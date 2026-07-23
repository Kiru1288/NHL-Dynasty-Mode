"""Shared helpers and franchise startup."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403
import logging

from app.sim_engine.entities.coach import CoachRole, generate_coach  # noqa: E402
from app.sim_engine.league import generate_regular_season_schedule  # noqa: E402
from app.sim_engine.league.schedule_generator import GameSlot, _safe_team_id, _safe_id_str, _safe_slot_team_id  # noqa: E402
from app.sim_engine.league.standings import StandingsTable  # noqa: E402
from app.sim_engine.franchise.session import FranchiseSession  # noqa: E402
from app.sim_engine.franchise.calendar import (  # noqa: E402
    build_season_calendar,
    calendar_day_to_dict,
    last_regular_season_index,
    map_abstract_schedule_to_calendar,
)
from app.sim_engine.franchise.schedule import (  # noqa: E402
    _calendar_iso_for_day,
    _finalize_schedule_after_generation,
    _schedule_quality_summary,
)

try:
    from app.sim_engine.world import calendar as world_calendar  # noqa: E402
    from app.sim_engine.world import chemistry as world_chemistry  # noqa: E402
    from app.sim_engine.world import durability as world_durability  # noqa: E402
    from app.sim_engine.world import fatigue as world_fatigue  # noqa: E402
    from app.sim_engine.world import injuries as world_injuries  # noqa: E402
    from app.sim_engine.world import morale as world_morale  # noqa: E402
    from app.sim_engine.world import momentum as world_momentum  # noqa: E402
except Exception:
    world_momentum = None  # type: ignore
    world_fatigue = None  # type: ignore
    world_morale = None  # type: ignore
    world_chemistry = None  # type: ignore
    world_injuries = None  # type: ignore
    world_durability = None  # type: ignore
    world_calendar = None  # type: ignore

_startup_log = logging.getLogger("uvicorn.error")


def _display_team(t: Any) -> str:
    city = str(getattr(t, "city", "") or "").strip()
    name = str(getattr(t, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return rs._team_name(t)


def _franchise_startup_stage(msg: str) -> None:
    """Always-on lightweight startup tracing (see post /api/franchise/start)."""
    _startup_log.info("[franchise start] %s", msg)
def _fr_dbg_enabled() -> bool:
    return os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1"
def _fr_dbg(msg: str) -> None:
    if _fr_dbg_enabled():
        print(f"[franchise debug] {msg}")
def _ensure_session_event_lists(session: FranchiseSession) -> None:
    if not hasattr(session, "calendar_events") or session.calendar_events is None:
        session.calendar_events = []
    if not hasattr(session, "pending_ui_popups") or session.pending_ui_popups is None:
        session.pending_ui_popups = []
    if not hasattr(session, "pending_decisions") or session.pending_decisions is None:
        session.pending_decisions = []
    if not hasattr(session, "notifications") or session.notifications is None:
        session.notifications = []
    if not hasattr(session, "timeline") or session.timeline is None:
        session.timeline = []
def _append_unique_dict_event(rows: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
    eid = str(event.get("id") or "").strip()
    if eid and any(isinstance(x, dict) and str(x.get("id") or "") == eid for x in rows):
        return
    rows.append(event)
def _normalized_notification(
    *,
    notification_id: str,
    notification_type: str,
    text: str,
    priority: str = "LOW",
    calendar_day: int = 0,
    calendar_iso: str = "",
    team_id: str = "",
    player_id: str = "",
    source: str = "franchise",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "id": str(notification_id),
        "type": str(notification_type or "system"),
        "text": str(text or ""),
        "priority": str(priority or "LOW").upper(),
        "calendar_day": int(calendar_day),
        "date": int(calendar_day),
        "calendar_iso": str(calendar_iso or ""),
        "team_id": _safe_id_str(team_id),
        "player_id": str(player_id or ""),
        "source": str(source or "franchise"),
    }
    if extra:
        row.update(extra)
    return row
def _normalized_timeline_event(
    *,
    event_id: str,
    event_type: str,
    text: str,
    calendar_day: int = 0,
    calendar_iso: str = "",
    team_id: str = "",
    player_id: str = "",
    priority: str = "LOW",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "id": str(event_id),
        "type": str(event_type or "system"),
        "text": str(text or ""),
        "headline": str(text or ""),
        "calendar_day": int(calendar_day),
        "date": int(calendar_day),
        "calendar_iso": str(calendar_iso or ""),
        "team_id": _safe_id_str(team_id),
        "player_id": str(player_id or ""),
        "priority": str(priority or "LOW").upper(),
    }
    if extra:
        row.update(extra)
    return row
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _same_calendar_day(row: Dict[str, Any], day_idx: int, day_iso: str = "") -> bool:
    """True when a notification/popup/event row belongs to the given calendar day."""
    if not isinstance(row, dict):
        return False
    try:
        target = int(day_idx)
    except (TypeError, ValueError):
        return False
    iso = str(day_iso or "").strip()
    row_iso = str(row.get("calendar_iso") or "").strip()
    if iso and row_iso and iso == row_iso:
        return True
    cd = row.get("calendar_day")
    if cd is not None and cd != "":
        try:
            if int(cd) == target:
                return True
        except (TypeError, ValueError):
            pass
    d = row.get("date")
    if d is None or d == "":
        return False
    try:
        return int(d) == target
    except (TypeError, ValueError):
        return bool(iso) and str(d).strip() == iso


def _franchise_log_injury_and_ui(
    session: FranchiseSession,
    *,
    player_id: str,
    player_name: str,
    team_id: str,
    team_abbrev: str,
    tier: str,
    games: int,
    injury_type: str,
    calendar_day: int,
    calendar_iso: str = "",
    game_day_injury: bool = False,
) -> None:
    _ensure_session_event_lists(session)

    if getattr(session, "injury_log_all", None) is None:
        session.injury_log_all = []
    if getattr(session, "injury_log_major", None) is None:
        session.injury_log_major = []

    cur_date = int(calendar_day)
    pid = str(player_id or "")
    tid_cmp = _safe_id_str(team_id).strip()
    utid = str(getattr(session, "user_team_id", "") or "").strip()
    tier_l = str(tier or "").lower().strip()
    team_abbr = str(team_abbrev or "").strip()
    player_label = str(player_name or "Player").strip()

    if not calendar_iso:
        calendar_iso = _calendar_iso_for_day(session, cur_date)

    if any(
        str(i.get("player_id") or "") == pid and _same_calendar_day(i, cur_date, calendar_iso)
        for i in session.injury_log_all
        if isinstance(i, dict)
    ):
        return

    is_user_team = bool(tid_cmp and utid) and tid_cmp.lower() == utid.lower()
    is_major = tier_l == "major"
    is_moderate = tier_l == "moderate"

    base_id = f"injury:{cur_date}:{tid_cmp}:{pid}"

    injury_log_row = {
        "id": f"injlog:{cur_date}:{tid_cmp}:{pid}",
        "player_id": pid,
        "player_name": player_label,
        "team_id": tid_cmp,
        "team_abbrev": team_abbr,
        "team_abbr": team_abbr,
        "tier": str(tier or ""),
        "severity": str(tier or ""),
        "injury_type": str(injury_type or tier or ""),
        "games": int(games),
        "games_initial": int(games),
        "games_remaining": int(games),
        "games_remaining_at_log": int(games),
        "type": str(injury_type or tier or ""),
        "date": cur_date,
        "calendar_day": cur_date,
        "calendar_iso": str(calendar_iso or ""),
        "status": "INJURED",
    }

    session.injury_log_all.append(injury_log_row)

    if is_major:
        session.injury_log_major.append(
            {
                "id": f"injmajor:{cur_date}:{tid_cmp}:{pid}",
                "player": player_label,
                "player_name": player_label,
                "player_id": pid,
                "tier": tier,
                "games": int(games),
                "team_id": tid_cmp,
                "team_abbrev": team_abbr,
                "calendar_day": cur_date,
                "calendar_iso": str(calendar_iso or ""),
            }
        )

    priority = "HIGH" if is_major else "MEDIUM"
    summary = (
        f"{player_label} ({team_abbr or tid_cmp}) is expected to miss {int(games)} games "
        f"with {injury_type or tier}."
    )
    if game_day_injury:
        summary += " This occurred during today's scheduled game."

    calendar_event = {
        "id": base_id,
        "kind": "injury",
        "type": "injury_report",
        "calendar_day": cur_date,
        "date": cur_date,
        "calendar_iso": str(calendar_iso or ""),
        "title": "Injury Report",
        "headline": f"{player_label} injured",
        "summary": summary,
        "description": summary,
        "team_id": tid_cmp,
        "team_abbrev": team_abbr,
        "team_abbr": team_abbr,
        "player_id": pid,
        "player_name": player_label,
        "priority": priority,
        "tier": str(tier or ""),
        "severity": str(tier or ""),
        "injury_type": str(injury_type or tier or ""),
        "games": int(games),
        "games_remaining": int(games),
        "game_day_injury": bool(game_day_injury),
        "surfaces": ["calendar", "storylines", "notifications"] + (["popup"] if is_user_team or is_major else []),
        "effects": {
            "availability_games_delta": -int(games),
            "depth_stress_delta": 1 if int(games) >= 2 else 0,
        },
        "effect_summary": f"Projected absence: {int(games)} game(s).",
    }

    _append_unique_dict_event(session.calendar_events, calendar_event)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"notif:{base_id}",
            notification_type="injury",
            text=f"{player_label} ({team_abbr or tid_cmp}) out {int(games)} games ({injury_type or tier}).",
            priority=priority,
            calendar_day=cur_date,
            calendar_iso=str(calendar_iso or ""),
            team_id=tid_cmp,
            player_id=pid,
            source="injury_engine",
            extra={"game_day_injury": bool(game_day_injury)},
        )
    )

    _record_storyline(
        session,
        {
            "id": f"story:{base_id}",
            "type": "injury",
            "kind": "injury",
            "headline": f"{player_label} sidelined",
            "details": f"{team_abbr or tid_cmp} lose {player_label} for {int(games)} games ({injury_type or tier})",
            "cause": f"{player_label} suffered a {injury_type or tier} injury.",
            "effects": {
                "availability_games_delta": -int(games),
                "depth_stress_delta": 1 if int(games) >= 2 else 0,
            },
            "effect_summary": f"Projected absence: {int(games)} game(s).",
            "team": tid_cmp,
            "team_id": tid_cmp,
            "team_abbrev": team_abbr,
            "player_id": pid,
            "player_name": player_label,
            "players": [player_label],
            "priority": priority,
            "date": cur_date,
            "calendar_day": cur_date,
            "calendar_iso": str(calendar_iso or ""),
            "surfaces": ["storylines", "calendar"],
        },
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:{base_id}",
            event_type="injury",
            text=f"{team_abbr or tid_cmp}: {player_label} injured ({int(games)}g)",
            calendar_day=cur_date,
            calendar_iso=str(calendar_iso or ""),
            team_id=tid_cmp,
            player_id=pid,
            priority=priority,
            extra={"surfaces": ["timeline", "calendar"]},
        )
    )

    # User-team injuries always popup. Major league injuries also popup as league news.
    should_popup = is_user_team or is_major
    if should_popup:
        same_day_injury_popups = [
            p
            for p in session.pending_ui_popups
            if isinstance(p, dict)
            and str(p.get("kind") or "") == "injury"
            and _same_calendar_day(p, cur_date, calendar_iso)
        ]

        if len(same_day_injury_popups) < 4:
            requires_decision = bool(is_user_team and (is_moderate or is_major))
            popup = {
                **calendar_event,
                "id": base_id,
                "kind": "injury",
                "date": cur_date,
                "calendar_day": cur_date,
                "requires_decision": requires_decision,
                "decision_id": base_id if requires_decision else "",
                "popup_scope": "user_team" if is_user_team else "league_news",
                "choices": (
                    [
                        {"id": "call_up_player", "label": "Call Up Depth Player"},
                        {"id": "shuffle_lines", "label": "Shuffle Lines"},
                        {"id": "play_short_roster", "label": "Play Short Roster"},
                        {"id": "place_on_ir", "label": "Place On IR"},
                    ]
                    if requires_decision
                    else []
                ),
            }

            _append_unique_dict_event(session.pending_ui_popups, popup)

            if requires_decision:
                _append_unique_dict_event(
                    session.pending_decisions,
                    {
                        "id": base_id,
                        "kind": "injury_decision",
                        "type": "injury_decision",
                        "calendar_day": cur_date,
                        "date": cur_date,
                        "calendar_iso": str(calendar_iso or ""),
                        "team_id": tid_cmp,
                        "team_abbrev": team_abbr,
                        "player_id": pid,
                        "player_name": player_label,
                        "title": "Injury Decision Required",
                        "summary": summary,
                        "choices": popup["choices"],
                        "resolved": False,
                    },
                )
def _find_player_on_team(session: FranchiseSession, team_id: str, player_name: str, player_id: str = "") -> Any:
    """Resolve roster player from franchise session."""
    tid = str(team_id or "").strip()
    tm = session.team_by_id.get(tid)
    if tm is None:
        for k, t in (session.team_by_id or {}).items():
            if str(k).lower() == tid.lower():
                tm = t
                break
    if tm is None:
        return None
    pid = str(player_id or "").strip()
    roster = list(getattr(tm, "roster", None) or [])
    if pid:
        for p in roster:
            if str(getattr(p, "id", "") or "") == pid:
                return p
    needle = str(player_name or "").strip().lower()
    if not needle:
        return None
    for p in roster:
        ident = getattr(p, "identity", None)
        labels = [
            str(getattr(p, "name", "") or ""),
            str(getattr(ident, "full_name", "") or "") if ident else "",
            str(getattr(ident, "name", "") or "") if ident else "",
        ]
        for lab in labels:
            if lab and (needle == lab.lower() or needle in lab.lower()):
                return p
    return None


def _franchise_tick_conduct_and_resolve(session: FranchiseSession, calendar_idx: int, day_meta: Dict[str, Any]) -> None:
    """After team games, emit resolution storylines when conduct suspensions expire."""
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
    from app.sim_engine.franchise.storyline_conduct import resolve_conduct_if_cleared  # noqa: WPS433

    if str(day_meta.get("segment") or "") not in ("preseason", "regular", "playoffs"):
        return

    _ensure_session_event_lists(session)
    iso = str(day_meta.get("iso") or "")
    cur_date = int(calendar_idx)

    for tm in session.team_by_id.values():
        tid = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "").strip()
        for pl in getattr(tm, "roster", None) or []:
            res = resolve_conduct_if_cleared(pl)
            if not res:
                continue
            pname = str(getattr(pl, "name", "") or "Player")
            sid = str(res.get("storyline_id") or f"conduct:{tid}:{getattr(pl, 'id', '')}")
            headline = f"UPDATE: {pname} Cleared to Return After Conduct Review"
            summary = str(res.get("resolution_summary") or "")
            _record_storyline(
                session,
                {
                    "id": f"story:resolve:{sid}:{cur_date}",
                    "storyline_id": sid,
                    "type": "legal_trouble",
                    "kind": "legal_trouble",
                    "category": "legal_trouble",
                    "headline": headline,
                    "summary": summary,
                    "details": summary,
                    "team_id": tid,
                    "team": tid,
                    "player_id": str(getattr(pl, "id", "") or ""),
                    "player_name": pname,
                    "priority": "MEDIUM",
                    "date": cur_date,
                    "calendar_day": cur_date,
                    "calendar_iso": iso,
                    "status": "resolved",
                    "arc_status": "resolved",
                    "overall_after": res.get("overall_after_return"),
                    "overall_before": res.get("overall_before_penalty"),
                    "effect_summary": summary,
                    "follow_up": "Watch next game for on-ice response.",
                    "surfaces": ["storylines", "notifications", "calendar"],
                },
            )
            session.notifications.append(
                _normalized_notification(
                    notification_id=f"notif:conduct_resolve:{sid}:{cur_date}",
                    notification_type="storyline",
                    text=f"{pname}: {summary}",
                    priority="MEDIUM",
                    calendar_day=cur_date,
                    calendar_iso=iso,
                    team_id=tid,
                    player_id=str(getattr(pl, "id", "") or ""),
                    source="conduct_engine",
                    extra={"arc_status": "resolved", "storyline_id": sid},
                )
            )


def _legal_gm_choice_options() -> List[Dict[str, Any]]:
    return [
        {"id": "suspend_internally", "label": "Suspend player internally"},
        {"id": "wait_league", "label": "Wait for league investigation"},
        {"id": "trade_immediately", "label": "Trade player immediately"},
        {"id": "support_program", "label": "Send player to support program"},
        {"id": "release_statement", "label": "Release statement"},
        {"id": "do_nothing", "label": "Do nothing and risk backlash"},
    ]


def _franchise_fanout_player_storylines(session: FranchiseSession, calendar_idx: int, day_meta: Dict[str, Any]) -> None:
    """Run one league-wide player storyline tick and surface notifications/popups like injuries."""
    if str(day_meta.get("segment") or "") not in ("preseason", "regular"):
        return
    from app.sim_engine.franchise.serialization import _franchise_team_abbrev  # noqa: WPS433
    from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433
    from app.sim_engine.franchise.engine_core import _estimate_return_from_games_remaining  # noqa: WPS433
    from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
        apply_conduct_suspension,
        apply_storyline_ovr_nudge,
        build_conduct_storyline_fields,
        build_impact_storyline_fields,
    )
    from app.sim_engine.franchise.storyline_engine import (  # noqa: WPS433
        ensure_cpu_storyline_cause,
        log_blocked_storyline,
        migrate_session_storyline_state,
        should_block_random_storyline_for_user,
        validate_storyline_before_effects,
    )

    migrate_session_storyline_state(session)
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is not None:
        setattr(league, "_franchise_user_team_id", str(session.user_team_id or ""))
    spp = getattr(sim, "run_player_storyline_pass", None)
    if not callable(spp):
        return
    try:
        raw = spp(sim.rng, int(session.season_calendar_year), franchise_tick=True)
    except TypeError:
        raw = spp(sim.rng, int(session.season_calendar_year))
    if not isinstance(raw, dict):
        return
    chaos = raw.get("league_delta") or {}
    try:
        session.chaos_index = _clamp(
            float(session.chaos_index) + float(chaos.get("chaos_index", 0) or 0),
            0.08,
            0.98,
        )
    except Exception:
        pass
    consequences = list(raw.get("narrative_consequences") or [])
    if not consequences:
        return

    _ensure_session_event_lists(session)
    utid = str(session.user_team_id)
    cur_date = int(calendar_idx)
    iso = str(day_meta.get("iso") or "")

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
        is_u = 1 if str(row.get("team_id") or "") == utid else 0
        tier = str(row.get("arc_tier") or "")
        tr = 0 if tier == "major" else 1 if tier == "mid" else 2
        return (-is_u, tr, str(row.get("player_name") or ""))

    rows = sorted(consequences, key=_sort_key)
    note_cap = 3
    notes = [
        n
        for n in (session.notifications or [])
        if isinstance(n, dict)
        and str(n.get("type") or "") in ("player_story", "storyline", "legal_trouble")
        and _same_calendar_day(n, cur_date, iso)
    ]
    popup_cap = 2
    legal_popup_cap = 1
    popups_today = [
        p
        for p in (session.pending_ui_popups or [])
        if isinstance(p, dict)
        and str(p.get("kind") or "") in ("storyline", "legal_trouble")
        and _same_calendar_day(p, cur_date, iso)
    ]

    legal_popups_today = sum(
        1 for p in popups_today if isinstance(p, dict) and str(p.get("kind") or "") == "legal_trouble"
    )

    def _apply_row_player_impact(
        row: Dict[str, Any],
        *,
        tid: str,
        pname: str,
        base_id: str,
        is_legal: bool,
        legal_sev: str,
        tier: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pl = _find_player_on_team(session, tid, pname, str(row.get("player_id") or ""))
        if pl is None:
            return {}, {}
        pol = str(row.get("storyline_polarity") or "negative").lower()
        if pol == "positive":
            return {}, {}
        should_impact = (
            is_user_team
            or (is_legal and legal_sev == "major")
            or (is_legal and is_user_team)
            or (tier == "major")
            or (tier == "mid" and is_user_team)
        )
        if is_user_team and should_block_random_storyline_for_user(row, session, user_team_id=utid):
            log_blocked_storyline(
                session,
                row,
                "no registered franchise cause (random engine storyline blocked for user team)",
            )
            return {}, {}
        if not should_impact:
            return {}, {}
        rng = getattr(session.sim, "rng", None)
        meta: Dict[str, Any] = {}
        if is_legal and legal_sev == "major":
            meta = apply_conduct_suspension(
                pl,
                severity="major",
                storyline_id=base_id,
                cause_type="LOW_CHARACTER_CONFLICT",
                cause_event_id=str(row.get("cause_event_id") or ""),
                rng=rng,
            )
            ret_est, ret_iso = _estimate_return_from_games_remaining(
                session, int(meta.get("games_remaining") or 0)
            )
            fields = build_conduct_storyline_fields(meta, return_estimate=ret_est, return_date=ret_iso)
            if meta.get("games_remaining"):
                fields["impact_reason"] = "League investigation / indefinite leave penalty"
            return meta, fields
        sev_key = legal_sev if is_legal else tier
        meta = apply_storyline_ovr_nudge(
            pl,
            tier=str(sev_key or tier or "minor"),
            legal_severity=legal_sev if is_legal else "",
            storyline_id=base_id,
            rng=rng,
        )
        return meta, build_impact_storyline_fields(meta)

    def _storyline_presentation(
        *,
        is_legal: bool,
        legal_sev: str,
        event_type: str,
        tier: str,
        is_user_team: bool,
    ) -> Dict[str, str]:
        if is_legal and legal_sev == "major":
            return {
                "presentation_type": "legal_major",
                "source_label": "League Office Report",
                "theme": "danger",
                "icon": "§",
            }
        if is_legal:
            return {
                "presentation_type": "legal_minor",
                "source_label": "League Statement",
                "theme": "neutral",
                "icon": "§",
            }
        et = str(event_type or "").lower()
        if et in ("scandal", "locker_room_issue", "team_conflict"):
            return {
                "presentation_type": "locker_room",
                "source_label": "Insider Alert",
                "theme": "warning",
                "icon": "!",
            }
        if et in ("breakout", "emergence", "clutch_run", "leader_emergence", "confidence_surge"):
            return {
                "presentation_type": "positive",
                "source_label": "Team Report",
                "theme": "positive",
                "icon": "▲",
            }
        if "goalie" in et or et == "goalie_meltdown":
            return {
                "presentation_type": "goalie",
                "source_label": "Net Report",
                "theme": "info",
                "icon": "◎",
            }
        if is_user_team:
            return {
                "presentation_type": "team_story",
                "source_label": "Team Statement Released",
                "theme": "info",
                "icon": "◆",
            }
        return {
            "presentation_type": "league_news",
            "source_label": "League Wire",
            "theme": "neutral",
            "icon": "◉",
        }

    for row in rows:
        if len(notes) >= note_cap:
            break
        tid_row = str(row.get("team_id") or "")
        if tid_row == utid and should_block_random_storyline_for_user(row, session, user_team_id=utid):
            log_blocked_storyline(
                session,
                row,
                "no registered franchise cause (random engine storyline blocked for user team)",
            )
            continue
        is_user_team = tid_row == utid
        if not is_user_team:
            row = ensure_cpu_storyline_cause(session, row, tid_row)
            pol = str(row.get("storyline_polarity") or "negative").lower()
            if pol == "negative" and str(row.get("cause_type") or ""):
                cpu_sl = {
                    "team_id": tid_row,
                    "player_id": str(row.get("player_id") or ""),
                    "cause_type": row.get("cause_type"),
                    "cause_event_id": row.get("cause_event_id"),
                    "tone": "negative",
                    "effects": {},
                    "headline": str(row.get("storyline_text") or ""),
                }
                if not validate_storyline_before_effects(session, cpu_sl):
                    log_blocked_storyline(session, row, "CPU storyline missing valid cause")
                    continue
        st = str(row.get("storyline_text") or "").strip()
        if not st:
            continue
        pname = str(row.get("player_name") or "Player")
        tid = str(row.get("team_id") or "")
        is_user_team = tid == utid
        tm = session.team_by_id.get(tid)
        abbrev = _franchise_team_abbrev(tm) if tm is not None else (tid[:3].upper() if tid else "?")
        tier = str(row.get("arc_tier") or "")
        event_type = str(row.get("event_type") or "generic")
        pool = str(row.get("pool") or "")
        legal_sev = str(row.get("legal_severity") or "").lower()
        is_legal = event_type == "legal_trouble" or pool == "legal_crime"
        priority = "HIGH" if tier == "major" or is_legal else "MEDIUM"
        notif_type = "legal_trouble" if is_legal else "storyline"
        base_id = f"storyline:{cur_date}:{tid}:{abs(hash(st + pname)) % 10_000_000}"

        conduct_meta: Dict[str, Any] = {}
        conduct_fields: Dict[str, Any] = {}
        ret_est, ret_iso = "", ""
        conduct_meta, conduct_fields = _apply_row_player_impact(
            row,
            tid=tid,
            pname=pname,
            base_id=base_id,
            is_legal=is_legal,
            legal_sev=legal_sev,
            tier=tier,
        )
        if conduct_fields.get("games_remaining"):
            ret_est = str(conduct_fields.get("return_estimate") or "")
            ret_iso = str(conduct_fields.get("return_date") or "")

        notif_extra = {
            "player_name": pname,
            "storyline_text": st,
            "arc_tier": tier,
            "event_type": event_type,
            "legal_severity": legal_sev,
            "is_user_team": is_user_team,
        }
        if conduct_fields:
            notif_extra.update(conduct_fields)
            gr = int(conduct_fields.get("games_remaining") or 0)
            ovr_d = conduct_fields.get("overall_delta")
            notif_tail = ""
            if gr > 0:
                notif_tail = f" OUT {gr}G · Return {conduct_fields.get('return_estimate') or ret_est}"
            if ovr_d is not None and int(ovr_d) != 0:
                notif_tail += (
                    f" · OVR {conduct_fields.get('overall_before')}→{conduct_fields.get('overall_after')} ({int(ovr_d):+d})"
                )
        else:
            notif_tail = ""

        session.notifications.append(
            _normalized_notification(
                notification_id=f"notif:{base_id}",
                notification_type=notif_type,
                text=f"{pname} ({abbrev}): {st}{notif_tail}",
                priority=priority,
                calendar_day=cur_date,
                calendar_iso=iso,
                team_id=tid,
                player_id=str(row.get("player_id") or ""),
                source="player_storyline_engine",
                extra=notif_extra,
            )
        )
        notes.append(1)

        headline = f"{pname}: {st[:90]}{'…' if len(st) > 90 else ''}"
        if is_legal and legal_sev == "major":
            headline = f"BREAKING: League Opens Investigation Into Player Conduct — {pname}"
        elif is_legal:
            headline = f"League Reviewing Off-Ice Matter — {pname}"

        summary = st
        if is_legal and legal_sev == "major":
            summary = (
                f"The league has opened a formal review involving {pname}. "
                f"The player is away from the team while the organization gathers information. "
                f"— {st}"
            )
        elif is_legal:
            summary = (
                f"League and team officials are monitoring an off-ice matter involving {pname}. "
                f"No suspension announced at this time. — {st}"
            )

        _record_storyline(
            session,
            {
                "id": f"story:{base_id}",
                "storyline_id": base_id,
                "type": "player_storyline" if not is_legal else "legal_trouble",
                "kind": "legal_trouble" if is_legal else "storyline",
                "category": "legal_trouble" if is_legal else str(event_type or "storyline"),
                "headline": headline,
                "details": st,
                "summary": summary,
                "team": tid,
                "team_id": tid,
                "team_abbrev": abbrev,
                "player_id": str(row.get("player_id") or ""),
                "player_name": pname,
                "players": [pname],
                "priority": priority,
                "date": cur_date,
                "calendar_day": cur_date,
                "calendar_iso": iso,
                "event_type": event_type,
                "legal_severity": legal_sev,
                "arc_tier": tier,
                "arc_status": conduct_fields.get("arc_status", "active"),
                "status": "active",
                "surfaces": ["calendar", "storylines", "notifications"]
                + (["popup"] if is_user_team or is_legal else []),
                **conduct_fields,
            },
        )

        trim = 72
        short = st if len(st) <= trim else st[: trim - 1] + "…"
        session.timeline.append(
            _normalized_timeline_event(
                event_id=f"timeline:{base_id}",
                event_type="legal_trouble" if is_legal else "storyline",
                text=f"Storyline: {pname} ({abbrev}) — {short}",
                calendar_day=cur_date,
                calendar_iso=iso,
                team_id=tid,
                priority=priority,
            )
        )

        should_popup = is_user_team or (is_legal and legal_sev == "major")
        if is_legal and not is_user_team and legal_popups_today >= legal_popup_cap:
            should_popup = False
        if should_popup and len(popups_today) < popup_cap:
            requires_decision = bool(is_user_team and is_legal and legal_sev == "major")
            popup_kind = "legal_trouble" if is_legal else "storyline"
            pres = _storyline_presentation(
                is_legal=is_legal,
                legal_sev=legal_sev,
                event_type=event_type,
                tier=tier,
                is_user_team=is_user_team,
            )
            popup_title = pres["source_label"]
            if is_legal and legal_sev == "major":
                popup_title = "League Office Report — BREAKING"
            elif is_user_team:
                popup_title = "Team Statement Released"
            popup = {
                "id": base_id,
                "kind": popup_kind,
                "title": popup_title,
                "headline": headline,
                "summary": summary,
                "description": summary,
                "player_name": pname,
                "player_id": str(row.get("player_id") or ""),
                "team_id": tid,
                "team_abbrev": abbrev,
                "team_abbr": abbrev,
                "storyline_text": st,
                "story_report": summary,
                "franchise_impact": conduct_fields.get("effect_summary") or "",
                "impact_reason": conduct_fields.get("impact_reason") or "",
                "event_type": event_type,
                "legal_severity": legal_sev,
                "tier": tier,
                "arc_tier": tier,
                "priority": priority,
                "date": cur_date,
                "calendar_day": cur_date,
                "calendar_iso": iso,
                "is_user_team": is_user_team,
                "popup_scope": "user_team" if is_user_team else "league_news",
                "requires_decision": requires_decision,
                "decision_id": base_id if requires_decision else "",
                "choices": _legal_gm_choice_options() if requires_decision else [],
                "surfaces": ["popup", "storylines", "notifications", "calendar"],
                **pres,
                **conduct_fields,
            }
            _append_unique_dict_event(session.pending_ui_popups, popup)
            popups_today.append(popup)
            if is_legal:
                legal_popups_today += 1

            if requires_decision:
                _append_unique_dict_event(
                    session.pending_decisions,
                    {
                        "id": base_id,
                        "kind": "legal_storyline_decision",
                        "type": "legal_storyline_decision",
                        "priority": "HIGH",
                        "calendar_day": cur_date,
                        "date": cur_date,
                        "calendar_iso": iso,
                        "title": popup_title,
                        "description": summary,
                        "options": _legal_gm_choice_options(),
                        "meta": {
                            "storyline_id": base_id,
                            "team_id": tid,
                            "player_name": pname,
                            "legal_severity": legal_sev,
                            "cause": st,
                        },
                    },
                )

    if len(session.timeline) > 200:
        session.timeline = session.timeline[-200:]


def resolve_user_team(teams: List[Any], query: str) -> Any:
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Team query is empty.")
    matches: List[Any] = []
    for t in teams:
        raw_tid = getattr(t, "team_id", None)
        if raw_tid is not None and str(raw_tid).lower() == q:
            matches.append(t)
            continue
        tid = str(rs._team_id(t)).lower()
        disp = _display_team(t).lower()
        nm = str(getattr(t, "name", "") or "").lower()
        ct = str(getattr(t, "city", "") or "").lower()
        if q == tid or q in disp or q in nm or q in ct or q in f"{ct} {nm}".strip():
            matches.append(t)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No team matched {query!r}. Try city, nickname, or numeric team id.")
    hint = ", ".join(_display_team(x) for x in matches[:6])
    raise ValueError(f"Ambiguous team {query!r}; matches include: {hint}")
def _use_world_modules() -> bool:
    return all(
        m is not None
        for m in (
            world_momentum,
            world_fatigue,
            world_morale,
            world_chemistry,
            world_injuries,
            world_durability,
            world_calendar,
        )
    )
def apply_coach_archetype(coach: Any, archetype: str, rng: random.Random) -> None:
    arch = (archetype or "balanced").lower().replace(" ", "_").replace("-", "_")
    try:
        if arch in ("development", "development_first", "teacher"):
            coach.usage.trust_youth = _clamp(float(coach.usage.trust_youth) + 0.12)
            coach.usage.trust_veterans = _clamp(float(coach.usage.trust_veterans) - 0.05)
            coach.development.skill_growth_multiplier = min(
                1.15, float(coach.development.skill_growth_multiplier) + 0.06
            )
        elif arch in ("defense_first", "defensive", "structure"):
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) - 0.10)
            coach.usage.penalty_kill_conservatism = _clamp(
                float(coach.usage.penalty_kill_conservatism) + 0.08
            )
        elif arch in ("aggressive", "attack", "offense_first"):
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) + 0.12)
            coach.tactics.offensive_activation = _clamp(
                float(coach.tactics.offensive_activation) + 0.08
            )
        elif arch in ("players_coach", "culture", "leader"):
            coach.usage.meritocracy = _clamp(float(coach.usage.meritocracy) + 0.10)
            coach.room_temperature = _clamp(float(coach.room_temperature) + 0.08)
        else:
            # balanced: small random identity nudge
            coach.tactics.risk_tolerance = _clamp(float(coach.tactics.risk_tolerance) + rng.uniform(-0.03, 0.03))
    except Exception:
        pass
def _chaos_index(sim: Any, league: Any) -> float:
    ctx = getattr(league, "_tuning_context", None) or {}
    return float(ctx.get("chaos_index", getattr(league, "_chaos_index", 0.5)) or 0.5)
def start_franchise(
    *,
    team_query: str,
    head_coach_name: str,
    coach_archetype: str,
    seed: Optional[int] = None,
    games_per_team: int = 82,
    season_start_year: Optional[int] = None,
    injuries_enabled: bool = True,
) -> FranchiseSession:
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    _franchise_startup_stage("SimEngine import complete; constructing engine")
    master = seed if seed is not None else random.randrange(1, 10**9)
    sim = SimEngine(seed=master, debug=False)
    _franchise_startup_stage("SimEngine constructed")
    league = sim.league
    try:
        setattr(league, "_runner_sim_engine", sim)
    except Exception:
        pass

    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        raise RuntimeError("League has no teams after initialization.")
    _franchise_startup_stage(f"team resolution: {len(teams)} clubs in league")

    user_team = resolve_user_team(teams, team_query)
    _tid = getattr(user_team, "team_id", None)
    if _tid is not None:
        uid = str(_tid)
    else:
        _oid = getattr(user_team, "id", None)
        uid = str(_oid) if _oid is not None else rs._team_id(user_team)
    sim.team = user_team
    _franchise_startup_stage(f"user team resolved -> {uid}")

    coach = generate_coach(sim.rng, f"HIRE_{uid}", CoachRole.HEAD_COACH)
    coach.name = (head_coach_name or "Head Coach").strip() or "Head Coach"
    apply_coach_archetype(coach, coach_archetype, sim.rng)
    user_team.coach = coach
    sim.coach = coach
    _franchise_startup_stage("head coach generated and assigned")

    gp = int(games_per_team)
    if gp < 4:
        gp = 4
    if gp > 82:
        gp = 82
    season_y = int(season_start_year) if season_start_year is not None else 2025
    _franchise_startup_stage(f"generating abstract schedule ({gp} GP template)")
    schedule_raw = generate_regular_season_schedule(sim.rng, teams, gp)
    _franchise_startup_stage(f"abstract schedule slots={len(schedule_raw)}")
    by_abs: Dict[int, List[Any]] = defaultdict(list)
    for slot in schedule_raw:
        by_abs[int(slot.day)].append(slot)
    abstract_keys = sorted(by_abs.keys())

    _franchise_startup_stage(f"building NHL season calendar year={season_y}")
    cal_objs = build_season_calendar(season_y)
    nhl_cal = [calendar_day_to_dict(c) for c in cal_objs]
    last_reg_idx = last_regular_season_index(cal_objs)
    _franchise_startup_stage(f"calendar rows={len(nhl_cal)} last_regular_idx={last_reg_idx}")
    _franchise_startup_stage("map_abstract_schedule_to_calendar")
    day_map = map_abstract_schedule_to_calendar(cal_objs, abstract_keys)
    _franchise_startup_stage("abstract days mapped to calendar indices")
    by_day: Dict[int, List[Any]] = defaultdict(list)
    schedule: List[Any] = []
    for old in abstract_keys:
        nid = int(day_map[int(old)])
        for slot in by_abs[old]:
            gs = GameSlot(day=nid, home_id=slot.home_id, away_id=slot.away_id, is_playoff=slot.is_playoff)
            by_day[nid].append(gs)
            schedule.append(gs)
    _sched_dbg = os.environ.get("NHL_FRANCHISE_SCHEDULE_DEBUG") == "1"

    _franchise_startup_stage("_finalize_schedule_after_generation (smooth + repair + validate)")
    by_day, schedule, schedule_diagnostics = _finalize_schedule_after_generation(
        by_day,
        nhl_cal,
        user_id=uid,
    )
    schedule_diagnostics["quality"] = _schedule_quality_summary(by_day, nhl_cal)
    if _sched_dbg:
        print("[franchise schedule] diagnostics", schedule_diagnostics)
    he = list(schedule_diagnostics.get("hard_errors") or [])
    if he:
        _fr_dbg(f"schedule hard-validation warning at startup: {he[0]}")

    if not schedule:
        raise RuntimeError(
            "Franchise startup failed: schedule is empty after calendar mapping and finalization."
        )

    days_sorted = sorted(by_day.keys())
    _franchise_startup_stage(
        f"schedule finalized game_dates={len(days_sorted)} total_slots={len(schedule)} "
        f"validation_ok={bool(schedule_diagnostics.get('startup_validation_ok'))}"
    )
    standings = StandingsTable(teams)
    team_by_id: Dict[str, Any] = {}
    team_ids: List[str] = []
    for idx, t in enumerate(teams):
        # Must match schedule_generator._safe_team_id: team_id may be 0 (falsy) but is valid.
        tid = _safe_team_id(t, idx)
        team_ids.append(tid)
        team_by_id[tid] = t

    sim._preseason_line_synergy_refresh(teams, sim.rng)
    strength_map = sim._build_strength_map(teams)
    use_world = _use_world_modules()
    play_days: Dict[str, Any] = {}
    if use_world and world_calendar is not None:
        play_days = world_calendar.build_team_play_days(schedule)

    _franchise_startup_stage("creating FranchiseSession")
    session = FranchiseSession(
        session_id=FranchiseSession.new_id(),
        sim=sim,
        user_team_id=uid,
        head_coach_name=coach.name,
        coach_archetype=coach_archetype,
        season_calendar_year=season_y,
        games_per_team_schedule=gp,
        calendar_days_finished=0,
        schedule=schedule,
        by_day=dict(by_day),
        days_sorted=days_sorted,
        nhl_calendar=nhl_cal,
        calendar_cursor=0,
        nhl_regular_season_last_index=last_reg_idx,
        standings=standings,
        team_by_id=team_by_id,
        team_ids=team_ids,
        strength_map=strength_map,
        prev_calendar_day=None,
        last_game_day={tid: None for tid in team_ids},
        play_days=play_days,
        injury_log_major=[],
        chaos_index=_chaos_index(sim, league),
        use_world=use_world,
        injuries_enabled=bool(injuries_enabled),
        preseason_applied=True,
    )
    session.schedule_diagnostics = schedule_diagnostics
    session.notifications = getattr(session, "notifications", None) or []
    session.timeline = getattr(session, "timeline", None) or []
    session.pending_ui_popups = getattr(session, "pending_ui_popups", None) or []
    session.calendar_events = getattr(session, "calendar_events", None) or []
    session.pending_decisions = getattr(session, "pending_decisions", None) or []
    start_iso = _calendar_iso_for_day(session, 0)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"system:franchise_ready:{uid}",
            notification_type="system",
            text=f"Franchise ready ΓÇö {_display_team(user_team)} ({uid}).",
            priority="LOW",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
        )
    )

    session.notifications.append(
        _normalized_notification(
            notification_id=f"system:coach_hired:{uid}:{season_y}",
            notification_type="system",
            text=(
                f"Hired {coach.name} ({coach_archetype}). NHL calendar {season_y}ΓÇô{season_y + 1} ┬╖ "
                f"{len(nhl_cal)} days ┬╖ {len(days_sorted)} game dates ┬╖ ~{gp} GP."
            ),
            priority="LOW",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
        )
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:welcome:{uid}:{season_y}",
            event_type="system",
            text="Welcome to Franchise Mode. Advance the day to begin the regular season.",
            calendar_day=0,
            calendar_iso=start_iso,
            team_id=uid,
            priority="LOW",
        )
    )
    try:
        from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy

        bootstrap_full_league_hierarchy(league, sim.rng)
        npl = len(getattr(league, "players", None) or [])
        session.notifications.append(
            f"League depth online ΓÇö NHL affiliates (AHL/ECHL), UFA pools, overseas, juniors (~{npl} player records)."
        )
    except Exception as e:
        session.notifications.append(f"League depth bootstrap skipped: {e}")
    try:
        snapshot_draft_rank_prev(session, sim)
    except Exception:
        pass
    _franchise_startup_stage("start_franchise complete; returning session")
    return session
