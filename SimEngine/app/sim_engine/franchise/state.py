"""State payload, caches, trades, and popups."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403
from app.sim_engine.franchise.serialization import _normalize_storyline_payload  # noqa: E402


def invalidate_session_payload_caches(session: FranchiseSession, reason: str = "") -> None:
    """Drop cached read-model payloads after mutating session state."""
    session._cached_draft_class_rankings = None
    session._cached_trade_assets_payload = None


def get_cached_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    cached = getattr(session, "_cached_draft_class_rankings", None)
    if isinstance(cached, dict) and cached:
        return cached
    payload = build_draft_class_rankings(session, sim)
    session._cached_draft_class_rankings = payload
    return payload


def _storyline_dedupe_key(ev: Dict[str, Any]) -> str:
    sk = str(ev.get("stable_key") or "").strip()
    if sk:
        return sk
    return f"{ev.get('type')}|{ev.get('date')}|{str(ev.get('headline') or '')[:200]}"
def _merge_simengine_league_news_into_storylines(session: FranchiseSession) -> None:
    """One-time merge of latest SimEngine league season news_events into the franchise feed."""
    sim = getattr(session, "sim", None)
    if sim is None:
        return
    hist = list(getattr(sim, "league_history", None) or [])
    if not hist:
        return
    last = hist[-1]
    sig = (int(getattr(last, "year", 0) or 0), id(last))
    if getattr(session, "_merged_engine_news_sig", None) == sig:
        return
    setattr(session, "_merged_engine_news_sig", sig)
    nev = list(getattr(last, "news_events", None) or [])
    for raw in nev[-150:]:
        if not isinstance(raw, dict):
            continue
        _record_storyline(session, raw)
def _record_storyline(session: FranchiseSession, event: Dict[str, Any]) -> None:
    raw = event if isinstance(event, dict) else {}
    try:
        from app.sim_engine.franchise.storyline_engine import enrich_storyline_for_narrative_universe  # noqa: WPS433

        raw = enrich_storyline_for_narrative_universe(session, raw)
    except Exception:
        pass
    ev = _normalize_storyline_payload(raw)
    if not ev.get("headline"):
        return
    dq = getattr(session, "_storyline_dedupe", None)
    if dq is None:
        dq = []
        session._storyline_dedupe = dq
    dk = _storyline_dedupe_key(ev)
    if dk in dq:
        return
    dq.append(dk)
    if len(dq) > 500:
        session._storyline_dedupe = dq[-400:]
    if getattr(session, "storyline_events", None) is None:
        session.storyline_events = []
    session.storyline_events.append(ev)
    if len(session.storyline_events) > 400:
        session.storyline_events = session.storyline_events[-400:]
def _storyline_choices_payload(session: FranchiseSession) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in list(getattr(session, "pending_decisions", None) or []):
        kind = str(d.get("kind") or "")
        if kind in ("franchise_critical_notice",):
            continue
        opts = list(d.get("options") or [])
        if not opts:
            continue
        meta = dict(d.get("meta") or {})
        storyline_id = str(meta.get("storyline_id") or d.get("storyline_id") or d.get("id") or "")
        if not storyline_id:
            storyline_id = str(d.get("id") or "")
        action_options: List[Dict[str, Any]] = []
        for idx, opt in enumerate(opts):
            oid = str(opt.get("id") or f"opt_{idx}")
            action_options.append(
                {
                    "id": oid,
                    "label": str(opt.get("label") or oid.replace("_", " ").title()),
                    "effects": dict(opt.get("effects") or {}),
                    "effect_summary": str(opt.get("effect_summary") or "").strip(),
                }
            )
        rows.append(
            {
                "storyline_id": storyline_id,
                "decision_id": str(d.get("id") or ""),
                "kind": kind,
                "priority": str(d.get("priority") or "MEDIUM").upper(),
                "title": str(d.get("title") or "Storyline choice"),
                "description": str(d.get("description") or ""),
                "team_id": str(meta.get("team_id") or meta.get("team") or ""),
                "player_id": str(meta.get("player_id") or ""),
                "player_name": str(meta.get("player_name") or ""),
                "cause": str(meta.get("cause") or ""),
                "action_options": action_options,
            }
        )
    return rows
def dismiss_franchise_popups(session: FranchiseSession, popup_ids: List[str]) -> None:
    """Remove dismissed popups from the pending queue (archive is unchanged)."""
    if not popup_ids:
        return
    drop = {str(x).strip() for x in popup_ids if str(x).strip()}
    session.pending_ui_popups = [p for p in (session.pending_ui_popups or []) if str(p.get("id") or "") not in drop]
def _append_showcase_popup(session: FranchiseSession, dedupe_key: str, payload: Dict[str, Any]) -> None:
    if dedupe_key in session.shown_event_keys:
        return
    session.shown_event_keys.add(dedupe_key)
    pid = f"pop_{uuid.uuid4().hex[:12]}"
    body = dict(payload)
    body["id"] = pid
    session.pending_ui_popups.append(body)
    arch = list(getattr(session, "showcase_archive", None) or [])
    arch.append(dict(body))
    session.showcase_archive = arch[-48:]


def _record_trade_package_notifications(
    session: FranchiseSession,
    exec_result: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    moved_players = [
        m for m in (exec_result.get("moved_players") or exec_result.get("moved_assets") or [])
        if str(m.get("asset_type") or m.get("type") or "player").lower() == "player"
        and m.get("applied", True)
    ]
    if not moved_players:
        return
    trade_day = int(getattr(session, "calendar_cursor", 0) or 0)
    try:
        from app.sim_engine.franchise.storyline_engine import (  # noqa: WPS433
            migrate_session_storyline_state,
            record_decision_event,
            resolve_culprit_traded_storylines,
        )

        migrate_session_storyline_state(session)
        record_decision_event(
            session,
            {
                "event_type": "PLAYER_TRADED",
                "team_id": str(session.user_team_id),
                "player_ids": [str(m.get("asset_id") or m.get("id") or "") for m in moved_players],
                "trade_id": str(exec_result.get("trade_id") or f"trade_exec_{trade_day}"),
                "severity": "medium",
            },
        )
        resolve_culprit_traded_storylines(session, moved_players)
    except Exception:
        pass

    headline = str(exec_result.get("headline") or "")
    if not headline:
        headline_bits = []
        for m in moved_players[:4]:
            src = (
                _display_team(session.team_by_id.get(m["source_team_id"]))
                if session.team_by_id.get(m.get("source_team_id"))
                else m.get("source_team_id")
            )
            dst = (
                _display_team(session.team_by_id.get(m["acquiring_team_id"]))
                if session.team_by_id.get(m.get("acquiring_team_id"))
                else m.get("acquiring_team_id")
            )
            headline_bits.append(f"{m.get('player_name')}: {src} -> {dst}")
        headline = "TRADE EXECUTED: " + "; ".join(headline_bits)

    notif = _normalize_notification_payload(
        {
            "type": "trade",
            "priority": "HIGH",
            "title": "Trade Executed",
            "headline": headline,
            "text": headline,
            "source": "trade_hub",
        },
        index=len(session.notifications or []),
    )
    session.notifications.append(notif)
    _record_storyline(
        session,
        {
            "type": "trade",
            "priority": "HIGH",
            "headline": headline,
            "details": f"Moved {len(moved_players)} players via Trade Hub package.",
            "players": [str(m.get("player_name") or "") for m in moved_players],
            "team_id": str(session.user_team_id),
        },
    )
    trade_iso = _calendar_iso_for_day(session, trade_day)
    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:trade_hub:{trade_day}:{uuid.uuid4().hex[:8]}",
            event_type="trade",
            text=f"TRADE HUB: moved {len(moved_players)} player(s).",
            calendar_day=trade_day,
            calendar_iso=trade_iso,
            team_id=str(session.user_team_id),
            priority="HIGH",
        )
    )


def execute_trade_package(session: FranchiseSession, *, assets_by_team: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Deprecated alias — delegates to trade_service.execute_franchise_trade (see docs/TRADE_SYSTEM.md)."""
    from app.sim_engine.franchise.trade_service import execute_franchise_trade

    exec_result = execute_franchise_trade(
        session,
        assets_by_team=dict(assets_by_team or {}),
        record_notifications_fn=_record_trade_package_notifications,
    )
    invalidate_session_payload_caches(session, reason="trade_exec")
    moved_players = [
        m for m in (exec_result.get("moved_players") or exec_result.get("moved_assets") or [])
        if m.get("applied", True)
    ]
    headline = str(exec_result.get("headline") or "")
    return {
        "moved_assets": moved_players,
        "headline": headline,
        "moved_players": len(moved_players),
        "trade_id": exec_result.get("trade_id"),
        "execution": exec_result,
    }


def invalidate_session_payload_caches(session: FranchiseSession, reason: str = "") -> None:
    """Drop cached read-model payloads after mutating session state."""
    session._cached_draft_class_rankings = None
    session._cached_trade_assets_payload = None


def get_cached_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    cached = getattr(session, "_cached_draft_class_rankings", None)
    if isinstance(cached, dict) and cached:
        return cached
    payload = build_draft_class_rankings(session, sim)
    session._cached_draft_class_rankings = payload
    return payload


def build_state_payload(session: FranchiseSession) -> Dict[str, Any]:
    _sync_nhl_calendar_bounds(session)
    # Schedule cadence is smoothed in start_franchise only. Re-running _smooth_league_schedule
    # here made every GET /api/franchise/state take minutes (full league re-optimization).
    sim = session.sim
    user_team = session.team_by_id.get(session.user_team_id)
    rec = None
    if session.standings and user_team is not None:
        tid = str(
            getattr(user_team, "team_id", None)
            if getattr(user_team, "team_id", None) is not None
            else rs._team_id(user_team)
        )
        rec = session.standings.records.get(tid) or session.standings.records.get(session.user_team_id)

    roster_rows: List[Dict[str, Any]] = []
    if user_team is not None:
        for p in getattr(user_team, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            row = _serialize_player_row(p, include_ratings=True, session=session, _team=user_team)
            pid = row.get("player_id") or ""
            st = session.player_season_stats.get(pid)
            if st:
                row["season_stats"] = {
                    "gp": int(st.get("gp", 0)),
                    "g": int(st.get("g", 0)),
                    "a": int(st.get("a", 0)),
                    "pts": int(st.get("pts", 0)),
                    "sog": int(st.get("sog", 0)),
                    "pim": int(st.get("pim", 0)),
                    "hit": int(st.get("hit", st.get("hits", 0)) or 0),
                    "blk": int(st.get("blk", st.get("blocks", 0)) or 0),
                    "toi": round((int(st.get("toi_sec", 0) or 0) / max(1, int(st.get("gp", 0) or 1))) / 60.0, 1),
                    "ga": int(st.get("ga", 0)),
                    "w": int(st.get("w", 0)),
                    "l": int(st.get("l", 0)),
                    "otl": int(st.get("otl", 0)),
                }
            roster_rows.append(row)
        roster_rows.sort(key=lambda x: -float(x.get("ovr") or 0))

    cal_rec = _user_team_record_from_game_results(session)
    uid_s = str(session.user_team_id or "")

    standings_rows: List[Dict[str, Any]] = []
    if session.standings:
        for tid, r in session.standings.records.items():
            tid_s = str(tid)
            if uid_s and tid_s == uid_s and int(cal_rec.get("gp") or 0) > 0:
                standings_rows.append(
                    {
                        "team_id": tid_s,
                        "name": getattr(r, "name", tid),
                        "gp": int(cal_rec["gp"]),
                        "w": int(cal_rec["w"]),
                        "l": int(cal_rec["l"]),
                        "otl": int(cal_rec["otl"]),
                        "pts": int(cal_rec["pts"]),
                    }
                )
            else:
                standings_rows.append(
                    {
                        "team_id": tid_s,
                        "name": getattr(r, "name", tid),
                        "gp": getattr(r, "gp", 0),
                        "w": getattr(r, "wins", 0),
                        "l": getattr(r, "losses", 0),
                        "otl": getattr(r, "otl", 0),
                        "pts": getattr(r, "points", 0),
                    }
                )
        standings_rows.sort(key=lambda x: (-x["pts"], -(x["w"] - x["l"])))

    cap_info = _team_cap_snapshot(user_team, sim) if user_team is not None else {"salary_cap": 92.0, "cap_hit": 0.0, "cap_space": 92.0}
    cap_hint = str(getattr(user_team, "cap_pressure", "moderate") if user_team else "?")
    strat = str(getattr(user_team, "strategy", "balanced") if user_team else "?")

    day_display = "Off-season"
    prog = None
    nhl_today = _nhl_today_payload(session)
    nhl_strip = _nhl_calendar_strip(session)
    season_lbl = f"{session.season_calendar_year}ΓÇô{int(session.season_calendar_year) + 1}"
    if session.phase == "regular" and session.nhl_calendar:
        last = int(session.nhl_regular_season_last_index)
        cur = int(session.calendar_cursor)
        if cur <= last:
            cd = session.nhl_calendar[cur]
            wd = str(cd.get("weekday") or "").strip()
            day_display = (
                f"Next league day: {cd.get('iso', '')}"
                + (f" ({wd})" if wd else "")
                + f" ΓÇö {cd.get('ui_phase', '')}"
            )
            prog = f"{cur + 1} / {last + 1}"
        else:
            day_display = "Regular season complete ΓÇö advance for playoffs"
            prog = f"{last + 1} / {last + 1}"
    elif session.phase == "complete":
        day_display = f"Season complete ΓÇö Cup: {session.champion_id or '?'}"

    try:
        _merge_simengine_league_news_into_storylines(session)
    except Exception:
        pass

    notifications_raw = list(session.notifications[-56:])
    notifications_norm = [_normalize_notification_payload(n, i) for i, n in enumerate(notifications_raw)]
    storylines_norm = [_normalize_storyline_payload(ev if isinstance(ev, dict) else {"headline": str(ev or "")}) for ev in list(getattr(session, "storyline_events", None) or [])[-300:]]
    storyline_choices = _storyline_choices_payload(session)
    injuries_payload = _build_injuries_payload(session)
    injury_history_payload = _build_injury_history_payload(session)

    payload = {
        "session_id": session.session_id,
        "user_team_id": str(session.user_team_id),
        "phase": session.phase,
        "season_year": session.season_calendar_year,
        "games_per_team_schedule": int(getattr(session, "games_per_team_schedule", 82) or 82),
        "calendar_summary": day_display,
        "progress": prog,
        "nhl_season_label": season_lbl,
        "nhl_today": nhl_today,
        "nhl_calendar_strip": nhl_strip,
        "nhl_calendar_full": _nhl_calendar_full_with_slates(session),
        "season_anchor_events": season_anchor_event_markers(int(session.season_calendar_year)),
        "team": {
            "id": session.user_team_id,
            "name": _display_team(user_team) if user_team else session.user_team_id,
            "coach": session.head_coach_name,
            "coach_archetype": session.coach_archetype,
            "salary_cap": float(cap_info["salary_cap"]),
            "cap_hit": float(cap_info["cap_hit"]),
            "cap_space": float(cap_info["cap_space"]),
            "cap_limit": float(cap_info["salary_cap"]),
            "record": (
                {
                    "gp": int(cal_rec["gp"]),
                    "w": int(cal_rec["w"]),
                    "l": int(cal_rec["l"]),
                    "otl": int(cal_rec["otl"]),
                    "pts": int(cal_rec["pts"]),
                }
                if int(cal_rec.get("gp") or 0) > 0
                else (
                    {
                        "gp": getattr(rec, "gp", 0),
                        "w": getattr(rec, "wins", 0),
                        "l": getattr(rec, "losses", 0),
                        "otl": getattr(rec, "otl", 0),
                        "pts": getattr(rec, "points", 0),
                    }
                    if rec
                    else None
                )
            ),
            "cap_pressure": cap_hint,
            "strategy": strat,
        },
        "pending_decisions": _pending_decision_snapshot(session),
        "pendingDecisions": _pending_decision_snapshot(session),
        "storyline_choices": storyline_choices,
        "notifications": notifications_norm,
        "timeline": list(session.timeline[-80:]),
        "storyline_events": storylines_norm,
        "injuries": injuries_payload,
        "injury_history": injury_history_payload,
        "roster": roster_rows[:28],
        "calendar_events": list(getattr(session, "calendar_events", []) or []),
        "schedule_diagnostics": getattr(session, "schedule_diagnostics", {}) or {},
        "roster_browser": _build_roster_browser(sim, str(session.user_team_id), franchise_session=session),
        "draft_class_rankings": build_draft_class_rankings(session, sim),
        "standings": standings_rows[:32],
        "stats_central": _build_stats_central_payload(session),
        "schedule_upcoming": _build_schedule_upcoming(session, limit=14),
        "flags": {
            "playoffs_done": session.playoffs_simulated,
            "can_advance": len(session.pending_decisions) == 0 and session.phase != "complete",
        },
        "pending_ui_popups": list(getattr(session, "pending_ui_popups", None) or []),
        "pendingUiPopups": list(getattr(session, "pending_ui_popups", None) or []),
        "showcase_archive": list(getattr(session, "showcase_archive", None) or [])[-24:],
    }
    try:
        from app.sim_engine.franchise.offseason import build_offseason_state_extras

        extras = build_offseason_state_extras(session)
        extra_flags = dict(extras.pop("flags", {}) or {})
        payload.update(extras)
        payload["flags"] = {**payload.get("flags", {}), **extra_flags}
        payload["phase"] = str(session.phase)
        payload["season_phase"] = str(getattr(session, "season_phase", session.phase) or session.phase)
        payload["next_important_event"] = str(getattr(session, "next_important_event", "") or "")
        payload["playoff_payload"] = dict(getattr(session, "playoff_payload", None) or {})
    except Exception:
        pass
    return payload
