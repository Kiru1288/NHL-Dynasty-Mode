"""Pending user decisions and storyline choices."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

def _maybe_enqueue_post_day_decisions(session: FranchiseSession, user_lines: List[str]) -> None:
    """Lightweight GM prompts derived from engine state (no extra full-season sim)."""
    sim = session.sim
    user_team = session.team_by_id.get(session.user_team_id)
    if user_team is None:
        return
    r = sim.rng

    # Injury prompts for user club (moderate + major; cap 2/day, same calendar day only)
    just_finished_idx = int(session.calendar_cursor) - 1
    decisions_added = 0
    for inj in reversed((session.injury_log_all or [])[-15:]):
        if int(inj.get("date", -1)) != just_finished_idx:
            continue
        if str(inj.get("team_id") or "") != str(session.user_team_id):
            continue
        tier = str(inj.get("tier") or "").lower()
        if tier not in ("major", "moderate"):
            continue
        pname = str(inj.get("player_name") or inj.get("player") or "Player")
        games = int(inj.get("games") or 0)
        storyline_id = f"story_injury_protocol_{str(inj.get('id') or pname).replace(' ', '_')}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "injury_protocol",
                "priority": "CRITICAL" if tier == "major" else "HIGH",
                "title": "Medical staff report",
                "description": f"{pname} ΓÇö {tier} injury (~{games} games). Choose how you message the room.",
                "options": [
                    {
                        "id": "transparent",
                        "label": "Transparent update (builds trust)",
                        "effects": {"morale_delta": 2, "media_noise_delta": 1},
                        "effect_summary": "Boosts room trust, slightly increases media cycle.",
                    },
                    {
                        "id": "minimize",
                        "label": "Minimize publicly (reduces media noise)",
                        "effects": {"morale_delta": 1, "media_noise_delta": -1},
                        "effect_summary": "Keeps coverage quiet with a small trust bump.",
                    },
                    {
                        "id": "next_man",
                        "label": "Next-man-up rhetoric (pressure on depth)",
                        "effects": {"morale_delta": 0, "depth_pressure_delta": 2},
                        "effect_summary": "Signals urgency; depth players absorb extra pressure.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "injury": inj,
                    "player_name": pname,
                    "team_id": str(session.user_team_id),
                    "cause": f"{pname} suffered a {tier} injury (~{games} games).",
                },
            }
        )
        decisions_added += 1
        if decisions_added >= 2:
            break

    if user_lines and r.random() < 0.22:
        roster = [p for p in (getattr(user_team, "roster", None) or []) if not getattr(p, "retired", False)]
        if roster:
            p = r.choice(roster)
            ident = getattr(p, "identity", None)
            nm = str(getattr(ident, "name", None) or getattr(p, "name", None) or "Player")
            role = float(getattr(getattr(p, "psych", None), "role_satisfaction", 0.55) or 0.55)
            if role < 0.62 or r.random() < 0.3:
                dec_id = f"dec_{uuid.uuid4().hex[:12]}"
                storyline_id = f"story_ice_time_{str(nm).replace(' ', '_').lower()}_{just_finished_idx}"
                session.pending_decisions.append(
                    {
                        "id": dec_id,
                        "storyline_id": storyline_id,
                        "kind": "ice_time",
                        "priority": "MEDIUM",
                        "title": f"{nm} wants a larger role",
                        "description": "Agents and internal scouts disagree on fit. Your call.",
                        "options": [
                            {
                                "id": "promote",
                                "label": "Promote usage (+ morale short-term, fatigue risk)",
                                "effects": {"morale_delta": 2, "fatigue_delta": 2},
                                "effect_summary": "Immediate confidence boost with heavier workload risk.",
                            },
                            {
                                "id": "steady",
                                "label": "Hold structure (stable room)",
                                "effects": {"morale_delta": 1, "fatigue_delta": 0},
                                "effect_summary": "Keeps current deployment and room balance intact.",
                            },
                            {
                                "id": "bench_msg",
                                "label": "Send message with minutes cut (discipline)",
                                "effects": {"morale_delta": -2, "fatigue_delta": -1},
                                "effect_summary": "Lowers satisfaction but protects energy and hierarchy.",
                            },
                        ],
                        "meta": {
                            "storyline_id": storyline_id,
                            "player_name": nm,
                            "team_id": str(session.user_team_id),
                            "cause": f"{nm}'s camp requested a larger role.",
                        },
                    }
                )

    if not user_lines and r.random() < 0.12:
        storyline_id = f"story_trade_inquiry_{just_finished_idx}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "trade_inquiry",
                "priority": "MEDIUM",
                "title": "Trade desk ping",
                "description": "Rival GM floats a futures-for-help concept. No names on paper yet.",
                "options": [
                    {
                        "id": "listen",
                        "label": "Stay open ΓÇö scouting will dig",
                        "effects": {"trade_activity_delta": 2, "asset_risk_delta": 1},
                        "effect_summary": "Increases market optionality with mild valuation risk.",
                    },
                    {
                        "id": "decline",
                        "label": "Decline politely",
                        "effects": {"trade_activity_delta": -1, "asset_risk_delta": -1},
                        "effect_summary": "Preserves assets and short-term stability.",
                    },
                    {
                        "id": "counter",
                        "label": "Counter with salary retention ask",
                        "effects": {"trade_activity_delta": 1, "cap_flex_delta": 1},
                        "effect_summary": "Keeps talks alive while targeting cap leverage.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "team_id": str(session.user_team_id),
                    "cause": "A rival GM floated a futures-for-help concept.",
                },
            }
        )
def _auto_resolve_pending_decisions(session: FranchiseSession) -> None:
    """
    Bulk sim helper only.

    Manual Advance Day should not call this.
    This is intentionally conservative and logs every forced choice.
    """
    while getattr(session, "pending_decisions", None):
        d = session.pending_decisions[0]

        if not isinstance(d, dict):
            session.pending_decisions.pop(0)
            continue

        opts = d.get("options") or d.get("choices") or []

        if not opts:
            session.timeline.append(
                f"AUTO-RESOLVE: removed decision with no options ({d.get('id') or d.get('kind') or 'unknown'})."
            )
            session.pending_decisions.pop(0)
            continue

        first = opts[0] if isinstance(opts[0], dict) else {"id": str(opts[0])}
        choice_id = str(first.get("id") or first.get("choice_id") or "")

        if not choice_id:
            session.timeline.append(
                f"AUTO-RESOLVE: removed malformed decision ({d.get('id') or d.get('kind') or 'unknown'})."
            )
            session.pending_decisions.pop(0)
            continue

        session.timeline.append(
            f"AUTO-RESOLVE: {d.get('kind') or d.get('type') or 'decision'} "
            f"{d.get('id') or ''} -> {choice_id}"
        )

        apply_decision(session, str(d.get("id") or ""), choice_id)
def _apply_injury_decision_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    injury = dict(meta.get("injury") or {})
    player_name = str(meta.get("player_name") or injury.get("player_name") or injury.get("player") or "Player")
    player_id = str(meta.get("player_id") or injury.get("player_id") or "")
    choice_id = str(choice.get("id") or "")

    effects: Dict[str, Any] = {}

    if user_team is None:
        return effects

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    if choice_id == "transparent":
        changed = _nudge_team_room(user_team, morale=0.012, confidence=0.006, role_satisfaction=0.018)
        if target is not None:
            _nudge_player_psych(target, morale=0.025, confidence=0.01, role_satisfaction=0.03)
        effects.update({"room_trust_delta": 2, "media_noise_delta": 1, "players_affected": changed})

    elif choice_id == "minimize":
        changed = _nudge_team_room(user_team, morale=0.004, confidence=0.004, role_satisfaction=-0.004)
        effects.update({"room_trust_delta": 1, "media_noise_delta": -1, "players_affected": changed})

    elif choice_id == "next_man":
        changed = _nudge_team_room(user_team, morale=0.002, confidence=0.012, role_satisfaction=-0.008)
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) + 0.06)
        effects.update({"depth_pressure_delta": 2, "confidence_delta": 1, "players_affected": changed})

    elif choice_id == "call_up_player":
        setattr(user_team, "_needs_callup", True)
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) - 0.02)
        changed = _nudge_team_room(user_team, morale=0.004, confidence=0.004)
        effects.update({"callup_flag": 1, "depth_pressure_delta": -1, "players_affected": changed})

    elif choice_id == "shuffle_lines":
        setattr(user_team, "_lines_shuffled", True)
        changed = _nudge_team_room(user_team, morale=0.002, confidence=0.006, role_satisfaction=0.006)
        effects.update({"line_chemistry_volatility": 1, "players_affected": changed})

    elif choice_id == "play_short_roster":
        setattr(user_team, "_depth_pressure", float(getattr(user_team, "_depth_pressure", 0.0) or 0.0) + 0.10)
        changed = _nudge_team_room(user_team, morale=-0.004, confidence=-0.004, role_satisfaction=-0.012)
        effects.update({"depth_pressure_delta": 3, "fatigue_risk_delta": 2, "players_affected": changed})

    elif choice_id == "place_on_ir":
        setattr(user_team, "_ir_management_used", int(getattr(user_team, "_ir_management_used", 0) or 0) + 1)
        setattr(user_team, "_needs_callup", True)
        effects.update({"ir_used": 1, "callup_flag": 1, "cap_flexibility_delta": 1})

    return effects
def _apply_ice_time_decision_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    player_name = str(meta.get("player_name") or "")
    player_id = str(meta.get("player_id") or "")
    choice_id = str(choice.get("id") or "")

    effects: Dict[str, Any] = {}

    if user_team is None:
        return effects

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    if target is None:
        return effects

    if choice_id == "promote":
        _nudge_player_psych(target, morale=0.02, confidence=0.035, role_satisfaction=0.10)
        setattr(target, "_temporary_role_boost_games", 5)
        effects.update({"role_satisfaction_delta": 3, "confidence_delta": 2, "temporary_role_boost_games": 5})

    elif choice_id == "bench_msg":
        _nudge_player_psych(target, morale=-0.015, confidence=-0.012, role_satisfaction=-0.12)
        setattr(target, "_accountability_pressure_games", 4)
        effects.update({"role_satisfaction_delta": -3, "accountability_pressure_games": 4})

    else:
        _nudge_player_psych(target, morale=0.006, confidence=0.006, role_satisfaction=0.02)
        effects.update({"role_satisfaction_delta": 1})

    return effects
def _apply_generic_storyline_choice_effect(
    session: FranchiseSession,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fallback for any future storyline choice.
    Reads choice.effects and converts them into visible player/team nudges.
    """
    user_team = session.team_by_id.get(str(session.user_team_id))
    meta = dict(decision.get("meta") or {})
    player_name = str(meta.get("player_name") or "")
    player_id = str(meta.get("player_id") or "")
    raw_effects = dict(choice.get("effects") or {})

    applied: Dict[str, Any] = {}

    if user_team is None:
        return applied

    target = _find_player_on_team_by_id_or_name(user_team, player_id=player_id, player_name=player_name)

    morale = float(raw_effects.get("morale_delta", raw_effects.get("morale", 0)) or 0) / 100.0
    confidence = float(raw_effects.get("confidence_delta", raw_effects.get("confidence", 0)) or 0) / 100.0
    role = float(raw_effects.get("role_satisfaction_delta", raw_effects.get("role", 0)) or 0) / 100.0

    if target is not None:
        _nudge_player_psych(target, morale=morale, confidence=confidence, role_satisfaction=role)
        applied["target_player_affected"] = 1
    elif any([morale, confidence, role]):
        changed = _nudge_team_room(
            user_team,
            morale=morale * 0.35,
            confidence=confidence * 0.35,
            role_satisfaction=role * 0.35,
        )
        applied["players_affected"] = changed

    for k, v in raw_effects.items():
        applied[k] = v

    return applied
def _pending_decision_snapshot(session: FranchiseSession) -> List[Dict[str, Any]]:
    """
    Small frontend-safe decision payload.
    Used when Advance Day stops before simming.
    """
    out: List[Dict[str, Any]] = []

    for index, d in enumerate(list(getattr(session, "pending_decisions", None) or [])):
        if not isinstance(d, dict):
            continue

        meta = dict(d.get("meta") or {})
        opts = list(d.get("options") or d.get("choices") or [])

        safe_options: List[Dict[str, Any]] = []
        for opt in opts:
            if not isinstance(opt, dict):
                continue

            safe_options.append(
                {
                    "id": str(opt.get("id") or opt.get("choice_id") or ""),
                    "label": str(opt.get("label") or opt.get("title") or opt.get("text") or "Choice"),
                    "description": str(opt.get("description") or opt.get("details") or ""),
                }
            )

        out.append(
            {
                "id": str(d.get("id") or f"decision:{index}"),
                "kind": str(d.get("kind") or d.get("type") or "decision"),
                "title": str(
                    d.get("title")
                    or d.get("headline")
                    or meta.get("title")
                    or "Decision Required"
                ),
                "message": str(
                    d.get("message")
                    or d.get("text")
                    or d.get("details")
                    or meta.get("message")
                    or "Resolve this item before advancing."
                ),
                "priority": str(d.get("priority") or meta.get("priority") or "HIGH").upper(),
                "calendar_day": int(
                    d.get("calendar_day")
                    or d.get("date")
                    or meta.get("calendar_day")
                    or getattr(session, "calendar_cursor", 0)
                    or 0
                ),
                "calendar_iso": str(
                    d.get("calendar_iso")
                    or d.get("calendarIso")
                    or meta.get("calendar_iso")
                    or _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
                    or ""
                ),
                "team_id": str(d.get("team_id") or meta.get("team_id") or ""),
                "player_id": str(d.get("player_id") or meta.get("player_id") or ""),
                "player_name": str(d.get("player_name") or meta.get("player_name") or ""),
                "options": safe_options,
                "meta": meta,
            }
        )

    return out
def _advance_blocked_result(
    session: FranchiseSession,
    *,
    reason: str,
    message: str,
) -> Dict[str, Any]:
    """
    Return a normal object instead of forcing the frontend to interpret a crash.
    """
    return {
        "status": "blocked",
        "mode": "day",
        "reason": str(reason or "blocked"),
        "message": str(message or "Resolve pending decisions before advancing."),
        "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
        "pending_decisions": _pending_decision_snapshot(session),
    }
def auto_resolve_franchise_decisions(session: FranchiseSession) -> None:
    """Public alias for API use before a single manual advance."""
    _auto_resolve_pending_decisions(session)
def apply_storyline_choice(session: FranchiseSession, storyline_id: str, choice_id: str) -> None:
    """
    Resolve a choice by storyline id.

    Frontend story cards usually know storyline_id, not pending decision id.
    """
    sid = str(storyline_id or "").strip()
    cid = str(choice_id or "").strip()

    if not sid:
        raise ValueError("Storyline id is required.")

    if not cid:
        raise ValueError("Choice id is required.")

    for d in list(getattr(session, "pending_decisions", None) or []):
        if not isinstance(d, dict):
            continue

        meta = dict(d.get("meta") or {})
        dec_story_id = str(
            meta.get("storyline_id")
            or d.get("storyline_id")
            or d.get("id")
            or ""
        )

        if dec_story_id == sid:
            apply_decision(session, str(d.get("id") or ""), cid)
            return

    raise ValueError(f"Storyline choice target not found: {sid}")
def apply_decision(session: FranchiseSession, decision_id: str, choice_id: str) -> None:
    """
    Resolve a pending user decision and apply real effects.

    This now:
    - validates decision id
    - validates choice id
    - applies injury/ice-time/storyline effects
    - writes visible feedback to timeline/calendar/storylines/notifications
    - removes matching popups so the UI does not keep showing stale blockers
    """
    did = str(decision_id or "").strip()
    cid = str(choice_id or "").strip()

    if not did:
        raise ValueError("Decision id is required.")

    if not cid:
        raise ValueError("Choice id is required.")

    pending = list(getattr(session, "pending_decisions", None) or [])

    for i, d in enumerate(pending):
        if not isinstance(d, dict):
            continue

        if str(d.get("id") or "") != did:
            continue

        kind = str(d.get("kind") or d.get("type") or "decision")
        options = list(d.get("options") or d.get("choices") or [])
        chosen: Optional[Dict[str, Any]] = None

        for opt in options:
            if not isinstance(opt, dict):
                continue
            if str(opt.get("id") or opt.get("choice_id") or "") == cid:
                chosen = opt
                break

        if chosen is None:
            raise ValueError(f"Choice {cid!r} not found for decision {did!r}.")

        # Remove the decision first so retry loops do not double-apply it.
        session.pending_decisions.pop(i)

        effects: Dict[str, Any] = {}

        if kind in ("injury_protocol", "injury_decision"):
            effects.update(_apply_injury_decision_effect(session, d, chosen))

        elif kind == "ice_time":
            effects.update(_apply_ice_time_decision_effect(session, d, chosen))

        elif kind == "wjc_u20_loan":
            meta = dict(d.get("meta") or {})
            pid = str(meta.get("player_id") or "").strip()

            if pid:
                if not hasattr(session, "wjc_nhl_u20_loan") or session.wjc_nhl_u20_loan is None:
                    session.wjc_nhl_u20_loan = {}

                session.wjc_nhl_u20_loan[pid] = bool(cid == "loan")
                effects["wjc_loan"] = 1 if cid == "loan" else 0

        elif kind == "retirement_decision":
            from app.sim_engine.franchise.retirement import apply_retirement_decision

            effects.update(apply_retirement_decision(session, d, cid))

        else:
            effects.update(_apply_generic_storyline_choice_effect(session, d, chosen))

        # Merge visible declared effects with actual applied effects.
        declared = dict(chosen.get("effects") or {})
        final_effects = {**declared, **effects}

        meta = dict(d.get("meta") or {})
        title = str(d.get("title") or d.get("headline") or "Decision Resolved")
        label = str(chosen.get("label") or cid)
        player_name = str(meta.get("player_name") or d.get("player_name") or "")

        headline = f"{title}: {label}"
        summary = f"You chose: {label}."
        if player_name:
            summary = f"{player_name} ΓÇö {summary}"

        if chosen.get("effect_summary"):
            summary += f" {chosen.get('effect_summary')}"

        _append_decision_feedback(
            session,
            decision=d,
            choice=chosen,
            headline=headline,
            summary=summary,
            priority=str(d.get("priority") or "MEDIUM").upper(),
            effects=final_effects,
        )

        # Clear stale UI popups tied to this decision/storyline.
        story_id = str(meta.get("storyline_id") or d.get("storyline_id") or "")
        next_popups: List[Dict[str, Any]] = []

        for popup in list(getattr(session, "pending_ui_popups", None) or []):
            if not isinstance(popup, dict):
                continue

            popup_decision_id = str(popup.get("decision_id") or popup.get("id") or "")
            popup_story_id = str(popup.get("storyline_id") or "")

            if popup_decision_id == did:
                continue

            if story_id and popup_story_id == story_id:
                continue

            next_popups.append(popup)

        session.pending_ui_popups = next_popups
        session.timeline.append(f"Decision ({kind}): {did} -> {cid}")

        return

    raise ValueError(f"Decision {did!r} not found.")
def _player_display_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    if ident is not None and getattr(ident, "name", None):
        return str(ident.name)
    return str(getattr(player, "name", None) or "Player")
def _find_player_on_team_by_id_or_name(team: Any, *, player_id: str = "", player_name: str = "") -> Optional[Any]:
    pid = str(player_id or "").strip()
    pname = str(player_name or "").strip().lower()

    for p in getattr(team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue

        p_id = str(
            getattr(p, "player_id", None)
            or getattr(p, "id", None)
            or getattr(p, "uid", None)
            or ""
        ).strip()

        if pid and p_id and p_id == pid:
            return p

        pn = _player_display_name(p).strip().lower()

        if pname and pn == pname:
            return p

    return None
def _nudge_player_psych(
    player: Any,
    *,
    morale: float = 0.0,
    confidence: float = 0.0,
    role_satisfaction: float = 0.0,
) -> None:
    psych = getattr(player, "psych", None)

    if psych is None:
        return

    for attr, delta in (
        ("morale", morale),
        ("confidence", confidence),
        ("role_satisfaction", role_satisfaction),
    ):
        if not delta:
            continue

        try:
            cur = float(getattr(psych, attr, 0.5) or 0.5)
            setattr(psych, attr, _clamp(cur + float(delta)))
        except Exception:
            pass
def _nudge_team_room(
    team: Any,
    *,
    morale: float = 0.0,
    confidence: float = 0.0,
    role_satisfaction: float = 0.0,
    limit: int = 28,
) -> int:
    changed = 0

    for p in getattr(team, "roster", None) or []:
        if changed >= int(limit):
            break

        if getattr(p, "retired", False):
            continue

        if getattr(p, "psych", None) is None:
            continue

        _nudge_player_psych(
            p,
            morale=morale,
            confidence=confidence,
            role_satisfaction=role_satisfaction,
        )
        changed += 1

    return changed
def _append_decision_feedback(
    session: FranchiseSession,
    *,
    decision: Dict[str, Any],
    choice: Dict[str, Any],
    headline: str,
    summary: str,
    priority: str = "MEDIUM",
    effects: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Every resolved user choice should leave visible evidence:
    timeline + notification + storyline + calendar event.
    """
    _ensure_session_event_lists(session)

    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    iso = _calendar_iso_for_day(session, cur)
    did = str(decision.get("id") or uuid.uuid4().hex[:10])
    cid = str(choice.get("id") or "choice")
    kind = str(decision.get("kind") or decision.get("type") or "decision")
    meta = dict(decision.get("meta") or {})
    team_id = str(meta.get("team_id") or decision.get("team_id") or getattr(session, "user_team_id", "") or "")
    player_id = str(meta.get("player_id") or decision.get("player_id") or "")
    player_name = str(meta.get("player_name") or decision.get("player_name") or "")

    event_id = f"decision:{did}:{cid}"

    row = {
        "id": event_id,
        "kind": "decision_result",
        "type": "decision_result",
        "calendar_day": cur,
        "date": cur,
        "calendar_iso": iso,
        "title": headline,
        "headline": headline,
        "summary": summary,
        "description": summary,
        "priority": str(priority or "MEDIUM").upper(),
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "decision_kind": kind,
        "choice_id": cid,
        "choice_label": str(choice.get("label") or cid),
        "effects": effects or dict(choice.get("effects") or {}),
        "effect_summary": str(choice.get("effect_summary") or choice.get("effectSummary") or ""),
        "surfaces": ["calendar", "storylines", "notifications", "timeline"],
    }

    _append_unique_dict_event(session.calendar_events, row)

    session.notifications.append(
        _normalized_notification(
            notification_id=f"notif:{event_id}",
            notification_type="decision_result",
            text=summary,
            priority=str(priority or "MEDIUM").upper(),
            calendar_day=cur,
            calendar_iso=iso,
            team_id=team_id,
            player_id=player_id,
            source="user_decision",
            extra={
                "decision_kind": kind,
                "choice_id": cid,
                "choice_label": str(choice.get("label") or cid),
            },
        )
    )

    _record_storyline(
        session,
        {
            "id": f"story:{event_id}",
            "type": "decision_result",
            "kind": "decision_result",
            "headline": headline,
            "details": summary,
            "cause": str(decision.get("description") or decision.get("message") or ""),
            "effects": effects or dict(choice.get("effects") or {}),
            "effect_summary": str(choice.get("effect_summary") or choice.get("effectSummary") or ""),
            "team": team_id,
            "team_id": team_id,
            "player_id": player_id,
            "player_name": player_name,
            "players": [player_name] if player_name else [],
            "priority": str(priority or "MEDIUM").upper(),
            "date": cur,
            "calendar_day": cur,
            "calendar_iso": iso,
            "surfaces": ["storylines", "calendar"],
        },
    )

    session.timeline.append(
        _normalized_timeline_event(
            event_id=f"timeline:{event_id}",
            event_type="decision_result",
            text=f"{headline}: {summary}",
            calendar_day=cur,
            calendar_iso=iso,
            team_id=team_id,
            player_id=player_id,
            priority=str(priority or "MEDIUM").upper(),
            extra={"choice_id": cid, "decision_kind": kind},
        )
    )
