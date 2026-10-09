"""
Franchise session trade service — bridges API to SimEngine trade module.
"""

from __future__ import annotations
import os

from typing import Any, Dict, List, Optional

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()

from app.sim_engine.trades.trade_evaluator import evaluate_trade_package  # noqa: E402
from app.sim_engine.trades.trade_executor import execute_validated_trade  # noqa: E402
from app.sim_engine.trades.trade_history import get_trade_history  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import (  # noqa: E402
    audit_pick_registry_integrity,
    ensure_franchise_pick_registry,
    serialize_team_picks,
    tradeable_draft_year,
    upcoming_draft_year,
)
from app.sim_engine.trades.trade_value import TRADE_VALUE_FORMULA_VERSION, pick_value_hint
from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot  # noqa: E402
from app.sim_engine.economy.team_needs import TeamNeeds  # noqa: E402
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

# Kept in backend (uvicorn-watched) so formula bumps reload the API process.
# Keep in sync with SimEngine TRADE_VALUE_FORMULA_VERSION.
TRADE_ASSETS_CACHE_VERSION = int(TRADE_VALUE_FORMULA_VERSION) + 5  # bump: clause kick-in year


def _summarize_team_needs(needs: Dict[str, float], direction: str) -> Dict[str, Any]:
    labels: List[str] = []
    if float(needs.get("top_line_forward", 0.0)) >= 0.55:
        labels.append("Top-line F")
    if float(needs.get("top_4_defense", 0.0)) >= 0.55:
        labels.append("Top-4 D")
    if float(needs.get("goalie", 0.0)) >= 0.55:
        labels.append("G")
    if float(needs.get("depth_forward", 0.0)) >= 0.55:
        labels.append("Depth F")
    window = str(direction or "unknown").lower()
    shopping: List[str] = []
    values: List[str] = []
    if window == "rebuild":
        values.extend(["Picks", "Prospects", "Cap space"])
        shopping.append("Veteran salary")
    elif window == "contender":
        values.extend(["NHL-ready talent", "Playoff experience"])
        shopping.append("Rental upgrades")
    else:
        values.append("Balanced assets")
    return {
        "needs_short": labels,
        "shopping": shopping,
        "values": values,
    }


def _organizational_depth_summary(team: Any) -> Dict[str, Any]:
    try:
        from services.roster_compliance import summarize_team_roster_capacity

        capacity = summarize_team_roster_capacity(team)
    except Exception:
        capacity = {}

    roster = list(getattr(team, "roster", None) or [])
    ahl = list(getattr(team, "ahl_roster", None) or [])

    def _ovr(p: Any) -> float:
        fn = getattr(p, "ovr", None)
        try:
            v = float(fn() if callable(fn) else fn or 0)
        except Exception:
            v = 0.0
        return v * 99.0 if v <= 1.5 else v

    def _pos(p: Any) -> str:
        try:
            from services.roster_compliance import position_bucket

            return position_bucket(p)
        except Exception:
            ident = getattr(p, "identity", None)
            pos = getattr(ident, "position", None) if ident else getattr(p, "position", "")
            s = str(getattr(pos, "value", pos) or "").upper()
            if s in ("LW", "RW", "W", "F"):
                return "F"
            return s

    fwds = sorted([p for p in roster if _pos(p) in ("C", "F")], key=_ovr, reverse=True)
    defs = sorted([p for p in roster if _pos(p) == "D"], key=_ovr, reverse=True)
    gs = sorted([p for p in roster if _pos(p) == "G"], key=_ovr, reverse=True)
    prospects = sorted(ahl, key=_ovr, reverse=True)[:6]

    return {
        "nhl_count": int(capacity.get("nhl_count") if capacity else len(roster)),
        "ahl_count": int(capacity.get("ahl_count") if capacity else len(ahl)),
        "top_six_forwards": len(fwds[:6]),
        "top_four_defense": len(defs[:4]),
        "goalies": int(capacity.get("goalies") if capacity else len(gs)),
        "forwards": int(capacity.get("forwards") or 0),
        "defense": int(capacity.get("defense") or 0),
        "prospect_count": len(prospects),
        "pipeline_strength": round(
            sum(_ovr(p) for p in prospects[:3]) / max(1, min(3, len(prospects))),
            1,
        ) if prospects else 0.0,
    }


def _team_roster_capacity_payload(team: Any) -> Dict[str, Any]:
    from services.roster_compliance import summarize_team_roster_capacity

    return summarize_team_roster_capacity(team)


def _team_contract_slots_payload(team: Any, league: Any = None) -> Dict[str, Any]:
    from services.roster_compliance import summarize_team_contract_slots

    return summarize_team_contract_slots(team, league=league)


def _trade_context(session: Any) -> Dict[str, Any]:
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is not None:
        # Trade value reads current-season production from the live ledger.
        try:
            setattr(league, "player_season_stats", getattr(session, "player_season_stats", None))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        # Pick / finish projections read each club's live record.
        try:
            from app.sim_engine.trades.trade_value import sync_trade_standings

            sync_trade_standings(league, getattr(session, "standings", None))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    cal = getattr(session, "nhl_calendar", None) or []
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 192) or 192))
    from services.franchise_sim import publish_trade_deadline_state

    deadline = publish_trade_deadline_state(session)
    deadline_phase = float(deadline["deadline_phase"])
    calendar_iso = ""
    if 0 <= cursor < len(cal):
        calendar_iso = str(cal[cursor].get("iso") or "")

    season_y = int(getattr(session, "season_calendar_year", 2025) or 2025)
    draft_y = upcoming_draft_year(season_y)
    draft_done = bool(getattr(session, "draft_completed", False))
    trade_y = tradeable_draft_year(season_y, draft_completed=draft_done)
    try:
        from services.franchise_entry_draft import build_known_pick_slots

        known_slots = build_known_pick_slots(session)
    except Exception:
        known_slots = {}
    draft_state = getattr(session, "draft_state", None) or {}
    draft_day_live = bool(draft_state.get("draft_started")) and not bool(
        draft_state.get("draft_completed") or draft_done
    )
    # Clause consents (NMC / NTC waivers) are windowed: drop expired ones first.
    from app.sim_engine.trades.clause_consent import prune_expired_consents, waiver_window

    try:
        prune_expired_consents(session)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    clause_window = waiver_window(session)
    return {
        "sim": sim,
        "league": league,
        "team_by_id": dict(session.team_by_id or {}),
        "user_team_id": str(session.user_team_id),
        "season_year": season_y,
        "draft_year": draft_y,
        "tradeable_draft_year": trade_y,
        "draft_completed": draft_done,
        "draft_day_trade": draft_day_live,
        "known_pick_slots": known_slots,
        "season_is_calendar": True,
        "use_upcoming_draft_year": True,
        "calendar_cursor": cursor,
        "calendar_iso": calendar_iso,
        "regular_season_last_index": max_d,
        "deadline_phase": deadline_phase,
        "days_to_deadline": deadline["days_to_deadline"],
        "trade_deadline_passed": deadline["freeze_active"],
        "tank_pressure_by_team": dict(getattr(session, "transcendent_tank_pressure", None) or {}),
        "transcendent_active": bool(getattr(session, "transcendent_draft_prospect_id", None)),
        "standings": getattr(session, "standings", None),
        "ntc_waivers": dict(getattr(session, "ntc_waivers", None) or {}),
        "clause_consent_authoritative": True,
        "clause_window_key": clause_window.get("key"),
        "player_season_stats": getattr(session, "player_season_stats", None),
        # Rare "offer you can't refuse" packages the Trade Finder surfaced: still legal,
        # but the desperate club has already agreed to the price.
        "desperate_offer_keys": {
            k for k, exp in (getattr(session, "_desperate_offers", None) or {}).items()
            if int(exp or 0) >= int(cursor or 0)
        },
    }


def request_ntc_waiver(
    session: Any,
    *,
    player_id: str,
    source_team_id: str,
    destination_team_id: str,
) -> Dict[str, Any]:
    """Ask an NTC player to waive for a destination; caches the result on the session."""
    ctx = _trade_context(session)
    team_by_id = ctx["team_by_id"] or {}
    src = team_by_id.get(str(source_team_id))
    dest = team_by_id.get(str(destination_team_id))
    if src is None:
        raise ValueError(f"Unknown source team: {source_team_id}")
    if dest is None:
        raise ValueError(f"Unknown destination team: {destination_team_id}")

    from app.sim_engine.trades.trade_asset import find_player_on_team_roster

    player, _ = find_player_on_team_roster(src, str(player_id))
    if player is None:
        raise ValueError(f"Player {player_id} not found on {source_team_id} NHL roster")

    from app.sim_engine.trades.clause_consent import request_clause_waiver

    decision = request_clause_waiver(
        session, player, source_team=src, destination_team=dest, origin="trade_hub",
        team_by_id=team_by_id, context=ctx,
    )
    return decision


def _ensure_trade_infrastructure(session: Any) -> Dict[str, Any]:
    """Prepare the league for a trade call and return the trade context (built once)."""
    ctx = _trade_context(session)
    league = ctx["league"]
    if league is None:
        return ctx
    try:
        setattr(league, "season_year", int(ctx["season_year"]))
        setattr(league, "current_season", int(ctx["season_year"]))
        setattr(league, "draft_year", int(ctx.get("tradeable_draft_year") or ctx["draft_year"]))
        setattr(league, "draft_completed", bool(ctx.get("draft_completed")))
        setattr(league, "season_is_calendar", True)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    ensure_franchise_pick_registry(
        league,
        season_calendar_year=int(ctx["season_year"]),
        years_ahead=4,
        draft_completed=bool(ctx.get("draft_completed")),
    )
    try:
        from services.franchise_entry_draft import build_known_pick_slots

        ctx["known_pick_slots"] = build_known_pick_slots(session)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return ctx


def _trade_state_signature(session: Any, assets_by_team: Dict[str, Any]) -> str:
    import hashlib

    league = getattr(getattr(session, "sim", None), "league", None)
    parts = [
        str(int(getattr(session, "calendar_cursor", 0) or 0)),
        str(getattr(session, "phase", "") or ""),
        str(len(getattr(league, "trade_history", None) or [])),
        str(len(getattr(session, "ntc_waivers", None) or {})),
    ]
    team_by_id = getattr(session, "team_by_id", None) or {}
    for tid in sorted(str(t) for t in (assets_by_team or {}).keys()):
        tm = team_by_id.get(tid)
        if tm is None:
            continue
        for attr in ("roster", "ahl_roster", "prospect_pool"):
            parts.append(",".join(sorted(str(getattr(p, "id", "")) for p in (getattr(tm, attr, None) or []))))
        # Injury status changes a player's value; don't reuse an evaluation from before he got hurt.
        try:
            from app.sim_engine.economy.team_needs import is_player_injured  # noqa: WPS433

            parts.append(",".join(sorted(str(getattr(p, "id", "")) for p in (getattr(tm, "roster", None) or []) if is_player_injured(p))))
        except Exception:
            pass
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


GM_PATIENCE_WARN = 3  # rejected proposals to one GM in a day before he warns you
GM_PATIENCE_LIMIT = 5  # ...and before he stops taking calls until tomorrow
PING_PONG_DAYS = 30


def _partner_of(session: Any, assets_by_team: Dict[str, Any]) -> str:
    utid = str(getattr(session, "user_team_id", "") or "")
    for tid in (assets_by_team or {}).keys():
        if str(tid) != utid:
            return str(tid)
    return ""


def _ping_pong_block(session: Any, assets_by_team: Dict[str, Any]) -> Optional[str]:
    """U21: a GM won't hand back a player he just traded you (or take back one he just sent)."""
    utid = str(getattr(session, "user_team_id", "") or "")
    partner = _partner_of(session, assets_by_team)
    if not partner:
        return None
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    league = getattr(getattr(session, "sim", None), "league", None)
    hist = list(getattr(league, "trade_history", None) or []) if league is not None else []
    moving = {}
    for tid, assets in (assets_by_team or {}).items():
        for a in assets or []:
            if str((a or {}).get("type") or "") in ("player", "prospect"):
                moving[str(a.get("id") or "")] = str(tid)
    for row in reversed(hist[-200:]):
        if not isinstance(row, dict):
            continue
        day = row.get("calendar_day")
        if day is None or cur - int(day or 0) > PING_PONG_DAYS:
            continue
        teams = {str(t) for t in (row.get("participating_teams") or [])}
        if teams != {utid, partner}:
            continue
        for m in list(row.get("moved_players") or []):
            pid = str((m or {}).get("asset_id") or "")
            # He'd be going straight back to the club that just traded him away.
            if pid in moving and str(m.get("source_team_id")) == moving[pid]:
                name = str(m.get("player_name") or "that player")
                return f"They just made the trade for {name} — they aren't reversing it this soon."
    return None


def evaluate_franchise_trade(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    record_rumor_fallout: bool = False,
) -> Dict[str, Any]:
    """Evaluate a package. ``record_rumor_fallout`` marks a real proposal (Propose button):
    it counts against the GM's patience and leaks rumours when rejected (U9, U24)."""
    partner = _partner_of(session, assets_by_team)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    patience = dict(getattr(session, "_gm_patience", None) or {})
    pkey = f"{cur}|{partner}"
    if record_rumor_fallout and partner and int(patience.get(pkey, 0) or 0) >= GM_PATIENCE_LIMIT:
        return {
            "accepted": False,
            "can_execute": False,
            "verdict": "rejected",
            "rejection_reasons": ["Their GM has stopped taking your calls today. Try again tomorrow."],
            "gm_patience": {"rejections_today": int(patience.get(pkey, 0)), "limit": GM_PATIENCE_LIMIT},
            "cap_impact": {},
        }
    public = dict(_evaluate_franchise_trade_core(session, assets_by_team=assets_by_team))
    block = _ping_pong_block(session, assets_by_team)
    if block:
        public["accepted"] = False
        public["verdict"] = "rejected"
        public["rejection_reasons"] = [block] + [r for r in list(public.get("rejection_reasons") or []) if r != block]
    if record_rumor_fallout and partner:
        if not public.get("accepted"):
            n = int(patience.get(pkey, 0) or 0) + 1
            patience = {k: v for k, v in patience.items() if str(k).startswith(f"{cur}|")}
            patience[pkey] = n
            session._gm_patience = patience
            public["gm_patience"] = {"rejections_today": n, "limit": GM_PATIENCE_LIMIT}
            if n >= GM_PATIENCE_WARN:
                left = GM_PATIENCE_LIMIT - n
                msg = (
                    "Their GM is losing patience with these offers."
                    + (f" {left} more and he stops taking your calls today." if left > 0 else " He's done for today.")
                )
                public["rejection_reasons"] = list(public.get("rejection_reasons") or []) + [msg]
        try:
            from app.sim_engine.franchise.storyline_engine import record_trade_hub_evaluation  # noqa: WPS433

            record_trade_hub_evaluation(session, public, dict(assets_by_team or {}), proposal_submitted=True)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return public


def _evaluate_franchise_trade_core(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    record_rumor_fallout: bool = False,
) -> Dict[str, Any]:
    import hashlib

    stats_rev = int(getattr(session, "_stats_revision", 0) or 0)
    assets_key = hashlib.sha256(repr(sorted((assets_by_team or {}).items())).encode()).hexdigest()[:24]
    # The verdict depends on rosters, the calendar and the trade log, not just stats:
    # key on all of them so an executed trade or a roster move never serves a stale verdict.
    cache_key = f"{stats_rev}|{assets_key}|{_trade_state_signature(session, assets_by_team)}"
    cached = getattr(session, "_trade_eval_cache", None)
    if isinstance(cached, dict) and str(cached.get("key") or "") == cache_key:
        payload = cached.get("payload")
        if isinstance(payload, dict):
            out = dict(payload)
            if not out.get("trade_id"):
                cur = int(getattr(session, "calendar_cursor", 0) or 0)
                out["trade_id"] = f"trade_{cur}_{assets_key[:8]}"
            return out
    ctx = _ensure_trade_infrastructure(session)
    try:
        from services.franchise_sim import ensure_player_financials

        league = ctx["league"]
        season_y = int(ctx["season_year"])
        touched: set[str] = set()
        for assets in (assets_by_team or {}).values():
            for asset in assets or []:
                if str(asset.get("type") or "").lower() != "player":
                    continue
                pid = str(asset.get("id") or "")
                tid = str(asset.get("team") or "")
                if not pid or pid in touched:
                    continue
                touched.add(pid)
                team = (ctx["team_by_id"] or {}).get(tid)
                if team is None:
                    continue
                for p in getattr(team, "roster", None) or []:
                    if str(getattr(p, "id", "") or "") == pid:
                        ensure_player_financials(p, league, season_y, team=team)
                        break
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    result = evaluate_trade_package(
        dict(assets_by_team or {}),
        league=ctx["league"],
        team_by_id=ctx["team_by_id"],
        context=ctx,
        user_team_id=ctx["user_team_id"],
    )
    public = {k: v for k, v in result.items() if not str(k).startswith("_")}
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    public["trade_id"] = f"trade_{cur}_{assets_key[:8]}"
    # Always expose human-readable rejection reasons + dual-team cap impact for Trade Hub.
    reasons = public.get("rejection_reasons")
    if not isinstance(reasons, list):
        public["rejection_reasons"] = [str(reasons)] if reasons else []
    else:
        public["rejection_reasons"] = [str(r) for r in reasons if r]
    cap_impact = dict(public.get("cap_impact") or {})
    public["cap_impact"] = {str(tid): dict(payload or {}) for tid, payload in cap_impact.items()}
    try:
        from services.franchise_sim import preview_trade_fan_reaction  # noqa: WPS433

        partner_id = ""
        utid = str(ctx.get("user_team_id") or "")
        for tid in (assets_by_team or {}).keys():
            if str(tid) != utid:
                partner_id = str(tid)
                break
        public["fan_reaction"] = preview_trade_fan_reaction(
            session,
            dict(assets_by_team or {}),
            public,
            partner_team_id=partner_id or None,
        )
        from services.franchise_sim import build_trade_review_payload  # noqa: WPS433

        public["trade_review"] = build_trade_review_payload(
            session,
            public,
            dict(assets_by_team or {}),
            partner_team_id=partner_id or None,
            user_team_id=utid,
            fan_reaction=public.get("fan_reaction"),
        )
        review = public.get("trade_review") or {}
        if isinstance(review, dict) and review.get("block_detail"):
            public["block_detail"] = review.get("block_detail")
    except Exception:
        public.setdefault("rejection_reasons", public.get("rejection_reasons") or [])
        public.setdefault("cap_impact", public.get("cap_impact") or {})
    # Evaluation previews must never damage player OVR.
    if record_rumor_fallout:
        try:
            from app.sim_engine.franchise.storyline_engine import record_trade_hub_evaluation  # noqa: WPS433

            record_trade_hub_evaluation(
                session,
                public,
                dict(assets_by_team or {}),
                proposal_submitted=True,
            )
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    session._trade_eval_cache = {"key": cache_key, "payload": public}
    return public


def process_pending_trade_waivers(session: Any) -> List[Dict[str, Any]]:
    """Put non-exempt players a CPU club sent down after a trade on waivers (U10)."""
    league = getattr(getattr(session, "sim", None), "league", None)
    queue = list(getattr(league, "_pending_trade_waivers", None) or []) if league is not None else []
    if not queue:
        return []
    try:
        league._pending_trade_waivers = []
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        from services.waivers import place_on_waivers
    except Exception:
        return out
    user_tid = str(getattr(session, "user_team_id", "") or "")
    teams = {str(getattr(t, "team_id", "")): t for t in list(getattr(league, "teams", None) or [])}
    for item in queue:
        tid = str((item or {}).get("team_id") or "")
        pid = str((item or {}).get("player_id") or "")
        team = teams.get(tid)
        if team is None or not pid or tid == user_tid:
            continue
        player = next((p for p in list(getattr(team, "ahl_roster", None) or []) if str(getattr(p, "id", "")) == pid), None)
        if player is None:
            continue
        try:
            res = place_on_waivers(session, team, player, reason="trade_roster_room", manual=False)
            if res and res.get("ok", True):
                out.append({"team_id": tid, "player_id": pid})
            else:
                # NMC or similar: he stays in the AHL as before.
                continue
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return out


OFFSEASON_TRADE_WINDOWS = {
    # stage → (label, max deals)
    "salary_cap": ("after the Cup", 3),
    "draft": ("on the draft floor", 4),
    "free_agency": ("on July 1", 3),
}


def run_offseason_cpu_trade_window(session: Any, stage: str) -> List[Dict[str, Any]]:
    """C6: CPU clubs trade in the summer too — after the Cup, at the draft and on July 1."""
    window = OFFSEASON_TRADE_WINDOWS.get(str(stage or ""))
    if window is None:
        return []
    done = set(getattr(session, "_offseason_trade_windows_done", None) or set())
    key = f"{int(getattr(session, 'season_calendar_year', 0) or 0)}:{stage}"
    if key in done:
        return []
    done.add(key)
    session._offseason_trade_windows_done = done
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return []
    label, cap = window
    try:
        from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

        try:
            setattr(league, "_franchise_user_team_id", str(getattr(session, "user_team_id", "") or ""))
            setattr(league, "_franchise_phase", str(getattr(session, "phase", "") or "offseason"))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        rows = propose_and_execute_cpu_trades(
            league,
            max_executions=int(cap),
            calendar_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
            regular_season_last_index=int(getattr(session, "nhl_regular_season_last_index", 0) or 192),
            offseason=True,
        )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
        return []
    session._trade_eval_cache = None
    try:
        from services.franchise_sim import _record_storyline

        for t in rows or []:
            _record_storyline(
                session,
                {
                    "type": "trade",
                    "priority": "MEDIUM",
                    "date": int(getattr(session, "calendar_cursor", 0) or 0),
                    "headline": str(t.get("headline") or "Trade completed"),
                    "summary": str(t.get("reason_text") or f"A deal {label}."),
                    "team": str(t.get("to_team_id") or ""),
                    "from_team_id": str(t.get("from_team_id") or ""),
                    "trade_id": str(t.get("trade_id") or ""),
                    "trade_category": str(t.get("trade_category") or ""),
                    "offseason_window": label,
                },
            )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        process_pending_trade_waivers(session)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return list(rows or [])


def _user_roster_overflow(session: Any, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tid = str(ctx.get("user_team_id") or "")
    team = (ctx.get("team_by_id") or {}).get(tid)
    if team is None:
        return None
    n = len(list(getattr(team, "roster", None) or []))
    if n <= 23:
        return None
    return {
        "roster_size": n,
        "limit": 23,
        "over_by": n - 23,
        "message": f"Your NHL roster is at {n}. Send down or waive {n - 23} player{'s' if n - 23 != 1 else ''} before the next game.",
    }


def execute_franchise_trade(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    record_notifications_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate and execute trade; raises ValueError on failure."""
    ctx = _ensure_trade_infrastructure(session)
    session._trade_eval_cache = None
    block = _ping_pong_block(session, assets_by_team)
    if block:
        raise ValueError(block)

    evaluation = evaluate_trade_package(
        dict(assets_by_team or {}),
        league=ctx["league"],
        team_by_id=ctx["team_by_id"],
        context=ctx,
        user_team_id=ctx["user_team_id"],
    )
    if not bool(evaluation.get("accepted")):
        # Trade rumor fallout only applies to real rejected proposals.
        try:
            from app.sim_engine.franchise.storyline_engine import record_trade_hub_evaluation  # noqa: WPS433

            public_eval = {k: v for k, v in evaluation.items() if not str(k).startswith("_")}
            record_trade_hub_evaluation(
                session,
                public_eval,
                dict(assets_by_team or {}),
                proposal_submitted=True,
            )
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    exec_result = execute_validated_trade(
        evaluation,
        league=ctx["league"],
        team_by_id=ctx["team_by_id"],
        context=ctx,
        user_team_id=ctx["user_team_id"],
    )

    try:
        from app.sim_engine.franchise.storyline_engine import apply_trade_universe_relocation  # noqa: WPS433

        history = exec_result.get("history_record") if isinstance(exec_result, dict) else {}
        for move in list((history or {}).get("moved_players") or []):
            if str((move or {}).get("asset_type") or "") != "player":
                continue
            pid = str((move or {}).get("asset_id") or "")
            from_tid = str((move or {}).get("source_team_id") or "")
            to_tid = str((move or {}).get("acquiring_team_id") or "")
            if pid and from_tid and to_tid and from_tid != to_tid:
                apply_trade_universe_relocation(
                    session,
                    player_id=pid,
                    from_team_id=from_tid,
                    to_team_id=to_tid,
                )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    try:
        from services.trade_demand_engine import clear_demands_on_trade

        moved: List[str] = []
        for side_assets in (assets_by_team or {}).values():
            for a in side_assets or []:
                if str((a or {}).get("type") or "") in ("player", "prospect"):
                    moved.append(str((a or {}).get("id") or ""))
        clear_demands_on_trade(session, moved)
        if exec_result and (not isinstance(exec_result, dict) or exec_result.get("ok", True)):
            from app.sim_engine.trades.clause_consent import consume_consents

            consume_consents(session, moved)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    try:
        placed = process_pending_trade_waivers(session)
        warn = _user_roster_overflow(session, ctx)
        if isinstance(exec_result, dict):
            if placed:
                exec_result["waiver_placements"] = placed
            if warn:
                exec_result["roster_warning"] = warn
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    if record_notifications_fn is not None:
        record_notifications_fn(session, exec_result, ctx)

    # Live draft: refresh on-clock owner immediately after any pick trade.
    try:
        state = getattr(session, "draft_state", None) or {}
        if state.get("draft_started") and not state.get("draft_completed"):
            from services.draft_pick_ownership import sync_draft_clock_after_trade

            sync = sync_draft_clock_after_trade(session)
            if isinstance(exec_result, dict):
                exec_result["draft_clock"] = sync
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    return exec_result


def build_trade_assets_payload(session: Any) -> Dict[str, Any]:
    ctx = _ensure_trade_infrastructure(session)
    league = ctx["league"]
    if os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1":
        audit = audit_pick_registry_integrity(
            league,
            start_year=int(ctx.get("tradeable_draft_year") or ctx["draft_year"]),
            years_ahead=4,
            rounds=7,
        )
        if not audit.get("ok"):
            first = (audit.get("errors") or ["unknown registry mismatch"])[0]
            print(f"[trade debug] pick registry drift before trade assets build: {first}")
    sim = ctx["sim"]
    needs_model = TeamNeeds()
    teams_out: Dict[str, Any] = {}

    def _repair_mislabeled_affiliate_spcs(tm: Any) -> None:
        """Fix affiliates that received NHL money/two-way flags but kept AHL/ECHL type labels."""
        for attr in ("ahl_roster", "echl_roster"):
            for p in list(getattr(tm, attr, None) or []):
                c = getattr(p, "contract", None)
                if c is None:
                    continue
                if isinstance(c, dict):
                    ctype = str(c.get("contract_type") or c.get("type") or "").upper()
                    two_way = bool(c.get("two_way") or c.get("is_two_way"))
                    src = str(c.get("source") or "")
                    yrs = int(c.get("years_remaining") or c.get("years") or 0)
                    aav = float(c.get("aav_m") or c.get("cap_hit_m") or 0)
                    if ctype in ("AHL", "ECHL", "AHL_ECHL") and yrs > 0 and (two_way or src == "affiliate_nhl_spc" or aav >= 0.7):
                        c["type"] = "STANDARD"
                        c["contract_type"] = "STANDARD"
                        c["is_nhl_spc"] = True
                        c["nhl_spc"] = True
                        c["standard_player_contract"] = True
                        c["two_way"] = True
                        p.contract = c
                        p.signed_status = "signed"
                else:
                    ctype = str(getattr(c, "contract_type", None) or getattr(c, "type", "") or "").upper()
                    two_way = bool(getattr(c, "two_way", False) or getattr(c, "is_two_way", False))
                    src = str(getattr(c, "source", "") or "")
                    try:
                        yrs = int(getattr(c, "years_remaining", None) or getattr(c, "years", 0) or 0)
                        aav = float(getattr(c, "aav_m", None) or getattr(c, "cap_hit_m", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if ctype in ("AHL", "ECHL", "AHL_ECHL") and yrs > 0 and (two_way or src == "affiliate_nhl_spc" or aav >= 0.7):
                        try:
                            c.type = "STANDARD"
                            c.contract_type = "STANDARD"
                            c.two_way = True
                            c.is_nhl_spc = True
                            c.nhl_spc = True
                            c.standard_player_contract = True
                            p.signed_status = "signed"
                        except Exception:
                            _swallowed_log.debug("suppressed exception", exc_info=True)

    cap_kw = {
        "season_year": int(ctx["season_year"]),
        "calendar_cursor": int(ctx["calendar_cursor"]),
        "regular_season_last_index": int(ctx["regular_season_last_index"]),
    }

    for tid, team in (ctx["team_by_id"] or {}).items():
        _repair_mislabeled_affiliate_spcs(team)
        cap_full: Dict[str, Any] = {}
        try:
            from services.contract_economy import sync_team_cap_fields

            cap_full = sync_team_cap_fields(team, league, sim, **cap_kw)
            snap = dict(cap_full.get("_raw") or {})
            if not snap:
                snap = calculate_team_cap_snapshot(
                    team,
                    league=league,
                    sim=sim,
                    season_label=f"{ctx['season_year']}-{(ctx['season_year'] + 1) % 100:02d}",
                    calendar_cursor=ctx["calendar_cursor"],
                    regular_season_last_index=ctx["regular_season_last_index"],
                )
        except Exception:
            snap = calculate_team_cap_snapshot(
                team,
                league=league,
                sim=sim,
                season_label=f"{ctx['season_year']}-{(ctx['season_year'] + 1) % 100:02d}",
                calendar_cursor=ctx["calendar_cursor"],
                regular_season_last_index=ctx["regular_season_last_index"],
            )
        needs = getattr(team, "needs", None) or needs_model.evaluate(team)
        picks = serialize_team_picks(
            league,
            tid,
            value_hint_fn=lambda row, _team=team: pick_value_hint(row, league, _team, context=ctx),
            min_year=int(ctx.get("tradeable_draft_year") or ctx.get("draft_year") or ctx["season_year"]),
        )
        # NHL rule: only picks in the next three drafts can be traded.
        _first_draft = int(ctx.get("tradeable_draft_year") or ctx.get("draft_year") or ctx["season_year"])
        picks = [it for it in picks if int(it.get("year") or 0) <= _first_draft + 2]
        for item in picks:
            orig_tid = str(item.get("original_team_id") or tid)
            orig_team = (ctx["team_by_id"] or {}).get(orig_tid) or team
            try:
                # Local import so importlib.reload(trade_value) is picked up without
                # requiring a full server restart / new franchise save.
                from app.sim_engine.trades.trade_value import (  # noqa: WPS433
                    evaluate_pick_asset_value as _eval_pick,
                )

                detail = _eval_pick(item, orig_team, orig_team, league, context=ctx)
                item["trade_value"] = detail.get("trade_value") or detail.get("total")
                item["value_tier"] = detail.get("value_tier")
                item["projected_slot"] = detail.get("projected_slot")
                item["projected_range"] = detail.get("projected_range")
                item["pick_value_context"] = detail.get("pick_value_context")
                item["value_hint"] = item["trade_value"]
                item["value_debug"] = detail.get("value_debug") or {}
            except Exception:
                item["value_debug"] = {}
        direction = str(getattr(team, "gm_window", getattr(team, "window", "unknown")) or "unknown")
        needs_summary = _summarize_team_needs(needs, direction)

        outlook: Dict[str, Any] = {}
        try:
            from services.franchise_sim import compute_team_playoff_outlook

            outlook = compute_team_playoff_outlook(session, team)
        except Exception:
            outlook = {}

        player_values: Dict[str, Any] = {}
        try:
            from services.franchise_sim import _serialize_player_trade_block

            user_tid = str(ctx.get("user_team_id") or "")
            acq_team = ctx["team_by_id"].get(user_tid) if user_tid else team
            fa_open = bool(getattr(session, "free_agency_open", False))
            sy_trade = int(getattr(session, "season_calendar_year", 2025) or 2025)
            outcomes = getattr(session, "resign_phase_outcomes", None) or {}
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                pid = str(getattr(p, "id", "") or "")
                if not pid:
                    continue
                phase_row = outcomes.get(pid) if isinstance(outcomes, dict) else None
                if isinstance(phase_row, dict) and str(phase_row.get("phase_status") or "") == "released":
                    continue
                if fa_open:
                    try:
                        from services.contract_economy import _walks_without_new_deal

                        if _walks_without_new_deal(p, sy_trade):
                            continue
                    except Exception:
                        pass
                player_values[pid] = _serialize_player_trade_block(
                    p,
                    source_team=team,
                    acquiring_team=acq_team or team,
                    league=league,
                    session=session,
                    trade_context=ctx,
                )
            from app.sim_engine.trades.trade_asset import player_holds_nhl_spc

            for attr in ("ahl_roster", "echl_roster"):
                for p in getattr(team, attr, None) or []:
                    if getattr(p, "retired", False):
                        continue
                    pid = str(getattr(p, "id", "") or "")
                    if not pid or pid in player_values:
                        continue
                    # Include every affiliate so the UI can show SPC tradeables
                    # and explain why pure AHL/ECHL deals are blocked.
                    player_values[pid] = _serialize_player_trade_block(
                        p,
                        source_team=team,
                        acquiring_team=acq_team or team,
                        league=league,
                        session=session,
                        trade_context=ctx,
                    )
                    if not player_holds_nhl_spc(p):
                        player_values[pid]["tradeable"] = False
                        if not player_values[pid].get("trade_block_reason"):
                            player_values[pid]["trade_block_reason"] = (
                                "Affiliate-only contract — NHL SPC required to trade"
                            )

            # Unsigned drafted prospects are held as rights, not roster spots,
            # but the rights themselves are tradeable and must be listed.
            for p in getattr(team, "prospect_pool", None) or []:
                if getattr(p, "retired", False):
                    continue
                pid = str(getattr(p, "id", "") or "")
                if not pid or pid in player_values:
                    continue
                row = _serialize_player_trade_block(
                    p,
                    source_team=team,
                    acquiring_team=acq_team or team,
                    league=league,
                    session=session,
                    trade_context=ctx,
                )
                row["roster_level"] = "prospect"
                row["is_draft_rights"] = True
                player_values[pid] = row
        except Exception:
            player_values = {}

        teams_out[str(tid)] = {
            "picks": picks,
            "players": player_values,
            "cap": {
                "usable_cap_space": snap.get("usableCapSpace")
                if snap.get("usableCapSpace") is not None
                else cap_full.get("usable_cap_space_m"),
                "total_cap_hit": snap.get("totalCapHit")
                if snap.get("totalCapHit") is not None
                else cap_full.get("total_cap_hit_m"),
                "upper_limit": snap.get("upperLimit")
                if snap.get("upperLimit") is not None
                else cap_full.get("upper_limit_m"),
                "effective_cap_limit": snap.get("effectiveCapLimit")
                if snap.get("effectiveCapLimit") is not None
                else cap_full.get("upper_limit_m"),
                "ltir_pool": snap.get("ltirPool") if snap.get("ltirPool") is not None else cap_full.get("ltir_pool_m"),
                "is_using_ltir": snap.get("isUsingLTIR"),
                "projected_deadline_space": snap.get("projectedDeadlineSpace"),
                "retained_salary": snap.get("retainedSalary")
                if snap.get("retainedSalary") is not None
                else cap_full.get("retained_salary_m"),
                "retained_slots_used": snap.get("retainedSlotsUsed"),
                "retained_slots_max": snap.get("retainedSlotsMax"),
                "projected_cap_space": snap.get("usableCapSpace")
                if snap.get("usableCapSpace") is not None
                else cap_full.get("usable_cap_space_m"),
                "buried_cap_hit": snap.get("buriedCapHit") if snap.get("buriedCapHit") is not None else cap_full.get("buried_cap_hit_m"),
                "buyout_cap_hit": snap.get("buyoutCapHit") if snap.get("buyoutCapHit") is not None else cap_full.get("buyout_cap_hit_m"),
                "incoming_cap_supported": True,
            },
            "team_direction": direction,
            "needs": needs,
            "needs_summary": needs_summary,
            "depth": _organizational_depth_summary(team),
            "roster_capacity": _team_roster_capacity_payload(team),
            "contract_slots": _team_contract_slots_payload(team, league),
            **outlook,
        }

    return {
        "teams": teams_out,
        "formula_version": int(TRADE_ASSETS_CACHE_VERSION),
        "season_year": int(ctx.get("season_year") or 0),
        "draft_year": int(ctx.get("draft_year") or 0),
        "tradeable_draft_year": int(ctx.get("tradeable_draft_year") or ctx.get("draft_year") or 0),
        "draft_completed": bool(ctx.get("draft_completed")),
    }


def build_trade_market_payload(session: Any) -> Dict[str, Any]:
    ctx = _ensure_trade_infrastructure(session)
    league = ctx["league"]
    if league is None:
        return {"recent_trades": [], "market_temperature": "Cool", "teams": {}}

    deadline_phase = float(ctx.get("deadline_phase", 0.0) or 0.0)
    if deadline_phase > 0.55:
        temperature = "Hot"
    elif deadline_phase > 0.25:
        temperature = "Warm"
    else:
        temperature = "Cool"

    hist = get_trade_history(league, limit=12)
    team_labels: Dict[str, Any] = {}
    for tid, team in (ctx["team_by_id"] or {}).items():
        direction = str(getattr(team, "gm_window", getattr(team, "window", "unknown")) or "unknown")
        needs = getattr(team, "needs", None) or TeamNeeds().evaluate(team)
        team_labels[str(tid)] = {
            "direction": direction,
            "needs_short": _summarize_team_needs(needs, direction).get("needs_short") or [],
        }

    runtime = getattr(league, "cpu_market_runtime", None) or {}
    adaptive = runtime.get("adaptive_knobs") if isinstance(runtime, dict) else None
    telemetry = runtime.get("telemetry") if isinstance(runtime, dict) else None

    return {
        "market_temperature": temperature,
        "deadline_phase": round(deadline_phase, 3),
        "recent_trades": hist,
        "team_labels": team_labels,
        "adaptive_market": adaptive if isinstance(adaptive, dict) else None,
        "market_telemetry": telemetry if isinstance(telemetry, dict) else None,
    }


def get_franchise_trade_history(
    session: Any,
    *,
    team_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    ctx = _trade_context(session)
    league = ctx["league"]
    if league is None:
        return {"history": []}
    hist = get_trade_history(league, team_id=team_id, limit=limit)
    return {"history": hist}
