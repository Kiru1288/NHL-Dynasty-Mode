"""
Franchise session trade service — bridges API to SimEngine trade module.
"""

from __future__ import annotations
import os

import uuid
from typing import Any, Dict, List, Optional

from app.sim_engine.franchise.paths import ensure_simengine_path

ensure_simengine_path()

from app.sim_engine.trades.trade_evaluator import evaluate_trade_package  # noqa: E402
from app.sim_engine.trades.trade_executor import execute_validated_trade  # noqa: E402
from app.sim_engine.trades.trade_history import get_trade_history  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import (  # noqa: E402
    audit_pick_registry_integrity,
    ensure_draft_pick_registry,
    serialize_team_picks,
)
from app.sim_engine.trades.trade_value import evaluate_pick_asset_value, pick_value_hint  # noqa: E402
from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot  # noqa: E402
from app.sim_engine.economy.team_needs import TeamNeeds  # noqa: E402


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
        "nhl_count": len(roster),
        "ahl_count": len(ahl),
        "top_six_forwards": len(fwds[:6]),
        "top_four_defense": len(defs[:4]),
        "goalies": len(gs),
        "prospect_count": len(prospects),
        "pipeline_strength": round(
            sum(_ovr(p) for p in prospects[:3]) / max(1, min(3, len(prospects))),
            1,
        ) if prospects else 0.0,
    }


def _trade_context(session: Any) -> Dict[str, Any]:
    sim = session.sim
    league = getattr(sim, "league", None)
    cal = getattr(session, "nhl_calendar", None) or []
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 192) or 192))
    md = max(40, int(max(120, max_d) * 0.56))
    deadline_phase = max(0.0, min(1.0, (float(cursor) - float(md)) / max(20.0, float(max_d) * 0.2)))
    calendar_iso = ""
    if 0 <= cursor < len(cal):
        calendar_iso = str(cal[cursor].get("iso") or "")

    return {
        "sim": sim,
        "league": league,
        "team_by_id": dict(session.team_by_id or {}),
        "user_team_id": str(session.user_team_id),
        "season_year": int(getattr(session, "season_calendar_year", 2025) or 2025),
        "calendar_cursor": cursor,
        "calendar_iso": calendar_iso,
        "regular_season_last_index": max_d,
        "deadline_phase": deadline_phase,
    }


def _ensure_trade_infrastructure(session: Any) -> None:
    ctx = _trade_context(session)
    league = ctx["league"]
    if league is None:
        return
    ensure_draft_pick_registry(league, start_year=ctx["season_year"], years_ahead=4)


def evaluate_franchise_trade(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    record_rumor_fallout: bool = False,
) -> Dict[str, Any]:
    _ensure_trade_infrastructure(session)
    ctx = _trade_context(session)
    try:
        from app.sim_engine.franchise.engine import ensure_player_financials

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
        pass
    result = evaluate_trade_package(
        dict(assets_by_team or {}),
        league=ctx["league"],
        team_by_id=ctx["team_by_id"],
        context=ctx,
        user_team_id=ctx["user_team_id"],
    )
    public = {k: v for k, v in result.items() if not str(k).startswith("_")}
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
    except Exception:
        pass
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
            pass
    return public


def execute_franchise_trade(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    record_notifications_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate and execute trade; raises ValueError on failure."""
    _ensure_trade_infrastructure(session)
    ctx = _trade_context(session)

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
            pass

    exec_result = execute_validated_trade(
        evaluation,
        league=ctx["league"],
        team_by_id=ctx["team_by_id"],
        context=ctx,
        user_team_id=ctx["user_team_id"],
    )

    if record_notifications_fn is not None:
        record_notifications_fn(session, exec_result, ctx)

    return exec_result


def build_trade_assets_payload(session: Any) -> Dict[str, Any]:
    _ensure_trade_infrastructure(session)
    ctx = _trade_context(session)
    league = ctx["league"]
    if os.environ.get("NHL_FRANCHISE_DEBUG", "0") == "1":
        audit = audit_pick_registry_integrity(
            league,
            start_year=int(ctx["season_year"]),
            years_ahead=4,
            rounds=7,
        )
        if not audit.get("ok"):
            first = (audit.get("errors") or ["unknown registry mismatch"])[0]
            print(f"[trade debug] pick registry drift before trade assets build: {first}")
    sim = ctx["sim"]
    needs_model = TeamNeeds()
    teams_out: Dict[str, Any] = {}

    for tid, team in (ctx["team_by_id"] or {}).items():
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
            value_hint_fn=lambda row: pick_value_hint(row, league, team, context=ctx),
        )
        for item in picks:
            try:
                detail = evaluate_pick_asset_value(item, team, team, league, context=ctx)
                item["value_debug"] = detail.get("value_debug") or {}
            except Exception:
                item["value_debug"] = {}
        direction = str(getattr(team, "gm_window", getattr(team, "window", "unknown")) or "unknown")
        needs_summary = _summarize_team_needs(needs, direction)

        player_values: Dict[str, Any] = {}
        try:
            from app.sim_engine.franchise.engine import _serialize_player_trade_block

            user_tid = str(ctx.get("user_team_id") or "")
            acq_team = ctx["team_by_id"].get(user_tid) if user_tid else team
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                pid = str(getattr(p, "id", "") or "")
                if not pid:
                    continue
                player_values[pid] = _serialize_player_trade_block(
                    p,
                    source_team=team,
                    acquiring_team=acq_team or team,
                    league=league,
                    session=session,
                )
        except Exception:
            player_values = {}

        teams_out[str(tid)] = {
            "picks": picks,
            "players": player_values,
            "cap": {
                "usable_cap_space": snap.get("usableCapSpace"),
                "total_cap_hit": snap.get("totalCapHit"),
                "upper_limit": snap.get("upperLimit"),
                "retained_salary": snap.get("retainedSalary"),
                "retained_slots_used": snap.get("retainedSlotsUsed"),
                "retained_slots_max": snap.get("retainedSlotsMax"),
                "projected_cap_space": snap.get("usableCapSpace"),
                "incoming_cap_supported": True,
            },
            "team_direction": direction,
            "needs": needs,
            "needs_summary": needs_summary,
            "depth": _organizational_depth_summary(team),
        }

    return {"teams": teams_out}


def build_trade_market_payload(session: Any) -> Dict[str, Any]:
    _ensure_trade_infrastructure(session)
    ctx = _trade_context(session)
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

    return {
        "market_temperature": temperature,
        "deadline_phase": round(deadline_phase, 3),
        "recent_trades": hist,
        "team_labels": team_labels,
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
