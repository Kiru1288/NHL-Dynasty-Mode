"""Presentation data for the Draft Lottery Night screens.

The lottery itself runs in franchise_offseason._run_draft_lottery. This module adds the
context the UI needs to tell the story: each lottery club's record, pre-lottery slot,
team state (tanking / rebuilding / bubble), GM and fan mood, plus the top prospects of
the draft class at lottery time. The result is snapshotted on the payload once
(``presentation_v1``) so records do not drift after the season rolls over.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

PRESENTATION_VERSION = 1

_PROSPECT_KEYS = (
    "key",
    "name",
    "position",
    "age",
    "nationality",
    "country",
    "team",
    "team_name",
    "league",
    "league_display",
    "rank",
    "potential_score",
    "ovr_range",
    "gp",
    "goals",
    "assists",
    "points",
    "ppg",
    "wins",
    "save_pct",
    "gaa",
    "stock_label",
    "stock_trend",
    "stock_delta",
    "archetype",
    "player_type",
    "handedness",
    "height",
    "weight",
    "headshot_id",
    "face_variant",
    "scouting_confidence",
    "boom_bust",
)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _team_state(tank: Dict[str, Any], team: Any, points_back: int) -> Dict[str, str]:
    mode = str((tank or {}).get("tank_mode") or "").lower()
    status = str(getattr(team, "team_status", "") or getattr(team, "gm_window", "") or "").lower()
    if mode == "hard_tank":
        return {"key": "tanking", "label": "Tanking"}
    if 0 <= points_back <= 6:
        return {"key": "bubble", "label": "Bubble miss"}
    if mode == "seller":
        return {"key": "selling", "label": "Selling"}
    if "tank" in status:
        return {"key": "tanking", "label": "Tanking"}
    if "rebuild" in status or "retool" in status:
        return {"key": "rebuilding", "label": "Rebuilding"}
    if "contend" in status or "playoff" in status:
        return {"key": "slumping", "label": "Down year"}
    return {"key": "rebuilding", "label": "Rebuilding"}


def _mood_word(value: float) -> str:
    if value >= 68:
        return "Upbeat"
    if value >= 55:
        return "Patient"
    if value >= 42:
        return "Restless"
    return "Angry"


def _top_prospects(session: Any, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        from services.franchise_sim import get_cached_draft_class_rankings

        board = get_cached_draft_class_rankings(session, session.sim) or {}
    except Exception:
        _log.debug("lottery presentation: draft board unavailable", exc_info=True)
        return []
    entries = [e for e in (board.get("entries") or []) if isinstance(e, dict)]
    entries.sort(key=lambda e: int(_num(e.get("rank"), 999)))
    out: List[Dict[str, Any]] = []
    for e in entries[:limit]:
        row = {k: e.get(k) for k in _PROSPECT_KEYS if e.get(k) is not None}
        row["id"] = str(e.get("key") or e.get("player_id") or e.get("name") or "")
        out.append(row)
    return out


def build_lottery_presentation(session: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return {field, top_prospects, lottery_meta} for the given lottery payload."""
    from services.franchise_sim import _build_standings_rows, _display_team

    picks = list(payload.get("picks") or payload.get("final_order") or [])
    pre = list(payload.get("pre_lottery_order") or [])
    try:
        rows = _build_standings_rows(session) or []
    except Exception:
        rows = []
    row_by_tid = {str(r.get("team_id") or ""): r for r in rows}

    playoff_ids: set = set()
    table = getattr(session, "standings", None)
    if table is not None and hasattr(table, "playoff_status_map"):
        try:
            playoff_ids = {
                str(tid)
                for tid, info in (table.playoff_status_map() or {}).items()
                if isinstance(info, dict) and info.get("is_playoff_team")
            }
        except Exception:
            playoff_ids = set()
    cutoff_pts = None
    if playoff_ids:
        po_pts = [int(_num(row_by_tid[t].get("pts"))) for t in playoff_ids if t in row_by_tid]
        if po_pts:
            cutoff_pts = min(po_pts)

    tank_map = getattr(session, "transcendent_tank_pressure", None) or {}
    fan_profiles = getattr(session, "fan_profiles", None) or {}
    uid = str(getattr(session, "user_team_id", "") or "")

    pick_by_lottery_tid: Dict[str, Dict[str, Any]] = {}
    for p in picks:
        tid = str(p.get("lottery_team_id") or p.get("original_owner_team_id") or p.get("team_id") or "")
        if tid:
            pick_by_lottery_tid[tid] = p

    if not pre:
        pre = [
            {"lottery_rank": int(_num(p.get("original_rank"), i + 1)), "team_id": str(p.get("lottery_team_id") or p.get("team_id") or "")}
            for i, p in enumerate(picks)
        ]
        pre.sort(key=lambda r: r["lottery_rank"])

    field: List[Dict[str, Any]] = []
    for i, entry in enumerate(pre, start=1):
        tid = str(entry.get("team_id") or "")
        rank = int(_num(entry.get("lottery_rank"), i))
        tm = session.team_by_id.get(tid) if getattr(session, "team_by_id", None) else None
        srow = row_by_tid.get(tid) or {}
        snap_rec = entry.get("record") if isinstance(entry.get("record"), dict) else None
        if snap_rec:
            record = dict(snap_rec)
        elif int(_num(srow.get("gp"))) > 0:
            record = {
                "gp": int(_num(srow.get("gp"))),
                "w": int(_num(srow.get("w"))),
                "l": int(_num(srow.get("l"))),
                "otl": int(_num(srow.get("otl"))),
                "pts": int(_num(srow.get("pts"))),
            }
            if srow.get("gf") is not None:
                record["gf"] = int(_num(srow.get("gf")))
            if srow.get("ga") is not None:
                record["ga"] = int(_num(srow.get("ga")))
        else:
            record = {"pts": int(_num(entry.get("points"))), "w": int(_num(entry.get("wins")))}
        pts = int(_num(record.get("pts"), _num(entry.get("points"))))
        points_back = int(cutoff_pts - pts) if cutoff_pts is not None else -1
        tank = tank_map.get(tid) if isinstance(tank_map, dict) else None
        tank = tank if isinstance(tank, dict) else {}
        state = _team_state(tank, tm, points_back)
        fan = fan_profiles.get(tid) if isinstance(fan_profiles, dict) else None
        pick = pick_by_lottery_tid.get(tid) or {}
        row: Dict[str, Any] = {
            "team_id": tid,
            "team_name": entry.get("team_name") or (_display_team(tm) if tm is not None else tid),
            "abbrev": str(srow.get("abbrev") or getattr(tm, "abbreviation", "") or "").upper(),
            "conference": str(srow.get("conference") or getattr(tm, "conference", "") or ""),
            "division": str(srow.get("division") or getattr(tm, "division", "") or ""),
            "lottery_rank": rank,
            "record": record,
            "points_back": points_back,
            "team_state": state["key"],
            "team_state_label": state["label"],
            "tank_pressure": int(_num(tank.get("tank_pressure"))),
            "owns_own_first": tank.get("owns_own_first"),
            "gm_name": str(getattr(tm, "gm_name", "") or "") if tm is not None else "",
            "odds": pick.get("odds"),
            "base_odds": pick.get("base_odds"),
            "final_pick": pick.get("pick"),
            "movement": pick.get("movement"),
            "pick_owner_team_id": pick.get("team_id"),
            "pick_owner_team_name": pick.get("team_name"),
            "is_traded": bool(pick.get("is_traded")),
            "is_user": bool(uid and (tid == uid or str(pick.get("team_id") or "") == uid)),
        }
        if isinstance(fan, dict) and fan.get("fan_confidence") is not None:
            conf = round(_num(fan.get("fan_confidence"), 55.0))
            row["fan_confidence"] = conf
            row["fan_mood"] = _mood_word(conf)
            if fan.get("fan_trust_in_gm") is not None:
                row["fan_trust_in_gm"] = round(_num(fan.get("fan_trust_in_gm"), 55.0))
        field.append(row)

    meta = {
        "version": PRESENTATION_VERSION,
        "user_team_id": uid,
        "draft_year": int(_num(getattr(session, "season_calendar_year", 0))) + 1,
        "playoff_cutoff_pts": cutoff_pts,
        "max_jump": 10,
        "draws": 2,
    }
    return {"field": field, "top_prospects": _top_prospects(session), "lottery_meta": meta}


def ensure_lottery_presentation(session: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach presentation data once (snapshot) and store it back on the session."""
    if not isinstance(payload, dict) or not (payload.get("picks") or payload.get("final_order")):
        return payload
    if payload.get("presentation_v1") and payload.get("field"):
        return payload
    try:
        extra = build_lottery_presentation(session, payload)
    except Exception:
        _log.debug("lottery presentation build failed", exc_info=True)
        return payload
    out = dict(payload)
    out.update(extra)
    out["presentation_v1"] = True
    try:
        session.draft_lottery_payload = out
    except Exception:
        pass
    return out
