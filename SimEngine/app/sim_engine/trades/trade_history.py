"""
Structured trade history on league state.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


def _safe_str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def ensure_trade_history(league: Any) -> List[Dict[str, Any]]:
    hist = getattr(league, "trade_history", None)
    if not isinstance(hist, list):
        hist = []
        setattr(league, "trade_history", hist)
    return hist


def append_trade_record(league: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    hist = ensure_trade_history(league)
    rec = dict(record)
    if not rec.get("trade_id"):
        rec["trade_id"] = f"trade_{uuid.uuid4().hex[:12]}"
    hist.append(rec)
    setattr(league, "trade_history", hist)
    return rec


def get_trade_history(
    league: Any,
    *,
    team_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    hist = list(ensure_trade_history(league))
    if team_id:
        tid = _safe_str(team_id)
        filtered: List[Dict[str, Any]] = []
        for row in hist:
            teams = row.get("participating_teams") or []
            if tid in [str(t) for t in teams]:
                filtered.append(row)
        hist = filtered
    hist = hist[-max(1, int(limit)) :]
    return [serialize_trade_record(r) for r in hist]


def serialize_trade_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trade_id": record.get("trade_id"),
        "calendar_day": record.get("calendar_day"),
        "calendar_iso": record.get("calendar_iso"),
        "season_year": record.get("season_year"),
        "participating_teams": list(record.get("participating_teams") or []),
        "assets_by_team": dict(record.get("assets_by_team") or {}),
        "moved_players": list(record.get("moved_players") or []),
        "moved_picks": list(record.get("moved_picks") or []),
        "retained_salary": list(record.get("retained_salary") or []),
        "cap_impact": dict(record.get("cap_impact") or {}),
        "value_scores": dict(record.get("value_scores") or {}),
        "fairness_gap": record.get("fairness_gap"),
        "accepted": bool(record.get("accepted")),
        "rejection_reasons": list(record.get("rejection_reasons") or []),
        "headline": record.get("headline"),
        "user_involved": bool(record.get("user_involved")),
    }
