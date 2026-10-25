"""Background scouting notes for the draft board.

Exact overall stays off the public card. The user gets a range, a projection,
and a few tool tags. Confidence rises as the draft gets closer.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _ovr99(player: Any) -> float:
    raw = getattr(player, "ovr", None)
    if raw is None:
        raw = getattr(player, "overall", None)
    try:
        value = float(raw() if callable(raw) else raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 1.5:
        value *= 99.0
    return value


def _pot99(player: Any) -> float:
    ratings = getattr(player, "ratings", None) or {}
    raw = None
    if isinstance(ratings, dict):
        raw = ratings.get("dev_potential")
    if raw is None:
        raw = getattr(player, "potential", None)
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        value = 0.0
    if 0 < value <= 1.5:
        value *= 99.0
    return value


def scout_report(player: Any, *, drafts_away: int = 1) -> Dict[str, Any]:
    """Estimated picture of a prospect. drafts_away 0 is draft week."""
    ovr = _ovr99(player)
    pot = _pot99(player) or ovr
    pos = str(
        getattr(getattr(getattr(player, "identity", None), "position", None), "value", None)
        or getattr(getattr(player, "identity", None), "position", "")
        or getattr(player, "position", "")
        or ""
    ).upper()
    spread = 6 if drafts_away >= 2 else (4 if drafts_away == 1 else 2)
    lo = max(40, int(round(ovr)) - spread)
    hi = min(99, int(round(ovr)) + max(1, spread - 2))
    tags: List[str] = []
    ratings = getattr(player, "ratings", None) or {}
    shot = float(ratings.get("shot") or ratings.get("shooting") or 0) if isinstance(ratings, dict) else 0
    skate = float(ratings.get("skating") or ratings.get("speed") or 0) if isinstance(ratings, dict) else 0
    iq = float(ratings.get("hockey_iq") or ratings.get("passing") or 0) if isinstance(ratings, dict) else 0
    if shot >= 85 or (ovr >= 70 and shot == 0 and pot >= 85 and "D" not in pos and pos != "G"):
        tags.append("Elite Shot")
    if iq >= 84:
        tags.append("High Hockey IQ")
    if 0 < skate < 72:
        tags.append("Skating Needs Work")
    elif skate >= 86:
        tags.append("Excellent Skater")
    if pot - ovr >= 12:
        tags.append("Boom-or-Bust Prospect")
    if "D" in pos and ovr >= 68:
        tags.append("NHL-Ready Defender" if ovr >= 74 else "Defensive Concerns" if pot < 78 else "Puck-Moving Defence")
    if pos == "G":
        projection = "Starter projection" if pot >= 84 else "Backup projection"
    elif "D" in pos:
        projection = "Top-4 Defenseman" if pot >= 84 else "Bottom-pair projection"
    elif pot >= 88:
        projection = "Top-6 Forward Projection"
    elif pot >= 80:
        projection = "Middle-six projection"
    else:
        projection = "Bottom-6 projection"
    if pot >= 92:
        tags.append("Elite Potential")
    confidence = "High" if drafts_away <= 0 else ("Medium" if drafts_away == 1 else "Low")
    return {
        "ovr_range": [lo, hi],
        "projection": projection,
        "tags": tags[:4] or ["Balanced tools"],
        "confidence": confidence,
        "summary": f"{projection}. {' · '.join(tags[:3]) or 'Balanced tools'}. Confidence {confidence.lower()}.",
    }


def _drafts_away(row: Dict[str, Any]) -> int:
    """June is draft month. Fall is a full year out."""
    month = row.get("calendar_month")
    try:
        month = int(month)
    except (TypeError, ValueError):
        month = 0
    if month >= 9:
        return 2
    if month >= 4:
        return 0
    if month >= 1:
        return 1
    return 1


def attach_auto_scout(player: Any, row: Dict[str, Any]) -> None:
    report = scout_report(player, drafts_away=_drafts_away(row))
    row["auto_scout"] = report
    row["scout_tags"] = report["tags"]
    row["scout_projection"] = report["projection"]
    row["scout_confidence"] = report["confidence"]
    row["scout_summary"] = report["summary"]
    row["ovr_range"] = {"low": report["ovr_range"][0], "high": report["ovr_range"][1]}
    row["public_ovr_low"] = report["ovr_range"][0]
    row["public_ovr_high"] = report["ovr_range"][1]
    # The board shows the range. The exact overall stays off the card.
    row["ovr_revealed"] = False
