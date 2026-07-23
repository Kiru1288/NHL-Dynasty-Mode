"""Franchise scouting API — draft-class prospects, assignments, and GM-driven coverage."""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.sim_engine.franchise.serialization import build_draft_class_rankings
from app.sim_engine.franchise.session import FranchiseSession
from app.sim_engine.franchise.state import invalidate_session_payload_caches

DRAFT_OVR_REVEAL_THRESHOLD = 72.0

DEFAULT_SCOUTING_BUDGET = 2_500_000


def _draft_franchise_date_context(session: FranchiseSession) -> Dict[str, Any]:
    month: Optional[int] = None
    phase = str(getattr(session, "phase", "regular") or "regular")
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    cal = list(getattr(session, "nhl_calendar", None) or [])
    if cal and 0 <= cur < len(cal):
        cd = cal[cur]
        if isinstance(cd, dict):
            raw_month = cd.get("month")
            if raw_month is not None:
                month = int(raw_month)
            elif cd.get("iso"):
                try:
                    month = date.fromisoformat(str(cd["iso"])).month
                except Exception:
                    month = None
    if month is None:
        offseason_stage = str(getattr(session, "offseason_stage", "") or "")
        stage_month = {
            "draft_lottery": 5,
            "draft": 6,
            "re_sign": 7,
            "free_agency": 7,
            "development_report": 7,
        }
        month = stage_month.get(offseason_stage)
    return {"month": month, "phase": phase, "calendar_cursor": cur}


def _draft_period_label(month: Optional[int]) -> str:
    if month is None:
        return "Scouting season"
    if month in (8, 9, 10):
        return "Early season scouting"
    if month in (11, 12, 1):
        return "Mid-season scouting"
    if month in (2, 3):
        return "Late-season scouting"
    if month == 4:
        return "Combine week"
    if month in (5, 6):
        return "Draft week"
    if month == 7:
        return "Offseason"
    return "Scouting season"


def _draft_scout_completion(rank: int, month: Optional[int], key: str) -> float:
    base = 22.0 + max(0, 40 - min(rank, 120)) * 0.35
    month_bonus = {
        8: 0, 9: 4, 10: 8, 11: 12, 12: 16, 1: 20,
        2: 28, 3: 36, 4: 48, 5: 58, 6: 68, 7: 12,
    }.get(int(month) if month is not None else 0, 0)
    jitter = (abs(hash(str(key))) % 11) - 5
    return float(max(8.0, min(88.0, base + month_bonus + jitter)))


def _draft_ovr_range(true_ovr: float, scouted: float) -> Dict[str, float]:
    spread = max(4.0, 18.0 - (float(scouted) * 0.16))
    low = max(40.0, float(true_ovr) - spread)
    high = min(99.0, float(true_ovr) + spread * 0.65)
    if scouted >= DRAFT_OVR_REVEAL_THRESHOLD:
        low = high = round(float(true_ovr), 1)
    return {"low": round(low, 1), "high": round(high, 1)}

ACTION_SCOUTED_GAIN: Dict[str, float] = {
    "region_sweep": 6.0,
    "player_focus": 10.0,
    "live_viewing": 14.0,
    "video_review": 5.0,
    "character_check": 8.0,
    "analytics_deep_dive": 9.0,
    "interview": 7.0,
    "dinner": 11.0,
    "combine": 12.0,
    "private_workout": 15.0,
    "medical_review": 10.0,
}

INTENSITY_MULT = {
    "light": 0.72,
    "normal": 1.0,
    "heavy": 1.32,
    "all_in": 1.65,
}

POST_ROUTE_ACTION = {
    "assign": "assign",
    "reassign": "reassign",
    "cancel": "cancel",
    "interview": "interview",
    "dinner": "dinner",
    "combine": "combine",
    "private-workout": "private_workout",
    "private_workout": "private_workout",
    "request-medical": "medical_review",
    "medical_review": "medical_review",
    "focus": "player_focus",
}


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    return s.strip("-") or "unknown"


def _ensure_scouting_state(session: FranchiseSession) -> Dict[str, Any]:
    raw = getattr(session, "scouting_state", None)
    if not isinstance(raw, dict):
        raw = {}
        session.scouting_state = raw
    raw.setdefault("budget", DEFAULT_SCOUTING_BUDGET)
    raw.setdefault("used_budget", 0.0)
    raw.setdefault("prospects", {})
    raw.setdefault("assignments", [])
    raw.setdefault("watchlist", [])
    return raw


def _scouting_phase(session: FranchiseSession) -> str:
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    if month is None or month in (8, 9, 10):
        return "early"
    if month == 11:
        return "mid"
    if month in (12, 1):
        return "mid"
    if month in (2, 3):
        return "late"
    if month == 4:
        return "combine"
    if month in (5, 6):
        return "draft_week"
    if month == 7:
        return "offseason"
    return "early"


def _season_label(session: FranchiseSession) -> str:
    y = int(getattr(session, "season_calendar_year", 2025) or 2025)
    return f"{y}–{y + 1}"


def _intensity_mult(intensity: str) -> float:
    return float(INTENSITY_MULT.get(str(intensity or "normal").lower(), 1.0))


def _scout_pool(session: FranchiseSession) -> List[Dict[str, Any]]:
    """Procedural scout staff tied to franchise session (no fixed player identities)."""
    seed = abs(hash(str(session.session_id))) % 10_000
    regions = [
        ("North America", "NA"),
        ("Europe", "EU"),
        ("Scandinavia", "SCAND"),
        ("Russia / CIS", "CIS"),
        ("Goaltending", "G"),
    ]
    scouts: List[Dict[str, Any]] = []
    for i, (region, code) in enumerate(regions):
        sid = f"scout-{code.lower()}"
        quality = 68 + ((seed + i * 17) % 28)
        scouts.append(
            {
                "id": sid,
                "scout_id": sid,
                "name": f"Regional Scout {code}",
                "role": "Pro Scout" if i else "Director of Amateur Scouting",
                "region": region,
                "country": "",
                "quality": quality,
                "rating": quality,
                "workload": 0,
                "specialty": region,
            }
        )
    return scouts


def _prospect_overlay(state: Dict[str, Any], prospect_id: str) -> Dict[str, Any]:
    prospects = state.get("prospects") if isinstance(state.get("prospects"), dict) else {}
    row = prospects.get(prospect_id)
    return dict(row) if isinstance(row, dict) else {}


def _set_prospect_overlay(state: Dict[str, Any], prospect_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    prospects = state.setdefault("prospects", {})
    cur = dict(prospects.get(prospect_id) or {})
    cur.update(patch)
    prospects[prospect_id] = cur
    return cur


def _scouted_pct(entry: Mapping[str, Any], overlay: Mapping[str, Any], month: Optional[int]) -> float:
    if overlay.get("scouted_percentage") is not None:
        return float(max(0.0, min(100.0, overlay["scouted_percentage"])))
    if entry.get("scouted_percentage") is not None:
        return float(max(0.0, min(100.0, entry["scouted_percentage"])))
    rank = int(entry.get("rank") or 999)
    key = str(entry.get("key") or entry.get("id") or "")
    base = float(_draft_scout_completion(rank, month, key))
    bonus = float(overlay.get("scouted_bonus") or 0.0)
    return float(max(0.0, min(100.0, base + bonus)))


def _normalize_prospect(
    entry: Mapping[str, Any],
    overlay: Mapping[str, Any],
    month: Optional[int],
) -> Dict[str, Any]:
    pid = str(entry.get("key") or entry.get("id") or "")
    country = str(entry.get("country") or "Unknown")
    region = str(entry.get("region") or "")
    if not region:
        cl = country.lower()
        if cl in ("canada", "united states", "usa", "us"):
            region = "North America"
        elif cl in ("sweden", "finland", "norway", "denmark"):
            region = "Scandinavia"
        elif cl in ("russia", "belarus", "kazakhstan"):
            region = "Russia / CIS"
        else:
            region = "Europe" if country not in ("Unknown", "") else "International"

    scouted = _scouted_pct(entry, overlay, month)
    potential = float(entry.get("potential_score") or entry.get("true_ovr") or 70)
    true_ovr = float(entry.get("true_ovr") or potential or 70)
    ovr_revealed = scouted >= float(DRAFT_OVR_REVEAL_THRESHOLD)
    raw_range = entry.get("ovr_range")
    if isinstance(raw_range, dict) and raw_range.get("low") is not None and raw_range.get("high") is not None:
        ovr_range = {
            "low": round(float(raw_range["low"]), 1),
            "high": round(float(raw_range["high"]), 1),
        }
    else:
        ovr_range = _draft_ovr_range(true_ovr, scouted)
    floor = float(ovr_range["low"])

    traits = list(overlay.get("traits") or [])
    if not traits and entry.get("player_type"):
        traits = [str(entry.get("player_type"))]
    red_flags = list(overlay.get("red_flags") or [])
    notes = list(overlay.get("notes") or overlay.get("reports") or [])

    stock = str(overlay.get("draft_stock") or entry.get("stock_change") or "Stable")
    if isinstance(stock, (int, float)):
        stock = "Rising" if float(stock) > 1 else "Falling" if float(stock) < -1 else "Stable"

    skills = {
        k: entry[k]
        for k in (
            "skating",
            "shooting",
            "passing",
            "defense",
            "physical",
            "hockey_iq",
            "compete",
            "poise",
            "consistency",
            "coachability",
        )
        if entry.get(k) is not None
    }

    out: Dict[str, Any] = {
        "id": pid,
        "key": pid,
        "name": str(entry.get("name") or "Unknown"),
        "position": str(entry.get("position") or "F"),
        "country": country,
        "region": region,
        "league": str(
            entry.get("league_display")
            or entry.get("league_code")
            or entry.get("league_name")
            or ""
        ),
        "league_code": str(entry.get("league_code") or ""),
        "league_name": str(entry.get("league_name") or ""),
        "league_display": str(entry.get("league_display") or ""),
        "team": str(entry.get("team_name") or ""),
        "team_name": str(entry.get("team_name") or ""),
        "team_id": str(entry.get("team_id") or ""),
        "age": int(entry.get("age") or 0),
        "rank": int(entry.get("rank") or 0),
        "scouted_percentage": round(scouted, 1),
        "scouted": round(scouted, 1),
        "upside": round(min(99.0, potential), 1),
        "floor": round(floor, 1),
        "ovr_range": ovr_range,
        "ovr_confidence": str(entry.get("ovr_confidence") or ("exact" if ovr_revealed else "range")),
        "ovr_revealed": bool(ovr_revealed),
        "risk": str(entry.get("risk") or "Medium"),
        "projection": str(entry.get("projection") or ""),
        "traits": traits,
        "red_flags": red_flags,
        "notes": notes,
        "skills": skills,
        "watchlist": bool(
            overlay.get("watchlist")
            or entry.get("watchlist")
            or pid in (overlay.get("watchlist_ids") or [])
        ),
        "target": bool(overlay.get("target") or entry.get("target")),
        "do_not_draft": bool(overlay.get("do_not_draft") or entry.get("do_not_draft")),
        "assigned_scout": str(overlay.get("assigned_scout") or entry.get("assigned_scout") or ""),
        "draft_stock": stock,
        "combine_status": str(overlay.get("combine_status") or "Not started"),
        "interview_status": str(overlay.get("interview_status") or "Not interviewed"),
        "dinner_status": str(overlay.get("dinner_status") or "Not scheduled"),
        "handedness": str(entry.get("handedness") or ""),
        "height": entry.get("height"),
        "weight": entry.get("weight"),
    }
    if ovr_revealed and entry.get("true_ovr") is not None:
        out["true_ovr"] = round(float(entry.get("true_ovr")), 1)

    for k in (
        "gp",
        "games_played",
        "goals",
        "assists",
        "points",
        "ppg",
        "points_per_game",
        "wins",
        "save_pct",
        "gaa",
        "shutouts",
        "pim",
        "production_context",
        "translation_risk",
        "scoring_environment",
        "league_difficulty",
        "production_adjusted_score",
        "league_scoring_profile",
    ):
        if entry.get(k) is not None:
            out[k] = entry[k]

    prod_adj = entry.get("production_adjusted_score")
    trans = str(entry.get("translation_risk") or "")
    if trans == "High" and prod_adj is not None:
        out["risk"] = "High"
        if "Translation risk" not in red_flags:
            red_flags.append("Translation risk")
    elif trans == "Medium" and str(out.get("risk") or "") == "Medium" and prod_adj is not None:
        try:
            if float(prod_adj) < float(entry.get("ppg") or entry.get("points_per_game") or 0) * 0.55:
                out["risk"] = "Medium-High"
        except (TypeError, ValueError):
            pass

    ctx = str(entry.get("production_context") or "")
    if ctx in ("Risky scorer", "Boom/Bust", "Overager scoring"):
        if ctx not in traits:
            traits.append(ctx)
    out["traits"] = traits
    out["red_flags"] = red_flags
    return out


def _draft_entries(session: FranchiseSession) -> List[Dict[str, Any]]:
    sim = getattr(session, "sim", None)
    board = build_draft_class_rankings(session, sim)
    return list(board.get("entries") or [])


def get_scouting_prospects(session: FranchiseSession) -> Dict[str, Any]:
    state = _ensure_scouting_state(session)
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    entries = _draft_entries(session)
    prospects = [
        _normalize_prospect(e, _prospect_overlay(state, str(e.get("key") or "")), month)
        for e in entries
    ]
    message = None if prospects else "No draft-class prospects in the active universe yet."
    return {"prospects": prospects, "total": len(prospects), "message": message}


def get_scouting_state(session: FranchiseSession) -> Dict[str, Any]:
    state = _ensure_scouting_state(session)
    ctx = _draft_franchise_date_context(session)
    scouts = _scout_pool(session)
    workload = sum(1 for a in state.get("assignments") or [] if str(a.get("status")) == "active")
    for sc in scouts:
        sc["workload"] = workload // max(1, len(scouts))

    return {
        "date": ctx.get("generated_at") or ctx.get("label") or "",
        "season": _season_label(session),
        "phase": _scouting_phase(session),
        "scouting_phase": _scouting_phase(session),
        "period_label": _draft_period_label(ctx.get("month")),
        "draft_year": ctx.get("draft_year"),
        "scouts": scouts,
        "staff": scouts,
        "budget": float(state.get("budget") or DEFAULT_SCOUTING_BUDGET),
        "used_budget": float(state.get("used_budget") or 0.0),
        "budget_remaining": float(state.get("budget") or DEFAULT_SCOUTING_BUDGET)
        - float(state.get("used_budget") or 0.0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _aggregate_world(prospects: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_country: Dict[str, Dict[str, Any]] = {}
    for p in prospects:
        country = str(p.get("country") or "Unknown")
        cid = _slug(country)
        if cid not in by_country:
            by_country[cid] = {
                "id": cid,
                "name": country,
                "region": str(p.get("region") or ""),
                "prospect_count": 0,
                "scouted_sum": 0.0,
            }
        by_country[cid]["prospect_count"] += 1
        by_country[cid]["scouted_sum"] += float(p.get("scouted_percentage") or 0)

    countries: List[Dict[str, Any]] = []
    for cid, row in sorted(by_country.items(), key=lambda x: -x[1]["prospect_count"]):
        cnt = max(1, row["prospect_count"])
        avg = row["scouted_sum"] / cnt
        seed = abs(hash(cid)) % 1000
        countries.append(
            {
                "id": cid,
                "name": row["name"],
                "region": row["region"],
                "prospect_count": row["prospect_count"],
                "scouted_average": round(avg, 1),
                "cost": 35_000 + row["prospect_count"] * 4_500 + (seed % 12) * 1_000,
                "effort": min(95, 40 + row["prospect_count"] * 3 + (seed % 20)),
                "difficulty": min(90, 25 + (seed % 40)),
                "safety_risk": 0,
                "political_risk": 0,
                "corruption_risk": min(35, seed % 30),
                "travel_notes": [],
            }
        )

    by_region: Dict[str, Dict[str, Any]] = {}
    for c in countries:
        reg = str(c.get("region") or "International")
        rid = _slug(reg)
        if rid not in by_region:
            by_region[rid] = {
                "id": rid,
                "name": reg,
                "prospect_count": 0,
                "country_count": 0,
                "scouted_sum": 0.0,
            }
        by_region[rid]["prospect_count"] += int(c["prospect_count"])
        by_region[rid]["country_count"] += 1
        by_region[rid]["scouted_sum"] += float(c["scouted_average"]) * int(c["prospect_count"])

    regions: List[Dict[str, Any]] = []
    for rid, row in sorted(by_region.items(), key=lambda x: -x[1]["prospect_count"]):
        cnt = max(1, row["prospect_count"])
        regions.append(
            {
                "id": rid,
                "name": row["name"],
                "prospect_count": row["prospect_count"],
                "country_count": row["country_count"],
                "scouted_average": round(row["scouted_sum"] / cnt, 1),
            }
        )

    return {"countries": countries, "regions": regions}


def get_scouting_world(session: FranchiseSession) -> Dict[str, Any]:
    prospects_payload = get_scouting_prospects(session)
    prospects = prospects_payload.get("prospects") or []
    world = _aggregate_world(prospects)
    ctx = _draft_franchise_date_context(session)
    world["generated_at"] = ctx.get("generated_at")
    world["message"] = prospects_payload.get("message")
    return world


def get_scouting_assignments(session: FranchiseSession) -> Dict[str, Any]:
    state = _ensure_scouting_state(session)
    assignments = list(state.get("assignments") or [])
    active = [a for a in assignments if str(a.get("status")) == "active"]
    completed = [a for a in assignments if str(a.get("status")) != "active"]
    return {
        "assignments": assignments,
        "active": active,
        "completed": completed,
    }


def _resolve_action(body: Mapping[str, Any], route_action: str) -> str:
    action = str(body.get("action") or route_action or "player_focus").lower().replace("-", "_")
    if action in ACTION_SCOUTED_GAIN:
        return action
    if route_action in ACTION_SCOUTED_GAIN:
        return route_action
    return "player_focus"


def _estimate_cost(body: Mapping[str, Any], action: str) -> float:
    explicit = body.get("estimated_cost")
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    base = 18_000.0 + ACTION_SCOUTED_GAIN.get(action, 8.0) * 2_200.0
    return base * _intensity_mult(str(body.get("intensity") or "normal"))


def _apply_to_targets(
    session: FranchiseSession,
    state: Dict[str, Any],
    body: Mapping[str, Any],
    action: str,
    *,
    cancel: bool = False,
) -> Tuple[List[str], float]:
    """Returns (affected_prospect_ids, cost)."""
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    entries = _draft_entries(session)
    entry_by_key = {str(e.get("key")): e for e in entries if e.get("key")}

    target_type = str(body.get("target_type") or "player").lower()
    target_id = str(
        body.get("target_id")
        or body.get("prospect_id")
        or body.get("player_id")
        or (body.get("context") or {}).get("prospect_id")
        or ""
    )
    country_id = str(
        body.get("country_id")
        or (body.get("context") or {}).get("country_id")
        or ""
    )

    affected: List[str] = []
    if target_type in ("player", "prospect") and target_id:
        affected = [target_id] if target_id in entry_by_key else []
    elif target_type == "country" and country_id:
        for e in entries:
            if _slug(str(e.get("country") or "")) == country_id:
                affected.append(str(e.get("key")))
    elif target_type == "region" and target_id:
        for e in entries:
            if _slug(str(e.get("region") or "")) == target_id:
                affected.append(str(e.get("key")))
    elif action == "region_sweep" and country_id:
        for e in entries:
            if _slug(str(e.get("country") or "")) == country_id:
                affected.append(str(e.get("key")))

    if not affected and target_id and target_id in entry_by_key:
        affected = [target_id]

    cost = 0.0 if cancel else _estimate_cost(body, action)
    gain = 0.0 if cancel else ACTION_SCOUTED_GAIN.get(action, 8.0) * _intensity_mult(str(body.get("intensity") or "normal"))

    for pid in affected:
        entry = entry_by_key.get(pid) or {}
        overlay = _prospect_overlay(state, pid)
        if cancel:
            continue
        current = _scouted_pct(entry, overlay, month)
        new_pct = min(100.0, current + gain)
        patch: Dict[str, Any] = {
            "scouted_percentage": round(new_pct, 1),
            "scouted_bonus": float(overlay.get("scouted_bonus") or 0.0) + max(0.0, new_pct - current),
        }
        note = f"{action.replace('_', ' ').title()} completed (+{round(new_pct - current, 1)}% coverage)."
        notes = list(overlay.get("notes") or [])
        notes.append(note)
        patch["notes"] = notes[-12:]

        if action == "interview":
            patch["interview_status"] = "Completed"
        elif action == "dinner":
            patch["dinner_status"] = "Completed"
        elif action == "combine":
            patch["combine_status"] = "Completed"
        elif action == "medical_review":
            flags = list(overlay.get("red_flags") or [])
            if new_pct >= 75 and "Medical — cleared pending final review" not in flags:
                flags.append("Medical — cleared pending final review")
            patch["red_flags"] = flags
        elif action == "character_check":
            traits = list(overlay.get("traits") or [])
            traits.append("Character profile updated")
            patch["traits"] = traits[-8:]

        _set_prospect_overlay(state, pid, patch)

    return affected, cost


def _upsert_assignment(
    state: Dict[str, Any],
    body: Mapping[str, Any],
    action: str,
    cost: float,
    affected: List[str],
    *,
    status: str = "active",
) -> Dict[str, Any]:
    assignments: List[Dict[str, Any]] = list(state.get("assignments") or [])
    aid = str(body.get("assignment_id") or uuid.uuid4())
    row = {
        "id": aid,
        "assignment_id": aid,
        "scout_id": body.get("scout_id"),
        "target_type": body.get("target_type") or ("player" if affected else "country"),
        "target_id": body.get("target_id") or (affected[0] if affected else ""),
        "action": action,
        "intensity": body.get("intensity") or "normal",
        "status": status,
        "progress": 100.0 if status == "completed" else min(95.0, 20.0 + len(affected) * 4),
        "cost": cost,
        "estimated_cost": cost,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "affected_prospects": affected,
        "context": body.get("context") or {},
    }
    replaced = False
    for i, a in enumerate(assignments):
        if str(a.get("id")) == aid:
            assignments[i] = row
            replaced = True
            break
    if not replaced:
        assignments.append(row)
    state["assignments"] = assignments[-80:]
    return row


def _apply_scouting_meta_patch(
    session: FranchiseSession,
    state: Dict[str, Any],
    body: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    ctx = body.get("context") if isinstance(body.get("context"), dict) else {}
    meta_patch = body.get("meta_patch")
    if not isinstance(meta_patch, dict):
        meta_patch = ctx.get("meta_patch") if isinstance(ctx.get("meta_patch"), dict) else None
    meta_only = bool(body.get("meta_only") or ctx.get("meta_only"))
    if not meta_only or not isinstance(meta_patch, dict):
        return None

    pid = str(
        body.get("prospect_id")
        or body.get("player_id")
        or body.get("target_id")
        or ctx.get("prospect_id")
        or ""
    )
    if not pid:
        return {"ok": False, "message": "meta_patch requires prospect_id."}

    allowed = {
        "watchlist",
        "target",
        "do_not_draft",
        "assigned_scout",
        "requested_reports",
        "traits",
        "notes",
        "red_flags",
        "combine_status",
        "interview_status",
        "dinner_status",
        "draft_stock",
    }
    patch = {k: v for k, v in meta_patch.items() if k in allowed}
    if not patch:
        return {"ok": False, "message": "meta_patch contained no allowed fields."}

    _set_prospect_overlay(state, pid, patch)
    invalidate_session_payload_caches(session, "scouting_meta")
    if patch.get("watchlist"):
        watchlist = list(state.get("watchlist") or [])
        if pid not in watchlist:
            watchlist.append(pid)
        state["watchlist"] = watchlist
    elif patch.get("watchlist") is False:
        state["watchlist"] = [x for x in list(state.get("watchlist") or []) if str(x) != pid]

    return {
        "ok": True,
        "message": "Scouting metadata updated.",
        "prospects": get_scouting_prospects(session).get("prospects"),
        "used_budget": float(state.get("used_budget") or 0.0),
    }


def apply_scouting_command(
    session: FranchiseSession,
    body: Mapping[str, Any],
    route_action: str,
) -> Dict[str, Any]:
    state = _ensure_scouting_state(session)
    meta_result = _apply_scouting_meta_patch(session, state, body)
    if meta_result is not None:
        return meta_result

    action = _resolve_action(body, POST_ROUTE_ACTION.get(route_action, route_action))

    if route_action == "cancel":
        aid = str(body.get("assignment_id") or "")
        for a in state.get("assignments") or []:
            if str(a.get("id")) == aid:
                a["status"] = "cancelled"
        return {
            "ok": True,
            "message": "Scouting assignment cancelled.",
            "assignments": state.get("assignments"),
        }

    affected, cost = _apply_to_targets(session, state, body, action)
    budget = float(state.get("budget") or DEFAULT_SCOUTING_BUDGET)
    used = float(state.get("used_budget") or 0.0)
    if used + cost > budget and cost > 0:
        return {
            "ok": False,
            "message": "Scouting budget exceeded. Reallocate funds or reduce intensity.",
        }

    state["used_budget"] = used + cost
    assignment = _upsert_assignment(state, body, action, cost, affected, status="completed")
    invalidate_session_payload_caches(session, "scouting_action")

    return {
        "ok": True,
        "message": f"Scouting action applied to {len(affected)} prospect(s).",
        "assignment": assignment,
        "assignments": state.get("assignments"),
        "prospects": get_scouting_prospects(session).get("prospects"),
        "used_budget": state["used_budget"],
    }
