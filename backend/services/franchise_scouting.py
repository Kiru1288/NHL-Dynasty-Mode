"""Franchise scouting API — draft-class prospects, assignments, and GM-driven coverage.

CANONICAL PATH for live franchise scouting overlay (scouted_percentage, assignments):
this module + franchise_sim.build_draft_class_rankings(). Legacy SimEngine prospect.py
and draft_class_generator.py are not used for live board intel.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from services.franchise_sim import get_cached_draft_class_rankings, ensure_prospect_stats_current_for_scouting
from services.franchise_session import FranchiseSession
from services.franchise_sim import invalidate_session_payload_caches

DRAFT_OVR_REVEAL_THRESHOLD = 72.0
PUBLIC_INTEL_CLEAR_THROUGH_RANK = 45

DEFAULT_SCOUTING_BUDGET = 2_500_000

# Passive dedicated-file growth (per calendar day) — caps below locked-read threshold
# unless the GM runs explicit scouting actions.
PASSIVE_ASSIGNED_DAILY = 0.42
PASSIVE_WATCHLIST_DAILY = 0.16
PASSIVE_TARGET_BONUS = 0.20
PASSIVE_REGION_DAILY = 0.10
PASSIVE_TOP_RANK_DAILY = 0.06
PASSIVE_TOP_RANK_CUTOFF = 120
PASSIVE_MAX_WITHOUT_ACTION = 62.0
PASSIVE_TARGET_CAP = 78.0
PASSIVE_ASSIGN_KICKSTART = 3.0
PASSIVE_FULL_FILE_DAYS = 14
MAX_ACTIVE_DEPLOYMENTS = 4
MAX_ASSIGNED_PASSIVE_TARGETS = 64
ASSIGNMENT_FREE_SLOTS = 12
ASSIGNMENT_CAP_SLOTS_DEFAULT = 5
SPOTLIGHT_MAX_TARGETS = 5
SPOTLIGHT_PASSIVE_DAILY_BONUS = 10.0
SPOTLIGHT_MIN_RANK = 100
HIDDEN_CEILING_POOL_SIZE = 20
CEILING_FOG_DECAY_MONTH = 10
PUBLIC_OVR_DISAGREEMENT_SPAN = 5.0
FILM_STUDY_GAIN = 12.0
FILM_STUDY_HOURS = 4
DEPLOYMENT_SALARY_PER_WEEK = 48_000
DISCOVERY_REP_LATE_BLOOMER = 8
DISCOVERY_REP_EARLY_FLAMER = 5


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
            "draft_combine": 5,
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
    "film_study": FILM_STUDY_GAIN,
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
    "film-study": "film_study",
    "film_study": "film_study",
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
    raw.setdefault("active_deployments", [])
    raw.setdefault("scout_spotlight", [])
    raw.setdefault("scout_reputation", 50.0)
    raw.setdefault("hidden_ceiling_pool", [])
    raw.setdefault("public_miss_markers", {})
    raw.setdefault("assignment_cap_slots", ASSIGNMENT_CAP_SLOTS_DEFAULT)
    raw.setdefault("discoveries", [])
    return raw


def _assignment_slot_limit(state: Mapping[str, Any]) -> int:
    slots = int(state.get("assignment_cap_slots") or ASSIGNMENT_CAP_SLOTS_DEFAULT)
    return min(MAX_ASSIGNED_PASSIVE_TARGETS, ASSIGNMENT_FREE_SLOTS + max(0, slots) * 10)


def _count_assigned_prospects(state: Mapping[str, Any]) -> int:
    prospects = state.get("prospects") if isinstance(state.get("prospects"), dict) else {}
    return sum(
        1
        for ov in prospects.values()
        if isinstance(ov, dict) and str(ov.get("assigned_scout") or "").strip()
    )


def _spotlight_ids(state: Mapping[str, Any]) -> List[str]:
    return [str(x) for x in (state.get("scout_spotlight") or []) if str(x).strip()]


def get_hidden_ceiling_pool(session: FranchiseSession) -> set:
    state = _ensure_scouting_state(session)
    return {str(x) for x in (state.get("hidden_ceiling_pool") or []) if str(x).strip()}


def ensure_scouting_markers(session: FranchiseSession, entries: List[Mapping[str, Any]]) -> None:
    """Seed hidden-ceiling sleepers (~20) and public-miss tags for the draft class."""
    state = _ensure_scouting_state(session)
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    if state.get("markers_season") == sy and state.get("hidden_ceiling_pool"):
        return

    pool_candidates: List[Tuple[float, int, str]] = []
    misses: Dict[str, str] = {}
    for e in entries:
        key = str(e.get("key") or e.get("id") or "")
        if not key:
            continue
        rank = int(e.get("rank") or 999)
        true_pot = float(e.get("true_potential_score") or e.get("potential_score") or 0)
        consensus = float(e.get("consensus_potential_score") or true_pot * 0.82)
        gap = true_pot - consensus
        bust = bool(e.get("is_bust_risk") or e.get("character_concerns"))

        if rank >= PUBLIC_INTEL_CLEAR_THROUGH_RANK + 1 and (
            gap >= 5.0 or (_scouting_rng(session, key, "hid") > 0.9 and gap >= 2.5)
        ):
            pool_candidates.append((gap, rank, key))

        if rank >= 130 and gap >= 7.0 and true_pot >= 78.0:
            misses[key] = "late_bloomer"
        elif 12 <= rank <= 38 and (bust or gap <= -5.0):
            misses[key] = "early_flamer"

    pool_candidates.sort(key=lambda x: (-x[0], x[1]))
    state["hidden_ceiling_pool"] = [k for _, _, k in pool_candidates[:HIDDEN_CEILING_POOL_SIZE]]
    state["public_miss_markers"] = misses
    state["markers_season"] = sy

    for key, mtype in misses.items():
        ov = _prospect_overlay(state, key)
        if not ov.get("public_miss_type"):
            _set_prospect_overlay(state, key, {"public_miss_type": mtype})


def _film_study_eligible(state: Mapping[str, Any], prospect_id: str) -> bool:
    watchlist = {str(x) for x in (state.get("watchlist") or [])}
    overlay = _prospect_overlay(state, prospect_id)
    if bool(overlay.get("watchlist")) or prospect_id in watchlist:
        return True
    return bool(str(overlay.get("assigned_scout") or "").strip())


def _apply_film_study_reveals(entry: Mapping[str, Any], patch: Dict[str, Any]) -> None:
    """Reveal off-sheet traits from tape — decision-making, maturity, system fit."""
    reveals: List[str] = []
    iq = entry.get("iq_rating") or entry.get("hockey_iq")
    coach = entry.get("coachability")
    compete = entry.get("compete") or entry.get("competitiveness")
    if iq is not None:
        reveals.append(f"Hockey IQ reads as {float(iq):.0f} on tape")
    if coach is not None:
        reveals.append(f"Coachability grade {float(coach):.0f}")
    if compete is not None:
        reveals.append(f"Compete level {float(compete):.0f}")
    role = str(entry.get("player_type") or entry.get("projection") or "").strip()
    if role:
        reveals.append(f"System fit: {role}")
    patch["film_study_complete"] = True
    patch["film_study_hours"] = FILM_STUDY_HOURS
    traits = list(patch.get("traits") or [])
    for line in reveals:
        if line not in traits:
            traits.append(line)
    patch["traits"] = traits[-10:]
    notes = list(patch.get("notes") or [])
    notes.append(f"Film study ({FILM_STUDY_HOURS}h) — off-ice traits unlocked.")
    patch["notes"] = notes[-12:]


def _maybe_award_discovery(
    session: FranchiseSession,
    state: Dict[str, Any],
    prospect_id: str,
    overlay: Mapping[str, Any],
    entry: Mapping[str, Any],
    new_pct: float,
) -> None:
    if new_pct < DRAFT_OVR_REVEAL_THRESHOLD:
        return
    if overlay.get("miss_discovered"):
        return
    mtype = str(overlay.get("public_miss_type") or (state.get("public_miss_markers") or {}).get(prospect_id) or "")
    if not mtype:
        return
    rep = float(state.get("scout_reputation") or 50.0)
    bonus = DISCOVERY_REP_LATE_BLOOMER if mtype == "late_bloomer" else DISCOVERY_REP_EARLY_FLAMER
    state["scout_reputation"] = round(min(99.0, rep + bonus), 1)
    discoveries = list(state.get("discoveries") or [])
    discoveries.append(
        {
            "prospect_id": prospect_id,
            "name": str(entry.get("name") or ""),
            "type": mtype,
            "rep_bonus": bonus,
            "rank": int(entry.get("rank") or 0),
        }
    )
    state["discoveries"] = discoveries[-24:]
    _set_prospect_overlay(state, prospect_id, {"miss_discovered": True})


def _deployment_cost_multiplier(state: Mapping[str, Any]) -> float:
    n = len(list(state.get("active_deployments") or []))
    return 1.0 + max(0, n) * 0.14


def _charge_deployment_salary(
    session: FranchiseSession,
    state: Dict[str, Any],
    *,
    cur_cursor: int,
) -> bool:
    deployments = list(state.get("active_deployments") or [])
    if not deployments:
        return False
    week_bucket = int(cur_cursor) // 7
    last_bucket = int(state.get("deployment_salary_week") or -1)
    if week_bucket <= last_bucket:
        return False
    weekly = len(deployments) * DEPLOYMENT_SALARY_PER_WEEK
    state["used_budget"] = float(state.get("used_budget") or 0.0) + float(weekly)
    state["deployment_salary_week"] = week_bucket
    state["deployment_salary_last"] = weekly
    return True


def _register_coverage_deployment(
    state: Dict[str, Any],
    session: FranchiseSession,
    *,
    kind: str,
    deploy_id: str,
) -> None:
    """Track an active regional/country deployment for passive daily coverage."""
    slug = _slug(deploy_id)
    if not slug or slug == "unknown":
        return
    deployments: List[Dict[str, Any]] = [
        d for d in list(state.get("active_deployments") or [])
        if not (str(d.get("kind") or "") == kind and _slug(str(d.get("id") or "")) == slug)
    ]
    deployments.append({
        "kind": kind,
        "id": deploy_id,
        "since_cursor": int(getattr(session, "calendar_cursor", 0) or 0),
    })
    state["active_deployments"] = deployments[-MAX_ACTIVE_DEPLOYMENTS:]


def _weeks_until_draft(month: Optional[int]) -> Optional[float]:
    if month is None:
        return None
    if month in (6, 7):
        return 0.0
    if month < 6:
        return round((6 - month) * 4.3, 1)
    return round((12 - month + 6) * 4.3, 1)


def _passive_cap_for_overlay(
    overlay: Mapping[str, Any],
    *,
    assigned: bool,
    is_target: bool,
    cur_cursor: int,
) -> float:
    """Cap for dedicated-file passive growth; assigned scouts unlock 100% after 14 days."""
    if assigned:
        assigned_at = overlay.get("assigned_at_cursor")
        if assigned_at is not None:
            try:
                if max(0, cur_cursor - int(assigned_at)) >= PASSIVE_FULL_FILE_DAYS:
                    return 100.0
            except (TypeError, ValueError):
                pass
        return PASSIVE_TARGET_CAP
    if is_target:
        return PASSIVE_TARGET_CAP
    return PASSIVE_MAX_WITHOUT_ACTION


def _passive_daily_rate(
    entry: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    assigned: bool,
    on_watchlist: bool,
    is_target: bool,
    in_deployed_region: bool,
    assigned_under_cap: bool,
) -> float:
    rate = 0.0
    if assigned and assigned_under_cap:
        rate += PASSIVE_ASSIGNED_DAILY
    if on_watchlist:
        rate += PASSIVE_WATCHLIST_DAILY
    if is_target:
        rate += PASSIVE_TARGET_BONUS
    if in_deployed_region:
        rate += PASSIVE_REGION_DAILY
    rank = int(entry.get("rank") or 999)
    try:
        current = float(overlay.get("scouted_percentage") or 0.0)
    except (TypeError, ValueError):
        current = 0.0
    if current <= 0.0 and rank <= PASSIVE_TOP_RANK_CUTOFF:
        rate += PASSIVE_TOP_RANK_DAILY
    return rate


def _scout_file_timeline(
    session: FranchiseSession,
    entry: Mapping[str, Any],
    overlay: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project days to OVR reveal (72%) and full file (100%) from passive growth."""
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    cur_cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    try:
        current = float(overlay.get("scouted_percentage") or 0.0)
    except (TypeError, ValueError):
        current = 0.0

    assigned = bool(str(overlay.get("assigned_scout") or "").strip())
    watchlist = {str(x) for x in (state.get("watchlist") or [])}
    pid = str(entry.get("key") or entry.get("id") or "")
    on_watchlist = bool(overlay.get("watchlist")) or pid in watchlist
    is_target = bool(overlay.get("target"))
    deployments = list(state.get("active_deployments") or [])
    deployed_regions = {
        _slug(str(d.get("id") or ""))
        for d in deployments
        if str(d.get("kind") or "") == "region"
    }
    deployed_countries = {
        _slug(str(d.get("id") or ""))
        for d in deployments
        if str(d.get("kind") or "") == "country"
    }
    country_slug = _slug(str(entry.get("country") or ""))
    region_slug = _slug(str(entry.get("region") or ""))
    if not region_slug:
        region_slug = _slug(_region_for_country(str(entry.get("country") or "")))
    in_deployed = country_slug in deployed_countries or region_slug in deployed_regions

    prospects = state.get("prospects") if isinstance(state.get("prospects"), dict) else {}
    assigned_count = sum(
        1
        for ov in prospects.values()
        if isinstance(ov, dict) and str(ov.get("assigned_scout") or "").strip()
    )
    assigned_under_cap = assigned_count <= _assignment_slot_limit(state)
    spotlight = set(_spotlight_ids(state))
    pid_key = str(entry.get("key") or entry.get("id") or "")

    daily = _passive_daily_rate(
        entry,
        overlay,
        assigned=assigned,
        on_watchlist=on_watchlist,
        is_target=is_target,
        in_deployed_region=in_deployed,
        assigned_under_cap=assigned_under_cap,
    )
    if pid_key in spotlight and assigned:
        daily += SPOTLIGHT_PASSIVE_DAILY_BONUS
    if month == 7:
        daily *= 0.35

    cap = _passive_cap_for_overlay(
        overlay, assigned=assigned, is_target=is_target, cur_cursor=cur_cursor
    )
    weeks_draft = _weeks_until_draft(month)

    def _days_to(target: float) -> Optional[int]:
        if daily <= 0.0 or current >= min(target, cap):
            return 0 if current >= target else None
        need = min(target, cap) - current
        if need <= 0:
            return 0
        return int(math.ceil(need / daily))

    days_72 = _days_to(DRAFT_OVR_REVEAL_THRESHOLD)
    days_100 = _days_to(100.0) if cap >= 100.0 else None
    days_full_cap = _days_to(cap) if cap < 100.0 else days_100

    assigned_at = overlay.get("assigned_at_cursor")
    days_until_full_unlock = None
    if assigned and assigned_at is not None:
        try:
            days_until_full_unlock = max(0, PASSIVE_FULL_FILE_DAYS - (cur_cursor - int(assigned_at)))
        except (TypeError, ValueError):
            days_until_full_unlock = PASSIVE_FULL_FILE_DAYS

    return {
        "dedicated_pct": round(current, 1),
        "passive_daily_pct": round(daily, 2),
        "passive_cap_pct": round(cap, 1),
        "days_to_ovr_reveal": days_72,
        "days_to_full_file": days_100,
        "days_to_passive_cap": days_full_cap,
        "weeks_until_draft": weeks_draft,
        "ovr_reveal_before_draft": (
            days_72 is not None and weeks_draft is not None and days_72 <= weeks_draft * 7
        ) if days_72 is not None and weeks_draft is not None else None,
        "full_file_before_draft": (
            days_100 is not None and weeks_draft is not None and days_100 <= weeks_draft * 7
        ) if days_100 is not None and weeks_draft is not None else None,
        "days_until_passive_full_unlock": days_until_full_unlock,
        "passive_cap_tooltip": (
            "Passive files cap at 62% without an assigned scout. "
            "Assign a scout — after 14 days on file, passive growth can reach 100%. "
            "Deep-dive actions (interview, medical) still accelerate locked reads."
        ),
    }


def _scouting_intel_rules() -> Dict[str, Any]:
    return {
        "public_ceiling_clear_through_rank": PUBLIC_INTEL_CLEAR_THROUGH_RANK,
        "public_ceiling_label": (
            f"Public ceiling intel is clear through pick ~{PUBLIC_INTEL_CLEAR_THROUGH_RANK}. "
            "Later picks need your dedicated file (YOU column) to reopen ceiling fog."
        ),
        "passive_cap_without_assign": PASSIVE_MAX_WITHOUT_ACTION,
        "passive_cap_assigned": PASSIVE_TARGET_CAP,
        "passive_full_file_days": PASSIVE_FULL_FILE_DAYS,
        "passive_cap_tooltip": (
            "Passive files cap at 62% without an assigned scout. "
            "Assign a scout — after 14 days on file, passive growth can reach 100%. "
            "Deep-dive actions (interview, medical) still accelerate locked reads."
        ),
        "ovr_reveal_threshold": DRAFT_OVR_REVEAL_THRESHOLD,
        "max_active_deployments": MAX_ACTIVE_DEPLOYMENTS,
        "region_sweep_gain_pct": ACTION_SCOUTED_GAIN["region_sweep"],
        "region_passive_daily_pct": PASSIVE_REGION_DAILY,
        "spotlight_max_targets": SPOTLIGHT_MAX_TARGETS,
        "spotlight_daily_bonus_pct": SPOTLIGHT_PASSIVE_DAILY_BONUS,
        "hidden_ceiling_pool_size": HIDDEN_CEILING_POOL_SIZE,
        "ceiling_fog_decay_month": CEILING_FOG_DECAY_MONTH,
        "public_ovr_disagreement_span": PUBLIC_OVR_DISAGREEMENT_SPAN,
        "film_study_hours": FILM_STUDY_HOURS,
        "deployment_salary_per_week": DEPLOYMENT_SALARY_PER_WEEK,
        "assignment_free_slots": ASSIGNMENT_FREE_SLOTS,
        "assignment_cap_slots_default": ASSIGNMENT_CAP_SLOTS_DEFAULT,
    }


def apply_passive_scouting_progress(session: FranchiseSession, *, days: int = 1) -> bool:
    """Apply passive dedicated-file growth for assigned/shortlist/deployed targets."""
    step_days = max(1, int(days or 1))
    state = _ensure_scouting_state(session)
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    offseason_gain_mult = 0.35 if month == 7 else 1.0

    entries = _draft_entries(session)
    if not entries:
        return False
    ensure_scouting_markers(session, entries)

    prospects = state.get("prospects") if isinstance(state.get("prospects"), dict) else {}
    watchlist = {str(x) for x in (state.get("watchlist") or [])}
    spotlight = set(_spotlight_ids(state))
    deployments = list(state.get("active_deployments") or [])
    deployed_regions = {
        _slug(str(d.get("id") or ""))
        for d in deployments
        if str(d.get("kind") or "") == "region"
    }
    deployed_countries = {
        _slug(str(d.get("id") or ""))
        for d in deployments
        if str(d.get("kind") or "") == "country"
    }

    assigned_limit = _assignment_slot_limit(state)
    assigned_count = _count_assigned_prospects(state)
    cur_cursor = int(getattr(session, "calendar_cursor", 0) or 0)

    changed = _charge_deployment_salary(session, state, cur_cursor=cur_cursor)
    for entry in entries:
        pid = str(entry.get("key") or entry.get("id") or "")
        if not pid:
            continue
        overlay = _prospect_overlay(state, pid)
        try:
            current = float(overlay.get("scouted_percentage") or 0.0)
        except (TypeError, ValueError):
            current = 0.0
        if current >= 100.0:
            continue

        gain = 0.0
        assigned = str(overlay.get("assigned_scout") or "").strip()
        on_watchlist = bool(overlay.get("watchlist")) or pid in watchlist
        is_target = bool(overlay.get("target"))

        if assigned and assigned_count <= assigned_limit:
            gain += PASSIVE_ASSIGNED_DAILY
        if pid in spotlight and assigned:
            gain += SPOTLIGHT_PASSIVE_DAILY_BONUS
        if on_watchlist:
            gain += PASSIVE_WATCHLIST_DAILY
        if is_target:
            gain += PASSIVE_TARGET_BONUS

        country_slug = _slug(str(entry.get("country") or ""))
        region_slug = _slug(str(entry.get("region") or ""))
        if not region_slug:
            region_slug = _slug(_region_for_country(str(entry.get("country") or "")))
        if country_slug in deployed_countries or region_slug in deployed_regions:
            gain += PASSIVE_REGION_DAILY

        rank = int(entry.get("rank") or 999)
        if current <= 0.0 and rank <= PASSIVE_TOP_RANK_CUTOFF:
            gain += PASSIVE_TOP_RANK_DAILY

        if gain <= 0.0:
            continue

        gain *= float(step_days) * offseason_gain_mult

        cap = _passive_cap_for_overlay(
            overlay,
            assigned=bool(assigned),
            is_target=is_target,
            cur_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
        )

        new_pct = min(cap, current + gain)
        if new_pct <= current + 0.005:
            continue

        _set_prospect_overlay(state, pid, {"scouted_percentage": round(new_pct, 1)})
        _maybe_award_discovery(session, state, pid, overlay, entry, new_pct)
        changed = True

    if changed:
        session._cached_scouting_prospects_payload = None
        session._cached_scouting_world_payload = None
    return changed


def _region_for_country(country: str) -> str:
    cl = str(country or "").strip().lower()
    if cl in ("canada", "united states", "usa", "us"):
        return "North America"
    if cl in ("sweden", "finland", "norway", "denmark"):
        return "Scandinavia"
    if cl in ("russia", "belarus", "kazakhstan"):
        return "Russia / CIS"
    if cl and cl not in ("unknown", ""):
        return "Europe"
    return "International"


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
    spec_map = {
        "North America": "USA / CHL / NCAA",
        "Europe": "Europe / CIS",
        "Scandinavia": "Scandinavia",
        "Russia / CIS": "Russia / CIS",
        "Goaltending": "Goaltending",
    }
    for i, (region, code) in enumerate(regions):
        sid = f"scout-{code.lower()}"
        quality = 68 + ((seed + i * 17) % 28)
        speed_mult = 0.82 + (quality - 68) / 35.0
        days_to_72 = int(round(DRAFT_OVR_REVEAL_THRESHOLD / max(0.15, PASSIVE_ASSIGNED_DAILY * speed_mult)))
        tier = "Elite" if quality >= 88 else "Strong" if quality >= 78 else "Average"
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
                "accuracy": quality,
                "evaluation_accuracy": quality,
                "workload": 0,
                "specialty": region,
                "specialization": spec_map.get(region, region),
                "speed_days_to_72": days_to_72,
                "scout_tier": tier,
                "character_evaluation": min(95, quality + 4 + (seed % 7)),
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
    session: Optional[FranchiseSession] = None,
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
        "spotlight": bool(overlay.get("spotlight"))
        or (session is not None and pid in _spotlight_ids(_ensure_scouting_state(session))),
        "public_miss_type": str(overlay.get("public_miss_type") or ""),
        "film_study_complete": bool(overlay.get("film_study_complete")),
        "miss_discovered": bool(overlay.get("miss_discovered")),
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
        "actual_stats",
        "projected_stats",
        "stats_mode",
        "recent_form",
        "projected_gp",
        "projected_goals",
        "projected_assists",
        "projected_points",
        "projected_ppg",
        "stock_delta",
        "stock_label",
        "stock_trend",
        "stock_reason",
        "scouting_confidence",
        "preseason_rank",
        "previous_rank",
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
    if session is not None:
        state = _ensure_scouting_state(session)
        try:
            dedicated = float(overlay.get("scouted_percentage") or 0.0)
        except (TypeError, ValueError):
            dedicated = 0.0
        out["dedicated_scouted_percentage"] = round(dedicated, 1)
        out["scout_timeline"] = _scout_file_timeline(session, entry, overlay, state)
    return out


def _draft_entries(session: FranchiseSession) -> List[Dict[str, Any]]:
    sim = getattr(session, "sim", None)
    rev = int(getattr(session, "_prospect_revision", 0) or 0)
    cached = getattr(session, "_cached_draft_class_rankings", None)
    cached_rev = int(getattr(session, "_cached_draft_class_rankings_rev", -1) or -1)
    if isinstance(cached, dict) and cached.get("entries") and cached_rev == rev:
        return list(cached.get("entries") or [])
    board = get_cached_draft_class_rankings(session, sim)
    return list(board.get("entries") or [])


def get_scouting_prospects(session: FranchiseSession) -> Dict[str, Any]:
    # Prefer warm draft-board cache. Avoid a second full junior-stats sync when the
    # calendar line is already current — that sync was dominating Draft Class opens.
    iso = ""
    try:
        cal = getattr(session, "nhl_calendar", None) or []
        cur = int(getattr(session, "calendar_cursor", 0) or 0)
        if 0 <= cur < len(cal):
            iso = str(cal[cur].get("iso") or cal[cur].get("date") or "")
    except Exception:
        iso = ""
    last_iso = str(getattr(session, "_prospect_stats_synced_iso", "") or "")
    if not (iso and last_iso and iso == last_iso):
        ensure_prospect_stats_current_for_scouting(session)
    cached = getattr(session, "_cached_scouting_prospects_payload", None)
    if isinstance(cached, dict) and cached:
        return cached
    state = _ensure_scouting_state(session)
    ctx = _draft_franchise_date_context(session)
    month = ctx.get("month")
    entries = _draft_entries(session)
    ensure_scouting_markers(session, entries)
    prospects = [
        _normalize_prospect(e, _prospect_overlay(state, str(e.get("key") or "")), month, session)
        for e in entries
    ]
    message = None if prospects else "No draft-class prospects in the active universe yet."
    payload = {"prospects": prospects, "total": len(prospects), "message": message}
    session._cached_scouting_prospects_payload = payload
    return payload


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
        "weeks_until_draft": _weeks_until_draft(ctx.get("month")),
        "intel_rules": _scouting_intel_rules(),
        "scouts": scouts,
        "staff": scouts,
        "budget": float(state.get("budget") or DEFAULT_SCOUTING_BUDGET),
        "used_budget": float(state.get("used_budget") or 0.0),
        "budget_remaining": float(state.get("budget") or DEFAULT_SCOUTING_BUDGET)
        - float(state.get("used_budget") or 0.0),
        "active_deployments": list(state.get("active_deployments") or []),
        "scout_spotlight": _spotlight_ids(state),
        "scout_reputation": round(float(state.get("scout_reputation") or 50.0), 1),
        "assignment_slot_limit": _assignment_slot_limit(state),
        "assignment_slots_used": _count_assigned_prospects(state),
        "assignment_cap_slots": int(state.get("assignment_cap_slots") or ASSIGNMENT_CAP_SLOTS_DEFAULT),
        "hidden_ceiling_pool_size": len(list(state.get("hidden_ceiling_pool") or [])),
        "discoveries": list(state.get("discoveries") or [])[-8:],
        "deployment_salary_per_week": DEPLOYMENT_SALARY_PER_WEEK,
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
        sweep_gain = float(ACTION_SCOUTED_GAIN["region_sweep"])
        passive_daily = float(PASSIVE_REGION_DAILY)
        pc = int(row["prospect_count"])
        countries.append(
            {
                "id": cid,
                "name": row["name"],
                "region": row["region"],
                "prospect_count": pc,
                "scouted_average": round(avg, 1),
                "cost": 35_000 + pc * 4_500 + (seed % 12) * 1_000,
                "effort": min(95, 40 + pc * 3 + (seed % 20)),
                "difficulty": min(90, 25 + (seed % 40)),
                "safety_risk": 0,
                "political_risk": 0,
                "corruption_risk": min(35, seed % 30),
                "travel_notes": [],
                "deploy_sweep_gain_pct": sweep_gain,
                "deploy_passive_daily_pct": passive_daily,
                "deploy_roi_label": (
                    f"{pc} prospects · +{sweep_gain:.0f}% each now · +{passive_daily:.2f}%/day passive"
                ),
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
        sweep_gain = float(ACTION_SCOUTED_GAIN["region_sweep"])
        passive_daily = float(PASSIVE_REGION_DAILY)
        pc = int(row["prospect_count"])
        regions.append(
            {
                "id": rid,
                "name": row["name"],
                "prospect_count": pc,
                "country_count": row["country_count"],
                "scouted_average": round(row["scouted_sum"] / cnt, 1),
                "deploy_sweep_gain_pct": sweep_gain,
                "deploy_passive_daily_pct": passive_daily,
                "deploy_roi_label": (
                    f"{pc} prospects · +{sweep_gain:.0f}% each now · +{passive_daily:.2f}%/day passive"
                ),
            }
        )

    return {"countries": countries, "regions": regions}


def get_scouting_world(session: FranchiseSession) -> Dict[str, Any]:
    cached = getattr(session, "_cached_scouting_world_payload", None)
    if isinstance(cached, dict) and cached:
        return cached
    prospects_payload = get_scouting_prospects(session)
    prospects = prospects_payload.get("prospects") or []
    world = _aggregate_world(prospects)
    ctx = _draft_franchise_date_context(session)
    world["generated_at"] = ctx.get("generated_at")
    world["message"] = prospects_payload.get("message")
    world["intel_rules"] = _scouting_intel_rules()
    session._cached_scouting_world_payload = world
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


def _estimate_cost(body: Mapping[str, Any], action: str, state: Optional[Mapping[str, Any]] = None) -> float:
    explicit = body.get("estimated_cost")
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    base = 18_000.0 + ACTION_SCOUTED_GAIN.get(action, 8.0) * 2_200.0
    cost = base * _intensity_mult(str(body.get("intensity") or "normal"))
    if state is not None:
        cost *= _deployment_cost_multiplier(state)
    return cost


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

    cost = 0.0 if cancel else _estimate_cost(body, action, state)
    gain = 0.0 if cancel else ACTION_SCOUTED_GAIN.get(action, 8.0) * _intensity_mult(str(body.get("intensity") or "normal"))

    if action == "film_study" and not cancel:
        if target_type in ("player", "prospect") and target_id:
            if not _film_study_eligible(state, target_id):
                return [], 0.0
        else:
            return [], 0.0

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
        elif action == "film_study":
            _apply_film_study_reveals(entry, patch)

        _set_prospect_overlay(state, pid, patch)
        _maybe_award_discovery(session, state, pid, {**overlay, **patch}, entry, new_pct)

    if action == "region_sweep":
        if target_type == "country" and country_id:
            _register_coverage_deployment(state, session, kind="country", deploy_id=country_id)
        elif target_type == "region" and target_id:
            _register_coverage_deployment(state, session, kind="region", deploy_id=target_id)
        elif country_id:
            _register_coverage_deployment(state, session, kind="country", deploy_id=country_id)

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
        "spotlight",
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

    overlay_before = _prospect_overlay(state, pid)
    if patch.get("spotlight"):
        rank = 999
        for e in _draft_entries(session):
            if str(e.get("key") or "") == pid:
                rank = int(e.get("rank") or 999)
                break
        if rank < SPOTLIGHT_MIN_RANK:
            return {
                "ok": False,
                "message": f"Scout Spotlight is for rank {SPOTLIGHT_MIN_RANK}+ sleepers only.",
            }
        spotlight = _spotlight_ids(state)
        if pid not in spotlight and len(spotlight) >= SPOTLIGHT_MAX_TARGETS:
            return {
                "ok": False,
                "message": f"Scout Spotlight full ({SPOTLIGHT_MAX_TARGETS} slots). Remove one first.",
            }
        if pid not in spotlight:
            spotlight.append(pid)
        state["scout_spotlight"] = spotlight[:SPOTLIGHT_MAX_TARGETS]
        patch["spotlight"] = True
    elif patch.get("spotlight") is False:
        state["scout_spotlight"] = [x for x in _spotlight_ids(state) if x != pid]
        patch["spotlight"] = False

    if patch.get("assigned_scout") and str(patch.get("assigned_scout") or "").strip():
        if not str(overlay_before.get("assigned_scout") or "").strip():
            if _count_assigned_prospects(state) >= _assignment_slot_limit(state):
                return {
                    "ok": False,
                    "message": (
                        "Assignment cap reached — each extra slot beyond the free tier "
                        "consumes a prospect cap slot. Clear a scout or raise cap slots."
                    ),
                }
            try:
                current = float(overlay_before.get("scouted_percentage") or 0.0)
            except (TypeError, ValueError):
                current = 0.0
            patch["scouted_percentage"] = round(
                min(100.0, max(current, current + PASSIVE_ASSIGN_KICKSTART)),
                1,
            )
        patch["assigned_at_cursor"] = int(getattr(session, "calendar_cursor", 0) or 0)
    elif "assigned_scout" in patch and not str(patch.get("assigned_scout") or "").strip():
        patch["assigned_at_cursor"] = None

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
        "scout_reputation": round(float(state.get("scout_reputation") or 50.0), 1),
        "scout_spotlight": _spotlight_ids(state),
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

    if action == "film_study":
        pid = str(
            body.get("prospect_id")
            or body.get("player_id")
            or body.get("target_id")
            or (body.get("context") or {}).get("prospect_id")
            or ""
        )
        if not pid or not _film_study_eligible(state, pid):
            return {
                "ok": False,
                "message": "Film study requires an assigned scout or watchlist slot (4 hours).",
            }

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
    if action == "film_study" and not affected:
        return {
            "ok": False,
            "message": "Film study requires an assigned scout or watchlist slot (4 hours).",
        }
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


def _scouting_rng(session: FranchiseSession, *parts: Any) -> float:
    seed = hashlib.md5(
        f"{session.session_id}:scout:{':'.join(str(p) for p in parts)}".encode()
    ).hexdigest()
    return int(seed[:8], 16) / 0xFFFFFFFF


def _league_code_bucket(entry: Mapping[str, Any]) -> str:
    code = str(entry.get("league_code") or entry.get("league") or "").upper()
    if "NCAA" in code:
        return "NCAA"
    if code.startswith("EU_") or code in ("SHL", "LIIGA", "KHL"):
        return "Europe"
    if code.startswith("CHL") or code in ("OHL", "WHL", "QMJHL", "USHL"):
        return "CHL"
    return "CHL"


def ensure_team_scouting_profiles(session: FranchiseSession) -> Dict[str, Dict[str, Any]]:
    """Deterministic per-team scouting departments for CPU draft boards."""
    state = _ensure_scouting_state(session)
    existing = state.get("team_profiles")
    if isinstance(existing, dict) and len(existing) >= len(session.team_ids or []):
        return existing

    profiles: Dict[str, Dict[str, Any]] = {}
    regions = ["North America", "Europe", "Scandinavia", "Russia / CIS", "Goaltending"]
    league_opts = ["CHL", "NCAA", "Europe"]

    for tid in session.team_ids or []:
        team = session.team_by_id.get(str(tid))
        dev_q = float(getattr(team, "development_quality", 0.5) or 0.5)
        risk = float(getattr(team, "risk_tolerance", 0.5) or 0.5)
        archetype = str(getattr(team, "archetype", "") or "").lower()
        seed = _scouting_rng(session, "profile", tid)

        scouting_quality = round(55 + dev_q * 35 + (seed - 0.5) * 12, 1)
        budget = int(1_400_000 + dev_q * 1_800_000 + seed * 400_000)
        n_scouts = 4 + int(dev_q * 4) + int(seed * 3)

        reg_strength = regions[int(seed * 5) % len(regions)]
        reg_weak = regions[int((1.0 - seed) * 5) % len(regions)]
        league_bias = league_opts[int(seed * 3) % len(league_opts)]

        public_trust = round(0.35 + (1.0 - dev_q) * 0.25 + seed * 0.15, 2)
        internal_trust = round(0.45 + dev_q * 0.35, 2)
        interview_trust = round(0.4 + dev_q * 0.3 + (seed - 0.5) * 0.1, 2)
        combine_trust = round(0.35 + dev_q * 0.4, 2)

        if "analytics" in archetype:
            public_trust = min(0.85, public_trust + 0.15)
            league_bias = "NCAA"
        if "old" in archetype or "traditional" in archetype:
            internal_trust = min(0.9, internal_trust + 0.12)
            league_bias = "CHL"

        profiles[str(tid)] = {
            "team_id": str(tid),
            "scouting_quality": scouting_quality,
            "amateur_scouting_budget": budget,
            "number_of_scouts": n_scouts,
            "regional_strengths": [reg_strength],
            "regional_weaknesses": [reg_weak],
            "league_biases": [league_bias],
            "public_board_trust": public_trust,
            "internal_board_trust": internal_trust,
            "interview_trust": interview_trust,
            "combine_trust": combine_trust,
            "risk_tolerance": round(risk, 2),
            "sleeper_detection": round(0.3 + dev_q * 0.45 + seed * 0.2, 2),
            "red_flag_detection": round(0.25 + dev_q * 0.5 + (1.0 - risk) * 0.15, 2),
            "goalie_scouting_quality": round(scouting_quality * (0.85 + seed * 0.2), 1),
            "defenseman_scouting_quality": round(scouting_quality * (0.9 + (1.0 - seed) * 0.15), 1),
            "forward_scouting_quality": round(scouting_quality * (0.95 + seed * 0.1), 1),
            "European_scouting_quality": round(scouting_quality * (0.7 + seed * 0.35), 1),
            "CHL_scouting_quality": round(scouting_quality * (0.85 + (1.0 - seed) * 0.2), 1),
            "NCAA_scouting_quality": round(scouting_quality * (0.75 + seed * 0.3), 1),
            "GM_scout_alignment": round(0.45 + dev_q * 0.35, 2),
            "GM_scout_disagreement_chance": round(0.08 + risk * 0.22, 2),
            "off_board_tendency": round(0.1 + risk * 0.35 + (1.0 - public_trust) * 0.2, 2),
            "do_not_draft_strictness": round(0.35 + (1.0 - risk) * 0.4, 2),
        }

    state["team_profiles"] = profiles
    session.scouting_state = state
    return profiles


def get_team_scouting_profile(session: FranchiseSession, team_id: str) -> Dict[str, Any]:
    profiles = ensure_team_scouting_profiles(session)
    return dict(profiles.get(str(team_id)) or {})


def get_team_prospect_impression(
    session: FranchiseSession,
    team_id: str,
    prospect_id: str,
) -> Dict[str, Any]:
    state = _ensure_scouting_state(session)
    impressions = state.get("team_impressions") if isinstance(state.get("team_impressions"), dict) else {}
    team_row = impressions.get(str(team_id)) if isinstance(impressions.get(str(team_id)), dict) else {}
    return dict(team_row.get(str(prospect_id)) or {})


def _combine_label(score: float) -> str:
    if score >= 82:
        return "Elite"
    if score >= 72:
        return "Strong"
    if score >= 62:
        return "Average"
    if score >= 50:
        return "Below Average"
    return "Poor"


def _generate_combine_prospect_results(
    session: FranchiseSession,
    entry: Mapping[str, Any],
) -> Dict[str, Any]:
    pid = str(entry.get("key") or "")
    true_skating = float(entry.get("skating") or entry.get("true_ovr") or 70)
    true_strength = float(entry.get("physical") or entry.get("true_ovr") or 68)
    true_agility = float(entry.get("compete") or entry.get("true_ovr") or 70)
    true_endurance = float(entry.get("consistency") or entry.get("true_ovr") or 68)
    noise = (_scouting_rng(session, "combine", pid, "phys") - 0.5) * 14.0

    skating = max(35, min(99, true_skating + noise))
    strength = max(35, min(99, true_strength + (_scouting_rng(session, pid, "str") - 0.5) * 12))
    agility = max(35, min(99, true_agility + (_scouting_rng(session, pid, "agi") - 0.5) * 12))
    endurance = max(35, min(99, true_endurance + (_scouting_rng(session, pid, "end") - 0.5) * 10))
    combine_score = round((skating + strength + agility + endurance) / 4.0, 1)

    med_roll = _scouting_rng(session, pid, "med")
    medical_flag = med_roll < 0.12 or bool(entry.get("injury_history"))
    medical_risk = "High" if med_roll < 0.05 else "Moderate" if medical_flag else "Low"

    char_roll = _scouting_rng(session, pid, "char")
    interview_score = round(50 + char_roll * 40 + (float(entry.get("coachability") or 70) - 70) * 0.3, 1)
    if entry.get("character_concerns"):
        interview_score = max(35, interview_score - 18)

    stock_delta = 0
    if combine_score >= 78:
        stock_delta = 3 + int(_scouting_rng(session, pid, "rise") * 4)
    elif combine_score <= 52:
        stock_delta = -2 - int(_scouting_rng(session, pid, "fall") * 4)
    if medical_flag and medical_risk == "High":
        stock_delta -= 4

    return {
        "prospect_id": pid,
        "combine_invited": True,
        "combine_attended": True,
        "combine_score": combine_score,
        "skating_test_score": round(skating, 1),
        "strength_test_score": round(strength, 1),
        "agility_test_score": round(agility, 1),
        "endurance_score": round(endurance, 1),
        "medical_flag": medical_flag,
        "medical_risk_level": medical_risk,
        "interview_score": interview_score,
        "interview_summary": (
            "Strong leadership and compete level in team interviews."
            if interview_score >= 72
            else "Some questions about consistency and compete."
            if interview_score < 55
            else "Solid interview — no major red flags."
        ),
        "combine_stock_delta": stock_delta,
        "combine_stock_reason": (
            "Elite testing numbers"
            if stock_delta >= 3
            else "Medical concern surfaced"
            if medical_flag and stock_delta < 0
            else "Underwhelming athletic testing"
            if stock_delta <= -2
            else "Met expectations"
        ),
    }


def _generate_team_impression(
    session: FranchiseSession,
    team_id: str,
    entry: Mapping[str, Any],
    combine: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    pid = str(entry.get("key") or "")
    pub_rank = int(entry.get("rank") or 999)
    league = _league_code_bucket(entry)
    region = str(entry.get("region") or "North America")
    pos = str(entry.get("position") or "").upper()

    iq_noise = (_scouting_rng(session, team_id, pid, "iq") - 0.5) * 2.0
    quality = float(profile.get("scouting_quality") or 60) / 100.0
    league_fit = 1.0
    if league in (profile.get("league_biases") or []):
        league_fit = 1.12
    elif league == "Europe" and float(profile.get("European_scouting_quality") or 60) < 62:
        league_fit = 0.88

    region_fit = 1.0
    if region in (profile.get("regional_strengths") or []):
        region_fit = 1.1
    elif region in (profile.get("regional_weaknesses") or []):
        region_fit = 0.85

    interview_imp = float(combine.get("interview_score") or 60)
    interview_imp += (iq_noise + (quality - 0.5) * 8) * float(profile.get("interview_trust") or 0.5)
    combine_imp = float(combine.get("combine_score") or 60) * float(profile.get("combine_trust") or 0.5)
    medical_imp = -12.0 if combine.get("medical_flag") and combine.get("medical_risk_level") == "High" else (
        -4.0 if combine.get("medical_flag") else 0.0
    )
    if float(profile.get("red_flag_detection") or 0.5) > 0.65 and combine.get("medical_flag"):
        medical_imp -= 4.0

    board_delta = (interview_imp - 60) * 0.08 + (combine_imp - 60) * 0.06 + medical_imp * 0.5
    board_delta *= league_fit * region_fit
    if pos == "G":
        gq = float(profile.get("goalie_scouting_quality") or 60)
        board_delta += (gq - 65) * 0.05

    risk_delta = medical_imp * 0.3
    if entry.get("character_concerns") and float(profile.get("do_not_draft_strictness") or 0.5) > 0.55:
        risk_delta -= 6.0
        board_delta -= 3.0

    confidence_delta = round((quality - 0.45) * 6 + league_fit * 2, 2)
    do_not_draft = False
    if combine.get("medical_risk_level") == "High" and float(profile.get("do_not_draft_strictness") or 0) > 0.7:
        do_not_draft = True
    if entry.get("character_concerns") and float(profile.get("risk_tolerance") or 0.5) < 0.35:
        do_not_draft = True

    scout_favorite = False
    sleeper_tag = False
    concern_tag = False
    if board_delta >= 4.0 and _scouting_rng(session, team_id, pid, "fav") < float(profile.get("off_board_tendency") or 0.2) + 0.15:
        scout_favorite = True
    if pub_rank > 45 and board_delta >= 5.0 and float(profile.get("sleeper_detection") or 0) > 0.55:
        sleeper_tag = True
    if board_delta <= -5.0 or do_not_draft:
        concern_tag = True

    scout_note = ""
    gm_note = ""
    if scout_favorite:
        scout_note = "Regional scout pushing hard — strong internal conviction."
    elif sleeper_tag:
        scout_note = "Late-round sleeper with tools that translate."
    elif concern_tag:
        scout_note = "Internal concern — recommend passing unless value extreme."
    if float(profile.get("GM_scout_disagreement_chance") or 0) > 0.25 and _scouting_rng(session, team_id, pid, "gm") < 0.3:
        gm_note = "GM prefers safer floor over scout's upside case."
        board_delta -= 2.0

    return {
        "team_id": str(team_id),
        "prospect_id": pid,
        "interview_impression": _combine_label(interview_imp),
        "combine_impression": _combine_label(combine_imp),
        "medical_impression": str(combine.get("medical_risk_level") or "Low"),
        "private_meeting_impression": "Not held",
        "scout_note": scout_note,
        "gm_note": gm_note,
        "board_delta": round(board_delta, 2),
        "risk_delta": round(risk_delta, 2),
        "confidence_delta": confidence_delta,
        "do_not_draft": do_not_draft,
        "scout_favorite": scout_favorite,
        "sleeper_tag": sleeper_tag,
        "concern_tag": concern_tag,
    }


def _select_combine_invites(session: FranchiseSession, entries: List[Dict[str, Any]]) -> List[str]:
    if not entries:
        return []
    invited: set = set()
    ranked = sorted(entries, key=lambda e: int(e.get("rank") or 999))

    for e in ranked[:40]:
        invited.add(str(e.get("key")))

    for e in ranked[34:58]:
        if _scouting_rng(session, "invite", e.get("key")) < 0.45:
            invited.add(str(e.get("key")))

    for e in entries:
        if str(e.get("risk") or "") == "High" and int(e.get("rank") or 999) <= 80:
            invited.add(str(e.get("key")))
        if int(e.get("stock_delta") or 0) >= 4:
            invited.add(str(e.get("key")))
        if float(e.get("scouting_confidence") or 50) < 50 and int(e.get("rank") or 999) <= 100:
            if _scouting_rng(session, "unc", e.get("key")) < 0.35:
                invited.add(str(e.get("key")))

    goalies = [e for e in entries if str(e.get("position") or "").upper() == "G"]
    goalies.sort(key=lambda e: int(e.get("rank") or 999))
    for e in goalies[:8]:
        invited.add(str(e.get("key")))

    cap = 75 if len(entries) > 120 else min(90, max(55, len(entries) // 2))
    if len(invited) > cap:
        ordered = sorted(invited, key=lambda pid: next(
            (int(e.get("rank") or 999) for e in entries if str(e.get("key")) == pid), 999
        ))
        invited = set(ordered[:cap])
    return sorted(invited, key=lambda pid: next(
        (int(e.get("rank") or 999) for e in entries if str(e.get("key")) == pid), 999
    ))


def _apply_public_combine_adjustments(
    session: FranchiseSession,
    entries: List[Dict[str, Any]],
    combine_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from services.franchise_entry_draft import append_stock_history_snapshot

    adjusted = []
    for e in entries:
        row = dict(e)
        pid = str(row.get("key") or "")
        comb = combine_map.get(pid) or {}
        if not comb.get("combine_invited"):
            adjusted.append(row)
            continue
        delta = int(comb.get("combine_stock_delta") or 0)
        if delta:
            old_rank = int(row.get("rank") or 0)
            new_rank = max(1, old_rank - delta)
            row["rank"] = new_rank
            row["combine_stock_delta"] = delta
            row["combine_stock_reason"] = comb.get("combine_stock_reason")
            row["stock_reason"] = comb.get("combine_stock_reason") or row.get("stock_reason")
            if delta > 0:
                row["stock_label"] = "Riser"
            elif delta < 0:
                row["stock_label"] = "Faller"
            append_stock_history_snapshot(
                session, row, event_source="combine", date_label="Combine"
            )
        adjusted.append(row)

    adjusted.sort(key=lambda x: int(x.get("rank") or 999))
    for i, row in enumerate(adjusted, start=1):
        row["rank"] = i
    return adjusted


def run_franchise_draft_combine(session: FranchiseSession) -> Dict[str, Any]:
    """Run combine once per offseason — invites, testing, CPU impressions, final board prep."""
    if getattr(session, "draft_combine_done", False) and session.draft_combine_payload:
        return dict(session.draft_combine_payload)

    state = _ensure_scouting_state(session)
    profiles = ensure_team_scouting_profiles(session)
    board = get_cached_draft_class_rankings(session, session.sim)
    entries = list(board.get("entries") or [])
    invite_ids = _select_combine_invites(session, entries)
    invite_set = set(invite_ids)

    combine_results: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        pid = str(e.get("key") or "")
        if pid not in invite_set:
            combine_results[pid] = {"prospect_id": pid, "combine_invited": False, "combine_attended": False}
            continue
        combine_results[pid] = _generate_combine_prospect_results(session, e)

    team_impressions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tid in session.team_ids or []:
        team_impressions[str(tid)] = {}
        profile = profiles.get(str(tid)) or {}
        for pid in invite_ids:
            entry = next((x for x in entries if str(x.get("key")) == pid), {})
            team_impressions[str(tid)][pid] = _generate_team_impression(
                session, str(tid), entry, combine_results.get(pid) or {}, profile
            )

    adjusted_entries = _apply_public_combine_adjustments(session, entries, combine_results)
    state["team_impressions"] = team_impressions
    state["combine_results"] = combine_results
    session.scouting_state = state

    invited_rows = []
    for pid in invite_ids:
        entry = next((x for x in adjusted_entries if str(x.get("key")) == pid), {})
        comb = combine_results.get(pid) or {}
        uid = str(session.user_team_id)
        user_imp = (team_impressions.get(uid) or {}).get(pid) or {}
        user_overlay = _prospect_overlay(state, pid)
        scouted = float(user_overlay.get("scouted_percentage") or entry.get("scouting_confidence") or 40)
        row = {**entry, **comb, **user_imp}
        if scouted < 55:
            for field in (
                "skating_test_score", "strength_test_score", "agility_test_score",
                "endurance_score", "interview_score",
            ):
                if field in row and row[field] is not None:
                    val = float(row[field])
                    row[field] = f"{_combine_label(val)} range"
            row["combine_score"] = _combine_label(float(comb.get("combine_score") or 60))
        row["final_board_delta"] = user_imp.get("board_delta")
        row["scout_favorite"] = user_imp.get("scout_favorite") or user_overlay.get("target")
        row["do_not_draft"] = user_imp.get("do_not_draft") or user_overlay.get("do_not_draft")
        invited_rows.append(row)

    top_testers = sorted(
        [combine_results[p] for p in invite_ids if combine_results.get(p, {}).get("combine_score")],
        key=lambda x: -float(x.get("combine_score") or 0),
    )[:8]
    medical_concerns = [
        combine_results[p] for p in invite_ids
        if combine_results.get(p, {}).get("medical_flag")
    ][:8]
    best_interviews = sorted(
        [combine_results[p] for p in invite_ids],
        key=lambda x: -float(x.get("interview_score") or 0),
    )[:6]
    worst_interviews = sorted(
        [combine_results[p] for p in invite_ids if float(combine_results[p].get("interview_score") or 100) < 55],
        key=lambda x: float(x.get("interview_score") or 0),
    )[:4]
    risers = [r for r in invited_rows if int(r.get("combine_stock_delta") or 0) >= 2][:8]
    fallers = [r for r in invited_rows if int(r.get("combine_stock_delta") or 0) <= -2][:8]

    meeting_options = []
    for pid in invite_ids[:12]:
        entry = next((x for x in adjusted_entries if str(x.get("key")) == pid), {})
        meeting_options.append({
            "prospect_id": pid,
            "name": entry.get("name"),
            "position": entry.get("position"),
            "rank": entry.get("rank"),
            "already_met": bool(
                _prospect_overlay(state, pid).get("dinner_status", "").lower().startswith("complet")
            ),
        })

    payload = {
        "completed": True,
        "invite_count": len(invite_ids),
        "invited_prospect_ids": invite_ids,
        "prospects": invited_rows,
        "combine_results": combine_results,
        "top_testers": top_testers,
        "medical_concerns": medical_concerns,
        "best_interviews": best_interviews,
        "worst_interviews": worst_interviews,
        "late_risers": risers,
        "late_fallers": fallers,
        "meeting_options": meeting_options,
        "final_rankings": adjusted_entries[:60],
        "user_team_impressions": team_impressions.get(str(session.user_team_id)) or {},
        "draft_year": int(session.season_calendar_year) + 1,
    }
    session.draft_combine_payload = payload
    session.draft_combine_done = True
    invalidate_session_payload_caches(session, "draft_combine")
    return payload


def apply_combine_user_meeting(
    session: FranchiseSession,
    prospect_id: str,
    meeting_type: str,
) -> Dict[str, Any]:
    """User-team private interview or dinner during combine — stronger board impact."""
    if not getattr(session, "draft_combine_done", False):
        run_franchise_draft_combine(session)

    state = _ensure_scouting_state(session)
    pid = str(prospect_id or "")
    uid = str(session.user_team_id)
    impressions = state.setdefault("team_impressions", {})
    team_row = impressions.setdefault(uid, {})
    imp = dict(team_row.get(pid) or get_team_prospect_impression(session, uid, pid))

    mt = str(meeting_type or "interview").lower()
    bonus = 4.5 if mt == "dinner" else 2.5
    imp["private_meeting_impression"] = "Excellent" if mt == "dinner" else "Positive"
    imp["board_delta"] = round(float(imp.get("board_delta") or 0) + bonus, 2)
    imp["confidence_delta"] = round(float(imp.get("confidence_delta") or 0) + 3.0, 2)
    if mt == "dinner":
        imp["scout_favorite"] = True
        imp["scout_note"] = "Private dinner — GM and scouts aligned on fit."
    imp["private_meeting_summary"] = (
        "Strong character and leadership in private setting."
        if mt == "dinner"
        else "Positive one-on-one interview — compete level stood out."
    )
    team_row[pid] = imp
    state["team_impressions"] = impressions

    overlay_patch: Dict[str, Any] = {}
    if mt == "dinner":
        overlay_patch["dinner_status"] = "Completed"
        overlay_patch["target"] = True
    else:
        overlay_patch["interview_status"] = "Completed"
    _set_prospect_overlay(state, pid, overlay_patch)
    session.scouting_state = state

    if session.draft_combine_payload:
        payload = dict(session.draft_combine_payload)
        for row in payload.get("prospects") or []:
            if str(row.get("prospect_id") or row.get("key")) == pid:
                row.update(imp)
                row["private_meeting_summary"] = imp.get("private_meeting_summary")
        payload["user_team_impressions"] = impressions.get(uid) or {}
        session.draft_combine_payload = payload

    invalidate_session_payload_caches(session, "combine_meeting")
    return {
        "ok": True,
        "impression": imp,
        "draft_combine": session.draft_combine_payload,
    }


def get_draft_combine_payload(session: FranchiseSession) -> Dict[str, Any]:
    if getattr(session, "draft_combine_done", False) and session.draft_combine_payload:
        return dict(session.draft_combine_payload)
    return run_franchise_draft_combine(session)
