# backend/sim_perspective.py
"""
Franchise perspective simulation runner (GM / front-office POV).

Engine responsibility: all hockey logic lives in SimEngine + run_sim.simulate_universe_year pipeline.
Perspective responsibility: team selection, longitudinal framing, relevance filtering, and POV log text.

This module reuses run_sim.run_simulation_core so the simulated universe matches normal runs.
Universe-wide timeline text is discarded to os.devnull; only perspective/ logs are produced for this mode.
Optional `--interactive-draft`: when the user team is on the clock, stdin prompts for pick or `auto` (see build_user_draft_pick_callback).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# run_sim integration (same package root as run_sim.py)
# -----------------------------------------------------------------------------
import run_sim as rs


# =============================================================================
# Output: hide universe timeline (engine still runs); perspective goes to ./perspective/
# =============================================================================


class _DevNullTimelineRunOutput(rs.RunOutput):
    """Keeps RunOutput contract; timeline lines go to the bit bucket (no mixed-in noise)."""

    def open(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._timeline_fp = open(os.devnull, "w", encoding="utf-8")


class _QuietRunnerLogger(rs.RunnerLogger):
    """RunnerLogger without console spam during perspective runs."""

    def emit(self, line: str, level: str = "normal", also_print: bool = False) -> None:  # type: ignore[override]
        super().emit(line, level, also_print=False)


# =============================================================================
# Team resolution
# =============================================================================


def _display_team(t: Any) -> str:
    city = str(getattr(t, "city", "") or "").strip()
    name = str(getattr(t, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return rs._team_name(t)


def resolve_user_team(teams: Sequence[Any], query: str) -> Any:
    """
    Match by team_id (string), city, nickname, or full "City Name" substring.
    Raises ValueError if ambiguous or missing.
    """
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Team query is empty.")
    matches: List[Any] = []
    for t in teams:
        tid = str(rs._team_id(t)).lower()
        disp = _display_team(t).lower()
        nm = str(getattr(t, "name", "") or "").lower()
        ct = str(getattr(t, "city", "") or "").lower()
        if q == tid or q in disp or q in nm or q in ct or q in f"{ct} {nm}".strip():
            matches.append(t)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No team matched {query!r}. Try city or nickname (e.g. Ottawa, Senators).")
    hint = ", ".join(_display_team(x) for x in matches[:6])
    raise ValueError(f"Ambiguous team {query!r}; matches include: {hint}")


# =============================================================================
# User-controlled draft (perspective mode only; engine hook via setattr)
# Replace INTERACTIVE_DRAFT_INPUT with a UI/API callable when wiring a frontend.
# =============================================================================

INTERACTIVE_DRAFT_INPUT: Callable[[str], str] = input


def _prospect_ceiling_mid(p: Any) -> float:
    dr = getattr(p, "draft_value_range", None)
    if isinstance(dr, (list, tuple)) and len(dr) >= 2:
        try:
            return (float(dr[0]) + float(dr[1])) / 2.0
        except (TypeError, ValueError):
            pass
    return 0.55


def _prospect_potential_phrase(mid: float) -> str:
    if mid >= 0.82:
        return "Elite potential"
    if mid >= 0.74:
        return "High ceiling"
    if mid >= 0.68:
        return "Top 6 / Top 4 upside"
    if mid >= 0.60:
        return "Top 9 / depth upside"
    return "Project / longshot"


def _prospect_display_ovr_int(p: Any) -> int:
    return int(rs.clamp(_prospect_ceiling_mid(p) * 99.0, 40.0, 99.0))


def _prospect_pos_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    if ident is None:
        return "?"
    pv = getattr(ident, "position", "?")
    return str(getattr(pv, "value", pv) if pv is not None else "?")


def _prospect_name_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    if ident is not None:
        nm = getattr(ident, "name", None)
        if nm:
            return str(nm)
    return str(getattr(p, "name", "Unknown"))


def build_user_draft_pick_callback(
    *,
    display_name: str,
    mem: "FranchisePerspectiveMemory",
    input_fn: Callable[[str], str] = INTERACTIVE_DRAFT_INPUT,
) -> Callable[..., Optional[str]]:
    """
    Returns a callback for SimEngine.user_draft_pick_callback.
    Prospect id or None (None => engine runs AI pick).
    """

    def _cb(
        engine: Any,
        team_id: str,
        pick_number: int,
        board: Any,
        drafted_ids: Any,
        league_events: Any,
        prospect_by_id: Dict[str, Any],
        ai_pick_fn: Callable[[], Tuple[Optional[str], Dict[str, Any]]],
    ) -> Optional[str]:
        rows: List[Tuple[str, float, Any]] = []
        for pid, p in prospect_by_id.items():
            pids = str(pid).strip()
            if not pids or pids in drafted_ids:
                continue
            rows.append((pids, _prospect_ceiling_mid(p), p))
        if not rows:
            mem.pending_draft_log.append(
                f"[USER DECISION REQUIRED] Pick #{pick_number} — no eligible prospects left in pool; delegating to AI.\n"
            )
            cid, _ = ai_pick_fn()
            return cid

        rows.sort(key=lambda x: -x[1])
        top = rows[:10]
        banner = (
            "\n==============================\n"
            "[USER DECISION REQUIRED]\n"
            "You are on the clock.\n"
            f"Team: {display_name}\n"
            f"Pick: #{pick_number}\n"
            "==============================\n"
            "Top available prospects:\n"
        )
        print(banner, flush=True)
        mem.pending_draft_log.append(banner)

        for i, (pids, mid, p) in enumerate(top):
            line = (
                f"  {i + 1}. {_prospect_name_str(p)} – {_prospect_pos_str(p)} – "
                f"{_prospect_display_ovr_int(p)} OVR (ceiling proxy) – {_prospect_potential_phrase(mid)}\n"
            )
            print(line.rstrip(), flush=True)
            mem.pending_draft_log.append(line)

        tail = (
            "\nEnter selection (1–10), or type 'auto' for AI.\n"
        )
        print(tail, flush=True)
        mem.pending_draft_log.append(tail)

        idx_by_display = {i + 1: top[i][0] for i in range(len(top))}

        while True:
            try:
                choice = input_fn("Your pick: ").strip()
            except (EOFError, KeyboardInterrupt, Exception):
                cid, _ = ai_pick_fn()
                cname = "?"
                if cid and cid in prospect_by_id:
                    cname = _prospect_name_str(prospect_by_id[cid])
                mem.pending_draft_log.append(
                    f"[USER PICK] {display_name} — input unavailable; AI selected {cname} at #{pick_number}\n"
                )
                return cid

            low = choice.lower()
            if low == "auto":
                cid, _ = ai_pick_fn()
                cname = "?"
                if cid and cid in prospect_by_id:
                    cname = _prospect_name_str(prospect_by_id[cid])
                mem.pending_draft_log.append(
                    f"[USER PICK] {display_name} delegated pick #{pick_number} to AI — {cname}\n"
                )
                return cid

            if choice.isdigit():
                idx = int(choice)
                pid_sel = idx_by_display.get(idx)
                if pid_sel:
                    p_obj = prospect_by_id.get(pid_sel)
                    if p_obj is None or pid_sel in drafted_ids:
                        mem.pending_draft_log.append(
                            f"[USER PICK] Invalid selection #{idx} (pool drift); AI will pick for {display_name} at #{pick_number}.\n"
                        )
                        cid, _ = ai_pick_fn()
                        return cid
                    nm = _prospect_name_str(p_obj)
                    log_ln = f"[USER PICK] {display_name} selected {nm} at #{pick_number}\n"
                    print(log_ln, flush=True)
                    mem.pending_draft_log.append(log_ln)
                    return pid_sel

            msg = "Invalid input — enter a listed number (1–10) or 'auto'.\n"
            print(msg, flush=True)
            mem.pending_draft_log.append(msg)

    return _cb


def clear_user_draft_hooks(sim: Any) -> None:
    if sim is None:
        return
    try:
        sim.user_draft_pick_callback = None
        sim.user_draft_team_id = None
    except Exception:
        pass


def attach_user_draft_hooks(sim: Any, *, user_team: Any, mem: "FranchisePerspectiveMemory", enabled: bool) -> None:
    clear_user_draft_hooks(sim)
    if not enabled or sim is None:
        return
    try:
        disp = _display_team(user_team)
        sim.user_draft_team_id = str(rs._team_id(user_team))
        sim.user_draft_pick_callback = build_user_draft_pick_callback(display_name=disp, mem=mem)
    except Exception:
        clear_user_draft_hooks(sim)


# =============================================================================
# Longitudinal franchise memory (perspective layer only — does not affect sim RNG)
# =============================================================================


def _new_team_memory() -> Dict[str, Any]:
    """Perspective-only longitudinal state (does not touch engine)."""
    return {
        "missed_playoffs_streak": 0,
        "years_since_contention": 0,
        "last_strategy": None,
        "core_age_trend": [],
        "years_stuck_middle": 0,
        "rebuild_seasons": 0,
        "points_history": [],
        "bucket_history": [],
        "failed_strategy_count": 0,
        "roster_insight_history": [],
    }


@dataclass
class FranchisePerspectiveMemory:
    consecutive_playoff_misses: int = 0
    consecutive_playoff_appearances: int = 0
    years_since_deep_run: int = 0  # proxy: finals appearance not tracked per team here
    last_points: Optional[int] = None
    last_bucket: Optional[str] = None
    last_season_finish_rank: Optional[int] = None
    prior_year_summary: str = ""
    team_memory: Dict[str, Any] = field(default_factory=_new_team_memory)
    # Buffered during run_simulation_core draft phase; flushed in format_year_block (narrative order).
    pending_draft_log: List[str] = field(default_factory=list)

    def record_season_end(
        self,
        *,
        made_playoffs: bool,
        points: int,
        bucket: str,
        rank: int,
        summary_line: str,
    ) -> None:
        if made_playoffs:
            self.consecutive_playoff_misses = 0
            self.consecutive_playoff_appearances += 1
        else:
            self.consecutive_playoff_misses += 1
            self.consecutive_playoff_appearances = 0
        self.last_points = points
        self.last_bucket = bucket
        self.last_season_finish_rank = rank
        self.prior_year_summary = summary_line


# =============================================================================
# Structured reads (defensive)
# =============================================================================


def _player_ovr_norm(p: Any) -> float:
    f = getattr(p, "ovr", None)
    try:
        ov = float(f() if callable(f) else f)
    except Exception:
        return 0.5
    return ov / 99.0 if ov > 1.5 else float(ov)


def _roster_players(team: Any) -> List[Any]:
    return [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]


def roster_structure_snapshot(team: Any) -> Dict[str, Any]:
    players = _roster_players(team)
    if not players:
        return {"n": 0, "mean_ovr": None, "top3_mean": None, "by_pos": {}, "u24": 0, "prime": 0, "vet": 0}
    ovrs = sorted((_player_ovr_norm(p) for p in players), reverse=True)
    by_pos: Dict[str, int] = {}
    u24 = prime = vet = 0
    for p in players:
        ident = getattr(p, "identity", None)
        age = int(getattr(ident, "age", 0) or 0) if ident is not None else 0
        if age < 24:
            u24 += 1
        elif age <= 29:
            prime += 1
        else:
            vet += 1
        pos = "?"
        if ident is not None:
            pv = getattr(ident, "position", "?")
            pos = str(getattr(pv, "value", pv) if pv is not None else "?")
        by_pos[pos] = by_pos.get(pos, 0) + 1
    top3 = ovrs[: min(3, len(ovrs))]
    return {
        "n": len(players),
        "mean_ovr": sum(ovrs) / len(ovrs),
        "top3_mean": sum(top3) / len(top3) if top3 else None,
        "by_pos": by_pos,
        "u24": u24,
        "prime": prime,
        "vet": vet,
    }


def prospect_pipeline_snapshot(team: Any, limit: int = 8) -> Dict[str, Any]:
    pool = list(getattr(team, "prospect_pool", None) or [])
    scored: List[Tuple[float, Any]] = []
    for pr in pool:
        dr = getattr(pr, "draft_value_range", None)
        if isinstance(dr, (list, tuple)) and len(dr) >= 2:
            mid = (float(dr[0]) + float(dr[1])) / 2.0
        else:
            mid = 0.55
        scored.append((mid, pr))
    scored.sort(key=lambda x: -x[0])
    top: List[str] = []
    for mid, pr in scored[:limit]:
        ident = getattr(pr, "identity", None)
        nm = (getattr(ident, "name", None) if ident is not None else None) or getattr(
            pr, "name", None
        ) or getattr(pr, "prospect_name", None) or "Prospect"
        y = int(getattr(pr, "development_years_remaining", -1) or -1)
        arch = str(getattr(pr, "_dev_archetype", "") or "") or "unset"
        top.append(f"{nm}  ceiling~{mid:.2f}  dev_yrs_left={y}  arch={arch}")
    return {"count": len(pool), "top_lines": top, "thin": len(pool) < 6}


def _standing_for_team(standings: Sequence[Any], tid: str) -> Optional[Any]:
    for s in standings:
        if str(getattr(s, "team_id", "")) == str(tid):
            return s
    return None


def _teams_by_id(teams: Sequence[Any]) -> Dict[str, Any]:
    return {str(rs._team_id(t)): t for t in teams}


def _playoff_cut(n_teams: int) -> int:
    return 16 if n_teams >= 16 else max(1, n_teams // 2)


def event_relevance_score(
    ev: rs.UniverseEvent,
    *,
    user_tid: str,
    user_div: str,
    user_conf: str,
    teams_by_id: Dict[str, Any],
    standings: Sequence[Any],
) -> float:
    """Heuristic 0–100: what matters to our boardroom."""
    et = (getattr(ev, "type", "") or "").upper()
    teams_hit = [str(x) for x in (getattr(ev, "teams", None) or [])]
    if user_tid in teams_hit:
        return 100.0
    score = 0.0
    bucket_rank: Dict[str, float] = {}
    for s in standings:
        bucket_rank[str(s.team_id)] = 1.0 if s.bucket == "contender" else 0.65 if s.bucket == "playoff" else 0.35

    for otid in teams_hit:
        ot = teams_by_id.get(otid)
        if ot is None:
            continue
        odiv = str(getattr(ot, "division", "") or "")
        oconf = str(getattr(ot, "conference", "") or "")
        br = bucket_rank.get(otid, 0.4)
        if odiv and odiv == user_div:
            score = max(score, 55.0 + 15.0 * br)
        elif oconf and oconf == user_conf:
            score = max(score, 38.0 + 12.0 * br)
        else:
            score = max(score, 15.0 + 8.0 * br)

    if et in ("TRADE", "SIGNING") and score < 20.0:
        score = 18.0
    if et == "ERA_SHIFT":
        score = max(score, 28.0)
    if et == "NOTE" and "lottery" in (getattr(ev, "headline", "") or "").lower():
        score = max(score, 40.0)
    return min(100.0, score + float(getattr(ev, "impact_score", 0.0) or 0.0) * 12.0)


def cap_pressure_phrase(team: Any, ustate: rs.UniverseState) -> str:
    """Derive phrasing from runner/econ hooks when present."""
    for attr in ("cap_pressure_band", "last_cap_pressure", "_cap_pressure"):
        v = getattr(team, attr, None)
        if isinstance(v, str) and v:
            return v
    try:
        usage = float(getattr(team, "cap_usage_pct", 0.0) or 0.0)
        if usage >= 0.92:
            return "high (usage at/near ceiling)"
        if usage >= 0.78:
            return "moderate"
        if usage > 0:
            return "comfortable"
    except Exception:
        pass
    return f"league cap ~${float(ustate.salary_cap_m):.1f}M — team-level usage unavailable"


def coach_snapshot(team: Any) -> str:
    ch = getattr(team, "coach", None)
    if ch is None:
        return "No coach object on team."
    for attr in ("overall_rating", "rating", "quality"):
        v = getattr(ch, attr, None)
        if isinstance(v, (int, float)):
            return f"Coach rating proxy ~{int(v)}"
    nm = getattr(ch, "name", None)
    return f"Coach present ({nm or 'unnamed'}); rating fields not exposed"


def org_profiles_snapshot(team: Any) -> Tuple[str, str, str]:
    mp = getattr(team, "market_profile", None)
    op = getattr(team, "ownership_profile", None)
    ts = getattr(team, "team_state", None)
    market = "unknown"
    own = "unknown"
    org = "unknown"
    if mp is not None:
        try:
            market = (
                f"media_pressure~{float(getattr(mp, 'media_pressure', 0.5)):.2f} "
                f"fan_exp~{float(getattr(mp, 'fan_expectations', 0.5)):.2f}"
            )
        except Exception:
            market = str(mp)
    if op is not None:
        try:
            own = (
                f"patience~{float(getattr(op, 'patience', 0.5)):.2f} "
                f"ambition~{float(getattr(op, 'ambition', 0.5)):.2f}"
            )
        except Exception:
            own = str(op)
    if ts is not None:
        try:
            org = (
                f"pressure~{float(getattr(ts, 'organizational_pressure', 0.5)):.2f} "
                f"morale~{float(getattr(ts, 'team_morale', 0.5)):.2f} "
                f"status={getattr(ts, 'status', '?')}"
            )
        except Exception:
            org = str(ts)
    return market, own, org


def window_and_posture(archetype: str, bucket: str, mem: FranchisePerspectiveMemory) -> Tuple[str, str]:
    arch = (archetype or "balanced").lower()
    if arch in ("rebuild", "draft_and_develop"):
        window = "Development / accumulation phase"
        posture = "Patience unless clear playoff leverage appears."
    elif arch in ("win_now", "chaos_agent"):
        window = "Short-horizon competitive push"
        posture = "Aggressive repair mode if slide persists."
    elif arch == "contender":
        window = "Open contention window"
        posture = "Selective buying if injury or matchup gap emerges."
    else:
        window = "Balanced / opportunistic"
        posture = "Hold flexibility unless division arms race forces response."

    if mem.consecutive_playoff_misses >= 3:
        posture += f" Internal heat elevated after {mem.consecutive_playoff_misses} missed postseasons."
    elif mem.consecutive_playoff_appearances >= 3:
        posture += " Sustained playoff runs raise expectation management risk."
    if bucket == "rebuild" and arch in ("win_now", "contender"):
        posture += " Tension: identity says push but results sit in lottery band."
    return window, posture


def narrative_season_beat(
    *,
    bucket: str,
    rank: int,
    n_teams: int,
    pts_delta: Optional[int],
    made_playoffs: bool,
) -> str:
    """Single paragraph internal beat — grounded, not melodrama."""
    cut = _playoff_cut(n_teams)
    parts: List[str] = []
    if pts_delta is None:
        parts.append("Season baseline established for the org chart.")
    elif pts_delta >= 8:
        parts.append("Standings surge versus prior year — internal narrative shifts toward validation of the core.")
    elif pts_delta <= -8:
        parts.append("Material points regression — depth and special teams questions surface in scouting meetings.")
    else:
        parts.append("Year-to-year points mostly flat — story becomes process and injury luck, not raw climb.")
    if made_playoffs:
        parts.append(f"Postseason berth secured (finish ~{rank}/{n_teams}) — evaluation mode switches to matchup viability.")
    else:
        parts.append(f"Missed playoffs (finish ~{rank}/{n_teams}) — lottery positioning and trade ethics enter the agenda.")
    if bucket == "contender" and not made_playoffs:
        parts.append("Underperformance vs roster expectation; leadership review pressure is real.")
    return " ".join(parts)


# =============================================================================
# Front-office intelligence (perspective-only; heuristic / synthetic where data thin)
# =============================================================================

INSIGHT_TEMPLATES: Dict[str, List[str]] = {
    "scoring_issue": [
        "Secondary scoring is unreliable when the top line is checked hard.",
        "Offense collapses beyond the first unit — depth forwards aren't driving play.",
        "Production is overly concentrated; special teams may be propping up totals.",
    ],
    "defense_issue": [
        "Blue line lacks a true shutdown presence in tough minutes.",
        "Defensive-zone coverage is inconsistent shift-to-shift.",
        "Top pair carries an outsized burden; third pair is a target for matchups.",
    ],
    "depth_issue": [
        "Middle six lacks drivers — wins feel borrowed rather than earned.",
        "Bottom six is defensively safe but offers little counterpunch.",
        "Forward group is deep in names but thin in impact past the top nine.",
    ],
    "age_issue": [
        "Age cluster on the back end raises succession risk within two seasons.",
        "Core is exiting prime; win-now trades carry asymmetric downside.",
        "Youth exists but hasn't displaced expensive declining minutes yet.",
    ],
    "goalie_issue": [
        "Team defensive metrics lean hard on one netminder — injury risk is systemic.",
        "Tandem quality drops sharply after the starter — workload management is critical.",
    ],
    "balanced": [
        "Roster shape is serviceable without a single fatal flaw — execution and health matter most.",
        "No screaming hole on paper; separation comes from coaching and special teams.",
    ],
}


def _deterministic_unit_float(key: str) -> float:
    """0..1 stable pseudo-random from key (reproducible; avoids randomized built-in hash())."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    v = int.from_bytes(digest[:4], "big", signed=False)
    return (v % 10_000) / 10_000.0


def compute_team_strength_profile(
    team: Any,
    diagnosis: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    """
    Balanced 0–1 vector — no single input dominates (each axis blends several signals).
    """
    pts = int(metrics.get("pts") or 82)
    rank = int(metrics.get("rank") or 16)
    n_teams = max(1, int(metrics.get("n_teams") or 32))
    made_playoffs = bool(metrics.get("made_playoffs"))
    won_cup = bool(metrics.get("won_cup"))
    pts_pct = rs.clamp(pts / 164.0, 0.0, 1.0)
    rank_pct = rs.clamp(1.0 - (rank - 1) / max(1, n_teams - 1), 0.0, 1.0)
    lg = float(metrics.get("league_avg_ovr") or 0.72)
    mean_ovr = float(metrics.get("mean_ovr") or 0.72)
    top3 = float(metrics.get("top3_mean") or mean_ovr)

    s_pts = pts_pct
    s_rank = rank_pct
    s_roster = rs.clamp(mean_ovr / max(0.55, lg), 0.0, 1.12) / 1.12
    s_elite = rs.clamp((top3 - 0.55) / 0.38, 0.0, 1.0)
    s_playoff = 1.0 if won_cup else (0.88 if made_playoffs else 0.42 + 0.38 * pts_pct)
    strength_score = (s_pts + s_rank + s_roster + s_elite + s_playoff) / 5.0

    depth = float(diagnosis.get("depth_score") or 0.5)
    star = float(diagnosis.get("star_dependency_index") or 0.4)
    grel = float(diagnosis.get("goalie_reliance_factor") or 0.35)
    si = float(diagnosis.get("structural_integrity_score") or 0.5)
    w_depth = 1.0 - rs.clamp(depth, 0.0, 1.0)
    w_star = rs.clamp(star / 1.15, 0.0, 1.0)
    w_goalie = rs.clamp(grel * 0.9, 0.0, 1.0)
    struct_bump = 0.07 if si < 0.30 else (0.035 if si < 0.45 else 0.0)
    weakness_score = rs.clamp((w_depth + w_star + w_goalie) / 3.0 + struct_bump, 0.0, 1.0)

    age_read = str(diagnosis.get("age_curve_position") or "")
    if "ASCENDING" in age_read:
        age_stab = 0.72
    elif "DECLINING" in age_read:
        age_stab = 0.42
    else:
        age_stab = 0.58
    cap_eff = str(metrics.get("cap_efficiency") or "SOLID").upper()
    if cap_eff in ("STRONG FLEX", "MANAGEABLE"):
        cap_s = 0.74
    elif cap_eff in ("SOLID",):
        cap_s = 0.62
    elif cap_eff in ("TIGHT",):
        cap_s = 0.48
    elif cap_eff in ("CRAMPED", "POOR"):
        cap_s = 0.36
    else:
        cap_s = 0.55
    stability_score = rs.clamp((si * 0.45 + age_stab * 0.30 + cap_s * 0.25), 0.0, 1.0)

    ph: List[int] = list(metrics.get("points_history") or [])
    momentum = 0.5
    if len(ph) >= 3:
        if ph[-1] >= ph[-2] and ph[-2] >= ph[-3] - 2:
            momentum = 0.72
        elif ph[-1] < ph[-2] - 5 and ph[-2] < ph[-3] - 2:
            momentum = 0.28
    elif len(ph) >= 2:
        momentum = 0.62 if ph[-1] > ph[-2] + 4 else (0.38 if ph[-1] < ph[-2] - 4 else 0.5)
    pipe = str(metrics.get("pipeline_strength") or "MEDIUM")
    if pipe == "HIGH":
        pipe_t = 0.84
    elif pipe == "MEDIUM":
        pipe_t = 0.62
    else:
        pipe_t = 0.38
    trajectory_score = rs.clamp(0.55 * momentum + 0.45 * pipe_t + 0.05 * (pts_pct - 0.5), 0.0, 1.0)

    return {
        "strength_score": round(float(strength_score), 3),
        "weakness_score": round(float(weakness_score), 3),
        "stability_score": round(float(stability_score), 3),
        "trajectory_score": round(float(trajectory_score), 3),
    }


def classify_flaw_severity(issues: Dict[str, Any]) -> str:
    """How much roster/standings flaws should sway the grid (MINOR < MODERATE < CRITICAL)."""
    score = 0
    if issues.get("high_star_dep"):
        score += 1
    if issues.get("low_depth"):
        score += 1
    if issues.get("high_goalie_rel"):
        score += 1
    if issues.get("struct_low"):
        score += 1
    if issues.get("declining_core"):
        score += 2
    if issues.get("playoff_miss_streak") and int(issues.get("playoff_miss_streak") or 0) >= 4:
        score += 2
    if score >= 6:
        return "CRITICAL"
    if score >= 3:
        return "MODERATE"
    return "MINOR"


def compute_pipeline_impact_score(
    scored_prospects: List[Tuple[float, Any]],
    team_needs: List[str],
    weakness_codes: List[str],
) -> float:
    """0–1 how much the pipeline should calm or sharpen external urgency."""
    if not scored_prospects:
        return 0.15
    hi = float(scored_prospects[0][0])
    pr0 = scored_prospects[0][1]
    y = int(getattr(pr0, "development_years_remaining", 3) or 3)
    ident = getattr(pr0, "identity", None)
    pos = "?"
    if ident is not None:
        pv = getattr(ident, "position", "?")
        pos = str(getattr(pv, "value", pv) or "?")

    def _matches() -> bool:
        p = pos.upper()
        if "defense_issue" in weakness_codes and p == "D":
            return hi >= 0.62
        if "scoring_issue" in weakness_codes and p in ("LW", "RW", "C"):
            return hi >= 0.62
        if "goalie_issue" in weakness_codes and p == "G":
            return hi >= 0.58
        if "depth_issue" in weakness_codes:
            return hi >= 0.66
        return hi >= 0.74

    align = 0.35 + (0.35 if _matches() else 0.0)
    quality = rs.clamp((hi - 0.52) / 0.38, 0.0, 1.0)
    eta = rs.clamp((4 - min(4, max(1, y))) / 3.0, 0.0, 1.0)
    depth_bonus = min(0.12, 0.03 * min(12, len(scored_prospects)))
    return round(rs.clamp(0.25 * align + 0.40 * quality + 0.30 * eta + depth_bonus, 0.0, 1.0), 3)


def _strategy_tier_index(code: str) -> int:
    order = ("HARD_REBUILD", "SELL", "RETOOL", "HOLD", "BUY")
    try:
        return order.index(code)
    except ValueError:
        return 3


def _strategy_from_tier_index(idx: int) -> str:
    order = ("HARD_REBUILD", "SELL", "RETOOL", "HOLD", "BUY")
    return order[rs.clamp(idx, 0, len(order) - 1)]


def apply_pipeline_strategy_shift(strategy: str, pipeline_impact: float) -> Tuple[str, List[str]]:
    """High pipeline impact reduces aggression; low impact increases it (one tier)."""
    notes: List[str] = []
    idx = _strategy_tier_index(strategy)
    if pipeline_impact >= 0.68:
        idx = max(0, idx - 1)
        notes.append("Pipeline impact high — reduced external urgency (patience justified).")
    elif pipeline_impact <= 0.32:
        idx = min(4, idx + 1)
        notes.append("Pipeline impact low — slightly higher urgency to address holes.")
    return _strategy_from_tier_index(idx), notes


def apply_gm_patience(
    strategy: str,
    trajectory_score: float,
    recent_success: bool,
    trajectory_trend: str,
) -> Tuple[str, List[str]]:
    notes: List[str] = []
    s = strategy
    if recent_success and s == "HARD_REBUILD":
        s = "SELL"
        notes.append("GM patience: recent playoff team — no full strip posture.")
    if trajectory_trend == "up" or trajectory_score >= 0.62:
        if s == "HARD_REBUILD":
            s = "SELL"
            notes.append("GM patience: upward trajectory — prefer measured teardown.")
        elif s == "SELL":
            s = "RETOOL"
            notes.append("GM patience: momentum — sell softened to surgical retool.")
        elif s == "RETOOL":
            s = "HOLD"
            notes.append("GM patience: improving club — hold and evaluate before bigger moves.")
    if trajectory_score >= 0.58 and trajectory_trend == "up" and s in ("SELL", "RETOOL"):
        if s == "SELL":
            s = "RETOOL"
            notes.append("GM patience: bias toward consolidation over fire sale.")
    return s, notes


def final_decision_validator(
    profile: str,
    strategy: str,
    diagnosis: Dict[str, Any],
    profile_vec: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Strong / upward / contender-family teams must not get panic outcomes."""
    notes: List[str] = []
    s = strategy
    ss = float(profile_vec.get("strength_score") or 0.5)
    ws = float(profile_vec.get("weakness_score") or 0.5)
    ts = float(profile_vec.get("trajectory_score") or 0.5)

    def bump(msg: str) -> None:
        notes.append(f"FINAL VALIDATION: {msg}")

    if ss > ws + 0.08 and s in ("HARD_REBUILD", "SELL"):
        s = "HOLD" if ss >= 0.58 else "RETOOL"
        bump("Net strength outweighs flaws — strategy softened from panic lane.")
    if ts >= 0.60 and s in ("HARD_REBUILD", "SELL"):
        s = "RETOOL" if s == "HARD_REBUILD" else "HOLD"
        bump("Trajectory healthy — no panic strip.")
    fam = profile.upper()
    if fam.startswith("CONTENDER (STRONG)") or profile == "TRUE CONTENDER":
        if s in ("SELL", "HARD_REBUILD"):
            s = "HOLD"
            bump("Strong contender profile — sell/strip disallowed.")
    return s, notes


def _balanced_issue_risk(
    profile: str,
    diagnosis: Dict[str, Any],
    profile_vec: Dict[str, float],
) -> Tuple[str, str]:
    """Strength + weakness + implication (less binary than 'structurally flawed')."""
    ss = float(profile_vec.get("strength_score") or 0.5)
    ws = float(profile_vec.get("weakness_score") or 0.5)
    depth = float(diagnosis.get("depth_score") or 0.5)
    star = float(diagnosis.get("star_dependency_index") or 0.4)
    base_issue = str(diagnosis.get("issue") or "")
    base_risk = str(diagnosis.get("risk") or "")
    strength_bit = (
        "The club shows competitive strengths in the standings and top-end talent, "
        if ss >= 0.55
        else "Results are mixed relative to talent, "
    )
    if ws > ss and depth < 0.46:
        weak_bit = "but underlying depth and support scoring remain concerns worth monitoring."
    elif ws <= ss + 0.05:
        weak_bit = "while roster balance is workable if health and special teams hold."
    else:
        weak_bit = "with some structural pressure if stars miss time or depth thins."
    if star > 0.52 and depth >= 0.42:
        weak_bit = (
            "though the game plan leans heavily on a few drivers — injury or cold streaks swing outcomes."
        )
    impl = " Implication: stay patient unless flaws worsen or the window clearly closes."
    if profile.upper().startswith("REBUILD"):
        impl = " Implication: protect futures while converting young NHL minutes into clearer answers."
    if profile == "EMERGING TEAM":
        impl = " Implication: avoid win-now shortcuts that mortgage the age curve."
    issue = f"{strength_bit}{weak_bit}{impl}"
    if len(issue) > 320:
        issue = base_issue
    risk = base_risk
    if ws > ss + 0.12:
        risk = f"{base_risk} Weighted read: downside scenarios cluster on depth and regression, not a single star."
    return issue, risk


def calculate_structural_integrity(
    depth_score: float,
    star_dependency_index: float,
    goalie_reliance_factor: float,
    u24: int,
    vet: int,
    roster_n: int,
) -> Dict[str, Any]:
    """
    Composite 0..1 score → LOW / MODERATE / STABLE / ELITE (avoids everything reading LOW).
    """
    age_balance = 1.0 - rs.clamp(abs(u24 - vet) / max(8, roster_n * 0.5), 0.0, 1.0) * 0.35
    depth_component = rs.clamp(depth_score, 0.0, 1.0) * 0.34
    star_component = rs.clamp(1.0 - star_dependency_index / 1.2, 0.0, 1.0) * 0.18
    goalie_component = rs.clamp(1.0 - goalie_reliance_factor * 0.85, 0.0, 1.0) * 0.13
    score = rs.clamp(depth_component + star_component + goalie_component + age_balance * 0.2, 0.0, 1.0)
    if score >= 0.80:
        label = "ELITE"
    elif score >= 0.60:
        label = "STABLE"
    elif score >= 0.30:
        label = "MODERATE"
    else:
        label = "LOW"
    return {"score": round(score, 3), "label": label}


def _mean_ovr_nhl99(mean_norm: float) -> float:
    """Proxy NHL-style OVR (0–99) from normalized 0–1 engine OVR."""
    return rs.clamp(mean_norm * 99.0, 50.0, 99.0)


def _league_avg_roster_ovr(league: Any) -> float:
    teams = rs._get_league_teams(league)
    vals: List[float] = []
    for t in teams:
        snap = roster_structure_snapshot(t)
        m = snap.get("mean_ovr")
        if m is not None and snap.get("n", 0) > 0:
            vals.append(float(m))
    return sum(vals) / len(vals) if vals else 0.72


def _player_age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    if ident is not None:
        try:
            return int(getattr(ident, "age", 0) or 0)
        except (TypeError, ValueError):
            pass
    return 26


def _player_position_key(p: Any) -> str:
    ident = getattr(p, "identity", None)
    if ident is None:
        return "?"
    pv = getattr(ident, "position", "?")
    return str(getattr(pv, "value", pv) if pv is not None else "?")


def diagnose_team(team_data: Dict[str, Any], league_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert roster + standings signals into a hockey-meaningful profile.
    team_data: roster_structure_snapshot output + extras (pts, rank, n_teams, bucket, made_playoffs, players list)
    league_context: league_avg_ovr, salary_cap_m
    """
    n = int(team_data.get("n") or 0)
    mean_ovr = float(team_data.get("mean_ovr") or 0.72)
    top3 = float(team_data.get("top3_mean") or mean_ovr)
    pts = int(team_data.get("pts") or 82)
    rank = int(team_data.get("rank") or 16)
    n_teams = max(1, int(team_data.get("n_teams") or 32))
    bucket = str(team_data.get("bucket") or "bubble")
    made_playoffs = bool(team_data.get("made_playoffs"))
    u24 = int(team_data.get("u24") or 0)
    vet = int(team_data.get("vet") or 0)
    players: List[Any] = list(team_data.get("players") or [])

    lg = float(league_context.get("league_avg_ovr") or 0.72)
    pts_pct = rs.clamp(pts / 164.0, 0.0, 1.0)
    rank_pct = rs.clamp(1.0 - (rank - 1) / max(1, n_teams - 1), 0.0, 1.0)

    ovrs = sorted((_player_ovr_norm(p) for p in players), reverse=True) if players else [mean_ovr]
    rest_mean = sum(ovrs[3:]) / max(1, len(ovrs) - 3) if len(ovrs) > 3 else mean_ovr
    depth_gap = max(0.0, top3 - rest_mean)
    depth_score = rs.clamp(1.0 - depth_gap * 4.0, 0.0, 1.0) * rs.clamp(rest_mean / max(0.55, top3), 0.0, 1.1)

    star_dep = rs.clamp((top3 - mean_ovr) / max(0.08, 1.0 - mean_ovr + 0.01), 0.0, 1.2)

    g_ovrs = [_player_ovr_norm(p) for p in players if _player_position_key(p) == "G"]
    sk_ovrs = [_player_ovr_norm(p) for p in players if _player_position_key(p) != "G"]
    g_rel = 0.35
    if g_ovrs and sk_ovrs:
        g_rel = rs.clamp((max(g_ovrs) - (sum(sk_ovrs) / len(sk_ovrs))) * 2.0 + 0.35, 0.0, 1.0)

    top5_ages = sorted([_player_age(p) for p in sorted(players, key=_player_ovr_norm, reverse=True)[:5]])
    mean_top5_age = sum(top5_ages) / max(1, len(top5_ages))
    if mean_top5_age <= 25.5 and u24 >= 6:
        age_curve = "ASCENDING (young core carries top minutes)"
    elif mean_top5_age >= 30.5 or vet >= 8:
        age_curve = "DECLINING (top minutes concentrated in older players)"
    else:
        age_curve = "MIXED / PLATEAU"

    contender_score = rs.clamp(
        0.38 * pts_pct + 0.28 * rank_pct + 0.22 * (mean_ovr / max(0.55, lg)) + 0.12 * depth_score,
        0.0,
        1.0,
    )
    if bucket in ("contender", "playoff") and made_playoffs:
        contender_score = min(1.0, contender_score + 0.06)
    if bucket == "rebuild" and pts_pct < 0.48:
        contender_score *= 0.85

    si_pack = calculate_structural_integrity(
        float(depth_score), float(star_dep), float(g_rel), u24, vet, max(1, n)
    )
    struct_integrity = si_pack["label"]
    struct_integrity_score = float(si_pack["score"])

    phist = list(team_data.get("points_history") or [])
    won_cup_flag = bool(team_data.get("won_cup"))
    _pre_stub = {
        "depth_score": float(depth_score),
        "star_dependency_index": float(star_dep),
        "goalie_reliance_factor": float(g_rel),
        "structural_integrity_score": struct_integrity_score,
        "age_curve_position": age_curve,
    }
    _pre_metrics: Dict[str, Any] = {
        "pts": pts,
        "rank": rank,
        "n_teams": n_teams,
        "made_playoffs": made_playoffs,
        "won_cup": won_cup_flag,
        "league_avg_ovr": lg,
        "mean_ovr": mean_ovr,
        "top3_mean": top3,
        "points_history": phist,
        "pipeline_strength": "MEDIUM",
        "cap_efficiency": "SOLID",
    }
    _bal0 = compute_team_strength_profile(None, _pre_stub, _pre_metrics)
    s0, w0 = float(_bal0["strength_score"]), float(_bal0["weakness_score"])

    high_pts_weak = pts >= 98 and (depth_score < 0.42 or star_dep > 0.48)
    low_pts_young = pts <= 86 and u24 >= 7 and mean_top5_age <= 26
    mid_pts = 88 <= pts <= 100
    false_contender_raw = high_pts_weak or (
        pts >= 96 and (struct_integrity == "LOW" or struct_integrity_score < 0.35)
    )
    weakness_dominates = w0 >= s0 - 0.02

    pts_trend_up = len(phist) >= 2 and phist[-1] > phist[-2] + 3

    profile = "BUBBLE (STALLED)"
    issue = "Standings and roster signals sit in the mushy middle."
    risk = "Small injuries or shooting luck swing playoff odds hard."
    if won_cup_flag:
        profile = "CONTENDER (STRONG)"
        issue = "Cup-caliber outcome; sustain depth and cap sequencing."
        risk = "Complacency and contract inflation."
    elif pts_pct >= 0.58 and depth_score >= 0.5 and star_dep <= 0.42 and made_playoffs:
        profile = "CONTENDER (STRONG)"
        issue = "High floor from depth and star support."
        risk = "Injury to a top driver still hurts, but structure absorbs shock better than most."
    elif false_contender_raw and weakness_dominates:
        profile = "CONTENDER (FRINGE)"
        issue = "Top-heavy scoring profile with depth concerns — competitive now but fragile if luck or health turns."
        risk = "Regression possible if secondary scoring or defensive support does not hold."
    elif false_contender_raw and not weakness_dominates:
        profile = "CONTENDER (FRINGE)" if pts_pct >= 0.52 else "BUBBLE (UPWARD)" if pts_trend_up else "BUBBLE (STALLED)"
        issue = (
            "The club shows competitive strengths in the standings and top-end talent, "
            "but underlying depth concerns remain — the picture is good, not automatic."
        )
        risk = "Implication: monitor middle-six and defensive support before treating the group as a locked-in favorite."
    elif low_pts_young:
        profile = "EMERGING TEAM"
        issue = "Results lag talent age curve — internal timeline still ahead of standings."
        risk = "Rushing the window with bad veteran adds."
    elif vet >= 9 and pts_pct < 0.52 and mean_top5_age >= 29:
        profile = "COLLAPSING CORE"
        issue = "Aging top of roster without replacement clarity."
        risk = "Sharp cliff if trades/extensions misfire."
    elif team_data.get("stalled_rebuild"):
        profile = "REBUILD (STALLED)"
        issue = "Multiple seasons without upward trajectory or pipeline conversion."
        risk = "Fan and owner pressure forces panic trades."
    elif not made_playoffs and bucket == "rebuild":
        profile = "REBUILD (EARLY)"
        issue = "Lottery geography acceptable if picks convert."
        risk = "Culture drift if losing extends without visible young NHL impact."
    elif mid_pts and not made_playoffs:
        profile = "BUBBLE (STALLED)"
        issue = "Too good to tank, not good enough to matter in April."
        risk = "Expensive mediocrity without a deliberate lane."
    elif made_playoffs:
        profile = "CONTENDER (FRINGE)" if contender_score >= 0.46 else "BUBBLE (UPWARD)"
        issue = (
            "Playoff qualification validates the season, but the weighted read still leaves room "
            "for execution and depth to decide how long the run lasts."
        )
        risk = "Health, matchups, and secondary scoring typically decide how far this profile goes."
    else:
        profile = "BUBBLE (UPWARD)" if pts_trend_up else "BUBBLE (STALLED)"

    issue_flags = {
        "high_star_dep": star_dep > 0.52,
        "low_depth": depth_score < 0.40,
        "high_goalie_rel": g_rel > 0.62,
        "struct_low": struct_integrity_score < 0.35,
        "declining_core": "DECLINING" in age_curve,
        "playoff_miss_streak": int(team_data.get("missed_playoff_streak") or 0),
    }

    return {
        "profile": profile,
        "issue": issue,
        "risk": risk,
        "structural_integrity": struct_integrity,
        "structural_integrity_score": struct_integrity_score,
        "contender_score": round(contender_score, 3),
        "depth_score": round(float(depth_score), 3),
        "star_dependency_index": round(float(star_dep), 3),
        "goalie_reliance_factor": round(float(g_rel), 3),
        "age_curve_position": age_curve,
        "mean_top5_age": round(mean_top5_age, 1),
        "depth_gap": round(depth_gap, 3),
        "issue_flags": issue_flags,
        "strength_profile": _bal0,
    }


def estimate_cap_structure(
    team: Any,
    roster_snapshot: Dict[str, Any],
    ustate: rs.UniverseState,
    *,
    year: int,
    user_tid: str,
) -> Dict[str, Any]:
    """
    Synthetic cap usage bounded to realistic NHL-style bands (never above 108%).
    Mean roster OVR (proxy 0–99) maps to target usage; payroll is back-solved from cap.
    """
    players = _roster_players(team)
    cap_m = float(ustate.salary_cap_m)
    if not players or cap_m <= 1.0:
        return {
            "payroll_m": 0.0,
            "usage_pct": 0.0,
            "efficiency": "UNKNOWN",
            "dead_weight": [],
            "value_contracts": [],
            "notes": "Insufficient roster or cap context.",
        }

    ovrs_n = [_player_ovr_norm(p) for p in players]
    mean_norm = sum(ovrs_n) / len(ovrs_n)
    avg_nhl99 = _mean_ovr_nhl99(mean_norm)
    if avg_nhl99 < 82:
        lo_u, hi_u = 90.0, 95.0
    elif avg_nhl99 < 85:
        lo_u, hi_u = 95.0, 100.0
    elif avg_nhl99 < 86:
        lo_u, hi_u = 98.0, 101.0
    elif avg_nhl99 < 89:
        lo_u, hi_u = 100.0, 104.0
    else:
        lo_u, hi_u = 104.0, 108.0

    j = _deterministic_unit_float(f"cap|{year}|{user_tid}|{mean_norm:.4f}")
    base_usage = lo_u + (hi_u - lo_u) * j
    jitter = (_deterministic_unit_float(f"capj|{year}|{user_tid}") - 0.5) * 4.0
    usage = rs.clamp(base_usage + jitter, 93.0, 108.0)
    payroll = cap_m * (usage / 100.0)

    rows: List[Tuple[str, float, float, int]] = []
    for p in players:
        o = _player_ovr_norm(p)
        age = _player_age(p)
        ident = getattr(p, "identity", None)
        nm = getattr(ident, "name", "?") if ident else "?"
        base = 0.85 + o * o * 10.5
        if _player_position_key(p) == "G":
            base *= 1.08
        if age >= 32:
            base *= 1.05
        elif age <= 22:
            base *= 0.88
        aav = rs.clamp(base, 0.775, 14.5)
        rows.append((str(nm), float(o), float(aav), age))

    ovrs_sorted = sorted(rows, key=lambda x: -x[1])
    names_top8 = {x[0] for x in ovrs_sorted[:8]}
    top3_mean = sum(x[1] for x in ovrs_sorted[:3]) / min(3, len(ovrs_sorted))
    rest_mean = sum(x[1] for x in ovrs_sorted[3:]) / max(1, len(ovrs_sorted) - 3) if len(ovrs_sorted) > 3 else top3_mean
    star_gap = max(0.0, top3_mean - rest_mean)
    vet_cnt = sum(1 for _, _, _, ag in rows if ag >= 30)
    depth_vs_stars = rs.clamp(1.0 - star_gap * 5.0, 0.0, 1.0)
    age_cost_stress = rs.clamp(vet_cnt / max(8, len(rows)), 0.0, 1.0)

    dead_weight: List[str] = []
    value_contracts: List[str] = []
    for nm, ovr, aav, age in rows:
        if ovr >= 0.78 and nm not in names_top8 and aav >= 5.0:
            dead_weight.append(f"{nm} (~${aav:.2f}M, OVR~{ovr:.2f}) — high cost outside top-8 impact tier")
        if ovr >= 0.72 and aav <= 4.2:
            value_contracts.append(f"{nm} (~${aav:.2f}M, OVR~{ovr:.2f})")

    if depth_vs_stars >= 0.55 and age_cost_stress <= 0.45:
        eff = "STRONG FLEX"
    elif depth_vs_stars >= 0.4:
        eff = "SOLID"
    elif usage >= 104 or age_cost_stress >= 0.55:
        eff = "CRAMPED"
    elif usage >= 99:
        eff = "TIGHT"
    else:
        eff = "MANAGEABLE"

    notes = (
        f"Usage model from mean roster OVR proxy ~{avg_nhl99:.0f} (band {lo_u:.0f}-{hi_u:.0f}% target, "
        f"clamped ≤108%). Depth-vs-stars and age-cost inform efficiency read."
    )

    return {
        "payroll_m": round(payroll, 2),
        "usage_pct": round(usage, 1),
        "efficiency": eff,
        "dead_weight": dead_weight[:5],
        "value_contracts": value_contracts[:5],
        "notes": notes,
    }


def _diagnosis_weakness_codes(diagnosis: Dict[str, Any], needs: List[str]) -> List[str]:
    """Map diagnosis + positional needs to insight / pipeline weakness buckets."""
    codes: List[str] = []
    need_blob = " ".join(needs).lower()
    depth = float(diagnosis.get("depth_score") or 0.5)
    star = float(diagnosis.get("star_dependency_index") or 0.4)
    grel = float(diagnosis.get("goalie_reliance_factor") or 0.35)
    if star > 0.48 or depth < 0.44:
        codes.append("scoring_issue")
    if "defense" in need_blob or depth < 0.36:
        codes.append("defense_issue")
    if depth < 0.42:
        codes.append("depth_issue")
    if "DECLINING" in str(diagnosis.get("age_curve_position") or ""):
        codes.append("age_issue")
    if grel > 0.62 or "goalie" in need_blob:
        codes.append("goalie_issue")
    if not codes:
        codes.append("balanced")
    return codes[:3]


def _pick_template_line(category: str, year: int, user_tid: str, slot: int, avoid: set) -> str:
    opts = INSIGHT_TEMPLATES.get(category) or INSIGHT_TEMPLATES["balanced"]
    n = len(opts)
    for k in range(n):
        idx = (int(_deterministic_unit_float(f"ins|{year}|{user_tid}|{category}|{slot}|{k}") * 10_007) + k) % n
        cand = opts[idx]
        if cand not in avoid:
            return cand
    return opts[int(_deterministic_unit_float(f"insfallback|{year}|{user_tid}|{category}") * 10_007) % n]


def generate_roster_insight(
    rstruct: Dict[str, Any],
    diagnosis: Dict[str, Any],
    players: List[Any],
    *,
    tm: Dict[str, Any],
    year: int,
    user_tid: str,
    p_eval: Dict[str, Any],
) -> List[str]:
    """Varied roster bullets + mandatory pipeline touchpoint; avoids repeats within 3 seasons."""
    by_pos = rstruct.get("by_pos") or {}
    needs = _infer_positional_needs(by_pos if isinstance(by_pos, dict) else {})
    hist = list(tm.get("roster_insight_history") or [])
    avoid: set = set()
    for block in hist[-3:]:
        for ph in block.get("phrases") or []:
            avoid.add(ph)

    weak_codes = _diagnosis_weakness_codes(diagnosis, needs)
    lines: List[str] = []
    picked: List[str] = []
    lines.append(f"Positional stress: {', '.join(needs)}.")
    for i, cat in enumerate(weak_codes[:2]):
        line = _pick_template_line(cat, year, user_tid, i, avoid)
        lines.append(line)
        picked.append(line)

    pipe_line = p_eval.get("pipeline_roster_hook") or (
        f"Prospect read: {p_eval.get('top_label', '?')} — {p_eval.get('pipeline_summary', p_eval.get('need_alignment', ''))}"
    )
    lines.append(pipe_line)
    picked.append(pipe_line)

    hist.append({"year": year, "phrases": picked})
    tm["roster_insight_history"] = hist[-8:]
    return lines


def adjust_for_context(strategy: str, trajectory: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Soften strategy when trending up + strong; sharpen when playoff drought persists."""
    notes: List[str] = []
    s = strategy
    trend = str(trajectory.get("trajectory_trend") or "neutral")
    strong = bool(trajectory.get("last_season_strong"))
    missed = int(trajectory.get("missed_streak") or trajectory.get("missed_playoffs_streak") or 0)

    if trend == "up" and strong:
        if s == "HARD_REBUILD":
            s = "SELL"
            notes.append("Context: upward trajectory + strong season — softened from full strip.")
        elif s == "SELL":
            s = "RETOOL"
            notes.append("Context: momentum positive — prefer surgical moves over fire sale.")
        elif s == "RETOOL":
            s = "HOLD"
            notes.append("Context: results validating build — pause aggressive restructuring.")

    if missed >= 3:
        if s == "HOLD":
            s = "RETOOL"
            notes.append("Context: 3+ missed playoffs — increase urgency to reshape middle roster.")
        elif s == "RETOOL":
            s = "SELL"
            notes.append("Context: extended playoff drought — asset-recovery posture warranted.")

    return s, notes


def enforce_strategy_coherence(
    diagnosis: Dict[str, Any],
    trajectory: Dict[str, Any],
    strategy: str,
) -> Tuple[str, List[str]]:
    """Profile-allowed strategies only; emits coherence adjustment notes when overriding."""
    notes: List[str] = []
    s = strategy
    prof = str(diagnosis.get("profile") or "")
    age_curve = str(diagnosis.get("age_curve_position") or "")
    trend = str(trajectory.get("trajectory_trend") or "neutral")
    mean_top5 = float(diagnosis.get("mean_top5_age") or 27.0)

    def bump(msg: str) -> None:
        notes.append(f"COHERENCE ADJUSTMENT: {msg}")

    if prof == "EMERGING TEAM" and "ASCENDING" in age_curve and trend in ("up", "neutral"):
        if s in ("HARD_REBUILD", "SELL", "RETOOL", "BUY"):
            bump("Strategy overridden to match diagnosis profile.")
            s = "HOLD"
    if prof in ("FALSE CONTENDER", "CONTENDER (FRINGE)"):
        if s == "BUY":
            bump("Strategy overridden to match diagnosis profile.")
            s = "RETOOL"
    if prof.startswith("CONTENDER (STRONG)") or prof == "TRUE CONTENDER":
        if s in ("HARD_REBUILD", "SELL"):
            bump("Strategy overridden to match diagnosis profile.")
            s = "HOLD"
    if prof.startswith("BUBBLE"):
        if s not in ("HOLD", "RETOOL"):
            bump("Strategy overridden to match diagnosis profile.")
            s = "RETOOL"
    if prof in ("STALLED REBUILD", "REBUILD (STALLED)"):
        if s == "BUY":
            bump("Strategy overridden to match diagnosis profile.")
            s = "HARD_REBUILD"
    return s, notes


def validate_front_office_output(diagnosis: Dict[str, Any], trajectory: Dict[str, Any], strategy: str) -> Tuple[str, List[str]]:
    """Final pass — re-apply coherence rules if context steps created illegal combos."""
    s2, n2 = enforce_strategy_coherence(diagnosis, trajectory, strategy)
    if s2 != strategy and not any("COHERENCE ADJUSTMENT" in x for x in n2):
        n2.append("COHERENCE ADJUSTMENT: Auto-corrected illegal strategy transition.")
    return s2, n2


def _infer_positional_needs(by_pos: Dict[str, int]) -> List[str]:
    needs: List[str] = []
    f = sum(by_pos.get(p, 0) for p in ("C", "LW", "RW"))
    d = sum(by_pos.get(p, 0) for p in ("D",))
    g = by_pos.get("G", 0)
    if by_pos.get("C", 0) < 3:
        needs.append("center depth")
    if min(by_pos.get("LW", 0), by_pos.get("RW", 0)) < 3:
        needs.append("wing balance")
    if d < 6:
        needs.append("defense quantity")
    if g < 2:
        needs.append("goalie coverage")
    if f > 16:
        needs.append("forward logjam — trade consolidation candidate")
    return needs or ["balanced chart (no glaring hole)"]


def evaluate_prospect_pipeline(
    team: Any,
    team_needs: List[str],
    league_avg_ceiling: float,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dx = diagnosis or {}
    weakness_codes = _diagnosis_weakness_codes(dx, team_needs)
    need_blob = " ".join(team_needs).lower()

    pool = list(getattr(team, "prospect_pool", None) or [])
    if not pool:
        return {
            "top_label": "None",
            "top_tier": "N/A",
            "classifications": [],
            "pipeline_strength": "LOW",
            "impact_years": "N/A",
            "need_alignment": "Pipeline empty — cannot cover roster holes.",
            "pipeline_summary": "Empty pool — external acquisition required for structural holes.",
            "pipeline_roster_hook": "PIPELINE GAP: No prospects on hand; roster holes likely need trade/FA solutions.",
            "addresses_weakness": False,
            "pipeline_impact": "",
            "pipeline_gap": "PIPELINE GAP: Pool empty vs current weaknesses — external acquisition required.",
        }

    scored: List[Tuple[float, Any]] = []
    for pr in pool:
        dr = getattr(pr, "draft_value_range", None)
        if isinstance(dr, (list, tuple)) and len(dr) >= 2:
            mid = (float(dr[0]) + float(dr[1])) / 2.0
        else:
            mid = 0.55
        scored.append((mid, pr))
    scored.sort(key=lambda x: -x[0])

    classifications: List[str] = []
    top_tier = "Role Player"
    top_name = "Prospect"
    top_pos = "?"
    impact_years = 3

    for idx, (mid, pr) in enumerate(scored[:12]):
        ident = getattr(pr, "identity", None)
        pos = "?"
        if ident is not None:
            pv = getattr(ident, "position", "?")
            pos = str(getattr(pv, "value", pv) or "?")
        nm = (getattr(ident, "name", None) if ident else None) or "Prospect"
        y = int(getattr(pr, "development_years_remaining", 3) or 3)
        arch = str(getattr(pr, "_dev_archetype", "") or "")
        bust_risk = arch in ("STALLED_DEVELOPER", "SAFE_LOW_CEILING") and mid < 0.62
        if mid >= 0.82 and y <= 2:
            tier = "Franchise Piece"
        elif mid >= 0.74:
            tier = "Core Player"
        elif mid >= 0.62:
            tier = "Role Player"
        elif bust_risk:
            tier = "Bust Risk"
        else:
            tier = "Trade Asset / Lottery Ticket"
        classifications.append(f"{nm} ({pos}): {tier}  ceiling~{mid:.2f}  ETA~{y}y")
        if idx == 0:
            top_tier = tier
            top_name = nm
            top_pos = pos
            impact_years = max(1, min(5, y))

    hi = scored[0][0] if scored else 0.55
    pipeline_impact_score = compute_pipeline_impact_score(scored, team_needs, weakness_codes)

    def _pos_matches_weakness(pos: str, codes: List[str]) -> bool:
        p = pos.upper()
        if "defense_issue" in codes and p == "D":
            return True
        if "scoring_issue" in codes and p in ("LW", "RW", "C"):
            return True
        if "goalie_issue" in codes and p == "G":
            return True
        if "depth_issue" in codes and p in ("LW", "RW", "C", "D"):
            return hi >= 0.68
        return False

    addresses = _pos_matches_weakness(top_pos, weakness_codes) and hi >= 0.64
    strength = "LOW"
    if len(pool) >= 10 and hi >= 0.72:
        strength = "HIGH"
    elif len(pool) >= 7 or hi >= 0.68:
        strength = "MEDIUM"

    if addresses:
        need_align = (
            f"PIPELINE IMPACT: {top_name} ({top_pos}) aligns with a stated structural weakness "
            f"— reduces urgency to pay retail for that spot externally."
        )
        pipeline_summary = (
            f"Top prospect {top_name} projects toward a current weakness; patience can substitute for a costly add."
        )
        pipeline_impact = need_align
        pipeline_gap = ""
    else:
        need_align = "Top prospect profile does not clearly patch the team's primary structural issues."
        pipeline_summary = "Pipeline head does not obviously solve the diagnosis weaknesses — market action may still be required."
        pipeline_impact = ""
        pipeline_gap = (
            "PIPELINE GAP: Top of the pool does not solve the current structural issue — external acquisition likely required."
        )

    if any("defense" in n for n in team_needs) and top_pos == "D" and hi >= 0.66:
        need_align = "Blue-line help visible in pipeline head — eases pressure on trade market for D help."
    if "center" in need_blob and top_pos == "C" and hi >= 0.75:
        need_align = "High-ceiling center prospect may absorb top-nine minutes if development holds."
    if "wing" in need_blob and top_pos in ("LW", "RW") and hi >= 0.70:
        need_align = "Wing pipeline head matches chart stress — less need to overpay for rental scoring."

    hook = need_align
    if pipeline_gap and not addresses:
        hook = f"{pipeline_gap} ({pipeline_summary})"
    elif addresses:
        hook = f"{pipeline_impact} ({pipeline_summary})"

    return {
        "top_label": top_name,
        "top_tier": top_tier,
        "classifications": classifications[:10],
        "pipeline_strength": strength,
        "impact_years": str(impact_years),
        "need_alignment": need_align,
        "pipeline_summary": pipeline_summary,
        "pipeline_roster_hook": hook,
        "addresses_weakness": addresses,
        "pipeline_impact": pipeline_impact,
        "pipeline_gap": pipeline_gap,
        "pipeline_impact_score": pipeline_impact_score,
    }


def update_team_identity(
    diagnosis: Dict[str, Any],
    pts: int,
    made_playoffs: bool,
    runner_archetype: str,
    pts_delta: Optional[int],
) -> str:
    """
    Narrative identity aligned to structure + performance (fixes 'rebuild' + 101 pts contradiction).
    """
    prof = diagnosis.get("profile", "")
    si = diagnosis.get("structural_integrity", "MODERATE")
    if prof == "CONTENDER (FRINGE)":
        return "fringe_contender — good season with structural caveats"
    if prof.startswith("CONTENDER ("):
        return "legitimate_contender"
    if prof == "FALSE CONTENDER":
        return "fake_contender — points overstate structural health"
    if prof == "TRUE CONTENDER":
        return "legitimate_contender"
    if prof == "EMERGING TEAM":
        return "emerging_core — results still catching up to youth"
    if prof == "REBUILD (EARLY)":
        return "rebuild_lane — accept pain if picks and young NHL minutes convert"
    if prof in ("STALLED REBUILD", "REBUILD (STALLED)", "COLLAPSING CORE"):
        return "direction_crisis — timeline and roster block don't match"
    if prof in ("STUCK MIDDLE", "BUBBLE (STALLED)"):
        return "expensive_middle — lane selection overdue"
    if prof == "BUBBLE (UPWARD)":
        return "trajectory_up — validate with second season"
    if made_playoffs and si in ("ELITE", "STABLE", "MODERATE"):
        return "playoff_caliber_structure"
    if pts >= 100 and si == "LOW":
        return "hollow_regular_season — fix depth before investing futures"
    if runner_archetype == "rebuild" and pts >= 96:
        return "rebuild_outperforming_label — consider retool vs strip"
    if runner_archetype in ("win_now", "contender") and pts < 88:
        return "underachieving_push — roster or coaching mismatch"
    if pts_delta is not None and pts_delta >= 10:
        return "trajectory_up — validate with second season"
    if pts_delta is not None and pts_delta <= -10:
        return "trajectory_down — intervention window open"
    return "balanced_watch — no single story dominates"


def determine_strategy(
    diagnosis: Dict[str, Any],
    prospect_eval: Dict[str, Any],
    standings_row: Dict[str, Any],
    trajectory: Dict[str, Any],
    *,
    profile_vec: Optional[Dict[str, float]] = None,
    flaw_severity: str = "MODERATE",
) -> Tuple[str, str]:
    """
    Weighted net read (strength vs weakness) drives base posture; profile and pipeline modulate.
    """
    prof = diagnosis.get("profile", "")
    depth = float(diagnosis.get("depth_score") or 0.5)
    star = float(diagnosis.get("star_dependency_index") or 0.4)
    pipe = prospect_eval.get("pipeline_strength", "MEDIUM")
    missed = int(trajectory.get("missed_streak") or 0)
    stuck = int(trajectory.get("years_stuck_middle") or 0)
    pts_hist: List[int] = list(trajectory.get("points_history") or [])
    trend_down = len(pts_hist) >= 2 and pts_hist[-1] < pts_hist[-2] - 5
    trend_up = len(pts_hist) >= 2 and pts_hist[-1] > pts_hist[-2] + 5
    addresses_weak = bool(prospect_eval.get("addresses_weakness"))
    vec = profile_vec or diagnosis.get("strength_profile") or {}
    ss = float(vec.get("strength_score") or 0.5)
    ws = float(vec.get("weakness_score") or 0.5)
    stab = float(vec.get("stability_score") or 0.5)
    trajs = float(vec.get("trajectory_score") or 0.5)
    pipe_imp = float(prospect_eval.get("pipeline_impact_score") or 0.5)

    net = ss - ws
    if flaw_severity == "MINOR":
        net += 0.08
    elif flaw_severity == "CRITICAL":
        net -= 0.12

    if net >= 0.40:
        idx = 4
        reason = f"Weighted read favors strengths (net {net:+.2f}) — short window to add if the fit is clean."
    elif net >= 0.20:
        idx = 3
        reason = f"Slight lean to the positive side (net {net:+.2f}) — patience and selective improvement."
    elif net >= -0.20:
        idx = 2
        reason = f"Push-pull roster (net {net:+.2f}) — reshape middle pieces without assuming total failure."
    elif net >= -0.40:
        idx = 1
        reason = f"Flaws outweigh positives (net {net:+.2f}) — recoup value before decline accelerates."
    else:
        idx = 0
        reason = f"Severe imbalance (net {net:+.2f}) — prioritize picks and cap clarity."

    if trajs >= 0.62:
        idx = min(4, idx + 1)
        reason += " Trajectory component supports a slightly more aggressive posture."
    if stab >= 0.66:
        idx = max(0, idx - 1)
        reason += " Stability read argues for less volatility in roster moves."

    strategy = _strategy_from_tier_index(idx)
    strategy, _pipe_notes = apply_pipeline_strategy_shift(strategy, pipe_imp)
    reason += (
        f" Pipeline impact score {pipe_imp:.2f} applied."
        if pipe_imp >= 0.68 or pipe_imp <= 0.32
        else ""
    )

    if prof == "EMERGING TEAM":
        base = "Youth curve ahead of standings — protect futures; only surgical adds."
        if addresses_weak:
            base += f" Pipeline ({prospect_eval.get('top_label')}) eases a diagnosed weakness — less need to force trades."
        elif pipe in ("HIGH", "MEDIUM"):
            base += " System depth acceptable — let NHL group grow into results."
        return ("HOLD", base)

    if prof.startswith("CONTENDER (STRONG)") and depth >= 0.48:
        r = "Short window open — targeted add if cap room allows."
        if addresses_weak:
            r += " Pipeline already points at one hole — shop selectively."
        if strategy in ("HARD_REBUILD", "SELL", "RETOOL"):
            strategy = "BUY" if net >= 0.25 else "HOLD"
            r = "Contender core intact — avoid teardown; " + r.lower()
        elif strategy == "HOLD" and net >= 0.35:
            strategy = "BUY"
        return (strategy, r)

    if prof in ("COLLAPSING CORE", "REBUILD (STALLED)", "STALLED REBUILD"):
        if net > -0.12 and flaw_severity == "MINOR":
            return ("RETOOL", "Stress is real but weighted read resists a full panic strip — start with controlled moves.")
        return ("HARD_REBUILD", "Timeline broken — prioritize picks and clear aging money where possible.")

    if missed >= 4 and not trend_up and not prof.startswith("CONTENDER (") and prof not in (
        "CONTENDER (FRINGE)",
        "BUBBLE (UPWARD)",
        "BUBBLE (STALLED)",
        "EMERGING TEAM",
    ):
        if ss > ws + 0.05:
            strategy = "RETOOL"
            reason = "Long miss streak, but strength signal resists labeling the roster as hopeless — retool first."
        else:
            strategy = "HARD_REBUILD"
            reason = "Extended miss cycle without lift — reset asset base."

    if prof == "CONTENDER (FRINGE)" or (star > 0.52 and depth < 0.45 and flaw_severity != "MINOR"):
        if strategy == "BUY":
            strategy = "HOLD"
        if strategy == "HARD_REBUILD":
            strategy = "RETOOL"
        r2 = "Fringe contender profile — clean up middle roster and cap without assuming the group is fraudulent."
        if not prospect_eval.get("addresses_weakness"):
            r2 += " Pipeline does not obviously patch the hole — keep external options open."
        return (strategy, r2)

    if stuck >= 2 or prof in ("STUCK MIDDLE", "BUBBLE (STALLED)"):
        if strategy == "HOLD" and net < -0.05:
            strategy = "RETOOL"
        reason = "Middle band stuck — pick a lane with asset trades, but weight both strengths and flaws."

    if prof in ("BUBBLE (UPWARD)", "BUBBLE (STALLED)"):
        if strategy in ("HARD_REBUILD", "SELL") and trajs >= 0.55:
            strategy = "RETOOL"
        if strategy == "BUY" and net < 0.30:
            strategy = "HOLD"

    if trend_down and standings_row.get("made_playoffs"):
        strategy = "HOLD"
        reason = "Still in tournament mix — fix process before selling core pieces."
    elif trend_down and not standings_row.get("made_playoffs") and net < 0.15:
        strategy = _strategy_from_tier_index(max(_strategy_tier_index(strategy), 1))
        reason = "Downward points trend without playoffs — recoup assets before values dip further."

    if prospect_eval.get("top_tier") == "Franchise Piece" and not prof.startswith("CONTENDER (STRONG)"):
        if strategy in ("BUY", "HARD_REBUILD"):
            strategy = "HOLD"
        reason = "Franchise prospect in system — avoid mortgaging that timeline for marginal NHL help."

    if depth < 0.38 and pipe == "LOW" and strategy == "BUY":
        strategy = "HOLD"
        reason = "Depth and pipeline both thin — cannot justify all-in buy."

    if addresses_weak and strategy in ("SELL", "HARD_REBUILD"):
        strategy = "RETOOL"
        reason = (
            f"Top prospect {prospect_eval.get('top_label')} tracks a weakness — prefer surgical change over full strip."
        )

    return (strategy, reason)


def _update_narrative_memory(
    tm: Dict[str, Any],
    *,
    mem: FranchisePerspectiveMemory,
    pts: int,
    bucket: str,
    made_playoffs: bool,
    diagnosis_profile: str,
    mean_top5_age: float,
    strategy: str,
) -> None:
    ph = list(tm.get("points_history") or [])
    ph.append(int(pts))
    tm["points_history"] = ph[-5:]
    bh = list(tm.get("bucket_history") or [])
    bh.append(str(bucket))
    tm["bucket_history"] = bh[-5:]

    tm["missed_playoffs_streak"] = int(mem.consecutive_playoff_misses)

    if made_playoffs or bucket in ("contender", "playoff"):
        tm["years_since_contention"] = 0
    else:
        tm["years_since_contention"] = int(tm.get("years_since_contention") or 0) + 1

    mid = 88 <= pts <= 100
    if mid and not made_playoffs:
        tm["years_stuck_middle"] = int(tm.get("years_stuck_middle") or 0) + 1
    elif pts > 102 or made_playoffs:
        tm["years_stuck_middle"] = 0

    if "REBUILD" in diagnosis_profile or bucket == "rebuild":
        tm["rebuild_seasons"] = int(tm.get("rebuild_seasons") or 0) + 1
    else:
        tm["rebuild_seasons"] = 0

    cat = list(tm.get("core_age_trend") or [])
    cat.append(round(float(mean_top5_age), 1))
    tm["core_age_trend"] = cat[-6:]

    prev = tm.get("last_strategy")
    if prev == "BUY" and not made_playoffs and pts < 90:
        tm["failed_strategy_count"] = int(tm.get("failed_strategy_count") or 0) + 1
    tm["last_strategy"] = strategy


def _trade_implication_for_user(headline: str, user_tid: str, display_name: str) -> Tuple[str, str, str]:
    """
    Heuristic parse of runner trade headlines → (move summary, reason, direction signal).
    """
    h = headline or ""
    low = h.lower()
    tid_l = str(user_tid).lower()
    nick = display_name.split()[-1].lower() if display_name else ""
    reason = "Roster refresh / asset swap per hockey ops read of value."
    direction = "HOLD"
    move = "Trade touched our organization."

    if "from " + tid_l in low or (nick and f"from {nick}" in low):
        move = "Outgoing player(s) / cap or pick cost."
        reason = "Likely cap, timeline, or fit mismatch driving exit."
        direction = "SELL / RESET"
    elif "acquires" in low and (tid_l in low or (nick and nick in low)):
        move = "Incoming player(s) or picks."
        reason = "Addressing immediate NHL hole or futures."
        direction = "BUY / ADD"
    if "pick" in low and "prospect" in low:
        reason += " Futures-heavy structure suggests long-game emphasis."
        direction = "REBUILD / RETOOL signal" if direction.startswith("SELL") else direction
    return move, reason, direction


def _emit_transaction_analyses(
    lines_out: List[str],
    trade_events: List[rs.UniverseEvent],
    user_tid: str,
    display_name: str,
) -> None:
    if not trade_events:
        lines_out.append("  (No trades logged involving our club this season.)\n")
        return
    for ev in trade_events[:8]:
        move, reason, direction = _trade_implication_for_user(ev.headline, user_tid, display_name)
        lines_out.append(f"  • {ev.headline}\n")
        lines_out.append(f"    Move read: {move}\n")
        lines_out.append(f"    Likely driver: {reason}\n")
        lines_out.append(f"    Direction signal: {direction}\n")


def emit_front_office_intelligence(
    *,
    year: int,
    user_team: Any,
    user_tid: str,
    display_name: str,
    ustate: rs.UniverseState,
    uni_result: rs.UniverseYearResult,
    league: Any,
    mem: FranchisePerspectiveMemory,
    lines_out: List[str],
    arch: str,
    bucket: str,
    pts: int,
    rank: int,
    n_teams: int,
    made_playoffs: bool,
    won_cup: bool,
    pts_delta: Optional[int],
    rstruct: Dict[str, Any],
    players: List[Any],
    scored_evs: List[Tuple[float, rs.UniverseEvent]],
) -> None:
    """Appended after legacy season sections; does not remove prior output."""
    lg_avg = _league_avg_roster_ovr(league)
    tm = mem.team_memory
    stalled = int(tm.get("rebuild_seasons") or 0) >= 3 and bucket == "rebuild" and not made_playoffs
    ph_full = list(tm.get("points_history") or []) + [pts]

    team_data = {
        **rstruct,
        "pts": pts,
        "rank": rank,
        "n_teams": n_teams,
        "bucket": bucket,
        "made_playoffs": made_playoffs,
        "won_cup": won_cup,
        "players": players,
        "stalled_rebuild": stalled or (int(mem.consecutive_playoff_misses) >= 4 and bucket == "rebuild"),
        "points_history": ph_full,
        "missed_playoff_streak": int(mem.consecutive_playoff_misses),
    }
    league_ctx = {"league_avg_ovr": lg_avg, "salary_cap_m": float(ustate.salary_cap_m)}
    dx = diagnose_team(team_data, league_ctx)

    needs = _infer_positional_needs(rstruct.get("by_pos") or {})
    p_eval = evaluate_prospect_pipeline(user_team, needs, lg_avg, diagnosis=dx)
    cap_x = estimate_cap_structure(
        user_team, rstruct, ustate, year=int(year), user_tid=str(user_tid)
    )

    metrics_for_profile: Dict[str, Any] = {
        "pts": pts,
        "rank": rank,
        "n_teams": n_teams,
        "made_playoffs": made_playoffs,
        "won_cup": won_cup,
        "league_avg_ovr": lg_avg,
        "mean_ovr": float(rstruct.get("mean_ovr") or 0.72),
        "top3_mean": float(rstruct.get("top3_mean") or rstruct.get("mean_ovr") or 0.72),
        "points_history": ph_full,
        "pipeline_strength": p_eval.get("pipeline_strength", "MEDIUM"),
        "cap_efficiency": cap_x.get("efficiency", "SOLID"),
    }
    full_vec = compute_team_strength_profile(user_team, dx, metrics_for_profile)
    dx["strength_profile"] = full_vec
    flaw_sev = classify_flaw_severity(dx.get("issue_flags") or {})
    if flaw_sev in ("MINOR", "MODERATE") and dx.get("profile") not in ("COLLAPSING CORE", "REBUILD (STALLED)"):
        bal_i, bal_r = _balanced_issue_risk(str(dx.get("profile", "")), dx, full_vec)
        dx["issue"] = bal_i
        dx["risk"] = bal_r

    narrative_id = update_team_identity(dx, pts, made_playoffs, arch, pts_delta)
    traj_trend = "neutral"
    if len(ph_full) >= 2:
        if ph_full[-1] > ph_full[-2] + 5:
            traj_trend = "up"
        elif ph_full[-1] < ph_full[-2] - 5:
            traj_trend = "down"
    traj = {
        "missed_streak": int(mem.consecutive_playoff_misses),
        "years_stuck_middle": int(tm.get("years_stuck_middle") or 0),
        "points_history": ph_full,
        "trajectory_trend": traj_trend,
        "last_season_strong": bool(made_playoffs or pts >= 98),
    }
    st_row = {"made_playoffs": made_playoffs, "pts": pts, "rank": rank}
    strategy, strat_reason = determine_strategy(
        dx, p_eval, st_row, traj, profile_vec=full_vec, flaw_severity=flaw_sev
    )
    co_notes: List[str] = []
    strategy, n_pat = apply_gm_patience(
        strategy, float(full_vec.get("trajectory_score") or 0.5), made_playoffs, traj_trend
    )
    co_notes.extend(n_pat)
    strategy, n_coh = enforce_strategy_coherence(dx, traj, strategy)
    co_notes.extend(n_coh)
    strategy, n_ctx = adjust_for_context(strategy, traj)
    co_notes.extend(n_ctx)
    strategy, n_val = validate_front_office_output(dx, traj, strategy)
    co_notes.extend(n_val)
    strategy, n_fin = final_decision_validator(dx.get("profile", ""), strategy, dx, full_vec)
    co_notes.extend(n_fin)

    lines_out.append("\n")
    lines_out.append("=== FRONT OFFICE REPORT ===\n")
    lines_out.append(f"Season {year}  |  {display_name}\n")
    lines_out.append("\nTEAM DIAGNOSIS\n")
    lines_out.append(f"  Profile: {dx['profile']}\n")
    lines_out.append(f"  Issue: {dx['issue']}\n")
    lines_out.append(f"  Risk: {dx['risk']}\n")
    lines_out.append(f"  Structural integrity: {dx['structural_integrity']}\n")
    lines_out.append(
        f"  Scores (0–1 scale): contender={dx['contender_score']}  depth={dx['depth_score']}  "
        f"star-dependency={dx['star_dependency_index']}  goalie-reliance={dx['goalie_reliance_factor']}\n"
    )
    lines_out.append(f"  Age curve read: {dx['age_curve_position']} (top-5 mean age {dx['mean_top5_age']})\n")
    lines_out.append(f"  Narrative identity (performance-aligned): {narrative_id}\n")

    lines_out.append("\nROSTER ANALYSIS\n")
    for ln in generate_roster_insight(
        rstruct,
        dx,
        players,
        tm=tm,
        year=int(year),
        user_tid=str(user_tid),
        p_eval=p_eval,
    ):
        lines_out.append(f"  • {ln}\n")

    lines_out.append("\nCAP ANALYSIS (synthetic)\n")
    lines_out.append(f"  Estimated payroll: ~${cap_x['payroll_m']:.2f}M  vs cap ${float(ustate.salary_cap_m):.1f}M\n")
    lines_out.append(f"  Estimated cap usage: ~{cap_x['usage_pct']}%\n")
    lines_out.append(f"  Efficiency read: {cap_x['efficiency']}\n")
    lines_out.append(f"  Note: {cap_x['notes']}\n")
    if cap_x["dead_weight"]:
        lines_out.append("  Dead-weight style contracts (proxy):\n")
        for d in cap_x["dead_weight"][:4]:
            lines_out.append(f"    - {d}\n")
    else:
        lines_out.append("  Dead-weight proxy: none flagged beyond normal middle-class noise.\n")
    if cap_x["value_contracts"]:
        lines_out.append("  Value spots:\n")
        for v in cap_x["value_contracts"][:4]:
            lines_out.append(f"    - {v}\n")

    lines_out.append("\nPROSPECT OUTLOOK\n")
    lines_out.append(f"  Top of pool: {p_eval['top_label']} — tier: {p_eval['top_tier']}\n")
    lines_out.append(f"  Pipeline strength: {p_eval['pipeline_strength']}\n")
    lines_out.append(f"  NHL impact timeline (headline prospect): ~{p_eval['impact_years']} season(s)\n")
    lines_out.append(f"  Need alignment: {p_eval['need_alignment']}\n")
    if p_eval.get("pipeline_impact"):
        lines_out.append(f"  {p_eval['pipeline_impact']}\n")
    if p_eval.get("pipeline_gap"):
        lines_out.append(f"  {p_eval['pipeline_gap']}\n")
    for c in p_eval["classifications"][:6]:
        lines_out.append(f"    • {c}\n")

    lines_out.append("\nFRANCHISE TRAJECTORY (memory)\n")
    fan = "STABLE"
    if int(mem.consecutive_playoff_misses) >= 3:
        fan = "DECLINING"
    elif strategy == "BUY" and (
        dx["profile"].startswith("CONTENDER (STRONG)") or dx["profile"] == "TRUE CONTENDER"
    ):
        fan = "ELEVATED"
    pressure_lvl = "MEDIUM"
    if int(mem.consecutive_playoff_misses) >= 4 or dx["profile"] in (
        "STALLED REBUILD",
        "REBUILD (STALLED)",
        "COLLAPSING CORE",
        "STUCK MIDDLE",
        "BUBBLE (STALLED)",
    ):
        pressure_lvl = "HIGH"
    elif (
        dx["profile"].startswith("CONTENDER (STRONG)")
        or dx["profile"] == "TRUE CONTENDER"
        or dx["profile"] == "EMERGING TEAM"
    ) and int(mem.consecutive_playoff_misses) == 0:
        pressure_lvl = "ELEVATED" if strategy == "BUY" else "MODERATE"
    lines_out.append(f"  Pressure level: {pressure_lvl}\n")
    lines_out.append(f"  Missed-playoff streak: {mem.consecutive_playoff_misses}\n")
    lines_out.append(f"  Years since contention signal: {tm.get('years_since_contention', 0)}\n")
    lines_out.append(f"  Years stuck middle band: {tm.get('years_stuck_middle', 0)}\n")
    lines_out.append(f"  Core age trail (recent): {tm.get('core_age_trend', [])}\n")
    lines_out.append(f"  Last season strategy tag: {tm.get('last_strategy')}\n")
    lines_out.append(f"  Fan sentiment (synthetic): {fan}\n")
    if stalled or dx["profile"] in ("STALLED REBUILD", "REBUILD (STALLED)"):
        lines_out.append(f"  Status note: STALLED REBUILD read ({int(tm.get('rebuild_seasons') or 0)} seasons in rebuild bucket without lift).\n")

    lines_out.append("\nFRONT OFFICE DIRECTION\n")
    lines_out.append(f"  Strategy: {strategy}\n")
    lines_out.append(f"  Reason: {strat_reason}\n")
    for note in co_notes:
        lines_out.append(f"  {note}\n")
    if p_eval.get("pipeline_summary"):
        lines_out.append(f"  Pipeline context: {p_eval['pipeline_summary']}\n")
    if p_eval["top_tier"] in ("Franchise Piece", "Core Player"):
        lines_out.append(
            f"  Prospect lever: {p_eval['top_label']} ({p_eval['top_tier']}) influences patience — avoid selling that path for marginal upgrades.\n"
        )

    trade_evs = [ev for _, ev in scored_evs if (getattr(ev, "type", "") or "").upper() == "TRADE" and user_tid in [str(x) for x in (getattr(ev, "teams", None) or [])]]
    if not trade_evs:
        trade_evs = [ev for _, ev in scored_evs if (getattr(ev, "type", "") or "").upper() == "TRADE" and user_tid.lower() in (ev.headline or "").lower()]
    lines_out.append("\nTRANSACTION ANALYSIS (our club)\n")
    _emit_transaction_analyses(lines_out, trade_evs, user_tid, display_name)

    verdict_core = (
        f"This group profiles as {dx['profile'].lower()}: {dx['issue']} "
        f"Front office should execute a {strategy} posture — {strat_reason.lower()}"
    )
    lines_out.append("\nFINAL VERDICT\n")
    lines_out.append(f"  What we are: {narrative_id.replace('_', ' ')}.\n")
    lines_out.append(f"  What we should do: {strategy} — {strat_reason}\n")
    lines_out.append(f"  One-line: {verdict_core}\n")

    _update_narrative_memory(
        tm,
        mem=mem,
        pts=pts,
        bucket=bucket,
        made_playoffs=made_playoffs,
        diagnosis_profile=dx["profile"],
        mean_top5_age=float(dx.get("mean_top5_age") or 27.0),
        strategy=strategy,
    )


def format_year_block(
    *,
    year: int,
    user_team: Any,
    user_tid: str,
    display_name: str,
    ustate: rs.UniverseState,
    uni_result: Optional[rs.UniverseYearResult],
    league_season_result: Any,
    sim: Any,
    league: Any,
    mem: FranchisePerspectiveMemory,
    lines_out: List[str],
) -> None:
    if uni_result is None:
        lines_out.append(f"\n### Season {year} — universe layer skipped (mode).\n")
        return

    standings = uni_result.standings
    st = _standing_for_team(standings, user_tid)
    teams = rs._get_league_teams(league)
    teams_by_id = _teams_by_id(teams)
    n_teams = len(standings) or len(teams) or 32
    if st is None:
        lines_out.append(f"\n### Season {year} — no standings row for team_id={user_tid}; skipping detail.\n")
        return

    rank = 1 + sum(1 for x in standings if (x.points, x.goal_diff) > (st.points, st.goal_diff))

    cut = _playoff_cut(n_teams)
    made_playoffs = rank <= cut
    champ = str(getattr(uni_result, "playoff_champion_id", "") or "")
    won_cup = champ == user_tid

    arch = str((ustate.team_archetypes or {}).get(user_tid, "balanced"))
    bucket = st.bucket if st is not None else "unknown"
    pts = int(st.points) if st is not None else 0
    pts_delta: Optional[int] = None
    if mem.last_points is not None:
        pts_delta = pts - mem.last_points

    window, posture = window_and_posture(arch, bucket, mem)
    rstruct = roster_structure_snapshot(user_team)
    players = _roster_players(user_team)
    pipe = prospect_pipeline_snapshot(user_team)
    market_s, own_s, org_s = org_profiles_snapshot(user_team)

    lines_out.append(f"\n{'=' * 72}\n")
    lines_out.append(f"SEASON {year}  |  {display_name}\n")
    lines_out.append(f"{'=' * 72}\n")
    lines_out.append("Franchise identity snapshot\n")
    lines_out.append(f"  Runner archetype: {arch}\n")
    lines_out.append(f"  Competitive window read: {window}\n")
    lines_out.append(f"  Finish: {pts} pts  (~{rank}/{n_teams})  bucket={bucket}  playoffs={'yes' if made_playoffs else 'no'}\n")
    if won_cup:
        lines_out.append("  Outcome: Stanley Cup champion — expectations reset upward sharply.\n")
    elif champ:
        lines_out.append(f"  League champion: team {champ} (not us).\n")

    lines_out.append("\nInternal front-office brief\n")
    lines_out.append(f"  Market / media: {market_s}\n")
    lines_out.append(f"  Ownership levers: {own_s}\n")
    lines_out.append(f"  Org state: {org_s}\n")
    lines_out.append(f"  Cap read: {cap_pressure_phrase(user_team, ustate)}\n")
    lines_out.append(f"  {coach_snapshot(user_team)}\n")
    lines_out.append(f"  Trade / deadline posture: {posture}\n")

    if mem.prior_year_summary:
        lines_out.append(f"\nCarryover context: {mem.prior_year_summary}\n")

    lines_out.append("\nSeason flow (internal)\n")
    beat = narrative_season_beat(
        bucket=bucket, rank=rank, n_teams=n_teams, pts_delta=pts_delta, made_playoffs=made_playoffs
    )
    lines_out.append(f"  {beat}\n")

    if mem.pending_draft_log:
        lines_out.append("\nEntry draft (interactive / logged)\n")
        lines_out.extend(mem.pending_draft_log)
        mem.pending_draft_log.clear()

    lines_out.append("\nRoster structure (our club)\n")
    if rstruct["n"] == 0:
        lines_out.append("  Roster empty or unreadable.\n")
    else:
        lines_out.append(
            f"  Skaters/goalies: {rstruct['n']}  mean OVR~{float(rstruct['mean_ovr']):.3f}  "
            f"top-3 mean~{float(rstruct['top3_mean']):.3f}\n"
        )
        lines_out.append(f"  Age mix: U24 {rstruct['u24']} | prime {rstruct['prime']} | 30+ {rstruct['vet']}\n")
        lines_out.append(f"  Positional counts: {rstruct['by_pos']}\n")

    lines_out.append("\nProspect pipeline — our view\n")
    lines_out.append(f"  Pool size: {pipe['count']}  ({'thin' if pipe['thin'] else 'healthy'} vs org minimums)\n")
    for ln in pipe["top_lines"][:6]:
        lines_out.append(f"    • {ln}\n")
    if not pipe["top_lines"]:
        lines_out.append("    (No ranked prospects in pool view.)\n")

    lines_out.append("\nLeague events that matter to us (filtered)\n")
    user_div = str(getattr(user_team, "division", "") or "")
    user_conf = str(getattr(user_team, "conference", "") or "")
    scored_evs: List[Tuple[float, rs.UniverseEvent]] = []
    for ev in uni_result.events:
        sc = event_relevance_score(
            ev, user_tid=user_tid, user_div=user_div, user_conf=user_conf, teams_by_id=teams_by_id, standings=standings
        )
        if sc >= 32.0:
            scored_evs.append((sc, ev))
    scored_evs.sort(key=lambda x: -x[0])
    if not scored_evs:
        lines_out.append("  No high-relevance external events crossed the board threshold this year.\n")
    for sc, ev in scored_evs[:14]:
        lines_out.append(f"  [{sc:4.0f}] ({ev.type}) {ev.headline}\n")

    lines_out.append("\nPlayoff / consequence framing\n")
    if won_cup:
        lines_out.append(
            "  Championship validates the current path; extension and cap sequencing become the dominant summer work.\n"
        )
    elif made_playoffs:
        lines_out.append(
            "  Playoff ticket punched — the honest question is whether the run was structural or matchup-luck driven.\n"
        )
    else:
        lines_out.append(
            "  Lottery geography hurts morale but can accelerate a retool if ownership stays aligned on timeline.\n"
        )

    lines_out.append("\nEngine league season (structural) — our takeaway\n")
    if league_season_result is not None:
        try:
            champ2 = getattr(getattr(league_season_result, "playoff_result", None), "champion_id", None)
            lines_out.append(f"  Structural sim champion team_id={champ2!r} (cross-check vs universe macro champion).\n")
        except Exception:
            lines_out.append("  Playoff object present; details omitted (safe fallback).\n")
    else:
        lines_out.append("  No league_season_result returned for this year (engine optional path).\n")

    # Awards mentioning our players
    if league_season_result is not None and getattr(league_season_result, "awards", None):
        hits: List[str] = []
        try:
            for aname, aw in league_season_result.awards.items():
                if str(getattr(aw, "winner_team_id", "")) == user_tid:
                    hits.append(f"{aname}: {getattr(aw, 'winner_name', '?')}")
        except Exception:
            pass
        if hits:
            lines_out.append("  Internal highlight reel (awards tied to our roster):\n")
            for h in hits[:8]:
                lines_out.append(f"    • {h}\n")

    summary = (
        f"{year}: {pts} pts, rank {rank}/{n_teams}, playoffs={'Y' if made_playoffs else 'N'}, identity={arch}, bucket={bucket}."
    )
    mem.record_season_end(
        made_playoffs=made_playoffs,
        points=pts,
        bucket=bucket,
        rank=rank,
        summary_line=summary,
    )

    lines_out.append("\nYear-end strategic summary\n")
    lines_out.append(f"  {summary}\n")
    if mem.consecutive_playoff_misses >= 2:
        lines_out.append(
            f"  Pressure note: {mem.consecutive_playoff_misses} consecutive misses — patience narrative is thinning.\n"
        )
    if pipe["thin"] and arch not in ("rebuild", "draft_and_develop"):
        lines_out.append("  Structural risk: thin prospect floor while timeline says compete — futures trades are costly.\n")

    emit_front_office_intelligence(
        year=year,
        user_team=user_team,
        user_tid=user_tid,
        display_name=display_name,
        ustate=ustate,
        uni_result=uni_result,
        league=league,
        mem=mem,
        lines_out=lines_out,
        arch=arch,
        bucket=bucket,
        pts=pts,
        rank=rank,
        n_teams=n_teams,
        made_playoffs=made_playoffs,
        won_cup=won_cup,
        pts_delta=pts_delta,
        rstruct=rstruct,
        players=players,
        scored_evs=scored_evs,
    )


def build_per_year_callback(
    *,
    user_team: Any,
    display_name: str,
    mem: FranchisePerspectiveMemory,
    lines_out: List[str],
) -> Callable[..., None]:
    user_tid = str(rs._team_id(user_team))

    def _cb(
        *,
        year: int,
        ustate: rs.UniverseState,
        uni_result: Optional[rs.UniverseYearResult],
        career_result: Any,
        league_season_result: Any,
        sim: Any,
        league: Any,
    ) -> None:
        format_year_block(
            year=year,
            user_team=user_team,
            user_tid=user_tid,
            display_name=display_name,
            ustate=ustate,
            uni_result=uni_result,
            league_season_result=league_season_result,
            sim=sim,
            league=league,
            mem=mem,
            lines_out=lines_out,
        )

    return _cb


def run_perspective(
    *,
    team_query: str,
    seed: int,
    years: int,
    start_year: int,
    mode: str,
    debug: bool,
    write_json: bool,
    interactive_draft: bool = False,
) -> Tuple[int, Path]:
    """
    Execute the same core loop as run_sim.main with a sink universe log and POV output.
    Returns (exit_code, perspective_file_path).
    """
    base_dir = Path(__file__).resolve().parent
    perspective_dir = base_dir / "perspective"
    perspective_dir.mkdir(parents=True, exist_ok=True)

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"perspective_{stamp}_seed{seed}.txt"
    out_path = perspective_dir / out_name
    latest_path = perspective_dir / "sim_perspective_latest.txt"

    run_cfg = rs.RunConfig(
        seed=seed,
        years=years,
        start_year=start_year,
        debug=debug,
        mode=mode,
        log_level="minimal",
        output_dir=str(perspective_dir / "_sink_run"),
        write_json=write_json,
        write_debug_state=False,
        pretty_print=False,
        flush_each_year=True,
        per_year_callback=None,
    )
    uni_cfg = rs.UniverseConfig()
    scn_cfg = rs.load_scenario_file(None)

    sink_dir = base_dir / run_cfg.output_dir
    sink_dir.mkdir(parents=True, exist_ok=True)
    sink_timeline = sink_dir / "universe_timeline_sink.log"
    sink_out = _DevNullTimelineRunOutput(run_dir=sink_dir, pretty=False, timeline_path=sink_timeline)
    sink_out.open()
    sink_logger = _QuietRunnerLogger(out=sink_out, log_level="minimal", flush_each_year=True)

    master_rng = rs.rng_from_seed(seed)
    meta_rng = rs.rng_from_seed(rs.split_seed(seed, "meta"))

    sim = None
    if rs.SimEngine is not None:
        try:
            try:
                sim = rs.SimEngine(seed=seed, debug=debug)  # type: ignore[misc]
            except TypeError:
                sim = rs.SimEngine(seed=seed)  # type: ignore[misc]
        except Exception:
            sim = None

    # Bootstrap league (mirror main)
    try:
        league_obj = getattr(sim, "league", None) if sim else None
        if league_obj is not None:
            teams = getattr(league_obj, "teams", None)
            if not teams:
                if hasattr(league_obj, "generate_default_teams"):
                    league_obj.generate_default_teams()
                elif sim and hasattr(sim, "generate_league"):
                    sim.generate_league()
                elif sim and hasattr(sim, "setup_league"):
                    sim.setup_league()
    except Exception:
        pass

    league = getattr(sim, "league", None) if sim else None
    if league is not None:
        try:
            setattr(league, "_runner_sim_engine", sim)
        except Exception:
            pass

    teams = rs._get_league_teams(league)
    if not teams:
        text = "FATAL: No teams in league; cannot run perspective mode.\n"
        out_path.write_text(text, encoding="utf-8")
        return 2, out_path

    career_rng = rs.rng_from_seed(rs.split_seed(seed, "career"))
    if sim is not None:
        if getattr(sim, "player", None) is None and hasattr(sim, "__dict__"):
            sim.__dict__["player"] = rs._create_random_player(career_rng)
        if getattr(sim, "team", None) is None:
            pick = rs._pick_player_team(list(teams), career_rng)
            if pick is not None and hasattr(sim, "__dict__"):
                sim.__dict__["team"] = pick

    try:
        user_team = resolve_user_team(teams, team_query)
    except ValueError as e:
        out_path.write_text(f"FATAL: {e}\n", encoding="utf-8")
        return 2, out_path

    display_name = _display_team(user_team)
    pov_lines: List[str] = []
    mem = FranchisePerspectiveMemory()

    run_cfg.per_year_callback = build_per_year_callback(
        user_team=user_team,
        display_name=display_name,
        mem=mem,
        lines_out=pov_lines,
    )

    arche, coach_ids, coach_sec = rs._init_team_identities(teams, meta_rng)
    ustate = rs.UniverseState(
        salary_cap_m=float(uni_cfg.salary_cap_start),
        cap_growth_rate=float(uni_cfg.cap_growth_rate_mean),
        inflation_factor=1.0,
        league_health=rs.clamp(uni_cfg.league_health_target + meta_rng.uniform(-0.05, 0.05), 0.10, 0.95),
        parity_index=rs.clamp(uni_cfg.parity_target + meta_rng.uniform(-0.05, 0.05), 0.10, 0.90),
        chaos_index=rs.clamp(uni_cfg.chaos + meta_rng.uniform(-0.05, 0.05), 0.10, 0.95),
        active_era=rs._choose_era(meta_rng),
        waiver_priority=[rs._team_id(t) for t in teams] if teams else [],
        team_archetypes=arche,
        coach_ids=coach_ids,
        coach_security=coach_sec,
        rng_traces=[],
        tuning_report={},
    )

    pov_lines.append("=" * 72 + "\n")
    pov_lines.append("FRANCHISE PERSPECTIVE SIMULATION (GM / front-office POV)\n")
    pov_lines.append("=" * 72 + "\n")
    pov_lines.append(f"Mode          : perspective\n")
    pov_lines.append(f"Seed          : {seed}\n")
    pov_lines.append(f"Years         : {years}\n")
    pov_lines.append(f"Start season  : {start_year}\n")
    pov_lines.append(f"Universe mode : {mode}\n")
    pov_lines.append(f"User team     : {display_name} (id={rs._team_id(user_team)})\n")
    pov_lines.append(f"Interactive draft: {'on (your picks when on the clock)' if interactive_draft else 'off (full AI draft)'}\n")
    pov_lines.append(
        f"Timestamp UTC : {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}\n"
    )
    pov_lines.append(f"Output file   : {out_path}\n")
    pov_lines.append("\nEngine note: underlying simulation matches run_sim.run_simulation_core; ")
    pov_lines.append("universe timeline is discarded to reduce noise. This file is the authoritative POV log.\n")

    narrative_context: Dict[str, Any] = {}
    last_context: Dict[str, Any] = {"phase": "init", "year": None}

    attach_user_draft_hooks(sim, user_team=user_team, mem=mem, enabled=bool(interactive_draft))

    exit_code = 0
    try:
        ustate = rs.run_simulation_core(
            run_cfg=run_cfg,
            uni_cfg=uni_cfg,
            scn_cfg=scn_cfg,
            league=league,
            sim=sim,
            ustate=ustate,
            narrative_context=narrative_context,
            logger=sink_logger,
            out=sink_out,
            last_context=last_context,
        )
        if run_cfg.mode == "regression":
            rs.regression_compare_or_write(run_cfg, sink_out, sink_logger)
        pov_lines.append("\n" + "=" * 72 + "\n")
        pov_lines.append("RUN COMPLETE — universe state advanced; review season blocks above for narrative continuity.\n")
    except Exception as e:
        exit_code = 1
        pov_lines.append("\n[FATAL] Perspective run crashed.\n")
        pov_lines.append(f"{type(e).__name__}: {e}\n")
        pov_lines.append(traceback.format_exc())
    finally:
        clear_user_draft_hooks(sim)
        try:
            sink_out.close()
        except Exception:
            pass

    text = "".join(pov_lines)
    out_path.write_text(text, encoding="utf-8")
    try:
        latest_path.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return exit_code, out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL Franchise Mode — franchise perspective (GM POV) simulation")
    p.add_argument("--team", type=str, default="Ottawa Senators", help="Team city, nickname, or id (default: Ottawa Senators).")
    p.add_argument("--seed", type=int, default=None, help="Master seed (default: time_ns).")
    p.add_argument("--years", type=int, default=10, help="Seasons to simulate.")
    p.add_argument("--start_year", type=int, default=2025, help="First calendar year label.")
    p.add_argument("--mode", type=str, default="combined", choices=rs.RUN_MODES, help="Same modes as run_sim.py.")
    p.add_argument("--debug", action="store_true", help="Forward engine debug flag when supported.")
    p.add_argument("--json_sink", action="store_true", help="Also write universe_year_*.json into perspective/_sink_run/.")
    p.add_argument(
        "--interactive-draft",
        action="store_true",
        help="When your team is on the clock, prompt for pick (1–10) or 'auto' for AI. No effect without sim engine.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv if argv is not None else sys.argv[1:])
    seed = args.seed if args.seed is not None else int(getattr(__import__("time"), "time_ns")())
    code, path = run_perspective(
        team_query=args.team,
        seed=seed,
        years=int(args.years),
        start_year=int(args.start_year),
        mode=str(args.mode),
        debug=bool(args.debug),
        write_json=bool(args.json_sink),
        interactive_draft=bool(getattr(args, "interactive_draft", False)),
    )
    print(f">>> Perspective log written to: {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
