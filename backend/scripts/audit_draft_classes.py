"""Large draft-class simulation audit.

Usage:
  python backend/scripts/audit_draft_classes.py --classes 20 --fast
  python backend/scripts/audit_draft_classes.py --classes 100 --full
  python backend/scripts/audit_draft_classes.py --force-transcendent
  python backend/scripts/audit_draft_classes.py --force-goalie-class weak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.franchise_sim import build_draft_class_rankings  # noqa: E402
from services.draft_ranking_logic import compute_tank_pressure_for_team  # noqa: E402
from services.draft_audit_session import create_audit_session  # noqa: E402
import run_sim as rs  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import (  # noqa: E402
    ensure_draft_pick_registry,
    team_owns_own_first,
    transfer_pick,
    canonical_pick_id,
)

GOALIE_CLASS_FORCE_VALUES = ("weak", "normal", "strong", "elite", "generational")


def _eta_years(row: Dict[str, Any]) -> int:
    eta = row.get("eta") or row.get("eta_years")
    if isinstance(eta, dict):
        return int(eta.get("years") if eta.get("years") is not None else 99)
    if isinstance(eta, (int, float)):
        return int(eta)
    return 99


def _true_pot(row: Dict[str, Any]) -> float:
    return float(row.get("true_potential_score") or row.get("potential_score") or 0)


def _low_pot_top32_violation(row: Dict[str, Any]) -> bool:
    pot = _true_pot(row)
    if pot >= 60:
        return False
    reason = str(row.get("ranking_reason") or row.get("ranking_flag") or "")
    if reason in ("production_only_risk", "nhl_ready_low_ceiling"):
        return False
    if row.get("generational_goalie") or row.get("is_transcendent"):
        return False
    ovr = float(row.get("true_ovr") or 0)
    if ovr >= 74 and float(row.get("production_adjusted_score") or 0) >= 1.0:
        return False
    return True


def _audit_one_class(session: Any, *, label: str, debug: bool = False) -> Dict[str, Any]:
    board = build_draft_class_rankings(session, session.sim)
    entries: List[Dict[str, Any]] = list(board.get("entries") or [])
    pipeline = dict(board.get("goalie_pipeline") or {})

    goalies = [e for e in entries if str(e.get("position") or "").upper() == "G"]
    top_g_ranks = [i + 1 for i, e in enumerate(entries) if str(e.get("position") or "").upper() == "G"]
    top_goalie_rank = min(top_g_ranks) if top_g_ranks else 0
    top10 = entries[:10]
    top32 = entries[:32]
    top20 = entries[:20]
    transcendent = [e for e in entries if e.get("is_transcendent")]

    low_pot_top20_true = [e for e in top20 if _true_pot(e) < 70]
    low_pot_top20_visible = [e for e in top20 if float(e.get("potential_score") or 0) < 70]
    low_pot_top32_true = [e for e in top32 if _low_pot_top32_violation(e)]

    short_top10 = []
    for e in top10:
        if str(e.get("position") or "").upper() in ("G", "D", "LD", "RD"):
            continue
        h = int(e.get("height_cm") or 0)
        prod = float(e.get("production_adjusted_score") or e.get("ppg") or 0)
        if 0 < h < 176 and prod < 1.0 and not e.get("is_transcendent"):
            short_top10.append(e)

    eta_top10_4y = []
    for e in top10:
        if str(e.get("position") or "").upper() == "G":
            continue
        if _eta_years(e) >= 4:
            eta_top10_4y.append(e)
            if debug:
                eta = e.get("eta") or {}
                print(
                    "ETA_VIOLATION",
                    e.get("name"),
                    "rank", e.get("rank"),
                    "pos", e.get("position"),
                    "ovr", e.get("true_ovr"),
                    "pot", _true_pot(e),
                    "age", e.get("age"),
                    "league", e.get("league_code"),
                    "eta", eta,
                )

    bad_team_names = [
        e for e in entries[:64]
        if "EU_J" in str(e.get("team_name") or "").upper()
        or " CHL " in str(e.get("team_name") or "").upper()
    ]

    low_pot_debug = []
    for e in low_pot_top32_true[:5]:
        low_pot_debug.append({
            "rank": e.get("rank"),
            "name": e.get("name"),
            "true_pot": _true_pot(e),
            "visible_pot": e.get("potential_score"),
            "ovr": e.get("true_ovr"),
            "reason": e.get("ranking_reason"),
        })

    gclass = str(pipeline.get("goalie_class_strength") or getattr(session.sim.league, "goalie_class_strength", "normal"))

    return {
        "label": label,
        "total": len(entries),
        "goalie_count": len(goalies),
        "top_goalie_rank": top_goalie_rank,
        "goalies_top32": sum(1 for e in entries[:32] if str(e.get("position") or "").upper() == "G"),
        "goalies_top10": sum(1 for e in entries[:10] if str(e.get("position") or "").upper() == "G"),
        "goalies_top64": sum(1 for e in entries[:64] if str(e.get("position") or "").upper() == "G"),
        "goalie_class_strength": gclass,
        "top10_pot_avg_true": round(sum(_true_pot(e) for e in top10) / max(1, len(top10)), 2),
        "top32_pot_min_true": min((_true_pot(e) for e in top32), default=0),
        "low_pot_top20_true": len(low_pot_top20_true),
        "low_pot_top20_visible": len(low_pot_top20_visible),
        "low_pot_top32_true": len(low_pot_top32_true),
        "top32_low_potential_exception_count": sum(
            1 for e in top32
            if _true_pot(e) < 64 and not _low_pot_top32_violation(e)
        ),
        "short_top10": len(short_top10),
        "eta_top10_4y": len(eta_top10_4y),
        "bad_team_names": len(bad_team_names),
        "transcendent_count": len(transcendent),
        "goalie_pipeline": pipeline,
        "low_pot_debug": low_pot_debug,
    }


def _entity_tid(team: Any) -> str:
    try:
        return str(rs._team_id(team))
    except Exception:
        tid = getattr(team, "team_id", None)
        if tid is None:
            tid = getattr(team, "id", "")
        return str(tid or "")


def _forced_transcendent_checks() -> Dict[str, Any]:
    os.environ["NHL_FORCE_TRANSCENDENT"] = "1"
    try:
        from app.sim_engine import league_hierarchy_bootstrap as lhb

        lhb.TRANSCENDENT_FORCE_DEBUG = True
    except Exception:
        pass

    session = create_audit_session(99991, fast=True)
    league = session.sim.league
    board = build_draft_class_rankings(session, session.sim)
    entries = board.get("entries") or []
    transcendent = [e for e in entries if e.get("is_transcendent")]
    rank_ok = bool(transcendent) and int(transcendent[0].get("rank") or 99) == 1

    from services.draft_prospect_profile import build_prospect_profile

    profile = build_prospect_profile(transcendent[0]) if transcendent else {}
    fx_ok = bool((profile.get("special_fx") or {}).get("halo"))
    origin_ok = bool((profile.get("origin_story") or {}).get("full_text"))
    storyline_ok = any(
        str(s.get("type") or "") == "TRANSCENDENT_DRAFT_PROSPECT"
        for s in (getattr(session, "storyline_events", None) or [])
    )
    tank_map = getattr(session, "transcendent_tank_pressure", None) or {}
    tank_ok = bool(tank_map)

    ensure_draft_pick_registry(league, start_year=2026, years_ahead=4)
    teams = list(getattr(league, "teams", None) or [])
    if len(teams) >= 2:
        traded_team = teams[0]
        receiver = teams[1]
        tid = _entity_tid(traded_team)
        rid = _entity_tid(receiver)
        pick_id = canonical_pick_id(2027, 1, tid)
        transfer_pick(league, pick_id, rid)
        capped = compute_tank_pressure_for_team(
            traded_team,
            transcendent_present=True,
            owns_own_first=False,
            pick_ownership_reason="pick_traded",
        )
        no_hard = capped.get("tank_mode") != "hard_tank"
    else:
        no_hard = True

    contender = next((t for t in teams if "contend" in str(getattr(t, "gm_window", "")).lower()), teams[-1] if teams else None)
    if contender is not None:
        setattr(contender, "team_status", "playoff_contender")
        low = compute_tank_pressure_for_team(contender, transcendent_present=True, owns_own_first=True)
        contender_low = int(low.get("tank_pressure") or 0) < 30
    else:
        contender_low = True

    return {
        "ranks_first": rank_ok,
        "profile_special_fx": fx_ok,
        "profile_origin_story": origin_ok,
        "storyline_created": storyline_ok,
        "tank_pressure_created": tank_ok,
        "traded_pick_no_hard_tank": no_hard,
        "contender_low_pressure": contender_low,
        "passed": all([rank_ok, fx_ok, origin_ok, storyline_ok, tank_ok, no_hard, contender_low]),
    }


def _goalie_class_expectations(gclass: str, row: Dict[str, Any]) -> Dict[str, Any]:
    gclass = gclass.lower()
    top32_cap = {"weak": 0, "normal": 1, "strong": 2, "elite": 3, "generational": 3}.get(gclass, 1)
    top10_cap = {"weak": 0, "normal": 0, "strong": 1, "elite": 1, "generational": 2}.get(gclass, 0)
    top_g_ok = True
    if gclass == "weak":
        top_g_ok = row["goalies_top32"] == 0 and row["top_goalie_rank"] >= 40
    elif gclass == "normal":
        top_g_ok = row["goalies_top32"] <= 1 and row["top_goalie_rank"] >= 20
    elif gclass == "strong":
        top_g_ok = row["goalies_top32"] <= 2 and row["top_goalie_rank"] >= 12
    elif gclass == "elite":
        top_g_ok = row["goalies_top32"] <= 3 and row["goalies_top10"] <= 1
        if row["top_goalie_rank"] > 0:
            top_g_ok = top_g_ok and row["top_goalie_rank"] <= 20
    elif gclass == "generational":
        top_g_ok = row["goalies_top32"] <= 3 and row["goalies_top10"] <= 2
        if row["top_goalie_rank"] > 0:
            top_g_ok = top_g_ok and row["top_goalie_rank"] <= 5
    visible_ok = row["goalie_count"] >= 12
    pot_ok = row["low_pot_top32_true"] == 0 and row["top32_pot_min_true"] >= 60
    return {
        "visible_goalies_ok": visible_ok,
        "top_goalie_distribution_ok": top_g_ok,
        "top32_cap": top32_cap,
        "top10_cap": top10_cap,
        "pot_floor_ok": pot_ok,
        "passed": visible_ok and top_g_ok and pot_ok,
    }


def run_forced_goalie_class(gclass: str, *, classes: int = 5, fast: bool = True) -> Dict[str, Any]:
    gclass = gclass.lower()
    if gclass not in GOALIE_CLASS_FORCE_VALUES:
        raise ValueError(f"Unknown goalie class: {gclass}")

    os.environ["NHL_FORCE_GOALIE_CLASS"] = gclass
    os.environ.pop("NHL_FORCE_TRANSCENDENT", None)
    rows: List[Dict[str, Any]] = []
    try:
        for i in range(classes):
            seed = 50_000 + i
            session = create_audit_session(seed, fast=fast)
            audit = _audit_one_class(session, label=f"{gclass}_{i}")
            expectations = _goalie_class_expectations(gclass, audit)
            audit["expectations"] = expectations
            rows.append(audit)
    finally:
        os.environ.pop("NHL_FORCE_GOALIE_CLASS", None)

    summary = {
        "forced_goalie_class": gclass,
        "classes_tested": classes,
        "goalie_count_min": min(r["goalie_count"] for r in rows),
        "goalie_count_avg": round(sum(r["goalie_count"] for r in rows) / len(rows), 2),
        "top_goalie_rank_avg": round(
            sum(r["top_goalie_rank"] for r in rows if r["top_goalie_rank"]) / max(1, len([r for r in rows if r["top_goalie_rank"]])),
            2,
        ),
        "goalies_top32_avg": round(sum(r["goalies_top32"] for r in rows) / len(rows), 2),
        "goalies_top10_avg": round(sum(r["goalies_top10"] for r in rows) / len(rows), 2),
        "top32_pot_min_worst": min(r["top32_pot_min_true"] for r in rows),
        "low_potential_top32_violations": sum(r["low_pot_top32_true"] for r in rows),
        "per_class": rows,
        "passed": sum(1 for r in rows if r["expectations"]["passed"]) >= max(4, len(rows) - 1),
    }
    return summary


def run_audit(
    *,
    classes: int = 100,
    force_transcendent: bool = False,
    force_goalie_class: Optional[str] = None,
    fast: bool = True,
) -> Dict[str, Any]:
    os.environ.pop("NHL_FORCE_TRANSCENDENT", None)
    os.environ.pop("NHL_FORCE_GOALIE_CLASS", None)

    if force_goalie_class:
        return run_forced_goalie_class(force_goalie_class, classes=min(5, classes), fast=fast)

    if force_transcendent:
        os.environ["NHL_FORCE_TRANSCENDENT"] = "1"
        forced = _forced_transcendent_checks()
        os.environ.pop("NHL_FORCE_TRANSCENDENT", None)
        return {"forced_transcendent": forced, "passed": bool(forced.get("passed"))}

    rows: List[Dict[str, Any]] = []
    for i in range(classes):
        seed = 10_000 + i
        session = create_audit_session(seed, fast=fast)
        rows.append(_audit_one_class(session, label=f"class_{i}", debug=(i == 0)))

    goalie_counts = [r["goalie_count"] for r in rows]
    top_g_ranks = [r["top_goalie_rank"] for r in rows if r["top_goalie_rank"]]
    pipeline_avg = {
        "draft_eligible_dev_goalies": round(
            sum((r.get("goalie_pipeline") or {}).get("draft_eligible_dev_goalies", 0) for r in rows)
            / max(1, len(rows)),
            1,
        ),
        "goalies_on_live_board": round(
            sum((r.get("goalie_pipeline") or {}).get("goalies_on_live_board", 0) for r in rows)
            / max(1, len(rows)),
            1,
        ),
    }

    class_dist: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    class_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        gclass = str(r.get("goalie_class_strength") or "normal")
        class_counts[gclass] += 1
        class_dist[gclass]["goalie_count"] += r["goalie_count"]
        class_dist[gclass]["top_goalie_rank"] += r["top_goalie_rank"]
        class_dist[gclass]["goalies_top32"] += r["goalies_top32"]
        class_dist[gclass]["goalies_top10"] += r["goalies_top10"]
    goalie_rank_distribution_by_class = {}
    for gclass, totals in class_dist.items():
        n = max(1, class_counts[gclass])
        goalie_rank_distribution_by_class[gclass] = {
            "samples": class_counts[gclass],
            "goalie_count_avg": round(totals["goalie_count"] / n, 2),
            "top_goalie_rank_avg": round(totals["top_goalie_rank"] / n, 2),
            "goalies_top32_avg": round(totals["goalies_top32"] / n, 2),
            "goalies_top10_avg": round(totals["goalies_top10"] / n, 2),
        }

    summary = {
        "mode": "fast" if fast else "full",
        "classes_tested": classes,
        "goalie_count_min": min(goalie_counts) if goalie_counts else 0,
        "goalie_count_avg": round(sum(goalie_counts) / max(1, len(goalie_counts)), 2),
        "top_goalie_rank_avg": round(sum(top_g_ranks) / max(1, len(top_g_ranks)), 2) if top_g_ranks else 0,
        "goalies_top32_avg": round(sum(r["goalies_top32"] for r in rows) / max(1, len(rows)), 2),
        "goalies_top10_avg": round(sum(r["goalies_top10"] for r in rows) / max(1, len(rows)), 2),
        "goalies_top64_avg": round(sum(r["goalies_top64"] for r in rows) / max(1, len(rows)), 2),
        "goalie_rank_distribution_by_class": goalie_rank_distribution_by_class,
        "top10_pot_avg_true": round(sum(r["top10_pot_avg_true"] for r in rows) / max(1, len(rows)), 2),
        "top32_pot_min_worst": min((r["top32_pot_min_true"] for r in rows), default=0),
        "low_potential_top20_violations": sum(r["low_pot_top20_true"] for r in rows),
        "low_potential_top32_violations": sum(r["low_pot_top32_true"] for r in rows),
        "top32_low_potential_exception_count": sum(r["top32_low_potential_exception_count"] for r in rows),
        "low_potential_top20_visible_violations": sum(r["low_pot_top20_visible"] for r in rows),
        "short_top10_violations": sum(r["short_top10"] for r in rows),
        "eta_top10_4y_violations": sum(r["eta_top10_4y"] for r in rows),
        "team_name_format_violations": sum(r["bad_team_names"] for r in rows),
        "transcendent_occurrences": sum(r["transcendent_count"] for r in rows),
        "goalie_pipeline_avg": pipeline_avg,
        "sample_low_pot_debug": rows[0].get("low_pot_debug") if rows else [],
        "sample_goalie_pipeline": rows[0].get("goalie_pipeline") if rows else {},
    }
    summary["passed"] = (
        summary["low_potential_top20_violations"] == 0
        and summary["low_potential_top32_violations"] == 0
        and summary["top32_pot_min_worst"] >= 60
        and summary["short_top10_violations"] == 0
        and summary["eta_top10_4y_violations"] == 0
        and summary["team_name_format_violations"] == 0
        and summary["goalie_count_min"] >= 12
        and summary["goalie_count_avg"] >= 14
        and summary["goalies_top32_avg"] <= 2.5
        and summary["goalies_top10_avg"] <= 0.5
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft class large audit")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--force-transcendent", action="store_true")
    parser.add_argument("--force-goalie-class", choices=GOALIE_CLASS_FORCE_VALUES)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    args = parser.parse_args()
    fast = not args.full if args.full else (args.fast or True)
    result = run_audit(
        classes=max(1, args.classes),
        force_transcendent=args.force_transcendent,
        force_goalie_class=args.force_goalie_class,
        fast=fast,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
