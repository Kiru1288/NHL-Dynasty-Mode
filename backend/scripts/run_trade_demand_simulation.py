#!/usr/bin/env python3
"""Simulate trade stability, agents, and crisis timers across 8 player/team scenarios."""

from typing import Any, Dict, List, Optional
import json
import sys
import time
import types
from pathlib import Path

# Bootstrap SimEngine path
_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
from services.franchise_paths import ensure_simengine_path  # noqa: E402

ensure_simengine_path()

import random  # noqa: E402

from app.sim_engine.franchise.player_agent_engine import (  # noqa: E402
    agent_public_view,
    ensure_player_agent,
)
from app.sim_engine.franchise.trade_stability_engine import (  # noqa: E402
    apply_daily_stability_update,
    apply_trade_hub_exposure,
    formal_demand_eligible,
    gather_player_concerns,
    primary_complaint_from_pressures,
    stability_to_escalation_level,
    update_player_stability,
)
from services.trade_demand_engine import (  # noqa: E402
    build_trade_demand_crisis_payload,
    evaluate_trade_demand_ntc_waiver,
    open_trade_demand,
    process_trade_demand_day,
    sync_trade_demand_crises,
)


ESCALATION_LABELS = {
    0: "Stable",
    1: "Angst / Upset",
    2: "Apathy / Annoyance",
    3: "Anger / Trade Request",
    4: "Full Crisis",
}

CRISIS_STAGE_LABELS = {
    1: "Formal demand (best leverage)",
    2: "Leaking / public pressure",
    3: "Severe leverage loss",
    4: "Distressed asset",
}


def _rec(w: int, l: int, otl: int = 0):
    r = types.SimpleNamespace(wins=w, losses=l, ot_losses=otl)
    return r


def _player(
    pid: str,
    name: str,
    *,
    character: int,
    mental: int,
    ovr: float = 84,
    role_sat: float = 0.35,
    gm_trust: float = 0.55,
    coach_trust: float = 0.50,
    competitiveness: int = 75,
    loyalty: int = 55,
    ego: float = 0.55,
    trade_heat: int = 0,
    trade_attempts: int = 0,
    career_demands: int = 0,
    clause: str = "",
    approved_teams: Optional[List[str]] = None,
):
    approved = list(approved_teams or [])
    clause_upper = str(clause or "").upper()
    is_nmc = "NMC" in clause_upper and "M-NTC" not in clause_upper and "MNTC" not in clause_upper
    is_mntc = "M-NTC" in clause_upper or "MNTC" in clause_upper
    is_ntc = "NTC" in clause_upper and not is_nmc and not is_mntc
    contract = types.SimpleNamespace(
        clause=clause,
        clause_type=clause,
        trade_clause=clause,
        years_remaining=3,
        term=3,
        no_move_clause=is_nmc,
        no_trade_clause=is_ntc,
        modified_no_trade_teams=len(approved) if is_mntc else 0,
        trade_list_size=len(approved) if approved else (10 if is_mntc else 0),
        approved_trade_teams=approved,
        approved_destinations=approved,
    )
    p = types.SimpleNamespace(
        id=pid,
        player_id=pid,
        name=name,
        character=character,
        mental=mental,
        retired=False,
        identity=types.SimpleNamespace(name=name, age=28),
        psych=types.SimpleNamespace(
            morale=0.45,
            confidence=0.50,
            role_satisfaction=role_sat,
            coach_trust=coach_trust,
        ),
        traits=types.SimpleNamespace(
            ego=ego,
            competitiveness=competitiveness / 100.0,
            coachability=character / 100.0,
            mental_toughness=mental / 100.0,
            work_ethic=0.55,
            leadership=0.50,
            volatility=0.35,
            patience=0.45,
        ),
        chemistry_profile={
            "competitiveness": competitiveness,
            "loyalty": loyalty,
            "adaptability": mental,
            "belonging": 50,
        },
        contract=contract,
        _franchise_storyline_state={
            "gm_trust": gm_trust,
            "trade_rumor_heat": trade_heat,
            "trade_attempt_count": trade_attempts,
            "career_trade_demand_count": career_demands,
            "was_recently_shopped": trade_attempts > 0,
        },
    )
    p.ovr = lambda o=ovr: o
    return p


def _team(tid: str, name: str, abbr: str, roster, *, situation: str):
    t = types.SimpleNamespace(team_id=tid, id=tid, abbr=abbr, name=name, roster=roster, situation=situation)
    return t


def build_scenarios():
    """Eight players on eight teams in different competitive contexts."""
    specs = [
        {
            "pid": "p1_kucherov",
            "name": "Nikita Kucherov",
            "team_id": "TBL",
            "team_name": "Tampa Bay Lightning",
            "abbr": "TBL",
            "situation": "Contending (Cup favorite)",
            "record": (42, 18, 4),
            "character": 92,
            "mental": 94,
            "role_sat": 0.28,
            "gm_trust": 0.78,
            "competitiveness": 95,
            "loyalty": 72,
            "ovr": 91,
            "shop": False,
            "clause": "NMC",
        },
        {
            "pid": "p2_bedard",
            "name": "Connor Bedard",
            "team_id": "CHI",
            "team_name": "Chicago Blackhawks",
            "abbr": "CHI",
            "situation": "Rebuilding (bottom feeder)",
            "record": (14, 42, 6),
            "character": 58,
            "mental": 55,
            "role_sat": 0.62,
            "gm_trust": 0.48,
            "competitiveness": 88,
            "loyalty": 45,
            "ego": 0.78,
            "ovr": 86,
            "shop": False,
            "clause": "",
        },
        {
            "pid": "p3_matthews",
            "name": "Auston Matthews",
            "team_id": "TOR",
            "team_name": "Toronto Maple Leafs",
            "abbr": "TOR",
            "situation": "Contending (playoff pressure)",
            "record": (38, 24, 4),
            "character": 78,
            "mental": 72,
            "role_sat": 0.18,
            "gm_trust": 0.52,
            "competitiveness": 90,
            "loyalty": 50,
            "ovr": 92,
            "shop": True,
            "trade_attempts": 2,
            "trade_heat": 35,
            "clause": "M-NTC",
            "approved_teams": ["TBL", "EDM", "VGK", "COL", "DAL", "FLA", "CAR"],
        },
        {
            "pid": "p4_suzuki",
            "name": "Nick Suzuki",
            "team_id": "MTL",
            "team_name": "Montreal Canadiens",
            "abbr": "MTL",
            "situation": "Losing (market pressure)",
            "record": (18, 38, 6),
            "character": 68,
            "mental": 60,
            "role_sat": 0.40,
            "gm_trust": 0.38,
            "competitiveness": 82,
            "loyalty": 68,
            "ovr": 84,
            "shop": True,
            "trade_attempts": 3,
            "trade_heat": 55,
            "clause": "NTC",
        },
        {
            "pid": "p5_mcdavid",
            "name": "Connor McDavid",
            "team_id": "EDM",
            "team_name": "Edmonton Oilers",
            "abbr": "EDM",
            "situation": "Contending (winning, star role)",
            "record": (40, 20, 6),
            "character": 88,
            "mental": 90,
            "role_sat": 0.82,
            "gm_trust": 0.85,
            "competitiveness": 96,
            "loyalty": 80,
            "ovr": 96,
            "shop": False,
            "clause": "",
        },
        {
            "pid": "p6_keller",
            "name": "Clayton Keller",
            "team_id": "UTA",
            "team_name": "Utah Hockey Club",
            "abbr": "UTA",
            "situation": "Rebuilding (small market)",
            "record": (16, 40, 6),
            "character": 55,
            "mental": 58,
            "role_sat": 0.55,
            "gm_trust": 0.42,
            "competitiveness": 70,
            "loyalty": 38,
            "ego": 0.72,
            "ovr": 85,
            "shop": False,
            "clause": "NTC",
        },
        {
            "pid": "p7_panarin",
            "name": "Artemi Panarin",
            "team_id": "NYR",
            "team_name": "New York Rangers",
            "abbr": "NYR",
            "situation": "Bubble / inconsistent",
            "record": (28, 28, 6),
            "character": 74,
            "mental": 68,
            "role_sat": 0.25,
            "gm_trust": 0.35,
            "coach_trust": 0.32,
            "competitiveness": 85,
            "loyalty": 42,
            "ovr": 88,
            "shop": False,
            "clause": "M-NTC",
            "approved_teams": ["NYR", "TBL", "TOR", "EDM", "VGK"],
        },
        {
            "pid": "p8_miller",
            "name": "J.T. Miller",
            "team_id": "VAN",
            "team_name": "Vancouver Canucks",
            "abbr": "VAN",
            "situation": "Underachieving contender",
            "record": (18, 38, 6),
            "character": 64,
            "mental": 72,
            "role_sat": 0.22,
            "gm_trust": 0.35,
            "coach_trust": 0.30,
            "competitiveness": 92,
            "loyalty": 48,
            "ego": 0.68,
            "ovr": 87,
            "career_demands": 2,
            "shop": False,
            "clause": "NTC",
        },
    ]

    window_by_situation = {
        "Contending": "contending",
        "Rebuilding": "rebuild",
        "Losing": "retool",
        "Bubble": "retool",
        "Underachieving": "retool",
    }

    teams = []
    standings = {}
    player_map = {}

    for sp in specs:
        ego = sp.get("ego", 0.55)
        player = _player(
            sp["pid"],
            sp["name"],
            character=sp["character"],
            mental=sp["mental"],
            ovr=sp["ovr"],
            role_sat=sp["role_sat"],
            gm_trust=sp["gm_trust"],
            coach_trust=sp.get("coach_trust", 0.50),
            competitiveness=sp["competitiveness"],
            loyalty=sp["loyalty"],
            ego=ego,
            trade_heat=sp.get("trade_heat", 0),
            trade_attempts=sp.get("trade_attempts", 0),
            career_demands=sp.get("career_demands", 0),
            clause=str(sp.get("clause") or ""),
            approved_teams=list(sp.get("approved_teams") or []),
        )
        player_map[sp["pid"]] = player
        window = "contending"
        for key, val in window_by_situation.items():
            if key in sp["situation"]:
                window = val
                break
        team = _team(sp["team_id"], sp["team_name"], sp["abbr"], [player], situation=sp["situation"])
        team.gm_window = window
        teams.append(team)
        w, l, otl = sp["record"]
        standings[sp["team_id"]] = _rec(w, l, otl)
        sp["_player"] = player
        sp["_team"] = team
        sp["_shop"] = sp.get("shop", False)

    league = types.SimpleNamespace(teams=teams)
    session = types.SimpleNamespace(
        user_team_id="TOR",
        trade_demands={},
        trade_stability_state={},
        pending_ui_popups=[],
        storyline_events=[],
        notifications=[],
        standings=types.SimpleNamespace(records=standings),
        agent_relationships={},
        sim=types.SimpleNamespace(league=league, rng=random.Random(42)),
        calendar_cursor=55,
        season_calendar_year=2025,
    )
    return session, specs, player_map


def _top_pressures(pressures: dict, n: int = 3):
    items = sorted((pressures or {}).items(), key=lambda kv: -kv[1])[:n]
    return [{"component": k, "pressure": round(v, 2)} for k, v in items]


def run_simulation():
    session, specs, player_map = build_scenarios()
    rng = session.sim.rng

    print("=" * 78)
    print("TRADE DEMAND SYSTEM — 8 PLAYER / 8 TEAM SIMULATION")
    print("=" * 78)
    print()

    # Phase 1: extended stability drift (days 42–80)
    print("PHASE 1: Daily Stability Drift (days 42–80)")
    print("-" * 78)
    baseline_rows = []

    for day in range(42, 81):
        for sp in specs:
            apply_daily_stability_update(session, sp["_player"], sp["_team"], day)

    for sp in specs:
        player = sp["_player"]
        team = sp["_team"]
        ensure_player_agent(player, session)
        agent = agent_public_view(player, session)
        snap = gather_player_concerns(session, player, team)
        row = session.trade_stability_state.get(sp["pid"], {})
        level = int(row.get("escalation_level") or 0)
        clause = getattr(getattr(player, "contract", None), "clause", "") or "None"
        baseline_rows.append(
            {
                "player": sp["name"],
                "team": sp["abbr"],
                "situation": sp["situation"],
                "character": sp["character"],
                "mental": sp["mental"],
                "clause": clause,
                "agent": agent.get("name"),
                "agent_style": agent.get("style_label"),
                "win_pct": round(snap.winning_satisfaction, 1),
                "role_sat": round(snap.role_satisfaction, 1),
                "gm_trust": round(snap.gm_trust, 1),
                "stability": row.get("trade_stability_score"),
                "target": row.get("target_stability_score"),
                "escalation": level,
                "escalation_label": ESCALATION_LABELS.get(level, "?"),
                "top_pressures": _top_pressures(row.get("pressures")),
                "readiness_ovr": (row.get("readiness_penalties") or {}).get("ovr_readiness", 0),
            }
        )

    for r in sorted(baseline_rows, key=lambda x: float(x["stability"] or 0)):
        print(
            f"  {r['player']:<22} ({r['team']}) {r['situation'][:28]:<28} "
            f"CHR={r['character']} MNT={r['mental']} {str(r['clause']):<5} "
            f"Stability={r['stability']:>5} (target {r.get('target', '?')})  L{r['escalation']} {r['escalation_label']}"
        )
        print(f"    Agent: {r['agent']} ({r['agent_style']})")
        print(
            f"    Context: win_sat={r['win_pct']} role={r['role_sat']} gm_trust={r['gm_trust']}  "
            f"OVR pen={r['readiness_ovr']}"
        )
        tops = ", ".join(f"{p['component']}={p['pressure']}" for p in r["top_pressures"])
        print(f"    Top pressures: {tops}")
        print()

    # Phase 2: Trade hub exposure for shopped players
    print("PHASE 2: Trade Hub Exposure (rejected proposal)")
    print("-" * 78)
    for sp in specs:
        if not sp.get("_shop"):
            continue
        player = sp["_player"]
        before = float(session.trade_stability_state.get(sp["pid"], {}).get("trade_stability_score") or 0)
        result = apply_trade_hub_exposure(session, player, attempt_n=sp.get("trade_attempts", 1), rejection_kind="rejected")
        team = sp["_team"]
        row = update_player_stability(session, player, team)
        after = float(row.get("trade_stability_score") or 0)
        print(
            f"  {sp['name']} ({sp['abbr']}): stability {before:.1f} -> {after:.1f}  "
            f"(delta {result.get('stability_delta', 0):+.2f}, heat +{result.get('heat_delta', 0)})  "
            f"-> L{row.get('escalation_level')} {ESCALATION_LABELS.get(int(row.get('escalation_level') or 0))}"
        )
    print()

    # Phase 3: Formal demands via unified daily processor
    print("PHASE 3: Formal Trade Demands + Crisis Timer Start")
    print("-" * 78)
    day_result = process_trade_demand_day(session, 55, {"iso": "2025-02-15"})
    print(f"  Daily pass: {day_result.get('opened')} formal demand(s), {day_result.get('warnings')} warning(s)")
    demands_opened = []

    for sp in specs:
        demand = session.trade_demands.get(sp["pid"])
        if not isinstance(demand, dict) or demand.get("status") != "open":
            st = session.trade_stability_state.get(sp["pid"], {})
            escalation = int(st.get("escalation_level") or 0)
            eligible = formal_demand_eligible(st)
            print(f"  {sp['name']}: no formal demand (L{escalation}, eligible={eligible})")
            continue

        crisis = demand.get("crisis") or {}
        agent = demand.get("agent") or {}
        demands_opened.append((sp, demand))
        clause = getattr(getattr(sp["_player"], "contract", None), "clause", "") or "None"
        print(f"  {sp['name']} ({sp['abbr']}) — FORMAL DEMAND [{clause}]")
        print(f"    Agent: {agent.get('name')} ({agent.get('style_label')})")
        print(f"    Complaint: {demand.get('primary_complaint')}")
        print(f"    Stability: {demand.get('trade_stability_score')}  Reason: {demand.get('reason_code')}")
        print(
            f"    Crisis timer: {crisis.get('initial_seconds')}s start "
            f"({crisis.get('initial_seconds', 360) // 60}:{crisis.get('initial_seconds', 360) % 60:02d})"
        )
        print(f"    Trade value: {demand.get('value_before')} -> {demand.get('value_after')}")
        print(f"    Destinations ({demand.get('destination_count')}): {', '.join(demand.get('preferred_destinations') or [])[:80]}")
        print()

    if not demands_opened:
        print("  (No organic formal demands — seeding NTC/crisis lab cases)")
        print()
        lab_pids = ("p8_miller", "p4_suzuki", "p3_matthews", "p6_keller")
        for sp in specs:
            if sp["pid"] not in lab_pids:
                continue
            st = session.trade_stability_state.get(sp["pid"], {})
            if int(st.get("escalation_level") or 0) < 2:
                st = update_player_stability(session, sp["_player"], sp["_team"])
                st["escalation_level"] = max(3, int(st.get("escalation_level") or 0))
                st["trade_stability_score"] = min(float(st.get("trade_stability_score") or 50), 28.0)
                session.trade_stability_state[sp["pid"]] = st
            demand = open_trade_demand(
                session,
                sp["_player"],
                sp["_team"],
                reason="role",
                calendar_idx=80,
                rng=rng,
                stability_row=st,
                force_formal=True,
            )
            if demand.get("status") == "open":
                demands_opened.append((sp, demand))
                clause = getattr(getattr(sp["_player"], "contract", None), "clause", "") or "None"
                crisis = demand.get("crisis") or {}
                agent = demand.get("agent") or {}
                print(f"  {sp['name']} ({sp['abbr']}) — LAB FORMAL DEMAND [{clause}]")
                print(f"    Agent: {agent.get('name')} ({agent.get('style_label')})")
                print(f"    Crisis timer: {crisis.get('initial_seconds')}s")
                print()

    # Phase 4: Crisis timer tick simulation
    print("PHASE 4: Crisis Timer Progression (simulated real-time)")
    print("-" * 78)

    tick_points = [
        (0, "Demand delivered"),
        (120, "2:00 elapsed"),
        (240, "4:00 elapsed"),
        (360, "6:00 — deadline"),
    ]

    for sp, demand in demands_opened:
        if str(demand.get("team_id")) != session.user_team_id:
            continue  # focus user-team crisis for ticker demo

        pid = sp["pid"]
        print(f"  Tracking crisis: {sp['name']} (user team {sp['abbr']})")
        book = session.trade_demands.get(pid, {})
        crisis = book.get("crisis") or {}
        initial = int(crisis.get("initial_seconds") or 360)

        for elapsed, label in tick_points:
            if elapsed > initial and elapsed > 0:
                continue
            remaining = max(0, initial - elapsed)
            crisis["remaining_seconds"] = remaining
            crisis["last_sync_unix"] = time.time()
            book["crisis"] = crisis
            sync_trade_demand_crises(session, elapsed_hint=0)

            stage = int(book.get("crisis_stage") or 1)
            payload = build_trade_demand_crisis_payload(session)
            print(
                f"    [{label}] remaining={remaining:>3}s  stage={stage} "
                f"({CRISIS_STAGE_LABELS.get(stage, '?')})  "
                f"TV after={book.get('value_after')}  dests={book.get('destination_count')}"
            )
            if book.get("leaked"):
                print("      >> Demand leaked to media")
            if book.get("public_demand"):
                print("      >> Public trade demand")
        print()

    # Also tick non-user demands briefly
    for sp, demand in demands_opened:
        if str(demand.get("team_id")) == session.user_team_id:
            continue
        pid = sp["pid"]
        book = session.trade_demands.get(pid, {})
        crisis = book.get("crisis") or {}
        initial = int(crisis.get("initial_seconds") or 360)
        crisis["remaining_seconds"] = max(0, initial - (initial // 2))
        crisis["last_sync_unix"] = time.time()
        sync_trade_demand_crises(session)
        print(
            f"  {sp['name']} ({sp['abbr']}) at 50% timer ({crisis.get('remaining_seconds')}s left): "
            f"stage={book.get('crisis_stage')} TV={book.get('value_after')} dests={book.get('destination_count')}"
        )

    print()
    print("PHASE 6: NTC / M-NTC Waiver Willingness During Crisis")
    print("-" * 78)
    team_by_abbr = {sp["abbr"]: sp["_team"] for sp in specs}
    probe_destinations = ["TBL", "EDM", "CHI", "UTA", "NYR"]

    for sp, demand in demands_opened:
        player = sp["_player"]
        source = sp["_team"]
        clause = getattr(getattr(player, "contract", None), "clause", "") or "None"
        print(f"  {sp['name']} ({sp['abbr']}) — clause={clause}, crisis stage={demand.get('crisis_stage')}")
        approved = set(demand.get("preferred_destinations") or [])
        for dest_abbr in probe_destinations:
            if dest_abbr == sp["abbr"]:
                continue
            dest_team = team_by_abbr.get(dest_abbr)
            if dest_team is None:
                # synthetic destination team for cross-league probe
                dest_team = _team(dest_abbr, dest_abbr, dest_abbr, [], situation="Probe")
                dest_team.gm_window = "contending" if dest_abbr in ("TBL", "EDM", "NYR") else "rebuild"
            bucket = "approved" if dest_abbr in approved else "blocked"
            result = evaluate_trade_demand_ntc_waiver(
                session, player, source, dest_team, demand_row=demand,
            )
            if result.get("reason_code") == "no_ntc":
                print(f"    {dest_abbr}: no waiver needed")
                continue
            if result.get("reason_code") == "nmc_hard_block":
                print(f"    {dest_abbr}: NMC — cannot waive")
                continue
            if result.get("reason_code") == "mntc_approved":
                print(f"    {dest_abbr}: M-NTC approved list — auto yes")
                continue
            verdict = "WAIVE YES" if result.get("accepted") else "WAIVE NO"
            print(
                f"    {dest_abbr} ({bucket}): {verdict}  "
                f"chance={result.get('accept_chance')}  reason={result.get('reason_code')}"
            )
        snap = demand.get("ntc_waiver_snapshot") or {}
        if snap.get("samples"):
            print(f"    Snapshot samples: {len(snap.get('samples') or [])} evaluated")
        print()

    print("PHASE 5: Summary Matrix")
    print("-" * 78)
    summary = []
    for sp in specs:
        pid = sp["pid"]
        st = session.trade_stability_state.get(pid, {})
        demand = session.trade_demands.get(pid, {})
        agent = agent_public_view(sp["_player"], session)
        summary.append(
            {
                "player": sp["name"],
                "team": sp["abbr"],
                "situation": sp["situation"],
                "character": sp["character"],
                "mental": sp["mental"],
                "agent": agent.get("name"),
                "stability": st.get("trade_stability_score"),
                "escalation": st.get("escalation_level"),
                "formal_demand": demand.get("status") == "open",
                "crisis_seconds": (demand.get("crisis") or {}).get("initial_seconds"),
                "crisis_stage": demand.get("crisis_stage"),
                "value_after": demand.get("value_after"),
            }
        )

    print(json.dumps(summary, indent=2))
    print()
    print("Done.")


if __name__ == "__main__":
    run_simulation()
