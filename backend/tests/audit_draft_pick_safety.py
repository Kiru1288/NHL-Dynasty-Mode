"""
Draft-pick ownership and valuation audit runner.

Run:
  python backend/tests/audit_draft_pick_safety.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.franchise_sim import start_franchise
from services.trade_service import build_trade_assets_payload
from services.franchise_entry_draft import (
    build_full_draft_order,
    execute_cpu_draft_pick,
    execute_user_draft_pick,
    get_entry_draft_payload,
    initialize_entry_draft,
)
from app.sim_engine.trades.trade_pick_registry import (
    audit_pick_registry_integrity,
    ensure_draft_pick_registry,
    get_pick_by_id,
    transfer_pick,
)
from app.sim_engine.trades.trade_value import evaluate_pick_asset_value


def _ok(msg: str) -> None:
    print(f"PASS {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise AssertionError(msg)


def _team_id(team: Any) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    return str(tid) if tid is not None else ""


def _all_unresolved_rows(league: Any) -> List[Dict[str, Any]]:
    reg = dict(getattr(league, "draft_pick_registry", {}) or {})
    out: List[Dict[str, Any]] = []
    for row in reg.values():
        if isinstance(row, dict) and not bool(row.get("resolved")):
            out.append(row)
    return out


def _assert_registry_shape(league: Any) -> None:
    reg = dict(getattr(league, "draft_pick_registry", {}) or {})
    seen: set[str] = set()
    for pid, row in reg.items():
        if pid in seen:
            _fail(f"duplicate pick id in registry: {pid}")
        seen.add(pid)
        for key in ("pick_id", "year", "round", "original_team_id", "current_owner_team_id", "resolved"):
            if key not in row:
                _fail(f"missing field {key} in pick row {pid}")
    _ok("registry rows include required fields")
    _ok("no duplicate pick IDs")


def _pick_owner_lists(league: Any) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for tm in list(getattr(league, "teams", None) or []):
        tid = _team_id(tm)
        out[tid] = set(str(x) for x in (getattr(tm, "owned_pick_ids", None) or []))
    return out


def _assert_one_owner_per_unresolved_pick(league: Any) -> None:
    owner_lists = _pick_owner_lists(league)
    for row in _all_unresolved_rows(league):
        pid = str(row.get("pick_id") or "")
        matches = [tid for tid, ids in owner_lists.items() if pid in ids]
        if len(matches) != 1:
            _fail(f"pick {pid} expected in exactly one owned_pick_ids list, found {matches}")
        owner = str(row.get("current_owner_team_id") or "")
        if matches[0] != owner:
            _fail(f"pick {pid} owner mismatch list={matches[0]} registry={owner}")
    _ok("every unresolved pick exists in exactly one team owned_pick_ids")


def _assert_expected_pick_count(league: Any, start_year: int, years_ahead: int = 4, rounds: int = 7) -> None:
    by_owner: Dict[Tuple[str, int], int] = {}
    for row in _all_unresolved_rows(league):
        owner = str(row.get("current_owner_team_id") or "")
        year = int(row.get("year") or 0)
        by_owner[(owner, year)] = by_owner.get((owner, year), 0) + 1
    for tm in list(getattr(league, "teams", None) or []):
        tid = _team_id(tm)
        for y in range(start_year, start_year + years_ahead):
            got = by_owner.get((tid, y), 0)
            if got != rounds:
                _fail(f"team {tid} expected {rounds} unresolved picks in {y}, found {got}")
    _ok("every team starts with expected picks per year/round")


def _find_tradeable_first(league: Any, draft_year: int) -> Dict[str, Any]:
    reg = dict(getattr(league, "draft_pick_registry", {}) or {})
    for row in reg.values():
        if not isinstance(row, dict):
            continue
        if bool(row.get("resolved")):
            continue
        if int(row.get("year") or 0) != int(draft_year):
            continue
        if int(row.get("round") or 0) != 1:
            continue
        return row
    raise AssertionError("no tradeable first-round pick found")


def _find_new_owner(league: Any, old_owner: str) -> str:
    for tm in list(getattr(league, "teams", None) or []):
        tid = _team_id(tm)
        if tid and tid != old_owner:
            return tid
    raise AssertionError("no new owner found")


def _find_slot_index(order: List[Dict[str, Any]], pick_id: str) -> int:
    for i, slot in enumerate(order):
        if str(slot.get("pick_id") or "") == str(pick_id):
            return i
    raise AssertionError(f"draft order slot not found for pick {pick_id}")


def _run_until_pick(session: Any, target_pick_id: str) -> Dict[str, Any]:
    # Advances draft to target slot and executes the pick at that slot.
    safety = 0
    while safety < 400:
        safety += 1
        state = dict(getattr(session, "draft_state", None) or {})
        order = list(state.get("draft_order") or [])
        overall = int(state.get("overall_pick") or 1)
        slot = order[overall - 1] if 1 <= overall <= len(order) else {}
        current_pick_id = str(slot.get("pick_id") or "")
        owner = str(slot.get("team_id") or "")
        if current_pick_id == str(target_pick_id):
            if owner == str(session.user_team_id):
                payload = get_entry_draft_payload(session)
                available = list(payload.get("available_prospects") or [])
                if not available:
                    _fail("no available prospects on user target pick")
                return execute_user_draft_pick(session, str(available[0].get("key") or ""))
            return execute_cpu_draft_pick(session)
        if owner == str(session.user_team_id):
            payload = get_entry_draft_payload(session)
            available = list(payload.get("available_prospects") or [])
            if not available:
                _fail("no available prospects on user pick while advancing to target")
            execute_user_draft_pick(session, str(available[0].get("key") or ""))
        else:
            execute_cpu_draft_pick(session)
    _fail(f"failed to reach target pick {target_pick_id} within safety loop")
    return {}


def _pick_value_map(session: Any) -> Dict[str, Dict[str, Any]]:
    assets = build_trade_assets_payload(session)
    teams = dict(assets.get("teams") or {})
    out: Dict[str, Dict[str, Any]] = {}
    for tid, block in teams.items():
        for p in list((block or {}).get("picks") or []):
            pid = str(p.get("pick_id") or "")
            if pid:
                out[pid] = dict(p)
    return out


def main() -> None:
    session = start_franchise(
        team_query="Toronto",
        head_coach_name="Audit Coach",
        coach_archetype="balanced",
        seed=77,
    )
    league = session.sim.league
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)

    ensure_draft_pick_registry(league, start_year=season_year, years_ahead=4, rounds=7)
    _ok("registry initialized")

    audit = audit_pick_registry_integrity(league, start_year=season_year, years_ahead=4, rounds=7)
    if not audit.get("ok"):
        _fail(f"registry integrity audit failed: {(audit.get('errors') or ['unknown'])[0]}")
    _ok("registry integrity audit passed")

    _assert_registry_shape(league)
    _assert_one_owner_per_unresolved_pick(league)
    _assert_expected_pick_count(league, season_year, years_ahead=4, rounds=7)

    # Trade-like ownership move for a first-round pick in the entry draft year.
    draft_year = season_year + 1
    pick = _find_tradeable_first(league, draft_year)
    pick_id = str(pick["pick_id"])
    original_owner = str(pick["current_owner_team_id"])
    new_owner = _find_new_owner(league, original_owner)
    transfer_pick(league, pick_id, new_owner)
    moved = get_pick_by_id(league, pick_id) or {}
    if str(moved.get("current_owner_team_id")) != new_owner:
        _fail(f"traded pick owner not updated for {pick_id}")
    _ok(f"traded pick moved from {original_owner} to {new_owner}")

    owner_lists = _pick_owner_lists(league)
    if pick_id in owner_lists.get(original_owner, set()):
        _fail(f"old owner still has traded pick in owned_pick_ids: {pick_id}")
    if pick_id not in owner_lists.get(new_owner, set()):
        _fail(f"new owner missing traded pick in owned_pick_ids: {pick_id}")
    _ok("owned_pick_ids updated after traded pick move")

    assets = build_trade_assets_payload(session)
    new_owner_picks = list(((assets.get("teams") or {}).get(new_owner) or {}).get("picks") or [])
    if not any(str(p.get("pick_id") or "") == pick_id for p in new_owner_picks):
        _fail(f"trade assets missing traded pick under new owner: {pick_id}")
    _ok("trade assets reflect new owner")

    order = build_full_draft_order(session)
    idx = _find_slot_index(order, pick_id)
    slot = order[idx]
    if str(slot.get("team_id") or "") != new_owner:
        _fail(f"draft order owner mismatch for {pick_id}: expected {new_owner}, got {slot.get('team_id')}")
    if str(slot.get("original_owner_team_id") or "") != str(pick.get("original_team_id") or ""):
        _fail(f"draft slot original owner mismatch for {pick_id}")
    _ok("draft order reflects registry owner and preserves original owner")

    # Execute draft pick and verify resolution.
    session.draft_combine_done = True
    initialize_entry_draft(session)
    _run_until_pick(session, pick_id)
    resolved = get_pick_by_id(league, pick_id) or {}
    if not bool(resolved.get("resolved")):
        _fail(f"pick not marked resolved after selection: {pick_id}")
    if not str(resolved.get("selected_prospect_id") or ""):
        _fail(f"selected_prospect_id missing for resolved pick: {pick_id}")
    _ok("draft pick resolved and selected_prospect_id stored")

    assets_after = build_trade_assets_payload(session)
    all_after = []
    for block in dict(assets_after.get("teams") or {}).values():
        all_after.extend(list((block or {}).get("picks") or []))
    if any(str(p.get("pick_id") or "") == pick_id for p in all_after):
        _fail(f"resolved pick still present in trade assets: {pick_id}")
    _ok("resolved pick removed from trade assets")

    try:
        transfer_pick(league, pick_id, original_owner)
        _fail(f"resolved pick transfer unexpectedly succeeded: {pick_id}")
    except ValueError:
        _ok("attempting to trade resolved pick fails")

    # Valuation sanity checks.
    value_map = _pick_value_map(session)
    rows = list((getattr(league, "draft_pick_registry", {}) or {}).values())
    firsts = [r for r in rows if isinstance(r, dict) and int(r.get("year") or 0) == season_year and int(r.get("round") or 0) == 1]
    if len(firsts) >= 2:
        scored = []
        for r in firsts:
            pid = str(r.get("pick_id") or "")
            hint = float((value_map.get(pid) or {}).get("value_hint") or 0.0)
            dbg = (value_map.get(pid) or {}).get("value_debug") or {}
            scored.append((pid, hint, float(dbg.get("projected_finish_risk", 0.0) or 0.0), dbg))
        scored.sort(key=lambda x: x[2], reverse=True)
        high = scored[0]
        low = scored[-1]
        if high[1] <= low[1]:
            _fail(
                f"suspicious valuation: high-risk first {high[0]} ({high[1]:.2f}) <= low-risk first {low[0]} ({low[1]:.2f})"
            )
        _ok("bad-team/high-risk first is valued above contender/low-risk first")

    sample_owner = str(firsts[0].get("current_owner_team_id") if firsts else session.user_team_id)
    rounds_now = []
    for rnd in (1, 2, 3, 7):
        pid = f"{season_year}-round{rnd}-{sample_owner}"
        item = value_map.get(pid)
        if item:
            rounds_now.append((rnd, float(item.get("value_hint") or 0.0)))
    if len(rounds_now) >= 3:
        rounds_now.sort(key=lambda x: x[0])
        vals = [v for _, v in rounds_now]
        if not all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            _fail(f"round value ordering suspicious: {rounds_now}")
        _ok("round hierarchy sane (1st > 2nd > 3rd > late rounds)")

    # Protection/condition sensitivity checks on synthetic variants.
    sample_first = next(
        (
            r
            for r in rows
            if isinstance(r, dict)
            and not bool(r.get("resolved"))
            and int(r.get("round") or 0) == 1
            and int(r.get("year") or 0) == season_year
        ),
        None,
    )
    if sample_first:
        tid = str(sample_first.get("current_owner_team_id") or "")
        team = session.team_by_id.get(tid)
        ctx = {
            "season_year": season_year,
            "team_by_id": dict(session.team_by_id or {}),
            "deadline_phase": 0.0,
        }
        clean = dict(sample_first)
        clean["protection"] = None
        clean["conditions"] = None
        protected = dict(clean)
        protected["protection"] = "top-10"
        conditional = dict(clean)
        conditional["conditions"] = "if lottery then slides to next year"
        v_clean = float(evaluate_pick_asset_value(clean, team, team, league, context=ctx).get("total") or 0.0)
        v_prot = float(evaluate_pick_asset_value(protected, team, team, league, context=ctx).get("total") or 0.0)
        v_cond = float(evaluate_pick_asset_value(conditional, team, team, league, context=ctx).get("total") or 0.0)
        if not (v_clean > v_prot and v_clean > v_cond):
            _fail(
                f"protection/condition discounts not applied strongly enough: clean={v_clean:.2f}, protected={v_prot:.2f}, conditional={v_cond:.2f}"
            )
        _ok("protected/conditional first valued below unprotected first")

    # Future-risk retention check for declining/rebuild windows when available.
    risky_teams = [
        t for t in list(session.team_by_id.values()) if str(getattr(t, "gm_window", getattr(t, "window", "")) or "").lower() in ("declining", "rebuild")
    ]
    if risky_teams:
        tm = risky_teams[0]
        tid = _team_id(tm)
        now_pid = f"{season_year}-round1-{tid}"
        fut_pid = f"{season_year + 3}-round1-{tid}"
        now_val = float((value_map.get(now_pid) or {}).get("value_hint") or 0.0)
        fut_val = float((value_map.get(fut_pid) or {}).get("value_hint") or 0.0)
        if fut_val <= 0.0 or now_val <= fut_val:
            _fail(
                f"future-risk sanity failed for declining/rebuild team {tid}: now={now_val:.2f}, future={fut_val:.2f}"
            )
        _ok("declining/rebuild future first keeps non-trivial risk value")

    for pid, item in value_map.items():
        dbg = item.get("value_debug") or {}
        fv = dbg.get("final_value")
        vh = item.get("value_hint")
        if fv is not None and vh is not None and abs(float(fv) - float(vh)) > 0.11:
            _fail(f"value_debug.final_value mismatch for {pid}: {fv} vs {vh}")
    _ok("value_debug.final_value matches serialized value_hint")

    print("PASS draft pick safety audit complete")


if __name__ == "__main__":
    main()
