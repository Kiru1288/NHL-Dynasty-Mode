"""Headless season through the trade deadline with a forced transcendent.

Writes a weekly tank log. Does not burn the one-shot request file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["NHL_FORCE_TRANSCENDENT"] = "1"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "before"
    out = ROOT / "docs" / f"tank_probe_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from services.franchise_sim import (
        _calendar_trade_deadline_index,
        advance_franchise_bulk,
        build_draft_class_rankings,
        start_franchise,
    )
    from services.transcendent_tank_behavior import flush_tank_week_log

    t0 = time.time()
    print("starting franchise", flush=True)
    session = start_franchise(
        team_query="Ottawa Senators",
        head_coach_name="General Manager",
        coach_archetype="balanced",
        seed=4242,
        games_per_team=82,
        injuries_enabled=True,
        player_universe="generated",
        warm_draft_cache=False,
    )
    print(f"started in {time.time() - t0:.0f}s", flush=True)
    board = build_draft_class_rankings(session, session.sim)
    star = None
    for row in board.get("prospects") or board.get("entries") or []:
        if row.get("is_transcendent") or row.get("cant_miss") and int(row.get("overall") or row.get("true_ovr") or 0) >= 99:
            star = row
            break
    if star is None:
        entries = board.get("rankings") or board.get("board") or []
        if isinstance(board, dict):
            for v in board.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and ("name" in v[0] or "key" in v[0]):
                    entries = v
                    break
        for row in entries:
            if isinstance(row, dict) and (row.get("is_transcendent") or int(row.get("true_ovr") or 0) >= 99):
                star = row
                break
    print("star", (star or {}).get("name"), "id", getattr(session, "transcendent_draft_prospect_id", None), flush=True)
    deadline = _calendar_trade_deadline_index(session)
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    print(f"simming to deadline idx {deadline} from {cursor}", flush=True)
    guard = 0
    last_stop = ""
    while cursor < int(deadline or cursor) and guard < 40:
        guard += 1
        step = min(14, int(deadline) - cursor)
        result = advance_franchise_bulk(
            session,
            mode="days",
            count=step,
            auto_resolve_decisions=True,
        )
        last_stop = str((result or {}).get("stopped_reason") or (result or {}).get("status") or "")
        # Day mode pauses for a waiver, a meeting, or a press hit. A headless season answers none of them.
        session.pending_ui_popups = []
        session.pending_decisions = []
        session.universe_interaction_queue = []
        new_cursor = int(getattr(session, "calendar_cursor", 0) or 0)
        print(f"  cursor {cursor}->{new_cursor} stop={last_stop}", flush=True)
        if new_cursor <= cursor:
            break
        cursor = new_cursor
    result = {"stopped_reason": last_stop, "status": "ok"}
    flush_tank_week_log(session, session.sim, week=int(getattr(session, "calendar_cursor", 0) or 0) // 7, reason="deadline")
    log = list(getattr(session, "tank_debug_log", None) or [])
    by_team: dict = {}
    for row in log:
        by_team.setdefault(row["team_id"], []).append(row)
    deadline_rows = []
    seen = set()
    for row in reversed(log):
        if row["team_id"] in seen:
            continue
        if row.get("reason") == "deadline" or True:
            seen.add(row["team_id"])
            deadline_rows.append(row)
    real = {"sell", "dump", "sign"}
    summary = {
        "label": label,
        "elapsed_s": round(time.time() - t0, 1),
        "stopped": (result or {}).get("stopped_reason") or (result or {}).get("status"),
        "cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "deadline_idx": deadline,
        "star": {
            "name": (star or {}).get("name"),
            "id": getattr(session, "transcendent_draft_prospect_id", None),
        },
        "teams": len(deadline_rows),
        "false_contender_labels": [
            r["team_id"] for r in deadline_rows
            if (not r.get("is_contender")) and "contend" in str(r.get("label_status") or "")
        ],
        "strength_contenders": [r["team_id"] for r in deadline_rows if r.get("is_contender")],
        "non_contenders": [r["team_id"] for r in deadline_rows if not r.get("is_contender")],
        "seller_or_hard": [
            r["team_id"] for r in deadline_rows
            if not r.get("is_contender") and r.get("mode") in ("seller", "hard_tank", "soft_sell")
        ],
        "with_real_move": [],
        "with_story_only": [],
        "modes": {},
    }
    for tid, weeks in by_team.items():
        acts = {a for w in weeks for a in (w.get("actions") or [])}
        last = weeks[-1]
        summary["modes"][str(last.get("mode"))] = summary["modes"].get(str(last.get("mode")), 0) + 1
        if not last.get("is_contender"):
            if acts & real:
                summary["with_real_move"].append(tid)
            elif acts:
                summary["with_story_only"].append(tid)
    nc = max(1, len(summary["non_contenders"]))
    summary["real_move_pct"] = round(100.0 * len(summary["with_real_move"]) / nc, 1)
    summary["rows"] = log
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
