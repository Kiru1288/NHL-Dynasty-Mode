"""
DEV ONLY — jump a franchise straight to Draft Lottery / Combine / Entry Draft UI.

Delete this file AND remove the register_dev_routes() import block from main.py when done testing.

Usage (API must be running — restart after first adding this file):

  # New franchise → Draft Combine screen
  curl -X POST http://127.0.0.1:8000/api/dev/start-and-jump ^
    -H "Content-Type: application/json" ^
    -d "{\"team_query\":\"Toronto\",\"stage\":\"draft_combine\"}"

  # Existing session → Entry Draft
  curl -X POST http://127.0.0.1:8000/api/dev/jump-to-draft ^
    -H "Content-Type: application/json" ^
    -H "x-franchise-session: YOUR_SESSION_ID" ^
    -d "{\"stage\":\"draft\"}"

Stages: draft_lottery | draft_combine | draft

Then in the browser console on the franchise app:
  localStorage.setItem('nhl_franchise_session_id', 'PASTE_SESSION_ID');
  location.reload();

Or use the PowerShell helper:
  .\\backend\\dev_jump_to_draft.ps1 -Team Toronto -Stage draft_combine
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

import services.franchise_sim as franchise_sim
from services.franchise_session import FranchiseSession
from services.franchise_store import get_session, save_session

DraftStage = Literal["draft_lottery", "draft_combine", "draft"]
VALID_STAGES = ("draft_lottery", "draft_combine", "draft")


class DevJumpBody(BaseModel):
    stage: DraftStage = "draft_combine"
    team_query: Optional[str] = None
    head_coach_name: str = "Dev Coach"
    coach_archetype: str = "balanced"
    seed: Optional[int] = 42


def dev_prepare_draft_ui(session: FranchiseSession, stage: str = "draft_combine") -> dict[str, Any]:
    """Force session into offseason at a draft stage with hydrated backend payloads."""
    from services.franchise_offseason import (
        _prepare_draft_payload,
        _run_draft_combine,
        _run_draft_lottery,
        build_offseason_state_extras,
        complete_playoffs,
    )
    from services.franchise_sim import build_state_payload, invalidate_session_payload_caches

    stage = str(stage or "draft_combine").lower()
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}")

    if not session.playoffs_simulated:
        complete_playoffs(session)

    session.phase = "offseason"
    session.season_phase = "offseason"
    session.regular_season_complete = True
    session.retirements_processed = True
    session.development_report_done = True
    session.retirements_payload = session.retirements_payload or {"retirements": []}
    session.development_report_payload = session.development_report_payload or {"done": True}
    session.salary_cap_payload = session.salary_cap_payload or {"new_season_cap": 92.0}
    if not session.awards_payload:
        session.awards_payload = {"awards": {}, "items": []}
        session.awards_generated = True

    session.offseason_stage = stage
    session.next_important_event = stage
    session.draft_lottery_done = False
    session.draft_lottery_payload = {}
    session.draft_combine_done = False
    session.draft_combine_payload = {}
    session.draft_completed = False
    session.draft_state = {}
    session.draft_payload = {}

    _run_draft_lottery(session)
    if stage in ("draft_combine", "draft"):
        _run_draft_combine(session)
    if stage == "draft":
        _prepare_draft_payload(session)

    invalidate_session_payload_caches(session, f"dev_jump_{stage}")
    state = build_state_payload(session)
    extras = build_offseason_state_extras(session)
    state.update(extras)

    return {
        "ok": True,
        "session_id": session.session_id,
        "stage": stage,
        "phase": session.phase,
        "offseason_stage": session.offseason_stage,
        "draft_lottery_done": session.draft_lottery_done,
        "draft_combine_done": session.draft_combine_done,
        "draft_started": bool((session.draft_state or {}).get("draft_started")),
        "user_team_id": session.user_team_id,
        "champion_id": session.champion_id,
        "state": state,
        "ui_hint": (
            f"Set localStorage nhl_franchise_session_id = {session.session_id} then reload Calendar."
        ),
    }


def register_dev_routes(app) -> None:
    """Register dev-only routes. Remove import from main.py when deleting this file."""

    @app.post("/api/dev/start-and-jump")
    def dev_start_and_jump(body: DevJumpBody) -> dict[str, Any]:
        team = str(body.team_query or "Toronto").strip()
        if not team:
            raise HTTPException(status_code=400, detail="team_query required")
        try:
            session = franchise_sim.start_franchise(
                team_query=team,
                head_coach_name=body.head_coach_name,
                coach_archetype=body.coach_archetype,
                seed=body.seed,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        save_session(session)
        try:
            return dev_prepare_draft_ui(session, body.stage)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/dev/jump-to-draft")
    def dev_jump_existing(
        body: DevJumpBody,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        sid = str(x_franchise_session or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Missing x-franchise-session header")
        session = get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found — start API may have restarted")
        try:
            result = dev_prepare_draft_ui(session, body.stage)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        save_session(session)
        return result

    @app.post("/api/dev/rebootstrap-dev-leagues")
    def dev_rebootstrap_dev_leagues(
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        from services.transcendent_tank_behavior import rebootstrap_development_leagues

        sid = str(x_franchise_session or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Missing x-franchise-session header")
        session = get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = rebootstrap_development_leagues(session)
        franchise_sim.invalidate_session_payload_caches(session, "dev_rebootstrap")
        save_session(session)
        return {
            **result,
            "session_id": session.session_id,
            "state": franchise_sim.build_state_payload(session),
        }

    @app.get("/api/dev/draft-stages")
    def dev_list_stages() -> dict[str, Any]:
        return {
            "stages": list(VALID_STAGES),
            "descriptions": {
                "draft_lottery": "Draft Lottery cinematic",
                "draft_combine": "Draft Combine (invites, testing, meetings)",
                "draft": "NHL Entry Draft floor",
            },
        }

    @app.get("/api/dev/pick-audit")
    def dev_pick_audit(
        team_id: Optional[str] = None,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        from app.sim_engine.trades.trade_pick_registry import audit_pick_registry_integrity
        from services.trade_service import build_trade_assets_payload

        sid = str(x_franchise_session or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Missing x-franchise-session header")
        session = get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        league = getattr(getattr(session, "sim", None), "league", None)
        if league is None:
            raise HTTPException(status_code=400, detail="League not initialized")

        audit = audit_pick_registry_integrity(
            league,
            start_year=int(getattr(session, "season_calendar_year", 2025) or 2025),
            years_ahead=4,
            rounds=7,
        )
        payload = build_trade_assets_payload(session)
        teams = dict(payload.get("teams") or {})
        target_team = str(team_id or session.user_team_id or "").strip()
        picks = list((teams.get(target_team) or {}).get("picks") or []) if target_team else []
        return {
            "ok": bool(audit.get("ok")),
            "audit": audit,
            "team_id": target_team,
            "owned_picks": picks,
            "count": len(picks),
        }


if __name__ == "__main__":
    import json
    import sys
    import urllib.request

    api = "http://127.0.0.1:8000"
    team = "Toronto"
    stage = "draft_combine"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--team" and i + 1 < len(args):
            team = args[i + 1]
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]

    payload = json.dumps({
        "team_query": team,
        "stage": stage,
        "head_coach_name": "Dev Coach",
        "coach_archetype": "balanced",
        "seed": 42,
    }).encode()
    req = urllib.request.Request(
        f"{api}/api/dev/start-and-jump",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Failed — is the API running? ({api})")
        print(e)
        sys.exit(1)

    print("\n=== DEV DRAFT UI READY ===")
    print(f"session_id: {data.get('session_id')}")
    print(f"stage:      {data.get('offseason_stage')}")
    print(f"team:       {data.get('user_team_id')}")
    print("\nBrowser console:")
    print(f"  localStorage.setItem('nhl_franchise_session_id', '{data.get('session_id')}');")
    print("  location.reload();")
    print()
