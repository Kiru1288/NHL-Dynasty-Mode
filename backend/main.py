from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.franchise_sim import (
    advance_franchise_day,
    apply_decision,
    build_state_payload,
    list_teams_summary,
    start_franchise,
)
from services.franchise_store import get_session, save_session

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="NHL Franchise Mode API", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FranchiseStartBody(BaseModel):
    team_query: str = Field(..., description="team id, city, or nickname")
    head_coach_name: str = Field(..., min_length=1, max_length=80)
    coach_archetype: str = Field(
        default="balanced",
        description="balanced | development | defense_first | aggressive | players_coach",
    )
    seed: Optional[int] = None


class FranchiseDecisionBody(BaseModel):
    decision_id: str
    choice_id: str


def _session_or_404(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Franchise-Session header")
    s = get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown or expired franchise session")
    return s


@app.on_event("startup")
async def _startup_banner() -> None:
    log.info(
        "NHL Franchise API v0.2.1 — interactive mode: "
        "/api/franchise/teams, /start, /state, /advance, /decision "
        "(stop any old uvicorn still serving /api/sim/run or /api/runs on this port)"
    )


@app.get("/api/franchise/teams")
def get_franchise_teams() -> dict[str, Any]:
    try:
        teams = list_teams_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"teams": teams}


@app.post("/api/franchise/start")
def post_franchise_start(body: FranchiseStartBody) -> dict[str, Any]:
    try:
        session = start_franchise(
            team_query=body.team_query,
            head_coach_name=body.head_coach_name,
            coach_archetype=body.coach_archetype,
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    save_session(session)
    return {
        "session_id": session.session_id,
        "state": build_state_payload(session),
    }


@app.get("/api/franchise/state")
def get_franchise_state(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return build_state_payload(s)


@app.post("/api/franchise/advance")
def post_franchise_advance(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        step = advance_franchise_day(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"step": step, "state": build_state_payload(s)}


@app.post("/api/franchise/decision")
def post_franchise_decision(
    body: FranchiseDecisionBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        apply_decision(s, body.decision_id, body.choice_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"state": build_state_payload(s)}


@app.get("/api/health")
def health() -> dict:
    root = Path(__file__).resolve().parent.parent / "SimEngine"
    return {
        "ok": True,
        "api_version": "0.2.1",
        "mode": "interactive_franchise",
        "franchise_endpoints": True,
        "simengine": str(root),
        "run_sim_on_disk": (root / "run_sim.py").is_file(),
    }
