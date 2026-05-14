from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.franchise_sim import (
    advance_franchise_bulk,
    advance_franchise_day,
    apply_decision,
    apply_storyline_choice,
    auto_resolve_franchise_decisions,
    build_state_payload,
    dismiss_franchise_popups,
    execute_trade_package,
    get_franchise_game_detail,
    list_teams_summary,
    snapshot_draft_rank_prev,
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
    games_per_team: int = Field(
        default=82,
        ge=4,
        le=82,
        description="Regular-season games per team (e.g. 82, 42, 15)",
    )
    season_start_year: Optional[int] = Field(
        default=None,
        ge=2005,
        le=2040,
        description="September year for NHL calendar (e.g. 2025 for 2025–26); default 2025",
    )


class FranchiseAdvanceBody(BaseModel):
    mode: str = Field(
        default="day",
        description="day | days | game | games | season",
    )
    count: int = Field(default=1, ge=1, le=320)
    auto_resolve: bool = Field(
        default=True,
        description="If true, pending GM prompts pick the first option during bulk sims.",
    )


class FranchiseDecisionBody(BaseModel):
    decision_id: str
    choice_id: str


class FranchisePopupDismissBody(BaseModel):
    ids: list[str] = Field(default_factory=list, description="Popup ids to remove from the pending queue")


class FranchiseTradeBody(BaseModel):
    assets_by_team: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class FranchiseStorylineChoiceBody(BaseModel):
    storyline_id: str
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
def post_franchise_start(body: FranchiseStartBody) -> Any:
    try:
        session = start_franchise(
            team_query=body.team_query,
            head_coach_name=body.head_coach_name,
            coach_archetype=body.coach_archetype,
            seed=body.seed,
            games_per_team=body.games_per_team,
            season_start_year=body.season_start_year,
        )
    except ValueError as e:
        log.warning("POST /api/franchise/start validation: %s", e)
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error_type": "franchise_start_validation",
                "message": str(e),
            },
        )
    except Exception as e:
        log.error("POST /api/franchise/start failed: %s", e)
        log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error_type": "franchise_start_failed",
                "message": str(e),
            },
        )
    try:
        save_session(session)
    except Exception as e:
        log.error("POST /api/franchise/start save_session failed: %s", e)
        log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error_type": "franchise_start_save_failed",
                "message": str(e),
            },
        )
    state = build_state_payload(session)
    log.info(
        "POST /api/franchise/start ok session_id=%s schedule_slots=%s",
        session.session_id,
        len(getattr(session, "schedule", None) or []),
    )
    return {"ok": True, "session_id": session.session_id, "state": state}


@app.get("/api/franchise/state")
def get_franchise_state(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return build_state_payload(s)


@app.get("/api/franchise/game/{game_id}")
def get_franchise_game(game_id: str, x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    box = get_franchise_game_detail(s, game_id)
    if not box:
        raise HTTPException(status_code=404, detail="Game not found or recap not available for this save.")
    return {"game": box}


@app.post("/api/franchise/advance")
def post_franchise_advance(
    body: Optional[FranchiseAdvanceBody] = Body(default=None),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    b = body or FranchiseAdvanceBody()
    mode = (b.mode or "day").strip().lower()
    allowed = {"day", "days", "game", "games", "season"}
    if mode not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode {b.mode!r}; use one of: {', '.join(sorted(allowed))}",
        )

    simple_one_day = mode == "day" and int(b.count) == 1

    try:
        if simple_one_day:
            if b.auto_resolve:
                auto_resolve_franchise_decisions(s)
            step = advance_franchise_day(s)
            # Post-day GM prompts use the same default as bulk sim: first option when auto_resolve is on.
            if b.auto_resolve:
                auto_resolve_franchise_decisions(s)
        else:
            step = advance_franchise_bulk(
                s,
                mode=mode,
                count=int(b.count),
                auto_resolve_decisions=bool(b.auto_resolve),
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = build_state_payload(s)
    if isinstance(step, dict):
        if step.get("bulk"):
            if int(step.get("steps_completed") or 0) > 0:
                try:
                    snapshot_draft_rank_prev(s, s.sim)
                except Exception:
                    pass
        elif step.get("status") == "ok":
            try:
                snapshot_draft_rank_prev(s, s.sim)
            except Exception:
                pass
    save_session(s)
    return {"step": step, "state": state}


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


@app.post("/api/franchise/storyline/choice")
def post_franchise_storyline_choice(
    body: FranchiseStorylineChoiceBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        apply_storyline_choice(s, body.storyline_id, body.choice_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"state": build_state_payload(s)}


@app.post("/api/franchise/popup/dismiss")
def post_franchise_popup_dismiss(
    body: FranchisePopupDismissBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    dismiss_franchise_popups(s, list(body.ids or []))
    save_session(s)
    return {"state": build_state_payload(s)}


@app.post("/api/franchise/trade")
def post_franchise_trade(
    body: FranchiseTradeBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        result = execute_trade_package(s, assets_by_team=dict(body.assets_by_team or {}))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"state": build_state_payload(s), "trade_result": result}


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
