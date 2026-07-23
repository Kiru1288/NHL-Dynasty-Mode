from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Optional

from services._simengine_bootstrap import ensure_simengine_path

ensure_simengine_path()

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import services.franchise_sim as franchise_sim
from services.franchise_sim import (
    advance_franchise_bulk,
    advance_franchise_day,
    advance_season_phase,
    apply_decision,
    apply_storyline_choice,
    auto_resolve_franchise_decisions,
    continue_franchise_offseason,
    dismiss_franchise_popups,
    enter_franchise_playoffs,
    execute_franchise_draft_pick,
    generate_franchise_next_season,
    get_cached_trade_assets_payload,
    get_contract_office,
    get_franchise_chemistry_report,
    get_franchise_game_detail,
    list_teams_summary,
    snapshot_draft_rank_prev,
)
from services.trade_service import (
    build_trade_market_payload,
    execute_franchise_trade,
    evaluate_franchise_trade,
    get_franchise_trade_history,
    request_ntc_waiver,
)
from services.franchise_store import (
    active_session_count,
    api_instance_id,
    clear_all_sessions,
    get_session,
    live_code_revision,
    save_session,
    set_live_code_revision,
)
from services.franchise_scouting import (
    apply_scouting_command,
    get_scouting_assignments,
    get_scouting_prospects,
    get_scouting_state,
    get_scouting_world,
)

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
    expose_headers=["X-API-Instance-Id", "X-API-Code-Revision"],
)


@app.middleware("http")
async def _stamp_backend_identity(request, call_next):
    """Let the frontend detect backend restarts/code updates on every response."""
    response = await call_next(request)
    try:
        response.headers["X-API-Instance-Id"] = str(api_instance_id())
        fp = _api_code_fingerprint()
        response.headers["X-API-Code-Revision"] = str(fp.get("revision") or "")
    except Exception:
        pass
    return response


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
    injuries_enabled: bool = Field(
        default=True,
        description="When false, the sim will not generate new injuries for this franchise.",
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


class FranchiseNtcWaiveBody(BaseModel):
    player_id: str = Field(..., min_length=1)
    source_team_id: str = Field(..., min_length=1)
    destination_team_id: str = Field(..., min_length=1)


class FranchiseStorylineChoiceBody(BaseModel):
    storyline_id: str
    choice_id: str


class FranchiseAdvancePhaseBody(BaseModel):
    target: Optional[str] = Field(default=None, description="Optional explicit phase target")


class FranchiseOffseasonStageBody(BaseModel):
    stage: Optional[str] = Field(default=None, description="Optional offseason stage key")


def _session_or_404(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing X-Franchise-Session header")
    s = get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown or expired franchise session")
    return s


def _api_code_fingerprint() -> dict[str, Any]:
    """
    Prove which backend modules are live.

    Fingerprints mtimes across backend services + SimEngine franchise/sim surfaces so
    a browser refresh after edits cannot keep serving an obsolete in-memory save.
    Cached briefly — health/middleware call this often.
    """
    import hashlib
    import time

    cached = getattr(_api_code_fingerprint, "_cache", None)
    now = time.monotonic()
    if isinstance(cached, dict) and (now - float(cached.get("at", 0))) < 0.75:
        return cached["value"]

    sim_path = Path(franchise_sim.__file__).resolve()
    backend_root = Path(__file__).resolve().parent
    services_root = sim_path.parent
    engine_root = backend_root.parent / "SimEngine" / "app" / "sim_engine"

    watch_files: list[Path] = [
        Path(__file__).resolve(),
        sim_path,
        engine_root / "engine.py",
    ]
    watch_globs = [
        (services_root, "*.py"),
        (engine_root / "systems", "*.py"),
        (engine_root / "franchise", "*.py"),
        (engine_root / "league", "*.py"),
        (engine_root / "generation", "*.py"),
        (engine_root / "world", "*.py"),
    ]
    for folder, pattern in watch_globs:
        if folder.is_dir():
            watch_files.extend(sorted(folder.glob(pattern)))

    # De-dupe while keeping stable order.
    seen: set[str] = set()
    mtimes: list[str] = []
    for p in watch_files:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        try:
            mtimes.append(f"{p.name}:{int(p.stat().st_mtime_ns)}")
        except Exception:
            mtimes.append(f"{p.name}:0")

    revision = hashlib.sha1("|".join(mtimes).encode("utf-8", errors="ignore")).hexdigest()[:16]
    value = {
        "services_root": str(services_root),
        "franchise_sim": str(sim_path),
        "revision": revision,
        "watched_files": len(mtimes),
        "features": {
            "contract_bootstrap": hasattr(franchise_sim, "_ensure_league_roster_contracts"),
            "full_contract_office": hasattr(franchise_sim, "_build_free_agent_row"),
            "draft_board_v2": hasattr(franchise_sim, "_draft_stock_reason"),
            "lineup_persistence": hasattr(franchise_sim, "save_franchise_lines"),
            "lines_route": any(getattr(r, "path", None) == "/api/franchise/lines" for r in app.routes),
            "chemistry_profile_contract": True,
            "saved_line_deployment": True,
            "stale_save_invalidation": True,
        },
    }
    _api_code_fingerprint._cache = {"at": now, "value": value}  # type: ignore[attr-defined]
    set_live_code_revision(revision)
    return value


@app.on_event("startup")
async def _startup_banner() -> None:
    fp = _api_code_fingerprint()
    log.info(
        "NHL Franchise API v0.2.1 — interactive mode: "
        "/api/franchise/teams, /start, /state, /advance, /decision "
        "(stop any old uvicorn still serving /api/sim/run or /api/runs on this port)"
    )
    log.info("Live franchise engine: %s", fp["franchise_sim"])
    log.info("Code revision: %s (watched=%s)", fp.get("revision"), fp.get("watched_files"))
    log.info("Code features: %s", fp["features"])


@app.get("/api/franchise/teams")
def get_franchise_teams() -> dict[str, Any]:
    try:
        teams = list_teams_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"teams": teams}


@app.post("/api/franchise/reset")
def reset_franchise_sessions() -> dict[str, Any]:
    cleared = clear_all_sessions()
    return {"ok": True, "cleared_sessions": cleared, "active_sessions": active_session_count()}


@app.post("/api/franchise/start")
def post_franchise_start(body: FranchiseStartBody) -> Any:
    try:
        session = franchise_sim.start_franchise(
            team_query=body.team_query,
            head_coach_name=body.head_coach_name,
            coach_archetype=body.coach_archetype,
            seed=body.seed,
            games_per_team=body.games_per_team,
            season_start_year=body.season_start_year,
            injuries_enabled=bool(body.injuries_enabled),
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
    state = franchise_sim.build_state_payload(session, include_heavy=False)
    log.info(
        "POST /api/franchise/start ok session_id=%s schedule_slots=%s",
        session.session_id,
        len(getattr(session, "schedule", None) or []),
    )
    return {"ok": True, "session_id": session.session_id, "state": state}


@app.get("/api/franchise/state")
def get_franchise_state(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return franchise_sim.build_state_payload_safe(s, include_heavy=False)


@app.get("/api/franchise/stats-central")
def get_franchise_stats_central(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.json_safe import json_safe

    s = _session_or_404(x_franchise_session)
    return json_safe(franchise_sim.get_cached_stats_central_payload(s))


@app.get("/api/franchise/draft-class/detail")
def get_franchise_draft_class_detail(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    sim = s.sim
    return franchise_sim.get_cached_draft_class_detail_payload(s, sim)


@app.get("/api/franchise/state/heavy")
def get_franchise_state_heavy(
    include_roster_browser: bool = True,
    include_draft_class_rankings: bool = True,
    include_draft_class_hud: bool = True,
    include_nhl_calendar_full: bool = False,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    full = franchise_sim.build_state_payload_safe(s, include_heavy=True)
    out: dict[str, Any] = {"session_id": str(getattr(s, "session_id", "") or "")}
    if include_roster_browser:
        out["roster_browser"] = full.get("roster_browser") or {}
    if include_draft_class_rankings:
        out["draft_class_rankings"] = full.get("draft_class_rankings") or {}
    if include_draft_class_hud:
        out["draft_class_hud"] = full.get("draft_class_hud") or {}
    if include_nhl_calendar_full:
        out["nhl_calendar_full"] = full.get("nhl_calendar_full") or []
    return out


@app.get("/api/franchise/league-operations")
def get_franchise_league_operations(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    from services.league_operations import build_league_operations_payload

    return {"league_operations": build_league_operations_payload(s)}


@app.get("/api/franchise/contract-office")
def get_franchise_contract_office(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_contract_office(s)


@app.get("/api/franchise/free-agents/{player_id}")
def get_franchise_free_agent_detail(player_id: str, x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.contract_economy import build_free_agent_detail

    s = _session_or_404(x_franchise_session)
    return build_free_agent_detail(s, player_id)


def _contract_action_route(action: str, body: dict[str, Any], session_header: Optional[str]) -> dict[str, Any]:
    from services.contract_economy import build_contract_office, handle_contract_action

    s = _session_or_404(session_header)
    result = handle_contract_action(s, action, body or {})
    if result.get("ok"):
        save_session(s)
    office = build_contract_office(s)
    result["office"] = office
    return result


@app.post("/api/franchise/contracts/offer")
def post_contract_offer(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("offer", body, x_franchise_session)


@app.post("/api/franchise/contracts/re-sign")
def post_contract_resign(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("re-sign", body, x_franchise_session)


@app.post("/api/franchise/contracts/sign-free-agent")
def post_contract_sign_fa(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("sign-free-agent", body, x_franchise_session)


@app.post("/api/franchise/contracts/qualify-rfa")
def post_contract_qualify_rfa(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("qualify-rfa", body, x_franchise_session)


@app.post("/api/franchise/contracts/release-rights")
def post_contract_release_rights(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("release-rights", body, x_franchise_session)


@app.post("/api/franchise/contracts/buyout")
def post_contract_buyout(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("buyout", body, x_franchise_session)


@app.post("/api/franchise/contracts/waive")
def post_contract_waive(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("waive", body, x_franchise_session)


@app.post("/api/franchise/contracts/bury")
def post_contract_bury(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("bury", body, x_franchise_session)


@app.post("/api/franchise/contracts/offer-sheet")
def post_contract_offer_sheet(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("offer-sheet", body, x_franchise_session)


@app.post("/api/franchise/contracts/arbitration-file")
def post_contract_arbitration_file(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("arbitration-file", body, x_franchise_session)


@app.post("/api/franchise/contracts/arbitration-settle")
def post_contract_arbitration_settle(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("arbitration-settle", body, x_franchise_session)


@app.post("/api/franchise/contracts/sign-elc")
def post_contract_sign_elc(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("sign-elc", body, x_franchise_session)


@app.post("/api/franchise/contracts/evaluate-elc")
def post_contract_evaluate_elc(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("evaluate-elc", body, x_franchise_session)


@app.get("/api/franchise/lines")
def get_franchise_lines(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return {"ok": True, "lines": dict(getattr(s, "lines", None) or {})}


@app.post("/api/franchise/lines")
def post_franchise_lines(
    body: dict[str, Any] = Body(...),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        result = franchise_sim.save_franchise_lines(s, body or {})
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    save_session(s)
    return result


@app.get("/api/franchise/game/{game_id}")
def get_franchise_game(game_id: str, x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    box = get_franchise_game_detail(s, game_id)
    if not box:
        raise HTTPException(status_code=404, detail="Game not found or recap not available for this save.")
    return {"game": box}


@app.get("/api/franchise/chemistry")
def get_franchise_chemistry(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_franchise_chemistry_report(s)


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
    except Exception as e:
        # Surface sim crashes as JSON (not a dropped socket → Axios "Network Error").
        log.exception("POST /api/franchise/advance failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Advance failed: {e}") from e

    # Safe payload: never fail after a long sim, and skip building multi‑MB calendars.
    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
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
    try:
        save_session(s)
    except Exception:
        pass
    return {"step": step, "state": state}


@app.post("/api/franchise/playoffs/enter")
def post_franchise_enter_playoffs(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        step = enter_franchise_playoffs(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = franchise_sim.build_state_payload(s, include_heavy=False)
    save_session(s)
    return {"step": step, "state": state}


@app.post("/api/franchise/playoffs/action")
def post_franchise_playoff_action(
    body: dict[str, Any] = Body(default=None),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_playoffs import handle_playoff_action, slim_live_for_client, json_safe

    s = _session_or_404(x_franchise_session)
    b = body or {}
    action = str(b.get("action") or b.get("mode") or "").strip()
    if not action:
        raise HTTPException(status_code=400, detail="action is required")
    try:
        result = handle_playoff_action(s, action, b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playoff action failed: {e}") from e
    try:
        state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
        if isinstance(state.get("playoff_live"), dict):
            state["playoff_live"] = slim_live_for_client(state["playoff_live"])
        pp = state.get("playoff_payload")
        if isinstance(pp, dict) and isinstance(pp.get("live_state"), dict):
            pp["live_state"] = slim_live_for_client(pp["live_state"])
            # Don't re-send full series copies on every action — client uses playoff_live.
            for key in ("series", "series_list", "first_round", "first_round_matchups", "matchups"):
                pp.pop(key, None)
        # Lines may contain player objects / callables from older saves — keep IDs only.
        if isinstance(state.get("lines"), dict):
            state["lines"] = json_safe(state["lines"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State build failed after playoffs: {e}") from e
    try:
        save_session(s)
    except Exception:
        pass
    return json_safe({"ok": True, "result": result, "state": state})


@app.post("/api/franchise/season/advance-phase")
def post_franchise_advance_phase(
    body: Optional[FranchiseAdvancePhaseBody] = Body(default=None),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    b = body or FranchiseAdvancePhaseBody()
    try:
        step = advance_season_phase(s, target=b.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = franchise_sim.build_state_payload(s, include_heavy=False)
    save_session(s)
    return {"step": step, "state": state}


@app.post("/api/franchise/offseason/continue")
def post_franchise_offseason_continue(
    body: dict[str, Any] = Body(default=None),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.json_safe import json_safe

    s = _session_or_404(x_franchise_session)
    b = body or {}
    from_stage = str(b.get("from_stage") or b.get("stage") or b.get("offseason_stage") or "").strip()
    try:
        step = continue_franchise_offseason(s, from_stage=from_stage or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
    try:
        save_session(s)
    except Exception:
        pass
    # step may include awards/retirements blobs with non-JSON leftovers.
    return json_safe({"step": step, "state": state})


@app.post("/api/franchise/next-season/generate")
def post_franchise_generate_next_season(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.json_safe import json_safe

    s = _session_or_404(x_franchise_session)
    try:
        step = generate_franchise_next_season(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
    save_session(s)
    return json_safe({"step": step, "state": state})


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
    return {"state": franchise_sim.build_state_payload(s, include_heavy=False)}


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
    return {"state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/popup/dismiss")
def post_franchise_popup_dismiss(
    body: FranchisePopupDismissBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    dismiss_franchise_popups(s, list(body.ids or []))
    save_session(s)
    return {"state": franchise_sim.build_state_payload_safe(s, include_heavy=False)}


@app.post("/api/franchise/trade")
def post_franchise_trade(
    body: FranchiseTradeBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        result = execute_franchise_trade(s, assets_by_team=dict(body.assets_by_team or {}))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    franchise_sim.invalidate_session_payload_caches(s, reason="trade_exec")
    save_session(s)
    return {"state": franchise_sim.build_state_payload(s, include_heavy=False), "trade_result": result}


@app.post("/api/franchise/trade/evaluate")
def post_franchise_trade_evaluate(
    body: FranchiseTradeBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        evaluation = evaluate_franchise_trade(s, assets_by_team=dict(body.assets_by_team or {}))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"evaluation": evaluation}


@app.post("/api/franchise/trade/ntc-waive")
def post_franchise_ntc_waive(
    body: FranchiseNtcWaiveBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        decision = request_ntc_waiver(
            s,
            player_id=str(body.player_id),
            source_team_id=str(body.source_team_id),
            destination_team_id=str(body.destination_team_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"ok": True, "decision": decision}


@app.get("/api/franchise/trade/history")
def get_franchise_trade_history_route(
    team_id: Optional[str] = None,
    limit: int = 50,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_franchise_trade_history(s, team_id=team_id, limit=limit)


@app.get("/api/franchise/trade/assets")
def get_franchise_trade_assets(
    force: bool = False,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_cached_trade_assets_payload(s, force=bool(force))


class FranchiseDraftPickBody(BaseModel):
    player_id: str
    drafting_team_id: Optional[str] = None
    pick_round: int = 1
    pick_overall: int = 1
    request_id: Optional[str] = None


@app.post("/api/franchise/draft/pick")
def post_franchise_draft_pick(
    body: FranchiseDraftPickBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    team_id = str(body.drafting_team_id or s.user_team_id or "")
    try:
        result = execute_franchise_draft_pick(
            s,
            drafting_team_id=team_id,
            player_id=str(body.player_id),
            pick_round=int(body.pick_round),
            pick_overall=int(body.pick_overall),
            request_id=body.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {
        "ok": True,
        "pick_result": result.get("pick_result") or result,
        "draft": result.get("draft"),
        "state": franchise_sim.build_state_payload(s, include_heavy=False),
    }


@app.get("/api/franchise/entry-draft/state")
def get_entry_draft_state_route(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import get_entry_draft_state

    s = _session_or_404(x_franchise_session)
    return {"draft": get_entry_draft_state(s), "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/entry-draft/start")
def post_entry_draft_start(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Start live Entry Draft — source of truth is backend/services/franchise_entry_draft.py (not SimEngine franchise mirror)."""
    from services.franchise_entry_draft import initialize_entry_draft

    s = _session_or_404(x_franchise_session)
    try:
        payload = initialize_entry_draft(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {"draft": payload, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/entry-draft/cpu-pick")
def post_entry_draft_cpu_pick(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import execute_cpu_draft_pick

    s = _session_or_404(x_franchise_session)
    try:
        result = execute_cpu_draft_pick(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {
        "ok": True,
        "pick_result": result.get("pick_result"),
        "draft": result.get("draft"),
        "state": franchise_sim.build_state_payload(s, include_heavy=False),
    }


@app.post("/api/franchise/entry-draft/sim-to-user-pick")
def post_entry_draft_sim_to_user(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import sim_entry_draft_to_user_pick

    s = _session_or_404(x_franchise_session)
    try:
        result = sim_entry_draft_to_user_pick(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/entry-draft/sim-round")
def post_entry_draft_sim_round(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import sim_entry_draft_round

    s = _session_or_404(x_franchise_session)
    try:
        result = sim_entry_draft_round(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/entry-draft/complete")
def post_entry_draft_complete(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import complete_entry_draft

    s = _session_or_404(x_franchise_session)
    try:
        result = complete_entry_draft(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.get("/api/franchise/entry-draft/results")
def get_entry_draft_results(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import get_draft_recap, get_entry_draft_payload

    s = _session_or_404(x_franchise_session)
    return {
        "draft": get_entry_draft_payload(s),
        "recap": get_draft_recap(s),
        "state": franchise_sim.build_state_payload(s, include_heavy=False),
    }


@app.get("/api/franchise/draft-combine/state")
def get_draft_combine_state(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Draft Combine offseason stage — live path is backend/services/franchise_scouting.py."""
    from services.franchise_scouting import get_draft_combine_payload

    s = _session_or_404(x_franchise_session)
    return {
        "draft_combine": get_draft_combine_payload(s),
        "state": franchise_sim.build_state_payload(s, include_heavy=False),
    }


class CombineMeetingBody(BaseModel):
    prospect_id: str
    meeting_type: str = "interview"


@app.post("/api/franchise/draft-combine/meeting")
def post_draft_combine_meeting(
    body: CombineMeetingBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_scouting import apply_combine_user_meeting

    s = _session_or_404(x_franchise_session)
    result = apply_combine_user_meeting(s, body.prospect_id, body.meeting_type)
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.get("/api/franchise/trade/market")
def get_franchise_trade_market(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return build_trade_market_payload(s)


class ScoutingCommandBody(BaseModel):
    scout_id: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    prospect_id: Optional[str] = None
    player_id: Optional[str] = None
    assignment_id: Optional[str] = None
    action: Optional[str] = None
    intensity: Optional[str] = "normal"
    estimated_cost: Optional[float] = None
    country_id: Optional[str] = None
    phase: Optional[str] = None
    context: Optional[dict[str, Any]] = None


def _scouting_post(route_action: str, body: ScoutingCommandBody, x_franchise_session: Optional[str]):
    s = _session_or_404(x_franchise_session)
    payload = body.model_dump(exclude_none=True)
    result = apply_scouting_command(s, payload, route_action)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message") or "Scouting action failed")
    save_session(s)
    return result


@app.get("/api/franchise/scouting/state")
def get_franchise_scouting_state(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_scouting_state(s)


@app.get("/api/franchise/scouting/world")
def get_franchise_scouting_world(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_scouting_world(s)


@app.get("/api/franchise/scouting/prospects")
def get_franchise_scouting_prospects(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_scouting_prospects(s)


@app.get("/api/franchise/scouting/assignments")
def get_franchise_scouting_assignments(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return get_scouting_assignments(s)


@app.post("/api/franchise/scouting/assign")
def post_franchise_scouting_assign(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("assign", body, x_franchise_session)


@app.post("/api/franchise/scouting/reassign")
def post_franchise_scouting_reassign(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("reassign", body, x_franchise_session)


@app.post("/api/franchise/scouting/cancel")
def post_franchise_scouting_cancel(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("cancel", body, x_franchise_session)


@app.post("/api/franchise/scouting/interview")
def post_franchise_scouting_interview(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("interview", body, x_franchise_session)


@app.post("/api/franchise/scouting/dinner")
def post_franchise_scouting_dinner(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("dinner", body, x_franchise_session)


@app.post("/api/franchise/scouting/combine")
def post_franchise_scouting_combine(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("combine", body, x_franchise_session)


@app.post("/api/franchise/scouting/private-workout")
def post_franchise_scouting_private_workout(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("private-workout", body, x_franchise_session)


@app.post("/api/franchise/scouting/request-medical")
def post_franchise_scouting_request_medical(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("request-medical", body, x_franchise_session)


@app.post("/api/franchise/scouting/focus")
def post_franchise_scouting_focus(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("focus", body, x_franchise_session)


from services.fan_reactions_api import register_fan_reactions_routes

register_fan_reactions_routes(app, _session_or_404)

# DEV ONLY — delete dev_jump_to_draft.py and remove this block when done testing draft UI.
try:
    from dev_jump_to_draft import register_dev_routes

    register_dev_routes(app)
except ImportError:
    pass

@app.get("/api/health")
def health() -> dict:
    root = Path(__file__).resolve().parent.parent / "SimEngine"
    fp = _api_code_fingerprint()
    return {
        "ok": True,
        "api_version": "0.2.1",
        "mode": "interactive_franchise",
        "franchise_endpoints": True,
        "instance_id": api_instance_id(),
        "code_revision": fp.get("revision"),
        "live_code_revision": live_code_revision(),
        "active_sessions": active_session_count(),
        "simengine": str(root),
        "run_sim_on_disk": (root / "run_sim.py").is_file(),
        "code": fp,
    }
