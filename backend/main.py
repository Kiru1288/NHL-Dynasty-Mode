from __future__ import annotations

import logging
import random
import traceback
from pathlib import Path
from typing import Any, Optional

from services._simengine_bootstrap import ensure_simengine_path

ensure_simengine_path()

from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import services.franchise_sim as franchise_sim
from services.franchise_sim import (
    advance_franchise_bulk,
    advance_franchise_day,
    advance_franchise_to_next_user_game,
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
    reopen_franchise_offseason_stage,
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

# CRA falls back to 3001/3002 when another app already owns 3000.
_LOCAL_UI_ORIGINS = [
    f"http://{host}:{port}"
    for host in ("127.0.0.1", "localhost")
    for port in (3000, 3001, 3002)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_LOCAL_UI_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-API-Instance-Id", "X-API-Code-Revision"],
)


@app.middleware("http")
async def _stamp_backend_identity(request, call_next):
    """Let the frontend detect backend restarts/code updates on every response.

    Also records per-route latency via perf_profiler (NHL_PERF=0 to disable).
    """
    import time

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        response.headers["X-API-Instance-Id"] = str(api_instance_id())
        fp = _api_code_fingerprint()
        response.headers["X-API-Code-Revision"] = str(fp.get("revision") or "")
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    except Exception:
        pass
    try:
        from services.perf_profiler import record as perf_record

        path = request.url.path
        perf_record(
            f"http.{request.method} {path}",
            elapsed_ms,
            meta={"status": getattr(response, "status_code", None)},
        )
    except Exception:
        pass
    return response


@app.get("/api/perf/snapshot")
def get_perf_snapshot(top_n: int = 40) -> dict[str, Any]:
    from services.perf_profiler import snapshot

    return snapshot(top_n=max(1, min(200, int(top_n or 40))))


@app.post("/api/perf/reset")
def post_perf_reset() -> dict[str, Any]:
    from services.perf_profiler import reset

    return reset()


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
        description="September year for NHL calendar (e.g. 2026 for 2026–27). Defaults to the current NHL season.",
    )
    injuries_enabled: bool = Field(
        default=True,
        description="When false, the sim will not generate new injuries for this franchise.",
    )
    player_universe: str = Field(
        default="generated",
        description='Player universe mode: "generated" (default) or "real_nhl".',
    )


class FranchiseAdvanceBody(BaseModel):
    mode: str = Field(
        default="day",
        description="day | days | game | games | season | next_game",
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
    # Health + every response stamp this; 5s is plenty for stale-code detection
    # without re-statting dozens of .py files on every click.
    if isinstance(cached, dict) and (now - float(cached.get("at", 0))) < 5.0:
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
            player_universe=str(getattr(body, "player_universe", None) or "generated"),
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
def get_franchise_state(
    crisis_tick: int = Query(default=0, alias="crisis_tick"),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    return franchise_sim.build_state_payload_safe(s, include_heavy=False, crisis_tick=bool(crisis_tick))


@app.get("/api/franchise/crisis")
def get_franchise_crisis(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Lightweight trade-demand timer sync — avoids full GET /state every 2s."""
    s = _session_or_404(x_franchise_session)
    from services.trade_demand_engine import build_trade_demand_crisis_payload  # noqa: WPS433

    return {
        "trade_demand_crisis": build_trade_demand_crisis_payload(s, tick_timers=True),
        "stats_revision": int(getattr(s, "_stats_revision", 0) or 0),
        "narrative_revision": int(getattr(s, "_narrative_revision", 0) or 0),
    }


@app.get("/api/franchise/narrative")
def get_franchise_narrative(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.franchise_sim import get_cached_narrative_universe_payload, _narrative_cache_revision  # noqa: WPS433

    s = _session_or_404(x_franchise_session)
    return {
        "narrative_revision": _narrative_cache_revision(s),
        "narrative_universe": get_cached_narrative_universe_payload(s),
    }


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


@app.get("/api/franchise/draft-class/prospect/{prospect_id}/profile")
def get_franchise_draft_prospect_profile(
    prospect_id: str,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    sim = s.sim
    profile = franchise_sim.get_draft_prospect_profile_payload(s, sim, prospect_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Prospect not found on the active draft board")
    return {"prospect_id": str(prospect_id), "profile": profile}


@app.get("/api/franchise/state/heavy")
def get_franchise_state_heavy(
    include_roster_browser: bool = True,
    include_draft_class_rankings: bool = True,
    include_draft_class_hud: bool = True,
    include_nhl_calendar_full: bool = False,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Build only the requested heavy domains — do not compute unused board/roster work."""
    s = _session_or_404(x_franchise_session)
    sim = s.sim
    out: dict[str, Any] = {
        "session_id": str(getattr(s, "session_id", "") or ""),
        "stats_revision": int(getattr(s, "_stats_revision", 0) or 0),
        "prospect_revision": int(getattr(s, "_prospect_revision", 0) or 0),
    }
    if include_roster_browser:
        out["roster_browser"] = franchise_sim.get_cached_roster_browser(
            s, sim, str(getattr(s, "user_team_id", "") or "")
        )
    draft_board = None
    if include_draft_class_rankings or include_draft_class_hud:
        draft_board = franchise_sim.get_cached_draft_class_rankings(s, sim)
    if include_draft_class_rankings:
        out["draft_class_rankings"] = draft_board or {}
    if include_draft_class_hud:
        user_team = getattr(s, "team_by_id", {}).get(str(getattr(s, "user_team_id", "") or ""))
        out["draft_class_hud"] = franchise_sim.get_cached_draft_class_hud(
            s,
            user_team,
            {},
            [],
            (draft_board or {}).get("entries"),
        )
    if include_nhl_calendar_full:
        out["nhl_calendar_full"] = franchise_sim._get_cached_nhl_calendar_full(s, full=True)
    return out


@app.get("/api/franchise/league-operations")
def get_franchise_league_operations(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    from services.league_operations import get_cached_league_operations_payload

    return {"league_operations": get_cached_league_operations_payload(s)}


@app.get("/api/franchise/contract-office")
def get_franchise_contract_office(x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.contract_economy import get_cached_contract_office

    s = _session_or_404(x_franchise_session)
    return get_cached_contract_office(s)


@app.get("/api/franchise/free-agents/{player_id}")
def get_franchise_free_agent_detail(player_id: str, x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.contract_economy import build_free_agent_detail

    s = _session_or_404(x_franchise_session)
    return build_free_agent_detail(s, player_id)


def _contract_action_route(action: str, body: dict[str, Any], session_header: Optional[str]) -> dict[str, Any]:
    from services.contract_economy import handle_contract_action

    payload = body or {}
    s = _session_or_404(session_header)
    result = handle_contract_action(s, action, payload)
    status = str(result.get("status") or "")
    evaluate_only = bool(payload.get("evaluate_only")) or status == "evaluated"
    read_only = evaluate_only or action in ("preview-elc-offer", "evaluate-elc")

    # Previews must not persist or rebuild negotiation/office side-effects.
    if read_only:
        from services.contract_economy import get_cached_contract_office

        result["office"] = get_cached_contract_office(s)
        return result

    save_session(s)
    try:
        from services.franchise_offseason import _prepare_resign_payload
        resign = _prepare_resign_payload(s, force=True)
        result["re_sign"] = resign.get("re_sign") or resign.get("contracts")
        result["contracts"] = result["re_sign"]
        save_session(s)
    except Exception:
        pass
    if action in ("sign-elc", "prospect-rights", "submit-elc-offer"):
        try:
            from services.franchise_offseason import _run_prospect_rights_stage

            rights = _run_prospect_rights_stage(s, force=True)
            result["prospect_rights"] = rights.get("prospect_rights")
            save_session(s)
        except Exception:
            pass
    from services.contract_economy import get_cached_contract_office

    s._cached_contract_office_payload = None
    office = get_cached_contract_office(s)
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


@app.get("/api/franchise/roster/moves")
def get_roster_moves(player_id: str, x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.roster_moves import available_roster_moves

    s = _session_or_404(x_franchise_session)
    return available_roster_moves(s, player_id)


@app.post("/api/franchise/roster/move")
def post_roster_move(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    from services.roster_moves import execute_roster_move
    from services.franchise_sim import build_state_payload

    s = _session_or_404(x_franchise_session)
    result = execute_roster_move(s, body or {})
    if result.get("ok"):
        save_session(s)
        try:
            result["state"] = build_state_payload(s)
        except Exception:
            pass
    return result


@app.post("/api/franchise/contracts/offer-sheet")
def post_contract_offer_sheet(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("offer-sheet", body, x_franchise_session)


@app.post("/api/franchise/contracts/match-offer-sheet")
def post_contract_match_offer_sheet(
    body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    return _contract_action_route("match-offer-sheet", body, x_franchise_session)


@app.post("/api/franchise/contracts/decline-offer-sheet")
def post_contract_decline_offer_sheet(
    body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    return _contract_action_route("decline-offer-sheet", body, x_franchise_session)


@app.post("/api/franchise/contracts/arbitration-file")
def post_contract_arbitration_file(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("arbitration-file", body, x_franchise_session)


@app.post("/api/franchise/contracts/arbitration-settle")
def post_contract_arbitration_settle(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("arbitration-settle", body, x_franchise_session)


@app.post("/api/franchise/contracts/sign-elc")
def post_contract_sign_elc(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("sign-elc", body, x_franchise_session)


@app.post("/api/franchise/contracts/prospect-rights")
def post_contract_prospect_rights(
    body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    return _contract_action_route("prospect-rights", body, x_franchise_session)


@app.post("/api/franchise/contracts/evaluate-elc")
def post_contract_evaluate_elc(body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)) -> dict[str, Any]:
    return _contract_action_route("evaluate-elc", body, x_franchise_session)


@app.post("/api/franchise/contracts/preview-elc-offer")
def post_contract_preview_elc_offer(
    body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    return _contract_action_route("preview-elc-offer", body, x_franchise_session)


@app.post("/api/franchise/contracts/submit-elc-offer")
def post_contract_submit_elc_offer(
    body: dict[str, Any] = Body(...), x_franchise_session: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    return _contract_action_route("submit-elc-offer", body, x_franchise_session)


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
    allowed = {"day", "days", "game", "games", "season", "next_game"}
    if mode not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode {b.mode!r}; use one of: {', '.join(sorted(allowed))}",
        )

    simple_one_day = mode == "day" and int(b.count) == 1

    try:
        if mode == "next_game":
            step = advance_franchise_to_next_user_game(
                s,
                auto_resolve_decisions=bool(b.auto_resolve),
            )
        elif simple_one_day and b.auto_resolve:
            # Calendar / quick advance: same light bulk path as multi-day sims (~seconds, not minutes).
            step = advance_franchise_bulk(
                s,
                mode="days",
                count=1,
                auto_resolve_decisions=True,
            )
        elif simple_one_day:
            step = advance_franchise_day(s)
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


@app.post("/api/franchise/offseason/reopen-stage")
def post_franchise_offseason_reopen_stage(
    body: dict[str, Any] = Body(default=None),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Step back to Free Agency (or Re-Sign) from a blocked Roster Check."""
    from services.json_safe import json_safe

    s = _session_or_404(x_franchise_session)
    b = body or {}
    stage = str(b.get("stage") or b.get("offseason_stage") or "free_agency").strip()
    try:
        step = reopen_franchise_offseason_stage(s, stage)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
    try:
        save_session(s)
    except Exception:
        pass
    return json_safe({"step": step, "state": state})


@app.get("/api/franchise/free-agency/desk")
def get_franchise_free_agency_desk(
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Free Agency Wire payload for Hub / standalone screen (any season phase)."""
    from services.json_safe import json_safe
    from services.franchise_offseason import build_free_agency_desk

    s = _session_or_404(x_franchise_session)
    try:
        desk = build_free_agency_desk(s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return json_safe(desk)


@app.post("/api/franchise/free-agency/advance-day")
def post_franchise_fa_advance_day(
    body: dict[str, Any] = Body(default_factory=dict),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.json_safe import json_safe
    from services.franchise_offseason import advance_free_agency_day

    s = _session_or_404(x_franchise_session)
    days = int((body or {}).get("days") or (body or {}).get("count") or 1)
    try:
        result = advance_free_agency_day(s, days=days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
    return json_safe({**result, "state": state})


@app.post("/api/franchise/contracts/advance-day")
def post_franchise_contracts_advance_day(
    body: dict[str, Any] = Body(default_factory=dict),
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Sim one (or more) days in the exclusive own-FA negotiating window."""
    from services.json_safe import json_safe
    from services.franchise_offseason import advance_contract_negotiation_day

    s = _session_or_404(x_franchise_session)
    days = int((body or {}).get("days") or (body or {}).get("count") or 1)
    try:
        result = advance_contract_negotiation_day(s, days=days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    state = franchise_sim.build_state_payload_safe(s, include_heavy=False)
    return json_safe({**result, "state": state})


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
    franchise_sim.invalidate_session_payload_caches(s, "storyline_choice")
    save_session(s)
    return {"state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/popup/dismiss")
def post_franchise_popup_dismiss(
    body: FranchisePopupDismissBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    dismiss_franchise_popups(s, list(body.ids or []))
    from services.franchise_sim import _bump_interaction_revision  # noqa: WPS433

    _bump_interaction_revision(s)
    save_session(s)
    return {"ok": True, "dismissed": list(body.ids or [])}


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
    return {"draft": get_entry_draft_state(s)}


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


class DraftDayTradeAcceptBody(BaseModel):
    offer: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/franchise/entry-draft/accept-trade")
def post_entry_draft_accept_trade(
    body: DraftDayTradeAcceptBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    from services.franchise_entry_draft import accept_draft_day_trade_offer

    s = _session_or_404(x_franchise_session)
    try:
        result = accept_draft_day_trade_offer(s, dict(body.offer or {}))
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
    return franchise_sim.get_cached_trade_market_payload(s)


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


@app.post("/api/franchise/scouting/film-study")
def post_franchise_scouting_film_study(
    body: ScoutingCommandBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    return _scouting_post("film-study", body, x_franchise_session)


class PlayerMeetingResolveBody(BaseModel):
    interaction_id: Optional[str] = None
    meeting_id: Optional[str] = None
    choice_id: str


class PlayerMeetingStartBody(BaseModel):
    player_id: str
    interaction_type: str


@app.post("/api/franchise/player-meetings/resolve")
def post_player_meeting_resolve(
    body: PlayerMeetingResolveBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        if body.meeting_id:
            result = franchise_sim.advance_player_meeting(s, body.meeting_id, body.choice_id)
        elif body.interaction_id:
            result = franchise_sim.resolve_player_meeting(s, body.interaction_id, body.choice_id)
        else:
            raise HTTPException(status_code=400, detail="meeting_id or interaction_id required")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.post("/api/franchise/player-meetings/start")
def post_player_meeting_start(
    body: PlayerMeetingStartBody,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    try:
        result = franchise_sim.start_player_meeting(s, body.player_id, body.interaction_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


@app.get("/api/franchise/player-meetings/player/{player_id}")
def get_player_meeting_detail(
    player_id: str,
    x_franchise_session: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    s = _session_or_404(x_franchise_session)
    detail = franchise_sim.get_player_meeting_detail_payload(s, player_id)
    # Detail-only response — frontend reads `detail` only; attaching full state
    # forced a full narrative_universe rebuild on every player click (multi-second).
    return {"detail": detail}


from services.fan_reactions_api import register_fan_reactions_routes

register_fan_reactions_routes(app, _session_or_404)


class BurnerPostBody(BaseModel):
    text: str
    market_key: Optional[str] = None


class BurnerPreviewBody(BaseModel):
    text: str
    market_key: Optional[str] = None


@app.get("/api/franchise/{session_id}/social-feed")
def get_social_feed(session_id: str) -> dict[str, Any]:
    s = _session_or_404(session_id)
    from datetime import datetime, timedelta
    from services.franchise_sim import _calendar_iso_for_day  # noqa: WPS433

    def _parse_iso(raw: str) -> datetime | None:
        text = str(raw or "")[:10]
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None

    def _is_recent(item: dict[str, Any], current_iso: str, max_days: int = 2) -> bool:
        today = _parse_iso(current_iso)
        item_day = _parse_iso(str(item.get("calendar_iso") or item.get("created_at") or item.get("date") or ""))
        if today is None or item_day is None:
            return True
        return item_day >= today - timedelta(days=max_days)

    def _is_broken_social_text(text: str) -> bool:
        raw = str(text or "").strip()
        if len(raw) < 8:
            return True
        lower = raw.lower()
        if "the player" in lower:
            return True
        if "(0 ovr)" in lower:
            return True
        if any(token in lower for token in ("0 points in 0 games", "through 0 gp", "0 starts", "0.00 ppg through 0")):
            return True
        if "{" in raw and "}" in raw:
            return True
        return False

    payload = {
        "social_posts": list(getattr(s, "social_posts", None) or []),
        "reddit_threads": list(getattr(s, "reddit_threads", None) or []),
    }
    current_iso = _calendar_iso_for_day(s, int(getattr(s, "calendar_idx", 0) or 0))
    posts = list(
        payload.get("social_posts")
        or []
    )
    posts = [p for p in posts if _is_recent(p, current_iso) and not _is_broken_social_text(str(p.get("text") or ""))]
    posts.sort(key=lambda p: str(p.get("calendar_iso") or p.get("created_at") or ""), reverse=True)
    posts = posts[:60]
    threads = list(payload.get("reddit_threads") or [])
    threads = [
        t for t in threads
        if _is_recent(t, current_iso)
        and not _is_broken_social_text(str(t.get("body") or ""))
        and not _is_broken_social_text(str(t.get("title") or ""))
    ]
    threads.sort(key=lambda t: str(t.get("created_at") or t.get("calendar_iso") or ""), reverse=True)
    threads = threads[:40]
    return {"puckr": posts, "icehole": threads}


@app.get("/api/franchise/{session_id}/burner")
def get_burner_state(session_id: str) -> dict[str, Any]:
    s = _session_or_404(session_id)
    from app.sim_engine.franchise.burner_engine import burner_state_payload  # noqa: WPS433

    return burner_state_payload(s)


@app.post("/api/franchise/{session_id}/burner/preview")
def preview_burner_post(session_id: str, body: BurnerPreviewBody) -> dict[str, Any]:
    s = _session_or_404(session_id)
    from app.sim_engine.franchise.burner_engine import preview_burner_risk  # noqa: WPS433
    from app.sim_engine.franchise.storyline_engine import _market_key_for_team  # noqa: WPS433

    mk = body.market_key or _market_key_for_team(s, str(getattr(s, "user_team_id", "") or ""))
    return preview_burner_risk(s, body.text, mk)


@app.post("/api/franchise/{session_id}/burner/post")
def post_burner(session_id: str, body: BurnerPostBody) -> dict[str, Any]:
    s = _session_or_404(session_id)
    from app.sim_engine.franchise.burner_engine import submit_burner_post  # noqa: WPS433
    from app.sim_engine.franchise.storyline_engine import _market_key_for_team  # noqa: WPS433

    mk = body.market_key or _market_key_for_team(s, str(getattr(s, "user_team_id", "") or ""))
    result = submit_burner_post(s, body.text, mk, random.Random())
    franchise_sim.invalidate_session_payload_caches(s, "burner_post")
    save_session(s)
    return {**result, "state": franchise_sim.build_state_payload(s, include_heavy=False)}


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
