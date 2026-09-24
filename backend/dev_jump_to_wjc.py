"""
DEV ONLY — jump a franchise to the World Juniors desk with real draft-class prospects.

Delete this file, dev_jump_to_wjc.ps1, and remove register_dev_routes() from main.py when done.

Usage (API must be running — restart after adding this file):

  # New franchise → WJC day 1 (live popup) + write tmp snapshot
  curl -X POST http://127.0.0.1:8000/api/dev/start-and-jump ^
    -H "Content-Type: application/json" ^
    -d "{\"team_query\":\"Toronto\",\"stage\":\"wjc_live\",\"persist_snapshot\":true}"

  # Existing session
  curl -X POST http://127.0.0.1:8000/api/dev/jump-to-wjc ^
    -H "Content-Type: application/json" ^
    -H "x-franchise-session: YOUR_SESSION_ID" ^
    -d "{\"stage\":\"wjc_live\"}"

  # Reload snapshot after API restart
  curl -X POST http://127.0.0.1:8000/api/dev/load-wjc-snapshot

Stages: wjc_preview | wjc_live | wjc_mid | wjc_final

Browser:
  localStorage.setItem('nhl_franchise_session_id', 'PASTE_SESSION_ID');
  location.reload();

PowerShell:
  .\\backend\\dev_jump_to_wjc.ps1 -Team Ottawa -Stage wjc_live
  .\\backend\\dev_jump_to_wjc.ps1 -LoadSnapshot
  .\\backend\\dev_jump_to_wjc.ps1 -DeleteSnapshot
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

import services.franchise_sim as franchise_sim
from services.franchise_session import FranchiseSession
from services.franchise_store import get_session, save_session

WjcStage = Literal["wjc_preview", "wjc_live", "wjc_mid", "wjc_final"]
VALID_STAGES = ("wjc_preview", "wjc_live", "wjc_mid", "wjc_final")

TMP_DIR = Path(__file__).resolve().parent / "tmp"
SNAPSHOT_PATH = TMP_DIR / "dev_wjc_test_save.pkl"


class DevJumpBody(BaseModel):
    stage: WjcStage = "wjc_live"
    team_query: Optional[str] = None
    head_coach_name: str = "Dev Coach"
    coach_archetype: str = "balanced"
    seed: Optional[int] = 42
    persist_snapshot: bool = Field(default=True, description="Write dev_wjc_test_save.pkl after jump")
    rebootstrap_dev_leagues: bool = Field(
        default=True,
        description="Refresh CHL/NCAA/etc. pools so WJC rosters use real tracked prospects",
    )


def _wjc_real_prospect_count(rows: Any) -> int:
    n = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or "")
        if pid.startswith("wjc_npc_"):
            continue
        if bool(row.get("is_npc")):
            continue
        if str(row.get("prospect_classification") or "") == "tournament_npc":
            continue
        n += 1
    return n


def _find_calendar_index_for_iso(session: FranchiseSession, target_iso: str) -> int:
    cal = getattr(session, "nhl_calendar", None) or []
    target = str(target_iso or "")[:10]
    for i, row in enumerate(cal):
        if not isinstance(row, dict):
            continue
        iso = str(row.get("iso") or row.get("date") or row.get("calendar_iso") or "")[:10]
        if iso == target:
            return i
    raise ValueError(f"No NHL calendar row for {target_iso} (calendar len={len(cal)})")


def _wjc_day_offset_for_stage(stage: str) -> int:
    if stage == "wjc_preview":
        return -1
    if stage == "wjc_mid":
        return 4
    if stage == "wjc_final":
        return 10
    return 0


def _hydrate_real_prospect_pools(session: FranchiseSession) -> dict[str, Any]:
    from services.transcendent_tank_behavior import rebootstrap_development_leagues

    reb = rebootstrap_development_leagues(session)
    try:
        franchise_sim.snapshot_draft_rank_prev(session, session.sim, force=True)
    except Exception as exc:
        reb = {**reb, "draft_rank_snapshot_error": str(exc)}
    return reb


def dev_prepare_wjc_ui(
    session: FranchiseSession,
    stage: str = "wjc_live",
    *,
    rebootstrap: bool = True,
) -> dict[str, Any]:
    """Land on the WJC calendar window with a full national-tournament bundle (real prospects)."""
    from services.franchise_sim import (
        _calendar_iso_for_day,
        _ensure_wjc_tournament_bundle,
        _push_wjc_live_popup,
        _wjc_calendar_dates,
        _wjc_client_visible_prospects,
        _wjc_day_index_for_iso,
        _wjc_live_tournament_payload,
        build_state_payload,
        invalidate_session_payload_caches,
    )

    stage = str(stage or "wjc_live").lower()
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}")

    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    wjc_dates = _wjc_calendar_dates(sy)
    if not wjc_dates:
        raise ValueError("WJC calendar dates unavailable for season year")

    rebootstrap_meta: dict[str, Any] = {}
    if rebootstrap:
        rebootstrap_meta = _hydrate_real_prospect_pools(session)

    evaluated = getattr(session, "wjc_stock_evaluated_seasons", None)
    if not isinstance(evaluated, set):
        session.wjc_stock_evaluated_seasons = set()
        evaluated = session.wjc_stock_evaluated_seasons
    evaluated.discard(sy)
    session.wjc_tournament_bundle = None

    pending = list(getattr(session, "pending_ui_popups", None) or [])
    session.pending_ui_popups = [
        p
        for p in pending
        if not (isinstance(p, dict) and (p.get("kind") == "wjc_tournament" or p.get("wjc_live")))
    ]

    if session.phase not in ("regular", "preseason"):
        session.phase = "regular"
        session.season_phase = "regular"
    session.regular_season_complete = False

    offset = _wjc_day_offset_for_stage(stage)
    if offset < 0:
        first = wjc_dates[0]
        from datetime import timedelta

        preview = (first - timedelta(days=1)).isoformat()
        try:
            session.calendar_cursor = _find_calendar_index_for_iso(session, preview)
        except ValueError:
            session.calendar_cursor = _find_calendar_index_for_iso(session, first.isoformat())
        _ensure_wjc_tournament_bundle(session)
        iso = _calendar_iso_for_day(session, int(session.calendar_cursor))
        invalidate_session_payload_caches(session, f"dev_jump_{stage}")
        state = build_state_payload(session)
        bundle = getattr(session, "wjc_tournament_bundle", None) or {}
        visible = _wjc_client_visible_prospects(bundle.get("tournament_prospects"))
        return _wjc_result(
            session,
            stage,
            state,
            iso=iso,
            wjc_day=None,
            wjc_phase="upcoming",
            tournament_prospects=visible,
            rebootstrap_meta=rebootstrap_meta,
        )

    day_idx = min(max(0, offset), len(wjc_dates) - 1)
    target_iso = wjc_dates[day_idx].isoformat()
    session.calendar_cursor = _find_calendar_index_for_iso(session, target_iso)
    iso = _calendar_iso_for_day(session, int(session.calendar_cursor))

    _ensure_wjc_tournament_bundle(session)
    n_days = len(wjc_dates)
    d_idx = _wjc_day_index_for_iso(iso, sy)
    if d_idx is None:
        d_idx = day_idx
    payload = _wjc_live_tournament_payload(session, iso, d_idx, n_days)
    if stage in ("wjc_live", "wjc_mid"):
        _push_wjc_live_popup(session, payload)
    elif stage == "wjc_final":
        _push_wjc_live_popup(session, payload)

    invalidate_session_payload_caches(session, f"dev_jump_{stage}")
    state = build_state_payload(session)
    visible = _wjc_client_visible_prospects(payload.get("tournament_prospects"))
    return _wjc_result(
        session,
        stage,
        state,
        iso=iso,
        wjc_day=payload.get("wjc_day"),
        wjc_phase=payload.get("wjc_phase"),
        tournament_prospects=visible,
        rebootstrap_meta=rebootstrap_meta,
        live_popup=stage != "wjc_preview",
    )


def _wjc_result(
    session: FranchiseSession,
    stage: str,
    state: dict[str, Any],
    *,
    iso: str,
    wjc_day: Any,
    wjc_phase: Any,
    tournament_prospects: list,
    rebootstrap_meta: dict[str, Any],
    live_popup: bool = False,
) -> dict[str, Any]:
    real_n = _wjc_real_prospect_count(tournament_prospects)
    return {
        "ok": True,
        "session_id": session.session_id,
        "stage": stage,
        "calendar_iso": iso,
        "wjc_day": wjc_day,
        "wjc_phase": wjc_phase,
        "user_team_id": session.user_team_id,
        "real_prospect_count": real_n,
        "tournament_prospect_count": len(tournament_prospects or []),
        "rebootstrap": rebootstrap_meta,
        "live_popup_enqueued": live_popup,
        "state": state,
        "snapshot_path": str(SNAPSHOT_PATH),
        "delete_hint": f"Delete {SNAPSHOT_PATH} (or run -DeleteSnapshot) when WJC testing is done.",
        "ui_hint": (
            f"Set localStorage nhl_franchise_session_id = {session.session_id} then reload Calendar."
        ),
    }


def write_wjc_snapshot(session: FranchiseSession) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_PATH.open("wb") as fh:
        pickle.dump(session, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return SNAPSHOT_PATH


def load_wjc_snapshot() -> FranchiseSession:
    if not SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(
            f"No WJC snapshot at {SNAPSHOT_PATH}. Run start-and-jump with persist_snapshot first."
        )
    with SNAPSHOT_PATH.open("rb") as fh:
        session = pickle.load(fh)
    if not isinstance(session, FranchiseSession):
        raise TypeError("Snapshot is not a FranchiseSession")
    save_session(session)
    return session


def delete_wjc_snapshot() -> bool:
    if SNAPSHOT_PATH.is_file():
        SNAPSHOT_PATH.unlink()
        return True
    return False


def register_dev_routes(app) -> None:
    """Register dev-only WJC routes. Remove import from main.py when deleting this file."""

    @app.post("/api/dev/start-and-jump")
    @app.post("/api/dev/wjc/start-and-jump")
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
            result = dev_prepare_wjc_ui(
                session,
                body.stage,
                rebootstrap=body.rebootstrap_dev_leagues,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        save_session(session)
        if body.persist_snapshot:
            path = write_wjc_snapshot(session)
            result["snapshot_written"] = str(path)
        return result

    @app.post("/api/dev/jump-to-wjc")
    def dev_jump_existing(
        body: DevJumpBody,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        sid = str(x_franchise_session or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Missing x-franchise-session header")
        session = get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found — API may have restarted")
        try:
            result = dev_prepare_wjc_ui(
                session,
                body.stage,
                rebootstrap=body.rebootstrap_dev_leagues,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        save_session(session)
        if body.persist_snapshot:
            path = write_wjc_snapshot(session)
            result["snapshot_written"] = str(path)
        return result

    @app.post("/api/dev/load-wjc-snapshot")
    def dev_load_wjc_snapshot() -> dict[str, Any]:
        try:
            session = load_wjc_snapshot()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (TypeError, pickle.UnpicklingError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid snapshot: {e}") from e
        franchise_sim.invalidate_session_payload_caches(session, "dev_wjc_snapshot_load")
        state = franchise_sim.build_state_payload(session)
        bundle = getattr(session, "wjc_tournament_bundle", None) or {}
        visible = franchise_sim._wjc_client_visible_prospects(bundle.get("tournament_prospects"))
        return {
            "ok": True,
            "session_id": session.session_id,
            "real_prospect_count": _wjc_real_prospect_count(visible),
            "state": state,
            "ui_hint": (
                f"Set localStorage nhl_franchise_session_id = {session.session_id} then reload."
            ),
        }

    @app.delete("/api/dev/wjc-snapshot")
    def dev_delete_wjc_snapshot() -> dict[str, Any]:
        deleted = delete_wjc_snapshot()
        return {"ok": True, "deleted": deleted, "path": str(SNAPSHOT_PATH)}

    @app.get("/api/dev/wjc-snapshot")
    def dev_wjc_snapshot_info() -> dict[str, Any]:
        exists = SNAPSHOT_PATH.is_file()
        size = SNAPSHOT_PATH.stat().st_size if exists else 0
        return {
            "path": str(SNAPSHOT_PATH),
            "exists": exists,
            "bytes": size,
            "delete_hint": f"DELETE {SNAPSHOT_PATH} or POST load after API restart",
        }

    @app.get("/api/dev/wjc-stages")
    def dev_list_wjc_stages() -> dict[str, Any]:
        return {
            "stages": list(VALID_STAGES),
            "descriptions": {
                "wjc_preview": "Day before tournament — nations + real prospect pool, desk upcoming",
                "wjc_live": "Dec 26 (day 1) — live WJC popup on calendar",
                "wjc_mid": "Mid tournament (day 5) — live scores and stock movement",
                "wjc_final": "Final day — medals and complete phase",
            },
        }

    @app.post("/api/dev/rebootstrap-dev-leagues")
    def dev_rebootstrap_dev_leagues(
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        sid = str(x_franchise_session or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Missing x-franchise-session header")
        session = get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        meta = _hydrate_real_prospect_pools(session)
        franchise_sim.invalidate_session_payload_caches(session, "dev_rebootstrap")
        save_session(session)
        return {
            **meta,
            "session_id": session.session_id,
            "state": franchise_sim.build_state_payload(session),
        }


if __name__ == "__main__":
    import json
    import sys
    import urllib.request

    api = "http://127.0.0.1:8000"
    team = "Toronto"
    stage = "wjc_live"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--team" and i + 1 < len(args):
            team = args[i + 1]
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]

    payload = json.dumps(
        {
            "team_query": team,
            "stage": stage,
            "head_coach_name": "Dev Coach",
            "coach_archetype": "balanced",
            "seed": 42,
            "persist_snapshot": True,
            "rebootstrap_dev_leagues": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{api}/api/dev/start-and-jump",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Failed — is the API running? ({api})")
        print(e)
        sys.exit(1)

    print("\n=== DEV WJC UI READY ===")
    print(f"session_id: {data.get('session_id')}")
    print(f"stage:      {data.get('stage')}")
    print(f"team:       {data.get('user_team_id')}")
    print(f"wjc_day:    {data.get('wjc_day')}")
    print(f"real prospects in bundle: {data.get('real_prospect_count')}")
    if data.get("snapshot_written"):
        print(f"snapshot:   {data.get('snapshot_written')}")
    print("\nBrowser console:")
    print(f"  localStorage.setItem('nhl_franchise_session_id', '{data.get('session_id')}');")
    print("  location.reload();")
    print()
