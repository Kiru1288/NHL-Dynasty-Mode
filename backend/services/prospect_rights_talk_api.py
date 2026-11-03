"""HTTP routes for rights-held prospect talks and signing interest (prospect_rights_talk.py)."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from fastapi import APIRouter, Body, Header, HTTPException
except ImportError:  # pragma: no cover - tooling without FastAPI
    APIRouter = None  # type: ignore[assignment]

from services.franchise_store import get_session, save_session

router = APIRouter() if APIRouter is not None else None


if router is not None:

    def _session(sid: Optional[str]) -> Any:
        if not sid:
            raise HTTPException(status_code=400, detail="Missing X-Franchise-Session header")
        s = get_session(sid)
        if s is None:
            raise HTTPException(status_code=404, detail="Unknown or expired franchise session")
        return s


    @router.get("/api/franchise/prospect-talk/interest")
    def prospect_signing_interest(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        from services.json_safe import json_safe
        from services.prospect_rights_talk import interest_board

        return json_safe(interest_board(_session(x_franchise_session)))


    @router.post("/api/franchise/prospect-talk")
    def prospect_talk(
        body: Dict[str, Any] = Body(default=None),
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        from services.json_safe import json_safe
        from services.prospect_rights_talk import talk

        s = _session(x_franchise_session)
        b = body or {}
        out = talk(s, str(b.get("player_id") or ""), str(b.get("topic") or ""), str(b.get("approach") or ""))
        if b.get("approach") and out.get("ok"):
            save_session(s)
        return json_safe(out)
