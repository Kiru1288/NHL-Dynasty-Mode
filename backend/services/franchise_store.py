"""In-memory franchise sessions (MVP: single-process only)."""

from __future__ import annotations

import uuid
from typing import Dict, Optional

from services.franchise_session import FranchiseSession

# Changes every time the API process starts — lets the frontend drop stale session ids.
API_INSTANCE_ID: str = uuid.uuid4().hex

_SESSIONS: Dict[str, FranchiseSession] = {}

# Disk/code fingerprint currently served by /api/health. When it changes, old in-memory
# saves are obsolete (backend edits while uvicorn stayed up) and must not survive refresh.
_LIVE_CODE_REVISION: str = ""


def api_instance_id() -> str:
    return API_INSTANCE_ID


def live_code_revision() -> str:
    return _LIVE_CODE_REVISION


def set_live_code_revision(revision: Optional[str]) -> int:
    """
    Publish the current code fingerprint. If it changed from a previous non-empty value,
    wipe every in-memory franchise save so refresh cannot resurrect stale state.
    Returns number of sessions cleared.
    """
    global _LIVE_CODE_REVISION
    rev = str(revision or "").strip()
    if not rev:
        return 0
    if not _LIVE_CODE_REVISION:
        _LIVE_CODE_REVISION = rev
        return 0
    if rev == _LIVE_CODE_REVISION:
        return 0
    cleared = clear_all_sessions()
    _LIVE_CODE_REVISION = rev
    return cleared


def active_session_count() -> int:
    return len(_SESSIONS)


def clear_all_sessions() -> int:
    count = len(_SESSIONS)
    _SESSIONS.clear()
    return count


def get_session(session_id: str) -> FranchiseSession | None:
    session = _SESSIONS.get(session_id)
    if session is None:
        return None
    stamped = str(getattr(session, "code_revision", "") or "").strip()
    live = _LIVE_CODE_REVISION
    # Unstampped sessions (pre-fix) and mismatched stamps are treated as expired.
    if live and stamped != live:
        _SESSIONS.pop(session_id, None)
        return None
    return session


def save_session(session: FranchiseSession) -> None:
    if _LIVE_CODE_REVISION and not str(getattr(session, "code_revision", "") or "").strip():
        session.code_revision = _LIVE_CODE_REVISION
    _SESSIONS[session.session_id] = session


def delete_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
