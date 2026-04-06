"""In-memory franchise sessions (MVP: single-process only)."""

from __future__ import annotations

from typing import Dict

from services.franchise_session import FranchiseSession

_SESSIONS: Dict[str, FranchiseSession] = {}


def get_session(session_id: str) -> FranchiseSession | None:
    return _SESSIONS.get(session_id)


def save_session(session: FranchiseSession) -> None:
    _SESSIONS[session.session_id] = session


def delete_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
