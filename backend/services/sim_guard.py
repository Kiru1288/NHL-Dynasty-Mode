"""One sim at a time per franchise session.

The hub overlay, the calendar, the FA desk and a retried request could each start an
advance on the same session. Two sims mutating one session concurrently (FastAPI runs
sync routes on a threadpool) corrupted day state and could leave the second request
waiting on work the first one already consumed. The second request now gets a quick
409 and the client resyncs instead.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_GUARD = threading.Lock()
_LOCKS: Dict[str, threading.Lock] = {}


def _lock_for(session_id: str) -> threading.Lock:
    with _GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[session_id] = lock
        return lock


@contextmanager
def exclusive_sim(session_id: Optional[str]) -> Iterator[bool]:
    """Yields True when this request owns the session's sim slot, False when busy.

    The lock is always released, even if the sim raises.
    """
    key = str(session_id or "").strip() or "_default"
    lock = _lock_for(key)
    got = lock.acquire(blocking=False)
    try:
        yield got
    finally:
        if got:
            lock.release()
