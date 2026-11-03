"""Read-only overlay so an unsaved lineup can be scored without touching the session.

``preview_franchise_chemistry`` used to assign the preview sheet to
``session.lines`` and restore it afterwards. Requests run concurrently on the
FastAPI threadpool, so a lines save that landed in between was overwritten by the
restore. The overlay exposes the preview ``lines`` (and a private chemistry-report
cache) to the scorer while every other attribute reads and writes the real session.
"""

from __future__ import annotations

from typing import Any, Dict

_LOCAL = ("lines", "_cached_chemistry_report")


class LinesPreviewSession:
    __slots__ = ("_base", "lines", "_cached_chemistry_report")

    def __init__(self, base: Any, preview_lines: Dict[str, Any]) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "lines", preview_lines)
        object.__setattr__(self, "_cached_chemistry_report", None)

    def __getattr__(self, name: str) -> Any:  # only called for attributes not on the overlay
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LOCAL:
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_base"), name, value)
