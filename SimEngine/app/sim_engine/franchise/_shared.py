"""Shared imports and constants for franchise submodules."""
from __future__ import annotations

import bisect
import hashlib
import logging
import os
import random
import time
import uuid
from dataclasses import is_dataclass, replace
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.paths import ensure_simengine_path

ensure_simengine_path()
import run_sim as rs  # noqa: E402
from app.sim_engine.league.schedule_generator import (  # noqa: E402
    GameSlot,
    _safe_id_str,
    _safe_slot_team_id,
    _safe_team_id,
)

_startup_log = logging.getLogger("uvicorn.error")


