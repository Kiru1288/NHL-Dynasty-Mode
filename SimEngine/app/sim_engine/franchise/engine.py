"""
LEGACY split-package facade — NOT imported by backend/main.py.

Live franchise HTTP API: backend/services/franchise_sim.py + backend/main.py.
Do not edit this package expecting in-game changes; use backend/services instead.
"""

from __future__ import annotations

from app.sim_engine.franchise.engine_core import *  # noqa: F401,F403
from app.sim_engine.franchise.common import *  # noqa: F401,F403
from app.sim_engine.franchise.schedule import *  # noqa: F401,F403
from app.sim_engine.franchise.contracts import *  # noqa: F401,F403
from app.sim_engine.franchise.serialization import *  # noqa: F401,F403
from app.sim_engine.franchise.advance import *  # noqa: F401,F403
from app.sim_engine.franchise.state import *  # noqa: F401,F403
from app.sim_engine.franchise.decisions import *  # noqa: F401,F403
from app.sim_engine.franchise.events import *  # noqa: F401,F403
from app.sim_engine.franchise.progression import *  # noqa: F401,F403

from app.sim_engine.franchise.offseason import (  # noqa: F401
    advance_season_phase,
    build_offseason_state_extras,
    continue_offseason,
    generate_next_season,
    complete_playoffs,
)

from app.sim_engine.franchise.api_bridge import (  # noqa: F401
    enter_franchise_playoffs,
    execute_franchise_draft_pick,
    get_cached_trade_assets_payload,
    get_contract_office,
    get_franchise_chemistry_report,
)


def continue_franchise_offseason(session):
    return continue_offseason(session)


def generate_franchise_next_season(session):
    return generate_next_season(session)
