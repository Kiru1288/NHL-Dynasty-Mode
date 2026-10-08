"""
LEGACY split-package facade — NOT imported by backend/main.py.

Live franchise HTTP API: backend/services/franchise_sim.py + backend/main.py.
Do not edit this package expecting in-game changes; use backend/services instead.
"""

from __future__ import annotations


from app.sim_engine.franchise.offseason import continue_offseason, generate_next_season

from app.sim_engine.franchise.api_bridge import get_contract_office


def continue_franchise_offseason(session):
    return continue_offseason(session)


def generate_franchise_next_season(session):
    return generate_next_season(session)
