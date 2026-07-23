"""
Franchise mode orchestration — session state, calendar, offseason, scouting, API engine.

Game rules and entities live under app.sim_engine.*; this package wires them into
interactive franchise mode (formerly backend/services).
"""

from app.sim_engine.franchise.paths import ensure_simengine_path

ensure_simengine_path()
