"""
Economy system package for NHL Franchise Mode sim.

This package provides modular, integration-friendly AI systems for:
- player valuation
- team needs evaluation
- roster management + lineup optimization
- free agency signings
- trade market generation
- waiver claims

Do not add I/O here; the runner/engine owns logging.
"""

from __future__ import annotations

from app.sim_engine.economy.player_value import PlayerValue, evaluate_player_value
from app.sim_engine.economy.team_needs import TeamNeeds, evaluate_team_needs
from app.sim_engine.economy.roster_manager import RosterManager, manage_roster
from app.sim_engine.economy.lineup_ai import LineupAI, generate_lineup
from app.sim_engine.economy.signing_ai import SigningAI, evaluate_free_agents
from app.sim_engine.economy.trade_ai import TradeAI, evaluate_trade_market
from app.sim_engine.economy.waiver_ai import WaiverAI, process_waivers

__all__ = [
    "PlayerValue",
    "evaluate_player_value",
    "TeamNeeds",
    "evaluate_team_needs",
    "RosterManager",
    "manage_roster",
    "LineupAI",
    "generate_lineup",
    "SigningAI",
    "evaluate_free_agents",
    "TradeAI",
    "evaluate_trade_market",
    "WaiverAI",
    "process_waivers",
]

