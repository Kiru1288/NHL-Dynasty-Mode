"""
Franchise trade engine — asset model, valuation, rules, evaluation, execution, history.
"""

from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    RetainedSalaryRecord,
    TradePackage,
    normalize_trade_package,
)
from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
from app.sim_engine.trades.trade_executor import execute_validated_trade
from app.sim_engine.trades.trade_history import (
    append_trade_record,
    ensure_trade_history,
    get_trade_history,
    serialize_trade_record,
)
from app.sim_engine.trades.trade_pick_registry import (
    draft_year_from_context,
    ensure_draft_pick_registry,
    ensure_franchise_pick_registry,
    get_pick_by_id,
    get_team_owned_picks,
    retire_draft_year_picks,
    serialize_team_picks,
    tradeable_draft_year,
    transfer_pick,
    upcoming_draft_year,
    validate_pick_ownership,
)

__all__ = [
    "DraftPickTradeAsset",
    "PlayerTradeAsset",
    "RetainedSalaryRecord",
    "TradePackage",
    "normalize_trade_package",
    "evaluate_trade_package",
    "execute_validated_trade",
    "append_trade_record",
    "ensure_trade_history",
    "get_trade_history",
    "serialize_trade_record",
    "draft_year_from_context",
    "ensure_draft_pick_registry",
    "ensure_franchise_pick_registry",
    "get_pick_by_id",
    "get_team_owned_picks",
    "retire_draft_year_picks",
    "serialize_team_picks",
    "tradeable_draft_year",
    "transfer_pick",
    "upcoming_draft_year",
    "validate_pick_ownership",
]
