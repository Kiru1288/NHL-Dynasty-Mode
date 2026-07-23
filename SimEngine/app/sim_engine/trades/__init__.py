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
    ensure_draft_pick_registry,
    get_pick_by_id,
    get_team_owned_picks,
    serialize_team_picks,
    transfer_pick,
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
    "ensure_draft_pick_registry",
    "get_pick_by_id",
    "get_team_owned_picks",
    "serialize_team_picks",
    "transfer_pick",
    "validate_pick_ownership",
]
