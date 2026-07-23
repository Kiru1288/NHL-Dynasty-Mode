"""
Trade AI — ambient CPU-CPU trades via the full trade engine.

Expose:
- class TradeAI
- evaluate_trade_market(league)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades


@dataclass
class TradeAI:
    base_trades: int = 2
    max_trades: int = 6
    fairness_tolerance: float = 7.0

    def evaluate_trade_market(
        self,
        league: Any,
        *,
        max_executions: Optional[int] = None,
        calendar_cursor: int = 0,
        regular_season_last_index: int = 192,
    ) -> List[Dict[str, Any]]:
        if len(list(getattr(league, "teams", None) or [])) < 2:
            return []

        chaos = 0.5
        try:
            chaos = float(
                getattr(getattr(league, "balance", None), "chaos_index", None)
                or getattr(league, "chaos_index", 0.5)
                or 0.5
            )
        except Exception:
            pass
        target = self.base_trades + int(round(chaos * 4.0))
        target = max(self.base_trades, min(self.max_trades, target))
        if max_executions is not None:
            target = max(0, min(int(max_executions), target))

        return propose_and_execute_cpu_trades(
            league,
            max_executions=target,
            calendar_cursor=int(calendar_cursor or getattr(league, "calendar_cursor", 0) or 0),
            regular_season_last_index=int(
                regular_season_last_index or getattr(league, "regular_season_last_index", 192) or 192
            ),
            fairness_gap_max=float(self.fairness_tolerance),
        )


_DEFAULT = TradeAI()


def evaluate_trade_market(
    league: Any,
    *,
    max_executions: Optional[int] = None,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
) -> List[Dict[str, Any]]:
    return _DEFAULT.evaluate_trade_market(
        league,
        max_executions=max_executions,
        calendar_cursor=calendar_cursor,
        regular_season_last_index=regular_season_last_index,
    )
