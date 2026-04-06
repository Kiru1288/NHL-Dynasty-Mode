"""
Trade AI.

Expose:
- class TradeAI
- evaluate_trade_market(league)

Trades should occur 2–6 times per season (scaled by chaos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.economy.player_value import PlayerValue
from app.sim_engine.economy.team_needs import TeamNeeds


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _player_ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            return 0.0
    return _safe_float(getattr(player, "ovr", None), 0.0)


def _is_prospect(player: Any) -> bool:
    age = _safe_int(getattr(player, "age", 25), 25)
    return age <= 21


@dataclass
class TradeAI:
    """
    Lightweight, chaos-responsive trade generator.
    """

    base_trades: int = 2
    max_trades: int = 6
    fairness_tolerance: float = 0.12  # allowed value mismatch

    def __post_init__(self) -> None:
        self.value_model = PlayerValue()
        self.needs_model = TeamNeeds()

    def evaluate_trade_market(self, league: Any) -> List[Dict[str, Any]]:
        """
        Returns list of trade dicts with keys:
        - from_team_id, to_team_id
        - outgoing (list player names), incoming (list player names)
        - headline string for logging
        """
        teams: List[Any] = list(getattr(league, "teams", None) or [])
        if len(teams) < 2:
            return []

        # Determine trade volume by chaos if present
        chaos = _safe_float(getattr(getattr(league, "balance", None), "chaos_index", None), _safe_float(getattr(league, "chaos_index", None), 0.5))
        target_trades = self.base_trades + int(round(chaos * 4.0))
        target_trades = max(self.base_trades, min(self.max_trades, target_trades))

        # Precompute needs
        needs_by_id: Dict[str, Dict[str, float]] = {}
        for t in teams:
            tid = str(getattr(t, "team_id", getattr(t, "id", "")))
            needs_by_id[tid] = getattr(t, "needs", None) or self.needs_model.evaluate(t)

        # Candidate sellers: rebuild/declining; buyers: contenders/emerging bubble
        def gm_window(t: Any) -> str:
            w = str(getattr(t, "window", getattr(t, "gm_window", "")) or "").lower()
            if w in ("rebuild", "emerging", "contender", "declining"):
                return w
            st = str(getattr(t, "status", "") or "").lower()
            ar = str(getattr(t, "archetype", "") or "").lower()
            blob = st + " " + ar
            if "rebuild" in blob or "tank" in blob:
                return "rebuild"
            if "declin" in blob:
                return "declining"
            if "contend" in blob or "win" in blob:
                return "contender"
            return "emerging"

        sellers = [t for t in teams if gm_window(t) in ("rebuild", "declining") or ("rebuild" in str(getattr(t, "status", "")).lower())]
        buyers = [t for t in teams if gm_window(t) in ("contender", "emerging") or ("contend" in str(getattr(t, "status", "")).lower())]
        if not sellers:
            sellers = teams[:]
        if not buyers:
            buyers = teams[:]

        trades: List[Dict[str, Any]] = []
        used_pairs = set()

        for _ in range(target_trades * 3):  # attempt budget
            if len(trades) >= target_trades:
                break
            seller = sellers[_safe_int(_ % len(sellers), 0)]
            buyer = buyers[_safe_int((_ * 7) % len(buyers), 0)]
            if seller is buyer:
                continue
            sid = str(getattr(seller, "team_id", "S"))
            bid = str(getattr(buyer, "team_id", "B"))
            key = (sid, bid)
            if key in used_pairs:
                continue
            used_pairs.add(key)

            s_roster = list(getattr(seller, "roster", None) or [])
            b_roster = list(getattr(buyer, "roster", None) or [])
            if not s_roster or not b_roster:
                continue

            # Seller offers: best non-prospect skater or goalie if seller has surplus
            s_candidates = sorted(
                s_roster,
                key=lambda p: (self.value_model.evaluate(p, team=seller), _player_ovr(p)),
                reverse=True,
            )
            s_offer = next((p for p in s_candidates if not _is_prospect(p)), None) or s_candidates[0]

            # Buyer sends: prospect + low-value contract (value close to offer)
            b_candidates = sorted(
                b_roster,
                key=lambda p: (self.value_model.evaluate(p, team=buyer), _player_ovr(p)),
                reverse=True,
            )
            b_prospect = next((p for p in reversed(b_candidates) if _is_prospect(p)), None)
            if b_prospect is None:
                # fallback: mid roster piece
                b_prospect = b_candidates[-1]

            v_offer = self.value_model.evaluate(s_offer, team=buyer)
            v_back = self.value_model.evaluate(b_prospect, team=seller)

            # Fairness check
            if abs(v_offer - v_back) > self.fairness_tolerance:
                continue

            # Execute trade (mutate rosters)
            try:
                s_roster.remove(s_offer)
                b_roster.append(s_offer)
                b_roster.remove(b_prospect)
                s_roster.append(b_prospect)
                seller.roster = s_roster
                buyer.roster = b_roster
            except Exception:
                continue

            headline = (
                "TRADE: "
                + f"{getattr(buyer, 'name', bid)} acquire {getattr(s_offer, 'name', 'Player')}; "
                + f"{getattr(seller, 'name', sid)} receive {getattr(b_prospect, 'name', 'Asset')}"
            )
            trades.append(
                {
                    "from_team_id": sid,
                    "to_team_id": bid,
                    "outgoing": [getattr(s_offer, "name", "Player")],
                    "incoming": [getattr(b_prospect, "name", "Asset")],
                    "headline": headline,
                }
            )

        return trades


_DEFAULT = TradeAI()


def evaluate_trade_market(league: Any) -> List[Dict[str, Any]]:
    return _DEFAULT.evaluate_trade_market(league)

