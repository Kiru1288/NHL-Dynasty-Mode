"""
Free agency signing AI.

Expose:
- class SigningAI
- evaluate_free_agents(team, free_agent_pool)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.sim_engine.economy.player_value import PlayerValue, player_economy_ability_01, estimate_fa_market_aav_m
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
    """Canonical 0–1 ability for signing heuristics."""
    try:
        from app.sim_engine.entities.player import player_current_ovr_01

        return float(player_current_ovr_01(player))
    except Exception:
        pass
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            v = float(fn())
        except Exception:
            return 0.0
    else:
        v = _safe_float(getattr(player, "ovr", None), 0.0)
    if v > 1.5:
        return max(0.0, min(1.0, v / 99.0))
    return max(0.0, min(1.0, v))


def _player_pos(player: Any) -> str:
    p = getattr(player, "position", "")
    s = str(getattr(p, "value", p)).upper()
    if s in ("LW", "RW", "W"):
        return "W"
    return s


@dataclass
class SigningAI:
    """
    Deterministic signer: prioritizes needs + value + cap affordability.
    """

    max_signings_per_offseason: int = 5
    cap_buffer_m: float = 0.75

    def __post_init__(self) -> None:
        self.value_model = PlayerValue()
        self.needs_model = TeamNeeds()

    def evaluate_free_agents(
        self, team: Any, free_agent_pool: List[Any], league: Any = None,
    ) -> List[Any]:
        needs = getattr(team, "needs", None) or self.needs_model.evaluate(team)
        roster = list(getattr(team, "roster", None) or [])

        # competitive posture influences preference (GM window + legacy labels)
        arche = str(getattr(team, "status", getattr(team, "archetype", "")) or "").lower()
        gmw = str(getattr(team, "window", getattr(team, "gm_window", "")) or "").lower()
        contender = gmw == "contender" or ("contend" in arche) or ("win" in arche) or ("contender" in arche)
        rebuild = gmw == "rebuild" or ("rebuild" in arche) or ("tank" in arche)
        draft_focus = str(getattr(team, "gm_strategy", "") or "").lower() == "draft_focus"

        cap_space_m = _safe_float(getattr(team, "cap_space_m", None), _safe_float(getattr(team, "cap_space", None), 10.0))

        def need_score(p: Any) -> float:
            pos = _player_pos(p)
            # map needs buckets to positions roughly
            if pos in ("C", "W"):
                return max(needs.get("top_line_forward", 0.0), needs.get("depth_forward", 0.0))
            if pos == "D":
                return needs.get("top_4_defense", 0.0)
            if pos == "G":
                return needs.get("goalie", 0.0)
            return 0.0

        def preference(p: Any) -> Tuple[float, float, float]:
            val = self.value_model.evaluate(p, team=team)
            ability = player_economy_ability_01(p)
            age = _safe_int(getattr(p, "age", 25), 25)
            n = need_score(p)
            youth = 1.0 if age <= 23 else 0.0
            prime = 1.0 if 24 <= age <= 30 else 0.0
            # contender wants proven ability; rebuild / draft_focus wants youth/value
            if contender:
                style = ability * 0.35 + prime * 0.08
            elif rebuild or draft_focus:
                style = val * 0.28 + youth * 0.14 + (0.06 if age <= 26 else 0.0)
            else:
                style = val * 0.22 + ability * 0.12 + (0.04 if age <= 28 else 0.0)
            return (n + val + style, ability, -age if rebuild else age)

        # shortlist sorted by desirability
        pool_sorted = sorted(list(free_agent_pool), key=preference, reverse=True)
        chosen: List[Any] = []
        for p in pool_sorted:
            if len(chosen) >= self.max_signings_per_offseason:
                break
            # affordability: if player has an attached cap_hit_m/aav_m, respect it; otherwise estimate by ovr
            cap_hit_m = _safe_float(getattr(p, "cap_hit_m", None), _safe_float(getattr(p, "contract_aav_m", None), 0.0))
            if cap_hit_m <= 0.0:
                cap_hit_m = estimate_fa_market_aav_m(p, league)
            if cap_space_m - cap_hit_m < self.cap_buffer_m:
                continue
            chosen.append(p)
            cap_space_m -= cap_hit_m
        return chosen

    def run_offseason(self, league: Any) -> List[str]:
        logs: List[str] = []
        free_agents: List[Any] = list(getattr(league, "free_agents", None) or [])
        if not free_agents:
            return logs

        teams = list(getattr(league, "teams", None) or [])
        for team in teams:
            roster = list(getattr(team, "roster", None) or [])
            if len(roster) >= 23:
                continue
            picks = self.evaluate_free_agents(team, free_agents, league)
            for p in picks:
                if p not in free_agents:
                    continue
                free_agents.remove(p)
                roster.append(p)
                # attach a rough AAV for logging/cap
                aav = _safe_float(getattr(p, "cap_hit_m", None), 0.0)
                if aav <= 0.0:
                    aav = estimate_fa_market_aav_m(p, league)
                    try:
                        p.cap_hit_m = aav
                    except Exception:
                        pass
                logs.append(f"SIGNING: {getattr(team, 'name', getattr(team, 'team_id', 'Team'))} signs {getattr(p, 'name', 'UFA')} at ${aav:.1f}M AAV")
            try:
                team.roster = roster
            except Exception:
                pass

        try:
            league.free_agents = free_agents
        except Exception:
            pass
        return logs


_DEFAULT = SigningAI()


def evaluate_free_agents(team: Any, free_agent_pool: List[Any]) -> List[Any]:
    return _DEFAULT.evaluate_free_agents(team, free_agent_pool)

