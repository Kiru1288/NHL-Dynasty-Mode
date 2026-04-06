"""
Player valuation model.

Output normalized to [0, 1]:
0.0 = replacement-level asset
1.0 = franchise cornerstone
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


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
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            return 0.0
    return _safe_float(getattr(player, "overall", None), _safe_float(getattr(player, "ovr", None), 0.0))


def _career_phase(player: Any) -> str:
    phase = getattr(player, "career_phase", None)
    if isinstance(phase, str) and phase:
        return phase.upper()
    age = _safe_int(getattr(player, "age", 25), 25)
    if age < 22:
        return "PROSPECT"
    if age <= 29:
        return "PRIME"
    if age <= 33:
        return "VETERAN"
    return "DECLINE"


def _contract_cap_hit_m(player: Any) -> float:
    # common shapes:
    # - player.cap_hit_m
    # - player.contract.cap_hit_m or cap_hit
    # - player.contract_aav_m
    for k in ("cap_hit_m", "contract_aav_m", "aav_m"):
        v = getattr(player, k, None)
        if v is not None:
            return _safe_float(v, 0.0)
    c = getattr(player, "contract", None)
    if c is not None:
        for k in ("cap_hit_m", "cap_hit", "aav_m", "aav"):
            v = getattr(c, k, None)
            if v is not None:
                return _safe_float(v, 0.0)
    return 0.0


def _team_fit(player: Any, team: Optional[Any]) -> float:
    """
    Team-fit is intentionally simple and robust:
    - contenders slightly prefer PRIME/VETERAN
    - rebuild slightly prefers PROSPECT/young
    - if team has an archetype string, use it as a mild nudge
    """
    if team is None:
        return 0.0
    arche = str(getattr(team, "archetype", getattr(team, "status", getattr(team, "team_status", ""))) or "").lower()
    phase = _career_phase(player)
    age = _safe_int(getattr(player, "age", 25), 25)
    if "rebuild" in arche or "tank" in arche:
        return 0.10 if (phase == "PROSPECT" or age <= 23) else (-0.05 if age >= 30 else 0.0)
    if "win" in arche or "contend" in arche or "contender" in arche:
        if phase in ("PRIME", "VETERAN"):
            return 0.08
        if phase == "DECLINE":
            return -0.06
        return 0.0
    return 0.0


@dataclass
class PlayerValue:
    """
    Value model tuned for decision-making (signings/trades/waivers), not realism.
    """

    # Typical cap hit scale for penalty (in millions)
    cap_hit_scale_m: float = 10.0

    def evaluate(self, player: Any, *, team: Optional[Any] = None) -> float:
        ovr = _clamp(_player_ovr(player), 0.0, 1.0)
        age = _safe_int(getattr(player, "age", 25), 25)
        phase = _career_phase(player)
        cap_hit_m = _contract_cap_hit_m(player)

        # age factor: peak around 26-28, declines outside window
        if age <= 18:
            age_factor = 0.10
        elif age <= 21:
            age_factor = 0.14
        elif age <= 24:
            age_factor = 0.10
        elif age <= 30:
            age_factor = 0.06
        elif age <= 34:
            age_factor = -0.02
        else:
            age_factor = -0.08

        # potential factor: proxies via phase (prospects more upside variance)
        potential_factor = 0.08 if phase == "PROSPECT" else (0.02 if phase == "PRIME" else 0.0)

        # contract penalty: expensive contracts lower value unless elite ovr
        contract_penalty = _clamp((cap_hit_m / max(self.cap_hit_scale_m, 0.1)) * 0.18, 0.0, 0.25)
        if ovr >= 0.80:
            contract_penalty *= 0.55

        team_fit = _team_fit(player, team)

        # Example formula shape requested by user
        value = (ovr * 0.70) + age_factor + potential_factor + team_fit - contract_penalty

        # Normalize: map rough [-0.2..0.9] into [0..1]
        value = (value + 0.20) / 1.10
        return _clamp(value, 0.0, 1.0)


_DEFAULT_MODEL = PlayerValue()


def evaluate_player_value(player: Any, team: Optional[Any] = None) -> float:
    return _DEFAULT_MODEL.evaluate(player, team=team)

