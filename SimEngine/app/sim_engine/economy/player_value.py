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
    """Canonical 0–1 ability for FA/waiver/roster economy decisions."""
    try:
        from app.sim_engine.entities.player import player_current_ovr_01

        return float(player_current_ovr_01(player))
    except Exception:
        pass
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            v = float(ovr_fn())
        except Exception:
            v = 0.0
    else:
        v = _safe_float(getattr(player, "overall", None), _safe_float(getattr(player, "ovr", None), 0.0))
    if v > 1.5:
        return _clamp(v / 99.0, 0.0, 1.0)
    return _clamp(v, 0.0, 1.0)


def _player_age(player: Any) -> int:
    age = _safe_int(getattr(player, "age", None), 0)
    if age > 0:
        return age
    ident = getattr(player, "identity", None)
    return _safe_int(getattr(ident, "age", 25), 25)


def player_ovr_display(player: Any) -> float:
    """Current ability on the 0–99 display scale."""
    return _player_ovr(player) * 99.0


def player_potential_display(player: Any, *, ovr_display: Optional[float] = None) -> float:
    """Development ceiling on the 0–99 display scale."""
    ovr_01 = (float(ovr_display) / 99.0) if ovr_display is not None else _player_ovr(player)
    return _player_potential_01(player, ovr_01) * 99.0


_PRO_ASSIGNMENT_POOLS = frozenset({"nhl", "ahl", "echl"})
_PRO_ROSTER_LOCS = frozenset({"nhl", "ahl", "echl", "minors", "assigned_ahl", "assigned_echl"})


def _has_pro_hockey_assignment(player: Any) -> bool:
    """Signed/minors assignments and pro stat lines override pipeline valuation."""
    pool = str(getattr(player, "pool_context", "") or getattr(player, "_pool_context", "") or "").lower()
    if pool in _PRO_ASSIGNMENT_POOLS:
        return True
    roster_loc = str(getattr(player, "roster_location", "") or "").lower()
    if roster_loc in _PRO_ROSTER_LOCS:
        return True
    org = str(getattr(player, "organizational_status", "") or "").lower()
    if org in ("nhl", "ahl", "echl") or org.startswith("assigned_"):
        return True
    stats = getattr(player, "season_stats", None) or {}
    if isinstance(stats, dict):
        gp = _safe_int(stats.get("gp") or stats.get("gamesPlayed"), 0)
        if gp >= 10:
            return True
    return False


def is_prospect_for_valuation(
    player: Any,
    *,
    age: Optional[int] = None,
    ovr_display: Optional[float] = None,
) -> bool:
    """
    Shared pipeline detector for trade / FA / waiver / signing systems.

    Rights-held, unsigned, junior/college assets only — never AHL/NHL assignments
    or `prospect_status` labels alone.
    """
    if _has_pro_hockey_assignment(player):
        return False

    age_i = _player_age(player) if age is None else int(age)
    ovr_d = player_ovr_display(player) if ovr_display is None else float(ovr_display)

    pool = str(getattr(player, "pool_context", "") or getattr(player, "_pool_context", "") or "").lower()
    if pool == "prospect":
        return True
    if pool.startswith("junior") or pool in ("chl", "ncaa", "ushl", "college", "development"):
        return True

    roster_loc = str(getattr(player, "roster_location", "") or "").lower()
    if roster_loc in ("prospect", "junior", "ncaa", "chl", "development"):
        return True

    signed = str(getattr(player, "signed_status", "") or "").lower()
    rights = getattr(player, "nhl_rights_team_id", None) or getattr(player, "rights_team_id", None)
    if rights and age_i <= 22 and ovr_d < 75 and signed in ("unsigned", "rights", "rights_only", ""):
        return True

    return False


def prospect_valuation_ovr(
    player: Any,
    *,
    ovr_display: Optional[float] = None,
    pot_display: Optional[float] = None,
) -> float:
    """Discounted ceiling anchor for pipeline assets (0–99, below raw POT)."""
    ovr_d = player_ovr_display(player) if ovr_display is None else float(ovr_display)
    pot_d = player_potential_display(player, ovr_display=ovr_d) if pot_display is None else float(pot_display)
    return max(ovr_d, pot_d * 0.85)


def player_economy_ability_01(player: Any) -> float:
    """Unified 0–1 ability anchor for economy AI (FA, waivers, signings)."""
    ovr_d = player_ovr_display(player)
    if is_prospect_for_valuation(player, ovr_display=ovr_d):
        val_d = prospect_valuation_ovr(player, ovr_display=ovr_d)
        return _clamp(val_d / 99.0, 0.0, 1.0)
    return _player_ovr(player)


def estimate_fa_market_aav_m(player: Any, league: Any = None) -> float:
    """Franchise contract curve when available; SimEngine fallback otherwise."""
    try:
        from services.contract_economy import compute_market_value

        return float(compute_market_value(player, league))
    except Exception:
        pass
    return round(1.0 + 9.0 * max(0.0, player_economy_ability_01(player) - 0.50), 3)


def _player_potential_01(player: Any, current_01: float) -> float:
    """Development ceiling on 0–1, preferring chapter profile potential when present."""
    try:
        from app.sim_engine.entities.chapter_attributes import get_player_chapters

        chapters = get_player_chapters(player)
        pot = chapters.get("potential")
        if pot is not None:
            return _clamp(float(pot) / 99.0, current_01, 1.0)
    except Exception:
        pass
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        for key in ("dev_potential", "potential", "pot"):
            if key in ratings:
                v = _safe_float(ratings.get(key), 0.0)
                if v > 1.5:
                    return _clamp(v / 99.0, current_01, 1.0)
                if v > 0:
                    return _clamp(v, current_01, 1.0)
    pot = _safe_float(getattr(player, "potential", 0), 0.0)
    if pot > 1.5:
        return _clamp(pot / 99.0, current_01, 1.0)
    if pot > 0:
        return _clamp(pot, current_01, 1.0)
    return current_01


def _career_phase(player: Any) -> str:
    phase = getattr(player, "career_phase", None)
    if isinstance(phase, str) and phase:
        return phase.upper()
    age = _player_age(player)
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
        ovr = _player_ovr(player)
        ovr_display = ovr * 99.0
        pot_01 = _player_potential_01(player, ovr)
        age = _player_age(player)
        phase = _career_phase(player)
        is_prospect = is_prospect_for_valuation(player, age=age, ovr_display=ovr_display)
        ability_01 = (
            prospect_valuation_ovr(player, ovr_display=ovr_display, pot_display=pot_01 * 99.0) / 99.0
            if is_prospect
            else ovr
        )
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

        # potential factor: small nudge once ceiling is already the ability anchor
        potential_factor = 0.04 if is_prospect else (0.02 if phase == "PRIME" else 0.0)

        # contract penalty: expensive contracts lower value unless elite ability
        ability_display = ability_01 * 99.0
        contract_penalty = _clamp((cap_hit_m / max(self.cap_hit_scale_m, 0.1)) * 0.18, 0.0, 0.25)
        if ability_display >= 80.0:
            contract_penalty *= 0.55
        elif ability_display >= 76.0:
            contract_penalty *= 0.75
        if is_prospect and cap_hit_m <= 0.05:
            contract_penalty *= 0.35

        team_fit = _team_fit(player, team)

        upside = max(0.0, (pot_01 - ovr) * 99.0)
        upside_factor = 0.0 if is_prospect else (_clamp(upside * 0.004, 0.0, 0.06) if age <= 25 else 0.0)

        value = (ability_01 * 0.70) + age_factor + potential_factor + team_fit - contract_penalty + upside_factor

        try:
            market_m = estimate_fa_market_aav_m(player)
            market_01 = _clamp(market_m / 12.0, 0.0, 1.0)
            value = value * 0.55 + market_01 * 0.45
        except Exception:
            pass

        # Normalize: map rough [-0.2..0.9] into [0..1]
        value = (value + 0.20) / 1.10
        return _clamp(value, 0.0, 1.0)


_DEFAULT_MODEL = PlayerValue()


def evaluate_player_value(player: Any, team: Optional[Any] = None) -> float:
    return _DEFAULT_MODEL.evaluate(player, team=team)

