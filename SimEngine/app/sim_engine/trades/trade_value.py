"""
Primary NHL player-and-pick package valuation path (uncapped relative scale).

Separate stacks still exist for draft-day trades, cap-casualty partner scoring,
and some ambient heuristics — do not treat this module as the only valuation system.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from app.sim_engine.economy.team_needs import TeamNeeds, is_player_injured
from app.sim_engine.economy.cap_engine import player_cap_hit_millions
from app.sim_engine.economy.player_value import (
    is_prospect_for_valuation,
    prospect_valuation_ovr,
)
from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    TradePackage,
    find_player_on_team_roster,
    player_display_name,
)
from app.sim_engine.trades.trade_pick_registry import get_pick_by_id
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

logger = logging.getLogger(__name__)

# Bump when the talent curve / depth-star spread changes so Trade Hub caches rebuild
# without requiring a new franchise save.
TRADE_VALUE_FORMULA_VERSION = 16  # v15: exponential star curve, age/goalie/position, term surplus, consolidation
# Soft ceiling used only for UI-relative clamps / legacy helpers — player totals are uncapped.
TRADE_VALUE_SOFT_CEIL = 450.0
LEAGUE_MINIMUM_AAV_M = 0.775
# Contract burden: value lost per $1M of overpay per remaining season.
CONTRACT_BURDEN_PER_M_YEAR = 2.0
CONTRACT_BURDEN_MAX = 90.0
# Hard floor for a player's final trade value (albatross contracts).
PLAYER_VALUE_FLOOR = -60.0
# Reduced fallback when enrichment fails — never return raw OVR (82 ≈ elite TV).
TRADE_VALUE_FALLBACK_SCALE = 0.55
TRADE_VALUE_FALLBACK_FLOOR = 3.0
TRADE_VALUE_FALLBACK_CEIL = 45.0

# --- v14 rebalance knobs ---------------------------------------------------------
# Future picks: modest per-year discount, never crushing.
FUTURE_PICK_DISCOUNT_PER_YEAR = 0.07
FUTURE_PICK_DISCOUNT_CAP = 0.18
# Prospect premium (age <= 22, potential >= 80): value floor by ceiling, strongly
# convex so a 90-potential teenager is a premium asset even at ~70 OVR.
PROSPECT_PREMIUM_MAX_AGE = 23
PROSPECT_PREMIUM_MIN_POT = 74.0
# v16: prospects are the league's scarcest currency — floors roughly 1.5x v15 and
# extended down to mid-ceiling (74+) kids so a pipeline is never "worthless".
_PROSPECT_FLOOR_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (74.0, 8.0),
    (77.0, 14.0),
    (80.0, 26.0),
    (82.0, 38.0),
    (84.0, 55.0),
    (86.0, 82.0),
    (88.0, 118.0),
    (90.0, 165.0),
    (92.0, 215.0),
    (94.0, 265.0),
    (97.0, 330.0),
)
# AHL-only veterans (age 23+, low ceiling): near-zero filler.
AHL_FILLER_BASE = 1.0
AHL_FILLER_PER_OVR = 0.6  # per OVR point above 70
AHL_HIGH_CEILING_MULT = 0.75  # 23+ AHL players who still project (POT >= 80)
# Depth roles (rank on their own NHL roster): value multipliers + role-fair AAV.
DEPTH_ROLE_MULT = {
    "third_line_f": 0.72,
    "fourth_line_f": 0.50,
    "third_pair_d": 0.62,
    "extra_d": 0.45,
}
# Role-fair AAV from the real market (v15.1): 3rd-line forwards sign for ~$3-4M,
# 4th-liners ~$1.5M, third-pair D ~$3M. The old $2.2M / $1.2M bar turned ordinary
# depth deals into negative-value contracts.
DEPTH_ROLE_AAV_M = {
    "third_line_f": 3.4,
    "fourth_line_f": 1.6,
    "third_pair_d": 3.0,
    "extra_d": 1.4,
}
DEPTH_ROLE_LABEL = {
    "third_line_f": "Third-line role",
    "fourth_line_f": "Fourth-line / extra forward",
    "third_pair_d": "Third-pair defenceman",
    "extra_d": "Depth / extra defenceman",
}
DEPTH_ROLE_MAX_VAL_OVR = 81.0  # legit top-6 talent buried on a deep club is not "depth"
ROLE_BURDEN_PER_M_YEAR = 2.0
ROLE_BURDEN_TOLERANCE_M = 0.25
ROLE_BURDEN_MAX = 28.0


def _clamp(x: float, lo: float = 0.0, hi: float = TRADE_VALUE_SOFT_CEIL) -> float:
    return lo if x < lo else hi if x > hi else x


def player_value_tier(total: float) -> str:
    """Tier bands on the uncapped scale (stars can exceed 100)."""
    v = float(total)
    if v >= 120:
        return "Franchise"
    if v >= 90:
        return "Elite"
    if v >= 60:
        return "Top Asset"
    if v >= 38:
        return "Useful"
    if v >= 18:
        return "Depth"
    if v >= 0:
        return "Replacement"
    return "Negative Value"


def pick_value_tier(total: float) -> str:
    """Pick tiers on the shared uncapped asset scale."""
    v = float(total)
    if v >= 110:
        return "FRANCHISE"
    if v >= 85:
        return "ELITE"
    if v >= 60:
        return "TOP ASSET"
    if v >= 40:
        return "USEFUL"
    if v >= 20:
        return "DEPTH"
    return "LOW"


def _protection_discount(protection: Any, rnd: int) -> float:
    if not protection:
        return 0.0
    prot = str(protection).lower().replace("_", "-")
    if rnd != 1:
        return 3.0
    if "lottery" in prot:
        return 12.0
    if "top" in prot:
        return 8.0
    return 8.0


def _pick_projected_range(proj: Dict[str, Any], rnd: int) -> str:
    league_rank = proj.get("league_rank")
    points_pct = proj.get("points_pct")
    window = str(proj.get("window") or "").lower()
    n_teams = 32

    if rnd > 1:
        if league_rank is None and points_pct is None and not window:
            return "UNKNOWN"
        if window == "contender" or (league_rank is not None and league_rank <= 10):
            return "LATE"
        if window == "rebuild" or (league_rank is not None and league_rank >= 24):
            return "MID"
        return "MID"

    if league_rank is not None:
        if league_rank >= max(1, n_teams - 6):
            return "LOTTERY"
        if league_rank >= max(1, n_teams - 12):
            return "TOP 10"
        if league_rank <= 6:
            return "CONTENDER"
        if league_rank <= 14:
            return "LATE"
        return "MID"

    if points_pct is not None:
        if points_pct < 0.42:
            return "LOTTERY"
        if points_pct < 0.48:
            return "TOP 10"
        if points_pct > 0.58:
            return "CONTENDER"
        if points_pct > 0.53:
            return "LATE"
        return "MID"

    if window == "rebuild":
        return "LOTTERY"
    if window == "contender":
        return "CONTENDER"
    if window in ("declining", "emerging"):
        return "TOP 10"
    return "UNKNOWN"


def slot_curve_value(overall_slot: int) -> float:
    """Value of a known draft slot on the shared uncapped asset scale.

    Two-term decay: a steep lottery term plus a long tail so mid-round picks keep
    some currency. Proven superstars still sit above even the first overall pick.
    """
    slot = max(1, int(overall_slot))
    k = float(slot - 1)
    # v15: lottery picks are priced like the stars they usually become —
    # #1 ~160 · #3 ~135 · #5 ~115 · #10 ~79 · #16 ~53 · #24 ~35 · #32 ~26 · #48 ~18 · #80 ~12.
    # Lottery and late firsts carry more weight than a replaceable everyday NHLer.
    return _clamp(3.0 + 168.0 * math.exp(-0.088 * k) + 42.0 * math.exp(-0.010 * k), 3.0, 220.0)


def _known_pick_slot(pick_row: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[int]:
    """Overall slot when the draft board has already resolved this pick."""
    for key in ("overall_pick", "overall", "known_slot"):
        val = pick_row.get(key)
        if val not in (None, ""):
            try:
                slot = int(val)
            except (TypeError, ValueError):
                continue
            if slot >= 1:
                return slot
    slots = ctx.get("known_pick_slots")
    if isinstance(slots, dict):
        pid = str(pick_row.get("pick_id") or "")
        val = slots.get(pid)
        if val not in (None, ""):
            try:
                slot = int(val)
            except (TypeError, ValueError):
                return None
            if slot >= 1:
                return slot
    return None


def _pick_projected_slot(proj: Dict[str, Any], rnd: int) -> Optional[int]:
    league_rank = proj.get("league_rank")
    if league_rank is not None and rnd == 1:
        # Rank 1 is the best club, so their first-round pick sits at the back of the round.
        return max(1, min(32, 33 - int(league_rank)))
    points_pct = proj.get("points_pct")
    if points_pct is not None and rnd == 1:
        return int(_clamp(32 - (points_pct - 0.35) * 48.0, 1, 32))
    window = str(proj.get("window") or "").lower()
    if rnd != 1:
        return None
    # No usable record (preseason / early October): project from direction. A rebuilding
    # club picks near the top, a contender near the back (this mapping used to be inverted).
    if window == "rebuild":
        return 5
    if window == "contender":
        return 26
    if window == "declining":
        return 12
    return None


def _pick_value_context(
    proj: Dict[str, Any],
    *,
    years_out: int,
    protection: Any,
    original_team: Any,
) -> str:
    reasons: List[str] = []
    window = str(proj.get("window") or "").lower()
    if window == "rebuild":
        reasons.append("Bad team")
    elif window == "contender":
        reasons.append("Contender")
    elif window in ("declining", "emerging"):
        reasons.append("Bubble team")

    if years_out >= 2:
        reasons.append("Future uncertainty")
    elif years_out == 1:
        reasons.append("Next-year outlook")

    if protection:
        reasons.append("Protected pick")

    cap_pressure = _team_cap_pressure(original_team) if original_team is not None else ""
    if cap_pressure in ("high", "critical", "trapped"):
        reasons.append("Cap trouble")

    core = float(proj.get("core_strength") or 0.0)
    if core > 0 and core < 72:
        reasons.append("Weak roster")
    elif core >= 83:
        reasons.append("Strong roster")

    if not reasons:
        return "League-average projection"
    return " · ".join(reasons[:3])


def _prospect_draft_tier(player: Any) -> float:
    for key in ("draft_tier", "prospect_tier", "prospect_grade"):
        raw = getattr(player, key, None)
        if raw is None:
            continue
        s = str(raw).upper()
        if s in ("A+", "FRANCHISE"):
            return 1.0
        if s in ("A", "ELITE"):
            return 0.85
        if s in ("B+", "TOP"):
            return 0.65
        if s in ("B",):
            return 0.45
        if s in ("C+", "C"):
            return 0.25
    return 0.0


def _scouting_confidence(player: Any) -> float:
    for key in ("scouting_confidence", "scout_confidence", "scouted_pct"):
        try:
            v = float(getattr(player, key, 0) or 0)
            if v > 0:
                return v / 100.0 if v > 1.5 else v
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return 0.5


def _prospect_upside_score(player: Any, ovr: float, age: int, pot: float) -> float:
    """Youth upside for trade value — bounded so age alone cannot mint stars.

    Age bonuses require real upside (potential above current OVR) or an already
    established NHL floor. Draft tier and high-pot/low-ovr premiums still apply,
    but the total cannot overwhelm intrinsic talent.
    """
    if age >= 23:
        return 0.0
    upside = max(0.0, pot - ovr)
    tier = _prospect_draft_tier(player)
    confidence = _scouting_confidence(player)
    base = upside * 0.18
    # Age bump only when there is projection or proven NHL ability.
    if upside >= 4.0 or ovr >= 75:
        if age <= 20:
            base += 1.6
        elif age <= 21:
            base += 1.0
        elif age <= 22:
            base += 0.5
    elif age <= 21 and upside >= 2.5:
        base += 0.6
    if tier >= 0.85:
        base += 4.0
    elif tier >= 0.65:
        base += 2.5
    elif tier >= 0.45:
        base += 1.2
    if pot >= 88 and ovr < 74:
        base += 2.5
    elif pot >= 84 and ovr < 70:
        base += 1.5
    # Established NHL regulars should not get prospect-style upside bags.
    if ovr >= 78 and upside < 3.0:
        base *= 0.55
    elif ovr >= 76 and upside < 2.0:
        base *= 0.70
    # Low-overall depth prospects stay cheap even when young.
    if ovr < 70:
        base *= 0.55
    elif ovr < 74:
        base *= 0.75
    base *= 0.75 + 0.5 * confidence
    return _clamp(base, 0.0, 8.0)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def sync_trade_standings(league: Any, standings: Any = None) -> int:
    """Stamp each club with its live record (gp, pts, pts_pct, rank) for pick/finish projections.

    Teams carry no gp/w/l attributes, so pick values used to fall back to the club's window
    label for every projection. Reads a StandingsTable when given, else the engine's
    per-tick snapshot on the league. Returns the number of clubs stamped.
    """
    rows: Dict[str, Dict[str, float]] = {}
    recs = getattr(standings, "records", None) if standings is not None else None
    if isinstance(recs, dict) and recs:
        for tid, rec in recs.items():
            w = int(getattr(rec, "wins", 0) or 0)
            l_ = int(getattr(rec, "losses", 0) or 0)
            otl = int(getattr(rec, "otl", 0) or getattr(rec, "ot_losses", 0) or 0)
            gp = w + l_ + otl
            pts = 2 * w + otl
            rows[str(tid)] = {"gp": gp, "pts": pts, "pts_pct": pts / max(1, 2 * gp)}
    else:
        snap = getattr(league, "_cpu_standings_snapshot", None) or {}
        for tid, row in snap.items():
            if isinstance(row, dict):
                rows[str(tid)] = {"gp": int(row.get("gp") or 0), "pts": int(row.get("pts") or 0),
                                  "pts_pct": float(row.get("pts_pct") or 0.0)}
    if not rows:
        return 0
    ranked = sorted(rows.items(), key=lambda kv: (-kv[1]["pts_pct"] if kv[1]["gp"] else 0.0, -kv[1]["pts"]))
    n = 0
    teams = list(getattr(league, "teams", None) or [])
    by_id = {str(getattr(t, "team_id", getattr(t, "id", ""))): t for t in teams}
    for rank, (tid, row) in enumerate(ranked, start=1):
        t = by_id.get(tid)
        if t is None:
            continue
        try:
            setattr(t, "_trade_standings", {**row, "rank": rank, "n_teams": len(ranked)})
            n += 1
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return n


def _team_points_pct(team: Any) -> Optional[float]:
    if team is None:
        return None
    live = getattr(team, "_trade_standings", None)
    if isinstance(live, dict) and int(live.get("gp") or 0) >= 5:
        return float(live.get("pts_pct") or 0.0)
    gp = _safe_float(getattr(team, "gp", getattr(team, "games_played", 0)), 0.0)
    pts = _safe_float(getattr(team, "pts", getattr(team, "points", 0)), 0.0)
    if gp > 0 and pts >= 0:
        return pts / max(1.0, gp * 2.0)
    w = _safe_float(getattr(team, "w", getattr(team, "wins", 0)), 0.0)
    l = _safe_float(getattr(team, "l", getattr(team, "losses", 0)), 0.0)
    otl = _safe_float(getattr(team, "otl", getattr(team, "ot_losses", 0)), 0.0)
    gp2 = w + l + otl
    if gp2 <= 0:
        return None
    pts2 = w * 2.0 + otl
    return pts2 / max(1.0, gp2 * 2.0)


def _team_core_strength(team: Any) -> float:
    roster = list(getattr(team, "roster", None) or [])
    if not roster:
        return 0.0
    vals: List[float] = []
    for p in roster:
        try:
            vals.append(_player_ovr(p))
        except Exception:
            continue
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    top = vals[:10]
    return sum(top) / max(1, len(top))


def _team_league_rank(team: Any, team_by_id: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if team is None:
        return None
    live = getattr(team, "_trade_standings", None)
    if isinstance(live, dict) and int(live.get("gp") or 0) >= 5 and live.get("rank"):
        return int(live["rank"])
    for key in ("league_rank", "overall_rank", "standings_rank"):
        v = getattr(team, key, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    if not isinstance(team_by_id, dict) or len(team_by_id) < 2:
        return None
    ranked: List[Tuple[float, str]] = []
    for tid, tm in team_by_id.items():
        pct = _team_points_pct(tm)
        if pct is None:
            continue
        ranked.append((pct, str(tid)))
    if not ranked:
        return None
    ranked.sort(key=lambda x: -x[0])
    my_id = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
    for idx, (_, tid) in enumerate(ranked, start=1):
        if tid == my_id:
            return idx
    return None


def _projected_finish_risk(team: Any, *, team_by_id: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    p_pct = _team_points_pct(team)
    core = _team_core_strength(team)
    window = _team_window(team)
    score = 0.0
    if p_pct is not None:
        if p_pct < 0.42:
            score += 16.0
        elif p_pct < 0.47:
            score += 11.0
        elif p_pct < 0.51:
            score += 7.0
        elif p_pct > 0.58:
            score -= 8.0
        elif p_pct > 0.54:
            score -= 4.0
    if core > 0:
        if core < 70:
            score += 10.0
        elif core < 74:
            score += 6.0
        elif core > 83:
            score -= 9.0
        elif core > 79:
            score -= 5.0
    if window == "rebuild":
        score += 5.0
    elif window == "declining":
        score += 3.0
    elif window == "contender":
        score -= 4.0
    league_rank = _team_league_rank(team, team_by_id)
    if league_rank is not None:
        n_teams = len(team_by_id) if isinstance(team_by_id, dict) and team_by_id else 32
        if league_rank >= max(1, n_teams - 4):
            score += 12.0
        elif league_rank >= max(1, n_teams - 8):
            score += 6.0
        elif league_rank <= 5:
            score -= 6.0
    return {
        "projected_risk_score": round(score, 2),
        "points_pct": round(p_pct, 4) if p_pct is not None else None,
        "core_strength": round(core, 2),
        "window": window,
        "league_rank": league_rank,
    }


def _player_ovr(player: Any) -> float:
    """Canonical 0–99 display OVR aligned with roster / FA economy."""
    try:
        from app.sim_engine.entities.player import player_current_ovr_01

        return float(player_current_ovr_01(player)) * 99.0
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            v = float(fn())
        except Exception:
            v = 0.0
    else:
        v = _safe_float(getattr(player, "overall", None), _safe_float(fn, 0.0))
    if v <= 1.5:
        return v * 99.0
    return v


def _player_potential_ovr(player: Any, current_ovr: float) -> float:
    """Development ceiling on the same 0–99 scale as current OVR."""
    try:
        from app.sim_engine.entities.chapter_attributes import get_player_chapters

        chapters = get_player_chapters(player)
        pot = chapters.get("potential")
        if pot is not None:
            return max(float(current_ovr), float(pot))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        for key in ("dev_potential", "potential", "pot"):
            if key in ratings:
                v = _safe_float(ratings.get(key), 0.0)
                if v > 1.5:
                    return v
                if v > 0:
                    return v * 99.0
    pot = _safe_float(getattr(player, "potential", 0), 0.0)
    if pot <= 1.5 and pot > 0:
        return pot * 99.0
    if pot > 1.5:
        return pot
    return current_ovr


def _trade_valuation_ovr(
    player: Any,
    *,
    ovr: float,
    pot: float,
    age: int,
    is_prospect_val: bool,
) -> float:
    """Ability anchor for trade value — blends ceiling for pipeline and young pros."""
    if is_prospect_val:
        val = prospect_valuation_ovr(player, ovr_display=ovr, pot_display=pot)
        upside = max(0.0, pot - ovr)
        if upside >= 5.0:
            # Younger = more of the ceiling priced in (an 18-year-old #1 pick is bought
            # for who he becomes). Was a flat 0.50 capped at +9.
            lift = 0.64 if age <= 19 else 0.56 if age <= 21 else 0.48
            val = max(val, ovr + min(13.0, upside * lift), pot * 0.88)
        return min(val, pot * 0.95)

    upside = max(0.0, pot - ovr)
    if upside < 2.5:
        return ovr
    if age >= 27:
        return ovr
    if age == 26:
        return min(max(ovr, ovr + min(2.0, upside * 0.28)), pot * 0.94)

    if ovr < 74:
        ceiling = max(ovr, pot * 0.88)
    elif ovr < 80:
        ceiling = max(ovr, pot * 0.93)
    else:
        ceiling = max(ovr, pot * 0.96)

    if age <= 21:
        weight = min(0.62, 0.30 + upside * 0.032)
        max_lift = min(upside * 0.72, 12.0)
    elif age <= 23:
        weight = min(0.52, 0.24 + upside * 0.026)
        max_lift = min(upside * 0.60, 10.0)
    elif age <= 25:
        weight = min(0.40, 0.16 + upside * 0.020)
        max_lift = min(upside * 0.48, 7.5)
    else:
        weight = 0.22
        max_lift = min(upside * 0.35, 4.0)

    blended = ovr * (1.0 - weight) + ceiling * weight
    val = min(max(ovr, blended), ovr + max_lift, pot * 0.97)
    if age <= 23 and ovr >= 76 and upside >= 5.0:
        val = max(val, ovr + min(8.0, upside * 0.58))
    # v15: on the exponential curve every projected point is +16% value, so the
    # projection premium for an established NHL player is capped tighter.
    hard_lift = 5.0 if age <= 21 else 4.0 if age <= 23 else 2.5
    return min(val, ovr + hard_lift)


def _talent_base(ovr: float) -> float:
    """
    Exponential ability curve (v15). Real NHL trade markets are winner-take-all: one star
    is worth far more than several good players, so value roughly multiplies by ~1.16 per
    OVR point above 80.

    Anchors (prime age, fair contract, before context):
      70 4th line ~10 · 75 bottom-6 ~18 · 78 middle-6 ~30 · 80 top-6 ~38
      83 1st line / top pair ~60 · 86 star ~94 · 88 elite ~126
      90 superstar ~170 · 92 MVP-level ~230 · 94 generational ~310
    """
    o = float(ovr or 0.0)
    if o <= 0:
        return 3.0
    if o < 70.0:
        anchor = 3.0 + max(0.0, o - 60.0) * 0.45
    elif o < 80.0:
        # Everyday NHLers (fourth line through middle six) sit under a late first.
        anchor = 6.0 + (o - 70.0) * 1.7
    elif o < 92.0:
        anchor = 32.0 * math.exp(0.155 * (o - 80.0))
    else:
        # Past MVP level the curve keeps climbing, but more gently.
        anchor = 32.0 * math.exp(0.155 * 12.0) * math.exp(0.10 * (o - 92.0))
    return max(3.0, float(anchor))


def _expected_cap_m(ovr: float) -> float:
    """Fair-market AAV expectation (millions) for contract surplus/deficit math.

    Mirrors the rating baseline of the franchise FA market
    (backend contract_economy.compute_market_value) so a contract the league
    itself would sign is not priced as a cap dump. Production / age multipliers
    are left out — age and term risk are handled by the trade-value mods.
    """
    o = float(ovr or 0.0)
    if o < 70.0:
        out = LEAGUE_MINIMUM_AAV_M + max(0.0, o - 55.0) * 0.035
    elif o < 78.0:
        out = 1.15 + (o - 70.0) * 0.18
        if o < 75.0:
            out = min(out, LEAGUE_MINIMUM_AAV_M + 0.85 + max(0.0, o - 65.0) * 0.10)
    else:
        # Real market (v15.1): 80 ~$3.6M · 84 ~$5.6M · 86 ~$7.2M · 88 ~$8.9M · 90 ~$10.5M ·
        # 93 ~$13M. The lower curve called fairly-paid regulars "overpaid".
        out = LEAGUE_MINIMUM_AAV_M + (o - 58.0) * 0.13 + max(0.0, o - 82.0) * 0.70
    return _clamp(out, LEAGUE_MINIMUM_AAV_M, 16.0)


def reduced_trade_value_fallback(player: Any, *, reason: str = "") -> float:
    """Deterministic safe fallback: scaled talent base, capped well below star territory."""
    ovr = _player_ovr(player)
    try:
        base = float(_talent_base(ovr))
    except Exception:
        base = max(TRADE_VALUE_FALLBACK_FLOOR, min(35.0, float(ovr or 0.0) * 0.4))
    total = _clamp(
        base * TRADE_VALUE_FALLBACK_SCALE,
        TRADE_VALUE_FALLBACK_FLOOR,
        TRADE_VALUE_FALLBACK_CEIL,
    )
    logger.warning(
        "trade_value fallback used (total=%.2f ovr=%.1f reason=%s)",
        total,
        ovr,
        reason or "unknown",
    )
    return float(total)


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None:
        return _safe_int(getattr(ident, "age", 0), 25)
    return _safe_int(getattr(player, "age", 25), 25)


def _player_pos(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", None) if ident else getattr(player, "position", "")
    s = str(getattr(pos, "value", pos) or "").upper()
    if s in ("LW", "RW", "W", "F"):
        return "W"
    if s in ("C",):
        return "C"
    if s in ("D", "LD", "RD"):
        return "D"
    if s in ("G",):
        return "G"
    return s or "F"


def _team_window(team: Any) -> str:
    for key in ("gm_window", "window"):
        w = str(getattr(team, key, "") or "").lower()
        if w in ("rebuild", "contender", "declining", "emerging"):
            return w
    st = str(getattr(team, "status", "") or "").lower()
    arch = str(getattr(team, "archetype", "") or "").lower()
    blob = st + " " + arch
    if "rebuild" in blob or "tank" in blob:
        return "rebuild"
    if "contend" in blob or "win" in blob:
        return "contender"
    if "declin" in blob:
        return "declining"
    return "emerging"


def _team_cap_pressure(team: Any) -> str:
    return str(getattr(team, "cap_pressure_tier", getattr(team, "cap_pressure", "moderate")) or "moderate").lower()


def _contract_field(player: Any, *keys: str, default: Any = None) -> Any:
    """Read a contract field from dict or object contracts (real-NHL uses dicts)."""
    c = getattr(player, "contract", None)
    for obj in (c, player):
        if obj is None:
            continue
        for key in keys:
            if isinstance(obj, dict):
                if key in obj and obj.get(key) is not None:
                    return obj.get(key)
            else:
                if hasattr(obj, key):
                    val = getattr(obj, key, None)
                    if val is not None:
                        return val
    return default


def _contract_years(player: Any) -> int:
    for key in ("years_remaining", "term_remaining", "remaining_years", "term", "years"):
        v = _safe_int(_contract_field(player, key, default=0), 0)
        if v > 0:
            return v
    # Derive from expiry_year when remaining term was lost on a dict contract.
    expiry_year = _safe_int(_contract_field(player, "expiry_year", default=0), 0)
    if expiry_year > 0:
        # Franchise season is usually stamped on the contract effective year.
        eff = _safe_int(
            _contract_field(player, "effective_season", "season_year", default=0),
            0,
        )
        if eff > 0:
            return max(0, expiry_year - eff)
    return 0


def _expiry_status(player: Any) -> str:
    for key in ("expiry_status", "ufa_rfa_status", "rights_status", "rights"):
        val = str(_contract_field(player, key, default="") or "").strip().upper()
        if val in ("UFA", "RFA", "ELC"):
            return val
    ctype = str(_contract_field(player, "contract_type", "type", default="") or "").strip().upper()
    if ctype == "ELC":
        return "ELC"
    if _is_elc_contract(player):
        return "ELC"
    age = _player_age(player)
    return "RFA" if age < 27 else "UFA"


def _contract_type_label(player: Any) -> str:
    ctype = str(_contract_field(player, "contract_type", "type", default="") or "").strip().upper()
    return ctype


def _is_elc_contract(player: Any) -> bool:
    """True ELC only — do not treat every cheap under-25 deal as entry-level."""
    if _contract_type_label(player) == "ELC":
        return True
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        if bool(c.get("is_elc") or c.get("entry_level") or c.get("elc")):
            return True
        label = str(c.get("type") or c.get("contract_type") or "").upper()
        if label in ("ELC", "ENTRY", "ENTRY_LEVEL"):
            return True
        # Spotrac / real-NHL tags sometimes set rights without type.
        if str(c.get("source") or "").lower().startswith("real_nhl") and bool(c.get("entry_level_contract")):
            return True
    elif c is not None:
        if bool(getattr(c, "is_elc", False) or getattr(c, "entry_level", False)):
            return True
        label = str(getattr(c, "type", None) or getattr(c, "contract_type", None) or "").upper()
        if label in ("ELC", "ENTRY", "ENTRY_LEVEL"):
            return True
    # Heuristic last resort: league-min AAV + young + short remaining term only.
    cap = player_cap_hit_millions(player)
    years = _contract_years(player)
    age = _player_age(player)
    if 0 < cap <= 0.95 and age <= 24 and 0 < years <= 3:
        return True
    return False


def _injury_games_out(player: Any) -> int:
    for key in ("_world_injury_games_remaining", "injury_games_remaining", "games_out", "games_remaining"):
        val = getattr(player, key, None)
        if val is not None:
            try:
                g = int(val)
                if g > 0:
                    return g
            except (TypeError, ValueError):
                continue
    health = getattr(player, "health", None)
    if health is not None:
        val = getattr(health, "injury_games_remaining", None) or getattr(health, "games_out", None)
        if val is not None:
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                pass
    return 0


def _injury_value_mod(
    player: Any,
    *,
    pos: str,
    ovr: float,
    window: str,
    deadline_phase: float,
    need_mod: float,
) -> float:
    if not is_player_injured(player):
        return 0.0
    games = _injury_games_out(player)
    severity = min(1.0, games / 30.0) if games > 0 else 0.45
    discount = 2.0 + 7.0 * severity
    if ovr >= 82:
        discount += 1.5
    if window == "contender" and deadline_phase > 0.4 and games > 0 and games <= 14 and ovr >= 76:
        discount *= 0.55
    if need_mod >= 6.0 and pos == "G" and games <= 21:
        discount *= 0.65
    return -discount


def _season_games_remaining(league: Any, season_games: int = 82) -> int:
    snap = getattr(league, "_cpu_standings_snapshot", None) or {}
    gps = []
    for row in snap.values() if isinstance(snap, dict) else []:
        try:
            gps.append(int((row or {}).get("gp") or 0))
        except Exception:
            continue
    gp = (sum(gps) / len(gps)) if gps else 0.0
    return max(0, int(round(season_games - gp)))


def _injury_value_pct(player: Any, *, window: str, league: Any, years: int, deadline_phase: float) -> float:
    """Share of a player's trade value lost to his current injury.

    A flat few-point discount meant a season-ending injury barely mattered, so you
    could hand a CPU club a star who wouldn't play again this year at near full price.
    Now the club loses the share of this season he'll miss (weighted by how much it
    cares about this season), spread over his term, plus a re-injury risk."""
    if not is_player_injured(player):
        return 0.0
    games = _injury_games_out(player)
    remaining = _season_games_remaining(league)
    if games <= 0:
        return 0.03
    if remaining <= 0:
        lost = 0.0  # offseason: he'll be back by camp, mostly
    else:
        lost = min(1.0, games / max(8.0, float(remaining)))
    now_weight = {"contender": 0.9, "win_now": 0.9, "rebuild": 0.35}.get(str(window or "").lower(), 0.6)
    if deadline_phase > 0.5 and now_weight >= 0.6:
        now_weight = min(1.0, now_weight + 0.1)
    term = max(1, int(years or 1))
    pct = lost * now_weight / (term ** 0.5)
    pct += 0.08 if games >= 40 else 0.04 if games >= 15 else 0.01
    return max(0.0, min(0.75, pct))


def _elc_value_mod(
    player: Any,
    *,
    ovr: float,
    age: int,
    pot: float,
    cap_hit: float,
    expiry: str,
    window: str,
    cap_pressure: str,
    is_prospect_val: bool = False,
) -> float:
    if expiry != "ELC" and not _is_elc_contract(player):
        return 0.0
    mod = 0.0
    if window == "rebuild":
        if age <= 22:
            if is_prospect_val:
                mod += 1.25
            else:
                mod += 2.5 + min(2.5, max(0.0, pot - ovr) * 0.12)
        elif age <= 24:
            mod += 1.5
    elif window == "contender":
        if age <= 23 and ovr >= 72:
            mod += 2.0
        elif age <= 22:
            mod -= 1.5
    if cap_pressure in ("cap_hell", "critical") and cap_hit <= 1.05:
        mod += 2.5
    elif cap_pressure in ("cap_hell", "critical") and cap_hit > 2.0:
        mod -= 1.0
    return mod


def _rental_market_mod(
    *,
    ovr: float,
    age: int,
    years: int,
    expiry: str,
    pos: str,
    window: str,
    deadline_phase: float,
    need_mod: float,
) -> float:
    if years > 1 or expiry != "UFA" or ovr < 74:
        return 0.0
    rental = 2.5 + max(0.0, ovr - 74.0) * 0.45
    if age >= 34:
        rental += 1.5
    elif age >= 30:
        rental += 0.8
    if window == "contender":
        rental *= 0.85 + 0.95 * max(0.0, deadline_phase)
        if deadline_phase >= 0.75:
            rental += 1.25
        if need_mod >= 5.0:
            rental += 2.0
    elif window == "rebuild":
        rental *= 0.35
    elif window == "declining":
        rental *= 0.55
    if pos == "G" and need_mod >= 6.0:
        rental += 2.5
    if deadline_phase > 0.65 and ovr >= 82:
        rental += 2.0
    return rental


def _bad_contract_score(player: Any, ovr: float, cap_hit: float, years: int, age: int) -> float:
    expected = _expected_cap_m(ovr)
    if cap_hit <= expected + 0.75:
        return 0.0
    overpay = cap_hit - expected
    ratio = cap_hit / max(0.75, expected)
    term_risk = 1.0
    if years >= 5 and age >= 30:
        term_risk = 1.25
    elif years >= 4 and age >= 32:
        term_risk = 1.35
    score = max(0.0, (overpay / max(0.5, expected)) * term_risk * max(0.0, ratio - 1.0))
    bad_type = _contract_field(player, "bad_contract_type", default=None)
    if bad_type:
        score = max(score, 0.35)
    try:
        tagged = float(
            getattr(player, "bad_contract_score", 0)
            or _contract_field(player, "bad_contract_score", default=0)
            or 0
        )
        if tagged > 0:
            score = max(score, tagged)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return min(2.0, score)


def _contract_burden(
    *,
    cap_hit: float,
    years: int,
    age: int,
    expected_cap: float,
    acquiring_window: str,
    cap_pressure: str,
) -> float:
    """Dead-money cost of an overpaid contract (<= 0), applied outside the context clamp.

    Scales with overpay dollars x remaining seasons so long albatross deals can
    push a mid-tier player below zero. `cap_hit` must already be net of retention.
    """
    tolerance = max(0.75, expected_cap * 0.20)
    overpay = cap_hit - expected_cap - tolerance
    if overpay <= 0:
        return 0.0
    term = max(1, int(years or 0))
    term_risk = 1.0
    if age >= 34:
        term_risk = 1.35
    elif age >= 31 and term >= 4:
        term_risk = 1.2
    burden = overpay * term * term_risk * CONTRACT_BURDEN_PER_M_YEAR
    if acquiring_window == "rebuild" and cap_pressure not in ("cap_hell", "critical"):
        burden *= 0.85  # rebuilders with room absorb dumps for assets
    elif acquiring_window == "contender":
        burden *= 1.1
    if cap_pressure in ("cap_hell", "critical"):
        burden *= 1.25
    return -min(CONTRACT_BURDEN_MAX, burden)


def _ledger_production(player: Any, league: Any) -> Optional[Tuple[float, int]]:
    """(production score, gp) from the live season ledger — session.player_season_stats,
    mirrored onto ``league.player_season_stats``. player.season_stats is year-keyed during
    a franchise, so the flat read below never saw the current season and trade value was
    frozen on last year's real-NHL line."""
    reg = getattr(league, "player_season_stats", None) if league is not None else None
    if not isinstance(reg, dict):
        return None
    row = reg.get(str(getattr(player, "id", "") or ""))
    if not isinstance(row, dict):
        return None
    gp = _safe_int(row.get("gp"), 0)
    if gp <= 0:
        return None
    if _player_pos(player) == "G":
        ga = _safe_float(row.get("ga"), 0.0)
        sa = _safe_float(row.get("sa") or row.get("shots_against"), 0.0)
        if sa > 0:
            sv = 1.0 - ga / sa
        else:
            # Ledger tracks GA but not shots: map GAA onto the SV% scale (~0.011 SV% per goal).
            sv = 0.903 + (3.0 - ga / gp) * 0.011
        return _clamp((sv - 0.88) * 120.0, 0.0, 18.0), gp
    pts = _safe_float(row.get("pts"), _safe_float(row.get("g"), 0) + _safe_float(row.get("a"), 0))
    return _clamp((pts / gp) * 14.0, 0.0, 16.0), gp


def _production_score(player: Any, league: Any = None) -> float:
    prior = _prior_production_score(player)
    cur = _ledger_production(player, league)
    if cur is None:
        return prior
    score, gp = cur
    w = gp / (gp + 20.0)  # ~50/50 at 20 GP, mostly this season by the deadline
    return score * w + prior * (1.0 - w)


def _prior_production_score(player: Any) -> float:
    st = getattr(player, "season_stats", None) or {}
    if isinstance(st, dict) and "gp" not in st and "pts" not in st and "points" not in st:
        # Year-keyed sim sync or empty — do not treat nested seasons as flat gp.
        st = {}
    if not isinstance(st, dict) or not st:
        # Prefer boxed import stats (prior NHL season) before career archive.
        imp = getattr(player, "real_nhl_import_stats", None) or {}
        if isinstance(imp, dict) and imp:
            st = {
                "gp": imp.get("gamesPlayed") or imp.get("gp"),
                "g": imp.get("goals") or imp.get("g"),
                "a": imp.get("assists") or imp.get("a"),
                "pts": imp.get("points") or imp.get("pts"),
                "sv_pct": imp.get("savePct") or imp.get("sv_pct"),
                "savePct": imp.get("savePct"),
            }
        else:
            career = getattr(player, "career_stats", None) or {}
            seasons = career.get("seasons") if isinstance(career, dict) else None
            if isinstance(seasons, list) and seasons:
                st = seasons[-1] if isinstance(seasons[-1], dict) else {}
    if isinstance(st, dict) and st:
        gp = max(1, _safe_int(st.get("gp") or st.get("gamesPlayed"), 1))
        pts = _safe_float(
            st.get("pts"),
            _safe_float(st.get("points"), _safe_float(st.get("g"), 0) + _safe_float(st.get("a"), 0)),
        )
        ppg = pts / gp
        if _player_pos(player) == "G":
            sv = _safe_float(st.get("sv_pct") or st.get("savePct"), 0.905)
            return _clamp((sv - 0.88) * 120.0, 0.0, 18.0)
        return _clamp(ppg * 14.0, 0.0, 16.0)
    return 0.0


def _clause_penalty(player: Any) -> float:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        nmc = bool(c.get("no_move_clause") or c.get("nmc"))
        ntc = bool(c.get("no_trade_clause") or c.get("ntc"))
        mntc = _safe_int(c.get("modified_no_trade_teams") or c.get("mntc"), 0)
    else:
        clauses = getattr(c, "clauses", None) if c else None
        nmc = bool(
            getattr(clauses, "noMoveClause", False)
            if clauses
            else getattr(c, "no_move_clause", False)
            if c
            else False
        )
        ntc = bool(
            getattr(clauses, "noTradeClause", False)
            if clauses
            else getattr(c, "no_trade_clause", False)
            if c
            else False
        )
        mntc = _safe_int(
            getattr(clauses, "modifiedNoTradeTeams", 0)
            if clauses
            else getattr(c, "modified_no_trade_teams", 0)
            if c
            else 0
        )
    if nmc:
        return 6.0
    # Modified lists also set the no-trade flag. Price the list, not a full NTC.
    if mntc > 0:
        return 2.5
    if ntc:
        return 4.0
    return 0.0


def _ntc_waived_for_player(player: Any, context: Optional[Dict[str, Any]] = None) -> tuple[bool, float]:
    """Return (waived, value_penalty_pct) from package/context waive markers."""
    ctx = context or {}
    pid = str(getattr(player, "id", "") or "")
    if bool(ctx.get("ntc_waived")):
        return True, float(ctx.get("ntc_value_penalty_pct") or 0.08)
    waivers = ctx.get("ntc_waivers") or {}
    if isinstance(waivers, dict) and pid:
        entry = waivers.get(pid)
        if isinstance(entry, dict) and bool(entry.get("accepted")):
            return True, float(entry.get("value_penalty_pct") or 0.08)
        if entry is True:
            return True, 0.08
    return False, 0.0


_NEEDS_MODEL = TeamNeeds()


def prospect_floor_value(age: int, ovr: float, pot: float, confidence: float = 0.5) -> float:
    """Minimum trade value of a young high-ceiling player (v14), 0 when not eligible.

    Scales strongly with potential (80 POT ~16 · 86 ~42 · 90 ~78 · 94 ~112), trimmed
    for age (less runway) and for older players still far from their ceiling.
    """
    age_i = int(age or 0)
    pot_f = float(pot or 0.0)
    if age_i <= 0 or age_i > PROSPECT_PREMIUM_MAX_AGE or pot_f < PROSPECT_PREMIUM_MIN_POT:
        return 0.0
    anchors = _PROSPECT_FLOOR_ANCHORS
    if pot_f >= anchors[-1][0]:
        base = anchors[-1][1]
    else:
        base = anchors[0][1]
        for (p0, v0), (p1, v1) in zip(anchors, anchors[1:]):
            if p0 <= pot_f <= p1:
                base = v0 + (v1 - v0) * (pot_f - p0) / (p1 - p0)
                break
    age_f = 1.0 if age_i <= 19 else 0.96 if age_i == 20 else 0.90 if age_i == 21 else 0.82 if age_i == 22 else 0.72
    if age_i >= 21 and pot_f - float(ovr or 0.0) > 12.0:
        age_f *= 0.88
    conf = max(0.0, min(1.0, float(confidence if confidence is not None else 0.5)))
    return round(base * age_f * (0.9 + 0.2 * conf), 2)


def _org_location(player: Any, source_team: Any) -> str:
    """'nhl' / 'ahl' / 'echl' / 'prospect' / '' — where the player sits in his org."""
    pid = str(getattr(player, "id", "") or "")
    if source_team is not None and pid:
        for attr, loc in (
            ("roster", "nhl"),
            ("injured_reserve", "nhl"),
            ("scratches", "nhl"),
            ("ahl_roster", "ahl"),
            ("echl_roster", "echl"),
            ("prospect_pool", "prospect"),
        ):
            for p in getattr(source_team, attr, None) or []:
                if p is player or str(getattr(p, "id", "") or "") == pid:
                    return loc
    loc = str(getattr(player, "roster_location", "") or getattr(player, "pool_context", "") or "").lower()
    if loc in ("ahl", "echl", "nhl"):
        return loc
    return ""


_ROLE_RANK_CACHE: Dict[int, Tuple[Tuple[Any, ...], Dict[str, Tuple[str, int]]]] = {}


def _team_depth_ranks(team: Any) -> Dict[str, Tuple[str, int]]:
    """{player_id: (group 'F'/'D', 1-based OVR rank in group)} for a team's NHL roster."""
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    sig = tuple((str(getattr(p, "id", "") or ""), getattr(p, "_ovr_memo", None)) for p in roster)
    key = id(team)
    hit = _ROLE_RANK_CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    groups: Dict[str, List[Tuple[float, str]]] = {"F": [], "D": []}
    for p in roster:
        pos = _player_pos(p)
        grp = "D" if pos == "D" else ("F" if pos in ("C", "W", "F", "LW", "RW") else "")
        if not grp:
            continue
        try:
            groups[grp].append((_player_ovr(p), str(getattr(p, "id", "") or "")))
        except Exception:
            continue
    ranks: Dict[str, Tuple[str, int]] = {}
    for grp, rows in groups.items():
        rows.sort(key=lambda r: -r[0])
        for idx, (_, pid) in enumerate(rows, start=1):
            ranks[pid] = (grp, idx)
    if len(_ROLE_RANK_CACHE) > 256:
        _ROLE_RANK_CACHE.clear()
    _ROLE_RANK_CACHE[key] = (sig, ranks)
    return ranks


def _depth_role(player: Any, source_team: Any, org_loc: str) -> str:
    """Depth role on his current NHL club ('' when top-6 F / top-4 D / not on NHL roster)."""
    if org_loc != "nhl" or source_team is None:
        return ""
    row = _team_depth_ranks(source_team).get(str(getattr(player, "id", "") or ""))
    if row is None:
        return ""
    grp, rank = row
    if grp == "F":
        if rank <= 6:
            return ""
        return "third_line_f" if rank <= 9 else "fourth_line_f"
    if rank <= 4:
        return ""
    return "third_pair_d" if rank <= 6 else "extra_d"


def _role_contract_burden(cap_hit: float, years: int, role_aav: float) -> float:
    """Negative value when a depth player's AAV exceeds what his role is worth (<= 0)."""
    overpay = float(cap_hit) - float(role_aav) - ROLE_BURDEN_TOLERANCE_M
    if overpay <= 0:
        return 0.0
    return -min(ROLE_BURDEN_MAX, overpay * max(1, int(years or 0)) * ROLE_BURDEN_PER_M_YEAR)


def _position_value_mult(pos: str, val_ovr: float) -> float:
    """Scarcity by position: top centres and No.1 defencemen cost the most; goalies are
    notoriously volatile and the market pays far less for them than for skaters of the
    same rating (elite starters excepted)."""
    if pos == "C":
        return 1.08
    if pos == "D":
        return 1.05 if val_ovr >= 82 else 1.0
    if pos == "G":
        # The crease is the biggest single lever on team strength in this sim (see
        # trades/lineup_impact.py), so a real starter is priced like a top skater.
        # The old flat 0.55 let you buy an 85+ starter for a mid prospect. Backups and
        # fringe goalies still carry the market discount.
        if val_ovr >= 90:
            return 1.0
        if val_ovr >= 85:
            return 0.9
        if val_ovr >= 80:
            return 0.72
        return 0.55
    return 1.0


def _age_value_mult(age: int, val_ovr: float, pos: str) -> float:
    """Decline curve. Teams buy the next five years, so a 33-year-old is a short window
    no matter how good he is today. Goalies age a little more gracefully."""
    a = int(age or 0)
    table = {30: 0.93, 31: 0.86, 32: 0.78, 33: 0.70, 34: 0.62, 35: 0.55}
    if a < 30:
        return 1.0
    mult = table.get(a, 0.48)
    if pos == "G":
        return min(1.0, mult + 0.06)
    if val_ovr >= 90:
        mult = min(1.0, mult + 0.05)
    return mult


def _term_surplus_value(
    *,
    val_ovr: float,
    age: int,
    cap_hit: float,
    expected_cap: float,
    years: int,
    expiry: str,
    is_prospect_val: bool,
) -> float:
    """Value of cost control: cap savings per season x seasons of control.

    A star on a cheap long deal is the most valuable thing in a capped league; the same
    player on an expiring deal is a rental. RFA years after the deal ends add a little
    (the club still controls him). Only good players produce meaningful surplus.
    """
    if is_prospect_val and cap_hit <= 0.05:
        return 0.0
    surplus_m = expected_cap - cap_hit
    if surplus_m <= 0.25:
        return 0.0
    yrs = max(0, int(years or 0))
    control = min(yrs, 6)
    if str(expiry).upper() in ("RFA", "ELC") and age <= 25:
        control += 1.5  # still controlled after the deal (RFA rights)
    if control <= 0:
        return 0.0
    tier = max(0.12, min(1.35, (float(val_ovr) - 74.0) / 11.0))
    value = surplus_m * min(control, 7.0) * 1.7 * tier
    return round(min(45.0, value), 2)


def evaluate_player_asset_value(
    player: Any,
    source_team: Any,
    acquiring_team: Any,
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    retained_pct: float = 0.0,
) -> Dict[str, Any]:
    try:
        # House-rule negative asset (Brady Tkachuk chaos) — keep import soft.
        if bool(getattr(player, "brady_tkachuk_chaos", False)) or bool(
            getattr(player, "locker_room_cancer", False)
        ):
            try:
                nhl_id = int(getattr(player, "nhl_player_id", 0) or 0)
            except Exception:
                nhl_id = 0
            name = str(getattr(getattr(player, "identity", None), "name", "") or "").lower()
            if nhl_id == 8480801 or ("brady" in name and "tkachuk" in name) or bool(
                getattr(player, "brady_tkachuk_chaos", False)
            ):
                return {
                    "total": -42.0,
                    "base": -42.0,
                    "context_mod": 0.0,
                    "tier": "negative",
                    "brady_tkachuk_chaos": True,
                    "risk_flags": [
                        "Locker-room CANCER",
                        "Active substance / rehab storyline",
                        "Negative asset — clubs pay to move him",
                    ],
                    "contract_flags": ["Toxic asset"],
                    "explain": [
                        "House rule: Brady Tkachuk is a negative trade asset",
                        "CANCER tag depresses every offer sheet",
                    ],
                    "retained_pct_supported": True,
                }
        res = _evaluate_player_asset_value_impl(
            player,
            source_team,
            acquiring_team,
            league,
            context=context,
            retained_pct=retained_pct,
        )
        # Team identity: CPU clubs pay a little more for players who fit how they
        # play (a run-and-gun club chases speed and finishing; a heavy club chases
        # size and bite) and a little less for poor fits. User club unaffected.
        try:
            ident = getattr(acquiring_team, "team_identity", None) if acquiring_team is not None else None
            if isinstance(ident, dict) and not ident.get("is_user") and isinstance(res, dict):
                from app.sim_engine.systems.team_identity import fit_reason, player_identity_fit

                fit = player_identity_fit(ident, player)
                tot = float(res.get("total") or 0.0)
                if tot > 0:
                    mult = 1.0 + 0.14 * (fit - 0.5)
                    res["total"] = round(tot * mult, 3)
                    res["identity_fit"] = round(fit, 3)
                    note = fit_reason(ident, player, fit)
                    if note:
                        res["explain"] = list(res.get("explain") or []) + [note]
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return res
    except Exception as exc:
        pid = str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")
        logger.exception(
            "evaluate_player_asset_value failed for player_id=%s: %s",
            pid,
            exc,
        )
        fb = reduced_trade_value_fallback(player, reason=f"{type(exc).__name__}:{exc}")
        return {
            "total": fb,
            "base": fb,
            "context_mod": 0.0,
            "tier": player_value_tier(fb),
            "fallback": True,
            "fallback_reason": str(exc),
            "retained_pct_supported": True,
        }


def _evaluate_player_asset_value_impl(
    player: Any,
    source_team: Any,
    acquiring_team: Any,
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    retained_pct: float = 0.0,
) -> Dict[str, Any]:
    ctx = context or {}
    ovr = _player_ovr(player)
    age = _player_age(player)
    pos = _player_pos(player)
    full_cap_hit = player_cap_hit_millions(player)
    retained_frac = _clamp(float(retained_pct or 0.0), 0.0, 50.0) / 100.0
    # Acquiring club only carries the non-retained share — price the contract on that.
    cap_hit = full_cap_hit * (1.0 - retained_frac)
    years = _contract_years(player)
    expiry = _expiry_status(player)
    pot = _player_potential_ovr(player, ovr)
    is_prospect_val = is_prospect_for_valuation(player, age=age, ovr_display=ovr)
    val_ovr = _trade_valuation_ovr(
        player,
        ovr=ovr,
        pot=pot,
        age=age,
        is_prospect_val=is_prospect_val,
    )
    contract_ref = val_ovr
    talent_fit = min(1.0, max(0.35, val_ovr / 84.0))

    base_core = round(_talent_base(val_ovr), 2)

    age_mod = 0.0
    if age <= 22:
        age_mod = 3.0 if val_ovr >= 76 else 1.5 if val_ovr >= 70 else 0.5
    elif age <= 26:
        age_mod = 2.0 if val_ovr >= 74 else 0.5
        if age >= 25 and val_ovr < 88 and pot <= ovr + 3.0:
            age_mod -= 1.25
    elif age <= 30:
        age_mod = 1.0
    elif age <= 33:
        # Elite stars still in their window; do not dump them solely for age 31–33.
        age_mod = -0.5 if val_ovr >= 88 else (-1.0 if val_ovr >= 84 else -2.0)
    else:
        age_mod = -5.0 - (age - 33) * 0.8
        if val_ovr >= 90:
            age_mod = max(age_mod, -3.5)
        elif val_ovr >= 86:
            age_mod = max(age_mod, -5.5)

    upside = max(0.0, pot - ovr)
    residual_upside = max(0.0, pot - val_ovr)
    prospect_upside = _prospect_upside_score(player, ovr, age, pot)
    if is_prospect_val:
        # Ceiling is already in base_core — keep only a small scout-tier nudge.
        prospect_upside *= 0.35
        potential_mod = _clamp(prospect_upside, 0.0, 4.0) if age <= 25 else 0.0
    elif age <= 25:
        potential_mod = _clamp(residual_upside * 0.12, 0.0, 4.0) + prospect_upside * 0.65
        if val_ovr < 76:
            potential_mod = min(potential_mod, 2.0 + prospect_upside * 0.75)
        # Soft cap: residual upside nudges only — talent anchor carries most of the ceiling.
        potential_mod = min(potential_mod, 8.0 if val_ovr >= 78 else 6.5)
    else:
        potential_mod = _clamp(residual_upside * 0.06, 0.0, 2.0)

    production_mod = min(_production_score(player, league), 8.0) * talent_fit
    # Young high-upside assets are priced on projection until production proves otherwise.
    if age <= 23 and upside >= 6.0:
        production_mod *= 0.50
    elif age <= 25 and upside >= 4.0:
        production_mod *= 0.68
    elif age >= 28 and ovr >= 78 and val_ovr == ovr:
        production_mod = min(production_mod, 5.5)

    pos_mod = 0.0
    if pos == "C":
        pos_mod = 1.5
    elif pos == "D":
        pos_mod = 1.2
    elif pos == "G":
        pos_mod = 1.0

    expected_cap = _expected_cap_m(contract_ref)
    # Mild overpay only — overpay beyond the market tolerance is priced by
    # _contract_burden outside the context clamp. Positive surplus is priced per
    # remaining season below (term_surplus), outside the clamp.
    overpay_tolerance = max(0.75, expected_cap * 0.20)
    contract_mod = _clamp((expected_cap - cap_hit) * 1.4, -overpay_tolerance * 1.4, 0.0)
    term_surplus = _term_surplus_value(
        val_ovr=contract_ref, age=age, cap_hit=cap_hit, expected_cap=expected_cap,
        years=years, expiry=expiry, is_prospect_val=is_prospect_val,
    )
    if is_prospect_val and cap_hit <= 0.05:
        contract_mod = min(contract_mod, 1.25)
    # Cheap replacement / depth AAV is not franchise surplus — clamp positive
    # surplus for sub-star talent so league-min deals cannot mint premium assets.
    if contract_ref < 76 and contract_mod > 0:
        contract_mod = min(contract_mod, 1.5 if contract_ref >= 72 else 0.75)
    elif contract_ref < 80 and contract_mod > 0:
        contract_mod = min(contract_mod, 3.0)
    elif contract_ref < 85 and contract_mod > 0:
        contract_mod = min(contract_mod, 4.0)
    elif contract_ref < 88 and contract_mod > 0:
        contract_mod = min(contract_mod, 4.75)
    elif contract_ref < 91 and contract_mod > 0:
        contract_mod = min(contract_mod, 5.5)
    if contract_mod > 0 and age >= 30:
        contract_mod *= max(0.45, 1.0 - (age - 29) * 0.08)
    if years <= 1 and expiry == "UFA" and not is_prospect_val:
        # Elite rentals retain meaningful value; fringe UFAs take the full discount.
        if ovr >= 88:
            contract_mod -= 1.0
        elif ovr >= 82:
            contract_mod -= 2.0
        else:
            contract_mod -= 3.0

    needs = _NEEDS_MODEL.evaluate(acquiring_team, context=ctx)
    # Needs can nudge price but must not flatten OVR gaps (roster-spot dumps vs stars).
    need_scale = 3.2 if val_ovr >= 84 else 4.2 if val_ovr >= 78 else 5.0
    need_mod = 0.0
    if pos in ("C", "W"):
        need_mod = max(needs.get("top_line_forward", 0.0), needs.get("depth_forward", 0.0)) * need_scale * talent_fit
    elif pos == "D":
        need_mod = needs.get("top_4_defense", 0.0) * need_scale * talent_fit
    elif pos == "G":
        need_mod = needs.get("goalie", 0.0) * (need_scale + 0.8) * talent_fit
    # Bound need so positional desperation cannot turn fringe players into stars.
    if val_ovr < 76:
        need_mod = _clamp(need_mod, 0.0, 3.0)
    elif val_ovr < 82:
        need_mod = _clamp(need_mod, 0.0, 5.0)
    else:
        need_mod = _clamp(need_mod, 0.0, 4.5)

    window = _team_window(acquiring_team)
    cap_pressure = _team_cap_pressure(acquiring_team)
    deadline_phase = _safe_float(ctx.get("deadline_phase"), 0.0)

    elc_mod = _elc_value_mod(
        player,
        ovr=ovr,
        age=age,
        pot=pot,
        cap_hit=cap_hit,
        expiry=expiry,
        window=window,
        cap_pressure=cap_pressure,
        is_prospect_val=is_prospect_val,
    )

    cap_dump_mod = _contract_burden(
        cap_hit=cap_hit,
        years=years,
        age=age,
        expected_cap=expected_cap,
        acquiring_window=window,
        cap_pressure=cap_pressure,
    )

    injury_mod = _injury_value_mod(
        player,
        pos=pos,
        ovr=ovr,
        window=window,
        deadline_phase=deadline_phase,
        need_mod=need_mod,
    )
    # Real injury cost is applied proportionally below (the flat mod was clamped to
    # a few points); keep only a token amount inside the context stack.
    injury_pct = _injury_value_pct(player, window=window, league=league, years=years, deadline_phase=deadline_phase)
    if injury_pct > 0:
        injury_mod = max(injury_mod, -1.0)

    rental_mod = _rental_market_mod(
        ovr=ovr,
        age=age,
        years=years,
        expiry=expiry,
        pos=pos,
        window=window,
        deadline_phase=deadline_phase,
        need_mod=need_mod,
    )

    window_mod = 0.0
    if window == "rebuild":
        if age <= 23 and val_ovr >= 74:
            window_mod = 4.0
        elif age <= 23:
            window_mod = 2.0 + prospect_upside * 0.35
        elif age <= 26 and cap_hit <= expected_cap:
            window_mod = 2.0
        elif age >= 30:
            window_mod = -5.0
        else:
            window_mod = -1.5
    elif window == "contender":
        if 24 <= age <= 32 and ovr >= 80 and not is_prospect_val:
            window_mod = 3.5
        elif age <= 22 and (ovr < 78 or is_prospect_val):
            window_mod = -2.5
        elif age >= 33 and cap_hit > expected_cap:
            window_mod -= 4.0

    market_mod = rental_mod
    if not is_prospect_val and ovr >= 88:
        market_mod += 2.0
    if contract_mod >= 4.0 and market_mod > 1.5:
        market_mod = 1.5 + (market_mod - 1.5) * 0.55

    risk_mod = -_clause_penalty(player) * 0.75
    waived, waive_pct = _ntc_waived_for_player(player, ctx)
    waive_mod = 0.0
    if waived and waive_pct > 0:
        # Slight post-waive discount: player is movable but still burned leverage.
        waive_mod = -max(2.5, min(9.0, base_core * float(waive_pct)))
        risk_mod += waive_mod
    mult = _safe_float(getattr(player, "_systemic_trade_value_mult", 1.0), 1.0)
    crisis_mult = _safe_float(getattr(player, "_crisis_trade_value_mult", 1.0), 1.0)
    distressed = _safe_float(getattr(player, "_crisis_distressed_asset", 0.0), 0.0)
    if crisis_mult != 1.0:
        risk_mod += (crisis_mult - 1.0) * 8.0
    elif mult != 1.0:
        risk_mod += (mult - 1.0) * 5.0
    if distressed > 0:
        risk_mod -= min(18.0, distressed * 0.45)

    risk_flags: List[str] = []
    contract_flags: List[str] = []
    if bool(getattr(player, "_trade_demand_active", False) or getattr(player, "trade_demand_active", False)):
        stage = int(getattr(player, "_crisis_trade_stage", 0) or 0)
        if stage >= 4 or distressed > 0:
            risk_mod -= 12.0
            risk_flags.append("Distressed asset — may require sweetener to move")
        elif stage >= 3:
            risk_mod -= 9.0
            risk_flags.append("Trade demand crisis — leverage collapsing")
        elif stage >= 2:
            risk_mod -= 6.5
            risk_flags.append("Trade demand leaking — value falling")
        else:
            risk_mod -= 6.0 if bool(getattr(player, "locker_room_disruptor", False)) else 3.5
            risk_flags.append("Active trade demand — value depressed")
    if bool(getattr(player, "locker_room_disruptor", False)):
        risk_flags.append("Locker-room disruptor")
        risk_mod -= 4.0
    if waived:
        risk_flags.append("NTC waived — slightly reduced trade value")
        contract_flags.append("NTC_WAIVED")
    if _clause_penalty(player) >= 4.0:
        risk_flags.append("NTC/NMC limits trade options")
    if age >= 33 and cap_hit > expected_cap + 1.0:
        risk_flags.append("Aging expensive profile")
    if years >= 5 and age >= 31:
        contract_flags.append("Long term on older player")
    if cap_hit > expected_cap + 2.5:
        contract_flags.append("Above-market cap hit")
    elif cap_hit < expected_cap - 1.5 and years >= 2:
        contract_flags.append("Team-friendly deal")
    if expiry == "UFA" and years <= 1:
        contract_flags.append("Pending UFA")
    if expiry == "ELC" or _is_elc_contract(player):
        contract_flags.append("ELC — cost-controlled")
    if is_player_injured(player):
        games = _injury_games_out(player)
        risk_flags.append(f"Injured ({games}g out)" if games > 0 else "Currently injured")
    bad_score = _bad_contract_score(player, ovr, cap_hit, years, age)
    if bad_score >= 0.35:
        contract_flags.append("Cap dump / negative-value contract")
    if pos == "G" and age <= 27:
        risk_flags.append("Goalie volatility")

    context_parts = (
        age_mod,
        potential_mod,
        production_mod,
        contract_mod,
        need_mod,
        window_mod,
        risk_mod,
        pos_mod,
        elc_mod,
        injury_mod,
    )
    context_raw = sum(context_parts)
    # Context can nudge but must not erase star vs depth gaps. Bonuses and
    # penalties are bounded separately so a stack of bonuses cannot silently
    # absorb an injury / risk discount (and vice versa).
    if base_core >= 75.0:
        ctx_lo, ctx_hi = -10.0, 6.0
    elif base_core >= 45.0:
        ctx_lo, ctx_hi = -12.0, 8.0
    else:
        ctx_lo, ctx_hi = -14.0, 8.0
    # Replacement-level veterans: a stack of small bonuses (age, position, need, window,
    # cheap deal) added up to +8 on a ~10 base — AHL depth priced like real assets.
    if not is_prospect_val and val_ovr < 74.0:
        ctx_hi = 2.0 if age >= 24 else 4.0
    context_mod = _clamp(sum(v for v in context_parts if v > 0), 0.0, ctx_hi) + _clamp(
        sum(v for v in context_parts if v < 0), ctx_lo, 0.0
    )
    # Deadline / rental market premium is time-limited and self-bounded — keep it
    # outside the clamp so contenders visibly pay up at the deadline.
    market_applied = _clamp(market_mod, -15.0, 15.0)

    components = {
        "talent": base_core,
        "base": base_core,
        "age": round(age_mod, 2),
        "potential": round(potential_mod, 2),
        "prospect_upside": round(prospect_upside, 2),
        "valued_on": "potential" if is_prospect_val else "overall",
        "valuation_ovr": round(val_ovr, 1),
        "production": round(production_mod, 2),
        "contract": round(contract_mod, 2),
        "team_need": round(need_mod, 2),
        "team_window": round(window_mod, 2),
        "market": round(market_mod, 2),
        "rental": round(rental_mod, 2),
        "risk": round(risk_mod, 2),
        "ntc_waive": round(waive_mod, 2),
        "position": round(pos_mod, 2),
        "elc": round(elc_mod, 2),
        "cap_dump": round(cap_dump_mod, 2),
        "injury": round(injury_mod, 2),
        "context_cap": round(context_mod - context_raw, 2),
    }
    total = base_core + context_mod + market_applied
    if injury_pct > 0 and total > 0:
        components["injury_pct"] = round(injury_pct, 3)
        total *= (1.0 - injury_pct)
    # Extra star premium / depth tax on top of the uncapped talent curve.
    # Pipeline assets skip star-premium — ceiling is already discounted in val_ovr.
    if not is_prospect_val:
        if val_ovr < 73.0:
            total -= (73.0 - val_ovr) * 1.5
        elif val_ovr < 77.0:
            total -= (77.0 - val_ovr) * 0.65
        elif val_ovr < 80.0:
            total -= (80.0 - val_ovr) * 0.30
    # Pipeline assets stay below lottery picks and proven NHL stars.
    if is_prospect_val:
        # Elite-ceiling premium: blue-chip prospects are the scarcest currency in the
        # league. Without it a 90+ potential #1 pick priced like a starting goalie.
        if pot >= 82.0:
            total += (pot - 82.0) * 7.0 * (0.75 + 0.5 * _scouting_confidence(player))
        signed = str(getattr(player, "signed_status", "") or "").lower()
        unsigned = signed in ("unsigned", "rights", "rights_only", "") and cap_hit <= 0.05
        # Cap was "never above a #1 pick" (~93); a developing elite prospect is worth more
        # than the pick that bought him, so the cap rises with his ceiling.
        prospect_cap = slot_curve_value(1) + max(0.0, pot - 84.0) * 10.0 - (2.0 if unsigned else 0.0)
        total = min(total, max(prospect_cap, 55.0))

    # --- v15: position scarcity, age curve, goalie discount, contract term -------
    pos_mult = _position_value_mult(pos, val_ovr)
    age_mult = 1.0 if is_prospect_val else _age_value_mult(age, val_ovr, pos)
    if total > 0:
        total *= pos_mult * age_mult
    total += term_surplus * (age_mult if total > 0 else 1.0)
    components["position_mult"] = round(pos_mult, 3)
    components["age_mult"] = round(age_mult, 3)
    components["term_surplus"] = round(term_surplus, 2)

    # --- v14 rebalance: org level, depth role, prospect premium ------------------
    org_loc = _org_location(player, source_team)
    total_pre_role = float(total)
    low_ceiling = pot < PROSPECT_PREMIUM_MIN_POT or pot <= ovr + 3.0
    depth_role = ""
    role_mult = 1.0
    role_burden = 0.0
    if org_loc in ("ahl", "echl") and age >= 23 and not is_prospect_val:
        if low_ceiling:
            # AHL-only veteran: a roster filler, not a trade chip.
            depth_role = "ahl_filler"
            filler_cap = AHL_FILLER_BASE + max(0.0, val_ovr - 70.0) * AHL_FILLER_PER_OVR
            if total > filler_cap:
                role_mult = filler_cap / total if total > 0 else 1.0
                total = filler_cap
        elif total > 0:
            depth_role = "ahl_projectable"
            role_mult = AHL_HIGH_CEILING_MULT
            total *= role_mult
    else:
        role = _depth_role(player, source_team, org_loc)
        if role and val_ovr < DEPTH_ROLE_MAX_VAL_OVR and (low_ceiling or age >= 26):
            depth_role = role
            role_mult = DEPTH_ROLE_MULT[role]
            if total > 0:
                total *= role_mult
            role_burden = _role_contract_burden(cap_hit, years, DEPTH_ROLE_AAV_M[role])
    role_adjust = float(total) - total_pre_role
    total_pre_floor = float(total)
    prospect_floor = prospect_floor_value(age, ovr, pot, _scouting_confidence(player))
    prospect_floor_applied = False
    if prospect_floor > 0:
        if window == "rebuild":
            prospect_floor *= 1.08
        elif window == "contender":
            prospect_floor *= 0.94
        # Real risk (injury, demand crisis, disruptor) still bites the premium.
        prospect_floor += min(0.0, risk_mod + injury_mod)
        if prospect_floor > total:
            total = prospect_floor
            prospect_floor_applied = True
    components["org_level"] = org_loc or "nhl"
    components["depth_role"] = depth_role
    components["role_mult"] = round(role_mult, 3)
    components["role_contract"] = round(role_burden, 2)
    components["prospect_floor"] = round(prospect_floor, 2)
    # Numeric drivers for the Trade Hub value panel.
    components["role_adjust"] = round(role_adjust, 2)
    components["prospect_premium"] = round(float(total) - total_pre_floor, 2) if prospect_floor_applied else 0.0

    # Hockey value floor: a near-minimum deal can be waived/buried for almost
    # nothing, so it should never cost a sweetener to move.
    hockey_floor = 0.0 if cap_hit <= LEAGUE_MINIMUM_AAV_M + 0.25 else -15.0
    # Injury: scale with how much of the remaining season (and term) the player misses.
    # The flat context discount alone moved a 92 OVR star only ~4% for a 60-game injury.
    inj_games = _injury_games_out(player) if is_player_injured(player) else 0
    if inj_games > 0 and total > 0:
        ictx = context or {}
        last_idx = max(40, int(ictx.get("regular_season_last_index", 192) or 192))
        cur = int(ictx.get("calendar_cursor", 0) or 0)
        season_left = max(0.08, min(1.0, (last_idx - cur) / float(last_idx))) if cur <= last_idx else 1.0
        games_left = max(8.0, 82.0 * season_left)
        share_missed = min(1.0, inj_games / games_left)
        term = max(1, int(years or 1))
        weight = 0.6 if term <= 1 else 0.6 / (1.0 + 0.5 * (term - 1))
        inj_discount = share_missed * weight + (0.08 if inj_games > games_left else 0.0)
        inj_discount = max(0.0, min(0.65, inj_discount))
        components["injury_share_discount"] = round(inj_discount, 3)
        total *= 1.0 - inj_discount
    total = max(hockey_floor, float(total))
    # Contract burden sits outside the context clamp so albatross deals go negative.
    # Depth players are also measured against what their ROLE is worth.
    total = max(PLAYER_VALUE_FLOOR, total + min(cap_dump_mod, role_burden))
    tier = player_value_tier(total)

    explain: List[str] = []
    if is_prospect_val and val_ovr >= 84:
        explain.append("Elite prospect ceiling")
    elif not is_prospect_val and ovr >= 85:
        explain.append("Elite NHL talent")
    if age <= 23 and pot > ovr + 5:
        explain.append("Strong upside relative to current rating")
    if prospect_upside >= 5:
        explain.append("Elite prospect upside")
    if term_surplus >= 8:
        explain.append(f"Cost-controlled contract (+{term_surplus:.0f} surplus over {years} yr)")
    if age_mult <= 0.8:
        explain.append(f"Age {age} — decline years discount")
    if pos == "G":
        explain.append("Goalie market discount")
    if cap_dump_mod < 0 and total < 0:
        explain.append("Contract cost outweighs on-ice value")
    elif cap_dump_mod <= -3.0 or contract_mod <= -1.5:
        explain.append("Expensive contract for current production")
    if need_mod >= 8:
        explain.append("Fills a positional need for acquiring team")
    if rental_mod >= 4.0:
        explain.append("Deadline rental premium")
    if waived and waive_mod < 0:
        explain.append("NTC waived — slightly reduced trade value")
    if elc_mod >= 2.0:
        explain.append("Cost-controlled ELC upside")
    if cap_dump_mod <= -6.0:
        explain.append("Negative-value contract — requires sweetener")
    if injury_mod <= -4.0:
        explain.append("Injury discount on trade value")
    if window == "rebuild" and age >= 30:
        explain.append("Older profile — less valuable to rebuilding team")
    if window == "rebuild" and age <= 23:
        explain.append("Youth valued by rebuilding team")
    if prospect_floor_applied:
        explain.insert(0, f"Prospect premium — {int(round(pot))} potential at age {age}")
    if depth_role == "ahl_filler":
        explain.insert(0, "AHL depth — filler value only")
    elif depth_role in DEPTH_ROLE_LABEL:
        explain.insert(0, f"{DEPTH_ROLE_LABEL[depth_role]} — depth value")
    if role_burden < 0 and role_burden <= cap_dump_mod:
        explain.append(f"Cap hit above what a {DEPTH_ROLE_LABEL.get(depth_role, 'depth').lower()} is worth")
        if "Cap dump / negative-value contract" not in contract_flags:
            contract_flags.append("Cap hit exceeds role")
    for flag in risk_flags[:2]:
        if flag not in explain:
            explain.append(flag)

    cap_impact = {
        "incoming_cap_m": round(cap_hit, 3),
        "full_cap_m": round(full_cap_hit, 3),
        "retained_pct": round(retained_frac * 100.0, 1),
        "expected_cap_m": round(expected_cap, 3),
        "years_remaining": years,
        "retained_pct_supported": True,
    }

    return {
        "asset_id": str(getattr(player, "id", "")),
        "type": "player",
        "name": player_display_name(player),
        "total": round(total, 2),
        "trade_value": round(total, 2),
        "value_tier": tier,
        "components": components,
        "breakdown": components,
        "explain": explain,
        "risk_flags": risk_flags,
        "contract_flags": contract_flags,
        "cap_impact": cap_impact,
    }


def evaluate_pick_asset_value(
    pick_row: Dict[str, Any],
    acquiring_team: Any,
    source_team: Any,
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = context or {}
    year = _safe_int(pick_row.get("year"), 0)
    rnd = _safe_int(pick_row.get("round"), 7)
    # Prefer upcoming draft year so "current" capital matches Entry Draft consumption.
    if ctx.get("draft_year") is not None:
        anchor = _safe_int(ctx.get("draft_year"), year)
    elif ctx.get("season_is_calendar") or ctx.get("use_upcoming_draft_year"):
        from app.sim_engine.trades.trade_pick_registry import upcoming_draft_year

        anchor = upcoming_draft_year(_safe_int(ctx.get("season_year"), year))
    else:
        anchor = _safe_int(ctx.get("season_year"), year)

    round_base = {
        1: 58.0,
        2: 28.0,
        3: 16.0,
        4: 10.0,
        5: 7.0,
        6: 5.0,
        7: 3.5,
    }.get(rnd, 3.0)

    years_out = max(0, year - anchor)
    # Proportional (not flat -4/yr): a flat discount erased every late-round pick
    # two drafts out while barely touching a first.
    age_discount = round_base * min(0.45, years_out * 0.12)
    base = max(1.0, round_base - age_discount)

    # Once the board is set, the exact slot is known and replaces the
    # round-average base plus the standings guesswork that estimates it.
    known_slot = _known_pick_slot(pick_row, ctx) if years_out == 0 else None
    team_by_id = ctx.get("team_by_id") or {}
    orig_tid = str(pick_row.get("original_team_id") or "")
    original_team = team_by_id.get(orig_tid) if isinstance(team_by_id, dict) else None
    proj = _projected_finish_risk(original_team, team_by_id=team_by_id if isinstance(team_by_id, dict) else None)
    expected_slot: Optional[float] = None
    if known_slot is not None:
        base = slot_curve_value(known_slot)
        age_discount = 0.0
    else:
        # Price the pick where it is expected to land on the real slot curve. Team strength
        # enters ONCE, through the projected slot, weighted by how settled the standings are.
        # (It used to be applied four times — finish risk, two lottery nudges and a quality
        # penalty — which sank most 1sts to their floor, level with an average 2nd.)
        n_teams = len(team_by_id) if isinstance(team_by_id, dict) and len(team_by_id) >= 2 else 32
        mid = (n_teams + 1) / 2.0
        rank = proj.get("league_rank")
        if rank is None and proj.get("points_pct") is not None:
            rank = _clamp(n_teams - (float(proj["points_pct"]) - 0.35) * 48.0 * (n_teams / 32.0), 1, n_teams)
        gp = _safe_float(getattr(original_team, "gp", getattr(original_team, "games_played", 0)), 0.0) if original_team is not None else 0.0
        if years_out == 0:
            certainty = min(1.0, gp / 60.0)
        elif years_out == 1:
            certainty = 0.2
        else:
            certainty = 0.0
        in_round = mid if rank is None else certainty * float(rank) + (1.0 - certainty) * mid
        expected_slot = (rnd - 1) * n_teams + in_round
        # Modest future discount (v14): 7%/yr, capped at 18% — was 10%/yr to 30%.
        base = slot_curve_value(int(round(expected_slot))) * (1.0 - min(FUTURE_PICK_DISCOUNT_CAP, FUTURE_PICK_DISCOUNT_PER_YEAR * years_out))
        age_discount = 0.0

    window = _team_window(acquiring_team)
    window_mod = 0.0
    if window == "rebuild":
        window_mod = 6.0 if rnd <= 2 else 3.0
    elif window == "contender":
        window_mod = -3.0 if rnd == 1 else -1.5

    market_mod = 0.0
    if ctx.get("deadline_phase", 0.0) > 0.4 and window == "contender" and rnd <= 2:
        market_mod -= 3.0

    # Team strength is already inside the projected slot — no separate finish-risk term.
    original_team_mod = 0.0

    points_pct = proj.get("points_pct")
    lottery_mod = 0.0
    league_rank = proj.get("league_rank")
    if rnd == 1:
        if league_rank is not None:
            n_teams = len(team_by_id) if isinstance(team_by_id, dict) and team_by_id else 32
            if league_rank >= max(1, n_teams - 4):
                lottery_mod += 12.0
            elif league_rank >= max(1, n_teams - 10):
                lottery_mod += 6.0
            elif league_rank <= 5:
                lottery_mod -= 6.0
        if points_pct is not None:
            if points_pct < 0.42:
                lottery_mod += 8.5
            elif points_pct < 0.47:
                lottery_mod += 5.5
            elif points_pct < 0.51:
                lottery_mod += 3.0
            elif points_pct > 0.58:
                lottery_mod -= 5.5
            elif points_pct > 0.54:
                lottery_mod -= 3.0

    future_mod = 0.0
    if years_out >= 1:
        future_mod -= min(5.0, years_out * 2.2)
        orig_window = str(proj.get("window") or "")
        if orig_window in ("rebuild", "declining"):
            future_mod += min(6.0, years_out * 1.4)
        elif orig_window == "contender":
            future_mod -= min(3.0, years_out * 0.9)

    protection = pick_row.get("protection")
    conditions = pick_row.get("conditions")
    prot_discount = _protection_discount(protection, rnd)
    risk_mod = -prot_discount if prot_discount else 0.0
    if conditions:
        risk_mod -= 3.0

    injury_factor = 0.0
    if original_team is not None:
        roster = list(getattr(original_team, "roster", None) or [])
        injured_core = 0
        for p in roster[:12]:
            if is_player_injured(p):
                injured_core += 1
        if injured_core >= 3:
            injury_factor += min(3.0, injured_core * 0.45)
        elif injured_core >= 1 and rnd == 1:
            injury_factor += 0.8

    components = {
        "base": round(base, 2),
        "original_team_projection": round(original_team_mod, 2),
        "lottery": round(lottery_mod, 2),
        "future_risk": round(future_mod, 2),
        "team_window": round(window_mod, 2),
        "market": round(market_mod, 2),
        "risk": round(risk_mod, 2),
        "injury": round(injury_factor, 2),
    }
    # Team/market modifiers are sized for a first-round pick. Scale them to the round
    # so a strong original club trims a 3rd a little instead of zeroing it.
    if rnd >= 2 and known_slot is None:
        scale = max(0.2, min(1.0, round_base / 28.0))
        for key in ("original_team_projection", "future_risk", "team_window", "market"):
            components[key] = round(components[key] * scale, 2)
    total = max(0.5, float(sum(components.values())), 0.45 * float(components["base"]))

    # Crown-jewel spectrum: lottery/rebuild clubs' 1sts vs contender late 1sts.
    if False and rnd == 1 and known_slot is None and original_team is not None:  # folded into projected slot
        risk = float(proj.get("projected_risk_score") or 0.0)
        league_rank = proj.get("league_rank")
        n_teams = len(team_by_id) if isinstance(team_by_id, dict) and team_by_id else 32
        quality_mod = 0.0
        if risk >= 14.0 or (league_rank is not None and league_rank >= max(1, n_teams - 5)):
            quality_mod += 22.0 + min(18.0, max(0.0, risk - 10.0) * 1.1)
        elif risk >= 8.0 or (league_rank is not None and league_rank >= max(1, n_teams - 10)):
            quality_mod += 10.0 + min(8.0, max(0.0, risk - 6.0) * 0.9)
        elif risk <= -2.0 or (league_rank is not None and league_rank <= 8):
            quality_mod -= 14.0 + (min(6.0, abs(min(0.0, risk))) if risk < 0 else 0.0)
        elif risk <= 2.0 or (league_rank is not None and league_rank <= 14):
            quality_mod -= 6.0
        if str(proj.get("window") or "") == "rebuild" and quality_mod > 0:
            quality_mod += 4.0
        elif str(proj.get("window") or "") == "contender" and quality_mod < 0:
            quality_mod -= 4.0
        components["original_team_quality"] = round(quality_mod, 2)
        total = max(0.5, total + quality_mod)
    elif False and rnd == 2 and known_slot is None and original_team is not None:  # folded into projected slot
        risk = float(proj.get("projected_risk_score") or 0.0)
        quality_mod = _clamp(risk * 0.35, -4.0, 8.0)
        components["original_team_quality"] = round(quality_mod, 2)
        total = max(0.5, total + quality_mod)

    explain = [f"Round {rnd} pick in {year}"]
    if original_team is not None:
        explain.append(f"Original team risk score {proj.get('projected_risk_score')}")
    if window == "rebuild":
        explain.append("High value to rebuilding team")
    if rnd == 1 and years_out == 0:
        explain.append("Current-year first-round capital")
    if protection:
        explain.append("Protection lowers expected conveyance value")
    if conditions:
        explain.append("Conditional structure lowers certainty")

    projected_range = _pick_projected_range(proj, rnd)
    projected_slot = _pick_projected_slot(proj, rnd)
    if known_slot is not None:
        explain.insert(0, f"Pick #{known_slot} overall (slot known)")
        projected_slot = known_slot
        projected_range = f"#{known_slot}"
    pick_context = _pick_value_context(
        proj,
        years_out=years_out,
        protection=protection,
        original_team=original_team,
    )
    tier = pick_value_tier(total)

    return {
        "asset_id": str(pick_row.get("pick_id", "")),
        "type": "pick",
        "name": pick_row.get("display") or pick_row.get("pick_id"),
        "total": round(total, 2),
        "trade_value": round(total, 2),
        "value_tier": tier,
        "projected_slot": projected_slot,
        "projected_range": projected_range,
        "pick_value_context": pick_context,
        "components": components,
        "value_debug": {
            "pick_id": str(pick_row.get("pick_id", "")),
            "year": int(year or 0),
            "round": int(rnd or 0),
            "original_team_id": orig_tid,
            "current_owner_team_id": str(pick_row.get("current_owner_team_id") or ""),
            "known_overall_slot": known_slot,
            "base_round_value": round(base, 2),
            "year_discount": round(age_discount, 2),
            "projected_finish_risk": proj.get("projected_risk_score"),
            "projected_finish_rank": None,
            "lottery_probability": None,
            "points_pct": proj.get("points_pct"),
            "original_team_points_pct": proj.get("points_pct"),
            "original_team_strength_score": proj.get("core_strength"),
            "core_strength": proj.get("core_strength"),
            "core_roster_strength": proj.get("core_strength"),
            "goalie_factor": 0.0,
            "prospect_pool_factor": 0.0,
            "age_curve_factor": 0.0,
            "team_window_factor": 0.0,
            "injury_factor": round(injury_factor, 2),
            "window": proj.get("window"),
            "lottery_factor": round(lottery_mod, 2),
            "future_factor": round(future_mod, 2),
            "scarcity_premium": round(lottery_mod + max(0.0, original_team_mod), 2) if rnd == 1 else 0.0,
            "market_premium": round(market_mod, 2),
            "protection_discount": -prot_discount if prot_discount else 0.0,
            "projected_range": projected_range,
            "projected_slot": projected_slot,
            "pick_value_context": pick_context,
            "condition_discount": -3.0 if conditions else 0.0,
            "market_factor": round(market_mod, 2),
            "acquiring_team_window_factor": round(window_mod, 2),
            "final_value": round(total, 2),
        },
        "explain": explain,
    }


def evaluate_asset_value(
    asset: Union[PlayerTradeAsset, DraftPickTradeAsset],
    source_team: Any,
    acquiring_team: Any,
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(asset, PlayerTradeAsset):
        src = source_team
        from app.sim_engine.trades.trade_asset import find_player_in_organization

        player, _loc, _idx = find_player_in_organization(src, asset.player_id)
        if player is None:
            player, _ = find_player_on_team_roster(src, asset.player_id)
        if player is None:
            return {
                "asset_id": asset.player_id,
                "type": "player",
                "name": asset.player_name or asset.player_id,
                "total": 0.0,
                "components": {},
                "explain": ["Player not found on source organization"],
            }
        return evaluate_player_asset_value(
            player,
            source_team,
            acquiring_team,
            league,
            context={
                **dict(context or {}),
                "ntc_waived": bool(getattr(asset, "ntc_waived", False)),
                "ntc_value_penalty_pct": 0.08 if bool(getattr(asset, "ntc_waived", False)) else 0.0,
            },
            retained_pct=asset.retained_pct,
        )

    row = get_pick_by_id(league, asset.pick_id) or {
        "pick_id": asset.pick_id,
        "year": asset.year,
        "round": asset.round,
        "original_team_id": asset.original_team_id,
    }
    return evaluate_pick_asset_value(row, acquiring_team, source_team, league, context=context)


def _pos_group(pos: str) -> str:
    p = str(pos or "").upper()
    if p.startswith("G"):
        return "G"
    if p in ("D", "LD", "RD"):
        return "D"
    if p == "C":
        return "C"
    return "W"


def _seller_loss_premium(team: Any, player_id: str) -> float:
    """Extra weight a club puts on losing one of its own players (0..0.22).

    Market value is nearly identical from every club's chair, which made trades zero-sum:
    the CPU only ever took deals that were even or better for itself, so the user could
    never come out ahead and a club never felt the hole a departure leaves. A club now
    weighs (a) where the player sits on its own depth chart and (b) how big the drop is
    to the next man at his position. Contenders feel it more, rebuilders less.
    """
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    me = next((p for p in roster if str(getattr(p, "id", "")) == str(player_id)), None)
    if me is None:
        return 0.0
    grp = _pos_group(_player_pos(me))
    peers = sorted((_player_ovr(p) for p in roster if _pos_group(_player_pos(p)) == grp), reverse=True)
    my_ovr = _player_ovr(me)
    rank = sum(1 for v in peers if v > my_ovr + 1e-6)
    slots = {"G": 1, "D": 2, "C": 2, "W": 4}.get(grp, 2)
    prem = 0.0
    if rank < slots:
        prem += 0.08 if grp != "G" else 0.12
    nxt = next((v for v in peers if v < my_ovr - 1e-6), None)
    if nxt is not None and rank < slots + 1:
        prem += min(0.08, max(0.0, (my_ovr - nxt - 2.0) * 0.012))
    window = _team_window(team)
    mult = 1.35 if window == "contender" else 0.45 if window == "rebuild" else 1.0
    return max(0.0, min(0.22, prem * mult))


def _buyer_fit_premium(team: Any, player: Any) -> float:
    """Extra weight a club puts on a player who would start for it at a thin position (0..0.16).

    Mirrors _seller_loss_premium: a seller moving depth and a buyer filling a hole can both
    come out ahead, which is how real hockey trades get made.
    """
    if team is None or player is None:
        return 0.0
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    grp = _pos_group(_player_pos(player))
    peers = sorted((_player_ovr(p) for p in roster if _pos_group(_player_pos(p)) == grp), reverse=True)
    slots = {"G": 1, "D": 4, "C": 2, "W": 4}.get(grp, 2)
    starters = peers[:slots]
    ovr = _player_ovr(player)
    if len(starters) < slots:
        prem = 0.10
    else:
        worst = starters[-1]
        if ovr <= worst:
            return 0.0
        prem = 0.04 + min(0.08, (ovr - worst) * 0.012)
    window = _team_window(team)
    mult = 1.3 if window == "contender" else 0.5 if window == "rebuild" else 1.0
    return max(0.0, min(0.16, prem * mult))


def evaluate_package_value(
    package: TradePackage,
    team_id: str,
    league: Any,
    team_by_id: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tid = str(team_id)
    incoming_vals: List[Dict[str, Any]] = []
    outgoing_vals: List[Dict[str, Any]] = []

    for asset in package.incoming_by_team.get(tid, []):
        src = team_by_id.get(asset.source_team_id if hasattr(asset, "source_team_id") else "")
        acq = team_by_id.get(tid)
        if src is None or acq is None:
            continue
        row = dict(evaluate_asset_value(asset, src, acq, league, context=context))
        if getattr(asset, "type", "") == "player":
            try:
                from app.sim_engine.trades.trade_asset import find_player_in_organization

                pl, _loc, _i = find_player_in_organization(src, str(getattr(asset, "player_id", "") or ""))
                prem = _buyer_fit_premium(acq, pl) if pl is not None else 0.0
            except Exception:
                prem = 0.0
            m_total = float(row.get("total", 0.0) or 0.0)
            if prem > 0 and m_total > 0:
                row["market_total"] = m_total
                row["buyer_fit_premium"] = round(prem, 3)
                row["total"] = round(m_total * (1.0 + prem), 2)
        incoming_vals.append(row)

    # Outgoing: what this club gives up. Blend the market's view (value to the receiver)
    # with the club's own view (its fit, its depth chart, its needs) so a club feels the
    # loss of its own top centre or only good goalie, and a deal can help both sides.
    out_vals: List[Dict[str, Any]] = []
    src_team = team_by_id.get(tid)
    for asset in package.outgoing_by_team.get(tid, []):
        acq = team_by_id.get(asset.acquiring_team_id if hasattr(asset, "acquiring_team_id") else "")
        if src_team is None or acq is None:
            continue
        market = evaluate_asset_value(asset, src_team, acq, league, context=context)
        row = dict(market)
        if getattr(asset, "type", "") == "player":
            try:
                prem = _seller_loss_premium(src_team, str(getattr(asset, "player_id", "") or ""))
            except Exception:
                prem = 0.0
            m_total = float(market.get("total", 0.0) or 0.0)
            if prem > 0 and m_total > 0:
                row["market_total"] = m_total
                row["seller_loss_premium"] = round(prem, 3)
                row["total"] = round(m_total * (1.0 + prem), 2)
        out_vals.append(row)
    outgoing_vals = out_vals

    raw_out = sum(v.get("total", 0.0) for v in out_vals)
    raw_in = sum(v.get("total", 0.0) for v in incoming_vals)
    # Quality over quantity: both sides are summed with diminishing weights, so four
    # middling pieces no longer add up to one star (roster spots and contracts are finite).
    out_total = effective_package_total([v.get("total", 0.0) for v in out_vals])
    in_total = effective_package_total([v.get("total", 0.0) for v in incoming_vals])
    net = in_total - out_total

    return {
        "incoming": incoming_vals,
        "outgoing": out_vals,
        "incoming_total": round(in_total, 2),
        "outgoing_total": round(out_total, 2),
        "incoming_raw_total": round(raw_in, 2),
        "outgoing_raw_total": round(raw_out, 2),
        "net": round(net, 2),
    }


CONSOLIDATION_WEIGHTS = (1.0, 0.80, 0.64, 0.52, 0.42)
CONSOLIDATION_TAIL = 0.35


def effective_package_total(values: List[float]) -> float:
    """Package worth with diminishing returns on extra pieces (best asset counts fully,
    the 2nd at 80%, 3rd 64%, ...). Negative contracts always count in full."""
    pos = sorted((float(v) for v in values if float(v) > 0), reverse=True)
    neg = sum(float(v) for v in values if float(v) < 0)
    total = 0.0
    for i, v in enumerate(pos):
        w = CONSOLIDATION_WEIGHTS[i] if i < len(CONSOLIDATION_WEIGHTS) else CONSOLIDATION_TAIL
        total += v * w
    return round(total + neg, 2)


def pick_value_hint(row: Dict[str, Any], league: Any, team: Any, context: Optional[Dict[str, Any]] = None) -> float:
    ctx = context or {}
    orig_tid = str(row.get("original_team_id") or getattr(team, "team_id", getattr(team, "id", "")) or "")
    team_by_id = ctx.get("team_by_id") or {}
    orig_team = team_by_id.get(orig_tid) if isinstance(team_by_id, dict) else None
    eval_team = orig_team or team
    val = evaluate_pick_asset_value(row, eval_team, eval_team, league, context=ctx)
    return float(val.get("total", 0.0))
