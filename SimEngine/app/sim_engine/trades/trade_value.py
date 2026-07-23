"""
Authoritative backend trade valuation (0-100 scale per team perspective).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from app.sim_engine.economy.team_needs import TeamNeeds, is_player_injured
from app.sim_engine.economy.cap_engine import player_cap_hit_millions
from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    TradePackage,
    find_player_on_team_roster,
    player_display_name,
)
from app.sim_engine.trades.trade_pick_registry import get_pick_by_id

# Bump when the talent curve / depth-star spread changes so Trade Hub caches rebuild
# without requiring a new franchise save.
TRADE_VALUE_FORMULA_VERSION = 3


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return lo if x < lo else hi if x > hi else x


def player_value_tier(total: float) -> str:
    v = float(total)
    if v >= 90:
        return "Franchise"
    if v >= 75:
        return "Elite"
    if v >= 55:
        return "Top Asset"
    if v >= 35:
        return "Useful"
    if v >= 18:
        return "Depth"
    return "Negative Value"


def pick_value_tier(total: float) -> str:
    """Hidden pick tiers aligned with Trade Hub bar scale (0-100)."""
    v = float(total)
    if v >= 90:
        return "FRANCHISE"
    if v >= 75:
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


def _pick_projected_slot(proj: Dict[str, Any], rnd: int) -> Optional[int]:
    league_rank = proj.get("league_rank")
    if league_rank is not None and rnd == 1:
        return int(league_rank)
    points_pct = proj.get("points_pct")
    if points_pct is not None and rnd == 1:
        return int(_clamp(32 - (points_pct - 0.35) * 48.0, 1, 32))
    window = str(proj.get("window") or "").lower()
    if rnd != 1:
        return None
    if window == "rebuild":
        return 28
    if window == "contender":
        return 6
    if window == "declining":
        return 18
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
            pass
    return 0.5


def _prospect_upside_score(player: Any, ovr: float, age: int, pot: float) -> float:
    if age >= 23:
        return 0.0
    upside = max(0.0, pot - ovr)
    tier = _prospect_draft_tier(player)
    confidence = _scouting_confidence(player)
    base = upside * 0.22
    if age <= 20:
        base += 2.5
    elif age <= 21:
        base += 1.5
    elif age <= 22:
        base += 0.8
    if tier >= 0.85:
        base += 6.0
    elif tier >= 0.65:
        base += 4.0
    elif tier >= 0.45:
        base += 2.0
    if pot >= 88 and ovr < 76:
        base += 4.0
    elif pot >= 84 and ovr < 72:
        base += 2.5
    base *= 0.75 + 0.5 * confidence
    return _clamp(base, 0.0, 14.0)


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


def _team_points_pct(team: Any) -> Optional[float]:
    if team is None:
        return None
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
    ranked.sort(key=lambda x: x[0])
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


def _talent_base(ovr: float) -> float:
    """
    Aggressive ability curve — depth is cheap roster chips; stars demand real return.

    Approx anchors (before contract/need nudges):
      70 4th-line → ~10 · 75 bottom-6 → ~20 · 78 middle-6 → ~31
      82 top-6 → ~48 · 85 star → ~66 · 88 superstar → ~86 · 90+ franchise → ~96
    """
    o = float(ovr or 0.0)
    if o <= 0:
        return 3.0
    if o < 70.0:
        # 62 → ~3 · 68 → ~9
        anchor = 3.0 + max(0.0, o - 60.0) * 1.0
    elif o < 76.0:
        # 70 → ~10 · 75 → ~20
        anchor = 10.0 + (o - 70.0) * 2.0
    elif o < 81.0:
        # 76 → ~24 · 80 → ~40
        anchor = 20.0 + (o - 75.0) * 4.0
    elif o < 85.0:
        # 81 → ~46 · 84 → ~60
        anchor = 40.0 + (o - 80.0) * 5.0
    elif o < 88.0:
        # 85 → ~68 · 87 → ~80
        anchor = 60.0 + (o - 84.0) * 6.5
    elif o < 91.0:
        # 88 → ~88 · 90 → ~96
        anchor = 80.0 + (o - 87.0) * 5.5
    else:
        # 91+ → ~97–99
        anchor = 96.0 + min(3.0, (o - 91.0) * 1.0)
    return _clamp(anchor, 3.0, 99.0)


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


def _contract_years(player: Any) -> int:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            v = _safe_int(getattr(obj, key, 0), 0)
            if v > 0:
                return v
    return 0


def _expiry_status(player: Any) -> str:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("expiry_status", "ufa_rfa_status", "rights_status", "rights"):
            val = str(getattr(obj, key, "") or "").strip().upper()
            if val in ("UFA", "RFA", "ELC"):
                return val
        ctype = str(getattr(obj, "contract_type", getattr(obj, "type", "")) or "").strip().upper()
        if ctype == "ELC":
            return "ELC"
    if _is_elc_contract(player):
        return "ELC"
    age = _player_age(player)
    return "RFA" if age < 27 else "UFA"


def _contract_type_label(player: Any) -> str:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        ctype = str(getattr(obj, "contract_type", getattr(obj, "type", "")) or "").strip().upper()
        if ctype:
            return ctype
    return ""


def _is_elc_contract(player: Any) -> bool:
    if _contract_type_label(player) == "ELC":
        return True
    cap = player_cap_hit_millions(player)
    return 0 < cap <= 1.05 and _player_age(player) <= 25


def _injury_games_out(player: Any) -> int:
    for key in ("injury_games_remaining", "games_out", "games_remaining"):
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
) -> float:
    if expiry != "ELC" and not _is_elc_contract(player):
        return 0.0
    mod = 0.0
    if window == "rebuild":
        if age <= 22:
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
    expected = _clamp(0.75 + max(0.0, ovr - 72.0) * 0.22, 0.85, 14.0)
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
    c = getattr(player, "contract", None)
    bad_type = getattr(c, "bad_contract_type", None) if c else None
    if bad_type:
        score = max(score, 0.35)
    try:
        tagged = float(getattr(player, "bad_contract_score", 0) or getattr(c, "bad_contract_score", 0) or 0)
        if tagged > 0:
            score = max(score, tagged)
    except Exception:
        pass
    return min(2.0, score)


def _cap_dump_value_mod(
    player: Any,
    *,
    ovr: float,
    cap_hit: float,
    years: int,
    age: int,
    expected_cap: float,
    source_window: str,
    acquiring_window: str,
    cap_pressure: str,
) -> float:
    bad = _bad_contract_score(player, ovr, cap_hit, years, age)
    overpay = cap_hit - expected_cap
    if bad < 0.22 and overpay <= 1.25:
        return 0.0
    mod = -min(14.0, overpay * 2.4 + bad * 6.5)
    if years >= 4:
        mod -= min(3.0, (years - 3) * 0.9)
    if source_window in ("rebuild", "declining") and bad >= 0.35:
        mod -= 1.5
    if acquiring_window == "rebuild" and cap_pressure not in ("cap_hell", "critical") and bad >= 0.4:
        mod += min(4.0, bad * 5.0)
    if acquiring_window == "contender" and bad >= 0.35:
        mod -= 2.0
    return mod


def _production_score(player: Any) -> float:
    st = getattr(player, "season_stats", None) or {}
    if isinstance(st, dict):
        gp = max(1, _safe_int(st.get("gp"), 1))
        pts = _safe_float(st.get("pts"), _safe_float(st.get("g"), 0) + _safe_float(st.get("a"), 0))
        ppg = pts / gp
        if _player_pos(player) == "G":
            sv = _safe_float(st.get("sv_pct"), 0.905)
            return _clamp((sv - 0.88) * 120.0, 0.0, 18.0)
        return _clamp(ppg * 14.0, 0.0, 16.0)
    return 0.0


def _clause_penalty(player: Any) -> float:
    c = getattr(player, "contract", None)
    clauses = getattr(c, "clauses", None) if c else None
    nmc = bool(getattr(clauses, "noMoveClause", False) if clauses else getattr(c, "no_move_clause", False) if c else False)
    ntc = bool(getattr(clauses, "noTradeClause", False) if clauses else getattr(c, "no_trade_clause", False) if c else False)
    mntc = _safe_int(getattr(clauses, "modifiedNoTradeTeams", 0) if clauses else getattr(c, "modified_no_trade_teams", 0) if c else 0)
    if nmc:
        return 6.0
    if ntc:
        return 4.0
    if mntc > 0:
        return 2.5
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


def evaluate_player_asset_value(
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
    cap_hit = player_cap_hit_millions(player)
    years = _contract_years(player)
    expiry = _expiry_status(player)
    pot = _player_potential_ovr(player, ovr)
    talent_fit = min(1.0, max(0.35, ovr / 82.0))

    base_core = round(_talent_base(ovr), 2)

    age_mod = 0.0
    if age <= 22:
        age_mod = 3.0 if ovr >= 76 else 1.5 if ovr >= 70 else 0.5
    elif age <= 26:
        age_mod = 2.0 if ovr >= 74 else 0.5
    elif age <= 30:
        age_mod = 1.0
    elif age <= 33:
        age_mod = -2.0
    else:
        age_mod = -5.0 - (age - 33) * 0.8

    upside = max(0.0, pot - ovr)
    prospect_upside = _prospect_upside_score(player, ovr, age, pot)
    if age <= 25:
        potential_mod = _clamp(upside * 0.14, 0.0, 5.0) + prospect_upside
        if ovr < 76:
            potential_mod = min(potential_mod, 3.0 + prospect_upside)
    else:
        potential_mod = _clamp(upside * 0.06, 0.0, 2.0)

    production_mod = min(_production_score(player), 8.0) * talent_fit

    pos_mod = 0.0
    if pos == "C":
        pos_mod = 1.5
    elif pos == "D":
        pos_mod = 1.2
    elif pos == "G":
        pos_mod = 1.0

    expected_cap = _clamp(0.75 + max(0.0, ovr - 72.0) * 0.22, 0.85, 14.0)
    contract_mod = _clamp((expected_cap - cap_hit) * 1.4, -8.0, 6.0)
    if years <= 1 and expiry == "UFA":
        contract_mod -= 3.0
    elif years >= 4 and cap_hit < expected_cap:
        contract_mod += 2.0
    elif years >= 5 and age >= 30 and cap_hit > expected_cap + 1.5:
        contract_mod -= 4.0
    if age >= 32 and cap_hit > expected_cap + 2.0:
        contract_mod -= 3.0

    needs = _NEEDS_MODEL.evaluate(acquiring_team, context=ctx)
    # Needs can nudge price but must not flatten OVR gaps (roster-spot dumps vs stars).
    need_scale = 3.2 if ovr >= 84 else 4.2 if ovr >= 78 else 5.0
    need_mod = 0.0
    if pos in ("C", "W"):
        need_mod = max(needs.get("top_line_forward", 0.0), needs.get("depth_forward", 0.0)) * need_scale * talent_fit
    elif pos == "D":
        need_mod = needs.get("top_4_defense", 0.0) * need_scale * talent_fit
    elif pos == "G":
        need_mod = needs.get("goalie", 0.0) * (need_scale + 0.8) * talent_fit
    need_mod = _clamp(need_mod, 0.0, 7.0 if ovr < 82 else 4.5)

    window = _team_window(acquiring_team)
    source_window = _team_window(source_team)
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
    )

    cap_dump_mod = _cap_dump_value_mod(
        player,
        ovr=ovr,
        cap_hit=cap_hit,
        years=years,
        age=age,
        expected_cap=expected_cap,
        source_window=source_window,
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
        if age <= 23 and ovr >= 74:
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
        if 24 <= age <= 32 and ovr >= 80:
            window_mod = 3.5
        elif age <= 22 and ovr < 78:
            window_mod = -2.5
        elif age >= 33 and cap_hit > expected_cap:
            window_mod -= 4.0

    if cap_pressure in ("cap_hell", "critical") and cap_hit > expected_cap + 1.0:
        contract_mod -= 3.0

    market_mod = rental_mod
    if ovr >= 88:
        market_mod += 2.0

    risk_mod = -_clause_penalty(player) * 0.75
    waived, waive_pct = _ntc_waived_for_player(player, ctx)
    waive_mod = 0.0
    if waived and waive_pct > 0:
        # Slight post-waive discount: player is movable but still burned leverage.
        waive_mod = -max(2.5, min(9.0, base_core * float(waive_pct)))
        risk_mod += waive_mod
    mult = _safe_float(getattr(player, "_systemic_trade_value_mult", 1.0), 1.0)
    if mult != 1.0:
        risk_mod += (mult - 1.0) * 5.0

    risk_flags: List[str] = []
    contract_flags: List[str] = []
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

    if retained_pct > 0:
        contract_mod += min(5.0, retained_pct * 0.08)

    context_raw = (
        age_mod
        + potential_mod
        + production_mod
        + contract_mod
        + need_mod
        + window_mod
        + market_mod
        + risk_mod
        + pos_mod
        + elc_mod
        + cap_dump_mod
        + injury_mod
    )
    # Context can nudge but must not erase star vs depth gaps.
    if base_core >= 75.0:
        context_mod = _clamp(context_raw, -10.0, 6.0)
    elif base_core >= 45.0:
        context_mod = _clamp(context_raw, -12.0, 8.0)
    else:
        context_mod = _clamp(context_raw, -14.0, 8.0)

    components = {
        "talent": base_core,
        "base": base_core,
        "age": round(age_mod, 2),
        "potential": round(potential_mod, 2),
        "prospect_upside": round(prospect_upside, 2),
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
    total = base_core + context_mod
    # Hard star premium / depth tax — 4th-liners stay cheap; superstars are scarce.
    if ovr >= 90.0:
        total += 4.0 + (ovr - 90.0) * 1.5
    elif ovr >= 87.0:
        total += 5.0 + (ovr - 87.0) * 1.5
    elif ovr >= 84.0:
        total += 3.5
    elif ovr < 73.0:
        total -= (73.0 - ovr) * 1.6
    elif ovr < 77.0:
        total -= (77.0 - ovr) * 0.9
    elif ovr < 80.0:
        total -= (80.0 - ovr) * 0.4
    total = _clamp(total, 0.0, 100.0)
    tier = player_value_tier(total)

    explain: List[str] = []
    if ovr >= 85:
        explain.append("Elite NHL talent")
    if age <= 23 and pot > ovr + 5:
        explain.append("Strong upside relative to current rating")
    if prospect_upside >= 5:
        explain.append("Elite prospect upside")
    if contract_mod >= 4:
        explain.append("Favorable contract relative to performance")
    if contract_mod <= -6:
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
    for flag in risk_flags[:2]:
        if flag not in explain:
            explain.append(flag)

    cap_impact = {
        "incoming_cap_m": round(cap_hit, 3),
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
    age_discount = years_out * 4.0
    base = max(2.0, round_base - age_discount)

    window = _team_window(acquiring_team)
    window_mod = 0.0
    if window == "rebuild":
        window_mod = 6.0 if rnd <= 2 else 3.0
    elif window == "contender":
        window_mod = -3.0 if rnd == 1 else -1.5

    market_mod = 0.0
    if ctx.get("deadline_phase", 0.0) > 0.4 and window == "contender" and rnd <= 2:
        market_mod -= 3.0

    team_by_id = ctx.get("team_by_id") or {}
    orig_tid = str(pick_row.get("original_team_id") or "")
    original_team = team_by_id.get(orig_tid) if isinstance(team_by_id, dict) else None
    proj = _projected_finish_risk(original_team, team_by_id=team_by_id if isinstance(team_by_id, dict) else None)
    original_team_mod = proj["projected_risk_score"] * (1.25 if rnd == 1 else 0.65)

    points_pct = proj.get("points_pct")
    lottery_mod = 0.0
    league_rank = proj.get("league_rank")
    if rnd == 1:
        if league_rank is not None:
            n_teams = len(team_by_id) if isinstance(team_by_id, dict) and team_by_id else 32
            if league_rank >= max(1, n_teams - 4):
                lottery_mod += 14.0
            elif league_rank >= max(1, n_teams - 10):
                lottery_mod += 7.0
            elif league_rank <= 5:
                lottery_mod -= 7.0
        if points_pct is not None:
            if points_pct < 0.42:
                lottery_mod += 10.0
            elif points_pct < 0.47:
                lottery_mod += 6.5
            elif points_pct < 0.51:
                lottery_mod += 3.5
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
    total = _clamp(sum(components.values()), 0.5, 100.0)

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
        player, _ = find_player_on_team_roster(src, asset.player_id)
        if player is None:
            return {
                "asset_id": asset.player_id,
                "type": "player",
                "name": asset.player_name or asset.player_id,
                "total": 0.0,
                "components": {},
                "explain": ["Player not found on source roster"],
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
        incoming_vals.append(evaluate_asset_value(asset, src, acq, league, context=context))

    for asset in package.outgoing_by_team.get(tid, []):
        src = team_by_id.get(tid)
        acq = team_by_id.get(asset.acquiring_team_id if hasattr(asset, "acquiring_team_id") else "")
        if src is None or acq is None:
            continue
        outgoing_vals.append(evaluate_asset_value(asset, src, acq, league, context=context))

    in_total = sum(v.get("total", 0.0) for v in incoming_vals)
    out_total = sum(
        evaluate_asset_value(a, team_by_id.get(tid), team_by_id.get(a.acquiring_team_id), league, context=context).get("total", 0.0)
        for a in package.outgoing_by_team.get(tid, [])
        if team_by_id.get(tid) and team_by_id.get(a.acquiring_team_id)
    )

    # Outgoing value to this team = what they give up (value to receiving teams averaged)
    out_vals: List[Dict[str, Any]] = []
    for asset in package.outgoing_by_team.get(tid, []):
        src = team_by_id.get(tid)
        acq = team_by_id.get(asset.acquiring_team_id)
        if src and acq:
            out_vals.append(evaluate_asset_value(asset, src, acq, league, context=context))

    out_total = sum(v.get("total", 0.0) for v in out_vals)
    in_total = sum(v.get("total", 0.0) for v in incoming_vals)
    net = in_total - out_total

    return {
        "incoming": incoming_vals,
        "outgoing": out_vals,
        "incoming_total": round(in_total, 2),
        "outgoing_total": round(out_total, 2),
        "net": round(net, 2),
    }


def pick_value_hint(row: Dict[str, Any], league: Any, team: Any, context: Optional[Dict[str, Any]] = None) -> float:
    ctx = context or {}
    orig_tid = str(row.get("original_team_id") or getattr(team, "team_id", getattr(team, "id", "")) or "")
    team_by_id = ctx.get("team_by_id") or {}
    orig_team = team_by_id.get(orig_tid) if isinstance(team_by_id, dict) else None
    eval_team = orig_team or team
    val = evaluate_pick_asset_value(row, eval_team, eval_team, league, context=ctx)
    return float(val.get("total", 0.0))
