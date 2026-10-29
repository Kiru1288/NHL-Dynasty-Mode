"""Usage-driven development, applied once at season end.

Young NHL players grow from NHL ice time (any amount helps; more helps more), and AHL
players grow from AHL ice time and production (services/ahl_league ledger). Growth is a
share of the gap to potential, scaled by age, and never passes potential.
"""
from __future__ import annotations

from typing import Any, Dict, List
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

AGE_FACTOR = {17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.8, 23: 0.6, 24: 0.45, 25: 0.3}
MAX_GAIN = 6.0


def _ovr99(p: Any) -> float:
    from services.lineup_integrity import player_ovr

    return float(player_ovr(p))


def _pot99(p: Any, ovr: float) -> float:
    try:
        from app.sim_engine.trades.trade_value import _player_potential_ovr

        return float(_player_potential_ovr(p, ovr))
    except Exception:
        return ovr


def _age(p: Any) -> int:
    try:
        return int(getattr(getattr(p, "identity", None), "age", 99) or 99)
    except Exception:
        return 99


def _raise_ovr(p: Any, gain: float) -> float:
    from app.sim_engine.entities.player import persist_recomputed_ovr
    from services.real_nhl_roster_importer import align_attribute_ovr_to_target

    before = _ovr99(p)
    target = min(0.99, (before + gain) / 99.0)
    align_attribute_ovr_to_target(p, target, rounds=30)
    try:
        persist_recomputed_ovr(p)
        p._invalidate_ovr_memo()
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return _ovr99(p) - before


def nhl_gain(age: int, ovr: float, pot: float, gp: int, toi_pg: float) -> float:
    af = AGE_FACTOR.get(age, 0.0)
    gap = max(0.0, pot - ovr)
    if af <= 0 or gap <= 0.3 or gp <= 0:
        return 0.0
    usage = min(1.0, toi_pg / 18.0)
    games = 0.35 + 0.65 * min(1.0, gp / 60.0)
    g = gap * 0.17 * af * (0.55 + 0.45 * usage) * games
    if gp >= 20 and age <= 22:
        g = max(g, 1.0)  # any real NHL run is worth at least a point for a kid
    return min(g, gap, MAX_GAIN)


def ahl_gain(age: int, ovr: float, pot: float, line: Dict[str, Any]) -> float:
    af = AGE_FACTOR.get(age, 0.0)
    gap = max(0.0, pot - ovr)
    gp = int(line.get("gp") or 0)
    if af <= 0 or gap <= 0.3 or gp <= 0:
        return 0.0
    usage = min(1.0, float(line.get("toi_per_gp_min") or 0.0) / 18.0)
    if str(line.get("pos") or "") == "G":
        sv = line.get("save_pct") or 0.900
        prod = max(0.0, min(1.6, 1.0 + (float(sv) - 0.905) * 40.0))
    else:
        exp_ppg = 0.30 if str(line.get("pos") or "") == "D" else 0.55
        prod = max(0.0, min(1.6, float(line.get("ppg") or 0.0) / exp_ppg))
    games = 0.35 + 0.65 * min(1.0, gp / 50.0)
    g = gap * 0.20 * af * (0.40 + 0.35 * usage + 0.35 * prod) * games
    return min(g, gap, MAX_GAIN)


def apply_season_ice_time_development(session: Any) -> Dict[str, Any]:
    """Retired as a separate growth source: it added up to +6 OVR on top of the in-season
    pulses and the year-end budget (growth stacked ~1.5x a year). Ice time now feeds the
    single annual budget through ``toi_quality`` (stamped by _dev_stamp_season_production)."""
    if not getattr(session, "_legacy_ice_time_growth", False):
        return {"skipped": True, "reason": "folded_into_development_budget"}
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    if int(getattr(session, "_ice_dev_applied_year", 0) or 0) == sy:
        return {"skipped": True}
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"skipped": True}
    pss = getattr(session, "player_season_stats", None) or {}
    try:
        from services.ahl_league import get_player_ahl_season_line
    except Exception:
        get_player_ahl_season_line = None  # type: ignore
    from services.lineup_integrity import player_key

    out = {"nhl": 0, "ahl": 0, "nhl_total": 0.0, "ahl_total": 0.0}
    log: List[Dict[str, Any]] = []
    for team in getattr(league, "teams", None) or []:
        for attr, kind in (("roster", "nhl"), ("ahl_roster", "ahl")):
            for p in list(getattr(team, attr, None) or []):
                age = _age(p)
                if age > 25 or getattr(p, "retired", False):
                    continue
                pid = player_key(p)
                ovr = _ovr99(p)
                pot = _pot99(p, ovr)
                gain = 0.0
                nhl_row = pss.get(pid) if isinstance(pss, dict) else None
                if isinstance(nhl_row, dict) and int(nhl_row.get("gp") or 0) > 0:
                    gp = int(nhl_row.get("gp") or 0)
                    toi_pg = float(nhl_row.get("toi_sec") or 0) / 60.0 / max(1, gp)
                    if str(nhl_row.get("position") or "").upper().startswith("G"):
                        toi_pg = 18.0 * min(1.0, gp / 50.0)
                    gain += nhl_gain(age, ovr, pot, gp, toi_pg)
                    src = "nhl"
                else:
                    src = kind
                if get_player_ahl_season_line is not None:
                    line = get_player_ahl_season_line(session, pid)
                    if line and int(line.get("gp") or 0) > 0:
                        gain += ahl_gain(age, ovr + gain, pot, line) * (0.6 if gain > 0 else 1.0)
                        src = "ahl" if src != "nhl" else "nhl+ahl"
                gain = min(gain, max(0.0, pot - ovr), MAX_GAIN)
                if gain < 0.25:
                    continue
                try:
                    real = _raise_ovr(p, gain)
                except Exception:
                    continue
                if real <= 0:
                    continue
                hist = list(getattr(p, "ice_time_dev_history", None) or [])
                hist.append({"season": sy, "source": src, "gain": round(real, 2), "ovr_after": round(ovr + real, 1)})
                setattr(p, "ice_time_dev_history", hist[-8:])
                key = "nhl" if "nhl" in src else "ahl"
                out[key] += 1
                out[f"{key}_total"] += real
                if team is not None and str(getattr(team, "team_id", "")) == str(getattr(session, "user_team_id", "")):
                    log.append({"player_id": pid, "name": getattr(getattr(p, "identity", None), "name", ""), "source": src, "gain": round(real, 1)})
    session._ice_dev_applied_year = sy
    session.ice_time_dev_report = {"season": sy, "user_team": sorted(log, key=lambda r: -r["gain"])[:25], **{k: round(v, 1) if isinstance(v, float) else v for k, v in out.items()}}
    return out


# ---------------------------------------------------------------------------
# Season performance vs expectation → potential (runs once at season end,
# before the annual development budget so the new runway is used this year).
# ---------------------------------------------------------------------------

# Minimum runway (0-1 scale) a young NHLer keeps above his current OVR. Imported
# real-NHL "pot" sits only 2-4 points above OVR, so year one used up most of it and
# year two+ had nothing left to grow into. Players still develop until ~26-27.
YOUNG_RUNWAY_FLOOR = {18: 0.06, 19: 0.06, 20: 0.055, 21: 0.048, 22: 0.04, 23: 0.03, 24: 0.02, 25: 0.012, 26: 0.006}


def _season_performance_index(p: Any, row: Dict[str, Any], ovr: float) -> float:
    """Season result vs what a player of this OVR normally posts. 0 = as expected,
    +0.5 = 50% better, -0.5 = 50% worse. Clamped to [-0.8, 1.2]."""
    gp = int(row.get("gp") or 0)
    if gp <= 0:
        return 0.0
    pos = str(row.get("position") or getattr(p, "position", "") or "").upper()
    if pos.startswith("G"):
        sv = float(row.get("save_pct") or row.get("sv_pct") or 0.0)
        shots = float(row.get("shots_against") or row.get("goalie_shots_against") or 0.0)
        if sv <= 0 and shots > 0:
            sv = float(row.get("saves") or 0.0) / shots
        if sv > 1.5:
            sv /= 100.0
        if sv <= 0:
            return 0.0
        exp_sv = 0.906 + max(-0.012, min(0.014, (ovr - 82.0) * 0.0009))
        return max(-0.8, min(1.2, (sv - exp_sv) * 45.0))
    try:
        from app.sim_engine.progression.development import _expected_ppg_for_ovr

        exp_ppg = float(_expected_ppg_for_ovr(ovr))
    except Exception:
        exp_ppg = 0.55
    # PPG-for-OVR anchors run a bit high for forwards and far high for defence.
    exp_ppg *= 0.8 if pos.startswith("D") else 0.87
    pts = float(row.get("points") or row.get("pts") or 0.0)
    ppg = pts / float(gp)
    return max(-0.8, min(1.2, ppg / max(0.08, exp_ppg) - 1.0))


def apply_season_performance_potential(session: Any) -> Dict[str, Any]:
    """Once per season: move potential (and keep a young-player runway) from how the
    season went against expectations. Great seasons raise potential at any age (less
    for veterans); poor seasons from young players trim it a little. The annual
    development pass then grows OVR into that runway."""
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    if int(getattr(session, "_perf_potential_year", 0) or 0) == sy:
        return {"skipped": True}
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"skipped": True}
    try:
        from app.sim_engine.progression.development import resolve_development_profile
        from app.sim_engine.entities.player import display_rating
        from services.lineup_integrity import player_key
    except Exception:
        return {"skipped": True}
    pss = getattr(session, "player_season_stats", None) or {}
    raised = lowered = floored = 0
    for team in getattr(league, "teams", None) or []:
        for p in list(getattr(team, "roster", None) or []):
            if getattr(p, "retired", False):
                continue
            try:
                age = _age(p)
                ovr = _ovr99(p)
                profile = resolve_development_profile(p)
                cur = float(profile.get("current_ovr") or ovr / 99.0)
                exp = float(profile.get("expected_ceiling") or cur)
                mx = float(profile.get("maximum_ceiling") or exp)
                row = pss.get(player_key(p)) if isinstance(pss, dict) else None
                row = row if isinstance(row, dict) else {}
                gp = int(row.get("gp") or 0)
                perf = _season_performance_index(p, row, ovr) if gp >= 25 else 0.0
                weight = min(1.0, gp / 70.0)
                if age <= 26:
                    age_mult = 1.0
                elif age <= 30:
                    age_mult = 0.7
                elif age <= 34:
                    age_mult = 0.45
                else:
                    age_mult = 0.25
                new_exp = exp
                if perf > 0.12:
                    new_exp += min(0.035, (perf - 0.12) * 0.045) * age_mult * weight
                elif perf < -0.3 and age <= 24:
                    new_exp -= min(0.012, (-perf - 0.3) * 0.03) * weight
                floor = YOUNG_RUNWAY_FLOOR.get(age, 0.06 if age < 18 else 0.0)
                if floor > 0:
                    floor *= max(0.5, min(1.5, 1.0 + perf * 0.8))
                    if cur + floor > new_exp:
                        new_exp = cur + floor
                        floored += 1
                new_exp = max(cur, min(0.99, new_exp))
                if abs(new_exp - exp) < 0.002:
                    continue
                if new_exp > exp:
                    raised += 1
                    # Older players who blew past expectations also get a real OVR bump
                    # through the existing career-year path in apply_player_development.
                    if age >= 27 and perf >= 0.35 and gp >= 50:
                        try:
                            p._dev_season_points_surplus = max(
                                float(getattr(p, "_dev_season_points_surplus", 0.0) or 0.0), 12.0
                            )
                        except Exception:
                            _swallowed_log.debug("suppressed exception", exc_info=True)
                else:
                    lowered += 1
                profile["expected_ceiling"] = new_exp
                profile["maximum_ceiling"] = min(0.99, max(mx, new_exp + 0.02))
                if "active_development_ceiling" in profile:
                    profile["active_development_ceiling"] = new_exp
                p.development_profile = profile
                from app.sim_engine.progression.potential import write_player_potential99

                write_player_potential99(p, float(new_exp) * 99.0)
                p.season_potential_delta = round((new_exp - exp) * 99.0, 1)
                hist = list(getattr(p, "potential_history", None) or [])
                hist.append({"season": sy, "pot_before": round(exp * 99.0, 1), "pot_after": round(new_exp * 99.0, 1), "performance": round(perf, 2)})
                p.potential_history = hist[-10:]
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
    session._perf_potential_year = sy
    return {"raised": raised, "lowered": lowered, "runway_floor": floored}
