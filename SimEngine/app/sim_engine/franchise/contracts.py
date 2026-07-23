"""Contracts, cap hits, and contract office."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

def _player_cap_hit_millions(player: Any) -> float:
    for key in ("cap_hit_m", "contract_aav_m", "aav_m"):
        try:
            v = float(getattr(player, key, 0) or 0)
            if v > 0:
                return v
        except Exception:
            pass
    c = getattr(player, "contract", None)
    if c is not None:
        for key in ("cap_hit_m", "cap_hit", "aav_m", "aav", "salary_aav"):
            try:
                v = float(getattr(c, key, 0) or 0)
                if v <= 0:
                    continue
                # Convert dollars to millions when stored as raw salary.
                if key in ("salary_aav", "aav", "cap_hit") and v > 250:
                    return v / 1_000_000.0
                return v
            except Exception:
                pass
    return 0.0
def _team_cap_snapshot(team: Any, sim: Any) -> Dict[str, float]:
    econ = ((getattr(getattr(sim, "league", None), "get_league_context", lambda: {})() or {}).get("economics") or {})
    cap_raw = float(econ.get("salary_cap", 92_000_000.0) or 92_000_000.0)
    salary_cap_m = cap_raw / 1_000_000.0 if cap_raw > 250 else cap_raw
    payroll_m = 0.0
    for p in (getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        payroll_m += _player_cap_hit_millions(p)
    if payroll_m <= 0.0:
        payroll_m = float(getattr(team, "total_cap_hit", 0) or 0)
        if payroll_m > 250:
            payroll_m /= 1_000_000.0
    cap_space_m = max(0.0, salary_cap_m - payroll_m)
    return {
        "salary_cap": round(float(salary_cap_m), 3),
        "cap_hit": round(float(payroll_m), 3),
        "cap_space": round(float(cap_space_m), 3),
    }
