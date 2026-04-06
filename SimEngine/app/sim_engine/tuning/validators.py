# app/sim_engine/tuning/validators.py
"""
Safety net: league health, rosters, prospect pipeline, cap sanity. Auto-corrects in place where possible.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from app.sim_engine.tuning.era_modifiers import resolve_era_profile


def _ovr(p: Any) -> float:
    fn = getattr(p, "ovr", None)
    if callable(fn):
        try:
            v = float(fn())
        except Exception:
            v = 0.5
    else:
        v = float(getattr(p, "ovr", 0.5))
    return v / 99.0 if v > 1.5 else v


def _team_id(team: Any) -> str:
    for a in ("team_id", "id", "abbr", "code"):
        v = getattr(team, a, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    return str(id(team))


def _global_unassigned_developable(league: Any) -> int:
    """World-pool inventory aged ~prospect (not on an NHL org). Complements team prospect_pool counts."""
    gp = getattr(league, "global_player_pool", None)
    if gp is None:
        gp = getattr(league, "global_prospect_pool", None)
    if not isinstance(gp, list):
        return 0
    n = 0
    for p in gp:
        if getattr(p, "team_id", None):
            continue
        if str(getattr(p, "status", "") or "") != "global":
            continue
        try:
            ag = int(getattr(p, "age", 0) or 0)
        except (TypeError, ValueError):
            ag = 0
        if 15 <= ag <= 24:
            n += 1
    return n


def validate_league_state(
    league_state: MutableMapping[str, Any],
    league: Any,
    teams: Sequence[Any],
    *,
    rng: Optional[random.Random] = None,
    year: int = 0,
) -> Dict[str, Any]:
    """
    Detect stat explosion / talent collapse signals; auto-correct macro indices.
    """
    report: Dict[str, Any] = {"issues": [], "fixes": []}
    if not teams:
        report["issues"].append("no_teams")
        return report

    all_ovr: List[float] = []
    n_players = 0
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            try:
                all_ovr.append(_ovr(p))
                n_players += 1
            except Exception:
                pass

    if n_players == 0:
        report["issues"].append("zero_active_players")
        league_state["league_health"] = max(0.05, float(league_state.get("league_health", 0.4) or 0.4) - 0.20)
        report["fixes"].append("league_health_emergency_low")
        r0 = rng if rng is not None else random.Random(7919)
        sim0 = getattr(league, "_runner_sim_engine", None)
        if sim0 is not None and hasattr(sim0, "ecosystem_operational_repairs"):
            try:
                rep0 = sim0.ecosystem_operational_repairs(list(teams), r0, int(year))
                report["fixes"].extend(rep0)
            except Exception as ex:
                report["fixes"].append(f"ecosystem_repair_failed:{type(ex).__name__}")
        return report

    mean_o = sum(all_ovr) / len(all_ovr)
    if mean_o < 0.58:
        report["issues"].append("talent_collapse_mean")
        if n_players < max(320, int(len(teams)) * 14):
            league_state["league_health"] = max(0.08, float(league_state.get("league_health", 0.5) or 0.5) - 0.12)
            r_coll = rng if rng is not None else random.Random(1201)
            simc = getattr(league, "_runner_sim_engine", None)
            if simc is not None and hasattr(simc, "ecosystem_operational_repairs"):
                try:
                    report["fixes"].extend(simc.ecosystem_operational_repairs(list(teams), r_coll, int(year)))
                except Exception as ex:
                    report["fixes"].append(f"ecosystem_on_talent_collapse:{type(ex).__name__}")
        else:
            league_state["parity_index"] = min(0.92, float(league_state.get("parity_index", 0.5) or 0.5) + 0.06)
            league_state["league_health"] = min(0.92, float(league_state.get("league_health", 0.5) or 0.5) + 0.05)
        report["fixes"].append("honest_health_on_collapse_or_boost_if_rosters_full")
    if mean_o > 0.78:
        report["issues"].append("stat_inflation_mean")
        league_state["chaos_index"] = min(0.95, float(league_state.get("chaos_index", 0.5) or 0.5) + 0.04)
        league_state["parity_index"] = max(0.12, float(league_state.get("parity_index", 0.5) or 0.5) - 0.05)
        report["fixes"].append("increase_chaos_reduce_parity")

    top = max(all_ovr)
    if top > 0.985:
        report["issues"].append("elite_saturation_top")
        report["fixes"].append("flagged_for_player_normalize")
    elif top > 0.97:
        report["fixes"].append("elite_elevated_watch_only")

    if top > 0.985:
        elite_nudged = 0
        ranked: List[Tuple[Any, float]] = []
        for t in teams:
            for p in getattr(t, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                try:
                    ranked.append((p, _ovr(p)))
                except Exception:
                    pass
        ranked.sort(key=lambda x: x[1], reverse=True)
        for p, ov in ranked[:6]:
            if ov < 0.93:
                break
            rmap = getattr(p, "ratings", None)
            if not isinstance(rmap, dict):
                continue
            for k in list(rmap.keys()):
                try:
                    rmap[k] = int(max(20, min(99, round(float(rmap[k]) * 0.997))))
                except (TypeError, ValueError):
                    pass
            elite_nudged += 1
        if elite_nudged:
            report["fixes"].append(f"elite_soft_cap_ratings_n={elite_nudged}")

    prospect_total = 0
    for t in teams:
        prospect_total += len(getattr(t, "prospect_pool", None) or [])
    thin_threshold = max(10, int(len(teams) * 1.35))
    world_inv = _global_unassigned_developable(league)
    gp = getattr(league, "global_player_pool", None) or getattr(league, "global_prospect_pool", None)
    gp_n = len(gp) if isinstance(gp, list) else 0
    boot = bool(getattr(league, "_global_player_pool_bootstrapped", False))
    pre_ecosystem = (not boot and gp_n < 120) or (gp_n < 40 and prospect_total == 0)
    world_ok = world_inv >= max(64, int(len(teams) * 2)) or (
        boot and gp_n >= max(200, int(len(teams) * 5))
    )
    effective_pipe = prospect_total + min(world_inv, int(len(teams) * 9)) // 3
    if pre_ecosystem and prospect_total < thin_threshold:
        report["fixes"].append(
            f"pipeline_thin_pre_global_cycle org={prospect_total} global_n={gp_n} (expected before draft/bootstrap)"
        )
    elif prospect_total < thin_threshold and not world_ok:
        report["issues"].append("thin_prospect_pipeline")
        league_state["league_health"] = max(0.10, float(league_state.get("league_health", 0.5) or 0.5) - 0.04)
        report["fixes"].append("operational_pipeline_repair_attempt")
        r1 = rng if rng is not None else random.Random(int(year) + 517)
        sim = getattr(league, "_runner_sim_engine", None)
        if sim is not None and hasattr(sim, "ecosystem_operational_repairs"):
            try:
                rep = sim.ecosystem_operational_repairs(list(teams), r1, int(year))
                for ln in rep:
                    report["fixes"].append(str(ln))
            except Exception as ex:
                report["fixes"].append(f"ecosystem_repair_exc:{type(ex).__name__}")
        try:
            setattr(league, "_boost_next_draft_class", True)
            prev = float(getattr(league, "_pipeline_dev_boost_one_year", 1.0) or 1.0)
            setattr(league, "_pipeline_dev_boost_one_year", max(prev, 1.065))
        except Exception:
            pass
    elif prospect_total < thin_threshold and world_ok:
        report["fixes"].append(
            f"pipeline_org_shallow_world_ok org={prospect_total} world_dev={world_inv} effective~{effective_pipe}"
        )

    setattr(league, "_tuning_validation_league", dict(report))
    return report


def validate_team(team: Any, league_state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Roster size sanity, cap-ish attributes. Auto-correct overload/underload.
    """
    report: Dict[str, Any] = {"team_id": _team_id(team), "issues": [], "fixes": []}
    roster = list(getattr(team, "roster", None) or [])
    active = [p for p in roster if not getattr(p, "retired", False)]

    if len(active) == 0:
        report["issues"].append("empty_roster")
        return report

    MAX_ROSTER = 28
    if len(active) > MAX_ROSTER:
        report["issues"].append("roster_overflow")
        # Remove lowest OVR players until at cap (auto-correct)
        scored = [(p, _ovr(p)) for p in active]
        scored.sort(key=lambda x: x[1])
        overflow = len(active) - MAX_ROSTER
        to_drop = [p for p, _ in scored[:overflow]]
        for p in to_drop:
            try:
                roster.remove(p)
            except ValueError:
                pass
        setattr(team, "roster", roster)
        report["fixes"].append(f"trimmed_{len(to_drop)}_contracts")

    if len(active) < 18:
        report["issues"].append("roster_understaffed")
        report["fixes"].append("noted_understaffed")

    cap_hit = getattr(team, "total_cap_hit", None)
    cap_space = getattr(team, "cap_space", None)
    if cap_hit is not None and cap_space is not None:
        try:
            ch, cs = float(cap_hit), float(cap_space)
            if cs < -2_000_000:
                report["issues"].append("severe_cap_over")
                factor = 0.985
                for p in active:
                    c = getattr(p, "contract", None)
                    if c is not None and hasattr(c, "aav"):
                        try:
                            setattr(c, "aav", max(750_000.0, float(c.aav) * factor))
                        except Exception:
                            pass
                report["fixes"].append("soft_contract_trim")
        except (TypeError, ValueError):
            pass

    return report


def validate_player_distribution(players: Sequence[Any], league_state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Distribution health; auto-correct spread via targeted micro-nudges on outliers.
    """
    report: Dict[str, Any] = {"issues": [], "fixes": [], "adjusted": 0}
    plist = [p for p in players if p is not None and not getattr(p, "retired", False)]
    if len(plist) < 8:
        return report

    ovrs = sorted(_ovr(p) for p in plist)
    mean = sum(ovrs) / len(ovrs)
    var = sum((x - mean) ** 2 for x in ovrs) / len(ovrs)
    std = var ** 0.5
    era = resolve_era_profile(str(league_state.get("active_era", "") or ""))

    if std < 0.045:
        report["issues"].append("distribution_too_tight")
        # Widen slightly: nudge top/bottom decile
        plist_sorted = sorted(plist, key=_ovr)
        n = max(1, len(plist_sorted) // 10)
        for p in plist_sorted[:n]:
            r = getattr(p, "ratings", None)
            if r and isinstance(r, dict):
                for k in list(r.keys()):
                    try:
                        r[k] = int(max(20, min(99, round(float(r[k]) * 0.996))))
                    except (TypeError, ValueError):
                        pass
                report["adjusted"] += 1
        for p in plist_sorted[-n:]:
            r = getattr(p, "ratings", None)
            if r and isinstance(r, dict):
                for k in list(r.keys()):
                    try:
                        r[k] = int(max(20, min(99, round(float(r[k]) * 1.004))))
                    except (TypeError, ValueError):
                        pass
                report["adjusted"] += 1
        report["fixes"].append("widened_distribution_deciles")

    if mean < 0.60 and era.get("prospect_growth_boost", 1.0) < 1.0:
        report["issues"].append("low_mean_in_low_dev_era")
        report["fixes"].append("recommend_era_shift_next_year")

    return report


def run_full_validation(
    league_state: MutableMapping[str, Any],
    league: Any,
    teams: Sequence[Any],
    all_players: Sequence[Any],
    *,
    rng: Optional[random.Random] = None,
    year: int = 0,
) -> Dict[str, Any]:
    """Run all validators; return aggregated report for logging."""
    out: Dict[str, Any] = {
        "league": validate_league_state(league_state, league, teams, rng=rng, year=int(year)),
        "teams": [],
        "players": {},
    }
    for t in teams:
        out["teams"].append(validate_team(t, league_state))
    out["players"] = validate_player_distribution(all_players, league_state)
    return out
