"""
Contract economy stress audit — simulates multiple offseason cycles and validates
the living cap/contract system without adding new gameplay features.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Path bootstrap for standalone script execution
_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
_SIM = _ROOT / "SimEngine"
for _p in (str(_BACKEND), str(_SIM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class ValidationIssue:
    severity: str  # error | warning | info
    code: str
    message: str
    season: Optional[int] = None
    team_id: Optional[str] = None
    player_id: Optional[str] = None


@dataclass
class SeasonSnapshot:
    season_year: int
    over_cap_teams: List[Dict[str, Any]] = field(default_factory=list)
    over_cap_after_compliance: List[str] = field(default_factory=list)
    cpu_signings: List[Dict[str, Any]] = field(default_factory=list)
    cpu_rfa_re_signed: int = 0
    cpu_rfa_walked: int = 0
    cpu_rfa_stranded: List[str] = field(default_factory=list)
    rfa_rights_total: int = 0
    unsigned_ufa_count: int = 0
    elc_count: int = 0
    pipeline_elc_count: int = 0
    unsigned_prospect_count: int = 0
    prospect_promotions: int = 0
    elc_signed: int = 0
    buyout_dead_cap_m: float = 0.0
    buried_cap_total_m: float = 0.0
    teams_at_slot_limit: List[str] = field(default_factory=list)
    positional_imbalance: List[Dict[str, Any]] = field(default_factory=list)
    cap_mismatch_teams: List[str] = field(default_factory=list)
    missing_contract_players: List[str] = field(default_factory=list)
    raw_dollar_contracts: List[str] = field(default_factory=list)
    nmc_violations: List[str] = field(default_factory=list)
    cpu_signings_by_band: Dict[str, int] = field(default_factory=dict)
    cpu_signings_by_position: Dict[str, int] = field(default_factory=dict)
    unsigned_ufas_by_band: Dict[str, int] = field(default_factory=dict)
    post_fa_shape_issues: List[Dict[str, Any]] = field(default_factory=list)
    waiver_exposed: int = 0
    waiver_claimed: int = 0
    waiver_cleared: int = 0
    waiver_claims: List[Dict[str, Any]] = field(default_factory=list)
    waiver_exposed_by_band: Dict[str, int] = field(default_factory=dict)
    waiver_claims_by_band: Dict[str, int] = field(default_factory=dict)
    buried_contracts: int = 0
    buried_cap_relief_m: float = 0.0
    buyouts_executed: int = 0
    buyout_cap_cleared_m: float = 0.0
    cap_casualty_trades: int = 0
    cap_casualty_cap_cleared_m: float = 0.0
    cap_casualty_trade_rows: List[Dict[str, Any]] = field(default_factory=list)
    teams_forced_cap_casualty: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    seed: int
    seasons_simulated: int
    start_year: int
    end_year: int
    issues: List[ValidationIssue] = field(default_factory=list)
    season_snapshots: List[SeasonSnapshot] = field(default_factory=list)
    worst_cap_teams: List[Dict[str, Any]] = field(default_factory=list)
    biggest_bad_contracts: List[Dict[str, Any]] = field(default_factory=list)
    best_bargains: List[Dict[str, Any]] = field(default_factory=list)
    top_unsigned_ufas: List[Dict[str, Any]] = field(default_factory=list)
    cpu_weird_signings: List[Dict[str, Any]] = field(default_factory=list)
    cpu_signings_by_band_total: Dict[str, int] = field(default_factory=dict)
    cpu_signings_by_position_total: Dict[str, int] = field(default_factory=dict)
    unsigned_ufas_by_band_final: Dict[str, int] = field(default_factory=dict)
    post_fa_shape_issues_total: List[Dict[str, Any]] = field(default_factory=list)
    waiver_exposed_total: int = 0
    waiver_claimed_total: int = 0
    waiver_cleared_total: int = 0
    waiver_by_band_total: Dict[str, int] = field(default_factory=dict)
    buyouts_total: int = 0
    buyout_cap_cleared_total_m: float = 0.0
    cap_casualty_trades_total: int = 0
    cap_casualty_cap_cleared_total_m: float = 0.0
    cap_casualty_trade_log: List[Dict[str, Any]] = field(default_factory=list)
    players_lost: List[str] = field(default_factory=list)
    stars_sample: List[Dict[str, Any]] = field(default_factory=list)
    depth_sample: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _player_id(p: Any) -> str:
    return str(_get(p, "id", "") or _get(p, "player_id", "") or "")


def _team_id(t: Any) -> str:
    return str(_get(t, "team_id", "") or _get(t, "id", "") or "")


def _player_ovr(p: Any) -> float:
    fn = _get(p, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


def _player_pos(p: Any) -> str:
    ident = _get(p, "identity", None)
    pos = _get(ident, "position", None) if ident else None
    if pos is not None and hasattr(pos, "value"):
        pos = pos.value
    return str(pos or _get(p, "position", "C") or "C").upper()


def _iter_all_players(league: Any) -> List[Any]:
    seen: Set[str] = set()
    out: List[Any] = []
    for team in _get(league, "teams", None) or []:
        for p in _get(team, "roster", None) or []:
            pid = _player_id(p)
            if pid and pid not in seen:
                seen.add(pid)
                out.append(p)
        for p in _get(team, "prospect_pool", None) or []:
            pid = _player_id(p)
            if pid and pid not in seen:
                seen.add(pid)
                out.append(p)
        for entry in _get(team, "reserve_list", None) or []:
            p = entry.get("player_ref") if isinstance(entry, dict) else None
            if p is not None:
                pid = _player_id(p)
                if pid and pid not in seen:
                    seen.add(pid)
                    out.append(p)
        for r in _get(team, "rfa_rights", None) or []:
            p = r.get("player_ref") if isinstance(r, dict) else None
            if p is not None:
                pid = _player_id(p)
                if pid and pid not in seen:
                    seen.add(pid)
                    out.append(p)
    for p in _get(league, "free_agents", None) or []:
        pid = _player_id(p)
        if pid and pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def _collect_player_registry(league: Any) -> Dict[str, str]:
    """player_id -> location label"""
    reg: Dict[str, str] = {}
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        for p in _get(team, "roster", None) or []:
            reg[_player_id(p)] = f"roster:{tid}"
        for p in _get(team, "prospect_pool", None) or []:
            reg[_player_id(p)] = f"prospect:{tid}"
        for entry in _get(team, "reserve_list", None) or []:
            pid = str(entry.get("player_id", "")) if isinstance(entry, dict) else ""
            if pid:
                reg[pid] = f"reserve:{tid}"
        for r in _get(team, "rfa_rights", None) or []:
            pid = str(r.get("player_id", "")) if isinstance(r, dict) else ""
            if pid:
                reg[pid] = f"rfa_rights:{tid}"
    for p in _get(league, "free_agents", None) or []:
        reg[_player_id(p)] = "ufa_pool"
    return reg


def _is_pipeline_elc(player: Any) -> bool:
    from services.contract_economy import has_elc_contract

    if not has_elc_contract(player):
        return False
    c = _get(player, "contract", None)
    if isinstance(c, dict):
        return str(c.get("source", "")).lower() == "elc"
    return str(_get(c, "source", "")).lower() == "elc"


def _seed_audit_prospect_pipeline(session: Any) -> int:
    """Inject draft-style org prospects when the sim has no global pipeline depth."""
    from types import SimpleNamespace

    from services.contract_economy import add_to_reserve_list

    league = session.sim.league
    rng = session.sim.rng
    season = int(session.season_calendar_year)
    positions = ["C", "LW", "RW", "D", "G"]
    added = 0
    for idx, team in enumerate(_get(league, "teams", None) or []):
        pool = getattr(team, "prospect_pool", None)
        if pool is None:
            team.prospect_pool = []
            pool = team.prospect_pool
        tid = _team_id(team)
        need = max(0, 2 - len(pool))
        for j in range(need):
            age = 20 + rng.randint(0, 4)
            pos = positions[(idx + j) % len(positions)]
            pid = f"audit-prospect-{tid}-{season}-{j}"
            p = SimpleNamespace(
                id=pid,
                player_id=pid,
                identity=SimpleNamespace(name=f"Prospect {pid[-10:]}", age=age, position=pos),
                position=pos,
                age=age,
                ovr=lambda o=0.55 + rng.random() * 0.22: o,
                development_years_remaining=0 if age >= 22 else 1,
                draft_value_range=(0.52, 0.78),
                signed_status="unsigned",
                entry_level_contract_eligible=True,
                team_id=tid,
            )
            pool.append(p)
            add_to_reserve_list(
                team,
                p,
                draft_year=season - 1,
                draft_overall=10 + j,
                added_season=season,
            )
            added += 1
    return added


def bootstrap_audit_session(seed: int = 424242, *, full_franchise: bool = False) -> Any:
    """Create a FranchiseSession with contracts bootstrapped. Fast path by default."""
    from services._simengine_bootstrap import ensure_simengine_path

    ensure_simengine_path()

    if full_franchise:
        from services.franchise_sim import start_franchise

        return start_franchise(
            team_query="Toronto",
            head_coach_name="Audit Coach",
            coach_archetype="balanced",
            seed=seed,
            games_per_team=82,
        )

    from app.sim_engine.engine import SimEngine
    from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy
    from app.sim_engine.league.schedule_generator import _safe_team_id
    from services.franchise_session import FranchiseSession
    from services.franchise_sim import _ensure_league_roster_contracts

    sim = SimEngine(seed=seed, debug=False)
    league = sim.league
    teams = list(_get(league, "teams", None) or [])
    if not teams:
        raise RuntimeError("No teams in league")

    try:
        bootstrap_full_league_hierarchy(league, sim.rng)
    except Exception:
        pass

    season_y = 2025
    _ensure_league_roster_contracts(league, season_y)

    try:
        from services.contract_economy import install_prospect_contract_hooks

        install_prospect_contract_hooks(league)
        depth_fn = getattr(sim, "ensure_prospect_pipeline_depth", None)
        if callable(depth_fn):
            depth_fn(season_y, sim.rng)
    except Exception:
        pass

    team_by_id: Dict[str, Any] = {}
    team_ids: List[str] = []
    for idx, t in enumerate(teams):
        tid = _safe_team_id(t, idx)
        team_ids.append(tid)
        team_by_id[tid] = t

    user_tid = team_ids[0]
    sim.team = team_by_id[user_tid]

    session = FranchiseSession(
        session_id=f"audit-{seed}",
        sim=sim,
        user_team_id=user_tid,
        head_coach_name="Audit Coach",
        coach_archetype="balanced",
        season_calendar_year=season_y,
        team_by_id=team_by_id,
        team_ids=team_ids,
        phase="offseason",
        season_phase="offseason",
        offseason_stage="salary_cap",
    )
    pool_total = sum(len(getattr(t, "prospect_pool", None) or []) for t in teams)
    if pool_total == 0:
        try:
            _seed_audit_prospect_pipeline(session)
        except Exception:
            pass
    return session


def _scan_contract_money_units(league: Any) -> List[str]:
    bad: List[str] = []
    for p in _iter_all_players(league):
        c = _get(p, "contract", None)
        if c is None:
            continue
        if isinstance(c, dict):
            for k in ("aav_m", "cap_hit_m", "aav", "cap_hit", "salary_m"):
                v = c.get(k)
                if v is not None and abs(float(v)) > 250:
                    bad.append(_player_id(p))
                    break
        else:
            for k in ("aav_m", "cap_hit_m", "aav", "cap_hit"):
                v = _get(c, k, None)
                if v is not None and abs(float(v)) > 250:
                    bad.append(_player_id(p))
                    break
    return bad


def _scan_missing_contracts(league: Any) -> List[str]:
    missing: List[str] = []
    for team in _get(league, "teams", None) or []:
        for p in _get(team, "roster", None) or []:
            if _get(p, "retired", False):
                continue
            if _get(p, "is_buried", False) or _get(p, "buried", False):
                continue
            from services.contract_economy import has_active_contract

            if not has_active_contract(p):
                if _get(p, "entry_level_contract_eligible", False):
                    continue
                missing.append(_player_id(p))
    return missing


def _scan_fake_elc_contracts(league: Any) -> List[str]:
    from services.contract_economy import ELC_AAV_M, ELC_AAV_TOLERANCE, normalize_contract_payload

    bad: List[str] = []
    for p in _iter_all_players(league):
        c = normalize_contract_payload(p)
        if str(c.get("type") or c.get("contract_type") or "").upper() != "ELC":
            continue
        cap = float(c.get("cap_hit_m") or c.get("aav_m") or 0)
        if cap <= 0 or abs(cap - ELC_AAV_M) > ELC_AAV_TOLERANCE:
            bad.append(_player_id(p))
    return bad


def _scan_malformed_active_contracts(league: Any) -> List[str]:
    from services.contract_economy import has_active_contract

    bad: List[str] = []
    for p in _iter_all_players(league):
        if _get(p, "contract", None) is None:
            continue
        if has_active_contract(p):
            continue
        if _get(p, "entry_level_contract_eligible", False) and str(_get(p, "signed_status", "")).lower() == "unsigned":
            continue
        bad.append(_player_id(p))
    return bad


def _scan_contract_slot_violations(league: Any) -> List[str]:
    from services.contract_economy import get_team_cap_snapshot_full

    bad: List[str] = []
    for team in _get(league, "teams", None) or []:
        snap = get_team_cap_snapshot_full(team, league)
        if int(snap.get("contract_slots_used", 0)) > int(snap.get("contract_slots_limit", 50)):
            bad.append(_team_id(team))
    return bad


def _scan_waiver_gp_exempt_issues(league: Any) -> List[str]:
    from services.contract_economy import has_active_contract, has_true_elc_contract, is_waiver_exempt, normalize_contract_payload

    bad: List[str] = []
    for team in _get(league, "teams", None) or []:
        for p in _get(team, "roster", None) or []:
            if not has_active_contract(p) or has_true_elc_contract(p):
                continue
            age = int(_get(_get(p, "identity", None), "age", _get(p, "age", 0)) or 0)
            if 21 <= age <= 22 and is_waiver_exempt(p, team, league):
                ctype = str(normalize_contract_payload(p).get("type") or "").upper()
                if ctype == "STANDARD":
                    bad.append(_player_id(p))
    return bad


def _bootstrap_trim_summary(league: Any) -> Dict[str, Any]:
    log = list(_get(league, "_bootstrap_cap_trim_log", None) or [])
    total_reduction = round(sum(float(r.get("reduction_m", 0) or 0) for r in log), 3)
    elc_touched = sum(
        1 for r in log
        if float(r.get("old_aav_m", 0) or 0) > 0
        and abs(float(r.get("old_aav_m", 0)) - 0.95) <= 0.01
    )
    return {
        "trim_count": len(log),
        "total_reduction_m": total_reduction,
        "true_elc_trim_attempts": elc_touched,
    }


def _team_positional_counts(team: Any) -> Dict[str, int]:
    from services.contract_economy import _position_bucket

    counts: Dict[str, int] = {"C": 0, "LW": 0, "RW": 0, "LD": 0, "RD": 0, "G": 0, "D": 0}
    for p in _get(team, "roster", None) or []:
        if _get(p, "retired", False) or _get(p, "is_buried", False):
            continue
        pos = _position_bucket(p)
        if pos in counts:
            counts[pos] += 1
        elif pos == "D":
            counts["D"] += 1
    return counts


def _count_by_ovr_band(players: List[Any]) -> Dict[str, int]:
    from services.contract_economy import ovr_band

    bands = {"90+": 0, "85-89": 0, "80-84": 0, "75-79": 0, "70-74": 0, "under-70": 0}
    for p in players:
        bands[ovr_band(_player_ovr(p))] = bands.get(ovr_band(_player_ovr(p)), 0) + 1
    return bands


def _aggregate_signing_bands(signings: List[Dict[str, Any]]) -> Dict[str, int]:
    bands = {"90+": 0, "85-89": 0, "80-84": 0, "75-79": 0, "70-74": 0, "under-70": 0}
    for s in signings:
        band = str(s.get("ovr_band", "") or "")
        if not band and s.get("overall") is not None:
            from services.contract_economy import ovr_band
            band = ovr_band(float(s.get("overall", 0)))
        if band in bands:
            bands[band] += 1
    return bands


def _aggregate_signing_positions(signings: List[Dict[str, Any]]) -> Dict[str, int]:
    pos_counts: Dict[str, int] = {}
    for s in signings:
        pos = str(s.get("position", "C") or "C")
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pos_counts


def _merge_band_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0) + int(v)


def _analyze_cpu_signings(signings: List[Dict[str, Any]], league: Any) -> List[Dict[str, Any]]:
    from services.contract_economy import (
        CPU_POSITION_OVERLOAD,
        evaluate_team_position_needs,
        get_team_competitive_window,
    )

    weird: List[Dict[str, Any]] = []
    for s in signings:
        tid = str(s.get("team_id", ""))
        team = next((t for t in (_get(league, "teams", None) or []) if _team_id(t) == tid), None)
        if team is None:
            continue
        pid = str(s.get("player_id", ""))
        player = next((p for p in _iter_all_players(league) if _player_id(p) == pid), None)
        if player is None:
            continue
        pos = str(s.get("position", _player_pos(player)))
        ctx = evaluate_team_position_needs(team, league)
        counts = ctx["counts"]
        ovr = float(s.get("overall", _player_ovr(player)))
        aav = float(s.get("aav_m", 0))
        window = str(s.get("window", get_team_competitive_window(team, league)) or "")
        fit = float(s.get("fit_score", 0))

        reasons: List[str] = []
        if counts.get(pos, 0) >= CPU_POSITION_OVERLOAD.get(pos, 5) and ovr < ctx["best_ovr"].get(pos, 0) + 3:
            reasons.append(f"overload_{pos}")
        if counts.get("G", 0) >= 3 and pos == "G":
            reasons.append("goalie_overload")
        if window == "rebuilder" and aav >= 6.0 and ovr < 86:
            reasons.append("rebuilder_big_overpay")
        if aav >= 8.0 and ovr < 80:
            reasons.append("depth_star_money")
        if ovr < 70 and fit < 0.45:
            reasons.append("mistake_signing")
        if fit < 0.35:
            reasons.append("low_fit_score")
        if reasons:
            weird.append({**s, "reasons": reasons, "position": pos, "overall": round(ovr)})
    return weird


def _validate_waiver_integrity(league: Any) -> List[str]:
    errors: List[str] = []
    seen_on_teams: Dict[str, List[str]] = {}
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        for p in _get(team, "roster", None) or []:
            pid = _player_id(p)
            if pid:
                seen_on_teams.setdefault(pid, []).append(tid)
    for pid, tids in seen_on_teams.items():
        if len(tids) > 1:
            errors.append(f"duplicate_roster:{pid}:{'|'.join(tids)}")

    for entry in _get(league, "waiver_wire", None) or []:
        if entry.get("claimed_by") and entry.get("player_ref") is not None:
            pid = str(entry.get("player_id", ""))
            orig = str(entry.get("original_team_id", ""))
            claim = str(entry.get("claimed_by", ""))
            player = entry.get("player_ref")
            on_orig = any(
                _player_id(p) == pid
                for t in (_get(league, "teams", None) or [])
                if _team_id(t) == orig
                for p in (_get(t, "roster", None) or [])
            )
            if on_orig:
                errors.append(f"claimed_still_on_original:{pid}:{orig}->{claim}")
    return errors


def _validate_buried_contracts(league: Any) -> List[str]:
    from services.contract_economy import has_active_contract

    errors: List[str] = []
    for team in _get(league, "teams", None) or []:
        for p in _all_rostered(team):
            if (_get(p, "is_buried", False) or _get(p, "buried", False)) and not has_active_contract(p):
                errors.append(_player_id(p))
    return errors


def _validate_cap_casualty_trades(
    league: Any,
    trade_rows: List[Dict[str, Any]],
    *,
    season_year: Optional[int] = None,
) -> List[ValidationIssue]:
    from services.contract_economy import (
        get_team_cap_snapshot_full,
        has_nmc,
        has_true_elc_contract,
        is_core_player_protected,
    )
    from app.sim_engine.trades.trade_pick_registry import audit_pick_registry_integrity

    issues: List[ValidationIssue] = []
    season = season_year

    for trade in trade_rows or []:
        seller_tid = str(trade.get("seller_team_id", "") or "")
        buyer_tid = str(trade.get("buyer_team_id", "") or "")
        for mp in trade.get("players_moved") or []:
            pid = str(mp.get("asset_id") or mp.get("player_id") or "")
            for team in _get(league, "teams", None) or []:
                for p in _get(team, "roster", None) or []:
                    if _player_id(p) != pid:
                        continue
                    if has_nmc(p):
                        issues.append(ValidationIssue(
                            "error", "CAP_CASUALTY_NMC_TRADED",
                            f"NMC player {pid} moved in cap casualty trade",
                            season=season, team_id=seller_tid, player_id=pid,
                        ))
                    if has_true_elc_contract(p) and _team_id(team) == buyer_tid:
                        issues.append(ValidationIssue(
                            "error", "CAP_CASUALTY_ELC_DUMP",
                            f"True ELC {pid} traded as cap casualty",
                            season=season, player_id=pid,
                        ))
                    if is_core_player_protected(p, team, league) and _team_id(team) == seller_tid:
                        issues.append(ValidationIssue(
                            "warning", "CAP_CASUALTY_CORE_TRADED",
                            f"Core player {pid} traded as cap casualty",
                            season=season, team_id=seller_tid, player_id=pid,
                        ))
                    pot = float(_get(p, "ratings", {}).get("dev_potential", 0) if isinstance(_get(p, "ratings", {}), dict) else 0)
                    age = int(_get(_get(p, "identity", None), "age", _get(p, "age", 30)) or 30)
                    if age <= 23 and pot >= 78 and _team_id(team) == seller_tid:
                        issues.append(ValidationIssue(
                            "warning", "CAP_CASUALTY_YOUNG_POTENTIAL",
                            f"Young high-potential player {pid} traded as cap casualty",
                            season=season, team_id=seller_tid, player_id=pid,
                        ))

        if float(trade.get("buyer_cap_after_m", 0) or 0) < -0.01:
            issues.append(ValidationIssue(
                "error", "CAP_CASUALTY_BUYER_OVER_CAP",
                f"Buyer {buyer_tid} over cap after cap casualty trade",
                season=season, team_id=buyer_tid,
            ))
        if not trade.get("solved_cap") and float(trade.get("cap_cleared_m", 0) or 0) < 0.25:
            issues.append(ValidationIssue(
                "warning", "CAP_CASUALTY_FAILED_RELIEF",
                f"Seller {seller_tid} cap casualty trade did not meaningfully clear cap",
                season=season, team_id=seller_tid,
            ))

        bad_money = float(trade.get("bad_score", 0) or 0)
        picks_to_buyer = [
            p for p in (trade.get("picks_moved") or [])
            if str(p.get("acquiring_team_id", "")) == buyer_tid
        ]
        if bad_money >= 0.35 and not picks_to_buyer:
            issues.append(ValidationIssue(
                "warning", "CAP_CASUALTY_BAD_CONTRACT_NO_COMP",
                f"Buyer {buyer_tid} absorbed bad contract without compensation",
                season=season, team_id=buyer_tid,
            ))

    try:
        audit = audit_pick_registry_integrity(league)
        if not audit.get("ok"):
            issues.append(ValidationIssue(
                "error", "CAP_CASUALTY_PICK_REGISTRY",
                str((audit.get("errors") or ["pick registry mismatch"])[0]),
                season=season,
            ))
    except Exception as exc:
        issues.append(ValidationIssue(
            "error", "CAP_CASUALTY_PICK_REGISTRY",
            str(exc), season=season,
        ))

    dup = _validate_waiver_integrity(league)
    for err in dup:
        if err.startswith("duplicate_roster"):
            issues.append(ValidationIssue(
                "error", "CAP_CASUALTY_DUPLICATE_PLAYER", err, season=season,
            ))

    for trade in trade_rows or []:
        seller_tid = str(trade.get("seller_team_id", "") or "")
        seller = next((t for t in (_get(league, "teams", None) or []) if _team_id(t) == seller_tid), None)
        if seller is not None:
            snap = get_team_cap_snapshot_full(seller, league, season_year=season_year)
            if snap["usable_cap_space_m"] < -0.01:
                issues.append(ValidationIssue(
                    "warning", "CAP_CASUALTY_STILL_OVER_CAP",
                    f"Team {seller_tid} still over cap after cap casualty pass",
                    season=season, team_id=seller_tid,
                ))

    return issues


def _all_rostered(team: Any) -> List[Any]:
    return [p for p in (_get(team, "roster", None) or []) if not _get(p, "retired", False)]


def _aggregate_waiver_bands(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    bands = {"90+": 0, "85-89": 0, "80-84": 0, "75-79": 0, "70-74": 0, "under-70": 0}
    for row in rows:
        band = str(row.get("ovr_band", "") or "")
        if not band and row.get("overall") is not None:
            from services.contract_economy import ovr_band
            band = ovr_band(float(row.get("overall", 0)))
        if band in bands:
            bands[band] += 1
    return bands


def _validate_post_fa_audit(session: Any, cpu_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    from services.contract_economy import validate_post_fa_roster_shape

    league = session.sim.league
    issues = list(cpu_result.get("post_fa_issues") or [])
    if issues:
        return issues
    for team in _get(league, "teams", None) or []:
        issues.extend(validate_post_fa_roster_shape(team, league, session.sim))
    return issues


def simulate_offseason_cycle(session: Any) -> SeasonSnapshot:
    from app.sim_engine.economy.cap_engine import advance_league_salary_cap, player_cap_hit_millions
    from services.contract_economy import (
        compute_bad_contract_score,
        compute_contract_tags,
        compute_fair_aav,
        compute_team_needs,
        get_team_cap_snapshot_full,
        has_nmc,
        run_cap_compliance_before_season,
        run_cpu_free_agency,
        run_cpu_rfa_decisions,
        run_prospect_promotion_pass,
        team_cap_snapshot_legacy_compat,
    )
    from services.franchise_offseason import _advance_salary_cap
    from services.franchise_sim import _team_cap_snapshot

    league = session.sim.league
    season_year = int(session.season_calendar_year)
    snap = SeasonSnapshot(season_year=season_year)

    _advance_salary_cap(session)

    fa_before = len(_get(league, "free_agents", None) or [])
    rfa_before = sum(len(_get(t, "rfa_rights", None) or []) for t in (_get(league, "teams", None) or []))
    cpu_rfa = run_cpu_rfa_decisions(session)
    snap.cpu_rfa_re_signed = int(cpu_rfa.get("re_signed_count", 0) or 0)
    snap.cpu_rfa_walked = int(cpu_rfa.get("walked_count", 0) or 0)
    cpu = run_cpu_free_agency(session)
    snap.cpu_signings = list(cpu.get("signings") or [])
    snap.cpu_signings_by_band = _aggregate_signing_bands(snap.cpu_signings)
    snap.cpu_signings_by_position = _aggregate_signing_positions(snap.cpu_signings)
    snap.post_fa_shape_issues = _validate_post_fa_audit(session, cpu)
    snap.unsigned_ufas_by_band = _count_by_ovr_band(list(_get(league, "free_agents", None) or []))

    promo = run_prospect_promotion_pass(session)
    snap.prospect_promotions = int(promo.get("promoted", 0) or 0)
    snap.elc_signed = int(promo.get("elc_signed", 0) or 0)

    over_before: List[Dict[str, Any]] = []
    cap_mismatch: List[str] = []
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        full = get_team_cap_snapshot_full(team, league, session.sim, season_year=season_year)
        if full["usable_cap_space_m"] < -0.01:
            over_before.append({
                "team_id": tid,
                "cap_space_m": full["usable_cap_space_m"],
                "total_hit_m": full["total_cap_hit_m"],
            })
        legacy = team_cap_snapshot_legacy_compat(full)
        simple = _team_cap_snapshot(team, session.sim, season_year=season_year)
        if abs(legacy["cap_space"] - simple["cap_space"]) > 0.05:
            cap_mismatch.append(tid)

    compliance = run_cap_compliance_before_season(session)
    pipeline = compliance.get("pipeline") or {}
    waived_rows = pipeline.get("waived") or []
    claim_rows = pipeline.get("claims") or []
    cleared_rows = pipeline.get("cleared") or []
    buyout_rows = pipeline.get("buyouts") or []
    cap_casualty_rows = pipeline.get("cap_casualty_trades") or []

    history = _get(league, "waiver_history", None) or []
    history_exposed = [
        h for h in history
        if int(h.get("season_year", 0) or 0) == season_year
    ]

    snap.waiver_exposed = max(len(waived_rows), len(history_exposed))
    snap.waiver_claimed = len(claim_rows)
    snap.waiver_cleared = len(cleared_rows)
    snap.waiver_claims = list(claim_rows)
    history_band_rows = history_exposed or waived_rows
    snap.waiver_exposed_by_band = _aggregate_waiver_bands(
        [{"ovr_band": e.get("ovr_band"), "overall": e.get("overall")} for e in history_band_rows]
        or [{"ovr_band": e.get("waiver_entry", {}).get("ovr_band"), "overall": e.get("waiver_entry", {}).get("overall")} for e in waived_rows if e.get("waiver_entry")]
        or waived_rows
    )
    snap.waiver_claims_by_band = _aggregate_waiver_bands(claim_rows)
    snap.buyouts_executed = len(buyout_rows)
    snap.buyout_cap_cleared_m = round(sum(float(b.get("cap_savings_m", 0) or 0) for b in buyout_rows), 3)
    snap.cap_casualty_trades = len(cap_casualty_rows)
    snap.cap_casualty_cap_cleared_m = round(
        sum(float(t.get("cap_cleared_m", 0) or 0) for t in cap_casualty_rows), 3,
    )
    snap.cap_casualty_trade_rows = list(cap_casualty_rows)
    snap.teams_forced_cap_casualty = sorted({
        str(t.get("seller_team_id", "")) for t in cap_casualty_rows if t.get("seller_team_id")
    })

    from services.contract_economy import get_team_bury_relief_total

    for team in _get(league, "teams", None) or []:
        buried_n = sum(
            1 for p in _all_rostered(team)
            if _get(p, "is_buried", False) or _get(p, "buried", False)
        )
        snap.buried_contracts += buried_n
        snap.buried_cap_relief_m += get_team_bury_relief_total(team)

    snap.over_cap_teams = over_before

    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        full = get_team_cap_snapshot_full(team, league, session.sim, season_year=season_year)
        if full["usable_cap_space_m"] < -0.01:
            snap.over_cap_after_compliance.append(tid)

    snap.cap_mismatch_teams = cap_mismatch
    snap.raw_dollar_contracts = _scan_contract_money_units(league)
    snap.missing_contract_players = _scan_missing_contracts(league)

    user_tid = str(_get(session, "user_team_id", "") or "")
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        rights = _get(team, "rfa_rights", None) or []
        snap.rfa_rights_total += len(rights)
        # After the CPU RFA pass, no non-user team should still be holding RFA rights:
        # every restricted free agent must be re-signed or released to the market.
        if tid != user_tid and rights:
            snap.cpu_rfa_stranded.append(tid)
        full = get_team_cap_snapshot_full(team, league, session.sim, season_year=season_year)
        snap.buried_cap_total_m += full.get("buried_cap_hit_m", 0)
        snap.buyout_dead_cap_m += full.get("buyout_cap_hit_m", 0)
        if full.get("contract_slots_used", 0) >= full.get("contract_slots_limit", 50):
            snap.teams_at_slot_limit.append(tid)

        needs = compute_team_needs(team)
        counts = _team_positional_counts(team)
        gaps: List[str] = []
        if needs.get("G") == "high" and counts.get("G", 0) >= 2:
            gaps.append("goalie_need_unfilled")
        if needs.get("C") == "high" and counts.get("C", 0) >= 4:
            gaps.append("center_surplus")
        if counts.get("G", 0) >= 4:
            gaps.append("goalie_overload")
        if counts.get("C", 0) >= 6:
            gaps.append("center_overload")
        if gaps:
            snap.positional_imbalance.append({"team_id": tid, "counts": counts, "needs": needs, "flags": gaps})

        for p in _get(team, "roster", None) or []:
            if has_nmc(p) and (_get(p, "is_buried", False) or _get(p, "waiver_status") == "buried"):
                snap.nmc_violations.append(_player_id(p))

    snap.unsigned_ufa_count = len(_get(league, "free_agents", None) or [])

    for team in _get(league, "teams", None) or []:
        for p in _get(team, "prospect_pool", None) or []:
            if _get(p, "entry_level_contract_eligible", False) and _get(p, "signed_status", "") == "unsigned":
                snap.unsigned_prospect_count += 1
            c = _get(p, "contract", None)
            ctype = ""
            if isinstance(c, dict):
                ctype = str(c.get("contract_type") or c.get("type") or "")
            elif c is not None:
                ctype = str(_get(c, "contract_type", "") or "")
            if ctype == "ELC":
                snap.elc_count += 1
                if _is_pipeline_elc(p):
                    snap.pipeline_elc_count += 1
        for p in _get(team, "roster", None) or []:
            c = _get(p, "contract", None)
            ctype = ""
            if isinstance(c, dict):
                ctype = str(c.get("contract_type") or c.get("type") or "")
            elif c is not None:
                ctype = str(_get(c, "contract_type", "") or "")
            if ctype == "ELC":
                snap.elc_count += 1
                if _is_pipeline_elc(p):
                    snap.pipeline_elc_count += 1

    session.season_calendar_year = season_year + 1
    return snap


def _collect_league_contract_profiles(league: Any) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    from app.sim_engine.economy.cap_engine import player_cap_hit_millions
    from services.contract_economy import compute_bad_contract_score, compute_contract_tags, compute_fair_aav

    bad: List[Dict] = []
    bargains: List[Dict] = []
    stars: List[Dict] = []
    for team in _get(league, "teams", None) or []:
        tid = _team_id(team)
        for p in _get(team, "roster", None) or []:
            if _get(p, "retired", False):
                continue
            ovr = _player_ovr(p)
            aav = player_cap_hit_millions(p)
            if aav <= 0:
                continue
            fair = compute_fair_aav(p, team, league)
            score = compute_bad_contract_score(p, team)
            tags = compute_contract_tags(p, team)
            row = {
                "player_id": _player_id(p),
                "name": str(_get(_get(p, "identity", None), "name", _get(p, "name", "")) or ""),
                "team_id": tid,
                "overall": round(ovr),
                "aav_m": round(aav, 3),
                "fair_aav_m": round(fair, 3),
                "bad_score": score,
                "tags": tags,
            }
            if score >= 0.35:
                bad.append(row)
            if "Bargain" in tags:
                bargains.append(row)
            if ovr >= 88:
                stars.append(row)
    bad.sort(key=lambda r: -r["bad_score"])
    bargains.sort(key=lambda r: r["aav_m"] - r["fair_aav_m"])
    stars.sort(key=lambda r: -r["overall"])
    return bad, bargains, stars


def run_stress_audit(
    *,
    seasons: int = 8,
    seed: int = 424242,
    full_franchise: bool = False,
) -> AuditReport:
    from app.sim_engine.economy.cap_engine import player_cap_hit_millions
    from services.contract_economy import get_team_cap_snapshot_full

    session = bootstrap_audit_session(seed=seed, full_franchise=full_franchise)
    league = session.sim.league
    start_registry = _collect_player_registry(league)
    start_year = int(session.season_calendar_year)

    report = AuditReport(
        seed=seed,
        seasons_simulated=seasons,
        start_year=start_year,
        end_year=start_year + seasons - 1,
    )

    # Bootstrap baseline (before any simulated seasons)
    bad_boot, bargains_boot, stars_boot = _collect_league_contract_profiles(league)
    from services.contract_economy import get_team_cap_snapshot_full, validate_franchise_cap_at_start

    cap_issues_boot = validate_franchise_cap_at_start(league, start_year)
    for msg in cap_issues_boot:
        report.issues.append(ValidationIssue("error", "BOOTSTRAP_OVER_CAP", msg, season=start_year))

    if bad_boot:
        report.summary["bad_contracts_at_bootstrap"] = len(bad_boot)
    else:
        report.issues.append(ValidationIssue(
            "warning", "NO_BAD_CONTRACTS_AT_BOOT",
            "No bad contracts detected at franchise bootstrap — may be too rare or threshold too high",
            season=start_year,
        ))

    start_caps = []
    for team in _get(league, "teams", None) or []:
        snap = get_team_cap_snapshot_full(team, league, session.sim, season_year=start_year)
        start_caps.append(snap["upper_limit_m"])
    report.summary["bootstrap_cap_upper_m"] = round(sum(start_caps) / max(1, len(start_caps)), 2)

    boot_trim = _bootstrap_trim_summary(league)
    report.summary["bootstrap_trim_count"] = boot_trim["trim_count"]
    report.summary["bootstrap_trim_reduction_m"] = boot_trim["total_reduction_m"]
    if boot_trim["true_elc_trim_attempts"]:
        report.issues.append(ValidationIssue(
            "error", "BOOTSTRAP_ELC_TRIM",
            f"Bootstrap trim touched {boot_trim['true_elc_trim_attempts']} true ELC contracts",
            season=start_year,
        ))
    if boot_trim["trim_count"] >= 120:
        report.issues.append(ValidationIssue(
            "warning", "BOOTSTRAP_TRIM_EXCESSIVE",
            f"Bootstrap trimmed {boot_trim['trim_count']} contracts ({boot_trim['total_reduction_m']}M total)",
            season=start_year,
        ))

    for pid in _scan_fake_elc_contracts(league):
        report.issues.append(ValidationIssue(
            "error", "FAKE_ELC", f"Player {pid} has ELC label with non-ELC cap hit", season=start_year,
        ))
    for pid in _scan_malformed_active_contracts(league):
        report.issues.append(ValidationIssue(
            "error", "MALFORMED_ACTIVE_CONTRACT", f"Player {pid} has invalid active contract state", season=start_year,
        ))
    for tid in _scan_contract_slot_violations(league):
        report.issues.append(ValidationIssue(
            "error", "CONTRACT_SLOT_OVER", f"Team {tid} exceeds contract slot limit", season=start_year,
        ))

    all_cpu_signings: List[Dict[str, Any]] = []
    worst_cap: Dict[str, float] = {}
    cpu_bands_total: Dict[str, int] = {}
    cpu_pos_total: Dict[str, int] = {}
    post_fa_all: List[Dict[str, Any]] = []
    waiver_exposed_total = 0
    waiver_claimed_total = 0
    waiver_cleared_total = 0
    waiver_band_total: Dict[str, int] = {}
    buyouts_total = 0
    buyout_cap_total = 0.0
    buried_contracts_total = 0
    cpu_rfa_re_signed_total = 0
    cpu_rfa_walked_total = 0
    cap_casualty_total = 0
    cap_casualty_cleared_total = 0.0
    cap_casualty_log_all: List[Dict[str, Any]] = []

    for _ in range(seasons):
        ss = simulate_offseason_cycle(session)
        report.season_snapshots.append(ss)
        all_cpu_signings.extend(ss.cpu_signings)
        _merge_band_counts(cpu_bands_total, ss.cpu_signings_by_band)
        for pos, n in ss.cpu_signings_by_position.items():
            cpu_pos_total[pos] = cpu_pos_total.get(pos, 0) + int(n)
        post_fa_all.extend(ss.post_fa_shape_issues)

        waiver_exposed_total += ss.waiver_exposed
        waiver_claimed_total += ss.waiver_claimed
        waiver_cleared_total += ss.waiver_cleared
        _merge_band_counts(waiver_band_total, ss.waiver_exposed_by_band)
        buyouts_total += ss.buyouts_executed
        buyout_cap_total += ss.buyout_cap_cleared_m
        buried_contracts_total += ss.buried_contracts
        cpu_rfa_re_signed_total += ss.cpu_rfa_re_signed
        cpu_rfa_walked_total += ss.cpu_rfa_walked
        cap_casualty_total += ss.cap_casualty_trades
        cap_casualty_cleared_total += ss.cap_casualty_cap_cleared_m
        cap_casualty_log_all.extend(ss.cap_casualty_trade_rows)

        for cc_issue in _validate_cap_casualty_trades(
            league, ss.cap_casualty_trade_rows, season_year=ss.season_year,
        ):
            report.issues.append(cc_issue)

        dup_errors = _validate_waiver_integrity(league)
        for err in dup_errors:
            report.issues.append(ValidationIssue(
                "error", "WAIVER_INTEGRITY", err, season=ss.season_year,
            ))
        buried_missing = _validate_buried_contracts(league)
        if buried_missing:
            report.issues.append(ValidationIssue(
                "error", "BURIED_CONTRACT_LOST",
                f"{len(buried_missing)} buried players missing contracts",
                season=ss.season_year,
            ))

        for pid in _scan_fake_elc_contracts(league):
            report.issues.append(ValidationIssue(
                "error", "FAKE_ELC", f"Player {pid} has ELC label with non-ELC cap hit", season=ss.season_year,
            ))
        for tid in _scan_contract_slot_violations(league):
            report.issues.append(ValidationIssue(
                "error", "CONTRACT_SLOT_OVER", f"Team {tid} exceeds contract slot limit", season=ss.season_year,
            ))
        gp_exempt = _scan_waiver_gp_exempt_issues(league)
        if len(gp_exempt) >= 25:
            report.issues.append(ValidationIssue(
                "warning", "WAIVER_GP_DATA_MISSING",
                f"{len(gp_exempt)} age 21-22 standard contracts treated waiver-exempt with missing GP",
                season=ss.season_year,
            ))

        for claim in ss.waiver_claims:
            if float(claim.get("overall", 0) or 0) >= 86:
                report.issues.append(ValidationIssue(
                    "warning", "HIGH_OVR_WAIVER_CLAIM",
                    f"Team {claim.get('to_team_id')} claimed {claim.get('player_id')} OVR {claim.get('overall')}",
                    season=ss.season_year,
                ))

        for issue in ss.post_fa_shape_issues:
            code = str(issue.get("code", ""))
            if code in ("unmet_primary_need",):
                continue
            report.issues.append(ValidationIssue(
                "warning",
                f"POST_FA_{code.upper()}",
                f"Team {issue.get('team_id')}: {code}",
                season=ss.season_year,
                team_id=str(issue.get("team_id", "")),
            ))

        for oc in ss.over_cap_teams:
            tid = oc["team_id"]
            worst_cap[tid] = min(worst_cap.get(tid, 0.0), float(oc["cap_space_m"]))

        if ss.over_cap_after_compliance:
            for tid in ss.over_cap_after_compliance:
                report.issues.append(ValidationIssue(
                    "error", "OVER_CAP_AFTER_COMPLIANCE",
                    f"Team {tid} still over cap after compliance pass",
                    season=ss.season_year, team_id=tid,
                ))

        if ss.cpu_rfa_stranded:
            for tid in ss.cpu_rfa_stranded:
                report.issues.append(ValidationIssue(
                    "error", "CPU_RFA_STRANDED",
                    f"CPU team {tid} left RFA rights unresolved after RFA pass",
                    season=ss.season_year, team_id=tid,
                ))

        if ss.cap_mismatch_teams:
            for tid in ss.cap_mismatch_teams:
                report.issues.append(ValidationIssue(
                    "error", "CAP_SNAPSHOT_MISMATCH",
                    f"Full vs legacy cap snapshot mismatch for {tid}",
                    season=ss.season_year, team_id=tid,
                ))

        if ss.raw_dollar_contracts:
            report.issues.append(ValidationIssue(
                "error", "RAW_DOLLAR_CONTRACT",
                f"{len(ss.raw_dollar_contracts)} contracts stored in raw dollars",
                season=ss.season_year,
            ))

        if ss.missing_contract_players:
            report.issues.append(ValidationIssue(
                "error", "MISSING_CONTRACT",
                f"{len(ss.missing_contract_players)} active roster players missing contracts",
                season=ss.season_year,
            ))

        if ss.nmc_violations:
            for pid in ss.nmc_violations:
                report.issues.append(ValidationIssue(
                    "error", "NMC_ILLEGAL_BURY",
                    f"NMC player {pid} buried/waived",
                    season=ss.season_year, player_id=pid,
                ))

    end_registry = _collect_player_registry(league)
    for pid, loc in start_registry.items():
        if pid and pid not in end_registry:
            report.players_lost.append(pid)

    if report.players_lost:
        report.issues.append(ValidationIssue(
            "warning", "PLAYER_LOST",
            f"{len(report.players_lost)} players no longer traceable in league",
        ))

    bad, bargains, stars = _collect_league_contract_profiles(league)
    report.biggest_bad_contracts = bad[:15]
    report.best_bargains = bargains[:15]
    report.stars_sample = stars[:12]
    report.depth_sample = [r for r in bad if r["overall"] < 78][:8]

    fa_pool = list(_get(league, "free_agents", None) or [])
    fa_rows = []
    for p in fa_pool:
        fa_rows.append({
            "player_id": _player_id(p),
            "name": str(_get(_get(p, "identity", None), "name", "") or ""),
            "overall": round(_player_ovr(p)),
            "age": int(_get(_get(p, "identity", None), "age", _get(p, "age", 0)) or 0),
        })
    fa_rows.sort(key=lambda r: -r["overall"])
    report.top_unsigned_ufas = fa_rows[:20]
    report.unsigned_ufas_by_band_final = _count_by_ovr_band(fa_pool)
    report.cpu_signings_by_band_total = cpu_bands_total
    report.cpu_signings_by_position_total = cpu_pos_total
    report.post_fa_shape_issues_total = post_fa_all[:40]
    report.waiver_exposed_total = waiver_exposed_total
    report.waiver_claimed_total = waiver_claimed_total
    report.waiver_cleared_total = waiver_cleared_total
    report.waiver_by_band_total = waiver_band_total
    report.buyouts_total = buyouts_total
    report.buyout_cap_cleared_total_m = round(buyout_cap_total, 3)
    report.cap_casualty_trades_total = cap_casualty_total
    report.cap_casualty_cap_cleared_total_m = round(cap_casualty_cleared_total, 3)
    report.cap_casualty_trade_log = cap_casualty_log_all[:80]

    if waiver_exposed_total == 0 and seasons >= 3:
        report.issues.append(ValidationIssue(
            "warning", "WAIVER_PIPELINE_INACTIVE",
            "No players exposed to waivers during audit — waiver pass may not be triggering",
            season=report.end_year,
        ))

    high_band_unsigned = report.unsigned_ufas_by_band_final.get("85-89", 0) + report.unsigned_ufas_by_band_final.get("90+", 0)
    if high_band_unsigned >= 15:
        report.issues.append(ValidationIssue(
            "warning", "UFA_QUALITY_POOL",
            f"{high_band_unsigned} unsigned UFAs at 85+ OVR — CPU demand may be too weak",
            season=report.end_year,
        ))

    under70_signings = cpu_bands_total.get("under-70", 0)
    if under70_signings >= max(8, len(all_cpu_signings) // 5):
        report.issues.append(ValidationIssue(
            "warning", "CPU_LOW_OVR_SIGNINGS",
            f"CPU signed {under70_signings} sub-70 OVR UFAs — check need-based filters",
            season=report.end_year,
        ))

    report.cpu_weird_signings = _analyze_cpu_signings(all_cpu_signings, league)[:25]
    for w in report.cpu_weird_signings:
        report.issues.append(ValidationIssue(
            "warning", "CPU_WEIRD_SIGNING",
            f"{w.get('player_id')} -> {w.get('team_id')}: {','.join(w.get('reasons', []))}",
        ))

    report.worst_cap_teams = [
        {"team_id": tid, "worst_cap_space_m": round(v, 3)}
        for tid, v in sorted(worst_cap.items(), key=lambda x: x[1])[:10]
    ]

    final_snaps = report.season_snapshots[-1] if report.season_snapshots else None
    if final_snaps:
        end_caps = []
        for team in _get(session.sim.league, "teams", None) or []:
            snap = get_team_cap_snapshot_full(team, session.sim.league, session.sim, season_year=final_snaps.season_year)
            end_caps.append(snap["upper_limit_m"])
        report.summary["final_cap_upper_m"] = round(sum(end_caps) / max(1, len(end_caps)), 2)

    report.summary.update({
        "total_issues": len(report.issues),
        "errors": sum(1 for i in report.issues if i.severity == "error"),
        "warnings": sum(1 for i in report.issues if i.severity == "warning"),
        "bad_contracts_at_bootstrap": report.summary.get("bad_contracts_at_bootstrap", 0),
        "bootstrap_cap_upper_m": report.summary.get("bootstrap_cap_upper_m"),
        "over_cap_incidents": sum(len(s.over_cap_teams) for s in report.season_snapshots),
        "compliance_failures": sum(len(s.over_cap_after_compliance) for s in report.season_snapshots),
        "cpu_signings_total": len(all_cpu_signings),
        "cpu_weird_signings": len(report.cpu_weird_signings),
        "cpu_signings_by_band": cpu_bands_total,
        "cpu_signings_by_position": cpu_pos_total,
        "unsigned_ufas_by_band_final": report.unsigned_ufas_by_band_final,
        "post_fa_shape_issues": len(post_fa_all),
        "waiver_exposed_total": waiver_exposed_total,
        "waiver_claimed_total": waiver_claimed_total,
        "waiver_cleared_total": waiver_cleared_total,
        "waiver_by_band_total": waiver_band_total,
        "buyouts_total": buyouts_total,
        "buyout_cap_cleared_total_m": round(buyout_cap_total, 3),
        "cpu_rfa_re_signed_total": cpu_rfa_re_signed_total,
        "cpu_rfa_walked_total": cpu_rfa_walked_total,
        "cap_casualty_trades_total": cap_casualty_total,
        "cap_casualty_cap_cleared_total_m": round(cap_casualty_cleared_total, 3),
        "buried_contracts_total": buried_contracts_total,
        "rfa_rights_final": final_snaps.rfa_rights_total if final_snaps else 0,
        "unsigned_ufas_final": final_snaps.unsigned_ufa_count if final_snaps else 0,
        "elc_count_final": final_snaps.elc_count if final_snaps else 0,
        "pipeline_elc_count_final": final_snaps.pipeline_elc_count if final_snaps else 0,
        "unsigned_prospects_final": final_snaps.unsigned_prospect_count if final_snaps else 0,
        "prospect_promotions_total": sum(s.prospect_promotions for s in report.season_snapshots),
        "elc_signed_total": sum(s.elc_signed for s in report.season_snapshots),
        "buyout_dead_cap_final_m": round(final_snaps.buyout_dead_cap_m, 3) if final_snaps else 0,
        "buried_cap_final_m": round(final_snaps.buried_cap_total_m, 3) if final_snaps else 0,
        "teams_at_slot_limit_final": len(final_snaps.teams_at_slot_limit) if final_snaps else 0,
        "positional_imbalance_teams": len(final_snaps.positional_imbalance) if final_snaps else 0,
        "players_lost": len(report.players_lost),
        "bad_contracts_found": len(bad),
        "bargains_found": len(bargains),
    })

    summary = report.summary
    if summary.get("pipeline_elc_count_final", 0) == 0 and summary.get("prospect_promotions_total", 0) > 0:
        report.issues.append(ValidationIssue(
            "warning", "ELC_PIPELINE_EMPTY",
            "Prospects were promoted but no pipeline ELC contracts exist — auto ELC hook may be broken",
            season=report.end_year,
        ))
    elif summary.get("pipeline_elc_count_final", 0) == 0 and summary.get("prospect_promotions_total", 0) == 0:
        report.issues.append(ValidationIssue(
            "warning", "ELC_PIPELINE_EMPTY",
            "No pipeline ELC contracts after audit — prospect promotion produced zero signed ELCs",
            season=report.end_year,
        ))
    report.summary["warnings"] = sum(1 for i in report.issues if i.severity == "warning")
    report.summary["total_issues"] = len(report.issues)

    return report


def format_audit_report(report: AuditReport) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("CONTRACT ECONOMY STRESS AUDIT")
    lines.append("=" * 72)
    lines.append(f"Seed: {report.seed}  |  Seasons: {report.seasons_simulated}  |  Years: {report.start_year}-{report.end_year}")
    lines.append("")
    s = report.summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Errors: {s.get('errors', 0)}  |  Warnings: {s.get('warnings', 0)}")
    lines.append(f"  Over-cap incidents (pre-compliance): {s.get('over_cap_incidents', 0)}")
    lines.append(f"  Compliance failures (still over cap): {s.get('compliance_failures', 0)}")
    lines.append(f"  CPU signings: {s.get('cpu_signings_total', 0)}  |  Weird: {s.get('cpu_weird_signings', 0)}")
    lines.append(f"  RFA rights (final): {s.get('rfa_rights_final', 0)}")
    lines.append(f"  Unsigned UFAs: {s.get('unsigned_ufas_final', 0)}")
    lines.append(f"  ELC contracts: {s.get('elc_count_final', 0)}  |  Pipeline ELCs: {s.get('pipeline_elc_count_final', 0)}  |  Unsigned prospects: {s.get('unsigned_prospects_final', 0)}")
    lines.append(f"  Prospect promotions: {s.get('prospect_promotions_total', 0)}  |  ELC signed (pass): {s.get('elc_signed_total', 0)}")
    lines.append(f"  Buyout dead cap: ${s.get('buyout_dead_cap_final_m', 0):.1f}M  |  Buried cap: ${s.get('buried_cap_final_m', 0):.1f}M")
    lines.append(f"  Teams at slot limit: {s.get('teams_at_slot_limit_final', 0)}")
    lines.append(f"  Positional imbalance flags: {s.get('positional_imbalance_teams', 0)}")
    lines.append(f"  Players lost from system: {s.get('players_lost', 0)}")
    lines.append(f"  Bad contracts: {s.get('bad_contracts_found', 0)}  |  Bargains: {s.get('bargains_found', 0)}")
    if s.get("bad_contracts_at_bootstrap") is not None:
        lines.append(f"  Bad contracts at bootstrap: {s.get('bad_contracts_at_bootstrap', 0)}")
    if s.get("bootstrap_cap_upper_m"):
        lines.append(f"  Cap upper: ${s.get('bootstrap_cap_upper_m', 0):.1f}M -> ${s.get('final_cap_upper_m', s.get('bootstrap_cap_upper_m', 0)):.1f}M")
    lines.append("")

    cpu_bands = s.get("cpu_signings_by_band") or report.cpu_signings_by_band_total
    if cpu_bands:
        lines.append("CPU UFA SIGNINGS BY OVR BAND")
        for band in ("90+", "85-89", "80-84", "75-79", "70-74", "under-70"):
            lines.append(f"  {band}: {cpu_bands.get(band, 0)}")
        lines.append("")

    ufa_bands = s.get("unsigned_ufas_by_band_final") or report.unsigned_ufas_by_band_final
    if ufa_bands:
        lines.append("UNSIGNED UFAs BY OVR BAND (final)")
        for band in ("90+", "85-89", "80-84", "75-79", "70-74", "under-70"):
            lines.append(f"  {band}: {ufa_bands.get(band, 0)}")
        lines.append("")

    cpu_pos = s.get("cpu_signings_by_position") or report.cpu_signings_by_position_total
    if cpu_pos:
        lines.append("CPU SIGNINGS BY POSITION")
        for pos in ("C", "LW", "RW", "LD", "RD", "G"):
            if cpu_pos.get(pos, 0):
                lines.append(f"  {pos}: {cpu_pos.get(pos, 0)}")
        lines.append("")

    if s.get("waiver_exposed_total", 0) or s.get("waiver_claimed_total", 0):
        lines.append("WAIVER PIPELINE")
        lines.append(f"  Exposed: {s.get('waiver_exposed_total', 0)}  |  Claimed: {s.get('waiver_claimed_total', 0)}  |  Cleared: {s.get('waiver_cleared_total', 0)}")
        wb = s.get("waiver_by_band_total") or report.waiver_by_band_total
        if wb:
            lines.append("  Exposed by band: " + ", ".join(f"{k}={v}" for k, v in wb.items() if v))
        lines.append(f"  Buyouts: {s.get('buyouts_total', 0)}  |  Cap cleared via buyout: ${s.get('buyout_cap_cleared_total_m', 0):.1f}M")
        lines.append(
            f"  Cap casualty trades: {s.get('cap_casualty_trades_total', 0)}  |  "
            f"Cap cleared: ${s.get('cap_casualty_cap_cleared_total_m', 0):.1f}M"
        )
        lines.append(f"  Buried contracts (season-end total): {s.get('buried_contracts_total', 0)}")
        lines.append("")

    if report.worst_cap_teams:
        lines.append("WORST CAP TEAMS (lowest cap space seen)")
        for row in report.worst_cap_teams[:8]:
            lines.append(f"  {row['team_id']}: {row['worst_cap_space_m']:.2f}M")
        lines.append("")

    if report.biggest_bad_contracts:
        lines.append("BIGGEST BAD CONTRACTS")
        for r in report.biggest_bad_contracts[:8]:
            lines.append(f"  {r['name']} ({r['team_id']}) OVR {r['overall']} @ ${r['aav_m']:.1f}M fair ${r['fair_aav_m']:.1f}M score {r['bad_score']:.2f}")
        lines.append("")

    if report.best_bargains:
        lines.append("BEST BARGAINS")
        for r in report.best_bargains[:6]:
            lines.append(f"  {r['name']} ({r['team_id']}) OVR {r['overall']} @ ${r['aav_m']:.1f}M")
        lines.append("")

    if report.stars_sample:
        lines.append("STAR SALARIES (OVR 88+)")
        for r in report.stars_sample[:8]:
            lines.append(f"  {r['name']} OVR {r['overall']} @ ${r['aav_m']:.1f}M (fair ${r['fair_aav_m']:.1f}M)")
        lines.append("")

    if report.top_unsigned_ufas:
        lines.append("TOP UNSIGNED UFAs")
        for r in report.top_unsigned_ufas[:8]:
            lines.append(f"  {r['name']} OVR {r['overall']} age {r['age']}")
        lines.append("")

    if report.cpu_weird_signings:
        lines.append("CPU WEIRD SIGNINGS")
        for w in report.cpu_weird_signings[:10]:
            lines.append(f"  {w.get('player_id')} -> {w.get('team_id')} ${w.get('aav_m', 0):.1f}M [{', '.join(w.get('reasons', []))}]")
        lines.append("")

    errors = [i for i in report.issues if i.severity == "error"]
    warnings = [i for i in report.issues if i.severity == "warning"]
    if errors:
        lines.append(f"ERRORS ({len(errors)})")
        for i in errors[:20]:
            lines.append(f"  [{i.code}] {i.message}")
        lines.append("")
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)})")
        for i in warnings[:15]:
            lines.append(f"  [{i.code}] {i.message}")
        lines.append("")

    if s.get("unsigned_prospects_final", 0) > 0:
        lines.append("NOTE: Unsigned ELC-eligible prospects remain on reserve lists (manual sign or await promotion).")
        lines.append("")

    status = "PASS" if s.get("errors", 0) == 0 else "FAIL"
    lines.append(f"AUDIT STATUS: {status}")
    lines.append("=" * 72)
    return "\n".join(lines)


def report_to_dict(report: AuditReport) -> Dict[str, Any]:
    return {
        "seed": report.seed,
        "seasons_simulated": report.seasons_simulated,
        "summary": report.summary,
        "worst_cap_teams": report.worst_cap_teams,
        "biggest_bad_contracts": report.biggest_bad_contracts[:10],
        "best_bargains": report.best_bargains[:10],
        "top_unsigned_ufas": report.top_unsigned_ufas[:10],
        "cpu_weird_signings": report.cpu_weird_signings[:10],
        "players_lost": report.players_lost[:20],
        "issues": [
            {"severity": i.severity, "code": i.code, "message": i.message, "season": i.season}
            for i in report.issues
        ],
    }
