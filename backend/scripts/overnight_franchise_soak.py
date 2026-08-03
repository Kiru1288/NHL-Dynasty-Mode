#!/usr/bin/env python3
"""Overnight franchise soak — deterministic multi-seed full-loop audit.

Writes compact FAIL/WARN/CHECKPOINT/FIX lines to backend/logs/soak/.
Minimize stdout: aggregates + failures only.

Usage:
  python scripts/overnight_franchise_soak.py
  python scripts/overnight_franchise_soak.py --seeds 4 --seasons 2 --budget-hours 6
  python scripts/overnight_franchise_soak.py --quick   # 2 seeds x 1 season
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SIMAPP = ROOT / "SimEngine" / "app"
for p in (str(BACKEND), str(SIMAPP)):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG_DIR = BACKEND / "logs" / "soak"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# One team per division (ATL/MET/CEN/PAC approximations via known franchises)
SEED_TEAMS: List[Tuple[int, str, str]] = [
    (1042, "Buffalo Sabres", "rebuild"),
    (2042, "Toronto Maple Leafs", "contend"),
    (3042, "Chicago Blackhawks", "rebuild"),
    (4042, "Colorado Avalanche", "contend"),
    (5042, "San Jose Sharks", "young"),
    (6042, "Boston Bruins", "old"),
    (7042, "Seattle Kraken", "average"),
    (8042, "Florida Panthers", "capped"),
]


@dataclass
class SoakStats:
    seeds_done: int = 0
    seasons_done: int = 0
    games: int = 0
    save_cycles: int = 0
    fails: int = 0
    warns: int = 0
    workflows: Counter = field(default_factory=Counter)
    fail_codes: Counter = field(default_factory=Counter)
    dist: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    confirmed: List[Dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.time)


class SoakLog:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.path = LOG_DIR / f"soak_{run_id}.log"
        self.summary_path = LOG_DIR / f"soak_{run_id}_summary.json"
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> None:
        parts = [f"{kind}"]
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            else:
                s = str(v).replace("|", "/").replace("\n", " ")
                if len(s) > 180:
                    s = s[:177] + "..."
                parts.append(f"{k}={s}")
        line = "|".join(parts)
        self._fh.write(line + "\n")
        self._fh.flush()
        if kind in ("FAIL", "CHECKPOINT", "FATAL", "FIX"):
            print(line, flush=True)

    def close(self) -> None:
        self._fh.close()


def _pid(p: Any) -> str:
    return str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")


def _tid(t: Any) -> str:
    return str(getattr(t, "team_id", None) or getattr(t, "id", "") or "")


def _ovr(p: Any) -> float:
    try:
        from app.sim_engine.franchise.storyline_conduct import get_effective_ovr_display

        return float(get_effective_ovr_display(p))
    except Exception:
        try:
            o = getattr(p, "overall", None)
            if o is not None:
                return float(o)
            fn = getattr(p, "ovr", None)
            v = float(fn() if callable(fn) else fn)
            return v * 99.0 if v <= 1.5 else v
        except Exception:
            return 0.0


def _pos(p: Any) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    if pos is None:
        pos = getattr(p, "position", None)
    return str(getattr(pos, "value", pos) or "?").upper()


def _finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def fingerprint(session: Any) -> str:
    """Compact critical-state hash for save/reload comparisons."""
    parts: List[str] = []
    parts.append(f"ph={getattr(session, 'phase', '')}")
    parts.append(f"sp={getattr(session, 'season_phase', '')}")
    parts.append(f"os={getattr(session, 'offseason_stage', '')}")
    parts.append(f"cur={getattr(session, 'calendar_cursor', 0)}")
    parts.append(f"sy={getattr(session, 'season_calendar_year', 0)}")
    parts.append(f"gr={len(getattr(session, 'game_results', []) or [])}")
    parts.append(f"pg={len(getattr(session, 'processed_game_ids', None) or getattr(session, '_processed_game_ids', None) or [])}")
    league = getattr(getattr(session, "sim", None), "league", None)
    reg = getattr(league, "draft_pick_registry", None) or getattr(league, "pick_registry", None) or {}
    if isinstance(reg, dict):
        owners = sorted(f"{k}:{v.get('current_owner') or v.get('owner')}" for k, v in list(reg.items())[:80] if isinstance(v, dict))
        parts.append("picks=" + ",".join(owners[:40]))
    roster_sig = []
    for tid, tm in sorted((session.team_by_id or {}).items(), key=lambda x: str(x[0])):
        ids = sorted(_pid(p) for p in (getattr(tm, "roster", None) or []) if _pid(p))
        roster_sig.append(f"{tid}:{len(ids)}:{hashlib.md5(','.join(ids).encode()).hexdigest()[:8]}")
    parts.append("rosters=" + ";".join(roster_sig[:32]))
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def clone_via_pickle(session: Any) -> Any:
    return pickle.loads(pickle.dumps(session, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# Level A invariants
# ---------------------------------------------------------------------------

def check_invariants(session: Any, log: SoakLog, *, seed: int, season: int, tag: str, stats: SoakStats) -> int:
    fails = 0
    date = ""
    try:
        from services.franchise_sim import _calendar_iso_for_day

        date = str(_calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)) or "")
    except Exception:
        date = str(getattr(session, "current_date", "") or "")

    def fail(code: str, **kw: Any) -> None:
        nonlocal fails
        fails += 1
        stats.fails += 1
        stats.fail_codes[code] += 1
        log.write("FAIL", seed=seed, season=season, date=date, stage=tag, system=kw.pop("system", tag), code=code, **kw)

    def warn(code: str, **kw: Any) -> None:
        stats.warns += 1
        log.write("WARN", seed=seed, season=season, date=date, stage=tag, system=kw.pop("system", tag), code=code, **kw)

    teams = list((session.team_by_id or {}).values())
    if len(teams) != 32:
        fail("TEAM_COUNT", system="league", n=len(teams))

    tids = [_tid(t) for t in teams]
    if len(tids) != len(set(tids)):
        fail("DUP_TEAM_ID", system="league")

    seen_nhl: Dict[str, str] = {}
    dup_nhl = 0
    missing_identity = 0
    bad_age = 0
    nan_rating = 0
    for tm in teams:
        tid = _tid(tm)
        roster = list(getattr(tm, "roster", None) or [])
        local_ids = [_pid(p) for p in roster]
        if len([x for x in local_ids if x]) != len(set(x for x in local_ids if x)):
            fail("DUP_ON_ROSTER", system="roster", team=tid)
        for p in roster:
            pid = _pid(p)
            if not pid:
                missing_identity += 1
                continue
            if pid in seen_nhl and seen_nhl[pid] != tid:
                dup_nhl += 1
                if dup_nhl <= 5:
                    log.write(
                        "WARN",
                        seed=seed,
                        season=season,
                        date=date,
                        stage=tag,
                        system="roster",
                        code="PLAYER_MULTI_ASSIGN_EX",
                        player=pid,
                        teams=f"{seen_nhl[pid]}|{tid}",
                    )
            else:
                seen_nhl[pid] = tid
            age = getattr(getattr(p, "identity", None), "age", None)
            if age is None:
                age = getattr(p, "age", None)
            try:
                ai = int(age)
                if ai < 15 or ai > 55:
                    bad_age += 1
            except (TypeError, ValueError):
                bad_age += 1
            o = _ovr(p)
            if not _finite(o) or o < 20 or o > 99:
                nan_rating += 1

    if dup_nhl:
        fail("PLAYER_MULTI_ASSIGN", system="roster", count=dup_nhl)
    if missing_identity:
        fail("MISSING_PLAYER_ID", system="roster", count=missing_identity)
    if bad_age:
        fail("BAD_AGE", system="roster", count=bad_age)
    if nan_rating:
        fail("BAD_OVR", system="roster", count=nan_rating)

    # Draft pick uniqueness
    league = getattr(getattr(session, "sim", None), "league", None)
    reg = getattr(league, "draft_pick_registry", None) or getattr(league, "pick_registry", None) or {}
    if isinstance(reg, dict) and reg:
        try:
            from app.sim_engine.trades.trade_pick_registry import audit_pick_registry_integrity

            audit = audit_pick_registry_integrity(league)
            if audit and not audit.get("ok", True):
                fail("PICK_REGISTRY", system="draft_picks", detail=audit.get("reason") or audit.get("errors"))
        except Exception as e:
            warn("PICK_AUDIT_EXC", system="draft_picks", err=type(e).__name__)

    # Schedule sanity (when calendar present)
    cal = list(getattr(session, "nhl_calendar", None) or [])
    if cal:
        cursor = int(getattr(session, "calendar_cursor", 0) or 0)
        if cursor < 0 or cursor > len(cal) + 5:
            fail("BAD_CURSOR", system="calendar", cursor=cursor, cal_len=len(cal))

    # Standings reconcile sample
    try:
        from services.franchise_sim import _build_standings_rows

        rows = _build_standings_rows(session) or []
        for r in rows[:32]:
            gp = int(r.get("gp") or 0)
            w = int(r.get("w") or r.get("wins") or 0)
            l = int(r.get("l") or r.get("losses") or 0)
            otl = int(r.get("otl") or 0)
            pts = int(r.get("pts") or r.get("points") or 0)
            if gp and w + l + otl > gp + 1:
                fail("STANDINGS_GP", system="standings", team=r.get("team_id"), gp=gp, w=w, l=l, otl=otl)
            if pts and pts != 2 * w + otl and abs(pts - (2 * w + otl)) > 0:
                # Some schemas use ROW differently; soft warn
                if abs(pts - (2 * w + otl)) > 2:
                    warn("STANDINGS_PTS", system="standings", team=r.get("team_id"), pts=pts, expect=2 * w + otl)
    except Exception as e:
        warn("STANDINGS_EXC", system="standings", err=type(e).__name__)

    # Cap finite
    for tm in teams:
        for attr in ("total_cap_hit", "cap_space", "cap_space_m", "salary_cap_m"):
            v = getattr(tm, attr, None)
            if v is not None and not _finite(v):
                fail("CAP_NAN", system="cap", team=_tid(tm), attr=attr)

    # Stats sample — no NaN in season_stats
    nan_stats = 0
    for tm in teams:
        for p in list(getattr(tm, "roster", None) or [])[:30]:
            ss = getattr(p, "season_stats", None) or {}
            if isinstance(ss, dict):
                for k, v in ss.items():
                    if isinstance(v, (int, float)) and not _finite(v):
                        nan_stats += 1
    if nan_stats:
        fail("STAT_NAN", system="stats", count=nan_stats)

    return fails


def collect_scoring_dist(session: Any, stats: SoakStats) -> None:
    pts_f: List[float] = []
    pts_d: List[float] = []
    gp_zero = 0
    active = 0
    pstats = getattr(session, "player_season_stats", None) or {}
    for tm in (session.team_by_id or {}).values():
        for p in getattr(tm, "roster", None) or []:
            if _pos(p) == "G":
                continue
            active += 1
            pid = _pid(p)
            ss = pstats.get(pid) if isinstance(pstats, dict) else None
            if not isinstance(ss, dict):
                ss = getattr(p, "season_stats", None) or {}
            if not isinstance(ss, dict):
                continue
            gp = int(ss.get("gp") or 0)
            pts = int(ss.get("pts") or ((ss.get("g") or 0) + (ss.get("a") or 0)))
            if gp <= 0:
                gp_zero += 1
                continue
            if _pos(p) == "D":
                pts_d.append(float(pts))
            else:
                pts_f.append(float(pts))
    if pts_f:
        stats.dist["fwd_pts_max"].append(max(pts_f))
        stats.dist["fwd_pts_p95"].append(sorted(pts_f)[int(0.95 * (len(pts_f) - 1))])
    if pts_d:
        stats.dist["d_pts_max"].append(max(pts_d))
        stats.dist["d_pts_p95"].append(sorted(pts_d)[int(0.95 * (len(pts_d) - 1))])
    if active:
        stats.dist["zero_gp_share"].append(gp_zero / max(1, active))
    # Hard fail if almost nobody has GP after a completed regular season.
    if active >= 100 and gp_zero >= active * 0.85 and bool(getattr(session, "regular_season_complete", False)):
        stats.fails += 1
        stats.fail_codes["STATS_MISSING"] += 1



def collect_fa_dist(session: Any, stats: SoakStats, log: SoakLog, seed: int, season: int) -> None:
    """Assert elite UFAs only after the market has had real calendar time."""
    try:
        from services.fa_market_engine import tick_free_agency_market

        day = int(getattr(session, "fa_market_day", 0) or 0)
        # Opening-day elites are expected; only fail after ~3 weeks of market.
        if day < 21:
            tick_free_agency_market(session, days=max(1, 21 - day), max_signings_per_day=20)
    except Exception as e:
        log.write("WARN", seed=seed, season=season, system="free_agency", code="FA_TICK_MISS", err=type(e).__name__)
        stats.warns += 1

    league = getattr(getattr(session, "sim", None), "league", None)
    fas = list(getattr(league, "free_agents", None) or [])
    stats.dist["fa_pool"].append(float(len(fas)))
    market_day = int(getattr(session, "fa_market_day", 0) or 0)
    elite_unsigned = []
    for p in fas:
        o = _ovr(p)
        if o >= 88:
            elite_unsigned.append((_pid(p), o))
    elite_unsigned.sort(key=lambda x: -x[1])
    if elite_unsigned:
        stats.dist["elite_ufa"].append(float(len(elite_unsigned)))
        for pid, o in elite_unsigned[:5]:
            # Soft warn early; hard fail only late-market + true superstar.
            severe = o >= 92 and market_day >= 21
            log.write(
                "FAIL" if severe else "WARN",
                seed=seed,
                season=season,
                system="free_agency",
                code="ELITE_UNSIGNED",
                player=pid,
                ovr=o,
                pool=len(fas),
                market_day=market_day,
            )
            if severe:
                stats.fails += 1
                stats.fail_codes["ELITE_UNSIGNED"] += 1


# ---------------------------------------------------------------------------
# Workflow drivers
# ---------------------------------------------------------------------------

def start_seed(seed: int, team: str, log: SoakLog) -> Any:
    from services.franchise_sim import start_franchise
    from services.franchise_store import save_session

    t0 = time.perf_counter()
    session = start_franchise(
        team_query=team,
        head_coach_name="Soak Coach",
        coach_archetype="balanced",
        seed=seed,
        player_universe="generated",
    )
    session._audit_schedule_invariants = True
    try:
        save_session(session)
    except Exception:
        pass
    log.write("CHECKPOINT", seed=seed, season=0, stage="created", ms=int((time.perf_counter() - t0) * 1000), teams=len(session.team_by_id or {}))
    return session


def smoke_state_payload(session: Any, log: SoakLog, seed: int, season: int, phase: str, stats: SoakStats) -> None:
    from services.franchise_sim import build_state_payload_safe

    t0 = time.perf_counter()
    try:
        payload = build_state_payload_safe(session)
        ms = (time.perf_counter() - t0) * 1000
        stats.dist["state_ms"].append(ms)
        stats.workflows[f"state:{phase}"] += 1
        if not isinstance(payload, dict):
            log.write("FAIL", seed=seed, season=season, system="api", code="STATE_NOT_DICT", phase=phase)
            stats.fails += 1
            return
        json.dumps(payload, default=str)
        if ms > 8000:
            log.write("WARN", seed=seed, season=season, system="perf", code="STATE_SLOW", ms=int(ms), phase=phase)
            stats.warns += 1
    except Exception as e:
        log.write("FAIL", seed=seed, season=season, system="api", code="STATE_500", phase=phase, err=f"{type(e).__name__}:{e}")
        stats.fails += 1
        stats.fail_codes["STATE_500"] += 1


def save_reload_cycle(session: Any, log: SoakLog, seed: int, season: int, stats: SoakStats) -> Any:
    import copy

    for attr in list(vars(session).keys()):
        if attr.endswith("_lock"):
            try:
                delattr(session, attr)
            except Exception:
                pass
    fp1 = fingerprint(session)
    try:
        # Prefer deepcopy: some Player.rating proxies mutate during pickle iteration.
        clone = copy.deepcopy(session)
    except Exception:
        try:
            clone = clone_via_pickle(session)
        except Exception as e:
            log.write("FAIL", seed=seed, season=season, system="save", code="CLONE_FAIL", err=f"{type(e).__name__}:{e}")
            stats.fails += 1
            stats.fail_codes["CLONE_FAIL"] += 1
            return session
    fp2 = fingerprint(clone)
    stats.save_cycles += 1
    stats.workflows["save_reload"] += 1
    if fp1 != fp2:
        log.write("FAIL", seed=seed, season=season, system="save", code="FINGERPRINT_MISMATCH", a=fp1, b=fp2)
        stats.fails += 1
        stats.fail_codes["FINGERPRINT_MISMATCH"] += 1
    else:
        log.write("CHECKPOINT", seed=seed, season=season, stage="save_ok", fp=fp1)
    return clone


def advance_regular_season(session: Any, log: SoakLog, seed: int, season: int, stats: SoakStats) -> None:
    from services.franchise_sim import advance_franchise_bulk

    # Chunked season advance with checkpoints
    guard = 0
    while guard < 40:
        guard += 1
        phase = str(getattr(session, "phase", "") or "")
        if phase not in ("regular", "preseason"):
            break
        if phase == "regular" and bool(getattr(session, "regular_season_complete", False)):
            break
        t0 = time.perf_counter()
        result = advance_franchise_bulk(
            session,
            mode="season" if guard > 1 or phase == "regular" else "days",
            count=21 if phase == "preseason" else 1,
            auto_resolve_decisions=True,
        )
        ms = (time.perf_counter() - t0) * 1000
        stats.dist["bulk_ms"].append(ms)
        steps = int(result.get("steps_completed") or 0)
        stats.games += max(0, steps)  # proxy; refined later
        stopped = str(result.get("stopped_reason") or "")
        st = str(result.get("status") or "")
        log.write(
            "CHECKPOINT",
            seed=seed,
            season=season,
            stage=f"bulk_{phase}",
            steps=steps,
            stopped=stopped,
            status=st,
            ms=int(ms),
            cursor=getattr(session, "calendar_cursor", 0),
        )
        if stopped in ("regular_complete", "phase"):
            break
        if st == "blocked" and stopped == "pending_decisions":
            # force clear if auto-resolve failed
            pd = list(getattr(session, "pending_decisions", None) or [])
            if pd:
                log.write("WARN", seed=seed, season=season, system="decisions", code="STUCK_DECISIONS", n=len(pd))
                stats.warns += 1
                session.pending_decisions = []
                continue
            break
        if st not in ("ok",) and stopped not in ("count", "regular_complete", "phase"):
            if stopped == "guard_limit":
                log.write("FAIL", seed=seed, season=season, system="advance", code="GUARD_LIMIT")
                stats.fails += 1
                stats.fail_codes["GUARD_LIMIT"] += 1
            break
        if steps == 0:
            break

    # Count completed games more accurately
    gr = getattr(session, "game_results", None) or []
    if isinstance(gr, list):
        stats.games = max(stats.games, len(gr))


def run_playoffs_and_offseason(session: Any, log: SoakLog, seed: int, season: int, stats: SoakStats) -> None:
    from services.franchise_sim import advance_season_phase
    from services.franchise_offseason import OFFSEASON_STAGES, continue_offseason, generate_next_season
    from services.franchise_entry_draft import (
        complete_entry_draft,
        initialize_entry_draft,
    )

    # Enter playoffs / complete
    for _ in range(6):
        phase = str(getattr(session, "phase", "") or "")
        if phase in ("offseason", "preseason"):
            break
        try:
            if phase == "regular":
                advance_season_phase(session, target="playoff_ready")
            elif phase == "playoff_ready":
                advance_season_phase(session, target="playoffs")
            elif phase in ("playoffs", "post_cup"):
                advance_season_phase(session)
            else:
                advance_season_phase(session)
            stats.workflows[f"phase:{getattr(session, 'phase', '')}"] += 1
        except Exception as e:
            log.write("FAIL", seed=seed, season=season, system="playoffs", code="PHASE_ADV_FAIL", err=f"{type(e).__name__}:{e}", phase=phase)
            stats.fails += 1
            stats.fail_codes["PHASE_ADV_FAIL"] += 1
            break

    smoke_state_payload(session, log, seed, season, str(getattr(session, "phase", "")), stats)
    check_invariants(session, log, seed=seed, season=season, tag="post_cup", stats=stats)
    collect_scoring_dist(session, stats)

    # Walk offseason stages
    for _ in range(len(OFFSEASON_STAGES) + 8):
        phase = str(getattr(session, "phase", "") or "")
        stage = str(getattr(session, "offseason_stage", "") or "")
        if phase == "preseason":
            break
        if phase == "post_cup":
            try:
                continue_offseason(session)
            except Exception as e:
                log.write("FAIL", seed=seed, season=season, system="offseason", code="CONTINUE_FAIL", err=type(e).__name__, stage=stage)
                stats.fails += 1
                break
            continue

        if stage == "draft" or (stage == "" and not getattr(session, "draft_completed", False) and getattr(session, "draft_lottery_done", False)):
            try:
                initialize_entry_draft(session)
                complete_entry_draft(session)
                stats.workflows["draft_complete"] += 1
                log.write("CHECKPOINT", seed=seed, season=season, stage="draft_done", picks=len((getattr(session, "draft_state", {}) or {}).get("draft_results") or []))
            except Exception as e:
                log.write("FAIL", seed=seed, season=season, system="draft", code="DRAFT_FAIL", err=f"{type(e).__name__}:{e}")
                stats.fails += 1
                stats.fail_codes["DRAFT_FAIL"] += 1
                # try to mark complete to unblock
                try:
                    st = getattr(session, "draft_state", None) or {}
                    st["draft_completed"] = True
                    session.draft_state = st
                    session.draft_completed = True
                except Exception:
                    pass

        if stage == "roster_cleanup" or getattr(session, "next_important_event", "") == "generate_next_season":
            try:
                # Force a fresh fill before generate — stale cleanup payloads block rollover.
                from services.franchise_offseason import _run_roster_cleanup

                _run_roster_cleanup(session, force=True)
                generate_next_season(session)
                stats.workflows["next_season"] += 1
                log.write("CHECKPOINT", seed=seed, season=season, stage="next_season", sy=getattr(session, "season_calendar_year", None))
                break
            except Exception as e:
                log.write("FAIL", seed=seed, season=season, system="rollover", code="NEXT_SEASON_FAIL", err=f"{type(e).__name__}:{e}")
                stats.fails += 1
                stats.fail_codes["NEXT_SEASON_FAIL"] += 1
                break

        try:
            before = stage
            continue_offseason(session, from_stage=before or None)
            after = str(getattr(session, "offseason_stage", "") or "")
            stats.workflows[f"offseason:{after or before}"] += 1
            if after == before and phase == "offseason" and after not in ("roster_cleanup",):
                # stalled stage
                log.write("WARN", seed=seed, season=season, system="offseason", code="STAGE_STALL", stage=after)
                stats.warns += 1
                # force hydrate/advance attempt once more then skip
                if after == "free_agency":
                    collect_fa_dist(session, stats, log, seed, season)
                # nudge
                if after == "draft" and getattr(session, "draft_completed", False):
                    session.offseason_stage = "draft_review"
                elif after and after != "roster_cleanup":
                    # move along if ready
                    pass
                else:
                    break
            if after == "free_agency":
                collect_fa_dist(session, stats, log, seed, season)
                smoke_state_payload(session, log, seed, season, "free_agency", stats)
            if after in ("re_sign", "awards", "draft_lottery"):
                smoke_state_payload(session, log, seed, season, after, stats)
                check_invariants(session, log, seed=seed, season=season, tag=after, stats=stats)
        except Exception as e:
            log.write("FAIL", seed=seed, season=season, system="offseason", code="CONTINUE_FAIL", err=f"{type(e).__name__}:{e}", stage=stage)
            stats.fails += 1
            stats.fail_codes["CONTINUE_FAIL"] += 1
            break


def run_one_season(session: Any, log: SoakLog, seed: int, season: int, stats: SoakStats) -> Any:
    smoke_state_payload(session, log, seed, season, "season_start", stats)
    check_invariants(session, log, seed=seed, season=season, tag="season_start", stats=stats)

    # Mid-season save
    advance_regular_season(session, log, seed, season, stats)
    session = save_reload_cycle(session, log, seed, season, stats)

    # If still in regular after pickle swap, finish
    if str(getattr(session, "phase", "")) in ("regular", "preseason") and not bool(getattr(session, "regular_season_complete", False)):
        advance_regular_season(session, log, seed, season, stats)

    check_invariants(session, log, seed=seed, season=season, tag="season_end", stats=stats)
    collect_scoring_dist(session, stats)
    smoke_state_payload(session, log, seed, season, "season_end", stats)
    session._soak_last_games = len(getattr(session, "game_results", []) or [])

    run_playoffs_and_offseason(session, log, seed, season, stats)

    # Ensure we land in preseason/regular for next loop
    ph = str(getattr(session, "phase", "") or "")
    if ph == "preseason":
        try:
            from services.franchise_sim import advance_season_phase

            advance_season_phase(session, target="regular")
        except Exception:
            session.phase = "regular"
            session.season_phase = "regular"

    stats.seasons_done += 1
    # game_results are cleared on generate_next_season — prefer season history / cumulative.
    hist = list(getattr(session, "season_history", None) or [])
    last_hist = hist[-1] if hist else {}
    games_logged = int(last_hist.get("game_results_count") or 0) or int(getattr(session, "_soak_last_games", 0) or 0)
    log.write(
        "CHECKPOINT",
        seed=seed,
        season=season,
        stage="season_complete",
        games=games_logged,
        errors=stats.fails,
        warnings=stats.warns,
        phase=getattr(session, "phase", None),
    )
    return session


def lightweight_progression(log: SoakLog, stats: SoakStats, seasons: int = 10) -> None:
    """Fast yearly-checkpoint progression using audit session + development tick."""
    from services.draft_audit_session import create_fast_audit_session

    seed = 91042
    try:
        session = create_fast_audit_session(seed, team_query="Toronto")
    except Exception as e:
        log.write("FAIL", seed=seed, season=0, system="progression", code="FAST_BOOT_FAIL", err=type(e).__name__)
        stats.fails += 1
        return

    ovr_hist: List[float] = []
    for yr in range(seasons):
        try:
            from app.sim_engine.league_hierarchy_bootstrap import tick_extra_league_development

            tick_extra_league_development(session.sim, session.sim.rng)
        except Exception:
            pass
        # Age the league so veteran decline can offset prospect growth.
        for tm in (session.team_by_id or {}).values():
            for p in list(getattr(tm, "roster", None) or []):
                ident = getattr(p, "identity", None)
                if ident is not None and getattr(ident, "age", None) is not None:
                    try:
                        ident.age = int(ident.age) + 1
                    except Exception:
                        pass
                elif getattr(p, "age", None) is not None:
                    try:
                        p.age = int(p.age) + 1
                    except Exception:
                        pass
        session.development_report_done = False
        session.development_report_completed_season = 0
        session.development_report_payload = {}
        try:
            from app.sim_engine.progression.development import apply_offseason_development

            apply_offseason_development(session.sim.league, session.sim.rng, season_year=2025 + yr)
        except Exception:
            try:
                from services.franchise_offseason import _run_offseason_development

                _run_offseason_development(session)
            except Exception as e:
                if yr == 0:
                    log.write("WARN", seed=seed, system="progression", code="DEV_TICK_MISS", err=type(e).__name__)
                    stats.warns += 1
        # sample mean ovr
        vals = []
        for tm in list((session.team_by_id or {}).values())[:8]:
            for p in list(getattr(tm, "roster", None) or [])[:12]:
                vals.append(_ovr(p))
        if vals:
            mean = sum(vals) / len(vals)
            ovr_hist.append(mean)
            stats.dist["league_mean_ovr"].append(mean)
        session.season_calendar_year = int(getattr(session, "season_calendar_year", 2025) or 2025) + 1
        check_invariants(session, log, seed=seed, season=yr + 1, tag="light_prog", stats=stats)
        log.write("CHECKPOINT", seed=seed, season=yr + 1, stage="light_prog", mean_ovr=(sum(vals) / len(vals) if vals else 0))

    if len(ovr_hist) >= 3:
        drift = ovr_hist[-1] - ovr_hist[0]
        # >6 pts mean OVR over a decade without replacement is systemic inflation.
        if abs(drift) > 6:
            log.write("FAIL", seed=seed, system="progression", code="OVR_DRIFT", drift=drift, start=ovr_hist[0], end=ovr_hist[-1])
            stats.fails += 1
            stats.fail_codes["OVR_DRIFT"] += 1
    stats.workflows["light_progression"] += 1


def run_api_http_smoke(log: SoakLog, stats: SoakStats) -> None:
    """Hit live uvicorn if available."""
    import urllib.request

    base = os.environ.get("SOAK_API", "http://127.0.0.1:8000")
    try:
        with urllib.request.urlopen(base + "/api/franchise/teams", timeout=10) as resp:
            body = resp.read()
            if resp.status != 200:
                log.write("FAIL", system="api", code="TEAMS_STATUS", status=resp.status)
                stats.fails += 1
            else:
                data = json.loads(body)
                n = len(data.get("teams") or data.get("items") or [])
                log.write("CHECKPOINT", system="api", stage="teams", n=n)
                stats.workflows["http_teams"] += 1
    except Exception as e:
        log.write("WARN", system="api", code="HTTP_UNAVAILABLE", err=type(e).__name__)
        stats.warns += 1


def write_summary(log: SoakLog, stats: SoakStats, seeds_planned: int, seasons_planned: int) -> Dict[str, Any]:
    dist_sum = {k: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "mean": (sum(v) / len(v) if v else None)} for k, v in stats.dist.items()}
    summary = {
        "run_id": log.run_id,
        "elapsed_s": time.time() - stats.started,
        "seeds_planned": seeds_planned,
        "seeds_done": stats.seeds_done,
        "seasons_planned": seasons_planned,
        "seasons_done": stats.seasons_done,
        "games_proxy": stats.games,
        "save_cycles": stats.save_cycles,
        "fails": stats.fails,
        "warns": stats.warns,
        "fail_codes": dict(stats.fail_codes),
        "workflows": dict(stats.workflows),
        "distributions": dist_sum,
        "log": str(log.path),
    }
    log.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.write("CHECKPOINT", stage="summary", **{k: summary[k] for k in ("seeds_done", "seasons_done", "fails", "warns")})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seasons", type=int, default=3)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-light-prog", action="store_true")
    ap.add_argument("--skip-http", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.seasons = min(args.seasons, 1)
        args.budget_hours = min(args.budget_hours, 1.5)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = SoakLog(run_id)
    stats = SoakStats()
    log.write("CHECKPOINT", stage="start", seeds=args.seeds, seasons=args.seasons, budget_h=args.budget_hours)

    # Phase 1 map (compact)
    log.write(
        "CHECKPOINT",
        stage="map",
        create="franchise_sim.start_franchise",
        advance="franchise_sim.advance_franchise_bulk",
        playoffs="franchise_offseason.advance_season_phase/complete_playoffs",
        offseason="franchise_offseason.continue_offseason",
        draft="franchise_entry_draft.complete_entry_draft",
        rollover="franchise_offseason.generate_next_season",
        state="franchise_sim.build_franchise_state_payload",
        store="franchise_store.save_session",
    )

    if not args.skip_http:
        run_api_http_smoke(log, stats)
    if not args.skip_light_prog:
        try:
            lightweight_progression(log, stats, seasons=10)
        except Exception as e:
            log.write("FAIL", system="progression", code="LIGHT_FATAL", err=f"{type(e).__name__}:{e}")
            stats.fails += 1

    deadline = time.time() + args.budget_hours * 3600
    matrix = SEED_TEAMS[: max(1, args.seeds)]

    for seed, team, archetype in matrix:
        if time.time() > deadline:
            log.write("CHECKPOINT", stage="budget_stop", seed=seed)
            break
        log.write("CHECKPOINT", seed=seed, stage="seed_start", team=team, arch=archetype)
        try:
            session = start_seed(seed, team, log)
            check_invariants(session, log, seed=seed, season=0, tag="init", stats=stats)
            smoke_state_payload(session, log, seed, 0, "init", stats)
            for season_i in range(1, args.seasons + 1):
                if time.time() > deadline:
                    break
                session = run_one_season(session, log, seed, season_i, stats)
            stats.seeds_done += 1
        except Exception as e:
            log.write("FATAL", seed=seed, err=f"{type(e).__name__}:{e}", tb=traceback.format_exc().splitlines()[-3:])
            stats.fails += 1
            stats.fail_codes["SEED_FATAL"] += 1

    summary = write_summary(log, stats, args.seeds, args.seasons)
    log.close()
    print(json.dumps({k: summary[k] for k in ("run_id", "seeds_done", "seasons_done", "fails", "warns", "fail_codes", "elapsed_s")}, indent=2))
    return 1 if stats.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
