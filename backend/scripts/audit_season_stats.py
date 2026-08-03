"""Start a franchise, advance N days, audit NHL team/player stats (in-process)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
SIM = ROOT / "SimEngine" / "app"
for p in (str(BACKEND), str(SIM), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services import franchise_sim  # noqa: E402


def _i(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    print("starting franchise...")
    t0 = time.time()
    session = franchise_sim.start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Audit Coach",
        coach_archetype="balanced",
        seed=4242,
        player_universe="generated",
        games_per_team=82,
    )
    print("start", f"{time.time() - t0:.1f}s", "session", session.session_id)

    days = 45
    print(f"advancing {days} days...")
    t0 = time.time()
    step = franchise_sim.advance_franchise_bulk(
        session,
        mode="days",
        count=days,
        auto_resolve_decisions=True,
    )
    print("advance", f"{time.time() - t0:.1f}s")
    if isinstance(step, dict):
        print(
            "steps_completed",
            step.get("steps_completed"),
            "status",
            step.get("status"),
            "bulk",
            step.get("bulk"),
        )
    print("phase", getattr(session, "phase", None), "cursor", getattr(session, "calendar_cursor", None))

    print("building stats-central...")
    t0 = time.time()
    sc = franchise_sim.get_cached_stats_central_payload(session)
    print("stats", f"{time.time() - t0:.1f}s")
    integrity = sc.get("integrity") or {}
    print("INTEGRITY", json.dumps(integrity, indent=2)[:4000])

    skaters = sc.get("skaters") or sc.get("players") or []
    goalies = sc.get("goalies") or []
    teams = sc.get("team_analytics") or sc.get("league_team_stats") or sc.get("teams") or []
    print("n_skaters", len(skaters), "n_goalies", len(goalies), "n_teams", len(teams))

    tops = sorted(skaters, key=lambda row: (-_i(row.get("pts")), -_i(row.get("g"))))[:15]
    print("\nTOP SCORERS:")
    for row in tops:
        print(
            f"  {row.get('name')} {row.get('team_id')} GP={_i(row.get('gp'))} "
            f"G={_i(row.get('g'))} A={_i(row.get('a'))} PTS={_i(row.get('pts'))} "
            f"SOG={_i(row.get('sog'))} TOI={_i(row.get('toi_sec'))} "
            f"+/-={_i(row.get('plus_minus'))} gf_on={_f(row.get('gf_on'))} "
            f"ga_on={_f(row.get('ga_on'))} gd={_f(row.get('goal_differential_on_ice'))} "
            f"CF%={row.get('cf_pct')} WAR={row.get('war')}"
        )

    pm_nonzero = sum(1 for row in skaters if _i(row.get("plus_minus")) != 0)
    gf_on_nz = sum(1 for row in skaters if _f(row.get("gf_on")) > 0)
    toi_zero_gp = sum(1 for row in skaters if _i(row.get("gp")) >= 5 and _i(row.get("toi_sec")) <= 0)
    pts_mismatch = sum(1 for row in skaters if _i(row.get("pts")) != _i(row.get("g")) + _i(row.get("a")))
    war_zero = sum(1 for row in skaters if _i(row.get("gp")) >= 10 and abs(_f(row.get("war"))) < 1e-9)
    print("\nSKATER ISSUES:")
    print("  plus_minus nonzero", pm_nonzero, "/", len(skaters))
    print("  gf_on nonzero", gf_on_nz)
    print("  gp>=5 toi=0", toi_zero_gp)
    print("  pts!=g+a", pts_mismatch)
    print("  gp>=10 war~0", war_zero)

    print("\nGOALIE SAMPLE:")
    for row in sorted(goalies, key=lambda x: -_i(x.get("gp")))[:12]:
        print(
            f"  {row.get('name')} GP={_i(row.get('gp'))} "
            f"W={_i(row.get('w'))} L={_i(row.get('l'))} OTL={_i(row.get('otl'))} "
            f"GA={_i(row.get('ga'))} SA={_i(row.get('shots_against'))} "
            f"SV%={row.get('save_pct')} GAA={row.get('gaa')} GSAx={row.get('gsax')}"
        )

    print("\nTEAM GF AUDIT:")
    mismatches = 0
    for t in teams:
        tid = str(t.get("team_id") or "")
        gf = _i(t.get("gf"))
        ga = _i(t.get("ga"))
        gp = _i(t.get("gp"))
        w = _i(t.get("w"))
        l = _i(t.get("l"))
        otl = _i(t.get("otl"))
        sk_g = sum(_i(row.get("g")) for row in skaters if str(row.get("team_id")) == tid)
        sk_a = sum(_i(row.get("a")) for row in skaters if str(row.get("team_id")) == tid)
        g_w = sum(_i(row.get("w")) for row in goalies if str(row.get("team_id")) == tid)
        g_l = sum(_i(row.get("l")) for row in goalies if str(row.get("team_id")) == tid)
        g_otl = sum(_i(row.get("otl")) for row in goalies if str(row.get("team_id")) == tid)
        g_ga = sum(_i(row.get("ga")) for row in goalies if str(row.get("team_id")) == tid)
        flag = ""
        if gf != sk_g:
            flag += f" GF!={sk_g}"
        if ga != g_ga:
            flag += f" GA!={g_ga}"
        if w != g_w or l != g_l or otl != g_otl:
            flag += f" WLOTL goalie={g_w}-{g_l}-{g_otl}"
        if flag:
            mismatches += 1
        print(
            f"  {t.get('team_name') or tid} GP={gp} W-L-OTL={w}-{l}-{otl} "
            f"GF/GA={gf}/{ga} skG={sk_g} skA={sk_a} gWLOTL={g_w}-{g_l}-{g_otl} "
            f"gGA={g_ga}{flag}"
        )
    print("team mismatch count", mismatches)
    print(
        "cf>0",
        sum(1 for row in skaters if _f(row.get("cf")) > 0),
        "xgf>0",
        sum(1 for row in skaters if _f(row.get("xgf")) > 0),
        "hit>0",
        sum(1 for row in skaters if _i(row.get("hit")) > 0),
        "blk>0",
        sum(1 for row in skaters if _i(row.get("blk")) > 0),
    )

    shoot = [r for r in skaters if _i(r.get("sog")) >= 15]
    if shoot:
        sh_list = [_i(r.get("g")) / max(1, _i(r.get("sog"))) for r in shoot]
        print(
            "\nPLAYER SH% (sog>=15): median",
            round(sorted(sh_list)[len(sh_list) // 2], 3),
            "p90",
            round(sorted(sh_list)[int(len(sh_list) * 0.9)], 3),
            "max",
            round(max(sh_list), 3),
        )
    wars = [_f(r.get("war")) for r in skaters if _i(r.get("gp")) >= 8]
    if wars:
        wars_s = sorted(wars)
        print(
            "WAR (gp>=8): median",
            round(wars_s[len(wars_s) // 2], 3),
            "p90",
            round(wars_s[int(len(wars_s) * 0.9)], 3),
            "max",
            round(max(wars), 3),
        )
    cfs = [_f(r.get("cf_pct")) for r in skaters if _i(r.get("gp")) >= 8 and _f(r.get("cf")) > 0]
    if cfs:
        cfs_s = sorted(cfs)
        print(
            "CF% (gp>=8): median",
            round(cfs_s[len(cfs_s) // 2], 3),
            "p10",
            round(cfs_s[int(len(cfs_s) * 0.1)], 3),
            "p90",
            round(cfs_s[int(len(cfs_s) * 0.9)], 3),
        )

    # Raw ledger spot-check (pre-enrichment)
    raw = getattr(session, "player_season_stats", {}) or {}
    sample = list(raw.values())[:5]
    print("\nRAW LEDGER SAMPLE KEYS:", sorted((sample[0] or {}).keys()) if sample else [])
    for row in list(raw.values())[:3]:
        if not isinstance(row, dict):
            continue
        print(
            {
                "name": row.get("name"),
                "pos": row.get("position"),
                "gp": row.get("gp"),
                "g": row.get("g"),
                "a": row.get("a"),
                "pts": row.get("pts"),
                "toi": row.get("toi_sec"),
                "plus_minus": row.get("plus_minus"),
                "gf_on": row.get("gf_on"),
                "ga_on": row.get("ga_on"),
                "cf": row.get("cf"),
                "hit": row.get("hit"),
                "blk": row.get("blk"),
            }
        )


if __name__ == "__main__":
    main()
