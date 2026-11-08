"""One generated season. Overall change has to follow the box score."""
from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.entities.player import display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.progression.development import (  # noqa: E402
    _expected_ppg_for_ovr,
    _is_defense,
    _is_goalie,
)
from services.franchise_sim import (  # noqa: E402
    _auto_resolve_pending_decisions,
    advance_franchise_bulk,
    start_franchise,
)
from services.prospect_potential_boost import rebalance_prospect_ceilings  # noqa: E402
from services.roster_compliance import fill_team_ahl_minimum  # noqa: E402


def _fill_user_ahl(session) -> None:
    league = getattr(getattr(session, "sim", None), "league", None)
    rng = getattr(getattr(session, "sim", None), "rng", None)
    user_id = str(getattr(session, "user_team_id", "") or "")
    if league is None or rng is None:
        return
    for club in list(getattr(league, "teams", None) or []):
        tid = str(getattr(club, "team_id", None) or getattr(club, "id", "") or "")
        if tid == user_id:
            fill_team_ahl_minimum(club, rng)
            return


def _clear_user_trade_demands(session) -> None:
    utid = str(getattr(session, "user_team_id", "") or "")
    book = getattr(session, "trade_demands", None)
    if isinstance(book, dict):
        for row in book.values():
            if isinstance(row, dict) and str(row.get("team_id") or "") == utid and str(row.get("status") or "") == "open":
                row["status"] = "resolved"
    popups = getattr(session, "pending_ui_popups", None)
    if isinstance(popups, list):
        session.pending_ui_popups = [
            p for p in popups
            if not (isinstance(p, dict) and str(p.get("presentation_type") or "").startswith("trade_demand"))
        ]


def _advance_until_season_ends(session) -> None:
    for guard in range(12):
        phase = str(getattr(session, "phase", "") or "")
        if phase == "regular" and bool(getattr(session, "regular_season_complete", False)):
            return
        if phase in ("playoffs", "offseason", "postseason", "draft"):
            return
        _fill_user_ahl(session)
        _clear_user_trade_demands(session)
        _auto_resolve_pending_decisions(session)
        t0 = time.time()
        step = advance_franchise_bulk(session, mode="season", count=1, auto_resolve_decisions=True)
        print(
            f"pass {guard} {time.time() - t0:.1f}s phase={getattr(session, 'phase', None)} "
            f"stopped={step.get('stopped_reason')} games={len(getattr(session, 'game_results', []) or [])}",
            flush=True,
        )
        reason = str(step.get("stopped_reason") or "")
        if reason == "ahl_roster":
            _fill_user_ahl(session)
            continue
        if reason == "trade_demand":
            _clear_user_trade_demands(session)
            continue
        if reason in ("regular_complete", "phase", "playoff_ready"):
            return
        if int(step.get("steps_completed") or 0) <= 0:
            return


def _ovr(player) -> int:
    try:
        return int(round(display_rating(player_current_ovr_01(player))))
    except Exception:
        return 0


def _name(player) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "full_name", None) or getattr(player, "name", "") or "?")


def _stat(session, player) -> dict:
    book = getattr(session, "player_season_stats", None) or {}
    pid = str(getattr(player, "id", "") or "")
    row = book.get(pid)
    return row if isinstance(row, dict) else {}


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def main() -> int:
    t0 = time.time()
    session = start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Audit Coach",
        coach_archetype="balanced",
        seed=4242,
        player_universe="generated",
    )
    rebalance_prospect_ceilings(session)
    print(f"start {time.time() - t0:.1f}s", flush=True)
    _advance_until_season_ends(session)
    print(f"season {time.time() - t0:.1f}s phase={getattr(session, 'phase', None)}", flush=True)

    league = getattr(session.sim, "league", None)
    skaters = []
    goalies = []
    for team in list(getattr(league, "teams", None) or []):
        for player in list(getattr(team, "roster", None) or []):
            if getattr(player, "retired", False):
                continue
            start = getattr(player, "season_start_ovr", None)
            if start is None:
                continue
            end = _ovr(player)
            delta = end - int(start)
            row = _stat(session, player)
            gp = int(row.get("gp") or row.get("games_played") or 0)
            if _is_goalie(player):
                sv = row.get("save_pct", row.get("sv_pct"))
                try:
                    sv_f = float(sv) if sv is not None else None
                except (TypeError, ValueError):
                    sv_f = None
                if sv_f is not None and sv_f > 1.5:
                    sv_f = sv_f / 100.0
                goalies.append((delta, sv_f, gp, int(start), _name(player)))
                continue
            pts = int(row.get("points") or row.get("pts") or 0)
            if pts <= 0:
                pts = int(row.get("g") or row.get("goals") or 0) + int(row.get("a") or row.get("assists") or 0)
            if gp < 20:
                continue
            exp = _expected_ppg_for_ovr(float(start)) * gp
            if _is_defense(player):
                exp *= 0.80
            surplus = pts - exp
            skaters.append((surplus, delta, gp, pts, int(start), end, _name(player)))

    skaters.sort()
    print(f"\nNHL skaters with 20+ games: {len(skaters)}", flush=True)
    if len(skaters) < 80:
        print("not enough skaters to judge the season", flush=True)
        return 1

    def _bucket(lo, hi):
        rows = [d for s, d, *_ in skaters if lo <= s < hi]
        return len(rows), _median(rows)

    print("surplus bucket -> count, median overall change", flush=True)
    for label, lo, hi in (
        ("under -20", -999, -20),
        ("-20 to 0", -20, 0),
        ("0 to 15", 0, 15),
        ("15 to 30", 15, 30),
        ("30 or more", 30, 999),
    ):
        n, med = _bucket(lo, hi)
        print(f"  {label:12} n={n:3}  median delta={med:+.1f}", flush=True)

    bad_up = [row for row in skaters if row[0] < -8 and row[1] >= 3 and row[2] >= 30]
    good_flat = [row for row in skaters if row[0] > 25 and row[1] <= 0 and row[2] >= 40]
    print(f"\ngained +3 or more while 8+ points under the line: {len(bad_up)}", flush=True)
    for surplus, delta, gp, pts, start, end, name in sorted(bad_up, key=lambda r: -r[1])[:12]:
        print(f"  {name:24} {start}->{end} ({delta:+d})  {pts} pts in {gp}  surplus {surplus:+.0f}", flush=True)
    print(f"no overall gain on a season 25+ points over the line: {len(good_flat)}", flush=True)
    for surplus, delta, gp, pts, start, end, name in sorted(good_flat, key=lambda r: -r[0])[:12]:
        print(f"  {name:24} {start}->{end} ({delta:+d})  {pts} pts in {gp}  surplus {surplus:+.0f}", flush=True)

    quarter = max(1, len(skaters) // 4)
    low = [d for _, d, *_ in skaters[:quarter]]
    high = [d for _, d, *_ in skaters[-quarter:]]
    print(
        f"\nbottom quarter surplus median delta {_median(low):+.1f}   "
        f"top quarter {_median(high):+.1f}",
        flush=True,
    )

    played = [g for g in goalies if g[2] >= 20 and g[1] is not None]
    print(f"\ngoalies with 20+ games: {len(played)}", flush=True)
    for delta, sv, gp, start, name in sorted(played, key=lambda r: -(r[1] or 0))[:8]:
        print(f"  {name:24} sv={sv:.3f} GP={gp:3} {start}->{start + delta} ({delta:+d})", flush=True)

    failed = False
    if _median(high) < _median(low) + 0.5:
        print("FAIL the best seasons did not grow more than the worst ones", flush=True)
        failed = True
    if len(bad_up) > 8:
        print("FAIL too many poor seasons still jumped overall", flush=True)
        failed = True
    if len(good_flat) > 12:
        print("FAIL too many strong seasons produced no overall gain", flush=True)
        failed = True
    if not failed:
        print("\nprogression follows the box score", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
