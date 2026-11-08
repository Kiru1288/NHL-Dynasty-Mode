"""One generated season. Prints scoring shape, overall movement, and rights potential."""
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
from app.sim_engine.progression.potential import read_player_potential99  # noqa: E402
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
            print(f"regular season complete after {guard} passes", flush=True)
            return
        if phase in ("playoffs", "offseason", "postseason", "draft"):
            print(f"left the regular season at phase={phase}", flush=True)
            return
        _fill_user_ahl(session)
        _clear_user_trade_demands(session)
        _auto_resolve_pending_decisions(session)
        t0 = time.time()
        step = advance_franchise_bulk(session, mode="season", count=1, auto_resolve_decisions=True)
        print(
            f"pass {guard} {time.time() - t0:.1f}s phase={getattr(session, 'phase', None)} "
            f"stopped={step.get('stopped_reason')} steps={step.get('steps_completed')} "
            f"cursor={getattr(session, 'calendar_cursor', None)} "
            f"games={len(getattr(session, 'game_results', []) or [])}",
            flush=True,
        )
        reason = str(step.get("stopped_reason") or "")
        if reason == "ahl_roster":
            _fill_user_ahl(session)
            continue
        if reason == "trade_demand":
            _clear_user_trade_demands(session)
            continue
        if reason in ("regular_complete", "phase"):
            return
        if int(step.get("steps_completed") or 0) <= 0:
            print("advance made no progress", flush=True)
            return
    print("stopped after the pass guard", flush=True)


def _ovr(player) -> int:
    try:
        return int(round(display_rating(player_current_ovr_01(player))))
    except Exception:
        return 0


def _pot(player) -> int:
    try:
        return int(round(float(read_player_potential99(player))))
    except Exception:
        return 0


def _is_goalie(player) -> bool:
    pos = getattr(player, "position", None) or getattr(getattr(player, "identity", None), "position", "")
    token = str(getattr(pos, "value", pos) or "").upper()
    return token in {"G", "GK", "GOALIE"} or "GOAL" in token


def _name(player) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "name", None) or getattr(player, "name", None) or getattr(player, "id", "?"))


def _stat(session, player) -> dict:
    pid = str(getattr(player, "id", "") or getattr(player, "player_id", "") or "")
    book = getattr(session, "player_season_stats", None) or {}
    row = book.get(pid) if isinstance(book, dict) else None
    return row if isinstance(row, dict) else {}


def _rights(league):
    out = []
    for team in list(getattr(league, "teams", None) or []):
        tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
        for attr in ("prospect_pool", "ahl_roster", "echl_roster"):
            for player in list(getattr(team, attr, None) or []):
                out.append((tid, attr, player))
    return out


def _hist(values, edges):
    labels = []
    for lo, hi in zip(edges, edges[1:]):
        labels.append((f"{lo}-{hi - 1}", sum(lo <= v < hi for v in values)))
    labels.append((f"{edges[-1]}+", sum(v >= edges[-1] for v in values)))
    return labels


def main() -> None:
    print("starting franchise", flush=True)
    t0 = time.time()
    session = start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Audit Coach",
        coach_archetype="balanced",
        seed=4242,
        player_universe="generated",
    )
    print(f"start {time.time() - t0:.1f}s phase={getattr(session, 'phase', None)}", flush=True)
    league = session.sim.league
    changed = rebalance_prospect_ceilings(session)
    print(f"prospect ceilings adjusted {changed}", flush=True)

    print("simulating through the regular season", flush=True)

    nhl = []
    for team in list(getattr(league, "teams", None) or []):
        for player in list(getattr(team, "roster", None) or []):
            if getattr(player, "retired", False) or _is_goalie(player):
                continue
            nhl.append(player)
    start_ovr = {id(p): _ovr(p) for p in nhl}
    rights_before = [_pot(p) for _, _, p in _rights(league)]
    print(
        f"opening NHL skaters {len(nhl)}  95+={sum(v >= 95 for v in start_ovr.values())} "
        f"99={sum(v >= 99 for v in start_ovr.values())}",
        flush=True,
    )
    print(
        f"opening rights potential n={len(rights_before)} "
        f"max={max(rights_before) if rights_before else 0} "
        f"90+={sum(v >= 90 for v in rights_before)} "
        f"bins={_hist(rights_before, [70, 80, 85, 90, 95])}",
        flush=True,
    )

    t0 = time.time()
    _advance_until_season_ends(session)
    print(f"calendar run {time.time() - t0:.1f}s phase={getattr(session, 'phase', None)}", flush=True)

    rows = []
    for team in list(getattr(league, "teams", None) or []):
        tid = str(getattr(team, "abbreviation", None) or getattr(team, "team_id", "") or "")
        for player in list(getattr(team, "roster", None) or []):
            if _is_goalie(player) or getattr(player, "retired", False):
                continue
            st = _stat(session, player)
            g = int(st.get("g") or st.get("goals") or 0)
            a = int(st.get("a") or st.get("assists") or 0)
            gp = int(st.get("gp") or st.get("games_played") or 0)
            rows.append((g + a, g, a, gp, _ovr(player), start_ovr.get(id(p := player), _ovr(p)), _name(player), tid))
    rows.sort(reverse=True)
    goals = sorted((r[1] for r in rows), reverse=True)
    print("\nTOP 15 POINTS", flush=True)
    for pts, g, a, gp, ovr, began, name, tid in rows[:15]:
        print(f"  {name:24} {tid:4} GP={gp:3} G={g:3} A={a:3} PTS={pts:3} OVR={ovr:3} was={began}", flush=True)
    print(
        f"\n60+ goals: {sum(g >= 60 for g in goals)}   50+ goals: {sum(g >= 50 for g in goals)}   "
        f"40+ goals: {sum(g >= 40 for g in goals)}",
        flush=True,
    )
    print(f"goal leader {goals[0] if goals else 0}   10th goal scorer {goals[9] if len(goals) > 9 else 0}", flush=True)
    print("TOP 10 GOALS", flush=True)
    for pts, g, a, gp, ovr, began, name, tid in sorted(rows, key=lambda r: (r[1], r[0]), reverse=True)[:10]:
        print(f"  {name:24} {tid:4} GP={gp:3} G={g:3} A={a:3} PTS={pts:3} OVR={ovr:3}", flush=True)
    top_pts = rows[:10]
    both = sum(1 for pts, g, a, *_ in top_pts if g >= 30 and a >= 30)
    skewed = sum(1 for pts, g, a, *_ in top_pts if g >= 50 and a < 25)
    print(f"top 10 point getters with 30+ goals AND 30+ assists: {both}/10", flush=True)
    print(f"top 10 point getters with 50+ goals and under 25 assists: {skewed}/10", flush=True)
    team_goals = {}
    team_gp = {}
    for pts, g, a, gp, ovr, began, name, tid in rows:
        team_goals[tid] = team_goals.get(tid, 0) + g
        team_gp[tid] = max(team_gp.get(tid, 0), gp)
    gp_ref = max(team_gp.values()) if team_gp else 1
    gpgs = [team_goals[t] / max(1, team_gp[t] or gp_ref) for t in team_goals]
    if gpgs:
        print(f"team goals/game  min={min(gpgs):.2f}  avg={sum(gpgs)/len(gpgs):.2f}  max={max(gpgs):.2f}", flush=True)

    end_ovr = [_ovr(p) for p in nhl]
    deltas = [_ovr(p) - start_ovr.get(id(p), _ovr(p)) for p in nhl]
    print(
        f"\nNHL skaters now 95+={sum(v >= 95 for v in end_ovr)}  99={sum(v >= 99 for v in end_ovr)}",
        flush=True,
    )
    print(
        f"overall change  median={sorted(deltas)[len(deltas)//2]}  "
        f"+4 or more={sum(d >= 4 for d in deltas)}  +6 or more={sum(d >= 6 for d in deltas)}  "
        f"max={max(deltas) if deltas else 0}",
        flush=True,
    )

    rights_after = [_pot(p) for _, _, p in _rights(league)]
    print(
        f"rights potential now n={len(rights_after)} "
        f"max={max(rights_after) if rights_after else 0} "
        f"bins={_hist(rights_after, [70, 80, 85, 90, 95])}",
        flush=True,
    )
    print(f"rights at 90+  {sum(v >= 90 for v in rights_after)} / {len(rights_after)}", flush=True)
    by_team = {}
    for tid, _attr, player in _rights(league):
        by_team.setdefault(tid, []).append(_pot(player))
    crowded = sum(1 for pots in by_team.values() if sum(v >= 90 for v in pots) >= 3)
    print(f"clubs with 3 or more 90+ rights prospects: {crowded}", flush=True)

    reg = getattr(session, "conduct_incidents", None) or {}
    on_ice = [
        inc for inc in (reg.values() if isinstance(reg, dict) else [])
        if isinstance(inc, dict) and str(inc.get("category") or "") == "on_ice"
    ]
    print(f"on-ice suspensions created: {len(on_ice)}", flush=True)


if __name__ == "__main__":
    main()
