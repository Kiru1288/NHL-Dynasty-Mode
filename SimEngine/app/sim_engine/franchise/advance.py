"""Day advancement and game simulation."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403
from app.sim_engine.franchise.session import FranchiseSession  # noqa: F401
from app.sim_engine.league.standings import StandingsTable  # noqa: F401
from app.sim_engine.franchise.common import (  # noqa: F401
    _clamp,
    _display_team,
    _ensure_session_event_lists,
    _fr_dbg,
    _franchise_log_injury_and_ui,
    world_calendar,
    world_chemistry,
    world_fatigue,
    world_injuries,
    world_momentum,
    world_morale,
)
from app.sim_engine.franchise.schedule import (  # noqa: F401
    _calendar_iso_for_day,
    _sync_nhl_calendar_bounds,
    _team_plays_on_day,
)
from app.sim_engine.franchise.serialization import (  # noqa: F401
    _coerce_final_score,
    _franchise_team_abbrev,
    _game_result_calendar_index,
    _goalie_availability_status,
    _name_str,
    _offense_opportunity_weight,
    _ovr_weight,
    _player_role_usage_mult,
    _pos_str,
    _regular_season_is_truly_complete,
    _remaining_regular_games_count,
    _saved_game_is_final,
    _skaters,
    _validate_final_game_result_payload,
)
from app.sim_engine.franchise.engine_core import _validate_schedule_hard  # noqa: F401
from app.sim_engine.franchise.events import (  # noqa: F401
    _maybe_enqueue_showcase_popups,
    _maybe_enqueue_wjc_loan_decisions,
)
from app.sim_engine.franchise.decisions import (  # noqa: F401
    _advance_blocked_result,
    _auto_resolve_pending_decisions,
    _maybe_enqueue_post_day_decisions,
    _pending_decision_snapshot,
)
from app.sim_engine.franchise.state import _record_storyline  # noqa: F401
from app.sim_engine.franchise.progression import _run_franchise_season_end_progression  # noqa: F401

def _stat_ensure(session: FranchiseSession, p: Any, team_id: str) -> Dict[str, Any]:
    reg = session.player_season_stats
    pid = str(getattr(p, "id", "") or "")
    if not pid:
        return {}
    if pid not in reg:
        reg[pid] = {
            "player_id": pid,
            "name": _name_str(p),
            "team_id": str(team_id),
            "position": _pos_str(p),
            "gp": 0,
            "g": 0,
            "a": 0,
            "pts": 0,
            "sog": 0,
            "pim": 0,
            "hit": 0,
            "blk": 0,
            "toi_sec": 0,
            "ga": 0,
            "w": 0,
            "l": 0,
            "otl": 0,
        }
    row = reg[pid]
    row["name"] = _name_str(p)
    row["position"] = _pos_str(p)
    row["team_id"] = str(team_id)
    return row
def _stat_add(session: FranchiseSession, p: Any, team_id: str, **kwargs: int) -> None:
    row = _stat_ensure(session, p, team_id)
    if not row:
        return
    for k, v in kwargs.items():
        if v:
            row[k] = int(row.get(k, 0)) + int(v)
    row["pts"] = int(row.get("g", 0)) + int(row.get("a", 0))
def _pick_assist(rng: random.Random, skaters: List[Any], scorer: Any) -> Optional[Any]:
    pool = [s for s in skaters if s is not scorer]
    if not pool:
        return None
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.35) for s in pool]
    return rng.choices(pool, weights=w, k=1)[0]
def _scoring_chunk(
    session: FranchiseSession,
    rng: random.Random,
    skaters: List[Any],
    tid: str,
    goals: int,
) -> List[str]:
    if not skaters or goals <= 0:
        return []
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.45) for s in skaters]
    scorers = rng.choices(skaters, weights=w, k=int(goals))
    high: List[str] = []
    for scorer, ng in Counter(scorers).items():
        _stat_add(session, scorer, tid, g=int(ng))
        for _ in range(int(ng)):
            if rng.random() < 0.78:
                ap = _pick_assist(rng, skaters, scorer)
                if ap:
                    _stat_add(session, ap, tid, a=1)
                    if rng.random() < 0.46:
                        ap2 = _pick_assist(rng, [x for x in skaters if x is not scorer and x is not ap], scorer)
                        if ap2:
                            _stat_add(session, ap2, tid, a=1)
        nm = _name_str(scorer)
        high.append(f"{nm} ├ù{ng}" if ng > 1 else nm)
    return high
def _goalie_game(
    session: FranchiseSession,
    rng: random.Random,
    goalies: List[Any],
    tid: str,
    ga: int,
    won: bool,
    otl_loss: bool,
) -> Optional[Dict[str, Any]]:
    if not goalies:
        return None
    w = [_ovr_weight(g) for g in goalies]
    g0 = rng.choices(goalies, weights=w, k=1)[0]
    if won:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), w=1)
    elif otl_loss:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), otl=1)
    else:
        _stat_add(session, g0, tid, gp=1, ga=int(ga), l=1)
    shots_against = max(int(ga) * 3 + rng.randint(18, 34), int(ga) + 12)
    return {
        "player_id": str(getattr(g0, "id", "") or ""),
        "name": _name_str(g0),
        "ga": int(ga),
        "saves": int(shots_against - int(ga)),
        "shots_against": int(shots_against),
    }
def _skater_box_rows(
    session: FranchiseSession,
    rng: random.Random,
    team: Any,
    tid: str,
    team_shots: int,
) -> Dict[str, Dict[str, Any]]:
    """Per-skater game row shells (g/a filled by play-by-play)."""
    rows: Dict[str, Dict[str, Any]] = {}
    sk = _skaters(team)
    if not sk:
        return rows
    shot_weights = [max(0.001, _offense_opportunity_weight(p) ** 1.35) for p in sk]
    shot_owners = rng.choices(sk, weights=shot_weights, k=max(0, int(team_shots)))
    shot_counts = Counter(shot_owners)

    for p in sk:
        pid = str(getattr(p, "id", "") or "")
        if not pid:
            continue
        sog = int(shot_counts.get(p, 0))
        pos = _pos_str(p).upper()
        usage = _player_role_usage_mult(p)
        if pos == "D":
            toi_min = int(rng.randint(16, 24) * usage)
        else:
            toi_min = int(rng.randint(10, 18) * usage)
        toi_min = max(7, min(28, toi_min))
        toi = int(toi_min * 60 + rng.randint(0, 55))
        pim = int(rng.choices([0, 2, 4, 6], weights=[0.56, 0.28, 0.12, 0.04], k=1)[0])
        if pos == "D":
            hit = int(rng.choices([0, 1, 2, 3, 4], weights=[0.17, 0.28, 0.30, 0.18, 0.07], k=1)[0] * max(0.7, min(1.5, usage)))
            blk = int(rng.choices([0, 1, 2, 3, 4], weights=[0.10, 0.26, 0.33, 0.21, 0.10], k=1)[0] * max(0.8, min(1.5, usage)))
        else:
            hit = int(rng.choices([0, 1, 2, 3], weights=[0.29, 0.37, 0.24, 0.10], k=1)[0] * max(0.7, min(1.4, usage)))
            blk = int(rng.choices([0, 1, 2], weights=[0.64, 0.29, 0.07], k=1)[0] * max(0.7, min(1.3, usage)))
        _stat_add(session, p, tid, gp=1, sog=sog, pim=pim, hit=hit, blk=blk, toi_sec=toi)
        rows[pid] = {
            "player_id": pid,
            "name": _name_str(p),
            "position": _pos_str(p),
            "g": 0,
            "a": 0,
            "sog": sog,
            "pim": pim,
            "hit": hit,
            "blk": blk,
            "toi_sec": toi,
        }
    return rows
def _goals_play_by_play(
    session: FranchiseSession,
    rng: random.Random,
    skaters: List[Any],
    tid: str,
    goals: int,
    rows_by_pid: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Register goals + assists on season stats; return summary strings and scoring events."""
    events: List[Dict[str, Any]] = []
    high: List[str] = []
    if not skaters or goals <= 0:
        return high, events
    w = [max(0.001, _offense_opportunity_weight(s) ** 1.5) for s in skaters]
    for _ in range(int(goals)):
        scorer = rng.choices(skaters, weights=w, k=1)[0]
        spid = str(getattr(scorer, "id", "") or "")
        _stat_add(session, scorer, tid, g=1)
        if spid in rows_by_pid:
            rows_by_pid[spid]["g"] = int(rows_by_pid[spid].get("g", 0)) + 1
        assist_names: List[str] = []
        if rng.random() < 0.78:
            ap = _pick_assist(rng, skaters, scorer)
            if ap:
                _stat_add(session, ap, tid, a=1)
                apid = str(getattr(ap, "id", "") or "")
                if apid in rows_by_pid:
                    rows_by_pid[apid]["a"] = int(rows_by_pid[apid].get("a", 0)) + 1
                assist_names.append(_name_str(ap))
                if rng.random() < 0.46:
                    pool2 = [x for x in skaters if x is not scorer and x is not ap]
                    if pool2:
                        ap2 = _pick_assist(rng, pool2, scorer)
                        if ap2:
                            _stat_add(session, ap2, tid, a=1)
                            ap2id = str(getattr(ap2, "id", "") or "")
                            if ap2id in rows_by_pid:
                                rows_by_pid[ap2id]["a"] = int(rows_by_pid[ap2id].get("a", 0)) + 1
                            assist_names.append(_name_str(ap2))
        per = int(rng.choices([1, 2, 3], weights=[0.34, 0.42, 0.24])[0])
        mm = int(rng.randint(0, 19))
        ss = int(rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
        rv = rng.random()
        strength = "EV"
        if rv < 0.23:
            strength = "PP"
        elif rv < 0.29:
            strength = "SH"
        events.append(
            {
                "for_team_id": str(tid),
                "period": per,
                "clock": f"{mm}:{ss:02d}",
                "scorer": _name_str(scorer),
                "scorer_id": spid,
                "assists": assist_names,
                "strength": strength,
            }
        )
        high.append(_name_str(scorer))
    return high, events
def _accumulate_franchise_game_stats(
    session: FranchiseSession,
    *,
    home: Any,
    away: Any,
    hid: str,
    aid: str,
    hg: int,
    ag: int,
    ot: bool,
    calendar_day: int,
    rng: random.Random,
    calendar_iso: str = "",
) -> None:
    """
    Single pipeline: SimEngine accumulates skater/goalie stats from the same _simulate_game outcome.

    Hard rules:
    - game box must be a dict
    - final score in box must match simulated score
    - no tied final
    - no negative score
    - append only a clean completed game box
    """
    sim = session.sim
    gid = uuid.uuid4().hex[:14]

    hg = _coerce_final_score(hg)
    ag = _coerce_final_score(ag)

    if hg == ag:
        raise RuntimeError(
            f"Stat accumulation refused tied final on day {calendar_day}: {hid} {hg}, {aid} {ag}"
        )

    box = sim.accumulate_unified_game_stats(
        rng,
        home,
        away,
        str(hid),
        str(aid),
        int(hg),
        int(ag),
        bool(ot),
        session.player_season_stats,
        build_game_payload=True,
        calendar_day=int(calendar_day),
        calendar_iso=str(calendar_iso or ""),
        game_id=gid,
    )

    if not isinstance(box, dict):
        raise RuntimeError(
            f"Stat accumulation failed on calendar day {calendar_day}: SimEngine returned no game box."
        )

    box_hg = _coerce_final_score(box.get("home_goals", box.get("home_score", hg)))
    box_ag = _coerce_final_score(box.get("away_goals", box.get("away_score", ag)))

    if box_hg != int(hg) or box_ag != int(ag):
        raise RuntimeError(
            f"Stat/game mismatch on day {calendar_day}: sim score {hid} {hg}, {aid} {ag}; "
            f"box score {box_hg}-{box_ag}."
        )

    box.update(
        {
            "game_id": str(box.get("game_id") or gid),
            "id": str(box.get("id") or box.get("game_id") or gid),
            "home_id": str(hid),
            "away_id": str(aid),
            "home_name": _display_team(home),
            "away_name": _display_team(away),
            "home_goals": int(hg),
            "away_goals": int(ag),
            "home_score": int(hg),
            "away_score": int(ag),
            "overtime": bool(ot),
            "ot": bool(ot),
            "day": int(calendar_day),
            "calendar_day": int(calendar_day),
            "iso": str(calendar_iso or box.get("iso") or ""),
            "calendar_iso": str(calendar_iso or box.get("calendar_iso") or box.get("iso") or ""),
            "status": "final",
            "completed": True,
            "is_final": True,
            "simmed": True,
        }
    )

    session.game_results.append(box)

    if len(session.game_results) > 2400:
        session.game_results = session.game_results[-1800:]
def _franchise_enqueue_critical_notice(
    session: FranchiseSession, *, title: str, description: str, source: str
) -> None:
    if any(
        str(d.get("kind") or "") == "franchise_critical_notice" and str((d.get("meta") or {}).get("source") or "") == source
        for d in (session.pending_decisions or [])
    ):
        return
    dec_id = f"dec_{uuid.uuid4().hex[:12]}"
    session.pending_decisions.insert(
        0,
        {
            "id": dec_id,
            "kind": "franchise_critical_notice",
            "priority": "CRITICAL",
            "title": title,
            "description": description,
            "options": [{"id": "ack", "label": "Acknowledge"}],
            "meta": {"source": source},
        },
    )
def _franchise_daily_league_tick(session: FranchiseSession, calendar_idx: int) -> None:
    """Waivers / trades / call-ups (SimEngine helpers) before the day's games ΓÇö mutates league rosters."""
    if int(getattr(session, "_last_socio_tick_idx", -99)) == int(calendar_idx):
        return
    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    st = session.standings
    if not teams or st is None:
        return
    rng = sim.rng
    utid = str(session.user_team_id)
    max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
    news_tmp: List[Dict[str, Any]] = []
    ctr: Dict[str, int] = {"trade_executions": 0, "waiver_claims": 0, "major_injuries": 0}
    try:
        sim._standings_sync_team_metrics(st, teams)
        sim._season_daily_socio_economics(rng, int(calendar_idx), max_d, st, teams, news_tmp, ctr)
    except Exception:
        return
    iso = ""
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= int(calendar_idx) < len(cal):
        iso = str(cal[int(calendar_idx)].get("iso") or "")
    for ev in news_tmp:
        ev2 = dict(ev)
        ev2.setdefault("priority", "MEDIUM")
        ev2["date"] = int(ev2.get("date") or calendar_idx)
        ev2["calendar_iso"] = iso
        _record_storyline(session, ev2)
        if str(ev2.get("type")) == "trade" and utid:
            ft = str(ev2.get("from_team_id") or "")
            tt = str(ev2.get("team") or "")
            if utid in (ft, tt):
                _franchise_enqueue_critical_notice(
                    session,
                    title="League office: trade register",
                    description=str(ev2.get("headline") or "A trade involving your club was processed."),
                    source=f"trade_{calendar_idx}_{ft}_{tt}",
                )
    setattr(session, "_last_socio_tick_idx", int(calendar_idx))
def _franchise_fanout_player_storylines(session: FranchiseSession, calendar_idx: int, day_meta: Dict[str, Any]) -> None:
    from app.sim_engine.franchise.common import _franchise_fanout_player_storylines as _fanout  # noqa: WPS433

    _fanout(session, calendar_idx, day_meta)


def _maybe_roll_storyline_arc(session: FranchiseSession, day_meta: Dict[str, Any], rng: random.Random) -> None:
    """Disabled — data-driven storylines come from storyline_engine + game ledger."""
    return


def _simulate_franchise_slot(session: FranchiseSession, slot: Any) -> Tuple[Optional[str], Optional[str]]:
    """Simulate one scheduled league game. Returns (user_summary_line_or_none, league_line_or_none)."""
    sim = session.sim
    teams = list(sim.league.teams)
    r = sim.rng
    user_tid = str(session.user_team_id)

    # team_by_id is keyed by str(team_id); slots may carry int ids (dataclasses do not enforce types).
    hid = str(getattr(slot, "home_id", "") or "")
    aid = str(getattr(slot, "away_id", "") or "")
    home = session.team_by_id.get(hid)
    away = session.team_by_id.get(aid)
    if home is None or away is None:
        return None, None
    d = int(slot.day)
    cal = getattr(session, "nhl_calendar", None) or []
    cal_iso = ""
    if 0 <= int(d) < len(cal):
        cal_iso = str(cal[int(d)].get("iso") or "")

    h_goal = _goalie_availability_status(home)
    a_goal = _goalie_availability_status(away)
    if int(h_goal["total"]) <= 0 or int(a_goal["total"]) <= 0:
        _fr_dbg(f"goalie availability failure on day {d}: {hid} total={h_goal['total']} {aid} total={a_goal['total']}")
        _franchise_enqueue_critical_notice(
            session,
            title="Roster integrity issue",
            description="A scheduled game has no listed goalie on one side. Resolve roster integrity before advancing.",
            source=f"goalie-missing:{hid}:{aid}:{d}",
        )
        raise RuntimeError("Cannot simulate game without at least one goalie on each roster.")
    if bool(h_goal["forced_injured_start"]) or bool(a_goal["forced_injured_start"]):
        forced_team = hid if bool(h_goal["forced_injured_start"]) else aid
        forced_tm = home if forced_team == hid else away
        forced_name = _display_team(forced_tm)
        _fr_dbg(f"forced injured goalie start: day={d} team={forced_team} ({forced_name})")
        _record_storyline(
            session,
            {
                "type": "injury",
                "priority": "HIGH" if forced_team == user_tid else "MEDIUM",
                "headline": f"{forced_name} emergency goalie start",
                "team_id": forced_team,
                "team": forced_team,
                "calendar_iso": cal_iso,
                "date": int(d),
                "cause": "All healthy goalies unavailable due to injuries.",
                "effects": {"goalie_availability_delta": -1, "stability_delta": -2},
                "effect_summary": "Emergency start required; elevated goals-against volatility.",
            },
        )

    if session.use_world and world_momentum is not None:
        if session.prev_calendar_day is not None and d > session.prev_calendar_day:
            span = float(d - session.prev_calendar_day)
            world_momentum.decay_all_teams(teams, span * 0.06)
        session.prev_calendar_day = d

        for tid, tm in ((hid, home), (aid, away)):
            lg = session.last_game_day.get(tid)
            if lg is not None:
                gap = d - lg - 1
                if gap > 0:
                    world_fatigue.rest_roster(tm, gap, r)
            session.last_game_day[tid] = d

        hb2b = bool(
            session.play_days and world_calendar.is_back_to_back(session.play_days.get(hid, set()), d)
        )
        ab2b = bool(
            session.play_days and world_calendar.is_back_to_back(session.play_days.get(aid, set()), d)
        )

        hm = world_momentum.team_strength_modifier(home)
        am = world_momentum.team_strength_modifier(away)
        hc = world_chemistry.team_strength_modifier(home)
        ac = world_chemistry.team_strength_modifier(away)
        hf = world_fatigue.team_fatigue_strength_factor(home)
        af = world_fatigue.team_fatigue_strength_factor(away)
        hmr = world_morale.team_morale_strength_factor(home)
        amr = world_morale.team_morale_strength_factor(away)

        h_scale = max(0.93, min(1.07, hm * hc * hf * hmr)) * float(sim._roster_injury_depth_penalty(home))
        a_scale = max(0.93, min(1.07, am * ac * af * amr)) * float(sim._roster_injury_depth_penalty(away))

        base_noise = 1.0 + 0.22 * (session.chaos_index - 0.5)
        nh = world_chemistry.chemistry_chaos_dampen(home, base_noise)
        na = world_chemistry.chemistry_chaos_dampen(away, base_noise)
        _, ih = sim._identity_runner_strength_noise_factors(home)
        _, ia = sim._identity_runner_strength_noise_factors(away)
        noise_scale = 0.5 * (nh + na) * (0.5 * (ih + ia))

        world_fatigue.tick_roster_fatigue_for_game(home, r, hb2b, session.schedule, d, hid)
        world_fatigue.tick_roster_fatigue_for_game(away, r, ab2b, session.schedule, d, aid)

        hg, ag, ot = sim._simulate_game(
            r,
            home,
            away,
            session.strength_map,
            home_strength_scale=h_scale,
            away_strength_scale=a_scale,
            noise_scale=noise_scale,
            light_mode=bool(getattr(session, "_light_game_stat_accumulation", False)),
        )

        world_momentum.update_momentum_after_game(home, hg, ag, r)
        world_momentum.update_momentum_after_game(away, ag, hg, r)
        blow = abs(hg - ag) >= 3
        world_chemistry.update_after_game(home, hg > ag, blow, r)
        world_chemistry.update_after_game(away, ag > hg, blow, r)

        for p in getattr(home, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            world_morale.update_after_team_result(
                p,
                hg > ag,
                hg - ag,
                r,
                role_satisfaction_proxy=float(
                    getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                ),
            )
        for p in getattr(away, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            world_morale.update_after_team_result(
                p,
                ag > hg,
                ag - hg,
                r,
                role_satisfaction_proxy=float(
                    getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                ),
            )

        for tm in (home, away):
            for pl in getattr(tm, "roster", None) or []:
                if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                    world_injuries.tick_games_missed(pl)
                if int(getattr(pl, "_world_conduct_games_remaining", 0) or 0) > 0:
                    from app.sim_engine.franchise.storyline_conduct import tick_conduct_games_missed  # noqa: WPS433

                    tick_conduct_games_missed(pl)

        if getattr(session, "injuries_enabled", True):
            for tm in (home, away):
                ev = world_injuries.maybe_injure_roster_subset(
                    tm, r, session.chaos_index, max_checks=8
                )
                tid_inj = next((str(v) for v in (getattr(tm, "team_id", None), getattr(tm, "id", None)) if v is not None), "")
                abbrev = _franchise_team_abbrev(tm)
                slot_user_game = user_tid in (hid, aid)
                for label, tier, games, pid in ev:
                    _franchise_log_injury_and_ui(
                        session,
                        player_id=pid,
                        player_name=label,
                        team_id=tid_inj,
                        team_abbrev=abbrev,
                        tier=str(tier),
                        games=int(games),
                        injury_type=str(tier),
                        calendar_day=int(d),
                        calendar_iso=cal_iso,
                        game_day_injury=bool(slot_user_game and tid_inj.lower() == user_tid.lower()),
                    )
    else:
        _, nh = sim._identity_runner_strength_noise_factors(home)
        _, na = sim._identity_runner_strength_noise_factors(away)
        id_noise = 0.5 * (nh + na)
        h_inj = float(sim._roster_injury_depth_penalty(home))
        a_inj = float(sim._roster_injury_depth_penalty(away))
        hg, ag, ot = sim._simulate_game(
            r,
            home,
            away,
            session.strength_map,
            home_strength_scale=h_inj,
            away_strength_scale=a_inj,
            noise_scale=id_noise,
            light_mode=bool(getattr(session, "_light_game_stat_accumulation", False)),
        )

        if world_injuries is not None:
            for tm in (home, away):
                for pl in getattr(tm, "roster", None) or []:
                    if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                        world_injuries.tick_games_missed(pl)
            if getattr(session, "injuries_enabled", True):
                for tm in (home, away):
                    ev = world_injuries.maybe_injure_roster_subset(
                        tm, r, session.chaos_index, max_checks=8
                    )
                    tid_inj = next((str(v) for v in (getattr(tm, "team_id", None), getattr(tm, "id", None)) if v is not None), "")
                    abbrev = _franchise_team_abbrev(tm)
                    slot_user_game = user_tid in (hid, aid)
                    for label, tier, games, pid in ev:
                        _franchise_log_injury_and_ui(
                            session,
                            player_id=pid,
                            player_name=label,
                            team_id=tid_inj,
                            team_abbrev=abbrev,
                            tier=str(tier),
                            games=int(games),
                            injury_type=str(tier),
                            calendar_day=int(d),
                            calendar_iso=cal_iso,
                            game_day_injury=bool(slot_user_game and tid_inj.lower() == user_tid.lower()),
                        )

        hg, ag = _validate_final_game_result_payload(
        home_id=hid,
        away_id=aid,
        home_goals=hg,
        away_goals=ag,
        calendar_day=d,
    )

    session.standings.record_game(slot.home_id, slot.away_id, hg, ag, overtime=ot)

    _accumulate_franchise_game_stats(
        session,
        home=home,
        away=away,
        hid=hid,
        aid=aid,
        hg=int(hg),
        ag=int(ag),
        ot=bool(ot),
        calendar_day=d,
        rng=r,
        calendar_iso=cal_iso,
    )
    

    hn = (_display_team(home) or "?")[:24]
    an = (_display_team(away) or "?")[:24]
    league_line = f"{hn} {int(hg)}-{int(ag)} {an}{' OT' if ot else ''}"

    user_line: Optional[str] = None
    if hid == user_tid or aid == user_tid:
        opp = away if hid == user_tid else home
        won = (hg > ag) if hid == user_tid else (ag > hg)
        wl = "W" if won else "L"
        gs = f"{hg}-{ag}"
        if ot:
            gs += " OT"
        user_line = f"{wl} vs {_display_team(opp)} ({gs}) ΓÇö calendar day {d}"

    return user_line, league_line
def _simulate_slots_for_day(
    session: FranchiseSession,
    calendar_day: int,
    slots: List[Any],
) -> Tuple[List[str], List[str]]:
    """
    Simulate every scheduled slot for one calendar day.

    After simulation, verify that every slot generated one real completed result.
    This prevents silent partial days where standings/games drift apart.
    """
    lines: List[str] = []
    league_lines: List[str] = []

    expected_keys: set = set()

    for slot in slots or []:
        hid = str(getattr(slot, "home_id", "") or "")
        aid = str(getattr(slot, "away_id", "") or "")

        if hid and aid:
            expected_keys.add((hid, aid))

        ul, ll = _simulate_franchise_slot(session, slot)

        if ul:
            lines.append(ul)

        if ll:
            league_lines.append(ll)

    # Verify result store has a valid final for every scheduled slot.
    saved_for_day = [
        g
        for g in (getattr(session, "game_results", None) or [])
        if isinstance(g, dict) and _game_result_calendar_index(g) == int(calendar_day)
    ]

    completed_keys: set = set()

    for g in saved_for_day:
        if not _saved_game_is_final(g):
            continue

        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")

        try:
            _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=g.get("home_goals", g.get("home_score")),
                away_goals=g.get("away_goals", g.get("away_score")),
                calendar_day=int(calendar_day),
            )
        except ValueError:
            continue

        completed_keys.add((hid, aid))

    missing = sorted(expected_keys - completed_keys)

    if missing:
        raise RuntimeError(
            f"Game result integrity error on calendar day {calendar_day}: "
            f"{len(missing)} scheduled game(s) did not produce a valid final result. "
            f"First missing: {missing[0][0]} vs {missing[0][1]}"
        )

    return lines, league_lines
def _purge_retired_from_extra_pools(session: FranchiseSession, player: Any) -> None:
    league = getattr(session.sim, "league", None)
    if league is None:
        return
    for attr in ("free_agents", "overseas_free_agents"):
        lst = getattr(league, attr, None)
        if not lst:
            continue
        try:
            if player in lst:
                lst.remove(player)
        except Exception:
            pass
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            pls = tm.get("players")
            if isinstance(pls, list) and player in pls:
                try:
                    pls.remove(player)
                except Exception:
                    pass
    for tm in getattr(league, "teams", None) or []:
        for attr in ("ahl_roster", "echl_roster"):
            lst = getattr(tm, attr, None)
            if not lst:
                continue
            try:
                if player in lst:
                    lst.remove(player)
            except Exception:
                pass
def _depth_pool_progression_tick(session: FranchiseSession) -> None:
    """Periodic full progression pass on non-NHL depth (prospects, overseas, FA, minors)."""
    from app.sim_engine.progression import run_player_progression

    league = getattr(session.sim, "league", None)
    if league is None:
        return
    rng = session.sim.rng
    pool: List[Any] = []
    for p in getattr(league, "free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append(p)
    for p in getattr(league, "overseas_free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append(p)
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append(p)
        for p in getattr(tm, "echl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append(p)
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if not getattr(p, "retired", False):
                    pool.append(p)
    if not pool:
        return
    rng.shuffle(pool)
    for p in pool[: min(72, len(pool))]:
        try:
            _, retired = run_player_progression(p, rng)
            if retired:
                setattr(p, "retired", True)
                _purge_retired_from_extra_pools(session, p)
        except Exception:
            pass
def _finalize_regular_calendar_day(
    session: FranchiseSession,
    day_meta: Dict[str, Any],
    user_lines: List[str],
    league_lines: List[str],
    *,
    day_ordinal: int,
) -> None:
    """Timeline, GM prompts, and league-wide off-day development after a calendar slate completes."""
    session.calendar_days_finished = int(getattr(session, "calendar_days_finished", 0) or 0) + 1

    iso = str(day_meta.get("iso") or "")
    ui_phase = str(day_meta.get("ui_phase") or "")
    total_reg_days = int(getattr(session, "nhl_regular_season_last_index", 0) or 0) + 1
    day_label = f"{iso} ┬╖ {ui_phase} ┬╖ league day {int(day_ordinal)} / {total_reg_days}"
    session.timeline.append(day_label)
    if league_lines:
        cap = 10
        bits = league_lines[:cap]
        tail = len(league_lines) - len(bits)
        slate = " ┬╖ ".join(bits)
        if tail > 0:
            slate += f" ΓÇª +{tail} more"
        session.timeline.append(f"League: {slate}")
    for ln in user_lines[:6]:
        session.timeline.append(ln)
    utid = str(session.user_team_id)
    user_tm = session.team_by_id.get(utid) or session.team_by_id.get(session.user_team_id)
    uname = (_display_team(user_tm) or "Your club")[:28]
    if not user_lines:
        if league_lines:
            session.timeline.append(f"{uname}: no game today.")
        else:
            session.timeline.append("League: quiet day (no games on the calendar).")
    if session.standings:
        rr = session.standings.records.get(utid) or session.standings.records.get(session.user_team_id)
        if rr is not None:
            session.timeline.append(
                f"{uname} record: {getattr(rr, 'wins', 0)}-{getattr(rr, 'losses', 0)}-{getattr(rr, 'otl', 0)} "
                f"({getattr(rr, 'points', 0)} pts)"
            )
    if len(session.timeline) > 200:
        session.timeline = session.timeline[-200:]

    just_idx = int(session.calendar_cursor) - 1
    try:
        from app.sim_engine.franchise.storyline_engine import franchise_record_data_storylines  # noqa: WPS433

        franchise_record_data_storylines(session, just_idx, day_meta, rng=session.sim.rng)
    except Exception:
        pass
    try:
        from app.sim_engine.franchise.storyline_engine import franchise_cause_storyline_daily_pass  # noqa: WPS433

        franchise_cause_storyline_daily_pass(session, just_idx, day_meta, rng=session.sim.rng)
    except Exception:
        pass
    _franchise_fanout_player_storylines(session, just_idx, day_meta)
    from app.sim_engine.franchise.common import _franchise_tick_conduct_and_resolve  # noqa: WPS433

    _franchise_tick_conduct_and_resolve(session, just_idx, day_meta)

    _maybe_enqueue_post_day_decisions(session, user_lines)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import tick_extra_league_development

        tick_extra_league_development(session.sim, session.sim.rng)
    except Exception:
        pass
    # Keep prospect league stats moving with the calendar (GP/G/A/P/PPG and
    # stock movement must drift in-season; mirrors backend franchise_sim).
    try:
        from app.sim_engine.generation.prospect_league_scoring import advance_all_development_league_stats
        from app.sim_engine.franchise.state import invalidate_session_payload_caches

        cal_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
        if cal_iso:
            advance_all_development_league_stats(
                session.sim,
                cal_iso,
                season_year=int(getattr(session, "season_calendar_year", 2025) or 2025),
                rng=getattr(session.sim, "rng", None),
            )
            invalidate_session_payload_caches(session, reason="prospect_stats")
    except Exception:
        pass
    if int(session.calendar_days_finished) % 5 == 0:
        _depth_pool_progression_tick(session)

    _maybe_enqueue_wjc_loan_decisions(session, day_meta)
    _maybe_enqueue_showcase_popups(session, day_meta)
    _maybe_roll_storyline_arc(session, day_meta, session.sim.rng)
def _split_preseason_from_regular_if_needed(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    """At first regular-season day, snapshot preseason stats and reset regular-season counters."""
    if str(day_meta.get("segment") or "") != "regular":
        return
    if bool(getattr(session, "_regular_stats_split_done", False)):
        return
    try:
        # Keep preseason snapshots available for UI or later diagnostics.
        session.preseason_standings_snapshot = session.standings
    except Exception:
        pass
    try:
        session.preseason_player_stats_snapshot = dict(getattr(session, "player_season_stats", None) or {})
    except Exception:
        session.preseason_player_stats_snapshot = {}
    try:
        session.preseason_game_results_snapshot = list(getattr(session, "game_results", None) or [])
    except Exception:
        session.preseason_game_results_snapshot = []

    # Start fresh regular-season records.
    try:
        teams = list(getattr(getattr(session, "sim", None), "league", None).teams)
        session.standings = StandingsTable(teams)
    except Exception:
        pass
    session.player_season_stats = {}
    session.game_results = []
    session.timeline.append("REGULAR SEASON: preseason stats archived; regular-season records reset.")
    setattr(session, "_regular_stats_split_done", True)
def _enter_postseason(session: FranchiseSession) -> Dict[str, Any]:
    """Regular season finished — open playoff-ready flow (bracket UI, then postseason)."""
    from app.sim_engine.franchise.offseason import _transition_to_playoff_ready

    return _transition_to_playoff_ready(session)
def advance_franchise_day(session: FranchiseSession) -> Dict[str, Any]:
    """
    Advance exactly one NHL calendar day.

    Sacred rules:
    - Do not auto-resolve user decisions here.
    - Do not silently mutate the schedule here.
    - If a daily tick creates a user-facing blocking decision, stop before games.
    - Validate schedule before simming.
    - Sim games once.
    - Clear only the current day after successful simulation.
    - Move cursor exactly once after successful simulation.
    """
    _ensure_session_event_lists(session)

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="pending_decisions",
            message="Resolve pending decisions before advancing.",
        )

    _sync_nhl_calendar_bounds(session)

    if session.phase == "complete":
        return {
            "status": "complete",
            "mode": "day",
            "message": "Season and playoffs finished. Start a new franchise to continue.",
            "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
            "iso": _calendar_iso_for_day(
                session,
                int(getattr(session, "calendar_cursor", 0) or 0),
            ),
        }

    if session.phase == "regular" and int(session.calendar_cursor) > int(session.nhl_regular_season_last_index):
        if not _regular_season_is_truly_complete(session):
            remaining = _remaining_regular_games_count(session)
            raise RuntimeError(
                f"Regular season boundary reached but {remaining} regular-season game(s) remain unsimulated."
            )

        return _enter_postseason(session)

    cal = getattr(session, "nhl_calendar", None) or []

    if not cal:
        raise RuntimeError("Franchise session missing NHL calendar data.")

    idx = int(session.calendar_cursor)

    if idx < 0 or idx >= len(cal):
        raise RuntimeError(f"Calendar cursor out of range: {idx} / {len(cal)}.")

    day_meta = cal[idx]
    day_ordinal = idx + 1

    _split_preseason_from_regular_if_needed(session, day_meta)

    # 1. Daily league office/storyline/injury/trade/news tick.
    # If this creates decisions, stop BEFORE games are simulated.
    _franchise_daily_league_tick(session, idx)

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="daily_tick_decision",
            message="A league-office alert needs your attention before this calendar date can be simulated.",
        )

    # 2. Off-day injury check for the user's team.
    # This can create an injury popup/decision. If it does, stop before games.
    if (
        world_injuries is not None
        and getattr(session, "injuries_enabled", True)
        and not _team_plays_on_day(
            session.by_day,
            idx,
            str(session.user_team_id),
        )
    ):
        user_team = session.team_by_id.get(str(session.user_team_id))

        if user_team is not None:
            iso_row = str(day_meta.get("iso") or "")
            events = world_injuries.maybe_injure_roster_subset(
                user_team,
                session.sim.rng,
                session.chaos_index,
                max_checks=1,
                low_intensity=True,
            )

            tid_inj = str(
                getattr(user_team, "team_id", None)
                or getattr(user_team, "id", None)
                or ""
            )
            abbrev = _franchise_team_abbrev(user_team)

            for label, tier, games, pid in events:
                _franchise_log_injury_and_ui(
                    session,
                    player_id=pid,
                    player_name=label,
                    team_id=tid_inj,
                    team_abbrev=abbrev,
                    tier=str(tier),
                    games=int(games),
                    injury_type=str(tier),
                    calendar_day=int(idx),
                    calendar_iso=iso_row,
                )

    if getattr(session, "pending_decisions", None):
        return _advance_blocked_result(
            session,
            reason="injury_decision",
            message="An injury decision needs your attention before this calendar date can be simulated.",
        )

    # 3. Hard schedule validation.
    # Do not repair here. Runtime repair makes the calendar lie.
    day_schedule_errors = _validate_schedule_hard(session.by_day, cal, day_filter=idx)

    if day_schedule_errors:
        _fr_dbg(f"schedule hard-validation failed on day {idx}: {day_schedule_errors[0]}")
        raise RuntimeError(
            f"Schedule integrity error at {day_meta.get('iso') or idx}: {day_schedule_errors[0]}"
        )

    slots = list(session.by_day.get(idx, []) or [])

    # 4. User double-booking should already be impossible after group 1 fixes.
    # If it still happens, fail loudly instead of silently shifting the calendar.
    utid = str(session.user_team_id)

    user_slots = [
        sl
        for sl in slots
        if _safe_slot_team_id(sl, "home_id") == utid
        or _safe_slot_team_id(sl, "away_id") == utid
    ]

    if len(user_slots) > 1:
        raise RuntimeError(
            f"Schedule integrity error at {day_meta.get('iso') or idx}: user team has "
            f"{len(user_slots)} games on the same day. Fix schedule generation, not runtime advance."
        )

    # 5. Sim the actual day.
    user_lines, league_lines = _simulate_slots_for_day(session, idx, slots)

    # 6. Only after successful simulation do we clear the slate and move the cursor.
    session.by_day[idx] = []
    session.calendar_cursor = int(session.calendar_cursor) + 1

    _finalize_regular_calendar_day(
        session,
        day_meta,
        user_lines,
        league_lines,
        day_ordinal=day_ordinal,
    )

    return {
        "status": "ok",
        "mode": "day",
        "calendar_index": idx,
        "next_calendar_index": int(session.calendar_cursor),
        "iso": str(day_meta.get("iso") or ""),
        "user_game_summaries": user_lines,
        "league_game_summaries": league_lines,
        "games_simulated": int(len(slots)),
        "pending_decisions": _pending_decision_snapshot(session),
    }
def advance_franchise_one_game(session: FranchiseSession) -> Dict[str, Any]:
    """One real NHL calendar day (same as advance day ΓÇö game-by-game calendar progression removed)."""
    return advance_franchise_day(session)
def advance_franchise_bulk(
    session: FranchiseSession,
    *,
    mode: str = "day",
    count: int = 1,
    auto_resolve_decisions: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple advance steps server-side.

    Manual/day advancement must not auto-resolve decisions.
    Full season/bulk sim may opt into auto-resolve by explicitly passing true.
    """
    raw = (mode or "day").strip().lower()

    if raw == "day":
        eff_mode, eff_count = "days", max(1, int(count))
    else:
        eff_mode, eff_count = raw, max(1, int(count))

    steps: List[Dict[str, Any]] = []
    guard = 0
    max_iter = 6000
    stopped: Optional[str] = None

    while guard < max_iter:
        guard += 1

        if auto_resolve_decisions and getattr(session, "pending_decisions", None):
            _auto_resolve_pending_decisions(session)

        if getattr(session, "pending_decisions", None):
            stopped = "pending_decisions"
            break

        if eff_mode == "days":
            step = advance_franchise_day(session)
        elif eff_mode == "games":
            step = advance_franchise_one_game(session)
        elif eff_mode == "season":
            step = advance_franchise_day(session)
        else:
            step = advance_franchise_day(session)

        steps.append(step)

        st = str(step.get("status") or "")

        if st == "blocked":
            stopped = str(step.get("reason") or "blocked")
            break

        if st != "ok":
            stopped = st
            break

        if eff_mode == "days":
            eff_count -= 1
            if eff_count <= 0:
                stopped = "count"
                break

        elif eff_mode == "games":
            eff_count -= 1
            if eff_count <= 0:
                stopped = "count"
                break

        elif eff_mode == "season":
            if session.phase != "regular":
                stopped = "phase"
                break

        else:
            stopped = "count"
            break

    if guard >= max_iter:
        stopped = "guard_limit"

    last = steps[-1] if steps else {
        "status": "blocked" if getattr(session, "pending_decisions", None) else "noop",
        "reason": "pending_decisions" if getattr(session, "pending_decisions", None) else "noop",
        "pending_decisions": _pending_decision_snapshot(session),
    }

    tail = steps[-20:] if len(steps) > 20 else steps

    return {
        "status": last.get("status", "noop"),
        "bulk": True,
        "steps_completed": len(steps),
        "stopped_reason": stopped,
        "last_step": last,
        "recent_steps": tail,
        "pending_decisions": _pending_decision_snapshot(session),
        "calendar_index": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
    }
