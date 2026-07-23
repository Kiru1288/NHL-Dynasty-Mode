"""Schedule generation, validation, and repair."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

def _sync_nhl_calendar_bounds(session: FranchiseSession) -> None:
    """Recompute last preseason+regular calendar index from stored rows (guards stale default 0)."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return
    last = 0
    for i, row in enumerate(cal):
        seg = str(row.get("segment") or "")
        if seg in ("preseason", "regular"):
            last = i
    session.nhl_regular_season_last_index = int(last)
def _calendar_iso_for_day(session: FranchiseSession, day_idx: int) -> str:
    """Best-effort ISO date lookup for a franchise calendar index."""
    cal = getattr(session, "nhl_calendar", None) or []
    try:
        idx = int(day_idx)
    except Exception:
        idx = 0

    if 0 <= idx < len(cal):
        row = cal[idx] or {}
        iso = row.get("iso") or row.get("date") or row.get("calendar_iso")
        if iso:
            return str(iso)

    return ""
def _team_plays_on_day(by_day: Dict[int, List[Any]], day_idx: int, team_id: str) -> bool:
    tid = _safe_id_str(team_id)
    for sl in by_day.get(int(day_idx), []) or []:
        if _safe_slot_team_id(sl, "home_id") == tid or _safe_slot_team_id(sl, "away_id") == tid:
            return True
    return False
def _can_place_user_game(
    by_day: Dict[int, List[Any]],
    day_idx: int,
    user_id: str,
    opp_id: str,
    nhl_cal: List[Dict[str, Any]],
) -> bool:
    """
    Legacy/user-game placement gate.

    This now mirrors the strict schedule rules:
    - only preseason/regular
    - allowed game date
    - no double-booking
    - no 4-in-4 / 5-in-7 created for either team
    """
    idx = int(day_idx)

    if idx < 0 or idx >= len(nhl_cal):
        return False

    row = nhl_cal[idx] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")

    if seg not in ("preseason", "regular"):
        return False

    ag = row.get("allows_games")
    if ag is None:
        ag = row.get("allowsGames")

    if ag is False:
        return False

    if _team_plays_on_day(by_day, idx, user_id):
        return False

    if _team_plays_on_day(by_day, idx, opp_id):
        return False

    # Build a temporary fake slot so the same cadence validator is used.
    probe = GameSlot(
        day=idx,
        home_id=_safe_id_str(user_id),
        away_id=_safe_id_str(opp_id),
        is_playoff=False,
    )

    if _would_create_bad_cadence_for_slot(
        probe,
        idx,
        by_day,
        old_day=None,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True
def _slot_key(slot: Any) -> Tuple[str, str, bool]:
    return (
        _safe_slot_team_id(slot, "home_id"),
        _safe_slot_team_id(slot, "away_id"),
        bool(getattr(slot, "is_playoff", False)),
    )
def _team_ids_for_slot(slot: Any) -> Tuple[str, str]:
    return (_safe_slot_team_id(slot, "home_id"), _safe_slot_team_id(slot, "away_id"))
def _slot_has_team(slot: Any, team_id: str) -> bool:
    tid = _safe_id_str(team_id)
    h, a = _team_ids_for_slot(slot)
    return tid == h or tid == a
def _regular_game_indices(nhl_cal: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for i, day in enumerate(nhl_cal or []):
        seg = str(day.get("segment") or day.get("season_segment") or "")
        if seg != "regular":
            continue
        ag = day.get("allows_games")
        if ag is None:
            ag = day.get("allowsGames")
        if ag is False:
            continue
        out.append(i)
    return out
def _slot_is_regular_season(slot: Any, nhl_cal: List[Dict[str, Any]], day_idx: int) -> bool:
    if bool(getattr(slot, "is_playoff", False)):
        return False
    idx = int(day_idx)
    if idx < 0 or idx >= len(nhl_cal):
        return False
    row = nhl_cal[idx]
    seg = str(row.get("segment") or row.get("season_segment") or "")
    return seg == "regular"
def _build_team_game_days(
    by_day: Dict[int, List[Any]],
    *,
    nhl_cal: Optional[List[Dict[str, Any]]] = None,
    regular_only: bool = False,
) -> Dict[str, List[int]]:
    acc: Dict[str, List[int]] = defaultdict(list)
    for d, slots in (by_day or {}).items():
        di = int(d)
        for sl in slots or []:
            if regular_only and nhl_cal is not None:
                if not _slot_is_regular_season(sl, nhl_cal, di):
                    continue
            h, a = _team_ids_for_slot(sl)
            if h:
                acc[h].append(di)
            if a:
                acc[a].append(di)
    return {tid: sorted(set(ds)) for tid, ds in acc.items()}
def _team_schedule_penalty(days: List[int]) -> float:
    """Deterministic cadence penalty for one team's sorted unique regular-season game days."""
    if not days:
        return 0.0
    if len(days) != len(set(days)):
        return 1e12
    ds = sorted(days)
    ds_set = set(ds)
    pen = 0.0
    lo, hi = ds[0], ds[-1]

    for start in range(lo, hi - 1):
        if all((start + k) in ds_set for k in range(3)):
            pen += 7500.0
    for start in range(lo, hi - 2):
        if all((start + k) in ds_set for k in range(4)):
            pen += 60000.0

    for i in range(len(ds) - 1):
        gap = ds[i + 1] - ds[i]
        if gap == 1:
            pen += 28.0
        elif gap == 2:
            pen += 5.0
        elif gap == 3:
            pass
        elif gap == 4:
            pen += 15.0
        elif gap >= 5:
            pen += 95.0 + float(gap - 5) * 38.0

    for w in range(lo, hi - 5):
        inc = sum(1 for x in ds if w <= x <= w + 6)
        if inc > 4:
            pen += float(inc - 4) * 3200.0
        elif inc < 2:
            if w >= ds[0] + 7 and w + 6 <= ds[-1] - 7:
                pen += 650.0 * float(2 - inc)

    return pen
def _team_has_impossible_cadence(days: List[int]) -> Optional[str]:
    """
    Hard NHL-style cadence validation for one team's regular-season game days.
    This is stricter than the soft penalty model.
    """
    if not days:
        return None

    ds = sorted(int(x) for x in days)

    if len(ds) != len(set(ds)):
        return "duplicate game day"

    lo, hi = ds[0], ds[-1]

    # HARD: no 4 games in 4 nights.
    for start in range(lo, hi + 1):
        games_4 = sum(1 for d in ds if start <= d <= start + 3)
        if games_4 >= 4:
            return f"4 games in 4 days starting day {start}"

    # HARD: no 5 games in 7 nights.
    for start in range(lo, hi + 1):
        games_7 = sum(1 for d in ds if start <= d <= start + 6)
        if games_7 >= 5:
            return f"5 games in 7 days starting day {start}"

    return None
def _validate_league_cadence_hard(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
) -> List[str]:
    """
    League-wide hard cadence validator.
    Separate from slot integrity: this catches the stuff that makes the schedule feel fake.
    """
    errors: List[str] = []

    team_days = _build_team_game_days(
        by_day or {},
        nhl_cal=nhl_cal,
        regular_only=True,
    )

    for team_id, days in sorted(team_days.items()):
        bad = _team_has_impossible_cadence(days)
        if bad:
            errors.append(f"Team {team_id}: {bad}.")

    return errors
def _get_regular_row_allows_games(nhl_cal: List[Dict[str, Any]], day_idx: int) -> bool:
    """
    Defensive calendar read. Some rows use allows_games, some allowsGames.
    """
    idx = int(day_idx)

    if idx < 0 or idx >= len(nhl_cal):
        return False

    row = nhl_cal[idx] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")

    if seg != "regular":
        return False

    allows = row.get("allows_games")
    if allows is None:
        allows = row.get("allowsGames")

    return allows is not False
def _slot_would_be_legal_after_move(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    old_day: int,
    max_games_per_day: int = 18,
) -> bool:
    """
    One strict legality gate used by BOTH smoothing and conflict repair.
    This prevents the repair pass from creating the exact schedule nonsense
    the smoother was trying to remove.
    """
    target_day = int(target_day)
    old_day = int(old_day)

    if target_day == old_day:
        return False

    if not _get_regular_row_allows_games(nhl_cal, target_day):
        return False

    slots_on_target = list(by_day.get(target_day, []) or [])

    if len(slots_on_target) >= int(max_games_per_day):
        return False

    home_id, away_id = _team_ids_for_slot(slot)

    if not home_id or not away_id:
        return False

    if home_id == away_id:
        return False

    if _team_plays_on_day(by_day, target_day, home_id):
        return False

    if _team_plays_on_day(by_day, target_day, away_id):
        return False

    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True
def _league_schedule_penalty(by_day: Dict[int, List[Any]], nhl_cal: List[Dict[str, Any]]) -> float:
    raw: Dict[str, List[int]] = defaultdict(list)
    for d, slots in (by_day or {}).items():
        di = int(d)
        for sl in slots or []:
            if not _slot_is_regular_season(sl, nhl_cal, di):
                continue
            for tid in _team_ids_for_slot(sl):
                if tid != "":
                    raw[tid].append(di)
    total = 0.0
    for _tid, days in raw.items():
        if len(days) != len(set(days)):
            total += 1e12
        total += _team_schedule_penalty(sorted(set(days)))
    return total
def _league_penalty_from_team_days(team_days: Dict[str, List[int]]) -> float:
    """League penalty as sum of per-team penalties (no cross-team terms)."""
    total = 0.0
    for ds in team_days.values():
        if len(ds) != len(set(ds)):
            return 1e12
        total += _team_schedule_penalty(sorted(set(ds)))
    return total
def _pair_penalty_after_day_move(
    team_days: Dict[str, List[int]],
    t_home: str,
    t_away: str,
    old_d: int,
    new_d: int,
) -> float:
    """Penalty contribution for the two teams if a shared game moves old_d -> new_d."""
    acc = 0.0
    for tid in sorted({_safe_id_str(t_home), _safe_id_str(t_away)}):
        if tid == "":
            continue
        base = sorted(set(team_days.get(tid, [])))
        if int(old_d) not in base:
            return 1e12
        nxt = sorted((set(base) - {int(old_d)}) | {int(new_d)})
        if len(nxt) != len(set(nxt)):
            acc += 1e12
        acc += _team_schedule_penalty(nxt)
    return acc
def _can_place_slot_on_day(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    *,
    old_day: int,
    eligible_set: set,
    max_games_per_day: int,
    nhl_cal: List[Dict[str, Any]],
) -> bool:
    target_day = int(target_day)
    old_day = int(old_day)

    if target_day == old_day:
        return False
    if target_day not in eligible_set:
        return False
    if target_day < 0 or target_day >= len(nhl_cal):
        return False

    row = nhl_cal[target_day] or {}
    seg = str(row.get("segment") or row.get("season_segment") or "")
    if seg != "regular":
        return False

    allows_games = row.get("allows_games")
    if allows_games is None:
        allows_games = row.get("allowsGames")
    if allows_games is False:
        return False

    if len(by_day.get(target_day, []) or []) >= int(max_games_per_day):
        return False

    t1, t2 = _team_ids_for_slot(slot)

    if not t1 or not t2:
        return False
    if t1 == t2:
        return False

    if _team_plays_on_day(by_day, target_day, t1):
        return False
    if _team_plays_on_day(by_day, target_day, t2):
        return False

    # HARD cadence rules: do not allow smoothing to create unrealistic stretches.
    if _would_create_bad_cadence_for_slot(
        slot,
        target_day,
        by_day,
        old_day=old_day,
        max_games_in_4=3,
        max_games_in_7=4,
    ):
        return False

    return True
def _clone_slot_at_day(slot: Any, new_day: int) -> Any:
    nd = int(new_day)
    if is_dataclass(slot) and not isinstance(slot, type):
        try:
            return replace(slot, day=nd)
        except TypeError:
            pass
    return GameSlot(
        day=nd,
        home_id=str(getattr(slot, "home_id", "") or ""),
        away_id=str(getattr(slot, "away_id", "") or ""),
        is_playoff=bool(getattr(slot, "is_playoff", False)),
    )
def _find_slot_on_day(by_day: Dict[int, List[Any]], day: int, key: Tuple[str, str, bool]) -> Optional[Any]:
    for sl in by_day.get(int(day), []) or []:
        if _slot_key(sl) == key:
            return sl
    return None
def _move_slot(by_day: Dict[int, List[Any]], slot: Any, old_day: int, new_day: int) -> None:
    """Move one slot between day buckets and keep slot.day in sync."""
    old_day = int(old_day)
    new_day = int(new_day)
    src = by_day.get(old_day, []) or []
    key = _slot_key(slot)
    pick_idx = -1
    for i, sl in enumerate(src):
        if _slot_key(sl) == key:
            pick_idx = i
            break
    if pick_idx < 0:
        return
    removed = src.pop(pick_idx)
    if not src:
        by_day.pop(old_day, None)
    else:
        by_day[old_day] = src
    moved = _clone_slot_at_day(removed, new_day)
    by_day.setdefault(new_day, []).append(moved)
def _schedule_quality_summary(
    by_day: Dict[int, List[Any]],
    nhl_cal: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    td = _build_team_game_days(
        by_day or {},
        nhl_cal=nhl_cal,
        regular_only=bool(nhl_cal),
    )

    max4 = 0
    max7 = 0
    n4 = 0
    n5in7 = 0
    n3 = 0
    ng5 = 0
    worst_gap = 0
    worst_team = ""

    for tid, days in td.items():
        if not days:
            continue

        ds = sorted(days)
        ds_set = set(ds)
        lo, hi = ds[0], ds[-1]

        team_max4 = 0
        team_max7 = 0

        for start in range(lo, hi + 1):
            c4 = sum(1 for x in ds if start <= x <= start + 3)
            c7 = sum(1 for x in ds if start <= x <= start + 6)
            team_max4 = max(team_max4, c4)
            team_max7 = max(team_max7, c7)
            max4 = max(max4, c4)
            max7 = max(max7, c7)

        if team_max4 >= 4:
            n4 += 1

        if team_max7 >= 5:
            n5in7 += 1

        has3 = any(
            all((start + k) in ds_set for k in range(3))
            for start in range(lo, hi - 1)
        )

        if has3 and team_max4 < 4:
            n3 += 1

        for i in range(len(ds) - 1):
            gap = ds[i + 1] - ds[i]

            if gap >= 5:
                ng5 += 1

            if gap > worst_gap:
                worst_gap = gap
                worst_team = str(tid)

    lp = _league_schedule_penalty(by_day or {}, nhl_cal or []) if nhl_cal else 0.0

    return {
        "max_games_in_4_days": int(max4),
        "max_games_in_7_days": int(max7),
        "teams_with_4_in_4": int(n4),
        "teams_with_5_in_7": int(n5in7),
        "teams_with_3_in_3": int(n3),
        "teams_with_5_day_gaps": int(ng5),
        "worst_gap_days": int(worst_gap),
        "worst_gap_team": worst_team,
        "league_penalty": float(lp),
        "hard_cadence_ok": bool(n4 == 0 and n5in7 == 0),
    }
def _would_create_bad_cadence_for_slot(
    slot: Any,
    target_day: int,
    by_day: Dict[int, List[Any]],
    *,
    old_day: Optional[int],
    max_games_in_4: int = 3,
    max_games_in_7: int = 4,
) -> bool:
    """
    Hard rejection used by the schedule smoother.
    Prevents the move from creating 4-in-4 or 5-in-7 type nonsense.
    """
    target_day = int(target_day)
    team_ids = [x for x in _team_ids_for_slot(slot) if x]

    for tid in team_ids:
        days = []
        for d, slots in (by_day or {}).items():
            di = int(d)
            for sl in slots or []:
                if old_day is not None and di == int(old_day) and _slot_key(sl) == _slot_key(slot):
                    continue
                if _slot_has_team(sl, tid):
                    days.append(di)

        days.append(target_day)
        days = sorted(set(days))

        for start in range(target_day - 3, target_day + 1):
            games_4 = sum(1 for d in days if start <= d <= start + 3)
            if games_4 > int(max_games_in_4):
                return True

        for start in range(target_day - 6, target_day + 1):
            games_7 = sum(1 for d in days if start <= d <= start + 6)
            if games_7 > int(max_games_in_7):
                return True

    return False
def _finalize_schedule_after_generation(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    user_id: Optional[str] = None,
) -> Tuple[Dict[int, List[Any]], List[Any], Dict[str, Any]]:
    """
    Startup-only schedule repair/finalizer.

    This is now much stricter:
    - smoother runs harder
    - conflict repair obeys cadence
    - validation includes hard NHL cadence rules
    - schedule diagnostics tell you exactly whether startup schedule is trustworthy
    """
    started_at = time.perf_counter()
    before = _schedule_quality_summary(dict(by_day), nhl_cal)

    raw_by_day: Dict[int, List[Any]] = {
        int(k): list(v or [])
        for k, v in (by_day or {}).items()
    }

    smooth_error: Optional[str] = None
    repair_error: Optional[str] = None
    second_smooth_error: Optional[str] = None

    try:
        _franchise_startup_stage("schedule finalize: smoothing pass 1")
        fixed = _smooth_league_schedule(
            dict(raw_by_day),
            nhl_cal=nhl_cal,
            user_id=user_id,
            max_passes=1,
            max_games_per_day=18,
            max_teams_per_round=10,
            max_inner_steps=10,
            max_suspicious_pairs=12,
        )
        _franchise_startup_stage("schedule finalize: smoothing pass 1 complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _smooth_league_schedule failed; using mapped schedule."
        )
        fixed = dict(raw_by_day)
        smooth_error = str(e)

    try:
        _franchise_startup_stage("schedule finalize: conflict repair")
        fixed = _repair_regular_day_conflicts(fixed, nhl_cal)
        _franchise_startup_stage("schedule finalize: conflict repair complete")
    except Exception as e:
        _startup_log.exception(
            "[franchise start] _repair_regular_day_conflicts failed; continuing with pre-repair slate."
        )
        repair_error = str(e)

    # Skip the second expensive smoothing pass at startup.
    # The first pass + strict repair + hard validation is enough and avoids long UI stalls.

    schedule: List[Any] = []
    aligned: Dict[int, List[Any]] = {}

    for d in sorted(fixed.keys()):
        di = int(d)
        row: List[Any] = []

        for sl in fixed[di]:
            if int(getattr(sl, "day", -1) or -1) != di:
                sl = _clone_slot_at_day(sl, di)

            row.append(sl)
            schedule.append(sl)

        aligned[di] = row

    hard_errors = _validate_schedule_hard(aligned, nhl_cal)
    after = _schedule_quality_summary(aligned, nhl_cal)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    startup_validation_ok = len(hard_errors) == 0

    diagnostics: Dict[str, Any] = {
        "before_smooth": before,
        "after_smooth": after,
        "hard_errors": hard_errors,
        "startup_validation_ok": startup_validation_ok,
        "locked_at_startup": True,
        "smooth_error": smooth_error,
        "repair_error": repair_error,
        "second_smooth_error": second_smooth_error,
        "finalize_elapsed_ms": elapsed_ms,
        "strict_schedule_rules": {
            "max_games_in_4_days": 3,
            "max_games_in_7_days": 4,
            "max_games_per_day": 18,
            "regular_only_conflict_repair": True,
            "blocked_dates_respected": True,
        },
    }

    if hard_errors:
        _startup_log.warning(
            "[franchise start] schedule hard-validation reported %s issue(s); first: %s",
            len(hard_errors),
            hard_errors[0],
        )
    else:
        _startup_log.info(
            "[franchise start] schedule hard-validation passed. Quality=%s elapsed_ms=%s",
            after,
            elapsed_ms,
        )

    return aligned, schedule, diagnostics
def _repair_regular_day_conflicts(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
) -> Dict[int, List[Any]]:
    """
    Resolve regular-season same-day double-bookings by moving extra slots
    to the nearest legal NHL-style date.

    Important:
    - Does not move games onto blocked dates.
    - Does not move games outside regular season.
    - Does not create 4-in-4.
    - Does not create 5-in-7.
    - Does not double-book either team.
    - Prefers nearby dates, not random future dumping.
    """
    if not by_day:
        return by_day

    regular_days = _regular_game_indices(nhl_cal)

    if not regular_days:
        return by_day

    regular_days = sorted(int(x) for x in regular_days)
    moved_total = 0
    failed_moves: List[str] = []

    def _candidate_repair_days(old_day: int) -> List[int]:
        """
        Prefer close dates around the original slot, alternating future/past.
        This keeps the schedule stable and stops everything from being dumped forward.
        """
        old_day = int(old_day)
        out: List[int] = []
        seen: set = set()

        for radius in range(1, 15):
            for cand in (old_day + radius, old_day - radius):
                if cand in seen:
                    continue

                seen.add(cand)

                if cand in regular_days:
                    out.append(cand)

        # If near search fails, use the full regular list by distance.
        if not out:
            out = sorted(
                regular_days,
                key=lambda d: (abs(int(d) - old_day), int(d)),
            )

        return out

    def _pick_target(old_day: int, slot: Any) -> Optional[int]:
        for cand in _candidate_repair_days(old_day):
            if _slot_would_be_legal_after_move(
                slot,
                cand,
                by_day,
                nhl_cal,
                old_day=old_day,
                max_games_per_day=18,
            ):
                return int(cand)

        return None

    for d in sorted(int(x) for x in list(by_day.keys())):
        if d not in regular_days:
            continue

        slots = list(by_day.get(d, []) or [])

        if len(slots) <= 1:
            continue

        team_counts: Dict[str, int] = defaultdict(int)

        for sl in slots:
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            if h:
                team_counts[h] += 1

            if a:
                team_counts[a] += 1

        conflict_slots: List[Any] = []

        for sl in slots:
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            if team_counts.get(h, 0) > 1 or team_counts.get(a, 0) > 1:
                conflict_slots.append(sl)

        for sl in conflict_slots:
            cur_slots = list(by_day.get(d, []) or [])
            h = _safe_slot_team_id(sl, "home_id")
            a = _safe_slot_team_id(sl, "away_id")

            # Recompute after prior moves.
            c_h = sum(1 for x in cur_slots if _slot_has_team(x, h))
            c_a = sum(1 for x in cur_slots if _slot_has_team(x, a))

            if c_h <= 1 and c_a <= 1:
                continue

            target = _pick_target(d, sl)

            if target is None:
                failed_moves.append(f"Day {d}: could not legally move {h} vs {a}.")
                continue

            moved = _clone_slot_at_day(sl, int(target))

            removed = False
            next_src: List[Any] = []

            for existing in cur_slots:
                if not removed and _slot_key(existing) == _slot_key(sl):
                    removed = True
                    continue

                next_src.append(existing)

            if not removed:
                failed_moves.append(f"Day {d}: failed to remove source slot {h} vs {a}.")
                continue

            if next_src:
                by_day[d] = next_src
            else:
                by_day.pop(d, None)

            by_day.setdefault(int(target), []).append(moved)
            moved_total += 1

    if moved_total:
        _fr_dbg(f"schedule conflict repair moved {moved_total} regular-season slots")

    if failed_moves:
        _fr_dbg("schedule conflict repair unresolved: " + " | ".join(failed_moves[:8]))

    return by_day
def _schedule_suspicious_pairs(
    days: List[int],
    by_day: Dict[int, List[Any]],
    team_id: str,
    nhl_cal: List[Dict[str, Any]],
    *,
    max_pairs: int,
) -> List[Tuple[int, Any]]:
    if not days:
        return []
    ds = sorted(days)
    ds_set = set(ds)
    flagged: set = set()
    lo, hi = ds[0], ds[-1]
    for start in range(lo, hi - 2):
        if all((start + k) in ds_set for k in range(4)):
            for x in (start, start + 1, start + 2, start + 3):
                flagged.add(x)
    for start in range(lo, hi - 1):
        if all((start + k) in ds_set for k in range(3)):
            flagged.update([start, start + 1, start + 2])
    for i in range(len(ds) - 1):
        if ds[i + 1] - ds[i] >= 5:
            flagged.add(ds[i])
            flagged.add(ds[i + 1])
    for w in range(lo, hi - 5):
        inc = sum(1 for x in ds if w <= x <= w + 6)
        if inc >= 5:
            for x in ds:
                if w <= x <= w + 6:
                    flagged.add(x)
    pairs: List[Tuple[int, Any]] = []
    for d in sorted(flagged):
        slots = list(by_day.get(d, []) or [])
        slots.sort(key=lambda s: _slot_key(s))
        for sl in slots:
            if _slot_is_regular_season(sl, nhl_cal, d) and _slot_has_team(sl, team_id):
                pairs.append((d, sl))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs
def _candidate_eligible_calendar_days(
    old_day: int,
    eligible_list: List[int],
    *,
    max_index_radius: int,
) -> List[int]:
    """Neighbors along the ordered list of eligible regular game nights (skips league off-days)."""
    if not eligible_list:
        return []
    old_day = int(old_day)
    pos = bisect.bisect_left(eligible_list, old_day)
    if pos < len(eligible_list) and eligible_list[pos] == old_day:
        center = pos
    elif pos > 0 and eligible_list[pos - 1] == old_day:
        center = pos - 1
    else:
        center = min(max(pos - 1, 0), len(eligible_list) - 1)
    seen: set = set()
    out: List[int] = []
    for w in range(1, int(max_index_radius) + 1):
        for j in (center - w, center + w):
            if 0 <= j < len(eligible_list):
                c = int(eligible_list[j])
                if c != old_day and c not in seen:
                    seen.add(c)
                    out.append(c)
    return out
def _smooth_league_schedule(
    by_day: Dict[int, List[Any]],
    *,
    nhl_cal: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    max_passes: int = 2,
    max_games_per_day: int = 22,
    max_teams_per_round: int = 8,
    max_inner_steps: int = 12,
    max_suspicious_pairs: int = 10,
) -> Dict[int, List[Any]]:
    _ = user_id
    if not nhl_cal:
        return {int(k): list(v or []) for k, v in (by_day or {}).items()}
    out: Dict[int, List[Any]] = {int(k): list(v or []) for k, v in (by_day or {}).items()}
    eligible_list = _regular_game_indices(nhl_cal)
    eligible_set = set(eligible_list)
    if not eligible_set:
        return out

    n_teams_round = max(4, min(int(max_teams_per_round), 32))
    n_inner = max(4, min(int(max_inner_steps), 60))
    n_sus = max(4, min(int(max_suspicious_pairs), 40))
    for _pass_i in range(max(1, int(max_passes))):
        improved_outer = False
        for _inner in range(n_inner):
            team_days = _build_team_game_days(out, nhl_cal=nhl_cal, regular_only=True)
            pen0 = _league_penalty_from_team_days(team_days)
            best_pen = pen0
            best_tuple: Optional[Tuple[int, int, Tuple[str, str, bool]]] = None
            team_order = sorted(team_days.keys(), key=lambda t: (-_team_schedule_penalty(team_days[t]), t))[
                :n_teams_round
            ]
            for tid in team_order:
                sus = _schedule_suspicious_pairs(
                    team_days.get(tid, []),
                    out,
                    tid,
                    nhl_cal,
                    max_pairs=n_sus,
                )
                for old_d, sl in sus:
                    key = _slot_key(sl)
                    near4 = _candidate_eligible_calendar_days(int(old_d), eligible_list, max_index_radius=3)
                    near4_set = set(near4)
                    wide8_only = [
                        c
                        for c in _candidate_eligible_calendar_days(int(old_d), eligible_list, max_index_radius=6)
                        if c not in near4_set
                    ]
                    t_home, t_away = _team_ids_for_slot(sl)
                    before_pair = 0.0
                    for tmid in sorted({str(t_home or ""), str(t_away or "")}):
                        if tmid:
                            before_pair += _team_schedule_penalty(sorted(set(team_days.get(tmid, []))))
                    for new_d in near4 + wide8_only:
                        if not _can_place_slot_on_day(
                            sl,
                            new_d,
                            out,
                            old_day=int(old_d),
                            eligible_set=eligible_set,
                            max_games_per_day=int(max_games_per_day),
                            nhl_cal=nhl_cal,
                        ):
                            continue
                        sl_here = _find_slot_on_day(out, int(old_d), key)
                        if sl_here is None:
                            continue
                        after_pair = _pair_penalty_after_day_move(
                            team_days, t_home, t_away, int(old_d), int(new_d)
                        )
                        new_pen = pen0 - before_pair + after_pair
                        if new_pen >= pen0 - 1e-9:
                            continue
                        if new_pen < best_pen - 1e-12:
                            best_pen = new_pen
                            best_tuple = (int(old_d), int(new_d), key)
                        elif abs(new_pen - best_pen) < 1e-12:
                            cand = (int(old_d), int(new_d), key)
                            if best_tuple is None or cand < (best_tuple[0], best_tuple[1], best_tuple[2]):
                                best_tuple = cand
            if best_tuple is None or best_pen >= pen0 - 1e-9:
                break
            o_d, n_d, key = best_tuple
            sl_apply = _find_slot_on_day(out, o_d, key)
            if sl_apply is None:
                break
            _move_slot(out, sl_apply, o_d, n_d)
            improved_outer = True
        if not improved_outer:
            break
    return out
def _smooth_user_team_schedule(by_day: Dict[int, List[Any]], *, user_id: str, nhl_cal: List[Dict[str, Any]]) -> Dict[int, List[Any]]:
    """Backward-compatible entry: smooth the full league (user + CPU) on regular-season nights."""
    if not nhl_cal:
        return {int(k): list(v or []) for k, v in (by_day or {}).items()}
    return _smooth_league_schedule(
        by_day,
        nhl_cal=nhl_cal,
        user_id=(user_id or None),
        max_passes=2,
        max_games_per_day=22,
        max_teams_per_round=8,
        max_inner_steps=12,
        max_suspicious_pairs=10,
    )
