"""Remaining franchise orchestration."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

def _validate_schedule_hard(
    by_day: Dict[int, List[Any]],
    nhl_cal: List[Dict[str, Any]],
    *,
    day_filter: Optional[int] = None,
) -> List[str]:
    """
    Hard schedule integrity validator used at startup and before daily simulation.

    This now checks:
    - games outside allowed calendar segments
    - blocked dates
    - slot.day mismatch
    - missing teams
    - self-play
    - same-day double-booking
    - duplicate exact matchup on same day
    - impossible NHL cadence like 4-in-4 and 5-in-7
    """
    errs: List[str] = []

    if day_filter is not None:
        days = [int(day_filter)]
    else:
        days = sorted(int(d) for d in (by_day or {}).keys())

    seen_matchups: set = set()

    for d in days:
        slots = list((by_day or {}).get(d, []) or [])

        if not slots:
            continue

        row = nhl_cal[d] if 0 <= d < len(nhl_cal) else {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        allows = row.get("allows_games")
        if allows is None:
            allows = row.get("allowsGames")

        in_game_segment = seg in ("preseason", "regular")

        if not in_game_segment:
            errs.append(
                f"Day {d}: games scheduled outside preseason/regular segment ({seg or 'unknown'})."
            )

        if allows is False:
            errs.append(f"Day {d}: games scheduled on a blocked calendar date.")

        strict_team_uniqueness = seg == "regular"
        team_seen: set = set()

        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            sday = int(getattr(sl, "day", d) or d)

            if sday != d:
                errs.append(f"Day {d}: slot day mismatch ({sday}).")

            if hid == "" or aid == "":
                errs.append(f"Day {d}: slot missing team id(s).")
                continue

            if hid == aid:
                errs.append(f"Day {d}: self-play slot ({hid}).")

            if strict_team_uniqueness and (hid in team_seen or aid in team_seen):
                errs.append(f"Day {d}: team double-booked ({hid} vs {aid}).")

            team_seen.add(hid)
            team_seen.add(aid)

            key = (d, min(hid, aid), max(hid, aid))

            if strict_team_uniqueness and key in seen_matchups:
                errs.append(f"Day {d}: duplicate matchup detected ({hid} vs {aid}).")

            seen_matchups.add(key)

    # Only run full cadence scan when validating the whole schedule.
    # If day_filter is supplied, we are doing a quick daily slot check.
    if day_filter is None:
        errs.extend(_validate_league_cadence_hard(by_day, nhl_cal))

    return errs
def _stable_storyline_id(raw: Dict[str, Any]) -> str:
    explicit = str(raw.get("id") or "").strip()
    if explicit:
        return explicit
    parts = [
        str(raw.get("type") or raw.get("tone") or "storyline"),
        str(raw.get("calendar_iso") or raw.get("date") or raw.get("day") or ""),
        str(raw.get("team_id") or raw.get("team") or ""),
        str(raw.get("player_id") or raw.get("player_name") or ""),
        str(raw.get("headline") or raw.get("title") or raw.get("text") or raw.get("summary") or ""),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:12]
    return f"story_{digest}"
def _get_player_health_status(player: Any) -> str:
    """Return INJURED | DAY_TO_DAY | HEALTHY from player.health if present."""
    h = getattr(player, "health", None)
    if h is None:
        return "HEALTHY"
    st: Any = None
    if isinstance(h, dict):
        st = h.get("injury_status")
    else:
        st = getattr(h, "injury_status", None)
    if st is None:
        return "HEALTHY"
    val = getattr(st, "value", None)
    raw = str(val if val is not None else getattr(st, "name", None) or st).lower().replace("-", "_")
    if "day_to_day" in raw or raw == "daytoday":
        return "DAY_TO_DAY"
    if raw in ("injured", "injury", "out"):
        return "INJURED"
    if raw in ("healthy", "health"):
        return "HEALTHY"
    if "injur" in raw and "healthy" not in raw:
        return "INJURED"
    return "HEALTHY"
def _get_live_injury_games_remaining(player: Any) -> int:
    return max(0, int(getattr(player, "_world_injury_games_remaining", 0) or 0))
def _get_live_injury_tier(player: Any) -> Optional[str]:
    t = getattr(player, "_world_injury_tier", None)
    if t is not None and str(t).strip():
        return str(t).strip().lower()
    return None
def _is_player_live_injured(player: Any) -> bool:
    if _get_live_injury_games_remaining(player) > 0:
        return True
    return _get_player_health_status(player) in ("INJURED", "DAY_TO_DAY")
def _find_latest_injury_log_for_player(session: FranchiseSession, player_id: str) -> Dict[str, Any]:
    pid = str(player_id or "")
    for inj in reversed(list(getattr(session, "injury_log_all", None) or [])):
        if not isinstance(inj, dict):
            continue
        if str(inj.get("player_id") or "") == pid:
            return dict(inj)
    return {}
def _estimate_return_from_games_remaining(session: FranchiseSession, games_remaining: int) -> Tuple[str, str]:
    gr = int(max(0, games_remaining))
    if gr <= 0:
        return "", ""
    estimate = f"In {gr} games"
    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    counted = 0
    end_idx = cur
    for i in range(cur, min(len(cal), cur + 400)):
        row = cal[i]
        seg = str(row.get("segment") or "")
        if seg not in ("preseason", "regular", "playoffs"):
            continue
        if row.get("allows_games") is False:
            continue
        counted += 1
        if counted >= gr:
            end_idx = i
            break
    iso = ""
    if cal and 0 <= end_idx < len(cal):
        iso = str(cal[end_idx].get("iso") or "")
    return estimate, iso
def _tier_human_label(tier: Optional[str]) -> str:
    t = str(tier or "minor").lower()
    if t == "major":
        return "Significant injury"
    if t == "moderate":
        return "Moderate injury"
    return "Minor injury"
def _build_active_injury_row(session: FranchiseSession, player: Any, team: Any, team_id: str) -> Dict[str, Any]:
    pname = _name_str(player)
    pos = _pos_str(player)
    ab = _franchise_team_abbrev(team)
    gr = _get_live_injury_games_remaining(player)
    status = _get_player_health_status(player)
    if gr > 0 and status == "HEALTHY":
        status = "INJURED"
    tier_guess = _get_live_injury_tier(player)
    if tier_guess is None and _is_player_live_injured(player):
        tier_guess = "minor"
    tier = str(tier_guess or "minor").lower()
    meta = _find_latest_injury_log_for_player(session, str(getattr(player, "id", "") or ""))
    tid_s = str(team_id)
    pid_s = str(getattr(player, "id", "") or "")
    dkey = meta.get("calendar_day", meta.get("date", "unk"))
    stable_id = f"injury:{tid_s}:{pid_s}:{dkey}"
    cal_day = int(meta.get("calendar_day", meta.get("date", 0)) or 0)
    cal_iso = str(meta.get("calendar_iso") or "")
    if not cal_iso and cal_day >= 0:
        calrows = getattr(session, "nhl_calendar", None) or []
        if cal_day < len(calrows):
            cal_iso = str(calrows[cal_day].get("iso") or "")
    ret_est, ret_iso = _estimate_return_from_games_remaining(session, gr)
    inj_label = _tier_human_label(tier)
    desc = f"{pname} ({ab}): {inj_label}, {gr} games remaining."
    games_initial = int(meta.get("games_initial", meta.get("games", gr)) or gr)
    return {
        "id": stable_id,
        "player_id": pid_s,
        "player_name": pname,
        "team_id": tid_s,
        "team_abbr": ab,
        "team_abbrev": ab,
        "position": pos,
        "status": status,
        "injury_status": status,
        "health_status": status,
        "injury": inj_label,
        "injury_type": tier,
        "tier": tier,
        "severity": tier,
        "games_remaining": gr,
        "days_remaining": gr,
        "duration": f"{gr} games" if gr else "0 games",
        "return_estimate": ret_est,
        "return_date": ret_iso,
        "calendar_day": cal_day,
        "calendar_iso": cal_iso,
        "date": cal_iso or (str(cal_day) if cal_day else ""),
        "description": desc,
        "source": "live_player_state",
        "games_initial": games_initial,
    }
def _country_in_wjc_pool(code: str) -> bool:
    return str(code or "") in set(_wjc_pool_codes())
def _player_ovr01(p: Any) -> float:
    ovr_f = getattr(p, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
    except Exception:
        ov = 0.55
    if ov > 1.5:
        ov = ov / 99.0
    return float(ov)
def _collect_user_wjc_prospects(session: FranchiseSession, rng: random.Random) -> List[Dict[str, Any]]:
    """U20 on your AHL affiliate, plus NHL U20 only if the user loaned them to their WJC country."""
    out: List[Dict[str, Any]] = []
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        return out
    loans = getattr(session, "wjc_nhl_u20_loan", None) or {}

    def _row(p: Any, *, roster: str) -> None:
        if getattr(p, "retired", False):
            return
        ident = getattr(p, "identity", None)
        if ident is None:
            return
        age = int(getattr(ident, "age", 99) or 99)
        if age > 20:
            return
        pid = str(getattr(p, "id", "") or "")
        nm = str(getattr(ident, "name", None) or "?")
        bc = str(getattr(ident, "birth_country", "") or "")
        code = _wjc_country_for_birth(rng, bc)
        if not _country_in_wjc_pool(code):
            # If the country is not in this year's tournament field, player does not participate.
            return
        lab = _wjc_country_label(code)
        ov = _player_ovr01(p)
        cut = 0.62 + 0.08 * rng.random()
        made = bool(ov >= cut or rng.random() < 0.28)
        note = (
            f"Named to {lab} U20 national roster."
            if made
            else f"Released from {lab} U20 national camp before the tournament."
        )
        out.append(
            {
                "player_id": pid,
                "name": nm,
                "age": age,
                "nationality": bc,
                "wjc_country": code,
                "wjc_country_label": lab,
                "made_wjc_team": made,
                "note": note,
                "roster": roster,
            }
        )

    for p in getattr(ut, "ahl_roster", None) or []:
        _row(p, roster="AHL")

    for p in getattr(ut, "roster", None) or []:
        if not loans.get(str(getattr(p, "id", "") or ""), False):
            continue
        _row(p, roster="NHL (loaned)")

    out.sort(key=lambda x: (-int(x.get("made_wjc_team") or 0), str(x.get("roster") or ""), str(x.get("name") or "")))
    return out
def _ensure_wjc_tournament_bundle(session: FranchiseSession) -> None:
    sy = int(session.season_calendar_year)
    b = getattr(session, "wjc_tournament_bundle", None)
    if isinstance(b, dict) and int(b.get("season_sy", -1)) == sy:
        return
    rng = _rng_for_event(session, f"wjc_bundle_{sy}")
    core = _simulate_wjc_national_bundle(rng)
    session.wjc_tournament_bundle = {"season_sy": sy, **core}
