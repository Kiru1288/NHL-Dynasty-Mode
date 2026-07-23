"""WJC, All-Star, and showcase events."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

def _rng_for_event(session: FranchiseSession, label: str) -> random.Random:
    base = abs(hash(f"{session.session_id}|{label}|{int(session.season_calendar_year)}")) % (2**31 - 1)
    return random.Random(int(base) or 1)
def _wjc_calendar_dates(season_y: int) -> List[date]:
    """Dec 26 (season_y) through Jan 5 (season_y+1), inclusive."""
    out: List[date] = []
    d = date(season_y, 12, 26)
    end = date(season_y + 1, 1, 5)
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out
def _wjc_day_index_for_iso(iso: str, season_y: int) -> Optional[int]:
    try:
        y, m, dd = (int(x) for x in iso.split("-"))
        cur = date(y, m, dd)
    except (TypeError, ValueError):
        return None
    for i, d in enumerate(_wjc_calendar_dates(season_y)):
        if d == cur:
            return i
    return None
def _wjc_country_for_birth(rng: random.Random, birth_country: str) -> str:
    bc = str(birth_country or "").strip().lower()
    pairs = [
        (("canada", "can"), "CAN"),
        (("united states", "u.s", "usa", "america"), "USA"),
        (("sweden", "sverige"), "SWE"),
        (("finland", "suomi"), "FIN"),
        (("czech", "czechia"), "CZE"),
        (("slovak", "slovakia"), "SVK"),
        (("germany", "deutsch"), "GER"),
        (("latvia", "latv"), "LAT"),
        (("russia", "╤Ç╨╛╤ü╤ü", "rossiya"), "RUS"),
        (("kazakh", "╥¢╨░╨╖╨░"), "KAZ"),
        (("denmark", "norway", "austria", "switzerland"), "GER"),
    ]
    for hints, code in pairs:
        if any(h in bc for h in hints):
            return code
    return ""
def _wjc_countries_meta() -> List[Tuple[str, str]]:
    """National programs only (no NHL clubs)."""
    return [
        ("CAN", "Canada"),
        ("USA", "United States"),
        ("RUS", "Russia"),
        ("FIN", "Finland"),
        ("SWE", "Sweden"),
        ("GER", "Germany"),
        ("CZE", "Czechia"),
        ("LAT", "Latvia"),
        ("KAZ", "Kazakhstan"),
    ]
def _wjc_country_label(code: str) -> str:
    for c, lab in _wjc_countries_meta():
        if c == code:
            return lab
    return str(code or "?")
def _wjc_pool_codes() -> List[str]:
    return [c for c, _ in _wjc_countries_meta()]
def _rr_standings_from_slice(codes: List[str], label_by: Dict[str, str], rr_slice: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    st: Dict[str, Dict[str, int]] = {
        c: {"gp": 0, "w": 0, "otl": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for c in codes
    }
    for g in rr_slice:
        home = str(g["home"])
        away = str(g["away"])
        hg = int(g["home_goals"])
        ga = int(g["away_goals"])
        st[home]["gp"] += 1
        st[away]["gp"] += 1
        st[home]["gf"] += hg
        st[home]["ga"] += ga
        st[away]["gf"] += ga
        st[away]["ga"] += hg
        if hg > ga:
            st[home]["w"] += 1
            st[away]["l"] += 1
            st[home]["pts"] += 2
        else:
            st[away]["w"] += 1
            st[home]["l"] += 1
            st[away]["pts"] += 2

    def _row_key(c: str) -> Tuple[int, int, int, int]:
        s = st[c]
        return (-int(s["pts"]), -(int(s["w"]) - int(s["l"])), -(int(s["gf"]) - int(s["ga"])), -int(s["gf"]))

    ordered = sorted(codes, key=_row_key)
    rows: List[Dict[str, Any]] = []
    for rank, c in enumerate(ordered, start=1):
        s = st[c]
        rows.append(
            {
                "place": rank,
                "code": c,
                "label": label_by[c],
                "gp": s["gp"],
                "w": s["w"],
                "otl": s["otl"],
                "l": s["l"],
                "gf": s["gf"],
                "ga": s["ga"],
                "pts": s["pts"],
            }
        )
    return rows
def _simulate_wjc_national_bundle(rng: random.Random) -> Dict[str, Any]:
    """Full U20 worlds ΓÇö national teams only (deterministic from rng)."""
    countries = _wjc_countries_meta()
    codes = [c for c, _ in countries]
    label_by = {c: lab for c, lab in countries}

    rr_games: List[Dict[str, Any]] = []
    for i, hi in enumerate(codes):
        for aj in codes[i + 1 :]:
            home, away = (hi, aj) if rng.random() < 0.5 else (aj, hi)
            hg = rng.randint(1, 5)
            ga = rng.randint(1, 5)
            if hg == ga:
                ga = min(6, ga + 1)
                if hg == ga:
                    hg = max(1, hg - 1)
            rr_games.append(
                {
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ga,
                    "home_label": label_by[home],
                    "away_label": label_by[away],
                }
            )

    st_full = _rr_standings_from_slice(codes, label_by, rr_games)
    code_order = [r["code"] for r in st_full]

    def _play_pair(a: str, b: str, label: str, lb: Dict[str, str]) -> Dict[str, Any]:
        home, away = (a, b) if rng.random() < 0.5 else (b, a)
        hg = rng.randint(2, 5)
        ga = rng.randint(1, 4)
        if hg == ga:
            hg = min(6, hg + 1)
        w = home if hg > ga else away
        l = away if hg > ga else home
        return {
            "round": label,
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ga,
            "winner": w,
            "loser": l,
            "home_label": lb[home],
            "away_label": lb[away],
            "winner_label": lb[w],
            "loser_label": lb[l],
        }

    playoff_pool = code_order[:8] if len(code_order) >= 8 else list(code_order)
    while len(playoff_pool) < 8 and playoff_pool:
        playoff_pool.append(playoff_pool[-1])
    s1, s2, s3, s4, s5, s6, s7, s8 = (
        playoff_pool[0],
        playoff_pool[1],
        playoff_pool[2],
        playoff_pool[3],
        playoff_pool[4],
        playoff_pool[5],
        playoff_pool[6],
        playoff_pool[7],
    )
    qf = [
        _play_pair(s1, s8, "Quarterfinal", label_by),
        _play_pair(s2, s7, "Quarterfinal", label_by),
        _play_pair(s3, s6, "Quarterfinal", label_by),
        _play_pair(s4, s5, "Quarterfinal", label_by),
    ]
    w_qf = [g["winner"] for g in qf]
    sf = [_play_pair(w_qf[0], w_qf[1], "Semifinal", label_by), _play_pair(w_qf[2], w_qf[3], "Semifinal", label_by)]
    w_sf = [g["winner"] for g in sf]
    l_sf = [g["loser"] for g in sf]
    bronze = _play_pair(l_sf[0], l_sf[1], "Bronze", label_by)
    gold = _play_pair(w_sf[0], w_sf[1], "Gold Medal", label_by)
    medals = {
        "gold": gold["winner"],
        "silver": gold["loser"],
        "bronze": bronze["winner"],
        "fourth": bronze["loser"],
    }
    medal_labels = {k: label_by.get(v, v) for k, v in medals.items()}

    return {
        "countries": [{"code": c, "label": lab} for c, lab in countries],
        "rr_games": rr_games,
        "playoffs": {"quarterfinals": qf, "semifinals": sf, "bronze": bronze, "gold": gold},
        "medals": medals,
        "medal_labels": medal_labels,
    }
def _strip_wjc_live_pending(session: FranchiseSession) -> None:
    session.pending_ui_popups = [p for p in (session.pending_ui_popups or []) if not p.get("wjc_live")]
def _push_wjc_live_popup(session: FranchiseSession, payload: Dict[str, Any]) -> None:
    """Replace any previous WJC live overlay so bulk sims only surface the latest day."""
    _strip_wjc_live_pending(session)
    pid = f"pop_{uuid.uuid4().hex[:12]}"
    body = dict(payload)
    body["id"] = pid
    body["wjc_live"] = True
    session.pending_ui_popups.append(body)
def _wjc_live_tournament_payload(session: FranchiseSession, iso: str, d_idx: int, n_days: int) -> Dict[str, Any]:
    sy = int(session.season_calendar_year)
    _ensure_wjc_tournament_bundle(session)
    bundle = getattr(session, "wjc_tournament_bundle", None) or {}
    rr_all: List[Dict[str, Any]] = list(bundle.get("rr_games") or [])
    countries = list(bundle.get("countries") or [])
    codes = [str(c["code"]) for c in countries]
    label_by = {str(c["code"]): str(c["label"]) for c in countries}
    rng = _rng_for_event(session, f"wjc_prospects_{sy}")

    n_rr = len(rr_all)
    rr_through = min(n_rr, max(1, ((d_idx + 1) * n_rr + n_days - 1) // n_days))
    rr_slice = rr_all[:rr_through]
    standings = _rr_standings_from_slice(codes, label_by, rr_slice)

    po_all = bundle.get("playoffs") or {}
    po_out: Dict[str, Any] = {}
    if d_idx >= 7 and rr_through >= n_rr:
        po_out["quarterfinals"] = po_all.get("quarterfinals") or []
    if d_idx >= 8 and rr_through >= n_rr:
        po_out["semifinals"] = po_all.get("semifinals") or []
    if d_idx >= 9 and rr_through >= n_rr:
        po_out["bronze"] = po_all.get("bronze")
    if d_idx >= 10 and rr_through >= n_rr:
        po_out["gold"] = po_all.get("gold")

    complete = bool(d_idx >= 10 and rr_through >= n_rr)
    medals = bundle.get("medal_labels") if complete else {}
    user_prospects = _collect_user_wjc_prospects(session, rng)

    return {
        "kind": "wjc_tournament",
        "wjc_live": True,
        "wjc_phase": "complete" if complete else "live",
        "calendar_iso": iso,
        "wjc_day": d_idx + 1,
        "wjc_days_total": n_days,
        "title": f"World Juniors (U20) ΓÇö day {d_idx + 1} of {n_days}",
        "season_label": f"{sy}ΓÇô{sy + 1}",
        "countries": countries,
        "round_robin_games": rr_slice,
        "round_robin_total": n_rr,
        "standings": standings,
        "playoffs": po_out,
        "medal_labels": medals if complete else {},
        "medals_final": complete,
        "user_prospects": user_prospects,
    }
def _maybe_enqueue_wjc_loan_decisions(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    """After Christmas Day, before the first WJC calendar date, offer NHL U20 loan releases (national teams)."""
    iso_done = str(day_meta.get("iso") or "")
    sy = int(session.season_calendar_year)
    if iso_done != f"{sy}-12-25":
        return
    if getattr(session, "wjc_loan_prompts_enqueued", False):
        return
    ut = session.team_by_id.get(str(session.user_team_id))
    if ut is None:
        session.wjc_loan_prompts_enqueued = True
        return
    offered = False
    for p in getattr(ut, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        ident = getattr(p, "identity", None)
        if ident is None:
            continue
        age = int(getattr(ident, "age", 99) or 99)
        if age > 20:
            continue
        pid = str(getattr(p, "id", "") or "")
        nm = str(getattr(ident, "name", None) or "?")
        bc = str(getattr(ident, "birth_country", "") or "")
        nat = _wjc_country_for_birth(session.sim.rng, bc)
        if not _country_in_wjc_pool(nat):
            continue
        nat_lab = _wjc_country_label(nat)
        storyline_id = f"story_wjc_loan_{pid}"
        dec_id = f"dec_{uuid.uuid4().hex[:12]}"
        session.pending_decisions.append(
            {
                "id": dec_id,
                "storyline_id": storyline_id,
                "kind": "wjc_u20_loan",
                "title": f"World Juniors ΓÇö {nm}",
                "description": (
                    f"{nm} ({age}) is U20-eligible for {nat_lab}. "
                    "Loan him to the national junior tournament roster (WJC recap only ΓÇö no NHL club in the IIHF bracket), "
                    "or keep him with your NHL club."
                ),
                "options": [
                    {
                        "id": "keep",
                        "label": "Keep on NHL roster",
                        "effects": {"chemistry_delta": 0, "prospect_exposure_delta": -1},
                        "effect_summary": "Retains NHL depth, limits international development reps.",
                    },
                    {
                        "id": "loan",
                        "label": f"Loan to {nat_lab} U20",
                        "effects": {"chemistry_delta": 1, "prospect_exposure_delta": 2},
                        "effect_summary": "Improves tournament exposure and confidence, temporarily reduces NHL depth.",
                    },
                ],
                "meta": {
                    "storyline_id": storyline_id,
                    "player_id": pid,
                    "player_name": nm,
                    "wjc_country": nat,
                    "wjc_country_label": nat_lab,
                    "team_id": str(session.user_team_id),
                    "cause": "National team requested U20 availability during World Juniors.",
                },
            }
        )
        offered = True
    session.wjc_loan_prompts_enqueued = True
    if offered:
        session.notifications.append("World Juniors: loan decisions needed for eligible U20 NHL players (Hub).")
def _simulate_showcase_score(rng: random.Random) -> Tuple[int, int, bool]:
    """Return (home_goals, away_goals, overtime) for an outdoor / exhibition tilt."""
    hg = rng.randint(2, 5)
    ag = rng.randint(2, 5)
    ot = rng.random() < 0.22
    if not ot and hg == ag:
        ag += rng.choice([-1, 1])
        ag = max(1, min(6, ag))
    if ot and hg == ag:
        hg += 1
    return hg, ag, ot
def _allstar_game_payload(session: FranchiseSession, rng: random.Random) -> Dict[str, Any]:
    pool: List[Tuple[float, str, str]] = []
    for tid, tm in session.team_by_id.items():
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            ident = getattr(p, "identity", None)
            if ident is None:
                continue
            ovr_f = getattr(p, "ovr", None)
            try:
                ov = float(ovr_f() if callable(ovr_f) else ovr_f)
            except Exception:
                ov = 0.6
            if ov > 1.5:
                ov = ov / 99.0
            nm = str(getattr(ident, "name", None) or "?")
            pool.append((ov, str(tid), nm))
    pool.sort(key=lambda x: -x[0])
    top = pool[:24]
    rng.shuffle(top)
    team_a = [x for x in top[:12]]
    team_b = [x for x in top[12:24]]
    ha, hb = rng.randint(4, 8), rng.randint(3, 7)
    if rng.random() < 0.5:
        ha, hb = hb, ha
    ut = str(session.user_team_id)
    user_names_a = [nm for ov, tid, nm in team_a if tid == ut]
    user_names_b = [nm for ov, tid, nm in team_b if tid == ut]
    return {
        "kind": "allstar_game",
        "title": "NHL All-Star Game",
        "season_label": f"{session.season_calendar_year}ΓÇô{int(session.season_calendar_year) + 1}",
        "team_a_label": "Team Pacific / Metro",
        "team_b_label": "Team Atlantic / Central",
        "team_a_score": ha,
        "team_b_score": hb,
        "team_a": [{"name": nm, "is_user": tid == ut} for ov, tid, nm in team_a],
        "team_b": [{"name": nm, "is_user": tid == ut} for ov, tid, nm in team_b],
        "user_allstars": user_names_a + user_names_b,
    }
def _maybe_enqueue_showcase_popups(session: FranchiseSession, day_meta: Dict[str, Any]) -> None:
    iso = str(day_meta.get("iso") or "")
    tags = {str(x).lower() for x in (day_meta.get("tags") or [])}
    sy = int(session.season_calendar_year)
    y2 = sy + 1

    if "world_juniors" in tags:
        d_idx = _wjc_day_index_for_iso(iso, sy)
        if d_idx is not None:
            n_days = len(_wjc_calendar_dates(sy))
            pl = _wjc_live_tournament_payload(session, iso, d_idx, n_days)
            _push_wjc_live_popup(session, pl)
            if pl.get("wjc_phase") == "complete" and f"wjc_final_arch_{sy}" not in session.shown_event_keys:
                session.shown_event_keys.add(f"wjc_final_arch_{sy}")
                snap = {k: v for k, v in pl.items() if k not in ("wjc_live",)}
                arch = list(getattr(session, "showcase_archive", None) or [])
                arch.append(snap)
                session.showcase_archive = arch[-48:]

    if iso == f"{sy}-12-31" and "winter_classic" in tags:
        rk = f"winter_classic_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            ov = _special_event_overlay(session, day_meta) or {}
            home = ov.get("home") or {"abbr": "?", "name": "TBD"}
            away = ov.get("away") or {"abbr": "?", "name": "TBD"}
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "winter_classic",
                    "title": str(ov.get("title") or "Winter Classic"),
                    "iso": iso,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )

    if iso == f"{y2}-01-13" and "heritage_classic" in tags:
        rk = f"heritage_classic_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            ov = _special_event_overlay(session, day_meta) or {}
            home = ov.get("home") or {"abbr": "?", "name": "TBD"}
            away = ov.get("away") or {"abbr": "?", "name": "TBD"}
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "heritage_classic",
                    "title": str(ov.get("title") or "Heritage Classic"),
                    "iso": iso,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )

    if iso == f"{y2}-02-03" and "allstar_break" in tags:
        rk = f"allstar_game_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            _append_showcase_popup(session, rk, _allstar_game_payload(session, rng))

    if iso == f"{y2}-03-14" and "four_nations" in tags:
        rk = f"four_nations_{sy}"
        if rk not in session.shown_event_keys:
            rng = _rng_for_event(session, rk)
            teams = ["CAN", "USA", "SWE", "FIN"]
            rng.shuffle(teams)
            a, b = teams[0], teams[1]
            hg, ag, ot = _simulate_showcase_score(rng)
            _append_showcase_popup(
                session,
                rk,
                {
                    "kind": "showcase_game",
                    "subkind": "four_nations",
                    "title": "4 Nations Face-Off ΓÇö Final",
                    "iso": iso,
                    "home": {"abbr": a, "name": a, "id": ""},
                    "away": {"abbr": b, "name": b, "id": ""},
                    "home_goals": hg,
                    "away_goals": ag,
                    "overtime": ot,
                },
            )
