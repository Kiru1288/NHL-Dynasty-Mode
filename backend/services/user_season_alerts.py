"""Daily GM alerts for the user's club (cheap: one pass over ~23 players + 32 standings rows).

* Suspensions: a popup when one of the user's players is suspended, and when the
  suspension ends he is put back into the even-strength slot he held (or the best
  open / fill-in slot of his position) instead of silently staying a scratch.
* Playoff clinch: real mathematical clinch / elimination marks (x / y / z / p / e)
  for the standings tables, and a one-time popup when the user's club clinches.
* Trade deadline: alerts ~7 days and 1 day before the deadline. These interrupt
  week / month sims so the GM is not simmed straight past the deadline.

Entry point: ``after_user_day(session)`` (called once per simulated calendar day).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)

CLINCH_RANK = {"p": 4, "z": 3, "y": 2, "x": 1}
CLINCH_LABEL = {
    "p": "Clinched the Presidents' Trophy",
    "z": "Clinched first place in the conference",
    "y": "Clinched the division",
    "x": "Clinched a playoff spot",
    "e": "Eliminated from playoff contention",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _season(session: Any) -> int:
    return int(getattr(session, "season_calendar_year", 0) or 0)


def _iso_for(session: Any, idx: int) -> str:
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= idx < len(cal) and isinstance(cal[idx], dict):
        row = cal[idx]
        return str(row.get("iso") or row.get("date") or "")[:10]
    return ""


def _user_team(session: Any) -> Any:
    by_id = getattr(session, "team_by_id", None) or {}
    uid = getattr(session, "user_team_id", None)
    return by_id.get(str(uid)) or by_id.get(uid)


def _team_abbr(team: Any) -> str:
    for attr in ("abbreviation", "abbr", "abbrev", "code"):
        v = getattr(team, attr, None)
        if v:
            return str(v)
    return ""


def _popup(session: Any, dedupe_key: str, payload: Dict[str, Any]) -> bool:
    keys = getattr(session, "shown_event_keys", None)
    if isinstance(keys, set) and dedupe_key in keys:
        return False
    from services.franchise_sim import _append_showcase_popup

    team = _user_team(session)
    body = {
        "kind": "breaking_news",
        "gm_alert": True,
        "team_id": str(getattr(session, "user_team_id", "") or ""),
        "team_abbr": _team_abbr(team) if team is not None else "",
        "calendar_iso": _iso_for(session, max(0, int(getattr(session, "calendar_cursor", 0) or 0) - 1)),
        **payload,
    }
    _append_showcase_popup(session, dedupe_key, body)
    try:
        notes = getattr(session, "notifications", None)
        if isinstance(notes, list):
            notes.append(str(payload.get("headline") or ""))
            if len(notes) > 400:
                del notes[: len(notes) - 400]
    except Exception:
        log.debug("notification append failed", exc_info=True)
    return True


# ---------------------------------------------------------------------------
# A4: suspensions
# ---------------------------------------------------------------------------

def _ids_of(player: Any) -> set:
    from services.lineup_integrity import canonical_id, player_key

    raw = str(getattr(player, "id", "") or "")
    out = {player_key(player), canonical_id(raw), raw}
    out.discard("")
    return out


def _restore_returning_player(session: Any, player: Any, entry: Dict[str, Any]) -> Optional[str]:
    """Put a player back into the saved sheet. Returns the slot label, or None."""
    from services.lineup_integrity import (
        fits_slot,
        is_available,
        mark_user_lines_changed,
        normalize_lines,
        index_roster,
        iter_slots,
        lookup_player,
        player_key,
        player_ovr,
        saved_even_strength,
        user_roster,
        _set_slot,
    )

    lines = saved_even_strength(session)
    if lines is None or not is_available(player):
        return None
    payload = normalize_lines(lines)
    mine = _ids_of(player)
    rows = list(iter_slots(payload, required_only=True))
    for row in rows:
        if row["player_id"] in mine:
            return row["label"]  # never left the sheet

    key = player_key(player)
    before_ids = set(entry.get("lineup_ids") or [])
    idx = index_roster(user_roster(session))
    target = None
    # 1. His own slot, whoever is filling in there now.
    for row in rows:
        if (
            row["group"] == entry.get("group")
            and row["unit_id"] == entry.get("unit_id")
            and row["slot"] == entry.get("slot")
            and fits_slot(player, row["slot"])
        ):
            target = row
            break
    # 2. Top-most empty slot of his position.
    if target is None:
        empties = [r for r in rows if not r["player_id"] and fits_slot(player, r["slot"])]
        if empties:
            target = min(empties, key=lambda r: r["index"])
    # 3. Weakest fill-in (someone not in the lineup when he was suspended).
    if target is None:
        fillins = []
        for r in rows:
            if not r["player_id"] or r["player_id"] in before_ids or not fits_slot(player, r["slot"]):
                continue
            occ = lookup_player(idx, r["player_id"])
            fillins.append((player_ovr(occ) if occ is not None else 0.0, r))
        if fillins:
            target = min(fillins, key=lambda t: t[0])[1]
    if target is None:
        return None

    _set_slot(payload, target["group"], target["unit_id"], target["slot"], key)
    block = dict((getattr(session, "lines", None) or {}).get("even_strength") or {})
    block["lines"] = payload
    history = list(block.get("integrity_log") or [])[-11:]
    history.append({
        "reason": "suspension_return",
        "vacated": [],
        "placed": [{"label": target["label"], "player_id": key, "player_name": entry.get("name") or ""}],
        "filled": [],
    })
    block["integrity_log"] = history
    block.setdefault("source", "user")
    session.lines["even_strength"] = block
    mark_user_lines_changed(session)
    return target["label"]


def _suspension_watch(session: Any) -> None:
    from services.lineup_integrity import (
        is_suspended,
        iter_slots,
        normalize_lines,
        player_key,
        player_name,
        saved_even_strength,
        user_roster,
    )

    roster = user_roster(session)
    watch: Dict[str, Dict[str, Any]] = dict(getattr(session, "_user_suspension_watch", None) or {})
    seen: set = set()
    changed = False
    for p in roster:
        key = player_key(p)
        if not key:
            continue
        seen.add(key)
        suspended = bool(is_suspended(p))
        if suspended and key not in watch:
            entry: Dict[str, Any] = {"name": player_name(p), "season": _season(session)}
            lines = saved_even_strength(session)
            if lines is not None:
                payload = normalize_lines(lines)
                rows = list(iter_slots(payload, required_only=True))
                entry["lineup_ids"] = [r["player_id"] for r in rows if r["player_id"]]
                mine = _ids_of(p)
                for r in rows:
                    if r["player_id"] in mine:
                        entry.update({"group": r["group"], "unit_id": r["unit_id"], "slot": r["slot"], "label": r["label"]})
                        break
            watch[key] = entry
            changed = True
            # Take him out of the lineup now and dress the best available replacement.
            try:
                from services.lineup_integrity import reconcile_user_lineup

                rep = reconcile_user_lineup(session, auto_fill=True, reason="suspension")
                filled = [f for f in (rep.get("filled") or []) if f.get("label") == entry.get("label")]
                if filled:
                    entry["replacement"] = filled[0].get("player_name")
            except Exception:
                log.exception("suspension lineup replace failed")
            try:
                from app.sim_engine.franchise.storyline_conduct import get_conduct_games_remaining

                games = int(get_conduct_games_remaining(p) or 0)
            except Exception:
                games = 0
            leave = str(getattr(p, "_world_conduct_status", "") or "") not in ("", "league_suspended") and bool(
                getattr(p, "_conduct_incident_id", None)
            )
            what = "placed on leave" if leave else "suspended"
            span = f" for {games} game{'s' if games != 1 else ''}" if games > 0 else ""
            slot_txt = (
                f" {entry['replacement']} takes his spot at {entry['label']}."
                if entry.get("replacement") and entry.get("label")
                else (f" His spot at {entry['label']} has been filled." if entry.get("label") else "")
            )
            _popup(
                session,
                f"gm_susp:{key}:{_season(session)}:{int(getattr(session, 'calendar_cursor', 0) or 0)}",
                {
                    "source_label": "Discipline",
                    "theme": "warning",
                    "headline": f"{entry['name']} {what}{span}",
                    "body": (
                        f"{entry['name']} has been {what}{span} and cannot dress.{slot_txt} "
                        "He returns to his line automatically when he is eligible again."
                    ),
                    "player_id": str(getattr(p, "id", "") or ""),
                    "player_name": entry["name"],
                    "games_remaining": games,
                    "actions": [{"id": "lines", "label": "Edit Lines", "primary": True}],
                },
            )
        elif not suspended and key in watch:
            entry = watch.pop(key)
            changed = True
            label = None
            try:
                label = _restore_returning_player(session, p, entry)
            except Exception:
                log.exception("suspension return lineup restore failed")
            name = entry.get("name") or player_name(p)
            _popup(
                session,
                f"gm_susp_end:{key}:{_season(session)}:{int(getattr(session, 'calendar_cursor', 0) or 0)}",
                {
                    "source_label": "Discipline",
                    "theme": "positive",
                    "headline": f"{name} returns from suspension",
                    "body": (
                        f"{name} is eligible again and is back in the lineup at {label}."
                        if label
                        else f"{name} is eligible again. Every slot of his position is taken, so slot him back in Edit Lines."
                    ),
                    "player_id": str(getattr(p, "id", "") or ""),
                    "player_name": name,
                    "actions": [{"id": "lines", "label": "Edit Lines"}],
                },
            )
    for key in [k for k in watch if k not in seen]:
        watch.pop(key, None)  # traded / sent down / released while suspended
        changed = True
    if changed or getattr(session, "_user_suspension_watch", None) is None:
        session._user_suspension_watch = watch


# ---------------------------------------------------------------------------
# A5: playoff clinch marks
# ---------------------------------------------------------------------------

def _spots_needed_threat(rows_conf: List[Dict[str, Any]], me: Dict[str, Any], ahead: Iterable[Dict[str, Any]]) -> bool:
    """True when ``me`` makes the playoffs even if every team in ``ahead`` finishes above it.

    NHL format per conference: top 3 of each division + the 2 best remaining (wild cards).
    Conferences that are too small or not split into divisions use top half (max 8).
    """
    ahead = list(ahead)
    divs: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows_conf:
        divs.setdefault(str(r.get("division") or ""), []).append(r)
    nhl_format = len(divs) >= 2 and all(d for d in divs) and all(len(v) >= 4 for v in divs.values())
    if not nhl_format:
        spots = min(8, max(1, len(rows_conf) // 2))
        return len(ahead) < spots
    my_div = str(me.get("division") or "")
    by_div: Dict[str, int] = {}
    for r in ahead:
        d = str(r.get("division") or "")
        by_div[d] = by_div.get(d, 0) + 1
    in_mine = by_div.get(my_div, 0)
    if in_mine < 3:
        return True
    wc_ahead = (in_mine - 3) + sum(max(0, n - 3) for d, n in by_div.items() if d != my_div)
    return wc_ahead < 2


def compute_clinch_marks(rows: Sequence[Dict[str, Any]], season_games: int) -> Dict[str, str]:
    """team_id -> mark ("p", "z", "y", "x", "e" or ""). Ties count against the team (conservative)."""
    games = int(season_games or 0)
    if games <= 0 or not rows:
        return {}
    data = []
    for r in rows:
        try:
            gp = int(r.get("gp") or 0)
            pts = int(r.get("pts") if r.get("pts") is not None else r.get("points") or 0)
        except (TypeError, ValueError):
            continue
        data.append({
            "team_id": str(r.get("team_id") or ""),
            "conference": str(r.get("conference") or ""),
            "division": str(r.get("division") or ""),
            "pts": pts,
            "max": pts + 2 * max(0, games - gp),
            "gp": gp,
        })
    if not data or all(d["gp"] <= 0 for d in data):
        return {}
    by_conf: Dict[str, List[Dict[str, Any]]] = {}
    for d in data:
        by_conf.setdefault(d["conference"], []).append(d)
    out: Dict[str, str] = {}
    for me in data:
        conf_rows = by_conf.get(me["conference"]) or []
        others = [o for o in conf_rows if o is not me]
        threats = [o for o in others if o["max"] >= me["pts"]]
        mark = ""
        if _spots_needed_threat(conf_rows, me, threats):
            mark = "x"
            if not any(o["max"] >= me["pts"] for o in others if o["division"] == me["division"]) and me["division"]:
                mark = "y"
            if not threats:
                mark = "z"
                if not any(o is not me and o["max"] >= me["pts"] for o in data):
                    mark = "p"
        else:
            sure_ahead = [o for o in others if o["pts"] > me["max"]]
            if not _spots_needed_threat(conf_rows, me, sure_ahead):
                mark = "e"
        out[me["team_id"]] = mark
    return out


def annotate_clinch_rows(rows: List[Dict[str, Any]], season_games: int) -> None:
    """Tag standings rows in place with ``clinch`` and ``clinched_playoff_spot``."""
    try:
        marks = compute_clinch_marks(rows, season_games)
    except Exception:
        log.debug("clinch marks failed", exc_info=True)
        return
    for r in rows:
        m = marks.get(str(r.get("team_id") or ""), "")
        r["clinch"] = m
        r["clinched_playoff_spot"] = m in CLINCH_RANK
        r["eliminated"] = m == "e"


def _season_games(session: Any) -> int:
    st = getattr(session, "standings", None)
    return int(getattr(st, "season_games", 0) or getattr(session, "games_per_team_schedule", 0) or 82)


def _standings_rows_for_clinch(session: Any) -> List[Dict[str, Any]]:
    st = getattr(session, "standings", None)
    recs = getattr(st, "records", None) or {}
    by_id = getattr(session, "team_by_id", None) or {}
    rows = []
    for tid, rec in recs.items():
        tm = by_id.get(tid) or by_id.get(str(tid))
        rows.append({
            "team_id": str(tid),
            "conference": str(getattr(rec, "conference", "") or getattr(tm, "conference", "") or ""),
            "division": str(getattr(rec, "division", "") or getattr(tm, "division", "") or ""),
            "gp": int(getattr(rec, "gp", 0) or 0),
            "pts": int(getattr(rec, "points", 0) or 0),
        })
    return rows


def _clinch_watch(session: Any) -> None:
    if str(getattr(session, "phase", "") or "") != "regular":
        return
    marks = compute_clinch_marks(_standings_rows_for_clinch(session), _season_games(session))
    uid = str(getattr(session, "user_team_id", "") or "")
    mark = marks.get(uid, "")
    if mark not in CLINCH_RANK:
        return
    season = _season(session)
    prev = dict(getattr(session, "_user_clinch_marks", None) or {})
    best_prev = str(prev.get(str(season)) or "")
    if CLINCH_RANK.get(best_prev, 0) >= CLINCH_RANK[mark]:
        return
    prev[str(season)] = mark
    session._user_clinch_marks = prev
    first = not best_prev
    team = _user_team(session)
    tname = str(getattr(team, "name", "") or "Your club")
    body = (
        f"{tname} have mathematically locked up a postseason berth. "
        "Whatever happens the rest of the way, there will be playoff hockey."
        if mark == "x" or first
        else f"{tname} just raised the bar: {CLINCH_LABEL[mark].lower()}."
    )
    if first and mark != "x":
        body += f" They also {CLINCH_LABEL[mark].lower()}."
    _popup(
        session,
        f"gm_clinch:{season}:{mark}",
        {
            "source_label": "Playoff Race",
            "theme": "positive",
            "headline": CLINCH_LABEL[mark] + "!" if first or mark != "x" else CLINCH_LABEL[mark],
            "body": body,
            "clinch": mark,
            "actions": [{"id": "standings", "label": "View Standings"}],
        },
    )


# ---------------------------------------------------------------------------
# A6: trade deadline approaching
# ---------------------------------------------------------------------------

def _deadline_watch(session: Any) -> None:
    if str(getattr(session, "phase", "") or "") not in ("regular", "preseason"):
        return
    from services.franchise_sim import _calendar_trade_deadline_index

    idx = _calendar_trade_deadline_index(session)
    if idx is None:
        return
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    left = int(idx) - cur
    if left < 0 or left > 7:
        return
    season = _season(session)
    deadline_iso = _iso_for(session, int(idx))
    if left <= 1:
        tag, when = "1", ("today" if left == 0 else "tomorrow")
    else:
        tag, when = "7", f"in {left} days"
    _popup(
        session,
        f"gm_deadline:{season}:{tag}",
        {
            "source_label": "Trade Deadline",
            "theme": "warning",
            "headline": f"Trade deadline {when}",
            "body": (
                f"The NHL trade deadline is {when}{f' ({deadline_iso})' if deadline_iso else ''}. "
                "After it passes, trades are frozen until the season ends — buy, sell, or stand pat now."
            ),
            "interrupt_sim": True,
            "actions": [{"id": "tradehub", "label": "Open Trade Hub", "primary": True}],
        },
    )


# ---------------------------------------------------------------------------

def after_user_day(session: Any) -> None:
    """Run every daily GM watcher; never raises."""
    for fn in (_suspension_watch, _clinch_watch, _deadline_watch):
        try:
            fn(session)
        except Exception:
            log.exception("user season alert %s failed", getattr(fn, "__name__", fn))
