"""
Player consent to waive trade protection (NMC / NTC / M-NTC).

One source of truth for:

* whether a protected player may be traded to a destination right now
  (``consent_allows`` / ``clause_trade_permission``);
* asking a player to waive — from a GM meeting (general ask) or the Trade Hub
  (destination ask). The answer is deterministic for the player + season + waiver
  window: a single seeded roll per window, compared against how appealing the
  destination is. Re-asking can never re-roll;
* anti-spam: a "no" cannot be re-asked inside the same window, never sooner than
  ``REASK_MIN_DAYS`` sim days, and at most once before the trade deadline in-season.
  Every ask costs a little trust; repeated asks cost more;
* clearing consent once it is used (the player is traded) or once the window closes.

Windows: ``<season>-in`` runs from training camp through deadline day; ``<season>-off``
runs from the day after the deadline through the offseason until the league year
rolls over. Consent granted in a window expires with that window.

Storage (session attributes, kept under the historic name so trade context plumbing
— ``ctx["ntc_waivers"]`` — keeps working):

* ``session.ntc_waivers``: ``{player_id: consent_record}`` — granted consents only.
* ``session.clause_waiver_asks``: ``{player_id: {"history": [...], "windows": {...}}}``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import player_display_name
from app.sim_engine.trades.trade_rules import (
    _clause_summary,
    _dest_on_list,
    _market_size,
    _stable_unit_roll,
    _team_strength_proxy,
)

#: Minimum sim days before a declined waiver can be asked again (even across windows).
REASK_MIN_DAYS = 30
#: Destination-specific "no" answers allowed per window before the player shuts it down.
MAX_DECLINES_PER_WINDOW = 2
#: Trade value haircut once a clause is waived (burned leverage).
WAIVED_VALUE_PENALTY_PCT = 0.08
#: NMC holders are harder to move than NTC holders.
NMC_CHANCE_SCALE = 0.70
#: Trust cost per ask; extra cost for each further ask in the same window; decline cost.
ASK_TRUST_COST = 1.5
REPEAT_ASK_TRUST_COST = 2.5
DECLINE_TRUST_COST = 4.0
DECLINE_MORALE_COST = 3.0
ACCEPT_TRUST_GAIN = 1.0

_INACTIVE_STATUSES = frozenset({"used", "expired", "declined", "revoked"})
_OFF_PHASES = frozenset({"playoffs", "postseason", "playoff_ready", "post_cup", "complete"})
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ---------------------------------------------------------------------------
# Dates / windows
# ---------------------------------------------------------------------------


def _parse_iso(iso: str) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat(str(iso or "")[:10])
    except (TypeError, ValueError):
        return None


def _add_days(iso: str, days: int) -> str:
    d = _parse_iso(iso)
    if d is None:
        return ""
    return (d + _dt.timedelta(days=int(days))).isoformat()


def format_iso_label(iso: str) -> str:
    """``2027-03-11`` -> ``Mar 11, 2027``."""
    d = _parse_iso(iso)
    if d is None:
        return str(iso or "")
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def _deadline_index(cal: List[Dict[str, Any]]) -> Optional[int]:
    tagged = [
        i for i, d in enumerate(cal)
        if isinstance(d, dict) and "trade_deadline" in tuple(d.get("tags") or ())
    ]
    if tagged:
        return tagged[-1]
    for i, d in enumerate(cal):
        if isinstance(d, dict) and str(d.get("iso") or "")[5:10] == "03-10":
            return i
    return None


def current_sim_iso(session: Any) -> str:
    """Today's sim date. Offseason free-agency days advance past the calendar's last day."""
    cal = list(getattr(session, "nhl_calendar", None) or [])
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    iso = ""
    if cal:
        idx = max(0, min(cursor, len(cal) - 1))
        iso = str((cal[idx] or {}).get("iso") or "")
    if bool(getattr(session, "free_agency_open", False)):
        season = int(getattr(session, "season_calendar_year", 2025) or 2025)
        fa_day = int(getattr(session, "fa_market_day", 0) or 0)
        fa_iso = _add_days(f"{season + 1}-07-01", max(0, fa_day - 1))
        if fa_iso and (not iso or fa_iso > iso):
            iso = fa_iso
    return iso


def waiver_window(session: Any) -> Dict[str, Any]:
    """The waiver window the session is in right now (see module docstring)."""
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    cal = list(getattr(session, "nhl_calendar", None) or [])
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    phase = str(getattr(session, "phase", "") or "").lower()
    today = current_sim_iso(session)
    d_idx = _deadline_index(cal) if cal else None
    deadline_iso = str((cal[d_idx] or {}).get("iso") or "") if d_idx is not None else f"{season + 1}-03-10"
    offseason_now = phase in _OFF_PHASES or phase in {
        "offseason",
        "draft",
        "entry_draft",
        "free_agency",
        "freeagency",
        "resign",
        "re_sign",
    } or bool(getattr(session, "free_agency_open", False))
    # Summer is its own waiver window. A "yes" from before the deadline does not
    # carry into July, including for deals signed before this offseason.
    if offseason_now:
        in_season = False
    elif d_idx is not None:
        in_season = cursor <= d_idx and phase not in _OFF_PHASES
    else:
        in_season = phase not in _OFF_PHASES
    next_camp_iso = f"{season + 1}-09-15"
    if in_season:
        ends_iso = deadline_iso
        reopen_iso = _add_days(deadline_iso, 1) or deadline_iso
        label = "before the trade deadline"
        ends_label = f"the trade deadline ({format_iso_label(deadline_iso)})"
    else:
        ends_iso = _add_days(next_camp_iso, -1)
        reopen_iso = next_camp_iso
        label = "this offseason"
        ends_label = f"the start of the {season + 1}-{(season + 2) % 100:02d} season"
    return {
        "key": f"{season}-{'in' if in_season else 'off'}",
        "season_year": season,
        "phase": "in" if in_season else "off",
        "label": label,
        "today_iso": today,
        "deadline_iso": deadline_iso,
        "ends_iso": ends_iso,
        "ends_label": ends_label,
        "reopen_iso": reopen_iso,
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _team_id(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")


def _consent_store(session: Any) -> Dict[str, Any]:
    store = getattr(session, "ntc_waivers", None)
    if not isinstance(store, dict):
        store = {}
        try:
            session.ntc_waivers = store
        except Exception:
            pass
    return store


def _ask_store(session: Any) -> Dict[str, Any]:
    store = getattr(session, "clause_waiver_asks", None)
    if not isinstance(store, dict):
        store = {}
        try:
            session.clause_waiver_asks = store
        except Exception:
            pass
    return store


def _bump_revision(session: Any) -> None:
    try:
        session._clause_consent_revision = int(getattr(session, "_clause_consent_revision", 0) or 0) + 1
    except Exception:
        pass


def consent_revision(session: Any) -> int:
    return int(getattr(session, "_clause_consent_revision", 0) or 0)


def migrate_consent_store(session: Any) -> None:
    """Fold pre-window entries (``pid`` / ``pid->dest`` caches) into the windowed layout.

    Old saves stored Trade Hub answers without a window. Accepted ones are kept as
    consent for the current window; declined ones become declines in the current window.
    """
    store = _consent_store(session)
    if not store or all(isinstance(v, dict) and v.get("window_key") for v in store.values()):
        return
    win = waiver_window(session)
    asks = _ask_store(session)
    migrated: Dict[str, Any] = {}
    for key, entry in list(store.items()):
        if isinstance(entry, dict) and entry.get("window_key"):
            migrated[str(key)] = entry
            continue
        pid = str(key).split("->", 1)[0]
        dest = str(key).split("->", 1)[1] if "->" in str(key) else None
        if not isinstance(entry, dict):
            continue
        dest = str(entry.get("destination_team_id") or dest or "") or None
        if bool(entry.get("accepted")):
            prev = migrated.get(pid) or {}
            allowed = list(prev.get("allowed_team_ids") or [])
            if dest and dest not in allowed:
                allowed.append(dest)
            migrated[pid] = {
                **{k: v for k, v in entry.items() if k not in ("cached",)},
                "player_id": pid,
                "accepted": True,
                "status": "granted",
                "scope": "teams" if allowed else "any",
                "allowed_team_ids": allowed,
                "destination_team_id": allowed[0] if len(allowed) == 1 else None,
                "source_team_id": str(entry.get("source_team_id") or "") or None,
                "season_year": win["season_year"],
                "window_key": win["key"],
                "granted_iso": win["today_iso"],
                "expires_iso": win["ends_iso"],
                "expires_label": win["ends_label"],
                "origin": str(entry.get("origin") or "trade_hub"),
                "value_penalty_pct": float(entry.get("value_penalty_pct") or WAIVED_VALUE_PENALTY_PCT),
            }
        elif entry.get("can_request"):
            book = asks.setdefault(pid, {"history": [], "windows": {}})
            book.setdefault("history", []).append({
                "window_key": win["key"],
                "season_year": win["season_year"],
                "iso": win["today_iso"],
                "destination_team_id": dest,
                "accepted": False,
                "origin": "trade_hub",
                "reason": str(entry.get("reason") or ""),
                "reask_after_iso": _reask_after(win),
            })
    session.ntc_waivers = migrated
    _bump_revision(session)


def prune_expired_consents(session: Any) -> int:
    """Drop consents whose window has closed. Returns how many were removed."""
    migrate_consent_store(session)
    store = _consent_store(session)
    if not store:
        return 0
    key = waiver_window(session)["key"]
    dead = [
        pid for pid, entry in store.items()
        if not isinstance(entry, dict)
        or str(entry.get("status") or "granted") in _INACTIVE_STATUSES
        or (entry.get("window_key") and str(entry.get("window_key")) != key)
    ]
    for pid in dead:
        store.pop(pid, None)
    # Ask history older than two league years carries no weight.
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)
    for pid, book in list(_ask_store(session).items()):
        hist = [h for h in (book.get("history") or []) if int(h.get("season_year") or season) >= season - 1]
        book["history"] = hist[-24:]
        wins = dict(book.get("windows") or {})
        book["windows"] = {k: v for k, v in wins.items() if str(k).split("-")[0].isdigit() and int(str(k).split("-")[0]) >= season - 1}
    if dead:
        _bump_revision(session)
    return len(dead)


def consume_consents(session: Any, player_ids: Iterable[Any]) -> List[str]:
    """Clear consent for players who were just traded (consent is single-use)."""
    store = _consent_store(session)
    used: List[str] = []
    for raw in player_ids or []:
        pid = str(raw or "")
        if pid and pid in store:
            store.pop(pid, None)
            used.append(pid)
        for key in [k for k in store if str(k).startswith(f"{pid}->")]:
            store.pop(key, None)
    if used:
        _bump_revision(session)
    return used


# ---------------------------------------------------------------------------
# Consent checks (pure — usable from trade rules with only a context dict)
# ---------------------------------------------------------------------------


def consent_allows(
    entry: Any,
    destination_team_id: Optional[str],
    *,
    window_key: Optional[str] = None,
    source_team_id: Optional[str] = None,
) -> bool:
    """True when ``entry`` (a consent record) lets the player go to ``destination_team_id``."""
    if entry is True:
        return True
    if not isinstance(entry, dict) or not bool(entry.get("accepted")):
        return False
    if str(entry.get("status") or "granted") in _INACTIVE_STATUSES:
        return False
    if window_key and entry.get("window_key") and str(entry.get("window_key")) != str(window_key):
        return False
    src = str(entry.get("source_team_id") or "")
    if source_team_id and src and src != str(source_team_id):
        return False
    dest = str(destination_team_id or "")
    allowed = [str(t) for t in (entry.get("allowed_team_ids") or []) if str(t)]
    if allowed:
        return bool(dest) and dest in allowed
    single = str(entry.get("destination_team_id") or "")
    if single:
        return dest == single
    return True


def lookup_consent(waivers: Any, player_id: str, destination_team_id: Optional[str] = None) -> Any:
    if not isinstance(waivers, dict):
        return None
    pid = str(player_id or "")
    entry = waivers.get(pid)
    if entry is None and destination_team_id:
        entry = waivers.get(f"{pid}->{destination_team_id}")
    return entry


def active_consent(
    session: Any,
    player_id: str,
    *,
    source_team_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The live consent record for this player (any destination), or None."""
    migrate_consent_store(session)
    entry = _consent_store(session).get(str(player_id))
    if not isinstance(entry, dict) or not bool(entry.get("accepted")):
        return None
    if str(entry.get("status") or "granted") in _INACTIVE_STATUSES:
        return None
    if entry.get("window_key") and str(entry.get("window_key")) != waiver_window(session)["key"]:
        return None
    src = str(entry.get("source_team_id") or "")
    if source_team_id and src and src != str(source_team_id):
        return None
    return entry


def clause_trade_permission(
    player: Any,
    *,
    source_team_id: str,
    destination_team_id: Optional[str],
    waivers: Any = None,
    window_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Is this player's clause satisfied for a move to ``destination_team_id``?

    Returns ``{"protected", "clause_label", "allowed", "waived", "reason", "entry"}``.
    """
    clause = _clause_summary(player)
    label = str(clause.get("label") or "None")
    pid = str(getattr(player, "id", "") or "")
    dest = str(destination_team_id or "")
    approved = [str(t) for t in (clause.get("approved_destinations") or [])]
    out: Dict[str, Any] = {
        "protected": label != "None",
        "clause_label": label,
        "allowed": True,
        "waived": False,
        "reason": "",
        "entry": None,
        "approved_destinations": approved,
    }
    if label == "None":
        return out
    # Modified NTC: approved destinations need no waiver.
    if not clause.get("nmc") and int(clause.get("mntc") or 0) > 0 and _dest_on_list(dest, approved):
        return out
    entry = lookup_consent(waivers, pid, dest)
    if consent_allows(entry, dest, window_key=window_key, source_team_id=source_team_id):
        out.update({"waived": True, "entry": entry if isinstance(entry, dict) else None})
        return out
    out["allowed"] = False
    if clause.get("nmc"):
        out["reason"] = "No-movement clause — ask him to waive it"
    elif clause.get("ntc") and int(clause.get("mntc") or 0) <= 0:
        out["reason"] = "No-trade clause — ask him to waive it"
    else:
        out["reason"] = "Modified no-trade clause — destination not on his list"
    return out


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _reask_after(win: Dict[str, Any]) -> str:
    floor = _add_days(win.get("today_iso") or "", REASK_MIN_DAYS)
    reopen = str(win.get("reopen_iso") or "")
    return max(floor, reopen) if floor and reopen else (floor or reopen)


def _player_entity(session: Any, player_id: str) -> Dict[str, Any]:
    ents = getattr(session, "universe_players", None)
    if isinstance(ents, dict):
        ent = ents.get(str(player_id))
        if isinstance(ent, dict):
            return ent
    return {}


def _apply_mood(session: Any, player_id: str, deltas: Dict[str, float]) -> List[Dict[str, Any]]:
    """Trust / morale cost on the player's universe profile, when one exists."""
    ent = _player_entity(session, player_id)
    if not ent or not deltas:
        return []
    out: List[Dict[str, Any]] = []
    try:
        from app.sim_engine.franchise.storyline_engine import _u_apply_profile_delta

        for key, delta in deltas.items():
            if delta:
                out.append(_u_apply_profile_delta(ent, f"state.{key}", float(delta)))
    except Exception:
        for key, delta in deltas.items():
            state = ent.setdefault("state", {})
            before = float(state.get(key, 55.0) or 55.0)
            after = max(0.0, min(100.0, before + float(delta)))
            state[key] = after
            out.append({"field": f"state.{key}", "before": before, "after": after, "delta": round(after - before, 2)})
    return out


def _broken_promises(session: Any, player_id: str) -> int:
    return sum(
        1
        for p in (getattr(session, "universe_promises", None) or [])
        if isinstance(p, dict)
        and str(p.get("player_id") or "") == str(player_id)
        and str(p.get("status") or "") == "broken"
    )


def _relationship_adjustment(session: Any, player: Any, source_team: Any) -> Tuple[float, List[str]]:
    """How the player's mood toward his current club shifts his willingness to move."""
    pid = str(getattr(player, "id", "") or "")
    ent = _player_entity(session, pid)
    state = dict(ent.get("state") or {})
    life = dict(ent.get("life") or {})
    notes: List[str] = []
    adj = 0.0
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if user_tid and _team_id(source_team) == user_tid and state:
        trust = float(state.get("gm_trust", 55) or 55)
        adj += (trust - 55.0) * 0.004
        if trust >= 65:
            notes.append("trusts management")
        elif trust <= 42:
            notes.append("low trust in management")
    morale = float(state.get("morale", 55) or 55) if state else 55.0
    if morale <= 40:
        adj += 0.06
        notes.append("unhappy where he is")
    role = float(state.get("role_satisfaction", 55) or 55) if state else 55.0
    if role <= 42:
        adj += 0.05
        notes.append("wants a bigger role")
    if float(life.get("community_connection") or 0) >= 60:
        adj -= 0.08
        notes.append("rooted in the community")
    if float(life.get("relocation_strain") or 0) >= 50:
        adj -= 0.05
        notes.append("family does not want to move")
    if _broken_promises(session, pid):
        adj -= 0.12
        notes.append("feels promises were broken")
    if bool(getattr(player, "_trade_demand_active", False) or getattr(player, "trade_demand_active", False)):
        adj += 0.35
        notes.append("has asked out")
    return adj, notes


def _player_age(player: Any) -> int:
    try:
        ident = getattr(player, "identity", None)
        return int(getattr(ident, "age", None) or getattr(player, "age", 28) or 28)
    except Exception:
        return 28


def destination_chance(
    player: Any,
    *,
    source_team: Any,
    destination_team: Any = None,
    context: Optional[Dict[str, Any]] = None,
    rel_adj: float = 0.0,
) -> float:
    """Probability-scale willingness to waive for this destination (None = any fair trade)."""
    ctx = context or {}
    clause = _clause_summary(player)
    src_quality = _team_strength_proxy(source_team, ctx) if source_team is not None else 0.48
    if destination_team is not None:
        dest_quality = _team_strength_proxy(destination_team, ctx)
        dest_market = _market_size(destination_team)
        dest_window = str(getattr(destination_team, "gm_window", None) or getattr(destination_team, "window", "") or "").lower()
    else:
        dest_quality, dest_market, dest_window = 0.5, "medium", ""
    chance = 0.38 + (dest_quality - 0.45) * 0.55
    if dest_market == "large":
        chance += 0.10
    elif dest_market == "small":
        chance -= 0.14
    if "contend" in dest_window:
        chance += 0.12
    if "rebuild" in dest_window or "tank" in dest_window:
        chance -= 0.16
    if dest_quality + 0.08 < src_quality:
        chance -= 0.10
    age = _player_age(player)
    if age >= 33:
        chance -= 0.08
    elif age <= 26:
        chance += 0.04
    chance += float(rel_adj or 0.0)
    if clause.get("nmc"):
        chance *= NMC_CHANCE_SCALE
    return max(0.03, min(0.88, chance))


def preview_waiver_chance(session: Any, player: Any, source_team: Any) -> Dict[str, Any]:
    """Read-only odds for a general (any-destination) waiver ask, with the reasons."""
    adj, notes = _relationship_adjustment(session, player, source_team)
    chance = destination_chance(player, source_team=source_team, destination_team=None, rel_adj=adj)
    return {"chance": round(chance, 3), "notes": notes}


def ask_status(
    session: Any,
    player_id: str,
    destination_team_id: Optional[str] = None,
    *,
    source_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Can the GM ask this player (again) right now? Explains why not."""
    migrate_consent_store(session)
    win = waiver_window(session)
    pid = str(player_id or "")
    dest = str(destination_team_id or "") or None
    entry = active_consent(session, pid, source_team_id=source_team_id)
    if entry is not None and (dest is None or consent_allows(entry, dest, window_key=win["key"], source_team_id=source_team_id)):
        return {
            "can_ask": False,
            "status": "granted",
            "reason": f"Already agreed — waiver on file until {entry.get('expires_label') or win['ends_label']}",
            "reask_after_iso": None,
            "window_key": win["key"],
        }
    book = _ask_store(session).get(pid) or {}
    hist = list(book.get("history") or [])
    today = str(win.get("today_iso") or "")
    in_window = [h for h in hist if str(h.get("window_key") or "") == win["key"]]
    declines_now = [h for h in in_window if not h.get("accepted")]
    blocking: List[Dict[str, Any]] = []
    for h in hist:
        if h.get("accepted"):
            continue
        h_dest = str(h.get("destination_team_id") or "") or None
        same_question = h_dest is None or dest is None or h_dest == dest
        same_window = str(h.get("window_key") or "") == win["key"]
        reask = str(h.get("reask_after_iso") or "")
        too_soon = bool(reask and today and today < reask)
        if same_question and (same_window or too_soon):
            blocking.append(h)
    if len(declines_now) >= MAX_DECLINES_PER_WINDOW and not blocking:
        blocking = declines_now
    if blocking:
        reask = max(str(h.get("reask_after_iso") or "") for h in blocking) or win["reopen_iso"]
        return {
            "can_ask": False,
            "status": "declined",
            "reason": f"Already asked — he said no (re-ask after {format_iso_label(reask)})",
            "reask_after_iso": reask,
            "reask_after_label": format_iso_label(reask),
            "window_key": win["key"],
        }
    return {
        "can_ask": True,
        "status": "none",
        "reason": "",
        "reask_after_iso": None,
        "window_key": win["key"],
        "asks_this_window": len(in_window),
    }


def _window_snapshot(session: Any, player: Any, source_team: Any, win: Dict[str, Any]) -> Dict[str, Any]:
    """Seeded roll + mood snapshot, fixed at the first ask of the window."""
    pid = str(getattr(player, "id", "") or "")
    book = _ask_store(session).setdefault(pid, {"history": [], "windows": {}})
    wins = book.setdefault("windows", {})
    snap = wins.get(win["key"])
    if isinstance(snap, dict) and "roll" in snap:
        return snap
    rel_adj, notes = _relationship_adjustment(session, player, source_team)
    snap = {
        "roll": round(_stable_unit_roll(f"clause-waive|{pid}|{win['season_year']}|{win['key']}"), 6),
        "rel_adj": round(rel_adj, 4),
        "notes": notes,
        "asks": 0,
    }
    wins[win["key"]] = snap
    return snap


_ACCEPT_REASONS = (
    ("contend", "Sees a better chance to win elsewhere"),
    ("fresh_start", "Open to a fresh start"),
    ("big_market", "Attracted to the destination market"),
    ("term_left", "Wants to play out his deal somewhere that values him"),
)
_DECLINE_REASONS = (
    ("desire_to_stay", "Wants to stay with his current club"),
    ("family", "Family situation — not prepared to relocate"),
    ("direction", "Unconvinced about the destination's direction"),
)


def _reason(accepted: bool, roll: float, destination_team: Any, notes: List[str]) -> Tuple[str, str]:
    pool = _ACCEPT_REASONS if accepted else _DECLINE_REASONS
    code, text = pool[int(roll * 1000) % len(pool)]
    if destination_team is not None:
        window = str(getattr(destination_team, "gm_window", None) or "").lower()
        market = _market_size(destination_team)
        if accepted and "contend" in window:
            code, text = "contend", "Sees a better chance to win with the destination"
        elif accepted and market == "large":
            code, text = "big_market", "Attracted to the destination market"
        elif not accepted and ("rebuild" in window or "tank" in window):
            code, text = "team_bad", "Not interested in joining a rebuilding club"
        elif not accepted and market == "small":
            code, text = "small_market", "Does not want to move to a small market"
    if notes:
        text = f"{text} ({notes[0]})"
    return code, text


def request_clause_waiver(
    session: Any,
    player: Any,
    *,
    source_team: Any,
    destination_team: Any = None,
    origin: str = "trade_hub",
    team_by_id: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    apply_mood: bool = True,
) -> Dict[str, Any]:
    """Ask a protected player to waive. Deterministic per player + window; never re-rolls.

    ``destination_team=None`` is a general ask (meetings): yes means any team for a full
    NTC / NMC, or a list of acceptable teams for a modified NTC.
    """
    migrate_consent_store(session)
    win = waiver_window(session)
    pid = str(getattr(player, "id", "") or "")
    pname = player_display_name(player)
    src_id = _team_id(source_team)
    dest_id = _team_id(destination_team) if destination_team is not None else None
    clause = _clause_summary(player)
    label = str(clause.get("label") or "None")
    base = {
        "player_id": pid,
        "player_name": pname,
        "clause_label": label,
        "source_team_id": src_id,
        "destination_team_id": dest_id,
        "window_key": win["key"],
        "origin": origin,
    }
    if label == "None":
        return {**base, "ok": True, "accepted": True, "can_request": False, "reason": "No trade protection — no waiver required.", "reason_code": "no_ntc", "accept_chance": 1.0, "value_penalty_pct": 0.0}
    approved = [str(t) for t in (clause.get("approved_destinations") or [])]
    if dest_id and not clause.get("nmc") and int(clause.get("mntc") or 0) > 0 and _dest_on_list(dest_id, approved):
        return {**base, "ok": True, "accepted": True, "can_request": False, "reason": "Destination is already on his approved list.", "reason_code": "mntc_approved", "accept_chance": 1.0, "value_penalty_pct": 0.0}

    status = ask_status(session, pid, dest_id, source_team_id=src_id)
    if not status["can_ask"]:
        entry = active_consent(session, pid, source_team_id=src_id) if status["status"] == "granted" else None
        return {
            **base,
            "ok": status["status"] == "granted",
            "accepted": status["status"] == "granted",
            "blocked": status["status"] != "granted",
            "cached": True,
            "can_request": False,
            "reason": status["reason"],
            "reason_code": "already_granted" if status["status"] == "granted" else "already_asked",
            "reask_after_iso": status.get("reask_after_iso"),
            "reask_after_label": status.get("reask_after_label"),
            "consent": dict(entry) if entry else None,
            "value_penalty_pct": float((entry or {}).get("value_penalty_pct") or 0.0),
        }

    ctx = dict(context or {})
    snap = _window_snapshot(session, player, source_team, win)
    dest_id = _team_id(destination_team)
    roll = round(_stable_unit_roll(f"clause-waive|{pid}|{win['season_year']}|{win['key']}|{dest_id}"), 6)
    rel_adj = float(snap.get("rel_adj") or 0.0)
    chance = destination_chance(player, source_team=source_team, destination_team=destination_team, context=ctx, rel_adj=rel_adj)
    accepted = roll < chance
    code, reason = _reason(accepted, roll, destination_team, list(snap.get("notes") or []))

    # Mood cost — every ask spends goodwill; repeated asks in a window spend more.
    asks_before = int(snap.get("asks") or 0)
    snap["asks"] = asks_before + 1
    mood: Dict[str, float] = {"gm_trust": -ASK_TRUST_COST - (REPEAT_ASK_TRUST_COST * asks_before)}
    if accepted:
        mood["gm_trust"] += ACCEPT_TRUST_GAIN
    else:
        mood["gm_trust"] -= DECLINE_TRUST_COST
        mood["morale"] = -DECLINE_MORALE_COST
    mood_receipts = _apply_mood(session, pid, mood) if apply_mood else []

    reask = None if accepted else _reask_after(win)
    _ask_store(session).setdefault(pid, {"history": [], "windows": {}}).setdefault("history", []).append({
        "window_key": win["key"],
        "season_year": win["season_year"],
        "iso": win["today_iso"],
        "cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "destination_team_id": dest_id,
        "accepted": bool(accepted),
        "origin": origin,
        "reason": reason,
        "reask_after_iso": reask,
    })

    consent: Optional[Dict[str, Any]] = None
    if accepted:
        store = _consent_store(session)
        prev = active_consent(session, pid, source_team_id=src_id)
        if dest_id:
            allowed = list((prev or {}).get("allowed_team_ids") or [])
            if prev is not None and not allowed:
                scope, allowed = "any", []
            else:
                scope = "teams"
                if dest_id not in allowed:
                    allowed.append(dest_id)
        elif int(clause.get("mntc") or 0) > 0 and not clause.get("nmc"):
            # Modified NTC (a list clause): a general yes names the clubs he would accept.
            scope, allowed = "teams", _acceptable_teams(player, source_team, team_by_id or ctx.get("team_by_id") or {}, ctx, roll, rel_adj, approved)
        else:
            scope, allowed = "any", []
        consent = {
            "player_id": pid,
            "player_name": pname,
            "clause_label": label,
            "accepted": True,
            "status": "granted",
            "scope": scope,
            "allowed_team_ids": allowed,
            "destination_team_id": allowed[0] if scope == "teams" and len(allowed) == 1 else None,
            "source_team_id": src_id,
            "season_year": win["season_year"],
            "window_key": win["key"],
            "granted_iso": win["today_iso"],
            "expires_iso": win["ends_iso"],
            "expires_label": win["ends_label"],
            "origin": origin,
            "reason": reason,
            "reason_code": code,
            "value_penalty_pct": WAIVED_VALUE_PENALTY_PCT,
        }
        store[pid] = consent
    _bump_revision(session)

    return {
        **base,
        "ok": True,
        "accepted": bool(accepted),
        "can_request": True,
        "cached": False,
        "blocked": False,
        "reason": reason,
        "reason_code": code,
        "accept_chance": round(chance, 3),
        "roll": round(roll, 4),
        "consent": dict(consent) if consent else None,
        "reask_after_iso": reask,
        "reask_after_label": format_iso_label(reask) if reask else None,
        "expires_iso": win["ends_iso"] if accepted else None,
        "expires_label": win["ends_label"] if accepted else None,
        "mood": mood_receipts,
        "value_penalty_pct": WAIVED_VALUE_PENALTY_PCT if accepted else 0.0,
        "value_note": (
            f"Clause waived until {win['ends_label']} — trade value slightly reduced"
            if accepted
            else f"{label} stays in force — re-ask after {format_iso_label(reask)}"
        ),
    }


def _acceptable_teams(
    player: Any,
    source_team: Any,
    team_by_id: Dict[str, Any],
    ctx: Dict[str, Any],
    roll: float,
    rel_adj: float,
    approved: List[str],
) -> List[str]:
    """Clubs a modified-NTC player would accept this window (his list), best first."""
    src_id = _team_id(source_team)
    scored: List[Tuple[float, str]] = []
    for tid, team in (team_by_id or {}).items():
        if str(tid) == src_id:
            continue
        scored.append((destination_chance(player, source_team=source_team, destination_team=team, context=ctx, rel_adj=rel_adj), str(tid)))
    scored.sort(reverse=True)
    pid = str(getattr(player, "id", "") or "")
    season = int(ctx.get("season_year", 2025) or 2025)
    window_key = str(ctx.get("clause_window_key") or f"{season}-in")
    ok = []
    for ch, tid in scored:
        team_roll = _stable_unit_roll(f"clause-waive|{pid}|{season}|{window_key}|{tid}")
        if team_roll < ch:
            ok.append(tid)
    out = list(approved)
    for tid in ok:
        if tid not in out:
            out.append(tid)
    return out


def consent_status_for_player(
    session: Any,
    player: Any,
    *,
    source_team_id: str,
    destination_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """UI row: clause, consent status, expiry, and whether the Ask button is live."""
    clause = _clause_summary(player)
    label = str(clause.get("label") or "None")
    if label == "None":
        return {"clause_label": "None", "status": "none", "can_ask": False}
    pid = str(getattr(player, "id", "") or "")
    win = waiver_window(session)
    entry = active_consent(session, pid, source_team_id=source_team_id)
    status = ask_status(session, pid, destination_team_id, source_team_id=source_team_id)
    out: Dict[str, Any] = {
        "clause_label": label,
        "status": status["status"],
        "can_ask": bool(status["can_ask"]),
        "ask_block_reason": status.get("reason") or "",
        "reask_after_iso": status.get("reask_after_iso"),
        "reask_after_label": status.get("reask_after_label"),
        "window_key": win["key"],
        "window_label": win["label"],
    }
    if entry is not None:
        allowed = list(entry.get("allowed_team_ids") or [])
        allows_dest = consent_allows(entry, destination_team_id, window_key=win["key"], source_team_id=source_team_id) if destination_team_id else True
        out.update({
            "status": "granted" if allows_dest else "granted_other",
            "scope": entry.get("scope") or ("teams" if allowed else "any"),
            "allowed_team_ids": allowed,
            "expires_iso": entry.get("expires_iso"),
            "expires_label": entry.get("expires_label"),
            "origin": entry.get("origin"),
            "reason": entry.get("reason") or "",
            "summary": (
                f"Agreed to waive his {label} until {entry.get('expires_label') or win['ends_label']}"
                + ("" if not allowed else f" — {len(allowed)} approved club{'s' if len(allowed) != 1 else ''}")
            ),
        })
        if not allows_dest:
            out["can_ask"] = bool(status["can_ask"])
    elif status["status"] == "declined":
        out["summary"] = status["reason"]
    return out
