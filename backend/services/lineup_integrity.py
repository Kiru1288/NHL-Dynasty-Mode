"""Lineup integrity: validate, reconcile and auto-fill an Edit Lines payload.

The pure helpers (``evaluate_lineup``, ``reconcile_lines``, ``empty_lines``…)
take a roster list and a lines dict, so any club or league lineup (the AHL
lineup included) can reuse them. The ``*_user_*`` helpers bind them to the
user's NHL club and ``session.lines["even_strength"]``.

Lines payload shape (same as the Edit Lines screen and the sim engine)::

    {"forwards": [{"id": "f1", "name": "Line 1", "slots": {"LW": pid, "C": pid, "RW": pid}}, ...x4],
     "defense":  [{"id": "d1", "name": "Pair 1", "slots": {"LD": pid, "RD": pid}}, ...x3],
     "goalies":  [{"id": "g1", "name": "Goalies", "slots": {"Starter": pid, "Backup": pid}}]}

A *hole* is a required slot the sim cannot dress: blank, or holding a player
who is no longer on the roster (sent down, traded, waived, retired) or who is
listed twice. Injured / suspended players in a slot are not holes — the sim
engine covers them from healthy scratches — but they are reported.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

FORWARD_SLOTS: Tuple[str, ...] = ("LW", "C", "RW")
DEFENSE_SLOTS: Tuple[str, ...] = ("LD", "RD")
GOALIE_SLOTS: Tuple[str, ...] = ("Starter", "Backup")
OPTIONAL_GOALIE_SLOTS: Tuple[str, ...] = ("Third",)

DEFAULT_STRUCTURE: Dict[str, int] = {"forwards": 4, "defense": 3}

BUCKET_LABEL = {"F": "forward", "D": "defenseman", "G": "goalie"}


# ---------------------------------------------------------------------------
# Player helpers
# ---------------------------------------------------------------------------

def canonical_id(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    return f"NHL_{s}" if s.isdigit() else s


def player_key(player: Any) -> str:
    try:
        from app.sim_engine.systems.chemistry import _canonical_player_id  # noqa: WPS433

        pid = _canonical_player_id(player)
        if pid:
            return str(pid)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return canonical_id(getattr(player, "id", "") or "")


def player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "name", None) or getattr(player, "name", None) or "Player")


def position_bucket(player: Any) -> str:
    from services.roster_compliance import position_bucket as _bucket

    b = _bucket(player)
    if b in ("F", "D", "G"):
        return b
    ident = getattr(player, "identity", None)
    raw = str(getattr(getattr(ident, "position", None), "value", getattr(ident, "position", "")) or "").upper()
    raw = raw.split(".")[-1]
    if raw in ("D", "LD", "RD"):
        return "D"
    if raw == "G":
        return "G"
    if raw in ("C", "LW", "RW", "F", "W"):
        return "F"
    return "OTHER"


def position_code(player: Any) -> str:
    from services.roster_compliance import position_code as _code

    code = _code(player).split(".")[-1]
    if not code or code == "?":
        ident = getattr(player, "identity", None)
        pos = getattr(ident, "position", None)
        code = str(getattr(pos, "value", pos) or "").upper().split(".")[-1]
    return code


def slot_bucket(slot: str) -> str:
    s = str(slot or "")
    if s in FORWARD_SLOTS:
        return "F"
    if s in DEFENSE_SLOTS:
        return "D"
    if s in GOALIE_SLOTS or s in OPTIONAL_GOALIE_SLOTS:
        return "G"
    return "OTHER"


def player_ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


def is_dressable(player: Any) -> bool:
    """On the club's NHL list and not assigned to the minors / retired."""
    from services.roster_compliance import is_buried_or_minors, is_retired

    return player is not None and not is_retired(player) and not is_buried_or_minors(player)


def is_injured(player: Any) -> bool:
    """Same rule the sim engine uses to sideline a player for a game."""
    try:
        if int(getattr(player, "_world_injury_games_remaining", 0) or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    health = getattr(player, "health", None)
    status = getattr(health, "injury_status", None) if health is not None else None
    if status is None:
        return False
    name = str(getattr(status, "name", status) or "").upper().split(".")[-1]
    return bool(name) and name != "HEALTHY"


def is_suspended(player: Any) -> bool:
    try:
        from app.sim_engine.franchise.storyline_conduct import is_under_conduct_suspension  # noqa: WPS433

        return bool(is_under_conduct_suspension(player))
    except Exception:
        return False


def is_available(player: Any) -> bool:
    return is_dressable(player) and not is_injured(player) and not is_suspended(player)


def fits_slot(player: Any, slot: str) -> bool:
    b = slot_bucket(slot)
    if b == "OTHER":
        return False
    pb = position_bucket(player)
    if pb == "OTHER":
        # Unknown position data: never strip a skater the user placed.
        return b in ("F", "D")
    return pb == b


def _slot_fit_bonus(player: Any, slot: str) -> float:
    """Natural position edge in OVR points (mirrors Edit Lines' position-safe build:
    a winger at centre is a real downgrade, off-wing is a small one)."""
    code = position_code(player)
    if code == slot:
        return 10.0
    if slot in ("LW", "RW") and code in ("LW", "RW", "W"):
        return 7.0
    if slot in ("LW", "RW") and code == "C":
        return 5.0
    if slot in ("LD", "RD") and code in ("D", "LD", "RD"):
        return 8.0
    return 0.0


# ---------------------------------------------------------------------------
# Lines payload helpers (pure)
# ---------------------------------------------------------------------------

def empty_lines(structure: Optional[Dict[str, int]] = None, include_third: bool = False) -> Dict[str, Any]:
    st = dict(DEFAULT_STRUCTURE, **(structure or {}))
    gslots = {slot: "" for slot in GOALIE_SLOTS}
    if include_third:
        gslots["Third"] = ""
    return {
        "forwards": [
            {"id": f"f{n}", "name": f"Line {n}", "slots": {slot: "" for slot in FORWARD_SLOTS}}
            for n in range(1, int(st["forwards"]) + 1)
        ],
        "defense": [
            {"id": f"d{n}", "name": f"Pair {n}", "slots": {slot: "" for slot in DEFENSE_SLOTS}}
            for n in range(1, int(st["defense"]) + 1)
        ],
        "goalies": [{"id": "g1", "name": "Goalies", "slots": gslots}],
    }


def has_any_assignment(lines: Any) -> bool:
    if not isinstance(lines, dict):
        return False
    for group in ("forwards", "defense", "goalies"):
        for unit in lines.get(group) or []:
            if isinstance(unit, dict) and any(str(v or "").strip() for v in (unit.get("slots") or {}).values()):
                return True
    return False


def normalize_lines(lines: Any, structure: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Deep copy with every required unit / slot present (extra unit keys are kept)."""
    src = lines if isinstance(lines, dict) else {}
    include_third = any(
        isinstance(u, dict) and "Third" in (u.get("slots") or {})
        for u in (src.get("goalies") or [])
    )
    base = empty_lines(structure, include_third=include_third)
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in src.items() if k not in ("forwards", "defense", "goalies")}
    for group in ("forwards", "defense", "goalies"):
        incoming = [u for u in (src.get(group) or []) if isinstance(u, dict)]
        units = []
        for index, fallback in enumerate(base[group]):
            unit = next((u for u in incoming if str(u.get("id") or "") == fallback["id"]), None)
            if unit is None and index < len(incoming):
                unit = incoming[index]
            unit = copy.deepcopy(unit or {})
            slots_in = dict(unit.get("slots") or {})
            slots = {slot: str(slots_in.get(slot) or "").strip() for slot in fallback["slots"]}
            units.append({**unit, "id": fallback["id"], "name": unit.get("name") or fallback["name"], "slots": slots})
        out[group] = units
    return out


def slot_label(group: str, index: int, slot: str) -> str:
    if group == "forwards":
        return f"Line {index} {slot}"
    if group == "defense":
        return f"Pair {index} {slot}"
    return f"Goalie {slot}"


def iter_slots(lines: Dict[str, Any], *, required_only: bool = True) -> Iterable[Dict[str, Any]]:
    for group in ("forwards", "defense", "goalies"):
        for index, unit in enumerate(lines.get(group) or [], start=1):
            if not isinstance(unit, dict):
                continue
            for slot, pid in (unit.get("slots") or {}).items():
                if required_only and slot in OPTIONAL_GOALIE_SLOTS:
                    continue
                yield {
                    "group": group,
                    "index": index,
                    "unit_id": str(unit.get("id") or ""),
                    "slot": slot,
                    "player_id": str(pid or "").strip(),
                    "label": slot_label(group, index, slot),
                    "key": f"{group}:{unit.get('id')}:{slot}",
                    "bucket": slot_bucket(slot),
                }


def _set_slot(lines: Dict[str, Any], group: str, unit_id: str, slot: str, value: str) -> None:
    for unit in lines.get(group) or []:
        if isinstance(unit, dict) and str(unit.get("id") or "") == unit_id:
            slots = dict(unit.get("slots") or {})
            slots[slot] = str(value or "")
            unit["slots"] = slots
            return


def index_roster(roster: Sequence[Any]) -> Dict[str, Any]:
    """Map raw / canonical / bare-numeric ids -> player, dressable players only."""
    out: Dict[str, Any] = {}
    for p in roster or []:
        if not is_dressable(p):
            continue
        raw = str(getattr(p, "id", "") or "").strip()
        canon = player_key(p)
        for key in (raw, canon, canonical_id(raw), canon.replace("NHL_", "", 1) if canon.startswith("NHL_") else ""):
            if key and key not in out:
                out[key] = p
    return out


def lookup_player(index: Dict[str, Any], raw: Any) -> Optional[Any]:
    s = str(raw or "").strip()
    if not s:
        return None
    hit = index.get(s) or index.get(canonical_id(s))
    if hit is None and s.startswith("NHL_"):
        hit = index.get(s.replace("NHL_", "", 1))
    return hit


def _name_for_unknown(name_lookup: Optional[Dict[str, str]], pid: str) -> str:
    if not name_lookup:
        return ""
    return str(name_lookup.get(pid) or name_lookup.get(canonical_id(pid)) or "")


def evaluate_lineup(
    roster: Sequence[Any],
    lines: Any,
    *,
    structure: Optional[Dict[str, int]] = None,
    name_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Read-only check of a lines payload against a roster."""
    payload = normalize_lines(lines, structure)
    idx = index_roster(roster)
    gaps: List[Dict[str, Any]] = []
    injured: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    assigned: set = set()
    for row in iter_slots(payload):
        pid = row["player_id"]
        if not pid:
            gaps.append({**row, "reason": "empty", "player_name": ""})
            continue
        player = lookup_player(idx, pid)
        if player is None:
            gaps.append({**row, "reason": "not_on_roster", "player_name": _name_for_unknown(name_lookup, pid)})
            continue
        key = player_key(player)
        if key in seen:
            gaps.append({**row, "reason": "duplicate", "player_name": player_name(player)})
            continue
        seen[key] = row["label"]
        assigned.add(key)
        if not fits_slot(player, row["slot"]):
            gaps.append({**row, "reason": "wrong_position", "player_name": player_name(player)})
            continue
        if is_injured(player) or is_suspended(player):
            injured.append({
                "label": row["label"],
                "player_id": key,
                "player_name": player_name(player),
                "status": "injured" if is_injured(player) else "suspended",
            })

    extras: Dict[str, List[Any]] = {"F": [], "D": [], "G": []}
    for p in idx.values():
        key = player_key(p)
        if key in assigned or not is_available(p):
            continue
        b = position_bucket(p)
        if b in extras and p not in extras[b]:
            extras[b].append(p)

    need: Dict[str, int] = {"F": 0, "D": 0, "G": 0}
    for gap in gaps:
        if gap["bucket"] in need:
            need[gap["bucket"]] += 1
    shortage = {b: max(0, need[b] - len(extras[b])) for b in need}
    unfillable = []
    remaining = dict(shortage)
    for gap in reversed(gaps):
        b = gap["bucket"]
        if remaining.get(b, 0) > 0:
            unfillable.append(gap["label"])
            remaining[b] -= 1
    unfillable.reverse()

    return {
        "has_lineup": has_any_assignment(payload),
        "complete": not gaps,
        "gaps": gaps,
        "gap_labels": [g["label"] for g in gaps],
        "injured": injured,
        "available_extras": {b: len(v) for b, v in extras.items()},
        "shortage": shortage,
        "fillable": not any(shortage.values()),
        "unfillable": unfillable,
    }


def _best_candidate(pool: List[Any], slot: str, prefer: Sequence[str] = ()) -> Optional[Any]:
    fits = [p for p in pool if fits_slot(p, slot)]
    if not fits:
        return None
    pref = set(prefer or ())
    return max(
        fits,
        key=lambda p: (player_key(p) in pref, player_ovr(p) + _slot_fit_bonus(p, slot), player_name(p)),
    )


def reconcile_lines(
    roster: Sequence[Any],
    lines: Any,
    *,
    auto_fill: bool = False,
    prefer_ids: Sequence[str] = (),
    structure: Optional[Dict[str, int]] = None,
    name_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (new_lines, report).

    1. Blank slots holding players no longer on the roster, duplicates and
       position misfits (reported as ``vacated``).
    2. Place ``prefer_ids`` (call-ups / acquisitions) into empty slots of their
       position (reported as ``placed``).
    3. With ``auto_fill``, fill the remaining empty slots with the best healthy
       unassigned player of the right position (reported as ``filled``).
    """
    payload = normalize_lines(lines, structure)
    before = copy.deepcopy(payload)
    idx = index_roster(roster)
    vacated: List[Dict[str, Any]] = []
    seen: set = set()

    for row in iter_slots(payload, required_only=False):
        pid = row["player_id"]
        if not pid:
            continue
        player = lookup_player(idx, pid)
        reason = ""
        if player is None:
            reason = "not_on_roster"
        elif player_key(player) in seen:
            reason = "duplicate"
        elif not fits_slot(player, row["slot"]):
            reason = "wrong_position"
        if reason:
            _set_slot(payload, row["group"], row["unit_id"], row["slot"], "")
            if row["slot"] not in OPTIONAL_GOALIE_SLOTS:
                vacated.append({
                    "label": row["label"],
                    "player_id": pid,
                    "player_name": player_name(player) if player is not None else _name_for_unknown(name_lookup, pid),
                    "reason": reason,
                })
            continue
        key = player_key(player)
        seen.add(key)
        if key != pid:
            _set_slot(payload, row["group"], row["unit_id"], row["slot"], key)

    def _pool(only: Optional[set] = None) -> List[Any]:
        out = []
        for p in idx.values():
            key = player_key(p)
            if key in seen or p in out or not is_available(p):
                continue
            if only is not None and key not in only:
                continue
            out.append(p)
        return out

    placed: List[Dict[str, Any]] = []
    prefer = {canonical_id(x) for x in (prefer_ids or ()) if x}
    prefer |= {player_key(lookup_player(idx, x)) for x in (prefer_ids or ()) if lookup_player(idx, x) is not None}
    if prefer:
        newcomers = sorted(_pool(prefer), key=lambda p: -player_ovr(p))
        for p in newcomers:
            empties = [r for r in iter_slots(payload) if not r["player_id"] and fits_slot(p, r["slot"])]
            if not empties:
                continue
            # Top-most open unit first; natural position breaks ties inside it.
            top_index = min(r["index"] for r in empties)
            same_unit = [r for r in empties if r["index"] == top_index]
            target = max(same_unit, key=lambda r: _slot_fit_bonus(p, r["slot"]))
            key = player_key(p)
            _set_slot(payload, target["group"], target["unit_id"], target["slot"], key)
            seen.add(key)
            placed.append({"label": target["label"], "player_id": key, "player_name": player_name(p)})

    filled: List[Dict[str, Any]] = []
    if auto_fill:
        for row in list(iter_slots(payload)):
            if row["player_id"]:
                continue
            pick = _best_candidate(_pool(), row["slot"], prefer=tuple(prefer))
            if pick is None:
                continue
            key = player_key(pick)
            _set_slot(payload, row["group"], row["unit_id"], row["slot"], key)
            seen.add(key)
            filled.append({"label": row["label"], "player_id": key, "player_name": player_name(pick)})

    report = evaluate_lineup(roster, payload, structure=structure, name_lookup=name_lookup)
    report.update({
        "vacated": vacated,
        "placed": placed,
        "filled": filled,
        "changed": payload != before,
    })
    return payload, report


# ---------------------------------------------------------------------------
# Session bindings (user's NHL club, even-strength sheet)
# ---------------------------------------------------------------------------

def user_team(session: Any) -> Any:
    tid = str(getattr(session, "user_team_id", "") or "")
    by_id = getattr(session, "team_by_id", None) or {}
    return by_id.get(tid) or by_id.get(getattr(session, "user_team_id", None))


def user_roster(session: Any) -> List[Any]:
    team = user_team(session)
    return list(getattr(team, "roster", None) or []) if team is not None else []


def saved_even_strength(session: Any) -> Optional[Dict[str, Any]]:
    """The user's saved even-strength sheet, or None when the club runs on auto lines."""
    root = getattr(session, "lines", None)
    if not isinstance(root, dict):
        return None
    block = root.get("even_strength")
    if not isinstance(block, dict):
        return None
    inner = block.get("lines") if isinstance(block.get("lines"), dict) else block
    if not isinstance(inner, dict):
        return None
    if not (inner.get("forwards") or inner.get("defense") or inner.get("goalies")):
        return None
    return inner


def _org_name_lookup(session: Any) -> Dict[str, str]:
    """Names for ids that left the NHL list (minors / prospects) so vacated slots read well."""
    team = user_team(session)
    out: Dict[str, str] = {}
    for attr in ("ahl_roster", "echl_roster", "prospect_pool", "roster"):
        for p in list(getattr(team, attr, None) or []):
            raw = str(getattr(p, "id", "") or "")
            if raw:
                out[raw] = player_name(p)
                out[canonical_id(raw)] = player_name(p)
                out[player_key(p)] = player_name(p)
    return out


def _roster_snapshot(session: Any) -> List[str]:
    return sorted({player_key(p) for p in user_roster(session) if is_dressable(p) and player_key(p)})


def evaluate_user_lineup(session: Any) -> Dict[str, Any]:
    lines = saved_even_strength(session)
    if lines is None:
        return {
            "has_lineup": False,
            "complete": True,
            "gaps": [],
            "gap_labels": [],
            "injured": [],
            "available_extras": {},
            "shortage": {"F": 0, "D": 0, "G": 0},
            "fillable": True,
            "unfillable": [],
        }
    return evaluate_lineup(user_roster(session), lines, name_lookup=_org_name_lookup(session))


def reconcile_user_lineup(
    session: Any,
    *,
    auto_fill: bool = False,
    prefer_ids: Sequence[str] = (),
    reason: str = "",
) -> Dict[str, Any]:
    """Bring the saved even-strength sheet in line with the NHL roster.

    Players who joined the NHL roster since the last reconcile (call-ups,
    signings, trade acquisitions) are placed into open slots automatically.
    Mutates ``session.lines`` only when something changed.
    """
    lines = saved_even_strength(session)
    snapshot_now = _roster_snapshot(session)
    if lines is None:
        session._lineup_roster_snapshot = snapshot_now
        return {**evaluate_user_lineup(session), "vacated": [], "placed": [], "filled": [], "changed": False}

    prior = getattr(session, "_lineup_roster_snapshot", None)
    newcomers = [pid for pid in snapshot_now if prior is not None and pid not in set(prior)]
    prefer = list(dict.fromkeys([*(prefer_ids or ()), *newcomers]))
    new_lines, report = reconcile_lines(
        user_roster(session),
        lines,
        auto_fill=auto_fill,
        prefer_ids=prefer,
        name_lookup=_org_name_lookup(session),
    )
    session._lineup_roster_snapshot = snapshot_now
    if report.get("changed"):
        block = dict((session.lines or {}).get("even_strength") or {})
        block["lines"] = new_lines
        log = list(block.get("integrity_log") or [])[-11:]
        log.append({
            "reason": str(reason or ""),
            "vacated": report.get("vacated") or [],
            "placed": report.get("placed") or [],
            "filled": report.get("filled") or [],
        })
        block["integrity_log"] = log
        block.setdefault("source", "user")
        session.lines["even_strength"] = block
        mark_user_lines_changed(session)
    return report


def user_lineup_slot_for(session: Any, player: Any) -> Optional[str]:
    """Label of the saved even-strength slot this player holds ("Line 3 LW"), if any."""
    lines = saved_even_strength(session)
    if lines is None or player is None:
        return None
    ids = {player_key(player), canonical_id(getattr(player, "id", "")), str(getattr(player, "id", "") or "")}
    ids.discard("")
    for row in iter_slots(normalize_lines(lines), required_only=False):
        if row["player_id"] in ids:
            return row["label"]
    return None


def mark_user_lines_changed(session: Any) -> None:
    """Bust caches that read the saved sheet (lineup flags, chemistry report)."""
    session._lines_revision = int(getattr(session, "_lines_revision", 0) or 0) + 1
    session._cached_chemistry_report = None


def note_lines_saved(session: Any) -> None:
    """Call after the user saves lines: the current roster is the new baseline."""
    session._lineup_roster_snapshot = _roster_snapshot(session)
    mark_user_lines_changed(session)


def describe_gaps(report: Dict[str, Any], limit: int = 6) -> str:
    labels = list(report.get("gap_labels") or [])
    if not labels:
        return ""
    head = ", ".join(labels[:limit]) + ("…" if len(labels) > limit else "")
    noun = "slot" if len(labels) == 1 else "slots"
    msg = f"Your lineup has {len(labels)} empty {noun} ({head})."
    short = {b: n for b, n in (report.get("shortage") or {}).items() if n}
    if short:
        parts = [f"{n} {BUCKET_LABEL.get(b, b)}{'' if n == 1 else 's'}" for b, n in short.items()]
        msg += f" No healthy extra to fill them — call up {', '.join(parts)} from the AHL."
    else:
        msg += " Fill them in Edit Lines before simulating."
    return msg


def emergency_recall_for_shortage(session: Any, shortage: Dict[str, int]) -> List[Dict[str, Any]]:
    """Auto-sim only: recall the best healthy affiliate SPC for each missing position.

    Respects the 23-man active limit; never demotes anyone.
    """
    from services.contract_economy import _recall_affiliate_player, uses_nhl_contract_slot
    from services.roster_compliance import ACTIVE_ROSTER_MAX, summarize_team_roster_capacity

    team = user_team(session)
    league = getattr(getattr(session, "sim", None), "league", None)
    if team is None:
        return []
    recalled: List[Dict[str, Any]] = []
    for bucket in ("G", "D", "F"):
        for _ in range(int(shortage.get(bucket) or 0)):
            if int(summarize_team_roster_capacity(team).get("nhl_count") or 0) >= ACTIVE_ROSTER_MAX:
                return recalled
            pool = []
            for attr in ("ahl_roster", "echl_roster"):
                for p in list(getattr(team, attr, None) or []):
                    if getattr(p, "retired", False) or position_bucket(p) != bucket:
                        continue
                    if is_injured(p) or is_suspended(p):
                        continue
                    try:
                        if not uses_nhl_contract_slot(p):
                            continue
                    except Exception:
                        continue
                    pool.append((attr, p))
            if not pool:
                break
            attr, pick = max(pool, key=lambda row: player_ovr(row[1]))
            if not _recall_affiliate_player(team, pick, league, attr):
                break
            recalled.append({"player_id": player_key(pick), "player_name": player_name(pick), "bucket": bucket, "from": attr})
    return recalled


def ensure_two_goalies(session: Any, *, allow_recall: bool) -> Dict[str, Any]:
    """An NHL club must dress two goalies. Returns {ok, healthy, recalled, message}.

    With ``allow_recall`` the best healthy affiliate goalie is called up; if the active
    roster is full, the lowest-rated skater is sent down to make the spot.
    """
    team = user_team(session)
    if team is None:
        return {"ok": True, "healthy": 2, "recalled": []}

    def _healthy_goalies() -> List[Any]:
        out = []
        for p in list(getattr(team, "roster", None) or []):
            if position_bucket(p) != "G" or getattr(p, "retired", False):
                continue
            if is_injured(p) or is_suspended(p) or getattr(p, "in_minors", False):
                continue
            out.append(p)
        return out

    healthy = _healthy_goalies()
    recalled: List[Dict[str, Any]] = []
    if len(healthy) < 2 and allow_recall:
        from services.contract_economy import _recall_affiliate_player
        from services.roster_compliance import ACTIVE_ROSTER_MAX, summarize_team_roster_capacity

        league = getattr(getattr(session, "sim", None), "league", None)
        for _ in range(2 - len(healthy)):
            pool = [
                (attr, p)
                for attr in ("ahl_roster", "echl_roster")
                for p in list(getattr(team, attr, None) or [])
                if position_bucket(p) == "G" and not getattr(p, "retired", False) and not is_injured(p) and not is_suspended(p)
            ]
            if not pool:
                break
            if int(summarize_team_roster_capacity(team).get("nhl_count") or 0) >= ACTIVE_ROSTER_MAX:
                try:
                    from app.sim_engine.trades.roster_balance import auto_send_down_overflow, send_down_candidates

                    skaters = [c for c in send_down_candidates(team, count=3) if position_bucket(c) != "G"]
                    if not skaters:
                        break
                    victim = skaters[0]
                    team.roster = [x for x in list(team.roster or []) if x is not victim]
                    ahl = list(getattr(team, "ahl_roster", None) or [])
                    ahl.append(victim)
                    team.ahl_roster = ahl
                    victim.in_minors = True
                    victim.roster_location = "ahl"
                    recalled.append({"player_id": player_key(victim), "player_name": player_name(victim), "bucket": position_bucket(victim), "to": "ahl_roster"})
                except Exception:
                    break
            attr, pick = max(pool, key=lambda row: player_ovr(row[1]))
            if not _recall_affiliate_player(team, pick, league, attr):
                break
            recalled.append({"player_id": player_key(pick), "player_name": player_name(pick), "bucket": "G", "from": attr})
        healthy = _healthy_goalies()
    ok = len(healthy) >= 2
    msg = ""
    if not ok:
        msg = (
            f"You only have {len(healthy)} healthy goalie{'s' if len(healthy) != 1 else ''} on the NHL roster — "
            "a club must dress two. Call one up from the AHL or sign a free agent (he can report to the AHL "
            "if your roster or cap is full)."
        )
    return {"ok": ok, "healthy": len(healthy), "recalled": recalled, "message": msg}


def _assign_affiliate(player: Any, team: Any, level: str) -> None:
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    try:
        player.in_minors = True
        player.roster_location = level
        player.organizational_status = "minors"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        ctx = getattr(player, "context", None)
        if ctx is not None:
            ctx.current_team_id = f"{level.upper()}_{tid}" if tid else level.upper()
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _team_label

        _set_assignment(player, org_nhl_team_id=tid, level=level, club=_team_label(team))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def ensure_affiliate_goalies(session: Any, *, min_goalies: int = 2) -> List[Dict[str, Any]]:
    """Every AHL affiliate keeps at least two healthy goalies.

    Call-ups, trades and injuries could strip an affiliate bare, leaving the AHL club
    (and any future emergency recall) with nobody in net. Refill in order: the ECHL
    affiliate's best goalie, then the best available free-agent goalie on an AHL deal.
    """
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return []
    season_year = int(getattr(session, "season_calendar_year", 0) or getattr(league, "season_year", 2026) or 2026)
    moves: List[Dict[str, Any]] = []
    for team in list(getattr(league, "teams", None) or []):
        ahl = list(getattr(team, "ahl_roster", None) or [])

        def _healthy(lst: List[Any]) -> List[Any]:
            return [
                p for p in lst
                if position_bucket(p) == "G" and not getattr(p, "retired", False) and not is_injured(p)
            ]

        need = min_goalies - len(_healthy(ahl))
        if need <= 0:
            continue
        echl = list(getattr(team, "echl_roster", None) or [])
        for g in sorted(_healthy(echl), key=player_ovr, reverse=True)[:need]:
            echl = [x for x in echl if x is not g]
            ahl.append(g)
            _assign_affiliate(g, team, "ahl")
            need -= 1
            moves.append({"team_id": str(getattr(team, "team_id", "") or ""), "player_name": player_name(g), "from": "echl"})
        team.echl_roster = echl
        team.ahl_roster = ahl
        if need <= 0:
            continue
        try:
            from services.contract_economy import prune_owned_from_fa_pools, sign_minor_or_tryout_contract

            prune_owned_from_fa_pools(league)
        except Exception:
            continue
        for _ in range(need):
            pool: List[Tuple[str, Any]] = []
            for attr in ("free_agents", "overseas_free_agents"):
                for p in list(getattr(league, attr, None) or []):
                    if position_bucket(p) == "G" and not getattr(p, "retired", False) and not is_injured(p):
                        pool.append((attr, p))
            # Best goalie who would actually take an AHL deal (no NHL-calibre starters).
            pool = [row for row in pool if player_ovr(row[1]) <= 76.0] or pool
            if not pool:
                break
            attr, g = max(pool, key=lambda row: player_ovr(row[1]))
            try:
                sign_minor_or_tryout_contract(
                    g, team, league, season_year,
                    {"contract_category": "ahl", "years": 1, "aav_m": 0.1},
                )
            except Exception:
                break
            try:
                setattr(league, attr, [x for x in list(getattr(league, attr, None) or []) if x is not g])
                team.prospect_pool = [x for x in list(getattr(team, "prospect_pool", None) or []) if x is not g]
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            ahl = list(getattr(team, "ahl_roster", None) or [])
            ahl.append(g)
            team.ahl_roster = ahl
            _assign_affiliate(g, team, "ahl")
            moves.append({"team_id": str(getattr(team, "team_id", "") or ""), "player_name": player_name(g), "from": attr})
    return moves
