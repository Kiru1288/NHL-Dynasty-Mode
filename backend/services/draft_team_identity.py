"""
Club-specific Entry Draft boards for CPU teams.

Every CPU club ranks the class on its own board instead of following the public
consensus list:

* a consensus anchor (public rank) blended with the club's private scouting read
  of each prospect's real ability (hidden OVR/potential plus noise sized by the
  scouting department's quality and its coverage of that league);
* an organizational identity seeded per club: size, skill vs two-way, age,
  CHL / Europe / US-path preference, risk appetite and positional emphasis;
* evaluation noise seeded per club + draft year + prospect;
* occasional "our guy" conviction, where a staff falls for a prospect the public
  board has 10-25+ spots later and takes him at its own slot.

Values live on a log-rank scale: the same conviction moves a pick a few spots at
the top of round one and many spots on day two, which is how real boards behave.

The live draft (services/franchise_entry_draft.py) asks this module for each
club's board and its selection. Nothing here mutates draft state; per-draft
results are memoized inside the draft cache dict the caller passes in.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

RANK_K = 5.0  # log-rank offset: ln(rank + K)
UNRANKED_RANK = 330

# Calibration (see the reach-distribution notes in the module docstring).
PRIVATE_WEIGHT_BASE = 0.64
PRIVATE_WEIGHT_SLOPE = 0.58  # per unit of public_board_trust
PRIVATE_WEIGHT_RANGE = (0.14, 0.5)
BOARD_NOISE_BASE = 0.06
BOARD_NOISE_SPAN = 0.10  # extra noise for the weakest scouting departments
CONVICTION_BASE = 0.16
CONVICTION_SLOPE = 0.42  # per unit of off_board_tendency

# Talent composite the scouts are trying to read (current ability vs ceiling).
OVR_WEIGHT = 0.35
POT_WEIGHT = 0.65

_SKILL_STYLES = {"sniper": 1.0, "playmaker": 1.0, "scoring_forward": 1.0, "offensive_defenseman": 0.9, "mobile_defenseman": 0.4}
_TWO_WAY_STYLES = {"grinder": -1.0, "defensive_defenseman": -1.0, "two_way_defenseman": -0.6, "two_way": -0.35, "power_forward": -0.1}


def _u(*parts: Any) -> float:
    """Deterministic uniform [0, 1)."""
    raw = ":".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:12], 16) / float(1 << 48)


def _g(*parts: Any) -> float:
    """Deterministic standard normal (Box-Muller on two seeded uniforms)."""
    u1 = max(1e-9, _u(*parts, "g1"))
    u2 = _u(*parts, "g2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _key(entry: Dict[str, Any]) -> str:
    return str(entry.get("key") or entry.get("prospect_id") or entry.get("player_id") or "")


def _pub_rank(entry: Dict[str, Any]) -> int:
    try:
        r = int(entry.get("rank") or entry.get("public_rank") or 0)
    except (TypeError, ValueError):
        r = 0
    return r if 0 < r < 900 else UNRANKED_RANK


def _league_bucket(entry: Dict[str, Any]) -> str:
    code = str(entry.get("league_code") or entry.get("league") or "").upper()
    if code.startswith("EU_") or code in ("SHL", "LIIGA", "KHL", "MHL", "DEL", "NL"):
        return "EUR"
    if "NCAA" in code or "USHL" in code or code.startswith("US"):
        return "US"
    return "CHL"


def _season_seed(session: Any) -> str:
    return str(getattr(session, "session_id", "") or "draft")


# ---------------------------------------------------------------------------
# Hidden ability the scouts are trying to read
# ---------------------------------------------------------------------------


def prospect_true_talent(session: Any, entries: List[Dict[str, Any]], cache: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """pid -> (current OVR, potential) from the live player entity, memoized per draft cache."""
    memo = cache.get("true_talent")
    if isinstance(memo, dict) and memo.get("_n") == len(entries):
        return memo
    out: Dict[str, Any] = {}
    try:
        from services.franchise_entry_draft import _build_dev_home_index, _lookup_prospect_player
        from services.franchise_sim import _draft_potential99, _player_display_ovr99

        index = _build_dev_home_index(session)
    except Exception:
        index = None
    for e in entries:
        pid = _key(e)
        if not pid:
            continue
        ovr = pot = None
        if index is not None:
            try:
                player, _blk, _tm = _lookup_prospect_player(session, pid, index)
                if player is not None:
                    ovr = float(_player_display_ovr99(player))
                    pot = float(_draft_potential99(player, ovr))
            except Exception:
                ovr = pot = None
        if not ovr:
            # No live entity: fall back to the public estimates (still fogged).
            ovr = _f(e.get("current_ovr_estimate") or e.get("floor_score"), 60.0)
            pot = _f(e.get("expected_ceiling_estimate") or e.get("potential_score") or e.get("ceiling_score"), ovr + 8.0)
        out[pid] = (float(ovr), float(max(pot or 0.0, ovr)))
    out["_n"] = len(entries)
    cache["true_talent"] = out
    return out


# ---------------------------------------------------------------------------
# Organizational draft identity
# ---------------------------------------------------------------------------


def team_draft_identity(
    session: Any,
    team_id: str,
    draft_year: int,
    *,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tid = str(team_id)
    memo = (cache or {}).get("team_identity") if cache is not None else None
    if isinstance(memo, dict) and tid in memo:
        return memo[tid]

    seed = _season_seed(session)
    profile: Dict[str, Any] = {}
    try:
        from services.franchise_scouting import get_team_scouting_profile

        profile = get_team_scouting_profile(session, tid) or {}
    except Exception:
        profile = {}
    phil = ""
    try:
        from services.franchise_entry_draft import get_team_draft_philosophy

        phil_map = (cache or {}).get("team_philosophies") or {}
        phil_row = phil_map.get(tid) or get_team_draft_philosophy(session, tid)
        phil = str((phil_row or {}).get("philosophy") or "")
    except Exception:
        phil = ""
    ideo = dict(((getattr(session, "cpu_franchise_profiles", None) or {}).get(tid) or {}).get("ideology") or {})
    team = (getattr(session, "team_by_id", None) or {}).get(tid)
    archetype = str(getattr(team, "archetype", "") or "").lower()

    q = _f(profile.get("scouting_quality"), 65.0)
    pbt = _f(profile.get("public_board_trust"), 0.55)
    off_board = _f(profile.get("off_board_tendency"), 0.3)
    sleeper = _f(profile.get("sleeper_detection"), 0.6)

    # Organizational traits are stable for the franchise (seeded per club) with a
    # small per-draft drift so a staff's taste evolves year to year.
    def trait(name: str, spread: float = 1.0) -> float:
        base = _g(seed, "identity", tid, name)
        drift = _g(seed, "identity", tid, name, int(draft_year)) * 0.25
        return _clamp((base * 0.6 + drift) * spread, -1.0, 1.0)

    size_pref = trait("size")
    style_pref = trait("style")
    age_pref = trait("age")
    risk_pref = trait("risk", 0.7)
    if any(k in archetype for k in ("grit", "heavy", "physical", "old", "traditional")):
        size_pref = _clamp(size_pref + 0.45, -1.0, 1.0)
        style_pref = _clamp(style_pref - 0.3, -1.0, 1.0)
    if "analytics" in archetype:
        size_pref = _clamp(size_pref - 0.35, -1.0, 1.0)
        style_pref = _clamp(style_pref + 0.3, -1.0, 1.0)
    if phil in ("boom_bust_gambler", "high_upside", "rebuilder_upside"):
        risk_pref = _clamp(risk_pref + (0.7 if phil == "boom_bust_gambler" else 0.45), -1.0, 1.0)
        age_pref = _clamp(age_pref + 0.25, -1.0, 1.0)
    elif phil in ("safe_floor", "contender_timeline"):
        risk_pref = _clamp(risk_pref - 0.55, -1.0, 1.0)
        age_pref = _clamp(age_pref - 0.25, -1.0, 1.0)
    risk_pref = _clamp(risk_pref + (_f(ideo.get("risk_tolerance"), 0.5) - 0.5) * 0.8, -1.0, 1.0)

    league_pref = {"CHL": 0.0, "EUR": 0.0, "US": 0.0}
    for bias in profile.get("league_biases") or []:
        b = str(bias)
        bucket = "EUR" if b == "Europe" else ("US" if b in ("NCAA", "USHL") else "CHL")
        league_pref[bucket] += 0.55
    eur_q = _f(profile.get("European_scouting_quality"), q)
    league_pref["EUR"] += _clamp((eur_q - 68.0) / 20.0, -0.6, 0.6)
    for bucket in league_pref:
        league_pref[bucket] = _clamp(league_pref[bucket] + _g(seed, "league", tid, bucket) * 0.3, -1.0, 1.0)

    pos_pref = {"C": 0.0, "D": 0.0, "G": 0.0, "W": 0.0}
    if phil == "center_priority":
        pos_pref["C"] += 0.6
    if phil == "defense_first" or "defense" in archetype:
        pos_pref["D"] += 0.6
    if phil == "goalie_tolerant":
        pos_pref["G"] += 0.6
    pos_pref["G"] += (_f(ideo.get("goaltending_investment"), 0.5) - 0.5) * 1.2
    for p in pos_pref:
        pos_pref[p] = _clamp(pos_pref[p] + _g(seed, "pos", tid, p) * 0.25, -1.0, 1.0)

    # How much the staff trusts its own read over the consensus list.
    private_weight = _clamp(
        PRIVATE_WEIGHT_BASE - PRIVATE_WEIGHT_SLOPE * pbt + 0.10 * (sleeper - 0.6) + 0.06 * _g(seed, "trust", tid, int(draft_year)),
        PRIVATE_WEIGHT_RANGE[0],
        PRIVATE_WEIGHT_RANGE[1],
    )
    qn = _clamp((q - 50.0) / 40.0, 0.0, 1.0)
    noise_sd = BOARD_NOISE_BASE + BOARD_NOISE_SPAN * (1.0 - qn)  # board noise, log-rank units
    scout_sd = 0.9 + 3.2 * (1.0 - qn)  # error on the talent composite, rating points
    need_scale = _clamp(0.45 + _f(ideo.get("positional_need_draft_bias"), 0.45), 0.6, 1.25)
    if phil == "need_focused":
        need_scale *= 1.35
    elif phil == "bpa_heavy" or _f(ideo.get("best_player_available_bias"), 0.55) >= 0.62:
        need_scale *= 0.7
    conviction = _clamp(
        CONVICTION_BASE + CONVICTION_SLOPE * off_board + (0.08 if phil in ("off_board_scout", "boom_bust_gambler") else 0.0),
        0.0,
        0.45,
    )

    tags: List[str] = []
    if size_pref >= 0.45:
        tags.append("Size and snarl")
    elif size_pref <= -0.45:
        tags.append("Skates over size")
    if style_pref >= 0.45:
        tags.append("Skill first")
    elif style_pref <= -0.45:
        tags.append("Two-way builders")
    if age_pref >= 0.5:
        tags.append("Young upside")
    elif age_pref <= -0.5:
        tags.append("Older, readier")
    best_league = max(league_pref.items(), key=lambda kv: kv[1])
    if best_league[1] >= 0.45:
        tags.append({"CHL": "CHL pipeline", "EUR": "Euro scouting shop", "US": "US college path"}[best_league[0]])
    if private_weight >= 0.38:
        tags.append("Trusts own list")
    elif private_weight <= 0.22:
        tags.append("Follows consensus")

    identity = {
        "team_id": tid,
        "draft_year": int(draft_year),
        "philosophy": phil,
        "scouting_quality": round(q, 1),
        "private_weight": round(private_weight, 3),
        "noise_sd": round(noise_sd, 3),
        "scout_sd": round(scout_sd, 2),
        "size_pref": round(size_pref, 3),
        "style_pref": round(style_pref, 3),
        "age_pref": round(age_pref, 3),
        "risk_pref": round(risk_pref, 3),
        "league_pref": {k: round(v, 3) for k, v in league_pref.items()},
        "pos_pref": {k: round(v, 3) for k, v in pos_pref.items()},
        "need_scale": round(need_scale, 3),
        "conviction_rate": round(conviction, 3),
        "tags": tags[:3],
    }
    if cache is not None:
        cache.setdefault("team_identity", {})[tid] = identity
    return identity


def _identity_adjustment(identity: Dict[str, Any], entry: Dict[str, Any], talent: Tuple[float, float]) -> float:
    adj = 0.0
    h = _f(entry.get("height_cm"), 0.0)
    w = _f(entry.get("weight"), 0.0)
    if h > 0 or w > 0:
        z = 0.0
        if h > 0:
            z += _clamp((h - 185.0) / 5.0, -2.0, 2.0) * 0.5
        if w > 0:
            z += _clamp((w - 190.0) / 12.0, -2.0, 2.0) * 0.5
        adj += 0.04 * identity["size_pref"] * z

    style = str(entry.get("playstyle") or "").lower()
    sz = _SKILL_STYLES.get(style, _TWO_WAY_STYLES.get(style, 0.0))
    off_r, def_r = _f(entry.get("offence"), 0.0), _f(entry.get("defence"), 0.0)
    if off_r and def_r:
        sz += _clamp((off_r - def_r) / 10.0, -1.0, 1.0) * 0.5
    adj += 0.035 * identity["style_pref"] * _clamp(sz, -1.5, 1.5)

    age = _f(entry.get("age"), 19.0)
    az = 1.0 if age <= 18 else (0.0 if age < 20 else -1.2)
    adj += 0.03 * identity["age_pref"] * az
    if age >= 20 and identity["age_pref"] > -0.5:
        adj -= 0.03  # most staffs discount overagers

    adj += 0.05 * identity["league_pref"].get(_league_bucket(entry), 0.0)

    ovr, pot = talent
    gap_z = _clamp((pot - ovr - 11.0) / 5.0, -1.5, 1.5)
    adj += 0.035 * identity["risk_pref"] * gap_z

    pos = str(entry.get("position") or "").upper()
    pkey = "G" if pos == "G" else ("D" if pos.endswith("D") or pos == "D" else ("C" if pos == "C" else "W"))
    adj += 0.05 * identity["pos_pref"].get(pkey, 0.0)
    return adj


# ---------------------------------------------------------------------------
# Club board over the whole class (memoized per draft)
# ---------------------------------------------------------------------------


def _scouting_event_delta(session: Any, team_id: str, entry: Dict[str, Any], phil_row: Dict[str, Any]) -> Tuple[float, bool, bool, List[str]]:
    """Combine / interview / dinner / do-not-draft signals on the club's file."""
    try:
        from services.franchise_entry_draft import _scouting_event_adjustments, _scouting_overlay

        overlay = _scouting_overlay(session, _key(entry), team_id=team_id)
        delta, _meta, notes = _scouting_event_adjustments(session, team_id, entry, overlay, phil_row)
        favorite = bool(overlay.get("target") or overlay.get("scout_favorite"))
        dnd = bool(overlay.get("do_not_draft"))
        return float(delta), favorite, dnd, list(notes or [])
    except Exception:
        return 0.0, False, False, []


def _draft_slots_for_team(state: Dict[str, Any], team_id: str) -> List[int]:
    out: List[int] = []
    for idx, slot in enumerate(state.get("draft_order") or []):
        if not isinstance(slot, dict):
            continue
        if str(slot.get("team_id") or "") == str(team_id):
            try:
                out.append(int(slot.get("overall_pick") or idx + 1))
            except (TypeError, ValueError):
                out.append(idx + 1)
    return sorted(out)


def team_class_board(
    session: Any,
    team_id: str,
    entries: List[Dict[str, Any]],
    cache: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """pid -> {value, private_rank, story, ...} for the club's whole-class board."""
    tid = str(team_id)
    boards = cache.setdefault("team_class_boards", {})
    memo = boards.get(tid)
    if isinstance(memo, dict) and memo.get("_n") == len(entries):
        return memo

    state = getattr(session, "draft_state", None) or {}
    draft_year = int(state.get("draft_year") or int(getattr(session, "season_calendar_year", 2025) or 2025) + 1)
    seed = _season_seed(session)
    identity = team_draft_identity(session, tid, draft_year, cache=cache)
    talent = prospect_true_talent(session, entries, cache)
    phil_row = ((cache.get("team_philosophies") or {}).get(tid)) or {"philosophy": identity.get("philosophy"), "risk_tolerance": 0.5}

    profile: Dict[str, Any] = {}
    try:
        from services.franchise_scouting import get_team_scouting_profile

        profile = get_team_scouting_profile(session, tid) or {}
    except Exception:
        profile = {}
    q = _f(profile.get("scouting_quality"), 65.0)
    league_q = {
        "EUR": _f(profile.get("European_scouting_quality"), q),
        "US": _f(profile.get("NCAA_scouting_quality"), q),
        "CHL": _f(profile.get("CHL_scouting_quality"), q),
    }

    # 1) The scouts' read of real ability, ranked across the whole class.
    reads: List[Tuple[str, float]] = []
    for e in entries:
        pid = _key(e)
        if not pid or pid not in talent:
            continue
        ovr, pot = talent[pid]
        pw = _clamp(POT_WEIGHT + 0.08 * float(identity["risk_pref"]), 0.5, 0.78)
        composite = (1.0 - pw) * ovr + pw * pot
        coverage = _clamp((league_q.get(_league_bucket(e), q) - 50.0) / 40.0, 0.0, 1.0)
        sd = identity["scout_sd"] * (1.0 + 0.6 * (1.0 - coverage))
        reads.append((pid, composite + _g(seed, "read", tid, draft_year, pid) * sd))
    reads.sort(key=lambda x: -x[1])
    private_rank = {pid: i + 1 for i, (pid, _c) in enumerate(reads)}

    w = identity["private_weight"]
    board: Dict[str, Any] = {}
    for e in entries:
        pid = _key(e)
        if not pid:
            continue
        pub = _pub_rank(e)
        prv = private_rank.get(pid, pub)
        base = -(1.0 - w) * math.log(pub + RANK_K) - w * math.log(prv + RANK_K)
        ident = _identity_adjustment(identity, e, talent.get(pid, (60.0, 68.0)))
        noise = _g(seed, "board", tid, draft_year, pid) * identity["noise_sd"]
        ev_delta, favorite, dnd, notes = _scouting_event_delta(session, tid, e, phil_row)
        # Event deltas are on the legacy point scale (~1 point per public slot).
        event_adj = _clamp(ev_delta * 0.03, -0.35, 0.3)
        if favorite:
            event_adj += 0.12
        value = base + ident + noise + event_adj
        if dnd:
            value -= 3.0
        pos = str(e.get("position") or "").upper()
        if pos == "G" and pub <= 40 and identity["pos_pref"].get("G", 0.0) < 0.5:
            value -= 0.07  # most clubs will not spend a high pick on a goalie
        board[pid] = {
            "value": value,
            "public_rank": pub,
            "private_rank": prv,
            "identity_adj": round(ident, 4),
            "event_adj": round(event_adj, 4),
            "favorite": favorite,
            "do_not_draft": dnd,
            "notes": notes[:3],
            "conviction": 0.0,
            "story": "our_guy" if favorite else None,
        }

    # 2) "Our guy": some staffs fall for a prospect the public has well behind
    #    their own slot and plan to take him there.
    slots = _draft_slots_for_team(state, tid)
    crushes = 0
    used: set = set()
    for slot_no in slots[:3]:
        if crushes >= 2:
            break
        rnd = (slot_no - 1) // 32 + 1
        rate = identity["conviction_rate"] + (0.05 if rnd >= 2 else 0.0)
        if _u(seed, "crush", tid, draft_year, slot_no) >= rate:
            continue
        lo, hi = (slot_no + 9, slot_no + 26) if rnd == 1 else (slot_no + 10, slot_no + 40)
        cands = []
        for pid, row in board.items():
            if pid in used or row["do_not_draft"] or not (lo <= row["public_rank"] <= hi):
                continue
            pick_score = (
                -math.log(row["private_rank"] + RANK_K)
                + row["identity_adj"] * 2.0
                + _g(seed, "crush-pick", tid, slot_no, pid) * 0.15
            )
            cands.append((pick_score, pid))
        if not cands:
            continue
        cands.sort(reverse=True)
        pid = cands[0][1]
        row = board[pid]
        slot_value = -math.log(slot_no + RANK_K)
        need = max(0.0, slot_value - row["value"])
        bump = need * (0.8 + 0.5 * _u(seed, "crush-size", tid, slot_no)) + 0.03
        row["value"] += bump
        row["conviction"] = round(bump, 4)
        row["story"] = "our_guy"
        used.add(pid)
        crushes += 1

    board["_n"] = len(entries)
    boards[tid] = board
    return board


# ---------------------------------------------------------------------------
# Situational board (needs) + selection
# ---------------------------------------------------------------------------


def _need_adjustment(entry: Dict[str, Any], needs: List[Dict[str, Any]], identity: Dict[str, Any], overall: int) -> float:
    if not needs:
        return 0.0
    pos = str(entry.get("position") or "").upper()
    is_d = pos == "D" or pos.endswith("D")
    shoots = str(entry.get("handedness") or entry.get("shoots") or "").upper()
    adj = 0.0
    for n in needs[:3]:
        if not isinstance(n, dict):
            continue
        cat = str(n.get("category") or "")
        pri = _f(n.get("priority"), 0.7)
        if cat in ("Franchise Center", "Center Depth") and pos == "C":
            adj += 0.06 * pri
        elif cat == "Right-Shot Defense" and is_d and (shoots.startswith("R") or pos in ("RD", "RHD")):
            adj += 0.06 * pri
        elif cat == "Goalie Pipeline" and pos == "G":
            adj += 0.05 * pri
        elif cat in ("Top-Six Winger", "Wing Depth") and pos in ("LW", "RW", "W"):
            adj += 0.035 * pri
    # Top of round one is best-player-available almost everywhere.
    scale = 0.45 if overall <= 10 else (0.8 if overall <= 32 else 1.0)
    return adj * scale * float(identity.get("need_scale") or 1.0)


def _reach_window(overall: int) -> int:
    if overall <= 32:
        return 32
    if overall <= 96:
        return 60
    return 90


def score_available_for_team(
    session: Any,
    team_id: str,
    overall: int,
    available: List[Dict[str, Any]],
    cache: Dict[str, Any],
    *,
    entries: Optional[List[Dict[str, Any]]] = None,
    needs: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """[(score, entry, board_row)] best-first for the club on the clock."""
    entries = entries if entries is not None else available
    board = team_class_board(session, team_id, entries, cache)
    state = getattr(session, "draft_state", None) or {}
    draft_year = int(state.get("draft_year") or int(getattr(session, "season_calendar_year", 2025) or 2025) + 1)
    identity = team_draft_identity(session, team_id, draft_year, cache=cache)
    if needs is None:
        needs = list((cache.get("team_needs") or {}).get(str(team_id)) or [])
    window = _reach_window(overall)
    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for e in available:
        pid = _key(e)
        row = board.get(pid)
        if row is None:
            # Late addition to the board (shouldn't happen mid-draft): consensus only.
            row = {"value": -math.log(_pub_rank(e) + RANK_K), "public_rank": _pub_rank(e), "private_rank": _pub_rank(e), "story": None, "conviction": 0.0}
        score = float(row["value"]) + _need_adjustment(e, needs, identity, overall)
        if row["public_rank"] > overall + window and not row.get("conviction"):
            score -= 1.0  # a club does not take a name this far off every list
        scored.append((score, e, row))
    scored.sort(key=lambda x: -x[0])
    return scored


def select_cpu_prospect(
    session: Any,
    team_id: str,
    overall: int,
    available: List[Dict[str, Any]],
    cache: Dict[str, Any],
    *,
    entries: Optional[List[Dict[str, Any]]] = None,
    needs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not available:
        raise ValueError("No prospects available")
    scored = score_available_for_team(session, team_id, overall, available, cache, entries=entries, needs=needs)
    return scored[0][1]


def board_story_for_pick(row: Optional[Dict[str, Any]], overall: int) -> Optional[str]:
    """Short floor-language reason when a club's own board drove the pick."""
    if not row:
        return None
    pub = int(row.get("public_rank") or 0)
    prv = int(row.get("private_rank") or 0)
    if row.get("story") == "our_guy" and pub > overall + 6:
        return f"Their guy all along — the staff had him far above his public #{pub} and would not risk waiting."
    if pub > overall + 8 and prv and prv < pub - 10:
        return f"Off the public board: their scouts graded him well above consensus #{pub}."
    if pub > overall + 8:
        return f"Organizational fit over consensus — taken ahead of public #{pub}."
    if pub < overall - 8:
        return f"Value too good to pass: consensus #{pub} slid to them."
    return None
