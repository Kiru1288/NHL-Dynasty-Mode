"""
CPU team assessment — status, needs, surplus and wants.

This is what a GM "knows" before picking up the phone:

* status   — contender / bubble / seller / tank, from the standings and the playoff-line
             gap, blended with the front office's stated direction early in the season.
             Deadline flags: panic buyer (expected contender sliding), chaser (bubble just
             outside), late flip to seller.
* needs    — lineup slots scored against the LEAGUE (not fixed OVR targets), so a team
             with a weak 2C or no right-shot top-4 D shows a real hole.
* surplus  — players a club can spare: depth beyond the lineup, a blocked prospect, a
             third goalie, expiring veterans on non-contenders, veterans on tankers, bad
             contracts, disruptors / demanders.
* spend    — what a buyer will pay with: picks (protecting what it values) and prospects.

Assessments are cached on the league per day (``league._cpu_assessments``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import player_holds_nhl_spc, team_id_of

# ---------------------------------------------------------------------------
# Lineup model
# ---------------------------------------------------------------------------

#: slot → (position group, first depth rank, last depth rank) — ranks are 1-based.
SLOTS: Dict[str, Tuple[str, int, int]] = {
    "C_TOP2": ("C", 1, 2),
    "W_TOP6": ("W", 1, 4),
    "LD_TOP4": ("LD", 1, 2),
    "RD_TOP4": ("RD", 1, 2),
    "G_START": ("G", 1, 1),
    "F_BOTTOM6": ("F", 7, 12),
    "D_BOTTOM": ("D", 5, 6),
}
SLOT_LABELS = {
    "C_TOP2": "top-six centre",
    "W_TOP6": "top-six winger",
    "LD_TOP4": "top-four left defence",
    "RD_TOP4": "top-four right defence",
    "G_START": "starting goaltender",
    "F_BOTTOM6": "bottom-six forward",
    "D_BOTTOM": "third-pair defence",
}
#: Dressed lineup per group (NHL roster).
LINEUP_COUNTS = {"C": 4, "W": 8, "D": 6, "G": 2}
#: OVR below the league median at a slot that counts as a full (1.0) need.
NEED_FULL_GAP = 6.0

STATUS_CONTENDER = "contender"
STATUS_BUBBLE = "bubble"
STATUS_SELLER = "seller"
STATUS_TANK = "tank"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def player_ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0.0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


# ---------------------------------------------------------------------------
# Form — current-season performance relative to what the league's own data says a
# player of that OVR produces. GMs used to rank their roster on OVR alone, so the
# same depth names were "surplus" every day of every save regardless of how anyone
# was actually playing. Form is fit per pass from the live season ledger
# (league.player_season_stats), so it adapts to whatever the sim produces.
# ---------------------------------------------------------------------------

#: pid → form adjustment in OVR points (±FORM_MAX), set by assess_league each pass.
_FORM: Dict[str, float] = {}
FORM_MAX = 5.0
FORM_MIN_GP = 8
FORM_MIN_SAMPLE = 20


def player_rating(player: Any) -> float:
    """OVR adjusted for current-season form — what the GM actually sees."""
    return player_ovr(player) + _FORM.get(player_pid(player), 0.0)


def player_form(player: Any) -> float:
    return _FORM.get(player_pid(player), 0.0)


def _linfit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """(slope, intercept, residual std) of y ~ x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx) if vx > 1e-9 else 0.0
    icpt = my - slope * mx
    res = [y - (slope * x + icpt) for x, y in zip(xs, ys)]
    sd = (sum(r * r for r in res) / max(1, n - 2)) ** 0.5
    return slope, icpt, sd


def compute_form(league: Any, teams: List[Any]) -> Dict[str, float]:
    reg = getattr(league, "player_season_stats", None)
    if not isinstance(reg, dict) or not reg:
        return {}
    groups: Dict[str, List[Tuple[str, float, float, int]]] = {"F": [], "D": [], "G": []}
    for tm in teams:
        for p in list(getattr(tm, "roster", None) or []):
            row = reg.get(player_pid(p))
            if not isinstance(row, dict):
                continue
            gp = _safe_int(row.get("gp"), 0)
            if gp < FORM_MIN_GP:
                continue
            g = position_group(p)
            key = "G" if g == "G" else ("D" if g in ("LD", "RD") else "F")
            if key == "G":
                # Lower GAA is better → negate so "higher is better" like points.
                perf = -(_safe_float(row.get("ga"), 0.0) / gp)
                sa = _safe_float(row.get("sa") or row.get("shots_against"), 0.0)
                if sa > 0:
                    perf = 1.0 - _safe_float(row.get("ga"), 0.0) / sa  # true SV% when tracked
            else:
                perf = _safe_float(row.get("pts"), 0.0) / gp
            groups[key].append((player_pid(p), player_ovr(p), perf, gp))
    out: Dict[str, float] = {}
    for key, rows in groups.items():
        if len(rows) < FORM_MIN_SAMPLE:
            continue
        slope, icpt, sd = _linfit([r[1] for r in rows], [r[2] for r in rows])
        if sd <= 1e-9:
            continue
        for pid, ovr, perf, gp in rows:
            z = (perf - (slope * ovr + icpt)) / sd
            shrink = gp / (gp + 15.0)  # small samples barely move the needle
            out[pid] = round(max(-FORM_MAX, min(FORM_MAX, z * 2.0)) * shrink, 2)
    return out


def player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    return _safe_int(getattr(ident, "age", getattr(player, "age", 27)), 27)


def player_pid(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _raw_position(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", None) or getattr(player, "position", None)
    return str(getattr(pos, "value", pos) or "").upper()


def _shoots(player: Any) -> str:
    ident = getattr(player, "identity", None)
    s = getattr(ident, "shoots", None) or getattr(player, "shoots", None)
    return str(getattr(s, "value", s) or "L").upper()[:1]


def position_group(player: Any) -> str:
    """C / W / LD / RD / G."""
    pos = _raw_position(player)
    if pos in ("G", "GOALIE", "GOALTENDER"):
        return "G"
    if pos in ("C", "CENTER", "CENTRE"):
        return "C"
    if pos in ("LW", "RW", "W", "F", "WING"):
        return "W"
    if pos == "LD":
        return "LD"
    if pos == "RD":
        return "RD"
    if pos in ("D", "DEFENSE", "DEFENCE"):
        return "RD" if _shoots(player) == "R" else "LD"
    return "W"


def _group_key(group: str) -> str:
    """Collapse LD/RD to D for lineup counts."""
    return "D" if group in ("LD", "RD") else group


def _is_injured(player: Any) -> bool:
    try:
        from app.sim_engine.economy.team_needs import is_player_injured

        return bool(is_player_injured(player))
    except Exception:
        return False


def _cfield(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def contract_years_left(player: Any) -> int:
    # Contracts are dicts; getattr() on them always gave 0, so every player looked like an
    # expiring rental and "bad contract" (needs 2+ years) could never trigger.
    c = getattr(player, "contract", None)
    years = 0
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years"):
            years = max(years, _safe_int(_cfield(obj, key, 0), 0))
    return years


def cap_hit_m(player: Any) -> float:
    c = getattr(player, "contract", None)
    for obj in (c, player):
        if obj is None:
            continue
        for key in ("cap_hit_m", "aav_m", "cap_hit"):
            v = _cfield(obj, key, None)
            if v is not None:
                f = _safe_float(v, 0.0)
                return f / 1_000_000.0 if f > 1000 else f
    return 0.0


def is_rental(player: Any) -> bool:
    return contract_years_left(player) <= 1 and player_age(player) >= 26


def potential_ovr(player: Any) -> float:
    for obj in (player, getattr(player, "identity", None)):
        if obj is None:
            continue
        for key in ("potential_ovr", "potential", "ceiling", "pot"):
            v = getattr(obj, key, None)
            if v is None or callable(v):
                continue
            f = _safe_float(v, 0.0)
            if f > 0:
                return f * 99.0 if f <= 1.5 else f
    return player_ovr(player)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SurplusItem:
    player: Any
    reason: str  # depth | positional | goalie | blocked_prospect | expiring_veteran | veteran_selloff | bad_contract | locker_room
    group: str
    priority: float  # how keen the club is to move him (0–1)


@dataclass
class TeamAssessment:
    team_id: str
    status: str
    points_pct: float = 0.5
    playoff_gap_pts: float = 0.0
    games_played: int = 0
    rank: int = 16
    panic_buyer: bool = False
    chaser: bool = False
    late_seller: bool = False
    cap_space_m: float = 0.0
    cap_tight: bool = False
    needs: Dict[str, float] = field(default_factory=dict)
    slot_floor: Dict[str, float] = field(default_factory=dict)  # weakest OVR currently in each slot
    surplus: List[SurplusItem] = field(default_factory=list)
    spend_prospects: List[Any] = field(default_factory=list)
    protects_first: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_buyer(self) -> bool:
        return self.status == STATUS_CONTENDER or self.panic_buyer or self.chaser

    @property
    def is_seller(self) -> bool:
        return self.status in (STATUS_SELLER, STATUS_TANK) or self.late_seller

    def top_needs(self, floor: float = 0.2) -> List[Tuple[str, float]]:
        return sorted(((k, v) for k, v in self.needs.items() if v >= floor), key=lambda kv: -kv[1])

    def surplus_ids(self) -> Dict[str, SurplusItem]:
        return {player_pid(s.player): s for s in self.surplus}


# ---------------------------------------------------------------------------
# Lineup / needs
# ---------------------------------------------------------------------------


def _nhl_lineup_by_group(team: Any) -> Dict[str, List[Any]]:
    """Healthy NHL roster by group, OVR-desc; F is C+W combined."""
    out: Dict[str, List[Any]] = {"C": [], "W": [], "LD": [], "RD": [], "G": [], "F": [], "D": []}
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False) or _is_injured(p):
            continue
        g = position_group(p)
        out[g].append(p)
        out["F" if g in ("C", "W") else ("D" if g in ("LD", "RD") else "G")].append(p)
    for k in out:
        out[k].sort(key=player_rating, reverse=True)
    # Thin sides borrow from the other side (a left-shot D can play the right).
    return out


def _slot_value(lineup: Dict[str, List[Any]], slot: str) -> Tuple[float, float]:
    """(average OVR across the slot, weakest OVR in the slot). Missing bodies count as 55."""
    group, lo, hi = SLOTS[slot]
    players = lineup.get(group) or []
    if group in ("LD", "RD") and len(players) < hi:
        other = lineup.get("RD" if group == "LD" else "LD") or []
        players = players + [p for p in other[2:]]  # 3rd+ D from other side cover a hole
    vals = [player_rating(p) for p in players[lo - 1 : hi]]
    vals += [55.0] * max(0, (hi - lo + 1) - len(vals))
    return sum(vals) / len(vals), min(vals)


def _league_slot_medians(teams: List[Any]) -> Dict[str, float]:
    per_slot: Dict[str, List[float]] = {s: [] for s in SLOTS}
    for tm in teams:
        lineup = _nhl_lineup_by_group(tm)
        for slot in SLOTS:
            per_slot[slot].append(_slot_value(lineup, slot)[0])
    return {s: (median(v) if v else 70.0) for s, v in per_slot.items()}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _profile_direction(league: Any, tid: str) -> str:
    prof = (getattr(league, "cpu_franchise_profiles", None) or {}).get(tid) or {}
    return str(prof.get("team_direction") or prof.get("competitive_window") or "").upper()


def _preseason_status(team: Any, direction: str) -> str:
    window = str(getattr(team, "gm_window", "") or "").lower()
    if direction in ("DEEP_REBUILD",):
        return STATUS_TANK
    if direction in ("REBUILDING", "SELLER", "CAP_CORRECTION") or window == "rebuild":
        return STATUS_SELLER
    if direction in ("CONTENDER", "ALL_IN_CONTENDER", "PLAYOFF_BUYER") or window == "contender":
        return STATUS_CONTENDER
    return STATUS_BUBBLE


def _standings_status(
    *,
    rank: int,
    gap_pts: float,
    gp: int,
    n_teams: int,
    preseason: str,
) -> str:
    if gp < 12:
        return preseason
    if rank <= max(4, n_teams // 5) or gap_pts >= 10.0:
        status = STATUS_CONTENDER
    elif gap_pts >= -5.0:
        status = STATUS_BUBBLE
    elif rank > n_teams - 3 or gap_pts <= -16.0:
        status = STATUS_TANK
    else:
        status = STATUS_SELLER
    # Before ~half-season the front office's plan still carries weight.
    if gp < 40:
        if preseason == STATUS_CONTENDER and status in (STATUS_BUBBLE, STATUS_SELLER) and gap_pts >= -8.0:
            status = STATUS_CONTENDER if gap_pts >= -2.0 else STATUS_BUBBLE
        if preseason in (STATUS_SELLER, STATUS_TANK) and status == STATUS_BUBBLE and gap_pts < 3.0:
            status = STATUS_SELLER
    return status


# ---------------------------------------------------------------------------
# Surplus
# ---------------------------------------------------------------------------


def _organization_players(team: Any) -> List[Tuple[Any, str]]:
    out: List[Tuple[Any, str]] = [(p, "nhl") for p in list(getattr(team, "roster", None) or [])]
    for p in list(getattr(team, "ahl_roster", None) or []):
        if player_holds_nhl_spc(p):
            out.append((p, "ahl"))
    return out


def _expected_cap_hit(ovr: float) -> float:
    return max(0.85, (ovr - 64.0) * 0.42)


def _find_surplus(
    team: Any,
    lineup: Dict[str, List[Any]],
    *,
    status: str,
    slot_avgs: Dict[str, float],
    medians: Dict[str, float],
    cap_tight: bool,
) -> Tuple[List[SurplusItem], List[Any]]:
    items: List[SurplusItem] = []
    seen: set = set()

    nhl_goalie_count = sum(1 for q in list(getattr(team, "roster", None) or []) if position_group(q) == "G")

    def add(p: Any, reason: str, group: str, priority: float) -> None:
        pid = player_pid(p)
        if not pid or pid in seen:
            return
        # Never shop the last NHL goalie (rules forbid leaving a club without one).
        if group == "G" and nhl_goalie_count < 2 and any(q is p for q in (getattr(team, "roster", None) or [])):
            return
        seen.add(pid)
        items.append(SurplusItem(player=p, reason=reason, group=group, priority=max(0.0, min(1.0, priority))))

    org = _organization_players(team)
    nhl_by_key: Dict[str, List[Any]] = {"C": [], "W": [], "D": [], "G": []}
    for p, loc in org:
        if loc == "nhl" and not getattr(p, "retired", False):
            nhl_by_key[_group_key(position_group(p))].append(p)
    for k in nhl_by_key:
        nhl_by_key[k].sort(key=player_rating, reverse=True)

    # Players the club will never shop.
    core: set = set()
    for k, plist in nhl_by_key.items():
        keep = {"C": 2, "W": 3, "D": 2, "G": 1}[k]
        core.update(player_pid(p) for p in plist[:keep])
    for p, _ in org:
        if player_age(p) <= 23 and (player_ovr(p) >= 78 or potential_ovr(p) >= 86):
            if status in (STATUS_SELLER, STATUS_TANK):
                core.add(player_pid(p))  # rebuilders keep young core

    # Flags first — these move regardless of depth.
    for p, _ in org:
        if bool(getattr(p, "_trade_demand_active", False)) or bool(getattr(p, "locker_room_disruptor", False)):
            add(p, "locker_room", position_group(p), 0.9)

    # Depth beyond the dressed lineup (+1 spare).
    for k, plist in nhl_by_key.items():
        extra = plist[LINEUP_COUNTS[k] + (1 if k != "G" else 0):]
        for p in extra:
            if player_ovr(p) >= 68 and player_pid(p) not in core:
                add(p, "goalie" if k == "G" else "depth", position_group(p), 0.55)

    # Positional strength: a club well above the league at a slot can spare its last man there.
    for slot, (group, lo, hi) in SLOTS.items():
        if slot in ("F_BOTTOM6", "D_BOTTOM", "G_START"):
            continue
        if slot_avgs.get(slot, 0.0) - medians.get(slot, 70.0) >= 4.0:
            key = _group_key(group)
            plist = nhl_by_key.get(key) or []
            if len(plist) > LINEUP_COUNTS[key]:
                cand = plist[LINEUP_COUNTS[key] - 1]
                if player_pid(cand) not in core:
                    add(cand, "positional", position_group(cand), 0.45)

    # Third NHL-calibre goalie anywhere in the org.
    goalies = sorted((p for p, _ in org if position_group(p) == "G"), key=player_ovr, reverse=True)
    for g in goalies[2:]:
        if player_ovr(g) >= 68:
            add(g, "goalie", "G", 0.6)

    # Blocked prospect on a club that is trying to win.
    if status in (STATUS_CONTENDER, STATUS_BUBBLE):
        for p, loc in org:
            if player_age(p) > 23 or player_pid(p) in core:
                continue
            key = _group_key(position_group(p))
            ahead = [q for q in nhl_by_key.get(key, []) if player_age(q) >= 24 and player_rating(q) > player_rating(p)]
            if loc == "ahl" and len(ahead) >= LINEUP_COUNTS[key] - 1 and potential_ovr(p) >= 76:
                add(p, "blocked_prospect", position_group(p), 0.5)

    # Non-contenders: their best expiring veterans; tankers: veterans generally.
    if status in (STATUS_SELLER, STATUS_TANK):
        nhl = [p for p, loc in org if loc == "nhl" and not (player_pid(p) in core and player_ovr(p) >= 86)]
        rentals = sorted((p for p in nhl if is_rental(p) and player_ovr(p) >= 72), key=player_ovr, reverse=True)
        for p in rentals[:4]:
            add(p, "expiring_veteran", position_group(p), 0.85 if status == STATUS_TANK else 0.75)
        if status == STATUS_TANK:
            vets = sorted((p for p in nhl if player_age(p) >= 29 and player_ovr(p) >= 74), key=player_ovr, reverse=True)
            for p in vets[:3]:
                add(p, "veteran_selloff", position_group(p), 0.6)

    # Lineup regulars who are underperforming their rating this season — the GM is
    # open to moving them (a change of scenery), not just its 13th forward.
    for k, plist in nhl_by_key.items():
        if k == "G":
            continue
        for p in plist[: LINEUP_COUNTS[k]]:
            if player_pid(p) in core:
                continue
            f = player_form(p)
            if f <= -1.5:
                add(p, "underperformer", position_group(p), min(0.7, 0.35 + (-f) * 0.06))

    # Sellers listen on any non-core veteran with term, not only expiring rentals.
    if status in (STATUS_SELLER, STATUS_TANK):
        vets = [
            p for p, loc in org
            if loc == "nhl" and player_pid(p) not in core and player_age(p) >= 27
            and player_ovr(p) >= 72 and not is_rental(p)
        ]
        vets.sort(key=player_rating, reverse=True)
        for p in vets[:3]:
            add(p, "seller_vet", position_group(p), 0.45 if status == STATUS_SELLER else 0.55)

    # Lineup churn: the next man up (13th F / 7th D) is outplaying the weakest-rated
    # regular whose OVR says he belongs ahead of him — that regular becomes movable.
    # One per position group at most, and only when the data actually says so.
    for k, plist in nhl_by_key.items():
        if k == "G" or len(plist) <= LINEUP_COUNTS[k]:
            continue
        nxt = plist[LINEUP_COUNTS[k]]
        if player_form(nxt) < 1.0:
            continue
        regs = [q for q in plist[: LINEUP_COUNTS[k]] if player_pid(q) not in core and player_ovr(q) > player_ovr(nxt)]
        if not regs:
            continue
        weakest = min(regs, key=player_rating)
        if player_rating(nxt) >= player_rating(weakest) - 0.5:
            add(weakest, "lineup_churn", position_group(weakest), 0.35)

    # Contracts the club would rather not carry.
    for p, loc in org:
        if loc != "nhl" or player_pid(p) in core:
            continue
        over = cap_hit_m(p) - _expected_cap_hit(player_ovr(p))
        if over >= 1.75 and contract_years_left(p) >= 2:
            add(p, "bad_contract", position_group(p), 0.7 if cap_tight else 0.35)

    # Prospects a buyer will spend (not core, not the only prospect at a position).
    spend: List[Any] = []
    if status in (STATUS_CONTENDER, STATUS_BUBBLE):
        pool = list(getattr(team, "prospect_pool", None) or [])
        ahl_young = [p for p, loc in org if loc == "ahl" and player_age(p) <= 23]
        for p in pool + ahl_young:
            if player_pid(p) in core:
                continue
            spend.append(p)
        spend.sort(key=potential_ovr, reverse=True)
    return items, spend[:6]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _cap_space(team: Any, league: Any) -> float:
    """Usable cap space from the authoritative cap snapshot (season cap table, LTIR,
    buried/retained/dead money). The old version used ``league.salary_cap_m or 88``,
    which lags at $88M on 2025+ franchises and made nearly every club look capped out."""
    try:
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        sy = getattr(league, "season_year", None)
        label = f"{int(sy)}-{(int(sy) + 1) % 100:02d}" if sy else None
        snap = calculate_team_cap_snapshot(team, league=league, season_label=label)
        return float(snap.get("usableCapSpace", 0.0) or 0.0)
    except Exception:
        try:
            from app.sim_engine.economy.cap_engine import team_active_roster_cap_hit_millions

            cap = _safe_float(getattr(league, "salary_cap_m", None), 0.0) or 88.0
            return cap - float(team_active_roster_cap_hit_millions(team))
        except Exception:
            return 0.0


def assess_team(
    team: Any,
    league: Any,
    *,
    medians: Dict[str, float],
    standings_rank: Dict[str, Tuple[int, float, float, int]],
    n_teams: int,
    deadline_phase: float,
) -> TeamAssessment:
    tid = team_id_of(team)
    direction = _profile_direction(league, tid)
    preseason = _preseason_status(team, direction)
    rank, pts_pct, gap_pts, gp = standings_rank.get(tid, (n_teams // 2, 0.5, 0.0, 0))
    status = _standings_status(rank=rank, gap_pts=gap_pts, gp=gp, n_teams=n_teams, preseason=preseason)

    a = TeamAssessment(team_id=tid, status=status, points_pct=pts_pct, playoff_gap_pts=gap_pts, games_played=gp, rank=rank)

    # Deadline psychology.
    if deadline_phase >= 0.3 and gp >= 30:
        expected_contender = preseason == STATUS_CONTENDER
        if expected_contender and gap_pts < 2.0 and status != STATUS_TANK:
            a.panic_buyer = True
            a.notes.append("Expected contender sliding toward the playoff line")
            if status == STATUS_SELLER and gap_pts > -9.0:
                a.status = STATUS_BUBBLE
        if a.status == STATUS_BUBBLE and -6.0 <= gap_pts < 0.0:
            a.chaser = True
            a.notes.append("Chasing the last playoff spot")
        if deadline_phase >= 0.5 and a.status == STATUS_BUBBLE and gap_pts < -6.0 and not a.panic_buyer:
            a.late_seller = True
            a.status = STATUS_SELLER
            a.notes.append("Fell out of the race — flipping to seller")

    a.cap_space_m = round(_cap_space(team, league), 2)
    cap_pressure = getattr(team, "cap_pressure", "")
    a.cap_tight = a.cap_space_m < 1.5 or str(cap_pressure).lower() in ("cap_hell", "critical")

    lineup = _nhl_lineup_by_group(team)
    slot_avgs: Dict[str, float] = {}
    for slot in SLOTS:
        avg, floor = _slot_value(lineup, slot)
        slot_avgs[slot] = avg
        a.slot_floor[slot] = floor
        gap = medians.get(slot, 70.0) - avg
        need = max(0.0, min(1.0, gap / NEED_FULL_GAP))
        if a.status == STATUS_CONTENDER or a.panic_buyer:
            need = min(1.0, need * 1.2 + (0.1 if gap > -1.5 else 0.0))  # contenders chase marginal upgrades
        a.needs[slot] = round(need, 3)
    # Injury holes hit hardest where the lineup has no cover.
    try:
        from app.sim_engine.economy.team_needs import _injury_need_boost

        boost = _injury_need_boost(list(getattr(team, "roster", None) or []))
        if boost.get("goalie"):
            a.needs["G_START"] = max(a.needs["G_START"], float(boost["goalie"]))
        if boost.get("top_4_defense"):
            for s in ("LD_TOP4", "RD_TOP4"):
                a.needs[s] = max(a.needs[s], float(boost["top_4_defense"]) * 0.8)
        if boost.get("top_line_forward"):
            for s in ("C_TOP2", "W_TOP6"):
                a.needs[s] = max(a.needs[s], float(boost["top_line_forward"]) * 0.8)
    except Exception:
        pass

    a.surplus, a.spend_prospects = _find_surplus(
        team, lineup, status=a.status, slot_avgs=slot_avgs, medians=medians, cap_tight=a.cap_tight,
    )
    prof = (getattr(league, "cpu_franchise_profiles", None) or {}).get(tid) or {}
    ideo = prof.get("ideology") or {}
    a.protects_first = _safe_float(ideo.get("draft_pick_protection"), 0.5) >= 0.62 and not a.panic_buyer
    return a


def _standings_table(league: Any, teams: List[Any]) -> Dict[str, Tuple[int, float, float, int]]:
    """tid → (rank, pts_pct, gap to 16th place in points, games played)."""
    snap = getattr(league, "_cpu_standings_snapshot", None) or {}
    rows = []
    for tm in teams:
        tid = team_id_of(tm)
        r = snap.get(tid) or {}
        rows.append((tid, _safe_float(r.get("pts_pct"), 0.5), _safe_int(r.get("gp"), 0)))
    rows.sort(key=lambda r: -r[1])
    n = len(rows)
    line_idx = min(n - 1, max(0, n // 2 - 1))
    line_pct = rows[line_idx][1] if rows else 0.5
    out: Dict[str, Tuple[int, float, float, int]] = {}
    for i, (tid, pct, gp) in enumerate(rows):
        out[tid] = (i + 1, pct, (pct - line_pct) * 2.0 * gp, gp)
    return out


def assess_league(
    league: Any,
    *,
    calendar_cursor: int,
    deadline_phase: float,
    days_to_deadline: int = 99,
    force: bool = False,
) -> Dict[str, TeamAssessment]:
    """Assess every club. Cached weekly; daily inside the last two weeks before the deadline."""
    global _FORM
    cache = getattr(league, "_cpu_assessments", None)
    cached_form = getattr(league, "_cpu_form", None)
    if isinstance(cached_form, dict):
        _FORM = cached_form
    refresh_every = 1 if 0 <= days_to_deadline <= 14 else 7
    if (
        not force
        and isinstance(cache, dict)
        and isinstance(cache.get("by_team"), dict)
        and 0 <= int(calendar_cursor) - int(cache.get("day", -999)) < refresh_every
        and int(cache.get("n_trades", -1)) == len(getattr(league, "trade_history", None) or [])
    ):
        return cache["by_team"]
    teams = list(getattr(league, "teams", None) or [])
    try:
        _FORM = compute_form(league, teams)
    except Exception:
        _FORM = {}
    try:
        setattr(league, "_cpu_form", _FORM)
    except Exception:
        pass
    medians = _league_slot_medians(teams)
    table = _standings_table(league, teams)
    out = {
        team_id_of(tm): assess_team(
            tm, league, medians=medians, standings_rank=table, n_teams=len(teams), deadline_phase=deadline_phase,
        )
        for tm in teams
        if team_id_of(tm)
    }
    try:
        setattr(
            league,
            "_cpu_assessments",
            {"day": int(calendar_cursor), "by_team": out, "medians": medians, "n_trades": len(getattr(league, "trade_history", None) or [])},
        )
    except Exception:
        pass
    return out


def slot_for_player(player: Any, assessment: TeamAssessment) -> Tuple[str, float]:
    """Best slot this player would take on the assessed team, and his OVR gain over its weakest occupant."""
    group = position_group(player)
    ovr = player_rating(player)
    if group == "G":
        cands = ["G_START"]
    elif group == "C":
        cands = ["C_TOP2", "W_TOP6", "F_BOTTOM6"]
    elif group == "W":
        cands = ["W_TOP6", "F_BOTTOM6"]
    else:
        cands = [f"{group}_TOP4", "D_BOTTOM"]
    best_slot, best_score, best_gain = cands[-1], -1.0, 0.0
    for slot in cands:
        gain = ovr - float(assessment.slot_floor.get(slot, 60.0))
        if gain <= 0:
            continue
        score = assessment.needs.get(slot, 0.0) * min(1.0, gain / 5.0)
        if score > best_score:
            best_slot, best_score, best_gain = slot, score, gain
    return best_slot, max(0.0, best_gain)


def need_fill_score(player: Any, assessment: TeamAssessment) -> Tuple[float, str]:
    """0–1: how much this player fixes a real hole for the assessed team."""
    slot, gain = slot_for_player(player, assessment)
    if gain <= 0:
        return 0.0, slot
    return round(assessment.needs.get(slot, 0.0) * min(1.0, gain / 5.0), 3), slot
