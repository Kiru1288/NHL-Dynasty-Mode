"""
Needs-and-surplus deal matching for CPU-CPU trades.

Every plan starts from a reason a real GM would call about:

* a buyer's lineup hole  ↔  another club's surplus (sale for futures, or a hockey swap
  where the return fills the seller's own hole)
* a cap-strapped club dumping a bad contract on a club with room (pick attached)
* a club moving a disruptive player out of the room

Plans are scored (need filled, how well it fits both clubs' status, deadline urgency)
and only plans above ``MIN_PLAN_SCORE`` are attempted. No reason → no trade.

Package construction depends on what the SELLER wants:
  seller / tank          → picks sized to value (+ a prospect on big deals)
  contender / bubble     → a player who fills the seller's own need, topped up with a pick

Deadline psychology feeds the buyer's ``premium`` — the share over value it will pay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.sim_engine.trades.team_assessment import (
    SLOT_LABELS,
    STATUS_BUBBLE,
    STATUS_CONTENDER,
    STATUS_SELLER,
    STATUS_TANK,
    SurplusItem,
    TeamAssessment,
    cap_hit_m,
    is_rental,
    need_fill_score,
    player_age,
    player_ovr,
    player_pid,
    position_group,
)
from app.sim_engine.trades.trade_asset import team_id_of

MIN_PLAN_SCORE = 0.32
MIN_NEED_FILL = 0.12
MAX_PLANS_PER_BUYER = 4
#: Deals sellers time for the deadline.
SELLOFF_MOTIVES = frozenset({"rental_purchase", "tank_selloff", "seller_futures", "late_selloff"})
#: Each GM's own read of each player (per save, stable within it). Without this every save
#: that starts from the same Real NHL rosters ranked the same deals first and moved the
#: same players.
GM_TASTE_SPREAD = 0.22
#: Plans kept per target player (his best two destinations), so one player can't crowd the list.
MAX_PLANS_PER_TARGET = 2


def gm_taste(league: Any, team_id: str, player: Any) -> float:
    """Stable per-save preference of club ``team_id`` for ``player`` in [-spread, +spread]."""
    import zlib

    salt = getattr(league, "_cpu_gm_taste_salt", None)
    if salt is None:
        rng = getattr(league, "rng", None)
        try:
            salt = int(rng.randint(1, 2_000_000_000)) if hasattr(rng, "randint") else 0
        except Exception:
            salt = 0
        if not salt:
            import random as _random

            salt = _random.randrange(1, 2_000_000_000)
        try:
            setattr(league, "_cpu_gm_taste_salt", int(salt))
        except Exception:
            pass
    key = f"{salt}|{team_id}|{player_pid(player)}".encode("utf-8")
    u = (zlib.crc32(key) & 0xFFFFFFFF) / 0xFFFFFFFF
    return (u * 2.0 - 1.0) * GM_TASTE_SPREAD


def _daily_jitter(rng: Any) -> float:
    """Day-to-day noise (mean ≈ the old +0..0.12 so plan volume is unchanged)."""
    return rng.random() * 0.20 - 0.04


@dataclass
class DealPlan:
    motive: str
    buyer: Any  # receives ``target``
    seller: Any  # sends ``target``
    target: Any
    score: float
    need_slot: str = ""
    need_fill: float = 0.0
    premium: float = 0.0  # buyer pays up to (1 + premium) × value
    motivated_seller: bool = False
    return_player: Optional[Any] = None
    buyer_picks: List[Dict[str, Any]] = field(default_factory=list)
    seller_pick: Optional[Dict[str, Any]] = None
    target_value: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    reason_text: str = ""
    trade_category: str = ""
    fail_reason: str = ""


@dataclass
class MatcherTools:
    """Proposer-side helpers (kept out of this module to avoid import cycles)."""

    player_value: Callable[[Any, Any, Any], float]  # (player, from_team, to_team) → value
    pick_for_value: Callable[..., Tuple[Optional[Dict[str, Any]], float]]
    tradeable: Callable[[Any, str], bool]  # (player, acquiring_team_id) → bool
    team_abbr: Callable[[Any], str]
    # (seller, buyer, target, return_player) → value of the roster player going back (0 if none)
    filler_value: Callable[[Any, Any, Any, Any], float] = lambda *_a: 0.0


def buyer_premium(a: TeamAssessment, *, deadline_phase: float, days_to_deadline: int, rental: bool) -> float:
    """Share over value a buyer will pay — the deadline overpay."""
    p = 0.0
    if a.status == STATUS_CONTENDER:
        p += 0.18 * deadline_phase
    if a.panic_buyer:
        p += 0.12 + 0.30 * deadline_phase
    if a.chaser:
        p += 0.10 * deadline_phase
    if days_to_deadline == 0:
        p += 0.08
    if rental and deadline_phase >= 0.6:
        p += 0.05  # rentals go for more the closer the deadline
    return round(min(0.55, p), 3)


def _status_fit(buyer: TeamAssessment, seller: TeamAssessment, item: SurplusItem) -> float:
    """How natural this pairing is for both front offices (0–1)."""
    fit = 0.0
    if buyer.is_buyer:
        fit += 0.5
    elif buyer.status == STATUS_BUBBLE:
        fit += 0.25
    if seller.is_seller:
        fit += 0.4
    elif item.reason in ("depth", "positional", "goalie", "blocked_prospect", "bad_contract", "locker_room"):
        fit += 0.3
    # Rebuilders don't buy veterans; contenders don't buy blocked kids.
    if buyer.status == STATUS_TANK:
        fit -= 0.6
    if buyer.status == STATUS_SELLER and player_age(item.player) >= 27:
        fit -= 0.4
    return max(0.0, min(1.0, fit))


def _motive_for(buyer: TeamAssessment, seller: TeamAssessment, item: SurplusItem, *, deadline_phase: float) -> str:
    if item.reason == "locker_room":
        return "locker_room"
    if buyer.panic_buyer and deadline_phase >= 0.3:
        return "panic_buy"
    if seller.status == STATUS_TANK and item.reason in ("expiring_veteran", "veteran_selloff"):
        return "tank_selloff"
    if seller.late_seller:
        return "late_selloff"
    if seller.is_seller and is_rental(item.player) and deadline_phase >= 0.2:
        return "rental_purchase"
    if seller.is_seller:
        return "seller_futures"
    return "hockey_swap"


_REASON_BY_MOTIVE = {
    "panic_buy": "PANIC_BUY",
    "tank_selloff": "TANK_SELLOFF",
    "late_selloff": "PLAYOFF_ODDS_COLLAPSE",
    "rental_purchase": "DEADLINE_RENTAL",
    "seller_futures": "REBUILDING_FUTURES",
    "hockey_swap": "NEED_FOR_SURPLUS_SWAP",
    "locker_room": "LOCKER_ROOM_DISRUPTOR_MOVED",
    "cap_dump": "CAP_RELIEF",
    "depth_add": "PLAYOFF_DEPTH",
}

_CATEGORY_BY_MOTIVE = {
    "panic_buy": "desperation_trade",
    "tank_selloff": "tank_trade",
    "late_selloff": "deadline_selloff",
    "rental_purchase": "deadline_rental",
    "seller_futures": "futures_trade",
    "hockey_swap": "hockey_trade",
    "locker_room": "locker_room_trade",
    "cap_dump": "cap_trade",
    "depth_add": "depth_trade",
}


def _reason_text(plan: DealPlan, tools: MatcherTools, buyer_a: TeamAssessment, seller_a: TeamAssessment) -> str:
    b = tools.team_abbr(plan.buyer)
    s = tools.team_abbr(plan.seller)
    name = str(getattr(plan.target, "name", None) or "the player")
    slot = SLOT_LABELS.get(plan.need_slot, "depth")
    over = f" — paid a {int(round(plan.premium * 100))}% deadline premium" if plan.premium >= 0.1 else ""
    m = plan.motive
    if m == "panic_buy":
        return f"{b}, sliding toward the playoff line, pushed chips in for {name} to shore up their {slot}{over}."
    if m == "tank_selloff":
        return f"{s} kept stripping the roster for futures; {b} landed {name} for their {slot}."
    if m == "late_selloff":
        return f"{s} fell out of the race and flipped to sellers — {name} heads to {b}."
    if m == "rental_purchase":
        return f"{b} rented {name} to fill their {slot} for the stretch run{over}."
    if m == "seller_futures":
        return f"{s} turned {name} into futures; {b} fills their {slot}."
    if m == "locker_room":
        return f"{s} moved {name} out of a fractured room; {b} takes the gamble."
    if m == "cap_dump":
        return f"{s} bought cap room by attaching a pick to {name}; {b} had space to absorb him."
    if m == "depth_add":
        return f"{b} added {name} as {slot} insurance for the stretch run."
    ret = str(getattr(plan.return_player, "name", None) or "")
    tail = f" for {ret}" if ret else ""
    return f"{b} filled their {slot} with {name}{tail} — a surplus-for-need swap."


def _pick_bundle(
    tools: MatcherTools,
    league: Any,
    team: Any,
    ctx: Dict[str, Any],
    *,
    target: float,
    protect_first: bool,
    max_picks: int = 2,
    max_firsts: int = 1,
    acquirer: Any = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Picks summing close to ``target`` value (sized to the player — not always a 1st)."""
    picks: List[Dict[str, Any]] = []
    total = 0.0
    exclude: set = set()
    remaining = target
    firsts = 0
    for _ in range(max_picks):
        if remaining < 4.0:
            break
        row, val = tools.pick_for_value(
            league, team, ctx=ctx, target=remaining, exclude_pick_ids=exclude, protect_first=protect_first,
            no_first=firsts >= max_firsts, acquirer=acquirer,
        )
        if row is None:
            break
        picks.append(row)
        if int(row.get("round") or 7) == 1:
            firsts += 1
        exclude.add(str(row.get("pick_id") or ""))
        total += val
        remaining = target - total
        if remaining < 0.25 * target:
            break
    return picks, total


def _spare_roster_players(team: Any, a: TeamAssessment) -> List[Tuple[Any, str]]:
    """NHL players a club can part with in a hockey trade: beyond its top guys at a
    position, and not at a slot where the club itself has a real need."""
    from app.sim_engine.trades.team_assessment import SLOTS, slot_for_player

    keep = {"C": 2, "W": 3, "LD": 1, "RD": 1, "G": 1}
    by_group: Dict[str, List[Any]] = {}
    for p in list(getattr(team, "roster", None) or []):
        by_group.setdefault(position_group(p), []).append(p)
    out: List[Tuple[Any, str]] = []
    for g, plist in by_group.items():
        if g == "G":
            continue
        plist.sort(key=player_ovr, reverse=True)
        for p in plist[keep.get(g, 2):]:
            slot, _ = slot_for_player(p, a)
            if a.needs.get(slot, 0.0) >= 0.35:
                continue
            out.append((p, "spare"))
    return out


def _young_currency(
    tools: MatcherTools, plan: DealPlan, ba: TeamAssessment, shortfall: float, used: set,
) -> Optional[Tuple[Any, float]]:
    """Prospect or spare young NHL player (≤25) whose value best covers ``shortfall``."""
    sid = team_id_of(plan.seller)
    pool: List[Any] = list(ba.spend_prospects)
    pool += [p for p, _ in _spare_roster_players(plan.buyer, ba) if player_age(p) <= 25]
    best: Optional[Tuple[Any, float]] = None
    best_err = float("inf")
    for p in pool:
        if player_pid(p) in used or p is plan.target or not tools.tradeable(p, sid):
            continue
        v = tools.player_value(p, plan.buyer, plan.seller)
        if v <= 0.0 or v > shortfall * 1.6:
            continue
        err = abs(v - shortfall)
        if err < best_err:
            best, best_err = (p, v), err
    return best


def _best_return_player(
    tools: MatcherTools,
    *,
    buyer_a: TeamAssessment,
    seller_a: TeamAssessment,
    buyer: Any,
    seller: Any,
    target_value: float,
    used: set,
) -> Tuple[Optional[Any], float, float]:
    """Buyer surplus that fills the SELLER's need, valued near (≤) the target. (player, value, fill)."""
    best: Tuple[Optional[Any], float, float] = (None, 0.0, 0.0)
    best_score = 0.0
    sid = team_id_of(seller)
    # Labelled surplus first, then any spare the buyer is deep at (not his top players,
    # not a slot he himself needs).
    pool: List[SurplusItem] = list(buyer_a.surplus)
    listed = {player_pid(i.player) for i in pool}
    for p, reason in _spare_roster_players(buyer, buyer_a):
        if player_pid(p) not in listed:
            pool.append(SurplusItem(player=p, reason=reason, group=position_group(p), priority=0.3))
    for item in pool:
        p = item.player
        if player_pid(p) in used or item.reason == "locker_room":
            continue
        fill, _ = need_fill_score(p, seller_a)
        if fill < MIN_NEED_FILL and item.reason != "blocked_prospect":
            continue
        if seller_a.is_seller and player_age(p) >= 28:
            continue  # rebuilders want youth back, not another veteran
        if not tools.tradeable(p, sid):
            continue
        val = tools.player_value(p, buyer, seller)
        if val > target_value * 1.15 or val < target_value * 0.35:
            continue
        score = fill + (0.35 if item.reason == "blocked_prospect" and seller_a.is_seller else 0.0) + val / max(1.0, target_value) * 0.3
        if score > best_score:
            best, best_score = (p, val, fill), score
    return best


HOCKEY_MIN_FILL = 0.10
HOCKEY_NEED_FLOOR = 0.20
DEPTH_SLOTS = ("F_BOTTOM6", "D_BOTTOM", "G_START")


def _tradeable_pool(team: Any, a: TeamAssessment, used: set) -> List[Any]:
    """Labelled surplus + spares — what a club will put in a hockey trade."""
    seen: set = set()
    out: List[Any] = []
    # Never swap away a club's top-two NHL goalies (its tandem).
    nhl_goalies = sorted(
        (g for g in list(getattr(team, "roster", None) or []) if position_group(g) == "G"), key=player_ovr, reverse=True,
    )
    tandem = {player_pid(g) for g in nhl_goalies[:2]}
    for p in [i.player for i in a.surplus if i.reason != "locker_room"] + [p for p, _ in _spare_roster_players(team, a)]:
        pid = player_pid(p)
        if pid in tandem:
            continue
        if pid and pid not in seen and pid not in used:
            seen.add(pid)
            out.append(p)
    return out


def _hockey_swap_plans(
    team_by_tid: Dict[str, Any],
    assessments: Dict[str, TeamAssessment],
    *,
    rng: Any,
    used_players: set,
    taste: Callable[[str, Any], float] = lambda _t, _p: 0.0,
) -> List[DealPlan]:
    """Mismatched depth: A's spare fills B's hole AND B's spare fills A's hole."""
    tids = [t for t in team_by_tid if t in assessments and assessments[t].status != STATUS_TANK]
    pools = {t: _tradeable_pool(team_by_tid[t], assessments[t], used_players) for t in tids}
    plans: List[DealPlan] = []
    for i, a_id in enumerate(tids):
        aa = assessments[a_id]
        for b_id in tids[i + 1:]:
            ba = assessments[b_id]
            # The buyer's own read of the player breaks near-ties (was always the argmax).
            best_x: Tuple[Any, float, str] = (None, 0.0, "")  # A -> B
            best_x_key = float("-inf")
            for x in pools[a_id]:
                fill, slot = need_fill_score(x, ba)
                if fill < HOCKEY_MIN_FILL or ba.needs.get(slot, 0.0) < HOCKEY_NEED_FLOOR:
                    continue
                key = fill + taste(b_id, x)
                if key > best_x_key:
                    best_x, best_x_key = (x, fill, slot), key
            if best_x[0] is None or best_x[1] < HOCKEY_MIN_FILL:
                continue
            best_y: Tuple[Any, float, str] = (None, 0.0, "")  # B -> A
            best_y_key = float("-inf")
            for y in pools[b_id]:
                if position_group(y) == position_group(best_x[0]):
                    continue  # a swap moves depth between positions, not like-for-like
                fill, slot = need_fill_score(y, aa)
                if fill < HOCKEY_MIN_FILL or aa.needs.get(slot, 0.0) < HOCKEY_NEED_FLOOR:
                    continue
                key = fill + taste(a_id, y)
                if key > best_y_key:
                    best_y, best_y_key = (y, fill, slot), key
            if best_y[0] is None or best_y[1] < HOCKEY_MIN_FILL:
                continue
            score = (
                0.2
                + 0.45 * (best_x[1] + best_y[1])
                + 0.5 * (taste(b_id, best_x[0]) + taste(a_id, best_y[0]))
                + _daily_jitter(rng)
            )
            plans.append(
                DealPlan(
                    motive="hockey_swap",
                    buyer=team_by_tid[b_id],
                    seller=team_by_tid[a_id],
                    target=best_x[0],
                    return_player=best_y[0],
                    score=round(score, 3),
                    need_slot=best_x[2],
                    need_fill=best_x[1],
                )
            )
    return plans


def _depth_market_plans(
    team_by_tid: Dict[str, Any],
    assessments: Dict[str, TeamAssessment],
    *,
    rng: Any,
    used_players: set,
    taste: Callable[[str, Any], float] = lambda _t, _p: 0.0,
) -> List[DealPlan]:
    """Deadline-week insurance: contenders/bubble clubs add depth from other clubs' spare depth."""
    buyers = [
        t for t in team_by_tid
        if t in assessments and (assessments[t].is_buyer or assessments[t].status == STATUS_BUBBLE)
    ]
    supply: List[Tuple[str, Any]] = []
    for tid, tm in team_by_tid.items():
        a = assessments.get(tid)
        if a is None:
            continue
        # AHL veterans on NHL deals + labelled NHL depth surplus.
        for p in list(getattr(tm, "ahl_roster", None) or []):
            if player_age(p) >= 24 and player_pid(p) not in used_players:
                supply.append((tid, p))
        for item in a.surplus:
            if item.reason in ("depth", "goalie", "positional", "expiring_veteran") and player_pid(item.player) not in used_players:
                supply.append((tid, item.player))
    plans: List[DealPlan] = []
    for bid in buyers:
        ba = assessments[bid]
        ranked: List[Tuple[float, str, Any, str]] = []
        for sid, p in supply:
            if sid == bid:
                continue
            g = position_group(p)
            slot = "G_START" if g == "G" else ("D_BOTTOM" if g in ("LD", "RD") else "F_BOTTOM6")
            if slot == "G_START":
                # Backup/insurance goalie only when the tandem is thin.
                if ba.needs.get("G_START", 0.0) < 0.05:
                    continue
            # Insurance = a comparable depth body, not necessarily an upgrade.
            gain = player_ovr(p) - float(ba.slot_floor.get(slot, 60.0))
            if gain < -1.5:
                continue
            fit = min(1.0, (gain + 2.0) / 6.0) * (0.5 + ba.needs.get(slot, 0.0))
            ranked.append((fit + taste(bid, p), sid, p, slot))
        ranked.sort(key=lambda r: -r[0])
        for fit, sid, p, slot in ranked[:3]:
            plans.append(
                DealPlan(
                    motive="depth_add",
                    buyer=team_by_tid[bid],
                    seller=team_by_tid[sid],
                    target=p,
                    score=round(0.34 + 0.3 * fit + _daily_jitter(rng), 3),
                    need_slot=slot,
                    need_fill=round(fit, 3),
                )
            )
    return plans


def generate_plans(
    league: Any,
    *,
    teams: List[Any],
    assessments: Dict[str, TeamAssessment],
    ctx: Dict[str, Any],
    tools: MatcherTools,
    rng: Any,
    used_players: set,
) -> List[DealPlan]:
    deadline_phase = float(ctx.get("deadline_phase") or 0.0)
    days_left = int(ctx.get("days_to_deadline", 99) or 99)
    team_by_tid = {team_id_of(t): t for t in teams}
    plans: List[DealPlan] = []

    def taste(tid: str, player: Any) -> float:
        return gm_taste(league, tid, player)

    # Everyone's surplus, indexed once.
    market: List[Tuple[Any, TeamAssessment, SurplusItem]] = []
    for tid, tm in team_by_tid.items():
        a = assessments.get(tid)
        if a is None:
            continue
        for item in a.surplus:
            if player_pid(item.player) in used_players or item.reason == "bad_contract":
                continue
            market.append((tm, a, item))

    # 1) Buyer holes ↔ surplus.
    for bid, buyer in team_by_tid.items():
        ba = assessments.get(bid)
        if ba is None or ba.status == STATUS_TANK:
            continue
        deadline_week = 0 <= days_left <= 7
        # Deadline week: contenders also shop for depth/insurance on small holes.
        need_floor = (0.04 if deadline_week else 0.15) if ba.is_buyer else 0.30
        min_fill = 0.04 if (deadline_week and ba.is_buyer) else MIN_NEED_FILL
        if not ba.top_needs(need_floor):
            continue
        cands: List[DealPlan] = []
        for seller, sa, item in market:
            sid = team_id_of(seller)
            if sid == bid:
                continue
            fill, slot = need_fill_score(item.player, ba)
            if fill < min_fill or ba.needs.get(slot, 0.0) < need_floor:
                continue
            status_fit = _status_fit(ba, sa, item)
            if status_fit <= 0.0:
                continue
            motive = _motive_for(ba, sa, item, deadline_phase=deadline_phase)
            urgency = 0.0
            if ba.is_buyer:
                urgency += 0.25 * deadline_phase
            if ba.panic_buyer:
                urgency += 0.2
            score = (
                fill * 0.9
                + status_fit * 0.35
                + item.priority * 0.25
                + urgency
                + taste(bid, item.player)
                + _daily_jitter(rng)
            )
            cands.append(
                DealPlan(
                    motive=motive,
                    buyer=buyer,
                    seller=seller,
                    target=item.player,
                    score=round(score, 3),
                    need_slot=slot,
                    need_fill=fill,
                    premium=buyer_premium(ba, deadline_phase=deadline_phase, days_to_deadline=days_left, rental=is_rental(item.player)),
                    motivated_seller=item.reason in ("locker_room",) or sa.status == STATUS_TANK or sa.late_seller,
                )
            )
        cands.sort(key=lambda p: -p.score)
        plans.extend(cands[: MAX_PLANS_PER_BUYER * (3 if deadline_week else 1)])

    # 2) Cap dumps: tight club + bad contract → club with room that is not trying to win now.
    for sid, seller in team_by_tid.items():
        sa = assessments.get(sid)
        if sa is None or not sa.cap_tight:
            continue
        for item in sa.surplus:
            if item.reason != "bad_contract" or player_pid(item.player) in used_players:
                continue
            hit = cap_hit_m(item.player)
            takers = [
                (t, assessments[team_id_of(t)])
                for t in teams
                if team_id_of(t) != sid
                and team_id_of(t) in assessments
                and assessments[team_id_of(t)].cap_space_m >= hit + 1.0
                and assessments[team_id_of(t)].status in (STATUS_SELLER, STATUS_TANK, STATUS_BUBBLE)
            ]
            if not takers:
                continue
            taker, _ta = takers[int(rng.random() * len(takers)) % len(takers)]
            plans.append(
                DealPlan(
                    motive="cap_dump",
                    buyer=taker,
                    seller=seller,
                    target=item.player,
                    score=round(0.45 + item.priority * 0.3 + rng.random() * 0.1, 3),
                    motivated_seller=True,
                )
            )

    # 3) Hockey trades between clubs with mismatched depth (all season).
    plans.extend(_hockey_swap_plans(team_by_tid, assessments, rng=rng, used_players=used_players, taste=taste))
    # 4) Deadline-week depth/insurance adds.
    if 0 <= days_left <= 7:
        plans.extend(_depth_market_plans(team_by_tid, assessments, rng=rng, used_players=used_players, taste=taste))

    # Seller patience: rentals and sell-offs fetch the most at the deadline, so sellers
    # mostly hold their veterans until then (a few sell early).
    timing = 0.25 + 0.75 * deadline_phase
    plans = [
        p for p in plans
        if p.score >= MIN_PLAN_SCORE
        and (p.motive not in SELLOFF_MOTIVES or rng.random() < timing)
    ]
    plans.sort(key=lambda p: -p.score)
    # One surplus player who fits many clubs' holes (a spare goalie, say) used to fill the
    # top of the list with a plan per buyer, so he was the first deal tried in every save.
    per_target: Dict[str, int] = {}
    diverse: List[DealPlan] = []
    for p in plans:
        pid = player_pid(p.target)
        if per_target.get(pid, 0) >= MAX_PLANS_PER_TARGET:
            continue
        per_target[pid] = per_target.get(pid, 0) + 1
        diverse.append(p)
    return diverse


def build_package_for_plan(
    plan: DealPlan,
    league: Any,
    *,
    assessments: Dict[str, TeamAssessment],
    ctx: Dict[str, Any],
    tools: MatcherTools,
    used_players: set,
) -> bool:
    """Fill in the return side of ``plan`` according to what the seller wants."""
    ba = assessments[team_id_of(plan.buyer)]
    sa = assessments[team_id_of(plan.seller)]
    if not tools.tradeable(plan.target, team_id_of(plan.buyer)):
        return False
    value = tools.player_value(plan.target, plan.seller, plan.buyer)
    plan.target_value = value

    if plan.motive == "hockey_swap" and plan.return_player is not None:
        # Player-for-player; the light side tops up with a pick so values line up.
        if not tools.tradeable(plan.return_player, team_id_of(plan.seller)):
            return False
        rv = tools.player_value(plan.return_player, plan.buyer, plan.seller)
        if value <= 0.0 or rv <= 0.0 or not (0.40 * value <= rv <= 1.6 * value):
            plan.fail_reason = "values_too_far_apart"
            return False
        diff = value - rv
        if diff >= 5.0:
            # Light side tops up — up to two picks, no firsts for a depth-level gap.
            picks, _ = _pick_bundle(
                tools, league, plan.buyer, ctx, target=diff, protect_first=True, max_picks=2,
                max_firsts=1 if diff >= 25.0 else 0, acquirer=plan.seller,
            )
            plan.buyer_picks = picks
        elif diff <= -5.0:
            row, _ = tools.pick_for_value(league, plan.seller, ctx=ctx, target=-diff, protect_first=True, acquirer=plan.buyer)
            if row is not None:
                plan.seller_pick = row
    elif plan.motive == "depth_add":
        # Insurance depth for a mid/late pick (never a 1st).
        row, _ = tools.pick_for_value(
            league, plan.buyer, ctx=ctx, target=max(1.0, value), protect_first=True, no_first=True, acquirer=plan.seller,
        )
        if row is None:
            plan.fail_reason = "no_depth_pick"
            return False
        plan.buyer_picks = [row]
    elif plan.motive == "cap_dump":
        # Seller pays the taker to take the contract: pick sized to the negative value.
        sweetener = max(6.0, -value + 6.0) if value < 6.0 else 6.0
        row, _ = tools.pick_for_value(league, plan.seller, ctx=ctx, target=sweetener, protect_first=True)
        if row is None:
            return False
        plan.seller_pick = row
        # Taker's side is nominal ("future considerations"): its cheapest pick.
        token, _ = tools.pick_for_value(league, plan.buyer, ctx=ctx, target=0.5, protect_first=True)
        if token is None:
            return False
        plan.buyer_picks = [token]
    else:
        pay = max(0.0, value) * (1.0 + plan.premium)
        if plan.motive == "locker_room":
            pay *= 0.85  # moving a problem — the club takes a little less
        # Only contender-window clubs (or a panicking one) will move a near-term 1st;
        # everyone else pays in 2nds/3rds/prospects (the value engine refuses otherwise).
        buyer_window = str(getattr(plan.buyer, "gm_window", "") or "").lower()
        firsts_ok = buyer_window == "contender" or plan.motive == "panic_buy"
        if sa.is_seller or plan.motive == "locker_room":
            # Futures: picks sized to value, a prospect on the big ones.
            prospect = None
            if pay >= 40.0 and ba.spend_prospects:
                for pr in ba.spend_prospects:
                    if player_pid(pr) in used_players or not tools.tradeable(pr, team_id_of(plan.seller)):
                        continue
                    pv = tools.player_value(pr, plan.buyer, plan.seller)
                    if 0.2 * pay <= pv <= 0.75 * pay:
                        prospect = (pr, pv)
                        break
            remaining = pay
            if prospect is not None:
                plan.return_player = prospect[0]
                remaining -= prospect[1]
            # A full roster sends a depth player back — that's part of the payment.
            remaining -= tools.filler_value(plan.seller, plan.buyer, plan.target, plan.return_player)
            picks, total = _pick_bundle(
                tools, league, plan.buyer, ctx, target=remaining,
                protect_first=ba.protects_first,
                max_firsts=(2 if (player_ovr(plan.target) >= 85 or plan.motive == "panic_buy") else 1) if firsts_ok else 0,
                max_picks=3 if not firsts_ok else 2,
                acquirer=plan.seller,
            )
            plan.buyer_picks = picks
            filler_v = tools.filler_value(plan.seller, plan.buyer, plan.target, plan.return_player)
            paid = total + filler_v + (prospect[1] if prospect is not None else 0.0)
            if paid < 0.8 * pay and prospect is None:
                # Short on picks (spent earlier in the season): pay the gap with youth.
                young = _young_currency(tools, plan, ba, pay - paid, used_players)
                if young is not None:
                    plan.return_player, yv = young
                    prospect = young
                    filler_v = tools.filler_value(plan.seller, plan.buyer, plan.target, plan.return_player)
                    paid = total + filler_v + yv
            if plan.return_player is None and not picks and filler_v <= 0.0:
                return False
            # Can't afford it → not a real offer (saves the market pass an attempt).
            if paid < 0.8 * pay:
                plan.fail_reason = "cant_afford"
                return False
            # Sellers won't take a pile of small assets for a star.
            if value >= 75.0 and prospect is None and not any(int(r.get("round") or 7) == 1 for r in picks):
                plan.fail_reason = "star_needs_premium_asset"
                return False
        else:
            # Hockey trade: the return must fill the seller's own hole.
            ret, rv, _fill = _best_return_player(
                tools, buyer_a=ba, seller_a=sa, buyer=plan.buyer, seller=plan.seller,
                target_value=pay, used=used_players,
            )
            if ret is None:
                # Contender/bubble clubs will still move genuine surplus for futures.
                pay_left = pay - tools.filler_value(plan.seller, plan.buyer, plan.target, None)
                picks, _ = _pick_bundle(
                    tools, league, plan.buyer, ctx, target=pay_left, protect_first=ba.protects_first, acquirer=plan.seller,
                    max_firsts=1 if firsts_ok else 0, max_picks=2 if firsts_ok else 3,
                )
                if not picks:
                    return False
                plan.buyer_picks = picks
                if plan.motive == "hockey_swap":
                    plan.motive = "surplus_sale"
            else:
                plan.return_player = ret
                short = pay - rv
                if short >= 5.0:
                    picks, _ = _pick_bundle(
                        tools, league, plan.buyer, ctx, target=short, protect_first=True, max_picks=1, acquirer=plan.seller,
                        max_firsts=1 if firsts_ok else 0,
                    )
                    plan.buyer_picks = picks

    code = _REASON_BY_MOTIVE.get(plan.motive, "ROSTER_BALANCE")
    plan.reason_codes = [code] + ([f"NEED_{plan.need_slot}"] if plan.need_slot else [])
    if plan.premium >= 0.1:
        plan.reason_codes.append("DEADLINE_OVERPAY")
    plan.trade_category = _CATEGORY_BY_MOTIVE.get(plan.motive, "hockey_trade")
    plan.reason_text = _reason_text(plan, tools, ba, sa)
    return True


# ---------------------------------------------------------------------------
# Post-deadline: AHL-only depth swaps
# ---------------------------------------------------------------------------

AHL_MIN = {"G": 2, "D": 6, "F": 11}


def _ahl_group(player: Any) -> str:
    g = position_group(player)
    return "D" if g in ("LD", "RD") else ("G" if g == "G" else "F")


def plan_ahl_depth_swap(teams: List[Any], *, rng: Any, used_players: set) -> Optional[Tuple[Any, Any, Any, Any]]:
    """(team_a, player_from_a, team_b, player_from_b): each club fixes an AHL shortage with the other's spare."""
    counts: Dict[str, Dict[str, List[Any]]] = {}
    for tm in teams:
        by: Dict[str, List[Any]] = {"G": [], "D": [], "F": []}
        for p in list(getattr(tm, "ahl_roster", None) or []):
            if player_pid(p) in used_players:
                continue
            by[_ahl_group(p)].append(p)
        counts[team_id_of(tm)] = by
    order = list(teams)
    rng.shuffle(order)
    for a in order:
        ca = counts.get(team_id_of(a)) or {}
        short = [g for g, n in AHL_MIN.items() if len(ca.get(g, [])) < n]
        spare_a = [g for g, n in AHL_MIN.items() if len(ca.get(g, [])) > n]
        if not short or not spare_a:
            continue
        need = short[0]
        for b in order:
            if b is a:
                continue
            cb = counts.get(team_id_of(b)) or {}
            if len(cb.get(need, [])) <= AHL_MIN[need]:
                continue
            for give_group in spare_a:
                if len(cb.get(give_group, [])) >= AHL_MIN[give_group] + 1:
                    continue  # b doesn't want more of what a is giving
                pa = sorted(ca[give_group], key=player_ovr)[0]
                pb = sorted(cb[need], key=player_ovr)[0]
                if abs(player_ovr(pa) - player_ovr(pb)) <= 6.0:
                    return a, pa, b, pb
    return None
