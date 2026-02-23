# backend/app/sim_engine/draft_sim/draft_sim.py
"""
Draft Simulation Orchestrator (NHL-grade, emergent, non-deterministic)

GOALS
- Simulate a realistic NHL Entry Draft from a franchise-sim perspective.
- Operate on *perception* (team draft boards), not truth.
- Enable: lottery -> board building -> on-the-clock decisions -> trades -> picks -> narratives.
- Keep it modular and testable. This file is the orchestrator + core glue.

HARD RULES
✅ Draft boards are noisy/biased per team (scouting ≠ truth).
✅ Teams make mistakes.
✅ Trades can be emotional/irrational sometimes.
✅ "Best player available" is not always selected.
❌ This module does NOT assign NHL ratings.
❌ This module does NOT simulate junior seasons.
❌ This module does NOT decide final career outcomes.
   (That happens later in projection/convert-to-player systems.)

DEPENDENCY PHILOSOPHY
- Prefer Python stdlib only for the simulation logic.
- Integrate with your existing Team/Prospect entities via duck-typing / Protocols.
- If your codebase has Team/Prospect dataclasses, this should "just work" as long as
  they expose the attributes used below.

INTEGRATION NOTES (expected attributes)
Prospect-like object:
- pid: str (unique id)
- name: str
- position: str  (e.g., "C", "LW", "RW", "D", "G")
- shoots: str
- age: int
- signals (optional): dict of "ceil", "floor", "certainty", etc.
- traits (optional): dict of personality traits in [0,1] (coachability, volatility, ego, etc.)
- risk (optional): float in [0,1] (higher = riskier)
- hype (optional): float in [0,1] (higher = more public hype)
- certainty (optional): float in [0,1] (higher = more certain projection)

Team-like object:
- tid: str (unique id, e.g. "TEAM_06")
- name: str
- style: str ("draft_and_develop", "win_now", etc.)
- context (optional): dict-like with keys used below (pressure, stability, dev_modifier, etc.)
- coach (optional): object or dict-like with risk_tolerance, pace_preference, lost_room, etc.

If your actual entities differ, either:
- adapt in a small adapter layer, OR
- change `get_*` helper functions below to read from your real structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import random


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DraftPick:
    year: int
    overall: int
    round: int
    original_team_id: str
    current_team_id: str


@dataclass
class DraftTrade:
    year: int
    timestamp: int  # pick counter at which it happened
    team_a: str
    team_b: str
    assets_a: List[str]  # human readable
    assets_b: List[str]
    note: str = ""


@dataclass
class DraftSelection:
    year: int
    pick: DraftPick
    team_id: str
    prospect_id: str
    prospect_name: str
    position: str
    rationale: str
    tag: str  # "steal" / "reach" / "chalk" / "panic" / "fit" etc.


@dataclass
class DraftNarrative:
    year: int
    selection_overall: int
    key: str
    text: str


@dataclass
class TeamDraftBoardEntry:
    prospect_id: str
    score: float
    tier: int
    do_not_draft: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class TeamDraftBoard:
    team_id: str
    entries: List[TeamDraftBoardEntry]  # sorted by score desc
    tiers: Dict[int, List[str]] = field(default_factory=dict)


@dataclass
class DraftSimConfig:
    year: int = 2025
    rounds: int = 7
    teams: int = 32

    # Lottery
    lottery_top_n: int = 16
    lottery_winners: int = 2
    lottery_noise: float = 0.15  # how chaotic lottery is relative to weights

    # Boards / scouting
    base_board_noise: float = 0.12     # per-team perception noise
    market_pressure_noise: float = 0.06  # extra noise from pressure environments
    mistake_rate: float = 0.08         # chance team makes a "bad" pick even if obvious
    do_not_draft_rate: float = 0.05    # baseline probability a team flags a prospect as DND

    # Trades
    enable_draft_trades: bool = True
    trade_chance_base: float = 0.05      # per pick chance a trade attempt triggers
    trade_chance_tier_gap: float = 0.10  # increases when tier gap is large
    trade_chance_target_fall: float = 0.12
    trade_max_attempts_per_pick: int = 2
    trade_value_fuzz: float = 0.12       # irrationality / imperfect valuation
    trade_accept_threshold: float = 0.02 # how close values need to be to accept

    # Pick value curve (approx)
    pick_value_power: float = 1.15

    # Narrative
    steal_threshold: float = 0.25   # relative difference between consensus and pick slot
    reach_threshold: float = 0.25

    # Determinism
    seed: Optional[int] = None


@dataclass
class DraftSimResult:
    year: int
    lottery_order_top16: List[str]
    draft_order: List[str]               # team_id per overall pick (length = rounds*teams)
    picks: List[DraftSelection]
    trades: List[DraftTrade]
    narratives: List[DraftNarrative]
    boards: Dict[str, TeamDraftBoard]    # per-team draft board (optional to keep)


# ---------------------------------------------------------------------------
# Helpers: safe attribute access / normalization
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x

def get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def get_nested(obj: Any, path: Tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for p in path:
        cur = get_attr(cur, p, None)
        if cur is None:
            return default
    return cur

def prospect_trait(p: Any, key: str, default: float = 0.5) -> float:
    # support p.traits dict OR direct attribute
    traits = get_attr(p, "traits", None)
    if isinstance(traits, dict) and key in traits:
        return float(traits.get(key, default))
    return float(get_attr(p, key, default))

def prospect_signal(p: Any, key: str, default: float = 0.5) -> float:
    signals = get_attr(p, "signals", None)
    if isinstance(signals, dict) and key in signals:
        return float(signals.get(key, default))
    return float(get_attr(p, key, default))

def team_context(t: Any, key: str, default: float = 0.5) -> float:
    ctx = get_attr(t, "context", None)
    if isinstance(ctx, dict) and key in ctx:
        return float(ctx.get(key, default))
    return float(get_attr(t, key, default))

def team_style(t: Any) -> str:
    return str(get_attr(t, "style", get_attr(t, "archetype", "balanced")))

def coach_risk_tolerance(t: Any, default: float = 0.5) -> float:
    # support t.coach.risk_tolerance OR t.coach dict OR t.context["coach_risk"] etc
    coach = get_attr(t, "coach", None)
    if coach is not None:
        v = get_attr(coach, "risk_tolerance", None)
        if v is not None:
            return float(v)
        if isinstance(coach, dict) and "risk_tolerance" in coach:
            return float(coach["risk_tolerance"])
    return float(team_context(t, "coach_risk_tolerance", default))

def coach_lost_room(t: Any) -> bool:
    coach = get_attr(t, "coach", None)
    if coach is None:
        return bool(team_context(t, "lost_room", False))
    v = get_attr(coach, "lost_room", None)
    if v is None and isinstance(coach, dict):
        v = coach.get("lost_room", None)
    if v is None:
        v = team_context(t, "lost_room", False)
    return bool(v)

def team_pressure(t: Any) -> float:
    # "market_pressure" or "team_pressure" both exist in your dumps
    return float(team_context(t, "market_pressure", team_context(t, "team_pressure", 0.5)))

def team_stability(t: Any) -> float:
    return float(team_context(t, "stability", 0.5))

def team_dev_modifier(t: Any) -> float:
    return float(team_context(t, "dev_modifier", 0.8))


# ---------------------------------------------------------------------------
# Lottery (simple but tunable)
# ---------------------------------------------------------------------------

def run_draft_lottery(team_ids: List[str], cfg: DraftSimConfig, rng: random.Random) -> List[str]:
    """
    Returns TOP16 order as a list of team_ids.

    Real NHL lottery is more complex. This is a tunable approximation:
    - Lower-ranked teams have higher weight
    - Two winners jump
    - Remaining teams slotted in order
    """
    top = team_ids[:cfg.lottery_top_n]

    # weight by rank (worse = higher). Add noise to avoid determinism.
    # rank 1 (worst) -> highest weight, rank 16 -> lowest weight.
    weights: List[float] = []
    for i, _tid in enumerate(top, start=1):
        base = (cfg.lottery_top_n + 1 - i) / cfg.lottery_top_n  # 1.0 down to ~0.0625
        noisy = base * (1.0 + rng.uniform(-cfg.lottery_noise, cfg.lottery_noise))
        weights.append(max(0.0001, noisy))

    # pick winners without replacement weighted
    remaining = top[:]
    remaining_weights = weights[:]
    winners: List[str] = []

    for _ in range(cfg.lottery_winners):
        total = sum(remaining_weights)
        r = rng.random() * total
        acc = 0.0
        pick_idx = 0
        for i, w in enumerate(remaining_weights):
            acc += w
            if acc >= r:
                pick_idx = i
                break
        winners.append(remaining.pop(pick_idx))
        remaining_weights.pop(pick_idx)

    # NHL-like: winners go to #1 and #2, rest follow in original order with winners removed
    result = winners + remaining
    return result


# ---------------------------------------------------------------------------
# Board Building (per-team perception + bias)
# ---------------------------------------------------------------------------

def _prospect_base_value(p: Any) -> float:
    """
    A compact "truth-ish" base value used only as a starting signal for perception.
    This should NOT be a deterministic OVR. It's a blend of ceiling/floor/certainty/hype.
    """
    ceil = prospect_signal(p, "ceiling", prospect_signal(p, "ceil", 0.6))
    floor = prospect_signal(p, "floor",  prospect_signal(p, "flr", 0.3))
    certainty = prospect_signal(p, "certainty", 0.5)
    hype = prospect_signal(p, "hype", 0.5)
    risk = prospect_signal(p, "risk", prospect_trait(p, "risk", 0.5))

    # ceiling matters more than floor, but certainty tempers it; hype adds slight boost.
    # risk reduces base slightly, but most of risk is handled via team-specific bias.
    base = 0.55 * ceil + 0.25 * floor + 0.15 * certainty + 0.05 * hype
    base *= (1.0 - 0.10 * clamp(risk))
    return clamp(base, 0.0, 1.0)

def _team_bias_weights(t: Any) -> Dict[str, float]:
    """
    Map team style/archetype into weighting preferences.
    """
    style = team_style(t).lower()

    # default balanced
    w_upside = 0.55
    w_safety = 0.45
    w_fit = 0.10

    if "win_now" in style or "contend" in style:
        w_upside = 0.45
        w_safety = 0.55
        w_fit = 0.18
    elif "draft" in style or "develop" in style or "rebuild" in style:
        w_upside = 0.62
        w_safety = 0.38
        w_fit = 0.12
    elif "chaos" in style:
        w_upside = 0.70
        w_safety = 0.30
        w_fit = 0.08

    # coach risk tolerance shifts upside appetite
    rt = clamp(coach_risk_tolerance(t, 0.5))
    w_upside = clamp(w_upside + (rt - 0.5) * 0.20, 0.30, 0.80)
    w_safety = clamp(1.0 - w_upside, 0.20, 0.70)

    return {"upside": w_upside, "safety": w_safety, "fit": w_fit}

def build_team_draft_board(
    team: Any,
    prospects: List[Any],
    cfg: DraftSimConfig,
    rng: random.Random,
) -> TeamDraftBoard:
    """
    Build a per-team board with:
    - perception noise
    - team bias (upside vs safety vs fit)
    - do-not-draft flags
    - tiering
    """
    bias = _team_bias_weights(team)
    pressure = team_pressure(team)
    stability = team_stability(team)
    dev = team_dev_modifier(team)
    lost_room = coach_lost_room(team)

    entries: List[TeamDraftBoardEntry] = []

    # team-specific DND intensity: unstable/lost room orgs are less tolerant of "headaches"
    dnd_boost = 0.05 + (1.0 - stability) * 0.08 + (0.10 if lost_room else 0.0)
    dnd_base = clamp(cfg.do_not_draft_rate + dnd_boost, 0.02, 0.25)

    for p in prospects:
        pid = str(get_attr(p, "pid", get_attr(p, "id", "")))
        name = str(get_attr(p, "name", pid))
        pos = str(get_attr(p, "position", "N/A"))

        # "truth-ish" signal, then perception noise
        base = _prospect_base_value(p)

        # derive upside and safety signals
        ceil = prospect_signal(p, "ceiling", prospect_signal(p, "ceil", 0.6))
        floor = prospect_signal(p, "floor",  prospect_signal(p, "flr", 0.3))
        certainty = prospect_signal(p, "certainty", 0.5)
        risk = prospect_signal(p, "risk", prospect_trait(p, "risk", 0.5))

        upside = clamp(ceil * (0.75 + 0.25 * certainty), 0.0, 1.0)
        safety = clamp((0.5 * floor + 0.5 * certainty) * (1.0 - 0.35 * risk), 0.0, 1.0)

        # fit: simple proxy using personality + dev environment
        coachability = prospect_trait(p, "coachability", 0.5)
        volatility = prospect_trait(p, "volatility", 0.5)
        ego = prospect_trait(p, "ego", 0.5)
        mental = prospect_trait(p, "mental_toughness", 0.5)
        adaptability = prospect_trait(p, "adaptability", 0.5)

        fit = 0.30 * coachability + 0.25 * mental + 0.20 * adaptability + 0.25 * dev
        fit -= 0.20 * volatility + 0.10 * ego
        fit = clamp(fit, 0.0, 1.0)

        # perception noise (pressure adds chaos; stable orgs reduce noise a bit)
        noise = cfg.base_board_noise
        noise += cfg.market_pressure_noise * (pressure - 0.5) * 2.0
        noise += 0.04 * (0.5 - stability)  # unstable -> more noise
        noise = max(0.0, noise)

        perceived = (
            0.55 * base
            + bias["upside"] * 0.35 * upside
            + bias["safety"] * 0.35 * safety
            + bias["fit"] * 0.10 * fit
        )

        perceived *= (1.0 + rng.uniform(-noise, noise))
        perceived = clamp(perceived, 0.0, 1.0)

        # DND logic (team-specific, prospect-specific red flags)
        dnd = False
        reasons: List[str] = []

        # baseline random DND
        if rng.random() < dnd_base:
            dnd = True
            reasons.append("internal red flag")

        # prospect-specific red flags (weighted by team context)
        # unstable/lost-room orgs fear volatility/ego more
        volatility_flag = volatility > (0.78 - 0.10 * (1.0 - stability))
        ego_flag = ego > (0.80 - 0.10 * (1.0 - stability))
        coachability_flag = coachability < (0.25 + 0.10 * (1.0 - dev))
        risk_flag = risk > (0.80 - 0.10 * coach_risk_tolerance(team, 0.5))

        if lost_room and volatility_flag:
            dnd = True
            reasons.append("volatile personality (room risk)")
        if ego_flag and pressure > 0.6:
            dnd = True
            reasons.append("ego/media risk in pressure market")
        if coachability_flag and dev < 0.7:
            dnd = True
            reasons.append("low coachability (dev mismatch)")
        if risk_flag and (team_style(team).lower().find("win") >= 0):
            # contenders avoid extreme risk more often
            dnd = True
            reasons.append("boom/bust too extreme for contender")

        entries.append(
            TeamDraftBoardEntry(
                prospect_id=pid,
                score=perceived,
                tier=0,
                do_not_draft=dnd,
                reasons=reasons,
            )
        )

    # sort by perceived score
    entries.sort(key=lambda e: e.score, reverse=True)

    # tiering: compress into tiers based on score gaps (simple heuristic)
    tiers: Dict[int, List[str]] = {}
    tier = 1
    last = None
    for e in entries:
        if last is None:
            e.tier = tier
            tiers.setdefault(tier, []).append(e.prospect_id)
            last = e.score
            continue

        gap = last - e.score
        # bigger gap => new tier
        if gap > 0.08 and tier < 6:
            tier += 1
        e.tier = tier
        tiers.setdefault(tier, []).append(e.prospect_id)
        last = e.score

    return TeamDraftBoard(team_id=str(get_attr(team, "tid", get_attr(team, "id", "TEAM"))), entries=entries, tiers=tiers)


# ---------------------------------------------------------------------------
# Pick Value Curve + Trade Engine (lightweight but believable)
# ---------------------------------------------------------------------------

def pick_value(overall_pick: int, cfg: DraftSimConfig) -> float:
    """
    Approximate pick value curve: early picks are exponentially more valuable.
    Returns value in (0,1].
    """
    # overall_pick starts at 1
    x = max(1, overall_pick)
    return 1.0 / (x ** cfg.pick_value_power)

def board_best_available(board: TeamDraftBoard, available_ids: set[str]) -> Optional[TeamDraftBoardEntry]:
    for e in board.entries:
        if e.prospect_id in available_ids and not e.do_not_draft:
            return e
    # if all filtered, allow DND as last resort
    for e in board.entries:
        if e.prospect_id in available_ids:
            return e
    return None

def board_find_target_tier(board: TeamDraftBoard, available_ids: set[str], max_tier: int = 2) -> Optional[TeamDraftBoardEntry]:
    # look for a "target" in top tiers still available
    for e in board.entries:
        if e.prospect_id in available_ids and e.tier <= max_tier and not e.do_not_draft:
            return e
    return None

def estimate_pick_tier_quality(board: TeamDraftBoard, available_ids: set[str]) -> Tuple[int, float]:
    """
    Returns (best_tier, best_score) for available players on this board.
    """
    best = board_best_available(board, available_ids)
    if best is None:
        return (999, 0.0)
    return (best.tier, best.score)

def try_trade_up(
    pick_idx: int,
    draft_order: List[str],
    teams_by_id: Dict[str, Any],
    boards: Dict[str, TeamDraftBoard],
    available_ids: set[str],
    cfg: DraftSimConfig,
    rng: random.Random
) -> Optional[Tuple[int, DraftTrade, List[str]]]:
    """
    Attempt a trade around current pick.
    Returns (new_pick_idx, trade, updated_draft_order) or None.
    """
    if not cfg.enable_draft_trades:
        return None

    cur_team_id = draft_order[pick_idx]
    cur_team = teams_by_id[cur_team_id]
    cur_board = boards[cur_team_id]

    # Determine if this team has a falling target / big tier gap
    cur_tier, cur_score = estimate_pick_tier_quality(cur_board, available_ids)
    target = board_find_target_tier(cur_board, available_ids, max_tier=1)

    # Look ahead a few picks
    lookahead = min(len(draft_order) - 1, pick_idx + 6)
    if lookahead <= pick_idx:
        return None

    # Trade trigger probability
    p = cfg.trade_chance_base
    if cur_tier >= 3:
        p += cfg.trade_chance_tier_gap  # board looks flat / desperation triggers moves
    if target is not None:
        p += cfg.trade_chance_target_fall

    # Risk tolerant orgs trade more
    p += (coach_risk_tolerance(cur_team, 0.5) - 0.5) * 0.06
    p = clamp(p, 0.0, 0.30)

    if rng.random() > p:
        return None

    # Choose a partner (someone ahead of you)
    possible_slots = list(range(pick_idx, lookahead + 1))
    # trading "up" means lower index
    partner_idx = pick_idx - rng.randint(1, min(4, pick_idx)) if pick_idx > 0 else None
    if partner_idx is None:
        return None

    partner_team_id = draft_order[partner_idx]
    if partner_team_id == cur_team_id:
        return None

    partner_team = teams_by_id[partner_team_id]

    # Value calculation: current team wants partner pick, partner wants compensation
    want_value = pick_value(partner_idx + 1, cfg)
    give_value = pick_value(pick_idx + 1, cfg)

    # Extra premium for moving up
    premium = (want_value - give_value) * (1.05 + rng.uniform(-cfg.trade_value_fuzz, cfg.trade_value_fuzz))
    premium = max(0.0, premium)

    # Partner willingness based on their style / risk / stability
    partner_style = team_style(partner_team).lower()
    partner_rt = coach_risk_tolerance(partner_team, 0.5)
    partner_stab = team_stability(partner_team)

    willingness = 0.02
    if "draft" in partner_style or "rebuild" in partner_style:
        willingness += 0.03
    if partner_rt < 0.45:
        willingness += 0.01  # conservative likes certainty of extra asset
    willingness += (partner_stab - 0.5) * 0.02
    willingness = clamp(willingness, 0.0, 0.08)

    # Accept if premium >= threshold-ish (with fuzz)
    accept_line = cfg.trade_accept_threshold + rng.uniform(-0.01, 0.02) - willingness
    if premium < accept_line:
        return None

    # Execute trade: swap pick owners for those slots (simple "move up / move down")
    updated = draft_order[:]
    updated[partner_idx], updated[pick_idx] = updated[pick_idx], updated[partner_idx]

    trade = DraftTrade(
        year=cfg.year,
        timestamp=pick_idx + 1,
        team_a=cur_team_id,
        team_b=partner_team_id,
        assets_a=[f"Pick {pick_idx+1} -> Pick {partner_idx+1}"],
        assets_b=[f"Pick {partner_idx+1} -> Pick {pick_idx+1}"],
        note="Draft-day pick swap (simplified premium model)",
    )

    return (partner_idx, trade, updated)


# ---------------------------------------------------------------------------
# Pick Logic (on-the-clock decision)
# ---------------------------------------------------------------------------

def decide_pick(
    pick_idx: int,
    team_id: str,
    teams_by_id: Dict[str, Any],
    board: TeamDraftBoard,
    available: Dict[str, Any],
    cfg: DraftSimConfig,
    rng: random.Random
) -> DraftSelection:
    """
    Select a prospect for a team at a given pick based on board + context.
    Includes mistakes, reaches, and "fit" decisions.
    """
    team = teams_by_id[team_id]
    available_ids = set(available.keys())

    # Candidate pool: top N available on their board
    pool: List[TeamDraftBoardEntry] = []
    for e in board.entries:
        if e.prospect_id in available_ids:
            pool.append(e)
        if len(pool) >= 12:
            break

    if not pool:
        # should never happen, but fail safe
        pid, p = next(iter(available.items()))
        return DraftSelection(
            year=cfg.year,
            pick=DraftPick(cfg.year, pick_idx + 1, (pick_idx // cfg.teams) + 1, team_id, team_id),
            team_id=team_id,
            prospect_id=pid,
            prospect_name=str(get_attr(p, "name", pid)),
            position=str(get_attr(p, "position", "N/A")),
            rationale="fallback pick (no board candidates)",
            tag="fallback",
        )

    # Mistake logic: sometimes pick isn't #1 on their board
    mistake = rng.random() < cfg.mistake_rate * (1.0 + (team_pressure(team) - 0.5) * 0.5)
    mistake = bool(mistake)

    # Fit logic: if team dev is strong, they may chase upside; if weak, they take safe
    dev = team_dev_modifier(team)
    rt = coach_risk_tolerance(team, 0.5)

    # weighting for selection sampling among the pool
    weights: List[float] = []
    for e in pool:
        # base weight from score
        w = max(0.0001, e.score)

        # penalize do-not-draft but still allow if nothing else
        if e.do_not_draft:
            w *= 0.20

        # style influences "upside" tolerance: use prospect risk/ceil vs floor
        p = available[e.prospect_id]
        ceil = prospect_signal(p, "ceiling", prospect_signal(p, "ceil", 0.6))
        floor = prospect_signal(p, "floor", prospect_signal(p, "flr", 0.3))
        risk = prospect_signal(p, "risk", prospect_trait(p, "risk", 0.5))
        cert = prospect_signal(p, "certainty", 0.5)

        upside = clamp(ceil * (0.75 + 0.25 * cert), 0.0, 1.0)
        safety = clamp((floor + cert) * 0.5 * (1.0 - 0.35 * risk), 0.0, 1.0)

        # if dev strong and risk tolerance high => boost upside candidates
        w *= (1.0 + (dev - 0.75) * 0.20 * upside)
        w *= (1.0 + (rt - 0.5) * 0.18 * upside)

        # if dev weak or risk tolerance low => boost safety
        w *= (1.0 + (0.75 - dev) * 0.18 * safety)
        w *= (1.0 + (0.5 - rt) * 0.16 * safety)

        weights.append(max(0.0001, w))

    # If not a mistake, heavily favor top of board
    if not mistake:
        # multiply weights by rank decay
        for i in range(len(weights)):
            weights[i] *= 1.0 / (1.0 + 0.28 * i)

    # sample from pool
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    chosen_entry = pool[0]
    for e, w in zip(pool, weights):
        acc += w
        if acc >= r:
            chosen_entry = e
            break

    chosen = available[chosen_entry.prospect_id]

    # rationale + tag
    rationale_bits = []
    if chosen_entry.do_not_draft:
        rationale_bits.append("ignored red flags (thin board)")
    if mistake:
        rationale_bits.append("on-the-clock variance / imperfect scouting")
    if team_pressure(team) > 0.62:
        rationale_bits.append("market pressure influenced decision")
    if coach_lost_room(team):
        rationale_bits.append("coach/room dynamics influenced 'safe' preference")

    # classify pick as chalk/reach/steal based on *team board* rank
    # rank within their own board
    board_rank = next((i for i, e in enumerate(board.entries) if e.prospect_id == chosen_entry.prospect_id), 999)
    if board_rank <= 2:
        tag = "chalk"
    elif board_rank <= 7:
        tag = "fit"
    else:
        tag = "reach" if not mistake else "panic"

    rationale = ", ".join(rationale_bits) if rationale_bits else "best available on team board"

    return DraftSelection(
        year=cfg.year,
        pick=DraftPick(cfg.year, pick_idx + 1, (pick_idx // cfg.teams) + 1, team_id, team_id),
        team_id=team_id,
        prospect_id=str(get_attr(chosen, "pid", get_attr(chosen, "id", ""))),
        prospect_name=str(get_attr(chosen, "name", chosen_entry.prospect_id)),
        position=str(get_attr(chosen, "position", "N/A")),
        rationale=rationale,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Narratives (steal/reach/slide)
# ---------------------------------------------------------------------------

def build_narratives(
    picks: List[DraftSelection],
    consensus_rank: Dict[str, int],
    cfg: DraftSimConfig
) -> List[DraftNarrative]:
    narratives: List[DraftNarrative] = []
    for sel in picks:
        pid = sel.prospect_id
        if pid not in consensus_rank:
            continue
        c_rank = consensus_rank[pid]  # 1-based
        actual = sel.pick.overall

        # relative difference
        # e.g., consensus 3 but picked 12 -> fell
        diff = (actual - c_rank) / max(1, c_rank)

        if diff >= cfg.steal_threshold:
            narratives.append(DraftNarrative(
                year=cfg.year,
                selection_overall=actual,
                key="steal",
                text=f"{sel.prospect_name} fell to #{actual} (consensus ~#{c_rank}). Labeled as a potential steal.",
            ))
        elif diff <= -cfg.reach_threshold:
            narratives.append(DraftNarrative(
                year=cfg.year,
                selection_overall=actual,
                key="reach",
                text=f"{sel.prospect_name} went #{actual} (consensus ~#{c_rank}). Seen as a reach or strong internal conviction pick.",
            ))
    return narratives


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def draft_sim(
    teams: List[Any],
    prospects: List[Any],
    cfg: Optional[DraftSimConfig] = None
) -> DraftSimResult:
    """
    Run full draft sim.

    Parameters
    - teams: list of team objects (must expose tid/id and optional context/coach/style)
    - prospects: list of prospect objects (must expose pid/id, name, position, optional traits/signals)
    - cfg: DraftSimConfig

    Returns DraftSimResult with:
    - lottery order top16
    - full draft order for all picks
    - selections
    - trades
    - narratives
    - boards (per-team)
    """
    cfg = cfg or DraftSimConfig()
    rng = random.Random(cfg.seed)

    # Team id mapping
    team_ids: List[str] = []
    teams_by_id: Dict[str, Any] = {}
    for t in teams:
        tid = str(get_attr(t, "tid", get_attr(t, "id", "")))
        if not tid:
            raise ValueError("Team missing tid/id")
        team_ids.append(tid)
        teams_by_id[tid] = t

    if cfg.teams != len(team_ids):
        # not fatal; adapt to provided list
        cfg.teams = len(team_ids)

    # Build a crude "standings order" as provided order:
    # team_ids[0] = worst, team_ids[-1] = best (caller decides).
    standings_order = team_ids[:]

    # Lottery top16
    lottery_top16 = run_draft_lottery(standings_order, cfg, rng)

    # Build round 1 order:
    # - top16 replaced by lottery result
    # - remaining remain in standings order (minus top16 teams)
    remaining = [tid for tid in standings_order if tid not in standings_order[:cfg.lottery_top_n]]
    round1_order = lottery_top16 + remaining

    # Build full draft order (snake could be used for later rounds; NHL uses same order each round)
    draft_order: List[str] = []
    for _r in range(cfg.rounds):
        draft_order.extend(round1_order)

    # Available prospects map
    available: Dict[str, Any] = {}
    for p in prospects:
        pid = str(get_attr(p, "pid", get_attr(p, "id", "")))
        if not pid:
            raise ValueError("Prospect missing pid/id")
        available[pid] = p

    # Consensus rank = sort prospects by base value (this is NOT "truth", just public-ish)
    # Used only for narratives (steal/reach).
    consensus_sorted = sorted(list(available.values()), key=_prospect_base_value, reverse=True)
    consensus_rank: Dict[str, int] = {}
    for i, p in enumerate(consensus_sorted, start=1):
        pid = str(get_attr(p, "pid", get_attr(p, "id", "")))
        consensus_rank[pid] = i

    # Build boards
    boards: Dict[str, TeamDraftBoard] = {}
    for tid in team_ids:
        boards[tid] = build_team_draft_board(teams_by_id[tid], list(available.values()), cfg, rng)

    trades: List[DraftTrade] = []
    picks: List[DraftSelection] = []

    # Draft loop
    pick_idx = 0
    while pick_idx < len(draft_order) and available:
        # Trade attempts
        if cfg.enable_draft_trades:
            for _ in range(cfg.trade_max_attempts_per_pick):
                attempt = try_trade_up(
                    pick_idx=pick_idx,
                    draft_order=draft_order,
                    teams_by_id=teams_by_id,
                    boards=boards,
                    available_ids=set(available.keys()),
                    cfg=cfg,
                    rng=rng
                )
                if attempt is None:
                    continue
                new_idx, trade, updated_order = attempt
                trades.append(trade)
                draft_order = updated_order
                pick_idx = new_idx
                break  # after a successful trade, proceed to pick

        team_id = draft_order[pick_idx]
        board = boards[team_id]

        sel = decide_pick(
            pick_idx=pick_idx,
            team_id=team_id,
            teams_by_id=teams_by_id,
            board=board,
            available=available,
            cfg=cfg,
            rng=rng
        )

        # If we accidentally chose something not available (shouldn't happen), fallback.
        if sel.prospect_id not in available:
            # fallback to best available by board
            best = board_best_available(board, set(available.keys()))
            if best is None:
                pid, p = next(iter(available.items()))
            else:
                pid = best.prospect_id
                p = available[pid]
            sel.prospect_id = pid
            sel.prospect_name = str(get_attr(p, "name", pid))
            sel.position = str(get_attr(p, "position", "N/A"))
            sel.rationale = "fallback due to availability mismatch"
            sel.tag = "fallback"

        # remove from available
        chosen_obj = available.pop(sel.prospect_id, None)
        if chosen_obj is None:
            # extremely defensive: pick someone else
            pid, p = next(iter(available.items()))
            sel.prospect_id = pid
            sel.prospect_name = str(get_attr(p, "name", pid))
            sel.position = str(get_attr(p, "position", "N/A"))
            sel.rationale = "fallback (already taken)"
            sel.tag = "fallback"
            available.pop(pid, None)

        picks.append(sel)
        pick_idx += 1

    narratives = build_narratives(picks, consensus_rank, cfg)

    return DraftSimResult(
        year=cfg.year,
        lottery_order_top16=lottery_top16,
        draft_order=draft_order,
        picks=picks,
        trades=trades,
        narratives=narratives,
        boards=boards,
    )


# ---------------------------------------------------------------------------
# Convenience: pretty printers (optional)
# ---------------------------------------------------------------------------

def summarize_draft(result: DraftSimResult, top_n: int = 16) -> str:
    lines = []
    lines.append(f"==== DRAFT {result.year} SUMMARY ====")
    lines.append("Lottery Top16 Order:")
    for i, tid in enumerate(result.lottery_order_top16[:top_n], start=1):
        lines.append(f"  #{i:02d}: {tid}")
    lines.append("")
    lines.append("Top Picks:")
    for sel in result.picks[:top_n]:
        lines.append(
            f"  #{sel.pick.overall:02d} {sel.team_id} selects {sel.prospect_name} ({sel.position}) [{sel.tag}]"
        )
    if result.trades:
        lines.append("")
        lines.append(f"Trades ({len(result.trades)}):")
        for tr in result.trades[:min(10, len(result.trades))]:
            lines.append(f"  t={tr.timestamp} {tr.team_a} <-> {tr.team_b} | {tr.note}")
    if result.narratives:
        lines.append("")
        lines.append("Narratives:")
        for n in result.narratives[:min(12, len(result.narratives))]:
            lines.append(f"  #{n.selection_overall:02d} [{n.key}] {n.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Minimal self-test (only runs if executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny smoke test with stub teams/prospects (safe to delete)
    @dataclass
    class StubTeam:
        tid: str
        name: str
        style: str = "draft_and_develop"
        context: Dict[str, float] = field(default_factory=lambda: {
            "market_pressure": 0.5,
            "stability": 0.5,
            "dev_modifier": 0.8,
        })
        coach: Dict[str, Any] = field(default_factory=lambda: {
            "risk_tolerance": 0.5,
            "lost_room": False,
        })

    @dataclass
    class StubProspect:
        pid: str
        name: str
        position: str
        age: int = 18
        signals: Dict[str, float] = field(default_factory=dict)
        traits: Dict[str, float] = field(default_factory=dict)

    teams = [StubTeam(tid=f"TEAM_{i:02d}", name=f"Team {i}") for i in range(1, 33)]
    prospects = []
    for i in range(1, 250):
        prospects.append(
            StubProspect(
                pid=f"P{i:03d}",
                name=f"Prospect {i}",
                position=random.choice(["C", "LW", "RW", "D", "G"]),
                signals={
                    "ceil": random.random(),
                    "flr": random.random() * 0.7,
                    "certainty": random.random(),
                    "risk": random.random(),
                    "hype": random.random(),
                },
                traits={
                    "coachability": random.random(),
                    "volatility": random.random(),
                    "ego": random.random(),
                    "mental_toughness": random.random(),
                    "adaptability": random.random(),
                },
            )
        )

    cfg = DraftSimConfig(year=2025, seed=123, rounds=7, teams=32)
    res = draft_sim(teams, prospects, cfg)
    print(summarize_draft(res, top_n=16))
