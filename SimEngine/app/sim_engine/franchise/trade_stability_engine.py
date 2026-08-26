"""Interconnected Trade Stability Score (0–100) and escalation levels.

Player concerns feed component pressures with personality × context interactions.
No single variable should independently trigger a formal trade demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.player_agent_engine import (
    ensure_player_agent,
    get_agent_gm_relationship,
)

STABILITY_STABLE_MIN = 70
STABILITY_ANGST_MIN = 55
STABILITY_APATHY_MIN = 40
STABILITY_ANGER_MIN = 20

CRISIS_DEADLINE_MAX = 360


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_0_100(raw: Any, default: float = 50.0) -> float:
    try:
        v = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        v = default
    if v <= 1.5:
        v *= 100.0
    return _clamp(v, 0.0, 100.0)


def _player_id(player: Any) -> str:
    return str(_get(player, "id", "") or _get(player, "player_id", "") or "")


def _player_ovr(player: Any) -> float:
    raw = _get(player, "ovr", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = 70.0
    try:
        o = float(raw if raw is not None else _get(player, "overall", 70) or 70)
    except Exception:
        o = 70.0
    if o <= 1.5:
        o *= 99.0
    return o


def _player_character_0_100(player: Any) -> int:
    try:
        chapters = player.get_chapter_ratings() if hasattr(player, "get_chapter_ratings") else {}
        if isinstance(chapters, dict) and chapters.get("Character"):
            return int(_clamp(float(chapters["Character"]), 55.0, 99.0))
    except Exception:
        pass
    c = getattr(player, "character", None)
    if c is not None:
        try:
            ci = int(c)
            if 20 <= ci <= 99:
                return ci
        except (TypeError, ValueError):
            pass
    tr = getattr(player, "traits", None)
    if tr is None:
        return 74
    blend = (
        0.22 * float(getattr(tr, "coachability", 0.5))
        + 0.20 * float(getattr(tr, "mental_toughness", 0.5))
        + 0.18 * float(getattr(tr, "work_ethic", 0.5))
        + 0.16 * float(getattr(tr, "leadership", 0.5))
        + 0.14 * float(getattr(tr, "competitiveness", 0.5))
        + 0.10 * (1.0 - float(getattr(tr, "volatility", 0.5)))
    )
    return int(round(_clamp(blend, 0.55, 0.95) * 100.0))


def _player_mental_0_100(player: Any) -> int:
    try:
        chapters = player.get_chapter_ratings() if hasattr(player, "get_chapter_ratings") else {}
        if isinstance(chapters, dict) and chapters.get("Mental"):
            return int(_clamp(float(chapters["Mental"]), 50.0, 99.0))
    except Exception:
        pass
    chem = getattr(player, "chemistry_profile", None) or {}
    if isinstance(chem, dict):
        for key in ("mental", "resilience", "adaptability"):
            if chem.get(key):
                return int(_clamp(float(chem[key]), 50.0, 99.0))
    tr = getattr(player, "traits", None)
    if tr is None:
        return 72
    blend = (
        0.45 * float(getattr(tr, "mental_toughness", 0.5))
        + 0.30 * (1.0 - float(getattr(tr, "volatility", 0.5)))
        + 0.25 * float(getattr(tr, "patience", 0.5))
    )
    return int(round(_clamp(blend, 0.45, 0.98) * 100.0))


def ensure_player_storyline_state(player: Any) -> Dict[str, Any]:
    st = getattr(player, "_franchise_storyline_state", None)
    if not isinstance(st, dict):
        st = {}
        setattr(player, "_franchise_storyline_state", st)
    for k, default in (
        ("trade_attempt_count", 0),
        ("was_recently_shopped", False),
        ("trade_rumor_heat", 0),
        ("gm_trust", 0.72),
        ("career_trade_demand_count", 0),
        ("season_trade_demand_count", 0),
        ("previous_trade_demand_severity", 0),
        ("previous_trade_demand_team", ""),
        ("previous_trade_demand_reason", ""),
        ("broken_promises", 0),
    ):
        st.setdefault(k, default)
    return st


def ensure_trade_stability_state(session: Any) -> Dict[str, Any]:
    book = getattr(session, "trade_stability_state", None)
    if not isinstance(book, dict):
        book = {}
        session.trade_stability_state = book
    return book


@dataclass
class PlayerConcernSnapshot:
    role_satisfaction: float = 50.0
    gm_trust: float = 50.0
    coach_trust: float = 50.0
    winning_satisfaction: float = 50.0
    competitiveness: float = 50.0
    character: float = 74.0
    mental: float = 72.0
    loyalty: float = 50.0
    ego: float = 50.0
    ambition: float = 50.0
    professionalism: float = 74.0
    resilience: float = 50.0
    contract_satisfaction: float = 55.0
    contract_security: float = 55.0
    trade_exposure: float = 0.0
    broken_promises: float = 0.0
    development_satisfaction: float = 55.0
    team_belonging: float = 55.0
    locker_room_relationships: float = 55.0
    leadership_treatment: float = 55.0
    performance_vs_deployment: float = 55.0
    career_stage_pressure: float = 0.0
    nhl_experience: float = 50.0
    has_ntc: bool = False
    family_satisfaction: float = 60.0
    relocation_strain: float = 0.0
    media_stress: float = 0.0
    organizational_direction: float = 55.0
    recent_team_performance: float = 50.0
    previous_trade_demands: int = 0
    agent_patience: float = 0.5
    agent_pressure: float = 0.0
    pressures: Dict[str, float] = field(default_factory=dict)


def _team_win_pct(session: Any, team: Any) -> float:
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    standings = getattr(session, "standings", None)
    try:
        rec = getattr(standings, "records", None) or {}
        row = rec.get(tid)
        if row is None:
            return 0.50
        wins = int(_get(row, "wins", 0) or 0)
        losses = int(_get(row, "losses", 0) or 0)
        otl = int(_get(row, "ot_losses", 0) or _get(row, "otl", 0) or 0)
        gp = wins + losses + otl
        if gp < 8:
            return 0.50
        pts = wins * 2 + otl
        return pts / max(1, gp * 2)
    except Exception:
        return 0.50


def gather_player_concerns(session: Any, player: Any, team: Any) -> PlayerConcernSnapshot:
    from app.sim_engine.systems.chemistry import ensure_player_chemistry_profile, safe_get_psych

    psych = safe_get_psych(player)
    chem = ensure_player_chemistry_profile(player)
    pst = ensure_player_storyline_state(player)
    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))

    role = _to_0_100(psych.get("role_satisfaction", 0.5) * 100.0 if psych.get("role_satisfaction", 0.5) <= 1.5 else psych.get("role_satisfaction", 50))
    morale = _to_0_100(psych.get("morale", 0.5) * 100.0 if psych.get("morale", 0.5) <= 1.5 else psych.get("morale", 50))
    conf = _to_0_100(psych.get("confidence", 0.5) * 100.0 if psych.get("confidence", 0.5) <= 1.5 else psych.get("confidence", 50))

    coach_trust_raw = getattr(getattr(player, "psych", None), "coach_trust", None)
    if coach_trust_raw is None:
        coach_trust = (conf + role) / 2.0
    else:
        coach_trust = _to_0_100(coach_trust_raw)

    win_pct = _team_win_pct(session, team)
    winning_sat = _clamp(35.0 + win_pct * 65.0, 0.0, 100.0)

    character = float(_player_character_0_100(player))
    mental = float(_player_mental_0_100(player))
    competitiveness = _to_0_100(chem.get("competitiveness", chem.get("compete", 50)))
    loyalty = _to_0_100(chem.get("loyalty", 50))
    ego = _to_0_100(getattr(getattr(player, "traits", None), "ego", 0.5) * 100.0)
    ambition = _to_0_100(chem.get("ambition", chem.get("drive", competitiveness)))
    resilience = _to_0_100(chem.get("resilience", chem.get("adaptability", mental)))
    belonging = _to_0_100(chem.get("belonging", chem.get("team_player", morale)))

    trade_exposure = min(
        100.0,
        int(pst.get("trade_rumor_heat") or 0) * 0.85 + int(pst.get("trade_attempt_count") or 0) * 8.0,
    )
    if pst.get("was_recently_shopped"):
        trade_exposure = min(100.0, trade_exposure + 12.0)

    gm_trust = _to_0_100(float(pst.get("gm_trust", 0.72)) * 100.0 if float(pst.get("gm_trust", 0.72)) <= 1.5 else pst.get("gm_trust", 72))

    contract = getattr(player, "contract", None)
    has_ntc = False
    contract_sat = 55.0
    contract_sec = 55.0
    if contract is not None:
        clause = str(
            _get(contract, "clause", "")
            or _get(contract, "clause_type", "")
            or _get(contract, "trade_clause", "")
            or ""
        ).upper()
        has_ntc = "NTC" in clause or "NMC" in clause or "NO MOVE" in clause or "NO TRADE" in clause
        yrs = int(_get(contract, "years_remaining", 0) or _get(contract, "term", 0) or 0)
        contract_sec = _clamp(40.0 + yrs * 8.0, 0.0, 100.0)

    age = int(getattr(getattr(player, "identity", None), "age", None) or getattr(player, "age", 27) or 27)
    ovr = _player_ovr(player)
    career_pressure = 0.0
    if age >= 32 and ovr >= 80:
        career_pressure = min(35.0, (age - 31) * 4.0)
    dev_sat = 70.0 if age >= 26 else _clamp(role + (winning_sat - 50) * 0.25, 0.0, 100.0)

    perf_vs_deploy = role
    if ovr >= 84 and role < 45:
        perf_vs_deploy = min(role, 35.0)

    prev_demands = int(pst.get("career_trade_demand_count") or 0)
    agent_patience = float(agent.get("patience", 0.5) or 0.5) + (float(gm_rel.get("agent_gm_trust", 0.55)) - 0.5) * 0.2

    return PlayerConcernSnapshot(
        role_satisfaction=role,
        gm_trust=gm_trust,
        coach_trust=coach_trust,
        winning_satisfaction=winning_sat,
        competitiveness=competitiveness,
        character=character,
        mental=mental,
        loyalty=loyalty,
        ego=ego,
        ambition=ambition,
        professionalism=character,
        resilience=resilience,
        contract_satisfaction=contract_sat,
        contract_security=contract_sec,
        trade_exposure=trade_exposure,
        broken_promises=float(int(pst.get("broken_promises") or 0) * 18.0),
        development_satisfaction=dev_sat,
        team_belonging=belonging,
        locker_room_relationships=belonging,
        leadership_treatment=coach_trust,
        performance_vs_deployment=perf_vs_deploy,
        career_stage_pressure=career_pressure,
        nhl_experience=min(100.0, max(0.0, (age - 18) * 4.5)),
        has_ntc=has_ntc,
        family_satisfaction=60.0,
        relocation_strain=0.0,
        media_stress=min(100.0, trade_exposure * 0.35),
        organizational_direction=winning_sat * 0.6 + gm_trust * 0.4,
        recent_team_performance=winning_sat,
        previous_trade_demands=prev_demands,
        agent_patience=_clamp(agent_patience, 0.05, 0.98),
        agent_pressure=max(0.0, (1.0 - agent_patience) * 20.0),
    )


def _dissatisfaction(satisfaction: float) -> float:
    return _clamp(100.0 - satisfaction, 0.0, 100.0)


def character_tolerance_bonus(character: float) -> float:
    c = float(character)
    if c >= 92:
        return 14.0
    if c >= 85:
        return 9.0
    if c >= 77:
        return 4.0
    if c >= 70:
        return 0.0
    if c >= 63:
        return -5.0
    if c >= 55:
        return -11.0
    return -16.0


def mental_resilience_bonus(mental: float) -> float:
    m = float(mental)
    if m >= 90:
        return 10.0
    if m >= 80:
        return 6.0
    if m >= 70:
        return 2.0
    if m >= 60:
        return -2.0
    return -8.0


def _role_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.role_satisfaction)
    sens = 0.85 + (snap.ego / 100.0) * 0.45
    if snap.competitiveness >= 75:
        if snap.winning_satisfaction >= 65:
            dissat *= 0.42
        elif snap.winning_satisfaction < 42:
            dissat *= 1.38
    if snap.character < 68:
        dissat *= 1.12
    elif snap.character >= 85:
        dissat *= 0.82
    return dissat * sens * 0.22


def _management_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (_dissatisfaction(snap.gm_trust) * 0.55 + _dissatisfaction(snap.coach_trust) * 0.45)
    if snap.loyalty >= 72 and snap.gm_trust >= 55:
        dissat *= 0.68
    return dissat * 0.20


def _winning_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.winning_satisfaction)
    weight = 0.12 + (snap.competitiveness / 100.0) * 0.18
    if snap.competitiveness >= 78 and snap.winning_satisfaction < 45:
        dissat *= 1.55
    return dissat * weight


def _contract_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (_dissatisfaction(snap.contract_satisfaction) * 0.6 + _dissatisfaction(snap.contract_security) * 0.4)
    return dissat * 0.14


def _development_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.development_satisfaction)
    if snap.ambition >= 72:
        dissat *= 1.28
    return dissat * 0.12


def _belonging_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (
        _dissatisfaction(snap.team_belonging) * 0.45
        + _dissatisfaction(snap.locker_room_relationships) * 0.35
        + _dissatisfaction(snap.leadership_treatment) * 0.20
    )
    return dissat * 0.14


def _trade_exposure_pressure(snap: PlayerConcernSnapshot) -> float:
    base = snap.trade_exposure
    if snap.mental >= 88 and snap.character >= 82:
        base *= 0.35
    elif snap.mental < 62:
        base *= 1.45
    elif snap.mental < 70:
        base *= 1.15
    if snap.character < 65:
        base *= 1.22
    return base * 0.18


def _personal_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (
        _dissatisfaction(snap.family_satisfaction) * 0.5
        + snap.relocation_strain * 0.25
        + snap.media_stress * 0.25
    )
    return dissat * 0.08


def _organizational_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.organizational_direction)
    return dissat * 0.10 + snap.career_stage_pressure * 0.06


def _performance_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.performance_vs_deployment)
    if snap.ego >= 75:
        dissat *= 1.18
    return dissat * 0.12


def compute_component_pressures(snap: PlayerConcernSnapshot) -> Dict[str, float]:
    pressures = {
        "role": _role_pressure(snap),
        "management": _management_pressure(snap),
        "winning": _winning_pressure(snap),
        "contract": _contract_pressure(snap),
        "development": _development_pressure(snap),
        "belonging": _belonging_pressure(snap),
        "trade_exposure": _trade_exposure_pressure(snap),
        "personal": _personal_pressure(snap),
        "coach": _management_pressure(snap) * 0.35,
        "organizational": _organizational_pressure(snap),
        "performance": _performance_pressure(snap),
    }
    if snap.broken_promises > 0:
        pressures["broken_promise"] = min(28.0, snap.broken_promises * 1.1)
    if snap.previous_trade_demands > 0:
        pressures["demand_history"] = min(18.0, snap.previous_trade_demands * 4.5)
    snap.pressures = pressures
    return pressures


def compute_trade_stability(snap: PlayerConcernSnapshot) -> Tuple[float, Dict[str, float]]:
    pressures = compute_component_pressures(snap)
    cumulative = sum(pressures.values()) + snap.agent_pressure * 0.35

    loyalty_buffer = (snap.loyalty / 100.0) * 6.0
    if snap.loyalty >= 75 and snap.gm_trust >= 58:
        loyalty_buffer += 4.0

    char_buf = character_tolerance_bonus(snap.character)
    mental_buf = mental_resilience_bonus(snap.mental)

    winning_mit = 0.0
    if snap.winning_satisfaction >= 62:
        winning_mit = min(8.0, (snap.winning_satisfaction - 60) * 0.15)

    relationship_mit = 0.0
    if snap.gm_trust >= 65 and snap.coach_trust >= 60:
        relationship_mit = min(7.0, (snap.gm_trust + snap.coach_trust - 120) * 0.08)

    agent_mod = (snap.agent_patience - 0.5) * 6.0

    score = 100.0 - cumulative + loyalty_buffer + char_buf + mental_buf + winning_mit + relationship_mit + agent_mod
    score = _clamp(score, 0.0, 100.0)
    return round(score, 2), pressures


def stability_to_escalation_level(stability: float) -> int:
    s = float(stability)
    if s >= STABILITY_STABLE_MIN:
        return 0
    if s >= STABILITY_ANGST_MIN:
        return 1
    if s >= STABILITY_APATHY_MIN:
        return 2
    if s >= STABILITY_ANGER_MIN:
        return 3
    return 4


def character_escalation_skip(character: float, stability: float) -> int:
    """Low character can skip warning stages."""
    c = float(character)
    if c >= 77:
        return 0
    if c < 58 and stability < STABILITY_ANGER_MIN + 8:
        return 2
    if c < 65 and stability < STABILITY_APATHY_MIN + 6:
        return 1
    return 0


def readiness_penalties(stability: float, character: float, mental: float, escalation: int) -> Dict[str, float]:
    """Mental stress + character disengagement (separate channels)."""
    if escalation < 2:
        return {"mental_stress": 0.0, "character_disengagement": 0.0, "ovr_readiness": 0.0}

    stress_base = max(0.0, (55.0 - float(stability)) * 0.06)
    mental_stress = stress_base * _clamp(1.35 - float(mental) / 100.0, 0.15, 1.25)

    disengage_base = max(0.0, (50.0 - float(character)) * 0.05) * (escalation - 1) * 0.35
    if escalation >= 4:
        disengage_base += 2.5
    character_disengagement = disengage_base * _clamp(1.2 - float(mental) / 120.0, 0.2, 1.0)

    ovr = -min(6.0, mental_stress + character_disengagement)
    return {
        "mental_stress": round(mental_stress, 2),
        "character_disengagement": round(character_disengagement, 2),
        "ovr_readiness": round(ovr, 2),
    }


def apply_readiness_to_player(player: Any, penalties: Dict[str, float]) -> None:
    ovr_pen = float(penalties.get("ovr_readiness") or 0.0)
    if ovr_pen >= 0:
        return
    st = ensure_player_storyline_state(player)
    st["trade_demand_readiness_penalty"] = ovr_pen
    try:
        setattr(player, "_trade_demand_readiness_penalty", ovr_pen)
    except Exception:
        pass


def clear_demand_temporary_modifiers(player: Any) -> None:
    st = ensure_player_storyline_state(player)
    for key in ("trade_demand_readiness_penalty",):
        st.pop(key, None)
    for attr in (
        "_trade_demand_readiness_penalty",
        "_trade_demand_active",
        "_systemic_trade_value_mult",
        "_crisis_trade_value_mult",
        "_crisis_distressed_asset",
        "_crisis_trade_stage",
        "locker_room_disruptor",
    ):
        try:
            if hasattr(player, attr):
                delattr(player, attr)
        except Exception:
            try:
                setattr(player, attr, False if attr == "locker_room_disruptor" else None)
            except Exception:
                pass


def update_player_stability(session: Any, player: Any, team: Any) -> Dict[str, Any]:
    snap = gather_player_concerns(session, player, team)
    score, pressures = compute_trade_stability(snap)
    escalation = stability_to_escalation_level(score)
    skip = character_escalation_skip(snap.character, score)
    effective_escalation = min(4, escalation + skip)
    penalties = readiness_penalties(score, snap.character, snap.mental, effective_escalation)
    apply_readiness_to_player(player, penalties)

    pid = _player_id(player)
    book = ensure_trade_stability_state(session)
    prev = book.get(pid) if isinstance(book.get(pid), dict) else {}
    prev_level = int(prev.get("escalation_level") or 0)
    row = {
        "player_id": pid,
        "trade_stability_score": score,
        "escalation_level": effective_escalation,
        "pressures": {k: round(v, 2) for k, v in pressures.items()},
        "character": int(snap.character),
        "mental": int(snap.mental),
        "readiness_penalties": penalties,
        "prev_escalation_level": prev_level,
    }
    book[pid] = row
    return row


def apply_trade_hub_exposure(
    session: Any,
    player: Any,
    *,
    attempt_n: int = 1,
    rejection_kind: str = "rejected",
) -> Dict[str, Any]:
    """Feed trade hub shopping into cumulative stability — rarely instant formal demand."""
    pst = ensure_player_storyline_state(player)
    if rejection_kind == "technical_no_fallout":
        return {"stability_delta": 0.0}

    mental = _player_mental_0_100(player)
    character = _player_character_0_100(player)

    if rejection_kind == "soft_blocked":
        heat_delta = 1
        stability_delta = -0.5
    else:
        heat_delta = 10 + min(8, attempt_n * 2)
        stability_delta = -1.5 - min(4.0, attempt_n * 0.75)
        if mental >= 88 and character >= 82:
            stability_delta = max(stability_delta, -2.0)
        elif mental < 62:
            stability_delta -= 2.5
        if character < 65:
            stability_delta -= 1.5

    pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + heat_delta)
    pst["was_recently_shopped"] = True
    gm_drop = 0.01 if rejection_kind == "soft_blocked" else 0.03 + min(0.06, attempt_n * 0.015)
    if mental >= 88:
        gm_drop *= 0.45
    pst["gm_trust"] = _clamp(float(pst.get("gm_trust", 0.72)) - gm_drop, 0.05, 1.0)

    team = None
    league = getattr(getattr(session, "sim", None), "league", None)
    pid = _player_id(player)
    if league is not None:
        for tm in getattr(league, "teams", None) or []:
            for p in getattr(tm, "roster", None) or []:
                if _player_id(p) == pid:
                    team = tm
                    break
            if team:
                break
    if team is not None:
        row = update_player_stability(session, player, team)
        book = ensure_trade_stability_state(session)
        cur = float(row.get("trade_stability_score") or 70.0)
        book[pid]["trade_stability_score"] = _clamp(cur + stability_delta, 0.0, 100.0)
        book[pid]["escalation_level"] = stability_to_escalation_level(book[pid]["trade_stability_score"])

    return {"stability_delta": stability_delta, "heat_delta": heat_delta}


def crisis_stage_from_remaining(initial_seconds: int, remaining_seconds: int) -> int:
    if remaining_seconds <= 0:
        return 4
    if initial_seconds <= 0:
        initial_seconds = CRISIS_DEADLINE_MAX
    ratio = remaining_seconds / float(initial_seconds)
    if ratio > 2.0 / 3.0:
        return 1
    if ratio > 1.0 / 3.0:
        return 2
    return 3


def crisis_trade_value_multiplier(crisis_stage: int) -> float:
    return {
        1: 0.93,
        2: 0.80,
        3: 0.52,
        4: 0.15,
    }.get(int(crisis_stage or 1), 0.93)


def crisis_distressed_asset_cost(base_value: float, crisis_stage: int) -> float:
    if crisis_stage < 4:
        return 0.0
    return max(8.0, base_value * 0.35)


def primary_complaint_from_pressures(pressures: Dict[str, float]) -> str:
    if not pressures:
        return "general dissatisfaction"
    top = max(pressures.items(), key=lambda kv: kv[1])
    labels = {
        "role": "deployment and ice time",
        "management": "management trust",
        "winning": "team competitiveness",
        "contract": "contract situation",
        "development": "development path",
        "belonging": "place in the locker room",
        "trade_exposure": "being shopped in trade talks",
        "broken_promise": "broken organizational promises",
        "performance": "role relative to production",
        "organizational": "organizational direction",
    }
    return labels.get(top[0], top[0].replace("_", " "))
