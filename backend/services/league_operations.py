"""
League Operations — league-wide revenue, CBA, cap forecast, relocation risk.
Backend-driven payload for the League Operations screen and Franchise Pulse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.franchise_session import FranchiseSession

# Tunable revenue bases (millions USD, annual)
_MARKET_BASE_REVENUE_M = {
    "large": 248.0,
    "medium": 168.0,
    "small": 118.0,
}

_MARKET_EXPENSE_RATIO = {
    "large": 0.78,
    "medium": 0.82,
    "small": 0.86,
}

_CBA_KEY_RULES = [
    "50/50 Revenue",
    "Hard Cap",
    "Escrow",
    "7-Year UFA",
    "8-Year Re-Sign",
    "Entry Deals",
    "No Tax Equalizer",
]

_RULE_CHANGE_TEMPLATES = [
    ("Cap Smoothing", "owners", 0.58),
    ("Escrow Cut", "players", 0.64),
    ("Play-In Round", "fans", 0.52),
    ("Lottery Reform", "fans", 0.48),
    ("Contract Limits", "owners", 0.55),
    ("NTC Rules", "players", 0.44),
    ("LTIR Audit", "owners", 0.61),
    ("Tax Adjustment", "owners", 0.38),
    ("Revenue Sharing", "small_markets", 0.72),
    ("Buyout Rules", "players", 0.41),
]

_CAP_GAUGE_POSITION = {
    "Rare Drop": 8,
    "Cap Freeze": 26,
    "Flat Cap": 48,
    "Small Rise": 72,
    "Big Jump": 92,
}

_CAP_TAGS = {
    "Big Jump": ["Escrow Hit", "Markets Strong", "Small Teams Okay"],
    "Small Rise": ["Escrow Hit", "Markets Strong", "Small Teams Okay"],
    "Flat Cap": ["Small Teams Weak", "Flat Risk"],
    "Cap Freeze": ["Escrow Miss", "Flat Risk"],
    "Rare Drop": ["Cap Drop Risk", "Fan Freeze"],
}


def _revenue_status_label(profit: float, revenue: float, win_pct: float, relocation_risk: float) -> str:
    if profit >= 18:
        return "Surge"
    if profit >= 8:
        return "Profit"
    if profit >= 2:
        return "Thin"
    if profit >= -2:
        return "Flat"
    if relocation_risk >= 0.55 or profit < -10:
        return "Risk"
    if profit < -2:
        return "Loss"
    return "Flat"


def _revenue_yoy_delta(team_id: str, season_year: int, revenue: float, win_pct: float) -> Dict[str, Any]:
    import hashlib

    h = int(hashlib.md5(f"{team_id}:{season_year - 1}".encode()).hexdigest()[:8], 16)
    drift = (h % 120) / 1000.0 - 0.04
    perf_shift = (win_pct - 0.5) * 0.14
    prior = revenue * (0.93 + drift - perf_shift * 0.35)
    prior = max(revenue * 0.78, prior)
    delta = round(revenue - prior, 1)
    if delta >= 4:
        direction = "up"
    elif delta <= -4:
        direction = "down"
    else:
        direction = "flat"
    return {"revenue_yoy_delta": delta, "revenue_yoy_direction": direction}


def _market_pressure_reason(team_row: Dict[str, Any], team: Any) -> str:
    att = _safe_float(team_row.get("attendance_rate", 0.7), 0.7)
    profit = _safe_float(team_row.get("profit", 0), 0)
    tier = str(team_row.get("market_tier", ""))
    arena = _team_arena_quality(team)
    fan = _safe_float(team_row.get("fan_sentiment", 55), 55)
    ownership = getattr(team, "ownership", None)
    patience = _safe_float(getattr(ownership, "patience", 0.55), 0.55) if ownership else 0.55

    if att < 0.52:
        return "Gate"
    if arena < 0.42:
        return "Arena"
    if profit < -4:
        return "Revenue"
    if tier == "Small":
        return "Small Market"
    if patience < 0.38:
        return "Owner"
    if fan < 42:
        return "Attendance"
    return "Lease"


def _relocation_reason_tag(team_row: Dict[str, Any], team: Any) -> str:
    pressure = _market_pressure_reason(team_row, team)
    mapping = {
        "Gate": "Gate",
        "Arena": "Arena",
        "Revenue": "Revenue",
        "Small Market": "Market",
        "Owner": "Owner",
        "Attendance": "Gate",
        "Lease": "Lease",
    }
    return mapping.get(pressure, "Stable")


def _build_cap_drivers(league_state: Dict[str, Any], cap: Dict[str, Any]) -> List[Dict[str, str]]:
    escrow = _safe_float(league_state.get("escrow_progress", 1.0), 1.0)
    health = _safe_float(league_state.get("revenue_health", 0.55), 0.55)
    small_be = _safe_float(league_state.get("small_market_break_even", 0.5), 0.5)
    losing = _safe_int(league_state.get("losing_teams_count", 0), 0)
    cba_p = _safe_float(league_state.get("cba_pressure", 0.3), 0.3)

    return [
        {"label": "Escrow", "sign": "+" if escrow >= 1.0 else "-"},
        {"label": "Revenue", "sign": "+" if health >= 0.55 else "-"},
        {"label": "Small Markets", "sign": "+" if small_be >= 0.65 else "-"},
        {"label": "Loss Teams", "sign": "+" if losing <= 10 else "-"},
        {"label": "CBA", "sign": "-" if cba_p >= 0.55 else "+"},
    ]


def _build_owner_mood(league_state: Dict[str, Any], cap: Dict[str, Any]) -> Dict[str, str]:
    pressure = _safe_float(league_state.get("cba_pressure", 0.3), 0.3)
    health = _safe_float(league_state.get("revenue_health", 0.55), 0.55)
    cap_type = str(cap.get("cap_change_type", ""))
    losing = _safe_int(league_state.get("losing_teams_count", 0), 0)

    if pressure >= 0.65:
        owners = "Hostile"
    elif pressure >= 0.45:
        owners = "Cautious"
    else:
        owners = "Calm"

    if "Drop" in cap_type or "Freeze" in cap_type:
        players = "Angry"
    elif "Jump" in cap_type:
        players = "Hungry"
    elif pressure >= 0.5:
        players = "Restless"
    else:
        players = "Stable"

    if health >= 0.65 and losing <= 8:
        fans = "Strong"
    elif health >= 0.5:
        fans = "Stable"
    elif health >= 0.4:
        fans = "Cool"
    else:
        fans = "Cold"

    if pressure >= 0.6 or losing >= 18:
        media = "Loud"
    elif pressure >= 0.4:
        media = "Watching"
    else:
        media = "Quiet"

    return {"owners": owners, "players": players, "fans": fans, "media": media}


def _build_league_pulse(league_state: Dict[str, Any], cap: Dict[str, Any], relocation: Dict[str, Any]) -> List[str]:
    pills: List[str] = []
    cap_type = str(cap.get("cap_change_type", ""))
    if "Jump" in cap_type or "Rise" in cap_type:
        pills.append("Cap Rising")
    elif "Drop" in cap_type:
        pills.append("Cap Falling")
    elif "Freeze" in cap_type or "Flat" in cap_type:
        pills.append("Cap Flat")

    pressure = _safe_float(league_state.get("cba_pressure", 0.3), 0.3)
    if pressure >= 0.55:
        pills.append("CBA Hot")
    else:
        pills.append("CBA Quiet")

    health = _safe_float(league_state.get("revenue_health", 0.55), 0.55)
    if health >= 0.6:
        pills.append("Revenue Strong")
    elif health >= 0.45:
        pills.append("Revenue Mixed")
    else:
        pills.append("Revenue Weak")

    losing = _safe_int(league_state.get("losing_teams_count", 0), 0)
    if losing >= 15:
        pills.append("Crisis Watch")
    else:
        pills.append("No Crisis")

    stability = str(relocation.get("league_stability", ""))
    if stability != "Stable":
        pills.append("Relocation Watch")
    else:
        pills.append("Markets Stable")

    return pills[:5]


def _league_health_label(revenue_health: float, losing_teams: int) -> str:
    if revenue_health >= 0.62 and losing_teams <= 6:
        return "Healthy League"
    if revenue_health >= 0.5 and losing_teams <= 12:
        return "Stable League"
    if revenue_health >= 0.4:
        return "Mixed League"
    return "Stressed League"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _team_market_tier(team: Any) -> Tuple[str, str]:
    market = getattr(team, "market", None)
    raw = str(getattr(market, "market_size", "") or getattr(team, "market_size", "") or "medium").lower()
    if raw in ("large", "big", "major"):
        return "large", "Large"
    if raw in ("small", "minor"):
        return "small", "Small"
    return "medium", "Mid"


def _team_player_ovr(player: Any) -> float:
    for key in ("overall", "ovr", "rating"):
        v = getattr(player, key, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    try:
        from app.sim_engine.engine import _franchise_player_ovr99  # noqa: WPS433

        return float(_franchise_player_ovr99(player))
    except Exception:
        return 75.0


def _player_display_name(player: Any) -> str:
    for key in ("name", "full_name", "player_name"):
        v = getattr(player, key, None)
        if v:
            return str(v)
    first = str(getattr(player, "first_name", "") or getattr(player, "firstName", "") or "")
    last = str(getattr(player, "last_name", "") or getattr(player, "lastName", "") or "")
    combined = f"{first} {last}".strip()
    return combined or "Unknown"


def _team_star_metrics(team: Any) -> Dict[str, Any]:
    roster = list(getattr(team, "roster", None) or [])
    elite_95: List[float] = []
    strong_92: List[float] = []
    mild_88: List[float] = []
    top_ovr = 0.0
    top_name = ""
    for p in roster:
        if getattr(p, "retired", False):
            continue
        ovr = _team_player_ovr(p)
        if ovr > top_ovr:
            top_ovr = ovr
            top_name = _player_display_name(p)
        if ovr >= 95:
            elite_95.append(ovr)
        elif ovr >= 92:
            strong_92.append(ovr)
        elif ovr >= 88:
            mild_88.append(ovr)
    star_power = round(
        len(elite_95) * 1.0
        + len(strong_92) * 0.55
        + len(mild_88) * 0.22
        + max(0.0, (top_ovr - 88) * 0.04),
        2,
    )
    return {
        "elite_95": elite_95,
        "strong_92": strong_92,
        "mild_88": mild_88,
        "superstar_count": len(elite_95),
        "stars_95": len(elite_95),
        "stars_90": len(strong_92) + len(elite_95),
        "star_power": star_power,
        "top_ovr": top_ovr,
        "top_player_overall": round(top_ovr, 1),
        "top_player_name": top_name,
    }


def _diminishing_stack(values: List[float], base_per: float, decay: Tuple[float, ...]) -> float:
    total = 0.0
    for i, ovr in enumerate(sorted(values, reverse=True)):
        tier_mult = decay[i] if i < len(decay) else decay[-1]
        ovr_bonus = max(0.0, (ovr - 88) * 0.08)
        total += (base_per + ovr_bonus) * tier_mult
    return total


def _fan_star_multiplier(fan_sentiment: float) -> float:
    if fan_sentiment >= 68:
        return 1.18
    if fan_sentiment >= 55:
        return 1.0
    if fan_sentiment >= 42:
        return 0.72
    if fan_sentiment >= 35:
        return 0.48
    return 0.32


def _market_star_multiplier(tier_key: str) -> float:
    return {"small": 1.48, "medium": 1.0, "large": 0.68}.get(tier_key, 1.0)


def calculate_superstar_revenue_boost(
    stars: Dict[str, Any],
    tier_key: str,
    fan_sentiment: float,
    win_pct: float,
    *,
    in_playoffs: bool = False,
) -> Dict[str, Any]:
    """Jersey / gate / TV / sponsor / fan / marketability / playoff hype — backend only."""
    elite = list(stars.get("elite_95") or [])
    strong = list(stars.get("strong_92") or [])
    mild = list(stars.get("mild_88") or [])

    raw = 0.0
    raw += _diminishing_stack(elite, 22.0, (1.0, 0.62, 0.38, 0.25))
    raw += _diminishing_stack(strong, 10.5, (1.0, 0.58, 0.35))
    raw += _diminishing_stack(mild, 4.2, (1.0, 0.5, 0.3))

    if raw <= 0:
        return {
            "superstar_revenue_boost": 0.0,
            "superstar_tags": [],
            "superstar_channels": {},
        }

    market_mult = _market_star_multiplier(tier_key)
    fan_mult = _fan_star_multiplier(fan_sentiment)
    perf_mult = 0.92 + win_pct * 0.16
    playoff_mult = 1.12 if in_playoffs else 1.0

    jersey = raw * 0.28 * market_mult * fan_mult
    tickets = raw * 0.24 * market_mult * (0.85 + win_pct * 0.3) * fan_mult
    national_tv = raw * 0.16 * (1.1 if stars.get("superstar_count", 0) >= 1 else 0.6)
    sponsors = raw * 0.18 * market_mult * fan_mult * perf_mult
    fan_interest = raw * 0.08 * fan_mult
    marketability = raw * 0.04 * market_mult
    playoff_hype = raw * 0.02 * playoff_mult * (1.0 + win_pct * 0.5)

    total = round(
        (jersey + tickets + national_tv + sponsors + fan_interest + marketability + playoff_hype)
        * perf_mult,
        1,
    )

    tags: List[str] = []
    if stars.get("superstar_count", 0) >= 2:
        tags.append("Superstar")
    elif stars.get("superstar_count", 0) >= 1 or stars.get("top_ovr", 0) >= 95:
        tags.append("Star Boost")
    if jersey >= tickets and jersey >= sponsors:
        tags.append("Jersey Spike")
    elif national_tv >= tickets:
        tags.append("TV Draw")
    elif tickets >= sponsors:
        tags.append("Star Boost")
    else:
        tags.append("Jersey Spike")

    # Dedupe while preserving order, max 2 for UI
    seen: set = set()
    unique_tags: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return {
        "superstar_revenue_boost": total,
        "superstar_tags": unique_tags[:2],
        "superstar_channels": {
            "jersey_sales": round(jersey, 1),
            "ticket_demand": round(tickets, 1),
            "national_games": round(national_tv, 1),
            "sponsorships": round(sponsors, 1),
            "fan_interest": round(fan_interest, 1),
            "marketability": round(marketability, 1),
            "playoff_hype": round(playoff_hype, 1),
        },
    }


def _team_win_pct(session: FranchiseSession, team_id: str) -> float:
    rec = None
    if session.standings:
        rec = session.standings.records.get(team_id)
    if rec is not None:
        gp = max(1, int(getattr(rec, "gp", 0) or 0))
        w = int(getattr(rec, "wins", 0) or getattr(rec, "w", 0) or 0)
        return _clamp(w / gp, 0.0, 1.0)
    return 0.5


def _team_fan_sentiment(session: FranchiseSession, team_id: str) -> float:
    try:
        from services.franchise_sim import _ensure_team_fan_profile  # noqa: WPS433

        profile = _ensure_team_fan_profile(session, team_id)
        return _clamp(_safe_float(profile.get("fan_confidence", 55.0), 55.0), 0.0, 100.0)
    except Exception:
        return 55.0


def _team_trade_heat(session: FranchiseSession, team_id: str) -> float:
    try:
        from services.franchise_sim import _ensure_team_fan_profile  # noqa: WPS433

        profile = _ensure_team_fan_profile(session, team_id)
        return _clamp(_safe_float(profile.get("recent_trade_heat", 0.0), 0.0), 0.0, 100.0)
    except Exception:
        return 0.0


def _team_arena_quality(team: Any) -> float:
    state = getattr(team, "state", None)
    if state is not None:
        fh = getattr(state, "financial_health", None)
        if fh is not None:
            return _clamp(_safe_float(fh, 0.5), 0.2, 1.0)
    rep = getattr(team, "reputation", None)
    if rep is not None:
        lr = getattr(rep, "league_reputation", 0.5)
        return _clamp(0.45 + _safe_float(lr, 0.5) * 0.35, 0.35, 0.95)
    return 0.55


def _playoff_revenue_bonus(session: FranchiseSession, team_id: str) -> float:
    uid = str(session.user_team_id or "")
    phase = str(getattr(session, "phase", "") or "").lower()
    bonus = 0.0
    if phase in ("playoffs", "postseason", "post_cup"):
        bonus += 6.0
    champ = str(getattr(session, "champion_id", "") or getattr(session, "stanley_cup_winner", "") or "")
    if champ and champ == team_id:
        bonus += 22.0
    elif phase == "post_cup" and team_id == uid:
        bonus += 4.0
    return bonus


def calculate_team_revenue(
    session: FranchiseSession,
    team: Any,
    team_id: str,
    *,
    is_user: bool = False,
) -> Dict[str, Any]:
    from services.franchise_sim import _display_team, _franchise_team_abbrev  # noqa: WPS433

    tier_key, tier_label = _team_market_tier(team)
    base = _MARKET_BASE_REVENUE_M[tier_key]
    win_pct = _team_win_pct(session, team_id)
    fan_sent = _team_fan_sentiment(session, team_id)
    trade_heat = _team_trade_heat(session, team_id)
    stars = _team_star_metrics(team)
    arena = _team_arena_quality(team)

    perf_mult = 0.82 + win_pct * 0.38
    fan_mult = 0.70 + (fan_sent / 100.0) * 0.45
    arena_mult = 0.92 + arena * 0.14

    phase = str(getattr(session, "phase", "") or "").lower()
    in_playoffs = phase in ("playoffs", "postseason", "post_cup")
    star_boost = calculate_superstar_revenue_boost(
        stars,
        tier_key,
        fan_sent,
        win_pct,
        in_playoffs=in_playoffs,
    )
    superstar_m = _safe_float(star_boost.get("superstar_revenue_boost", 0), 0.0)

    revenue = base * perf_mult * fan_mult * arena_mult
    revenue += superstar_m
    revenue += _playoff_revenue_bonus(session, team_id)

    if is_user and trade_heat >= 55:
        revenue *= 1.0 - _clamp((trade_heat - 50) / 120.0, 0.0, 0.22)
    if is_user and fan_sent < 35:
        revenue *= 0.88
    elif is_user and fan_sent < 45:
        revenue *= 0.94

    expense_ratio = _MARKET_EXPENSE_RATIO[tier_key]
    payroll_m = _safe_float(getattr(team, "payroll_m", 0) or 0)
    if payroll_m <= 0:
        try:
            from services.franchise_sim import _team_nhl_payroll_m  # noqa: WPS433

            payroll_m = _team_nhl_payroll_m(team)
        except Exception:
            payroll_m = base * 0.55
    expense_ratio += _clamp(payroll_m / max(base, 1.0) - 0.65, 0.0, 0.12) * 0.15

    expenses = revenue * expense_ratio + payroll_m * 0.08
    profit = revenue - expenses
    attendance_rate = _clamp(0.55 + win_pct * 0.25 + (fan_sent / 100.0) * 0.22 + stars["star_power"] * 0.03, 0.35, 0.99)

    superstar_tags = list(star_boost.get("superstar_tags") or [])
    reason_tags: List[str] = list(superstar_tags)
    if tier_key == "large":
        reason_tags.append("Big Market")
    if win_pct >= 0.58:
        reason_tags.append("Playoff Cash")
    if win_pct < 0.42:
        reason_tags.append("Weak Gate")
    if fan_sent < 40:
        reason_tags.append("Fan Freeze")
    if is_user and trade_heat >= 60:
        reason_tags.append("Boycott Risk")
    if profit < 0:
        reason_tags.append("Small Loss")
    elif profit > base * 0.15:
        reason_tags.append("Merch Spike")
    if arena < 0.45:
        reason_tags.append("Arena Drag")

    # Max 2 chips — superstar tags first, then strongest other signals
    deduped: List[str] = []
    seen_tags: set = set()
    for tag in reason_tags:
        if tag not in seen_tags:
            seen_tags.add(tag)
            deduped.append(tag)
    reason_tags = deduped[:2]

    relocation_risk = calculate_relocation_risk(team, revenue, profit, fan_sent, win_pct, tier_key)
    revenue_status = _revenue_status_label(profit, revenue, win_pct, relocation_risk)
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    yoy = _revenue_yoy_delta(team_id, sy, revenue, win_pct)

    abbr = _franchise_team_abbrev(team)
    name = _display_team(team)

    row: Dict[str, Any] = {
        "id": team_id,
        "name": name,
        "abbreviation": abbr,
        "logo": abbr,
        "market_size": tier_key,
        "market_tier": tier_label,
        "revenue": round(revenue, 1),
        "expenses": round(expenses, 1),
        "profit": round(profit, 1),
        "attendance_rate": round(attendance_rate, 3),
        "fan_sentiment": round(fan_sent, 1),
        "star_power": stars["star_power"],
        "superstar_count": stars["superstar_count"],
        "top_player_overall": stars["top_player_overall"],
        "top_player_name": stars["top_player_name"],
        "superstar_revenue_boost": superstar_m,
        "superstar_tags": superstar_tags,
        "playoff_revenue": round(_playoff_revenue_bonus(session, team_id), 1),
        "relocation_risk": round(relocation_risk, 3),
        "relocation_risk_label": _relocation_label(relocation_risk),
        "revenue_status": revenue_status,
        "reason_tags": reason_tags,
        "trade_heat": round(trade_heat, 1),
        "revenue_yoy_delta": yoy["revenue_yoy_delta"],
        "revenue_yoy_direction": yoy["revenue_yoy_direction"],
    }
    row["market_pressure"] = _market_pressure_reason(row, team)
    return row


def calculate_relocation_risk(
    team: Any,
    revenue: float,
    profit: float,
    fan_sentiment: float,
    win_pct: float,
    market_tier: str,
) -> float:
    risk = 0.12
    if market_tier == "small":
        risk += 0.18
    elif market_tier == "medium":
        risk += 0.08

    if profit < -8:
        risk += 0.22
    elif profit < 0:
        risk += 0.10

    if fan_sentiment < 40:
        risk += 0.12
    elif fan_sentiment < 50:
        risk += 0.05

    if win_pct < 0.38:
        risk += 0.10

    ownership = getattr(team, "ownership", None)
    if ownership is not None:
        patience = _safe_float(getattr(ownership, "patience", 0.55), 0.55)
        risk += (0.65 - patience) * 0.15

    arena = _team_arena_quality(team)
    if arena < 0.42:
        risk += 0.14

    return _clamp(risk, 0.05, 0.95)


def _relocation_label(risk: float) -> str:
    if risk >= 0.62:
        return "High"
    if risk >= 0.38:
        return "Med"
    return "Low"


def calculate_league_revenue(teams: List[Dict[str, Any]]) -> float:
    return round(sum(_safe_float(t.get("revenue", 0)) for t in teams), 1)


def calculate_escrow_progress(league_revenue: float, league_health: float) -> Dict[str, float]:
    target = round(league_revenue * 0.115, 1)
    collected = round(target * _clamp(0.82 + league_health * 0.28, 0.65, 1.12), 1)
    progress = collected / max(target, 1.0)
    return {
        "escrow_target": target,
        "escrow_collected": collected,
        "escrow_progress": round(progress, 3),
    }


def calculate_salary_cap_projection(
    session: FranchiseSession,
    league_state: Dict[str, Any],
) -> Dict[str, Any]:
    from services.franchise_sim import _resolve_league_salary_cap_m  # noqa: WPS433

    sim = session.sim
    league = getattr(sim, "league", None)
    current_cap = _resolve_league_salary_cap_m(league)

    escrow_progress = _safe_float(league_state.get("escrow_progress", 1.0), 1.0)
    losing_teams = _safe_int(league_state.get("losing_teams_count", 0), 0)
    small_market_break_even = _safe_float(league_state.get("small_market_break_even", 0.5), 0.5)
    revenue_health = _safe_float(league_state.get("revenue_health", 0.55), 0.55)
    cba_pressure = _safe_float(league_state.get("cba_pressure", 0.3), 0.3)

    if escrow_progress >= 1.05 and small_market_break_even >= 0.75:
        cap_change_type = "Big Jump"
        delta_pct = 0.068
    elif escrow_progress >= 1.0 and losing_teams <= 8:
        cap_change_type = "Small Rise"
        delta_pct = 0.032
    elif losing_teams >= 22 or revenue_health < 0.45:
        cap_change_type = "Rare Drop"
        delta_pct = -0.018
    elif losing_teams >= 15:
        cap_change_type = "Flat Cap"
        delta_pct = 0.004
    elif escrow_progress < 0.92:
        cap_change_type = "Cap Freeze"
        delta_pct = 0.0
    else:
        cap_change_type = "Small Rise"
        delta_pct = 0.022

    if cba_pressure >= 0.72:
        delta_pct *= 0.65

    projected = round(current_cap * (1.0 + delta_pct), 2)
    cap_change = round(projected - current_cap, 2)

    return {
        "salary_cap": round(current_cap, 2),
        "projected_salary_cap": projected,
        "cap_change": cap_change,
        "cap_change_type": cap_change_type,
        "cap_tags": _CAP_TAGS.get(cap_change_type, ["Flat Risk"]),
        "cap_gauge_position": _CAP_GAUGE_POSITION.get(cap_change_type, 48),
    }


def calculate_cba_pressure(league_state: Dict[str, Any]) -> float:
    losing = _safe_int(league_state.get("losing_teams_count", 0), 0)
    health = _safe_float(league_state.get("revenue_health", 0.55), 0.55)
    escrow = _safe_float(league_state.get("escrow_progress", 1.0), 1.0)
    pressure = 0.25 + (losing / 32.0) * 0.35 + (1.0 - health) * 0.25
    if escrow < 0.95:
        pressure += 0.08
    return round(_clamp(pressure, 0.1, 0.95), 3)


def _build_cba_block(session: FranchiseSession, pressure: float) -> Dict[str, Any]:
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    start_year = sy - ((sy - 2020) % 8)
    end_year = start_year + 7
    years_remaining = max(0, end_year - sy)
    bargaining_deadline = f"{end_year - 1}-06-30"

    if pressure >= 0.72:
        pressure_level = "High"
    elif pressure >= 0.48:
        pressure_level = "Med"
    else:
        pressure_level = "Low"

    return {
        "current_agreement": f"CBA {start_year}-{end_year}",
        "start_year": start_year,
        "end_year": end_year,
        "bargaining_deadline": bargaining_deadline,
        "years_remaining": years_remaining,
        "pressure_level": pressure_level,
        "pressure": round(pressure, 3),
        "key_rules": list(_CBA_KEY_RULES),
        "potential_changes": _build_rule_changes(pressure, session),
    }


def _build_rule_changes(pressure: float, session: FranchiseSession) -> List[Dict[str, Any]]:
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    out: List[Dict[str, Any]] = []
    for name, faction, base_support in _RULE_CHANGE_TEMPLATES:
        support = base_support + (pressure - 0.5) * 0.12
        if faction == "small_markets":
            support += 0.06
        support = _clamp(support, 0.18, 0.88)
        owner_support = _clamp(support + (0.08 if "Cap" in name or "Tax" in name else -0.04), 0.1, 0.95)
        player_support = _clamp(1.0 - owner_support + 0.15, 0.1, 0.95)
        fan_reaction = _clamp(0.42 + support * 0.35, 0.2, 0.9)
        likelihood = support * 0.85 + (0.1 if sy % 3 == 0 else 0.0)
        if likelihood >= 0.62:
            status = "Likely"
        elif likelihood >= 0.48:
            status = "Open"
        else:
            status = "Long"
        out.append(
            {
                "name": name,
                "support_pct": round(support * 100),
                "owner_support": round(owner_support * 100),
                "player_support": round(player_support * 100),
                "fan_reaction": round(fan_reaction * 100),
                "likelihood": round(likelihood * 100),
                "status": status,
                "label": f"{name} | {round(support * 100)}% | {status}",
            }
        )
    return out[:8]


def _fan_sentiment_label(score: float) -> str:
    if score >= 68:
        return "Hot"
    if score >= 55:
        return "Warm"
    if score >= 42:
        return "Cool"
    return "Cold"


def _build_user_impact(user_row: Dict[str, Any], cap_change: float, league_revenue: float) -> Dict[str, Any]:
    fan_sent = _safe_float(user_row.get("fan_sentiment", 55), 55)
    trade_heat = _safe_float(user_row.get("trade_heat", 0), 0)
    boycott_risk = "High" if fan_sent < 38 or trade_heat >= 65 else ("Med" if fan_sent < 48 or trade_heat >= 45 else "Low")
    cap_share = (_safe_float(user_row.get("revenue", 0)) / max(league_revenue, 1.0)) * cap_change
    backlash = "Heavy" if trade_heat >= 60 else ("Light" if trade_heat >= 35 else "None")

    superstar_m = _safe_float(user_row.get("superstar_revenue_boost", 0), 0)
    superstar_tags = list(user_row.get("superstar_tags") or [])
    impact_tags = list(superstar_tags) if superstar_tags else list(user_row.get("reason_tags") or [])
    superstar_impact_label = ""
    if superstar_m >= 1.0:
        if user_row.get("superstar_count", 0) >= 2:
            superstar_impact_label = "Superstar Boost"
        else:
            superstar_impact_label = f"Star Pull +${round(superstar_m):.0f}M"

    return {
        "revenue": user_row.get("revenue", 0),
        "fan_sentiment": fan_sent,
        "fan_label": _fan_sentiment_label(fan_sent),
        "boycott_risk": boycott_risk,
        "trade_backlash": backlash,
        "cap_contribution_m": round(cap_share, 2),
        "market_health": user_row.get("revenue_status", "Even"),
        "reason_tags": impact_tags[:2],
        "superstar_revenue_boost": superstar_m,
        "superstar_impact_label": superstar_impact_label,
        "top_player_name": user_row.get("top_player_name", ""),
        "top_player_overall": user_row.get("top_player_overall", 0),
    }


def build_franchise_pulse(session: FranchiseSession, ops: Dict[str, Any]) -> Dict[str, Any]:
    user = dict(ops.get("user_team") or {})
    cap = dict(ops.get("cap") or {})
    return {
        "revenue_m": user.get("revenue", 0),
        "revenue_label": f"${_safe_float(user.get('revenue', 0)):.0f}M",
        "fan_spending": user.get("fan_label") or _fan_sentiment_label(_safe_float(user.get("fan_sentiment", 55))),
        "fan_sentiment": user.get("fan_sentiment", 55),
        "fan_label": user.get("fan_label") or _fan_sentiment_label(_safe_float(user.get("fan_sentiment", 55))),
        "boycott_risk": user.get("boycott_risk", "Low"),
        "cap_contribution_m": user.get("cap_contribution_m", 0),
        "cap_pull_label": f"{user.get('cap_contribution_m', 0):+.1f}M",
        "market_health": user.get("market_health", "Even"),
        "cap_change_type": cap.get("cap_change_type", "Flat Cap"),
    }


def build_league_operations_payload(session: FranchiseSession) -> Dict[str, Any]:
    uid = str(session.user_team_id or "")
    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    season_label = f"{sy}–{sy + 1}"

    team_rows: List[Dict[str, Any]] = []
    for tid, team in (session.team_by_id or {}).items():
        if team is None:
            continue
        row = calculate_team_revenue(session, team, str(tid), is_user=(str(tid) == uid))
        team_rows.append(row)

    team_rows.sort(key=lambda r: -_safe_float(r.get("revenue", 0)))

    max_rev = max((_safe_float(t.get("revenue", 0)) for t in team_rows), default=1.0)
    max_profit = max((abs(_safe_float(t.get("profit", 0))) for t in team_rows), default=1.0)
    for i, row in enumerate(team_rows):
        row["rank"] = i + 1
        row["revenue_bar_pct"] = round(_safe_float(row.get("revenue", 0)) / max(max_rev, 1.0) * 100, 1)
        prof = _safe_float(row.get("profit", 0))
        row["profit_bar_pct"] = round(abs(prof) / max(max_profit, 1.0) * 100, 1)
        row["profit_positive"] = prof >= 0
        row["fan_heat_pct"] = round(_safe_float(row.get("fan_sentiment", 55)), 1)
        row["relocation_bar_pct"] = round(_safe_float(row.get("relocation_risk", 0)) * 100, 1)

    league_revenue = calculate_league_revenue(team_rows)
    total_superstar_boost = round(sum(_safe_float(t.get("superstar_revenue_boost", 0)) for t in team_rows), 1)
    large_rev = sum(_safe_float(t["revenue"]) for t in team_rows if t.get("market_tier") == "Large")
    small_teams = [t for t in team_rows if t.get("market_tier") == "Small"]
    small_profitable = sum(1 for t in small_teams if _safe_float(t.get("profit", 0)) >= 0)
    small_market_break_even = small_profitable / max(len(small_teams), 1)
    losing_teams = sum(1 for t in team_rows if _safe_float(t.get("profit", 0)) < -2)
    avg_fan = sum(_safe_float(t.get("fan_sentiment", 55)) for t in team_rows) / max(len(team_rows), 1)
    revenue_health = _clamp(
        0.35 + (league_revenue / max(len(team_rows) * 175.0, 1)) * 0.35 + (avg_fan / 100.0) * 0.2 - (losing_teams / 32.0) * 0.25,
        0.15,
        0.95,
    )
    superstar_share = total_superstar_boost / max(league_revenue, 1.0)
    if superstar_share >= 0.035:
        revenue_health = _clamp(revenue_health + min(0.06, superstar_share * 0.45), 0.15, 0.95)

    league_state = {
        "league_revenue": league_revenue,
        "total_superstar_revenue_boost": total_superstar_boost,
        "superstar_revenue_share": round(superstar_share, 4),
        "losing_teams_count": losing_teams,
        "small_market_break_even": round(small_market_break_even, 3),
        "revenue_health": round(revenue_health, 3),
    }
    escrow = calculate_escrow_progress(league_revenue, revenue_health)
    league_state.update(escrow)
    league_state["cba_pressure"] = calculate_cba_pressure(league_state)

    cap = calculate_salary_cap_projection(session, league_state)
    cba = _build_cba_block(session, league_state["cba_pressure"])

    watchlist = sorted(team_rows, key=lambda r: -_safe_float(r.get("relocation_risk", 0)))[:5]
    relocation = {
        "watchlist": [
            {
                "id": t["id"],
                "abbreviation": t["abbreviation"],
                "risk": t.get("relocation_risk_label", "Low"),
                "risk_score": t.get("relocation_risk", 0),
                "risk_bar_pct": t.get("relocation_bar_pct", 0),
                "reason": _relocation_reason_tag(t, session.team_by_id.get(t["id"])),
                "pressure": t.get("market_pressure", "Stable"),
                "label": f"{t['abbreviation']} | {t.get('relocation_risk_label', 'Low')} | {t.get('market_pressure', 'Stable')}",
            }
            for t in watchlist
        ],
        "highest_risk_team": watchlist[0]["abbreviation"] if watchlist else "",
        "league_stability": "Stable" if revenue_health >= 0.55 and losing_teams <= 12 else ("Shaky" if losing_teams <= 20 else "Weak"),
    }

    cap_drivers = _build_cap_drivers(league_state, cap)
    owner_mood = _build_owner_mood(league_state, cap)
    league_pulse = _build_league_pulse(league_state, cap, relocation)
    health_label = _league_health_label(revenue_health, losing_teams)

    user_row = next((t for t in team_rows if str(t.get("id")) == uid), team_rows[0] if team_rows else {})
    user_impact = _build_user_impact(user_row, _safe_float(cap.get("cap_change", 0)), league_revenue)
    user_team = {**user_row, **user_impact}

    rule_changes = cba.get("potential_changes") or []

    return {
        "season": season_label,
        "current_year": sy,
        "salary_cap": cap["salary_cap"],
        "projected_salary_cap": cap["projected_salary_cap"],
        "cap_change": cap["cap_change"],
        "cap_change_type": cap["cap_change_type"],
        "cap_tags": cap.get("cap_tags", []),
        "league_revenue": league_revenue,
        "escrow_target": escrow["escrow_target"],
        "escrow_collected": escrow["escrow_collected"],
        "escrow_progress": escrow["escrow_progress"],
        "large_market_revenue": round(large_rev, 1),
        "small_market_break_even": round(small_market_break_even, 3),
        "small_market_gap": round(max(0.0, 0.75 - small_market_break_even), 3),
        "losing_teams_count": losing_teams,
        "revenue_health": round(revenue_health, 3),
        "cba": cba,
        "relocation": relocation,
        "teams": team_rows,
        "user_team": user_team,
        "rule_changes": rule_changes,
        "cap": cap,
        "cap_drivers": cap_drivers,
        "cap_gauge_position": cap.get("cap_gauge_position", 48),
        "owner_mood": owner_mood,
        "league_pulse": league_pulse,
        "league_health_label": health_label,
        "league_revenue_b": round(league_revenue / 1000.0, 2),
        "total_superstar_revenue_boost": total_superstar_boost,
    }
