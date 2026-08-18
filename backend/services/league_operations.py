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


def _revenue_yoy_delta(
    session: FranchiseSession,
    team_id: str,
    season_year: int,
    revenue: float,
    win_pct: float,
) -> Dict[str, Any]:
    """Prefer persisted prior-season revenue; seed history when missing."""
    history = getattr(session, "market_revenue_history", None)
    if not isinstance(history, dict):
        history = {}
        try:
            session.market_revenue_history = history
        except Exception:
            pass

    tid = str(team_id)
    team_hist = history.get(tid)
    if not isinstance(team_hist, dict):
        team_hist = {}
        history[tid] = team_hist

    prior_key = str(int(season_year) - 1)
    cur_key = str(int(season_year))
    prior = team_hist.get(prior_key)
    if prior is None:
        # First observation: invent a soft prior once, then persist both years so
        # subsequent reads are stable (no MD5 churn every request).
        import hashlib

        h = int(hashlib.md5(f"{tid}:{prior_key}".encode()).hexdigest()[:8], 16)
        drift = (h % 120) / 1000.0 - 0.04
        perf_shift = (win_pct - 0.5) * 0.14
        prior = revenue * (0.93 + drift - perf_shift * 0.35)
        prior = max(revenue * 0.78, prior)
        team_hist[prior_key] = round(float(prior), 1)
    else:
        prior = float(prior)

    team_hist[cur_key] = round(float(revenue), 1)
    delta = round(float(revenue) - float(prior), 1)
    if delta >= 4:
        direction = "up"
    elif delta <= -4:
        direction = "down"
    else:
        direction = "flat"
    return {
        "revenue_yoy_delta": delta,
        "revenue_yoy_direction": direction,
        "revenue_prior_m": round(float(prior), 1),
    }


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


# Nations whose diaspora / national pride creates outsized NHL market interest
# ("Yao Ming effect") when a player from that country joins a club.
_GLOBAL_DRAW_NATIONS = frozenset(
    {
        "china",
        "chinese",
        "chn",
        "japan",
        "japanese",
        "jpn",
        "south korea",
        "korea",
        "republic of korea",
        "korean",
        "kor",
        "india",
        "indian",
        "ind",
        "nigeria",
        "nigerian",
        "nga",
        "kenya",
        "kenyan",
        "ken",
        "philippines",
        "filipino",
        "phl",
        "mexico",
        "mexican",
        "mex",
        "brazil",
        "brazilian",
        "bra",
        "argentina",
        "argentinian",
        "arg",
        "south africa",
        "zaf",
        "rsa",
        "ghana",
        "gha",
        "jamaica",
        "jam",
        "vietnam",
        "vnm",
        "indonesia",
        "idn",
        "pakistan",
        "pak",
        "egypt",
        "egy",
        "morocco",
        "mar",
        "colombia",
        "col",
        "peru",
        "per",
        "chile",
        "chl",
        "thailand",
        "tha",
        "taiwan",
        "twn",
        "hong kong",
        "hkg",
    }
)


def _player_birth_country(player: Any) -> str:
    ident = getattr(player, "identity", None)
    for src in (
        getattr(ident, "birth_country", None) if ident is not None else None,
        getattr(ident, "nationality", None) if ident is not None else None,
        getattr(player, "nationality", None),
        getattr(player, "birth_country", None),
        getattr(player, "birthCountry", None),
        getattr(player, "country", None),
    ):
        if src:
            return str(src).strip()
    return ""


def _is_global_draw_nation(country: str) -> bool:
    raw = str(country or "").strip().lower()
    if not raw:
        return False
    if raw in _GLOBAL_DRAW_NATIONS:
        return True
    # Match compound labels ("China / Hong Kong", "South Korea")
    return any(token in raw for token in ("china", "japan", "korea", "nigeria", "india", "philippines", "mexico", "brazil"))


def calculate_global_draw_revenue_boost(team: Any) -> Dict[str, Any]:
    """Yao Ming effect — rare-nation pride markets explode jersey / TV / gate demand.

    Even a mid-tier NHL regular from China, Japan, Korea, Nigeria, etc. moves the
    needle; stars create franchise-altering revenue spikes.
    """
    draws: List[Dict[str, Any]] = []
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        country = _player_birth_country(p)
        if not _is_global_draw_nation(country):
            continue
        ovr = _team_player_ovr(p)
        if ovr < 68.0:
            continue
        # Base: growing hockey markets pay up once a countryman reaches the NHL.
        raw = max(0.0, (ovr - 66.0) * 1.65)
        if ovr >= 88:
            raw *= 1.55
        elif ovr >= 82:
            raw *= 1.28
        elif ovr >= 76:
            raw *= 1.12
        draws.append(
            {
                "name": _player_display_name(p),
                "country": country,
                "ovr": round(ovr, 1),
                "raw": raw,
            }
        )

    if not draws:
        return {
            "global_draw_revenue_boost": 0.0,
            "global_draw_players": [],
            "global_draw_tags": [],
            "global_draw_channels": {},
        }

    draws.sort(key=lambda d: d["raw"], reverse=True)
    decay = (1.0, 0.55, 0.32, 0.2)
    total_raw = 0.0
    for i, row in enumerate(draws):
        total_raw += row["raw"] * (decay[i] if i < len(decay) else decay[-1])

    # Channel mix leans jersey + international TV — classic Yao profile.
    jersey = total_raw * 0.38
    national_tv = total_raw * 0.28
    sponsors = total_raw * 0.18
    tickets = total_raw * 0.10
    marketability = total_raw * 0.06
    total = round(jersey + national_tv + sponsors + tickets + marketability, 1)

    tags = ["Global Draw"]
    if draws[0]["ovr"] >= 88 or total >= 28.0:
        tags.append("Yao Effect")

    return {
        "global_draw_revenue_boost": total,
        "global_draw_players": [
            {"name": d["name"], "country": d["country"], "ovr": d["ovr"]} for d in draws[:4]
        ],
        "global_draw_tags": tags[:2],
        "global_draw_channels": {
            "jersey_sales": round(jersey, 1),
            "national_games": round(national_tv, 1),
            "sponsorships": round(sponsors, 1),
            "ticket_demand": round(tickets, 1),
            "marketability": round(marketability, 1),
        },
    }


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
        gp = int(getattr(rec, "gp", 0) or 0)
        # Neutral sample when the season hasn't started — using max(1, gp) made
        # 0-0 clubs read as 0% and 3-0 clubs as 100%, which inflated camp revenue.
        if gp <= 0:
            return 0.5
        w = int(getattr(rec, "wins", 0) or getattr(rec, "w", 0) or 0)
        return _clamp(w / gp, 0.0, 1.0)
    return 0.5


def _season_ticket_summer_revenue_m(tier_key: str) -> float:
    """Camp / summer baseline from season tickets — ~$80–110M before the gate opens."""
    return {
        "large": 108.0,
        "medium": 94.0,
        "small": 82.0,
    }.get(tier_key, 94.0)


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

    phase = str(getattr(session, "phase", "") or "").lower()
    global_draw = calculate_global_draw_revenue_boost(team)
    global_draw_m = _safe_float(global_draw.get("global_draw_revenue_boost", 0), 0.0)

    # Camp / summer: season-ticket book, not the full in-season gate.
    if phase in ("preseason", "offseason"):
        summer = _season_ticket_summer_revenue_m(tier_key)
        fan_adj = 0.92 + (fan_sent / 100.0) * 0.14
        revenue = summer * fan_adj
        revenue += stars.get("star_power", 0) * 1.5
        # International interest still lifts summer merch / deposits.
        revenue += global_draw_m * 0.62
        if is_user and fan_sent < 40:
            revenue *= 0.94
        expense_ratio = _MARKET_EXPENSE_RATIO[tier_key]
        payroll_m = _safe_float(getattr(team, "payroll_m", 0) or 0)
        if payroll_m <= 0:
            try:
                from services.franchise_sim import _team_nhl_payroll_m  # noqa: WPS433

                payroll_m = _team_nhl_payroll_m(team)
            except Exception:
                payroll_m = summer * 0.55
        expenses = revenue * expense_ratio + payroll_m * 0.06
        profit = revenue - expenses
        summer_tags = ["Season Tickets", "Summer Books"]
        for tag in list(global_draw.get("global_draw_tags") or []):
            if tag not in summer_tags:
                summer_tags.insert(0, tag)
        relocation_risk = calculate_relocation_risk(team, revenue, profit, fan_sent, win_pct, tier_key)
        revenue_status = _revenue_status_label(profit, revenue, win_pct, relocation_risk)
        sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
        yoy = _revenue_yoy_delta(session, team_id, sy, revenue, win_pct)
        abbr = _franchise_team_abbrev(team) if team is not None else ""
        name = _display_team(team) if team is not None else team_id
        row = {
            "id": str(team_id),
            "team_id": str(team_id),
            "name": name,
            "team_name": name,
            "abbreviation": abbr,
            "abbr": abbr,
            "logo": abbr,
            "market_size": tier_key,
            "market_tier": tier_label,
            "market_tier_key": tier_key,
            "revenue": round(revenue, 1),
            "revenue_m": round(revenue, 1),
            "expenses": round(expenses, 1),
            "profit": round(profit, 1),
            "win_pct": round(win_pct, 3),
            "fan_sentiment": round(fan_sent, 1),
            "trade_heat": round(trade_heat, 1),
            "arena_quality": round(arena, 3),
            "attendance_rate": round(
                _clamp(0.70 + (fan_sent / 100.0) * 0.18 + min(0.08, global_draw_m * 0.002), 0.55, 0.98),
                3,
            ),
            "star_power": stars.get("star_power", 0),
            "superstar_count": stars.get("superstar_count", 0),
            "top_player_overall": stars.get("top_player_overall", 0),
            "top_player_name": stars.get("top_player_name", ""),
            "superstar_revenue_boost": 0.0,
            "superstar_tags": [],
            "global_draw_revenue_boost": round(global_draw_m * 0.62, 1),
            "global_draw_players": list(global_draw.get("global_draw_players") or []),
            "global_draw_tags": list(global_draw.get("global_draw_tags") or []),
            "playoff_revenue": 0.0,
            "relocation_risk": round(relocation_risk, 3),
            "relocation_risk_label": _relocation_label(relocation_risk),
            "revenue_status": revenue_status,
            "reason_tags": summer_tags[:2],
            "revenue_yoy_delta": yoy["revenue_yoy_delta"],
            "revenue_yoy_direction": yoy["revenue_yoy_direction"],
            "revenue_prior_m": yoy.get("revenue_prior_m"),
            "conduct_revenue_modifier": 1.0,
            "is_user": bool(is_user),
            "revenue_profile": "summer_season_tickets",
        }
        row["market_pressure"] = _market_pressure_reason(row, team)
        return row

    perf_mult = 0.82 + win_pct * 0.38
    fan_mult = 0.70 + (fan_sent / 100.0) * 0.45
    arena_mult = 0.92 + arena * 0.14

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
    revenue += global_draw_m
    revenue += _playoff_revenue_bonus(session, team_id)

    conduct_rev_mult = 1.0
    try:
        from app.sim_engine.franchise.conduct_incidents import get_team_revenue_modifier  # noqa: WPS433

        conduct_rev_mult = float(get_team_revenue_modifier(session, team_id) or 1.0)
        revenue *= conduct_rev_mult
    except Exception:
        conduct_rev_mult = 1.0

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
    attendance_rate = _clamp(
        0.55
        + win_pct * 0.25
        + (fan_sent / 100.0) * 0.22
        + stars["star_power"] * 0.03
        + min(0.10, global_draw_m * 0.0025),
        0.35,
        0.99,
    )

    superstar_tags = list(star_boost.get("superstar_tags") or [])
    reason_tags: List[str] = list(global_draw.get("global_draw_tags") or []) + list(superstar_tags)
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
    if conduct_rev_mult < 0.97:
        reason_tags.append("Conduct Fallout")

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
    yoy = _revenue_yoy_delta(session, team_id, sy, revenue, win_pct)

    abbr = _franchise_team_abbrev(team)
    name = _display_team(team)

    row: Dict[str, Any] = {
        "id": str(team_id),
        "team_id": str(team_id),
        "name": name,
        "team_name": name,
        "abbreviation": abbr,
        "abbr": abbr,
        "logo": abbr,
        "market_size": tier_key,
        "market_tier": tier_label,
        "market_tier_key": tier_key,
        "revenue": round(revenue, 1),
        "revenue_m": round(revenue, 1),
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
        "global_draw_revenue_boost": round(global_draw_m, 1),
        "global_draw_players": list(global_draw.get("global_draw_players") or []),
        "global_draw_tags": list(global_draw.get("global_draw_tags") or []),
        "playoff_revenue": round(_playoff_revenue_bonus(session, team_id), 1),
        "relocation_risk": round(relocation_risk, 3),
        "relocation_risk_label": _relocation_label(relocation_risk),
        "revenue_status": revenue_status,
        "reason_tags": reason_tags,
        "trade_heat": round(trade_heat, 1),
        "revenue_yoy_delta": yoy["revenue_yoy_delta"],
        "revenue_yoy_direction": yoy["revenue_yoy_direction"],
        "revenue_prior_m": yoy.get("revenue_prior_m"),
        "conduct_revenue_modifier": round(conduct_rev_mult, 3),
        "is_user": bool(is_user),
        "revenue_profile": "in_season",
        "win_pct": round(win_pct, 3),
        "arena_quality": round(arena, 3),
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


def calculate_escrow_progress(
    league_revenue: float,
    league_health: float,
    session: Optional[FranchiseSession] = None,
) -> Dict[str, float]:
    """Track a lightweight escrow ledger on the session; formula remains the target."""
    target = round(league_revenue * 0.115, 1)
    collected_formula = round(target * _clamp(0.82 + league_health * 0.28, 0.65, 1.12), 1)

    ledger: Dict[str, Any] = {}
    if session is not None:
        raw = getattr(session, "escrow_ledger", None)
        if isinstance(raw, dict):
            ledger = raw
        else:
            ledger = {}
            try:
                session.escrow_ledger = ledger
            except Exception:
                pass
        sy = str(int(getattr(session, "season_calendar_year", 2025) or 2025))
        season_row = ledger.get(sy)
        if not isinstance(season_row, dict):
            season_row = {
                "target_m": target,
                "collected_m": collected_formula,
                "entries": [],
            }
            ledger[sy] = season_row
        else:
            # Blend toward current formula so health shifts move the ledger without wiping history.
            prev = _safe_float(season_row.get("collected_m"), collected_formula)
            season_row["target_m"] = target
            season_row["collected_m"] = round(prev * 0.65 + collected_formula * 0.35, 1)
            entries = season_row.get("entries")
            if not isinstance(entries, list):
                entries = []
                season_row["entries"] = entries
            if len(entries) < 48:
                entries.append(
                    {
                        "day": int(getattr(session, "calendar_cursor", 0) or 0),
                        "collected_m": season_row["collected_m"],
                        "target_m": target,
                    }
                )
        collected = round(_safe_float(season_row.get("collected_m"), collected_formula), 1)
    else:
        collected = collected_formula

    progress = collected / max(target, 1.0)
    return {
        "escrow_target": target,
        "escrow_collected": collected,
        "escrow_progress": round(progress, 3),
        "escrow_ledger_active": bool(session is not None),
    }


def calculate_salary_cap_projection(
    session: FranchiseSession,
    league_state: Dict[str, Any],
) -> Dict[str, Any]:
    from services.franchise_sim import ensure_session_nhl_salary_cap  # noqa: WPS433

    current_cap = float(ensure_session_nhl_salary_cap(session))

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


def build_cap_forecast_series(session: FranchiseSession, cap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Authoritative multi-year sketch from current + next scalars (not a full HRR model)."""
    from app.sim_engine.economy.cap_engine import nhl_upper_limit_millions  # noqa: WPS433

    sy = int(getattr(session, "season_calendar_year", 2025) or 2025)
    current = _safe_float(cap.get("salary_cap"), nhl_upper_limit_millions(sy))
    projected = _safe_float(cap.get("projected_salary_cap"), current)
    growth = projected - current
    if abs(growth) < 0.05:
        growth = current * 0.025
    uncertainty = max(0.8, abs(growth) * 0.35 + 1.2)
    series: List[Dict[str, Any]] = []
    for i in range(5):
        year = sy + i
        if i == 0:
            value = current
            source = "current"
        elif i == 1:
            value = projected
            source = "projected_next"
        else:
            # Blend model growth with published NHL table when available.
            table = nhl_upper_limit_millions(year)
            extrapolated = current + growth * i * (0.85 + (0.92 ** i) * 0.2)
            value = round((table * 0.55 + extrapolated * 0.45), 1)
            source = "extrapolated"
        band = uncertainty * (0.7 + i * 0.35)
        series.append(
            {
                "year": year,
                "season": f"{year}-{year + 1}",
                "label": str(year),
                "cap": round(float(value), 1),
                "salary_cap": round(float(value), 1),
                "low": round(float(value) - band, 1),
                "high": round(float(value) + band, 1),
                "source": source,
            }
        )
    return series


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
        "display_only": True,
        "interactive": False,
        "brief": (
            "Intelligence display only — negotiations are pressure estimates. "
            "They do not change cap rules, LTIR, lottery, or contract limits in-sim."
        ),
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


def _league_ops_cache_key(session: FranchiseSession) -> tuple:
    return (
        int(getattr(session, "_stats_revision", 0) or 0),
        int(getattr(session, "season_calendar_year", 0) or 0),
        int(getattr(session, "calendar_cursor", 0) or 0),
        str(getattr(session, "phase", "") or ""),
        str(session.user_team_id or ""),
    )


def invalidate_league_ops_cache(session: FranchiseSession) -> None:
    session._cached_league_operations = None
    session._cached_league_operations_key = None


def get_cached_league_operations_payload(session: FranchiseSession) -> Dict[str, Any]:
    """Reuse league-ops until stats/day/phase identity changes — identical math, less waste."""
    key = _league_ops_cache_key(session)
    cached = getattr(session, "_cached_league_operations", None)
    cached_key = getattr(session, "_cached_league_operations_key", None)
    if isinstance(cached, dict) and cached and cached_key == key:
        return cached
    payload = build_league_operations_payload(session)
    session._cached_league_operations = payload
    session._cached_league_operations_key = key
    return payload


def slim_league_operations_for_state(ops: Dict[str, Any]) -> Dict[str, Any]:
    """Lean /state only needs pulse scalars + user row — full 32-team table is on /league-operations."""
    if not isinstance(ops, dict):
        return {}
    out = {k: v for k, v in ops.items() if k != "teams"}
    # Keep a tiny user-only teams stub so older UI that indexes teams[0] still works.
    user = ops.get("user_team")
    if isinstance(user, dict):
        out["teams"] = [user]
    else:
        out["teams"] = []
    out["_slim"] = True
    return out


def build_league_operations_payload(session: FranchiseSession) -> Dict[str, Any]:
    from services.perf_profiler import span

    with span("league_ops.build"):
        return _build_league_operations_payload_impl(session)


def _build_league_operations_payload_impl(session: FranchiseSession) -> Dict[str, Any]:
    from services.franchise_sim import (  # noqa: WPS433
        _sync_session_phase_from_calendar,
        ensure_session_nhl_salary_cap,
    )

    try:
        _sync_session_phase_from_calendar(session)
    except Exception:
        pass
    ensure_session_nhl_salary_cap(session)

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
    escrow = calculate_escrow_progress(league_revenue, revenue_health, session=session)
    league_state.update(escrow)
    league_state["cba_pressure"] = calculate_cba_pressure(league_state)

    cap = calculate_salary_cap_projection(session, league_state)
    cap_forecast = build_cap_forecast_series(session, cap)
    cba = _build_cba_block(session, league_state["cba_pressure"])

    def _team_id(t: Dict[str, Any]) -> str:
        return str(t.get("id") or t.get("team_id") or "")

    def _team_abbr(t: Dict[str, Any]) -> str:
        return str(t.get("abbreviation") or t.get("abbr") or "")

    watchlist = sorted(team_rows, key=lambda r: -_safe_float(r.get("relocation_risk", 0)))[:5]
    relocation = {
        "watchlist": [
            {
                "id": _team_id(t),
                "abbreviation": _team_abbr(t),
                "risk": t.get("relocation_risk_label", "Low"),
                "risk_score": t.get("relocation_risk", 0),
                "risk_bar_pct": t.get("relocation_bar_pct", 0),
                "reason": _relocation_reason_tag(t, session.team_by_id.get(_team_id(t))),
                "pressure": t.get("market_pressure", "Stable"),
                "label": f"{_team_abbr(t)} | {t.get('relocation_risk_label', 'Low')} | {t.get('market_pressure', 'Stable')}",
            }
            for t in watchlist
            if _team_id(t)
        ],
        "highest_risk_team": _team_abbr(watchlist[0]) if watchlist else "",
        "league_stability": "Stable" if revenue_health >= 0.55 and losing_teams <= 12 else ("Shaky" if losing_teams <= 20 else "Weak"),
    }

    cap_drivers = _build_cap_drivers(league_state, cap)
    owner_mood = _build_owner_mood(league_state, cap)
    league_pulse = _build_league_pulse(league_state, cap, relocation)
    health_label = _league_health_label(revenue_health, losing_teams)

    user_row = next((t for t in team_rows if str(t.get("id")) == uid or str(t.get("team_id")) == uid), team_rows[0] if team_rows else {})
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
        "cap_forecast": cap_forecast,
        "cap_forecast_note": "Projection sketch from current ceiling + next-year model (not full HRR accounting).",
        "league_revenue": league_revenue,
        "escrow_target": escrow["escrow_target"],
        "escrow_collected": escrow["escrow_collected"],
        "escrow_progress": escrow["escrow_progress"],
        "escrow_ledger_active": escrow.get("escrow_ledger_active", False),
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
        "phase": str(getattr(session, "phase", "") or ""),
    }
