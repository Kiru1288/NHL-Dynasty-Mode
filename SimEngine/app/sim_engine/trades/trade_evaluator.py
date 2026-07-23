"""
Trade package evaluation — value scoring, AI interest, acceptance decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    TradePackage,
    normalize_trade_package,
)
from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
from app.sim_engine.trades.trade_rules import validate_trade_rules

CPU_AMBIENT_FAIRNESS_GAP_MAX = 7.0
CPU_AMBIENT_MIN_INTEREST = 0.50
from app.sim_engine.trades.trade_value import (
    evaluate_asset_value,
    evaluate_package_value,
    evaluate_player_asset_value,
    evaluate_pick_asset_value,
)
from app.sim_engine.trades.trade_pick_registry import get_pick_by_id
from app.sim_engine.trades.trade_asset import find_player_on_team_roster, player_display_name
from app.sim_engine.economy.team_needs import TeamNeeds


def _team_display(team: Any, team_id: str) -> str:
    for key in ("name", "team_name", "display_name"):
        val = getattr(team, key, None) if team is not None else None
        if val:
            return str(val)
    return str(team_id)


def _team_window(team: Any) -> str:
    # If season is underway, use current pace to avoid stale window labels.
    try:
        gp = float(getattr(team, "gp", getattr(team, "games_played", 0)) or 0)
        pts = float(getattr(team, "pts", getattr(team, "points", 0)) or 0)
    except Exception:
        gp = 0.0
        pts = 0.0
    if gp >= 10:
        p_pct = pts / max(1.0, gp * 2.0)
        if p_pct >= 0.55:
            return "contender"
        if p_pct <= 0.43:
            return "rebuild"

    for key in ("gm_window", "window"):
        w = str(getattr(team, key, "") or "").lower()
        if w in ("rebuild", "contender", "declining", "emerging"):
            return w
    return "unknown"


def _is_severe_overpay(net_for_team: float, incoming_vals: List[Dict[str, Any]]) -> bool:
    """True when a team receives clearly premium surplus value."""
    if float(net_for_team) >= 16.0:
        return True
    premium_count = sum(1 for v in incoming_vals if float(v.get("total", 0.0) or 0.0) >= 65.0)
    return float(net_for_team) >= 12.0 and premium_count >= 1


def _deadline_phase(context: Optional[Dict[str, Any]]) -> float:
    if not context:
        return 0.0
    return float(context.get("deadline_phase", 0.0) or 0.0)


def _asset_bucket(asset: Dict[str, Any]) -> str:
    val = float(asset.get("total", 0.0) or 0.0)
    typ = str(asset.get("type") or "").lower()
    if typ == "pick":
        dbg = asset.get("value_debug") or {}
        round_guess = 7
        try:
            round_guess = int(dbg.get("round", dbg.get("pick_round", 7)) or 7)
        except Exception:
            round_guess = 7
        if val >= 70 or round_guess == 1:
            return "premium"
        if val >= 40 or round_guess == 2:
            return "strong"
        if val >= 18:
            return "medium"
        return "low"
    if val >= 75:
        return "premium"
    if val >= 45:
        return "strong"
    if val >= 20:
        return "medium"
    return "low"


def _draft_floor_pick_swap(
    context: Optional[Dict[str, Any]],
    league: Any,
    incoming: List[Any],
    outgoing_year: int,
    outgoing_round: int,
) -> bool:
    """True when draft-day context swaps for another same-class pick (slot move)."""
    if not (context or {}).get("draft_day_trade"):
        return False
    for asset in incoming:
        if not isinstance(asset, DraftPickTradeAsset):
            continue
        row = get_pick_by_id(league, asset.pick_id) or {}
        yr = int(row.get("year", asset.year or 0) or 0)
        rnd = int(row.get("round", asset.round or 99) or 99)
        if yr == outgoing_year and abs(rnd - outgoing_round) <= 1:
            return True
    return False


def _pick_sale_guardrails(
    team_id: str,
    package: TradePackage,
    team_by_id: Dict[str, Any],
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    reasons: List[str] = []
    team = team_by_id.get(team_id)
    if team is None:
        return reasons
    season_year = int((context or {}).get("season_year", 2025) or 2025)
    window = _team_window(team)
    outgoing = package.outgoing_by_team.get(team_id, [])
    incoming = package.incoming_by_team.get(team_id, [])
    incoming_vals: List[Dict[str, Any]] = []
    for asset in incoming:
        src = team_by_id.get(getattr(asset, "source_team_id", ""))
        if src is None:
            continue
        try:
            incoming_vals.append(evaluate_asset_value(asset, src, team, league, context=context))
        except Exception:
            continue
    incoming_buckets = [_asset_bucket(v) for v in incoming_vals if isinstance(v, dict)]
    low_count = sum(1 for b in incoming_buckets if b == "low")
    med_count = sum(1 for b in incoming_buckets if b == "medium")
    has_strong_or_better = any(b in ("premium", "strong") for b in incoming_buckets)

    for asset in outgoing:
        if not isinstance(asset, DraftPickTradeAsset):
            continue
        row = get_pick_by_id(league, asset.pick_id) or {}
        rnd = int(row.get("round", asset.round or 7) or 7)
        yr = int(row.get("year", asset.year or season_year) or season_year)
        orig = str(row.get("original_team_id") or "")
        owner = str(row.get("current_owner_team_id") or "")
        protection = row.get("protection")
        pick_val = evaluate_pick_asset_value(row, team, team, league, context=context)
        dbg = pick_val.get("value_debug") or {}
        risk = float(dbg.get("projected_finish_risk", 0.0) or 0.0)
        points_pct = dbg.get("points_pct")
        likely_lottery = rnd == 1 and (
            risk >= 9.0 or (points_pct is not None and float(points_pct) < 0.5)
        )
        premium_pick = rnd == 1 and (yr <= season_year + 1 or likely_lottery or not protection)

        if premium_pick and not has_strong_or_better:
            reasons.append(
                "AI rejected: quantity of low-value picks does not satisfy premium asset requirement."
            )

        if premium_pick and (low_count >= 3 or (low_count + med_count) >= 5):
            reasons.append(
                "AI rejected: package lacks a premium asset for a likely lottery first."
            )

        if window == "rebuild" and owner == team_id and rnd == 1 and yr <= season_year + 1 and not protection:
            if not _draft_floor_pick_swap(context, league, incoming, yr, rnd):
                reasons.append(
                    "AI rejected: rebuilding team will not move its own unprotected first."
                )

        if window != "contender" and owner == team_id and rnd == 1 and yr <= season_year + 1:
            if not _draft_floor_pick_swap(context, league, incoming, yr, rnd):
                reasons.append(
                    "AI rejected: non-contender heavily protects current/next-year first-round picks."
                )

        if likely_lottery and not any(b == "premium" for b in incoming_buckets):
            reasons.append(
                "AI rejected: likely lottery first requires premium market return."
            )

        if orig and orig != owner and likely_lottery and not has_strong_or_better:
            reasons.append(
                "AI rejected: weak-origin premium pick requires strong incoming asset."
            )
    return reasons


def _ai_interest_for_team(
    team_id: str,
    package: TradePackage,
    team_by_id: Dict[str, Any],
    league: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    user_team_id: Optional[str] = None,
    rules_ok: bool = True,
) -> tuple[float, List[str]]:
    """Return (interest 0-1, rejection reasons for this team)."""
    reasons: List[str] = []
    if not rules_ok:
        return 0.0, reasons

    if user_team_id and str(team_id) == str(user_team_id):
        return 1.0, reasons

    team = team_by_id.get(team_id)
    if team is None:
        return 0.0, ["Team not found"]

    val = evaluate_package_value(package, team_id, league, team_by_id, context=context)
    net = float(val.get("net", 0.0))
    window = _team_window(team)

    # Base interest from net value received
    if net >= 12:
        interest = 0.85
    elif net >= 5:
        interest = 0.72
    elif net >= 0:
        interest = 0.55
    elif net >= -8:
        interest = 0.38
        # Ambient CPU market: near-even hockey swaps should not die on float noise.
        if (context or {}).get("cpu_ambient_trade") and net >= -2.75:
            interest = 0.55
    else:
        interest = 0.18
        reasons.append(f"Package net value ({net:.1f}) is too unfavorable")

    # Rebuilder pick protection
    if window == "rebuild":
        for asset in package.outgoing_by_team.get(team_id, []):
            if isinstance(asset, DraftPickTradeAsset):
                row = get_pick_by_id(league, asset.pick_id) or {}
                rnd = int(row.get("round", asset.round) or 7)
                if rnd == 1:
                    interest = min(interest, 0.22)
                    reasons.append("Team is rebuilding and will not move a first-round pick without a premium return")
                elif rnd == 2:
                    interest = min(interest, 0.42)
            if isinstance(asset, PlayerTradeAsset):
                src = team_by_id.get(team_id)
                if src:
                    p, _ = find_player_on_team_roster(src, asset.player_id)
                    if p is not None:
                        ident = getattr(p, "identity", None)
                        age = int(getattr(ident, "age", getattr(p, "age", 25)) or 25)
                        ovr_fn = getattr(p, "ovr", None)
                        ovr = float(ovr_fn() if callable(ovr_fn) else ovr_fn or 0)
                        if ovr <= 1.5:
                            ovr *= 99
                        if age <= 22 and ovr >= 78:
                            interest = min(interest, 0.30)
                            reasons.append("Rebuilding team reluctant to move elite young prospect")

    # Anti-quantity-spam guardrails for premium draft capital.
    reasons.extend(
        _pick_sale_guardrails(
            team_id,
            package,
            team_by_id,
            league,
            context=context,
        )
    )
    # Contender deadline boost for fit
    if window == "contender" and _deadline_phase(context) > 0.35:
        if net >= -3:
            interest = min(1.0, interest + 0.12)

    # Star consolidation tax — trading away best asset for many small pieces
    outgoing = val.get("outgoing") or []
    incoming = val.get("incoming") or []
    severe_overpay = _is_severe_overpay(net, incoming)
    if severe_overpay:
        pick_lock_tokens = (
            "rebuilding team will not move",
            "non-contender heavily protects",
            "likely lottery first requires",
            "premium pick requires",
            "first-round",
            "first round",
        )
        reasons = [r for r in reasons if not any(tok in r.lower() for tok in pick_lock_tokens)]

    if any("premium" in r.lower() or "first" in r.lower() for r in reasons):
        interest = min(interest, 0.18)
    elif severe_overpay and window != "contender":
        # Non-contenders should still engage on clear overpays.
        interest = max(interest, 0.64)

    if outgoing:
        best_out = max(float(x.get("total", 0)) for x in outgoing)
        if best_out >= 75 and len(incoming) >= 3:
            small_in = sum(1 for x in incoming if float(x.get("total", 0)) < 25)
            if small_in >= 2 and sum(float(x.get("total", 0)) for x in incoming) < best_out - 5:
                interest = min(interest, 0.25)
                reasons.append("Too many low-value assets do not equal the outgoing star")

    # Cap relief acceptance for cap-strapped teams
    cap_tier = str(getattr(team, "cap_pressure_tier", getattr(team, "cap_pressure", "")) or "").lower()
    if cap_tier in ("cap_hell", "critical") and net >= -5:
        interest = min(1.0, interest + 0.08)

    tank_map = (context or {}).get("tank_pressure_by_team") or {}
    tank_row = tank_map.get(str(team_id)) or {}
    tank_pressure = int(tank_row.get("tank_pressure") or getattr(team, "_franchise_tank_pressure", 0) or 0)
    tank_mode = str(tank_row.get("tank_mode") or getattr(team, "_franchise_tank_mode", "none") or "none")
    if tank_pressure >= 30 and window != "contender":
        outgoing_assets = package.outgoing_by_team.get(team_id, [])
        incoming_assets = package.incoming_by_team.get(team_id, [])
        outgoing_players = [a for a in outgoing_assets if isinstance(a, PlayerTradeAsset)]
        incoming_picks = [a for a in incoming_assets if isinstance(a, DraftPickTradeAsset)]
        if outgoing_players and incoming_picks and tank_pressure >= 50:
            bonus = 0.05 + max(0.0, (tank_pressure - 50) * 0.003)
            if tank_mode == "hard_tank" and tank_row.get("owns_own_first", True):
                bonus += 0.04
            if tank_mode == "hard_tank" and not tank_row.get("owns_own_first", True):
                bonus = min(bonus, 0.03)
            interest = min(1.0, interest + bonus)
            reasons.append("Transcendent lottery chase — willing to move veterans for futures")
        if tank_pressure >= 70 and net >= -4:
            interest = min(1.0, interest + 0.04)
    if tank_pressure >= 50 and window != "contender" and _deadline_phase(context) > 0.25:
        incoming_players = [a for a in package.incoming_by_team.get(team_id, []) if isinstance(a, PlayerTradeAsset)]
        for asset in incoming_players:
            src = team_by_id.get(str(asset.team_id or "")) or team
            p, _ = find_player_on_team_roster(src, asset.player_id)
            if p is None:
                continue
            ident = getattr(p, "identity", None)
            age = int(getattr(ident, "age", getattr(p, "age", 25)) or 25)
            if age >= 30:
                interest = max(0.12, interest - 0.14)
                reasons.append("Tank mode — not buying rentals at the deadline")
                break

    interest = max(0.0, min(1.0, interest))
    # Draft-floor same-class pick swaps: allow slot moves without premium futures tax.
    if (context or {}).get("draft_day_trade"):
        incoming_assets = package.incoming_by_team.get(team_id, [])
        outgoing_assets = package.outgoing_by_team.get(team_id, [])
        in_picks = [a for a in incoming_assets if isinstance(a, DraftPickTradeAsset)]
        out_picks = [a for a in outgoing_assets if isinstance(a, DraftPickTradeAsset)]
        if in_picks and out_picks and net >= -14:
            interest = max(interest, 0.58)
            reasons = [
                r
                for r in reasons
                if "first-round" not in r.lower()
                and "first round" not in r.lower()
                and "premium" not in r.lower()
                and "lottery" not in r.lower()
            ]
    accept_threshold = 0.52 if window == "contender" else 0.58 if window == "rebuild" else 0.55
    if interest < accept_threshold and not reasons:
        reasons.append("Package does not meet team valuation threshold")

    return interest, reasons


def _accept_threshold_for_team(team: Any) -> float:
    window = _team_window(team)
    if window == "contender":
        return 0.52
    if window == "rebuild":
        return 0.58
    return 0.55


def _summarize_team_needs(team: Any) -> Dict[str, Any]:
    needs_model = TeamNeeds()
    needs = getattr(team, "needs", None) or needs_model.evaluate(team)
    labels: List[str] = []
    if float(needs.get("top_line_forward", 0.0)) >= 0.55:
        labels.append("Top-line forward")
    if float(needs.get("top_4_defense", 0.0)) >= 0.55:
        labels.append("Top-4 defense")
    if float(needs.get("goalie", 0.0)) >= 0.55:
        labels.append("Goalie")
    if float(needs.get("depth_forward", 0.0)) >= 0.55:
        labels.append("Depth forward")
    window = _team_window(team)
    shopping: List[str] = []
    values: List[str] = []
    if window == "rebuild":
        values.extend(["Draft picks", "Young prospects", "Cap flexibility"])
        shopping.append("Veteran cap hits")
    elif window == "contender":
        values.extend(["NHL-ready talent", "Playoff experience"])
        shopping.append("Rental upgrades")
    else:
        values.append("Balanced value")
    return {
        "needs": labels,
        "shopping": shopping,
        "values": values,
        "window": window,
    }


def _build_verdict(
    *,
    can_execute: bool,
    accepted: bool,
    blocking: List[str],
    warnings: List[str],
    user_net: float,
) -> str:
    blob = " ".join(blocking + warnings).lower()
    if not can_execute:
        if "cap" in blob or "salary" in blob:
            return "cap_illegal"
        if "roster" in blob and ("maximum" in blob or "exceed" in blob):
            return "roster_illegal"
        if "nmc" in blob or "ntc" in blob or "clause" in blob:
            return "ntc_nmc_conflict"
        if "not found" in blob or "does not own" in blob or "unavailable" in blob:
            return "player_unavailable"
        if "pick" in blob and "own" in blob:
            return "asset_not_owned"
        return "blocked"
    if not accepted:
        if user_net < -10:
            return "trade_value_too_low"
        return "rejected"
    if warnings:
        return "needs_adjustment"
    if abs(user_net) <= 4:
        return "accepted"
    if user_net > 8:
        return "needs_adjustment"
    return "accepted"


def _build_explanation(
    *,
    verdict: str,
    user_team_id: Optional[str],
    partner_id: Optional[str],
    team_by_id: Dict[str, Any],
    user_net: float,
    partner_net: float,
    blocking: List[str],
    ai_reasons_by_team: Dict[str, List[str]],
    value_breakdown: Dict[str, Any],
    team_needs_impact: Dict[str, Any],
) -> str:
    partner = team_by_id.get(str(partner_id or ""))
    partner_name = _team_display(partner, str(partner_id or "Partner"))

    if blocking:
        return blocking[0]
    if verdict == "rejected":
        reasons = ai_reasons_by_team.get(str(partner_id or ""), [])
        if reasons:
            return f"{partner_name} rejects: {reasons[0]}"
        if partner_net < -5:
            return f"{partner_name} would lose too much value in this deal."
        return f"{partner_name} is not interested at the current offer."
    if verdict == "trade_value_too_low":
        return "This is an overpay based on projected surplus value."
    if verdict == "cap_illegal":
        return "One or more teams cannot absorb the cap hit after this trade."
    if verdict == "ntc_nmc_conflict":
        return "A player has a no-trade or no-movement clause blocking the deal."
    if verdict == "needs_adjustment" and user_net > 8:
        return "You are giving up significantly more value than you receive."
    if verdict == "needs_adjustment" and user_net < -8:
        return "This deal improves your roster but may be hard to get approved."
    if verdict == "accepted":
        incoming = (value_breakdown.get(str(user_team_id or ""), {}) or {}).get("incoming") or []
        if incoming:
            names = [str(x.get("name", "")) for x in incoming[:2] if x.get("name")]
            if names:
                return f"Balanced deal — you add {', '.join(names)}."
        return "Both sides receive fair value for their competitive window."
    needs = team_needs_impact.get(str(partner_id or ""), {})
    if needs.get("fills_need"):
        return f"{partner_name} likes the fit but wants more compensation."
    return "Package is structurally valid — review value balance before proposing."


def _suggest_counteroffers(
    package: TradePackage,
    *,
    user_team_id: Optional[str],
    partner_id: Optional[str],
    team_by_id: Dict[str, Any],
    league: Any,
    user_net: float,
    partner_net: float,
    context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if not partner_id or not user_team_id:
        return suggestions
    partner = team_by_id.get(str(partner_id))
    if partner is None:
        return suggestions
    window = _team_window(partner)
    if partner_net < -4:
        if window == "rebuild":
            suggestions.append({
                "label": "Add a draft pick",
                "explanation": "Rebuilding teams usually want future assets over marginal NHL pieces.",
            })
        else:
            suggestions.append({
                "label": "Sweeten with a higher-value piece",
                "explanation": "Partner net value is negative — add a better asset or retain less salary.",
            })
    if user_net < -6:
        suggestions.append({
            "label": "Ask for an additional pick or prospect",
            "explanation": "You are overpaying relative to incoming value.",
        })
    outgoing = package.outgoing_by_team.get(str(partner_id), [])
    for asset in outgoing:
        if not isinstance(asset, PlayerTradeAsset):
            continue
        p, _ = find_player_on_team_roster(partner, asset.player_id)
        if p is None:
            continue
        ovr_fn = getattr(p, "ovr", None)
        ovr = float(ovr_fn() if callable(ovr_fn) else ovr_fn or 0)
        if ovr <= 1.5:
            ovr *= 99
        if ovr >= 84:
            suggestions.append({
                "label": f"Request more for {player_display_name(p)}",
                "explanation": "Elite outgoing talent rarely moves without a premium return.",
            })
            break
    return suggestions[:3]


def _team_needs_impact_for_trade(
    package: TradePackage,
    team_id: str,
    team_by_id: Dict[str, Any],
    league: Any,
    *,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    team = team_by_id.get(team_id)
    if team is None:
        return {"fills_need": False, "weakens": [], "strengthens": []}
    needs = getattr(team, "needs", None) or TeamNeeds().evaluate(team)
    strengthens: List[str] = []
    weakens: List[str] = []
    fills_need = False

    for asset in package.incoming_by_team.get(team_id, []):
        if not isinstance(asset, PlayerTradeAsset):
            if isinstance(asset, DraftPickTradeAsset):
                if _team_window(team) == "rebuild":
                    strengthens.append("Draft capital")
                    fills_need = True
            continue
        src = team_by_id.get(asset.source_team_id)
        if src is None:
            continue
        p, _ = find_player_on_team_roster(src, asset.player_id)
        if p is None:
            continue
        pos = str(getattr(getattr(p, "identity", None), "position", getattr(p, "position", "")) or "")
        pos_u = str(getattr(pos, "value", pos)).upper()
        if pos_u in ("C", "LW", "RW", "W", "F") and float(needs.get("top_line_forward", 0)) >= 0.5:
            strengthens.append("Forward depth")
            fills_need = True
        if pos_u in ("D", "LD", "RD") and float(needs.get("top_4_defense", 0)) >= 0.5:
            strengthens.append("Defense")
            fills_need = True
        if pos_u == "G" and float(needs.get("goalie", 0)) >= 0.5:
            strengthens.append("Goaltending")
            fills_need = True

    for asset in package.outgoing_by_team.get(team_id, []):
        if isinstance(asset, PlayerTradeAsset):
            p, _ = find_player_on_team_roster(team, asset.player_id)
            if p is not None:
                weakens.append(player_display_name(p))

    return {
        "fills_need": fills_need,
        "strengthens": strengthens[:4],
        "weakens": weakens[:4],
        "priority_needs": _summarize_team_needs(team).get("needs") or [],
    }


def evaluate_trade_package(
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    *,
    league: Any,
    team_by_id: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    user_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    ensure_draft_pick_registry(
        league,
        start_year=ctx.get("season_year"),
        years_ahead=4,
    )

    package = normalize_trade_package(assets_by_team, team_by_id=team_by_id)
    rules = validate_trade_rules(
        package,
        league,
        team_by_id,
        context=ctx,
        user_team_id=user_team_id,
    )

    score_for_teams: Dict[str, float] = {}
    value_breakdown: Dict[str, Any] = {}
    interest_level: Dict[str, float] = {}
    rejection_reasons: List[str] = list(rules.get("blocking_reasons") or [])
    warnings: List[str] = list(rules.get("warnings") or [])
    ai_reasons_by_team: Dict[str, List[str]] = {}

    for tid in package.participating_team_ids:
        pb = evaluate_package_value(package, tid, league, team_by_id, context=ctx)
        score_for_teams[tid] = round(float(pb.get("net", 0.0)) + 50.0, 2)
        value_breakdown[tid] = pb
        interest, team_reasons = _ai_interest_for_team(
            tid,
            package,
            team_by_id,
            league,
            context=ctx,
            user_team_id=user_team_id,
            rules_ok=bool(rules.get("ok")),
        )
        interest_level[tid] = round(interest, 3)
        if team_reasons:
            ai_reasons_by_team[tid] = team_reasons

    nets = [float(value_breakdown[t].get("net", 0.0)) for t in package.participating_team_ids if t in value_breakdown]
    fairness_gap = round(max(nets) - min(nets), 2) if len(nets) >= 2 else 0.0

    can_execute = bool(rules.get("ok"))

    # AI acceptance — all non-user teams must meet interest threshold.
    # cap_casualty_trade bypasses AI interest: over-cap teams may need salary relief
    # even when a partner would normally reject the package. Hard validation still applies.
    accepted = can_execute
    if accepted and (ctx or {}).get("cap_casualty_trade"):
        accepted = True
    elif accepted:
        for tid in package.participating_team_ids:
            if user_team_id and str(tid) == str(user_team_id):
                continue
            team_obj = team_by_id.get(tid)
            threshold = _accept_threshold_for_team(team_obj) if team_obj is not None else 0.55
            if (ctx or {}).get("cpu_ambient_trade"):
                # Keep a firm floor, but do not let rebuild thresholds (0.58) block
                # near-fair ambient depth swaps that already pass fairness_gap.
                threshold = max(0.50, min(0.55, threshold))
                if fairness_gap <= CPU_AMBIENT_FAIRNESS_GAP_MAX and interest_level.get(tid, 0.0) >= 0.50:
                    threshold = min(threshold, 0.50)
            if interest_level.get(tid, 0.0) < threshold:
                accepted = False
                tname = _team_display(team_obj, tid)
                for r in ai_reasons_by_team.get(tid, []):
                    msg = f"{tname}: {r}"
                    if msg not in rejection_reasons:
                        rejection_reasons.append(msg)
        if accepted and (ctx or {}).get("cpu_ambient_trade") and fairness_gap > CPU_AMBIENT_FAIRNESS_GAP_MAX:
            # Draft-floor same-class pick swaps can have larger projected-value gaps by slot.
            draft_gap_max = 22.0 if (ctx or {}).get("draft_day_trade") else CPU_AMBIENT_FAIRNESS_GAP_MAX
            if fairness_gap > draft_gap_max:
                accepted = False
                msg = f"Ambient CPU trade fairness gap too wide ({fairness_gap} > {draft_gap_max})"
                if msg not in rejection_reasons:
                    rejection_reasons.append(msg)

    if can_execute and not accepted and not rejection_reasons:
        rejection_reasons.append("One or more teams would reject this package based on value and team direction")

    partner_id = next(
        (tid for tid in package.participating_team_ids if not user_team_id or str(tid) != str(user_team_id)),
        None,
    )
    user_breakdown = value_breakdown.get(str(user_team_id or ""), {}) if user_team_id else {}
    partner_breakdown = value_breakdown.get(str(partner_id or ""), {}) if partner_id else {}
    user_net = float(user_breakdown.get("net", 0.0))
    partner_net = float(partner_breakdown.get("net", 0.0))
    user_in = float(user_breakdown.get("incoming_total", 0.0))
    user_out = float(user_breakdown.get("outgoing_total", 0.0))
    partner_in = float(partner_breakdown.get("incoming_total", 0.0))
    partner_out = float(partner_breakdown.get("outgoing_total", 0.0))

    if partner_id:
        p_incoming = partner_breakdown.get("incoming") or []
        if _is_severe_overpay(partner_net, p_incoming):
            lock_tokens = (
                "rebuilding team will not move",
                "non-contender heavily protects",
                "likely lottery first requires",
            )
            rejection_reasons = [r for r in rejection_reasons if not any(t in r.lower() for t in lock_tokens)]

    team_needs_impact: Dict[str, Any] = {}
    for tid in package.participating_team_ids:
        team_needs_impact[tid] = _team_needs_impact_for_trade(
            package, tid, team_by_id, league, context=ctx
        )

    verdict = _build_verdict(
        can_execute=can_execute,
        accepted=accepted,
        blocking=rejection_reasons,
        warnings=warnings,
        user_net=user_net,
    )
    explanation = _build_explanation(
        verdict=verdict,
        user_team_id=user_team_id,
        partner_id=partner_id,
        team_by_id=team_by_id,
        user_net=user_net,
        partner_net=partner_net,
        blocking=rejection_reasons,
        ai_reasons_by_team=ai_reasons_by_team,
        value_breakdown=value_breakdown,
        team_needs_impact=team_needs_impact,
    )
    counteroffers = _suggest_counteroffers(
        package,
        user_team_id=user_team_id,
        partner_id=partner_id,
        team_by_id=team_by_id,
        league=league,
        user_net=user_net,
        partner_net=partner_net,
        context=ctx,
    )

    scouting_confidence = 1.0
    if user_team_id and partner_id:
        partner_incoming = package.incoming_by_team.get(str(user_team_id), [])
        unknown_count = 0
        total_players = 0
        for asset in partner_incoming:
            if not isinstance(asset, PlayerTradeAsset):
                continue
            total_players += 1
            src = team_by_id.get(asset.source_team_id)
            if src is None:
                unknown_count += 1
                continue
            p, _ = find_player_on_team_roster(src, asset.player_id)
            if p is None:
                unknown_count += 1
        if total_players > 0:
            scouting_confidence = round(max(0.35, 1.0 - (unknown_count / total_players) * 0.45), 2)

    asset_breakdown = {
        "user": {
            "incoming": user_breakdown.get("incoming") or [],
            "outgoing": user_breakdown.get("outgoing") or [],
            "incoming_total": user_in,
            "outgoing_total": user_out,
            "net": user_net,
        },
        "partner": {
            "incoming": partner_breakdown.get("incoming") or [],
            "outgoing": partner_breakdown.get("outgoing") or [],
            "incoming_total": partner_in,
            "outgoing_total": partner_out,
            "net": partner_net,
        },
    }

    immersion: Dict[str, Any] = {}
    if partner_id:
        partner_team = team_by_id.get(str(partner_id))
        if partner_team is not None:
            p_needs = _summarize_team_needs(partner_team)
            immersion["partner_needs"] = p_needs.get("needs") or []
            immersion["partner_values"] = p_needs.get("values") or []
            immersion["partner_window"] = p_needs.get("window")
            immersion["market_temperature"] = (
                "Hot" if float(ctx.get("deadline_phase", 0.0)) > 0.5 else "Warm" if float(ctx.get("deadline_phase", 0.0)) > 0.25 else "Cool"
            )

    normalized_assets = {
        tid: [
            {
                "type": a.type,
                "id": getattr(a, "player_id", None) or getattr(a, "pick_id", None),
                "team": a.source_team_id,
                "retained": getattr(a, "retained_pct", 0),
            }
            for a in package.incoming_by_team.get(tid, [])
        ]
        for tid in package.participating_team_ids
    }

    return {
        "accepted": accepted,
        "can_execute": can_execute,
        "verdict": verdict,
        "score": round((user_net + 50.0), 2),
        "user_value": round(user_in, 2),
        "opposing_value": round(partner_in, 2),
        "value_delta": round(user_net, 2),
        "participating_teams": list(package.participating_team_ids),
        "score_for_teams": score_for_teams,
        "fairness_gap": fairness_gap,
        "interest_level": interest_level,
        "rejection_reasons": rejection_reasons,
        "warnings": warnings,
        "cap_impact": rules.get("cap_impact") or {},
        "roster_impact": rules.get("roster_impact") or {},
        "contract_slot_impact": rules.get("contract_slot_impact") or {},
        "team_needs_impact": team_needs_impact,
        "clause_impact": rules.get("clause_impact") or {},
        "value_breakdown": value_breakdown,
        "asset_breakdown": asset_breakdown,
        "scouting_confidence": scouting_confidence,
        "explanation": explanation,
        "suggested_counteroffers": counteroffers,
        "counteroffer": counteroffers[0] if counteroffers else None,
        "immersion": immersion,
        "normalized_assets": normalized_assets,
        "_package": package,
        "_rules": rules,
    }
