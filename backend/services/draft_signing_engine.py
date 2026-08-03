"""
Prospect ELC signing decisions — delegates structured offers to elc_offer_engine.
Kept for backwards-compatible imports from contract_economy / tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def evaluate_elc_signing_decision(
    player: Any,
    team: Any,
    *,
    season_year: int,
    promote_to_nhl: bool = False,
    assignment_plan: Optional[str] = None,
    development_promise: Optional[str] = None,
    template_id: str = "standard_elc",
    offer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.elc_offer_engine import (
        build_offer_from_template,
        evaluate_offer_acceptance,
        prior_reject_count,
    )

    # promote_to_nhl and assignment_plan are mutually exclusive: a player either goes
    # straight to the active NHL roster, or gets a developmental assignment (AHL,
    # junior, college, Europe, camp invite). Never derive one from the other.
    built = offer or build_offer_from_template(
        player,
        season_year=season_year,
        template_id=template_id,
        development_promise=development_promise,
        assignment_plan=None if promote_to_nhl else assignment_plan,
    )
    # Session may be absent in unit tests
    session = getattr(team, "_session_ref", None)
    rejects = prior_reject_count(session, str(getattr(player, "id", "") or "")) if session else 0
    decision = evaluate_offer_acceptance(
        player, team, built, season_year=season_year, prior_rejects=rejects
    )
    return {
        **decision,
        "path": getattr(player, "development_path", None),
        "elc_slide_eligible": bool(getattr(player, "elc_slide_eligible", True)),
        "elc_slide_years_remaining": getattr(player, "elc_slide_years_remaining", 1),
        "offer": built,
    }


def attempt_sign_elc_with_decision(
    session: Any,
    player: Any,
    team: Any,
    *,
    season_year: int,
    promote_to_nhl: bool = False,
    assignment_plan: Optional[str] = None,
    development_promise: Optional[str] = None,
    template_id: str = "standard_elc",
    offer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from services.elc_offer_engine import submit_elc_offer

    # promote_to_nhl and assignment_plan are mutually exclusive (see
    # evaluate_elc_signing_decision). Signing straight onto the NHL roster should
    # never also run a developmental (e.g. AHL) assignment on the same player —
    # that used to fire apply_post_elc_assignment("assign_ahl") and then immediately
    # promote_prospect_to_nhl, leaving in_minors=True on an "NHL" player.
    result = submit_elc_offer(
        session,
        player,
        team,
        season_year=season_year,
        template_id=template_id,
        offer=offer,
        development_promise=development_promise,
        assignment_plan=None if promote_to_nhl else assignment_plan,
    )
    if result.get("signed") and promote_to_nhl:
        league = getattr(getattr(session, "sim", None), "league", None)
        try:
            from services.contract_economy import promote_prospect_to_nhl

            promote_prospect_to_nhl(player, team, league, season_year, auto_elc=False)
        except Exception:
            pass
    return result
