"""
Draft rights engine: league-specific NHL rights independent of active club affiliation.

Drafting transfers NHL rights only. The player keeps current_team_id / current_league_id
on his junior, NCAA, or European club while nhl_rights_team_id records organizational claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

RIGHTS_STATUS = (
    "drafted_unsigned",
    "exclusive_rights",
    "indefinite_european_rights",
    "college_rights",
    "rights_expiring",
    "rights_relinquished",
    "draft_reentry",
    "unrestricted_free_agent",
    "signed",
)

RIGHTS_TYPE = (
    "chl_exclusive",
    "ncaa_college",
    "european_exclusive",
    "european_indefinite",
    "ushl_junior",
    "other_amateur",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _league_code(block: Optional[Dict], entry: Optional[Dict] = None) -> str:
    code = str((block or {}).get("league_code") or (entry or {}).get("league_code") or "").upper()
    if not code:
        code = str((entry or {}).get("league") or (entry or {}).get("league_name") or "").upper()
    return code


def development_path_for(entry: Optional[Dict] = None, block: Optional[Dict] = None) -> str:
    code = _league_code(block, entry)
    if code.startswith("NCAA") or "NCAA" in code:
        return "NCAA"
    if code.startswith("EU_") or any(x in code for x in ("SHL", "LIIGA", "DEL", "KHL", "CZE", "SVK", "SUI")):
        return "Europe"
    if code.startswith("CHL") or code in ("OHL", "WHL", "QMJHL", "USHL"):
        return "Junior"
    return "Junior"


def infer_rights_type(dev_path: str, league_code: str = "") -> str:
    code = (league_code or "").upper()
    path = (dev_path or "").upper()
    if path == "NCAA" or code.startswith("NCAA") or "NCAA" in code:
        return "ncaa_college"
    if path == "EUROPE" or code.startswith("EU_"):
        # First-year exclusive European claim; later passes may convert to indefinite.
        return "european_exclusive"
    if code == "USHL":
        return "ushl_junior"
    if path == "JUNIOR" or code.startswith("CHL") or code in ("OHL", "WHL", "QMJHL"):
        return "chl_exclusive"
    return "other_amateur"


def _rights_status_for_type(rights_type: str) -> str:
    if rights_type == "ncaa_college":
        return "college_rights"
    if rights_type in ("european_exclusive", "european_indefinite"):
        return "indefinite_european_rights" if rights_type == "european_indefinite" else "exclusive_rights"
    if rights_type in ("chl_exclusive", "ushl_junior"):
        return "exclusive_rights"
    return "drafted_unsigned"


def compute_rights_window(
    *,
    rights_type: str,
    draft_year: int,
    player_age: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Approximate CBA-style windows (game abstraction, not literal CBA counsel).

    CHL-style: exclusive claim through draft_year + 2 (June 1 signing deadline style).
    NCAA: retained while enrolled; model expiry at draft_year + 4 (typical graduation window).
    European exclusive: draft_year + 2, then may become indefinite or re-entry.
    """
    dy = int(draft_year)
    age = int(player_age or 18)
    if rights_type == "ncaa_college":
        expiry_year = dy + max(2, 4 - max(0, age - 18))
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-08-15",
            "rights_expiry_date": f"{expiry_year}-08-15",
            "rights_is_exclusive": False,
            "rights_can_reenter_draft": True,
        }
    if rights_type in ("chl_exclusive", "ushl_junior"):
        expiry_year = dy + 2
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-06-01",
            "rights_expiry_date": f"{expiry_year}-06-01",
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": True,
        }
    if rights_type == "european_exclusive":
        expiry_year = dy + 2
        return {
            "rights_expiry_year": expiry_year,
            "rights_signing_deadline": f"{expiry_year}-06-15",
            "rights_expiry_date": f"{expiry_year}-06-15",
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": True,
        }
    if rights_type == "european_indefinite":
        return {
            "rights_expiry_year": None,
            "rights_signing_deadline": None,
            "rights_expiry_date": None,
            "rights_is_exclusive": True,
            "rights_can_reenter_draft": False,
        }
    expiry_year = dy + 2
    return {
        "rights_expiry_year": expiry_year,
        "rights_signing_deadline": f"{expiry_year}-06-01",
        "rights_expiry_date": f"{expiry_year}-06-01",
        "rights_is_exclusive": True,
        "rights_can_reenter_draft": True,
    }


def build_draft_rights_fields(
    *,
    team_id: str,
    draft_year: int,
    block: Optional[Dict] = None,
    entry: Optional[Dict] = None,
    player: Any = None,
    acquired_date: Optional[str] = None,
) -> Dict[str, Any]:
    code = _league_code(block, entry)
    dev_path = development_path_for(entry, block)
    rights_type = infer_rights_type(dev_path, code)
    age = None
    if player is not None:
        try:
            age = int(getattr(player, "age", None) or (entry or {}).get("age") or 18)
        except Exception:
            age = int((entry or {}).get("age") or 18)
    else:
        age = int((entry or {}).get("age") or 18)
    window = compute_rights_window(rights_type=rights_type, draft_year=draft_year, player_age=age)
    status = _rights_status_for_type(rights_type)
    return {
        "rights_type": rights_type,
        "rights_acquired_date": acquired_date or _now_iso(),
        "rights_expiry_date": window.get("rights_expiry_date"),
        "rights_expiry_year": window.get("rights_expiry_year"),
        "rights_signing_deadline": window.get("rights_signing_deadline"),
        "rights_team_id": str(team_id),
        "rights_status": status,
        "rights_source_league": code or dev_path,
        "rights_is_exclusive": bool(window.get("rights_is_exclusive")),
        "rights_can_reenter_draft": bool(window.get("rights_can_reenter_draft")),
        "development_path": dev_path,
        "nhl_rights_team_id": str(team_id),
        "organizational_status": "unsigned_drafted",
        "signed_status": "unsigned",
    }


def affiliation_snapshot(
    player: Any,
    block: Optional[Dict] = None,
    tm: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Capture active club/league without overwriting NHL rights ownership."""
    current_team_id = None
    if tm is not None:
        current_team_id = str(tm.get("team_id") or tm.get("id") or tm.get("name") or "")
    if not current_team_id:
        current_team_id = str(
            getattr(player, "current_team_id", None)
            or getattr(player, "team_id", None)
            or getattr(player, "dev_team_id", None)
            or ""
        )
    current_league_id = None
    if block is not None:
        current_league_id = str(block.get("league_code") or block.get("id") or block.get("name") or "")
    if not current_league_id:
        current_league_id = str(
            getattr(player, "current_league_id", None)
            or getattr(player, "league_id", None)
            or getattr(player, "dev_league_id", None)
            or ""
        )
    return {
        "current_team_id": current_team_id or None,
        "current_league_id": current_league_id or None,
    }


def apply_draft_rights(
    player: Any,
    *,
    nhl_team_id: str,
    draft_year: int,
    pick_meta: Optional[Dict[str, Any]] = None,
    block: Optional[Dict] = None,
    tm: Optional[Dict] = None,
    entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assign NHL draft rights while leaving the player on his development club.

    Does NOT set team_id to the NHL organization. Does NOT remove from junior/NCAA/EU roster.
    """
    pick_meta = pick_meta or {}
    entry = entry or {}
    aff = affiliation_snapshot(player, block, tm)
    rights = build_draft_rights_fields(
        team_id=nhl_team_id,
        draft_year=draft_year,
        block=block,
        entry=entry,
        player=player,
    )
    overall = int(pick_meta.get("overall_pick") or 1)

    setattr(player, "drafted", True)
    setattr(player, "drafted_by", str(nhl_team_id))
    setattr(player, "draft_team_id", str(nhl_team_id))
    setattr(player, "draft_year", int(draft_year))
    setattr(player, "draft_round", int(pick_meta.get("round") or 1))
    setattr(player, "draft_pick_number", int(pick_meta.get("pick_in_round") or 1))
    setattr(player, "draft_overall_pick", overall)
    pick_id = str(pick_meta.get("pick_id") or "").strip()
    if pick_id:
        setattr(player, "draft_pick_id", pick_id)
    orig_pick_team = str(
        pick_meta.get("original_owner_team_id")
        or pick_meta.get("original_team_id")
        or nhl_team_id
    )
    setattr(player, "draft_pick_original_team_id", orig_pick_team)
    setattr(
        player,
        "draft_pick_was_traded",
        bool(pick_meta.get("is_traded")) or (orig_pick_team != str(nhl_team_id)),
    )

    setattr(player, "current_team_id", aff["current_team_id"])
    setattr(player, "current_league_id", aff["current_league_id"])
    # Keep active club id as the development team — never the NHL org.
    if aff["current_team_id"]:
        setattr(player, "team_id", aff["current_team_id"])
    setattr(player, "dev_team_id", aff["current_team_id"])
    setattr(player, "dev_league_id", aff["current_league_id"])

    for k, v in rights.items():
        setattr(player, k, v)

    setattr(player, "prospect_status", "org_prospect")
    setattr(player, "post_draft_league", rights["development_path"])
    setattr(player, "entry_level_contract_eligible", True)
    setattr(player, "status", "prospect")
    setattr(player, "elc_slide_eligible", True)
    setattr(player, "elc_slide_years_remaining", 1)
    setattr(player, "nhl_games_played_this_season", int(getattr(player, "nhl_games_played_this_season", 0) or 0))
    setattr(player, "contract_burned", False)

    return {**aff, **rights}


def available_rights_actions(player: Any) -> List[Dict[str, Any]]:
    """Legal decision options for Prospect Rights — never invent illegal paths."""
    signed = str(getattr(player, "signed_status", "unsigned") or "unsigned").lower()
    status = str(getattr(player, "rights_status", "") or "").lower()
    path = str(getattr(player, "development_path", "") or getattr(player, "post_draft_league", "") or "").upper()
    rights_type = str(getattr(player, "rights_type", "") or "").lower()
    age = _safe_player_age(player)
    env = development_environment_assessment(player)
    actions: List[Dict[str, Any]] = []
    if signed == "signed" or status == "signed":
        return [{"id": "already_signed", "label": "Already signed", "enabled": False}]
    if status in ("rights_relinquished", "unrestricted_free_agent", "draft_reentry"):
        return [{"id": "no_rights", "label": "No active rights", "enabled": False}]

    elc_ok = bool(getattr(player, "entry_level_contract_eligible", True))
    if elc_ok:
        actions.append({
            "id": "sign_elc",
            "label": "Sign to ELC",
            "enabled": True,
            "starts_elc": True,
            "uses_contract_slot": True,
            "can_slide": bool(getattr(player, "elc_slide_eligible", False)),
        })
    actions.append({"id": "keep_unsigned", "label": "Keep unsigned", "enabled": True})
    if "NCAA" in path or rights_type == "ncaa_college":
        actions.append({"id": "keep_college", "label": "Keep in college", "enabled": True})
    elif "EUROPE" in path or rights_type.startswith("european"):
        actions.append({"id": "keep_europe", "label": "Keep in Europe", "enabled": True})
    else:
        actions.append({"id": "return_junior", "label": "Return to junior", "enabled": True})
    # AHL is a professional roster — an unsigned prospect cannot be assigned there
    # until an ELC is in place. Surface the option (so the UI can show it, disabled)
    # but never let it fire until the player is actually signed.
    actions.append({
        "id": "assign_ahl",
        "label": "Assign to AHL",
        "enabled": False,
        "blocked_reason": "Sign an ELC first — AHL assignment requires a contract",
    })
    actions.append({"id": "invite_camp", "label": "Invite to training camp", "enabled": True})
    actions.append({"id": "delay", "label": "Delay decision", "enabled": True})
    expiry = getattr(player, "rights_expiry_year", None)
    if expiry is not None:
        actions.append({
            "id": "allow_expire",
            "label": "Allow rights to expire",
            "enabled": True,
            "warning": "Relinquishes organizational claim at deadline",
        })
    for action in actions:
        tradeoffs = _rights_action_tradeoffs(player, action, env)
        action["pros"] = tradeoffs["pros"]
        action["cons"] = tradeoffs["cons"]
        if tradeoffs.get("summary"):
            action["summary"] = tradeoffs["summary"]
    return actions


def _rights_action_tradeoffs(player: Any, action: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    """Pros/cons from existing rights, environment, and eligibility signals only."""
    aid = str(action.get("id") or "")
    path = str(getattr(player, "development_path", "") or getattr(player, "post_draft_league", "") or "")
    eta = getattr(player, "nhl_eta", None)
    grade = str(env.get("grade") or "")
    env_reasons = list(env.get("reasons") or [])
    slide = bool(getattr(player, "elc_slide_eligible", False))
    slide_years = getattr(player, "elc_slide_years_remaining", None)
    expiry = getattr(player, "rights_expiry_year", None)
    pros: List[str] = []
    cons: List[str] = []
    summary = None

    if aid == "sign_elc":
        pros.append("Locks exclusive NHL rights into an ELC")
        if action.get("can_slide") or slide:
            pros.append(
                f"ELC can slide{f' - {slide_years}y left' if slide_years is not None else ''}"
            )
        if grade in ("ideal", "good"):
            pros.append(f"Development environment graded {grade}")
        if env_reasons:
            pros.extend(env_reasons[:2])
        cons.append("Uses one of 50 organization contract slots")
        if grade in ("poor", "risky"):
            cons.append(f"Environment graded {grade} — development risk")
        if eta is not None and int(eta) >= 4:
            cons.append(f"Long runway (ETA {eta}y) before NHL impact")
        summary = "Turn the prospect pro on an entry-level deal"
    elif aid in ("keep_unsigned", "delay"):
        pros.append("Preserves a contract slot for other signings")
        if path:
            pros.append(f"Continues current path ({path})")
        if expiry is not None:
            cons.append(f"Rights still tick toward {expiry}")
        cons.append("No roster control until signed")
        summary = "Leave unsigned and revisit later"
    elif aid == "keep_college":
        pros.append("NCAA development continues without burning an ELC year")
        pros.append("Preserves a contract slot")
        if expiry is not None:
            cons.append(f"College rights still expire {expiry}")
        cons.append("No immediate pro assignment control")
        summary = "Keep the prospect in school"
    elif aid == "keep_europe":
        pros.append("Keeps European development path intact")
        pros.append("Preserves a contract slot")
        if grade in ("poor", "risky"):
            cons.append(f"Environment graded {grade}")
        if expiry is not None:
            cons.append(f"European exclusive window tracked through {expiry}")
        summary = "Leave the prospect overseas"
    elif aid == "return_junior":
        pros.append("Age-appropriate junior minutes when eligible")
        pros.append("Preserves a contract slot")
        age = _safe_player_age(player)
        if age >= 21:
            cons.append("Older junior return can stall growth")
        if env_reasons:
            cons.extend([r for r in env_reasons if "outgrown" in str(r).lower()][:1])
        summary = "Send the prospect back to junior"
    elif aid == "assign_ahl":
        if action.get("enabled"):
            pros.append("Pro introduction against men")
            if grade in ("ideal", "good"):
                pros.append(f"Environment supports AHL path ({grade})")
            cons.append("Requires AHL eligibility under junior-return rules")
            if eta is not None and int(eta) >= 3:
                cons.append("Still a multi-year NHL project")
        else:
            cons.append(action.get("blocked_reason") or "Not currently AHL-eligible")
        summary = "Assign to the AHL affiliate"
    elif aid == "invite_camp":
        pros.append("Evaluation look without a permanent roster burn")
        pros.append("Can still keep junior/college/Europe path afterward")
        cons.append("Does not replace a signed ELC for long-term control")
        summary = "Invite to training camp for a look"
    elif aid == "allow_expire":
        pros.append("Frees organizational attention and slot pressure")
        cons.append(action.get("warning") or "Relinquishes organizational claim at deadline")
        cons.append("Cannot reclaim exclusive rights after expiry")
        summary = "Let rights lapse"
    else:
        if action.get("blocked_reason"):
            cons.append(action["blocked_reason"])
        if action.get("warning"):
            cons.append(action["warning"])

    return {
        "pros": pros[:4],
        "cons": cons[:4],
        "summary": summary,
    }


def rights_card_payload(player: Any, *, team: Any = None, season_year: Optional[int] = None) -> Dict[str, Any]:
    expiry = getattr(player, "rights_expiry_year", None)
    path = str(getattr(player, "development_path", "") or getattr(player, "post_draft_league", "") or "")
    status = str(getattr(player, "rights_status", "") or "")
    signed = str(getattr(player, "signed_status", "unsigned") or "unsigned")
    env = development_environment_assessment(player)
    actions = available_rights_actions(player)
    recommended = next((a for a in actions if a.get("enabled") and a.get("id") != "allow_expire"), None)

    # Attach live ELC acceptance signals onto the Sign to ELC action when possible.
    elc_eval = None
    if team is not None and season_year is not None:
        try:
            from services.draft_signing_engine import evaluate_elc_signing_decision

            elc_eval = evaluate_elc_signing_decision(player, team, season_year=int(season_year))
            for action in actions:
                if action.get("id") != "sign_elc":
                    continue
                reasons = list(elc_eval.get("reasons") or [])
                if elc_eval.get("accepted"):
                    action.setdefault("pros", [])
                    action["pros"] = list(dict.fromkeys((action.get("pros") or []) + reasons[:3]))[:5]
                    action["acceptance_outlook"] = "Likely to accept"
                else:
                    action.setdefault("cons", [])
                    action["cons"] = list(dict.fromkeys((action.get("cons") or []) + reasons[:3]))[:5]
                    action["acceptance_outlook"] = "May decline"
                if elc_eval.get("reason"):
                    action["evaluation_reason"] = elc_eval.get("reason")
                break
        except Exception:
            elc_eval = None

    decision_status = str(getattr(player, "rights_decision_status", None) or (
        "signed" if signed == "signed" else "pending"
    ))

    ovr = _safe_player_ovr(player)
    out = {
        "rights_through": expiry,
        "rights_status": status,
        "rights_type": getattr(player, "rights_type", None),
        "rights_signing_deadline": getattr(player, "rights_signing_deadline", None),
        "returning_to": path,
        "expected_role": getattr(player, "expected_role", None) or "Org prospect",
        "elc_decision": "Signed" if signed == "signed" else "Unsigned",
        "eta": getattr(player, "nhl_eta", None),
        "organizational_status": getattr(player, "organizational_status", None),
        "current_league_id": getattr(player, "current_league_id", None),
        "current_team_id": getattr(player, "current_team_id", None),
        "nhl_rights_team_id": getattr(player, "nhl_rights_team_id", None)
        or getattr(player, "rights_team_id", None),
        "elc_slide_eligible": bool(getattr(player, "elc_slide_eligible", False)),
        "elc_slide_years_remaining": getattr(player, "elc_slide_years_remaining", None),
        "nhl_games_played_this_season": getattr(player, "nhl_games_played_this_season", 0),
        "contract_burned": bool(getattr(player, "contract_burned", False)),
        "entry_level_contract_eligible": bool(getattr(player, "entry_level_contract_eligible", True)),
        "available_actions": actions,
        "development_environment": env,
        "recommended_action": (recommended or {}).get("id"),
        "recommended_label": (recommended or {}).get("label"),
        "path_visual": _path_visual(path),
        "elc_evaluation": elc_eval,
        "decision_status": decision_status,
        "overall": int(ovr) if ovr else None,
        "nhl_readiness": getattr(player, "nhl_readiness", None),
        "nhl_eta_label": getattr(player, "nhl_eta_label", None),
        "draft_year": getattr(player, "draft_year", None),
        "draft_overall_pick": getattr(player, "draft_overall_pick", None)
        or getattr(player, "overall_pick", None),
        "draft_pick_id": getattr(player, "draft_pick_id", None),
        "draft_pick_original_team_id": getattr(player, "draft_pick_original_team_id", None),
        "draft_pick_was_traded": bool(getattr(player, "draft_pick_was_traded", False)),
    }
    # Structured ELC negotiation payload (authoritative — do not invent in UI)
    if season_year is not None:
        try:
            from services.elc_offer_engine import legal_elc_terms, list_offer_templates

            legal = legal_elc_terms(player, int(season_year))
            templates = list_offer_templates(player, int(season_year))
            out["legal_elc_terms"] = legal
            out["offer_templates"] = [
                {
                    "template_id": t["template_id"],
                    "label": t["label"],
                    "summary": t["summary"],
                    "term_years": t["term_years"],
                    "aav_display": t["aav_display"],
                    "signing_bonus_display": t["signing_bonus_display"],
                    "schedule_a_display": t["schedule_a_display"],
                    "schedule_b_display": t["schedule_b_display"],
                    "slide_eligible": t["slide_eligible"],
                }
                for t in templates
            ]
            if team is not None and templates:
                from services.elc_offer_engine import evaluate_offer_acceptance, build_offer_from_template

                rec = build_offer_from_template(
                    player, season_year=int(season_year), template_id="standard_elc"
                )
                acc = evaluate_offer_acceptance(player, team, rec, season_year=int(season_year))
                out["elc_acceptance_summary"] = {
                    "acceptance_pct": acc.get("acceptance_pct"),
                    "outlook_label": acc.get("outlook_label"),
                    "decision": acc.get("decision"),
                    "main_positive": acc.get("main_positive"),
                    "main_concern": acc.get("main_concern"),
                    "agent_wants": acc.get("agent_wants"),
                }
        except Exception:
            pass
    return out


def _detach_from_development_leagues(league: Any, player: Any) -> None:
    for block in getattr(league, "development_leagues", None) or []:
        if not isinstance(block, dict):
            continue
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            players = tm.get("players")
            if isinstance(players, list) and player in players:
                try:
                    players.remove(player)
                except ValueError:
                    pass


def move_prospect_to_ahl(league: Any, player: Any, team: Any) -> bool:
    """Physically place a prospect on the club's AHL roster.

    Setting development_path alone leaves the player on his junior club and
    invisible to every roster/trade surface, which all read the roster lists.
    """
    if team is None:
        return False
    ahl = list(getattr(team, "ahl_roster", None) or [])
    pid = str(getattr(player, "id", "") or "")
    if not any(str(getattr(p, "id", "") or "") == pid for p in ahl):
        ahl.append(player)
        team.ahl_roster = ahl
    _detach_from_development_leagues(league, player)
    for attr, val in (
        ("roster_location", "ahl"),
        ("in_minors", True),
        ("current_league_id", "AHL"),
        ("team_id", str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")),
    ):
        try:
            setattr(player, attr, val)
        except Exception:
            pass
    try:
        from services.draft_player_registry import register_player

        register_player(league, player)
    except Exception:
        pass
    return True


def remove_prospect_from_ahl(team: Any, player: Any) -> None:
    """Undo an AHL assignment when the club reroutes the prospect elsewhere."""
    if team is None:
        return
    ahl = list(getattr(team, "ahl_roster", None) or [])
    pid = str(getattr(player, "id", "") or "")
    kept = [p for p in ahl if str(getattr(p, "id", "") or "") != pid]
    if len(kept) != len(ahl):
        team.ahl_roster = kept
        try:
            setattr(player, "roster_location", None)
            setattr(player, "in_minors", False)
        except Exception:
            pass


def apply_prospect_rights_decision(
    session: Any,
    player: Any,
    team: Any,
    action_id: str,
    *,
    season_year: int,
) -> Dict[str, Any]:
    """Persist a Prospect Rights stage decision onto the live player/org."""
    aid = str(action_id or "").strip()
    legal = {str(a.get("id")): a for a in available_rights_actions(player)}
    action = legal.get(aid)
    if not action:
        return {"ok": False, "reason": f"Action not available: {aid}"}
    if action.get("enabled") is False:
        return {
            "ok": False,
            "reason": action.get("blocked_reason") or "Action disabled",
            "action": action,
        }

    league = getattr(getattr(session, "sim", None), "league", None)

    if aid == "sign_elc":
        from services.draft_signing_engine import attempt_sign_elc_with_decision

        result = attempt_sign_elc_with_decision(
            session,
            player,
            team,
            season_year=int(season_year),
            promote_to_nhl=False,
        )
        if result.get("ok"):
            try:
                setattr(player, "rights_decision_status", "signed_elc")
                setattr(player, "rights_decision_action", aid)
            except Exception:
                pass
        return result

    # Non-signing path decisions — mutate existing development / rights fields only.
    if aid == "keep_college":
        setattr(player, "development_path", "NCAA")
        setattr(player, "post_draft_league", "NCAA")
        remove_prospect_from_ahl(team, player)
    elif aid == "keep_europe":
        setattr(player, "development_path", "EUROPE")
        setattr(player, "post_draft_league", "EUROPE")
        remove_prospect_from_ahl(team, player)
    elif aid == "return_junior":
        # Keep junior family path when already junior; otherwise mark JUNIOR.
        cur = str(getattr(player, "development_path", "") or "")
        if "JUNIOR" not in cur.upper() and "CHL" not in cur.upper() and "USHL" not in cur.upper():
            setattr(player, "development_path", "JUNIOR")
            setattr(player, "post_draft_league", "JUNIOR")
        remove_prospect_from_ahl(team, player)
    elif aid == "assign_ahl":
        # Belt-and-suspenders: available_rights_actions() already disables this for
        # unsigned prospects, but never physically move an unsigned player onto a
        # professional AHL roster even if this handler is reached some other way.
        signed_now = str(getattr(player, "signed_status", "unsigned") or "unsigned").lower() == "signed"
        if not signed_now:
            return {
                "ok": False,
                "reason": "Sign an ELC first — AHL assignment requires a contract",
            }
        setattr(player, "development_path", "AHL")
        setattr(player, "post_draft_league", "AHL")
        move_prospect_to_ahl(league, player, team)
    elif aid == "invite_camp":
        try:
            setattr(player, "training_camp_invite", True)
        except Exception:
            pass
    elif aid == "allow_expire":
        setattr(player, "rights_status", "rights_relinquished")
        setattr(player, "organizational_status", "rights_relinquished")
        try:
            setattr(player, "nhl_rights_team_id", None)
            setattr(player, "rights_team_id", None)
            setattr(player, "rights_is_exclusive", False)
        except Exception:
            pass
        # Drop from reserve / pool when relinquishing claim.
        try:
            from services.contract_economy import remove_from_reserve_list, _player_id

            remove_from_reserve_list(team, _player_id(player))
            pool = list(getattr(team, "prospect_pool", None) or [])
            if player in pool:
                pool.remove(player)
                team.prospect_pool = pool
        except Exception:
            pass
        remove_prospect_from_ahl(team, player)
    elif aid in ("keep_unsigned", "delay"):
        pass  # explicit no-op path decision
    else:
        return {"ok": False, "reason": f"Unhandled rights action: {aid}"}

    try:
        setattr(player, "rights_decision_status", aid)
        setattr(player, "rights_decision_action", aid)
        setattr(player, "rights_decision_season", int(season_year))
    except Exception:
        pass

    # Mirror decision onto reserve list row when present.
    try:
        from services.contract_economy import _player_id

        pid = _player_id(player)
        reserve = list(getattr(team, "reserve_list", None) or [])
        changed = False
        for entry in reserve:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("player_id") or "") != pid:
                continue
            entry["decision_status"] = aid
            entry["rights_decision_action"] = aid
            if aid in ("keep_college", "keep_europe", "return_junior", "assign_ahl"):
                entry["current_league_id"] = getattr(player, "development_path", None)
            changed = True
        if changed:
            team.reserve_list = reserve
    except Exception:
        pass

    _ = league  # reserved for future league-wide sync
    return {
        "ok": True,
        "action_id": aid,
        "player_id": str(getattr(player, "id", "") or ""),
        "decision_status": aid,
        "development_path": getattr(player, "development_path", None),
        "rights_status": getattr(player, "rights_status", None),
    }


def _safe_player_ovr(player: Any) -> float:
    """Resolve overall when it may be a property, method, or missing."""
    for key in ("overall", "ovr", "current_ovr", "true_ovr"):
        raw = getattr(player, key, None)
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, (int, float)):
            try:
                raw = raw()
            except TypeError:
                continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _safe_player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    for raw in (
        getattr(ident, "age", None) if ident is not None else None,
        getattr(player, "age", None),
    ):
        if raw is None:
            continue
        if callable(raw) and not isinstance(raw, (int, float)):
            try:
                raw = raw()
            except TypeError:
                continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 18


def development_environment_assessment(player: Any) -> Dict[str, Any]:
    """Public development-environment grade from observable signals only."""
    path = str(getattr(player, "development_path", "") or "").upper()
    age = _safe_player_age(player)
    ovr = _safe_player_ovr(player)
    reasons: List[str] = []
    grade = "acceptable"
    if ovr >= 70 and age <= 21:
        grade = "ideal"
        reasons.append("Ready for meaningful pro minutes")
    elif ovr >= 64:
        grade = "good"
        reasons.append("Tools project a usable role with ice time")
    elif ovr > 0 and ovr < 55 and "NCAA" not in path and "JUNIOR" not in path:
        grade = "risky"
        reasons.append("Current ability may be overmatched")
    elif age <= 18 and ("JUNIOR" in path or "NCAA" in path):
        grade = "good"
        reasons.append("Age-appropriate development league")
    elif age >= 21 and ("JUNIOR" in path):
        grade = "poor"
        reasons.append("May have outgrown junior competition")
    else:
        reasons.append("Standard developmental placement")
    return {"grade": grade, "reasons": reasons[:3], "path": path or None}


def _path_visual(path: str) -> List[str]:
    p = (path or "").upper()
    if "NCAA" in p:
        return ["College", "AHL", "NHL"]
    if "EUROPE" in p:
        return ["Europe", "AHL", "NHL"]
    return ["Junior", "AHL", "NHL"]


def _iter_org_prospects(league: Any) -> List[Tuple[Any, Any]]:
    out: List[Tuple[Any, Any]] = []
    for team in getattr(league, "teams", None) or []:
        for p in list(getattr(team, "prospect_pool", None) or []):
            out.append((team, p))
        for entry in list(getattr(team, "reserve_list", None) or []):
            if not isinstance(entry, dict):
                continue
            # Prefer live player from org pools; registry resolution happens upstream.
            ref = entry.get("player_ref")
            if ref is not None and all(ref is not x for _, x in out):
                out.append((team, ref))
    return out


def process_draft_rights_deadlines(
    session: Any,
    league: Any,
    season_year: int,
) -> Dict[str, Any]:
    """
    Evaluate CHL/NCAA/European rights expiry, re-entry eligibility, relinquishment, and UFA.
    """
    sy = int(season_year)
    notifications: List[Dict[str, Any]] = []
    expired: List[Dict[str, Any]] = []
    reentry: List[Dict[str, Any]] = []
    extended_eu: List[Dict[str, Any]] = []
    reviewed = 0

    from services.draft_player_registry import get_player, ensure_players_by_id

    ensure_players_by_id(league)

    for team in getattr(league, "teams", None) or []:
        tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
        pool = list(getattr(team, "prospect_pool", None) or [])
        reserve = list(getattr(team, "reserve_list", None) or [])
        reserve_by_id = {str(e.get("player_id")): e for e in reserve if isinstance(e, dict)}

        for player in list(pool):
            pid = str(getattr(player, "id", "") or "")
            if not pid:
                continue
            if str(getattr(player, "signed_status", "") or "").lower() == "signed":
                setattr(player, "rights_status", "signed")
                continue
            if str(getattr(player, "rights_status", "") or "") in (
                "rights_relinquished",
                "unrestricted_free_agent",
                "draft_reentry",
            ):
                continue

            reviewed += 1
            rtype = str(getattr(player, "rights_type", "") or "")
            expiry_year = getattr(player, "rights_expiry_year", None)
            deadline = str(getattr(player, "rights_signing_deadline", "") or "")

            if expiry_year is not None and int(expiry_year) - sy <= 1 and int(expiry_year) >= sy:
                setattr(player, "rights_status", "rights_expiring")
                notifications.append({
                    "type": "signing_deadline",
                    "team_id": tid,
                    "player_id": pid,
                    "player_name": str(getattr(player, "name", "") or pid),
                    "rights_expiry_year": int(expiry_year),
                    "rights_signing_deadline": deadline,
                    "message": f"Rights expire {deadline or expiry_year}",
                })

            if expiry_year is None or int(expiry_year) > sy:
                continue

            # Expired
            can_reenter = bool(getattr(player, "rights_can_reenter_draft", True))
            if rtype == "european_exclusive":
                # Convert to indefinite European exclusive claim rather than auto-UFA.
                setattr(player, "rights_type", "european_indefinite")
                setattr(player, "rights_status", "indefinite_european_rights")
                setattr(player, "rights_expiry_year", None)
                setattr(player, "rights_expiry_date", None)
                setattr(player, "rights_signing_deadline", None)
                setattr(player, "rights_can_reenter_draft", False)
                extended_eu.append({"player_id": pid, "team_id": tid})
                continue

            if can_reenter:
                setattr(player, "rights_status", "draft_reentry")
                setattr(player, "organizational_status", "draft_reentry")
                reentry.append({"player_id": pid, "team_id": tid, "name": getattr(player, "name", "")})
            else:
                setattr(player, "rights_status", "unrestricted_free_agent")
                setattr(player, "organizational_status", "ufa")

            setattr(player, "nhl_rights_team_id", None)
            setattr(player, "rights_team_id", None)
            setattr(player, "rights_is_exclusive", False)
            setattr(player, "prospect_status", "free_agent")

            # Drop from org membership but leave active development affiliation intact.
            try:
                pool.remove(player)
                team.prospect_pool = pool
            except Exception:
                pass
            if pid in reserve_by_id:
                try:
                    team.reserve_list = [e for e in reserve if str(e.get("player_id")) != pid]
                except Exception:
                    pass

            expired.append({
                "player_id": pid,
                "former_team_id": tid,
                "outcome": "draft_reentry" if can_reenter else "unrestricted_free_agent",
            })
            notifications.append({
                "type": "rights_relinquished" if not can_reenter else "draft_reentry",
                "team_id": tid,
                "player_id": pid,
                "player_name": str(getattr(player, "name", "") or pid),
                "message": (
                    f"{getattr(player, 'name', pid)} eligible for draft re-entry"
                    if can_reenter
                    else f"{getattr(player, 'name', pid)} became an unrestricted free agent"
                ),
            })

            # Sync reserve row if still present elsewhere via registry
            _ = get_player(league, pid)

    payload = {
        "season_year": sy,
        "reviewed": reviewed,
        "expired": expired,
        "reentry_eligible": reentry,
        "european_indefinite": extended_eu,
        "notifications": notifications,
        "processed_at": _now_iso(),
    }
    try:
        session.draft_rights_review_payload = payload
    except Exception:
        pass
    return payload


def should_cpu_auto_sign_elc(
    player: Any,
    team: Any,
    *,
    season_year: int,
    league: Any = None,
) -> Tuple[bool, str]:
    """
    Automatic ELC only with a clear organizational reason — not blanket roster cleanup.
    """
    if str(getattr(player, "signed_status", "") or "").lower() == "signed":
        return False, "already_signed"
    if not bool(getattr(player, "entry_level_contract_eligible", False)):
        return False, "not_elc_eligible"

    expiry = getattr(player, "rights_expiry_year", None)
    status = str(getattr(player, "rights_status", "") or "")
    readiness_raw = getattr(player, "nhl_readiness", None) or getattr(player, "overall", None)
    if readiness_raw is None:
        # `ovr` is a method on real player entities — resolve it like the rest of
        # the codebase instead of calling float() on a bound method.
        ov = getattr(player, "ovr", None)
        if callable(ov):
            try:
                ov = ov()
            except Exception:
                ov = None
        readiness_raw = ov
    try:
        readiness = float(readiness_raw if readiness_raw is not None else 50)
    except (TypeError, ValueError):
        readiness = 50.0
    if readiness <= 1.5:  # 0–1 scale from ovr() → 0–99 scale
        readiness *= 99.0
    age = int(getattr(player, "age", 18) or 18)
    path = str(getattr(player, "development_path", "") or "")

    # Rights about to expire / marked expiring
    if status == "rights_expiring" or (expiry is not None and int(expiry) <= int(season_year) + 1):
        return True, "rights_expiration_risk"

    # Clearly NHL-ready and not in multi-year NCAA return
    if readiness >= 72 and path != "NCAA":
        return True, "prospect_readiness"

    # Older juniors / European pros ready to turn pro
    if age >= 21 and readiness >= 64 and path in ("Junior", "Europe"):
        return True, "age_and_readiness"

    # Cap / slots: do not auto-sign if org is near contract limit
    try:
        from services.contract_economy import validate_contract_slots

        slots = validate_contract_slots(team, league, additional=1)
        if not slots.get("ok"):
            return False, "contract_slots_full"
    except Exception:
        pass

    return False, "no_clear_reason"
