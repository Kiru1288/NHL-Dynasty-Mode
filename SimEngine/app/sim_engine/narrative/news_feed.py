"""
News Feed — turns narrative events into media-style log output.

Consumes player journey events, storylines, and major league events;
produces formatted lines for the simulation log.
Does not alter simulation state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_narrative_section(
    journey_events: List[Dict[str, Any]],
    storyline_events: List[Dict[str, Any]],
    major_headlines: Optional[List[Any]] = None,
    awards: Optional[Dict[str, Any]] = None,
    playoff_champion_name: Optional[str] = None,
    year: Optional[int] = None,
    *,
    narrative_context: Optional[Dict[str, Any]] = None,
    league: Any = None,
) -> List[str]:
    """
    Produce log lines for the NARRATIVE EVENTS section.
    Returns list of strings (including section header and PLAYER JOURNEY:/STORYLINE:/NEWS: lines).
    """
    lines: List[str] = []
    content_added = bool(journey_events or storyline_events or awards or playoff_champion_name)
    lines.append("")
    lines.append("==============================")
    lines.append("NARRATIVE EVENTS")
    lines.append("==============================")
    lines.append("")

    nc = narrative_context if narrative_context is not None else {}
    et = nc.get("narrative_effect_trace_lines") or []
    if et:
        content_added = True
        lines.append("------------------------------")
        lines.append("EFFECT TRACE (progression / performance hooks)")
        lines.append("------------------------------")
        for ln in et[:26]:
            lines.append(f"  {ln}")
        lines.append("")

    act = nc.get("narrative_activation_lines") or []
    if act:
        content_added = True
        lines.append("------------------------------")
        lines.append("NARRATIVE ACTIVATIONS")
        lines.append("------------------------------")
        for ln in act[:22]:
            lines.append(f"  {ln}")
        lines.append("")

    resl = nc.get("narrative_resolution_lines") or []
    if resl:
        content_added = True
        lines.append("------------------------------")
        lines.append("RESOLUTION")
        lines.append("------------------------------")
        for ln in resl[:22]:
            lines.append(f"  {ln}")
        lines.append("")

    for ev in journey_events:
        msg = ev.get("message") or ""
        if msg:
            lines.append("PLAYER JOURNEY:")
            lines.append(msg)
            lines.append("")

    for ev in storyline_events:
        msg = ev.get("message") or ""
        if msg:
            lines.append("STORYLINE:")
            lines.append(msg)
            lines.append("")

    # Season-long digest: connected arcs (data-driven, not random copy)
    if narrative_context is not None and league is not None and year is not None:
        try:
            from app.sim_engine.narrative.player_journeys import build_player_journey_digest_lines

            digest = build_player_journey_digest_lines(narrative_context, league, int(year), max_lines=10)
        except Exception:
            digest = []
        if digest:
            content_added = True
            lines.append("------------------------------")
            lines.append("PLAYER JOURNEYS (season digest)")
            lines.append("------------------------------")
            lines.append("")
            for para in digest:
                lines.append(f"  • {para}")
            lines.append("")

        major_types = {
            "dynasty",
            "collapse",
            "rebuild",
            "rivalry",
            "historic_season",
            "redemption_arc",
            "contender_window",
        }
        st_digest: List[str] = []
        for ev in storyline_events:
            st = str(ev.get("storyline_type") or "")
            if st not in major_types:
                continue
            m = ev.get("message") or ""
            if m and m not in st_digest:
                st_digest.append(str(m))
            if len(st_digest) >= 8:
                break
        if st_digest:
            content_added = True
            lines.append("------------------------------")
            lines.append("KEY STORYLINES")
            lines.append("------------------------------")
            lines.append("")
            for para in st_digest:
                lines.append(f"  • {para}")
            lines.append("")

    # News: from awards and cup
    if awards:
        for award_name, award in awards.items():
            winner = getattr(award, "winner_name", None) or (award.get("winner_name") if isinstance(award, dict) else None)
            if winner and award_name and "Stanley" not in str(award_name):
                lines.append("NEWS:")
                lines.append(f"{winner} has captured the {award_name} after a dominant season.")
                lines.append("")
    if playoff_champion_name and year:
        lines.append("NEWS:")
        lines.append(f"The {playoff_champion_name} are celebrating their Stanley Cup championship.")
        lines.append("")

    # Major headlines as NEWS (trades, big signings)
    if major_headlines:
        for e in major_headlines[:5]:
            headline = getattr(e, "headline", None) or (e.get("headline") if isinstance(e, dict) else None)
            typ = (getattr(e, "type", "") or (e.get("type") if isinstance(e, dict) else "")) or ""
            if headline and typ.upper() in ("TRADE", "SIGNING") and getattr(e, "impact_score", 0.6) >= 0.55:
                lines.append("NEWS:")
                lines.append(str(headline))
                lines.append("")

    if not content_added and not (awards or playoff_champion_name) and not major_headlines:
        lines.append("(No narrative events this season.)")
        lines.append("")

    return lines
