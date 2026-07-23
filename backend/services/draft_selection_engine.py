"""
Shared draft selection engine façade.

Franchise live draft and SimEngine universe draft both call into this module so
selection policy stays one place — orchestration may still differ.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.draft_board_engine import enrich_board_entry_with_team_scouting, select_best_prospect


def build_uncertain_team_board(
    available: List[Dict[str, Any]],
    *,
    team_id: str,
    scouting_quality: float = 60.0,
    draft_year: int = 2026,
    interview_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    interview_map = interview_map or {}
    rows = [
        enrich_board_entry_with_team_scouting(
            e,
            team_id=team_id,
            scouting_quality=scouting_quality,
            draft_year=draft_year,
            interview=interview_map.get(str(e.get("key") or e.get("prospect_id") or "")),
        )
        for e in available
    ]
    rows.sort(
        key=lambda r: (
            -float(r.get("scouting_confidence") or 0),
            int(r.get("team_board_rank_hint") or 999),
        )
    )
    for i, row in enumerate(rows):
        row["team_board_rank"] = i + 1
        # Do not expose true attributes on the board payload shape.
        row.pop("true_overall_hidden", None)
    return rows


def cpu_select_from_board(
    board_rows: List[Dict[str, Any]],
    *,
    overall_pick: int,
    philosophy: str,
    needs: Optional[List[Dict[str, Any]]] = None,
    noise_fn=None,
) -> Dict[str, Any]:
    return select_best_prospect(
        board_rows,
        overall_pick=overall_pick,
        philosophy=philosophy,
        needs=needs,
        noise_fn=noise_fn,
    )


def apply_universe_selection_rights(
    prospect: Any,
    *,
    nhl_team_id: str,
    draft_year: int,
    overall_pick: int,
    round_num: int = 1,
    pick_in_round: int = 1,
    league_code: str = "",
) -> Dict[str, Any]:
    """
    Shared rights assignment for SimEngine universe draft picks.
    Leaves active club affiliation alone when possible.
    """
    from services.draft_rights_engine import apply_draft_rights

    entry = {
        "league_code": league_code or getattr(prospect, "league_code", "") or "",
        "age": getattr(prospect, "age", 18),
    }
    return apply_draft_rights(
        prospect,
        nhl_team_id=str(nhl_team_id),
        draft_year=int(draft_year),
        pick_meta={
            "round": int(round_num),
            "pick_in_round": int(pick_in_round),
            "overall_pick": int(overall_pick),
        },
        block={"league_code": entry["league_code"]} if entry["league_code"] else None,
        tm=None,
        entry=entry,
    )
