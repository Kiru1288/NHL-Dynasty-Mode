"""Players the club did not re-sign leave the roster when free agency opens."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from services.contract_economy import release_unresigned_free_agents  # noqa: E402


def _player(pid, name, years, expiry, aav, age=30, pending=False):
    return SimpleNamespace(
        id=pid,
        name=name,
        age=age,
        retired=False,
        pending_july1_expiry=pending,
        user_signed=False,
        rights_status="UFA",
        position="C",
        contract={
            "years_remaining": years,
            "expiry_year": expiry,
            "aav_m": aav,
            "cap_hit_m": aav,
            "rights_status": "UFA",
            "pending_july1_expiry": pending,
        },
        cap_hit_m=aav,
        aav_m=aav,
    )


def _world(roster, season_year=2026, outcomes=None):
    team = SimpleNamespace(
        team_id="OTT",
        id="OTT",
        roster=list(roster),
        ahl_roster=[],
        echl_roster=[],
        rfa_rights=[],
    )
    league = SimpleNamespace(teams=[team], free_agents=[])
    session = SimpleNamespace(
        sim=SimpleNamespace(league=league, rng=None),
        season_calendar_year=season_year,
        july1_contracts_expired=True,
        user_team_id="OTT",
        resign_phase_outcomes=outcomes or {},
        phase="offseason",
        free_agency_open=True,
    )
    return session, team, league


def test_unsigned_ufa_leaves_roster_and_cap_even_if_july1_already_ran():
    unsigned = _player("walks", "Unsigned Vet", 1, 2027, 6.5, age=31, pending=False)
    stayed = _player("stays", "Signed Core", 4, 2030, 8.0, age=28)
    session, team, league = _world([unsigned, stayed])
    out = release_unresigned_free_agents(session)
    ids = [p.id for p in team.roster]
    assert ids == ["stays"]
    assert any(p.id == "walks" for p in league.free_agents)
    assert unsigned.cap_hit_m == 0
    assert out["cap_freed_m"] == 6.5
    assert out["user_released"][0]["name"] == "Unsigned Vet"
    assert "6.50M" in session.fa_cap_release_note["text"]


def test_let_go_on_the_desk_leaves_even_with_years_left_on_the_sheet():
    let_go = _player("gone", "Let Go", 3, 2029, 4.25, age=26)
    session, team, league = _world(
        [let_go],
        outcomes={
            "gone": {"phase_status": "released", "player_id": "gone"},
        },
    )
    release_unresigned_free_agents(session)
    assert [p.id for p in team.roster] == []
    on_market = any(p.id == "gone" for p in league.free_agents)
    on_rights = any(
        isinstance(row, dict) and str(row.get("player_id") or "") == "gone"
        for row in team.rfa_rights
    )
    assert on_market or on_rights
    assert let_go.cap_hit_m == 0


def test_accepted_resign_stays_on_the_roster():
    signed = _player("kept", "Re-Signed", 1, 2027, 5.0, age=29, pending=True)
    session, team, league = _world(
        [signed],
        outcomes={
            "kept": {"phase_status": "accepted", "player_id": "kept"},
        },
    )
    out = release_unresigned_free_agents(session)
    assert [p.id for p in team.roster] == ["kept"]
    assert league.free_agents == []
    assert out["released"] == 0
    assert signed.cap_hit_m == 5.0
