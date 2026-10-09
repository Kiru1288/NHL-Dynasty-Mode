"""Regression tests for the cap / contract / free-agency fixes (no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _yearly_row(name: str, pid: int, age: int, cells: str) -> str:
    return (
        "<tr>"
        f'<td><a href="https://www.spotrac.com/nhl/player/_/id/{pid}/x" class="link">{name}</a></td>'
        "<td>G</td>"
        f"<td>{age}</td>"
        f"{cells}"
        "</tr>"
    )


def test_minor_table_contracts_are_parsed():
    from services.real_nhl_contracts import _parse_yearly_team_html

    nhl = _yearly_row(
        "Linus Ullmark",
        1,
        33,
        '<td data-sort="8250000">$8,250,000</td>' * 3,
    )
    minor = _yearly_row(
        "Jackson Parsons",
        2,
        22,
        '<td><span style="display: none">929167</span>$929,167</td>' * 2
        + '<td><span style="display: none">4</span><div class="pill-rfa">RFA</div></td>',
    )
    html = (
        '<table id="dataTable-active"><tbody>' + nhl + "</tbody></table>"
        '<table id="table" class="table yearly-minors-table"><tbody>' + minor + "</tbody></table>"
    )
    out = _parse_yearly_team_html(html, 2026)
    parsons = out["jackson parsons"]
    assert parsons["aav_m"] == 0.929
    assert parsons["years_remaining"] == 2
    assert parsons["rights_status"] == "RFA"
    assert parsons["spotrac_minor"] is True
    assert parsons["contract_type"] == "ELC"
    assert "spotrac_minor" not in out["linus ullmark"]


def test_cheap_veteran_deal_is_not_an_elc():
    from services.real_nhl_contracts import _parse_yearly_row

    row = _yearly_row("Nikolas Matinpalo", 3, 28, '<td data-sort="875000">$875,000</td>')
    entry = _parse_yearly_row(row, 2026)
    assert entry["contract_type"] == "STANDARD"


def test_estimated_farmhand_contract_clamped_to_minimum_level():
    from services.contract_economy import apply_contract_to_player
    from services.real_nhl_roster_importer import _clamp_estimated_contract_to_minor_level

    player = types.SimpleNamespace(contract=None)
    apply_contract_to_player(
        player,
        {
            "aav_m": 3.25,
            "cap_hit_m": 3.25,
            "years": 2,
            "years_remaining": 2,
            "contract_type": "STANDARD",
            "rights_status": "RFA",
            "source": "estimated",
            "is_nhl_spc": True,
        },
        2026,
    )
    assert _clamp_estimated_contract_to_minor_level(player, season_year=2026)
    assert player.contract["aav_m"] <= 0.85 + 0.15 + 1e-9
    assert player.cap_hit_m == player.contract["cap_hit_m"]
    assert player.contract["two_way"] is True


def test_real_contract_is_not_clamped():
    from services.contract_economy import apply_contract_to_player
    from services.real_nhl_roster_importer import _clamp_estimated_contract_to_minor_level

    player = types.SimpleNamespace(contract=None)
    apply_contract_to_player(
        player,
        {"aav_m": 3.25, "cap_hit_m": 3.25, "years": 2, "years_remaining": 2, "source": "real_nhl_spotrac"},
        2026,
    )
    assert not _clamp_estimated_contract_to_minor_level(player, season_year=2026)
    assert player.contract["aav_m"] == 3.25


def test_offer_sheet_eligibility_has_no_salary_ceiling():
    from services import contract_economy as ce

    star = types.SimpleNamespace(age=24, identity=types.SimpleNamespace(age=24))
    kid = types.SimpleNamespace(age=19, identity=types.SimpleNamespace(age=19))
    assert ce.rfa_offer_sheet_eligible(star, {"previous_aav_m": 9.5})
    assert not ce.rfa_offer_sheet_eligible(kid, {"previous_aav_m": 0.95})


def _cap_player(aav: float, years: int, *, pending: bool = False):
    p = types.SimpleNamespace(
        retired=False,
        pending_july1_expiry=pending,
        contract={"aav_m": aav, "cap_hit_m": aav, "years_remaining": years},
        cap_hit_m=aav,
    )
    for flag in ("is_buried", "buried", "in_minors", "on_ir", "on_ltir", "is_ir", "is_ltir"):
        setattr(p, flag, False)
    return p


def test_opening_day_books_after_year_burn_keep_final_year_deals():
    from app.sim_engine.economy.cap_engine import team_following_season_active_cap_hit_millions

    roster = [
        _cap_player(5.0, 2),  # two seasons left
        _cap_player(3.0, 1),  # next season is his final year (or a 1-year deal signed this summer)
        _cap_player(8.0, 1, pending=True),  # unsigned UFA leaving July 1
    ]
    before_burn = types.SimpleNamespace(roster=list(roster), _contract_year_burned=False)
    after_burn = types.SimpleNamespace(roster=list(roster), _contract_year_burned=True)
    # Before the salary-cap stage burns a year, 1-year deals are the ones expiring.
    assert abs(team_following_season_active_cap_hit_millions(before_burn) - 5.0) < 1e-6
    # After the burn, only the deferred July-1 UFA leaves.
    assert abs(team_following_season_active_cap_hit_millions(after_burn) - 8.0) < 1e-6


def test_cpu_rfas_held_for_offer_sheet_window():
    from services import contract_economy as ce

    star = types.SimpleNamespace(id="p1", age=23, identity=types.SimpleNamespace(age=23, name="Star Rfa"))
    entry = {"player_id": "p1", "name": "Star Rfa", "overall": 85, "player_ref": star}
    team = types.SimpleNamespace(team_id="T2", id="T2", rfa_rights=[entry], roster=[], name="Club")
    league = types.SimpleNamespace(teams=[team])
    session = types.SimpleNamespace(
        sim=types.SimpleNamespace(league=league), season_calendar_year=2026, user_team_id="T1"
    )
    orig = ce._player_ovr
    ce._player_ovr = lambda p: 85.0
    try:
        out = ce.run_cpu_rfa_decisions(session)
    finally:
        ce._player_ovr = orig
    assert out["deferred_count"] == 1
    assert team.rfa_rights and team.rfa_rights[0]["qualified"] is True
    assert 3 <= team.rfa_rights[0]["cpu_resolve_day"] <= 6


def test_cpu_bids_meet_the_ask_and_desperate_clubs_exceed_it():
    import random

    from services.fa_market_engine import _cpu_offer_vs_ask

    rng = random.Random(4)
    normal = [
        _cpu_offer_vs_ask(ask=5.0, ovr=83, cap_space_m=20.0, need=0.3, window="bubble", roster_count=22, rng=rng)
        for _ in range(200)
    ]
    desperate = [
        _cpu_offer_vs_ask(ask=5.0, ovr=83, cap_space_m=20.0, need=0.8, window="contender", roster_count=18, rng=rng)
        for _ in range(200)
    ]
    assert min(normal) >= 5.0 * 0.99 - 1e-9
    assert min(desperate) >= 5.0 - 1e-9
    assert max(desperate) > 5.0 * 1.08
    # A club that can't get near the ask stays out instead of lowballing.
    assert _cpu_offer_vs_ask(ask=5.0, ovr=83, cap_space_m=3.0, need=0.8, window="bubble", roster_count=22, rng=rng) is None


def test_user_target_signed_by_cpu_is_reported_same_day():
    from services.franchise_offseason import _user_targets_lost_to_cpu

    session = types.SimpleNamespace(
        user_team_id="5",
        resign_negotiations={"p1": {"pending_offer": {"aav_m": 2.0, "years": 2, "context": "ufa"}}},
    )
    tick = {"signings": [{"player_id": "p1", "name": "Some Winger", "team_id": "9", "team_name": "Blue Jackets", "aav_m": 2.4, "years": 3}]}
    lost = _user_targets_lost_to_cpu(session, tick)
    assert len(lost) == 1
    assert lost[0]["signed_team_name"] == "Blue Jackets"
    assert "Blue Jackets" in lost[0]["feedback"]
    assert session.resign_negotiations["p1"]["pending_offer"] is None
