"""ELC and league-minimum deals do not count against the cap."""

import random
from types import SimpleNamespace

from app.sim_engine.economy.cap_engine import player_cap_hit_millions
from app.sim_engine.progression.regression import apply_regression
from services.franchise_entry_draft import _revealed_ability_payload
from services.franchise_sim import shown_ovr_and_potential


def _deal(contract_type: str, aav: float) -> SimpleNamespace:
    contract = {
        "type": contract_type,
        "contract_type": contract_type,
        "aav_m": aav,
        "cap_hit_m": aav,
        "years": 3,
        "years_remaining": 3,
    }
    return SimpleNamespace(contract=contract, aav_m=aav, cap_hit_m=aav)


def test_elc_and_league_minimum_do_not_count_against_the_cap():
    elc = _deal("ELC", 0.95)
    league_min = _deal("STANDARD", 0.85)
    under_min = _deal("STANDARD", 0.75)
    real = _deal("STANDARD", 6.5)

    assert player_cap_hit_millions(elc) == 0.0
    assert player_cap_hit_millions(league_min) == 0.0
    assert player_cap_hit_millions(under_min) == 0.0
    assert player_cap_hit_millions(real) == 6.5
    # The stored ELC stays $0.95 so true-ELC checks still recognize it.
    assert elc.cap_hit_m == 0.95
    assert elc.contract["cap_hit_m"] == 0.95


def test_quiet_veteran_does_not_lose_overall():
    player = SimpleNamespace(
        age=34,
        peak_age=27,
        morale=0.72,
        ratings={"off_shot": 80, "off_pass": 78, "def_gap": 74},
    )
    before = dict(player.ratings)
    apply_regression(player, random.Random(3))
    assert player.ratings == before


def test_draft_reveal_matches_the_shared_card():
    player = SimpleNamespace(
        ovr=lambda: 78,
        ratings={"dev_potential": 86},
        potential=86,
    )
    ovr, pot = shown_ovr_and_potential(player)
    revealed = _revealed_ability_payload(player)
    assert revealed["drafted_overall"] == ovr
    assert revealed["drafted_potential"] == pot
    assert revealed["overall"] == ovr
    assert pot == 86
