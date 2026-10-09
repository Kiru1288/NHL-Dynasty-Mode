"""Unit tests for Spotrac current-season AAV overlay (no network)."""

from __future__ import annotations

from services.real_nhl_contracts import _merge_cap_aav_over_yearly


def test_merge_prefers_cap_sheet_aav_over_extension_yearly():
    yearly = {
        "shane pinto": {
            "name": "Shane Pinto",
            "aav_m": 7.5,
            "cap_hit_m": 7.5,
            "years_remaining": 4,
            "years": 4,
            "spotrac_id": 1,
            "source": "real_nhl_spotrac",
        },
        "jordan spence": {
            "name": "Jordan Spence",
            "aav_m": 5.0,
            "cap_hit_m": 5.0,
            "years_remaining": 4,
            "years": 4,
            "spotrac_id": 2,
            "source": "real_nhl_spotrac",
        },
    }
    cap = {
        "shane pinto": {
            "name": "Shane Pinto",
            "aav_m": 3.75,
            "cap_hit_m": 3.75,
            "spotrac_id": 1,
            "source": "real_nhl_spotrac_cap",
        },
        "jordan spence": {
            "name": "Jordan Spence",
            "aav_m": 1.5,
            "cap_hit_m": 1.5,
            "spotrac_id": 2,
            "source": "real_nhl_spotrac_cap",
        },
    }
    merged = _merge_cap_aav_over_yearly(yearly, cap)
    assert merged["shane pinto"]["aav_m"] == 3.75
    assert merged["shane pinto"]["years_remaining"] == 1
    assert merged["shane pinto"]["extension_aav_m"] == 7.5
    assert merged["jordan spence"]["aav_m"] == 1.5
    assert merged["jordan spence"]["years_remaining"] == 1
    assert merged["jordan spence"].get("extension_years_remaining") == 4


def test_sync_term_from_flat_season_cap_hits_greig_like():
    from services.contract_economy import _sync_term_from_season_cap_hits

    c = {
        "aav_m": 3.25,
        "cap_hit_m": 3.25,
        "years_remaining": 1,
        "years": 1,
        "season_cap_hits": [3.25, 3.25, 3.25],
        "expiry_year": 2027,
    }
    _sync_term_from_season_cap_hits(c, 2026)
    assert c["years_remaining"] == 3
    assert c["expiry_year"] == 2029
    assert c["nhl_salary_by_year_m"] == [3.25, 3.25, 3.25]


def test_clause_schedule_does_not_block_until_kick_in_year():
    from services.contract_economy import advance_dict_contract_one_season, normalize_contract_dict

    c = normalize_contract_dict({
        "aav_m": 8.0,
        "cap_hit_m": 8.0,
        "years_remaining": 3,
        "season_cap_hits": [8.0, 8.0, 8.0],
        "clause_by_year": ["", "NTC", "NMC"],
        "no_trade_clause": True,
        "ntc_mode": "FULL",
        "effective_season": 2026,
        "source": "real_nhl_spotrac",
    })
    assert c["no_trade_clause"] is False
    assert c["nmc"] is False
    assert c["clause_display"] == "NTC starts 2027-28 · NMC 2028-29"
    assert c["clause_kicks_in_year"] == 2027
    advance_dict_contract_one_season(c, 2026)
    assert c["no_trade_clause"] is True
    assert c["clause_type"] == "NTC"
    assert "NTC" in c["clause_display"]
    assert "NMC 2028-29" in c["clause_display"]


def test_merge_keeps_yearly_term_when_cap_aav_matches():
    yearly = {
        "drake batherson": {
            "name": "Drake Batherson",
            "aav_m": 5.0,
            "cap_hit_m": 5.0,
            "years_remaining": 3,
            "years": 3,
            "spotrac_id": 9,
            "source": "real_nhl_spotrac",
        },
    }
    cap = {
        "drake batherson": {
            "name": "Drake Batherson",
            "aav_m": 5.0,
            "cap_hit_m": 5.0,
            "years_remaining": 1,
            "years": 1,
            "spotrac_id": 9,
            "source": "real_nhl_spotrac_cap",
        },
    }
    merged = _merge_cap_aav_over_yearly(yearly, cap)
    assert merged["drake batherson"]["years_remaining"] == 3
    assert merged["drake batherson"]["aav_m"] == 5.0


def test_resign_desk_keeps_next_season_deals_and_moves_a_signing_forward():
    """A year that covers next season is not this July's UFA. Signed deals are left alone."""
    from services.contract_economy import (
        forward_years_from_reported,
        is_current_ufa_class,
        stamp_forward_contract_term,
    )

    # CapWages "2 years" in 2026-27. After that season was played, 2027-28 is still owed.
    assert forward_years_from_reported(2, 2026, season_already_played=True) == 1
    assert forward_years_from_reported(2, 2026, season_already_played=False) == 2
    # Final year of 2026-27 is a UFA once that season has been played.
    assert forward_years_from_reported(1, 2026, season_already_played=True) == 0

    player = type("P", (), {})()
    contract = {
        "aav_m": 8.0,
        "cap_hit_m": 8.0,
        "years_remaining": 1,
        "expiry_year": 2027,
        "source": "real_nhl_spotrac_cap",
        "pending_july1_expiry": True,
    }
    assert stamp_forward_contract_term(
        contract, player, 1, 2026, season_already_played=True
    )
    assert contract["years_remaining"] == 1
    assert contract["expiry_year"] == 2028
    assert "pending_july1_expiry" not in contract
    assert player.pending_july1_expiry is False
    assert is_current_ufa_class(
        years_remaining=1,
        expiry_year=2028,
        season_year=2026,
        pending_july1=False,
        extension_signed=False,
    ) is False
    assert is_current_ufa_class(
        years_remaining=1,
        expiry_year=2027,
        season_year=2026,
        pending_july1=True,
        extension_signed=False,
    ) is True

    signed = {"source": "signed", "user_signed": True, "years_remaining": 4, "aav_m": 6.5}
    assert stamp_forward_contract_term(signed, None, 1, 2026, season_already_played=True) is False
