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
