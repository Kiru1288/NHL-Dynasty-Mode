"""Real NHL import must not surface seasons/awards from the franchise opening year onward."""
from services.real_nhl_roster_importer import (
    drop_nhl_awards_from_franchise_open,
    drop_nhl_import_from_franchise_open,
)


def test_drop_career_seasons_from_franchise_open_year():
    seasons = [
        {"season": "2024-25", "gp": 82},
        {"season": "2025-26", "gp": 80},
        {"season": "2026-27", "gp": 10},
    ]
    out = drop_nhl_import_from_franchise_open(seasons, 2026)
    assert [s["season"] for s in out] == ["2024-25", "2025-26"]


def test_drop_nhl_awards_from_franchise_open_year():
    awards = [
        {"award_name": "Selke", "season": "2025-26"},
        {"award_name": "Hart", "season": "2026-27"},
    ]
    out = drop_nhl_awards_from_franchise_open(awards, 2026)
    assert [a["season"] for a in out] == ["2025-26"]
