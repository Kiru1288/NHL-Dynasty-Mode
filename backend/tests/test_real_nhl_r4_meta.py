"""Tests for Real NHL R4 overrides, draft/body wiring, and Spotrac contract parsing."""

from __future__ import annotations

from types import SimpleNamespace

from services.real_nhl_contracts import (
    is_real_nhl_contract,
    match_contract_for_player,
    normalize_player_name,
    _parse_yearly_team_html,
)
from services.real_nhl_roster_importer import (
    apply_r4_target,
    load_r4_overrides,
    _apply_draft_and_body,
)
from app.sim_engine.generation.player_headshots import (
    headshot_fields_from_player,
    merge_headshot_into_row,
    valid_nhl_headshot_url,
)


def test_r4_pack_loads_and_pins_mcdavid():
    overrides = load_r4_overrides()
    assert 8478402 in overrides
    ovr, profile, ov = apply_r4_target(
        nhl_id=8478402,
        target_ovr=0.90,
        profile="two_way",
        overrides=overrides,
    )
    assert ov is not None
    assert ovr >= 0.96
    assert profile == "playmaker"


def test_r4_unknown_player_unchanged():
    ovr, profile, ov = apply_r4_target(
        nhl_id=1,
        target_ovr=0.72,
        profile="grinder",
        overrides={},
    )
    assert ov is None
    assert ovr == 0.72
    assert profile == "grinder"


def test_draft_and_body_from_landing():
    ident = SimpleNamespace(
        height_cm=180,
        weight_kg=80.0,
        draft_year=0,
        draft_round=0,
        draft_pick=0,
    )
    player = SimpleNamespace(identity=ident)
    teams = [SimpleNamespace(abbreviation="EDM", team_id="team_edm", id="team_edm")]
    landing = {
        "heightInInches": 73,
        "weightInPounds": 194,
        "draftDetails": {
            "year": 2015,
            "round": 1,
            "pickInRound": 1,
            "overallPick": 1,
            "teamAbbrev": "EDM",
        },
    }
    _apply_draft_and_body(
        player,
        roster_row={"heightInInches": 72, "weightInPounds": 190},
        landing=landing,
        teams=teams,
    )
    assert player.drafted is True
    assert player.draft_year == 2015
    assert player.draft_round == 1
    assert player.draft_overall_pick == 1
    assert player.draft_team_abbr == "EDM"
    assert player.drafted_by == "team_edm"
    assert ident.height_cm == 185  # 73 * 2.54
    assert abs(float(ident.weight_kg) - 88.0) < 1.0


def test_undrafted_landing():
    ident = SimpleNamespace(height_cm=180, weight_kg=80.0, draft_year=0, draft_round=0, draft_pick=0)
    player = SimpleNamespace(identity=ident)
    _apply_draft_and_body(
        player,
        roster_row={},
        landing={"heightInInches": 72, "weightInPounds": 200},
        teams=[],
    )
    assert player.undrafted is True
    assert player.drafted is False


def test_spotrac_yearly_html_parse_mcdavid_like_row():
    html = """
    <table><thead><tr><th>Player (1)</th></tr></thead>
    <tbody>
      <tr>
        <td><a href="https://www.spotrac.com/nhl/player/_/id/17891/connor-mcdavid">Connor McDavid</a> NMC</td>
        <td data-sort="C">C</td>
        <td data-sort="30">30</td>
        <td data-sort="12500000">$12,500,000</td>
        <td data-sort="12500000">$12,500,000</td>
        <td data-sort="5">UFA</td>
        <td data-sort="-10">&nbsp;</td>
      </tr>
    </tbody></table>
    """
    parsed = _parse_yearly_team_html(html)
    key = normalize_player_name("Connor McDavid")
    assert key in parsed
    row = parsed[key]
    assert row["aav_m"] == 12.5
    assert row["years_remaining"] == 2
    assert row["rights_status"] == "UFA"
    assert row["no_move_clause"] is True
    assert row["source"].startswith("real_nhl")


def test_match_contract_by_name():
    contracts = {
        "EDM": {
            normalize_player_name("Connor McDavid"): {
                "aav_m": 12.5,
                "years_remaining": 2,
                "source": "real_nhl_spotrac",
            }
        }
    }
    hit = match_contract_for_player("Connor McDavid", "EDM", contracts)
    assert hit is not None
    assert hit["aav_m"] == 12.5
    assert is_real_nhl_contract(SimpleNamespace(contract=hit, real_nhl_contract=True))


def test_duplicate_name_contracts_prefer_role_aav():
    key = normalize_player_name("Elias Pettersson")
    contracts = {
        "VAN": {
            key: [
                {
                    "aav_m": 11.6,
                    "years_remaining": 8,
                    "spotrac_id": 1,
                    "source": "real_nhl_spotrac",
                },
                {
                    "aav_m": 0.913,
                    "years_remaining": 1,
                    "spotrac_id": 2,
                    "source": "real_nhl_spotrac",
                },
            ]
        }
    }
    fwd = match_contract_for_player("Elias Pettersson", "VAN", contracts, position_code="C")
    dman = match_contract_for_player("Elias Pettersson", "VAN", contracts, position_code="D")
    assert fwd is not None and dman is not None
    assert float(fwd["aav_m"]) > float(dman["aav_m"])
    assert float(fwd["aav_m"]) >= 10.0
    assert float(dman["aav_m"]) < 2.0


def test_merge_roster_rows_supplements_thin_current_roster():
    from services.real_nhl_roster_importer import _merge_roster_rows

    current = [{"id": 1, "lastName": {"default": "A"}}]
    prior = [
        {"id": 1, "lastName": {"default": "A-old"}},
        {"id": 2, "lastName": {"default": "Star"}},
    ]
    merged = _merge_roster_rows(current, prior)
    ids = {int(r["id"]) for r in merged}
    assert ids == {1, 2}
    assert next(r for r in merged if int(r["id"]) == 1)["lastName"]["default"] == "A"


def test_pick_stats_prefers_full_season_over_injury_sample():
    from services.real_nhl_roster_importer import pick_stats_row

    primary = {1: {"gamesPlayed": 50, "points": 25}}
    secondary = {1: {"gamesPlayed": 77, "points": 63}}
    row = pick_stats_row(1, is_goalie=False, primary=primary, secondary=secondary)
    assert row is secondary[1]


def test_current_nhl_season_start_year_uses_july_cutoff():
    from datetime import date

    from services.nhl_season_calendar import current_nhl_season_start_year

    assert current_nhl_season_start_year(date(2026, 8, 15)) == 2026
    assert current_nhl_season_start_year(date(2026, 6, 15)) == 2025
    assert current_nhl_season_start_year(date(2027, 1, 10)) == 2026


def test_real_nhl_headshot_fields_layer_over_generated_fallback():
    player = SimpleNamespace(
        id="NHL_8478402",
        nhl_player_id=8478402,
        nhl_headshot_url="https://assets.nhle.com/mugs/nhl/20252026/EDM/8478402.png",
        avatar_seed=123,
        headshot_id=7,
        face_variant=7,
    )
    fields = headshot_fields_from_player(player)
    assert fields["nhl_player_id"] == 8478402
    assert fields["portrait_source"] == "nhl"
    assert fields["nhl_headshot_url"].startswith("https://assets.nhle.com/")
    assert fields["headshot_id"] == 7
    assert fields["avatar_seed"] == 123


def test_invalid_nhl_headshot_url_keeps_generated_metadata_only():
    player = SimpleNamespace(
        id="NHL_1",
        nhl_player_id=1,
        nhl_headshot_url="https://example.invalid/player.png",
        avatar_seed=456,
        headshot_id=8,
        face_variant=8,
    )
    row = merge_headshot_into_row({"player_id": "NHL_1"}, player)
    assert valid_nhl_headshot_url(player.nhl_headshot_url) == ""
    assert "nhl_headshot_url" not in row
    assert "portrait_source" not in row
    assert row["headshot_id"] == 8


def test_old_save_without_nhl_fields_still_gets_generated_headshot():
    player = SimpleNamespace(
        id="legacy-player",
        name="Legacy Player",
        age=27,
        position="C",
    )
    row = merge_headshot_into_row({"player_id": "legacy-player"}, player)
    assert 1 <= int(row["headshot_id"]) <= 60
    assert int(row["avatar_seed"]) > 0
    assert "nhl_headshot_url" not in row


def test_roster_serializer_exposes_nhl_headshot_metadata():
    from services.franchise_sim import _serialize_player_row

    identity = SimpleNamespace(
        name="Imported Player",
        position="C",
        shoots="L",
        age=26,
        birth_year=2000,
        birth_month=1,
        birth_day=1,
        birth_country="Canada",
        height_cm=183,
        weight_kg=86,
    )
    player = SimpleNamespace(
        id="NHL_8478402",
        identity=identity,
        ovr=lambda: 0.9,
        contract={},
        nhl_player_id=8478402,
        nhl_headshot_url="https://assets.nhle.com/mugs/nhl/20252026/EDM/8478402.png",
        avatar_seed=123,
        headshot_id=7,
        face_variant=7,
    )
    row = _serialize_player_row(player)
    assert row["nhl_player_id"] == 8478402
    assert row["portrait_source"] == "nhl"
    assert row["nhl_headshot_url"].endswith("/8478402.png")
    assert row["headshot_id"] == 7


def test_landing_fetches_are_deduplicated_and_cached(monkeypatch):
    import services.real_nhl_roster_importer as importer

    calls = []
    importer._PLAYER_LANDING_CACHE.clear()

    def fake_fetch(player_id):
        calls.append(player_id)
        return {"playerId": player_id, "headshot": f"https://assets.nhle.com/{player_id}.png"}

    monkeypatch.setattr(importer, "_fetch_player_landing", fake_fetch)
    first = importer.fetch_landings_by_id([7, 7, 8], max_workers=1)
    second = importer.fetch_landings_by_id([7, 8], max_workers=1)

    assert set(first) == {7, 8}
    assert set(second) == {7, 8}
    assert calls == [7, 8]
