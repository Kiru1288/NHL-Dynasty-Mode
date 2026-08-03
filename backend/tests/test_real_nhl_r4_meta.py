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
