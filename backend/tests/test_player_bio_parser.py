"""Tests for player_bios.txt parser."""

from datetime import date

from services.player_bio_parser import (
    load_player_bio_registry,
    normalize_nationality,
    parse_dob,
    parse_height_cm,
    parse_nationalities,
    parse_player_bios,
    parse_weight_kg,
    resolve_wjc_country_code,
)


def test_normalize_nationality_and_dual():
    assert normalize_nationality("Canada 🇨🇦") == "Canada"
    assert normalize_nationality("USA / Canada") == "USA"
    assert parse_nationalities("Switzerland/Canada") == ["Switzerland", "Canada"]


def test_parse_imperial_measures():
    assert parse_height_cm("6'2\"") == 188
    assert parse_height_cm("5'11\"") == 180
    assert parse_weight_kg("204 lbs") == 93


def test_parse_dob_formats():
    assert parse_dob("2006-11-04") == (2006, 11, 4)
    assert parse_dob("Jul. 5, 2004") == (2004, 7, 5)
    assert parse_dob("Aug. 15, 2004") == (2004, 8, 15)


def test_wjc_dual_citizen():
    code = resolve_wjc_country_code("USA / Canada", nationalities=["USA", "Canada"])
    assert code in ("USA", "CAN")


def test_parse_sample_lines():
    sample = (
        "Carter Bear (LW/C): 78 ovr, 84 cha, 84 off, 78 def, 82 tra, 82 men, 78 phy, 88 pot"
        " | Age: 19 | DOB: 2006-11-04 | Height: 6'0\" | Weight: 187 lbs | Nationality: Canada\n"
        "Jake Richard (RW/LW): Age 22 | DOB: Aug. 15, 2004 | 6'1\" | 194 lbs | USA / Canada\n"
        "Player\tAge\tBirthday\tHeight\tWeight\tNationalityFraser Minten\t22\tJul. 5, 2004\t6'2\"\t204 lbs\tCanada"
    )
    reg = parse_player_bios(sample, as_of=date(2026, 9, 15))
    assert reg.parse_stats["parsed"] == 3
    bear = reg.lookup("Carter Bear")
    assert bear is not None
    assert bear.age == 19
    assert bear.height_cm == 183
    assert bear.nationality == "Canada"
    richard = reg.lookup("Jake Richard")
    assert richard is not None
    assert "USA" in richard.nationalities


def test_wjc_age_eligibility_jan4_cutoff():
    from datetime import date

    from services.player_bio_parser import wjc_age_eligible, wjc_eligibility_cutoff

    cutoff = wjc_eligibility_cutoff(2025)
    assert cutoff == date(2026, 1, 4)
    born_jan5 = {"birth_year": 2006, "birth_month": 1, "birth_day": 5}
    born_jan3 = {"birth_year": 2006, "birth_month": 1, "birth_day": 3}
    assert wjc_age_eligible(born_jan5, 2025) is True
    assert wjc_age_eligible(born_jan3, 2025) is False


def test_russia_in_wjc_pool():
    from services.player_bio_parser import WJC_COUNTRY_META, resolve_wjc_country_code

    codes = [c for c, _ in WJC_COUNTRY_META]
    assert "RUS" in codes
    assert resolve_wjc_country_code("Russia") == "RUS"
    reg = load_player_bio_registry(as_of=date(2026, 9, 15))
    assert reg.parse_stats["parsed"] >= 500
    assert reg.lookup("Konsta Helenius") is not None
    assert reg.lookup("Fraser Minten") is not None


def test_wjc_federation_merge_non_pool():
    from services.player_bio_parser import merge_wjc_federation_code

    assert merge_wjc_federation_code("Belarus") == "RUS"
    assert merge_wjc_federation_code("Kazakhstan") == "RUS"
    assert merge_wjc_federation_code("Norway") == "DEN"
    assert merge_wjc_federation_code("Austria") == "GER"


def test_wjc_junior_league_dual_tiebreak():
    from services.player_bio_parser import resolve_wjc_country_code, wjc_code_from_junior_league

    assert wjc_code_from_junior_league("QMJHL") == "CAN"
    assert wjc_code_from_junior_league("USHL") == "USA"
    code = resolve_wjc_country_code(
        "Canada / Latvia",
        nationalities=["Canada", "Latvia"],
        junior_league="LHL Latvia",
    )
    assert code == "LAT"
    code_us = resolve_wjc_country_code(
        "Canada / USA",
        nationalities=["Canada", "USA"],
        junior_league="USHL",
    )
    assert code_us == "USA"
