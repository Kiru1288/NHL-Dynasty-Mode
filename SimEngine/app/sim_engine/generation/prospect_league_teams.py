"""
Real junior / development team registry for prospect assignment and league display.

Used by league_hierarchy_bootstrap (generation) and draft_ranking_logic (display fixes).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

TeamSpec = Tuple[str, str]  # (city_key, full_name)


def _norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return text


def _teams(*rows: TeamSpec) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for city, name in rows:
        out.append({"city": city, "name": name, "key": _norm_key(name)})
    return out


# --- Canada / USA junior ---
OHL_TEAMS = _teams(
    ("Barrie", "Barrie Colts"),
    ("Brampton", "Brampton Steelheads"),
    ("Brantford", "Brantford Bulldogs"),
    ("Erie", "Erie Otters"),
    ("Flint", "Flint Firebirds"),
    ("Guelph", "Guelph Storm"),
    ("Kingston", "Kingston Frontenacs"),
    ("Kitchener", "Kitchener Rangers"),
    ("London", "London Knights"),
    ("Niagara", "Niagara IceDogs"),
    ("North Bay", "North Bay Battalion"),
    ("Oshawa", "Oshawa Generals"),
    ("Ottawa", "Ottawa 67's"),
    ("Owen Sound", "Owen Sound Attack"),
    ("Peterborough", "Peterborough Petes"),
    ("Saginaw", "Saginaw Spirit"),
    ("Sarnia", "Sarnia Sting"),
    ("Sault Ste. Marie", "Soo Greyhounds"),
    ("Sudbury", "Sudbury Wolves"),
    ("Windsor", "Windsor Spitfires"),
)

WHL_TEAMS = _teams(
    ("Brandon", "Brandon Wheat Kings"),
    ("Calgary", "Calgary Hitmen"),
    ("Edmonton", "Edmonton Oil Kings"),
    ("Everett", "Everett Silvertips"),
    ("Kamloops", "Kamloops Blazers"),
    ("Kelowna", "Kelowna Rockets"),
    ("Lethbridge", "Lethbridge Hurricanes"),
    ("Medicine Hat", "Medicine Hat Tigers"),
    ("Moose Jaw", "Moose Jaw Warriors"),
    ("Penticton", "Penticton Vees"),
    ("Portland", "Portland Winterhawks"),
    ("Prince Albert", "Prince Albert Raiders"),
    ("Prince George", "Prince George Cougars"),
    ("Red Deer", "Red Deer Rebels"),
    ("Regina", "Regina Pats"),
    ("Saskatoon", "Saskatoon Blades"),
    ("Seattle", "Seattle Thunderbirds"),
    ("Spokane", "Spokane Chiefs"),
    ("Swift Current", "Swift Current Broncos"),
    ("Tri-City", "Tri-City Americans"),
    ("Vancouver", "Vancouver Giants"),
    ("Victoria", "Victoria Royals"),
    ("Wenatchee", "Wenatchee Wild"),
)

QMJHL_TEAMS = _teams(
    ("Baie-Comeau", "Baie-Comeau Drakkar"),
    ("Blainville-Boisbriand", "Blainville-Boisbriand Armada"),
    ("Sydney", "Cape Breton Eagles"),
    ("Charlottetown", "Charlottetown Islanders"),
    ("Chicoutimi", "Chicoutimi Sagueneens"),
    ("Drummondville", "Drummondville Voltigeurs"),
    ("Gatineau", "Gatineau Olympiques"),
    ("Halifax", "Halifax Mooseheads"),
    ("Moncton", "Moncton Wildcats"),
    ("St. John's", "Newfoundland Regiment"),
    ("Quebec City", "Quebec Remparts"),
    ("Rimouski", "Rimouski Oceanic"),
    ("Rouyn-Noranda", "Rouyn-Noranda Huskies"),
    ("Saint John", "Saint John Sea Dogs"),
    ("Shawinigan", "Shawinigan Cataractes"),
    ("Sherbrooke", "Sherbrooke Phoenix"),
    ("Val-d'Or", "Val-d'Or Foreurs"),
    ("Victoriaville", "Victoriaville Tigres"),
)

USHL_TEAMS = _teams(
    ("Cedar Rapids", "Cedar Rapids RoughRiders"),
    ("Chicago", "Chicago Steel"),
    ("Des Moines", "Des Moines Buccaneers"),
    ("Dubuque", "Dubuque Fighting Saints"),
    ("Fargo", "Fargo Force"),
    ("Green Bay", "Green Bay Gamblers"),
    ("Lincoln", "Lincoln Stars"),
    ("Madison", "Madison Capitols"),
    ("Muskegon", "Muskegon Lumberjacks"),
    ("Omaha", "Omaha Lancers"),
    ("Sioux City", "Sioux City Musketeers"),
    ("Sioux Falls", "Sioux Falls Stampede"),
    ("Kearney", "Tri-City Storm"),
    ("Plymouth", "USNTDP Juniors"),
    ("Waterloo", "Waterloo Black Hawks"),
    ("Youngstown", "Youngstown Phantoms"),
)

NCAA_TEAMS = _teams(
    ("Boston", "Boston College"),
    ("Boston", "Boston University"),
    ("Ann Arbor", "Michigan"),
    ("Grand Forks", "North Dakota"),
    ("Denver", "Denver"),
    ("Hamden", "Quinnipiac"),
    ("Minneapolis", "Minnesota"),
    ("Madison", "Wisconsin"),
    ("Providence", "Providence College"),
    ("Ithaca", "Cornell"),
    ("New Haven", "Yale"),
    ("Cambridge", "Harvard"),
)

# --- European development / pro club systems (U20-style assignment) ---
SHL_TEAMS = _teams(
    ("Gavle", "Brynas IF"),
    ("Stockholm", "Djurgardens IF"),
    ("Gothenburg", "Frolunda HC"),
    ("Karlstad", "Farjestad BK"),
    ("Jonkoping", "HV71"),
    ("Leksand", "Leksands IF"),
    ("Linkoping", "Linkoping HC"),
    ("Lulea", "Lulea HF"),
    ("Malmo", "Malmo Redhawks"),
    ("Orebro", "Orebro HK"),
    ("Angelholm", "Rogle BK"),
    ("Skelleftea", "Skelleftea AIK"),
    ("Timra", "Timra IK"),
    ("Vaxjo", "Vaxjo Lakers"),
)

LIIGA_TEAMS = _teams(
    ("Helsinki", "HIFK"),
    ("Hameenlinna", "HPK"),
    ("Tampere", "Ilves"),
    ("Mikkeli", "Jukurit"),
    ("Jyvaskyla", "JYP"),
    ("Kuopio", "KalPa"),
    ("Espoo", "Kiekko-Espoo"),
    ("Kouvola", "KooKoo"),
    ("Oulu", "Karpat"),
    ("Rauma", "Lukko"),
    ("Lahti", "Pelicans"),
    ("Lappeenranta", "SaiPa"),
    ("Vaasa", "Sport"),
    ("Tampere", "Tappara"),
    ("Turku", "TPS"),
    ("Pori", "Assat"),
)

CZECH_TEAMS = _teams(
    ("Liberec", "Bili Tygri Liberec"),
    ("Mladá Boleslav", "BK Mladá Boleslav"),
    ("Pardubice", "HC Dynamo Pardubice"),
    ("Karlovy Vary", "HC Energie Karlovy Vary"),
    ("Brno", "HC Kometa Brno"),
    ("Litvinov", "HC Litvinov"),
    ("Trinec", "HC Ocelari Trinec"),
    ("Olomouc", "HC Olomouc"),
    ("Plzen", "HC Skoda Plzen"),
    ("Prague", "HC Sparta Praha"),
    ("Ostrava", "HC Vitkovice Ridera"),
    ("Ceske Budejovice", "Motor Ceske Budejovice"),
    ("Hradec Kralove", "Mountfield HK"),
    ("Kladno", "Rytiri Kladno"),
)

SLOVAK_TEAMS = _teams(
    ("Kosice", "HC Kosice"),
    ("Presov", "HC Presov"),
    ("Bratislava", "HC Slovan Bratislava"),
    ("Banska Bystrica", "HC 05 Banska Bystrica"),
    ("Zilina", "Vlci Zilina"),
    ("Trencin", "HK Dukla Trencin"),
    ("Nitra", "HK Nitra"),
    ("Poprad", "HK Poprad"),
    ("Zvolen", "HKM Zvolen"),
    ("Liptovsky Mikulas", "HK 32 Liptovsky Mikulas"),
    ("Michalovce", "HK Dukla Michalovce"),
    ("Spisska Nova Ves", "HK Spisska Nova Ves"),
)

SWISS_TEAMS = _teams(
    ("Zurich", "ZSC Lions"),
    ("Bern", "SC Bern"),
    ("Davos", "HC Davos"),
    ("Lausanne", "Lausanne HC"),
    ("Geneva", "Geneve-Servette HC"),
    ("Lugano", "HC Lugano"),
    ("Fribourg", "HC Fribourg-Gotteron"),
    ("Zug", "EV Zug"),
    ("Biel", "EHC Biel-Bienne"),
    ("Langnau", "SCL Tigers"),
    ("Rapperswil-Jona", "SC Rapperswil-Jona Lakers"),
    ("Ambri-Piotta", "HC Ambri-Piotta"),
    ("Kloten", "EHC Kloten"),
    ("Porrentruy", "HC Ajoie"),
)

DEL_TEAMS = _teams(
    ("Augsburg", "Augsburger Panther"),
    ("Berlin", "Eisbaren Berlin"),
    ("Bremerhaven", "Fischtown Pinguins"),
    ("Dresden", "Dresdner Eislowen"),
    ("Frankfurt", "Lowen Frankfurt"),
    ("Ingolstadt", "ERC Ingolstadt"),
    ("Iserlohn", "Iserlohn Roosters"),
    ("Cologne", "Kolner Haie"),
    ("Mannheim", "Adler Mannheim"),
    ("Munich", "EHC Red Bull Munchen"),
    ("Nuremberg", "Nurnberg Ice Tigers"),
    ("Schwenningen", "Schwenninger Wild Wings"),
    ("Straubing", "Straubing Tigers"),
    ("Wolfsburg", "Grizzlys Wolfsburg"),
)

KHL_TEAMS = _teams(
    ("Kazan", "Ak Bars Kazan"),
    ("Khabarovsk", "Amur Khabarovsk"),
    ("Omsk", "Avangard Omsk"),
    ("Yekaterinburg", "Avtomobilist Yekaterinburg"),
    ("Astana", "Barys Astana"),
    ("Moscow", "CSKA Moscow"),
    ("Minsk", "Dinamo Minsk"),
    ("Moscow", "Dynamo Moscow"),
    ("Shanghai", "Shanghai Dragons"),
    ("Togliatti", "Lada Togliatti"),
    ("Yaroslavl", "Lokomotiv Yaroslavl"),
    ("Magnitogorsk", "Metallurg Magnitogorsk"),
    ("Nizhnekamsk", "Neftekhimik Nizhnekamsk"),
    ("Ufa", "Salavat Yulaev Ufa"),
    ("Cherepovets", "Severstal Cherepovets"),
    ("Novosibirsk", "Sibir Novosibirsk"),
    ("Saint Petersburg", "SKA Saint Petersburg"),
    ("Sochi", "HC Sochi"),
    ("Moscow", "Spartak Moscow"),
    ("Nizhny Novgorod", "Torpedo Nizhny Novgorod"),
    ("Chelyabinsk", "Traktor Chelyabinsk"),
    ("Vladivostok", "Admiral Vladivostok"),
)

NORWAY_TEAMS = _teams(
    ("Oslo", "Vålerenga Ishockey"),
    ("Stavanger", "Stavanger Oilers"),
)

DENMARK_TEAMS = _teams(
    ("Copenhagen", "Rungsted Seier Capital"),
    ("Herning", "Herning Blue Fox"),
)

AUSTRIA_TEAMS = _teams(
    ("Vienna", "Vienna Capitals"),
    ("Villach", "EC VSV"),
)

LEAGUE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "CHL_OHL": {"display": "OHL", "parent": "CHL", "teams": OHL_TEAMS},
    "CHL_WHL": {"display": "WHL", "parent": "CHL", "teams": WHL_TEAMS},
    "CHL_QMJHL": {"display": "QMJHL", "parent": "CHL", "teams": QMJHL_TEAMS},
    "OHL": {"display": "OHL", "parent": "CHL", "teams": OHL_TEAMS},
    "WHL": {"display": "WHL", "parent": "CHL", "teams": WHL_TEAMS},
    "QMJHL": {"display": "QMJHL", "parent": "CHL", "teams": QMJHL_TEAMS},
    "USHL": {"display": "USHL", "parent": "USHL", "teams": USHL_TEAMS},
    "NCAA": {"display": "NCAA", "parent": "NCAA", "teams": NCAA_TEAMS},
    "EU_J_SHL": {"display": "J20 Nationell", "parent": "Sweden", "teams": SHL_TEAMS},
    "EU_J_LIIGA": {"display": "U20 SM-sarja", "parent": "Finland", "teams": LIIGA_TEAMS},
    "EU_J_DEL": {"display": "DEL", "parent": "Germany", "teams": DEL_TEAMS},
    "EU_J_SWISS": {"display": "NL", "parent": "Switzerland", "teams": SWISS_TEAMS},
    "EU_J_CZ": {"display": "Czech Extraliga", "parent": "Czechia", "teams": CZECH_TEAMS},
    "EU_J_SK": {"display": "Slovak Extraliga", "parent": "Slovakia", "teams": SLOVAK_TEAMS},
    "EU_J_KHL_JR": {"display": "MHL", "parent": "Russia", "teams": KHL_TEAMS},
    "EU_J_NOR": {"display": "Norway", "parent": "Norway", "teams": NORWAY_TEAMS},
    "EU_J_DEN": {"display": "Denmark", "parent": "Denmark", "teams": DENMARK_TEAMS},
    "EU_J_AUT": {"display": "Austria", "parent": "Austria", "teams": AUSTRIA_TEAMS},
}

_NATIONALITY_ALIASES = {
    "usa": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "russian federation": "Russia",
    "russia": "Russia",
    "slovak republic": "Slovakia",
    "slovakia": "Slovakia",
    "latvia": "Latvia",
    "kazakhstan": "Kazakhstan",
    "japan": "Japan",
    "sweden": "Sweden",
    "finland": "Finland",
    "canada": "Canada",
    "germany": "Germany",
    "switzerland": "Switzerland",
    "belarus": "Belarus",
    "norway": "Norway",
    "denmark": "Denmark",
    "austria": "Austria",
    "france": "France",
    "south korea": "South Korea",
    "korea": "South Korea",
    "china": "China",
    "australia": "Australia",
    "uk": "UK",
    "united kingdom": "UK",
    "great britain": "UK",
    "nigeria": "Nigeria",
    "south africa": "South Africa",
    "india": "India",
}


def normalize_nationality(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Canada"
    key = _norm_key(raw)
    return _NATIONALITY_ALIASES.get(key, raw)


def league_display_name(league_code: str) -> str:
    code = str(league_code or "").strip().upper()
    meta = LEAGUE_REGISTRY.get(code)
    return str(meta.get("display") if meta else code or "")


# Weighted nationality pools per development league block (must sum ~1.0)
LEAGUE_NATIONALITY_WEIGHTS: Dict[str, List[Tuple[str, float]]] = {
    # Canadian major-junior: strong Canadian majority, USA is the largest import.
    # Immigration-hub diversity in these leagues is expressed via heritage NAMES on
    # Canadian-nationality players (see name_generator), not via foreign birth country.
    "CHL_OHL": [
        ("Canada", 0.86), ("USA", 0.085), ("Czechia", 0.016), ("Slovakia", 0.011),
        ("Sweden", 0.006), ("Finland", 0.005), ("Russia", 0.004), ("Germany", 0.003),
        ("Latvia", 0.002), ("Switzerland", 0.0015),
    ],
    "CHL_WHL": [
        ("Canada", 0.835), ("USA", 0.11), ("Czechia", 0.018), ("Slovakia", 0.011),
        ("Sweden", 0.007), ("Finland", 0.006), ("Russia", 0.004), ("Germany", 0.003),
        ("Latvia", 0.0015),
    ],
    # QMJHL keeps slightly more European + francophone diversity than OHL/WHL.
    "CHL_QMJHL": [
        ("Canada", 0.775), ("USA", 0.085), ("France", 0.045), ("Czechia", 0.028),
        ("Slovakia", 0.02), ("Sweden", 0.014), ("Finland", 0.011), ("Switzerland", 0.006),
        ("Germany", 0.005), ("Russia", 0.003),
    ],
    "USHL": [
        ("USA", 0.755), ("Canada", 0.145), ("Sweden", 0.024), ("Finland", 0.014),
        ("Czechia", 0.011), ("Slovakia", 0.007), ("Germany", 0.006), ("Russia", 0.004),
        # rare imports, incl. non-traditional / non-hockey markets
        ("Japan", 0.005), ("Australia", 0.005), ("South Korea", 0.004), ("China", 0.004),
        ("UK", 0.004), ("Latvia", 0.004),
    ],
    "NCAA": [
        ("USA", 0.70), ("Canada", 0.185), ("Sweden", 0.024), ("Finland", 0.014),
        ("Czechia", 0.011), ("Slovakia", 0.007), ("Germany", 0.008), ("Russia", 0.005),
        ("Latvia", 0.004),
        # US college draws the widest set of non-traditional markets
        ("Japan", 0.005), ("Australia", 0.005), ("UK", 0.005), ("South Korea", 0.004),
        ("China", 0.004), ("Nigeria", 0.004), ("South Africa", 0.004), ("India", 0.004),
    ],
    "EU_J_SHL": [
        ("Sweden", 0.82), ("Finland", 0.10), ("Denmark", 0.025), ("Norway", 0.015),
        ("Czechia", 0.012), ("USA", 0.008), ("Canada", 0.005), ("Germany", 0.003),
        ("Latvia", 0.002),
    ],
    "EU_J_LIIGA": [
        ("Finland", 0.84), ("Sweden", 0.10), ("USA", 0.015), ("Canada", 0.012),
        ("Czechia", 0.008), ("Russia", 0.005), ("Germany", 0.004), ("Latvia", 0.002),
    ],
    "EU_J_DEL": [
        ("Germany", 0.74), ("USA", 0.08), ("Canada", 0.05), ("Czechia", 0.04),
        ("Switzerland", 0.03), ("Austria", 0.03), ("Sweden", 0.02), ("Finland", 0.015),
    ],
    "EU_J_SWISS": [
        ("Switzerland", 0.76), ("Germany", 0.10), ("Austria", 0.05), ("USA", 0.03),
        ("Canada", 0.025), ("Sweden", 0.015), ("Finland", 0.01), ("Czechia", 0.01),
    ],
    "EU_J_CZ": [
        ("Czechia", 0.76), ("Slovakia", 0.16), ("Germany", 0.025), ("USA", 0.015),
        ("Canada", 0.012), ("Sweden", 0.01), ("Finland", 0.008), ("Russia", 0.005),
    ],
    "EU_J_SK": [
        ("Slovakia", 0.72), ("Czechia", 0.20), ("Germany", 0.025), ("USA", 0.015),
        ("Canada", 0.012), ("Sweden", 0.008), ("Finland", 0.008), ("Russia", 0.005),
    ],
    "EU_J_KHL_JR": [
        ("Russia", 0.52), ("Belarus", 0.12), ("Kazakhstan", 0.11), ("Latvia", 0.08),
        ("Czechia", 0.04), ("Finland", 0.03), ("USA", 0.025),
        ("Germany", 0.015), ("Sweden", 0.01),
    ],
    "EU_J_NOR": [
        ("Norway", 0.84), ("Sweden", 0.08), ("Denmark", 0.04), ("USA", 0.02),
        ("Canada", 0.015), ("Finland", 0.005),
    ],
    "EU_J_DEN": [
        ("Denmark", 0.82), ("Sweden", 0.08), ("Norway", 0.05), ("Germany", 0.025),
        ("USA", 0.015), ("Canada", 0.01),
    ],
    "EU_J_AUT": [
        ("Austria", 0.68), ("Germany", 0.16), ("Switzerland", 0.06), ("Czechia", 0.04),
        ("USA", 0.025), ("Canada", 0.015), ("Slovakia", 0.015), ("Sweden", 0.01),
    ],
}

# Weighted league routes by nationality (primary pipelines + rare imports)
NATIONALITY_LEAGUE_WEIGHTS: Dict[str, List[Tuple[str, float]]] = {
    "Canada": [
        ("CHL_OHL", 0.34), ("CHL_WHL", 0.33), ("CHL_QMJHL", 0.28), ("USHL", 0.03),
        ("NCAA", 0.017), ("EU_J_SHL", 0.002),
    ],
    "USA": [
        ("USHL", 0.38), ("NCAA", 0.32), ("CHL_WHL", 0.12), ("CHL_OHL", 0.10),
        ("CHL_QMJHL", 0.05), ("EU_J_SHL", 0.015), ("EU_J_DEL", 0.005),
    ],
    "Sweden": [
        ("EU_J_SHL", 0.78), ("EU_J_LIIGA", 0.08), ("EU_J_DEL", 0.04), ("CHL_WHL", 0.03),
        ("USHL", 0.025), ("NCAA", 0.015), ("CHL_OHL", 0.01),
    ],
    "Finland": [
        ("EU_J_LIIGA", 0.78), ("EU_J_SHL", 0.10), ("EU_J_DEL", 0.04), ("CHL_WHL", 0.025),
        ("USHL", 0.02), ("NCAA", 0.015), ("CHL_OHL", 0.01),
    ],
    "Russia": [
        ("EU_J_KHL_JR", 0.72), ("EU_J_CZ", 0.06), ("EU_J_DEL", 0.05), ("EU_J_LIIGA", 0.04),
        ("CHL_WHL", 0.04), ("USHL", 0.03), ("NCAA", 0.025), ("CHL_OHL", 0.015),
    ],
    "Czechia": [
        ("EU_J_CZ", 0.62), ("EU_J_SK", 0.14), ("EU_J_DEL", 0.06), ("EU_J_SHL", 0.05),
        ("EU_J_LIIGA", 0.04), ("CHL_WHL", 0.035), ("USHL", 0.025), ("CHL_OHL", 0.02),
    ],
    "Slovakia": [
        ("EU_J_SK", 0.58), ("EU_J_CZ", 0.18), ("EU_J_SHL", 0.06), ("EU_J_LIIGA", 0.05),
        ("CHL_QMJHL", 0.035), ("CHL_OHL", 0.03), ("USHL", 0.025), ("NCAA", 0.02),
    ],
    "Germany": [
        ("EU_J_DEL", 0.55), ("EU_J_SWISS", 0.10), ("EU_J_AUT", 0.08), ("EU_J_CZ", 0.06),
        ("EU_J_SHL", 0.05), ("USHL", 0.05), ("NCAA", 0.045), ("CHL_WHL", 0.03),
    ],
    "Switzerland": [
        ("EU_J_SWISS", 0.62), ("EU_J_DEL", 0.12), ("EU_J_AUT", 0.08), ("EU_J_SHL", 0.06),
        ("EU_J_LIIGA", 0.04), ("USHL", 0.035), ("NCAA", 0.025), ("CHL_WHL", 0.02),
    ],
    "Latvia": [
        ("EU_J_KHL_JR", 0.28), ("EU_J_LIIGA", 0.22), ("EU_J_SHL", 0.18), ("EU_J_CZ", 0.10),
        ("EU_J_SK", 0.08), ("EU_J_DEL", 0.05), ("CHL_WHL", 0.035), ("USHL", 0.025),
        ("CHL_OHL", 0.02),
    ],
    "Kazakhstan": [
        ("EU_J_KHL_JR", 0.62), ("EU_J_CZ", 0.08), ("EU_J_DEL", 0.06), ("EU_J_LIIGA", 0.05),
        ("EU_J_SHL", 0.04), ("CHL_WHL", 0.035), ("USHL", 0.03), ("NCAA", 0.025),
    ],
    "Japan": [
        ("USHL", 0.28), ("NCAA", 0.24), ("CHL_WHL", 0.12), ("CHL_OHL", 0.10),
        ("EU_J_DEL", 0.08), ("EU_J_SHL", 0.06), ("EU_J_LIIGA", 0.05), ("CHL_QMJHL", 0.04),
    ],
    # Non-traditional / non-hockey markets: develop mostly through North America.
    "South Korea": [
        ("NCAA", 0.42), ("USHL", 0.34), ("CHL_WHL", 0.10), ("CHL_OHL", 0.08), ("EU_J_DEL", 0.06),
    ],
    "China": [
        ("NCAA", 0.40), ("USHL", 0.34), ("EU_J_KHL_JR", 0.10), ("EU_J_DEL", 0.08), ("CHL_WHL", 0.08),
    ],
    "Australia": [
        ("NCAA", 0.40), ("USHL", 0.36), ("CHL_WHL", 0.10), ("CHL_OHL", 0.08), ("EU_J_DEL", 0.06),
    ],
    "UK": [
        ("NCAA", 0.40), ("USHL", 0.34), ("EU_J_DEL", 0.10), ("EU_J_SHL", 0.08), ("CHL_WHL", 0.08),
    ],
    "Nigeria": [
        ("NCAA", 0.52), ("USHL", 0.40), ("CHL_OHL", 0.05), ("CHL_WHL", 0.03),
    ],
    "South Africa": [
        ("NCAA", 0.52), ("USHL", 0.40), ("CHL_OHL", 0.05), ("CHL_WHL", 0.03),
    ],
    "India": [
        ("NCAA", 0.50), ("USHL", 0.40), ("CHL_OHL", 0.06), ("CHL_WHL", 0.04),
    ],
    "Belarus": [
        ("EU_J_KHL_JR", 0.68), ("EU_J_CZ", 0.08), ("EU_J_DEL", 0.06), ("EU_J_LIIGA", 0.05),
        ("CHL_WHL", 0.04), ("USHL", 0.035), ("NCAA", 0.025),
    ],
    "Norway": [
        ("EU_J_NOR", 0.55), ("EU_J_SHL", 0.18), ("EU_J_DEN", 0.10), ("EU_J_DEL", 0.06),
        ("USHL", 0.045), ("NCAA", 0.035), ("CHL_WHL", 0.03),
    ],
    "Denmark": [
        ("EU_J_DEN", 0.55), ("EU_J_SHL", 0.16), ("EU_J_NOR", 0.10), ("EU_J_DEL", 0.08),
        ("USHL", 0.045), ("NCAA", 0.035), ("CHL_WHL", 0.03),
    ],
    "Austria": [
        ("EU_J_AUT", 0.52), ("EU_J_DEL", 0.18), ("EU_J_SWISS", 0.12), ("EU_J_CZ", 0.06),
        ("USHL", 0.045), ("NCAA", 0.035), ("CHL_WHL", 0.03),
    ],
}

NATIONALITY_PRIMARY_LEAGUES: Dict[str, List[str]] = {
    nat: [code for code, weight in pairs if weight >= 0.08]
    for nat, pairs in NATIONALITY_LEAGUE_WEIGHTS.items()
}

IMPORT_LEAGUE_CHANCE = 0.06
RARE_IMPORT_MAX_WEIGHT = 0.05
RARE_IMPORT_MIN_WEIGHT = 0.004


def _weighted_pick(rng, pairs: List[Tuple[str, float]], fallback: str) -> str:
    items = [(k, max(0.0, float(w))) for k, w in pairs if k and w > 0]
    if not items:
        return fallback
    total = sum(w for _, w in items)
    roll = rng.random() * total
    acc = 0.0
    for key, weight in items:
        acc += weight
        if roll <= acc:
            return key
    return items[-1][0]


def choose_nationality_for_league(rng, league_code: str) -> str:
    code = str(league_code or "").strip().upper()
    pairs = LEAGUE_NATIONALITY_WEIGHTS.get(code) or LEAGUE_NATIONALITY_WEIGHTS.get("CHL_OHL", [])
    return normalize_nationality(_weighted_pick(rng, pairs, "Canada"))


def pick_league_for_nationality(rng, nationality: str, *, allow_import: bool = True) -> str:
    nat = normalize_nationality(nationality)
    pairs = list(NATIONALITY_LEAGUE_WEIGHTS.get(nat) or NATIONALITY_LEAGUE_WEIGHTS.get("Canada", []))
    if not allow_import:
        pairs = [(code, weight) for code, weight in pairs if weight >= 0.05]
    if not pairs:
        pairs = NATIONALITY_LEAGUE_WEIGHTS["Canada"]
    if allow_import and rng.random() >= IMPORT_LEAGUE_CHANCE:
        primary = [(code, weight) for code, weight in pairs if weight >= 0.05]
        if primary:
            pairs = primary
    return _weighted_pick(rng, pairs, "CHL_OHL")


def pick_import_league(rng, nationality: str) -> str:
    return pick_league_for_nationality(rng, nationality, allow_import=True)


def _league_weight_for_nationality(nationality: str, league_code: str) -> float:
    nat = normalize_nationality(nationality)
    code = str(league_code or "").strip().upper()
    for league, weight in NATIONALITY_LEAGUE_WEIGHTS.get(nat, []):
        if league == code:
            return float(weight)
    return 0.0


def validate_prospect_league_fit(nationality: str, league_code: str) -> bool:
    """Return True when nationality + league is allowed (primary, secondary, or rare import)."""
    weight = _league_weight_for_nationality(nationality, league_code)
    if weight >= RARE_IMPORT_MIN_WEIGHT:
        return True
    code = str(league_code or "").strip().upper()
    nat = normalize_nationality(nationality)
    league_nat_pairs = LEAGUE_NATIONALITY_WEIGHTS.get(code, [])
    for candidate, w in league_nat_pairs:
        if normalize_nationality(candidate) == nat and w >= RARE_IMPORT_MIN_WEIGHT:
            return True
    return False


def league_fit_tier(nationality: str, league_code: str) -> str:
    weight = _league_weight_for_nationality(nationality, league_code)
    if weight >= 0.12:
        return "primary"
    if weight >= 0.04:
        return "secondary"
    if weight >= RARE_IMPORT_MIN_WEIGHT:
        return "rare_import"
    code = str(league_code or "").strip().upper()
    nat = normalize_nationality(nationality)
    for candidate, w in LEAGUE_NATIONALITY_WEIGHTS.get(code, []):
        if normalize_nationality(candidate) == nat:
            if w >= 0.12:
                return "primary"
            if w >= 0.04:
                return "secondary"
            if w >= RARE_IMPORT_MIN_WEIGHT:
                return "rare_import"
    return "invalid"

_TEAM_LOOKUP: Dict[str, Dict[str, str]] = {}
for code, meta in LEAGUE_REGISTRY.items():
    for team in meta["teams"]:
        for token in {team["key"], _norm_key(team["city"]), _norm_key(team["name"])}:
            if token and token not in _TEAM_LOOKUP:
                _TEAM_LOOKUP[token] = {
                    "league_code": code,
                    "league_display": meta["display"],
                    "team_name": team["name"],
                    "city": team["city"],
                }


def teams_for_league(league_code: str) -> List[Dict[str, str]]:
    meta = LEAGUE_REGISTRY.get(str(league_code or "").upper()) or LEAGUE_REGISTRY.get(str(league_code or ""))
    if not meta:
        return []
    return list(meta["teams"])


def resolve_league_for_team(team_name: str) -> Optional[Dict[str, str]]:
    raw = str(team_name or "").strip()
    if not raw:
        return None
    key = _norm_key(raw)
    if key in _TEAM_LOOKUP:
        return dict(_TEAM_LOOKUP[key])
    for token in raw.split():
        hit = _TEAM_LOOKUP.get(_norm_key(token))
        if hit:
            return dict(hit)
    partial = [v for k, v in _TEAM_LOOKUP.items() if key and (key in k or k in key)]
    if len(partial) == 1:
        return dict(partial[0])
    return None


def normalize_team_display(team_name: str, league_code: str = "") -> str:
    hit = resolve_league_for_team(team_name)
    if hit:
        return hit["team_name"]
    return str(team_name or "").strip()


def _fallback_league_for_nationality(nationality: str) -> Optional[str]:
    nat = normalize_nationality(nationality)
    best_code: Optional[str] = None
    best_weight = 0.0
    for code, pairs in LEAGUE_NATIONALITY_WEIGHTS.items():
        for candidate, weight in pairs:
            if normalize_nationality(candidate) == nat and float(weight) > best_weight:
                best_code = code
                best_weight = float(weight)
    if best_code and best_weight >= RARE_IMPORT_MIN_WEIGHT:
        return best_code
    for code, pairs in LEAGUE_NATIONALITY_WEIGHTS.items():
        for candidate, weight in pairs:
            if normalize_nationality(candidate) == nat:
                return code
    return None


def apply_prospect_league_team_fix(row: Dict[str, Any]) -> Dict[str, Any]:
    """Correct league/team mismatches on draft board rows (e.g. Rimouski + NCAA)."""
    import random

    out = dict(row)
    nat = normalize_nationality(out.get("nationality") or out.get("country") or "")
    code = str(out.get("league_code") or "").strip().upper()

    team_raw = str(out.get("team_name") or out.get("team") or "").strip()
    hit = resolve_league_for_team(team_raw)
    if hit:
        out["team_name"] = hit["team_name"]
        out["team"] = hit["team_name"]
        code = hit["league_code"]
        out["league_code"] = code

    legacy_display = str(out.get("league_display") or out.get("league") or "").strip()
    if legacy_display and ("Pro Jr" in legacy_display or re.search(r"\bjr\s*/", legacy_display, re.I)):
        code = str(out.get("league_code") or code).strip().upper()

    display = league_display_name(code)
    if display:
        out["league_code"] = code
        out["league_display"] = display
        out["league"] = display
        out["league_name"] = display

    tier = league_fit_tier(nat, code)
    if tier == "invalid" and nat:
        seed_key = str(out.get("key") or out.get("name") or team_raw or "0")
        rng = random.Random(abs(hash(seed_key)) & 0xFFFFFFFF)
        new_code = None
        for attempt in range(14):
            candidate = pick_league_for_nationality(
                rng,
                nat,
                allow_import=attempt >= 4,
            )
            if validate_prospect_league_fit(nat, candidate):
                new_code = candidate
                break
        if not new_code:
            new_code = _fallback_league_for_nationality(nat)
        teams = teams_for_league(new_code) if new_code else []
        if teams and new_code:
            team = teams[abs(hash(f"{seed_key}:{new_code}")) % len(teams)]
            out["league_code"] = new_code
            out["league_display"] = league_display_name(new_code)
            out["league"] = out["league_display"]
            out["league_name"] = out["league_display"]
            out["team_name"] = team["name"]
            out["team"] = team["name"]
            code = new_code
            tier = league_fit_tier(nat, code)

    out.pop("league_parent", None)
    out.pop("league_sub", None)
    out.pop("leagueParent", None)
    out.pop("leagueSub", None)

    out["league_fit_tier"] = tier
    out["is_import_story"] = tier == "rare_import"
    return out
