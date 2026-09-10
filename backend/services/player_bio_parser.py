"""
Parse player_bios.txt — authoritative age, DOB, height, weight, and nationality.

Applied to dynasty-spawned AHL/prospect pools and matched onto real NHL imports
by normalized name so trade value, body sim, archetype inference, and WJC
eligibility use real biographical data instead of random placeholders.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from services.dynasty_ratings_parser import normalize_player_name

PLAYER_BIOS_PATH = Path(__file__).resolve().parent.parent / "data" / "player_bios.txt"
ROOT_BIOS_PATH = Path(__file__).resolve().parent.parent.parent / "age.txt"

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

NATIONALITY_ALIASES: Dict[str, str] = {
    "can": "Canada",
    "canada": "Canada",
    "usa": "USA",
    "u.s.a": "USA",
    "u.s.": "USA",
    "u.s": "USA",
    "united states": "USA",
    "america": "USA",
    "swe": "Sweden",
    "sweden": "Sweden",
    "sverige": "Sweden",
    "fin": "Finland",
    "finland": "Finland",
    "suomi": "Finland",
    "cze": "Czechia",
    "czech": "Czechia",
    "czechia": "Czechia",
    "czech republic": "Czechia",
    "svk": "Slovakia",
    "slovakia": "Slovakia",
    "ger": "Germany",
    "germany": "Germany",
    "deutschland": "Germany",
    "sui": "Switzerland",
    "switzerland": "Switzerland",
    "swiss": "Switzerland",
    "den": "Denmark",
    "denmark": "Denmark",
    "lat": "Latvia",
    "latvia": "Latvia",
    "latvian": "Latvia",
    "rus": "Russia",
    "russia": "Russia",
    "belarus": "Belarus",
    "blr": "Belarus",
    "ukraine": "Ukraine",
    "ukr": "Ukraine",
    "norway": "Norway",
    "nor": "Norway",
    "austria": "Austria",
    "aut": "Austria",
    "hungary": "Hungary",
    "hun": "Hungary",
    "italy": "Italy",
    "ita": "Italy",
    "lithuania": "Lithuania",
    "ltu": "Lithuania",
    "poland": "Poland",
    "pol": "Poland",
    "kazakhstan": "Kazakhstan",
    "kaz": "Kazakhstan",
    "france": "France",
    "fra": "France",
    "uk": "UK",
    "united kingdom": "UK",
    "england": "UK",
    "japan": "Japan",
    "south korea": "South Korea",
    "china": "China",
    "australia": "Australia",
    "mexico": "Mexico",
    "brazil": "Brazil",
}

WJC_COUNTRY_CODES: Dict[str, str] = {
    "Canada": "CAN",
    "USA": "USA",
    "Russia": "RUS",
    "Sweden": "SWE",
    "Finland": "FIN",
    "Czechia": "CZE",
    "Slovakia": "SVK",
    "Germany": "GER",
    "Switzerland": "SUI",
    "Denmark": "DEN",
    "Latvia": "LAT",
}

# Authoritative WJC nation pool (code, display label).
WJC_COUNTRY_META: List[Tuple[str, str]] = [
    ("CAN", "Canada"),
    ("USA", "United States"),
    ("RUS", "Russia"),
    ("SWE", "Sweden"),
    ("FIN", "Finland"),
    ("CZE", "Czechia"),
    ("SVK", "Slovakia"),
    ("GER", "Germany"),
    ("SUI", "Switzerland"),
    ("DEN", "Denmark"),
    ("LAT", "Latvia"),
]

WJC_POOL_CODES: frozenset = frozenset(c for c, _ in WJC_COUNTRY_META)

# Non-pool birth countries merged into the nearest WJC federation for tournament assignment.
WJC_FEDERATION_MERGE: Dict[str, str] = {
    "Belarus": "RUS",
    "Kazakhstan": "RUS",
    "Norway": "DEN",
    "Austria": "GER",
    "France": "SUI",
    "UK": "DEN",
    "Lithuania": "LAT",
    "Ukraine": "LAT",
    "Hungary": "SVK",
    "Poland": "CZE",
    "Italy": "SUI",
}

# WJC code → name_generator nationality key.
WJC_CODE_TO_NAME_NAT: Dict[str, str] = {
    "CAN": "Canada",
    "USA": "USA",
    "RUS": "Russia",
    "SWE": "Sweden",
    "FIN": "Finland",
    "CZE": "Czechia",
    "SVK": "Slovakia",
    "GER": "Germany",
    "SUI": "Switzerland",
    "DEN": "Denmark",
    "LAT": "Latvia",
}

# Junior-league hints → WJC federation (dual-citizen tie-break + row resolution).
JUNIOR_LEAGUE_WJC_HINTS: Tuple[Tuple[str, str], ...] = (
    ("OHL", "CAN"),
    ("WHL", "CAN"),
    ("QMJHL", "CAN"),
    ("CHL", "CAN"),
    ("USHL", "USA"),
    ("NCAA", "USA"),
    ("NTDP", "USA"),
    ("J20", "SWE"),
    ("NATIONELL", "SWE"),
    ("SHL", "SWE"),
    ("LIIGA", "FIN"),
    ("SM-SARJA", "FIN"),
    ("MHL", "RUS"),
    ("KHL", "RUS"),
    ("VHL", "RUS"),
    ("DEL", "GER"),
    ("NL", "SUI"),
    ("SWISS", "SUI"),
    ("CZECH", "CZE"),
    ("EXTRALIGA", "CZE"),
    ("SLOVAK", "SVK"),
    ("LATV", "LAT"),
    ("LHL", "LAT"),
    ("DENMARK", "DEN"),
    ("NORWAY", "DEN"),
    ("METAL", "LAT"),
)

WJC_ROSTER_MIN = 14
WJC_ROSTER_MAX = 22
WJC_REAL_MIN_BEFORE_FILLER = 8

BIO_SUFFIX_RE = re.compile(
    r"\|\s*Age:\s*(\d+)\s*\|\s*DOB:\s*([^|]+?)\s*\|\s*Height:\s*([^|]+?)\s*"
    r"\|\s*Weight:\s*([^|]+?)\s*\|\s*Nationality:\s*(.+?)\s*$",
    re.IGNORECASE,
)
PIPE_BIO_RE = re.compile(
    r"^(.+?)\s*(?:\([^)]+\))?\s*:\s*Age\s+(\d+)\s*\|\s*DOB:\s*([^|]+?)\s*"
    r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$",
    re.IGNORECASE,
)
TAB_BIO_RE = re.compile(
    r"^([^\t]+)\t(\d{1,2})\t([^\t]+)\t([^\t]+)\t(\d+\s*lbs?)\t([^\t]+?)\s*$",
    re.IGNORECASE,
)
TURNS_AGE_RE = re.compile(r"turns\s+(\d+)", re.IGNORECASE)
EMOJI_RE = re.compile(
    r"[\U0001F1E0-\U0001F1FF\U0001F300-\U0001FAFF\U00002700-\U000027BF]+"
)


@dataclass
class PlayerBioEntry:
    raw_name: str
    lookup_keys: List[str]
    age: int
    birth_year: int
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None
    height_cm: int = 0
    weight_kg: int = 0
    nationality: str = "Canada"
    nationalities: List[str] = field(default_factory=list)
    wjc_country: str = ""


@dataclass
class PlayerBioRegistry:
    entries: List[PlayerBioEntry] = field(default_factory=list)
    by_name: Dict[str, List[PlayerBioEntry]] = field(default_factory=dict)
    parse_stats: Dict[str, int] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[PlayerBioEntry]:
        key = normalize_player_name(name)
        hits = self.by_name.get(key) or []
        if hits:
            return hits[-1]
        parts = key.split()
        if len(parts) >= 2:
            last = normalize_player_name(parts[-1])
            hits = self.by_name.get(last) or []
            if len(hits) == 1:
                return hits[0]
        return None


def _strip_noise(text: str) -> str:
    s = EMOJI_RE.sub("", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_nationality(raw: str) -> str:
    s = _strip_noise(raw).lower()
    s = re.sub(r"[^a-z0-9\s/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "Canada"
    token = s.split("/")[0].strip()
    if token in NATIONALITY_ALIASES:
        return NATIONALITY_ALIASES[token]
    for key, val in NATIONALITY_ALIASES.items():
        if key in token or token in key:
            return val
    return token.title() if token else "Canada"


def parse_nationalities(raw: str) -> List[str]:
    s = _strip_noise(raw)
    if not s:
        return ["Canada"]
    parts = re.split(r"[/,]", s)
    out: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        nat = normalize_nationality(part)
        if nat and nat not in seen:
            seen.add(nat)
            out.append(nat)
    return out or ["Canada"]


def parse_height_cm(raw: str) -> Optional[int]:
    s = str(raw or "").strip().lower()
    if not s:
        return None
    nums = re.findall(r"(\d+)['\"]", s)
    if len(nums) >= 2:
        ft, inch = int(nums[0]), int(nums[1])
        return int(round((ft * 12 + inch) * 2.54))
    m = re.search(r"(\d+)['\"]\s*(\d+)", s)
    if m:
        ft, inch = int(m.group(1)), int(m.group(2))
        return int(round((ft * 12 + inch) * 2.54))
    m = re.search(r"(\d+)['\"]", s)
    if m:
        ft = int(m.group(1))
        inch_m = re.search(r"['\"]\s*(\d+)", s)
        inch = int(inch_m.group(1)) if inch_m else 0
        return int(round((ft * 12 + inch) * 2.54))
    return None


def parse_weight_kg(raw: str) -> Optional[int]:
    s = str(raw or "").strip().lower()
    m = re.search(r"(\d+)\s*lbs?", s)
    if not m:
        return None
    lbs = int(m.group(1))
    return int(round(lbs * 0.453592))


def _month_num(token: str) -> Optional[int]:
    t = str(token or "").strip().lower().replace(".", "")[:3]
    return MONTHS.get(t)


def parse_dob(raw: str) -> Optional[Tuple[int, int, int]]:
    s = _strip_noise(raw).strip()
    if not s:
        return None
    iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if iso:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    m = re.match(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m:
        mo = _month_num(m.group(1))
        if mo:
            return int(m.group(3)), mo, int(m.group(2))
    return None


def _age_from_dob(dob: Tuple[int, int, int], as_of: date) -> int:
    y, m, d = dob
    age = as_of.year - y
    if (as_of.month, as_of.day) < (m, d):
        age -= 1
    return max(15, age)


def wjc_eligibility_cutoff(season_sy: int) -> date:
    """IIHF U20 cutoff — player must be under 20 on January 4 (season_sy+1)."""
    return date(int(season_sy) + 1, 1, 4)


def _birth_parts_from_source(source: Any) -> Optional[Tuple[int, int, int]]:
    if source is None:
        return None
    bd = getattr(source, "birth_date", None)
    if bd is None and isinstance(source, dict):
        bd = source.get("birth_date")
    if isinstance(bd, str) and bd.count("-") >= 2:
        try:
            y, m, d = (int(x) for x in bd.split("-")[:3])
            if y > 1900:
                return y, m, d
        except (TypeError, ValueError):
            pass
    ident = getattr(source, "identity", None) if not isinstance(source, dict) else None
    row = source if isinstance(source, dict) else {}
    by = int(
        getattr(ident, "birth_year", 0)
        or row.get("birth_year")
        or getattr(source, "birth_year", 0)
        or 0
    )
    bm = int(getattr(ident, "birth_month", 0) or row.get("birth_month") or 0)
    bday = int(getattr(ident, "birth_day", 0) or row.get("birth_day") or 0)
    if by > 1900 and bm and bday:
        return by, bm, bday
    if by > 1900:
        return by, 7, 1
    return None


def player_age_on(source: Any, as_of: date) -> int:
    dob = _birth_parts_from_source(source)
    if dob:
        return _age_from_dob(dob, as_of)
    if isinstance(source, dict):
        try:
            return int(source.get("age") or 99)
        except (TypeError, ValueError):
            return 99
    ident = getattr(source, "identity", None)
    if ident is not None:
        try:
            return int(getattr(ident, "age", 99) or 99)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(source, "age", 99) or 99)
    except (TypeError, ValueError):
        return 99


def wjc_age_eligible(source: Any, season_sy: int) -> bool:
    """True when the player is 19 or younger on the Jan 4 eligibility cutoff."""
    cutoff = wjc_eligibility_cutoff(season_sy)
    return player_age_on(source, cutoff) < 20


def wjc_country_label(code: str) -> str:
    for c, lab in WJC_COUNTRY_META:
        if c == str(code or "").upper():
            return lab
    return str(code or "?")


def wjc_code_from_junior_league(league: str) -> str:
    """Map a junior league label/code to a WJC federation when unambiguous."""
    blob = str(league or "").strip().upper()
    if not blob:
        return ""
    for hint, code in JUNIOR_LEAGUE_WJC_HINTS:
        if hint in blob:
            return code
    return ""


def merge_wjc_federation_code(nationality: str) -> str:
    """Return WJC pool code for a nationality, including non-pool merge targets."""
    nat = normalize_nationality(nationality)
    if nat in WJC_COUNTRY_CODES:
        return WJC_COUNTRY_CODES[nat]
    merged = WJC_FEDERATION_MERGE.get(nat)
    if merged:
        if merged in WJC_POOL_CODES:
            return merged
        if merged in WJC_COUNTRY_CODES:
            return WJC_COUNTRY_CODES[merged]
    return ""


def resolve_wjc_country_for_player(player: Any, rng: Any = None) -> str:
    """Unified federation pick — preset wjc_country, dual passport, then birth country."""
    preset = str(getattr(player, "wjc_country", "") or "")
    if preset in WJC_POOL_CODES:
        return preset
    ident = getattr(player, "identity", None)
    bc = str(getattr(ident, "birth_country", "") or getattr(player, "birth_country", "") or "")
    dual = getattr(player, "dual_nationality", None)
    junior_league = str(
        getattr(player, "junior_league", "")
        or getattr(player, "league", "")
        or getattr(player, "league_name", "")
        or getattr(ident, "junior_league", "")
        or ""
    )
    if isinstance(dual, list) and dual:
        code = resolve_wjc_country_code(
            bc, nationalities=dual, rng=rng, junior_league=junior_league
        )
        if code:
            return code
    return resolve_wjc_country_code(bc, rng=rng, junior_league=junior_league)


def resolve_wjc_country_code(
    nationality: str,
    *,
    nationalities: Optional[List[str]] = None,
    rng: Any = None,
    junior_league: str = "",
) -> str:
    """Pick IIHF WJC federation from nationality string (dual citizens prefer WJC-eligible)."""
    nats = list(nationalities or parse_nationalities(nationality))
    eligible_codes: List[str] = []
    for n in nats:
        code = merge_wjc_federation_code(n)
        if code and code not in eligible_codes:
            eligible_codes.append(code)
    if not eligible_codes:
        return merge_wjc_federation_code(nationality)
    if len(eligible_codes) == 1:
        return eligible_codes[0]

    league_code = wjc_code_from_junior_league(junior_league)
    if league_code and league_code in eligible_codes:
        return league_code

    # Prefer the smaller federation when still ambiguous (Latvia over Canada, etc.).
    pool_order = [c for c, _ in WJC_COUNTRY_META]
    ordered = sorted(eligible_codes, key=lambda c: pool_order.index(c) if c in pool_order else 99)
    if len(ordered) >= 2 and rng is not None:
        # Weight toward smaller nations but allow either passport.
        weights = [max(1, 12 - pool_order.index(c)) for c in ordered]
        pick = rng.choices(ordered, weights=weights, k=1)[0]
        return pick
    return ordered[0]


def _lookup_keys(name: str) -> List[str]:
    primary = str(name or "").strip()
    keys = {normalize_player_name(primary)}
    parts = primary.split()
    if len(parts) >= 2:
        keys.add(normalize_player_name(parts[-1]))
    return sorted(keys)


def _index_entry(registry: PlayerBioRegistry, entry: PlayerBioEntry) -> None:
    registry.entries.append(entry)
    for lk in entry.lookup_keys:
        registry.by_name.setdefault(lk, []).append(entry)


def _build_entry(
    *,
    name: str,
    age: int,
    dob_raw: str,
    height_raw: str,
    weight_raw: str,
    nationality_raw: str,
    as_of: date,
) -> Optional[PlayerBioEntry]:
    name = _strip_noise(name)
    if not name or len(name) < 3:
        return None
    dob = parse_dob(dob_raw)
    birth_year = int(as_of.year) - int(age)
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None
    if dob:
        birth_year, birth_month, birth_day = dob
        age = _age_from_dob(dob, as_of)
    h_cm = parse_height_cm(height_raw)
    w_kg = parse_weight_kg(weight_raw)
    if not h_cm or not w_kg:
        return None
    nats = parse_nationalities(nationality_raw)
    nat = nats[0]
    entry = PlayerBioEntry(
        raw_name=name,
        lookup_keys=_lookup_keys(name),
        age=int(age),
        birth_year=int(birth_year),
        birth_month=birth_month,
        birth_day=birth_day,
        height_cm=int(h_cm),
        weight_kg=int(w_kg),
        nationality=nat,
        nationalities=nats,
        wjc_country=resolve_wjc_country_code(nat, nationalities=nats),
    )
    return entry


def _parse_dynasty_bio_line(line: str, as_of: date) -> Optional[PlayerBioEntry]:
    m = BIO_SUFFIX_RE.search(line)
    if not m:
        return None
    prefix = line[: m.start()].strip()
    name_part = prefix.split(":", 1)[0].strip()
    name_part = re.sub(r"\([^)]+\)\s*$", "", name_part).strip()
    return _build_entry(
        name=name_part,
        age=int(m.group(1)),
        dob_raw=m.group(2),
        height_raw=m.group(3),
        weight_raw=m.group(4),
        nationality_raw=m.group(5),
        as_of=as_of,
    )


def _parse_pipe_bio_line(line: str, as_of: date) -> Optional[PlayerBioEntry]:
    m = PIPE_BIO_RE.match(line)
    if not m:
        return None
    age = int(m.group(2))
    turns = TURNS_AGE_RE.search(m.group(0))
    if turns:
        age = int(turns.group(1))
    return _build_entry(
        name=m.group(1),
        age=age,
        dob_raw=m.group(3),
        height_raw=m.group(4),
        weight_raw=m.group(5),
        nationality_raw=m.group(6),
        as_of=as_of,
    )


def _parse_tab_bio_line(line: str, as_of: date) -> Optional[PlayerBioEntry]:
    if "\t" not in line:
        return None
    m = TAB_BIO_RE.match(line)
    if m:
        name = re.sub(r"^Nationality", "", m.group(1), flags=re.I).strip()
        if name.lower() not in ("player", "age", "birthday", "height", "weight", "nationality"):
            return _build_entry(
                name=name,
                age=int(m.group(2)),
                dob_raw=m.group(3),
                height_raw=m.group(4),
                weight_raw=m.group(5),
                nationality_raw=m.group(6),
                as_of=as_of,
            )

    # Merged header rows: Player\tAge\t...\tNationalityName\t22\tDOB\t...
    parts = [p.strip() for p in line.split("\t") if p.strip()]
    age_idx = None
    for i, part in enumerate(parts):
        if re.fullmatch(r"\d{1,2}", part):
            try:
                age_val = int(part)
            except ValueError:
                continue
            if 15 <= age_val <= 45 and i + 4 < len(parts):
                age_idx = i
                break
    if age_idx is None:
        return None
    name_raw = parts[age_idx - 1] if age_idx > 0 else ""
    name_raw = re.sub(r"^Nationality", "", name_raw, flags=re.I).strip()
    if not name_raw or name_raw.lower() in ("player", "age", "birthday", "height", "weight", "nationality"):
        return None
    return _build_entry(
        name=name_raw,
        age=int(parts[age_idx]),
        dob_raw=parts[age_idx + 1],
        height_raw=parts[age_idx + 2],
        weight_raw=parts[age_idx + 3],
        nationality_raw=parts[age_idx + 4] if age_idx + 4 < len(parts) else parts[-1],
        as_of=as_of,
    )


def parse_player_bios(text: str, *, as_of: Optional[date] = None) -> PlayerBioRegistry:
    registry = PlayerBioRegistry()
    as_of = as_of or date(2026, 9, 15)
    skipped = 0
    parsed = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("format:") or low.startswith("copy-ready"):
            continue
        if line.startswith("===") or line.startswith("--"):
            continue
        if re.match(r"^player\tage\tbirthday\theight\tweight\tnationality\s*$", low):
            continue
        if low in ("forwards", "defence", "defense", "goalies", "prospects"):
            continue
        if "independently confirm" in low or "organizational prospect list" in low:
            continue

        entry = (
            _parse_dynasty_bio_line(line, as_of)
            or _parse_pipe_bio_line(line, as_of)
            or _parse_tab_bio_line(line, as_of)
        )
        if entry is None:
            skipped += 1
            continue
        _index_entry(registry, entry)
        parsed += 1

    registry.parse_stats = {
        "parsed": parsed,
        "skipped": skipped,
        "unique_names": len({e.raw_name for e in registry.entries}),
    }
    return registry


def load_player_bio_registry(
    bios_path: Optional[Path] = None,
    *,
    as_of: Optional[date] = None,
) -> PlayerBioRegistry:
    path = bios_path or PLAYER_BIOS_PATH
    if not path.exists():
        path = ROOT_BIOS_PATH
    text = path.read_text(encoding="utf-8")
    return parse_player_bios(text, as_of=as_of)


def apply_player_bio_to_player(
    player: Any,
    entry: PlayerBioEntry,
    *,
    as_of_year: Optional[int] = None,
    reapply_body: bool = True,
) -> bool:
    """Apply authoritative bio fields to a live player entity."""
    ident = getattr(player, "identity", None)
    if ident is None:
        return False

    as_of = date(int(as_of_year or 2026), 9, 15)
    age = int(entry.age)
    if entry.birth_month and entry.birth_day:
        age = _age_from_dob((entry.birth_year, entry.birth_month, entry.birth_day), as_of)

    try:
        ident.age = age
        ident.birth_year = int(entry.birth_year)
        ident.birth_country = str(entry.nationality)
        ident.height_cm = int(entry.height_cm)
        ident.weight_kg = int(entry.weight_kg)
        if entry.birth_month:
            ident.birth_month = int(entry.birth_month)
        if entry.birth_day:
            ident.birth_day = int(entry.birth_day)
    except Exception:
        return False

    try:
        player.age = age
    except Exception:
        pass

    if entry.birth_month and entry.birth_day:
        bd = f"{entry.birth_year:04d}-{entry.birth_month:02d}-{entry.birth_day:02d}"
        setattr(player, "birth_date", bd)

    total_in = int(round(entry.height_cm / 2.54))
    setattr(player, "height_cm", int(entry.height_cm))
    setattr(player, "weight_kg", int(entry.weight_kg))
    setattr(player, "height_in", total_in)
    setattr(player, "weight_lb", int(round(entry.weight_kg / 0.453592)))
    setattr(player, "nationality", entry.nationality)
    setattr(player, "birth_country", entry.nationality)
    setattr(player, "player_bio_import", True)
    if entry.nationalities:
        setattr(player, "dual_nationality", entry.nationalities)
    if entry.wjc_country:
        setattr(player, "wjc_country", entry.wjc_country)
        setattr(player, "international_country", entry.wjc_country)

    if reapply_body:
        try:
            from app.sim_engine.generation.prospect_body import apply_body_tradeoffs_to_ratings
            import random

            seed = abs(hash(str(getattr(player, "id", "") or entry.raw_name))) & 0xFFFFFFFF
            apply_body_tradeoffs_to_ratings(player, random.Random(seed))
        except Exception:
            pass
        try:
            from app.sim_engine.generation.prospect_identity import refresh_player_identity

            refresh_player_identity(player, force=True)
        except Exception:
            pass

    return True


def apply_player_bio_by_name(
    player: Any,
    registry: PlayerBioRegistry,
    *,
    as_of_year: Optional[int] = None,
) -> bool:
    ident = getattr(player, "identity", None)
    name = str(getattr(ident, "name", "") or getattr(player, "name", "") or "")
    entry = registry.lookup(name)
    if entry is None:
        return False
    return apply_player_bio_to_player(player, entry, as_of_year=as_of_year)


def apply_player_bios_to_league(league: Any, *, as_of_year: Optional[int] = None) -> Dict[str, int]:
    """Match all league players to bio registry by name."""
    registry = load_player_bio_registry(as_of=date(int(as_of_year or 2026), 9, 15))
    applied = 0
    players = list(getattr(league, "players", None) or [])
    for team in getattr(league, "teams", None) or []:
        players.extend(getattr(team, "roster", None) or [])
        players.extend(getattr(team, "ahl_roster", None) or [])
        players.extend(getattr(team, "echl_roster", None) or [])
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            players.extend(tm.get("players") or [])
    players.extend(getattr(league, "free_agents", None) or [])
    players.extend(getattr(league, "overseas_free_agents", None) or [])

    seen_ids: Set[str] = set()
    for p in players:
        pid = str(getattr(p, "id", id(p)))
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if apply_player_bio_by_name(p, registry, as_of_year=as_of_year):
            applied += 1

    return {"applied": applied, **registry.parse_stats}
