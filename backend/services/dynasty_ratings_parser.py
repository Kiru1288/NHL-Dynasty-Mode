"""
Parse dynasty_ratings.txt — authoritative chapter ratings for NHL / AHL / prospects.

Format mirrors chapter_attributes.py (overall, character, offence, … / goalie chapters).
Optional dynasty_ratings_patches.txt applies last-minute overall overrides (83 → 85).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

DYNASTY_RATINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "dynasty_ratings.txt"
DYNASTY_PATCHES_PATH = Path(__file__).resolve().parent.parent / "data" / "dynasty_ratings_patches.txt"

TEAM_HEADER_RE = re.compile(r"^===\s*(.+?)\s*===$")
SECTION_RE = re.compile(r"^--\s*(.+?)\s*--$", re.IGNORECASE)

SKATER_LINE_RE = re.compile(
    r"^(.+?):\s*(\d+)\s*ovr,\s*(\d+)\s*cha,\s*(\d+)\s*off,\s*(\d+)\s*def,\s*"
    r"(\d+)\s*tra,\s*(\d+)\s*men,\s*(\d+)\s*phy,\s*(\d+)\s*pot\s*$",
    re.IGNORECASE,
)
GOALIE_LINE_RE = re.compile(
    r"^(.+?):\s*(\d+)\s*ovr,\s*(\d+)\s*glove,\s*(\d+)\s*blocker,\s*(\d+)\s*stick,\s*(\d+)\s*pot\s*$",
    re.IGNORECASE,
)
PATCH_LINE_RE = re.compile(r"^\s*(\d+)\s*(?:→|->|=>)\s*(\d+)\s*$")
POSITION_TOKENS = frozenset({"C", "LW", "RW", "D", "G", "F", "LD", "RD", "W"})

# Full header name (normalized) -> NHL abbr
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "anaheim ducks": "ANA",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "calgary flames": "CGY",
    "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "columbus blue jackets": "CBJ",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "nashville predators": "NSH",
    "new jersey devils": "NJD",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "san jose sharks": "SJS",
    "seattle kraken": "SEA",
    "st. louis blues": "STL",
    "st louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "utah mammoth": "UTA",
    "utah hockey club": "UTA",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "washington capitals": "WSH",
    "winnipeg jets": "WPG",
}


def normalize_player_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _team_abbr_from_header(header: str) -> str:
    key = normalize_player_name(header.replace(".", ""))
    abbr = TEAM_NAME_TO_ABBR.get(key)
    if abbr:
        return abbr
    # Last-resort: first three letters (unused fallback)
    return key[:3].upper() if key else ""


def _parse_level_from_section(section: str) -> Tuple[str, str]:
    s = str(section or "").strip().upper()
    if "NHL FORWARD" in s:
        return "nhl", "forward"
    if "NHL DEFENCE" in s or "NHL DEFENSE" in s:
        return "nhl", "defense"
    if "NHL GOALIE" in s:
        return "nhl", "goalie"
    if s.startswith("AHL"):
        return "ahl", "mixed"
    if s.startswith("ECHL"):
        return "echl", "mixed"
    if "PROSPECT" in s:
        return "prospect", "mixed"
    return "", ""


def _looks_like_position_hint(paren: str) -> bool:
    raw = str(paren or "").strip().upper()
    if not raw:
        return False
    if raw in POSITION_TOKENS:
        return True
    if "/" in raw:
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        return all(p in POSITION_TOKENS or p in {"W", "F"} for p in parts)
    return False


def _parse_name_field(raw: str) -> Optional[Tuple[str, List[str], Optional[str]]]:
    text = str(raw or "").strip()
    if not text or "[see" in text.lower():
        return None
    if text.lower().startswith("rated above"):
        return None

    primary = text
    position_hint: Optional[str] = None
    aliases: List[str] = []

    paren_m = re.search(r"\(([^)]+)\)", text)
    if paren_m:
        paren = paren_m.group(1).strip()
        primary = text[: paren_m.start()].strip()
        if _looks_like_position_hint(paren):
            position_hint = paren
        else:
            aliases.append(paren)

    keys: Set[str] = set()
    if primary:
        keys.add(normalize_player_name(primary))
        parts = primary.split()
        if parts:
            keys.add(normalize_player_name(parts[-1]))
    for alias in aliases:
        keys.add(normalize_player_name(alias))

    if not primary or not keys:
        return None
    return primary, sorted(keys), position_hint


@dataclass
class DynastyRatingEntry:
    raw_name: str
    lookup_keys: List[str]
    team_abbr: str
    level: str
    position_hint: Optional[str]
    chapters: Dict[str, int]
    is_goalie: bool
    patched: bool = False


@dataclass
class DynastyRatingsRegistry:
    entries: List[DynastyRatingEntry] = field(default_factory=list)
    by_team_level: Dict[Tuple[str, str], List[DynastyRatingEntry]] = field(default_factory=dict)
    by_name: Dict[str, List[DynastyRatingEntry]] = field(default_factory=dict)
    parse_stats: Dict[str, Any] = field(default_factory=dict)

    def match_player(
        self,
        team_abbr: str,
        level: str,
        full_name: str,
        *,
        last_name: str = "",
    ) -> Optional[DynastyRatingEntry]:
        abbr = str(team_abbr or "").upper()
        lvl = str(level or "nhl").lower()
        pool = list(self.by_team_level.get((abbr, lvl), []))

        full_key = normalize_player_name(full_name)
        last_key = normalize_player_name(last_name or (full_name.split()[-1] if full_name else ""))

        if full_key:
            primary_hits = [
                e for e in pool if normalize_player_name(e.raw_name) == full_key
            ]
            if len(primary_hits) == 1:
                return primary_hits[0]
            key_hits = [e for e in pool if full_key in e.lookup_keys]
            if len(key_hits) == 1:
                only = key_hits[0]
                primary_norm = normalize_player_name(only.raw_name)
                if " " not in full_key and " " in primary_norm:
                    pass
                else:
                    return only
            for entry in self.by_name.get(full_key, []):
                if entry.team_abbr == abbr and entry.level == lvl:
                    primary_norm = normalize_player_name(entry.raw_name)
                    if " " not in full_key and " " in primary_norm:
                        continue
                    return entry

        if last_key:
            first_key = ""
            if full_key and " " in full_key:
                first_key = full_key.split()[0]
            if first_key:
                for entry in pool:
                    if last_key not in entry.lookup_keys:
                        continue
                    entry_first = normalize_player_name(
                        (entry.raw_name or "").split()[0] if entry.raw_name else ""
                    )
                    if entry_first == first_key:
                        return entry
                # Txt rows are often last-name only ("Peterka: 86 ovr"). Those never have a
                # first name to compare, so accept them when the last name is unique on this team.
                last_only = [
                    e for e in pool
                    if last_key in e.lookup_keys and " " not in normalize_player_name(e.raw_name or "")
                ]
                if len(last_only) == 1:
                    return last_only[0]
            elif not full_key:
                last_hits = [e for e in pool if last_key in e.lookup_keys]
                if len(last_hits) == 1:
                    return last_hits[0]
                global_last = [
                    e for e in self.by_name.get(last_key, [])
                    if e.team_abbr == abbr and e.level == lvl
                ]
                if len(global_last) == 1:
                    return global_last[0]

        if lvl == "nhl" and full_key and " " in full_key:
            for alt_lvl in ("ahl", "prospect", "echl"):
                alt_pool = list(self.by_team_level.get((abbr, alt_lvl), []))
                for entry in alt_pool:
                    if full_key in entry.lookup_keys:
                        return entry
        return None

    def match_anywhere(
        self,
        full_name: str,
        *,
        last_name: str = "",
        claimed: Optional[set] = None,
        unmatched_last_counts: Optional[Dict[str, int]] = None,
    ) -> Optional[DynastyRatingEntry]:
        """League-wide fallback for players the txt lists under another club (trades,
        stale team headers). Exact full-name matches win; a last-name-only row is used
        only when that last name is unique among unclaimed txt rows AND among the still
        unmatched players, so two Hughes can never share one rating."""
        taken = claimed or set()
        full_key = normalize_player_name(full_name)
        last_key = normalize_player_name(last_name or (full_name.split()[-1] if full_name else ""))
        if full_key and " " in full_key:
            exact = [
                e for e in self.by_name.get(full_key, [])
                if id(e) not in taken and normalize_player_name(e.raw_name or "") == full_key
            ]
            if len(exact) == 1:
                return exact[0]
        if last_key:
            rows = [e for e in self.by_name.get(last_key, []) if id(e) not in taken]
            last_only = [e for e in rows if " " not in normalize_player_name(e.raw_name or "")]
            if (
                len(rows) == 1
                and len(last_only) == 1
                and int((unmatched_last_counts or {}).get(last_key, 1)) == 1
            ):
                return last_only[0]
        return None

    def entries_for_team(self, team_abbr: str, level: str) -> List[DynastyRatingEntry]:
        return list(self.by_team_level.get((str(team_abbr or "").upper(), str(level or "").lower()), []))


def _index_entry(registry: DynastyRatingsRegistry, entry: DynastyRatingEntry) -> None:
    registry.entries.append(entry)
    key = (entry.team_abbr, entry.level)
    registry.by_team_level.setdefault(key, []).append(entry)
    for lk in entry.lookup_keys:
        registry.by_name.setdefault(lk, []).append(entry)


def parse_dynasty_ratings(text: str) -> DynastyRatingsRegistry:
    registry = DynastyRatingsRegistry()
    current_team = ""
    current_abbr = ""
    current_level = ""
    skipped = 0
    parsed = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"=+", line):
            continue
        if line.lower().startswith("format:") or line.lower().startswith("goalies:"):
            continue
        if line.lower().startswith("nhl dynasty mode"):
            continue

        team_m = TEAM_HEADER_RE.match(line)
        if team_m:
            current_team = team_m.group(1).strip()
            current_abbr = _team_abbr_from_header(current_team)
            continue

        sec_m = SECTION_RE.match(line)
        if sec_m:
            current_level, _group = _parse_level_from_section(sec_m.group(1))
            continue

        if line.lower() in ("forwards", "defence", "defense", "goalies"):
            continue

        if not current_abbr or not current_level:
            continue

        sk_m = SKATER_LINE_RE.match(line)
        g_m = GOALIE_LINE_RE.match(line)
        if not sk_m and not g_m:
            continue

        name_raw = (sk_m or g_m).group(1).strip()
        parsed_name = _parse_name_field(name_raw)
        if not parsed_name:
            skipped += 1
            continue

        primary, lookup_keys, position_hint = parsed_name
        if sk_m:
            chapters = {
                "overall": int(sk_m.group(2)),
                "character": int(sk_m.group(3)),
                "offence": int(sk_m.group(4)),
                "defence": int(sk_m.group(5)),
                "transition": int(sk_m.group(6)),
                "mental": int(sk_m.group(7)),
                "physical": int(sk_m.group(8)),
                "potential": int(sk_m.group(9)),
            }
            is_goalie = False
        else:
            chapters = {
                "overall": int(g_m.group(2)),
                "glove": int(g_m.group(3)),
                "blocker": int(g_m.group(4)),
                "stick": int(g_m.group(5)),
                "potential": int(g_m.group(6)),
            }
            is_goalie = True

        entry = DynastyRatingEntry(
            raw_name=primary,
            lookup_keys=lookup_keys,
            team_abbr=current_abbr,
            level=current_level,
            position_hint=position_hint,
            chapters=chapters,
            is_goalie=is_goalie,
        )
        _index_entry(registry, entry)
        parsed += 1

    registry.parse_stats = {
        "parsed": parsed,
        "skipped": skipped,
        "teams": len({e.team_abbr for e in registry.entries}),
    }
    return registry


def apply_overall_patches(registry: DynastyRatingsRegistry, patch_text: str) -> int:
    """Apply Name + old→new blocks; returns count of patches applied."""
    if not patch_text.strip():
        return 0

    pending_name: Optional[str] = None
    applied = 0

    def _apply_to_name(name: str, new_ovr: int) -> None:
        nonlocal applied
        key = normalize_player_name(name)
        hits = registry.by_name.get(key, [])
        if not hits:
            # try last name
            last = normalize_player_name(name.split()[-1] if name else "")
            hits = registry.by_name.get(last, [])
        for entry in hits:
            old = int(entry.chapters.get("overall") or new_ovr)
            pot = entry.chapters.get("potential")
            if pot is not None:
                pot = int(pot)
                if new_ovr < old:
                    # A downgrade lowers the ceiling by the same amount (never below the new OVR).
                    pot = max(int(new_ovr), pot - (old - int(new_ovr)))
                pot = max(pot, int(new_ovr))
                entry.chapters["potential"] = pot
            entry.chapters["overall"] = int(new_ovr)
            entry.patched = True
            applied += 1

    for raw_line in patch_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            pending_name = None
            continue
        if re.fullmatch(r"\d+", line):
            continue
        patch_m = PATCH_LINE_RE.match(line)
        if patch_m and pending_name:
            _apply_to_name(pending_name, int(patch_m.group(2)))
            pending_name = None
            continue
        if patch_m:
            continue
        if ":" in line:
            continue
        pending_name = line

    return applied


_REGISTRY_CACHE: Optional[DynastyRatingsRegistry] = None
_REGISTRY_CACHE_STAMP: Optional[tuple] = None


def _ratings_files_stamp() -> tuple:
    out = []
    for p in (DYNASTY_RATINGS_PATH, DYNASTY_PATCHES_PATH):
        try:
            out.append(p.stat().st_mtime_ns)
        except OSError:
            out.append(0)
    return tuple(out)


def load_dynasty_ratings_registry(
    ratings_path: Optional[Path] = None,
    patches_path: Optional[Path] = None,
) -> DynastyRatingsRegistry:
    global _REGISTRY_CACHE, _REGISTRY_CACHE_STAMP
    stamp = _ratings_files_stamp() if ratings_path is None and patches_path is None else None
    if (
        ratings_path is None
        and patches_path is None
        and _REGISTRY_CACHE is not None
        and stamp == _REGISTRY_CACHE_STAMP
    ):
        return _REGISTRY_CACHE
    rp = ratings_path or DYNASTY_RATINGS_PATH
    pp = patches_path or DYNASTY_PATCHES_PATH
    text = rp.read_text(encoding="utf-8")
    registry = parse_dynasty_ratings(text)
    if pp.exists():
        patch_text = pp.read_text(encoding="utf-8")
        n = apply_overall_patches(registry, patch_text)
        registry.parse_stats["patches_applied"] = n
    if ratings_path is None and patches_path is None:
        _REGISTRY_CACHE = registry
        _REGISTRY_CACHE_STAMP = stamp
    return registry


def _team_abbr_from_team_obj(team: Any) -> str:
    if team is None:
        return ""
    for attr in ("abbr", "code", "abbreviation", "short_name", "team_abbr"):
        val = getattr(team, attr, None)
        if val:
            return str(val).upper()
    return ""


def _roster_level_from_kind(roster_kind: str) -> str:
    rk = str(roster_kind or "").lower()
    if rk == "ahl":
        return "ahl"
    if rk == "echl":
        return "echl"
    if rk in ("prospect", "prospects"):
        return "prospect"
    return "nhl"


def _chapter_values_flat(chapters: Dict[str, int]) -> bool:
    vals = [int(v) for v in (chapters or {}).values() if v is not None]
    return len(vals) >= 3 and len(set(vals)) <= 1


def ensure_dynasty_chapter_profile_on_player(
    player: Any,
    *,
    team: Any = None,
    roster_kind: str = "",
) -> bool:
    """Re-attach dynasty_ratings.txt chapters when profile is missing or collapsed to one number."""
    try:
        from app.sim_engine.entities.chapter_attributes import get_player_chapters
    except Exception:
        return False

    chapters = get_player_chapters(player)
    if chapters and not _chapter_values_flat(chapters):
        return False

    tagged = bool(
        getattr(player, "dynasty_ratings_import", False)
        or str(getattr(player, "real_nhl_rating_note", "") or "") == "dynasty_txt"
    )
    if not chapters and not tagged:
        return False
    if chapters and _chapter_values_flat(chapters) and not tagged:
        return False

    registry = load_dynasty_ratings_registry()
    ident = getattr(player, "identity", None)
    name = str(getattr(ident, "name", None) or getattr(player, "name", "") or "")
    if not name.strip():
        return False
    abbr = _team_abbr_from_team_obj(team)
    level = _roster_level_from_kind(roster_kind)
    last = name.split()[-1] if name else ""
    entry = registry.match_player(abbr, level, name, last_name=last)
    if entry is None and level != "nhl":
        entry = registry.match_player(abbr, "nhl", name, last_name=last)
    if entry is None:
        return False

    apply_dynasty_entry_to_player(
        player,
        entry,
        seed=getattr(player, "id", None) or getattr(player, "player_id", None),
        align_rounds=6,
    )
    return True


def apply_dynasty_entry_to_player(
    player: Any,
    entry: DynastyRatingEntry,
    *,
    seed: Optional[int] = None,
    align_rounds: int = 40,
) -> None:
    """Attach chapter profile from txt and sync legacy sim ratings."""
    from app.sim_engine.entities.chapter_attributes import (
        ensure_player_attribute_profile,
        sync_legacy_ratings_from_profile,
    )
    from app.sim_engine.entities.player import persist_recomputed_ovr

    ensure_player_attribute_profile(
        player,
        chapters=dict(entry.chapters),
        seed=seed,
        regenerate=True,
    )
    sync_legacy_ratings_from_profile(player, overwrite=True)
    try:
        player._invalidate_ovr_memo()
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    target = float(entry.chapters.get("overall", 75)) / 99.0
    target = max(0.30, min(0.99, target))
    # persist_recomputed_ovr between nudges — a stale ovr() memo makes this
    # loop slam NHL stars to 99 and leave AHL/juniors on the 40 floor.
    from services.real_nhl_roster_importer import align_attribute_ovr_to_target

    align_attribute_ovr_to_target(player, target, rounds=max(1, int(align_rounds)))
    persist_recomputed_ovr(player)
    try:
        player._invalidate_ovr_memo()
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    setattr(player, "dynasty_ratings_import", True)
    setattr(player, "_dynasty_entry_key", f"{entry.team_abbr}|{entry.level}|{entry.raw_name}")
    setattr(player, "dynasty_rating_source", entry.level)
    setattr(player, "real_nhl_rating_note", "dynasty_txt")


def _position_from_hint(hint: Optional[str], *, is_goalie: bool = False) -> Any:
    from app.sim_engine.entities.player import Position

    if is_goalie:
        return Position.G
    raw = str(hint or "F").upper().split("/")[0].strip()
    mapping = {
        "C": Position.C,
        "LW": Position.LW,
        "RW": Position.RW,
        "W": Position.RW,
        "F": Position.C,
        "D": Position.D,
        "LD": Position.D,
        "RD": Position.D,
        "G": Position.G,
    }
    return mapping.get(raw, Position.C)


_SPAWN_BIO_REGISTRY_CACHE: Dict[int, Any] = {}


def _spawn_bio_registry(as_of_year: int) -> Any:
    year = int(as_of_year or 2026)
    cached = _SPAWN_BIO_REGISTRY_CACHE.get(year)
    if cached is not None:
        return cached
    from services.player_bio_parser import load_player_bio_registry

    reg = load_player_bio_registry(as_of=date(year, 9, 15))
    _SPAWN_BIO_REGISTRY_CACHE[year] = reg
    return reg


def spawn_player_from_dynasty_entry(
    entry: DynastyRatingEntry,
    *,
    rng: Any,
    pool_context: str,
    used_names: Set[str],
    league_players: List[Any],
    as_of_year: Optional[int] = None,
    align_rounds: int = 40,
    apply_bio: bool = True,
    bio_registry: Optional[Any] = None,
) -> Any:
    """Create a named player from a dynasty txt entry (AHL / prospect pools)."""
    from app.sim_engine.engine import build_role_shaped_ratings
    from app.sim_engine.entities.player import (
        BackstoryType,
        BackstoryUpbringing,
        DevResources,
        IdentityBio,
        Player,
        PressureLevel,
        Shoots,
        SupportLevel,
        UpbringingType,
        assign_skater_archetype,
    )

    pos = _position_from_hint(entry.position_hint, is_goalie=entry.is_goalie)
    target_ovr = float(entry.chapters.get("overall", 70)) / 99.0
    target_ovr = max(0.30, min(0.92, target_ovr))
    archetype = assign_skater_archetype(pos, rng)
    ratings = build_role_shaped_ratings(position=pos, target_ovr=target_ovr, rng=rng)
    age_lo, age_hi = (18, 22) if pool_context == "prospect" else (20, 28)
    age = rng.randint(int(age_lo), int(age_hi))
    try:
        year = int(as_of_year or 0)
    except (TypeError, ValueError):
        year = 0
    if year < 2000:
        year = date.today().year
    birth_year = int(year) - int(age)
    seed = rng.randint(1, 2_000_000_000)
    name = str(entry.raw_name)
    used_names.add(name)
    identity = IdentityBio(
        name=name,
        age=age,
        birth_year=birth_year,
        birth_country="CAN",
        birth_city="Unknown",
        height_cm=185,
        weight_kg=86,
        position=pos,
        shoots=Shoots.L if rng.random() < 0.58 else Shoots.R,
        draft_year=max(2015, birth_year + 18),
        draft_round=2,
        draft_pick=15,
    )
    backstory = BackstoryUpbringing(
        backstory=BackstoryType.GRINDER,
        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
        family_support=SupportLevel.MEDIUM,
        early_pressure=PressureLevel.MODERATE,
        dev_resources=DevResources.LOCAL,
    )
    player = Player(
        identity=identity,
        backstory=backstory,
        ratings=ratings,
        rng_seed=seed,
        archetype=archetype,
        pool_context=pool_context,
        enforce_floor_on_init=False,
    )
    apply_dynasty_entry_to_player(player, entry, seed=seed, align_rounds=align_rounds)
    if apply_bio:
        try:
            from services.player_bio_parser import apply_player_bio_by_name

            registry = bio_registry or _spawn_bio_registry(int(year))
            apply_player_bio_by_name(player, registry, as_of_year=year)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    league_players.append(player)
    return player
