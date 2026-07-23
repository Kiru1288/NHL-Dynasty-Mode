"""
Deterministic CSS headshot metadata for franchise players and prospects.

Cosmetic only — does not affect ratings, contracts, or simulation outcomes.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

HEADSHOT_MIN = 1
HEADSHOT_MAX = 60

SKIN_TONES = (
    "fair",
    "light",
    "medium",
    "olive",
    "tan",
    "brown",
    "deep",
)

HAIR_STYLES = (
    "swept",
    "crop",
    "buzz",
    "curly",
    "red_part",
    "bald",
    "flow",
    "spiky",
    "flat_top",
    "afro",
    "fade",
    "long_blond",
    "messy_dark",
    "grey",
    "helmet",
)

HAIR_COLORS = (
    "black",
    "dark_brown",
    "brown",
    "auburn",
    "red",
    "blond",
    "dirty_blond",
    "grey",
    "white",
)

FACIAL_HAIR = (
    "none",
    "stubble",
    "beard",
    "playoff_beard",
    "mustache",
    "goatee",
    "grey_beard",
)

EXPRESSIONS = (
    "neutral",
    "smile",
    "serious",
    "focused",
    "confident",
    "tired",
    "angry",
)

AGE_BUCKETS = (
    "prospect",
    "rookie",
    "prime",
    "veteran",
    "legend",
)

GOALIE_HEADSHOTS = (10, 21, 22, 36, 37, 38, 52, 55)
VETERAN_HEADSHOTS = (8, 9, 15, 17, 18, 29, 33, 41, 48)
ROOKIE_HEADSHOTS = (2, 3, 16, 19, 23, 27, 31, 44, 49)
PROSPECT_HEADSHOTS = (16, 27, 31, 44, 49, 50)


def _pick(options: tuple, seed: int, salt: str = "") -> str:
    if not options:
        return ""
    material = f"{seed}|{salt}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return str(options[idx])


def derive_avatar_seed(
    *,
    player_id: str = "",
    full_name: str = "",
    birth_year: int = 0,
    age: int = 0,
    nationality: str = "",
    position: str = "",
    shoots: str = "",
    team_id: str = "",
) -> int:
    material = "|".join(
        [
            str(player_id or "").strip(),
            str(full_name or "").strip().lower(),
            str(int(birth_year or 0)),
            str(int(age or 0)),
            str(nationality or "").strip().lower(),
            str(position or "").strip().upper(),
            str(shoots or "").strip().upper(),
            str(team_id or "").strip(),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16)
    if seed <= 0:
        seed = 1
    return seed


def _age_bucket(age: int) -> str:
    a = int(age or 0)
    if a <= 19:
        return "prospect"
    if a <= 22:
        return "rookie"
    if a <= 30:
        return "prime"
    if a <= 36:
        return "veteran"
    return "legend"


def _nationality_code(nationality: str) -> str:
    raw = str(nationality or "").strip().upper()
    mapping = {
        "CANADA": "CAN",
        "CAN": "CAN",
        "UNITED STATES": "USA",
        "USA": "USA",
        "US": "USA",
        "SWEDEN": "SWE",
        "SWE": "SWE",
        "FINLAND": "FIN",
        "FIN": "FIN",
        "CZECH REPUBLIC": "CZE",
        "CZECHIA": "CZE",
        "CZE": "CZE",
        "SLOVAKIA": "SVK",
        "SVK": "SVK",
        "RUSSIA": "RUS",
        "RUS": "RUS",
        "GERMANY": "GER",
        "GER": "GER",
        "SWITZERLAND": "SUI",
        "SUI": "SUI",
        "AUSTRIA": "AUT",
        "AUT": "AUT",
        "NORWAY": "NOR",
        "NOR": "NOR",
        "DENMARK": "DEN",
        "DEN": "DEN",
        "LATVIA": "LAT",
        "LAT": "LAT",
        "BELARUS": "BLR",
        "BLR": "BLR",
        "UKRAINE": "UKR",
        "UKR": "UKR",
        "FRANCE": "FRA",
        "FRA": "FRA",
    }
    if raw in mapping:
        return mapping[raw]
    for key, code in mapping.items():
        if key in raw or raw in key:
            return code
    if len(raw) >= 3:
        return raw[:3]
    return raw or "NHL"


def _headshot_id_for_profile(
    seed: int,
    *,
    age: int,
    position: str,
    facial_hair: str,
) -> int:
    pos = str(position or "C").strip().upper()
    bucket = _age_bucket(age)

    if pos == "G":
        pool = GOALIE_HEADSHOTS
    elif bucket == "prospect":
        pool = PROSPECT_HEADSHOTS
    elif bucket == "rookie":
        pool = ROOKIE_HEADSHOTS
    elif bucket in ("veteran", "legend") or facial_hair in ("beard", "playoff_beard", "grey_beard", "mustache"):
        pool = VETERAN_HEADSHOTS
    else:
        pool = tuple(range(HEADSHOT_MIN, HEADSHOT_MAX + 1))

    idx = seed % len(pool)
    return int(pool[idx])


def generate_player_headshot_metadata(
    *,
    player_id: str = "",
    full_name: str = "",
    age: int = 20,
    birth_year: int = 0,
    nationality: str = "",
    position: str = "C",
    shoots: str = "L",
    team_id: str = "",
    avatar_seed: Optional[int] = None,
) -> Dict[str, Any]:
    seed = int(avatar_seed) if avatar_seed is not None else derive_avatar_seed(
        player_id=player_id,
        full_name=full_name,
        birth_year=birth_year,
        age=age,
        nationality=nationality,
        position=position,
        shoots=shoots,
        team_id=team_id,
    )

    age_bucket = _age_bucket(age)
    skin_tone = _pick(SKIN_TONES, seed, "skin")
    hair_style = _pick(HAIR_STYLES, seed, "hair_style")
    hair_color = _pick(HAIR_COLORS, seed, "hair_color")
    facial_hair = _pick(FACIAL_HAIR, seed, "facial_hair")

    if age_bucket in ("veteran", "legend") and facial_hair == "none" and (seed % 5) < 3:
        facial_hair = "grey_beard" if age_bucket == "legend" else "beard"
    if age_bucket == "prospect":
        facial_hair = "none" if facial_hair not in ("none", "stubble") else facial_hair

    expression = _pick(EXPRESSIONS, seed, "expression")
    headshot_id = _headshot_id_for_profile(
        seed,
        age=age,
        position=position,
        facial_hair=facial_hair,
    )

    return {
        "avatar_seed": int(seed),
        "headshot_id": int(headshot_id),
        "face_variant": int(headshot_id),
        "skin_tone": skin_tone,
        "hair_style": hair_style,
        "hair_color": hair_color,
        "facial_hair": facial_hair,
        "expression": expression,
        "age_bucket": age_bucket,
        "nationality_code": _nationality_code(nationality),
    }


def _player_identity_fields(player: Any) -> Dict[str, Any]:
    ident = getattr(player, "identity", None)
    if ident is None:
        return {
            "player_id": str(getattr(player, "id", "") or getattr(player, "player_id", "") or ""),
            "full_name": str(getattr(player, "name", "") or ""),
            "age": int(getattr(player, "age", 20) or 20),
            "birth_year": 0,
            "nationality": "",
            "position": "C",
            "shoots": "L",
            "team_id": str(getattr(player, "team_id", "") or getattr(player, "current_team_id", "") or ""),
        }

    pos = getattr(ident, "position", "C")
    if hasattr(pos, "value"):
        pos = pos.value
    shoots = getattr(ident, "shoots", "L")
    if hasattr(shoots, "value"):
        shoots = shoots.value

    return {
        "player_id": str(getattr(player, "id", "") or getattr(player, "player_id", "") or ""),
        "full_name": str(getattr(ident, "name", "") or getattr(player, "name", "") or ""),
        "age": int(getattr(ident, "age", 20) or 20),
        "birth_year": int(getattr(ident, "birth_year", 0) or 0),
        "nationality": str(getattr(ident, "birth_country", "") or getattr(ident, "nationality", "") or ""),
        "position": str(pos or "C"),
        "shoots": str(shoots or "L"),
        "team_id": str(
            getattr(player, "team_id", "")
            or getattr(player, "current_team_id", "")
            or getattr(getattr(player, "context", None), "current_team_id", "")
            or ""
        ),
    }


def headshot_fields_from_player(player: Any) -> Dict[str, Any]:
    return {
        "avatar_seed": int(getattr(player, "avatar_seed", 0) or 0),
        "headshot_id": int(getattr(player, "headshot_id", 0) or getattr(player, "face_variant", 0) or 0),
        "face_variant": int(getattr(player, "face_variant", 0) or getattr(player, "headshot_id", 0) or 0),
        "skin_tone": str(getattr(player, "skin_tone", "") or ""),
        "hair_style": str(getattr(player, "hair_style", "") or ""),
        "hair_color": str(getattr(player, "hair_color", "") or ""),
        "facial_hair": str(getattr(player, "facial_hair", "") or ""),
        "expression": str(getattr(player, "expression", "") or ""),
        "age_bucket": str(getattr(player, "age_bucket", "") or ""),
        "nationality_code": str(getattr(player, "nationality_code", "") or ""),
    }


def apply_headshot_to_player(player: Any, meta: Dict[str, Any]) -> None:
    for key, value in meta.items():
        try:
            setattr(player, key, value)
        except Exception:
            pass


def ensure_player_headshot(player: Any) -> Dict[str, Any]:
    existing_id = getattr(player, "headshot_id", None)
    existing_seed = getattr(player, "avatar_seed", None)
    if existing_id and existing_seed:
        return headshot_fields_from_player(player)

    fields = _player_identity_fields(player)
    meta = generate_player_headshot_metadata(
        player_id=fields["player_id"],
        full_name=fields["full_name"],
        age=fields["age"],
        birth_year=fields["birth_year"],
        nationality=fields["nationality"],
        position=fields["position"],
        shoots=fields["shoots"],
        team_id=fields["team_id"],
        avatar_seed=int(existing_seed) if existing_seed else None,
    )
    apply_headshot_to_player(player, meta)
    return meta


def attach_headshot_to_profile_dict(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return profile

    if profile.get("headshot_id") and profile.get("avatar_seed"):
        return profile

    meta = generate_player_headshot_metadata(
        player_id=str(profile.get("id") or profile.get("player_id") or ""),
        full_name=str(profile.get("name") or profile.get("full_name") or ""),
        age=int(profile.get("age") or 18),
        birth_year=int(profile.get("birth_year") or 0),
        nationality=str(profile.get("nationality") or profile.get("birth_country") or ""),
        position=str(profile.get("position") or "C"),
        shoots=str(profile.get("shoots") or profile.get("handedness") or "L"),
        team_id=str(profile.get("team_id") or profile.get("current_team_id") or ""),
        avatar_seed=profile.get("avatar_seed"),
    )
    out = dict(profile)
    out.update(meta)
    return out


def merge_headshot_into_row(row: Dict[str, Any], player: Any = None) -> Dict[str, Any]:
    out = dict(row or {})
    if player is not None:
        ensure_player_headshot(player)
        out.update(headshot_fields_from_player(player))
        return out

    if out.get("headshot_id") and out.get("avatar_seed"):
        return out

    meta = generate_player_headshot_metadata(
        player_id=str(out.get("player_id") or out.get("id") or ""),
        full_name=str(out.get("name") or out.get("player_name") or ""),
        age=int(out.get("age") or 20),
        birth_year=int(out.get("birth_year") or 0),
        nationality=str(out.get("nationality") or out.get("country") or out.get("birth_country") or ""),
        position=str(out.get("position") or "C"),
        shoots=str(out.get("shoots") or out.get("handedness") or "L"),
        team_id=str(out.get("team_id") or ""),
        avatar_seed=out.get("avatar_seed"),
    )
    out.update(meta)
    return out
