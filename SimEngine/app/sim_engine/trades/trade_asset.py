"""
Normalized trade asset models and package parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


def _safe_str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class PlayerTradeAsset:
    type: str = "player"
    player_id: str = ""
    source_team_id: str = ""
    acquiring_team_id: str = ""
    retained_pct: float = 0.0
    retained_cap_hit_m: float = 0.0
    player_name: str = ""
    ntc_waived: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def asset_id(self) -> str:
        return self.player_id


@dataclass
class DraftPickTradeAsset:
    type: str = "pick"
    pick_id: str = ""
    year: int = 0
    round: int = 0
    original_team_id: str = ""
    current_owner_team_id: str = ""
    acquiring_team_id: str = ""
    source_team_id: str = ""
    protection: Optional[str] = None
    conditions: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def asset_id(self) -> str:
        return self.pick_id


@dataclass
class RetainedSalaryRecord:
    player_id: str
    player_name: str
    retaining_team_id: str
    benefiting_team_id: str
    original_cap_hit_m: float
    retained_pct: float
    retained_cap_hit_m: float
    seasons_remaining: int = 1


@dataclass
class TradePackage:
    assets_by_team: Dict[str, List[Dict[str, Any]]]
    participating_team_ids: List[str] = field(default_factory=list)
    normalized_assets: List[Union[PlayerTradeAsset, DraftPickTradeAsset]] = field(default_factory=list)
    incoming_by_team: Dict[str, List[Union[PlayerTradeAsset, DraftPickTradeAsset]]] = field(default_factory=dict)
    outgoing_by_team: Dict[str, List[Union[PlayerTradeAsset, DraftPickTradeAsset]]] = field(default_factory=dict)


_LEGACY_PICK_RE = re.compile(
    r"^pick-(?P<team>[^-]+)-(?P<year>\d{4})-r(?P<round>\d+)(?:-\d+)?$",
    re.IGNORECASE,
)
_CANONICAL_PICK_RE = re.compile(
    r"^(?P<year>\d{4})-round(?P<round>\d+)-(?P<team>.+)$",
    re.IGNORECASE,
)


def canonical_pick_id(year: int, round_num: int, original_team_id: str) -> str:
    slug = _team_slug(original_team_id)
    return f"{int(year)}-round{int(round_num)}-{slug}"


def _team_slug(team_id: str) -> str:
    tid = _safe_str(team_id).strip()
    if len(tid) <= 6:
        return tid.upper()
    return tid[-6:].upper()


def resolve_pick_id(raw_id: str, source_team_id: str = "") -> str:
    """Map frontend legacy IDs to canonical registry IDs when possible."""
    rid = _safe_str(raw_id).strip()
    if not rid:
        return rid
    m = _CANONICAL_PICK_RE.match(rid)
    if m:
        return canonical_pick_id(int(m.group("year")), int(m.group("round")), m.group("team"))
    m = _LEGACY_PICK_RE.match(rid)
    if m:
        team = m.group("team") or source_team_id
        return canonical_pick_id(int(m.group("year")), int(m.group("round")), team)
    return rid


def normalize_trade_package(
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    *,
    team_by_id: Optional[Dict[str, Any]] = None,
) -> TradePackage:
    if not isinstance(assets_by_team, dict) or not assets_by_team:
        raise ValueError("Trade package is empty.")

    participating = sorted(str(k) for k in assets_by_team.keys() if str(k))
    if len(participating) < 2:
        raise ValueError("Trade package requires at least two teams.")

    normalized: List[Union[PlayerTradeAsset, DraftPickTradeAsset]] = []
    incoming_by_team: Dict[str, List[Any]] = {tid: [] for tid in participating}
    outgoing_by_team: Dict[str, List[Any]] = {tid: [] for tid in participating}

    for acq_tid, assets in assets_by_team.items():
        acquiring_id = _safe_str(acq_tid)
        if acquiring_id not in participating:
            participating.append(acquiring_id)
            incoming_by_team.setdefault(acquiring_id, [])
            outgoing_by_team.setdefault(acquiring_id, [])
        for raw in assets or []:
            if not isinstance(raw, dict):
                continue
            asset_type = _safe_str(raw.get("type")).lower()
            source_tid = _safe_str(raw.get("team") or raw.get("source_team_id"))
            if not source_tid or source_tid == acquiring_id:
                continue
            if team_by_id is not None and source_tid not in team_by_id:
                raise ValueError(f"Unknown source team in trade package: {source_tid}")
            if team_by_id is not None and acquiring_id not in team_by_id:
                raise ValueError(f"Unknown acquiring team in trade package: {acquiring_id}")

            if asset_type == "pick":
                raw_pick_id = _safe_str(raw.get("id") or raw.get("pick_id"))
                pick_id = resolve_pick_id(raw_pick_id, source_tid)
                asset = DraftPickTradeAsset(
                    pick_id=pick_id,
                    year=int(raw.get("year") or 0),
                    round=int(raw.get("round") or 0),
                    original_team_id=_safe_str(raw.get("original_team_id") or source_tid),
                    current_owner_team_id=source_tid,
                    acquiring_team_id=acquiring_id,
                    source_team_id=source_tid,
                    protection=raw.get("protection"),
                    conditions=raw.get("conditions"),
                    raw=dict(raw),
                )
            else:
                pid = _safe_str(raw.get("id") or raw.get("player_id"))
                if not pid:
                    continue
                retained_raw = _safe_float(raw.get("retained"), 0.0)
                if retained_raw < 0 or retained_raw > 50:
                    raise ValueError(
                        f"Retained salary for player {pid} must be between 0% and 50% (got {retained_raw}%)"
                    )
                retained = retained_raw
                waived = bool(
                    raw.get("ntc_waived")
                    or raw.get("ntcWaived")
                    or raw.get("clause_waived")
                )
                asset = PlayerTradeAsset(
                    player_id=pid,
                    source_team_id=source_tid,
                    acquiring_team_id=acquiring_id,
                    retained_pct=retained,
                    player_name=_safe_str(raw.get("name") or raw.get("player_name")),
                    ntc_waived=waived,
                    raw=dict(raw),
                )
            normalized.append(asset)
            incoming_by_team.setdefault(acquiring_id, []).append(asset)
            outgoing_by_team.setdefault(source_tid, []).append(asset)

    if not normalized:
        raise ValueError("Trade package contains no movable assets.")

    return TradePackage(
        assets_by_team=dict(assets_by_team),
        participating_team_ids=sorted(set(participating)),
        normalized_assets=normalized,
        incoming_by_team=incoming_by_team,
        outgoing_by_team=outgoing_by_team,
    )


def find_player_on_team_roster(team: Any, player_id: str) -> Tuple[Optional[Any], int]:
    """Active NHL roster only — not AHL/prospect assignments."""
    pid = _safe_str(player_id)
    roster = list(getattr(team, "roster", None) or [])
    for i, p in enumerate(roster):
        if _safe_str(getattr(p, "id", "")) == pid:
            return p, i
    return None, -1


def find_player_on_ahl_roster(team: Any, player_id: str) -> Tuple[Optional[Any], int]:
    pid = _safe_str(player_id)
    ahl = list(getattr(team, "ahl_roster", None) or [])
    for i, p in enumerate(ahl):
        if _safe_str(getattr(p, "id", "")) == pid:
            return p, i
    return None, -1


def player_trade_roster_location(team: Any, player_id: str) -> str:
    """Return 'nhl', 'ahl', or '' if the player is not on either list."""
    if find_player_on_team_roster(team, player_id)[0] is not None:
        return "nhl"
    if find_player_on_ahl_roster(team, player_id)[0] is not None:
        return "ahl"
    return ""


def player_display_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    name = getattr(ident, "name", None) if ident else None
    if name:
        return str(name)
    return _safe_str(getattr(player, "name", "Player"))


def team_id_of(team: Any) -> str:
    # team_id=0 is valid; only fall back to "id" when team_id is truly absent.
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    return _safe_str(tid)
