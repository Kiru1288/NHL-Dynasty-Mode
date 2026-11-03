"""Stats + advanced analytics block for the free-agent evaluation panel (D7).

Adds ``detail["fa_analytics"]`` to the free-agent detail payload:
  last_season  — basic line (GP/G/A/P/+/-/TOI or goalie line)
  advanced     — xG (ixG, xGF%), possession (CF%, FF%, HDCF%), finishing (Sh%),
                 rates (P/60), value (GAR, WAR), goalie GSAx / SV% / GAA
  career       — NHL totals plus the last five NHL seasons
Sources, in order: live season ledger (enriched with the same analytics as Stats
Central), then the archived season compact line, then career seasons.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging as _logging_swallow

_swallowed_log = _logging_swallow.getLogger(__name__)

_SKATER_ADV = (
    ("ixg", "ixG", 1), ("xgf_pct", "xGF%", 1), ("cf_pct", "CF%", 1), ("ff_pct", "FF%", 1),
    ("hdcf_pct", "HDCF%", 1), ("sh_pct", "Sh%", 1), ("pts_per_60", "P/60", 2), ("ixg_per_60", "ixG/60", 2),
    ("pdo", "PDO", 1), ("gar", "GAR", 1), ("war", "WAR", 1), ("offensive_gar", "Off GAR", 1),
    ("defensive_gar", "Def GAR", 1),
)
_GOALIE_ADV = (
    ("sv_pct", "SV%", 3), ("gaa", "GAA", 2), ("gsax", "GSAx", 1), ("war", "WAR", 1),
)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _i(v: Any) -> int:
    n = _num(v)
    return int(n) if n is not None else 0


def _pct_norm(key: str, v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if key.endswith("_pct") and key not in ("sv_pct",) and abs(v) <= 1.0:
        return v * 100.0
    return v


def _ledger_row(session: Any, pid: str) -> Dict[str, Any]:
    row = dict(((getattr(session, "player_season_stats", None) or {}).get(pid) or {}))
    if _i(row.get("gp")) > 0:
        return row
    snap = getattr(session, "preseason_player_stats_snapshot", None) or {}
    if isinstance(snap, dict) and isinstance(snap.get(pid), dict) and _i(snap[pid].get("gp")) > 0:
        return dict(snap[pid])
    return {}


def _archive_row(session: Any, pid: str) -> Dict[str, Any]:
    arch = getattr(session, "player_season_archive", None) or {}
    if not isinstance(arch, dict):
        return {}
    for key in sorted(arch.keys(), reverse=True):
        rows = arch.get(key) or {}
        if isinstance(rows, dict) and isinstance(rows.get(pid), dict):
            out = dict(rows[pid])
            out.setdefault("season_year", key)
            return out
    return {}


def _career(player: Any) -> Dict[str, Any]:
    cs = getattr(player, "career_stats", None)
    seasons = [s for s in (cs.get("seasons") or [])] if isinstance(cs, dict) else []
    nhl = [s for s in seasons if isinstance(s, dict) and str(s.get("league") or "NHL").upper() == "NHL"]
    # Drop per-team splits when a TOT line exists for that season.
    tot_seasons = {str(s.get("season")) for s in nhl if str(s.get("team") or "").upper() == "TOT"}
    rows = [s for s in nhl if str(s.get("season")) not in tot_seasons or str(s.get("team") or "").upper() == "TOT"]
    rows.sort(key=lambda s: str(s.get("season") or ""))
    goalie = any(s.get("sv_pct") is not None or s.get("wins") is not None for s in rows)
    totals: Dict[str, Any] = {"seasons": len({str(s.get("season")) for s in rows}), "gp": sum(_i(s.get("gp")) for s in rows)}
    if goalie:
        totals["wins"] = sum(_i(s.get("wins") or s.get("w")) for s in rows)
        totals["shutouts"] = sum(_i(s.get("shutouts") or s.get("so")) for s in rows)
    else:
        g = sum(_i(s.get("g") or s.get("goals")) for s in rows)
        a = sum(_i(s.get("a") or s.get("assists")) for s in rows)
        totals.update({"g": g, "a": a, "pts": g + a})
        if totals["gp"]:
            totals["ppg"] = round((g + a) / totals["gp"], 2)
    recent = []
    for s in rows[-5:][::-1]:
        recent.append({
            "season": s.get("season"),
            "team": s.get("team") or s.get("team_abbrev"),
            "gp": _i(s.get("gp")),
            "g": s.get("g") if s.get("g") is not None else s.get("goals"),
            "a": s.get("a") if s.get("a") is not None else s.get("assists"),
            "pts": s.get("pts") if s.get("pts") is not None else s.get("points"),
            "plus_minus": s.get("plus_minus") if s.get("plus_minus") is not None else s.get("pm"),
            "sv_pct": s.get("sv_pct"),
            "gaa": s.get("gaa"),
            "wins": s.get("wins"),
        })
    return {"is_goalie": goalie, "totals": totals, "recent": recent}


def attach_fa_analytics(session: Any, player: Any, detail: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pid = str(detail.get("player_id") or detail.get("id") or getattr(player, "id", "") or "")
        raw = _ledger_row(session, pid)
        source = "season_ledger" if raw else ""
        enriched: Dict[str, Any] = {}
        if raw:
            try:
                from app.sim_engine.generation.player_analytics import enrich_player_row

                enriched = dict(enrich_player_row(raw) or {})
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
                enriched = dict(raw)
        else:
            enriched = _archive_row(session, pid)
            source = "season_archive" if enriched else ""
        pos = str(enriched.get("position") or detail.get("position") or "").upper()
        goalie = pos.startswith("G")
        last: Dict[str, Any] = {}
        if enriched:
            gp = _i(enriched.get("gp") or enriched.get("games_played"))
            if goalie:
                last = {
                    "gp": gp, "w": enriched.get("wins") or enriched.get("w"),
                    "sv_pct": enriched.get("sv_pct") or enriched.get("save_pct"),
                    "gaa": enriched.get("gaa"), "so": enriched.get("shutouts") or enriched.get("so"),
                }
            else:
                toi = _num(enriched.get("toi_sec"))
                last = {
                    "gp": gp,
                    "g": enriched.get("g") if enriched.get("g") is not None else enriched.get("goals"),
                    "a": enriched.get("a") if enriched.get("a") is not None else enriched.get("assists"),
                    "pts": enriched.get("pts") if enriched.get("pts") is not None else enriched.get("points"),
                    "plus_minus": enriched.get("plus_minus"),
                    "shots": enriched.get("shots") or enriched.get("sog"),
                    "hits": enriched.get("hits"),
                    "blocks": enriched.get("blocks") or enriched.get("blk"),
                    "toi": (f"{int(toi / gp // 60)}:{int(toi / gp % 60):02d}" if toi and gp else None),
                }
            last["season"] = enriched.get("season") or enriched.get("season_year")
        adv: List[Dict[str, Any]] = []
        for key, label, dp in (_GOALIE_ADV if goalie else _SKATER_ADV):
            v = _pct_norm(key, _num(enriched.get(key)))
            if v is None:
                continue
            if key == "pdo" and v < 2:
                v *= 1000.0
            adv.append({"key": key, "label": label, "value": round(v, dp)})
        # Quick read for the GM: what kind of player the numbers say he is.
        reads: List[str] = []
        if not goalie and enriched:
            xgf = _pct_norm("xgf_pct", _num(enriched.get("xgf_pct")))
            cf = _pct_norm("cf_pct", _num(enriched.get("cf_pct")))
            sh = _pct_norm("sh_pct", _num(enriched.get("sh_pct")))
            war = _num(enriched.get("war"))
            if xgf is not None and xgf >= 53:
                reads.append("Drives play: his team wins the expected-goals battle with him on the ice.")
            elif xgf is not None and xgf <= 46:
                reads.append("Gets outchanced: expected goals tilt the wrong way with him out there.")
            if cf is not None and cf >= 53:
                reads.append("Strong possession numbers (CF%).")
            if sh is not None and sh >= 16:
                reads.append("Shooting % is high — some regression risk on his goal total.")
            elif sh is not None and 0 < sh <= 7:
                reads.append("Shooting % is low — goals could bounce back.")
            if war is not None and war >= 2.0:
                reads.append("Top-of-roster value by WAR.")
        detail["fa_analytics"] = {
            "source": source or None,
            "is_goalie": goalie,
            "last_season": last,
            "advanced": adv,
            "reads": reads,
            "career": _career(player),
        }
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return detail
