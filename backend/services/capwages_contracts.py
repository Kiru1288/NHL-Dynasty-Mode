"""Fill 1-year cap-sheet stubs from CapWages years-remaining.

Spotrac's cap sheet stamps every active row as one year. CapWages publishes the
years still left on the real deal, counted from the 2026-27 season. Used only
when the stored contract has no multi-year grid and was not signed in-game.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _slug(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _parse_years_remaining(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    m = re.search(r"(\d+)\s*(UFA|RFA)?", text, re.I)
    if not m:
        return None
    years = int(m.group(1))
    rights = (m.group(2) or "").upper() or None
    return {"years": years, "rights": rights}


def fetch_capwages_term(name: str, *, timeout: float = 12.0) -> Optional[Dict[str, Any]]:
    slug = _slug(name)
    if not slug:
        return None
    url = f"https://capwages.com/players/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", "replace")
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
    )
    if not m:
        return None
    data = json.loads(m.group(1))
    player = ((data.get("props") or {}).get("pageProps") or {}).get("player") or {}
    parsed = _parse_years_remaining(player.get("yearsRemaining"))
    if not parsed:
        return None
    parsed["terms"] = str(player.get("terms") or "").strip()
    parsed["terms_details"] = str(player.get("termsDetails") or "").strip()
    parsed["slug"] = slug
    return parsed


def repair_stub_terms_from_capwages(
    players: list,
    season_year: int,
    *,
    season_already_played: bool,
    cache: Dict[str, Any],
) -> int:
    """Update 1-year stubs in place. `cache` is name-key → report or {'miss': True}."""
    from services.contract_economy import (
        _player_name,
        forward_years_from_reported,
        stamp_forward_contract_term,
    )
    from services.real_nhl_contracts import normalize_player_name

    stubs = []
    for player in players:
        c = getattr(player, "contract", None)
        if not isinstance(c, dict):
            continue
        if c.get("user_signed"):
            continue
        src = str(c.get("source") or "").lower()
        if src in ("signed", "re_sign", "ufa", "free_agent", "fa", "arbitration"):
            continue
        hits = [h for h in (c.get("season_cap_hits") or []) if float(h or 0) > 0.05]
        years = int(c.get("years_remaining") or 0)
        if len(hits) > 1 or years > 1:
            continue
        name = _player_name(player)
        key = normalize_player_name(name)
        if not key:
            continue
        stubs.append((player, c, key, name))

    missing = [row for row in stubs if row[2] not in cache]
    if missing:
        def _one(name: str):
            try:
                return fetch_capwages_term(name)
            except Exception:
                _log.debug("capwages term fetch failed for %s", name, exc_info=True)
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_one, name): key for _p, _c, key, name in missing}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    report = fut.result()
                except Exception:
                    report = None
                cache[key] = report if isinstance(report, dict) else {"miss": True}

    changed = 0
    for player, contract, key, _name in stubs:
        report = cache.get(key)
        if not isinstance(report, dict) or report.get("miss") or not report.get("years"):
            continue
        forward = forward_years_from_reported(
            int(report["years"]),
            int(season_year),
            season_already_played=season_already_played,
        )
        if forward <= 0:
            continue
        if stamp_forward_contract_term(
            contract,
            player,
            forward,
            int(season_year),
            season_already_played=season_already_played,
        ):
            if report.get("rights") in ("UFA", "RFA"):
                contract["rights_status"] = report["rights"]
                contract["rights_source"] = "capwages"
            changed += 1
    return changed
