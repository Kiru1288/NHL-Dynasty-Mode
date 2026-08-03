"""
Spotrac multi-year cap tables → AAV / years remaining / UFA-RFA for Real NHL import.

Unofficial scrape of public HTML. Soft-fails per team so import still succeeds.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT_S = 35

# NHL abbr → Spotrac team slug
SPOTRAC_TEAM_SLUGS: Dict[str, str] = {
    "ANA": "anaheim-ducks",
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars",
    "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers",
    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens",
    "NSH": "nashville-predators",
    "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken",
    "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WPG": "winnipeg-jets",
    "WSH": "washington-capitals",
}


def _http_get_text(url: str, *, timeout: float = HTTP_TIMEOUT_S) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def normalize_player_name(name: str) -> str:
    text = str(name or "").lower().strip()
    text = (
        text.replace("á", "a")
        .replace("à", "a")
        .replace("ä", "a")
        .replace("å", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("ï", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ö", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
        .replace("ç", "c")
        .replace("ř", "r")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("ć", "c")
        .replace("č", "c")
        .replace("ď", "d")
        .replace("ť", "t")
        .replace("ý", "y")
        .replace("ů", "u")
        .replace("ľ", "l")
        .replace("ł", "l")
        .replace("ø", "o")
        .replace("æ", "ae")
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_money_sort(raw: str) -> Optional[float]:
    try:
        v = float(str(raw).replace(",", "").strip())
    except Exception:
        return None
    # Cap hits are in absolute dollars on Spotrac data-sort.
    if v >= 500_000:
        return v / 1_000_000.0
    return None


def _parse_yearly_team_html(html: str) -> Dict[str, Dict[str, Any]]:
    """Return name_key → contract dict from a Spotrac multi-year team page."""
    out: Dict[str, Dict[str, Any]] = {}
    # Prefer the first large player table body after a "Player (" header.
    marker = html.find("Player (")
    if marker < 0:
        marker = 0
    tbody_s = html.find("<tbody", marker)
    if tbody_s < 0:
        return out
    tbody_e = html.find("</tbody>", tbody_s)
    if tbody_e < 0:
        return out
    tbody = html[tbody_s:tbody_e]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody, re.S)
    for row in rows:
        pl = re.search(
            r'nhl/player/_/id/(\d+)/([^"]+)"[^>]*>\s*([^<]+)',
            row,
        )
        if not pl:
            continue
        display = pl.group(3).strip()
        key = normalize_player_name(display)
        if not key:
            continue
        sorts = re.findall(r'data-sort="([^"]*)"', row)
        year_hits: List[float] = []
        for s in sorts:
            m = _parse_money_sort(s)
            if m is not None:
                year_hits.append(m)
        if not year_hits:
            continue
        aav_m = round(float(year_hits[0]), 3)
        years_remaining = len(year_hits)
        rights = "UFA"
        row_text = re.sub(r"<[^>]+>", " ", row)
        if re.search(r"\bRFA\b", row_text):
            rights = "RFA"
        elif re.search(r"\bUFA\b", row_text):
            rights = "UFA"
        nmc = bool(re.search(r"\bNMC\b", row_text))
        ntc = bool(re.search(r"\bNTC\b", row_text)) and not nmc
        # ELC heuristic: low AAV + short remaining term on young deals.
        ctype = "ELC" if aav_m <= 1.0 and years_remaining <= 3 else "STANDARD"
        entry = {
            "name": display,
            "spotrac_id": int(pl.group(1)),
            "aav_m": aav_m,
            "cap_hit_m": aav_m,
            "years_remaining": years_remaining,
            "years": years_remaining,
            "rights_status": rights,
            "contract_type": ctype,
            "no_move_clause": nmc,
            "no_trade_clause": ntc,
            "source": "real_nhl_spotrac",
        }
        # Same display name can appear twice on one club (e.g. Elias Pettersson C/D).
        existing = out.get(key)
        if existing is None:
            out[key] = entry
        elif isinstance(existing, list):
            if not any(int(x.get("spotrac_id") or 0) == entry["spotrac_id"] for x in existing):
                existing.append(entry)
        else:
            if int(existing.get("spotrac_id") or 0) != entry["spotrac_id"]:
                out[key] = [existing, entry]
    return out


def fetch_team_contracts(abbr: str, season_year: int) -> Dict[str, Any]:
    slug = SPOTRAC_TEAM_SLUGS.get(str(abbr or "").upper())
    if not slug:
        return {}
    year = int(season_year)
    urls = [
        f"https://www.spotrac.com/nhl/{slug}/yearly/_/year/{year}",
        f"https://www.spotrac.com/nhl/{slug}/yearly/_/year/{year + 1}",
        f"https://www.spotrac.com/nhl/{slug}/yearly",
    ]
    # Utah rename fallback
    if abbr.upper() == "UTA":
        urls.extend(
            [
                f"https://www.spotrac.com/nhl/utah-hockey-club/yearly/_/year/{year}",
                "https://www.spotrac.com/nhl/utah-hockey-club/yearly",
            ]
        )
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            html = _http_get_text(url)
            parsed = _parse_yearly_team_html(html)
            if parsed:
                return parsed
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return {}


def fetch_league_contracts_by_team(
    abbreviations: List[str],
    season_year: int,
    *,
    max_workers: int = 10,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
    """
    Returns (by_abbr → name_key → contract, failures).
    """
    by_team: Dict[str, Dict[str, Dict[str, Any]]] = {}
    failures: List[str] = []
    abbrs = [str(a).upper() for a in abbreviations if a]

    def _one(abbr: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        return abbr, fetch_team_contracts(abbr, season_year)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, a) for a in abbrs]
        for fut in as_completed(futs):
            try:
                abbr, parsed = fut.result()
                by_team[abbr] = parsed
                if not parsed:
                    failures.append(f"{abbr}: empty Spotrac yearly table")
            except Exception as e:
                failures.append(f"spotrac: {e}")
    return by_team, failures


def match_contract_for_player(
    player_name: str,
    team_abbr: str,
    contracts_by_team: Dict[str, Dict[str, Any]],
    *,
    prefer_higher_aav: Optional[bool] = None,
    position_code: str = "",
) -> Optional[Dict[str, Any]]:
    team_map = contracts_by_team.get(str(team_abbr or "").upper()) or {}
    if not team_map:
        return None
    key = normalize_player_name(player_name)
    hit = team_map.get(key)

    def _pick(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return dict(candidates[0])
        # Duplicate names on one club: prefer the AAV that matches role.
        # Star centres / scorers → higher AAV; depth / D namesakes → lower AAV.
        ranked = sorted(candidates, key=lambda c: float(c.get("aav_m") or 0), reverse=True)
        pos = str(position_code or "").upper()
        if prefer_higher_aav is True or pos in ("C", "LW", "RW", "L", "R", "W"):
            # When both are cheap ELCs, still return the higher of the two.
            return dict(ranked[0])
        if prefer_higher_aav is False or pos in ("D", "G"):
            # Defencemen / goalies sharing a forward's name usually have the lower hit.
            return dict(ranked[-1])
        # Unknown role: prefer non-ELC / higher AAV (stars are more often missing).
        non_elc = [c for c in ranked if float(c.get("aav_m") or 0) >= 2.0]
        return dict((non_elc or ranked)[0])

    if isinstance(hit, list):
        return _pick(hit)
    if isinstance(hit, dict):
        return dict(hit)
    # Soft fallback: unique last-name match on the club.
    parts = key.split()
    if not parts:
        return None
    last = parts[-1]
    candidates: List[Dict[str, Any]] = []
    for k, v in team_map.items():
        if not (k.endswith(" " + last) or k == last):
            continue
        if isinstance(v, list):
            candidates.extend(v)
        elif isinstance(v, dict):
            candidates.append(v)
    if len(candidates) == 1:
        return dict(candidates[0])
    if len(candidates) > 1 and key.count(" ") >= 1:
        # Only auto-resolve last-name collisions when first names also collide-free.
        return None
    return None


def is_real_nhl_contract(player: Any) -> bool:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        src = str(c.get("source") or "")
    else:
        src = str(getattr(c, "source", "") or "") if c is not None else ""
    if not src:
        src = str(getattr(player, "contract_source", "") or "")
    return src.startswith("real_nhl") or bool(getattr(player, "real_nhl_contract", False))
