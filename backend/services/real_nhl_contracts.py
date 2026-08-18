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


def _spotrac_team_slug(abbr: str) -> Optional[str]:
    return SPOTRAC_TEAM_SLUGS.get(str(abbr or "").upper())


def _parse_cap_table_rows(html: str, table_id: str) -> List[Dict[str, Any]]:
    """Parse one Spotrac cap-page table (active / dead / retained)."""
    marker = html.find(f'id="{table_id}"')
    if marker < 0:
        return []
    tbody_s = html.find("<tbody", marker)
    if tbody_s < 0:
        return []
    tbody_e = html.find("</tbody>", tbody_s)
    if tbody_e < 0:
        return []
    tbody = html[tbody_s:tbody_e]
    out: List[Dict[str, Any]] = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody, re.S):
        pl = re.search(
            r'nhl/player/_/id/(\d+)/([^"]+)"[^>]*>\s*([^<]+)',
            row,
        )
        if pl:
            display = pl.group(3).strip()
            spotrac_id = int(pl.group(1))
        else:
            pl2 = re.search(r">([A-ZÀ-ÖØ-Þ][^<]{1,60})</a>", row)
            if not pl2:
                continue
            display = pl2.group(1).strip()
            spotrac_id = 0
        key = normalize_player_name(display)
        if not key:
            continue
        sorts = re.findall(r'data-sort="([^"]*)"', row)
        money = [m for m in (_parse_money_sort(s) for s in sorts) if m is not None]
        if not money:
            continue
        # Cap Hit Total / Adjusted are the leading money sorts on these tables.
        aav_m = round(float(money[0]), 3)
        out.append(
            {
                "name": display,
                "name_key": key,
                "spotrac_id": spotrac_id,
                "aav_m": aav_m,
                "cap_hit_m": aav_m,
                "source": "real_nhl_spotrac_cap",
            }
        )
    return out


def fetch_team_cap_sheet(abbr: str, season_year: int) -> Dict[str, Any]:
    """
    Current-season Spotrac cap sheet: active AAVs + buyouts + retained.

    The yearly multi-year board often leads with NEXT season's extension AAV
    (e.g. Pinto $7.5M / Spence $5.0M) while the club is still on the prior deal.
    Cap-page active rows are the source of truth for *this* season's hit.
    """
    slug = _spotrac_team_slug(abbr)
    empty = {"active": {}, "buyouts": [], "retained": []}
    if not slug:
        return empty
    year = int(season_year)
    urls = [
        f"https://www.spotrac.com/nhl/{slug}/cap/_/year/{year}",
        f"https://www.spotrac.com/nhl/{slug}/cap",
    ]
    if str(abbr).upper() == "UTA":
        urls.extend(
            [
                f"https://www.spotrac.com/nhl/utah-hockey-club/cap/_/year/{year}",
                "https://www.spotrac.com/nhl/utah-hockey-club/cap",
            ]
        )
    html = ""
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            html = _http_get_text(url)
            if 'id="table_active"' in html or "Cap Space" in html:
                break
        except Exception as e:
            last_err = e
            html = ""
    if not html:
        if last_err:
            raise last_err
        return empty

    active_rows = _parse_cap_table_rows(html, "table_active")
    buyout_rows = _parse_cap_table_rows(html, "table_dead")
    retained_rows = _parse_cap_table_rows(html, "table_retained")
    active: Dict[str, Dict[str, Any]] = {}
    for row in active_rows:
        entry = {
            "name": row["name"],
            "spotrac_id": row["spotrac_id"],
            "aav_m": row["aav_m"],
            "cap_hit_m": row["cap_hit_m"],
            "years_remaining": 1,
            "years": 1,
            "rights_status": "UFA",
            "contract_type": "ELC" if row["aav_m"] <= 1.0 else "STANDARD",
            "no_move_clause": False,
            "no_trade_clause": False,
            "source": "real_nhl_spotrac_cap",
        }
        key = row["name_key"]
        existing = active.get(key)
        if existing is None:
            active[key] = entry
        elif isinstance(existing, list):
            existing.append(entry)
        else:
            active[key] = [existing, entry]
    return {
        "active": active,
        "buyouts": [
            {
                "player": r["name"],
                "amount_m": r["cap_hit_m"],
                "cap_hit_m": r["cap_hit_m"],
                "season": f"{year}-{str(year + 1)[-2:]}",
                "source": "real_nhl_spotrac_cap",
            }
            for r in buyout_rows
            if float(r.get("cap_hit_m") or 0) > 0
        ],
        "retained": [
            {
                "player": r["name"],
                "amount_m": r["cap_hit_m"],
                "cap_hit_m": r["cap_hit_m"],
                "season": f"{year}-{str(year + 1)[-2:]}",
                "seasons_remaining": 1,
                "source": "real_nhl_spotrac_cap",
            }
            for r in retained_rows
            if float(r.get("cap_hit_m") or 0) > 0
        ],
    }


def _merge_cap_aav_over_yearly(
    yearly: Dict[str, Any],
    cap_active: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay current-season AAV from the cap sheet onto yearly rows."""
    if not cap_active:
        return yearly
    out: Dict[str, Any] = dict(yearly or {})
    for key, cap_entry in cap_active.items():
        caps = cap_entry if isinstance(cap_entry, list) else [cap_entry]
        existing = out.get(key)
        if existing is None:
            out[key] = [dict(c) for c in caps] if len(caps) > 1 else dict(caps[0])
            continue

        def _overlay(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(dst)
            cap_aav = float(src.get("aav_m") or src.get("cap_hit_m") or 0)
            old_aav = float(merged.get("aav_m") or merged.get("cap_hit_m") or 0)
            merged["aav_m"] = cap_aav
            merged["cap_hit_m"] = cap_aav
            merged["source"] = "real_nhl_spotrac"
            # Yearly boards often lead with an already-signed extension AAV. When the
            # current-season hit disagrees, do not trust the future-year remaining count.
            if old_aav > 0 and abs(old_aav - cap_aav) > 0.25:
                merged["years_remaining"] = 1
                merged["years"] = 1
                merged["extension_aav_m"] = old_aav
            return merged

        if isinstance(existing, list):
            if len(existing) == 1 and len(caps) == 1:
                out[key] = _overlay(existing[0], caps[0])
            else:
                # Match by spotrac_id when possible; else replace with cap rows.
                by_id = {int(c.get("spotrac_id") or 0): c for c in caps if c.get("spotrac_id")}
                merged_rows = []
                for row in existing:
                    sid = int(row.get("spotrac_id") or 0)
                    if sid and sid in by_id:
                        merged_rows.append(_overlay(row, by_id.pop(sid)))
                    else:
                        merged_rows.append(dict(row))
                for leftover in by_id.values():
                    merged_rows.append(dict(leftover))
                out[key] = merged_rows if len(merged_rows) > 1 else merged_rows[0]
        else:
            out[key] = _overlay(existing, caps[0])
    return out


def fetch_team_contracts(
    abbr: str,
    season_year: int,
    *,
    cap_sheet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Team contracts for Real NHL import.

    Yearly table supplies term / clause context; cap sheet supplies *current*
    season AAV (and is used alone when yearly is empty).
    """
    slug = _spotrac_team_slug(abbr)
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
    yearly: Dict[str, Any] = {}
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            html = _http_get_text(url)
            parsed = _parse_yearly_team_html(html)
            if parsed:
                yearly = parsed
                break
        except Exception as e:
            last_err = e
            continue

    sheet = cap_sheet
    if sheet is None:
        try:
            sheet = fetch_team_cap_sheet(abbr, year)
        except Exception as e:
            last_err = e
            sheet = {}

    active = dict((sheet or {}).get("active") or {})
    if yearly and active:
        return _merge_cap_aav_over_yearly(yearly, active)
    if active:
        return active
    if yearly:
        return yearly
    if last_err:
        raise last_err
    return {}


def fetch_league_contracts_by_team(
    abbreviations: List[str],
    season_year: int,
    *,
    max_workers: int = 10,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str], Dict[str, Dict[str, Any]]]:
    """
    Returns (by_abbr → name_key → contract, failures, dead_by_abbr).

    dead_by_abbr holds {"buyouts": [...], "retained": [...]} from the cap sheet.
    """
    by_team: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dead_by_team: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    abbrs = [str(a).upper() for a in abbreviations if a]

    def _one(abbr: str) -> Tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        try:
            sheet = fetch_team_cap_sheet(abbr, season_year)
        except Exception:
            sheet = {"active": {}, "buyouts": [], "retained": []}
        contracts = fetch_team_contracts(abbr, season_year, cap_sheet=sheet)
        dead = {
            "buyouts": list(sheet.get("buyouts") or []),
            "retained": list(sheet.get("retained") or []),
        }
        return abbr, contracts, dead

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, a) for a in abbrs]
        for fut in as_completed(futs):
            try:
                abbr, parsed, dead = fut.result()
                by_team[abbr] = parsed
                dead_by_team[abbr] = dead
                if not parsed:
                    failures.append(f"{abbr}: empty Spotrac contract tables")
            except Exception as e:
                failures.append(f"spotrac: {e}")
    return by_team, failures, dead_by_team


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
