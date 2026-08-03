"""Draft prospect profile cards for DraftClass modal — backend-driven, no fake fields."""
from __future__ import annotations

import re

from typing import Any, Dict, List, Optional


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    """Return the first non-empty value from known real row fields."""
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _pos_bucket(pos: str) -> str:
    p = str(pos or "").strip().upper()
    if p == "G" or "GOAL" in p:
        return "G"
    if p in ("D", "LD", "RD", "LHD", "RHD") or "DEF" in p:
        return "D"
    if p in ("LW", "RW", "W", "WING", "WINGER"):
        return "W"
    if p in ("C", "CE", "CENTER", "CENTRE"):
        return "C"
    return "UNK"


def _competition_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    code = str(row.get("league_code") or "").upper()
    label = str(row.get("league_display") or row.get("league") or "").strip()
    if not label:
        label = "Junior"

    if code.startswith("EU_J_") or code in ("SHL", "LIIGA", "DEL", "EU_J_SHL", "EU_J_LIIGA", "EU_J_DEL"):
        comp_label = "Pro Jr" if "EU_J" in code else str(row.get("league") or "Pro")
    elif code in ("NCAA",) or "NCAA" in label.upper():
        comp_label = "NCAA"
    elif code in ("CHL_OHL", "CHL_WHL", "CHL_QMJHL", "OHL", "WHL", "QMJHL") or "CHL" in code:
        comp_label = "CHL"
    elif code == "USHL" or "USHL" in label.upper():
        comp_label = "USHL"
    elif code.startswith("EU_"):
        comp_label = "Pro"
    else:
        comp_label = label.split()[0][:12] if label else "Junior"

    level_score = _f(row.get("league_difficulty"))
    if level_score <= 0:
        level_map = {
            "NCAA": 72.0,
            "CHL": 58.0,
            "USHL": 52.0,
            "Pro Jr": 76.0,
            "Pro": 82.0,
            "SHL": 78.0,
            "LIIGA": 77.0,
            "DEL": 76.0,
        }
        level_score = level_map.get(comp_label, 50.0)

    adjustment = _first_present(
        row,
        [
            "production_adjusted_score",
            "league_adjusted_ppg",
            "adjusted_ppg",
            "translation_factor",
        ],
    )

    out: Dict[str, Any] = {
        "label": comp_label,
        "level_score": round(level_score, 1),
    }
    if adjustment is not None:
        out["adjustment"] = round(_f(adjustment), 3)
    return out


_TALENT_GRADE_RANK = {"A+": 6, "A": 5, "A-": 4, "B+": 3, "B": 2, "B-": 1, "C+": 0, "C": 0}


def _talent_rank(row: Dict[str, Any]) -> int:
    grade = str(row.get("talent_grade") or row.get("scout_tier") or "").upper()
    return _TALENT_GRADE_RANK.get(grade, 0)


_LEAGUE_PARENT_MAP = {
    "CHL_OHL": ("CHL", "OHL"),
    "CHL_WHL": ("CHL", "WHL"),
    "CHL_QMJHL": ("CHL", "QMJHL"),
    "OHL": ("CHL", "OHL"),
    "WHL": ("CHL", "WHL"),
    "QMJHL": ("CHL", "QMJHL"),
    "NCAA": ("NCAA", None),
    "USHL": ("USHL", None),
    "EU_J_SHL": ("Pro Jr", "SHL"),
    "EU_J_LIIGA": ("Pro Jr", "Liiga"),
    "EU_J_DEL": ("Pro Jr", "DEL"),
    "SHL": ("Pro", "SHL"),
    "LIIGA": ("Pro", "Liiga"),
    "DEL": ("Pro", "DEL"),
}


def _strip_trailing_junk(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if not s:
        return ""
    s = re.sub(r"\s+\d+$", "", s)
    for marker in (" CHL ", " EU_J ", " NCAA ", " USHL ", " AHL "):
        idx = s.find(marker)
        if idx > 0:
            s = s[:idx].strip()
    parts = s.split()
    if len(parts) >= 3 and parts[-1].isdigit():
        tail = " ".join(parts[-2:])
        if re.search(r"CHL|EU_J|NCAA|USHL|AHL|WHL|OHL|QMJ", tail, re.I):
            city = " ".join(parts[:-2]).strip()
            if city:
                return city
    return s


def _clean_team_name(row: Dict[str, Any]) -> str:
    raw = str(row.get("team_name") or row.get("team") or "").strip()
    if not raw:
        return ""
    cleaned = _strip_trailing_junk(raw)
    league_bits = set()
    code = str(row.get("league_code") or "").upper()
    parent, sub = _LEAGUE_PARENT_MAP.get(code, (None, None))
    for bit in (parent, sub, str(row.get("league_display") or ""), str(row.get("league") or "")):
        if bit:
            league_bits.add(bit.upper())
    parts = cleaned.split()
    filtered: List[str] = []
    for p in parts:
        if p.upper() in league_bits:
            continue
        if p.isdigit():
            continue
        filtered.append(p)
    return " ".join(filtered).strip() or cleaned


def _clean_league_parts(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    code = str(row.get("league_code") or "").upper()
    parent, sub = _LEAGUE_PARENT_MAP.get(code, (None, None))
    if not parent:
        label = str(row.get("league_display") or row.get("league") or "").strip()
        label = _strip_trailing_junk(label)
        if "QMJ" in code or "QMJ" in label.upper():
            parent, sub = "CHL", "QMJHL"
        elif "OHL" in code or "OHL" in label.upper():
            parent, sub = "CHL", "OHL"
        elif "WHL" in code or "WHL" in label.upper():
            parent, sub = "CHL", "WHL"
        elif "NCAA" in code or "NCAA" in label.upper():
            parent, sub = "NCAA", None
        elif "USHL" in code or "USHL" in label.upper():
            parent, sub = "USHL", None
        elif code.startswith("EU_J"):
            parent, sub = "Pro Jr", label.split()[0][:12] if label else None
        elif code.startswith("EU_"):
            parent, sub = "Pro", label.split()[0][:12] if label else None
        else:
            parent = label.split("/")[0].strip()[:14] if label else "Junior"
            sub = None
    display = f"{parent} / {sub}" if parent and sub else (parent or sub or "Junior")
    return {"parent": parent, "sub": sub, "display": display}


def _identity_badges(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    raw_pos = str(row.get("position") or "").strip().upper()
    pos = raw_pos or None

    hand = str(row.get("handedness") or "").strip()
    if hand.lower().startswith("l"):
        hand = "L"
    elif hand.lower().startswith("r"):
        hand = "R"
    elif hand:
        hand = hand[:1].upper()
    else:
        hand = None

    height = str(row.get("height") or "").strip() or None
    weight = row.get("weight")
    weight_label = f"{int(round(_f(weight)))} LBS" if weight else None
    age = _i(row.get("age"), 0)

    return {
        "position": pos,
        "handedness": hand,
        "height": height,
        "weight": weight_label,
        "age": f"{age}Y" if age > 0 else None,
    }


def _profile_header(
    badges: Dict[str, Optional[str]],
    *,
    team_line: str,
    league_line: str,
) -> Dict[str, str]:
    primary_parts = [badges.get("position"), badges.get("handedness"), badges.get("height")]
    secondary_parts = [badges.get("weight"), badges.get("age")]
    primary = " • ".join([str(p) for p in primary_parts if p])
    secondary = " • ".join([str(p) for p in secondary_parts if p])
    return {
        "primary_line": primary,
        "secondary_line": secondary,
        "team_line": team_line or "",
        "league_line": league_line or "Junior",
    }


def _player_tags(row: Dict[str, Any]) -> List[str]:
    """Position-aware scouting tags from real signals — max 5.

    Production tags require a meaningful sample. Ceiling-adjacent quality tags are
    withheld when draft attention is too low to lock a ceiling. Size tags must
    match the actual frame so a weak Physical grade can't sit next to "Physical".
    """
    pos = _pos_bucket(str(row.get("position") or ""))
    ovr = _f(row.get("true_ovr"))
    pot = _f(row.get("potential_score"))
    gap = max(0.0, pot - ovr)
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    prod_adj = _f(row.get("production_adjusted_score"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    gp = _i(row.get("gp") or row.get("games_played"))
    hcm = _i(row.get("height_cm"), 0)
    weight = _f(row.get("weight"))
    hidden_ceiling = bool(row.get("ceiling_hidden"))
    sample_ok = gp >= 15
    tags: List[str] = []

    # Ceiling / readiness descriptors — only when the ceiling itself is visible.
    if not hidden_ceiling:
        if pot >= 88:
            tags.append("Elite Ceiling")
        elif pot >= 82:
            tags.append("Top-End Upside")
        elif pot >= 76:
            tags.append("Everyday NHLer")
        if ovr >= 74 and gap < 12 and _i(row.get("age"), 18) >= 18:
            tags.append("NHL-Ready")
        elif gap >= 16 and pot >= 80:
            tags.append("High Runway")
        if row.get("is_gem"):
            tags.append("Gem")
        if row.get("is_bust_risk") or row.get("boom_bust"):
            tags.append("Boom/Bust")
        elif str(row.get("risk") or "") == "Low" and gap < 12 and not row.get("character_concerns"):
            tags.append("Safe Floor")

    # Observable frame tags — independent of ceiling, but must match the body.
    big_frame = hcm >= 193 or weight >= 220
    true_physical = hcm >= 196 and weight >= 210
    if true_physical:
        tags.append("Physical")
    elif big_frame:
        tags.append("Big Frame")
    if hcm > 0 and hcm < 178 and pos != "G":
        tags.append("Undersized")

    # Production / role tags need a real sample — never invent "Offensive D" from 6 GP.
    if sample_ok and not hidden_ceiling:
        if pos == "G":
            if pot >= 86:
                tags.append("Athletic")
            if ovr >= 74:
                tags.append("Calm")
            if pot >= 82:
                tags.append("Positioning")
            if ovr >= 70 and pot >= 84:
                tags.append("Glove")
        elif pos == "D":
            if ppg >= 0.65 or prod_adj >= 0.95:
                tags.extend(["Offensive D", "Puck Mover"])
            elif ppg >= 0.45:
                tags.append("Puck Mover")
            if ovr >= 72 and ppg < 0.40:
                tags.append("Shutdown")
            if ovr >= 70 and ppg >= 0.35:
                tags.append("Two-Way D")
        elif pos in ("C", "W"):
            total = max(1, goals + assists)
            g_ratio = goals / total
            a_ratio = assists / total
            if ppg >= 1.15 or prod_adj >= 1.25:
                tags.append("Goal Scorer")
            if ppg >= 0.85:
                tags.append("Production")
            if g_ratio >= 0.52 and goals >= 8:
                tags.extend(["Sniper", "Shot"])
            if a_ratio >= 0.52 and assists >= 10:
                tags.append("Playmaker")
            if weight >= 205 and hcm >= 185:
                tags.append("Power Forward")
            if ppg >= 0.50 and ovr >= 70:
                tags.append("Two-Way F")
            if pos == "C" and ppg < 0.55 and ovr >= 72:
                tags.append("Shutdown C")
            if ppg >= 0.70 and pos == "W":
                tags.append("Transition")
            if ovr >= 72 and pot >= 78:
                tags.append("High IQ")
        else:
            if pot >= 80:
                tags.append("High Upside")
            if ppg >= 0.85 or prod_adj >= 1.0:
                tags.append("Production")
    elif sample_ok and hidden_ceiling:
        # Late / low-attention: only soft style hints from sustained production, no quality sell.
        if pos == "D" and ppg >= 0.55:
            tags.append("Puck Mover")
        elif pos in ("C", "W") and ppg >= 0.90:
            tags.append("Production")
        if weight >= 205 and hcm >= 185 and pos in ("C", "W"):
            tags.append("Power Forward")

    seen = set()
    out: List[str] = []
    cap = 3 if hidden_ceiling else 5
    for t in tags:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= cap:
            break
    return out


def _fmt_height(row: Dict[str, Any]) -> str:
    h = row.get("height") or row.get("height_display")
    if h:
        return str(h)
    cm = _i(row.get("height_cm"), 0)
    if cm <= 0:
        return ""
    inches = round(cm / 2.54)
    return f"{inches // 12}'{inches % 12}\""


def _evidence_strengths(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evidence-based strengths — title + supporting fact. Max 3."""
    out: List[Dict[str, Any]] = []
    pos = _pos_bucket(str(row.get("position") or ""))
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points")) or (goals + assists)
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    prod_adj = _f(row.get("production_adjusted_score"))
    league = str(row.get("league_display") or row.get("league") or "junior")
    pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate"))
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    conf = _f(row.get("scouting_confidence"), 55)
    wjc_raw = row.get("wjc_stats") if isinstance(row.get("wjc_stats"), dict) else {}
    wjc_gp = _i(row.get("wjc_gp") or row.get("wjc_games") or wjc_raw.get("gp"))
    wjc_pts = _i(row.get("wjc_points") or wjc_raw.get("points"))
    sv = _f(row.get("save_pct") or row.get("sv_pct") or row.get("save_percentage"))
    gaa = _f(row.get("gaa") or row.get("goals_against_avg"))
    hidden_ceiling = bool(row.get("ceiling_hidden"))

    if pos == "G":
        if sv >= 0.915 and gp >= 20:
            out.append({
                "title": "Save rate",
                "fact": f".{str(int(round(sv * 1000))).zfill(3)} SV% over {gp} GP",
                "context": f"Workload-backed stop rate in {league}",
                "confidence": "High" if conf >= 65 else "Medium",
            })
        if gaa > 0 and gaa <= 2.65 and gp >= 15:
            out.append({
                "title": "Goals against",
                "fact": f"{gaa:.2f} GAA in {gp} games",
                "context": "Strong goals-against relative to junior workload",
                "confidence": "Medium",
            })
        if pot >= 84 and not hidden_ceiling:
            out.append({
                "title": "Ceiling tools",
                "fact": f"Scouted ceiling band supports starter projection",
                "context": "Athletic profile grades among stronger goalie prospects",
                "confidence": "Medium" if conf < 70 else "High",
            })
        if ovr >= 70:
            out.append({
                "title": "Current readiness",
                "fact": f"Scouted current ability already projects NHL backup floor",
                "context": "Higher present ability than most draft goalies",
                "confidence": "High" if conf >= 60 else "Medium",
            })
    else:
        if gp >= 20 and (ppg >= 0.95 or (prod_adj >= 1.1 and ppg >= 0.75)):
            out.append({
                "title": "Production",
                "fact": f"{points} points in {gp} GP ({ppg:.2f} PPG) in {league}",
                "context": "Above peer scoring rate for this draft class",
                "confidence": "High" if conf >= 60 else "Medium",
            })
        elif gp >= 20 and assists >= max(goals, 1) * 1.4 and assists >= 20:
            out.append({
                "title": "Playmaking",
                "fact": f"{assists} assists in {gp} games",
                "context": f"Primary distributor among draft {pos or 'skaters'}",
                "confidence": "High" if conf >= 55 else "Medium",
            })
        elif gp >= 20 and goals >= 20 and goals >= assists:
            out.append({
                "title": "Finishing",
                "fact": f"{goals} goals in {gp} games",
                "context": "Goal-driven production relative to class peers",
                "confidence": "Medium",
            })
        elif hidden_ceiling and gp >= 15 and points > 0:
            # Late-round fog: still surface the counting line so drafting isn't a coin flip.
            out.append({
                "title": "Counting line",
                "fact": f"{goals}G-{assists}A-{points}P in {gp} GP ({ppg:.2f} PPG) · {league}",
                "context": "Public production available while ceiling stays ungraded",
                "confidence": "Medium" if gp >= 25 else "Low",
            })
        if wjc_gp > 0 and wjc_pts >= 4:
            out.append({
                "title": "WJC production",
                "fact": f"{wjc_pts} points in {wjc_gp} WJC games",
                "context": "Produced against international draft peers",
                "confidence": "High",
            })
        gap = max(0.0, pot - ovr)
        if pot >= 82 and gap >= 12 and not hidden_ceiling:
            out.append({
                "title": "Ceiling runway",
                "fact": f"Scouted ceiling well above current ability band",
                "context": "Large development gap with upside tools intact",
                "confidence": "Medium" if conf < 65 else "High",
            })
        elif ovr >= 72 and gap < 10 and conf >= 60:
            out.append({
                "title": "Present ability",
                "fact": f"Current ability grades among the more NHL-ready in class",
                "context": "Narrower outcome range than raw projects",
                "confidence": "High",
            })
        hcm = _i(row.get("height_cm"), 0)
        weight = _f(row.get("weight"))
        if (hcm >= 193 or weight >= 215) and pos == "D":
            out.append({
                "title": "Size",
                "fact": f"{_fmt_height(row) or f'{hcm} cm'}, {int(weight) if weight else '—'} lb",
                "context": "Above peer size for draft defensemen",
                "confidence": "High",
            })
        stock = _i(row.get("stock_delta") or row.get("rank_movement"), 0)
        catalyst = str(row.get("stock_reason") or row.get("movement_catalyst") or "")
        if stock >= 8 and catalyst:
            out.append({
                "title": "Stock rise",
                "fact": f"Rose {stock} spots — {catalyst}",
                "context": "Season events moved public board consensus",
                "confidence": "Medium",
            })

    # Deduplicate by title, cap at 3
    seen = set()
    clean: List[Dict[str, Any]] = []
    for item in out:
        t = item["title"]
        if t in seen:
            continue
        seen.add(t)
        clean.append(item)
        if len(clean) >= 3:
            break
    return clean


def _evidence_weaknesses(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evidence-based concerns — only when data supports them. Max 3."""
    out: List[Dict[str, Any]] = []
    pos = _pos_bucket(str(row.get("position") or ""))
    gp = _i(row.get("gp") or row.get("games_played"))
    points = _i(row.get("points"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    if points <= 0:
        points = goals + assists
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate"))
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    conf = _f(row.get("scouting_confidence"), 55)
    age = _i(row.get("age"), 18)
    hcm = _i(row.get("height_cm"), 0)
    weight = _f(row.get("weight"))
    height_disp = _fmt_height(row)
    sv = _f(row.get("save_pct") or row.get("sv_pct") or row.get("save_percentage"))

    if pos == "G":
        if sv > 0 and sv < 0.900 and gp >= 15:
            out.append({
                "title": "Save rate",
                "fact": f".{str(int(round(sv * 1000))).zfill(3)} SV% across {gp} GP",
                "context": "Stop rate trails stronger goalie peers",
                "confidence": "Medium",
            })
        if 0 < hcm < 185:
            out.append({
                "title": "Frame",
                "fact": f"{height_disp or f'{hcm} cm'} — below typical NHL goalie size band",
                "context": "Size relative to pro goalie standards",
                "confidence": "High",
            })
        if ovr > 0 and pot - ovr >= 14 and conf < 60:
            out.append({
                "title": "Projection volatility",
                "fact": "Wide ceiling gap with limited scouting confidence",
                "context": "Outcome range remains unsettled",
                "confidence": "Low",
            })
    else:
        if gp >= 25 and ppg > 0 and ppg < 0.45 and pos in ("C", "W"):
            out.append({
                "title": "Limited offense",
                "fact": f"{ppg:.2f} PPG over {gp} games",
                "context": "Scoring rate below draft-forward peers",
                "confidence": "High" if conf >= 55 else "Medium",
            })
        if pos == "D" and gp >= 25 and ppg < 0.25:
            out.append({
                "title": "Puck production",
                "fact": f"{points} points in {gp} GP ({ppg:.2f} PPG)",
                "context": "Offense trails draft defensemen at similar usage",
                "confidence": "Medium",
            })
        if pos == "D" and 0 < hcm < 183:
            out.append({
                "title": "Size concern",
                "fact": f"{height_disp or f'{hcm} cm'}{f', {int(weight)} lb' if weight else ''}",
                "context": "Below typical size band for NHL defensemen",
                "confidence": "High",
            })
        elif pos in ("C", "W") and 0 < hcm < 175:
            out.append({
                "title": "Size concern",
                "fact": f"{height_disp or f'{hcm} cm'}{f', {int(weight)} lb' if weight else ''}",
                "context": "Undersized relative to NHL forward standards",
                "confidence": "High",
            })
        if age >= 20:
            out.append({
                "title": "Older prospect",
                "fact": f"Age {age} — older than most of the draft class",
                "context": "Less remaining physical development runway",
                "confidence": "High",
            })
        if conf > 0 and conf < 45:
            out.append({
                "title": "Scouting uncertainty",
                "fact": f"Scouting confidence at {conf:.0f}%",
                "context": "Projection reliability remains limited",
                "confidence": "Low",
            })
        stock = _i(row.get("stock_delta") or row.get("rank_movement"), 0)
        catalyst = str(row.get("stock_reason") or row.get("movement_catalyst") or "")
        if stock <= -8 and catalyst:
            out.append({
                "title": "Stock drop",
                "fact": f"Fell {abs(stock)} spots — {catalyst}",
                "context": "Public board moved against him during the season",
                "confidence": "Medium",
            })
        if row.get("is_bust_risk") or (str(row.get("risk") or "") == "High" and pot - ovr >= 14):
            out.append({
                "title": "Development volatility",
                "fact": "Wide outcome range between floor and ceiling",
                "context": "Boom/bust profile supported by risk flags",
                "confidence": "Medium",
            })

    seen = set()
    clean: List[Dict[str, Any]] = []
    for item in out:
        t = item["title"]
        if t in seen:
            continue
        seen.add(t)
        clean.append(item)
        if len(clean) >= 3:
            break
    return clean


def _projection_notes(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Up to two projection notes distinguishing floor vs ceiling likelihood."""
    notes: List[Dict[str, Any]] = []
    hidden_ceiling = bool(row.get("ceiling_hidden"))
    pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate"))
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    gap = max(0.0, pot - ovr)
    nhl_prob = _f((row.get("potential") or {}).get("probability") if isinstance(row.get("potential"), dict) else 0)
    if nhl_prob <= 0:
        # Approximate from potential block inputs without calling full builder
        conf = _f(row.get("scouting_confidence"), 55)
        nhl_prob = max(15.0, min(90.0, 48.0 + (conf - 50.0) * 0.28 + (pot - 70.0) * 0.25 - gap * 0.3))
    age = _i(row.get("age"), 18)
    risk = str(row.get("risk") or "")

    if hidden_ceiling:
        # Ceiling is withheld — only floor / age-based reads are honest here.
        notes.append({
            "title": "Ceiling unestablished",
            "fact": "Limited draft-year spotlight on this pick",
            "context": "Project the upside yourself from production, analytics, age, size and tools",
        })
        if age >= 20:
            notes.append({
                "title": "Overager",
                "fact": f"Age {age} — older than most of the class",
                "context": "Less physical runway remaining",
            })
        return notes[:2]

    if ovr >= 70 and gap < 10 and nhl_prob >= 55:
        notes.append({
            "title": "Safe floor",
            "fact": f"NHL probability ~{nhl_prob:.0f}% with a narrower ceiling gap",
            "context": "Higher current ability, more limited upside swing",
        })
    elif gap >= 14 and ovr < 68:
        notes.append({
            "title": "High-upside swing",
            "fact": f"Large ceiling gap; NHL probability ~{nhl_prob:.0f}%",
            "context": "Lower present readiness with unfinished tools",
        })
    if age >= 20 and ovr >= 68:
        notes.append({
            "title": "Overager polish",
            "fact": f"Age {age} with stronger current production/ability than peers",
            "context": "Less physical runway remaining",
        })
    elif age <= 17 and gap >= 12:
        notes.append({
            "title": "Long runway",
            "fact": f"One of the younger players; development timeline stretches out",
            "context": "Ceiling depends on multi-year growth",
        })
    if risk == "High" and not notes:
        notes.append({
            "title": "Volatile projection",
            "fact": "Risk profile widens possible outcomes",
            "context": "Ceiling likelihood remains modest despite tools",
        })
    return notes[:2]


def _strengths_list(row: Dict[str, Any]) -> List[Any]:
    """Legacy string list plus structured evidence for UI progressive disclosure."""
    evidence = _evidence_strengths(row)
    if evidence:
        return [f"{e['title']} — {e['fact']}" for e in evidence]
    # Minimal fallback only when no evidence fires
    return []


def _concerns_list(row: Dict[str, Any]) -> List[Any]:
    evidence = _evidence_weaknesses(row)
    if evidence:
        return [f"{e['title']} — {e['fact']}" for e in evidence]
    return []


# NHL role ladders (ascending quality). Labels are position-agnostic — no "D"/"F"/"G"
# suffix — and each index doubles as the colour tier the UI lights up on the dossier.
_FORWARD_TIERS = ["Depth", "Bottom 6", "Middle 6", "Top 6", "Top Line", "Franchise"]
_DEFENSE_TIERS = ["Depth", "Top 6", "Bottom 4", "Top 4", "Top 2", "Franchise"]
_GOALIE_TIERS = ["AHL Starter", "Bench", "Tandem", "Starter", "Franchise"]


def _defense_projection(row: Dict[str, Any], ovr: float, pot: float, rank: int) -> int:
    """Return the defenseman role tier index (0..5) — aligned with peak-OVR upside bands."""
    talent = _talent_rank(row)
    # Franchise role only for true franchise ceilings (matches frontend peak bands).
    if pot >= 92 or (rank <= 2 and pot >= 90) or (rank == 1 and talent >= 6 and pot >= 88):
        return 5  # Franchise
    if pot >= 88 or (rank <= 5 and pot >= 86):
        return 4  # Top 2
    if pot >= 84 or (rank <= 12 and pot >= 80) or ovr >= 80:
        return 3  # Top 4
    if pot >= 74 or ovr >= 70 or (rank <= 40 and pot >= 71):
        return 2  # Bottom 4
    if pot >= 69 or ovr >= 64:
        return 1  # Top 6 (fringe NHL regular)
    return 0  # Depth


def _forward_projection(row: Dict[str, Any], ovr: float, pot: float, rank: int) -> int:
    """Return the forward role tier index (0..5) — aligned with peak-OVR upside bands.

    Ladder: Depth, Bottom 6, Middle 6, Top 6, Top Line, Franchise
    Franchise only at true franchise ceilings (92+), not merely #1 + 84 pot.
    """
    talent = _talent_rank(row)
    if pot >= 92 or (rank <= 2 and pot >= 90) or (rank == 1 and talent >= 6 and pot >= 88):
        return 5  # Franchise
    if pot >= 88 or (rank <= 6 and pot >= 86):
        return 4  # Top Line (elite band)
    if pot >= 84 or (rank <= 10 and pot >= 82) or ovr >= 82:
        return 4  # Top Line (high-upside band — matches HIGH UPSIDE)
    if pot >= 80 or (rank <= 15 and pot >= 76) or ovr >= 76:
        return 3  # Top 6
    if pot >= 72 or ovr >= 70 or (rank <= 45 and pot >= 70):
        return 2  # Middle 6
    if pot >= 67 or ovr >= 64:
        return 1  # Bottom 6
    return 0  # Depth


def _goalie_projection(row: Dict[str, Any], ovr: float, pot: float, rank: int) -> int:
    """Return the goalie role tier index (0..4)."""
    if row.get("generational_goalie"):
        return 4  # Franchise
    if (rank <= 8 and pot >= 82) or pot >= 86 or ovr >= 74:
        return 3  # Starter
    if pot >= 78 or ovr >= 70 or (rank <= 20 and pot >= 76):
        return 2  # Tandem
    if pot >= 72 or ovr >= 66:
        return 1  # Bench
    return 0  # AHL Starter


def _unknown_projection(ovr: float, pot: float, rank: int) -> int:
    if (rank <= 5 and pot >= 80) or pot >= 86:
        return 4
    if pot >= 78 or ovr >= 72:
        return 3
    if pot >= 72 or ovr >= 66:
        return 2
    if pot >= 67 or ovr >= 62:
        return 1
    return 0


def _projection_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate") or row.get("floor_score"))
    if ovr <= 0 and pot <= 0:
        return None

    # When the ceiling is withheld, the projected NHL path can only reflect the visible
    # floor — the guaranteed role — not the hidden upside.
    hidden_ceiling = bool(row.get("ceiling_hidden"))
    if hidden_ceiling:
        floor_only = _f(row.get("floor_score") or row.get("current_ovr_estimate") or ovr)
        if floor_only > 0:
            pot = floor_only
            if ovr <= 0:
                ovr = floor_only

    pos = _pos_bucket(str(row.get("position") or ""))
    rank = _i(row.get("rank"), 99)
    rank_boost = max(0.0, (32.0 - min(rank, 32)) * 0.35)
    role_score = ovr * 0.50 + pot * 0.38 + rank_boost

    if pos == "G":
        ladder = _GOALIE_TIERS
        tier = _goalie_projection(row, ovr, pot, rank)
    elif pos == "D":
        ladder = _DEFENSE_TIERS
        tier = _defense_projection(row, ovr, pot, rank)
    elif pos in ("C", "W"):
        ladder = _FORWARD_TIERS
        tier = _forward_projection(row, ovr, pot, rank)
    else:
        ladder = _FORWARD_TIERS
        tier = _unknown_projection(ovr, pot, rank)

    tier = max(0, min(len(ladder) - 1, int(tier)))
    label = ladder[tier]
    tier_max = len(ladder) - 1

    conf = min(95.0, max(35.0, _f(row.get("scouting_confidence"), 55)))
    if rank <= 5:
        conf = min(95.0, conf + 6.0)
    elif rank <= 12:
        conf = min(92.0, conf + 3.0)

    return {
        "label": label,
        "tier": tier,
        "tier_max": tier_max,
        "role_score": round(role_score, 1),
        "confidence": round(conf, 1),
        "hidden_ceiling": hidden_ceiling,
        "based_on": "floor" if hidden_ceiling else "ceiling",
    }


def _potential_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scout-visible ceiling block. `rating` is estimated ceiling, not final OVR.

    `probability` is NHL outcome odds derived from rank/tools/risk — not a renamed potential score.
    """
    pot = _f(row.get("expected_ceiling_estimate") or row.get("potential_score"))
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr"))
    if pot <= 0:
        return None

    gap = max(0.0, pot - ovr)
    rank = _i(row.get("rank"), 99)

    # Ceiling readability gate: for low-attention (late) prospects the ceiling AND the
    # numeric floor are withheld — a raw "62 floor" is itself a quality tell. The user
    # projects both ends from production, age, size and tools.
    if row.get("ceiling_hidden"):
        return {
            "rating": None,
            "floor": None,
            "floor_label": "Ungraded",
            "band": None,
            "probability": None,
            "nhl_probability": None,
            "risk": None,
            "label": None,
            "hidden": True,
            "hint": None,
        }
    talent = _talent_rank(row)
    conf = _f(row.get("scouting_confidence"), 55)
    # NHL probability from readiness signals — independent of raw ceiling number.
    prob = 48.0 + (conf - 50.0) * 0.28
    prob += min(12.0, max(-8.0, (pot - 70.0) * 0.35))
    prob -= gap * 0.35

    if rank <= 3:
        prob += 10.0
    elif rank <= 8:
        prob += 6.0
    elif rank <= 15:
        prob += 3.0

    prob += talent * 1.5

    if row.get("is_bust_risk") or row.get("boom_bust"):
        prob -= 16.0
    if row.get("is_gem"):
        prob += 6.0

    age = _i(row.get("age"), 18)
    if age >= 20:
        prob -= 7.0
    elif age <= 17:
        prob += 4.0

    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    prod_adj = _f(row.get("production_adjusted_score"))
    if ppg >= 1.1 or prod_adj >= 1.2:
        prob += 5.0
    elif ppg > 0 and ppg < 0.4:
        try:
            from services.draft_ranking_logic import infer_prospect_role

            role = infer_prospect_role(row)
            if role in ("defensive_defenseman", "shutdown_center", "grinder", "two_way_defenseman"):
                prob -= 2.0
            else:
                prob -= 6.0
        except Exception:
            prob -= 6.0

    if str(row.get("translation_risk") or "").lower() in ("high", "very high"):
        prob -= 8.0

    prob = max(18.0, min(92.0, prob))
    risk = str(row.get("risk") or "").strip() or ("High" if row.get("is_bust_risk") else "Medium")

    # Anti-correlated floor/ceiling band (see compute_prospect_outcome_band): high-ceiling
    # swings sit far below their ceiling; safe prospects sit just under theirs.
    outcome_band = str(row.get("outcome_band") or "")
    floor_val = _f(row.get("floor_score"))
    if floor_val <= 0:
        floor_val = round(max(38.0, ovr - gap * 0.35), 1)

    if outcome_band:
        ceiling_label = {
            "Boom/Bust": "Boom/bust",
            "Safe Floor": "Safe floor",
            "Balanced": "Balanced",
        }.get(outcome_band, "Standard")
        if rank <= 5 and pot >= 82 and outcome_band != "Boom/Bust":
            ceiling_label = "Elite ceiling"
    elif rank <= 5 and pot >= 78 and not row.get("is_bust_risk"):
        ceiling_label = "Elite ceiling"
    elif gap >= 18:
        ceiling_label = "High ceiling"
    elif prob >= 72 and gap < 10:
        ceiling_label = "Safe floor"
    else:
        ceiling_label = "Standard"

    return {
        "rating": round(pot, 1),  # estimated projected ceiling (scout-visible)
        "floor": round(floor_val, 1),  # reliable/downside outcome (anti-correlated w/ ceiling)
        "band": outcome_band or None,
        "probability": round(prob, 1),  # nhl_probability — not a potential alias
        "nhl_probability": round(prob, 1),
        "risk": risk,
        "label": ceiling_label,
    }


def _gem_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Development-risk read on a four-step ladder with a real-data reason.

    tier drives the UI colour: 0 = Major Risk (red), 1 = Neutral, 2 = Safe (green),
    3 = GEM (gold). ``reason`` is always sourced from real prospect signals so the
    dossier can show *why* next to the label.
    """
    pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate"))
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate"))
    gap = max(0.0, pot - ovr)
    conf = _f(row.get("scouting_confidence"), 55)
    age = _i(row.get("age"), 18)
    hcm = _i(row.get("height_cm"), 0)
    band = str(row.get("outcome_band") or "")
    concerns = bool(row.get("character_concerns"))
    boom = bool(row.get("is_bust_risk") or row.get("boom_bust"))
    trans_risk = str(row.get("translation_risk") or "").lower()

    # Ceiling withheld: the boom/bust read is a ceiling-vs-floor variance statement, so it
    # can't be shown. Only surface a risk when it comes from a non-ceiling signal (character).
    if row.get("ceiling_hidden"):
        if concerns:
            return {
                "label": "Major Risk",
                "tier": 0,
                "reason": "Character concerns",
                "score": round(ovr, 1),
                "reason_codes": ["character"],
            }
        return {
            "label": "Unknown",
            "tier": 1,
            "reason": "Ceiling unestablished — evaluate the profile yourself",
            "score": round(ovr, 1),
            "reason_codes": ["ceiling_hidden"],
        }

    if row.get("is_gem"):
        return {
            "label": "GEM",
            "tier": 3,
            "reason": "Pipeline value above draft slot",
            "score": round(pot, 1),
            "reason_codes": ["pipeline_steal"],
        }

    if boom or concerns or band == "Boom/Bust" or (gap >= 16 and conf < 60):
        if concerns:
            reason = "Character concerns"
        elif trans_risk in ("high", "very high"):
            reason = "League translation risk"
        elif 0 < hcm < 178:
            reason = "Undersized frame"
        elif age >= 20:
            reason = "Limited growth runway"
        elif boom or band == "Boom/Bust":
            reason = "Wide boom-or-bust range"
        else:
            reason = "Unproven projection"
        return {
            "label": "Major Risk",
            "tier": 0,
            "reason": reason,
            "score": round(ovr, 1),
            "reason_codes": ["bust_risk"],
        }

    if (band == "Safe Floor" or gap < 10) and conf >= 60:
        return {
            "label": "Safe",
            "tier": 2,
            "reason": "Reliable — narrow outcome range",
            "score": round(ovr, 1),
            "reason_codes": ["low_risk"],
        }

    reason = "Ceiling depends on development" if gap >= 12 else "Standard development curve"
    return {
        "label": "Neutral",
        "tier": 1,
        "reason": reason,
        "score": round(pot, 1),
        "reason_codes": ["balanced"],
    }


def _roster_counts(roster_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"C": 0, "W": 0, "D": 0, "G": 0, "RHD": 0, "LHD": 0, "UNK": 0}
    for p in roster_rows or []:
        raw_pos = str(p.get("position") or p.get("pos") or "").upper()
        pos = _pos_bucket(raw_pos)
        hand = str(p.get("handedness") or p.get("shoots") or "").upper()

        if pos == "G":
            counts["G"] += 1
        elif pos == "D":
            counts["D"] += 1
            if hand.startswith("R") or raw_pos in ("RD", "RHD"):
                counts["RHD"] += 1
            else:
                counts["LHD"] += 1
        elif pos == "W":
            counts["W"] += 1
        elif pos == "C":
            counts["C"] += 1
        else:
            counts["UNK"] += 1
    return counts


def _team_fit_block(
    row: Dict[str, Any],
    roster_rows: List[Dict[str, Any]],
    team_status: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not roster_rows:
        return None

    counts = _roster_counts(roster_rows)
    pos = _pos_bucket(str(row.get("position") or ""))
    hand = str(row.get("handedness") or "").lower()

    need_match = 50.0
    if pos == "G" and counts["G"] < 2:
        need_match = 92.0
    elif pos == "D" and hand.startswith("r") and counts["RHD"] < 3:
        need_match = 88.0
    elif pos == "D" and counts["D"] < 6:
        need_match = 84.0
    elif pos == "C" and counts["C"] < 3:
        need_match = 80.0
    elif pos == "W" and counts["W"] < 4:
        need_match = 78.0
    elif pos == "UNK":
        need_match = 46.0
    else:
        need_match = 52.0

    ts_key = str((team_status or {}).get("key") or "")
    ovr = _f(row.get("true_ovr"))
    age = _i(row.get("age"), 18)
    timeline_match = 55.0

    if ts_key in ("rebuilding", "tanking", "middling"):
        timeline_match = 78.0 if age <= 18 else (62.0 if age <= 19 else 48.0)
    elif ts_key in ("playoff_contender", "cup_contender"):
        timeline_match = 82.0 if ovr >= 72 else (58.0 if age <= 19 else 42.0)

    if pos == "D":
        depth = counts["D"]
    elif pos == "G":
        depth = counts["G"]
    elif pos == "C":
        depth = counts["C"]
    elif pos == "W":
        depth = counts["W"]
    else:
        depth = counts["C"] + counts["W"] + counts["D"]

    path_score = max(30.0, min(95.0, 88.0 - depth * 4.5))
    score = need_match * 0.45 + timeline_match * 0.30 + path_score * 0.25

    reasons: List[str] = []
    if pos == "G" and counts["G"] < 2:
        reasons.append(f"Thin goalie pipeline; only {counts['G']} NHL-depth goalie(s) on roster.")
    elif pos == "D" and hand.startswith("r") and counts["RHD"] < 3:
        reasons.append(f"Right-shot defense shortage; {counts['RHD']} RHD currently counted.")
    elif pos == "D" and counts["D"] < 6:
        reasons.append(f"Defense depth is thin ({counts['D']} D on roster).")
    elif pos == "C" and counts["C"] < 3:
        reasons.append(f"Center depth is thin ({counts['C']} centers).")
    elif pos == "W" and counts["W"] >= 6:
        reasons.append(f"Strong wing depth ({counts['W']}) makes NHL path crowded.")
    elif pos == "W" and counts["W"] < 4:
        reasons.append(f"Needs wing depth ({counts['W']} wingers).")
    if ts_key in ("rebuilding", "tanking", "middling") and age <= 19:
        reasons.append("Rebuild window can wait on development timeline.")
    elif ts_key in ("playoff_contender", "cup_contender") and ovr >= 72:
        reasons.append("Near-ready ability matches a competitive window.")
    elif ts_key in ("playoff_contender", "cup_contender") and age <= 18 and ovr < 68:
        reasons.append("Long timeline conflicts with an aging competitive core.")
    if timeline_match >= 75:
        reasons.append("Development timeline aligns with club window.")
    if path_score < 50:
        reasons.append("Crowded prospect path at this position.")

    # Late / low-attention prospects must not print a loud 77% "green light" fit score
    # that fights Do-Not-Draft and sells the pick. Cap the meter and keep only the
    # positional roster note — no rebuild/timeline boost packaging.
    rank = _i(row.get("rank"), 99)
    hidden_ceiling = bool(row.get("ceiling_hidden"))
    if hidden_ceiling or rank >= 160:
        score = min(score, 52.0)
        # Prefer the positional need line only — drop timeline marketing.
        reasons = [r for r in reasons if "shortage" in r.lower() or "thin" in r.lower() or "Needs" in r or "crowded" in r.lower()][:2]
        label = "Need noted" if need_match >= 78 else "Neutral"
        return {
            "score": None,  # no loud percentage on unestablished prospects
            "label": label,
            "reasons": reasons[:2],
            "fit_strengths": [],
            "fit_concerns": [],
            "need_match": round(need_match, 1),
            "timeline_match": round(timeline_match, 1),
            "path_score": round(path_score, 1),
            "position_depth": {
                "C": counts["C"],
                "W": counts["W"],
                "D": counts["D"],
                "G": counts["G"],
                "RHD": counts["RHD"],
                "LHD": counts["LHD"],
            },
            "note_only": True,
        }

    label = "Neutral"
    if score >= 82:
        label = "Elite Fit"
    elif score >= 68:
        label = "Good Fit"
    elif score < 45:
        label = "Poor Fit"

    fit_strengths = [r for r in reasons if "thin" in r.lower() or "shortage" in r.lower() or "Needs" in r or "Near-ready" in r or "aligns" in r.lower() or "wait" in r.lower()][:2]
    fit_concerns = [r for r in reasons if "crowded" in r.lower() or "conflicts" in r.lower() or "Crowded" in r][:2]

    return {
        "score": round(max(20.0, min(98.0, score)), 1),
        "label": label,
        "reasons": reasons[:4],
        "fit_strengths": fit_strengths,
        "fit_concerns": fit_concerns,
        "need_match": round(need_match, 1),
        "timeline_match": round(timeline_match, 1),
        "path_score": round(path_score, 1),
        "position_depth": {
            "C": counts["C"],
            "W": counts["W"],
            "D": counts["D"],
            "G": counts["G"],
            "RHD": counts["RHD"],
            "LHD": counts["LHD"],
        },
    }


def _eta_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from services.draft_ranking_logic import calculate_prospect_eta

    rank = _i(row.get("rank"), 99)
    eta = calculate_prospect_eta(row, final_rank=rank if rank < 90 else None)
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    if ovr <= 0 and not eta:
        return None
    return eta


def _willingness_block(
    row: Dict[str, Any],
    roster_rows: List[Dict[str, Any]],
    team_status: Optional[Dict[str, Any]],
    team_fit: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not roster_rows or not team_status:
        return None

    score = 50.0
    reasons: List[str] = []
    ts_key = str(team_status.get("key") or "")

    if ts_key in ("cup_contender", "playoff_contender"):
        score += 12.0
        reasons.append("competitive_window")
    elif ts_key in ("tanking", "rebuilding"):
        score -= 4.0
        reasons.append("rebuild_phase")

    if team_fit and team_fit.get("need_match", 0) >= 80:
        score += 14.0
        reasons.append("positional_need")
    elif team_fit and team_fit.get("path_score", 0) >= 75:
        score += 8.0
        reasons.append("roster_path")

    nat = str(row.get("nationality") or row.get("country") or "").lower()
    code = str(row.get("league_code") or "").upper()
    if nat in ("canada", "united states", "usa") and (code.startswith("CHL") or code in ("NCAA", "USHL")):
        score += 6.0
        reasons.append("na_pipeline")

    score = max(22.0, min(94.0, score))
    if score < 30:
        return None

    label = "High" if score >= 72 else ("Med" if score >= 52 else "Low")
    return {"score": round(score, 1), "label": label, "reason_codes": reasons}


def _comparison_basis(row: Dict[str, Any], pos: str, ppg: float) -> List[str]:
    basis: List[str] = []
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    total = goals + assists
    rank = _i(row.get("rank"), 99)
    pot = _f(row.get("potential_score"))
    ovr = _f(row.get("true_ovr"))
    hand = str(row.get("handedness") or "").lower()

    if pos == "G":
        basis.append("goalie")
    elif pos == "D":
        basis.append("right-shot D" if hand.startswith("r") else "left-shot D")
    elif pos == "W":
        basis.append("wing")
    elif pos == "C":
        basis.append("center")
    else:
        basis.append("unknown position")

    if rank <= 5:
        basis.append("top rank")
    if pot >= 82:
        basis.append("high ceiling")
    if ovr >= 74:
        basis.append("NHL-ready traits")
    if ppg >= 0.95:
        basis.append("high production")

    if total > 0:
        g_ratio = goals / total
        a_ratio = assists / total
        if g_ratio >= 0.52 and goals >= 8:
            basis.append("goal-heavy profile")
        elif a_ratio >= 0.52 and assists >= 10:
            basis.append("playmaking lean")

    return basis[:4]


def _comparison_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pos = _pos_bucket(str(row.get("position") or ""))
    ovr = _f(row.get("true_ovr"))
    pot = _f(row.get("potential_score"))
    rank = _i(row.get("rank"), 99)
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    hand = str(row.get("handedness") or "")
    gp = _i(row.get("gp") or row.get("games_played"))
    hcm = _i(row.get("height_cm"), 0)
    weight = _f(row.get("weight"))
    hidden_ceiling = bool(row.get("ceiling_hidden"))
    sample_ok = gp >= 15

    if ovr <= 0 and not hidden_ceiling:
        return None

    analytics = row.get("analytics") if isinstance(row.get("analytics"), dict) else {}
    off_war = _f(analytics.get("offensive_war"))
    def_war = _f(analytics.get("defensive_war"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    total = goals + assists
    g_ratio = (goals / total) if total else 0.0
    a_ratio = (assists / total) if total else 0.0
    has_war = sample_ok and (abs(off_war) > 0.001 or abs(def_war) > 0.001)
    # Tiny samples must not invent "Offensive D" from a hot 6-game PPG.
    offensive_lean = (off_war > def_war + 0.12) if has_war else (sample_ok and ppg >= 0.70)
    defensive_lean = (def_war > off_war + 0.12) if has_war else (sample_ok and ppg < 0.40 and ovr >= 68)

    # Low-attention / small-sample: style from frame + handedness only — no role sell.
    if hidden_ceiling or not sample_ok:
        if pos == "G":
            archetype = "Project Goalie"
        elif pos == "D":
            side = "RHD" if hand.lower().startswith("r") else "LHD"
            if hcm >= 193 or weight >= 215:
                archetype = f"Big-frame {side}"
            else:
                archetype = f"Mobile {side}"
        elif pos == "W":
            archetype = "Power Winger" if (weight >= 205 and hcm >= 185) else "Winger"
        elif pos == "C":
            archetype = "Power Center" if (weight >= 205 and hcm >= 185) else "Center"
        else:
            archetype = "Skater"
        return {
            "label": "STYLE",
            "archetype": archetype,
            "confidence": 40.0,
            "basis": ["frame", "handedness"] if pos == "D" else ["position"],
            "provisional": True,
        }

    if pos == "G":
        if row.get("generational_goalie"):
            archetype = "Franchise Goalie"
        elif rank <= 5 and pot >= 84:
            archetype = "Franchise Goalie"
        else:
            archetype = "Starting Goalie" if ovr >= 74 else "Project Goalie"
    elif pos == "D":
        side = "RHD" if hand.lower().startswith("r") else "LHD"
        if rank <= 3 and pot >= 80:
            archetype = f"Franchise {side}"
        elif offensive_lean and not defensive_lean:
            archetype = f"Offensive {side}"
        elif defensive_lean and not offensive_lean:
            archetype = f"Defensive {side}"
        elif ovr >= 72 or (rank <= 14 and pot >= 74):
            archetype = f"Two-Way {side}"
        else:
            archetype = f"Mobile {side}" if ppg >= 0.45 else f"Stay-Home {side}"
    elif pos in ("C", "W"):
        wing = pos == "W"
        if rank <= 3 and pot >= 82:
            archetype = "Franchise Forward"
        elif offensive_lean or ppg >= 0.85:
            if g_ratio >= 0.55 and goals >= 8:
                archetype = "Sniper"
            elif a_ratio >= 0.55 and assists >= 10:
                archetype = "Playmaker"
            else:
                archetype = "Scoring Winger" if wing else "Scoring Center"
        elif defensive_lean:
            archetype = "Two-Way Winger" if wing else "Two-Way Center"
        elif ppg >= 0.55:
            archetype = "Middle-Six Winger" if wing else "Middle-Six Center"
        else:
            archetype = "Energy Winger" if wing else "Checking Center"
    else:
        archetype = "Unknown skater"

    conf = min(88.0, max(42.0, _f(row.get("scouting_confidence"), 50) * 0.9))
    if rank <= 5:
        conf = min(92.0, conf + 4.0)

    return {
        "label": "STYLE",
        "archetype": archetype,
        "confidence": round(conf, 1),
        "basis": _comparison_basis(row, pos, ppg),
    }


def _ui_labels() -> Dict[str, str]:
    return {
        "projection": "NHL Path",
        "potential": "Ceiling",
        "gem": "Scout Read",
        "team_fit": "Fit",
        "eta": "Arrival",
        "player_comparison": "Style Match",
        "competition": "League Test",
    }


def _ui_scores(
    *,
    competition: Optional[Dict[str, Any]],
    potential: Optional[Dict[str, Any]],
    team_fit: Optional[Dict[str, Any]],
    eta: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if potential:
        if potential.get("rating") is not None:
            scores["ceiling"] = round(_f(potential.get("rating")), 1)
        if potential.get("probability") is not None:
            scores["certainty"] = round(_f(potential.get("probability")), 1)
    if team_fit and team_fit.get("score") is not None:
        scores["fit"] = round(_f(team_fit.get("score")), 1)
    if eta and eta.get("confidence") is not None:
        scores["arrival"] = round(_f(eta.get("confidence")), 1)
    if competition and competition.get("level_score") is not None:
        scores["competition"] = round(_f(competition.get("level_score")), 1)
    return scores


def _ui_priority(row: Dict[str, Any], gem: Optional[Dict[str, Any]]) -> List[str]:
    pos = _pos_bucket(str(row.get("position") or ""))
    gem_label = str((gem or {}).get("label") or "")

    if gem_label == "GEM":
        return ["gem", "potential", "projection", "team_fit", "eta"]
    if gem_label == "Major Risk" or row.get("is_bust_risk") or row.get("boom_bust"):
        return ["gem", "concerns", "potential", "projection", "eta"]
    if pos == "G":
        return ["projection", "potential", "eta", "competition", "concerns"]
    return ["projection", "potential", "team_fit", "eta", "competition"]


def _translation_note(row: Dict[str, Any]) -> Optional[str]:
    """Age / league leap warning — especially for Euro juniors and overagers."""
    age = _i(row.get("age"), 18)
    code = str(row.get("league_code") or "").upper()
    parent = str(row.get("league_parent") or row.get("leagueLevel") or "").lower()
    league = str(row.get("league_display") or row.get("league") or "")
    bits: List[str] = []
    euro = code.startswith("EU_") or "pro jr" in parent or "czech" in league.lower() or "shl" in league.lower() or "liiga" in league.lower()
    if euro and age <= 18:
        bits.append("Euro junior leap — NHL translation still unproven")
    elif code in ("NCAA",) or "ncaa" in league.lower():
        bits.append("NCAA path — longer development runway before NHL minutes")
    if age <= 17:
        bits.append("Young for the class — multi-year growth still ahead")
    elif age >= 20:
        bits.append("Overager — less physical runway remaining")
    return " · ".join(bits[:2]) if bits else None


def _readiness_label_for_row(row: Dict[str, Any]) -> str:
    """Current-ability readiness label — not ETA, not hidden ceiling quality."""
    age = _i(row.get("age"), 18)
    gp = _i(row.get("gp") or row.get("games_played"))
    if row.get("ceiling_hidden"):
        if age <= 17:
            return "Multi-year project"
        if age >= 20:
            return "Near-term decision"
        if gp < 15:
            return "Early-season read"
        code = str(row.get("league_code") or "").upper()
        if code.startswith("EU_"):
            return "Translation TBD"
        return "Development TBD"
    ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    pos = str(row.get("position") or "").upper()
    is_goalie = pos == "G"
    # Readiness is about playing NHL games now — OVR + age + position gates.
    if age <= 16:
        return "Long-term project"
    if is_goalie:
        if ovr >= 74 and age >= 22:
            return "NHL ready"
        if ovr >= 70 and age >= 20:
            return "Close"
        if ovr >= 64:
            return "Developing"
        return "Long-term project"
    if ovr >= 76 and age >= 18:
        return "NHL ready"
    if ovr >= 72 and age >= 18:
        return "Close"
    if ovr >= 66:
        return "Developing"
    if ovr >= 58:
        return "Long-term project"
    return "At Risk"


def _micro_summary(
    *,
    row: Dict[str, Any],
    tags: List[str],
    projection: Optional[Dict[str, Any]],
    potential: Optional[Dict[str, Any]],
    eta: Optional[Dict[str, Any]],
    translation: Optional[str],
    comparison: Optional[Dict[str, Any]],
) -> str:
    pos = _pos_bucket(str(row.get("position") or ""))
    role = str((projection or {}).get("label") or "")
    ceiling = str((potential or {}).get("label") or "")
    eta_label = str((eta or {}).get("label") or "")
    archetype = str((comparison or {}).get("archetype") or "")
    gp = _i(row.get("gp") or row.get("games_played"))
    age = _i(row.get("age"), 18)
    height = _fmt_height(row)
    league = str(row.get("league_display") or row.get("league") or "junior")
    weight = _f(row.get("weight"))

    if row.get("ceiling_hidden"):
        parts: List[str] = []
        if height and weight:
            parts.append(f"{height}, {int(weight)} lb {age}Y")
        elif age:
            parts.append(f"{age}-year-old")
        if league:
            parts.append(f"in {league}")
        if gp > 0 and gp < 15:
            parts.append(f"only {gp} GP so far — sample too small to grade")
        elif gp >= 15:
            ppg = _f(row.get("ppg") or row.get("points_per_game"))
            if ppg > 0:
                parts.append(f"{ppg:.2f} PPG through {gp} GP")
        frame = "Big frame" if ("Physical" in tags or "Big Frame" in tags) else None
        if frame:
            parts.append(frame.lower())
        if parts:
            return "; ".join(parts[:3]) + "."
        return "Low draft attention — judge him from the tape and the numbers."

    if "Boom/Bust" in tags and ceiling:
        return f"Boom/bust profile, {ceiling.lower()}."
    if "Gem" in tags and role:
        return f"Hidden value with {role.lower()} path."
    if "Small frame" in tags or "Undersized" in tags:
        if "Playmaker" in tags:
            return "Undersized playmaker, skill-driven upside."
        return "Undersized skater with real upside."
    if translation == "Strong" and archetype:
        return f"{archetype}, strong league translation."
    if eta_label in ("Now", "1Y") and role:
        return f"{role}, quick NHL arrival."
    if "Safe Pick" in tags and role:
        return f"Safe profile with {role.lower()} path."
    if "Sniper" in tags:
        return "Goal-heavy scorer with shooting pop."
    if "Playmaker" in tags:
        return "Playmaker with production-driven upside."
    if pos == "D" and role:
        return f"Mobile defender with {role.lower()} path."
    if pos == "G" and role:
        return f"Goalie project with {role.lower()} upside."
    if role:
        return f"{role} profile with steady upside."
    return "More viewings needed to lock the grade."


def _analytics_from_row_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill prospect analytics from season totals.

    WAR / possession are intentional scouting signals:
    - Scale with *current* ability (OVR) so higher-overall kids print higher WAR
    - Reward production that exceeds what that OVR normally produces (gem finder)
    - Never use hidden true potential — late-round ceiling fog stays intact, but
      over-productive tools still light up WAR so users can hunt gems.
    """
    existing = row.get("analytics") if isinstance(row.get("analytics"), dict) else {}
    out: Dict[str, Any] = dict(existing or {})
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points"))
    if points <= 0:
        points = goals + assists
    pos = str(row.get("position") or "").upper()
    is_goalie = pos == "G" or "GOAL" in pos
    is_d = pos in ("D", "LD", "RD") or pos.startswith("D")
    ceiling_hidden = bool(row.get("ceiling_hidden"))

    if is_goalie:
        if out.get("gsax") is None and gp > 0:
            sv = _f(row.get("save_pct") or row.get("sv_pct"), 0.0)
            if sv > 1.5:
                sv = sv / 100.0
            if sv > 0:
                shots_against = gp * 28.0
                out["gsax"] = round((sv - 0.905) * shots_against, 2)
        if out.get("quality_starts") is None and gp > 0:
            wins = _i(row.get("wins"))
            out["quality_starts"] = max(wins // 2, min(gp, int(round(gp * 0.45))))
        return {k: v for k, v in out.items() if v is not None}

    shots = _i(out.get("shots") or row.get("shots") or row.get("sog") or row.get("shots_on_goal"))
    if shots <= 0 and gp > 0:
        expected_sh = 0.12 if pos in ("C", "LW", "RW", "F", "W") else 0.07
        shots = max(goals * 6, int(round(goals / max(0.04, expected_sh)))) if goals else max(gp * 2, points * 2)
        shots = max(gp, shots)
    if shots > 0:
        out.setdefault("shots", shots)
        if gp > 0:
            out.setdefault("shot_rate", round(shots / float(gp), 2))
        if goals >= 0 and shots > 0:
            out.setdefault("shooting_pct", round((goals / float(shots)) * 100.0, 1))

    if out.get("primary_points") is None and points > 0:
        prim = int(round(goals + assists * 0.55))
        out["primary_points"] = max(goals, min(points, prim))

    if out.get("plus_minus") is None and row.get("plus_minus") is not None:
        out["plus_minus"] = _i(row.get("plus_minus"))
    elif out.get("plus_minus") is None and row.get("plusMinus") is not None:
        out["plus_minus"] = _i(row.get("plusMinus"))

    if gp >= 3:
        ppg = _f(row.get("ppg"))
        if ppg <= 0 and gp > 0:
            ppg = points / float(gp)
        # Observable ability only. When ceiling is fogged, ignore potential_score so
        # WAR cannot leak the hidden ceiling — gems must show up via production vs OVR.
        ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall"))
        if ovr <= 0:
            ovr = 58.0
        ability = max(0.40, min(0.92, ovr / 99.0))
        if not ceiling_hidden:
            pot = _f(row.get("potential_score") or row.get("expected_ceiling_estimate"))
            # Mild visible-ceiling blend only — never the sole driver.
            if pot >= 70:
                ability = min(0.92, ability * 0.82 + (pot / 99.0) * 0.18)

        shot_rate = _f(out.get("shot_rate"))
        if shot_rate <= 0 and gp > 0 and shots > 0:
            shot_rate = shots / float(gp)
        sh_pct = _f(out.get("shooting_pct"))
        plus_minus = out.get("plus_minus")
        pm_rate = (float(plus_minus) / float(gp)) if plus_minus is not None and gp > 0 else 0.0

        # What PPG this current ability typically posts in junior (CHL-ish).
        if is_d:
            expected_ppg = 0.18 + ability * 0.72
        else:
            expected_ppg = 0.22 + ability * 1.05
        surplus = ppg - expected_ppg

        # Reliability scales with sample; still usable early (gem hunting before GP 15).
        sample = min(1.0, (gp / 22.0) ** 0.65)

        # Ability floor: a 72 OVR kid clears ~+1.0 WAR baseline at full sample.
        ability_war = (ability - 0.52) * 5.4 * sample
        prod_war = (ppg - (0.35 if is_d else 0.48)) * (2.1 if is_d else 2.6) * sample
        # Overproduction vs ability = late-round gem signal (does not need potential).
        gem_war = max(0.0, surplus) * (3.2 if is_d else 3.8) * sample
        # Underproduction soft penalty so empty-calorie low-OVR points don't dominate.
        under_pen = min(0.0, surplus) * 1.1 * sample
        pm_war = pm_rate * 0.45 * sample
        shot_war = max(-0.35, min(0.55, (shot_rate - (1.6 if is_d else 2.2)) * 0.14)) * sample
        sh_war = 0.0
        if sh_pct > 0:
            sh_war = max(-0.25, min(0.35, (sh_pct - (7.5 if is_d else 10.0)) * 0.03)) * sample

        off_share = 0.42 if is_d else 0.68
        raw_war = ability_war + prod_war + gem_war + under_pen + pm_war + shot_war + sh_war
        war = round(max(-1.8, min(4.8, raw_war)), 2)
        off_war = round(max(-1.5, min(3.6, war * off_share + gem_war * 0.35)), 2)
        def_war = round(max(-1.2, min(2.6, war - off_war)), 2)

        # Possession tracks ability + surplus so high-OVR / gem kids don't all sit at 50%.
        poss = 50.0 + (ability - 0.58) * 22.0 + surplus * 6.5 + pm_rate * 1.8
        if is_d:
            poss += (ability - 0.55) * 4.0
        xgf = round(max(41.0, min(62.0, poss)), 1)
        cf = round(max(40.0, min(63.0, xgf + (ability - 0.60) * 3.0 + (0.8 if is_d else -0.4))), 1)

        # Always publish the scouting-signal WAR (overwrite stale thin-sample cookies).
        out["war"] = war
        out["offensive_war"] = off_war
        out["defensive_war"] = def_war
        out["xgf_pct"] = xgf
        out["cf_pct"] = cf
        out["analytics_signal"] = "gem_finder" if (ceiling_hidden and surplus >= 0.18 and war >= 1.15) else "standard"

    if out.get("toi") is None and gp > 0:
        ppg = _f(row.get("ppg"))
        if ppg <= 0:
            ppg = points / float(gp)
        ovr = _f(row.get("true_ovr") or row.get("current_ovr_estimate"))
        ability = (ovr / 99.0) if ovr > 0 else 0.58
        if is_d:
            toi = 17.5 + ability * 6.5 + min(3.0, ppg * 2.8)
        else:
            toi = 13.5 + ability * 7.0 + min(4.5, ppg * 3.5)
        out["toi"] = round(min(24.0 if not is_d else 26.0, toi), 1)

    return {k: v for k, v in out.items() if v is not None}


def _observable_nhl_odds(row: Dict[str, Any]) -> Optional[float]:
    """NHL odds from public signals only — never true/scouted ceiling when fogged."""
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points")) or (goals + assists)
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / float(gp)
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr") or row.get("scouted_overall_estimate"))
    age = _i(row.get("age"), 18)
    rank = _i(row.get("rank"), 120)
    conf = _f(row.get("scouting_confidence"), 50)
    prod_adj = _f(row.get("production_adjusted_score"))
    pos = _pos_bucket(str(row.get("position") or ""))

    odds = 28.0 + (conf - 50.0) * 0.18
    if ovr > 0:
        odds += (ovr - 55.0) * 0.55
    if gp >= 20:
        if pos == "G":
            sv = _f(row.get("save_pct") or row.get("sv_pct"))
            if sv > 1.5:
                sv = sv / 100.0
            if sv >= 0.915:
                odds += 10.0
            elif sv >= 0.900:
                odds += 4.0
        else:
            if ppg >= 1.1 or prod_adj >= 1.25:
                odds += 12.0
            elif ppg >= 0.85 or prod_adj >= 1.05:
                odds += 7.0
            elif ppg >= 0.55:
                odds += 3.0
            elif ppg > 0 and ppg < 0.35:
                odds -= 6.0
    if age <= 17:
        odds += 3.0
    elif age >= 20:
        odds -= 5.0
    # Public board slot is an attention signal, not a ceiling leak.
    if rank <= 64:
        odds += 4.0
    elif rank <= 96:
        odds += 1.0
    elif rank >= 150:
        odds -= 4.0
    if row.get("is_bust_risk") or row.get("character_concerns"):
        odds -= 8.0
    return round(max(8.0, min(72.0, odds)), 1)


def _fog_projection_notes(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Decision notes for ceiling-fogged prospects — production / frame / age only."""
    notes: List[Dict[str, Any]] = []
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points")) or (goals + assists)
    ppg = _f(row.get("ppg") or row.get("points_per_game"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / float(gp)
    league = str(row.get("league_display") or row.get("league") or "junior")
    age = _i(row.get("age"), 18)
    height = _fmt_height(row)
    weight = _f(row.get("weight"))
    prod_adj = _f(row.get("production_adjusted_score"))
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr"))
    pos = _pos_bucket(str(row.get("position") or ""))

    notes.append({
        "title": "Ceiling fogged",
        "fact": "True upside is ungraded — project from production, age, size, and league context",
    })
    if gp >= 15 and pos != "G":
        notes.append({
            "title": "Season production",
            "fact": f"{gp} GP · {goals}G-{assists}A-{points}P · {ppg:.2f} PPG in {league}",
        })
        if prod_adj >= 1.15:
            notes.append({
                "title": "League translation",
                "fact": "Production grades above peer expectation for this league",
            })
        elif prod_adj > 0 and prod_adj < 0.85:
            notes.append({
                "title": "League translation",
                "fact": "Raw counting stats need a discount vs stronger leagues",
            })
    elif gp >= 15 and pos == "G":
        sv = _f(row.get("save_pct") or row.get("sv_pct"))
        gaa = _f(row.get("gaa"))
        bits = [f"{gp} GP"]
        if sv > 0:
            if sv > 1.5:
                sv = sv / 100.0
            bits.append(f".{str(int(round(sv * 1000))).zfill(3)} SV%")
        if gaa > 0:
            bits.append(f"{gaa:.2f} GAA")
        notes.append({"title": "Season workload", "fact": " · ".join(bits) + f" ({league})"})
    if height and weight:
        notes.append({
            "title": "Frame",
            "fact": f"{height}, {int(weight)} lb at age {age}",
        })
    if age <= 17:
        notes.append({"title": "Age", "fact": "Young for the class — longer development runway"})
    elif age >= 20:
        notes.append({"title": "Age", "fact": "Overager — less physical runway; need NHL traits sooner"})
    if ovr > 0:
        if ovr >= 64:
            notes.append({"title": "Present ability", "fact": "Current tools already project a usable pro floor"})
        elif ovr <= 52:
            notes.append({"title": "Present ability", "fact": "Raw project — multi-year growth required before NHL minutes"})
    return notes[:5]


def _rank_history_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer stored stock trail; otherwise build checkpoints from real board ranks."""
    raw = row.get("rank_history") or row.get("stock_history") or []
    if isinstance(raw, list) and len(raw) >= 2:
        return list(raw)
    points: List[Dict[str, Any]] = []
    for label, keys in (
        ("Preseason", ("preseason_rank", "preseasonRank")),
        ("Midseason", ("midseason_rank", "midseasonRank")),
        ("Current", ("rank", "central_rank", "public_rank")),
    ):
        rank = None
        for k in keys:
            if row.get(k) is not None:
                try:
                    rank = int(row.get(k))
                except Exception:
                    rank = None
                if rank:
                    break
        if not rank:
            continue
        if points and int(points[-1].get("rank") or 0) == rank:
            if label != "Current":
                continue
        points.append({
            "date_label": label,
            "date": label,
            "label": label,
            "rank": rank,
            "event_source": "board_checkpoint",
        })
    # If history is thin/flat, backfill a season arc from stock delta / preseason gap.
    ranks = [int(p.get("rank") or 0) for p in points]
    flat = (not points) or len(points) < 2 or (len(set(ranks)) <= 1)
    if flat:
        current = None
        for k in ("rank", "central_rank", "public_rank"):
            if row.get(k) is not None:
                try:
                    current = int(row.get(k))
                except Exception:
                    current = None
                if current:
                    break
        pre = _i(row.get("preseason_rank") or row.get("preseasonRank"))
        delta = row.get("stock_change")
        if delta is None:
            delta = row.get("stock_delta")
        if delta is None:
            nested = row.get("draft_stock") if isinstance(row.get("draft_stock"), dict) else {}
            delta = nested.get("delta_rank") or nested.get("deltaRank") or nested.get("stock_heat")
        try:
            delta_n = int(delta) if delta is not None else 0
        except Exception:
            delta_n = 0
        if current and pre and pre != current:
            delta_n = int(pre) - int(current)
        if current and delta_n:
            earlier = max(1, int(current) + int(delta_n))
            mid = max(1, int(round((earlier + current) / 2.0)))
            points = [
                {
                    "date_label": "Preseason",
                    "date": "Preseason",
                    "label": "Preseason",
                    "rank": earlier,
                    "event_source": "stock_delta",
                },
                {
                    "date_label": "Midseason",
                    "date": "Midseason",
                    "label": "Midseason",
                    "rank": mid,
                    "event_source": "stock_delta",
                },
                {
                    "date_label": "Current",
                    "date": "Current",
                    "label": "Current",
                    "rank": int(current),
                    "event_source": "board_checkpoint",
                },
            ]
        elif current:
            points = [
                {
                    "date_label": "Preseason",
                    "date": "Preseason",
                    "label": "Preseason",
                    "rank": int(pre or current),
                    "event_source": "board_checkpoint",
                },
                {
                    "date_label": "Current",
                    "date": "Current",
                    "label": "Current",
                    "rank": int(current),
                    "event_source": "board_checkpoint",
                },
            ]
        elif isinstance(raw, list) and raw:
            return list(raw)
    return points


def build_prospect_profile(
    row: Dict[str, Any],
    *,
    roster_rows: Optional[List[Dict[str, Any]]] = None,
    team_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one draft board entry into modal profile payload."""
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points"))
    if points <= 0 and (goals or assists):
        points = goals + assists

    ppg = row.get("ppg")
    if ppg is None and gp > 0:
        ppg = round(points / gp, 3)

    team_fit = _team_fit_block(row, roster_rows or [], team_status)
    league_parts = _clean_league_parts(row)
    clean_team = _clean_team_name(row)
    league_display = league_parts.get("display") or str(row.get("league_display") or row.get("league") or "") or "Junior"
    scout_conf = round(_f(row.get("scouting_confidence"), 55), 1)
    # Dedicated user scouting (overlay) — ambient games-played confidence is NOT a file.
    user_scout = _f(
        row.get("scouted_percentage")
        or row.get("user_scouted_percentage")
        or row.get("team_scout_pct"),
        0.0,
    )
    dedicated_file = user_scout >= 20.0
    ceiling_hidden = bool(row.get("ceiling_hidden"))
    if ceiling_hidden and not dedicated_file:
        # Ambient board confidence (often 75–90% from GP) must not read as a locked file.
        scout_conf = min(scout_conf, 42.0)
    comp = _competition_block(row)

    translation_label = None
    if comp and comp.get("adjustment") is not None:
        adj = _f(comp.get("adjustment"))
        if adj >= 1.1:
            translation_label = "Strong"
        elif adj >= 0.85:
            translation_label = "Avg"
        else:
            translation_label = "Risk"

    badges = _identity_badges(row)
    potential = _potential_block(row)
    # Floor-based role + age/league ETA stay available even when ceiling is fogged —
    # late-round drafting needs a decision surface without leaking true upside.
    projection = _projection_block(row)
    gem = None if ceiling_hidden else _gem_block(row)
    eta = _eta_block(row)
    ready = _readiness_label_for_row(row)
    if ceiling_hidden:
        eta_label = str((eta or {}).get("label") or "").strip().upper()
        if eta is None or eta_label in ("", "UNKNOWN", "N/A", "NONE"):
            if ready:
                eta = {"label": ready, "years": None, "confidence": "low", "based_on": "observable"}
    comparison = _comparison_block(row)
    tags = _player_tags(row)
    is_transcendent = bool(row.get("is_transcendent") or row.get("transcendent_talent"))
    intel_label = str(row.get("intel_label") or "")
    if ceiling_hidden and not dedicated_file:
        intel_label = "Limited"
    evidence_strengths = _evidence_strengths(row)
    evidence_weaknesses = _evidence_weaknesses(row)
    projection_notes = _fog_projection_notes(row) if ceiling_hidden else _projection_notes(row)
    translation_note = _translation_note(row)
    readiness_label = ready
    sample_thin = gp > 0 and gp < 15
    sample_note = f"Small sample ({gp} GP)" if sample_thin else (f"{gp} GP sample" if gp >= 15 else None)
    analytics = _analytics_from_row_stats(row)
    rank_history = _rank_history_from_row(row)

    # Development profile classification from real signals
    pot_v = _f((potential or {}).get("rating") if potential else 0) or _f(row.get("potential_score"))
    ovr_v = _f(row.get("true_ovr") or row.get("current_ovr_estimate") or row.get("scouted_overall_estimate"))
    gap_v = max(0.0, pot_v - ovr_v)
    nhl_prob = _f((potential or {}).get("probability") if potential else 0)
    if ceiling_hidden or nhl_prob <= 0:
        nhl_prob = _f(_observable_nhl_odds(row))
    age_v = _i(row.get("age"), 18)
    if ceiling_hidden:
        # Upside is unknown, so only age-based / floor-based reads are honest here.
        if age_v >= 20:
            development_profile = "Overager"
        elif ovr_v >= 64 and gp >= 30:
            development_profile = "Tools Present"
        elif gp >= 20 and _f(row.get("ppg") or (points / float(gp) if gp else 0)) >= 0.85:
            development_profile = "Producer — Ceiling TBD"
        else:
            development_profile = "Projection Unknown"
    elif age_v >= 20:
        development_profile = "Overager"
    elif ovr_v >= 70 and gap_v < 10 and nhl_prob >= 55:
        development_profile = "Safe Pick"
    elif gap_v >= 14 and ovr_v < 68:
        development_profile = "Upside Pick"
    elif ovr_v < 64 and gap_v >= 12:
        development_profile = "Raw Project"
    else:
        development_profile = "Balanced"

    ovr_low = row.get("overall_range_low") or row.get("scouted_overall_low")
    ovr_high = row.get("overall_range_high") or row.get("scouted_overall_high")
    if ovr_low is None and ovr_v > 0:
        spread = max(1, int(round((100 - scout_conf) / 25)))
        ovr_low = max(40, int(round(ovr_v)) - spread)
        ovr_high = min(99, int(round(ovr_v)) + spread)
    pot_range = row.get("potential_range")
    pot_low = pot_high = None
    if ceiling_hidden:
        pot_low = pot_high = None
    elif isinstance(pot_range, (list, tuple)) and len(pot_range) >= 2:
        pot_low, pot_high = pot_range[0], pot_range[1]
    elif pot_v > 0:
        spread = max(2, int(round((100 - scout_conf) / 20)))
        pot_low = max(50, int(round(pot_v)) - spread)
        pot_high = min(99, int(round(pot_v)) + spread)

    preseason = _i(row.get("preseason_rank")) or None
    current_rank = _i(row.get("rank")) or None
    midseason = _i(row.get("midseason_rank")) or None
    rank_movement = None
    if preseason and current_rank:
        rank_movement = preseason - current_rank  # positive = rose

    wjc_stats = None
    wjc_raw = row.get("wjc_stats") if isinstance(row.get("wjc_stats"), dict) else None
    wjc_gp = _i(row.get("wjc_gp") or row.get("wjc_games") or (wjc_raw or {}).get("gp"))
    if wjc_gp > 0 or wjc_raw:
        wjc_stats = {
            "played": True,
            "games": wjc_gp or _i((wjc_raw or {}).get("gp")),
            "goals": _i(row.get("wjc_goals") or (wjc_raw or {}).get("goals")),
            "assists": _i(row.get("wjc_assists") or (wjc_raw or {}).get("assists")),
            "points": _i(row.get("wjc_points") or (wjc_raw or {}).get("points")),
            "team": row.get("wjc_team") or (wjc_raw or {}).get("team"),
            "year": row.get("wjc_year") or (wjc_raw or {}).get("year"),
            "result": row.get("wjc_result") or (wjc_raw or {}).get("result"),
        }
        if wjc_stats["games"] and wjc_stats["points"] is not None:
            wjc_stats["ppg"] = round(wjc_stats["points"] / max(1, wjc_stats["games"]), 2)

    playoff_stats = None
    if _i(row.get("playoff_gp") or row.get("playoff_games")) > 0:
        pog = _i(row.get("playoff_gp") or row.get("playoff_games"))
        pop = _i(row.get("playoff_points"))
        playoff_stats = {
            "games": pog,
            "goals": _i(row.get("playoff_goals")),
            "assists": _i(row.get("playoff_assists")),
            "points": pop,
            "ppg": round(pop / max(1, pog), 2) if pop else None,
        }

    profile: Dict[str, Any] = {
        "id": str(row.get("key") or row.get("id") or ""),
        "playerId": str(row.get("key") or row.get("id") or ""),
        "rank": _i(row.get("rank")),
        "publicRank": current_rank,
        "userRank": _i(row.get("team_board_rank") or row.get("user_rank")) or None,
        "teamRank": _i(row.get("team_board_rank")) or None,
        "preseasonRank": preseason,
        "midseasonRank": midseason,
        "currentRank": current_rank,
        "rankMovement": rank_movement,
        "movementDirection": "up" if (rank_movement or 0) > 0 else ("down" if (rank_movement or 0) < 0 else "flat"),
        "movementCatalysts": [
            c for c in [
                row.get("stock_reason"),
                row.get("movement_catalyst"),
                row.get("stock_label"),
            ] if c
        ][:3],
        "rankHistory": rank_history,
        "stock_history": rank_history,
        "name": str(row.get("name") or ""),
        "fullName": str(row.get("name") or ""),
        "firstName": str(row.get("first_name") or "").strip() or None,
        "lastName": str(row.get("last_name") or "").strip() or None,
        "position": str(row.get("position") or ""),
        "positionGroup": _pos_bucket(str(row.get("position") or "")),
        "position_bucket": _pos_bucket(str(row.get("position") or "")),
        "handedness": str(row.get("handedness") or ""),
        "height": row.get("height") or _fmt_height(row) or None,
        "heightDisplay": row.get("height") or _fmt_height(row) or None,
        "weight": row.get("weight"),
        "age": _i(row.get("age")) or None,
        "team": clean_team or str(row.get("team_name") or row.get("team") or ""),
        "club": clean_team or str(row.get("team_name") or row.get("team") or ""),
        "currentClub": clean_team or str(row.get("team_name") or row.get("team") or ""),
        "league": league_display,
        "league_display": league_display,
        "leagueLevel": league_parts.get("parent"),
        "league_short": league_parts.get("parent"),
        "league_detail": league_parts.get("sub"),
        "league_parent": league_parts.get("parent"),
        "league_sub": league_parts.get("sub"),
        "countryCode": str(row.get("country_code") or "") or None,
        "country_code": str(row.get("country_code") or "") or None,
        "nationality": str(row.get("nationality") or row.get("country") or "") or None,
        "identity_badges": badges,
        "profile_header": _profile_header(
            badges,
            team_line=clean_team or str(row.get("team_name") or row.get("team") or ""),
            league_line=league_display,
        ),
        "scout_confidence": scout_conf,
        "scoutingConfidence": scout_conf,
        "overallConfidence": scout_conf,
        "dedicatedScoutFile": dedicated_file,
        "intel_label": intel_label or (
            "Limited" if (ceiling_hidden and not dedicated_file)
            else ("Locked" if scout_conf >= 91 else "Solid" if scout_conf >= 56 else "Limited")
        ),
        "potential_range": None if ceiling_hidden else row.get("potential_range"),
        # Current ability stays visible — fog only hides ceiling / true upside.
        "scoutedOverall": None if (ceiling_hidden and not dedicated_file and not (
            bool(row.get("ovr_revealed")) or scout_conf >= 88.0
        )) else (ovr_v or None),
        "overallRangeLow": ovr_low,
        "overallRangeHigh": ovr_high,
        "scoutedPotentialLow": pot_low,
        "scoutedPotentialHigh": pot_high,
        "nhlProbability": nhl_prob or None,
        "ceilingHidden": ceiling_hidden,
        "ceilingVisibility": row.get("ceiling_visibility"),
        "ceilingState": str(row.get("ceiling_state") or ("hidden" if ceiling_hidden else "clear")),
        "ceilingHint": (
            "Ungraded — project upside from production, age, size, and league context"
            if ceiling_hidden else None
        ),
        "readinessLabel": readiness_label,
        "translationNote": translation_note,
        "sampleNote": sample_note,
        "sampleThin": sample_thin,
        "stockReason": row.get("stock_reason") or row.get("movement_catalyst") or None,
        "ceilingLikelihood": (
            "Ungraded" if ceiling_hidden
            else "High" if nhl_prob >= 70 and gap_v < 12
            else "Moderate" if nhl_prob >= 50
            else "Low" if nhl_prob > 0
            else None
        ),
        "developmentProfile": development_profile,
        "developmentVolatility": (
            "High" if ceiling_hidden
            else "Low" if gap_v < 8 else "Medium" if gap_v < 14 else "High"
        ),
        "tags": tags,
        "strengths": (
            [e.get("title") for e in evidence_strengths if isinstance(e, dict) and e.get("title")][:3]
            if ceiling_hidden
            else _strengths_list(row)
        ),
        "strengthsEvidence": evidence_strengths[:3] if ceiling_hidden else evidence_strengths,
        "weaknessesEvidence": evidence_weaknesses[:3] if ceiling_hidden else evidence_weaknesses,
        "projectionNotes": projection_notes,
        "concerns": (
            [e.get("title") for e in evidence_weaknesses if isinstance(e, dict) and e.get("title")][:3]
            if ceiling_hidden
            else _concerns_list(row)
        ),
        "translation": translation_label,
        "stats": {
            "games": gp,
            "goals": goals,
            "assists": assists,
            "points": points,
            "ppg": round(_f(ppg), 3) if ppg is not None and gp > 0 else None,
            "analytics": analytics or None,
            "sampleNote": sample_note,
            "sampleThin": sample_thin,
            "league": league_display,
            "shots": analytics.get("shots") if analytics else row.get("shots"),
            "toi": analytics.get("toi") if analytics else None,
        },
        "analytics": analytics or None,
        "wjcStats": wjc_stats,
        "playoffStats": playoff_stats,
        "competition": comp,
        "projection": projection,
        "potential": potential,
        "gem": gem,
        "team_fit": team_fit,
        "teamFit": team_fit,
        "eta": eta,
        "estimatedNhlArrival": (eta or {}).get("label") if isinstance(eta, dict) else None,
        "player_comparison": comparison,
        "ui_labels": _ui_labels(),
        "ui_scores": _ui_scores(
            competition=comp,
            potential=potential,
            team_fit=team_fit,
            eta=eta,
        ),
        "ui_priority": _ui_priority(row, gem),
        "micro_summary": _micro_summary(
            row=row,
            tags=tags,
            projection=projection,
            potential=potential,
            eta=eta,
            translation=translation_label,
            comparison=comparison,
        ),
    }
    if is_transcendent:
        # Narrative metadata stays server-side / storyline only. Do not ship
        # is_transcendent / aura / special_fx on browser-facing dossiers.
        origin = row.get("origin_story")
        if isinstance(origin, dict) and origin:
            # Origin prose is public colour; omit if it encodes hidden-tier labels.
            safe_origin = {
                k: v
                for k, v in origin.items()
                if str(k).lower() not in ("true_potential", "hidden_tier", "pipeline_tier")
            }
            if safe_origin:
                profile["origin_story"] = safe_origin
    elif isinstance(row.get("origin_story"), dict):
        profile["origin_story"] = row.get("origin_story")
    return profile


def build_prospect_profiles_by_id(
    entries: List[Dict[str, Any]],
    *,
    roster_rows: Optional[List[Dict[str, Any]]] = None,
    team_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in entries or []:
        pid = str(row.get("key") or row.get("id") or "")
        if not pid:
            continue
        out[pid] = build_prospect_profile(row, roster_rows=roster_rows, team_status=team_status)
    return out