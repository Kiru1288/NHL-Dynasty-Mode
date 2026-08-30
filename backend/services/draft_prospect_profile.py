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


def _parse_score_range(raw: Any) -> tuple[Optional[float], Optional[float]]:
    """Normalize board range payloads (dict, list, tuple) to (low, high)."""
    if raw is None:
        return None, None
    if isinstance(raw, dict):
        lo = raw.get("low")
        hi = raw.get("high")
        try:
            low_v = float(lo) if lo is not None else None
            high_v = float(hi) if hi is not None else None
        except (TypeError, ValueError):
            return None, None
        if low_v is None or high_v is None:
            return None, None
        if high_v < low_v:
            low_v, high_v = high_v, low_v
        return low_v, high_v
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            low_v = float(raw[0])
            high_v = float(raw[1])
        except (TypeError, ValueError):
            return None, None
        if high_v < low_v:
            low_v, high_v = high_v, low_v
        return low_v, high_v
    return None, None


def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    """Return the first non-empty value from known real row fields."""
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


_PLAY_STYLE_LABELS = {
    "TWO_WAY_F": "Two-way forward",
    "TWO_WAY_D": "Two-way defenseman",
    "TWO_WAY_W": "Two-way winger",
    "POWER_FORWARD": "Power forward",
    "SNIPER": "Sniper",
    "PLAYMAKER": "Playmaker",
    "GRINDER": "Grinder",
    "OFFENSIVE_D": "Offensive defenseman",
    "SHUTDOWN_D": "Shutdown defenseman",
    "PUCK_MOVING_G": "Puck-moving goalie",
    "BUTTERFLY": "Butterfly",
    "HYBRID": "Hybrid",
    "ATHLETIC": "Athletic",
}


def _humanize_play_style(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    if not s:
        return None
    key = s.upper().replace(" ", "_").replace("-", "_")
    if key in _PLAY_STYLE_LABELS:
        return _PLAY_STYLE_LABELS[key]
    if "_" in s or (s.isupper() and len(s) > 2):
        return s.replace("_", " ").title()
    return s


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
        if len(clean) >= 6:
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
        if len(clean) >= 6:
            break
    return clean


def _score_tier(score: float) -> str:
    if score >= 88:
        return "Elite"
    if score >= 78:
        return "Very High"
    if score >= 68:
        return "High"
    if score >= 58:
        return "Above Average"
    if score >= 50:
        return "Average"
    if score >= 42:
        return "Below Average"
    return "Disastrous"


def _attr_val(row: Dict[str, Any], *keys: str) -> Optional[float]:
    chapters = (row.get("chapter_profile") or {}).get("chapters") or {}
    for key in keys:
        for src in (row, chapters):
            raw = src.get(key) if isinstance(src, dict) else None
            if raw is not None:
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    return v
    return None


def _dossier_archetype_block(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = (
        row.get("dossier_archetype")
        or row.get("prospect_archetype")
        or row.get("archetype")
        or (row.get("player_comparison") or {}).get("archetype")
    )
    label = str(raw or "").strip()
    if not label:
        pos = _pos_bucket(str(row.get("position") or ""))
        label = "Goaltender" if pos == "G" else ("Defenseman" if pos == "D" else "Forward")
    human = label.replace("_", " ").title()
    if label.isupper() and len(label) > 3:
        human = label.replace("_", " ").title()
    return {"key": str(raw or label).upper().replace(" ", "_"), "label": human, "source": "player_creation"}


def _dossier_play_style_block(row: Dict[str, Any], archetype: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("dossier_play_style") or row.get("play_style") or row.get("playstyle")
    if not raw:
        chem = row.get("chemistry_playstyle")
        raw = chem or row.get("play_style_bucket")
    label = _humanize_play_style(raw)
    if not label:
        arch_key = str(archetype.get("key") or "").upper()
        arch_map = {
            "SNIPER": "North-south finisher",
            "PLAYMAKER": "East-west distributor",
            "POWER_FORWARD": "Power north-south",
            "TWO_WAY_F": "Two-way detail",
            "TWO_WAY_D": "Two-way defense",
            "OFFENSIVE_D": "Puck-moving defense",
            "SHUTDOWN_D": "Shutdown defense",
            "DEFENSIVE_D": "Shutdown defense",
            "BUTTERFLY_G": "Butterfly",
            "HYBRID_G": "Hybrid",
        }
        for prefix, style in arch_map.items():
            if arch_key.startswith(prefix) or prefix in arch_key:
                label = style
                break
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points")) or (goals + assists)
    ppg = _f(row.get("ppg"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    if not label and gp >= 10:
        if goals >= assists * 1.15 and goals >= 12:
            label = "Volume shooter"
        elif assists >= goals * 1.2 and assists >= 14:
            label = "Playmaking pace"
        elif ppg >= 0.95:
            label = "Primary scorer"
        elif ppg <= 0.35 and _pos_bucket(str(row.get("position") or "")) == "D":
            label = "Stay-at-home defense"
    return {"label": label or "Balanced", "source": "player_creation"}


def _dossier_tools_block(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    pos = _pos_bucket(str(row.get("position") or ""))
    if pos == "G":
        keys = [
            ("Glove", "glove", "glove_score"),
            ("Blocker", "blocker", "blocker_score"),
            ("Reflexes", "reflexes", "reflex_score"),
            ("Rebound", "rebound_control", "rebound_score"),
            ("Positioning", "positioning", "positioning_score"),
            ("Puck-handling", "puck_handling", "puck_handling_score"),
        ]
    else:
        keys = [
            ("Skating", "skating", "skating_rating"),
            ("Shot", "shooting", "shooting_rating"),
            ("Vision", "passing", "passing_rating", "hockey_iq", "iq_rating"),
            ("Defense", "defense", "def_rating"),
            ("Physical", "physical", "physical_rating"),
            ("IQ", "hockey_iq", "iq_rating"),
        ]
    out: List[Dict[str, Any]] = []
    for label, *fields in keys:
        val = _attr_val(row, *fields)
        if val is None:
            continue
        score = int(round(val))
        out.append({
            "label": label,
            "score": score,
            "grade": score,
            "tier": _score_tier(float(score)),
            "text": str(score),
            "locked": False,
        })
    return out


def _zone_map_from_tools(row: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    pos = _pos_bucket(str(row.get("position") or ""))
    by_label = {t["label"]: t for t in tools if t.get("label")}
    if pos == "G":
        def _avg(labels: List[str]) -> Optional[float]:
            vals = [float(by_label[l]["score"]) for l in labels if l in by_label]
            return sum(vals) / len(vals) if vals else None
        zones = {
            "rebound": _avg(["Rebound", "Reflexes"]),
            "angles": _avg(["Positioning"]),
            "range": _avg(["Puck-handling", "Glove"]),
        }
        return {"type": "crease", "zones": {k: {"value": round(v), "tier": _score_tier(v)} for k, v in zones.items() if v is not None}}
    def _avg(labels: List[str]) -> Optional[float]:
        vals = [float(by_label[l]["score"]) for l in labels if l in by_label]
        return sum(vals) / len(vals) if vals else None
    offensive = _avg(["Shot", "Vision"])
    transition = _avg(["Skating", "IQ"])
    defensive = _avg(["Defense"])
    play_style = str(row.get("dossier_play_style") or row.get("playstyle") or "").lower()
    if offensive and "shutdown" in play_style:
        offensive = offensive * 0.82
        defensive = (defensive or offensive or 0) * 1.08 if defensive else None
    if transition and "transition" in play_style:
        transition = min(99.0, transition * 1.06)
    zones = {}
    if defensive is not None:
        zones["defensive"] = {"value": round(defensive), "tier": _score_tier(defensive)}
    if transition is not None:
        zones["transition"] = {"value": round(transition), "tier": _score_tier(transition)}
    if offensive is not None:
        zones["offensive"] = {"value": round(offensive), "tier": _score_tier(offensive)}
    hero = "defensive" if pos == "D" and defensive and (not offensive or defensive >= offensive) else "offensive"
    return {"type": "rink", "hero_zone": hero, "zones": zones}


def _off_ice_frame_block(row: Dict[str, Any]) -> Dict[str, Any]:
    chapters = (row.get("chapter_profile") or {}).get("chapters") or {}
    char_score = _f(row.get("character_score") or chapters.get("character"))
    rows = []
    for label, key in (("Physical", "physical"), ("Mental", "mental"), ("Character", "character"), ("Transition", "transition")):
        val = _f(chapters.get(key) or row.get(key))
        if val <= 0 and label == "Character":
            val = char_score
        if val <= 0:
            continue
        detail = f"{label} {int(round(val))} — {_score_tier(val)}"
        if label == "Character" and char_score >= 82:
            detail = f"Character {int(round(val))} — recognized leader on file"
        elif label == "Character" and char_score > 0 and char_score < 50:
            detail = f"Character {int(round(val))} — disastrous attitude flagged"
        rows.append({"label": label, "score": int(round(val)), "pips": max(0, min(10, int(round(val / 10)))), "detail": detail})
    return {"rows": rows}


def _attribute_strength_weakness_entries(row: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strengths: List[Dict[str, Any]] = []
    weaknesses: List[Dict[str, Any]] = []
    pos = _pos_bucket(str(row.get("position") or ""))
    tools = _dossier_tools_block(row)
    for tool in tools:
        score = float(tool.get("score") or 0)
        label = str(tool.get("label") or "")
        if score >= 82:
            strengths.append({
                "title": f"{label} tool",
                "fact": f"{label} grades {int(round(score))} — {_score_tier(score)} for his age group",
                "context": "Carrying attribute on the scouting card",
                "confidence": "High",
            })
        elif score > 0 and score < 62:
            weaknesses.append({
                "title": f"{label} lag",
                "fact": f"{label} at {int(round(score))} — {_score_tier(score)} relative to draft peers",
                "context": "Development gap tied to on-ice tools",
                "confidence": "High",
            })
    hcm = _i(row.get("height_cm"), 0)
    weight = _f(row.get("weight"))
    if pos == "D" and hcm >= 193:
        strengths.append({
            "title": "Defensive frame",
            "fact": f"{_fmt_height(row)} frame with {int(weight)} lb listed weight" if weight else _fmt_height(row),
            "context": "Size profile matches modern NHL defense usage",
            "confidence": "High",
        })
    elif pos in ("C", "W") and weight >= 205 and hcm >= 185:
        strengths.append({
            "title": "Power frame",
            "fact": f"{int(weight)} lb on a {_fmt_height(row) or f'{hcm} cm'} frame",
            "context": "Physical profile supports net-front and board work",
            "confidence": "High",
        })
    char_score = _f(row.get("character_score"))
    if char_score >= 84:
        strengths.append({
            "title": "Leadership makeup",
            "fact": f"Character score {int(round(char_score))} — high coachability and room presence",
            "context": "Projects as a culture add through development",
            "confidence": "High",
        })
    elif char_score > 0 and char_score < 50:
        weaknesses.append({
            "title": "Character risk",
            "fact": f"Character score {int(round(char_score))} with attitude concerns on file",
            "context": "Off-ice reliability is a draft-day factor",
            "confidence": "High",
        })
    analytics = row.get("analytics") if isinstance(row.get("analytics"), dict) else {}
    war = _f(analytics.get("war"))
    if war >= 1.4:
        strengths.append({
            "title": "Analytics surplus",
            "fact": f"WAR {war:+.2f} relative to junior production baseline",
            "context": "Stat profile supports recent stock movement",
            "confidence": "High",
        })
    elif war <= -0.35 and _i(row.get("gp")) >= 15:
        weaknesses.append({
            "title": "Analytics drag",
            "fact": f"WAR {war:+.2f} — production and process trail draft slot",
            "context": "Stat ledger is a headwind on the public board",
            "confidence": "High",
        })
    return strengths, weaknesses


def _full_strengths_weaknesses(row: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_s = _evidence_strengths(row)
    base_w = _evidence_weaknesses(row)
    attr_s, attr_w = _attribute_strength_weakness_entries(row)
    seen_s: set = set()
    seen_w: set = set()
    strengths: List[Dict[str, Any]] = []
    weaknesses: List[Dict[str, Any]] = []
    for item in base_s + attr_s:
        title = str(item.get("title") or "")
        if not title or title in seen_s:
            continue
        seen_s.add(title)
        strengths.append(item)
        if len(strengths) >= 6:
            break
    for item in base_w + attr_w:
        title = str(item.get("title") or "")
        if not title or title in seen_w:
            continue
        seen_w.add(title)
        weaknesses.append(item)
        if len(weaknesses) >= 6:
            break
    return strengths, weaknesses


def _scout_report_narrative(
    row: Dict[str, Any],
    *,
    archetype: Dict[str, Any],
    play_style: Dict[str, Any],
    potential: Optional[Dict[str, Any]],
    projection: Optional[Dict[str, Any]],
) -> str:
    name = str(row.get("name") or "Prospect").strip()
    age = _i(row.get("age"), 18)
    gp = _i(row.get("gp") or row.get("games_played"))
    goals = _i(row.get("goals"))
    assists = _i(row.get("assists"))
    points = _i(row.get("points")) or (goals + assists)
    ppg = _f(row.get("ppg"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    league = str(row.get("league_display") or row.get("league") or "junior")
    arch = archetype.get("label") or "prospect"
    style = play_style.get("label") or "balanced"
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr") or row.get("scouted_overall_estimate"))
    peak = _f((potential or {}).get("rating") or row.get("potential_score"))
    floor = _f((potential or {}).get("floor"))
    role = str((projection or {}).get("label") or "")
    parts = [f"{name} ({age}Y) profiles as a {arch.lower()} with a {style.lower()} game in {league}."]
    if gp > 0:
        if _pos_bucket(str(row.get("position") or "")) == "G":
            sv = row.get("save_pct") or row.get("sv_pct")
            parts.append(f"Season line: {gp} GP, {row.get('wins', 0)} W, {sv} SV%.")
        else:
            parts.append(f"Season line: {goals}G-{assists}A-{points}P in {gp} GP ({ppg:.2f} PPG).")
    if ovr > 0:
        parts.append(f"Present ability grades near {int(round(ovr))} OVR.")
    if peak > 0 and not row.get("ceiling_hidden"):
        parts.append(f"Peak projection {int(round(peak))} OVR{f' with floor near {int(round(floor))}' if floor > 0 else ''}.")
    if role:
        parts.append(f"Projects as {role.lower()}.")
    return " ".join(parts)


def _development_trajectory_line(
    row: Dict[str, Any],
    *,
    potential: Optional[Dict[str, Any]],
    projection: Optional[Dict[str, Any]],
    ovr_v: float,
    pot_v: float,
    nhl_prob: float,
) -> str:
    if row.get("ceiling_hidden"):
        if ovr_v > 0:
            return f"Ceiling withheld — present ability {int(round(ovr_v))} OVR; upside must be inferred from production and tools."
        return "Ceiling withheld — trajectory depends on production, age, and tools."
    floor = _f((potential or {}).get("floor"))
    peak = _f((potential or {}).get("rating") or pot_v)
    likely = str((projection or {}).get("label") or (potential or {}).get("band") or "")
    gap = max(0.0, peak - ovr_v) if peak > 0 and ovr_v > 0 else 0.0
    bits = []
    if ovr_v > 0:
        bits.append(f"Now {int(round(ovr_v))} OVR")
    if peak > 0:
        bits.append(f"peak {int(round(peak))}")
    if floor > 0:
        bits.append(f"floor {int(round(floor))}")
    if likely:
        bits.append(f"likely outcome {likely.lower()}")
    if nhl_prob > 0:
        bits.append(f"{int(round(nhl_prob))}% NHL probability")
    if gap >= 14:
        bits.append("wide runway remaining")
    elif gap <= 6 and ovr_v >= 68:
        bits.append("narrow gap to peak")
    return " · ".join(bits) + "." if bits else "Development path still forming from backend grades."


def _intel_desk_tags(row: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    public_rank = _i(row.get("public_rank") or row.get("central_rank"))
    team_rank = _i(row.get("team_board_rank") or row.get("user_rank"))
    scout_rank = _i(row.get("rank"))
    if public_rank > 0 and team_rank > 0 and abs(public_rank - team_rank) >= 12:
        tags.append("Public vs personal board split")
    elif public_rank > 0 and scout_rank > 0 and abs(public_rank - scout_rank) >= 12:
        tags.append("Public vs personal board split")
    if bool(row.get("character_concerns")):
        tags.append("Character concern")
    if bool(row.get("is_overager") or row.get("overager")) or _i(row.get("age"), 18) >= 20:
        tags.append("Overager")
    stock_delta = _i(row.get("stock_delta") or row.get("stock_change"))
    if stock_delta == 0:
        nested = row.get("draft_stock") if isinstance(row.get("draft_stock"), dict) else {}
        stock_delta = _i(nested.get("delta_rank") or nested.get("deltaRank") or nested.get("weekly_stock_delta"))
    if stock_delta >= 50:
        tags.append("Stock surge (+50)")
    elif stock_delta <= -50:
        tags.append("Stock collapse (-50)")
    if bool(row.get("injured") or row.get("injury_status") or row.get("prospect_injured")):
        tags.append("Injury")
    return tags


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


def _rank_nhl_prior(rank: int) -> float:
    """Public board prior — thin user files should not crater elite prospects."""
    if rank <= 1:
        return 88.0
    if rank <= 3:
        return 84.0
    if rank <= 8:
        return 76.0
    if rank <= 15:
        return 66.0
    if rank <= 32:
        return 54.0
    return 40.0


def _analytics_model_grade(row: Dict[str, Any], analytics: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Stable analytics desk grade from production + rank (not random)."""
    analytics = analytics or {}
    rank = _i(row.get("rank"), 99)
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr") or row.get("scouted_overall_estimate"))
    score = ovr if ovr > 0 else 62.0
    war = _f(analytics.get("war"))
    gp = _i(row.get("gp") or row.get("games_played"))
    points = _i(row.get("points"))
    ppg = _f(row.get("ppg"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    if war:
        score += war * 3.5
    if ppg >= 1.5:
        score += 8.0
    elif ppg >= 1.0:
        score += 4.0
    elif ppg >= 0.7:
        score += 2.0
    if rank <= 3:
        score += 8.0
    elif rank <= 8:
        score += 5.0
    elif rank <= 15:
        score += 2.0
    return int(max(52.0, min(96.0, round(score))))


def _potential_block(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scout-visible ceiling block. `rating` is estimated ceiling, not final OVR.

    `probability` is NHL outcome odds derived from rank/tools/risk — not a renamed potential score.
    """
    pot = _f(row.get("expected_ceiling_estimate") or row.get("potential_score"))
    pr_lo, pr_hi = _parse_score_range(row.get("potential_range") or row.get("ceiling_range"))
    if pr_hi is not None and pr_hi > 0:
        pot = pr_hi
    elif pr_lo is not None and pr_lo > 0:
        pot = max(pot, pr_lo)
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
    rank_prior = _rank_nhl_prior(rank)
    # NHL probability from readiness signals — independent of raw ceiling number.
    prob = 48.0 + (conf - 50.0) * 0.18
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

    # Thin user scouting file: blend toward public board prior so #1 picks aren't 71% NHL.
    file_weight = max(0.0, min(1.0, (conf - 12.0) / 72.0))
    prob = rank_prior * (1.0 - file_weight) + prob * file_weight

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
    """Prospect analytics come from prospect_league_scoring.derive_prospect_analytics only."""
    existing = row.get("analytics") if isinstance(row.get("analytics"), dict) else {}
    if existing:
        return {k: v for k, v in existing.items() if v is not None}
    out: Dict[str, Any] = {}
    for key in (
        "war", "offensive_war", "defensive_war", "xgf_pct", "cf_pct", "ff_pct",
        "shooting_pct", "plus_minus", "primary_points", "shot_rate", "shots",
        "gsax", "quality_starts", "defensive_impact", "quality_of_competition",
        "quality_of_teammates", "toi",
    ):
        if row.get(key) is not None:
            out[key] = row.get(key)
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
    """Only stored weekly board history — never synthesize preseason arcs."""
    raw = row.get("rank_history") or row.get("stock_history") or []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if entry is None:
            continue
        if isinstance(entry, (int, float)) and int(entry) > 0:
            out.append({
                "date_label": f"Week {len(out) + 1}",
                "label": f"Week {len(out) + 1}",
                "rank": int(entry),
                "event_source": "stored",
            })
            continue
        if not isinstance(entry, dict):
            continue
        rank = _i(entry.get("rank") or entry.get("public_rank") or entry.get("board_rank") or entry.get("value"))
        if rank <= 0:
            continue
        label = str(
            entry.get("date_label")
            or entry.get("label")
            or entry.get("week_label")
            or entry.get("date")
            or entry.get("event")
            or f"Week {len(out) + 1}"
        )
        out.append({
            "date_label": label,
            "date": entry.get("date") or entry.get("calendar_iso") or label,
            "label": label,
            "rank": rank,
            "previous_rank": entry.get("previous_rank") or entry.get("prev_rank"),
            "movement": entry.get("movement") or entry.get("delta_rank") or entry.get("delta"),
            "reason": entry.get("reason") or entry.get("stock_reason"),
            "event_source": entry.get("event_source") or "weekly_board",
            "gp": entry.get("gp"),
            "injury": entry.get("injury") or entry.get("injured"),
        })
    return out


def _normalize_outcome_segments(segs: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    total = sum(max(0, _f(s.get("weight") or s.get("w"))) for s in segs)
    if total <= 0:
        return None
    out: List[Dict[str, Any]] = []
    for seg in segs:
        w = max(0.0, _f(seg.get("weight") or seg.get("w")))
        if w <= 0:
            continue
        pct = (w / total) * 100.0
        out.append({
            "key": str(seg.get("key") or ""),
            "label": str(seg.get("label") or ""),
            "weight": round(w, 2),
            "pct": round(pct, 1),
        })
    return out or None


def _outcome_distribution_block(row: Dict[str, Any], potential: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Career outcome split from rank, OVR, peak, floor, character, and analytics."""
    if not potential or potential.get("hidden"):
        return None
    nhl = _f(potential.get("nhl_probability") or potential.get("probability"))
    if nhl <= 0:
        return None
    rank = _i(row.get("rank"), 120)
    ovr = _f(row.get("current_ovr_estimate") or row.get("true_ovr") or row.get("scouted_overall_estimate"))
    peak = _f(potential.get("rating") or row.get("potential_score"))
    floor = _f(potential.get("floor"))
    gap = max(0.0, peak - ovr) if peak > 0 and ovr > 0 else 0.0
    band = str(row.get("outcome_band") or potential.get("band") or "Balanced")
    is_goalie = str(row.get("position") or "").upper() == "G"
    analytics = row.get("analytics") if isinstance(row.get("analytics"), dict) else {}
    war = _f(analytics.get("war"))
    char = _f(row.get("character_score"))
    bust_risk = 0.08
    if rank <= 5:
        bust_risk = 0.04
    elif rank <= 15:
        bust_risk = 0.06
    elif rank >= 120:
        bust_risk = 0.22
    if band == "Boom/Bust":
        bust_risk += 0.08
    if bool(row.get("character_concerns")):
        bust_risk += 0.06
    if war <= -0.4:
        bust_risk += 0.05
    elif war >= 1.5:
        bust_risk -= 0.03
    if gap >= 16:
        bust_risk += 0.04
    bust_risk = max(0.03, min(0.35, bust_risk))
    star_share = 0.10
    if rank <= 3 and peak >= 88:
        star_share = 0.38
    elif rank <= 8 and peak >= 84:
        star_share = 0.28
    elif rank <= 15 and peak >= 80:
        star_share = 0.20
    elif rank <= 32:
        star_share = 0.14
    if char >= 84:
        star_share += 0.04
    if war >= 1.8:
        star_share += 0.03
    star_share = max(0.05, min(0.45, star_share))
    non_nhl = max(0.0, 100.0 - nhl)
    bust = round(non_nhl * bust_risk)
    if is_goalie:
        minor = max(0.0, non_nhl - bust)
        starter = round(nhl * star_share)
        platoon = round(nhl * 0.32)
        backup = max(0.0, nhl - starter - platoon)
        segs = _normalize_outcome_segments([
            {"key": "bust", "label": "Bust", "weight": bust},
            {"key": "ahl", "label": "AHL/ECHL", "weight": minor},
            {"key": "mid", "label": "NHL Backup", "weight": backup},
            {"key": "top", "label": "Platoon", "weight": platoon},
            {"key": "star", "label": "Starter", "weight": starter},
        ])
    else:
        ahl = max(0.0, non_nhl - bust)
        star = round(nhl * star_share)
        top6 = round(nhl * (0.34 if floor >= 68 else 0.28))
        mid6 = max(0.0, nhl - star - top6)
        segs = _normalize_outcome_segments([
            {"key": "bust", "label": "Bust", "weight": bust},
            {"key": "ahl", "label": "AHL", "weight": ahl},
            {"key": "mid", "label": "Mid-6", "weight": mid6},
            {"key": "top", "label": "Top-6", "weight": top6},
            {"key": "star", "label": "Star+", "weight": star},
        ])
    if not segs:
        return None
    return {
        "source": "backend_model",
        "nhl_probability": round(nhl, 1),
        "outcome_band": band or None,
        "outcome_volatility": row.get("outcome_volatility"),
        "label": f"Career model · {band or 'Standard'} · {round(nhl)}% NHL",
        "segments": segs,
    }


def _scouting_history_block(row: Dict[str, Any], character_read: Optional[Dict[str, Any]], analytics: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
    """Backend scouting desk rows — only stored reports and stat-backed notes."""
    stored = row.get("scouting_desk") or row.get("scouting_history") or row.get("scout_reports")
    if isinstance(stored, list) and stored:
        return stored[:6]

    entries: List[Dict[str, Any]] = []
    grade = row.get("scouted_overall_estimate") or row.get("true_ovr") or row.get("current_ovr_estimate")
    notes = row.get("notes") or row.get("scout_reports") or row.get("reports") or []
    if isinstance(notes, list):
        for idx, note in enumerate(notes[:4]):
            text = note if isinstance(note, str) else str((note or {}).get("text") or (note or {}).get("summary") or "")
            text = text.strip()
            if not text:
                continue
            entries.append({
                "scout": str((note or {}).get("scout") if isinstance(note, dict) else "Regional scout"),
                "meta": str((note or {}).get("region") if isinstance(note, dict) else "Regional file"),
                "quote": text,
                "grade": (note or {}).get("grade") if isinstance(note, dict) else round(_f(grade), 0) if grade else None,
                "grade_label": "GRADE",
                "tone": "cyan" if idx % 2 else "green",
                "locked": False,
            })

    stock_reason = str(row.get("stock_reason") or row.get("movement_catalyst") or row.get("weekly_stock_reason") or "").strip()
    gp = _i(row.get("gp") or row.get("games_played"))
    points = _i(row.get("points"))
    ppg = _f(row.get("ppg"))
    if ppg <= 0 and gp > 0 and points > 0:
        ppg = points / gp
    war = _f((analytics or {}).get("war"))
    if stock_reason:
        entries.append({
            "scout": "Board movement",
            "meta": "Weekly stock",
            "quote": stock_reason,
            "grade": round(_f(grade), 0) if grade else None,
            "grade_label": "BOARD",
            "tone": "amber" if "fall" in stock_reason.lower() or "drop" in stock_reason.lower() else "green",
            "locked": False,
        })
    if gp > 0 and ppg > 0:
        stat_line = f"{points}P in {gp} GP · {ppg:.2f} PPG"
        if war:
            stat_line += f" · WAR {war:+.2f}"
        entries.append({
            "scout": "Stat ledger",
            "meta": "Season production",
            "quote": stat_line,
            "grade": _analytics_model_grade(row, analytics),
            "grade_label": "MODEL",
            "tone": "cyan",
            "locked": False,
        })
    if bool(row.get("injured") or row.get("prospect_injured") or row.get("injury_status")):
        inj_note = str(row.get("injury_note") or row.get("injury_status") or "Missed time this season")
        entries.append({
            "scout": "Medical file",
            "meta": "Injury",
            "quote": inj_note,
            "grade": None,
            "grade_label": "STATUS",
            "tone": "amber",
            "locked": False,
        })
    char_headline = (character_read or {}).get("headline")
    if char_headline and bool(row.get("character_concerns")):
        entries.append({
            "scout": "Character file",
            "meta": "Off-ice",
            "quote": str(char_headline),
            "grade": row.get("character_score"),
            "grade_label": "CHAR",
            "tone": "amber",
            "locked": False,
        })
    return entries or None


def _prospect_character_read(row: Dict[str, Any]) -> Dict[str, Any]:
    """Character read from backend character_score and psych traits — no interviews."""
    concerns = bool(row.get("character_concerns"))
    char_score = _f(
        row.get("character_score")
        or (row.get("chapter_profile") or {}).get("chapters", {}).get("character"),
    )
    if char_score <= 0:
        char_score = 0.0

    def _trait(label: str, key: str, fallback_delta: float = 0.0) -> Dict[str, Any]:
        val = _f(row.get(key))
        if val <= 0 and char_score > 0:
            try:
                from services.draft_ranking_logic import _stable_unit  # noqa: WPS433

                spread = (_stable_unit(
                    str(row.get("key") or row.get("id") or row.get("name") or ""),
                    f"char_trait_{key}",
                ) - 0.5) * 22.0
            except Exception:
                spread = 0.0
            val = max(20.0, min(99.0, char_score + fallback_delta + spread))
        tier = _score_tier(val) if val > 0 else "Unknown"
        return {"label": label, "tier": tier, "score": int(round(val)) if val > 0 else None}

    if concerns:
        return {
            "headline": str(row.get("attitude_label") or "Disastrous attitude"),
            "confidence": int(min(95, max(40, _f(row.get("scouting_confidence"), 55)))),
            "traits": [
                _trait("Competitive Drive", "competitiveness", -4),
                _trait("Coachability", "coachability", -10),
                _trait("Leadership", "maturity", -12),
                _trait("Work Ethic", "work_ethic", -6),
                _trait("Social Adjustment", "sociability", -14),
            ],
            "character_concerns": True,
            "attitude": "disastrous",
            "leader": False,
        }

    leader = char_score >= 84 or _f(row.get("leadership")) >= 82
    headline = "Recognized leader" if leader else (
        "Strong character" if char_score >= 74 else (
            "Average makeup" if char_score >= 58 else "Character questions" if char_score >= 50 else "Below average makeup"
        )
    )
    return {
        "headline": headline,
        "confidence": int(min(95, max(40, _f(row.get("scouting_confidence"), 55)))),
        "traits": [
            _trait("Competitive Drive", "competitiveness"),
            _trait("Coachability", "coachability"),
            _trait("Leadership", "leadership", -2),
            _trait("Work Ethic", "work_ethic"),
            _trait("Social Adjustment", "sociability", -4),
        ],
        "character_concerns": False,
        "attitude": "leader" if leader else "standard",
        "leader": leader,
    }


def build_prospect_profile(
    row: Dict[str, Any],
    *,
    roster_rows: Optional[List[Dict[str, Any]]] = None,
    team_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one draft board entry into modal profile payload."""
    row = dict(row or {})
    try:
        from services.draft_ranking_logic import _apply_character_integrity, _derive_dossier_identity

        _derive_dossier_identity(row, row.get("_player"))
        _apply_character_integrity(row, row.get("_player"))
    except Exception:
        pass
    actual = row.get("actual_stats") if isinstance(row.get("actual_stats"), dict) else {}
    gp = _i(actual.get("gp") or actual.get("games_played") or row.get("gp") or row.get("games_played"))
    goals = _i(actual.get("goals") if actual else row.get("goals"))
    assists = _i(actual.get("assists") if actual else row.get("assists"))
    points = _i(actual.get("points") if actual else row.get("points"))
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
    evidence_strengths, evidence_weaknesses = _full_strengths_weaknesses(row)
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
    if ovr_low is None or ovr_high is None:
        cur_lo, cur_hi = _parse_score_range(row.get("current_ovr_range"))
        if cur_lo is not None and cur_hi is not None:
            ovr_low = ovr_low if ovr_low is not None else cur_lo
            ovr_high = ovr_high if ovr_high is not None else cur_hi
    if ovr_low is None or ovr_high is None:
        pub_lo, pub_hi = _parse_score_range(
            {
                "low": row.get("public_ovr_low"),
                "high": row.get("public_ovr_high"),
            }
        )
        if pub_lo is not None and pub_hi is not None:
            ovr_low = ovr_low if ovr_low is not None else pub_lo
            ovr_high = ovr_high if ovr_high is not None else pub_hi
    if ovr_low is None and ovr_v > 0:
        spread = max(1, int(round((100 - scout_conf) / 25)))
        ovr_low = max(40, int(round(ovr_v)) - spread)
        ovr_high = min(99, int(round(ovr_v)) + spread)
    pot_range = row.get("potential_range")
    if pot_range is None:
        pot_range = row.get("ceiling_range")
    pot_low, pot_high = _parse_score_range(pot_range)
    if ceiling_hidden:
        pot_low = pot_high = None
    elif pot_low is None and pot_v > 0:
        spread = max(2, int(round((100 - scout_conf) / 20)))
        pot_low = max(50, int(round(pot_v)) - spread)
        pot_high = min(99, int(round(pot_v)) + spread)
    if pot_low is not None and pot_high is not None and not ceiling_hidden:
        try:
            from services.draft_ranking_logic import _peak_range_max_span

            rank_n = _i(row.get("rank")) or 999
            max_span = _peak_range_max_span(rank_n)
            if float(pot_high) - float(pot_low) > max_span:
                center = (float(pot_low) + float(pot_high)) / 2.0
                pot_low = int(round(max(50.0, center - max_span * 0.48)))
                pot_high = int(round(min(99.0, center + max_span * 0.52)))
        except Exception:
            pass

    headroom_delta = None
    if not ceiling_hidden and ovr_high is not None and pot_high is not None:
        try:
            headroom_delta = int(round(float(pot_high) - float(ovr_high)))
        except (TypeError, ValueError):
            headroom_delta = None

    file_depth_label = None
    if ovr_low is not None and ovr_high is not None:
        file_depth_label = f"Now {int(ovr_low)}–{int(ovr_high)} OVR"

    prospect_revision = _i(row.get("_prospect_revision") or row.get("prospect_revision")) or None

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

    character_read = _prospect_character_read(row)
    archetype_block = _dossier_archetype_block(row)
    play_style_block = _dossier_play_style_block(row, archetype_block)
    tools_block = _dossier_tools_block(row)
    zone_map = _zone_map_from_tools(row, tools_block)
    off_ice_frame = _off_ice_frame_block(row)
    intel_tags = _intel_desk_tags(row)
    scout_report = _scout_report_narrative(
        row,
        archetype=archetype_block,
        play_style=play_style_block,
        potential=potential,
        projection=projection,
    )
    development_trajectory = _development_trajectory_line(
        row,
        potential=potential,
        projection=projection,
        ovr_v=ovr_v,
        pot_v=pot_v,
        nhl_prob=nhl_prob,
    )
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
            bool(row.get("ovr_revealed")) or user_scout >= 72.0
        )) else (ovr_v or None),
        "overallRangeLow": ovr_low,
        "overallRangeHigh": ovr_high,
        "scoutedPotentialLow": pot_low,
        "scoutedPotentialHigh": pot_high,
        "now_range": (
            {"low": ovr_low, "high": ovr_high}
            if ovr_low is not None and ovr_high is not None
            else None
        ),
        "peak_range": (
            {"low": pot_low, "high": pot_high, "hidden": False}
            if not ceiling_hidden and pot_low is not None and pot_high is not None
            else {"low": None, "high": None, "hidden": True}
        ),
        "headroom_delta": headroom_delta,
        "file_depth_label": file_depth_label,
        "prospect_revision": prospect_revision,
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
        "strengths": [f"{e['title']} — {e['fact']}" for e in evidence_strengths if e.get("title")],
        "strengthsEvidence": evidence_strengths,
        "weaknessesEvidence": evidence_weaknesses,
        "projectionNotes": projection_notes,
        "concerns": [f"{e['title']} — {e['fact']}" for e in evidence_weaknesses if e.get("title")],
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
        "micro_summary": scout_report,
        "scout_report": scout_report,
        "development_trajectory": development_trajectory,
        "developmentTrajectory": development_trajectory,
        "archetype": archetype_block,
        "play_style_block": play_style_block,
        "tools": tools_block,
        "zone_map": zone_map,
        "off_ice_frame": off_ice_frame,
        "intel_desk_tags": intel_tags,
        "intelDeskTags": intel_tags,
        "character_read": character_read,
        "character_score": _i(row.get("character_score")) or None,
        "character_concerns": bool(row.get("character_concerns")),
        "attitude_label": row.get("attitude_label"),
        "chapter_profile": row.get("chapter_profile"),
        "chapterProfileFogged": bool((row.get("chapter_profile") or {}).get("fogged")),
        "chapterProfileHidden": bool((row.get("chapter_profile") or {}).get("hidden")),
        "scouted_percentage": _f(row.get("scouted_percentage"), 0.0) or None,
        "play_style": play_style_block.get("label") or _humanize_play_style(
            row.get("play_style")
            or row.get("playstyle")
            or row.get("dossier_play_style")
            or row.get("archetype")
        ),
        "prospect_role": row.get("prospect_role") or row.get("prospectRole"),
        "scouting_history": _scouting_history_block(row, character_read, analytics),
        "outcome_distribution": _outcome_distribution_block(row, potential),
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
    prospect_revision: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in entries or []:
        pid = str(row.get("key") or row.get("id") or "")
        if not pid:
            continue
        row_payload = dict(row)
        if prospect_revision is not None and not row_payload.get("_prospect_revision"):
            row_payload["_prospect_revision"] = int(prospect_revision)
        out[pid] = build_prospect_profile(row_payload, roster_rows=roster_rows, team_status=team_status)
    return out