"""On-ice fit read for a forward line or defence pair (D9).

Separate from the personality/familiarity chemistry score: this answers "do these
players' games fit together?" from their real attribute groups, in words the GM can
act on, e.g. "Excellent — elite playmaker + elite shooter, compatible tempo".

Reads:
  playmaker (pm_ avg), shooter (off_ avg), defender (def_), physical (phy_),
  tempo (skg_ avg), puck need (high offence with low off-puck defence = needs the
  puck), usage tier (overall).
"""
from __future__ import annotations

from typing import Any, Dict, List

VERDICTS = ((80, "Excellent"), (66, "Good"), (52, "Fair"), (0, "Poor"))


def _avg(p: Any, prefix: str, default: float = 65.0) -> float:
    r = getattr(p, "ratings", None) or {}
    vals = []
    for k, v in r.items():
        if str(k).startswith(prefix):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    return sum(vals) / len(vals) if vals else default


def _ovr(p: Any) -> float:
    for attr in ("overall", "ovr"):
        v = getattr(p, attr, None)
        try:
            v = v() if callable(v) else v
            f = float(v)
            return f * 99.0 if f <= 1.5 else f
        except (TypeError, ValueError):
            continue
    return 70.0


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    n = str(getattr(ident, "name", None) or getattr(p, "name", "") or "Player")
    return n.split()[-1] if n else "Player"


def _shoots(p: Any) -> str:
    for src in (getattr(p, "identity", None), p):
        v = getattr(src, "shoots", None) if src is not None else None
        if v is None:
            v = getattr(src, "handedness", None) if src is not None else None
        if v is not None:
            s = str(getattr(v, "value", v)).upper()
            if s.startswith("L"):
                return "L"
            if s.startswith("R"):
                return "R"
    return ""


def profile(p: Any) -> Dict[str, Any]:
    pm, sh, df, ph, sk = _avg(p, "pm_"), _avg(p, "off_"), _avg(p, "def_"), _avg(p, "phy_"), _avg(p, "skg_")
    tags: List[str] = []
    if pm >= 80:
        tags.append("elite playmaker")
    elif pm >= 74 and pm >= sh + 3:
        tags.append("playmaker")
    if sh >= 80:
        tags.append("elite shooter")
    elif sh >= 74 and sh >= pm + 3:
        tags.append("shooter")
    if df >= 76:
        tags.append("defensive conscience")
    if ph >= 76:
        tags.append("physical presence")
    if sk >= 80:
        tags.append("burner")
    # A puck-dominant dual threat (passes AND shoots at a high level, little off-puck game).
    puck_need = (max(pm, sh) - df) if min(pm, sh) >= 76 else 0
    return {
        "name": _name(p), "pm": pm, "sh": sh, "df": df, "ph": ph, "sk": sk,
        "ovr": _ovr(p), "tags": tags, "needs_puck": puck_need >= 12,
        "shoots": _shoots(p),
    }


def _verdict(score: float) -> str:
    for cut, label in VERDICTS:
        if score >= cut:
            return label
    return "Poor"


def line_fit_read(players: List[Any], unit: str = "forward") -> Dict[str, Any]:
    ps = [p for p in (players or []) if p is not None]
    if len(ps) < 2:
        return {}
    pr = [profile(p) for p in ps]
    score = 62.0
    reasons: List[str] = []
    concerns: List[str] = []

    def has(tag: str) -> List[Dict[str, Any]]:
        return [x for x in pr if any(t == tag or t.endswith(" " + tag) for t in x["tags"])]

    makers, shooters = has("playmaker"), has("shooter")
    elite_m = [x for x in pr if "elite playmaker" in x["tags"]]
    elite_s = [x for x in pr if "elite shooter" in x["tags"]]
    if makers and shooters and (set(id(x) for x in makers) != set(id(x) for x in shooters) or len(pr) >= 3):
        if elite_m and elite_s and elite_m[0] is not elite_s[0]:
            score += 16
            reasons.append(f"Elite playmaker ({elite_m[0]['name']}) + elite shooter ({elite_s[0]['name']})")
        else:
            score += 10
            reasons.append(f"Playmaker ({makers[0]['name']}) feeds a shooter ({shooters[0]['name']})")
    elif len(shooters) >= 2 and not makers and unit == "forward":
        score -= 6
        concerns.append("Two shooters, nobody to set them up")
    elif len(makers) >= 2 and not shooters and unit == "forward":
        score -= 6
        concerns.append("Two passers, no finisher — they'll pass up shots")
    needy = [x for x in pr if x["needs_puck"]]
    if len(needy) >= 2:
        score -= 10
        concerns.append(f"Both {needy[0]['name']} and {needy[1]['name']} need the puck")
    spread_sk = max(x["sk"] for x in pr) - min(x["sk"] for x in pr)
    if spread_sk <= 7:
        score += 6
        reasons.append("Compatible tempo")
    elif spread_sk >= 15:
        score -= 8
        slow = min(pr, key=lambda x: x["sk"])
        concerns.append(f"Tempo mismatch — {slow['name']} can't keep pace")
    spread_ovr = max(x["ovr"] for x in pr) - min(x["ovr"] for x in pr)
    if spread_ovr <= 6:
        score += 5
        reasons.append("Compatible usage tier")
    elif spread_ovr >= 14:
        score -= 6
        low = min(pr, key=lambda x: x["ovr"])
        concerns.append(f"Usage mismatch — {low['name']} is a tier below his linemates")
    if unit == "forward":
        if any("defensive conscience" in x["tags"] for x in pr):
            score += 5
            reasons.append("A defensive conscience covers for the skill guys")
        elif sum(x["df"] for x in pr) / len(pr) < 66:
            score -= 6
            concerns.append("Nobody minds the defensive zone")
        if any("physical presence" in x["tags"] for x in pr) and (makers or shooters):
            score += 3
            reasons.append("Physical presence wins pucks for the skill")
    else:
        a, b = pr[0], pr[1]
        mobile = max(a["sk"], b["sk"]) >= 74 or max(a["pm"], b["pm"]) >= 74
        stay = max(a["df"], b["df"]) >= 74
        if mobile and stay and abs((a["pm"] - a["df"]) - (b["pm"] - b["df"])) >= 6:
            score += 12
            mover = a if a["pm"] - a["df"] > b["pm"] - b["df"] else b
            other = b if mover is a else a
            reasons.append(f"Puck mover ({mover['name']}) + stay-at-home partner ({other['name']})")
        elif a["needs_puck"] and b["needs_puck"]:
            score -= 8
            concerns.append("Two offensive D — rush chances against")
        elif max(a["df"], b["df"]) < 68:
            score -= 8
            concerns.append("Neither is a reliable defender")
        if a["shoots"] and b["shoots"]:
            if a["shoots"] != b["shoots"]:
                score += 4
                reasons.append("Left/right shot balance")
            else:
                score -= 4
                concerns.append("Same-side shots — one plays his off side")
    score = max(10.0, min(98.0, score))
    verdict = _verdict(score)
    summary = "; ".join(reasons[:3]) if verdict in ("Excellent", "Good") and reasons else "; ".join((concerns or reasons)[:3])
    return {
        "score": int(round(score)),
        "verdict": verdict,
        "summary": f"Chemistry: {verdict}" + (f" — {summary}" if summary else ""),
        "reasons": reasons,
        "concerns": concerns,
        "players": [{"name": x["name"], "tags": x["tags"]} for x in pr],
    }
