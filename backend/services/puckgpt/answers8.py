"""Reasoning modes: the question categories the fixed handlers didn't cover.

Each mode is a *way of thinking*, not a topic: describe, group, interpret, check a
premise, project, plan for a contingency, optimise under the cap, and so on. Every
answer is built from the save, names the yardstick it used, and says plainly what
the data cannot show. Earlier layers (team planner, verdicts, events, game records,
analytics rules, intent handlers) keep everything they already answer.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .answers import _answer, _cap_space, _resolve_players, _standing, _team_label
from .data import LiveView

Answer = Dict[str, Any]

MODES: List[Tuple[str, str]] = [
    ("self_reflect", r"\bwhat (information|data|info|stats?) (would|do) you need\b|\bwhat can'?t you (see|know|answer|tell)\b|\blimits? of (your|the) (data|knowledge)\b|\bhow (do|did) you (know|work that out|calculate|decide)\b|\bwhat (don'?t|do not) you (know|track)\b"),
    ("multi_step", r"\b(biggest|main|top) (weakness|hole|need|problem)\b.*\b(targets?|who (could|can|should|would) (fix|fill|help)|trade for)\b|\bfind (realistic |me )?(trade )?targets\b.*\b(weakness|hole|need)\b"),
    ("adversarial", r"\bprove me (right|wrong)\b|\bchange my mind\b|\bam i (right|wrong)\b|\bi (think|believe|bet) .*\b(actually|secretly|really)\b|\bconvince me\b"),
    ("premise", r"\bis [a-z' .-]+ (actually |really |just )?(bad|terrible|awful|the problem|overrated|washed)\b[^?]*\bor (is it|it'?s)\b|\bwhy (is|are|was|were) (our|my|the) (goalie|goaltender|keeper|defen[cs]e|offen[cs]e|power ?play|penalty ?kill|coach|top line|captain|[a-z]+) (so )?(terrible|awful|bad|garbage|trash|horrible|the (problem|reason))\b|\bresponsible for (every|all|each) (loss|losses|goal)\b|\bwhy (does|do) (he|they|our [a-z]+) (always|never)\b"),
    ("exploratory", r"\bsomething (interesting|surprising|cool|weird|unusual|crazy|wild|fun|random)\b|\bsomething i (might|may|probably) (not|haven'?t)\b|\bsurprise me\b|\bfun fact\b|\bhidden (trend|stat|story)\b|\banything (interesting|unusual|weird)\b"),
    ("creative", r"\b(write|tell|invent|create|give|spin|craft|make)\b[^?]{0,40}?\b(story|narrative|recap|article|headline|press release|hype (video|script)|poem)\b"),
    ("scenario", r"\b(trade|sell|move) (two|2|three|3|both|our|some) (veterans?|vets|older players)\b.*\b(promote|call up|bring up)\b|\bpromote (our|the|my) prospects\b.*\b(trade|sell)\b"),
    ("counterfactual", r"\bwhat (might|would|could) have happened if\b|\bif (we|i) had (drafted|signed|kept|traded|not traded|taken)\b|\bshould (we|i) have (drafted|taken|picked|kept|signed)\b|\bin hindsight\b"),
    ("contingency", r"\b(if|when) (our|my|the) (starting |number one |no\.? ?1 |top )?(goalie|starter|captain|best player|top (center|centre|d|defen[cs]eman)|[a-z]+) (gets|got|goes|went|is) (hurt|injured|down|out)\b|\bbackup plan\b|\bcontingency\b|\bplan b\b"),
    ("hypothetical", r"\bif (we|i) (trade|traded|lose|lost|move|moved|sell|sold|dump|waive|buy out) [a-z' .-]+?,? (what|how)\b|\bwhat (if|happens if|would happen if) (we|i) (trade|traded|lose|lost|move|moved|sell|sold|dump|scratch|bench|waive|bought out|buy out|add|added|sign|signed)\b|\bwhat if [a-z .'-]+ (got traded|left|retired|got hurt)\b"),
    ("optimize", r"\b(strongest|best|optimal|ideal) (roster|lineup|team) (we|i) can (build|make|afford|ice)\b|\boptimi[sz]e\b.*\b(roster|lineup|cap)\b|\bbest (roster|lineup) under the cap\b"),
    ("allocate", r"\b(how|where) should (i|we) (spend|allocate|use|invest) (our |my |the )?(cap|cap space|money|budget|room)\b|\ballocate\b.*\b(cap|space|money)\b|\bwhat should i do with (my|our) cap\b"),
    ("prioritize", r"\bwhich (roster )?(weakness|hole|need|problem|area) (should|do|would) (i|we) (address|fix|tackle|solve) first\b|\bprioriti[sz]e\b|\bwhat should (i|we) fix first\b|\bbiggest (roster )?(weakness|hole|need)\b|\bwhere are we (weakest|thinnest)\b"),
    ("multi_criteria", r"\bbest combination of\b|\bbest (balance|mix) of\b|\b(age|ability|salary|potential|cost)\b[^?]*,[^?]*\b(age|ability|salary|potential|cost)\b[^?]*\b(target|player|option)\b"),
    ("retro", r"\b(past|previous|my|our) (management |gm )?(decisions?|moves?)\b.*\b(best|biggest|worst|positive|negative|impact)\b|\b(best|worst) (decision|move) (i|we)('ve| have)? (ever )?made\b"),
    ("coach", r"\bcoach(es|ing)?\b.*\b(good|bad|right|wrong|decision|hire|hiring|fire|firing|mistake)\b|\b(hire|hiring|fire|firing) (the |a |our )?coach\b"),
    ("history", r"\b(over|in) the (last|past) (\d+|two|three|four|five|six|seven|eight|nine|ten) (seasons|years)\b|\bseason by season\b|\bsince (we|i) took over\b|\bfranchise (history|over time)\b"),
    ("trend", r"\bhow (much|far) (has|have|did) (our|my|the) [a-z ]+ (improved|declined|dropped|changed|grown|fallen|gotten (better|worse))\b|\bwhy (has|have|did) (our|my|the) [a-z ]+ (declined|dropped|fallen|slipped|improved|gotten (worse|better)|fallen off)\b|\b(has|is) (our|my) [a-z ]+ (improving|declining|getting (better|worse))\b"),
    ("anomaly", r"\b(far |way |well )?(above|below|over|under) (their |his )?expectations\b|\b(over|under)perform(ing|ers)?\b|\boutliers?\b|\banomal\w*\b"),
    ("award_race", r"\b(who('s| is| will)?|whos) (going to |gonna )?(win|winning|lead|leading|favou?rite for|the favou?rite (for|in)) (the )?(art ross|rocket|richard|hart|vezina|norris|calder|selke|ted lindsay|lindsay|jennings)\b|\b(art ross|rocket richard|hart|vezina|norris|calder|selke) (race|odds|favou?rite|leader|chances?)\b"),
    ("playoff_race", r"\bwho (will|is going to|gonna|'?ll) make (the )?playoffs\b|\bplayoff (race|picture|odds|chances?) (in|for) the (atlantic|metro|metropolitan|central|pacific|east|west|eastern|western)\b|\bwho (makes|is making|gets) (in|the playoffs)\b|\b(atlantic|metro|metropolitan|central|pacific|eastern|western) (division |conference )?(race|standings|playoff)\b|\bin the (atlantic|metro|metropolitan|central|pacific|east|west)\b.*\bplayoffs?\b|\bplayoffs?\b.*\bin the (atlantic|metro|metropolitan|central|pacific|east|west)\b"),
    ("overview", r"\boverview\b|\bstate of (the|my|our) (franchise|team|club|organi[sz]ation)\b|\bwhere (do we|are we) stand\b|\bsummar(y|i[sz]e) (my|our|the) (team|franchise|season|club)\b|\bhow('s| is) (my|our|the) franchise\b|\bbrief me\b|\bscouting report on (my|our) team\b"),
    ("strengths", r"\b(biggest |main |key |greatest )?strengths?\b|\bwhat (are we|is my team|is our team) good at\b|\bwhat do we do (well|best)\b|\bour best (area|asset)\b"),
    ("categorize", r"\b(group|sort|categori[sz]e|split|bucket|classify) (my|our|the) (roster|players|team|lineup)\b|\bwhich (of my |of our )?players are (prospects|starters|declining|veterans)\b|\bwho('s| is) (rising|declining|peaking)\b.*\b(and|or)\b"),
    ("sustain", r"\bsustainable\b|\bunsustainable\b|\bregress(ion)?\b.*\b(coming|due|expect)\b|\b(shooting|save) (percentage|pct|%)\b.*\b(real|luck|lucky|hold|last)\b|\bpdo\b|\b(luck|lucky)\b.*\bor (good|real|skill)\b|\bfor real\b"),
    ("interpret", r"\bwhat does (it|this|that|our [a-z ]+) (suggest|mean|tell (us|you|me)|say|indicate)\b|\b(bad|poor|losing|good|winning) record despite\b|\bdespite (strong|good|great|bad|poor|weak) (scoring|defen[cs]e|goaltending|possession)\b"),
    ("correlate", r"\b(do|does|are|is) (our|my|the) (highest[- ]paid|most expensive|best[- ]paid|richest|biggest contracts?)\b|\bpay (vs\.?|versus|and) (production|points|performance)\b|\bcorrelat\w*\b|\brelationship between\b|\bdoes (age|size|pay|salary|ice time) (matter|affect|predict)\b"),
    ("pattern", r"\bpatterns?\b|\bwhy (do )?we keep (losing|winning|blowing)\b|\bwhen do we (usually )?(lose|win)\b|\bwhat do (our|the|my) (losses|wins) have in common\b|\btrends? in (our )?(losses|wins|results)\b|\bwhy do we keep (losing|winning)\b"),
    ("predict", r"\bon pace\b|\bpace for\b|\bprojected?\b|\bprojection\b|\bfinish (the season )?with\b|\bend (up |the season )?with\b|\bhow many (points|goals|wins|assists) (will|could|might|would|is)\b|\bpredict\w*\b|\bforecast\b"),
    ("confidence", r"\bhow (confident|sure|certain) are you\b|\bconfidence (level|in|that)\b|\bhow reliable is (that|this)\b"),
    ("speculate", r"\b(infer|outlook|future)\b[^?]*\bprospects?\b|\bprospects?'?s? (future|outlook|ceiling)\b|\bcould [a-z' ]{1,30}?(eventually |ever |one day |someday )?(become|be|turn into) (a |an )?(superstar|star|elite|franchise player|all[- ]star|first[- ]liner|number one)\b|\bsuperstar potential\b|\bfuture (superstar|star|franchise player)\b|\bwill (he|[a-z]+) (ever )?be (a |an )?(star|superstar|elite)\b"),
    ("normative", r"\bshould (we|i) (trade|move|sell|keep|re-?sign|extend|let go|part with) (a |an |our |my |the )?(popular|fan[- ]favou?rite|aging|ageing|old|older|veteran|captain|franchise player|face of the franchise)\b"),
    ("exciting", r"\bmost (exciting|fun|entertaining|electric|dynamic|skilled|fun to watch|watchable)\b|\bfan favou?rite\b|\bhighlight[- ]reel\b"),
    ("debate", r"\bis (building|drafting|rebuilding|tanking|spending|buying)\b.*\bbetter than\b|\b(draft|drafting) (vs\.?|versus|or) (free agency|fa|signing)\b|\b(free agency|fa) (vs\.?|versus|or) (the )?draft\b"),
    ("theory", r"\bwhat makes (a |an )?(successful|good|great|winning|elite) (nhl )?(rebuild|team|contender|gm|general manager|dynasty|franchise|goalie|defen[cs]eman|power ?play)\b|\bhow do (you|teams|nhl teams) (rebuild|win the cup|build a (contender|dynasty)|tank)\b"),
]


def match(q: str) -> Optional[str]:
    q = " ".join(str(q or "").lower().replace("’", "'").split())
    for mode, pat in MODES:
        if re.search(pat, q):
            return mode
    return None


# --------------------------------------------------------------------------- shared helpers
def _tid(view: LiveView, q: str, slots: Dict[str, Any]) -> str:
    teams = [str(t) for t in (slots.get("teams") or [])]
    if re.search(r"\b(we|us|our|my)\b", q) or not teams:
        return view.user_tid
    return teams[0]


def _nhl(view: LiveView, tid: str) -> List[Dict[str, Any]]:
    return [r for r in view.team_rows(tid) if r]


def _cap(r: Dict[str, Any]) -> float:
    return float((r.get("contract") or {}).get("cap_hit_m") or 0.0)


def _is_d(r: Dict[str, Any]) -> bool:
    return (r.get("pos") or "") in ("D", "LD", "RD")


def _grp(r: Dict[str, Any]) -> str:
    p = r.get("pos") or ""
    return "G" if p == "G" else ("D" if _is_d(r) else ("C" if p == "C" else "W"))


def _league_rank(view: LiveView, tid: str, key: str, high_good: bool = True) -> Tuple[int, int, float]:
    rows = [r for r in view.standings() if r.get("gp")]
    def val(r):
        gp = float(r.get("gp") or 1)
        if key in ("gf_pg", "ga_pg"):
            return float(r.get("gf" if key == "gf_pg" else "ga") or 0) / gp
        if key == "pts_pct":
            return float(r.get("pts") or 0) / (2 * gp)
        return float(r.get(key) or 0)
    rows.sort(key=val, reverse=high_good)
    for i, r in enumerate(rows):
        if str(r.get("team_id") or r.get("id") or "") == str(tid):
            return i + 1, len(rows), val(r)
    return len(rows), len(rows), 0.0


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _group_strength(view: LiveView, tid: str) -> Dict[str, Tuple[float, int]]:
    """Average overall of each position group's top slots, and its league rank."""
    slots = {"C": 3, "W": 6, "D": 4, "G": 1}
    def avg(t, g):
        vals = sorted((r["ovr"] for r in _nhl(view, t) if _grp(r) == g), reverse=True)[:slots[g]]
        return statistics.mean(vals) if vals else 0.0
    out = {}
    for g in slots:
        mine = avg(tid, g)
        league = sorted((avg(t, g) for t in view.teams), reverse=True)
        out[g] = (mine, 1 + sum(1 for v in league if v > mine))
    return out


_GNAME = {"C": "centres", "W": "wingers", "D": "defence", "G": "goaltending"}


def _player(view: LiveView, slots: Dict[str, Any], conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = _resolve_players(view, slots)
    if rows:
        return rows[0]
    for pid in (conv.get("players") or [])[:1]:
        if pid in view.players:
            return view.row(pid)
    return None


# --------------------------------------------------------------------------- modes
def overview(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    team = _team_label(view, tid)
    st = _standing(view, tid) or {}
    gp = int(st.get("gp") or 0)
    rank, n, _ = _league_rank(view, tid, "pts_pct")
    groups = _group_strength(view, tid)
    best = min(groups, key=lambda g: groups[g][1])
    worst = max(groups, key=lambda g: groups[g][1])
    space = _cap_space(view, tid)
    rows = _nhl(view, tid)
    young = [r for r in view.team_rows(tid, ("Prospect", "AHL", "ECHL")) if (r.get("pot") or 0) >= 80]
    top = max((r for r in rows if r.get("pos") != "G"), key=lambda r: (r.get("pts") or 0), default=None)
    from .derived import team_status

    head = f"The {team}: " + (f"{st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)}, {_ord(rank)} of {n} by points %." if gp else "no games played yet.")
    notes = [f"Identity: {team_status(view, tid)}. Strongest group: {_GNAME[best]} ({_ord(groups[best][1])} in the league); weakest: {_GNAME[worst]} ({_ord(groups[worst][1])}).",
             f"Cap space: ${space:.2f}M." if space is not None else "",
             f"Leading scorer: {top['name']} ({top.get('pts', 0)} pts)." if top and gp else "",
             f"Pipeline: {len(young)} prospects with 80+ potential." if young else "Pipeline: no prospects with 80+ potential.",
             f"Average age {statistics.mean([r['age'] for r in rows]):.1f}." if rows else ""]
    return _answer(head, analysis=[x for x in notes if x], followups=["What are our biggest strengths?", "Which weakness should I fix first?", "How confident are you we make the playoffs?"])


def strengths(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    team = _team_label(view, tid)
    weak = bool(re.search(r"\bweak", q))
    groups = _group_strength(view, tid)
    facts = []
    for g, (v, r) in groups.items():
        facts.append((r, f"{_GNAME[g].capitalize()}: {v:.1f} average overall in the top slots ({_ord(r)})."))
    for key, label, high in (("gf_pg", "Scoring", True), ("ga_pg", "Keeping it out", False), ("pp_pct", "Power play", True), ("pk_pct", "Penalty kill", True)):
        r, n, val = _league_rank(view, tid, key, high)
        if val:
            shown = f"{val * 100:.1f}%" if key in ("pp_pct", "pk_pct") else f"{val:.2f} per game"
            facts.append((r, f"{label}: {shown} ({_ord(r)})."))
    facts.sort(key=lambda x: x[0], reverse=weak)
    lead = facts[0][1] if facts else "Not enough data yet."
    return _answer(f"{'Biggest weakness' if weak else 'Biggest strength'} for the {team} — {lead}",
                   analysis=[f for _, f in facts[1:6]] + ["Ranked against all 32 clubs; position groups use the top 3 C, 6 W, 4 D and the starting goalie."],
                   followups=["Which weakness should I fix first?", "Give me an overview of my franchise"])


def categorize(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    buckets: Dict[str, List[str]] = {"Prospects (developing)": [], "Rising": [], "Established core": [], "Depth": [], "Declining veterans": []}
    for r in view.team_rows(tid, ("NHL", "AHL", "ECHL", "Prospect")):
        age, ovr, pot = r.get("age") or 0, r.get("ovr") or 0, r.get("pot") or 0
        start = r.get("season_start_ovr") or ovr
        if r.get("level") != "NHL" and age <= 23:
            b = "Prospects (developing)"
        elif age <= 25 and pot - ovr >= 3:
            b = "Rising"
        elif age >= 31 and (ovr < start - 0.5 or age >= 34):
            b = "Declining veterans"
        elif ovr >= 80:
            b = "Established core"
        else:
            b = "Depth"
        buckets[b].append(f"{r['name']} ({r.get('pos')}, {age}, {ovr:.0f}/{pot:.0f})")
    rows = [[k, len(v), ", ".join(v[:6]) + (" …" if len(v) > 6 else "")] for k, v in buckets.items()]
    return _answer(f"{_team_label(view, tid)} grouped: " + ", ".join(f"{len(v)} {k.split(' (')[0].lower()}" for k, v in buckets.items()) + ".",
                   analysis=["Prospects: outside the NHL and 23 or younger. Rising: 25 or younger with 3+ points to grow. Declining: 31+ and below their opening-night overall, or 34+. Core: 80+ overall."],
                   table={"cols": ["Group", "Count", "Players (pos, age, OVR/POT)"], "rows": rows})


def sustain(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    p = _player(view, slots, conv)
    if p and p.get("pos") != "G":
        st = view.stats.get(p["id"]) or {}
        g, sog = float(st.get("g") or 0), float(st.get("sog") or 0)
        sh = g / sog if sog else 0.0
        lg = _league_sh(view)
        verdict = "likely to cool off" if sh > lg * 1.35 and sog >= 40 else ("due for more goals" if sh < lg * 0.7 and sog >= 40 else "about sustainable")
        return _answer(f"{p['name']} is shooting {sh * 100:.1f}% on {int(sog)} shots against a league {lg * 100:.1f}% — {verdict}.",
                       analysis=["Shooting % swings a lot under ~100 shots; finishers sustain 2-4 points above average, few sustain 8+."])
    st = _standing(view, tid) or {}
    rows = _nhl(view, tid)
    g = sum(float((view.stats.get(r["id"]) or {}).get("g") or 0) for r in rows if r.get("pos") != "G")
    s = sum(float((view.stats.get(r["id"]) or {}).get("sog") or 0) for r in rows if r.get("pos") != "G")
    sh = g / s if s else 0.0
    from .answers7 import _league_sv

    goalie = max((r for r in rows if r.get("pos") == "G"), key=lambda r: r.get("gp") or 0, default=None)
    sv = float(goalie.get("sv_pct") or 0) if goalie else 0.0
    pdo = (sh + sv) * 100 if sv else None
    lg_sh, lg_sv = _league_sh(view), _league_sv(view)
    lg_pdo = (lg_sh + lg_sv) * 100
    if pdo is None:
        return _answer("Not enough shots yet to judge.")
    verdict = "running hot — expect some regression" if pdo > lg_pdo + 1.5 else ("running cold — results should improve" if pdo < lg_pdo - 1.5 else "close to normal — results look earned")
    return _answer(f"{_team_label(view, tid)}: {sh * 100:.1f}% shooting + {sv:.3f} starter save % = PDO {pdo:.1f} (league {lg_pdo:.1f}), {verdict}.",
                   analysis=["PDO (shooting % + save %) drifts back toward the league level over a season; teams far above it usually fall back.",
                             f"League shooting {lg_sh * 100:.1f}%, save % {lg_sv:.3f}.", f"Record so far: {st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)}."])


def _league_sh(view) -> float:
    g = s = 0.0
    for st in view.stats.values():
        if isinstance(st, dict):
            g += float(st.get("g") or 0)
            s += float(st.get("sog") or 0)
    return g / s if s else 0.095


def interpret(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    gf_r, n, gf = _league_rank(view, tid, "gf_pg", True)
    ga_r, _, ga = _league_rank(view, tid, "ga_pg", False)
    pts_r, _, pct = _league_rank(view, tid, "pts_pct", True)
    from .answers6 import _select, _summary

    s = _summary(_select(view, tid, {}, None))
    exp_pct = (gf ** 2) / (gf ** 2 + ga ** 2) if gf + ga else 0.5
    gap = pct - exp_pct
    lines = [f"Scoring {_ord(gf_r)} ({gf:.2f}/game), goals against {_ord(ga_r)} ({ga:.2f}/game), points % {_ord(pts_r)} ({pct:.3f}).",
             f"Goal totals alone predict about a {exp_pct:.3f} points % (Pythagorean); you're {'above' if gap > 0 else 'below'} that by {abs(gap):.3f}.",
             f"One-goal games: {s['one_goal']}; overtime/shootout: {s['ot']}."]
    if gap < -0.03:
        reading = "the results are worse than the play — close games are going the other way, which usually evens out"
    elif gap > 0.03:
        reading = "the results are better than the play — winning the close ones, which rarely lasts"
    elif gf_r <= n // 3 and ga_r > 2 * n // 3:
        reading = "it's a defence problem: the scoring is real but the goals against cancel it"
    else:
        reading = "the record is a fair reflection of the goal differential"
    return _answer(f"What the numbers suggest for the {_team_label(view, tid)}: {reading}.", analysis=lines,
                   followups=["Are our results sustainable?", "What patterns appear in our losses?"])


def correlate(view, q, slots, conv) -> Answer:
    tid = None if re.search(r"\b(league|nhl|all teams|every team)\b", q) else _tid(view, q, slots)
    rows = [r for t in ([tid] if tid else list(view.teams)) for r in _nhl(view, t) if r.get("pos") != "G" and (r.get("gp") or 0) >= 15 and _cap(r) > 0]
    if len(rows) < 6:
        return _answer("Not enough skaters with games and contracts to measure that yet.")
    xs, ys = [_cap(r) for r in rows], [float(r.get("pts") or 0) for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    var = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    r = cov / var if var else 0.0
    strength = "strong" if abs(r) >= 0.6 else ("moderate" if abs(r) >= 0.35 else "weak")
    top_paid = sorted(rows, key=_cap, reverse=True)[:5]
    top_pts = {x["id"] for x in sorted(rows, key=lambda x: -(x.get("pts") or 0))[:5]}
    overlap = sum(1 for x in top_paid if x["id"] in top_pts)
    scope = _team_label(view, tid) if tid else "the league"
    return _answer(f"Pay and points in {scope}: a {strength} {'positive' if r > 0 else 'negative'} link (r = {r:.2f}). {overlap} of the 5 highest-paid are among the 5 top scorers.",
                   analysis=["Correlation across skaters with 15+ GP; it shows a link, not that pay causes points (defensive roles and age skew it).",
                             "Highest paid: " + ", ".join(f"{x['name']} ${_cap(x):.1f}M/{x.get('pts', 0)} pts" for x in top_paid) + "."],
                   followups=["Best bargain", "Worst contract"])


def pattern(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    from .answers6 import _select

    games = _select(view, tid, {}, None)
    if len(games) < 10:
        return _answer("Need at least 10 games to see patterns.")
    losses = [g for g in games if not g["won"]]
    wins = [g for g in games if g["won"]]
    def pct(lst, f):
        return 100.0 * sum(1 for g in lst if f(g)) / len(lst) if lst else 0.0
    facts = []
    facts.append((abs(pct(losses, lambda g: g["home"]) - pct(games, lambda g: g["home"])), f"{pct(losses, lambda g: not g['home']):.0f}% of losses came on the road (road games are {pct(games, lambda g: not g['home']):.0f}% of the schedule)."))
    facts.append((pct(losses, lambda g: g["gf"] <= 1), f"Scored 0-1 goals in {pct(losses, lambda g: g['gf'] <= 1):.0f}% of losses — {'scoring dries up' if pct(losses, lambda g: g['gf'] <= 1) > 45 else 'not a scoring drought issue'}."))
    facts.append((pct(losses, lambda g: g["ga"] >= 5), f"Allowed 5+ in {pct(losses, lambda g: g['ga'] >= 5):.0f}% of losses — {'blowouts are the problem' if pct(losses, lambda g: g['ga'] >= 5) > 30 else 'few blowouts'}."))
    facts.append((pct(losses, lambda g: abs(g['gf'] - g['ga']) <= 1 or g['extra']), f"{pct(losses, lambda g: abs(g['gf'] - g['ga']) <= 1 or g['extra']):.0f}% of losses were by one goal or in overtime."))
    strong = {}
    for g in games:
        o = g["opp"]
        strong.setdefault(o, [0, 0])
        strong[o][0 if g["won"] else 1] += 1
    nemesis = max(strong.items(), key=lambda kv: (kv[1][1] - kv[1][0], kv[1][1]))
    if nemesis[1][1] >= 2:
        facts.append((nemesis[1][1] * 10, f"Toughest opponent: {_team_label(view, nemesis[0])} ({nemesis[1][0]}-{nemesis[1][1]})."))
    facts.sort(key=lambda x: -x[0])
    return _answer(f"Pattern in the {_team_label(view, tid)}'s {len(losses)} losses: {facts[0][1]}", analysis=[f for _, f in facts[1:]] + [f"Wins average {statistics.mean([g['gf'] for g in wins]) if wins else 0:.1f} goals for; losses {statistics.mean([g['gf'] for g in losses]) if losses else 0:.1f}."],
                   followups=["What does our record suggest?", "Our record on the road"])


def predict(view, q, slots, conv) -> Answer:
    p = _player(view, slots, conv)
    season_len = 82
    if p:
        gp = int(p.get("gp") or 0)
        team_gp = int((_standing(view, p.get("team_id")) or {}).get("gp") or gp or 1)
        left = max(0, season_len - team_gp)
        if gp < 5:
            return _answer(f"{p['name']} has only {gp} games; too early to project.")
        rate_pts, rate_g = (p.get("pts") or 0) / gp, (p.get("g") or 0) / gp
        avail = gp / max(1, team_gp)
        proj_pts = (p.get("pts") or 0) + rate_pts * left * avail
        proj_g = (p.get("g") or 0) + rate_g * left * avail
        spread = 1.96 * (max(1.0, rate_pts * left * avail)) ** 0.5
        return _answer(f"{p['name']} is on pace for about {proj_pts:.0f} points ({proj_g:.0f} goals) — likely range {max(p.get('pts') or 0, proj_pts - spread):.0f}-{proj_pts + spread:.0f}.",
                       analysis=[f"{p.get('pts', 0)} pts in {gp} GP ({rate_pts:.2f}/game) with {left} team games left, at his {avail * 100:.0f}% availability so far.",
                                 "Range is a rough 95% band from scoring randomness; injuries, role changes and slumps aren't in it."])
    tid = _tid(view, q, slots)
    st = _standing(view, tid) or {}
    gp = int(st.get("gp") or 0)
    if gp < 5:
        return _answer("Too early in the season to project.")
    pace = float(st.get("pts") or 0) / gp * season_len
    return _answer(f"The {_team_label(view, tid)} are on pace for about {pace:.0f} points ({st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)} after {gp}).",
                   analysis=["~96 points is the usual playoff line; ±8 points is a normal swing over the rest of a season.", "Pace assumes the same points rate; it doesn't adjust for schedule strength."],
                   followups=["How confident are you we make the playoffs?", "Are our results sustainable?"])


def confidence(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    st = _standing(view, tid) or {}
    gp = int(st.get("gp") or 0)
    if gp < 5:
        return _answer("Low confidence on anything season-level yet: under 5 games is mostly noise.",
                       analysis=["Confidence rises with sample size: ~20 games tells you a little, ~40 a lot, and season-long rates settle near 60."])
    pace = float(st.get("pts") or 0) / gp * 82
    sd = 8.0 * ((82 - gp) / 82) ** 0.5
    import math

    z = (pace - 96.0) / max(1.0, sd)
    p = 0.5 * (1 + math.erf(z / 2 ** 0.5))
    level = "high" if abs(z) >= 1.5 else ("moderate" if abs(z) >= 0.7 else "low")
    return _answer(f"Playoff chances for the {_team_label(view, tid)}: about {p * 100:.0f}% — {level} confidence.",
                   analysis=[f"On pace for {pace:.0f} points against a ~96-point line, with about ±{sd:.0f} points of uncertainty from the {82 - gp} games left.",
                             "Confidence here is about sample size and margin; injuries and trades can move it more than the math allows for."])


def speculate(view, q, slots, conv) -> Answer:
    p = _player(view, slots, conv)
    if not p:
        return _answer("Which player? Name him and I'll weigh his odds.")
    age, ovr, pot = p.get("age") or 0, float(p.get("ovr") or 0), float(p.get("pot") or 0)
    room = max(0.0, pot - ovr)
    years = max(0, 26 - age)
    reach = min(pot, ovr + room * min(1.0, 0.25 * years + 0.2))
    star = 88.0
    chance = max(0.02, min(0.9, (pot - 84.0) / 12.0 if pot > 84 else 0.03))
    if ovr >= star:
        head = f"{p['name']} is already elite at {ovr:.0f}."
    else:
        head = f"{p['name']} ({age}, {ovr:.0f} now, {pot:.0f} potential): roughly a {chance * 100:.0f}% shot at becoming an 88+ star; a fair expectation is about {reach:.0f} by his mid-20s."
    return _answer(head, analysis=["Potential is the ceiling, not the forecast — most players settle below it. Growth slows sharply after 24-25.",
                                   f"His season so far: {p.get('pts', 0)} pts in {p.get('gp', 0)} GP." if p.get("gp") else "No NHL games this season, so production doesn't inform this yet."],
                   followups=["Who has the highest ceiling?", "Is he worth it?"])


def exciting(view, q, slots, conv) -> Answer:
    tid = None if re.search(r"\b(league|nhl)\b", q) else _tid(view, q, slots)
    rows = [r for t in ([tid] if tid else list(view.teams)) for r in _nhl(view, t) if r.get("pos") != "G" and (r.get("gp") or 0) >= 10]
    if not rows:
        return _answer("Not enough games yet.")
    def score(r):
        gp = r.get("gp") or 1
        youth = max(0, 26 - (r.get("age") or 26)) * 0.4
        return (r.get("g") or 0) / gp * 6 + (r.get("pts") or 0) / gp * 3 + (r.get("shg") or 0) * 0.8 + (r.get("gwg") or 0) * 0.5 + youth + max(0, (r.get("pot") or 0) - 85) * 0.2
    rows.sort(key=score, reverse=True)
    r = rows[0]
    return _answer(f"Most exciting: {r['name']} ({r.get('team')}) — {r.get('g', 0)} goals, {r.get('pts', 0)} points in {r.get('gp')} games at {r.get('age')}.",
                   analysis=["'Exciting' is subjective; this blends goal rate, points rate, shorthanded and game-winning goals, youth and ceiling.",
                             "Next: " + ", ".join(f"{x['name']} ({x.get('team')})" for x in rows[1:5]) + "."])


def premise(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    team = _team_label(view, tid)
    from .answers7 import _league_sv

    named = _player(view, slots, {})
    if named and named.get("pos") == "G" or re.search(r"goalie|goaltend|keeper", q):
        g = named if (named and named.get("pos") == "G") else max((r for r in _nhl(view, tid) if r.get("pos") == "G"), key=lambda r: r.get("gp") or 0, default=None)
        if not g or not g.get("sv_pct"):
            return _answer("No goalie stats yet to check that.")
        lg = _league_sv(view)
        ga_r, n, ga = _league_rank(view, tid, "ga_pg", False)
        st = view.stats.get(g["id"]) or {}
        sa = float(st.get("shots_against") or st.get("goalie_shots_against") or 0)
        gp = max(1, int(g.get("gp") or 1))
        if g["sv_pct"] >= lg:
            head = f"The data doesn't support blaming {g['name']}: his .{int(g['sv_pct'] * 1000):03d} save % is at or above the league's .{int(lg * 1000):03d}."
            why = f"He's facing {sa / gp:.1f} shots a game; the goals against ({_ord(ga_r)} of {n}) come from volume in front of him, not his saves."
        else:
            saved = (lg - g["sv_pct"]) * sa
            head = f"Partly fair: {g['name']} is at .{int(g['sv_pct'] * 1000):03d} vs league .{int(lg * 1000):03d} — about {saved:.0f} extra goals over {gp} games."
            why = "That's a real gap, but no goalie is responsible for every loss — check the shots and chances allowed too."
        return _answer(head, analysis=[why], followups=["Which goalie should start?", "What patterns appear in our losses?"])
    # Generic premise check: compare the named area with league rank.
    area = next((k for k, pat in (("ga_pg", r"defen[cs]e|keeping|goals against"), ("gf_pg", r"offen[cs]e|scoring|attack"),
                                   ("pp_pct", r"power ?play"), ("pk_pct", r"penalty ?kill")) if re.search(pat, q)), None)
    if not area:
        return _answer("I'd want to check that claim against the numbers first — which part of the team do you mean (goalie, defence, offence, power play, penalty kill)?")
    r, n, v = _league_rank(view, tid, area, area != "ga_pg")
    shown = f"{v * 100:.1f}%" if area in ("pp_pct", "pk_pct") else f"{v:.2f} per game"
    fair = r > (2 * n) // 3
    head = (f"The numbers back you up: the {team} rank {_ord(r)} of {n} there ({shown})." if fair else
            f"The numbers don't really support that: the {team} rank {_ord(r)} of {n} there ({shown}), {'mid-pack' if r > n // 3 else 'near the top'}.")
    return _answer(head, analysis=["Bad stretches feel bigger than they are; the full-season rank is the fairer test."], followups=["What are our biggest strengths?", "Why are we losing?"])


def adversarial(view, q, slots, conv) -> Answer:
    p = _player(view, slots, conv)
    tid = _tid(view, q, slots)
    from .answers7 import _impact_rows

    rows = _impact_rows(view, tid)
    if not rows:
        return _answer("Not enough games to test that claim yet.")
    if not p and re.search(r"\bworst\b", q):
        by_ovr = sorted((r for r in rows if r.get("pos") != "G"), key=lambda r: r.get("ovr") or 0)
        p = by_ovr[0] if by_ovr else None
    if not p:
        return _answer("Name the player and I'll test the claim against the numbers.")
    me = next((r for r in rows if r["id"] == p["id"]), None)
    if not me:
        return _answer(f"{p['name']} hasn't played enough (15+ GP) to judge.")
    rank = 1 + sum(1 for r in rows if r["impact"] > me["impact"])
    value_rank = 1 + sum(1 for r in rows if (r["impact"] - 0.6 * r["cap"]) > (me["impact"] - 0.6 * me["cap"]))
    verdict = "You're right" if value_rank <= 3 else ("Partly right" if value_rank <= len(rows) // 3 else "The numbers say no")
    return _answer(f"{verdict}: {p['name']} ranks {_ord(rank)} of {len(rows)} in impact and {_ord(value_rank)} in impact for the money.",
                   analysis=[f"His line: {me['why']}; cap ${me['cap']:.2f}M.", "Impact = goals given or cost versus what his rating should produce; value also subtracts salary."],
                   followups=["Who carried us?", "Best bargain"])


def exploratory(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    facts = []
    from .answers6 import _select, _summary

    s = _summary(_select(view, tid, {}, None))
    if s["gp"] >= 10:
        if s["one_goal_n"]:
            facts.append((abs(s["one_goal_w"] / s["one_goal_n"] - 0.5) * 10, f"You're {s['one_goal']} in one-goal games — {'clutch or lucky' if s['one_goal_w'] / s['one_goal_n'] > 0.6 else ('snakebitten' if s['one_goal_w'] / s['one_goal_n'] < 0.4 else 'right on the coin flip')}."))
        home = _summary(_select(view, tid, {"venue": "home"}, None))
        road = _summary(_select(view, tid, {"venue": "road"}, None))
        if home["gp"] and road["gp"]:
            facts.append((abs(home["pts_pct"] - road["pts_pct"]) * 10, f"Home {home['pts_pct']:.3f} vs road {road['pts_pct']:.3f} points %."))
    rows = _nhl(view, tid)
    sk = [r for r in rows if r.get("pos") != "G" and (r.get("gp") or 0) >= 10]
    if sk:
        d = max((r for r in sk if _is_d(r)), key=lambda r: r.get("pts") or 0, default=None)
        if d and (d.get("pts") or 0) >= 10:
            f_rank = 1 + sum(1 for r in sk if (r.get("pts") or 0) > (d.get("pts") or 0))
            facts.append((6 - min(5, f_rank), f"Defenceman {d['name']} is {_ord(f_rank)} on the team in scoring ({d.get('pts')} pts)."))
        pm = max(sk, key=lambda r: abs(r.get("plus_minus") or 0))
        facts.append((abs(pm.get("plus_minus") or 0) / 6, f"{pm['name']} sits at {pm.get('plus_minus', 0):+d}, the most extreme plus/minus on the team."))
        old = max(sk, key=lambda r: r.get("age") or 0)
        facts.append((1.5, f"Oldest regular: {old['name']} at {old.get('age')} with {old.get('pts', 0)} pts."))
    if not facts:
        return _answer("Play a few more games and I'll have something to dig up.")
    facts.sort(key=lambda x: -x[0])
    return _answer(f"Something you might not have noticed: {facts[0][1]}", analysis=[f for _, f in facts[1:4]], followups=["Tell me something else interesting", "What patterns appear in our losses?"])


def creative(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    team = _team_label(view, tid)
    from .answers6 import _select, _summary

    games = _select(view, tid, {"playoffs": bool(re.search(r"playoff|postseason", q))}, None)
    if not games:
        games = _select(view, tid, {}, None)
    if not games:
        return _answer(f"No games yet to build the {team} story from.")
    s = _summary(games)
    last5 = games[-5:]
    best = max(games, key=lambda g: g["gf"] - g["ga"])
    worst = min(games, key=lambda g: g["gf"] - g["ga"])
    top = max((r for r in _nhl(view, tid) if r.get("pos") != "G"), key=lambda r: r.get("pts") or 0, default=None)
    form = "".join("W" if g["won"] else "L" for g in last5)
    story = [f"It's been a {_rec(s)} ride for the {team}, {s['gf']} goals scored and {s['ga']} allowed.",
             f"The high point came {best['date']}: a {best['gf']}-{best['ga']} statement {'at home' if best['home'] else 'on the road'} against the {_team_label(view, best['opp'])}.",
             f"The low point, {worst['date']}: a {worst['gf']}-{worst['ga']} night against the {_team_label(view, worst['opp'])} that nobody in the room wants to rewatch."]
    if top:
        story.append(f"Through it all, {top['name']} has carried the mail — {top.get('g', 0)} goals, {top.get('pts', 0)} points.")
    story.append(f"Lately the sheet reads {form}. " + ("The momentum is real." if form.count("W") >= 4 else ("The pressure is building." if form.count("L") >= 4 else "Every night is a coin flip.")))
    return _answer(f"THE {team.upper()} STORY", analysis=story + ["Every score, date and stat above is from this save; only the wording is invented."])


def _rec(s: Dict[str, Any]) -> str:
    return f"{s['w']}-{s['l']}-{s['otl']}"


def prioritize(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    groups = _group_strength(view, tid)
    n = len(view.teams) or 32
    needs = sorted(groups.items(), key=lambda kv: -kv[1][1])
    impact_w = {"G": 1.4, "D": 1.15, "C": 1.1, "W": 1.0}
    ranked = sorted(needs, key=lambda kv: -(kv[1][1] / n) * impact_w[kv[0]])
    g, (v, r) = ranked[0]
    return _answer(f"Fix {_GNAME[g]} first: {v:.1f} average overall in the top slots, {_ord(r)} of {n}.",
                   analysis=["Order: " + " → ".join(f"{_GNAME[k]} ({_ord(rr)})" for k, (_, rr) in ranked) + ".",
                             "Weighted by how much each spot moves results: a starting goalie most, then defence, centres, wingers."],
                   followups=["Identify our biggest weakness and find trade targets", "How should I allocate my cap space?"])


def allocate(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    space = _cap_space(view, tid) or 0.0
    groups = _group_strength(view, tid)
    n = len(view.teams) or 32
    weights = {g: max(0.05, r / n) * {"G": 1.4, "D": 1.15, "C": 1.1, "W": 1.0}[g] for g, (_, r) in groups.items()}
    total = sum(weights.values())
    reserve = min(space * 0.15, 2.0)
    spend = max(0.0, space - reserve)
    rows = [[_GNAME[g].capitalize(), _ord(groups[g][1]), f"${spend * w / total:.2f}M"] for g, w in sorted(weights.items(), key=lambda kv: -kv[1])]
    return _answer(f"With ${space:.2f}M of room: keep ~${reserve:.2f}M in reserve, put the rest where you rank lowest — {rows[0][0].lower()} first ({rows[0][2]}).",
                   analysis=["Split is proportional to how far each group trails the league, weighted by impact (goalie > D > C > W).",
                             "A reserve covers injuries and in-season trades; in a cap-rising year you can run it thinner."],
                   table={"cols": ["Group", "League rank", "Suggested spend"], "rows": rows})


def multi_criteria(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    space = _cap_space(view, tid) or 0.0
    from .answers3 import _sellers

    sellers = set(_sellers(view))
    cands = [view.row(p) for p, (t, lv) in view.where.items() if lv == "FA" or (lv == "NHL" and t in sellers)]
    cands = [r for r in cands if r and r.get("pos") != "G" and (r.get("ovr") or 0) >= 74]
    if not cands:
        return _answer("No realistic targets found.")
    def score(r):
        cap = _cap(r)
        age_s = max(0.0, 1 - abs((r.get("age") or 27) - 25) / 10)
        ability = ((r.get("ovr") or 0) - 70) / 20
        pot = max(0.0, ((r.get("pot") or 0) - (r.get("ovr") or 0))) / 10
        cost = 1 - min(1.0, cap / max(1.0, space + 2.0))
        return 0.40 * ability + 0.20 * age_s + 0.15 * pot + 0.25 * cost
    cands.sort(key=score, reverse=True)
    top = cands[:8]
    r0 = top[0]
    return _answer(f"Best all-round target: {r0['name']} ({r0.get('team') or 'FA'}, {r0.get('pos')}, {r0.get('age')}, {r0.get('ovr', 0):.0f}/{r0.get('pot', 0):.0f}, ${_cap(r0):.2f}M).",
                   analysis=["Weights: ability 40%, cost against your space 25%, age (peak ~25) 20%, growth left 15%. Sellers and free agents only.", f"Your cap space: ${space:.2f}M."],
                   table={"cols": ["Player", "Team", "Pos", "Age", "OVR/POT", "Cap", "Score"],
                          "rows": [[r["name"], r.get("team") or "FA", r.get("pos"), r.get("age"), f"{r.get('ovr', 0):.0f}/{r.get('pot', 0):.0f}", f"${_cap(r):.2f}M", f"{score(r) * 100:.0f}"] for r in top]})


def multi_step(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    groups = _group_strength(view, tid)
    g, (v, r) = max(groups.items(), key=lambda kv: kv[1][1])
    space = _cap_space(view, tid) or 0.0
    from .answers3 import _sellers, _strength_with, _cup_odds

    sellers = set(_sellers(view))
    cands = [view.row(p) for p, (t, lv) in view.where.items() if (lv == "FA" or (lv == "NHL" and t in sellers))]
    cands = [c for c in cands if c and _grp(c) == g and (c.get("age") or 40) <= 33 and _cap(c) <= space + 4.0 and (c.get("ovr") or 0) > v]
    cands.sort(key=lambda c: -(c.get("ovr") or 0))
    base = _strength_with(view, tid)
    rows = []
    for c in cands[:5]:
        new = _strength_with(view, tid, add=c)
        rows.append([c["name"], c.get("team") or "FA", f"{c.get('ovr', 0):.0f}", f"${_cap(c):.2f}M", f"+{new - base:.2f}", f"{_cup_odds(view, base):.1f}% → {_cup_odds(view, new):.1f}%"])
    steps = [f"Step 1 — weakest group: {_GNAME[g]} ({v:.1f} avg in the top slots, {_ord(r)} in the league).",
             f"Step 2 — targets: {len(cands)} realistic {_GNAME[g]} upgrades from sellers and free agents within ${space + 4.0:.1f}M (your space plus a matching salary back).",
             "Step 3 — effect: team strength and rough Cup odds with each one added."]
    if not rows:
        return _answer(f"Your biggest hole is {_GNAME[g]}, but no realistic upgrade fits right now.", analysis=steps[:2])
    return _answer(f"Weakest spot is {_GNAME[g]}; best fix: {rows[0][0]} ({rows[0][1]}), lifting team strength {rows[0][4]}.", analysis=steps,
                   table={"cols": ["Target", "Team", "OVR", "Cap", "Strength", "Cup odds"], "rows": rows})


def contingency(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    rows = _nhl(view, tid)
    want_g = bool(re.search(r"goalie|goaltend|starter|keeper", q))
    p = _player(view, slots, conv)
    if not p:
        pool = [r for r in rows if (r.get("pos") == "G") == want_g] if want_g else [r for r in rows if r.get("pos") != "G"]
        p = max(pool, key=lambda r: (r.get("gp") or 0) if want_g else (r.get("ovr") or 0), default=None)
    if not p:
        return _answer("Couldn't find who you mean.")
    grp = _grp(p)
    depth = sorted([r for r in view.team_rows(tid, ("NHL", "AHL")) if _grp(r) == grp and r["id"] != p["id"]], key=lambda r: -(r.get("ovr") or 0))
    from .answers3 import _strength_with, _best_target

    base = _strength_with(view, tid)
    without = _strength_with(view, tid, remove_ids=(p["id"],))
    space = _cap_space(view, tid) or 0.0
    tgt = _best_target(view, "G" if grp == "G" else grp, space)
    plan = [f"Next man up: {depth[0]['name']} ({depth[0].get('ovr', 0):.0f}, {depth[0].get('level')})." if depth else "No internal replacement at that position.",
            f"Team strength drops {base - without:.2f} without {p['name']}.",
            f"Call-up after that: {depth[1]['name']} ({depth[1].get('ovr', 0):.0f})." if len(depth) > 1 else "",
            f"Outside option: {tgt['name']} ({tgt.get('team') or 'FA'}, {tgt.get('ovr', 0):.0f}, ${_cap(tgt):.2f}M)." if tgt else "",
            f"Cap space to work with: ${space:.2f}M."]
    return _answer(f"If {p['name']} goes down: lean on {depth[0]['name'] if depth else 'a trade'}, then {'add ' + tgt['name'] if tgt else 'look for a rental'} if it's long-term.",
                   analysis=[x for x in plan if x], followups=["Which goalie should start?", "Injury report"])


def hypothetical(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    p = _player(view, slots, conv)
    if not p and re.search(r"\bcaptain\b", q):
        cap = next((r for r in _nhl(view, tid) if "captain" in str(r.get("role") or "").lower()), None)
        p = cap or max(_nhl(view, tid), key=lambda r: r.get("ovr") or 0, default=None)
    if not p:
        return _answer("Which player? Name him and I'll run the what-if.")
    from .answers3 import _strength_with, _cup_odds, _rank_of

    adding = bool(re.search(r"\b(add|added|sign|signed|acquire|get)\b", q)) and p.get("team_id") != tid
    base = _strength_with(view, tid)
    new = _strength_with(view, tid, add=p) if adding else _strength_with(view, tid, remove_ids=(p["id"],))
    space = (_cap_space(view, tid) or 0.0) + (-_cap(p) if adding else _cap(p))
    verb = "adding" if adding else "losing"
    return _answer(f"What-if {verb} {p['name']}: team strength {base:.2f} → {new:.2f}, rough Cup odds {_cup_odds(view, base):.1f}% → {_cup_odds(view, new):.1f}%.",
                   analysis=[f"Cap space would become ${space:.2f}M.",
                             f"He's producing {p.get('pts', 0)} pts in {p.get('gp', 0)} GP." if p.get("gp") else "",
                             "Strength uses the top 12 F, 6 D and the starting goalie; it can't capture chemistry, leadership or what comes back in a trade."],
                   followups=[f"Should I trade {p['name']}?", "Best trade targets"])


def scenario(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    rows = _nhl(view, tid)
    vets = sorted([r for r in rows if (r.get("age") or 0) >= 30 and r.get("pos") != "G"], key=lambda r: -_cap(r))[:2]
    pros = sorted([r for r in view.team_rows(tid, ("AHL", "Prospect", "ECHL")) if r.get("pos") != "G"], key=lambda r: -(r.get("ovr") or 0))[:2]
    if not vets or not pros:
        return _answer("Need at least one veteran (30+) and one prospect to model that.")
    from .answers3 import _strength_with, _cup_odds

    base = _strength_with(view, tid)
    after = _strength_with(view, tid, remove_ids=tuple(v["id"] for v in vets))
    promoted = after
    try:
        # the prospects step into the spots the veterans leave
        cur_rows = [r for r in rows if r["id"] not in {v["id"] for v in vets}] + pros
        f = sorted((r["ovr"] for r in cur_rows if _grp(r) in ("C", "W")), reverse=True)[:12]
        d = sorted((r["ovr"] for r in cur_rows if _grp(r) == "D"), reverse=True)[:6]
        g = sorted((r["ovr"] for r in cur_rows if _grp(r) == "G"), reverse=True)[:1]
        promoted = statistics.mean(f) * 0.5 + statistics.mean(d) * 0.3 + g[0] * 0.2 if f and d and g else after
    except Exception:
        promoted = after
    freed = sum(_cap(v) for v in vets)
    age_now = statistics.mean([r["age"] for r in rows]) if rows else 0
    return _answer(f"Trade {', '.join(v['name'] for v in vets)} and promote {', '.join(p['name'] for p in pros)}: strength {base:.2f} → {promoted:.2f} now, Cup odds {_cup_odds(view, base):.1f}% → {_cup_odds(view, promoted):.1f}%, ${freed:.2f}M freed.",
                   analysis=[f"Prospects promoted: " + ", ".join(f"{p['name']} {p.get('ovr', 0):.0f}/{p.get('pot', 0):.0f}" for p in pros) + " — they should grow 2-5 a year while the veterans decline.",
                             f"Roster average age drops from about {age_now:.1f}.", "What comes back in the trades isn't modelled; that usually offsets part of the drop."])


def counterfactual(view, q, slots, conv) -> Answer:
    from .answers7 import _draft

    tid = view.user_tid
    p = _player(view, slots, conv)
    drafted = []
    for pid, pl in view.players.items():
        d = _draft(pl)
        if d["overall"] and d["year"]:
            drafted.append((pid, d))
    if p:
        d = next((dd for pid, dd in drafted if pid == p["id"]), None)
        if not d:
            return _answer(f"No draft record for {p['name']}.")
        same = [view.row(pid) for pid, dd in drafted if dd["year"] == d["year"] and dd["overall"] and abs(dd["overall"] - d["overall"]) <= 10 and pid != p["id"]]
        same = [r for r in same if r]
        better = sorted([r for r in same if (r.get("ovr") or 0) > (p.get("ovr") or 0)], key=lambda r: -(r.get("ovr") or 0))
        head = (f"{p['name']} went #{d['overall']} in {d['year']}. Within 10 picks of him, {len(better)} players are better today"
                + (f" — the best is {better[0]['name']} ({better[0].get('ovr', 0):.0f} vs {p.get('ovr', 0):.0f})." if better else "; he was the right call."))
        return _answer(head, analysis=["Hindsight compares current overall only; at the time, scouting saw ranges, not these outcomes."],
                       table={"cols": ["Player", "Pick", "OVR", "POT"], "rows": [[r["name"], "", f"{r.get('ovr', 0):.0f}", f"{r.get('pot', 0):.0f}"] for r in better[:6]]})
    mine = [(pid, d) for pid, d in drafted if view.where.get(pid, ("", ""))[0] == tid]
    if not mine:
        return _answer("Name a player and I'll compare him with who was taken around him.")
    worst = None
    for pid, d in mine:
        r = view.row(pid)
        peers = [view.row(x) for x, dd in drafted if dd["year"] == d["year"] and dd["overall"] and abs(dd["overall"] - d["overall"]) <= 10]
        best_peer = max((x for x in peers if x), key=lambda x: x.get("ovr") or 0, default=None)
        if best_peer and r:
            miss = (best_peer.get("ovr") or 0) - (r.get("ovr") or 0)
            if worst is None or miss > worst[0]:
                worst = (miss, r, best_peer, d)
    if not worst or worst[0] <= 0:
        return _answer("Your picks hold up — nobody taken near them is clearly better today.")
    miss, r, peer, d = worst
    return _answer(f"Biggest 'what if': at #{d['overall']} in {d['year']} you have {r['name']} ({r.get('ovr', 0):.0f}); {peer['name']} went nearby and is {peer.get('ovr', 0):.0f} now.",
                   analysis=["Compares each of your current players with the best player taken within 10 picks of him in the same draft."])


def normative(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    p = _player(view, slots, conv)
    rows = _nhl(view, tid)
    if not p:
        vets = [r for r in rows if (r.get("age") or 0) >= 30]
        p = max(vets, key=lambda r: _cap(r), default=None)
    if not p:
        return _answer("No veterans on the roster to weigh.")
    from .derived import team_status

    status = team_status(view, tid)
    age = p.get("age") or 0
    pros = sum(1 for r in view.team_rows(tid, ("AHL", "Prospect", "ECHL")) if _grp(r) == _grp(p) and (r.get("pot") or 0) >= (p.get("ovr") or 0) - 2)
    for_trade = [f"He's {age}; most players lose 1-2 overall a year from 31-32."] if age >= 31 else []
    if status in ("rebuilding", "middle"):
        for_trade.append(f"Team status is {status}: his value now is worth more than his wins later.")
    if pros:
        for_trade.append(f"{pros} prospect(s) at his position could take the minutes.")
    against = []
    if status == "contender":
        against.append("You're contending; losing a regular now costs real playoff odds.")
    if (p.get("pts") or 0) >= 0.6 * max(1, p.get("gp") or 1):
        against.append(f"He's still producing ({p.get('pts', 0)} pts in {p.get('gp', 0)} GP).")
    against.append("Popular veterans anchor the room; trading one can dent morale and how free agents see you.")
    lean = "Trade him" if len(for_trade) > len(against) else ("Keep him" if len(against) > len(for_trade) else "It's a coin flip — let the return decide")
    return _answer(f"{lean}: weighing {p['name']} ({age}, {p.get('ovr', 0):.0f}, ${_cap(p):.2f}M).",
                   analysis=["For trading: " + (" ".join(for_trade) or "nothing pressing."), "For keeping: " + " ".join(against),
                             "This is a judgement call; the deciding factor should be the return, not the principle."],
                   followups=[f"Should I trade {p['name']}?", "Should we rebuild, retool, or remain contenders?"])


def debate(view, q, slots, conv) -> Answer:
    tid = view.user_tid
    space = _cap_space(view, tid) or 0.0
    pipe = sum(1 for r in view.team_rows(tid, ("AHL", "Prospect", "ECHL")) if (r.get("pot") or 0) >= 82)
    return _answer("Both work; which is better depends on your window, cap and pipeline.",
                   analysis=["Draft: cheap, controllable, peaks 4-7 years out; most picks after the top 10 never become regulars. Builds sustained winners.",
                             "Free agency: immediate and proven, but you pay for past production, usually into the decline years, and term is the risk.",
                             "Most Cup winners are draft-built cores topped up with trades and a few targeted signings.",
                             f"For you: ${space:.2f}M of space and {pipe} prospect(s) with 82+ potential — " + ("the pipeline supports drafting and developing." if pipe >= 3 else "a thin pipeline makes targeted signings more useful.")])


def theory(view, q, slots, conv) -> Answer:
    if re.search(r"rebuild|tank", q):
        pts = ["Sell veterans at peak value for picks and young players, early rather than late.", "Pile up top-10 picks: that's where franchise players come from.",
               "Give young players real minutes; development needs ice time.", "Keep the cap clean so you can add when the core is 22-25.", "Know when to stop: 3-4 seasons is normal, 6+ usually means something's wrong."]
        head = "What makes a successful rebuild:"
    elif re.search(r"goalie|goaltend", q):
        head, pts = "What makes an elite goalie:", ["Save % above .915 over a full season, especially on high-danger shots.", "Consistency — few bad nights.", "Positioning and rebound control over highlight saves."]
    else:
        pts = ["A top-pair defenceman and a game-breaking centre.", "A goalie at or above league-average save %.", "Depth: 3 lines that can score and a third pair that doesn't bleed.",
               "Cap efficiency: value contracts and young players on cheap deals.", "Special teams in the top half."]
        head = "What makes a contender:"
    return _answer(head + " " + pts[0], analysis=pts[1:] + ["General hockey principles, not from your save — ask 'how do we measure up' to compare."],
                   followups=["Give me an overview of my franchise", "Should we rebuild, retool, or remain contenders?"])


def self_reflect(view, q, slots, conv) -> Answer:
    return _answer("What I can and can't see in this save:",
                   analysis=["Can: ratings and all sub-ratings, potential, contracts, this season's per-game results and player stats, special teams totals, injuries, trades, draft records, archived seasons and the league storyline log.",
                             "Can't: shot locations and per-shift data (so no true zone starts or quality of competition), per-game special-teams splits, coaching decisions, and trade offers you declined.",
                             "Limited: older seasons keep totals, not game-by-game detail; expected goals are the sim's own model.",
                             "To judge a roster better I'd want more games (40+ makes rates reliable) and line combinations over time."],
                   followups=["How confident are you we make the playoffs?", "Tell me something interesting about my franchise"])


def optimize(view, q, slots, conv) -> Answer:
    """Strongest lineup reachable with your cap: keep or swap each slot for the best value free agent."""
    tid = view.user_tid
    space = _cap_space(view, tid) or 0.0
    rows = _nhl(view, tid)
    fas = [view.row(p) for p, (t, lv) in view.where.items() if lv == "FA"]
    fas = [r for r in fas if r and (r.get("ovr") or 0) >= 70]
    need = {"C": 4, "W": 8, "D": 6, "G": 2}
    swaps = []
    budget = space
    for g, n in need.items():
        mine = sorted((r for r in rows if _grp(r) == g), key=lambda r: r.get("ovr") or 0)
        pool = sorted((r for r in fas if _grp(r) == g), key=lambda r: -((r.get("ovr") or 0)))
        for weak in mine[:max(0, len(mine) - (n - 2))]:
            best = next((f for f in pool if (f.get("ovr") or 0) >= (weak.get("ovr") or 0) + 2 and _cap(f) - _cap(weak) <= budget), None)
            if best:
                budget -= max(0.0, _cap(best) - _cap(weak))
                pool.remove(best)
                swaps.append((weak, best))
    if not swaps:
        return _answer(f"Your current lineup is already the strongest you can afford with ${space:.2f}M of space — no free agent is a 2+ overall upgrade that fits.",
                       followups=["How should I allocate my cap space?", "Best trade targets"])
    from .answers3 import _strength_with

    base = _strength_with(view, tid)
    lift = sum((b.get("ovr") or 0) - (w.get("ovr") or 0) for w, b in swaps)
    return _answer(f"Strongest affordable lineup: {len(swaps)} swap(s), +{lift:.0f} overall across the lineup, ${space - budget:.2f}M of your ${space:.2f}M used.",
                   analysis=["Greedy pass: replaces your weakest players at each position with the best free agent 2+ overall better that still fits the remaining space.",
                             f"Current team strength {base:.2f}. Trades aren't included — they need a partner's agreement."],
                   table={"cols": ["Out", "OVR", "In (FA)", "OVR", "Extra cap"],
                          "rows": [[w["name"], f"{w.get('ovr', 0):.0f}", b["name"], f"{b.get('ovr', 0):.0f}", f"${max(0.0, _cap(b) - _cap(w)):.2f}M"] for w, b in swaps]})


def trend(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    from .answers6 import _select, _summary

    games = _select(view, tid, {}, None)
    if len(games) < 10:
        return _answer("Need at least 10 games to see a trend.")
    half = len(games) // 2
    a, b = _summary(games[:half]), _summary(games[half:])
    area = "ga" if re.search(r"defen[cs]|goals against|keeping|goalie", q) else ("gf" if re.search(r"scor|offen[cs]|goals for|attack", q) else "pts")
    def pg(s, k):
        return s[k] / s["gp"] if s["gp"] else 0.0
    if area == "pts":
        d = b["pts_pct"] - a["pts_pct"]
        head = f"{_team_label(view, tid)}: points % {a['pts_pct']:.3f} in the first {a['gp']} games → {b['pts_pct']:.3f} in the last {b['gp']} ({'up' if d > 0 else 'down'} {abs(d):.3f})."
    else:
        x, y = pg(a, area), pg(b, area)
        word = "goals for" if area == "gf" else "goals against"
        better = (y > x) if area == "gf" else (y < x)
        head = f"{_team_label(view, tid)}: {word} {x:.2f}/game early → {y:.2f}/game lately — {'better' if better else 'worse'} by {abs(y - x):.2f}."
    notes = [f"First half {_rec(a)} ({a['gf']} GF, {a['ga']} GA); second half {_rec(b)} ({b['gf']} GF, {b['ga']} GA)."]
    if re.search(r"\bwhy\b", q):
        injured = [r for r in _nhl(view, tid) if r.get("injured") and (r.get("ovr") or 0) >= 78]
        if injured:
            notes.append("Likely factor — injuries: " + ", ".join(f"{r['name']} ({r['ovr']:.0f})" for r in injured[:4]) + ".")
        g = max((r for r in _nhl(view, tid) if r.get("pos") == "G"), key=lambda r: r.get("gp") or 0, default=None)
        if g and g.get("sv_pct"):
            from .answers7 import _league_sv

            notes.append(f"Starter {g['name']} is at .{int(g['sv_pct'] * 1000):03d} vs league .{int(_league_sv(view) * 1000):03d}.")
        notes.append("Halves are compared on results; the save keeps season totals for special teams, so PP/PK can't be split.")
    return _answer(head, analysis=notes, followups=["What patterns appear in our losses?", "Are our results sustainable?"])


def history(view, q, slots, conv) -> Answer:
    tid = _tid(view, q, slots)
    try:
        from .data import archive

        doc = archive(view.session)
    except Exception:
        doc = {}
    seasons = sorted((doc.get("seasons") or {}).items(), key=lambda kv: int(kv[0]))
    rows = []
    for yr, sec in seasons:
        st = (sec.get("standings") or {}).get(str(tid)) or {}
        if not st:
            continue
        champ = "🏆" if str(sec.get("champion_id") or "") == str(tid) else ""
        rows.append([sec.get("label") or yr, f"{st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)}", st.get("pts", "—"), st.get("gf", "—"), st.get("ga", "—"), champ])
    st_now = _standing(view, tid) or {}
    if st_now.get("gp"):
        rows.append(["This season", f"{st_now.get('w', 0)}-{st_now.get('l', 0)}-{st_now.get('otl', 0)}", st_now.get("pts"), st_now.get("gf"), st_now.get("ga"), ""])
    if len(rows) < 2:
        return _answer("Only this season is on record so far — finish a season and the year-by-year view fills in.",
                       analysis=["Completed seasons are archived with standings, Cup winners, awards and draft results."])
    return _answer(f"{_team_label(view, tid)} season by season ({len(rows)} seasons):", table={"cols": ["Season", "Record", "Pts", "GF", "GA", "Cup"], "rows": rows[-10:]},
                   analysis=["Earlier seasons keep totals only, so trends are compared on records and goals."])


def anomaly(view, q, slots, conv) -> Answer:
    from .answers7 import _impact_rows

    tid = view.user_tid if re.search(r"\b(my|our|we)\b", q) else None
    rows = [r for t in ([tid] if tid else list(view.teams)) for r in _impact_rows(view, t)]
    if not rows:
        return _answer("Not enough games yet to find outliers.")
    rows.sort(key=lambda r: r["impact"])
    over, under = rows[::-1][:5], rows[:5]
    return _answer(f"Furthest above expectations: {over[0]['name']} ({over[0].get('team')}, {over[0]['impact']:+.1f} goals). Furthest below: {under[0]['name']} ({under[0].get('team')}, {under[0]['impact']:+.1f}).",
                   analysis=["Expectation = what a player's overall and position usually produce; goalies against an average goalie on the same shots."],
                   table={"cols": ["Player", "Team", "OVR", "Impact", "Why"], "rows": [[r["name"], r.get("team"), f"{r.get('ovr', 0):.0f}", f"{r['impact']:+.1f}", r["why"]] for r in over + under]})


def retro(view, q, slots, conv) -> Answer:
    from .answers7 import _trade_verdict, _bust_or_steal

    worst = bool(re.search(r"\b(worst|negative|bad|biggest mistake)\b", q))
    t = _trade_verdict(view, ("worst" if worst else "best") + " trade i made", {})
    d = _bust_or_steal(view, "bust" if worst else "steal", "my " + q)
    head = ("Your worst calls: " if worst else "Your best calls: ") + t["answer"]
    return _answer(head, analysis=["Draft side: " + d["answer"], "Signings aren't stored with their results yet, so trades and picks are what can be judged."],
                   table=t.get("table"), followups=["Biggest draft steal", "Most lopsided trade"])


def coach(view, q, slots, conv) -> Answer:
    return _answer("Coaching decisions aren't recorded in the save, so I can't judge a hire or firing directly.",
                   analysis=["What I can do: compare results before and after a date — e.g. 'our record since January' or 'how are we doing lately'."],
                   followups=["How are we doing lately?", "What patterns appear in our losses?"])


DIVISIONS = {
    "atlantic": {"BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"},
    "metropolitan": {"CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"},
    "central": {"CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG", "ARI"},
    "pacific": {"ANA", "CGY", "EDM", "LAK", "SJS", "SEA", "VAN", "VGK"},
}
DIVISIONS["east"] = DIVISIONS["atlantic"] | DIVISIONS["metropolitan"]
DIVISIONS["west"] = DIVISIONS["central"] | DIVISIONS["pacific"]


def _abbr(view, tid: str) -> str:
    return str((view.team_meta.get(tid) or {}).get("abbr") or tid)[:3].upper()


def _odds(pts: float, gp: int, line: float) -> float:
    import math

    if gp <= 0:
        return 0.5
    pace = pts / gp * 82
    sd = max(1.0, 8.0 * ((82 - gp) / 82) ** 0.5)
    return 0.5 * (1 + math.erf((pace - line) / sd / 2 ** 0.5))


def playoff_race(view, q, slots, conv) -> Answer:
    key = next((k for k in ("atlantic", "metropolitan", "metro", "central", "pacific", "eastern", "east", "western", "west") if re.search(rf"\b{k}\b", q)), None)
    key = {"metro": "metropolitan", "eastern": "east", "western": "west"}.get(key, key)
    if key is None:
        mine = _abbr(view, view.user_tid)
        key = next((k for k in ("atlantic", "metropolitan", "central", "pacific") if mine in DIVISIONS[k]), "east")
    members = DIVISIONS[key]
    rows = []
    for r in view.standings():
        tid = str(r.get("team_id") or r.get("id") or "")
        if _abbr(view, tid) not in members or not r.get("gp"):
            continue
        rows.append((tid, float(r.get("pts") or 0), int(r.get("gp") or 0), r))
    if not rows:
        return _answer(f"No games yet in the {key.title()}.")
    rows.sort(key=lambda x: (-x[1] / max(1, x[2]), -x[1]))
    # Top 3 in a division make it outright; the rest chase wild cards (~96-point line).
    out = []
    for i, (tid, pts, gp, r) in enumerate(rows):
        line = 92.0 if (key in DIVISIONS and len(members) <= 9 and i < 3) else 96.0
        out.append((tid, pts, gp, _odds(pts, gp, line), r))
    out.sort(key=lambda x: -x[3])
    likely = [_team_label(view, t) for t, _, _, o, _ in out if o >= 0.5]
    head = f"{key.title()} playoff picture: most likely in — " + (", ".join(likely[:5]) if likely else "nobody is clearly on pace yet") + "."
    table = {"cols": ["Team", "Record", "Pts", "GP", "Pace", "Playoff odds"],
             "rows": [[_team_label(view, t), f"{r.get('w', 0)}-{r.get('l', 0)}-{r.get('otl', 0)}", int(p), g, f"{p / max(1, g) * 82:.0f}", f"{o * 100:.0f}%"] for t, p, g, o, r in out]}
    return _answer(head, analysis=["Odds come from each team's points pace against the usual cut line (top 3 in a division ~92 points, wild card ~96), with uncertainty shrinking as games are played.",
                                   "Schedule strength and injuries aren't in the model."], table=table,
                   followups=["How confident are you we make the playoffs?", "Which team allows the fewest goals?"])


_AWARDS = {
    "art ross": ("Art Ross (points)", lambda r: r.get("pts") or 0, lambda r: r.get("pos") != "G", "pts"),
    "rocket": ("Rocket Richard (goals)", lambda r: r.get("g") or 0, lambda r: r.get("pos") != "G", "g"),
    "richard": ("Rocket Richard (goals)", lambda r: r.get("g") or 0, lambda r: r.get("pos") != "G", "g"),
    "norris": ("Norris (top defenseman)", lambda r: (r.get("pts") or 0) * 0.7 + (r.get("plus_minus") or 0) * 0.5 + (r.get("ovr") or 0) * 0.4, lambda r: (r.get("pos") or "") in ("D", "LD", "RD"), "pts"),
    "calder": ("Calder (rookie)", lambda r: r.get("pts") or 0, lambda r: r.get("pos") != "G" and (r.get("age") or 99) <= 22, "pts"),
    "vezina": ("Vezina (goalie)", lambda r: (r.get("sv_pct") or 0) * 1000 + (r.get("w") or 0) * 0.3, lambda r: r.get("pos") == "G" and (r.get("gp") or 0) >= 10, "sv"),
    "hart": ("Hart (MVP)", lambda r: (r.get("pts") or 0) + (r.get("gwg") or 0) * 2 + (r.get("ovr") or 0) * 0.3, lambda r: r.get("pos") != "G", "pts"),
    "selke": ("Selke (defensive forward)", lambda r: (r.get("plus_minus") or 0) + (r.get("fo_pct") or 0) * 20 + (r.get("ovr") or 0) * 0.2, lambda r: (r.get("pos") or "") in ("C", "LW", "RW", "W", "F"), "pm"),
}


def award_race(view, q, slots, conv) -> Answer:
    key = next((k for k in _AWARDS if k in q), "hart")
    label, score, eligible, kind = _AWARDS[key]
    rows = [r for t in view.teams for r in _nhl(view, t) if eligible(r) and (r.get("gp") or 0) >= 5]
    if not rows:
        return _answer(f"No one qualifies for the {label} race yet.")
    rows.sort(key=score, reverse=True)
    top = rows[0]
    def line(r):
        if kind == "sv":
            return f".{int((r.get('sv_pct') or 0) * 1000):03d} SV%, {r.get('w', 0)} W"
        team_gp = int((_standing(view, r.get("team_id")) or {}).get("gp") or r.get("gp") or 1)
        val = r.get("g") if kind == "g" else r.get("pts")
        return f"{val} in {r.get('gp')} GP (pace {(val or 0) / max(1, team_gp) * 82:.0f})"
    lead = score(top) - score(rows[1]) if len(rows) > 1 else 0
    return _answer(f"{label} favourite right now: {top['name']} ({top.get('team')}) — {line(top)}.",
                   analysis=[f"Lead over 2nd: {lead:.1f} on this measure." if len(rows) > 1 else "",
                             "Awards are only announced at season's end; this is the current race, not a result. Voting awards (Hart, Norris, Selke) are estimated from production and rating."],
                   table={"cols": ["#", "Player", "Team", "Line"], "rows": [[i + 1, r["name"], r.get("team"), line(r)] for i, r in enumerate(rows[:10])]},
                   followups=["Who's winning the Rocket Richard?", "Calder race", "Vezina race"])


HANDLERS: Dict[str, Callable[..., Answer]] = {
    "overview": overview, "strengths": strengths, "categorize": categorize, "sustain": sustain, "interpret": interpret,
    "correlate": correlate, "pattern": pattern, "predict": predict, "confidence": confidence, "speculate": speculate,
    "exciting": exciting, "premise": premise, "adversarial": adversarial, "exploratory": exploratory, "creative": creative,
    "prioritize": prioritize, "allocate": allocate, "multi_criteria": multi_criteria, "multi_step": multi_step,
    "contingency": contingency, "hypothetical": hypothetical, "scenario": scenario, "counterfactual": counterfactual,
    "normative": normative, "debate": debate, "theory": theory, "self_reflect": self_reflect,
    "optimize": optimize, "award_race": award_race, "playoff_race": playoff_race, "trend": trend, "history": history, "anomaly": anomaly, "retro": retro, "coach": coach,
}


def run(mode: str, view: LiveView, q: str, slots: Dict[str, Any], conv: Dict[str, Any]) -> Answer:
    q = " ".join(str(q or "").lower().replace("’", "'").split())
    return HANDLERS[mode](view, q, slots, conv)
