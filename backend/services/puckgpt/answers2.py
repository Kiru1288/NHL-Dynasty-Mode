"""PuckGPT, second layer: a general ranking engine plus the questions that need
memory, reasoning or several facts at once.

``route`` reads the normalised question and the conversation and decides when a
question belongs here instead of the first-layer intents. It runs before the
intent model's answer is used, so specific phrasings win over the model's guess.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from .answers import (
    HANDLERS as BASE, POS_NAMES, SINGULAR, _answer, _card, _cap_space, _money, _player_table, _pos_ok,
    _resolve_players, _standing, _team_label,
)
from .data import LiveView, archive
from .metrics import METRICS, find_metric, fmt, metric_value

Answer = Dict[str, Any]
RANK_WORDS = {"best", "worst", "top", "most", "least", "highest", "lowest", "fewest", "bottom", "who", "which", "rank",
              "ranking", "leader", "leaders", "leads", "biggest", "strongest", "weakest", "fastest", "slowest", "smartest",
              "toughest", "oldest", "youngest", "cheapest", "underrated", "better"}
LOW_WORDS = {"worst", "least", "lowest", "fewest", "bottom", "weakest", "slowest", "unhappy", "unhappiest", "hates", "wants",
             "youngest", "cheapest"}


# =========================================================================== scope helpers
def _division_tids(view: LiveView, tid: str, key: str = "division") -> List[str]:
    t = view.teams.get(tid)
    want = str(getattr(t, key, "") or "")
    return [x for x, tm in view.teams.items() if str(getattr(tm, key, "") or "") == want] if want else []


def _sellers(view: LiveView, n: int = 12) -> set:
    st = sorted(view.standings(), key=lambda r: r.get("pts", 0))
    return {str(r.get("team_id")) for r in st[:n]}


def _pool(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any], raw: str) -> Tuple[List[str], str]:
    """Player ids in scope, and a label for the answer."""
    w = set(raw.split())
    levels = ("NHL",)
    if "prospect" in w or "prospects" in w or "junior" in w:
        levels = ("Prospect", "AHL", "ECHL")
    if slots.get("self") or "my" in w or "our" in w or "we" in w or "us" in w:
        tid = view.user_tid
        if "prospect" in w or "prospects" in w:
            return [p for p, (t, lv) in view.where.items() if t == tid and lv in ("Prospect", "AHL", "ECHL")], "in your system"
        return [p for p, (t, lv) in view.where.items() if t == tid and lv == "NHL"], f"on the {_team_label(view, tid)}"
    if "their" in w and ctx.get("team"):
        tid = ctx["team"]
        return [p for p, (t, lv) in view.where.items() if t == tid and lv == "NHL"], f"on the {_team_label(view, tid)}"
    if slots.get("teams"):
        tids = {str(x) for x in slots["teams"]}
        return [p for p, (t, lv) in view.where.items() if t in tids and lv in levels], f"on the {_team_label(view, next(iter(tids)))}"
    if "division" in w or "conference" in w:
        key = "division" if "division" in w else "conference"
        tids = set(_division_tids(view, view.user_tid, key))
        name = str(getattr(view.teams.get(view.user_tid), key, "") or "")
        return [p for p, (t, lv) in view.where.items() if t in tids and lv == "NHL"], f"in the {name} {key.title()}"
    if "available" in w or "afford" in w or "free" in w:
        sellers = _sellers(view)
        ids = [p for p, (t, lv) in view.where.items() if lv == "FA" or (lv == "NHL" and t in sellers and t != view.user_tid)]
        return ids, "available (free agents and likely sellers)"
    return [p for p, (t, lv) in view.where.items() if lv in levels], ("among prospects" if levels != ("NHL",) else "in the league")


def _ranked(view: LiveView, ids: List[str], metric: str, low: bool, slots: Dict[str, Any]) -> List[Tuple[float, Dict[str, Any]]]:
    out = []
    pos = slots.get("position")
    for pid in ids:
        r = view.row(pid)
        if not r or not _pos_ok(r, pos):
            continue
        if slots.get("age_max") and not (0 < r["age"] <= slots["age_max"]):
            continue
        if slots.get("age_min") and r["age"] < slots["age_min"]:
            continue
        if slots.get("cap_max") and (r["contract"]["cap_hit_m"] or 0) > slots["cap_max"]:
            continue
        v = metric_value(view, pid, metric)
        if v is None:
            continue
        out.append((v, r))
    higher = METRICS.get(metric, {}).get("higher", True)
    out.sort(key=lambda vr: vr[0], reverse=(higher != low))
    return out


def _metric_table(rows: List[Tuple[float, Dict[str, Any]]], metric: str) -> Dict[str, Any]:
    t = _player_table([r for _v, r in rows], ("gp", "pts"))
    t["cols"].append(METRICS[metric]["label"])
    for line, (v, _r) in zip(t["rows"], rows):
        line.append(fmt(metric, v))
    return t


# =========================================================================== generic ranking
def rank_metric(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    metric = slots["metric"]
    m = METRICS[metric]
    low = bool(slots.get("low"))
    ids, where = _pool(view, slots, ctx.get("conv", {}), raw)
    if raw.startswith("need") or " need " in f" {raw} ":
        sellers = _sellers(view)
        ids = [p for p, (t, lv) in view.where.items() if (lv == "FA" or (lv == "NHL" and t in sellers)) and t != view.user_tid]
        where = "you could get (free agents and likely sellers)"
    if "nobody" in raw or "underrated" in raw:
        ids = [p for p in ids if view.row(p).get("ovr", 99) <= 87]
        where += ", outside the stars"
    rows = _ranked(view, ids, metric, low, slots)
    if not rows:
        return _answer(f"No one {where} has a {m['label'].lower()} on record yet.",
                       analysis=["Stats fill in as games are played; ratings are always there for rostered players."])
    shown = rows[: int(slots.get("limit") or 10)]
    v0, top = shown[0]
    word = ("Lowest" if m["higher"] else "Highest") if low else ("Highest" if m["higher"] else "Lowest")
    if metric in ("value",):
        word = "Worst value" if low else "Best value for the money"
    elif metric == "morale":
        word = "Unhappiest" if low else "Happiest"
    label = POS_NAMES.get(slots.get("position") or "", "player")
    head = (f"{word} {m['label'].lower()} {where}: {top['name']} ({fmt(metric, v0)})."
            if metric not in ("value", "morale") else f"{word} {SINGULAR.get(label, label)} {where}: {top['name']} ({fmt(metric, v0)}).")
    values = [v for v, _ in rows]
    analysis = [f"Ranked {len(rows)} players {where} by {m['label'].lower()}."]
    if len(values) >= 5:
        analysis.append(f"Median {fmt(metric, statistics.median(values))}; {top['name']} is {top['pos']}, {top['age']}, {top['ovr']:.0f} overall.")
    return _answer(head, analysis=analysis, table=_metric_table(shown, metric), cards=[_card(r) for _v, r in shown[:6]],
                   list_ids=[r["id"] for _v, r in shown], followups=[f"Compare {shown[0][1]['name']} and {shown[1][1]['name']}" if len(shown) > 1 else "Best trade targets"])


def player_metric(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    metric = slots["metric"]
    m = METRICS[metric]
    if not rows:
        return _answer(f"Whose {m['label'].lower()}? Name the player.")
    r = rows[0]
    v = metric_value(view, r["id"], metric)
    if v is None:
        return _answer(f"{r['name']} has no {m['label'].lower()} on record.", focus_ids=[r["id"]])
    league = _ranked(view, [p for p, (t, lv) in view.where.items() if lv == "NHL"], metric, False, {"position": None})
    rank = next((i + 1 for i, (_v, x) in enumerate(league) if x["id"] == r["id"]), None)
    same = [vr for vr in league if _pos_ok(vr[1], "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else "F"))]
    prank = next((i + 1 for i, (_v, x) in enumerate(same) if x["id"] == r["id"]), None)
    analysis = []
    if rank:
        analysis.append(f"{rank} of {len(league)} in the league" + (f"; {prank} of {len(same)} at his position." if prank else "."))
    return _answer(f"{r['name']}: {m['label'].lower()} {fmt(metric, v)}.", analysis=analysis, rows=[r], extra=("gp", "pts"), focus_ids=[r["id"]])


# =========================================================================== conversation follow-ups
def _ctx_players(view: LiveView, slots: Dict[str, Any], conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = _resolve_players(view, slots)
    if rows:
        return rows
    return [view.row(pid) for pid in (conv.get("players") or []) if pid in view.players]


def other_player(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    conv = ctx.get("conv", {})
    cur = (conv.get("players") or [None])[0]
    pool = list(conv.get("compare") or []) + list(conv.get("list") or []) + list(conv.get("history") or [])[::-1]
    other = next((p for p in pool if p != cur and p in view.players), None)
    if not other:
        return _answer("Which other player? I only have one in this conversation.")
    return BASE["player_card"](view, {"players": [[other]]}, ctx)


def previous_player(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    hist = [p for p in (ctx.get("conv", {}).get("history") or []) if p in view.players]
    if not hist:
        return _answer("You haven't asked about a player yet in this conversation.")
    cur = (ctx.get("conv", {}).get("players") or [None])[0]
    earlier = [p for p in hist if p != cur]
    pick = earlier[-1] if earlier else hist[-1]
    out = BASE["player_card"](view, {"players": [[pick]]}, ctx)
    names = ", ".join(dict.fromkeys(view.row(p)["name"] for p in hist[::-1]))
    out.setdefault("analysis", []).append(f"Players you've asked about, newest first: {names}.")
    return out


def player_age(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Who? Name the player.")
    r = rows[0]
    p = view.players[r["id"]]
    ident = getattr(p, "identity", None)
    born = f"{getattr(ident, 'birth_year', '')}-{int(getattr(ident, 'birth_month', 0) or 0):02d}-{int(getattr(ident, 'birth_day', 0) or 0):02d}"
    return _answer(f"{r['name']} is {r['age']}.", analysis=[f"Born {born}."], rows=[r], extra=("gp", "pts"), focus_ids=[r["id"]])


def player_awards(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return BASE["awards"](view, slots, ctx)
    r = rows[0]
    try:
        from services.award_history import player_awards_history

        res = player_awards_history(view.session, r["id"])
    except Exception:
        res = {}
    save = [a for a in res.get("save_awards") or res.get("sim_awards") or [] if int(a.get("finish") or 1) == 1]
    nhl = res.get("nhl_awards") or []
    if not save and not nhl:
        return _answer(f"{r['name']} hasn't won a trophy.", rows=[r], extra=("gp", "pts"), focus_ids=[r["id"]])
    lines = [f"{a.get('season')}: {a.get('award_name') or a.get('award')}" for a in save]
    lines += [f"{a.get('season') or a.get('seasonId') or ''} {a.get('trophy') or a.get('name') or a.get('award_name') or ''} (real NHL)".strip() for a in nhl[:10]]
    return _answer(f"{r['name']} has won {len(save) + len(nhl)} award{'s' if len(save) + len(nhl) != 1 else ''}.", analysis=lines,
                   rows=[r], extra=("gp", "pts"), focus_ids=[r["id"]])


def _morale_reasons(view: LiveView, pid: str) -> List[str]:
    p = view.players.get(pid)
    ps = getattr(p, "psych", None)
    r = view.row(pid)
    out = []
    def low(attr: str, text: str, cut: float = 0.42) -> None:
        v = getattr(ps, attr, None)
        if v is not None and float(v) < cut:
            out.append(f"{text} ({float(v) * 100:.0f}/100)")
    if ps is not None:
        low("role_satisfaction", "Unhappy with his role")
        low("ice_time_satisfaction", "Wants more ice time")
        low("coach_relationship", "Poor relationship with the coach")
        low("trust_in_management", "Doesn't trust management")
        low("locker_room_fit", "Doesn't fit the room")
        low("career_satisfaction", "Frustrated with his career")
    st = _standing(view, r.get("team_id"))
    if st and int(st.get("league_rank") or st.get("rank") or 0) >= 24:
        out.append(f"Team is losing ({st.get('w')}-{st.get('l')}-{st.get('otl')})")
    if r["contract"]["years_left"] <= 1 and r["contract"]["cap_hit_m"]:
        out.append("Contract year")
    demands = getattr(view.session, "trade_demands", None) or {}
    if isinstance(demands, dict) and pid in demands:
        out.append("Has asked for a trade")
    return out


def player_morale(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return team_morale(view, slots, ctx)
    r = rows[0]
    mv = metric_value(view, r["id"], "morale")
    reasons = _morale_reasons(view, r["id"])
    mood = "unhappy" if (mv or 50) < 45 else ("content" if (mv or 50) < 65 else "happy")
    return _answer(f"{r['name']} is {mood} (morale {fmt('morale', mv)}).",
                   analysis=reasons or ["No single complaint stands out; his morale is close to normal."],
                   rows=[r], extra=("gp", "pts", "cap"), focus_ids=[r["id"]])


def team_morale(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    ids = [p for p, (t, lv) in view.where.items() if t == view.user_tid and lv == "NHL"]
    vals = [(metric_value(view, p, "morale"), view.row(p)) for p in ids]
    vals = [(v, r) for v, r in vals if v is not None]
    if not vals:
        return _answer("No morale readings yet.")
    vals.sort(key=lambda vr: vr[0])
    avg = statistics.mean(v for v, _ in vals)
    lows = vals[:5]
    analysis = [f"Team average morale {avg:.0f}/100."]
    for v, r in lows[:3]:
        why = _morale_reasons(view, r["id"])
        analysis.append(f"{r['name']} {v:.0f}: {why[0] if why else 'no specific complaint'}.")
    st = _standing(view, view.user_tid)
    if st and int(st.get("league_rank") or 0) >= 20:
        analysis.append("Losing drags everyone's mood; results are the biggest lever.")
    t = _metric_table(lows, "morale")
    return _answer(f"Lowest morale: {lows[0][1]['name']} ({lows[0][0]:.0f}).", analysis=analysis, table=t,
                   list_ids=[r["id"] for _v, r in lows])


def afford(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    space = _cap_space(view, view.user_tid)
    pointed = slots.get("players") or re.search(r"\b(him|he|his)\b", raw)
    rows = _ctx_players(view, slots, ctx.get("conv", {})) if pointed else []
    if not rows:
        out = rank_metric(view, {**slots, "metric": "ovr", "cap_max": max(space or 0.0, 0.0) or 0.9},
                          {**ctx, "raw": raw + " available"})
        out["answer"] = f"Best players you can afford with {_money(space or 0)} in space: " + out["answer"].split(": ", 1)[-1]
        return out
    r = rows[0]
    cap = r["contract"]["cap_hit_m"] or 0.0
    if space is None:
        return _answer("I can't read your cap space right now.")
    if cap <= space:
        return _answer(f"Yes. {r['name']} costs {_money(cap)} and you have {_money(space)} in space.",
                       analysis=[f"You'd have {_money(space - cap)} left."], rows=[r], extra=("cap", "status"), focus_ids=[r["id"]])
    short = cap - space
    mine = sorted((x for x in view.team_rows(view.user_tid) if (x["contract"]["cap_hit_m"] or 0) >= short * 0.8),
                  key=lambda x: (x["ovr"], -(x["contract"]["cap_hit_m"] or 0)))[:3]
    return _answer(f"Not without sending money back. {r['name']} costs {_money(cap)}; you have {_money(space)}.",
                   analysis=[f"You're {_money(short)} short."] + ([f"Salary that could go the other way: {', '.join(f'{m['name']} {_money(m['contract']['cap_hit_m'])}' for m in mine)}."] if mine else []),
                   rows=[r], extra=("cap", "status"), focus_ids=[r["id"]])


def available(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return BASE["find_players"](view, {**slots, "filters": ["free"]}, ctx)
    r = rows[0]
    if r["level"] == "FA":
        return _answer(f"Yes, {r['name']} is an unsigned free agent.", rows=[r], extra=("status",), focus_ids=[r["id"]])
    if r["team_id"] == view.user_tid:
        return _answer(f"{r['name']} is already yours.", rows=[r], extra=("cap",), focus_ids=[r["id"]])
    sellers = _sellers(view)
    c = r["contract"]
    notes = [f"Under contract with the {_team_label(view, r['team_id'])}: {_money(c['cap_hit_m'])} x {c['years_left']}y."]
    if c["nmc"]:
        notes.append("He has a no-move clause, so he'd have to agree.")
    verdict = "Probably" if r["team_id"] in sellers else "Only by trade, and they aren't sellers"
    notes.append(f"The {_team_label(view, r['team_id'])} are {'among the likely sellers' if r['team_id'] in sellers else 'not a seller right now'}.")
    return _answer(f"{verdict}: {r['name']} would have to come by trade.", analysis=notes, rows=[r], extra=("cap", "status"), focus_ids=[r["id"]])


def drafted_by_us(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Who? Name the player.")
    r = rows[0]
    try:
        from services.award_history import _draft_payload_from_player
        from services.real_nhl_roster_importer import ensure_player_nhl_landing_data, ensure_prospect_draft_from_nhl

        p = view.players[r["id"]]
        if getattr(p, "real_nhl_import", False):
            ensure_player_nhl_landing_data(p, fetch=True, league=view.league)
        else:
            ensure_prospect_draft_from_nhl(p, league=view.league)
        d = _draft_payload_from_player(p)
    except Exception:
        d = {}
    if not d.get("drafted"):
        return _answer(f"{r['name']} was never drafted." if d.get("undrafted") else f"I don't have {r['name']}'s draft record.", focus_ids=[r["id"]])
    team = str(d.get("drafted_by_team_id") or d.get("draft_team_id") or "")
    who = _team_label(view, team) if team else (d.get("draft_team_abbr") or "another team")
    ours = team == view.user_tid
    pick = f"{d.get('draft_year')}, round {d.get('draft_round')}, #{d.get('draft_overall_pick')} overall"
    return _answer(f"{'Yes' if ours else 'No'}: {r['name']} was drafted by the {who} ({pick}).", focus_ids=[r["id"]])


def brother(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Whose brother? Name the player.")
    r = rows[0]
    last = r["name"].split()[-1].lower()
    sibs = [view.row(p) for p in view.players if p != r["id"] and view.row(p)["name"].split()[-1].lower() == last]
    if not sibs:
        return _answer(f"No other {r['name'].split()[-1]} is in this save.", focus_ids=[r["id"]])
    s = sibs[0]
    return _answer(f"{s['name']} ({s['pos']}, {s['team'] or s['level']}, {s['ovr']:.0f} OVR).",
                   analysis=["Same family name; the save doesn't record family ties, so this is the namesake."],
                   rows=sibs[:4], extra=("gp", "pts", "cap"), focus_ids=[s["id"]])


def move_position(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Move who? Name the player.")
    r = rows[0]
    raw = ctx.get("raw", "")
    to = "RD" if re.search(r"\b(right defenseman|right d|rd)\b", raw) else "LD" if re.search(r"\b(left defenseman|left d|ld)\b", raw) else \
        "C" if "center" in raw else "W"
    notes = []
    if to in ("RD", "LD"):
        hand = r["shoots"] or "?"
        natural = (to == "RD" and hand == "R") or (to == "LD" and hand == "L")
        notes.append(f"He shoots {hand}. {'That is his natural side.' if natural else 'Playing his off side costs a little on breakouts and one-timers, but plenty of D do it.'}")
        if r["pos"] not in ("D", "LD", "RD"):
            notes.append("He isn't a defenseman, which is a much bigger change than switching sides.")
    elif to == "C":
        fo = metric_value(view, r["id"], "fo_pct")
        notes.append(f"Faceoffs this season: {fmt('fo_pct', fo)}." if fo is not None else "No faceoff record yet.")
    t = metric_value(view, r["id"], "ch:transition")
    d = metric_value(view, r["id"], "ch:defence")
    if t is not None and d is not None:
        notes.append(f"Transition {t:.0f}, defence {d:.0f}.")
    return _answer(f"{r['name']} at {to}: workable" if notes else f"{r['name']} at {to}.", analysis=notes, rows=[r], extra=("gp", "pts"), focus_ids=[r["id"]])


def pair_with(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    conv = ctx.get("conv", {})
    if len(rows) < 2 and conv.get("players"):
        rows = [view.row(conv["players"][0])] + rows
    if len(rows) < 2:
        return _answer("Name both players, e.g. “would Rielly work with Tanev”.")
    a, b = rows[0], rows[1]
    notes = []
    if a["pos"] in ("D", "LD", "RD") and b["pos"] in ("D", "LD", "RD"):
        notes.append("Opposite hands: a natural left-right pair." if a["shoots"] != b["shoots"] else "Same shooting hand: one plays his off side.")
    ta, tb = metric_value(view, a["id"], "ch:transition") or 0, metric_value(view, b["id"], "ch:transition") or 0
    da, db = metric_value(view, a["id"], "ch:defence") or 0, metric_value(view, b["id"], "ch:defence") or 0
    if (ta - da) * (tb - db) < 0:
        notes.append("Their styles complement: one leans on puck-moving, the other on defending.")
    elif ta > da and tb > db:
        notes.append("Both lean offensive; the pair may leak chances against.")
    else:
        notes.append("Both lean defensive; the pair may struggle to move the puck.")
    try:
        from app.sim_engine.chemistry import pair_chemistry  # optional engine hook

        notes.append(f"Engine chemistry {pair_chemistry(view.players[a['id']], view.players[b['id']]):.0f}.")
    except Exception:
        pass
    return _answer(f"{a['name']} with {b['name']}: {'good fit' if (ta - da) * (tb - db) < 0 or a['shoots'] != b['shoots'] else 'it can work'}.",
                   analysis=notes, rows=[a, b], extra=("gp", "pts", "plus_minus"), focus_ids=[a["id"], b["id"]])


def compare_current(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Compare who? Name the player.")
    r = rows[0]
    grp = "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else r["pos"])
    mine = sorted((x for x in view.team_rows(view.user_tid) if _pos_ok(x, grp) and x["id"] != r["id"]), key=lambda x: -x["ovr"])
    if not mine:
        return _answer(f"You have nobody at {grp} to compare with.")
    return BASE["compare"](view, {"players": [[r["id"]], [mine[0]["id"]]], "angle": slots.get("angle")}, ctx)


def fit_check(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return BASE["fit_partner"](view, slots, ctx)
    r = rows[0]
    grp = "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else "F")
    mine = sorted((x for x in view.team_rows(view.user_tid) if _pos_ok(x, grp) and x["id"] != r["id"]), key=lambda x: -x["ovr"])
    slot = sum(1 for x in mine if x["ovr"] > r["ovr"]) + 1
    tier = {"D": ["top pair", "top pair", "second pair", "second pair", "third pair", "third pair"],
            "F": ["first line"] * 3 + ["second line"] * 3 + ["third line"] * 3 + ["fourth line"] * 3,
            "G": ["starter", "backup", "third string"]}[grp]
    where = tier[slot - 1] if slot - 1 < len(tier) else "press box"
    space = _cap_space(view, view.user_tid)
    cap = r["contract"]["cap_hit_m"] or 0
    notes = [f"He'd rank #{slot} among your {POS_NAMES.get(grp, grp)}: your {where}."]
    if space is not None and r["team_id"] != view.user_tid:
        notes.append("He fits under your cap." if cap <= space else f"Cap: you'd need to clear {_money(cap - space)}.")
    return _answer(f"{r['name']} would slot onto your {where}.", analysis=notes, rows=[r] + mine[:4], extra=("cap", "pts"), focus_ids=[r["id"]])


def worth_it(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Is who worth it? Name the player.")
    return BASE["player_contract"](view, {"players": [[rows[0]["id"]]]}, ctx)


def repeat_last(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    conv = ctx.get("conv", {})
    last = conv.get("intent")
    if last in ("trade_advice", "fit_check", "afford", "available", "worth_it") and conv.get("players"):
        fn = BASE.get(last) or HANDLERS2.get(last)
        out = fn(view, {"players": [[conv["players"][0]]]}, ctx)
        out.setdefault("analysis", []).insert(0, "Picking up your last question.")
        return out
    return _answer("Do what? Tell me the move and I'll weigh it, e.g. “should I trade Marner”.")


# =========================================================================== history
def worst_trade(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    archive(view.session)
    try:
        from services.legacy_hall import trade_history

        trades = trade_history(view.session, team_id=view.user_tid, limit=100).get("trades") or []
    except Exception:
        trades = []
    if not trades:
        return _answer("You haven't made a trade in this save yet.")
    best = "best" in ctx.get("raw", "")
    def net(t: Dict[str, Any]) -> float:
        got = sum(int((p.get("after_with_new_team") or {}).get("pts") or 0) for p in t.get("players") or [] if str(p.get("to_team_id")) == view.user_tid)
        gave = sum(int((p.get("after_with_new_team") or {}).get("pts") or 0) for p in t.get("players") or [] if str(p.get("to_team_id")) != view.user_tid)
        return got - gave
    trades.sort(key=net, reverse=best)
    t = trades[0]
    moved = "; ".join(f"{p.get('name')} → {view.team_meta.get(str(p.get('to_team_id')), {}).get('abbr', '?')}" for p in t.get("players") or [])
    return _answer(f"{'Best' if best else 'Worst'} trade so far ({t.get('season')}): {moved}.",
                   analysis=[f"Net points since, for you minus for them: {net(t):+d}.", "Judged on points scored after the trade by the players each side got."])


def biggest_upset(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    doc = archive(view.session)
    best = None
    for y, s in (doc.get("seasons") or {}).items():
        stand = s.get("standings") or {}
        for ser in s.get("series") or []:
            hi, lo, win = str(ser.get("team_high_id") or ""), str(ser.get("team_low_id") or ""), str(ser.get("winner_id") or ser.get("winner") or "")
            if not hi or not lo or win != lo:
                continue
            gap = int((stand.get(hi) or {}).get("pts") or 0) - int((stand.get(lo) or {}).get("pts") or 0)
            if best is None or gap > best[0]:
                best = (gap, y, lo, hi)
    if not best:
        return _answer("No playoff upsets on record yet; history starts after the first playoffs.")
    gap, y, lo, hi = best
    return _answer(f"Biggest upset: {y}, the {_team_label(view, lo)} beat the {_team_label(view, hi)}, who had {gap} more points.")


def season_record(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    stat = slots.get("stat") or "pts"
    stat = stat if stat in ("g", "a", "pts", "pim", "hit", "blk", "w", "so") else "pts"
    doc = archive(view.session)
    best = []
    for y, s in (doc.get("seasons") or {}).items():
        for ln in s.get("player_lines") or []:
            if (ln.get("scope") or "regular") == "regular" and ln.get(stat) is not None:
                if slots.get("position") and not _pos_ok({"pos": str(ln.get("pos") or ""), "shoots": ""}, slots["position"]):
                    continue
                best.append((int(ln.get(stat) or 0), ln.get("name"), y))
    for pid, (t, lv) in view.where.items():
        if lv == "NHL":
            r = view.row(pid)
            if slots.get("position") and not _pos_ok(r, slots["position"]):
                continue
            best.append((int(r.get(stat) or 0), r["name"], f"{view.season} (in progress)"))
    best.sort(reverse=True)
    if not best or best[0][0] == 0:
        return _answer("No season records yet.")
    name = {"g": "goals", "a": "assists", "pts": "points", "pim": "PIM", "hit": "hits", "blk": "blocks", "w": "wins", "so": "shutouts"}[stat]
    who = " by a defenseman" if slots.get("position") == "D" else ""
    return _answer(f"Most {name} in a season{who}: {best[0][1]}, {best[0][0]} ({best[0][2]}).",
                   table={"cols": ["Player", name.title(), "Season"], "rows": [[n, v, y] for v, n, y in best[:10]]},
                   analysis=["Covers every season this save has finished plus the one in progress."])


def first_captain(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    try:
        from services.legacy_hall import ensure_legacy

        lg = ensure_legacy(view.session)
        tenures = (lg.get("captaincy") or {}).get(view.user_tid) or []
    except Exception:
        tenures = []
    caps = [t for t in tenures if isinstance(t, dict) and (t.get("role") in (None, "C", "captain"))]
    if not caps:
        return _answer("No captaincy is on record yet for your team in this save.")
    caps.sort(key=lambda t: (t.get("start_season") or 0, str(t.get("start_iso") or "")))
    c = caps[0]
    return _answer(f"First captain on record: {c.get('name')} (from {c.get('start_season')}).",
                   analysis=["Records start when this save began; real-world history before it isn't tracked."])


def last_season(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    doc = archive(view.session)
    seasons = sorted((doc.get("seasons") or {}).items(), key=lambda kv: -int(kv[0]))
    if not seasons:
        return _answer("This is the first season of the save, so there's no last year yet.")
    y, s = seasons[0]
    me = (s.get("standings") or {}).get(view.user_tid) or {}
    lines = [f"You finished {me.get('w')}-{me.get('l')}-{me.get('otl')} with {me.get('pts')} points (rank {me.get('league_rank')})."] if me else []
    if s.get("champion_id"):
        lines.append(f"The {_team_label(view, str(s['champion_id']))} won the Cup.")
    if me.get("playoff_round_reached") is not None:
        lines.append(f"Playoffs: reached round {me.get('playoff_round_reached')}." if me.get("made_playoffs") else "You missed the playoffs.")
    for a in (s.get("awards") or [])[:4]:
        lines.append(f"{a.get('award') or a.get('award_name')}: {a.get('name') or a.get('player_name') or a.get('winner')}")
    return _answer(f"Last season ({y}):", analysis=lines)


def show_last_trade(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    out = BASE["transactions"](view, {**slots, "self": True}, {**ctx, "raw": "my trades"})
    tb = out.get("table")
    if tb and tb.get("rows"):
        out["table"] = {**tb, "rows": tb["rows"][:1]}
        out["answer"] = "Your most recent trade:"
    return out


def unsupported_note(text: str) -> Any:
    def _h(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
        return _answer(text, analysis=["Ask me about players, ratings, stats, contracts, standings or this save's history."])
    return _h


# =========================================================================== reasoning
def prospects_stalling(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    ids = [p for p, (t, lv) in view.where.items() if t == view.user_tid and lv in ("Prospect", "AHL", "ECHL", "NHL") and view.row(p)["age"] <= 23]
    rows = []
    for p in ids:
        r = view.row(p)
        start = r.get("season_start_ovr")
        if start:
            rows.append((r["ovr"] - float(start), r))
    if not rows:
        return _answer("No young players have a season-start rating to compare yet.")
    rows.sort(key=lambda vr: vr[0])
    stalled = [(d, r) for d, r in rows if d <= 0.2]
    notes = [f"{len(stalled)} of {len(rows)} players 23 and under haven't gained since the season started."]
    for d, r in stalled[:5]:
        why = []
        if r["level"] == "NHL" and (r.get("toi_pg") or 0) < 12:
            why.append(f"only {r.get('toi_pg') or 0} min a night")
        if r["level"] in ("Prospect",) and not r.get("gp"):
            why.append("not playing games anywhere")
        if (r.get("gp") or 0) < 15:
            why.append("growth checkpoints come at about 20, 41 and 61 games")
        gp = r.get("gp") or 0
        if gp and r.get("pts") is not None and r["pos"] != "G":
            why.append(f"{r['pts']} points in {gp} games")
        notes.append(f"{r['name']} ({r['age']}, {r['ovr']:.0f}/{r['pot']:.0f}): {', '.join(why) or 'production below his line'}.")
    notes.append("Growth in season follows production, ice time, morale and team environment, so playing time is the lever you control.")
    t = _player_table([r for _d, r in rows[:10]], ("gp", "pts"))
    t["cols"].append("Change")
    for line, (d, _r) in zip(t["rows"], rows[:10]):
        line.append(f"{d:+.1f}")
    return _answer("Why your young players are stalling:", analysis=notes, table=t, list_ids=[r["id"] for _d, r in rows[:10]])


def playoff_history(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    doc = archive(view.session)
    seasons = sorted((doc.get("seasons") or {}).items(), key=lambda kv: int(kv[0]))
    rows, missed = [], 0
    for y, s in seasons:
        me = (s.get("standings") or {}).get(view.user_tid) or {}
        if me:
            rows.append([y, me.get("pts"), me.get("league_rank"), "yes" if me.get("made_playoffs") else "no"])
            missed += 0 if me.get("made_playoffs") else 1
    out = BASE["diagnose"](view, slots, ctx)
    if rows:
        out["answer"] = f"You've missed the playoffs {missed} of {len(rows)} seasons. This year: " + out.get("answer", "")
        out["table"] = {"cols": ["Season", "Points", "Rank", "Playoffs"], "rows": rows}
    else:
        out.setdefault("analysis", []).insert(0, "No finished seasons yet, so this is about this year only.")
    return out


def luck(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    me = _standing(view, view.user_tid)
    if not me or not me.get("gp"):
        return _answer("No games yet.")
    gp = float(me["gp"])
    gd = int(me.get("gf") or 0) - int(me.get("ga") or 0)
    expected_pts = round(gp * (1.0 + gd / gp * 0.35), 0)
    games = view.team_games(view.user_tid)
    one_goal_l = sum(1 for _d, won, gf, ga in games if not won and ga - gf == 1)
    notes = [f"Goal difference {gd:+d} over {int(gp)} games; teams with that usually have about {expected_pts:.0f} points, you have {me.get('pts')}.",
             f"{one_goal_l} one-goal losses and {me.get('otl')} overtime/shootout losses."]
    goalie = [r for r in view.team_rows(view.user_tid) if r["pos"] == "G" and r.get("sv_pct")]
    if goalie:
        g = max(goalie, key=lambda r: r.get("gp") or 0)
        notes.append(f"Starter {g['name']} at {g['sv_pct']:.3f} save %.")
    verdict = "You've been unlucky: the underlying numbers are better than the record." if me.get("pts", 0) < expected_pts - 2 \
        else "The record matches the play; it isn't just bad luck."
    return _answer(verdict, analysis=notes)


def players_want_out(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    demands = getattr(view.session, "trade_demands", None) or {}
    ids = [p for p in (demands.keys() if isinstance(demands, dict) else []) if p in view.players and view.where[p][0] == view.user_tid]
    if ids:
        rows = [view.row(p) for p in ids]
        return _answer(f"{len(rows)} of your players have asked out.", analysis=[f"{r['name']}: {'; '.join(_morale_reasons(view, r['id'])[:2]) or 'see his dossier'}" for r in rows],
                       rows=rows, extra=("cap", "status"))
    out = team_morale(view, slots, ctx)
    out["answer"] = "Nobody has formally asked for a trade. " + out.get("answer", "")
    return out


# =========================================================================== strategy (trade-offs)
def strategy(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    mine = view.team_rows(view.user_tid)
    space = _cap_space(view, view.user_tid) or 0.0
    sellers = _sellers(view)
    others = [view.row(p) for p, (t, lv) in view.where.items() if lv in ("NHL", "FA") and t != view.user_tid]
    notes, rows = [], []
    if "younger" in raw:
        olds = sorted((r for r in mine if r["age"] >= 30), key=lambda r: r["ovr"])[:4]
        young = sorted((r for r in others if r["age"] <= 24 and r["ovr"] >= 80 and (r["team_id"] in sellers or r["level"] == "FA")), key=lambda r: -r["ovr"])[:6]
        notes.append("Move your weakest veterans and replace them with young players who already play at an NHL level.")
        notes.append("Sell: " + ", ".join(f"{r['name']} ({r['age']}, {r['ovr']:.0f})" for r in olds))
        rows = young
        head = "Younger without falling off: swap aging depth for 80+ rated players 24 and under."
    elif "asset" in raw or "spending" in raw:
        rows = sorted((r for r in others if r["level"] == "FA" and (r["contract"]["cap_hit_m"] or 0.8) <= max(space, 0.8)), key=lambda r: -r["ovr"])[:8]
        head = "Without spending assets, free agency is the only route: best unsigned players you can fit."
        notes.append(f"Cap space {_money(space)}.")
    elif "defen" in raw and "offen" in raw:
        rows = sorted((r for r in others if r["pos"] in ("D", "LD", "RD") and (metric_value(view, r["id"], "ch:defence") or 0) >= 80
                       and (metric_value(view, r["id"], "ch:offence") or 0) >= 78), key=lambda r: -r["ovr"])[:8]
        head = "Improve defense without losing offense: target two-way defensemen (80+ defence, 78+ offence)."
    elif "now" in raw and ("later" in raw or "rebuild" in raw):
        rows = sorted((r for r in others if r["age"] <= 25 and r["ovr"] >= 82 and r["contract"]["years_left"] >= 3), key=lambda r: -r["ovr"])[:8]
        head = "Win now and later: players 25 or younger, already 82+, signed three-plus years."
    elif "veteran" in raw and "contend" in raw:
        rows = sorted((r for r in mine if r["age"] >= 30), key=lambda r: (r["ovr"], -(r["contract"]["cap_hit_m"] or 0)))[:6]
        head = "Trade veterans and still contend: move the low-impact ones, keep the core."
        notes.append("These are your 30+ players with the least impact for their cap hit; your best veterans stay.")
    elif "cap" in raw and "talent" in raw:
        scored = []
        for r in mine:
            v = metric_value(view, r["id"], "value")
            if v is not None:
                scored.append((v, r))
        scored.sort(key=lambda vr: vr[0])
        rows = [r for _v, r in scored[:6]]
        head = "Free cap without losing talent: move the contracts that pay most above their value."
    elif "tough" in raw and "slow" in raw:
        rows = sorted((r for r in others if (metric_value(view, r["id"], "ch:physical") or 0) >= 82 and (metric_value(view, r["id"], "ch:transition") or 0) >= 80),
                      key=lambda r: -r["ovr"])[:8]
        head = "Tougher without getting slower: physical 82+ and transition 80+."
    elif "goalie" in raw and ("cheap" in raw or "cheaply" in raw):
        gs = sorted((r for r in others if r["pos"] == "G" and (r["contract"]["cap_hit_m"] or 0) <= 4.0), key=lambda r: -r["ovr"])[:8]
        rows = gs
        head = "A good goalie cheaply: best goalies at $4M or less."
    elif "prospect" in raw:
        rows = sorted((view.row(p) for p, (t, lv) in view.where.items() if lv in ("Prospect", "AHL") and t in sellers), key=lambda r: -r["pot"])[:8]
        head = "Fastest way to a deeper prospect pool: buy prospects from rebuilding teams, or trade a veteran for picks plus a prospect."
    elif "tank" in raw:
        out = BASE["standings"](view, slots, {**ctx, "raw": "should i tank"})
        out.setdefault("analysis", []).append("Profit isn't something this save tracks, so I can only speak to the hockey side.")
        return out
    else:
        return BASE["diagnose"](view, slots, ctx)
    notes.insert(0, "These goals pull against each other, so this is the compromise that gives up the least.")
    return _answer(head, analysis=notes, rows=rows, extra=("cap", "status", "pts"))


# =========================================================================== draft / prospects
def draft_lists(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    from services.award_history import _draft_payload_from_player

    def dp(pid: str) -> Dict[str, Any]:
        return _draft_payload_from_player(view.players[pid])

    nhl = [p for p, (t, lv) in view.where.items() if lv == "NHL"]
    if "not a prospect" in raw or "anymore" in raw or "graduated" in raw:
        rows = sorted((view.row(p) for p in nhl if view.row(p)["age"] <= 23 and (view.row(p).get("gp") or 0) >= 1), key=lambda r: -r["ovr"])[:10]
        return _answer(f"Best graduated prospect: {rows[0]['name']} ({rows[0]['age']}, {rows[0]['ovr']:.0f})." if rows else "Nobody yet.",
                       analysis=["Young players already playing NHL games, so no longer prospects."], rows=rows, extra=("gp", "pts"))
    if "late bloomer" in raw:
        rows = sorted((view.row(p) for p in nhl if view.row(p)["age"] >= 24 and view.row(p).get("season_start_ovr")),
                      key=lambda r: -(r["ovr"] - float(r["season_start_ovr"])))[:10]
        return _answer(f"Best late bloomer this season: {rows[0]['name']}." if rows else "No risers yet.",
                       analysis=["Players 24 and older who've gained the most since the season started."], rows=rows, extra=("gp", "pts"))
    if "outside" in raw and ("round" in raw or "first" in raw):
        rows = sorted((view.row(p) for p in nhl if (dp(p).get("draft_round") or 0) >= 2), key=lambda r: -r["ovr"])[:10]
        return _answer(f"Best player drafted after round one: {rows[0]['name']} ({rows[0]['ovr']:.0f})." if rows else "No draft records loaded yet.",
                       rows=rows, extra=("gp", "pts"), analysis=["Real draft records load from NHL.com as players are opened; unknowns are left out."])
    if "steal" in raw or "bust" in raw:
        scored = []
        for p in nhl:
            d = dp(p)
            ov = d.get("draft_overall_pick")
            if ov:
                r = view.row(p)
                # Expected overall for the pick slot: #1 ~ 92, #32 ~ 84, #100 ~ 78, #200 ~ 74.
                exp = 92 - 8 * (min(ov, 224) ** 0.5 - 1) / 13.97 * 2.2
                scored.append((r["ovr"] - exp, r, ov))
        steal = "steal" in raw
        scored.sort(key=lambda x: x[0], reverse=steal)
        if not scored:
            return _answer("No draft records loaded yet.")
        d0, r0, ov0 = scored[0]
        t = _player_table([r for _d, r, _o in scored[:10]], ("gp", "pts"))
        t["cols"].append("Pick")
        for line, (_d, _r, ov) in zip(t["rows"], scored[:10]):
            line.append(f"#{ov}")
        return _answer(f"Biggest draft {'steal' if steal else 'bust'}: {r0['name']}, picked #{ov0}, now {r0['ovr']:.0f}.", table=t,
                       analysis=["Judged by current overall against what that pick slot usually becomes."])
    pros = [p for p, (t, lv) in view.where.items() if lv in ("Prospect", "AHL", "ECHL") and view.row(p)["age"] <= 23]
    if re.search(r"not draft(ed)? by us|other team|drafted by (someone|another)", raw):
        rows = sorted((view.row(p) for p in pros if view.where[p][0] != view.user_tid), key=lambda r: -r["pot"])[:10]
        return _answer(f"Best prospect outside your system: {rows[0]['name']} ({rows[0]['team']}, {rows[0]['pot']:.0f} POT)." if rows else "None.",
                       rows=rows, extra=("gp", "pts"))
    if "underrated" in raw:
        rows = sorted((view.row(p) for p in pros), key=lambda r: -(r["pot"] - 0.3 * r["ovr"] + (metric_value(view, r["id"], "pts") or 0) * 0.2))[:10]
        return _answer(f"Underrated prospect: {rows[0]['name']} ({rows[0]['team']}, {rows[0]['ovr']:.0f}/{rows[0]['pot']:.0f})." if rows else "None.",
                       analysis=["High potential relative to his current rating, so he's easy to overlook."], rows=rows, extra=("gp", "pts"))
    if "fail" in raw or "floor" in raw or "bust risk" in raw:
        rows = sorted((view.row(p) for p in pros if view.row(p)["age"] >= 20), key=lambda r: (r["pot"] - r["ovr"] < 3, r["ovr"]))[:10]
        rows = sorted(rows, key=lambda r: (r["ovr"] + r["pot"]))[:10]
        return _answer(f"Most likely to fall short: {rows[0]['name']} ({rows[0]['age']}, {rows[0]['ovr']:.0f}/{rows[0]['pot']:.0f})." if rows else "None.",
                       analysis=["Older prospects whose current and projected ratings are both low; the floor is where they already are."],
                       rows=rows, extra=("gp", "pts"))
    if "ceiling" in raw:
        return rank_metric(view, {**slots, "metric": "pot"}, {**ctx, "raw": raw + " prospects"})
    return BASE["top_prospects"](view, slots, ctx)


# =========================================================================== trade logic
def trade_logic(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    mine = view.team_rows(view.user_tid)
    m = re.search(r"over (\d{2})", raw)
    if "everyone" in raw or "everybody" in raw:
        if m:
            age = int(m.group(1))
            rows = sorted((r for r in mine if r["age"] > age), key=lambda r: -r["ovr"])
            notes = [f"That's {len(rows)} players and {sum(r['contract']['cap_hit_m'] or 0 for r in rows):.1f}M of cap."]
            core = [r["name"] for r in rows if r["ovr"] >= 88]
            if core:
                notes.append(f"It would include your core ({', '.join(core[:4])}); a full sell-off like that resets you to a lottery team.")
            nmc = [r["name"] for r in rows if r["contract"]["nmc"]]
            if nmc:
                notes.append(f"No-move clauses need consent: {', '.join(nmc)}.")
            return _answer(f"Players over {age}:", analysis=notes, rows=rows, extra=("cap", "status"))
        core = sorted(mine, key=lambda r: -(r["ovr"] + (r["pot"] - r["ovr"]) * 0.5 - max(0, r["age"] - 28)))[:5]
        rest = [r for r in mine if r not in core]
        return _answer("Untouchables (your top five by overall, upside and age):", analysis=[", ".join(r["name"] for r in core),
                       "Everyone else is listed as available, weakest value first."], rows=sorted(rest, key=lambda r: r["ovr"]), extra=("cap", "status"))
    if "hurt" in raw and "least" in raw:
        rows = sorted(mine, key=lambda r: (r["ovr"], -(r["contract"]["cap_hit_m"] or 0)))[:8]
        return _answer(f"The trade that hurts least moves {rows[0]['name']}.", analysis=["Lowest-rated players first, with bigger cap hits higher up because moving them frees space."],
                       rows=rows, extra=("cap", "status"))
    if "top" in raw and re.search(r"top (\d+)", raw):
        st = sorted(view.standings(), key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
        me = _standing(view, view.user_tid) or {}
        n = int(re.search(r"top (\d+)", raw).group(1))
        target = st[n - 1] if len(st) >= n else None
        gap = int(target.get("pts") or 0) - int(me.get("pts") or 0) if target else 0
        return _answer(f"You're {me.get('league_rank') or me.get('rank')}th; the #{n} team has {gap} more points.",
                       analysis=["Trades lift you roughly one or two points per overall point added to a top-six forward or top-four D over a season.",
                                 "Ask “best trade targets” for who to go after."])
    if "trading down" in raw or "trade down" in raw:
        return _answer("Trading down pays when the board is flat around your pick: you give a few spots and get an extra second-rounder or better.",
                       analysis=["It's worth it when the player you'd take is likely still there a few picks later.",
                                 "It isn't when a clear top talent sits at your slot."], followups=["Draft history"])
    pos = None
    if re.search(r"\b(goalie|goaltending|goaltender)\b", raw):
        pos = "G"
    elif re.search(r"\b(defenseman|defense|defence)\b", raw):
        pos = "D"
    elif re.search(r"\b(center|forward|winger|scoring)\b", raw):
        pos = "F"
    if "prospect" in raw:
        st = sorted(view.standings(), key=lambda r: r.get("pts", 0))[:8]
        return _answer("Rebuilding teams value prospects most:", analysis=[", ".join(str(r.get("name")) for r in st)],
                       table={"cols": ["Team", "PTS"], "rows": [[r.get("name"), r.get("pts")] for r in st]})
    players = _ctx_players(view, slots, ctx.get("conv", {})) if re.search(r"\b(him|he|this player|this guy)\b", raw) or slots.get("players") else []
    if players and ("overpay" in raw or "who would" in raw or "who wants" in raw):
        r = players[0]
        pos = "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else "F")
    if pos:
        needs = []
        for tid in view.teams:
            if tid == view.user_tid:
                continue
            group = sorted((x["ovr"] for x in view.team_rows(tid) if _pos_ok(x, pos)), reverse=True)[: (2 if pos == "G" else 4 if pos == "D" else 6)]
            if not group:
                continue
            space = _cap_space(view, tid) or 0.0
            needs.append((statistics.mean(group), space, tid))
        needs.sort(key=lambda x: (x[0], -x[1]))
        label = {"G": "goaltending", "D": "defensemen", "F": "forwards"}[pos]
        rows = [[_team_label(view, t), f"{avg:.1f}", _money(sp)] for avg, sp, t in needs[:8]]
        head = (f"Teams most likely to pay for {players[0]['name']}:" if players and "overpay" in raw else
                f"Teams most desperate for {label}:")
        return _answer(head + f" the {_team_label(view, needs[0][2])} lead the list.", table={"cols": ["Team", f"Top {label} avg OVR", "Cap space"], "rows": rows},
                       analysis=["Weakest group at the position first; cap space shown because a needy team still has to fit the salary."])
    if "blockbuster" in raw:
        mine_sorted = sorted(mine, key=lambda r: -r["ovr"])
        surplus_pos = statistics.mode([("D" if r["pos"] in ("D", "LD", "RD") else "F") for r in mine_sorted[:8]]) if mine_sorted else "F"
        others = [view.row(p) for p, (t, lv) in view.where.items() if lv == "NHL" and t != view.user_tid and t in _sellers(view)]
        want = "F" if surplus_pos == "D" else "D"
        target = max((r for r in others if _pos_ok(r, want)), key=lambda r: r["ovr"], default=None)
        give = next((r for r in mine_sorted[3:] if _pos_ok(r, surplus_pos)), None)
        if not target or not give:
            return _answer("I can't find a realistic blockbuster right now.")
        return _answer(f"Most realistic blockbuster: {give['name']} for {target['name']} ({target['team']}).",
                       analysis=[f"You're deep at {surplus_pos}, they're selling, and the ratings are within reach ({give['ovr']:.0f} vs {target['ovr']:.0f}).",
                                 "Expect to add a pick or prospect to bridge the gap."], rows=[give, target], extra=("cap", "status"))
    return BASE["trade_target"](view, slots, ctx)


def leaders_career_fr(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = ctx.get("raw", "")
    mine = any(w in raw.split() for w in ("we", "our", "us", "my", "we ve", "weve"))
    stat = slots.get("stat") or "pts"
    return BASE["leaders_career"](view, {**slots, "stat": stat, "self": mine or slots.get("self")}, {**ctx, "raw": raw + (" franchise" if mine else "")})


def traded_away(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    conv = ctx.get("conv", {})
    tid = str((slots.get("teams") or [None])[0] or ("" if "we" in ctx.get("raw", "").split() else conv.get("team") or "") or view.user_tid)
    try:
        from services.legacy_hall import trade_history

        trades = trade_history(view.session, team_id=tid, limit=100).get("trades") or []
    except Exception:
        trades = []
    gone = []
    for t in trades:
        for p in t.get("players") or []:
            if str(p.get("to_team_id")) != tid and str(p.get("player_id")) in view.players:
                gone.append(view.row(str(p.get("player_id"))))
    if not gone:
        return _answer(f"The {_team_label(view, tid)} haven't traded anyone away in this save yet.")
    gone.sort(key=lambda r: -r["ovr"])
    return _answer(f"Best player the {_team_label(view, tid)} traded away: {gone[0]['name']} ({gone[0]['ovr']:.0f}, now {gone[0]['team']}).",
                   rows=gone[:10], extra=("gp", "pts", "cap"))


def playoff_performer(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    try:
        from services.legacy_hall import leaders

        res = leaders(view.session, stat="pts", scope="playoffs", limit=10)
    except Exception:
        res = {"rows": []}
    rows = res.get("rows") or []
    if not rows:
        return _answer("No playoffs have been played in this save yet, so there are no playoff performers on record.")
    return _answer(f"Best playoff performer: {rows[0].get('name')} ({rows[0].get('value')} playoff points).",
                   table={"cols": ["Player", "Season", "Playoff PTS", "GP"], "rows": [[r.get("name"), r.get("season"), r.get("value"), r.get("gp")] for r in rows]})


def player_card_ctx(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _ctx_players(view, slots, ctx.get("conv", {}))
    if not rows:
        return _answer("Who? Name the player.")
    return BASE["player_card"](view, {"players": [[rows[0]["id"]]]}, ctx)


HANDLERS2 = {
    "leaders_career_fr": leaders_career_fr, "traded_away": traded_away, "playoff_performer": playoff_performer,
    "player_card_ctx": player_card_ctx,
    "rank_metric": rank_metric, "player_metric": player_metric, "other_player": other_player, "previous_player": previous_player,
    "player_age": player_age, "player_awards": player_awards, "player_morale": player_morale, "team_morale": team_morale,
    "afford": afford, "available": available, "drafted_by_us": drafted_by_us, "brother": brother, "move_position": move_position,
    "pair_with": pair_with, "compare_current": compare_current, "fit_check": fit_check, "worth_it": worth_it,
    "repeat_last": repeat_last, "worst_trade": worst_trade, "biggest_upset": biggest_upset, "season_record": season_record,
    "first_captain": first_captain, "last_season": last_season, "show_last_trade": show_last_trade,
    "prospects_stalling": prospects_stalling, "playoff_history": playoff_history, "luck": luck,
    "players_want_out": players_want_out, "strategy": strategy, "draft_lists": draft_lists, "trade_logic": trade_logic,
    "biggest_goal": unsupported_note("Individual goals aren't stored with their moment or stakes, so I can't pick the biggest one."),
    "point_streak": unsupported_note("Player point streaks aren't tracked game by game yet, so I can't find the longest."),
    "almost_trade": unsupported_note("Trades you didn't complete aren't saved, so there's no record of near-misses."),
    "attendance": unsupported_note("Attendance and finances aren't something this save tracks."),
}

LABELS2 = {
    "leaders_career_fr": "All-time records", "traded_away": "Traded away", "playoff_performer": "Playoff performers",
    "player_card_ctx": "Player profile",
    "rank_metric": "Ranking", "player_metric": "Player rating", "other_player": "The other player", "previous_player": "Earlier player",
    "player_age": "Age", "player_awards": "Awards", "player_morale": "Morale", "team_morale": "Team morale", "afford": "Can I afford him",
    "available": "Availability", "drafted_by_us": "Draft record", "brother": "Namesake", "move_position": "Position change",
    "pair_with": "Pairing fit", "compare_current": "Compare with my guy", "fit_check": "Lineup fit", "worth_it": "Worth it?",
    "repeat_last": "Follow-up", "worst_trade": "Trade review", "biggest_upset": "Upsets", "season_record": "Season records",
    "first_captain": "Captains", "last_season": "Last season", "show_last_trade": "Last trade", "prospects_stalling": "Prospect growth",
    "playoff_history": "Playoff history", "luck": "Luck check", "players_want_out": "Trade requests", "strategy": "Strategy",
    "draft_lists": "Draft and prospects", "trade_logic": "Trade market", "biggest_goal": "Not tracked", "point_streak": "Not tracked",
    "almost_trade": "Not tracked", "attendance": "Not tracked",
}


# =========================================================================== router
# (pattern on the normalised question, intent, needs a player in the conversation)
RULES: List[Tuple[str, str, bool]] = [
    (r"\balmost (made|did)\b|\bnearly (made|did)\b", "almost_trade", False),
    (r"\bnot draft(ed)? by us\b|\bdrafted by (someone|another|other)", "draft_lists", False),
    (r"\btrad(e|ing) down\b", "trade_logic", False),
    (r"\bprospect pool\b", "strategy", False),
    (r"\b(top|good|great|elite|starting|starter|number one) goalie\b.*\b(cheap|cheaply|cheaper|budget|bargain)\b", "strategy", False),
    (r"\b(the other (guy|one|player)|other (guy|one))\b", "other_player", False),
    (r"\b(asking|asked) about (before|earlier)|from before|earlier player\b", "previous_player", False),
    (r"\bhow old\b", "player_age", True),
    (r"\b(win|won) anything\b|\bhis awards\b|\bany (awards|trophies)\b", "player_awards", True),
    (r"\bwhy is (he|PLAYER) (unhappy|upset|mad|angry)\b|\bwhy (does|do) (he|PLAYER) want out\b", "player_morale", True),
    (r"\b(can i|can we) afford\b|\bafford (him|it)\b", "afford", False),
    (r"\b(still available|is (he|PLAYER) available|can (i|we) get (him|PLAYER))\b", "available", True),
    (r"\bdraft(ed)? by us\b|\b(did|was) (we|he) draft|\bour (draft )?pick\b", "drafted_by_us", True),
    (r"\bbrother\b", "brother", True),
    (r"\bmove (him|PLAYER) to\b|\bswitch (him|PLAYER) to\b|\bplay (him|PLAYER) (at|on)\b", "move_position", True),
    (r"\b(work|play|pair|fit) (well )?with\b", "pair_with", True),
    (r"\b(compare|vs|versus) (him|PLAYER) (to|with|and) my\b|\bmy current (guy|one|player)\b", "compare_current", True),
    (r"\b(would|does|will) (he|PLAYER) fit\b|\bfit (our|my) (roster|team|lineup)\b", "fit_check", True),
    (r"\bis (he|PLAYER) worth it\b|^worth it\b", "worth_it", True),
    (r"^(should i|should we) do it\b|^do it\b", "repeat_last", False),
    (r"^what about (him|PLAYER)$", "player_card_ctx", True),
    (r"\b(worst|best) trade\b", "worst_trade", False),
    (r"\bupset\b", "biggest_upset", False),
    (r"\bin a season\b|\bsingle season\b|\bseason record\b", "season_record", False),
    (r"\bfirst captain\b|\bcaptain(s)? (history|ever)\b", "first_captain", False),
    (r"\bwhat happened last (year|season)\b|\blast (year|season) (recap|summary)\b", "last_season", False),
    (r"\b(show|see) (that|the|my last) trade\b|\bthat trade again\b", "show_last_trade", False),
    (r"\bbiggest goal\b", "biggest_goal", False),
    (r"\bpoint streak\b", "point_streak", False),
    (r"\balmost (made|did)\b", "almost_trade", False),
    (r"\battendance\b", "attendance", False),
    (r"\bprospects? (stall|not develop|not growing)|\b(young|kid|prospect)s? (guy |player )?(is )?(not|isnt|arent) (growing|developing)\b|\bstalling\b|\bnot (growing|developing)\b", "prospects_stalling", False),
    (r"\b(miss|missing|missed) (the )?playoffs\b|\bno playoffs\b", "playoff_history", False),
    (r"\bgood but\b|\bunlucky\b|\bbad luck\b", "luck", False),
    (r"\bmorale (dropping|down|low|falling)\b|\bwhy is morale\b|\bteam morale\b", "team_morale", False),
    (r"\b(players?|guys|everyone|anyone) (want|wants) out\b|\bwho (wants|want) out\b|\btrade request|\bhates being here\b|\bunhappy players\b", "players_want_out", False),
    (r"\b(prospect|prospects|drafted|draft|player)\b.*\b(anymore|graduated|outside|out side|steals?|bust|underrated|fail|floor|ceiling)\b|\blate bloomer\b|\bdraft (steals?|bust)\b|\bunderrated prospect", "draft_lists", False),
    (r"\btrade everyone\b|\btrade everybody\b|\bhurts? (us )?(the )?least\b|\btrade (my way )?into (the )?top\b|\btrad(e|ing) down\b|\boverpa(y|ys|id|ying)\b|\bdesperate for\b|\bvalues? prospects?\b|\bblockbuster\b", "trade_logic", False),
    (r"\b(ever had|all time|of all time|franchise history)\b", "leaders_career_fr", False),
    (r"\b(traded away|they traded|we traded|gave away)\b", "traded_away", False),
    (r"\bplayoff (performer|performers|player|star)\b|\bbest in the playoffs\b", "playoff_performer", False),
    (r"\b(can i|i can) (actually )?afford\b|\bwithin (my|our) (cap|budget)\b", "afford", False),
]


_GOALS = {"younger", "asset", "assets", "spending", "spend", "rebuild", "contend", "competitive", "tougher", "slower",
          "cheaply", "cheap", "profitable", "immediately", "talent", "veterans", "offense", "offence"}
_LINKS = {"but", "without", "while", "still", "and", "later"}


def _is_strategy(words: List[str]) -> bool:
    """Two goals joined by "but", "without", "while"... in any order."""
    w = set(words)
    goals = w & _GOALS
    if "cap" in w and "space" in w:
        goals.add("cap")
    if "now" in w and ("later" in w or "rebuild" in w):
        return True
    return bool(goals) and bool(w & _LINKS) and (len(goals) >= 2 or bool(w & {"without", "while", "but"}))


def route(it: Any, view: LiveView, conv: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """A second-layer intent and extra slots, or None to keep the first layer's answer."""
    text = it.rewritten.lower()
    words = text.split()
    slots = dict(it.slots)
    # Player names in the text read as PLAYER for the patterns.
    probe = text
    for cands in slots.get("players") or []:
        for pid in cands[:1]:
            name = view.row(pid).get("name", "") if pid in view.players else ""
            if name:
                probe = probe.replace(name.lower(), "PLAYER")
    has_player = bool(slots.get("players")) or bool(conv.get("players"))
    if _is_strategy(words):
        return "strategy", slots
    for pattern, intent, need_player in RULES:
        if re.search(pattern, probe):
            if need_player and not has_player:
                continue
            return intent, slots

    if len(slots.get("teams") or []) >= 2 and not slots.get("players"):
        return None  # two teams: the team comparison owns this
    metric = find_metric(words)
    scoped = bool(set(words) & {"division", "conference", "available", "their", "nobody", "underrated"})
    career = bool(set(words) & {"ever", "career", "all"}) or "all time" in text
    keep = {"contracts_value", "standings", "streak", "champions", "draft_history", "transactions", "injuries", "help"}
    if career:
        keep.add("leaders_career")
    if metric and it.intent not in keep:
        if metric in ("ovr",) and not scoped and it.intent in ("best_on_team", "find_players", "compare", "player_card", "fit_partner"):
            return None
        if metric == "pot" and it.intent == "top_prospects":
            return None
        slots["metric"] = metric
        slots["low"] = bool(set(words) & LOW_WORDS) and not (set(words) & {"best", "most", "top", "highest"} and metric == "morale")
        n_players = len(slots.get("players") or [])
        if n_players >= 2 or (it.intent == "compare" and conv.get("compare")):
            return None  # comparisons keep their own handler
        if n_players == 1:
            return "player_metric", slots
        if set(words) & RANK_WORDS or text.startswith(("need", "who")):
            return "rank_metric", slots
    return None
