"""PuckGPT entry point: question in, answer out.

Confidence bands decide the reply:
  high   -> answer directly
  medium -> answer, and say what was assumed, with the other readings as chips
  low    -> ask, offering the likeliest readings as one-tap questions
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List

from .answers import HANDLERS, INTENT_LABELS
from .answers2 import HANDLERS2, LABELS2, route

HANDLERS = {**HANDLERS, **HANDLERS2}
INTENT_LABELS = {**INTENT_LABELS, **LABELS2}
from .data import live_view
from .nlu import interpret

log = logging.getLogger(__name__)

_CTX: Dict[str, Dict[str, Any]] = {}
_CTX_LOCK = threading.Lock()

_L10N = {
    "read_as": {"en": "Read as", "fr": "Compris comme", "es": "Entendido como", "de": "Verstanden als",
                "sv": "Tolkat som", "fi": "Tulkittu", "ru": "Понято как"},
    "did_you_mean": {"en": "Did you mean", "fr": "Vouliez-vous dire", "es": "¿Quisiste decir", "de": "Meinten Sie",
                     "sv": "Menade du", "fi": "Tarkoititko", "ru": "Вы имели в виду"},
    "not_sure": {"en": "I'm not sure what you're after.", "fr": "Je ne suis pas sûr de ce que vous cherchez.",
                 "es": "No estoy seguro de lo que buscas.", "de": "Ich bin nicht sicher, was Sie suchen.",
                 "sv": "Jag är inte säker på vad du menar.", "fi": "En ole varma, mitä haet.", "ru": "Не уверен, что вы ищете."},
    "answer_in_en": {"en": "", "fr": "Réponse en anglais.", "es": "Respuesta en inglés.", "de": "Antwort auf Englisch.",
                     "sv": "Svar på engelska.", "fi": "Vastaus englanniksi.", "ru": "Ответ на английском."},
}

# One example question per intent, used for clarification chips.
_EXAMPLE = {
    "best_on_team": "Who is my best {pos}?", "find_players": "Best {pos} in the league", "fit_partner": "Who fits my second pair?",
    "trade_target": "Best trade targets", "trade_advice": "Should I trade {player}?", "player_card": "Tell me about {player}",
    "player_contract": "{player} contract", "compare": "Compare {player} and …", "leaders_season": "Most {stat} this season",
    "leaders_career": "Most {stat} all time", "streak": "Longest win streak", "diagnose": "Why are we losing?",
    "standings": "Standings", "champions": "Past Cup winners", "awards": "Award winners", "draft_history": "My best draft class",
    "top_prospects": "My top prospects", "most_improved": "Who improved the most?", "contracts_value": "Best contracts",
    "goalie_choice": "Which goalie should start?", "rank_metric": "Best transition rating in the league", "carried_us": "Who carried us?", "injuries": "Injury report",
    "transactions": "My trades", "team_card": "How is my team doing?", "social": "What are fans saying?", "help": "What can you do?", "unsupported": "What can you do?", "hall_of_fame": "Hall of Fame",
}


def _t(key: str, lang: str) -> str:
    return _L10N[key].get(lang) or _L10N[key]["en"]


def _example(intent: str, slots: Dict[str, Any], view: Any) -> str:
    from .answers import POS_NAMES, STAT_NAMES

    name = ""
    for cands in slots.get("players") or []:
        if cands and cands[0] in view.players:
            name = view.row(cands[0])["name"]
            break
    text = _EXAMPLE.get(intent, intent)
    pos = POS_NAMES.get(slots.get("position") or "", "player")
    pos = pos[:-1] if pos.endswith("s") else pos
    return text.format(pos=pos, player=name or "him", stat=STAT_NAMES.get(slots.get("stat") or "pts", "points"))


def _ctx_for(session: Any) -> Dict[str, Any]:
    sid = str(getattr(session, "session_id", "") or id(session))
    with _CTX_LOCK:
        return _CTX.setdefault(sid, {})


def _finish(out: Dict[str, Any], it: Any, q: str, t0: float) -> Dict[str, Any]:
    """Stamp the interpretation, and say which parts of the question the answer couldn't honour."""
    from .answers4 import caveats

    # The planner and reasoning layers already say what they could and couldn't split.
    own = str(it.intent or "").split(":")[0] in ("team_plan", "game_records", "reason", "verdict", "events")
    notes = [] if own else caveats(q.lower(), str(it.intent or ""))
    if notes:
        out.setdefault("analysis", [])[0:0] = notes
        it.band = "medium"
    try:
        from .data import live_view
        from .evidence import attach

        attach(live_view(_SESSION_FOR_EVIDENCE.get("s")), out, str(it.intent or ""), q.lower(), it.slots or {}) if _SESSION_FOR_EVIDENCE.get("s") is not None else None
    except Exception:
        log.debug("analytics check failed", exc_info=True)
    out.pop("_evidence_done", None)
    out["interpretation"] = it.as_dict()
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out.pop("list_ids", None)
    return out


_SESSION_FOR_EVIDENCE: Dict[str, Any] = {}


def ask(session: Any, query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    _SESSION_FOR_EVIDENCE["s"] = session
    from .normalize import normalize

    # Slang, texting shorthand and typos are cleaned once, so every layer reads the same words.
    q = normalize(str(query or "").strip()) or str(query or "").strip()
    view = live_view(session)
    ctx = _ctx_for(session)
    if not q:
        return {**HANDLERS["help"](view, {}, {}), "interpretation": None}

    # Dates go straight to the game archive.
    from services.sim_history import _games_on, _parse_date

    date = _parse_date(q.lower())
    if date:
        res = _games_on(session, date)
        return {"answer": res.get("answer"), "analysis": [], "table": {"cols": ["Date", "Game"], "rows": [[h.get("date"), h.get("summary")] for h in res.get("hits") or []]},
                "interpretation": {"intent": "games_on_date", "confidence": 1.0, "band": "high", "rewritten": f"games on {date}", "slots": {"date": date}},
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    it = interpret(q, view.world, ctx)
    from . import answers6

    try:
        early = answers6.plan(q, it.slots, view)
    except Exception:
        log.exception("puckgpt team planner failed")
        early = None
    # Questions with an opponent, a time window, a split or "which team" go straight to
    # the planner; plain "what's our record" still uses the fixed team answers.
    if early and (early["rank"] or early.get("opp") or early["window"] or early["metric"] in ("pp", "pk", "one_goal", "ot", "so", "gf_pg", "ga_pg")):
        try:
            out = answers6.run(early, view)
        except Exception:
            log.exception("puckgpt team plan run failed")
            out = None
        if out:
            it.intent, it.confidence, it.band = f"team_plan:{early['metric']}", 0.88, "high"
            it.rewritten = " ".join(q.split())
            if early.get("focus"):
                ctx["team"] = str(early["focus"])
            ctx["intent"] = it.intent
            return _finish(out, it, q, t0)
    from . import answers7

    vk = answers7.verdict_kind(q)
    if vk or answers7.is_event_question(q):
        try:
            out = answers7.verdict(view, vk, q, it.slots) if vk else answers7.events(view, q, it.slots)
        except Exception:
            log.exception("puckgpt verdict/event layer failed")
            out = None
        if out:
            it.intent, it.confidence, it.band = (f"verdict:{vk}" if vk else "events:search"), 0.85, "high"
            it.rewritten = " ".join(q.split())
            ctx["intent"] = it.intent
            return _finish(out, it, q, t0)
    from . import answers8

    mode = answers8.match(q)
    if mode:
        try:
            out = answers8.run(mode, view, q, it.slots, ctx)
        except Exception:
            log.exception("puckgpt reasoning mode %s failed", mode)
            out = None
        if out:
            it.intent, it.confidence, it.band = f"reason:{mode}", 0.85, "high"
            it.rewritten = " ".join(q.split())
            focus = out.get("focus_ids") or []
            if focus:
                ctx["players"] = focus[:4]
            ctx["intent"] = it.intent
            return _finish(out, it, q, t0)
    from . import answers5

    kind = answers5.match(q)
    if kind:
        try:
            out = answers5.run(kind, view, q, it.slots)
        except Exception:
            log.exception("puckgpt game records %s failed", kind)
            out = {"answer": "That question hit a snag reading the save.", "analysis": ["Try rephrasing, or ask something narrower."]}
        it.intent, it.confidence, it.band = f"game_records:{kind}", 0.9, "high"
        it.rewritten = " ".join(q.split())
        ctx["intent"] = it.intent
        return _finish(out, it, q, t0)
    from . import answers3, history

    past = history.match(q)
    if past:
        try:
            out = history.run(past, view, q, team=str((it.slots.get("teams") or [""])[0] or ""))
        except Exception:
            log.exception("puckgpt history %s failed", past.__name__)
            out = {"answer": "That question hit a snag reading the save's history.", "analysis": ["Try rephrasing, or ask something narrower."]}
        it.intent, it.confidence, it.band = f"history:{past.__name__}", 0.9, "high"
        it.rewritten = " ".join(q.split())
        focus = out.get("focus_ids") or []
        if focus:
            ctx["players"] = focus[:4]
            ctx["history"] = ([p for p in ctx.get("history") or [] if p != focus[0]] + focus[:1])[-20:]
        ctx["intent"] = it.intent
        return _finish(out, it, q, t0)
    hit = answers3.match(q)
    if hit:
        action, arg = hit
        try:
            out = answers3.run(action, arg, view, q, it.slots, ctx)
        except Exception:
            log.exception("puckgpt analytics %s failed", action)
            out = {"answer": "That question hit a snag reading the save.", "analysis": ["Try rephrasing, or ask something narrower."]}
        name = arg if isinstance(arg, str) else arg[0]
        it.intent, it.confidence, it.band = f"{action}:{name}", 0.9, "high"
        it.rewritten = " ".join(q.split())  # this layer reads the question as typed
        focus = out.get("focus_ids") or []
        if focus:
            ctx["players"] = focus[:4]
            ctx["history"] = ([p for p in ctx.get("history") or [] if p != focus[0]] + focus[:1])[-20:]
        if out.get("list_ids"):
            ctx["list"] = out["list_ids"][:15]
        ctx["intent"] = it.intent
        return _finish(out, it, q, t0)
    from . import answers4

    try:
        found = answers4.detail(view, q.lower(), it, ctx)
    except Exception:
        log.exception("puckgpt detail layer failed")
        found = None
    if found:
        it.intent, out = found
        it.confidence, it.band = max(it.confidence, 0.8), "high"
        focus = out.get("focus_ids") or []
        if focus:
            ctx["players"] = focus[:4]
            ctx["history"] = ([p for p in ctx.get("history") or [] if p != focus[0]] + focus[:1])[-20:]
        if out.get("list_ids"):
            ctx["list"] = out["list_ids"][:15]
        ctx["intent"] = it.intent
        return _finish(out, it, q, t0)
    # Generic team/game planner: any record/goals/special-teams question about clubs,
    # with any window or split, before falling back to the fixed intent handlers.
    from . import answers6

    try:
        tplan = answers6.plan(q, it.slots, view)
    except Exception:
        log.exception("puckgpt team planner failed")
        tplan = None
    if tplan and (it.band != "high" or it.intent in ("team_card", "standings", "diagnose", "help", "unsupported", "streak")):
        try:
            out = answers6.run(tplan, view)
        except Exception:
            log.exception("puckgpt team plan run failed")
            out = None
        if out:
            it.intent, it.confidence, it.band = f"team_plan:{tplan['metric']}", 0.85, "high"
            it.rewritten = " ".join(q.split())
            if tplan.get("focus"):
                ctx["team"] = str(tplan["focus"])
            ctx["intent"] = it.intent
            return _finish(out, it, q, t0)
    routed = route(it, view, ctx)
    if routed:
        it.intent, it.slots = routed
        it.confidence = max(it.confidence, 0.7)
        it.band = "high" if it.confidence >= 0.55 else "medium"
    handler = HANDLERS.get(it.intent) or HANDLERS["help"]
    hctx = {"raw": it.rewritten.lower(), "language": it.language, "conv": ctx}

    if it.band == "low":
        options = [it.intent] + [a["intent"] for a in it.alternatives]
        seen, chips = set(), []
        for intent in options:
            if intent in seen or intent not in HANDLERS:
                continue
            seen.add(intent)
            chips.append({"label": INTENT_LABELS.get(intent, intent), "query": _example(intent, it.slots, view)})
        for intent in ("diagnose", "best_on_team", "trade_target", "help"):
            if len(chips) >= 4:
                break
            if intent not in seen:
                seen.add(intent)
                chips.append({"label": INTENT_LABELS[intent], "query": _example(intent, it.slots, view)})
        out = {"answer": _t("not_sure", it.language), "analysis": [f"{_t('did_you_mean', it.language)}:"],
               "clarify": chips[:4], "followups": []}
    else:
        try:
            out = handler(view, it.slots, hctx)
        except Exception:
            log.exception("puckgpt handler %s failed", it.intent)
            out = {"answer": "That question hit a snag reading the save.", "analysis": ["Try rephrasing, or ask something narrower."]}
        if it.band == "medium" or it.assumptions:
            note = f"{_t('read_as', it.language)}: {INTENT_LABELS.get(it.intent, it.intent)}"
            if it.assumptions:
                note += " — " + "; ".join(it.assumptions)
            out.setdefault("analysis", []).insert(0, note)
            out["clarify"] = [{"label": INTENT_LABELS.get(a["intent"], a["intent"]), "query": _example(a["intent"], it.slots, view)}
                              for a in it.alternatives[:3] if a["intent"] in HANDLERS]
        if it.language != "en" and _t("answer_in_en", it.language):
            out.setdefault("analysis", []).append(_t("answer_in_en", it.language))

        # Remember what was shown so "his", "which one" and "who's better" resolve next time.
        focus = out.get("focus_ids") or []
        players = [c[0] for c in it.slots.get("players") or [] if c]
        ctx["intent"] = it.intent
        if focus or players:
            ctx["players"] = (focus or players)[:4]
            hist = [p for p in ctx.get("history") or [] if p not in ctx["players"][:1]]
            ctx["history"] = (hist + ctx["players"][:1])[-20:]
        if it.slots.get("teams"):
            ctx["team"] = str(it.slots["teams"][0])
        elif out.get("list_ids"):
            # "his" after a ranked list means the player at the top of it.
            ctx["players"] = out["list_ids"][:1]
        if out.get("list_ids"):
            ctx["list"] = out["list_ids"][:15]
        if it.intent == "compare" and len(focus) >= 2:
            ctx["compare"] = focus[:4]

    return _finish(out, it, q, t0)
