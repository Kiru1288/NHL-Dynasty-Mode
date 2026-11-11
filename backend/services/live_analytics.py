"""Live analytics, coaching notifications and three stars for the SimTracker (contract v4).

Everything counted here comes from REAL engine events (attempts, goals, penalties, on-ice lists, lines) plus the
motion track's on-ice presence for TOI / shifts / line time / matchups (labelled as estimates). Faceoffs are the
motion track's derived faceoffs (labelled derived). Hits, giveaways and takeaways are not tracked by the engine
and are reported as null rather than invented.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SIDES = ("home", "away")
HD = frozenset({"high_danger", "rebound", "breakaway"})


def _opp(s: Optional[str]) -> str:
    return "away" if s == "home" else "home"


def _pct(a: float, b: float) -> Optional[float]:
    return round(a / (a + b), 3) if (a + b) > 0 else None


def _clock(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def _blank_team() -> Dict[str, Any]:
    return {"cf": 0, "ca": 0, "sf": 0, "sa": 0, "xgf": 0.0, "xga": 0.0, "hdcf": 0, "hdca": 0, "gf": 0, "ga": 0,
            "pp_g": 0, "pp_o": 0, "pk_k": 0, "pk_f": 0, "fo_w": 0, "fo_l": 0, "blocks": 0}


def _blank_row() -> Dict[str, Any]:
    return {"cf": 0, "ca": 0, "xgf": 0.0, "xga": 0.0, "gf": 0, "ga": 0}


def _blank_player() -> Dict[str, Any]:
    return {"g": 0, "a1": 0, "a2": 0, "sog": 0, "att": 0, "cf": 0, "ca": 0, "xgf": 0.0, "xga": 0.0, "gf": 0, "ga": 0,
            "blocks": 0, "fo_w": 0, "fo_l": 0}


# ───────────────────────────── notification text (5 languages)
_N: Dict[str, Dict[str, Tuple[str, str]]] = {
    "en": {
        "pull_goalie": ("Pull the goalie?", "Down {diff} with {left} left in the third, and you're at full strength."),
        "timeout": ("Use your timeout?", "{opp} has {run} of the last {n} attempts and you trail by {diff} with {left} to play."),
        "fatigue": ("{name}'s line is gassed", "{name} has been out for {shift} (fatigue {fat}%), change before they get hemmed in."),
        "matchup": ("Bad matchup on the ice", "Their {oline} ({ocf}% CF tonight) is out against your {line}, get a top unit on."),
        "line_struggling": ("{line} getting caved", "{line}: {cf} CF / {ca} CA in {toi} together, shelter them or shorten the bench?"),
        "line_hot": ("{line} is rolling", "{line}: {cf} CF / {ca} CA and {xgf} xGF in {toi}, give them more ice."),
        "goalie": ("{goalie} is struggling", "{goalie}: {svp} on {sa} shots, {gsax} goals saved above expected."),
        "goalie_opp": ("Their goalie looks shaky", "{goalie} is {svp} on {sa} shots tonight, keep shooting."),
        "pp_unit": ("Power play misfiring", "Power play {ppg}/{ppo} with {pps} shots in {pptoi}, try a more aggressive setup?"),
        "pk_unit": ("Penalty kill leaking", "They have {ppg} power-play goals on {ppo} chances, tighten the box?"),
        "momentum": ("They're taking over", "{opp} has {run} of the last {n} shot attempts."),
        "penalty_trouble": ("Too many penalties", "{pim} penalty minutes already, calm the forecheck down?"),
    },
    "fr": {
        "pull_goalie": ("Retirer le gardien?", "Vous tirez de l'arrière par {diff} avec {left} à jouer, à forces égales."),
        "timeout": ("Prendre votre temps d'arrêt?", "{opp} a {run} des {n} derniers tirs tentés, écart de {diff} avec {left} à jouer."),
        "fatigue": ("Le trio de {name} est épuisé", "{name} est sur la glace depuis {shift} (fatigue {fat} %), changez."),
        "matchup": ("Mauvais duel sur la glace", "Leur {oline} ({ocf} % CF) affronte votre {line}, envoyez une unité de pointe."),
        "line_struggling": ("{line} en difficulté", "{line} : {cf} CF / {ca} CA en {toi}, la protéger ou raccourcir le banc?"),
        "line_hot": ("{line} en feu", "{line} : {cf} CF / {ca} CA et {xgf} xGF en {toi}, donnez-lui plus de temps."),
        "goalie": ("{goalie} en difficulté", "{goalie} : {svp} sur {sa} tirs, {gsax} buts sauvés au-dessus des attentes."),
        "goalie_opp": ("Leur gardien chancelle", "{goalie} : {svp} sur {sa} tirs, continuez de lancer."),
        "pp_unit": ("Avantage numérique stérile", "Avantage numérique {ppg}/{ppo}, {pps} tirs en {pptoi}, plus agressif?"),
        "pk_unit": ("Infériorité numérique perméable", "{ppg} buts en avantage numérique sur {ppo} occasions, resserrer le box?"),
        "momentum": ("Ils prennent le dessus", "{opp} a {run} des {n} derniers tirs tentés."),
        "penalty_trouble": ("Trop de pénalités", "Déjà {pim} minutes de pénalité, calmer l'échec avant?"),
    },
    "es": {
        "pull_goalie": ("¿Sacar al portero?", "Abajo por {diff} con {left} por jugar y en igualdad numérica."),
        "timeout": ("¿Tiempo muerto?", "{opp} tiene {run} de los últimos {n} tiros, abajo por {diff} con {left} por jugar."),
        "fatigue": ("La línea de {name} está fundida", "{name} lleva {shift} en el hielo (fatiga {fat} %), haz el cambio."),
        "matchup": ("Mal emparejamiento", "Su {oline} ({ocf} % CF) está contra tu {line}, saca una línea titular."),
        "line_struggling": ("{line} sufriendo", "{line}: {cf} CF / {ca} CA en {toi}, ¿protegerla o acortar el banquillo?"),
        "line_hot": ("{line} imparable", "{line}: {cf} CF / {ca} CA y {xgf} xGF en {toi}, dale más hielo."),
        "goalie": ("{goalie} con problemas", "{goalie}: {svp} en {sa} tiros, {gsax} goles salvados sobre lo esperado."),
        "goalie_opp": ("Su portero duda", "{goalie} está en {svp} con {sa} tiros, sigan tirando."),
        "pp_unit": ("El power play no funciona", "Power play {ppg}/{ppo} con {pps} tiros en {pptoi}, ¿más agresivo?"),
        "pk_unit": ("La inferioridad hace agua", "{ppg} goles en power play en {ppo} ocasiones, ¿cerrar la caja?"),
        "momentum": ("Están dominando", "{opp} tiene {run} de los últimos {n} tiros intentados."),
        "penalty_trouble": ("Demasiadas penalizaciones", "Ya {pim} minutos de castigo, ¿calmar el forecheck?"),
    },
    "de": {
        "pull_goalie": ("Torhüter rausnehmen?", "{diff} Tore hinten, noch {left} im letzten Drittel, volle Stärke."),
        "timeout": ("Auszeit nehmen?", "{opp} hat {run} der letzten {n} Abschlüsse, {diff} hinten, noch {left}."),
        "fatigue": ("Die Reihe um {name} ist platt", "{name} ist seit {shift} auf dem Eis (Müdigkeit {fat} %), wechseln."),
        "matchup": ("Falsches Duell auf dem Eis", "Ihre {oline} ({ocf} % CF) spielt gegen deine {line}, bring eine Top-Reihe."),
        "line_struggling": ("{line} unter Druck", "{line}: {cf} CF / {ca} CA in {toi}, schützen oder die Bank verkürzen?"),
        "line_hot": ("{line} läuft heiß", "{line}: {cf} CF / {ca} CA und {xgf} xGF in {toi}, mehr Eiszeit geben."),
        "goalie": ("{goalie} wackelt", "{goalie}: {svp} bei {sa} Schüssen, {gsax} Tore über Erwartung."),
        "goalie_opp": ("Ihr Torhüter wackelt", "{goalie} hält {svp} bei {sa} Schüssen, weiter schießen."),
        "pp_unit": ("Überzahl harmlos", "Überzahl {ppg}/{ppo} mit {pps} Schüssen in {pptoi}, aggressiver spielen?"),
        "pk_unit": ("Unterzahl löchrig", "{ppg} Überzahltore bei {ppo} Chancen, die Box enger machen?"),
        "momentum": ("Sie übernehmen", "{opp} hat {run} der letzten {n} Abschlussversuche."),
        "penalty_trouble": ("Zu viele Strafen", "Schon {pim} Strafminuten, den Forecheck beruhigen?"),
    },
    "sv": {
        "pull_goalie": ("Plocka målvakten?", "Under med {diff} och {left} kvar, fulla lag."),
        "timeout": ("Ta timeout?", "{opp} har {run} av de senaste {n} skottförsöken, under med {diff} och {left} kvar."),
        "fatigue": ("{name}s kedja är slut", "{name} har varit inne {shift} (trötthet {fat} %), byt."),
        "matchup": ("Dålig matchning", "Deras {oline} ({ocf} % CF) mot din {line}, skicka ut en toppkedja."),
        "line_struggling": ("{line} pressas", "{line}: {cf} CF / {ca} CA på {toi}, skydda dem eller korta bänken?"),
        "line_hot": ("{line} går som tåget", "{line}: {cf} CF / {ca} CA och {xgf} xGF på {toi}, ge dem mer istid."),
        "goalie": ("{goalie} har det tufft", "{goalie}: {svp} på {sa} skott, {gsax} mål räddade över förväntan."),
        "goalie_opp": ("Deras målvakt vacklar", "{goalie} har {svp} på {sa} skott, fortsätt skjuta."),
        "pp_unit": ("Powerplayet fungerar inte", "Powerplay {ppg}/{ppo} med {pps} skott på {pptoi}, mer aggressivt?"),
        "pk_unit": ("Boxplayet läcker", "{ppg} powerplaymål på {ppo} chanser, tajtare box?"),
        "momentum": ("De tar över", "{opp} har {run} av de senaste {n} skottförsöken."),
        "penalty_trouble": ("För många utvisningar", "Redan {pim} utvisningsminuter, lugna forecheckingen?"),
    },
}
_ACT: Dict[str, Dict[str, str]] = {
    "en": {"pull": "Pull the goalie", "timeout": "Call timeout", "next": "Send out {line}", "shorten": "Shorten the bench",
           "roll": "Roll four lines", "pp_aggr": "Aggressive power play", "pk_pass": "Passive penalty kill", "shoot": "Shoot more",
           "calm": "Passive forecheck", "defend": "Play it safe"},
    "fr": {"pull": "Retirer le gardien", "timeout": "Temps d'arrêt", "next": "Envoyer {line}", "shorten": "Raccourcir le banc",
           "roll": "Quatre trios", "pp_aggr": "Avantage agressif", "pk_pass": "Infériorité prudente", "shoot": "Lancer davantage",
           "calm": "Échec avant prudent", "defend": "Jouer prudemment"},
    "es": {"pull": "Sacar al portero", "timeout": "Tiempo muerto", "next": "Sacar {line}", "shorten": "Acortar el banquillo",
           "roll": "Rotar cuatro líneas", "pp_aggr": "Power play agresivo", "pk_pass": "Inferioridad prudente", "shoot": "Tirar más",
           "calm": "Forecheck pasivo", "defend": "Jugar seguro"},
    "de": {"pull": "Torhüter raus", "timeout": "Auszeit", "next": "{line} aufs Eis", "shorten": "Bank verkürzen",
           "roll": "Vier Reihen rotieren", "pp_aggr": "Aggressives Überzahlspiel", "pk_pass": "Vorsichtige Unterzahl", "shoot": "Mehr schießen",
           "calm": "Passiver Forecheck", "defend": "Sicher spielen"},
    "sv": {"pull": "Plocka målvakten", "timeout": "Ta timeout", "next": "Skicka ut {line}", "shorten": "Korta bänken",
           "roll": "Rulla fyra kedjor", "pp_aggr": "Aggressivt powerplay", "pk_pass": "Försiktigt boxplay", "shoot": "Skjut mer",
           "calm": "Passiv forechecking", "defend": "Spela säkert"},
}


class LiveAnalytics:
    """Incremental counters fed by every live event; build() renders snapshot.analytics."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop
        self.team = {"home": _blank_team(), "away": _blank_team()}
        self.lines: Dict[str, Dict[str, Any]] = {}      # "home:F1" -> counters
        self.players: Dict[str, Dict[str, Any]] = {}
        self.goalies: Dict[str, Dict[str, Any]] = {}
        self.match_cf: Dict[Tuple[str, str], List[int]] = {}
        self.att_sides: List[str] = []
        self.pp_shots = {"home": 0, "away": 0}
        self.pp_toi = {"home": 0.0, "away": 0.0}
        self.notes: List[Dict[str, Any]] = []
        self._nid = 0
        self._last_new_gc = -999.0
        self._cool: Dict[str, float] = {}
        self._seen: set = set()

    # ── event intake
    def _p(self, pid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not pid:
            return None
        r = self.players.get(pid)
        if r is None:
            r = self.players[pid] = _blank_player()
        return r

    def _unit_key(self, side: str, ids: List[str], strength: str) -> Optional[str]:
        kind = "PP" if strength == "PP" else ("PK" if strength == "SH" else None)
        if kind is None:
            return None
        P = getattr(self.loop.motion, "pres", None) or {}
        reg = ((P.get("units") or {}).get(side) or {}).get(kind) or []
        s = set(ids or [])
        best, bo = 0, -1
        for i, u in enumerate(reg):
            o = len(u & s)
            if o > bo:
                best, bo = i, o
        return f"{kind}{best + 1}"

    def on_event(self, ev: Dict[str, Any]) -> None:
        try:
            self._on_event(ev)
        except Exception:
            logger.warning("analytics failed on %s", ev.get("type"), exc_info=True)

    def _on_event(self, ev: Dict[str, Any]) -> None:
        typ = ev.get("type")
        side = ev.get("side")
        meta = ev.get("meta") or {}
        if typ in ("attempt", "goal") and side in SIDES:
            o = _opp(side)
            oc = ev.get("outcome")
            xg = float(ev.get("xg") or 0.0)
            hd = ev.get("chance") in HD
            T, U = self.team[side], self.team[o]
            T["cf"] += 1
            U["ca"] += 1
            T["xgf"] += xg
            U["xga"] += xg
            if hd:
                T["hdcf"] += 1
                U["hdca"] += 1
            if oc in ("SAVED", "GOAL"):
                T["sf"] += 1
                U["sa"] += 1
            if oc == "GOAL":
                T["gf"] += 1
                U["ga"] += 1
                if ev.get("strength") == "PP":
                    T["pp_g"] += 1
            if oc == "BLOCKED":
                U["blocks"] += 1
                b = self._p(ev.get("blocker_id"))
                if b is not None:
                    b["blocks"] += 1
            self.att_sides = (self.att_sides + [side])[-12:]
            if ev.get("strength") == "PP":
                self.pp_shots[side] += int(oc in ("SAVED", "GOAL"))
            sh = self._p(ev.get("shooter_id"))
            if sh is not None:
                sh["att"] += 1
                sh["sog"] += int(oc in ("SAVED", "GOAL"))
                if oc == "GOAL":
                    sh["g"] += 1
            if oc == "GOAL":
                for i, a in enumerate(ev.get("assist_ids") or []):
                    r = self._p(a)
                    if r is not None:
                        r["a1" if i == 0 else "a2"] += 1
            on = ev.get("on_ice") or {}
            for s_, sign in ((side, 1), (o, -1)):
                for pid in on.get(s_) or []:
                    r = self._p(pid)
                    if sign > 0:
                        r["cf"] += 1
                        r["xgf"] += xg
                        r["gf"] += int(oc == "GOAL")
                    else:
                        r["ca"] += 1
                        r["xga"] += xg
                        r["ga"] += int(oc == "GOAL")
            keys: Dict[str, List[str]] = {}
            for s_ in SIDES:
                rec = (meta.get("lines") or {}).get(s_) or {}
                ks = []
                if rec.get("F") is not None:
                    ks.append(f"F{int(rec['F']) + 1}")
                if rec.get("D") is not None:
                    ks.append(f"D{int(rec['D']) + 1}")
                if not ks:
                    st_ = ev.get("strength") if s_ == side else {"PP": "SH", "SH": "PP"}.get(ev.get("strength"), "")
                    uk = self._unit_key(s_, on.get(s_) or [], st_ or "")
                    if uk:
                        ks.append(uk)
                keys[s_] = ks
                for k in ks:
                    L = self.lines.setdefault(f"{s_}:{k}", _blank_row())
                    if s_ == side:
                        L["cf"] += 1
                        L["xgf"] += xg
                        L["gf"] += int(oc == "GOAL")
                    else:
                        L["ca"] += 1
                        L["xga"] += xg
                        L["ga"] += int(oc == "GOAL")
            hf = next((k for k in keys.get("home", []) if k.startswith(("F", "PP", "PK"))), None)
            af = next((k for k in keys.get("away", []) if k.startswith(("F", "PP", "PK"))), None)
            if hf and af:
                m = self.match_cf.setdefault((hf, af), [0, 0])
                m[0 if side == "home" else 1] += 1
            gid = ev.get("goalie_id")
            if gid and not ev.get("empty_net") and oc in ("SAVED", "GOAL", "MISSED"):
                G = self.goalies.setdefault(gid, {"side": o, "sa": 0, "ga": 0, "xga": 0.0, "hd_sa": 0, "hd_ga": 0})
                G["xga"] += xg
                if oc in ("SAVED", "GOAL"):
                    G["sa"] += 1
                    G["ga"] += int(oc == "GOAL")
                    if hd:
                        G["hd_sa"] += 1
                        G["hd_ga"] += int(oc == "GOAL")
        elif typ == "penalty" and side in SIDES:
            self.team[_opp(side)]["pp_o"] += 1
            self.team[side]["pk_f"] += 1
        elif typ == "penalty_end" and side in SIDES:
            if meta.get("reason") == "expired":
                self.team[side]["pk_k"] += 1
        elif typ == "faceoff":
            w = meta.get("winner_side")
            if w in SIDES:
                self.team[w]["fo_w"] += 1
                self.team[_opp(w)]["fo_l"] += 1
                a, b = self._p(meta.get("winner_id")), self._p(meta.get("loser_id"))
                if a is not None:
                    a["fo_w"] += 1
                if b is not None:
                    b["fo_l"] += 1

    def tick(self, gc: float, dt: float) -> None:
        mp = self.loop.manpower(gc) if self.loop._engine_captured else {}
        pp = mp.get("pp_side")
        if pp in SIDES and dt > 0:
            self.pp_toi[pp] += dt

    # ── rendering
    def _toi(self) -> Tuple[Dict[str, Dict[str, Any]], bool]:
        loop = self.loop
        P = getattr(loop.motion, "pres", None) or {}
        pres = P.get("p") or {}
        tot = sum(r["toi"] for r in pres.values())
        elapsed = float(loop.game_sec_elapsed)
        if pres and tot >= 0.6 * 12.0 * max(1.0, elapsed):
            return pres, True
        out = {}
        for pid, side in loop.book.side.items():
            out[pid] = {"side": side, "toi": loop.shifts.toi_est(pid, side), "shifts": loop.shifts.shifts.get(pid, 0),
                        "on": pid in loop.shifts.on_since, "since": loop.shifts.on_since.get(pid, elapsed)}
        return out, False

    @staticmethod
    def _game_score(r: Dict[str, Any]) -> float:
        return (0.75 * r["g"] + 0.7 * r["a1"] + 0.55 * r["a2"] + 0.075 * r["sog"] + 0.05 * r["blocks"] + 0.01 * r["fo_w"]
                - 0.01 * r["fo_l"] + 0.05 * r["cf"] - 0.05 * r["ca"] + 0.15 * r["gf"] - 0.15 * r["ga"])

    def build(self) -> Dict[str, Any]:
        loop = self.loop
        now = float(loop.game_sec_elapsed)
        book = loop.book
        toi, from_motion = self._toi()
        st = loop.state
        teams = {}
        for s in SIDES:
            T = self.team[s]
            o = self.team[_opp(s)]
            shp = (T["gf"] / T["sf"]) if T["sf"] else 0.0
            svp = (1.0 - o["gf"] / o["sf"]) if o["sf"] else 1.0
            teams[s] = {"cf": T["cf"], "ca": T["ca"], "cf_pct": _pct(T["cf"], T["ca"]), "sf": T["sf"], "sa": T["sa"],
                        "xgf": round(T["xgf"], 2), "xga": round(T["xga"], 2), "xg_pct": _pct(T["xgf"], T["xga"]),
                        "hdcf": T["hdcf"], "hdca": T["hdca"], "gf": T["gf"], "ga": T["ga"],
                        "pp": f"{T['pp_g']}/{T['pp_o']}", "pk": f"{T['pk_k']}/{T['pk_f']}", "fo_w": T["fo_w"], "fo_l": T["fo_l"],
                        "hits": None, "blocks": T["blocks"], "giveaways": None, "takeaways": None,
                        "pdo": round(100.0 * (shp + svp), 1) if (T["sf"] or o["sf"]) else None}
        P = getattr(loop.motion, "pres", None) or {}
        pline = P.get("line") or {}
        starts = P.get("starts") or {}
        lines: Dict[str, List[Dict[str, Any]]] = {"home": [], "away": []}
        for s in SIDES:
            ln = loop.lines.get(s) or {}
            defs: List[Tuple[str, List[str]]] = [(f"F{i + 1}", list(g or [])) for i, g in enumerate(ln.get("F") or [])]
            defs += [(f"D{i + 1}", list(g or [])) for i, g in enumerate(ln.get("D") or [])]
            for kind in ("PP", "PK"):
                for i in range(2):
                    key = f"{kind}{i + 1}"
                    if f"{s}:{key}" in pline or f"{s}:{key}" in self.lines:
                        defs.append((key, list((pline.get(f"{s}:{key}") or {}).get("ids") or [])))
            for key, ids in defs:
                c = self.lines.get(f"{s}:{key}") or _blank_row()
                pl = pline.get(f"{s}:{key}") or {}
                zs = starts.get(f"{s}:{key}") or {}
                fats = [loop.shifts.fatigue(p, now) for p in ids] if ids else []
                lines[s].append({"key": key, "player_ids": ids, "toi_sec": round(pl.get("toi", 0.0), 1), "shifts": int(pl.get("shifts", 0)),
                                 "cf": c["cf"], "ca": c["ca"], "cf_pct": _pct(c["cf"], c["ca"]), "xgf": round(c["xgf"], 2),
                                 "xga": round(c["xga"], 2), "xg_pct": _pct(c["xgf"], c["xga"]), "gf": c["gf"], "ga": c["ga"],
                                 "oz_starts": int(zs.get("oz", 0)), "dz_starts": int(zs.get("dz", 0)),
                                 "last_shift_sec": round(pl.get("last_shift", 0.0), 1),
                                 "fatigue": round(sum(fats) / len(fats), 2) if fats else None})
        players: Dict[str, Dict[str, Any]] = {}
        for pid, side in book.side.items():
            if book.pos(pid) == "G":
                continue
            r = self.players.get(pid) or _blank_player()
            tr = toi.get(pid) or {}
            sh = int(tr.get("shifts", 0) or 0)
            t_ = float(tr.get("toi", 0.0) or 0.0)
            pm = int((st.p.get(pid) or {}).get("plus_minus", 0))
            gs = self._game_score(r)
            players[pid] = {"side": side, "toi_sec": round(t_, 1), "shifts": sh, "avg_shift_sec": round(t_ / sh, 1) if sh else None,
                            "g": r["g"], "a": r["a1"] + r["a2"], "pts": r["g"] + r["a1"] + r["a2"], "sog": r["sog"], "att": r["att"],
                            "cf": r["cf"], "ca": r["ca"], "cf_pct": _pct(r["cf"], r["ca"]), "xgf": round(r["xgf"], 2), "xga": round(r["xga"], 2),
                            "plus_minus": pm, "hits": None, "blocks": r["blocks"], "fo_w": r["fo_w"], "fo_l": r["fo_l"],
                            "fatigue": loop.shifts.fatigue(pid, now),
                            "current_shift_sec": round(max(0.0, now - float(tr.get("since", now))), 1) if tr.get("on") else 0.0,
                            "rating_tonight": round(max(0.0, min(10.0, 5.0 + 1.6 * gs)), 1)}
        goalies: Dict[str, Dict[str, Any]] = {}
        for gid, G in self.goalies.items():
            sv = G["sa"] - G["ga"]
            goalies[gid] = {"side": G["side"], "sa": G["sa"], "sv": sv, "ga": G["ga"], "sv_pct": round(sv / G["sa"], 3) if G["sa"] else None,
                            "xga": round(G["xga"], 2), "gsax": round(G["xga"] - G["ga"], 2),
                            "hd_sv_pct": round((G["hd_sa"] - G["hd_ga"]) / G["hd_sa"], 3) if G["hd_sa"] else None}
        cur = P.get("cur") or {}
        top = sorted((P.get("match") or {}).items(), key=lambda kv: -kv[1])[:6]
        matchups = {"on_ice_now": {"home_line": (cur.get("home") or {}).get("F") or (cur.get("home") or {}).get("U"),
                                   "away_line": (cur.get("away") or {}).get("F") or (cur.get("away") or {}).get("U")},
                    "top": [{"home_line": k[0], "away_line": k[1], "toi_sec": round(v, 1),
                             "cf_pct_home": _pct(*(self.match_cf.get(k) or [0, 0]))} for k, v in top]}
        mins = int(now // 60)
        note = (f"Small sample: {mins} minutes played. " if now < 1200 else f"{mins} minutes played. ")
        note += ("TOI, shifts and line time are estimates from the on-ice track; faceoffs are derived; hits are not tracked."
                 if from_motion else "TOI and shifts are estimates from the engine's units; faceoffs are derived; hits are not tracked.")
        return {"teams": teams, "lines": lines, "players": players, "goalies": goalies, "matchups": matchups, "sample_note": note}

    # ── three stars
    def stars(self) -> List[Dict[str, Any]]:
        loop = self.loop
        book = loop.book
        cands: List[Tuple[float, str, str, str]] = []
        for pid, r in self.players.items():
            side = book.side.get(pid)
            if side not in SIDES or book.pos(pid) == "G":
                continue
            gs = self._game_score(r)
            a = r["a1"] + r["a2"]
            bits = []
            if r["g"]:
                bits.append(f"{r['g']}G")
            if a:
                bits.append(f"{a}A")
            if not bits:
                bits.append(f"{r['sog']} SOG")
            cands.append((gs, pid, side, " ".join(bits)))
        for gid, G in self.goalies.items():
            sv = G["sa"] - G["ga"]
            gs = -0.75 * G["ga"] + 0.1 * sv + 0.3 * (G["xga"] - G["ga"])
            line = f"{sv} saves" + (f" {sv / G['sa']:.3f}".replace("0.", ".") if G["sa"] else "")
            cands.append((gs, gid, G["side"], line))
        cands.sort(key=lambda c: -c[0])
        out = []
        for gs, pid, side, line in cands[:3]:
            row = {"player_id": pid, "side": side, "name": book.name(pid), "line": line, "score": round(gs, 2),
                   "position": book.pos(pid), "number": book.number(pid)}
            row.update(loop.headshot_fields(pid))
            out.append(row)
        return out

    # ── coaching notifications
    def _txt(self, kind: str, **kw: Any) -> Tuple[str, str]:
        lang = self.loop.lg.lang
        t, b = (_N.get(lang) or _N["en"]).get(kind) or _N["en"][kind]
        return t.format(**kw), b.format(**kw)

    def _act(self, key: str, kind: str, payload: Dict[str, Any], **kw: Any) -> Optional[Dict[str, Any]]:
        """Only offer orders the coach controller accepts right now."""
        coach = self.loop.coach
        try:
            ok, _why = coach._validate(kind, payload)
        except Exception:
            ok = False
        if not ok:
            return None
        lab = ((_ACT.get(self.loop.lg.lang) or _ACT["en"]).get(key) or _ACT["en"][key]).format(**kw)
        return {"label": lab, "order": {"kind": kind, "payload": dict(payload)}}

    def notify(self, an: Dict[str, Any], pt: float) -> None:
        loop = self.loop
        user = loop.lg.user_side
        if user not in SIDES or loop.finished or loop.phase in ("pregame", "shootout") or loop.intermission:
            self._expire(pt)
            return
        gc = float(loop.game_sec_elapsed)
        opp = _opp(user)
        book = loop.book
        period, left = loop.clock_at(gc)
        mine, theirs = (loop.lg.score_home, loop.lg.score_away) if user == "home" else (loop.lg.score_away, loop.lg.score_home)
        diff = int(theirs) - int(mine)
        cands: List[Tuple[int, str, str, Tuple[str, str], List[Optional[Dict[str, Any]]]]] = []
        mp = loop.manpower(gc) if loop._engine_captured else {}
        my_sk = mp.get("home_skaters" if user == "home" else "away_skaters", 5)
        their_sk = mp.get("away_skaters" if user == "home" else "home_skaters", 5)
        team_say = lambda s: loop.team_say(s, loop.lg.lang)  # noqa: E731
        if period >= 3 and 1 <= diff <= 2 and left <= (150.0 if diff == 1 else 210.0) and not loop.goalie_pulled.get(user) and my_sk >= their_sk:
            cands.append((3, "pull_goalie", f"pull:{period}", self._txt("pull_goalie", diff=diff, left=_clock(left)),
                          [self._act("pull", "pull_goalie", {"pull": True})]))
        run = self.att_sides[-8:].count(opp)
        if period >= 3 and left <= 300.0 and 1 <= diff <= 2 and not loop.timeout_used:
            cands.append((2, "timeout", "timeout_late", self._txt("timeout", opp=team_say(opp), run=run, n=len(self.att_sides[-8:]), diff=diff, left=_clock(left)),
                          [self._act("timeout", "timeout", {})]))
        players = an["players"]
        lines = {L["key"]: L for L in an["lines"].get(user) or []}
        olines = {L["key"]: L for L in an["lines"].get(opp) or []}
        cur = an["matchups"]["on_ice_now"]
        my_line = cur.get(f"{user}_line")
        their_line = cur.get(f"{opp}_line")
        on_mine = [p for p in loop.state.on_ice.get(user) or [] if p in players]
        tired = sorted(on_mine, key=lambda p: -players[p]["current_shift_sec"])
        if tired and (players[tired[0]]["current_shift_sec"] >= 70.0 or players[tired[0]]["fatigue"] >= 0.85):
            p0 = tired[0]
            nxt = None
            if my_line and my_line.startswith("F"):
                nxt = f"F{int(my_line[1:]) % 4 + 1}"
            acts = [self._act("next", "next_line", {"line": nxt}, line=nxt)] if nxt else []
            cands.append((2, "fatigue", f"fat:{p0}:{int(gc // 120)}", self._txt("fatigue", name=book.short(p0), shift=_clock(players[p0]["current_shift_sec"]),
                                                                                  fat=int(round(100 * players[p0]["fatigue"]))), acts))
        if their_line and my_line and their_line.startswith("F") and my_line in ("F3", "F4"):
            oL = olines.get(their_line) or {}
            best_o = max((L for L in olines.values() if L["key"].startswith("F")), key=lambda L: (L["cf"] - L["ca"], -int(L["key"][1:])), default=None)
            if best_o is not None and best_o["key"] == their_line and (oL.get("cf", 0) + oL.get("ca", 0)) >= 4:
                ocf = int(round(100 * (oL.get("cf_pct") or 0.5)))
                cands.append((2, "matchup", f"match:{their_line}:{my_line}:{int(gc // 180)}", self._txt("matchup", oline=their_line, ocf=ocf, line=my_line),
                              [self._act("next", "next_line", {"line": "F1"}, line="F1")]))
        for key, L in lines.items():
            n = L["cf"] + L["ca"]
            if n >= 8 and L["toi_sec"] >= 150.0 and (L["cf_pct"] or 0.5) <= 0.33:
                cands.append((2, "line_struggling", f"struggle:{key}:{L['ca']}", self._txt("line_struggling", line=key, cf=L["cf"], ca=L["ca"], toi=_clock(L["toi_sec"])),
                              [self._act("shorten", "shorten_bench", {"on": True})]))
            if key.startswith("F") and n >= 8 and (L["cf_pct"] or 0.5) >= 0.65:
                cands.append((1, "line_hot", f"hot:{key}:{L['cf'] // 4}", self._txt("line_hot", line=key, cf=L["cf"], ca=L["ca"], xgf=f"{L['xgf']:.1f}", toi=_clock(L["toi_sec"])),
                              [self._act("next", "next_line", {"line": key}, line=key)]))
        for gid, G in an["goalies"].items():
            if G["sa"] < 10 or G["sv_pct"] is None:
                continue
            svp = f"{G['sv_pct']:.3f}".replace("0.", ".")
            if G["side"] == user and (G["sv_pct"] < 0.86 or G["gsax"] <= -1.5):
                cands.append((2, "goalie", f"goalie:{gid}:{G['ga']}", self._txt("goalie", goalie=book.short(gid), svp=svp, sa=G["sa"], gsax=f"{G['gsax']:+.1f}"),
                              [self._act("timeout", "timeout", {})]))
            elif G["side"] == opp and G["sv_pct"] < 0.85:
                cands.append((1, "goalie", f"goalie_opp:{gid}:{G['ga']}", self._txt("goalie_opp", goalie=book.short(gid), svp=svp, sa=G["sa"]),
                              [self._act("shoot", "tactics", {"shot_selection": "aggressive"})]))
        T, O = self.team[user], self.team[opp]
        if T["pp_o"] >= 2 and T["pp_g"] == 0 and self.pp_shots[user] <= T["pp_o"]:
            cands.append((2, "pp_unit", f"pp:{T['pp_o']}", self._txt("pp_unit", ppg=T["pp_g"], ppo=T["pp_o"], pps=self.pp_shots[user], pptoi=_clock(self.pp_toi[user])),
                          [self._act("pp_aggr", "tactics", {"pp_style": "aggressive"})]))
        if O["pp_g"] >= 2:
            cands.append((2, "pk_unit", f"pk:{O['pp_g']}", self._txt("pk_unit", ppg=O["pp_g"], ppo=O["pp_o"]),
                          [self._act("pk_pass", "tactics", {"pk_style": "passive"})]))
        if len(self.att_sides) >= 8 and run >= 6:
            acts = [self._act("defend", "tactics", {"defense": "passive"})]
            if not loop.timeout_used:
                acts.insert(0, self._act("timeout", "timeout", {}))
            cands.append((2, "momentum", f"mom:{int(gc // 240)}", self._txt("momentum", opp=team_say(opp), run=run, n=8), acts))
        pim = int(loop.state.stats[user]["pim"])
        if pim >= 8:
            cands.append((1, "penalty_trouble", f"pim:{pim // 4}", self._txt("penalty_trouble", pim=pim),
                          [self._act("calm", "tactics", {"forecheck": "passive"})]))
        cands.sort(key=lambda c: -c[0])
        for prio, kind, key, (title, body), acts in cands:
            if key in self._seen:
                continue
            if gc - self._cool.get(kind, -1e9) < (90.0 if prio >= 3 else 240.0):
                continue
            if prio < 3 and gc - self._last_new_gc < 60.0:
                continue
            self._seen.add(key)
            self._cool[kind] = gc
            self._last_new_gc = gc
            self._nid += 1
            self.notes.append({"id": f"n{self._nid}", "pt": round(pt, 2), "t": round(gc, 2), "kind": kind, "priority": prio,
                               "title": title, "body": body, "actions": [a for a in acts if a][:3],
                               "expires_pt": round(pt + (60.0 if prio >= 3 else 120.0), 2), "status": "new"})
            break
        self.notes = self.notes[-40:]
        self._expire(pt)

    def _expire(self, pt: float) -> None:
        for n in self.notes:
            if n["status"] == "new" and pt > float(n["expires_pt"]):
                n["status"] = "expired"

    def ack(self, nid: str, status: str) -> Optional[Dict[str, Any]]:
        for n in self.notes:
            if n["id"] == nid:
                n["status"] = status if status in ("acted", "dismissed") else "dismissed"
                return dict(n)
        return None
