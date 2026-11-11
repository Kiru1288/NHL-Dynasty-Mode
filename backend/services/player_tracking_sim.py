"""Live Sim (SimTracker): registry, instrumented SimEngine pass, live loop, coaches, API router.

One authoritative engine pass (same seed as replays) is captured through hooks; events are
stepped to the client on the engine's own game clock. Coach orders re-run the engine with
the order gated to unplayed attempts and splice the new future onto what was already shown.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import sys
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import APIRouter, Header, HTTPException
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore

from services.franchise_store import get_session

logger = logging.getLogger(__name__)

# ═══ 0. Imports, constants, feature flags

LIVE_SIM_ENABLED = os.environ.get("LIVE_SIM_ENABLED", "1") != "0"
LIVE_TRACKING_ENABLED = True
LIVE_COMMENTARY_ENABLED = True
LIVE_TTS_ENABLED = os.environ.get("LIVE_TTS_ENABLED", "0") == "1"
LIVE_TTS_WORKER_URL = os.environ.get("LIVE_TTS_WORKER_URL", "http://127.0.0.1:8765")
LIVE_MAX_GAMES = int(os.environ.get("LIVE_MAX_GAMES", "8"))
LIVE_EVENT_BUFFER = 600
LIVE_IDLE_EXPIRE_SEC = 3 * 3600
CROWD_HOME_LIFT_ENABLED = False

TTS_QA_CORPUS: List[Dict[str, str]] = [
    {"category": "goal", "profile": "hype", "text": "He scores!", "expected_speak": "He scores"},
    {"category": "pp", "profile": "calm", "text": "PP goal — 3–2", "expected_speak": "power play goal, 3 2"},
]

# Kiru product decisions (§18) as wired defaults for later stages:
COMMENTATOR_PERSONA = os.environ.get("LIVE_COMMENTATOR_PERSONA", "The Booth")  # config label only; original lines, no impersonation
TTS_VOICE_MODE = "stock"  # stock voices only (browser speechSynthesis, or local kokoro/pyttsx3); no voice cloning
SHOOTOUT_MODEL = "breakaway"  # D4: round-by-round breakaway simulator (Stage 9)
PENALTY_LABEL_MODE = "duration_only"  # engine has duration, not infraction type
PRONUNCIATION_STORE = "dict"  # D6: in-module dict until data file is approved
USE_SOUNDTRACK_FOLDER_AUDIO = True  # D7: optional UI stings from frontend/soundtrack

_OPEN_STATUSES = frozenset({"pregame", "live", "intermission", "overtime", "shootout", "paused", "final"})

_REGULATION_PERIOD_SEC = 1200.0
_REGULATION_GAME_SEC = 3600.0

_registry_lock = threading.Lock()
_LIVE: Dict[str, "LiveGame"] = {}


# ═══ 1. Data contracts

def _period_label(period: int, *, shootout: bool = False, ot_period: int = 0) -> str:
    if shootout:
        return "SO"
    if period > 3:
        return "OT" if ot_period <= 1 else f"{ot_period}OT"
    return {1: "1ST", 2: "2ND", 3: "3RD"}.get(period, f"P{period}")


def _clock_display(sec_left_period: float) -> str:
    """MM:SS time left in the period (contract format)."""
    sec = max(0, int(math.ceil(float(sec_left_period) - 1e-6)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ═══ 2. Registry

@dataclass
class LiveGame:
    live_id: str
    session_id: str
    game_key: Dict[str, Any]
    status: str = "pregame"
    seq: int = 0
    game_sec_elapsed: float = 0.0
    period: int = 1
    score_home: int = 0
    score_away: int = 0
    shots_home: int = 0
    shots_away: int = 0
    xg_home: float = 0.0
    xg_away: float = 0.0
    seed: int = 0
    user_side: str = "home"
    home_id: str = ""
    away_id: str = ""
    is_playoff: bool = False
    calendar_day: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Filled at start; loop uses these in Stage 2+
    home: Any = None
    away: Any = None
    sim: Any = None
    scratch: Dict[str, Any] = field(default_factory=dict)
    strength: Dict[str, float] = field(default_factory=dict)
    event_ring: List[Dict[str, Any]] = field(default_factory=list)
    loop: Any = None
    paused: bool = False
    speed: int = 1
    coach_orders: List[Dict[str, Any]] = field(default_factory=list)
    captions: List[Dict[str, Any]] = field(default_factory=list)
    alerts_buf: List[Dict[str, Any]] = field(default_factory=list)
    tracking_frames: List[Dict[str, Any]] = field(default_factory=list)
    lang: str = "en"


def has_open_live_game(session_id: str) -> Optional[str]:
    """Return live_id when a non-terminal live game exists for this session."""
    if not session_id:
        return None
    with _registry_lock:
        lg = _LIVE.get(str(session_id))
        if lg is None:
            return None
        if lg.status in ("committed", "aborted"):
            return None
        _touch(lg)
        return lg.live_id


def _touch(lg: LiveGame) -> None:
    lg.last_access = time.time()


def _evict_stale_unlocked() -> None:
    now = time.time()
    with _registry_lock:
        drop = [
            sid
            for sid, lg in _LIVE.items()
            if lg.status in ("committed", "aborted")
            or (now - lg.last_access) > LIVE_IDLE_EXPIRE_SEC
        ]
        # Leave room for the game about to be inserted (LRU among the survivors).
        survivors = sorted((kv for kv in _LIVE.items() if kv[0] not in drop), key=lambda kv: kv[1].last_access)
        excess = len(survivors) - max(0, LIVE_MAX_GAMES - 1)
        for sid, _ in survivors[: max(0, excess)]:
            drop.append(sid)
        for sid in drop:
            _LIVE.pop(sid, None)


def _find_user_slot(session: Any) -> Optional[Any]:
    from app.sim_engine.league.schedule_generator import _safe_slot_team_id

    uid = str(getattr(session, "user_team_id", "") or "")
    day = int(getattr(session, "calendar_cursor", 0) or 0)
    slots = list((getattr(session, "by_day", None) or {}).get(day) or [])
    for slot in slots:
        hid = _safe_slot_team_id(slot, "home_id")
        aid = _safe_slot_team_id(slot, "away_id")
        if uid in (hid, aid):
            return slot
    return None


def _read_strength_scales(session: Any, home: Any, away: Any, hid: str, aid: str, day: int, is_playoff: bool) -> Tuple[float, float, float]:
    """Mirror slot strength/scale reads without mutating session world state."""
    sim = session.sim
    try:
        from app.sim_engine.world import chemistry as world_chemistry
        from app.sim_engine.world import fatigue as world_fatigue
        from app.sim_engine.world import morale as world_morale
        from app.sim_engine.world import momentum as world_momentum
    except Exception:
        return 1.0, 1.0, 1.0

    h_scale = 1.0
    a_scale = 1.0
    noise_scale = 1.0
    if getattr(session, "use_world", False) and world_momentum is not None:
        hm = world_momentum.team_strength_modifier(home)
        am = world_momentum.team_strength_modifier(away)
        hc = world_chemistry.team_strength_modifier(home)
        ac = world_chemistry.team_strength_modifier(away)
        hf = world_fatigue.team_fatigue_strength_factor(home)
        af = world_fatigue.team_fatigue_strength_factor(away)
        hmr = world_morale.team_morale_strength_factor(home)
        amr = world_morale.team_morale_strength_factor(away)
        h_scale = max(0.93, min(1.07, hm * hc * hf * hmr)) * float(sim._roster_injury_depth_penalty(home))
        a_scale = max(0.93, min(1.07, am * ac * af * amr)) * float(sim._roster_injury_depth_penalty(away))
        try:
            from app.sim_engine.franchise.storyline_coverage import apply_matchup_to_scales

            h_scale, a_scale, _ = apply_matchup_to_scales(
                session,
                hid,
                aid,
                h_scale,
                a_scale,
                {"game_id": f"{hid}_{aid}_{day}", "is_playoff": is_playoff},
            )
        except Exception:
            logger.debug("apply_matchup_to_scales failed", exc_info=True)
        base_noise = 1.0 + 0.22 * (float(getattr(session, "chaos_index", 0.5) or 0.5) - 0.5)
        _, ih = sim._identity_runner_strength_noise_factors(home)
        _, ia = sim._identity_runner_strength_noise_factors(away)
        noise_scale = 0.5 * base_noise * (0.5 * (ih + ia))
    return float(h_scale), float(a_scale), float(noise_scale)


def _pid(p: Any) -> str:
    from app.sim_engine.engine import _id_str

    return _id_str(p, "id")


def _penalty_duration_label(dur_sec: float) -> str:
    """Engine provides duration only — no infraction type."""
    mins = int(round(float(dur_sec) / 60.0))
    if mins >= 5:
        return "5-minute major"
    if mins >= 4:
        return "double minor (4 min)"
    return "2-minute minor"


# ═══ 3. Localization — original wording only (no catchphrases, no impersonation)

LIVE_LANGUAGES: List[Dict[str, str]] = [
    {"code": "en", "label": "English"},
    {"code": "fr", "label": "Français"},
    {"code": "es", "label": "Español"},
    {"code": "de", "label": "Deutsch"},
    {"code": "sv", "label": "Svenska"},
]
_LANG_CODES = tuple(x["code"] for x in LIVE_LANGUAGES)


def _norm_lang(v: Any) -> str:
    s = str(v or "").strip().lower()[:2]
    return s if s in _LANG_CODES else "en"


_NUM_WORDS: Dict[str, List[str]] = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"],
    "fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf", "vingt"],
    "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte"],
    "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn", "zwanzig"],
    "sv": ["noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio", "elva", "tolv", "tretton", "fjorton", "femton", "sexton", "sjutton", "arton", "nitton", "tjugo"],
}
_SCORE_JOIN_TTS = {"en": "-", "fr": " à ", "es": " a ", "de": " zu ", "sv": " "}
_PP_WORD = {"en": "power play", "fr": "avantage numérique", "es": "power play", "de": "Überzahl", "sv": "powerplay"}
_SOG_WORD = {"en": "shots on goal", "fr": "tirs au but", "es": "tiros a puerta", "de": "Torschüsse", "sv": "skott på mål"}


def _num_word(n: int, lang: str) -> str:
    words = _NUM_WORDS.get(lang) or _NUM_WORDS["en"]
    return words[n] if 0 <= int(n) < len(words) else str(n)


def _score_text(a: int, b: int) -> str:
    return f"{int(a)}–{int(b)}"


def _score_tts(a: int, b: int, lang: str) -> str:
    return f"{_num_word(a, lang)}{_SCORE_JOIN_TTS.get(lang, ' to ')}{_num_word(b, lang)}"


# Period names with article, lower case (capitalised by _cap when a sentence starts with them).
_PERIOD_NAME: Dict[str, Dict[int, str]] = {
    "en": {1: "the 1st period", 2: "the 2nd period", 3: "the 3rd period", 4: "overtime", 5: "the shootout"},
    "fr": {1: "la 1re période", 2: "la 2e période", 3: "la 3e période", 4: "la prolongation", 5: "la fusillade"},
    "es": {1: "el primer periodo", 2: "el segundo periodo", 3: "el tercer periodo", 4: "la prórroga", 5: "la tanda de penaltis"},
    "de": {1: "das 1. Drittel", 2: "das 2. Drittel", 3: "das 3. Drittel", 4: "die Verlängerung", 5: "das Penaltyschießen"},
    "sv": {1: "första perioden", 2: "andra perioden", 3: "tredje perioden", 4: "förlängningen", 5: "straffläggningen"},
}

# Contract chance buckets -> noun phrase used in play-by-play.
_CHANCE_PHRASE: Dict[str, Dict[str, str]] = {
    "en": {"high_danger": "shot from in tight", "slot": "shot from the slot", "rebound": "rebound try", "point": "point shot",
           "perimeter": "shot from the outside", "breakaway": "shot on the rush", "": "shot"},
    "fr": {"high_danger": "tir de près", "slot": "tir de l'enclave", "rebound": "tir sur le retour", "point": "tir de la pointe",
           "perimeter": "tir de l'extérieur", "breakaway": "tir en contre-attaque", "": "tir"},
    "es": {"high_danger": "tiro desde muy cerca", "slot": "tiro desde el slot", "rebound": "tiro de rebote", "point": "tiro desde la línea azul",
           "perimeter": "tiro desde fuera", "breakaway": "tiro al contraataque", "": "tiro"},
    "de": {"high_danger": "Schuss aus kurzer Distanz", "slot": "Schuss aus dem Slot", "rebound": "Nachschuss", "point": "Schuss von der blauen Linie",
           "perimeter": "Schuss von außen", "breakaway": "Schuss im Konter", "": "Schuss"},
    "sv": {"high_danger": "skott från nära håll", "slot": "skott från slottet", "rebound": "returskott", "point": "skott från blå linjen",
           "perimeter": "skott från utsidan", "breakaway": "skott i kontring", "": "skott"},
}

_STRENGTH_SFX: Dict[str, Dict[str, str]] = {
    "en": {"PP": " on the power play", "SH": " shorthanded", "EN": " into the empty net", "4v4": " at 4-on-4", "3v3": " at 3-on-3"},
    "fr": {"PP": " en avantage numérique", "SH": " en désavantage numérique", "EN": " dans un filet désert", "4v4": " à 4 contre 4", "3v3": " à 3 contre 3"},
    "es": {"PP": " en power play", "SH": " en inferioridad numérica", "EN": " a portería vacía", "4v4": " en 4 contra 4", "3v3": " en 3 contra 3"},
    "de": {"PP": " in Überzahl", "SH": " in Unterzahl", "EN": " ins leere Tor", "4v4": " bei 4 gegen 4", "3v3": " bei 3 gegen 3"},
    "sv": {"PP": " i powerplay", "SH": " i boxplay", "EN": " i tom kasse", "4v4": " i fyra mot fyra", "3v3": " i tre mot tre"},
}

# Engine gives duration only (PENALTY_LABEL_MODE) — never an invented infraction.
_PEN_LABEL: Dict[str, Dict[int, str]] = {
    "en": {2: "2-minute minor", 4: "double minor", 5: "5-minute major"},
    "fr": {2: "pénalité mineure de 2 minutes", 4: "double mineure", 5: "majeure de 5 minutes"},
    "es": {2: "menor de 2 minutos", 4: "doble menor", 5: "mayor de 5 minutos"},
    "de": {2: "2-Minuten-Strafe", 4: "2+2 Minuten", 5: "5-Minuten-Strafe"},
    "sv": {2: "2 minuter", 4: "2+2 minuter", 5: "5 minuter"},
}

_DESC: Dict[str, Dict[str, str]] = {
    "en": {
        "saved": "{shooter}, {chance} — saved by {goalie}", "missed": "{shooter}, {chance} — wide of the net",
        "blocked": "{shooter}, {chance} — blocked by {blocker}", "en_kept_out": "{shooter}, {chance} at the empty net — kept out",
        "goal": "GOAL {team} — {shooter} ({assists}), {chance}{sfx}. {score_line}", "assists": "assists: {names}", "unassisted": "unassisted",
        "penalty": "Penalty {team} — {player}, {label}", "penalty_end": "{team} back to full strength",
        "penalty_end_goal": "{team} back to full strength after the power-play goal",
        "period_start": "Start of {period}", "period_end": "End of {period}",
        "faceoff": "Faceoff at center ice", "faceoff_goal": "Faceoff at center ice after the goal",
        "so_goal": "Shootout round {round}: {shooter} scores on {goalie}", "so_miss": "Shootout round {round}: {shooter} stopped by {goalie}",
        "final": "Final — {score_line}{dec}", "coach": "{team} bench: {what}",
        "dec_OT": " (OT)", "dec_SO": " (SO)",
    },
    "fr": {
        "saved": "{shooter}, {chance} — arrêt de {goalie}", "missed": "{shooter}, {chance} — à côté du filet",
        "blocked": "{shooter}, {chance} — bloqué par {blocker}", "en_kept_out": "{shooter}, {chance} vers le filet désert — repoussé",
        "goal": "BUT {team} — {shooter} ({assists}), {chance}{sfx}. {score_line}", "assists": "aides : {names}", "unassisted": "sans aide",
        "penalty": "Pénalité {team} — {player}, {label}", "penalty_end": "{team} de retour à forces égales",
        "penalty_end_goal": "{team} de retour à forces égales après le but en avantage numérique",
        "period_start": "Début de {period}", "period_end": "Fin de {period}",
        "faceoff": "Mise au jeu au centre", "faceoff_goal": "Mise au jeu au centre après le but",
        "so_goal": "Fusillade, ronde {round} : {shooter} déjoue {goalie}", "so_miss": "Fusillade, ronde {round} : {shooter} arrêté par {goalie}",
        "final": "Final — {score_line}{dec}", "coach": "Banc de {team} : {what}",
        "dec_OT": " (prol.)", "dec_SO": " (fusillade)",
    },
    "es": {
        "saved": "{shooter}, {chance} — atajada de {goalie}", "missed": "{shooter}, {chance} — desviado",
        "blocked": "{shooter}, {chance} — bloqueado por {blocker}", "en_kept_out": "{shooter}, {chance} a portería vacía — no entra",
        "goal": "GOL {team} — {shooter} ({assists}), {chance}{sfx}. {score_line}", "assists": "asistencias: {names}", "unassisted": "sin asistencia",
        "penalty": "Penalización {team} — {player}, {label}", "penalty_end": "{team} vuelve a estar completo",
        "penalty_end_goal": "{team} vuelve a estar completo tras el gol en power play",
        "period_start": "Comienza {period}", "period_end": "Termina {period}",
        "faceoff": "Saque neutral en el centro", "faceoff_goal": "Saque neutral en el centro tras el gol",
        "so_goal": "Tanda de penaltis, ronda {round}: {shooter} bate a {goalie}", "so_miss": "Tanda de penaltis, ronda {round}: {goalie} detiene a {shooter}",
        "final": "Final — {score_line}{dec}", "coach": "Banquillo de {team}: {what}",
        "dec_OT": " (prórroga)", "dec_SO": " (penaltis)",
    },
    "de": {
        "saved": "{shooter}, {chance} — Parade von {goalie}", "missed": "{shooter}, {chance} — am Tor vorbei",
        "blocked": "{shooter}, {chance} — geblockt von {blocker}", "en_kept_out": "{shooter}, {chance} aufs leere Tor — nicht drin",
        "goal": "TOR {team} — {shooter} ({assists}), {chance}{sfx}. {score_line}", "assists": "Vorlagen: {names}", "unassisted": "ohne Vorlage",
        "penalty": "Strafe {team} — {player}, {label}", "penalty_end": "{team} wieder vollzählig",
        "penalty_end_goal": "{team} nach dem Überzahltor wieder vollzählig",
        "period_start": "{Period} beginnt", "period_end": "{Period} ist beendet",
        "faceoff": "Bully am Mittelpunkt", "faceoff_goal": "Bully am Mittelpunkt nach dem Tor",
        "so_goal": "Penaltyschießen, Runde {round}: {shooter} trifft gegen {goalie}", "so_miss": "Penaltyschießen, Runde {round}: {goalie} pariert gegen {shooter}",
        "final": "Endstand — {score_line}{dec}", "coach": "Bank {team}: {what}",
        "dec_OT": " (n.V.)", "dec_SO": " (n.P.)",
    },
    "sv": {
        "saved": "{shooter}, {chance} — räddning av {goalie}", "missed": "{shooter}, {chance} — utanför mål",
        "blocked": "{shooter}, {chance} — blockerat av {blocker}", "en_kept_out": "{shooter}, {chance} mot tom kasse — går inte in",
        "goal": "MÅL {team} — {shooter} ({assists}), {chance}{sfx}. {score_line}", "assists": "assist: {names}", "unassisted": "utan assist",
        "penalty": "Utvisning {team} — {player}, {label}", "penalty_end": "{team} fulltaliga igen",
        "penalty_end_goal": "{team} fulltaliga igen efter powerplaymålet",
        "period_start": "{Period} börjar", "period_end": "{Period} är slut",
        "faceoff": "Tekning vid mittcirkeln", "faceoff_goal": "Tekning vid mittcirkeln efter målet",
        "so_goal": "Straffläggning, omgång {round}: {shooter} sätter den på {goalie}", "so_miss": "Straffläggning, omgång {round}: {goalie} räddar från {shooter}",
        "final": "Slut — {score_line}{dec}", "coach": "Bänken hos {team}: {what}",
        "dec_OT": " (förl.)", "dec_SO": " (straffar)",
    },
}

# Two-person booth. "pbp" = play-by-play (factual, present tense); "color" = analyst lines that
# are only used when the stat they quote is real (computed from emitted events).
COMMENTARY_I18N: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "faceoff_open": ["Puck is dropped, and we're underway.", "Here's the opening faceoff."],
        "period_start": ["{Period} is underway.", "Puck drop to start {period}."],
        "period_end": ["That's the end of {period}.", "The horn sounds to end {period}."],
        "save": ["{shooter} with a {chance}, {goalie} makes the save.", "{goalie} turns it aside.", "Shot by {shooter}, saved by {goalie}.", "{goalie} with the stop on {shooter}.", "{shooter} tests {goalie}, who holds on."],
        "save_big": ["Big save by {goalie} on {shooter}.", "{goalie} gets across and denies {shooter}.", "{shooter} in close, and {goalie} shuts the door."],
        "miss": ["{shooter} misses the net.", "{shooter} fires wide.", "Off target from {shooter}.", "{shooter} sends it wide of the net."],
        "miss_hd": ["{shooter} in alone, and he can't hit the net.", "Great look for {shooter}, just wide."],
        "block": ["{blocker} blocks the shot.", "{shooter}'s shot is blocked by {blocker}.", "Shot from {shooter} doesn't get through, {blocker} gets a piece."],
        "en_miss": ["{shooter} tries for the empty net, and it stays out."],
        "goal": ["Scores! {shooter}, {from_where}, and {score_phrase}.", "{shooter} scores {from_where}. {Score_phrase}.", "In it goes, {shooter}. {Score_phrase}."],
        "goal_pp": ["Power-play goal, {shooter}. {Score_phrase}.", "{shooter} converts on the power play. {Score_phrase}."],
        "goal_sh": ["Shorthanded goal, {shooter}. {Score_phrase}."],
        "goal_en": ["{shooter} into the empty net. {Score_phrase}."],
        "score_lead": ["{team} leads {score}"],
        "score_ahead": ["{team} takes the lead, {score}"],
        "score_within": ["{team} pulls within one, {score}"],
        "score_cut": ["{team} cuts the lead to {score}"],
        "score_tie": ["we're tied at {tie}"],
        "penalty": ["{player} goes to the box, {label}.", "Penalty to {player}, {label}."],
        "pp_start": ["{team} will go to the power play.", "{team} on the power play."],
        "pk_kill": ["{team} kills off the penalty.", "Back to even strength."],
        "coach_pull": ["{team} pulls the goalie for an extra attacker.", "{team}'s net is empty, six attackers on."],
        "coach_return": ["The goalie is back in for {team}."],
        "coach_timeout": ["{team} calls timeout."],
        "coach_change": ["Change on the {team} bench: {what}."],
        "so_start": ["We'll need a shootout."],
        "so_goal": ["{shooter} scores in the shootout.", "{shooter} beats {goalie}."],
        "so_save": ["{goalie} stops {shooter}.", "{shooter} is denied by {goalie}."],
        "final": ["That's the final. {team} wins it, {score}.", "Final horn. {team} takes it, {score}."],
        "late_close": ["Under two minutes left in a one-goal game."],
        "late_tied": ["Two minutes to go, and it's still tied."],
        "fill_oz": ["{team} cycling in the {opp} zone.", "{team} with sustained pressure down low.", "{team} keeping the puck in the {opp} end."],
        "fill_nz": ["{team} moving it up through the neutral zone.", "{team} regrouping in the middle.", "{team} looking for room through center ice."],
        "fill_dz": ["{team} working it out of its own end.", "{team} trying to break out.", "{team} sorting it out behind its own net."],
        "fill_stop": ["Whistle, and we'll have a faceoff."],
        "c_shots": ["That's his {nth} shot on goal tonight.", "{Nth} shot on goal tonight for {player}."],
        "c_share": ["{team} has had {n} of the last {m} attempts.", "{n} of the last {m} attempts belong to {team}."],
        "c_goalie": ["{goalie} has stopped {sv} of {sa} tonight.", "{goalie} is {sv} for {sa} so far."],
        "c_pp": ["{team}'s power play tonight: {g} for {o}."],
        "c_xg": ["Expected goals: {team} {a}, {opp} {b}.", "On expected goals it's {team} {a}, {opp} {b}."],
        "c_run": ["That's {n} straight goals for {team}."],
        "c_points": ["{player} has {pts} points tonight.", "A {pts}-point night so far for {player}."],
        "c_period_shots": ["Shots in the period: {team} {a}, {opp} {b}."],
        "c_first": ["First goal of the night.", "That opens the scoring."],
        "c_pk": ["{team} has killed {k} of {n} tonight."],
        "c_trail": ["{team} down {n} with {time} to play."],
    },
    "fr": {
        "faceoff_open": ["La rondelle est mise au jeu, c'est parti.", "Voici la mise au jeu initiale."],
        "period_start": ["Début de {period}.", "On lance {period}."],
        "period_end": ["C'est la fin de {period}.", "La sirène met fin à {period}."],
        "save": ["{shooter} décoche un {chance}, arrêt de {goalie}.", "{goalie} fait l'arrêt.", "Lancer de {shooter}, {goalie} bloque.", "{goalie} repousse le tir de {shooter}.", "{shooter} met {goalie} à l'épreuve, il ne cède pas."],
        "save_big": ["Gros arrêt de {goalie} devant {shooter}.", "{goalie} se déplace et frustre {shooter}.", "{shooter} seul en face, et {goalie} ferme la porte."],
        "miss": ["{shooter} rate le filet.", "Le lancer de {shooter} passe à côté.", "{shooter} vise à côté.", "{shooter} envoie la rondelle à côté du filet."],
        "miss_hd": ["{shooter} seul devant, mais il rate la cible.", "Belle occasion pour {shooter}, juste à côté."],
        "block": ["{blocker} bloque le lancer.", "Le tir de {shooter} est bloqué par {blocker}.", "Le tir de {shooter} ne passe pas, {blocker} s'interpose."],
        "en_miss": ["{shooter} vise le filet désert, mais rate."],
        "goal": ["Il compte! {shooter}, {from_where}, et {score_phrase}.", "But de {shooter} {from_where}. {Score_phrase}.", "C'est dedans, {shooter}. {Score_phrase}."],
        "goal_pp": ["But en avantage numérique, {shooter}. {Score_phrase}.", "{shooter} profite de l'avantage numérique. {Score_phrase}."],
        "goal_sh": ["But en désavantage numérique, {shooter}. {Score_phrase}."],
        "goal_en": ["{shooter} dans le filet désert. {Score_phrase}."],
        "score_lead": ["{team} mène {score}"],
        "score_ahead": ["{team} prend les devants {score}"],
        "score_within": ["{team} revient à un but, {score}"],
        "score_cut": ["{team} réduit l'écart, {score}"],
        "score_tie": ["c'est égal {score}"],
        "penalty": ["{player} écope d'une {label}.", "Pénalité à {player}, {label}."],
        "pp_start": ["{team} profitera d'un avantage numérique.", "{team} en avantage numérique."],
        "pk_kill": ["{team} écoule la pénalité.", "Retour à forces égales."],
        "coach_pull": ["{team} retire son gardien pour un sixième attaquant.", "Filet désert pour {team}, six attaquants sur la glace."],
        "coach_return": ["Le gardien de {team} reprend sa place."],
        "coach_timeout": ["{team} demande son temps d'arrêt."],
        "coach_change": ["Changement derrière le banc de {team} : {what}."],
        "so_start": ["Il faudra une fusillade."],
        "so_goal": ["{shooter} marque en fusillade.", "{shooter} déjoue {goalie}."],
        "so_save": ["{goalie} stoppe {shooter}.", "{shooter} est frustré par {goalie}."],
        "final": ["C'est terminé. Victoire de {team}, {score}.", "Fin du match : {team} l'emporte {score}."],
        "late_close": ["Moins de deux minutes à jouer, un but d'écart."],
        "late_tied": ["Deux minutes à faire, toujours égal."],
        "fill_oz": ["{team} fait tourner la rondelle dans la zone de {opp}.", "{team} maintient la pression en zone adverse.", "{team} garde la rondelle dans le territoire de {opp}."],
        "fill_nz": ["{team} remonte la rondelle en zone neutre.", "{team} se replie en zone neutre.", "{team} cherche de l'espace en zone neutre."],
        "fill_dz": ["{team} tente de sortir de sa zone.", "{team} relance depuis sa zone.", "{team} s'organise derrière son filet."],
        "fill_stop": ["Coup de sifflet, mise au jeu à venir."],
        "c_shots": ["C'est son {nth} tir cadré ce soir.", "{Nth} tir cadré de la soirée pour {player}."],
        "c_share": ["{team} a obtenu {n} des {m} dernières tentatives.", "{n} des {m} dernières tentatives pour {team}."],
        "c_goalie": ["{goalie} a stoppé {sv} des {sa} tirs ce soir.", "{goalie} : {sv} arrêts sur {sa} tirs jusqu'ici."],
        "c_pp": ["Avantage numérique ce soir pour {team} : {g} en {o}."],
        "c_xg": ["Buts attendus : {team} {a}, {opp} {b}."],
        "c_run": ["{n} buts de suite pour {team}."],
        "c_points": ["{player} en est à {pts} points ce soir."],
        "c_period_shots": ["Tirs au but dans la période : {team} {a}, {opp} {b}."],
        "c_first": ["Premier but de la soirée.", "Voilà la marque ouverte."],
        "c_pk": ["{team} a écoulé {k} pénalités sur {n} ce soir."],
        "c_trail": ["{team} tire de l'arrière par {n}, il reste {time} à jouer."],
    },
    "es": {
        "faceoff_open": ["Cae el disco y arranca el partido.", "Saque neutral inicial."],
        "period_start": ["Arranca {period}.", "Comienza {period}."],
        "period_end": ["Termina {period}.", "Se acaba {period}."],
        "save": ["{shooter} con un {chance}, atajada de {goalie}.", "{goalie} detiene el disparo.", "Tiro de {shooter}, para {goalie}.", "{goalie} contiene el disparo de {shooter}.", "{shooter} pone a prueba a {goalie}, que responde."],
        "save_big": ["Gran atajada de {goalie} ante {shooter}.", "{goalie} se estira y le niega el gol a {shooter}.", "{shooter} solo, y {goalie} lo detiene."],
        "miss": ["{shooter} no encuentra la portería.", "Disparo desviado de {shooter}.", "{shooter} tira fuera.", "{shooter} manda el disco fuera de la portería."],
        "miss_hd": ["{shooter} solo ante el portero, y la manda fuera.", "Ocasión clara para {shooter}, por poco."],
        "block": ["{blocker} bloquea el tiro.", "El disparo de {shooter} lo bloquea {blocker}.", "El tiro de {shooter} no pasa, {blocker} lo intercepta."],
        "en_miss": ["{shooter} busca la portería vacía, pero no entra."],
        "goal": ["¡Gol! {shooter}, {from_where}, y {score_phrase}.", "Marca {shooter} {from_where}. {Score_phrase}.", "Adentro, {shooter}. {Score_phrase}."],
        "goal_pp": ["Gol en power play de {shooter}. {Score_phrase}.", "{shooter} aprovecha la superioridad. {Score_phrase}."],
        "goal_sh": ["Gol en inferioridad numérica, {shooter}. {Score_phrase}."],
        "goal_en": ["{shooter} a portería vacía. {Score_phrase}."],
        "score_lead": ["{team} gana {score}"],
        "score_ahead": ["{team} se pone por delante, {score}"],
        "score_within": ["{team} se acerca a un gol, {score}"],
        "score_cut": ["{team} recorta distancias, {score}"],
        "score_tie": ["empate a {tie}"],
        "penalty": ["{player} va al banquillo de castigo, {label}.", "Penalización a {player}, {label}."],
        "pp_start": ["{team} tendrá superioridad numérica.", "{team} en power play."],
        "pk_kill": ["{team} supera la inferioridad numérica.", "De vuelta a cinco contra cinco."],
        "coach_pull": ["{team} retira al portero para sumar un atacante.", "Portería vacía para {team}, seis atacantes en pista."],
        "coach_return": ["Vuelve el portero de {team}."],
        "coach_timeout": ["{team} pide tiempo muerto."],
        "coach_change": ["Ajuste en el banquillo de {team}: {what}."],
        "so_start": ["Habrá tanda de penaltis."],
        "so_goal": ["{shooter} marca en la tanda.", "{shooter} bate a {goalie}."],
        "so_save": ["{goalie} detiene a {shooter}.", "{goalie} gana el duelo a {shooter}."],
        "final": ["Final del partido. Gana {team}, {score}.", "Se acabó: {team} se impone {score}."],
        "late_close": ["Menos de dos minutos y un gol de diferencia."],
        "late_tied": ["Dos minutos por jugar y sigue el empate."],
        "fill_oz": ["{team} mueve el disco en la zona de {opp}.", "{team} mantiene la presión en zona rival.", "{team} mantiene el disco en campo de {opp}."],
        "fill_nz": ["{team} avanza por la zona neutral.", "{team} reorganiza en la zona neutral.", "{team} busca espacios por el centro."],
        "fill_dz": ["{team} intenta salir de su zona.", "{team} construye desde atrás.", "{team} se organiza detrás de su portería."],
        "fill_stop": ["Silbato, habrá saque neutral."],
        "c_shots": ["Es su {nth} tiro a puerta esta noche."],
        "c_share": ["{team} suma {n} de los últimos {m} intentos.", "{n} de los últimos {m} intentos son de {team}."],
        "c_goalie": ["{goalie} ha detenido {sv} de {sa} tiros esta noche.", "{goalie} lleva {sv} paradas en {sa} tiros."],
        "c_pp": ["Power play de {team} esta noche: {g} de {o}."],
        "c_xg": ["Goles esperados: {team} {a}, {opp} {b}."],
        "c_run": ["{n} goles seguidos de {team}."],
        "c_points": ["{player} ya suma {pts} puntos esta noche."],
        "c_period_shots": ["Tiros a puerta en el periodo: {team} {a}, {opp} {b}."],
        "c_first": ["Primer gol de la noche.", "Se abre el marcador."],
        "c_pk": ["{team} ha superado {k} de {n} inferioridades esta noche."],
        "c_trail": ["{team} pierde por {n} con {time} por jugar."],
    },
    "de": {
        "faceoff_open": ["Der Puck ist eingeworfen, das Spiel läuft.", "Da ist das Eröffnungsbully."],
        "period_start": ["{Period} beginnt.", "{Period} läuft."],
        "period_end": ["{Period} ist vorbei.", "Die Sirene beendet {period}."],
        "save": ["{shooter} mit dem {chance}, {goalie} hält.", "{goalie} pariert.", "Schuss von {shooter}, gehalten von {goalie}.", "{goalie} entschärft den Schuss von {shooter}.", "{shooter} prüft {goalie}, der sicher bleibt."],
        "save_big": ["Starke Parade von {goalie} gegen {shooter}.", "{goalie} ist rechtzeitig unten und stoppt {shooter}.", "{shooter} frei vor dem Tor, aber {goalie} bleibt Sieger."],
        "miss": ["{shooter} verfehlt das Tor.", "{shooter} schießt vorbei.", "Daneben von {shooter}.", "{shooter} setzt die Scheibe neben das Tor."],
        "miss_hd": ["{shooter} ganz frei, aber er trifft das Tor nicht.", "Gute Chance für {shooter}, knapp vorbei."],
        "block": ["{blocker} blockt den Schuss.", "Der Schuss von {shooter} wird von {blocker} geblockt.", "Der Schuss von {shooter} kommt nicht durch, {blocker} ist dazwischen."],
        "en_miss": ["{shooter} zielt aufs leere Tor, aber daneben."],
        "goal": ["Tor! {shooter}, {from_where}, und {score_phrase}.", "{shooter} trifft {from_where}. {Score_phrase}.", "Drin, {shooter}. {Score_phrase}."],
        "goal_pp": ["Überzahltor durch {shooter}. {Score_phrase}.", "{shooter} nutzt die Überzahl. {Score_phrase}."],
        "goal_sh": ["Unterzahltor durch {shooter}. {Score_phrase}."],
        "goal_en": ["{shooter} ins leere Tor. {Score_phrase}."],
        "score_lead": ["{team} führt {score}"],
        "score_ahead": ["{team} geht in Führung, {score}"],
        "score_within": ["{team} verkürzt auf ein Tor, {score}"],
        "score_cut": ["{team} verkürzt auf {score}"],
        "score_tie": ["es steht {score}"],
        "penalty": ["{player} muss auf die Strafbank, {label}.", "Strafe gegen {player}, {label}."],
        "pp_start": ["{team} bekommt ein Überzahlspiel.", "{team} in Überzahl."],
        "pk_kill": ["{team} übersteht die Unterzahl.", "Wieder fünf gegen fünf."],
        "coach_pull": ["{team} nimmt den Torhüter für einen sechsten Feldspieler vom Eis.", "Leeres Tor bei {team}, sechs Angreifer auf dem Eis."],
        "coach_return": ["Der Torhüter von {team} ist zurück im Tor."],
        "coach_timeout": ["{team} nimmt die Auszeit."],
        "coach_change": ["Umstellung bei {team}: {what}."],
        "so_start": ["Es geht ins Penaltyschießen."],
        "so_goal": ["{shooter} verwandelt.", "{shooter} überwindet {goalie}."],
        "so_save": ["{goalie} hält gegen {shooter}.", "{shooter} scheitert an {goalie}."],
        "final": ["Das Spiel ist aus. {team} gewinnt {score}.", "Schlusssirene: {team} siegt {score}."],
        "late_close": ["Weniger als zwei Minuten, ein Tor Unterschied."],
        "late_tied": ["Noch zwei Minuten, weiter unentschieden."],
        "fill_oz": ["{team} lässt die Scheibe im Drittel von {opp} laufen.", "{team} setzt sich im Angriffsdrittel fest.", "{team} hält die Scheibe im Drittel von {opp}."],
        "fill_nz": ["{team} baut über die neutrale Zone auf.", "{team} sortiert sich in der neutralen Zone.", "{team} sucht den Weg durch die Mitte."],
        "fill_dz": ["{team} versucht, sich zu befreien.", "{team} baut aus dem eigenen Drittel auf.", "{team} ordnet sich hinter dem eigenen Tor."],
        "fill_stop": ["Pfiff, es gibt ein Bully."],
        "c_shots": ["Das ist heute sein {nth} Torschuss."],
        "c_share": ["{team} hatte {n} der letzten {m} Abschlüsse.", "{n} der letzten {m} Abschlüsse gingen auf das Konto von {team}."],
        "c_goalie": ["{goalie} hat heute {sv} von {sa} Schüssen gehalten.", "{goalie} steht bei {sv} Paraden aus {sa} Schüssen."],
        "c_pp": ["Überzahl heute für {team}: {g} aus {o}."],
        "c_xg": ["Erwartete Tore: {team} {a}, {opp} {b}."],
        "c_run": ["{n} Tore in Folge für {team}."],
        "c_points": ["{player} steht heute schon bei {pts} Punkten."],
        "c_period_shots": ["Torschüsse im Drittel: {team} {a}, {opp} {b}."],
        "c_first": ["Das erste Tor des Abends.", "Damit ist der Bann gebrochen."],
        "c_pk": ["{team} hat heute {k} von {n} Unterzahlspielen überstanden."],
        "c_trail": ["{team} liegt {n} zurück, noch {time} zu spielen."],
    },
    "sv": {
        "faceoff_open": ["Pucken släpps och matchen är igång.", "Där är öppningstekningen."],
        "period_start": ["{Period} är igång.", "Nu startar {period}."],
        "period_end": ["{Period} är slut.", "Signalen går och {period} är över."],
        "save": ["{shooter} med ett {chance}, {goalie} räddar.", "{goalie} gör räddningen.", "Skott från {shooter}, räddning av {goalie}.", "{goalie} tar hand om skottet från {shooter}.", "{shooter} prövar {goalie}, som står emot."],
        "save_big": ["Stark räddning av {goalie} på {shooter}.", "{goalie} glider över och stoppar {shooter}.", "{shooter} helt fri, men {goalie} står emot."],
        "miss": ["{shooter} missar målet.", "{shooter} skjuter utanför.", "Utanför från {shooter}.", "{shooter} skickar pucken utanför målet."],
        "miss_hd": ["{shooter} helt fri, men han träffar inte målet.", "Fin chans för {shooter}, strax utanför."],
        "block": ["{blocker} blockerar skottet.", "Skottet från {shooter} blockeras av {blocker}.", "Skottet från {shooter} kommer inte fram, {blocker} är i vägen."],
        "en_miss": ["{shooter} siktar mot den tomma kassen, men missar."],
        "goal": ["Mål! {shooter}, {from_where}, och {score_phrase}.", "{shooter} gör mål {from_where}. {Score_phrase}.", "Där satt den, {shooter}. {Score_phrase}."],
        "goal_pp": ["Powerplaymål av {shooter}. {Score_phrase}.", "{shooter} utnyttjar powerplayet. {Score_phrase}."],
        "goal_sh": ["Mål i boxplay av {shooter}. {Score_phrase}."],
        "goal_en": ["{shooter} i tom kasse. {Score_phrase}."],
        "score_lead": ["{team} leder med {score}"],
        "score_ahead": ["{team} tar ledningen, {score}"],
        "score_within": ["{team} reducerar till ett måls underläge, {score}"],
        "score_cut": ["{team} reducerar till {score}"],
        "score_tie": ["det står {score}"],
        "penalty": ["{player} åker ut, {label}.", "Utvisning på {player}, {label}."],
        "pp_start": ["{team} får spela powerplay.", "{team} i powerplay."],
        "pk_kill": ["{team} klarar boxplayet.", "Fulltaligt igen."],
        "coach_pull": ["{team} plockar målvakten för en extra utespelare.", "Tom kasse för {team}, sex utespelare på isen."],
        "coach_return": ["Målvakten är tillbaka i målet för {team}."],
        "coach_timeout": ["{team} tar timeout."],
        "coach_change": ["Justering i båset hos {team}: {what}."],
        "so_start": ["Det blir straffläggning."],
        "so_goal": ["{shooter} sätter straffen.", "{shooter} överlistar {goalie}."],
        "so_save": ["{goalie} stoppar {shooter}.", "{shooter} stoppas av {goalie}."],
        "final": ["Slut. {team} vinner med {score}.", "Slutsignal: {team} tar segern, {score}."],
        "late_close": ["Under två minuter kvar och ett mål skiljer."],
        "late_tied": ["Två minuter kvar och fortfarande lika."],
        "fill_oz": ["{team} trycker på i anfallszonen.", "{team} håller pucken i offensiv zon.", "{team} sätter press mot målet."],
        "fill_nz": ["{team} för upp pucken genom mittzonen.", "{team} samlar ihop sig i mittzonen.", "{team} letar efter ytor genom mitten."],
        "fill_dz": ["{team} försöker spela sig ur egen zon.", "{team} bygger upp bakifrån.", "{team} organiserar sig bakom eget mål."],
        "fill_stop": ["Avblåsning, det blir tekning."],
        "c_shots": ["Det är hans {nth} skott på mål i kväll."],
        "c_share": ["{team} har haft {n} av de senaste {m} avsluten.", "{n} av de senaste {m} avsluten har gått till {team}."],
        "c_goalie": ["{goalie} har räddat {sv} av {sa} skott i kväll.", "{goalie} står på {sv} räddningar av {sa} skott."],
        "c_pp": ["Powerplay i kväll för {team}: {g} på {o}."],
        "c_xg": ["Förväntade mål: {team} {a}, {opp} {b}."],
        "c_run": ["{n} raka mål för {team}."],
        "c_points": ["{player} har redan {pts} poäng i kväll."],
        "c_period_shots": ["Skott på mål i perioden: {team} {a}, {opp} {b}."],
        "c_first": ["Kvällens första mål.", "Där öppnas målskyttet."],
        "c_pk": ["{team} har klarat {k} av {n} boxplay i kväll."],
        "c_trail": ["{team} ligger under med {n}, {time} kvar att spela."],
    },
}

_FROM_WHERE: Dict[str, Dict[str, str]] = {
    "en": {"high_danger": "from in tight", "slot": "from the slot", "rebound": "on the rebound", "point": "from the point", "perimeter": "from distance", "breakaway": "on the rush", "": "from the slot"},
    "fr": {"high_danger": "de près", "slot": "de l'enclave", "rebound": "sur le retour", "point": "de la pointe", "perimeter": "de loin", "breakaway": "en contre-attaque", "": "de l'enclave"},
    "es": {"high_danger": "desde muy cerca", "slot": "desde el slot", "rebound": "en el rebote", "point": "desde la línea azul", "perimeter": "desde lejos", "breakaway": "al contraataque", "": "desde el slot"},
    "de": {"high_danger": "aus kurzer Distanz", "slot": "aus dem Slot", "rebound": "im Nachschuss", "point": "von der blauen Linie", "perimeter": "aus der Distanz", "breakaway": "im Konter", "": "aus dem Slot"},
    "sv": {"high_danger": "från nära håll", "slot": "från slottet", "rebound": "på returen", "point": "från blå linjen", "perimeter": "från distans", "breakaway": "i kontring", "": "från slottet"},
}

_ORDINAL: Dict[str, List[str]] = {
    "en": ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"],
    "fr": ["", "premier", "deuxième", "troisième", "quatrième", "cinquième", "sixième", "septième", "huitième", "neuvième", "dixième"],
    "es": ["", "primer", "segundo", "tercer", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo"],
    "de": ["", "erster", "zweiter", "dritter", "vierter", "fünfter", "sechster", "siebter", "achter", "neunter", "zehnter"],
    "sv": ["", "första", "andra", "tredje", "fjärde", "femte", "sjätte", "sjunde", "åttonde", "nionde", "tionde"],
}

_MINUTE_WORD = {"en": ("minute", "minutes"), "fr": ("minute", "minutes"), "es": ("minuto", "minutos"), "de": ("Minute", "Minuten"), "sv": ("minut", "minuter")}

_WHISTLE: Dict[str, Dict[str, str]] = {
    "en": {"freeze": "{goalie} froze the puck", "out": "Puck out of play", "penalty": "Penalty to {player}", "goal": "Goal, {player}", "period_end": "End of the period", "final": "Final horn"},
    "fr": {"freeze": "{goalie} a immobilisé la rondelle", "out": "Rondelle hors jeu", "penalty": "Pénalité à {player}", "goal": "But de {player}", "period_end": "Fin de la période", "final": "Fin du match"},
    "es": {"freeze": "{goalie} atrapó el disco", "out": "Disco fuera de juego", "penalty": "Penalización a {player}", "goal": "Gol de {player}", "period_end": "Fin del periodo", "final": "Final del partido"},
    "de": {"freeze": "{goalie} hat die Scheibe festgehalten", "out": "Puck aus dem Spiel", "penalty": "Strafe gegen {player}", "goal": "Tor durch {player}", "period_end": "Drittelende", "final": "Spielende"},
    "sv": {"freeze": "{goalie} täckte pucken", "out": "Pucken ur spel", "penalty": "Utvisning på {player}", "goal": "Mål av {player}", "period_end": "Periodslut", "final": "Slutsignal"},
}

_SITUATION: Dict[str, Dict[str, str]] = {
    "en": {"pp": "{team} power play, {time} left", "en": "{team} has the goalie pulled", "oz": "{team} pressing in the {opp} zone",
           "nz": "{team} carrying through the neutral zone", "dz": "{team} breaking out of its own end", "loose": "loose puck",
           "stop": "stoppage, faceoff coming", "goal": "{team} celebrating the goal", "intermission": "Intermission",
           "pregame": "Teams lining up for the opening faceoff", "final": "Final: {score_line}", "shootout": "Shootout"},
    "fr": {"pp": "{team} en avantage numérique, {time} à écouler", "en": "{team} a retiré son gardien", "oz": "{team} presse dans la zone de {opp}",
           "nz": "{team} traverse la zone neutre", "dz": "{team} sort de sa zone", "loose": "rondelle libre",
           "stop": "arrêt de jeu, mise au jeu à venir", "goal": "{team} célèbre le but", "intermission": "Entracte",
           "pregame": "Les équipes s'alignent pour la mise au jeu initiale", "final": "Final : {score_line}", "shootout": "Fusillade"},
    "es": {"pp": "{team} en power play, quedan {time}", "en": "{team} ha retirado al portero", "oz": "{team} presiona en la zona de {opp}",
           "nz": "{team} cruza la zona neutral", "dz": "{team} sale de su zona", "loose": "disco suelto",
           "stop": "juego detenido, viene un saque neutral", "goal": "{team} celebra el gol", "intermission": "Descanso",
           "pregame": "Los equipos se alinean para el saque inicial", "final": "Final: {score_line}", "shootout": "Tanda de penaltis"},
    "de": {"pp": "{team} in Überzahl, noch {time}", "en": "{team} hat den Torhüter gezogen", "oz": "{team} drückt im Drittel von {opp}",
           "nz": "{team} läuft durch die neutrale Zone", "dz": "{team} befreit sich aus dem eigenen Drittel", "loose": "freie Scheibe",
           "stop": "Unterbrechung, gleich Bully", "goal": "{team} jubelt über das Tor", "intermission": "Drittelpause",
           "pregame": "Die Teams stellen sich zum Eröffnungsbully auf", "final": "Endstand: {score_line}", "shootout": "Penaltyschießen"},
    "sv": {"pp": "{team} i powerplay, {time} kvar", "en": "{team} har plockat målvakten", "oz": "{team} trycker på i anfallszonen mot {opp}",
           "nz": "{team} för pucken genom mittzonen", "dz": "{team} spelar sig ur egen zon", "loose": "lös puck",
           "stop": "avblåst, tekning väntar", "goal": "{team} firar målet", "intermission": "Periodpaus",
           "pregame": "Lagen ställer upp för öppningstekningen", "final": "Slut: {score_line}", "shootout": "Straffläggning"},
}


def _dec(v: float, lang: str) -> str:
    s = f"{float(v):.1f}"
    return s if lang == "en" else s.replace(".", ",")


def _time_tts(sec: float, lang: str) -> str:
    sec = max(0, int(math.ceil(float(sec))))
    m, s = sec // 60, sec % 60
    one, many = _MINUTE_WORD.get(lang, _MINUTE_WORD["en"])
    if m == 0:
        return f"{_num_word(s, lang) if s <= 12 else s} " + {"en": "seconds", "fr": "secondes", "es": "segundos", "de": "Sekunden", "sv": "sekunder"}.get(lang, "seconds")
    word = one if m == 1 else many
    if m == 1:
        lead = {"en": "one", "fr": "une", "es": "un", "de": "eine", "sv": "en"}.get(lang, "one")
        return f"{lead} {word} {s}" if s else f"{lead} {word}"
    return f"{_num_word(m, lang) if m <= 12 else m} {word} {s}" if s else f"{_num_word(m, lang) if m <= 12 else m} {word}"


_ALERT_I18N: Dict[str, Dict[str, str]] = {
    "en": {
        "goal": "{team} goal — {player}", "penalty": "{player} penalized: {label}",
        "pp": "Power play for {team} ({label})", "pk": "Penalty kill for {team} ({label})",
        "fatigue": "{player} is gassed — change lines?", "hot": "{player} is hot: {pts} pts, {sog} SOG",
        "cold": "{player} is struggling ({pm} on ice) — sit him?", "goalie": "{player} is struggling: {ga} GA, {svp} save pct",
        "momentum": "Momentum swinging to {team}", "timeout_suggest": "Two quick goals against — call a timeout?",
        "pull_suggest": "Down {n} late — pull the goalie?",
    },
    "fr": {
        "goal": "But de {team} — {player}", "penalty": "Pénalité à {player} : {label}",
        "pp": "Avantage numérique pour {team} ({label})", "pk": "Désavantage numérique pour {team} ({label})",
        "fatigue": "{player} est épuisé — changer de trio?", "hot": "{player} est en feu : {pts} pts, {sog} tirs",
        "cold": "{player} connaît des difficultés ({pm}) — le laisser au banc?", "goalie": "{player} en difficulté : {ga} buts accordés, efficacité {svp}",
        "momentum": "L'élan passe du côté de {team}", "timeout_suggest": "Deux buts rapides contre nous — prendre le temps d'arrêt?",
        "pull_suggest": "En retard de {n} en fin de match — retirer le gardien?",
    },
    "es": {
        "goal": "Gol de {team} — {player}", "penalty": "Penalización a {player}: {label}",
        "pp": "Power play para {team} ({label})", "pk": "Inferioridad numérica para {team} ({label})",
        "fatigue": "{player} está agotado — ¿cambiar de línea?", "hot": "{player} está encendido: {pts} pts, {sog} tiros",
        "cold": "{player} lo está pasando mal ({pm}) — ¿sentarlo?", "goalie": "{player} tiene problemas: {ga} goles en contra, {svp} de atajadas",
        "momentum": "El impulso cambia hacia {team}", "timeout_suggest": "Dos goles rápidos en contra — ¿pedir tiempo muerto?",
        "pull_suggest": "Perdiendo por {n} al final — ¿sacar al portero?",
    },
    "de": {
        "goal": "Tor für {team} — {player}", "penalty": "Strafe gegen {player}: {label}",
        "pp": "Überzahl für {team} ({label})", "pk": "Unterzahl für {team} ({label})",
        "fatigue": "{player} ist platt — Reihe wechseln?", "hot": "{player} ist heiß: {pts} Pkt., {sog} Schüsse",
        "cold": "{player} tut sich schwer ({pm}) — auf die Bank?", "goalie": "{player} wackelt: {ga} Gegentore, Fangquote {svp}",
        "momentum": "Das Momentum kippt zu {team}", "timeout_suggest": "Zwei schnelle Gegentore — Auszeit nehmen?",
        "pull_suggest": "Rückstand von {n} kurz vor Schluss — Torhüter raus?",
    },
    "sv": {
        "goal": "Mål för {team} — {player}", "penalty": "Utvisning på {player}: {label}",
        "pp": "Powerplay för {team} ({label})", "pk": "Boxplay för {team} ({label})",
        "fatigue": "{player} är slut — byt kedja?", "hot": "{player} är het: {pts} poäng, {sog} skott",
        "cold": "{player} har det tungt ({pm}) — bänka honom?", "goalie": "{player} har det kämpigt: {ga} insläppta, räddningsprocent {svp}",
        "momentum": "Momentum svänger till {team}", "timeout_suggest": "Två snabba baklängesmål — ta timeout?",
        "pull_suggest": "Underläge med {n} sent — plocka målvakten?",
    },
}

_MOMENTUM_LABEL: Dict[str, Tuple[str, str, str]] = {
    "en": ("Home pushing", "Even", "Away pushing"),
    "fr": ("Les locaux poussent", "Égal", "Les visiteurs poussent"),
    "es": ("Presiona el local", "Parejo", "Presiona el visitante"),
    "de": ("Heimteam drückt", "Ausgeglichen", "Gäste drücken"),
    "sv": ("Hemmalaget trycker på", "Jämnt", "Bortalaget trycker på"),
}

# Short, generic, original crowd chants. {team} = home nickname, {goalie} = home goalie surname.
_CHANTS: Dict[str, Dict[str, List[str]]] = {
    "en": {"team": ["Let's go {team}!", "{team}! {team}!"], "goalie": ["{goalie}! {goalie}!"], "more": ["We want more!", "One more goal!"],
           "defense": ["Defense! Defense!"], "shoot": ["Shoot! Shoot!"], "rival": ["Beat {rival}!", "This is our house!"]},
    "fr": {"team": ["Allez {team}!", "{team}! {team}!"], "goalie": ["{goalie}! {goalie}!"], "more": ["On en veut encore!", "Encore un!"],
           "defense": ["Défense! Défense!"], "shoot": ["Lance! Lance!"], "rival": ["Battez {rival}!", "On est chez nous!"]},
    "es": {"team": ["¡Vamos {team}!", "¡{team}! ¡{team}!"], "goalie": ["¡{goalie}! ¡{goalie}!"], "more": ["¡Queremos más!", "¡Otro gol!"],
           "defense": ["¡Defensa! ¡Defensa!"], "shoot": ["¡Tira! ¡Tira!"], "rival": ["¡A ganarle a {rival}!", "¡Esta es nuestra casa!"]},
    "de": {"team": ["Auf geht's {team}!", "{team}! {team}!"], "goalie": ["{goalie}! {goalie}!"], "more": ["Wir wollen mehr!", "Noch eins!"],
           "defense": ["Verteidigung! Verteidigung!"], "shoot": ["Schieß! Schieß!"], "rival": ["Schlagt {rival}!", "Das ist unser Haus!"]},
    "sv": {"team": ["Heja {team}!", "{team}! {team}!"], "goalie": ["{goalie}! {goalie}!"], "more": ["Vi vill ha mer!", "Ett till!"],
           "defense": ["Försvar! Försvar!"], "shoot": ["Skjut! Skjut!"], "rival": ["Slå {rival}!", "Det här är vårt hus!"]},
}

# Round-5 broadcast calls (build-ups, goal-call pieces, replays, play narration, rule calls) live in
# services/live_calls.py and are merged here; this file keeps working without it.
try:
    from services import live_calls as _LC
except Exception:  # pragma: no cover - optional module
    _LC = None
if _LC is not None:
    for _lang, _rows in _LC.CALLS.items():
        _dst = COMMENTARY_I18N.setdefault(_lang, {})
        for _k, _v in _rows.items():
            _dst[_k] = (list(_dst.get(_k) or []) + list(_v)) if (_lang != "en" and _k in ("save", "miss", "block")) else list(_v)
    _CHANTS = _LC.CHANTS
    for _lang, _rows in _LC.WHISTLE.items():
        _WHISTLE.setdefault(_lang, {}).update(_rows)


_COACH_WHAT: Dict[str, Dict[str, str]] = {
    "en": {"pull": "goalie pulled for an extra attacker", "return": "goalie back in net", "timeout": "timeout",
           "shorten_bench": "shortening the bench", "roll_lines": "rolling all four lines", "normal_rotation": "normal rotation",
           "next_line": "sending out {unit}", "bench_player": "benching {player}", "unbench": "{player} back in the rotation",
           "tactics": "new tactics ({detail})", "line": "line {n}", "pair": "pair {n}"},
    "fr": {"pull": "gardien retiré pour un attaquant supplémentaire", "return": "gardien de retour devant le filet", "timeout": "temps d'arrêt",
           "shorten_bench": "on raccourcit le banc", "roll_lines": "les quatre trios en rotation", "normal_rotation": "rotation normale",
           "next_line": "on envoie {unit}", "bench_player": "{player} cloué au banc", "unbench": "{player} de retour dans la rotation",
           "tactics": "nouvelle tactique ({detail})", "line": "le trio {n}", "pair": "la paire {n}"},
    "es": {"pull": "portero fuera por un atacante extra", "return": "portero de vuelta en la portería", "timeout": "tiempo muerto",
           "shorten_bench": "se acorta la rotación", "roll_lines": "rotan las cuatro líneas", "normal_rotation": "rotación normal",
           "next_line": "sale {unit}", "bench_player": "{player} al banquillo", "unbench": "{player} vuelve a la rotación",
           "tactics": "nueva táctica ({detail})", "line": "la línea {n}", "pair": "la pareja {n}"},
    "de": {"pull": "Torhüter raus für einen zusätzlichen Feldspieler", "return": "Torhüter zurück im Tor", "timeout": "Auszeit",
           "shorten_bench": "die Bank wird verkürzt", "roll_lines": "alle vier Reihen rotieren", "normal_rotation": "normale Rotation",
           "next_line": "{unit} aufs Eis", "bench_player": "{player} auf die Bank", "unbench": "{player} zurück in der Rotation",
           "tactics": "neue Taktik ({detail})", "line": "Reihe {n}", "pair": "Verteidigerpaar {n}"},
    "sv": {"pull": "målvakten ute för en extra utespelare", "return": "målvakten tillbaka i mål", "timeout": "timeout",
           "shorten_bench": "bänken kortas", "roll_lines": "fyra kedjor rullar", "normal_rotation": "normal rotation",
           "next_line": "{unit} skickas ut", "bench_player": "{player} bänkad", "unbench": "{player} tillbaka i rotationen",
           "tactics": "ny taktik ({detail})", "line": "kedja {n}", "pair": "backpar {n}"},
}

_RIVAL_NAME: Dict[str, Dict[Any, str]] = {
    "en": {1: "Rivals", 2: "Rivalry", 3: "Bitter rivalry", "div": "Division rivals"},
    "fr": {1: "Rivaux", 2: "Rivalité", 3: "Rivalité féroce", "div": "Rivaux de division"},
    "es": {1: "Rivales", 2: "Rivalidad", 3: "Rivalidad encarnizada", "div": "Rivales de división"},
    "de": {1: "Rivalen", 2: "Rivalität", 3: "Erzrivalen", "div": "Divisionsrivalen"},
    "sv": {1: "Rivaler", 2: "Rivalitet", 3: "Bitter rivalitet", "div": "Divisionsrivaler"},
}


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _fmt(template: str, params: Dict[str, Any]) -> str:
    try:
        return str(template).format_map(_SafeDict(params))
    except Exception:
        return str(template)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _loc(table: Dict[str, Dict[Any, Any]], lang: str, key: Any, default: Any = "") -> Any:
    row = table.get(lang) or table.get("en") or {}
    if key in row:
        return row[key]
    return (table.get("en") or {}).get(key, default)


def _speakable(text: str, lang: str) -> str:
    """TTS-friendly line: no symbols, PP/SOG spoken out."""
    import re

    s = str(text or "")
    if lang == "en":
        s = re.sub(r"(\d)–(\d)", r"\1-\2", s)  # 9–0 -> nine-zero
    s = s.replace("—", ", ").replace("–", " ").replace("...", "…")
    s = re.sub(r"\s*…\s*", "… ", s).replace("… !", "…!").replace("… ?", "…?").replace("… .", "….")
    s = re.sub(r"#\d+", "", s)
    s = re.sub(r"\bPP\b", _PP_WORD.get(lang, "power play"), s)
    s = re.sub(r"\bSOG\b", _SOG_WORD.get(lang, "shots on goal"), s)
    if _LC is not None:  # no raw abbreviations in speech: 4v5, PK, OT, xG ...
        _w = _NUM_WORDS.get(lang) or _NUM_WORDS["en"]
        s = re.sub(r"\b([3-6])\s?(?:v|vs\.?|on)\s?([3-6])\b", lambda m_: f"{_w[int(m_.group(1))]} {_LC.VS_WORD.get(lang, 'on')} {_w[int(m_.group(2))]}", s)
        for _ab, _full in (_LC.ABBR.get(lang) or _LC.ABBR["en"]).items():
            s = re.sub(r"\b" + re.escape(_ab) + r"\b", _full, s)
    pct = {"en": " percent", "fr": " pour cent", "es": " por ciento", "de": " Prozent", "sv": " procent"}.get(lang, " percent")
    sep = "." if lang == "en" else ","
    if lang == "en":
        s = re.sub(r"\ba (?:\.(\d)(\d)(\d)) save percentage", lambda m_: f"a save percentage of {m_.group(1)}{m_.group(2)}.{m_.group(3)}", s)
    s = re.sub(r"(?<![\d])\.(\d)(\d)(\d)(?!\d)", lambda m_: f"{m_.group(1)}{m_.group(2)}{sep}{m_.group(3)}{pct}", s)  # .912 -> 91.2 percent
    words = _NUM_WORDS.get(lang) or _NUM_WORDS["en"]
    num_re = re.compile(r"(?<![\d.,:])(\d+)(?!\d|[.,:]\d|[A-Za-zÀ-ÿ])")

    def _sentence(sent: str) -> str:
        # one style per sentence: words up to twenty, digits for the whole sentence if any number is bigger
        nums = [int(x) for x in num_re.findall(sent)]
        if any(n > 20 for n in nums) or re.search(r"\d[.,]\d", sent):
            return sent
        return num_re.sub(lambda m_: words[int(m_.group(1))] if int(m_.group(1)) < len(words) else m_.group(1), sent)

    s = " ".join(_sentence(x) for x in re.split(r"(?<=[.!?])\s+", s))
    s = s.replace("(", ", ").replace(")", "").replace(":", ",").replace(";", ",")
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s{2,}", " ", s)
    if lang == "en":
        s = re.sub(r"\b([Aa]) (?=(?:8\d*|11|18)\b|(?:8\d*|11|18)-|eight|eleven|eighteen|eighty)", r"\1n ", s)
    s = s.strip(" ,")
    s = re.sub(r"((?<!…)[.!?] )([a-zà-ÿ])", lambda m_: m_.group(1) + m_.group(2).upper(), s)
    s = s.replace("…", "...")  # ellipsis = held pause for the voice engine
    return s[:1].upper() + s[1:]


def _en_article(s: str) -> str:
    import re

    return re.sub(r"\b([Aa]) (?=(?:8\d*|11|18)\b|(?:8\d*|11|18)-)", r"\1n ", s)


# ═══ 4. Player helpers + instrumented engine pass

_MISSING = object()
_ENGINE_PATCH_LOCK = threading.RLock()
_SEG_FUNC = "_simulate_segment"
_OT_TICK_SEC = 18.0  # engine's non-chronological (OT) attempt spacing
_HD_CHANCES = frozenset({"high_danger", "rebound", "breakaway"})
_SIDES = ("home", "away")

_CHANCE_MAP = {
    "LOW_DANGER_PERIMETER": "perimeter", "POINT_SHOT": "point", "RUSH_MEDIUM": "slot", "SLOT": "slot",
    "HIGH_DANGER_SLOT": "high_danger", "NET_FRONT": "high_danger", "REBOUND": "rebound", "ONE_TIMER": "slot",
    "PP_SLOT": "slot", "PP_ONE_TIMER": "slot", "SH_RUSH": "slot",
}

_TACTIC_KEYS = ("forecheck", "defense", "shot_selection", "pp_style", "pk_style")
_TACTIC_VALUES = ("passive", "balanced", "aggressive")
# Multipliers on engine inputs (all bounded again by the engine's own clamps).
_TACTIC_OFF = {"forecheck": {"aggressive": 1.06, "passive": 0.95}, "shot_selection": {"aggressive": 0.98, "passive": 1.03}}
_TACTIC_SUP = {"forecheck": {"aggressive": 0.94, "passive": 1.05}, "defense": {"aggressive": 0.95, "passive": 1.08}}
_TACTIC_CTX_ATK = {"forecheck": {"aggressive": 1.05, "passive": 0.96}, "shot_selection": {"aggressive": 0.96, "passive": 1.05}}
_TACTIC_CTX_DEF = {"defense": {"aggressive": 1.05, "passive": 0.94}, "forecheck": {"aggressive": 1.03, "passive": 0.98}}
_PP_ATK = {"aggressive": 1.07, "passive": 0.95}   # own PP chances
_PP_RISK = {"aggressive": 1.10, "passive": 0.92}  # opponent shorthanded chances against our PP
_PK_DEF = {"aggressive": 1.04, "passive": 0.95}   # opponent PP chances against our PK
_PK_ATK = {"aggressive": 1.12, "passive": 0.85}   # our shorthanded chances


def _default_tactics() -> Dict[str, Any]:
    return {k: "balanced" for k in _TACTIC_KEYS}


def _contract_chance(raw: str, tags: List[str], rebound: bool) -> Optional[str]:
    if rebound:
        return "rebound"
    r = str(raw or "").upper()
    if r in ("RUSH_MEDIUM", "SH_RUSH") and "odd_man_look" in (tags or []):
        return "breakaway"
    return _CHANCE_MAP.get(r)


def _segment_frame() -> Any:
    """Frame of the engine's _simulate_segment that called the current hook (or None)."""
    try:
        f = sys._getframe(2)
    except ValueError:
        return None
    for _ in range(4):
        if f is None:
            return None
        if f.f_code.co_name == _SEG_FUNC:
            return f
        f = f.f_back
    return None


def _opp(side: str) -> str:
    return "away" if side == "home" else "home"


def _truthy(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(v)


def _pname(p: Any) -> str:
    if p is None:
        return ""
    ident = getattr(p, "identity", None)
    n = getattr(ident, "name", None) if ident is not None else None
    if not n:
        n = getattr(p, "name", None)
    if not n:
        fn = (getattr(ident, "first_name", None) if ident is not None else None) or getattr(p, "first_name", None) or ""
        ln = (getattr(ident, "last_name", None) if ident is not None else None) or getattr(p, "last_name", None) or ""
        n = f"{fn} {ln}".strip()
    return str(n or "?")


def _short_name(name: str) -> str:
    parts = str(name or "").split()
    return parts[-1] if parts else str(name or "")


def _pnum(p: Any) -> Any:
    ident = getattr(p, "identity", None)
    for v in (
        getattr(p, "jersey_number", None),
        getattr(p, "number", None),
        getattr(ident, "jersey_number", None) if ident is not None else None,
        getattr(ident, "number", None) if ident is not None else None,
    ):
        if v is None or v == "":
            continue
        s = str(v).strip()
        return int(s) if s.isdigit() else s
    return None


def _ppos(p: Any) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident is not None else None
    if pos is None or str(getattr(pos, "value", pos) or "").strip() in ("", "?"):
        pos = getattr(p, "position", None)
    raw = str(getattr(pos, "value", pos) or "?").strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"LD", "RD", "LHD", "RHD", "DEF", "DEFENSE", "DEFENCE", "DEFENSEMAN", "D"}:
        return "D"
    if raw in {"GOALIE", "GOALTENDER", "G"}:
        return "G"
    if raw in {"CENTER", "CENTRE", "C"}:
        return "C"
    if raw in {"LEFT_WING", "LEFTWING", "LW"}:
        return "LW"
    if raw in {"RIGHT_WING", "RIGHTWING", "RW"}:
        return "RW"
    return raw or "?"


def _headshot_fields(p: Any, pid: str, name: str, pos: str) -> Dict[str, Any]:
    """Identity + portrait fields read by the frontend headshot resolver (no mutation of the player object)."""
    out: Dict[str, Any] = {"id": pid, "player_id": pid, "name": name, "position": pos}
    try:
        from app.sim_engine.generation.player_headshots import headshot_fields_from_player

        f = headshot_fields_from_player(p)
        out.update({k: v for k, v in f.items() if v not in (None, "", 0)})
    except Exception:
        logger.debug("headshot fields unavailable", exc_info=True)
    ident = getattr(p, "identity", None)
    for key, attr in (("age", "age"), ("nationality", "birth_country"), ("shoots", "shoots")):
        v = getattr(ident, attr, None) if ident is not None else getattr(p, key, None)
        if hasattr(v, "value"):
            v = v.value
        if v not in (None, ""):
            out[key] = v
    tid = getattr(p, "team_id", None) or getattr(p, "current_team_id", None)
    if tid:
        out["team_id"] = str(tid)
    return out


class _PlayerBook:
    """pid -> PlayerChip (+ side) for every player the engine put on the ice."""

    def __init__(self) -> None:
        self.chips: Dict[str, Dict[str, Any]] = {}
        self.side: Dict[str, str] = {}

    def add(self, p: Any, side: Optional[str]) -> str:
        if p is None:
            return ""
        try:
            pid = _pid(p)
        except Exception:
            pid = str(getattr(p, "id", "") or "")
        if not pid:
            return ""
        if pid not in self.chips:
            name = _pname(p)
            self.chips[pid] = {"player_id": pid, "name": name, "short": _short_name(name), "number": _pnum(p), "pos": _ppos(p),
                               "hs": _headshot_fields(p, pid, name, _ppos(p))}
        if side:
            self.side.setdefault(pid, side)
        return pid

    def chip(self, pid: Optional[str]) -> Optional[Dict[str, Any]]:
        c = self.chips.get(str(pid or ""))
        return dict(c) if c else None

    def name(self, pid: Optional[str]) -> str:
        c = self.chips.get(str(pid or ""))
        return c["name"] if c else ""

    def short(self, pid: Optional[str]) -> str:
        c = self.chips.get(str(pid or ""))
        return c["short"] if c else ""

    def number(self, pid: Optional[str]) -> Any:
        c = self.chips.get(str(pid or ""))
        return c["number"] if c else None

    def pos(self, pid: Optional[str]) -> str:
        c = self.chips.get(str(pid or ""))
        return c["pos"] if c else "?"


def _attempt_sig(a: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        a.get("seg"), round(float(a.get("local_t") or 0.0), 3), a.get("team_id"), a.get("shooter_id"),
        a.get("outcome"), round(float(a.get("xg") or 0.0), 4),
    )


class _EnginePass:
    """One instrumented SimEngine game (`_run_event_driven_game`).

    Hooks observe every shot attempt (with the real on-ice units, goalie, blocker,
    chance type, engine game clock and skater counts) and every penalty draw.
    Applied coach orders act through the same hooks, but only for attempts with
    index >= the order's `from_idx`; earlier attempts get the exact engine values,
    so a replay with the same seed reproduces everything the user already watched.
    Hooks never draw from the engine RNG.
    """

    def __init__(self, loop: "LiveEventLoop", effects: List[Dict[str, Any]]) -> None:
        self.loop = loop
        self.sim = loop.sim
        self.book = loop.book
        self.home_id = str(loop.lg.home_id)
        self.away_id = str(loop.lg.away_id)
        self.effects = [dict(e) for e in (effects or [])]
        # CPU coach may only change play once the user already did, and never before the
        # first user order (keeps every replay's shown prefix identical to the original pass).
        self.cpu_enabled = bool(self.effects)
        self.cpu_from = min((int(e.get("from_idx", 0)) for e in self.effects), default=0)
        self.user_side = loop.lg.user_side
        self.thread = threading.get_ident()
        self.attempts: List[Dict[str, Any]] = []
        self.coach_raw: List[Dict[str, Any]] = []
        self.penalties: List[Dict[str, Any]] = []
        self.ctx: Dict[str, Any] = {}
        self.cur: Optional[Dict[str, Any]] = None
        self._seg_k = 0
        self._last_local = -1.0
        self._last_seg: Optional[str] = None
        self.pulled: Dict[str, Optional[Dict[str, Any]]] = {"home": None, "away": None}
        self.last_return_idx = {"home": -10, "away": -10}
        self.goals = {"home": 0, "away": 0}
        self.line_at: Dict[int, Dict[str, Tuple[Any, Any, str]]] = {}
        self._used: Dict[str, int] = {}
        self._draws = {"home": 0, "away": 0}
        self._done_pull: set = set()
        self.origs: Dict[str, Any] = {}
        self.orig_credit: Any = None

    # ── context
    def _mine(self, fl: Any) -> bool:
        try:
            return threading.get_ident() == self.thread and fl is not None and fl.get("self") is self.sim
        except Exception:
            return False

    def _team_side(self, team: Any) -> Optional[str]:
        if team is self.loop.lg.home:
            return "home"
        if team is self.loop.lg.away:
            return "away"
        return None

    def _grab_ctx(self, fl: Any) -> None:
        for key in ("home_units", "away_units", "home_gl", "away_gl", "home_starter", "away_starter", "home_dressed", "away_dressed"):
            try:
                self.ctx[key] = fl.get(key)
            except Exception:
                self.ctx[key] = None
        for side in _SIDES:
            for p in list(self.ctx.get(f"{side}_dressed") or []) + list(self.ctx.get(f"{side}_gl") or []):
                self.book.add(p, side)
            st = self.ctx.get(f"{side}_starter")
            if st is not None:
                self.book.add(st, side)
            if self.ctx.get(f"{side}_pen") is None:
                try:
                    self.ctx[f"{side}_pen"] = fl.get(f"{side}_penalty_intervals") or []
                except Exception:
                    self.ctx[f"{side}_pen"] = []
        self.ctx["captured"] = True

    def _begin(self, fl: Any) -> Dict[str, Any]:
        i = len(self.attempts)
        if self.cur is not None and self.cur["idx"] == i:
            return self.cur
        chrono = bool(fl.get("chronological"))
        local_t = float(fl.get("game_sec_elapsed") or 0.0)
        if chrono:
            seg = "reg"
        else:
            if self._last_seg in (None, "reg") or local_t <= self._last_local:
                self._seg_k += 1
            seg = f"ot{self._seg_k}"
        if not self.ctx.get("captured"):
            self._grab_ctx(fl)
        if self._last_seg == "reg" and seg != "reg":
            for s in _SIDES:
                if self.pulled[s]:
                    self._goalie_return(s, i, "reg", _REGULATION_GAME_SEC, 0, "period_end")
        self._last_seg = seg
        self._last_local = local_t
        self.cur = {"idx": i, "seg": seg, "local_t": local_t, "total": int(fl.get("total") or 0)}
        self._apply_pull_effects(i, seg, local_t)
        if self.cpu_enabled and i >= self.cpu_from:
            self._cpu_coach(i, seg, local_t)
        return self.cur

    def skaters_at(self, seg: str, local_t: float, strength: Optional[str] = None, side: Optional[str] = None) -> Tuple[int, int]:
        if seg == "reg":
            hb = sum(1 for s, d in (self.ctx.get("home_pen") or []) if s <= local_t < s + d)
            ab = sum(1 for s, d in (self.ctx.get("away_pen") or []) if s <= local_t < s + d)
            return max(3, 5 - hb), max(3, 5 - ab)
        base = 5 if self.loop.lg.is_playoff else 3
        h = a = base
        st = str(strength or "EV").upper()
        if st == "PP" and side:
            h, a = (base + 1, base) if side == "home" else (base, base + 1)
        elif st == "SH" and side:
            h, a = (base, base + 1) if side == "home" else (base + 1, base)
        return h, a

    # ── effects
    def _active(self, kind: str, side: str, i: int) -> List[Dict[str, Any]]:
        return [
            e for e in self.effects
            if e.get("kind") == kind and e.get("side") == side and int(e.get("from_idx", 0)) <= i
            and (e.get("until_idx") is None or i < int(e["until_idx"]))
        ]

    def _tac(self, side: str, i: int) -> Dict[str, Any]:
        t = _default_tactics()
        for e in self._active("tactics", side, i):
            t.update(e.get("tactics") or {})
        return t

    @staticmethod
    def _mult(table: Dict[str, Dict[str, float]], tac: Dict[str, Any]) -> float:
        m = 1.0
        for k, opts in table.items():
            m *= float(opts.get(str(tac.get(k)), 1.0))
        return m

    def _goalie_return(self, side: str, i: int, seg: str, local_t: float, total: int, reason: str) -> None:
        self.pulled[side] = None
        self.last_return_idx[side] = i
        phase = 2.5 if reason == "goal" else (-2.0 if reason == "period_end" else 1.4)
        self.coach_raw.append({
            "action": "return", "side": side, "by": "auto" if reason in ("goal", "period_end") else "user",
            "aidx": i, "seg": seg, "local_t": local_t, "total": total, "phase": phase, "reason": reason,
            "silent": reason == "order",
        })

    def _apply_pull_effects(self, i: int, seg: str, local_t: float) -> None:
        total = int((self.cur or {}).get("total") or 0)
        for e in self.effects:
            if e.get("kind") != "pull" or e.get("id") in self._done_pull or int(e.get("from_idx", 0)) > i:
                continue
            self._done_pull.add(e.get("id"))
            s = e["side"]
            if e.get("pull", True):
                if not self.pulled[s]:
                    self.pulled[s] = {"idx": i, "by": e.get("by", "user")}
                    self.coach_raw.append({
                        "action": "pull", "side": s, "by": e.get("by", "user"), "aidx": i, "seg": seg,
                        "local_t": local_t, "total": total, "phase": 1.5, "silent": e.get("by", "user") == "user",
                    })
            elif self.pulled[s]:
                self._goalie_return(s, i, seg, local_t, total, "order")

    def _cpu_coach(self, i: int, seg: str, local_t: float) -> None:
        cpu = _opp(self.user_side)
        if seg != "reg" or self.pulled[cpu] or i - self.last_return_idx[cpu] < 2:
            return
        diff = self.goals[cpu] - self.goals[self.user_side]
        if diff not in (-1, -2):
            return
        if local_t < _REGULATION_GAME_SEC - (95.0 if diff == -1 else 160.0):
            return
        h, a = self.skaters_at("reg", local_t)
        mine, theirs = (h, a) if cpu == "home" else (a, h)
        if mine < theirs:
            return
        self.pulled[cpu] = {"idx": i, "by": "cpu"}
        self.coach_raw.append({
            "action": "pull", "side": cpu, "by": "cpu", "aidx": i, "seg": seg, "local_t": local_t,
            "total": int((self.cur or {}).get("total") or 0), "phase": 1.5, "silent": False,
        })

    def _deploy(self, side: str, units: Dict[str, Any], unit: List[Any], i: int) -> List[Any]:
        lines = list(units.get("lines") or [])
        pairs = list(units.get("pairs") or [])
        sh = units.get("_shift") if isinstance(units.get("_shift"), dict) else {}
        li = int(sh.get("li", 0) or 0)
        pi = int(sh.get("pi", 0) or 0)
        nli, npi = li, pi
        self._draws[side] += 1
        forced = False
        for e in self.effects:
            if e.get("side") != side or e.get("kind") not in ("next_line", "rest") or int(e.get("from_idx", 0)) > i:
                continue
            used = self._used.get(e["id"], 0)
            if used >= int(e.get("count", 2)):
                continue
            self._used[e["id"]] = used + 1
            if e["kind"] == "rest":
                nli, npi = 0, 0
            else:
                if e.get("F") is not None:
                    nli = int(e["F"])
                if e.get("D") is not None:
                    npi = int(e["D"])
            forced = True
            break
        if not forced:
            modes = self._active("deploy", side, i)
            mode = modes[-1].get("mode") if modes else "normal"
            n = self._draws[side]
            if mode == "short":
                if nli >= 3:
                    nli = n % 3
                if npi >= 2:
                    npi = n % 2
            elif mode == "roll":
                nli = (n // 2) % 4
                npi = (n // 2) % 3
        if (nli, npi) == (li, pi):
            return self._double(side, units, unit, i, nli, npi)
        line = list(lines[nli]) if 0 <= nli < len(lines) and lines[nli] else (list(lines[li]) if 0 <= li < len(lines) else [])
        pair = list(pairs[npi]) if 0 <= npi < len(pairs) and pairs[npi] else (list(pairs[pi]) if 0 <= pi < len(pairs) else [])
        if not line or not pair:
            return unit
        for pl in line:
            setattr(pl, "_gm_shift_line_idx", nli)
        for pl in pair:
            setattr(pl, "_gm_shift_pair_idx", npi)
        return self._double(side, units, line + pair, i, nli, npi)

    def _double(self, side: str, units: Dict[str, Any], unit: List[Any], i: int, li: int, pi: int) -> List[Any]:
        """Double-shift: the player also takes the 3rd/4th-line (or 3rd-pair) shifts."""
        dbl = [e for e in self._active("double", side, i)]
        if not dbl:
            return unit
        out = list(unit)
        ids = [_pid(p) for p in out]
        for e in dbl:
            pid = str(e.get("player_id") or "")
            if not pid or pid in ids:
                continue
            obj = None
            for grp in list(units.get("lines") or []) + list(units.get("pairs") or []):
                for p in grp:
                    if _pid(p) == pid:
                        obj = p
            if obj is None:
                continue
            is_d = self.sim._gm_pos_str(obj) == "D"
            if (is_d and pi < 2) or ((not is_d) and li < 2):
                continue
            for j, p in enumerate(out):
                if (self.sim._gm_pos_str(p) == "D") == is_d:
                    out[j] = obj
                    ids[j] = pid
                    break
        return out

    def _bench(self, side: str, units: Dict[str, Any], unit: List[Any], i: int) -> List[Any]:
        benched = {str(e.get("player_id")) for e in self._active("bench", side, i)}
        if not benched or not any(_pid(p) in benched for p in unit):
            return unit
        on = {_pid(p) for p in unit}
        pool_f = [p for ln in reversed(list(units.get("lines") or [])) for p in ln]
        pool_d = [p for pr in reversed(list(units.get("pairs") or [])) for p in pr]
        out: List[Any] = []
        for p in unit:
            pid = _pid(p)
            if pid not in benched:
                out.append(p)
                continue
            pool = pool_d if self.sim._gm_pos_str(p) == "D" else pool_f
            rep = next((q for q in pool if _pid(q) not in on and _pid(q) not in benched), None)
            if rep is None:
                out.append(p)
                continue
            on.add(_pid(rep))
            out.append(rep)
        return out

    def _note_line(self, side: str, units: Dict[str, Any], unit: List[Any], i: int, st: str) -> None:
        rec = self.line_at.setdefault(i, {})
        if st != "EV":
            rec[side] = (None, None, st)
            return
        ids = {_pid(p) for p in unit}
        F = D = None
        for j, ln in enumerate(units.get("lines") or []):
            if ln and sum(1 for p in ln if _pid(p) in ids) >= 2:
                F = j
                break
        for j, pr in enumerate(units.get("pairs") or []):
            if pr and any(_pid(p) in ids for p in pr):
                D = j
                break
        rec[side] = (F, D, st)

    # ── wrappers (installed as instance attributes on the sim for the pass only)
    def w_gm_on_ice_unit(self, units: Any, strength: Any, rng: Any, *a: Any, **kw: Any) -> Any:
        res = self.origs["_gm_on_ice_unit"](units, strength, rng, *a, **kw)
        fr = _segment_frame()
        if fr is None:
            return res
        fl = fr.f_locals
        if not self._mine(fl):
            return res
        try:
            unit, extra = res
            cur = self._begin(fl)
            side = "home" if units is self.ctx.get("home_units") else ("away" if units is self.ctx.get("away_units") else None)
            if side is None:
                return res
            i = cur["idx"]
            st = str(strength or "EV").upper()
            new = list(unit)
            if st == "EV":
                new = self._deploy(side, units, new, i)
            new = self._bench(side, units, new, i)
            self._note_line(side, units, new, i, st)
            return new, extra
        except Exception:
            logger.warning("live on-ice hook failed", exc_info=True)
            return res

    def w_team_offense_skill(self, team: Any, *a: Any, **kw: Any) -> Any:
        v = self.origs["_team_offense_skill"](team, *a, **kw)
        if a or kw:
            return v
        fr = _segment_frame()
        if fr is None or not self._mine(fr.f_locals):
            return v
        try:
            side = self._team_side(team)
            if side is None:
                return v
            m = self._mult(_TACTIC_OFF, self._tac(side, len(self.attempts)))
            if self.pulled.get(side):
                m *= 1.12  # extra attacker
            return v * m if m != 1.0 else v
        except Exception:
            return v

    def w_team_defense_suppression(self, team: Any, *a: Any, **kw: Any) -> Any:
        v = self.origs["_team_defense_suppression"](team, *a, **kw)
        fr = _segment_frame()
        if fr is None or not self._mine(fr.f_locals):
            return v
        try:
            side = self._team_side(team)
            if side is None:
                return v
            m = self._mult(_TACTIC_SUP, self._tac(side, len(self.attempts)))
            return v * m if m != 1.0 else v
        except Exception:
            return v

    def w_gm_chance_context(self, *a: Any, **kw: Any) -> Any:
        out = self.origs["_gm_chance_context"](*a, **kw)
        fr = _segment_frame()
        if fr is None:
            return out
        fl = fr.f_locals
        if not self._mine(fl) or not isinstance(out, dict):
            return out
        try:
            side = str(fl.get("side") or "")
            if side not in _SIDES:
                return out
            i = len(self.attempts)
            st = str(fl.get("strength") or "EV").upper()
            ta, td = self._tac(side, i), self._tac(_opp(side), i)
            adj = self._mult(_TACTIC_CTX_ATK, ta) * self._mult(_TACTIC_CTX_DEF, td)
            if st == "PP":
                adj *= _PP_ATK.get(str(ta.get("pp_style")), 1.0) * _PK_DEF.get(str(td.get("pk_style")), 1.0)
            elif st == "SH":
                adj *= _PK_ATK.get(str(ta.get("pk_style")), 1.0) * _PP_RISK.get(str(td.get("pp_style")), 1.0)
            if self.pulled.get(_opp(side)):
                adj *= 2.0  # shooting at an empty net
            if adj == 1.0:
                return out
            out = dict(out)
            out["situational_adj"] = float(out.get("situational_adj", 1.0) or 1.0) * adj
            return out
        except Exception:
            return out

    def w_gm_goalie_save_adjustment(self, g: Any, *a: Any, **kw: Any) -> Any:
        v = self.origs["_gm_goalie_save_adjustment"](g, *a, **kw)
        fr = _segment_frame()
        if fr is None or not self._mine(fr.f_locals):
            return v
        for s in _SIDES:
            if self.pulled[s] and g is not None and g is self.ctx.get(f"{s}_starter"):
                return 0.0  # nobody in net
        return v

    def w_gm_generate_penalties(self, rng: Any, dressed_sk: Any, team_id: Any, ledger: Any, *a: Any, **kw: Any) -> Any:
        recs: List[Tuple[Any, str, int]] = []
        inst = self.sim.__dict__
        prev = inst.get("_gm_ledger_add", _MISSING)
        base_add = getattr(type(self.sim), "_gm_ledger_add").__get__(self.sim) if prev is _MISSING else prev

        def _rec_add(*aa: Any, **kk: Any) -> Any:
            if "pim" in kk and len(aa) >= 3:
                recs.append((aa[1], str(aa[2]), int(kk.get("pim") or 0)))
            return base_add(*aa, **kk)

        inst["_gm_ledger_add"] = _rec_add
        try:
            out = self.origs["_gm_generate_penalties"](rng, dressed_sk, team_id, ledger, *a, **kw)
        finally:
            if prev is _MISSING:
                inst.pop("_gm_ledger_add", None)
            else:
                inst["_gm_ledger_add"] = prev
        try:
            side = "home" if str(team_id) == self.home_id else "away"
            self.ctx[f"{side}_pen"] = out[3]  # live list: the engine truncates it on PP goals
            self.ctx[f"{side}_pen_recs"] = recs
            for p, _tid, _m in recs:
                self.book.add(p, side)
        except Exception:
            logger.warning("live penalty capture failed", exc_info=True)
        return out

    def credit_hook(self, *args: Any, **kwargs: Any) -> Any:
        fr = _segment_frame()
        fl = fr.f_locals if fr is not None else None
        if fl is None or not self._mine(fl):
            return self.orig_credit(*args, **kwargs)
        try:
            cur = self._begin(fl)
        except Exception:
            logger.warning("live attempt hook failed", exc_info=True)
            return self.orig_credit(*args, **kwargs)
        tid = str(kwargs.get("team_id") or "")
        side = str(fl.get("side") or "")
        if side not in _SIDES:
            side = "home" if tid == self.home_id else "away"
        en = bool(self.pulled.get(_opp(side)))
        call_kw = kwargs
        if en and kwargs.get("defending_goalie") is not None:
            call_kw = dict(kwargs)
            call_kw["defending_goalie"] = None  # empty net: no goalie shot/GA credit
        res = self.orig_credit(*args, **call_kw)
        try:
            self._record(fl, cur, kwargs, side, en)
        except Exception:
            logger.warning("live shot capture failed", exc_info=True)
            self.attempts.append({
                "idx": cur["idx"], "seg": cur["seg"], "local_t": cur["local_t"], "total": cur["total"], "side": side,
                "team_id": tid, "outcome": str(kwargs.get("outcome") or ""), "xg": float(kwargs.get("raw_xg") or 0.0),
                "atk_ids": [], "def_ids": [], "assist_ids": [], "broken": True,
            })
        try:
            if str(kwargs.get("outcome") or "") == "GOAL" and bool(fl.get("track_goals", True)):
                self.goals[side] += 1
                for s in _SIDES:
                    if self.pulled[s]:
                        self._goalie_return(s, cur["idx"], cur["seg"], cur["local_t"], cur["total"], "goal")
        except Exception:
            logger.warning("live goal bookkeeping failed", exc_info=True)
        return res

    def _record(self, fl: Any, cur: Dict[str, Any], kw: Dict[str, Any], side: str, en: bool) -> None:
        dside = _opp(side)
        shooter = kw.get("shooter")
        blocker = kw.get("blocker")
        dg = fl.get("def_goalie")
        ctxd = fl.get("ctx")
        ctxd = ctxd if isinstance(ctxd, dict) else {}
        tags = [str(x) for x in (ctxd.get("tags") or [])]
        chance_raw = str(fl.get("chance") or "")
        rebound = bool(fl.get("is_pp_rebound")) or chance_raw.upper() == "REBOUND"
        st = str(kw.get("strength") or "EV").upper()
        sk_h, sk_a = self.skaters_at(cur["seg"], cur["local_t"], st, side)
        rec = {
            "idx": cur["idx"], "seg": cur["seg"], "local_t": cur["local_t"], "total": cur["total"],
            "side": side, "team_id": str(kw.get("team_id") or ""), "opp_team_id": str(kw.get("opp_team_id") or ""),
            "shooter_id": self.book.add(shooter, side) or None,
            "goalie_id": None if en else (self.book.add(dg, dside) or None),
            "blocker_id": (self.book.add(blocker, dside) or None) if blocker is not None else None,
            "atk_ids": [x for x in (self.book.add(p, side) for p in (kw.get("attacking_skaters") or [])) if x],
            "def_ids": [x for x in (self.book.add(p, dside) for p in (kw.get("defending_skaters") or [])) if x],
            "outcome": str(kw.get("outcome") or ""), "xg": float(kw.get("raw_xg") or 0.0),
            "strength_engine": st, "chance_raw": chance_raw, "chance": _contract_chance(chance_raw, tags, rebound),
            "rebound": rebound, "screened": bool(ctxd.get("screened")), "tags": tags,
            "sk_home": sk_h, "sk_away": sk_a, "empty_net": en, "extra_attacker": bool(self.pulled.get(side)),
            "lines": dict(self.line_at.get(cur["idx"]) or {}), "assist_ids": [],
        }
        self.attempts.append(rec)

    # ── after the pass
    def finalize(self, result: Dict[str, Any]) -> None:
        goals = [a for a in self.attempts if a.get("outcome") == "GOAL"]
        se = list(result.get("scoring_events") or [])
        if len(se) == len(goals):
            for g, s in zip(goals, se):
                if str(s.get("scorer_id") or "") != str(g.get("shooter_id") or ""):
                    continue
                g["assist_ids"] = [str(x) for x in (s.get("assist_ids") or []) if x]
                names = list(s.get("assists") or [])
                for j, pid in enumerate(g["assist_ids"]):
                    if pid not in self.book.chips and j < len(names):
                        nm = str(names[j])
                        self.book.chips[pid] = {"player_id": pid, "name": nm, "short": _short_name(nm), "number": None, "pos": "?"}
                        self.book.side.setdefault(pid, g["side"])
        else:
            logger.info("live: scoring_events (%d) != captured goals (%d); assists omitted", len(se), len(goals))
        self.penalties = []
        for side in _SIDES:
            iv = list(self.ctx.get(f"{side}_pen") or [])
            recs = list(self.ctx.get(f"{side}_pen_recs") or [])
            tid = self.home_id if side == "home" else self.away_id
            for j, (start, dur) in enumerate(iv):
                if j < len(recs):
                    p, mins = recs[j][0], int(recs[j][2] or 2)
                else:
                    p, mins = None, max(2, int(round(float(dur) / 60.0)))
                pid = self.book.add(p, side) if p is not None else None
                full = float(mins) * 60.0
                self.penalties.append({
                    "key": f"pen{side}{j}", "side": side, "team_id": tid, "player_id": pid or None,
                    "start": float(start), "full": full, "dur": float(dur), "mins": mins,
                    "ended_by_goal": float(dur) < full - 0.5,
                })

    def pen_sig(self, upto_t: float) -> List[Tuple[Any, ...]]:
        return sorted(
            (p["key"], round(p["start"], 2), p["mins"], p["player_id"]) for p in self.penalties if p["start"] <= upto_t
        )


# ═══ 5. Observers — consume emitted events only; never change the engine result


def _empty_team_stats() -> Dict[str, Any]:
    return {
        "goals": 0, "sog": 0, "attempts": 0, "missed": 0, "blocked": 0, "blocks": 0, "xg": 0.0, "hd_chances": 0,
        "pp_goals": 0, "pp_opps": 0, "pim": 0, "saves": 0, "sv_pct": None, "_ga_goalie": 0,
    }


class _GameState:
    """Real counts from emitted events: team stats, player lines, goalie lines, on-ice."""

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.stats = {"home": _empty_team_stats(), "away": _empty_team_stats()}
        self.p: Dict[str, Dict[str, Any]] = {}
        self.g: Dict[str, Dict[str, int]] = {}
        self.on_ice: Dict[str, List[str]] = {"home": [], "away": []}
        self.on_line: Dict[str, Dict[str, Any]] = {"home": {"F": None, "D": None}, "away": {"F": None, "D": None}}

    def prow(self, pid: str) -> Dict[str, Any]:
        r = self.p.get(pid)
        if r is None:
            r = {"g": 0, "a": 0, "sog": 0, "plus_minus": 0, "ga_on": 0}
            self.p[pid] = r
        return r

    def grow(self, pid: str) -> Dict[str, int]:
        r = self.g.get(pid)
        if r is None:
            r = {"sa": 0, "ga": 0}
            self.g[pid] = r
        return r

    def apply(self, ev: Dict[str, Any], raw: Dict[str, Any]) -> None:
        typ = ev["type"]
        if typ in ("attempt", "goal"):
            a = raw.get("att") or {}
            side = ev["side"]
            dside = _opp(side)
            st = self.stats[side]
            oc = ev.get("outcome")
            st["attempts"] += 1
            st["xg"] = round(st["xg"] + float(ev.get("xg") or 0.0), 4)
            if ev.get("chance") in _HD_CHANCES:
                st["hd_chances"] += 1
            if oc == "BLOCKED":
                st["blocked"] += 1
                self.stats[dside]["blocks"] += 1
            elif oc == "MISSED":
                st["missed"] += 1
            elif oc in ("SAVED", "GOAL"):
                st["sog"] += 1
                if ev.get("shooter_id"):
                    self.prow(ev["shooter_id"])["sog"] += 1
                gid = ev.get("goalie_id")
                if gid and not ev.get("empty_net"):
                    gr = self.grow(gid)
                    gr["sa"] += 1
                    if oc == "SAVED":
                        self.stats[dside]["saves"] += 1
                    else:
                        gr["ga"] += 1
                        self.stats[dside]["_ga_goalie"] += 1
            if oc == "GOAL":
                st["goals"] += 1
                if str(a.get("strength_engine") or "") == "PP":
                    st["pp_goals"] += 1
                if ev.get("shooter_id"):
                    self.prow(ev["shooter_id"])["g"] += 1
                for pid in ev.get("assist_ids") or []:
                    self.prow(pid)["a"] += 1
                if str(a.get("strength_engine") or "EV") != "PP":  # engine's +/- rule
                    for pid in a.get("atk_ids") or []:
                        self.prow(pid)["plus_minus"] += 1
                    for pid in a.get("def_ids") or []:
                        self.prow(pid)["plus_minus"] -= 1
                for pid in a.get("def_ids") or []:
                    self.prow(pid)["ga_on"] += 1
            sv = self.stats[dside]
            faced = sv["saves"] + sv["_ga_goalie"]
            sv["sv_pct"] = round(sv["saves"] / faced, 3) if faced else None
            for s in _SIDES:
                ids = (ev.get("on_ice") or {}).get(s)
                if ids:
                    self.on_ice[s] = list(ids)
            for s, rec in (a.get("lines") or {}).items():
                if s in _SIDES and rec:
                    self.on_line[s] = {"F": rec[0], "D": rec[1]}
        elif typ == "penalty":
            pen = raw.get("pen") or {}
            side = pen.get("side")
            if side in _SIDES:
                self.stats[side]["pim"] += int(pen.get("mins") or 0)
                self.stats[_opp(side)]["pp_opps"] += 1

    def team_stats(self, side: str) -> Dict[str, Any]:
        return {k: v for k, v in self.stats[side].items() if not k.startswith("_")}


class MomentumModel:
    """Rolling pressure model: xG, shots, high-danger chances, goals and penalties drawn,
    exponentially decayed (≈90 s time constant, 3-minute window). + = home."""

    TAU = 90.0
    WINDOW = 180.0

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.value = 0.0
        self.history: List[float] = []
        self.pts: List[Tuple[float, str, float]] = []
        self._last_sample_t = -999.0
        self._zone = 0
        self.swing: Optional[str] = None

    def on_event(self, ev: Dict[str, Any]) -> None:
        typ = ev["type"]
        t = float(ev["t"])
        side: Optional[str] = None
        w = 0.0
        if typ in ("attempt", "goal") and ev.get("side") in _SIDES:
            side = ev["side"]
            oc = ev.get("outcome")
            w = 0.25 + 4.0 * float(ev.get("xg") or 0.0)
            if oc in ("SAVED", "GOAL"):
                w += 0.2
            elif oc == "BLOCKED":
                w -= 0.1
            if ev.get("chance") in _HD_CHANCES:
                w += 0.4
            if typ == "goal":
                w += 1.6
        elif typ == "penalty" and ev.get("side") in _SIDES:
            side, w = _opp(ev["side"]), 0.7
        elif typ == "period_start":
            self.pts = [(pt, s, ww * 0.5) for pt, s, ww in self.pts]
        if side:
            self.pts.append((t, side, max(0.05, w)))
        self.recompute(t)

    def calm(self, factor: float = 0.5) -> None:
        self.pts = [(pt, s, ww * factor) for pt, s, ww in self.pts]

    def recompute(self, t: float) -> None:
        self.pts = [p for p in self.pts if t - p[0] <= self.WINDOW]
        H = sum(w * math.exp(-(t - pt) / self.TAU) for pt, s, w in self.pts if s == "home")
        A = sum(w * math.exp(-(t - pt) / self.TAU) for pt, s, w in self.pts if s == "away")
        self.value = max(-1.0, min(1.0, (H - A) / (H + A + 2.0)))
        last = self.history[-1] if self.history else 0.0
        if t - self._last_sample_t >= 20.0 or abs(self.value - last) > 0.25:
            self.history.append(round(self.value, 3))
            self.history = self.history[-30:]
            self._last_sample_t = t
        zone = 1 if self.value > 0.35 else (-1 if self.value < -0.35 else 0)
        self.swing = None
        if zone != 0 and zone != self._zone:
            if t > 120.0:
                self.swing = "home" if zone > 0 else "away"
            self._zone = zone

    def label(self, lang: str) -> str:
        lab = _MOMENTUM_LABEL.get(lang) or _MOMENTUM_LABEL["en"]
        if self.value > 0.25:
            return lab[0]
        if self.value < -0.25:
            return lab[2]
        return lab[1]


class CrowdModel:
    """Home crowd. Level = f(fan meter, rivalry tier, score state, momentum, recent events).
    Presentation only — CROWD_HOME_LIFT_ENABLED stays False (never feeds the engine)."""

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.rng = random.Random(int(loop.lg.seed) ^ 0xC40D)
        self.level = 0.35
        self.mood = "buzz"
        self.chant: Optional[str] = None
        self._chant_until = -1.0
        self._spike = (0.0, 0.0)
        self._mood_force: Optional[Tuple[str, float]] = None

    def _set_chant(self, key: str, t: float, ttl: float = 30.0, **params: Any) -> None:
        lang = self.loop.lg.lang
        opts = _loc(_CHANTS, lang, key, []) or []
        if not opts:
            return
        base = {"team": self.loop.team_nick("home"), "goalie": self.loop.book.short(self.loop.current_goalie("home")),
                "rival": self.loop.team_nick("away")}
        base.update(params)
        self.chant = _fmt(self.rng.choice(opts), base)
        self._chant_until = t + ttl

    def on_event(self, ev: Dict[str, Any]) -> None:
        loop = self.loop
        t = float(ev["t"])
        typ = ev["type"]
        side = ev.get("side")
        tier = int(loop.rivalry.get("tier") or 0)
        if typ == "goal":
            if side == "home":
                self._spike = (0.45, t)
                self._mood_force = ("roar", t + 25.0)
                diff = int(ev["score_after"]["home"]) - int(ev["score_after"]["away"])
                if diff == 0:
                    key = "equal"
                elif diff >= 3 and int(ev.get("period") or 1) >= 2:
                    key = "more"
                elif tier >= 2 and self.rng.random() < 0.4:
                    key = "rival"
                else:
                    key = "team"
                self._set_chant(key, t)
            else:
                self._spike = (-0.25, t)
                self._mood_force = ("boo" if tier >= 2 else "quiet", t + 25.0)
                self.chant = None
        elif typ == "attempt" and side == "away" and ev.get("outcome") == "SAVED" and int(ev.get("importance") or 1) >= 3:
            self._spike = (0.2, t)
            if ev.get("goalie_id"):
                self._set_chant("goalie", t, 20.0, goalie=loop.book.short(ev.get("goalie_id")))
        elif typ == "penalty":
            if side == "home":
                self._mood_force = ("boo", t + 12.0)
            else:
                self._spike = (0.12, t)
                self._set_chant("shoot", t, 40.0)
        elif typ == "coach" and (ev.get("meta") or {}).get("action") == "pull" and side == "home":
            self._set_chant("team", t, 40.0)
        elif typ == "final":
            hs, as_ = int(ev["score_after"]["home"]), int(ev["score_after"]["away"])
            self._mood_force = ("roar" if hs > as_ else "quiet", t + 9999.0)
            if hs > as_:
                self._set_chant("team", t, 9999.0)
            else:
                self.chant = None
        self.recompute(t)
        if self.chant is None and self.level > 0.6 and self.rng.random() < 0.15:
            self._set_chant(self._context_chant(t, tier), t)

    def _context_chant(self, t: float, tier: int) -> str:
        """Chant that fits the score, period and rivalry (no 'one more goal' while leading early)."""
        loop = self.loop
        diff = int(loop.lg.score_home) - int(loop.lg.score_away)
        period, left = loop.clock_at(t)
        if period >= 3 and left <= 300.0:
            if diff in (-1, 0):
                return "one_more"
            if diff == 1:
                return "defense" if self.rng.random() < 0.6 else "hold"
        if diff >= 3 and period >= 2:
            return "more"
        if tier >= 1 and self.rng.random() < 0.35:
            return "rival"
        return "team"

    def recompute(self, t: float) -> None:
        loop = self.loop
        fan = loop.fan_meter
        fan01 = (float(fan) / 100.0) if fan is not None else 0.55
        tier = int(loop.rivalry.get("tier") or 0)
        diff = int(loop.lg.score_home) - int(loop.lg.score_away)
        period, left = loop.clock_at(t)
        late = period >= 3 and left <= 300.0 and not loop.finished
        level = 0.22 + 0.33 * fan01 + 0.07 * tier
        level += 0.06 if diff > 0 else (-0.04 if diff < 0 else 0.0)
        if late and abs(diff) <= 1:
            level += 0.10
        level += 0.12 * float(loop.momentum.value)
        sv, st = self._spike
        level += sv * math.exp(-max(0.0, t - st) / 45.0)
        self.level = max(0.03, min(1.0, level))
        if self._mood_force and t < self._mood_force[1]:
            self.mood = self._mood_force[0]
        else:
            self._mood_force = None
            if late and diff < 0 and self.level < 0.55:
                self.mood = "restless"
            elif self.level < 0.28:
                self.mood = "quiet"
            elif self.level < 0.48:
                self.mood = "buzz"
            elif self.level < 0.7:
                self.mood = "loud"
            else:
                self.mood = "roar"
        if self.chant and t > self._chant_until:
            self.chant = None


class AlertEngine:
    """Bench alerts for the user (localized). `action` is a ready /coach body when sensible."""

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.items: List[Dict[str, Any]] = []
        self._cool: Dict[str, float] = {}
        self._against: List[float] = []
        self._n = 0
        self._last_tick = -999.0

    def add(self, kind: str, severity: str, params: Dict[str, Any], t: float, seq: int, *,
            player_id: Optional[str] = None, action: Optional[Dict[str, Any]] = None, ttl: float = 12.0,
            cooldown: float = 90.0, key: str = "", text_key: Optional[str] = None) -> None:
        ck = f"{kind}:{player_id or ''}:{key}"
        if t - self._cool.get(ck, -1e9) < cooldown:
            return
        self._cool[ck] = t
        self._n += 1
        lang = self.loop.lg.lang
        self.items.append({
            "id": f"al{seq}-{self._n}", "seq": seq, "t": round(t, 2), "severity": severity, "kind": kind,
            "text": _fmt(_loc(_ALERT_I18N, lang, text_key or kind), params), "player_id": player_id,
            "action": action, "ttl_sec": ttl,
        })
        self.items = self.items[-60:]

    def _pname(self, pid: Optional[str]) -> str:
        return self.loop.book.short(pid) or self.loop.book.name(pid)

    def on_event(self, ev: Dict[str, Any]) -> None:
        loop = self.loop
        user = loop.lg.user_side
        t = float(ev["t"])
        seq = int(ev["seq"])
        typ = ev["type"]
        side = ev.get("side")
        lang = loop.lg.lang
        if typ == "goal":
            self.add("goal", "high", {"team": loop.team_say(side, lang), "player": self._pname(ev.get("shooter_id"))},
                     t, seq, player_id=ev.get("shooter_id"), cooldown=0.0, key=str(seq))
            if side != user:
                self._against = [x for x in self._against if t - x <= 180.0] + [t]
                if len(self._against) >= 2 and not loop.timeout_used and int(ev["period"]) <= 3:
                    self.add("timeout_suggest", "med", {}, t, seq, action={"kind": "timeout", "payload": {}}, ttl=30.0, cooldown=300.0)
                for pid in (ev.get("on_ice") or {}).get(user) or []:
                    pr = loop.state.p.get(pid) or {}
                    if int(pr.get("plus_minus", 0)) <= -2:
                        pm = int(pr.get("plus_minus", 0))
                        self.add("cold", "low", {"player": self._pname(pid), "pm": f"{pm:+d}"}, t, seq, player_id=pid,
                                 action={"kind": "bench_player", "payload": {"player_id": pid}}, ttl=20.0, cooldown=600.0, key=str(pm))
            for pid in [ev.get("shooter_id")] + list(ev.get("assist_ids") or []):
                self._hot(pid, t, seq)
        if typ in ("attempt", "goal"):
            if ev.get("outcome") in ("SAVED", "GOAL"):
                self._hot(ev.get("shooter_id"), t, seq)
                gid = ev.get("goalie_id")
                if gid:
                    gr = loop.state.g.get(gid) or {"sa": 0, "ga": 0}
                    sa, ga = int(gr["sa"]), int(gr["ga"])
                    svp = (sa - ga) / sa if sa else 1.0
                    if ga >= 3 or (sa >= 10 and svp < 0.850):
                        gside = _opp(side) if side in _SIDES else None
                        act = {"kind": "timeout", "payload": {}} if gside == user and not loop.timeout_used else None
                        self.add("goalie", "med" if gside == user else "low",
                                 {"player": self._pname(gid), "ga": ga, "svp": f"{svp:.3f}".lstrip("0")},
                                 t, seq, player_id=gid, action=act, ttl=20.0, cooldown=900.0, key=str(ga))
        elif typ == "penalty":
            meta = ev.get("meta") or {}
            label = meta.get("label") or ""
            self.add("penalty", "med", {"player": meta.get("player_short") or meta.get("player_name") or "", "label": label},
                     t, seq, player_id=meta.get("player_id"), cooldown=0.0, key=str(seq))
            pp_side = _opp(side) if side in _SIDES else None
            if pp_side == user:
                self.add("pp", "med", {"team": loop.team_say(user, lang), "label": label}, t, seq, ttl=20.0, cooldown=0.0, key=str(seq))
            elif pp_side:
                self.add("pp", "low", {"team": loop.team_say(user, lang), "label": label}, t, seq, ttl=20.0, cooldown=0.0,
                         key=str(seq), text_key="pk")
        if loop.momentum.swing and loop.momentum.swing != user:
            act = {"kind": "timeout", "payload": {}} if not loop.timeout_used else None
            self.add("momentum", "med", {"team": loop.team_say(loop.momentum.swing, lang)}, t, seq, action=act, ttl=20.0, cooldown=240.0)

    def _hot(self, pid: Optional[str], t: float, seq: int) -> None:
        if not pid:
            return
        loop = self.loop
        pr = loop.state.p.get(pid) or {}
        pts = int(pr.get("g", 0)) + int(pr.get("a", 0))
        sog = int(pr.get("sog", 0))
        if pts < 2 and sog < 4:
            return
        act = None
        if loop.book.side.get(pid) == loop.lg.user_side:
            li = loop.line_index_of(pid)
            if li is not None:
                act = {"kind": "next_line", "payload": {"line": li}}
        self.add("hot", "low", {"player": self._pname(pid), "pts": pts, "sog": sog}, t, seq, player_id=pid, action=act,
                 ttl=15.0, cooldown=1e9, key=f"{min(pts, 9)}-{min(sog // 4, 3)}")

    def tick(self, t: float) -> None:
        loop = self.loop
        if t - self._last_tick < 10.0 or loop.finished or loop.intermission:
            return
        self._last_tick = t
        user = loop.lg.user_side
        seq = int(loop.lg.seq)
        lang = loop.lg.lang
        for pid in loop.state.on_ice.get(user) or []:
            f = loop.shifts.fatigue(pid, t)
            if f >= 0.8:
                cur = loop.state.on_line.get(user) or {}
                F = cur.get("F")
                nxt = f"F{((int(F) + 1) % 4) + 1}" if F is not None else "F2"
                self.add("fatigue", "med" if f >= 0.9 else "low", {"player": self._pname(pid)}, t, seq, player_id=pid,
                         action={"kind": "next_line", "payload": {"line": nxt}}, ttl=15.0, cooldown=120.0)
        period, left = loop.clock_at(t)
        mine = loop.lg.score_home if user == "home" else loop.lg.score_away
        theirs = loop.lg.score_away if user == "home" else loop.lg.score_home
        n = int(theirs) - int(mine)
        if period == 3 and n in (1, 2) and left <= (120.0 if n == 1 else 180.0) and not loop.goalie_pulled.get(user):
            mp = loop.manpower(t)
            sk_me = mp["home_skaters"] if user == "home" else mp["away_skaters"]
            sk_them = mp["away_skaters"] if user == "home" else mp["home_skaters"]
            if sk_me >= sk_them:
                self.add("pull_suggest", "high", {"n": n}, t, seq, action={"kind": "pull_goalie", "payload": {"pull": True}},
                         ttl=30.0, cooldown=1e9, key=str(n))
        _ = lang

    def active(self, now: float) -> List[Dict[str, Any]]:
        return [a for a in self.items if now - float(a["t"]) <= float(a["ttl_sec"]) or a["kind"] in ("goal",)][-12:]


def _fr_elide(s: str) -> str:
    import re

    return re.sub(r"\b([dD])e ([AEIOUYHÉÈÊÂÎaeiouyhéèêâî])", r"\1'\2", re.sub(r"\b([lL])e ([AEIOUYÉÈaeiouyéè])", r"\1'\2", s))


class CommentaryWriter:
    """Two-person booth (pbp + color). Original text only; persona is a label.

    Pacing (game seconds): goals (priority 3) are called immediately; other lines keep >= 6 s
    apart; priority-2 lines are delayed to the next free slot (dropped if > 10 s late);
    when nothing has been said for ~20 s a filler line (situation or a real stat) is added.
    Colour lines appear at most once per ~3 pbp lines and only quote numbers that exist.
    """

    PERSONA = COMMENTATOR_PERSONA
    MIN_GAP = 6.0
    FILL_GAP = 15.0

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.rng = random.Random(int(loop.lg.seed) ^ 0xC033)
        self.lines: List[Dict[str, Any]] = []
        self._last_t = -999.0
        self._recent: List[str] = []
        self._recent_lines: List[List[str]] = []
        self._said: set = set()
        self._last_momentum_t = -999.0
        self._pbp_since_color = 3
        self._n = 0
        self._att_sides: List[str] = []
        self._run = {"side": None, "n": 0}
        self._period_sog: Dict[int, Dict[str, int]] = {}
        self._pk = {"home": [0, 0], "away": [0, 0]}  # [killed, faced] — completed penalties only
        self._last_poss: Optional[str] = None
        # booth stories / pacing
        self._fam: Dict[str, Tuple[int, Any]] = {}       # colour stat family -> (period, value) last said
        self._last_fill_t = -999.0
        self._last_imp2_t = 0.0
        self.story_q: List[Dict[str, Any]] = []           # pending story lines {due, text, speaker, key, topic}
        self._story_meta: Optional[Dict[str, Any]] = None
        self._interrupted: Optional[Dict[str, Any]] = None
        self._cutin = False
        self.next_story_t = 70.0 + self.rng.uniform(0.0, 50.0)
        self._facts: Optional[List[Dict[str, Any]]] = None
        self._last_story_cat: Optional[str] = None
        self.stories_told = 0
        self._used_lines: set = set()
        self._fam_counts: Dict[str, int] = {}
        self._told_game: set = set()
        self._last_family: Optional[str] = None
        self._cat_counts: Dict[str, int] = {}
        self._last_pt = -999.0
        self._last_nar_pt = -999.0
        self._last_fo_pt = -999.0
        self._last_rule_pt = -999.0
        self._last_lc_pt: Dict[str, float] = {"home": -999.0, "away": -999.0}
        self._bu_done: set = set()

    # ── template choice: never reuse one of the last 8 templates
    def _pick(self, lang: str, key: str) -> Tuple[str, str]:
        opts = _loc(COMMENTARY_I18N, lang, key, []) or []
        if not opts:
            return "", ""
        order = list(range(len(opts)))
        self.rng.shuffle(order)
        ban = getattr(self, "_ban_tids", None) or set()
        if key.startswith("fill_"):
            order = [i for i in order if f"{key}#{i}" not in ban]
            if not order:
                return "", ""
        for idx in order:
            tid = f"{key}#{idx}"
            if tid not in self._recent:
                return opts[idx], tid
        oldest = min(order, key=lambda i: self._recent.index(f"{key}#{i}") if f"{key}#{i}" in self._recent else -1)
        return opts[oldest], f"{key}#{oldest}!"

    def _fresh(self, lang: str, parts: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> bool:
        for key, _p, _q in parts:
            opts = _loc(COMMENTARY_I18N, lang, key, []) or []
            if opts and all(f"{key}#{i}" in self._recent for i in range(len(opts))):
                return False
        return True

    def _render(self, lang: str, parts: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> Tuple[str, str, List[str]]:
        texts, speaks, tids = [], [], []
        for key, params, tts_params in parts:
            tpl, tid = self._pick(lang, key)
            if not tpl:
                continue
            tids.append(tid)
            texts.append(_fmt(tpl, params))
            speaks.append(_fmt(tpl, {**params, **tts_params}))
        return " ".join(x for x in texts if x).strip(), " ".join(x for x in speaks if x).strip(), tids

    # ── presentation time (pt): lines are placed and paced on the presentation clock
    def _pt(self, gc: float) -> float:
        mt = getattr(self.loop, "motion", None)
        if mt is not None and hasattr(mt, "pt_at"):
            try:
                return float(mt.pt_at(float(gc)))
            except Exception:
                logger.debug("pt_at failed", exc_info=True)
        return float(gc)

    def _gc(self, pt: float, fallback: float) -> float:
        mt = getattr(self.loop, "motion", None)
        if mt is not None and hasattr(mt, "gc_at"):
            try:
                g = float(mt.gc_at(float(pt)))
                return min(max(g, fallback), fallback + 12.0)
            except Exception:
                pass
        return fallback

    def _pt_free(self, pt: float, before: float, after: float) -> bool:
        for c in self.lines[-40:]:
            cp = float(c.get("pt", c["t"]))
            if pt - before < cp < pt + after:
                return False
        return True

    @staticmethod
    def _hint(profile: str, priority: int, speaker: str, kind: str) -> Dict[str, Any]:
        energy = {"hype": 0.95, "build": 0.72, "calm": 0.42, "somber": 0.3}.get(profile, 0.45)
        if kind == "story":
            energy = min(energy, 0.38)
        elif kind == "buildup":
            energy = 0.7
        elif kind in ("narration", "rule"):
            energy = 0.5 if kind == "narration" else 0.48
        if speaker == "color":
            energy = max(0.2, energy - 0.08)
        pace = "fast" if (profile == "hype" or kind == "buildup") else ("slow" if kind in ("story", "replay") or profile == "somber" else "normal")
        if priority >= 3 and profile == "hype":
            pause = 0
        elif kind in ("buildup", "payoff"):
            pause = 120
        elif kind in ("story", "replay", "color"):
            pause = 600
        else:
            pause = 300
        return {"energy": round(energy, 2), "pace": pace, "pause_before_ms": pause}

    def caption_tail(self) -> List[Dict[str, Any]]:
        return sorted(self.lines[-80:], key=lambda c: float(c.get("pt", c["t"])))

    def say(self, t: float, seq: int, parts: List[Tuple[str, Dict[str, Any], Dict[str, Any]]], priority: int,
            profile: str, speaker: str = "pbp", kind: str = "event", *, pt: Optional[float] = None, gap: Optional[float] = None,
            force_color: bool = False, hint: Optional[Dict[str, Any]] = None) -> bool:
        if not LIVE_COMMENTARY_ENABLED or not parts:
            return False
        loop = self.loop
        lt = float(t)
        base_pt = float(pt) if pt is not None else self._pt(lt)
        lpt = base_pt
        if self.story_q and kind != "story" and (priority < 2 or (speaker == "color" and priority < 3)):
            return False  # let a running story breathe; routine plays and stat asides are skipped
        mg = self.MIN_GAP if gap is None else float(gap)
        if priority >= 3:
            pass
        elif lpt < self._last_pt - 0.01:  # placed behind the newest line (derived events / narration): needs a free slot
            if not self._pt_free(lpt, mg, min(mg, 4.0)):
                return False
        elif lpt - self._last_pt < mg:
            if priority < 2:
                return False
            lpt = self._last_pt + mg
            if lpt - base_pt > 10.0:
                return False
            lt = self._gc(lpt, lt)
        if speaker == "color" and self._pbp_since_color < 2 and not force_color:
            return False
        lang = loop.lg.lang
        if (priority <= 1 or speaker == "color") and not self._fresh(lang, parts):
            return False  # never reuse a template from the last 8 lines for optional lines
        text, speak, tids = self._render(lang, parts)
        if not text:
            return False
        if self._cutin and speaker == "pbp" and priority >= 2:
            ci = self.rng.choice(_booth().CUTIN.get(lang) or _booth().CUTIN["en"])
            text, speak = f"{ci} {text}", f"{ci} {speak}"
            self._cutin = False
        if lang == "fr":
            text, speak = _fr_elide(text), _fr_elide(speak)
        self._recent_lines = (self._recent_lines + [[x.rstrip("!") for x in tids]])[-8:]
        self._recent = [x for ln in self._recent_lines for x in ln]  # template ids used in the last 8 lines
        if speaker == "color":
            self._pbp_since_color = 0
        else:
            self._pbp_since_color += 1
        self._last_t = max(self._last_t, lt)
        self._last_pt = max(self._last_pt, lpt)
        self._n += 1
        tts = _speakable(speak, lang)
        if speaker == "color" and profile == "hype":
            profile = "build"
        pr = int(max(1, min(3, priority)))
        line = {
            "id": f"c{seq}-{self._n}", "seq": seq, "t": round(lt, 2), "pt": round(lpt, 2), "priority": pr,
            "profile": profile, "speaker": speaker, "kind": kind, "lang": lang, "text": text, "tts_text": tts,
            "tts_hint": dict(hint or self._hint(profile, pr, speaker, kind)), "audio_key": None, "audio_url": None,
            "real_ts": int(time.time() * 1000), "expires_at_t": round(lt + 20.0, 2), "expires_at_pt": round(lpt + 20.0, 2),
        }
        _tts_attach(line)
        self.lines.append(line)
        self.lines = self.lines[-400:]
        return True

    # ── helpers
    def _team(self, side: Optional[str]) -> str:
        return self.loop.team_say(side, self.loop.lg.lang) if side in _SIDES else ""

    def _score_phrase(self, h: int, a: int, scorer: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        lang = self.loop.lg.lang
        if h == a:
            tpl, _ = self._pick(lang, "score_tie")
            p = {"score": _score_text(h, a), "tie": str(h)}
            q = {"score": _score_tts(h, a, lang), "tie": _num_word(h, lang)}
        else:
            lead = "home" if h > a else "away"
            hi, lo = max(h, a), min(h, a)
            key = "score_lead"
            team_side = lead
            if scorer in _SIDES:
                if scorer != lead:
                    key = "score_within" if hi - lo == 1 else "score_cut"
                    team_side = scorer
                elif hi - lo == 1:
                    key = "score_ahead"
            tpl, _ = self._pick(lang, key)
            p = {"score": _score_text(hi, lo), "team": self._team(team_side)}
            q = {"score": _score_tts(hi, lo, lang)}
        text = _fmt(tpl, p)
        speak = _fmt(tpl, {**p, **q})
        return ({"score_phrase": text, "Score_phrase": _cap(text)}, {"score_phrase": speak, "Score_phrase": _cap(speak)})

    def _fam_ok(self, fam: str, period: int, value: Any, *, min_change: float = 0.0) -> bool:
        """Each analyst stat family at most once per period unless its value moved meaningfully."""
        last = self._fam.get(fam)
        if last is not None and len(last) > 2 and self.loop.game_sec_elapsed - float(last[2]) < 600.0:
            return False  # at most one line per stat family per ~10 game minutes
        if last is None or last[0] != period:
            return True
        try:
            return min_change > 0 and abs(float(value) - float(last[1])) >= min_change
        except (TypeError, ValueError):
            return False

    def _color_after(self, ev: Dict[str, Any], t: float, seq: int) -> None:
        """At most one real-stat colour line after notable events (per-family cooldowns)."""
        loop = self.loop
        lang = loop.lg.lang
        st = loop.state
        typ = ev["type"]
        side = ev.get("side")
        period = int(ev.get("period") or 1)
        cands: List[Tuple[str, Dict[str, Any], Dict[str, Any], Any, float]] = []
        if typ == "goal":
            total = int(ev["score_after"]["home"]) + int(ev["score_after"]["away"])
            if total == 1:
                cands.append(("c_first", {}, {}, 1, 0.0))
            if self._run["n"] >= 2:
                n = self._run["n"]
                cands.append(("c_run", {"n": n, "team": self._team(side)}, {"n": _num_word(n, lang)}, n, 1.0))
            for pid in [ev.get("shooter_id")] + list(ev.get("assist_ids") or []):
                pr = st.p.get(pid or "") or {}
                pts = int(pr.get("g", 0)) + int(pr.get("a", 0))
                if pts >= 2:
                    cands.append(("c_points", {"player": loop.book.short(pid), "pts": pts}, {"pts": _num_word(pts, lang)}, f"{pid}:{pts}", 0.0))
                    break
            if ev.get("strength") == "PP" and side in _SIDES:
                s_ = st.stats[side]
                done = max(1, int(s_["pp_opps"]))
                cands.append(("c_pp", {"team": self._team(side), "g": s_["pp_goals"], "o": done},
                              {"g": _num_word(s_["pp_goals"], lang), "o": _num_word(done, lang)}, s_["pp_goals"], 1.0))
        elif typ == "attempt" and ev.get("outcome") == "SAVED":
            gid = ev.get("goalie_id")
            gr = st.g.get(gid or "") or {}
            sa, ga = int(gr.get("sa", 0)), int(gr.get("ga", 0))
            if gid and sa >= 8 and int(ev.get("importance") or 1) >= 2:
                cands.append(("c_goalie", {"goalie": loop.book.short(gid), "sv": sa - ga, "sa": sa}, {}, sa, 10.0))
            sog = int((st.p.get(ev.get("shooter_id") or "") or {}).get("sog", 0))
            if 3 <= sog <= 10:
                nth = _ORDINAL.get(lang, _ORDINAL["en"])[sog]
                cands.append(("c_shots", {"nth": nth, "Nth": _cap(nth), "player": loop.book.short(ev.get("shooter_id"))}, {}, sog, 2.0))
        elif typ == "penalty_end" and side in _SIDES:
            k, n = self._pk[side]
            act = loop._cap is not None and any(p["side"] == side and p["start"] <= t < p["start"] + p["dur"] for p in loop._cap.penalties)
            if n >= 2 and not act:
                cands.append(("c_pk", {"team": self._team(side), "k": k, "n": n}, {"k": _num_word(k, lang), "n": _num_word(n, lang)}, n, 1.0))
        elif typ == "period_end":
            ps = self._period_sog.get(period) or {}
            if ps and period < 3:
                lead = "home" if ps.get("home", 0) >= ps.get("away", 0) else "away"
                cands.append(("c_period_shots", {"team": self._team(lead), "opp": self._team(_opp(lead)),
                                                 "a": ps.get(lead, 0), "b": ps.get(_opp(lead), 0)}, {}, period, 0.0))
        cands = [c for c in cands if self._fam_ok(c[0], period, c[3], min_change=c[4])]
        if cands:
            key, p, q, val, _mc = cands[0] if typ == "goal" else self.rng.choice(cands)
            cpt = (float(ev["pt"]) + 3.0) if ev.get("pt") is not None else None
            if self.say(t + 3.0, seq, [(key, p, q)], 2, "calm", speaker="color", kind="color", pt=cpt):
                self._fam[key] = (period, val, self.loop.game_sec_elapsed)

    def _track(self, ev: Dict[str, Any]) -> None:
        typ = ev["type"]
        side = ev.get("side")
        if typ in ("attempt", "goal") and side in _SIDES:
            self._att_sides = (self._att_sides + [side])[-11:]
            if ev.get("outcome") in ("SAVED", "GOAL"):
                row = self._period_sog.setdefault(int(ev["period"]), {"home": 0, "away": 0})
                row[side] += 1
        if typ == "goal" and side in _SIDES:
            if self._run["side"] == side:
                self._run["n"] += 1
            else:
                self._run = {"side": side, "n": 1}
        if typ == "penalty_end" and side in _SIDES:
            self._pk[side][1] += 1
            if (ev.get("meta") or {}).get("reason") == "expired":
                self._pk[side][0] += 1

    def on_event(self, ev: Dict[str, Any]) -> None:
        self._track(ev)
        loop = self.loop
        lang = loop.lg.lang
        t = float(ev["t"])
        seq = int(ev["seq"])
        typ = ev["type"]
        imp = int(ev.get("importance") or 1)
        if imp >= 2 and typ in ("attempt", "goal", "penalty", "coach", "shootout_attempt"):
            self._last_imp2_t = t
            if self.story_q and (imp >= 3 or typ in ("goal", "penalty", "coach")):
                self._interrupt(t)
        if typ == "final":
            self._interrupted = None
            self.story_q = []
            self.lines = [c for c in self.lines if c["t"] <= t + 1e-6 or c.get("kind") == "event"]
            self._last_t = min(self._last_t, t)
        side = ev.get("side")
        meta = ev.get("meta") or {}
        book = loop.book
        sh = book.short(ev.get("shooter_id"))
        gl = book.short(ev.get("goalie_id"))
        team = self._team(side)
        chance = _loc(_CHANCE_PHRASE, lang, ev.get("chance") or "", "")
        said = False
        ept = float(ev["pt"]) if ev.get("pt") is not None else None
        bu = False
        if typ in ("attempt", "goal") and not ev.get("empty_net"):
            bu = self._buildup_call(ev, ept)
        if typ in ("faceoff", "stoppage", "rule", "line_change", "delayed_penalty", "hit"):
            if typ == "faceoff" and meta.get("reason") == "period_start" and int(ev["period"]) == 1:
                self.say(t, seq, [("faceoff_open", {}, {})], 2, "calm", pt=ept)
            elif ev.get("authority") == "derived":
                self._rule_call(ev, ept)
            return
        elif typ == "period_start":
            p = int(ev["period"])
            if p == 5:
                said = self.say(t, seq, [("so_start", {}, {})], 2, "build")
            elif p >= 2:
                nm = _loc(_PERIOD_NAME, lang, p)
                said = self.say(t, seq, [("period_start", {"period": nm, "Period": _cap(nm)}, {})], 2, "calm")
        elif typ == "period_end":
            nm = _loc(_PERIOD_NAME, lang, int(ev["period"]))
            said = self.say(t, seq, [("period_end", {"period": nm, "Period": _cap(nm)}, {})], 2, "calm")
        elif typ == "attempt":
            oc = ev.get("outcome")
            imp = int(ev.get("importance") or 1)
            blk = book.short(ev.get("blocker_id"))
            if ev.get("empty_net") and oc == "SAVED":
                said = self.say(t, seq, [("en_miss", {"shooter": sh}, {})], 2, "build", pt=ept)
            elif bu and oc in ("SAVED", "MISSED", "BLOCKED"):  # payoff of the build-up call
                key = {"SAVED": "pay_save_big" if imp >= 3 else "pay_save", "MISSED": "pay_miss", "BLOCKED": "pay_block"}[oc]
                shot_p = self._shot_phrase(ev)
                if shot_p and self.rng.random() < 0.6:
                    key = key.replace("pay_", "pay_typed_")
                said = self.say(t, seq, [(key, {"goalie": gl, "shooter": sh, "blocker": blk, "Shot": shot_p}, {})], max(2, imp),
                                "build" if imp >= 3 else "calm", kind="payoff", pt=ept, gap=1.6)
            elif oc == "SAVED":
                said = self.say(t, seq, [("save_big" if imp >= 3 else "save", {"goalie": gl, "shooter": sh, "chance": chance}, {})],
                                imp, "build" if imp >= 3 else "calm", pt=ept)
            elif oc == "MISSED":
                hd = ev.get("chance") in _HD_CHANCES
                said = self.say(t, seq, [("miss_hd" if hd else "miss", {"shooter": sh}, {})], 2 if hd else 1, "calm", pt=ept)
            elif oc == "BLOCKED":
                said = self.say(t, seq, [("block", {"blocker": blk, "shooter": sh}, {})], 1, "calm", pt=ept)
            if oc == "SAVED" and imp >= 3 and said and ept is not None:
                self._save_replay(ev, ept)
        elif typ == "goal":
            h, a = int(ev["score_after"]["home"]), int(ev["score_after"]["away"])
            sp, sq = self._score_phrase(h, a, side)
            stren = ev.get("strength")
            said = self.say(t, seq, self._goal_parts(ev, sp, sq), 3, "hype", pt=ept, gap=1.6)
            if said and ept is not None:
                self._goal_replay(ev, ept)
        elif typ == "penalty":
            pp_side = _opp(side) if side in _SIDES else None
            parts = [("penalty", {"player": meta.get("player_short") or "", "label": meta.get("label") or ""}, {})]
            if pp_side:
                parts.append(("pp_start", {"team": self._team(pp_side)}, {}))
            said = self.say(t, seq, parts, 2, "calm")
        elif typ == "penalty_end":
            if meta.get("reason") == "expired" and meta.get("manpower_restored"):
                said = self.say(t, seq, [("pk_kill", {"team": team}, {})], 1, "calm")
        elif typ == "coach":
            act = meta.get("action")
            key = {"pull": "coach_pull", "return": "coach_return", "timeout": "coach_timeout"}.get(str(act), "coach_change")
            said = self.say(t, seq, [(key, {"team": team, "what": meta.get("what") or ""}, {})], 2 if act != "return" else 1,
                            "build" if act == "pull" else "calm")
        elif typ == "shootout_attempt":
            said = self.say(t, seq, [("so_goal" if ev.get("outcome") == "GOAL" else "so_save", {"shooter": sh, "goalie": gl}, {})],
                            3 if ev.get("outcome") == "GOAL" else 2, "build")
        elif typ == "final":
            h, a = int(ev["score_after"]["home"]), int(ev["score_after"]["away"])
            win = "home" if h > a else "away"
            hi, lo = max(h, a), min(h, a)
            said = self.say(t, seq, [("final", {"team": self._team(win), "score": _score_text(hi, lo)}, {"score": _score_tts(hi, lo, lang)})],
                            3, "hype" if win == loop.lg.user_side else "somber")
        self._cutin = False
        inter = typ == "period_end" and int(ev["period"]) < 3 or (typ == "period_end" and int(ev["period"]) == 3 and (loop._result or {}).get("overtime"))
        if (said and typ in ("goal", "attempt", "period_end")) or typ == "penalty_end":
            self._color_after(ev, t, seq)
        if inter:
            self._intermission_segment(t, seq)


    # ── round 5: build-up calls, goal-call structure, replays, rule calls, light play narration
    def _lane(self, side: Optional[str], gc: float, y: float) -> str:
        lang = self.loop.lg.lang
        mt = self.loop.motion
        d = mt.dir(side, gc) if hasattr(mt, "dir") and side in _SIDES else 1.0
        row = _LC.LANE.get(lang, _LC.LANE["en"]) if _LC is not None else {"l": "left", "r": "right"}
        return row["l"] if y * d > 0 else row["r"]

    def _buildup_call(self, ev: Dict[str, Any], ept: Optional[float]) -> bool:
        """PBP talks through the play 2-4 s BEFORE a dangerous shot, from the motion track's plan."""
        mt = getattr(self.loop, "motion", None)
        key = (ev.get("meta") or {}).get("event_key")
        if ept is None or mt is None or not hasattr(mt, "buildup") or not key or key in self._bu_done:
            return False
        info = mt.buildup(key)
        if not info or not info.get("shooter") or info.get("shooter") != ev.get("shooter_id"):
            return False
        imp = int(ev.get("importance") or 1)
        dangerous = ev["type"] == "goal" or imp >= 2 or ev.get("chance") in ("breakaway", "high_danger", "slot", "rebound")
        if not dangerous and self.rng.random() > 0.3:
            return False
        if self.story_q and imp < 2 and ev["type"] != "goal":
            return False
        bpt = ept - self.rng.uniform(2.2, 3.6)
        if bpt - self._last_pt < 3.0 or not self._pt_free(bpt, 3.0, 1.2):
            return False
        book = self.loop.book
        sh = book.short(info["shooter"])
        ps = book.short(info.get("passer")) if info.get("passer") else ""
        seg = info.get("seg") or {}
        y = float((seg.get("p1") or (0.0, 0.0))[1])
        lane = self._lane(ev.get("side"), float(ev["t"]), y)
        ptype = info.get("ptype")
        if (ev.get("meta") or {}).get("shot_type") == "wraparound":
            k = "bu_wrap"
        elif ps and ps != sh and ptype in ("bank", "saucer", "stretch", "drop"):
            k = "bu_" + ptype
        elif info.get("rush") or ev.get("chance") == "breakaway":
            k = "bu_rush"
        elif ps and ps != sh and ev.get("strength") == "PP":
            k = "bu_pp"
        elif ps and ps != sh:
            k = "bu_pass"
        else:
            k = "bu_carry"
            p0 = seg.get("p0")
            if p0 is not None and info.get("from_motion", True):
                y0 = abs(float(p0[1]))
                if abs(y) < 12.0 and y0 - abs(y) > 8.0:
                    k = "bu_cut"       # he really did cut in from the wide lane
                elif abs(y) > 17.0 and y0 > 15.0:
                    k = "bu_wide"      # he stayed wide on that side
        self._bu_done.add(key)
        tb = self._gc(bpt, float(ev["t"]) - (ept - bpt))
        return self.say(min(tb, float(ev["t"])), int(ev["seq"]), [(k, {"player": sh, "passer": ps, "lane": lane}, {})], 2, "build",
                        kind="buildup", pt=bpt, gap=3.0)

    def _goal_parts(self, ev: Dict[str, Any], sp: Dict[str, Any], sq: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        loop = self.loop
        lang = loop.lg.lang
        book = loop.book
        stren = ev.get("strength")
        ot = int(ev.get("period") or 1) >= 4
        rise = "goal_rise_ot" if ot else {"PP": "goal_rise_pp", "SH": "goal_rise_sh", "EN": "goal_rise_en"}.get(stren, "goal_rise")
        shot_p = self._shot_phrase(ev)
        if rise == "goal_rise" and shot_p and self.rng.random() < 0.5:
            return [("goal_rise_typed", {"Shot": shot_p}, {})] + self._goal_parts_rest(ev, sp, sq)
        return [(rise, {}, {})] + self._goal_parts_rest(ev, sp, sq)

    def _shot_phrase(self, ev: Dict[str, Any]) -> str:
        """Capitalised shot-type phrase ('Slap shot from the point', 'One-timer', ...) or ''."""
        st = (ev.get("meta") or {}).get("shot_type")
        if not st or _LC is None:
            return ""
        row = getattr(_LC, "SHOT_NAME", {}).get(self.loop.lg.lang) or getattr(_LC, "SHOT_NAME", {}).get("en") or {}
        key = "slap_point" if st == "slap" and ev.get("chance") == "point" else st
        return _cap(row.get(key) or row.get(st) or "")

    def _goal_parts_rest(self, ev: Dict[str, Any], sp: Dict[str, Any], sq: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        loop = self.loop
        lang = loop.lg.lang
        book = loop.book
        stren = ev.get("strength")
        fw = _loc(_FROM_WHERE, lang, ev.get("chance") or "", "")
        sid = ev.get("shooter_id")
        name = book.name(sid) or book.short(sid)
        parts: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [
            ("goal_name", {"surname": book.short(sid), "name": name, "from_where": (", " + fw) if fw and stren != "EN" else ""}, {}),
        ]
        ast = [book.short(x) for x in (ev.get("assist_ids") or []) if book.short(x)]
        if len(ast) >= 2:
            parts.append(("goal_assist2", {"a1": ast[0], "a2": ast[1]}, {}))
        elif ast:
            parts.append(("goal_assist1", {"a1": ast[0]}, {}))
        elif stren != "EN":
            parts.append(("goal_assist0", {}, {}))
        parts.append(("goal_score", sp, sq))
        return parts

    def _goal_replay(self, ev: Dict[str, Any], ept: float) -> None:
        """Analyst breakdown 6-9 s after the goal (during the celebration; the game clock is stopped)."""
        if int(ev.get("period") or 1) >= 4 or ev.get("strength") == "EN":
            return
        mt = getattr(self.loop, "motion", None)
        key = (ev.get("meta") or {}).get("event_key")
        info = (mt.buildup(key) if mt is not None and hasattr(mt, "buildup") and key else None) or {}
        book = self.loop.book
        sh = book.short(ev.get("shooter_id"))
        passer = info.get("passer") or ((ev.get("assist_ids") or [None])[0])
        ps = book.short(passer) if passer else ""
        if ev.get("chance") == "rebound":
            k = "rp_reb"
        elif info.get("rush") or ev.get("chance") == "breakaway":
            k = "rp_rush"
        elif ps and ps != sh and ev.get("strength") == "PP":
            k = "rp_pp"
        elif ps and ps != sh:
            k = "rp_pass"
        else:
            k = "rp_carry"
        self.say(float(ev["t"]), int(ev["seq"]), [(k, {"shooter": sh, "passer": ps}, {})], 2, "calm", speaker="color", kind="replay",
                 pt=ept + self.rng.uniform(6.0, 9.0), gap=4.0, force_color=True)

    def _save_replay(self, ev: Dict[str, Any], ept: float) -> None:
        mt = getattr(self.loop, "motion", None)
        if mt is None or not hasattr(mt, "stoppage_after") or mt.stoppage_after(float(ev["t"])) not in ("freeze", "out"):
            return  # only when play stops after the save (time for a replay)
        key = (ev.get("meta") or {}).get("event_key")
        info = (mt.buildup(key) if key else None) or {}
        book = self.loop.book
        ps = book.short(info.get("passer")) if info.get("passer") else ""
        k = "rp_save_pass" if ps and info.get("passer") != ev.get("shooter_id") else "rp_save"
        self.say(float(ev["t"]), int(ev["seq"]), [(k, {"goalie": book.short(ev.get("goalie_id")), "shooter": book.short(ev.get("shooter_id")), "passer": ps}, {})],
                 1, "calm", speaker="color", kind="replay", pt=ept + self.rng.uniform(4.0, 5.5), gap=3.0, force_color=True)

    def _rule_call(self, ev: Dict[str, Any], ept: Optional[float]) -> None:
        """Offside, icing, glass, freezes, faceoff wins, line changes, delayed penalties, hits (derived events)."""
        loop = self.loop
        lang = loop.lg.lang
        book = loop.book
        typ = ev["type"]
        m = ev.get("meta") or {}
        t = float(ev["t"])
        seq = int(ev["seq"])
        side = ev.get("side")
        team = self._team(side)
        pt = ept if ept is not None else self._pt(t)
        if typ == "stoppage":
            r = m.get("reason")
            if r == "freeze":
                if self.rng.random() < 0.55:
                    self.say(t, seq, [("rule_freeze", {"goalie": book.short(m.get("player_id"))}, {})], 1, "calm", kind="rule", pt=pt + 0.4, gap=3.5)
                return
            key = {"offside": "rule_offside", "icing": "rule_icing", "glass": "rule_glass", "out": "rule_out"}.get(str(r))
            if key and (r != "out" or self.rng.random() < 0.5):
                if self.say(t, seq, [(key, {"team": team}, {})], 2 if r in ("offside", "icing", "glass") else 1, "calm", kind="rule",
                            pt=pt + 0.4, gap=3.0):
                    self._last_rule_pt = pt
            return
        if typ == "rule":
            r = m.get("rule")
            key = {"icing_waved": "rule_icing_waved", "sh_clear": "rule_sh_clear"}.get(str(r))
            if key:
                self.say(t, seq, [(key, {"team": team}, {})], 1, "calm", kind="rule", pt=pt, gap=3.5)
            return
        if typ == "faceoff":
            r = m.get("reason")
            p_say = {"after_goal": 0.75, "period_start": 0.8, "icing": 0.55, "offside": 0.45, "penalty": 0.6}.get(str(r), 0.3)
            if pt - self._last_fo_pt < 25.0 or self.rng.random() > p_say:
                return
            loc = str(m.get("location") or "")
            row = _LC.FO_WHERE.get(lang, _LC.FO_WHERE["en"]) if _LC is not None else {"center": "", "neutral": "", "end": ""}
            if loc == "center":
                where = row["center"]
            elif loc in ("home_end", "away_end"):
                where = _fmt(row["end"], {"team": self._team("home" if loc == "home_end" else "away")})
            else:
                where = row["neutral"]
            if self.say(t, seq, [("fo_win", {"winner": book.short(m.get("winner_id")), "team": team, "where": where}, {})], 1, "calm",
                        kind="rule", pt=pt + 0.6, gap=3.0):
                self._last_fo_pt = pt
            return
        if typ == "line_change":
            if not side or pt - self._last_lc_pt.get(side, -999.0) < 90.0 or not m.get("center_id") or self.rng.random() > 0.45:
                return
            key = "line_change" if m.get("on_the_fly") else "line_change_stop"
            if self.say(t, seq, [(key, {"team": team, "center": book.short(m.get("center_id"))}, {})], 1, "calm", kind="rule", pt=pt, gap=4.0):
                self._last_lc_pt[side] = pt
            return
        if typ == "delayed_penalty":
            pen_side = m.get("side")
            self.say(t, seq, [("delayed_pen", {"team": self._team(pen_side), "ben": self._team(m.get("beneficiary"))}, {})], 2, "build",
                     kind="rule", pt=pt, gap=3.0)
            return
        if typ == "hit":
            if self.rng.random() < 0.6:
                self.say(t, seq, [("hit", {"hitter": book.short(m.get("hitter_id")), "victim": book.short(m.get("victim_id"))}, {})], 1, "build",
                         kind="rule", pt=pt + 0.3, gap=4.0)

    def on_hints(self, hints: List[Dict[str, Any]]) -> None:
        """Light play narration from the motion track (who has it, where) so the booth is never silent
        for long: one line when nothing has been said for ~7 s of presentation time."""
        loop = self.loop
        if not LIVE_COMMENTARY_ENABLED or not hints or loop.finished or loop.phase in ("pregame", "shootout"):
            return
        lang = loop.lg.lang
        book = loop.book
        mt = loop.motion
        for h in hints:
            pt = float(h["pt"])
            # narration is seasoning, not wallpaper: >= 22 s apart and only into a real silence (3-8 s gaps stay)
            if self.story_q or pt - self._last_nar_pt < 22.0:
                continue
            if not self._pt_free(pt, 9.0, 5.0):
                continue
            if h["kind"] == "pass" and (not h.get("rcv") or h.get("rcv") == h.get("pid")):
                continue
            side = h.get("side")
            if side not in _SIDES:
                continue
            gc = float(h.get("t", 0.0))
            fx, fy = h.get("from") or (0.0, 0.0)
            tx, ty = h.get("to") or (0.0, 0.0)
            u = fx * (mt.dir(side, gc) if hasattr(mt, "dir") else 1.0)
            player = book.short(h.get("pid")) if h.get("pid") else ""
            if h["kind"] == "dump":
                k, p = ("nar_rim" if h.get("ptype") == "rim" else "nar_dump"), {"team": self._team(side)}
            elif h["kind"] == "pass" and h.get("rcv") and player:
                ru = tx * (mt.dir(side, gc) if hasattr(mt, "dir") else 1.0)
                k = "nar_point" if 30.0 <= ru <= 45.0 and u > 45.0 else "nar_pass"
                if h.get("ptype") in ("bank", "saucer", "drop", "stretch"):
                    k = "nar_" + h["ptype"]
                p = {"player": player, "rcv": book.short(h.get("rcv"))}
            elif h["kind"] == "carry" and player:
                if h.get("entry"):
                    k = "nar_entry"
                else:
                    k = "nar_dz" if u < -25.0 else ("nar_oz" if u > 25.0 else "nar_nz")
                p = {"player": player, "lane": self._lane(side, gc, ty)}
            else:
                continue
            r_ = self.rng.random()
            try:
                pp_side = loop.manpower(gc).get("pp_side")
            except Exception:
                pp_side = None
            if pp_side == side and u > 25.0 and r_ < 0.4:
                k, p = "nar_pp", {"team": self._team(side), "player": player or self._team(side)}
            elif r_ < 0.2 and k != "nar_dump" and pt - getattr(self, "_last_team_nar_pt", -999.0) >= 60.0:
                # a team-level line at most once a minute, never the same template twice in a period
                k = {"nar_dz": "fill_dz", "nar_nz": "fill_nz", "nar_entry": "fill_oz"}.get(k, "fill_oz")
                p = {"team": self._team(side), "opp": self._team(_opp(side))}
            per = loop.clock_at(gc)[0]
            if getattr(self, "_ban_period", None) != per:
                self._ban_period, self._ban_tids = per, set()
            if h.get("pid") and h.get("pid") == getattr(self, "_last_nar_pid", None) and pt - self._last_nar_pt < 25.0:
                continue  # don't follow one player around the rink
            if self.say(gc, int(loop.lg.seq), [(k, p, {})], 1, "calm", kind="narration", pt=pt, gap=5.0):
                self._last_nar_pt = pt
                self._last_nar_pid = h.get("pid")
                if k.startswith("fill_"):
                    self._last_team_nar_pt = pt
                    self._ban_tids.update(x.rstrip("!") for x in (self._recent_lines[-1] if self._recent_lines else []))

    # ── booth stories
    def _story_facts(self, t: float) -> List[Dict[str, Any]]:
        loop = self.loop
        bs = _booth()
        if self._facts is None:
            try:
                self._facts = bs.build_facts(loop.session, loop.lg.home, loop.lg.away, loop.lg.home_id, loop.lg.away_id,
                                             coach=str(getattr(loop.session, "head_coach_name", "") or ""),
                                             user_tid=str(getattr(loop.session, "user_team_id", "") or ""))
            except Exception:
                logger.warning("booth facts failed", exc_info=True)
                self._facts = []
        pss = dict(getattr(loop.session, "player_season_stats", None) or {})
        on = []
        for side in _SIDES:
            for pid in loop.state.on_ice.get(side) or []:
                r = pss.get(pid)
                if r:
                    on.append((pid, {**r, "name": loop.book.name(pid) or r.get("name")}))
        live = bs.live_facts(self._team("home"), self._team("away"), int(loop.state.stats["home"]["sog"]),
                             int(loop.state.stats["away"]["sog"]), on, t)
        if getattr(self, "_xfacts", None) is None:
            try:
                self._xfacts = bs.extra_facts(loop.session, loop.lg.home, loop.lg.away, loop.lg.home_id, loop.lg.away_id)
            except Exception:
                logger.warning("booth extra facts failed", exc_info=True)
                self._xfacts = []
        live += bs.live_colour_facts(self._team("home"), self._team("away"), int(loop.lg.score_home), int(loop.lg.score_away), t)
        return list(self._facts) + list(self._xfacts) + live

    def _told(self) -> Tuple[Dict[str, int], int]:
        sess = self.loop.session
        told = getattr(sess, "_booth_told", None)
        if not isinstance(told, dict):
            told = {}
            try:
                setattr(sess, "_booth_told", told)
            except Exception:
                pass
        return told, len(getattr(sess, "game_results", None) or [])

    def _pick_story(self, t: float) -> Optional[Dict[str, Any]]:
        lang = self.loop.lg.lang
        told, game_no = self._told()
        for k in self._said:
            if k.startswith("story:"):
                told.setdefault(k[6:], game_no)
        on = set(self.loop.state.on_ice.get("home") or []) | set(self.loop.state.on_ice.get("away") or [])
        bs = _booth()
        st = bs.select2(lang, self._story_facts(t), told, game_no, self.rng, on_ice=on, used_lines=self._used_lines,
                        cat_counts=self._cat_counts, fam_counts=self._fam_counts, last_family=self._last_family, last_cat=self._last_story_cat,
                        told_game=self._told_game)
        if st is None:
            return None
        told[st["key"]] = game_no
        if len(told) > 4000:
            for k in sorted(told, key=lambda k: told[k])[:1500]:
                told.pop(k, None)
        self._said.add("story:" + st["key"])
        self._last_story_cat = st["cat"]
        self._last_family = st["family"]
        self._fam_counts[st["family"]] = self._fam_counts.get(st["family"], 0) + 1
        self._cat_counts[st["cat"]] = self._cat_counts.get(st["cat"], 0) + 1
        self.stories_told += 1
        st["lines"] = self._shape(st, lang)
        return st

    def _shape(self, st: Dict[str, Any], lang: str) -> List[Tuple[str, str]]:
        """Bridge in, payoff, a reply from the other seat, sometimes a hand-back to play. 2-5 lines."""
        bs = _booth()
        setup = st["setup"]
        if self.rng.random() < 0.6 and setup.rstrip().endswith("?"):  # bridge into questions only (no stacked intros)
            pool = bs.BRIDGES.get(lang, bs.BRIDGES["en"])
            opts = (pool.get("crowd") or []) if st["cat"] in ("crowd_arena", "crowd_live") else []
            opts = opts or pool.get(st["family"]) or []
            opts = [x for x in opts if ("br", x) not in self._used_lines]
            if opts:
                br = self.rng.choice(opts)
                self._used_lines.add(("br", br))
                low = br.endswith(",") or br.endswith(" but")
                setup = f"{br} {setup[:1].lower() + setup[1:]}" if low else f"{br} {setup}"
        lines: List[Tuple[str, str]] = [("pbp", setup), ("color", st["payoff"])]
        r = self.rng.random()
        fam_rep = (getattr(bs, "REPLIES_BY_FAMILY", {}).get(lang) or {}).get(st["family"]) or bs.REPLIES.get(lang) or []
        fam_rep = [x for x in fam_rep if ("rep", x) not in self._used_lines]
        if r < 0.25 or not st.get("react"):
            pass
        elif r < 0.6 or not fam_rep:
            lines.append(("pbp", st["react"]))
        else:
            rp = self.rng.choice(fam_rep)
            self._used_lines.add(("rep", rp))
            lines.append(("pbp", rp))
            lines.append(("color", st["react"]))
        hbs = [x for x in (bs.HANDBACKS.get(lang) or bs.HANDBACKS["en"]) if ("hb", x) not in self._used_lines]
        if len(lines) >= 3 and lines[-1][0] == "color" and hbs and self.rng.random() < 0.5:
            f = self.loop.motion.last_frame if getattr(self.loop, "motion", None) is not None else None
            side = (f or {}).get("pos")
            if side in _SIDES:
                hb = self.rng.choice(hbs)
                self._used_lines.add(("hb", hb))
                lines.append(("pbp", hb.format(team=self._team(side))))
        return lines

    def _queue_story(self, st: Dict[str, Any], t0: float, gap: Tuple[float, float] = (5.5, 7.5)) -> float:
        t = t0
        self._story_meta_lines = []
        for i, (spk, line) in enumerate(st["lines"]):
            item = {"due": t, "text": line, "speaker": spk, "key": st["key"], "topic": st.get("topic") or "",
                    "cat": st["cat"], "family": st.get("family"), "idx": i}
            self.story_q.append(item)
            self._story_meta_lines.append(item)
            t += self.rng.uniform(*gap)
        return t

    def _emit_story_line(self, item: Dict[str, Any], t: float, seq: int, pt: Optional[float] = None) -> None:
        lang = self.loop.lg.lang
        text = item["text"]
        if lang == "fr":
            text = _fr_elide(text)
        elif lang == "en":
            text = _en_article(text)
        self._n += 1
        tts = _speakable(text, lang)
        lpt = float(pt) if pt is not None else max(self._pt(t), self._last_pt + 5.0)
        line = {
            "id": f"s{seq}-{self._n}", "seq": seq, "t": round(t, 2), "pt": round(lpt, 2), "priority": 1, "profile": "calm",
            "speaker": item["speaker"], "kind": "story", "story": item["cat"], "family": item.get("family"), "lang": lang, "text": text, "tts_text": tts,
            "tts_hint": self._hint("calm", 1, item["speaker"], "story"), "audio_key": None, "audio_url": None,
            "real_ts": int(time.time() * 1000), "expires_at_t": round(t + 20.0, 2), "expires_at_pt": round(lpt + 20.0, 2),
        }
        _tts_attach(line)
        self.lines.append(line)
        self.lines = self.lines[-400:]
        self._last_t = max(self._last_t, t)
        self._last_pt = max(self._last_pt, lpt)

    def _interrupt(self, t: float) -> None:
        started = any(x.get("emitted") for x in self._story_meta_lines) if getattr(self, "_story_meta_lines", None) else False
        rest = [x for x in self.story_q if x["idx"] >= 1 and not x.get("resume")] if started else []
        resumed = any(x.get("resumed") for x in self.story_q)
        topic = (self.story_q[0] or {}).get("topic") if self.story_q else ""
        self._interrupted = {"lines": rest, "topic": topic, "t": t} if rest and not resumed else None
        self.story_q = []
        self._cutin = bool(started or resumed)  # only cut in if the booth was actually mid-story

    def _intermission_segment(self, t: float, seq: int) -> None:
        """Longer booth segment during the intermission (2 stories, 4-6 lines)."""
        self.story_q = []
        self._interrupted = None
        tt = max(t, self._last_t) + 0.5
        ppt = max(self._pt(t) + 3.0, self._last_pt + 4.0)
        for _ in range(2):
            st = self._pick_story(t)
            if st is None:
                break
            for spk, line in st["lines"]:
                self._emit_story_line({"text": line, "speaker": spk, "cat": st["cat"], "family": st.get("family")}, t, seq, pt=ppt)
                tt += 0.5
                ppt += 6.0
        self.next_story_t = t + 60.0

    def _lull(self, t: float) -> bool:
        loop = self.loop
        if t - self._last_imp2_t < 25.0:
            return False
        try:
            if loop.manpower(t)["state"] not in ("EV", "4v4", "3v3"):
                return False
        except Exception:
            pass
        period, left = loop.clock_at(t)
        diff = abs(int(loop.lg.score_home) - int(loop.lg.score_away))
        if period >= 3 and left <= 120.0 and diff <= 1:
            return False
        return True

    def _story_gap(self, t: float) -> float:
        loop = self.loop
        period, left = loop.clock_at(t)
        diff = abs(int(loop.lg.score_home) - int(loop.lg.score_away))
        g = self.rng.uniform(110.0, 200.0)
        if diff >= 3:
            g *= 0.65
        if period == 1:
            g *= 0.85
        if period >= 3 and diff <= 1:
            g *= 1.6
        return g

    def tick(self, t: float) -> None:
        loop = self.loop
        if loop.finished or loop.intermission or loop.phase in ("pregame", "shootout"):
            return
        seq = int(loop.lg.seq)
        period, left = loop.clock_at(t)
        diff = abs(int(loop.lg.score_home) - int(loop.lg.score_away))
        if period == 3 and left <= 120.0 and "late" not in self._said and diff <= 1:
            self._said.add("late")
            self.story_q = []
            self.say(t, seq, [("late_tied" if diff == 0 else "late_close", {}, {})], 2, "build")
            return
        # running story: emit due lines, spaced so speech fits
        if self.story_q:
            item = self.story_q[0]
            if item["due"] <= t and t - self._last_t >= 5.0:
                self.story_q.pop(0)
                item["emitted"] = True
                self._emit_story_line(item, max(item["due"], self._last_t + 5.0), seq)
                if not self.story_q:
                    self.next_story_t = t + self._story_gap(t)
            return
        if self._lull(t):
            if self._interrupted and t - float(self._interrupted["t"]) >= 25.0:
                intr, self._interrupted = self._interrupted, None
                if intr.get("lines") and intr.get("topic") and self.rng.random() < 0.5:
                    lang = loop.lg.lang
                    res = self.rng.choice(_booth().RESUME.get(lang) or _booth().RESUME["en"]).format(topic=intr["topic"])
                    tt = t
                    for i, x in enumerate([{"text": res, "speaker": "pbp", "cat": intr["lines"][0]["cat"], "family": intr["lines"][0].get("family"), "idx": 0, "resume": True}] + intr["lines"]):
                        self.story_q.append({**x, "due": tt, "resumed": True})
                        tt += self.rng.uniform(5.5, 7.5)
                    return
            if t >= self.next_story_t and t - self._last_t >= 6.0:
                st = self._pick_story(t)
                if st is not None:
                    self._queue_story(st, t)
                    return
                self.next_story_t = t + 60.0
        # analyst stat in a quiet spell (families on cooldown) or, rarely, a filler call
        if t - self._last_t < 25.0:
            return
        lang = loop.lg.lang
        user = loop.lg.user_side
        mine = loop.lg.score_home if user == "home" else loop.lg.score_away
        theirs = loop.lg.score_away if user == "home" else loop.lg.score_home
        if self._pbp_since_color >= 2:
            cands: List[Tuple[str, Dict[str, Any], Dict[str, Any], Any, float]] = []
            if len(self._att_sides) >= 8:
                for s in _SIDES:
                    n = self._att_sides.count(s)
                    if n >= 8:
                        cands.append(("c_share", {"team": self._team(s), "n": n, "m": len(self._att_sides)}, {"n": _num_word(n, lang), "m": _num_word(len(self._att_sides), lang)}, n, 99.0))
            hx, ax = float(loop.state.stats["home"]["xg"]), float(loop.state.stats["away"]["xg"])
            if t > 600.0 and max(hx, ax) >= 0.8:
                lead = "home" if hx >= ax else "away"
                cands.append(("c_xg", {"team": self._team(lead), "opp": self._team(_opp(lead)), "a": _dec(max(hx, ax), lang),
                                       "b": _dec(min(hx, ax), lang)}, {}, round(max(hx, ax), 1), 99.0))
            if period == 3 and theirs > mine and left <= 420.0:
                n = int(theirs) - int(mine)
                cands.append(("c_trail", {"team": self._team(user), "n": n, "time": _clock_display(left).lstrip("0")},
                              {"n": _num_word(n, lang), "time": _time_tts(left, lang)}, n, 99.0))
            cands = [c for c in cands if self._fam_ok(c[0], period, c[3], min_change=c[4])]
            if cands:
                key, p, q, val, _m = self.rng.choice(cands)
                if self.say(t, seq, [(key, p, q)], 1, "calm", speaker="color", kind="color"):
                    self._fam[key] = (period, val, self.loop.game_sec_elapsed)
                    return
        if t - self._last_fill_t < 160.0:
            return
        sit = loop.motion.situation_key() if getattr(loop, "motion", None) is not None else None
        if not sit or sit[0] == "stop" or sit[1] not in _SIDES:
            return
        kind_, side = sit
        key = {"oz": "fill_oz", "nz": "fill_nz", "dz": "fill_dz"}.get(kind_)
        if key and self.say(t, seq, [(key, {"team": self._team(side), "opp": self._team(_opp(side))}, {})], 1, "calm", kind="filler"):
            self._last_fill_t = t


def _booth() -> Any:
    from services import booth_stories as bs

    return bs


_SHOT_ZONES = {
    "high_danger": ((78.0, 86.0), (0.0, 9.0)), "slot": ((64.0, 77.0), (0.0, 14.0)), "rebound": ((82.0, 87.0), (0.0, 6.0)),
    "point": ((28.0, 38.0), (8.0, 34.0)), "perimeter": ((48.0, 76.0), (22.0, 38.0)), "breakaway": ((70.0, 82.0), (0.0, 8.0)),
}


def _assign_roles(ids: List[str], book: _PlayerBook) -> List[Tuple[str, str]]:
    ds = [p for p in ids if book.pos(p) == "D"]
    fs = [p for p in ids if p not in ds]
    if not ds and len(fs) >= 4:  # positions unknown: last two act as D
        ds, fs = fs[-2:], fs[:-2]
    out: List[Tuple[str, str]] = list(zip(ds[:2], ("LD", "RD")))
    out += [(p, "D") for p in ds[2:]]
    order = sorted(fs, key=lambda p: {"C": 0, "LW": 1, "RW": 2}.get(book.pos(p), 3))
    roles = ["C", "LW", "RW"]
    for j, p in enumerate(order):
        out.append((p, roles[j] if j < len(roles) else "F"))
    return out


class TrackingModel:
    """Shot-location estimates (deterministic per event key) + legacy frame view of the motion track."""

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop

    @staticmethod
    def home_dir(period: int) -> float:
        return 1.0 if int(period) in (1, 3, 5) else -1.0

    def attack_dir(self, side: str, period: int) -> float:
        d = self.home_dir(period)
        return d if side == "home" else -d

    def shot_loc_for(self, key: str, chance: Optional[str], direction: float) -> Dict[str, Any]:
        rng = random.Random(zlib.crc32(f"{int(self.loop.lg.seed)}|shot|{key}".encode()))
        (x0, x1), (y0, y1) = _SHOT_ZONES.get(str(chance or ""), ((60.0, 80.0), (0.0, 16.0)))
        x = rng.uniform(x0, x1)
        y = max(-40.0, min(40.0, rng.uniform(y0, y1) * (1.0 if rng.random() < 0.5 else -1.0)))
        return {"x": round(x * direction, 1), "y": round(y * direction, 1), "est": True}

    @property
    def frames(self) -> List[Dict[str, Any]]:
        """Latest motion frame in the v1 `tracking.frames` shape (compatibility)."""
        mt = getattr(self.loop, "motion", None)
        f = mt.last_frame if mt is not None else None
        if not f:
            return []
        book = self.loop.book
        roles = mt.roles
        toks = [{"id": pid, "side": book.side.get(pid), "x": x, "y": y, "role": roles.get(pid, "G" if book.pos(pid) == "G" else "F")}
                for pid, x, y in f["p"]]
        pk = f.get("pk")
        return [{"t": f["t"], "tokens": toks, "puck": {"x": pk[0], "y": pk[1]} if pk else None}]


# ═══ 5b. Motion track — play-flow model between real events (presentation only)

_MX, _MY = 97.0, 40.0          # skater bounds (rink is ±100 x ±42.5)
_NET_X = 89.0
_V_SKATE = 30.5                # < contract cap 32 ft/s
_A_SKATE = 24.0                # ft/s^2
_V_GOALIE = 16.0
_CARRY = 21.0                  # nominal carry speed
_PASS = 72.0                   # pass speed
_SHOT = 100.0                  # shot speed (< 120 cap)
_BENCH = {"home": (-22.0, 39.0), "away": (22.0, 39.0)}
_ANCHOR_TYPES = frozenset({"attempt", "goal", "penalty", "period_start", "period_end", "shootout_attempt", "final"})


def _clampxy(x: float, y: float) -> Tuple[float, float]:
    return max(-_MX, min(_MX, x)), max(-_MY, min(_MY, y))


class MotionTrack:
    """Deterministic 0.5 s motion frames for the span just simulated.

    Real events are anchors: each attempt's shot leaves from its shot_loc exactly at its t.
    Between anchors a possession script (faceoff -> breakout -> neutral zone carry/dump ->
    entry -> cycle -> shot, with turnovers) moves the puck; skaters steer toward role spots
    for the manpower (5v5, PP 1-3-1, PK box, 4v4, 3v3, 6v5) with speed/acceleration caps;
    goalies track the puck on the crease arc. Never feeds the engine.
    """

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.t_last: Optional[float] = None
        self.pos: Dict[str, List[float]] = {}
        self.plan: List[Dict[str, Any]] = []
        self.plan_key: Optional[Tuple[Any, Any]] = None
        self.last_frame: Optional[Dict[str, Any]] = None
        self.roles: Dict[str, str] = {}
        self.last_whistle: Optional[Tuple[str, Dict[str, Any]]] = None
        self._anchor_cache: Tuple[Any, List[Dict[str, Any]]] = (None, [])
        self._att_cache: Tuple[Any, List[Dict[str, Any]]] = (None, [])
        self._lineup_cache: Dict[float, Dict[str, List[str]]] = {}
        self._cur_seg: Optional[Dict[str, Any]] = None
        self._so_t: Optional[float] = None
        self._te: Dict[str, float] = {}

    # ── anchors / timing
    def _anchors(self) -> List[Dict[str, Any]]:
        tl = self.loop._timeline
        key = (id(tl), len(tl))
        if self._anchor_cache[0] != key:
            self._anchor_cache = (key, [r for r in tl if r.get("type") in _ANCHOR_TYPES])
            self._att_cache = (key, [r for r in tl if r.get("type") in ("attempt", "goal") and (r.get("att") or {}).get("atk_ids")])
            self._so_t = next((float(r["t"]) for r in tl if r.get("key") == "ps5"), None)
            self._lineup_cache = {}
            goal_ts = [float(r["t"]) for r in self._anchor_cache[1] if r.get("type") == "goal"]
            te: Dict[str, float] = {}
            for r in self._anchor_cache[1]:
                t = float(r["t"])
                if r["type"] == "period_start" and t > 0.0:
                    t += 0.25  # the period_end frame at T stays "intermission"; faceoff from T+0.5
                elif r["type"] in ("period_end", "final") and any(abs(t - g) < 0.01 for g in goal_ts):
                    t += 6.0  # let a game-ending goal play out
                te[r["key"]] = t
            self._te = te
        return self._anchor_cache[1]

    def _attempts(self) -> List[Dict[str, Any]]:
        self._anchors()
        return self._att_cache[1]

    def _pair(self, t: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """(anchor in effect at t, next anchor) using effective anchor times."""
        prev = nxt = None
        pte = -1e18
        nte = 1e18
        te = self._te
        for r in self._anchors():
            x = te.get(r["key"], float(r["t"]))
            if x <= t + 1e-9:
                if x >= pte:
                    prev, pte = r, x
            elif x < nte:
                nxt, nte = r, x
        return prev, nxt

    def te(self, r: Optional[Dict[str, Any]]) -> float:
        if r is None:
            return 0.0
        self._anchors()
        return self._te.get(r["key"], float(r["t"]))

    def _dir(self, side: str, t: float) -> float:
        self._anchors()
        if self._so_t is not None and t >= self._so_t - 1e-9:
            period = 5
        else:
            period = self.loop.clock_at(max(0.0, t - 1e-6))[0]  # frame at a period's last instant keeps its ends
        d = 1.0 if period in (1, 3, 5) else -1.0
        return d if side == "home" else -d

    def _abs(self, side: str, t: float, u: float, v: float) -> Tuple[float, float]:
        d = self._dir(side, t)
        return _clampxy(u * d, v * d)

    def _rel(self, side: str, t: float, x: float, y: float) -> Tuple[float, float]:
        d = self._dir(side, t)
        return x * d, y * d

    def _shot_loc(self, raw: Dict[str, Any]) -> Tuple[float, float]:
        loc = raw.get("shot_loc") or {}
        return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))

    # ── lineups (deterministic from anchors)
    def lineup_at(self, t: float) -> Dict[str, List[str]]:
        k = round(t * 2) / 2.0
        hit = self._lineup_cache.get(k)
        if hit is not None:
            return hit
        atts = self._attempts()
        prev = nxt = None
        for r in atts:
            if float(r["t"]) <= t + 1e-9:
                prev = r
            else:
                nxt = r
                break
        cur = prev or nxt
        out = {"home": [], "away": []}
        if cur is not None:
            use = cur
            if prev is not None and nxt is not None:
                tp, tn = float(prev["t"]), float(nxt["t"])
                sw = max(tp + 0.5 * (tn - tp), tn - 7.0)
                if (prev.get("att") or {}).get("outcome") == "GOAL":
                    sw = min(sw, tp + 6.0)
                for r in self._anchors():
                    if tp < float(r["t"]) < tn and r["type"] in ("penalty", "period_start"):
                        sw = min(sw, float(r["t"]) + (2.0 if r["type"] == "penalty" else 0.0))
                        break
                if t >= sw:
                    use = nxt
            a = use.get("att") or {}
            s = a.get("side")
            if s in _SIDES:
                out[s] = list(a.get("atk_ids") or [])
                out[_opp(s)] = list(a.get("def_ids") or [])
        for side in _SIDES:
            if not out[side]:
                ln = self.loop.lines[side]
                F = [p for p in ((ln.get("F") or [[]])[0] or [])]
                D = [p for p in ((ln.get("D") or [[]])[0] or [])]
                out[side] = (F + D)[:5]
        # penalized players sit in the box; on-ice counts follow the real manpower
        cap = self.loop._cap
        boxed = set()
        need = {"home": 5, "away": 5}
        if t >= _REGULATION_GAME_SEC and not self.loop.lg.is_playoff and (self.loop._result or {}).get("overtime"):
            need = {"home": 3, "away": 3}
        elif cap is not None:
            for pen in cap.penalties:
                if pen["start"] <= t < pen["start"] + pen["dur"]:
                    need[pen["side"]] = max(3, need[pen["side"]] - 1)
                    if pen.get("player_id"):
                        boxed.add(pen["player_id"])
        book = self.loop.book
        for side in _SIDES:
            ids = [p for p in out[side] if p not in boxed]
            pool = [p for grp in (self.loop.lines[side].get("F") or []) + (self.loop.lines[side].get("D") or []) for p in grp]
            while len(ids) < need[side]:
                have_d = sum(1 for p in ids if book.pos(p) == "D")
                cand = [p for p in pool if p not in ids and p not in boxed]
                pref = [p for p in cand if (book.pos(p) == "D") == (have_d < 2)] or cand
                if not pref:
                    break
                ids.append(pref[0])
            while len(ids) > need[side]:
                fw = [p for p in ids if book.pos(p) != "D"]
                ids.remove(fw[-1] if len(fw) > 1 else ids[-1])
            out[side] = ids
        if len(self._lineup_cache) > 4000:
            self._lineup_cache.clear()
        self._lineup_cache[k] = out
        return out

    def goalie_at(self, side: str, t: float) -> Optional[str]:
        pulled = False
        for (tt, s, v) in self.loop._pull_hist:
            if s == side and tt <= t + 1e-9:
                pulled = v
        if pulled:
            return None
        g = self.loop.lines[side].get("G") or []
        return g[0] if g else None

    # ── plan building
    def _rng(self, *parts: Any) -> random.Random:
        return random.Random(zlib.crc32(("|".join(str(p) for p in (self.loop.lg.seed,) + parts)).encode()))

    @staticmethod
    def _seg(t0: float, t1: float, kind: str, p0: Tuple[float, float], p1: Tuple[float, float], side: Optional[str] = None,
             car: Optional[str] = None, rcv: Optional[str] = None, ph: str = "play", fo: Optional[Tuple[float, float]] = None,
             why: Optional[Tuple[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {"t0": float(t0), "t1": float(max(t1, t0 + 0.01)), "kind": kind, "p0": p0, "p1": p1, "side": side,
                "car": car, "rcv": rcv, "ph": ph, "fo": fo, "why": why}

    def _role_pid(self, side: str, t: float, role: str) -> Optional[str]:
        ids = self.lineup_at(t)[side]
        roles = dict((r, p) for p, r in _assign_roles(ids, self.loop.book))
        for r in {"D1": ("LD", "RD", "D"), "D2": ("RD", "LD", "D"), "W": ("LW", "RW", "C", "F"), "W2": ("RW", "LW", "F", "C"),
                  "C": ("C", "LW", "RW", "F")}.get(role, (role,)):
            if r in roles:
                return roles[r]
        return ids[0] if ids else None

    def _faceoff_then(self, t0: float, dot: Tuple[float, float], nxt: Optional[Dict[str, Any]], rng: random.Random,
                      segs: List[Dict[str, Any]]) -> Tuple[float, Tuple[float, float], Optional[str], Optional[str]]:
        segs.append(self._seg(t0, t0 + 1.0, "hold", dot, dot, ph="faceoff", fo=dot))
        target = (nxt.get("att") or {}).get("side") if nxt is not None else None
        win = target if (target in _SIDES and rng.random() < 0.6) else ("home" if rng.random() < 0.5 else "away")
        du, dv = self._rel(win, t0, *dot)
        back = self._abs(win, t0, du - 18.0, dv + rng.uniform(-6, 6))
        d1 = self._role_pid(win, t0 + 1.0, "D1")
        dist = math.hypot(back[0] - dot[0], back[1] - dot[1])
        t1 = t0 + 1.0 + max(0.35, dist / _PASS)
        segs.append(self._seg(t0 + 1.0, t1, "pass", dot, back, side=win, rcv=d1))
        return t1, back, win, d1

    def _aftermath(self, A: Dict[str, Any], tB: float, nxt: Optional[Dict[str, Any]], rng: random.Random
                   ) -> Tuple[List[Dict[str, Any]], float, Tuple[float, float], Optional[str], Optional[str]]:
        """Segments right after anchor A; returns (segs, t_end, puck, possessing side, carrier)."""
        segs: List[Dict[str, Any]] = []
        tA = float(A["t"]) + (0.01 if A["type"] == "period_start" else 0.0)
        typ = A["type"]
        if typ in ("period_end", "final"):
            tA = self.te(A)
        avail = max(0.5, tB - tA)
        book = self.loop.book
        if typ == "period_start":
            if int(A.get("period") or 1) == 5:
                return segs, tA, (0.0, 0.0), None, None
            t1, p, s, c = self._faceoff_then(tA, (0.0, 0.0), nxt, rng, segs)
            return segs, t1, p, s, c
        if typ in ("period_end", "final"):
            segs.append(self._seg(tA, tA + 9999.0, "hold", self._puck_now(tA), self._puck_now(tA),
                                  ph="intermission" if typ == "period_end" else "stoppage",
                                  why=("period_end" if typ == "period_end" else "final", {})))
            return segs, tA + 9999.0, self._puck_now(tA), None, None
        if typ == "penalty":
            pen = A.get("pen") or {}
            ps = pen.get("side") if pen.get("side") in _SIDES else "home"
            p0 = self._puck_now(tA)
            dot = self._abs(ps, tA, -69.0, 22.0 if rng.random() < 0.5 else -22.0)
            dur = max(min(4.0, max(1.0, avail - 3.0)), math.hypot(dot[0] - p0[0], dot[1] - p0[1]) / 100.0)
            segs.append(self._seg(tA, tA + dur, "dead", p0, dot, ph="stoppage", fo=dot,
                                  why=("penalty", {"player": book.short(pen.get("player_id"))})))
            t1, p, s, c = self._faceoff_then(tA + dur, dot, nxt, rng, segs)
            return segs, t1, p, s, c
        if typ == "shootout_attempt":
            r = A.get("so") or {}
            side = r.get("side") or "home"
            p0 = self._abs(side, tA, 66.0, 0.0)
            p1 = self._abs(side, tA, 80.0, rng.uniform(-3, 3))
            net = self._abs(side, tA, 89.5 if r.get("goal") else 86.5, rng.uniform(-2, 2))
            segs.append(self._seg(tA, tA + 0.5, "carry", p0, p1, side=side, car=r.get("shooter_id"), ph="shootout"))
            segs.append(self._seg(tA + 0.5, tA + 0.9, "shot", p1, net, side=side, ph="shootout"))
            segs.append(self._seg(tA + 0.9, tA + 9999.0, "hold", net, net, ph="shootout"))
            return segs, tA + 9999.0, net, None, None
        # attempts / goals
        a = A.get("att") or {}
        X = a.get("side") if a.get("side") in _SIDES else "home"
        L = self._shot_loc(A)
        lu, lv = self._rel(X, tA, *L)
        oc = a.get("outcome")
        tS = math.ceil(tA * 2.0 - 1e-9) / 2.0  # release on the frame grid: frames around t both show the puck at shot_loc
        if tS > tA + 1e-9 and tB - tA > 1.5:
            segs.append(self._seg(tA, tS, "hold", L, L, side=X, car=a.get("shooter_id")))
            avail -= tS - tA
            tA = tS
        if oc == "GOAL":
            net = self._abs(X, tA, 90.5, max(-2.5, min(2.5, lv * 0.08)))
            tau = max(0.25, math.hypot(net[0] - L[0], net[1] - L[1]) / _SHOT)
            segs.append(self._seg(tA, tA + tau, "shot", L, net, side=X))
            g_end = tA + tau + min(5.0, max(1.5, avail - 6.0))
            segs.append(self._seg(tA + tau, g_end, "hold", net, net, side=X, ph="goal",
                                  why=("goal", {"player": book.short(a.get("shooter_id"))})))
            if a.get("seg") != "reg":
                segs.append(self._seg(g_end, tA + 9999.0, "hold", net, net, ph="goal"))
                return segs, tA + 9999.0, net, None, None
            d_end = g_end + max(min(3.0, max(1.0, avail - (g_end - tA) - 2.0)), 0.95)
            segs.append(self._seg(g_end, d_end, "dead", net, (0.0, 0.0), ph="stoppage", fo=(0.0, 0.0)))
            t1, p, s, c = self._faceoff_then(d_end, (0.0, 0.0), nxt, rng, segs)
            return segs, t1, p, s, c
        if oc == "SAVED" and not a.get("empty_net"):
            gp = self._abs(X, tA, 86.0, max(-3.0, min(3.0, lv * 0.12)))
            tau = max(0.2, math.hypot(gp[0] - L[0], gp[1] - L[1]) / _SHOT)
            segs.append(self._seg(tA, tA + tau, "shot", L, gp, side=X))
            if avail >= 13.0 and rng.random() < 0.4:
                dot = self._abs(X, tA, 69.0, 22.0 if lv >= 0 else -22.0)
                segs.append(self._seg(tA + tau, tA + tau + 3.0, "dead", gp, dot, ph="stoppage", fo=dot,
                                      why=("freeze", {"goalie": book.short(a.get("goalie_id"))})))
                t1, p, s, c = self._faceoff_then(tA + tau + 3.0, dot, nxt, rng, segs)
                return segs, t1, p, s, c
            reb = self._abs(X, tA, rng.uniform(72.0, 80.0), rng.uniform(-14.0, 14.0))
            segs.append(self._seg(tA + tau, tA + tau + 0.5, "loose", gp, reb))
            return segs, tA + tau + 0.5, reb, None, None
        if oc == "BLOCKED":
            net = self._abs(X, tA, 89.0, 0.0)
            bp = (L[0] + 0.15 * (net[0] - L[0]), L[1] + 0.15 * (net[1] - L[1]))
            tau = max(0.12, math.hypot(bp[0] - L[0], bp[1] - L[1]) / _SHOT)
            segs.append(self._seg(tA, tA + tau, "shot", L, bp, side=X))
            dfl = _clampxy(bp[0] + rng.uniform(-12, 12), bp[1] + rng.uniform(-12, 12))
            segs.append(self._seg(tA + tau, tA + tau + 0.6, "loose", bp, dfl))
            return segs, tA + tau + 0.6, dfl, None, None
        # MISSED (or empty-net shot that stayed out): wide, off the end boards
        sgn = 1.0 if (lv >= 0) else -1.0
        wide = self._abs(X, tA, 97.0, sgn * rng.uniform(6.0, 22.0))
        tau = max(0.2, math.hypot(wide[0] - L[0], wide[1] - L[1]) / 110.0)
        segs.append(self._seg(tA, tA + tau, "shot", L, wide, side=X))
        rim = self._abs(X, tA, 90.0, sgn * 37.0)
        segs.append(self._seg(tA + tau, tA + tau + 0.8, "loose", wide, rim))
        if avail >= 12.0 and rng.random() < 0.22:
            dot = self._abs(X, tA, 69.0, 22.0 * sgn)
            segs.append(self._seg(tA + tau + 0.8, tA + tau + 3.8, "dead", rim, dot, ph="stoppage", fo=dot, why=("out", {})))
            t1, p, s, c = self._faceoff_then(tA + tau + 3.8, dot, nxt, rng, segs)
            return segs, t1, p, s, c
        return segs, tA + tau + 0.8, rim, None, None

    def _route(self, X: str, t0: float, p0: Tuple[float, float], car: Optional[str], tB: float,
               L: Optional[Tuple[float, float]], shooter: Optional[str], rng: random.Random) -> List[Dict[str, Any]]:
        """Possession script for team X from p0 at t0 to the shot at L at tB (L None: just keep possession)."""
        u0, v0 = self._rel(X, t0, *p0)
        s = 1.0 if rng.random() < 0.5 else -1.0
        wps: List[Tuple[str, float, float, str]] = []
        if u0 < -30.0:
            if rng.random() < 0.5:
                wps.append(("pass", -78.0, -s * 14.0, "D2"))
            wps.append(("pass", -30.0, s * 30.0, "W"))
            wps.append(("carry", 4.0, s * 24.0, "W"))
        elif u0 < 22.0:
            wps.append(("carry", 8.0, max(-25.0, min(25.0, v0)), "CAR"))
        if u0 < 25.0:
            if rng.random() < 0.7:
                wps.append(("carry", 30.0, s * 20.0, "W" if u0 < -30.0 else "CAR"))
            else:
                wps.append(("loose", 88.0, s * 33.0, "-"))
                wps.append(("carry", 80.0, s * 30.0, "W"))
        cycle = [("carry", 82.0, s * 30.0, "W"), ("pass", 38.0, s * 20.0, "D1"), ("pass", 38.0, -s * 20.0, "D2"),
                 ("pass", 62.0, -s * 30.0, "W2"), ("carry", 80.0, -s * 28.0, "W2"), ("pass", 60.0, s * 28.0, "W")]
        if L is not None:
            lu, lv = self._rel(X, t0, *L)
            final = ("pass" if shooter and shooter != car else "carry", lu, lv, "SHOOTER")
        else:
            final = None

        def dur(kind: str, a: Tuple[float, float], b: Tuple[float, float]) -> float:
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            return max(0.35, d / (_PASS if kind in ("pass", "loose") else _CARRY))

        T = max(0.5, tB - t0)

        def total(lst: List[Tuple[str, float, float, str]]) -> float:
            tt, prev = 0.0, (u0, v0)
            for k, u, v, _r in lst:
                tt += dur(k, prev, (u, v))
                prev = (u, v)
            return tt

        body = list(wps)
        ci = 0
        while final is not None and total(body + [final]) < 0.75 * T and ci < 24:
            body.append(cycle[ci % len(cycle)])
            ci += 1
        if final is None:
            while total(body) < 0.9 * T and ci < 24:
                body.append(cycle[ci % len(cycle)])
                ci += 1
        seq = body + ([final] if final is not None else [])
        while len(seq) > 1 and total(seq) > T:
            seq.pop(0)
        durs, prev = [], (u0, v0)
        for k, u, v, _r in seq:
            durs.append(dur(k, prev, (u, v)))
            prev = (u, v)
        tot = sum(durs) or 1.0
        if tot < T:
            carry_idx = [i for i, (k, *_x) in enumerate(seq) if k == "carry"]
            extra = T - tot
            if carry_idx:
                ctot = sum(durs[i] for i in carry_idx)
                for i in carry_idx:
                    durs[i] += extra * (durs[i] / ctot)
            else:
                durs[0] += extra
        else:
            scale = T / tot
            durs = [d * scale for d in durs]
        segs: List[Dict[str, Any]] = []
        t = t0
        cur = (u0, v0)
        holder = car
        for (k, u, v, role), d in zip(seq, durs):
            t1 = t + d
            pa = self._abs(X, t, *cur)
            pb = self._abs(X, t, u, v)
            if role == "SHOOTER":
                who = shooter or holder
            elif role == "CAR":
                who = holder or self._role_pid(X, t, "C")
            elif role == "-":
                who = None
            else:
                who = self._role_pid(X, t1, role)
            if k == "carry":
                if who and holder and who != holder:
                    mid = (pa[0] + 0.5 * (pb[0] - pa[0]), pa[1] + 0.5 * (pb[1] - pa[1]))
                    pt = min(t1 - 0.1, t + max(0.35, math.hypot(mid[0] - pa[0], mid[1] - pa[1]) / _PASS))
                    segs.append(self._seg(t, pt, "pass", pa, mid, side=X, car=holder, rcv=who))
                    t, pa = pt, mid
                segs.append(self._seg(t, t1, "carry", pa, pb, side=X, car=who or holder))
                holder = who or holder
            elif k == "pass":
                segs.append(self._seg(t, t1, "pass", pa, pb, side=X, car=holder, rcv=who))
                holder = who
            else:  # dump-in
                segs.append(self._seg(t, t1, "loose", pa, pb, side=X))
                holder = None
            cur = (u, v)
            t = t1
        return segs

    def _buildup(self, t0: float, p0: Tuple[float, float], side0: Optional[str], car0: Optional[str],
                 B: Optional[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
        segs: List[Dict[str, Any]] = []
        if B is None:
            return [self._seg(t0, t0 + 9999.0, "hold", p0, p0, ph="stoppage")]
        tB = self.te(B)
        if B["type"] in ("attempt", "goal"):
            a = B.get("att") or {}
            X = a.get("side") if a.get("side") in _SIDES else "home"
            L: Optional[Tuple[float, float]] = self._shot_loc(B)
            shooter = a.get("shooter_id")
        elif B["type"] == "shootout_attempt" or (B["type"] == "period_start" and int(B.get("period") or 0) == 5):
            return [self._seg(t0, tB, "hold", p0, p0, ph="shootout")]
        else:
            X = side0 if side0 in _SIDES else ("home" if rng.random() < 0.5 else "away")
            L, shooter = None, None
        t, p, side, car = t0, p0, side0, car0
        if side is None or car is None:
            rec = X if rng.random() < 0.6 or tB - t < 6.0 else _opp(X)
            nearest = self._nearest(rec, p, t)
            dt = min(1.2, max(0.4, (tB - t) * 0.2))
            segs.append(self._seg(t, t + dt, "loose", p, p))
            t, side, car = t + dt, rec, nearest
        if side != X and tB - t > 8.0:
            dY = min(0.35 * (tB - t), 12.0)
            Y = side
            yu, yv = self._rel(Y, t, *p)
            q = self._abs(Y, t, min(yu + rng.uniform(25.0, 45.0), 20.0), max(-28.0, min(28.0, yv + rng.uniform(-10, 10))))
            segs.append(self._seg(t, t + dY, "carry", p, q, side=Y, car=car))
            segs.append(self._seg(t + dY, t + dY + 0.6, "loose", q, q))
            t, p = t + dY + 0.6, q
            side, car = X, self._nearest(X, q, t)
        elif side != X:
            segs.append(self._seg(t, t + 0.4, "loose", p, p))
            t += 0.4
            side, car = X, self._nearest(X, p, t)
        if L is not None:
            tq = math.floor(tB * 2.0) / 2.0
            if tq <= t + 0.2:
                tq = tB
            segs += self._route(X, t, p, car, tq, L, shooter, rng)
            if tq < tB:
                segs.append(self._seg(tq, tB, "hold", L, L, side=X, car=shooter))
        else:
            segs += self._route(X, t, p, car, tB, L, shooter, rng)
        return segs

    def _nearest(self, side: str, p: Tuple[float, float], t: float) -> Optional[str]:
        ids = self.lineup_at(t)[side]
        best, bd = None, 1e9
        for pid in ids:
            q = self.pos.get(pid)
            d = math.hypot(q[0] - p[0], q[1] - p[1]) if q else 50.0
            if d < bd:
                best, bd = pid, d
        return best

    def _puck_now(self, t: float) -> Tuple[float, float]:
        if self._cur_seg is not None:
            return self._seg_puck(self._cur_seg, t)
        return (0.0, 0.0)

    @staticmethod
    def _seg_puck(s: Dict[str, Any], t: float) -> Tuple[float, float]:
        t0, t1 = s["t0"], s["t1"]
        a = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
        if s["kind"] == "carry":
            a = a * a * (3.0 - 2.0 * a)
        return (s["p0"][0] + (s["p1"][0] - s["p0"][0]) * a, s["p0"][1] + (s["p1"][1] - s["p0"][1]) * a)

    def _ensure_plan(self, t: float) -> None:
        A, B = self._pair(t)
        key = (A["key"] if A else None, B["key"] if B else None)
        if key == self.plan_key and self.plan:
            return
        rng = self._rng(key[0], key[1])
        if A is None:
            self.plan = self._buildup(t, (0.0, 0.0), None, None, B, rng)
        elif self.plan_key is not None and key[0] == self.plan_key[0] and self._cur_seg is not None:
            # same interval, future anchor changed (coach re-sim): continue from where we are
            cs = self._cur_seg
            keep = [s for s in self.plan if s["t0"] <= cs["t0"] + 1e-9]
            t0 = cs["t1"]
            p0 = cs["p1"]
            side, car = cs.get("side"), (cs.get("rcv") or cs.get("car"))
            if cs["kind"] in ("dead",) or cs["ph"] in ("goal", "intermission"):
                self.plan = keep + self._buildup(t0, p0, None, None, B, rng)
            else:
                self.plan = keep + self._buildup(t0, p0, side, car, B, rng)
        else:
            tB = self.te(B) if B else self.te(A) + 9999.0
            segs, t1, p, side, car = self._aftermath(A, tB, B, rng)
            if t1 < tB - 0.05 and B is not None:
                segs += self._buildup(t1, p, side, car, B, rng)
            self.plan = segs
            if A["type"] == "period_start":
                self._teleport_faceoff(float(A["t"]) + 0.01, int(A.get("period") or 1))
        self.plan_key = key

    def _seg_at(self, t: float) -> Dict[str, Any]:
        best = None
        for s in self.plan:
            if s["t0"] <= t + 1e-9:
                best = s
            else:
                break
        return best or (self.plan[0] if self.plan else self._seg(t, t + 1, "hold", (0.0, 0.0), (0.0, 0.0), ph="stoppage"))

    # ── players
    def _teleport_faceoff(self, t: float, period: int) -> None:
        if period == 5:
            self.pos.clear()
            return
        lu = self.lineup_at(t + 0.01)
        self.pos = {}
        for side in _SIDES:
            for pid, (x, y) in self._faceoff_spots(side, t, (0.0, 0.0), lu[side]).items():
                self.pos[pid] = [x, y, 0.0, 0.0]
            g = self.goalie_at(side, t)
            if g:
                self.pos[g] = [-_NET_X * self._dir(side, t), 0.0, 0.0, 0.0]

    def _faceoff_spots(self, side: str, t: float, dot: Tuple[float, float], ids: List[str]) -> Dict[str, Tuple[float, float]]:
        d = self._dir(side, t)
        off = {"C": (-2.5, 0.0), "LW": (-3.5, 16.0), "RW": (-3.5, -16.0), "LD": (-24.0, 11.0), "RD": (-24.0, -11.0),
               "F": (-8.0, 6.0), "D": (-24.0, 0.0)}
        out = {}
        for pid, role in _assign_roles(ids, self.loop.book):
            ou, ov = off.get(role, (-10.0, 0.0))
            out[pid] = _clampxy(dot[0] + ou * d, dot[1] + ov * d)
        return out

    def _spots(self, side: str, t: float, seg: Dict[str, Any], puck: Tuple[float, float], ids: List[str],
               n_opp: int) -> Dict[str, Tuple[float, float]]:
        roles = _assign_roles(ids, self.loop.book)
        self.roles.update({p: r for p, r in roles})
        rmap = {r: p for p, r in roles}
        pu, pv = self._rel(side, t, *puck)
        s = 1.0 if pv >= 0 else -1.0
        n = len(ids)
        poss = seg.get("side")
        spot: Dict[str, Tuple[float, float]] = {}

        def put(role_opts: Tuple[str, ...], u: float, v: float) -> None:
            for r in role_opts:
                p = rmap.get(r)
                if p and p not in spot:
                    spot[p] = (u, v)
                    return

        if seg["ph"] in ("faceoff", "stoppage") and seg.get("fo") is not None:
            return self._faceoff_spots(side, t, seg["fo"], ids)
        Wst, Wwk = ("LW", "RW") if s > 0 else ("RW", "LW")
        Dst, Dwk = ("LD", "RD") if s > 0 else ("RD", "LD")
        if seg["ph"] == "goal":
            scorer_side = seg.get("side")
            for pid in ids:
                q = self.pos.get(pid)
                if q is None:
                    continue
                if side == scorer_side:
                    spot[pid] = self._rel(side, t, *_clampxy(puck[0] - 8.0 * self._dir(side, t), puck[1] * 0.6 + (zlib.crc32(pid.encode()) % 9 - 4)))
                else:
                    spot[pid] = self._rel(side, t, q[0], q[1])
        elif poss == side:
            if pu >= 22.0:
                if n > n_opp and n >= 5:  # PP 1-3-1 (+ extra attacker)
                    put((Dst, "LD", "D"), 40.0, 0.0)
                    put((Wst, "LW", "F"), 62.0, 28.0 * s)
                    put((Dwk, "RD", "D"), 62.0, -28.0 * s)
                    put(("C",), 68.0, 0.0)
                    put((Wwk, "RW", "F"), 83.0, 3.0)
                    put(("F", "D"), 76.0, -12.0 * s)
                elif n >= 5:
                    put(("C",), 70.0, -0.25 * pv)
                    put((Wst,), 62.0, 30.0 * s)
                    put((Wwk,), 81.0, -7.0 * s)
                    put((Dst, "D"), 38.0, 23.0 * s)
                    put((Dwk, "D"), 36.0, -14.0 * s)
                    put(("F",), 76.0, -14.0 * s)
                elif n == 4:
                    if n > n_opp:
                        put((Dst, "D"), 40.0, 0.0)
                        put((Wst, "C", "F"), 62.0, 28.0 * s)
                        put(("C", Wwk, "F"), 62.0, -28.0 * s)
                        put((Dwk, Wwk, "D", "F"), 82.0, 3.0)
                    else:
                        put(("C",), 66.0, 0.0)
                        put((Wst, Wwk, "F"), 58.0, 30.0 * s)
                        put((Dst, "D"), 38.0, 20.0 * s)
                        put((Dwk, "D"), 38.0, -20.0 * s)
                else:
                    put(("C",), 64.0, 14.0 * s)
                    put((Wst, Wwk, "F"), 72.0, -16.0 * s)
                    put((Dst, Dwk, "D"), 38.0, 0.0)
            else:
                put(("C",), min(pu + 10.0, 70.0), 0.2 * pv)
                put((Wst,), min(pu + 22.0, 72.0), 28.0 * s)
                put((Wwk,), min(pu + 18.0, 70.0), -26.0 * s)
                put((Dst, "D"), max(-86.0, min(pu - 18.0, 30.0)), 16.0 * s)
                put((Dwk, "D"), max(-86.0, min(pu - 20.0, 28.0)), -16.0 * s)
                put(("F",), min(pu + 14.0, 70.0), -10.0 * s)
        else:
            fwd = [p for p, r in roles if r in ("C", "LW", "RW", "F")]
            f1 = None
            if fwd and poss is not None:
                f1 = min(fwd, key=lambda p: math.hypot((self.pos.get(p) or [0, 0])[0] - puck[0], (self.pos.get(p) or [0, 0])[1] - puck[1]))
            if pu <= -22.0:
                if n < n_opp or (n == 4 and n_opp >= 5):  # PK box / triangle
                    sh = 0.25 * pv
                    if n >= 4:
                        put(("C", "LW", "F"), -58.0, 12.0 + sh)
                        put(("LW", "RW", "C", "F"), -58.0, -12.0 + sh)
                        put(("LD", "D"), -80.0, 10.0 + 0.2 * pv)
                        put(("RD", "D"), -80.0, -10.0 + 0.2 * pv)
                    else:
                        put(("C", "LW", "F"), -60.0, sh)
                        put(("LD", "D"), -80.0, 9.0)
                        put(("RD", "D"), -80.0, -9.0)
                else:
                    put(("C",), -72.0, 0.3 * pv)
                    put((Wst,), -40.0, 22.0 * s)
                    put((Wwk,), -42.0, -20.0 * s)
                    put((Dst, "D"), -80.0, 13.0 * s)
                    put((Dwk, "D"), -83.0, -4.0 * s)
                    put(("F",), -60.0, -8.0 * s)
            else:
                put(("C",), max(-84.0, pu - 12.0), 0.3 * pv)
                put(("LW",), max(-84.0, pu - 14.0), 18.0)
                put(("RW",), max(-84.0, pu - 14.0), -18.0)
                put(("LD", "D"), max(-84.0, min(pu - 32.0, 40.0)), 14.0)
                put(("RD", "D"), max(-84.0, min(pu - 32.0, 40.0)), -14.0)
                put(("F",), max(-84.0, pu - 20.0), 0.0)
            if f1 is not None:
                nu, nv = -1.0, 0.0
                spot[f1] = (pu + nu * 4.0, pv + nv)
            # shadow: blend toward the nearest attacker, staying on the net side
            opp_ids = [p for p in self.lineup_at(t)[_opp(side)] if p in self.pos]
            for pid, (u, v) in list(spot.items()):
                if pid == f1 or not opp_ids:
                    continue
                ax, ay = self._abs(side, t, u, v)
                near = min(opp_ids, key=lambda q: math.hypot(self.pos[q][0] - ax, self.pos[q][1] - ay))
                qu, qv = self._rel(side, t, self.pos[near][0], self.pos[near][1])
                if math.hypot(qu - u, qv - v) < 20.0:
                    spot[pid] = (0.65 * u + 0.35 * (qu - 5.0), 0.65 * v + 0.35 * qv)
        for pid in ids:
            if pid not in spot:
                spot[pid] = (pu - 10.0, 0.0)
        return {pid: self._abs(side, t, u, v) for pid, (u, v) in spot.items()}

    def _step(self, t: float, dt: float, seg: Dict[str, Any], lineup: Dict[str, List[str]], goalies: Dict[str, Optional[str]]) -> Tuple[float, float]:
        puck = self._seg_puck(seg, t)
        targets: Dict[str, Tuple[float, float]] = {}
        for side in _SIDES:
            targets.update(self._spots(side, t, seg, puck, lineup[side], len(lineup[_opp(side)])))
        if seg["kind"] == "carry" and seg.get("car") in targets:
            targets[seg["car"]] = puck
        nxt_rcv = None
        for s in self.plan:
            if s["t0"] >= t - 1e-9 and s["t0"] <= t + 2.0 and s["kind"] == "pass" and s.get("rcv"):
                nxt_rcv = s
                break
        if seg["kind"] == "pass" and seg.get("rcv") in targets:
            targets[seg["rcv"]] = seg["p1"]
        elif nxt_rcv is not None and nxt_rcv["rcv"] in targets and nxt_rcv["rcv"] != seg.get("car"):
            targets[nxt_rcv["rcv"]] = nxt_rcv["p1"]
        if seg["kind"] == "loose" and seg["ph"] == "play":
            for side in _SIDES:
                c = None
                bd = 1e9
                for pid in lineup[side]:
                    q = self.pos.get(pid)
                    if q is not None:
                        d = math.hypot(q[0] - puck[0], q[1] - puck[1])
                        if d < bd:
                            c, bd = pid, d
                if c:
                    targets[c] = puck
        slow = seg["ph"] in ("goal", "intermission")
        for pid, (tx, ty) in targets.items():
            st = self.pos.get(pid)
            if st is None:
                continue
            dx, dy = tx - st[0], ty - st[1]
            dist = math.hypot(dx, dy)
            vmax = 9.0 if slow else _V_SKATE
            want = min(vmax, dist * 1.8)
            if dist > 1e-6:
                vx, vy = dx / dist * want, dy / dist * want
            else:
                vx = vy = 0.0
            ax, ay = vx - st[2], vy - st[3]
            amag = math.hypot(ax, ay)
            lim = _A_SKATE * dt
            if amag > lim:
                ax, ay = ax / amag * lim, ay / amag * lim
            st[2] += ax
            st[3] += ay
            sp = math.hypot(st[2], st[3])
            if sp > _V_SKATE:
                st[2], st[3] = st[2] / sp * _V_SKATE, st[3] / sp * _V_SKATE
            st[0], st[1] = _clampxy(st[0] + st[2] * dt, st[1] + st[3] * dt)
        for side in _SIDES:
            g = goalies.get(side)
            if not g:
                continue
            st = self.pos.get(g)
            od = self._dir(side, t)
            nx = -_NET_X * od
            if st is None:
                st = [nx, 0.0, 0.0, 0.0]
                self.pos[g] = st
            vx, vy = puck[0] - nx, puck[1]
            if vx * (-od) >= 0:  # puck behind the goal line
                tx, ty = nx + od * 1.0, max(-3.0, min(3.0, puck[1]))
            else:
                m = math.hypot(vx, vy) or 1.0
                tx, ty = nx + vx / m * 4.5, vy / m * 4.5
            dx, dy = tx - st[0], ty - st[1]
            d = math.hypot(dx, dy)
            step = min(d, _V_GOALIE * dt)
            if d > 1e-6:
                st[0] += dx / d * step
                st[1] += dy / d * step
        return puck

    def _sync_lineup(self, t: float, lineup: Dict[str, List[str]], goalies: Dict[str, Optional[str]]) -> None:
        keep = set(lineup["home"]) | set(lineup["away"]) | {g for g in goalies.values() if g}
        for pid in list(self.pos):
            if pid not in keep:
                del self.pos[pid]
        for side in _SIDES:
            for pid in lineup[side]:
                if pid not in self.pos:
                    bx, by = _BENCH[side]
                    self.pos[pid] = [bx + (zlib.crc32(pid.encode()) % 7 - 3) * 1.5, by, 0.0, 0.0]

    # ── public
    def frame_at(self, t: float) -> Dict[str, Any]:
        self._ensure_plan(t)
        lineup = self.lineup_at(t)
        goalies = {s: self.goalie_at(s, t) for s in _SIDES}
        seg = self._seg_at(t)
        so = seg["ph"] == "shootout"
        if so:
            car = seg.get("car")
            lineup = {"home": [], "away": []}
            if car and self.loop.book.side.get(car) in _SIDES:
                lineup[self.loop.book.side[car]] = [car]
                if car not in self.pos:
                    x0, y0 = seg["p0"]
                    self.pos[car] = [x0, y0, 0.0, 0.0]
        if not so:
            lineup = {s: list(v) for s, v in lineup.items()}
            for s in _SIDES:
                if goalies[s] is None and lineup[s]:  # pulled goalie: sixth skater
                    pool = [p for grp in (self.loop.lines[s].get("F") or []) for p in grp if p not in lineup[s]]
                    if pool:
                        lineup[s].append(pool[0])
        self._sync_lineup(t, lineup, goalies)
        for sub in (0.25, 0.5):
            tt = t - 0.5 + sub
            sg = self._seg_at(tt)
            self._cur_seg = sg
            puck = self._step(tt, 0.25, sg, lineup, goalies)
        self._cur_seg = seg
        puck = self._seg_puck(seg, t)
        if seg["why"] is not None and seg["kind"] in ("dead", "hold") and seg["ph"] in ("stoppage", "goal", "intermission"):
            self.last_whistle = seg["why"]
        c = seg.get("car") if seg["kind"] in ("carry", "hold") else None
        side = seg.get("side")
        if c is not None and (side not in _SIDES or c not in lineup.get(side, [])):
            c = None
        pos_side = side if seg["kind"] in ("carry", "pass", "shot") or (seg["kind"] == "hold" and c) else None
        players = []
        for s in _SIDES:
            for pid in lineup[s] + ([goalies[s]] if goalies[s] else []):
                st = self.pos.get(pid)
                if st is not None:
                    players.append([pid, round(st[0], 1), round(st[1], 1)])
        f = {"t": round(t, 2), "ph": seg["ph"], "pk": [round(puck[0], 1), round(puck[1], 1)], "c": c, "pos": pos_side, "p": players}
        self.last_frame = f
        return f

    def generate_to(self, t_end: float, *, skip: bool = False) -> Dict[str, Any]:
        t = 0.0 if self.t_last is None else self.t_last + 0.5
        frames: List[Dict[str, Any]] = []
        if self.loop.finished:
            fin = next((r for r in reversed(self.loop._timeline) if r.get("type") == "final"), None)
            if fin is not None:
                t_end = max(float(t_end), self.te(fin) + 1.0)
        if skip and t_end - t > 30.0:
            t = math.floor((t_end - 10.0) * 2.0) / 2.0
            self.plan_key = None
            self.plan = []
            self.pos = {}
        guard = 0
        while t <= float(t_end) + 1e-9 and guard < 2400:
            try:
                frames.append(self.frame_at(t))
            except Exception:
                logger.warning("motion frame failed at t=%.1f", t, exc_info=True)
                if self.last_frame is not None:
                    frames.append(dict(self.last_frame, t=round(t, 2)))
            self.t_last = t
            t += 0.5
            guard += 1
        start = frames[0]["t"] if frames else (self.t_last if self.t_last is not None else 0.0)
        return {"from_t": start, "to_t": self.t_last if self.t_last is not None else 0.0, "step": 0.5,
                "frames": frames, "authority": "model_estimated", "skipped": bool(skip)}

    def situation_key(self) -> Optional[Tuple[str, Optional[str]]]:
        f = self.last_frame
        if not f:
            return None
        if f["ph"] in ("stoppage", "faceoff"):
            return ("stop", None)
        if f["ph"] != "play" or f.get("pos") not in _SIDES or not f.get("pk"):
            return None
        side = f["pos"]
        u, _v = self._rel(side, f["t"], f["pk"][0], f["pk"][1])
        return ("oz" if u > 25.0 else ("dz" if u < -25.0 else "nz"), side)

    def zone(self) -> Optional[str]:
        f = self.last_frame
        if not f or not f.get("pk"):
            return None
        hx = f["pk"][0] * self._dir("home", f["t"])
        return "home_d" if hx < -25.0 else ("away_d" if hx > 25.0 else "neutral")


_CLOCK_NOTE = ("frames: t == pt (presentation time, contiguous 0.5 s, keeps running through stoppages); "
               "gc = game clock (engine seconds, frozen during stoppages/celebrations/faceoff set-ups). "
               "Play frames by pt, display gc. Events: t = gc, pt = when to show it.")


def _make_motion(loop: "LiveEventLoop") -> Any:
    """Motion v2 (services/live_motion.py: rules staging + pt/gc clocks); falls back to MotionTrack."""
    try:
        from services import live_motion as _lm

        mt = _lm.MotionV2(loop)
        loop._motion_v2 = True
        return mt
    except Exception:
        logger.warning("live_motion unavailable, using the built-in motion track", exc_info=True)
        loop._motion_v2 = False
        return MotionTrack(loop)


class ShiftModel:
    """Estimated TOI / shifts / fatigue from the units the engine actually put on the ice
    at each attempt (scaled so a team's skater-seconds match elapsed time)."""

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.toi: Dict[str, float] = {}
        self.shifts: Dict[str, int] = {}
        self.on_since: Dict[str, float] = {}
        self.last_off: Dict[str, float] = {}
        self.cur: Dict[str, set] = {"home": set(), "away": set()}
        self.last_t = 0.0

    def _credit(self, t: float) -> None:
        dt = max(0.0, min(90.0, t - self.last_t))
        if dt > 0:
            for side in _SIDES:
                for pid in self.cur[side]:
                    self.toi[pid] = self.toi.get(pid, 0.0) + dt
        self.last_t = max(self.last_t, t)

    def observe(self, t: float, on_ice: Dict[str, List[str]]) -> None:
        self._credit(t)
        for side in _SIDES:
            new = set((on_ice or {}).get(side) or [])
            if not new:
                continue
            for pid in self.cur[side] - new:
                self.on_since.pop(pid, None)
                self.last_off[pid] = t
            for pid in new - self.cur[side]:
                self.shifts[pid] = self.shifts.get(pid, 0) + 1
                self.on_since[pid] = t
            self.cur[side] = new

    def advance(self, t: float) -> None:
        self._credit(t)

    def intermission(self, t: float) -> None:
        for side in _SIDES:
            for pid in list(self.cur[side]):
                self.on_since[pid] = t

    def rest(self, side: str, t: float) -> None:
        for pid in list(self.cur.get(side) or []):
            self.on_since[pid] = t
        for pid in list(self.last_off):
            if self.loop.book.side.get(pid) == side:
                self.last_off[pid] = t - 600.0

    def toi_est(self, pid: str, side: str) -> float:
        raw = self.toi.get(pid, 0.0)
        tot = sum(v for p, v in self.toi.items() if self.loop.book.side.get(p) == side)
        target = 5.0 * min(self.last_t, _REGULATION_GAME_SEC + 1200.0)
        scale = min(3.0, target / tot) if tot > 0 else 1.0
        return raw * scale

    def fatigue(self, pid: str, t: float) -> float:
        side = self.loop.book.side.get(pid, "home")
        base = 0.25 * min(1.0, self.toi_est(pid, side) / 1500.0)
        if pid in self.on_since:
            return round(min(1.0, base + 0.75 * min(1.0, max(0.0, t - self.on_since[pid]) / 75.0)), 3)
        off = t - self.last_off.get(pid, -1e9)
        return round(min(1.0, base + 0.35 * math.exp(-max(0.0, off) / 45.0)), 3)


# ═══ 6. Coaches


def _parse_line_ref(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """{line: "F2"} / {line: "D1"} (1-based) or {F: 0-based, D: 0-based}."""
    F = D = None
    ref = payload.get("line")
    if isinstance(ref, str) and len(ref.strip()) >= 2:
        r = ref.strip().upper()
        try:
            n = int(r[1:]) - 1
        except ValueError:
            n = None
        if n is not None:
            if r[0] in ("F", "L"):
                F = n
            elif r[0] == "D":
                D = n
    elif isinstance(ref, (int, float)):
        F = int(ref) - 1
    for key, cur in (("F", F), ("D", D)):
        v = payload.get(key)
        if cur is None and v is not None:
            try:
                if key == "F":
                    F = int(v)
                else:
                    D = int(v)
            except (TypeError, ValueError):
                pass
    return F, D


class CoachController:
    """User orders: validated, applied immediately by re-simulating the rest of the game
    from the next unplayed attempt (same seed => identical past), or rejected with a reason."""

    ALLOWED = ("roll_lines", "shorten_bench", "next_line", "bench_player", "pull_goalie", "timeout", "tactics")

    def __init__(self, loop: "LiveEventLoop") -> None:
        self.loop = loop
        self.orders: List[Dict[str, Any]] = []
        self.pending: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self.last_result: Optional[Dict[str, Any]] = None

    def applied(self) -> List[Dict[str, Any]]:
        return [dict(o) for o in self.orders if o.get("status") == "applied"]

    def _finish(self, order: Dict[str, Any], status: str, reason: str, effect: str) -> Dict[str, Any]:
        order["status"] = status
        order["reason"] = reason
        order["effect"] = effect
        self.orders.append(order)
        self.orders = self.orders[-60:]
        self.last_result = {"status": status, "reason": reason or effect}
        self.loop.lg.coach_orders = self.applied()
        return dict(order)

    def _validate(self, kind: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        loop = self.loop
        user = loop.lg.user_side
        if loop.finished or loop.lg.status in ("final", "committed", "aborted"):
            return False, "The game is over."
        if loop.phase == "shootout":
            return False, "No coaching changes during the shootout."
        if kind not in self.ALLOWED:
            return False, f"Unknown order kind: {kind}"
        if kind == "pull_goalie":
            if _truthy(payload.get("pull"), True):
                if loop.goalie_pulled.get(user):
                    return False, "The goalie is already pulled."
                mp = loop.manpower(loop.game_sec_elapsed)
                me = mp["home_skaters"] if user == "home" else mp["away_skaters"]
                them = mp["away_skaters"] if user == "home" else mp["home_skaters"]
                if me < them:
                    return False, "Can't pull the goalie while shorthanded."
            elif not loop.goalie_pulled.get(user):
                return False, "The goalie is not pulled."
        elif kind == "timeout":
            if loop.timeout_used:
                return False, "Timeout already used."
            if loop.intermission:
                return False, "Can't call a timeout during intermission."
        elif kind == "next_line":
            ref = str(payload.get("line") or "").strip().upper()
            if ref.startswith(("PP", "PK")):
                return False, "Power-play and penalty-kill units are picked by the coach AI; use pp_style / pk_style tactics."
            F, D = _parse_line_ref(payload)
            if F is None and D is None:
                return False, "Pick a line (F1-F4) or pair (D1-D3)."
            if F is not None and not 0 <= F <= 3:
                return False, "Forward lines are F1-F4."
            if D is not None and not 0 <= D <= 2:
                return False, "Defence pairs are D1-D3."
        elif kind == "bench_player":
            pid = str(payload.get("player_id") or "")
            if not pid or loop.book.side.get(pid) != user:
                return False, "That player isn't dressed for your team."
            if loop.book.pos(pid) == "G":
                return False, "Use pull_goalie for goalies."
            if "double_shift" in payload:
                if _truthy(payload.get("double_shift")) and pid in loop.double_shift:
                    return False, "Already double-shifting."
                if not _truthy(payload.get("double_shift")) and pid not in loop.double_shift:
                    return False, "Not double-shifting."
                if _truthy(payload.get("double_shift")) and pid in loop.benched:
                    return False, "That player is benched."
            else:
                bench = _truthy(payload.get("bench"), pid not in loop.benched)
                if bench and len(loop.benched) >= 3:
                    return False, "You can bench at most three skaters."
                if not bench and pid not in loop.benched:
                    return False, "That player isn't benched."
        elif kind == "tactics":
            tac = {k: str(payload.get(k)).lower() for k in _TACTIC_KEYS if k in payload}
            if not tac:
                return False, "No tactic given (forecheck, defense, shot_selection, pp_style, pk_style)."
            bad = [k for k, v in tac.items() if v not in _TACTIC_VALUES]
            if bad:
                return False, f"Tactic values are passive, balanced or aggressive ({', '.join(bad)})."
        return True, ""

    def submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate now; the engine re-simulation runs at the start of the next /advance
        (status "applying" until then) so /coach always answers fast."""
        loop = self.loop
        kind = str((body or {}).get("kind") or "").strip()
        payload = dict((body or {}).get("payload") or {})
        t = float(loop.game_sec_elapsed)
        if kind == "notification_ack":  # bookkeeping only: no engine re-simulation
            res = loop.analytics.ack(str(payload.get("id") or ""), str(payload.get("status") or "dismissed")) if loop.analytics is not None else None
            return {"id": str(uuid.uuid4()), "kind": kind, "payload": payload, "issued_t": round(t, 2), "effective_from_t": round(t, 2),
                    "status": "applied" if res else "rejected", "reason": "" if res else "Unknown notification.",
                    "effect": "notification marked " + str((res or {}).get("status") or ""), "notification": res}
        order = {
            "id": str(uuid.uuid4()), "kind": kind, "payload": payload, "issued_t": round(t, 2),
            "effective_from_t": round(t, 2), "status": "queued", "reason": "", "effect": "",
        }
        ok, why = self._validate(kind, payload)
        if not ok:
            return self._finish(order, "rejected", why, "")
        if kind == "timeout" and any(k == "timeout" for k, _p, _o in self.pending):
            return self._finish(order, "rejected", "Timeout already requested.", "")
        if not loop._engine_captured:
            status, reason, effect = self._apply_batch([(kind, payload, order)])[0]
            return self._finish(order, status, reason, effect)
        order["status"] = "applying"
        order["effect"] = "Re-simulating the rest of the game with this order; applied on the next advance."
        self.pending.append((kind, payload, order))
        self.last_result = {"status": "applying", "reason": ""}
        return dict(order)

    def pending_orders(self) -> List[Dict[str, Any]]:
        return [dict(o) for _k, _p, o in self.pending]

    def process_pending(self) -> int:
        if not self.pending:
            return 0
        batch, self.pending = self.pending, []
        try:
            results = self._apply_batch(batch)
        except Exception as e:
            logger.exception("live coach batch failed")
            results = [("rejected", f"Order failed ({type(e).__name__}).", "")] * len(batch)
        for (_k, _p, order), (status, reason, effect) in zip(batch, results):
            self._finish(order, status, reason, effect)
        return len(batch)

    def _apply_batch(self, batch: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> List[Tuple[str, str, str]]:
        loop = self.loop
        effects = [dict(e) for e in loop._effects]
        prepared: List[Optional[Tuple[Dict[str, Any], str]]] = []
        results: List[Optional[Tuple[str, str, str]]] = []
        for kind, payload, order in batch:
            ok, why = self._validate(kind, payload)
            if not ok:
                prepared.append(None)
                results.append(("rejected", why, ""))
                continue
            pres, human = self._effects_for(kind, payload, order, effects)
            prepared.append((pres, human))
            results.append(None)
        if all(p is None for p in prepared):
            return [r or ("rejected", "", "") for r in results]
        ok, why = loop._resim(effects)
        if ok:
            if not loop._engine_captured:
                where = "from the opening faceoff"
            else:
                p, left = loop.clock_at(loop.game_sec_elapsed)
                where = f"from {_clock_display(left)} of the {_period_label(p)} on (the next unplayed shot attempt)"
        out: List[Tuple[str, str, str]] = []
        for (kind, payload, order), prep, res in zip(batch, prepared, results):
            if res is not None:
                out.append(res)
                continue
            pres, human = prep  # type: ignore[misc]
            if ok:
                effect = f"{human}. The rest of the game was re-simulated {where} with this order; the score can change."
                order["effect"] = effect
                loop.apply_order_presentation(pres, order)
                out.append(("applied", "", effect))
            elif kind == "timeout":
                effect = f"Timeout taken (bench rested in the fatigue model). Play was NOT re-simulated ({why}); the result is unchanged."
                order["effect"] = effect
                loop.apply_order_presentation(pres, order)
                out.append(("applied", why, effect))
            else:
                out.append(("rejected", f"Could not apply this order: {why}.", ""))
        return out

    def _effects_for(self, kind: str, payload: Dict[str, Any], order: Dict[str, Any],
                     effects: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        loop = self.loop
        user = loop.lg.user_side
        k = loop._emitted_attempts
        base = {"id": order["id"], "order_id": order["id"], "side": user, "from_idx": k, "by": "user"}
        pres: Dict[str, Any] = {"kind": kind}
        lang = loop.lg.lang
        W = lambda key, **kw: _fmt(_loc(_COACH_WHAT, lang, key), kw)  # noqa: E731
        if kind == "pull_goalie":
            pull = _truthy(payload.get("pull"), True)
            effects.append({**base, "kind": "pull", "pull": pull})
            human = "Goalie pulled for an extra attacker until the next goal" if pull else "Goalie back in net"
            pres.update(action="pull" if pull else "return", what=W("pull" if pull else "return"), pull=pull)
        elif kind == "timeout":
            effects.append({**base, "kind": "rest", "count": 2})
            human = "Timeout: top line and top pair rested and sent out for the next shifts"
            pres.update(action="timeout", what=W("timeout"))
        elif kind in ("shorten_bench", "roll_lines"):
            on = _truthy(payload.get("on"), True)
            mode = ("short" if kind == "shorten_bench" else "roll") if on else "normal"
            effects.append({**base, "kind": "deploy", "mode": mode})
            human = {"short": "Bench shortened: 4th line and 3rd pair sat at even strength",
                     "roll": "Rolling all four lines and three pairs evenly",
                     "normal": "Normal rotation restored"}[mode]
            pres.update(action=kind if on else "normal_rotation", what=W(kind if on else "normal_rotation"), mode=mode)
        elif kind == "next_line":
            F, D = _parse_line_ref(payload)
            effects.append({**base, "kind": "next_line", "F": F, "D": D, "count": 2})
            unit = " / ".join(x for x in (
                W("line", n=F + 1) if F is not None else "", W("pair", n=D + 1) if D is not None else "") if x)
            human = "Next shifts: " + " ".join(x for x in (f"F{F + 1}" if F is not None else "", f"D{D + 1}" if D is not None else "") if x)
            pres.update(action="next_line", what=W("next_line", unit=unit))
        elif kind == "bench_player":
            pid = str(payload.get("player_id"))
            pname = loop.book.short(pid) or loop.book.name(pid)
            if "double_shift" in payload:
                on = _truthy(payload.get("double_shift"))
                if on:
                    effects.append({**base, "kind": "double", "player_id": pid, "until_idx": None})
                else:
                    for e in effects:
                        if e.get("kind") == "double" and str(e.get("player_id")) == pid and e.get("until_idx") is None:
                            e["until_idx"] = k
                human = f"{pname} double-shifting (takes 3rd/4th-line shifts)" if on else f"{pname} back to his own line"
                pres.update(action="double_shift" if on else "unbench", what=W("next_line", unit=pname) if on else W("unbench", player=pname),
                            player_id=pid, double=on)
            else:
                bench = _truthy(payload.get("bench"), pid not in loop.benched)
                if bench:
                    effects.append({**base, "kind": "bench", "player_id": pid, "until_idx": None})
                else:
                    for e in effects:
                        if e.get("kind") == "bench" and str(e.get("player_id")) == pid and e.get("until_idx") is None:
                            e["until_idx"] = k
                human = f"{pname} benched" if bench else f"{pname} back in the rotation"
                pres.update(action="bench_player" if bench else "unbench",
                            what=W("bench_player" if bench else "unbench", player=pname), player_id=pid, bench=bench)
        else:  # tactics
            tac = dict(loop.tactics)
            for e in effects:
                if e.get("kind") == "tactics" and e.get("side") == user:
                    tac.update(e.get("tactics") or {})
            tac.update({kk: str(payload[kk]).lower() for kk in _TACTIC_KEYS if kk in payload})
            effects.append({**base, "kind": "tactics", "tactics": tac})
            detail = ", ".join(f"{kk.replace('_', ' ')} {payload[kk]}" for kk in _TACTIC_KEYS if kk in payload)
            human = f"Tactics: {detail}"
            pres.update(action="tactics", what=W("tactics", detail=detail), tactics=tac)
        return pres, human


# ═══ 7. Live loop


class LiveEventLoop:
    """Live game built from authoritative SimEngine passes, stepped for the client.

    Time source: the engine's own `game_sec_elapsed` inside `_simulate_segment`
    (read from the hooked frame at every shot attempt) for regulation; overtime uses
    the engine's per-segment attempt clock, scaled to fit the OT period. Coach orders
    re-run the engine with the same seed and the order gated to attempts that have not
    been shown yet; the already-shown prefix is verified identical before splicing.
    """

    def __init__(self, lg: LiveGame, session: Any) -> None:
        self.lg = lg
        self.session = session
        self.sim = lg.sim
        self.rng = random.Random(int(lg.seed) ^ 0x7E1C)
        self.book = _PlayerBook()
        self.game_sec_elapsed = 0.0
        self.phase = "pregame"
        self.finished = False
        self.intermission = False
        self._int_period = 0
        self._skip_intermission = False
        self.timeout_used = False
        self.cpu_timeout_used = False
        self.goalie_pulled = {"home": False, "away": False}
        self.tactics = _default_tactics()
        self.deploy_mode = "normal"
        self.benched: set = set()
        self.double_shift: set = set()
        self._engine_captured = False
        self._result: Optional[Dict[str, Any]] = None
        self._scratch: Dict[str, Dict[str, Any]] = {}
        self._cap: Optional[_EnginePass] = None
        self._timeline: List[Dict[str, Any]] = []
        self._timeline_idx = 0
        self._emitted_attempts = 0
        self._effects: List[Dict[str, Any]] = []
        self._so_rounds: List[Dict[str, Any]] = []
        self._so_tally = {"home": 0, "away": 0}
        self._goals_for_t: Dict[str, List[float]] = {"home": [], "away": []}
        self.home_goalie = (lg.scratch or {}).get("home_goalie")
        self.away_goalie = (lg.scratch or {}).get("away_goalie")
        self.ot_len = 300.0
        try:
            self.ot_len = 60.0 * float((self.sim._board_game_rules() or {}).get("ot_minutes", 5) or 5)
        except Exception:
            logger.debug("board rules unavailable", exc_info=True)
        self.lines: Dict[str, Dict[str, List[Any]]] = {"home": {"F": [], "D": [], "G": []}, "away": {"F": [], "D": [], "G": []}}
        self.home_scale = float(lg.strength.get("home_scale", 1.0))
        self.away_scale = float(lg.strength.get("away_scale", 1.0))
        self.noise_scale = float(lg.strength.get("noise_scale", 1.0))
        self._strength_map: Dict[str, float] = {}
        self._team_cards: Optional[Dict[str, Any]] = None
        self.fan_meter: Optional[float] = None
        self.rivalry: Dict[str, Any] = {"tier": 0, "name_key": None}
        self.state = _GameState(self)
        self.coach = CoachController(self)
        self.momentum = MomentumModel(self)
        self.crowd = CrowdModel(self)
        self.alerts = AlertEngine(self)
        self.commentary = CommentaryWriter(self)
        try:
            from services.live_analytics import LiveAnalytics
            self.analytics: Any = LiveAnalytics(self)
        except Exception:
            logger.warning("live analytics unavailable", exc_info=True)
            self.analytics = None
        self._an_t = 0.0
        self.tracking = TrackingModel(self)
        self.shifts = ShiftModel(self)
        self.motion = _make_motion(self)
        self._pull_hist: List[Tuple[float, str, bool]] = []
        self.last_pass_ms = 0.0
        self._motion_on = True
        self._motion_buf: List[Dict[str, Any]] = []
        self._motion_v2 = bool(getattr(self, "_motion_v2", False))
        self._snap_cache: Tuple[Any, Optional[Dict[str, Any]]] = (None, None)
        self._roster_cache: Tuple[Any, Dict[str, Any]] = (None, {})

    @classmethod
    def attach(cls, lg: LiveGame, session: Any) -> "LiveEventLoop":
        loop = cls(lg, session)
        loop.bootstrap_pregame()
        return loop

    # ── setup
    def bootstrap_pregame(self) -> None:
        self._strength_map = dict(getattr(self.session, "strength_map", None) or {})
        try:
            h_m, a_m, _ = self.sim._tactic_game_goal_mults(self.lg.home, self.lg.away, home_b2b=False, away_b2b=False, event_path=True)
            self.home_scale *= float(h_m)
            self.away_scale *= float(a_m)
        except Exception:
            logger.debug("live: tactic goal mults unavailable", exc_info=True)  # returns None on the event path
        for side, team in (("home", self.lg.home), ("away", self.lg.away)):
            dep = getattr(team, "_franchise_deployed_lineup", None)
            if isinstance(dep, dict) and dep.get("ok"):
                F = [[self.book.add(p, side) for p in (ln or [])] for ln in (dep.get("forward_lines") or [])][:4]
                D = [[self.book.add(p, side) for p in (pr or [])] for pr in (dep.get("defense_pairs") or [])][:3]
                self.lines[side]["F"], self.lines[side]["D"] = F, D
            g = self.home_goalie if side == "home" else self.away_goalie
            if g is not None:
                self.lines[side]["G"] = [self.book.add(g, side)]
        try:
            from services import franchise_sim as fs

            prof = fs._ensure_team_fan_profile(self.session, self.lg.home_id)
            self.fan_meter = float(prof.get("fan_confidence")) if prof.get("fan_confidence") is not None else None
        except Exception:
            logger.warning("live: fan meter unavailable", exc_info=True)
        self.rivalry = self._read_rivalry()

    def _read_rivalry(self) -> Dict[str, Any]:
        tier, key = 0, None
        try:
            from services.legacy_hall import rivalries

            for r in (rivalries(self.session, tid=self.lg.home_id, limit=50) or {}).get("rivalries") or []:
                if {str(x) for x in r.get("teams") or []} == {str(self.lg.home_id), str(self.lg.away_id)}:
                    s = float(r.get("score") or 0.0)
                    tier = 3 if s >= 70 else (2 if s >= 40 else 1)
                    key = tier
                    break
        except Exception:
            logger.debug("rivalry lookup failed", exc_info=True)
        try:
            hd = str(getattr(self.lg.home, "division", "") or "")
            ad = str(getattr(self.lg.away, "division", "") or "")
            if hd and hd == ad and tier < 1:
                tier, key = 1, "div"
        except Exception:
            pass
        return {"tier": tier, "name_key": key}

    # ── names / helpers
    def abbr(self, side: Optional[str]) -> Optional[str]:
        if side not in _SIDES:
            return None
        cards = self.team_cards()
        return (cards.get(side) or {}).get("abbr")

    def team_cards(self) -> Dict[str, Any]:
        if self._team_cards is None:
            lg = self.lg
            self._team_cards = {
                "home": _team_card(self.session, lg.home, lg.home_id, is_user=(lg.user_side == "home")),
                "away": _team_card(self.session, lg.away, lg.away_id, is_user=(lg.user_side == "away")),
            }
        return self._team_cards

    def team_nick(self, side: str) -> str:
        name = str((self.team_cards().get(side) or {}).get("name") or "")
        city = str((self.team_cards().get(side) or {}).get("city") or "")
        if city and name.startswith(city):
            nick = name[len(city):].strip()
            if nick:
                return nick
        return _short_name(name) or (self.abbr(side) or "")

    def team_say(self, side: Optional[str], lang: str) -> str:
        if side not in _SIDES:
            return ""
        card = self.team_cards().get(side) or {}
        return str(card.get("city") or "") or self.team_nick(side)

    def current_goalie(self, side: str) -> Optional[str]:
        if self.goalie_pulled.get(side):
            return None
        g = self.lines[side].get("G") or []
        return g[0] if g else None

    def line_index_of(self, pid: str) -> Optional[str]:
        side = self.book.side.get(pid)
        if side not in _SIDES:
            return None
        for j, ln in enumerate(self.lines[side].get("F") or []):
            if pid in ln:
                return f"F{j + 1}"
        for j, pr in enumerate(self.lines[side].get("D") or []):
            if pid in pr:
                return f"D{j + 1}"
        return None

    # ── clock (single source of truth)
    def _ot_seg_len(self) -> float:
        return _REGULATION_PERIOD_SEC if self.lg.is_playoff else self.ot_len

    def clock_at(self, t: float, period: Optional[int] = None) -> Tuple[int, float]:
        t = float(t)
        if period is None:
            period = min(3, 1 + int(t // _REGULATION_PERIOD_SEC)) if t < _REGULATION_GAME_SEC else 4
        period = int(period)
        if period >= 5:
            return 5, 0.0
        if period <= 3:
            return period, max(0.0, period * _REGULATION_PERIOD_SEC - t)
        seg = self._ot_seg_len()
        into = max(0.0, t - _REGULATION_GAME_SEC)
        if self.lg.is_playoff:
            into = into % seg
        return 4, max(0.0, seg - into)

    def clock_state(self) -> Dict[str, Any]:
        t = float(self.game_sec_elapsed)
        if self.phase == "shootout" or (self.finished and self._so_rounds):
            period, left = 5, 0.0
        elif self.intermission:
            period, left = self._int_period or self.clock_at(t)[0], 0.0
        elif self.finished:
            period, left = self.clock_at(t)
            if t >= _REGULATION_GAME_SEC and not (self._result or {}).get("overtime"):
                period, left = 3, 0.0
        else:
            period, left = self.clock_at(t)
        return {
            "period": period, "period_label": _period_label(period, shootout=(period == 5)),
            "sec_elapsed_game": round(t, 2), "sec_left_period": round(left, 2), "display": _clock_display(left),
            "intermission": bool(self.intermission),
        }

    # ── manpower
    def manpower(self, t: float) -> Dict[str, Any]:
        h = a = 5
        pp_side: Optional[str] = None
        pp_left: Optional[float] = None
        cap = self._cap
        if t >= _REGULATION_GAME_SEC and (self._result or {}).get("overtime"):
            h = a = 5 if self.lg.is_playoff else 3
        elif cap is not None:
            act = [p for p in cap.penalties if p["start"] <= t < p["start"] + p["dur"]]
            hb = sum(1 for p in act if p["side"] == "home")
            ab = sum(1 for p in act if p["side"] == "away")
            h, a = max(3, 5 - hb), max(3, 5 - ab)
            if h != a:
                short = "home" if h < a else "away"
                pp_side = _opp(short)
                lefts = [p["start"] + p["full"] - t for p in act if p["side"] == short]
                pp_left = round(max(0.0, min(lefts)), 1) if lefts else None
        user = self.lg.user_side
        mine, theirs = (h, a) if user == "home" else (a, h)
        if self.goalie_pulled["home"] or self.goalie_pulled["away"]:
            state = "EN"
        elif mine > theirs:
            state = "PP"
        elif mine < theirs:
            state = "SH"
        else:
            state = {5: "EV", 4: "4v4", 3: "3v3"}.get(mine, "EV")
        return {
            "state": state, "perspective": user,
            "home_skaters": h + (1 if self.goalie_pulled["home"] else 0),
            "away_skaters": a + (1 if self.goalie_pulled["away"] else 0),
            "pp_side": pp_side, "pp_sec_left": pp_left,
            "empty_net": {"home": bool(self.goalie_pulled["home"]), "away": bool(self.goalie_pulled["away"])},
        }

    def _strength_for(self, side: Optional[str], t: float) -> str:
        mp = self.manpower(t)
        if side not in _SIDES:
            side = "home"
        h = mp["home_skaters"] - (1 if self.goalie_pulled["home"] else 0)
        a = mp["away_skaters"] - (1 if self.goalie_pulled["away"] else 0)
        mine, theirs = (h, a) if side == "home" else (a, h)
        if mine > theirs:
            return "PP"
        if mine < theirs:
            return "SH"
        return {5: "EV", 4: "4v4", 3: "3v3"}.get(mine, "EV")

    # ── engine
    def _run_pass(self, effects: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], _EnginePass]:
        from app.sim_engine.gameplay import game_analytics_ledger as gal

        sim = self.sim
        lg = self.lg
        cap = _EnginePass(self, effects)
        ledger: Dict[str, Dict[str, Any]] = {}
        rng = random.Random(int(lg.seed))  # same seed every pass => identical prefix
        names = ("_gm_on_ice_unit", "_team_offense_skill", "_team_defense_suppression", "_gm_chance_context",
                 "_gm_goalie_save_adjustment", "_gm_generate_penalties")
        self._restore_team_state()  # every pass starts from the same pre-game caches => identical replays
        with _ENGINE_PATCH_LOCK:
            orig_credit = gal.credit_shot_attempt_event
            saved = {n: sim.__dict__.get(n, _MISSING) for n in names}
            cap.origs = {n: getattr(sim, n) for n in names}
            cap.orig_credit = orig_credit
            prev_ledger = getattr(sim, "_gm_current_ledger", None)
            owned = None
            try:
                gal.credit_shot_attempt_event = cap.credit_hook  # type: ignore[assignment]
                for n in names:
                    sim.__dict__[n] = getattr(cap, "w" + n)
                owned = sim._gm_begin_game_scope()
                sim._gm_prime_game_player_caches(lg.home, lg.away, [], [])
                sim._gm_current_ledger = getattr(self.session, "player_season_stats", None)
                result = sim._run_event_driven_game(
                    rng, lg.home, lg.away, lg.home_id, lg.away_id, ledger,
                    strength_map=self._strength_map, home_strength_scale=self.home_scale,
                    away_strength_scale=self.away_scale, noise_scale=self.noise_scale, light_mode=False,
                    is_playoff=bool(lg.is_playoff), calendar_day=int(lg.calendar_day), home_b2b=False, away_b2b=False,
                    home_goalie=self.home_goalie, away_goalie=self.away_goalie,
                )
            finally:
                gal.credit_shot_attempt_event = orig_credit  # type: ignore[assignment]
                for n in names:
                    if saved[n] is _MISSING:
                        sim.__dict__.pop(n, None)
                    else:
                        sim.__dict__[n] = saved[n]
                try:
                    sim._gm_end_game_scope(owned)
                except Exception:
                    logger.warning("live: end game scope failed", exc_info=True)
                sim._gm_current_ledger = prev_ledger
        cap.finalize(result)
        return result, ledger, cap

    def _state_objects(self) -> List[Any]:
        objs: List[Any] = [self.lg.home, self.lg.away]
        for team in (self.lg.home, self.lg.away):
            for attr in ("roster", "players"):
                lst = getattr(team, attr, None)
                if isinstance(lst, (list, tuple)) and lst:
                    objs.extend(lst)
                    break
        return objs

    @staticmethod
    def _copy_val(v: Any) -> Any:
        if isinstance(v, dict):
            return dict(v)
        if isinstance(v, list):
            return list(v)
        if isinstance(v, set):
            return set(v)
        return v

    def _snapshot_team_state(self) -> None:
        """Pre-game private state of both teams and their players (engine caches, chemistry,
        deployment memos, development/streak counters). One engine pass mutates these in place,
        so every pass (first or coach re-sim) starts from this snapshot: replays are identical and
        the committed pass's side effects are applied exactly once."""
        self._team_base = {}
        for o in self._state_objects():
            try:
                d = vars(o)
            except TypeError:
                continue
            self._team_base[id(o)] = (o, {k: self._copy_val(v) for k, v in d.items() if k.startswith("_") or k == "chemistry_profile"})

    def _restore_team_state(self) -> None:
        base = getattr(self, "_team_base", None)
        if base is None:
            self._snapshot_team_state()
            return
        for o, b in base.values():
            d = vars(o)
            for k in [k for k in d if (k.startswith("_") or k == "chemistry_profile") and k not in b]:
                del d[k]
            for k, v in b.items():
                d[k] = self._copy_val(v)

    def _capture_engine_timeline(self, *, go_live: bool = True) -> None:
        if self._engine_captured:
            return
        t0 = time.perf_counter()
        result, ledger, cap = self._run_pass(self._effects)
        self._adopt(result, ledger, cap, splice=False)
        self._engine_captured = True
        self.last_pass_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.info("live engine pass %.0f ms (%d attempts)", self.last_pass_ms, len(cap.attempts))
        if go_live and self.phase == "pregame":
            self.phase = "live"
        self._refresh_status()

    def prepare(self) -> None:
        """Run the authoritative engine pass ahead of the first /advance (called from /start)."""
        self._capture_engine_timeline(go_live=False)

    def _resim(self, effects: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if not self._engine_captured:
            self._effects = [dict(e) for e in effects]
            return True, ""
        k = self._emitted_attempts
        if self._cap is None or k >= len(self._cap.attempts):
            return False, "no unplayed shot attempts left to re-simulate"
        try:
            t0 = time.perf_counter()
            result, ledger, cap = self._run_pass(effects)
            self.last_pass_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        except Exception as e:
            logger.exception("live re-sim failed")
            return False, f"engine re-simulation failed ({type(e).__name__})"
        old = [_attempt_sig(a) for a in self._cap.attempts[:k]]
        new = [_attempt_sig(a) for a in cap.attempts[:k]]
        if old != new:
            logger.warning("live re-sim diverged before t=%.1f (%d shown attempts); keeping original timeline", self.game_sec_elapsed, k)
            return False, "the replay did not reproduce the play already shown"
        if self._cap.pen_sig(self.game_sec_elapsed) != cap.pen_sig(self.game_sec_elapsed):
            return False, "the replay did not reproduce the penalties already shown"
        self._effects = [dict(e) for e in effects]
        self._adopt(result, ledger, cap, splice=True)
        return True, ""

    def _adopt(self, result: Dict[str, Any], ledger: Dict[str, Dict[str, Any]], cap: _EnginePass, *, splice: bool) -> None:
        self._result = result
        self._scratch = ledger
        self._cap = cap
        self._so_rounds = self._shootout_rounds(result, cap)
        result["_so_rounds"] = [dict(r) for r in self._so_rounds]
        for side in _SIDES:
            units = cap.ctx.get(f"{side}_units") or {}
            F = [[self.book.add(p, side) for p in (ln or [])] for ln in (units.get("lines") or [])][:4]
            D = [[self.book.add(p, side) for p in (pr or [])] for pr in (units.get("pairs") or [])][:3]
            if any(F):
                self.lines[side]["F"] = F
            if any(D):
                self.lines[side]["D"] = D
            st = cap.ctx.get(f"{side}_starter")
            gl = [g for g in (cap.ctx.get(f"{side}_gl") or []) if g is not None]
            G = []
            if st is not None:
                G.append(self.book.add(st, side))
            for g in gl:
                gid = self.book.add(g, side)
                if gid and gid not in G:
                    G.append(gid)
            if G:
                self.lines[side]["G"] = G[:2]
        tl = self._build_timeline(result, cap)
        if not splice:
            self._timeline = tl
            self._timeline_idx = 0
            return
        emitted = self._timeline[: self._timeline_idx]
        keys = {r["key"] for r in emitted}
        now = float(self.game_sec_elapsed)
        k = self._emitted_attempts
        rest: List[Dict[str, Any]] = []
        for r in tl:
            if r["key"] in keys:
                continue
            if r.get("aidx") is not None:
                if int(r["aidx"]) < k:
                    continue
            elif float(r["t"]) < now:
                continue
            if float(r["t"]) < now:
                r["t"] = now
            rest.append(r)
        rest.sort(key=lambda r: (float(r["t"]), float(r["phase"])))
        self._timeline = emitted + rest

    def _shootout_rounds(self, result: Dict[str, Any], cap: _EnginePass) -> List[Dict[str, Any]]:
        """Round-by-round presentation of the ENGINE's shootout decision (winner never changes)."""
        if not result.get("shootout"):
            return []
        home_win = bool(result.get("shootout_home_win"))
        rng = random.Random(int(self.lg.seed) ^ 0x5070)
        sim = self.sim

        def pool(side: str) -> List[Any]:
            team = self.lg.home if side == "home" else self.lg.away
            try:
                sk = list(sim._gm_skaters(team))
                return sorted(sk, key=lambda p: sim._gm_offensive_skill_composite(p), reverse=True)[:9]
            except Exception:
                return [p for p in (cap.ctx.get(f"{side}_dressed") or [])][:9]

        pools = {"home": pool("home"), "away": pool("away")}
        goalies = {"home": cap.ctx.get("home_starter"), "away": cap.ctx.get("away_starter")}

        def prob(shooter: Any, goalie: Any) -> float:
            try:
                from app.sim_engine.gameplay.game_analytics_ledger import resolve_goal_probability

                fin = float(sim._gm_finishing_adjustment(shooter))
                g_adj = float(sim._gm_goalie_save_adjustment(goalie, "SLOT")) if goalie is not None else 1.0
                return min(0.62, max(0.08, float(resolve_goal_probability(0.28, fin, 2.0 - g_adj)) * 0.95))
            except Exception:
                return 0.32

        def mk(side: str, rnd: int, goal: bool) -> Dict[str, Any]:
            pl = pools[side]
            shooter = pl[(rnd - 1) % len(pl)] if pl else None
            g = goalies[_opp(side)]
            return {"round": rnd, "side": side, "team_id": self.lg.home_id if side == "home" else self.lg.away_id,
                    "shooter_id": self.book.add(shooter, side) or None, "goalie_id": self.book.add(g, _opp(side)) or None,
                    "goal": bool(goal)}

        def simulate() -> Tuple[List[Dict[str, Any]], Optional[bool]]:
            rounds: List[Dict[str, Any]] = []
            score = {"home": 0, "away": 0}
            for rnd in range(1, 16):
                for idx, side in enumerate(("home", "away")):
                    pl = pools[side]
                    shooter = pl[(rnd - 1) % len(pl)] if pl else None
                    goal = rng.random() < prob(shooter, goalies[_opp(side)])
                    rounds.append(mk(side, rnd, goal))
                    score[side] += int(goal)
                    if rnd <= 3:
                        left_h = 3 - rnd
                        left_a = 3 - rnd + (1 if idx == 0 else 0)
                        if score["home"] > score["away"] + left_a or score["away"] > score["home"] + left_h:
                            return rounds, score["home"] > score["away"]
                if rnd >= 3 and score["home"] != score["away"]:
                    return rounds, score["home"] > score["away"]
            return rounds, None

        for _ in range(40):
            rounds, winner = simulate()
            if winner is not None and winner == home_win:
                return rounds
        forced: List[Dict[str, Any]] = []
        for rnd in range(1, 4):
            forced += [mk("home", rnd, False), mk("away", rnd, False)]
        forced += [mk("home", 4, home_win), mk("away", 4, not home_win)]
        return forced

    @staticmethod
    def _reg_t(local_t: float) -> float:
        """Engine clock, except the first 12 s after each period start are compressed into
        6-12 s so no shot is shown before the opening faceoff (monotonic)."""
        t = float(local_t)
        ps = math.floor(t / _REGULATION_PERIOD_SEC) * _REGULATION_PERIOD_SEC
        x = t - ps
        return ps + 6.0 + 0.5 * x if x < 12.0 else t

    def _attempt_t(self, a: Dict[str, Any]) -> float:
        if a.get("seg") == "reg":
            return self._reg_t(float(a.get("local_t") or 0.0))
        return self._ot_time(str(a.get("seg") or "ot1"), float(a.get("local_t") or 0.0), int(a.get("total") or 0))

    def _ot_time(self, seg: str, local_t: float, total: int) -> float:
        try:
            k = max(1, int(seg[2:] or 1))
        except ValueError:
            k = 1
        seg_len = self._ot_seg_len()
        base = _REGULATION_GAME_SEC + (k - 1) * seg_len
        scale = min(1.0, (seg_len - 8.0) / max(1.0, total * _OT_TICK_SEC))
        return min(base + seg_len - 1.0, base + 6.0 + local_t * scale)

    def _build_timeline(self, result: Dict[str, Any], cap: _EnginePass) -> List[Dict[str, Any]]:
        tl: List[Dict[str, Any]] = []

        def add(key: str, t: float, phase: float, typ: str, **kw: Any) -> None:
            tl.append({"key": key, "t": float(t), "phase": float(phase), "type": typ, **kw})

        overtime = bool(result.get("overtime"))
        for p in (1, 2, 3):
            add(f"ps{p}", (p - 1) * _REGULATION_PERIOD_SEC, 0, "period_start", period=p)
            add(f"fo{p}", (p - 1) * _REGULATION_PERIOD_SEC, 1, "faceoff", period=p, meta={"reason": "period_start", "winner": None, "location": "center"})
            add(f"pe{p}", p * _REGULATION_PERIOD_SEC, -1, "period_end", period=p, intermission_after=(p < 3 or overtime))
        score = {"home": 0, "away": 0}
        for a in cap.attempts:
            t = self._attempt_t(a)
            goal = a.get("outcome") == "GOAL"
            if goal:
                score[a["side"]] += 1
            sl = None
            if a.get("side") in _SIDES:
                sl = self.tracking.shot_loc_for(f"a{a['idx']}|{a.get('shooter_id')}|{round(float(a.get('local_t') or 0.0), 2)}",
                                                a.get("chance"), self.tracking.attack_dir(a["side"], self.clock_at(t)[0]))
            add(f"a{a['idx']}", t, 2, "goal" if goal else "attempt", aidx=a["idx"], att=a, score_after=dict(score), shot_loc=sl)
            if goal and a.get("seg") == "reg":
                add(f"fg{a['idx']}", t, 6, "faceoff", aidx=a["idx"], meta={"reason": "after_goal", "winner": None, "location": "center"})
        t_end = _REGULATION_GAME_SEC
        if overtime:
            add("ps4", _REGULATION_GAME_SEC, 0, "period_start", period=4)
            add("fo4", _REGULATION_GAME_SEC, 1, "faceoff", period=4, meta={"reason": "period_start", "winner": None, "location": "center"})
            ot_goal = next((a for a in cap.attempts if a.get("seg") != "reg" and a.get("outcome") == "GOAL"), None)
            if ot_goal is not None:
                t_end = self._attempt_t(ot_goal)
            else:
                t_end = _REGULATION_GAME_SEC + self._ot_seg_len() * max(1, cap._seg_k)
            add("pe4", t_end, 7, "period_end", period=4, intermission_after=False)
        for pen in cap.penalties:
            add(pen["key"], pen["start"], 4, "penalty", pen=pen)
            end = pen["start"] + pen["dur"]
            if end < _REGULATION_GAME_SEC:
                add("e" + pen["key"], end, 3, "penalty_end", pen=pen)
        for c in cap.coach_raw:
            if c.get("silent"):
                continue
            if c.get("reason") == "period_end":
                t = _REGULATION_GAME_SEC
            elif c.get("seg") == "reg":
                t = self._reg_t(float(c.get("local_t") or 0.0))
            else:
                t = self._ot_time(str(c.get("seg")), float(c.get("local_t") or 0.0), int(c.get("total") or 0))
            add(f"c{c['action']}{c['side']}{c['aidx']}", t, float(c.get("phase") or 5), "coach", aidx=c.get("aidx"), coach=c)
        if self._so_rounds:
            add("ps5", t_end, 8, "period_start", period=5)
            for j, r in enumerate(self._so_rounds):
                add(f"so{j}", t_end + 1.0 + j, 9, "shootout_attempt", period=5, so=r)
            t_end = t_end + 1.0 + len(self._so_rounds)
        add("final", t_end, 10, "final", score_after={
            "home": int(result.get("display_home_goals") or 0), "away": int(result.get("display_away_goals") or 0)})
        tl.sort(key=lambda r: (float(r["t"]), float(r["phase"])))
        return tl

    # ── stepping
    def _refresh_status(self) -> None:
        lg = self.lg
        if lg.status in ("committed", "aborted"):
            return
        if self.finished:
            s = "final"
        elif lg.paused:
            s = "paused"
        elif self.intermission:
            s = "intermission"
        elif self.phase == "shootout":
            s = "shootout"
        elif self.phase == "pregame":
            s = "pregame"
        elif self.game_sec_elapsed >= _REGULATION_GAME_SEC and (self._result or {}).get("overtime"):
            s = "overtime"
        else:
            s = "live"
        lg.status = s
        lg.game_sec_elapsed = float(self.game_sec_elapsed)
        lg.period = int(self.clock_state()["period"])

    def resume_from_intermission(self) -> None:
        if self.intermission:
            self.intermission = False
            self.shifts.intermission(self.game_sec_elapsed)
            self._refresh_status()

    def step_once(self) -> bool:
        return self.step_until(self.game_sec_elapsed + self.rng.uniform(6.0, 14.0), cap_sec=14.0) > 0

    def step_until(self, until_t: float, *, cap_sec: float = 60.0) -> int:
        if not self._engine_captured:
            self._capture_engine_timeline()
        if self.finished or self.intermission:
            self._refresh_status()
            return 0
        if self.phase == "pregame":
            self.phase = "live"
        target = min(float(until_t), self.game_sec_elapsed + float(cap_sec))
        live = self._motion_on and bool(getattr(self.motion, "live_mode", False))
        if not live:
            n = self._emit_due_events()
            self._motion_step()
            while not self.finished and not self.intermission and self.game_sec_elapsed < target:
                self.game_sec_elapsed = min(target, self.game_sec_elapsed + 4.0)
                n += self._emit_due_events()
                self._motion_step()
                self._tick(self.game_sec_elapsed)
            self._refresh_status()
            return n
        # forward play: the motion runs first (its warped game clock decides when each real event is reached),
        # then the events up to the clock the motion has actually played are emitted
        n = self._emit_due_events()
        guard = 0
        while not self.finished and not self.intermission and self.game_sec_elapsed < target - 1e-6 and guard < 80:
            guard += 1
            goal = min(target, self.game_sec_elapsed + 4.0)
            cap = self._pending_intermission_t()
            if cap is not None and cap <= goal:
                goal = cap
            before = self.motion.realized_gc()
            self._motion_step(goal)
            reached = self.motion.realized_gc()
            self.game_sec_elapsed = max(self.game_sec_elapsed, min(goal, reached))
            n += self._emit_due_events()
            self._tick(self.game_sec_elapsed)
            if reached <= before + 1e-9 and self.game_sec_elapsed < goal - 1e-6:
                break   # motion can't advance (e.g. waiting at an intermission)
        self._refresh_status()
        return n

    def _pending_intermission_t(self) -> Optional[float]:
        if self._skip_intermission:
            return None
        for raw in self._timeline[self._timeline_idx:]:
            if raw.get("type") == "period_end" and raw.get("intermission_after"):
                return float(raw["t"])
        return None

    def _motion_step(self, gc_end: Optional[float] = None) -> None:
        if self._motion_on:
            try:
                self._motion_buf.extend(self.motion.generate_to(self.game_sec_elapsed if gc_end is None else gc_end)["frames"])
            except Exception:
                logger.warning("motion generation failed", exc_info=True)
            self._drain_motion()

    def _drain_motion(self) -> None:
        """Derived rule events (offside, icing, faceoffs, line changes, ...) and play hints from the motion track."""
        mt = self.motion
        if not hasattr(mt, "drain_events"):
            return
        try:
            for d in mt.drain_events():
                ev = self._derived_event(d)
                if ev is not None:
                    self._push_event(ev)
            self.commentary.on_hints(mt.drain_hints())
            self.lg.captions = self.commentary.caption_tail()
        except Exception:
            logger.warning("derived motion events failed", exc_info=True)

    def _derived_event(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lg = self.lg
        typ = str(d.get("type"))
        t = float(d.get("t", self.game_sec_elapsed))
        meta = dict(d.get("meta") or {})
        if typ == "rule":
            typ = "stoppage" if meta.get("rule") not in ("icing_waved", "sh_clear") else "rule"
            meta.setdefault("reason", meta.get("rule"))
        lg.seq += 1
        period, left = self.clock_at(t)
        side = meta.get("side") if meta.get("side") in _SIDES else (meta.get("winner") if meta.get("winner") in _SIDES else None)
        ev: Dict[str, Any] = {
            "seq": lg.seq, "t": round(t, 2), "pt": round(float(d.get("pt", t)), 2), "period": period, "clock": _clock_display(left),
            "type": typ, "team_id": (lg.home_id if side == "home" else lg.away_id) if side else None,
            "team_abbr": self.abbr(side) if side else None, "side": side, "strength": self._strength_for(None, t),
            "shooter_id": None, "shooter_name": None, "shooter_number": None, "goalie_id": None, "goalie_name": None,
            "blocker_id": None, "blocker_name": None, "assist_ids": [], "assist_names": [], "outcome": None, "xg": 0.0,
            "chance": None, "rebound": False, "empty_net": False, "shot_loc": None,
            "on_ice": {"home": list(self.state.on_ice["home"]), "away": list(self.state.on_ice["away"])},
            "score_after": {"home": lg.score_home, "away": lg.score_away}, "importance": 1, "description": "",
            "meta": meta, "authority": "derived",
        }
        book = self.book
        for k in ("player_id", "winner_id", "loser_id", "hitter_id", "victim_id", "center_id"):
            if meta.get(k):
                meta[k.replace("_id", "_name")] = book.name(meta[k]) or None
        if typ == "faceoff":
            ft = getattr(self.motion, "fo_tally", None)
            if ft:
                meta["tally"] = {s: {"won": ft[s][0], "taken": ft[s][1]} for s in _SIDES}
        ev["description"] = self._describe_derived(ev)
        try:
            self.commentary.on_event(ev)
        except Exception:
            logger.warning("commentary failed on derived %s", typ, exc_info=True)
        return ev

    def _describe_derived(self, ev: Dict[str, Any]) -> str:
        m = ev.get("meta") or {}
        team = ev.get("team_abbr") or ""
        typ = ev["type"]
        lang = self.lg.lang
        W = lambda key, **kw: _fmt(_loc(_WHISTLE, lang, key, key), kw)  # noqa: E731
        if typ == "faceoff":
            return f"{_loc(_DESC, lang, 'faceoff')} {m.get('winner_name') or ''} ({team})".strip()
        if typ == "stoppage":
            r = m.get("reason")
            if r == "freeze":
                return W("freeze", goalie=self.book.short(m.get("player_id")) or "")
            return W(r if r in ("offside", "icing", "glass", "out") else "out", team=team)
        if typ == "line_change":
            return f"{team}: {m.get('center_name') or ''}".strip(": ")
        if typ == "delayed_penalty":
            return W("delayed", team=self.abbr(m.get("side")) if m.get("side") in _SIDES else team)
        if typ == "hit":
            return f"{m.get('hitter_name') or ''} → {m.get('victim_name') or ''}"
        if typ == "rule":
            return W(m.get("rule") or "out", team=team)
        return ""

    def take_motion(self, *, skipped: bool = False) -> Dict[str, Any]:
        """Frames produced since the last call (contiguous 0.5 s grid)."""
        if skipped:
            self._motion_buf = []
            return self.motion.generate_to(self.game_sec_elapsed, skip=True)
        self._motion_step()
        frames, self._motion_buf = self._motion_buf, []
        mt = self.motion
        last = mt.t_last if mt.t_last is not None else 0.0
        return {"from_t": frames[0]["t"] if frames else last, "to_t": frames[-1]["t"] if frames else last,
                "from_pt": frames[0]["t"] if frames else last, "to_pt": frames[-1]["t"] if frames else last,
                "gc": round(float(getattr(mt, "gc_last", self.game_sec_elapsed)), 2), "step": 0.5,
                "frames": frames, "authority": "model_estimated", "skipped": False, "clock": _CLOCK_NOTE}

    def pt_now(self) -> float:
        mt = self.motion
        return float(mt.t_last) if getattr(mt, "t_last", None) is not None else 0.0

    def gc_for_pt(self, pt: float) -> float:
        """Game clock reached when presentation time `pt` is played (for /advance until_pt)."""
        mt = self.motion
        if hasattr(mt, "gc_for_pt"):
            try:
                return float(mt.gc_for_pt(pt))
            except Exception:
                logger.warning("gc_for_pt failed", exc_info=True)
        return float(pt)

    def pt_of_raw(self, raw: Dict[str, Any]) -> float:
        mt = self.motion
        if hasattr(mt, "pt_for_raw"):
            return mt.pt_for_raw(raw)
        return round(float(raw.get("t", self.game_sec_elapsed)), 2)

    def run_to_final(self) -> None:
        """Sim to the final horn (skips client pacing and intermissions)."""
        self._skip_intermission = True
        self._motion_on = False
        try:
            if not self._engine_captured:
                self._capture_engine_timeline()
            self.resume_from_intermission()
            guard = 0
            while not self.finished and guard < 400:
                self.step_until(self.game_sec_elapsed + 600.0, cap_sec=600.0)
                self.resume_from_intermission()
                guard += 1
            if not self.finished:
                logger.error("live run_to_final did not reach final (idx=%d/%d)", self._timeline_idx, len(self._timeline))
        finally:
            self._skip_intermission = False
            self._motion_on = True

    def headshot_fields(self, pid: Optional[str]) -> Dict[str, Any]:
        """Fields the frontend PlayerHeadshot / resolvePlayerHeadshot helper reads (generated portrait + NHL photo)."""
        c = self.book.chips.get(str(pid or "")) or {}
        return dict(c.get("hs") or {})

    def _tick(self, t: float) -> None:
        if self.analytics is not None:
            try:
                self.analytics.tick(t, max(0.0, t - self._an_t))
            except Exception:
                logger.debug("analytics tick failed", exc_info=True)
            self._an_t = t
        try:
            self.shifts.advance(t)
            self.momentum.recompute(t)
            self.crowd.recompute(t)
            self.alerts.tick(t)
            self.commentary.tick(t)
        except Exception:
            logger.warning("live tick observers failed", exc_info=True)

    def _emit_due_events(self) -> int:
        n = 0
        while self._timeline_idx < len(self._timeline):
            raw = self._timeline[self._timeline_idx]
            if float(raw["t"]) > self.game_sec_elapsed + 1e-9:
                break
            self._timeline_idx += 1
            if raw["type"] in ("attempt", "goal") and raw.get("aidx") is not None:
                self._emitted_attempts += 1
            if raw["type"] == "faceoff" and getattr(self, "_motion_v2", False):
                continue  # faceoffs come from the motion track (staged at the right dot, winner derived)
            ev = self._to_live_event(raw)
            self._push_event(ev)
            n += 1
            self._cpu_reactions(ev)
            if ev["type"] == "final":
                self.finished = True
                self.phase = "final"
                self.game_sec_elapsed = float(raw["t"])
                break
            if ev["type"] == "period_end" and raw.get("intermission_after") and not self._skip_intermission:
                self.intermission = True
                self._int_period = int(ev["period"])
                self.game_sec_elapsed = float(raw["t"])
                break
        self._refresh_status()
        return n

    def _cpu_reactions(self, ev: Dict[str, Any]) -> None:
        """CPU bench: a timeout after two quick goals against. Cosmetic — engine timeouts change nothing."""
        if ev["type"] != "goal" or self.cpu_timeout_used or int(ev["period"]) > 3:
            return
        cpu = _opp(self.lg.user_side)
        if ev.get("side") == cpu:
            return
        t = float(ev["t"])
        lst = [x for x in self._goals_for_t[self.lg.user_side] if t - x <= 150.0] + [t]
        self._goals_for_t[self.lg.user_side] = lst
        if len(lst) >= 2:
            self.cpu_timeout_used = True
            self.push_coach_event(cpu, "timeout", "cpu", what=_loc(_COACH_WHAT, self.lg.lang, "timeout"),
                                  effect="CPU timeout (presentation only; play unchanged)")

    def push_coach_event(self, side: str, action: str, by: str, *, what: str = "", effect: str = "",
                         order_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = {"key": f"uc{self.lg.seq + 1}", "t": float(self.game_sec_elapsed), "phase": 5.0, "type": "coach",
               "coach": {"action": action, "side": side, "by": by, "what": what, "effect": effect,
                         "order_id": order_id, "payload": dict(payload or {})}}
        ev = self._to_live_event(raw)
        self._push_event(ev)
        return ev

    def apply_order_presentation(self, pres: Dict[str, Any], order: Dict[str, Any]) -> None:
        user = self.lg.user_side
        t = float(self.game_sec_elapsed)
        kind = pres.get("kind")
        if kind == "timeout":
            self.timeout_used = True
            self.shifts.rest(user, t)
            self.momentum.calm(0.5)
        elif kind == "pull_goalie":
            self.goalie_pulled[user] = bool(pres.get("pull"))
        elif kind == "tactics":
            self.tactics = dict(pres.get("tactics") or self.tactics)
        elif kind in ("shorten_bench", "roll_lines"):
            self.deploy_mode = str(pres.get("mode") or "normal")
        elif kind == "bench_player":
            pid = str(pres.get("player_id"))
            if "double" in pres:
                (self.double_shift.add if pres["double"] else self.double_shift.discard)(pid)
            elif pres.get("bench"):
                self.benched.add(pid)
                self.double_shift.discard(pid)
            else:
                self.benched.discard(pid)
        self.push_coach_event(user, str(pres.get("action") or kind), "user", what=str(pres.get("what") or ""),
                              effect=str(order.get("effect") or ""), order_id=order.get("id"), payload=order.get("payload"))

    # ── event building
    def _to_live_event(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        lg = self.lg
        lg.seq += 1
        typ = str(raw.get("type"))
        t = float(raw.get("t", self.game_sec_elapsed))
        period, left = self.clock_at(t, raw.get("period"))
        if typ == "period_end":
            left = 0.0
        ev: Dict[str, Any] = {
            "seq": lg.seq, "t": round(t, 2), "period": period, "clock": _clock_display(left), "type": typ,
            "team_id": None, "team_abbr": None, "side": None, "strength": self._strength_for(None, t),
            "shooter_id": None, "shooter_name": None, "shooter_number": None,
            "goalie_id": None, "goalie_name": None, "blocker_id": None, "blocker_name": None,
            "assist_ids": [], "assist_names": [], "outcome": None, "xg": 0.0, "chance": None,
            "rebound": False, "empty_net": False, "shot_loc": None,
            "on_ice": {"home": list(self.state.on_ice["home"]), "away": list(self.state.on_ice["away"])},
            "score_after": {"home": lg.score_home, "away": lg.score_away}, "importance": 1, "description": "",
            "meta": dict(raw.get("meta") or {}), "authority": "derived",
        }
        book = self.book

        def set_side(side: Optional[str]) -> None:
            if side in _SIDES:
                ev["side"] = side
                ev["team_id"] = lg.home_id if side == "home" else lg.away_id
                ev["team_abbr"] = self.abbr(side)

        if typ in ("attempt", "goal"):
            a = raw.get("att") or {}
            side = a.get("side")
            set_side(side)
            ev.update({
                "shooter_id": a.get("shooter_id"), "shooter_name": book.name(a.get("shooter_id")) or None,
                "shooter_number": book.number(a.get("shooter_id")),
                "goalie_id": a.get("goalie_id"), "goalie_name": book.name(a.get("goalie_id")) or None,
                "blocker_id": a.get("blocker_id"), "blocker_name": book.name(a.get("blocker_id")) or None,
                "outcome": a.get("outcome") or None, "xg": round(float(a.get("xg") or 0.0), 4),
                "chance": a.get("chance"), "rebound": bool(a.get("rebound")), "empty_net": bool(a.get("empty_net")),
                "authority": "authoritative",
            })
            if a.get("broken"):
                ev["meta"]["capture_incomplete"] = True
            if side in _SIDES and (a.get("atk_ids") or a.get("def_ids")):
                ev["on_ice"] = {side: list(a.get("atk_ids") or []), _opp(side): list(a.get("def_ids") or [])}
            ev["strength"] = self._attempt_strength(a)
            if raw.get("shot_loc"):
                ev["shot_loc"] = dict(raw["shot_loc"])
            if typ == "goal":
                ev["assist_ids"] = list(a.get("assist_ids") or [])
                ev["assist_names"] = [book.name(x) for x in ev["assist_ids"]]
                sc = raw.get("score_after") or {}
                lg.score_home = int(sc.get("home", lg.score_home))
                lg.score_away = int(sc.get("away", lg.score_away))
                ev["score_after"] = {"home": lg.score_home, "away": lg.score_away}
                ev["importance"] = 3
                try:
                    sid_ = a.get("shooter_id")
                    row = (getattr(self.session, "player_season_stats", None) or {}).get(sid_) or {}
                    ev["meta"]["season_goals"] = int(row.get("g", 0) or 0) + int((self.state.p.get(sid_) or {}).get("g", 0) or 0) + 1
                    ev["meta"]["scorer_headshot"] = {"player_id": sid_, "name": book.name(sid_), "position": book.pos(sid_), **self.headshot_fields(sid_)}
                except Exception:
                    ev["meta"]["season_goals"] = None
            else:
                hd = ev["chance"] in _HD_CHANCES
                ev["importance"] = 3 if (ev["outcome"] == "SAVED" and hd and ev["xg"] >= 0.12) else (2 if hd else 1)
            lines = a.get("lines") or {}
            ev["meta"].update({
                "chance_raw": a.get("chance_raw"), "tags": list(a.get("tags") or []), "screened": bool(a.get("screened")),
                "strength_engine": a.get("strength_engine"), "skaters": {"home": a.get("sk_home"), "away": a.get("sk_away")},
                "extra_attacker": bool(a.get("extra_attacker")), "engine_segment": a.get("seg"),
                "engine_t": round(float(a.get("local_t") or 0.0), 2),
                "lines": {s: {"F": v[0], "D": v[1], "unit": v[2]} for s, v in lines.items() if v},
            })
            if ev["outcome"] in ("SAVED", "GOAL"):
                lg.shots_home += int(side == "home")
                lg.shots_away += int(side == "away")
            if side == "home":
                lg.xg_home = round(lg.xg_home + ev["xg"], 4)
            elif side == "away":
                lg.xg_away = round(lg.xg_away + ev["xg"], 4)
        elif typ in ("penalty", "penalty_end"):
            pen = raw.get("pen") or {}
            set_side(pen.get("side"))
            mins = int(pen.get("mins") or 2)
            ev["authority"] = "authoritative"
            ev["strength"] = self._strength_for(pen.get("side"), t + (0.01 if typ == "penalty" else 0.0))
            ev["meta"].update({
                "player_id": pen.get("player_id"), "player_name": book.name(pen.get("player_id")) or None,
                "player_short": book.short(pen.get("player_id")) or None, "minutes": mins,
                "label": _loc(_PEN_LABEL, lg.lang, mins, f"{mins} min"), "duration_sec": pen.get("full"), "infraction": None,
            })
            if typ == "penalty":
                ev["importance"] = 2
            else:
                ev["meta"]["reason"] = "pp_goal" if pen.get("ended_by_goal") else "expired"
                mp = self.manpower(t)
                ev["meta"]["manpower_restored"] = mp["home_skaters"] == mp["away_skaters"]
        elif typ in ("period_start", "period_end"):
            ev["importance"] = 2
            if typ == "period_start" and period == 5:
                self.phase = "shootout"
            if typ == "period_start" and period == 4:
                ev["strength"] = "EV" if lg.is_playoff else "3v3"
        elif typ == "faceoff":
            ev["meta"].setdefault("winner", None)
        elif typ == "coach":
            c = raw.get("coach") or {}
            side = c.get("side")
            set_side(side)
            act = str(c.get("action") or "")
            if act == "pull" and side in _SIDES:
                self.goalie_pulled[side] = True
                self._pull_hist.append((t, side, True))
            elif act == "return" and side in _SIDES:
                self.goalie_pulled[side] = False
                self._pull_hist.append((t, side, False))
            ev["importance"] = 2
            ev["authority"] = "authoritative" if c.get("by") in ("cpu", "auto") else "derived"
            what = c.get("what") or _loc(_COACH_WHAT, lg.lang, act if act in ("pull", "return", "timeout") else "timeout")
            ev["meta"].update({"action": act, "by": c.get("by"), "what": what, "effect": c.get("effect") or "",
                               "order_id": c.get("order_id"), "reason": c.get("reason"), "payload": c.get("payload") or {}})
            if act in ("pull", "return") and c.get("by") == "cpu":
                ev["meta"]["effect"] = "CPU bench decision inside the engine re-simulation"
        elif typ == "shootout_attempt":
            r = raw.get("so") or {}
            side = r.get("side")
            set_side(side)
            if side in _SIDES and r.get("goal"):
                self._so_tally[side] += 1
            ev.update({
                "shooter_id": r.get("shooter_id"), "shooter_name": book.name(r.get("shooter_id")) or None,
                "shooter_number": book.number(r.get("shooter_id")),
                "goalie_id": r.get("goalie_id"), "goalie_name": book.name(r.get("goalie_id")) or None,
                "outcome": "GOAL" if r.get("goal") else "SAVED", "chance": "breakaway", "strength": "EV",
                "importance": 3 if r.get("goal") else 2, "authority": "derived",
            })
            ev["on_ice"] = {side: [r.get("shooter_id")] if r.get("shooter_id") else [], _opp(side): []} if side in _SIDES else ev["on_ice"]
            ev["meta"].update({"round": r.get("round"), "so_home": self._so_tally["home"], "so_away": self._so_tally["away"],
                               "note": "Round-by-round shootout presentation of the engine's shootout result"})
        elif typ == "final":
            sc = raw.get("score_after") or {}
            lg.score_home = int(sc.get("home", lg.score_home))
            lg.score_away = int(sc.get("away", lg.score_away))
            ev["score_after"] = {"home": lg.score_home, "away": lg.score_away}
            ev["importance"] = 3
            ev["authority"] = "authoritative"
            if self._so_rounds:
                ev["period"], ev["clock"] = 5, _clock_display(0)
            ev["meta"].update(self.final_info() or {})
        ev["description"] = self._describe(ev, raw)
        try:
            ev["pt"] = self.pt_of_raw(raw) if typ != "coach" else round(max(self.pt_now(), self.pt_of_raw(raw)), 2)
        except Exception:
            logger.warning("pt for %s failed", typ, exc_info=True)
            ev["pt"] = round(t, 2)
        if raw.get("key"):
            ev["meta"]["event_key"] = raw.get("key")
        if typ in ("attempt", "goal") and hasattr(self.motion, "shot_info"):
            try:
                si = self.motion.shot_info(raw.get("key"))
                if si.get("shot_type"):
                    ev["meta"]["shot_type"] = si["shot_type"]          # presentation (derived from the chance type)
                    ev["meta"]["deflected"] = bool(si.get("deflected"))
            except Exception:
                logger.debug("shot_info failed", exc_info=True)
        try:
            self.state.apply(ev, raw)
            if typ in ("attempt", "goal"):
                self.shifts.observe(t, ev["on_ice"])
            self.momentum.on_event(ev)
            self.crowd.on_event(ev)
            self.alerts.on_event(ev)
            self.commentary.on_event(ev)
        except Exception:
            logger.warning("live observers failed on %s", typ, exc_info=True)
        return ev

    def _attempt_strength(self, a: Dict[str, Any]) -> str:
        if a.get("empty_net"):
            return "EN"
        side = a.get("side")
        h, w = int(a.get("sk_home") or 5), int(a.get("sk_away") or 5)
        mine, theirs = (h, w) if side == "home" else (w, h)
        if mine > theirs:
            return "PP"
        if mine < theirs:
            return "SH"
        return {5: "EV", 4: "4v4", 3: "3v3"}.get(mine, "EV")

    def _describe(self, ev: Dict[str, Any], raw: Dict[str, Any]) -> str:
        lang = self.lg.lang
        D = lambda key: _loc(_DESC, lang, key)  # noqa: E731
        book = self.book
        typ = ev["type"]
        team = ev.get("team_abbr") or ""
        score_line = f"{self.abbr('away')} {ev['score_after']['away']}, {self.abbr('home')} {ev['score_after']['home']}"
        if typ in ("attempt", "goal"):
            chance = _loc(_CHANCE_PHRASE, lang, ev.get("chance") or "", "")
            sh = book.short(ev.get("shooter_id")) or "?"
            if typ == "goal":
                names = ", ".join(book.short(x) for x in ev.get("assist_ids") or [] if book.short(x))
                assists = _fmt(D("assists"), {"names": names}) if names else D("unassisted")
                sfx = _loc(_STRENGTH_SFX, lang, ev.get("strength"), "")
                return _fmt(D("goal"), {"team": team, "shooter": sh, "assists": assists, "chance": chance, "sfx": sfx, "score_line": score_line})
            oc = ev.get("outcome")
            if oc == "SAVED" and ev.get("empty_net"):
                return _fmt(D("en_kept_out"), {"shooter": sh, "chance": chance})
            if oc == "SAVED":
                return _fmt(D("saved"), {"shooter": sh, "chance": chance, "goalie": book.short(ev.get("goalie_id")) or "?"})
            if oc == "BLOCKED":
                return _fmt(D("blocked"), {"shooter": sh, "chance": chance, "blocker": book.short(ev.get("blocker_id")) or "?"})
            return _fmt(D("missed"), {"shooter": sh, "chance": chance})
        meta = ev.get("meta") or {}
        if typ == "penalty":
            return _fmt(D("penalty"), {"team": team, "player": meta.get("player_short") or "?", "label": meta.get("label") or ""})
        if typ == "penalty_end":
            return _fmt(D("penalty_end_goal" if meta.get("reason") == "pp_goal" else "penalty_end"), {"team": team})
        if typ in ("period_start", "period_end"):
            nm = _loc(_PERIOD_NAME, lang, int(ev["period"]))
            return _fmt(D(typ), {"period": nm, "Period": _cap(nm)})
        if typ == "faceoff":
            return D("faceoff_goal" if meta.get("reason") == "after_goal" else "faceoff")
        if typ == "coach":
            return _fmt(D("coach"), {"team": team, "what": meta.get("what") or ""})
        if typ == "shootout_attempt":
            key = "so_goal" if ev.get("outcome") == "GOAL" else "so_miss"
            return _fmt(D(key), {"round": meta.get("round"), "shooter": book.short(ev.get("shooter_id")) or "?",
                                 "goalie": book.short(ev.get("goalie_id")) or "?"})
        if typ == "final":
            dec = (meta.get("decision") or "REG")
            return _fmt(D("final"), {"score_line": score_line, "dec": D(f"dec_{dec}") if dec in ("OT", "SO") else ""})
        return ""

    def _push_event(self, ev: Dict[str, Any]) -> None:
        lg = self.lg
        if self.analytics is not None:
            self.analytics.on_event(ev)
        lg.event_ring.append(ev)
        if len(lg.event_ring) > LIVE_EVENT_BUFFER:
            lg.event_ring = lg.event_ring[-LIVE_EVENT_BUFFER:]
        lg.captions = self.commentary.caption_tail()
        lg.alerts_buf = self.alerts.items[-40:]
        lg.tracking_frames = self.tracking.frames

    def final_info(self) -> Optional[Dict[str, Any]]:
        if not self.finished and self.lg.status != "final":
            if not (self._timeline_idx and self._timeline and self._timeline[self._timeline_idx - 1].get("type") == "final"):
                return None
        r = self._result or {}
        hg, ag = int(r.get("display_home_goals") or 0), int(r.get("display_away_goals") or 0)
        return {"winner_side": "home" if hg > ag else "away",
                "decision": "SO" if r.get("shootout") else ("OT" if r.get("overtime") else "REG")}

    def build_result(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if not self._engine_captured:
            self._capture_engine_timeline()
        assert self._result is not None
        return dict(self._result), dict(self._scratch)


def commit_live_game(session: Any, lg: LiveGame) -> Dict[str, Any]:
    """Commit exactly the engine pass the user watched (latest pass incl. coach-order re-sims)."""
    from services.franchise_store import save_session
    from services import franchise_sim as fs

    loop = lg.loop
    if loop is None:
        raise HTTPException(status_code=500, detail={"reason": "no_loop", "message": "Live loop missing."})
    if lg.status == "committed" or lg.scratch.get("committed"):
        raise HTTPException(status_code=409, detail={"reason": "already_committed", "message": "This game was already committed."})
    if not loop.finished:
        raise HTTPException(status_code=409, detail={"reason": "not_final", "message": "Finish the game before committing."})

    result, scratch = loop.build_result()
    sim = lg.sim
    hid, aid = lg.home_id, lg.away_id
    hg = int(result.get("display_home_goals") or 0)
    ag = int(result.get("display_away_goals") or 0)
    if (hg, ag) != (int(lg.score_home), int(lg.score_away)):
        logger.error("live commit: shown score %s-%s != engine %s-%s", lg.score_home, lg.score_away, hg, ag)
        raise HTTPException(status_code=500, detail={"reason": "score_mismatch", "message": "Shown score and engine result disagree; not committing."})
    ot = bool(result.get("overtime"))
    key = (str(hid), str(aid), int(hg), int(ag), bool(ot), False)
    entry = {"_key": key, "result": result, "scratch": scratch}
    setattr(sim, "_pending_event_game", entry)
    pend = getattr(sim, "_pending_event_games", None)
    if not isinstance(pend, dict):
        pend = {}
    if len(pend) > 64:
        pend.clear()
    pend[key] = entry
    setattr(sim, "_pending_event_games", pend)

    score_res = sim._score_result_from_event(result, lg.home, lg.away, hid, aid, is_playoff=bool(lg.is_playoff))
    session._live_score_override = {
        "key": (str(hid), str(aid), int(lg.calendar_day)),
        "score_result": score_res,
    }

    try:
        step = fs.advance_franchise_day(session)
        save_session(session)
    except Exception as e:
        session._live_score_override = None
        pend.pop(key, None)
        logger.exception("live commit failed")
        raise HTTPException(status_code=500, detail={"reason": "commit_failed", "message": str(e)}) from e

    lg.scratch["committed"] = True
    lg.status = "committed"
    with _registry_lock:
        if _LIVE.get(str(lg.session_id)) is lg:
            _LIVE.pop(str(lg.session_id), None)
    return {
        "ok": True,
        "day_result": step,
        "score": {"home": hg, "away": ag},
    }


def _try_lock(lg: LiveGame, timeout: float = 0.5) -> bool:
    return lg.lock.acquire(timeout=timeout)


def _events_since(lg: LiveGame, since: int) -> Dict[str, Any]:
    ring = list(lg.event_ring)
    events = [e for e in ring if int(e.get("seq", 0)) > since]
    oldest = int(ring[0].get("seq", 0)) if ring else 0
    return {
        "events": events,
        "alerts": list(lg.alerts_buf),
        "captions": list(lg.captions),
        "frames": list(lg.tracking_frames),
        "last_seq": lg.seq,
        "gap": bool(ring) and since + 1 < oldest,
    }


def _run_parity_cli(argv: List[str]) -> int:
    import argparse

    from services.franchise_paths import ensure_simengine_path

    ensure_simengine_path()
    from app.sim_engine.league.schedule_generator import _safe_slot_team_id
    from services.franchise_sim import start_franchise

    p = argparse.ArgumentParser()
    p.add_argument("--parity", action="store_true")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    args, _ = p.parse_known_args(argv)
    if not args.parity:
        return 0
    s = start_franchise(team_query="Ottawa Senators", head_coach_name="Parity", coach_archetype="balanced", seed=int(args.seed))
    s.session_id = f"parity-{args.seed}"
    uid = str(s.user_team_id)
    failed = matched = 0
    checked = 0
    for d, slots in sorted((s.by_day or {}).items()):
        if checked >= int(args.games):
            break
        for sl in slots or []:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            if uid not in (hid, aid):
                continue
            s.calendar_cursor = int(d)
            try:
                lg = build_live_game(s)
            except Exception:
                continue
            loop = lg.loop
            loop.run_to_final()
            live_h = int(lg.score_home)
            live_a = int(lg.score_away)
            assert loop._result is not None
            eng_h = int(loop._result.get("display_home_goals") or 0)
            eng_a = int(loop._result.get("display_away_goals") or 0)
            if live_h == eng_h and live_a == eng_a:
                matched += 1
            else:
                failed += 1
                print(f"FAIL day={d} live={live_h}-{live_a} engine={eng_h}-{eng_a}")
            checked += 1
            with _registry_lock:
                _LIVE.pop(str(s.session_id), None)
            break
    print(f"Parity: {matched} passed, {failed} failed, {checked} games (display score vs stepped replay)")
    return 1 if failed else 0


def build_live_game(session: Any, lang: str = "en") -> LiveGame:
    """Pre-game only: lineup attach, scales, starters — no score, no fatigue tick."""
    from app.sim_engine.league.schedule_generator import _safe_slot_team_id
    from services import franchise_sim as fs

    if str(getattr(session, "phase", "") or "") not in ("regular", "preseason"):
        raise HTTPException(status_code=409, detail={"reason": "wrong_phase", "message": "Live sim is only for the regular season."})

    slot = _find_user_slot(session)
    if slot is None:
        raise HTTPException(status_code=409, detail={"reason": "no_user_game_today", "message": "No game for your club today."})

    lineup_block = fs.user_lineup_advance_gate(session, auto_fill=False)
    if lineup_block is not None:
        raise HTTPException(
            status_code=409,
            detail={"reason": "lineup_blocked", "message": "Complete your lines first.", "step": lineup_block},
        )

    hid = _safe_slot_team_id(slot, "home_id")
    aid = _safe_slot_team_id(slot, "away_id")
    home = session.team_by_id.get(hid)
    away = session.team_by_id.get(aid)
    if home is None or away is None:
        raise HTTPException(status_code=404, detail={"reason": "teams_missing", "message": "Could not resolve clubs for this game."})

    h_goal = fs._ensure_goalie_for_game(home)
    a_goal = fs._ensure_goalie_for_game(away)
    if int(h_goal.get("total", 0) or 0) <= 0 or int(a_goal.get("total", 0) or 0) <= 0:
        raise HTTPException(
            status_code=409,
            detail={"reason": "goalie_shortage", "message": "Fix goalie availability before starting live sim."},
        )

    uid = str(session.user_team_id)
    user_side = "home" if uid == hid else "away"
    day = int(getattr(session, "calendar_cursor", 0) or 0)
    is_playoff = bool(getattr(slot, "is_playoff", False))
    sim = session.sim

    session_id = str(getattr(session, "session_id", "") or "")
    season_y = int(getattr(session, "season_calendar_year", 0) or 0)
    seed = zlib.crc32(f"{session_id}|{season_y}|{hid}|{aid}|{day}|live".encode()) & 0xFFFFFFFF
    rng_pre = random.Random(seed ^ 0x51A7E)

    h_scale, a_scale, noise_scale = _read_strength_scales(session, home, away, hid, aid, day, is_playoff)

    fs._attach_franchise_saved_lineups(session, home, away, home_id=hid, away_id=aid, user_tid=uid)
    starters: Any = None
    try:
        starters = sim._gm_pick_game_starters(rng_pre, home, away, calendar_day=day, home_b2b=False, away_b2b=False, is_playoff=is_playoff)
    except TypeError:
        starters = sim._gm_pick_game_starters(rng_pre, home, away)
    # Pass these starters to every engine pass (fast sim does the same) instead of relying on a
    # stale sim._locked_*_goalie left over from some other game.
    h_start, a_start = starters if isinstance(starters, (tuple, list)) and len(starters) == 2 else (None, None)

    live_id = str(uuid.uuid4())
    lg = LiveGame(
        live_id=live_id,
        session_id=session_id,
        game_key={"home_id": hid, "away_id": aid, "day": day, "is_playoff": is_playoff},
        seed=int(seed),
        user_side=user_side,
        home_id=hid,
        away_id=aid,
        is_playoff=is_playoff,
        calendar_day=day,
        home=home,
        away=away,
        sim=sim,
        strength={"home_scale": h_scale, "away_scale": a_scale, "noise_scale": noise_scale},
        scratch={"home_goalie": h_start, "away_goalie": a_start},
        lang=_norm_lang(lang),
    )
    lg.loop = LiveEventLoop.attach(lg, session)
    return lg


_TEAM_COLORS: Dict[str, List[str]] = {
    "ANA": ["#f47a20", "#b9975b"], "ARI": ["#8c2633", "#e2d6b5"], "BOS": ["#ffb81c", "#111111"], "BUF": ["#003087", "#ffb81c"],
    "CAR": ["#cc0000", "#a2aaad"], "CBJ": ["#002654", "#ce1126"], "CGY": ["#c8102e", "#f1be48"], "CHI": ["#cf0a2c", "#ff671b"],
    "COL": ["#6f263d", "#236192"], "DAL": ["#006847", "#8f8f8c"], "DET": ["#ce1126", "#ffffff"], "EDM": ["#ff4c00", "#041e42"],
    "FLA": ["#c8102e", "#b9975b"], "LAK": ["#a2aaad", "#111111"], "MIN": ["#154734", "#a6192e"], "MTL": ["#af1e2d", "#192168"],
    "NJD": ["#ce1126", "#000000"], "NSH": ["#ffb81c", "#041e42"], "NYI": ["#00539b", "#f47d30"], "NYR": ["#0038a8", "#ce1126"],
    "OTT": ["#c52032", "#c2912c"], "PHI": ["#f74902", "#000000"], "PIT": ["#fcb514", "#000000"], "SEA": ["#99d9d9", "#e9072b"],
    "SJS": ["#006d75", "#ea7200"], "STL": ["#002f87", "#fcb514"], "TBL": ["#002868", "#ffffff"], "TOR": ["#00205b", "#ffffff"],
    "UTA": ["#6cace4", "#010101"], "VAN": ["#00843d", "#00205b"], "VGK": ["#b4975a", "#333f42"], "WPG": ["#041e42", "#7b303e"],
    "WSH": ["#c8102e", "#041e42"],
}  # mirrors frontend BoxScoreArcade TEAM_COLORS


def _team_colors(tm: Any) -> List[str]:
    for attr in ("colors", "team_colors"):
        v = getattr(tm, attr, None)
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) and x.startswith("#") for x in v[:2]):
            return [str(x) for x in v[:2]] if len(v) >= 2 else [str(v[0]), "#ffffff"]
    prim = getattr(tm, "primary_color", None)
    if isinstance(prim, str) and prim.startswith("#"):
        sec = getattr(tm, "secondary_color", None)
        return [prim, sec if isinstance(sec, str) and sec.startswith("#") else "#ffffff"]
    abbr = ""
    try:
        from services import franchise_sim as fs

        abbr = fs._franchise_team_abbrev(tm)
    except Exception:
        abbr = str(getattr(tm, "abbr", "") or "")[:3].upper()
    if abbr in _TEAM_COLORS:
        return list(_TEAM_COLORS[abbr])
    h = zlib.crc32(abbr.encode()) & 0xFFFFFF
    return [f"#{h:06x}", f"#{(h ^ 0x555555) & 0xFFFFFF:06x}"]


def _team_card(session: Any, tm: Any, tid: str, *, is_user: bool) -> Dict[str, Any]:
    from services import franchise_sim as fs

    prof = fs._ensure_team_fan_profile(session, tid)
    score = int(prof.get("fan_confidence", 55) or 55)
    return {
        "id": tid,
        "abbr": fs._franchise_team_abbrev(tm),
        "name": fs._display_team(tm),
        "city": str(getattr(tm, "city", "") or ""),
        "colors": _team_colors(tm),
        "logo_token": fs._franchise_team_abbrev(tm),
        "fan_confidence": score,
        "fan_label": fs._fan_category_from_score(score),
        "is_user": bool(is_user),
    }


def _empty_manpower() -> Dict[str, Any]:
    return {
        "state": "EV", "perspective": "home", "home_skaters": 5, "away_skaters": 5,
        "pp_side": None, "pp_sec_left": None, "empty_net": {"home": False, "away": False},
    }


def _bench_block(loop: "LiveEventLoop", now: float) -> Dict[str, Any]:
    lg = loop.lg
    side = lg.user_side
    lines = loop.lines[side]
    F = [list(x) for x in (lines.get("F") or [])][:4]
    D = [list(x) for x in (lines.get("D") or [])][:3]
    G = list(lines.get("G") or [])[:2]
    on = set(loop.state.on_ice.get(side) or [])
    gid = loop.current_goalie(side)
    players: List[Dict[str, Any]] = []
    seen: set = set()

    def row(pid: str, label: str) -> None:
        if not pid or pid in seen:
            return
        seen.add(pid)
        chip = loop.book.chip(pid) or {"player_id": pid, "name": pid, "short": pid, "number": None, "pos": "?"}
        st = loop.state.p.get(pid) or {}
        g, a, sog = int(st.get("g", 0)), int(st.get("a", 0)), int(st.get("sog", 0))
        pm = int(st.get("plus_minus", 0))
        is_g = label == "G"
        players.append({
            **chip, "line_label": label, "on_ice": (pid == gid) if is_g else (pid in on),
            "toi_sec": round(loop.shifts.toi_est(pid, side), 1) if not is_g else round(min(now, _REGULATION_GAME_SEC + loop._ot_seg_len()), 1) if pid == gid else 0.0,
            "shifts": int(loop.shifts.shifts.get(pid, 0)) if not is_g else (1 if pid == gid else 0),
            "fatigue": loop.shifts.fatigue(pid, now) if not is_g else 0.0,
            "g": g, "a": a, "sog": sog, "plus_minus": pm,
            "hot": (g + a) >= 2 or sog >= 4,
            "cold": pm <= -2 or (now >= _REGULATION_PERIOD_SEC and sog == 0 and int(st.get("ga_on", 0)) >= 2),
            "benched": pid in loop.benched, "double_shift": pid in loop.double_shift,
        })

    for j, ln in enumerate(F):
        for pid in ln:
            row(pid, f"L{j + 1}")
    for j, pr in enumerate(D):
        for pid in pr:
            row(pid, f"D{j + 1}")
    for pid in G:
        row(pid, "G")
    cur = loop.state.on_line.get(side) or {}
    return {
        "user_side": side,
        "lines": {"F": F, "D": D, "G": G},
        "on_line": {"F": cur.get("F"), "D": cur.get("D")},
        "players": players,
    }


def _final_block(loop: "LiveEventLoop") -> Optional[Dict[str, Any]]:
    if not loop.finished:
        return None
    out = dict(loop.final_info() or {})
    if loop.analytics is not None:
        try:
            out["stars"] = loop.analytics.stars()
        except Exception:
            logger.warning("three stars failed", exc_info=True)
            out["stars"] = []
    return out


def _roster(loop: "LiveEventLoop") -> Dict[str, Any]:
    key = (len(loop.book.chips), id(loop._cap), int(loop.lg.score_home) + int(loop.lg.score_away))
    if loop._roster_cache[0] == key:
        return loop._roster_cache[1]
    out: Dict[str, Any] = {}
    pss = getattr(loop.session, "player_season_stats", None) or {}
    for pid, chip in loop.book.chips.items():
        side = loop.book.side.get(pid)
        if side in _SIDES:
            row = pss.get(pid) or {}
            live = loop.state.p.get(pid) or {}
            g = int(row.get("g", 0) or 0) + int(live.get("g", 0) or 0)
            a = int(row.get("a", 0) or 0) + int(live.get("a", 0) or 0)
            out[pid] = {"short": chip["short"], "name": chip["name"], "number": chip["number"], "pos": chip["pos"], "side": side,
                        "season_goals": g, "season_points": g + a, **loop.headshot_fields(pid)}
    loop._roster_cache = (key, out)
    return out


def _situation(loop: "LiveEventLoop", now: float, lang: str) -> Dict[str, Any]:
    lg = loop.lg
    S = lambda key, **kw: _fmt(_loc(_SITUATION, lang, key), kw)  # noqa: E731
    f = loop.motion.last_frame
    zone = loop.motion.zone()
    poss = f.get("pos") if f else None
    parts: List[str] = []
    if lg.status == "pregame":
        parts.append(S("pregame"))
    elif loop.finished:
        parts.append(S("final", score_line=f"{loop.abbr('away')} {lg.score_away}, {loop.abbr('home')} {lg.score_home}"))
    elif loop.intermission:
        parts.append(S("intermission"))
    elif loop.phase == "shootout":
        parts.append(S("shootout"))
    else:
        mp = loop.manpower(now) if loop._engine_captured else _empty_manpower()
        if mp.get("pp_side") in _SIDES and mp.get("pp_sec_left") is not None:
            parts.append(S("pp", team=loop.team_say(mp["pp_side"], lang), time=_clock_display(mp["pp_sec_left"]).lstrip("0")))
        for s in _SIDES:
            if loop.goalie_pulled.get(s):
                parts.append(S("en", team=loop.team_say(s, lang)))
        key = loop.motion.situation_key()
        ph = f.get("ph") if f else None
        if ph == "goal":
            parts.append(S("goal", team=loop.team_say(f.get("pos") or (loop.motion._cur_seg or {}).get("side"), lang)))
        elif key is not None:
            kind, side = key
            if kind == "stop":
                parts.append(S("stop"))
            else:
                parts.append(S(kind, team=loop.team_say(side, lang), opp=loop.team_say(_opp(side), lang)))
        elif ph == "play":
            parts.append(S("loose"))
    text = " — ".join(p for p in parts if p)
    text = _cap(text) + ("." if text and not text.endswith(".") else "")
    if lang == "fr":
        text = _fr_elide(text)
    w = loop.motion.last_whistle
    last_whistle = None
    if w:
        wk, wp = w
        last_whistle = _fmt(_loc(_WHISTLE, lang, wk), wp)
    return {"text": text, "zone": zone, "possession": poss, "last_whistle": last_whistle}


def snapshot_from_live(lg: LiveGame, session: Any, *, lite: bool = False) -> Dict[str, Any]:
    loop: Optional[LiveEventLoop] = lg.loop
    lang = _norm_lang(getattr(lg, "lang", "en"))
    if loop is None:
        raise HTTPException(status_code=500, detail="Live loop not initialized")
    now = float(loop.game_sec_elapsed)
    loop._refresh_status()
    ck = (lg.seq, round(now, 2), lang, lg.status, len(loop.coach.orders), len(loop.coach.pending), lg.paused,
          loop.motion.t_last, bool(lite))
    if loop._snap_cache[0] == ck and loop._snap_cache[1] is not None:
        return loop._snap_cache[1]
    teams = loop.team_cards()
    an = None
    if loop.analytics is not None:
        try:
            an = loop.analytics.build()
            loop.analytics.notify(an, loop.pt_now())
        except Exception:
            logger.warning("analytics build failed", exc_info=True)
    tts = ({"up": _live_tts_up(), "engine": "neural" if _live_tts_up() else None, "languages": []} if _live_tts_mod() is not None
           else (_tts_probe() if LIVE_TTS_ENABLED else {"up": False, "engine": None, "languages": []}))
    rv = loop.rivalry or {}
    tier = int(rv.get("tier") or 0)
    on_ice = {
        "home": [c for c in (loop.book.chip(p) for p in loop.state.on_ice["home"]) if c],
        "away": [c for c in (loop.book.chip(p) for p in loop.state.on_ice["away"]) if c],
        "home_goalie": loop.book.chip(loop.current_goalie("home")),
        "away_goalie": loop.book.chip(loop.current_goalie("away")),
    }
    snap = {
        "live_id": lg.live_id,
        "session_id": lg.session_id,
        "game_key": dict(lg.game_key),
        "status": lg.status,
        "seq": lg.seq,
        "lang": lang,
        "languages": [dict(x) for x in LIVE_LANGUAGES],
        "clock": loop.clock_state(),
        "pt": round(loop.pt_now(), 2), "gc": round(now, 2),
        "score": {"home": lg.score_home, "away": lg.score_away},
        "shots": {"home": lg.shots_home, "away": lg.shots_away},
        "xg": {"home": round(lg.xg_home, 3), "away": round(lg.xg_away, 3)},
        "teams": teams,
        "stats": {"home": loop.state.team_stats("home"), "away": loop.state.team_stats("away")},
        "manpower": loop.manpower(now) if loop._engine_captured else _empty_manpower(),
        "on_ice": on_ice,
        "bench": _bench_block(loop, now),
        "momentum": {"value": round(float(loop.momentum.value), 3), "label": loop.momentum.label(lang), "history": list(loop.momentum.history)},
        "crowd": {
            "level": round(float(loop.crowd.level), 3), "mood": loop.crowd.mood, "chant": loop.crowd.chant,
            "rivalry": {"tier": tier, "name": _loc(_RIVAL_NAME, lang, rv.get("name_key")) if rv.get("name_key") is not None else None},
            "fan_meter": loop.fan_meter,
        },
        "tracking": {"authority": "model_estimated", "model": True, "frames": loop.tracking.frames},
        "roster": _roster(loop),
        "situation": _situation(loop, now, lang),
        "alerts": loop.alerts.active(now),
        "analytics": an,
        "notifications": list(loop.analytics.notes) if loop.analytics is not None else [],
        "commentary": list(loop.commentary.lines[-30:]),
        "coach": {
            "timeout_used": bool(loop.timeout_used),
            "goalie_pulled": bool(loop.goalie_pulled.get(lg.user_side)),
            "tactics": dict(loop.tactics),
            "deployment": loop.deploy_mode,
            "benched": sorted(loop.benched),
            "double_shift": sorted(loop.double_shift),
            "pending": loop.coach.pending_orders(),
            "applied": loop.coach.applied()[-20:],
            "last_result": loop.coach.last_result,
            "cpu": {"timeout_used": bool(loop.cpu_timeout_used), "goalie_pulled": bool(loop.goalie_pulled.get(_opp(lg.user_side)))},
            "allowed_kinds": list(CoachController.ALLOWED) + ["notification_ack"],
        },
        "speed": int(lg.speed),
        "flags": {
            "tts_worker": bool(tts.get("up")),
            "tts_engine": (tts.get("engine") or "browser") if (LIVE_TTS_ENABLED and tts.get("up")) else "browser",
            "tts_enabled": bool(LIVE_TTS_ENABLED),
            "tracking": bool(LIVE_TRACKING_ENABLED),
            "commentator": COMMENTATOR_PERSONA,
        },
        "final": _final_block(loop),
        "product": {
            "shootout_model": SHOOTOUT_MODEL,
            "penalty_labels": PENALTY_LABEL_MODE,
            "tts_voice_mode": TTS_VOICE_MODE,
            "soundtrack_folder": USE_SOUNDTRACK_FOLDER_AUDIO,
        },
    }
    loop._snap_cache = (ck, snap)
    return snap


def _session(sid: Optional[str]) -> Any:
    if not sid:
        raise HTTPException(status_code=400, detail="Missing X-Franchise-Session header")
    s = get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown or expired franchise session")
    if not getattr(s, "session_id", None):
        setattr(s, "session_id", sid)
    return s


def _get_live_for_session(session_id: str) -> LiveGame:
    with _registry_lock:
        lg = _LIVE.get(str(session_id))
        if lg is None:
            raise HTTPException(status_code=404, detail={"reason": "no_live_game", "message": "No live game for this session."})
        if lg.status in ("committed", "aborted"):
            raise HTTPException(status_code=404, detail={"reason": "live_game_closed", "message": "Live game already finished."})
        _touch(lg)
        return lg


_ROUTER_PREFIX = "/api/franchise/live"
router = APIRouter(prefix=_ROUTER_PREFIX, tags=["live-sim"]) if APIRouter is not None else None


# ═══ 8. Local TTS tier (optional; default audio path is the browser's Web Speech API)

_TTS_STATUS: Dict[str, Any] = {"t": 0.0, "up": False, "engine": None, "languages": []}
_TTS_TEXT: Dict[str, Tuple[str, str, str]] = {}
_TTS_CACHE: Dict[str, bytes] = {}
_TTS_LOCK = threading.Lock()


def _tts_opener() -> Any:
    import urllib.request

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local worker: never via a proxy


# ── neural TTS client (services/live_tts.py, owned by the TTS agent). Optional: without it the old
# worker probe below is used, and without any worker the browser voice reads tts_text.
_LIVE_TTS: Dict[str, Any] = {"mod": None, "tried": 0.0, "up": False, "t": 0.0, "thread": None}


def _live_tts_mod() -> Any:
    now = time.time()
    if _LIVE_TTS["mod"] is None and now - float(_LIVE_TTS["tried"]) > 30.0:
        _LIVE_TTS["tried"] = now
        try:
            from services import live_tts as _m  # type: ignore
            _LIVE_TTS["mod"] = _m
        except ImportError:
            _LIVE_TTS["mod"] = None
        except Exception:
            logger.warning("live_tts import failed", exc_info=True)
            _LIVE_TTS["mod"] = None
    return _LIVE_TTS["mod"]


def _live_tts_refresher() -> None:
    while True:
        mod = _LIVE_TTS["mod"]
        try:
            _LIVE_TTS["up"] = bool(mod.tts_available()) if mod is not None else False
        except Exception:
            _LIVE_TTS["up"] = False
        _LIVE_TTS["t"] = time.time()
        time.sleep(4.0)


def _live_tts_up() -> bool:
    """Never blocks the request thread (F3): the worker probe runs on a daemon thread."""
    mod = _live_tts_mod()
    if mod is None:
        return False
    th = _LIVE_TTS["thread"]
    if th is None or not th.is_alive():
        th = threading.Thread(target=_live_tts_refresher, name="live-tts-probe", daemon=True)
        _LIVE_TTS["thread"] = th
        th.start()
    return bool(_LIVE_TTS["up"])


def _tts_attach(line: Dict[str, Any]) -> None:
    """Queue neural audio for a commentary line (non-blocking) and set audio_key / audio_url."""
    line.setdefault("audio_key", None)
    line.setdefault("audio_url", None)
    try:
        if _live_tts_up():
            key = _LIVE_TTS["mod"].request_clip(line.get("tts_text") or line.get("text"), line.get("lang") or "en",
                                                line.get("speaker") or "pbp", line.get("profile") or "calm", line.get("tts_hint"))
            if key:
                line["audio_key"] = key
                line["audio_url"] = f"{_ROUTER_PREFIX}/tts/{key}"
            return
        if _live_tts_mod() is None and LIVE_TTS_ENABLED:
            url = _tts_audio_url(line.get("tts_text") or "", line.get("lang") or "en", line.get("profile") or "calm")
            if url:
                line["audio_url"] = url
                line["audio_key"] = url.rsplit("/", 1)[-1]
    except Exception:
        logger.warning("tts request failed", exc_info=True)


def _tts_probe(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    with _TTS_LOCK:
        if not force and now - float(_TTS_STATUS["t"]) < 15.0:
            return dict(_TTS_STATUS)
        if not force:  # F3: refresh in the background, answer from the cache
            if not _TTS_STATUS.get("probing"):
                _TTS_STATUS["probing"] = True
                threading.Thread(target=_tts_probe_bg, daemon=True).start()
            return dict(_TTS_STATUS)
    up, engine, langs = False, None, []
    try:
        import json as _json

        with _tts_opener().open(LIVE_TTS_WORKER_URL.rstrip("/") + "/capabilities", timeout=0.6) as r:
            data = _json.loads(r.read().decode("utf-8") or "{}")
        engine = data.get("engine") or None
        langs = [str(x) for x in (data.get("languages") or [])]
        up = bool(engine and engine != "none")
    except Exception:
        up = False
    with _TTS_LOCK:
        _TTS_STATUS.update({"t": now, "up": up, "engine": engine, "languages": langs, "probing": False})
        return dict(_TTS_STATUS)


def _tts_probe_bg() -> None:
    try:
        _tts_probe(force=True)
    except Exception:
        with _TTS_LOCK:
            _TTS_STATUS["probing"] = False


def _tts_audio_url(text: str, lang: str, profile: str) -> Optional[str]:
    if not LIVE_TTS_ENABLED or not text:
        return None
    st = _tts_probe()
    if not st.get("up") or lang not in (st.get("languages") or []):
        return None
    key = hashlib.sha1(f"{lang}|{profile}|{text}".encode("utf-8")).hexdigest()[:20]
    with _TTS_LOCK:
        _TTS_TEXT[key] = (text, lang, profile)
        if len(_TTS_TEXT) > 400:
            for k in list(_TTS_TEXT)[:100]:
                _TTS_TEXT.pop(k, None)
    return f"{_ROUTER_PREFIX}/tts/{key}"


def _tts_fetch(key: str) -> Optional[bytes]:
    with _TTS_LOCK:
        if key in _TTS_CACHE:
            return _TTS_CACHE[key]
        item = _TTS_TEXT.get(key)
    if item is None:
        return None
    import json as _json
    import urllib.request

    text, lang, profile = item
    req = urllib.request.Request(
        LIVE_TTS_WORKER_URL.rstrip("/") + "/synth",
        data=_json.dumps({"text": text, "lang": lang, "profile": profile}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with _tts_opener().open(req, timeout=15.0) as r:
        if int(getattr(r, "status", 200)) != 200:
            return None
        wav = r.read()
    with _TTS_LOCK:
        _TTS_CACHE[key] = wav
        if len(_TTS_CACHE) > 200:
            for k in list(_TTS_CACHE)[:50]:
                _TTS_CACHE.pop(k, None)
    return wav


def _set_lang(lg: LiveGame, v: Any) -> None:
    if v:
        lg.lang = _norm_lang(v)


from contextlib import contextmanager


@contextmanager
def _session_slot(sid: str, timeout: float = 2.5) -> Any:
    """The session's sim slot (same lock as services.sim_guard.exclusive_sim), but WAITING up to
    `timeout` instead of failing at once: overlapping client calls queue briefly instead of 409."""
    lock = None
    try:
        from services import sim_guard as _sg

        lock = _sg._lock_for(str(sid or "").strip() or "_default")
    except Exception:
        lock = None
    if lock is None:
        from services.sim_guard import exclusive_sim

        with exclusive_sim(sid) as owned:
            yield owned
        return
    got = lock.acquire(timeout=timeout)
    try:
        yield got
    finally:
        if got:
            lock.release()


def _perf(name: str, t0: float, parts: Dict[str, float], payload: Dict[str, Any]) -> Dict[str, Any]:
    total = round((time.perf_counter() - t0) * 1000.0, 1)
    perf = {"total": total, **{k: round(v, 1) for k, v in parts.items()}}
    if total > 250.0:
        logger.warning("live %s slow: %s", name, perf)
    payload["perf_ms"] = perf
    return payload


def _snapshot_nowait(lg: LiveGame, session: Any, *, lite: bool = False) -> Dict[str, Any]:
    """Snapshot without waiting on a running /advance: use the cached one if the loop is busy."""
    got = _try_lock(lg, 0.05)
    try:
        if not got and lg.loop is not None and lg.loop._snap_cache[1] is not None:
            return lg.loop._snap_cache[1]
        return snapshot_from_live(lg, session, lite=lite)
    finally:
        if got:
            lg.lock.release()


if router is not None:

    @router.post("/start")
    def live_start(
        body: Optional[Dict[str, Any]] = None,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        from services.json_safe import json_safe

        t0 = time.perf_counter()
        parts: Dict[str, float] = {}
        if not LIVE_SIM_ENABLED:
            raise HTTPException(status_code=503, detail={"reason": "disabled", "message": "Live sim is disabled."})
        body = body or {}
        session = _session(x_franchise_session)
        session_id = str(x_franchise_session or getattr(session, "session_id", ""))
        if session_id:
            setattr(session, "session_id", session_id)

        _evict_stale_unlocked()
        with _registry_lock:
            existing = _LIVE.get(session_id)
        if existing is not None and existing.status not in ("committed", "aborted"):
            _set_lang(existing, body.get("lang"))
            return json_safe(_perf("start", t0, parts, dict(_snapshot_nowait(existing, session))))

        with _session_slot(session_id) as owned:
            if not owned:
                raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
            t1 = time.perf_counter()
            lg = build_live_game(session, lang=_norm_lang(body.get("lang")))
            parts["build"] = (time.perf_counter() - t1) * 1000.0
            t1 = time.perf_counter()
            try:
                lg.loop.prepare()  # engine pass now, so the first /advance is cheap
            except Exception:
                logger.exception("live: engine pass at start failed; will retry on first advance")
            parts["engine_pass"] = (time.perf_counter() - t1) * 1000.0
        with _registry_lock:
            existing = _LIVE.get(session_id)
            if existing is not None and existing.status not in ("committed", "aborted"):
                lg = existing  # a concurrent /start won the race
            else:
                _LIVE[session_id] = lg
        logger.info("live_sim start session=%s live_id=%s %s@%s", session_id, lg.live_id, lg.away_id, lg.home_id)
        return json_safe(_perf("start", t0, parts, dict(snapshot_from_live(lg, session))))

    @router.get("/state")
    def live_state(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        from services.json_safe import json_safe

        t0 = time.perf_counter()
        session = _session(x_franchise_session)
        lg = _get_live_for_session(str(x_franchise_session))
        return json_safe(_perf("state", t0, {}, dict(_snapshot_nowait(lg, session))))

    @router.post("/abort")
    def live_abort(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        session = _session(x_franchise_session)
        session_id = str(x_franchise_session or "")
        with _registry_lock:
            lg = _LIVE.pop(session_id, None)
        if lg is not None:
            lg.status = "aborted"
            logger.info("live_sim abort session=%s live_id=%s", session_id, lg.live_id)
        if hasattr(session, "_live_score_override"):
            session._live_score_override = None
        return {"ok": True, "aborted": bool(lg)}

    @router.get("/open")
    def live_open(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        """Cheap poll for calendar UI: is a live game in progress?"""
        sid = str(x_franchise_session or "")
        lid = has_open_live_game(sid)
        return {"open": bool(lid), "live_id": lid}

    @router.get("/events")
    def live_events(
        since_seq: int = 0,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        from services.json_safe import json_safe

        session = _session(x_franchise_session)
        lg = _get_live_for_session(str(x_franchise_session))
        payload = _events_since(lg, int(since_seq))
        payload["snapshot_lite"] = _snapshot_nowait(lg, session, lite=True)
        return json_safe(payload)

    @router.post("/advance")
    def live_advance(
        body: Optional[Dict[str, Any]] = None,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        from services.json_safe import json_safe

        t0 = time.perf_counter()
        parts: Dict[str, float] = {}
        body = body or {}
        session = _session(x_franchise_session)
        sid = str(x_franchise_session or "")
        lg = _get_live_for_session(sid)
        since = int(body.get("since_seq", 0) or 0)
        with _session_slot(sid) as owned:
            if not owned:
                raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
            if not _try_lock(lg, 2.5):
                raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
            try:
                loop = lg.loop
                if loop is None:
                    raise HTTPException(status_code=500, detail="Live loop not initialized")
                _set_lang(lg, body.get("lang"))
                if body.get("speed") is not None:
                    try:
                        lg.speed = max(1, min(16, int(body.get("speed"))))
                    except (TypeError, ValueError):
                        pass
                t1 = time.perf_counter()
                if not loop._engine_captured:
                    loop._capture_engine_timeline()
                    parts["engine_pass"] = (time.perf_counter() - t1) * 1000.0
                    t1 = time.perf_counter()
                if loop.coach.process_pending():
                    parts["coach_resim"] = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                skip = False
                if not lg.paused:
                    if body.get("sim_to_end"):
                        loop.run_to_final()
                        skip = True
                    else:
                        if loop.intermission:
                            loop.resume_from_intermission()  # this /advance skips the intermission
                        if body.get("until_pt") is not None:  # pace by presentation time (stoppages take real time)
                            until_t = loop.gc_for_pt(float(body.get("until_pt")))
                        else:
                            until_t = float(body.get("until_t", loop.game_sec_elapsed + 5.0))
                        loop.step_until(until_t, cap_sec=60.0)
                parts["step"] = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                motion = loop.take_motion(skipped=skip)
                parts["motion"] = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                payload = _events_since(lg, since)
                payload["motion"] = motion
                payload["snapshot_lite"] = snapshot_from_live(lg, session, lite=True)
                parts["snapshot"] = (time.perf_counter() - t1) * 1000.0
                _touch(lg)
                t1 = time.perf_counter()
                out = json_safe(payload)
                parts["serialize"] = (time.perf_counter() - t1) * 1000.0
                parts["frames"] = float(len(motion.get("frames") or []))
                return _perf("advance", t0, parts, out)
            finally:
                lg.lock.release()

    @router.post("/coach")
    def live_coach(
        body: Optional[Dict[str, Any]] = None,
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        """Validates and queues; the engine re-simulation runs at the start of the next /advance."""
        from services.json_safe import json_safe

        t0 = time.perf_counter()
        body = body or {}
        session = _session(x_franchise_session)
        sid = str(x_franchise_session or "")
        lg = _get_live_for_session(sid)
        if not _try_lock(lg, 2.5):
            raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
        try:
            loop = lg.loop
            if loop is None:
                raise HTTPException(status_code=500, detail="No loop")
            _set_lang(lg, body.get("lang"))
            order = loop.coach.submit(body)
            _touch(lg)
            return json_safe(_perf("coach", t0, {}, {"order": order, "snapshot": snapshot_from_live(lg, session)}))
        finally:
            lg.lock.release()

    @router.post("/pause")
    def live_pause(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        lg = _get_live_for_session(str(x_franchise_session))
        got = _try_lock(lg, 2.5)
        try:
            if lg.loop is not None and not lg.loop.finished:
                lg.paused = True
                lg.loop._refresh_status()
        finally:
            if got:
                lg.lock.release()
        return {"state": lg.status}

    @router.post("/resume")
    def live_resume(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        lg = _get_live_for_session(str(x_franchise_session))
        got = _try_lock(lg, 2.5)
        try:
            lg.paused = False
            if lg.loop is not None:
                lg.loop._refresh_status()
        finally:
            if got:
                lg.lock.release()
        return {"state": lg.status}

    @router.post("/commit")
    def live_commit(x_franchise_session: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        from services.json_safe import json_safe

        t0 = time.perf_counter()
        session = _session(x_franchise_session)
        sid = str(x_franchise_session or "")
        with _session_slot(sid, timeout=5.0) as owned:
            if not owned:
                raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
            lg = _get_live_for_session(sid)
            if not _try_lock(lg, 5.0):
                raise HTTPException(status_code=409, detail={"reason": "sim_busy", "busy": True})
            try:
                if lg.loop and lg.loop.coach.pending:
                    lg.loop.coach.process_pending()
                if lg.loop and not lg.loop.finished:
                    lg.paused = False
                    lg.loop.run_to_final()
                out = commit_live_game(session, lg)
                return json_safe(_perf("commit", t0, {}, out))
            finally:
                lg.lock.release()

    @router.get("/tts-status")
    def live_tts_status() -> Dict[str, Any]:
        mod = _live_tts_mod()
        if mod is not None:
            try:
                out = dict(mod.status() or {})
                out.setdefault("enabled", True)
                out["source"] = "live_tts"
                return out
            except Exception:
                logger.warning("live_tts.status failed", exc_info=True)
        st = _tts_probe(force=True)
        up = bool(st.get("up"))
        return {
            "worker": "up" if up else "down",
            "engine": (st.get("engine") or "browser") if (up and LIVE_TTS_ENABLED) else "browser",
            "worker_engine": st.get("engine"),
            "enabled": bool(LIVE_TTS_ENABLED),
            "languages": list(st.get("languages") or []) if up else [x["code"] for x in LIVE_LANGUAGES],
        }

    @router.get("/tts/{audio_key}")
    def live_tts_clip(audio_key: str) -> Any:
        from fastapi import Response

        mod = _live_tts_mod()
        if mod is not None:
            key = "".join(ch for ch in str(audio_key) if ch.isalnum())[:64]
            try:
                stt = mod.clip_status(key)
            except Exception:
                stt = "failed"
            if stt == "ready":
                got = mod.clip_bytes(key)
                if got:
                    data, mime = got
                    return Response(content=data, media_type=mime or "audio/wav")
            if stt == "pending":
                return Response(content=b"", media_type="text/plain", status_code=202)
            raise HTTPException(status_code=404, detail={"reason": "unknown_clip" if stt != "failed" else "failed", "key": key})
        if not LIVE_TTS_ENABLED:
            raise HTTPException(status_code=503, detail={"reason": "disabled"})
        key = "".join(ch for ch in str(audio_key) if ch.isalnum())[:40]
        try:
            wav = _tts_fetch(key)
        except Exception as e:
            logger.warning("tts fetch failed: %s", e)
            raise HTTPException(status_code=503, detail={"reason": "worker_unavailable"}) from e
        if not wav:
            raise HTTPException(status_code=404, detail={"reason": "unknown_clip", "key": key})
        return Response(content=wav, media_type="audio/wav")


if __name__ == "__main__":
    if "--tts-worker" in sys.argv:
        print("The local TTS worker moved to backend/tts_worker/worker.py (client: services/live_tts.py).", flush=True)
        raise SystemExit(2)
    if "--parity" in sys.argv:
        raise SystemExit(_run_parity_cli(sys.argv))
    if "--tts-qa" in sys.argv:
        print(f"TTS QA corpus: {len(TTS_QA_CORPUS)} entries.", flush=True)
        raise SystemExit(0)
