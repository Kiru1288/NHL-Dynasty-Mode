"""In-process client for the local neural TTS worker (backend/tts_worker/worker.py).

Stdlib only. The FastAPI backend calls:

    tts_available() -> bool                      worker up and ready (probe cached ~5 s)
    request_clip(text, lang, speaker, profile, hint) -> str | None
                                                 non-blocking; returns the audio key, or None
                                                 when the worker is down / language unsupported
    clip_status(key) -> "ready" | "pending" | "failed"
    clip_bytes(key)  -> (bytes, mime) | None
    status()         -> dict for the UI ("Neural voices: ON (Kokoro)" vs setup hint)

speaker: "pbp" | "color";  profile: hype | build | calm | somber
hint:    {"energy": 0..1, "pace": "slow"|"normal"|"fast", "pause_before_ms": int} or None
         (pause_before_ms is NOT baked into the clip; the player applies it.)

Clips are cached in memory (LRU, 300) and on disk at backend/data/tts_cache/.
Synthesis requests go to the worker from a small thread pool fed by a bounded
priority queue: goal calls / hype first; when the queue is full the lowest
priority (then oldest) item is dropped and reported "failed" so the client falls
back to the browser voice for that line. Lines waiting longer than
MAX_AGE_S are dropped the same way (they are already behind the playhead).

If the worker is down and backend/tts_worker/venv exists, the first probe starts
it in the background (set NHL_TTS_AUTOSTART=0 to disable).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
_WORKER_DIR = _BACKEND / "tts_worker"
CACHE_DIR = Path(os.environ.get("NHL_TTS_CACHE") or (_BACKEND / "data" / "tts_cache"))
WORKER_URL = (os.environ.get("NHL_TTS_URL") or os.environ.get("LIVE_TTS_WORKER_URL")
              or "http://127.0.0.1:8765").rstrip("/")
SETUP_HINT = "Run backend/start_tts.ps1"

PROBE_TTL_S = 5.0
PROBE_TIMEOUT_S = 0.6
SYNTH_TIMEOUT_S = 30.0
MEM_CACHE_MAX = 300
DISK_CACHE_MAX_FILES = 4000
QUEUE_MAX = 32
MAX_AGE_S = 90.0
FAILED_TTL_S = 60.0
N_THREADS = 2

def _load_prosody() -> Any:
    """Shared stdlib prosody module (tts_worker/prosody.py), loaded by path so the
    worker folder never lands on sys.path. Same speed -> same cache key as the worker."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("nhl_tts_prosody", str(_WORKER_DIR / "prosody.py"))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["nhl_tts_prosody"] = mod   # dataclasses need the module registered
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # pragma: no cover - keep the backend alive whatever happens
        return None


_prosody = _load_prosody()

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
_lock = threading.RLock()
_cv = threading.Condition(_lock)

_health: Dict[str, Any] = {"t": 0.0, "data": None, "probing": False, "err": None}
_mem: "OrderedDict[str, Tuple[bytes, str]]" = OrderedDict()
_queue: List[Dict[str, Any]] = []            # pending jobs (small; linear scans are fine)
_queued: Dict[str, Dict[str, Any]] = {}      # key -> job (queued or in flight)
_failed: Dict[str, Tuple[float, str]] = {}   # key -> (time, reason)
_threads: List[threading.Thread] = []
_seq = [0]
_stats = {"done": 0, "failed": 0, "dropped": 0, "stale": 0, "synth_ms": 0.0}
_autostart = {"t": 0.0, "proc": None}


# --------------------------------------------------------------------------
# worker probe


def _get_json(path: str, timeout: float) -> Dict[str, Any]:
    with _opener.open(WORKER_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def _probe(force: bool = False) -> Optional[Dict[str, Any]]:
    """Cached /health. Only one thread probes at a time; others use the last value."""
    now = time.time()
    with _lock:
        fresh = now - _health["t"] < PROBE_TTL_S
        if (fresh and not force) or _health["probing"]:
            return _health["data"]
        _health["probing"] = True
    data, err = None, None
    try:
        data = _get_json("/health", PROBE_TIMEOUT_S)
        if not isinstance(data, dict) or not data.get("ok"):
            data, err = None, "bad_health"
    except Exception as e:
        err = type(e).__name__
    with _lock:
        _health.update(t=time.time(), data=data, probing=False, err=err)
    if data is None:
        _maybe_autostart()
    return data


def _maybe_autostart() -> None:
    if os.environ.get("NHL_TTS_AUTOSTART", "1") == "0":
        return
    py = _WORKER_DIR / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        return
    with _lock:
        p = _autostart["proc"]
        if p is not None and p.poll() is None:
            return                                    # already starting / running
        if time.time() - _autostart["t"] < 60.0:
            return
        _autostart["t"] = time.time()
    try:
        logs = _BACKEND / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / "tts_worker.log", "ab")
        kw: Dict[str, Any] = {"cwd": str(_WORKER_DIR), "stdout": out, "stderr": subprocess.STDOUT,
                              "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED|NEW_GROUP|NO_WINDOW
        else:
            kw["start_new_session"] = True
        proc = subprocess.Popen([str(py), str(_WORKER_DIR / "worker.py")], **kw)
        with _lock:
            _autostart["proc"] = proc
    except Exception:
        pass


def tts_available() -> bool:
    h = _probe()
    return bool(h and h.get("ready") and h.get("languages"))


# --------------------------------------------------------------------------
# keys / caches


def _norm_lang(lang: Any) -> str:
    return str(lang or "en").strip().lower()[:2] or "en"


def _norm_speaker(speaker: Any) -> str:
    return "color" if str(speaker or "").strip().lower() == "color" else "pbp"


def _clean(text: Any) -> str:
    if _prosody is not None:
        return _prosody.clean_text(text)
    return " ".join(str(text or "").split())[:600]


def _speed(profile: Any, hint: Any, speaker: str) -> float:
    if _prosody is not None:
        return _prosody.speed_for(profile, hint, speaker)
    return {"hype": 1.2, "build": 1.1, "somber": 0.95}.get(str(profile), 1.05)


def _norm_profile(profile: Any) -> str:
    p = str(profile or "calm").strip().lower()
    return p if p in ("hype", "build", "calm", "somber") else "calm"


def _priority(profile: str, hint: Any, text: str, lang: str, speaker: str) -> int:
    if _prosody is not None:
        return _prosody.priority_for(profile, hint, text, lang, speaker)
    return {"hype": 80, "build": 60, "somber": 50, "calm": 30}.get(profile, 30)


def audio_key(engine: str, voice: str, text: str, speed: float, profile: str = "", pron: str = "") -> str:
    """hash(engine, voice, text, speed) (+ profile, which sets gain/loudness, and the
    pronunciation-table version so edited names re-render)."""
    raw = f"v1|{engine}|{voice}|{speed:.2f}|{profile}|{pron}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _disk_path(key: str) -> Optional[Path]:
    for ext in (".ogg", ".wav"):
        p = CACHE_DIR / f"{key}{ext}"
        if p.exists():
            return p
    return None


def _mem_put(key: str, data: bytes, mime: str) -> None:
    with _lock:
        _mem[key] = (data, mime)
        _mem.move_to_end(key)
        while len(_mem) > MEM_CACHE_MAX:
            _mem.popitem(last=False)


def _disk_put(key: str, data: bytes, mime: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = CACHE_DIR / f"{key}{'.ogg' if 'ogg' in mime else '.wav'}"
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
    except OSError:
        pass


def _have(key: str) -> bool:
    with _lock:
        if key in _mem:
            return True
    return _disk_path(key) is not None


# --------------------------------------------------------------------------
# queue + threads


def _ensure_threads() -> None:
    with _lock:
        alive = [t for t in _threads if t.is_alive()]
        _threads[:] = alive
        for i in range(N_THREADS - len(alive)):
            t = threading.Thread(target=_run, name=f"live-tts-{len(_threads)}", daemon=True)
            _threads.append(t)
            t.start()
        if not getattr(_ensure_threads, "_pruned", False):
            _ensure_threads._pruned = True  # type: ignore[attr-defined]
            threading.Thread(target=_prune_disk, name="live-tts-prune", daemon=True).start()


def _fail(job: Dict[str, Any], reason: str) -> None:
    _failed[job["key"]] = (time.time(), reason)
    _queued.pop(job["key"], None)


def _enqueue(job: Dict[str, Any]) -> bool:
    """Caller holds _lock. Bounded: evict the lowest-priority (oldest) job if full."""
    if len(_queue) >= QUEUE_MAX:
        victim = min(_queue, key=lambda j: (j["prio"], j["seq"]))
        if victim["prio"] > job["prio"]:
            _failed[job["key"]] = (time.time(), "queue_full")
            _stats["dropped"] += 1
            return False
        _queue.remove(victim)
        _fail(victim, "dropped")
        _stats["dropped"] += 1
    _queue.append(job)
    _queued[job["key"]] = job
    _cv.notify()
    return True


def _take() -> Dict[str, Any]:
    with _cv:
        while True:
            now = time.time()
            for j in [j for j in _queue if now - j["t"] > MAX_AGE_S]:
                _queue.remove(j)
                _fail(j, "stale")
                _stats["stale"] += 1
            if _queue:
                job = max(_queue, key=lambda j: (j["prio"], -j["seq"]))
                _queue.remove(job)
                job["inflight"] = True
                return job
            _cv.wait(timeout=5.0)


def _synth(job: Dict[str, Any]) -> Tuple[bytes, str]:
    body = json.dumps({"text": job["text"], "lang": job["lang"], "speaker": job["speaker"],
                       "profile": job["profile"], "hint": job["hint"], "speed": job["speed"]}).encode("utf-8")
    req = urllib.request.Request(WORKER_URL + "/synth", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=SYNTH_TIMEOUT_S) as r:
        data = r.read()
        mime = r.headers.get("Content-Type", "audio/wav").split(";")[0].strip()
    if not data:
        raise RuntimeError("empty audio")
    return data, mime


def _run() -> None:
    while True:
        job = _take()
        t0 = time.time()
        try:
            data, mime = _synth(job)
        except Exception as e:
            with _lock:
                _fail(job, type(e).__name__)
                _stats["failed"] += 1
                if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
                    _health["t"] = 0.0          # re-probe soon: worker probably went away
            continue
        _mem_put(job["key"], data, mime)
        _disk_put(job["key"], data, mime)
        with _lock:
            _queued.pop(job["key"], None)
            _failed.pop(job["key"], None)
            _stats["done"] += 1
            _stats["synth_ms"] += (time.time() - t0) * 1000.0


def _prune_disk() -> None:
    try:
        files = sorted(CACHE_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - DISK_CACHE_MAX_FILES)]:
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


# --------------------------------------------------------------------------
# public API


def request_clip(text: Any, lang: Any = "en", speaker: Any = "pbp", profile: Any = "calm",
                 hint: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Queue synthesis of one line (non-blocking). Returns the audio key, or None when
    neural TTS is unavailable for this language (caller keeps the browser voice)."""
    t = _clean(text)
    if not t:
        return None
    h = _probe()
    if not h or not h.get("ready"):
        return None
    lg, sp, prof = _norm_lang(lang), _norm_speaker(speaker), _norm_profile(profile)
    v = ((h.get("voices") or {}).get(lg) or {}).get(sp)
    if lg not in (h.get("languages") or []) or not v:
        return None
    hint = hint if isinstance(hint, dict) else None
    speed = _speed(prof, hint, sp)
    key = audio_key(str(v.get("engine")), str(v.get("voice")) + str(v.get("timbre_st", "")), t, speed, prof,
                    str(h.get("pron_version", "")))
    with _lock:
        if key in _mem or key in _queued:
            return key
        ft = _failed.get(key)
        if ft and time.time() - ft[0] < FAILED_TTL_S and ft[1] not in ("stale", "dropped", "queue_full"):
            return key                              # recent hard failure: don't hammer the worker
    if _disk_path(key) is not None:
        return key
    with _lock:
        _failed.pop(key, None)
        _seq[0] += 1
        job = {"key": key, "text": t, "lang": lg, "speaker": sp, "profile": prof, "hint": hint,
               "speed": speed, "prio": _priority(prof, hint, t, lg, sp), "seq": _seq[0], "t": time.time()}
        _enqueue(job)
    _ensure_threads()
    return key


def clip_status(key: str) -> str:
    if not key:
        return "failed"
    with _lock:
        if key in _mem:
            return "ready"
        if key in _queued:
            return "pending"
        if key in _failed:
            return "failed"
    return "ready" if _disk_path(key) is not None else "failed"


def clip_bytes(key: str) -> Optional[Tuple[bytes, str]]:
    if not key or any(c not in "0123456789abcdef" for c in key):
        return None
    with _lock:
        hit = _mem.get(key)
        if hit is not None:
            _mem.move_to_end(key)
            return hit
    p = _disk_path(key)
    if p is None:
        return None
    try:
        data = p.read_bytes()
    except OSError:
        return None
    mime = "audio/ogg" if p.suffix == ".ogg" else "audio/wav"
    _mem_put(key, data, mime)
    return data, mime


def status() -> Dict[str, Any]:
    h = _probe()
    up = bool(h and h.get("ready") and h.get("languages"))
    with _lock:
        qn = len(_queue)
        inflight = sum(1 for j in _queued.values() if j.get("inflight"))
        st = dict(_stats)
        mem_n = len(_mem)
    engine = (h or {}).get("engine") if up else None
    voices = {lang: {sp: v.get("voice") for sp, v in d.items()} for lang, d in ((h or {}).get("voices") or {}).items()} if up else {}
    if up:
        label = f"Neural voices: ON ({str(engine).capitalize()})"
    elif h and h.get("loading"):
        label = "Neural voices: loading..."
    else:
        label = f"Browser voices (robotic) - {SETUP_HINT}"
    return {
        "worker": "up" if up else ("loading" if h and h.get("loading") else "down"),
        "engine": engine,
        "engines": {k: bool(v.get("loaded")) for k, v in ((h or {}).get("engines") or {}).items()},
        "voices": voices,
        "languages": list((h or {}).get("languages") or []) if up else [],
        "format": (h or {}).get("format"),
        "queue": qn,
        "inflight": inflight,
        "cache": {"memory": mem_n, "dir": str(CACHE_DIR)},
        "stats": {**{k: v for k, v in st.items() if k != "synth_ms"},
                  "avg_ms": round(st["synth_ms"] / st["done"]) if st["done"] else None,
                  "worker": (h or {}).get("stats")},
        "label": label,
        "setup_hint": None if up else SETUP_HINT,
        "url": WORKER_URL,
    }
