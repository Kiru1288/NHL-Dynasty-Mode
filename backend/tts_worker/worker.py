#!/usr/bin/env python
"""Local neural TTS worker for the live-game commentary (NHL Dynasty Mode).

A separate process on purpose: espeak-ng / phonemizer (GPL) stay out of the
FastAPI process. Stdlib HTTP server on 127.0.0.1:8765.

  GET  /health         -> {"ok", "ready", "engine", "engines", "voices", "languages", "format", "pron_version"}
  GET  /capabilities   -> legacy shape {"engine", "languages", "voices"} (older backend probe)
  POST /synth          {text, lang, speaker, profile, hint, speed?}
                       -> audio bytes (audio/ogg Opus if available, else audio/wav)
                          headers: X-Engine, X-Voice, X-Speed, X-Duration-Ms, X-Synth-Ms

Engines
  kokoro  Kokoro-82M via kokoro-onnx (onnxruntime, CPU)   en, fr, es, it, pt
  piper   Piper voices via piper-tts, or straight onnxruntime + phonemizer   de, sv, fr (analyst)
  fake    --fake: harmonic test tones, for testing without models

Run:  venv\\Scripts\\python.exe worker.py [--data ..\\data\\tts] [--port 8765] [--fake]
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import os
import sys
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dsp  # noqa: E402
import prosody  # noqa: E402

log = logging.getLogger("tts_worker")
OUT_SR = 24000
VERSION = 1

# --------------------------------------------------------------------------
# voice table (C4). Override per voice with <data>/voices.json, e.g.
#   {"en": {"color": {"engine": "kokoro", "voice": "am_fenrir", "lang": "en-us"}}}
# or env NHL_TTS_VOICE_EN_COLOR=am_fenrir (kokoro voice id) .

DEFAULT_VOICES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "en": {
        # warm, clear American baritone (Kokoro grade C+) -> energetic at speed 1.15-1.25
        "pbp": {"engine": "kokoro", "voice": "am_michael", "lang": "en-us"},
        # older, deeper British timbre (grade C): unmistakably a different person
        "color": {"engine": "kokoro", "voice": "bm_george", "lang": "en-gb"},
    },
    "fr": {
        "pbp": {"engine": "kokoro", "voice": "ff_siwis", "lang": "fr-fr"},   # only French Kokoro voice
        "color": {"engine": "piper", "voice": "fr_FR-gilles-low"},          # male, CC0
    },
    "es": {
        "pbp": {"engine": "kokoro", "voice": "em_alex", "lang": "es"},
        "color": {"engine": "kokoro", "voice": "ef_dora", "lang": "es"},
    },
    "it": {
        "pbp": {"engine": "kokoro", "voice": "im_nicola", "lang": "it"},
        "color": {"engine": "kokoro", "voice": "if_sara", "lang": "it"},
    },
    "pt": {
        "pbp": {"engine": "kokoro", "voice": "pm_alex", "lang": "pt-br"},
        "color": {"engine": "kokoro", "voice": "pf_dora", "lang": "pt-br"},
    },
    "de": {
        "pbp": {"engine": "piper", "voice": "de_DE-thorsten-high"},        # male, CC0
        "color": {"engine": "piper", "voice": "de_DE-kerstin-low"},        # female, CC0
    },
    "sv": {
        "pbp": {"engine": "piper", "voice": "sv_SE-nst-medium"},           # CC0
        # Only one permissively licensed Swedish speaker exists, so the analyst is the
        # same model rendered 2 semitones lower (formants too) and a touch slower.
        "color": {"engine": "piper", "voice": "sv_SE-nst-medium", "timbre_st": -2.0, "speed_mul": 0.97},
    },
}

# Used when a configured voice's model is missing: same language, other engine.
FALLBACK_VOICES: Dict[str, List[Dict[str, Any]]] = {
    "fr": [{"engine": "piper", "voice": "fr_FR-siwis-medium"}],
    "es": [{"engine": "piper", "voice": "es_ES-davefx-medium"}, {"engine": "piper", "voice": "es_MX-ald-medium"}],
    "de": [{"engine": "piper", "voice": "de_DE-thorsten-medium"}],
}

PROFILE_LUFS = {"hype": -15.5, "build": -16.0, "calm": -16.0, "somber": -16.5}


# --------------------------------------------------------------------------
# engines


class FakeEngine:
    """Deterministic harmonic 'speech' (syllable envelope) for tests without models."""
    name = "fake"

    def __init__(self) -> None:
        self.ready = True

    def has_voice(self, voice: str) -> bool:
        return True

    def synth(self, parts: List[Tuple[str, str]], voice: str, lang: str, speed: float,
              speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        text = prosody.parts_to_text(parts)
        sr = 22050 if voice.startswith(("de_", "sv_", "fr_FR")) else 24000
        f0 = 105.0 if voice in ("bm_george", "fr_FR-gilles-low", "de_DE-kerstin-low") else 135.0
        sylls = max(1, int(len([c for c in text if c.isalpha()]) / 3.2))
        dur = sylls * 0.19 / max(0.5, speed)
        t = np.arange(int(dur * sr)) / sr
        f = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * 0.7 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        sig = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 12))
        env = np.clip(np.sin(np.pi * (t * sylls / max(dur, 1e-3)) % np.pi), 0, 1) ** 0.6
        # model-style edge silence that the trimmer must remove
        out = np.concatenate([np.zeros(int(0.12 * sr)), sig * env * 0.25, np.zeros(int(0.2 * sr))])
        time.sleep(0.01)
        return out.astype(np.float32), sr


class KokoroEngine:
    name = "kokoro"

    def __init__(self, data_dir: Path) -> None:
        from kokoro_onnx import Kokoro  # type: ignore
        model = None
        for cand in ("kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx"):
            if (data_dir / cand).exists():
                model = data_dir / cand
                break
        voices = data_dir / "voices-v1.0.bin"
        if model is None or not voices.exists():
            raise FileNotFoundError(f"Kokoro model/voices not found in {data_dir}")
        self.model_file = model.name
        self.k = Kokoro(str(model), str(voices))
        self.voices = set(self.k.get_voices())
        sig = inspect.signature(self.k.create).parameters
        self._has_pauses = "clause_pause" in sig
        self._tok = getattr(self.k, "tokenizer", None)
        self.ready = True

    def has_voice(self, voice: str) -> bool:
        return voice in self.voices

    def synth(self, parts: List[Tuple[str, str]], voice: str, lang: str, speed: float,
              speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        speed = float(min(2.0, max(0.5, speed)))
        kw: Dict[str, Any] = {}
        if self._has_pauses:
            kw.update(sentence_pause=0.0, clause_pause=0.12)
        has_ipa = any(k == "ipa" for k, _ in parts)
        if has_ipa and self._tok is not None:
            ph = []
            for k, s in parts:
                if k == "ipa":
                    ph.append(s)
                elif s.strip():
                    ph.append(self._tok.phonemize(s, lang))
            phonemes = " ".join(p for p in ph if p)
            audio, sr = self.k.create(phonemes, voice=voice, speed=speed, is_phonemes=True, **kw)
        else:
            text = prosody.parts_to_text(parts)
            audio, sr = self.k.create(text, voice=voice, speed=speed, lang=lang, **kw)
        return np.asarray(audio, dtype=np.float32), int(sr)


class _OnnxPiperVoice:
    """Piper VITS voice run directly with onnxruntime; phonemes from phonemizer/espeak-ng
    (espeak-ng shared library from espeakng-loader, installed with kokoro-onnx)."""

    def __init__(self, model: Path) -> None:
        import onnxruntime as ort  # type: ignore
        with open(str(model) + ".json", "r", encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        self.sess = ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])
        self.sr = int(self.cfg["audio"]["sample_rate"])
        self.idmap: Dict[str, List[int]] = self.cfg["phoneme_id_map"]
        self.inf = self.cfg.get("inference", {})
        self.num_speakers = int(self.cfg.get("num_speakers", 1))
        self.espeak_voice = (self.cfg.get("espeak") or {}).get("voice", "en-us")
        self._backend = None

    def _phonemize(self, text: str) -> str:
        if self._backend is None:
            from phonemizer.backend import EspeakBackend  # type: ignore
            try:
                import espeakng_loader  # type: ignore
                from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                if hasattr(EspeakWrapper, "set_data_path"):
                    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            except Exception:
                pass
            self._backend = EspeakBackend(self.espeak_voice, preserve_punctuation=True,
                                          with_stress=True, language_switch="remove-flags")
        return self._backend.phonemize([text], strip=True)[0]

    def ids(self, phonemes: str) -> List[int]:
        m = self.idmap
        out = list(m["^"]) + list(m["_"])
        for ch in unicodedata.normalize("NFD", phonemes):
            if ch in m:
                out += m[ch] + m["_"]
        return out + list(m["$"])

    def synth(self, parts: List[Tuple[str, str]], speed: float, speaker_id: Optional[int]) -> np.ndarray:
        ph = " ".join(s if k == "ipa" else self._phonemize(s) for k, s in parts if s.strip())
        ids = self.ids(ph)
        args = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([float(self.inf.get("noise_scale", 0.667)), 1.0 / max(0.5, speed),
                                float(self.inf.get("noise_w", 0.8))], dtype=np.float32),
        }
        if self.num_speakers > 1:
            args["sid"] = np.array([speaker_id or 0], dtype=np.int64)
        return np.asarray(self.sess.run(None, args)[0], dtype=np.float32).squeeze()


class PiperEngine:
    name = "piper"

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "piper"
        self._voices: Dict[str, Any] = {}
        self._lock = threading.Lock()
        try:
            from piper import PiperVoice, SynthesisConfig  # type: ignore  # piper-tts >= 1.3
            self._PiperVoice, self._SynCfg = PiperVoice, SynthesisConfig
            self.backend = "piper-tts"
        except Exception:
            self._PiperVoice = self._SynCfg = None
            import onnxruntime  # noqa: F401  (raises if missing)
            import phonemizer  # type: ignore  # noqa: F401
            self.backend = "onnxruntime"
        self.available = sorted(p.name[:-5] for p in self.dir.glob("*.onnx")) if self.dir.exists() else []
        if not self.available:
            raise FileNotFoundError(f"no Piper voices in {self.dir}")
        self.ready = True

    def has_voice(self, voice: str) -> bool:
        return voice in self.available

    def _load(self, voice: str) -> Any:
        with self._lock:
            v = self._voices.get(voice)
            if v is None:
                path = self.dir / f"{voice}.onnx"
                v = self._PiperVoice.load(str(path)) if self._PiperVoice else _OnnxPiperVoice(path)
                self._voices[voice] = v
            return v

    def preload(self, voice: str) -> None:
        if self.has_voice(voice):
            self._load(voice)

    def synth(self, parts: List[Tuple[str, str]], voice: str, lang: str, speed: float,
              speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        v = self._load(voice)
        if self._PiperVoice is not None:
            # piper-tts takes raw phonemes inline as [[ ... ]]
            text = "".join(f" [[ {s} ]] " if k == "ipa" else s for k, s in parts)
            cfg = self._SynCfg(length_scale=1.0 / max(0.5, speed), speaker_id=speaker_id)
            chunks = [c.audio_float_array for c in v.synthesize(text, syn_config=cfg)]
            sr = int(v.config.sample_rate)
            audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
            return audio.astype(np.float32), sr
        return v.synth(parts, speed, speaker_id), v.sr


# --------------------------------------------------------------------------
# the worker


class TTSWorker:
    def __init__(self, data_dir: Path, fake: bool = False) -> None:
        self.data_dir = data_dir
        self.fake = fake
        self.engines: Dict[str, Any] = {}
        self.engine_errors: Dict[str, str] = {}
        self.voice_table: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.ready = False
        self.loading = True
        self.lock = threading.Lock()          # one synthesis at a time: CPU-bound, espeak is global
        self.stats = {"clips": 0, "synth_ms_total": 0.0, "audio_ms_total": 0.0, "errors": 0}

    # ---- setup
    def load(self) -> None:
        t0 = time.time()
        if self.fake:
            self.engines["kokoro"] = FakeEngine()
            self.engines["piper"] = FakeEngine()
        else:
            for name, cls in (("kokoro", KokoroEngine), ("piper", PiperEngine)):
                try:
                    self.engines[name] = cls(self.data_dir)
                    log.info("engine %s loaded", name)
                except Exception as e:  # missing package or files
                    self.engine_errors[name] = f"{type(e).__name__}: {e}"[:300]
                    log.warning("engine %s unavailable: %s", name, self.engine_errors[name])
        self.voice_table = self._resolve_voices()
        # preload Piper voices + warm every voice once so the first real line is fast
        for lang, spk in self.voice_table.items():
            for sp, v in spk.items():
                try:
                    self._render("Ready.", lang, sp, "calm", None)
                except Exception as e:
                    log.warning("warm-up %s/%s failed: %s", lang, sp, e)
        self.ready = bool(self.voice_table)
        self.loading = False
        log.info("ready in %.1fs: %s", time.time() - t0, {k: {s: v["voice"] for s, v in d.items()} for k, d in self.voice_table.items()})

    def _overrides(self) -> Dict[str, Any]:
        table = copy.deepcopy(DEFAULT_VOICES)
        f = self.data_dir / "voices.json"
        if f.exists():
            try:
                user = json.loads(f.read_text(encoding="utf-8"))
                for lang, d in user.items():
                    if isinstance(d, dict):
                        table.setdefault(lang, {}).update(d)
            except Exception as e:
                log.warning("voices.json ignored: %s", e)
        for lang in list(table):
            for sp in ("pbp", "color"):
                env = os.environ.get(f"NHL_TTS_VOICE_{lang.upper()}_{sp.upper()}")
                if env:
                    cur = dict(table[lang].get(sp, {}))
                    cur["voice"] = env
                    cur["engine"] = "piper" if "-" in env else "kokoro"
                    table[lang][sp] = cur
        return table

    def _usable(self, v: Dict[str, Any]) -> bool:
        e = self.engines.get(v.get("engine", ""))
        return bool(e) and e.has_voice(v.get("voice", ""))

    def _resolve_voices(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for lang, spk in self._overrides().items():
            res: Dict[str, Dict[str, Any]] = {}
            for sp in ("pbp", "color"):
                v = spk.get(sp)
                if v and self._usable(v):
                    res[sp] = dict(v)
                    continue
                for fb in FALLBACK_VOICES.get(lang, []):
                    if self._usable(fb):
                        res[sp] = dict(fb)
                        break
            # one speaker missing -> reuse the other with a timbre shift so they still differ
            if "pbp" in res and "color" not in res:
                res["color"] = dict(res["pbp"], timbre_st=-2.0, speed_mul=0.97, derived=True)
            elif "color" in res and "pbp" not in res:
                res["pbp"] = dict(res["color"], timbre_st=+1.5, derived=True)
            if res:
                out[lang] = res
        return out

    # ---- synthesis
    def _render(self, text: str, lang: str, speaker: str, profile: str, hint: Any,
                speed: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        v = self.voice_table[lang][speaker if speaker in ("pbp", "color") else "pbp"]
        eng = self.engines[v["engine"]]
        base_speed = float(speed) if speed else prosody.speed_for(profile, hint, speaker)
        base_speed *= float(v.get("speed_mul", 1.0))
        timbre = float(v.get("timbre_st", 0.0))
        segs = prosody.plan_segments(text, lang, profile, hint, speaker, speed=base_speed)
        allow_ipa = v["engine"] in ("kokoro", "piper")
        pieces: List[np.ndarray] = [dsp.silence(30, OUT_SR)]
        for seg in segs:
            parts = prosody.apply_pronunciations(seg.text, lang, allow_ipa=allow_ipa)
            st = seg.pitch_st + timbre
            # pitch-shifting by resampling scales duration by 2^(-st/12): pre-compensate
            syn_speed = seg.speed * (2.0 ** (-st / 12.0))
            audio, sr = eng.synth(parts, v["voice"], v.get("lang", lang), syn_speed, v.get("speaker_id"))
            audio = dsp.trim_silence(audio, sr)
            if sr != OUT_SR:
                audio = dsp.resample(audio, sr, OUT_SR)
            if abs(st) > 1e-3:
                audio = dsp.pitch_shift(audio, st)
            if seg.gain_db:
                audio = audio * dsp.db_to_lin(seg.gain_db)
            pieces.append(dsp.fade(audio, OUT_SR, 4.0, 8.0))
            if seg.pause_after_ms:
                pieces.append(dsp.silence(seg.pause_after_ms, OUT_SR))
        pieces.append(dsp.silence(60, OUT_SR))
        x = np.concatenate(pieces)
        x = dsp.master(x, OUT_SR, target_lufs=PROFILE_LUFS.get(prosody.norm_profile(profile), -16.0))
        meta = {"engine": v["engine"], "voice": v["voice"] + (f"{timbre:+.0f}st" if timbre else ""),
                "speed": round(base_speed, 2), "segments": len(segs)}
        return x, meta

    def synth(self, req: Dict[str, Any]) -> Tuple[bytes, str, Dict[str, Any]]:
        text = prosody.clean_text(req.get("text"))
        lang = str(req.get("lang") or "en").lower()[:2]
        speaker = "color" if str(req.get("speaker") or "pbp").lower() == "color" else "pbp"
        profile = prosody.norm_profile(req.get("profile"))
        hint = req.get("hint") if isinstance(req.get("hint"), dict) else None
        speed = req.get("speed")
        if not text:
            raise ValueError("empty_text")
        if lang not in self.voice_table:
            raise LookupError(lang)
        t0 = time.time()
        with self.lock:
            x, meta = self._render(text, lang, speaker, profile, hint, float(speed) if speed else None)
        data, mime = dsp.encode(x, OUT_SR, prefer=str(req.get("format") or "ogg"))
        ms = (time.time() - t0) * 1000.0
        meta.update(synth_ms=round(ms), duration_ms=round(len(x) * 1000.0 / OUT_SR), mime=mime)
        self.stats["clips"] += 1
        self.stats["synth_ms_total"] += ms
        self.stats["audio_ms_total"] += meta["duration_ms"]
        return data, mime, meta

    def health(self) -> Dict[str, Any]:
        engines = {}
        for name in ("kokoro", "piper"):
            e = self.engines.get(name)
            if e is not None:
                info: Dict[str, Any] = {"loaded": True}
                if isinstance(e, KokoroEngine):
                    info["model"] = e.model_file
                if isinstance(e, PiperEngine):
                    info.update(backend=e.backend, voices=e.available)
                if isinstance(e, FakeEngine):
                    info["fake"] = True
                engines[name] = info
            elif name in self.engine_errors:
                engines[name] = {"loaded": False, "error": self.engine_errors[name]}
        used = [v["engine"] for d in self.voice_table.values() for v in d.values()]
        primary = "kokoro" if "kokoro" in used else (used[0] if used else "none")
        if self.fake:
            primary = "fake"
        st = self.stats
        return {
            "ok": True, "version": VERSION, "ready": self.ready, "loading": self.loading,
            "engine": primary, "engines": engines,
            "voices": {lang: {sp: {"engine": v["engine"], "voice": v["voice"],
                                   **({"timbre_st": v["timbre_st"]} if v.get("timbre_st") else {})}
                              for sp, v in d.items()} for lang, d in self.voice_table.items()},
            "languages": sorted(self.voice_table) if self.ready else [],
            "format": "ogg" if dsp.opus_available() else "wav",
            "pron_version": prosody.pron_version(),
            "stats": {"clips": st["clips"], "avg_synth_ms": round(st["synth_ms_total"] / st["clips"]) if st["clips"] else None,
                      "rtf": round(st["synth_ms_total"] / st["audio_ms_total"], 3) if st["audio_ms_total"] else None,
                      "errors": st["errors"]},
        }


def make_handler(worker: TTSWorker):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NHLTTS/1"

        def _json(self, code: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                self._json(200, worker.health())
            elif self.path.startswith("/capabilities"):
                h = worker.health()
                self._json(200, {"engine": h["engine"] if h["ready"] else "none", "languages": h["languages"],
                                 "gpu": False, "voices": sorted({v["voice"] for d in h["voices"].values() for v in d.values()})})
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/synth"):
                self._json(404, {"error": "not_found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                if n > 32768:
                    self._json(413, {"error": "too_large"})
                    return
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._json(400, {"error": "bad_json"})
                return
            if not worker.ready:
                self._json(503, {"error": "loading" if worker.loading else "no_engine"})
                return
            try:
                data, mime, meta = worker.synth(req)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            except LookupError as e:
                self._json(503, {"error": "no_engine_for_lang", "lang": str(e)})
                return
            except Exception as e:
                worker.stats["errors"] += 1
                log.exception("synth failed")
                self._json(500, {"error": "synth_failed", "detail": f"{type(e).__name__}: {e}"[:300]})
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Engine", meta["engine"])
            self.send_header("X-Voice", meta["voice"])
            self.send_header("X-Speed", str(meta["speed"]))
            self.send_header("X-Duration-Ms", str(meta["duration_ms"]))
            self.send_header("X-Synth-Ms", str(meta["synth_ms"]))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("%s " + fmt, self.address_string(), *args)

    return Handler


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NHL Dynasty live commentary TTS worker")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("NHL_TTS_PORT", "8765")))
    ap.add_argument("--data", default=os.environ.get("NHL_TTS_DATA") or str(HERE.parent / "data" / "tts"))
    ap.add_argument("--fake", action="store_true", help="test engine (no models needed)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    worker = TTSWorker(Path(args.data).resolve(), fake=args.fake)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(worker))
    srv.daemon_threads = True
    threading.Thread(target=worker.load, name="tts-load", daemon=True).start()
    log.info("TTS worker listening on http://%s:%d (data=%s)", args.host, args.port, args.data)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
