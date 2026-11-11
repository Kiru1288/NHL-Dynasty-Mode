# Neural commentary voices (TTS worker)

The live sim's commentary is spoken by a local, offline, free neural TTS instead of
the browser's robotic `speechSynthesis`. It runs as a separate process
(`tts_worker/worker.py`, `http://127.0.0.1:8765`). That keeps the GPL espeak-ng and
phonemizer code out of the FastAPI process. The backend talks to it through
`services/live_tts.py`. No cloud, no paid APIs, no voice cloning.

## Setup (Windows, one time)

```powershell
cd backend
.\start_tts.ps1          # creates tts_worker\venv, installs packages, downloads models, starts the worker, prints status
.\start_api.ps1          # start the API as usual; it finds the worker by itself
```

- Re-running `start_tts.ps1` is safe. Finished steps are skipped, and a running worker is left alone.
- `.\start_all.ps1` runs both steps in one go (`start_api.ps1` itself is unchanged).
- Once the venv exists, the API also starts the worker in the background if it isn't running (`NHL_TTS_AUTOSTART=0` turns that off). In practice: run `start_tts.ps1` once, then start the API as normal.
- Options: `-Restart`, `-Stop`, `-Foreground` (see the log live), `-Int8` (smaller ~90 MB Kokoro model, slightly lower quality), `-ExtraVoices` (fallback Piper voices), `-Python <path>`.
- Python: the worker needs **Python 3.10–3.13** (kokoro-onnx doesn't support 3.14 yet). The script prefers `py -3.12`, then 3.11, 3.13, 3.10. If none is installed it tells you to run `winget install -e --id Python.Python.3.12`. The game API can stay on 3.14.
- If PowerShell blocks scripts: `powershell -ExecutionPolicy Bypass -File .\start_tts.ps1`
- Disk: ~310 MB Kokoro model + 27 MB voices + ~250 MB Piper voices in `backend/data/tts/`. Clip cache: `backend/data/tts_cache/`. Both are git-ignored.
- Logs: `backend/logs/tts_worker.log`. Status: `http://127.0.0.1:8765/health`.

## Voices

| lang | play-by-play (energetic) | analyst (different timbre) |
|---|---|---|
| en | Kokoro `am_michael`: warm American baritone, the most natural US male voice (Kokoro grade C+) | Kokoro `bm_george`: older, deeper British voice (grade C), clearly a different person |
| fr | Kokoro `ff_siwis` (the only French Kokoro voice) | Piper `fr_FR-gilles-low`, male |
| es | Kokoro `em_alex`, male | Kokoro `ef_dora`, female |
| it | Kokoro `im_nicola` | Kokoro `if_sara` |
| pt | Kokoro `pm_alex` (pt-BR) | Kokoro `pf_dora` |
| de | Piper `de_DE-thorsten-high`, male | Piper `de_DE-kerstin-low`, female |
| sv | Piper `sv_SE-nst-medium` | the same model 2 semitones lower (pitch and formants) and 3 % slower |

There is only one permissively licensed Swedish speaker (sv `alma` is trained on the same NST speaker, and `lisa` has no stated licence). That's why the Swedish analyst is a re-pitched variant.

Change a voice in `backend/data/tts/voices.json`, then run `start_tts.ps1 -Restart`:

```json
{ "en": { "color": { "engine": "kokoro", "voice": "am_fenrir", "lang": "en-us" } } }
```

You can also set an environment variable such as `NHL_TTS_VOICE_EN_COLOR=am_fenrir`. Other Kokoro male options: `am_puck` (brighter), `am_fenrir` (deep), `bm_fable`, `bm_lewis`, `am_eric`.

## Expressiveness

- **Speed** comes from profile and hint: hype 1.15–1.25, build 1.06–1.14, calm 1.02–1.08, somber 0.92–0.98. `energy` shifts it by ±0.05 and `pace` by +0.03/−0.04 within the band. The analyst is 0.03 slower.
- **Pauses**: each sentence is synthesised separately, with 120–350 ms of silence between sentences (shorter when fast or hype, longer when somber). Sentences over 16 words are also split at commas (≈150 ms). Shorter commas are left to the model's own phrasing.
- **Goal calls**: a leading call ("Scores!", "He scores!", "But !", "¡Gol!", "Tor!", "Mål!") becomes its own segment. It's rendered faster (+0.08), louder (+2…4 dB) and 1–2 semitones brighter, followed by a ~160 ms beat, then the rest of the line. In hype lines the excitement decays sentence by sentence, and in build lines it rises.
- **Post-processing** (numpy only): 80 Hz high-pass, −1.5 dB at 300 Hz, +2.5 dB presence at 3.2 kHz, a little air at 8 kHz, then BS.1770 loudness normalisation to −16 LUFS (hype −15.5, somber −16.5), 2.5:1 compression, re-normalisation and a −1 dBFS look-ahead peak limiter. Every clip sits at the same level over the crowd bed.
- **Output**: OGG/Opus at 24 kHz (~15 KB per line) when libsndfile supports it, otherwise 16-bit WAV.

## Name pronunciation

Overrides live in `tts_worker/pronunciations.json` (name → respelling, optionally IPA). The worker reloads the file when it changes, and the clip cache key includes the table's version. For example `"Stützle": "Stoots-lee"`, `"Ullmark": "Ool-mark"`, `"Zub": "Zoob"`, `"Lafrenière": "La-fren-yair"`. Use `{"say": "...", "ipa": "ʃtˈʏtslə", "langs": ["en"]}` to give exact phonemes (Kokoro and piper-tts accept them). Unlisted names with ö/ä/å/ü/č/š/ž/ř/é get a simple letter fallback in English commentary. In other languages the native espeak rules are used.

## HTTP API (worker)

- `GET /health` → `{ok, ready, loading, engine, engines, voices{lang:{pbp,color}}, languages, format, pron_version, stats}`
- `GET /capabilities` → legacy `{engine, languages, voices}`
- `POST /synth {text, lang, speaker: pbp|color, profile: hype|build|calm|somber, hint: {energy, pace, pause_before_ms}, speed?}` → audio bytes, with the headers `X-Engine`, `X-Voice`, `X-Speed`, `X-Duration-Ms` and `X-Synth-Ms`. `pause_before_ms` is not baked in; the player applies it.

## Licences (C11)

| component | licence | notes |
|---|---|---|
| Kokoro-82M weights + voices (hexgrad) | **Apache-2.0** | downloaded from the kokoro-onnx GitHub releases |
| kokoro-onnx | **MIT** | |
| onnxruntime | MIT | |
| Piper (rhasspy/piper) | **MIT** | voices and model format |
| piper-tts 1.3+ (OHF-Voice/piper1-gpl) | GPL-3.0 | optional, worker process only; without it a built-in onnxruntime route is used |
| espeak-ng (espeakng-loader), phonemizer | GPL-3.0 | worker process only, never imported by the API |
| numpy / soundfile | BSD (libsndfile LGPL) | |
| Piper voice `de_DE-thorsten-high` | **CC0** | Thorsten-Voice dataset |
| Piper voice `de_DE-kerstin-low` | **CC0** | |
| Piper voice `sv_SE-nst-medium` | **CC0** | NST dataset (Språkbanken), trained by KBLab |
| Piper voice `fr_FR-gilles-low` | **CC0** | |
| `fr_FR-siwis-medium` (only with `-ExtraVoices`) | CC-BY-4.0 | attribution: SIWIS database |
| `es_ES-davefx-medium` (only with `-ExtraVoices`) | CC0 | |

Licences are taken from each voice's `MODEL_CARD` in rhasspy/piper-voices. Deliberately **not** used: `fr_FR-tom` (AGPL-3.0), `fr_FR-upmc` (CC-BY-SA-4.0), `de_DE-pavoque` (CC-BY-NC-SA-4.0, non-commercial), `sv_SE-lisa` (no licence stated).

## Testing without models

`python worker.py --fake --port 8799` serves harmonic test tones through the full pipeline (segmenting, pauses, DSP, encoding, HTTP). It's useful for checking `services/live_tts.py` without downloading anything.
