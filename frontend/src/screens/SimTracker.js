import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { teamPalette } from "../components/boxscore/BoxScoreArcade";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { baseURL, getFranchiseSessionId } from "../services/api";
import {
  liveAbort,
  liveAdvance,
  liveCommit,
  liveCoach,
  livePause,
  liveResume,
  liveStart,
  liveState,
  liveTtsStatus,
  liveTtsUrl,
} from "../services/franchiseService";

/* ==========================================================================
   Live game sim screen (contract v2 + v3 presentation time).
   The server simulates ahead and returns a dense motion track; this screen
   buffers it and plays it back on its own requestAnimationFrame clock. The
   playback head (not the server) drives the clock, score, plays, commentary
   and alerts the viewer sees. Markers are moved through refs, never through
   per-frame React state. If the server sends no `motion`, playback falls
   back to the server clock and whatever tracking frames the snapshot has.
   ========================================================================== */

const SPEEDS = [1, 2, 4, 8];
// Broadcast pace slows the on-ice motion (the game clock slows with it) so plays are readable; Real time = 1:1.
const PACE_KEY = "nhl.live.pace";
const PACES = { broadcast: 0.6, real: 1 };
// Buffer kept ahead of the playback head (presentation seconds) and chunk size asked per /advance
// (game seconds). Both grow with speed so 4x / 8x never starve (F2).
const bufferMin = (speed) => Math.max(8, 3 * speed);
const chunkSec = (speed) => Math.max(16, 8 * speed);
const FETCH_GAP_MS = 150;
const LABEL_LAYOUT_MS = 100; // label collision pass at ~10 Hz (F1)
const CLIP_WAIT_MS = 1500; // neural clip must be ready this soon after reveal, else browser voice
const REPLAY_RATE = 0.5;
// Spoken commentary uses the browser's speech voices (or neural clips when the local voice worker is running).
const BROWSER_VOICE = true;
const STOP_PHASES = new Set(["stoppage", "offside", "icing", "penalty", "goal", "intermission"]);
const STOP_LABEL = {
  offside: "Offside",
  icing: "Icing",
  frozen: "Frozen puck",
  out_of_play: "Out of play",
  // Backend reason codes (live_motion.py)
  freeze: "Frozen puck",
  out: "Out of play",
  glass: "Over the glass",
  waved_off: "Icing waved off",
  goal: "Goal",
  penalty: "Penalty",
  hand_pass: "Hand pass",
  high_stick: "High stick",
  intermission: "End of period",
  opening: "Opening faceoff",
};
const LANG_KEY = "nhl.live.commentaryLang";
const SOUND_KEY = "nhl.live.sound";
const FALLBACK_LANGS = [
  { code: "en", label: "English" },
  { code: "fr", label: "Français" },
  { code: "es", label: "Español" },
  { code: "de", label: "Deutsch" },
  { code: "sv", label: "Svenska" },
];
const VOICE_PREFS = {
  en: ["en-US", "en-CA", "en-GB"],
  fr: ["fr-CA", "fr-FR"],
  es: ["es-ES", "es-MX"],
  de: ["de-DE"],
  sv: ["sv-SE"],
};
// Broadcast pace: play-by-play callers talk fast. These are multiplied by the user's voice-speed setting.
const PROFILE_VOICE = {
  hype: { rate: 1.3, pitch: 1.14 },
  build: { rate: 1.24, pitch: 1.06 },
  calm: { rate: 1.16, pitch: 1 },
  somber: { rate: 1.02, pitch: 0.92 },
};
// Voices that sound human (Edge "Online (Natural)", Chrome "Google …", macOS premium/enhanced).
const NATURAL_RE = /natural|neural|online|google|premium|enhanced|wavenet/i;
// Words that get a lift when they appear in a phrase — the "call" words of a broadcast.
const EMPH_WORDS = {
  en: /\b(scores?|goal|what a|stopped|denied|robbed|big save|breakaway|empty net|overtime)\b/i,
  fr: /(^but\b|\bcompte\b|quel arrêt|arrêt|échappée|filet désert|prolongation)/i,
  es: /(gol|atajada|paradón|escapada|portería vacía|prórroga)/i,
  de: /(\btor\b|parade|alleingang|leeres tor|verlängerung)/i,
  sv: /(\bmål\b|räddning|friläge|tom kasse|förlängning)/i,
};
function hash01(str) {
  let h = 2166136261;
  const t = String(str);
  for (let i = 0; i < t.length; i += 1) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
  return ((h >>> 0) % 1000) / 1000;
}
const clampN = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
/**
 * Break a line into breath phrases and give each its own rate/pitch so it sounds spoken, not read:
 * call words lift, the opening of a hype call jumps, the last phrase drops into a cadence, and a small
 * per-phrase jitter stops the flat robot rhythm. Each phrase becomes its own utterance (natural pauses).
 */
function prosodyChunks(text, { profile, speaker, lang, rateMul, sameVoice, seed }) {
  const raw = String(text || "")
    .split(/(?<=[.!?;,])\s+/)
    .map((x) => x.trim())
    .filter(Boolean);
  const parts = [];
  for (const seg of raw) {
    if (parts.length && (seg.length < 4 || parts[parts.length - 1].length < 4)) parts[parts.length - 1] += ` ${seg}`;
    else parts.push(seg);
  }
  const base = PROFILE_VOICE[profile] || PROFILE_VOICE.calm;
  const color = speaker === "color";
  const emphRe = EMPH_WORDS[String(lang || "en").split("-")[0]] || EMPH_WORDS.en;
  return parts.map((seg, i) => {
    let rate = base.rate * rateMul * (color ? 0.93 : 1);
    let pitch = base.pitch * (color && sameVoice ? 0.82 : 1);
    let volume = color ? 0.9 : 1;
    const exclaim = /!$/.test(seg);
    if (exclaim || emphRe.test(seg)) {
      pitch *= exclaim ? 1.14 : 1.07;
      rate *= 0.96; // stretch the call word slightly
      volume = 1;
    }
    if (i === 0 && profile === "hype") {
      pitch *= 1.08;
      rate *= 1.05;
    }
    if (i === parts.length - 1 && parts.length > 1 && !exclaim) {
      pitch *= 0.95; // falling cadence at the end of a thought
      rate *= 0.97;
    }
    if (/\b(one|two|three|four|five|six|seven|eight|nine|ten|un|deux|trois|uno|dos|tres|eins|zwei|drei|ett|två|tre)\b/i.test(seg)) {
      rate *= 0.95; // say the score clearly
    }
    const j = hash01(`${seed}:${i}`) - 0.5;
    rate *= 1 + j * 0.06;
    pitch *= 1 - j * 0.06;
    return { text: seg, rate: clampN(rate, 0.6, 2), pitch: clampN(pitch, 0.5, 2), volume };
  });
}
const STALE_MS = 6000;
const UTTER_TIMEOUT_MS = 12000;
const WATCHDOG_MS = 10000;
const QUEUE_CAP = 3;
const MAX_EVENTS = 500;
const MAX_LINES = 120;
const SHOT_MARKERS = 6;
const SHOT_FADE_SEC = 45; // shot markers fade out over this many presentation seconds
const TRAIL_LEN = 16;

const TACTICS = [
  { key: "forecheck", label: "Forecheck", opts: [["passive", "Sit back"], ["balanced", "Balanced"], ["aggressive", "Hard"]], tip: "How hard your forwards chase the puck in the other team's zone." },
  { key: "defense", label: "D posture", opts: [["passive", "Collapse"], ["balanced", "Balanced"], ["aggressive", "Pinch"]], tip: "Collapse protects the slot; Pinch keeps pucks in but risks odd-man rushes." },
  { key: "shot_selection", label: "Shooting", opts: [["passive", "Patient"], ["balanced", "Balanced"], ["aggressive", "Fire away"]], tip: "Patient waits for better looks; Fire away takes more, lower-quality shots." },
];
const SPECIAL = [
  { key: "pp_style", label: "Power play", opts: [["passive", "Set up"], ["aggressive", "Shoot"]], tip: "Set up works the umbrella for a clean look; Shoot throws pucks at the net early." },
  { key: "pk_style", label: "Penalty kill", opts: [["passive", "Box"], ["aggressive", "Pressure"]], tip: "Box protects the middle; Pressure chases the puck carrier." },
];
const KIND_LABEL = {
  roll_lines: "Roll lines",
  shorten_bench: "Shorten bench",
  next_line: "Next line",
  bench_player: "Bench player",
  pull_goalie: "Pull goalie",
  timeout: "Timeout",
  tactics: "Tactics",
  open_lines: "Open lines",
  blender: "Shuffle lines",
};

/* ---------- helpers ---------- */
const num = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
const has = (v) => v !== undefined && v !== null && v !== "";
const upper = (s) => String(s || "").toUpperCase();
const errMsg = (e, fb) => {
  const d = e?.response?.data?.detail;
  if (typeof d === "string") return d;
  return d?.message || e?.message || fb;
};
function mmss(sec) {
  const s = Math.max(0, Math.ceil(num(sec) - 1e-6));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function periodName(p) {
  const n = num(p, 1);
  if (n === 1) return "1st";
  if (n === 2) return "2nd";
  if (n === 3) return "3rd";
  if (n === 4) return "OT";
  if (n === 5) return "SO";
  return `P${n}`;
}
function hexRgb(hex) {
  const h = String(hex || "").replace("#", "").trim();
  const f = h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6);
  const n = parseInt(f, 16);
  if (!/^[0-9a-f]{6}$/i.test(f) || !Number.isFinite(n)) return null;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function luminance(hex) {
  const c = hexRgb(hex);
  if (!c) return 0.3;
  return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
}
/** Weighted RGB distance ("redmean"), 0..~765. */
function colorDist(a, b) {
  const A = hexRgb(a);
  const B = hexRgb(b);
  if (!A || !B) return 999;
  const rm = (A[0] + B[0]) / 2;
  const dr = A[0] - B[0];
  const dg = A[1] - B[1];
  const db = A[2] - B[2];
  return Math.sqrt((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db);
}
function darken(hex, f) {
  const c = hexRgb(hex);
  if (!c) return "#0b1844";
  return `#${c.map((v) => Math.round(v * (1 - f)).toString(16).padStart(2, "0")).join("")}`;
}
const inkOn = (hex) => (luminance(hex) > 0.58 ? "#071233" : "#ffffff");
function teamColors(card) {
  const c = Array.isArray(card?.colors) ? card.colors.filter(Boolean) : [];
  const pal = teamPalette(card || {});
  return { primary: c[0] || pal.primary || "#0b1844", secondary: c[1] || pal.secondary || "#ffe7c4", tertiary: c[2] || null };
}
/**
 * Kits: home wears its dark primary; away wears white trimmed in its primary.
 * If the away trim is too light to read on white, or too close to the home
 * kit, fall back to the away secondary. `id` is the one colour each team uses
 * everywhere in the UI (scorebug, markers, plays, stats, momentum).
 */
function kitColors(teams) {
  const h = teamColors(teams?.home);
  const a = teamColors(teams?.away);
  let homeKit = h.primary;
  if (luminance(homeKit) > 0.55) homeKit = luminance(h.secondary) <= 0.55 ? h.secondary : darken(homeKit, 0.5);
  // Away trim must read on white AND differ from the home kit (DET red vs OTT red -> not red/red).
  const bad = (c) => !hexRgb(c) || luminance(c) > 0.78 || colorDist(c, homeKit) < 150;
  const awayTrim = [a.primary, a.secondary, a.tertiary, darken(a.primary, 0.55), "#0b1844"].find((c) => c && !bad(c)) || "#0b1844";
  // chip = how each team is painted in UI: home = dark kit fill; away = white fill, trim ink + trim outline.
  return {
    home: { ...h, kit: homeKit, kitInk: inkOn(homeKit), trim: "#071233", id: homeKit, idInk: inkOn(homeKit), chip: { "--st-bg": homeKit, "--st-fg": inkOn(homeKit), "--st-line": "#071233" } },
    away: { ...a, kit: "#fbfbf6", kitInk: awayTrim, trim: awayTrim, id: awayTrim, idInk: inkOn(awayTrim), chip: { "--st-bg": "#fbfbf6", "--st-fg": awayTrim, "--st-line": awayTrim } },
  };
}
const otherSide = (s) => (s === "home" ? "away" : s === "away" ? "home" : null);
function readLang() {
  try {
    return window.localStorage.getItem(LANG_KEY) || "en";
  } catch {
    return "en";
  }
}
function writeLang(code) {
  try {
    window.localStorage.setItem(LANG_KEY, code);
  } catch {
    /* storage blocked */
  }
}
function readSoundPrefs() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(SOUND_KEY) || "{}");
    return { crowd: raw.crowd !== false, voice: raw.voice !== false, volume: Number.isFinite(raw.volume) ? raw.volume : 0.8 };
  } catch {
    return { crowd: true, voice: true, volume: 0.8 };
  }
}
function writeSoundPrefs(p) {
  try {
    window.localStorage.setItem(SOUND_KEY, JSON.stringify(p));
  } catch {
    /* storage blocked */
  }
}
function fmtPct(v) {
  if (!has(v)) return null;
  const n = num(v);
  if (n > 1) return `${n.toFixed(1)}%`;
  return n >= 1 ? "1.000" : `.${String(Math.round(n * 1000)).padStart(3, "0")}`;
}
/** Home attacks +x in periods 1 and 3 (contract); flipped in even periods. */
function homeAttacksRight(period) {
  return num(period, 1) % 2 === 1;
}
function netXFor(ev) {
  const x = num(ev?.shot_loc?.x);
  if (Math.abs(x) > 25) return x >= 0 ? 89 : -89;
  const right = homeAttacksRight(ev?.period);
  if (ev?.side === "home") return right ? 89 : -89;
  if (ev?.side === "away") return right ? -89 : 89;
  return x >= 0 ? 89 : -89;
}
/** Clock derived from playback time. OT length comes from the server when it can be worked out. */
function clockFromT(t, otLen = 300) {
  const s = Math.max(0, num(t));
  if (s < 3600) {
    const p = Math.min(3, Math.floor(s / 1200) + 1);
    return { period: p, left: p * 1200 - s };
  }
  return { period: 4, left: Math.max(0, 3600 + otLen - s) };
}
function rankVoices(voices, code) {
  if (!Array.isArray(voices) || !voices.length) return [];
  const raw = String(code || "en").replace("_", "-");
  const base = raw.toLowerCase().split("-")[0];
  const norm = (v) => String(v?.lang || "").replace("_", "-").toLowerCase();
  const tags = [...(raw.includes("-") ? [raw.toLowerCase()] : []), ...(VOICE_PREFS[base] || [base]).map((t) => t.toLowerCase())];
  const score = (v) => {
    const l = norm(v);
    let s = tags.indexOf(l);
    s = s < 0 ? (l.split("-")[0] === base ? 50 : 999) : s * 10;
    if (s >= 999) return 999;
    // A natural/neural voice in any accent of the language beats a robotic desktop voice in the exact locale.
    if (NATURAL_RE.test(v.name || "")) s -= 60;
    return s;
  };
  return voices.filter((v) => score(v) < 999).sort((a, b) => score(a) - score(b));
}
function resolveAudioUrl(u) {
  const s = String(u || "");
  if (!s) return "";
  if (/^(https?:|blob:|data:)/i.test(s)) return s;
  if (s.startsWith("/")) return `${baseURL || ""}${s}`;
  return liveTtsUrl(s);
}
/** Presentation time of a frame / event / line: `pt` when the server sends it, else `t`. */
const ptOf = (o, d = 0) => (has(o?.pt) ? num(o.pt) : has(o?.t) ? num(o.t) : d);
function firstSentence(s, max = 90) {
  const t = String(s || "").split(/(?<=[.!?])\s/)[0] || "";
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}
/** Old-style tracking frames -> v2 motion frames (fallback when `motion` is missing). */
function convertTracking(frames) {
  if (!Array.isArray(frames)) return [];
  return frames
    .filter((f) => f && has(f.t))
    .map((f) => ({
      t: num(f.t),
      ph: "play",
      pk: f.puck && has(f.puck.x) ? [num(f.puck.x), num(f.puck.y)] : null,
      c: null,
      pos: null,
      p: (Array.isArray(f.tokens) ? f.tokens : []).filter((k) => k && has(k.id)).map((k) => [k.id, num(k.x), num(k.y), k.side]),
    }));
}
function shortName(info, id) {
  const s = info?.short || (info?.name ? String(info.name).trim().split(/\s+/).slice(-1)[0] : "");
  return s ? upper(s).slice(0, 11) : has(info?.number) ? "" : `#${id}`;
}

function deriveStars(events) {
  const tally = new Map();
  const get = (id, name, side) => {
    const key = String(id ?? name ?? "");
    if (!key) return null;
    if (!tally.has(key)) tally.set(key, { id: key, name: name || "Unknown", side, g: 0, a: 0, sv: 0, ga: 0, goalie: false });
    const row = tally.get(key);
    if (name && row.name === "Unknown") row.name = name;
    return row;
  };
  for (const e of events) {
    if (!e || num(e.period) >= 5) continue;
    if (e.type === "goal") {
      const s = get(e.shooter_id, e.shooter_name, e.side);
      if (s) s.g += 1;
      (e.assist_ids || []).forEach((aid, i) => {
        const a = get(aid, (e.assist_names || [])[i], e.side);
        if (a) a.a += 1;
      });
      if (has(e.goalie_id) || has(e.goalie_name)) {
        const g = get(e.goalie_id, e.goalie_name, otherSide(e.side));
        if (g) {
          g.goalie = true;
          g.ga += 1;
        }
      }
    } else if (e.type === "attempt" && upper(e.outcome) === "SAVED" && (has(e.goalie_id) || has(e.goalie_name))) {
      const g = get(e.goalie_id, e.goalie_name, otherSide(e.side));
      if (g) {
        g.goalie = true;
        g.sv += 1;
      }
    }
  }
  const scored = [...tally.values()].map((r) => {
    let v = r.g * 3 + r.a * 2;
    if (r.goalie) v += r.sv * 0.12 + (r.ga === 0 && r.sv >= 15 ? 2 : 0) - r.ga * 0.4;
    const line = r.goalie ? `${r.sv} SV${r.ga === 0 && r.sv > 0 ? " · shutout" : ""}` : `${r.g}G ${r.a}A`;
    return { ...r, v, line };
  });
  return scored.filter((r) => r.v > 0).sort((a, b) => b.v - a.v).slice(0, 3);
}


/* ==========================================================================
   Audio: two-voice booth (Web Speech API, or audio_url when provided).
   Staleness is measured in REAL time from when a line was revealed.
   Chrome stalls long utterances: a watchdog nudges pause/resume every 10 s,
   and an utterance whose onend never fires is cancelled after 12 s.
   ========================================================================== */
function readPref(key, fallback) {
  try {
    const v = window.localStorage.getItem(key);
    return v == null ? fallback : v;
  } catch {
    return fallback;
  }
}
function writePref(key, v) {
  try {
    window.localStorage.setItem(key, String(v));
  } catch {
    /* ignore */
  }
}
function useBooth({ enabled, volume, lang, lines, clipsRef }) {
  const synth = typeof window !== "undefined" && window.speechSynthesis ? window.speechSynthesis : null;
  const [voices, setVoices] = useState(() => {
    try {
      return synth ? synth.getVoices() : [];
    } catch {
      return [];
    }
  });
  const [current, setCurrent] = useState(null);
  const [primed, setPrimed] = useState(false);
  const [stalls, setStalls] = useState(0);
  const seenRef = useRef(new Set());
  const queueRef = useRef([]);
  const busyRef = useRef(false);
  const tokenRef = useRef(0);
  const audioRef = useRef(null);
  const killRef = useRef(null);
  const capRef = useRef(null);
  const seatsRef = useRef({ pbp: null, color: null });
  const [lastPath, setLastPath] = useState(""); // "neural" | "browser" | ""
  const cfgRef = useRef({});

  const [rateMul, setRateMulState] = useState(() => clampN(Number(readPref("st.voice.rate", "1")) || 1, 0.8, 1.5));
  const [pbpPick, setPbpPickState] = useState(() => readPref(`st.voice.pbp.${lang}`, ""));
  const [colorPick, setColorPickState] = useState(() => readPref(`st.voice.color.${lang}`, ""));
  useEffect(() => {
    setPbpPickState(readPref(`st.voice.pbp.${lang}`, ""));
    setColorPickState(readPref(`st.voice.color.${lang}`, ""));
  }, [lang]);
  const setRateMul = useCallback((v) => {
    const n = clampN(Number(v) || 1, 0.8, 1.5);
    setRateMulState(n);
    writePref("st.voice.rate", n);
  }, []);
  const setPbpPick = useCallback((name) => {
    setPbpPickState(name);
    writePref(`st.voice.pbp.${lang}`, name);
  }, [lang]);
  const setColorPick = useCallback((name) => {
    setColorPickState(name);
    writePref(`st.voice.color.${lang}`, name);
  }, [lang]);
  const currentVoiceRef = useRef(null);

  const ranked = useMemo(() => rankVoices(voices, lang), [voices, lang]);
  const pbpVoice = ranked.find((v) => v.name === pbpPick) || ranked[0] || null;
  const colorVoice =
    colorPick === "__same__" ? null :
    ranked.find((v) => v.name === colorPick && v.name !== pbpVoice?.name) ||
    ranked.find((v) => v !== pbpVoice && v.name !== pbpVoice?.name) ||
    null;
  cfgRef.current = { enabled, volume, lang, pbpVoice, colorVoice, voices, rateMul };

  useEffect(() => {
    if (!synth) return undefined;
    const update = () => {
      try {
        setVoices(synth.getVoices() || []);
      } catch {
        /* ignore */
      }
    };
    update();
    if (synth.addEventListener) synth.addEventListener("voiceschanged", update);
    else synth.onvoiceschanged = update;
    return () => {
      if (synth.removeEventListener) synth.removeEventListener("voiceschanged", update);
      else synth.onvoiceschanged = null;
    };
  }, [synth]);

  const haltOutput = useCallback(() => {
    tokenRef.current += 1;
    clearTimeout(killRef.current);
    clearTimeout(capRef.current);
    Object.values(seatsRef.current).forEach((a) => {
      try {
        a?.pause();
      } catch {
        /* ignore */
      }
    });
    try {
      synth?.cancel();
    } catch {
      /* ignore */
    }
    if (audioRef.current) {
      try {
        audioRef.current.pause();
      } catch {
        /* ignore */
      }
      audioRef.current = null;
    }
    busyRef.current = false;
  }, [synth]);

  const playNextRef = useRef(() => {});
  playNextRef.current = () => {
    const q = queueRef.current;
    const now = Date.now();
    while (q.length && now - num(q[0]._at, now) > (q[0].kind === "story" ? 45000 : STALE_MS)) q.shift();
    const line = q.shift();
    const cfg = cfgRef.current;
    if (!line || !cfg.enabled) {
      busyRef.current = false;
      setCurrent(null);
      return;
    }
    const token = ++tokenRef.current;
    const done = () => {
      if (tokenRef.current !== token) return;
      clearTimeout(killRef.current);
      busyRef.current = false;
      setCurrent(null);
      playNextRef.current();
    };
    busyRef.current = true;
    setCurrent(line);
    const hint = line.tts_hint || {};
    const pauseMs = clampN(num(hint.pause_before_ms, 0), 0, 900);
    // Neural clip (prefetched when the line arrived, ahead of reveal). One Audio element per seat, so
    // play-by-play and analyst cut each other off like a real booth.
    const wantsClip = !!(line.audio_url || line.audio_key);
    const rec = wantsClip ? clipsRef?.current?.get(line.id) : null;
    if (rec && rec.url) {
      const seat = line.speaker === "color" ? "color" : "pbp";
      const other = seat === "color" ? "pbp" : "color";
      try {
        seatsRef.current[other]?.pause();
      } catch {
        /* ignore */
      }
      let a = seatsRef.current[seat];
      if (!a) {
        a = new Audio();
        a.preload = "auto";
        seatsRef.current[seat] = a;
      }
      a.onended = done;
      a.onerror = done;
      a.src = rec.url;
      const energy = clampN(num(hint.energy, line.profile === "hype" ? 0.9 : 0.55), 0, 1);
      a.volume = clampN(num(cfg.volume, 0.8) * (0.8 + 0.2 * energy), 0, 1);
      audioRef.current = a;
      setLastPath("neural");
      const start = () => {
        if (tokenRef.current !== token) return;
        a.play().catch(done);
      };
      if (pauseMs) killRef.current = setTimeout(start, pauseMs);
      else start();
      capRef.current = setTimeout(() => {
        if (tokenRef.current === token) done();
      }, 30000);
      return;
    }
    if (wantsClip && (!rec || rec.state === "loading") && Date.now() - num(line._at, 0) < CLIP_WAIT_MS) {
      // Not ready yet: wait a moment (up to 1.5 s after reveal), then try again.
      killRef.current = setTimeout(() => {
        if (tokenRef.current !== token) return;
        tokenRef.current += 1;
        queueRef.current.unshift(line);
        busyRef.current = false;
        playNextRef.current();
      }, 120);
      return;
    }
    if (!BROWSER_VOICE) {
      // Captions only: hold each line just long enough to read, so the call keeps up with the play.
      const hold = line.kind === "story" ? 1100 + String(line.text || "").length * 34 : 650 + String(line.text || "").length * 24;
      killRef.current = setTimeout(done, Math.min(line.kind === "story" ? 4200 : 2400, hold));
      return;
    }
    setLastPath("browser");
    const isColor = line.speaker === "color";
    const lineVoices = line.lang && line.lang !== cfg.lang ? rankVoices(cfg.voices, line.lang) : null;
    const pbp = lineVoices ? lineVoices[0] : cfg.pbpVoice;
    const col = lineVoices ? lineVoices[1] || null : cfg.colorVoice;
    const voice = isColor ? col || pbp : pbp;
    if (!synth || !voice) {
      // No voice for this language: captions only, held long enough to read.
      killRef.current = setTimeout(done, Math.min(5000, 1200 + String(line.text || "").length * 45));
      return;
    }
    const fullText = String(line.tts_text || line.text || "");
    const chunks = prosodyChunks(fullText, {
      profile: line.profile,
      speaker: line.speaker,
      lang: line.lang || cfg.lang,
      rateMul: num(cfg.rateMul, 1),
      // Same voice for both seats? Shift the colour analyst's pitch so they still sound like two people.
      sameVoice: isColor && !col,
      seed: line.id,
    });
    if (!chunks.length) {
      done();
      return;
    }
    const vol = Math.max(0, Math.min(1, num(cfg.volume, 0.8)));
    currentVoiceRef.current = voice;
    const utterances = chunks.map((c, i) => {
      const u = new SpeechSynthesisUtterance(c.text);
      u.voice = voice;
      u.lang = voice.lang;
      u.rate = c.rate;
      u.pitch = c.pitch;
      u.volume = vol * c.volume;
      u.onerror = done;
      if (i === chunks.length - 1) u.onend = done;
      return u;
    });
    killRef.current = setTimeout(() => {
      if (tokenRef.current !== token) return;
      setStalls((n) => n + 1);
      try {
        synth.cancel();
      } catch {
        /* ignore */
      }
      done();
    }, Math.max(UTTER_TIMEOUT_MS, 3000 + (fullText.length * 95) / num(cfg.rateMul, 1)));
    const speakAll = () => {
      if (tokenRef.current !== token) return;
      try {
        utterances.forEach((u) => synth.speak(u));
      } catch {
        done();
      }
    };
    if (pauseMs) setTimeout(speakAll, pauseMs);
    else speakAll();
  };

  // Enqueue newly revealed lines.
  useEffect(() => {
    const seen = seenRef.current;
    const fresh = (lines || []).filter((l) => l && has(l.id) && !seen.has(l.id));
    fresh.forEach((l) => seen.add(l.id));
    if (!enabled || !fresh.length) return;
    const now = Date.now();
    for (const l of fresh) {
      const isStory = l.kind === "story";
      // Stories are told in sequence (intermissions arrive in one burst), so they get a longer life than play calls.
      if (now - num(l._at, 0) > (isStory ? 45000 : STALE_MS)) continue;
      if (num(l.priority) >= 3) {
        haltOutput();
        queueRef.current = [l];
        playNextRef.current();
        continue;
      }
      const q = queueRef.current;
      q.push(l);
      const cap = q.every((x) => x.kind === "story") ? 8 : QUEUE_CAP;
      while (q.length > cap) {
        let drop = 0;
        for (let i = 1; i < q.length; i += 1) if (num(q[i].priority) < num(q[drop].priority)) drop = i;
        q.splice(drop, 1);
      }
      if (!busyRef.current) playNextRef.current();
    }
  }, [lines, enabled, haltOutput]);

  // Chrome watchdog: keep the engine from going silent mid-game.
  useEffect(() => {
    if (!synth || !enabled) return undefined;
    const id = setInterval(() => {
      try {
        // Only network voices (Google/Online) suffer Chrome's long-utterance stall; nudging local voices makes them stutter.
        if (synth.speaking && !synth.paused && currentVoiceRef.current && currentVoiceRef.current.localService === false) {
          synth.pause();
          synth.resume();
        }
      } catch {
        /* ignore */
      }
    }, WATCHDOG_MS);
    return () => clearInterval(id);
  }, [synth, enabled]);

  useEffect(() => {
    if (!enabled) {
      queueRef.current = [];
      haltOutput();
      setCurrent(null);
    }
  }, [enabled, haltOutput]);

  useEffect(() => () => haltOutput(), [haltOutput]);

  /** Must run inside the "Sound on" click: browsers only allow speech after a user gesture. */
  const prime = useCallback(() => {
    if (!synth) return;
    try {
      synth.cancel();
      setVoices(synth.getVoices() || []);
      const u = new SpeechSynthesisUtterance(".");
      u.volume = 0;
      u.onend = () => setPrimed(true);
      synth.speak(u);
      synth.resume();
      setTimeout(() => setPrimed(true), 600);
    } catch {
      /* ignore */
    }
  }, [synth]);

  let status = "off";
  if (!synth) status = "unsupported";
  else if (!voices.length) status = "loading";
  else if (!pbpVoice) status = "novoice";
  else if (current) status = "speaking";
  else status = "ready";

  return {
    current,
    speaking: !!current,
    status,
    primed,
    stalls,
    pbpName: pbpVoice?.name || "",
    colorName: colorVoice?.name || "",
    choices: ranked.map((v) => ({ name: v.name, natural: NATURAL_RE.test(v.name || ""), lang: v.lang })),
    hasNatural: ranked.some((v) => NATURAL_RE.test(v.name || "")),
    rateMul,
    setRateMul,
    setPbpPick,
    setColorPick,
    supported: !!synth,
    lastPath,
    prime,
  };
}

/* ==========================================================================
   Audio: procedural crowd bed + goal horn (WebAudio, no files)
   ========================================================================== */
const MOOD_BAND = { quiet: 420, buzz: 640, loud: 860, roar: 1150, restless: 560, boo: 300 };

function useCrowdAudio({ enabled, level, mood, duck, goalPulse, hornPulse, volume }) {
  const ctxRef = useRef(null);
  const nodesRef = useRef(null);
  const baseRef = useRef(0.1);

  const unlock = useCallback(() => {
    if (ctxRef.current) {
      ctxRef.current.resume?.();
      return;
    }
    const AC = typeof window !== "undefined" ? window.AudioContext || window.webkitAudioContext : null;
    if (!AC) return;
    const ctx = new AC();
    const len = Math.floor(ctx.sampleRate * 3);
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch += 1) {
      const d = buf.getChannelData(ch);
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      for (let i = 0; i < len; i += 1) {
        const w = Math.random() * 2 - 1;
        b0 = 0.99765 * b0 + w * 0.099046;
        b1 = 0.963 * b1 + w * 0.2965164;
        b2 = 0.57 * b2 + w * 1.0526913;
        d[i] = (b0 + b1 + b2 + w * 0.1848) * 0.2;
      }
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    const band = ctx.createBiquadFilter();
    band.type = "bandpass";
    band.frequency.value = 640;
    band.Q.value = 0.5;
    const low = ctx.createBiquadFilter();
    low.type = "lowpass";
    low.frequency.value = 2600;
    const crowd = ctx.createGain();
    crowd.gain.value = 0;
    const master = ctx.createGain();
    master.gain.value = 0;
    src.connect(band);
    band.connect(low);
    low.connect(crowd);
    crowd.connect(master);
    master.connect(ctx.destination);
    src.start();
    ctxRef.current = ctx;
    nodesRef.current = { band, low, crowd, master };
  }, []);

  // master volume / mute
  useEffect(() => {
    const ctx = ctxRef.current;
    const n = nodesRef.current;
    if (!ctx || !n) return;
    n.master.gain.setTargetAtTime(enabled ? Math.max(0, Math.min(1, volume)) * 0.9 : 0, ctx.currentTime, 0.15);
  }, [enabled, volume]);

  // crowd bed follows level / mood, ducked while the commentator speaks
  useEffect(() => {
    const ctx = ctxRef.current;
    const n = nodesRef.current;
    if (!ctx || !n) return;
    const lvl = Math.max(0, Math.min(1, num(level, 0.3)));
    const base = (0.05 + lvl * 0.34) * (duck ? 0.42 : 1);
    baseRef.current = base;
    n.crowd.gain.setTargetAtTime(base, ctx.currentTime, 0.35);
    n.band.frequency.setTargetAtTime(MOOD_BAND[mood] || 640, ctx.currentTime, 0.6);
    n.band.Q.setTargetAtTime(mood === "boo" ? 1.6 : 0.5, ctx.currentTime, 0.6);
  }, [level, mood, duck, enabled]);

  // boo: low moaning voices + short whistles
  const prevMood = useRef(mood);
  useEffect(() => {
    const was = prevMood.current;
    prevMood.current = mood;
    const ctx = ctxRef.current;
    const n = nodesRef.current;
    if (!ctx || !n || !enabled || mood !== "boo" || was === "boo") return;
    const t = ctx.currentTime;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 520;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.16, t + 0.4);
    g.gain.linearRampToValueAtTime(0.12, t + 1.6);
    g.gain.linearRampToValueAtTime(0, t + 2.4);
    lp.connect(g);
    g.connect(n.master);
    [118, 131, 147, 163].forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = "triangle";
      o.frequency.setValueAtTime(f, t);
      o.frequency.linearRampToValueAtTime(f * 0.86, t + 2.3);
      o.detune.value = (i - 1.5) * 9;
      o.connect(lp);
      o.start(t);
      o.stop(t + 2.5);
    });
    [0.3, 1.1].forEach((off) => {
      const o = ctx.createOscillator();
      const wg = ctx.createGain();
      o.type = "sine";
      o.frequency.setValueAtTime(2700, t + off);
      o.frequency.linearRampToValueAtTime(2300, t + off + 0.45);
      wg.gain.setValueAtTime(0, t + off);
      wg.gain.linearRampToValueAtTime(0.035, t + off + 0.05);
      wg.gain.linearRampToValueAtTime(0, t + off + 0.5);
      o.connect(wg);
      wg.connect(n.master);
      o.start(t + off);
      o.stop(t + off + 0.55);
    });
  }, [mood, enabled]);

  // goal swell
  const prevGoal = useRef(goalPulse);
  useEffect(() => {
    if (prevGoal.current === goalPulse) return;
    prevGoal.current = goalPulse;
    const ctx = ctxRef.current;
    const n = nodesRef.current;
    if (!ctx || !n || !enabled) return;
    const t = ctx.currentTime;
    n.crowd.gain.cancelScheduledValues(t);
    n.crowd.gain.setValueAtTime(n.crowd.gain.value, t);
    n.crowd.gain.linearRampToValueAtTime(0.78, t + 0.35);
    n.crowd.gain.setTargetAtTime(baseRef.current, t + 1.4, 1.6);
    n.band.frequency.cancelScheduledValues(t);
    n.band.frequency.setValueAtTime(n.band.frequency.value, t);
    n.band.frequency.linearRampToValueAtTime(1500, t + 0.3);
    n.band.frequency.setTargetAtTime(MOOD_BAND[mood] || 640, t + 1.5, 1.4);
  }, [goalPulse, enabled, mood]);

  // home goal horn
  const prevHorn = useRef(hornPulse);
  useEffect(() => {
    if (prevHorn.current === hornPulse) return;
    prevHorn.current = hornPulse;
    const ctx = ctxRef.current;
    const n = nodesRef.current;
    if (!ctx || !n || !enabled) return;
    const t = ctx.currentTime + 0.05;
    const shaper = ctx.createWaveShaper();
    const curve = new Float32Array(256);
    for (let i = 0; i < 256; i += 1) {
      const x = (i / 255) * 2 - 1;
      curve[i] = Math.tanh(x * 2.4);
    }
    shaper.curve = curve;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 1300;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.2, t + 0.07);
    g.gain.setValueAtTime(0.2, t + 2.5);
    g.gain.linearRampToValueAtTime(0, t + 3.2);
    shaper.connect(lp);
    lp.connect(g);
    g.connect(n.master);
    [69, 138, 174, 207].forEach((f) => {
      const o = ctx.createOscillator();
      o.type = "sawtooth";
      o.frequency.setValueAtTime(f * 0.97, t);
      o.frequency.linearRampToValueAtTime(f, t + 0.12);
      o.connect(shaper);
      o.start(t);
      o.stop(t + 3.3);
    });
  }, [hornPulse, enabled]);

  useEffect(
    () => () => {
      try {
        ctxRef.current?.close();
      } catch {
        /* ignore */
      }
      ctxRef.current = null;
      nodesRef.current = null;
    },
    []
  );

  /** Procedural SFX: pass, shot, boards, whistle, organ stings, final horn. No files. */
  const sfx = useCallback(
    (kind) => {
      const ctx = ctxRef.current;
      const n = nodesRef.current;
      if (!ctx || !n || !enabled) return;
      const t = ctx.currentTime + 0.01;
      const out = ctx.createGain();
      out.connect(n.master);
      const noise = (dur, type, freq, q, peak) => {
        const len = Math.max(1, Math.floor(ctx.sampleRate * dur));
        const b = ctx.createBuffer(1, len, ctx.sampleRate);
        const d = b.getChannelData(0);
        for (let i = 0; i < len; i += 1) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
        const src = ctx.createBufferSource();
        src.buffer = b;
        const f = ctx.createBiquadFilter();
        f.type = type;
        f.frequency.value = freq;
        f.Q.value = q;
        const g = ctx.createGain();
        g.gain.value = peak;
        src.connect(f);
        f.connect(g);
        g.connect(out);
        src.start(t);
      };
      const tone = (type, f0, f1, start, dur, peak) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = type;
        o.frequency.setValueAtTime(f0, t + start);
        if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t + start + dur);
        g.gain.setValueAtTime(0, t + start);
        g.gain.linearRampToValueAtTime(peak, t + start + 0.012);
        g.gain.exponentialRampToValueAtTime(0.0008, t + start + dur);
        o.connect(g);
        g.connect(out);
        o.start(t + start);
        o.stop(t + start + dur + 0.05);
        return o;
      };
      if (kind === "pass") {
        noise(0.05, "bandpass", 2400, 3, 0.18);
        tone("sine", 980, 700, 0, 0.06, 0.05);
      } else if (kind === "shot") {
        noise(0.09, "highpass", 1800, 0.8, 0.42);
        tone("sine", 160, 60, 0, 0.12, 0.2);
      } else if (kind === "boards") {
        tone("sine", 95, 50, 0, 0.28, 0.32);
        noise(0.18, "lowpass", 420, 0.7, 0.3);
      } else if (kind === "whistle") {
        const o = tone("sine", 2850, 2850, 0, 0.55, 0.07);
        const lfo = ctx.createOscillator();
        const lg = ctx.createGain();
        lfo.frequency.value = 28;
        lg.gain.value = 70;
        lfo.connect(lg);
        lg.connect(o.frequency);
        lfo.start(t);
        lfo.stop(t + 0.6);
      } else if (kind === "organ" || kind === "charge") {
        // Rink organ: square + sine stack per note. "charge" = the classic rising call.
        const notes = kind === "charge" ? [392, 523.25, 659.25, 783.99, 659.25, 783.99] : [523.25, 659.25, 783.99, 1046.5];
        const step = kind === "charge" ? 0.16 : 0.12;
        notes.forEach((f, i) => {
          const dur = kind === "charge" && i >= 4 ? 0.45 : step * 0.95;
          tone("square", f, f, i * step, dur, 0.035);
          tone("sine", f / 2, f / 2, i * step, dur, 0.05);
        });
      } else if (kind === "alert") {
        // coaching decision chime (priority 3)
        tone("sine", 880, 880, 0, 0.18, 0.09);
        tone("sine", 1318.5, 1318.5, 0.14, 0.3, 0.08);
      } else if (kind === "horn") {
        [69, 138, 174].forEach((f) => tone("sawtooth", f, f, 0, 2.2, 0.06));
      }
    },
    [enabled]
  );

  return { unlock, sfx };
}


/* ==========================================================================
   Small pieces
   ========================================================================== */
const Est = ({ title = "Estimated by the sim, not measured" }) => (
  <sup className="st-est" title={title}>
    est.
  </sup>
);

/** One fatigue bar everywhere: green -> amber -> red, always labelled as an estimate. */
function Fatigue({ value }) {
  if (!has(value)) return <span className="st-fat is-none" title="No fatigue data" />;
  const f = Math.max(0, Math.min(1, num(value)));
  const lvl = f > 0.72 ? "is-red" : f > 0.45 ? "is-amber" : "is-green";
  return (
    <span className={`st-fat ${lvl}`} title={`Fatigue ${Math.round(f * 100)}% (estimated)`}>
      <i style={{ width: `${Math.max(6, Math.round(f * 100))}%` }} />
    </span>
  );
}

/** A team chip in its identity (home dark kit / away white with trim). */
const TeamChip = ({ side, colors, teams, children, className = "" }) => (
  <span className={`st-chip is-${side} ${className}`} style={colors[side]?.chip}>
    {children || teams?.[side]?.abbr || (side === "home" ? "HME" : "AWY")}
  </span>
);

function Scorebug({ teams, colors, score, sog, period, clockText, manpower, flashSide, final, intermission, late, stopped }) {
  const mp = manpower || {};
  const state = upper(mp.state) || "EV";
  const block = (side) => {
    const card = teams?.[side] || {};
    return (
      <div className={`st-bug__team is-${side}${flashSide === side ? " is-flash" : ""}${card.is_user ? " is-user" : ""}`} style={colors[side].chip} title={card.name || ""}>
        <b className="st-bug__abbr">
          {card.abbr || (side === "home" ? "HOME" : "AWAY")}
          {card.is_user ? <span className="st-bug__you">you</span> : null}
        </b>
        <strong className="st-bug__score" key={`${side}-${num(score?.[side])}`}>
          {num(score?.[side])}
        </strong>
        {has(sog?.[side]) ? (
          <span className="st-bug__sog">
            <small>SOG</small> {sog[side]}
          </span>
        ) : null}
      </div>
    );
  };
  const periodLabel = final ? `Final${final.decision && final.decision !== "REG" ? ` / ${final.decision}` : ""}` : intermission ? "Intermission" : periodName(period);
  // D11: "OTT PP 0:52" in the PP team's colours, never a bare "PP 4V5".
  const ppSide = mp.pp_side;
  const showPP = !final && ppSide && (state === "PP" || state === "SH");
  const en = !final && state === "EN";
  return (
    <div className={`st-bug${late ? " is-late" : ""}${stopped ? " is-stopped" : ""}`} role="group" aria-label="Scoreboard">
      {block("away")}
      <div className="st-bug__mid">
        <span className="st-bug__period">{periodLabel}</span>
        {!final ? <span className="st-bug__clock">{clockText}</span> : null}
        {showPP ? (
          <span className="st-bug__pp" style={colors[ppSide].chip} title={`${teams?.[ppSide]?.name || ""} power play`}>
            {teams?.[ppSide]?.abbr} PP{has(mp.pp_sec_left) ? ` ${mmss(mp.pp_sec_left)}` : ""}
          </span>
        ) : en ? (
          <span className="st-bug__pp is-en" title="A goalie is pulled for an extra attacker">
            Extra attacker
          </span>
        ) : !final && (state === "4V4" || state === "3V3") ? (
          <span className="st-bug__pp is-even">{state === "3V3" ? "3 on 3" : "4 on 4"}</span>
        ) : null}
      </div>
      {block("home")}
    </div>
  );
}

/** Head-and-shoulders silhouettes, generated once. */
function crowdPath(count, w, h, seed) {
  let d = "";
  let arms = "";
  const step = w / count;
  for (let i = 0; i < count; i += 1) {
    const j = hash01(`${seed}:${i}`);
    const x = i * step + step / 2 + (j - 0.5) * step * 0.35;
    const hy = h * (0.42 + j * 0.12);
    const r = step * (0.2 + j * 0.05);
    d += `M${(x - r * 2.2).toFixed(1)},${h} Q${(x - r * 2.1).toFixed(1)},${(hy + r * 1.6).toFixed(1)} ${x.toFixed(1)},${(hy + r * 1.5).toFixed(1)} Q${(x + r * 2.1).toFixed(1)},${(hy + r * 1.6).toFixed(1)} ${(x + r * 2.2).toFixed(1)},${h} Z `;
    d += `M${(x + r).toFixed(1)},${hy.toFixed(1)} a${r.toFixed(1)},${r.toFixed(1)} 0 1 0 ${(-2 * r).toFixed(1)},0 a${r.toFixed(1)},${r.toFixed(1)} 0 1 0 ${(2 * r).toFixed(1)},0 Z `;
    if (j > 0.35) arms += `M${(x - r * 1.6).toFixed(1)},${(hy + r * 1.8).toFixed(1)} L${(x - r * 2.4).toFixed(1)},${(hy - r * 1.6).toFixed(1)} M${(x + r * 1.6).toFixed(1)},${(hy + r * 1.8).toFixed(1)} L${(x + r * 2.4).toFixed(1)},${(hy - r * 1.6).toFixed(1)} `;
  }
  return { d, arms };
}

/** The crowd (D6): silhouettes that bob with the noise and throw their arms up on goals, a loudness meter and a mood word. */
function Crowd({ crowd, colors, drama, goal, chant }) {
  const c = crowd || {};
  const lvl = Math.max(0, Math.min(1, num(c.level, 0.25)));
  const mood = c.mood || "buzz";
  const tier = num(c.rivalry?.tier);
  const back = useMemo(() => crowdPath(46, 1000, 60, "b"), []);
  const front = useMemo(() => crowdPath(38, 1000, 60, "f"), []);
  const segs = 10;
  const lit = Math.round(lvl * segs);
  return (
    <div
      className={`st-crowd is-${mood}${drama ? ` is-${drama}` : ""}${goal ? " is-goal" : ""}${lvl > 0.75 ? " is-loud" : ""}`}
      style={{ "--st-crowd": lvl, "--st-crowd-dur": `${(1.5 - lvl * 1.0).toFixed(2)}s`, "--st-home": colors.home.kit, "--st-away": colors.away.trim }}
      title={crowd ? `Crowd: ${mood} (${Math.round(lvl * 100)}%)` : "No crowd data from the sim"}
    >
      <div className="st-crowd__meter" aria-label={`Crowd ${mood}`}>
        <span className="st-crowd__word">{crowd ? mood : "—"}</span>
        <span className="st-crowd__bars">
          {Array.from({ length: segs }, (_, i) => (
            <i key={i} className={i < lit ? "is-lit" : ""} />
          ))}
        </span>
      </div>
      <svg className="st-crowd__svg" viewBox="0 0 1000 60" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <g className="st-crowd__back">
          <path d={back.d} />
        </g>
        <g className="st-crowd__arms">
          <path d={front.arms} />
        </g>
        <g className="st-crowd__front">
          <path d={front.d} />
        </g>
        <rect className="st-crowd__pocket" x="760" y="0" width="120" height="60" />
      </svg>
      {chant ? (
        <span key={chant} className="st-crowd__chant">
          {chant}
        </span>
      ) : null}
      {tier > 0 ? (
        <span className="st-crowd__rival" title="Rivalry">
          {"★".repeat(Math.min(3, tier))} {c.rivalry?.name || "Rivals"}
        </span>
      ) : null}
    </div>
  );
}

/** Momentum (D24): tug-of-war rope in both team identities, with a label. */
function Momentum({ momentum, colors, teams }) {
  if (!momentum) return <div className="st-mom is-empty" />;
  const v = Math.max(-1, Math.min(1, num(momentum.value)));
  return (
    <div
      className="st-mom"
      style={{ "--st-home": colors.home.kit, "--st-away-line": colors.away.trim, "--st-knot": `${50 + v * 45}%` }}
      title="Momentum: a model estimate from recent shots and chances"
    >
      <TeamChip side="away" colors={colors} teams={teams} />
      <span className="st-mom__rope">
        <i className="st-mom__knot" />
      </span>
      <TeamChip side="home" colors={colors} teams={teams} />
      <span className="st-mom__label">{momentum.label || (Math.abs(v) < 0.15 ? "Even" : v > 0 ? `${teams?.home?.abbr || "Home"} pushing` : `${teams?.away?.abbr || "Away"} pushing`)}</span>
    </div>
  );
}

/* ---------- Rink ---------- */
function starPoints(cx, cy, r) {
  const out = [];
  for (let i = 0; i < 10; i += 1) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const ang = (Math.PI / 5) * i - Math.PI / 2;
    out.push(`${(cx + rad * Math.cos(ang)).toFixed(2)},${(cy + rad * Math.sin(ang)).toFixed(2)}`);
  }
  return out.join(" ");
}
// Ice is 200 x 85 ft; the SVG adds boards, glass, benches / boxes and a ring of seats around it.
const VB = { x: -110, y: -52, w: 220, h: 104 };
// Players may stand in the benches (+y) and penalty boxes (-y), just outside the boards.
const clampX = (x) => Math.max(-99.5, Math.min(99.5, num(x)));
const clampY = (y) => Math.max(-50.5, Math.min(50.5, num(y)));
const RED = "#c8102e";
const BLUE = "#0b3d91";
const ICE_LINE = 2 / 12; // 2-inch lines (goal lines, circles, hashes, creases)
// Marker sizes in feet (true-ish scale): skaters 2.4 ft, goalies 3 ft.
const SK_R = 0.85; // smaller bodies leave room to see skating lines and gaps
const G_R = 1.1;
const PUCK_R = 1.05; // drawn at about 2 ft so it reads (a real puck is 3 in)
const SHOT_TYPES = new Set(["slap", "one_timer", "onetimer", "snap", "wrist", "backhand", "tip", "wraparound", "deke"]);
const PUFFS = 6;
const AD_TEXT = ["NHL DYNASTY", "", "ZAMBONI FUEL", "PUCKSTOP", "NHL DYNASTY", "", "ICEBOX COLA", "BENCH BOSS"];

/** Rounded rect as a path (for even-odd rings). */
function rrPath(x, y, w, h, r) {
  return `M${x + r},${y} H${x + w - r} A${r},${r} 0 0 1 ${x + w},${y + r} V${y + h - r} A${r},${r} 0 0 1 ${x + w - r},${y + h} H${x + r} A${r},${r} 0 0 1 ${x},${y + h - r} V${y + r} A${r},${r} 0 0 1 ${x + r},${y} Z`;
}

/** Arena around the sheet: seats (muted), benches and boxes behind the glass, the white board face with dasher ads. */
function Arena({ colors, teams, boxed }) {
  const area = (side, x0, x1, y0, y1, label) => (
    <g key={`${label}${side}`} className={`st-area is-${side}${label === "box" && boxed?.[side] ? " is-busy" : ""}`}>
      <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx="1.2" className="st-area__floor" />
      <rect x={x0} y={label === "box" ? y1 - 0.9 : y0} width={x1 - x0} height="0.9" fill={colors[side].kit} stroke={colors[side].trim} strokeWidth="0.2" />
      <text x={(x0 + x1) / 2} y={(y0 + y1) / 2 + (label === "box" ? -0.4 : 0.5)} textAnchor="middle" dominantBaseline="middle" className="st-area__txt">
        {label === "box" ? `${teams?.[side]?.abbr || ""} BOX` : `${teams?.[side]?.abbr || ""} BENCH`}
      </text>
    </g>
  );
  // Ads on the board face, top and bottom straights only.
  const ads = [];
  [-1, 1].forEach((sy) => {
    for (let i = 0; i < 8; i += 1) {
      const x = -68 + i * 17;
      const txt = AD_TEXT[(i + (sy > 0 ? 3 : 0)) % AD_TEXT.length] || teams?.home?.abbr || "";
      const y = sy < 0 ? -44.4 : 42.6;
      ads.push(
        <g key={`ad${sy}${i}`} className={`st-ad${i % 3 === 1 ? " is-team" : ""}`} style={i % 3 === 1 ? { "--st-bg": colors.home.kit, "--st-fg": colors.home.kitInk } : undefined}>
          <rect x={x - 7.6} y={y} width="15.2" height="1.8" rx="0.4" />
          <text x={x} y={y + 0.95} textAnchor="middle" dominantBaseline="middle">
            {txt}
          </text>
        </g>
      );
    }
  });
  return (
    <g className="st-arena" aria-hidden="true">
      <path className="st-seats" d={`${rrPath(-110, -52, 220, 104, 36)} ${rrPath(-102.4, -44.9, 204.8, 89.8, 30.4)}`} fillRule="evenodd" />
      <rect className="st-seats__row" x="-107.6" y="-49.6" width="215.2" height="99.2" rx="33.6" />
      <rect className="st-seats__row" x="-105.2" y="-47.3" width="210.4" height="94.6" rx="31.4" />
      {area("home", -36, -6, 44.9, 50.6, "bench")}
      {area("away", 6, 36, 44.9, 50.6, "bench")}
      {area("home", -14, -2.5, -50.6, -44.9, "box")}
      {area("away", 2.5, 14, -50.6, -44.9, "box")}
      <path className="st-boardface" d={`${rrPath(-102, -44.5, 204, 89, 30)} ${rrPath(-100, -42.5, 200, 85, 28)}`} fillRule="evenodd" />
      {ads}
      <rect className="st-glass" x="-102.2" y="-44.7" width="204.4" height="89.4" rx="30.2" />
    </g>
  );
}

/** NHL markings at true widths: 1 ft red / blue lines, 2 in goal lines, circles, hashes, L-marks, creases, trapezoids. */
function RinkMarkings({ colors, teams }) {
  const L = ICE_LINE;
  const circles = [
    [-69, -22],
    [-69, 22],
    [69, -22],
    [69, 22],
  ];
  const neutral = [
    [-20, -22],
    [-20, 22],
    [20, -22],
    [20, 22],
  ];
  const net = (side) => {
    const s = side === "right" ? 1 : -1;
    const gx = 89 * s;
    const cx = gx - 4.47 * s;
    return (
      <g key={side}>
        <path d={`M${gx},-4 L${cx},-4 A6,6 0 0 ${s > 0 ? 0 : 1} ${cx},4 L${gx},4 Z`} className="st-crease" strokeWidth={L} />
        <path d={`M${gx},-3 L${gx + 2.2 * s},-3 Q${gx + 3.4 * s},-3 ${gx + 3.4 * s},-1.8 L${gx + 3.4 * s},1.8 Q${gx + 3.4 * s},3 ${gx + 2.2 * s},3 L${gx},3`} fill="url(#st-net-mesh)" stroke="#3a3f4f" strokeWidth="0.28" />
        <line x1={gx} y1="-3" x2={gx} y2="3" stroke={RED} strokeWidth="0.55" strokeLinecap="round" />
        {/* trapezoid: 18 ft at the goal line, 28 ft at the end boards */}
        <path d={`M${gx},-9 L${100 * s},-14 M${gx},9 L${100 * s},14`} stroke={RED} strokeWidth={L} fill="none" />
      </g>
    );
  };
  return (
    <g className="st-rink-lines">
      <circle cx="0" cy="0" r="9" fill={colors.home.kit} fillOpacity="0.1" stroke={colors.home.kit} strokeOpacity="0.25" strokeWidth="0.4" />
      <text x="0" y="0.4" className="st-rink-logo" fill={colors.home.kit} fillOpacity="0.26" textAnchor="middle" dominantBaseline="middle">
        {teams?.home?.abbr || ""}
      </text>
      <line x1="89" y1="-43" x2="89" y2="43" stroke={RED} strokeWidth={L} />
      <line x1="-89" y1="-43" x2="-89" y2="43" stroke={RED} strokeWidth={L} />
      <line x1="25.5" y1="-43" x2="25.5" y2="43" stroke={BLUE} strokeWidth="1" />
      <line x1="-25.5" y1="-43" x2="-25.5" y2="43" stroke={BLUE} strokeWidth="1" />
      <path d="M0,-43 V-9 M0,9 V43" stroke={RED} strokeWidth="1" />
      <path d="M0,-43 V-9 M0,9 V43" stroke="#ffffff" strokeWidth="0.12" strokeDasharray="1 1" opacity="0.7" />
      <circle cx="0" cy="0" r="15" fill="none" stroke={BLUE} strokeWidth={L} />
      <circle cx="0" cy="0" r="0.5" fill={BLUE} />
      <path d="M-10,-42.5 A10,10 0 0 0 10,-42.5" fill="none" stroke={RED} strokeWidth={L} />
      {circles.map(([cx, cy]) => (
        <g key={`c${cx}${cy}`}>
          <circle cx={cx} cy={cy} r="15" fill="none" stroke={RED} strokeWidth={L} />
          <circle cx={cx} cy={cy} r="1" fill={RED} />
          <path d={[-2.9, 2.9].map((dx) => `M${cx + dx},${cy - 14.72} v-2 M${cx + dx},${cy + 14.72} v2`).join(" ")} stroke={RED} strokeWidth={L} />
          <path
            d={[-1, 1]
              .flatMap((sx) => [-1, 1].map((sy) => `M${cx + sx * 6},${cy + sy * 0.75} H${cx + sx * 2} V${cy + sy * 3.75}`))
              .join(" ")}
            stroke={RED}
            strokeWidth={L}
            fill="none"
          />
        </g>
      ))}
      {neutral.map(([cx, cy]) => (
        <circle key={`n${cx}${cy}`} cx={cx} cy={cy} r="1" fill={RED} />
      ))}
      {net("right")}
      {net("left")}
    </g>
  );
}

const overlaps = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
const toPct = (x, y) => ({ left: `${((clampX(x) - VB.x) / VB.w) * 100}%`, top: `${((clampY(y) - VB.y) / VB.h) * 100}%` });

/**
 * The rink. Markers render once per on-ice set; the playback loop writes their
 * transforms through `apiRef.current.draw()`. Label collision and overlay
 * dodging run only when `layout` is true (~10 Hz).
 */
function Rink({ apiRef, onIce, roster, colors, teams, period, shots, headPt, celebration, saveFlash, drama, dramaText, alert, onAlertAction, onAlertDismiss, tracking, buffering, stop, faceoff, foWin, danger, hit, delayed, statPop, replay, onSkipReplay, boxed, scale = 1, strip = null, names = "key" }) {
  const markRef = useRef(new Map());
  const scaleRef = useRef(scale);
  scaleRef.current = scale;
  // Angled "broadcast" view of the rink (on by default; remembered per browser).
  const [tilt, setTilt] = useState(() => {
    try {
      return window.localStorage.getItem("st.view.tilt") !== "0";
    } catch {
      return true;
    }
  });
  const toggleTilt = () =>
    setTilt((v) => {
      try {
        window.localStorage.setItem("st.view.tilt", v ? "0" : "1");
      } catch {
        /* ignore */
      }
      return !v;
    });
  const namesRef = useRef(names);
  namesRef.current = names;
  const hoverRef = useRef(null);
  const selRef = useRef(null);
  const puckRef = useRef(null);
  const puckBodyRef = useRef(null);
  const puckShadowRef = useRef(null);
  const haloRef = useRef(null);
  const trailRef = useRef([]);
  const passRef = useRef(null);
  const puffRef = useRef([]);
  const svgRef = useRef(null);
  const rinkRef = useRef(null);
  const stRef = useRef({ lift: -1, loose: null, shot: null });

  useEffect(() => {
    apiRef.current = {
      draw(fr) {
        const marks = markRef.current;
        const car = has(fr.carrier) ? String(fr.carrier) : null;
        const list = fr.layout ? [] : null;
        fr.pos.forEach((xy, id) => {
          const el = marks.get(id);
          if (!el) return;
          const cx = clampX(xy[0]);
          const cy = clampY(xy[1]);
          el.g.setAttribute("transform", `translate(${cx.toFixed(2)} ${cy.toFixed(2)})`);
          if (!el.shown) {
            el.g.style.visibility = "visible";
            el.shown = true;
          }
          const isCar = id === car;
          if (el.car !== isCar) {
            el.g.classList.toggle("is-carrier", isCar);
            el.car = isCar;
          }
          if (list) list.push({ id, el, cx, cy });
        });
        if (list) {
          // Names (HFM style): numbers only by default; names for the carrier, the last passer / receiver,
          // goalies and hovered / selected players. "all" labels everyone that fits; "off" only hover / selection.
          const mode = namesRef.current;
          const recent = fr.recent || null;
          const hov = hoverRef.current;
          const sel = selRef.current;
          const sc = scaleRef.current;
          const half = SK_R * sc + 0.25;
          const off = half + 1.45;
          const px = fr.pk ? fr.pk[0] : 0;
          const py = fr.pk ? fr.pk[1] : 0;
          list.forEach((m) => {
            const key = m.id === car || m.id === hov || m.id === sel || (recent && recent.has(m.id));
            m.want = mode === "all" || (mode === "key" ? key || m.el.g0 : m.id === hov || m.id === sel);
            m.force = m.id === car || m.id === hov || m.id === sel || (mode !== "off" && m.el.g0);
            m.pr = m.id === hov || m.id === sel ? -2e6 : m.id === car ? -1e6 : m.el.g0 ? -1e5 : (m.cx - px) ** 2 + (m.cy - py) ** 2;
          });
          list.sort((a, b) => a.pr - b.pr);
          // Bodies and the puck are obstacles for labels.
          const placed = list.map((m) => ({ x: m.cx - half, y: m.cy - half, w: 2 * half, h: 2 * half }));
          if (fr.pk) placed.push({ x: px - 1.6, y: py - 1.6, w: 3.2, h: 3.2 });
          for (const m of list) {
            const { el, cx, cy } = m;
            if (!el.tag) continue;
            let pick = null;
            if (m.want) {
              const w = el.w;
              const h = 2.3;
              const tries = [
                [0, off],
                [0, -off],
                [w / 2 + half + 0.5, 0],
                [-(w / 2 + half + 0.5), 0],
              ];
              for (const [dx, dy] of tries) {
                const r = { x: cx + dx - w / 2, y: cy + dy - h / 2, w, h };
                if (r.x < -101 || r.x + r.w > 101 || r.y < -51 || r.y + r.h > 51) continue;
                if (!placed.some((q) => overlaps(q, r))) {
                  pick = [dx, dy, r];
                  break;
                }
              }
              if (!pick && m.force) pick = [0, cy > 40 ? -off : off, null];
            }
            if (pick) {
              if (pick[2]) placed.push(pick[2]);
              el.tag.setAttribute("transform", `translate(${pick[0].toFixed(2)} ${pick[1].toFixed(2)})`);
              if (el.hidden) {
                el.tag.style.visibility = "visible";
                el.hidden = false;
              }
            } else if (!el.hidden) {
              el.tag.style.visibility = "hidden";
              el.hidden = true;
            }
          }
          // Overlays must not sit on the puck: nudge them away.
          const rink = rinkRef.current;
          if (rink && fr.pk) {
            const rr = rink.getBoundingClientRect();
            const sx = rr.left + ((fr.pk[0] - VB.x) / VB.w) * rr.width;
            const sy = rr.top + ((fr.pk[1] - VB.y) / VB.h) * rr.height;
            const els = [...rink.querySelectorAll("[data-avoid]")];
            const stack = document.querySelector(".st-nstack");
            if (stack) els.push(stack);
            const pad = Math.max(18, rr.width * 0.02);
            for (const o of els) {
              const r = o.getBoundingClientRect();
              const dy0 = o._dodge || 0;
              const top = r.top - dy0;
              const bottom = r.bottom - dy0;
              const hitP = sx > r.left - pad && sx < r.right + pad && sy > top - pad && sy < bottom + pad;
              let dy = 0;
              if (hitP) dy = sy < (top + bottom) / 2 ? bottom - sy + pad : -(sy - top + pad);
              if (Math.abs(dy - dy0) > 2 && (hitP || dy0)) {
                o._dodge = dy;
                o.style.setProperty("--st-dodge", `${Math.round(dy)}px`);
                o.classList.toggle("is-dodge", !!dy);
              }
            }
          }
        }

        // Puck: on the stick in front of a moving carrier, lifted when pz says so, halo pulses only while loose.
        const pk = puckRef.current;
        const S = stRef.current;
        if (pk) {
          if (fr.pk) {
            pk.setAttribute("transform", `translate(${clampX(fr.pk[0]).toFixed(2)} ${clampY(fr.pk[1]).toFixed(2)})`);
            if (pk.style.visibility !== "visible") pk.style.visibility = "visible";
            const lift = Math.round(Math.min(6, Math.max(0, num(fr.pz))) * 10) / 10;
            if (lift !== S.lift) {
              S.lift = lift;
              const s = 1 + Math.min(0.5, lift * 0.12);
              puckBodyRef.current?.setAttribute("transform", lift > 0 ? `translate(0 ${(-lift * 0.45).toFixed(2)}) scale(${s.toFixed(2)})` : "");
              if (puckShadowRef.current) puckShadowRef.current.style.opacity = lift > 0 ? String(Math.min(0.45, 0.15 + lift * 0.08)) : "0";
            }
            const loose = !car;
            if (loose !== S.loose) {
              S.loose = loose;
              haloRef.current?.classList.toggle("is-loose", loose);
            }
          } else if (pk.style.visibility !== "hidden") pk.style.visibility = "hidden";
        }
        // Trail: longer and brighter when the puck is fast; a sharp streak on shots.
        const tr = fr.trail || [];
        const lines = trailRef.current;
        const spd = num(fr.puckSpeed);
        const shown = Math.min(lines.length, Math.round(3 + spd / 9));
        const bright = Math.min(1, 0.25 + spd / 90);
        const shot = !!fr.shot;
        if (shot !== S.shot) {
          S.shot = shot;
          lines.forEach((ln) => ln && ln.classList.toggle("is-shot", shot));
        }
        for (let i = 0; i < lines.length; i += 1) {
          const ln = lines[i];
          if (!ln) continue;
          const a = tr[tr.length - 2 - i];
          const b = tr[tr.length - 1 - i];
          if (!a || !b || i >= shown || spd < 6) {
            if (ln.style.visibility !== "hidden") ln.style.visibility = "hidden";
            continue;
          }
          ln.setAttribute("x1", clampX(a[0]).toFixed(2));
          ln.setAttribute("y1", clampY(a[1]).toFixed(2));
          ln.setAttribute("x2", clampX(b[0]).toFixed(2));
          ln.setAttribute("y2", clampY(b[1]).toFixed(2));
          ln.style.visibility = "visible";
          ln.style.opacity = String((bright * (1 - i / shown)).toFixed(2));
        }
        // Pass: faint dotted path for 0.6 s.
        const pl = passRef.current;
        if (pl) {
          const P = fr.pass;
          if (P && fr.pk && P.age < 600) {
            pl.setAttribute("x1", clampX(P.x).toFixed(2));
            pl.setAttribute("y1", clampY(P.y).toFixed(2));
            pl.setAttribute("x2", clampX(fr.pk[0]).toFixed(2));
            pl.setAttribute("y2", clampY(fr.pk[1]).toFixed(2));
            pl.style.opacity = String((0.55 * (1 - P.age / 600)).toFixed(2));
            pl.style.visibility = "visible";
          } else if (pl.style.visibility !== "hidden") pl.style.visibility = "hidden";
        }
        // Snow puffs on hard stops (frame `sp`).
        const puffs = fr.puffs || [];
        puffRef.current.forEach((c, i) => {
          if (!c) return;
          const p = puffs[i];
          if (!p || p.age > 700) {
            if (c.style.visibility !== "hidden") c.style.visibility = "hidden";
            return;
          }
          const k = p.age / 700;
          c.setAttribute("cx", clampX(p.x).toFixed(2));
          c.setAttribute("cy", clampY(p.y).toFixed(2));
          c.setAttribute("r", (0.8 + k * 1.8).toFixed(2));
          c.style.opacity = String((0.8 * (1 - k)).toFixed(2));
          c.style.visibility = "visible";
        });
        const svg = svgRef.current;
        if (svg && svg.dataset.ph !== (fr.ph || "play")) svg.dataset.ph = fr.ph || "play";
      },
    };
    return () => {
      apiRef.current = null;
    };
  }, [apiRef]);

  const homeRight = homeAttacksRight(period);
  const enOf = drama === "en-home" ? "home" : drama === "en-away" ? "away" : null;

  const marker = ({ id, side }) => {
    const info = roster.get(String(id)) || {};
    const col = colors[side] || colors.home;
    const isG = upper(info.pos) === "G";
    const numTxt = has(info.number) ? String(info.number) : "";
    const name = shortName(info, id);
    const w = Math.max(3.6, name.length * 1.12 + 1.6);
    const r = isG ? G_R : SK_R;
    const reg = (g) => {
      const k = String(id);
      if (!g) {
        markRef.current.delete(k);
        return;
      }
      const prev = markRef.current.get(k);
      if (prev && prev.g === g) return;
      const tag = g.querySelector(".st-tag");
      markRef.current.set(k, { g, tag, w, g0: isG, car: g.classList.contains("is-carrier"), hidden: !tag || tag.style.visibility !== "visible", shown: g.style.visibility === "visible" });
    };
    const tip = [numTxt ? `#${numTxt}` : "", info.name || name, info.pos || "", teams?.[side]?.abbr || ""].filter(Boolean).join(" · ");
    const k = String(id);
    return (
      <g
        key={`${side}-${id}`}
        ref={reg}
        className={`st-tok is-${side}${isG ? " is-g" : ""}`}
        style={{ visibility: "hidden" }}
        onMouseEnter={() => {
          hoverRef.current = k;
        }}
        onMouseLeave={() => {
          if (hoverRef.current === k) hoverRef.current = null;
        }}
        onClick={() => {
          selRef.current = selRef.current === k ? null : k;
        }}
      >
        <title>{`${tip} (position estimated)`}</title>
        <g transform={scale !== 1 ? `scale(${scale})` : undefined}>
          <circle className="st-tok__carry" r={r + 0.9} />
          {side === "away" ? (
            <rect x={-r} y={-r} width={2 * r} height={2 * r} rx={r * 0.35} fill={col.kit} stroke={col.trim} strokeWidth={isG ? 0.42 : 0.34} />
          ) : (
            <circle r={r} fill={col.kit} stroke={isG ? "#fbfbf6" : "#071233"} strokeWidth={isG ? 0.38 : 0.26} />
          )}
          <text y="0.08" textAnchor="middle" dominantBaseline="middle" fill={col.kitInk} className={`st-tok__num${numTxt.length > 1 ? " is-2" : ""}`}>
            {numTxt || (isG ? "G" : "")}
          </text>
        </g>
        {name ? (
          <g className="st-tag" style={{ visibility: "hidden" }}>
            <rect x={-w / 2} y="-1.15" width={w} height="2.3" rx="1.15" />
            <text y="0.1" textAnchor="middle" dominantBaseline="middle">
              {name}
            </text>
          </g>
        ) : null}
      </g>
    );
  };

  // Shot markers: small and faint so the ice stays clean (D19, round 5).
  const shotMark = (e) => {
    const age = Math.max(0, headPt - ptOf(e));
    if (age > SHOT_FADE_SEC) return null;
    const x = clampX(e.shot_loc.x);
    const y = clampY(e.shot_loc.y);
    const out = upper(e.outcome) || (e.type === "goal" ? "GOAL" : "");
    const op = Math.max(0.08, 0.45 * (1 - age / SHOT_FADE_SEC));
    const col = colors[e.side]?.trim || "#a9bfdc";
    const label = `${e.description || "Shot attempt"} — ${out === "GOAL" || e.type === "goal" ? "goal" : out === "SAVED" ? "saved" : out === "BLOCKED" ? "blocked" : "missed"}`;
    if (out === "GOAL" || e.type === "goal")
      return (
        <polygon key={`s${e.seq}`} points={starPoints(x, y, 1.6)} className="st-mk is-goal" opacity={Math.min(0.8, op + 0.25)}>
          <title>{label}</title>
        </polygon>
      );
    return (
      <circle key={`s${e.seq}`} cx={x} cy={y} r="0.7" className={`st-mk is-${out === "SAVED" ? "save" : out === "BLOCKED" ? "block" : "miss"}`} stroke={col} opacity={op}>
        <title>{label}</title>
      </circle>
    );
  };

  const foArrow = foWin ? (() => {
    const winDefendsLeft = (foWin.side === "home") === homeRight;
    const dx = winDefendsLeft ? -1 : 1;
    const x = clampX(foWin.x);
    const y = clampY(foWin.y);
    return (
      <g key={foWin.key} className="st-fo-arrow" style={colors[foWin.side]?.chip}>
        <path d={`M${x},${y} L${x + dx * 8},${y}`} />
        <path d={`M${x + dx * 8},${y} l${-dx * 2},-1.6 M${x + dx * 8},${y} l${-dx * 2},1.6`} />
      </g>
    );
  })() : null;

  return (
    <div className="st-rinkcell">
      <div ref={rinkRef} className={`st-rink${replay ? " is-replay" : ""}${tilt ? " is-3d" : ""}`}>
        <svg ref={svgRef} viewBox={`${VB.x} ${VB.y} ${VB.w} ${VB.h}`} preserveAspectRatio="xMidYMid meet" className="st-rink-svg" role="img" aria-label="Rink" data-ph="play">
          <defs>
            <clipPath id="st-rink-clip">
              <rect x="-100" y="-42.5" width="200" height="85" rx="28" ry="28" />
            </clipPath>
            <pattern id="st-ice-scratch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(24)">
              <line x1="0" y1="0" x2="0" y2="7" stroke="#8fb2d1" strokeOpacity="0.12" strokeWidth="0.09" />
              <line x1="3.4" y1="1" x2="3.4" y2="4.6" stroke="#8fb2d1" strokeOpacity="0.09" strokeWidth="0.07" />
            </pattern>
            <linearGradient id="st-ice-sheen" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#ffffff" stopOpacity="0" />
              <stop offset="0.45" stopColor="#ffffff" stopOpacity="0.55" />
              <stop offset="0.55" stopColor="#ffffff" stopOpacity="0" />
            </linearGradient>
            <radialGradient id="st-ice-vignette" cx="0.5" cy="0.5" r="0.62">
              <stop offset="0.72" stopColor="#9fb7d0" stopOpacity="0" />
              <stop offset="1" stopColor="#9fb7d0" stopOpacity="0.3" />
            </radialGradient>
            <pattern id="st-net-mesh" width="0.7" height="0.7" patternUnits="userSpaceOnUse">
              <rect width="0.7" height="0.7" fill="#f6f8fc" />
              <path d="M0 0L.7 .7M.7 0L0 .7" stroke="#a3adc2" strokeWidth="0.08" />
            </pattern>
          </defs>
          {/* layer order: arena -> ice -> markings -> shot marks -> trails -> players -> puck */}
          <Arena colors={colors} teams={teams} boxed={boxed} />
          <g clipPath="url(#st-rink-clip)">
            <rect x="-100" y="-42.5" width="200" height="85" className="st-ice" />
            <rect x="-100" y="-42.5" width="200" height="85" fill="url(#st-ice-scratch)" />
            <rect x="-100" y="-42.5" width="200" height="85" fill="url(#st-ice-sheen)" opacity="0.35" />
            <RinkMarkings colors={colors} teams={teams} />
            <rect x="-100" y="-42.5" width="200" height="85" fill="url(#st-ice-vignette)" />
            {enOf ? <circle cx={(enOf === "home" ? !homeRight : homeRight) ? 90.6 : -90.6} cy="0" r="6" className="st-emptynet" /> : null}
          </g>
          <rect x="-99.55" y="-42.05" width="199.1" height="84.1" rx="27.55" ry="27.55" className="st-rink-kick" />
          <rect x="-100" y="-42.5" width="200" height="85" rx="28" ry="28" className="st-rink-boards" />
          <rect x="-98.6" y="-41.1" width="197.2" height="82.2" rx="26.6" ry="26.6" className="st-rink-inner" />

          <g className="st-shots">{shots.map(shotMark)}</g>
          {faceoff ? (
            <g className="st-fo" aria-hidden="true">
              <circle cx={clampX(faceoff.x)} cy={clampY(faceoff.y)} r="4" />
            </g>
          ) : null}
          {foArrow}
          {hit ? <circle key={hit.key} cx={clampX(hit.x)} cy={clampY(hit.y)} r="2.2" className="st-hitring" /> : null}

          <g className="st-puffs" aria-hidden="true">
            {Array.from({ length: PUFFS }, (_, i) => (
              <circle
                key={i}
                ref={(el) => {
                  puffRef.current[i] = el;
                }}
                r="1"
                style={{ visibility: "hidden" }}
              />
            ))}
          </g>
          <line ref={passRef} className="st-passline" style={{ visibility: "hidden" }} />
          <g className="st-trail" aria-hidden="true">
            {Array.from({ length: TRAIL_LEN - 1 }, (_, i) => (
              <line
                key={i}
                ref={(el) => {
                  trailRef.current[i] = el;
                }}
                strokeWidth={Math.max(0.2, 0.9 - i * 0.06)}
                style={{ visibility: "hidden" }}
              />
            ))}
          </g>

          {onIce.map(marker)}

          <g ref={puckRef} className="st-puck" style={{ visibility: "hidden" }}>
            <ellipse ref={puckShadowRef} className="st-puck__shadow" cx="0.25" cy="0.35" rx={PUCK_R * 1.15} ry={PUCK_R * 0.8} style={{ opacity: 0 }} />
            <g ref={puckBodyRef}>
              <circle ref={haloRef} className="st-puck__halo" r={PUCK_R + 1} />
              <circle r={PUCK_R} fill="#0a0a0a" stroke="#ffffff" strokeWidth="0.26" />
            </g>
          </g>
        </svg>

        {strip}
        {tracking ? (
          <span className="st-estpos" title="Player and puck positions are generated by the sim to illustrate the play. They never affect the result.">
            Estimated positions
          </span>
        ) : (
          <span className="st-estpos is-off" title="The sim sent no tracking data for this stretch">
            No tracking data
          </span>
        )}
        <div className="st-deadbar" aria-hidden="true">
          <span className="st-deadbar__whistle" />
          <b>Whistle</b>
          <small>{stop && STOP_LABEL[stop.reason] ? STOP_LABEL[stop.reason] : "play is dead"}</small>
        </div>
        {replay ? (
          <div className="st-replayframe" aria-hidden="true">
            <i>Replay</i>
            <i>Slow motion</i>
            <i>Replay</i>
            <i>Not live</i>
          </div>
        ) : null}
        <div className="st-legend" data-avoid="" aria-label="Who is who on the rink">
          <span>
            <i className="is-dot" style={{ background: colors?.home?.kit, borderColor: colors?.home?.trim }} />
            {teams?.home?.abbr || "Home"}
          </span>
          <span>
            <i className="is-sq" style={{ background: colors?.away?.kit, borderColor: colors?.away?.trim }} />
            {teams?.away?.abbr || "Away"}
          </span>
          <span>
            <i className="is-g">G</i>goalie
          </span>
          <span>
            <i className="is-puck" />
            puck
          </span>
        </div>
        <button type="button" className={`st-tiltbtn${tilt ? " is-on" : ""}`} data-avoid="" onClick={toggleTilt} title={tilt ? "Switch to the flat top-down rink" : "Switch to the angled 3D rink"}>
          {tilt ? "3D" : "2D"}
        </button>
        {buffering ? <span className="st-buffer" title="Waiting for the sim to send the next stretch of play">buffering</span> : null}

        {replay ? (
          <div className="st-replaybug" data-avoid="">
            <span className="st-replaybug__reel" aria-hidden="true" />
            <b>Replay</b>
            <small>{replay.kind === "save" ? "the save" : "the goal"}</small>
            <button type="button" className="st-btn is-ghost is-sm" onClick={onSkipReplay} title="Back to live">
              Skip
            </button>
          </div>
        ) : null}

        {stop ? (
          <div key={stop.key} className={`st-stop is-${stop.reason}`} data-avoid="">
            <span className="st-stop__whistle" aria-hidden="true" />
            <b>{STOP_LABEL[stop.reason] || "Whistle"}</b>
          </div>
        ) : null}
        {foWin ? (
          <span key={`fw${foWin.key}`} className="st-fowin" style={{ ...toPct(foWin.x, foWin.y), ...colors[foWin.side]?.chip }}>
            {teams?.[foWin.side]?.abbr} win the draw
          </span>
        ) : null}
        {delayed ? (
          <span className="st-delayed" data-avoid="" title="Delayed penalty: play continues until the offending team touches the puck">
            <span className="st-delayed__arm" aria-hidden="true" />
            Delayed penalty · {teams?.[delayed.side]?.abbr || ""}
          </span>
        ) : null}
        {danger ? (
          <span key={danger.key} className="st-danger" data-avoid="" style={toPct(danger.x, danger.y)}>
            {danger.text}
          </span>
        ) : null}

        {dramaText && !replay ? (
          <span key={dramaText.split(" ")[0]} className={`st-drama is-${String(drama || "").split("-")[0]}`} data-avoid="">
            {dramaText}
          </span>
        ) : null}

        {alert ? (
          <div key={alert.id} className={`st-alert is-${alert.severity || "low"}`} data-avoid="">
            <p>{alert.text}</p>
            <div className="st-alert__row">
              {alert.action?.kind ? (
                <button type="button" className="st-btn is-sm" onClick={() => onAlertAction(alert)} title={`Send this order: ${KIND_LABEL[alert.action.kind] || alert.action.kind}`}>
                  {KIND_LABEL[alert.action.kind] || "Do it"}
                </button>
              ) : null}
              <button type="button" className="st-btn is-ghost is-sm" onClick={() => onAlertDismiss(alert.id)} title="Dismiss">
                Dismiss
              </button>
            </div>
          </div>
        ) : statPop ? (
          <div key={statPop.key} className="st-statpop" data-avoid="" style={colors[statPop.side]?.chip}>
            <b>{statPop.name}</b>
            <span>{statPop.text}</span>
          </div>
        ) : null}

        {saveFlash ? (
          <span key={saveFlash.key} className="st-saveflash" style={toPct(saveFlash.x, saveFlash.y)}>
            Save!
          </span>
        ) : null}

        {celebration ? (
          <div key={celebration.seq} className={`st-goalburst is-${celebration.side}`} style={colors[celebration.side]?.chip}>
            <strong>Goal!</strong>
            <em>
              {celebration.abbr ? `${celebration.abbr} · ` : ""}
              {celebration.score}
            </em>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** Lower-third under the rink (D14): never covers the play. */
function Caption({ line, live }) {
  if (!line) return <div className="st-caption is-empty" aria-hidden="true" />;
  const color = line.speaker === "color";
  return (
    <div key={line.id} className={`st-caption is-${line.profile || "calm"} is-${color ? "color" : "pbp"}${live ? " is-live" : " is-old"}`} aria-live="polite">
      <span>{color ? "Analyst" : "Play-by-play"}</span>
      <p>{line.text}</p>
    </div>
  );
}

/** Scorer card (E6), shown beside the replay. Season totals only when the sim sends them. */
function ScorerCard({ card, colors, teams }) {
  if (!card) return null;
  return (
    <div key={card.seq} className={`st-scorer is-${card.side}`} style={colors[card.side]?.chip}>
      <PlayerHeadshot player={card.player} size="md" className="st-scorer__face" teamColors={{ primary: colors[card.side]?.primary, secondary: colors[card.side]?.secondary }} showFlag={false} />
      <div className="st-scorer__txt">
        <small>{teams?.[card.side]?.abbr} goal</small>
        <b>
          {has(card.number) ? `#${card.number} ` : ""}
          {card.name}
        </b>
        {card.season ? <em>{card.season}</em> : null}
        <span>{card.assists?.length ? `Assists: ${card.assists.join(", ")}` : "Unassisted"}</span>
      </div>
    </div>
  );
}

/* ==========================================================================
   Drawers behind the bottom dock: Lines | Tactics | Special teams | Game
   ========================================================================== */
function LinesPanel({ bench, colors, onCoach, live }) {
  const [menu, setMenu] = useState(null);
  const side = bench?.user_side;
  const col = side ? colors[side] : colors.home;
  const players = useMemo(() => (Array.isArray(bench?.players) ? bench.players : []), [bench]);
  const byId = useMemo(() => {
    const m = new Map();
    players.forEach((p) => m.set(String(p.player_id), p));
    return m;
  }, [players]);

  const groups = useMemo(() => {
    const L = bench?.lines || {};
    const on = bench?.on_line || {};
    const out = [];
    const F = Array.isArray(L.F) ? L.F : [];
    const D = Array.isArray(L.D) ? L.D : [];
    const G = Array.isArray(L.G) ? L.G : [];
    if (F.length || D.length || G.length) {
      F.forEach((ids, i) => out.push({ key: `F${i + 1}`, label: `L${i + 1}`, code: `F${i + 1}`, ids: ids || [], on: on.F === i }));
      D.forEach((ids, i) => out.push({ key: `D${i + 1}`, label: `D${i + 1}`, code: `D${i + 1}`, ids: ids || [], on: on.D === i }));
      if (G.length) out.push({ key: "G", label: "G", code: null, ids: G.filter(has), on: false, goalies: true });
    } else if (players.length) {
      const by = new Map();
      players.forEach((p) => {
        const k = p.line_label || "—";
        if (!by.has(k)) by.set(k, []);
        by.get(k).push(p.player_id);
      });
      [...by.keys()].sort().forEach((k) => {
        const code = /^L(\d)$/.test(k) ? `F${k.slice(1)}` : /^D\d$/.test(k) ? k : null;
        out.push({ key: k, label: k, code, ids: by.get(k), on: false, goalies: k === "G" });
      });
    }
    return out.map((g) => {
      const ps = g.ids.map((id) => byId.get(String(id))).filter(Boolean);
      const allOn = ps.length > 0 && ps.every((p) => p.on_ice) && !g.goalies;
      return { ...g, ps, missing: g.ids.length - ps.length, on: g.on || allOn };
    });
  }, [bench, players, byId]);

  if (!bench || !groups.length) return <p className="st-empty">The sim hasn't sent your bench yet. Your lines show up here as soon as it does.</p>;

  return (
    <div className="st-lines" style={col.chip}>
      <div className="st-lines__quick">
        <button type="button" className="st-btn is-ghost" disabled={!live} onClick={() => onCoach("roll_lines", {})} title="Share ice time evenly across all four lines">
          Roll lines
        </button>
        <button type="button" className="st-btn is-ghost" disabled={!live} onClick={() => onCoach("blender", { on: true })} title="Shuffle the wingers across your top three lines (centres stay). Press again for a new mix.">
          Blender
        </button>
        <button type="button" className="st-btn is-ghost" disabled={!live} onClick={() => onCoach("blender", { on: false })} title="Go back to the lines you started the game with">
          Undo blender
        </button>
        <button type="button" className="st-btn is-ghost" disabled={!live} onClick={() => onCoach("shorten_bench", {})} title="Lean on your top two lines and pairs; they tire faster">
          Shorten bench
        </button>
        <span className="st-lines__key">
          Fatigue <Est />: <i className="is-green" /> fresh <i className="is-amber" /> tiring <i className="is-red" /> gassed
        </span>
      </div>
      <div className="st-lines__grid">
        {groups.map((g) => (
          <section key={g.key} className={`st-line${g.on ? " is-on" : ""}${g.goalies ? " is-g" : ""}`}>
            {g.code ? (
              <button type="button" className="st-line__tag" disabled={!live} title={`Send out ${g.label} at the next change`} onClick={() => onCoach("next_line", { line: g.code })}>
                {g.label}
              </button>
            ) : (
              <span className="st-line__tag">{g.label}</span>
            )}
            <div className="st-line__chips">
              {g.ps.map((p) => {
                const open = menu === p.player_id;
                return (
                  <div key={p.player_id} className={`st-pchip${p.on_ice ? " is-ice" : ""}${open ? " is-open" : ""}`}>
                    <button
                      type="button"
                      className="st-pchip__btn"
                      onClick={() => setMenu(open ? null : p.player_id)}
                      disabled={!live || g.goalies}
                      title={`${p.name || p.short}${has(p.toi_sec) ? ` · TOI ${mmss(p.toi_sec)} (est.)` : ""} · ${num(p.g)}G ${num(p.a)}A${has(p.plus_minus) ? ` · ${num(p.plus_minus) > 0 ? "+" : ""}${num(p.plus_minus)}` : ""}`}
                    >
                      <span className="st-pchip__num">{has(p.number) ? p.number : "–"}</span>
                      <span className="st-pchip__name">
                        {p.short || p.name}
                        {p.hot ? <span className="st-pchip__form is-hot" title="Hot tonight">hot</span> : null}
                        {p.cold ? <span className="st-pchip__form is-cold" title="Struggling tonight">cold</span> : null}
                      </span>
                      <Fatigue value={p.fatigue} />
                    </button>
                    {open ? (
                      <div className="st-pchip__menu">
                        <button
                          type="button"
                          className="st-btn is-ghost is-sm"
                          onClick={() => {
                            onCoach("bench_player", { player_id: p.player_id });
                            setMenu(null);
                          }}
                        >
                          {p.benched ? "Back in lineup" : "Sit him"}
                        </button>
                        <button
                          type="button"
                          className="st-btn is-ghost is-sm"
                          onClick={() => {
                            onCoach("bench_player", { player_id: p.player_id, double_shift: !p.double_shift });
                            setMenu(null);
                          }}
                        >
                          {p.double_shift ? "End double-shift" : "Double-shift"}
                        </button>
                      </div>
                    ) : null}
                  </div>
                );
              })}
              {g.missing > 0 ? <span className="st-line__gap">{g.missing} unknown</span> : null}
            </div>
            {g.on ? <span className="st-line__on">On ice</span> : null}
          </section>
        ))}
      </div>
    </div>
  );
}

function Seg({ label, opts, value, onPick, disabled, tip }) {
  return (
    <div className="st-seg" title={tip}>
      <span className="st-seg__label">{label}</span>
      <div className="st-seg__opts" role="radiogroup" aria-label={label}>
        {opts.map(([v, text]) => (
          <button key={v} type="button" role="radio" aria-checked={value === v} className={value === v ? "is-on" : ""} disabled={disabled} onClick={() => onPick(v)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

function TacticsPanel({ coach, live, onCoach, list, dflt }) {
  const tactics = coach?.tactics || {};
  return (
    <div className="st-tac">
      {list.map((t) => (
        <Seg key={t.key} label={t.label} tip={t.tip} opts={t.opts} value={tactics[t.key] || dflt} disabled={!live} onPick={(v) => onCoach("tactics", { [t.key]: v })} />
      ))}
      <p className="st-note">Changes start at the next whistle or line change. The rest of the game is re-simulated from there.</p>
    </div>
  );
}

function StatsTable({ stats, teams, colors, compact }) {
  const a = stats?.away || {};
  const h = stats?.home || {};
  const rows = [];
  const push = (label, key, fmt = (x) => x) => {
    if (!has(a[key]) && !has(h[key])) return;
    rows.push({ label, av: has(a[key]) ? fmt(a[key]) : "–", hv: has(h[key]) ? fmt(h[key]) : "–", an: num(a[key]), hn: num(h[key]) });
  };
  push("Shots on goal", "sog");
  push("Expected goals", "xg", (x) => num(x).toFixed(2));
  push("High-danger chances", "hd_chances");
  if (has(a.pp_opps) || has(h.pp_opps)) {
    rows.push({ label: "Power play", av: `${num(a.pp_goals)}/${num(a.pp_opps)}`, hv: `${num(h.pp_goals)}/${num(h.pp_opps)}`, an: num(a.pp_goals), hn: num(h.pp_goals) });
  }
  push("Attempts blocked", "blocked");
  push("Saves", "saves");
  push("Save %", "sv_pct", fmtPct);
  push("Penalty minutes", "pim");
  if (!rows.length) return <p className="st-empty">Stats appear once the puck drops.</p>;
  return (
    <div className={`st-stats${compact ? " is-compact" : ""}`}>
      <div className="st-stats__head">
        <TeamChip side="away" colors={colors} teams={teams} />
        <span />
        <TeamChip side="home" colors={colors} teams={teams} />
      </div>
      {rows.map((r) => (
        <div key={r.label} className="st-stats__row">
          <b className={r.an > r.hn ? "is-lead" : ""}>{r.av}</b>
          <span>{r.label}</span>
          <b className={r.hn > r.an ? "is-lead" : ""}>{r.hv}</b>
        </div>
      ))}
    </div>
  );
}

function Plays({ events, teams, colors }) {
  const rows = useMemo(
    () =>
      events
        .filter((e) => e && (e.type === "goal" || e.type === "penalty" || e.type === "period_start" || e.type === "period_end" || e.type === "final" || e.type === "coach" || num(e.importance) >= 2))
        .slice(-60)
        .reverse(),
    [events]
  );
  if (!rows.length) return <p className="st-empty">Big plays land here, newest on top.</p>;
  return (
    <ol className="st-plays">
      {rows.map((e) => {
        const col = colors[e.side];
        return (
          <li key={e.seq} className={`st-play is-${e.type}${num(e.importance) >= 3 ? " is-big" : ""}`}>
            <span className="st-play__time">
              {e.clock || ""} <small>{periodName(e.period)}</small>
            </span>
            {e.team_abbr || teams?.[e.side]?.abbr ? (
              <span className="st-play__team" style={col ? { "--st-kit": col.kit, "--st-kit-ink": col.kitInk, "--st-trim": col.trim } : undefined}>
                {e.team_abbr || teams?.[e.side]?.abbr}
              </span>
            ) : (
              <span className="st-play__team is-none" />
            )}
            <p>
              {e.type === "goal" ? <b className="st-play__goal">Goal</b> : null}
              {e.description || String(e.type || "").replace("_", " ")}
              {e.type === "goal" && e.score_after ? (
                <small className="st-play__score">
                  {" "}
                  {teams?.away?.abbr} {e.score_after.away} – {e.score_after.home} {teams?.home?.abbr}
                </small>
              ) : null}
            </p>
          </li>
        );
      })}
    </ol>
  );
}

function GamePanel({ stats, events, teams, colors, isFinal, simToEnd, onSimToEnd, onLeave, onCommit, working, languages, lang, onLang, compact, onCompact, names, onNames }) {
  return (
    <div className="st-gamepanel">
      <section>
        <h3 className="st-h">Box score</h3>
        <StatsTable stats={stats} teams={teams} colors={colors} />
      </section>
      <section className="st-gamepanel__plays">
        <h3 className="st-h">Big plays</h3>
        <Plays events={events} teams={teams} colors={colors} />
      </section>
      <section className="st-gamepanel__side">
        <h3 className="st-h">Game</h3>
        <label className="st-field" title="Language for commentary from now on">
          <span>Commentary</span>
          <select className="st-select" value={lang} onChange={(e) => onLang(e.target.value)}>
            {languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        {onNames ? (
          <Seg
            label="Names on markers"
            tip="Numbers only like a TV tracker; names for the puck carrier, last passer and goalies; or everyone (V)"
            opts={[
              ["off", "Off"],
              ["key", "Key players"],
              ["all", "All"],
            ]}
            value={names}
            onPick={onNames}
          />
        ) : null}
        {onCompact ? (
          <label className="st-check" title="Smaller player markers so the play reads tighter (C)">
            <input type="checkbox" checked={!!compact} onChange={onCompact} /> Compact markers
          </label>
        ) : null}
        <button type="button" className="st-btn is-ghost" onClick={onSimToEnd} disabled={isFinal || simToEnd} title="Skip to the final horn; no more coaching">
          {simToEnd ? "Simming…" : "Sim to end"}
        </button>
        {isFinal ? (
          <button type="button" className="st-btn" disabled={working} onClick={onCommit}>
            {working ? "Saving result & simming the league…" : "Commit result"}
          </button>
        ) : (
          <button type="button" className="st-btn is-ghost" onClick={onLeave} title="Walk out; the rest of the game is fast-simmed">
            Leave game
          </button>
        )}
      </section>
    </div>
  );
}

/* ---------- Help ---------- */
function Help({ colors, teams, onClose }) {
  return (
    <div className="st-modal" role="dialog" aria-modal="true" aria-label="How this screen works" onClick={onClose}>
      <div className="st-help" onClick={(e) => e.stopPropagation()}>
        <header className="st-help__head">
          <h2>How the live game works</h2>
          <button type="button" className="st-btn is-ghost is-sm" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="st-help__grid">
          <section>
            <h3 className="st-h">The screen</h3>
            <p>The rink and the score are the game. The line under the score says what is happening right now. Everything else lives behind the four tabs at the bottom.</p>
            <h3 className="st-h">Rink legend</h3>
            <ul className="st-help__legend">
              <li>
                <svg viewBox="-4 -4 8 8" aria-hidden="true">
                  <circle r="2.8" fill={colors.home.kit} stroke="#071233" strokeWidth="0.6" />
                </svg>
                {teams?.home?.abbr || "Home"} (home): dark circles
              </li>
              <li>
                <svg viewBox="-4 -4 8 8" aria-hidden="true">
                  <rect x="-2.7" y="-2.7" width="5.4" height="5.4" rx="1.3" fill="#fbfbf6" stroke={colors.away.trim} strokeWidth="0.8" />
                </svg>
                {teams?.away?.abbr || "Away"} (away): white squares
              </li>
              <li>
                <i className="st-help__ring" /> Gold ring: has the puck
              </li>
              <li>
                <i className="st-help__trail" /> Cyan streak: where the puck just went
              </li>
              <li>
                <svg viewBox="-4 -4 8 8" aria-hidden="true">
                  <polygon points={starPoints(0, 0, 3)} fill="#ffe23a" stroke="#071233" strokeWidth="0.4" />
                </svg>
                Shot marks: star = goal, white dot = save, dashed = missed or blocked. They fade after a while.
              </li>
              <li>Benches sit outside the top boards, penalty boxes outside the bottom boards. Penalised players skate to the box.</li>
              <li>After a whistle a stamp says why (offside, icing, frozen puck…). At a faceoff the dot is ringed and an arrow shows who won it.</li>
              <li>After goals (and some big saves) an instant replay runs in slow motion. Press R or Esc to skip. Replays are skipped at 4x and 8x.</li>
              <li>Markers show jersey numbers. Names appear for the puck carrier, the last passer and receiver, goalies, and anyone you hover or click (change it with V).</li>
              <li>The puck has a halo that pulses while it's loose, a streak on shots and a dotted line on passes. Positions are estimated by the sim to show the play and never change the result.</li>
            </ul>
          </section>
          <section>
            <h3 className="st-h">Coaching</h3>
            <p>Every order (line, tactic, timeout, goalie) starts at the next whistle or change and the sim re-plays the rest of the game from there. What you have already watched never changes.</p>
            <p>An order can be rejected, for example pulling the goalie on a penalty kill, a second timeout, or a line that isn't dressed. The reason shows next to the dock.</p>
            <ul>
              <li>
                <b>Timeout</b>: one per game. Rests your players and settles momentum.
              </li>
              <li>
                <b>Pull goalie</b>: extra attacker, empty net behind you. Click twice to confirm.
              </li>
              <li>
                <b>Roll lines / Shorten bench</b>: even ice time or lean on your top players.
              </li>
            </ul>
            <h3 className="st-h">Fatigue</h3>
            <p>
              Bars go <span className="st-help__fat is-green">green</span> to <span className="st-help__fat is-amber">amber</span> to <span className="st-help__fat is-red">red</span>. They are estimates from time on ice. Tired players lose races and give up chances.
            </p>
          </section>
          <section>
            <h3 className="st-h">Coaching notes, analytics</h3>
            <p>When a decision is worth making (a line getting caved, their top line out against your fourth, a tired pair, a cold goalie) a card pops up top-right with the numbers behind it and buttons to act. Cards fold into the bell after a few seconds; the bell keeps the full log and can pause the game on critical calls.</p>
            <p>Stats (S) opens the analytics drawer: line and player shot shares (CF%), expected goals, zone starts, ice time, goalies and matchups. Pin it to keep it beside the rink. The strip on the top boards shows who is on and how long the shift has run (amber after 45 s, red after 60 s).</p>
            <h3 className="st-h">Momentum, crowd</h3>
            <p>The rope under the rink is momentum: the knot slides toward the team creating more chances lately.</p>
            <p>The stands bounce with the crowd. Big moments, rivalries and chants make them louder.</p>
            <h3 className="st-h">Sound and language</h3>
            <p>Click Sound on once (browsers need a click before they talk). The booth reads the play in your browser's voices.</p>
            <p>
              Missing a voice? On Windows: Settings → Time &amp; language → Speech → Manage voices → Add voices, pick the language, then restart the browser. Edge also offers natural online voices.
            </p>
          </section>
          <section>
            <h3 className="st-h">Keys</h3>
            <dl className="st-keys">
              <dt>Space</dt>
              <dd>Pause / resume</dd>
              <dt>1–4</dt>
              <dd>Speed 1x, 2x, 4x, 8x</dd>
              <dt>T</dt>
              <dd>Timeout</dd>
              <dt>G</dt>
              <dd>Pull goalie (press twice)</dd>
              <dt>L</dt>
              <dd>Lines drawer</dd>
              <dt>M</dt>
              <dd>Sound on / off</dd>
              <dt>H or ?</dt>
              <dd>This help</dd>
              <dt>R</dt>
              <dd>Skip replay</dd>
              <dt>S</dt>
              <dd>Analytics drawer</dd>
              <dt>N</dt>
              <dd>Coaching log</dd>
              <dt>C</dt>
              <dd>Compact markers</dd>
              <dt>V</dt>
              <dd>Names: key players / all / off</dd>
              <dt>Esc</dt>
              <dd>Close drawer, help or replay</dd>
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   v4: on-ice strip, analytics drawer, coaching notifications
   ========================================================================== */
const OZ_KINDS = ["oz_draws", "oz_draw_line", "faceoff_line", "oz_line"];
const PAUSE_CRIT_KEY = "nhl.live.pauseCritical";
const COMPACT_KEY = "nhl.live.compactMarkers";
const NAMES_KEY = "nhl.live.markerNames";
const NOTIF_KIND_LABEL = {
  fatigue: "Fatigue",
  matchup: "Matchup",
  line_struggling: "Line struggling",
  line_hot: "Line hot",
  goalie: "Goalie",
  pp_unit: "Power play",
  pk_unit: "Penalty kill",
  timeout: "Timeout",
  pull_goalie: "Pull goalie",
  momentum: "Momentum",
  injury: "Injury",
  penalty_trouble: "Penalties",
  opponent_change: "Opponent change",
};
/** 0..1 or 0..100 -> 0..100 */
const pct100 = (v) => (has(v) ? (num(v) <= 1 ? num(v) * 100 : num(v)) : null);
const pctTxt = (v) => (pct100(v) === null ? "–" : `${Math.round(pct100(v))}%`);
const share = (a, b) => (num(a) + num(b) > 0 ? (100 * num(a)) / (num(a) + num(b)) : null);
const fx = (v, d = 1) => (has(v) ? num(v).toFixed(d) : "–");
const signed = (v, d = 0) => (has(v) ? `${num(v) > 0 ? "+" : ""}${num(v).toFixed(d)}` : "–");
const isFDKey = (k) => /^[FD]\d$/.test(String(k || ""));
function surname(roster, pid) {
  const r = roster.get(String(pid)) || {};
  return r.short || (r.name ? String(r.name).split(/\s+/).slice(-1)[0] : `#${pid}`);
}
/** A player object PlayerHeadshot can resolve (id + name + position + team + any headshot fields the sim sends). */
function facePlayer(pid, extra, roster, teams) {
  const r = roster.get(String(pid)) || {};
  const m = { ...r, ...(extra || {}) };
  const side = m.side || r.side;
  return {
    ...m,
    id: pid ?? m.id,
    player_id: pid ?? m.player_id,
    name: m.name || r.name || "",
    position: m.position || m.pos || r.pos || "",
    number: has(m.number) ? m.number : r.number,
    team_id: m.team_id || teams?.[side]?.id || teams?.[side]?.team_id || "",
  };
}
/** Which line keys are on the ice now for a side, from the live on-ice ids. */
function lineKeysOnIce(lines, ids) {
  const set = new Set(ids.map(String));
  let bestF = null;
  let bestD = null;
  let bestS = null;
  for (const l of lines || []) {
    const pids = (l.player_ids || []).map(String);
    if (!pids.length) continue;
    const n = pids.filter((p) => set.has(p)).length;
    const k = String(l.key || "");
    if (/^F/.test(k) && n >= 2 && (!bestF || n > bestF.n)) bestF = { k, n };
    else if (/^D/.test(k) && n >= 1 && (!bestD || n > bestD.n)) bestD = { k, n };
    else if (/^P[PK]/.test(k) && n >= Math.min(4, pids.length) && (!bestS || n > bestS.n)) bestS = { k, n };
  }
  if (bestS) return bestS.k;
  return [bestF?.k, bestD?.k].filter(Boolean).join(" · ");
}

/** Numbers along the top boards (HFM style): who is on, the line keys, and the shift clock. */
function OnIceStrip({ onIce, roster, colors, teams, analytics, shiftStart, gameSec }) {
  const side = (s) => {
    const ids = onIce.filter((p) => p.side === s).map((p) => p.id);
    const sk = ids.filter((id) => upper(roster.get(id)?.pos) !== "G");
    const order = (id) => ({ C: 0, LW: 1, RW: 2, F: 1, W: 1, D: 3, LD: 3, RD: 4, G: 9 }[upper(roster.get(id)?.pos)] ?? 5);
    ids.sort((a, b) => order(a) - order(b));
    let shift = 0;
    sk.forEach((id) => {
      const st = shiftStart.get(id);
      if (has(st)) shift = Math.max(shift, gameSec - st);
    });
    const key = lineKeysOnIce(analytics?.lines?.[s], ids);
    const lvl = shift > 60 ? "is-red" : shift > 45 ? "is-amber" : "";
    return (
      <div className={`st-onice__team is-${s}`}>
        <TeamChip side={s} colors={colors} teams={teams} />
        <span className="st-onice__nums">
          {ids.map((id) => {
            const r = roster.get(id) || {};
            const g = upper(r.pos) === "G";
            return (
              <span key={id} className={`st-onice__n${g ? " is-g" : ""}`} style={colors[s].chip} title={`${r.name || surname(roster, id)}${r.pos ? ` · ${r.pos}` : ""}`}>
                {has(r.number) ? r.number : g ? "G" : "?"}
              </span>
            );
          })}
        </span>
        {key ? <span className="st-onice__key">{key}</span> : null}
        {sk.length ? (
          <span className={`st-onice__shift ${lvl}`} title="Longest current shift on the ice (estimated)">
            {mmss(shift)}
          </span>
        ) : null}
      </div>
    );
  };
  if (!onIce.length) return null;
  return (
    <div className="st-onice" aria-label="On the ice now">
      {side("away")}
      {side("home")}
    </div>
  );
}

/* ---------- analytics drawer ---------- */
function SortTh({ k, label, sort, setSort, est, title }) {
  const on = sort.k === k;
  return (
    <th className={on ? "is-sorted" : ""} title={title}>
      <button type="button" onClick={() => setSort((s) => ({ k, dir: s.k === k ? -s.dir : -1 }))}>
        {label}
        {est ? <Est /> : null}
        {on ? <i>{sort.dir < 0 ? "▼" : "▲"}</i> : null}
      </button>
    </th>
  );
}
function sortRows(rows, sort, get) {
  const g = get || ((r, k) => r[k]);
  return [...rows].sort((a, b) => {
    const x = g(a, sort.k);
    const y = g(b, sort.k);
    if (typeof x === "string" || typeof y === "string") return sort.dir * String(x ?? "").localeCompare(String(y ?? ""), undefined, { numeric: true });
    return sort.dir * (num(x, -1e9) - num(y, -1e9));
  });
}
const PctBar = ({ v }) => {
  const p = pct100(v);
  if (p === null) return <span className="st-pbar is-none">–</span>;
  return (
    <span className={`st-pbar${p >= 55 ? " is-good" : p <= 45 ? " is-bad" : ""}`} title={`${p.toFixed(1)}%`}>
      <i style={{ width: `${Math.max(4, Math.min(100, p))}%` }} />
      <b>{Math.round(p)}</b>
    </span>
  );
};

function LinesTab({ an, side, setSide, userSide, roster, colors, teams, onIceIds, onCoach, live, ozKind }) {
  const [sort, setSort] = useState({ k: "order", dir: 1 });
  const lines = Array.isArray(an?.lines?.[side]) ? an.lines[side] : [];
  const order = ["F1", "F2", "F3", "F4", "D1", "D2", "D3", "PP1", "PP2", "PK1", "PK2"];
  const ice = new Set((onIceIds[side] || []).map(String));
  const rows = lines.map((l) => {
    const pids = (l.player_ids || []).map(String);
    const n = pids.filter((p) => ice.has(p)).length;
    return {
      ...l,
      order: order.indexOf(l.key) >= 0 ? order.indexOf(l.key) : 99,
      cfp: has(l.cf_pct) ? pct100(l.cf_pct) : share(l.cf, l.ca),
      xgp: has(l.xg_pct) ? pct100(l.xg_pct) : share(l.xgf, l.xga),
      gd: num(l.gf) - num(l.ga),
      ozp: share(l.oz_starts, l.dz_starts),
      onNow: pids.length > 0 && n >= Math.min(2, pids.length),
      names: pids.map((p) => surname(roster, p)).join(" · "),
    };
  });
  const sorted = sort.k === "order" ? sortRows(rows, { k: "order", dir: 1 }) : sortRows(rows, sort);
  const mine = side === userSide;
  return (
    <>
      <div className="st-an__bar">
        <div className="st-an__seg" role="radiogroup" aria-label="Team">
          {["home", "away"].map((s) => (
            <button key={s} type="button" role="radio" aria-checked={side === s} className={side === s ? "is-on" : ""} onClick={() => setSide(s)}>
              {teams?.[s]?.abbr || s}
              {s === userSide ? " (you)" : ""}
            </button>
          ))}
        </div>
        {mine ? (
          <span className="st-an__acts">
            <button type="button" className="st-btn is-ghost is-sm" disabled={!live} onClick={() => onCoach("shorten_bench", {})} title="Sit the 4th line and 3rd pair at even strength (F1–F3, D1–D2)">
              Shorten bench to F1–F3
            </button>
            <button type="button" className="st-btn is-ghost is-sm" disabled={!live} onClick={() => onCoach("roll_lines", {})} title="Even ice time for all four lines">
              Roll all four
            </button>
          </span>
        ) : null}
      </div>
      {!rows.length ? (
        <p className="st-empty">No line numbers yet — they build up as shifts are played.</p>
      ) : (
        <div className="st-an__scroll">
          <table className="st-tbl">
            <thead>
              <tr>
                <SortTh k="order" label="Line" sort={sort} setSort={setSort} />
                <th className="is-left">Players</th>
                <SortTh k="toi_sec" label="TOI" est sort={sort} setSort={setSort} />
                <SortTh k="shifts" label="Sh" est sort={sort} setSort={setSort} title="Shifts" />
                <SortTh k="cf" label="CF–CA" sort={sort} setSort={setSort} title="Shot attempts for / against with this line on" />
                <SortTh k="cfp" label="CF%" sort={sort} setSort={setSort} />
                <SortTh k="xgf" label="xGF–xGA" sort={sort} setSort={setSort} />
                <SortTh k="xgp" label="xG%" sort={sort} setSort={setSort} />
                <SortTh k="gd" label="GF–GA" sort={sort} setSort={setSort} />
                <SortTh k="ozp" label="OZ/DZ" sort={sort} setSort={setSort} title="Offensive / defensive zone faceoff starts" />
                <SortTh k="fatigue" label="Fat" est sort={sort} setSort={setSort} title="Average fatigue (estimated)" />
                {mine ? <th /> : null}
              </tr>
            </thead>
            <tbody>
              {sorted.map((l) => (
                <tr key={l.key} className={l.onNow ? "is-ice" : ""}>
                  <td>
                    <span className="st-tbl__key" style={colors[side].chip}>
                      {l.key}
                    </span>
                    {l.onNow ? <span className="st-tbl__on" title="On the ice now" /> : null}
                  </td>
                  <td className="is-left st-tbl__names">{l.names || "–"}</td>
                  <td>{has(l.toi_sec) ? mmss(l.toi_sec) : "–"}</td>
                  <td>{has(l.shifts) ? l.shifts : "–"}</td>
                  <td>
                    {num(l.cf)}–{num(l.ca)}
                  </td>
                  <td>
                    <PctBar v={l.cfp} colors={colors} />
                  </td>
                  <td>
                    {fx(l.xgf, 2)}–{fx(l.xga, 2)}
                  </td>
                  <td>
                    <PctBar v={l.xgp} colors={colors} />
                  </td>
                  <td>
                    {num(l.gf)}–{num(l.ga)}
                  </td>
                  <td>
                    {num(l.oz_starts)}/{num(l.dz_starts)}
                  </td>
                  <td>
                    <Fatigue value={l.fatigue} />
                  </td>
                  {mine ? (
                    <td className="st-tbl__act">
                      {isFDKey(l.key) ? (
                        <button type="button" className="st-btn is-sm" disabled={!live} onClick={() => onCoach("next_line", { line: l.key })} title={`Send ${l.key} out for the next shift`}>
                          Next shift
                        </button>
                      ) : null}
                      {ozKind && /^F\d$/.test(l.key) ? (
                        <button type="button" className="st-btn is-ghost is-sm" disabled={!live} onClick={() => onCoach(ozKind, { line: l.key })} title="Use this line for offensive-zone draws">
                          OZ draws
                        </button>
                      ) : null}
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function PlayersTab({ an, side, setSide, userSide, roster, colors, teams, bench, onCoach, live }) {
  const [sort, setSort] = useState({ k: "toi_sec", dir: -1 });
  const [menu, setMenu] = useState(null);
  const all = an?.players && typeof an.players === "object" ? an.players : {};
  const benchBy = new Map((bench?.players || []).map((p) => [String(p.player_id), p]));
  const rows = Object.entries(all)
    .filter(([id, p]) => p && (p.side || roster.get(id)?.side) === side && upper(roster.get(id)?.pos) !== "G")
    .map(([id, p]) => {
      const r = roster.get(id) || {};
      const fo = num(p.fo_w) + num(p.fo_l);
      return {
        ...p,
        id,
        name: r.short || r.name || `#${id}`,
        number: r.number,
        pos: r.pos,
        cfp: has(p.cf_pct) ? pct100(p.cf_pct) : share(p.cf, p.ca),
        xgd: has(p.xgf) || has(p.xga) ? num(p.xgf) - num(p.xga) : null,
        fop: fo ? (100 * num(p.fo_w)) / fo : null,
        pts: has(p.pts) ? p.pts : num(p.g) + num(p.a),
      };
    });
  const sorted = sortRows(rows, sort);
  const mine = side === userSide;
  return (
    <>
      <div className="st-an__bar">
        <div className="st-an__seg" role="radiogroup" aria-label="Team">
          {["home", "away"].map((s) => (
            <button key={s} type="button" role="radio" aria-checked={side === s} className={side === s ? "is-on" : ""} onClick={() => setSide(s)}>
              {teams?.[s]?.abbr || s}
              {s === userSide ? " (you)" : ""}
            </button>
          ))}
        </div>
        {mine ? <span className="st-note">Click one of your players to sit him or double-shift him.</span> : null}
      </div>
      {!rows.length ? (
        <p className="st-empty">No player numbers yet.</p>
      ) : (
        <div className="st-an__scroll">
          <table className="st-tbl is-players">
            <thead>
              <tr>
                <SortTh k="name" label="Player" sort={sort} setSort={setSort} />
                <SortTh k="toi_sec" label="TOI" est sort={sort} setSort={setSort} />
                <SortTh k="shifts" label="Sh" est sort={sort} setSort={setSort} title="Shifts" />
                <SortTh k="avg_shift_sec" label="Avg" est sort={sort} setSort={setSort} title="Average shift" />
                <SortTh k="current_shift_sec" label="Now" est sort={sort} setSort={setSort} title="Current shift" />
                <SortTh k="pts" label="G-A-P" sort={sort} setSort={setSort} />
                <SortTh k="sog" label="SOG" sort={sort} setSort={setSort} />
                <SortTh k="cfp" label="CF%" sort={sort} setSort={setSort} />
                <SortTh k="xgd" label="xG±" sort={sort} setSort={setSort} />
                <SortTh k="plus_minus" label="+/-" sort={sort} setSort={setSort} />
                <SortTh k="hits" label="Hit" sort={sort} setSort={setSort} />
                <SortTh k="blocks" label="Blk" sort={sort} setSort={setSort} />
                <SortTh k="fop" label="FO%" sort={sort} setSort={setSort} />
                <SortTh k="fatigue" label="Fat" est sort={sort} setSort={setSort} />
                <SortTh k="rating_tonight" label="GS" sort={sort} setSort={setSort} title="Game score tonight, 0–10" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((p) => {
                const b = benchBy.get(p.id) || {};
                const open = menu === p.id;
                const gs = has(p.rating_tonight) ? num(p.rating_tonight) : null;
                return (
                  <tr key={p.id} className={`${mine && live ? "is-click" : ""}${b.on_ice ? " is-ice" : ""}${open ? " is-open" : ""}`} onClick={() => mine && live && setMenu(open ? null : p.id)}>
                    <td className="is-left st-tbl__player">
                      <span className="st-tbl__num" style={colors[side].chip}>
                        {has(p.number) ? p.number : "–"}
                      </span>
                      {p.name}
                      {p.pos ? <small>{p.pos}</small> : null}
                      {open ? (
                        <span className="st-tbl__menu" onClick={(e) => e.stopPropagation()}>
                          <button
                            type="button"
                            className="st-btn is-ghost is-sm"
                            onClick={() => {
                              onCoach("bench_player", { player_id: p.id });
                              setMenu(null);
                            }}
                          >
                            {b.benched ? "Back in lineup" : "Sit him"}
                          </button>
                          <button
                            type="button"
                            className="st-btn is-ghost is-sm"
                            onClick={() => {
                              onCoach("bench_player", { player_id: p.id, double_shift: !b.double_shift });
                              setMenu(null);
                            }}
                          >
                            {b.double_shift ? "End double-shift" : "Double-shift"}
                          </button>
                        </span>
                      ) : null}
                    </td>
                    <td>{has(p.toi_sec) ? mmss(p.toi_sec) : "–"}</td>
                    <td>{has(p.shifts) ? p.shifts : "–"}</td>
                    <td>{has(p.avg_shift_sec) ? `${Math.round(num(p.avg_shift_sec))}s` : "–"}</td>
                    <td className={num(p.current_shift_sec) > 60 ? "is-red" : num(p.current_shift_sec) > 45 ? "is-amber" : ""}>{num(p.current_shift_sec) > 0 ? `${Math.round(num(p.current_shift_sec))}s` : "–"}</td>
                    <td>
                      {num(p.g)}-{num(p.a)}-{num(p.pts)}
                    </td>
                    <td>{has(p.sog) ? p.sog : "–"}</td>
                    <td>{p.cfp === null ? "–" : `${Math.round(p.cfp)}%`}</td>
                    <td>{p.xgd === null ? "–" : signed(p.xgd, 2)}</td>
                    <td>{has(p.plus_minus) ? signed(p.plus_minus) : "–"}</td>
                    <td>{has(p.hits) ? p.hits : "–"}</td>
                    <td>{has(p.blocks) ? p.blocks : "–"}</td>
                    <td>{p.fop === null ? "–" : `${Math.round(p.fop)}%`}</td>
                    <td>
                      <Fatigue value={p.fatigue} />
                    </td>
                    <td>{gs === null ? "–" : <span className={`st-gs${gs >= 7 ? " is-hi" : gs < 4 ? " is-lo" : ""}`}>{gs.toFixed(1)}</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function TeamTab({ an, colors, teams }) {
  const h = an?.teams?.home;
  const a = an?.teams?.away;
  if (!h || !a) return <p className="st-empty">Team numbers arrive with the next update from the sim.</p>;
  const ratio = (s) => {
    const m = /^(\d+)\s*\/\s*(\d+)$/.exec(String(s || ""));
    return m ? { n: Number(m[1]), d: Number(m[2]) } : null;
  };
  const rows = [];
  const add = (label, av, hv, aTxt, hTxt, title) => rows.push({ label, av: num(av), hv: num(hv), aTxt, hTxt, title });
  const fo = (t) => (num(t.fo_w) + num(t.fo_l) ? (100 * num(t.fo_w)) / (num(t.fo_w) + num(t.fo_l)) : null);
  const cfA = has(a.cf_pct) ? pct100(a.cf_pct) : share(a.cf, a.ca);
  const cfH = has(h.cf_pct) ? pct100(h.cf_pct) : share(h.cf, h.ca);
  if (cfA !== null || cfH !== null) add("CF%", cfA, cfH, pctTxt(cfA), pctTxt(cfH), `Shot attempts ${num(a.cf)}–${num(h.cf)}`);
  const xA = has(a.xg_pct) ? pct100(a.xg_pct) : share(a.xgf, a.xga);
  const xH = has(h.xg_pct) ? pct100(h.xg_pct) : share(h.xgf, h.xga);
  if (xA !== null || xH !== null) add("xG%", xA, xH, `${pctTxt(xA)} · ${fx(a.xgf, 2)}`, `${fx(h.xgf, 2)} · ${pctTxt(xH)}`);
  if (has(a.hdcf) || has(h.hdcf)) add("High-danger CF", a.hdcf, h.hdcf, num(a.hdcf), num(h.hdcf));
  if (has(a.sf) || has(h.sf)) add("Shots on goal", a.sf, h.sf, num(a.sf), num(h.sf));
  const ppA = ratio(a.pp);
  const ppH = ratio(h.pp);
  if (ppA || ppH) add("Power play", ppA ? ppA.n : 0, ppH ? ppH.n : 0, a.pp || "–", h.pp || "–");
  const pkA = ratio(a.pk);
  const pkH = ratio(h.pk);
  if (pkA || pkH) add("Penalty kill", pkA && pkA.d ? pkA.n / pkA.d : 0, pkH && pkH.d ? pkH.n / pkH.d : 0, a.pk || "–", h.pk || "–");
  const fA = fo(a);
  const fH = fo(h);
  if (fA !== null || fH !== null) add("Faceoffs", fA, fH, pctTxt(fA), pctTxt(fH), `${num(a.fo_w)}–${num(h.fo_w)} won`);
  if (has(a.pdo) || has(h.pdo)) add("PDO", a.pdo, h.pdo, fx(a.pdo, 3), fx(h.pdo, 3), "Shooting % + save %, luck indicator (1.000 = average)");
  if (has(a.hits) || has(h.hits)) add("Hits", a.hits, h.hits, num(a.hits), num(h.hits));
  if (has(a.blocks) || has(h.blocks)) add("Blocks", a.blocks, h.blocks, num(a.blocks), num(h.blocks));
  return (
    <div className="st-mirror" style={{ "--st-home-bg": colors.home.kit, "--st-away-line": colors.away.trim }}>
      <div className="st-mirror__head">
        <TeamChip side="away" colors={colors} teams={teams} />
        <span />
        <TeamChip side="home" colors={colors} teams={teams} />
      </div>
      {rows.map((r) => {
        const tot = Math.abs(r.av) + Math.abs(r.hv);
        const aw = tot ? (100 * Math.abs(r.av)) / tot : 50;
        return (
          <div key={r.label} className="st-mirror__row" title={r.title}>
            <b className={r.av > r.hv ? "is-lead" : ""}>{r.aTxt}</b>
            <span className="st-mirror__mid">
              <small>{r.label}</small>
              <span className="st-mirror__bar">
                <i className="is-away" style={{ width: `${aw}%` }} />
                <i className="is-home" style={{ width: `${100 - aw}%` }} />
              </span>
            </span>
            <b className={r.hv > r.av ? "is-lead" : ""}>{r.hTxt}</b>
          </div>
        );
      })}
    </div>
  );
}

function GoaliesTab({ an, roster, colors, teams }) {
  const g = an?.goalies && typeof an.goalies === "object" ? Object.entries(an.goalies) : [];
  if (!g.length) return <p className="st-empty">Goalie numbers arrive after the first shots.</p>;
  return (
    <div className="st-an__scroll">
      <table className="st-tbl">
        <thead>
          <tr>
            <th className="is-left">Goalie</th>
            <th>SA</th>
            <th>SV</th>
            <th>GA</th>
            <th>SV%</th>
            <th title="Expected goals against">xGA</th>
            <th title="Goals saved above expected">GSAx</th>
            <th title="High-danger save %">HDSV%</th>
          </tr>
        </thead>
        <tbody>
          {g.map(([id, x]) => {
            const s = x.side || roster.get(id)?.side || "home";
            return (
              <tr key={id}>
                <td className="is-left st-tbl__player">
                  <TeamChip side={s} colors={colors} teams={teams} />
                  {roster.get(id)?.name || surname(roster, id)}
                </td>
                <td>{num(x.sa)}</td>
                <td>{num(x.sv)}</td>
                <td>{num(x.ga)}</td>
                <td>{has(x.sv_pct) ? fmtPct(x.sv_pct) : "–"}</td>
                <td>{fx(x.xga, 2)}</td>
                <td className={num(x.gsax) > 0.5 ? "is-good" : num(x.gsax) < -0.5 ? "is-bad" : ""}>{has(x.gsax) ? signed(x.gsax, 2) : "–"}</td>
                <td>{has(x.hd_sv_pct) ? fmtPct(x.hd_sv_pct) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MatchupsTab({ an, colors, teams, userSide }) {
  const m = an?.matchups || {};
  const now = m.on_ice_now || {};
  const top = Array.isArray(m.top) ? m.top : [];
  return (
    <div className="st-match">
      <div className="st-match__now">
        <small>On the ice now</small>
        <div>
          <span className="st-tbl__key" style={colors.home.chip}>
            {teams?.home?.abbr} {now.home_line || "–"}
          </span>
          <b>vs</b>
          <span className="st-tbl__key" style={colors.away.chip}>
            {teams?.away?.abbr} {now.away_line || "–"}
          </span>
        </div>
      </div>
      {top.length ? (
        <table className="st-tbl">
          <thead>
            <tr>
              <th className="is-left">{teams?.home?.abbr}</th>
              <th className="is-left">{teams?.away?.abbr}</th>
              <th>
                TOI
                <Est />
              </th>
              <th title={`Shot-attempt share for ${teams?.home?.abbr || "home"} in this matchup`}>CF% {teams?.home?.abbr}</th>
            </tr>
          </thead>
          <tbody>
            {top.map((r, i) => (
              <tr key={`${r.home_line}-${r.away_line}-${i}`} className={r.home_line === now.home_line && r.away_line === now.away_line ? "is-ice" : ""}>
                <td className="is-left">{r.home_line}</td>
                <td className="is-left">{r.away_line}</td>
                <td>{has(r.toi_sec) ? mmss(r.toi_sec) : "–"}</td>
                <td>
                  <PctBar v={r.cf_pct_home} colors={colors} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="st-empty">Matchups build up as lines face each other.</p>
      )}
      {userSide ? <p className="st-note">Tip: send a line out with "Next shift" on the Lines tab when you like a matchup.</p> : null}
    </div>
  );
}

const AN_TABS = [
  ["lines", "Lines"],
  ["players", "Players"],
  ["team", "Team"],
  ["goalies", "Goalies"],
  ["matchups", "Matchups"],
];
function StatsPanel({ an, roster, colors, teams, userSide, bench, onIceIds, pinned, onPin, onClose, onCoach, live, ozKind }) {
  const [tab, setTab] = useState("lines");
  const [side, setSide] = useState(userSide || "home");
  return (
    <aside className={`st-an${pinned ? " is-pinned" : ""}`} aria-label="Game analytics">
      <header className="st-an__head">
        <h3>Analytics</h3>
        <nav className="st-an__tabs">
          {AN_TABS.map(([k, label]) => (
            <button key={k} type="button" className={tab === k ? "is-on" : ""} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </nav>
        <button type="button" className={`st-btn is-ghost is-sm${pinned ? " is-on" : ""}`} onClick={onPin} title={pinned ? "Unpin: float over the rink" : "Pin beside the rink"} aria-pressed={pinned}>
          {pinned ? "Pinned" : "Pin"}
        </button>
        <button type="button" className="st-btn is-ghost is-sm" onClick={onClose} title="Close (S)">
          Close
        </button>
      </header>
      {an?.sample_note ? <p className="st-an__note">{an.sample_note}</p> : null}
      {!an ? (
        <p className="st-empty">Analytics arrive with the next update from the sim.</p>
      ) : tab === "lines" ? (
        <LinesTab an={an} side={side} setSide={setSide} userSide={userSide} roster={roster} colors={colors} teams={teams} onIceIds={onIceIds} onCoach={onCoach} live={live} ozKind={ozKind} />
      ) : tab === "players" ? (
        <PlayersTab an={an} side={side} setSide={setSide} userSide={userSide} roster={roster} colors={colors} teams={teams} bench={bench} onCoach={onCoach} live={live} />
      ) : tab === "team" ? (
        <TeamTab an={an} colors={colors} teams={teams} />
      ) : tab === "goalies" ? (
        <GoaliesTab an={an} roster={roster} colors={colors} teams={teams} />
      ) : (
        <MatchupsTab an={an} colors={colors} teams={teams} userSide={userSide} />
      )}
      <p className="st-an__foot">
        Counts come from the game's events. TOI, shifts and fatigue are estimated from the on-ice track (<Est />).
      </p>
    </aside>
  );
}

/* ---------- coaching notifications ---------- */
function NotifCard({ n, onAct, onDismiss, live }) {
  const acts = Array.isArray(n.actions) ? n.actions.slice(0, 3) : [];
  // Routine advice clears itself so it never sits over the rink; critical calls (priority 3) wait for the coach.
  const dismissRef = useRef(onDismiss);
  dismissRef.current = onDismiss;
  const nid = n.id;
  const prio = num(n.priority, 2);
  useEffect(() => {
    if (prio >= 3) return undefined;
    const id = setTimeout(() => dismissRef.current({ id: nid }), 9000);
    return () => clearTimeout(id);
  }, [nid, prio]);
  return (
    <div className={`st-ncard is-p${num(n.priority, 2)}`} role="alert">
      <div className="st-ncard__top">
        <span className="st-ncard__kind">{NOTIF_KIND_LABEL[n.kind] || n.kind || "Coach"}</span>
        <button type="button" className="st-ncard__x" onClick={() => onDismiss(n)} aria-label="Dismiss" title="Dismiss">
          ×
        </button>
      </div>
      <b className="st-ncard__title">{n.title}</b>
      {n.body ? <p>{n.body}</p> : null}
      {acts.length ? (
        <div className="st-ncard__acts">
          {acts.map((a, i) => (
            <button key={i} type="button" className={`st-btn is-sm${i ? " is-ghost" : ""}`} disabled={!live || !a?.order?.kind} onClick={() => onAct(n, a)}>
              {a.label || KIND_LABEL[a?.order?.kind] || "Do it"}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function NotifLog({ list, statusOf, onAct, onDismiss, live, pauseCritical, setPauseCritical, onClose }) {
  const [kind, setKind] = useState("all");
  const kinds = [...new Set(list.map((n) => n.kind).filter(Boolean))];
  const rows = list.filter((n) => kind === "all" || n.kind === kind);
  const whenOf = (n) => {
    if (!has(n.t)) return "";
    const c = clockFromT(n.t);
    return `${periodName(c.period)} ${mmss(c.left)}`;
  };
  return (
    <div className="st-nlog" role="dialog" aria-label="Coaching notifications">
      <header>
        <b>Coaching log</b>
        <button type="button" className="st-btn is-ghost is-sm" onClick={onClose}>
          Close
        </button>
      </header>
      <label className="st-check">
        <input type="checkbox" checked={pauseCritical} onChange={() => setPauseCritical(!pauseCritical)} /> Pause the game on critical decisions
      </label>
      {kinds.length > 1 ? (
        <div className="st-nlog__filters">
          {["all", ...kinds].map((k) => (
            <button key={k} type="button" className={kind === k ? "is-on" : ""} onClick={() => setKind(k)}>
              {k === "all" ? "All" : NOTIF_KIND_LABEL[k] || k}
            </button>
          ))}
        </div>
      ) : null}
      {!rows.length ? (
        <p className="st-empty">No coaching notes yet.</p>
      ) : (
        <ol className="st-nlog__list">
          {rows.map((n) => {
            const st = statusOf(n);
            const acts = Array.isArray(n.actions) ? n.actions.slice(0, 3) : [];
            return (
              <li key={n.id} className={`is-p${num(n.priority, 1)} is-${st}`}>
                <div className="st-nlog__meta">
                  <span>{whenOf(n)}</span>
                  <span>{NOTIF_KIND_LABEL[n.kind] || n.kind}</span>
                  <em className={`st-nstat is-${st}`}>{st}</em>
                </div>
                <b>{n.title}</b>
                {n.body ? <p>{n.body}</p> : null}
                {st === "new" && acts.length ? (
                  <div className="st-ncard__acts">
                    {acts.map((a, i) => (
                      <button key={i} type="button" className={`st-btn is-sm${i ? " is-ghost" : ""}`} disabled={!live || !a?.order?.kind} onClick={() => onAct(n, a)}>
                        {a.label || "Do it"}
                      </button>
                    ))}
                    <button type="button" className="st-btn is-ghost is-sm" onClick={() => onDismiss(n)}>
                      Dismiss
                    </button>
                  </div>
                ) : null}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

/* ==========================================================================
   Screen
   ========================================================================== */
const DOCK_TABS = [
  ["lines", "Lines", "Your lines, fatigue and who's on the ice (L)"],
  ["tactics", "Tactics", "Forecheck, defensive posture and shooting"],
  ["special", "Special teams", "Power play and penalty kill style"],
  ["game", "Game", "Box score, big plays, language, sim to end, leave"],
];
const TabIcon = ({ k }) => (
  <svg className="st-tab__ico" viewBox="0 0 20 20" aria-hidden="true">
    {k === "lines" ? <path d="M3 5h14M3 10h14M3 15h14" /> : null}
    {k === "tactics" ? <path d="M4 4l4 4M8 4L4 8M12 13a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0M6 16c3 0 5-2 6-6l-2 1m2-1l1 2" /> : null}
    {k === "special" ? <path d="M3 7h6M6 4v6M11 13h6" /> : null}
    {k === "game" ? <path d="M4 3h12v14H4zM7 7h6M7 10h6M7 13h4" /> : null}
    {k === "stats" ? <path d="M3 16l4-5 3 3 7-9M3 3v14h14" /> : null}
  </svg>
);
const ordinal = (n) => {
  const v = n % 100;
  return `${n}${v >= 11 && v <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th"}`;
};
/** Short friendly text for a coach order (D8), never the server's technical sentence. */
function friendlyOrder(o) {
  const kind = o.kind;
  const p = o.payload || {};
  let what = KIND_LABEL[kind] || "Order";
  if (kind === "tactics") {
    const key = Object.keys(p)[0];
    const def = [...TACTICS, ...SPECIAL].find((t) => t.key === key);
    const opt = def?.opts.find(([v]) => v === p[key]);
    if (def) what = `${def.label}: ${opt ? opt[1] : p[key]}`;
  } else if (kind === "pull_goalie") what = p.pull === false ? "Goalie back in" : "Goalie pulled";
  else if (kind === "next_line" && p.line) what = `${String(p.line).replace("F", "L")} up next`;
  else if (kind === "bench_player") what = p.double_shift ? "Double-shift" : p.double_shift === false ? "Double-shift off" : "Bench change";
  const st = o.status || "queued";
  if (st === "rejected") return { status: st, text: `${what} — not allowed${o.reason ? `: ${firstSentence(o.reason, 70)}` : ""}` };
  if (st === "applying") return { status: st, text: `${what} — applying…` };
  if (st === "applied") return { status: st, text: `${what} — in effect` };
  return { status: st, text: `${what} — at the next whistle` };
}
/** Chants only when they fit the game (crowd lines can lag the score). */
function chantFits(text, homeDiff) {
  const t = String(text || "");
  if (/one more|tie (it|this)|need one/i.test(t)) return homeDiff === -1;
  if (/comeback|come back/i.test(t)) return homeDiff < 0;
  if (/sweep|so easy|scoreboard/i.test(t)) return homeDiff >= 2;
  return true;
}
function parseDot(dot) {
  if (Array.isArray(dot) && dot.length >= 2) return [num(dot[0]), num(dot[1])];
  if (dot && typeof dot === "object" && has(dot.x)) return [num(dot.x), num(dot.y)];
  if (typeof dot === "string" && /cent/i.test(dot)) return [0, 0];
  return null;
}

export default function SimTracker({ sessionId: sessionIdProp, onExit }) {
  const sessionId = sessionIdProp || getFranchiseSessionId();

  // Server state (ahead of the viewer) and displayed state (at the playback head).
  const [server, setServer] = useState(null);
  const [disp, setDisp] = useState(null);
  const [events, setEvents] = useState([]);
  const [lines, setLines] = useState([]);
  const [gameSec, setGameSec] = useState(0); // game clock at the head (gc)
  const [headPt, setHeadPt] = useState(0); // presentation time at the head (coarse, ~1 Hz)
  const [onIce, setOnIce] = useState([]);
  const [phase, setPhase] = useState({ ph: "play", pos: null, zone: null });
  const [buffering, setBuffering] = useState(false);
  const [hasTracking, setHasTracking] = useState(false);

  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    try {
      window.dispatchEvent(new CustomEvent("nhl-live-sim", { detail: true }));
    } catch {
      /* ignore */
    }
    return () => {
      try {
        window.dispatchEvent(new CustomEvent("nhl-live-sim", { detail: false }));
      } catch {
        /* ignore */
      }
    };
  }, []);
  const [paceMode, setPaceMode] = useState(() => {
    try {
      return window.localStorage.getItem(PACE_KEY) === "real" ? "real" : "broadcast";
    } catch {
      return "broadcast";
    }
  });
  const paceRef = useRef(PACES[paceMode] || 0.6);
  paceRef.current = PACES[paceMode] || 0.6;
  const togglePace = () => {
    const next = paceMode === "real" ? "broadcast" : "real";
    setPaceMode(next);
    try {
      window.localStorage.setItem(PACE_KEY, next);
    } catch {
      /* ignore */
    }
  };
  const [held, setHeld] = useState(false);
  const [simToEnd, setSimToEnd] = useState(false);
  const [booting, setBooting] = useState(true);
  const [err, setErr] = useState("");
  const [working, setWorking] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [lastOrder, setLastOrder] = useState(null);
  const [toast, setToast] = useState(null);
  const [dismissed, setDismissed] = useState(() => new Set());
  const [celebration, setCelebration] = useState(null);
  const [scorer, setScorer] = useState(null);
  const [saveFlash, setSaveFlash] = useState(null);
  const [stop, setStop] = useState(null);
  const [faceoff, setFaceoff] = useState(null);
  const [foWin, setFoWin] = useState(null);
  const [danger, setDanger] = useState(null);
  const [hit, setHit] = useState(null);
  const [delayed, setDelayed] = useState(null);
  const [statPop, setStatPop] = useState(null);
  const [replay, setReplay] = useState(null);
  const [periodCard, setPeriodCard] = useState(null);
  const [banner, setBanner] = useState(null);
  const [intro, setIntro] = useState(false);
  const [goalPulse, setGoalPulse] = useState(0);
  const [hornPulse, setHornPulse] = useState(0);
  const [lang, setLang] = useState(readLang);
  const [soundOn, setSoundOn] = useState(false);
  const [soundMenu, setSoundMenu] = useState(false);
  const [prefs, setPrefs] = useState(readSoundPrefs);
  const [drawer, setDrawer] = useState(null);
  const [help, setHelp] = useState(false);
  const [pullArmed, setPullArmed] = useState(false);
  const [ttsStatus, setTtsStatus] = useState(null);
  const [puckDropped, setPuckDropped] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const [statsPinned, setStatsPinned] = useState(false);
  const readFlag = (k) => {
    try {
      return window.localStorage.getItem(k) === "1";
    } catch {
      return false;
    }
  };
  const writeFlag = (k, v) => {
    try {
      window.localStorage.setItem(k, v ? "1" : "0");
    } catch {
      /* ignore */
    }
  };
  const [compact, setCompactState] = useState(() => readFlag(COMPACT_KEY));
  const [names, setNamesState] = useState(() => {
    try {
      const v = window.localStorage.getItem(NAMES_KEY);
      return v === "off" || v === "all" ? v : "key";
    } catch {
      return "key";
    }
  });
  const setNames = (v) => {
    setNamesState(v);
    try {
      window.localStorage.setItem(NAMES_KEY, v);
    } catch {
      /* ignore */
    }
  };
  const [pauseCritical, setPauseCriticalState] = useState(() => readFlag(PAUSE_CRIT_KEY));
  const [nStatus, setNStatus] = useState({});
  const [cards, setCards] = useState([]);
  const [logOpen, setLogOpen] = useState(false);
  const [logSeen, setLogSeen] = useState(() => new Set());
  const seenNotif = useRef(new Set());

  const serverRef = useRef(null);
  const dispRef = useRef(null);
  const sinceRef = useRef(0);
  const speedRef = useRef(speed);
  const heldRef = useRef(held);
  const introRef = useRef(false);
  const langRef = useRef(lang);
  const simToEndRef = useRef(false);
  const soundRef = useRef(false);
  const sfxRef = useRef(() => {});
  const rinkApi = useRef(null);
  const timers = useRef({});
  const rosterRef = useRef(new Map());
  const clipsRef = useRef(new Map());
  const seenLines = useRef(new Set());
  const seenEvents = useRef(new Set());
  const eventsRef = useRef(events);
  eventsRef.current = events;
  const pb = useRef({
    head: 0,
    vel: 0,
    bufferedTo: 0,
    simGame: 0,
    frames: [],
    hasMotion: false,
    started: false,
    pendEvents: [],
    pendLines: [],
    pendSnaps: [],
    fetching: false,
    lastFetch: 0,
    backoffUntil: 0,
    trail: [],
    trailAt: 0,
    sig: "",
    shownSec: -1,
    shownPt: -1,
    ph: "play",
    pos: null,
    zone: null,
    gc: 0,
    buffering: false,
    sides: new Map(),
    ready: false,
    lastLayout: 0,
    lastCarrier: null,
    passAt: 0,
    boardsAt: 0,
    puckIn: true,
    lastStopReason: null,
    lastStopPt: -1e9,
    stopCount: 0,
    foDot: null,
    pk: null,
    replay: null,
    pendingReplay: null,
    lastSaveReplay: -1e9,
    lastPopGc: -1e9,
    shots: new Map(),
    posOut: new Map(),
    shiftStart: new Map(),
    liveSig: "",
    carPrev: null,
    carV: [0, 0],
    prevC: null,
    pass: null,
    recent: new Map(),
    recentSet: new Set(),
    puffs: [],
    spFrame: null,
  });
  speedRef.current = speed;
  heldRef.current = held;
  introRef.current = intro;
  langRef.current = lang;
  simToEndRef.current = simToEnd;
  soundRef.current = soundOn && prefs.voice;

  const flash = useCallback((key, fn, ms) => {
    clearTimeout(timers.current[key]);
    timers.current[key] = setTimeout(fn, ms);
  }, []);
  useEffect(
    () => () => {
      Object.values(timers.current).forEach(clearTimeout);
      clipsRef.current.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    },
    []
  );

  /* ---------- neural clip prefetch (lines arrive ahead of reveal) ---------- */
  const prefetchClip = useCallback((line) => {
    const src = line.audio_url ? resolveAudioUrl(line.audio_url) : line.audio_key ? liveTtsUrl(line.audio_key) : "";
    const clips = clipsRef.current;
    if (!src || !has(line.id) || clips.has(line.id)) return;
    const rec = { state: "loading", url: null };
    clips.set(line.id, rec);
    let tries = 0;
    const go = async () => {
      tries += 1;
      try {
        const r = await fetch(src);
        if (r.status === 202) {
          if (tries < 30) setTimeout(go, 400);
          else rec.state = "failed";
          return;
        }
        if (!r.ok) {
          rec.state = "failed";
          return;
        }
        const blob = await r.blob();
        rec.url = URL.createObjectURL(blob);
        rec.state = "ready";
      } catch {
        if (tries < 3) setTimeout(go, 700);
        else rec.state = "failed";
      }
    };
    go();
    while (clips.size > 80) {
      const k = clips.keys().next().value;
      const old = clips.get(k);
      if (old?.url) URL.revokeObjectURL(old.url);
      clips.delete(k);
    }
  }, []);

  /* ---------- intake ---------- */
  const mergeServer = useCallback((s) => {
    if (!s || typeof s !== "object") return;
    const next = { ...(serverRef.current || {}), ...s };
    serverRef.current = next;
    setServer(next);
  }, []);

  const queueSnapshot = useCallback(
    (s, at) => {
      if (!s || typeof s !== "object") return;
      const P = pb.current;
      P.pendSnaps.push({ t: num(at, P.head), s });
      P.pendSnaps.sort((a, b) => a.t - b.t);
      if (Array.isArray(s.commentary)) {
        s.commentary.forEach((c) => {
          if (!c || !has(c.id) || seenLines.current.has(c.id)) return;
          seenLines.current.add(c.id);
          P.pendLines.push({ ...c, _t: ptOf(c, num(at, P.head)) });
          if (c.audio_url || c.audio_key) prefetchClip(c);
        });
        P.pendLines.sort((a, b) => a._t - b._t || num(a.seq) - num(b.seq));
      }
      if (s.tracking?.frames?.length) setHasTracking(true);
    },
    [prefetchClip]
  );

  const queueEvents = useCallback((list) => {
    if (!Array.isArray(list)) return;
    const P = pb.current;
    list.forEach((e) => {
      if (!e || seenEvents.current.has(e.seq)) return;
      seenEvents.current.add(e.seq);
      P.pendEvents.push({ ...e, _p: ptOf(e) });
    });
    P.pendEvents.sort((a, b) => a._p - b._p || num(a.seq) - num(b.seq));
  }, []);

  const appendFrames = useCallback((frames) => {
    const P = pb.current;
    if (!frames.length) return;
    const lastP = P.frames.length ? P.frames[P.frames.length - 1]._p : -Infinity;
    const add = [];
    for (const f of frames) {
      if (!f) continue;
      const p = ptOf(f, NaN);
      if (!Number.isFinite(p) || p <= lastP + 1e-6) continue;
      add.push({ ...f, _p: p, _g: has(f.gc) ? num(f.gc) : num(f.t, p) });
    }
    if (!add.length) return;
    if (!P.started) {
      // First real frames: start the head at the first one (pt can differ from the game clock).
      P.head = add[0]._p;
      P.started = true;
    }
    P.frames.push(...add);
    setHasTracking(true);
  }, []);

  /* ---------- boot ---------- */
  useEffect(() => {
    let dead = false;
    (async () => {
      setBooting(true);
      let s = null;
      try {
        s = await liveStart(sessionId, { lang: langRef.current });
      } catch (e) {
        try {
          s = await liveState(sessionId);
        } catch {
          if (!dead) setErr(errMsg(e, "Could not start the live game."));
        }
      }
      if (dead) return;
      if (s) {
        const P = pb.current;
        const t0 = num(s.clock?.sec_elapsed_game);
        P.head = t0;
        P.bufferedTo = t0;
        P.simGame = t0;
        P.gc = t0;
        P.ready = true;
        const old = (Array.isArray(s.commentary) ? s.commentary : []).filter((c) => c && has(c.id));
        old.forEach((c) => seenLines.current.add(c.id));
        // Notes that already exist when we attach go to the log only, already read.
        const oldN = (Array.isArray(s.notifications) ? s.notifications : []).filter((n) => n && has(n.id)).map((n) => n.id);
        oldN.forEach((id) => seenNotif.current.add(id));
        setLogSeen(new Set(oldN));
        setLines(old.slice(-MAX_LINES).map((c) => ({ ...c, _at: 0 })));
        if (Number.isFinite(Number(s.seq))) sinceRef.current = Number(s.seq);
        if (s.tracking?.frames?.length) appendFrames(convertTracking(s.tracking.frames));
        mergeServer(s);
        setDisp({ ...s, _p: t0 });
        setGameSec(t0);
        // Pregame intro (E5) for a fresh game; the buffer keeps filling behind it.
        if (t0 <= 1 && !s.final && s.status !== "final") setIntro(true);
      }
      setBooting(false);
    })();
    return () => {
      dead = true;
    };
  }, [sessionId, mergeServer, appendFrames]);

  // Voice engine status (D23).
  useEffect(() => {
    let dead = false;
    const poll = async () => {
      try {
        const st = await liveTtsStatus();
        if (!dead) setTtsStatus(st || null);
      } catch {
        if (!dead) setTtsStatus({ worker: "down", engine: "browser" });
      }
    };
    poll();
    const id = setInterval(poll, 30000);
    return () => {
      dead = true;
      clearInterval(id);
    };
  }, []);

  const srvStatus = server?.status;
  const serverFinal = srvStatus === "final" || srvStatus === "committed" || !!server?.final;
  const serverPaused = srvStatus === "paused";
  const serverInt = (srvStatus === "intermission" || !!server?.clock?.intermission) && !puckDropped;

  /* ---------- fetch (never blocks rendering) ---------- */
  const fetchRef = useRef(async () => {});
  fetchRef.current = async () => {
    const P = pb.current;
    P.fetching = true;
    P.lastFetch = performance.now();
    const before = P.bufferedTo;
    try {
      // until_t is game time on the server; the buffer is measured in presentation time.
      // Pace by presentation time when the server supports it: stoppages, skate-outs and celebrations
      // freeze the game clock, so asking by game time alone can stall the buffer at 0:00.
      const body = {
        until_t: P.simGame + chunkSec(speedRef.current),
        until_pt: Math.max(P.bufferedTo, P.head) + chunkSec(speedRef.current),
        since_seq: sinceRef.current,
        lang: langRef.current,
      };
      const toEnd = simToEndRef.current;
      if (toEnd) body.sim_to_end = true;
      const res = await liveAdvance(sessionId, body);
      const lite = res?.snapshot_lite || null;
      const m = res?.motion;
      queueEvents(res?.events);
      if (Number.isFinite(Number(res?.last_seq))) sinceRef.current = Number(res.last_seq);
      const liteT = num(lite?.clock?.sec_elapsed_game, P.simGame);
      P.simGame = Math.max(P.simGame, liteT);
      if (m && Array.isArray(m.frames) && m.frames.length) {
        P.hasMotion = true;
        appendFrames(m.frames);
        if (P.frames.length) P.bufferedTo = Math.max(P.bufferedTo, P.frames[P.frames.length - 1]._p);
      } else {
        // Fallback: no motion track. Play the span on the server clock with whatever tracking frames came back.
        if (!P.hasMotion && lite?.tracking?.frames?.length) appendFrames(convertTracking(lite.tracking.frames));
        P.bufferedTo = Math.max(P.bufferedTo, P.hasMotion ? P.bufferedTo : liteT);
      }
      // The horn (period end / final) can be timed a few seconds after the last movement frame.
      // Stretch the buffer to cover every queued event so playback reaches it instead of stalling at 0:00.
      if (P.pendEvents.length && (lite?.status === "intermission" || lite?.clock?.intermission || lite?.status === "final" || lite?.final)) {
        P.bufferedTo = Math.max(P.bufferedTo, P.pendEvents[P.pendEvents.length - 1]._p + 0.3);
      }
      if (lite) {
        mergeServer(lite);
        queueSnapshot(lite, P.bufferedTo);
      }
      const ended = lite?.status === "final" || !!lite?.final || (res?.events || []).some((e) => e?.type === "final");
      if (ended) {
        try {
          const full = await liveState(sessionId);
          mergeServer(full);
          queueSnapshot(full, P.bufferedTo);
        } catch {
          /* keep lite */
        }
        if (serverRef.current?.status !== "final" && !serverRef.current?.final) mergeServer({ status: "final" });
      }
      if (toEnd) {
        P.head = P.bufferedTo;
        P.vel = 0;
        P.trail = [];
        P.replay = null;
        P.pendingReplay = null;
        setReplay(null);
      }
      if (P.bufferedTo <= before + 1e-6) P.backoffUntil = performance.now() + 1000;
      setErr("");
    } catch (e) {
      P.backoffUntil = performance.now() + (e?.response?.status === 409 ? 400 : 2000);
      if (e?.response?.status !== 409) setErr(errMsg(e, "The sim didn't answer. Retrying…"));
    } finally {
      P.fetching = false;
    }
  };
  const canFetchRef = useRef(false);
  canFetchRef.current = !!server && !booting && !serverFinal && !serverPaused && !serverInt && !held;

  /* ---------- reveal (events, lines, snapshots) at the head ---------- */
  const revealRef = useRef(() => {});
  revealRef.current = (head) => {
    const P = pb.current;
    const evs = [];
    while (P.pendEvents.length && P.pendEvents[0]._p <= head + 1e-6) evs.push(P.pendEvents.shift());
    const ls = [];
    while (P.pendLines.length && P.pendLines[0]._t <= head + 1e-6) ls.push({ ...P.pendLines.shift(), _at: Date.now() });
    let snap = null;
    let snapAt = 0;
    while (P.pendSnaps.length && P.pendSnaps[0].t <= head + 1e-6) {
      const q = P.pendSnaps.shift();
      snap = { ...(snap || {}), ...q.s };
      snapAt = q.t;
    }
    if (evs.length) {
      setEvents((prev) => [...prev, ...evs].slice(-MAX_EVENTS));
      const tm = serverRef.current?.teams || {};
      const sfx = sfxRef.current;
      for (const e of evs) {
        const type = e.type;
        if (type === "stoppage") {
          P.lastStopReason = e.meta?.reason || "whistle";
          P.lastStopPt = e._p;
          if (STOP_PHASES.has(P.ph) && P.ph !== "goal") setStop({ reason: P.lastStopReason, key: e.seq });
          if (P.lastStopReason !== "penalty") setDelayed(null);
        } else if (type === "faceoff") {
          const d = parseDot(e.meta?.dot) || parseDot(e.meta?.location) || (P.pk ? [P.pk[0], P.pk[1]] : [0, 0]);
          P.foDot = d;
          const w = e.meta?.winner_side || (e.meta?.winner === "home" || e.meta?.winner === "away" ? e.meta.winner : null);
          if (w) {
            setFoWin({ side: w, x: d[0], y: d[1], key: e.seq });
            flash("fowin", () => setFoWin(null), 2200);
          }
        } else if (type === "delayed_penalty" || (type === "penalty" && e.meta?.delayed)) {
          setDelayed({ side: e.side });
        } else if (type === "penalty") {
          setDelayed(null);
        } else if (type === "hit") {
          const id = String(e.meta?.hitter_id ?? e.shooter_id ?? "");
          const at = e.shot_loc ? [num(e.shot_loc.x), num(e.shot_loc.y)] : P.posOut.get(id);
          if (at) {
            setHit({ x: at[0], y: at[1], key: e.seq });
            flash("hit", () => setHit(null), 900);
          }
          sfx("boards");
        } else if (type === "attempt") {
          sfx("shot");
          const out = upper(e.outcome);
          if (out === "SAVED" || out === "GOAL") {
            const k = String(e.shooter_id ?? e.shooter_name ?? "");
            P.shots.set(k, (P.shots.get(k) || 0) + 1);
          }
          const ch = e.chance;
          if (e.shot_loc && (ch === "high_danger" || ch === "breakaway" || ch === "rebound" || e.rebound)) {
            const xgv = has(e.xg) ? num(e.xg) : 0;
            // "High-danger" only when the number backs it up; a low value next to a big label reads as a bug.
            const label = ch === "breakaway" ? "Breakaway" : ch === "rebound" || e.rebound ? "Rebound" : xgv >= 0.15 || !xgv ? "High-danger chance" : "Scoring chance";
            const pct = xgv >= 0.1 ? ` · ${Math.round(xgv * 100)}% goal chance` : "";
            setDanger({ x: num(e.shot_loc.x), y: num(e.shot_loc.y), text: `${label}${pct}`, key: e.seq });
            flash("danger", () => setDanger(null), 1700);
          }
          if (out === "SAVED" && num(e.importance) >= 3) {
            setSaveFlash({ key: e.seq, x: netXFor(e) * 0.95, y: 0 });
            flash("save", () => setSaveFlash(null), 1400);
            if (e._p - P.lastSaveReplay > 90 && speedRef.current <= 2 && !simToEndRef.current) {
              P.lastSaveReplay = e._p;
              P.pendingReplay = { kind: "save", at: e._p + 0.8, from: e._p - 2.2, to: e._p + 0.5, anchor: e._p };
            }
          }
          if ((out === "SAVED" || out === "GOAL") && (e.goalie_id || e.goalie_name)) {
            const gk0 = String(e.goalie_id ?? e.goalie_name);
            if (!P.gSaves) P.gSaves = new Map();
            if (!P.gShots) P.gShots = new Map();
            P.gShots.set(gk0, (P.gShots.get(gk0) || 0) + 1);
            if (out === "SAVED") P.gSaves.set(gk0, (P.gSaves.get(gk0) || 0) + 1);
          }
          // Live stat pop-ups (E10), at most every ~2.5 game minutes.
          if (P.gc - P.lastPopGc > 150) {
            const d = dispRef.current || {};
            const gSide = otherSide(e.side);
            const k = String(e.shooter_id ?? e.shooter_name ?? "");
            const n = P.shots.get(k) || 0;
            const gk = String(e.goalie_id ?? e.goalie_name ?? "");
            const gsv = (P.gSaves && P.gSaves.get(gk)) || 0;
            const gsa = (P.gShots && P.gShots.get(gk)) || 0;
            if (out === "SAVED" && gSide && e.goalie_name && gsv > 0) {
              const sv = gsa ? gsv / gsa : null;
              void d;
              setStatPop({ key: e.seq, side: gSide, name: e.goalie_name, text: `${gsv} save${gsv === 1 ? "" : "s"}${sv != null && gsa >= 5 ? ` · ${fmtPct(sv)}` : ""} tonight` });
              P.lastPopGc = P.gc;
              flash("pop", () => setStatPop(null), 4200);
            } else if (n >= 3 && e.shooter_name && e.side) {
              setStatPop({ key: e.seq, side: e.side, name: e.shooter_name, text: `${n} shots on goal tonight` });
              P.lastPopGc = P.gc;
              flash("pop", () => setStatPop(null), 4200);
            }
          }
        } else if (type === "goal") {
          const sa = e.score_after;
          const k = String(e.shooter_id ?? e.shooter_name ?? "");
          P.shots.set(k, (P.shots.get(k) || 0) + 1);
          setDelayed(null);
          setCelebration({ seq: e.seq, side: e.side, abbr: tm[e.side]?.abbr || "", score: sa && has(sa.home) ? `${tm.away?.abbr || "AWY"} ${sa.away} – ${sa.home} ${tm.home?.abbr || "HME"}` : "" });
          flash("celeb", () => setCelebration(null), 1500);
          const info = rosterRef.current.get(String(e.shooter_id)) || {};
          const seasonN = num(e.meta?.season_goals ?? info.season_goals ?? info.season?.g ?? info.season?.goals, NaN);
          setScorer({
            seq: e.seq,
            side: e.side,
            name: e.shooter_name || info.name || "Goal",
            number: has(e.shooter_number) ? e.shooter_number : info.number,
            assists: Array.isArray(e.assist_names) ? e.assist_names : [],
            season: Number.isFinite(seasonN) && seasonN > 0 ? `His ${ordinal(seasonN)} of the season` : "",
            player: facePlayer(e.shooter_id, { ...(e.meta?.scorer_headshot || {}), name: e.shooter_name || info.name, side: e.side }, rosterRef.current, serverRef.current?.teams),
          });
          flash("scorer", () => setScorer(null), 6500);
          setGoalPulse((n) => n + 1);
          if (e.side === "home") setHornPulse((n) => n + 1);
          else sfx("whistle");
          if (speedRef.current <= 2 && !simToEndRef.current) P.pendingReplay = { kind: "goal", at: e._p + 1.8, from: e._p - 6, to: e._p + 0.8, anchor: e._p };
        } else if (type === "period_end") {
          sfx("horn");
          const per = num(e.period, 1);
          const all = [...eventsRef.current, ...evs].filter((x) => num(x.period) === per);
          const sog = { home: 0, away: 0 };
          const goals = { home: 0, away: 0 };
          let best = null;
          all.forEach((x) => {
            if (!x.side) return;
            if (x.type === "goal") {
              goals[x.side] += 1;
              sog[x.side] += 1;
            } else if (x.type === "attempt" && upper(x.outcome) === "SAVED") sog[x.side] += 1;
            const score = x.type === "goal" ? 10 + num(x.xg) : x.type === "attempt" ? num(x.xg) + num(x.importance) : -1;
            if (score > 0 && (!best || score > best.score)) best = { score, text: x.description };
          });
          setPeriodCard({ period: per, sog, goals, best: best?.text || "" });
          flash("pcard", () => setPeriodCard(null), 15000);
        } else if (type === "period_start" && num(e.period) > 1) {
          setBanner(num(e.period) >= 5 ? "Shootout" : num(e.period) === 4 ? "Overtime" : `${periodName(e.period)} period`);
          flash("banner", () => setBanner(null), 2600);
          setPeriodCard(null);
          sfx("organ");
        } else if (type === "final") {
          sfx("horn");
        }
      }
    }
    if (ls.length) setLines((prev) => [...prev, ...ls].slice(-MAX_LINES));
    if (snap) setDisp((prev) => ({ ...(prev || {}), ...snap, _p: snapAt }));
  };

  /* ---------- the playback loop (rAF; refs only, React state only on change) ---------- */
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const pos = new Map();
    const frameMap = (f) => {
      if (!f._m) {
        const m = new Map();
        (Array.isArray(f.p) ? f.p : []).forEach((r) => m.set(String(r[0]), r));
        // Penalised players sit in the box (drawn behind the -y glass).
        (Array.isArray(f.bx) ? f.bx : []).forEach((r) => {
          const id = String(r[0]);
          if (!m.has(id)) m.set(id, [r[0], num(r[1]), num(r[2]) < -41 ? -47.6 : num(r[2])]);
        });
        f._m = m;
      }
      return f._m;
    };
    /** Cubic Hermite with velocities estimated from the neighbouring frames (non-uniform time). */
    const herm = (p0, p1, p2, p3, t0, t1, t2, t3, k, m1o, m2o) => {
      const dt = t2 - t1 > 1e-6 ? t2 - t1 : 0.5;
      const v12 = (p2 - p1) / dt;
      const m1 = m1o !== undefined ? m1o : t2 - t0 > 1e-6 ? (p2 - p0) / (t2 - t0) : v12;
      const m2 = m2o !== undefined ? m2o : t3 - t1 > 1e-6 ? (p3 - p1) / (t3 - t1) : v12;
      const k2 = k * k;
      const k3 = k2 * k;
      return (2 * k3 - 3 * k2 + 1) * p1 + (k3 - 2 * k2 + k) * dt * m1 + (-2 * k3 + 3 * k2) * p2 + (k3 - k2) * dt * m2;
    };
    const cos2 = (ax, ay, bx, by) => {
      const d = Math.hypot(ax, ay) * Math.hypot(bx, by);
      return d > 1e-6 ? (ax * bx + ay * by) / d : 1;
    };
    /** Sample the buffered track at presentation time t. */
    const sample = (P, t) => {
      const fs = P.frames;
      const n = fs.length;
      if (!n) return null;
      let lo = 0;
      if (t >= fs[n - 1]._p) lo = n - 1;
      else if (t > fs[0]._p) {
        let hi = n - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (fs[mid]._p <= t) lo = mid;
          else hi = mid;
        }
      }
      const a = fs[lo];
      const b = fs[lo + 1] || null;
      const span = b ? b._p - a._p : 0;
      const k = b && span > 0 ? Math.max(0, Math.min(1, (t - a._p) / span)) : 0;
      const reach = Math.max(1, span / 0.5);
      const a0 = fs[lo - 1] || a;
      const b1 = fs[lo + 2] || b || a;
      const am = frameMap(a);
      const bm = b ? frameMap(b) : null;
      const a0m = frameMap(a0);
      const b1m = frameMap(b1);
      const t0 = a0._p;
      const t1 = a._p;
      const t2 = b ? b._p : a._p + 0.5;
      const t3 = b1 !== b && b1 ? b1._p : t2 + (t2 - t1);
      pos.clear();
      am.forEach((r, id) => {
        if (r[3]) P.sides.set(id, r[3]);
        const q = bm && bm.get(id);
        let out = P.posOut.get(id);
        if (!out) {
          out = [0, 0];
          P.posOut.set(id, out);
        }
        if (q) {
          const ax = num(r[1]);
          const ay = num(r[2]);
          const bx = num(q[1]);
          const by = num(q[2]);
          if (Math.abs(bx - ax) > 40 * reach || Math.abs(by - ay) > 40 * reach) {
            out[0] = k < 0.5 ? ax : bx;
            out[1] = k < 0.5 ? ay : by;
          } else {
            const p0 = a0m.get(id);
            const p3 = b1m.get(id);
            const ok0 = p0 && a0 !== a && Math.abs(num(p0[1]) - ax) < 30 && Math.abs(num(p0[2]) - ay) < 30;
            const ok3 = p3 && b1 !== b && Math.abs(num(p3[1]) - bx) < 30 && Math.abs(num(p3[2]) - by) < 30;
            out[0] = herm(ok0 ? num(p0[1]) : ax, ax, bx, ok3 ? num(p3[1]) : bx, ok0 ? t0 : t1, t1, t2, ok3 ? t3 : t2, k);
            out[1] = herm(ok0 ? num(p0[2]) : ay, ay, by, ok3 ? num(p3[2]) : by, ok0 ? t0 : t1, t1, t2, ok3 ? t3 : t2, k);
          }
          pos.set(id, out);
        } else if (k < 0.5) {
          out[0] = num(r[1]);
          out[1] = num(r[2]);
          pos.set(id, out);
        }
      });
      if (bm && k >= 0.5)
        bm.forEach((q, id) => {
          if (pos.has(id)) return;
          let out = P.posOut.get(id);
          if (!out) {
            out = [0, 0];
            P.posOut.set(id, out);
          }
          out[0] = num(q[1]);
          out[1] = num(q[2]);
          pos.set(id, out);
        });
      // Puck: same Hermite, but kinks stay sharp (shot release, deflection, pass reception).
      let pk = null;
      let pv = 0;
      if (Array.isArray(a.pk)) {
        const ax = num(a.pk[0]);
        const ay = num(a.pk[1]);
        if (b && Array.isArray(b.pk) && Math.abs(num(b.pk[0]) - ax) < 70 * reach && Math.abs(num(b.pk[1]) - ay) < 70 * reach) {
          const bx = num(b.pk[0]);
          const by = num(b.pk[1]);
          const dt = t2 - t1 > 1e-6 ? t2 - t1 : 0.5;
          const vx = (bx - ax) / dt;
          const vy = (by - ay) / dt;
          pv = Math.hypot(vx, vy);
          let m1x;
          let m1y;
          let m2x;
          let m2y;
          const has0 = a0 !== a && Array.isArray(a0.pk);
          const has3 = b1 && b1 !== b && Array.isArray(b1.pk);
          const ux = has0 ? (ax - num(a0.pk[0])) / Math.max(1e-6, t1 - t0) : vx;
          const uy = has0 ? (ay - num(a0.pk[1])) / Math.max(1e-6, t1 - t0) : vy;
          const wx = has3 ? (num(b1.pk[0]) - bx) / Math.max(1e-6, t3 - t2) : vx;
          const wy = has3 ? (num(b1.pk[1]) - by) / Math.max(1e-6, t3 - t2) : vy;
          if (!has0 || (pv > 20 && (cos2(ux, uy, vx, vy) < 0.6 || pv > 2 * Math.hypot(ux, uy) + 15))) {
            m1x = vx;
            m1y = vy;
          }
          if (!has3 || (Math.hypot(wx, wy) > 20 && (cos2(vx, vy, wx, wy) < 0.6 || Math.hypot(wx, wy) > 2 * pv + 15))) {
            m2x = vx;
            m2y = vy;
          }
          pk = [
            herm(has0 ? num(a0.pk[0]) : ax, ax, bx, has3 ? num(b1.pk[0]) : bx, has0 ? t0 : t1, t1, t2, has3 ? t3 : t2, k, m1x, m2x),
            herm(has0 ? num(a0.pk[1]) : ay, ay, by, has3 ? num(b1.pk[1]) : by, has0 ? t0 : t1, t1, t2, has3 ? t3 : t2, k, m1y, m2y),
          ];
        } else pk = [ax, ay];
      }
      const near = k < 0.5 || !b ? a : b;
      const gc = b && b._g >= a._g ? a._g + (b._g - a._g) * k : a._g;
      const pz = b ? num(a.pz) + (num(b.pz) - num(a.pz)) * k : num(a.pz);
      return { pos, pk, near, gc, pv, pz, pp: a.pp || null };
    };

    const loop = (now) => {
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      const P = pb.current;
      if (P.ready) {
        // Instant replay (E1/E2): the live head waits while a replay head re-runs the buffered frames in slow motion.
        if (!P.replay && P.pendingReplay && P.head >= P.pendingReplay.at) {
          const R = P.pendingReplay;
          P.pendingReplay = null;
          if (speedRef.current <= 2 && !simToEndRef.current && P.frames.length && P.frames[0]._p <= R.from + 0.5) {
            P.replay = { ...R, t: R.from };
            P.trail = [];
            setReplay({ kind: R.kind });
            // Bring the analyst's breakdown forward so it plays over the replay.
            const pull = [];
            const horizon = R.anchor + (R.kind === "goal" ? 16 : 9);
            P.pendLines = P.pendLines.filter((l) => {
              if (l.speaker === "color" && l._t <= horizon) {
                pull.push({ ...l, _at: Date.now() });
                return false;
              }
              return true;
            });
            if (pull.length) setLines((prev) => [...prev, ...pull].slice(-MAX_LINES));
          }
        }
        let viewT;
        const R = P.replay;
        if (R) {
          if (speedRef.current >= 4 || simToEndRef.current) R.skip = true;
          R.t += dt * REPLAY_RATE;
          P.vel = 0;
          viewT = R.t;
          if (R.t >= R.to || R.skip) {
            P.replay = null;
            P.trail = [];
            setReplay(null);
            viewT = P.head;
          }
        }
        let room = P.bufferedTo - P.head;
        const target = heldRef.current || introRef.current || P.replay ? 0 : speedRef.current * paceRef.current;
        if (!P.replay) {
          const ease = Math.max(0, Math.min(1, room / 1.5));
          P.vel += (target * ease - P.vel) * Math.min(1, dt * 5);
          if (room <= 1e-4) P.vel = 0;
          P.head = Math.min(P.bufferedTo, P.head + Math.max(0, P.vel) * dt);
          room = P.bufferedTo - P.head;
          viewT = P.head;
          revealRef.current(P.head);
        }

        const smp = sample(P, viewT);
        if (smp) {
          const { near } = smp;
          let pk = smp.pk;
          const ph = near.ph || "play";
          if (!P.replay) P.gc = smp.gc;
          const c0 = has(near.c) ? String(near.c) : null;
          // Puck on the stick: just in front of a moving carrier.
          const cp = c0 ? smp.pos.get(c0) : null;
          if (cp) {
            const pc = P.carPrev;
            if (pc && pc.id === c0 && viewT > pc.t) {
              const ivt = 1 / Math.max(1e-3, viewT - pc.t);
              P.carV[0] += ((cp[0] - pc.x) * ivt - P.carV[0]) * 0.25;
              P.carV[1] += ((cp[1] - pc.y) * ivt - P.carV[1]) * 0.25;
            } else {
              P.carV[0] = 0;
              P.carV[1] = 0;
            }
            P.carPrev = { id: c0, x: cp[0], y: cp[1], t: viewT };
            const sp = Math.hypot(P.carV[0], P.carV[1]);
            if (pk && ph === "play" && Math.hypot(pk[0] - cp[0], pk[1] - cp[1]) < 5 && sp > 2) pk = [cp[0] + (P.carV[0] / sp) * 1.7, cp[1] + (P.carV[1] / sp) * 1.7];
          } else P.carPrev = null;
          if (pk) P.pk = pk;
          const shot = !c0 && SHOT_TYPES.has(String(smp.pp || ""));
          // passes: dotted path from where the puck left the stick; names for passer + receiver for 2 s
          if (c0 !== P.prevC) {
            if (P.prevC && !c0 && pk && !shot) P.pass = { x: pk[0], y: pk[1], t0: now };
            if (P.prevC) P.recent.set(P.prevC, now + 2000);
            if (c0) P.recent.set(c0, now + 2000);
            P.prevC = c0;
          }
          if (P.pass && now - P.pass.t0 > 600) P.pass = null;
          // snow puffs on hard stops (frame sp)
          if (near !== P.spFrame) {
            P.spFrame = near;
            if (Array.isArray(near.sp))
              near.sp.forEach((id) => {
                const q = smp.pos.get(String(id));
                if (q) P.puffs.push({ x: q[0], y: q[1], t0: now });
              });
            while (P.puffs.length > PUFFS) P.puffs.shift();
          }
          // trail
          if (ph !== P.ph && (ph === "faceoff" || ph === "goal")) P.trail = [];
          if (pk && ph === "play" && now - P.trailAt > 30) {
            P.trail.push([pk[0], pk[1]]);
            P.trailAt = now;
            if (P.trail.length > TRAIL_LEN) P.trail.shift();
          } else if (ph !== "play" && P.trail.length && now - P.trailAt > 40) {
            P.trail.shift();
            P.trailAt = now;
          }
          // markers re-render only when the on-ice set changes
          let sig = "";
          smp.pos.forEach((_, id) => {
            sig += `${id},`;
          });
          if (sig !== P.sig) {
            P.sig = sig;
            setOnIce([...smp.pos.keys()].sort().map((id) => ({ id, side: rosterRef.current.get(id)?.side || P.sides.get(id) || "home" })));
          }
          const layout = now - P.lastLayout > LABEL_LAYOUT_MS;
          if (layout) {
            P.lastLayout = now;
            P.recentSet.clear();
            P.recent.forEach((exp, id) => {
              if (exp > now) P.recentSet.add(id);
              else P.recent.delete(id);
            });
          }
          const puffs = P.puffs.length ? P.puffs.map((q) => ({ x: q.x, y: q.y, age: now - q.t0 })) : null;
          if (puffs && puffs.every((q) => q.age > 700)) P.puffs = [];
          rinkApi.current?.draw({
            pos: smp.pos,
            pk,
            carrier: near.c,
            trail: P.trail,
            ph,
            layout,
            puckSpeed: smp.pv,
            shot,
            pz: smp.pz,
            pass: P.pass ? { x: P.pass.x, y: P.pass.y, age: now - P.pass.t0 } : null,
            puffs,
            recent: P.recentSet,
          });

          if (!P.replay) {
            // shift clocks (game seconds since each player stepped on)
            let lsig = "";
            smp.pos.forEach((_, id) => {
              lsig += `${id},`;
            });
            if (lsig !== P.liveSig) {
              P.liveSig = lsig;
              smp.pos.forEach((_, id) => {
                if (!P.shiftStart.has(id)) P.shiftStart.set(id, P.gc);
              });
              [...P.shiftStart.keys()].forEach((id) => {
                if (!smp.pos.has(id)) P.shiftStart.delete(id);
              });
            }
            const sfx = sfxRef.current;
            // SFX: passes on a carrier change within a team, boards when the puck reaches them.
            const c = has(near.c) ? String(near.c) : null;
            if (c && P.lastCarrier && c !== P.lastCarrier && ph === "play" && now - P.passAt > 160) {
              sfx("pass");
              P.passAt = now;
            }
            if (c) P.lastCarrier = c;
            if (pk) {
              const atBoards = Math.abs(pk[1]) > 40.5 || Math.abs(pk[0]) > 97;
              if (atBoards && P.puckIn && ph === "play" && now - P.boardsAt > 350) {
                sfx("boards");
                P.boardsAt = now;
              }
              P.puckIn = !atBoards;
            }
            // possession zone for the situation line
            let zone = null;
            const posSide = near.pos || null;
            if (posSide && pk) {
              const per = clockFromT(P.gc).period;
              const attacksRight = (posSide === "home") === homeAttacksRight(per);
              zone = Math.abs(pk[0]) <= 25 ? "nz" : pk[0] > 25 === attacksRight ? "oz" : "dz";
            }
            if (ph !== P.ph || posSide !== P.pos || zone !== P.zone) {
              const wasStop = STOP_PHASES.has(P.ph);
              const isStop = STOP_PHASES.has(ph) && ph !== "intermission";
              if (isStop && !wasStop && ph !== "goal") {
                let reason = ph !== "stoppage" ? ph : P.head - P.lastStopPt < 6 && P.lastStopReason ? P.lastStopReason : "whistle";
                if (reason === "whistle" && (P.gc < 1 || P.gc % 1200 < 1)) reason = "opening";
                setStop({ reason, key: `${Math.round(P.head * 10)}` });
                if (ph !== "goal") {
                  sfx("whistle");
                  P.stopCount += 1;
                  if (P.stopCount % 3 === 0) flash("organ", () => sfxRef.current("organ"), 1300);
                }
              } else if (!isStop && wasStop) setStop(null);
              if (ph === "faceoff" && P.ph !== "faceoff") {
                const d = P.foDot && P.head - P.lastStopPt < 30 ? P.foDot : pk ? [pk[0], pk[1]] : [0, 0];
                setFaceoff({ x: d[0], y: d[1] });
              } else if (ph !== "faceoff" && P.ph === "faceoff") setFaceoff(null);
              P.ph = ph;
              P.pos = posSide;
              P.zone = zone;
              setPhase({ ph, pos: posSide, zone });
            }
          }
          // keep ~25 s of frames behind the head for replays
          if (P.frames.length > 200 && P.frames[60]._p < P.head - 25 && !P.replay) P.frames.splice(0, 60);
        } else if (!P.replay) {
          P.gc = P.head; // no tracking at all: game clock follows the head
        }

        const sec = Math.floor(P.gc);
        if (sec !== P.shownSec) {
          P.shownSec = sec;
          setGameSec(P.gc);
        }
        const ptS = Math.floor(P.head);
        if (ptS !== P.shownPt) {
          P.shownPt = ptS;
          setHeadPt(P.head);
        }

        // Safety net: nothing more will be fetched (intermission, final) but events are still queued past the buffer.
        if (!canFetchRef.current && !heldRef.current && P.pendEvents.length && P.head >= P.bufferedTo - 0.25) {
          P.bufferedTo = Math.max(P.bufferedTo, P.pendEvents[P.pendEvents.length - 1]._p + 0.3);
        }
        const needMore = P.bufferedTo - P.head < bufferMin(speedRef.current);
        if (needMore && canFetchRef.current && !P.fetching && now - P.lastFetch > FETCH_GAP_MS && now > P.backoffUntil) fetchRef.current();
        const isBuf = !heldRef.current && !introRef.current && !P.replay && canFetchRef.current && room < 1 && P.vel < target * 0.5;
        if (isBuf !== P.buffering) {
          P.buffering = isBuf;
          setBuffering(isBuf);
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [flash]);

  const skipReplay = useCallback(() => {
    const R = pb.current.replay;
    if (R) R.skip = true;
  }, []);

  /* ---------- derived display ---------- */
  const view = disp || server || {};
  dispRef.current = view;
  const teams = useMemo(() => server?.teams || disp?.teams || {}, [server, disp]);
  const colors = useMemo(() => kitColors(teams), [teams]);

  const roster = useMemo(() => {
    const m = rosterRef.current;
    const put = (p, side) => {
      if (!p || !has(p.player_id)) return;
      const k = String(p.player_id);
      m.set(k, { ...(m.get(k) || {}), ...p, side: side || p.side || m.get(k)?.side });
    };
    const r = server?.roster;
    if (r && typeof r === "object") Object.entries(r).forEach(([id, p]) => put({ ...p, player_id: id }, p?.side));
    const oi = view.on_ice || {};
    (oi.home || []).forEach((p) => put(p, "home"));
    (oi.away || []).forEach((p) => put(p, "away"));
    put(oi.home_goalie, "home");
    put(oi.away_goalie, "away");
    (view.bench?.players || []).forEach((p) => put(p, view.bench?.user_side));
    return new Map(m);
  }, [server, view]);

  const score = useMemo(() => {
    let evT = null;
    for (let i = events.length - 1; i >= 0; i -= 1)
      if (events[i]?.score_after && has(events[i].score_after.home)) {
        evT = events[i];
        break;
      }
    if (evT && num(evT._p) >= num(view._p, -1)) return evT.score_after;
    return view.score || evT?.score_after || {};
  }, [events, view]);

  const otLen = useMemo(() => {
    const c = server?.clock;
    if (c && num(c.period) === 4 && has(c.sec_left_period)) return Math.max(60, num(c.sec_elapsed_game) - 3600 + num(c.sec_left_period));
    return 300;
  }, [server]);
  const playbackDone = pb.current.head >= pb.current.bufferedTo - 0.25 && pb.current.pendEvents.length === 0 && !replay;
  const isFinal = serverFinal && playbackDone;
  const isPausedView = held || (serverPaused && playbackDone);
  const intermission = !!view.clock?.intermission || (serverInt && playbackDone);
  const shootout = num(view.clock?.period) >= 5;
  const hc = clockFromT(gameSec, otLen);
  const period = shootout ? 5 : hc.period;
  const clockText = shootout ? "SO" : intermission ? mmss(view.clock?.sec_left_period ?? 0) : mmss(hc.left);
  const manpower = view.manpower || {};
  const mpState = upper(manpower.state) || "EV";
  const coach = useMemo(() => server?.coach || {}, [server]);
  const userSide = server?.bench?.user_side || (teams.home?.is_user ? "home" : teams.away?.is_user ? "away" : null);
  const stopped = STOP_PHASES.has(phase.ph) || phase.ph === "faceoff";

  // Drama: one state at a time, strongest first (E7).
  const diff = num(score.home) - num(score.away);
  let drama = null;
  let dramaText = "";
  if (!isFinal && !intermission) {
    const enSide = mpState === "EN" ? (manpower.pp_side ? otherSide(manpower.pp_side) : null) : null;
    if (shootout) {
      drama = "so";
      dramaText = "Shootout";
    } else if (mpState === "EN" || (coach.goalie_pulled && userSide)) {
      const enOf = coach.goalie_pulled && userSide ? userSide : enSide;
      drama = enOf ? `en-${enOf}` : "en";
      dramaText = `Empty net${enOf ? ` · ${teams?.[enOf]?.abbr || ""} goalie pulled` : ""}`;
    } else if (manpower.pp_side && (mpState === "PP" || mpState === "SH")) {
      drama = `pp-${manpower.pp_side}`;
      dramaText = "";
    } else if (period === 4) {
      drama = "ot";
      dramaText = `Overtime${mpState === "3V3" ? " · 3 on 3" : ""}`;
    } else if (period === 3 && hc.left <= 120 && Math.abs(diff) <= 1) {
      drama = "late";
      dramaText = diff === 0 ? "Final minutes · tied" : "Final minutes · one-goal game";
    }
  }
  const late = drama === "late" || drama === "ot";

  // Situation line from the frame at the head (A25), the server's sentence only while it's fresh.
  const situation = useMemo(() => {
    if (isFinal) return "Final horn.";
    if (intermission) return `${periodName(view.clock?.period)} intermission.`;
    if (replay) return replay.kind === "save" ? "Another look at that save." : "Another look at the goal.";
    if (stop && stop.reason === "opening") return "Players lining up for the opening faceoff.";
    if (stop && STOP_LABEL[stop.reason]) return `Whistle — ${STOP_LABEL[stop.reason].toLowerCase()}.${stop.reason !== "goal" ? " Faceoff coming up." : ""}`;
    if (phase.ph === "faceoff") return "Players lining up for the faceoff.";
    const fresh = view.situation?.text && headPt - num(view._p, -99) < 4;
    if (fresh) return view.situation.text;
    const name = (s) => teams?.[s]?.name || teams?.[s]?.abbr || "";
    const ppSide = manpower.pp_side;
    const pp = ppSide && (mpState === "PP" || mpState === "SH") ? ` ${name(ppSide)} power play${has(manpower.pp_sec_left) ? `, ${mmss(manpower.pp_sec_left)} left` : ""}.` : "";
    if (phase.pos && teams?.[phase.pos]) {
      const opp = name(otherSide(phase.pos));
      const z = phase.zone === "oz" ? `pressing in the ${opp} zone` : phase.zone === "dz" ? "breaking out of their own zone" : "carrying through the neutral zone";
      return `${name(phase.pos)} ${z}.${pp}`;
    }
    if (pp.trim()) return pp.trim();
    return phase.ph === "play" ? "Puck is in play." : view.situation?.text || "Waiting for the puck drop.";
  }, [view, isFinal, intermission, replay, stop, phase, manpower, mpState, teams, headPt]);

  const shots = useMemo(
    () => events.filter((e) => e && (e.type === "attempt" || e.type === "goal") && e.shot_loc && has(e.shot_loc.x) && has(e.shot_loc.y)).slice(-SHOT_MARKERS),
    [events]
  );

  // Who is in the penalty box (for the box highlight).
  const boxed = useMemo(() => {
    const out = { home: false, away: false };
    onIce.forEach(({ id, side }) => {
      const p = pb.current.posOut.get(id);
      if (p && p[1] < -42.6 && Math.abs(p[0]) < 16) out[side] = true;
    });
    return out;
  }, [onIce, headPt]); // eslint-disable-line react-hooks/exhaustive-deps

  const languages = Array.isArray(server?.languages) && server.languages.length ? server.languages : FALLBACK_LANGS;
  const langLabel = (languages.find((l) => l.code === lang) || FALLBACK_LANGS.find((l) => l.code === lang) || { label: lang }).label;

  /* ---------- audio ---------- */
  const booth = useBooth({ enabled: soundOn && prefs.voice, volume: prefs.volume, lang, lines, clipsRef });
  const crowdAudio = useCrowdAudio({
    enabled: soundOn && prefs.crowd,
    level: Math.min(1, num(view.crowd?.level, 0.3) + (late ? 0.15 : 0)) * (replay ? 0.6 : 1),
    mood: view.crowd?.mood,
    duck: booth.speaking,
    goalPulse,
    hornPulse,
    volume: prefs.volume,
  });
  sfxRef.current = crowdAudio.sfx;
  const updatePrefs = (patch) =>
    setPrefs((p) => {
      const next = { ...p, ...patch };
      writeSoundPrefs(next);
      return next;
    });
  const turnSoundOn = () => {
    crowdAudio.unlock();
    booth.prime();
    setSoundOn(true);
  };
  const toggleSound = () => {
    if (soundOn) setSoundOn(false);
    else turnSoundOn();
  };
  const pickLang = (code) => {
    setLang(code);
    writeLang(code);
  };
  const neural = ttsStatus && ttsStatus.worker === "up" && ttsStatus.engine && ttsStatus.engine !== "browser";
  const engineName = neural ? String(ttsStatus.engine).replace(/^\w/, (c) => c.toUpperCase()) : "";
  const voiceLabel = !soundOn
    ? null
    : !prefs.voice
    ? "Commentary off"
    : neural
    ? `Neural voices · ${engineName}`
    : booth.status === "unsupported"
    ? "No speech engine · captions"
    : booth.status === "novoice"
    ? `No ${langLabel} voice · captions`
    : "Commentary on";

  // Caption: what the booth is saying; the last line stays (dimmed) so the strip never jumps.
  const [, forceTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const lastLine = lines[lines.length - 1];
  const captionLine = booth.current || lastLine || null;
  const captionLive = !!booth.current || (lastLine && Date.now() - num(lastLine._at) < STALE_MS);

  // Chant (context-aware): shown for a few seconds when it changes, never during a replay.
  const [chant, setChant] = useState(null);
  const chantSrc = view.crowd?.chant || null;
  useEffect(() => {
    if (!chantSrc || !chantFits(chantSrc, diff)) return;
    setChant(chantSrc);
    flash("chant", () => setChant(null), 5500);
  }, [chantSrc]); // eslint-disable-line react-hooks/exhaustive-deps

  /* ---------- coach ---------- */
  const sendCoach = useCallback(
    async (kind, payload = {}) => {
      try {
        const r = await liveCoach(sessionId, { kind, payload });
        const order = r?.order || r || {};
        if (r?.snapshot) mergeServer({ coach: r.snapshot.coach, bench: r.snapshot.bench || serverRef.current?.bench });
        else if (r?.coach) mergeServer({ coach: r.coach });
        setLastOrder({ id: order.id, kind, payload, status: order.status || r?.status || "queued", reason: order.reason || "", key: Date.now() });
      } catch (e) {
        setLastOrder({ kind, payload, status: "rejected", reason: errMsg(e, "Order failed."), key: Date.now() });
      }
    },
    [sessionId, mergeServer]
  );
  const result = useMemo(() => {
    if (!lastOrder) return null;
    const applied = Array.isArray(coach.applied) ? coach.applied : [];
    const pending = Array.isArray(coach.pending) ? coach.pending : [];
    const srv = lastOrder.id ? [...applied, ...pending].find((o) => o.id === lastOrder.id) : null;
    const o = { ...lastOrder, ...(srv ? { status: srv.status, reason: srv.reason } : {}) };
    const f = friendlyOrder(o);
    return { ...f, key: `${lastOrder.key}-${f.status}` };
  }, [coach, lastOrder]);
  useEffect(() => {
    if (!result) return;
    setToast(result);
    flash("toast", () => setToast(null), result.status === "applying" ? 8000 : 3800);
  }, [result?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  const canCoach = !!server && !booting && !serverFinal;
  const live = canCoach && !serverInt;
  const pullLegal = live && mpState === "EV" && !coach.goalie_pulled;

  const doTimeout = () => {
    if (!live || coach.timeout_used) return;
    sendCoach("timeout", {});
  };
  const doPull = () => {
    if (coach.goalie_pulled) {
      sendCoach("pull_goalie", { pull: false });
      return;
    }
    if (!pullLegal) return;
    if (!pullArmed) {
      setPullArmed(true);
      flash("pull", () => setPullArmed(false), 4000);
      return;
    }
    setPullArmed(false);
    clearTimeout(timers.current.pull);
    sendCoach("pull_goalie", {});
  };

  /* ---------- alerts (revealed at the head) ---------- */
  const alert = useMemo(() => {
    const list = Array.isArray(view.alerts) ? view.alerts : [];
    return (
      list
        .filter(
          (a) =>
            a &&
            !dismissed.has(a.id) &&
            (has(a.pt) ? num(a.pt) <= headPt + 0.5 : !has(a.t) || num(a.t) <= gameSec + 0.5) &&
            (!has(a.ttl_sec) || !has(a.t) || gameSec - num(a.t) <= num(a.ttl_sec))
        )
        .sort((a, b) => ({ high: 3, med: 2, low: 1 }[b.severity] || 0) - ({ high: 3, med: 2, low: 1 }[a.severity] || 0) || num(b.seq) - num(a.seq))[0] || null
    );
  }, [view, dismissed, headPt, gameSec]);
  const dismissAlert = (id) => setDismissed((prev) => new Set([...prev, id]));

  /* ---------- coaching notifications (v4) ---------- */
  const hasNotifFeed = Array.isArray(server?.notifications) || Array.isArray(view.notifications);
  const notifList = useMemo(() => {
    // Items with pt can be read from the (ahead) server snapshot and revealed at the head; others wait for the displayed snapshot.
    const byId = new Map();
    (Array.isArray(view.notifications) ? view.notifications : []).forEach((n) => n && has(n.id) && byId.set(n.id, n));
    (Array.isArray(server?.notifications) ? server.notifications : []).forEach((n) => {
      if (n && has(n.id) && has(n.pt)) byId.set(n.id, { ...(byId.get(n.id) || {}), ...n });
    });
    return [...byId.values()].filter((n) => !has(n.pt) || num(n.pt) <= headPt + 0.5).sort((a, b) => ptOf(b, num(b.t)) - ptOf(a, num(a.t)));
  }, [view, server, headPt]);
  const notifById = useMemo(() => new Map(notifList.map((n) => [n.id, n])), [notifList]);
  const statusOf = useCallback(
    (n) => {
      if (nStatus[n.id]) return nStatus[n.id];
      const st = n.status || "new";
      if (st === "new" && has(n.expires_pt) && num(n.expires_pt) < headPt) return "expired";
      return st;
    },
    [nStatus, headPt]
  );
  const unread = notifList.filter((n) => !logSeen.has(n.id)).length;
  const ackNotif = useCallback(
    (id, status) => {
      setNStatus((p) => ({ ...p, [id]: status }));
      setCards((prev) => prev.filter((x) => x !== id));
      liveCoach(sessionId, { kind: "notification_ack", payload: { id, status } }).catch(() => {});
    },
    [sessionId]
  );
  const actNotif = (n, a) => {
    if (a?.order?.kind) sendCoach(a.order.kind, a.order.payload || {});
    ackNotif(n.id, "acted");
  };
  const dismissNotif = (n) => ackNotif(n.id, "dismissed");
  const setPauseCritical = (v) => {
    setPauseCriticalState(v);
    writeFlag(PAUSE_CRIT_KEY, v);
  };
  const toggleCompact = () =>
    setCompactState((v) => {
      writeFlag(COMPACT_KEY, !v);
      return !v;
    });
  const toggleLog = () =>
    setLogOpen((o) => {
      if (!o) setLogSeen(new Set(notifList.map((n) => n.id)));
      return !o;
    });
  useEffect(() => {
    if (logOpen) setLogSeen(new Set(notifList.map((n) => n.id)));
  }, [logOpen, notifList]);
  const toggleStats = () =>
    setStatsOpen((o) => {
      if (!o) setDrawer(null);
      return !o;
    });
  const alertAction = (a) => {
    if (a.action.kind === "open_lines") setDrawer("lines"); // a suggestion to rework the lines opens the Lines tab
    else sendCoach(a.action.kind, a.action.payload || {});
    dismissAlert(a.id);
  };

  /* ---------- game management ---------- */
  const togglePause = async () => {
    if (serverFinal) return;
    if (!held) {
      setHeld(true);
      try {
        await livePause(sessionId);
        mergeServer({ status: "paused" });
      } catch {
        /* local hold still applies */
      }
    } else {
      setHeld(false);
      try {
        await liveResume(sessionId);
        const s = await liveState(sessionId);
        mergeServer({ status: s?.status, clock: s?.clock, coach: s?.coach });
      } catch (e) {
        setErr(errMsg(e, "Resume failed."));
      }
    }
  };
  // The backend doesn't pause at intermission: the next /advance skips past it. Only /resume when truly paused.
  const dropPuck = async () => {
    setHeld(false);
    setPuckDropped(true);
    setPeriodCard(null);
    queueSnapshot({ clock: { ...(serverRef.current?.clock || {}), intermission: false } }, pb.current.head);
    if (serverRef.current?.status === "paused") {
      try {
        await liveResume(sessionId);
      } catch (e) {
        setErr(errMsg(e, "Resume failed."));
      }
    }
  };
  useEffect(() => {
    if (puckDropped && srvStatus !== "intermission" && !server?.clock?.intermission) setPuckDropped(false);
  }, [puckDropped, srvStatus, server]);
  const endIntro = useCallback(() => {
    setIntro(false);
    sfxRef.current("charge");
    setGoalPulse((n) => n + 1);
  }, []);
  useEffect(() => {
    if (!intro) return undefined;
    const id = setTimeout(endIntro, 9000);
    return () => clearTimeout(id);
  }, [intro, endIntro]);
  const onCommit = async () => {
    setWorking(true);
    try {
      await liveCommit(sessionId);
      onExit?.();
    } catch (e) {
      // 404 on a second commit: the result is already saved.
      if (e?.response?.status === 404) {
        onExit?.();
        return;
      }
      setErr(errMsg(e, "Commit failed."));
    } finally {
      setWorking(false);
    }
  };
  const onLeave = async () => {
    setWorking(true);
    try {
      await liveAbort(sessionId);
      onExit?.();
    } catch (e) {
      setErr(errMsg(e, "Could not leave the game."));
      setWorking(false);
      setConfirm(null);
    }
  };

  /* ---------- keyboard ---------- */
  const keysRef = useRef(() => {});
  keysRef.current = (e) => {
    const tag = String(e.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea" || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "Escape") {
      if (help) setHelp(false);
      else if (confirm) setConfirm(null);
      else if (logOpen) setLogOpen(false);
      else if (soundMenu) setSoundMenu(false);
      else if (replay) skipReplay();
      else if (intro) endIntro();
      else setDrawer(null);
      return;
    }
    if (k === " " || k === "Spacebar") {
      e.preventDefault();
      if (intro) endIntro();
      else togglePause();
    } else if (k >= "1" && k <= "4") setSpeed(SPEEDS[Number(k) - 1]);
    else if (k === "t" || k === "T") doTimeout();
    else if (k === "g" || k === "G") doPull();
    else if (k === "l" || k === "L") setDrawer((d) => (d === "lines" ? null : "lines"));
    else if (k === "h" || k === "H" || k === "?") setHelp((h) => !h);
    else if (k === "m" || k === "M") toggleSound();
    else if (k === "r" || k === "R") skipReplay();
    else if (k === "s" || k === "S") toggleStats();
    else if (k === "n" || k === "N") toggleLog();
    else if (k === "c" || k === "C") toggleCompact();
    else if (k === "v" || k === "V") setNames(names === "key" ? "all" : names === "all" ? "off" : "key");
  };
  useEffect(() => {
    const h = (e) => keysRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // New priority 2-3 notes pop as cards; priority 1 only lands in the log. Cards fold into the bell quickly.
  useEffect(() => {
    const fresh = notifList.filter((n) => !seenNotif.current.has(n.id));
    if (!fresh.length) return;
    fresh.forEach((n) => seenNotif.current.add(n.id));
    const pop = fresh.filter((n) => num(n.priority, 1) >= 2 && statusOf(n) === "new");
    if (!pop.length) return;
    const ids = pop.map((n) => n.id);
    setCards((prev) => [...ids, ...prev.filter((id) => !ids.includes(id))].slice(0, 3));
    const ms = 12000 / ({ 1: 1, 2: 1.5, 4: 2.5, 8: 3 }[speedRef.current] || 1);
    pop.forEach((n) => flash(`n${n.id}`, () => setCards((prev) => prev.filter((id) => id !== n.id)), num(n.priority) >= 3 ? ms * 1.4 : ms));
    if (pop.some((n) => num(n.priority) >= 3)) {
      sfxRef.current("alert");
      if (pauseCritical && !heldRef.current && !serverFinal) togglePause();
    }
  }, [notifList]); // eslint-disable-line react-hooks/exhaustive-deps

  // Final horn once the viewer reaches the end.
  const finalOnce = useRef(false);
  useEffect(() => {
    if (isFinal && !finalOnce.current) {
      finalOnce.current = true;
      sfxRef.current("horn");
    }
  }, [isFinal]);

  const stars = useMemo(() => {
    const srv = isFinal && Array.isArray(server?.final?.stars) && server.final.stars.length ? server.final.stars : null;
    if (srv)
      return srv.slice(0, 3).map((x) => {
        const pid = x.player_id ?? x.id;
        const r = rosterRef.current.get(String(pid)) || {};
        return { id: String(pid ?? x.name), side: x.side || r.side || "home", name: x.name || r.name || "", line: x.line || "", face: facePlayer(pid, x, rosterRef.current, teams), fromServer: true };
      });
    if (!(isFinal || intermission)) return [];
    return deriveStars(events).map((x) => ({ ...x, face: facePlayer(x.id, { name: x.name, side: x.side, ...(x.goalie ? { position: "G" } : {}) }, rosterRef.current, teams) }));
  }, [isFinal, intermission, events, server, teams]);
  const sog = { home: view.stats?.home?.sog, away: view.stats?.away?.sog };

  const analytics = view.analytics || server?.analytics || null;
  const onIceIds = useMemo(() => ({ home: onIce.filter((p) => p.side === "home").map((p) => p.id), away: onIce.filter((p) => p.side === "away").map((p) => p.id) }), [onIce]);
  const allowedKinds = server?.coach?.allowed || server?.coach?.allowed_kinds || server?.flags?.coach_kinds || null;
  const ozKind = Array.isArray(allowedKinds) ? OZ_KINDS.find((k) => allowedKinds.includes(k)) || null : null;

  /* ---------- overlays ---------- */
  let overlay = null;
  if (isFinal) {
    const fin = server?.final || {};
    overlay = (
      <div className="st-final">
        <span className="st-stamp is-big">{fin.decision && fin.decision !== "REG" ? `Final / ${fin.decision}` : "Final"}</span>
        <div className="st-final__score">
          {["away", "home"].map((s) => (
            <div key={s} className={`st-final__team is-${s}${fin.winner_side === s ? " is-win" : ""}`} style={colors[s].chip}>
              <b>{teams?.[s]?.abbr || s}</b>
              <strong>{num(server?.score?.[s] ?? score[s])}</strong>
              {has(server?.stats?.[s]?.sog) ? <small>SOG {server.stats[s].sog}</small> : null}
            </div>
          ))}
        </div>
        {stars.length ? (
          <ol className="st-stars" title={stars[0]?.fromServer ? "Three stars of the game" : "Worked out from this game's events"}>
            {stars.map((s, i) => (
              <li key={s.id} className={`st-star is-${i + 1}`} style={{ animationDelay: `${0.25 + i * 0.35}s` }}>
                <span className="st-star__n">{"★".repeat(i + 1)}</span>
                <PlayerHeadshot player={s.face} size="md" className="st-star__face" teamColors={{ primary: colors[s.side]?.primary, secondary: colors[s.side]?.secondary }} showFlag={false} />
                <b>{s.name}</b>
                <TeamChip side={s.side || "home"} colors={colors} teams={teams} />
                <small>{s.line}</small>
              </li>
            ))}
          </ol>
        ) : null}
        <div className="st-final__box">
          <StatsTable stats={server?.stats || view.stats} teams={teams} colors={colors} compact />
        </div>
        {working ? <p className="st-note">Saving the result and simming the rest of the league's day. This takes a few seconds.</p> : null}
        <div className="st-final__actions">
          <button type="button" className="st-btn is-lg" disabled={working || srvStatus === "committed"} onClick={onCommit}>
            {working ? "Saving result & simming the league…" : srvStatus === "committed" ? "Committed" : "Commit result"}
          </button>
          <button type="button" className="st-btn is-ghost" onClick={() => setConfirm("discard")} disabled={working}>
            Discard &amp; leave
          </button>
        </div>
      </div>
    );
  } else if (intermission || periodCard) {
    const pc = periodCard;
    overlay = (
      <div className="st-hold st-pcard">
        <span className="st-stamp is-big">{pc ? `End of the ${periodName(pc.period)}` : `${periodName(view.clock?.period)} intermission`}</span>
        {pc ? (
          <div className="st-pcard__grid">
            <div className="st-pcard__row">
              <TeamChip side="away" colors={colors} teams={teams} />
              <span>Goals</span>
              <TeamChip side="home" colors={colors} teams={teams} />
            </div>
            <div className="st-pcard__row is-num">
              <b>{pc.goals.away}</b>
              <span>this period</span>
              <b>{pc.goals.home}</b>
            </div>
            <div className="st-pcard__row is-num">
              <b>{pc.sog.away}</b>
              <span>shots on goal</span>
              <b>{pc.sog.home}</b>
            </div>
          </div>
        ) : null}
        {pc?.best ? (
          <p className="st-pcard__best">
            <small>Play of the period</small>
            {pc.best}
          </p>
        ) : null}
        {intermission && stars.length ? (
          <ol className="st-stars is-mini">
            {stars.map((s, i) => (
              <li key={s.id} className="st-star">
                <span className="st-star__n">{"★".repeat(i + 1)}</span>
                <PlayerHeadshot player={s.face} size="sm" className="st-star__face is-mini" showFlag={false} />
                <b>{s.name}</b>
                <small>{s.line}</small>
              </li>
            ))}
          </ol>
        ) : null}
        {intermission ? (
          <>
            <p>Zamboni's out. Check your lines and tactics, then drop the puck.</p>
            <button type="button" className="st-btn is-lg" onClick={dropPuck}>
              Drop the puck
            </button>
          </>
        ) : null}
      </div>
    );
  } else if (intro) {
    const rv = view.crowd?.rivalry;
    const goalie = (s) => view.on_ice?.[`${s}_goalie`]?.name || null;
    const rec = (s) => teams?.[s]?.record || (has(teams?.[s]?.wins) ? `${teams[s].wins}-${num(teams[s].losses)}-${num(teams[s].otl)}` : "");
    overlay = (
      <div className="st-intro">
        {num(rv?.tier) > 0 ? <span className="st-intro__rival">{rv?.name || "Rivalry night"}</span> : <span className="st-intro__rival is-plain">Tonight</span>}
        <div className="st-intro__teams">
          {["away", "home"].map((s, i) => (
            <React.Fragment key={s}>
              {i === 1 ? <span className="st-intro__at">at</span> : null}
              <div className={`st-intro__team is-${s}`} style={colors[s].chip}>
                <b>{teams?.[s]?.abbr}</b>
                <span>{teams?.[s]?.name}</span>
                {rec(s) ? <small>{rec(s)}</small> : null}
                {goalie(s) ? <em>In goal: {goalie(s)}</em> : null}
              </div>
            </React.Fragment>
          ))}
        </div>
        {view.situation?.text ? <p className="st-intro__story">{view.situation.text}</p> : null}
        <button type="button" className="st-btn is-lg" onClick={endIntro}>
          Drop the puck
        </button>
      </div>
    );
  } else if (isPausedView) {
    overlay = (
      <div className="st-hold is-pause">
        <span className="st-stamp is-big">Paused</span>
        <button type="button" className="st-btn is-lg" onClick={togglePause} title="Space">
          Resume
        </button>
      </div>
    );
  }

  if (!server) {
    return (
      <div className="st-root st-root--boot">
        <div className="st-bootcard">
          {booting && !err ? (
            <>
              <span className="st-stamp is-big">Dropping the puck…</span>
              <p>Building tonight's game.</p>
            </>
          ) : (
            <>
              <span className="st-stamp is-big">No live game</span>
              <p>{err || "The sim didn't return a game."}</p>
              <button type="button" className="st-btn" onClick={() => onExit?.()}>
                Back to calendar
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  const drawerBody =
    drawer === "lines" ? (
      <LinesPanel bench={view.bench || server.bench} colors={colors} onCoach={sendCoach} live={canCoach} />
    ) : drawer === "tactics" ? (
      <TacticsPanel coach={coach} live={canCoach} onCoach={sendCoach} list={TACTICS} dflt="balanced" />
    ) : drawer === "special" ? (
      <TacticsPanel coach={coach} live={canCoach} onCoach={sendCoach} list={SPECIAL} dflt="passive" />
    ) : drawer === "game" ? (
      <GamePanel
        stats={view.stats}
        events={events}
        teams={teams}
        colors={colors}
        isFinal={isFinal}
        simToEnd={simToEnd}
        onSimToEnd={() => setSimToEnd(true)}
        onLeave={() => setConfirm("leave")}
        onCommit={onCommit}
        working={working}
        languages={languages}
        lang={lang}
        onLang={pickLang}
        compact={compact}
        onCompact={toggleCompact}
        names={names}
        onNames={setNames}
      />
    ) : null;

  const dramaClass = drama ? ` is-drama is-${drama.split("-")[0]}` : "";
  return (
    <div className={`st-root${dramaClass}${celebration ? " is-goal" : ""}${replay ? " is-replay" : ""}`}>
      <header className="st-top">
        <div className="st-top__left">
          <div className="st-transport" role="group" aria-label="Playback">
            <span className={`st-live${isFinal ? " is-final" : isPausedView ? " is-paused" : replay ? " is-replay" : ""}`}>
              <i />
              {isFinal ? "Final" : isPausedView ? "Paused" : replay ? "Replay" : intermission ? "Int." : "Live"}
            </span>
            <button type="button" className={`st-playbtn${held ? " is-paused" : ""}`} onClick={togglePause} disabled={isFinal} title="Pause / resume (Space)" aria-label={held ? "Resume" : "Pause"}>
              <i />
            </button>
            <div className="st-speed" role="radiogroup" aria-label="Playback speed">
              {SPEEDS.map((s, i) => (
                <button key={s} type="button" role="radio" aria-checked={speed === s} className={speed === s ? "is-on" : ""} onClick={() => setSpeed(s)} disabled={isFinal} title={`Speed ${s}x (${i + 1})${s >= 4 ? " — replays are skipped" : ""}`}>
                  {s}x
                </button>
              ))}
            </div>
            <button
              type="button"
              className={`st-pace${paceMode === "real" ? " is-real" : ""}`}
              onClick={togglePace}
              title={paceMode === "real" ? "Real-time pace — click for slower broadcast pace" : "Broadcast pace (60%) — click for real time"}
            >
              {paceMode === "real" ? "Real time" : "Broadcast"}
            </button>
          </div>
        </div>

        <div className="st-top__mid">
          <Scorebug
            teams={teams}
            colors={colors}
            score={isFinal ? server.score || score : score}
            sog={sog}
            period={period}
            clockText={clockText}
            manpower={manpower}
            flashSide={celebration?.side}
            final={isFinal ? server.final || {} : null}
            intermission={intermission}
            late={late}
            stopped={stopped && !intermission && !isFinal}
          />
          <p className="st-situation" aria-live="polite">
            {situation}
          </p>
        </div>

        <div className="st-top__right">
          {!soundOn ? (
            <button type="button" className="st-btn st-soundon" onClick={turnSoundOn} title="Turn on crowd and commentary (M)">
              Sound on
            </button>
          ) : (
            <div className="st-soundwrap">
              <button
                type="button"
                className={`st-voice${neural ? " is-neural" : " is-browser"}${booth.speaking ? " is-speaking" : ""}${prefs.voice ? "" : " is-muted"}`}
                onClick={() => setSoundMenu((v) => !v)}
                title={neural ? "Natural neural voices from the local voice worker" : "Sound and commentary settings"}
              >
                <i />
                <span>
                  {voiceLabel}
                </span>
              </button>
              {soundMenu ? (
                <div className="st-pop">
                  <label className="st-check">
                    <input type="checkbox" checked={prefs.crowd} onChange={() => updatePrefs({ crowd: !prefs.crowd })} /> Crowd &amp; rink sounds
                  </label>
                  <label className="st-check">
                    <input type="checkbox" checked={prefs.voice} onChange={() => updatePrefs({ voice: !prefs.voice })} /> Commentary
                  </label>
                  <label className="st-field">
                    <span>Volume</span>
                    <input className="st-vol" type="range" min="0" max="1" step="0.05" value={prefs.volume} onChange={(e) => updatePrefs({ volume: Number(e.target.value) })} />
                  </label>
                  <label className="st-field">
                    <span>Language</span>
                    <select className="st-select" value={lang} onChange={(e) => pickLang(e.target.value)}>
                      {languages.map((l) => (
                        <option key={l.code} value={l.code}>
                          {l.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="st-pop__note">
                    {neural ? `Neural voices (${engineName}) are on. If a clip is late, that line falls back to the browser voice below.` : "The browser voice below reads the commentary."}
                  </p>
                  {BROWSER_VOICE && booth.choices && booth.choices.length ? (
                    <>
                      <label className="st-field">
                        <span>Browser play-by-play voice</span>
                        <select className="st-select" value={booth.pbpName} onChange={(e) => booth.setPbpPick(e.target.value)}>
                          {booth.choices.map((c) => (
                            <option key={c.name} value={c.name}>
                              {c.natural ? "★ " : ""}
                              {c.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="st-field">
                        <span>Browser analyst voice</span>
                        <select className="st-select" value={booth.colorName || "__same__"} onChange={(e) => booth.setColorPick(e.target.value)}>
                          <option value="__same__">Same voice, lower pitch</option>
                          {booth.choices
                            .filter((c) => c.name !== booth.pbpName)
                            .map((c) => (
                              <option key={c.name} value={c.name}>
                                {c.natural ? "★ " : ""}
                                {c.name}
                              </option>
                            ))}
                        </select>
                      </label>
                      <label className="st-field">
                        <span>Voice speed {Math.round(booth.rateMul * 100)}%</span>
                        <input type="range" min="0.8" max="1.5" step="0.05" value={booth.rateMul} onChange={(e) => booth.setRateMul(e.target.value)} />
                      </label>
                    </>
                  ) : null}
                  <p className="st-pop__note">
                    ★ = natural-sounding browser voice.
                    {!booth.hasNatural ? " Only robotic browser voices found — Microsoft Edge has free natural ones (Aria, Guy, Jenny)." : ""}
                    {booth.stalls ? ` Speech engine stalled ${booth.stalls}× and was restarted.` : ""}
                  </p>
                  {booth.status === "novoice" ? <p className="st-pop__note">Add a {langLabel} voice in Windows: Settings → Time &amp; language → Speech → Add voices.</p> : null}
                  <button type="button" className="st-btn is-ghost is-sm" onClick={() => setSoundOn(false)}>
                    Mute all (M)
                  </button>
                </div>
              ) : null}
            </div>
          )}
          <div className="st-bellwrap">
            <button type="button" className={`st-bell${unread ? " has-unread" : ""}${logOpen ? " is-on" : ""}`} onClick={toggleLog} title="Coaching log (N)" aria-label={`Coaching log, ${unread} unread`}>
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M10 2.5c-3 0-5 2.3-5 5.2v3.6L3.4 14h13.2L15 11.3V7.7c0-2.9-2-5.2-5-5.2zM8 15.5a2 2 0 0 0 4 0" />
              </svg>
              {unread ? <em>{unread > 9 ? "9+" : unread}</em> : null}
            </button>
            {logOpen ? (
              <NotifLog
                list={notifList}
                statusOf={statusOf}
                onAct={actNotif}
                onDismiss={dismissNotif}
                live={live}
                pauseCritical={pauseCritical}
                setPauseCritical={setPauseCritical}
                onClose={() => setLogOpen(false)}
              />
            ) : null}
          </div>
          <button type="button" className="st-help-btn" onClick={() => setHelp(true)} title="How this screen works (H)" aria-label="Help">
            ?
          </button>
        </div>
      </header>

      <main className={`st-stage${drama ? ` is-${drama.split("-")[0]}` : ""}${statsOpen && statsPinned ? " has-an-pinned" : ""}`}>
        <Crowd crowd={view.crowd} colors={colors} drama={drama ? drama.split("-")[0] : null} goal={!!celebration} chant={replay ? null : chant} />
        <Rink
          apiRef={rinkApi}
          onIce={onIce}
          roster={roster}
          colors={colors}
          teams={teams}
          period={period}
          shots={shots}
          headPt={headPt}
          celebration={celebration}
          saveFlash={saveFlash}
          drama={drama}
          dramaText={banner || dramaText}
          alert={live && !overlay && !replay && !hasNotifFeed ? alert : null}
          onAlertAction={alertAction}
          onAlertDismiss={dismissAlert}
          tracking={hasTracking}
          buffering={buffering && !overlay}
          stop={stop}
          faceoff={faceoff}
          foWin={foWin}
          danger={danger}
          hit={hit}
          delayed={delayed}
          statPop={replay ? null : statPop}
          replay={replay}
          onSkipReplay={skipReplay}
          boxed={boxed}
          scale={compact ? 0.85 : 1}
          names={names}
          strip={<OnIceStrip onIce={onIce} roster={roster} colors={colors} teams={teams} analytics={analytics} shiftStart={pb.current.shiftStart} gameSec={gameSec} />}
        />
        <Caption line={captionLine} live={captionLive} />
        <Momentum momentum={view.momentum} colors={colors} teams={teams} />
        {scorer && !overlay ? <ScorerCard card={scorer} colors={colors} teams={teams} /> : null}
        {cards.length && !overlay ? (
          <div className="st-nstack" aria-live="polite">
            {cards
              .map((id) => notifById.get(id))
              .filter((n) => n && statusOf(n) === "new")
              .map((n) => (
                <NotifCard key={n.id} n={n} onAct={actNotif} onDismiss={dismissNotif} live={live} />
              ))}
          </div>
        ) : null}
        {statsOpen ? (
          <StatsPanel
            an={analytics}
            roster={roster}
            colors={colors}
            teams={teams}
            userSide={userSide}
            bench={view.bench || server.bench}
            onIceIds={onIceIds}
            pinned={statsPinned}
            onPin={() => setStatsPinned((p) => !p)}
            onClose={() => setStatsOpen(false)}
            onCoach={sendCoach}
            live={canCoach}
            ozKind={ozKind}
          />
        ) : null}
        {overlay ? <div className="st-cover">{overlay}</div> : null}
        {drawer ? (
          <section className={`st-drawer is-${drawer}`} aria-label={DOCK_TABS.find((d) => d[0] === drawer)?.[1]}>
            <button type="button" className="st-drawer__x st-btn is-ghost is-sm" onClick={() => setDrawer(null)} title="Close (Esc)">
              Close
            </button>
            {drawerBody}
          </section>
        ) : null}
      </main>

      <footer className="st-dock">
        <nav className="st-tabs" aria-label="Coaching">
          {DOCK_TABS.map(([k, label, tip]) => (
            <button key={k} type="button" className={`st-tab${drawer === k ? " is-on" : ""}`} onClick={() => {
                if (statsOpen && !statsPinned) setStatsOpen(false);
                setDrawer((d) => (d === k ? null : k));
              }} title={tip} aria-expanded={drawer === k}>
              <TabIcon k={k} />
              {label}
              {k === "special" && (mpState === "PP" || mpState === "SH") ? <em className="st-tab__dot" /> : null}
            </button>
          ))}
          <button type="button" className={`st-tab${statsOpen ? " is-on" : ""}`} onClick={toggleStats} title="Line, player, team and goalie analytics (S)" aria-expanded={statsOpen}>
            <TabIcon k="stats" />
            Stats
          </button>
        </nav>
        {toast ? (
          <div key={toast.key} className={`st-ordertoast is-${toast.status}`} aria-live="polite">
            {toast.text}
          </div>
        ) : null}
        <div className="st-calls">
          <button type="button" className={`st-call${coach.timeout_used ? " is-used" : ""}`} disabled={!live || !!coach.timeout_used} onClick={doTimeout} title={coach.timeout_used ? "Timeout already used" : "Call your one timeout (T)"}>
            <b>{coach.timeout_used ? "Timeout used" : "Timeout"}</b>
            <small>{coach.timeout_used ? "one per game" : "T"}</small>
          </button>
          {coach.goalie_pulled ? (
            <button type="button" className="st-call is-hot" disabled={!live} onClick={doPull} title="Put the goalie back in (G)">
              <b>Recall goalie</b>
              <small>net is empty</small>
            </button>
          ) : (
            <button
              type="button"
              className={`st-call${pullArmed ? " is-armed" : ""}`}
              disabled={!pullLegal}
              onClick={doPull}
              title={mpState !== "EV" ? "Only at even strength" : pullArmed ? "Click again to confirm" : "Extra attacker, empty net (G, press twice)"}
            >
              <b>{pullArmed ? "Confirm pull?" : "Pull goalie"}</b>
              <small>{pullArmed ? "click again" : mpState !== "EV" ? "even strength only" : "G ×2"}</small>
            </button>
          )}
        </div>
      </footer>

      {help ? <Help colors={colors} teams={teams} onClose={() => setHelp(false)} /> : null}

      {confirm ? (
        <div className="st-modal" role="dialog" aria-modal="true">
          <div className="st-confirm">
            <span className="st-stamp is-big">{confirm === "discard" ? "Discard result?" : "Leave the game?"}</span>
            <p>
              {confirm === "discard"
                ? "This live result won't be saved. The night gets fast-simmed when you advance."
                : "You'll walk out of the rink. The rest of tonight's game gets fast-simmed when you advance."}
            </p>
            <div className="st-confirm__row">
              <button type="button" className="st-btn" onClick={() => setConfirm(null)} disabled={working}>
                Keep watching
              </button>
              <button type="button" className="st-btn is-ghost" onClick={onLeave} disabled={working}>
                {working ? "Leaving…" : confirm === "discard" ? "Discard & leave" : "Leave game"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {err ? (
        <div className="st-toast" role="alert">
          <b>!</b>
          <span>{err}</span>
          <button type="button" className="st-btn is-ghost is-sm" onClick={() => setErr("")}>
            OK
          </button>
        </div>
      ) : null}
    </div>
  );
}
