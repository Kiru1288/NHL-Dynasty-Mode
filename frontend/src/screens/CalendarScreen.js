// frontend/src/components/game/CalendarScreen.js
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { advanceFranchise } from "../services/franchiseService";
import { formatFranchiseApiError } from "../services/api";
import WorldJuniorsEvent from "../events/worldJuniors/WorldJuniorsEvent";
import { resolveWorldJuniorsPayload } from "../events/worldJuniors/WorldJuniorsMenu";

/**
 * CalendarScreen.js
 * Full visual overhaul inspired by a premium hockey franchise calendar layout.
 *
 * FIXED VERSION GOALS:
 * - Keeps backend-connected data paths intact.
 * - Does not hardcode real players.
 * - Does not require new files.
 * - Uses live franchiseState/game data wherever available.
 * - Properly renders special events directly on the calendar.
 * - Gives special events visible icons/badges on the day tile.
 * - Allows special-event-only dates to be visible, hoverable, clickable, and explainable.
 * - Fixes Team Only toggle so it can actually turn on/off.
 * - Fixes drawerOpen so the drawer can actually open.
 * - Removes duplicate special event helper definitions inside game tile.
 * - Merges event feeds instead of accidentally using only the first available array.
 * - Improves missing-score handling so unplayed games do not become fake 0-0 finals.
 */

const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const WEEKDAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

const LONG_WEEKDAY_NAMES = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

const SCREEN_KEYS = {
  office: "office",
  calendar: "calendar",
  lineup: "lineup",
  roster: "roster",
  prospects: "prospects",
  scouting: "scouting",
  analytics: "analytics",
  finances: "finances",
  inbox: "inbox",
  settings: "settings",
  storylines_report: SCREENS.STORYLINES,
  injury_report: "injury_report",
  draft: "draft",
  standings: "standings",
  league: "league",
};

const NAV_ALIAS_TO_SCREEN = {
  [SCREEN_KEYS.lineup]: SCREENS.ROSTER,
  [SCREEN_KEYS.roster]: SCREENS.ROSTER,
  [SCREEN_KEYS.scouting]: SCREENS.SCOUTING,
  [SCREEN_KEYS.analytics]: SCREENS.STATS,
  [SCREEN_KEYS.finances]: SCREENS.OFFICE,
  [SCREEN_KEYS.inbox]: SCREENS.OFFICE,
  [SCREEN_KEYS.standings]: SCREENS.STATS,
  [SCREEN_KEYS.league]: SCREENS.STATS,
  [SCREEN_KEYS.draft]: SCREENS.DRAFT_CLASS,
  [SCREEN_KEYS.storylines_report]: SCREENS.STORYLINES,
  
};

const EMPTY_ARRAY = Object.freeze([]);
const EMPTY_OBJECT = Object.freeze({});

const PRIORITY_ORDER = {
  CRITICAL: 0,
  HIGH: 1,
  MEDIUM: 2,
  LOW: 3,
};

const EVENT_TYPE_LABELS = {
  preseason_start: "Preseason Begins",
  preseason_finale: "Preseason Finale",
  opening_night: "Opening Night",
  home_opener: "Home Opener",
  thanksgiving_checkpoint: "Thanksgiving Checkpoint",
  wjc_start: "World Juniors Begin",
  wjc_semifinals: "World Juniors Semifinals",
  wjc_final: "World Juniors Final",
  roster_freeze: "Holiday Roster Freeze",
  winter_classic: "Winter Classic",
  heritage_classic: "Heritage Classic",
  stadium_series: "Stadium Series",
  all_star_weekend: "All-Star Weekend",
  allstar_game: "All-Star Game",
  trade_deadline: "Trade Deadline",
  playoff_push: "Playoff Race Push",
  regular_season_finale: "Regular Season Finale",
  playoffs_start: "Stanley Cup Playoffs Begin",
  draft_lottery: "Draft Lottery",
  conference_finals: "Conference Finals",
  stanley_cup_final: "Stanley Cup Final",
  stanley_cup_finals: "Stanley Cup Final",
  nhl_awards: "NHL Awards",
  nhl_draft: "NHL Draft",
  free_agency: "Free Agency Opens",
  development_camp: "Development Camp",
  training_camp: "Training Camp",
  injury: "Injury Report",
  injury_report: "Injury Report",
  trade: "Trade Bulletin",
  roster_move: "Roster Move",
  waiver: "Waiver Wire",
  callup: "Call-Up",
  milestone: "Milestone Watch",
  rivalry: "Rivalry Game",
  showcase_game: "Showcase Game",
  wjc_tournament: "World Juniors",
  four_nations_tournament: "4 Nations / International Tournament",
  four_nations_faceoff: "4 Nations Face-Off",
};

const LOGO_CONTEXT = (() => {
  try {
    return require.context("../logos", false, /\.(png|jpg|jpeg|webp|svg)$/i);
  } catch (_err) {
    return null;
  }
})();

const TEAM_LOGO_NAME_OVERRIDES = {
  ANA: "Anaheim",
  ARI: "Arizona",
  BOS: "Boston",
  BUF: "Buffalo",
  CGY: "Calgary",
  CAR: "Carolina",
  CHI: "Chicago",
  COL: "Colorado",
  CBJ: "Columbus",
  DAL: "Dallas",
  DET: "Detroit",
  EDM: "Edmonton",
  FLA: "Florida",
  LAK: "Los Angeles",
  MIN: "Minnesota",
  MTL: "Montreal",
  NSH: "Nashville",
  NJD: "New Jersey",
  NYI: "NY Islanders",
  NYR: "NY Rangers",
  OTT: "Ottawa",
  PHI: "Philadelphia",
  PIT: "Pittsburgh",
  SJS: "San Jose",
  SEA: "Seattle",
  STL: "St. Louis",
  TBL: "Tampa Bay",
  TOR: "Toronto",
  VAN: "Vancouver",
  VGK: "Vegas",
  WSH: "Washington",
  WPG: "Winnipeg",
};

const TEAM_ABBR_LOGO_ALIASES = {
  WIN: "WPG",
  WPG: "WPG",
  NJ: "NJD",
  DEV: "NJD",
  LAS: "VGK",
  VEG: "VGK",
  MON: "MTL",
  CAL: "CGY",
  WAS: "WSH",
  TB: "TBL",
  SJ: "SJS",
  LA: "LAK",
  NAS: "NSH",
  CLB: "CBJ",
  ARZ: "ARI",
  UTA: "UTA",
  STL: "STL",
  ST: "STL",
};

/** Canonical NHL name / city → abbreviation (never "NEW" / "ST." / "CAL"). */
const NHL_NAME_TO_ABBR = {
  "anaheim ducks": "ANA",
  anaheim: "ANA",
  ducks: "ANA",
  "arizona coyotes": "ARI",
  arizona: "ARI",
  coyotes: "ARI",
  "boston bruins": "BOS",
  boston: "BOS",
  bruins: "BOS",
  "buffalo sabres": "BUF",
  buffalo: "BUF",
  sabres: "BUF",
  "calgary flames": "CGY",
  calgary: "CGY",
  flames: "CGY",
  "carolina hurricanes": "CAR",
  carolina: "CAR",
  hurricanes: "CAR",
  "chicago blackhawks": "CHI",
  chicago: "CHI",
  blackhawks: "CHI",
  "colorado avalanche": "COL",
  colorado: "COL",
  avalanche: "COL",
  "columbus blue jackets": "CBJ",
  columbus: "CBJ",
  "blue jackets": "CBJ",
  "dallas stars": "DAL",
  dallas: "DAL",
  stars: "DAL",
  "detroit red wings": "DET",
  detroit: "DET",
  "red wings": "DET",
  "edmonton oilers": "EDM",
  edmonton: "EDM",
  oilers: "EDM",
  "florida panthers": "FLA",
  florida: "FLA",
  panthers: "FLA",
  "los angeles kings": "LAK",
  "los angeles": "LAK",
  kings: "LAK",
  "minnesota wild": "MIN",
  minnesota: "MIN",
  wild: "MIN",
  "montreal canadiens": "MTL",
  montreal: "MTL",
  canadiens: "MTL",
  "nashville predators": "NSH",
  nashville: "NSH",
  predators: "NSH",
  "new jersey devils": "NJD",
  "new jersey": "NJD",
  devils: "NJD",
  "new york islanders": "NYI",
  islanders: "NYI",
  "ny islanders": "NYI",
  "new york rangers": "NYR",
  rangers: "NYR",
  "ny rangers": "NYR",
  "ottawa senators": "OTT",
  ottawa: "OTT",
  senators: "OTT",
  "philadelphia flyers": "PHI",
  philadelphia: "PHI",
  flyers: "PHI",
  "pittsburgh penguins": "PIT",
  pittsburgh: "PIT",
  penguins: "PIT",
  "san jose sharks": "SJS",
  "san jose": "SJS",
  sharks: "SJS",
  "seattle kraken": "SEA",
  seattle: "SEA",
  kraken: "SEA",
  "st louis blues": "STL",
  "st. louis blues": "STL",
  "st louis": "STL",
  "st. louis": "STL",
  blues: "STL",
  "tampa bay lightning": "TBL",
  "tampa bay": "TBL",
  lightning: "TBL",
  "toronto maple leafs": "TOR",
  toronto: "TOR",
  "maple leafs": "TOR",
  "utah hockey club": "UTA",
  utah: "UTA",
  "vancouver canucks": "VAN",
  vancouver: "VAN",
  canucks: "VAN",
  "vegas golden knights": "VGK",
  vegas: "VGK",
  "golden knights": "VGK",
  "washington capitals": "WSH",
  washington: "WSH",
  capitals: "WSH",
  "winnipeg jets": "WPG",
  winnipeg: "WPG",
  jets: "WPG",
};

const KNOWN_NHL_ABBRS = new Set(Object.values(NHL_NAME_TO_ABBR));

const TEAM_LOGO_MAP = (() => {
  const map = new Map();
  if (!LOGO_CONTEXT) return map;

  const keys = LOGO_CONTEXT.keys();

  keys.forEach((key) => {
    const src = LOGO_CONTEXT(key);
    const rawFile = String(key || "").replace(/^.\//, "");
    const stem = rawFile.replace(/\.[^.]+$/, "");
    const cleaned = stem.replace(/\s+\d+$/, "").trim();
    const normalized = normalizeLogoToken(cleaned);

    if (normalized && !map.has(normalized)) {
      map.set(normalized, src);
    }
  });
  return map;
})();

function normalizeLogoToken(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "")
    .trim();
}

function normalizeKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

function sanitizeDisplayText(value) {
  if (value === null || value === undefined) return "";

  return String(value)
    .replace(/ΓÇÖ/g, "'")
    .replace(/ΓÇ£/g, '"')
    .replace(/ΓÇ¥/g, '"')
    .replace(/ΓÇô/g, "-")
    .replace(/ΓÇö/g, "-")
    .replace(/â€™/g, "'")
    .replace(/â€œ/g, '"')
    .replace(/â€\x9d/g, '"')
    .replace(/â€"/g, "-")
    .replace(/â€˜/g, "'")
    .replace(/\uFFFD/g, "'")
    .trim();
}

const SPECIAL_EVENT_LOGO_STEMS = {
  wjc: "wjc logo",
  world_juniors: "wjc logo",
  world_junior: "wjc logo",
  wjc_tournament: "wjc logo",
  wjc_semifinals: "wjc logo",

  heritage_classic: "heritage classic",
  heritage: "heritage classic",

  four_nations: "4nationslogo",
  four_nations_faceoff: "4nationslogo",
  four_nations_tournament: "4nationslogo",
  four_nations_logo: "4nationslogo",
  "4_nations": "4nationslogo",
  "4nations": "4nationslogo",
  "4nationslogo": "4nationslogo",

  trade_deadline: "trade deadline logo",
  deadline: "trade deadline logo",
  tdk: "trade deadline logo",

  winter_classic: "winterclassic",
  winter: "winterclassic",
};

function getLogoFromContextByStem(stem) {
  if (!LOGO_CONTEXT || !stem) return null;

  const target = normalizeLogoToken(stem);
  const keys = LOGO_CONTEXT.keys();

  const matchedKey = keys.find((key) => {
    const rawFile = String(key || "").replace(/^.\//, "");
    const fileStem = rawFile.replace(/\.[^.]+$/, "");
    return normalizeLogoToken(fileStem) === target;
  });

  return matchedKey ? LOGO_CONTEXT(matchedKey) : null;
}

function getSpecialEventLogoSrc(eventOrType) {
  const event =
    eventOrType && typeof eventOrType === "object"
      ? eventOrType
      : { type: eventOrType };

  const direct =
    event.logoSrc ||
    event.logo_src ||
    event.logo ||
    event.image ||
    event.image_src ||
    event.badge ||
    event.badge_src ||
    "";

  if (direct) return direct;

  const rawType =
    event.event_type ||
    event.eventType ||
    event.kind ||
    event.type ||
    event.category ||
    event.subtype ||
    event.eventKind ||
    "";

  const key = normalizeKey(rawType);
  const compactKey = normalizeLogoToken(rawType);

  const candidateKeys = [
    key,
    compactKey,
    event.logo_key,
    event.logoKey,
    event.event_logo,
    event.eventLogo,
  ]
    .map((value) => normalizeKey(value))
    .filter(Boolean);

  for (const candidate of candidateKeys) {
    if (SPECIAL_EVENT_LOGO_STEMS[candidate]) {
      const src = getLogoFromContextByStem(SPECIAL_EVENT_LOGO_STEMS[candidate]);
      if (src) return src;
    }
  }

  if (key.includes("wjc") || key.includes("world_junior") || key.includes("world_juniors")) {
    return getLogoFromContextByStem("wjc logo");
  }

  if (key.includes("heritage")) {
    return getLogoFromContextByStem("heritage classic");
  }

  if (
    key.includes("4_nations") ||
    key.includes("4nations") ||
    key.includes("four_nations") ||
    key.includes("nations_faceoff")
  ) {
    return getLogoFromContextByStem("4nationslogo");
  }

  if (key.includes("trade_deadline") || key === "tdk" || key.includes("deadline")) {
    return getLogoFromContextByStem("trade deadline logo");
  }

  if (key.includes("winter")) {
    return getLogoFromContextByStem("winterclassic");
  }

  return null;
}

function firstDefined(...values) {
  for (const value of values) {
    if (value !== null && value !== undefined && value !== "") return value;
  }

  return undefined;
}

function firstNumberOrNull(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;

    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }

  return null;
}

function firstNumber(...values) {
  const found = firstNumberOrNull(...values);
  return found === null ? 0 : found;
}

function safeNumber(value, fallback = "—") {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;

  if (Math.abs(number) >= 1000) {
    return number.toLocaleString();
  }

  return number;
}

function safeScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.round(n)) : null;
}

function normalizeArray(...values) {
  for (const value of values) {
    if (Array.isArray(value)) return value;

    if (value && typeof value === "object") {
      if (Array.isArray(value.items)) return value.items;
      if (Array.isArray(value.data)) return value.data;
      if (Array.isArray(value.results)) return value.results;
      if (Array.isArray(value.rows)) return value.rows;
      if (Array.isArray(value.games)) return value.games;
      if (Array.isArray(value.events)) return value.events;
    }
  }

  return [];
}

function normalizeArrayMerged(...values) {
  const output = [];

  values.forEach((value) => {
    if (Array.isArray(value)) {
      output.push(...value);
      return;
    }

    if (value && typeof value === "object") {
      if (Array.isArray(value.items)) output.push(...value.items);
      if (Array.isArray(value.data)) output.push(...value.data);
      if (Array.isArray(value.results)) output.push(...value.results);
      if (Array.isArray(value.rows)) output.push(...value.rows);
      if (Array.isArray(value.games)) output.push(...value.games);
      if (Array.isArray(value.events)) output.push(...value.events);
    }
  });

  return output;
}

function dedupeByStableKey(rows, keyBuilder) {
  const seen = new Set();
  const output = [];

  rows.forEach((row, index) => {
    const key = keyBuilder(row, index);
    if (!key || seen.has(key)) return;
    seen.add(key);
    output.push(row);
  });

  return output;
}

function getSpecialEventDefaultTitle(type) {
  const key = normalizeKey(type);

  if (EVENT_TYPE_LABELS[key]) return EVENT_TYPE_LABELS[key];

  if (key.includes("injury")) return "Injury Report";
  if (key.includes("trade_deadline")) return "Trade Deadline";
  if (key.includes("trade")) return "Trade Bulletin";
  if (key.includes("callup") || key.includes("call_up")) return "Call-Up";
  if (key.includes("waiver")) return "Waiver Wire";
  if (key.includes("roster")) return "Roster Move";
  if (key.includes("milestone")) return "Milestone Watch";
  if (key.includes("playoff")) return "Playoff Race Update";
  if (key.includes("stanley")) return "Stanley Cup Update";
  if (key.includes("draft")) return "Draft Update";
  if (key.includes("all_star") || key.includes("allstar")) return "All-Star Event";
  if (key.includes("winter")) return "Winter Classic";
  if (key.includes("heritage")) return "Heritage Classic";
  if (key.includes("stadium")) return "Stadium Series";
  if (key.includes("wjc") || key.includes("world_junior")) return "World Juniors";
  if (key.includes("free_agency")) return "Free Agency Opens";
  if (key.includes("rivalry")) return "Rivalry Night";
  if (key.includes("showcase")) return "Showcase Event";

  return "League Event";
}

function getSpecialEventIcon(type) {
  const key = normalizeKey(type);

  if (key.includes("trade_deadline")) return "⏳";
  if (key.includes("trade")) return "⇄";
  if (key.includes("draft_lottery")) return "🎲";
  if (key.includes("draft")) return "◈";
  if (key.includes("all_star") || key.includes("allstar")) return "★";
  if (key.includes("winter")) return "❄";
  if (key.includes("heritage") || key.includes("stadium")) return "🏟";
  if (key.includes("wjc") || key.includes("world")) return "🌍";
  if (key.includes("playoff") || key.includes("stanley")) return "🏆";
  if (key.includes("free_agency")) return "$";
  if (key.includes("injury")) return "✚";
  if (key.includes("rivalry")) return "⚔";
  if (key.includes("milestone")) return "★";
  if (key.includes("waiver")) return "↕";
  if (key.includes("callup") || key.includes("call_up")) return "↑";
  if (key.includes("roster")) return "⇅";
  if (key.includes("opening")) return "◉";
  if (key.includes("finale")) return "◌";
  if (key.includes("camp")) return "⌁";
  if (key.includes("award")) return "♛";
  if (key.includes("showcase")) return "◆";

  return "◆";
}

function getSpecialEventPriority(type, explicitPriority) {
  const explicit = String(explicitPriority || "").toUpperCase();
  if (["CRITICAL", "HIGH", "MEDIUM", "LOW"].includes(explicit)) return explicit;

  const key = normalizeKey(type);

  if (
    key.includes("trade_deadline") ||
    key.includes("playoffs_start") ||
    key.includes("stanley") ||
    key.includes("draft_lottery") ||
    key.includes("nhl_draft") ||
    key.includes("free_agency")
  ) {
    return "CRITICAL";
  }

  if (
    key.includes("winter") ||
    key.includes("heritage") ||
    key.includes("stadium") ||
    key.includes("all_star") ||
    key.includes("allstar") ||
    key.includes("wjc") ||
    key.includes("world_junior") ||
    key.includes("injury")
  ) {
    return "HIGH";
  }

  if (
    key.includes("trade") ||
    key.includes("milestone") ||
    key.includes("roster") ||
    key.includes("waiver") ||
    key.includes("callup") ||
    key.includes("rivalry")
  ) {
    return "MEDIUM";
  }

  return "LOW";
}

function getSpecialEventTone(type, priority) {
  const key = normalizeKey(type);
  const p = String(priority || "").toUpperCase();

  if (p === "CRITICAL") return "critical";
  if (key.includes("injury")) return "medical";
  if (key.includes("trade")) return "trade";
  if (key.includes("draft")) return "draft";
  if (key.includes("playoff") || key.includes("stanley")) return "playoff";
  if (key.includes("all_star") || key.includes("allstar")) return "star";
  if (key.includes("winter") || key.includes("heritage") || key.includes("stadium")) return "showcase";
  if (key.includes("wjc") || key.includes("world")) return "international";
  if (key.includes("milestone")) return "milestone";

  return p === "HIGH" ? "important" : "league";
}

function getWorldJuniorsEventPayload(franchiseState) {
  const resolved = resolveWorldJuniorsPayload(franchiseState, null);
  if (resolved.hasData && resolved.raw) return resolved.raw;
  return EMPTY_OBJECT;
}

function isWorldJuniorsCalendarEvent(event) {
  if (!event || typeof event !== "object") return false;

  const type = normalizeKey(
    event.type || event.kind || event.event_type || event.eventType || event.category || ""
  );
  const id = normalizeKey(event.id || event.event_id || event.eventId || "");

  return (
    type.includes("wjc") ||
    type.includes("world_junior") ||
    type.includes("world_juniors") ||
    id.includes("wjc") ||
    id.includes("world_junior")
  );
}

function isWorldJuniorsWindowActive(franchiseState) {
  if (findPendingWjcPopup(franchiseState)) return true;
  const wjc = getWorldJuniorsEventPayload(franchiseState);
  const status = normalizeKey(
    wjc?.status || wjc?.tournament_status || wjc?.phase || wjc?.wjc_phase || ""
  );

  return status === "active" || status === "live" || status === "complete";
}

function findPendingWjcPopup(franchiseState) {
  const pops = normalizeArrayMerged(
    franchiseState?.pending_ui_popups,
    franchiseState?.pendingUiPopups
  );

  return (
    pops.find(
      (pop) =>
        pop &&
        (pop.kind === "wjc_tournament" || pop.wjc_live === true || pop.wjc_phase)
    ) || null
  );
}

function getPlayoffEventPayload(franchiseState) {
  return (
    franchiseState?.playoffs ||
    franchiseState?.playoff_data ||
    franchiseState?.playoffData ||
    EMPTY_OBJECT
  );
}

function isPlayoffReady(franchiseState) {
  const phase = String(
    franchiseState?.season_phase || franchiseState?.phase || ""
  ).toLowerCase();
  return phase === "playoff_ready" || Boolean(franchiseState?.flags?.can_enter_playoffs);
}

function isPlayoffsStartCalendarEvent(event) {
  if (!event || typeof event !== "object") return false;
  const type = normalizeKey(
    event.type || event.kind || event.event_type || event.eventType || event.category || ""
  );
  const id = normalizeKey(event.id || event.event_id || event.eventId || "");
  return (
    type.includes("playoffs_start") ||
    type.includes("playoff_start") ||
    id.includes("playoffs_start") ||
    id.includes("playoff_start")
  );
}

function findPendingPlayoffPopup(franchiseState) {
  const pops = normalizeArrayMerged(
    franchiseState?.pending_ui_popups,
    franchiseState?.pendingUiPopups
  );

  return pops.find((pop) => pop && pop.kind === "playoff_start") || null;
}

function isDateInWorldJuniorsWindow(iso, seasonYear) {
  if (!iso || !seasonYear) return false;

  try {
    const [y, m, d] = iso.split("-").map((part) => Number(part));
    const dt = new Date(y, m - 1, d);
    const sy = Number(seasonYear);

    if (dt.getMonth() === 11 && dt.getDate() >= 26 && dt.getFullYear() === sy) return true;
    if (dt.getMonth() === 0 && dt.getDate() <= 5 && dt.getFullYear() === sy + 1) return true;
  } catch {
    return false;
  }

  return false;
}

function normalizeCalendarSpecialEvent(event, index = 0) {
  if (typeof event === "string") {
    return {
      id: `special-event-string-${index}`,
      date: "",
      title: event,
      headline: event,
      type: "league_event",
      priority: "MEDIUM",
      description: "",
      icon: "◆",
      logoSrc: null,
      tone: "league",
      raw: event,
    };
  }

  if (!event || typeof event !== "object") {
    return {
      id: `special-event-empty-${index}`,
      date: "",
      title: "League Event",
      headline: "League Event",
      type: "league_event",
      priority: "LOW",
      description: "",
      icon: "◆",
      logoSrc: null,
      tone: "league",
      raw: event,
    };
  }

  const type = normalizeKey(
    event.event_type ||
      event.eventType ||
      event.kind ||
      event.type ||
      event.category ||
      event.subtype ||
      event.eventKind ||
      "league_event"
  );

  const date =
    event.calendar_iso ||
    event.calendarIso ||
    event.date ||
    event.event_date ||
    event.eventDate ||
    event.start_date ||
    event.startDate ||
    event.day ||
    event.iso ||
    "";

  const title = sanitizeDisplayText(
    event.title ||
      event.headline ||
      event.name ||
      event.summary ||
      event.label ||
      event.subject ||
      getSpecialEventDefaultTitle(type)
  );

  const description = sanitizeDisplayText(
    event.description ||
      event.details ||
      event.body ||
      event.message ||
      event.text ||
      event.effect_summary ||
      event.summary ||
      ""
  );

  const priority = getSpecialEventPriority(type, event.priority || event.importance || event.severity);

  const id =
    event.id ||
    event.event_id ||
    event.eventId ||
    event.storyline_id ||
    event.storylineId ||
    event.notification_id ||
    event.notificationId ||
    `${type}-${toISODate(date) || "no-date"}-${index}`;

  return {
    ...event,
    id,
    date,
    title,
    headline: sanitizeDisplayText(event.headline || title),
    type,
    priority,
    description,
    icon: event.icon || getSpecialEventIcon(type),
    logoSrc: getSpecialEventLogoSrc({
      ...event,
      type,
    }),
    tone: event.tone || getSpecialEventTone(type, priority),
    effects: event.effects || EMPTY_OBJECT,
    effect_summary: event.effect_summary || event.effectSummary || "",
    team_id: event.team_id || event.teamId || event.team || "",
    player_id: event.player_id || event.playerId || "",
    player_name: event.player_name || event.playerName || event.player || "",
  };
}

function pickInjuryGamesRemaining(inj) {
  if (!inj || typeof inj !== "object") return 0;

  const keys = ["games_remaining", "injury_games_remaining", "days_remaining", "gamesRemaining"];

  for (let i = 0; i < keys.length; i += 1) {
    const n = Number(inj[keys[i]]);
    if (Number.isFinite(n) && n >= 0) return Math.floor(n);
  }

  return 0;
}

function normalizeInjuryRowForUi(inj, idx) {
  const gamesRemaining = pickInjuryGamesRemaining(inj);
  const teamAbbr = inj?.team_abbr || inj?.team_abbrev || inj?.teamAbbr || "";
  const teamId = inj?.team_id || inj?.teamId || inj?.team || "";
  const status = inj?.status || inj?.injury_status || inj?.health_status || "—";
  const injuryLabel =
    inj?.injury ||
    inj?.description ||
    inj?.injury_type ||
    inj?.injuryType ||
    inj?.tier ||
    inj?.severity ||
    "Injury";

  const returnText =
    inj?.return_estimate ||
    inj?.returnEstimate ||
    inj?.return_date ||
    inj?.returnDate ||
    (gamesRemaining > 0 ? `In ${gamesRemaining} games` : "") ||
    "";

  const duration = inj?.duration || (gamesRemaining > 0 ? `${gamesRemaining} games` : "");

  return {
    id: inj?.id || inj?.injury_id || inj?.injuryId || `inj-fallback-${idx}`,
    player: inj?.player_name || inj?.playerName || inj?.player || inj?.name || "Player",
    playerName: inj?.player_name || inj?.playerName || inj?.player || inj?.name || "Player",
    tier: inj?.tier || inj?.severity || inj?.injury_type || inj?.injuryType || "—",
    games: gamesRemaining,
    gamesRemaining,
    date: inj?.calendar_iso || inj?.calendarIso || inj?.date || inj?.return_date || inj?.returnDate || "",
    returnText,
    status,
    injuryLabel,
    description: inj?.description || "",
    teamAbbr,
    teamId,
    position: inj?.position || inj?.pos || "—",
    severity: inj?.severity || inj?.tier || inj?.injury_type || inj?.injuryType || "—",
    duration,
    raw: inj,
  };
}

function buildInjuryEventsFromRows(injuries) {
  return (injuries || [])
    .map((injury, index) => {
      const row = normalizeInjuryRowForUi(injury, index);
      const iso = toISODate(row.date);

      if (!iso) return null;

      return normalizeCalendarSpecialEvent(
        {
          id: `injury-event-${row.id}`,
          calendar_iso: iso,
          type: "injury_report",
          priority: row.teamId ? "HIGH" : "MEDIUM",
          title: `${row.playerName} Injury`,
          headline: `${row.playerName} Injury`,
          description:
            `${row.teamAbbr || row.teamId || "Team"} · ${row.injuryLabel}` +
            (row.gamesRemaining > 0 ? ` · ${row.gamesRemaining} games` : ""),
          icon: "✚",
          team_id: row.teamId,
          player_name: row.playerName,
          injury_row: row,
        },
        index
      );
    })
    .filter(Boolean);
}

function CalendarSpecialEventTile({ event, compact, onOpen }) {
  const priority = String(event?.priority || "MEDIUM").toLowerCase();
  const type = normalizeKey(event?.type || "league_event");
  const tone = normalizeKey(event?.tone || getSpecialEventTone(type, event?.priority));
  const logoSrc = event?.logoSrc || getSpecialEventLogoSrc(event);

  return (
    <div
      className={`nhlcal-special-event-tile priority-${priority} type-${type} tone-${tone} ${
        logoSrc ? "has-logo" : ""
      }`}
      role="button"
      tabIndex={0}
      title={event?.description || event?.title || "League Event"}
      onClick={(clickEvent) => {
        clickEvent.stopPropagation();
        onOpen?.(event);
      }}
      onKeyDown={(keyEvent) => {
        if (keyEvent.key === "Enter" || keyEvent.key === " ") {
          keyEvent.preventDefault();
          keyEvent.stopPropagation();
          onOpen?.(event);
        }
      }}
    >
      <div className="nhlcal-special-event-icon">
        {logoSrc ? (
          <img src={logoSrc} alt={`${event?.title || "Special event"} logo`} loading="lazy" />
        ) : (
          event?.icon || "◆"
        )}
      </div>

      <div className="nhlcal-special-event-copy">
        <strong title={event?.title || "League Event"}>
          {event?.title || "League Event"}
        </strong>
        {!compact && event?.description ? (
          <span title={event.description}>{event.description}</span>
        ) : null}
      </div>
    </div>
  );
}

function SpecialEventDetailsModal({ event, dateLabel, onClose, onOpenWorldJuniors, onOpenPlayoffs }) {
  if (!event) return null;

  const effects = event.effects && typeof event.effects === "object" ? Object.entries(event.effects) : [];
  const logoSrc = event.logoSrc || getSpecialEventLogoSrc(event);

  return (
    <div className="nhlcal-event-backdrop" onMouseDown={onClose} role="presentation">
      <div
        className={`nhlcal-event-modal tone-${normalizeKey(event.tone || "league")}`}
        onMouseDown={(mouseEvent) => mouseEvent.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="nhlcal-event-title"
      >
        <header className="nhlcal-event-modal-head">
          <div className={`nhlcal-event-modal-icon ${logoSrc ? "has-logo" : ""}`}>
            {logoSrc ? (
              <img src={logoSrc} alt={`${event.title || "Special event"} logo`} loading="lazy" />
            ) : (
              event.icon || "◆"
            )}
          </div>

          <div>
            <p>{event.priority || "LEAGUE"} EVENT · {dateLabel || formatLongDate(event.date)}</p>
            <h2 id="nhlcal-event-title">{sanitizeDisplayText(event.title) || "League Event"}</h2>
          </div>

          <button type="button" onClick={onClose} aria-label="Close event details">
            ×
          </button>
        </header>

        <div className="nhlcal-event-modal-body">
          {event.description ? (
            <p className="nhlcal-event-modal-description">{sanitizeDisplayText(event.description)}</p>
          ) : null}

          {event.effect_summary ? (
            <article className="nhlcal-event-modal-callout">
              <span>Effect</span>
              <strong>{sanitizeDisplayText(event.effect_summary)}</strong>
            </article>
          ) : null}

          {effects.length ? (
            <div className="nhlcal-event-effect-grid">
              {effects.map(([key, value]) => (
                <article key={key}>
                  <span>{key.replace(/_/g, " ")}</span>
                  <strong>
                    {Number(value) > 0 ? "+" : ""}
                    {String(value)}
                  </strong>
                </article>
              ))}
            </div>
          ) : null}

          {event.player_name ? (
            <article className="nhlcal-event-modal-callout">
              <span>Player</span>
              <strong>{event.player_name}</strong>
            </article>
          ) : null}

          {event.team_id ? (
            <article className="nhlcal-event-modal-callout">
              <span>Team</span>
              <strong>{event.team_id}</strong>
            </article>
          ) : null}

          {typeof onOpenWorldJuniors === "function" ? (
            <footer className="nhlcal-event-modal-actions">
              <button type="button" className="nhlcal-wjc-hub-button is-modal" onClick={onOpenWorldJuniors}>
                Open World Juniors Hub
              </button>
            </footer>
          ) : null}

          {typeof onOpenPlayoffs === "function" ? (
            <footer className="nhlcal-event-modal-actions">
              <button type="button" className="nhlcal-wjc-hub-button is-modal" onClick={onOpenPlayoffs}>
                Enter Playoffs
              </button>
            </footer>
          ) : null}
        </div>
      </div>
    </div>
  );
}
function InjuryReportFullModal({ injuries, userTeamId, activeTeam, onClose }) {
  const leagueCount = injuries.length;

  const userCount = injuries.filter((row) => {
    if (!row.teamId) return false;
    if (activeTeam && isSameTeamIdentifier(row.teamId, activeTeam)) return true;
    return String(row.teamId || "").toLowerCase() === String(userTeamId || "").toLowerCase();
  }).length;

  return (
    <div className="nhlcal-injury-backdrop" onMouseDown={onClose} role="presentation">
      <div
        className="nhlcal-injury-report-modal"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="nhlcal-injury-report-title"
      >
        <header className="nhlcal-injury-report-head">
          <div>
            <p className="nhlcal-injury-report-kicker">Medical</p>
            <h2 id="nhlcal-injury-report-title">Injury Report</h2>
            <p className="nhlcal-injury-report-stats">
              Active: <strong>{leagueCount}</strong> league-wide · Your club: <strong>{userCount}</strong>
            </p>
          </div>

          <button type="button" className="nhlcal-injury-report-close" onClick={onClose} aria-label="Close injury report">
            ×
          </button>
        </header>

        <div className="nhlcal-injury-report-body">
          {injuries.length ? (
            <table className="nhlcal-injury-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Pos</th>
                  <th>Status</th>
                  <th>Injury</th>
                  <th>Severity</th>
                  <th>GR</th>
                  <th>Return</th>
                </tr>
              </thead>
              <tbody>
                {injuries.map((row) => (
                  <tr key={row.id}>
                    <td>{row.playerName}</td>
                    <td>{row.teamAbbr || row.teamId || "—"}</td>
                    <td>{row.position}</td>
                    <td>{row.status}</td>
                    <td>{row.injuryLabel}</td>
                    <td>{row.severity}</td>
                    <td>{row.gamesRemaining}</td>
                    <td>{row.returnText || row.date || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="nhlcal-small-empty">No active injuries reported.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function CalendarScreen(props = {}) {
  const gameUI = useGameUI();

  const {
    franchiseState = gameUI?.franchiseState,
    state = null,
    gameState = null,
    data = null,
    onAdvanceDay = null,
    advanceDay = gameUI?.onAdvanceFranchise,
    setScreen = gameUI?.setScreen,
    navigate = null,
    onNavigate = null,
    selectedTeamId,
    teamId,
    gmTeamId,
  } = props;

  const rootState = useMemo(() => {
    return franchiseState || state || gameState || data || EMPTY_OBJECT;
  }, [franchiseState, state, gameState, data]);

  const controlledTeamId = useMemo(() => {
    return (
      selectedTeamId ||
      teamId ||
      gmTeamId ||
      rootState?.user_team_id ||
      rootState?.selected_team_id ||
      rootState?.controlled_team_id ||
      rootState?.gm_team_id ||
      rootState?.team_id ||
      rootState?.team?.id ||
      rootState?.team?.team_id ||
      rootState?.team?.abbr ||
      rootState?.team?.abbreviation ||
      null
    );
  }, [selectedTeamId, teamId, gmTeamId, rootState]);

  const normalized = useMemo(() => {
    return normalizeFranchiseState(rootState, controlledTeamId);
  }, [rootState, controlledTeamId]);

  const {
    currentDate,
    activeTeam,
    allTeams,
    games,
    standings,
    leagueEvents,
    notifications,
    players,
    prospects,
    injuries,
    draftClass,
    leagueState,
    statsCentral,
    finance,
    inbox,
  } = normalized;

  const [viewDate, setViewDate] = useState(() => {
    return toDateObject(currentDate) || new Date();
  });

  const [selectedDateISO, setSelectedDateISO] = useState(() => {
    return toISODate(currentDate || new Date());
  });

  const [activePanel, setActivePanel] = useState("game_preview");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [denseMode, setDenseMode] = useState(false);
  const [showOnlyTeamGames, setShowOnlyTeamGames] = useState(true);
  const [expandedGameKey, setExpandedGameKey] = useState("");
  const [hoveredDay, setHoveredDay] = useState(null);
  const [injuryReportOpen, setInjuryReportOpen] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [advanceBusy, setAdvanceBusy] = useState(false);
  const [choiceBusyId, setChoiceBusyId] = useState("");
  const [choiceError, setChoiceError] = useState("");
  const [worldJuniorsMenuOpen, setWorldJuniorsMenuOpen] = useState(false);
  const [wjcMenuSnapshot, setWjcMenuSnapshot] = useState(null);

  const calendarRootRef = useRef(null);
  const isAdvancing = advanceBusy || Boolean(gameUI?.advancing);

  const worldJuniorsEventData = useMemo(() => getWorldJuniorsEventPayload(rootState), [rootState]);

  const worldJuniorsWindowOpen = useMemo(() => {
    return (
      isWorldJuniorsWindowActive(rootState) ||
      isDateInWorldJuniorsWindow(toISODate(currentDate), rootState?.season_year || rootState?.seasonYear)
    );
  }, [rootState, currentDate]);

  useEffect(() => {
    const state = gameUI?.franchiseState || rootState;
    if (!isPlayoffReady(state)) return;
    if (findPendingPlayoffPopup(state) || state?.playoffs_generated) {
      gameUI?.openFranchiseEvent?.();
    }
  }, [gameUI?.franchiseState, rootState, gameUI]);

  useEffect(() => {
    const nextDate = toDateObject(currentDate);
    if (!nextDate) return;

    setViewDate((previous) => {
      if (previous.getFullYear() === nextDate.getFullYear() && previous.getMonth() === nextDate.getMonth()) {
        return previous;
      }

      return nextDate;
    });

    setSelectedDateISO(toISODate(nextDate));
  }, [currentDate]);

  useEffect(() => {
    setExpandedGameKey("");
  }, [selectedDateISO, denseMode, showOnlyTeamGames]);

  const selectedDate = useMemo(() => {
    return toDateObject(selectedDateISO) || toDateObject(currentDate) || new Date();
  }, [selectedDateISO, currentDate]);

  const monthGrid = useMemo(() => {
    return buildMonthGrid(viewDate);
  }, [viewDate]);

  const gamesByDate = useMemo(() => {
    const grouped = new Map();

    games.forEach((game) => {
      const iso = toISODate(game.date);
      if (!iso) return;

      if (!grouped.has(iso)) grouped.set(iso, []);
      grouped.get(iso).push(game);
    });

    grouped.forEach((list) => {
      list.sort((a, b) => {
        const aTeamGame = isTeamGame(a, activeTeam);
        const bTeamGame = isTeamGame(b, activeTeam);

        if (aTeamGame && !bTeamGame) return -1;
        if (!aTeamGame && bTeamGame) return 1;

        return String(a.time || "").localeCompare(String(b.time || ""));
      });
    });

    return grouped;
  }, [games, activeTeam]);

  const injuryUiRows = useMemo(() => {
    const raw = Array.isArray(injuries) ? injuries : EMPTY_ARRAY;
    return raw.map((inj, i) => normalizeInjuryRowForUi(inj, i));
  }, [injuries]);

  const calendarSpecialEvents = useMemo(() => {
    // Backend is now the source of truth.
    // Prefer calendar_events first so injuries/trades/storylines do not get duplicated.
    const anchorEvents = normalizeArrayMerged(
      rootState.season_anchor_events,
      rootState.seasonAnchorEvents
    );

    const canonicalEvents = normalizeArrayMerged(
      rootState.calendar_events,
      rootState.calendarEvents,
      anchorEvents
    );
  
    const fallbackEvents = canonicalEvents.length
      ? []
      : normalizeArrayMerged(
          rootState.special_events,
          rootState.specialEvents,
          rootState.league_events,
          rootState.leagueEvents,
          rootState.showcase_events,
          rootState.showcaseEvents,
          rootState.news_events,
          rootState.newsEvents
        );
  
    const normalizedRows = [...canonicalEvents, ...fallbackEvents]
      .map((event, index) => normalizeCalendarSpecialEvent(event, index))
      .filter((event) => toISODate(event.date));
  
    return dedupeByStableKey(normalizedRows, (event, index) => {
      const date = toISODate(event.date);
      const id = String(event.id || event.event_id || event.eventId || "").trim();
  
      if (id) return `${date}|${id}`;
  
      const team = String(event.team_id || event.teamId || "").trim();
      const player = String(event.player_id || event.playerId || event.player_name || "").trim();
      const type = String(event.type || event.kind || "league_event").trim();
      const headline = String(event.headline || event.title || "").trim();
  
      return `${date}|${type}|${team}|${player}|${headline}|${index}`;
    });
  }, [rootState]);

  const specialEventsByDate = useMemo(() => {
    const grouped = new Map();

    calendarSpecialEvents.forEach((event) => {
      const iso = toISODate(event.date);
      if (!iso) return;

      if (!grouped.has(iso)) grouped.set(iso, []);
      grouped.get(iso).push(event);
    });

    grouped.forEach((rows) => {
      rows.sort((a, b) => {
        const ap = PRIORITY_ORDER[String(a.priority || "MEDIUM").toUpperCase()] ?? 2;
        const bp = PRIORITY_ORDER[String(b.priority || "MEDIUM").toUpperCase()] ?? 2;

        if (ap !== bp) return ap - bp;

        const at = String(a.type || "");
        const bt = String(b.type || "");

        if (at !== bt) return at.localeCompare(bt);

        return String(a.title || "").localeCompare(String(b.title || ""));
      });
    });

    return grouped;
  }, [calendarSpecialEvents]);

  const userTeamGameByDate = useMemo(() => {
    const out = new Map();

    gamesByDate.forEach((rows, iso) => {
      const game = (rows || []).find((x) => isTeamGame(x, activeTeam));
      if (game) out.set(iso, game);
    });

    return out;
  }, [gamesByDate, activeTeam]);

  const selectedDayGamesRaw = useMemo(() => {
    return gamesByDate.get(selectedDateISO) || EMPTY_ARRAY;
  }, [gamesByDate, selectedDateISO]);

  const selectedDayEvents = useMemo(() => {
    return specialEventsByDate.get(selectedDateISO) || EMPTY_ARRAY;
  }, [specialEventsByDate, selectedDateISO]);

  const selectedDayGames = useMemo(() => {
    // The calendar tile can be Team Only, but the right-side selected-day panel
    // should still know the full league slate so it does not look empty/misleading.
    return selectedDayGamesRaw;
  }, [selectedDayGamesRaw]);

  const selectedTeamGame = useMemo(() => {
    const userGame = selectedDayGamesRaw.find((game) => isTeamGame(game, activeTeam));
    if (userGame) return userGame;
    return selectedDayGamesRaw[0] || null;
  }, [selectedDayGamesRaw, activeTeam]);

  const selectedPreviewIsUserGame = useMemo(() => {
    return Boolean(selectedTeamGame && isTeamGame(selectedTeamGame, activeTeam));
  }, [selectedTeamGame, activeTeam]);

  const selectedDayTeamGameCount = useMemo(() => {
    return selectedDayGamesRaw.filter((game) => isTeamGame(game, activeTeam)).length;
  }, [selectedDayGamesRaw, activeTeam]);

  const todayTeamGame = useMemo(() => {
    const todayISO = toISODate(currentDate || new Date());
    const todayGames = gamesByDate.get(todayISO) || EMPTY_ARRAY;

    return todayGames.find((game) => isTeamGame(game, activeTeam)) || todayGames[0] || null;
  }, [gamesByDate, currentDate, activeTeam]);

  const todaySpecialEvents = useMemo(() => {
    const todayISO = toISODate(currentDate || new Date());
    return specialEventsByDate.get(todayISO) || EMPTY_ARRAY;
  }, [specialEventsByDate, currentDate]);

  const monthTitle = useMemo(() => {
    return `${MONTH_NAMES[viewDate.getMonth()]} ${viewDate.getFullYear()}`;
  }, [viewDate]);

  const teamGames = useMemo(() => {
    return games
      .filter((game) => isTeamGame(game, activeTeam))
      .sort(sortGamesByDate);
  }, [games, activeTeam]);

  const previousTeamGame = useMemo(() => {
    const todayTime = startOfDay(toDateObject(currentDate) || new Date()).getTime();

    return (
      [...teamGames]
        .filter((game) => {
          const date = toDateObject(game.date);
          return date && startOfDay(date).getTime() < todayTime && isCompletedGame(game);
        })
        .sort(sortGamesByDate)
        .reverse()[0] || null
    );
  }, [teamGames, currentDate]);

  const nextTeamGames = useMemo(() => {
    const todayTime = startOfDay(toDateObject(currentDate) || new Date()).getTime();

    return teamGames
      .filter((game) => {
        const date = toDateObject(game.date);
        return date && startOfDay(date).getTime() >= todayTime && !isCompletedGame(game);
      })
      .sort(sortGamesByDate)
      .slice(0, 5);
  }, [teamGames, currentDate]);

  const leagueStandingsSnapshot = useMemo(() => {
    const divisionName = getDivisionName(activeTeam);
    if (divisionName && divisionName !== "League") {
      const div = buildDivisionStandings(standings, allTeams, activeTeam);
      if (div.rows.length >= 3) {
        return {
          ...div,
          label: `${divisionName} Division`,
          rows: div.rows.slice(0, 10),
        };
      }
    }
    const league = buildLeagueStandings(standings, allTeams);
    return { ...league, rows: league.rows.slice(0, 10) };
  }, [standings, allTeams, activeTeam]);

  const scheduleDiagnostics = useMemo(() => {
    return buildScheduleDiagnostics(teamGames, activeTeam, currentDate, standings);
  }, [teamGames, activeTeam, currentDate, standings]);

  const leagueStateRows = useMemo(() => {
    const rows = buildLeagueStateRows(games, currentDate, activeTeam);
    const visible = showOnlyTeamGames ? rows.filter((row) => row.involvesUserTeam) : rows;
    return visible.slice(0, 6);
  }, [games, currentDate, activeTeam, showOnlyTeamGames]);

  const gamePreview = useMemo(() => {
    return buildGamePreview(
      selectedTeamGame,
      activeTeam,
      allTeams,
      previousTeamGame,
      standings,
      statsCentral,
      {
        isUserTeamGame: selectedPreviewIsUserGame,
        games,
        currentDate,
      }
    );
  }, [
    selectedTeamGame,
    activeTeam,
    allTeams,
    previousTeamGame,
    standings,
    statsCentral,
    selectedPreviewIsUserGame,
    games,
    currentDate,
  ]);

  const quickTeamStats = useMemo(() => {
    return buildQuickTeamStats(activeTeam, standings, games, currentDate, statsCentral, allTeams);
  }, [activeTeam, standings, games, currentDate, statsCentral, allTeams]);

  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    if (!activeTeam) return;

    const row = findStandingForTeam(standings, activeTeam);
    if (standings.length && !row) {
      console.warn("[CalendarScreen] activeTeam could not be matched to standings row.", {
        team: getTeamDisplayName(activeTeam),
        teamId: getTeamId(activeTeam),
      });
    }

    const completed = completedTeamGamesUpTo(games, activeTeam, currentDate);
    if (completed.length > 0) {
      const gf = calculateGoalsFor(games, activeTeam, currentDate);
      const ga = calculateGoalsAgainst(games, activeTeam, currentDate);
      if (gf === 0 && ga === 0) {
        console.warn("[CalendarScreen] Completed team games exist but GF/GA ledger is zero.", {
          team: getTeamDisplayName(activeTeam),
          completedCount: completed.length,
        });
      }
    }
  }, [activeTeam, standings, games, currentDate]);

  const calendarInsights = useMemo(() => {
    return buildCalendarInsights(scheduleDiagnostics, activeTeam, standings, games, selectedDayEvents);
  }, [scheduleDiagnostics, activeTeam, standings, games, selectedDayEvents]);

  const inboxCount = useMemo(() => {
    return Array.isArray(inbox) ? inbox.filter((item) => !item.read && !item.is_read).length : 0;
  }, [inbox]);

  const activeMonthGames = useMemo(() => {
    const month = viewDate.getMonth();
    const year = viewDate.getFullYear();
    const scoped = showOnlyTeamGames ? games.filter((game) => isTeamGame(game, activeTeam)) : games;

    return scoped.filter((game) => {
      const date = toDateObject(game.date);
      return date && date.getMonth() === month && date.getFullYear() === year;
    });
  }, [games, viewDate, showOnlyTeamGames, activeTeam]);

  const activeMonthEvents = useMemo(() => {
    const month = viewDate.getMonth();
    const year = viewDate.getFullYear();

    return calendarSpecialEvents.filter((event) => {
      const date = toDateObject(event.date);
      return date && date.getMonth() === month && date.getFullYear() === year;
    });
  }, [calendarSpecialEvents, viewDate]);

  const selectedDateHeader = useMemo(() => {
    return formatLongDate(selectedDateISO);
  }, [selectedDateISO]);

  const activeTeamLabel = useMemo(() => {
    return getTeamDisplayName(activeTeam);
  }, [activeTeam]);

  const recentStorylines = useMemo(() => {
    const rows = Array.isArray(leagueEvents) ? leagueEvents : EMPTY_ARRAY;

    const out = rows
      .map((ev, idx) => {
        if (typeof ev === "string") {
          return {
            id: `evt-${idx}`,
            headline: sanitizeDisplayText(ev),
            date: "",
            priority: "MEDIUM",
            type: "storyline",
          };
        }

        return {
          id: ev?.id || ev?.storyline_id || `evt-${idx}`,
          headline: sanitizeDisplayText(
            ev?.headline || ev?.title || ev?.text || ev?.summary || "Storyline update"
          ),
          date: ev?.calendar_iso || ev?.date || "",
          priority: String(ev?.priority || "MEDIUM").toUpperCase(),
          type: String(ev?.type || ev?.event_type || "storyline").toLowerCase(),
          team_id: ev?.team_id || ev?.team,
          cause: sanitizeDisplayText(ev?.cause || ""),
          effects: ev?.effects || {},
          effect_summary: sanitizeDisplayText(ev?.effect_summary || ev?.effectSummary || ""),
        };
      })
      .filter((ev) => ev.headline)
      .sort((a, b) => {
        const pa = PRIORITY_ORDER[a.priority] ?? 2;
        const pb = PRIORITY_ORDER[b.priority] ?? 2;
        return pa - pb;
      });

    return out.slice(0, 6);
  }, [leagueEvents]);

  const storylineChoices = useMemo(() => {
    return rootState?.storyline_choices || rootState?.storylineChoices || EMPTY_ARRAY;
  }, [rootState?.storyline_choices, rootState?.storylineChoices]);

  const recentInjuryRows = useMemo(() => {
    return injuryUiRows
      .filter((row) => row.teamId && isSameTeamIdentifier(row.teamId, activeTeam))
      .slice(0, 6);
  }, [injuryUiRows, activeTeam]);

  const selectedDayInjuryRows = useMemo(() => {
    return injuryUiRows.filter((row) => {
      const rowIso = toISODate(row.date);
      if (!rowIso || rowIso !== selectedDateISO) return false;
      if (!row.teamId) return true;
      return isSameTeamIdentifier(row.teamId, activeTeam);
    });
  }, [injuryUiRows, selectedDateISO, activeTeam]);
  const handleNavigate = useCallback(
    (screenKey) => {
      setDrawerOpen(false);
      const resolved = NAV_ALIAS_TO_SCREEN[screenKey] || screenKey;

      if (typeof setScreen === "function") {
        setScreen(resolved);
        return;
      }

      if (typeof navigate === "function") {
        navigate(resolved);
        return;
      }

      if (typeof onNavigate === "function") {
        onNavigate(resolved);
      }
    },
    [setScreen, navigate, onNavigate]
  );

  const handleAdvanceCalendar = useCallback(
    async ({ mode = "day", count = 1 } = {}) => {
      if (isAdvancing) return;

      setAdvanceBusy(true);

      const payload = {
        mode,
        count: Math.max(1, Number(count) || 1),
        auto_resolve: true,
      };

      try {
        const result = await advanceFranchise(payload);

        if (typeof gameUI?.setFranchiseState === "function" && result?.state) {
          gameUI.setFranchiseState(result.state);
        } else if (typeof gameUI?.refreshFranchise === "function") {
          await gameUI.refreshFranchise();
        }

        const step = result?.step || {};
        const lastStep = step?.bulk ? step.last_step || step : step;
        const status = String(lastStep?.status || step?.status || "").toLowerCase();
        const nextState = result?.state || gameUI?.franchiseState || rootState;

        if (status === "blocked") {
          setActivePanel("events");
          return;
        }

        const bulkCompleted = Number(step?.steps_completed || 0);
        const bulkStopped = String(step?.stopped_reason || "").toLowerCase();

        if (step?.bulk && bulkCompleted <= 0) {
          if (bulkStopped === "pending_decisions") {
            setActivePanel("events");
          }
          return;
        }

        if (
          status &&
          !["ok", "complete", "postseason", "post_cup", "offseason", "preseason", "playoff_ready"].includes(status)
        ) {
          return;
        }

        const landedIso = toISODate(nextState?.nhl_today?.iso || nextState?.nhlToday?.iso || "");

        if (status === "post_cup" || status === "offseason") {
          gameUI?.openFranchiseEvent?.();
        }

        const playoffTriggered =
          status === "playoff_ready" ||
          isPlayoffReady(nextState) ||
          Boolean(findPendingPlayoffPopup(nextState));

        if (playoffTriggered) {
          const playoffPop = findPendingPlayoffPopup(nextState);
          if (playoffPop?.id && typeof gameUI?.onDismissShowcasePopups === "function") {
            try {
              await gameUI.onDismissShowcasePopups([playoffPop.id]);
            } catch {
              /* menu still opens */
            }
          }
          gameUI?.openFranchiseEvent?.();
        }

        const wjcTriggered =
          Boolean(findPendingWjcPopup(nextState)) ||
          isWorldJuniorsWindowActive(nextState) ||
          isDateInWorldJuniorsWindow(landedIso, nextState?.season_year || nextState?.seasonYear);

        if (wjcTriggered) {
          const wjcPop = findPendingWjcPopup(nextState);
          const resolved = resolveWorldJuniorsPayload(nextState, wjcPop);
          if (resolved.hasData && resolved.raw) {
            setWjcMenuSnapshot({ ...resolved.raw });
          }
          setWorldJuniorsMenuOpen(true);
        }
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[CalendarScreen] advance failed:", formatFranchiseApiError(error) || error);
        }
      } finally {
        setAdvanceBusy(false);
      }
    },
    [isAdvancing, gameUI, rootState]
  );

  const handleAdvanceDay = useCallback(() => {
    return handleAdvanceCalendar({ mode: "day", count: 1 });
  }, [handleAdvanceCalendar]);

  const handleAdvance15Days = useCallback(() => {
    return handleAdvanceCalendar({ mode: "days", count: 15 });
  }, [handleAdvanceCalendar]);

  const handleStorylineChoice = useCallback(
    async (storylineId, choiceId) => {
      const sid = String(storylineId || "");
      const cid = String(choiceId || "");

      if (!sid || !cid) return;

      const busyKey = `${sid}:${cid}`;

      setChoiceBusyId(busyKey);
      setChoiceError("");

      try {
        if (typeof gameUI?.onResolveStorylineChoice === "function") {
          await gameUI.onResolveStorylineChoice(sid, cid);
          return;
        }

        if (typeof rootState?.onResolveStorylineChoice === "function") {
          await rootState.onResolveStorylineChoice(sid, cid);
          return;
        }

        throw new Error("Storyline choice handler is not connected.");
      } catch (error) {
        setChoiceError(
          String(
            error?.response?.data?.detail ||
              error?.response?.data?.message ||
              error?.message ||
              "Could not resolve storyline choice."
          )
        );
      } finally {
        setChoiceBusyId("");
      }
    },
    [gameUI, rootState]
  );

  const openFranchiseEventMenu = useCallback(async () => {
    setDrawerOpen(false);
    setSettingsOpen(false);
    setInjuryReportOpen(false);
    setSelectedEvent(null);

    const playoffPop = findPendingPlayoffPopup(gameUI?.franchiseState || rootState);
    if (playoffPop?.id && typeof gameUI?.onDismissShowcasePopups === "function") {
      try {
        await gameUI.onDismissShowcasePopups([playoffPop.id]);
      } catch {
        /* keep menu open even if dismiss fails */
      }
    }

    gameUI?.openFranchiseEvent?.();
  }, [gameUI, rootState]);

  const handleEnterPlayoffs = useCallback(async () => {
    if (typeof gameUI?.onEnterPlayoffs !== "function") {
      throw new Error("Playoff entry handler is not connected.");
    }
    await gameUI.onEnterPlayoffs();
    gameUI?.openFranchiseEvent?.();
  }, [gameUI]);

  const closeWorldJuniorsMenu = useCallback(async () => {
    setWorldJuniorsMenuOpen(false);
    const state = gameUI?.franchiseState || rootState;
    const wjcPop = findPendingWjcPopup(state);
    if (wjcPop?.id && typeof gameUI?.onDismissShowcasePopups === "function") {
      try {
        await gameUI.onDismissShowcasePopups([wjcPop.id]);
      } catch {
        /* menu already closed */
      }
    }
    setWjcMenuSnapshot(null);
  }, [gameUI, rootState]);

  const openWorldJuniorsMenu = useCallback(() => {
    setDrawerOpen(false);
    setSettingsOpen(false);
    setInjuryReportOpen(false);
    setSelectedEvent(null);

    const state = gameUI?.franchiseState || rootState;
    const wjcPop = findPendingWjcPopup(state);
    const resolved = resolveWorldJuniorsPayload(state, wjcPop);
    if (resolved.hasData && resolved.raw) {
      setWjcMenuSnapshot({ ...resolved.raw });
    } else {
      setWjcMenuSnapshot(null);
    }

    setWorldJuniorsMenuOpen(true);
  }, [gameUI, rootState]);

  const closeAllOverlays = useCallback(() => {
    setDrawerOpen(false);
    setSettingsOpen(false);
    setInjuryReportOpen(false);
    setSelectedEvent(null);
    setWorldJuniorsMenuOpen(false);
    setWjcMenuSnapshot(null);
  }, []);

  const openDrawer = useCallback(() => {
    setSettingsOpen(false);
    setInjuryReportOpen(false);
    setSelectedEvent(null);
    setDrawerOpen(true);
  }, []);
  
  const closeDrawer = useCallback(() => {
    setDrawerOpen(false);
  }, []);
  
  const openSettings = useCallback(() => {
    setDrawerOpen(false);
    setInjuryReportOpen(false);
    setSelectedEvent(null);
    setSettingsOpen((value) => !value);
  }, []);
  
  const openInjuryReport = useCallback(() => {
    setDrawerOpen(false);
    setSettingsOpen(false);
    setSelectedEvent(null);
    setInjuryReportOpen(true);
  }, []);
  
  const closeInjuryReport = useCallback(() => {
    setInjuryReportOpen(false);
  }, []);
  
  const openSpecialEvent = useCallback(
    (event) => {
      if (isWorldJuniorsCalendarEvent(event)) {
        openWorldJuniorsMenu();
        return;
      }

      if (isPlayoffsStartCalendarEvent(event) || isPlayoffReady(gameUI?.franchiseState || rootState)) {
        openFranchiseEventMenu();
        return;
      }

      setDrawerOpen(false);
      setSettingsOpen(false);
      setInjuryReportOpen(false);
      setSelectedEvent(event);
    },
    [openWorldJuniorsMenu, openFranchiseEventMenu, gameUI, rootState]
  );
  
  const closeSpecialEvent = useCallback(() => {
    setSelectedEvent(null);
  }, []);

  const goPreviousMonth = useCallback(() => {
    setViewDate((previous) => new Date(previous.getFullYear(), previous.getMonth() - 1, 1));
  }, []);

  const goNextMonth = useCallback(() => {
    setViewDate((previous) => new Date(previous.getFullYear(), previous.getMonth() + 1, 1));
  }, []);

  const goCurrentMonth = useCallback(() => {
    const date = toDateObject(currentDate) || new Date();
    setViewDate(new Date(date.getFullYear(), date.getMonth(), 1));
    setSelectedDateISO(toISODate(date));
  }, [currentDate]);

  const setCalendarModeTeamOnly = useCallback(() => {
    setShowOnlyTeamGames(true);
  }, []);

  const setCalendarModeLeague = useCallback(() => {
    setShowOnlyTeamGames(false);
  }, []);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;

      const target = event.target;
      const tag = String(target?.tagName || "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target?.isContentEditable) {
        return;
      }

      event.preventDefault();
      if (event.key === "ArrowLeft") goPreviousMonth();
      else goNextMonth();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goPreviousMonth, goNextMonth]);

  return (
    <div className="nhlcal-root" ref={calendarRootRef}>
      <CalendarStyles />

      <aside className="nhlcal-sidebar">
        <button
          className="nhlcal-brand-button"
          type="button"
          onClick={() => handleNavigate(SCREEN_KEYS.office)}
          title="Office"
        >
          <span className="nhlcal-shield-icon">⌂</span>
        </button>

        <nav className="nhlcal-side-nav" aria-label="Main navigation">
          <SideNavButton
            active={false}
            icon="▦"
            label="Office"
            onClick={() => handleNavigate(SCREEN_KEYS.office)}
          />

          <SideNavButton
            active
            icon="◫"
            label="Calendar"
            onClick={() => handleNavigate(SCREEN_KEYS.calendar)}
          />

          <SideNavButton
            active={false}
            icon="📰"
            label="Storylines"
            onClick={() => handleNavigate(SCREEN_KEYS.storylines_report)}
          />

          <SideNavButton
            active={false}
            icon="🩺"
            label="Injury Report"
            onClick={openInjuryReport}
          />

          <SideNavButton
            active={false}
            icon="✉"
            label="Inbox"
            badge={inboxCount > 99 ? "99+" : inboxCount > 0 ? inboxCount : null}
            onClick={() => handleNavigate(SCREEN_KEYS.inbox)}
            title={inboxCount ? `${inboxCount} unread messages` : "Inbox"}
          />
        </nav>

        <button
          className="nhlcal-settings-button"
          type="button"
          onClick={openSettings}
          title="Settings"
        >
          <span>⚙</span>
          <small>Settings</small>
        </button>
      </aside>

      <main className="nhlcal-main">
        <header className="nhlcal-topbar">
          <section className="nhlcal-team-identity">
            <TeamIdentityBadge team={activeTeam} size="large" />
            <div>
              <p className="nhlcal-team-city">{getTeamCity(activeTeam)}</p>
              <h1>{activeTeamLabel}</h1>
            </div>
          </section>

          <section className="nhlcal-month-control" aria-label="Calendar month controls">
            <div className="nhlcal-month-row">
              <button type="button" onClick={goPreviousMonth} aria-label="Previous month">
                ‹
              </button>
              <h2>{monthTitle}</h2>
              <button type="button" onClick={goNextMonth} aria-label="Next month">
                ›
              </button>
              <button type="button" className="nhlcal-today-chip" onClick={goCurrentMonth}>
                Today
              </button>
            </div>
          </section>

          <section className="nhlcal-action-cluster">
            <div className="nhlcal-action-primary">
              {worldJuniorsWindowOpen ? (
                <button
                  className="nhlcal-wjc-hub-button"
                  type="button"
                  onClick={openWorldJuniorsMenu}
                  title="Open U-20 World Juniors tournament hub"
                >
                  <span>🌍</span>
                  World Juniors
                </button>
              ) : null}

              <button
                className={`nhlcal-advance-button ${isAdvancing ? "is-busy" : ""}`}
                type="button"
                onClick={handleAdvanceDay}
                disabled={isAdvancing}
              >
                <span>{isAdvancing ? "…" : "▶"}</span>
                {isAdvancing ? "Advancing" : "Advance Day"}
              </button>

              <button
                className={`nhlcal-advance-button nhlcal-advance-button-secondary ${isAdvancing ? "is-busy" : ""}`}
                type="button"
                onClick={handleAdvance15Days}
                disabled={isAdvancing}
                title="Sim 15 calendar days"
              >
                +15 Days
              </button>
            </div>

            <div className="nhlcal-action-secondary">
              <button
                className="nhlcal-menu-toggle"
                type="button"
                onClick={openDrawer}
                aria-label="Open command drawer"
                title="Open command drawer"
              >
                <span />
                <span />
                <span />
              </button>

              <button
                className="nhlcal-quick-link"
                type="button"
                onClick={() => handleNavigate(SCREEN_KEYS.storylines_report)}
              >
                Storylines
              </button>

              <button className="nhlcal-quick-link" type="button" onClick={openInjuryReport}>
                Injuries
              </button>

              <div className="nhlcal-date-chip compact">
                <span className="nhlcal-date-icon">◫</span>
                <div>
                  <strong>{formatShortDate(currentDate)}</strong>
                  <span>{formatWeekday(currentDate)}</span>
                </div>
              </div>
            </div>
          </section>
        </header>

        {!games.length ? (
          <div className="nhlcal-schedule-alert">
            Schedule data is missing or not loaded from the backend yet.
          </div>
        ) : null}

        <section className="nhlcal-stat-strip">
          {quickTeamStats.map((stat) => (
            <StatPill
              key={stat.key}
              icon={stat.icon}
              label={stat.label}
              value={stat.value}
              sub={stat.sub}
              tone={stat.tone}
              tooltip={stat.tooltip}
              loading={isAdvancing}
            />
          ))}
        </section>

        <section className="nhlcal-content-grid">
          <section className="nhlcal-calendar-panel">
            <div className="nhlcal-calendar-toolbar">
              <div className="nhlcal-calendar-toolbar-group">
                <button type="button" onClick={goCurrentMonth}>
                  Today
                </button>
                <button type="button" onClick={() => setDenseMode((value) => !value)}>
                  {denseMode ? "Comfort" : "Dense"}
                </button>
                <button
                  type="button"
                  className={showOnlyTeamGames ? "is-active" : ""}
                  onClick={setCalendarModeTeamOnly}
                  title="Show only your team's games on the calendar"
                >
                  Team Only
                </button>
                <button
                  type="button"
                  className={!showOnlyTeamGames ? "is-active" : ""}
                  onClick={setCalendarModeLeague}
                  title="Show the full league slate plus calendar events"
                >
                  League + Events
                </button>
              </div>
            </div>
            <div className="nhlcal-week-header">
              {WEEKDAY_NAMES.map((day) => (
                <div key={day}>{day}</div>
              ))}
            </div>

            <div className={`nhlcal-month-grid nhlcal-scroll-surface ${denseMode ? "is-dense" : ""}`}>
              {monthGrid.map((dayCell) => {
                const iso = toISODate(dayCell.date);
                const dayGamesRaw = gamesByDate.get(iso) || EMPTY_ARRAY;
                const daySpecialEvents = specialEventsByDate.get(iso) || EMPTY_ARRAY;
                const sortedDayEvents = [...daySpecialEvents].sort((a, b) => {
                  const ap = PRIORITY_ORDER[String(a.priority || "MEDIUM").toUpperCase()] ?? 2;
                  const bp = PRIORITY_ORDER[String(b.priority || "MEDIUM").toUpperCase()] ?? 2;
                  return ap - bp;
                });
                
                const criticalDayEvents = sortedDayEvents.filter(
                  (event) => String(event.priority || "").toUpperCase() === "CRITICAL"
                );
                
                const regularDayEvents = sortedDayEvents.filter(
                  (event) => String(event.priority || "").toUpperCase() !== "CRITICAL"
                );
                
                const visibleDayEvents = denseMode
                  ? [...criticalDayEvents, ...regularDayEvents.slice(0, Math.max(1, 2 - criticalDayEvents.length))]
                  : sortedDayEvents.slice(0, 4);

                const visibleGames = showOnlyTeamGames
                  ? userTeamGameByDate.get(iso)
                    ? [userTeamGameByDate.get(iso)]
                    : EMPTY_ARRAY
                  : dayGamesRaw;

                const teamGame = dayGamesRaw.find((game) => isTeamGame(game, activeTeam));
                const isSelected = iso === selectedDateISO;
                const isToday = iso === toISODate(currentDate || new Date());
                const isOtherMonth = dayCell.outsideMonth;
                const hasGames = visibleGames.length > 0;
                const hasRawGames = dayGamesRaw.length > 0;
                const hasSpecialEvents = daySpecialEvents.length > 0;
                const hasCriticalEvent = daySpecialEvents.some(
                  (event) => String(event.priority || "").toUpperCase() === "CRITICAL"
                );
                const hasHighEvent = daySpecialEvents.some(
                  (event) => String(event.priority || "").toUpperCase() === "HIGH"
                );

                const maxEvents = denseMode ? 1 : 2;
                const maxGames = denseMode ? 2 : hasSpecialEvents ? 1 : 2;

                return (
                  <button
                    key={iso}
                    type="button"
                    aria-label={`${formatLongDate(iso)}${teamGame ? ", your team plays" : ""}${
                      hasSpecialEvents ? `, ${daySpecialEvents.length} events` : ""
                    }${isToday ? ", today" : ""}`}
                    className={[
                      "nhlcal-day-cell",
                      isSelected ? "is-selected" : "",
                      isToday ? "is-today" : "",
                      isOtherMonth ? "is-muted" : "",
                      hasGames ? "has-games" : "",
                      hasRawGames ? "has-raw-games" : "",
                      teamGame ? "has-team-game" : "",
                      hasSpecialEvents ? "has-special-events" : "",
                      hasCriticalEvent ? "has-critical-event" : "",
                      hasHighEvent ? "has-high-event" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => setSelectedDateISO(iso)}
                    onMouseEnter={() => setHoveredDay(iso)}
                    onMouseLeave={() => setHoveredDay(null)}
                  >
                    <div className="nhlcal-day-number-row">
                      <span className="nhlcal-day-number">{dayCell.date.getDate()}</span>

                      <span className="nhlcal-day-marker-row">
                      {hasSpecialEvents ? (
                        <span className="nhlcal-event-corner-badge" title={`${daySpecialEvents.length} special event(s)`}>
                          {daySpecialEvents[0]?.logoSrc || getSpecialEventLogoSrc(daySpecialEvents[0]) ? (
                            <img
                              src={daySpecialEvents[0]?.logoSrc || getSpecialEventLogoSrc(daySpecialEvents[0])}
                              alt={`${daySpecialEvents[0]?.title || "Special event"} logo`}
                              loading="lazy"
                            />
                          ) : (
                            daySpecialEvents[0]?.icon || "◆"
                          )}
                        </span>
                      ) : null}

                        {teamGame ? (
                          <span className="nhlcal-corner-cut" title="Your team plays this day" aria-label="Your team plays" />
                        ) : null}
                      </span>
                    </div>

                    <div className="nhlcal-day-content">
                      {daySpecialEvents.length ? (
                        <div className="nhlcal-day-special-events">
                          {visibleDayEvents.map((event) => (
                            <CalendarSpecialEventTile
                              key={event.id}
                              event={event}
                              compact={denseMode}
                              onOpen={openSpecialEvent}
                            />
                          ))}

                          {daySpecialEvents.length > maxEvents ? (
                            <div
                              className="nhlcal-more-events"
                              role="button"
                              tabIndex={0}
                              onClick={(event) => {
                                event.stopPropagation();
                                setSelectedDateISO(iso);
                                setActivePanel("events");
                              }}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" || event.key === " ") {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  setSelectedDateISO(iso);
                                  setActivePanel("events");
                                }
                              }}
                            >
                              +{daySpecialEvents.length - maxEvents} event
                              {daySpecialEvents.length - maxEvents === 1 ? "" : "s"}
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      <div className="nhlcal-day-games">
                        {visibleGames.slice(0, maxGames).map((game, index) => {
                          const gameKey = getGameStableKey(game, index);

                          return (
                            <CalendarGameTile
                              key={gameKey}
                              game={game}
                              activeTeam={activeTeam}
                              allTeams={allTeams}
                              standings={standings}
                              compact={denseMode}
                              expanded={isSelected && expandedGameKey === gameKey}
                              onToggle={() => {
                                setSelectedDateISO(iso);
                                setExpandedGameKey((prev) => (prev === gameKey ? "" : gameKey));
                              }}
                            />
                          );
                        })}

                        {visibleGames.length > maxGames ? (
                          <div className="nhlcal-more-games">
                            +{visibleGames.length - maxGames} game{visibleGames.length - maxGames === 1 ? "" : "s"}
                          </div>
                        ) : null}

                        {!visibleGames.length && !daySpecialEvents.length ? (
                          <div
                            className="nhlcal-empty-day-line"
                            title={
                              showOnlyTeamGames && hasRawGames
                                ? `${dayGamesRaw.length} league game(s) hidden by Team Only filter`
                                : undefined
                            }
                          >
                            {showOnlyTeamGames && hasRawGames ? "League slate" : "Off day"}
                          </div>
                        ) : null}
                      </div>
                    </div>

                    {hoveredDay === iso && (dayGamesRaw.length > 0 || daySpecialEvents.length > 0) ? (
                      <div className="nhlcal-day-hover-card">
                        <strong>{formatLongDate(iso)}</strong>
                        {dayGamesRaw.length ? (
                          <span>
                            {dayGamesRaw.length} scheduled game{dayGamesRaw.length === 1 ? "" : "s"}
                          </span>
                        ) : null}
                        {daySpecialEvents.length ? (
                          <span>
                            {daySpecialEvents.length} special event{daySpecialEvents.length === 1 ? "" : "s"} ·{" "}
                            {daySpecialEvents
                              .slice(0, 3)
                              .map((event) => event.title)
                              .join(" / ")}
                          </span>
                        ) : null}
                      </div>
                    ) : null}
                  </button>
                );
              })}
            </div>

            <footer className="nhlcal-calendar-footer">
              <div className="nhlcal-legend">
                <span>
                  <i className="dot home" />
                  Home
                </span>
                <span>
                  <i className="dot away" />
                  Away
                </span>
                <span>
                  <i className="dot team-game" />
                  Team Game
                </span>
                <span>
                  <i className="dot win" />
                  Win
                </span>
                <span>
                  <i className="dot loss" />
                  Loss
                </span>
                <span>
                  <i className="dot otl" />
                  OTL
                </span>
                <span>
                  <i className="dot special" />
                  Special Event
                </span>
                <span>
                  <i className="dot critical" />
                  Critical Date
                </span>
              </div>

              <div className="nhlcal-calendar-actions">
                <button type="button" onClick={() => setSettingsOpen(true)}>
                  Calendar Settings
                </button>
              </div>
            </footer>
          </section>

          <aside className="nhlcal-right-rail">
            <div className="nhlcal-rail-preview-wrap">
              <GamePreviewCard
                activePanel={activePanel}
                setActivePanel={setActivePanel}
                preview={gamePreview}
                selectedDateHeader={selectedDateHeader}
                activeTeam={activeTeam}
                allTeams={allTeams}
                standings={standings}
                selectedDayGames={selectedDayGames}
                selectedDayGamesRaw={selectedDayGamesRaw}
                selectedDayEvents={selectedDayEvents}
                selectedDayInjuryRows={selectedDayInjuryRows}
                selectedDayTeamGameCount={selectedDayTeamGameCount}
                todayTeamGame={todayTeamGame}
                todaySpecialEvents={todaySpecialEvents}
                onNavigate={handleNavigate}
                onOpenEvent={openSpecialEvent}
                onOpenInjuryReport={openInjuryReport}
              />
            </div>

            <StandingsCard
              activeTeam={activeTeam}
              allTeams={allTeams}
              rows={leagueStandingsSnapshot.rows}
              groupLabel={leagueStandingsSnapshot.label}
              extended
              onOpenFull={() => handleNavigate(SCREEN_KEYS.standings)}
            />

            {nextTeamGames.length ? (
              <UpcomingStretchCard
                games={nextTeamGames}
                activeTeam={activeTeam}
                allTeams={allTeams}
              />
            ) : null}
          </aside>
        </section>

        <section className="nhlcal-bottom-grid nhlcal-bottom-grid--collapsed">
          <DiagnosticsPanel
            diagnostics={scheduleDiagnostics}
            insights={calendarInsights}
          />

          <MonthSnapshotPanel
            activeMonthGames={activeMonthGames}
            activeMonthEvents={activeMonthEvents}
            activeTeam={activeTeam}
            allTeams={allTeams}
            selectedDateISO={selectedDateISO}
          />
        </section>
      </main>

      {drawerOpen ? (
        <FranchiseDrawer
          activeTeam={activeTeam}
          rootState={rootState}
          onClose={closeDrawer}
          onNavigate={handleNavigate}
          players={players}
          prospects={prospects}
          injuries={injuries}
          draftClass={draftClass}
          standings={standings}
          notifications={notifications}
          leagueEvents={leagueEvents}
          finance={finance}
          calendarSpecialEvents={calendarSpecialEvents}
          nextTeamGames={nextTeamGames}
          leagueStateRows={leagueStateRows}
        />
      ) : null}

      {settingsOpen ? (
        <CalendarSettingsModal
          onClose={() => setSettingsOpen(false)}
          denseMode={denseMode}
          setDenseMode={setDenseMode}
          showOnlyTeamGames={showOnlyTeamGames}
          setShowOnlyTeamGames={setShowOnlyTeamGames}
          currentDate={currentDate}
          activeTeam={activeTeam}
          games={games}
          events={calendarSpecialEvents}
        />
      ) : null}

      {injuryReportOpen ? (
        <InjuryReportFullModal
          injuries={injuryUiRows}
          userTeamId={controlledTeamId || rootState?.user_team_id || ""}
          activeTeam={activeTeam}
          onClose={closeInjuryReport}
        />
      ) : null}

      {selectedEvent ? (
        <SpecialEventDetailsModal
          event={selectedEvent}
          dateLabel={formatLongDate(selectedEvent.date)}
          onClose={closeSpecialEvent}
          onOpenWorldJuniors={
            isWorldJuniorsCalendarEvent(selectedEvent) ? openWorldJuniorsMenu : undefined
          }
          onOpenPlayoffs={
            isPlayoffsStartCalendarEvent(selectedEvent) || isPlayoffReady(gameUI?.franchiseState || rootState)
              ? openFranchiseEventMenu
              : undefined
          }
        />
      ) : null}

      {worldJuniorsMenuOpen ? (
        <div className="nhlcal-wjc-menu-host" role="presentation">
          <WorldJuniorsEvent
            franchiseState={gameUI?.franchiseState || rootState}
            eventData={wjcMenuSnapshot || worldJuniorsEventData}
            onClose={closeWorldJuniorsMenu}
            onBackToHub={closeWorldJuniorsMenu}
            onSimNextTournamentDay={handleAdvanceDay}
            onOpenDraftBoard={(prospectId) => {
              closeWorldJuniorsMenu();
              if (typeof gameUI?.openDraftClassFromWjc === "function") {
                gameUI.openDraftClassFromWjc(prospectId);
              } else {
                handleNavigate(SCREEN_KEYS.draft);
              }
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

function SideNavButton({ active, icon, label, badge, onClick, title }) {
  return (
    <button
      type="button"
      className={`nhlcal-side-button ${active ? "is-active" : ""}`}
      onClick={onClick}
      title={title || label}
      aria-label={title || label}
    >
      <span className="nhlcal-side-icon">{icon}</span>
      <span className="nhlcal-side-label">{label}</span>
      {badge ? <em aria-label={`${badge} unread`}>{badge}</em> : null}
    </button>
  );
}

function StatPill({ icon, label, value, sub, tone, tooltip, loading = false }) {
  if (loading) {
    return (
      <article className="nhlcal-stat-pill is-skeleton">
        <div className="nhlcal-stat-icon" />
        <div>
          <span>{label}</span>
          <strong className="nhlcal-skeleton-bar" />
          <small className="nhlcal-skeleton-bar short" />
        </div>
      </article>
    );
  }

  return (
    <article
      className={`nhlcal-stat-pill ${tone ? `tone-${tone}` : ""}`}
      title={tooltip || undefined}
    >
      <div className="nhlcal-stat-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong className={value === "—" || value == null || value === "" ? "is-missing" : undefined}>
          {value == null || value === "" ? "—" : value}
        </strong>
        {sub ? <small>{sub}</small> : null}
      </div>
    </article>
  );
}

function CalendarGameTile({ game, activeTeam, allTeams, standings = [], compact, expanded, onToggle }) {
  const opponent = getOpponentFromGame(game, activeTeam, allTeams);
  const isHome = isHomeGame(game, activeTeam);
  const isUserGame = isTeamGame(game, activeTeam);
  const completed = isCompletedGame(game);
  const result = getGameResultLabel(game, activeTeam);
  const opponentName = getTeamAbbreviation(opponent);
  const relation = isHome ? "vs" : "@";
  const awayTeam = getAwayTeam(game, allTeams);
  const homeTeam = getHomeTeam(game, allTeams);

  const awayScore = safeScore(
    firstDefined(game.awayScore, game.away_score, game.away_goals, game.awayGoals, game.score?.away)
  );

  const homeScore = safeScore(
    firstDefined(game.homeScore, game.home_score, game.home_goals, game.homeGoals, game.score?.home)
  );

  const hasVisibleScore = awayScore !== null && homeScore !== null;
  const detailsTime = normalizeGameTime(game.time || game.start_time || game.startTime);
  const venue = game.venue || game.arena || getArenaName(homeTeam);
  const finalLabel = completed ? "FINAL" : detailsTime;
  const statusPillText = completed ? finalLabel : detailsTime || "TBD";
  const resultToken = String(result || "").trim().toUpperCase();

  const resultClass =
    completed && isUserGame
      ? resultToken.startsWith("OTL") || resultToken === "OT" || resultToken === "SO"
        ? "result-otl"
        : resultToken.startsWith("W")
          ? "result-win"
          : resultToken.startsWith("L")
            ? "result-loss"
            : ""
      : "";

  const opponentFullName = getTeamDisplayName(opponent);
  const awayFullName = getTeamDisplayName(awayTeam);
  const homeFullName = getTeamDisplayName(homeTeam);
  const scoreTooltip =
    hasVisibleScore && completed ? ` ${awayScore}-${homeScore}` : completed && result ? ` — ${result}` : "";
  const tileTooltip = isUserGame
    ? `${relation} ${opponentFullName}${completed && result ? ` — ${result}` : detailsTime ? ` · ${detailsTime}` : ""}`
    : `${awayFullName} @ ${homeFullName}${scoreTooltip || (detailsTime ? ` · ${detailsTime}` : "")}`;
  const tileAriaLabel = tileTooltip;

  return (
    <div
      className={`nhlcal-game-tile ${completed ? "is-final" : "is-upcoming"} ${isHome ? "is-home" : "is-away"} ${resultClass} ${
        expanded ? "is-expanded" : ""
      }`}
      title={tileTooltip}
      aria-label={tileAriaLabel}
      onClick={(event) => {
        event.stopPropagation();
        onToggle?.();
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.stopPropagation();
          onToggle?.();
        }
      }}
    >
      <div className="nhlcal-game-tile-accent" />

      <div className="nhlcal-game-tile-logo">
        <TeamIdentityBadge team={opponent} size={compact ? "tile-compact" : "tile-main"} />
      </div>

      <div className="nhlcal-game-tile-main">
        <div className="nhlcal-game-match-line">
          <span className={`nhlcal-game-relation ${isHome ? "home" : "away"}`}>
            {isUserGame ? relation : "NHL"}
          </span>

          <strong>
            {isUserGame ? (
              opponentName
            ) : (
              <>
                {getTeamAbbreviation(getAwayTeam(game, allTeams))} @ {getTeamAbbreviation(getHomeTeam(game, allTeams))}
              </>
            )}
          </strong>
        </div>

        <div className="nhlcal-game-meta-line">
          <span>{completed ? result : detailsTime || "TBD"}</span>
          {venue ? <em>{venue}</em> : null}
        </div>
      </div>

      <div className="nhlcal-game-tile-side">
        {hasVisibleScore ? (
          <div className="nhlcal-game-score-mini">
            <span>{awayScore}</span>
            <em>-</em>
            <span>{homeScore}</span>
          </div>
        ) : (
          <div className="nhlcal-game-status-pill">{statusPillText}</div>
        )}

        <span className="nhlcal-game-chevron">{expanded ? "⌃" : "⌄"}</span>
      </div>

      {expanded ? (
        <div className="nhlcal-game-expand-details">
          <div className="nhlcal-game-expand-header">
            <strong>{finalLabel}</strong>
            <span>{venue || getArenaName(homeTeam) || "TBD"}</span>
          </div>

          <div className="nhlcal-game-expand-row">
            <div>
              <TeamIdentityBadge team={awayTeam} size="mini" />
              <span>{getTeamAbbreviation(awayTeam)}</span>
              <small>{formatTeamRecord(awayTeam, standings)}</small>
            </div>

            <div>
              <TeamIdentityBadge team={homeTeam} size="mini" />
              <span>{getTeamAbbreviation(homeTeam)}</span>
              <small>{formatTeamRecord(homeTeam, standings)}</small>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function getGameStableKey(game, index = 0) {
  const gid = String(game?.id || game?.game_id || game?.gameId || "").trim();
  if (gid) return gid;

  const date = toISODate(game?.date || game?.game_date || game?.gameDate || "");
  const home = String(game?.homeId || game?.home_team_id || game?.home_id || game?.home || "");
  const away = String(game?.awayId || game?.away_team_id || game?.away_id || game?.away || "");
  const time = String(game?.time || game?.start_time || game?.startTime || "");

  return `${date}|${home}|${away}|${time}|${index}`;
}
function GamePreviewCard({
  activePanel,
  setActivePanel,
  preview,
  selectedDateHeader,
  activeTeam,
  allTeams,
  standings = [],
  selectedDayGames,
  selectedDayGamesRaw,
  selectedDayEvents,
  selectedDayInjuryRows,
  selectedDayTeamGameCount = 0,
  todayTeamGame,
  todaySpecialEvents,
  onNavigate,
  onOpenEvent,
  onOpenInjuryReport,
}) {
  const game = preview?.game || null;
  const home = preview?.homeTeam || null;
  const away = preview?.awayTeam || null;
  const previewTitle = preview?.previewLabel || (game ? "Game Preview" : "Calendar Day");

  const hasEvents = Array.isArray(selectedDayEvents) && selectedDayEvents.length > 0;
  const hasRawGames = Array.isArray(selectedDayGamesRaw) && selectedDayGamesRaw.length > 0;
  const hasInjuries = Array.isArray(selectedDayInjuryRows) && selectedDayInjuryRows.length > 0;

  const eventCount = selectedDayEvents?.length || 0;
  const gameCount = selectedDayGamesRaw?.length || selectedDayGames?.length || 0;
  const venue = preview?.venue || game?.venue || game?.arena || getArenaName(home);
  const hasAnalysis = Array.isArray(preview?.analysis) && preview.analysis.length > 0;
  const completedPreview = game ? isCompletedGame(game) : false;

  return (
    <section className="nhlcal-card nhlcal-preview-card nhlcal-scroll-surface">
      <header className="nhlcal-card-header">
        <div>
          <p>{selectedDateHeader}</p>
          <h3>{game ? previewTitle : hasEvents ? "Calendar Events" : "Calendar Day"}</h3>
        </div>

        <span className="nhlcal-header-pill">
          {gameCount} game{gameCount === 1 ? "" : "s"} · {selectedDayTeamGameCount} team · {eventCount} event
          {eventCount === 1 ? "" : "s"}
        </span>
      </header>

      <div className="nhlcal-selected-summary-strip">
        <span>{gameCount} league games</span>
        <span>{selectedDayTeamGameCount} team games</span>
        <span>{eventCount} events</span>
      </div>

      <div className={`nhlcal-tab-row ${hasAnalysis ? "nhlcal-tab-row-three" : "nhlcal-tab-row-two"}`}>
        <button
          type="button"
          className={activePanel === "game_preview" ? "is-active" : ""}
          onClick={() => setActivePanel("game_preview")}
        >
          Preview
        </button>

        {hasAnalysis ? (
          <button
            type="button"
            className={activePanel === "matchup_analysis" ? "is-active" : ""}
            onClick={() => setActivePanel("matchup_analysis")}
          >
            Matchup
          </button>
        ) : null}

        <button
          type="button"
          className={activePanel === "events" ? "is-active" : ""}
          onClick={() => setActivePanel("events")}
        >
          Events
        </button>
      </div>

      {activePanel === "events" ? (
        <SelectedDayEventsPanel
          events={selectedDayEvents}
          injuries={selectedDayInjuryRows}
          games={selectedDayGamesRaw}
          activeTeam={activeTeam}
          allTeams={allTeams}
          onOpenEvent={onOpenEvent}
          onOpenInjuryReport={onOpenInjuryReport}
        />
      ) : game ? (
        <>
          {!preview?.isUserTeamGame ? (
            <div className="nhlcal-preview-mode-banner">League Game Preview - your club is not on the ice this date.</div>
          ) : null}

          {preview?.missingFields?.length ? (
            <div className="nhlcal-preview-missing">
              Backend data missing for: {preview.missingFields.join(", ")}.
            </div>
          ) : null}

          <div className="nhlcal-matchup-stage nhlcal-matchup-stage-compact">
            <div className="nhlcal-matchup-team">
              <TeamIdentityBadge team={away} size="matchup" />
              <strong>{getTeamAbbreviation(away)}</strong>
              {preview?.awayRecord ? <span>{preview.awayRecord}</span> : null}
            </div>

            <div className="nhlcal-versus">
              <strong>{completedPreview ? "FINAL" : "VS"}</strong>
              <span>
                {completedPreview
                  ? getScoreLine(game)
                  : normalizeGameTime(game.time || game.start_time || game.startTime) || "Time TBD"}
              </span>
              {venue ? <small>{venue}</small> : null}
            </div>

            <div className="nhlcal-matchup-team">
              <TeamIdentityBadge team={home} size="matchup" />
              <strong>{getTeamAbbreviation(home)}</strong>
              {preview?.homeRecord ? <span>{preview.homeRecord}</span> : null}
            </div>
          </div>

          {activePanel === "game_preview" ? (
            preview.lines.length ? (
              <div className="nhlcal-preview-lines nhlcal-preview-lines-compact">
                {preview.lines.map((line) => (
                  <div key={line.label}>
                    <span>{line.label}</span>
                    <strong>{line.value}</strong>
                  </div>
                ))}
              </div>
            ) : null
          ) : preview.analysis.length ? (
            <div className="nhlcal-preview-lines nhlcal-preview-lines-compact">
              {preview.analysis.map((line) => (
                <div key={line.label}>
                  <span>{line.label}</span>
                  <strong>{line.value}</strong>
                </div>
              ))}
            </div>
          ) : (
            <div className="nhlcal-preview-empty-stats">Team stats will appear when backend data is available.</div>
          )}

          {hasEvents ? (
            <div className="nhlcal-selected-event-strip">
              {selectedDayEvents.slice(0, 3).map((event) => (
                <button
                  type="button"
                  key={event.id}
                  className={`nhlcal-selected-event-chip tone-${normalizeKey(event.tone || "league")}`}
                  title={event.title || event.description || "League Event"}
                  onClick={() => onOpenEvent?.(event)}
                >
                  <span>{event.icon || "◆"}</span>
                  <strong title={event.title || event.description || "League Event"}>{event.title}</strong>
                </button>
              ))}
              {activePanel === "game_preview" && !preview?.isUserTeamGame ? (
                <button
                  type="button"
                  className="nhlcal-selected-event-chip tone-league"
                  title="Open the Events tab for full details"
                  onClick={() => setActivePanel("events")}
                >
                  <span>◆</span>
                  <strong title="Open Events tab">View all events ({eventCount})</strong>
                </button>
              ) : null}
            </div>
          ) : null}

          <div className="nhlcal-wide-action muted" style={{ textAlign: "center" }}>
            Storylines + injury reports drive live sim effects.
          </div>
        </>
      ) : (
        <div className="nhlcal-empty-preview">
          <div className="nhlcal-empty-orb">{hasEvents ? selectedDayEvents[0]?.icon || "◆" : "◌"}</div>
          <h4>{hasEvents ? "Special Events Scheduled" : "No Featured Team Game"}</h4>
          <p>
            {hasEvents
              ? `${eventCount} league calendar event${eventCount === 1 ? "" : "s"} on this date — open Events for details, deadlines, and actions.`
              : `This day has ${selectedDayGames.length || "no"} visible game${
                  selectedDayGames.length === 1 ? "" : "s"
                }. Select another date or advance the calendar.`}
          </p>

          {hasEvents ? (
            <>
              <div className="nhlcal-selected-event-strip">
                {selectedDayEvents.slice(0, 3).map((event) => (
                  <button
                    type="button"
                    key={event.id}
                    className={`nhlcal-selected-event-chip tone-${normalizeKey(event.tone || "league")}`}
                    title={event.title || event.description || "League Event"}
                    onClick={() => onOpenEvent?.(event)}
                  >
                    <span>{event.icon || "◆"}</span>
                    <strong title={event.title || event.description || "League Event"}>{event.title}</strong>
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="nhlcal-wide-action"
                onClick={() => setActivePanel("events")}
              >
                Open Events Panel
              </button>
            </>
          ) : null}

          {!hasEvents && todayTeamGame ? (
            <p className="nhlcal-empty-subnote">
              Today: {getTeamAbbreviation(getAwayTeam(todayTeamGame, allTeams))} @{" "}
              {getTeamAbbreviation(getHomeTeam(todayTeamGame, allTeams))}
            </p>
          ) : null}

          {!hasEvents && !todayTeamGame && todaySpecialEvents?.length ? (
            <p className="nhlcal-empty-subnote">
              Today has {todaySpecialEvents.length} league event{todaySpecialEvents.length === 1 ? "" : "s"}.
            </p>
          ) : null}
        </div>
      )}
    </section>
  );
}

function SelectedDayEventsPanel({
  events,
  injuries,
  games,
  activeTeam,
  allTeams,
  onOpenEvent,
  onOpenInjuryReport,
}) {
  const visibleEvents = Array.isArray(events) ? events : EMPTY_ARRAY;
  const visibleInjuries = Array.isArray(injuries) ? injuries : EMPTY_ARRAY;
  const visibleGames = Array.isArray(games) ? games : EMPTY_ARRAY;

  return (
    <div className="nhlcal-selected-day-panel">
      {visibleEvents.length ? (
        <section className="nhlcal-selected-section">
          <header className="nhlcal-selected-section-head">
            <span>Special Events</span>
            <strong>{visibleEvents.length}</strong>
          </header>

          <div className="nhlcal-selected-event-list">
            {visibleEvents.map((event) => (
              <button
                key={event.id}
                type="button"
                className={`nhlcal-selected-event-row tone-${normalizeKey(event.tone || "league")}`}
                title={event.title || event.description || "League Event"}
                onClick={() => onOpenEvent?.(event)}
              >
                <span className="nhlcal-selected-event-icon">{event.icon || "◆"}</span>
                <div>
                  <strong title={event.title || event.description || "League Event"}>
                    {event.title || "League Event"}
                  </strong>
                  <small>
                    {event.priority || "MEDIUM"}
                    {event.description ? ` · ${event.description}` : ""}
                  </small>
                </div>
              </button>
            ))}
          </div>
        </section>
      ) : (
        <section className="nhlcal-selected-section">
          <header className="nhlcal-selected-section-head">
            <span>Special Events</span>
            <strong>0</strong>
          </header>
          <p className="nhlcal-small-empty">No league events are attached to this date.</p>
        </section>
      )}

      {visibleInjuries.length ? (
        <section className="nhlcal-selected-section">
          <header className="nhlcal-selected-section-head">
            <span>Date Injury Notes</span>
            <button type="button" onClick={onOpenInjuryReport}>
              Full Report
            </button>
          </header>

          <div className="nhlcal-selected-injury-list">
            {visibleInjuries.map((injury) => (
              <article key={injury.id}>
                <strong>{injury.playerName}</strong>
                <span>
                  {injury.teamAbbr || injury.teamId || getTeamAbbreviation(activeTeam)} · {injury.injuryLabel} ·{" "}
                  {injury.returnText || injury.status || "Active"}
                </span>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {visibleGames.length ? (
        <section className="nhlcal-selected-section">
          <header className="nhlcal-selected-section-head">
            <span>League Slate</span>
            <strong>{visibleGames.length}</strong>
          </header>

          <div className="nhlcal-selected-slate-list">
            {visibleGames.slice(0, 8).map((game, index) => {
              const away = getAwayTeam(game, allTeams);
              const home = getHomeTeam(game, allTeams);
              const scoreOrTime = isCompletedGame(game) ? getScoreLine(game) : normalizeGameTime(game.time);

              return (
                <article
                  key={game.id || `${game.date}-${index}`}
                  className={isTeamGame(game, activeTeam) ? "is-user-game" : ""}
                >
                  <span>{getTeamAbbreviation(away)}</span>
                  <em>@</em>
                  <span>{getTeamAbbreviation(home)}</span>
                  <strong>{scoreOrTime}</strong>
                </article>
              );
            })}

            {visibleGames.length > 8 ? (
              <p className="nhlcal-selected-more">+{visibleGames.length - 8} more games on league slate</p>
            ) : null}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function StandingsCard({ activeTeam, allTeams = [], rows, groupLabel = "League Standings", extended = false, onOpenFull }) {
  return (
    <section className={`nhlcal-card nhlcal-standings-card ${extended ? "nhlcal-standings-card--extended" : ""}`}>
      <header className="nhlcal-card-header compact nhlcal-standings-header">
        <div>
          <p>{groupLabel}</p>
          <h3>Standings Snapshot</h3>
        </div>

        <div className="nhlcal-standings-header-actions">
          {rows.length ? <span className="nhlcal-standings-count">{rows.length} teams</span> : null}
          <button type="button" className="nhlcal-standings-full-link" onClick={onOpenFull}>
            Full Standings
          </button>
        </div>
      </header>

      <div className="nhlcal-standings-table" role="table" aria-label={`${groupLabel} standings`}>
        <div className="nhlcal-standings-head" role="row">
          <span role="columnheader">#</span>
          <span role="columnheader">Team</span>
          <span role="columnheader">GP</span>
          <span role="columnheader">W</span>
          <span role="columnheader">L</span>
          <span role="columnheader">OTL</span>
          <span role="columnheader">PTS</span>
          <span role="columnheader">P%</span>
        </div>

        <div className="nhlcal-standings-body nhlcal-scroll-surface" role="rowgroup">
          {rows.length ? (
            rows.map((row, index) => {
              const team = resolveStandingsTeamRow(row, allTeams);
              const isUserTeam = isSameTeam(team, activeTeam);
              const gp = safeNumber(row.gp, 0);

              return (
                <div
                  key={row.id || getTeamId(team) || row.abbr || index}
                  className={`nhlcal-standings-row ${isUserTeam ? "is-user-team" : ""}`}
                  role="row"
                >
                  <span role="cell">{row.rank || index + 1}</span>
                  <span role="cell">
                    <TeamIdentityBadge team={team} size="mini" />
                    <strong title={getTeamDisplayName(team)}>{getTeamAbbreviation(team)}</strong>
                  </span>
                  <span role="cell">{gp}</span>
                  <span role="cell">{safeNumber(row.w, 0)}</span>
                  <span role="cell">{safeNumber(row.l, 0)}</span>
                  <span role="cell">{safeNumber(row.otl, 0)}</span>
                  <span role="cell">{safeNumber(row.pts, 0)}</span>
                  <span role="cell">{gp === 0 ? "—" : formatPointPct(row.pointPct, gp)}</span>
                </div>
              );
            })
          ) : (
            <div className="nhlcal-table-empty">Standings will appear once the season begins.</div>
          )}
        </div>
      </div>

      <footer className="nhlcal-standings-footer">
        <button type="button" className="nhlcal-standings-full-button" onClick={onOpenFull}>
          Open Full Standings
        </button>
      </footer>
    </section>
  );
}

function UpcomingStretchCard({ games, activeTeam, allTeams }) {
  return (
    <section className="nhlcal-card nhlcal-stretch-card">
      <header className="nhlcal-mini-header">
        <h3>Upcoming Stretch</h3>
        <span>Next {games.length || 0}</span>
      </header>

      <div className="nhlcal-stretch-list">
        {games.length ? (
          games.map((game, index) => {
            const opponent = getOpponentFromGame(game, activeTeam, allTeams);
            const isHome = isHomeGame(game, activeTeam);

            return (
              <div key={game.id || `${game.date}-${index}`} className="nhlcal-stretch-row">
                <span>{formatMonthDay(game.date)}</span>
                <TeamIdentityBadge team={opponent} size="mini" />
                <strong>
                  {isHome ? "vs" : "@"} {getTeamAbbreviation(opponent)}
                </strong>
                <em className={isHome ? "home" : "away"}>{isHome ? "Home" : "Away"}</em>
              </div>
            );
          })
        ) : (
          <p className="nhlcal-small-empty">No upcoming team games found.</p>
        )}
      </div>

      <button type="button" className="nhlcal-mini-button">
        View Full Schedule ›
      </button>
    </section>
  );
}

function LeagueStateCard({ rows }) {
  return (
    <section className="nhlcal-card nhlcal-league-card">
      <header className="nhlcal-mini-header">
        <h3>League State</h3>
        <span>Today</span>
      </header>

      <div className="nhlcal-league-list">
        {rows.length ? (
          rows.map((row, index) => (
            <div
              key={row.id || `${row.away}-${row.home}-${index}`}
              className={`nhlcal-league-row ${row.involvesUserTeam ? "is-highlight" : ""}`}
            >
              <span>{row.away}</span>
              <em>@</em>
              <span>{row.home}</span>
              <strong>{row.time}</strong>
            </div>
          ))
        ) : (
          <p className="nhlcal-small-empty">No league games loaded for today.</p>
        )}
      </div>

      <button type="button" className="nhlcal-mini-button">
        All Scores
      </button>
    </section>
  );
}

function StorylinesReportCard({ rows, storylineChoices, onChoose, busyChoiceId = "" }) {
  const choicesByStoryId = useMemo(() => {
    const map = new Map();

    (storylineChoices || []).forEach((row) => {
      map.set(String(row.storyline_id || row.decision_id || row.id || ""), row);
    });

    return map;
  }, [storylineChoices]);

  return (
    <section className="nhlcal-card nhlcal-stretch-card">
      <header className="nhlcal-mini-header">
        <h3>Storylines</h3>
        <span>Latest</span>
      </header>

      <div className="nhlcal-stretch-list nhlcal-storyline-list">
        {rows.length ? (
          rows.map((row, index) => {
            const choiceRow = choicesByStoryId.get(String(row.id || row.storyline_id || ""));

            return (
              <div key={row.id || index} className="nhlcal-storyline-row">
                <div className="nhlcal-storyline-topline">
                  <span>{formatMonthDay(row.date) || "Today"}</span>
                  <strong>{row.headline}</strong>
                  <em className={String(row.priority || "").toLowerCase()}>{row.priority}</em>
                </div>

                {row.cause ? <div className="nhlcal-subtext">Cause: {row.cause}</div> : null}
                {row.effect_summary ? <div className="nhlcal-subtext">Effect: {row.effect_summary}</div> : null}

                {Object.keys(row.effects || {}).length ? (
                  <div className="nhlcal-subtext">
                    {Object.entries(row.effects)
                      .map(([k, v]) => `${k.replace(/_/g, " ")} ${Number(v) > 0 ? "+" : ""}${v}`)
                      .join(" · ")}
                  </div>
                ) : null}

                {choiceRow && Array.isArray(choiceRow.action_options) ? (
                  <div className="nhlcal-storyline-choice-row">
                    {choiceRow.action_options.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        className="nhlcal-storyline-choice-button"
                        disabled={busyChoiceId === `${choiceRow.storyline_id}:${opt.id}`}
                        onClick={() => onChoose?.(choiceRow.storyline_id, opt.id)}
                        title={opt.effect_summary || ""}
                      >
                        {busyChoiceId === `${choiceRow.storyline_id}:${opt.id}` ? "Applying..." : opt.label}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })
        ) : (
          <p className="nhlcal-small-empty">No storyline events yet.</p>
        )}
      </div>
    </section>
  );
}

function InjuryReportCard({ rows, team, onOpenFull }) {
  return (
    <section className="nhlcal-card nhlcal-league-card">
      <header className="nhlcal-mini-header">
        <h3>Injury Report</h3>
        <button type="button" onClick={onOpenFull}>
          {getTeamAbbreviation(team)}
        </button>
      </header>

      <div className="nhlcal-league-list nhlcal-injury-mini-list">
        {rows.length ? (
          rows.map((row, index) => (
            <div key={row.id || index} className="nhlcal-injury-mini-row">
              <span>{row.player}</span>
              <em>{row.tier}</em>
              <strong>
                {row.gamesRemaining > 0
                  ? `${row.gamesRemaining}g`
                  : String(row.status || "").toUpperCase().includes("DAY")
                    ? "DTD"
                    : row.status || "—"}
              </strong>
              <small>{row.returnText || row.date || "active"}</small>
            </div>
          ))
        ) : (
          <p className="nhlcal-small-empty">No active injuries reported for your club.</p>
        )}
      </div>

      <button type="button" className="nhlcal-mini-button" onClick={onOpenFull}>
        Full Injury Report
      </button>
    </section>
  );
}

function DiagnosticsPanel({ diagnostics, insights }) {
  return (
    <section className="nhlcal-card nhlcal-diagnostics-panel">
      <header className="nhlcal-section-title">
        <div>
          <p>Schedule Diagnostics & Insights</p>
          <h3>Calendar Intelligence</h3>
        </div>
        <span>{diagnostics?.scheduleStrengthLabel || "Live"}</span>
      </header>

      <div className="nhlcal-diagnostic-grid">
        <DiagnosticTile
          label="Back-to-Backs"
          value={diagnostics.backToBacks}
          sub={diagnostics.backToBackRank}
          icon="▣"
        />

        <DiagnosticTile
          label="3-in-4 Stretches"
          value={diagnostics.threeInFour}
          sub={diagnostics.threeInFourRank}
          icon="⌁"
        />

        <DiagnosticTile
          label="Longest Road Trip"
          value={diagnostics.longestRoadTrip}
          sub={diagnostics.longestRoadTripDates}
          icon="⌖"
        />

        <DiagnosticTile
          label="Current Streak"
          value={diagnostics.currentStreak}
          sub={diagnostics.currentStreakSub}
          icon="↗"
          danger={String(diagnostics.currentStreak || "").startsWith("L")}
        />
      </div>

      <div className="nhlcal-insight-row">
        {insights.map((insight, index) => (
          <article key={`${insight.title}-${index}`} className="nhlcal-insight-pill">
            <span>{insight.icon}</span>
            <p>{insight.title}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function DiagnosticTile({ label, value, sub, icon, danger }) {
  return (
    <article className={`nhlcal-diagnostic-tile ${danger ? "is-danger" : ""}`}>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{sub}</small>
      </div>
      <em>{icon}</em>
    </article>
  );
}

function MonthSnapshotPanel({ activeMonthGames, activeMonthEvents, activeTeam, allTeams, selectedDateISO }) {
  const monthGames = activeMonthGames || EMPTY_ARRAY;
  const monthEvents = activeMonthEvents || EMPTY_ARRAY;

  const teamMonthGames = monthGames.filter((game) => isTeamGame(game, activeTeam));
  const completed = teamMonthGames.filter(isCompletedGame);
  const upcoming = teamMonthGames.filter((game) => !isCompletedGame(game));

  const homeGames = teamMonthGames.filter((game) => isHomeGame(game, activeTeam)).length;
  const awayGames = teamMonthGames.filter((game) => !isHomeGame(game, activeTeam)).length;

  const criticalEvents = monthEvents.filter((event) => String(event.priority || "").toUpperCase() === "CRITICAL").length;
  const highEvents = monthEvents.filter((event) => String(event.priority || "").toUpperCase() === "HIGH").length;

  const featuredOpponents = upcoming
    .slice(0, 4)
    .map((game) => getOpponentFromGame(game, activeTeam, allTeams))
    .filter(Boolean);

  return (
    <section className="nhlcal-card nhlcal-month-snapshot">
      <header className="nhlcal-section-title">
        <div>
          <p>Month Snapshot</p>
          <h3>{formatMonthYear(selectedDateISO)}</h3>
        </div>
        <span>
          {teamMonthGames.length} team games · {monthEvents.length} events
        </span>
      </header>

      <div className="nhlcal-snapshot-grid">
        <div>
          <span>Completed</span>
          <strong>{completed.length}</strong>
        </div>

        <div>
          <span>Upcoming</span>
          <strong>{upcoming.length}</strong>
        </div>

        <div>
          <span>Home</span>
          <strong>{homeGames}</strong>
        </div>

        <div>
          <span>Away</span>
          <strong>{awayGames}</strong>
        </div>

        <div>
          <span>Events</span>
          <strong>{monthEvents.length}</strong>
        </div>

        <div>
          <span>Critical</span>
          <strong>{criticalEvents}</strong>
        </div>

        <div>
          <span>High</span>
          <strong>{highEvents}</strong>
        </div>

        <div>
          <span>Special</span>
          <strong>{monthEvents.filter((event) => normalizeKey(event.type).includes("classic") || normalizeKey(event.type).includes("star")).length}</strong>
        </div>
      </div>

      <div className="nhlcal-opponent-strip">
        {featuredOpponents.length ? (
          featuredOpponents.map((team, index) => (
            <div key={`${getTeamId(team)}-${index}`}>
              <TeamIdentityBadge team={team} size="small" />
              <span>{getTeamAbbreviation(team)}</span>
            </div>
          ))
        ) : monthEvents.length ? (
          monthEvents.slice(0, 4).map((event, index) => (
            <div key={event.id || index} className="nhlcal-opponent-event">
              <strong>{event.icon || "◆"}</strong>
              <span>{event.title}</span>
            </div>
          ))
        ) : (
          <p>No upcoming opponents or special events loaded for this month.</p>
        )}
      </div>
    </section>
  );
}
function FranchiseDrawer({
  activeTeam,
  rootState,
  onClose,
  onNavigate,
  players,
  prospects,
  injuries,
  draftClass,
  standings,
  notifications,
  leagueEvents,
  finance,
  calendarSpecialEvents,
  nextTeamGames,
  leagueStateRows,
}) {
  const [tab, setTab] = useState("hub");
  const statsCentral = useMemo(() => {
    return rootState?.stats_central || rootState?.statsCentral || EMPTY_OBJECT;
  }, [rootState]);

  const topPlayers = useMemo(() => {
    const rows = normalizeArrayMerged(
      statsCentral?.user_leaders,
      statsCentral?.userLeaders,
      statsCentral?.leaders,
      statsCentral?.league_leaders,
      players
    );
  
    const teamRows = rows.filter((player) => {
      if (isGoalieRow(player)) return false;
  
      const tid =
        player?.team_id ||
        player?.teamId ||
        player?.team ||
        player?.team_abbr ||
        player?.teamAbbr ||
        "";
  
      if (!tid) return true;
  
      return isSameTeamIdentifier(tid, activeTeam);
    });
  
    return teamRows
      .map((player) => ({
        ...player,
        points: getPlayerPoints(player),
        goals: getPlayerGoals(player),
        assists: getPlayerAssists(player),
        gp: getPlayerGamesPlayed(player),
        ppg: getPlayerPointsPerGame(player),
      }))
      .sort((a, b) => {
        if (getPlayerPoints(b) !== getPlayerPoints(a)) {
          return getPlayerPoints(b) - getPlayerPoints(a);
        }
  
        if (getPlayerGoals(b) !== getPlayerGoals(a)) {
          return getPlayerGoals(b) - getPlayerGoals(a);
        }
  
        return String(a.name || a.player_name || "").localeCompare(String(b.name || b.player_name || ""));
      })
      .slice(0, 10);
  }, [statsCentral, players, activeTeam]);

  const teamProspects = useMemo(() => {
    return [...prospects]
      .filter((player) => {
        const playerTeamId =
          player.team_id ||
          player.teamId ||
          player.rights_team_id ||
          player.rightsTeamId ||
          player.team ||
          player.rightsTeam;

        return !playerTeamId || isSameTeamIdentifier(playerTeamId, activeTeam);
      })
      .sort((a, b) => {
        const aEta = nhlEtaSortValue(a.eta || a.nhl_eta || a.arrival_year);
        const bEta = nhlEtaSortValue(b.eta || b.nhl_eta || b.arrival_year);
        if (aEta !== bEta) return aEta - bEta;
        return getPotentialScore(b) - getPotentialScore(a);
      })
      .slice(0, 6);
  }, [prospects, activeTeam]);

  const visibleDraftClass = useMemo(() => {
    return [...draftClass]
      .sort((a, b) => {
        const ar = Number(a.rank || a.overall_rank || a.draft_rank || 999);
        const br = Number(b.rank || b.overall_rank || b.draft_rank || 999);
        if (ar !== br) return ar - br;
        return getPotentialScore(b) - getPotentialScore(a);
      })
      .slice(0, 8);
  }, [draftClass]);

  const teamInjuries = useMemo(() => {
    const raw = Array.isArray(injuries) ? injuries : EMPTY_ARRAY;
    const mapped = raw.map((inj, i) => normalizeInjuryRowForUi(inj, i));

    return mapped
      .filter((row) => row.teamId && isSameTeamIdentifier(row.teamId, activeTeam))
      .slice(0, 6);
  }, [injuries, activeTeam]);

  const importantEvents = useMemo(() => {
    return [...(calendarSpecialEvents || EMPTY_ARRAY)]
      .sort((a, b) => {
        const ap = PRIORITY_ORDER[String(a.priority || "MEDIUM").toUpperCase()] ?? 2;
        const bp = PRIORITY_ORDER[String(b.priority || "MEDIUM").toUpperCase()] ?? 2;
        if (ap !== bp) return ap - bp;
        return String(a.date || "").localeCompare(String(b.date || ""));
      })
      .slice(0, 8);
  }, [calendarSpecialEvents]);

  const drawerTabs = [
    { key: "hub", label: "Hub" },
    { key: "roster", label: "Roster" },
    { key: "draft", label: "Draft" },
    { key: "league", label: "League" },
    { key: "office", label: "Office" },
  ];

  return (
    <div className="nhlcal-drawer-backdrop" onMouseDown={onClose}>
      <aside className="nhlcal-drawer" onMouseDown={(event) => event.stopPropagation()}>
        <header className="nhlcal-drawer-header">
          <div>
            <p>Command Menu</p>
            <h2>{getTeamDisplayName(activeTeam)}</h2>
          </div>

          <button type="button" onClick={onClose}>
            ×
          </button>
        </header>

        <nav className="nhlcal-drawer-tabs">
          {drawerTabs.map((item) => (
            <button
              key={item.key}
              type="button"
              className={tab === item.key ? "is-active" : ""}
              onClick={() => setTab(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {tab === "hub" ? (
          <div className="nhlcal-drawer-body">
            <DrawerSection title="Quick Navigation">
              <div className="nhlcal-drawer-grid">
                <DrawerNavCard title="Office" sub="Owner, inbox, goals" onClick={() => onNavigate(SCREEN_KEYS.office)} />
                <DrawerNavCard title="Roster" sub="Depth, contracts, roles" onClick={() => onNavigate(SCREEN_KEYS.roster)} />
                <DrawerNavCard title="Lineup" sub="Lines, special teams" onClick={() => onNavigate(SCREEN_KEYS.lineup)} />
                <DrawerNavCard title="Scouting" sub="Reports, draft board" onClick={() => onNavigate(SCREEN_KEYS.scouting)} />
                <DrawerNavCard title="Analytics" sub="Team trends" onClick={() => onNavigate(SCREEN_KEYS.analytics)} />
                <DrawerNavCard title="Finances" sub="Budget, cap, revenue" onClick={() => onNavigate(SCREEN_KEYS.finances)} />
              </div>
            </DrawerSection>

            <DrawerSection title="Calendar Alerts">
              <div className="nhlcal-drawer-feed">
                {importantEvents.length ? (
                  importantEvents.map((event, index) => (
                    <article key={event.id || index} className={`nhlcal-drawer-event tone-${normalizeKey(event.tone || "league")}`}>
                      <strong>
                        <span>{event.icon || "◆"}</span> {event.title || event.headline || "League Event"}
                      </strong>
                      <p>
                        {formatMonthDay(event.date)} · {event.priority || "MEDIUM"}
                        {event.description ? ` · ${event.description}` : ""}
                      </p>
                    </article>
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No special calendar events are currently loaded.</p>
                )}
              </div>
            </DrawerSection>

            <DrawerSection title="Recent Notifications">
              <div className="nhlcal-drawer-feed">
                {(notifications || EMPTY_ARRAY).slice(0, 6).map((item, index) => (
                  <article key={item.id || index}>
                    <strong>{item.title || item.headline || item.type || "Notification"}</strong>
                    <p>{item.message || item.body || item.text || item.description || "No details available."}</p>
                  </article>
                ))}

                {!(notifications || EMPTY_ARRAY).length ? (
                  <p className="nhlcal-drawer-empty">No notifications are currently loaded.</p>
                ) : null}
              </div>
            </DrawerSection>
          </div>
        ) : null}

        {tab === "roster" ? (
          <div className="nhlcal-drawer-body">
            <DrawerSection title="Team Leaders">
              <div className="nhlcal-player-list">
                {topPlayers.length ? (
                  topPlayers.map((player, index) => (
                    <PlayerMiniRow key={player.id || player.player_id || player.name || index} player={player} index={index} />
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No player stat data loaded for this team yet.</p>
                )}
              </div>
            </DrawerSection>

            <DrawerSection title="Injury Watch">
              <div className="nhlcal-drawer-feed">
                {teamInjuries.length ? (
                  teamInjuries.map((injury, index) => (
                    <article key={injury.id || index}>
                      <strong>{injury.playerName || injury.player || "Injured Player"}</strong>
                      <p>
                        {injury.injuryLabel || injury.description || "Injury"} ·{" "}
                        {injury.duration ||
                          injury.returnText ||
                          (injury.gamesRemaining > 0 ? `${injury.gamesRemaining} games` : "") ||
                          "—"}
                        {injury.status ? ` · ${injury.status}` : ""}
                      </p>
                    </article>
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No active injuries found for this team.</p>
                )}
              </div>
            </DrawerSection>

            <DrawerSection title="Upcoming Stretch">
              <div className="nhlcal-drawer-feed">
                {(nextTeamGames || EMPTY_ARRAY).length ? (
                  nextTeamGames.map((game, index) => {
                    const opponent = getOpponentFromGame(game, activeTeam, []);
                    return (
                      <article key={game.id || index}>
                        <strong>
                          {formatMonthDay(game.date)} · {isHomeGame(game, activeTeam) ? "vs" : "@"}{" "}
                          {getTeamAbbreviation(opponent)}
                        </strong>
                        <p>{normalizeGameTime(game.time)} · {game.venue || game.arena || "Arena TBD"}</p>
                      </article>
                    );
                  })
                ) : (
                  <p className="nhlcal-drawer-empty">No upcoming team games loaded.</p>
                )}
              </div>
            </DrawerSection>
          </div>
        ) : null}

        {tab === "draft" ? (
          <div className="nhlcal-drawer-body">
            <DrawerSection title="Draft Board Snapshot">
              <div className="nhlcal-draft-board">
                {visibleDraftClass.length ? (
                  visibleDraftClass.map((player, index) => (
                    <DraftMiniRow key={player.id || player.player_id || player.name || index} player={player} index={index} />
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No draft class loaded yet.</p>
                )}
              </div>
            </DrawerSection>

            <DrawerSection title="Prospect Pipeline">
              <div className="nhlcal-player-list">
                {teamProspects.length ? (
                  teamProspects.map((player, index) => (
                    <ProspectMiniRow key={player.id || player.player_id || player.name || index} player={player} index={index} />
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No owned prospects found yet.</p>
                )}
              </div>
            </DrawerSection>
          </div>
        ) : null}

        {tab === "league" ? (
          <div className="nhlcal-drawer-body">
            <DrawerSection title="Today League Slate">
              <div className="nhlcal-drawer-feed">
                {(leagueStateRows || EMPTY_ARRAY).length ? (
                  leagueStateRows.map((row, index) => (
                    <article key={row.id || index}>
                      <strong>
                        {row.away} @ {row.home}
                      </strong>
                      <p>{row.time}{row.involvesUserTeam ? " · Your club" : ""}</p>
                    </article>
                  ))
                ) : (
                  <p className="nhlcal-drawer-empty">No league slate loaded for today.</p>
                )}
              </div>
            </DrawerSection>

            <DrawerSection title="League Events">
              <div className="nhlcal-drawer-feed">
                {(leagueEvents || EMPTY_ARRAY).slice(0, 8).map((event, index) => (
                  <article key={event.id || index}>
                    <strong>{event.title || event.headline || event.type || "League Event"}</strong>
                    <p>{event.message || event.description || event.summary || event.date || "No event details available."}</p>
                  </article>
                ))}

                {!(leagueEvents || EMPTY_ARRAY).length ? (
                  <p className="nhlcal-drawer-empty">No league events are currently loaded.</p>
                ) : null}
              </div>
            </DrawerSection>

            <DrawerSection title="Standings Pulse">
              <div className="nhlcal-drawer-feed">
                {(standings || EMPTY_ARRAY).slice(0, 8).map((row, index) => (
                  <article key={row.id || row.team_id || row.team || index}>
                    <strong>
                      {index + 1}. {getTeamDisplayName(row.team || row)}
                    </strong>
                    <p>
                      {safeNumber(row.wins || row.w, 0)}-{safeNumber(row.losses || row.l, 0)}-
                      {safeNumber(row.otl || row.ot || row.overtime_losses, 0)} ·{" "}
                      {safeNumber(row.points || row.pts, 0)} pts
                    </p>
                  </article>
                ))}
              </div>
            </DrawerSection>
          </div>
        ) : null}

        {tab === "office" ? (
          <div className="nhlcal-drawer-body">
            <DrawerSection title="Office Snapshot">
              <div className="nhlcal-office-grid">
                <OfficeMetric label="GM" value={rootState?.gm_name || rootState?.general_manager || "User GM"} />
                <OfficeMetric label="Budget" value={formatMoney(finance?.budget || rootState?.budget)} />
                <OfficeMetric label="Cap Space" value={formatMoney(finance?.cap_space || rootState?.cap_space)} />
                <OfficeMetric label="Owner Trust" value={formatPercentLoose(rootState?.owner_trust || rootState?.ownerTrust)} />
              </div>
            </DrawerSection>

            <DrawerSection title="Office Actions">
              <div className="nhlcal-drawer-grid">
                <DrawerNavCard title="Inbox" sub="Messages and tasks" onClick={() => onNavigate(SCREEN_KEYS.inbox)} />
                <DrawerNavCard title="Finances" sub="Cap and revenue" onClick={() => onNavigate(SCREEN_KEYS.finances)} />
                <DrawerNavCard title="Analytics" sub="Reports and trends" onClick={() => onNavigate(SCREEN_KEYS.analytics)} />
                <DrawerNavCard title="Settings" sub="Game options" onClick={() => onNavigate(SCREEN_KEYS.settings)} />
              </div>
            </DrawerSection>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function DrawerSection({ title, children }) {
  return (
    <section className="nhlcal-drawer-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function DrawerNavCard({ title, sub, onClick }) {
  return (
    <button type="button" className="nhlcal-drawer-nav-card" onClick={onClick}>
      <strong>{title}</strong>
      <span>{sub}</span>
    </button>
  );
}

function PlayerMiniRow({ player, index }) {
  const name = player?.name || player?.player_name || player?.player || "Player";
  const pos = player?.position || player?.pos || "—";

  if (isGoalieRow(player)) {
    return (
      <article className="nhlcal-player-mini-row">
        <span>{index + 1}</span>
        <div>
          <strong>{name}</strong>
          <small>
            {pos} · {getPlayerGamesPlayed(player)} GP · {formatSavePct(getGoalieSavePct(player))} SV% ·{" "}
            {Number(getGoalieGAA(player)).toFixed(2)} GAA
          </small>
        </div>
        <b>{formatSavePct(getGoalieSavePct(player))}</b>
      </article>
    );
  }

  return (
    <article className="nhlcal-player-mini-row">
      <span>{index + 1}</span>
      <div>
        <strong>{name}</strong>
        <small>
          {pos} · {getPlayerGamesPlayed(player)} GP · {getPlayerGoals(player)}G · {getPlayerAssists(player)}A
        </small>
      </div>
      <b>{getPlayerPoints(player)}P</b>
    </article>
  );
}

function DraftMiniRow({ player, index }) {
  const stock = Number(player.stock_change || player.stockChange || player.rank_change || 0);

  return (
    <article className="nhlcal-draft-mini-row">
      <span>{player.rank || player.overall_rank || index + 1}</span>
      <div>
        <strong>{getPlayerName(player)}</strong>
        <small>
          {getPlayerPosition(player)} · {player.league || player.current_league || player.country || "Scouted"}
        </small>
      </div>
      <em className={stock >= 0 ? "up" : "down"}>
        {stock >= 0 ? "+" : ""}
        {stock}
      </em>
    </article>
  );
}

function formatNhlEta(value, fallback = "TBD") {
  if (value == null || value === "") return fallback;
  if (typeof value === "object") {
    const label = value.label;
    const years = value.years;
    if (label === "Now" || years === 0) return "NHL Ready";
    if (label != null && label !== "") return String(label);
    if (years != null && years !== "") return `${years}Y`;
    return fallback;
  }
  return String(value);
}

function nhlEtaSortValue(value) {
  if (value == null || value === "") return 9999;
  if (typeof value === "object") {
    const years = Number(value.years);
    if (Number.isFinite(years)) return years;
    const label = String(value.label || "");
    if (label === "Now") return 0;
    const match = label.match(/(\d+)/);
    if (match) return Number(match[1]);
    return 9999;
  }
  const n = Number(value);
  if (Number.isFinite(n)) return n;
  const match = String(value).match(/(\d+)/);
  return match ? Number(match[1]) : 9999;
}

function ProspectMiniRow({ player, index }) {
  return (
    <article className="nhlcal-player-mini-row">
      <span>{index + 1}</span>
      <div>
        <strong>{getPlayerName(player)}</strong>
        <small>
          {getPlayerPosition(player)} · ETA {formatNhlEta(player.eta || player.nhl_eta || player.arrival_year, "TBD")}
        </small>
      </div>
      <em>{player.potential || player.ceiling || player.grade || "—"}</em>
    </article>
  );
}

function OfficeMetric({ label, value }) {
  return (
    <article className="nhlcal-office-metric">
      <span>{label}</span>
      <strong>{value || "—"}</strong>
    </article>
  );
}

function CalendarSettingsModal({
  onClose,
  denseMode,
  setDenseMode,
  showOnlyTeamGames,
  setShowOnlyTeamGames,
  currentDate,
  activeTeam,
  games,
  events,
}) {
  const totalGames = Array.isArray(games) ? games.length : 0;
  const teamGames = Array.isArray(games) ? games.filter((game) => isTeamGame(game, activeTeam)).length : 0;
  const totalEvents = Array.isArray(events) ? events.length : 0;
  const criticalEvents = Array.isArray(events)
    ? events.filter((event) => String(event.priority || "").toUpperCase() === "CRITICAL").length
    : 0;

  return (
    <div className="nhlcal-modal-backdrop" onMouseDown={onClose}>
      <section className="nhlcal-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>
            <p>Calendar Settings</p>
            <h2>Schedule Display</h2>
          </div>

          <button type="button" onClick={onClose}>
            ×
          </button>
        </header>

        <div className="nhlcal-settings-list">
          <button
            type="button"
            className={denseMode ? "is-active" : ""}
            onClick={() => setDenseMode((value) => !value)}
          >
            <span>Dense Calendar</span>
            <strong>{denseMode ? "On" : "Off"}</strong>
          </button>

          <button
            type="button"
            className={showOnlyTeamGames ? "is-active" : ""}
            onClick={() => setShowOnlyTeamGames(true)}
          >
            <span>Team Games Only</span>
            <strong>{showOnlyTeamGames ? "On" : "Off"}</strong>
          </button>

          <button
            type="button"
            className={!showOnlyTeamGames ? "is-active" : ""}
            onClick={() => setShowOnlyTeamGames(false)}
          >
            <span>League Games + Events</span>
            <strong>{!showOnlyTeamGames ? "On" : "Off"}</strong>
          </button>
        </div>

        <div className="nhlcal-settings-summary">
          <article>
            <span>Current Date</span>
            <strong>{formatLongDate(currentDate)}</strong>
          </article>

          <article>
            <span>Loaded League Games</span>
            <strong>{totalGames}</strong>
          </article>

          <article>
            <span>{getTeamAbbreviation(activeTeam)} Games</span>
            <strong>{teamGames}</strong>
          </article>

          <article>
            <span>Special Events</span>
            <strong>{totalEvents}</strong>
          </article>

          <article>
            <span>Critical Events</span>
            <strong>{criticalEvents}</strong>
          </article>
        </div>
      </section>
    </div>
  );
}

function TeamInitialBadge({ team, size = "small" }) {
  const abbr = getTeamAbbreviation(team);
  const seed = getTeamColorSeed(team);

  return (
    <span className={`nhlcal-team-badge size-${size}`} style={{ "--team-seed": seed }}>
      {abbr}
    </span>
  );
}

function getTeamLogoSrc(team) {
  if (!team) return null;

  const rawAbbr = String(getTeamAbbreviation(team) || "").toUpperCase();
  const mappedAbbr = TEAM_ABBR_LOGO_ALIASES[rawAbbr] || rawAbbr;
  const preferredName =
    TEAM_LOGO_NAME_OVERRIDES[mappedAbbr] || TEAM_LOGO_NAME_OVERRIDES[rawAbbr] || "";

  const candidates = [
    preferredName,
    TEAM_LOGO_NAME_OVERRIDES[rawAbbr],
    team.full_name,
    team.fullName,
    team.name,
    team.team_name,
    team.nickname,
    [team.city, team.name].filter(Boolean).join(" "),
    team.city,
    mappedAbbr,
    rawAbbr,
  ]
    .map((value) => String(value || "").trim())
    .filter(Boolean);

  const seen = new Set();

  for (let i = 0; i < candidates.length; i += 1) {
    const key = normalizeLogoToken(candidates[i]);
    if (!key || seen.has(key)) continue;
    seen.add(key);

    const src = TEAM_LOGO_MAP.get(key);
    if (src) return src;
  }

  return null;
}

function resolveStandingsTeamRow(row, allTeams = []) {
  if (!row) return null;

  const refs = [
    row.team,
    row.team_id,
    row.teamId,
    row.id,
    row.abbreviation,
    row.abbr,
    row.team_abbrev,
    row.team_abbreviation,
    row.name,
    row.team_name,
  ];

  let team = null;

  for (const ref of refs) {
    if (!ref) continue;
    team = findTeamByAny(allTeams, ref);
    if (team) break;
  }

  if (!team && row.team && typeof row.team === "object") {
    team = row.team;
  }

  if (!team) {
    team = normalizeTeam(row, 0);
  }

  return enrichTeamWithStandings(
    normalizeTeam(
      {
        ...team,
        abbreviation:
          canonicalizeTeamAbbr(
            team.abbreviation ||
              team.abbr ||
              row.abbreviation ||
              row.abbr ||
              row.abbrev ||
              row.team_abbrev ||
              row.team_abbreviation
          ) ||
          lookupNhlAbbrFromLabel(team.name || row.name || row.team_name) ||
          team.abbreviation ||
          team.abbr,
        name: team.name || row.name || row.team_name,
        city: team.city || row.city,
        division: team.division || row.division,
        conference: team.conference || row.conference,
      },
      0
    ),
    [row]
  );
}

function TeamIdentityBadge({ team, size = "small" }) {
  const src = getTeamLogoSrc(team);
  const label = getTeamDisplayName(team);

  if (!src) return <TeamInitialBadge team={team} size={size} />;

  return (
    <span className={`nhlcal-team-logo size-${size}`}>
      <img src={src} alt={`${label} logo`} loading="lazy" />
    </span>
  );
}

function normalizeFranchiseState(rootState, controlledTeamId) {
  const allTeams = normalizeTeams(rootState);
  const activeTeam = findActiveTeam(rootState, allTeams, controlledTeamId);
  const games = normalizeGames(rootState, allTeams);
  const standings = normalizeStandings(rootState, allTeams);

  const players = normalizeArrayMerged(
    rootState.players,
    rootState.roster,
    rootState.player_stats,
    rootState.playerStats,
    rootState.skater_stats,
    rootState.skaters,
    rootState.league_players
  );

  const prospects = normalizeArrayMerged(
    rootState.prospects,
    rootState.pipeline,
    rootState.team_prospects,
    rootState.prospect_pool
  );

  const draftClass = normalizeArrayMerged(
    rootState.draft_class,
    rootState.draftClass,
    rootState.scouting?.draft_class,
    rootState.scouting?.draftClass,
    rootState.prospect_rankings,
    rootState.draft_board
  );

  const injuries = normalizeArrayMerged(
    rootState.injuries,
    rootState.injuries_recent,
    rootState.injuries_active,
    rootState.injury_log,
    rootState.injuryLog,
    rootState.injury_log_all,
    rootState.active_injuries
  );

  const leagueEvents = normalizeArrayMerged(
    rootState.storyline_events,
    rootState.storylineEvents,
    rootState.league_events,
    rootState.leagueEvents,
    rootState.timeline,
    rootState.news_events,
    rootState.newsEvents
  );

  const notifications = normalizeArrayMerged(
    rootState.notifications,
    rootState.inbox_notifications,
    rootState.office_notifications,
    rootState.messages
  );

  const inbox = normalizeArrayMerged(rootState.inbox, rootState.messages, rootState.notifications);

  return {
    currentDate:
      rootState.current_date ||
      rootState.currentDate ||
      rootState.date ||
      rootState.today ||
      rootState.nhl_today?.iso ||
      rootState.calendar_date ||
      new Date(),
    activeTeam,
    allTeams,
    games,
    standings,
    leagueEvents,
    notifications,
    players,
    prospects,
    injuries,
    draftClass,
    leagueState: rootState.league_state || rootState.leagueState || EMPTY_OBJECT,
    statsCentral: rootState.stats_central || rootState.statsCentral || EMPTY_OBJECT,
    preseasonPlayerStats:
      rootState.preseason_player_stats_snapshot ||
      rootState.preseasonPlayerStats ||
      EMPTY_OBJECT,
    finance: rootState.finance || rootState.finances || EMPTY_OBJECT,
    inbox,
  };
}

function mergeUniqueTeams(primary, extras) {
  const out = [];

  const pushIfNew = (team) => {
    if (!team) return;
    const exists = out.some((t) => isSameTeam(t, team));
    if (!exists) out.push(team);
  };

  (primary || []).forEach(pushIfNew);
  (extras || []).forEach(pushIfNew);

  return out;
}

function enrichTeamWithStandings(team, standings) {
  if (!team) return team;

  const row = (standings || []).find((r) => isSameTeam(r.team || r, team));
  if (!row) return team;

  return normalizeTeam(
    {
      ...team,
      wins: firstNumber(team.wins, row.wins, row.w),
      losses: firstNumber(team.losses, row.losses, row.l),
      otl: firstNumber(team.otl, row.otl, row.ot),
      points: firstNumber(team.points, row.points, row.pts),
      gp: firstNumber(team.gp, row.gp),
      goals_for: firstNumber(team.goals_for, row.goals_for, row.gf),
      goals_against: firstNumber(team.goals_against, row.goals_against, row.ga),
      pp_pct: firstNumber(team.pp_pct, row.pp_pct, row.power_play_pct),
      pk_pct: firstNumber(team.pk_pct, row.pk_pct, row.penalty_kill_pct),
    },
    0
  );
}

function normalizeTeams(rootState) {
  const candidates = normalizeArrayMerged(
    rootState.teams,
    rootState.all_teams,
    rootState.allTeams,
    rootState.league?.teams,
    rootState.franchise?.teams
  );

  if (candidates.length) {
    const baseTeams = candidates.map((team, index) => normalizeTeam(team, index));

    const standingRows = normalizeArrayMerged(
      rootState.standings,
      rootState.league_standings,
      rootState.leagueStandings
    );

    const fromStandings = standingRows.map((row, index) =>
      normalizeTeam(
        {
          id: row.team_id || row.teamId || row.id || `stand-${index}`,
          team_id: row.team_id || row.teamId || row.id || `stand-${index}`,
          abbreviation: row.team_abbrev || row.team_abbreviation || row.abbreviation || row.abbr,
          name: row.name || row.team_name || row.teamName || row.team || row.team_display_name,
          city: row.city || row.team_city || "",
          wins: row.w || row.wins,
          losses: row.l || row.losses,
          otl: row.otl || row.ot,
          points: row.pts || row.points,
          gp: row.gp || row.games_played,
          goals_for: row.gf || row.goals_for,
          goals_against: row.ga || row.goals_against,
          pp_pct: row.pp_pct || row.power_play_pct,
          pk_pct: row.pk_pct || row.penalty_kill_pct,
        },
        index + 1000
      )
    );

    return mergeUniqueTeams(baseTeams, fromStandings);
  }

  const team = rootState.team || rootState.user_team || rootState.selected_team || rootState.active_team;
  if (team) return [normalizeTeam(team, 0)];

  return [
    normalizeTeam(
      {
        id: "USER",
        abbreviation: "CLB",
        name: "Club",
        city: "",
        wins: 0,
        losses: 0,
        otl: 0,
        points: 0,
      },
      0
    ),
  ];
}

function normalizeTeam(team, index = 0) {
  if (!team || typeof team !== "object") {
    return {
      id: String(team || index),
      team_id: String(team || index),
      abbreviation: String(team || "TBD").slice(0, 3).toUpperCase(),
      name: String(team || "Team"),
      city: "",
      division: "League",
      conference: "League",
      wins: 0,
      losses: 0,
      otl: 0,
      points: 0,
      gp: 0,
      goals_for: 0,
      goals_against: 0,
      pp_pct: 0,
      pk_pct: 0,
    };
  }

  const wins = firstNumber(team.wins, team.w, team.record?.wins, team.record?.w);
  const losses = firstNumber(team.losses, team.l, team.record?.losses, team.record?.l);
  const otl = firstNumber(team.otl, team.ot, team.overtime_losses, team.record?.otl, team.record?.ot);
  const gp = firstNumber(team.gp, team.games_played, team.gamesPlayed, wins + losses + otl);

  return {
    ...team,
    id: getTeamId(team) || team.id || team.team_id || team.abbreviation || team.abbr || `team-${index}`,
    team_id: team.team_id || team.id || team.abbreviation || team.abbr || `team-${index}`,
    abbreviation: getTeamAbbreviation(team),
    name: team.name || team.team_name || team.nickname || team.full_name || team.fullName || getTeamAbbreviation(team),
    city: team.city || team.location || team.market || team.team_city || "",
    division: team.division || team.division_name || team.div || "League",
    conference: team.conference || team.conf || "League",
    wins,
    losses,
    otl,
    gp,
    points: firstNumber(team.points, team.pts, wins * 2 + otl),
    goals_for: firstNumber(team.goals_for, team.goalsFor, team.gf, team.stats?.goals_for, team.stats?.gf),
    goals_against: firstNumber(team.goals_against, team.goalsAgainst, team.ga, team.stats?.goals_against, team.stats?.ga),
    pp_pct: firstNumber(team.pp_pct, team.power_play_pct, team.powerPlayPct, team.stats?.pp_pct),
    pk_pct: firstNumber(team.pk_pct, team.penalty_kill_pct, team.penaltyKillPct, team.stats?.pk_pct),
    home_wins: firstNumber(team.home_wins, team.homeWins, team.home?.wins, team.record?.home_wins),
    home_losses: firstNumber(team.home_losses, team.homeLosses, team.home?.losses, team.record?.home_losses),
    home_otl: firstNumber(team.home_otl, team.homeOtl, team.home?.otl, team.record?.home_otl),
    road_wins: firstNumber(team.road_wins, team.away_wins, team.roadWins, team.away?.wins, team.record?.road_wins),
    road_losses: firstNumber(team.road_losses, team.away_losses, team.roadLosses, team.away?.losses, team.record?.road_losses),
    road_otl: firstNumber(team.road_otl, team.away_otl, team.roadOtl, team.away?.otl, team.record?.road_otl),
  };
}
function findActiveTeam(rootState, allTeams, controlledTeamId) {
  const direct =
    rootState.team ||
    rootState.user_team ||
    rootState.selected_team ||
    rootState.active_team ||
    rootState.franchise_team ||
    null;

  if (direct && typeof direct === "object") return normalizeTeam(direct, 0);

  const id =
    controlledTeamId ||
    direct ||
    rootState.user_team_id ||
    rootState.controlled_team_id ||
    rootState.selected_team_id ||
    rootState.gm_team_id ||
    rootState.team_id;

  const found = allTeams.find((team) => isSameTeamIdentifier(id, team));
  if (found) return found;

  return allTeams[0] || normalizeTeam({ abbreviation: "CLB", name: "Club", city: "" }, 0);
}

function normalizeGames(rootState, allTeams) {
  const calendarExpanded = [];
  const nhlFull = Array.isArray(rootState.nhl_calendar_full) ? rootState.nhl_calendar_full : EMPTY_ARRAY;

  nhlFull.forEach((day) => {
    const iso = day?.iso || day?.date || day?.calendar_iso || null;
    const dayGames = Array.isArray(day?.games) ? day.games : EMPTY_ARRAY;

    dayGames.forEach((g) => {
      calendarExpanded.push({
        ...g,
        date: g?.date || g?.game_date || iso,
        home_team_id: g?.home_team_id || g?.home_id || g?.homeTeamId || g?.home,
        away_team_id: g?.away_team_id || g?.away_id || g?.awayTeamId || g?.away,
        home_team:
          g?.home_team ||
          g?.homeTeam || {
            id: g?.home_id || g?.home_team_id || g?.home,
            abbreviation: g?.home_abbr || g?.homeAbbr,
            name: g?.home_name || g?.homeName,
          },
        away_team:
          g?.away_team ||
          g?.awayTeam || {
            id: g?.away_id || g?.away_team_id || g?.away,
            abbreviation: g?.away_abbr || g?.awayAbbr,
            name: g?.away_name || g?.awayName,
          },
      });
    });
  });

  const upcomingExpanded = [];
  const upcomingBlocks = Array.isArray(rootState.schedule_upcoming) ? rootState.schedule_upcoming : EMPTY_ARRAY;

  upcomingBlocks.forEach((day) => {
    const iso = day?.iso || day?.date || day?.calendar_iso || null;
    const dayGames = Array.isArray(day?.games) ? day.games : EMPTY_ARRAY;

    dayGames.forEach((g) => {
      upcomingExpanded.push({
        ...g,
        date: g?.date || g?.game_date || iso,
        home_team_id: g?.home_team_id || g?.home_id || g?.homeTeamId || g?.home,
        away_team_id: g?.away_team_id || g?.away_id || g?.awayTeamId || g?.away,
      });
    });
  });

  const candidates = normalizeArrayMerged(
    rootState.games,
    rootState.schedule,
    rootState.calendar,
    rootState.league_schedule,
    rootState.leagueSchedule,
    rootState.game_results,
    rootState.results,
    rootState.completed_games,
    calendarExpanded,
    upcomingExpanded,
    rootState.season?.games,
    rootState.season?.schedule,
    rootState.franchise?.schedule
  );

  const uniqueRaw = [];
  const seen = new Set();

  candidates.forEach((g, index) => {
    const dt = toISODate(g?.date || g?.game_date || g?.gameDate || g?.day || g?.start_date || "");
    const hid = getLooseTeamIdentifier(g?.home_team_id || g?.home_id || g?.homeTeamId || g?.home || g?.home_team || g?.homeTeam);
    const aid = getLooseTeamIdentifier(g?.away_team_id || g?.away_id || g?.awayTeamId || g?.away || g?.away_team || g?.awayTeam);
    const gid = String(g?.id || g?.game_id || g?.gameId || "").trim();
    const time = String(g?.time || g?.start_time || g?.startTime || g?.puck_drop || "").trim();

    const key = gid || `${dt}|${hid}|${aid}|${time}|${index}`;

    if (seen.has(key)) return;
    seen.add(key);
    uniqueRaw.push(g);
  });

  return uniqueRaw
    .map((game, index) => normalizeGame(game, index, allTeams))
    .filter((game) => game.date);
}

function normalizeGame(game, index, allTeams) {
  if (!game || typeof game !== "object") {
    return {
      id: `game-${index}`,
      date: new Date(),
      homeId: null,
      awayId: null,
      homeTeam: null,
      awayTeam: null,
      time: "",
      status: "scheduled",
      completed: false,
      isFinal: false,
    };
  }

  const homeId =
    game.home_team_id ||
    game.homeTeamId ||
    game.home_id ||
    game.home ||
    game.home_team ||
    game.homeTeam ||
    game.home_abbr ||
    game.homeAbbr;

  const awayId =
    game.away_team_id ||
    game.awayTeamId ||
    game.away_id ||
    game.away ||
    game.away_team ||
    game.awayTeam ||
    game.away_abbr ||
    game.awayAbbr;

  const homeTeam =
    findTeamByAny(allTeams, homeId) ||
    normalizeTeam(game.home_team || game.homeTeam || game.home || homeId || "HOME", index);

  const awayTeam =
    findTeamByAny(allTeams, awayId) ||
    normalizeTeam(game.away_team || game.awayTeam || game.away || awayId || "AWAY", index);

  const homeScore = firstNumberOrNull(
    game.homeScore,
    game.home_score,
    game.home_goals,
    game.homeGoals,
    game.score?.home
  );

  const awayScore = firstNumberOrNull(
    game.awayScore,
    game.away_score,
    game.away_goals,
    game.awayGoals,
    game.score?.away
  );

  const explicitStatus = game.status || game.game_status || game.state || game.gameState || "";
  const normalizedStatus = String(explicitStatus || "").toLowerCase();

  const explicitFinalStatus = [
    "final",
    "completed",
    "complete",
    "played",
    "done",
    "simmed",
    "finished",
  ].includes(normalizedStatus);

  const explicitScheduledStatus = [
    "scheduled",
    "pregame",
    "preview",
    "upcoming",
    "not_started",
    "not-started",
    "ns",
  ].includes(normalizedStatus);

  const hasBothScores = homeScore !== null && awayScore !== null;
  const scoresAreTied = hasBothScores && Number(homeScore) === Number(awayScore);
  const zeroZero = hasBothScores && Number(homeScore) === 0 && Number(awayScore) === 0;

  const backendFinalFlag =
    game.completed === true ||
    game.is_final === true ||
    game.isFinal === true ||
    game.simmed === true ||
    game.played === true;

  // This is the important protection:
  // A game is final only when backend says final/completed/simmed AND scores are usable.
  // We do NOT let random placeholder 0-0 become a final.
  const isFinal =
    !explicitScheduledStatus &&
    hasBothScores &&
    !scoresAreTied &&
    (explicitFinalStatus || backendFinalFlag);

  const cleanHomeScore = isFinal ? Math.max(0, Math.round(Number(homeScore))) : null;
  const cleanAwayScore = isFinal ? Math.max(0, Math.round(Number(awayScore))) : null;

  return {
    ...game,
    id: game.id || game.game_id || game.gameId || `game-${index}`,
    date: game.date || game.game_date || game.gameDate || game.day || game.start_date || game.startDate,
    time: game.time || game.start_time || game.startTime || game.puck_drop || game.puckDrop || "",
    homeId: getTeamId(homeTeam),
    awayId: getTeamId(awayTeam),
    homeTeam,
    awayTeam,
    homeScore: cleanHomeScore,
    awayScore: cleanAwayScore,
    status: isFinal ? "final" : explicitStatus || "scheduled",
    completed: isFinal,
    isFinal,
    hasScore: isFinal,
    scoreRejectedAsPlaceholder: Boolean(hasBothScores && !isFinal && (zeroZero || scoresAreTied)),
  };
}
function normalizeStandings(rootState, allTeams) {
  const candidates = normalizeArrayMerged(
    rootState.standings,
    rootState.league_standings,
    rootState.leagueStandings,
    rootState.season?.standings
  );

  if (candidates.length) {
    return candidates.map((row, index) => {
      const team =
        findTeamByAny(
          allTeams,
          row.team_id ||
            row.teamId ||
            row.id ||
            row.team ||
            row.abbreviation ||
            row.abbr ||
            row.name
        ) || normalizeTeam(row.team && typeof row.team === "object" ? row.team : row, index);

      const wins = firstNumber(row.wins, row.w, row.record?.wins, team.wins);
      const losses = firstNumber(row.losses, row.l, row.record?.losses, team.losses);
      const otl = firstNumber(row.otl, row.ot, row.overtime_losses, row.record?.otl, team.otl);
      const gp = firstNumber(row.gp, row.games_played, row.gamesPlayed, wins + losses + otl);
      const pts = firstNumber(row.points, row.pts, wins * 2 + otl, team.points);
      const abbreviation =
        canonicalizeTeamAbbr(
          row.abbreviation || row.abbr || row.abbrev || row.team_abbrev || row.team_abbreviation || team.abbreviation || team.abbr
        ) ||
        lookupNhlAbbrFromLabel(team.name || row.name || row.team_name || row.team) ||
        getTeamAbbreviation(team);

      return {
        ...row,
        team,
        id: getTeamId(team),
        abbreviation,
        abbr: abbreviation,
        gp,
        w: wins,
        l: losses,
        otl,
        pts,
        rank: firstNumber(row.rank, row.league_rank, row.division_rank, index + 1),
        pointPct: gp ? pts / (gp * 2) : 0,
        division: row.division || team.division || "League",
        conference: row.conference || team.conference || "League",
        goals_for: firstNumberOrNull(row.goals_for, row.gf, row.team?.goals_for, row.team?.gf),
        goals_against: firstNumberOrNull(row.goals_against, row.ga, row.team?.goals_against, row.team?.ga),
        pp_pct: firstNumberOrNull(row.pp_pct, row.power_play_pct, row.team?.pp_pct, row.team?.power_play_pct),
        pk_pct: firstNumberOrNull(row.pk_pct, row.penalty_kill_pct, row.team?.pk_pct, row.team?.penalty_kill_pct),
      };
    });
  }

  return allTeams.map((team, index) => {
    const wins = firstNumber(team.wins, team.w);
    const losses = firstNumber(team.losses, team.l);
    const otl = firstNumber(team.otl, team.ot);
    const gp = firstNumber(team.gp, wins + losses + otl);
    const pts = firstNumber(team.points, team.pts, wins * 2 + otl);
    const abbreviation =
      canonicalizeTeamAbbr(team.abbreviation || team.abbr) ||
      lookupNhlAbbrFromLabel(team.name || team.team_name || team.full_name) ||
      getTeamAbbreviation(team);

    return {
      team,
      id: getTeamId(team),
      abbreviation,
      abbr: abbreviation,
      gp,
      w: wins,
      l: losses,
      otl,
      pts,
      rank: index + 1,
      pointPct: gp ? pts / (gp * 2) : 0,
      division: team.division || "League",
      conference: team.conference || "League",
      goals_for: firstNumberOrNull(team.goals_for, team.goalsFor, team.gf),
      goals_against: firstNumberOrNull(team.goals_against, team.goalsAgainst, team.ga),
      pp_pct: firstNumberOrNull(team.pp_pct, team.power_play_pct, team.powerPlayPct),
      pk_pct: firstNumberOrNull(team.pk_pct, team.penalty_kill_pct, team.penaltyKillPct),
    };
  });
}

function buildMonthGrid(viewDate) {
  const year = viewDate.getFullYear();
  const month = viewDate.getMonth();

  const firstOfMonth = new Date(year, month, 1);
  const startDay = firstOfMonth.getDay();
  const gridStart = new Date(year, month, 1 - startDay);

  const cells = [];

  for (let index = 0; index < 42; index += 1) {
    const date = new Date(gridStart);
    date.setDate(gridStart.getDate() + index);

    cells.push({
      date,
      outsideMonth: date.getMonth() !== month,
    });
  }

  return cells;
}

function findStatsCentralTeamRow(statsCentral, team) {
  if (!statsCentral || !team) return null;

  const direct = statsCentral.user_team_stats || statsCentral.userTeamStats || statsCentral.team_stats;
  if (direct && typeof direct === "object" && !Array.isArray(direct)) {
    if (isSameTeam(direct, team) || isSameTeamIdentifier(direct.team_id || direct.teamId, team)) {
      return direct;
    }
  }

  const pools = normalizeArrayMerged(
    statsCentral.league_team_stats,
    statsCentral.leagueTeamStats,
    statsCentral.teams,
    statsCentral.team_rows
  );

  return (
    pools.find(
      (row) =>
        isSameTeamIdentifier(row?.team_id || row?.teamId || row?.abbr || row?.name, team) ||
        isSameTeam(row, team)
    ) || null
  );
}

function completedTeamGamesUpTo(games, activeTeam, currentDate) {
  const currentTime = startOfDay(toDateObject(currentDate) || new Date()).getTime();

  return games.filter((game) => {
    if (!isTeamGame(game, activeTeam) || !isCompletedGame(game)) return false;

    const date = toDateObject(game.date);
    if (date && startOfDay(date).getTime() > currentTime) return false;

    return true;
  });
}

function normalizePctValue(value) {
  const n = firstNumberOrNull(value);
  if (n === null) return null;
  if (n > 0 && n <= 1) return n * 100;
  return n;
}

function isUsableSpecialTeamsPct(value, gp = 0) {
  const n = normalizePctValue(value);
  if (n === null) return false;
  if (!Number.isFinite(n)) return false;
  if (n <= 0) return false;
  if (gp < 0) return false;
  return true;
}

function resolveMetricWithSource(candidates) {
  for (const candidate of candidates) {
    const n = firstNumberOrNull(candidate.value);
    if (n === null) continue;
    if (candidate.requirePositive && n <= 0) continue;
    if (candidate.allowZeroIfGp && n === 0 && candidate.gp <= 0) continue;
    return { value: n, source: candidate.source };
  }
  return { value: null, source: "unavailable" };
}

function deriveSpecialTeamsFromGames(games, activeTeam, currentDate) {
  const completed = completedTeamGamesUpTo(games, activeTeam, currentDate);
  if (!completed.length) return { ppPct: null, pkPct: null, ppg: 0, ppo: 0, ppga: 0, oppPpo: 0 };

  let ppg = 0;
  let ppo = 0;
  let ppga = 0;
  let oppPpo = 0;

  completed.forEach((game) => {
    const home = isHomeGame(game, activeTeam);
    const pick = (...keys) => {
      for (const key of keys) {
        const n = firstNumberOrNull(game?.[key]);
        if (n !== null) return n;
      }
      return 0;
    };

    if (home) {
      ppg += pick("home_pp_goals", "homePpg", "home_ppg", "pp_goals");
      ppo += pick("home_ppo", "homePpo", "ppo");
      ppga += pick("home_ppga", "homePpga", "away_pp_goals");
      oppPpo += pick("home_opp_ppo", "homeOppPpo", "away_ppo");
    } else {
      ppg += pick("away_pp_goals", "awayPpg", "away_ppg", "pp_goals");
      ppo += pick("away_ppo", "awayPpo", "ppo");
      ppga += pick("away_ppga", "awayPpga", "home_pp_goals");
      oppPpo += pick("away_opp_ppo", "awayOppPpo", "home_ppo");
    }
  });

  return {
    ppPct: ppo > 0 ? (ppg / ppo) * 100 : null,
    pkPct: oppPpo > 0 ? (1 - ppga / oppPpo) * 100 : null,
    ppg,
    ppo,
    ppga,
    oppPpo,
  };
}

function buildTeamSeasonStatsBundle(activeTeam, standings, games, statsCentral, currentDate) {
  const row = findStandingForTeam(standings, activeTeam);
  const scRow = findStatsCentralTeamRow(statsCentral, activeTeam);
  const scRates = deriveMatchupTeamRates(statsCentral, activeTeam);
  const completed = completedTeamGamesUpTo(games, activeTeam, currentDate);
  const ledgerGF = completed.reduce((total, game) => total + getTeamScoreFromGame(game, activeTeam), 0);
  const ledgerGA = completed.reduce((total, game) => total + getOpponentScoreFromGame(game, activeTeam), 0);
  const gp = firstNumber(row?.gp, completed.length, activeTeam?.gp, 0);
  const derivedST = deriveSpecialTeamsFromGames(games, activeTeam, currentDate);

  const goalsFor = resolveMetricWithSource([
    { value: completed.length ? ledgerGF : null, source: "completed_games" },
    { value: row?.goals_for ?? row?.gf, source: "standings", allowZeroIfGp: true, gp },
    { value: scRow?.goals_for ?? scRow?.gf ?? scRates.gf, source: "stats_central", requirePositive: true },
    { value: activeTeam?.goals_for ?? activeTeam?.goalsFor ?? activeTeam?.gf, source: "team_object", requirePositive: true },
  ]);

  const goalsAgainst = resolveMetricWithSource([
    { value: completed.length ? ledgerGA : null, source: "completed_games" },
    { value: row?.goals_against ?? row?.ga, source: "standings", allowZeroIfGp: true, gp },
    { value: scRow?.goals_against ?? scRow?.ga ?? scRates.ga, source: "stats_central", requirePositive: true },
    { value: activeTeam?.goals_against ?? activeTeam?.goalsAgainst ?? activeTeam?.ga, source: "team_object", requirePositive: true },
  ]);

  const ppCandidates = [
    derivedST.ppPct,
    row?.pp_pct,
    row?.power_play_pct,
    scRow?.pp_pct,
    scRow?.power_play_pct,
    scRates.pp_pct > 0 ? scRates.pp_pct : null,
    activeTeam?.pp_pct,
    activeTeam?.power_play_pct,
  ];

  let ppPct = null;
  let ppSource = "unavailable";
  for (const candidate of ppCandidates) {
    if (!isUsableSpecialTeamsPct(candidate, gp)) continue;
    ppPct = normalizePctValue(candidate);
    ppSource =
      candidate === derivedST.ppPct
        ? "completed_games"
        : candidate === row?.pp_pct || candidate === row?.power_play_pct
          ? "standings"
          : candidate === scRow?.pp_pct || candidate === scRow?.power_play_pct || candidate === scRates.pp_pct
            ? "stats_central"
            : "team_object";
    break;
  }

  const pkCandidates = [
    derivedST.pkPct,
    row?.pk_pct,
    row?.penalty_kill_pct,
    scRow?.pk_pct,
    scRow?.penalty_kill_pct,
    scRates.pk_pct > 0 ? scRates.pk_pct : null,
    activeTeam?.pk_pct,
    activeTeam?.penalty_kill_pct,
  ];

  let pkPct = null;
  let pkSource = "unavailable";
  for (const candidate of pkCandidates) {
    if (!isUsableSpecialTeamsPct(candidate, gp)) continue;
    pkPct = normalizePctValue(candidate);
    pkSource =
      candidate === derivedST.pkPct
        ? "completed_games"
        : candidate === row?.pk_pct || candidate === row?.penalty_kill_pct
          ? "standings"
          : candidate === scRow?.pk_pct || candidate === scRow?.penalty_kill_pct || candidate === scRates.pk_pct
            ? "stats_central"
            : "team_object";
    break;
  }

  return {
    row,
    gp,
    completedCount: completed.length,
    goalsFor,
    goalsAgainst,
    ppPct,
    pkPct,
    sources: {
      goalsFor: goalsFor.source,
      goalsAgainst: goalsAgainst.source,
      ppPct: ppSource,
      pkPct: pkSource,
    },
  };
}

function logCalendarStatsDebug(bundle, activeTeam, standings) {
  if (process.env.NODE_ENV === "production") return;

  const row = findStandingForTeam(standings, activeTeam);
  if (!row && standings?.length) {
    console.warn("[CalendarScreen] activeTeam not matched in standings:", getTeamDisplayName(activeTeam));
  }

  if (bundle.completedCount > 0 && bundle.goalsFor.value === 0 && bundle.goalsAgainst.value === 0) {
    console.warn("[CalendarScreen] Completed games exist but GF/GA calculated to zero.", {
      team: getTeamDisplayName(activeTeam),
      completedCount: bundle.completedCount,
    });
  }

  console.debug("[CalendarScreen] quick stats sources", {
    team: getTeamDisplayName(activeTeam),
    gf: bundle.goalsFor,
    ga: bundle.goalsAgainst,
    pp: { value: bundle.ppPct, source: bundle.sources.ppPct },
    pk: { value: bundle.pkPct, source: bundle.sources.pkPct },
  });
}

function buildQuickTeamStats(activeTeam, standings, games, currentDate, statsCentral, allTeams = []) {
  const bundle = buildTeamSeasonStatsBundle(activeTeam, standings, games, statsCentral, currentDate);
  logCalendarStatsDebug(bundle, activeTeam, standings);

  const row = bundle.row;
  const wins = firstNumber(row?.w, activeTeam?.wins, activeTeam?.w);
  const losses = firstNumber(row?.l, activeTeam?.losses, activeTeam?.l);
  const otl = firstNumber(row?.otl, activeTeam?.otl, activeTeam?.ot);
  const gp = firstNumber(row?.gp, bundle.completedCount, wins + losses + otl);
  const pts = firstNumber(row?.pts, activeTeam?.points, activeTeam?.pts, wins * 2 + otl);

  const homeRecord = calculateHomeRoadRecord(games, activeTeam, "home", currentDate);
  const roadRecord = calculateHomeRoadRecord(games, activeTeam, "road", currentDate);

  const goalsFor = bundle.goalsFor.value;
  const goalsAgainst = bundle.goalsAgainst.value;
  const ppPct = bundle.ppPct;
  const pkPct = bundle.pkPct;

  const rankContext = { standings, games, statsCentral, currentDate, allTeams };

  const sourceLabel = (key) => {
    const source = bundle.sources[key];
    if (!source || source === "unavailable") return "Data unavailable";
    if (source === "completed_games") return "Calculated from completed games";
    if (source === "standings") return "Loaded from league standings";
    if (source === "stats_central") return "Loaded from stats central";
    if (source === "team_object") return "Loaded from team season object";
    return source;
  };

  return [
    {
      key: "record",
      icon: "◉",
      label: "Record",
      value: `${wins}-${losses}-${otl}`,
      sub: `${pts} pts${gp ? ` · ${gp} GP` : ""}`,
      tone: "cyan",
      tooltip: "Loaded from league standings",
    },
    {
      key: "home",
      icon: "⌂",
      label: "Home",
      value: homeRecord.record,
      sub: `${homeRecord.points} pts`,
      tone: "neutral",
      tooltip: "Calculated from completed home games",
    },
    {
      key: "road",
      icon: "⌖",
      label: "Road",
      value: roadRecord.record,
      sub: `${roadRecord.points} pts`,
      tone: "neutral",
      tooltip: "Calculated from completed road games",
    },
    {
      key: "goals_for",
      icon: "◎",
      label: "Goals For",
      value: goalsFor === null ? "—" : safeNumber(goalsFor, 0),
      sub: resolveLeagueRankLabel(activeTeam, "goals_for", rankContext) || null,
      tone: "green",
      tooltip: sourceLabel("goalsFor"),
    },
    {
      key: "goals_against",
      icon: "△",
      label: "Goals Against",
      value: goalsAgainst === null ? "—" : safeNumber(goalsAgainst, 0),
      sub: resolveLeagueRankLabel(activeTeam, "goals_against", { ...rankContext, lowerIsBetter: true }) || null,
      tone: "danger",
      tooltip: sourceLabel("goalsAgainst"),
    },
    {
      key: "power_play",
      icon: "↯",
      label: "Power Play",
      value: ppPct === null ? "—" : formatPercentLoose(ppPct),
      sub: resolveLeagueRankLabel(activeTeam, "pp_pct", rankContext) || null,
      tone: "gold",
      tooltip: sourceLabel("ppPct") === "Data unavailable"
        ? "Special teams % unavailable until PP opportunities are logged"
        : sourceLabel("ppPct"),
    },
    {
      key: "penalty_kill",
      icon: "▣",
      label: "Penalty Kill",
      value: pkPct === null ? "—" : formatPercentLoose(pkPct),
      sub: resolveLeagueRankLabel(activeTeam, "pk_pct", rankContext) || null,
      tone: "blue",
      tooltip: sourceLabel("pkPct") === "Data unavailable"
        ? "Special teams % unavailable until shorthanded opportunities are logged"
        : sourceLabel("pkPct"),
    },
  ];
}

function isVerifiedRecord(team, standings) {
  const row = findStandingForTeam(standings, team);
  const gp = firstNumber(row?.gp, 0);
  return gp > 0;
}

function getVerifiedTeamRecord(team, standings) {
  if (!isVerifiedRecord(team, standings)) return null;
  return formatTeamRecord(team, standings);
}

function isVerifiedGoalMetric(bundle, key) {
  const metric = key === "goalsFor" ? bundle.goalsFor : bundle.goalsAgainst;
  if (metric.value === null || metric.value === undefined) return false;
  return ["completed_games", "standings", "stats_central"].includes(metric.source);
}

function isVerifiedSpecialTeamsBundle(bundle) {
  return bundle.ppPct !== null && bundle.pkPct !== null;
}

function isVerifiedGoalieLabel(value) {
  const text = String(value || "").trim();
  if (!text) return false;
  return !/tbd|—/i.test(text);
}

function isVerifiedMeetingLabel(value) {
  const text = String(value || "").trim();
  return Boolean(text && text !== "No recent meeting");
}

function isVerifiedSeriesLabel(value) {
  const text = String(value || "").trim();
  return Boolean(text && text !== "—");
}

function buildGamePreview(
  game,
  activeTeam,
  allTeams,
  previousTeamGame,
  standings,
  statsCentral,
  options = {}
) {
  const { isUserTeamGame = true } = options;

  if (!game) {
    return {
      game: null,
      homeTeam: null,
      awayTeam: null,
      isUserTeamGame,
      previewLabel: "Calendar Day",
      awayRecord: null,
      homeRecord: null,
      lines: [],
      analysis: [],
      missingFields: [],
    };
  }

  const homeTeam = enrichTeamWithStandings(getHomeTeam(game, allTeams), standings);
  const awayTeam = enrichTeamWithStandings(getAwayTeam(game, allTeams), standings);
  const userInGame = isTeamGame(game, activeTeam);
  const userOpponent = userInGame
    ? enrichTeamWithStandings(getOpponentFromGame(game, activeTeam, allTeams), standings)
    : null;

  const previewSideTeam = userInGame ? activeTeam : homeTeam;
  const previewOpponent = userInGame ? userOpponent : awayTeam;

  if (process.env.NODE_ENV !== "production" && (!homeTeam || !awayTeam)) {
    console.warn("[CalendarScreen] GamePreviewCard missing home/away team data.", {
      gameId: game.id,
      home: game.homeId || game.home,
      away: game.awayId || game.away,
    });
  }

  const missingFields = [];
  if (!homeTeam?.abbreviation && !homeTeam?.name) missingFields.push("home_team");
  if (!awayTeam?.abbreviation && !awayTeam?.name) missingFields.push("away_team");

  const activeGoalie = userInGame
    ? getProjectedGoalie(game, activeTeam, "active")
    : getProjectedGoalie(game, homeTeam, "active");
  const opponentGoalie = userInGame
    ? getProjectedGoalie(game, userOpponent, "opponent")
    : getProjectedGoalie(game, awayTeam, "opponent");

  const lastMeeting =
    userInGame && previousTeamGame
      ? getLastMeetingLabel(previousTeamGame, activeTeam, userOpponent)
      : "No recent meeting";
  const series = userInGame
    ? getSeasonSeriesLabel(game, activeTeam, userOpponent)
    : getSeasonSeriesLabel(game, homeTeam, awayTeam);

  const activeBundle = buildTeamSeasonStatsBundle(previewSideTeam, standings, options.games || [], statsCentral, options.currentDate);
  const oppBundle = buildTeamSeasonStatsBundle(previewOpponent, standings, options.games || [], statsCentral, options.currentDate);

  const venue =
    game.venue ||
    game.arena ||
    game.location ||
    getArenaName(homeTeam) ||
    null;

  const lines = [];

  lines.push({
    label: `Goalie (${getTeamAbbreviation(previewSideTeam)})`,
    value: isVerifiedGoalieLabel(activeGoalie) ? activeGoalie : "Starter TBD",
  });

  lines.push({
    label: `Goalie (${getTeamAbbreviation(previewOpponent)})`,
    value: isVerifiedGoalieLabel(opponentGoalie) ? opponentGoalie : "Starter TBD",
  });

  if (isVerifiedMeetingLabel(lastMeeting)) {
    lines.push({ label: "Last Meeting", value: lastMeeting });
  }

  if (isVerifiedSeriesLabel(series)) {
    lines.push({ label: "Season Series", value: series });
  }

  const analysis = [];

  if (isVerifiedGoalMetric(activeBundle, "goalsFor") && isVerifiedGoalMetric(activeBundle, "goalsAgainst")) {
    analysis.push({
      label: `${getTeamAbbreviation(previewSideTeam)} GF / GA`,
      value: `${safeNumber(activeBundle.goalsFor.value, 0)} / ${safeNumber(activeBundle.goalsAgainst.value, 0)}`,
    });
  }

  if (isVerifiedGoalMetric(oppBundle, "goalsFor") && isVerifiedGoalMetric(oppBundle, "goalsAgainst")) {
    analysis.push({
      label: `${getTeamAbbreviation(previewOpponent)} GF / GA`,
      value: `${safeNumber(oppBundle.goalsFor.value, 0)} / ${safeNumber(oppBundle.goalsAgainst.value, 0)}`,
    });
  }

  if (isVerifiedSpecialTeamsBundle(activeBundle)) {
    analysis.push({
      label: `${getTeamAbbreviation(previewSideTeam)} PP / PK`,
      value: `${formatPercentLoose(activeBundle.ppPct)} / ${formatPercentLoose(activeBundle.pkPct)}`,
    });
  }

  if (isVerifiedSpecialTeamsBundle(oppBundle)) {
    analysis.push({
      label: `${getTeamAbbreviation(previewOpponent)} PP / PK`,
      value: `${formatPercentLoose(oppBundle.ppPct)} / ${formatPercentLoose(oppBundle.pkPct)}`,
    });
  }

  return {
    game,
    homeTeam,
    awayTeam,
    userOpponent,
    isUserTeamGame: isUserTeamGame && userInGame,
    previewLabel: isUserTeamGame && userInGame ? "Game Preview" : "League Game Preview",
    venue,
    missingFields,
    awayRecord: getVerifiedTeamRecord(awayTeam, standings),
    homeRecord: getVerifiedTeamRecord(homeTeam, standings),
    lines,
    analysis,
  };
}

function buildLeagueStandings(standings, allTeams) {
  const rows = standings
    .map((row) => ({
      ...row,
      team: resolveStandingsTeamRow(row, allTeams),
    }))
    .sort((a, b) => {
      if (b.pts !== a.pts) return b.pts - a.pts;
      if (b.pointPct !== a.pointPct) return b.pointPct - a.pointPct;
      if (b.w !== a.w) return b.w - a.w;
      return getTeamDisplayName(a.team).localeCompare(getTeamDisplayName(b.team));
    });

  return {
    label: "League Standings",
    rows: rows.map((row, index) => ({
      ...row,
      rank: index + 1,
    })),
  };
}

function buildDivisionStandings(standings, allTeams, activeTeam) {
  const division = getDivisionName(activeTeam);
  const filtered = (standings || []).filter((row) => {
    const team = resolveStandingsTeamRow(row, allTeams);
    const rowDiv = String(row.division || team?.division || "").trim();
    if (!division || division === "League") return true;
    return rowDiv && rowDiv.toLowerCase() === String(division).toLowerCase();
  });
  const scope = filtered.length ? filtered : standings;
  return buildLeagueStandings(scope, allTeams);
}

function deriveGoalsForAgainstFromStandings(standings, team) {
  if (!team || !Array.isArray(standings)) return { gf: 0, ga: 0 };

  const row = standings.find((r) => isSameTeam(r.team || r, team));
  if (!row) return { gf: 0, ga: 0 };

  return {
    gf: firstNumber(row.gf, row.goals_for, row.team?.gf, row.team?.goals_for),
    ga: firstNumber(row.ga, row.goals_against, row.team?.ga, row.team?.goals_against),
  };
}

function deriveMatchupTeamRates(statsCentral, team) {
  const rows = Array.isArray(statsCentral?.league_team_stats) ? statsCentral.league_team_stats : EMPTY_ARRAY;

  if (!rows.length || !team) return { gf: 0, ga: 0, pp_pct: 0, pk_pct: 0 };

  const row =
    rows.find(
      (r) =>
        isSameTeamIdentifier(r.team_id || r.teamId || r.abbr || r.abbreviation || r.name, team) ||
        isSameTeam(r, team)
    ) || null;

  if (!row) return { gf: 0, ga: 0, pp_pct: 0, pk_pct: 0 };

  return {
    gf: firstNumber(row.gf, row.goals_for),
    ga: firstNumber(row.ga, row.goals_against),
    pp_pct: firstNumberOrNull(row.pp_pct, row.power_play_pct) || 0,
    pk_pct: firstNumberOrNull(row.pk_pct, row.penalty_kill_pct) || 0,
  };
}

function buildScheduleDiagnostics(teamGames, activeTeam, currentDate, standings) {
  const sortedGames = [...teamGames].sort(sortGamesByDate);
  const backToBacks = countBackToBacks(sortedGames);
  const threeInFour = countThreeInFour(sortedGames);
  const longestRoadTripData = getLongestRoadTrip(sortedGames, activeTeam);
  const currentStreak = getCurrentStreak(sortedGames, activeTeam, currentDate);
  const nextTen = getNextGames(sortedGames, currentDate, 10);

  const divisionOpponents = nextTen.filter((game) => {
    const opponent = getOpponentFromGame(game, activeTeam, []);
    return getDivisionName(opponent) === getDivisionName(activeTeam);
  }).length;

  const topOpponentGames = nextTen.filter((game) => {
    const opponent = getOpponentFromGame(game, activeTeam, []);
    const row = findStandingForTeam(standings, opponent);
    return row && row.pointPct >= 0.6;
  }).length;

  return {
    backToBacks,
    backToBackRank: backToBacks ? "Monitor fatigue" : "Clean stretch",
    threeInFour,
    threeInFourRank: threeInFour ? "Travel risk" : "Stable spacing",
    longestRoadTrip: longestRoadTripData.length ? `${longestRoadTripData.length} games` : "0 games",
    longestRoadTripDates: longestRoadTripData.label || "No road trip loaded",
    currentStreak: currentStreak.label,
    currentStreakSub: currentStreak.sub,
    scheduleStrengthLabel:
      topOpponentGames >= 4 ? "Difficult" : topOpponentGames >= 2 ? "Balanced" : "Manageable",
    nextTenDivisionOpponents: divisionOpponents,
    nextTenTopOpponents: topOpponentGames,
  };
}

function sortGamesByDate(a, b) {
  const ad = startOfDay(toDateObject(a?.date) || new Date(0)).getTime();
  const bd = startOfDay(toDateObject(b?.date) || new Date(0)).getTime();

  if (ad !== bd) return ad - bd;

  return String(a?.time || "").localeCompare(String(b?.time || ""));
}

function buildCalendarInsights(diagnostics, activeTeam, standings, games, selectedDayEvents = EMPTY_ARRAY) {
  const insights = [];

  const criticalEvents = selectedDayEvents.filter((event) => String(event.priority || "").toUpperCase() === "CRITICAL");
  const highEvents = selectedDayEvents.filter((event) => String(event.priority || "").toUpperCase() === "HIGH");

  if (criticalEvents.length) {
    insights.push({
      icon: "!",
      title: `${criticalEvents.length} critical calendar event${criticalEvents.length === 1 ? "" : "s"} on selected date.`,
    });
  } else if (highEvents.length) {
    insights.push({
      icon: "★",
      title: `${highEvents.length} major league event${highEvents.length === 1 ? "" : "s"} attached to selected date.`,
    });
  }

  if (diagnostics.backToBacks > 0) {
    insights.push({
      icon: "!",
      title: `${diagnostics.backToBacks} back-to-back set${diagnostics.backToBacks === 1 ? "" : "s"} demand goalie rotation planning.`,
    });
  } else {
    insights.push({
      icon: "✓",
      title: "No loaded back-to-back sets in the visible team schedule.",
    });
  }

  if (diagnostics.nextTenDivisionOpponents > 0) {
    insights.push({
      icon: "+",
      title: `${diagnostics.nextTenDivisionOpponents} of the next 10 are divisional matchups.`,
    });
  }

  if (diagnostics.nextTenTopOpponents > 0) {
    insights.push({
      icon: "↯",
      title: `${diagnostics.nextTenTopOpponents} upcoming games are against high point-percentage teams.`,
    });
  }

  const row = findStandingForTeam(standings, activeTeam);

  if (row && row.pointPct < 0.45) {
    insights.push({
      icon: "⌁",
      title: "Current standings pace is below playoff range. Upcoming home games matter.",
    });
  } else if (row && row.pointPct >= 0.6) {
    insights.push({
      icon: "◎",
      title: "Strong standings pace. Manage fatigue without chasing every matchup.",
    });
  }

  if (!insights.length) {
    insights.push({
      icon: "◌",
      title: "Schedule profile is neutral based on currently loaded data.",
    });
  }

  return insights.slice(0, 3);
}

function buildLeagueStateRows(games, currentDate, activeTeam) {
  const today = toISODate(currentDate || new Date());

  return games
    .filter((game) => toISODate(game.date) === today)
    .sort((a, b) => {
      const aUser = isTeamGame(a, activeTeam);
      const bUser = isTeamGame(b, activeTeam);

      if (aUser && !bUser) return -1;
      if (!aUser && bUser) return 1;

      return String(a.time || "").localeCompare(String(b.time || ""));
    })
    .map((game) => ({
      id: game.id,
      away: getTeamAbbreviation(game.awayTeam),
      home: getTeamAbbreviation(game.homeTeam),
      time: isCompletedGame(game) ? getScoreLine(game) : normalizeGameTime(game.time),
      involvesUserTeam: isTeamGame(game, activeTeam),
    }));
}

function calculateHomeRoadRecord(games, activeTeam, type, currentDate) {
  const filtered = games.filter((game) => {
    if (!isTeamGame(game, activeTeam)) return false;
    if (!isCompletedGame(game)) return false;

    const date = toDateObject(game.date);
    const current = toDateObject(currentDate);

    if (date && current && startOfDay(date).getTime() > startOfDay(current).getTime()) return false;

    const home = isHomeGame(game, activeTeam);
    return type === "home" ? home : !home;
  });

  let wins = 0;
  let losses = 0;
  let otl = 0;

  filtered.forEach((game) => {
    const result = getGameResult(game, activeTeam);
    if (result === "W") wins += 1;
    if (result === "L") losses += 1;
    if (result === "OTL") otl += 1;
  });

  return {
    wins,
    losses,
    otl,
    points: wins * 2 + otl,
    record: `${wins}-${losses}-${otl}`,
  };
}

function calculateGoalsFor(games, activeTeam, currentDate = null) {
  return games.reduce((total, game) => {
    if (!isTeamGame(game, activeTeam) || !isCompletedGame(game)) return total;

    if (currentDate) {
      const date = toDateObject(game.date);
      const current = toDateObject(currentDate);
      if (date && current && startOfDay(date).getTime() > startOfDay(current).getTime()) return total;
    }

    return total + getTeamScoreFromGame(game, activeTeam);
  }, 0);
}

function calculateGoalsAgainst(games, activeTeam, currentDate = null) {
  return games.reduce((total, game) => {
    if (!isTeamGame(game, activeTeam) || !isCompletedGame(game)) return total;

    if (currentDate) {
      const date = toDateObject(game.date);
      const current = toDateObject(currentDate);
      if (date && current && startOfDay(date).getTime() > startOfDay(current).getTime()) return total;
    }

    return total + getOpponentScoreFromGame(game, activeTeam);
  }, 0);
}

function countBackToBacks(games) {
  let count = 0;

  for (let index = 1; index < games.length; index += 1) {
    const previous = toDateObject(games[index - 1].date);
    const current = toDateObject(games[index].date);
    if (!previous || !current) continue;

    const diff = Math.round((startOfDay(current).getTime() - startOfDay(previous).getTime()) / 86400000);
    if (diff === 1) count += 1;
  }

  return count;
}

function countThreeInFour(games) {
  let count = 0;

  for (let index = 0; index < games.length - 2; index += 1) {
    const first = toDateObject(games[index].date);
    const third = toDateObject(games[index + 2].date);
    if (!first || !third) continue;

    const diff = Math.round((startOfDay(third).getTime() - startOfDay(first).getTime()) / 86400000);
    if (diff <= 3) count += 1;
  }

  return count;
}

function getLongestRoadTrip(games, activeTeam) {
  let currentTrip = [];
  let bestTrip = [];

  games.forEach((game) => {
    if (!isHomeGame(game, activeTeam)) {
      currentTrip.push(game);

      if (currentTrip.length > bestTrip.length) {
        bestTrip = [...currentTrip];
      }
    } else {
      currentTrip = [];
    }
  });

  if (!bestTrip.length) {
    return {
      length: 0,
      label: "",
    };
  }

  const first = bestTrip[0];
  const last = bestTrip[bestTrip.length - 1];

  return {
    length: bestTrip.length,
    label: `${formatMonthDay(first.date)} - ${formatMonthDay(last.date)}`,
  };
}

function getCurrentStreak(games, activeTeam, currentDate) {
  const current = startOfDay(toDateObject(currentDate) || new Date()).getTime();

  const completed = games
    .filter((game) => {
      const date = toDateObject(game.date);
      return date && startOfDay(date).getTime() <= current && isCompletedGame(game);
    })
    .sort(sortGamesByDate)
    .reverse();

  if (!completed.length) {
    return {
      label: "—",
      sub: "No completed games",
    };
  }

  const firstResult = getGameResult(completed[0], activeTeam);

  if (!["W", "L", "OTL", "T"].includes(firstResult)) {
    return {
      label: "—",
      sub: "No valid streak data",
    };
  }

  let count = 0;

  for (const game of completed) {
    const result = getGameResult(game, activeTeam);
    if (result !== firstResult) break;
    count += 1;
  }

  const previous = completed[0];
  const opponent = getOpponentFromGame(previous, activeTeam, []);

  const resultWord =
    firstResult === "W"
      ? "win"
      : firstResult === "OTL"
        ? "OT loss"
        : firstResult === "T"
          ? "tie"
          : "loss";

  return {
    label: `${firstResult}${count}`,
    sub: `Last ${resultWord}: ${formatMonthDay(previous.date)} ${getTeamAbbreviation(opponent)}`,
  };
}

function getNextGames(games, currentDate, limit = 10) {
  const current = startOfDay(toDateObject(currentDate) || new Date()).getTime();

  return games
    .filter((game) => {
      const date = toDateObject(game.date);
      return date && startOfDay(date).getTime() >= current && !isCompletedGame(game);
    })
    .sort(sortGamesByDate)
    .slice(0, limit);
}

function getProjectedGoalie(game, team, type) {
  if (!game || !team) return "Projected goalie TBD";

  const teamId = getTeamId(team);
  const abbr = getTeamAbbreviation(team);

  const directKeys = [
    `${abbr}_goalie`,
    `${abbr?.toLowerCase()}_goalie`,
    `${teamId}_goalie`,
    type === "active" ? "active_goalie" : "opponent_goalie",
    type === "active" ? "user_goalie" : "opp_goalie",
  ];

  for (const key of directKeys) {
    if (game[key]) return formatGoalieValue(game[key]);
  }

  const isHome = isSameTeam(game.homeTeam, team);

  const goalie =
    (isHome
      ? game.home_goalie || game.homeGoalie || game.projected_home_goalie || game.projectedHomeGoalie
      : game.away_goalie || game.awayGoalie || game.projected_away_goalie || game.projectedAwayGoalie) || null;

  return goalie ? formatGoalieValue(goalie) : "Projected goalie TBD";
}

function formatGoalieValue(goalie) {
  if (!goalie) return "Projected goalie TBD";
  if (typeof goalie === "string") return goalie;

  const name = goalie.name || goalie.player_name || goalie.full_name || "Goalie";
  const savePct = firstNumberOrNull(goalie.save_pct, goalie.savePct, goalie.sv_pct, goalie.svPct);

  if (savePct !== null) return `${name} · ${formatSavePct(savePct)}`;
  return name;
}

function getLastMeetingLabel(previousTeamGame, activeTeam, opponent) {
  if (!previousTeamGame) return "No recent meeting";
  if (!opponent) return getScoreLine(previousTeamGame);

  const result = getGameResult(previousTeamGame, activeTeam);
  const score = getScoreLine(previousTeamGame);
  const date = formatMonthDay(previousTeamGame.date);

  return `${date} · ${result} ${score}`;
}

function getSeasonSeriesLabel(game, activeTeam, opponent) {
  if (!game) return "—";

  const userSeriesWins = firstNumberOrNull(
    game.user_series_wins,
    game.series?.user_wins,
    game.series?.active_wins,
    game.series_record?.wins
  );

  const oppSeriesWins = firstNumberOrNull(
    game.opponent_series_wins,
    game.series?.opponent_wins,
    game.series?.opp_wins,
    game.series_record?.losses
  );

  const ties = firstNumberOrNull(game.series_ties, game.series?.ties, game.series_record?.otl);

  if (userSeriesWins !== null || oppSeriesWins !== null || ties !== null) {
    return `${userSeriesWins || 0} - ${oppSeriesWins || 0} - ${ties || 0}`;
  }

  const activeAbbr = getTeamAbbreviation(activeTeam);
  const opponentAbbr = getTeamAbbreviation(opponent);

  if (game.season_series || game.seasonSeries) {
    return String(game.season_series || game.seasonSeries)
      .replaceAll("{USER}", activeAbbr)
      .replaceAll("{OPP}", opponentAbbr);
  }

  return "—";
}

function getTeamMetricForRanking(team, metricKey, standings, games, statsCentral, currentDate) {
  const bundle = buildTeamSeasonStatsBundle(team, standings, games, statsCentral, currentDate);

  if (metricKey === "goals_for") return bundle.goalsFor.value;
  if (metricKey === "goals_against") return bundle.goalsAgainst.value;
  if (metricKey === "pp_pct") return bundle.ppPct;
  if (metricKey === "pk_pct") return bundle.pkPct;

  return null;
}

function resolveLeagueRankLabel(activeTeam, metricKey, options = {}) {
  const { standings = [], statsCentral = {}, allTeams = [] } = options;

  const totalTeams = allTeams.length || standings.length || 32;
  const scRow = findStatsCentralTeamRow(statsCentral, activeTeam);

  const rankFieldByMetric = {
    goals_for: ["gf_league_rank", "goals_for_league_rank"],
    goals_against: ["ga_league_rank", "goals_against_league_rank"],
    pp_pct: ["pp_pct_league_rank", "pp_league_rank", "power_play_league_rank"],
    pk_pct: ["pk_pct_league_rank", "pk_league_rank", "penalty_kill_league_rank"],
  };

  for (const field of rankFieldByMetric[metricKey] || []) {
    const rank = firstNumberOrNull(scRow?.[field]);
    if (rank !== null && rank > 0) {
      return `League #${rank} of ${totalTeams}`;
    }
  }

  return "";
}

function rankLabel(activeTeam, standings, key, label = "NHL", lowerIsBetter = false) {
  return resolveLeagueRankLabel(activeTeam, key, { standings, lowerIsBetter });
}

function findStandingForTeam(standings, team) {
  if (!team || !standings?.length) return null;

  const direct = standings.find((row) => isSameTeam(row.team || row, team));
  if (direct) return direct;

  const abbr = getTeamAbbreviation(team).toLowerCase();
  if (abbr && abbr !== "TBD") {
    const byAbbr = standings.find((row) => {
      const candidate = row.team || row;
      return getTeamAbbreviation(candidate).toLowerCase() === abbr;
    });
    if (byAbbr) return byAbbr;
  }

  const teamId = getTeamId(team).toLowerCase();
  if (teamId) {
    return (
      standings.find((row) => {
        const candidate = row.team || row;
        return (
          String(row.team_id || row.teamId || row.id || "").toLowerCase() === teamId ||
          getTeamId(candidate).toLowerCase() === teamId
        );
      }) || null
    );
  }

  return null;
}

function findTeamByAny(teams, value) {
  if (!value) return null;

  if (typeof value === "object") {
    const id = getTeamId(value);
    return teams.find((team) => isSameTeamIdentifier(id, team) || isSameTeam(team, value)) || null;
  }

  return teams.find((team) => isSameTeamIdentifier(value, team)) || null;
}

function getHomeTeam(game, allTeams = []) {
  return (
    game?.homeTeam ||
    findTeamByAny(allTeams, game?.homeId || game?.home_team_id || game?.homeTeamId || game?.home) ||
    normalizeTeam(game?.home_team || game?.homeTeam || game?.home || "HOME")
  );
}

function getAwayTeam(game, allTeams = []) {
  return (
    game?.awayTeam ||
    findTeamByAny(allTeams, game?.awayId || game?.away_team_id || game?.awayTeamId || game?.away) ||
    normalizeTeam(game?.away_team || game?.awayTeam || game?.away || "AWAY")
  );
}

function getOpponentFromGame(game, activeTeam, allTeams = []) {
  if (!game || !activeTeam) return null;

  const home = getHomeTeam(game, allTeams);
  const away = getAwayTeam(game, allTeams);

  if (isSameTeam(home, activeTeam)) return away;
  if (isSameTeam(away, activeTeam)) return home;

  return home || away || null;
}

function isTeamGame(game, activeTeam) {
  if (!game || !activeTeam) return false;

  if (isSameTeam(game.homeTeam, activeTeam) || isSameTeam(game.awayTeam, activeTeam)) {
    return true;
  }

  const activeId = getTeamId(activeTeam);
  const homeRef = game.homeId || game.home_team_id || game.homeTeamId || game.home;
  const awayRef = game.awayId || game.away_team_id || game.awayTeamId || game.away;

  return (
    isSameTeamIdentifier(homeRef, activeTeam) ||
    isSameTeamIdentifier(awayRef, activeTeam) ||
    isSameTeamIdentifier(homeRef, activeId) ||
    isSameTeamIdentifier(awayRef, activeId)
  );
}

function isHomeGame(game, activeTeam) {
  if (!game || !activeTeam) return false;
  return isSameTeam(game.homeTeam, activeTeam);
}

function isCompletedGame(game) {
  if (!game) return false;

  if (game.completed === true || game.isFinal === true || game.is_final === true) {
    return hasValidFinalScore(game);
  }

  const status = String(game.status || game.game_status || game.state || "").toLowerCase();

  if (!["final", "completed", "complete", "played", "done", "simmed", "finished"].includes(status)) {
    return false;
  }

  return hasValidFinalScore(game);
}

function hasScore(game) {
  return hasValidFinalScore(game);
}

function hasValidFinalScore(game) {
  if (!game) return false;

  const home = firstNumberOrNull(
    game.homeScore,
    game.home_score,
    game.home_goals,
    game.homeGoals,
    game.score?.home
  );

  const away = firstNumberOrNull(
    game.awayScore,
    game.away_score,
    game.away_goals,
    game.awayGoals,
    game.score?.away
  );

  if (home === null || away === null) return false;

  const h = Number(home);
  const a = Number(away);

  if (!Number.isFinite(h) || !Number.isFinite(a)) return false;
  if (h < 0 || a < 0) return false;

  // Final hockey games cannot end tied.
  // This also kills fake 0-0 finals.
  if (h === a) return false;

  return true;
}
function getGameResult(game, activeTeam) {
  if (!game || !activeTeam || !isCompletedGame(game)) return "—";

  const teamScore = getTeamScoreFromGame(game, activeTeam);
  const oppScore = getOpponentScoreFromGame(game, activeTeam);

  if (teamScore === null || oppScore === null) return "—";

  if (teamScore > oppScore) return "W";

  const overtime =
    game.overtime ||
    game.went_ot ||
    game.wentOT ||
    game.ot ||
    String(game.result_type || game.resultType || "").toLowerCase().includes("ot") ||
    String(game.period_final || "").toLowerCase().includes("ot");

  if (teamScore < oppScore && overtime) return "OTL";
  if (teamScore < oppScore) return "L";

  return "—";
}

function getGameResultLabel(game, activeTeam) {
  if (!isCompletedGame(game)) return normalizeGameTime(game?.time || game?.start_time || game?.startTime);
  return `${getGameResult(game, activeTeam)} ${getScoreLine(game)}`;
}

function getTeamScoreFromGame(game, team) {
  if (!game || !team || !isCompletedGame(game)) return null;

  const isHome = isSameTeam(game.homeTeam, team);

  const score = isHome
    ? firstNumberOrNull(game.homeScore, game.home_score, game.home_goals, game.homeGoals, game.score?.home)
    : firstNumberOrNull(game.awayScore, game.away_score, game.away_goals, game.awayGoals, game.score?.away);

  return score === null ? null : Math.max(0, Math.round(Number(score)));
}

function getOpponentScoreFromGame(game, team) {
  if (!game || !team || !isCompletedGame(game)) return null;

  const isHome = isSameTeam(game.homeTeam, team);

  const score = isHome
    ? firstNumberOrNull(game.awayScore, game.away_score, game.away_goals, game.awayGoals, game.score?.away)
    : firstNumberOrNull(game.homeScore, game.home_score, game.home_goals, game.homeGoals, game.score?.home);

  return score === null ? null : Math.max(0, Math.round(Number(score)));
}

function getScoreLine(game) {
  if (!game || !isCompletedGame(game)) return "—";

  const away = firstNumberOrNull(
    game.awayScore,
    game.away_score,
    game.away_goals,
    game.awayGoals,
    game.score?.away
  );

  const home = firstNumberOrNull(
    game.homeScore,
    game.home_score,
    game.home_goals,
    game.homeGoals,
    game.score?.home
  );

  if (away === null || home === null) return "—";

  const a = Math.max(0, Math.round(Number(away)));
  const h = Math.max(0, Math.round(Number(home)));

  if (!Number.isFinite(a) || !Number.isFinite(h)) return "—";
  if (a === h) return "—";

  return `${a}-${h}`;
}
function getLooseTeamIdentifier(value) {
  if (!value) return "";

  if (typeof value === "object") {
    return (
      value.id ||
      value.team_id ||
      value.teamId ||
      value.abbreviation ||
      value.abbr ||
      value.short_name ||
      value.name ||
      value.full_name ||
      ""
    );
  }

  return String(value || "");
}

function getTeamId(team) {
  if (!team || typeof team !== "object") return String(team || "");

  return String(
    team.id ||
      team.team_id ||
      team.teamId ||
      team.abbreviation ||
      team.abbr ||
      team.short_name ||
      team.name ||
      ""
  );
}

function getTeamDisplayName(team) {
  if (!team) return "Club";
  if (typeof team === "string") return team;

  return (
    team.full_name ||
    team.fullName ||
    team.name ||
    team.team_name ||
    team.nickname ||
    team.abbreviation ||
    team.abbr ||
    "Club"
  );
}

function getTeamCity(team) {
  if (!team || typeof team !== "object") return "";
  return team.city || team.location || team.market || team.region || "";
}

function lookupNhlAbbrFromLabel(label) {
  const raw = String(label || "")
    .trim()
    .toLowerCase()
    .replace(/\./g, "")
    .replace(/\s+/g, " ");
  if (!raw) return "";
  if (NHL_NAME_TO_ABBR[raw]) return NHL_NAME_TO_ABBR[raw];

  // Disambiguate New York clubs — never collapse to a single abbr from "new york" alone.
  if (raw === "new york" || raw === "ny") return "";

  const keys = Object.keys(NHL_NAME_TO_ABBR).sort((a, b) => b.length - a.length);
  for (const key of keys) {
    if (key.length < 4) continue;
    if (raw.includes(key)) return NHL_NAME_TO_ABBR[key];
  }
  return "";
}

function canonicalizeTeamAbbr(raw) {
  const cleaned = String(raw || "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "");
  if (!cleaned) return "";
  if (TEAM_ABBR_LOGO_ALIASES[cleaned]) return TEAM_ABBR_LOGO_ALIASES[cleaned];
  if (KNOWN_NHL_ABBRS.has(cleaned)) return cleaned;
  // Ambiguous / broken truncations — never show these on the board.
  if (cleaned === "NEW" || cleaned === "NY") return "";
  return cleaned.length >= 2 && cleaned.length <= 3 ? cleaned : cleaned.slice(0, 3);
}

function getTeamAbbreviation(team) {
  if (!team) return "TBD";

  if (typeof team === "string") {
    const fromName = lookupNhlAbbrFromLabel(team);
    if (fromName) return fromName;
    const asAbbr = canonicalizeTeamAbbr(team);
    if (asAbbr && KNOWN_NHL_ABBRS.has(asAbbr)) return asAbbr;
    return asAbbr || "TBD";
  }

  const nameBits = [
    team.name,
    team.team_name,
    team.full_name,
    team.fullName,
    [team.city || team.location || team.market, team.nickname || team.mascot].filter(Boolean).join(" "),
    team.city || team.location || team.market,
  ]
    .map((v) => String(v || "").trim())
    .filter(Boolean);

  for (const bit of nameBits) {
    const hit = lookupNhlAbbrFromLabel(bit);
    if (hit) return hit;
  }

  const rawExplicit =
    team.abbreviation ||
    team.abbr ||
    team.team_abbrev ||
    team.team_abbreviation ||
    team.short_name ||
    team.shortName ||
    team.code ||
    "";

  const explicit = canonicalizeTeamAbbr(rawExplicit);
  if (explicit && KNOWN_NHL_ABBRS.has(explicit)) return explicit;
  // Allow non-NHL custom clubs with a clean 2–3 letter code, but never "NEW"/"CAL" truncations.
  if (explicit && explicit.length >= 2 && explicit.length <= 3 && !["NEW", "CAL", "ST"].includes(explicit)) {
    return explicit;
  }

  const rawFallback = team.id || team.team_id || "";
  const fromId = canonicalizeTeamAbbr(rawFallback);
  if (fromId && KNOWN_NHL_ABBRS.has(fromId)) return fromId;

  return "TBD";
}

function getDivisionName(team) {
  if (!team || typeof team !== "object") return "League";
  return team.division || team.division_name || team.div || "League";
}

function getConferenceName(team) {
  if (!team || typeof team !== "object") return "League";
  return team.conference || team.conference_name || team.conf || "League";
}

function getArenaName(team) {
  if (!team || typeof team !== "object") return "";
  return team.arena || team.venue || team.home_arena || team.homeArena || "";
}

function getTeamColorSeed(team) {
  const raw = getTeamAbbreviation(team);
  let hash = 0;

  for (let index = 0; index < raw.length; index += 1) {
    hash = raw.charCodeAt(index) + ((hash << 5) - hash);
  }

  const hue = Math.abs(hash) % 360;
  return `${hue} 78% 48%`;
}

function isSameTeam(a, b) {
  if (!a || !b) return false;

  const aId = getTeamId(a).toLowerCase();
  const bId = getTeamId(b).toLowerCase();
  const aAbbr = getTeamAbbreviation(a).toLowerCase();
  const bAbbr = getTeamAbbreviation(b).toLowerCase();
  const aName = getTeamDisplayName(a).toLowerCase();
  const bName = getTeamDisplayName(b).toLowerCase();

  return Boolean(
    (aId && bId && aId === bId) ||
      (aAbbr && bAbbr && aAbbr === bAbbr) ||
      (aName && bName && aName === bName)
  );
}

function isSameTeamIdentifier(identifier, team) {
  if (!identifier || !team) return false;

  const id = String(getLooseTeamIdentifier(identifier)).toLowerCase();
  const teamId = getTeamId(team).toLowerCase();
  const abbr = getTeamAbbreviation(team).toLowerCase();
  const name = getTeamDisplayName(team).toLowerCase();

  return id === teamId || id === abbr || id === name;
}

function getPlayerName(player) {
  if (!player) return "Player";
  if (typeof player === "string") return player;

  return (
    player.name ||
    player.player_name ||
    player.full_name ||
    player.fullName ||
    `${player.first_name || player.firstName || ""} ${player.last_name || player.lastName || ""}`.trim() ||
    "Player"
  );
}

function getPlayerPosition(player) {
  if (!player || typeof player !== "object") return "—";
  return player.position || player.pos || player.primary_position || player.primaryPosition || "—";
}

function getPlayerGoals(player) {
  return Math.max(
    0,
    firstNumber(
      player?.goals,
      player?.g,
      player?.stats?.goals,
      player?.season_stats?.goals,
      player?.seasonStats?.goals
    )
  );
}

function getPlayerAssists(player) {
  return Math.max(
    0,
    firstNumber(
      player?.assists,
      player?.a,
      player?.stats?.assists,
      player?.season_stats?.assists,
      player?.seasonStats?.assists
    )
  );
}

function getPlayerPoints(player) {
  const explicit = firstNumberOrNull(
    player?.points,
    player?.pts,
    player?.stats?.points,
    player?.season_stats?.points,
    player?.seasonStats?.points
  );

  if (explicit !== null) {
    return Math.max(0, Number(explicit));
  }

  return getPlayerGoals(player) + getPlayerAssists(player);
}
function isGoalieRow(player) {
  const pos = String(player?.position || player?.pos || "").toUpperCase();

  if (pos === "G" || pos === "GOALIE") return true;

  return Boolean(
    player?.is_goalie ||
      player?.stat_type === "goalie" ||
      player?.save_pct !== undefined ||
      player?.gaa !== undefined ||
      player?.shots_against !== undefined
  );
}

function getGoalieSavePct(player) {
  const raw = firstNumberOrNull(
    player?.save_pct,
    player?.savePct,
    player?.sv_pct,
    player?.stats?.save_pct,
    player?.season_stats?.save_pct
  );

  if (raw === null) return 0;

  const n = Number(raw);
  if (!Number.isFinite(n)) return 0;

  return n > 1 ? n / 100 : n;
}

function getGoalieGAA(player) {
  return firstNumber(
    player?.gaa,
    player?.goals_against_average,
    player?.stats?.gaa,
    player?.season_stats?.gaa
  );
}
function getPlayerGamesPlayed(player) {
  return Math.max(
    0,
    firstNumber(
      player?.gp,
      player?.games_played,
      player?.gamesPlayed,
      player?.stats?.gp,
      player?.season_stats?.gp,
      player?.seasonStats?.gp
    )
  );
}

function getPlayerPointsPerGame(player) {
  const gp = getPlayerGamesPlayed(player);
  if (!gp) return 0;
  return getPlayerPoints(player) / gp;
}

function getPotentialScore(player) {
  const raw = player?.potential_score || player?.potentialScore || player?.overall || player?.ovr || player?.rating;

  if (Number.isFinite(Number(raw))) return Number(raw);

  const grade = String(player?.potential || player?.ceiling || player?.grade || "").toUpperCase();

  if (grade.includes("A+")) return 98;
  if (grade.includes("A")) return 92;
  if (grade.includes("B+")) return 86;
  if (grade.includes("B")) return 80;
  if (grade.includes("C+")) return 74;
  if (grade.includes("C")) return 68;
  if (grade.includes("D")) return 60;

  return 0;
}

function formatPointPct(value, gp) {
  if (gp != null && Number(gp) === 0) return "—";

  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number <= 0) return ".000";

  return number.toFixed(3).replace(/^0/, "");
}

function formatSavePct(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return ".000";

  if (number > 1) {
    return (number / 100).toFixed(3).replace(/^0/, "");
  }

  return number.toFixed(3).replace(/^0/, "");
}

function formatPercentLoose(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  if (number <= 1 && number > 0) {
    return `${(number * 100).toFixed(1)}%`;
  }

  return `${number.toFixed(1)}%`;
}

function formatMoney(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  if (Math.abs(number) >= 1000000) {
    return `$${(number / 1000000).toFixed(1)}M`;
  }

  if (Math.abs(number) >= 1000) {
    return `$${(number / 1000).toFixed(1)}K`;
  }

  return `$${number.toLocaleString()}`;
}

function ordinal(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  const mod10 = number % 10;
  const mod100 = number % 100;

  if (mod10 === 1 && mod100 !== 11) return `${number}st`;
  if (mod10 === 2 && mod100 !== 12) return `${number}nd`;
  if (mod10 === 3 && mod100 !== 13) return `${number}rd`;

  return `${number}th`;
}

function toDateObject(value) {
  if (!value) return null;

  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value;
  }

  if (typeof value === "number") {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;

    const normalized = trimmed.includes("T") ? trimmed : `${trimmed}T00:00:00`;
    const date = new Date(normalized);

    return Number.isNaN(date.getTime()) ? null : date;
  }

  return null;
}

function toISODate(value) {
  const date = toDateObject(value);
  if (!date) return "";

  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");

  return `${year}-${month}-${day}`;
}

function startOfDay(date) {
  const next = new Date(date);
  next.setHours(0, 0, 0, 0);
  return next;
}

function formatShortDate(value) {
  const date = toDateObject(value);
  if (!date) return "Date TBD";

  return `${MONTH_NAMES[date.getMonth()].slice(0, 3)} ${date.getDate()}, ${date.getFullYear()}`;
}

function formatLongDate(value) {
  const date = toDateObject(value);
  if (!date) return "Date TBD";

  return `${LONG_WEEKDAY_NAMES[date.getDay()]}, ${MONTH_NAMES[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
}

function formatWeekday(value) {
  const date = toDateObject(value);
  if (!date) return "Day";

  return LONG_WEEKDAY_NAMES[date.getDay()];
}

function formatMonthDay(value) {
  const date = toDateObject(value);
  if (!date) return "TBD";

  return `${MONTH_NAMES[date.getMonth()].slice(0, 3)} ${date.getDate()}`;
}

function formatMonthYear(value) {
  const date = toDateObject(value);
  if (!date) return "Month";

  return `${MONTH_NAMES[date.getMonth()]} ${date.getFullYear()}`;
}

function normalizeGameTime(value) {
  if (!value) return "TBD";

  const raw = String(value).trim();

  if (/^\d{1,2}:\d{2}\s?(AM|PM)$/i.test(raw)) {
    return raw.toUpperCase().replace(/\s+/, " ");
  }

  if (/^\d{1,2}:\d{2}$/.test(raw)) {
    const [hourRaw, minute] = raw.split(":");
    let hour = Number(hourRaw);

    if (!Number.isFinite(hour)) return raw;

    const suffix = hour >= 12 ? "PM" : "AM";
    hour = hour % 12 || 12;

    return `${hour}:${minute} ${suffix}`;
  }

  return raw;
}

function formatTeamRecord(team, standings = []) {
  if (!team || typeof team !== "object") return "—";

  const row = findStandingForTeam(standings, team);
  const enriched = enrichTeamWithStandings(team, standings);

  const wins = firstNumber(row?.w, row?.wins, enriched.wins, enriched.w, enriched.record?.wins, enriched.record?.w);
  const losses = firstNumber(row?.l, row?.losses, enriched.losses, enriched.l, enriched.record?.losses, enriched.record?.l);
  const otl = firstNumber(
    row?.otl,
    row?.ot,
    enriched.otl,
    enriched.ot,
    enriched.overtime_losses,
    enriched.record?.otl,
    enriched.record?.ot
  );
  const gp = firstNumber(row?.gp, enriched.gp, wins + losses + otl);

  if (gp <= 0 && wins + losses + otl <= 0) return "—";

  return `${wins}-${losses}-${otl}`;
}
function CalendarStyles() {
  return (
    <style>{`
      .nhlcal-root {
        --bg: #04101a;
        --bg-2: #061522;
        --panel: rgba(9, 25, 38, 0.94);
        --panel-2: rgba(12, 35, 52, 0.94);
        --panel-3: rgba(15, 46, 66, 0.78);
        --line: rgba(156, 218, 236, 0.14);
        --line-2: rgba(115, 229, 241, 0.25);
        --line-strong: rgba(73, 231, 240, 0.5);
        --text: #e9f7fb;
        --muted: #8096a8;
        --muted-2: #607789;
        --cyan: #13d8e7;
        --cyan-soft: rgba(19, 216, 231, 0.13);
        --gold: #e9a83c;
        --gold-soft: rgba(233, 168, 60, 0.14);
        --green: #52df94;
        --green-soft: rgba(82, 223, 148, 0.13);
        --red: #ff606d;
        --red-soft: rgba(255, 96, 109, 0.13);
        --blue: #8ab4ff;
        --blue-soft: rgba(138, 180, 255, 0.13);
        --purple: #c992ff;
        --purple-soft: rgba(201, 146, 255, 0.14);
        --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);

        min-height: 100vh;
        height: 100vh;
        width: 100%;
        background:
          radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
          radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
          linear-gradient(180deg, #06131f 0%, #020a11 100%);
        color: var(--text);
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr);
        overflow: hidden;
        font-family:
          Inter,
          ui-sans-serif,
          system-ui,
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          sans-serif;
      }

      .nhlcal-root *,
      .nhlcal-root *::before,
      .nhlcal-root *::after {
        box-sizing: border-box;
      }

      .nhlcal-root button {
        font-family: inherit;
      }

      .nhlcal-sidebar {
        min-height: 100vh;
        background:
          linear-gradient(180deg, rgba(5, 16, 26, 0.98), rgba(3, 10, 17, 0.98)),
          radial-gradient(circle at 100% 14%, rgba(19, 216, 231, 0.14), transparent 34%);
        border-right: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        align-items: stretch;
        position: relative;
        z-index: 4;
      }

      .nhlcal-brand-button {
        height: 112px;
        border: 0;
        background: transparent;
        color: var(--text);
        display: grid;
        place-items: center;
        border-bottom: 1px solid var(--line);
        cursor: pointer;
      }

      .nhlcal-shield-icon {
        width: 30px;
        height: 34px;
        border: 2px solid rgba(223, 245, 250, 0.52);
        display: grid;
        place-items: center;
        color: rgba(223, 245, 250, 0.75);
        clip-path: polygon(50% 0, 92% 16%, 92% 72%, 50% 100%, 8% 72%, 8% 16%);
        font-size: 15px;
      }

      .nhlcal-side-nav {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 18px 0;
      }

      .nhlcal-side-button {
        width: 100%;
        min-height: 66px;
        border: 0;
        background: transparent;
        color: var(--muted);
        display: grid;
        place-items: center;
        gap: 4px;
        cursor: pointer;
        position: relative;
        transition:
          color 0.2s ease,
          background 0.2s ease,
          transform 0.2s ease;
      }

      .nhlcal-side-button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.035);
      }

      .nhlcal-side-button.is-active {
        color: var(--cyan);
        background:
          linear-gradient(90deg, rgba(19, 216, 231, 0.17), rgba(19, 216, 231, 0.03)),
          radial-gradient(circle at 100% 50%, rgba(19, 216, 231, 0.24), transparent 52%);
      }

      .nhlcal-side-button.is-active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 12px;
        bottom: 12px;
        width: 3px;
        border-radius: 999px;
        background: var(--cyan);
        box-shadow: 0 0 22px rgba(19, 216, 231, 0.8);
      }

      .nhlcal-side-icon {
        font-size: 22px;
        line-height: 1;
      }

      .nhlcal-side-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.02em;
      }

      .nhlcal-side-button em {
        position: absolute;
        right: 16px;
        top: 12px;
        min-width: 18px;
        height: 18px;
        border-radius: 999px;
        background: var(--cyan);
        color: #021016;
        font-size: 10px;
        display: grid;
        place-items: center;
        font-style: normal;
        font-weight: 900;
      }

      .nhlcal-settings-button {
        margin-top: auto;
        height: 88px;
        border: 0;
        border-top: 1px solid var(--line);
        background: transparent;
        color: var(--muted);
        display: grid;
        place-items: center;
        gap: 4px;
        cursor: pointer;
      }

      .nhlcal-settings-button span {
        font-size: 22px;
      }

      .nhlcal-settings-button small {
        font-size: 10px;
        font-weight: 800;
      }

      .nhlcal-main {
        min-width: 0;
        height: 100vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        padding: 10px 14px 10px;
      }

      .nhlcal-main::-webkit-scrollbar {
        width: 10px;
      }

      .nhlcal-main::-webkit-scrollbar-track {
        background: rgba(4, 16, 26, 0.72);
        border-radius: 999px;
      }

      .nhlcal-main::-webkit-scrollbar-thumb {
        background: rgba(110, 173, 191, 0.25);
        border-radius: 999px;
        border: 2px solid rgba(4, 16, 26, 0.72);
      }

      .nhlcal-main::-webkit-scrollbar-thumb:hover {
        background: rgba(19, 216, 231, 0.38);
      }

      .nhlcal-scroll-surface {
        scrollbar-width: thin;
        scrollbar-color: rgba(110, 173, 191, 0.35) rgba(4, 16, 26, 0.72);
      }

      .nhlcal-scroll-surface::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-track {
        background: rgba(4, 16, 26, 0.72);
        border-radius: 999px;
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(19, 216, 231, 0.34), rgba(110, 173, 191, 0.28));
        border-radius: 999px;
        border: 2px solid rgba(4, 16, 26, 0.72);
      }

      .nhlcal-scroll-surface::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(19, 216, 231, 0.52), rgba(110, 173, 191, 0.42));
      }

      .nhlcal-topbar {
        min-height: 56px;
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: minmax(180px, 0.9fr) minmax(240px, 1fr) minmax(260px, 0.95fr);
        align-items: center;
        gap: 10px;
      }

      .nhlcal-team-identity {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }

      .nhlcal-team-city {
        margin: 0 0 1px;
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .nhlcal-team-identity h1 {
        margin: 0;
        font-size: clamp(20px, 2vw, 30px);
        line-height: 0.95;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text);
        text-shadow: 0 0 24px rgba(19, 216, 231, 0.12);
      }

      .nhlcal-month-control {
        text-align: center;
        min-width: 0;
      }

      .nhlcal-month-control p {
        margin: 0 0 7px;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 0.36em;
        font-size: 12px;
        font-weight: 900;
      }

      .nhlcal-month-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        flex-wrap: nowrap;
      }

      .nhlcal-month-row h2 {
        margin: 0;
        font-size: clamp(18px, 1.8vw, 26px);
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
      }

      .nhlcal-month-row h2::first-letter {
        color: var(--cyan);
      }

      .nhlcal-month-row button {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          background 0.2s ease,
          transform 0.2s ease;
      }

      .nhlcal-month-row button:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.12);
        transform: translateY(-1px);
      }

      .nhlcal-action-cluster {
        justify-self: end;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        min-width: 0;
      }

      .nhlcal-action-primary,
      .nhlcal-action-secondary {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }

      .nhlcal-today-chip {
        height: 32px;
        min-width: 60px;
        border: 1px solid rgba(233, 168, 60, 0.35);
        border-radius: 8px;
        background: rgba(233, 168, 60, 0.08);
        color: #ffd88d;
        padding: 0 10px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        align-self: center;
      }

      .nhlcal-today-chip:hover {
        border-color: rgba(233, 168, 60, 0.55);
        background: rgba(233, 168, 60, 0.14);
      }

      .nhlcal-date-chip.compact {
        min-width: 0;
        height: auto;
        border-left: 0;
        padding-left: 0;
      }

      .nhlcal-advance-error-banner {
        flex: 0 0 auto;
        margin-top: 6px;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 96, 109, 0.28);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
        font-size: 11px;
        font-weight: 700;
      }

      .nhlcal-advance-error-banner.is-blocked {
        border-color: rgba(233, 168, 60, 0.32);
        background: rgba(233, 168, 60, 0.1);
        color: #ffd88d;
      }

      .nhlcal-schedule-alert {
        flex: 0 0 auto;
        margin-top: 8px;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 96, 109, 0.28);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
        font-size: 12px;
        font-weight: 700;
      }

      .nhlcal-menu-toggle {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(12, 31, 47, 0.72);
        display: grid;
        place-items: center;
        gap: 4px;
        padding: 12px;
        cursor: pointer;
      }

      .nhlcal-menu-toggle span {
        display: block;
        width: 19px;
        height: 2px;
        background: var(--text);
        border-radius: 999px;
      }

      .nhlcal-quick-link {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        padding: 9px 14px;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          transform 0.2s ease,
          background 0.2s ease;
      }

      .nhlcal-quick-link:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.12);
        transform: translateY(-1px);
      }

      .nhlcal-online-chip {
        display: grid;
        gap: 3px;
        padding-right: 6px;
      }

      .nhlcal-online-chip strong {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-online-chip span {
        color: #56dc75;
        font-size: 10px;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 0.1em;
      }

      .nhlcal-date-chip {
        min-width: 158px;
        height: 58px;
        border-left: 1px solid var(--line);
        padding-left: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .nhlcal-date-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: rgba(136, 180, 255, 0.12);
        border: 1px solid rgba(136, 180, 255, 0.14);
        color: #b8ceff;
      }

      .nhlcal-date-chip strong,
      .nhlcal-date-chip span {
        display: block;
      }

      .nhlcal-date-chip strong {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-date-chip span {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-wjc-hub-button {
        min-height: 44px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 14px;
        border: 1px solid rgba(0, 216, 223, 0.28);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.14), rgba(255, 255, 255, 0.03)),
          rgba(7, 22, 35, 0.88);
        color: #dffcff;
        cursor: pointer;
        padding: 0 14px;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition: 150ms ease;
      }

      .nhlcal-wjc-hub-button:hover {
        border-color: rgba(232, 165, 54, 0.38);
        transform: translateY(-1px);
      }

      .nhlcal-wjc-hub-button.is-modal {
        width: 100%;
        justify-content: center;
        min-height: 42px;
        margin-top: 12px;
      }

      .nhlcal-event-modal-actions {
        margin-top: 4px;
      }

      .nhlcal-wjc-menu-host {
        position: fixed;
        inset: 0;
        z-index: 12050;
        background: #08090c;
        overflow: hidden;
      }

      .nhlcal-wjc-menu-host .wjc-event-shell,
      .nhlcal-wjc-menu-host .wjc-stage-root {
        width: 100%;
        height: 100%;
        min-height: 100vh;
        min-height: 100dvh;
      }

      .nhlcal-advance-button {
        height: 44px;
        min-width: 132px;
        border: 0;
        border-radius: 7px;
        background:
          linear-gradient(180deg, #f4bd52, #d99023),
          radial-gradient(circle at 20% 0%, rgba(255, 255, 255, 0.5), transparent 30%);
        color: #1b1002;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
        box-shadow:
          0 14px 36px rgba(217, 144, 35, 0.28),
          inset 0 1px 0 rgba(255, 255, 255, 0.35);
        transition:
          transform 0.2s ease,
          filter 0.2s ease;
      }

      .nhlcal-advance-button-secondary {
        min-width: 92px;
        background:
          linear-gradient(180deg, rgba(19, 216, 231, 0.22), rgba(19, 216, 231, 0.08)),
          rgba(7, 22, 35, 0.92);
        color: #dffcff;
        border: 1px solid rgba(19, 216, 231, 0.32);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
      }

      .nhlcal-advance-button-secondary:hover:not(:disabled) {
        border-color: rgba(19, 216, 231, 0.52);
        filter: brightness(1.06);
      }

      .nhlcal-advance-button:disabled {
        cursor: not-allowed;
        opacity: 0.72;
        transform: none;
      }

      .nhlcal-advance-button.is-busy {
        filter: saturate(0.75) brightness(0.9);
      }

      .nhlcal-advance-alert {
        margin: 0 0 18px;
        border: 1px solid var(--line2);
        background: rgba(8, 23, 35, 0.92);
        border-radius: 16px;
        padding: 16px 18px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 18px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
      }

      .nhlcal-advance-alert strong {
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 12px;
        margin-bottom: 6px;
      }

      .nhlcal-advance-alert p {
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
      }

      .nhlcal-advance-alert ul {
        margin: 10px 0 0;
        padding-left: 18px;
        color: var(--text);
      }

      .nhlcal-advance-alert li {
        margin: 4px 0;
        color: rgba(232, 244, 251, 0.88);
      }

      .nhlcal-advance-alert button {
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: rgba(255, 255, 255, 0.08);
        color: var(--text);
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        cursor: pointer;
        white-space: nowrap;
      }

      .nhlcal-advance-alert.is-blocked {
        border-color: rgba(244, 189, 82, 0.42);
        background:
          radial-gradient(circle at 0% 0%, rgba(244, 189, 82, 0.16), transparent 38%),
          rgba(8, 23, 35, 0.94);
      }

      .nhlcal-advance-alert.is-blocked strong {
        color: #f4bd52;
      }

      .nhlcal-advance-alert.is-error {
        border-color: rgba(255, 100, 100, 0.42);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 100, 100, 0.16), transparent 38%),
          rgba(8, 23, 35, 0.94);
      }

      .nhlcal-advance-alert.is-error strong {
        color: #ff6464;
      }

      .nhlcal-advance-button:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
      }

      .nhlcal-advance-button span {
        margin-right: 10px;
      }

      .nhlcal-stat-strip {
        flex: 0 0 auto;
        margin-top: 6px;
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 0;
        border: 1px solid var(--line);
        background: rgba(8, 23, 35, 0.86);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow);
      }

      .nhlcal-stat-pill {
        min-height: 48px;
        padding: 6px 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
        background:
          linear-gradient(180deg, rgba(18, 42, 61, 0.45), rgba(6, 20, 31, 0.34)),
          radial-gradient(circle at 100% 0%, rgba(19, 216, 231, 0.05), transparent 52%);
      }

      .nhlcal-stat-pill:last-child {
        border-right: 0;
      }

      .nhlcal-stat-icon {
        width: 28px;
        height: 28px;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: rgba(148, 185, 205, 0.12);
        border: 1px solid rgba(148, 185, 205, 0.12);
        color: rgba(233, 247, 251, 0.8);
        font-size: 13px;
      }

      .nhlcal-stat-pill span,
      .nhlcal-stat-pill small {
        display: block;
      }

      .nhlcal-stat-pill span {
        color: var(--muted);
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 1000;
      }

      .nhlcal-stat-pill strong {
        display: block;
        margin-top: 2px;
        color: var(--text);
        font-size: 16px;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.02em;
        min-height: 16px;
      }

      .nhlcal-stat-pill strong.is-missing {
        color: rgba(233, 247, 251, 0.45);
      }

      .nhlcal-stat-pill.is-skeleton .nhlcal-stat-icon,
      .nhlcal-skeleton-bar {
        background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.12), rgba(255,255,255,0.05));
        background-size: 200% 100%;
        animation: nhlcal-skeleton 1.2s ease infinite;
        border-radius: 8px;
      }

      .nhlcal-stat-pill.is-skeleton .nhlcal-stat-icon {
        width: 34px;
        height: 34px;
      }

      .nhlcal-skeleton-bar {
        display: block;
        height: 18px;
        margin-top: 4px;
      }

      .nhlcal-skeleton-bar.short {
        height: 10px;
        width: 70%;
        margin-top: 8px;
      }

      @keyframes nhlcal-skeleton {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
      }

      .nhlcal-stat-pill small {
        margin-top: 4px;
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlcal-stat-pill.tone-cyan .nhlcal-stat-icon {
        color: var(--cyan);
        background: var(--cyan-soft);
      }

      .nhlcal-stat-pill.tone-green .nhlcal-stat-icon {
        color: var(--green);
        background: var(--green-soft);
      }

      .nhlcal-stat-pill.tone-danger .nhlcal-stat-icon {
        color: var(--red);
        background: var(--red-soft);
      }

      .nhlcal-stat-pill.tone-gold .nhlcal-stat-icon {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .nhlcal-stat-pill.tone-blue .nhlcal-stat-icon {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .nhlcal-content-grid {
        margin-top: 6px;
        flex: 1 1 auto;
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(272px, 318px);
        gap: 8px;
        overflow: hidden;
      }

      .nhlcal-calendar-panel {
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        border: 1px solid var(--line);
        border-radius: 12px;
        background:
          linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
          radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
        overflow: hidden;
        box-shadow: var(--shadow);
      }

      .nhlcal-calendar-toolbar {
        flex: 0 0 auto;
        padding: 8px 10px;
        border-bottom: 1px solid var(--line);
        background: rgba(5, 17, 27, 0.72);
      }

      .nhlcal-calendar-toolbar-group {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlcal-calendar-toolbar-group button {
        height: 30px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(14, 35, 50, 0.9);
        color: rgba(233, 247, 251, 0.82);
        padding: 0 10px;
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        cursor: pointer;
      }

      .nhlcal-calendar-toolbar-group button.is-active,
      .nhlcal-calendar-toolbar-group button:hover {
        border-color: var(--line-strong);
        color: var(--text);
        background: rgba(19, 216, 231, 0.11);
      }

      .nhlcal-week-header {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        height: 34px;
        border-bottom: 1px solid var(--line);
        background: rgba(5, 17, 27, 0.62);
        border-radius: 12px 12px 0 0;
        overflow: hidden;
      }

      .nhlcal-week-header div {
        display: grid;
        place-items: center;
        color: rgba(233, 247, 251, 0.88);
        text-transform: uppercase;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.12em;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
      }

      .nhlcal-week-header div:last-child {
        border-right: 0;
      }

      .nhlcal-month-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
      }

      .nhlcal-month-grid.nhlcal-scroll-surface {
        scrollbar-gutter: stable;
      }

      .nhlcal-day-cell {
        position: relative;
        min-height: 104px;
        border: 0;
        border-right: 1px solid rgba(156, 218, 236, 0.11);
        border-bottom: 1px solid rgba(156, 218, 236, 0.11);
        background:
          linear-gradient(180deg, rgba(11, 31, 45, 0.72), rgba(7, 22, 34, 0.72));
        color: var(--text);
        text-align: left;
        padding: 10px 11px;
        overflow: visible;
        cursor: pointer;
        transition:
          background 0.2s ease,
          box-shadow 0.2s ease,
          border-color 0.2s ease,
          transform 0.2s ease;
      }

      .nhlcal-month-grid.is-dense .nhlcal-day-cell {
        min-height: 96px;
        padding: 7px;
      }

      .nhlcal-day-cell.is-selected {
        box-shadow:
          inset 0 0 0 2px rgba(19, 216, 231, 0.95),
          0 0 24px rgba(19, 216, 231, 0.28);
        z-index: 5;
      }

      .nhlcal-day-cell.is-today:not(.is-selected)::after {
        content: "";
        position: absolute;
        inset: 5px;
        border: 1px dashed rgba(233, 168, 60, 0.42);
        border-radius: 8px;
        pointer-events: none;
      }

      .nhlcal-empty-day-line {
        color: rgba(128, 150, 168, 0.42);
        font-size: 12px;
        text-align: center;
        padding-top: 8px;
      }

      .nhlcal-day-cell:nth-child(7n) {
        border-right: 0;
      }

      .nhlcal-day-cell:nth-last-child(-n + 7) {
        border-bottom: 0;
      }

      .nhlcal-day-cell:hover {
        background:
          linear-gradient(180deg, rgba(16, 44, 62, 0.84), rgba(7, 25, 38, 0.82));
        z-index: 4;
      }

      .nhlcal-day-cell.is-muted {
        color: rgba(233, 247, 251, 0.38);
        background:
          linear-gradient(180deg, rgba(8, 18, 28, 0.72), rgba(5, 12, 19, 0.72));
      }

      .nhlcal-day-cell.has-team-game {
        background:
          radial-gradient(circle at 50% 50%, rgba(19, 216, 231, 0.11), transparent 72%),
          linear-gradient(180deg, rgba(9, 37, 51, 0.86), rgba(5, 22, 33, 0.86));
      }

      .nhlcal-day-cell.has-special-events {
        background:
          radial-gradient(circle at 12% 6%, rgba(233, 168, 60, 0.12), transparent 34%),
          linear-gradient(180deg, rgba(13, 34, 48, 0.82), rgba(6, 22, 34, 0.82));
      }

      .nhlcal-day-cell.has-critical-event {
        background:
          radial-gradient(circle at 12% 6%, rgba(255, 96, 109, 0.18), transparent 36%),
          radial-gradient(circle at 88% 12%, rgba(233, 168, 60, 0.13), transparent 34%),
          linear-gradient(180deg, rgba(38, 18, 29, 0.86), rgba(8, 22, 34, 0.86));
      }

      .nhlcal-day-cell.has-high-event:not(.has-critical-event) {
        background:
          radial-gradient(circle at 12% 6%, rgba(233, 168, 60, 0.17), transparent 36%),
          linear-gradient(180deg, rgba(31, 35, 37, 0.86), rgba(7, 22, 34, 0.86));
      }

      .nhlcal-day-number-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 22px;
        position: relative;
        z-index: 2;
      }

      .nhlcal-day-number {
        font-size: 14px;
        font-weight: 1000;
        color: rgba(233, 247, 251, 0.9);
      }

      .nhlcal-day-marker-row {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        gap: 5px;
      }

      .nhlcal-event-corner-badge {
        min-width: 22px;
        height: 22px;
        padding: 0 5px;
        display: inline-grid;
        place-items: center;
        border-radius: 999px;
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.2), transparent 36%),
          linear-gradient(180deg, rgba(233, 168, 60, 0.95), rgba(154, 87, 28, 0.95));
        border: 1px solid rgba(255, 214, 135, 0.45);
        color: #1b1002;
        font-size: 11px;
        font-weight: 1000;
        box-shadow: 0 0 16px rgba(233, 168, 60, 0.26);
      }
.nhlcal-event-corner-badge img {
  width: 18px;
  height: 18px;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.18));
}

.nhlcal-special-event-tile.has-logo {
  grid-template-columns: 34px minmax(0, 1fr);
}

.nhlcal-special-event-icon img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.16));
}

.nhlcal-special-event-tile.has-logo .nhlcal-special-event-icon {
  width: 34px;
  height: 30px;
  padding: 3px;
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.13);
}

.nhlcal-event-modal-icon.has-logo {
  padding: 8px;
  background: rgba(255, 255, 255, 0.08);
}

.nhlcal-event-modal-icon.has-logo img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  filter: drop-shadow(0 0 12px rgba(255, 255, 255, 0.18));
}

      .nhlcal-corner-cut {
        width: 0;
        height: 0;
        border-top: 13px solid rgba(19, 216, 231, 0.62);
        border-left: 13px solid transparent;
        position: absolute;
        top: -10px;
        right: -11px;
        filter: drop-shadow(0 0 8px rgba(19, 216, 231, 0.28));
        opacity: 0.85;
      }

      .nhlcal-day-content {
        margin-top: 8px;
        display: grid;
        gap: 8px;
        width: 100%;
      }

      .nhlcal-day-special-events,
      .nhlcal-day-games {
        display: grid;
        gap: 7px;
        width: 100%;
      }

      .nhlcal-empty-day-line {
        color: rgba(128, 150, 168, 0.42);
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding-top: 4px;
      }

      .nhlcal-special-event-tile {
        width: 100%;
        min-height: 38px;
        display: grid;
        grid-template-columns: 26px minmax(0, 1fr);
        align-items: center;
        gap: 7px;
        border-radius: 10px;
        padding: 6px 7px;
        text-align: left;
        border: 1px solid rgba(233, 168, 60, 0.24);
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.16), transparent 58%),
          linear-gradient(180deg, rgba(52, 36, 19, 0.78), rgba(20, 22, 25, 0.74));
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.06),
          0 2px 10px rgba(0, 0, 0, 0.16);
        cursor: pointer;
        transition:
          border-color 0.18s ease,
          transform 0.18s ease,
          background 0.18s ease;
      }

      .nhlcal-special-event-tile:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 214, 135, 0.48);
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.22), transparent 58%),
          linear-gradient(180deg, rgba(64, 43, 20, 0.88), rgba(23, 25, 29, 0.82));
      }

      .nhlcal-special-event-icon {
        width: 26px;
        height: 26px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: rgba(233, 168, 60, 0.18);
        border: 1px solid rgba(233, 168, 60, 0.2);
        color: #ffd88d;
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-special-event-copy {
        min-width: 0;
        display: grid;
        gap: 2px;
      }

      .nhlcal-special-event-copy strong {
        min-width: 0;
        color: rgba(255, 239, 211, 0.96);
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.01em;
        text-transform: none;
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        line-height: 1.2;
        max-height: 2.4em;
        word-break: break-word;
      }

      .nhlcal-special-event-copy span {
        min-width: 0;
        color: rgba(232, 203, 160, 0.72);
        font-size: 9px;
        font-weight: 700;
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 1;
        line-height: 1.2;
      }

      .nhlcal-special-event-tile.priority-critical {
        border-color: rgba(255, 96, 109, 0.42);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.22), transparent 58%),
          linear-gradient(180deg, rgba(68, 24, 31, 0.82), rgba(21, 16, 22, 0.78));
      }

      .nhlcal-special-event-tile.priority-critical .nhlcal-special-event-icon {
        background: rgba(255, 96, 109, 0.16);
        border-color: rgba(255, 96, 109, 0.3);
        color: #ffc4ca;
      }

      .nhlcal-special-event-tile.priority-high:not(.priority-critical) {
        border-color: rgba(233, 168, 60, 0.36);
      }

      .nhlcal-special-event-tile.tone-medical {
        border-color: rgba(255, 96, 109, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(60, 22, 31, 0.78), rgba(18, 20, 25, 0.74));
      }

      .nhlcal-special-event-tile.tone-medical .nhlcal-special-event-icon {
        color: #ffcbd1;
        background: rgba(255, 96, 109, 0.14);
        border-color: rgba(255, 96, 109, 0.28);
      }

      .nhlcal-special-event-tile.tone-trade {
        border-color: rgba(19, 216, 231, 0.32);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(16, 48, 58, 0.78), rgba(18, 23, 28, 0.74));
      }

      .nhlcal-special-event-tile.tone-trade .nhlcal-special-event-icon {
        color: #baf9ff;
        background: rgba(19, 216, 231, 0.13);
        border-color: rgba(19, 216, 231, 0.28);
      }

      .nhlcal-special-event-tile.tone-draft {
        border-color: rgba(201, 146, 255, 0.35);
        background:
          radial-gradient(circle at 0% 0%, rgba(201, 146, 255, 0.18), transparent 56%),
          linear-gradient(180deg, rgba(42, 28, 66, 0.78), rgba(18, 20, 28, 0.74));
      }

      .nhlcal-special-event-tile.tone-draft .nhlcal-special-event-icon {
        color: #ead7ff;
        background: rgba(201, 146, 255, 0.14);
        border-color: rgba(201, 146, 255, 0.28);
      }

      .nhlcal-special-event-tile.tone-playoff {
        border-color: rgba(82, 223, 148, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(82, 223, 148, 0.16), transparent 56%),
          linear-gradient(180deg, rgba(20, 58, 40, 0.78), rgba(16, 23, 23, 0.74));
      }

      .nhlcal-special-event-tile.tone-playoff .nhlcal-special-event-icon {
        color: #caffdf;
        background: rgba(82, 223, 148, 0.13);
        border-color: rgba(82, 223, 148, 0.28);
      }

      .nhlcal-special-event-tile.tone-showcase,
      .nhlcal-special-event-tile.tone-star,
      .nhlcal-special-event-tile.tone-international {
        border-color: rgba(138, 180, 255, 0.34);
        background:
          radial-gradient(circle at 0% 0%, rgba(138, 180, 255, 0.17), transparent 56%),
          linear-gradient(180deg, rgba(25, 39, 68, 0.78), rgba(16, 21, 29, 0.74));
      }

      .nhlcal-special-event-tile.tone-showcase .nhlcal-special-event-icon,
      .nhlcal-special-event-tile.tone-star .nhlcal-special-event-icon,
      .nhlcal-special-event-tile.tone-international .nhlcal-special-event-icon {
        color: #d6e3ff;
        background: rgba(138, 180, 255, 0.13);
        border-color: rgba(138, 180, 255, 0.28);
      }

      .nhlcal-more-events {
        min-height: 24px;
        border: 1px dashed rgba(233, 168, 60, 0.3);
        border-radius: 8px;
        background: rgba(233, 168, 60, 0.06);
        color: rgba(255, 220, 160, 0.86);
        display: grid;
        place-items: center;
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-game-tile {
        display: grid;
        grid-template-columns: 4px auto minmax(0, 1fr) auto;
        align-items: center;
        gap: 8px;
        width: 100%;
        min-height: 58px;
        min-width: 0;
        position: relative;
        border-radius: 12px;
        padding: 8px 9px;
        cursor: pointer;
        text-align: left;
        overflow: hidden;
        transition:
          background 120ms ease,
          border-color 120ms ease,
          box-shadow 120ms ease;
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.11), transparent 55%),
          linear-gradient(180deg, rgba(16, 44, 62, 0.82), rgba(7, 24, 36, 0.84));
        border: 1px solid rgba(156, 218, 236, 0.16);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.05),
          0 2px 10px rgba(0, 0, 0, 0.18);
      }

      .nhlcal-game-tile:hover {
        border-color: rgba(19, 216, 231, 0.34);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.08),
          0 6px 14px rgba(0, 0, 0, 0.25);
      }

      .nhlcal-game-tile.is-home .nhlcal-game-tile-accent {
        background: linear-gradient(180deg, rgba(21, 238, 255, 0.92), rgba(38, 160, 214, 0.72));
      }

      .nhlcal-game-tile.is-away .nhlcal-game-tile-accent {
        background: linear-gradient(180deg, rgba(129, 174, 214, 0.9), rgba(88, 122, 166, 0.68));
      }

      .nhlcal-game-tile.is-final {
        background:
          radial-gradient(circle at 0% 0%, rgba(76, 130, 158, 0.08), transparent 60%),
          linear-gradient(180deg, rgba(11, 31, 46, 0.84), rgba(5, 19, 29, 0.88));
      }

      .nhlcal-game-tile.is-final.result-win {
        background:
          radial-gradient(circle at 0% 0%, rgba(68, 204, 128, 0.16), transparent 58%),
          linear-gradient(180deg, rgba(17, 64, 44, 0.7), rgba(8, 35, 24, 0.72));
        border-color: rgba(95, 226, 155, 0.34);
      }

      .nhlcal-game-tile.is-final.result-loss {
        background:
          radial-gradient(circle at 0% 0%, rgba(238, 86, 86, 0.16), transparent 58%),
          linear-gradient(180deg, rgba(72, 24, 28, 0.7), rgba(36, 11, 14, 0.72));
        border-color: rgba(241, 120, 120, 0.34);
      }

      .nhlcal-game-tile.is-final.result-otl {
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.15), transparent 58%),
          linear-gradient(180deg, rgba(64, 45, 20, 0.68), rgba(35, 24, 12, 0.72));
        border-color: rgba(233, 168, 60, 0.3);
      }

      .nhlcal-game-tile.is-upcoming {
        border-color: rgba(19, 216, 231, 0.2);
      }

      .nhlcal-game-tile.is-expanded {
        border-color: rgba(19, 216, 231, 0.52);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.09),
          0 0 0 1px rgba(19, 216, 231, 0.18),
          0 10px 24px rgba(5, 20, 30, 0.35);
      }

      .nhlcal-game-tile-accent {
        width: 4px;
        height: 100%;
        min-height: 42px;
        border-radius: 999px;
        box-shadow: 0 0 8px rgba(19, 216, 231, 0.45);
      }

      .nhlcal-game-tile-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }

      .nhlcal-game-tile-main {
        min-width: 0;
        display: grid;
        gap: 3px;
      }

      .nhlcal-game-match-line {
        display: flex;
        align-items: center;
        gap: 5px;
        min-width: 0;
      }

      .nhlcal-game-match-line strong {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: rgba(233, 247, 251, 0.94);
        font-size: 12px;
        font-weight: 1000;
        letter-spacing: 0.04em;
      }

      .nhlcal-game-relation {
        flex: 0 0 auto;
        min-width: 26px;
        height: 18px;
        padding: 0 5px;
        border-radius: 999px;
        display: inline-grid;
        place-items: center;
        font-size: 9px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(233, 247, 251, 0.88);
        background: rgba(109, 153, 178, 0.26);
        border: 1px solid rgba(156, 218, 236, 0.2);
      }

      .nhlcal-game-relation.home {
        color: rgba(152, 246, 255, 0.95);
        background: rgba(19, 216, 231, 0.18);
        border-color: rgba(19, 216, 231, 0.38);
      }

      .nhlcal-game-relation.away {
        color: rgba(191, 214, 241, 0.94);
        background: rgba(105, 137, 175, 0.2);
        border-color: rgba(117, 156, 201, 0.35);
      }

      .nhlcal-game-meta-line {
        display: flex;
        align-items: center;
        gap: 5px;
        min-width: 0;
        color: var(--muted);
        font-size: 10px;
        font-weight: 850;
      }

      .nhlcal-game-meta-line span,
      .nhlcal-game-meta-line em {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-game-meta-line em {
        color: rgba(174, 206, 223, 0.68);
        font-style: normal;
      }

      .nhlcal-game-tile-side {
        display: grid;
        justify-items: end;
        gap: 3px;
        min-width: 38px;
      }

      .nhlcal-game-score-mini {
        min-width: 38px;
        height: 22px;
        padding: 0 6px;
        border-radius: 7px;
        background: rgba(0, 0, 0, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        color: rgba(233, 247, 251, 0.9);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-game-score-mini em {
        color: rgba(176, 205, 219, 0.78);
        font-style: normal;
      }

      .nhlcal-game-status-pill {
        min-width: 42px;
        height: 22px;
        padding: 0 6px;
        border-radius: 7px;
        border: 1px solid rgba(19, 216, 231, 0.24);
        background: rgba(19, 216, 231, 0.08);
        color: rgba(180, 238, 245, 0.95);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlcal-game-chevron {
        color: rgba(170, 206, 223, 0.78);
        font-size: 11px;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlcal-game-expand-details {
        grid-column: 1 / -1;
        margin-top: 6px;
        padding: 8px;
        border-radius: 9px;
        background: rgba(2, 10, 16, 0.32);
        border-top: 1px solid rgba(19, 216, 231, 0.18);
        display: grid;
        gap: 8px;
      }

      .nhlcal-game-expand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
      }

      .nhlcal-game-expand-header strong {
        color: rgba(233, 247, 251, 0.95);
        font-size: 11px;
        letter-spacing: 0.09em;
      }

      .nhlcal-game-expand-header span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
      }

      .nhlcal-game-expand-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }

      .nhlcal-game-expand-row > div {
        min-width: 0;
        display: grid;
        grid-template-columns: auto 1fr;
        grid-template-rows: auto auto;
        align-items: center;
        column-gap: 6px;
        row-gap: 1px;
      }

      .nhlcal-game-expand-row > div span {
        color: rgba(233, 247, 251, 0.92);
        font-size: 10px;
        font-weight: 900;
      }

      .nhlcal-game-expand-row > div small {
        color: var(--muted);
        font-size: 9px;
        grid-column: 2;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-tile {
        min-height: 46px;
        padding: 6px 7px;
        grid-template-columns: 3px auto minmax(0, 1fr) auto;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-meta-line em {
        display: none;
      }

      .nhlcal-month-grid.is-dense .nhlcal-game-match-line strong {
        font-size: 11px;
      }

      .nhlcal-month-grid.is-dense .nhlcal-special-event-tile {
        min-height: 34px;
        grid-template-columns: 24px minmax(0, 1fr);
        padding: 5px 6px;
      }

      .nhlcal-month-grid.is-dense .nhlcal-special-event-copy span {
        display: none;
      }

      .nhlcal-more-games {
        color: var(--muted);
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-day-hover-card {
        position: absolute;
        left: 10px;
        bottom: calc(100% - 10px);
        width: 245px;
        border: 1px solid var(--line-2);
        border-radius: 12px;
        background: rgba(4, 15, 24, 0.98);
        padding: 12px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.42);
        z-index: 20;
        pointer-events: none;
      }

      .nhlcal-day-hover-card strong,
      .nhlcal-day-hover-card span {
        display: block;
      }

      .nhlcal-day-hover-card strong {
        font-size: 12px;
        color: var(--text);
      }

      .nhlcal-day-hover-card span {
        margin-top: 5px;
        color: var(--muted);
        font-size: 11px;
        line-height: 1.35;
      }

      .nhlcal-calendar-footer {
        flex: 0 0 auto;
        min-height: 52px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 12px;
        border-top: 1px solid var(--line);
        background: rgba(5, 16, 25, 0.72);
        border-radius: 0 0 12px 12px;
      }

      .nhlcal-week-header {
        flex: 0 0 auto;
      }

      .nhlcal-legend {
        display: flex;
        align-items: center;
        gap: 18px;
        flex-wrap: wrap;
      }

      .nhlcal-legend span {
        display: flex;
        align-items: center;
        gap: 8px;
        color: rgba(233, 247, 251, 0.74);
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .dot {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        background: var(--muted-2);
      }

      .dot.home {
        background: var(--cyan);
      }

      .dot.away {
        background: #6b8090;
      }

      .dot.team-game {
        background: #0b707c;
      }

      .dot.win {
        background: #2fd67b;
      }

      .dot.loss {
        background: #ff6070;
      }

      .dot.otl {
        background: #e9a83c;
      }

      .dot.special {
        background: var(--gold);
      }

      .dot.critical {
        background: var(--red);
      }

      .nhlcal-calendar-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .nhlcal-calendar-actions button {
        height: 34px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(14, 35, 50, 0.9);
        color: rgba(233, 247, 251, 0.82);
        padding: 0 12px;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-calendar-actions button:hover,
      .nhlcal-calendar-actions button.is-active {
        border-color: var(--line-strong);
        color: var(--text);
        background: rgba(19, 216, 231, 0.11);
      }
      .nhlcal-calendar-footer .nhlcal-calendar-actions {
        display: none;
      }

      .nhlcal-calendar-footer {
        flex: 0 0 auto;
        min-height: 36px;
        padding: 6px 10px;
      }

              .nhlcal-right-rail {
        min-width: 0;
        min-height: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlcal-rail-preview-wrap {
        flex: 0 0 auto;
        min-height: 0;
        max-height: 32%;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      .nhlcal-rail-preview-wrap .nhlcal-preview-card {
        flex: 1 1 auto;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
      }

      .nhlcal-rail-preview-wrap .nhlcal-preview-card.nhlcal-scroll-surface {
        scrollbar-gutter: stable;
      }

      .nhlcal-selected-summary-strip {
        display: none;
      }

      .nhlcal-selected-summary-strip span {
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(156, 218, 236, 0.12);
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlcal-preview-mode-banner,
      .nhlcal-preview-missing {
        margin: 0 14px 10px;
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
      }

      .nhlcal-preview-mode-banner {
        border: 1px solid rgba(136, 180, 255, 0.24);
        background: rgba(136, 180, 255, 0.08);
        color: #cfe0ff;
      }

      .nhlcal-preview-missing {
        border: 1px solid rgba(255, 96, 109, 0.24);
        background: rgba(255, 96, 109, 0.08);
        color: #ffc4c9;
      }

      .nhlcal-tab-row button.is-active {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.14);
        color: var(--text);
        box-shadow: inset 0 -2px 0 var(--cyan);
      }

      .nhlcal-card {
        border: 1px solid var(--line);
        border-radius: 12px;
        background:
          linear-gradient(180deg, rgba(10, 30, 45, 0.94), rgba(5, 18, 29, 0.94)),
          radial-gradient(circle at 90% 0%, rgba(19, 216, 231, 0.07), transparent 38%);
        box-shadow: var(--shadow);
      }

      .nhlcal-card-header {
        min-height: 58px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        padding: 16px 18px 10px;
      }

      .nhlcal-card-header.compact {
        padding-bottom: 10px;
      }

      .nhlcal-card-header p,
      .nhlcal-card-header h3 {
        margin: 0;
      }

      .nhlcal-card-header p {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.13em;
        text-transform: uppercase;
      }

      .nhlcal-card-header h3 {
        margin-top: 4px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-card-header button,
      .nhlcal-mini-header button {
        border: 0;
        background: transparent;
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-card-header button:hover,
      .nhlcal-mini-header button:hover {
        color: var(--cyan);
      }

      .nhlcal-header-pill {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        padding: 6px 10px;
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-preview-card {
        overflow: hidden;
      }

      .nhlcal-preview-card .nhlcal-card-header {
        min-height: 46px;
        padding: 10px 12px 6px;
      }

      .nhlcal-preview-card .nhlcal-card-header h3 {
        font-size: 12px;
      }

      .nhlcal-preview-card .nhlcal-header-pill {
        font-size: 9px;
      }

      .nhlcal-matchup-stage {
        min-height: 118px;
        display: grid;
        grid-template-columns: 1fr 0.72fr 1fr;
        align-items: center;
        gap: 6px;
        padding: 6px 12px 12px;
      }

      .nhlcal-matchup-team {
        display: grid;
        place-items: center;
        text-align: center;
        gap: 6px;
      }

      .nhlcal-matchup-team strong {
        font-size: 20px;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.06em;
      }

      .nhlcal-matchup-team span {
        color: var(--muted);
        font-size: 12px;
        font-weight: 900;
      }

      .nhlcal-versus {
        display: grid;
        place-items: center;
        text-align: center;
      }

      .nhlcal-versus strong {
        width: 44px;
        height: 44px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(233, 247, 251, 0.72);
        font-size: 16px;
        font-weight: 1000;
      }

      .nhlcal-versus span {
        margin-top: 8px;
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-versus small {
        margin-top: 4px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-tab-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        padding: 0 14px;
        border-top: 1px solid var(--line);
      }

      .nhlcal-tab-row-three {
        grid-template-columns: repeat(3, 1fr);
      }

      .nhlcal-tab-row button {
        height: 42px;
        border: 0;
        border-bottom: 2px solid transparent;
        background: transparent;
        color: var(--muted);
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        cursor: pointer;
      }

      .nhlcal-tab-row button.is-active {
        color: var(--text);
        border-bottom-color: var(--cyan);
      }

      .nhlcal-preview-lines {
        padding: 10px 14px 14px;
        display: grid;
      }

      .nhlcal-preview-lines div {
        min-height: 31px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid rgba(156, 218, 236, 0.09);
      }

      .nhlcal-preview-lines div:last-child {
        border-bottom: 0;
      }

      .nhlcal-preview-lines span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-preview-lines strong {
        color: rgba(233, 247, 251, 0.9);
        font-size: 12px;
        font-weight: 1000;
        text-align: right;
      }

      .nhlcal-wide-action {
        width: calc(100% - 28px);
        min-height: 36px;
        margin: 0 14px 14px;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: rgba(15, 38, 55, 0.9);
        color: rgba(233, 247, 251, 0.88);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
        padding: 0 12px;
      }

      .nhlcal-wide-action:hover {
        border-color: var(--line-strong);
      }

      .nhlcal-wide-action.muted {
        margin-top: 12px;
        margin-bottom: 14px;
      }

      .nhlcal-empty-preview {
        min-height: 250px;
        display: grid;
        place-items: center;
        text-align: center;
        padding: 24px;
      }

      .nhlcal-empty-orb {
        width: 74px;
        height: 74px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.035);
        color: var(--cyan);
        font-size: 36px;
      }

      .nhlcal-empty-preview h4 {
        margin: 12px 0 0;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-empty-preview p {
        margin: 8px 0 0;
        max-width: 300px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
      }

      .nhlcal-empty-subnote {
        color: rgba(233, 247, 251, 0.7) !important;
        font-weight: 800;
      }

      .nhlcal-selected-event-strip {
        padding: 0 14px 2px;
        display: grid;
        gap: 7px;
      }

      .nhlcal-selected-event-chip {
        min-height: 34px;
        border-radius: 8px;
        border: 1px solid rgba(233, 168, 60, 0.24);
        background: rgba(233, 168, 60, 0.07);
        color: var(--text);
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr);
        align-items: center;
        gap: 7px;
        padding: 5px 8px;
        text-align: left;
        cursor: pointer;
      }

      .nhlcal-selected-event-chip span {
        width: 24px;
        height: 24px;
        border-radius: 7px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.14);
      }

      .nhlcal-selected-event-chip strong {
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-selected-day-panel {
        padding: 12px 14px 16px;
        display: grid;
        gap: 12px;
      }

      .nhlcal-selected-section {
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 12px;
      }

      .nhlcal-selected-section-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }

      .nhlcal-selected-section-head span {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-selected-section-head strong {
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-selected-section-head button {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(19, 216, 231, 0.07);
        color: var(--cyan);
        padding: 5px 9px;
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-selected-event-list,
      .nhlcal-selected-injury-list,
      .nhlcal-selected-slate-list {
        display: grid;
        gap: 8px;
      }

      .nhlcal-selected-event-row {
        width: 100%;
        min-height: 50px;
        border: 1px solid rgba(233, 168, 60, 0.2);
        border-radius: 10px;
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.1), transparent 50%),
          rgba(255, 255, 255, 0.025);
        color: var(--text);
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        align-items: center;
        gap: 10px;
        text-align: left;
        padding: 8px;
        cursor: pointer;
      }

      .nhlcal-selected-event-row:hover {
        border-color: rgba(233, 168, 60, 0.42);
      }

      .nhlcal-selected-event-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.12);
        color: #ffd88d;
      }

      .nhlcal-selected-event-row strong,
      .nhlcal-selected-event-row small {
        display: block;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlcal-selected-event-row strong {
        color: var(--text);
        font-size: 12px;
        font-weight: 1000;
        white-space: nowrap;
      }

      .nhlcal-selected-event-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
        line-height: 1.3;
      }

      .nhlcal-selected-injury-list article {
        min-height: 42px;
        border: 1px solid rgba(255, 96, 109, 0.14);
        border-radius: 8px;
        background: rgba(255, 96, 109, 0.055);
        padding: 8px 10px;
      }

      .nhlcal-selected-injury-list strong,
      .nhlcal-selected-injury-list span {
        display: block;
      }

      .nhlcal-selected-injury-list strong {
        color: rgba(255, 213, 217, 0.95);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-selected-injury-list span {
        margin-top: 3px;
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
      }

      .nhlcal-selected-slate-list article {
        min-height: 32px;
        display: grid;
        grid-template-columns: 1fr 16px 1fr auto;
        gap: 8px;
        align-items: center;
        border-bottom: 1px solid rgba(156, 218, 236, 0.07);
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-selected-slate-list article:last-child {
        border-bottom: 0;
      }

      .nhlcal-selected-slate-list article.is-user-game {
        color: var(--cyan);
      }

      .nhlcal-selected-slate-list article span:first-child {
        text-align: right;
      }

      .nhlcal-selected-slate-list article em {
        color: var(--muted);
        font-style: normal;
        text-align: center;
      }

      .nhlcal-selected-slate-list article strong {
        color: inherit;
        font-size: 10px;
        text-align: right;
      }

      .nhlcal-selected-more {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 10px;
        font-weight: 900;
        text-align: center;
      }

      .nhlcal-standings-card {
        flex: 1 1 0;
        min-height: 0;
        max-height: none;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        border-color: rgba(19, 216, 231, 0.22);
        box-shadow:
          0 12px 28px rgba(0, 0, 0, 0.28),
          inset 0 1px 0 rgba(19, 216, 231, 0.08);
      }

      .nhlcal-standings-card--extended {
        flex: 1 1 auto;
        min-height: 0;
      }

      .nhlcal-standings-header {
        flex: 0 0 auto;
        min-height: 46px;
        padding: 10px 12px 6px;
      }

      .nhlcal-standings-header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }

      .nhlcal-standings-count {
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-standings-full-link {
        border: 1px solid rgba(19, 216, 231, 0.24);
        border-radius: 8px;
        background: rgba(19, 216, 231, 0.08);
        color: var(--cyan);
        padding: 6px 10px;
        font-size: 10px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-standings-table {
        flex: 1 1 auto;
        min-height: 0;
        display: flex;
        flex-direction: column;
        padding: 0 10px 6px;
      }

      .nhlcal-standings-body {
        flex: 1 1 auto;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        border-top: 1px solid rgba(156, 218, 236, 0.08);
        scrollbar-gutter: stable;
        padding-right: 2px;
      }

      .nhlcal-standings-footer {
        flex: 0 0 auto;
        padding: 0 10px 10px;
      }

      .nhlcal-standings-full-button {
        width: 100%;
        min-height: 34px;
        border: 1px solid rgba(156, 218, 236, 0.18);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.04);
        color: rgba(233, 247, 251, 0.88);
        font-size: 10px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-standings-full-button:hover,
      .nhlcal-standings-full-link:hover {
        border-color: rgba(19, 216, 231, 0.42);
        background: rgba(19, 216, 231, 0.1);
      }

      .nhlcal-standings-head,
      .nhlcal-standings-row {
        display: grid;
        grid-template-columns: 22px minmax(0, 1fr) 24px 22px 22px 26px 28px 34px;
        align-items: center;
        gap: 4px;
      }

      .nhlcal-standings-head {
        flex: 0 0 auto;
        height: 26px;
        color: rgba(233, 247, 251, 0.64);
        text-transform: uppercase;
        font-size: 9px;
        font-weight: 1000;
        letter-spacing: 0.06em;
        border-bottom: 1px solid rgba(156, 218, 236, 0.12);
      }

      .nhlcal-standings-row {
        min-height: 34px;
        color: rgba(233, 247, 251, 0.8);
        font-size: 10px;
        font-weight: 800;
        border-bottom: 1px solid rgba(156, 218, 236, 0.07);
      }

      .nhlcal-standings-row > span:nth-child(2) {
        display: flex;
        align-items: center;
        gap: 7px;
        min-width: 0;
      }

      .nhlcal-standings-row > span:nth-child(2) strong {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-standings-row > span:not(:nth-child(2)),
      .nhlcal-standings-head > span:not(:nth-child(2)) {
        text-align: right;
      }

      .nhlcal-standings-row.is-user-team {
        color: var(--cyan);
        background: rgba(19, 216, 231, 0.1);
        box-shadow: inset 3px 0 0 var(--cyan);
      }

      .nhlcal-standings-row.is-user-team strong {
        color: var(--cyan);
      }

      .nhlcal-mini-card-row {
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        max-height: 96px;
        overflow: hidden;
      }

      .nhlcal-table-empty {
        padding: 14px 0;
        color: var(--muted);
        font-size: 11px;
        text-align: center;
      }

      .nhlcal-stretch-card,
      .nhlcal-league-card {
        min-height: 245px;
        overflow: hidden;
      }

      .nhlcal-mini-header {
        min-height: 54px;
        padding: 15px 15px 8px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
      }

      .nhlcal-mini-header h3 {
        margin: 0;
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-mini-header span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-stretch-list,
      .nhlcal-league-list {
        padding: 0 14px;
        display: grid;
        gap: 7px;
      }

      .nhlcal-stretch-row,
      .nhlcal-league-row {
        min-height: 25px;
        display: grid;
        align-items: center;
        gap: 7px;
        color: rgba(233, 247, 251, 0.82);
        font-size: 11px;
        font-weight: 900;
      }

      .nhlcal-stretch-row {
        grid-template-columns: 42px 22px minmax(0, 1fr) auto;
      }

      .nhlcal-stretch-row > span {
        color: var(--muted);
      }

      .nhlcal-stretch-row strong {
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }

      .nhlcal-stretch-row em {
        min-width: 45px;
        padding: 3px 6px;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.045);
        color: var(--muted);
        font-size: 9px;
        font-style: normal;
        text-align: center;
        text-transform: uppercase;
      }

      .nhlcal-stretch-row em.home {
        color: var(--cyan);
        background: var(--cyan-soft);
      }

      .nhlcal-stretch-row em.away {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .nhlcal-league-row {
        grid-template-columns: 1fr 14px 1fr auto;
      }

      .nhlcal-league-row span:first-child {
        text-align: right;
      }

      .nhlcal-league-row em {
        color: var(--muted);
        font-style: normal;
        text-align: center;
      }

      .nhlcal-league-row strong {
        color: var(--muted);
        font-size: 10px;
        text-align: right;
      }

      .nhlcal-league-row.is-highlight span,
      .nhlcal-league-row.is-highlight strong {
        color: var(--cyan);
      }

      .nhlcal-storyline-list {
        gap: 9px;
      }

      .nhlcal-storyline-row {
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 9px;
      }

      .nhlcal-storyline-topline {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr) auto;
        gap: 8px;
        align-items: center;
      }

      .nhlcal-storyline-topline span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 900;
      }

      .nhlcal-storyline-topline strong {
        min-width: 0;
        color: var(--text);
        font-size: 11px;
        font-weight: 1000;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-storyline-topline em {
        border-radius: 999px;
        padding: 3px 6px;
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        font-style: normal;
        font-size: 8px;
        font-weight: 1000;
      }

      .nhlcal-storyline-topline em.critical {
        color: var(--red);
        background: var(--red-soft);
      }

      .nhlcal-storyline-topline em.high {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .nhlcal-subtext {
        margin-top: 5px;
        color: var(--muted);
        font-size: 10px;
        line-height: 1.35;
        font-weight: 800;
      }

      .nhlcal-storyline-choice-row {
        margin-top: 7px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
      }

      .nhlcal-storyline-choice-button {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(19, 216, 231, 0.07);
        color: var(--cyan);
        min-height: 26px;
        padding: 0 9px;
        font-size: 9px;
        font-weight: 1000;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlcal-storyline-choice-button:disabled {
        cursor: not-allowed;
        opacity: 0.58;
        filter: saturate(0.65);
      }

      .nhlcal-injury-mini-list {
        gap: 8px;
      }

      .nhlcal-injury-mini-row {
        min-height: 38px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto;
        grid-template-rows: auto auto;
        gap: 3px 8px;
        align-items: center;
        border: 1px solid rgba(255, 96, 109, 0.1);
        border-radius: 9px;
        background: rgba(255, 96, 109, 0.04);
        padding: 8px;
      }

      .nhlcal-injury-mini-row span {
        min-width: 0;
        color: var(--text);
        font-size: 11px;
        font-weight: 1000;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .nhlcal-injury-mini-row em {
        color: rgba(255, 198, 205, 0.9);
        font-size: 10px;
        font-style: normal;
        font-weight: 900;
      }

      .nhlcal-injury-mini-row strong {
        color: var(--red);
        font-size: 10px;
        font-weight: 1000;
      }

      .nhlcal-injury-mini-row small {
        grid-column: 1 / -1;
        color: var(--muted);
        font-size: 10px;
        font-weight: 800;
      }

      .nhlcal-small-empty {
        margin: 18px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.4;
      }

      .nhlcal-mini-button {
        width: calc(100% - 28px);
        min-height: 34px;
        margin: 13px 14px 14px;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: rgba(15, 38, 55, 0.9);
        color: rgba(233, 247, 251, 0.78);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 10px;
        font-weight: 1000;
        cursor: pointer;
      }

      .nhlcal-bottom-grid {
        margin-top: 16px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 400px;
        gap: 16px;
      }

      .nhlcal-bottom-grid--collapsed {
        display: none;
      }

      .nhlcal-diagnostics-panel,
      .nhlcal-month-snapshot {
        padding: 16px;
      }

      .nhlcal-section-title {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 13px;
      }

      .nhlcal-section-title p,
      .nhlcal-section-title h3 {
        margin: 0;
      }

      .nhlcal-section-title p {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .nhlcal-section-title h3 {
        margin-top: 4px;
        color: rgba(233, 247, 251, 0.94);
        font-size: 14px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-section-title > span {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 6px 10px;
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlcal-diagnostic-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }

      .nhlcal-diagnostic-tile {
        min-height: 92px;
        border: 1px solid rgba(156, 218, 236, 0.11);
        border-radius: 9px;
        background:
          linear-gradient(180deg, rgba(15, 38, 56, 0.84), rgba(8, 25, 38, 0.78)),
          radial-gradient(circle at 100% 30%, rgba(19, 216, 231, 0.1), transparent 52%);
        padding: 13px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .nhlcal-diagnostic-tile span,
      .nhlcal-diagnostic-tile small {
        display: block;
      }

      .nhlcal-diagnostic-tile span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-diagnostic-tile strong {
        display: block;
        margin-top: 6px;
        font-size: 26px;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlcal-diagnostic-tile small {
        margin-top: 8px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-diagnostic-tile em {
        color: rgba(19, 216, 231, 0.56);
        font-size: 34px;
        font-style: normal;
      }

      .nhlcal-diagnostic-tile.is-danger strong {
        color: var(--red);
      }

      .nhlcal-diagnostic-tile.is-danger em {
        color: rgba(255, 96, 109, 0.62);
      }

      .nhlcal-insight-row {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-insight-pill {
        min-height: 52px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr);
        align-items: center;
        gap: 10px;
        padding: 10px;
      }

      .nhlcal-insight-pill span {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: var(--gold-soft);
        color: var(--gold);
        font-size: 12px;
        font-weight: 1000;
      }

      .nhlcal-insight-pill p {
        margin: 0;
        color: rgba(233, 247, 251, 0.78);
        font-size: 11px;
        line-height: 1.35;
        font-weight: 750;
      }

      .nhlcal-snapshot-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-snapshot-grid div {
        min-height: 74px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.035);
        display: grid;
        place-items: center;
        text-align: center;
      }

      .nhlcal-snapshot-grid span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-snapshot-grid strong {
        color: var(--text);
        font-size: 24px;
        font-weight: 1000;
      }

      .nhlcal-opponent-strip {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-opponent-strip div {
        min-height: 60px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        gap: 4px;
        padding: 8px;
      }

      .nhlcal-opponent-strip span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-align: center;
      }

      .nhlcal-opponent-strip p {
        grid-column: 1 / -1;
        margin: 0;
        color: var(--muted);
        font-size: 12px;
        text-align: center;
        align-self: center;
      }

      .nhlcal-opponent-event strong {
        color: var(--gold);
        font-size: 20px;
      }

      .nhlcal-team-badge {
        --team-seed: 185 78% 48%;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 999px;
        background:
          radial-gradient(circle at 30% 25%, rgba(255, 255, 255, 0.2), transparent 28%),
          linear-gradient(135deg, hsl(var(--team-seed) / 0.95), hsl(var(--team-seed) / 0.38));
        border: 1px solid hsl(var(--team-seed) / 0.58);
        color: #eaffff;
        font-weight: 1000;
        letter-spacing: 0.05em;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.34);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.16),
          0 0 20px hsl(var(--team-seed) / 0.16);
      }

      .nhlcal-team-badge.size-large {
        width: 94px;
        height: 94px;
        font-size: 23px;
        border-radius: 24px;
      }

      .nhlcal-team-badge.size-matchup {
        width: 86px;
        height: 86px;
        font-size: 20px;
        border-radius: 22px;
      }

      .nhlcal-team-badge.size-small {
        width: 34px;
        height: 34px;
        font-size: 10px;
      }

      .nhlcal-team-badge.size-tiny {
        width: 24px;
        height: 24px;
        font-size: 8px;
      }

      .nhlcal-team-badge.size-mini {
        width: 18px;
        height: 18px;
        font-size: 7px;
      }

      .nhlcal-team-badge.size-tile-main {
        width: 44px;
        height: 44px;
        font-size: 12px;
        border-radius: 12px;
      }

      .nhlcal-team-badge.size-tile-compact {
        width: 34px;
        height: 34px;
        font-size: 10px;
        border-radius: 10px;
      }

      .nhlcal-team-logo {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: rgba(2, 11, 18, 0.85);
        border: 1px solid rgba(103, 157, 183, 0.32);
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
      }

      .nhlcal-team-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        filter: drop-shadow(0 2px 5px rgba(0, 0, 0, 0.35));
      }

      .nhlcal-team-logo.size-large {
        width: 76px;
        height: 76px;
        border-radius: 18px;
        padding: 7px;
      }

      .nhlcal-matchup-stage-compact {
        min-height: 88px;
        padding: 4px 10px 8px;
      }

      .nhlcal-matchup-stage-compact .nhlcal-matchup-team strong {
        font-size: 16px;
      }

      .nhlcal-matchup-stage-compact .nhlcal-versus strong {
        width: 34px;
        height: 34px;
        font-size: 12px;
      }

      .nhlcal-preview-lines-compact div {
        min-height: 28px;
        padding: 4px 12px;
      }

      .nhlcal-preview-empty-stats {
        margin: 0 12px 10px;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px dashed rgba(156, 218, 236, 0.18);
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        text-align: center;
      }

      .nhlcal-tab-row-two {
        grid-template-columns: 1fr 1fr;
      }

      .nhlcal-team-logo.size-matchup {
        width: 44px;
        height: 44px;
        border-radius: 11px;
        padding: 4px;
      }

      .nhlcal-team-logo.size-small {
        width: 34px;
        height: 34px;
        padding: 3px;
      }

      .nhlcal-team-logo.size-tiny {
        width: 24px;
        height: 24px;
        padding: 2px;
      }

      .nhlcal-team-logo.size-mini {
        width: 20px;
        height: 20px;
        padding: 1px;
      }

      .nhlcal-team-logo.size-tile-main {
        width: 44px;
        height: 44px;
        padding: 3px;
        border-radius: 12px;
      }

      .nhlcal-team-logo.size-tile-compact {
        width: 34px;
        height: 34px;
        padding: 2px;
        border-radius: 10px;
      }

      .nhlcal-drawer-backdrop,
      .nhlcal-modal-backdrop,
      .nhlcal-event-backdrop {
        position: fixed;
        inset: 0;
        z-index: 100;
        background: rgba(0, 5, 10, 0.58);
        backdrop-filter: blur(7px);
        display: flex;
      }

      .nhlcal-drawer-backdrop {
        justify-content: flex-end;
      }

      .nhlcal-modal-backdrop,
      .nhlcal-event-backdrop {
        align-items: center;
        justify-content: center;
      }

      .nhlcal-injury-backdrop {
        position: fixed;
        inset: 0;
        z-index: 100;
        background: rgba(0, 5, 10, 0.58);
        backdrop-filter: blur(7px);
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nhlcal-event-modal {
        width: min(620px, 94vw);
        max-height: 88vh;
        overflow: auto;
        border: 1px solid var(--line-strong);
        border-radius: 16px;
        background:
          radial-gradient(circle at 0% 0%, rgba(233, 168, 60, 0.13), transparent 38%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-event-modal.tone-medical {
        border-color: rgba(255, 96, 109, 0.48);
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.16), transparent 38%),
          linear-gradient(180deg, rgba(32, 12, 20, 0.98), rgba(2, 9, 15, 0.98));
      }

      .nhlcal-event-modal.tone-trade {
        border-color: rgba(19, 216, 231, 0.45);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.14), transparent 38%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
      }

      .nhlcal-event-modal-head {
        display: grid;
        grid-template-columns: 54px minmax(0, 1fr) 42px;
        align-items: start;
        gap: 14px;
        padding: 22px;
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-event-modal-icon {
        width: 52px;
        height: 52px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        background: rgba(233, 168, 60, 0.13);
        border: 1px solid rgba(233, 168, 60, 0.25);
        color: #ffd88d;
        font-size: 24px;
      }

      .nhlcal-event-modal-head p,
      .nhlcal-event-modal-head h2 {
        margin: 0;
      }

      .nhlcal-event-modal-head p {
        color: var(--gold);
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .nhlcal-event-modal-head h2 {
        margin-top: 6px;
        color: var(--text);
        font-size: 25px;
        line-height: 1.05;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-event-modal-head button {
        width: 42px;
        height: 42px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 26px;
        line-height: 1;
        cursor: pointer;
      }

      .nhlcal-event-modal-body {
        padding: 18px 22px 22px;
        display: grid;
        gap: 12px;
      }

      .nhlcal-event-modal-description {
        margin: 0;
        color: rgba(233, 247, 251, 0.82);
        font-size: 13px;
        line-height: 1.55;
        font-weight: 750;
      }

      .nhlcal-event-modal-callout {
        border: 1px solid rgba(156, 218, 236, 0.12);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
      }

      .nhlcal-event-modal-callout span,
      .nhlcal-event-modal-callout strong {
        display: block;
      }

      .nhlcal-event-modal-callout span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlcal-event-modal-callout strong {
        margin-top: 5px;
        color: var(--text);
        font-size: 13px;
        line-height: 1.35;
      }

      .nhlcal-event-effect-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-event-effect-grid article {
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 11px;
      }

      .nhlcal-event-effect-grid span,
      .nhlcal-event-effect-grid strong {
        display: block;
      }

      .nhlcal-event-effect-grid span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-event-effect-grid strong {
        margin-top: 4px;
        color: var(--cyan);
        font-size: 18px;
        font-weight: 1000;
      }

      .nhlcal-injury-report-modal {
        width: min(760px, 96vw);
        max-height: 88vh;
        overflow: auto;
        z-index: 101;
        border: 1px solid var(--line-strong);
        border-radius: 16px;
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.12), transparent 38%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-injury-report-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 22px 22px 16px;
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-injury-report-kicker {
        margin: 0;
        color: var(--cyan);
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-injury-report-head h2 {
        margin: 6px 0 0;
        font-size: 26px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-injury-report-stats {
        margin: 10px 0 0;
        font-size: 13px;
        color: var(--muted);
      }

      .nhlcal-injury-report-close {
        width: 42px;
        height: 42px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 26px;
        line-height: 1;
        cursor: pointer;
        flex-shrink: 0;
      }

      .nhlcal-injury-report-body {
        padding: 16px 18px 22px;
      }

      .nhlcal-injury-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }

      .nhlcal-injury-table th,
      .nhlcal-injury-table td {
        text-align: left;
        padding: 8px 6px;
        border-bottom: 1px solid rgba(156, 218, 236, 0.08);
      }

      .nhlcal-injury-table th {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }

      .nhlcal-drawer {
        width: min(560px, 94vw);
        height: 100vh;
        overflow: auto;
        border-left: 1px solid var(--line-strong);
        background:
          radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.13), transparent 34%),
          linear-gradient(180deg, rgba(7, 21, 34, 0.98), rgba(2, 9, 15, 0.98));
        box-shadow: -30px 0 80px rgba(0, 0, 0, 0.55);
      }

      .nhlcal-drawer-header {
        min-height: 100px;
        padding: 25px 25px 18px;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
      }

      .nhlcal-drawer-header p,
      .nhlcal-drawer-header h2 {
        margin: 0;
      }

      .nhlcal-drawer-header p {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.16em;
      }

      .nhlcal-drawer-header h2 {
        margin-top: 6px;
        font-size: 32px;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-drawer-header button,
      .nhlcal-modal header button {
        width: 42px;
        height: 42px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.035);
        color: var(--text);
        font-size: 27px;
        line-height: 1;
        cursor: pointer;
      }

      .nhlcal-drawer-tabs {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        border-bottom: 1px solid var(--line);
      }

      .nhlcal-drawer-tabs button {
        height: 48px;
        border: 0;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
        border-bottom: 2px solid transparent;
        background: rgba(255, 255, 255, 0.02);
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
        font-weight: 1000;
        cursor: pointer;
      }

      .nhlcal-drawer-tabs button:last-child {
        border-right: 0;
      }

      .nhlcal-drawer-tabs button.is-active {
        color: var(--text);
        border-bottom-color: var(--cyan);
        background: rgba(19, 216, 231, 0.08);
      }

      .nhlcal-drawer-body {
        padding: 20px;
        display: grid;
        gap: 18px;
      }

      .nhlcal-drawer-section {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.025);
        padding: 16px;
      }

      .nhlcal-drawer-section h3 {
        margin: 0 0 12px;
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .nhlcal-drawer-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-drawer-nav-card {
        min-height: 74px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background:
          linear-gradient(180deg, rgba(18, 42, 60, 0.72), rgba(7, 23, 34, 0.72));
        color: var(--text);
        text-align: left;
        padding: 13px;
        cursor: pointer;
        transition:
          border-color 0.2s ease,
          transform 0.2s ease,
          background 0.2s ease;
      }

      .nhlcal-drawer-nav-card:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.08);
        transform: translateY(-1px);
      }

      .nhlcal-drawer-nav-card strong,
      .nhlcal-drawer-nav-card span {
        display: block;
      }

      .nhlcal-drawer-nav-card strong {
        font-size: 14px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-drawer-nav-card span {
        margin-top: 6px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
      }

      .nhlcal-drawer-feed {
        display: grid;
        gap: 10px;
      }

      .nhlcal-drawer-feed article {
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 12px;
      }

      .nhlcal-drawer-feed strong {
        display: block;
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-drawer-feed strong span {
        margin-right: 6px;
      }

      .nhlcal-drawer-feed p {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
      }

      .nhlcal-drawer-event {
        border-color: rgba(233, 168, 60, 0.14) !important;
      }

      .nhlcal-drawer-event.tone-medical {
        border-color: rgba(255, 96, 109, 0.22) !important;
        background: rgba(255, 96, 109, 0.04) !important;
      }

      .nhlcal-drawer-event.tone-trade {
        border-color: rgba(19, 216, 231, 0.22) !important;
        background: rgba(19, 216, 231, 0.035) !important;
      }

      .nhlcal-drawer-empty {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
      }

      .nhlcal-player-list,
      .nhlcal-draft-board {
        display: grid;
        gap: 8px;
      }

      .nhlcal-player-mini-row,
      .nhlcal-draft-mini-row {
        min-height: 54px;
        display: grid;
        grid-template-columns: 32px minmax(0, 1fr) auto;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(156, 218, 236, 0.09);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        padding: 8px 10px;
      }

      .nhlcal-player-mini-row > span,
      .nhlcal-draft-mini-row > span {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: rgba(19, 216, 231, 0.1);
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
      }

      .nhlcal-player-mini-row strong,
      .nhlcal-player-mini-row small,
      .nhlcal-draft-mini-row strong,
      .nhlcal-draft-mini-row small {
        display: block;
        min-width: 0;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }

      .nhlcal-player-mini-row strong,
      .nhlcal-draft-mini-row strong {
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      .nhlcal-player-mini-row small,
      .nhlcal-draft-mini-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .nhlcal-player-mini-row em,
      .nhlcal-draft-mini-row em {
        color: var(--cyan);
        font-style: normal;
        font-size: 14px;
        font-weight: 1000;
      }

      .nhlcal-draft-mini-row em.up {
        color: var(--green);
      }

      .nhlcal-draft-mini-row em.down {
        color: var(--red);
      }

      .nhlcal-office-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-office-metric {
        min-height: 70px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        text-align: center;
        padding: 10px;
      }

      .nhlcal-office-metric span,
      .nhlcal-office-metric strong {
        display: block;
      }

      .nhlcal-office-metric span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .nhlcal-office-metric strong {
        margin-top: 5px;
        color: var(--text);
        font-size: 18px;
        font-weight: 1000;
      }

      .nhlcal-modal {
        width: min(520px, 94vw);
        border: 1px solid var(--line-strong);
        border-radius: 16px;
        background:
          radial-gradient(circle at 10% 0%, rgba(19, 216, 231, 0.14), transparent 35%),
          linear-gradient(180deg, rgba(9, 27, 42, 0.98), rgba(3, 11, 18, 0.98));
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.58);
        overflow: hidden;
      }

      .nhlcal-modal header {
        min-height: 86px;
        padding: 21px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
      }

      .nhlcal-modal header p,
      .nhlcal-modal header h2 {
        margin: 0;
      }

      .nhlcal-modal header p {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.14em;
      }

      .nhlcal-modal header h2 {
        margin-top: 5px;
        font-size: 25px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .nhlcal-settings-list {
        padding: 18px 21px;
        display: grid;
        gap: 10px;
      }

      .nhlcal-settings-list button {
        min-height: 54px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        color: var(--text);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 0 14px;
        cursor: pointer;
      }

      .nhlcal-settings-list button.is-active {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.08);
      }

      .nhlcal-settings-list span {
        font-size: 13px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlcal-settings-list strong {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-settings-summary {
        border-top: 1px solid var(--line);
        padding: 18px 21px 21px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlcal-settings-summary article {
        min-height: 64px;
        border: 1px solid rgba(156, 218, 236, 0.1);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        text-align: center;
        padding: 10px;
      }

      .nhlcal-settings-summary span,
      .nhlcal-settings-summary strong {
        display: block;
      }

      .nhlcal-settings-summary span {
        color: var(--muted);
        font-size: 10px;
        font-weight: 1000;
        text-transform: uppercase;
      }

      .nhlcal-settings-summary strong {
        color: var(--text);
        font-size: 13px;
        font-weight: 1000;
      }

      @media (max-width: 1480px) {
        .nhlcal-topbar {
          grid-template-columns: 1fr;
          min-height: auto;
          gap: 16px;
        }

        .nhlcal-action-cluster {
          justify-self: stretch;
          justify-content: flex-start;
        }

        .nhlcal-month-control {
          text-align: left;
        }

        .nhlcal-month-row {
          justify-content: flex-start;
        }

        .nhlcal-stat-strip {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .nhlcal-content-grid,
        .nhlcal-bottom-grid {
          grid-template-columns: 1fr;
        }

        .nhlcal-right-rail {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          align-items: start;
        }

        .nhlcal-preview-card {
          grid-column: 1 / -1;
        }
      }

      @media (max-width: 1080px) {
        .nhlcal-root {
          grid-template-columns: 1fr;
        }

        .nhlcal-sidebar {
          min-height: auto;
          height: auto;
          flex-direction: row;
          border-right: 0;
          border-bottom: 1px solid var(--line);
          overflow-x: auto;
        }

        .nhlcal-brand-button,
        .nhlcal-settings-button {
          width: 86px;
          height: 76px;
          flex: 0 0 auto;
          border-bottom: 0;
          border-top: 0;
          border-right: 1px solid var(--line);
        }

        .nhlcal-side-nav {
          flex-direction: row;
          padding: 0;
        }

        .nhlcal-side-button {
          width: 86px;
          min-height: 76px;
          flex: 0 0 auto;
        }

        .nhlcal-side-button.is-active::before {
          top: auto;
          right: 12px;
          left: 12px;
          bottom: 0;
          width: auto;
          height: 3px;
        }

        .nhlcal-main {
          height: calc(100vh - 76px);
          padding: 18px;
        }

        .nhlcal-stat-strip {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlcal-content-grid,
        .nhlcal-bottom-grid,
        .nhlcal-right-rail {
          grid-template-columns: 1fr;
        }

        .nhlcal-standings-card--extended {
          min-height: 280px;
        }

        .nhlcal-calendar-footer {
          align-items: flex-start;
          flex-direction: column;
        }

        .nhlcal-calendar-actions {
          justify-content: flex-start;
        }

        .nhlcal-month-grid {
          min-width: 920px;
        }

        .nhlcal-calendar-panel {
          overflow-x: auto;
        }

        .nhlcal-week-header {
          min-width: 920px;
        }
      }

      @media (max-width: 720px) {
        .nhlcal-main {
          padding: 14px;
        }

        .nhlcal-team-identity h1 {
          font-size: 30px;
        }

        .nhlcal-month-row h2 {
          font-size: 30px;
          letter-spacing: 0.12em;
        }

        .nhlcal-action-cluster {
          gap: 8px;
        }

        .nhlcal-date-chip,
        .nhlcal-online-chip {
          display: none;
        }

        .nhlcal-advance-button {
          min-width: 150px;
        }

        .nhlcal-stat-strip {
          grid-template-columns: 1fr;
        }

        .nhlcal-diagnostic-grid,
        .nhlcal-insight-row,
        .nhlcal-snapshot-grid,
        .nhlcal-opponent-strip,
        .nhlcal-event-effect-grid,
        .nhlcal-settings-summary {
          grid-template-columns: 1fr;
        }

        .nhlcal-matchup-stage {
          grid-template-columns: 1fr;
          gap: 18px;
        }

        .nhlcal-drawer {
          width: 100vw;
        }

        .nhlcal-drawer-tabs {
          grid-template-columns: repeat(5, minmax(88px, 1fr));
          overflow-x: auto;
        }

        .nhlcal-drawer-grid,
        .nhlcal-office-grid {
          grid-template-columns: 1fr;
        }

        .nhlcal-event-modal-head {
          grid-template-columns: 48px minmax(0, 1fr) 38px;
          padding: 18px;
        }

        .nhlcal-event-modal-head h2 {
          font-size: 20px;
        }

        .nhlcal-injury-report-modal {
          width: 96vw;
        }

        .nhlcal-injury-report-body {
          overflow-x: auto;
        }

        .nhlcal-injury-table {
          min-width: 720px;
        }
      }
    `}</style>
  );
}

export default CalendarScreen;