import React, { useMemo, useState, useEffect, useCallback } from "react";
import "../styles/game-ui.css";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { api } from "../services/api";
import { formatProspectLeague, formatProspectTeam } from "../events/prospectDevelopment/prospectDevelopmentHelpers";
import { applyProspectLeagueTeamFix } from "../data/prospectLeagueTeams";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { flagApiUrl } from "../utils/countryFlags";

let TRANSCENDENT_BOSS_AUDIO_URL = null;
try {
  // Optional asset — safe when missing from build.
  TRANSCENDENT_BOSS_AUDIO_URL = require("../soundtrack/Super Mario 64 Soundtrack - Stage Boss - RadiatorRampardos.mp3");
} catch {
  TRANSCENDENT_BOSS_AUDIO_URL = null;
}

function playTranscendentBossSting() {
  if (!TRANSCENDENT_BOSS_AUDIO_URL) return;
  if (typeof window === "undefined") return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  try {
    const audio = new Audio(TRANSCENDENT_BOSS_AUDIO_URL);
    audio.volume = 0.42;
    audio.loop = false;
    const p = audio.play();
    if (p && typeof p.catch === "function") {
      p.catch(() => {});
    }
  } catch {
    // Missing codec / autoplay blocked — ignore.
  }
}

/** Backend ETA is often `{ label, years, confidence }` — never render the object as a React child. */
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

function displaySafeText(value) {
  if (value == null || value === "") return null;
  if (typeof value === "object") return formatNhlEta(value, null);
  return value;
}


const BOARD_NAV_ITEMS = [
  { key: "rank", label: "RANK", title: "Rank", icon: "rank" },
  { key: "forwards", label: "FWD", title: "Forwards", icon: "fwd" },
  { key: "defensemen", label: "D-MEN", title: "Defensemen", icon: "dmen" },
  { key: "goalies", label: "G", title: "Goalies", icon: "goalie" },
];

function normalizeProspectPosition(pos) {
  return String(pos || "").trim().toUpperCase();
}

function isForwardPosition(pos) {
  const p = normalizeProspectPosition(pos);
  return ["C", "LW", "RW", "F", "W"].includes(p) || p.includes("FORWARD");
}

function isDefensemanPosition(pos) {
  const p = normalizeProspectPosition(pos);
  return ["D", "LD", "RD", "LHD", "RHD"].includes(p) || p.includes("DEF");
}

function isGoaliePosition(pos) {
  const p = normalizeProspectPosition(pos);
  return p === "G" || p.includes("GOAL");
}

const PROFILE_TABS = ["OVERVIEW", "STATS", "ATTRIBUTES", "SCOUT REPORT", "CHARACTER"];
const LEAGUES = ["OHL", "WHL", "QMJHL", "NCAA", "USHL", "SHL", "LIIGA", "DEL", "CZECHIA"];
const COUNTRIES = ["Canada", "United States", "Sweden", "Finland", "Czechia", "Slovakia", "Germany", "Switzerland"];
const SCOUT_NAMES = [
  "Mike Brennan", "Sarah Chen", "Erik Lindholm", "Marc Dubois",
  "James Okafor", "Anna Kowalski", "Tyler Morrison", "Lisa Bergstrom",
];
const REPORT_TYPES = [
  { key: "potential", label: "Potential Comparison" },
  { key: "skills", label: "Skills Assessment" },
  { key: "style", label: "Playing Style" },
  { key: "strengths", label: "Strengths/Weaknesses" },
  { key: "character", label: "Character/Interviews" },
];
/** Client fallback when backend stats missing — mirrors junior inflation vs pro leagues. */
const LEAGUE_PPG_FALLBACK = {
  OHL: { min: 0.48, max: 2.4 },
  WHL: { min: 0.45, max: 2.2 },
  QMJHL: { min: 0.5, max: 2.5 },
  USHL: { min: 0.4, max: 1.7 },
  NCAA: { min: 0.32, max: 1.4 },
  SHL: { min: 0.18, max: 0.98 },
  LIIGA: { min: 0.22, max: 1.08 },
  DEL: { min: 0.2, max: 0.95 },
  CZECHIA: { min: 0.2, max: 0.95 },
  DEFAULT: { min: 0.38, max: 1.85 },
};

const LEAGUE_TEAMS = {
  OHL: ["London Knights", "Ottawa 67's", "Windsor Spitfires", "Saginaw Spirit", "Kitchener Rangers", "Guelph Storm"],
  WHL: ["Seattle Thunderbirds", "Portland Winterhawks", "Edmonton Oil Kings", "Kelowna Rockets", "Calgary Hitmen"],
  QMJHL: ["Quebec Remparts", "Halifax Mooseheads", "Moncton Wildcats", "Shawinigan Cataractes", "Rimouski Oceanic"],
  NCAA: ["Boston College", "Michigan", "North Dakota", "Denver", "Quinnipiac", "Boston University"],
  USHL: ["Dubuque Fighting Saints", "Omaha Lancers", "Fargo Force", "Chicago Steel"],
  SHL: ["HV71", "Frolunda HC", "Leksands IF", "Farjestad BK"],
  LIIGA: ["Tappara", "HIFK", "Karpat", "Ilves"],
  DEL: ["Eisbaren Berlin", "Adler Mannheim", "Kolner Haie"],
  CZECHIA: ["HC Kometa Brno", "Sparta Praha", "HC Olomouc"],
};
const EMPTY_SCOUTING_META = Object.freeze({
  watchlist: false,
  target: false,
  doNotDraft: false,
  assignedScout: null,
  lastViewed: null,
  requestedReports: {},
});

const OVR_REVEAL_THRESHOLD = 72;

const SCOUTING_ENDPOINTS = Object.freeze({
  prospects: "/api/franchise/scouting/prospects",
  focus: "/api/franchise/scouting/focus",
  interview: "/api/franchise/scouting/interview",
});

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function seededNumber(seed) {
  let value = seed;
  value = (value ^ 61) ^ (value >>> 16);
  value += value << 3;
  value ^= value >>> 4;
  value *= 0x27d4eb2d;
  value ^= value >>> 15;
  return Math.abs(value);
}

function pick(list, seed) {
  return list[seededNumber(seed) % list.length];
}

function projectionForRank(rank) {
  if (rank <= 3) return "TOP 3";
  if (rank <= 5) return "TOP 5";
  if (rank <= 10) return "TOP 10";
  if (rank <= 20) return "TOP 20";
  if (rank <= 32) return "1ST RD";
  if (rank <= 64) return "2ND RD";
  if (rank <= 96) return "3RD RD";
  return "LATE RD";
}

function talentGrade(rank, seed) {
  const swing = seededNumber(seed) % 4;
  if (rank <= 2) return swing > 1 ? "A+" : "A";
  if (rank <= 5) return swing > 1 ? "A" : "A-";
  if (rank <= 12) return swing > 1 ? "A-" : "B+";
  if (rank <= 25) return swing > 1 ? "B+" : "B";
  if (rank <= 50) return swing > 1 ? "B" : "B-";
  if (rank <= 80) return swing > 1 ? "B-" : "C+";
  return swing > 1 ? "C+" : "C";
}

/**
 * Nation mark for a prospect. Scouting files identify players by federation
 * code, so this returns the three-letter code used on the board rather than a
 * platform-dependent flag glyph. Flag artwork is handled separately by
 * flagApiUrl where a real image is available.
 */
function countryFlag(country) {
  const map = {
    Canada: "CAN",
    CAN: "CAN",
    CA: "CAN",
    USA: "USA",
    US: "USA",
    "United States": "USA",
    "United States of America": "USA",
    Sweden: "SWE",
    SWE: "SWE",
    SE: "SWE",
    Finland: "FIN",
    FIN: "FIN",
    FI: "FIN",
    Czechia: "CZE",
    "Czech Republic": "CZE",
    CZE: "CZE",
    CZ: "CZE",
    Slovakia: "SVK",
    SVK: "SVK",
    SK: "SVK",
    Germany: "GER",
    GER: "GER",
    DE: "GER",
    Switzerland: "SUI",
    SUI: "SUI",
    CH: "SUI",
    Russia: "RUS",
    RUS: "RUS",
    RU: "RUS",
  };
  const raw = String(country || "").trim();
  if (!raw) return "—";
  if (map[raw]) return map[raw];
  if (map[raw.toUpperCase()]) return map[raw.toUpperCase()];
  const iso = resolveCountryCode(raw);
  if (iso && map[iso]) return map[iso];
  if (iso && /^[A-Z]{2,3}$/.test(iso)) return iso;
  return "—";
}

function humanizeScoutReason(raw) {
  if (raw == null || raw === "") return null;
  const s = String(raw).trim();
  if (!s) return null;
  const COPY = {
    rebuild_needs_ceiling: "Rebuild pathway — club needs high-upside talent",
    rebuild_needs_ready: "Rebuild pathway — needs earlier NHL contributions",
    contender_needs_ready: "Contender fit — timeline matches NHL-ready help",
    contender_needs_ceiling: "Contender fit — still values long-term upside",
    depth_need: "Fills a clear organizational depth need",
    position_need: "Addresses a positional roster gap",
    timeline_mismatch: "Timeline mismatch with current competitive window",
    surplus_position: "Position is already deep on the depth chart",
  };
  if (COPY[s]) return COPY[s];
  if (/^[a-z0-9]+(_[a-z0-9]+)+$/i.test(s)) {
    return s
      .split("_")
      .filter(Boolean)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(" ");
  }
  return s;
}

function countryDisplayLabel(player, profile) {
  return (
    profile?.country
    || player?.country
    || player?.nationality
    || normalizeCountryCode(player)
    || profile?.country_code
    || "—"
  );
}

const COUNTRY_NAME_TO_ISO = {
  CAN: "CA",
  CANADA: "CA",
  Canada: "CA",
  USA: "US",
  US: "US",
  "UNITED STATES": "US",
  "United States": "US",
  "United States of America": "US",
  SWE: "SE",
  SWEDEN: "SE",
  Sweden: "SE",
  FIN: "FI",
  FINLAND: "FI",
  Finland: "FI",
  RUS: "RU",
  RUSSIA: "RU",
  Russia: "RU",
  "Russian Federation": "RU",
  CZE: "CZ",
  CZECHIA: "CZ",
  "CZECH REPUBLIC": "CZ",
  Czechia: "CZ",
  "Czech Republic": "CZ",
  SVK: "SK",
  SLOVAKIA: "SK",
  Slovakia: "SK",
  SUI: "CH",
  SWITZERLAND: "CH",
  Switzerland: "CH",
  GER: "DE",
  GERMANY: "DE",
  Germany: "DE",
  NOR: "NO",
  NORWAY: "NO",
  Norway: "NO",
  DEN: "DK",
  DENMARK: "DK",
  Denmark: "DK",
  LAT: "LV",
  LATVIA: "LV",
  Latvia: "LV",
  AUT: "AT",
  AUSTRIA: "AT",
  Austria: "AT",
  KAZ: "KZ",
  KAZAKHSTAN: "KZ",
  Kazakhstan: "KZ",
  BLR: "BY",
  BELARUS: "BY",
  Belarus: "BY",
  UKR: "UA",
  UKRAINE: "UA",
  Ukraine: "UA",
  POL: "PL",
  POLAND: "PL",
  Poland: "PL",
  FRA: "FR",
  FRANCE: "FR",
  France: "FR",
  GBR: "GB",
  "UNITED KINGDOM": "GB",
  "Great Britain": "GB",
  "United Kingdom": "GB",
  JPN: "JP",
  JAPAN: "JP",
  Japan: "JP",
};

function resolveCountryCode(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  if (/^[A-Za-z]{2}$/.test(s)) return s.toUpperCase();
  const upper = s.toUpperCase();
  if (COUNTRY_NAME_TO_ISO[upper]) return COUNTRY_NAME_TO_ISO[upper];
  if (COUNTRY_NAME_TO_ISO[s]) return COUNTRY_NAME_TO_ISO[s];
  return null;
}

function normalizeCountryCode(player) {
  if (!player || typeof player !== "object") return null;
  const candidates = [
    player.country_code,
    player.countryCode,
    player.nationality_code,
    player.nationalityCode,
    player.nationality,
    player.country,
    player.birth_country,
    player.birthCountry,
  ];
  for (const raw of candidates) {
    const code = resolveCountryCode(raw);
    if (code) return code;
  }
  return null;
}

function prospectRank(player, index = 0) {
  return (
    Number(player?.rank)
    || Number(player?.draft_rank)
    || Number(player?.draftRank)
    || Number(player?.consensus_rank)
    || Number(player?.consensusRank)
    || index + 1
  );
}

function prospectPpgValue(player) {
  const provided = Number(player?.ppg);
  if (Number.isFinite(provided)) return provided;
  const gp = Number(player?.gp ?? player?.gamesPlayed ?? player?.games_played) || 0;
  const pts = Number(player?.points) || 0;
  return gp > 0 ? pts / gp : 0;
}

function formatHandedness(handedness) {
  const h = String(handedness || "").trim().toLowerCase();
  if (!h) return null;
  if (h.startsWith("l")) return "L";
  if (h.startsWith("r")) return "R";
  return h.slice(0, 1).toUpperCase();
}

function cleanTeamName(player, profile) {
  const raw = profile?.team || player?.team || player?.teamName || "";
  if (profile?.team) return profile.team;
  return formatProspectTeam(
    { team_name: raw, team: raw, league_code: player?.leagueCode, league: player?.league },
    raw
  );
}

function cleanLeagueName(player, profile) {
  if (profile?.league) return profile.league;
  return formatProspectLeague({
    league_display: player?.leagueDisplay,
    league_code: player?.leagueCode,
    league: player?.league,
    league_name: player?.league,
    team_name: player?.team || player?.teamName,
    team: player?.team,
  }) || player?.league || "—";
}

function getPlayerIdentityBadges(player, profile) {
  const fromBackend = profile?.identity_badges;
  if (fromBackend && typeof fromBackend === "object") {
    return {
      position: fromBackend.position || player?.position || "—",
      handedness: fromBackend.handedness || formatHandedness(profile?.handedness || player?.handedness),
      height: fromBackend.height || profile?.height || player?.height,
      weight: fromBackend.weight || (profile?.weight ? `${Math.round(Number(profile.weight))} LBS` : null),
      age: fromBackend.age || (profile?.age ? `${profile.age}Y` : player?.age ? `${player.age}Y` : null),
    };
  }
  const weight = profile?.weight ?? player?.weight;
  return {
    position: player?.position || profile?.position || "—",
    handedness: formatHandedness(profile?.handedness || player?.handedness),
    height: profile?.height || player?.height,
    weight: weight != null && Number.isFinite(Number(weight)) ? `${Math.round(Number(weight))} LBS` : null,
    age: profile?.age ? `${profile.age}Y` : player?.age ? `${player.age}Y` : null,
  };
}

function getPlayerTags(player, profile) {
  if (profile?.ceilingHidden || profile?.potential?.hidden) {
    if (Array.isArray(profile?.tags) && profile.tags.length) return profile.tags;
    return [];
  }
  if (Array.isArray(profile?.tags) && profile.tags.length) return profile.tags;
  const tags = [];
  const pos = String(player?.position || profile?.position || "").toUpperCase();
  const ppg = Number(profile?.stats?.ppg ?? player?.ppg) || 0;
  const gp = Number(profile?.stats?.games ?? player?.gp) || 0;
  if (gp < 15) return tags.slice(0, 5);
  if (ppg >= 0.9) tags.push("Production");
  if (player?.isGem || profile?.gem?.label === "Gem") tags.push("Gem");
  if (player?.isBustRisk || profile?.gem?.label === "Risk") tags.push("Boom/Bust");
  if (pos !== "G" && pos !== "D" && ppg >= 0.55) tags.push("Goal Scorer");
  return tags.slice(0, 5);
}

function movementIndicator(stock) {
  const tone = getStockTone(stock);
  if (tone === "rise") return { glyph: "↑", text: stockBadgeText(stock), cls: "is-rise" };
  if (tone === "fall") return { glyph: "↓", text: stockBadgeText(stock), cls: "is-fall" };
  if (tone === "new") return { glyph: "◎", text: stockBadgeText(stock), cls: "is-new" };
  if (stock?.available && tone === "stable") return { glyph: "—", text: stockBadgeText(stock), cls: "is-stable" };
  return { glyph: "—", text: "—", cls: "is-muted" };
}

function movementDisplayText(stock) {
  if (!stock?.available && stock?.direction === "UNKNOWN") return "—";
  return stockBadgeText(stock, { compact: true });
}

function ProspectMetric({ label, value, align = "left", tone = "", valueStyle = null, title = null }) {
  return (
    <div
      className={`dc-prospect-metric dc-prospect-metric--${align}${tone ? ` ${tone}` : ""}`}
      title={title || undefined}
    >
      <span className="dc-prospect-metric__value" style={valueStyle || undefined}>{value}</span>
      <span className="dc-prospect-metric__label">{label}</span>
    </div>
  );
}

function prospectScoutingPct(player) {
  const n = Number(coalesce(player?.scoutingConfidence, player?.completion, player?.scouting_confidence));
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.min(100, Math.max(0, n)));
}

function prospectIntelTier(pct) {
  if (pct == null) return null;
  if (pct <= 35) return "Unknown";
  if (pct <= 55) return "Limited";
  if (pct <= 75) return "Solid";
  if (pct <= 90) return "Strong";
  return "Locked";
}

function prospectConfidenceFogClass(pct) {
  if (pct == null) return "dc-conf-fog--blind";
  if (pct >= 91) return "dc-conf-fog--locked";
  if (pct >= 76) return "dc-conf-fog--strong";
  if (pct >= 56) return "dc-conf-fog--solid";
  if (pct >= 36) return "dc-conf-fog--limited";
  if (pct >= 15) return "dc-conf-fog--unknown";
  return "dc-conf-fog--blind";
}

/**
 * Ceiling tiers reuse peakProjectionBand's thresholds so the board column and the
 * profile panel never disagree about what a potential number is worth. The ramp runs
 * cold to hot — slate, steel, blue, ice, cyan, gold — and the top two tiers earn a
 * filled chip so a franchise ceiling is unmistakable while scanning 300+ rows.
 */
const POTENTIAL_TIERS = [
  { min: 92, key: "generational", label: "Generational" },
  { min: 88, key: "elite", label: "Elite" },
  { min: 84, key: "high", label: "High upside" },
  { min: 80, key: "strong", label: "Strong upside" },
  { min: 75, key: "moderate", label: "Moderate upside" },
  { min: 70, key: "depth", label: "Depth upside" },
  { min: 0, key: "fringe", label: "Limited upside" },
];

function potentialTier(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return POTENTIAL_TIERS.find((t) => n >= t.min) || null;
}

/**
 * Projected ceiling for the board column. Fog controls precision — an unscouted kid
 * shows a range, never a settled grade — but the tier ramp reads off whatever the
 * scouts will commit to, so a range still carries its colour.
 */
function resolvePotentialEstimate(player) {
  const blank = (detail) => ({ text: "—", value: null, exact: false, tier: null, detail });
  if (player?.ceilingHidden) return blank("Ceiling unreported — assign a scout");

  const pct = prospectScoutingPct(player);
  if (pct == null || pct < 15) return blank("Ceiling not scouted");

  const range = player?.potentialRange;
  const score = Number(player?.potentialScore);
  const hasScore = Number.isFinite(score) && score > 0;
  const lo = range?.low != null ? Math.round(Number(range.low)) : NaN;
  const hi = range?.high != null ? Math.round(Number(range.high)) : NaN;
  const hasRange = Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo;

  // A grade only settles once the file is deep enough, or the band has closed to a
  // few points. Everything else stays a range so the colour can't oversell a guess.
  if (hasScore && (pct >= 91 || (pct >= 76 && hasRange && hi - lo <= 5))) {
    const v = Math.round(score);
    return { text: String(v), value: v, exact: true, tier: potentialTier(v), detail: "Projected ceiling" };
  }
  if (hasRange && hi > lo) {
    const mid = (lo + hi) / 2;
    return { text: `${lo}–${hi}`, value: mid, exact: false, tier: potentialTier(mid), detail: "Ceiling range" };
  }
  if (hasRange) {
    return { text: String(lo), value: lo, exact: false, tier: potentialTier(lo), detail: "Ceiling range" };
  }
  if (hasScore) {
    const v = Math.round(score);
    return { text: String(v), value: v, exact: false, tier: potentialTier(v), detail: "Ceiling estimate" };
  }
  const label = coalesce(player?.potentialLabel, player?.potential_label);
  if (label) return { text: String(label), value: null, exact: false, tier: null, detail: "Scout grade" };
  return blank("Ceiling not scouted");
}

function ScoutConfidenceMetric({ player }) {
  const pct = prospectScoutingPct(player);
  const tier = prospectIntelTier(pct) || player?.intelLabel || null;
  const fogClass = prospectConfidenceFogClass(pct);
  const title = pct != null
    ? `${tier || "Scouting"} · ${pct}% confidence`
    : "Scouting confidence unknown";

  return (
    <div
      className={`dc-prospect-metric dc-prospect-metric--center is-scout dc-scout-confidence ${fogClass}`}
      title={title}
    >
      <span className="dc-prospect-metric__value">{pct != null ? `${pct}%` : "—"}</span>
      <span className="dc-scout-confidence__bar" aria-hidden="true">
        <span className="dc-scout-confidence__fill" style={{ width: `${pct ?? 0}%` }} />
      </span>
    </div>
  );
}

function ProspectBoardColumnHeader() {
  return (
    <div className="dc-prospect-board__columns" aria-hidden="true">
      <span className="dc-prospect-board__col dc-prospect-board__col--rank">#</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--player">Player</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--pos">Pos</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--league">League</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--pot">Pot</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--proj">Proj</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--conf">Conf</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--stock">Stock</span>
    </div>
  );
}

function boardFilterLabel(activeBoardView) {
  const item = BOARD_NAV_ITEMS.find((x) => x.key === activeBoardView);
  return item && activeBoardView !== "rank" ? item.title : null;
}

function regionForCountry(country) {
  if (country === "Canada" || country === "United States") return "NORTH AMERICA";
  if (["Sweden", "Finland", "Czechia", "Slovakia", "Germany", "Switzerland"].includes(country)) return "EUROPE";
  return "INTERNATIONAL";
}

function parseMonthFromDate(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  const iso = s.match(/^(\d{4})-(\d{2})/);
  if (iso) return Number(iso[2]);
  const named = s.match(/\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/i);
  if (named) {
    const map = { january: 1, february: 2, march: 3, april: 4, may: 5, june: 6, july: 7, august: 8, september: 9, october: 10, november: 11, december: 12 };
    return map[named[1].toLowerCase()] || null;
  }
  return null;
}

function getFranchiseDateRaw(franchiseState) {
  return (
    franchiseState?.nhl_today?.iso ||
    franchiseState?.nhl_today?.date_label ||
    franchiseState?.current_date ||
    franchiseState?.currentDate ||
    franchiseState?.calendar?.current_date ||
    ""
  );
}

function getGPRangeForMonth(month) {
  if (!month || month === 8) return [0, 12];
  if (month === 9 || month === 10) return [0, 12];
  if (month === 11) return [8, 24];
  if (month === 12) return [15, 32];
  if (month === 1) return [25, 42];
  if (month === 2) return [35, 52];
  if (month === 3 || month === 4) return [45, 68];
  if (month >= 5 && month <= 7) return [50, 72];
  return [0, 12];
}

function getScoutingPeriodLabel(month) {
  if (!month) return "Early Season Draft Board";
  if (month >= 9 && month <= 10) return "Early Season Draft Board";
  if (month === 11) return "November Scouting Update";
  if (month === 12 || month === 1) return "Midseason Draft Board";
  if (month === 2 || month === 3) return "Late Season Draft Board";
  if (month >= 4 && month <= 6) return "Final Draft Rankings";
  return "Early Season Draft Board";
}

function getMaxCompletionForMonth(month) {
  if (!month || month <= 10) return 48;
  if (month === 11) return 58;
  if (month === 12) return 65;
  if (month === 1) return 72;
  if (month === 2) return 78;
  if (month === 3) return 85;
  if (month >= 4 && month <= 6) return 95;
  return 48;
}

function maxCompletionForRank(rank, monthMax) {
  if (rank <= 5) return Math.min(95, monthMax + 8);
  if (rank <= 15) return Math.min(90, monthMax);
  if (rank <= 32) return Math.min(82, monthMax - 6);
  return Math.min(70, monthMax - 14);
}

function buildDateContext(franchiseState) {
  const raw = getFranchiseDateRaw(franchiseState);
  const month = parseMonthFromDate(raw);
  const monthMax = getMaxCompletionForMonth(month);
  const [gpMin, gpMax] = getGPRangeForMonth(month);
  const seasonYear = Number(franchiseState?.season_year) || new Date().getFullYear();
  return {
    raw,
    month,
    monthMax,
    gpMin,
    gpMax,
    statsThrough: raw || "Early Season",
    periodLabel: getScoutingPeriodLabel(month),
    draftYear: seasonYear + 1,
    isPartialSeason: !month || month < 4 || month > 6,
  };
}

function pickTeamName(league, seed) {
  const teams = LEAGUE_TEAMS[league] || LEAGUE_TEAMS.OHL;
  return teams[seededNumber(seed + 99) % teams.length];
}

function generatePartialStats(position, rank, gp, seed, league = "OHL") {
  if (position === "G") {
    const wins = clamp(Math.floor(gp * 0.52) + (seededNumber(seed + 14) % 4) - 1, 0, gp);
    const savePct = (0.885 + (seededNumber(seed + 15) % 45) / 1000).toFixed(3);
    const gaa = (2.05 + (seededNumber(seed + 16) % 85) / 100).toFixed(2);
    return { gp, goals: 0, assists: 0, points: 0, wins, savePct, gaa };
  }
  const profile = LEAGUE_PPG_FALLBACK[league] || LEAGUE_PPG_FALLBACK.DEFAULT;
  const rankBoost = (100 - rank) / 120;
  const rate = clamp(
    0.42 + rankBoost + (seededNumber(seed + 12) % 30) / 100,
    profile.min,
    profile.max
  );
  const points = clamp(Math.round(gp * rate), 0, Math.round(gp * profile.max));
  const goalShare = 0.38 + (seededNumber(seed + 13) % 18) / 100;
  const goals = clamp(Math.round(points * goalShare), 0, points);
  const assists = Math.max(0, points - goals);
  return { gp, goals, assists, points, wins: 0, savePct: null, gaa: null };
}

function addPositionRanks(prospects) {
  const counters = {};
  return [...prospects]
    .sort((a, b) => a.rank - b.rank)
    .map((p) => {
      counters[p.position] = (counters[p.position] || 0) + 1;
      return { ...p, positionRank: counters[p.position] };
    });
}

function confidenceLabel(completion) {
  if (completion <= 30) return "Unknown";
  if (completion <= 55) return "Preliminary";
  if (completion <= 80) return "Moderate Confidence";
  if (completion < 95) return "High Confidence";
  return "Fully Scouted";
}

function attributeDisplay(exactValue, completion, attrSeed = 0, { wideFog = false } = {}) {
  const v = Number(exactValue);
  const hasVal = Number.isFinite(v) && v > 0;
  const base = hasVal ? v : 50;
  const fogBoost = wideFog ? 1.65 : 1.0;
  // Only fully lock when there is essentially no scouting file at all.
  if (completion <= 18) {
    return { text: "?", range: null, width: 0, locked: true, confidence: "Unknown" };
  }
  // Ceiling fog / early looks → banded ranges, never blank "?" when we have a tool value.
  if (completion <= 55 || wideFog) {
    const spread = Math.round(12 * fogBoost);
    const low = clamp(base - spread + (seededNumber(base + attrSeed) % 5), 40, 96);
    const high = clamp(low + spread + (wideFog ? 4 : 0), low, 96);
    return { text: `${low}–${high}`, range: [low, high], width: (low + high) / 2, locked: false, confidence: "Preliminary" };
  }
  if (completion <= 80) {
    const spread = 6;
    const low = clamp(base - spread + (seededNumber(base + attrSeed + 7) % 3), 45, 96);
    const high = clamp(low + spread, low, 96);
    return { text: `${low}–${high}`, range: [low, high], width: (low + high) / 2, locked: false, confidence: "Moderate Confidence" };
  }
  if (completion < 95) {
    const spread = 2;
    const low = clamp(base - spread, 45, 96);
    const high = clamp(base + spread, low, 96);
    return { text: `${low}–${high}`, range: [low, high], width: base, locked: false, confidence: "High Confidence" };
  }
  return { text: String(Math.round(base)), range: [base, base], width: base, locked: false, confidence: "Fully Scouted" };
}

function inferTeamNeeds(franchiseState) {
  const roster = franchiseState?.roster;
  if (!Array.isArray(roster) || !roster.length) return null;
  const counts = { C: 0, W: 0, D: 0, G: 0 };
  roster.forEach((p) => {
    const pos = String(p.position || p.pos || "").toUpperCase();
    if (pos === "G" || pos.includes("GOAL")) counts.G += 1;
    else if (pos === "D" || pos.includes("DEF")) counts.D += 1;
    else if (pos === "C") counts.C += 1;
    else if (pos === "LW" || pos === "RW" || pos === "W") counts.W += 1;
  });
  const needs = [];
  if (counts.G < 2) needs.push("Goalie");
  if (counts.D < 6) needs.push("Defensive D");
  if (counts.C < 3) needs.push("C");
  if (counts.W < 4) needs.push("Scoring Winger");
  if (counts.D < 8) needs.push("RHD");
  return needs.length ? needs : null;
}

function defaultProspectStub(rank) {
  const seed = seededNumber(`stub-${rank}`);
  return {
    id: `prospect-${rank}`,
    firstName: "Prospect",
    lastName: "Player",
    position: "C",
    country: "Canada",
    region: "NORTH AMERICA",
    league: "OHL",
    team: "—",
    playerType: "Two-Way Forward",
    projection: projectionForRank(rank),
    talent: talentGrade(rank, seed),
    completion: 30,
    stock: 0,
    gp: 0,
    goals: 0,
    assists: 0,
    points: 0,
    height: "6'0\"",
    weight: 185,
    age: 18,
    handedness: "Right",
    birthday: "",
    morale: 70,
    character: 70,
    fit: 65,
    compete: 65,
    leadership: 60,
    workEthic: 65,
    coachability: 65,
    consistency: 60,
    poise: 60,
    skating: 60,
    shooting: 60,
    passing: 60,
    defense: 58,
    physical: 58,
    hockeyIQ: 62,
    isGem: false,
    isBustRisk: false,
  };
}

function scoutingMetaFromProspect(prospect) {
  if (!prospect || typeof prospect !== "object") return { ...EMPTY_SCOUTING_META };
  return {
    watchlist: Boolean(prospect.watchlist),
    target: Boolean(prospect.target),
    doNotDraft: Boolean(prospect.do_not_draft || prospect.doNotDraft),
    assignedScout: prospect.assigned_scout || prospect.assignedScout || null,
    lastViewed: null,
    requestedReports: prospect.requested_reports || prospect.requestedReports || {},
    notes: Array.isArray(prospect.notes) ? prospect.notes : [],
  };
}

function scoutingStoreFromApiProspects(prospects) {
  const store = {};
  (Array.isArray(prospects) ? prospects : []).forEach((prospect) => {
    const id = String(prospect?.id || prospect?.key || "");
    if (!id) return;
    store[id] = scoutingMetaFromProspect(prospect);
  });
  return store;
}

function scoutingStoreFromFranchiseState(franchiseState) {
  const overlays = franchiseState?.scouting_state?.prospects;
  if (!overlays || typeof overlays !== "object") return {};
  const store = {};
  Object.entries(overlays).forEach(([id, overlay]) => {
    if (!overlay || typeof overlay !== "object") return;
    store[id] = {
      watchlist: Boolean(overlay.watchlist),
      target: Boolean(overlay.target),
      doNotDraft: Boolean(overlay.do_not_draft || overlay.doNotDraft),
      assignedScout: overlay.assigned_scout || overlay.assignedScout || null,
      lastViewed: overlay.last_viewed || overlay.lastViewed || null,
      requestedReports: overlay.requested_reports || overlay.requestedReports || {},
      notes: Array.isArray(overlay.notes) ? overlay.notes : [],
    };
  });
  return store;
}

function mergeScoutingStores(...stores) {
  return stores.reduce((acc, store) => {
    if (!store || typeof store !== "object") return acc;
    Object.entries(store).forEach(([id, meta]) => {
      acc[id] = { ...getScoutingMeta(acc, id), ...(meta || {}) };
    });
    return acc;
  }, {});
}

async function patchScoutingMeta(prospectId, metaPatch) {
  const res = await api.post(SCOUTING_ENDPOINTS.focus, {
    prospect_id: prospectId,
    target_id: prospectId,
    target_type: "player",
    context: {
      meta_only: true,
      meta_patch: metaPatch,
      prospect_id: prospectId,
    },
  });
  return res?.data || {};
}

function coalesce(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function kgToLbs(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 2.20462);
}

function normalizeLeagueCode(code) {
  const raw = String(code || "").trim();
  const map = {
    CHL_OHL: "OHL",
    CHL_WHL: "WHL",
    CHL_QMJHL: "QMJHL",
    EU_J_SHL: "J20 Nationell",
    EU_J_LIIGA: "U20 SM-sarja",
    EU_J_DEL: "DEL",
    EU_J_SWISS: "NL",
    EU_J_CZ: "Czech Extraliga",
    EU_J_SK: "Slovak Extraliga",
    EU_J_KHL_JR: "MHL",
    EU_J_NOR: "Norway",
    EU_J_DEN: "Denmark",
    EU_J_AUT: "Austria",
    USHL: "USHL",
    NCAA: "NCAA",
    OHL: "OHL",
    WHL: "WHL",
    QMJHL: "QMJHL",
    SHL: "SHL",
    LIIGA: "Liiga",
    DEL: "DEL",
    CZECHIA: "Czech Extraliga",
  };
  return map[raw] || raw || "Unknown";
}

function resolveLeagueDisplay(row, base) {
  const formatted = formatProspectLeague(row);
  if (formatted) return formatted;
  const normalized = normalizeLeagueCode(row?.league_code);
  return normalized !== "Unknown" ? normalized : coalesce(base?.league, "Unknown");
}

function hasBackendStats(row) {
  if (row?.actual_stats && typeof row.actual_stats === "object") return true;
  return row?.gp !== undefined && row?.gp !== null;
}

function resolveActualStatsRow(row) {
  const actual = row?.actual_stats;
  if (actual && typeof actual === "object") {
    return {
      ...row,
      gp: actual.gp ?? actual.games_played ?? row.gp,
      goals: actual.goals ?? row.goals,
      assists: actual.assists ?? row.assists,
      points: actual.points ?? row.points,
      ppg: actual.ppg ?? actual.points_per_game ?? row.ppg,
      pim: actual.pim ?? row.pim,
      wins: actual.wins ?? row.wins,
      losses: actual.losses ?? row.losses,
      ot_losses: actual.ot_losses ?? row.ot_losses,
      save_pct: actual.save_pct ?? row.save_pct,
      gaa: actual.gaa ?? row.gaa,
      shutouts: actual.shutouts ?? row.shutouts,
    };
  }
  return row;
}

function resolveProjectedStatsRow(row) {
  const projected = row?.projected_stats;
  if (projected && typeof projected === "object") {
    return {
      gp: projected.projected_gp ?? projected.gp ?? row.projected_gp,
      goals: projected.projected_goals ?? projected.goals ?? row.projected_goals,
      assists: projected.projected_assists ?? projected.assists ?? row.projected_assists,
      points: projected.projected_points ?? projected.points ?? row.projected_points,
      ppg: projected.projected_ppg ?? projected.ppg ?? row.projected_ppg,
      wins: projected.projected_wins ?? projected.wins ?? row.projected_wins,
      savePct: projected.projected_save_pct ?? projected.save_pct ?? row.projected_save_pct,
      gaa: projected.projected_gaa ?? projected.gaa ?? row.projected_gaa,
      shutouts: projected.projected_shutouts ?? projected.shutouts ?? row.projected_shutouts,
    };
  }
  return {
    gp: row?.projected_gp,
    goals: row?.projected_goals,
    assists: row?.projected_assists,
    points: row?.projected_points,
    ppg: row?.projected_ppg,
    wins: row?.projected_wins,
    savePct: row?.projected_save_pct,
    gaa: row?.projected_gaa,
    shutouts: row?.projected_shutouts,
  };
}

function attrFromBackend(row, base, snakeKey, camelKey) {
  const raw = coalesce(row?.[snakeKey], row?.[camelKey]);
  if (raw !== undefined && raw !== null && Number.isFinite(Number(raw))) {
    return Math.round(Number(raw));
  }
  return base[camelKey];
}

function resolveWeightLbs(row, base) {
  const explicit = coalesce(row?.weight_lbs, row?.weightLbs);
  if (explicit != null && Number.isFinite(Number(explicit))) {
    return Math.round(Number(explicit));
  }
  if (row?.weight != null && Number.isFinite(Number(row.weight))) {
    const n = Number(row.weight);
    if (n >= 120) return Math.round(n);
    const converted = kgToLbs(n);
    return converted != null ? converted : base.weight;
  }
  return base.weight;
}

function mapBackendStats(row, base, pos, rank, league = "OHL") {
  if (!hasBackendStats(row)) {
    return {
      gp: 0,
      goals: 0,
      assists: 0,
      points: 0,
      wins: 0,
      savePct: null,
      gaa: null,
      _statsFromBackend: false,
    };
  }

  const src = resolveActualStatsRow(row);
  const gp = Number(src.gp) || 0;
  if (pos === "G") {
    const saveRaw = coalesce(src.save_pct, src.savePct);
    const gaaRaw = coalesce(src.gaa);
    return {
      gp,
      goals: 0,
      assists: 0,
      points: 0,
      wins: Number(src.wins) || 0,
      losses: Number(src.losses) || 0,
      otLosses: Number(src.ot_losses) || 0,
      savePct: saveRaw != null ? String(saveRaw) : null,
      gaa: gaaRaw != null ? String(gaaRaw) : null,
      shutouts: Number(src.shutouts) || 0,
      _statsFromBackend: true,
    };
  }

  const goals = Number(src.goals) || 0;
  const assists = Number(src.assists) || 0;
  const points = src.points != null ? Number(src.points) || 0 : goals + assists;
  const ppg = gp > 0 ? Number((points / gp).toFixed(3)) : null;
  const plusRaw = coalesce(src.plus_minus, src.plusMinus);
  const shotsRaw = coalesce(src.shots, src.shots_on_goal, src.sog);
  return {
    gp,
    goals,
    assists,
    points,
    pim: src.pim != null ? Number(src.pim) || 0 : null,
    plusMinus: plusRaw != null && plusRaw !== "" ? Number(plusRaw) : null,
    shots: shotsRaw != null && shotsRaw !== "" ? Number(shotsRaw) : null,
    wins: 0,
    savePct: null,
    gaa: null,
    ppg,
    _statsFromBackend: true,
  };
}

function splitName(fullName) {
  const parts = String(fullName || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return { firstName: "Prospect", lastName: "Player" };
  if (parts.length === 1) return { firstName: parts[0], lastName: "Player" };
  return { firstName: parts.slice(0, -1).join(" "), lastName: parts[parts.length - 1] };
}

function mapBackendDraftBoard(entries, dateContext) {
  const rows = Array.isArray(entries) ? entries : [];
  if (!rows.length) return [];

  const mapped = rows.map((row, i) => {
    const fixedRow = applyProspectLeagueTeamFix(row);
    const rank = Number(fixedRow?.rank) || Number(fixedRow?.central_rank) || i + 1;
    const base = defaultProspectStub(rank);
    const nm = splitName(coalesce(row?.name, `${base.firstName} ${base.lastName}`));
    const pos = String(coalesce(row?.position, base.position) || "C").toUpperCase();
    const trend = String(row?.trend || "").toUpperCase();
    const backendStockRaw = coalesce(row?.stock_change, row?.stockChange, row?.stock_delta, row?.stockDelta);
    const backendStock = backendStockRaw != null && Number.isFinite(Number(backendStockRaw))
      ? Number(backendStockRaw)
      : null;
    const stock = backendStock != null ? backendStock : 0;
    const draftStock = normalizeBackendDraftStock(row, rank);
    const franchiseTier = normalizeBackendFranchiseTier(row);

    const league = resolveLeagueDisplay(fixedRow, base);
    const stats = mapBackendStats(fixedRow, base, pos, rank, league);
    const projected = resolveProjectedStatsRow(fixedRow);
    const recentForm = fixedRow?.recent_form && typeof fixedRow.recent_form === "object" ? fixedRow.recent_form : null;
    const scoutedRaw = coalesce(row?.scouted_percentage, row?.user_scouted_percentage, row?.team_scout_pct);
    const ambientRaw = coalesce(row?.scouting_confidence, row?.scout_grade, base.completion);
    // scouted_percentage=0 means "no dedicated file" — do not override ambient board confidence.
    const scoutedNum = scoutedRaw != null && scoutedRaw !== "" ? Number(scoutedRaw) : null;
    const ambientNum = ambientRaw != null && ambientRaw !== "" ? Number(ambientRaw) : null;
    const completion = Math.min(
      100,
      Math.max(
        0,
        Math.round(
          (Number.isFinite(scoutedNum) && scoutedNum > 0)
            ? scoutedNum
            : (Number.isFinite(ambientNum) ? ambientNum : (Number(base.completion) || 40))
        )
      )
    );
    const ovrRevealed = completion >= OVR_REVEAL_THRESHOLD || Boolean(row?.ovr_revealed);
    const rawRange = row?.ovr_range;
    const ovrRange = rawRange && rawRange.low != null && rawRange.high != null
      ? { low: Number(rawRange.low), high: Number(rawRange.high) }
      : null;

    const country = coalesce(row?.country, row?.nationality, base.country);
    const region = coalesce(row?.region, regionForCountry(country), base.region);
    const weightLbs = resolveWeightLbs(row, base);
    const countryCode = coalesce(
      row?.country_code,
      row?.countryCode,
      resolveCountryCode(country),
      resolveCountryCode(row?.birth_country),
      resolveCountryCode(row?.birthCountry),
    );

    return ensurePlayerHeadshotFields({
      ...base,
      ...stats,
      id: String(coalesce(row?.key, row?.id, base.id, `prospect-${rank}`)),
      name: coalesce(row?.name, `${nm.firstName} ${nm.lastName}`),
      rank,
      centralRank: Number(row?.central_rank) || rank,
      firstName: nm.firstName,
      lastName: nm.lastName,
      position: pos,
      age: Number(coalesce(row?.age, base.age)) || base.age,
      country,
      countryCode,
      region,
      league,
      leagueCode: coalesce(fixedRow?.league_code, fixedRow?.leagueCode),
      team: (() => {
        const cleaned = formatProspectTeam(fixedRow, coalesce(fixedRow?.team_name, fixedRow?.team, ""));
        if (cleaned && cleaned !== "—") return cleaned;
        return pickTeamName(league, seededNumber(String(fixedRow?.key || rank)));
      })(),
      leagueDisplay: formatProspectLeague(fixedRow),
      projection: coalesce(row?.projection, base.projection),
      talent: String(coalesce(row?.talent_grade, row?.scout_tier, base.talent) || "B"),
      playerType: coalesce(row?.player_type, row?.playerType, base.playerType),
      completion,
      stock,
      stockLabel: coalesce(row?.stock_label, row?.stockLabel),
      stockReason: coalesce(row?.stock_reason, row?.stockReason),
      stockDirection: coalesce(row?.stock_direction, row?.stockDirection),
      characterConcerns: Boolean(row?.character_concerns ?? row?.characterConcerns),
      scoutingConfidence: Number(coalesce(row?.scouting_confidence, row?.scoutingConfidence)) || null,
      intelLabel: coalesce(row?.intel_label, row?.intelLabel),
      isTranscendent: Boolean(row?.is_transcendent ?? row?.isTranscendent ?? row?.transcendent_talent),
      hometown: coalesce(row?.hometown, row?.birth_city, row?.birthCity),
      birthday: coalesce(row?.birthday, base.birthday),
      birthCity: coalesce(row?.hometown, row?.birth_city, row?.birthCity, country),
      height: coalesce(row?.height, base.height),
      weight: weightLbs,
      handedness: coalesce(row?.handedness, base.handedness),
      ceilingHidden: Boolean(row?.ceiling_hidden ?? row?.ceilingHidden),
      ceilingState: coalesce(row?.ceiling_state, row?.ceilingState),
      ceilingVisibility: row?.ceiling_visibility != null
        ? Number(row.ceiling_visibility)
        : (row?.ceilingVisibility != null ? Number(row.ceilingVisibility) : null),
      ceilingHint: coalesce(row?.ceiling_hint, row?.ceilingHint),
      floorScore: row?.floor_score != null
        ? Number(row.floor_score)
        : (row?.floorScore != null ? Number(row.floorScore) : null),
      potentialLabel: coalesce(row?.potential, row?.potential_label),
      potentialScore: (row?.ceiling_hidden ?? row?.ceilingHidden)
        ? null
        : (row?.potential_score != null ? Number(row.potential_score) : null),
      potentialRange: (row?.ceiling_hidden ?? row?.ceilingHidden)
        ? null
        : (row?.potential_range && row.potential_range.low != null && row.potential_range.high != null
          ? { low: Number(row.potential_range.low), high: Number(row.potential_range.high) }
          : null),
      riskLabel: coalesce(row?.risk, row?.risk_label),
      nhlEta: formatNhlEta(coalesce(row?.nhl_eta, row?.nhlEta), null),
      isGem: row?.is_gem != null ? Boolean(row.is_gem) : row?.isGem != null ? Boolean(row.isGem) : base.isGem,
      isBustRisk: row?.is_bust_risk != null
        ? Boolean(row.is_bust_risk)
        : row?.isBustRisk != null
        ? Boolean(row.isBustRisk)
        : base.isBustRisk,
      ovrHint: ovrRevealed ? Math.round(Number(coalesce(row?.true_ovr, 0)) || 0) : null,
      ovrRange,
      ovrRevealed,
      skating: attrFromBackend(row, base, "skating", "skating"),
      shooting: attrFromBackend(row, base, "shooting", "shooting"),
      passing: attrFromBackend(row, base, "passing", "passing"),
      defense: attrFromBackend(row, base, "defense", "defense"),
      physical: attrFromBackend(row, base, "physical", "physical"),
      hockeyIQ: attrFromBackend(row, base, "hockey_iq", "hockeyIQ"),
      compete: attrFromBackend(row, base, "compete", "compete"),
      leadership: attrFromBackend(row, base, "leadership", "leadership"),
      workEthic: attrFromBackend(row, base, "work_ethic", "workEthic"),
      coachability: attrFromBackend(row, base, "coachability", "coachability"),
      consistency: attrFromBackend(row, base, "consistency", "consistency"),
      poise: attrFromBackend(row, base, "poise", "poise"),
      morale: base.morale,
      character: base.character,
      fit: base.fit,
      watchlist: Boolean(row?.watchlist),
      target: Boolean(row?.target),
      doNotDraft: Boolean(row?.do_not_draft || row?.doNotDraft),
      assignedScout: coalesce(row?.assigned_scout, row?.assignedScout) || null,
      productionContext: coalesce(row?.production_context, row?.productionContext),
      translationRisk: coalesce(row?.translation_risk, row?.translationRisk),
      scoringEnvironment: coalesce(row?.scoring_environment, row?.scoringEnvironment),
      leagueDifficulty: coalesce(row?.league_difficulty, row?.leagueDifficulty),
      productionAdjustedScore: row?.production_adjusted_score ?? row?.productionAdjustedScore ?? null,
      ppg: stats.ppg != null
        ? Number(stats.ppg)
        : stats.gp > 0
        ? Number((stats.points / stats.gp).toFixed(3))
        : null,
      projectedGp: projected.gp != null ? Number(projected.gp) : null,
      projectedGoals: projected.goals != null ? Number(projected.goals) : null,
      projectedAssists: projected.assists != null ? Number(projected.assists) : null,
      projectedPoints: projected.points != null ? Number(projected.points) : null,
      projectedPpg: projected.ppg != null ? Number(projected.ppg) : null,
      projectedWins: projected.wins != null ? Number(projected.wins) : null,
      projectedSavePct: projected.savePct != null ? String(projected.savePct) : null,
      projectedGaa: projected.gaa != null ? String(projected.gaa) : null,
      recentForm,
      hasNoGames: stats._statsFromBackend && Number(stats.gp) === 0,
      draftStock,
      franchiseTier,
      backendStockSource: draftStock.source,
      draftRankReasonCodes: Array.isArray(row?.draft_rank_reason_codes)
        ? row.draft_rank_reason_codes
        : Array.isArray(row?.draftRankReasonCodes)
        ? row.draftRankReasonCodes
        : [],
      prospectRole: coalesce(row?.prospect_role, row?.prospectRole),
      roleAdjustedProduction: row?.role_adjusted_production ?? row?.roleAdjustedProduction ?? null,
      defensiveProjectionBonus: row?.defensive_projection_bonus ?? row?.defensiveProjectionBonus ?? null,
      analytics: row?.analytics && typeof row.analytics === "object" ? row.analytics : null,
    });
  });

  return addPositionRanks(mapped);
}

const FRANCHISE_TIER_ORDER = {
  franchise_swing: 1,
  core_upside: 2,
  debate_room: 3,
  safe_depth: 4,
  mystery_box: 5,
  late_flyer: 6,
  unclassified: 999,
};

const FRANCHISE_TIER_LABELS = {
  franchise_swing: "Franchise Swing",
  core_upside: "Core Upside",
  debate_room: "Debate Room",
  safe_depth: "Safe Depth",
  mystery_box: "Mystery Box",
  late_flyer: "Late Flyer",
  unclassified: "Unclassified",
};

function normalizeBackendDraftStock(row, rank = 0) {
  const nested = row?.draft_stock || row?.draftStock;
  if (nested && typeof nested === "object" && nested.source === "backend") {
    const direction = String(nested.direction || "UNKNOWN").toUpperCase();
    const unit = String(nested.stock_unit || nested.stockUnit || (String(nested.stock_mode || nested.stockMode) === "rank_change" ? "rank" : "heat"));
    const mode = String(nested.stock_mode || nested.stockMode || (unit === "rank" ? "rank_change" : "weekly_heat"));
    return {
      direction,
      deltaRank: Number(nested.delta_rank ?? nested.deltaRank ?? nested.display_delta) || 0,
      rankDelta: Number(nested.rank_delta ?? nested.rankDelta) || 0,
      stockHeat: Number(nested.stock_heat ?? nested.stockHeat) || 0,
      stockMode: mode,
      stockUnit: unit,
      previousRank: nested.previous_rank ?? nested.previousRank ?? null,
      currentRank: Number(nested.current_rank ?? nested.currentRank ?? rank) || rank,
      label: String(nested.label || "No Movement Data"),
      reason: String(nested.reason || ""),
      confidence: Number(nested.confidence) || 0,
      updatedAt: String(nested.updated_at || nested.updatedAt || ""),
      source: "backend",
      available: direction !== "UNKNOWN" || Boolean(nested.reason),
    };
  }
  const hasFlat = coalesce(row?.stock_change, row?.stockChange, row?.stock_delta, row?.stockDelta) != null
    || coalesce(row?.stock_label, row?.stockLabel, row?.stock_reason, row?.stockReason) != null
    || coalesce(row?.trend, row?.stock_direction, row?.stockDirection);
  if (!hasFlat) {
    return {
      direction: "UNKNOWN",
      deltaRank: 0,
      rankDelta: 0,
      stockHeat: 0,
      stockMode: "none",
      stockUnit: "heat",
      previousRank: null,
      currentRank: rank,
      label: "No Movement Data",
      reason: "",
      confidence: 0,
      updatedAt: "",
      source: "backend",
      available: false,
    };
  }
  const trend = String(row?.trend || row?.stock_direction || row?.stockDirection || "SAME").toUpperCase();
  const directionMap = { UP: "UP", DOWN: "DOWN", SAME: "STABLE", NEW: "NEW", FLAT: "STABLE", RISING: "UP", FALLING: "DOWN" };
  const direction = directionMap[trend] || "UNKNOWN";
  const delta = Number(coalesce(row?.stock_change, row?.stockChange, row?.stock_delta, row?.stockDelta)) || 0;
  const prev = row?.previous_rank ?? row?.previousRank ?? row?.rank_prev ?? row?.rankPrev;
  const mode = String(coalesce(row?.stock_mode, row?.stockMode) || "weekly_heat");
  const unit = String(coalesce(row?.stock_unit, row?.stockUnit) || (mode === "rank_change" ? "rank" : "heat"));
  return {
    direction,
    deltaRank: delta,
    rankDelta: Number(coalesce(row?.rank_delta, row?.rankDelta, row?.rank_change, row?.rankChange)) || 0,
    stockHeat: Number(coalesce(row?.stock_heat, row?.stockHeat, row?.weekly_stock_delta, row?.weeklyStockDelta)) || 0,
    stockMode: mode,
    stockUnit: unit,
    previousRank: prev != null ? Number(prev) : null,
    currentRank: Number(row?.rank ?? rank) || rank,
    label: String(coalesce(row?.stock_label, row?.stockLabel) || (direction === "STABLE" ? "Holding" : direction === "UP" ? "Rising" : direction === "DOWN" ? "Falling" : direction === "NEW" ? "New Entry" : "No Movement Data")),
    reason: String(coalesce(row?.stock_reason, row?.stockReason) || ""),
    confidence: Number(nestedConfidence(row)) || 0,
    updatedAt: String(row?.last_prospect_stat_update_date || ""),
    source: "backend",
    available: true,
  };
}

function nestedConfidence(row) {
  const nested = row?.draft_stock || row?.draftStock;
  if (nested && nested.confidence != null) return Number(nested.confidence);
  // Prefer sample-based stock confidence when present; do not fall back to scout %.
  return Number(coalesce(row?.stock_confidence, row?.stockConfidence)) || 0;
}

function normalizeBackendFranchiseTier(row) {
  const nested = row?.franchise_tier || row?.franchiseTier;
  if (nested && typeof nested === "object" && nested.source === "backend") {
    const key = String(nested.key || "unclassified");
    return {
      key,
      label: String(nested.label || FRANCHISE_TIER_LABELS[key] || "Unclassified"),
      order: Number(nested.order ?? FRANCHISE_TIER_ORDER[key]) || FRANCHISE_TIER_ORDER[key] || 999,
      confidence: Number(nested.confidence) || 0,
      reason: String(nested.reason || ""),
      source: "backend",
      available: key !== "unclassified" || Boolean(nested.reason),
    };
  }
  const flatKey = coalesce(row?.tier_key, row?.tierKey);
  if (flatKey) {
    const key = String(flatKey);
    return {
      key,
      label: String(coalesce(row?.tier_label, row?.tierLabel, FRANCHISE_TIER_LABELS[key]) || "Unclassified"),
      order: Number(FRANCHISE_TIER_ORDER[key]) || 999,
      confidence: Number(coalesce(row?.tier_confidence, row?.tierConfidence)) || 0,
      reason: String(coalesce(row?.tier_reason, row?.tierReason) || ""),
      source: "backend",
      available: true,
    };
  }
  return {
    key: "unclassified",
    label: "Unclassified",
    order: 999,
    confidence: 0,
    reason: "Backend tier data unavailable",
    source: "backend",
    available: false,
  };
}

function getStockTone(stock) {
  const dir = String(stock?.direction || "UNKNOWN").toUpperCase();
  if (dir === "UP") return "rise";
  if (dir === "DOWN") return "fall";
  if (dir === "NEW") return "new";
  if (dir === "STABLE") return "stable";
  return "neutral";
}

function stockMoverDirection(stock) {
  const dir = String(stock?.direction || "UNKNOWN").toUpperCase();
  const delta = Number(stock?.deltaRank);
  const numericDelta = Number.isFinite(delta) ? delta : 0;
  if (dir === "NEW") return "NEW";
  if (numericDelta > 0) return "UP";
  if (numericDelta < 0) return "DOWN";
  if (dir === "UP" || dir === "DOWN" || dir === "STABLE") return dir;
  return "UNKNOWN";
}

function buildStockMoversFromProspects(prospects) {
  const risers = [];
  const fallers = [];
  const list = Array.isArray(prospects) ? prospects : [];

  for (const p of list) {
    const stock = p?.draftStock;
    if (!stock) continue;
    if (!stock.available && String(stock.direction || "UNKNOWN").toUpperCase() === "UNKNOWN") continue;

    const movement = stockMoverDirection(stock);
    const delta = Number(stock.deltaRank);
    const numericDelta = Number.isFinite(delta) ? delta : 0;
    if (movement === "UP" && numericDelta <= 0) continue;
    if (movement === "DOWN" && numericDelta >= 0) continue;
    if (movement !== "UP" && movement !== "DOWN") continue;

    const item = {
      key: p.id,
      name: `${p.firstName || ""} ${p.lastName || ""}`.trim() || p.name || "Unknown",
      rank: Number(p.rank) || 0,
      delta_rank: numericDelta,
      deltaRank: numericDelta,
      label: stock.label || "",
    };

    if (movement === "UP") risers.push(item);
    else fallers.push(item);
  }

  risers.sort((a, b) => a.rank - b.rank || b.delta_rank - a.delta_rank);
  fallers.sort((a, b) => a.rank - b.rank || a.delta_rank - b.delta_rank);

  return {
    risers,
    fallers,
    source: "client",
  };
}

function getTierTone(tier) {
  const key = String(tier?.key || "unclassified");
  const map = {
    franchise_swing: "gold",
    core_upside: "cyan",
    debate_room: "purple",
    safe_depth: "blue",
    mystery_box: "warn",
    late_flyer: "muted",
    unclassified: "neutral",
  };
  return map[key] || "neutral";
}

function groupProspectsByBackendTier(prospects) {
  const groups = new Map();
  const sorted = [...prospects].sort((a, b) => {
    const oa = a.franchiseTier?.order ?? 999;
    const ob = b.franchiseTier?.order ?? 999;
    if (oa !== ob) return oa - ob;
    return a.rank - b.rank;
  });
  sorted.forEach((p) => {
    const key = p.franchiseTier?.key || "unclassified";
    const label = p.franchiseTier?.label || FRANCHISE_TIER_LABELS[key] || "Unclassified";
    if (!groups.has(key)) {
      groups.set(key, { key, label, order: p.franchiseTier?.order ?? 999, prospects: [] });
    }
    groups.get(key).prospects.push(p);
  });
  return [...groups.values()].sort((a, b) => a.order - b.order);
}

function stockBadgeText(stock, options = {}) {
  const compact = options === true || options?.compact;
  if (!stock?.available && stock?.direction === "UNKNOWN") return "—";
  const delta = Number(stock?.deltaRank) || 0;
  const unit = String(stock?.stockUnit || "heat");
  const suffix = !compact && unit === "rank" && delta !== 0 ? " rk" : "";
  if (delta > 0) return `+${delta}${suffix}`;
  if (delta < 0) return `${delta}${suffix}`;
  if (stock?.direction === "NEW") return "NEW";
  return "+0";
}

function recommendedAction(player, meta) {
  if (meta.doNotDraft) return "Remove from board — marked Do Not Draft.";
  if (meta.target) return "Pin maintained — prioritize in war room and schedule final viewings.";
  if (player.completion < 45) return "Request skills assessment and schedule in-person viewing.";
  if (player.isBustRisk || player.characterConcerns) return "Cross-check character report before committing pick capital.";
  if (player.rank <= 16) return "Maintain top-tier tracking — align with lottery positioning.";
  return "Monitor late-season production and regional scout updates.";
}

function ratingLabel(value) {
  if (value >= 90) return "Elite";
  if (value >= 82) return "Excellent";
  if (value >= 74) return "Good";
  if (value >= 64) return "Average";
  return "Concern";
}

function gradeFromValue(value) {
  if (value >= 94) return "A+";
  if (value >= 88) return "A";
  if (value >= 82) return "A-";
  if (value >= 76) return "B+";
  if (value >= 70) return "B";
  if (value >= 64) return "B-";
  if (value >= 58) return "C+";
  return "C";
}

function initials(player) {
  return `${player.firstName?.[0] || ""}${player.lastName?.[0] || ""}`.toUpperCase();
}

function fullName(player) {
  return `${player.firstName} ${player.lastName}`;
}

function normalizeId(v) {
  return String(v ?? "").trim().toLowerCase();
}

function toFiniteOrNull(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function collectStandings(franchiseState) {
  if (Array.isArray(franchiseState?.standings)) return franchiseState.standings;
  if (Array.isArray(franchiseState?.standings?.teams)) return franchiseState.standings.teams;
  if (Array.isArray(franchiseState?.league_standings)) return franchiseState.league_standings;
  return [];
}

function standingRowForUserTeam(franchiseState) {
  const tid = String(franchiseState?.team?.id ?? franchiseState?.user_team_id ?? "");
  if (!tid) return null;
  return collectStandings(franchiseState).find((r) => String(r.team_id || r.id) === tid) || null;
}

function safePickTeamIdentity(franchiseState) {
  const team = franchiseState?.team || {};
  const userTeam = franchiseState?.user_team || {};
  return {
    id: normalizeId(
      userTeam.id
      ?? userTeam.team_id
      ?? team.id
      ?? team.team_id
      ?? franchiseState?.user_team_id
    ),
    abbr: String(
      userTeam.abbreviation
      ?? userTeam.abbrev
      ?? team.abbreviation
      ?? team.abbrev
      ?? ""
    ).trim(),
  };
}

function resolveRecord(franchiseState) {
  const direct = franchiseState?.team?.record || franchiseState?.record;
  if (direct) {
    if (typeof direct === "string") return direct;
    const w = direct.w ?? direct.wins;
    const l = direct.l ?? direct.losses;
    const otl = direct.otl ?? direct.ot ?? direct.overtime_losses;
    if (w != null || l != null || otl != null) {
      return `${w ?? 0}-${l ?? 0}-${otl ?? 0}`;
    }
  }

  const row = standingRowForUserTeam(franchiseState);
  if (row) {
    const w = row.w ?? row.wins;
    const l = row.l ?? row.losses;
    const otl = row.otl ?? row.ot ?? row.overtime_losses;
    if (w != null || l != null || otl != null) {
      return `${w ?? 0}-${l ?? 0}-${otl ?? 0}`;
    }
  }

  const team = franchiseState?.team || {};
  const userTeam = franchiseState?.user_team || {};
  const standings = franchiseState?.standings || franchiseState?.team_standings || {};
  const wins = toFiniteOrNull(
    userTeam.wins ?? team.wins ?? standings.wins ?? standings.w ?? 0
  );
  const losses = toFiniteOrNull(
    userTeam.losses ?? team.losses ?? standings.losses ?? standings.l ?? 0
  );
  const otl = toFiniteOrNull(
    userTeam.ot_losses
    ?? userTeam.otl
    ?? team.ot_losses
    ?? team.otl
    ?? standings.ot_losses
    ?? standings.otl
    ?? 0
  );
  return `${wins ?? 0}-${losses ?? 0}-${otl ?? 0}`;
}

function resolveTeamLogo(franchiseState) {
  const team = franchiseState?.team || {};
  const userTeam = franchiseState?.user_team || {};
  const resolved = resolveFranchiseTeamLogo(userTeam, userTeam?.name || team?.name || "");
  if (resolved) return resolved;
  const logo = userTeam.logo ?? userTeam.logo_url ?? team.logo ?? team.logo_url ?? "";
  return typeof logo === "string" ? logo.trim() : "";
}

function resolveTeamAbbr(franchiseState) {
  const team = franchiseState?.team || {};
  const userTeam = franchiseState?.user_team || {};
  return String(
    userTeam.abbreviation
    ?? userTeam.abbrev
    ?? team.abbreviation
    ?? team.abbrev
    ?? team.short_name
    ?? team.name
    ?? "TEAM"
  ).trim();
}

function resolveTeamToken(franchiseState) {
  const abbr = resolveTeamAbbr(franchiseState);
  if (/[A-Za-z]/.test(abbr)) return abbr.toUpperCase().slice(0, 3);
  const team = franchiseState?.team || {};
  const userTeam = franchiseState?.user_team || {};
  const name = String(userTeam.name ?? team.name ?? "").trim();
  if (name) {
    const token = name.split(/\s+/).map((x) => x[0]).join("").toUpperCase();
    if (token) return token.slice(0, 3);
  }
  return "TM";
}

function matchesPickOwner(pick, teamIdentity) {
  const ownerFields = [
    pick?.owner_team_id,
    pick?.owning_team_id,
    pick?.current_owner_team_id,
    pick?.original_owner_team_id,
    pick?.team_id,
    pick?.current_team_id,
    pick?.to_team_id,
    pick?.owner_id,
    pick?.owner,
    pick?.team_abbreviation,
    pick?.team_abbrev,
    pick?.owner_abbreviation,
    pick?.owner_abbrev,
  ];
  return ownerFields.some((v) => {
    const id = normalizeId(v);
    if (!id) return false;
    return id === teamIdentity.id || id === normalizeId(teamIdentity.abbr);
  });
}

function matchesPickYear(pick, draftYear) {
  const raw = (
    pick?.draft_year
    ?? pick?.draftYear
    ?? pick?.year
    ?? pick?.season_year
    ?? pick?.season
    ?? pick?.round_year
  );
  if (raw == null || raw === "") return false;
  const n = Number(raw);
  if (Number.isFinite(n)) return n === Number(draftYear);
  const s = String(raw);
  if (s.includes(String(draftYear))) return true;
  const idBlob = String(pick?.pick_id ?? pick?.id ?? "");
  const idMatch = idBlob.match(/\b(\d{4})\b/);
  if (idMatch) {
    return Number(idMatch[1]) === Number(draftYear);
  }
  return false;
}

function flattenPickPool(pool) {
  if (Array.isArray(pool)) return pool;
  if (!pool || typeof pool !== "object") return [];
  const values = Object.values(pool);
  const out = [];
  for (const v of values) {
    if (Array.isArray(v)) out.push(...v);
    else if (v && typeof v === "object") {
      if (
        v.draft_year != null
        || v.year != null
        || v.round != null
        || v.owner_team_id != null
        || v.team_id != null
      ) {
        out.push(v);
      } else {
        out.push(...flattenPickPool(v));
      }
    }
  }
  return out;
}

function resolveOwnedPickCount(franchiseState, draftYear) {
  const teamIdentity = safePickTeamIdentity(franchiseState);
  const tradeTeams = franchiseState?.trade_assets?.teams || franchiseState?.tradeAssets?.teams || {};
  const tradeTeamEntries = tradeTeams && typeof tradeTeams === "object" ? Object.entries(tradeTeams) : [];
  const normalizedTeamId = normalizeId(teamIdentity.id);
  const normalizedTeamAbbr = normalizeId(teamIdentity.abbr);
  const keyMatchedTradeTeamBlock = tradeTeamEntries.find(([key]) => {
    const nk = normalizeId(key);
    if (!nk) return false;
    return (
      (normalizedTeamId && nk === normalizedTeamId)
      || (normalizedTeamAbbr && nk === normalizedTeamAbbr)
    );
  })?.[1] || null;
  const tradeTeamList = tradeTeams && typeof tradeTeams === "object" ? Object.values(tradeTeams) : [];
  const fallbackTradeTeamBlock = tradeTeamList.find((block) => {
    const bid = normalizeId(block?.team_id ?? block?.id ?? block?.owner_team_id);
    const babbr = normalizeId(block?.abbreviation ?? block?.abbr);
    return (
      (teamIdentity.id && bid && bid === teamIdentity.id)
      || (teamIdentity.abbr && babbr && babbr === normalizeId(teamIdentity.abbr))
    );
  }) || null;
  const directTradeTeamBlock = (
    (teamIdentity.id && tradeTeams[teamIdentity.id])
    || tradeTeams[String(franchiseState?.user_team_id ?? "")]
    || tradeTeams[String(franchiseState?.team?.id ?? "")]
    || keyMatchedTradeTeamBlock
    || fallbackTradeTeamBlock
    || null
  );
  const teamPicks = flattenPickPool(
    directTradeTeamBlock?.picks
    ?? directTradeTeamBlock?.draft_picks
    ?? []
  );
  if (!teamPicks.length || !draftYear) return null;
  const ownedForYear = teamPicks.filter((pick) => matchesPickYear(pick, draftYear));
  return ownedForYear.length;
}

function stockClass(stock) {
  if (stock > 0) return "draft-trend-flag--up";
  if (stock < 0) return "draft-trend-flag--down";
  return "draft-trend-flag--same";
}

function stockText(stock) {
  if (stock > 0) return `↟ +${stock}`;
  if (stock < 0) return `↡ ${stock}`;
  return "—";
}

function formatLeaderValue(value, { decimals = 0, suffix = "" } = {}) {
  if (value == null || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (decimals > 0) return `${n.toFixed(decimals)}${suffix}`;
  return `${Math.round(n)}${suffix}`;
}

function leaderPpg(player) {
  const gp = Number(player?.gp) || 0;
  if (gp <= 0) return null;
  return Number(prospectPpgValue(player).toFixed(3));
}

function shootingPctFor(player) {
  const shots = Number(player?.shots);
  const goals = Number(player?.goals);
  if (!Number.isFinite(shots) || shots <= 0 || !Number.isFinite(goals)) return null;
  return Number(((goals / shots) * 100).toFixed(1));
}

function leaderFirstPresent(sources, keys) {
  for (const key of keys) {
    for (const src of sources) {
      if (!src || typeof src !== "object") continue;
      const v = src[key];
      if (v != null && v !== "") return v;
    }
  }
  return undefined;
}

function leaderNum(v) {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function leaderPct(v) {
  const n = leaderNum(v);
  if (n == null) return null;
  if (n > 0 && n <= 1) return Number((n * 100).toFixed(1));
  return Number(n.toFixed(1));
}

function extractProspectAnalytics(player, profile) {
  const embedded = player?.analytics && typeof player.analytics === "object" ? player.analytics : null;
  const profileAnalytics = profile?.analytics && typeof profile.analytics === "object" ? profile.analytics : null;
  const statsAnalytics = profile?.stats?.analytics && typeof profile.stats.analytics === "object"
    ? profile.stats.analytics
    : null;
  const sources = [embedded, profileAnalytics, statsAnalytics, player, profile?.stats, profile, player?.rawData].filter(Boolean);
  const pick = (...keys) => leaderFirstPresent(sources, keys);
  const gp = Number(player?.gp) || 0;
  const shots = leaderNum(pick("shots", "sog", "shots_on_goal"));
  const shotRateRaw = leaderNum(pick("shot_rate", "shots_per_game", "sog_per_game"));
  return {
    xgf_pct: leaderPct(pick("xgf_pct", "xGF_pct", "expected_goals_pct")),
    cf_pct: leaderPct(pick("cf_pct", "corsi_pct", "corsi_percentage")),
    ff_pct: leaderPct(pick("ff_pct", "fenwick_pct", "fenwick_percentage")),
    war: leaderNum(pick("war", "WAR", "player_war")),
    offensive_war: leaderNum(pick("offensive_war", "off_war", "owar")),
    defensive_war: leaderNum(pick("defensive_war", "def_war", "dwar")),
    shooting_pct: shootingPctFor(player) ?? leaderPct(pick("shooting_pct", "sh_pct")),
    plus_minus: player?.plusMinus != null ? leaderNum(player.plusMinus) : leaderNum(pick("plus_minus", "plusMinus")),
    primary_points: leaderNum(pick("primary_points", "primary_pts")),
    shot_rate: shotRateRaw ?? (gp > 0 && shots != null ? Number((shots / gp).toFixed(2)) : null),
    toi: leaderNum(pick("toi", "toi_avg", "avg_toi", "time_on_ice")),
    defensive_impact: leaderNum(pick("defensive_impact", "defense_score")),
    quality_of_competition: leaderNum(pick("quality_of_competition", "qoc")),
    quality_of_teammates: leaderNum(pick("quality_of_teammates", "qot")),
    gsax: leaderNum(pick("gsax", "goals_saved_above_expected", "GSAx")),
    quality_starts: leaderNum(pick("quality_starts", "qs")),
  };
}

function formatLeaderPct(v) {
  if (v == null) return "—";
  return `${Number(v).toFixed(1)}%`;
}

function formatLeaderSigned(v, decimals = 1) {
  if (v == null) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (n > 0) return `+${n.toFixed(decimals)}`;
  return n.toFixed(decimals);
}

function formatLeaderStockDisplay(row, { emphasize = false } = {}) {
  const draft = row.draftStock || null;
  const delta = Number(draft?.deltaRank ?? row.stockDelta) || 0;
  const unit = String(draft?.stockUnit || "heat");
  const suffix = emphasize && unit === "rank" && delta !== 0 ? " rk" : "";
  if (delta > 0) return { text: `+${delta}${suffix}`, cls: "is-up" };
  if (delta < 0) return { text: `${delta}${suffix}`, cls: "is-down" };
  if (!draft?.available && draft?.direction === "UNKNOWN") {
    return { text: "—", cls: "is-neutral" };
  }
  if (emphasize) return { text: "+0", cls: "is-neutral" };
  return { text: "+0", cls: "is-neutral" };
}

const LEADER_MODE_META = {
  points: { eyebrow: "Draft Class", title: "Points Leaders", emphasis: "points" },
  goals: { eyebrow: "Draft Class", title: "Goals Leaders", emphasis: "goals" },
  assists: { eyebrow: "Draft Class", title: "Assists Leaders", emphasis: "assists" },
  ppg: { eyebrow: "Draft Class", title: "Points Per Game", emphasis: "ppg" },
  defense: { eyebrow: "Defensemen", title: "D Scoring Leaders", emphasis: "points" },
  goalies: { eyebrow: "Goaltenders", title: "Goalie Leaders", emphasis: "goalie" },
  stock: { eyebrow: "Stock Market", title: "Rising Prospects", emphasis: "stock" },
  analytics: { eyebrow: "Adaptive Analytics", title: "Advanced Stat Leaders", emphasis: "analytics" },
};

const LEADER_SORT_VIEW_MODES = [
  { key: "production", label: "Production" },
  { key: "analytics", label: "Analytics" },
  { key: "draft", label: "Draft Stock" },
];

const LEADER_METRICS = {
  gp: { key: "gp", label: "GP", group: "production", format: (v) => formatLeaderValue(v) },
  goals: { key: "goals", label: "G", group: "production", format: (v) => formatLeaderValue(v) },
  assists: { key: "assists", label: "A", group: "production", format: (v) => formatLeaderValue(v) },
  points: { key: "points", label: "PTS", group: "production", format: (v) => formatLeaderValue(v) },
  ppg: { key: "ppg", label: "PPG", group: "production", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
  shooting_pct: { key: "shooting_pct", label: "SH%", group: "efficiency", format: (v) => formatLeaderPct(v) },
  plus_minus: {
    key: "plus_minus",
    label: "+/-",
    group: "efficiency",
    format: (v) => (v == null ? "—" : (v > 0 ? `+${v}` : String(v))),
  },
  xgf_pct: { key: "xgf_pct", label: "xGF%", group: "analytics", format: (v) => formatLeaderPct(v) },
  cf_pct: { key: "cf_pct", label: "CF%", group: "analytics", format: (v) => formatLeaderPct(v) },
  ff_pct: { key: "ff_pct", label: "FF%", group: "analytics", format: (v) => formatLeaderPct(v) },
  war: { key: "war", label: "WAR", group: "analytics", format: (v) => formatLeaderSigned(v, 2) },
  offensive_war: { key: "offensive_war", label: "Off WAR", group: "analytics", format: (v) => formatLeaderSigned(v, 2) },
  defensive_war: { key: "defensive_war", label: "Def WAR", group: "analytics", format: (v) => formatLeaderSigned(v, 2) },
  defensive_impact: { key: "defensive_impact", label: "Def Imp", group: "analytics", format: (v) => formatLeaderSigned(v, 1) },
  primary_points: { key: "primary_points", label: "Prim P", group: "analytics", format: (v) => formatLeaderValue(v) },
  shot_rate: { key: "shot_rate", label: "Shots/G", group: "analytics", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
  quality_of_competition: { key: "quality_of_competition", label: "QoC", group: "analytics", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
  quality_of_teammates: { key: "quality_of_teammates", label: "QoT", group: "analytics", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
  draft_rank: { key: "draft_rank", label: "Draft", group: "draft", format: (v) => (v == null ? "—" : `#${v}`), sortAsc: true },
  stock_delta: { key: "stock_delta", label: "Stock", group: "draft", format: (v, row) => formatLeaderStockDisplay(row || {}, { emphasize: true }).text },
  scout_pct: { key: "scout_pct", label: "Scout", group: "draft", format: (v) => (v == null ? "—" : `${Math.round(v)}%`) },
  wins: { key: "wins", label: "W", group: "production", format: (v) => formatLeaderValue(v) },
  save_pct: { key: "save_pct", label: "SV%", group: "production", format: (v, row) => (row?.savePct != null ? row.savePct : v != null ? String(v) : "—") },
  gaa: { key: "gaa", label: "GAA", group: "production", format: (v, row) => (row?.gaa != null ? row.gaa : v != null ? String(v) : "—"), sortAsc: true },
  shutouts: { key: "shutouts", label: "SO", group: "production", format: (v) => formatLeaderValue(v) },
  gsax: { key: "gsax", label: "GSAx", group: "analytics", format: (v) => formatLeaderSigned(v, 2) },
  quality_starts: { key: "quality_starts", label: "QS", group: "analytics", format: (v) => formatLeaderValue(v) },
};

const LEADER_SORT_BAR_BY_VIEW = {
  production: ["points", "ppg", "goals", "assists", "gp", "shooting_pct", "plus_minus"],
  analytics: ["war", "offensive_war", "defensive_war", "xgf_pct", "cf_pct", "shooting_pct", "plus_minus", "defensive_impact"],
  draft: ["draft_rank", "stock_delta", "scout_pct"],
};

const LEADER_GOALIE_SORT_BAR = ["wins", "save_pct", "gaa", "gp", "shutouts", "gsax", "quality_starts"];

const LEADER_ROW_METRICS_SKATER = {
  production: ["gp", "goals", "assists", "points", "ppg"],
  efficiency: ["shooting_pct", "plus_minus"],
  analytics: ["xgf_pct", "cf_pct", "war", "offensive_war", "defensive_war"],
};

const LEADER_ROW_METRICS_GOALIE = {
  production: ["gp", "wins", "save_pct", "gaa", "shutouts"],
  analytics: ["gsax", "quality_starts"],
};

const LEADER_MODE_DEFAULT_SORT = {
  points: "points",
  goals: "goals",
  assists: "assists",
  ppg: "ppg",
  defense: "points",
  goalies: "wins",
  stock: "stock_delta",
  analytics: "war",
};

function getLeaderMetricRaw(row, metricKey) {
  if (!row || !metricKey) return null;
  if (metricKey === "gp") return leaderNum(row.gp);
  if (metricKey === "goals") return leaderNum(row.goals);
  if (metricKey === "assists") return leaderNum(row.assists);
  if (metricKey === "points") return leaderNum(row.points);
  if (metricKey === "ppg") return row.ppg != null ? leaderNum(row.ppg) : null;
  if (metricKey === "draft_rank") return leaderNum(row.draftRank);
  if (metricKey === "stock_delta") return leaderNum(row.stockDelta);
  if (metricKey === "scout_pct") return leaderNum(row.scoutPct);
  if (metricKey === "wins") return leaderNum(row.wins);
  if (metricKey === "shutouts") return leaderNum(row.shutouts);
  if (metricKey === "save_pct") {
    const raw = row.savePct;
    if (raw == null || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }
  if (metricKey === "gaa") {
    const raw = row.gaa;
    if (raw == null || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }
  const analytics = row.analytics || {};
  if (metricKey === "shooting_pct") return leaderNum(row.shootingPct ?? analytics.shooting_pct);
  if (metricKey === "plus_minus") return leaderNum(row.plusMinus ?? analytics.plus_minus);
  return leaderNum(analytics[metricKey]);
}

function formatLeaderMetric(row, metricKey) {
  const def = LEADER_METRICS[metricKey];
  if (!def) return "—";
  const raw = getLeaderMetricRaw(row, metricKey);
  if (raw == null && metricKey !== "save_pct" && metricKey !== "gaa" && metricKey !== "stock_delta") return "—";
  return def.format(raw, row);
}

function leaderSortMetricValue(row, metricKey) {
  return getLeaderMetricRaw(row, metricKey);
}

function sortLeaderRowsByMetric(rows, metricKey, direction = "desc") {
  if (!metricKey || !rows?.length) return rows || [];
  const def = LEADER_METRICS[metricKey];
  const invert = Boolean(def?.sortAsc);
  const desc = invert ? direction === "asc" : direction !== "asc";
  return [...rows].sort((a, b) => {
    const va = leaderSortMetricValue(a, metricKey);
    const vb = leaderSortMetricValue(b, metricKey);
    const aMissing = va == null || !Number.isFinite(Number(va));
    const bMissing = vb == null || !Number.isFinite(Number(vb));
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    const diff = Number(vb) - Number(va);
    return desc ? diff : -diff;
  });
}

function defaultSortForLeaderMode(leaderMode, prospects) {
  const key = LEADER_MODE_DEFAULT_SORT[leaderMode] || "points";
  if (leaderMode === "analytics") {
    const skaters = (prospects || []).filter((p) => p.position !== "G");
    for (const skater of skaters) {
      const analytics = extractProspectAnalytics(skater);
      if (analytics.war != null) return "war";
    }
    return "points";
  }
  return key;
}

function resolveHeroMetricKey(leaderMode, activeSortKey) {
  if (activeSortKey) return activeSortKey;
  return LEADER_MODE_DEFAULT_SORT[leaderMode] || "points";
}

function resolveSortViewMode(leaderMode) {
  if (leaderMode === "analytics") return "analytics";
  if (leaderMode === "stock") return "draft";
  return "production";
}

function sortBarMetricsForMode(leaderMode, sortViewMode) {
  if (leaderMode === "goalies") return LEADER_GOALIE_SORT_BAR;
  return LEADER_SORT_BAR_BY_VIEW[sortViewMode] || LEADER_SORT_BAR_BY_VIEW.production;
}

const LEADER_MODE_OPTIONS = [
  { key: "points", label: "Points" },
  { key: "goals", label: "Goals" },
  { key: "assists", label: "Assists" },
  { key: "ppg", label: "PPG" },
  { key: "defense", label: "D Scoring" },
  { key: "goalies", label: "Goalies" },
  { key: "stock", label: "Rising" },
  { key: "analytics", label: "Analytics" },
];

function sortProspectsForLeaderMode(prospects, leaderMode) {
  const skaters = prospects.filter((p) => p.position !== "G");
  const defense = skaters.filter((p) => isDefensemanPosition(p.position));
  if (leaderMode === "goalies") {
    return [...prospects]
      .filter((p) => p.position === "G")
      .sort((a, b) => Number(b.wins) - Number(a.wins) || Number(b.savePct) - Number(a.savePct));
  }
  if (leaderMode === "goals") {
    return [...skaters].sort((a, b) => b.goals - a.goals || b.points - a.points);
  }
  if (leaderMode === "assists") {
    return [...skaters].sort((a, b) => b.assists - a.assists || b.points - a.points);
  }
  if (leaderMode === "ppg") {
    return [...skaters]
      .filter((p) => Number(p.gp) > 0)
      .sort((a, b) => (leaderPpg(b) || 0) - (leaderPpg(a) || 0));
  }
  if (leaderMode === "defense") {
    return [...defense].sort((a, b) => b.points - a.points || b.goals - a.goals);
  }
  if (leaderMode === "stock") {
    return [...prospects].sort((a, b) => {
      const da = Number(a.draftStock?.deltaRank ?? a.stock) || 0;
      const db = Number(b.draftStock?.deltaRank ?? b.stock) || 0;
      return db - da;
    });
  }
  if (leaderMode === "analytics") {
    return [...skaters].sort((a, b) => {
      const aa = extractProspectAnalytics(a);
      const ba = extractProspectAnalytics(b);
      if (aa.war != null || ba.war != null) return (ba.war ?? -999) - (aa.war ?? -999);
      if (aa.xgf_pct != null || ba.xgf_pct != null) return (ba.xgf_pct ?? -999) - (aa.xgf_pct ?? -999);
      if (aa.cf_pct != null || ba.cf_pct != null) return (ba.cf_pct ?? -999) - (aa.cf_pct ?? -999);
      return b.points - a.points;
    });
  }
  return [...skaters].sort((a, b) => b.points - a.points);
}

function buildLeaderDisplayRow(player, profilesById) {
  const profile = profilesById?.[player.id];
  const gp = Number(player.gp) || 0;
  const ppg = leaderPpg(player);
  const stockDelta = Number(player.draftStock?.deltaRank ?? player.stock) || 0;
  const analytics = extractProspectAnalytics(player, profile);
  return {
    player,
    profile,
    id: player.id,
    draftRank: player.rank,
    name: fullName(player),
    position: player.position,
    handedness: player.handedness,
    league: player.leagueDisplay || player.league,
    team: player.team,
    gp,
    goals: player.goals,
    assists: player.assists,
    points: player.points,
    ppg,
    plusMinus: analytics.plus_minus,
    shootingPct: analytics.shooting_pct,
    scoutPct: player.scoutingConfidence ?? player.completion,
    stockDelta,
    stockLabel: player.draftStock?.label || player.stockLabel,
    wins: player.wins,
    savePct: player.savePct,
    gaa: player.gaa,
    shutouts: player.shutouts,
    analytics,
  };
}

function buildLeadersModalSummary(leaders, leaderMode) {
  if (!leaders?.length) return null;
  const top = leaders[0];
  const meta = LEADER_MODE_META[leaderMode] || LEADER_MODE_META.points;
  const skaters = leaders.filter((r) => r.position !== "G");
  const gpSkaters = skaters.filter((r) => r.gp > 0);
  const avgPpg = gpSkaters.length
    ? gpSkaters.reduce((sum, r) => sum + (r.ppg || 0), 0) / gpSkaters.length
    : null;
  const scoutVals = leaders.map((r) => r.scoutPct).filter((v) => v != null && Number.isFinite(Number(v)));
  const avgScout = scoutVals.length
    ? scoutVals.reduce((sum, v) => sum + Number(v), 0) / scoutVals.length
    : null;

  let leaderValue = "—";
  if (leaderMode === "goalies") leaderValue = top.savePct != null ? `SV% ${top.savePct}` : formatLeaderValue(top.wins, { suffix: " W" });
  else if (leaderMode === "goals") leaderValue = `${formatLeaderValue(top.goals)} G`;
  else if (leaderMode === "assists") leaderValue = `${formatLeaderValue(top.assists)} A`;
  else if (leaderMode === "ppg") leaderValue = top.ppg != null ? `${Number(top.ppg).toFixed(2)} PPG` : "—";
  else if (leaderMode === "stock") leaderValue = formatLeaderStockDisplay(top, { emphasize: true }).text;
  else if (leaderMode === "analytics") {
    const a = top.analytics || {};
    leaderValue = a.war != null ? `WAR ${formatLeaderSigned(a.war, 2)}` : a.xgf_pct != null ? `xGF% ${formatLeaderPct(a.xgf_pct)}` : `${formatLeaderValue(top.points)} PTS`;
  } else leaderValue = `${formatLeaderValue(top.points)} PTS`;

  return {
    meta,
    count: leaders.length,
    leaderName: top.name,
    leaderValue,
    avgPpg,
    avgScout,
  };
}

function LeaderStatPill({
  metricKey,
  row,
  hero = false,
  active = false,
  sortable = false,
  onClick,
}) {
  const def = LEADER_METRICS[metricKey];
  if (!def) return null;
  const className = [
    "dc-lm-pill",
    hero ? "is-hero" : "",
    active ? "is-sort-active" : "",
    sortable ? "is-sortable" : "",
  ].filter(Boolean).join(" ");
  const content = (
    <>
      <span className="dc-lm-pill__label">{def.label}</span>
      <strong className="dc-lm-pill__value">{formatLeaderMetric(row, metricKey)}</strong>
    </>
  );

  if (sortable && onClick) {
    return (
      <button
        type="button"
        className={className}
        onClick={() => onClick(metricKey)}
        title={`Sort by ${def.label}`}
        aria-pressed={active}
      >
        {content}
      </button>
    );
  }
  return <div className={className}>{content}</div>;
}

function LeadersSortBar({ leaderMode, sortViewMode, activeSortKey, onSort }) {
  const keys = sortBarMetricsForMode(leaderMode, sortViewMode);
  return (
    <div className="dc-lm-sort-bar">
      <span className="dc-lm-sort-bar__label">Sort by</span>
      <div className="dc-lm-sort-bar__pills">
        {keys.map((key) => {
          const def = LEADER_METRICS[key];
          if (!def) return null;
          return (
            <button
              key={key}
              type="button"
              className={`dc-lm-sort-pill${activeSortKey === key ? " is-active" : ""}`}
              onClick={() => onSort(key)}
              aria-pressed={activeSortKey === key}
            >
              {def.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LeaderModalRow({
  row,
  index,
  leaderMode,
  onSelect,
  heroMetricKey,
  activeSortMetric,
  onMetricSort,
  sortViewMode,
}) {
  const isGoalie = leaderMode === "goalies" || row.position === "G";
  const stock = formatLeaderStockDisplay(row, { emphasize: leaderMode === "stock" });
  const hand = row.handedness ? String(row.handedness)[0] : "—";
  const rowMetrics = isGoalie ? LEADER_ROW_METRICS_GOALIE : LEADER_ROW_METRICS_SKATER;
  const emphasizeGroup = sortViewMode || resolveSortViewMode(leaderMode);

  const renderCluster = (groupKey, metricKeys, label) => {
    if (!metricKeys?.length) return null;
    return (
      <div className={`dc-lm-row__cluster dc-lm-row__cluster--${groupKey}${emphasizeGroup === groupKey ? " is-emphasis-group" : ""}`}>
        <span className="dc-lm-row__cluster-label">{label}</span>
        <div className="dc-lm-row__cluster-pills">
          {metricKeys.map((key) => (
            <LeaderStatPill
              key={key}
              metricKey={key}
              row={row}
              hero={heroMetricKey === key}
              active={activeSortMetric === key}
              sortable={Boolean(onMetricSort)}
              onClick={onMetricSort}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <article className={`dc-lm-row${isGoalie ? " dc-lm-row--goalie" : ""}`}>
      <div className="dc-lm-row__identity">
        <span className="dc-lm-row__rank">{index + 1}</span>
        <div className="dc-lm-row__player">
          <button type="button" className="dc-lm-row__name" onClick={() => onSelect(row)}>
            {row.name}
          </button>
          <span className="dc-lm-row__meta">
            {row.position || "—"} · {hand} · {row.league || "—"}
            {row.team ? ` · ${row.team}` : ""}
          </span>
        </div>
      </div>

      <div className="dc-lm-row__stats">
        {renderCluster("production", rowMetrics.production, "Production")}
        {!isGoalie ? renderCluster("efficiency", rowMetrics.efficiency, "Efficiency") : null}
        {renderCluster(isGoalie ? "analytics" : "analytics", rowMetrics.analytics, "Analytics")}
      </div>

      <div className={`dc-lm-row__draft${emphasizeGroup === "draft" ? " is-emphasis-group" : ""}`}>
        <LeaderStatPill
          metricKey="draft_rank"
          row={row}
          hero={heroMetricKey === "draft_rank"}
          active={activeSortMetric === "draft_rank"}
          sortable={Boolean(onMetricSort)}
          onClick={onMetricSort}
        />
        <LeaderStatPill
          metricKey="scout_pct"
          row={row}
          hero={heroMetricKey === "scout_pct"}
          active={activeSortMetric === "scout_pct"}
          sortable={Boolean(onMetricSort)}
          onClick={onMetricSort}
        />
        <div className={`dc-lm-row__stock ${stock.cls}${heroMetricKey === "stock_delta" ? " is-hero" : ""}${activeSortMetric === "stock_delta" ? " is-sort-active" : ""}`}>
          {onMetricSort ? (
            <button type="button" className="dc-lm-row__stock-btn" onClick={() => onMetricSort("stock_delta")} title="Sort by Stock">
              <span>Stock</span>
              <strong>{stock.text}</strong>
            </button>
          ) : (
            <>
              <span>Stock</span>
              <strong>{stock.text}</strong>
            </>
          )}
        </div>
      </div>
    </article>
  );
}

function strengthList(player) {
  if (player.completion < 45) {
    return ["Insufficient scouting — strengths not confirmed yet."];
  }
  const pool = [
    player.hockeyIQ >= 78 && "High-end hockey IQ and reads pressure early",
    player.passing >= 78 && "Creates offense through seams and controlled entries",
    player.shooting >= 78 && "Dangerous release from the slot and circles",
    player.skating >= 78 && "Strong acceleration and edge control",
    player.defense >= 78 && "Reliable defensive habits away from the puck",
    player.physical >= 78 && "Competes hard on walls and around the crease",
    player.workEthic >= 78 && "High work rate with clear development habits",
    player.poise >= 78 && "Composed under pressure in late-game situations",
  ].filter(Boolean);

  return pool.length ? pool.slice(0, 5) : [
    "Projectable frame with room to develop",
    "Shows flashes of high-end processing",
    "Useful habits in transition",
  ];
}

function weaknessList(player) {
  if (player.completion < 45) {
    return ["More viewings needed before weaknesses can be confirmed."];
  }
  const pool = [
    player.skating < 70 && "Needs another gear in open ice",
    player.physical < 70 && "Could add strength before NHL minutes",
    player.defense < 70 && "Defensive reads are still inconsistent",
    player.shooting < 70 && "Shot selection can be predictable",
    player.passing < 70 && "Can force plays through traffic",
    player.consistency < 70 && "Game-to-game impact can fluctuate",
    player.coachability < 70 && "Scouts want quicker adjustments after feedback",
    player.leadership < 70 && "Still developing a louder presence in the room",
  ].filter(Boolean);

  return pool.length ? pool.slice(0, 4) : [
    "Needs pro pace adjustment",
    "Could become more consistent shift-to-shift",
    "Strength gains will decide ceiling",
  ];
}

function scoutReportSections(player, meta, profile) {
  const name = fullName(player);
  const charDone = meta?.requestedReports?.character === "complete" || player.completion >= 82;
  const p = profile || player?.profile || null;

  // Real backend evidence when available, otherwise the preliminary templated fallback.
  const strengths = Array.isArray(p?.strengths) ? p.strengths.filter(Boolean) : [];
  const concerns = Array.isArray(p?.concerns) ? p.concerns.filter((c) => c && c !== "Clear") : [];
  const projLabel = p?.projection?.label || null;
  const potRating = p?.potential?.rating != null ? Math.round(Number(p.potential.rating)) : null;
  const nhlProb = p?.potential?.probability != null ? Math.round(Number(p.potential.probability)) : null;
  const etaLabel = p?.eta?.label || p?.estimatedNhlArrival || null;
  const compLabel = p?.player_comparison?.label || p?.player_comparison?.summary || null;
  const volatility = p?.developmentVolatility || null;

  return {
    projection: projLabel
      ? `${name} projects as ${projLabel}${potRating ? ` (ceiling ~${potRating} OVR${nhlProb ? `, ${nhlProb}% NHL odds` : ""})` : ""}.`
      : (player.completion >= 40
        ? `${name} currently projects as ${player.projection} with ${player.talent} tier tools.`
        : "Projection remains preliminary — central scouting has limited viewings."),
    upside: strengths.length
      ? `Strengths: ${strengths.slice(0, 3).join(", ")}.`
      : (player.rank <= 16
        ? "Ceiling profiles as a top-six / top-pair contributor if development accelerates."
        : player.rank <= 32
        ? "Upside tied to whether standout tools become repeatable at pro pace."
        : "Longer runway with role-player floor and moderate upside."),
    risk: concerns.length
      ? `Concerns: ${concerns.slice(0, 3).join(", ")}${volatility ? ` · ${volatility} volatility` : ""}.`
      : (volatility === "High" || player.isBustRisk || player.riskLabel === "High"
        ? "Scouts flag volatility in consistency and translation risk."
        : (volatility === "Low" || player.riskLabel === "Low" || player.rank <= 10)
        ? "Low variance relative to tier — main risk is injury or stagnation."
        : "Medium variance — needs continued viewings through spring."),
    notes: scoutSummary(player, p),
    timeline: etaLabel
      || formatNhlEta(player.nhlEta, null)
      || (player.rank <= 8 ? "1–2 years to NHL readiness" : player.rank <= 32 ? "2–3 years development runway" : "3–5 years with AHL seasoning likely"),
    comparable: compLabel || `${player.playerType} — ${player.league} pace`,
    nextScout: player.completion < 55
      ? "Schedule in-person viewings and request skills assessment."
      : charDone
      ? "Monitor late-season production and interview at combine."
      : "Request character report and cross-check with regional scout.",
  };
}

function getScoutingMeta(store, id) {
  return store[id] || EMPTY_SCOUTING_META;
}

function nextReportDue(completion) {
  if (completion < 40) return "2–3 weeks";
  if (completion < 70) return "4–6 weeks";
  return "Post-season review";
}

function scoutSummary(player, profile) {
  const p = profile || player?.profile || null;
  // Prefer the backend's evidence-based micro summary; then synthesize from real
  // strengths/concerns/projection; only then fall back to rank-templated boilerplate.
  if (p) {
    const micro = typeof p.micro_summary === "string" ? p.micro_summary.trim() : "";
    if (micro) return micro;
    const strengths = Array.isArray(p.strengths) ? p.strengths.filter(Boolean) : [];
    const concerns = Array.isArray(p.concerns) ? p.concerns.filter((c) => c && c !== "Clear") : [];
    const proj = p.projection?.label || p.developmentProfile || null;
    const eta = p.eta?.label || p.estimatedNhlArrival || null;
    const parts = [];
    if (proj) parts.push(`projects as ${proj}`);
    if (strengths.length) parts.push(strengths.slice(0, 2).join(" and ").toLowerCase());
    if (concerns.length) parts.push(`must clean up ${concerns[0].toLowerCase()}`);
    if (eta && eta !== "Now") parts.push(`NHL arrival ${eta}`);
    if (parts.length) return `${fullName(player)} ${parts.join("; ")}.`;
  }

  if (player.rank <= 5) {
    return `${fullName(player)} grades as a potential franchise-level piece with high-end tools, strong detail, and a profile that should translate quickly if development stays on track.`;
  }

  if (player.rank <= 16) {
    return `${fullName(player)} projects as a top-half first-round talent with enough translatable traits to become a major NHL contributor.`;
  }

  if (player.rank <= 32) {
    return `${fullName(player)} has first-round upside, but the final projection depends on whether the weaker parts of the profile catch up to the standout tools.`;
  }

  return `${fullName(player)} is a longer-view prospect with useful traits, development variance, and enough upside to justify serious scouting attention.`;
}

/** CSS-only fallback avatar — PlayerHeadshot.js → PlayerHeadshot.css (not styles/playerHeadshot.css). */
function DraftClassHeadshot({ player, size = "md", board = false }) {
  return (
    <PlayerHeadshot
      player={player}
      size={size}
      variant={board ? "circle" : "card"}
      className={`dc-shared-headshot${board ? " dc-board-headshot" : ""}`}
      draftState="eligible"
    />
  );
}

function TopHeader({ onBack, dateContext, franchiseState }) {
  const logoUrl = resolveTeamLogo(franchiseState);
  const initials = resolveTeamToken(franchiseState);
  const record = resolveRecord(franchiseState);
  const ownedPickCount = resolveOwnedPickCount(franchiseState, dateContext?.draftYear);
  return (
    <header className="dc-topbar dc-topbar--draft-hud">
      <div className="dc-topbar__left">
        <button type="button" className="dc-back-btn dc-back-btn--hud" onClick={onBack}>
          ← Back
        </button>
        <small className="dc-record-mini">Record {record}</small>
      </div>

      <div className="dc-topbar__center dc-topbar__center-logo">
        {logoUrl ? (
          <img className="dc-team-logo" src={logoUrl} alt="Team logo" />
        ) : (
          <div className="dc-team-logo-fallback" aria-label="Team logo">{initials}</div>
        )}
        <small className="dc-date-mini">{dateContext?.statsThrough || "—"}</small>
      </div>

      <div className="dc-topbar__right dc-topbar__right-picks">
        <strong className="dc-pick-count-mini">{ownedPickCount ?? 0}</strong>
        <small>{dateContext?.draftYear || "—"} Picks</small>
      </div>
    </header>
  );
}

function DraftBoardNavRail({ activeBoardView, setActiveBoardView }) {
  return (
    <nav className="dc-board-nav" aria-label="Draft board navigation">
      <div className="dc-board-nav__stack">
        {BOARD_NAV_ITEMS.map((item) => {
          const active = activeBoardView === item.key;
          return (
            <button
              key={item.key}
              type="button"
              className={`dc-board-nav__item${active ? " is-active" : ""}`}
              onClick={() => setActiveBoardView(item.key)}
              title={item.title}
              aria-label={item.title}
              aria-pressed={active}
            >
              <span className={`dc-board-nav__icon dc-board-nav__icon--${item.icon}`} aria-hidden="true">
                {item.icon === "rank" ? <><i /><i /><i /></> : null}
                {item.icon === "fwd" ? <><i /><i /><i /></> : null}
                {item.icon === "dmen" ? <i className="dc-board-nav__shield" /> : null}
                {item.icon === "goalie" ? <i className="dc-board-nav__mask" /> : null}
              </span>
              <span className="dc-board-nav__label">{item.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

function parseIsoDate(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  const base = s.slice(0, 10);
  const d = new Date(base);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatDaysUntil(rawDays) {
  const n = Number(rawDays);
  if (!Number.isFinite(n)) return "—";
  if (n < 0) return "PASSED";
  if (n === 0) return "TODAY";
  return `${Math.floor(n)} DAYS`;
}

function inferEventDaysFallback(franchiseState, eventKey) {
  const nowRaw = franchiseState?.nhl_today?.iso || franchiseState?.current_date || "";
  const now = parseIsoDate(nowRaw);
  const markers = Array.isArray(franchiseState?.season_anchor_events) ? franchiseState.season_anchor_events : [];
  const row = markers.find((m) => String(m?.key || "") === eventKey);
  const event = parseIsoDate(row?.date);
  if (!now || !event) return "—";
  const delta = Math.floor((event.getTime() - now.getTime()) / 86400000);
  return formatDaysUntil(delta);
}

function statusIconForKey(key) {
  const k = String(key || "").toLowerCase();
  if (k.includes("cup")) return "◆";
  if (k.includes("playoff") || k.includes("contender")) return "◉";
  if (k.includes("tanking") || k.includes("tank")) return "✕";
  if (k.includes("rebuild")) return "△";
  return "◎";
}

function CommandStatStrip({ franchiseState, onOpenWjc }) {
  const hud = franchiseState?.draft_class_hud || {};
  const teamStatus = hud?.team_status || {
    key: "unknown",
    label: "Backend status unavailable",
    reason: "draft_class_hud.team_status missing from backend payload.",
  };
  const events = hud?.events || {};
  const wjcEvent = events?.wjc || {};
  const wjcValue =
    wjcEvent.display ||
    formatDaysUntil(wjcEvent.days_until) ||
    inferEventDaysFallback(franchiseState, "wjc_start");
  const wjcClickable = typeof onOpenWjc === "function" && wjcValue && wjcValue !== "—";
  const lotValue = events?.lottery?.display || formatDaysUntil(events?.lottery?.days_until) || inferEventDaysFallback(franchiseState, "draft_lottery");
  const draftValue = events?.draft?.display || formatDaysUntil(events?.draft?.days_until) || inferEventDaysFallback(franchiseState, "draft");

  const items = [
    { key: "status", icon: statusIconForKey(teamStatus?.key), label: "TEAM STATUS", value: teamStatus?.label || "Middling", clickable: false },
    { key: "wjc", icon: "◎", label: "WJC", value: wjcValue || "—", clickable: wjcClickable },
    { key: "lottery", icon: "◌", label: "LOTTERY", value: lotValue || "—", clickable: false },
    { key: "draft", icon: "▣", label: "DRAFT", value: draftValue || "—", clickable: false },
  ];

  return (
    <section className="dc-stat-strip dc-hud-strip" aria-label="Draft class HUD status">
      {items.map((item) => {
        const inner = (
          <>
            <span className="dc-hud-strip__icon" aria-hidden="true">{item.icon}</span>
            <div className="dc-hud-strip__text">
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </div>
          </>
        );
        if (item.clickable) {
          return (
            <button
              key={item.key}
              type="button"
              className="dc-hud-strip__item dc-hud-strip__item--clickable"
              title="Open World Juniors"
              onClick={onOpenWjc}
            >
              {inner}
            </button>
          );
        }
        return (
          <article key={item.key} className="dc-hud-strip__item" title={item.key === "status" ? (teamStatus?.reason || "") : ""}>
            {inner}
          </article>
        );
      })}
    </section>
  );
}

function ProspectBadges({ player, meta, showStockTier = false }) {
  const badges = [];
  if (meta.watchlist) badges.push({ key: "wl", text: "★ Watchlist", cls: "dc-badge--watch" });
  if (meta.target) badges.push({ key: "tg", text: "◎ Target", cls: "dc-badge--target" });
  if (meta.doNotDraft) badges.push({ key: "dnd", text: "⚠ DND", cls: "dc-badge--dnd" });
  if (showStockTier) {
    const stockTone = getStockTone(player.draftStock);
    if (stockTone === "rise") badges.push({ key: "up", text: player.draftStock?.label || "↑ Rising", cls: "dc-badge--rise" });
    if (stockTone === "fall") badges.push({ key: "dn", text: player.draftStock?.label || "↓ Falling", cls: "dc-badge--fall" });
    if (player.franchiseTier?.available) badges.push({ key: "tier", text: player.franchiseTier.label, cls: `dc-badge--tier-${getTierTone(player.franchiseTier)}` });
  }
  if (player.isGem) badges.push({ key: "gem", text: "◆ Gem", cls: "dc-badge--gem" });
  if (player.isBustRisk) badges.push({ key: "bust", text: "▲ Risk", cls: "dc-badge--bust" });
  if (player.characterConcerns) badges.push({ key: "char", text: "⚠ Character", cls: "dc-badge--bust" });
  if (!badges.length) return null;
  return (
    <div className="dc-badges">
      {badges.map((b) => <span key={b.key} className={`dc-badge ${b.cls}`}>{b.text}</span>)}
    </div>
  );
}

function StockBadge({ stock, compact = false }) {
  const tone = getStockTone(stock);
  const text = stockBadgeText(stock);
  return (
    <span className={`dc-stock-badge dc-stock-badge--${tone}`} title={stock?.reason || ""}>
      {compact ? text : text}
    </span>
  );
}

function TierBadge({ tier, compact = false }) {
  const tone = getTierTone(tier);
  return (
    <span className={`dc-tier-badge dc-tier-badge--${tone}`} title={tier?.reason || ""}>
      {compact ? (tier?.label || "Unclassified") : (tier?.label || "Unclassified")}
    </span>
  );
}

function ProspectIdentityBlock({ player }) {
  const countryLabel = player.country || player.nationality || normalizeCountryCode(player) || "";
  const flag = countryFlag(countryLabel);
  const flagUrl = flagApiUrl(countryLabel, 64);
  const [flagBroken, setFlagBroken] = useState(false);

  return (
    <div className="dc-prospect-identity">
      <div className="dc-prospect-identity__avatar-wrap">
        <DraftClassHeadshot player={player} size="md" board />
        {flagUrl && !flagBroken ? (
          <img
            className="dc-prospect-identity__flag-badge"
            src={flagUrl}
            alt={countryLabel || "Nationality"}
            title={countryLabel || "Nationality unknown"}
            loading="lazy"
            onError={() => setFlagBroken(true)}
          />
        ) : (
          <span
            className="dc-prospect-identity__flag-fallback"
            title={countryLabel || "Nationality unknown"}
            aria-label={countryLabel || "Nationality unknown"}
          >
            {flag}
          </span>
        )}
      </div>
    </div>
  );
}

function ProspectBoardRow({ player, index, selected, onSelect, meta }) {
  const rank = prospectRank(player, index);
  const countryCode = normalizeCountryCode(player);
  const movementCls = movementIndicator(player.draftStock).cls;
  const scoutPct = prospectScoutingPct(player);
  const fogClass = prospectConfidenceFogClass(scoutPct);
  const ceilingHidden = Boolean(player?.ceilingHidden);
  const profile = player?.profile || null;
  const ovr = resolveCurrentEstimate(player, profile, ceilingHidden, Boolean(profile?.dedicatedScoutFile));
  const pot = resolvePotentialEstimate(player);
  const league = player.leagueDisplay || player.league || "—";
  const pos = player.position || "—";
  const nationFull = player.country || player.nationality || countryCode || "";
  const rowFlagUrl = flagApiUrl(countryCode || nationFull, 32);
  const stockText = movementDisplayText(player.draftStock);
  const projection = player.projection || projectionForRank(rank);
  const confText = scoutPct != null ? `${Math.round(scoutPct)}%` : "—";

  return (
    <button
      type="button"
      className={`dc-prospect-row${selected ? " is-selected" : ""}${meta.doNotDraft ? " is-dnd" : ""}${player?.isTranscendent ? " prospect-card--transcendent" : ""}${rank > 32 ? " is-late-round" : ""}`}
      onClick={onSelect}
    >
      <div className="dc-prospect-row__rank" aria-label={`Rank ${rank}`}>
        <span>{rank}</span>
      </div>

      <div className="dc-prospect-row__player">
        {rowFlagUrl ? (
          <img
            className="dc-prospect-row__flag"
            src={rowFlagUrl}
            alt={nationFull || ""}
            title={nationFull || undefined}
            loading="lazy"
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
        ) : null}
        <strong className="dc-prospect-row__name">{player.firstName} {player.lastName}</strong>
      </div>

      <span className="dc-prospect-row__cell dc-prospect-row__pos">{pos}</span>
      <span className="dc-prospect-row__cell dc-prospect-row__league" title={league}>{league}</span>
      <span
        className={`dc-prospect-row__cell dc-prospect-row__pot${pot.tier ? ` is-pot-${pot.tier.key}` : ""}${pot.exact ? "" : " is-range"}`}
        title={`${pot.tier ? `${pot.tier.label} · ` : ""}${pot.detail} · Now ${ovr.text}`}
      >
        {pot.text}
      </span>
      <span className="dc-prospect-row__cell dc-prospect-row__proj">{projection}</span>
      <span className={`dc-prospect-row__cell dc-prospect-row__conf ${fogClass}`}>{confText}</span>
      <span className={`dc-prospect-row__cell dc-prospect-row__stock ${movementCls}`}>{stockText}</span>
    </button>
  );
}

function ProspectBoardPanel({ prospects, selectedProspectId, onOpenProspect, scoutingStore, activeBoardView }) {
  const filterLabel = boardFilterLabel(activeBoardView);

  return (
    <section className="dc-prospect-board">
      <header className="dc-prospect-board__head">
        <h2>Prospect Board</h2>
        <span>
          {prospects.length} prospect{prospects.length === 1 ? "" : "s"}
          {filterLabel ? ` · ${filterLabel}` : ""}
        </span>
      </header>
      <div className="dc-prospect-board__list dc-scroll-surface">
        <ProspectBoardColumnHeader />
        {!prospects.length ? (
          <div className="dc-board-standby">
            <span className="dc-board-standby__label">
              {filterLabel ? "No matching prospects" : "Board not yet populated"}
            </span>
            <p className="dc-empty-note dc-prospect-board__empty">
              {filterLabel
                ? `Nothing on the board matches the ${filterLabel} filter.`
                : "No prospects have been published to this draft class yet."}
            </p>
          </div>
        ) : (
          prospects.map((player, index) => (
            <ProspectBoardRow
              key={player.id}
              player={player}
              index={index}
              selected={player.id === selectedProspectId}
              onSelect={() => onOpenProspect(player)}
              meta={getScoutingMeta(scoutingStore, player.id)}
            />
          ))
        )}
      </div>
    </section>
  );
}

function ModalCloseButton({ onClick, label = "Close" }) {
  return (
    <button type="button" className="dc-modal-close" onClick={onClick} aria-label={label} title={label}>
      <svg className="dc-modal-close__x" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6.2 6.2l11.6 11.6M17.8 6.2L6.2 17.8" />
      </svg>
    </button>
  );
}

/** @deprecated Prefer ModalCloseButton — kept so older call sites keep working if reintroduced. */
function HockeySticksCloseButton(props) {
  return <ModalCloseButton {...props} />;
}

function ProfileMeter({ label, value, display, max = 100, tone = "cyan", title, note }) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  const barPct = max > 100 ? Math.max(0, Math.min(100, (num / max) * 100)) : Math.max(0, Math.min(100, num));
  const valueLabel = display ?? (max > 100 ? String(Math.round(num)) : `${Math.round(num)}%`);
  return (
    <div className={`dc-profile-meter dc-profile-meter--${tone}`} title={title || undefined}>
      <div className="dc-profile-meter__head">
        <span>{label}</span>
        <strong>{valueLabel}</strong>
      </div>
      <div className="dc-profile-meter__track">
        <i style={{ width: `${barPct}%` }} />
      </div>
      {note ? <p className="dc-profile-meter__note">{note}</p> : null}
    </div>
  );
}

function ProfileChip({ children, tone = "neutral" }) {
  if (!children) return null;
  return <span className={`dc-profile-chip dc-profile-chip--${tone}`}>{children}</span>;
}

function ProfileFactRow({ label, value, sub, title }) {
  const safeValue = displaySafeText(value);
  const safeSub = displaySafeText(sub);
  if (safeValue == null || safeValue === "") return null;
  return (
    <div className="dc-profile-fact" title={title || undefined}>
      <span className="dc-profile-fact__label">{label}</span>
      <div className="dc-profile-fact__body">
        <strong>{safeValue}</strong>
        {safeSub ? <small>{safeSub}</small> : null}
      </div>
    </div>
  );
}

function formatIdentityLine(player, profile) {
  const badges = getPlayerIdentityBadges(player, profile);
  const parts = [];
  if (badges.position && badges.position !== "—") parts.push(badges.position);
  if (badges.handedness) {
    const shoot = String(badges.handedness).toUpperCase().startsWith("L") ? "Shoots L" : "Shoots R";
    parts.push(shoot);
  }
  if (badges.height) parts.push(badges.height);
  if (badges.weight) {
    const wt = String(badges.weight).replace(/\s*LBS$/i, " lbs");
    parts.push(wt);
  }
  if (badges.age) {
    const age = String(badges.age).replace(/Y$/i, "");
    parts.push(`${age} yrs`);
  }
  return parts.join(" · ");
}

// Displayed tools use real scouted attributes — no cosmetic lift toward ceiling.
function prospectAttributeLift(profile) {
  return 0;
}

function ProspectAttributeStrip({ player, profile }) {
  const lift = prospectAttributeLift(profile);
  const ceilingHidden = Boolean(profile?.ceilingHidden || profile?.potential?.hidden);
  const dedicatedFile = Boolean(profile?.dedicatedScoutFile);
  const wideFog = ceilingHidden && !dedicatedFile;
  const ambient = Number(profile?.scout_confidence ?? player.completion ?? player.scoutingConfidence ?? 55);
  const completion = wideFog ? Math.min(Math.max(ambient, 52), 66) : Math.max(ambient, 40);
  const bump = (val) => {
    const n = Number(val);
    if (!Number.isFinite(n)) return val;
    return Math.min(94, Math.round(n + lift));
  };
  const attrs = [
    ["Skating", bump(player.skating), 1],
    ["Shot", bump(player.shooting), 2],
    ["Pass", bump(player.passing), 3],
    ["Defend", bump(player.defense), 4],
    ["Physical", bump(player.physical), 5],
    ["IQ", bump(player.hockeyIQ), 6],
  ];
  const hasAny = attrs.some(([, val]) => Number.isFinite(Number(val)));
  if (!hasAny) return null;
  return (
    <div className="dc-profile-attr-strip">
        <span className="dc-profile-tags__label">Tools</span>
      <div className="dc-profile-attr-strip__grid">
        {attrs.map(([label, val, seed]) => {
          const display = attributeDisplay(val, completion, seed, { wideFog });
          return (
            <div key={label} className={`dc-profile-attr-mini${display.locked ? " is-locked" : ""}`}>
              <span>{label}</span>
              <strong>{display.text}</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// NHL readiness — driven by current ability / ETA, never rank alone.
function nhlReadinessLabel(rank, ceiling, currentOvr, profile) {
  if (profile?.readinessLabel) return profile.readinessLabel;
  if (profile?.ceilingHidden || profile?.potential?.hidden) {
    const age = Number(profile?.age ?? 18);
    if (age <= 17) return "Multi-year project";
    if (age >= 20) return "Near-term decision";
    return "Development TBD";
  }
  const etaLabel = String(profile?.eta?.label || profile?.nhlEta || "").toUpperCase();
  if (etaLabel === "NOW") return "NHL ready";
  if (etaLabel === "1Y") return "1 year away";
  if (etaLabel === "2Y") return "2 years away";
  if (etaLabel === "3Y") return "3 years away";
  if (etaLabel === "4Y+" || etaLabel === "4Y") return "Long-term project";
  const cur = Number(currentOvr) || 0;
  const c = Number(ceiling) || 0;
  if (cur >= 76) return "NHL ready";
  if (cur >= 70 || (cur >= 64 && c >= 76)) return "1 year away";
  if (cur >= 60) return "2 years away";
  if (cur >= 54) return "3 years away";
  return "Long-term project";
}

// Compact season-line stats only — no prose ("small sample", stock blurbs, league names).
function seasonAnalyticsLine(analytics, gp, { ppg = null } = {}) {
  const games = Number(gp);
  const parts = [];
  const pushRaw = (label, text) => {
    if (text != null && text !== "" && parts.length < 4) parts.push(`${label} ${text}`);
  };
  const push = (obj, key, label, fmt) => {
    if (!obj || typeof obj !== "object") return;
    const v = Number(obj[key]);
    if (!Number.isFinite(v) || v === 0) return;
    pushRaw(label, fmt(v));
  };
  const pct = (v) => `${Math.round(v <= 1 ? v * 100 : v)}%`;

  if (Number.isFinite(Number(ppg)) && Number(ppg) > 0 && Number.isFinite(games) && games > 0) {
    pushRaw("PPG", Number(ppg).toFixed(2));
  }

  const sampleOk = Number.isFinite(games) && games >= 15;
  if (sampleOk && analytics && typeof analytics === "object") {
    push(analytics, "xgf_pct", "xGF", pct);
    push(analytics, "cf_pct", "CF", pct);
    push(analytics, "war", "WAR", (v) => v.toFixed(1));
  }
  if (analytics && typeof analytics === "object") {
    push(analytics, "shots", "SOG", (v) => String(Math.round(v)));
    push(analytics, "primary_points", "1stP", (v) => String(Math.round(v)));
    push(analytics, "shooting_pct", "SH%", (v) => `${Number(v).toFixed(1)}%`);
    push(analytics, "plus_minus", "+/-", (v) => (v > 0 ? `+${Math.round(v)}` : String(Math.round(v))));
  }
  return parts.length ? parts.join(" · ") : null;
}

/** Collapse consecutive same-rank samples so early movement isn't crushed onto the left edge. */
function compressTrajectoryPoints(points) {
  if (!Array.isArray(points) || points.length <= 2) return points || [];
  const out = [];
  for (let i = 0; i < points.length; i += 1) {
    const pt = points[i];
    const prev = out[out.length - 1];
    const isLast = i === points.length - 1;
    if (!prev || prev.rank !== pt.rank || isLast) {
      if (isLast && prev && prev.rank === pt.rank) {
        // Keep a single end marker; refresh label toward "Current".
        out[out.length - 1] = { ...pt, label: pt.label || prev.label };
      } else {
        out.push(pt);
      }
    }
  }
  // Cap length but always keep first + last.
  if (out.length <= 8) return out;
  const mid = out.slice(1, -1);
  const step = Math.ceil(mid.length / 6);
  const kept = [out[0]];
  for (let i = 0; i < mid.length; i += step) kept.push(mid[i]);
  const last = out[out.length - 1];
  if (kept[kept.length - 1] !== last) kept.push(last);
  return kept;
}

/** Rank / stock history points for value trajectory — backend trail, else season checkpoints. */
function buildValueTrajectoryPoints(profile, player) {
  const raw = profile?.rankHistory
    || profile?.stock_history
    || player?.rankHistory
    || player?.stockHistory
    || [];
  const points = [];
  const pushRank = (rank, label) => {
    const r = Number(rank);
    if (!Number.isFinite(r) || r <= 0) return;
    points.push({ x: points.length, rank: r, label: String(label || `#${points.length + 1}`) });
  };
  if (Array.isArray(raw) && raw.length) {
    raw.forEach((entry, i) => {
      if (entry == null) return;
      if (typeof entry === "number" && Number.isFinite(entry)) {
        pushRank(entry, String(i + 1));
        return;
      }
      if (typeof entry !== "object") return;
      const rank = Number(
        entry.rank ?? entry.public_rank ?? entry.board_rank ?? entry.central_rank ?? entry.value
      );
      const label = entry.date_label || entry.label || entry.event || entry.phase || entry.date || entry.event_source;
      pushRank(rank, label);
    });
  }

  // Always prefer a season-arc skeleton when the live trail is empty or a single sample.
  if (points.length < 2) {
    points.length = 0;
    pushRank(profile?.preseasonRank ?? player?.preseasonRank, "Preseason");
    pushRank(profile?.midseasonRank ?? player?.midseasonRank, "Midseason");
    pushRank(profile?.currentRank ?? profile?.rank ?? player?.rank ?? prospectRank(player), "Current");
  } else {
    // Relabel ends so the axis always reads Preseason → Current across the season.
    if (points[0]) points[0] = { ...points[0], label: "Preseason" };
    if (points[points.length - 1]) {
      points[points.length - 1] = { ...points[points.length - 1], label: "Current" };
    }
  }

  const flatOrThin = points.length < 2 || points.every((p) => p.rank === points[0].rank);
  if (flatOrThin) {
    const current = Number(
      points[points.length - 1]?.rank
      ?? profile?.currentRank
      ?? profile?.rank
      ?? player?.rank
      ?? prospectRank(player)
    );
    const pre = Number(profile?.preseasonRank ?? player?.preseasonRank);
    const seasonMove = Number.isFinite(pre) && Number.isFinite(current) && pre > 0 && current > 0
      ? pre - current
      : 0;
    const stock = player?.draftStock;
    let delta = Number(stock?.deltaRank ?? player?.stock);
    // Prefer full-season preseason→current movement over a single weekly tick.
    if (Number.isFinite(seasonMove) && seasonMove !== 0) delta = seasonMove;
    // Heat-mode +6 still means "rose"; convert to a board-rank shift for the chart.
    if (!Number.isFinite(delta) || delta === 0) {
      const heat = Number(stock?.stockHeat);
      if (Number.isFinite(heat) && heat !== 0) delta = heat;
    }
    if (Number.isFinite(current) && current > 0 && Number.isFinite(delta) && delta !== 0) {
      const earlier = Math.max(1, Math.round(current + delta));
      const mid = Math.max(1, Math.round((earlier + current) / 2));
      return [
        { x: 0, rank: earlier, label: "Preseason" },
        { x: 1, rank: mid, label: "Midseason" },
        { x: 2, rank: Math.round(current), label: "Current" },
      ];
    }
    // Flat board season: still draw a full-width line so the chart isn't a left stub.
    if (Number.isFinite(current) && current > 0) {
      return [
        { x: 0, rank: Math.round(current), label: "Preseason" },
        { x: 1, rank: Math.round(current), label: "Current" },
      ];
    }
  }
  return compressTrajectoryPoints(points).map((p, i) => ({ ...p, x: i }));
}

function formatSignalPct(v) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  return `${(n <= 1.5 ? n * 100 : n).toFixed(1)}%`;
}

function formatSignalSigned(v, decimals = 1) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  if (n > 0) return `+${n.toFixed(decimals)}`;
  return n.toFixed(decimals);
}

function formatSignalNum(v, decimals = 0) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  return Number(v).toFixed(decimals);
}

/** Current ability estimate — always available; ceiling fog does not blank present skill. */
function resolveCurrentEstimate(player, profile, ceilingHidden, dedicatedFile) {
  const exact = Number(profile?.scoutedOverall ?? profile?.currentOvrEstimate ?? player?.ovrHint);
  const revealed = Boolean(player?.ovrRevealed) || (
    Number(profile?.scout_confidence ?? player?.scoutingConfidence ?? player?.completion) >= OVR_REVEAL_THRESHOLD
  );
  // Exact OVR only when revealed / dedicated file — never as a late-round ceiling leak.
  if (revealed && (!ceilingHidden || dedicatedFile) && Number.isFinite(exact) && exact > 0) {
    return { text: String(Math.round(exact)), detail: "Scouted OVR", exact: true };
  }
  const low = Number(profile?.overallRangeLow ?? player?.ovrRange?.low);
  const high = Number(profile?.overallRangeHigh ?? player?.ovrRange?.high);
  if (Number.isFinite(low) && Number.isFinite(high) && low > 0 && high >= low) {
    return { text: `${Math.round(low)}–${Math.round(high)}`, detail: ceilingHidden ? "Present ability" : "Range", exact: false };
  }
  if (Number.isFinite(exact) && exact > 0 && !ceilingHidden) {
    return { text: String(Math.round(exact)), detail: "Estimate", exact: false };
  }
  return { text: "—", detail: "Not revealed", exact: false };
}

function truncateScoutLine(text, max = 148) {
  const s = String(text || "").trim();
  if (!s) return null;
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1).trim()}…`;
}

function ValueTrajectoryChart({ points, compact = false }) {
  if (!points?.length) {
    return (
      <div className={`dc-signal-chart dc-signal-chart--empty${compact ? " is-compact" : ""}`}>
        <span className="dc-profile-tags__label">Value trajectory</span>
        <p>No board movement history yet</p>
      </div>
    );
  }
  const ranks = points.map((p) => p.rank);
  const minR = Math.min(...ranks);
  const maxR = Math.max(...ranks);
  // Keep a usable vertical range even on flat seasons so the line isn't a 1px stub.
  const pad = Math.max(2, Math.round((maxR - minR) * 0.18) || 3);
  const yMin = Math.max(1, minR - pad);
  const yMax = Math.max(yMin + 4, maxR + pad);
  const w = compact ? 220 : 320;
  const h = compact ? 48 : 72;
  const left = 8;
  const right = 8;
  const top = 8;
  const bottom = 8;
  const spanX = Math.max(1, points.length - 1);
  const coords = points.map((pt, i) => {
    const x = left + ((w - left - right) * i) / spanX;
    // Lower board rank (#1) sits higher on the chart.
    const t = (pt.rank - yMin) / Math.max(1e-6, yMax - yMin);
    const y = top + t * (h - top - bottom);
    return { x, y, ...pt };
  });
  const poly = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const area = `${left},${h - bottom} ${poly} ${coords[coords.length - 1].x.toFixed(1)},${h - bottom}`;
  const first = coords[0];
  const last = coords[coords.length - 1];
  const rising = last.rank < first.rank;
  const falling = last.rank > first.rank;
  const gradId = compact ? "dcTrailGlowCompact" : "dcTrailGlow";
  const fillId = compact ? "dcTrailFillCompact" : "dcTrailFill";
  return (
    <div className={`dc-signal-chart${compact ? " is-compact" : ""}`}>
      <div className="dc-signal-chart__head">
        <span className="dc-profile-tags__label">Value trajectory</span>
        <strong className={rising ? "is-up" : falling ? "is-down" : ""}>
          {rising ? "RISING" : falling ? "FALLING" : "FLAT"}
        </strong>
        <span className="dc-signal-chart__delta">#{last.rank}</span>
      </div>
      <svg className="dc-signal-chart__svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(43,228,255,0.55)" />
            <stop offset="100%" stopColor="rgba(244,198,110,1)" />
          </linearGradient>
          <linearGradient id={fillId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(43,228,255,0.22)" />
            <stop offset="100%" stopColor="rgba(43,228,255,0)" />
          </linearGradient>
        </defs>
        <polygon fill={`url(#${fillId})`} points={area} />
        <polyline fill="none" stroke={`url(#${gradId})`} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" points={poly} />
        {coords.map((c, i) => (
          <circle
            key={`${c.rank}-${i}`}
            cx={c.x}
            cy={c.y}
            r={i === coords.length - 1 ? 3.6 : 2.4}
            fill={i === coords.length - 1 ? "#f4c66e" : "#2be4ff"}
            stroke="rgba(3,10,18,0.85)"
            strokeWidth="1"
          />
        ))}
      </svg>
      {!compact ? (
        <div className="dc-signal-chart__axis">
          <span>{first.label}</span>
          <span>{last.label}</span>
        </div>
      ) : null}
    </div>
  );
}

function SignalMetricTile({ label, value, tone }) {
  const empty = value == null || value === "" || value === "—";
  return (
    <div className={`dc-signal-metric${empty ? " is-empty" : ""}${tone ? ` is-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{empty ? "—" : value}</strong>
    </div>
  );
}

function SignalNationFlag({ country, size = 64, className = "" }) {
  const shiny = flagApiUrl(country, size, "shiny");
  const flat = flagApiUrl(country, size, "flat");
  const [src, setSrc] = useState(shiny || flat);
  useEffect(() => {
    setSrc(shiny || flat);
  }, [shiny, flat]);
  if (!src) {
    return <span className={`dc-signal-flag-fallback ${className}`}>{String(country || "—").slice(0, 3).toUpperCase()}</span>;
  }
  return (
    <img
      className={`dc-signal-flag ${className}`}
      src={src}
      alt=""
      loading="lazy"
      onError={(e) => {
        if (flat && e.currentTarget.src.indexOf("/flat/") === -1) {
          e.currentTarget.src = flat;
        }
      }}
    />
  );
}

function positionDisplayName(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "C") return "CENTRE";
  if (p === "LW") return "LEFT WING";
  if (p === "RW") return "RIGHT WING";
  if (p === "D" || p === "LD" || p === "RD") return "DEFENCE";
  if (p === "G") return "GOALTENDER";
  if (p === "F" || p === "W") return "FORWARD";
  return p || "—";
}

function heightWithCm(height) {
  const raw = String(height || "").trim();
  if (!raw) return { imperial: "—", metric: null };
  const m = raw.match(/(\d)\s*['′]\s*(\d{1,2})/);
  if (!m) return { imperial: raw, metric: null };
  const cm = Math.round((Number(m[1]) * 12 + Number(m[2])) * 2.54);
  return { imperial: `${m[1]}'${m[2]}"`, metric: Number.isFinite(cm) ? `${cm} CM` : null };
}

function weightWithKg(weight) {
  const n = Number(String(weight || "").replace(/[^\d.]/g, ""));
  if (!Number.isFinite(n) || n <= 0) return { lbs: "—", kg: null };
  return { lbs: `${Math.round(n)} LBS`, kg: `${Math.round(n / 2.20462)} KG` };
}

function resolveToolRows(player, profile) {
  const lift = prospectAttributeLift(profile);
  const ceilingHidden = Boolean(profile?.ceilingHidden || profile?.potential?.hidden);
  const dedicatedFile = Boolean(profile?.dedicatedScoutFile);
  const wideFog = ceilingHidden && !dedicatedFile;
  const ambient = Number(profile?.scout_confidence ?? player?.scoutingConfidence ?? player?.completion);
  const userScout = Number(profile?.userScoutPct ?? player?.scoutedPercentage);
  // Prefer ambient board confidence so Skill DNA is never blanked by scouted_percentage=0.
  let completion = Math.max(
    Number.isFinite(ambient) ? ambient : 0,
    (Number.isFinite(userScout) && userScout > 0) ? userScout : 0,
    52
  );
  if (wideFog) {
    // Keep preliminary ranges visible under ceiling fog; do not collapse to "?".
    completion = Math.min(Math.max(completion, 52), 66);
  }
  const bump = (val) => {
    const n = Number(val);
    if (!Number.isFinite(n)) return val;
    return Math.min(94, Math.round(n + lift));
  };
  const pickAttr = (...keys) => {
    for (const k of keys) {
      const v = player?.[k] ?? profile?.[k] ?? profile?.attributes?.[k] ?? profile?.tools?.[k];
      if (v != null && Number.isFinite(Number(v)) && Number(v) > 0) return Number(v);
    }
    return null;
  };
  return [
    ["Skating", bump(pickAttr("skating", "Skating")), 1],
    ["Shot", bump(pickAttr("shooting", "shot", "Shooting")), 2],
    ["Vision", bump(pickAttr("passing", "vision", "Passing")), 3],
    ["Defense", bump(pickAttr("defense", "Defence")), 4],
    ["Physical", bump(pickAttr("physical", "Physical")), 5],
    ["IQ", bump(pickAttr("hockeyIQ", "hockey_iq", "iq")), 6],
  ].map(([label, val, seed]) => {
    const display = attributeDisplay(val, completion, seed, { wideFog });
    const mid = display.range
      ? (display.range[0] + display.range[1]) / 2
      : (display.locked ? null : Number(val));
    return {
      label,
      text: display.text,
      locked: display.locked,
      mid: Number.isFinite(mid) ? mid : null,
      low: display.range ? display.range[0] : null,
      high: display.range ? display.range[1] : null,
    };
  });
}

function SkillDnaRadar({ tools, compositeText }) {
  const size = 240;
  const cx = size / 2;
  const cy = size / 2;
  const rMax = 72;
  const n = 6;
  const angleAt = (i) => (-Math.PI / 2) + (i * 2 * Math.PI) / n;
  const pt = (i, radius) => {
    const a = angleAt(i);
    return [cx + Math.cos(a) * radius, cy + Math.sin(a) * radius];
  };
  const rings = [0.35, 0.6, 0.85, 1];
  const values = tools.map((t) => (t.mid != null ? Math.max(0, Math.min(99, t.mid)) / 99 : 0));
  const hasSignal = tools.some((t) => t.mid != null && !t.locked);
  const poly = values
    .map((v, i) => {
      const [x, y] = pt(i, rMax * (hasSignal ? Math.max(0.12, v) : 0.12));
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="dc-skill-dna">
      <span className="dc-profile-tags__label">Skill DNA</span>
      <div className="dc-skill-dna__stage">
        <svg viewBox={`0 0 ${size} ${size}`} className="dc-skill-dna__svg" aria-hidden="true">
          {rings.map((scale) => (
            <polygon
              key={scale}
              className="dc-skill-dna__ring"
              points={Array.from({ length: n }, (_, i) => pt(i, rMax * scale).map((v) => v.toFixed(1)).join(",")).join(" ")}
            />
          ))}
          {tools.map((t, i) => {
            const [x2, y2] = pt(i, rMax);
            return <line key={t.label} className="dc-skill-dna__spoke" x1={cx} y1={cy} x2={x2} y2={y2} />;
          })}
          {hasSignal ? (
            <polygon className="dc-skill-dna__fill" points={poly} />
          ) : null}
          {tools.map((t, i) => {
            const [x, y] = pt(i, rMax + 22);
            return (
              <text key={`lab-${t.label}`} x={x} y={y} className="dc-skill-dna__label" textAnchor="middle" dominantBaseline="middle">
                {t.label}
              </text>
            );
          })}
          {tools.map((t, i) => {
            if (t.mid == null) return null;
            const [x, y] = pt(i, rMax * (t.mid / 99));
            return <circle key={`n-${t.label}`} cx={x} cy={y} r="3" className="dc-skill-dna__node" />;
          })}
        </svg>
        <div className="dc-skill-dna__core">
          <strong className={String(compositeText || "").length > 3 ? "is-range" : ""}>
            {compositeText || "—"}
          </strong>
          <span>OVERALL</span>
        </div>
      </div>
      <div className="dc-skill-dna__ranges">
        {tools.map((t) => (
          <div key={t.label} className={t.locked ? "is-locked" : ""}>
            <span>{t.label}</span>
            <strong>
              {t.locked
                ? "?"
                : (t.low != null && t.high != null
                  ? `${t.low}-${t.high}`
                  : t.text)}
            </strong>
            <em>OVR {t.locked || t.mid == null ? "—" : Math.round(t.mid)}</em>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProjectionEngineBar({ label, value, display, tone = "cyan", max = 100 }) {
  const n = Number(value);
  const ok = Number.isFinite(n);
  const pct = ok ? Math.max(0, Math.min(100, (n / max) * 100)) : 0;
  return (
    <div className={`dc-proj-engine__bar dc-proj-engine__bar--${tone}${ok ? "" : " is-empty"}`}>
      <div className="dc-proj-engine__bar-head">
        <span>{label}</span>
        <strong>{ok ? (display ?? Math.round(n)) : "—"}</strong>
      </div>
      <div className="dc-proj-engine__track">
        <i style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SegmentDots({ filled = 0, tone = "cyan", total = 10 }) {
  const n = Math.max(0, Math.min(total, Math.round(Number(filled) || 0)));
  return (
    <div className={`dc-segment-dots dc-segment-dots--${tone}`} aria-hidden="true">
      {Array.from({ length: total }, (_, i) => (
        <i key={i} className={i < n ? "is-on" : ""} />
      ))}
    </div>
  );
}

function DecisionLensRow({ label, value, dots, tone = "cyan" }) {
  const empty = value == null || value === "" || value === "—";
  return (
    <div className={`dc-lens-row${empty ? " is-empty" : ""}`}>
      <div className="dc-lens-row__copy">
        <span>{label}</span>
        <strong>{empty ? "—" : value}</strong>
      </div>
      <SegmentDots filled={empty ? 0 : dots} tone={tone} />
    </div>
  );
}

function ScoutingTrail({ points, confPct }) {
  const nodes = (points || []).slice(-6);
  const completed = nodes.length;
  const target = 6;
  if (!nodes.length) {
    return (
      <div className="dc-scout-trail dc-scout-trail--empty">
        <span className="dc-profile-tags__label">Scouting trail</span>
        <p>No viewing trail recorded</p>
        <div className="dc-scout-trail__progress">
          <span>Viewings completed — / {target}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="dc-scout-trail">
      <div className="dc-scout-trail__head">
        <span className="dc-profile-tags__label">Scouting trail</span>
        <span className="dc-scout-trail__progress">
          Viewings completed {Math.min(completed, target)} / {target}
        </span>
      </div>
      <div className="dc-scout-trail__track">
        {nodes.map((n, i) => (
          <div key={`${n.label}-${i}`} className={`dc-scout-trail__node${i === nodes.length - 1 ? " is-active" : ""}`}>
            <i />
            <span>{n.label}</span>
            <strong>#{n.rank}</strong>
          </div>
        ))}
      </div>
      <div className="dc-scout-trail__bar">
        <i style={{ width: `${Math.round((Math.min(completed, target) / target) * 100)}%` }} />
      </div>
      {confPct != null ? (
        <small className="dc-scout-trail__conf">{confPct}% file clarity</small>
      ) : null}
    </div>
  );
}

function upsideImpactFromProfile(ceilingOvr, franchiseTier, rank, ceilingHidden) {
  const band = peakProjectionBand(ceilingOvr, ceilingHidden);
  return { text: band.upside, dots: band.upsideDots, tone: band.tone };
}

/** NHL role from peak OVR — kept in lockstep with upsideImpactFromProfile bands. */
function projectedRoleFromPeak(ceilingOvr, position, ceilingHidden) {
  const band = peakProjectionBand(ceilingOvr, ceilingHidden);
  if (band.role === "—") return { text: "—", dots: 0, tone: "muted" };
  const pos = String(position || "").toUpperCase();
  let role = band.role;
  if (pos === "G") {
    if (role === "Franchise") role = "Franchise";
    else if (role === "Elite" || role === "Top Line") role = "Starter";
    else if (role === "Top 6") role = "Tandem";
    else if (role === "Middle 6") role = "Bench";
    else role = "AHL Starter";
  } else if (pos === "D" || pos === "LD" || pos === "RD") {
    if (role === "Top Line") role = "Top 2";
    else if (role === "Top 6") role = "Top 4";
    else if (role === "Middle 6") role = "Bottom 4";
    else if (role === "Bottom 6") role = "Top 6";
  }
  return { text: String(role).toUpperCase(), dots: band.roleDots, tone: band.tone === "gold" ? "gold" : "cyan" };
}

/** Draft pick window from board rank (not the same as NHL role). */
function draftWindowFromRank(rank) {
  const r = Number(rank);
  if (!Number.isFinite(r) || r <= 0) return "—";
  if (r <= 2) return "Top 2";
  if (r <= 5) return "Lottery";
  if (r <= 10) return "Top 10";
  if (r <= 15) return "Mid 1st";
  if (r <= 32) return "1st Round";
  if (r <= 64) return "2nd Round";
  if (r <= 96) return "3rd Round";
  return "Mid/Late";
}

/**
 * Shared peak-OVR bands so Upside Impact and Projected Role never disagree.
 * Franchise role only at true franchise ceilings (92+).
 */
function peakProjectionBand(ceilingOvr, ceilingHidden) {
  if (ceilingHidden) {
    return { upside: "—", role: "—", upsideDots: 0, roleDots: 0, tone: "muted" };
  }
  const pot = Number(ceilingOvr);
  if (!Number.isFinite(pot) || pot <= 0) {
    return { upside: "—", role: "—", upsideDots: 0, roleDots: 0, tone: "muted" };
  }
  if (pot >= 92) {
    return { upside: "FRANCHISE ALTERING", role: "Franchise", upsideDots: 10, roleDots: 10, tone: "gold" };
  }
  if (pot >= 88) {
    return { upside: "ELITE IMPACT", role: "Elite", upsideDots: 9, roleDots: 9, tone: "gold" };
  }
  if (pot >= 84) {
    return { upside: "HIGH UPSIDE", role: "Top Line", upsideDots: 7, roleDots: 7, tone: "cyan" };
  }
  if (pot >= 80) {
    return { upside: "STRONG UPSIDE", role: "Top 6", upsideDots: 6, roleDots: 6, tone: "cyan" };
  }
  if (pot >= 75) {
    return { upside: "MODERATE UPSIDE", role: "Middle 6", upsideDots: 5, roleDots: 5, tone: "cyan" };
  }
  if (pot >= 70) {
    return { upside: "DEPTH UPSIDE", role: "Bottom 6", upsideDots: 3, roleDots: 3, tone: "violet" };
  }
  return { upside: "LIMITED UPSIDE", role: "Depth", upsideDots: 2, roleDots: 2, tone: "violet" };
}

function riskTemperatureFromGem(gem, riskLabel, volatility) {
  const raw = String(gem?.label || riskLabel || volatility || "").trim();
  if (!raw) return { text: "—", dots: 0, tone: "muted" };
  const lower = raw.toLowerCase();
  if (lower.includes("bust") || lower.includes("elevat") || lower === "high") {
    return { text: "ELEVATED", dots: 8, tone: "violet" };
  }
  if (lower.includes("uncertain") || lower.includes("volatile")) {
    return { text: "VOLATILE", dots: 6, tone: "gold" };
  }
  if (lower.includes("gem") || lower.includes("safe") || lower === "low") {
    return { text: "LOW", dots: 2, tone: "cyan" };
  }
  if (lower.includes("neutral") || lower.includes("medium") || lower === "mod") {
    return { text: "NEUTRAL", dots: 5, tone: "muted" };
  }
  const tier = Number(gem?.tier);
  if (Number.isFinite(tier)) {
    return { text: raw.toUpperCase(), dots: Math.max(1, Math.min(10, tier * 2 + 2)), tone: "violet" };
  }
  return { text: raw.toUpperCase(), dots: 5, tone: "muted" };
}

function competitionLens(competition) {
  if (!competition || typeof competition !== "object") return { text: "—", dots: 0, tone: "muted" };
  const score = Number(competition.level_score);
  const label = String(competition.label || "").toUpperCase() || "—";
  let band = "MODERATE";
  let dots = 5;
  if (Number.isFinite(score)) {
    if (score >= 75) band = "HIGH";
    else if (score >= 60) band = "SOLID";
    else if (score < 45) band = "LIGHT";
    dots = Math.max(1, Math.min(10, Math.round(score / 10)));
  } else if (label && label !== "—") {
    const lower = label.toLowerCase();
    if (lower.includes("high") || lower.includes("elite")) { band = "HIGH"; dots = 8; }
    else if (lower.includes("solid") || lower.includes("strong")) { band = "SOLID"; dots = 6; }
    else if (lower.includes("light") || lower.includes("low")) { band = "LIGHT"; dots = 3; }
    else { band = label; dots = 5; }
  } else {
    return { text: "—", dots: 0, tone: "muted" };
  }
  return { text: band, dots, tone: "cyan", detail: label };
}

function ProspectProfileModal({
  player,
  profile,
  meta,
  onClose,
  compareIds = [],
  onToggleWatchlist,
  onToggleTarget,
  onToggleDND,
  onToggleCompare,
  onAssignScout,
}) {
  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const isTranscendent = Boolean(profile?.is_transcendent || profile?.transcendent_talent || player?.isTranscendent);

  useEffect(() => {
    if (!isTranscendent) return undefined;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return undefined;
    document.body.classList.add("dc-transcendent-shake");
    const t = window.setTimeout(() => document.body.classList.remove("dc-transcendent-shake"), 600);
    return () => {
      window.clearTimeout(t);
      document.body.classList.remove("dc-transcendent-shake");
    };
  }, [isTranscendent, player?.id]);

  if (!player) return null;

  const scoutMeta = meta || EMPTY_SCOUTING_META;
  const stats = profile?.stats || {};
  const gp = Number(stats.games ?? player.gp) || 0;
  const pts = Number(stats.points ?? player.points) || 0;
  const goals = Number(stats.goals ?? player.goals) || 0;
  const assists = Number(stats.assists ?? player.assists) || 0;
  const ppg = stats.ppg != null ? Number(stats.ppg) : (gp > 0 ? prospectPpgValue(player) : null);
  const isGoalie = isGoaliePosition(player.position);

  const pot = profile?.potential;
  const fit = profile?.team_fit || profile?.teamFit;
  const proj = profile?.projection;
  const gem = profile?.gem;
  const comparison = profile?.player_comparison;
  const competition = profile?.competition;
  const strengths = Array.isArray(profile?.strengths) ? profile.strengths : [];
  const tags = getPlayerTags(player, profile).filter((t) => {
    const primary = String(strengths[0] || "").toLowerCase();
    return !primary || String(t).toLowerCase() !== primary;
  });
  const scoutConf = Number(profile?.scout_confidence ?? player?.scoutingConfidence ?? player?.completion);
  const confPct = Number.isFinite(scoutConf) ? Math.round(scoutConf) : null;
  const intelTier = profile?.intel_label || prospectIntelTier(confPct);
  const team = cleanTeamName(player, profile);
  const league = cleanLeagueName(player, profile);
  const countryLabel = countryDisplayLabel(player, profile);
  const badges = getPlayerIdentityBadges(player, profile);
  const rank = prospectRank(player);
  const stock = player.draftStock || null;
  const trend = movementDisplayText(stock);
  const trendTone = getStockTone(stock);
  const fitReasons = (Array.isArray(fit?.reasons) ? fit.reasons : [])
    .map(humanizeScoutReason)
    .filter(Boolean);
  const fitNote = fitReasons.length
    ? fitReasons.slice(0, 2).join(" · ")
    : (fit?.label ? `${fit.label} organizational match` : null);
  const ceilingHidden = Boolean(profile?.ceilingHidden || pot?.hidden || player?.ceilingHidden);
  const dedicatedFile = Boolean(profile?.dedicatedScoutFile);
  const confNote = confPct == null
    ? null
    : ceilingHidden && !dedicatedFile
      ? "INCOMPLETE READ"
      : confPct >= 91
        ? "LOCKED READ"
        : confPct >= 56
          ? `${String(intelTier || "SOLID").toUpperCase()} COVERAGE`
          : "LIMITED LOOKS";
  const compareFull = (compareIds || []).length >= 3;
  const inCompare = (compareIds || []).includes(player.id);
  const reportBlurb = truncateScoutLine(scoutSummary(player, profile), 168);

  const ceilingRating = Number(pot?.rating);
  const currentOvr = Number(profile?.scoutedOverall ?? profile?.currentOvrEstimate);
  const readinessLabel = nhlReadinessLabel(rank, ceilingRating, currentOvr, profile);
  const sampleThin = Boolean(profile?.sampleThin || stats?.sampleThin || (gp > 0 && gp < 15));
  const playStyle = comparison?.archetype || player.playerType || null;
  const riskLabel = gem?.label || player.riskLabel || null;
  const volatility = profile?.developmentVolatility || null;
  const currentEstimate = resolveCurrentEstimate(player, profile, ceilingHidden, dedicatedFile);
  const trajectory = buildValueTrajectoryPoints(profile, player);
  const analytics = extractProspectAnalytics(player, profile);
  const tools = resolveToolRows(player, profile);
  const ht = heightWithCm(badges.height || player.height);
  const wt = weightWithKg(badges.weight || player.weight);
  const handRaw = badges.handedness || formatHandedness(player.handedness);
  const shoots = handRaw
    ? (String(handRaw).toUpperCase().startsWith("L") ? "SHOOTS LEFT" : "SHOOTS RIGHT")
    : "—";
  const ageNum = badges.age ? String(badges.age).replace(/Y$/i, "") : (player.age != null ? String(player.age) : "—");
  const posName = positionDisplayName(badges.position || player.position);
  const peakVal = !ceilingHidden && pot?.rating != null ? Number(pot.rating) : null;
  const volatilityRaw = !ceilingHidden ? String(profile?.developmentVolatility || "").trim() : "";
  const volatilityDisplay = (!volatilityRaw || volatilityRaw.toLowerCase() === "unknown")
    ? "—"
    : volatilityRaw.toUpperCase();
  const floorVal = !ceilingHidden && pot?.floor != null ? Number(pot.floor) : null;
  const boardYear = profile?.draft_year || player?.draftYear || null;
  const upside = upsideImpactFromProfile(peakVal ?? ceilingRating, player.franchiseTier, rank, ceilingHidden);
  const roleLens = projectedRoleFromPeak(peakVal ?? ceilingRating, player.position, ceilingHidden);
  const draftWindow = draftWindowFromRank(rank);
  const riskTemp = riskTemperatureFromGem(gem, riskLabel, volatility);
  const compLens = competitionLens(competition);
  const roleDots = roleLens.dots;
  const confDots = confPct != null ? Math.max(0, Math.min(10, Math.round(confPct / 10))) : 0;
  const confBand = confPct == null
    ? "—"
    : confPct >= 72
      ? "HIGH"
      : confPct >= 45
        ? "MODERATE"
        : "LOW";

  const warTone = (() => {
    const w = Number(analytics.war);
    if (!Number.isFinite(w)) return null;
    if (w >= 2.2 || profile?.analytics?.analytics_signal === "gem_finder") return "gem";
    if (w >= 1.2) return "hot";
    if (w < 0) return "cold";
    return null;
  })();

  const analyticsTiles = isGoalie
    ? [
        { label: "GP", value: gp > 0 ? String(gp) : "—" },
        { label: "W", value: formatSignalNum(player.wins, 0) },
        { label: "SV%", value: player.savePct != null ? String(player.savePct) : "—" },
        { label: "GAA", value: player.gaa != null ? String(player.gaa) : "—" },
        { label: "GSAx", value: formatSignalSigned(analytics.gsax, 2) },
        { label: "QS", value: formatSignalNum(analytics.quality_starts, 0) },
        { label: "SO", value: formatSignalNum(player.shutouts, 0) },
        { label: "TOI", value: analytics.toi != null ? formatSignalNum(analytics.toi, 1) : "—" },
      ]
    : [
        { label: "GP", value: gp > 0 ? String(gp) : "—" },
        { label: "G-A-P", value: gp > 0 ? `${goals}-${assists}-${pts}` : "—" },
        { label: "P/GP", value: ppg != null && Number.isFinite(ppg) && gp > 0 ? Number(ppg).toFixed(2) : "—" },
        { label: "SH%", value: formatSignalPct(analytics.shooting_pct) },
        { label: "Prim P", value: formatSignalNum(analytics.primary_points, 0) },
        { label: "WAR", value: formatSignalSigned(analytics.war, 2), tone: warTone },
        { label: "xGF%", value: formatSignalPct(analytics.xgf_pct) },
        { label: "CF%", value: formatSignalPct(analytics.cf_pct) },
        { label: "Shots/G", value: formatSignalNum(analytics.shot_rate, 2) },
        { label: "TOI", value: analytics.toi != null ? formatSignalNum(analytics.toi, 1) : "—" },
        { label: "+/-", value: formatSignalSigned(analytics.plus_minus, 0) },
      ];

  return (
    <div
      className={`dc-profile-modal dc-profile-modal--prospect${isTranscendent ? " prospect-modal--transcendent" : ""}${rank > 32 || (confPct != null && confPct < 45) ? " dc-profile-modal--uncertain" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-label={`${player.firstName} ${player.lastName} scouting profile`}
    >
      <button type="button" className="dc-profile-modal__backdrop" onClick={onClose} aria-label="Close" />
      <article className={`dc-signal-panel dc-signal-panel--premium${isTranscendent ? " aura-gold shake-on-open" : ""}`}>
        <ModalCloseButton onClick={onClose} label="Close prospect profile" />
        <header className="dc-signal-banner">
          <span>Franchise Intelligence</span>
          <strong>Prospect Scouting</strong>
        </header>

        {!profile ? (
          <p className="dc-empty-note dc-profile-modal__loading">Scouting profile loading…</p>
        ) : (
          <>
            <aside className="dc-signal-identity">
              <div className="dc-signal-portrait">
                <DraftClassHeadshot player={player} size="lg" />
                <SignalNationFlag country={countryLabel} size={64} className="dc-signal-portrait__flag" />
                <div className={`dc-signal-portrait__stock is-${trendTone || "muted"}`}>
                  <strong>#{rank}</strong>
                  <span>{trend}</span>
                </div>
              </div>

              <div className="dc-signal-identity__title">
                <h2>{player.firstName} {player.lastName}</h2>
                <p className="dc-signal-identity__pos">{posName}</p>
              </div>

              <ul className="dc-signal-bio">
                <li>
                  <SignalNationFlag country={countryLabel} size={32} />
                  <div>
                    <span>Nation</span>
                    <strong>{String(countryLabel || "—").toUpperCase()}</strong>
                  </div>
                </li>
                <li>
                  <div>
                    <span>Handedness</span>
                    <strong>{shoots}</strong>
                  </div>
                </li>
                <li>
                  <div>
                    <span>Age</span>
                    <strong>{ageNum !== "—" ? `AGE ${ageNum}` : "—"}</strong>
                  </div>
                </li>
                <li>
                  <div>
                    <span>Height</span>
                    <strong>{ht.imperial}{ht.metric ? ` / ${ht.metric}` : ""}</strong>
                  </div>
                </li>
                <li>
                  <div>
                    <span>Weight</span>
                    <strong>{wt.lbs}{wt.kg ? ` / ${wt.kg}` : ""}</strong>
                  </div>
                </li>
              </ul>

              <div className="dc-signal-board-card">
                <strong>#{rank}</strong>
                <span>{boardYear ? `${boardYear} Draft Board` : "Draft Board"}</span>
                <em className={`is-${trendTone || "muted"}`}>Stock {trend}</em>
              </div>

              <div className="dc-signal-club-card dc-signal-club-card--text">
                <div>
                  <strong>{team && team !== "—" ? team : "—"}</strong>
                  <span>{league || "—"}</span>
                </div>
              </div>
            </aside>

            <main className="dc-signal-core">
              <header className="dc-signal-core__head">
                <div>
                  <span className="dc-profile-tags__label">Scouting signal</span>
                  <h3>Skill & projection</h3>
                </div>
                {playStyle ? <ProfileChip tone="accent">{String(playStyle).toUpperCase()}</ProfileChip> : null}
              </header>

              <div className="dc-signal-core__grid">
                <SkillDnaRadar
                  tools={tools}
                  compositeText={currentEstimate.text}
                />

                <section className="dc-proj-engine">
                  <span className="dc-profile-tags__label">Projection engine</span>
                  <ProjectionEngineBar
                    label="Peak Forecast"
                    value={peakVal}
                    display={peakVal != null ? Math.round(peakVal) : "—"}
                    tone="gold"
                    max={99}
                  />
                  <ProjectionEngineBar
                    label="Baseline Outcome"
                    value={floorVal}
                    display={floorVal != null ? Math.round(floorVal) : "—"}
                    tone="cyan"
                    max={99}
                  />
                  <div className="dc-proj-engine__meta">
                    <div>
                      <span>Draft Window</span>
                      <strong>{draftWindow}</strong>
                    </div>
                    <div>
                      <span>Projected Role</span>
                      <strong>{roleLens.text !== "—" ? roleLens.text : (playStyle || "—")}</strong>
                    </div>
                    <div>
                      <span>Arrival Window</span>
                      <strong>{readinessLabel || "—"}</strong>
                    </div>
                    <div>
                      <span>Volatility</span>
                      <strong className={volatilityDisplay !== "—" ? "is-warn" : ""}>
                        {volatilityDisplay}
                      </strong>
                    </div>
                  </div>
                </section>
              </div>
            </main>

            <aside className="dc-signal-decision">
              <header className="dc-profile-zone-head">Decision lens</header>
              <DecisionLensRow
                label="Upside Impact"
                value={upside.text}
                dots={upside.dots}
                tone={upside.tone || "gold"}
              />
              <DecisionLensRow
                label="Projected Role"
                value={roleLens.text}
                dots={roleDots}
                tone={roleLens.tone || "cyan"}
              />
              <DecisionLensRow
                label="Risk Temperature"
                value={riskTemp.text}
                dots={riskTemp.dots}
                tone={riskTemp.tone || "violet"}
              />
              <DecisionLensRow
                label="Competition Level"
                value={compLens.text}
                dots={compLens.dots}
                tone="cyan"
              />
              <DecisionLensRow
                label="Scouting Confidence"
                value={confBand}
                dots={confDots}
                tone="violet"
              />
            </aside>

            <footer className="dc-signal-footer dc-signal-footer--premium">
              <div className="dc-signal-footer__stock">
                <ValueTrajectoryChart points={trajectory} />
              </div>
              <div className="dc-signal-season">
                <div className="dc-signal-season__head">
                  <span className="dc-profile-tags__label">Season & analytics</span>
                  {sampleThin ? <span className="dc-signal-sample">Thin sample</span> : null}
                </div>
                <div className="dc-signal-metrics">
                  {analyticsTiles.map((tile) => (
                    <SignalMetricTile key={tile.label} label={tile.label} value={tile.value} tone={tile.tone} />
                  ))}
                </div>
              </div>
              <div className="dc-signal-actionbar dc-signal-actionbar--icon">
                <button
                  type="button"
                  className={`dc-shortlist-icon${scoutMeta.watchlist ? " is-active" : ""}`}
                  onClick={onToggleWatchlist}
                  disabled={!onToggleWatchlist}
                  aria-label={scoutMeta.watchlist ? "Remove from shortlist" : "Add to shortlist"}
                  title={scoutMeta.watchlist ? "Remove from shortlist" : "Add to shortlist"}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M6 3.5h12a1 1 0 0 1 1 1V21l-7-4.2L5 21V4.5a1 1 0 0 1 1-1z" />
                  </svg>
                </button>
              </div>
            </footer>
          </>
        )}
      </article>
    </div>
  );
}

function SelectedProspectCommandCard({
  player, meta, dateContext, draftYear, allProspects, scoutingActions, compareIds, onRemoveCompare, onClearCompare,
}) {
  const [tab, setTab] = useState("OVERVIEW");
  const [openFile, setOpenFile] = useState(false);
  if (!player) {
    return (
      <section className="dc-selected-file dc-selected-file--empty">
        <p>Select a prospect from the tier board.</p>
      </section>
    );
  }

  const isGoalie = player.position === "G";
  const ppg = player.gp > 0 ? (player.points / player.gp).toFixed(2) : "—";
  const concern = player.characterConcerns
    ? "Character concerns flagged by backend scouting."
    : player.isBustRisk
    ? "Bust risk flagged — variance in projection."
    : player.riskLabel === "High"
    ? "High risk profile per backend risk label."
    : "No major red flags in current file.";

  return (
    <section className="dc-selected-file">
      <header className="dc-selected-file__hero">
        <DraftClassHeadshot player={player} size="lg" />
        <div className="dc-selected-file__identity">
          <span className="dc-selected-file__rank">#{player.rank}</span>
          <h2>{player.firstName} {player.lastName}</h2>
          <p>{player.position} · {countryFlag(player.country)} · {player.league}</p>
        </div>
        <div className="dc-selected-file__quick">
          <div><span>Top</span><strong>{player.projection}</strong></div>
          <div><span>Grade</span><strong>{player.talent}</strong></div>
          <div><span>Scout</span><strong>{player.completion}%</strong></div>
        </div>
      </header>

      <div className="dc-selected-file__grid">
        <article className="dc-glass-card">
          <h3>Stats</h3>
          {!isGoalie ? (
            <p>{player.gp} GP · {player.points} PTS · {ppg} P/GP</p>
          ) : (
            <p>{player.gp} GP · {player.wins} W · SV% {player.savePct || "—"}</p>
          )}
        </article>
        <article className="dc-glass-card">
          <h3>Tier</h3>
          <TierBadge tier={player.franchiseTier} />
        </article>
        <article className="dc-glass-card">
          <h3>Stock</h3>
          <StockBadge stock={player.draftStock} />
          <p>{player.draftStock?.available ? stockBadgeText(player.draftStock) : "—"}</p>
        </article>
        <article className="dc-glass-card">
          <h3>Concern</h3>
          <p>{player.characterConcerns ? "⚠ Character" : player.isBustRisk ? "⚠ Risk" : "✓ Clear"}</p>
        </article>
        <article className="dc-glass-card">
          <h3>Next</h3>
          <p>{meta.doNotDraft ? "Hold" : meta.target ? "Pin locked" : "Scout"}</p>
        </article>
      </div>

      <div className="dc-action-strip">{scoutingActions}</div>

      <div className="dc-file-toggle-wrap">
        <button type="button" className={`dc-btn dc-btn--secondary ${openFile ? "is-active" : ""}`} onClick={() => setOpenFile((v) => !v)}>
          {openFile ? "Close File" : "Open File"}
        </button>
      </div>

      {openFile ? (
        <>
          <div className="dc-selected-file__tabs">
            {PROFILE_TABS.map((t) => (
              <button key={t} type="button" className={`dc-profile-tab ${tab === t ? "is-active" : ""}`} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
          </div>

          <ComparisonTray compareIds={compareIds} prospects={allProspects} onRemove={onRemoveCompare} onClear={onClearCompare} />

          <div className="dc-selected-file__detail dc-scroll-surface">
            {tab === "OVERVIEW" && <OverviewTab player={player} meta={meta} dateContext={dateContext} draftYear={draftYear} />}
            {tab === "STATS" && <StatsTab player={player} dateContext={dateContext} allProspects={allProspects} />}
            {tab === "ATTRIBUTES" && <AttributesTab player={player} />}
            {tab === "SCOUT REPORT" && <ScoutReportTab player={player} meta={meta} profile={player.profile} />}
            {tab === "CHARACTER" && <CharacterTab player={player} meta={meta} profile={player.profile} />}
          </div>
        </>
      ) : null}
    </section>
  );
}

function StockExchangeRail({ boardMeta, prospects, onSelectPlayer, leadersProps }) {
  const summary = useMemo(() => {
    const movers = buildStockMoversFromProspects(prospects);
    if (movers.risers.length || movers.fallers.length) return movers;
    const backend = boardMeta?.stock_market_summary || boardMeta?.stockMarketSummary;
    if (backend?.source === "backend") return backend;
    return movers;
  }, [prospects, boardMeta?.stock_market_summary, boardMeta?.stockMarketSummary]);
  const hasSummary = Boolean(summary?.risers?.length || summary?.fallers?.length);

  const renderList = (title, items, tone) => {
    if (!items?.length) return null;
    return (
      <div className={`dc-stock-card dc-stock-card--${tone}`}>
        <h4>{title}</h4>
        {items.slice(0, 6).map((item) => {
          const id = item.key;
          const deltaRaw = item.delta_rank ?? item.deltaRank;
          const delta = Number.isFinite(Number(deltaRaw)) ? Number(deltaRaw) : 0;
          return (
            <button key={`${title}-${id}`} type="button" className="dc-stock-row" onClick={() => id && onSelectPlayer(id)}>
              <span>#{item.rank}</span>
              <span className="dc-stock-row__name">{item.name}</span>
              <span className={`dc-stock-row__delta ${delta > 0 ? "is-up" : delta < 0 ? "is-down" : ""}`}>
                {delta > 0 ? `+${delta}` : delta < 0 ? delta : "—"}
              </span>
            </button>
          );
        })}
      </div>
    );
  };

  return (
    <aside className="dc-stock-rail dc-scroll-surface">
      <header className="dc-stock-rail__head">
        <h2>Stock Board</h2>
      </header>
      <div className="dc-stock-rail__movement">
        {!hasSummary ? (
          <p className="dc-empty-note dc-stock-rail__empty">+0</p>
        ) : (
          <>
            {renderList("Risers", summary.risers, "rise")}
            {renderList("Fallers", summary.fallers, "fall")}
          </>
        )}
      </div>
      <div className="dc-stock-rail__leaders">
        <LeagueLeaders {...leadersProps} compact />
      </div>
    </aside>
  );
}

function IntelFeed({ boardMeta, prospects }) {
  const summary = useMemo(() => {
    const movers = buildStockMoversFromProspects(prospects);
    if (movers.risers.length || movers.fallers.length) return movers;
    const backend = boardMeta?.stock_market_summary || boardMeta?.stockMarketSummary;
    if (backend?.source === "backend") return backend;
    return movers;
  }, [prospects, boardMeta?.stock_market_summary, boardMeta?.stockMarketSummary]);
  const lines = [];
  if (summary?.risers?.length) {
    const top = summary.risers[0];
    const deltaRaw = top.delta_rank ?? top.deltaRank;
    const delta = Number.isFinite(Number(deltaRaw)) ? Number(deltaRaw) : 0;
    lines.push(`${top.name} ${delta >= 0 ? "↑" : "→"} ${Math.abs(delta)}`);
  }
  const cleanTop5 = prospects.find((p) => p.franchiseTier?.key === "franchise_swing" && getStockTone(p.draftStock) === "stable");
  if (cleanTop5) {
    lines.push(`${cleanTop5.lastName} stable`);
  }
  if (summary?.no_movement_count > 0) {
    lines.push(`${summary.no_movement_count} no history`);
  }
  // Honest standby: no movement to report yet, stated plainly on one line
  // instead of an empty broadcast band.
  const idle = !lines.length;
  if (idle) {
    lines.push("No rank movement reported yet");
  }
  return (
    <footer className={`dc-intel-feed${idle ? " is-idle" : ""}`}>
      <h3>Scouting Wire</h3>
      <ul>
        {lines.map((line, i) => <li key={i}>{line}</li>)}
      </ul>
    </footer>
  );
}

function LeagueLeadersModal({
  prospects,
  dateContext,
  leaderMode,
  setLeaderMode,
  profilesById,
  onClose,
  onSelectPlayer,
}) {
  const [metricSort, setMetricSort] = useState(null);
  const [sortViewMode, setSortViewMode] = useState("production");

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    setSortViewMode(resolveSortViewMode(leaderMode));
    setMetricSort((prev) => {
      if (prev?.mode === leaderMode) return prev;
      return {
        key: defaultSortForLeaderMode(leaderMode, prospects),
        dir: "desc",
        mode: leaderMode,
      };
    });
  }, [leaderMode, prospects]);

  const handleMetricSort = (metricKey) => {
    setMetricSort((prev) => {
      if (prev?.key === metricKey) {
        return { key: metricKey, dir: prev.dir === "desc" ? "asc" : "desc", mode: leaderMode };
      }
      return { key: metricKey, dir: "desc", mode: leaderMode };
    });
  };

  const leaders = useMemo(() => {
    const base = sortProspectsForLeaderMode(prospects, leaderMode).map((p) =>
      buildLeaderDisplayRow(p, profilesById)
    );
    if (metricSort?.key) {
      return sortLeaderRowsByMetric(base, metricSort.key, metricSort.dir);
    }
    return base;
  }, [prospects, leaderMode, profilesById, metricSort]);

  const sectionMeta = LEADER_MODE_META[leaderMode] || LEADER_MODE_META.points;
  const heroMetricKey = resolveHeroMetricKey(leaderMode, metricSort?.key);
  const sortLabel = metricSort?.key ? (LEADER_METRICS[metricSort.key]?.label || metricSort.key) : null;

  const handleRowClick = (row) => {
    if (row.player) onSelectPlayer(row.player);
  };

  return (
    <div className="dc-leaders-modal" role="dialog" aria-modal="true" aria-label="League Leaders">
      <button type="button" className="dc-profile-modal__backdrop" onClick={onClose} aria-label="Close" />
      <div className="dc-leaders-modal__panel dc-leaders-modal-sc">
        <ModalCloseButton onClick={onClose} label="Close leaders" />
        <header className="dc-leaders-modal__head">
          <div>
            <p className="dc-lm-eyebrow">League Leaders</p>
            <h2>{sectionMeta.title}</h2>
            {sortLabel ? (
              <p className="dc-lm-sort-label">
                Sorted by {sortLabel}
                {metricSort?.dir === "asc" ? " (low to high)" : ""}
              </p>
            ) : null}
            {dateContext?.statsThrough ? (
              <p className="dc-leaders-modal__context">Through {dateContext.statsThrough}</p>
            ) : null}
          </div>
        </header>

        <div className="dc-leader-tabs dc-leader-tabs--modal">
          {LEADER_MODE_OPTIONS.map((m) => (
            <button
              key={m.key}
              type="button"
              className={`dc-leader-tab ${leaderMode === m.key ? "is-active" : ""}`}
              onClick={() => setLeaderMode(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>

        {leaderMode !== "goalies" ? (
          <div className="dc-lm-view-toggle">
            {LEADER_SORT_VIEW_MODES.map((mode) => (
              <button
                key={mode.key}
                type="button"
                className={`dc-lm-view-toggle__btn${sortViewMode === mode.key ? " is-active" : ""}`}
                onClick={() => setSortViewMode(mode.key)}
              >
                {mode.label}
              </button>
            ))}
          </div>
        ) : null}

        <LeadersSortBar
          leaderMode={leaderMode}
          sortViewMode={sortViewMode}
          activeSortKey={metricSort?.key}
          onSort={handleMetricSort}
        />

        <div className="dc-leaders-modal__body">
          <section className="dc-lm-section">
            <div className="dc-lm-section__body dc-scroll-surface dc-leaders-modal__scroll">
              {!leaders.length ? (
                <p className="dc-empty-note">Not enough tracked data for this view yet.</p>
              ) : (
                <div className="dc-lm-row-stack">
                  {leaders.map((row, index) => (
                    <LeaderModalRow
                      key={row.id}
                      row={row}
                      index={index}
                      leaderMode={leaderMode}
                      onSelect={handleRowClick}
                      heroMetricKey={heroMetricKey}
                      activeSortMetric={metricSort?.key}
                      onMetricSort={handleMetricSort}
                      sortViewMode={sortViewMode}
                    />
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function LeagueLeaders({
  prospects,
  dateContext,
  leaderMode,
  setLeaderMode,
  onSelectPlayer,
  onOpenFullLeaders,
  regionFilter,
  compact = false,
}) {
  const filtered = useMemo(() => {
    let list = prospects;
    if (regionFilter && regionFilter !== "ALL PLAYERS") list = list.filter((p) => p.region === regionFilter);
    return list;
  }, [prospects, regionFilter]);

  const previewModes = LEADER_MODE_OPTIONS.filter((m) => m.key !== "defense" && m.key !== "analytics");

  const leaders = useMemo(() => {
    return sortProspectsForLeaderMode(filtered, leaderMode).slice(0, 8);
  }, [filtered, leaderMode]);

  return (
    <div className={`dc-leaders-panel${compact ? " dc-leaders-panel--compact" : ""}`}>
      <div className="dc-side-title">
        <h2>LEAGUE LEADERS</h2>
        {!compact && dateContext?.statsThrough ? (
          <span className="dc-leaders-date">through {dateContext.statsThrough}</span>
        ) : null}
      </div>
      <div className="dc-leader-tabs">
        {previewModes.map((m) => (
          <button key={m.key} type="button" className={`dc-leader-tab ${leaderMode === m.key ? "is-active" : ""}`} onClick={() => setLeaderMode(m.key)}>
            {m.label}
          </button>
        ))}
      </div>
      <div className="dc-leader-scroll">
        {!leaders.length ? (
          <p className="dc-empty-note">Not enough tracked data for this view yet.</p>
        ) : (
          leaders.map((p, index) => (
            <button type="button" key={p.id} className="dc-leader-row" onClick={() => onSelectPlayer(p)}>
              <span className="dc-leader-row__name">{index + 1}. {p.firstName[0]}. {p.lastName}</span>
              <span className="dc-leader-meta">{p.league}</span>
              {leaderMode === "goalies" ? (
                <strong>{p.wins}W · {p.savePct || "—"}</strong>
              ) : leaderMode === "goals" ? (
                <strong>{p.goals}G · {p.gp} GP</strong>
              ) : leaderMode === "assists" ? (
                <strong>{p.assists}A · {p.gp} GP</strong>
              ) : leaderMode === "ppg" ? (
                <strong>{leaderPpg(p) != null ? `${Number(leaderPpg(p)).toFixed(2)} PPG` : "—"}</strong>
              ) : leaderMode === "stock" ? (
                <strong>{stockText(Number(p.draftStock?.deltaRank ?? p.stock) || 0)}</strong>
              ) : (
                <strong>{p.points} PTS · {p.gp} GP</strong>
              )}
            </button>
          ))
        )}
      </div>
      <button type="button" className="dc-view-full dc-view-full--leaders" onClick={onOpenFullLeaders}>
        View Full Leaders
      </button>
    </div>
  );
}

function ScoutingPanel({ player, meta, onAssignScout, showAssign, setShowAssign }) {
  if (!player) return null;
  const conf = confidenceLabel(player.completion);
  return (
    <div className="dc-scouting-panel">
      <h3>SCOUTING DOSSIER</h3>
      <div className="dc-scout-grid">
        <div><span>Completion</span><strong>{player.completion}%</strong></div>
        <div><span>Confidence</span><strong>{conf}</strong></div>
        <div><span>Assigned Scout</span><strong>{meta.assignedScout || "Unassigned"}</strong></div>
        <div><span>Last Viewed</span><strong>{meta.lastViewed || "—"}</strong></div>
        <div><span>Next Report</span><strong>{nextReportDue(player.completion)}</strong></div>
      </div>
      {!meta.assignedScout && (
        <p className="dc-scout-prompt">Assign a regional scout to begin detailed reports.</p>
      )}
      {showAssign && (
        <div className="dc-scout-assign">
          {SCOUT_NAMES.map((name) => (
            <button key={name} type="button" className={meta.assignedScout === name ? "is-active" : ""} onClick={() => { onAssignScout(name); setShowAssign(false); }}>
              {name}
            </button>
          ))}
        </div>
      )}
      <div className="dc-report-list">
        <h4>Report Status</h4>
        {REPORT_TYPES.map((rt) => {
          const status = meta.requestedReports[rt.key];
          const autoComplete = rt.key !== "character" && player.completion >= 78;
          const charComplete = rt.key === "character" && (status === "complete" || player.completion >= 88);
          const label = charComplete || (autoComplete && status !== "pending") ? "Complete" : status === "pending" ? "Pending" : status === "requested" ? "Requested" : "Not Requested";
          return (
            <div key={rt.key} className={`dc-report-row dc-report-row--${label.toLowerCase().replace(/\s/g, "-")}`}>
              <span>{rt.label}</span>
              <em>{label}</em>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TeamNeedsPanel({ needs, strategy, player }) {
  return (
    <div className="dc-needs-panel">
      <h3>ORGANIZATIONAL CONTEXT</h3>
      {needs?.length ? (
        <p><span>Team Needs:</span> {needs.join(", ")}</p>
      ) : (
        <p className="dc-empty-note">Team need context unavailable.</p>
      )}
      <p><span>Draft Strategy:</span> {strategy || "BPA"}</p>
      {player && (
        <p className="dc-fit-note">
          <span>{fullName(player)} fit:</span> {player.fit >= 72 ? "Strong organizational fit" : player.fit >= 58 ? "Moderate fit — addresses depth" : "Peripheral fit at current rank"}
        </p>
      )}
    </div>
  );
}

function ScoutingPriorities({ onFilterAction }) {
  const items = [
    { label: "Review top 32", action: "top32" },
    { label: "Assign scouts to low-confidence targets", action: "lowconf" },
    { label: "Scout players near projected pick", action: "midround" },
    { label: "Check league leaders", action: "leaders" },
    { label: "Revisit risers/fallers", action: "movers" },
  ];
  return (
    <div className="dc-priorities-panel">
      <h3>SCOUTING PRIORITIES</h3>
      <ul>
        {items.map((item) => (
          <li key={item.action}>
            <button type="button" onClick={() => onFilterAction(item.action)}>{item.label}</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ComparisonTray({ compareIds, prospects, onRemove, onClear }) {
  if (!compareIds.length) return null;
  const players = compareIds.map((id) => prospects.find((p) => p.id === id)).filter(Boolean);
  return (
    <div className="dc-compare-tray">
      <div className="dc-compare-head">
        <strong>Compare ({players.length}/3)</strong>
        <button type="button" onClick={onClear}>Clear</button>
      </div>
      <div className="dc-compare-grid">
        {players.map((p) => (
          <div key={p.id} className="dc-compare-card">
            <button type="button" className="dc-compare-remove" onClick={() => onRemove(p.id)}>×</button>
            <b>{p.firstName} {p.lastName}</b>
            <span>#{p.rank} · {p.position}</span>
            <span>{p.projection} · {p.talent}</span>
            <span>Scout {p.completion}%</span>
            <span>{p.position !== "G" ? `${p.points} PTS / ${p.gp} GP` : `${p.wins}W · ${p.savePct}`}</span>
            <span>Fit {p.fit}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoutingActions({ meta, onToggleWatchlist, onToggleTarget, onToggleDND, onAssignToggle, onRequestReport, onCompare, compareFull, showAssign }) {
  const [scoutOpen, setScoutOpen] = useState(false);
  return (
    <div className="dc-scout-actions">
      <button type="button" className={`dc-btn dc-btn--primary ${scoutOpen ? "is-active" : ""}`} onClick={() => setScoutOpen((v) => !v)}>
        ◎ Scout
      </button>
      {scoutOpen ? (
        <div className="dc-scout-menu">
          <button type="button" className="dc-btn dc-btn--sub" onClick={onAssignToggle}>{showAssign ? "Close" : "Assign"}</button>
          <button type="button" className="dc-btn dc-btn--sub" onClick={() => onRequestReport("potential")}>Potential</button>
          <button type="button" className="dc-btn dc-btn--sub" onClick={() => onRequestReport("skills")}>Skills</button>
          <button type="button" className="dc-btn dc-btn--sub" onClick={() => onRequestReport("character")}>Character</button>
        </div>
      ) : null}
      <button type="button" className={`dc-btn dc-btn--secondary ${meta.watchlist ? "is-active" : ""}`} onClick={onToggleWatchlist}>
        ★ Watch
      </button>
      <button type="button" className={`dc-btn dc-btn--secondary ${meta.target ? "is-active" : ""}`} onClick={onToggleTarget}>
        ◉ Pin
      </button>
      <button type="button" className={`dc-btn dc-btn--secondary ${compareFull ? "is-disabled" : ""}`} onClick={onCompare} disabled={compareFull}>
        ⚖ Compare
      </button>
      <button type="button" className={`dc-btn dc-btn--danger ${meta.doNotDraft ? "is-active" : ""}`} onClick={onToggleDND}>
        ⚠ DND
      </button>
    </div>
  );
}

function RightSidePanel(props) {
  const [sideTab, setSideTab] = useState("leaders");
  return (
    <aside className="dc-right-panel">
      <div className="dc-side-tabs">
        {["leaders", "scouting", "context", "priorities"].map((t) => (
          <button key={t} type="button" className={sideTab === t ? "is-active" : ""} onClick={() => setSideTab(t)}>
            {t === "leaders" ? "Leaders" : t === "scouting" ? "Scouting" : t === "context" ? "Team" : "Tasks"}
          </button>
        ))}
      </div>
      {sideTab === "leaders" && <LeagueLeaders {...props.leadersProps} />}
      {sideTab === "scouting" && <ScoutingPanel {...props.scoutingProps} />}
      {sideTab === "context" && <TeamNeedsPanel {...props.contextProps} />}
      {sideTab === "priorities" && <ScoutingPriorities onFilterAction={props.onFilterAction} />}
    </aside>
  );
}

function OverviewTab({ player, meta, dateContext, draftYear }) {
  return (
    <div className="dc-profile-body">
      <div className="dc-profile-left">
        <DraftClassHeadshot player={player} size="lg" />
        <div className="dc-profile-name">
          <span>{player.firstName}</span>
          <strong>{player.lastName}</strong>
          <p>{player.position} · {player.playerType}</p>
          <small>{countryFlag(player.country)} {player.country}</small>
          <small>{player.team} ({player.league})</small>
          <small>{player.height} · {player.weight} lbs · Shoots {player.handedness} · Age {player.age}</small>
          <ProspectBadges player={player} meta={meta} />
        </div>
      </div>

      <div className="dc-profile-card dc-draft-projection">
        <span>DRAFT PROJECTION</span>
        <strong>{player.projection}</strong>
        <p>{confidenceLabel(player.completion)}</p>
        <small>Scout: {meta.assignedScout || "Unassigned"}</small>
      </div>

      <div className="dc-info-card">
        <h3>PLAYER INFO</h3>
        <div className="dc-info-grid">
          <span>Birthdate</span><b>{player.birthday}</b>
          <span>Hometown</span><b>{player.birthCity || player.country}</b>
          <span>Draft Eligible</span><b>{draftYear}</b>
          <span>Overall Rank</span><b>{player.rank}</b>
          <span>Position Rank</span><b>{player.position}-{player.positionRank}</b>
          <span>Partial Stats</span><b>{player.gp} GP through {dateContext.statsThrough || "season"}</b>
          <span>Height</span><b>{player.height}</b>
          <span>Weight</span><b>{player.weight} lbs</b>
        </div>
      </div>

      <div className="dc-list-card dc-list-card--good">
        <h3>STRENGTHS</h3>
        <ul>{strengthList(player).map((s) => <li key={s}>{s}</li>)}</ul>
      </div>

      <div className="dc-list-card dc-list-card--bad">
        <h3>WEAKNESSES</h3>
        <ul>{weaknessList(player).map((s) => <li key={s}>{s}</li>)}</ul>
      </div>

      <div className="dc-summary-card">
        <h3>SCOUT SUMMARY</h3>
        <p>{scoutSummary(player)}</p>
        {dateContext.isPartialSeason && <small className="dc-partial-note">Partial-season view — reports still developing.</small>}
      </div>
    </div>
  );
}

function StatsTab({ player, dateContext, allProspects }) {
  const isGoalie = player.position === "G";
  const ppg = player.gp > 0 ? (player.points / player.gp).toFixed(2) : "—";
  const hasProjection = player.projectedGp != null || player.projectedPoints != null;
  const leagueRank = useMemo(() => {
    const peers = allProspects.filter((p) => p.league === player.league && p.position !== "G");
    if (!peers.length || isGoalie) return null;
    const sorted = [...peers].sort((a, b) => b.points - a.points);
    const idx = sorted.findIndex((p) => p.id === player.id);
    return idx >= 0 ? idx + 1 : null;
  }, [allProspects, player, isGoalie]);

  return (
    <div className="dc-stat-layout">
      <div className="dc-stat-card">
        <h3>CURRENT SEASON STATS</h3>
        <p className="dc-stats-context">Actual stats through {dateContext.statsThrough || "current date"}</p>
        {player.hasNoGames ? (
          <p className="dc-sample-warn">No games yet — stats will populate as the prospect league season begins.</p>
        ) : null}
        {!isGoalie ? (
          <div className="dc-big-stat-grid">
            <div><span>GP</span><strong>{player.gp}</strong></div>
            <div><span>G</span><strong>{player.goals}</strong></div>
            <div><span>A</span><strong>{player.assists}</strong></div>
            <div><span>PTS</span><strong>{player.points}</strong></div>
            <div><span>P/GP</span><strong>{ppg}</strong></div>
            <div><span>STOCK</span><strong>{stockBadgeText(player.draftStock)}</strong></div>
          </div>
        ) : (
          <div className="dc-big-stat-grid">
            <div><span>GP</span><strong>{player.gp}</strong></div>
            <div><span>W</span><strong>{player.wins}</strong></div>
            <div><span>SV%</span><strong>{player.savePct || "—"}</strong></div>
            <div><span>GAA</span><strong>{player.gaa || "—"}</strong></div>
            <div><span>STOCK</span><strong>{stockBadgeText(player.draftStock)}</strong></div>
            <div><span>FIT</span><strong>{player.fit}</strong></div>
          </div>
        )}
        {player.gp > 0 && player.gp < 12 && <p className="dc-sample-warn">Small sample warning — stats may not be representative yet.</p>}
        {player.recentForm?.last_5_gp > 0 ? (
          <p className="dc-stats-context">
            Recent form: {player.recentForm.last_5_points} PTS in last {player.recentForm.last_5_gp} GP
          </p>
        ) : null}
      </div>

      <div className="dc-stat-card">
        <h3>SEASON PROJECTION</h3>
        <p className="dc-stats-context">
          {hasProjection ? "Full-season pace based on talent, role, and league environment" : "Based on preseason scouting"}
        </p>
        {!isGoalie ? (
          <div className="dc-big-stat-grid">
            <div><span>GP PACE</span><strong>{player.projectedGp ?? "—"}</strong></div>
            <div><span>G PACE</span><strong>{player.projectedGoals ?? "—"}</strong></div>
            <div><span>A PACE</span><strong>{player.projectedAssists ?? "—"}</strong></div>
            <div><span>PTS PACE</span><strong>{player.projectedPoints ?? "—"}</strong></div>
            <div><span>P/GP PACE</span><strong>{player.projectedPpg != null ? Number(player.projectedPpg).toFixed(2) : "—"}</strong></div>
            <div><span>CONF</span><strong>{player.scoutingConfidence != null ? `${player.scoutingConfidence}%` : `${player.completion}%`}</strong></div>
          </div>
        ) : (
          <div className="dc-big-stat-grid">
            <div><span>GP PACE</span><strong>{player.projectedGp ?? "—"}</strong></div>
            <div><span>W PACE</span><strong>{player.projectedWins ?? "—"}</strong></div>
            <div><span>SV% PACE</span><strong>{player.projectedSavePct ?? "—"}</strong></div>
            <div><span>GAA PACE</span><strong>{player.projectedGaa ?? "—"}</strong></div>
          </div>
        )}
      </div>

      <div className="dc-stat-card">
        <h3>LEAGUE CONTEXT</h3>
        {(player.productionContext || player.translationRisk || player.scoringEnvironment) ? (
          <div className="dc-scoring-tags" style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
            {player.productionContext ? <span className="dc-tag">{player.productionContext}</span> : null}
            {player.scoringEnvironment ? <span className="dc-tag">{player.scoringEnvironment}</span> : null}
            {player.leagueDifficulty ? <span className="dc-tag">{player.leagueDifficulty} league</span> : null}
            {player.translationRisk ? <span className="dc-tag dc-tag--warn">Translation risk: {player.translationRisk}</span> : null}
          </div>
        ) : null}
        <p>
          {leagueRank
            ? `Ranks ${leagueRank}${leagueRank === 1 ? "st" : leagueRank === 2 ? "nd" : leagueRank === 3 ? "rd" : "th"} in ${player.league} scoring among tracked draft-eligible skaters.`
            : `Tracked in ${player.league} with ${player.completion}% scouting confidence.`}
          {" "}Production will shift as the season progresses.
        </p>
        {player.productionAdjustedScore != null && !isGoalie ? (
          <p className="dc-stats-context">Adjusted production score: {Number(player.productionAdjustedScore).toFixed(2)} (league + age context)</p>
        ) : null}
      </div>

      <div className="dc-stat-card">
        <h3>DEVELOPMENT ETA</h3>
        <div className="dc-eta">
          {(() => {
            const etaLabel = formatNhlEta(player.nhlEta, null);
            const fallback = player.rank <= 8 ? "1-2 YEARS" : player.rank <= 32 ? "2-3 YEARS" : "3-5 YEARS";
            return (
              <>
                <strong>{etaLabel || fallback}</strong>
                <span>
                  {etaLabel
                    ? `${etaLabel} to NHL readiness based on central scouting.`
                    : player.rank <= 8
                    ? "Could challenge for NHL minutes quickly."
                    : player.rank <= 32
                    ? "Likely needs one or two years of development."
                    : "Longer runway with higher variance."}
                </span>
              </>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

function AttributeBar({ label, value, completion, seed }) {
  const display = attributeDisplay(value, completion, seed);
  return (
    <div className={`dc-attribute ${display.locked ? "is-locked" : ""}`}>
      <div>
        <span>{label}</span>
        <b>{display.text}</b>
        <em>{display.confidence}</em>
      </div>
      <div className="dc-attribute__track">
        <div style={{ width: `${display.width}%`, opacity: display.locked ? 0.15 : 1 }} />
      </div>
    </div>
  );
}

function AttributesTab({ player }) {
  const attrs = [
    ["Skating", player.skating, 1],
    ["Shooting", player.shooting, 2],
    ["Passing", player.passing, 3],
    ["Defense", player.defense, 4],
    ["Physical", player.physical, 5],
    ["Hockey IQ", player.hockeyIQ, 6],
  ];
  return (
    <div className="dc-attributes-layout">
      <div className="dc-attribute-card">
        <h3>PLAYER ATTRIBUTES</h3>
        <p className="dc-fog-note">Attributes shown as ranges until scouting completion is high ({confidenceLabel(player.completion)}).</p>
        {attrs.map(([label, val, seed]) => (
          <AttributeBar key={label} label={label} value={val} completion={player.completion} seed={seed} />
        ))}
      </div>

      <div className="dc-attribute-card">
        <h3>SCOUTING GRADES</h3>
        <div className="dc-grade-grid">
          {attrs.map(([label, val, seed]) => {
            const display = attributeDisplay(val, player.completion, seed);
            const grade = display.locked ? "?" : gradeFromValue(display.width);
            return (
              <div key={label}><span>{label}</span><strong>{grade}</strong></div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ScoutReportTab({ player, meta, profile }) {
  const p = profile || player?.profile || null;
  const sections = scoutReportSections(player, meta, p);
  const teamFitScore = p?.team_fit?.score ?? p?.teamFit?.score ?? player.fit;
  const teamFitLabel = p?.team_fit?.label ?? p?.teamFit?.label ?? null;
  return (
    <div className="dc-scout-layout">
      <div className="dc-scout-card">
        <h3>CURRENT PROJECTION</h3>
        <p>{sections.projection}</p>
        <h4>Upside</h4>
        <p>{sections.upside}</p>
        <h4>Scout Notes</h4>
        <p>{sections.notes}</p>
      </div>
      <div className="dc-scout-card">
        <h3>RISK</h3>
        <p>{sections.risk}</p>
        <h4>Development Timeline</h4>
        <p>{sections.timeline}</p>
        <h4>Comparable Role</h4>
        <p>{sections.comparable}</p>
      </div>
      <div className="dc-scout-card">
        <h3>WHAT TO SCOUT NEXT</h3>
        <p>{sections.nextScout}</p>
        <ul>
          <li>Scouting completion: {player.completion}%</li>
          <li>Confidence: {confidenceLabel(player.completion)}</li>
          <li>Draft movement: {stockText(player.stock)}{player.stockReason ? ` — ${player.stockReason}` : ""}</li>
          <li>Team fit score: {teamFitScore != null ? Math.round(Number(teamFitScore)) : "—"}{teamFitLabel ? ` (${teamFitLabel})` : ""}</li>
        </ul>
      </div>
    </div>
  );
}

function CharacterTab({ player, meta, profile }) {
  const p = profile || player?.profile || null;
  const teamFitScore = p?.team_fit?.score ?? p?.teamFit?.score ?? player.fit;
  const charAvailable = meta.requestedReports?.character === "complete" || meta.requestedReports?.character === "requested" || player.completion >= 82;
  if (!charAvailable) {
    return (
      <div className="dc-character-layout">
        <div className="dc-character-card dc-character-card--locked">
          <h3>CHARACTER REPORT</h3>
          <p>Character report not complete. Request a character/interview report or raise scouting completion.</p>
        </div>
      </div>
    );
  }

  const rows = [
    ["Competitiveness", "High compete level. Wants to be the difference.", player.compete],
    ["Leadership", "Teammates respond well to his habits.", player.leadership],
    ["Work Ethic", "Consistently looks to improve.", player.workEthic],
    ["Coachability", "Takes feedback and applies it.", player.coachability],
    ["Consistency", "Performance stability over long sample.", player.consistency],
    ["Poise", "Calm under pressure. Rarely rattled.", player.poise],
  ];

  return (
    <div className="dc-character-layout">
      <div className="dc-character-card">
        <h3>PERSONALITY & CHARACTER</h3>
        {rows.map(([label, note, value]) => (
          <div className="dc-character-row" key={label}>
            <span>{label}</span>
            <p>{player.completion >= 70 ? note : "Preliminary observation — needs follow-up."}</p>
            <strong>{player.completion >= 70 ? gradeFromValue(value) : "?"}</strong>
          </div>
        ))}
      </div>

      <div className="dc-character-card">
        <h3>MORALE & FIT</h3>
        <div className="dc-fit-row"><span>Morale</span><strong>{ratingLabel(player.morale)}</strong></div>
        <div className="dc-fit-row"><span>Character</span><strong>{ratingLabel(player.character)}</strong></div>
        <div className="dc-fit-row"><span>Willingness To Join Org</span><strong>{ratingLabel(player.fit)}</strong></div>
        <div className="dc-fit-row"><span>Team Need Fit</span><strong>{teamFitScore != null ? ratingLabel(teamFitScore) : "—"}</strong></div>
        <div className="dc-fit-row"><span>Potential Impact</span><strong>{player.rank <= 5 ? "Franchise" : player.rank <= 32 ? "Core" : "Depth"}</strong></div>
      </div>
    </div>
  );
}

function PlayerProfile({ player, meta, dateContext, draftYear, allProspects, scoutingActions, compareIds, onRemoveCompare, onClearCompare }) {
  const [tab, setTab] = useState("OVERVIEW");
  if (!player) {
    return <section className="dc-profile dc-profile--empty"><p>Select a prospect to view scouting dossier.</p></section>;
  }

  return (
    <section className="dc-profile">
      <div className="dc-profile-header">
        <h2>PLAYER PROFILE — {player.firstName} {player.lastName}</h2>
        <div className="dc-profile-tabs">
          {PROFILE_TABS.map((t) => (
            <button key={t} type="button" className={`dc-profile-tab ${tab === t ? "is-active" : ""}`} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {scoutingActions}

      <ComparisonTray compareIds={compareIds} prospects={allProspects} onRemove={onRemoveCompare} onClear={onClearCompare} />

      {tab === "OVERVIEW" && <OverviewTab player={player} meta={meta} dateContext={dateContext} draftYear={draftYear} />}
      {tab === "STATS" && <StatsTab player={player} dateContext={dateContext} allProspects={allProspects} />}
      {tab === "ATTRIBUTES" && <AttributesTab player={player} />}
      {tab === "SCOUT REPORT" && <ScoutReportTab player={player} meta={meta} profile={player.profile} />}
      {tab === "CHARACTER" && <CharacterTab player={player} meta={meta} profile={player.profile} />}
    </section>
  );
}

function BottomLegend({ onBack }) {
  return (
    <footer className="dc-bottom-legend">
      <span>Click row to select</span>
      <button type="button" className="dc-legend-back" onClick={onBack}>← Back to Office</button>
      <span>Esc to exit</span>
    </footer>
  );
}

export default function DraftClass() {
  const {
    franchiseState,
    setScreen,
    loading,
    openWorldJuniors,
    pendingDraftProspectId,
    setPendingDraftProspectId,
  } = useGameUI();
  const dateContext = useMemo(() => buildDateContext(franchiseState), [franchiseState]);
  const [activeBoardView, setActiveBoardView] = useState("rank");
  const [leaderMode, setLeaderMode] = useState("points");
  const [leadersModalOpen, setLeadersModalOpen] = useState(false);
  const [scoutingStore, setScoutingStore] = useState({});
  const [compareIds, setCompareIds] = useState([]);
  const [selectedProspect, setSelectedProspect] = useState(null);

  const patchProspectMeta = useCallback(async (prospectId, localPatch, apiPatch) => {
    if (!prospectId) return;
    setScoutingStore((prev) => ({
      ...prev,
      [prospectId]: { ...getScoutingMeta(prev, prospectId), ...localPatch },
    }));
    try {
      await patchScoutingMeta(prospectId, apiPatch || localPatch);
    } catch {
      // Keep optimistic UI; board refresh will reconcile later.
    }
  }, []);

  const toggleProspectWatchlist = useCallback((prospectId) => {
    const current = getScoutingMeta(scoutingStore, prospectId);
    const next = !current.watchlist;
    patchProspectMeta(prospectId, { watchlist: next }, { watchlist: next });
  }, [scoutingStore, patchProspectMeta]);

  const toggleProspectTarget = useCallback((prospectId) => {
    const current = getScoutingMeta(scoutingStore, prospectId);
    const next = !current.target;
    patchProspectMeta(prospectId, { target: next }, { target: next });
  }, [scoutingStore, patchProspectMeta]);

  const toggleProspectDnd = useCallback((prospectId) => {
    const current = getScoutingMeta(scoutingStore, prospectId);
    const next = !current.doNotDraft;
    patchProspectMeta(prospectId, { doNotDraft: next }, { do_not_draft: next });
  }, [scoutingStore, patchProspectMeta]);

  const assignProspectScout = useCallback((prospectId, scoutName) => {
    patchProspectMeta(prospectId, { assignedScout: scoutName }, { assigned_scout: scoutName });
  }, [patchProspectMeta]);

  const toggleProspectCompare = useCallback((prospectId) => {
    setCompareIds((prev) => {
      if (prev.includes(prospectId)) return prev.filter((id) => id !== prospectId);
      if (prev.length >= 3) return prev;
      return [...prev, prospectId];
    });
  }, []);

  const boardMeta = franchiseState?.draft_class_rankings || {};
  const usingFallback = !boardMeta?.entries?.length;
  const teamName = franchiseState?.team?.name || "";
  const strategy = String(franchiseState?.team?.strategy || "BPA").toUpperCase();
  const teamNeeds = useMemo(() => inferTeamNeeds(franchiseState), [franchiseState]);

  const rawProspects = useMemo(() => {
    const profiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
    const mapped = mapBackendDraftBoard(franchiseState?.draft_class_rankings?.entries, dateContext);
    return mapped.map((p) => ({ ...p, profile: profiles[p.id] || null }));
  }, [franchiseState?.draft_class_rankings?.entries, franchiseState?.draft_class_hud?.prospect_profiles_by_id, dateContext]);

  useEffect(() => {
    // Prefer local board + franchise scouting overlays. Hitting /scouting/prospects
    // on every Draft Class open rebuilt the same board again (~30s) in parallel
    // with state/heavy — keep that endpoint for the Scouting screen itself.
    const fromState = scoutingStoreFromFranchiseState(franchiseState);
    if (Object.keys(fromState).length) {
      setScoutingStore((prev) => mergeScoutingStores(fromState, prev));
    }
  }, [franchiseState?.session_id, franchiseState?.scouting_state, franchiseState?.draft_class_rankings?.entries]);

  const franchiseStateForHeader = franchiseState;

  useEffect(() => {
    const fromBoard = {};
    rawProspects.forEach((player) => {
      if (!player?.id) return;
      if (player.watchlist || player.target || player.doNotDraft || player.assignedScout) {
        fromBoard[player.id] = scoutingMetaFromProspect(player);
      }
    });
    if (!Object.keys(fromBoard).length) return;
    setScoutingStore((prev) => mergeScoutingStores(fromBoard, prev));
  }, [rawProspects]);

  const filteredProspects = useMemo(() => {
    let list = [...rawProspects];

    if (activeBoardView === "forwards") {
      list = list.filter((p) => isForwardPosition(p.position));
    } else if (activeBoardView === "defensemen") {
      list = list.filter((p) => isDefensemanPosition(p.position));
    } else if (activeBoardView === "goalies") {
      list = list.filter((p) => isGoaliePosition(p.position));
    }

    list.sort((a, b) => a.rank - b.rank);

    return list;
  }, [rawProspects, activeBoardView]);

  const selectedProspectId = selectedProspect?.id ?? null;

  useEffect(() => {
    if (!filteredProspects.length) {
      setSelectedProspect(null);
      return;
    }
    if (selectedProspectId && !filteredProspects.some((p) => p.id === selectedProspectId)) {
      setSelectedProspect(null);
    }
  }, [filteredProspects, selectedProspectId]);

  const openProspect = useCallback((player) => {
    if (!player) return;
    const transcendent = Boolean(player.isTranscendent || player?.profile?.is_transcendent);
    if (transcendent) {
      playTranscendentBossSting();
    }
    setSelectedProspect(player);
  }, []);

  useEffect(() => {
    if (!selectedProspect?.id) return;
    const stamp = dateContext.statsThrough || "Today";
    setScoutingStore((prev) => ({
      ...prev,
      [selectedProspect.id]: { ...getScoutingMeta(prev, selectedProspect.id), lastViewed: stamp },
    }));
  }, [selectedProspect?.id, dateContext.statsThrough]);

  const activeProfile = useMemo(() => {
    if (!selectedProspect) return null;
    const profiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
    return selectedProspect.profile || profiles[selectedProspect.id] || null;
  }, [selectedProspect, franchiseState?.draft_class_hud?.prospect_profiles_by_id]);

  useEffect(() => {
    if (!pendingDraftProspectId || !filteredProspects.length) return;
    const match = filteredProspects.find((p) => String(p.id) === String(pendingDraftProspectId));
    if (match) {
      setSelectedProspect(match);
    }
    setPendingDraftProspectId(null);
  }, [pendingDraftProspectId, filteredProspects, setPendingDraftProspectId]);

  const handleOpenWjc = useCallback(() => {
    if (typeof openWorldJuniors === "function") openWorldJuniors();
  }, [openWorldJuniors]);

  const handleBack = useCallback(() => setScreen(SCREENS.HUB), [setScreen]);

  useEffect(() => {
    function onKeyDown(e) {
      if (e.target?.tagName === "INPUT" || e.target?.tagName === "SELECT") return;
      if (e.key === "Escape") {
        if (leadersModalOpen) {
          setLeadersModalOpen(false);
          return;
        }
        if (selectedProspect) {
          setSelectedProspect(null);
          return;
        }
        handleBack();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleBack, selectedProspect, leadersModalOpen]);

  if (loading && !franchiseState) {
    return (
      <div className="game-root">
        <div className="game-canvas">
          <main className="dc-screen dc-screen--loading"><p>Loading draft class…</p></main>
        </div>
      </div>
    );
  }

  return (
    <div className="game-root">
      <div className="game-canvas">
        <main className="dc-root dc-screen register-ops" data-register="ops">
          <TopHeader
            onBack={handleBack}
            dateContext={dateContext}
            franchiseState={franchiseStateForHeader}
          />

          <CommandStatStrip
            franchiseState={franchiseStateForHeader || franchiseState}
            onOpenWjc={handleOpenWjc}
          />

          <div className="dc-command-grid">
            <aside className="dc-board-nav-rail">
              <DraftBoardNavRail
                activeBoardView={activeBoardView}
                setActiveBoardView={setActiveBoardView}
              />
            </aside>
            <ProspectBoardPanel
              prospects={filteredProspects}
              selectedProspectId={selectedProspectId}
              onOpenProspect={openProspect}
              scoutingStore={scoutingStore}
              activeBoardView={activeBoardView}
            />

            <StockExchangeRail
              boardMeta={boardMeta}
              prospects={filteredProspects}
              onSelectPlayer={(id) => {
                const p = filteredProspects.find((x) => x.id === id);
                if (p) openProspect(p);
              }}
              leadersProps={{
                prospects: filteredProspects,
                dateContext,
                leaderMode,
                setLeaderMode,
                onSelectPlayer: openProspect,
                onOpenFullLeaders: () => setLeadersModalOpen(true),
                regionFilter: "ALL PLAYERS",
                leagueFilter: "ALL",
              }}
            />
          </div>

          <IntelFeed boardMeta={boardMeta} prospects={filteredProspects} />
        </main>

        {selectedProspect ? (
          <ProspectProfileModal
            player={selectedProspect}
            profile={activeProfile}
            meta={getScoutingMeta(scoutingStore, selectedProspect.id)}
            compareIds={compareIds}
            onClose={() => setSelectedProspect(null)}
            onToggleWatchlist={() => toggleProspectWatchlist(selectedProspect.id)}
            onToggleTarget={() => toggleProspectTarget(selectedProspect.id)}
            onToggleDND={() => toggleProspectDnd(selectedProspect.id)}
            onToggleCompare={() => toggleProspectCompare(selectedProspect.id)}
            onAssignScout={(name) => assignProspectScout(selectedProspect.id, name)}
          />
        ) : null}

        {leadersModalOpen ? (
          <LeagueLeadersModal
            prospects={filteredProspects}
            dateContext={dateContext}
            leaderMode={leaderMode}
            setLeaderMode={setLeaderMode}
            profilesById={franchiseState?.draft_class_hud?.prospect_profiles_by_id || {}}
            onClose={() => setLeadersModalOpen(false)}
            onSelectPlayer={(player) => {
              setLeadersModalOpen(false);
              openProspect(player);
            }}
          />
        ) : null}

        <style>{`
          .dc-root.dc-screen {
            --dc-font-title: var(--font-broadcast-display);
            --dc-font-number: var(--font-mono-data);
            --dc-font-ui: var(--font-ops-ui);
            --dc-font-mono: var(--font-mono-data);
            --dc-text: var(--ops-text);
            --dc-muted: var(--ops-text-secondary);
            --dc-line: var(--ops-grid);
            --dc-cyan: var(--ops-cyan);
            --dc-green: var(--ops-success);
            --dc-red: var(--ops-injury);
            --dc-gold: var(--ops-gold);
            --dc-panel: var(--ops-panel);
            --dc-panel-soft: rgba(6, 21, 34, 0.72);

            /* Ceiling ramp: cold slate to hot gold, lightness climbing with the tier so
               the column reads as a gradient even without hue perception. */
            --dc-pot-fringe: #74849a;
            --dc-pot-depth: #9db0c6;
            --dc-pot-moderate: #8ab4ff;
            --dc-pot-strong: #38bdf8;
            --dc-pot-high: #13d8e7;
            --dc-pot-elite: #e9a83c;
            --dc-pot-generational: #ffc94d;

            height: 100%;
            width: 100%;
            display: grid;
            /* Four children: header, stat strip, board, wire. The board owns the
               flexible track — a fifth declared track handed it to the wire. */
            grid-template-rows: auto auto minmax(0, 1fr) auto;
            gap: 10px;
            padding: 12px;
            color: var(--dc-text);
            font-family: var(--dc-font-ui);
            background:
              radial-gradient(circle at 50% 42%, rgba(53, 185, 255, 0.1), transparent 28%),
              radial-gradient(circle at 10% 0%, rgba(43, 228, 255, 0.09), transparent 25%),
              radial-gradient(circle at 90% 8%, rgba(244, 198, 110, 0.07), transparent 20%),
              linear-gradient(180deg, #04101a, #02070d);
            overflow: hidden;
          }

          .dc-screen--loading { display: grid; place-items: center; }
          .dc-scroll-surface { overflow: auto; scrollbar-width: thin; scrollbar-color: rgba(43, 228, 255, 0.55) rgba(8, 21, 34, 0.55); }
          .dc-scroll-surface::-webkit-scrollbar { width: 10px; height: 10px; }
          .dc-scroll-surface::-webkit-scrollbar-track { background: rgba(8, 21, 34, 0.55); border-radius: 999px; }
          .dc-scroll-surface::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(43, 228, 255, 0.75), rgba(43, 228, 255, 0.38));
            border-radius: 999px;
            border: 2px solid rgba(8, 21, 34, 0.55);
          }
          .dc-scroll-surface::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(43, 228, 255, 0.98), rgba(43, 228, 255, 0.6));
          }

          .dc-topbar,
          .dc-stat-strip,
          .dc-board-nav,
          .dc-prospect-board,
          .dc-selected-file,
          .dc-stock-rail,
          .dc-intel-feed {
            border: 1px solid var(--dc-line);
            border-radius: 10px;
            background: var(--dc-panel);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.35);
          }

          .dc-topbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center;
            min-height: clamp(88px, 10vh, 112px);
            padding: 10px 16px;
            gap: 8px;
            backdrop-filter: blur(8px);
            background:
              linear-gradient(180deg, rgba(8, 24, 39, 0.84), rgba(5, 15, 26, 0.8)),
              radial-gradient(circle at 50% -18%, rgba(43,228,255,0.08), transparent 62%);
            box-shadow:
              inset 0 1px 0 rgba(255,255,255,0.04),
              0 0 0 1px rgba(80, 124, 158, 0.18),
              0 14px 28px rgba(0,0,0,0.28);
          }
          .dc-topbar--draft-hud .dc-topbar__left,
          .dc-topbar--draft-hud .dc-topbar__center,
          .dc-topbar--draft-hud .dc-topbar__right {
            min-width: 0;
            display: flex;
            align-items: center;
          }
          .dc-topbar--draft-hud .dc-topbar__left {
            justify-content: flex-start;
            gap: 10px;
            width: 100%;
          }
          .dc-topbar--draft-hud .dc-topbar__center {
            justify-content: center;
            flex-direction: column;
            gap: 2px;
          }
          .dc-topbar--draft-hud .dc-topbar__right {
            justify-content: flex-end;
            flex-direction: column;
            gap: 0;
            width: 100%;
          }
          .dc-back-btn--hud {
            padding: 4px 8px;
            font-size: 0.72rem;
            line-height: 1;
            height: 32px;
            border: 0;
            background: transparent;
            box-shadow: none;
            font-weight: 700;
          }
          .dc-record-mini {
            font-size: 0.92rem;
            letter-spacing: 0.04em;
            color: var(--dc-text);
            white-space: nowrap;
            height: 32px;
            display: inline-flex;
            align-items: center;
            padding: 0 2px;
            border: 0;
            background: transparent;
            font-family: var(--dc-font-mono);
            font-weight: 700;
          }
          .dc-topbar__center-logo .dc-team-logo {
            width: clamp(56px, 4.8vw, 74px);
            height: clamp(56px, 4.8vw, 74px);
            object-fit: contain;
            border-radius: 0;
            border: 0;
            background: transparent;
            box-shadow: none;
          }
          .dc-team-logo-fallback {
            width: clamp(56px, 4.8vw, 74px);
            height: clamp(56px, 4.8vw, 74px);
            border-radius: 0;
            border: 0;
            background: transparent;
            display: grid;
            place-items: center;
            font-family: var(--dc-font-title);
            font-size: 1rem;
            letter-spacing: 0.08em;
            color: var(--dc-cyan);
          }
          .dc-date-mini {
            font-size: 0.76rem;
            color: var(--dc-muted);
            letter-spacing: 0.04em;
            white-space: nowrap;
            line-height: 1;
            margin-top: 2px;
          }
          .dc-topbar__right-picks {
            border: 0;
            background: transparent;
            padding: 2px 0;
            min-width: 64px;
            min-height: 38px;
            width: fit-content;
            justify-self: end;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-shadow: none;
          }
          .dc-pick-count-mini {
            font-family: var(--dc-font-number);
            font-size: 1.45rem;
            line-height: 1;
            color: var(--dc-gold);
            text-align: center;
          }
          .dc-topbar__right-picks small {
            font-size: 0.8rem;
            color: rgba(226, 237, 246, 0.8);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            line-height: 1;
            margin-top: 1px;
            font-weight: 700;
          }
          .dc-topbar__left { padding-left: 6px; }
          .dc-topbar__right { padding-right: 6px; }
          .dc-back-btn, .dc-nav-chip, .dc-lens-btn, .dc-sort, .dc-mini-select, .dc-profile-tab, .dc-view-full {
            border: 1px solid var(--dc-line);
            border-radius: 4px;
            background: rgba(7, 18, 30, 0.9);
            color: var(--dc-text);
            cursor: pointer;
            font-size: 0.6875rem;
            padding: 5px 10px;
          }
          .dc-back-btn:hover, .dc-nav-chip:hover, .dc-lens-btn:hover, .dc-profile-tab:hover { border-color: rgba(43,228,255,0.55); }

          .dc-stat-strip {
            display: grid;
            grid-template-columns: repeat(4, max-content);
            justify-content: space-between;
            align-items: center;
            gap: 4px;
            min-height: 66px;
            padding: 8px 14px;
            border-radius: 10px;
            background:
              linear-gradient(180deg, rgba(4, 12, 20, 0.98), rgba(3, 9, 16, 0.98)),
              radial-gradient(circle at 50% -8%, rgba(43,228,255,0.045), transparent 65%);
            border: 1px solid rgba(73, 110, 138, 0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 10px 20px rgba(0,0,0,0.32);
          }
          .dc-hud-strip__item {
            display: flex;
            align-items: center;
            gap: 7px;
            min-width: 0;
            padding: 2px 0;
          }
          .dc-hud-strip__item--clickable {
            cursor: pointer;
            border: none;
            background: transparent;
            color: inherit;
            font: inherit;
            text-align: left;
            border-radius: 6px;
            padding: 4px 6px;
            margin: -2px -4px;
            transition: background 0.15s ease, box-shadow 0.15s ease;
          }
          .dc-hud-strip__item--clickable:hover,
          .dc-hud-strip__item--clickable:focus-visible {
            background: rgba(43, 228, 255, 0.08);
            box-shadow: 0 0 0 1px rgba(43, 228, 255, 0.22);
            outline: none;
          }
          .dc-hud-strip__icon {
            width: 32px;
            height: 32px;
            display: inline-grid;
            place-items: center;
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1;
            filter: drop-shadow(0 0 8px rgba(43,228,255,0.18));
          }
          .dc-hud-strip__text {
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 1px;
          }
          .dc-hud-strip__text small {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
            line-height: 1;
            white-space: nowrap;
          }
          .dc-hud-strip__text strong {
            font-size: 0.94rem;
            letter-spacing: 0.01em;
            color: var(--dc-text);
            line-height: 1.1;
            white-space: nowrap;
          }

          .dc-board-nav {
            width: 100%;
            height: 100%;
            min-height: 0;
            padding: 8px 5px;
            display: flex;
            flex-direction: column;
            border-radius: 10px;
            background:
              linear-gradient(180deg, rgba(4, 12, 20, 0.98), rgba(3, 9, 16, 0.98)),
              radial-gradient(circle at 50% -8%, rgba(43,228,255,0.045), transparent 65%);
            border: 1px solid rgba(73, 110, 138, 0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 10px 20px rgba(0,0,0,0.32);
          }
          .dc-board-nav__stack {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            /* Lens selector reads as a stacked rail at the top of the board,
               not four buttons floated apart down 500px of empty rail. */
            justify-content: flex-start;
            gap: 2px;
          }
          .dc-board-nav__item {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 5px;
            padding: 7px 4px;
            border: 0;
            border-radius: var(--radius-hud, 4px);
            background: transparent;
            color: rgba(138, 160, 182, 0.9);
            cursor: pointer;
            transition: color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
          }
          .dc-board-nav__item:hover {
            color: rgba(215, 232, 246, 0.96);
            background: rgba(255, 255, 255, 0.04);
          }
          .dc-board-nav__item.is-active {
            color: var(--dc-cyan);
            background: rgba(43, 228, 255, 0.08);
            box-shadow: inset 0 0 0 1px rgba(43, 228, 255, 0.28);
          }
          .dc-board-nav__label {
            font-size: 0.6875rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
          }
          .dc-board-nav__icon {
            width: 22px;
            height: 22px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            position: relative;
            opacity: 0.92;
          }
          .dc-board-nav__item.is-active .dc-board-nav__icon { opacity: 1; filter: drop-shadow(0 0 6px rgba(43,228,255,0.35)); }
          .dc-board-nav__icon--rank {
            flex-direction: column;
            gap: 3px;
            align-items: flex-start;
          }
          .dc-board-nav__icon--rank i {
            display: block;
            height: 3px;
            border-radius: 999px;
            background: currentColor;
          }
          .dc-board-nav__icon--rank i:nth-child(1) { width: 18px; }
          .dc-board-nav__icon--rank i:nth-child(2) { width: 14px; }
          .dc-board-nav__icon--rank i:nth-child(3) { width: 10px; }
          .dc-board-nav__icon--fwd {
            flex-direction: row;
            gap: 3px;
            align-items: flex-end;
          }
          .dc-board-nav__icon--fwd i {
            display: block;
            width: 4px;
            border-radius: 2px 2px 0 0;
            background: currentColor;
          }
          .dc-board-nav__icon--fwd i:nth-child(1) { height: 8px; }
          .dc-board-nav__icon--fwd i:nth-child(2) { height: 13px; }
          .dc-board-nav__icon--fwd i:nth-child(3) { height: 10px; }
          .dc-board-nav__shield {
            display: block;
            width: 14px;
            height: 16px;
            border: 2px solid currentColor;
            border-radius: 2px 2px 8px 8px;
            box-sizing: border-box;
          }
          .dc-board-nav__mask {
            display: block;
            width: 16px;
            height: 14px;
            border: 2px solid currentColor;
            border-radius: 8px 8px 4px 4px;
            box-sizing: border-box;
            position: relative;
          }
          .dc-board-nav__mask::after {
            content: "";
            position: absolute;
            left: 50%;
            bottom: -1px;
            width: 10px;
            height: 2px;
            transform: translateX(-50%);
            background: currentColor;
            border-radius: 1px;
          }
          .dc-command-grid {
            min-height: 0;
            display: grid;
            grid-template-columns: 84px minmax(0, 1fr) 360px;
            gap: 10px;
          }
          .dc-board-nav-rail { min-height: 0; display: flex; width: 84px; }
          .dc-prospect-board,
          .dc-stock-rail {
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
          }
          .dc-prospect-board__head,
          .dc-stock-rail__head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--dc-line);
            padding: 10px 12px;
            flex-shrink: 0;
          }
          .dc-prospect-board__head h2,
          .dc-stock-rail__head h2 {
            margin: 0;
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--dc-cyan);
          }
          .dc-prospect-board__head span {
            color: var(--dc-muted);
            font-size: 0.6875rem;
            font-family: var(--dc-font-mono);
          }
          .dc-prospect-board__list {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            padding: 0 4px 12px;
          }
          .dc-prospect-board__empty { padding: 0; }

          /* Intentional board-closed state instead of a stray sentence. */
          .dc-board-standby {
            margin: 14px 12px;
            padding: 14px 16px;
            border: 1px solid var(--dc-line);
            border-left: 3px solid var(--dc-gold);
            background: rgba(6, 21, 34, 0.6);
            max-width: 460px;
          }
          .dc-board-standby__label {
            display: block;
            margin-bottom: 5px;
            font-size: 0.6875rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dc-gold);
          }

          .dc-prospect-board__columns,
          .dc-prospect-row {
            display: grid;
            grid-template-columns: 44px minmax(160px, 1.5fr) 42px minmax(68px, 0.8fr) 68px 66px 52px 56px;
            align-items: center;
            gap: 10px;
            padding: 0 12px;
          }

          .dc-prospect-board__columns {
            position: sticky;
            top: 0;
            z-index: 2;
            min-height: 30px;
            margin-bottom: 2px;
            padding-top: 6px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--dc-line);
            background: rgba(6, 16, 28, 0.98);
          }

          .dc-prospect-board__col {
            font-size: var(--type-phase-label-size);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--dc-muted);
            font-weight: 900;
          }

          .dc-prospect-board__col--pot,
          .dc-prospect-board__col--conf,
          .dc-prospect-board__col--stock { text-align: right; }

          /* The Pot header sits over a padded chip, so pull it back into alignment. */
          .dc-prospect-board__col--pot { padding-right: 7px; }

          .dc-prospect-row {
            width: 100%;
            min-height: 40px;
            border: none;
            border-bottom: 1px solid rgba(156, 218, 236, 0.08);
            border-radius: 0;
            background: transparent;
            color: var(--dc-text);
            padding-top: 6px;
            padding-bottom: 6px;
            margin-bottom: 0;
            text-align: left;
            cursor: pointer;
            transition: background var(--motion-micro);
            font-size: var(--type-compact-size);
          }

          .dc-prospect-row:hover {
            background: var(--ops-cyan-soft);
          }

          .dc-prospect-row.is-selected {
            background: var(--ops-table-sel);
            box-shadow: inset 2px 0 0 var(--dc-cyan);
          }

          /* Late-round noise recedes — but never the ceiling, which is the whole
             reason to keep reading past pick 32. */
          .dc-prospect-row.is-late-round .dc-prospect-row__conf {
            color: var(--dc-muted);
          }

          /* Department signature: the rank ticket. Every board position is a
             numbered draft ticket with a perforated tear edge. */
          .dc-prospect-row__rank {
            position: relative;
            display: flex;
            align-items: center;
            padding-right: 8px;
          }

          .dc-prospect-row__rank::after {
            content: "";
            position: absolute;
            right: 0;
            top: 20%;
            bottom: 20%;
            width: 1px;
            background: repeating-linear-gradient(
              180deg,
              rgba(233, 168, 60, 0.5) 0 3px,
              transparent 3px 6px
            );
          }

          .dc-prospect-row__rank span {
            font-family: var(--dc-font-number);
            font-size: var(--type-table-value-size);
            font-weight: 800;
            color: var(--dc-gold);
            line-height: 1;
            font-variant-numeric: tabular-nums;
          }

          .dc-prospect-row.is-selected .dc-prospect-row__rank::after {
            background: var(--dc-cyan);
          }

          .dc-prospect-row__player {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
          }

          .dc-prospect-row__flag {
            width: 18px;
            height: 12px;
            object-fit: cover;
            border-radius: 1px;
            flex: 0 0 auto;
          }

          .dc-prospect-row__name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: var(--type-compact-size);
            font-weight: 700;
          }

          .dc-prospect-row__cell {
            font-family: var(--dc-font-mono);
            font-size: var(--type-table-value-size);
            font-weight: 700;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }

          .dc-prospect-row__league {
            color: var(--dc-muted);
            font-weight: 600;
          }

          /* The chip hugs its number and hangs off the right edge, so a filled tier
             reads as a badge on the grade rather than a highlight on the row. */
          .dc-prospect-row__pot {
            justify-self: end;
            display: inline-flex;
            align-items: center;
            box-sizing: border-box;
            padding: 2px 7px;
            border: 1px solid transparent;
            border-radius: 3px;
            color: var(--dc-text);
            font-variant-numeric: tabular-nums;
            transition: background var(--motion-micro), box-shadow var(--motion-micro);
          }

          .dc-prospect-row__pot.is-pot-fringe { color: var(--dc-pot-fringe); }
          .dc-prospect-row__pot.is-pot-depth { color: var(--dc-pot-depth); }
          .dc-prospect-row__pot.is-pot-moderate { color: var(--dc-pot-moderate); }
          .dc-prospect-row__pot.is-pot-strong { color: var(--dc-pot-strong); }

          /* Hue alone separates the cool tiers too weakly to scan, so the top three
             climb a second axis — wash, then border, then glow. */
          .dc-prospect-row__pot.is-pot-high {
            color: var(--dc-pot-high);
            background: rgba(19, 216, 231, 0.1);
            border-color: rgba(19, 216, 231, 0.24);
          }

          .dc-prospect-row__pot.is-pot-elite {
            color: var(--dc-pot-elite);
            background: rgba(233, 168, 60, 0.14);
            border-color: rgba(233, 168, 60, 0.36);
          }

          .dc-prospect-row__pot.is-pot-generational {
            color: var(--dc-pot-generational);
            background: linear-gradient(90deg, rgba(255, 201, 77, 0.05), rgba(255, 201, 77, 0.22));
            border-color: rgba(255, 201, 77, 0.52);
            box-shadow: 0 1px 12px rgba(255, 201, 77, 0.26);
          }

          /* An unsettled read keeps its hue but not the chip — the ceremony is earned
             by scouting the kid, so a fogged range can never masquerade as a lock. */
          .dc-prospect-row__pot.is-range {
            background: none;
            border-color: transparent;
            box-shadow: none;
            font-weight: 600;
          }

          .dc-prospect-row__conf.is-fog-heavy,
          .dc-prospect-row__conf.is-fog-medium {
            color: var(--dc-muted);
          }

          .dc-prospect-row__stock.is-up { color: var(--dc-green); }
          .dc-prospect-row__stock.is-down { color: var(--dc-red); }

          .dc-prospect-identity__avatar-wrap {
            position: relative;
            width: 92px;
            height: 92px;
            border-radius: 50%;
            overflow: hidden;
            border: 1px solid rgba(118, 200, 245, 0.32);
            background: linear-gradient(180deg, rgba(12, 30, 48, 0.98), rgba(6, 16, 28, 1));
            display: grid;
            place-items: center;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 14px rgba(0,0,0,0.24);
          }
          .dc-prospect-identity__flag-badge {
            position: absolute;
            right: 2px;
            bottom: 2px;
            width: 44px;
            height: 30px;
            object-fit: cover;
            border-radius: 4px;
            border: 1.5px solid rgba(200, 228, 245, 0.55);
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            z-index: 2;
          }
          .dc-prospect-identity__flag-fallback {
            position: absolute;
            right: 2px;
            bottom: 2px;
            min-width: 22px;
            height: 22px;
            padding: 0 3px;
            border-radius: 4px;
            background: rgba(8, 21, 34, 0.92);
            border: 1px solid rgba(118, 200, 245, 0.28);
            color: #e8f4ff;
            font-size: 0.78rem;
            line-height: 20px;
            text-align: center;
            z-index: 2;
          }
          .dc-prospect-identity .dc-board-headshot.player-headshot {
            --size: 72px;
            --jersey: #0f2a42;
            --jersey2: #163a56;
            --skin: #c89267;
            --skin-shadow: #875538;
            --hair: #2a211c;
            --hair2: #1a1410;
            position: relative;
            z-index: 1;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            filter: none !important;
          }
          .dc-prospect-identity .dc-board-headshot.player-headshot::before {
            background: linear-gradient(180deg, var(--jersey), var(--jersey2)) !important;
          }
          .dc-prospect-identity .dc-board-headshot .ph-flag { display: none; }
          .dc-prospect-identity .dc-board-headshot .ph-face {
            background:
              radial-gradient(circle at 34% 26%, rgba(255,255,255,0.22), transparent 18%),
              linear-gradient(145deg, var(--skin), var(--skin-shadow)) !important;
          }
          .dc-prospect-identity .dc-board-headshot .ph-hair {
            background: linear-gradient(160deg, var(--hair), var(--hair2)) !important;
          }
          .dc-prospect-identity .dc-board-headshot[class*="headshot-"] .ph-face {
            background:
              radial-gradient(circle at 34% 26%, rgba(255,255,255,0.22), transparent 18%),
              linear-gradient(145deg, var(--skin), var(--skin-shadow)) !important;
            box-shadow: none !important;
          }
          .dc-prospect-identity .dc-board-headshot[class*="headshot-"] .ph-hair {
            background: linear-gradient(160deg, var(--hair), var(--hair2)) !important;
            box-shadow: none !important;
          }

          .dc-prospect-row__identity { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
          .dc-prospect-row__name {
            display: block;
            font-family: var(--dc-font-title);
            font-size: 1.22rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            line-height: 1.15;
          }
          .dc-prospect-row__sub,
          .dc-prospect-row__team,
          .dc-prospect-row__nation-label {
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
          .dc-prospect-row__sub {
            font-size: 0.92rem;
            color: #b8d4ea;
            font-weight: 500;
          }
          .dc-prospect-row__team {
            font-size: 0.86rem;
            color: var(--dc-muted);
          }
          .dc-prospect-row__nation-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            color: var(--dc-cyan);
            font-family: var(--dc-font-mono);
            font-weight: 600;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
          }
          .dc-prospect-row__nation-flag {
            width: 22px;
            height: 15px;
            object-fit: cover;
            border-radius: 3px;
            border: 1px solid rgba(200, 228, 245, 0.45);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
            flex: 0 0 auto;
          }

          .dc-prospect-metric {
            display: flex;
            flex-direction: column;
            gap: 4px;
            min-width: 0;
          }
          .dc-prospect-metric--center { align-items: center; text-align: center; }
          .dc-prospect-metric--right { align-items: flex-end; text-align: right; }
          .dc-prospect-metric__value {
            font-family: var(--dc-font-mono);
            font-size: 1rem;
            font-weight: 600;
            color: #d8eeff;
            line-height: 1.3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
          }
          .dc-prospect-metric.is-scout .dc-prospect-metric__value {
            font-family: var(--dc-font-number);
            font-size: 1.28rem;
            color: var(--dc-cyan);
          }
          .dc-scout-confidence {
            gap: 5px;
          }
          .dc-scout-confidence__bar {
            display: block;
            width: 100%;
            max-width: 56px;
            height: 4px;
            margin: 0 auto;
            /* Confidence aperture: a measured scale, not a rounded meter. */
            border-radius: 1px;
            background:
              repeating-linear-gradient(90deg, rgba(255,255,255,0.16) 0 1px, transparent 1px 25%),
              rgba(118, 200, 245, 0.14);
            overflow: hidden;
          }
          .dc-scout-confidence__fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, rgba(43, 228, 255, 0.55), var(--dc-cyan));
            transition: opacity 0.2s ease;
          }
          .dc-conf-fog--locked .dc-prospect-metric__value,
          .dc-conf-fog--locked .dc-scout-confidence__fill {
            opacity: 1;
            filter: none;
          }
          .dc-conf-fog--strong .dc-prospect-metric__value { opacity: 0.88; filter: none; }
          .dc-conf-fog--strong .dc-scout-confidence__fill { opacity: 0.92; }
          .dc-conf-fog--solid .dc-prospect-metric__value { opacity: 0.72; filter: blur(0.35px); }
          .dc-conf-fog--solid .dc-scout-confidence__fill { opacity: 0.78; }
          .dc-conf-fog--limited .dc-prospect-metric__value { opacity: 0.54; filter: blur(0.7px); }
          .dc-conf-fog--limited .dc-scout-confidence__fill { opacity: 0.62; }
          .dc-conf-fog--unknown .dc-prospect-metric__value { opacity: 0.36; filter: blur(1.1px); }
          .dc-conf-fog--unknown .dc-scout-confidence__fill { opacity: 0.45; }
          .dc-conf-fog--blind .dc-prospect-metric__value { opacity: 0.24; filter: blur(1.5px); }
          .dc-conf-fog--blind .dc-scout-confidence__fill { opacity: 0.28; }
          .dc-prospect-metric.is-potential .dc-prospect-metric__value {
            font-family: var(--dc-font-number);
            font-size: 1.28rem;
            color: var(--dc-gold);
          }
          /* Ceiling withheld (low draft attention): the metric shows only the floor,
             tinted amber and de-emphasized to signal upside is unknown. */
          .dc-prospect-metric.is-ceiling-hidden .dc-prospect-metric__value {
            color: #f0b45a;
            opacity: 0.9;
            filter: none;
          }
          .dc-prospect-metric.is-ceiling-hidden .dc-prospect-metric__label {
            color: rgba(240, 180, 90, 0.75);
          }
          .dc-profile-roster-note {
            margin: 0 0 10px;
            padding: 8px 10px;
            font-size: 0.82rem;
            line-height: 1.35;
            color: rgba(200, 210, 222, 0.85);
            border-left: 2px solid rgba(100, 180, 200, 0.45);
          }
          .dc-profile-roster-note .dc-profile-tags__label {
            display: block;
            margin-bottom: 2px;
          }
          .dc-prospect-metric__label {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
            font-weight: 600;
          }
          .dc-prospect-row .dc-prospect-metric__label { display: none; }
          .dc-prospect-metric.is-rise .dc-prospect-metric__value { color: var(--dc-green); }
          .dc-prospect-metric.is-fall .dc-prospect-metric__value { color: var(--dc-red); }
          .dc-prospect-metric.is-new .dc-prospect-metric__value { color: var(--dc-cyan); }
          .dc-prospect-metric.is-stable .dc-prospect-metric__value { color: #9ec7e8; }
          .dc-prospect-metric.is-muted .dc-prospect-metric__value { color: var(--dc-muted); }

          .dc-selected-file { min-height: 0; display: flex; flex-direction: column; overflow: hidden; }

          .dc-selected-file__hero { position: relative; overflow: hidden; }
          .dc-selected-file__hero::before { content: ""; position: absolute; inset: -10% 30% -45% 30%; background: radial-gradient(circle, rgba(43,228,255,0.15), transparent 65%); pointer-events: none; }
          .dc-selected-file__identity h2 { margin: 2px 0; font-family: var(--dc-font-title); letter-spacing: 0.05em; font-size: 1.28rem; }
          .dc-selected-file__identity p { margin: 0; color: var(--dc-muted); font-size: 0.7rem; }
          .dc-selected-file__rank { color: var(--dc-gold); font-size: 0.7rem; letter-spacing: 0.08em; }
          .dc-selected-file__quick { margin-left: auto; display: grid; grid-template-columns: repeat(3, minmax(58px,1fr)); gap: 6px; }
          .dc-selected-file__quick div { border: 1px solid var(--dc-line); border-radius: 8px; background: var(--dc-panel-soft); padding: 5px; text-align: center; }
          .dc-selected-file__quick span { display: block; color: var(--dc-muted); font-size: 0.6875rem; text-transform: uppercase; }
          .dc-selected-file__quick strong { font-family: var(--dc-font-number); }

          .dc-selected-file__grid { padding: 8px 10px; display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 6px; }
          .dc-glass-card { border: 1px solid var(--dc-line); border-radius: 10px; background: var(--dc-panel-soft); padding: 7px; min-height: 66px; }
          .dc-glass-card h3 { margin: 0 0 4px; font-size: 0.6875rem; color: var(--dc-muted); text-transform: uppercase; letter-spacing: 0.07em; }
          .dc-glass-card p, .dc-glass-card small { margin: 0; font-size: 0.6875rem; }

          /* War-board marks are squared board notation, not pills. */
          .dc-stock-badge, .dc-tier-badge, .dc-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--dc-line);
            border-radius: var(--radius-ops, 2px);
            padding: 2px 6px;
            font-size: 0.6875rem;
            line-height: 1;
            letter-spacing: 0.06em;
            background: rgba(255,255,255,0.04);
          }
          .dc-stock-badge--rise { color: var(--dc-green); border-color: rgba(108,247,166,0.4); }
          .dc-stock-badge--fall { color: var(--dc-red); border-color: rgba(255,111,126,0.4); }
          .dc-stock-badge--new { color: var(--dc-cyan); border-color: rgba(43,228,255,0.45); }
          .dc-tier-badge--gold { color: var(--dc-gold); border-color: rgba(244,198,110,0.45); }
          .dc-tier-badge--cyan { color: var(--dc-cyan); border-color: rgba(43,228,255,0.45); }
          .dc-tier-badge--purple { color: #d8eeff; border-color: rgba(216,238,255,0.4); }
          .dc-badges { display: inline-flex; gap: 4px; flex-wrap: wrap; }
          .dc-badge--watch { color: #ffe39e; }
          .dc-badge--target { color: #a5e9ff; }
          .dc-badge--dnd, .dc-badge--bust { color: #ff9ca6; }

          .dc-action-strip { padding: 2px 10px 8px; }
          .dc-scout-actions { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
          .dc-scout-menu { display: inline-flex; gap: 5px; padding: 4px 6px; border: 1px solid var(--dc-line); border-radius: var(--radius-hud, 4px); background: rgba(7,18,30,0.7); }
          .dc-btn {
            border: 1px solid var(--dc-line);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.6875rem;
            cursor: pointer;
            color: var(--dc-text);
            background: rgba(8, 22, 36, 0.92);
          }
          .dc-btn--primary { border-color: rgba(43,228,255,0.65); box-shadow: 0 0 18px rgba(43,228,255,0.18); background: linear-gradient(180deg, rgba(27,83,124,0.9), rgba(8,31,50,0.9)); }
          .dc-btn--secondary { background: rgba(14,30,46,0.9); }
          .dc-btn--danger { border-color: rgba(255,111,126,0.55); color: #ffd6db; background: linear-gradient(180deg, rgba(95,34,42,0.9), rgba(43,17,22,0.9)); }
          .dc-btn.is-active { border-color: rgba(108,247,166,0.6); }
          .dc-btn.is-disabled, .dc-btn:disabled { opacity: 0.4; cursor: not-allowed; }
          .dc-file-toggle-wrap { padding: 0 10px 8px; }

          .dc-selected-file__tabs { display: flex; gap: 5px; flex-wrap: wrap; padding: 0 10px 8px; border-bottom: 1px solid var(--dc-line); }
          .dc-profile-tab.is-active { color: #061018; background: linear-gradient(180deg, #6df7ff, #47c8d8); border-color: transparent; }
          .dc-selected-file__detail { min-height: 0; flex: 1; padding: 8px 10px 10px; }

          .dc-stock-card { padding: 8px 10px; border-bottom: 1px solid var(--dc-line); position: relative; }
          .dc-stock-card::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(43,228,255,0.5), transparent); opacity: .55; }
          .dc-stock-card h4 { margin: 0 0 5px; font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dc-muted); }
          .dc-stock-card--rise h4 { color: var(--dc-green); }
          .dc-stock-card--fall h4 { color: var(--dc-red); }
          .dc-stock-row { width: 100%; border: 0; background: transparent; color: var(--dc-text); display: grid; grid-template-columns: 36px 1fr auto; gap: 5px; padding: 4px 0; text-align: left; cursor: pointer; font-family: var(--dc-font-mono); font-size: 0.6875rem; }
          .dc-stock-row:hover { color: var(--dc-cyan); }
          .dc-stock-row__delta.is-up { color: var(--dc-green); }
          .dc-stock-row__delta.is-down { color: var(--dc-red); }
          .dc-stock-rail__movement {
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            gap: 0;
            min-height: 0;
          }
          .dc-stock-rail__leaders {
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            margin-top: 0;
            border-top: 1px solid var(--dc-line);
            overflow: hidden;
          }
          .dc-leaders-panel {
            display: flex;
            flex-direction: column;
            min-height: 0;
            height: 100%;
            padding: 8px 10px 10px;
          }
          .dc-leaders-panel--compact .dc-side-title h2 { font-size: 0.6875rem; }
          .dc-side-title {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
          }
          .dc-side-title h2 {
            margin: 0;
            font-size: 0.6875rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--dc-cyan);
          }
          .dc-leaders-date {
            font-size: 0.6875rem;
            color: var(--dc-muted);
            font-family: var(--dc-font-mono);
            white-space: nowrap;
          }
          .dc-leaders-panel--compact .dc-leader-tabs { flex-wrap: wrap; gap: 4px; }
          .dc-leaders-panel--compact .dc-leader-tab { font-size: 0.6875rem; padding: 4px 7px; }
          .dc-leaders-panel--compact .dc-leader-row {
            font-size: 0.6875rem;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 6px;
            padding: 5px 0;
          }
          .dc-leaders-panel--compact .dc-leader-row__name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            min-width: 0;
          }
          .dc-leaders-panel--compact .dc-leader-meta {
            display: none;
          }
          .dc-leaders-panel--compact .dc-leader-scroll {
            flex: 1 1 auto;
            min-height: 0;
            overflow: auto;
            margin-top: 6px;
          }
          .dc-view-full--leaders {
            margin-top: 8px;
            width: 100%;
            padding: 8px 10px;
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-radius: 10px;
            border: 1px solid rgba(43, 228, 255, 0.35);
            background: linear-gradient(180deg, rgba(12, 34, 52, 0.95), rgba(8, 21, 34, 0.92));
            color: var(--dc-cyan);
            cursor: pointer;
            flex-shrink: 0;
          }
          .dc-view-full--leaders:hover {
            border-color: rgba(43, 228, 255, 0.65);
            background: linear-gradient(180deg, rgba(16, 42, 64, 0.98), rgba(10, 26, 42, 0.95));
          }
          .dc-leader-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
          }
          .dc-leader-tab {
            border: 1px solid rgba(118, 200, 245, 0.22);
            border-radius: var(--radius-ops, 2px);
            background: rgba(255,255,255,0.03);
            color: var(--dc-muted);
            padding: 4px 8px;
            font-size: 0.6875rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            cursor: pointer;
          }
          .dc-leader-tab.is-active {
            color: #061018;
            background: linear-gradient(180deg, #6df7ff, #47c8d8);
            border-color: transparent;
          }
          .dc-leader-row {
            width: 100%;
            border: 0;
            background: transparent;
            color: var(--dc-text);
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) 0.8fr auto;
            gap: 6px;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid rgba(118, 200, 245, 0.08);
            text-align: left;
            cursor: pointer;
            font-size: 0.6875rem;
          }
          .dc-leader-row:hover { color: var(--dc-cyan); }
          .dc-leader-row strong { font-family: var(--dc-font-mono); color: #d8eeff; white-space: nowrap; }
          .dc-leader-meta { color: var(--dc-muted); font-size: 0.6875rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
          .dc-leaders-modal {
            position: fixed;
            inset: 0;
            z-index: var(--z-modal, 1200);
            display: grid;
            place-items: center;
            padding: 16px;
          }
          .dc-leaders-modal-sc {
            --lm-panel: rgba(10, 28, 42, 0.94);
            --lm-line: rgba(150, 190, 210, 0.16);
            --lm-muted: #8ba0af;
            --lm-cyan: #00d8df;
            --lm-gold: #e8a536;
            --lm-green: #48d88b;
            --lm-red: #ff6464;
          }
          .dc-leaders-modal__panel {
            position: relative;
            width: min(70vw, 1280px);
            height: min(70vh, 860px);
            border: 1px solid rgba(118, 200, 245, 0.38);
            border-radius: 8px;
            background:
              radial-gradient(circle at 18% 12%, rgba(43, 228, 255, 0.1), transparent 34%),
              radial-gradient(circle at 88% 12%, rgba(232, 165, 54, 0.08), transparent 24%),
              linear-gradient(180deg, rgba(8, 21, 34, 0.97), rgba(5, 14, 24, 0.98));
            box-shadow:
              inset 0 1px 0 rgba(255,255,255,0.06),
              0 0 0 1px rgba(43, 228, 255, 0.12),
              0 28px 80px rgba(0,0,0,0.55);
            overflow: hidden;
            pointer-events: auto;
            display: flex;
            flex-direction: column;
          }
          .dc-leaders-modal__head {
            padding: 18px 20px 10px;
            border-bottom: 1px solid rgba(118, 200, 245, 0.14);
            flex-shrink: 0;
          }
          .dc-lm-eyebrow {
            margin: 0 0 4px;
            color: var(--lm-muted, var(--dc-muted));
            font-size: 0.6875rem;
            line-height: 1;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-weight: 800;
          }
          .dc-leaders-modal__head h2 {
            margin: 0;
            font-size: clamp(1.1rem, 2vw, 1.45rem);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #ffffff;
          }
          .dc-leaders-modal__context {
            margin: 6px 0 0;
            font-size: 0.6875rem;
            color: var(--lm-muted, var(--dc-muted));
            font-family: var(--dc-font-mono);
          }
          .dc-leader-tabs--modal {
            padding: 10px 16px 0;
            flex-shrink: 0;
          }
          .dc-leader-tabs--modal .dc-leader-tab {
            font-size: 0.6875rem;
            padding: 6px 10px;
          }
          .dc-leaders-modal__scroll {
            scrollbar-width: thin;
            scrollbar-color: rgba(0, 216, 223, 0.32) rgba(4, 16, 26, 0.72);
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar {
            width: 6px;
            height: 6px;
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar-track {
            background: rgba(4, 16, 26, 0.72);
            border-radius: 999px;
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(0, 216, 223, 0.42), rgba(118, 200, 245, 0.22));
            border: 1px solid rgba(0, 216, 223, 0.14);
            border-radius: 999px;
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(0, 216, 223, 0.58), rgba(118, 200, 245, 0.34));
          }
          .dc-lm-view-toggle {
            display: flex;
            gap: 6px;
            padding: 8px 16px 0;
            flex-shrink: 0;
          }
          .dc-lm-view-toggle__btn {
            border: 1px solid rgba(118, 200, 245, 0.16);
            background: rgba(255,255,255,0.03);
            color: var(--lm-muted, var(--dc-muted));
            border-radius: var(--radius-ops, 2px);
            padding: 5px 12px;
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
          }
          .dc-lm-view-toggle__btn.is-active {
            color: #041018;
            background: var(--lm-cyan, var(--dc-cyan));
            border-color: rgba(0, 216, 223, 0.65);
            box-shadow: 0 0 12px rgba(0, 216, 223, 0.22);
          }
          .dc-lm-sort-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 16px 0;
            flex-shrink: 0;
            flex-wrap: wrap;
          }
          .dc-lm-sort-bar__label {
            color: var(--lm-muted, var(--dc-muted));
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            flex: 0 0 auto;
          }
          .dc-lm-sort-bar__pills {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            min-width: 0;
          }
          .dc-lm-sort-pill {
            border: 1px solid rgba(118, 200, 245, 0.16);
            background: rgba(255,255,255,0.03);
            color: #d8eeff;
            border-radius: var(--radius-ops, 2px);
            padding: 4px 9px;
            font-size: 0.6875rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            cursor: pointer;
          }
          .dc-lm-sort-pill:hover {
            border-color: rgba(0, 216, 223, 0.42);
            background: rgba(0, 216, 223, 0.08);
          }
          .dc-lm-sort-pill.is-active {
            color: #041018;
            background: linear-gradient(180deg, rgba(0, 216, 223, 0.95), rgba(0, 180, 186, 0.88));
            border-color: rgba(0, 216, 223, 0.65);
            box-shadow: 0 0 10px rgba(0, 216, 223, 0.24);
          }
          .dc-leaders-modal__body {
            flex: 1 1 auto;
            min-height: 0;
            overflow: hidden;
            padding: 10px 16px 16px;
          }
          .dc-lm-section {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            background:
              linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.012)),
              var(--lm-panel, var(--dc-panel));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 12px 28px rgba(0,0,0,0.18);
            min-height: 0;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
          }
          .dc-lm-section__body {
            padding: 8px;
            min-height: 0;
            flex: 1 1 auto;
            overflow: auto;
          }
          .dc-lm-row-stack {
            display: grid;
            gap: 8px;
          }
          .dc-lm-row {
            display: grid;
            grid-template-columns: minmax(170px, 0.95fr) minmax(0, 2.2fr) auto;
            gap: 10px 14px;
            align-items: center;
            min-height: 88px;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid rgba(118, 200, 245, 0.14);
            background: rgba(8, 21, 34, 0.72);
          }
          .dc-lm-row--goalie {
            grid-template-columns: minmax(170px, 0.95fr) minmax(0, 1.8fr) auto;
          }
          .dc-lm-row__identity {
            display: flex;
            gap: 10px;
            align-items: center;
            min-width: 0;
          }
          .dc-lm-row__rank {
            flex: 0 0 auto;
            width: 30px;
            height: 30px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            font-family: var(--dc-font-number);
            font-weight: 800;
            font-size: 0.82rem;
            color: var(--lm-gold, var(--dc-gold));
            background: rgba(244, 198, 110, 0.08);
            border: 1px solid rgba(244, 198, 110, 0.22);
          }
          .dc-lm-row__player { min-width: 0; display: grid; gap: 2px; }
          .dc-lm-row__name {
            border: 0;
            background: transparent;
            color: #fff;
            font-size: 0.95rem;
            font-weight: 700;
            text-align: left;
            cursor: pointer;
            padding: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .dc-lm-row__name:hover { color: var(--lm-cyan, var(--dc-cyan)); }
          .dc-lm-row__meta {
            color: var(--lm-muted, var(--dc-muted));
            font-size: 0.6875rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .dc-lm-row__stats {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 10px;
            align-items: flex-start;
            min-width: 0;
          }
          .dc-lm-row__cluster {
            display: grid;
            gap: 4px;
            min-width: 0;
          }
          .dc-lm-row__cluster.is-emphasis-group .dc-lm-row__cluster-label {
            color: var(--lm-cyan, var(--dc-cyan));
          }
          .dc-lm-row__cluster-label {
            color: rgba(139, 160, 175, 0.85);
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }
          .dc-lm-row__cluster-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
          }
          .dc-lm-pill {
            display: inline-flex;
            align-items: baseline;
            gap: 5px;
            border-radius: var(--radius-ops, 2px);
            border: 1px solid rgba(118, 200, 245, 0.12);
            background: rgba(255,255,255,0.03);
            padding: 4px 8px;
            min-height: 0;
          }
          .dc-lm-pill__label {
            color: var(--lm-muted, var(--dc-muted));
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .dc-lm-pill__value {
            color: #e8f4ff;
            font-family: var(--dc-font-mono);
            font-size: 0.72rem;
            line-height: 1;
            font-weight: 700;
          }
          .dc-lm-pill.is-hero {
            border-color: rgba(0, 216, 223, 0.5);
            background: rgba(0, 216, 223, 0.1);
            box-shadow: 0 0 10px rgba(0, 216, 223, 0.14);
            padding: 5px 10px;
          }
          .dc-lm-pill.is-hero .dc-lm-pill__value {
            color: var(--lm-cyan, var(--dc-cyan));
            font-size: 0.88rem;
          }
          .dc-lm-pill.is-sortable {
            cursor: pointer;
            border: 1px solid rgba(118, 200, 245, 0.12);
            background: rgba(255,255,255,0.03);
            color: inherit;
            font: inherit;
            text-align: left;
          }
          .dc-lm-pill.is-sortable:hover {
            border-color: rgba(0, 216, 223, 0.38);
            background: rgba(0, 216, 223, 0.07);
          }
          .dc-lm-pill.is-sort-active {
            border-color: rgba(0, 216, 223, 0.55);
            background: rgba(0, 216, 223, 0.09);
            box-shadow: 0 0 0 1px rgba(0, 216, 223, 0.2);
          }
          .dc-lm-pill.is-sort-active .dc-lm-pill__value {
            color: var(--lm-cyan, var(--dc-cyan));
          }
          .dc-lm-row__draft {
            display: flex;
            flex-direction: column;
            gap: 5px;
            align-items: stretch;
            min-width: 72px;
          }
          .dc-lm-row__draft.is-emphasis-group .dc-lm-pill,
          .dc-lm-row__draft.is-emphasis-group .dc-lm-row__stock {
            border-color: rgba(232, 165, 54, 0.28);
          }
          .dc-lm-row__stock {
            border-radius: 999px;
            border: 1px solid rgba(118, 200, 245, 0.12);
            background: rgba(255,255,255,0.03);
            padding: 4px 8px;
            display: flex;
            align-items: baseline;
            gap: 5px;
          }
          .dc-lm-row__stock-btn {
            border: 0;
            background: transparent;
            padding: 0;
            display: flex;
            align-items: baseline;
            gap: 5px;
            cursor: pointer;
            color: inherit;
            font: inherit;
            width: 100%;
          }
          .dc-lm-row__stock span,
          .dc-lm-row__stock-btn span {
            color: var(--lm-muted, var(--dc-muted));
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .dc-lm-row__stock strong,
          .dc-lm-row__stock-btn strong {
            font-family: var(--dc-font-mono);
            font-size: 0.72rem;
            color: #d8eeff;
          }
          .dc-lm-row__stock.is-up strong,
          .dc-lm-row__stock-btn.is-up strong { color: var(--lm-green, var(--dc-green)); }
          .dc-lm-row__stock.is-down strong,
          .dc-lm-row__stock-btn.is-down strong { color: var(--lm-red, var(--dc-red)); }
          .dc-lm-row__stock.is-neutral strong,
          .dc-lm-row__stock-btn.is-neutral strong { color: var(--lm-muted, var(--dc-muted)); }
          .dc-lm-row__stock.is-hero,
          .dc-lm-row__stock.is-sort-active {
            border-color: rgba(0, 216, 223, 0.5);
            background: rgba(0, 216, 223, 0.1);
            box-shadow: 0 0 10px rgba(0, 216, 223, 0.14);
          }
          .dc-lm-sort-label {
            margin: 6px 0 0;
            color: var(--lm-cyan, var(--dc-cyan));
            font-size: 0.6875rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .dc-empty-note { color: var(--dc-muted); font-size: 0.6875rem; margin: 0; }

          .dc-intel-feed { padding: 6px 10px; }
          /* Standby wire collapses to a single labelled line. */
          .dc-intel-feed.is-idle {
            display: flex;
            align-items: baseline;
            gap: 12px;
            padding: 5px 10px;
            background: rgba(6, 21, 34, 0.6);
            box-shadow: none;
          }
          .dc-intel-feed.is-idle ul { margin: 0; }
          .dc-intel-feed.is-idle li { color: var(--dc-muted); }
          .dc-intel-feed.is-idle li::before { color: var(--dc-muted); }
          .dc-intel-feed h3 { margin: 0; font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dc-muted); }
          .dc-intel-feed ul { margin: 3px 0 0; padding: 0; list-style: none; display: flex; gap: 12px; white-space: nowrap; overflow: auto; font-family: var(--dc-font-mono); font-size: 0.6875rem; }
          .dc-intel-feed li::before { content: "• "; color: var(--dc-cyan); }

          .dc-shared-headshot.player-headshot.size-sm { --size: 40px; }
          .dc-shared-headshot.player-headshot.size-md { --size: 80px; }
          .dc-shared-headshot.player-headshot.size-lg { --size: 126px; }

          .dc-profile-body, .dc-stat-layout, .dc-attributes-layout, .dc-scout-layout, .dc-character-layout { display: grid; gap: 8px; }
          .dc-profile-card, .dc-info-card, .dc-list-card, .dc-summary-card, .dc-stat-card, .dc-attribute-card, .dc-scout-card, .dc-character-card {
            border: 1px solid var(--dc-line);
            border-radius: 10px;
            background: var(--dc-panel-soft);
            padding: 8px;
          }
          .dc-info-grid, .dc-big-stat-grid, .dc-grade-grid { font-size: 0.6875rem; }
          .dc-character-row, .dc-fit-row { font-size: 0.6875rem; }

          @media (max-width: 1600px) {
            .dc-command-grid { grid-template-columns: 84px minmax(0, 1fr) 320px; }
            /* Same eight-column ladder as the base board: the previous
               four-column override wrapped both the header and every row onto
               a second line, so ranks and grades no longer aligned. */
            .dc-prospect-board__columns,
            .dc-prospect-row {
              grid-template-columns: 40px minmax(150px, 1.5fr) 38px minmax(62px, 0.8fr) 64px 60px 48px 56px;
              gap: 9px;
              padding-left: 12px;
              padding-right: 12px;
            }
            .dc-prospect-row { min-height: 76px; }
            .dc-prospect-row__rank span { font-size: 1.45rem; }
            .dc-prospect-identity__avatar-wrap { width: 84px; height: 84px; }
            .dc-prospect-identity .dc-board-headshot.player-headshot { --size: 66px; }
            .dc-prospect-row__name { font-size: 1.1rem; }
            .dc-stock-rail { max-height: 100%; }
            .dc-stat-strip { grid-template-columns: repeat(2, minmax(0,1fr)); }
          }
          @media (max-width: 1200px) {
            .dc-root.dc-screen { grid-template-rows: auto auto minmax(0, 1fr) auto; }
            .dc-command-grid { grid-template-columns: 1fr; }
            .dc-board-nav-rail { grid-column: 1 / -1; width: 100%; }
            .dc-board-nav { height: auto; }
            .dc-board-nav__stack {
              flex-direction: row;
              justify-content: space-between;
              flex-wrap: wrap;
              gap: 4px;
            }
            .dc-board-nav__item {
              flex: 1 1 calc(25% - 4px);
              min-width: 72px;
              padding: 8px 6px;
            }
            .dc-prospect-board__columns { display: none; }
            .dc-prospect-row {
              grid-template-columns: 56px 76px 1fr;
              grid-template-areas:
                "rank avatar identity"
                "metrics metrics metrics";
              gap: 8px 12px;
              padding: 12px;
            }
            .dc-prospect-row__rank { grid-area: rank; }
            .dc-prospect-identity { grid-area: avatar; }
            .dc-prospect-row__identity { grid-area: identity; }
            .dc-prospect-row__metrics {
              grid-area: metrics;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 10px 12px;
            }
            .dc-prospect-row .dc-prospect-metric__label { display: block; }
            .dc-stat-strip { grid-template-columns: 1fr 1fr; }
            .dc-stock-rail { max-height: 280px; }
          }
          @media (max-width: 900px) {
            .dc-topbar { flex-wrap: wrap; }
          }
          @media (prefers-reduced-motion: reduce) {
            .dc-prospect-row { transition: none; }
          }

          .dc-profile-modal {
            position: fixed;
            inset: 0;
            z-index: var(--z-modal, 1200);
            display: grid;
            place-items: center;
            padding: clamp(8px, 1.2vh, 16px);
            pointer-events: none;
          }

          .dc-profile-modal--prospect .dc-signal-panel {
            border-color: var(--ops-grid-2);
            background:
              linear-gradient(180deg, rgba(6, 21, 34, 0.98), rgba(4, 13, 22, 0.98));
          }

          .dc-profile-modal--prospect .dc-signal-banner {
            background: linear-gradient(90deg, rgba(19, 216, 231, 0.08), transparent);
          }

          .dc-profile-modal--prospect .dc-signal-banner span {
            color: var(--ops-cyan);
          }

          .dc-profile-modal--uncertain .dc-signal-panel {
            border-style: dashed;
            border-color: rgba(128, 150, 168, 0.45);
          }

          .dc-profile-modal--uncertain .dc-profile-meter__track,
          .dc-profile-modal--uncertain .dc-skill-dna__fill {
            opacity: 0.55;
          }

          .dc-profile-modal--uncertain .dc-profile-attr-mini.is-locked {
            filter: blur(0.4px);
          }
          .dc-profile-modal__backdrop {
            position: absolute;
            inset: 0;
            border: 0;
            background: rgba(2, 8, 16, 0.72);
            backdrop-filter: blur(7px);
            pointer-events: auto;
            cursor: pointer;
          }
          .dc-signal-panel {
            position: relative;
            width: min(1360px, 97vw);
            height: min(94vh, 900px);
            height: min(94dvh, 900px);
            max-height: 94vh;
            max-height: 94dvh;
            border: 1px solid rgba(118, 200, 245, 0.42);
            border-radius: 10px;
            background:
              radial-gradient(ellipse 85% 70% at 12% -5%, rgba(56, 190, 255, 0.42), transparent 55%),
              radial-gradient(ellipse 70% 55% at 95% 105%, rgba(30, 140, 255, 0.32), transparent 52%),
              radial-gradient(ellipse 50% 40% at 48% 42%, rgba(40, 120, 210, 0.28), transparent 62%),
              linear-gradient(165deg, #143a62 0%, #0d2748 38%, #081c36 72%, #04101f 100%);
            box-shadow: 0 30px 80px rgba(0,0,0,0.62), inset 0 1px 0 rgba(118,200,245,0.22);
            overflow: hidden;
            pointer-events: auto;
            display: grid;
            grid-template-columns: minmax(210px, 240px) minmax(0, 1fr) minmax(210px, 250px);
            grid-template-rows: auto minmax(0, 1fr) auto;
            grid-template-areas:
              "banner banner banner"
              "identity core decision"
              "footer footer footer";
            gap: 0;
          }
          .dc-signal-panel--premium .dc-signal-identity {
            background:
              linear-gradient(180deg, rgba(10,28,44,0.96), rgba(5,14,24,0.88)),
              repeating-linear-gradient(0deg, transparent, transparent 11px, rgba(118,200,245,0.03) 12px);
          }


          .dc-skill-dna__ranges em {
            display: block;
            margin-top: 1px;
            font-style: normal;
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-cyan);
          }
          .dc-signal-club-card--text {
            grid-template-columns: 1fr;
          }
          .dc-signal-footer--premium {
            grid-template-columns: minmax(260px, 1fr) minmax(0, 1.35fr) auto;
            align-items: stretch;
          }
          .dc-signal-footer__stock { min-width: 0; }
          .dc-signal-actionbar--icon {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 4px 8px;
            align-self: center;
          }
          .dc-shortlist-icon {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            border: 1px solid rgba(244,198,110,0.55);
            background: linear-gradient(180deg, rgba(244,198,110,0.18), rgba(20,30,40,0.85));
            color: var(--dc-gold);
            cursor: pointer;
            display: grid;
            place-items: center;
            box-shadow: 0 0 16px rgba(244,198,110,0.2);
          }
          .dc-shortlist-icon svg {
            width: 22px;
            height: 22px;
            fill: none;
            stroke: currentColor;
            stroke-width: 1.8;
          }
          .dc-shortlist-icon.is-active {
            background: linear-gradient(180deg, #ffe08a, #f0bf55);
            color: #1a1204;
            box-shadow: 0 0 20px rgba(244,198,110,0.45);
          }
          .dc-shortlist-icon.is-active svg { fill: currentColor; stroke: currentColor; }
          .dc-signal-identity__board-no { display: none !important; }
          .dc-signal-assign { display: none !important; }

          .dc-signal-banner {
            grid-area: banner;
            display: flex;
            align-items: baseline;
            gap: 10px;
            padding: 8px 48px 8px 16px;
            border-bottom: 1px solid rgba(118, 200, 245, 0.16);
            background: linear-gradient(90deg, rgba(244,198,110,0.08), rgba(43,228,255,0.05), transparent);
          }
          .dc-signal-banner span {
            font-size: 0.6875rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--dc-gold);
            font-weight: 800;
          }
          .dc-signal-banner strong {
            font-family: var(--dc-font-title);
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #eaf6ff;
          }
          .dc-signal-flag {
            display: block;
            object-fit: cover;
            border: 1.5px solid rgba(255,255,255,0.92);
            border-radius: 3px;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.45), 0 4px 14px rgba(0,0,0,0.45), 0 0 18px rgba(43,228,255,0.18);
            background: #0a1520;
          }
          .dc-signal-flag-fallback {
            display: inline-grid;
            place-items: center;
            min-width: 28px;
            min-height: 18px;
            padding: 2px 6px;
            border: 1px solid rgba(255,255,255,0.7);
            color: #dff4ff;
            font-size: 0.6875rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            background: rgba(20,40,60,0.9);
          }
          .dc-signal-portrait {
            position: relative;
            align-self: center;
            width: 100%;
            display: grid;
            place-items: center;
            padding: 4px 0 8px;
          }
          .dc-signal-portrait .player-headshot { --size: 118px; }
          .dc-signal-portrait__flag {
            position: absolute;
            left: 10px;
            bottom: 10px;
            width: 36px;
            height: 24px;
            z-index: 2;
          }
          .dc-signal-portrait__stock {
            position: absolute;
            right: 6px;
            top: 4px;
            display: grid;
            justify-items: end;
            gap: 1px;
            padding: 4px 7px;
            border-radius: 6px;
            background: rgba(4,12,20,0.82);
            border: 1px solid rgba(244,198,110,0.45);
            box-shadow: 0 0 16px rgba(244,198,110,0.2);
          }
          .dc-signal-portrait__stock strong {
            font-family: var(--dc-font-number);
            font-size: 1.05rem;
            color: var(--dc-gold);
            line-height: 1;
          }
          .dc-signal-portrait__stock span {
            font-family: var(--dc-font-mono);
            font-size: 0.6875rem;
            color: var(--dc-cyan);
          }
          .dc-signal-portrait__stock.is-fall span { color: #ff8f9c; }
          .dc-signal-portrait__stock.is-rise span { color: var(--dc-green); }
          .dc-signal-identity__title {
            text-align: center;
            position: relative;
            padding-top: 2px;
          }
          .dc-signal-identity__board-no {
            position: absolute;
            right: 0;
            top: -6px;
            font-family: var(--dc-font-number);
            font-size: 1.55rem;
            color: rgba(244,198,110,0.88);
            text-shadow: 0 0 18px rgba(244,198,110,0.35);
            line-height: 1;
          }
          .dc-signal-identity__title h2 {
            margin: 0;
            font-family: var(--dc-font-title);
            font-size: clamp(1.05rem, 1.5vw, 1.35rem);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #f6d78a;
            text-shadow: 0 0 22px rgba(244,198,110,0.28);
            line-height: 1.1;
          }
          .dc-signal-identity__pos {
            margin: 4px 0 0;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            color: var(--dc-cyan);
            font-weight: 800;
          }
          .dc-signal-bio {
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 6px;
          }
          .dc-signal-bio li {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 8px;
            align-items: center;
            padding: 5px 6px;
            border: 1px solid rgba(118,200,245,0.12);
            background: rgba(255,255,255,0.02);
          }
          .dc-signal-bio .dc-signal-flag { width: 28px; height: 18px; }
          .dc-signal-bio span {
            display: block;
            font-size: 0.6875rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-signal-bio strong {
            display: block;
            font-size: 0.72rem;
            color: #e7f4ff;
            letter-spacing: 0.02em;
          }
          .dc-signal-board-card {
            padding: 8px 10px;
            border: 1px solid rgba(118,200,245,0.28);
            background: linear-gradient(180deg, rgba(43,228,255,0.08), rgba(8,18,28,0.5));
            box-shadow: none;
            display: grid;
            gap: 2px;
            text-align: center;
          }
          .dc-signal-board-card strong {
            font-family: var(--dc-font-number);
            font-size: 1.55rem;
            color: #e8f6ff;
            line-height: 1;
          }
          .dc-signal-board-card span {
            font-size: 0.6875rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #d7e8f6;
          }
          .dc-signal-board-card em {
            font-style: normal;
            font-family: var(--dc-font-mono);
            font-size: 0.6875rem;
            color: var(--dc-cyan);
          }
          .dc-signal-club-card {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 8px;
            align-items: center;
            padding: 8px;
            border: 1px solid rgba(118,200,245,0.16);
            background: rgba(6,16,28,0.7);
          }
          .dc-signal-club-card .dc-signal-flag { width: 34px; height: 22px; }
          .dc-signal-club-card strong,
          .dc-signal-club-card span { display: block; }
          .dc-signal-club-card strong { color: #e8f5ff; font-size: 0.78rem; }
          .dc-signal-club-card span { color: var(--dc-muted); font-size: 0.6875rem; }
          .dc-signal-assign {
            margin-top: auto;
            padding-top: 8px;
            border-top: 1px solid rgba(118,200,245,0.12);
          }
          .dc-signal-assign > strong {
            display: block;
            font-size: 0.78rem;
            color: var(--dc-cyan);
            letter-spacing: 0.06em;
          }
          .dc-signal-core__grid {
            display: grid;
            grid-template-columns: minmax(220px, 0.95fr) minmax(0, 1.05fr);
            gap: 12px;
            min-height: 0;
            flex: 1;
            align-items: stretch;
          }
          .dc-skill-dna {
            min-width: 0;
            display: grid;
            gap: 8px;
            padding: 10px 8px 8px;
            border: 1px solid rgba(118,200,245,0.16);
            background: radial-gradient(circle at 50% 38%, rgba(43,228,255,0.12), transparent 58%), rgba(4,12,22,0.55);
            justify-items: center;
            text-align: center;
          }
          .dc-skill-dna > .dc-profile-tags__label {
            justify-self: start;
            width: 100%;
            text-align: left;
          }
          .dc-skill-dna__stage {
            position: relative;
            width: min(100%, 248px);
            aspect-ratio: 1;
            margin: 0 auto;
          }
          .dc-skill-dna__svg {
            width: 100%;
            height: 100%;
            display: block;
            filter: drop-shadow(0 0 12px rgba(43,228,255,0.25));
          }
          .dc-skill-dna__ring { fill: none; stroke: rgba(118,200,245,0.18); stroke-width: 1; }
          .dc-skill-dna__spoke { stroke: rgba(118,200,245,0.16); stroke-width: 1; }
          .dc-skill-dna__fill {
            fill: rgba(43,228,255,0.18);
            stroke: rgba(43,228,255,0.95);
            stroke-width: 2;
            filter: drop-shadow(0 0 10px rgba(43,228,255,0.45));
          }
          .dc-skill-dna__node { fill: #f4c66e; stroke: #fff; stroke-width: 0.8; }
          .dc-skill-dna__label {
            fill: #9db8cc;
            font-size: 11px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
          }
          .dc-skill-dna__core {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 92px;
            height: 92px;
            margin: 0;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(8,24,40,0.96), rgba(8,20,34,0.72));
            box-shadow: 0 0 28px rgba(43,228,255,0.35), inset 0 0 18px rgba(43,228,255,0.12);
            pointer-events: none;
            box-sizing: border-box;
            padding: 0 4px;
          }
          .dc-skill-dna__core strong {
            font-family: var(--dc-font-number);
            font-size: 1.45rem;
            color: #f4d07a;
            line-height: 1;
            text-shadow: 0 0 16px rgba(244,198,110,0.45);
            display: block;
            width: 100%;
            text-align: center;
            letter-spacing: -0.02em;
          }
          .dc-skill-dna__core strong.is-range {
            font-size: 1.05rem;
            letter-spacing: -0.03em;
          }
          .dc-skill-dna__core span {
            font-size: 0.6875rem;
            letter-spacing: 0.16em;
            color: var(--dc-cyan);
            margin-top: 5px;
            font-weight: 800;
            display: block;
            width: 100%;
            text-align: center;
          }
          .dc-skill-dna__ranges {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 4px;
            width: 100%;
          }
          .dc-skill-dna__ranges > div {
            padding: 5px 4px;
            border: 1px solid rgba(118,200,245,0.12);
            background: rgba(255,255,255,0.02);
            text-align: center;
          }
          .dc-skill-dna__ranges span {
            display: block;
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-skill-dna__ranges strong {
            font-family: var(--dc-font-mono);
            font-size: 0.7rem;
            color: #d9eefc;
          }
          .dc-skill-dna__ranges .is-locked strong { color: #6d8296; }
          .dc-proj-engine {
            min-width: 0;
            display: grid;
            gap: 8px;
            padding: 8px 10px;
            border: 1px solid rgba(118,200,245,0.16);
            background: linear-gradient(180deg, rgba(10,24,38,0.85), rgba(5,12,20,0.7));
          }
          .dc-proj-engine__bar-head {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 3px;
          }
          .dc-proj-engine__bar-head span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-proj-engine__bar-head strong {
            font-family: var(--dc-font-number);
            font-size: 0.92rem;
            color: var(--dc-cyan);
          }
          .dc-proj-engine__bar--gold .dc-proj-engine__bar-head strong { color: var(--dc-gold); }
          .dc-proj-engine__bar--violet .dc-proj-engine__bar-head strong { color: #b8a8ff; }
          .dc-proj-engine__bar--teal .dc-proj-engine__bar-head strong { color: #5fd6c8; }
          .dc-proj-engine__track {
            height: 8px;
            border-radius: 2px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(118,200,245,0.14);
            overflow: hidden;
          }
          .dc-proj-engine__track i {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, rgba(43,228,255,0.45), rgba(43,228,255,0.95));
            box-shadow: 0 0 12px rgba(43,228,255,0.35);
          }
          .dc-proj-engine__bar--gold .dc-proj-engine__track i {
            background: linear-gradient(90deg, rgba(244,198,110,0.45), rgba(244,198,110,0.98));
            box-shadow: 0 0 12px rgba(244,198,110,0.4);
          }
          .dc-proj-engine__bar--violet .dc-proj-engine__track i {
            background: linear-gradient(90deg, rgba(150,120,255,0.45), rgba(184,168,255,0.95));
          }
          .dc-proj-engine__bar--teal .dc-proj-engine__track i {
            background: linear-gradient(90deg, rgba(40,160,150,0.45), rgba(95,214,200,0.95));
          }
          .dc-proj-engine__bar.is-empty .dc-proj-engine__track i { width: 0 !important; }
          .dc-proj-engine__meta {
            display: grid;
            gap: 6px;
            margin-top: 4px;
            padding-top: 8px;
            border-top: 1px solid rgba(118,200,245,0.12);
          }
          .dc-proj-engine__meta > div {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            align-items: baseline;
          }
          .dc-proj-engine__meta span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-proj-engine__meta strong {
            font-size: 0.74rem;
            color: #dff1ff;
            text-align: right;
          }
          .dc-proj-engine__meta strong.is-warn { color: var(--dc-gold); }
          .dc-signal-core__lower {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 10px;
            flex-shrink: 0;
          }
          .dc-field-notes,
          .dc-identity-tags {
            padding: 8px 10px;
            border: 1px solid rgba(118,200,245,0.14);
            background: rgba(255,255,255,0.02);
          }
          .dc-field-notes p {
            margin: 0;
            font-size: 0.74rem;
            line-height: 1.35;
            color: #c5dbea;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }
          .dc-identity-tags__row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .dc-profile-chip--gold {
            color: var(--dc-gold);
            border-color: rgba(244,198,110,0.55);
            background: rgba(244,198,110,0.12);
            box-shadow: 0 0 12px rgba(244,198,110,0.15);
          }
          .dc-profile-chip--violet {
            color: #c4b6ff;
            border-color: rgba(170,140,255,0.45);
            background: rgba(120,90,220,0.14);
          }
          .dc-lens-row {
            display: grid;
            gap: 5px;
            padding: 7px 0;
            border-bottom: 1px solid rgba(118,200,245,0.1);
          }
          .dc-lens-row__copy {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            align-items: center;
          }
          .dc-lens-row__copy span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-lens-row__copy strong {
            font-size: 0.72rem;
            color: #e8f5ff;
            text-align: right;
            letter-spacing: 0.02em;
          }
          .dc-lens-row.is-empty .dc-lens-row__copy strong { color: #6d8296; }
          .dc-segment-dots {
            display: grid;
            grid-template-columns: repeat(10, minmax(0, 1fr));
            gap: 3px;
          }
          .dc-segment-dots i {
            height: 6px;
            border-radius: 1px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(118,200,245,0.12);
          }
          .dc-segment-dots--cyan i.is-on {
            background: linear-gradient(180deg, #6df7ff, #2bb8d0);
            box-shadow: 0 0 8px rgba(43,228,255,0.45);
            border-color: transparent;
          }
          .dc-segment-dots--gold i.is-on {
            background: linear-gradient(180deg, #ffe08a, #f4c66e);
            box-shadow: 0 0 8px rgba(244,198,110,0.5);
            border-color: transparent;
          }
          .dc-segment-dots--violet i.is-on {
            background: linear-gradient(180deg, #d0c0ff, #9b7dff);
            box-shadow: 0 0 8px rgba(155,125,255,0.45);
            border-color: transparent;
          }
          .dc-segment-dots--muted i.is-on {
            background: linear-gradient(180deg, #9db0c0, #7a8fa0);
            box-shadow: none;
            border-color: transparent;
          }
          .dc-lens-note {
            margin: 0;
            font-size: 0.6875rem;
            line-height: 1.3;
            color: #8fa9bd;
          }
          .dc-signal-chart.is-compact .dc-signal-chart__svg { height: 44px; }
          .dc-signal-chart__head strong.is-up { color: var(--dc-green); }
          .dc-signal-chart__head strong.is-down { color: #ff8f9c; }
          .dc-signal-footer--premium {
            grid-template-columns: minmax(240px, 0.95fr) minmax(0, 1.2fr) auto;
            grid-template-rows: auto;
            gap: 10px 14px;
            align-items: stretch;
          }
          .dc-signal-footer--premium .dc-signal-actionbar {
            grid-column: auto;
          }
          .dc-scout-trail {
            min-width: 0;
            display: grid;
            gap: 6px;
          }
          .dc-scout-trail__head {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            align-items: baseline;
          }
          .dc-scout-trail__progress {
            font-size: 0.6875rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--dc-cyan);
          }
          .dc-scout-trail__track {
            display: flex;
            gap: 6px;
            overflow: hidden;
          }
          .dc-scout-trail__node {
            flex: 1;
            min-width: 0;
            display: grid;
            gap: 3px;
            justify-items: center;
            text-align: center;
          }
          .dc-scout-trail__node i {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: rgba(118,200,245,0.25);
            border: 1px solid rgba(118,200,245,0.35);
          }
          .dc-scout-trail__node.is-active i {
            background: #2be4ff;
            box-shadow: 0 0 0 3px rgba(43,228,255,0.25), 0 0 14px rgba(43,228,255,0.55);
          }
          .dc-scout-trail__node span {
            font-size: 0.6875rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--dc-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
          }
          .dc-scout-trail__node strong {
            font-family: var(--dc-font-mono);
            font-size: 0.6875rem;
            color: #d8eeff;
          }
          .dc-scout-trail__bar {
            height: 4px;
            border-radius: 2px;
            background: rgba(255,255,255,0.06);
            overflow: hidden;
          }
          .dc-scout-trail__bar i {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, rgba(43,228,255,0.4), rgba(244,198,110,0.9));
          }
          .dc-scout-trail__conf {
            font-size: 0.6875rem;
            color: var(--dc-muted);
          }
          .dc-scout-trail--empty p {
            margin: 4px 0;
            font-size: 0.7rem;
            color: var(--dc-muted);
          }
          .dc-signal-actionbar {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
          }
          .dc-signal-actionbar .dc-btn {
            width: 100%;
            justify-content: center;
            font-size: 0.6875rem;
            padding: 9px 8px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }
          .dc-btn--gold {
            border-color: rgba(244,198,110,0.7);
            color: #1a1204;
            font-weight: 800;
            background: linear-gradient(180deg, #ffe29a, #f0bf55);
            box-shadow: 0 0 18px rgba(244,198,110,0.35);
          }
          .dc-btn--gold.is-active {
            box-shadow: 0 0 22px rgba(244,198,110,0.55);
          }
          .dc-signal-assign-tray,
          .dc-signal-more {
            grid-column: 1 / -1;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }

          .dc-signal-identity {
            grid-area: identity;
            padding: 12px 12px 10px;
            border-right: 1px solid rgba(118, 200, 245, 0.14);
            display: flex;
            flex-direction: column;
            gap: 7px;
            min-height: 0;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(12,36,58,0.95), rgba(6,18,32,0.72));
          }
          .dc-signal-club-card {
            margin-top: auto;
          }
          .dc-signal-identity .player-headshot { --size: 118px; align-self: center; }
          .dc-signal-identity__rank {
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 8px;
          }
          .dc-signal-identity__rank strong {
            font-family: var(--dc-font-number);
            font-size: 1.35rem;
            color: var(--dc-cyan);
          }
          .dc-signal-identity__name {
            margin: 0;
            text-align: center;
            font-family: var(--dc-font-title);
            font-size: clamp(0.95rem, 1.4vw, 1.2rem);
            letter-spacing: 0.02em;
            line-height: 1.15;
            color: #eef7ff;
          }
          .dc-signal-identity__vitals {
            margin: 0;
            text-align: center;
            font-size: 0.6875rem;
            line-height: 1.35;
            color: #9db8cc;
          }
          .dc-signal-identity__org {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 8px;
            align-items: center;
            padding: 8px;
            border: 1px solid rgba(118, 200, 245, 0.12);
            background: rgba(255,255,255,0.02);
          }
          .dc-signal-identity__flag {
            width: 28px;
            height: 20px;
            object-fit: cover;
            border-radius: 2px;
          }
          .dc-signal-identity__org strong,
          .dc-signal-identity__org span {
            display: block;
            font-size: 0.6875rem;
            line-height: 1.25;
          }
          .dc-signal-identity__org strong { color: #d8eeff; }
          .dc-signal-identity__org span { color: var(--dc-muted); }
          .dc-signal-identity__facts {
            margin: 0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px 8px;
          }
          .dc-signal-identity__facts > div {
            display: flex;
            justify-content: space-between;
            gap: 6px;
            padding: 3px 0;
            border-bottom: 1px solid rgba(118, 200, 245, 0.08);
          }
          .dc-signal-identity__facts dt {
            margin: 0;
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-signal-identity__facts dd {
            margin: 0;
            font-family: var(--dc-font-mono);
            font-size: 0.72rem;
            color: #d8eeff;
          }
          .dc-signal-core {
            grid-area: core;
            padding: 14px 16px 10px;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .dc-signal-core__head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
            flex-shrink: 0;
          }
          .dc-signal-core__head h3 {
            margin: 2px 0 0;
            font-family: var(--dc-font-title);
            font-size: 1.05rem;
            letter-spacing: 0.03em;
            color: #e8f6ff;
          }
          .dc-signal-meters {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            flex-shrink: 0;
          }
          .dc-signal-estimate {
            padding: 8px 10px;
            border: 1px solid rgba(118, 200, 245, 0.16);
            background: linear-gradient(180deg, rgba(12,32,48,0.85), rgba(6,16,28,0.7));
            display: grid;
            gap: 2px;
            align-content: start;
          }
          .dc-signal-estimate strong {
            font-family: var(--dc-font-number);
            font-size: 1.35rem;
            color: var(--dc-cyan);
            line-height: 1.1;
          }
          .dc-signal-estimate small {
            font-size: 0.6875rem;
            color: var(--dc-muted);
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }
          .dc-signal-estimate.is-muted strong { color: #7f93a6; }
          .dc-signal-core .dc-profile-attr-strip {
            margin-top: 0;
            padding-top: 0;
            border-top: 0;
            flex-shrink: 0;
          }
          .dc-signal-core .dc-profile-attr-strip__grid {
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 6px;
          }
          .dc-signal-tags {
            margin-top: 0;
            padding-top: 0;
            border-top: 0;
            flex-shrink: 0;
          }
          .dc-signal-report {
            margin-top: 0;
            padding-top: 8px;
            border-top: 1px solid rgba(118, 200, 245, 0.12);
            flex-shrink: 0;
          }
          .dc-signal-report p {
            margin: 0;
            font-size: 0.74rem;
            line-height: 1.35;
            color: #c2d8ea;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }
          .dc-signal-origin-hook,
          .dc-signal-translation {
            margin: 0;
            font-size: 0.6875rem;
            line-height: 1.3;
            color: #8fa9bd;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }
          .dc-signal-decision {
            grid-area: decision;
            padding: 14px 12px 12px;
            border-left: 1px solid rgba(118, 200, 245, 0.14);
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 0;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(10,22,36,0.95), rgba(6,12,20,0.88));
          }
          .dc-signal-decision__block {
            display: grid;
            gap: 4px;
          }
          .dc-signal-decision__block > strong {
            font-size: 0.86rem;
            color: #dcefff;
            line-height: 1.25;
          }
          .dc-signal-dash { color: #7f93a6 !important; font-weight: 600; }
          .dc-signal-actions {
            margin-top: auto;
            display: grid;
            gap: 6px;
          }
          .dc-signal-actions .dc-btn {
            width: 100%;
            font-size: 0.7rem;
            padding: 7px 8px;
            justify-content: center;
          }
          .dc-signal-decision .dc-profile-assign {
            margin-top: 0;
            max-height: 88px;
            overflow: auto;
          }
          .dc-signal-footer {
            grid-area: footer;
            border-top: 1px solid rgba(118, 200, 245, 0.16);
            padding: 10px 14px 12px;
            display: grid;
            grid-template-columns: minmax(220px, 0.9fr) minmax(0, 1.4fr);
            gap: 12px 16px;
            background: rgba(3, 10, 18, 0.72);
            min-height: 0;
          }
          .dc-signal-chart {
            min-width: 0;
            display: grid;
            gap: 4px;
          }
          .dc-signal-chart__head {
            display: flex;
            align-items: baseline;
            gap: 8px;
          }
          .dc-signal-chart__head strong {
            font-family: var(--dc-font-number);
            color: var(--dc-gold);
            font-size: 0.95rem;
          }
          .dc-signal-chart__delta {
            font-family: var(--dc-font-mono);
            font-size: 0.6875rem;
            color: var(--dc-cyan);
          }
          .dc-signal-chart__svg {
            width: 100%;
            height: 64px;
            display: block;
            background: linear-gradient(180deg, rgba(43,228,255,0.04), transparent);
            border: 1px solid rgba(118, 200, 245, 0.1);
          }
          .dc-signal-chart__axis {
            display: flex;
            justify-content: space-between;
            font-size: 0.6875rem;
            color: var(--dc-muted);
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }
          .dc-signal-chart--empty p {
            margin: 8px 0 0;
            font-size: 0.72rem;
            color: var(--dc-muted);
          }
          .dc-signal-season {
            min-width: 0;
            display: grid;
            gap: 6px;
          }
          .dc-signal-season__head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
          }
          .dc-signal-sample {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #f4b467;
          }
          .dc-signal-metrics {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 6px;
          }
          .dc-signal-metric {
            padding: 6px 7px;
            border: 1px solid rgba(118, 200, 245, 0.12);
            background: rgba(255,255,255,0.02);
            display: grid;
            gap: 2px;
          }
          .dc-signal-metric span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-signal-metric strong {
            font-family: var(--dc-font-mono);
            font-size: 0.78rem;
            color: #d8eeff;
          }
          .dc-signal-metric.is-hot {
            border-color: rgba(43, 228, 255, 0.45);
            background: rgba(43, 228, 255, 0.08);
          }
          .dc-signal-metric.is-hot strong { color: var(--dc-cyan); }
          .dc-signal-metric.is-gem {
            border-color: rgba(244, 198, 110, 0.55);
            background: rgba(244, 198, 110, 0.12);
            box-shadow: 0 0 14px rgba(244, 198, 110, 0.18);
          }
          .dc-signal-metric.is-gem strong { color: var(--dc-gold); }
          .dc-signal-metric.is-cold strong { color: #ff8f9c; }
          .dc-signal-metric.is-empty strong { color: #6d8296; }
          .dc-profile-modal__loading {
            grid-column: 1 / -1;
            grid-row: 1 / -1;
            place-self: center;
            padding: 28px 20px;
            margin: 0;
            text-align: center;
          }
          .dc-profile-zone-head {
            margin: 0;
            font-size: 0.6875rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--dc-cyan);
            font-weight: 700;
          }
          .dc-modal-close {
            position: absolute;
            top: 8px;
            right: 10px;
            z-index: 3;
            width: 32px;
            height: 32px;
            border: 1px solid rgba(118, 200, 245, 0.28);
            border-radius: 8px;
            background: rgba(6, 16, 28, 0.92);
            cursor: pointer;
            display: grid;
            place-items: center;
            color: #e8f4ff;
            padding: 0;
            line-height: 0;
          }
          .dc-modal-close:hover {
            border-color: rgba(43, 228, 255, 0.55);
            background: rgba(10, 26, 42, 0.98);
            color: #fff;
          }
          .dc-modal-close__x {
            display: block;
            width: 14px;
            height: 14px;
            stroke: currentColor;
            stroke-width: 2.2;
            stroke-linecap: round;
            fill: none;
          }
          .dc-profile-hero__trend {
            font-family: var(--dc-font-mono);
            font-size: 0.72rem;
            color: var(--dc-muted);
          }
          .dc-profile-hero__trend.is-rise { color: var(--dc-green); }
          .dc-profile-hero__trend.is-fall { color: #ff8f9c; }
          .dc-profile-hero__flags {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            justify-content: center;
          }
          .dc-profile-fact {
            display: grid;
            grid-template-columns: 130px 1fr;
            gap: 8px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(118, 200, 245, 0.1);
          }
          .dc-profile-fact:last-child { border-bottom: 0; }
          .dc-profile-fact__label {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
            padding-top: 2px;
          }
          .dc-profile-fact__body strong {
            display: block;
            font-family: var(--dc-font-mono);
            font-size: 0.86rem;
            color: #d8eeff;
            line-height: 1.3;
            font-weight: 600;
          }
          .dc-profile-fact__body small {
            display: block;
            margin-top: 2px;
            font-size: 0.6875rem;
            color: var(--dc-cyan);
          }
          .dc-profile-fact__body--inline {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 6px;
          }
          .dc-risk-reason {
            font-style: normal;
            font-size: 0.6875rem;
            color: var(--dc-muted);
            line-height: 1.25;
          }
          /* Projection band reads as a board stencil. */
          .dc-proj-tier {
            display: inline-block;
            padding: 2px 8px;
            border-radius: var(--radius-ops, 2px);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border: 1px solid transparent;
            text-shadow: none;
          }
          .dc-proj-tier--0 { color: #9db1c4; border-color: rgba(157,177,196,0.4); background: rgba(157,177,196,0.1); text-shadow: none; }
          .dc-proj-tier--1 { color: #7fe0c0; border-color: rgba(127,224,192,0.45); background: rgba(127,224,192,0.12); }
          .dc-proj-tier--2 { color: #59e6a6; border-color: rgba(89,230,166,0.5); background: rgba(89,230,166,0.14); }
          .dc-proj-tier--3 { color: #35c8ff; border-color: rgba(53,200,255,0.55); background: rgba(53,200,255,0.16); }
          .dc-proj-tier--4 { color: #7fd8ff; border-color: rgba(127,216,255,0.6); background: rgba(127,216,255,0.16); }
          .dc-proj-tier--5 { color: #ffd24a; border-color: rgba(255,210,74,0.7); background: rgba(255,210,74,0.2); box-shadow: none; }
          .dc-risk-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: var(--radius-ops, 2px);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border: 1px solid transparent;
          }
          .dc-risk-tag--0 { color: #ff7d8b; border-color: rgba(255,125,139,0.55); background: rgba(255,125,139,0.14); }
          .dc-risk-tag--1 { color: #c8d8e6; border-color: rgba(200,216,230,0.4); background: rgba(200,216,230,0.08); }
          .dc-risk-tag--2 { color: #59e6a6; border-color: rgba(89,230,166,0.5); background: rgba(89,230,166,0.14); }
          .dc-risk-tag--3 { color: #ffd24a; border-color: rgba(255,210,74,0.7); background: rgba(255,210,74,0.2); box-shadow: none; }
          .dc-profile-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(118, 200, 245, 0.22);
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 0.6875rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #c8e4f8;
            background: rgba(255,255,255,0.03);
          }
          .dc-profile-chip--accent {
            color: var(--dc-cyan);
            border-color: rgba(43, 228, 255, 0.35);
            background: rgba(43, 228, 255, 0.08);
          }
          .dc-profile-chip--good { color: var(--dc-green); border-color: rgba(108,247,166,0.35); }
          .dc-profile-chip--warn { color: #ffb574; border-color: rgba(255,181,116,0.35); }
          .dc-profile-tags__label {
            display: block;
            font-size: 0.6875rem;
            letter-spacing: 0.1em;
            color: var(--dc-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
          }
          .dc-profile-tags__row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
          }
          .dc-profile-attr-strip__grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 6px;
          }
          .dc-profile-attr-mini {
            padding: 6px 7px;
            border: 1px solid rgba(118, 200, 245, 0.14);
            background: rgba(255,255,255,0.02);
            display: grid;
            gap: 2px;
          }
          .dc-profile-attr-mini span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--dc-muted);
          }
          .dc-profile-attr-mini strong {
            font-family: var(--dc-font-mono);
            font-size: 0.78rem;
            color: #d8eeff;
          }
          .dc-profile-attr-mini.is-locked strong { color: var(--dc-muted); opacity: 0.75; }
          .dc-profile-assign {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
          }
          .dc-profile-meter__head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 4px;
          }
          .dc-profile-meter__head span {
            font-size: 0.6875rem;
            letter-spacing: 0.08em;
            color: var(--dc-muted);
            text-transform: uppercase;
          }
          .dc-profile-meter__head strong {
            font-family: var(--dc-font-number);
            font-size: 0.88rem;
            color: var(--dc-cyan);
          }
          .dc-profile-meter--gold .dc-profile-meter__head strong { color: var(--dc-gold); }
          .dc-profile-meter--violet .dc-profile-meter__head strong { color: #b8a8ff; }
          .dc-profile-meter--amber .dc-profile-meter__head strong { color: #f4b467; }
          .dc-profile-meter__track {
            height: 7px;
            border-radius: 2px;
            background: rgba(255,255,255,0.06);
            overflow: hidden;
            border: 1px solid rgba(118, 200, 245, 0.14);
          }
          .dc-profile-meter__track i {
            display: block;
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, rgba(43,228,255,0.55), rgba(43,228,255,0.95));
          }
          .dc-profile-meter--gold .dc-profile-meter__track i {
            background: linear-gradient(90deg, rgba(244,198,110,0.55), rgba(244,198,110,0.95));
          }
          .dc-profile-meter--violet .dc-profile-meter__track i {
            background: linear-gradient(90deg, rgba(160,140,255,0.55), rgba(184,168,255,0.95));
          }
          .dc-profile-meter--amber .dc-profile-meter__track i {
            background: linear-gradient(90deg, rgba(244,180,103,0.50), rgba(244,180,103,0.92));
          }
          .dc-profile-meter__note {
            margin: 4px 0 0;
            font-size: 0.6875rem;
            line-height: 1.3;
            color: #9db8cc;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }
          @media (max-width: 1180px) {
            .dc-signal-panel {
              width: min(1180px, 97vw);
              grid-template-columns: minmax(150px, 176px) minmax(0, 1fr) minmax(168px, 196px);
            }
            .dc-signal-core .dc-profile-attr-strip__grid {
              grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .dc-signal-metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
          }
          @media (max-width: 980px) {
            .dc-signal-panel {
              height: min(94vh, 900px);
              height: min(94dvh, 900px);
              max-height: 94dvh;
              grid-template-columns: 1fr 1fr;
              grid-template-rows: auto auto minmax(0, 1fr) auto;
              grid-template-areas:
                "banner banner"
                "identity decision"
                "core core"
                "footer footer";
            }
            .dc-signal-core__grid { grid-template-columns: 1fr; }
            .dc-signal-core__lower { grid-template-columns: 1fr; }
            .dc-signal-actionbar { grid-template-columns: 1fr 1fr; }
            .dc-signal-identity { border-right: 0; border-bottom: 1px solid rgba(118,200,245,0.14); }
            .dc-signal-decision { border-left: 0; border-bottom: 1px solid rgba(118,200,245,0.14); }
            .dc-signal-footer { grid-template-columns: 1fr; }
          }
          @media (max-width: 720px) {
            .dc-signal-panel {
              width: 96vw;
              height: 92vh;
              grid-template-columns: 1fr;
              grid-template-areas:
                "banner"
                "identity"
                "core"
                "decision"
                "footer";
              overflow: auto;
            }
            .dc-signal-meters { grid-template-columns: 1fr; }
            .dc-signal-metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          }
          @media (max-height: 780px) {
            .dc-signal-panel { height: 96vh; }
            .dc-signal-identity .player-headshot,
            .dc-signal-portrait .player-headshot { --size: 88px; }
            .dc-signal-chart__svg { height: 40px; }
            .dc-signal-core { gap: 6px; padding: 8px 10px 6px; }
            .dc-signal-meters { gap: 7px; }
            .dc-skill-dna__ranges { display: none; }
            .dc-profile-meter__note { display: none; }
            .dc-signal-banner { padding-top: 5px; padding-bottom: 5px; }
          }

          .prospect-card--transcendent {
            border-color: rgba(244, 198, 110, 0.55) !important;
            box-shadow: 0 0 0 1px rgba(244, 198, 110, 0.25), 0 0 24px rgba(244, 198, 110, 0.18);
          }
          .prospect-modal--transcendent .dc-profile-modal__backdrop {
            background: rgba(8, 6, 2, 0.72);
          }
          .aura-gold {
            border-color: rgba(244, 198, 110, 0.65) !important;
            box-shadow:
              inset 0 0 40px rgba(244, 198, 110, 0.12),
              0 0 0 1px rgba(244, 198, 110, 0.35),
              0 0 48px rgba(244, 198, 110, 0.22);
          }
          @keyframes dc-shake {
            0%, 100% { transform: translateX(0); }
            20% { transform: translateX(-6px); }
            40% { transform: translateX(6px); }
            60% { transform: translateX(-4px); }
            80% { transform: translateX(4px); }
          }
          .shake-on-open { animation: dc-shake 0.55s ease-out; }
          body.dc-transcendent-shake .dc-signal-panel { animation: dc-shake 0.55s ease-out; }
          @media (prefers-reduced-motion: reduce) {
            .shake-on-open, body.dc-transcendent-shake .dc-signal-panel { animation: none !important; }
          }

          @media (max-width: 900px) {
            .dc-leaders-modal__panel { width: 94vw; height: 82vh; }
            .dc-lm-row,
            .dc-lm-row--goalie {
              grid-template-columns: 1fr;
              min-height: 0;
            }
            .dc-lm-row__stats,
            .dc-lm-row__draft {
              width: 100%;
            }
            .dc-lm-row__draft {
              flex-direction: row;
              flex-wrap: wrap;
              align-items: center;
            }
            .dc-profile-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
        `}</style>
      </div>
    </div>
  );
}