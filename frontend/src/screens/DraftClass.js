import React, { useMemo, useState, useEffect, useCallback, useRef } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { api } from "../services/api";
import { getDraftProspectProfile, prefetchDraftProspectProfile } from "../services/franchiseService";
import { formatProspectLeague, formatProspectTeam } from "../events/prospectDevelopment/prospectDevelopmentHelpers";
import { applyProspectLeagueTeamFix } from "../data/prospectLeagueTeams";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { flagApiUrl } from "../utils/countryFlags";
import {
  ProjectionRange,
  StockTrajectory,
  RankChange,
  PerformanceStrip,
  LeagueBadge,
} from "../components/franchise/commandVisuals";
import { BOARD_SOURCES, WORKSPACE_TABS, confMeaning, enrichProspectsForWarRoom, groupByPyramid, rankProspectsForSource, weeklyStockMovers } from "./draftWarRoom";
import { buildProspectDossierModel } from "../components/dossier/prospectDossierModel";
import { ProspectDossier, useDossierTab } from "../components/dossier/ProspectDossierView";
import { resolveOvrBands, buildStubProspectProfile } from "./prospectDossierHelpers";

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



const BOARD_NAV_ITEMS = [
  { key: "rank", label: "All", title: "All prospects", icon: "rank" },
  { key: "forwards", label: "Forwards", title: "Forwards", icon: "fwd" },
  { key: "defensemen", label: "Defense", title: "Defensemen", icon: "dmen" },
  { key: "goalies", label: "Goalies", title: "Goalies", icon: "goalie" },
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

const SCOUT_NAMES = [
  "Mike Brennan", "Sarah Chen", "Erik Lindholm", "Marc Dubois",
  "James Okafor", "Anna Kowalski", "Tyler Morrison", "Lisa Bergstrom",
];
/** Client fallback when backend stats missing — mirrors junior inflation vs pro leagues. */

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
  assign: "/api/franchise/scouting/assign",
  filmStudy: "/api/franchise/scouting/film-study",
});

const COVERAGE_DEPLOY_TARGETS = Object.freeze([
  { kind: "region", id: "north-america", label: "North America" },
  { kind: "region", id: "europe", label: "Europe" },
  { kind: "region", id: "scandinavia", label: "Scandinavia" },
  { kind: "country", id: "canada", label: "Canada" },
  { kind: "country", id: "united-states", label: "United States" },
  { kind: "country", id: "sweden", label: "Sweden" },
  { kind: "country", id: "finland", label: "Finland" },
]);

function slugifyCoverageId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "unknown";
}

function prospectMatchesDeployTarget(player, target) {
  if (!player || !target) return false;
  if (target.kind === "country") {
    const country = slugifyCoverageId(player.country || player.nationality || "");
    return country === slugifyCoverageId(target.id);
  }
  const region = slugifyCoverageId(player.region || regionForCountry(player.country));
  return region === slugifyCoverageId(target.id);
}

function countDeployTargets(prospects, target) {
  return (prospects || []).filter((p) => prospectMatchesDeployTarget(p, target)).length;
}

async function runCoverageDeploy(target, { intensity = "normal" } = {}) {
  const body = {
    action: "region_sweep",
    target_type: target.kind,
    target_id: target.id,
    intensity,
  };
  if (target.kind === "country") {
    body.country_id = target.id;
  }
  const res = await api.post(SCOUTING_ENDPOINTS.focus, body);
  return res?.data || {};
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
    elite_junior_production: "Elite junior production",
    high_junior_production: "High junior production",
    role_adjusted_production: "Role-adjusted production",
    defensive_defenseman_projection: "Defensive D projection",
    offensive_role_underproduction: "Offensive underproduction",
    production_concern: "Production concern",
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

function prospectPublicPct(player) {
  const n = Number(coalesce(
    player?.publicScoutingConfidence,
    player?.public_scouting_confidence,
    player?.scoutingConfidence,
    player?.scouting_confidence,
  ));
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.min(100, Math.max(0, n)));
}

function prospectDedicatedPct(player) {
  const n = Number(coalesce(player?.dedicatedScoutingPct, player?.scouted_percentage, player?.scoutedPercentage));
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.round(Math.min(100, Math.max(0, n)));
}

function prospectEffectivePct(player) {
  return prospectDedicatedPct(player) ?? prospectPublicPct(player);
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
  // "generational" key kept for the gold styling; the word itself is reserved for the
  // backend class tier (one every several drafts), never a raw potential number.
  { min: 94, key: "generational", label: "Franchise ceiling" },
  { min: 90, key: "elite", label: "Elite" },
  { min: 86, key: "high", label: "High upside" },
  { min: 82, key: "strong", label: "Strong upside" },
  { min: 77, key: "moderate", label: "Moderate upside" },
  { min: 72, key: "depth", label: "Depth upside" },
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

  const pct = prospectEffectivePct(player);
  if (pct == null || pct < 15) return blank("Ceiling not scouted");

  const range = player?.potentialRange;
  const score = Number(player?.potentialScore);
  const hasScore = Number.isFinite(score) && score > 0;
  const lo = range?.low != null ? Math.round(Number(range.low)) : NaN;
  const hi = range?.high != null ? Math.round(Number(range.high)) : NaN;
  const hasRange = Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo;

  // The column is a range unless the band has actually closed. Coverage changes
  // how wide the range is, not whether a range is shown.
  if (hasRange && hi - lo <= 1 && hasScore) {
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


function ScoutingFileSplit({ player, compact = false }) {
  const pub = prospectPublicPct(player);
  const you = prospectDedicatedPct(player);
  const effective = you ?? pub;
  const pubFog = prospectConfidenceFogClass(pub);
  const youFog = you != null ? prospectConfidenceFogClass(you) : "dc-conf-fog--blind";
  const conf = confMeaning(effective);
  const title = [
    pub != null ? `Public file ${pub}% — what the hockey world already knows` : "Public file unknown",
    you != null && you > 0 ? `Your file ${you}% — dedicated scouting coverage` : "No dedicated file — assign a scout or deploy regional coverage",
    conf.detail,
  ].join(" · ");

  return (
    <span
      className={`dc-prospect-row__cell dc-prospect-row__conf dc-scout-file-split${compact ? " dc-scout-file-split--compact" : ""}`}
      title={title}
    >
      <span className={`dc-scout-file-split__line is-public ${pubFog}`}>
        <em>PUB</em>
        <strong>{pub != null ? `${pub}%` : "—"}</strong>
      </span>
      <span className={`dc-scout-file-split__line is-yours ${youFog}${you == null ? " is-empty" : ""}`}>
        <em>YOU</em>
        <strong>{you != null ? `${you}%` : "—"}</strong>
      </span>
    </span>
  );
}

function ProspectBoardColumnHeader({ showConsensus }) {
  return (
    <div className="dc-prospect-board__columns" aria-hidden="true">
      <span className="dc-prospect-board__col dc-prospect-board__col--rank">#</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--player">Player</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--pos">Pos</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--league">League</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--pot">Pot</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--proj">Window</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--conf">Pub / You</span>
      <span className="dc-prospect-board__col dc-prospect-board__col--stock">Stock</span>
      {showConsensus ? (
        <span className="dc-prospect-board__col dc-prospect-board__col--pub">Public</span>
      ) : null}
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
  const stage = String(franchiseState?.offseason_stage || "").toLowerCase();
  const draftLive =
    stage === "draft" ||
    franchiseState?.draft?.draft_started ||
    franchiseState?.draft_payload?.draft_started;
  if (draftLive) {
    const sy = Number(franchiseState?.season_year || 0);
    if (sy) return `${sy + 1}-06-27`;
  }
  return (
    franchiseState?.franchise_today_iso ||
    franchiseState?.nhl_today?.iso ||
    franchiseState?.scouting_as_of_iso ||
    franchiseState?.nhl_today?.date_label ||
    franchiseState?.current_date ||
    franchiseState?.currentDate ||
    franchiseState?.calendar?.current_date ||
    ""
  );
}

function formatStatsThroughLabel(raw, month, periodLabel) {
  if (!raw) {
    if (month >= 4 && month <= 6) return "End of Season";
    return periodLabel?.replace(/ Draft Board$/i, "") || "Early Season";
  }
  if (/^\d{4}-\d{2}-\d{2}/.test(String(raw))) {
    if (month >= 4 && month <= 6) return "End of Season";
    if (month >= 9 && month <= 10) return "Early Season";
    if (month === 11) return "November";
    if (month === 12 || month === 1) return "Midseason";
    if (month === 2 || month === 3) return "Late Season";
  }
  return String(raw);
}

function isDraftLive(franchiseState) {
  const stage = String(franchiseState?.offseason_stage || "").toLowerCase();
  return (
    stage === "draft" ||
    franchiseState?.draft?.draft_started ||
    franchiseState?.draft_payload?.draft_started ||
    franchiseState?.draft_class_hud?.events?.draft?.display === "LIVE"
  );
}

function draftHudCalendarStale(franchiseState, hud = franchiseState?.draft_class_hud) {
  if (!hud || typeof hud !== "object") return false;
  const liveIso = String(
    franchiseState?.scouting_as_of_iso
    || franchiseState?.franchise_today_iso
    || franchiseState?.nhl_today?.iso
    || ""
  );
  if (!liveIso) return false;
  const hudIso = String(hud.scouting_as_of_iso || hud.franchise_today_iso || "");
  if (hudIso && hudIso !== liveIso) return true;
  const hudCursor = Number(hud.calendar_cursor ?? NaN);
  const liveCursor = Number(franchiseState?.calendar_cursor ?? NaN);
  return Number.isFinite(hudCursor) && Number.isFinite(liveCursor) && hudCursor !== liveCursor;
}

function resolveHudEventDisplay(franchiseState, hud, { eventKey, hudRow, draftLive = false } = {}) {
  if (draftLive) return "LIVE";
  if (!draftHudCalendarStale(franchiseState, hud) && hudRow) {
    if (hudRow.display) return hudRow.display;
    const formatted = formatDaysUntil(hudRow.days_until);
    if (formatted && formatted !== "—") return formatted;
  }
  return inferEventDaysFallback(franchiseState, eventKey);
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




function buildDateContext(franchiseState) {
  const raw = getFranchiseDateRaw(franchiseState);
  const month = parseMonthFromDate(raw);
  const monthMax = getMaxCompletionForMonth(month);
  const [gpMin, gpMax] = getGPRangeForMonth(month);
  const seasonYear = Number(franchiseState?.season_year) || new Date().getFullYear();
  const periodLabel = getScoutingPeriodLabel(month);
  return {
    raw,
    month,
    monthMax,
    gpMin,
    gpMax,
    statsThrough: formatStatsThroughLabel(raw, month, periodLabel),
    periodLabel,
    draftYear: seasonYear + 1,
    isPartialSeason: !month || month < 4 || month > 6,
    draftLive: isDraftLive(franchiseState),
  };
}

function pickTeamName(league, seed) {
  const teams = LEAGUE_TEAMS[league] || LEAGUE_TEAMS.OHL;
  return teams[seededNumber(seed + 99) % teams.length];
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

async function runProspectScoutAction(prospectId, action, extra = {}) {
  const endpoints = {
    film_study: SCOUTING_ENDPOINTS.filmStudy,
    player_focus: SCOUTING_ENDPOINTS.focus,
    assign: SCOUTING_ENDPOINTS.assign,
  };
  const path = endpoints[action] || SCOUTING_ENDPOINTS.focus;
  const res = await api.post(path, {
    prospect_id: prospectId,
    player_id: prospectId,
    target_id: prospectId,
    target_type: "player",
    action,
    ...extra,
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

function characterScoreFromRow(row, base) {
  const chapters = row?.chapter_profile?.chapters;
  if (chapters?.character != null && Number.isFinite(Number(chapters.character))) {
    return Math.round(Number(chapters.character));
  }
  if (row?.character_score != null && Number.isFinite(Number(row.character_score))) {
    return Math.round(Number(row.character_score));
  }
  if (row?.characterScore != null && Number.isFinite(Number(row.characterScore))) {
    return Math.round(Number(row.characterScore));
  }
  const mental = chapterScoreFromRow(row, "mental", null);
  if (mental != null) return mental;
  return null;
}

function chapterScoreFromRow(row, key, fallback = null) {
  const chapters = row?.chapter_profile?.chapters;
  if (chapters?.[key] != null && Number.isFinite(Number(chapters[key]))) {
    return Math.round(Number(chapters[key]));
  }
  const flat = row?.[key];
  if (flat != null && Number.isFinite(Number(flat))) {
    return Math.round(Number(flat));
  }
  return fallback;
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
    String(row?.trend || "").toUpperCase();
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
    const ambientRaw = coalesce(row?.public_scouting_confidence, row?.scouting_confidence, row?.scout_grade, base.completion);
    const scoutedNum = scoutedRaw != null && scoutedRaw !== "" ? Number(scoutedRaw) : null;
    const ambientNum = ambientRaw != null && ambientRaw !== "" ? Number(ambientRaw) : null;
    const dedicatedScoutingPct = Number.isFinite(scoutedNum) && scoutedNum > 0
      ? Math.round(Math.min(100, Math.max(0, scoutedNum)))
      : null;
    const publicScoutingConfidence = Number.isFinite(ambientNum)
      ? Math.round(Math.min(100, Math.max(0, ambientNum)))
      : null;
    const completion = dedicatedScoutingPct ?? publicScoutingConfidence ?? (Number(base.completion) || 40);
    // Only YOUR dedicated file reveals the real OVR (it fell back to league-wide
    // coverage, so well-known prospects you never scouted showed an exact OVR).
    const ovrRevealed = false;
    const rawRange = row?.ovr_range;
    let ovrRange = rawRange && rawRange.low != null && rawRange.high != null
      ? { low: Number(rawRange.low), high: Number(rawRange.high) }
      : null;
    if (!ovrRange) {
      const cur = row?.current_ovr_range;
      if (Array.isArray(cur) && cur.length >= 2) {
        ovrRange = { low: Number(cur[0]), high: Number(cur[1]) };
      } else if (cur && cur.low != null && cur.high != null) {
        ovrRange = { low: Number(cur.low), high: Number(cur.high) };
      } else if (row?.public_ovr_low != null && row?.public_ovr_high != null) {
        ovrRange = { low: Number(row.public_ovr_low), high: Number(row.public_ovr_high) };
      }
    }

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
      dedicatedScoutingPct,
      publicScoutingConfidence,
      stock,
      stockLabel: coalesce(row?.stock_label, row?.stockLabel),
      stockReason: coalesce(row?.stock_reason, row?.stockReason),
      stockDirection: coalesce(row?.stock_direction, row?.stockDirection),
      characterConcerns: Boolean(row?.character_concerns ?? row?.characterConcerns),
      scoutingConfidence: Number(coalesce(row?.scouting_confidence, row?.scoutingConfidence)) || null,
      intelLabel: coalesce(row?.intel_label, row?.intelLabel),
      isTranscendent: Boolean(row?.is_transcendent ?? row?.isTranscendent ?? row?.transcendent_talent),
      classTier: row?.class_tier || row?.classTier || null,
      classIndicators: row?.class_indicators || row?.classIndicators || null,
      hometown: coalesce(row?.hometown, row?.birth_city, row?.birthCity),
      birthday: coalesce(row?.birthday, base.birthday),
      birthCity: coalesce(row?.hometown, row?.birth_city, row?.birthCity, country),
      height: coalesce(row?.height, row?.height_display, base.height),
      heightCm: row?.height_cm != null ? Number(row.height_cm) : (row?.heightCm != null ? Number(row.heightCm) : null),
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
      potentialTrend: coalesce(row?.potential_trend, row?.potentialTrend),
      potentialTrendLabel: coalesce(row?.potential_trend_label, row?.potentialTrendLabel),
      potentialChangeEst: row?.potential_change_est != null ? Number(row.potential_change_est) : null,
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
      ovrHint: null,
      ovrRange,
      ovrRevealed,
      scoutTags: Array.isArray(row?.scout_tags) ? row.scout_tags : (row?.auto_scout?.tags || []),
      scoutProjection: row?.scout_projection || row?.auto_scout?.projection || "",
      scoutConfidence: row?.scout_confidence || row?.auto_scout?.confidence || "",
      scoutSummary: row?.scout_summary || row?.auto_scout?.summary || "",
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
      glove: attrFromBackend(row, base, "glove", "glove"),
      blocker: attrFromBackend(row, base, "blocker", "blocker"),
      reflexes: attrFromBackend(row, base, "reflexes", "reflexes"),
      reboundControl: attrFromBackend(row, base, "rebound_control", "reboundControl"),
      positioning: attrFromBackend(row, base, "positioning", "positioning"),
      puckHandling: attrFromBackend(row, base, "puck_handling", "puckHandling"),
      athleticism: attrFromBackend(row, base, "athleticism", "athleticism"),
      morale: base.morale,
      character: characterScoreFromRow(row, base),
      characterScore: characterScoreFromRow(row, base),
      chapterProfile: row?.chapter_profile || null,
      competitiveness: attrFromBackend(row, base, "competitiveness", "competitiveness"),
      sociability: attrFromBackend(row, base, "sociability", "sociability"),
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

function leaderPctFromSources(sources, keys) {
  for (const key of keys) {
    for (const src of sources) {
      if (!src || typeof src !== "object") continue;
      if (src[key] != null && src[key] !== "") return leaderPct(src[key]);
    }
  }
  return null;
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
  const xgf_pct = leaderPctFromSources(sources, ["xgf_pct", "xGF_pct", "expected_goals_pct"]);
  let cf_pct = leaderPctFromSources(sources, ["cf_pct", "corsi_pct", "corsi_percentage"]);
  // Avoid mirroring xGF% into CF% when Corsi was never supplied on the payload.
  if (cf_pct != null && xgf_pct != null && cf_pct === xgf_pct) {
    const hasCf = sources.some((src) => src && (
      src.cf_pct != null || src.corsi_pct != null || src.corsi_percentage != null
    ));
    if (!hasCf) cf_pct = null;
  }
  return {
    xgf_pct,
    cf_pct,
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




function getScoutingMeta(store, id) {
  return store[id] || EMPTY_SCOUTING_META;
}



/** CSS-only fallback avatar — PlayerHeadshot.js → PlayerHeadshot.css (not styles/playerHeadshot.css). */
function DraftClassHeadshot({ player, size = "md", board = false }) {
  const seed = String(player?.id || player?.name || "");
  let h = 2166136261;
  for (let i = 0; i < seed.length; i += 1) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const roll = Math.abs(h >>> 0) % 100;
  const rank = Number(player?.rank || player?.scoutRank || 99);
  const bloom = Boolean(player?.isTranscendent) || rank <= 5 || roll < 28;
  const zoom = !bloom && (rank <= 16 || roll < 58);
  const fx = bloom ? "bloom" : zoom ? "zoom" : "crop";
  return (
    <PlayerHeadshot
      player={player}
      size={size}
      variant={board ? "circle" : "card"}
      className={`dc-shared-headshot${board ? " dc-board-headshot" : ""} dc-headshot--${fx}`}
      draftState="eligible"
    />
  );
}

function TopHeader({ onBack, dateContext, franchiseState }) {
  const logoUrl = resolveTeamLogo(franchiseState);
  const initials = resolveTeamToken(franchiseState);
  const record = resolveRecord(franchiseState);
  const ownedPickCount = resolveOwnedPickCount(franchiseState, dateContext?.draftYear);
  const hud = franchiseState?.draft_class_hud || {};
  const draftLive = dateContext?.draftLive || isDraftLive(franchiseState);
  const draftValue = resolveHudEventDisplay(franchiseState, hud, {
    eventKey: "draft",
    hudRow: hud?.events?.draft,
    draftLive,
  });
  const coverage = hud?.coverage_label || hud?.scouting_coverage || dateContext?.periodLabel;
  return (
    <header className="dc-topbar dc-topbar--draft-hud dc-topbar--war">
      <div className="dc-topbar__left">
        <button type="button" className="dc-back-btn dc-back-btn--hud" onClick={onBack}>
          ← Back
        </button>
        <div>
          <small className="dc-record-mini">Scouting War Room</small>
          <div className="dc-war-meta">
            <span>Class <strong>{dateContext?.draftYear || "—"}</strong></span>
            <span>Picks <strong>{ownedPickCount ?? 0}</strong></span>
            <span>Record {record}</span>
          </div>
        </div>
      </div>

      <div className="dc-topbar__center dc-topbar__center-logo">
        {logoUrl ? (
          <img className="dc-team-logo" src={logoUrl} alt="Team logo" />
        ) : (
          <div className="dc-team-logo-fallback" aria-label="Team logo">{initials}</div>
        )}
        <small className="dc-date-mini">{coverage || dateContext?.statsThrough || "—"}</small>
      </div>

      <div className="dc-war-countdown" title="Days until the NHL Draft">
        <b>{draftValue || "—"}</b>
        <small>Until Draft</small>
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
  const nowRaw =
    franchiseState?.scouting_as_of_iso ||
    franchiseState?.franchise_today_iso ||
    franchiseState?.nhl_today?.iso ||
    franchiseState?.current_date ||
    "";
  const now = parseIsoDate(nowRaw);
  const markers = Array.isArray(franchiseState?.season_anchor_events) ? franchiseState.season_anchor_events : [];
  const row = markers.find((m) => String(m?.key || "") === eventKey);
  const event = parseIsoDate(row?.date);
  if (!now || !event) return "—";
  if (eventKey === "draft" && isDraftLive(franchiseState)) return "LIVE";
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
  const wjcValue = resolveHudEventDisplay(franchiseState, hud, {
    eventKey: "wjc_start",
    hudRow: wjcEvent,
  });
  const wjcClickable = typeof onOpenWjc === "function" && wjcValue && wjcValue !== "—";
  const lotValue = resolveHudEventDisplay(franchiseState, hud, {
    eventKey: "draft_lottery",
    hudRow: events?.lottery,
  });

  // The draft countdown lives in the header; this strip carries the other dates.
  const items = [
    { key: "status", icon: statusIconForKey(teamStatus?.key), label: "Team status", value: teamStatus?.label || "Middling", clickable: false },
    { key: "lottery", icon: "◌", label: "Draft lottery", value: lotValue || "—", clickable: false },
    { key: "wjc", icon: "◎", label: "World Juniors", value: wjcValue || "—", clickable: wjcClickable },
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





function wrNumClassForRank(rank) {
  const n = Number(rank);
  if (n <= 5) return "wr-num-gold";
  if (n <= 32) return "wr-num-cyan";
  return "wr-num-slate";
}

function wrNumClassForTier(tierKey) {
  if (tierKey === "generational" || tierKey === "elite") return "wr-num-gold";
  if (tierKey === "high" || tierKey === "strong") return "wr-num-cyan";
  return "wr-num-slate";
}


/** Current ability estimate for the board column. Ceiling fog does not blank present skill. */
function resolveCurrentEstimate(player, profile, ceilingHidden, dedicatedFile) {
  const exact = Number(profile?.scoutedOverall ?? profile?.currentOvrEstimate ?? player?.ovrHint);
  const revealed = false;
  if (revealed && (!ceilingHidden || dedicatedFile) && Number.isFinite(exact) && exact > 0) {
    return { text: String(Math.round(exact)), detail: "Scouted OVR", exact: true };
  }
  const low = Number(profile?.overallRangeLow ?? player?.ovrRange?.low);
  const high = Number(profile?.overallRangeHigh ?? player?.ovrRange?.high);
  if (Number.isFinite(low) && Number.isFinite(high) && low > 0 && high >= low) {
    const dash = "\u2013";
    return { text: `${Math.round(low)}${dash}${Math.round(high)}`, detail: ceilingHidden ? "Present ability" : "Range", exact: false };
  }
  if (Number.isFinite(exact) && exact > 0 && !ceilingHidden) {
    return { text: `~${Math.round(exact)}`, detail: "Estimate", exact: false };
  }
  return { text: "\u2014", detail: "Not revealed", exact: false };
}

function ProspectBoardRow({ player, index, selected, expanded, onSelect, onPrefetch, meta, showConsensus }) {
  const rank = player.boardRank || prospectRank(player, index);
  const countryCode = normalizeCountryCode(player);
  const scoutPct = prospectEffectivePct(player);
  const dedicatedPct = prospectDedicatedPct(player);
  prospectConfidenceFogClass(scoutPct);
  const ceilingHidden = Boolean(player?.ceilingHidden);
  const profile = player?.profile || null;
  const ovr = resolveCurrentEstimate(player, profile, ceilingHidden, Boolean(profile?.dedicatedScoutFile));
  const ovrBands = resolveOvrBands(profile, player, ceilingHidden);
  const pot = resolvePotentialEstimate(player);
  const league = player.leagueDisplay || player.league || "—";
  const pos = player.position || "—";
  const nationFull = player.country || player.nationality || countryCode || "";
  const rowFlagUrl = flagApiUrl(countryCode || nationFull, 32);
  const stock = player.draftStock || {};
  const delta = Number(stock.deltaRank) || 0;
  const topFive = rank <= 5;
  const boardOvrText = ovr.text !== "—" ? ovr.text : "—";
  const boardOvrTitle = ovrBands.peakText && !ceilingHidden && ovr.text !== "—"
    ? `Now ${ovr.text} · Peak ${ovrBands.peakText}`
    : (ovr.detail || "Present ability");
  const lo = pot.exact ? pot.value : (player?.potentialRange?.low ?? null);
  const hi = pot.exact ? pot.value : (player?.potentialRange?.high ?? null);
  const mid = pot.value;
  confMeaning(scoutPct);
  const tier = player.pyramidTier;

  const reasonCode = Array.isArray(player.draftRankReasonCodes) ? player.draftRankReasonCodes[0] : null;
  const reasonLabel = humanizeScoutReason(reasonCode);
  const divergence = player.boardDivergence;
  const ceilingFog = player.ceilingHidden
    ? "hidden"
    : (pot.exact ? "exact" : ((dedicatedPct == null || dedicatedPct < 76) && String(pot.text || "").includes("–") ? "range-fog" : "plain"));
  const rankNumClass = wrNumClassForRank(rank);
  const potNumClass = wrNumClassForTier(pot.tier?.key);
  const glowCeiling = pot.exact && (pot.tier?.key === "elite" || pot.tier?.key === "generational" || rank <= 5);

  return (
    <button
      type="button"
      className={`dc-prospect-row${selected ? " is-selected" : ""}${expanded ? " is-expanded" : ""}${meta?.doNotDraft ? " is-dnd" : ""}${player?.isTranscendent ? " prospect-card--transcendent" : ""}${rank > 32 ? " is-late-round" : ""}${topFive ? " is-top5" : ""}${player.overager ? " is-overager" : ""}`}
      onClick={onSelect}
      onMouseEnter={() => onPrefetch?.(player)}
      style={tier ? { "--tier-color": tier.color } : undefined}
    >
      <div className={`dc-prospect-row__rank ${rankNumClass}`} aria-label={`Rank ${rank}`}>
        <RankChange rank={rank} delta={delta} />
      </div>

      <div className="dc-prospect-row__player">
        <div className="dc-prospect-row__shot">
          <DraftClassHeadshot player={player} size={topFive ? "lg" : "md"} board />
        </div>
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
        <span className="dc-prospect-row__identity">
          <span className="dc-prospect-row__name-row">
            <strong className="dc-prospect-row__name">{player.firstName} {player.lastName}</strong>
            {topFive ? (
              <b
                className={`dc-prospect-row__ovr ${(ovr.exact || (ovrBands.peakText && !ceilingHidden)) ? "wr-num-gold" : "wr-num-slate"}`}
                title={boardOvrTitle}
              >
                {boardOvrText}
              </b>
            ) : null}
          </span>
          <span className="dc-prospect-row__sub">
            {player.teamDisplay || player.team || league} · {player.age || "—"} yrs
            {player.scoutProjection ? ` · ${player.scoutProjection}` : ""}
            {player.scoutConfidence ? ` · ${player.scoutConfidence} confidence` : ""}
            {Array.isArray(player.scoutTags) && player.scoutTags.length ? ` · ${player.scoutTags.slice(0, 2).join(", ")}` : ""}
            {player.overager ? <em className="dc-overage-pill">Overager</em> : null}
            {player.characterConcerns ? <em className="dc-char-pill">Character</em> : null}
            {reasonLabel ? <em className="wr-reason-chip">{reasonLabel}</em> : null}
            {divergence ? (
              <em className="wr-div-chip">YOU #{divergence.scoutRank} · PUBLIC #{divergence.publicRank}</em>
            ) : null}
            {player.goalieBoardCap ? <em className="wr-cap-chip">Goalie board cap</em> : null}
          </span>
        </span>
      </div>

      <span className="dc-prospect-row__cell dc-prospect-row__pos">{pos}</span>
      <span className="dc-prospect-row__cell dc-prospect-row__league" title={league}>
        <LeagueBadge league={league} />
      </span>
      <span
        className={`dc-prospect-row__cell dc-prospect-row__pot${pot.tier ? ` is-pot-${pot.tier.key}` : ""}${pot.exact ? "" : " is-range"}`}
        title={`${pot.tier ? `${pot.tier.label} · ` : ""}${pot.detail} · Now ${ovr.text}`}
      >
        <span
          className={`wr-ceiling-chip${ceilingFog === "hidden" ? " is-hidden" : ""}${ceilingFog === "range-fog" ? " is-range-fog" : ""}${ceilingFog === "exact" ? " is-exact" : ""}${glowCeiling ? " is-glow" : ""}`}
          aria-label={ceilingFog === "hidden" ? "Ceiling unreported" : undefined}
        >
          {ceilingFog === "hidden" ? null : (
            <ProjectionRange
              low={lo}
              mid={mid}
              high={hi}
              confidence={scoutPct}
              className={pot.exact ? potNumClass : ""}
            />
          )}
        </span>
        {ceilingFog !== "hidden" && (player.potentialTrend === "rising" || player.potentialTrend === "falling") ? (
          <span
            className={`dc-pot-trend is-${player.potentialTrend}`}
            title={player.potentialTrendLabel || (player.potentialTrend === "rising" ? "Scouts see the ceiling rising" : "Scouts see the ceiling slipping")}
          >
            {player.potentialTrend === "rising" ? "▲" : "▼"}
            {player.potentialChangeEst != null ? Math.abs(player.potentialChangeEst) : ""}
          </span>
        ) : null}
      </span>
      <span className="dc-prospect-row__cell dc-prospect-row__proj">{projectionForRank(rank)}</span>
      <ScoutingFileSplit player={player} compact />
      <span className="dc-prospect-row__cell dc-prospect-row__stock" title="Weekly stock">
        <StockTrajectory delta={delta} direction={stock.direction} />
      </span>
      {showConsensus ? (
        <span className="dc-prospect-row__cell dc-prospect-row__pub">#{player.publicRank || player.rank}</span>
      ) : null}
      {expanded ? (
        <div className="dc-mini-dossier">
          <PerformanceStrip player={player} />
        </div>
      ) : null}
    </button>
  );
}

function CoverageDeployPanel({ prospects, activeDeployments, onDeploy, busy, notice }) {
  const [open, setOpen] = useState(false);
  const activeKeys = new Set(
    (activeDeployments || []).map((d) => `${d.kind}:${slugifyCoverageId(d.id)}`)
  );

  return (
    <div className="dc-coverage-deploy">
      <button
        type="button"
        className={`dc-coverage-deploy__toggle${open ? " is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        title="Deploy regional scouting coverage — passive daily file growth plus an instant sweep"
      >
        Deploy coverage {activeDeployments?.length ? `(${activeDeployments.length})` : ""}
      </button>
      {notice ? <span className="dc-coverage-deploy__notice">{notice}</span> : null}
      {open ? (
        <div className="dc-coverage-deploy__menu">
          <p className="dc-coverage-deploy__hint">
            Sweep a region now (+6% each) and keep scouts on passive coverage. Max 4 active deployments.
            Assign scouts to your shortlist for passive daily growth — full file after 14 days assigned.
          </p>
          {COVERAGE_DEPLOY_TARGETS.map((target) => {
            const count = countDeployTargets(prospects, target);
            const active = activeKeys.has(`${target.kind}:${slugifyCoverageId(target.id)}`);
            const sweepTotal = count * 6;
            return (
              <button
                key={`${target.kind}-${target.id}`}
                type="button"
                className={`dc-coverage-deploy__opt${active ? " is-active" : ""}`}
                disabled={busy || count <= 0}
                onClick={() => onDeploy(target)}
                title={`Deploy coverage — ${count} prospect${count === 1 ? "" : "s"} · +6% each (${sweepTotal}% total)`}
              >
                <strong>{target.label}</strong>
                <span>
                  {count} on board · +6% ea ({sweepTotal}% total){active ? " · active" : ""}
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ProspectBoardPanel({
  prospects,
  selectedProspectId,
  onOpenProspect,
  onPrefetchProspect,
  scoutingStore,
  activeBoardView,
  boardSource,
  setBoardSource,
  showConsensus,
  setShowConsensus,
  pyramidMode,
  setPyramidMode,
  activeDeployments,
  onDeployCoverage,
  deployBusy,
  deployNotice,
}) {
  const filterLabel = boardFilterLabel(activeBoardView);
  const groups = pyramidMode ? groupByPyramid(prospects) : [{ key: "rank", prospects, color: null, label: null, rarity: null }];
  const renderRow = (player, index) => (
    <ProspectBoardRow
      key={player.id}
      player={player}
      index={index}
      selected={player.id === selectedProspectId}
      expanded={player.id === selectedProspectId}
      onSelect={() => onOpenProspect(player)}
      onPrefetch={onPrefetchProspect}
      meta={getScoutingMeta(scoutingStore, player.id)}
      showConsensus={showConsensus}
    />
  );

  return (
    <section className={`dc-prospect-board${showConsensus ? " is-consensus" : ""}`}>
      <header className="dc-prospect-board__head">
        <p className="dc-public-intel-note" title="Dedicated YOU file reopens ceiling fog on later picks">
          Public ceiling intel clear through pick ~45 · later picks need Deploy coverage or assigned scouts
        </p>
        <span>
          {prospects.length} prospect{prospects.length === 1 ? "" : "s"}
          {filterLabel ? ` · ${filterLabel}` : ""}
        </span>
      </header>
      <div className="dc-board-toolbar">
        <div className="dc-board-sources">
          {BOARD_SOURCES.map((src) => (
            <button
              key={src.id}
              type="button"
              className={boardSource === src.id ? "is-active" : ""}
              onClick={() => setBoardSource(src.id)}
            >
              {src.label}
            </button>
          ))}
        </div>
        <div className="dc-board-filters">
          <CoverageDeployPanel
            prospects={prospects}
            activeDeployments={activeDeployments}
            onDeploy={onDeployCoverage}
            busy={deployBusy}
            notice={deployNotice}
          />
          <button
            type="button"
            className={`dc-consensus-chip${pyramidMode ? " is-on" : ""}`}
            onClick={() => setPyramidMode((v) => !v)}
            aria-pressed={pyramidMode}
            title="Group the board into pyramid tiers. Off = strict rank order."
          >
            Pyramid tiers
          </button>
          <button
            type="button"
            className={`dc-consensus-chip${showConsensus ? " is-on" : ""}`}
            onClick={() => setShowConsensus((v) => !v)}
            aria-pressed={showConsensus}
            title="Public consensus is a reference filter, not the GM board"
          >
            Public consensus
          </button>
        </div>
      </div>
      <div className="dc-prospect-board__list dc-scroll-surface">
        <ProspectBoardColumnHeader showConsensus={showConsensus} />
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
        ) : pyramidMode ? (
          groups.map((group) => (
            <div key={group.key} className="dc-tier-group">
              <div className="dc-tier-band" style={{ "--tier-color": group.color }}>
                <strong>{group.label}</strong>
                <span>{group.rarity} · {group.prospects.length}</span>
              </div>
              {group.prospects.map((player, index) => renderRow(player, index))}
            </div>
          ))
        ) : (
          prospects.map((player, index) => renderRow(player, index))
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

function ProspectProfileModal({
  player,
  profile,
  profileLoading = false,
  profileEnriching = false,
  meta,
  franchiseState,
  allProspects = [],
  onClose,
  compareIds = [],
  onToggleWatchlist,
  onToggleTarget,
  onToggleDND,
  onToggleCompare,
  onAssignScout,
  onRunScoutAction,
  scoutActionBusy = false,
  scoutActionNote = "",
}) {
  const [dossierTab, setDossierTab] = useDossierTab();
  const [showAssignPanel, setShowAssignPanel] = useState(false);

  useEffect(() => {
    setShowAssignPanel(false);
  }, [player?.id]);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const prevHtml = document.documentElement.style.overflow;
    const prevBody = document.body.style.overflow;
    document.documentElement.style.overflow = "hidden";
    document.body.style.overflow = "hidden";
    return () => {
      document.documentElement.style.overflow = prevHtml;
      document.body.style.overflow = prevBody;
    };
  }, []);

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

  if (profileLoading && !profile) {
    return (
      <div
        className="dc-profile-modal dc-profile-modal--prospect dc-profile-modal--loading"
        role="dialog"
        aria-modal="true"
        aria-busy="true"
        aria-label={`Loading ${player.firstName} ${player.lastName} scouting profile`}
      >
        <button type="button" className="dc-profile-modal__backdrop" onClick={onClose} aria-label="Close" />
        <article className="dc-signal-panel dc-signal-panel--premium dc-signal-panel--dossier dc-dossier-shell dc-dossier-shell--loading">
          <ModalCloseButton onClick={onClose} label="Close prospect profile" />
          <div className="dc-dossier-loading">
            <DraftClassHeadshot player={player} size="lg" />
            <strong>{player.firstName} {player.lastName}</strong>
            <p>Opening scout file…</p>
          </div>
        </article>
      </div>
    );
  }

  const scoutMeta = meta || EMPTY_SCOUTING_META;
  const model = buildProspectDossierModel(player, profile || {}, {
    allProspects,
    userScoutPct: Number(profile?.userScoutPct ?? player?.dedicatedScoutingPct ?? player?.scoutedPercentage ?? 0),
  });
  const filmReports = scoutMeta.requestedReports || {};
  const filmDone = Boolean(filmReports.film_study || filmReports.filmStudy);
  const filmReady = Boolean(scoutMeta.watchlist || scoutMeta.assignedScout);
  const countryLabel = model.identity.nation || countryDisplayLabel(player, profile);
  const actions = [
    {
      id: "scout",
      label: "Scout player",
      hint: scoutActionBusy ? "Scouting…" : "One-off visit",
      disabled: scoutActionBusy || !onRunScoutAction,
      onClick: () => onRunScoutAction?.("player_focus"),
    },
    {
      id: "film",
      label: "Film study",
      hint: filmDone ? "Film study · done" : filmReady ? "1 per prospect" : "Shortlist or assign first",
      title: filmDone ? "Film study · done" : filmReady ? "4-hour film study" : "Shortlist or assign a scout first",
      disabled: scoutActionBusy || !onRunScoutAction || !filmReady || filmDone,
      onClick: () => onRunScoutAction?.("film_study"),
    },
    {
      id: "assign",
      label: scoutMeta.assignedScout ? `Scout: ${scoutMeta.assignedScout}` : "Assign scout",
      hint: "On assignment · file +~2%/day",
      primary: true,
      disabled: scoutActionBusy || !onAssignScout,
      onClick: () => setShowAssignPanel((v) => !v),
    },
    {
      id: "short",
      label: scoutMeta.watchlist ? "Shortlisted" : "Shortlist",
      hint: scoutMeta.watchlist ? "On your board" : "Pin to your board",
      disabled: !onToggleWatchlist,
      onClick: onToggleWatchlist,
    },
  ];

  return (
    <div
      className={`dc-profile-modal dc-profile-modal--prospect dc-profile-modal--file-open${isTranscendent ? " prospect-modal--transcendent" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-label={`${player.firstName} ${player.lastName} scouting profile`}
    >
      <button type="button" className="dc-profile-modal__backdrop" onClick={onClose} aria-label="Close" />
      <article className={`dc-signal-panel dc-signal-panel--premium dc-signal-panel--dossier dc-dossier-shell${isTranscendent ? " aura-gold shake-on-open" : ""}`}>
        <ModalCloseButton onClick={onClose} label="Close prospect profile" />
        {profileEnriching ? (
          <p className="dc-dossier-enriching" aria-live="polite">Syncing full scout file…</p>
        ) : null}
        {!profile ? (
          <p className="dc-empty-note dc-profile-modal__loading">Scouting profile loading…</p>
        ) : (
          <ProspectDossier
            model={model}
            portrait={(
              <>
                <DraftClassHeadshot player={player} size="lg" />
                <SignalNationFlag country={countryLabel} size={48} className="dd-flag" />
              </>
            )}
            tab={dossierTab}
            onTab={setDossierTab}
            actions={actions}
            statusNote={scoutActionNote}
            assignPanel={showAssignPanel ? (
              <div className="dd-assign">
                {SCOUT_NAMES.map((name) => (
                  <button
                    key={name}
                    type="button"
                    className={scoutMeta.assignedScout === name ? "is-active" : ""}
                    disabled={scoutActionBusy}
                    onClick={() => {
                      onAssignScout?.(name);
                      setShowAssignPanel(false);
                    }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            ) : null}
          />
        )}
      </article>
    </div>
  );
}

function IntelFeed({ prospects }) {
  const movers = useMemo(() => weeklyStockMovers(prospects, { minAbs: 1, limit: 80 }), [prospects]);
  const idle = !movers.length;
  return (
    <footer className={`dc-intel-feed dc-intel-feed--weekly${idle ? " is-idle" : ""}`}>
      <h3>Weekly stock</h3>
      <ul>
        {idle ? (
          <li>No weekly movement reported yet</li>
        ) : (
          movers.map((row) => (
            <li key={row.key} className={row.delta > 0 ? "is-up" : "is-down"}>
              #{row.rank} {row.name} {row.delta > 0 ? `+${row.delta}` : row.delta}
            </li>
          ))
        )}
      </ul>
    </footer>
  );
}

function StatsWorkspace({
  prospects,
  dateContext,
  leaderMode,
  setLeaderMode,
  profilesById,
  onSelectPlayer,
}) {
  const [metricSort, setMetricSort] = useState(null);
  const [sortViewMode, setSortViewMode] = useState("production");

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

  return (
    <div className="dc-stats-workspace dc-scroll-surface dc-leaders-modal-sc">
      <header className="dc-stats-workspace__head">
        <div>
          <p className="dc-lm-eyebrow">Draft class</p>
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
      <div className="dc-lm-row-stack">
        {!leaders.length ? (
          <p className="dc-empty-note">Not enough tracked data for this view yet.</p>
        ) : (
          leaders.map((row, index) => (
            <LeaderModalRow
              key={row.id}
              row={row}
              index={index}
              leaderMode={leaderMode}
              onSelect={(r) => r?.player && onSelectPlayer(r.player)}
              heroMetricKey={heroMetricKey}
              activeSortMetric={metricSort?.key}
              onMetricSort={handleMetricSort}
              sortViewMode={sortViewMode}
            />
          ))
        )}
      </div>
    </div>
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
















export default function DraftClass() {
  const {
    franchiseState,
    setScreen,
    loading,
    openWorldJuniors,
    pendingDraftProspectId,
    setPendingDraftProspectId,
    refreshFranchise,
    hydrateFranchiseHeavyState,
    hydrateFranchiseNarrative,
  } = useGameUI();
  const dateContext = useMemo(() => buildDateContext(franchiseState), [franchiseState]);

  useEffect(() => {
    hydrateFranchiseNarrative?.();
  }, [hydrateFranchiseNarrative]);

  useEffect(() => {
    if (typeof hydrateFranchiseHeavyState !== "function") return undefined;
    hydrateFranchiseHeavyState({
      includeRosterBrowser: false,
      includeDraftClassRankings: true,
      includeDraftClassHud: true,
    });
  }, [
    hydrateFranchiseHeavyState,
    franchiseState?.session_id,
    franchiseState?.prospect_revision,
    franchiseState?.franchise_today_iso,
    franchiseState?.calendar_cursor,
    franchiseState?.scouting_as_of_iso,
  ]);
  const [activeBoardView, setActiveBoardView] = useState("rank");
  const [boardSource, setBoardSource] = useState("scout");
  const [showConsensus, setShowConsensus] = useState(false);
  const [pyramidMode, setPyramidMode] = useState(false);
  const [workspace, setWorkspace] = useState("board");
  const [leaderMode, setLeaderMode] = useState("points");
  const [leadersModalOpen, setLeadersModalOpen] = useState(false);
  const [scoutingStore, setScoutingStore] = useState({});
  const [compareIds, setCompareIds] = useState([]);
  const [selectedProspect, setSelectedProspect] = useState(null);
  const [profileCache, setProfileCache] = useState({});
  const [profileLoadingId, setProfileLoadingId] = useState(null);
  const profileFetchRef = useRef(0);
  const profileCacheRef = useRef({});
  profileCacheRef.current = profileCache;
  const [deployBusy, setDeployBusy] = useState(false);
  const [deployNotice, setDeployNotice] = useState("");
  const [scoutActionBusy, setScoutActionBusy] = useState(false);
  const [scoutActionNote, setScoutActionNote] = useState("");

  const activeDeployments = useMemo(
    () => franchiseState?.scouting_state?.active_deployments || [],
    [franchiseState?.scouting_state?.active_deployments]
  );

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

  const requestProspectProfile = useCallback((prospectId) => {
    if (!prospectId) return Promise.resolve(null);
    const hudProfiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
    if (hudProfiles[prospectId] || profileCacheRef.current[prospectId]) {
      return Promise.resolve(hudProfiles[prospectId] || profileCacheRef.current[prospectId]);
    }
    const fetchId = profileFetchRef.current + 1;
    profileFetchRef.current = fetchId;
    setProfileLoadingId(prospectId);
    return getDraftProspectProfile(prospectId, {
      prospectRevision: franchiseState?.prospect_revision,
    })
      .then((loaded) => {
        if (profileFetchRef.current !== fetchId || !loaded) return loaded;
        setProfileCache((prev) => ({ ...prev, [prospectId]: loaded }));
        return loaded;
      })
      .catch(() => null)
      .finally(() => {
        if (profileFetchRef.current === fetchId) {
          setProfileLoadingId((cur) => (cur === prospectId ? null : cur));
        }
      });
  }, [franchiseState?.draft_class_hud?.prospect_profiles_by_id, franchiseState?.prospect_revision]);

  const prefetchProspectProfile = useCallback((player) => {
    if (!player?.id) return;
    const hudProfiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
    if (player.profile || hudProfiles[player.id] || profileCacheRef.current[player.id]) return;
    prefetchDraftProspectProfile(player.id, { prospectRevision: franchiseState?.prospect_revision });
  }, [franchiseState?.draft_class_hud?.prospect_profiles_by_id, franchiseState?.prospect_revision]);

  const openProspect = useCallback((player) => {
    if (!player?.id) return;
    const transcendent = Boolean(player.isTranscendent || player?.profile?.is_transcendent);
    if (transcendent) {
      playTranscendentBossSting();
    }
    setSelectedProspect(player);
    requestProspectProfile(player.id);
  }, [requestProspectProfile]);

  const runDossierScoutAction = useCallback(async (action) => {
    const prospectId = selectedProspect?.id;
    if (!prospectId || scoutActionBusy) return;
    setScoutActionBusy(true);
    setScoutActionNote("");
    try {
      const result = await runProspectScoutAction(prospectId, action, { intensity: "normal" });
      if (result?.ok === false) {
        setScoutActionNote(result.message || "Scouting action failed.");
      } else {
        setScoutActionNote(result.message || "Scouting file updated.");
        if (Array.isArray(result?.prospects)) {
          setScoutingStore((prev) => mergeScoutingStores(prev, scoutingStoreFromApiProspects(result.prospects)));
        }
      }
      await refreshFranchise();
      if (selectedProspect?.id) {
        profileFetchRef.current += 1;
        const rev = profileFetchRef.current;
        setProfileLoadingId(prospectId);
        try {
          const profilePayload = await requestProspectProfile(prospectId);
          if (profileFetchRef.current === rev && profilePayload) {
            setProfileCache((prev) => ({ ...prev, [prospectId]: profilePayload }));
          }
        } finally {
          if (profileFetchRef.current === rev) setProfileLoadingId(null);
        }
      }
    } catch (err) {
      setScoutActionNote(err?.response?.data?.detail || err?.message || "Scouting action failed.");
    } finally {
      setScoutActionBusy(false);
    }
  }, [selectedProspect?.id, scoutActionBusy, refreshFranchise, requestProspectProfile]);

  const handleDeployCoverage = useCallback(async (target) => {
    if (!target || deployBusy) return;
    setDeployBusy(true);
    setDeployNotice("");
    try {
      const result = await runCoverageDeploy(target);
      setDeployNotice(result?.message || `Coverage deployed to ${target.label}.`);
      await refreshFranchise();
    } catch (err) {
      setDeployNotice(err?.message || "Coverage deploy failed.");
    } finally {
      setDeployBusy(false);
    }
  }, [deployBusy, refreshFranchise]);

  const toggleProspectCompare = useCallback((prospectId) => {
    setCompareIds((prev) => {
      if (prev.includes(prospectId)) return prev.filter((id) => id !== prospectId);
      if (prev.length >= 3) return prev;
      return [...prev, prospectId];
    });
  }, []);

  String(franchiseState?.team?.strategy || "BPA").toUpperCase();

  const rawProspects = useMemo(() => {
    const profiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
    const mapped = mapBackendDraftBoard(franchiseState?.draft_class_rankings?.entries, dateContext);
    const withProfiles = mapped.map((p) => ({ ...p, profile: profiles[p.id] || null }));
    return enrichProspectsForWarRoom(withProfiles, profiles);
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

    const source = showConsensus ? "consensus" : boardSource;
    return rankProspectsForSource(list, source);
  }, [rawProspects, activeBoardView, boardSource, showConsensus]);

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
    const cached = selectedProspect.profile
      || profiles[selectedProspect.id]
      || profileCache[selectedProspect.id];
    if (cached) return cached;
    return buildStubProspectProfile(selectedProspect);
  }, [selectedProspect, franchiseState?.draft_class_hud?.prospect_profiles_by_id, profileCache]);

  const profileLoading = false;
  const profileEnriching = Boolean(
    selectedProspect?.id
    && profileLoadingId === selectedProspect.id
    && activeProfile?._stub
  );

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

          <nav className="dc-workspace-tabs" aria-label="Draft workspace">
            {WORKSPACE_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={workspace === tab.id ? "is-active" : ""}
                onClick={() => setWorkspace(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {workspace === "stats" ? (
            <StatsWorkspace
              prospects={filteredProspects}
              dateContext={dateContext}
              leaderMode={leaderMode}
              setLeaderMode={setLeaderMode}
              profilesById={franchiseState?.draft_class_hud?.prospect_profiles_by_id || {}}
              onSelectPlayer={openProspect}
            />
          ) : (
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
                onPrefetchProspect={prefetchProspectProfile}
                scoutingStore={scoutingStore}
                activeBoardView={activeBoardView}
                boardSource={boardSource}
                setBoardSource={setBoardSource}
                pyramidMode={pyramidMode}
                setPyramidMode={setPyramidMode}
                showConsensus={showConsensus}
                setShowConsensus={setShowConsensus}
                activeDeployments={activeDeployments}
                onDeployCoverage={handleDeployCoverage}
                deployBusy={deployBusy}
                deployNotice={deployNotice}
              />
            </div>
          )}

          <IntelFeed prospects={filteredProspects} />
        </main>

        {selectedProspect ? (
          <ProspectProfileModal
            player={selectedProspect}
            profile={activeProfile}
            profileLoading={profileLoading}
            profileEnriching={profileEnriching}
            meta={getScoutingMeta(scoutingStore, selectedProspect.id)}
            franchiseState={franchiseState}
            allProspects={rawProspects}
            compareIds={compareIds}
            onClose={() => setSelectedProspect(null)}
            onToggleWatchlist={() => toggleProspectWatchlist(selectedProspect.id)}
            onToggleTarget={() => toggleProspectTarget(selectedProspect.id)}
            onToggleDND={() => toggleProspectDnd(selectedProspect.id)}
            onToggleCompare={() => toggleProspectCompare(selectedProspect.id)}
            onAssignScout={(name) => assignProspectScout(selectedProspect.id, name)}
            onRunScoutAction={runDossierScoutAction}
            scoutActionBusy={scoutActionBusy}
            scoutActionNote={scoutActionNote}
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

        {null}
      </div>
    </div>
  );
}
