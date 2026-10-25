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

        <style>{`
          .dc-root.dc-screen {
            --dc-font-title: var(--dd-font-ui, Nunito, sans-serif);
            --dc-font-number: var(--dd-font-ui, Nunito, sans-serif);
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
            --dc-panel-soft: rgba(13, 37, 85, 0.72);

            /* Ceiling ramp: cold slate to hot gold, lightness climbing with the tier so
               the column reads as a gradient even without hue perception. */
            --dc-pot-fringe: #74849a;
            --dc-pot-depth: #9db0c6;
            --dc-pot-moderate: #8ab4ff;
            --dc-pot-strong: #38bdf8;
            --dc-pot-high: #2ee6d6;
            --dc-pot-elite: #e9a83c;
            --dc-pot-generational: #ffc94d;

            height: 100%;
            width: 100%;
            display: grid;
            /* Four children: header, stat strip, board, wire. The board owns the
               flexible track — a fifth declared track handed it to the wire. */
            grid-template-rows: auto auto auto minmax(0, 1fr) auto;
            gap: 8px;
            padding: 8px;
            color: var(--dc-text);
            font-family: var(--dc-font-ui);
            background:
              var(--wr-scan),
              radial-gradient(circle at 50% 42%, rgba(53, 185, 255, 0.1), transparent 28%),
              radial-gradient(circle at 10% 0%, rgba(43, 228, 255, 0.09), transparent 25%),
              radial-gradient(circle at 90% 8%, rgba(244, 198, 110, 0.07), transparent 20%),
              linear-gradient(180deg, #0c214d, #0b1b42);
            overflow: hidden;
          }

          .dc-screen--loading { display: grid; place-items: center; }
          .dc-scroll-surface { overflow: auto; scrollbar-width: thin; scrollbar-color: rgba(43, 228, 255, 0.55) rgba(14, 37, 85, 0.55); }
          .dc-scroll-surface::-webkit-scrollbar { width: 10px; height: 10px; }
          .dc-scroll-surface::-webkit-scrollbar-track { background: rgba(14, 37, 85, 0.55); border-radius: 999px; }
          .dc-scroll-surface::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(43, 228, 255, 0.75), rgba(43, 228, 255, 0.38));
            border-radius: 999px;
            border: 2px solid rgba(14, 37, 85, 0.55);
          }
          .dc-scroll-surface::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(43, 228, 255, 0.98), rgba(43, 228, 255, 0.6));
          }

          .dc-topbar,
          .dc-stat-strip,
          .dc-board-nav,
          .dc-selected-file,
          .dc-stock-rail,
          .dc-intel-feed {
            border: 1px solid var(--dc-line);
            border-radius: 10px;
            background: var(--dc-panel);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.35);
          }
          .dc-prospect-board {
            border: 1px solid var(--dc-line);
            border-radius: 0;
            background: var(--dc-panel);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.35);
            clip-path: var(--wr-cut);
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
              linear-gradient(180deg, rgba(14, 39, 89, 0.84), rgba(12, 32, 77, 0.8)),
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
            font-size: 1.2rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--dc-text);
            white-space: nowrap;
            height: 32px;
            display: inline-flex;
            align-items: center;
            padding: 0 2px;
            border: 0;
            background: transparent;
            font-family: var(--dc-font-title);
            font-weight: 800;
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
            background: rgba(14, 35, 81, 0.9);
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
              linear-gradient(180deg, rgba(12, 30, 72, 0.98), rgba(12, 28, 68, 0.98)),
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
              linear-gradient(180deg, rgba(12, 30, 72, 0.98), rgba(12, 28, 68, 0.98)),
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
            grid-template-columns: 84px minmax(0, 1fr);
            gap: 8px;
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
            border-bottom: 0;
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
          .dc-public-intel-note {
            margin: 4px 0 0;
            font-size: 0.625rem;
            line-height: 1.35;
            color: var(--dc-muted);
            letter-spacing: 0.02em;
            text-transform: none;
            max-width: 28rem;
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
            background: rgba(13, 37, 85, 0.6);
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
            grid-template-columns: 88px minmax(220px, 1.7fr) 42px minmax(72px, 0.7fr) 108px 72px 88px 78px;
            align-items: center;
            gap: 8px;
            padding: 0 10px;
          }
          .dc-prospect-board.is-consensus .dc-prospect-board__columns,
          .dc-prospect-board.is-consensus .dc-prospect-row {
            grid-template-columns: 88px minmax(200px, 1.6fr) 42px minmax(72px, 0.7fr) 108px 72px 88px 78px 52px;
          }

          .dc-prospect-board__columns {
            position: sticky;
            top: 0;
            z-index: 2;
            min-height: 30px;
            margin-bottom: 2px;
            padding-top: 6px;
            padding-bottom: 6px;
            border-bottom: 0;
            background: rgba(13, 33, 79, 0.98);
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
            font-size: 1.28rem;
            font-weight: 900;
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
            background: rgba(46, 230, 214, 0.1);
            border-color: rgba(46, 230, 214, 0.24);
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
            background: linear-gradient(180deg, rgba(12, 30, 48, 0.98), rgba(13, 33, 79, 1));
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
            background: rgba(14, 37, 85, 0.92);
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
            --hair: #1f2d4f;
            --hair2: #172444;
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
          .dc-prospect-row__team,
          .dc-prospect-row__nation-label {
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
          .dc-prospect-row__sub {
            margin: 0;
            font-size: 0.92rem;
            color: #b8d4ea;
            font-weight: 500;
            white-space: normal;
            overflow: visible;
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
          .dc-scout-menu { display: inline-flex; gap: 5px; padding: 4px 6px; border: 1px solid var(--dc-line); border-radius: var(--radius-hud, 4px); background: rgba(14, 35, 81, 0.7); }
          .dc-btn {
            border: 1px solid var(--dc-line);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.6875rem;
            cursor: pointer;
            color: var(--dc-text);
            background: rgba(14, 37, 86, 0.92);
          }
          .dc-btn--primary { border-radius: 0; clip-path: var(--wr-cut-sm); border-color: rgba(43,228,255,0.65); box-shadow: 0 0 18px rgba(43,228,255,0.18); background: linear-gradient(180deg, rgba(27,83,124,0.9), rgba(8,31,50,0.9)); }
          .dc-btn--secondary { background: rgba(14,30,46,0.9); }
          .dc-btn--danger { border-color: rgba(255,111,126,0.55); color: #ffd6db; background: linear-gradient(180deg, rgba(95,34,42,0.9), rgba(43,17,22,0.9)); }
          .dc-btn.is-active { border-color: rgba(108,247,166,0.6); }
          .dc-btn.is-disabled, .dc-btn:disabled { opacity: 0.4; cursor: not-allowed; }
          .dc-file-toggle-wrap { padding: 0 10px 8px; }

          .dc-selected-file__tabs { display: flex; gap: 5px; flex-wrap: wrap; padding: 0 10px 8px; border-bottom: 0; position: relative; }
          .dc-profile-tab.is-active { color: #0d214c; background: linear-gradient(180deg, #6df7ff, #47c8d8); border-color: transparent; }
          .dc-selected-file__detail { min-height: 0; flex: 1; padding: 8px 10px 10px; }

          .dc-stock-card { padding: 8px 10px; border-bottom: 0; position: relative; }
          .dc-stock-card::after { content: ""; position: absolute; left: 10px; right: 10px; bottom: 0; height: 1px; background: var(--wr-rule); }
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
            background: linear-gradient(180deg, rgba(12, 34, 52, 0.95), rgba(14, 37, 85, 0.92));
            color: var(--dc-cyan);
            cursor: pointer;
            flex-shrink: 0;
          }
          .dc-view-full--leaders:hover {
            border-color: rgba(43, 228, 255, 0.65);
            background: linear-gradient(180deg, rgba(16, 42, 64, 0.98), rgba(15, 40, 92, 0.95));
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
            color: #0d214c;
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
            --lm-panel: rgba(15, 42, 92, 0.94);
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
              linear-gradient(180deg, rgba(14, 37, 85, 0.97), rgba(12, 32, 76, 0.98));
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
            scrollbar-color: rgba(0, 216, 223, 0.32) rgba(12, 33, 77, 0.72);
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar {
            width: 6px;
            height: 6px;
          }
          .dc-leaders-modal__scroll::-webkit-scrollbar-track {
            background: rgba(12, 33, 77, 0.72);
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
            color: #0c214c;
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
            color: #0c214c;
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
            background: rgba(14, 37, 85, 0.72);
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
            background: rgba(13, 37, 85, 0.6);
            box-shadow: none;
          }
          .dc-intel-feed.is-idle ul { margin: 0; }
          .dc-intel-feed.is-idle li { color: var(--dc-muted); }
          .dc-intel-feed.is-idle li::before { color: var(--dc-muted); }
          .dc-intel-feed h3 { margin: 0; font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dc-muted); }
          .dc-intel-feed ul { margin: 3px 0 0; padding: 0; list-style: none; display: flex; gap: 12px; white-space: nowrap; overflow: auto; font-family: var(--dc-font-mono); font-size: 0.6875rem; }
          .dc-intel-feed li::before { content: "• "; color: var(--dc-cyan); }

          .dc-shared-headshot.player-headshot.size-sm { --size: 52px; }
          .dc-shared-headshot.player-headshot.size-md { --size: 72px; }
          .dc-shared-headshot.player-headshot.size-lg { --size: 148px; }

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
            .dc-command-grid { grid-template-columns: 84px minmax(0, 1fr); }
            /* Same eight-column ladder as the base board: the previous
               four-column override wrapped both the header and every row onto
               a second line, so ranks and grades no longer aligned. */
            .dc-prospect-board__columns,
            .dc-prospect-row {
              grid-template-columns: 88px minmax(200px, 1.6fr) 38px minmax(64px, 0.7fr) 96px 64px 80px 72px;
              gap: 9px;
              padding-left: 12px;
              padding-right: 12px;
            }
            .dc-prospect-row { min-height: 52px; }
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
            .wr-ceiling-chip.is-exact.is-glow,
            .wr-now-peak__chip.is-peak { animation: none; filter: none; }
          }

          .dc-profile-modal {
            position: fixed;
            inset: 0;
            z-index: var(--z-modal, 1200);
            display: flex;
            align-items: stretch;
            justify-content: stretch;
            padding: 0;
            pointer-events: none;
            background: rgba(11, 28, 68, 0.96);
          }

          .dc-profile-modal--prospect .dc-signal-panel.dc-dossier-shell {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
            max-height: 100%;
            margin: 0;
            flex: 1;
            border-radius: 0;
            border-color: var(--ops-grid-2);
            background:
              var(--wr-scan),
              linear-gradient(180deg, rgba(13, 37, 85, 0.98), rgba(12, 31, 74, 0.98));
          }

          .dc-profile-modal--prospect .dc-signal-panel:not(.dc-dossier-shell) {
            width: min(1120px, 98vw);
            height: min(92dvh, 920px);
            max-height: 92dvh;
            margin: auto;
            flex-shrink: 0;
            border-color: var(--ops-grid-2);
            background:
              var(--wr-scan),
              linear-gradient(180deg, rgba(13, 37, 85, 0.98), rgba(12, 31, 74, 0.98));
          }

          .dc-profile-modal--prospect .dc-signal-banner {
            background: linear-gradient(90deg, rgba(46, 230, 214, 0.08), transparent);
          }

          .dc-profile-modal--prospect .dc-signal-banner span {
            color: var(--ops-cyan);
          }

          .dc-dossier-shell--loading .dc-dossier-loading {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 24px;
            color: var(--dc-muted);
            text-align: center;
          }

          .dc-dossier-shell--loading .dc-dossier-loading strong {
            font-family: var(--dc-font-title);
            font-size: 1.35rem;
            letter-spacing: 0.04em;
            color: var(--dc-text);
          }

          .dc-dossier-shell--loading .dc-dossier-loading p {
            margin: 0;
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
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
            background: rgba(11, 28, 68, 0.72);
            backdrop-filter: blur(7px);
            pointer-events: auto;
            cursor: pointer;
          }
          .dc-signal-panel:not(.dc-dossier-shell) {
            position: relative;
            width: min(1360px, 97vw);
            height: min(94vh, 900px);
            height: min(94dvh, 900px);
            max-height: 94vh;
            max-height: 94dvh;
            border: 1px solid rgba(118, 200, 245, 0.42);
            border-radius: 0;
            clip-path: var(--wr-cut);
            background:
              var(--wr-scan),
              radial-gradient(ellipse 85% 70% at 12% -5%, rgba(56, 190, 255, 0.42), transparent 55%),
              radial-gradient(ellipse 70% 55% at 95% 105%, rgba(30, 140, 255, 0.32), transparent 52%),
              radial-gradient(ellipse 50% 40% at 48% 42%, rgba(40, 120, 210, 0.28), transparent 62%),
              linear-gradient(165deg, #143a62 0%, #0d2748 38%, #081c36 72%, #0c2152 100%);
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
          .dc-signal-panel.dc-dossier-shell,
          .dc-signal-panel--dossier {
            position: relative;
            width: 100%;
            height: 100%;
            max-height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            pointer-events: auto;
            border: 1px solid rgba(118, 200, 245, 0.42);
            border-radius: 0;
            clip-path: var(--wr-cut);
            grid-template-columns: unset;
            grid-template-rows: unset;
            grid-template-areas: unset;
          }
          .dc-signal-panel.dc-brochure:not(.dc-brochure--deep):not(.dc-dossier-shell) {
            width: min(1480px, 98vw);
            height: min(96vh, 980px);
            height: min(96dvh, 980px);
            grid-template-columns: minmax(220px, 260px) minmax(0, 1.35fr) minmax(280px, 340px);
            grid-template-rows: auto minmax(0, 1.2fr) minmax(220px, 0.95fr);
            grid-template-areas:
              "banner banner banner"
              "identity dna reports"
              "identity analytics analytics";
          }
          .dc-featured { display: none !important; }
          .dc-signal-panel--premium .dc-signal-identity {
            background:
              linear-gradient(180deg, rgba(10,28,44,0.96), rgba(12, 32, 76, 0.88)),
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
            background: linear-gradient(180deg, rgba(244,198,110,0.18), rgba(20, 43, 90, 0.85));
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
            color: #17233a;
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
            background: #0f2553;
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
            background: rgba(12, 30, 72, 0.82);
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
            background: linear-gradient(180deg, rgba(43,228,255,0.08), rgba(14, 35, 79, 0.5));
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
            background: rgba(13, 33, 79, 0.7);
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
            background: radial-gradient(circle at 50% 38%, rgba(43,228,255,0.12), transparent 58%), rgba(12, 30, 74, 0.55);
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
            width: min(100%, 360px);
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
            width: 108px;
            height: 108px;
            margin: 0;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(14, 39, 90, 0.96), rgba(14, 36, 85, 0.72));
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
            background: linear-gradient(180deg, rgba(15, 39, 88, 0.85), rgba(12, 30, 72, 0.7));
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
          .dc-signal-actionbar--dossier {
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: stretch;
          }
          .dc-dossier-scout-actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
          }
          .dc-dossier-scout-note {
            margin: 0;
            font-size: 0.72rem;
            color: var(--dc-cyan);
          }
          .dc-dossier-scout-assign {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
          }
          .dc-dossier-scout-assign button {
            border: 1px solid var(--dc-line);
            background: rgba(14, 37, 85, 0.8);
            color: var(--dc-text);
            font-size: 0.65rem;
            padding: 6px 8px;
            cursor: pointer;
          }
          .dc-dossier-scout-assign button.is-active {
            border-color: var(--dc-cyan);
            color: var(--dc-cyan);
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
            color: #17233a;
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

          .dc-signal-panel:not(.dc-dossier-shell) .dc-signal-identity {
            grid-area: identity;
            padding: 12px 12px 10px;
            border-right: 1px solid rgba(118, 200, 245, 0.14);
            display: flex;
            flex-direction: column;
            gap: 7px;
            min-height: 0;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(12,36,58,0.95), rgba(13, 35, 83, 0.72));
          }
          .dc-dossier-shell .dc-signal-identity {
            grid-area: unset;
            padding: 0;
            border-right: none;
            overflow: visible;
            background: transparent;
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
            background: linear-gradient(180deg, rgba(12,32,48,0.85), rgba(13, 33, 79, 0.7));
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
            background: linear-gradient(180deg, rgba(15, 37, 86, 0.95), rgba(13, 30, 72, 0.88));
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
            background: rgba(12, 29, 70, 0.72);
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
            height: 140px;
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
            background: rgba(13, 33, 79, 0.92);
            cursor: pointer;
            display: grid;
            place-items: center;
            color: #e8f4ff;
            padding: 0;
            line-height: 0;
          }
          .dc-modal-close:hover {
            border-color: rgba(43, 228, 255, 0.55);
            background: rgba(15, 40, 92, 0.98);
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
            .dc-signal-chart__svg { height: 72px; }
            .dc-signal-chart--weekly .dc-signal-chart__svg { height: 140px; }
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
            background: rgba(14, 26, 56, 0.72);
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
