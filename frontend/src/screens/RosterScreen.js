import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { enrichRosterPlayer } from "../game/rosterColumns";
import { GameFooter } from "../components/game/GameFooter";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { formatProspectLeague, formatProspectTeam } from "../events/prospectDevelopment/prospectDevelopmentHelpers";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { nationalityCode, ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import {
  getBaseOverall,
  getOverallDrop,
  getOverallTooltip,
  getUniversalOverall,
} from "../utils/playerOverall";

/**
 * RosterScreen.js
 *
 * Massive connected roster revamp.
 *
 * DESIGN RULES:
 * - No fake UI actions.
 * - No button appears active unless it actually changes local state or calls a real handler.
 * - No fake player values are invented for display.
 * - Missing backend values show as "—", "Unavailable", or "Not connected".
 * - Display overall is the universal backend OVR (effective_ovr / ovr).
 * - Potential is position-aware, age-aware, development-aware, and performance-aware.
 * - Skaters, defensemen, and goalies are evaluated differently.
 * - Existing franchiseState / roster_browser / draft_class_rankings paths are preserved.
 * - Styling later is designed to visually match CalendarScreen's premium dark NHL layout.
 */

const EMPTY_ARRAY = Object.freeze([]);
const EMPTY_OBJECT = Object.freeze({});

const TABLE_PAGE_SIZE = 16;

const DEFAULT_CAP_LIMIT = 83.5;
const NHL_ACTIVE_ROSTER_LIMIT = 23;
const NHL_CONTRACT_RESERVE_LIMIT = 50;

const PLAYER_POOLS = {
  ORGANIZATION: "organization",
  MY_PROSPECTS: "my_prospects",
  FREE_AGENTS: "free_agents",
  OVERSEAS: "overseas",
  DEVELOPMENT: "development",
  DRAFT_CLASS: "draft_class",
};

const PLAYER_SEARCH_MODES = {
  TEAM_ROSTERS: "team_rosters",
  NHL_LEAGUE: "nhl_league",
};

const PLAYER_SEARCH_MODE_OPTIONS = [
  { value: PLAYER_SEARCH_MODES.TEAM_ROSTERS, label: "Team Rosters" },
  { value: PLAYER_SEARCH_MODES.NHL_LEAGUE, label: "NHL Roster" },
];

const VIEW_MODES = {
  BOARD: "board",
  TABLE: "table",
  CARDS: "cards",
  LINES: "lines",
  RATINGS: "ratings",
};

const VIEW_MODE_OPTIONS = [
  { value: VIEW_MODES.BOARD, label: "Board" },
  { value: VIEW_MODES.TABLE, label: "Table" },
  { value: VIEW_MODES.CARDS, label: "Cards" },
  { value: VIEW_MODES.LINES, label: "Lines" },
  { value: VIEW_MODES.RATINGS, label: "Ratings" },
];

const PANEL_TABS = [
  { value: "overview", label: "Profile" },
  { value: "ratings", label: "Ratings" },
  { value: "production", label: "Stats" },
  { value: "contract", label: "Contract" },
  { value: "development", label: "Development" },
  { value: "history", label: "Timeline" },
];

const SORT_KEYS = [
  { value: "overall_desc", label: "Overall ↓" },
  { value: "overall_asc", label: "Overall ↑" },
  { value: "true_overall_desc", label: "True Overall ↓" },
  { value: "potential_score_desc", label: "Potential Score ↓" },
  { value: "age_asc", label: "Age ↑" },
  { value: "age_desc", label: "Age ↓" },
  { value: "name_asc", label: "Name A-Z" },
  { value: "name_desc", label: "Name Z-A" },
  { value: "points_desc", label: "Points ↓" },
  { value: "goals_desc", label: "Goals ↓" },
  { value: "assists_desc", label: "Assists ↓" },
  { value: "ppg_desc", label: "Points/Game ↓" },
  { value: "morale_desc", label: "Morale ↓" },
  { value: "fatigue_desc", label: "Fatigue ↓" },
  { value: "salary_desc", label: "Cap Hit ↓" },
  { value: "term_desc", label: "Term ↓" },
  { value: "asset_value_desc", label: "Asset Value ↓" },
];

const POSITION_FILTERS = ["ALL", "F", "C", "LW", "RW", "D", "LD", "RD", "G"];
const LEAGUE_FILTERS = ["ALL", "NHL", "AHL", "ECHL", "CHL", "NCAA", "EU", "INTL"];
const STATUS_FILTERS = [
  "All",
  "Active",
  "Injured",
  "Scratched",
  "Assigned",
  "Unsigned",
  "Draft Eligible",
];

const PLAYER_TYPE_FILTERS = [
  "ALL",
  "SNIPER",
  "PLAYMAKER",
  "POWER",
  "TWO-WAY",
  "DEFENSIVE",
  "ENFORCER",
  "GRINDER",
  "OFFENSIVE D",
  "DEFENSIVE D",
  "TWO-WAY D",
  "HYBRID",
  "BUTTERFLY",
  "STANDUP",
  "BALANCED",
];

const POTENTIAL_TIERS = {
  FRANCHISE: {
    label: "Franchise",
    score: 100,
    skaterMin: 92,
    goalieMin: 91,
  },
  ELITE: {
    label: "Elite",
    score: 92,
    skaterMin: 86,
    goalieMin: 85,
  },
  TOP_PAIR_D: {
    label: "Top Pair D",
    score: 87,
    skaterMin: 84,
    goalieMin: 0,
  },
  TOP_LINE: {
    label: "Top Line",
    score: 86,
    skaterMin: 84,
    goalieMin: 0,
  },
  STARTER: {
    label: "Starter",
    score: 86,
    skaterMin: 0,
    goalieMin: 83,
  },
  TOP_4_D: {
    label: "Top 4 D",
    score: 80,
    skaterMin: 78,
    goalieMin: 0,
  },
  TOP_6: {
    label: "Top 6",
    score: 80,
    skaterMin: 78,
    goalieMin: 0,
  },
  TANDEM: {
    label: "Tandem",
    score: 78,
    skaterMin: 0,
    goalieMin: 78,
  },
  MIDDLE_6: {
    label: "Middle 6",
    score: 70,
    skaterMin: 73,
    goalieMin: 0,
  },
  THIRD_PAIR_D: {
    label: "Third Pair D",
    score: 68,
    skaterMin: 72,
    goalieMin: 0,
  },
  BACKUP: {
    label: "Backup",
    score: 68,
    skaterMin: 0,
    goalieMin: 72,
  },
  BOTTOM_6: {
    label: "Bottom 6",
    score: 61,
    skaterMin: 68,
    goalieMin: 0,
  },
  DEPTH: {
    label: "Depth",
    score: 50,
    skaterMin: 0,
    goalieMin: 0,
  },
  AHL: {
    label: "AHL",
    score: 38,
    skaterMin: 0,
    goalieMin: 0,
  },
};

const POTENTIAL_ORDER = {
  Franchise: 100,
  Elite: 92,
  "Top Pair D": 87,
  "Top Line": 86,
  Starter: 86,
  "Top 4 D": 80,
  "Top 6": 80,
  Tandem: 78,
  "Middle 6": 70,
  "Third Pair D": 68,
  Backup: 68,
  "Bottom 6": 61,
  Depth: 50,
  AHL: 38,
  Unknown: 0,
  "—": 0,
};

const ATTRIBUTE_BUCKETS = {
  SKATER_OFFENSE: [
    "offense",
    "shooting",
    "shooting_accuracy",
    "shot_accuracy",
    "wrist_shot_accuracy",
    "slap_shot_accuracy",
    "shooting_power",
    "shot_power",
    "wrist_shot_power",
    "slap_shot_power",
    "finishing",
    "scoring",
    "goalscoring",
    "playmaking",
    "passing",
    "puck_control",
    "deking",
    "hands",
    "offensive_iq",
    "creativity",
    "vision",
  ],
  SKATER_DEFENSE: [
    "defense",
    "defensive_iq",
    "checking_defense",
    "stick_checking",
    "shot_blocking",
    "positioning",
    "backchecking",
    "gap_control",
    "takeaways",
    "faceoffs",
    "discipline",
  ],
  SKATER_SKATING: [
    "skating",
    "speed",
    "acceleration",
    "agility",
    "balance",
    "edgework",
    "endurance",
    "stamina",
  ],
  SKATER_PHYSICAL: [
    "physical",
    "strength",
    "checking",
    "body_checking",
    "hitting",
    "durability",
    "grit",
    "toughness",
    "net_front",
  ],
  SKATER_MENTAL: [
    "mental",
    "hockey_iq",
    "iq",
    "consistency",
    "clutch",
    "poise",
    "leadership",
    "work_ethic",
    "competitiveness",
  ],
  GOALIE_TECHNICAL: [
    "goalie",
    "goaltending",
    "reflexes",
    "glove",
    "blocker",
    "stick_low",
    "five_hole",
    "angles",
    "rebound_control",
    "save_recovery",
    "butterfly",
  ],
  GOALIE_ATHLETIC: [
    "speed",
    "agility",
    "lateral_movement",
    "flexibility",
    "explosiveness",
    "balance",
    "endurance",
    "stamina",
  ],
  GOALIE_MENTAL: [
    "mental",
    "hockey_iq",
    "iq",
    "poise",
    "composure",
    "consistency",
    "clutch",
    "focus",
    "vision",
  ],
  GOALIE_PUCK: [
    "puck_playing",
    "passing",
    "puck_control",
    "rebound_control",
    "breakout",
  ],
};

const POSITION_WEIGHTS = {
  C: {
    offense: 0.3,
    defense: 0.2,
    skating: 0.2,
    physical: 0.1,
    mental: 0.2,
  },
  LW: {
    offense: 0.36,
    defense: 0.14,
    skating: 0.22,
    physical: 0.12,
    mental: 0.16,
  },
  RW: {
    offense: 0.36,
    defense: 0.14,
    skating: 0.22,
    physical: 0.12,
    mental: 0.16,
  },
  F: {
    offense: 0.34,
    defense: 0.16,
    skating: 0.22,
    physical: 0.12,
    mental: 0.16,
  },
  D: {
    offense: 0.18,
    defense: 0.36,
    skating: 0.18,
    physical: 0.14,
    mental: 0.14,
  },
  LD: {
    offense: 0.18,
    defense: 0.36,
    skating: 0.18,
    physical: 0.14,
    mental: 0.14,
  },
  RD: {
    offense: 0.18,
    defense: 0.36,
    skating: 0.18,
    physical: 0.14,
    mental: 0.14,
  },
  G: {
    technical: 0.48,
    athletic: 0.2,
    mental: 0.22,
    puck: 0.1,
  },
};

const RATING_GROUP_LABELS = {
  offense: "Offense",
  defense: "Defense",
  skating: "Skating",
  physical: "Physical",
  mental: "Mental",
  technical: "Technical",
  athletic: "Athletic",
  puck: "Puck Play",
};

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function safeNumOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function safeStr(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function round1(n) {
  const value = safeNum(n, 0);
  return Math.round(value * 10) / 10;
}

function round0(n) {
  return Math.round(safeNum(n, 0));
}

function pickFirstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function normalizeKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

function normalizeMoneyMillions(value) {
  const raw = safeNum(value, 0);

  if (!raw) return 0;

  if (raw > 1000000) return raw / 1000000;
  if (raw > 1000) return raw / 1000000;

  return raw;
}

function formatMoneyMillions(value, empty = "—") {
  const n = normalizeMoneyMillions(value);
  if (!Number.isFinite(n) || n <= 0) return empty;
  return `$${n.toFixed(2)}M`;
}

function formatSignedNumber(value, decimals = 1) {
  const n = safeNum(value, 0);
  if (n > 0) return `+${n.toFixed(decimals)}`;
  if (n < 0) return n.toFixed(decimals);
  return Number(0).toFixed(decimals);
}

function formatPercent(value, empty = "—") {
  const n = safeNumOrNull(value);
  if (n === null) return empty;
  if (Math.abs(n) <= 1) return `${(n * 100).toFixed(1)}%`;
  return `${n.toFixed(1)}%`;
}

function formatDecimal(value, digits = 3, empty = "—") {
  const n = safeNumOrNull(value);
  if (n === null) return empty;
  return n.toFixed(digits);
}

function normalizeRatingScale(value) {
  const n = safeNumOrNull(value);
  if (n === null) return null;

  if (n <= 1 && n >= 0) return clamp(n * 100, 0, 100);
  if (n <= 10 && n > 1) return clamp(n * 10, 0, 100);

  return clamp(n, 0, 100);
}

function normalizePercentScale(value, fallback = 50) {
  const n = safeNumOrNull(value);
  if (n === null) return fallback;
  if (n >= 0 && n <= 1) return clamp(n * 100, 0, 100);
  return clamp(n, 0, 100);
}

function normalizePosition(position) {
  const raw = safeStr(position, "").trim().toUpperCase();

  if (!raw) return "—";

  if (["C", "LW", "RW", "F", "LD", "RD", "D", "G"].includes(raw)) return raw;
  if (raw.includes("GOAL")) return "G";
  if (raw.includes("LEFT") && raw.includes("DEF")) return "LD";
  if (raw.includes("RIGHT") && raw.includes("DEF")) return "RD";
  if (raw.includes("DEF")) return "D";
  if (raw.includes("CENTER") || raw === "CENTRE") return "C";
  if (raw.includes("LEFT") && raw.includes("WING")) return "LW";
  if (raw.includes("RIGHT") && raw.includes("WING")) return "RW";
  if (raw.includes("LW")) return "LW";
  if (raw.includes("RW")) return "RW";
  if (raw.includes("LD")) return "LD";
  if (raw.includes("RD")) return "RD";
  if (raw.includes("D")) return "D";
  if (raw.includes("G")) return "G";
  if (raw.includes("C")) return "C";

  return raw;
}

function isForwardPosition(position) {
  const p = normalizePosition(position);
  return ["C", "LW", "RW", "F"].includes(p);
}

function isDefensePosition(position) {
  const p = normalizePosition(position);
  return ["D", "LD", "RD"].includes(p);
}

function isGoaliePosition(position) {
  return normalizePosition(position) === "G";
}

function positionMatchesFilter(position, filter) {
  const p = normalizePosition(position);

  if (filter === "ALL") return true;
  if (filter === "F") return isForwardPosition(p);
  if (filter === "D") return isDefensePosition(p);

  return p === filter;
}

function getPositionClass(position) {
  if (isGoaliePosition(position)) return "goalie";
  if (isDefensePosition(position)) return "defense";
  if (isForwardPosition(position)) return "forward";
  return "unknown";
}

function getPositionDisplay(position) {
  const p = normalizePosition(position);

  if (p === "G") return "Goalie";
  if (p === "LD") return "Left Defense";
  if (p === "RD") return "Right Defense";
  if (p === "D") return "Defense";
  if (p === "C") return "Center";
  if (p === "LW") return "Left Wing";
  if (p === "RW") return "Right Wing";
  if (p === "F") return "Forward";

  return p || "—";
}

function initialsFromName(name) {
  const raw = safeStr(name, "").trim();
  if (!raw || raw === "—") return "—";

  const parts = raw.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();

  return `${parts[0][0] || ""}${parts[parts.length - 1][0] || ""}`.toUpperCase();
}

function getPlayerName(player) {
  return safeStr(
    pickFirstDefined(
      player?.name,
      player?.player_name,
      player?.playerName,
      player?.full_name,
      player?.fullName,
      player?.display_name,
      player?.displayName
    ),
    "Unnamed Player"
  );
}

function getPlayerId(player, fallback = "") {
  return safeStr(
    pickFirstDefined(
      player?.id,
      player?.player_id,
      player?.playerId,
      player?.uid,
      player?.key
    ),
    fallback
  );
}

function getPlayerTeamId(player) {
  return safeStr(
    pickFirstDefined(
      player?.team_id,
      player?.teamId,
      player?.team,
      player?.team_abbr,
      player?.teamAbbr,
      player?.rights_team_id,
      player?.rightsTeamId
    ),
    ""
  );
}

function buildFranchiseStatsLookup(franchiseState) {
  const map = new Map();
  const sc = franchiseState?.stats_central || EMPTY_OBJECT;
  const buckets = [
    sc.players,
    sc.skaters,
    sc.goalies,
    sc.user_team_skaters,
    sc.user_leaders,
    sc.leaders,
    franchiseState?.player_season_stats ? Object.values(franchiseState.player_season_stats) : EMPTY_ARRAY,
  ];

  buckets.forEach((rows) => {
    if (!Array.isArray(rows)) return;

    rows.forEach((row) => {
      const id = safeStr(getPlayerId(row, ""), "").toLowerCase();
      if (!id) return;
      map.set(id, row);
    });
  });

  return map;
}

function mergeFranchiseStatsIntoPlayer(player, statsLookup) {
  if (!player || !statsLookup?.size) return player;

  const id = safeStr(getPlayerId(player, ""), "").toLowerCase();
  const statsRow = statsLookup.get(id);
  if (!statsRow) return player;

  const mergedStats = {
    ...(player.season_stats || player.seasonStats || player.stats || EMPTY_OBJECT),
    gp: pickFirstDefined(statsRow.gp, statsRow.games_played),
    g: pickFirstDefined(statsRow.g, statsRow.goals),
    goals: pickFirstDefined(statsRow.goals, statsRow.g),
    a: pickFirstDefined(statsRow.a, statsRow.assists),
    assists: pickFirstDefined(statsRow.assists, statsRow.a),
    pts: pickFirstDefined(statsRow.pts, statsRow.points),
    points: pickFirstDefined(statsRow.points, statsRow.pts),
    ppg: pickFirstDefined(statsRow.ppg, statsRow.points_per_game),
    wins: pickFirstDefined(statsRow.w, statsRow.wins),
    w: pickFirstDefined(statsRow.w, statsRow.wins),
    losses: pickFirstDefined(statsRow.l, statsRow.losses),
    l: pickFirstDefined(statsRow.l, statsRow.losses),
    otl: pickFirstDefined(statsRow.otl, statsRow.ot),
    saves: pickFirstDefined(statsRow.saves, statsRow.sv),
    shots_against: pickFirstDefined(statsRow.shots_against, statsRow.sa),
    goals_against: pickFirstDefined(statsRow.goals_against, statsRow.ga),
    sv_pct: pickFirstDefined(statsRow.save_pct, statsRow.sv_pct, statsRow.savePct),
    save_pct: pickFirstDefined(statsRow.save_pct, statsRow.sv_pct, statsRow.savePct),
    gaa: statsRow.gaa,
    sog: pickFirstDefined(statsRow.sog, statsRow.shots),
    shots: pickFirstDefined(statsRow.shots, statsRow.sog),
    pim: statsRow.pim,
    plus_minus: pickFirstDefined(statsRow.plus_minus, statsRow.pm),
    toi: pickFirstDefined(statsRow.toi, statsRow.average_toi),
  };

  return {
    ...player,
    season_stats: mergedStats,
    stats: mergedStats,
    league: formatProspectLeague(player)
      || formatProspectLeague(statsRow)
      || player.league
      || statsRow.league_display
      || statsRow.league_code
      || statsRow.league_name,
    league_display: player.league_display || statsRow.league_display || formatProspectLeague(player) || formatProspectLeague(statsRow),
    team_name: formatProspectTeam(player, statsRow.team_name || player.team_name || statsRow.team_abbrev || statsRow.team_abbr),
  };
}

function isUserOwnedProspect(raw, userTeamId) {
  const uid = safeStr(userTeamId, "").toLowerCase();
  if (!uid) return false;

  if (raw?.is_user_prospect || raw?.isUserProspect || raw?.user_prospect || raw?.userProspect) {
    return true;
  }

  if (raw?.owned_by_user || raw?.ownedByUser || raw?.is_user || raw?.isUser) {
    return true;
  }

  const rightsFields = [
    raw?.drafted_by_team_id,
    raw?.draftedByTeamId,
    raw?.developed_by_team_id,
    raw?.developedByTeamId,
    raw?.rights_team_id,
    raw?.rightsTeamId,
    raw?.nhl_rights_id,
    raw?.nhlRightsId,
    raw?.owning_team_id,
    raw?.owningTeamId,
    raw?.org_team_id,
    raw?.orgTeamId,
    raw?.prospect_team_id,
    raw?.prospectTeamId,
  ];

  return rightsFields.some((field) => safeStr(field, "").toLowerCase() === uid);
}

function collectMyProspectRawPlayers(rb, franchiseState, userTeamId, userOrganization) {
  const rows = [];
  const seen = new Set();

  const push = (player, meta = EMPTY_OBJECT) => {
    if (!player) return;

    const id = getPlayerId(player, "");
    const name = getPlayerName(player);
    const dedupeKey = id || `${meta.league || "—"}-${name}`;

    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);

    rows.push({
      ...player,
      ...meta,
      _source: PLAYER_POOLS.MY_PROSPECTS,
    });
  };

  const orgName = userOrganization?.name || franchiseState?.team?.name || "Organization";

  (userOrganization?.ahl || EMPTY_ARRAY).forEach((player) => {
    push(player, {
      league: "AHL",
      team_name: player.team_name || orgName,
      pipeline_level: "AHL",
    });
  });

  (userOrganization?.echl || EMPTY_ARRAY).forEach((player) => {
    push(player, {
      league: "ECHL",
      team_name: player.team_name || orgName,
      pipeline_level: "ECHL",
    });
  });

  (rb?.development_leagues || EMPTY_ARRAY).forEach((league) => {
    const leagueCode = league?.league_code || league?.league_name || "DEV";
    const leagueDisplay = formatProspectLeague(league);

    (league?.teams || EMPTY_ARRAY).forEach((team) => {
      (team?.players || EMPTY_ARRAY).forEach((player) => {
        if (!isUserOwnedProspect(player, userTeamId)) return;

        const ctx = {
          ...player,
          league_code: player.league_code || leagueCode,
          league_name: player.league_name || league.league_name,
          league_display: player.league_display || leagueDisplay,
          team_name: player.team_name || team.name || team.team_name,
          team_id: player.team_id || team.team_id,
        };

        push(player, {
          league: formatProspectLeague(ctx) || leagueDisplay,
          league_display: formatProspectLeague(ctx) || leagueDisplay,
          league_code: ctx.league_code,
          league_name: ctx.league_name,
          team_name: formatProspectTeam(ctx, team.name || team.team_name),
          team_id: ctx.team_id,
          dev_league: leagueDisplay,
          pipeline_level: leagueDisplay || leagueCode,
        });
      });
    });
  });

  const extraPools = [
    franchiseState?.prospect_pool,
    franchiseState?.prospectPool,
    franchiseState?.prospects,
    franchiseState?.team_prospects,
    franchiseState?.teamProspects,
    franchiseState?.pipeline,
    franchiseState?.wjc_tournament_bundle?.user_prospects,
    ...(Array.isArray(franchiseState?.pending_ui_popups) ? franchiseState.pending_ui_popups : [])
      .filter((pop) => pop?.kind === "wjc_tournament" || pop?.wjc_live)
      .flatMap((pop) => pop.user_prospects || []),
    ...(Array.isArray(franchiseState?.showcase_archive) ? franchiseState.showcase_archive : [])
      .filter((arch) => arch?.kind === "wjc_tournament" || arch?.wjc_phase)
      .flatMap((arch) => arch.user_prospects || []),
  ];

  extraPools.forEach((pool) => {
    if (!Array.isArray(pool)) return;

    pool.forEach((player) => {
      if (userTeamId && !isUserOwnedProspect(player, userTeamId)) return;

      push(player, {
        league: player.league || player.dev_league || player.league_code || "PROSPECT",
        team_name: player.team_name || player.roster || orgName,
        pipeline_level: player.roster || player.league || "Prospect",
      });
    });
  });

  return rows;
}

function collectOrganizationPoolPlayers(organizations, poolKey = "nhl") {
  const rows = [];

  (Array.isArray(organizations) ? organizations : EMPTY_ARRAY).forEach((org) => {
    const teamName = safeStr(org?.name || org?.team_name, "—");
    const teamId = safeStr(org?.team_id || org?.id || org?.abbr, "");
    const pool = org?.[poolKey];

    if (!Array.isArray(pool)) return;

    pool.forEach((player) => {
      rows.push({
        ...player,
        _source: PLAYER_POOLS.ORGANIZATION,
        team_name: player?.team_name || teamName,
        team_id: player?.team_id || teamId,
        league: player?.league || poolKey.toUpperCase(),
      });
    });
  });

  return rows;
}

function normalizeArchetype(player) {
  const raw = safeStr(
    pickFirstDefined(
      player?.archetype,
      player?.player_type,
      player?.playerType,
      player?.style,
      player?.type
    ),
    ""
  );

  const key = normalizeKey(raw);

  if (!key) return "Balanced";
  if (key.includes("sniper")) return "Sniper";
  if (key.includes("playmaker")) return "Playmaker";
  if (key.includes("power")) return "Power Forward";
  if (key.includes("two_way_d") || key.includes("twoway_d")) return "Two-Way D";
  if (key.includes("offensive_d")) return "Offensive D";
  if (key.includes("defensive_d")) return "Defensive D";
  if (key.includes("two_way") || key.includes("twoway")) return "Two-Way";
  if (key.includes("defensive")) return "Defensive";
  if (key.includes("grinder")) return "Grinder";
  if (key.includes("enforcer")) return "Enforcer";
  if (key.includes("butterfly")) return "Butterfly";
  if (key.includes("hybrid")) return "Hybrid";
  if (key.includes("standup") || key.includes("stand_up")) return "Standup";

  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function getArchetypeColor(archetype) {
  const key = normalizeKey(archetype);

  if (key.includes("sniper")) return "#ff6868";
  if (key.includes("playmaker")) return "#5fc7ff";
  if (key.includes("power")) return "#ffa94d";
  if (key.includes("two_way")) return "#81f0a4";
  if (key.includes("offensive_d")) return "#8dc6ff";
  if (key.includes("defensive_d") || key.includes("defensive")) return "#6ee7b7";
  if (key.includes("grinder")) return "#d6b36a";
  if (key.includes("enforcer")) return "#ef7d7d";
  if (key.includes("butterfly") || key.includes("hybrid") || key.includes("standup")) return "#c4a7ff";

  return "#9aa7bd";
}

function normalizeContract(player) {
  const contract = player?.contract || EMPTY_OBJECT;

  const capHit = normalizeMoneyMillions(
    pickFirstDefined(
      contract.cap_hit,
      contract.capHit,
      contract.aav,
      contract.salary,
      player?.cap_hit,
      player?.capHit,
      player?.aav,
      player?.salary
    )
  );

  const salary = normalizeMoneyMillions(
    pickFirstDefined(
      contract.salary,
      contract.base_salary,
      contract.baseSalary,
      player?.salary,
      player?.base_salary,
      player?.baseSalary,
      capHit
    )
  );

  const term = safeNum(
    pickFirstDefined(
      contract.term,
      contract.years,
      contract.years_remaining,
      contract.yearsRemaining,
      player?.contract_term,
      player?.contractTerm,
      player?.term,
      player?.years_remaining,
      player?.yearsRemaining
    ),
    0
  );

  const rawExpiry = safeStr(
    pickFirstDefined(
      contract.expiry,
      contract.expires,
      contract.expiry_year,
      contract.expiryYear,
      player?.contract_expiry,
      player?.contractExpiry,
      player?.expiry
    ),
    ""
  );

  const type = safeStr(
    pickFirstDefined(
      contract.contract_type,
      contract.contractType,
      contract.type,
      player?.contract_type,
      player?.contractType
    ),
    term > 0 || capHit > 0 ? "Standard" : "Unsigned"
  );

  const isSigned = capHit > 0 || term > 0 || type.toLowerCase() !== "unsigned";

  let expiry = rawExpiry;
  if (!expiry || expiry.toLowerCase() === "unsigned") {
    if (term > 0) expiry = `${term} yr remaining`;
    else if (isSigned) expiry = "—";
    else expiry = "No contract";
  }

  const clause = safeStr(
    pickFirstDefined(
      contract.clause,
      contract.trade_clause,
      contract.tradeClause,
      player?.clause,
      player?.trade_clause,
      player?.tradeClause
    ),
    ""
  );

  return {
    capHit,
    salary,
    term,
    expiry,
    type,
    clause,
    isSigned,
  };
}

function normalizeSeasonStats(player) {
  const s = player?.season_stats || player?.seasonStats || player?.stats || EMPTY_OBJECT;

  const gp = safeNum(pickFirstDefined(s.gp, s.games_played, s.gamesPlayed, player?.gp), 0);
  const goals = safeNum(pickFirstDefined(s.g, s.goals, player?.goals, player?.g), 0);
  const assists = safeNum(pickFirstDefined(s.a, s.assists, player?.assists, player?.a), 0);
  const points = safeNum(pickFirstDefined(s.pts, s.points, player?.points, player?.pts, goals + assists), goals + assists);
  const shots = safeNum(pickFirstDefined(s.sog, s.shots, s.shots_on_goal, s.shotsOnGoal, player?.shots), 0);
  const hits = safeNum(pickFirstDefined(s.hits, player?.hits), 0);
  const blocks = safeNum(pickFirstDefined(s.blocks, s.blocked_shots, s.blockedShots, player?.blocks), 0);
  const takeaways = safeNum(pickFirstDefined(s.takeaways, s.tk, player?.takeaways), 0);
  const giveaways = safeNum(pickFirstDefined(s.giveaways, s.gv, player?.giveaways), 0);
  const pim = safeNum(pickFirstDefined(s.pim, s.penalty_minutes, s.penaltyMinutes, player?.pim), 0);
  const plusMinus = safeNum(pickFirstDefined(s.plus_minus, s.plusMinus, s.pm, player?.plus_minus, player?.pm), 0);

  const toi = safeNum(
    pickFirstDefined(
      s.toi,
      s.average_toi,
      s.avg_toi,
      s.averageToi,
      s.avgToi,
      player?.toi,
      player?.minutes,
      player?.average_toi,
      player?.averageToi
    ),
    0
  );

  const wins = safeNum(pickFirstDefined(s.wins, s.w, player?.wins), 0);
  const losses = safeNum(pickFirstDefined(s.losses, s.l, player?.losses), 0);
  const otl = safeNum(pickFirstDefined(s.otl, s.ot, s.overtime_losses, player?.otl), 0);
  const saves = safeNum(pickFirstDefined(s.saves, player?.saves), 0);
  const shotsAgainst = safeNum(pickFirstDefined(s.shots_against, s.shotsAgainst, s.sa, player?.shots_against), 0);
  const goalsAgainst = safeNum(pickFirstDefined(s.goals_against, s.goalsAgainst, s.ga, player?.goals_against), 0);

  const rawSvPct = pickFirstDefined(s.sv_pct, s.save_pct, s.savePct, player?.sv_pct, player?.save_pct);
  const svPct =
    safeNumOrNull(rawSvPct) !== null
      ? safeNum(rawSvPct, 0)
      : shotsAgainst > 0
        ? saves / shotsAgainst
        : 0;

  const rawGaa = pickFirstDefined(s.gaa, player?.gaa);
  const goalieToiSeconds = safeNum(
    pickFirstDefined(s.toi_sec, s.time_on_ice_sec, s.toi_total_sec, player?.toi_sec, player?.time_on_ice_sec),
    0
  );
  const goalieToiMinutes =
    goalieToiSeconds > 0
      ? goalieToiSeconds / 60
      : toi > 0 && gp > 0
        ? toi * gp
        : 0;
  const gaa =
    safeNumOrNull(rawGaa) !== null
      ? safeNum(rawGaa, 0)
      : goalieToiMinutes > 0
        ? (goalsAgainst * 60) / goalieToiMinutes
        : 0;

  return {
    gp,
    g: goals,
    a: assists,
    pts: points,
    ppg: gp > 0 ? points / gp : 0,
    goalsPerGame: gp > 0 ? goals / gp : 0,
    assistsPerGame: gp > 0 ? assists / gp : 0,
    shots,
    shootingPct: shots > 0 ? goals / shots : 0,
    hits,
    blocks,
    takeaways,
    giveaways,
    pim,
    plusMinus,
    toi,
    wins,
    losses,
    otl,
    saves,
    shotsAgainst,
    goalsAgainst,
    svPct,
    gaa,
  };
}

function getRawRating(player, keys) {
  for (const key of keys) {
    const direct = normalizeRatingScale(player?.[key]);
    if (direct !== null) return direct;

    const ratings = player?.ratings || player?.attributes || player?.rating || EMPTY_OBJECT;
    const nested = normalizeRatingScale(ratings?.[key]);
    if (nested !== null) return nested;

    const normalizedKey = normalizeKey(key);
    const nestedByKey = Object.entries(ratings || {}).find(([candidate]) => normalizeKey(candidate) === normalizedKey);
    if (nestedByKey) {
      const value = normalizeRatingScale(nestedByKey[1]);
      if (value !== null) return value;
    }
  }

  return null;
}

function collectBucketRows(player, bucketKey) {
  const keys = ATTRIBUTE_BUCKETS[bucketKey] || EMPTY_ARRAY;

  return keys
    .map((key) => {
      const value = getRawRating(player, [key]);
      if (value === null) return null;

      return {
        id: key,
        label: key
          .replace(/_/g, " ")
          .replace(/\b\w/g, (char) => char.toUpperCase()),
        value,
      };
    })
    .filter(Boolean);
}

function averageRows(rows) {
  if (!Array.isArray(rows) || !rows.length) return 0;
  return rows.reduce((sum, row) => sum + safeNum(row.value, 0), 0) / rows.length;
}

function normalizeRatingGroups(player) {
  if (Array.isArray(player?.rating_groups) && player.rating_groups.length) {
    return player.rating_groups.map((group, groupIndex) => ({
      key: normalizeKey(group?.title || group?.key || `group_${groupIndex}`),
      title: safeStr(group?.title || group?.label || `Group ${groupIndex + 1}`, "Attributes"),
      rows: Array.isArray(group?.rows)
        ? group.rows.map((row, rowIndex) => ({
            id: safeStr(row?.id || row?.key || `${groupIndex}-${rowIndex}`, `${groupIndex}-${rowIndex}`),
            label: safeStr(row?.label || row?.name || row?.id, "Rating"),
            value: normalizeRatingScale(pickFirstDefined(row?.value, row?.v, row?.rating, row?.score)) || 0,
          }))
        : [],
    }));
  }

  const pos = normalizePosition(player?.position || player?.pos);

  if (pos === "G") {
    return [
      {
        key: "technical",
        title: "Technical",
        rows: collectBucketRows(player, "GOALIE_TECHNICAL"),
      },
      {
        key: "athletic",
        title: "Athletic",
        rows: collectBucketRows(player, "GOALIE_ATHLETIC"),
      },
      {
        key: "mental",
        title: "Mental",
        rows: collectBucketRows(player, "GOALIE_MENTAL"),
      },
      {
        key: "puck",
        title: "Puck Play",
        rows: collectBucketRows(player, "GOALIE_PUCK"),
      },
    ];
  }

  return [
    {
      key: "offense",
      title: "Offense",
      rows: collectBucketRows(player, "SKATER_OFFENSE"),
    },
    {
      key: "defense",
      title: "Defense",
      rows: collectBucketRows(player, "SKATER_DEFENSE"),
    },
    {
      key: "skating",
      title: "Skating",
      rows: collectBucketRows(player, "SKATER_SKATING"),
    },
    {
      key: "physical",
      title: "Physical",
      rows: collectBucketRows(player, "SKATER_PHYSICAL"),
    },
    {
      key: "mental",
      title: "Mental",
      rows: collectBucketRows(player, "SKATER_MENTAL"),
    },
  ];
}

function buildRatingSummary(groups) {
  const summary = {};

  (groups || EMPTY_ARRAY).forEach((group) => {
    const key = normalizeKey(group.key || group.title);
    summary[key] = averageRows(group.rows);
  });

  return summary;
}

function countKnownRatings(groups) {
  return (groups || EMPTY_ARRAY).reduce((sum, group) => {
    return sum + (Array.isArray(group.rows) ? group.rows.length : 0);
  }, 0);
}

function getExplicitOverall(player) {
  const universal = getUniversalOverall(player);
  if (universal > 0) return universal;
  return safeNumOrNull(
    pickFirstDefined(
      player?.true_overall,
      player?.trueOverall,
      player?.overall,
      player?.ovr,
      player?.rating,
      player?.display_overall,
      player?.displayOverall
    )
  );
}

function buildOverallFromRatings(player, ratingSummary, seasonStats) {
  const pos = normalizePosition(player?.position || player?.pos);
  const explicitOverall = getExplicitOverall(player);
  const knownRatingCount = countKnownRatings(player?.rating_groups_normalized || EMPTY_ARRAY);

  const weights = POSITION_WEIGHTS[pos] || POSITION_WEIGHTS[getPositionClass(pos).toUpperCase()] || POSITION_WEIGHTS.F;

  let weighted = 0;
  let usedWeight = 0;

  Object.entries(weights).forEach(([key, weight]) => {
    const value = safeNumOrNull(ratingSummary[key]);

    if (value !== null && value > 0) {
      weighted += value * weight;
      usedWeight += weight;
    }
  });

  const ratingBased = usedWeight > 0 ? weighted / usedWeight : null;

  const productionBoost = buildProductionAdjustment(player, seasonStats);
  const healthAdjustment = buildHealthAdjustment(player);
  const moraleAdjustment = buildMoraleAdjustment(player);
  const fatigueAdjustment = buildFatigueAdjustment(player);

  let base;

  if (ratingBased !== null && explicitOverall !== null) {
    base = ratingBased * 0.65 + explicitOverall * 0.35;
  } else if (ratingBased !== null) {
    base = ratingBased;
  } else if (explicitOverall !== null) {
    base = explicitOverall;
  } else {
    base = 0;
  }

  const trueOverall =
    base > 0
      ? clamp(base + productionBoost + healthAdjustment + moraleAdjustment + fatigueAdjustment, 30, 99)
      : 0;

  // Universal display OVR = backend effective/ovr. Never remix with production.
  const universalDisplay = getUniversalOverall(player);
  const displayOverall =
    universalDisplay > 0
      ? universalDisplay
      : trueOverall > 0
        ? round0(trueOverall)
        : explicitOverall || 0;

  const confidence =
    knownRatingCount >= 15
      ? "High"
      : knownRatingCount >= 8
        ? "Medium"
        : explicitOverall !== null || universalDisplay > 0
          ? "Backend"
          : "Low";

  return {
    explicitOverall: explicitOverall || 0,
    calculatedOverall: ratingBased || 0,
    trueOverall: displayOverall || trueOverall,
    displayOverall,
    baseOverall: getBaseOverall(player) || displayOverall,
    overallDrop: getOverallDrop(player),
    confidence,
    knownRatingCount,
    adjustments: {
      production: productionBoost,
      health: healthAdjustment,
      morale: moraleAdjustment,
      fatigue: fatigueAdjustment,
    },
  };
}

function buildProductionAdjustment(player, stats) {
  const pos = normalizePosition(player?.position || player?.pos);
  const gp = safeNum(stats?.gp, 0);

  if (gp <= 0) return 0;

  if (pos === "G") {
    const svPct = safeNum(stats?.svPct, 0);
    const gaa = safeNum(stats?.gaa, 0);

    let boost = 0;

    if (svPct >= 0.925) boost += 2.4;
    else if (svPct >= 0.915) boost += 1.4;
    else if (svPct >= 0.905) boost += 0.4;
    else if (svPct > 0 && svPct < 0.895) boost -= 1.4;

    if (gaa > 0 && gaa <= 2.25) boost += 1.2;
    else if (gaa >= 3.3) boost -= 1.2;

    return clamp(boost, -3, 3);
  }

  const ppg = safeNum(stats?.ppg, 0);
  const plusMinus = safeNum(stats?.plusMinus, 0);
  const toi = safeNum(stats?.toi, 0);

  let boost = 0;

  if (ppg >= 1.05) boost += 2.4;
  else if (ppg >= 0.82) boost += 1.4;
  else if (ppg >= 0.6) boost += 0.6;
  else if (ppg <= 0.18 && gp >= 15) boost -= 1;

  if (isDefensePosition(pos)) {
    if (toi >= 23) boost += 1.1;
    else if (toi >= 20) boost += 0.6;
    else if (toi > 0 && toi < 13) boost -= 0.5;

    if (plusMinus >= 12) boost += 0.8;
    else if (plusMinus <= -12) boost -= 0.8;
  } else {
    if (toi >= 20) boost += 0.9;
    else if (toi >= 17) boost += 0.5;
    else if (toi > 0 && toi < 10) boost -= 0.5;
  }

  return clamp(boost, -3, 3);
}

function buildHealthAdjustment(player) {
  const status = normalizeHealth(player);
  if (!status.isInjured) return 0;

  if (status.gamesRemaining >= 25) return -3;
  if (status.gamesRemaining >= 10) return -2;
  if (status.gamesRemaining >= 3) return -1;

  return -0.5;
}

function buildMoraleAdjustment(player) {
  const morale = normalizePercentScale(player?.morale, 50);

  if (morale >= 90) return 1;
  if (morale >= 75) return 0.5;
  if (morale <= 25) return -1.2;
  if (morale <= 40) return -0.6;

  return 0;
}

function buildFatigueAdjustment(player) {
  const fatigue = normalizePercentScale(player?.fatigue, 0);

  if (fatigue >= 80) return -1.5;
  if (fatigue >= 65) return -0.9;
  if (fatigue >= 50) return -0.4;

  return 0;
}

function normalizeHealth(player) {
  const gamesRemaining = Math.max(
    0,
    safeNum(
      pickFirstDefined(
        player?.injury_games_remaining,
        player?.games_remaining,
        player?.days_remaining,
        player?.gamesRemaining,
        player?.injury?.games_remaining,
        player?.injury?.gamesRemaining
      ),
      0
    )
  );

  const rawStatus = safeStr(
    pickFirstDefined(
      player?.injury_status,
      player?.health_status,
      player?.availability_status,
      player?.status,
      player?.injury?.status
    ),
    ""
  );

  const injuryLabel = safeStr(
    pickFirstDefined(
      player?.injury_label,
      player?.injury_type,
      player?.injuryType,
      player?.injury,
      player?.injury?.label,
      player?.injury?.type,
      player?.injury?.description
    ),
    ""
  );

  const statusKey = normalizeKey(rawStatus);
  const injuryKey = normalizeKey(injuryLabel);

  const injuredByFlag =
    player?.is_injured === true ||
    player?.injured === true ||
    player?.isInjured === true ||
    gamesRemaining > 0;

  const injuredByText =
    Boolean(injuryKey) &&
    !["healthy", "none", "available", "active"].includes(injuryKey);

  const isInjured =
    injuredByFlag ||
    injuredByText ||
    statusKey.includes("injured") ||
    statusKey.includes("out") ||
    statusKey.includes("day_to_day") ||
    statusKey.includes("ltir");

  const isDayToDay =
    statusKey.includes("day") ||
    injuryKey.includes("day") ||
    rawStatus.toLowerCase().includes("day-to-day");

  const isLTIR =
    statusKey.includes("ltir") ||
    injuryKey.includes("ltir") ||
    rawStatus.toLowerCase().includes("ltir") ||
    injuryLabel.toLowerCase().includes("ltir");

  let label = "Healthy";

  if (isLTIR) {
    label = gamesRemaining > 0 ? `LTIR · ${gamesRemaining}g` : "LTIR";
  } else if (isDayToDay) {
    label = gamesRemaining > 0 ? `Day-to-day · ${gamesRemaining}g` : "Day-to-day";
  } else if (isInjured) {
    label = injuryLabel && injuryLabel !== "—"
      ? gamesRemaining > 0
        ? `${injuryLabel} · ${gamesRemaining}g`
        : injuryLabel
      : gamesRemaining > 0
        ? `Out · ${gamesRemaining}g`
        : "Injured";
  }

  return {
    isInjured,
    isDayToDay,
    isLTIR,
    gamesRemaining,
    label,
    rawStatus,
    injuryLabel,
  };
}

function normalizeRosterStatus(player, league) {
  const health = normalizeHealth(player);

  if (health.isInjured) return "Injured";
  if (player?.scratched === true || player?.is_scratched === true || player?.isScratched === true) return "Scratched";

  const signed = normalizeContract(player).isSigned;
  const poolStatus = safeStr(
    pickFirstDefined(player?.roster_status, player?.rosterStatus, player?.status),
    ""
  );

  const key = normalizeKey(poolStatus);

  if (key.includes("draft")) return "Draft Eligible";
  if (key.includes("unsigned") || !signed) {
    const source = normalizeKey(player?._source || player?.source || "");
    if (source.includes("draft")) return "Draft Eligible";
    if (source.includes("free")) return "Unsigned";
  }

  const lg = safeStr(league, "NHL").toUpperCase();

  if (lg && lg !== "NHL") return "Assigned";

  return "Active";
}

function getMoraleBand(morale) {
  const value = normalizePercentScale(morale, 50);

  if (value >= 85) return { label: "Excellent", tone: "good" };
  if (value >= 70) return { label: "Good", tone: "good" };
  if (value >= 55) return { label: "Stable", tone: "neutral" };
  if (value >= 40) return { label: "Shaky", tone: "warn" };

  return { label: "Poor", tone: "bad" };
}

function getFatigueBand(fatigue) {
  const value = normalizePercentScale(fatigue, 0);

  if (value <= 15) return { label: "Fresh", tone: "good" };
  if (value <= 35) return { label: "Managed", tone: "neutral" };
  if (value <= 60) return { label: "Heavy", tone: "warn" };

  return { label: "Exhausted", tone: "bad" };
}

function getHealthBand(player) {
  const health = normalizeHealth(player);

  if (!health.isInjured) return { label: "Healthy", tone: "good" };
  if (health.isDayToDay) return { label: health.label, tone: "warn" };

  return { label: health.label, tone: "bad" };
}

function getDevelopmentStage(player) {
  const age = safeNum(player?.age, 0);
  const trueOverall = safeNum(player?.trueOverall, safeNum(player?.ovr, 0));
  const potentialScore = safeNum(player?.potentialScore, 0);
  const growth = safeNum(player?.growth, 0);

  if (age <= 20) return "Early Prospect";
  if (age <= 23 && potentialScore >= 75) return "High-Upside Prospect";
  if (age <= 24 && trueOverall < 78) return "Developing";
  if (age <= 28 && trueOverall >= 82) return "Prime Core";
  if (age <= 29) return "Prime";
  if (age <= 32 && growth >= -0.4) return "Veteran Prime";
  if (age <= 35) return "Veteran";
  return "Late Career";
}

function getDevelopmentBand(player) {
  const growth = safeNum(player?.growth, 0);
  const age = safeNum(player?.age, 0);
  const potentialScore = safeNum(player?.potentialScore, 0);

  if (growth >= 1.5) return { label: "Surging", tone: "good" };
  if (growth >= 0.4) return { label: "Trending Up", tone: "good" };
  if (growth <= -1.2) return { label: "Regression Risk", tone: "bad" };
  if (growth <= -0.4) return { label: "Slight Decline", tone: "warn" };
  if (age <= 23 && potentialScore >= 80) return { label: "Patience Required", tone: "good" };

  return { label: "Stable", tone: "neutral" };
}

function inferGrowth(player) {
  return safeNum(
    pickFirstDefined(
      player?.growth_delta,
      player?.growthDelta,
      player?.dev_delta,
      player?.devDelta,
      player?.overall_delta,
      player?.overallDelta,
      player?.delta_ovr,
      player?.deltaOvr
    ),
    0
  );
}

function inferLeague(player, fallback = "NHL") {
  return safeStr(
    pickFirstDefined(
      player?.league,
      player?.league_name,
      player?.leagueName,
      player?.league_code,
      player?.leagueCode,
      player?.level
    ),
    fallback
  ).toUpperCase();
}

function inferTeamName(player, franchiseState) {
  return safeStr(
    pickFirstDefined(
      player?.team_name,
      player?.teamName,
      player?.team,
      player?.team_abbr,
      player?.teamAbbr,
      franchiseState?.team?.name,
      franchiseState?.team?.abbr,
      franchiseState?.team?.abbreviation
    ),
    "—"
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
  BEL: "BE",
  BELGIUM: "BE",
  Belgium: "BE",
  NLD: "NL",
  NETHERLANDS: "NL",
  Netherlands: "NL",
  ITA: "IT",
  ITALY: "IT",
  Italy: "IT",
  SVN: "SI",
  SLOVENIA: "SI",
  Slovenia: "SI",
  HRV: "HR",
  CROATIA: "HR",
  Croatia: "HR",
  EST: "EE",
  ESTONIA: "EE",
  Estonia: "EE",
  LTU: "LT",
  LITHUANIA: "LT",
  Lithuania: "LT",
  ROU: "RO",
  ROMANIA: "RO",
  Romania: "RO",
  BGR: "BG",
  BULGARIA: "BG",
  Bulgaria: "BG",
  HUN: "HU",
  HUNGARY: "HU",
  Hungary: "HU",
  ISL: "IS",
  ICELAND: "IS",
  Iceland: "IS",
  IRL: "IE",
  IRELAND: "IE",
  Ireland: "IE",
  AUS: "AU",
  AUSTRALIA: "AU",
  Australia: "AU",
  CHN: "CN",
  CHINA: "CN",
  China: "CN",
  KOR: "KR",
  "SOUTH KOREA": "KR",
  "South Korea": "KR",
  MEX: "MX",
  MEXICO: "MX",
  Mexico: "MX",
};

function resolveCountryCode(raw) {
  const text = String(raw || "").trim();
  if (!text || text === "—") return null;
  if (/^[A-Za-z]{2}$/.test(text)) return text.toUpperCase();

  const upper = text.toUpperCase();
  if (COUNTRY_NAME_TO_ISO[upper]) return COUNTRY_NAME_TO_ISO[upper];
  if (COUNTRY_NAME_TO_ISO[text]) return COUNTRY_NAME_TO_ISO[text];

  const nat = nationalityCode(text);
  if (nat && COUNTRY_NAME_TO_ISO[nat]) return COUNTRY_NAME_TO_ISO[nat];

  return null;
}

function normalizeRosterCountryCode(player) {
  if (!player) return null;

  const enriched = ensurePlayerHeadshotFields(player);
  const candidates = [
    enriched.country_code,
    enriched.countryCode,
    enriched.nationality_code,
    enriched.nationalityCode,
    enriched.nat,
    enriched.nationality,
    enriched.nation,
    enriched.country,
    enriched.birth_country,
    enriched.birthCountry,
    player.country_code,
    player.countryCode,
    player.nationality_code,
    player.nationalityCode,
    player.nat,
    player.nationality,
    player.nation,
    player.country,
    player.birth_country,
    player.birthCountry,
  ];

  for (let i = 0; i < candidates.length; i += 1) {
    const code = resolveCountryCode(candidates[i]);
    if (code) return code;
  }

  return null;
}

function flagApiUrl(countryCode, size = 64, style = "flat") {
  const iso2 = resolveCountryCode(countryCode) || (/^[A-Za-z]{2}$/.test(String(countryCode || "")) ? String(countryCode).toUpperCase() : null);
  if (!iso2) return null;
  return `https://flagsapi.com/${iso2}/${style}/${size}.png`;
}

function resolveRosterFlagLabel(player) {
  if (!player) return null;

  const enriched = ensurePlayerHeadshotFields(player);
  const code = nationalityCode(
    pickFirstDefined(
      enriched.nationality_code,
      enriched.nat,
      enriched.nationality,
      enriched.country,
      player.nat,
      player.nationality,
      player.country
    ) || ""
  );

  if (code) return code;

  return normalizeRosterCountryCode(enriched);
}

function rosterFlagUrl(player, size = 64, style = "flat") {
  const code = normalizeRosterCountryCode(player);
  return flagApiUrl(code, size, style);
}

function inferHandedness(player) {
  const position = normalizePosition(pickFirstDefined(player?.position, player?.pos));
  const isGoalie = isGoaliePosition(position);

  if (isGoalie) {
    const catches = safeStr(
      pickFirstDefined(player?.catches, player?.catch_hand, player?.catchHand, player?.catching_hand),
      ""
    );
    if (catches && catches !== "—") return catches;
  } else {
    const shoots = safeStr(
      pickFirstDefined(player?.shoots, player?.shoot_hand, player?.shootHand, player?.shooting_hand),
      ""
    );
    if (shoots && shoots !== "—") return shoots;
  }

  const generic = safeStr(
    pickFirstDefined(player?.handedness, player?.hand, player?.shoots, player?.catches),
    ""
  );

  return generic && generic !== "—" ? generic : "";
}

function resolveExplicitRole(player) {
  const raw = safeStr(
    pickFirstDefined(
      player?.line_role,
      player?.lineRole,
      player?.roster_role,
      player?.rosterRole,
      player?.depth_role,
      player?.depthRole
    ),
    ""
  );

  if (raw && raw !== "—") {
    return raw.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }

  return null;
}

function resolveExplicitSpecialTeams(player) {
  const raw = safeStr(
    pickFirstDefined(
      player?.special_teams,
      player?.specialTeams,
      player?.special_teams_role,
      player?.specialTeamsRole,
      player?.st_role,
      player?.stRole
    ),
    ""
  );

  return raw && raw !== "—" ? raw : null;
}

function resolveExplicitMinutes(player) {
  return safeNumOrNull(
    pickFirstDefined(
      player?.average_toi,
      player?.averageToi,
      player?.avg_toi,
      player?.avgToi,
      player?.toi,
      player?.minutes
    )
  );
}

function displayOverallValue(player) {
  const ovr = getUniversalOverall(player);
  return ovr > 0 ? ovr : "—";
}

function formatHeightDisplay(raw) {
  const h = safeStr(raw, "");
  if (!h || h === "—") return "—";
  if (h.includes("'") && !h.includes('"') && /\d'$/.test(h.trim())) return `${h}"`;
  return h;
}

function formatWeightDisplay(raw) {
  const w = safeStr(raw, "");
  if (!w || w === "—") return "—";
  if (/^\d+$/.test(w.trim())) return `${w} lb`;
  if (!/lb/i.test(w)) return `${w} lb`;
  return w;
}

function formatHandLabel(player) {
  const position = normalizePosition(pickFirstDefined(player?.position, player?.pos));
  const hand = inferHandedness(player);
  if (!hand) return "—";

  const clean = hand
    .replace(/^shoots[:\s.]*/i, "")
    .replace(/^catches[:\s.]*/i, "")
    .replace(/^\./, "")
    .trim();

  const prefix = isGoaliePosition(position) ? "Catches" : "Shoots";
  return `${prefix} ${clean}`;
}

function formatContractStatus(contract) {
  if (!contract) return "—";
  if (!contract.isSigned) return "Unsigned";
  if (contract.term > 0) return `Signed · ${contract.term} yr`;
  if (contract.capHit > 0) return "Signed";
  return contract.type || "—";
}

function formatContractExpiry(contract) {
  if (!contract) return "—";
  if (!contract.isSigned) return "—";
  const exp = safeStr(contract.expiry, "");
  if (exp && exp.toLowerCase() !== "unsigned" && exp.toLowerCase() !== "no contract") return exp;
  if (contract.term > 0) return `${contract.term} yr remaining`;
  return "—";
}

function displayStatValue(value, { allowZero = true } = {}) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return safeStr(value, "—");
  if (!allowZero && n === 0) return "—";
  return n;
}

function compactBoardStats(player) {
  if (!player) return "—";

  const stats = player.season_stats || EMPTY_OBJECT;

  if (isGoaliePosition(player.position)) {
    const gp = safeNum(stats.gp, 0);
    const wins = safeNum(stats.wins, 0);
    const losses = safeNum(stats.losses, 0);
    const otl = safeNum(stats.otl, 0);
    const sv = safeNum(stats.svPct, 0);
    const gaa = safeNum(stats.gaa, 0);

    if (!gp && !wins && !losses && !otl && !sv && !gaa) return "—";

    return `${gp} GP · ${wins}-${losses}-${otl} · ${sv ? formatDecimal(sv, 3) : "—"} SV% · ${gaa ? gaa.toFixed(2) : "—"} GAA`;
  }

  const gp = safeNum(stats.gp, 0);
  const goals = safeNum(stats.g, 0);
  const assists = safeNum(stats.a, 0);
  const points = safeNum(stats.pts, 0);

  if (!gp && !goals && !assists && !points) return "—";

  return `${gp} GP · ${goals} G · ${assists} A · ${points} PTS`;
}

function capHitDisplay(player) {
  const contract = player?.contract || EMPTY_OBJECT;
  const capHit = safeNum(contract.capHit, 0);

  if (!contract.isSigned && capHit <= 0) return "Unsigned";
  if (capHit <= 0) return "—";

  return formatMoneyMillions(capHit);
}

function potentialToneClass(score) {
  const value = safeNum(score, 0);
  if (value >= 88) return "is-elite";
  if (value >= 76) return "is-franchise";
  if (value >= 60) return "is-good";
  return "is-neutral";
}

function inferNationality(player) {
  return safeStr(
    pickFirstDefined(
      player?.nationality,
      player?.nation,
      player?.country,
      player?.nat
    ),
    "—"
  );
}

function inferHeight(player) {
  const display = safeStr(
    pickFirstDefined(player?.height_display, player?.heightDisplay),
    ""
  );
  if (display && display !== "—") return display;

  const cm = safeNum(player?.height_cm, 0);
  if (cm > 0) {
    const totalIn = Math.round(cm / 2.54);
    const ft = Math.floor(totalIn / 12);
    const inch = totalIn % 12;
    return `${ft}'${inch}"`;
  }

  return safeStr(pickFirstDefined(player?.hgt, player?.height), "—");
}

function inferWeight(player) {
  const lbs = safeNum(player?.weight, 0);
  if (lbs > 0) return `${Math.round(lbs)} lb`;

  const kg = safeNum(player?.weight_kg, 0);
  if (kg > 0) return `${Math.round(kg * 2.20462)} lb`;

  return safeStr(pickFirstDefined(player?.wgt), "—");
}

function inferRole(player, position, trueOverall) {
  const raw = safeStr(
    pickFirstDefined(
      player?.line_role,
      player?.lineRole,
      player?.role,
      player?.roster_role,
      player?.rosterRole,
      player?.depth_role,
      player?.depthRole
    ),
    ""
  );

  if (raw && raw !== "—") {
    return raw
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  const pos = normalizePosition(position);
  const ovr = safeNum(trueOverall, 0);

  if (pos === "G") {
    if (ovr >= 84) return "Starter";
    if (ovr >= 78) return "Tandem";
    if (ovr >= 72) return "Backup";
    return "Depth Goalie";
  }

  if (isDefensePosition(pos)) {
    if (ovr >= 84) return "Top Pair";
    if (ovr >= 78) return "Top 4";
    if (ovr >= 72) return "Third Pair";
    return "Depth Defense";
  }

  if (ovr >= 84) return "Top Line";
  if (ovr >= 78) return "Top 6";
  if (ovr >= 72) return "Middle 6";
  if (ovr >= 67) return "Bottom 6";

  return "Depth Forward";
}

function inferSpecialTeams(player, position, trueOverall, ratingSummary) {
  const raw = safeStr(
    pickFirstDefined(
      player?.special_teams,
      player?.specialTeams,
      player?.special_teams_role,
      player?.specialTeamsRole,
      player?.st_role,
      player?.stRole
    ),
    ""
  );

  if (raw && raw !== "—") return raw;

  const pos = normalizePosition(position);
  const offense = safeNum(ratingSummary?.offense, 0);
  const defense = safeNum(ratingSummary?.defense, 0);
  const technical = safeNum(ratingSummary?.technical, 0);
  const ovr = safeNum(trueOverall, 0);

  if (pos === "G") {
    if (ovr >= 84 || technical >= 84) return "Starter Usage";
    if (ovr >= 78) return "Tandem Usage";
    return "Backup Usage";
  }

  if (offense >= 82 && defense >= 78) return "PP + PK";
  if (offense >= 82) return "Power Play";
  if (defense >= 80) return "Penalty Kill";

  return "Even Strength";
}

function inferMinutes(player, position, trueOverall) {
  const direct = safeNumOrNull(
    pickFirstDefined(
      player?.minutes,
      player?.toi,
      player?.average_toi,
      player?.averageToi,
      player?.ice_time,
      player?.iceTime,
      player?.season_stats?.toi,
      player?.stats?.toi
    )
  );

  if (direct !== null) return direct;

  const pos = normalizePosition(position);
  const ovr = safeNum(trueOverall, 0);

  if (!ovr) return 0;

  if (pos === "G") return 0;

  if (isDefensePosition(pos)) {
    if (ovr >= 86) return 24;
    if (ovr >= 80) return 21;
    if (ovr >= 74) return 17;
    return 12;
  }

  if (ovr >= 86) return 20;
  if (ovr >= 80) return 17;
  if (ovr >= 74) return 14;
  if (ovr >= 68) return 10;

  return 7;
}

function getPotentialScoreFromRaw(raw) {
  const value = safeStr(raw, "");
  const exact = POTENTIAL_ORDER[value];

  if (exact !== undefined) return exact;

  const key = normalizeKey(value);

  if (key.includes("franchise")) return 100;
  if (key.includes("elite")) return 92;
  if (key.includes("top_pair")) return 87;
  if (key.includes("top_line")) return 86;
  if (key.includes("starter")) return 86;
  if (key.includes("top_4")) return 80;
  if (key.includes("top_6")) return 80;
  if (key.includes("tandem")) return 78;
  if (key.includes("middle_6")) return 70;
  if (key.includes("third_pair")) return 68;
  if (key.includes("backup")) return 68;
  if (key.includes("bottom_6")) return 61;
  if (key.includes("depth")) return 50;
  if (key.includes("ahl")) return 38;

  return 0;
}

function buildPotentialModel(player, overallModel, ratingSummary, seasonStats) {
  const rawPotential = pickFirstDefined(
    player?.potential,
    player?.potential_tier,
    player?.potentialTier,
    player?.ceiling,
    player?.projection,
    player?.projected_role,
    player?.projectedRole
  );

  const explicitScore = getPotentialScoreFromRaw(rawPotential);
  const pos = normalizePosition(player?.position || player?.pos);
  const age = safeNum(player?.age, 18);
  const trueOverall = safeNum(overallModel?.trueOverall, 0);
  const growth = inferGrowth(player);
  const ppg = safeNum(seasonStats?.ppg, 0);
  const gp = safeNum(seasonStats?.gp, 0);

  let score = explicitScore || 0;

  if (!score) {
    if (pos === "G") {
      score = estimateGoaliePotentialScore(age, trueOverall, growth, seasonStats, ratingSummary);
    } else if (isDefensePosition(pos)) {
      score = estimateDefensePotentialScore(age, trueOverall, growth, seasonStats, ratingSummary);
    } else {
      score = estimateForwardPotentialScore(age, trueOverall, growth, seasonStats, ratingSummary);
    }
  }

  const ageModifier = buildAgePotentialModifier(age, trueOverall, growth);
  const productionModifier = buildPotentialProductionModifier(pos, gp, ppg, seasonStats);
  const ratingModifier = buildPotentialRatingModifier(pos, ratingSummary);

  const finalScore = clamp(score + ageModifier + productionModifier + ratingModifier, 25, 100);
  const label = potentialLabelFromScore(pos, finalScore, trueOverall, age);

  return {
    rawPotential: rawPotential || "",
    potentialLabel: label,
    potentialScore: round0(finalScore),
    potentialConfidence: explicitScore ? "Backend" : overallModel?.knownRatingCount >= 8 ? "Calculated" : "Low",
    potentialBreakdown: {
      base: score,
      age: ageModifier,
      production: productionModifier,
      ratings: ratingModifier,
    },
  };
}

function estimateForwardPotentialScore(age, overall, growth, stats, ratings) {
  const offense = safeNum(ratings?.offense, 0);
  const skating = safeNum(ratings?.skating, 0);
  const mental = safeNum(ratings?.mental, 0);
  const ppg = safeNum(stats?.ppg, 0);

  let score = 50;

  if (overall >= 88) score = 90;
  else if (overall >= 84) score = 84;
  else if (overall >= 80) score = 78;
  else if (overall >= 75) score = 70;
  else if (overall >= 70) score = 60;

  if (age <= 20 && overall >= 74) score += 10;
  else if (age <= 22 && overall >= 76) score += 7;
  else if (age <= 24 && overall >= 78) score += 4;

  if (offense >= 86 || skating >= 86) score += 4;
  if (mental >= 84) score += 2;
  if (growth >= 1) score += 5;
  if (ppg >= 0.8 && age <= 25) score += 4;

  return score;
}

function estimateDefensePotentialScore(age, overall, growth, stats, ratings) {
  const defense = safeNum(ratings?.defense, 0);
  const skating = safeNum(ratings?.skating, 0);
  const mental = safeNum(ratings?.mental, 0);
  const toi = safeNum(stats?.toi, 0);

  let score = 50;

  if (overall >= 88) score = 90;
  else if (overall >= 84) score = 86;
  else if (overall >= 80) score = 80;
  else if (overall >= 75) score = 70;
  else if (overall >= 70) score = 60;

  if (age <= 21 && overall >= 73) score += 9;
  else if (age <= 23 && overall >= 76) score += 6;

  if (defense >= 86) score += 4;
  if (skating >= 84) score += 3;
  if (mental >= 84) score += 2;
  if (growth >= 1) score += 5;
  if (toi >= 20 && age <= 25) score += 4;

  return score;
}

function estimateGoaliePotentialScore(age, overall, growth, stats, ratings) {
  const technical = safeNum(ratings?.technical, 0);
  const athletic = safeNum(ratings?.athletic, 0);
  const mental = safeNum(ratings?.mental, 0);
  const svPct = safeNum(stats?.svPct, 0);

  let score = 50;

  if (overall >= 88) score = 90;
  else if (overall >= 84) score = 84;
  else if (overall >= 80) score = 78;
  else if (overall >= 75) score = 68;
  else if (overall >= 70) score = 58;

  if (age <= 22 && overall >= 72) score += 10;
  else if (age <= 25 && overall >= 75) score += 6;

  if (technical >= 85) score += 4;
  if (athletic >= 84) score += 3;
  if (mental >= 84) score += 2;
  if (growth >= 1) score += 5;
  if (svPct >= 0.915 && age <= 26) score += 4;

  return score;
}

function buildAgePotentialModifier(age, overall, growth) {
  if (age <= 20) return 6;
  if (age <= 22) return 4;
  if (age <= 24 && growth >= 0.5) return 2;
  if (age >= 31 && overall < 84) return -6;
  if (age >= 34) return -10;
  return 0;
}

function buildPotentialProductionModifier(pos, gp, ppg, stats) {
  if (gp < 10) return 0;

  if (pos === "G") {
    const svPct = safeNum(stats?.svPct, 0);
    if (svPct >= 0.92) return 4;
    if (svPct >= 0.91) return 2;
    if (svPct > 0 && svPct < 0.89) return -3;
    return 0;
  }

  if (ppg >= 1) return 4;
  if (ppg >= 0.75) return 2;
  if (ppg <= 0.15 && gp >= 20) return -2;

  return 0;
}

function buildPotentialRatingModifier(pos, ratings) {
  if (pos === "G") {
    const technical = safeNum(ratings?.technical, 0);
    const mental = safeNum(ratings?.mental, 0);
    if (technical >= 88 && mental >= 82) return 4;
    if (technical >= 84) return 2;
    return 0;
  }

  if (isDefensePosition(pos)) {
    const defense = safeNum(ratings?.defense, 0);
    const skating = safeNum(ratings?.skating, 0);
    if (defense >= 88 && skating >= 82) return 4;
    if (defense >= 84) return 2;
    return 0;
  }

  const offense = safeNum(ratings?.offense, 0);
  const skating = safeNum(ratings?.skating, 0);
  if (offense >= 88 && skating >= 82) return 4;
  if (offense >= 84 || skating >= 86) return 2;

  return 0;
}

function potentialLabelFromScore(position, score, overall, age) {
  const pos = normalizePosition(position);
  const s = safeNum(score, 0);
  const ovr = safeNum(overall, 0);

  if (s >= 96 && (age <= 27 || ovr >= 90)) return "Franchise";
  if (s >= 88) return "Elite";

  if (pos === "G") {
    if (s >= 83) return "Starter";
    if (s >= 75) return "Tandem";
    if (s >= 65) return "Backup";
    if (s >= 48) return "Depth";
    return "AHL";
  }

  if (isDefensePosition(pos)) {
    if (s >= 84) return "Top Pair D";
    if (s >= 76) return "Top 4 D";
    if (s >= 66) return "Third Pair D";
    if (s >= 50) return "Depth";
    return "AHL";
  }

  if (s >= 84) return "Top Line";
  if (s >= 76) return "Top 6";
  if (s >= 66) return "Middle 6";
  if (s >= 55) return "Bottom 6";
  if (s >= 45) return "Depth";

  return "AHL";
}

function calculateAssetValue(player) {
  const age = safeNum(player?.age, 0);
  const ovr = safeNum(player?.trueOverall, safeNum(player?.ovr, 0));
  const potentialScore = safeNum(player?.potentialScore, 0);
  const capHit = safeNum(player?.contract?.capHit, 0);
  const term = safeNum(player?.contract?.term, 0);
  const health = normalizeHealth(player);
  const growth = safeNum(player?.growth, 0);

  let score = 0;

  score += ovr * 1.5;
  score += potentialScore * 0.75;

  if (age <= 21) score += 12;
  else if (age <= 24) score += 8;
  else if (age <= 28) score += 5;
  else if (age >= 33) score -= 8;

  if (growth >= 1) score += 6;
  else if (growth <= -1) score -= 6;

  if (capHit > 0) {
    if (ovr >= 84 && capHit <= 5) score += 8;
    else if (ovr >= 80 && capHit <= 3) score += 6;
    else if (capHit >= 9 && ovr < 86) score -= 8;
  }

  if (term >= 4 && age <= 27 && ovr >= 80) score += 5;
  if (health.isInjured && health.gamesRemaining >= 15) score -= 8;

  let label = "Depth Asset";

  if (score >= 210) label = "Franchise Cornerstone";
  else if (score >= 185) label = "Elite Core";
  else if (score >= 160) label = "Core Piece";
  else if (score >= 135) label = "Useful NHL Asset";
  else if (score >= 110) label = "Depth / Role Asset";
  else if (score >= 90) label = "Replacement Level";
  else label = "Low Value";

  return {
    score: round0(score),
    label,
  };
}

function getOVRColor(ovr) {
  const n = safeNum(ovr, 0);

  if (n >= 92) return "#f8d26a";
  if (n >= 88) return "#8fd3ff";
  if (n >= 84) return "#b9f6ca";
  if (n >= 80) return "#eef4ff";
  if (n >= 74) return "#a8b4c8";

  return "#7c879a";
}

function toneClass(tone) {
  if (tone === "good") return "is-good";
  if (tone === "warn") return "is-warn";
  if (tone === "bad") return "is-bad";
  if (tone === "medical") return "is-medical";
  if (tone === "premium") return "is-premium";
  return "is-neutral";
}

function gradeFromOverall(ovr) {
  const n = safeNum(ovr, 0);

  if (n >= 94) return "A+";
  if (n >= 90) return "A";
  if (n >= 86) return "A-";
  if (n >= 82) return "B+";
  if (n >= 78) return "B";
  if (n >= 74) return "B-";
  if (n >= 70) return "C+";
  if (n >= 66) return "C";

  return "D";
}

function buildPlayerNote(player) {
  const pieces = [];

  if (!player) return "No player selected.";

  const pos = normalizePosition(player.position);
  const ovr = safeNum(player.trueOverall, safeNum(player.ovr, 0));
  const age = safeNum(player.age, 0);
  const potential = safeStr(player.potential, "—");
  const potentialScore = safeNum(player.potentialScore, 0);
  const health = normalizeHealth(player);
  const moraleBand = getMoraleBand(player.morale);
  const fatigueBand = getFatigueBand(player.fatigue);
  const asset = player.asset || calculateAssetValue(player);

  pieces.push(`${getPositionDisplay(pos)} profile with ${ovr || "unavailable"} calculated overall.`);

  if (potentialScore >= 88) {
    pieces.push(`High-end ${potential} ceiling should be protected unless the return is massive.`);
  } else if (potentialScore >= 76) {
    pieces.push(`${potential} projection gives the player real roster-planning value.`);
  } else if (potentialScore <= 50) {
    pieces.push("Projection is mostly depth-based unless development changes.");
  }

  if (age <= 23 && potentialScore >= 75) pieces.push("Still young enough for meaningful growth.");
  if (age >= 32 && ovr < 84) pieces.push("Age curve needs to be monitored.");
  if (health.isInjured) pieces.push(`Health flag: ${health.label}.`);
  if (moraleBand.tone === "bad") pieces.push("Morale is hurting the current read.");
  if (fatigueBand.tone === "bad") pieces.push("Fatigue is high enough to affect deployment.");
  if (asset?.label) pieces.push(`Asset read: ${asset.label}.`);

  return pieces.join(" ");
}

function normalizeLivePlayer(player, franchiseState, index) {
  const enriched = enrichRosterPlayer(player, index);
  const source = {
    ...enriched,
    ...player,
  };

  const position = normalizePosition(pickFirstDefined(source.position, source.pos));
  const age = safeNum(pickFirstDefined(source.age, enriched.age), 18);
  const league = inferLeague(source, "NHL");
  const contract = normalizeContract(source);
  const seasonStats = normalizeSeasonStats(source);
  const ratingGroups = normalizeRatingGroups(source);
  const ratingSummary = buildRatingSummary(ratingGroups);

  const overallModel = buildOverallFromRatings(
    {
      ...source,
      position,
      rating_groups_normalized: ratingGroups,
    },
    ratingSummary,
    seasonStats
  );

  const potentialModel = buildPotentialModel(
    {
      ...source,
      position,
      age,
    },
    overallModel,
    ratingSummary,
    seasonStats
  );

  const normalizedBase = {
    ...source,
    _draft: false,
    _source: source._source || "live_roster",
    key: getPlayerId(source, `${getPlayerName(source)}-${index}`),
    id: getPlayerId(source, `${index}`),
    name: getPlayerName(source),
    age,
    position,
    positionClass: getPositionClass(position),
    league,
    teamName: inferTeamName(source, franchiseState),
    nat: inferNationality(source),
    nationality_code:
      resolveCountryCode(
        pickFirstDefined(
          source?.nationality_code,
          source?.nationalityCode,
          source?.country_code,
          source?.countryCode,
          source?.nationality,
          source?.country,
          source?.nat,
          source?.nation,
          source?.birth_country,
          source?.birthCountry
        )
      ) || nationalityCode(inferNationality(source) !== "—" ? inferNationality(source) : ""),
    country_code:
      resolveCountryCode(
        pickFirstDefined(source?.country_code, source?.countryCode, source?.country, source?.birth_country)
      ) || null,
    hgt: inferHeight(source),
    wgt: inferWeight(source),
    hand: inferHandedness(source) || "Unknown",
    archetype: normalizeArchetype(source),
    morale: normalizePercentScale(source.morale, 50),
    fatigue: normalizePercentScale(source.fatigue, 0),
    growth: inferGrowth(source),
    contract,
    season_stats: seasonStats,
    rating_groups: ratingGroups,
    rating_summary: ratingSummary,
    explicitOverall: overallModel.explicitOverall,
    calculatedOverall: overallModel.calculatedOverall,
    trueOverall: overallModel.trueOverall,
    overallConfidence: overallModel.confidence,
    overallAdjustments: overallModel.adjustments,
    knownRatingCount: overallModel.knownRatingCount,
    base_ovr: overallModel.baseOverall || overallModel.displayOverall,
    effective_ovr: overallModel.displayOverall,
    overall_drop: overallModel.overallDrop || 0,
    ovr: overallModel.displayOverall,
    overall: overallModel.displayOverall,
    potential: potentialModel.potentialLabel,
    potentialScore: potentialModel.potentialScore,
    potentialConfidence: potentialModel.potentialConfidence,
    potentialBreakdown: potentialModel.potentialBreakdown,
  };

  const status = normalizeRosterStatus(normalizedBase, league);
  const role = inferRole(source, position, normalizedBase.trueOverall);
  const specialTeams = inferSpecialTeams(source, position, normalizedBase.trueOverall, ratingSummary);
  const minutes = inferMinutes(source, position, normalizedBase.trueOverall);
  const asset = calculateAssetValue({
    ...normalizedBase,
    status,
    role,
  });

  const output = {
    ...normalizedBase,
    status,
    role,
    roleLabel: role,
    specialTeams,
    minutes,
    explicitRole: resolveExplicitRole(source),
    explicitSpecialTeams: resolveExplicitSpecialTeams(source),
    explicitMinutes: resolveExplicitMinutes(source),
    stage: getDevelopmentStage({
      ...normalizedBase,
      role,
    }),
    asset,
    assetTag: asset.label,
    displayOvr: displayOverallValue(normalizedBase),
  };

  return {
    ...output,
    note: buildPlayerNote(output),
  };
}

/** Shared with TradeHub — identical display OVR/position/role as roster board. */
export function normalizeRosterBrowserPlayer(player, franchiseState, index = 0) {
  return normalizeLivePlayer(
    {
      ...player,
      _source: player?._source || "organization",
      league: player?.league || "NHL",
    },
    franchiseState,
    index,
  );
}

function normalizeDraftPlayer(row, index) {
  const source = {
    ...row,
    _draft: true,
    _source: "draft_class",
  };

  const position = normalizePosition(source.position || source.pos);
  const age = safeNum(source.age, 18);
  const contract = {
    capHit: 0,
    salary: 0,
    term: 0,
    expiry: "Draft rights not signed",
    type: "Unsigned",
    clause: "",
    isSigned: false,
  };

  const seasonStats = normalizeSeasonStats(source);
  const ratingGroups = normalizeRatingGroups(source);
  const ratingSummary = buildRatingSummary(ratingGroups);

  const overallModel = buildOverallFromRatings(
    {
      ...source,
      position,
      rating_groups_normalized: ratingGroups,
      overall: pickFirstDefined(source.true_ovr, source.trueOverall, source.overall, source.ovr),
    },
    ratingSummary,
    seasonStats
  );

  const potentialModel = buildPotentialModel(
    {
      ...source,
      position,
      age,
    },
    overallModel,
    ratingSummary,
    seasonStats
  );

  const rank = safeNum(
    pickFirstDefined(source.rank, source.overall_rank, source.overallRank, source.draft_rank, source.draftRank),
    index + 1
  );

  const trend = normalizeDraftTrend(source);

  const normalizedBase = {
    ...source,
    key: getPlayerId(source, `draft-${rank}-${index}`),
    id: getPlayerId(source, `draft-${rank}-${index}`),
    rank,
    rank_delta: safeNum(pickFirstDefined(source.rank_delta, source.rankDelta, source.stock_change, source.stockChange), 0),
    trendText: trend.text,
    trendClass: trend.className,
    name: getPlayerName(source),
    age,
    position,
    positionClass: getPositionClass(position),
    league: safeStr(
      pickFirstDefined(source.league, source.league_name, source.leagueName, source.league_code, source.leagueCode),
      "Draft"
    ),
    teamName: "Draft Eligible",
    nat: inferNationality(source),
    hgt: inferHeight(source),
    wgt: inferWeight(source),
    hand: inferHandedness(source) || "Unknown",
    archetype: normalizeArchetype(source),
    morale: 50,
    fatigue: 0,
    growth: inferGrowth(source),
    contract,
    season_stats: seasonStats,
    rating_groups: ratingGroups,
    rating_summary: ratingSummary,
    explicitOverall: overallModel.explicitOverall,
    calculatedOverall: overallModel.calculatedOverall,
    trueOverall: overallModel.trueOverall,
    overallConfidence: overallModel.confidence,
    overallAdjustments: overallModel.adjustments,
    knownRatingCount: overallModel.knownRatingCount,
    ovr: overallModel.displayOverall,
    true_ovr: safeNum(pickFirstDefined(source.true_ovr, source.trueOverall, overallModel.displayOverall), 0),
    scout_grade: safeStr(pickFirstDefined(source.scout_grade, source.scoutGrade, source.grade), "—"),
    scout_tier: safeStr(pickFirstDefined(source.scout_tier, source.scoutTier, source.tier), "—"),
    potential: potentialModel.potentialLabel,
    potentialScore: potentialModel.potentialScore,
    potentialConfidence: potentialModel.potentialConfidence,
    potentialBreakdown: potentialModel.potentialBreakdown,
    status: "Draft Eligible",
    role: "Prospect",
    roleLabel: "Prospect",
    specialTeams: "Development",
    minutes: 0,
  };

  const asset = calculateAssetValue(normalizedBase);

  return {
    ...normalizedBase,
    asset,
    assetTag: asset.label,
    stage: getDevelopmentStage(normalizedBase),
    note:
      safeStr(source.notes, "") && safeStr(source.notes, "") !== "—"
        ? safeStr(source.notes, "")
        : buildPlayerNote({
            ...normalizedBase,
            asset,
          }),
  };
}

function normalizeDraftTrend(row) {
  const trend = safeStr(pickFirstDefined(row?.trend, row?.stock_trend, row?.stockTrend), "SAME").toUpperCase();
  const delta = safeNum(pickFirstDefined(row?.rank_delta, row?.rankDelta, row?.stock_change, row?.stockChange), 0);

  if (trend === "UP" || delta > 0) {
    return {
      text: `▲${Math.abs(delta) || ""}`,
      className: "is-up",
    };
  }

  if (trend === "DOWN" || delta < 0) {
    return {
      text: `▼${Math.abs(delta) || ""}`,
      className: "is-down",
    };
  }

  if (trend === "NEW") {
    return {
      text: "NEW",
      className: "is-new",
    };
  }

  return {
    text: "—",
    className: "is-flat",
  };
}

function comparePlayers(a, b, sortKey) {
  const nameA = safeStr(a?.name, "").toLowerCase();
  const nameB = safeStr(b?.name, "").toLowerCase();

  const get = (player, path, fallback = 0) => {
    if (!player) return fallback;
    const parts = String(path).split(".");
    let cur = player;

    for (const part of parts) {
      cur = cur?.[part];
      if (cur === undefined || cur === null) return fallback;
    }

    return safeNum(cur, fallback);
  };

  switch (sortKey) {
    case "overall_asc":
      return get(a, "ovr") - get(b, "ovr") || get(a, "age") - get(b, "age");

    case "overall_desc":
      return get(b, "ovr") - get(a, "ovr") || get(a, "age") - get(b, "age");

    case "true_overall_desc":
      return get(b, "trueOverall") - get(a, "trueOverall") || get(b, "ovr") - get(a, "ovr");

    case "potential_score_desc":
      return get(b, "potentialScore") - get(a, "potentialScore") || get(a, "age") - get(b, "age");

    case "age_asc":
      return get(a, "age") - get(b, "age") || get(b, "ovr") - get(a, "ovr");

    case "age_desc":
      return get(b, "age") - get(a, "age") || get(b, "ovr") - get(a, "ovr");

    case "name_desc":
      return nameB.localeCompare(nameA);

    case "name_asc":
      return nameA.localeCompare(nameB);

    case "points_desc":
      return get(b, "season_stats.pts") - get(a, "season_stats.pts") || get(b, "season_stats.g") - get(a, "season_stats.g");

    case "goals_desc":
      return get(b, "season_stats.g") - get(a, "season_stats.g") || get(b, "season_stats.pts") - get(a, "season_stats.pts");

    case "assists_desc":
      return get(b, "season_stats.a") - get(a, "season_stats.a") || get(b, "season_stats.pts") - get(a, "season_stats.pts");

    case "ppg_desc":
      return get(b, "season_stats.ppg") - get(a, "season_stats.ppg") || get(b, "season_stats.pts") - get(a, "season_stats.pts");

    case "morale_desc":
      return get(b, "morale") - get(a, "morale") || get(b, "ovr") - get(a, "ovr");

    case "fatigue_desc":
      return get(b, "fatigue") - get(a, "fatigue") || get(b, "ovr") - get(a, "ovr");

    case "salary_desc":
      return get(b, "contract.capHit") - get(a, "contract.capHit") || get(b, "ovr") - get(a, "ovr");

    case "term_desc":
      return get(b, "contract.term") - get(a, "contract.term") || get(b, "contract.capHit") - get(a, "contract.capHit");

    case "asset_value_desc":
      return get(b, "asset.score") - get(a, "asset.score") || get(b, "potentialScore") - get(a, "potentialScore");

    default:
      return get(b, "ovr") - get(a, "ovr") || get(a, "age") - get(b, "age");
  }
}

function statLineForPlayer(player) {
  if (!player) return "—";

  const stats = player.season_stats || EMPTY_OBJECT;
  const pos = normalizePosition(player.position);

  if (pos === "G") {
    const wins = safeNum(stats.wins, 0);
    const losses = safeNum(stats.losses, 0);
    const otl = safeNum(stats.otl, 0);
    const sv = safeNum(stats.svPct, 0);
    const gaa = safeNum(stats.gaa, 0);

    if (!wins && !losses && !otl && !sv && !gaa) return "Stats unavailable";

    return `${wins}-${losses}-${otl} · ${formatDecimal(sv, 3)} SV% · ${gaa ? gaa.toFixed(2) : "—"} GAA`;
  }

  const gp = safeNum(stats.gp, 0);
  const goals = safeNum(stats.g, 0);
  const assists = safeNum(stats.a, 0);
  const points = safeNum(stats.pts, 0);

  if (!gp && !goals && !assists && !points) return "Stats unavailable";

  return `${gp} GP · ${goals} G · ${assists} A · ${points} PTS · ${stats.ppg ? stats.ppg.toFixed(2) : "0.00"} P/GP`;
}

function buildRosterWarnings(players, capInfo) {
  const nhlPlayers = (players || EMPTY_ARRAY).filter((p) => p.league === "NHL");
  const activeNhl = nhlPlayers.filter((p) => p.status === "Active" || p.status === "Scratched");

  const forwards = activeNhl.filter((p) => isForwardPosition(p.position));
  const defense = activeNhl.filter((p) => isDefensePosition(p.position));
  const goalies = activeNhl.filter((p) => isGoaliePosition(p.position));
  const injured = nhlPlayers.filter((p) => p.status === "Injured");

  const warnings = [];

  if (activeNhl.length > NHL_ACTIVE_ROSTER_LIMIT) {
    warnings.push({
      key: "active_limit",
      tone: "bad",
      title: "Active roster over limit",
      body: `${activeNhl.length}/${NHL_ACTIVE_ROSTER_LIMIT} active NHL players.`,
    });
  }

  if (forwards.length < 12) {
    warnings.push({
      key: "forwards_low",
      tone: "warn",
      title: "Forward group thin",
      body: `${forwards.length}/12 active forwards.`,
    });
  }

  if (defense.length < 6) {
    warnings.push({
      key: "defense_low",
      tone: "warn",
      title: "Defense group thin",
      body: `${defense.length}/6 active defensemen.`,
    });
  }

  if (goalies.length < 2) {
    warnings.push({
      key: "goalies_low",
      tone: "bad",
      title: "Goalie coverage problem",
      body: `${goalies.length}/2 active goalies.`,
    });
  }

  if (injured.length > 0) {
    warnings.push({
      key: "injuries",
      tone: "medical",
      title: "Injury impact",
      body: `${injured.length} NHL player${injured.length === 1 ? "" : "s"} currently injured.`,
    });
  }

  if (capInfo && safeNum(capInfo.capSpace, 0) < 0) {
    warnings.push({
      key: "cap_over",
      tone: "bad",
      title: "Over the cap",
      body: `${formatMoneyMillions(Math.abs(capInfo.capSpace))} over the limit.`,
    });
  }

  return warnings;
}

function buildLineGroups(players) {
  const nhl = (players || EMPTY_ARRAY)
    .filter((p) => p.league === "NHL")
    .filter((p) => p.status !== "Injured")
    .sort((a, b) => safeNum(b.trueOverall, b.ovr) - safeNum(a.trueOverall, a.ovr));

  const centers = nhl.filter((p) => normalizePosition(p.position) === "C");
  const leftWings = nhl.filter((p) => normalizePosition(p.position) === "LW");
  const rightWings = nhl.filter((p) => normalizePosition(p.position) === "RW");
  const flexibleForwards = nhl.filter((p) => normalizePosition(p.position) === "F");
  const defense = nhl.filter((p) => isDefensePosition(p.position));
  const leftD = defense.filter((p) => normalizePosition(p.position) === "LD");
  const rightD = defense.filter((p) => normalizePosition(p.position) === "RD");
  const genericD = defense.filter((p) => normalizePosition(p.position) === "D");
  const goalies = nhl.filter((p) => isGoaliePosition(p.position));

  const take = (list) => list.shift() || null;

  const forwardLines = [];

  for (let i = 0; i < 4; i += 1) {
    const lw = take(leftWings) || take(flexibleForwards) || take(centers) || take(rightWings);
    const c = take(centers) || take(flexibleForwards) || take(leftWings) || take(rightWings);
    const rw = take(rightWings) || take(flexibleForwards) || take(centers) || take(leftWings);

    forwardLines.push([lw, c, rw].filter(Boolean));
  }

  const defensePairs = [];

  for (let i = 0; i < 3; i += 1) {
    const ld = take(leftD) || take(genericD) || take(rightD);
    const rd = take(rightD) || take(genericD) || take(leftD);

    defensePairs.push([ld, rd].filter(Boolean));
  }

  return {
    forwards: forwardLines,
    defense: defensePairs,
    goalies: goalies.slice(0, 2),
    extras: {
      forwards: [...leftWings, ...centers, ...rightWings, ...flexibleForwards],
      defense: [...leftD, ...rightD, ...genericD],
      goalies: goalies.slice(2),
    },
  };
}
function MiniBadge({ text, tone = "neutral", title = "" }) {
  return (
    <span className={`nhlrost-mini-badge ${toneClass(tone)}`} title={title || text}>
      {text || "—"}
    </span>
  );
}

function RatingPill({ label, value, tone = "neutral" }) {
  const numeric = safeNum(value, 0);

  return (
    <article className={`nhlrost-rating-pill ${toneClass(tone)}`}>
      <span>{label}</span>
      <strong>{numeric ? round0(numeric) : "—"}</strong>
    </article>
  );
}

function ProgressBar({ label, value, max = 100, tone = "neutral", suffix = "" }) {
  const numeric = safeNum(value, 0);
  const pct = clamp((numeric / max) * 100, 0, 100);

  return (
    <div className={`nhlrost-progress ${toneClass(tone)}`}>
      <div className="nhlrost-progress__top">
        <span>{label}</span>
        <strong>
          {numeric ? round1(numeric) : "—"}
          {suffix}
        </strong>
      </div>
      <div className="nhlrost-progress__track">
        <span style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function InfoPair({ label, value, tone = "neutral" }) {
  return (
    <div className={`nhlrost-info-pair ${toneClass(tone)}`}>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function ToolbarSelect({ id, label, value, onChange, options, disabled = false, compact = false }) {
  return (
    <label className={`nhlrost-control ${compact ? "nhlrost-control--compact" : ""}`} htmlFor={id}>
      {!compact ? <span>{label}</span> : null}
      <select
        id={id}
        value={value}
        onChange={onChange}
        disabled={disabled}
        aria-label={compact ? label : undefined}
      >
        {(options || EMPTY_ARRAY).map((option) => {
          const optValue = option?.value ?? option;
          const optLabel = option?.label ?? option;

          return (
            <option key={String(optValue)} value={optValue}>
              {optLabel}
            </option>
          );
        })}
      </select>
    </label>
  );
}

function ToolbarInput({ id, label, value, onChange, placeholder, compact = false }) {
  return (
    <label className={`nhlrost-control ${compact ? "nhlrost-control--compact" : ""}`} htmlFor={id}>
      {!compact ? <span>{label}</span> : null}
      <input
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        aria-label={compact ? label : undefined}
      />
    </label>
  );
}

function ConnectedActionNotice({ title, body, tone = "neutral" }) {
  return (
    <article className={`nhlrost-action-notice ${toneClass(tone)}`}>
      <strong>{title}</strong>
      <p>{body}</p>
    </article>
  );
}

function EmptyPanel({ title = "No data", body = "Nothing is available for this view yet.", compact = false }) {
  return (
    <section className={`nhlrost-empty-panel ${compact ? "is-compact" : ""}`}>
      {!compact ? <div className="nhlrost-empty-panel__orb">◌</div> : null}
      <h3>{title}</h3>
      <p>{body}</p>
    </section>
  );
}

function PlayerAvatar({ player, size = "md" }) {
  const enriched = ensurePlayerHeadshotFields(player || {});
  const rosterSize = size === "sm" ? "xs" : size;
  const flagCode =
    enriched.nationality_code ||
    nationalityCode(
      pickFirstDefined(enriched.nat, enriched.nationality, enriched.country, player?.nat) || ""
    ) ||
    resolveRosterFlagLabel(enriched);

  return (
    <PlayerHeadshot
      player={enriched}
      size={rosterSize}
      variant="card"
      className="nhlrost-headshot"
      number={player?.num}
      flag={flagCode || null}
    />
  );
}

function PlayerIconPlate({ player }) {
  const posClass = player?.positionClass || getPositionClass(player?.position) || "unknown";
  const archColor = getArchetypeColor(player?.archetype);
  const potScore = safeNum(player?.potentialScore, 0);
  const accentOpacity = potScore > 0 ? Math.min(1, 0.4 + potScore / 220) : 0.55;
  const ovr = getUniversalOverall(player);
  const ovrColor = getOVRColor(ovr);
  const posLabel = player?.position || "—";

  return (
    <div
      className={`nhlrost-player-icon-plate pos-${posClass}`}
      style={{
        "--arch-color": archColor,
        "--arch-accent-opacity": accentOpacity,
      }}
      aria-hidden="true"
    >
      <div className="nhlrost-player-icon-plate__number" title={getOverallTooltip(player)}>
        <span style={{ color: ovrColor }}>{ovr > 0 ? ovr : "—"}</span>
      </div>
      <div className="nhlrost-player-icon-plate__portrait">
        <PlayerAvatar player={player} size="lg" />
        <span className="nhlrost-player-icon-plate__pos">{posLabel}</span>
      </div>
    </div>
  );
}

function OvrStack({ player }) {
  const ovr = getUniversalOverall(player);
  const base = getBaseOverall(player) || ovr;
  const drop = getOverallDrop(player);
  const confidence = safeStr(player?.overallConfidence, "Low");
  const growth = inferGrowth(player);
  const growthRounded = Math.round(growth);

  return (
    <div className="nhlrost-ovr-stack" title={getOverallTooltip(player)}>
      <strong style={{ color: getOVRColor(ovr) }}>{ovr || "—"}</strong>
      {growthRounded !== 0 ? (
        <em
          className={`nhlrost-ovr-growth ${growthRounded > 0 ? "is-up" : "is-down"}`}
          title={`Season OVR ${growthRounded > 0 ? "+" : ""}${growthRounded} from start`}
        >
          {growthRounded > 0 ? `+${growthRounded}` : `${growthRounded}`}
        </em>
      ) : null}
      <span>{confidence}</span>
      {drop >= 1 ? (
        <em className="nhlrost-ovr-drop" title={`Down ${drop} from base ${base}`}>
          ↓{drop}
        </em>
      ) : null}
    </div>
  );
}

function PotentialStack({ player }) {
  const score = safeNum(player?.potentialScore, 0);
  const label = safeStr(player?.potential, "—");
  const tone = score >= 88 ? "premium" : score >= 76 ? "good" : score >= 60 ? "neutral" : "warn";

  return (
    <div className="nhlrost-potential-stack">
      <MiniBadge text={label} tone={tone} />
      <span>{score ? `${score}/100` : "—"}</span>
    </div>
  );
}

function PlayerStatusStrip({ player }) {
  const moraleBand = getMoraleBand(player?.morale);
  const fatigueBand = getFatigueBand(player?.fatigue);
  const healthBand = getHealthBand(player);

  return (
    <div className="nhlrost-status-strip">
      <MiniBadge text={`Morale ${round0(player?.morale)}`} tone={moraleBand.tone} title={moraleBand.label} />
      <MiniBadge text={`Fatigue ${round0(player?.fatigue)}`} tone={fatigueBand.tone} title={fatigueBand.label} />
      <MiniBadge text={healthBand.label} tone={healthBand.tone} />
    </div>
  );
}

function compactStatLine(player) {
  const line = statLineForPlayer(player);
  return line === "Stats unavailable" ? "—" : line;
}

function groupPlayersForBoard(players) {
  const forwards = [];
  const defense = [];
  const goalies = [];
  const injured = [];
  const other = [];

  (players || EMPTY_ARRAY).forEach((player) => {
    const health = normalizeHealth(player);
    const isInjured = player.status === "Injured" || health.isInjured;

    if (isInjured) {
      injured.push(player);
      return;
    }

    if (isGoaliePosition(player.position)) {
      goalies.push(player);
    } else if (isDefensePosition(player.position)) {
      defense.push(player);
    } else if (isForwardPosition(player.position)) {
      forwards.push(player);
    } else {
      other.push(player);
    }
  });

  return { forwards, defense, goalies, injured, other };
}

function PlayerFlagBadge({ player, size = "sm" }) {
  const enriched = ensurePlayerHeadshotFields(player || {});
  const imgSize = size === "lg" ? 80 : 64;
  const url = rosterFlagUrl(enriched, imgSize, "flat");
  const label = resolveRosterFlagLabel(enriched);
  const [failed, setFailed] = useState(false);

  if (url && !failed) {
    return (
      <img
        className={`nhlrost-flag-badge ${size === "lg" ? "is-lg" : ""}`}
        src={url}
        alt={label || ""}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }

  if (label) {
    return <span className={`nhlrost-flag-fallback ${size === "lg" ? "is-lg" : ""}`}>{label}</span>;
  }

  return null;
}

function PotentialPill({ player, large = false }) {
  const label = safeStr(player?.potential, "—");
  const score = safeNum(player?.potentialScore, 0);
  const tone = potentialToneClass(score);

  return (
    <span className={`nhlrost-potential-pill ${tone} ${large ? "is-large" : ""}`}>
      {label}
    </span>
  );
}

function OvrPill({ player, large = false }) {
  const tone = potentialToneClass(safeNum(player?.potentialScore, 0));
  const growth = inferGrowth(player);
  const growthRounded = Math.round(growth);

  return (
    <span className={`nhlrost-ovr-pill ${tone} ${large ? "is-large" : ""}`}>
      <span className="nhlrost-ovr-pill__value">{displayOverallValue(player)}</span>
      {growthRounded !== 0 ? (
        <span
          className={`nhlrost-ovr-pill__delta ${growthRounded > 0 ? "is-up" : "is-down"}`}
          title={`Season overall ${growthRounded > 0 ? "+" : ""}${growthRounded}`}
        >
          {growthRounded > 0 ? `+${growthRounded}` : growthRounded}
        </span>
      ) : null}
    </span>
  );
}

function RosterPlayerFlag({ player }) {
  const enriched = ensurePlayerHeadshotFields(player || {});
  const url = rosterFlagUrl(enriched, 64, "flat");
  const label = resolveRosterFlagLabel(enriched);
  const [failed, setFailed] = useState(false);

  if (url && !failed) {
    return (
      <img
        className="nhlrost-board-row__flag"
        src={url}
        alt={label || ""}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }

  if (label) {
    return <span className="nhlrost-board-row__flag-fallback">{label}</span>;
  }

  return null;
}

function PremiumPlayerRow({ player, selected, onSelect, showTeam = false }) {
  return (
    <button
      type="button"
      className={`nhlrost-board-row ${selected ? "is-selected" : ""}${showTeam ? " has-team" : ""}`}
      onClick={() => onSelect(player)}
    >
      <PlayerIconPlate player={player} />

      <span className="nhlrost-board-row__identity">
        <span className="nhlrost-board-row__name-line">
          <RosterPlayerFlag player={player} />
          <strong>{player.name}</strong>
        </span>
        {showTeam && player.teamName && player.teamName !== "—" ? (
          <em className="nhlrost-board-row__team">{player.teamName}</em>
        ) : null}
      </span>

      <span className={`nhlrost-board-row__pos pos-${player.positionClass}`}>{player.position}</span>

      <span className="nhlrost-board-row__age">{player.age ? round0(player.age) : "—"}</span>

      <span className="nhlrost-board-row__ovr">
        <OvrPill player={player} />
      </span>

      <span className="nhlrost-board-row__pot">
        <PotentialPill player={player} />
      </span>

      <span className="nhlrost-board-row__stats">{compactBoardStats(player)}</span>

      <span className="nhlrost-board-row__cap">{capHitDisplay(player)}</span>
    </button>
  );
}

function RosterBoardView({ players, selectedPlayerKey, onSelectPlayer, showTeam = false }) {
  if (!players.length) {
    return (
      <EmptyPanel
        title="No players loaded"
        body="This pool has no connected players for the current franchise data."
      />
    );
  }

  return (
    <div className="nhlrost-board">
      <div className="nhlrost-board-list">
        {players.map((player, index) => (
          <PremiumPlayerRow
            key={player.key || `${player.name}-${index}`}
            player={player}
            selected={player.key === selectedPlayerKey}
            onSelect={onSelectPlayer}
            showTeam={showTeam}
          />
        ))}
      </div>
    </div>
  );
}

function ViewModeSegmented({ value, onChange }) {
  return (
    <div className="nhlrost-view-modes">
      <span className="nhlrost-view-modes__label">View</span>
      <div className="nhlrost-view-modes__buttons" role="group" aria-label="View mode">
        {VIEW_MODE_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={value === option.value ? "is-active" : ""}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const LEAGUE_POOL_OPTIONS = [
  { value: "nhl", label: "NHL" },
  { value: "ahl", label: "AHL" },
  { value: "echl", label: "ECHL" },
  { value: "rights", label: "Rights Held" },
];

function PlayerSearchModeSegmented({ value, onChange }) {
  return (
    <div className="nhlrost-search-mode-segment" role="group" aria-label="Player search mode">
      {PLAYER_SEARCH_MODE_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={value === option.value ? "is-active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function LeaguePoolSegmented({ value, onChange }) {
  return (
    <div className="nhlrost-pool-segment" role="group" aria-label="League pool">
      {LEAGUE_POOL_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={value === option.value ? "is-active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function PlayerInspectorStrip({ player, onExpand, onClose }) {
  return (
    <div className="nhlrost-inspector-strip">
      <PlayerAvatar player={player} size="sm" />

      <div className="nhlrost-inspector-strip__meta">
        <strong>{player.name}</strong>
        <span>
          {player.position} · OVR {player.ovr ? round0(player.ovr) : "—"} · {player.potential}
        </span>
      </div>

      <button type="button" className="nhlrost-chip-button" onClick={onExpand}>
        Expand Details
      </button>

      <button type="button" className="nhlrost-inspector-close" onClick={onClose} aria-label="Clear selection">
        ×
      </button>
    </div>
  );
}

function RosterWarnings({ warnings }) {
  if (!Array.isArray(warnings) || !warnings.length) {
    return (
      <div className="nhlrost-warning-clean">
        <span>✓</span>
        <strong>Roster check clean</strong>
        <p>No active roster warnings from the currently loaded data.</p>
      </div>
    );
  }

  return (
    <div className="nhlrost-warning-list">
      {warnings.map((warning) => (
        <article key={warning.key} className={`nhlrost-warning-card ${toneClass(warning.tone)}`}>
          <strong>{warning.title}</strong>
          <p>{warning.body}</p>
        </article>
      ))}
    </div>
  );
}

function RosterTable({
  players,
  selectedPlayerKey,
  onSelectPlayer,
  pageOffset = 0,
  showPoolColumn = false,
}) {
  if (!players.length) {
    return <EmptyPanel title="No players match these filters" body="Adjust search, position, or advanced filters." />;
  }

  return (
    <div className="nhlrost-table">
      <div className={`nhlrost-table__head ${showPoolColumn ? "has-pool" : ""}`}>
        <span>Name</span>
        <span>Pos</span>
        <span>OVR</span>
        <span>Potential</span>
        <span>Age</span>
        <span>Role</span>
        <span>Type</span>
        <span>Stats</span>
        <span>Status</span>
        {showPoolColumn ? <span>League</span> : null}
      </div>

      <div className="nhlrost-table__body">
        {players.map((player, index) => {
          const selected = player.key === selectedPlayerKey;
          const archetypeColor = getArchetypeColor(player.archetype);
          const healthBand = getHealthBand(player);
          const globalIndex = pageOffset + index;

          return (
            <button
              type="button"
              key={player.key || `${player.name}-${globalIndex}`}
              className={`nhlrost-row ${selected ? "is-selected" : ""}`}
              onClick={() => onSelectPlayer(player)}
            >
              <span className="nhlrost-row__name">
                <PlayerAvatar player={player} size="sm" />
                <span>
                  <strong>{player.name}</strong>
                  <em>{player.teamName}</em>
                </span>
              </span>

              <span className={`nhlrost-row__pos pos-${player.positionClass}`}>{player.position}</span>

              <span>
                <OvrStack player={player} />
              </span>

              <span>
                <PotentialStack player={player} />
              </span>

              <span className="nhlrost-row__age">{player.age}</span>

              <span className="nhlrost-row__role">{player.roleLabel || player.role}</span>

              <span>
                <em
                  className="nhlrost-archetype-tag"
                  style={{
                    "--arch-color": archetypeColor,
                  }}
                >
                  {player.archetype}
                </em>
              </span>

              <span className="nhlrost-row__stats">{compactStatLine(player)}</span>

              <span>
                <MiniBadge text={player.status} tone={healthBand.tone} />
              </span>

              {showPoolColumn ? <span className="nhlrost-row__pool">{player.league || player._source || "—"}</span> : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function DraftBoardTable({ players, selectedPlayerKey, onSelectPlayer, pageOffset = 0 }) {
  if (!players.length) {
    return <EmptyPanel title="No draft players match" body="The draft board is empty or filtered out." />;
  }

  return (
    <div className="nhlrost-draft-table">
      <div className="nhlrost-draft-table__head">
        <span>Rank</span>
        <span>Move</span>
        <span>Player</span>
        <span>Pos</span>
        <span>Age</span>
        <span>League</span>
        <span>True OVR</span>
        <span>Scout</span>
        <span>Projection</span>
        <span>Type</span>
      </div>

      <div className="nhlrost-draft-table__body">
        {players.map((player, index) => {
          const selected = player.key === selectedPlayerKey;

          return (
            <button
              type="button"
              key={player.key || `draft-${pageOffset + index}`}
              className={`nhlrost-draft-row ${selected ? "is-selected" : ""}`}
              onClick={() => onSelectPlayer(player)}
            >
              <span>{player.rank}</span>
              <span className={`nhlrost-draft-trend ${player.trendClass}`}>{player.trendText}</span>
              <span>
                <strong>{player.name}</strong>
                <em>{player.nat}</em>
              </span>
              <span>{player.position}</span>
              <span>{player.age}</span>
              <span>{player.league}</span>
              <span>{player.true_ovr || player.ovr || "—"}</span>
              <span>{player.scout_grade}</span>
              <span>
                <PotentialStack player={player} />
              </span>
              <span>{player.archetype}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RosterCards({ players, selectedPlayerKey, onSelectPlayer }) {
  if (!players.length) {
    return <EmptyPanel title="No cards to show" body="No players match the current filters." />;
  }

  return (
    <div className="nhlrost-card-grid">
      {players.map((player) => {
        const selected = player.key === selectedPlayerKey;
        const healthBand = getHealthBand(player);
        const moraleBand = getMoraleBand(player.morale);
        const fatigueBand = getFatigueBand(player.fatigue);

        return (
          <button
            type="button"
            key={player.key}
            className={`nhlrost-player-card ${selected ? "is-selected" : ""}`}
            onClick={() => onSelectPlayer(player)}
          >
            <div className="nhlrost-player-card__top">
              <PlayerAvatar player={player} size="lg" />

              <div className="nhlrost-player-card__identity">
                <strong>{player.name}</strong>
                <span>
                  {player.position} · {player.age} · {player.teamName}
                </span>
              </div>

              <div className="nhlrost-player-card__ovr">
                <strong style={{ color: getOVRColor(player.ovr) }}>{player.ovr ? round0(player.ovr) : "—"}</strong>
                <span>{gradeFromOverall(player.ovr)}</span>
              </div>
            </div>

            <div className="nhlrost-player-card__badges">
              <MiniBadge text={player.potential} tone={player.potentialScore >= 88 ? "premium" : player.potentialScore >= 76 ? "good" : "neutral"} />
              <MiniBadge text={player.archetype} />
              <MiniBadge text={player.roleLabel || player.role} />
              <MiniBadge text={player.status} tone={healthBand.tone} />
            </div>

            <div className="nhlrost-player-card__metrics">
              <InfoPair label="True OVR" value={player.trueOverall ? round1(player.trueOverall) : "—"} />
              <InfoPair label="Potential" value={player.potentialScore ? `${player.potentialScore}/100` : "—"} />
              <InfoPair label="Morale" value={`${round0(player.morale)} · ${moraleBand.label}`} tone={moraleBand.tone} />
              <InfoPair label="Fatigue" value={`${round0(player.fatigue)} · ${fatigueBand.label}`} tone={fatigueBand.tone} />
              <InfoPair label="Cap Hit" value={formatMoneyMillions(player.contract?.capHit)} />
              <InfoPair label="Asset" value={player.asset?.label || "—"} />
            </div>

            <div className="nhlrost-player-card__statline">{statLineForPlayer(player)}</div>
          </button>
        );
      })}
    </div>
  );
}

function LineView({ lineGroups, selectedPlayerKey, onSelectPlayer }) {
  const forwardRows = lineGroups?.forwards || EMPTY_ARRAY;
  const defenseRows = lineGroups?.defense || EMPTY_ARRAY;
  const goalies = lineGroups?.goalies || EMPTY_ARRAY;
  const extras = lineGroups?.extras || EMPTY_OBJECT;

  const renderPlayerChip = (player, label = "") => {
    if (!player) {
      return (
        <span className="nhlrost-line-chip is-empty">
          <em>{label || "Empty"}</em>
        </span>
      );
    }

    return (
      <button
        type="button"
        className={`nhlrost-line-chip ${player.key === selectedPlayerKey ? "is-selected" : ""}`}
        onClick={() => onSelectPlayer(player)}
      >
        <span>{label || player.position}</span>
        <strong>{player.name}</strong>
        <em>{player.ovr ? round0(player.ovr) : "—"} OVR · {player.potential}</em>
      </button>
    );
  };

  return (
    <div className="nhlrost-lines">
      <section className="nhlrost-line-section">
        <header>
          <span>Forward Lines</span>
          <strong>Position-aware auto read</strong>
        </header>

        {forwardRows.map((line, index) => (
          <article key={`f-line-${index}`} className="nhlrost-line-row">
            <strong>L{index + 1}</strong>
            <div>
              {renderPlayerChip(line[0], "LW")}
              {renderPlayerChip(line[1], "C")}
              {renderPlayerChip(line[2], "RW")}
            </div>
          </article>
        ))}
      </section>

      <section className="nhlrost-line-section">
        <header>
          <span>Defense Pairs</span>
          <strong>LD/RD balanced where possible</strong>
        </header>

        {defenseRows.map((pair, index) => (
          <article key={`d-pair-${index}`} className="nhlrost-line-row">
            <strong>D{index + 1}</strong>
            <div>
              {renderPlayerChip(pair[0], "LD")}
              {renderPlayerChip(pair[1], "RD")}
            </div>
          </article>
        ))}
      </section>

      <section className="nhlrost-line-section nhlrost-line-section--goalies">
        <header>
          <span>Goalies</span>
          <strong>Best available healthy goalies</strong>
        </header>

        <article className="nhlrost-line-row">
          <strong>G</strong>
          <div>
            {renderPlayerChip(goalies[0], "Starter")}
            {renderPlayerChip(goalies[1], "Backup")}
          </div>
        </article>
      </section>

      <section className="nhlrost-line-section nhlrost-line-section--extras">
        <header>
          <span>Extras</span>
          <strong>Not assigned to current auto lines</strong>
        </header>

        <div className="nhlrost-extra-grid">
          {(extras.forwards || EMPTY_ARRAY).map((player) => renderPlayerChip(player, "F"))}
          {(extras.defense || EMPTY_ARRAY).map((player) => renderPlayerChip(player, "D"))}
          {(extras.goalies || EMPTY_ARRAY).map((player) => renderPlayerChip(player, "G"))}
        </div>
      </section>

      <ConnectedActionNotice
        tone="neutral"
        title="Line editing not faked"
        body="This view reads and organizes loaded roster data. It does not pretend to save lines until a real backend lineup endpoint exists."
      />
    </div>
  );
}

function RatingsEngineView({ players, selectedPlayerKey, onSelectPlayer }) {
  if (!players.length) {
    return <EmptyPanel title="No ratings loaded" body="No players match the current filters." />;
  }

  return (
    <div className="nhlrost-ratings-grid-view">
      {players.map((player) => {
        const selected = player.key === selectedPlayerKey;

        return (
          <button
            type="button"
            key={player.key}
            className={`nhlrost-rating-card ${selected ? "is-selected" : ""}`}
            onClick={() => onSelectPlayer(player)}
          >
            <header>
              <PlayerAvatar player={player} size="sm" />
              <div>
                <strong>{player.name}</strong>
                <span>
                  {player.position} · {player.roleLabel || player.role}
                </span>
              </div>
              <OvrStack player={player} />
            </header>

            <div className="nhlrost-rating-card__bars">
              {(player.rating_groups || EMPTY_ARRAY).map((group) => {
                const avg = averageRows(group.rows);

                return (
                  <ProgressBar
                    key={group.key || group.title}
                    label={group.title}
                    value={avg}
                    tone={avg >= 84 ? "good" : avg >= 72 ? "neutral" : "warn"}
                  />
                );
              })}
            </div>

            <footer>
              <span>{player.knownRatingCount} known ratings</span>
              <strong>{player.overallConfidence} confidence</strong>
            </footer>
          </button>
        );
      })}
    </div>
  );
}

function PlayerOverviewPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No player selected" body="Choose a player from the roster board." />;
  }

  const stats = player.season_stats || EMPTY_OBJECT;
  const contract = player.contract || EMPTY_OBJECT;
  const gp = safeNum(stats.gp, 0);
  const hasSeasonGames = gp > 0;
  const healthBand = getHealthBand(player);
  const teamDisplay = player.teamName && player.teamName !== "—" ? player.teamName : "—";
  const leagueDisplay = player.league && player.league !== "—" ? player.league : "—";

  return (
    <section className="nhlrost-player-overview">
      <article className="nhlrost-profile-zone nhlrost-profile-zone--bio">
        <header className="nhlrost-profile-zone__head">
          <p>Player Profile</p>
          <h3>{getPositionDisplay(player.position)}</h3>
        </header>
        <div className="nhlrost-profile-kv-grid">
          <InfoPair label="Age" value={player.age || "—"} />
          <InfoPair label="Height" value={formatHeightDisplay(player.hgt)} />
          <InfoPair label="Weight" value={formatWeightDisplay(player.wgt)} />
          <InfoPair label="Nationality" value={player.nat && player.nat !== "—" ? player.nat : "—"} />
          <InfoPair label="Hand" value={formatHandLabel(player)} />
          <InfoPair label="Status" value={player.status || "Active"} tone={healthBand.tone} />
          <InfoPair label="Team" value={teamDisplay} />
          <InfoPair label="League" value={leagueDisplay} />
        </div>
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--ability">
        <header className="nhlrost-profile-zone__head">
          <p>Ability & Role</p>
          <h3>{player.roleLabel || player.role || "—"}</h3>
        </header>
        <div className="nhlrost-profile-summary-strip">
          <div className="nhlrost-profile-summary-item">
            <span>Current OVR</span>
            <OvrPill player={player} large />
          </div>
          <div className="nhlrost-profile-summary-item">
            <span>Potential</span>
            <PotentialPill player={player} large />
          </div>
          {player.asset?.label ? (
            <div className="nhlrost-profile-summary-item">
              <span>Asset Tier</span>
              <strong>{player.asset.label}</strong>
            </div>
          ) : null}
        </div>
        <div className="nhlrost-profile-kv-grid">
          <InfoPair label="Archetype" value={player.archetype || "—"} />
          <InfoPair label="Special Teams" value={player.explicitSpecialTeams || player.specialTeams || "—"} />
          <InfoPair label="Avg TOI" value={player.explicitMinutes != null ? `${round0(player.explicitMinutes)} min` : player.minutes ? `${round1(player.minutes)} min` : "—"} />
          <InfoPair label="Stage" value={player.stage || "—"} />
        </div>
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--contract">
        <header className="nhlrost-profile-zone__head">
          <p>Contract Snapshot</p>
          <h3>{capHitDisplay(player)}</h3>
        </header>
        <div className="nhlrost-profile-kv-grid">
          <InfoPair label="Status" value={formatContractStatus(contract)} />
          <InfoPair label="Term" value={contract.term ? `${contract.term} yr` : "—"} />
          <InfoPair label="Expiry" value={formatContractExpiry(contract)} />
          <InfoPair label="Type" value={contract.type || "—"} />
          <InfoPair label="Clause" value={contract.clause || "—"} />
          <InfoPair label="Morale" value={player.morale != null ? round0(player.morale) : "—"} />
        </div>
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--performance">
        <header className="nhlrost-profile-zone__head">
          <p>Season Performance</p>
          <h3>{hasSeasonGames ? `${gp} GP` : "No games played"}</h3>
        </header>
        {hasSeasonGames ? (
          <div className="nhlrost-profile-stat-band">
            {isGoaliePosition(player.position) ? (
              <>
                <InfoPair label="Record" value={`${displayStatValue(stats.wins)}-${displayStatValue(stats.losses)}-${displayStatValue(stats.otl)}`} />
                <InfoPair label="SV%" value={stats.svPct ? formatDecimal(stats.svPct, 3) : "—"} />
                <InfoPair label="GAA" value={stats.gaa ? stats.gaa.toFixed(2) : "—"} />
                <InfoPair label="Saves" value={displayStatValue(stats.saves)} />
              </>
            ) : (
              <>
                <InfoPair label="G" value={displayStatValue(stats.g)} />
                <InfoPair label="A" value={displayStatValue(stats.a)} />
                <InfoPair label="PTS" value={displayStatValue(stats.pts)} />
                <InfoPair label="P/GP" value={stats.ppg ? stats.ppg.toFixed(2) : displayStatValue(0)} />
                <InfoPair label="Shots" value={displayStatValue(stats.shots)} />
                <InfoPair label="+/-" value={stats.plusMinus != null ? formatSignedNumber(stats.plusMinus, 0) : "—"} />
                <InfoPair label="TOI/GP" value={stats.toi ? `${round1(stats.toi)} min` : "—"} />
              </>
            )}
          </div>
        ) : (
          <p className="nhlrost-muted-text">No regular-season games played.</p>
        )}
      </article>
    </section>
  );
}

function RatingsPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No ratings selected" body="Select a player to inspect ratings." />;
  }

  const groups = (player.rating_groups || EMPTY_ARRAY).filter((group) => group?.rows?.length);

  if (!groups.length) {
    return <EmptyPanel title="No ratings loaded" body="Backend rating groups are not available for this player." />;
  }

  return (
    <section className="nhlrost-ratings-layout">
      <div className="nhlrost-ratings-summary">
        {groups.map((group) => {
          const average = averageRows(group.rows);
          const top = [...(group.rows || [])].sort((a, b) => safeNum(b.value, 0) - safeNum(a.value, 0))[0];
          const low = [...(group.rows || [])].sort((a, b) => safeNum(a.value, 0) - safeNum(b.value, 0))[0];

          return (
            <article key={group.key || group.title} className="nhlrost-ratings-summary-card">
              <span>{group.title}</span>
              <strong>{average ? round0(average) : "—"}</strong>
              <em>
                {top?.label ? `High: ${top.label}` : "—"}
                {low?.label && low.id !== top?.id ? ` · Low: ${low.label}` : ""}
              </em>
            </article>
          );
        })}
      </div>

      <div className="nhlrost-detail-grid nhlrost-detail-grid--ratings">
        {groups.map((group) => {
          const average = averageRows(group.rows);
          const sortedRows = [...(group.rows || [])].sort(
            (a, b) => safeNum(b.value, 0) - safeNum(a.value, 0) || safeStr(a.label, "").localeCompare(safeStr(b.label, ""))
          );

          return (
            <article key={group.key || group.title} className="nhlrost-panel nhlrost-rating-group">
              <header>
                <div>
                  <p>{group.title}</p>
                  <h3>{average ? round0(average) : "—"}</h3>
                </div>
              </header>

              <div className="nhlrost-rating-row-list">
                {sortedRows.map((row) => (
                  <ProgressBar
                    key={row.id}
                    label={row.label}
                    value={row.value}
                    tone={row.value >= 85 ? "good" : row.value >= 72 ? "neutral" : "warn"}
                  />
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ProductionPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No player selected" body="Select a player to view stats." />;
  }

  const stats = player.season_stats || EMPTY_OBJECT;
  const isGoalie = isGoaliePosition(player.position);
  const gp = safeNum(stats.gp, 0);
  const hasSeasonGames = gp > 0;

  if (!hasSeasonGames) {
    return (
      <section className="nhlrost-stats-layout">
        <EmptyPanel
          compact
          title="No regular-season games played"
          body="Roster status, ratings, and contract data remain available in other tabs."
        />
      </section>
    );
  }

  if (isGoalie) {
    return (
      <section className="nhlrost-stats-layout">
        <article className="nhlrost-panel nhlrost-stats-band">
          <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
            <InfoPair label="GP" value={displayStatValue(stats.gp)} />
            <InfoPair label="Record" value={`${displayStatValue(stats.wins)}-${displayStatValue(stats.losses)}-${displayStatValue(stats.otl)}`} />
            <InfoPair label="SV%" value={stats.svPct ? formatDecimal(stats.svPct, 3) : "—"} />
            <InfoPair label="GAA" value={stats.gaa ? stats.gaa.toFixed(2) : "—"} />
            <InfoPair label="Saves" value={displayStatValue(stats.saves)} />
            <InfoPair label="Shots Against" value={displayStatValue(stats.shotsAgainst)} />
            <InfoPair label="Shutouts" value={displayStatValue(stats.shutouts)} />
            <InfoPair label="TOI/GP" value={stats.toi ? `${round1(stats.toi)} min` : "—"} />
          </div>
        </article>
      </section>
    );
  }

  return (
    <section className="nhlrost-stats-layout">
      <article className="nhlrost-panel nhlrost-stats-band">
        <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
          <InfoPair label="GP" value={displayStatValue(stats.gp)} />
          <InfoPair label="G" value={displayStatValue(stats.g)} />
          <InfoPair label="A" value={displayStatValue(stats.a)} />
          <InfoPair label="PTS" value={displayStatValue(stats.pts)} />
          <InfoPair label="P/GP" value={stats.ppg ? stats.ppg.toFixed(2) : displayStatValue(0)} />
          <InfoPair label="Shots" value={displayStatValue(stats.shots)} />
          <InfoPair label="SH%" value={stats.shootingPct ? `${(stats.shootingPct * 100).toFixed(1)}%` : "—"} />
          <InfoPair label="+/-" value={stats.plusMinus != null ? formatSignedNumber(stats.plusMinus, 0) : "—"} />
          <InfoPair label="TOI/GP" value={stats.toi ? `${round1(stats.toi)} min` : "—"} />
          <InfoPair label="Hits" value={displayStatValue(stats.hits)} />
          <InfoPair label="Blocks" value={displayStatValue(stats.blocks)} />
          <InfoPair label="PIM" value={displayStatValue(stats.pim)} />
        </div>
      </article>
    </section>
  );
}

function ContractPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No contract selected" body="Select a player to view contract details." />;
  }

  const contract = player.contract || EMPTY_OBJECT;
  const valueTone =
    contract.capHit >= 8 && getUniversalOverall(player) < 86
      ? "bad"
      : contract.capHit <= 3 && getUniversalOverall(player) >= 78
        ? "good"
        : "neutral";

  return (
    <section className="nhlrost-contract-layout">
      <article className="nhlrost-panel nhlrost-contract-panel">
        <div className="nhlrost-contract-hero">
          <span>Cap Hit</span>
          <strong className={toneClass(valueTone)}>{capHitDisplay(player)}</strong>
        </div>

        <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
          <InfoPair label="Salary" value={formatMoneyMillions(contract.salary)} />
          <InfoPair label="Term" value={contract.term ? `${contract.term} yr` : "—"} />
          <InfoPair label="Expiry" value={formatContractExpiry(contract)} />
          <InfoPair label="Type" value={contract.type || "—"} />
          <InfoPair label="Clause" value={contract.clause || "—"} />
          <InfoPair label="Status" value={formatContractStatus(contract)} />
        </div>
      </article>
    </section>
  );
}

function DevelopmentPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No development data" body="Select a player to view development." />;
  }

  const hasMorale = player.morale != null && Number.isFinite(Number(player.morale));
  const hasFatigue = player.fatigue != null && Number.isFinite(Number(player.fatigue));
  const hasGrowth = player.growth != null && Number.isFinite(Number(player.growth));

  return (
    <section className="nhlrost-development-layout">
      <article className="nhlrost-panel">
        <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
          <InfoPair label="Current OVR" value={displayOverallValue(player)} />
          <InfoPair label="Age" value={player.age || "—"} />
          <InfoPair label="Potential" value={player.potential || "—"} />
          <InfoPair label="Stage" value={player.stage || "—"} />
          <InfoPair label="Role" value={player.roleLabel || player.role || "—"} />
          {hasGrowth ? (
            <InfoPair label="Growth" value={formatSignedNumber(player.growth)} tone={player.growth > 0 ? "good" : player.growth < 0 ? "warn" : "neutral"} />
          ) : null}
          {hasMorale ? <InfoPair label="Morale" value={round0(player.morale)} /> : null}
          {hasFatigue ? <InfoPair label="Fatigue" value={round0(player.fatigue)} /> : null}
          <InfoPair label="Confidence" value={player.overallConfidence || "—"} />
        </div>
      </article>
    </section>
  );
}

function UsagePanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No usage selected" body="Select a player to view role and deployment." />;
  }

  return (
    <section className="nhlrost-detail-grid">
      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Usage</p>
            <h3>Deployment Read</h3>
          </div>
          <span>{player.roleLabel || player.role}</span>
        </header>

        <div className="nhlrost-stat-grid">
          <InfoPair label="Role" value={player.roleLabel || player.role} />
          <InfoPair label="Special Teams" value={player.specialTeams} />
          <InfoPair label="Average TOI" value={player.minutes ? `${round1(player.minutes)} min` : "—"} />
          <InfoPair label="Status" value={player.status} tone={getHealthBand(player).tone} />
          <InfoPair label="League" value={player.league} />
          <InfoPair label="Position Class" value={player.positionClass} />
        </div>
      </article>

      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Usage Logic</p>
            <h3>No Fake Save</h3>
          </div>
          <span>Read-only</span>
        </header>

        <p className="nhlrost-muted-text">
          This panel reads current player usage from backend fields when available. If no backend usage exists, it derives a read-only estimate from position and calculated overall. It does not pretend to save roster moves.
        </p>

        <ConnectedActionNotice
          tone="neutral"
          title="Roster actions hidden until connected"
          body="Call-ups, scratches, waivers, trade block, and lineup saves should only appear here after real backend handlers exist."
        />
      </article>
    </section>
  );
}

function HistoryPanel({ player, storylines }) {
  if (!player) {
    return <EmptyPanel title="No history selected" body="Select a player to view timeline events." compact />;
  }

  return (
    <section className="nhlrost-history-layout">
      {storylines.length ? (
        <div className="nhlrost-storyline-list">
          {storylines.map((event, index) => (
            <article key={event.id || event.storyline_id || index} className="nhlrost-storyline-card">
              <strong>{event.headline || event.title || event.type || "Storyline"}</strong>
              {event.season || event.date ? (
                <span>{event.season || event.date}</span>
              ) : null}
              {event.effect_summary ? <p>{event.effect_summary}</p> : null}
            </article>
          ))}
        </div>
      ) : (
        <EmptyPanel compact title="No recorded career events yet" body="Timeline events appear when this player is tied to franchise storylines." />
      )}
    </section>
  );
}

function PlayerProfileModal({ player, activeTab, setActiveTab, storylines, onClose }) {
  const modalBodyRef = React.useRef(null);

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  useEffect(() => {
    if (modalBodyRef.current) {
      modalBodyRef.current.scrollTop = 0;
    }
  }, [player?.key, player?.id, activeTab]);

  if (!player) return null;

  const teamLeague = [player.teamName, player.league]
    .filter((value) => value && value !== "—")
    .filter((value, index, list) => list.indexOf(value) === index)
    .join(" · ");
  const healthBand = getHealthBand(player);
  const flagCode =
    player.nationality_code ||
    nationalityCode(pickFirstDefined(player.nat, player.nationality, player.country) || "") ||
    resolveRosterFlagLabel(player);

  return (
    <div className="nhlrost-profile-modal" role="dialog" aria-modal="true" aria-label={`${player.name} profile`}>
      <button type="button" className="nhlrost-profile-modal__backdrop" onClick={onClose} aria-label="Close profile" />

      <div className="nhlrost-profile-modal__panel" tabIndex={-1}>
        <header className="nhlrost-profile-modal__hero">
          <div className="nhlrost-profile-modal__visual">
            <PlayerHeadshot
              player={ensurePlayerHeadshotFields(player)}
              size="md"
              variant="card"
              className="nhlrost-profile-modal__headshot"
              number={player.num}
              flag={flagCode || null}
            />
          </div>

          <div className="nhlrost-profile-modal__meta">
            <div className="nhlrost-profile-modal__identity-row">
              {flagCode ? <PlayerFlagBadge player={player} size="sm" /> : null}
              <h2 id="nhlrost-profile-title">{player.name}</h2>
            </div>
            <p>
              {getPositionDisplay(player.position)} · {player.age || "—"}
              {teamLeague ? ` · ${teamLeague}` : ""}
              {player.status && player.status !== "Active" ? ` · ${player.status}` : ""}
            </p>

            <div className="nhlrost-profile-modal__chips">
              <OvrPill player={player} large />
              <PotentialPill player={player} large />
              {player.roleLabel || player.role ? (
                <span className="nhlrost-profile-modal__role">{player.roleLabel || player.role}</span>
              ) : null}
              {healthBand.tone === "medical" || healthBand.tone === "bad" ? (
                <span className={`nhlrost-profile-modal__health ${toneClass(healthBand.tone)}`}>{healthBand.label}</span>
              ) : null}
            </div>
          </div>

          <button type="button" className="nhlrost-profile-modal__close" onClick={onClose} aria-label="Close profile">
            ×
          </button>
        </header>

        <DetailTabs activeTab={activeTab} setActiveTab={setActiveTab} />

        <div className="nhlrost-profile-modal__body" ref={modalBodyRef}>
          <DetailPanelRouter activeTab={activeTab} player={player} storylines={storylines} />
        </div>
      </div>
    </div>
  );
}

function DetailTabs({ activeTab, setActiveTab }) {
  return (
    <nav className="nhlrost-detail-tabs" aria-label="Player detail tabs">
      {PANEL_TABS.map((tab) => (
        <button
          key={tab.value}
          type="button"
          className={activeTab === tab.value ? "is-active" : ""}
          onClick={() => setActiveTab(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

function DetailPanelRouter({ activeTab, player, storylines }) {
  if (activeTab === "overview") return <PlayerOverviewPanel player={player} />;
  if (activeTab === "ratings") return <RatingsPanel player={player} />;
  if (activeTab === "production") return <ProductionPanel player={player} />;
  if (activeTab === "contract") return <ContractPanel player={player} />;
  if (activeTab === "development") return <DevelopmentPanel player={player} />;
  if (activeTab === "history") return <HistoryPanel player={player} storylines={storylines} />;

  return <PlayerOverviewPanel player={player} />;
}

function resolveRosterTeamToken(franchiseState) {
  const team = franchiseState?.user_team || franchiseState?.team || {};
  const abbr = String(
    team.abbreviation ?? team.abbrev ?? team.abbr ?? team.short_name ?? ""
  ).trim();

  if (abbr && /[A-Za-z]/.test(abbr)) return abbr.toUpperCase().slice(0, 3);

  const name = String(team.name ?? team.full_name ?? team.fullName ?? "").trim();
  if (name) {
    const token = name.split(/\s+/).map((part) => part[0]).join("").toUpperCase();
    if (token) return token.slice(0, 3);
  }

  return "TM";
}

function CommandBarTeamLogo({ franchiseState }) {
  const team = franchiseState?.user_team || franchiseState?.team || {};
  const teamName = safeStr(
    team.name || team.full_name || team.fullName || franchiseState?.team?.name,
    ""
  );

  const logoUrl = useMemo(() => {
    const resolved = resolveFranchiseTeamLogo(team, teamName);
    if (resolved) return resolved;

    const remote = team.logo ?? team.logo_url ?? team.logoUrl ?? team.crest ?? team.primaryLogo ?? "";
    return typeof remote === "string" ? remote.trim() : "";
  }, [team, teamName]);

  const fallbackToken = useMemo(() => resolveRosterTeamToken(franchiseState), [franchiseState]);
  const [imgFailed, setImgFailed] = useState(false);
  const showImg = Boolean(logoUrl) && !imgFailed;

  return (
    <div className="nhlrost-team-mark" aria-label={teamName || "Team logo"}>
      {showImg ? (
        <img
          className="nhlrost-team-logo-img"
          src={logoUrl}
          alt=""
          onError={() => setImgFailed(true)}
        />
      ) : (
        <span className="nhlrost-team-logo-fallback">{fallbackToken}</span>
      )}
    </div>
  );
}

export function RosterScreen() {
  const gameUI = useGameUI();

  const {
    franchiseState,
    rosterRowIndex,
    setRosterRowIndex,
    setScreen,
    refreshFranchise,
  } = gameUI;

  const rb = franchiseState?.roster_browser || EMPTY_OBJECT;
  const draftBoard = franchiseState?.draft_class_rankings || EMPTY_OBJECT;
  const organizations = useMemo(
    () => rb?.organizations || EMPTY_ARRAY,
    [rb?.organizations]
  );

  const userTeamId =
    franchiseState?.team?.id ||
    franchiseState?.team?.team_id ||
    franchiseState?.team?.abbr ||
    franchiseState?.team?.abbreviation ||
    franchiseState?.user_team_id ||
    franchiseState?.selected_team_id ||
    "";

  const [browseSource, setBrowseSource] = useState(PLAYER_POOLS.ORGANIZATION);
  const [searchMode, setSearchMode] = useState(PLAYER_SEARCH_MODES.TEAM_ROSTERS);
  const [orgTeamId, setOrgTeamId] = useState("");
  const [orgLevel, setOrgLevel] = useState("nhl");
  const [devLeagueIdx, setDevLeagueIdx] = useState(0);
  const [devTeamIdx, setDevTeamIdx] = useState(0);

  const [searchTerm, setSearchTerm] = useState("");
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [leagueFilter, setLeagueFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("All");
  const [typeFilter, setTypeFilter] = useState("ALL");
  const [roleFilter, setRoleFilter] = useState("ALL");
  const [sortKey, setSortKey] = useState("overall_desc");
  const [viewMode, setViewMode] = useState(VIEW_MODES.BOARD);
  const [activeTab, setActiveTab] = useState("overview");
  const [showCoreOnly, setShowCoreOnly] = useState(false);
  const [showWarningsOnly, setShowWarningsOnly] = useState(false);
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [detailsCollapsed, setDetailsCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [tablePage, setTablePage] = useState(0);
  const [selectedPlayerKey, setSelectedPlayerKey] = useState("");

  const defaultOrgId = useMemo(() => {
    const userOrg = organizations.find((org) => {
      return String(org?.team_id || org?.id || org?.abbr || "").toLowerCase() === String(userTeamId || "").toLowerCase();
    });

    const fallback = userOrg || organizations[0];

    return String(fallback?.team_id || fallback?.id || fallback?.abbr || "");
  }, [organizations, userTeamId]);

  useEffect(() => {
    if (!orgTeamId && defaultOrgId) {
      setOrgTeamId(defaultOrgId);
    }
  }, [defaultOrgId, orgTeamId]);

  const selectedOrganization = useMemo(() => {
    return (
      organizations.find((org) => {
        const orgId = String(org?.team_id || org?.id || org?.abbr || "").toLowerCase();
        return orgId === String(orgTeamId || defaultOrgId || "").toLowerCase();
      }) ||
      organizations[0] ||
      EMPTY_OBJECT
    );
  }, [organizations, orgTeamId, defaultOrgId]);

  const userOrganization = useMemo(() => {
    return (
      organizations.find((org) => {
        const orgId = String(org?.team_id || org?.id || org?.abbr || "").toLowerCase();
        return orgId === String(userTeamId || defaultOrgId || "").toLowerCase();
      }) ||
      selectedOrganization ||
      EMPTY_OBJECT
    );
  }, [organizations, userTeamId, defaultOrgId, selectedOrganization]);

  const franchiseStatsLookup = useMemo(() => buildFranchiseStatsLookup(franchiseState), [franchiseState]);

  const devLeagues = rb?.development_leagues || EMPTY_ARRAY;
  const devTeams = devLeagues[devLeagueIdx]?.teams || EMPTY_ARRAY;

  useEffect(() => {
    if (devLeagueIdx >= devLeagues.length) {
      setDevLeagueIdx(Math.max(0, devLeagues.length - 1));
      setDevTeamIdx(0);
    }
  }, [devLeagueIdx, devLeagues.length]);

  useEffect(() => {
    if (devTeamIdx >= devTeams.length) {
      setDevTeamIdx(Math.max(0, devTeams.length - 1));
    }
  }, [devTeamIdx, devTeams.length]);

  const rawPlayers = useMemo(() => {
    if (browseSource === PLAYER_POOLS.DRAFT_CLASS) {
      return EMPTY_ARRAY;
    }

    if (searchMode === PLAYER_SEARCH_MODES.NHL_LEAGUE) {
      return collectOrganizationPoolPlayers(organizations, "nhl");
    }

    if (!rb || !Object.keys(rb).length) {
      if (browseSource === PLAYER_POOLS.MY_PROSPECTS) {
        return collectMyProspectRawPlayers(rb, franchiseState, userTeamId, userOrganization);
      }

      return franchiseState?.roster || EMPTY_ARRAY;
    }

    if (browseSource === PLAYER_POOLS.FREE_AGENTS) {
      return (rb.free_agents || EMPTY_ARRAY).map((player) => ({
        ...player,
        _source: PLAYER_POOLS.FREE_AGENTS,
        league: player.league || "UFA",
      }));
    }

    if (browseSource === PLAYER_POOLS.OVERSEAS) {
      return (rb.overseas_free_agents || EMPTY_ARRAY).map((player) => ({
        ...player,
        _source: PLAYER_POOLS.OVERSEAS,
        league: player.league || player.league_code || "INTL",
      }));
    }

    if (browseSource === PLAYER_POOLS.MY_PROSPECTS) {
      return collectMyProspectRawPlayers(rb, franchiseState, userTeamId, userOrganization);
    }

    if (browseSource === PLAYER_POOLS.DEVELOPMENT) {
      const selectedLeague = devLeagues[devLeagueIdx] || EMPTY_OBJECT;
      const selectedTeam = (selectedLeague?.teams || EMPTY_ARRAY)[devTeamIdx] || EMPTY_OBJECT;

      return (selectedTeam?.players || EMPTY_ARRAY).map((player) => ({
        ...player,
        _source: PLAYER_POOLS.DEVELOPMENT,
        league:
          player.league ||
          selectedLeague.league_code ||
          selectedLeague.league_name ||
          "DEV",
        team_name: player.team_name || selectedTeam.name || selectedTeam.team_name,
      }));
    }

    const key = orgLevel === "ahl" ? "ahl" : orgLevel === "echl" ? "echl" : "nhl";

    const organizationRows = selectedOrganization?.[key];

    if (Array.isArray(organizationRows)) {
      return organizationRows.map((player) => ({
        ...player,
        _source: PLAYER_POOLS.ORGANIZATION,
        league: player.league || key.toUpperCase(),
        team_name: player.team_name || selectedOrganization.name || franchiseState?.team?.name,
      }));
    }

    return franchiseState?.roster || EMPTY_ARRAY;
  }, [
    browseSource,
    searchMode,
    organizations,
    rb,
    franchiseState,
    selectedOrganization,
    userOrganization,
    userTeamId,
    orgLevel,
    devLeagues,
    devLeagueIdx,
    devTeamIdx,
  ]);

  const players = useMemo(() => {
    if (browseSource === PLAYER_POOLS.DRAFT_CLASS) {
      const rows = draftBoard?.entries || draftBoard?.players || draftBoard?.rankings || EMPTY_ARRAY;
      return rows.map((row, index) => normalizeDraftPlayer(row, index));
    }

    return rawPlayers.map((player, index) => {
      const enriched = mergeFranchiseStatsIntoPlayer(player, franchiseStatsLookup);
      return normalizeLivePlayer(enriched, franchiseState, index);
    });
  }, [browseSource, draftBoard, rawPlayers, franchiseState, franchiseStatsLookup]);

  const roleOptions = useMemo(() => {
    const roles = Array.from(
      new Set(
        players
          .map((player) => safeStr(player.roleLabel || player.role, ""))
          .filter(Boolean)
          .filter((role) => role !== "—")
      )
    ).sort((a, b) => a.localeCompare(b));

    return ["ALL", ...roles];
  }, [players]);

  const filteredPlayers = useMemo(() => {
    const query = safeStr(searchTerm, "").trim().toLowerCase();

    const rows = players.filter((player) => {
      if (query) {
        const haystack = [
          player.name,
          player.position,
          player.archetype,
          player.teamName,
          player.league,
          player.potential,
          player.role,
          player.roleLabel,
          player.nat,
          player.status,
          player.asset?.label,
        ]
          .map((value) => safeStr(value, "").toLowerCase())
          .join(" ");

        if (!haystack.includes(query)) return false;
      }

      if (!positionMatchesFilter(player.position, positionFilter)) return false;

      if (leagueFilter !== "ALL") {
        const playerLeague = safeStr(player.league, "").toUpperCase();

        if (leagueFilter === "EU") {
          const isEuro =
            playerLeague.includes("EU") ||
            playerLeague.includes("SWE") ||
            playerLeague.includes("FIN") ||
            playerLeague.includes("RUS") ||
            playerLeague.includes("CZE") ||
            playerLeague.includes("SHL") ||
            playerLeague.includes("LIIGA");

          if (!isEuro) return false;
        } else if (leagueFilter === "CHL") {
          const isChl =
            playerLeague.includes("CHL") ||
            playerLeague.includes("OHL") ||
            playerLeague.includes("WHL") ||
            playerLeague.includes("QMJHL");

          if (!isChl) return false;
        } else if (!playerLeague.includes(leagueFilter)) {
          return false;
        }
      }

      if (statusFilter !== "All" && player.status !== statusFilter) return false;

      if (typeFilter !== "ALL") {
        const key = normalizeKey(player.archetype);
        const filterKey = normalizeKey(typeFilter);

        if (!key.includes(filterKey)) return false;
      }

      if (roleFilter !== "ALL" && safeStr(player.roleLabel || player.role, "") !== roleFilter) {
        return false;
      }

      if (showCoreOnly && safeNum(player.potentialScore, 0) < 76 && safeNum(player.ovr, 0) < 80) {
        return false;
      }

      if (showWarningsOnly) {
        const health = normalizeHealth(player);
        const morale = getMoraleBand(player.morale);
        const fatigue = getFatigueBand(player.fatigue);
        const capHit = safeNum(player.contract?.capHit, 0);

        const hasWarning =
          health.isInjured ||
          morale.tone === "bad" ||
          fatigue.tone === "bad" ||
          (capHit >= 8 && safeNum(player.ovr, 0) < 86) ||
          safeNum(player.growth, 0) <= -1;

        if (!hasWarning) return false;
      }

      return true;
    });

    return [...rows].sort((a, b) => comparePlayers(a, b, sortKey));
  }, [
    players,
    searchTerm,
    positionFilter,
    leagueFilter,
    statusFilter,
    typeFilter,
    roleFilter,
    showCoreOnly,
    showWarningsOnly,
    sortKey,
  ]);

  useEffect(() => {
    setTablePage(0);
  }, [
    browseSource,
    searchMode,
    orgTeamId,
    orgLevel,
    devLeagueIdx,
    devTeamIdx,
    searchTerm,
    positionFilter,
    leagueFilter,
    statusFilter,
    typeFilter,
    roleFilter,
    sortKey,
    showCoreOnly,
    showWarningsOnly,
    viewMode,
  ]);

  useEffect(() => {
    if (!filteredPlayers.length) {
      setSelectedPlayerKey("");
      setProfileOpen(false);
      if (typeof setRosterRowIndex === "function") setRosterRowIndex(0);
      return;
    }

    if (selectedPlayerKey && !filteredPlayers.some((player) => player.key === selectedPlayerKey)) {
      setSelectedPlayerKey("");
      setProfileOpen(false);
    }
  }, [filteredPlayers, selectedPlayerKey, setRosterRowIndex]);

  const selectedPlayer = useMemo(() => {
    if (!selectedPlayerKey) return null;
    return filteredPlayers.find((player) => player.key === selectedPlayerKey) || null;
  }, [filteredPlayers, selectedPlayerKey]);

  const selectedPlayerIndex = useMemo(() => {
    if (!selectedPlayer) return -1;
    return filteredPlayers.findIndex((player) => player.key === selectedPlayer.key);
  }, [filteredPlayers, selectedPlayer]);

  const highlightPlayer = useCallback(
    (player) => {
      if (!player) return;

      const index = filteredPlayers.findIndex((row) => row.key === player.key);
      setSelectedPlayerKey(player.key);

      if (index >= 0 && typeof setRosterRowIndex === "function") {
        setRosterRowIndex(index);
      }
    },
    [filteredPlayers, setRosterRowIndex]
  );

  const handleSelectPlayer = useCallback(
    (player) => {
      if (!player) return;

      highlightPlayer(player);
      setProfileOpen(true);
      setActiveTab("overview");
    },
    [highlightPlayer]
  );

  const clearSelection = useCallback(() => {
    setProfileOpen(false);
    setDetailsCollapsed(false);
  }, []);

  const selectedStorylines = useMemo(() => {
    if (!selectedPlayer) return EMPTY_ARRAY;

    const selectedId = String(selectedPlayer.id || selectedPlayer.player_id || "").toLowerCase();
    const selectedName = safeStr(selectedPlayer.name, "").toLowerCase();
    const rows = franchiseState?.storyline_events || franchiseState?.storylineEvents || EMPTY_ARRAY;

    return rows
      .filter((event) => {
        const eventPlayerId = String(event?.player_id || event?.playerId || "").toLowerCase();
        const eventPlayerName = safeStr(event?.player_name || event?.playerName || event?.player, "").toLowerCase();
        const eventPlayers = Array.isArray(event?.players)
          ? event.players.map((value) => safeStr(value, "").toLowerCase())
          : EMPTY_ARRAY;

        return (
          (selectedId && eventPlayerId === selectedId) ||
          (selectedName && eventPlayerName === selectedName) ||
          (selectedName && eventPlayers.includes(selectedName))
        );
      })
      .slice(-8)
      .reverse();
  }, [franchiseState, selectedPlayer]);

  const statsLite = useMemo(() => {
    const count = filteredPlayers.length;
    const avgOVR = count
      ? filteredPlayers.reduce((sum, player) => sum + safeNum(player.ovr, 0), 0) / count
      : 0;

    const avgAge = count
      ? filteredPlayers.reduce((sum, player) => sum + safeNum(player.age, 0), 0) / count
      : 0;

    const avgPotential = count
      ? filteredPlayers.reduce((sum, player) => sum + safeNum(player.potentialScore, 0), 0) / count
      : 0;

    const injured = filteredPlayers.filter((player) => player.status === "Injured").length;
    const unsigned = filteredPlayers.filter((player) => !player.contract?.isSigned).length;
    const core = filteredPlayers.filter((player) => player.potentialScore >= 76 || player.ovr >= 80).length;

    return {
      count,
      avgOVR,
      avgAge,
      avgPotential,
      injured,
      unsigned,
      core,
    };
  }, [filteredPlayers]);

  const orgSummary = useMemo(() => {
    const rows = players;
    const nhl = rows.filter((player) => player.league === "NHL");
    const ahl = rows.filter((player) => player.league === "AHL");
    const echl = rows.filter((player) => player.league === "ECHL");

    const activeNhl = nhl.filter((player) => player.status === "Active" || player.status === "Scratched");
    const forwards = activeNhl.filter((player) => isForwardPosition(player.position));
    const defense = activeNhl.filter((player) => isDefensePosition(player.position));
    const goalies = activeNhl.filter((player) => isGoaliePosition(player.position));

    return {
      total: rows.length,
      nhl: nhl.length,
      ahl: ahl.length,
      echl: echl.length,
      activeNhl: activeNhl.length,
      forwards: forwards.length,
      defense: defense.length,
      goalies: goalies.length,
    };
  }, [players]);

  const capInfo = useMemo(() => {
    const nhlPlayers = players.filter((player) => {
      return player.league === "NHL" && player.contract?.isSigned;
    });

    const snap = franchiseState?.team?.cap_snapshot || franchiseState?.cap_snapshot || null;

    const backendCapLimit =
      safeNum(snap?.upper_limit_m, 0) ||
      safeNum(franchiseState?.team?.salary_cap, 0) ||
      safeNum(franchiseState?.team?.cap_limit, 0) ||
      safeNum(franchiseState?.salary_cap, 0) ||
      safeNum(franchiseState?.cap_limit, 0) ||
      0;

    const backendCapHit =
      safeNum(snap?.total_cap_hit_m, 0) ||
      safeNum(franchiseState?.team?.cap_hit, 0) ||
      safeNum(franchiseState?.cap_hit, 0) ||
      0;

    const backendCapSpaceRaw =
      snap?.usable_cap_space_m ??
      franchiseState?.team?.cap_space ??
      franchiseState?.cap_space;

    const computedCapHit = nhlPlayers.reduce((sum, player) => {
      return sum + safeNum(player.contract?.capHit, 0);
    }, 0);

    const capUsed = backendCapHit > 0 ? backendCapHit : computedCapHit;
    const capSpace = backendCapSpaceRaw != null && Number.isFinite(Number(backendCapSpaceRaw))
      ? Number(backendCapSpaceRaw)
      : backendCapLimit - capUsed;

    const signedContracts = players.filter((player) => player.contract?.isSigned).length;

    return {
      capLimit: backendCapLimit,
      capUsed,
      capSpace,
      signedContracts,
      contractLimit: NHL_CONTRACT_RESERVE_LIMIT,
      activeLimit: NHL_ACTIVE_ROSTER_LIMIT,
      source: backendCapHit > 0 ? "Backend" : "Computed",
    };
  }, [players, franchiseState]);

  const systemSummary = useMemo(() => {
    const nhl = Array.isArray(userOrganization?.nhl)
      ? userOrganization.nhl.length
      : (rb?.counts?.nhl_contracted ?? orgSummary.nhl);

    const ahl = Array.isArray(userOrganization?.ahl)
      ? userOrganization.ahl.length
      : (rb?.counts?.ahl_contracted ?? orgSummary.ahl);

    const echl = Array.isArray(userOrganization?.echl)
      ? userOrganization.echl.length
      : (rb?.counts?.echl_contracted ?? orgSummary.echl);

    const canCountProspects = Boolean(
      rb?.development_leagues?.length ||
        franchiseState?.prospect_pool ||
        franchiseState?.prospectPool ||
        userOrganization?.ahl?.length ||
        userOrganization?.echl?.length
    );

    const prospects = canCountProspects
      ? collectMyProspectRawPlayers(rb, franchiseState, userTeamId, userOrganization).length
      : null;

    const systemTotal = nhl + ahl + echl + (prospects ?? 0);
    const rosterSpotsLine = `${capInfo.signedContracts}/${capInfo.contractLimit}`;
    const subParts = [`NHL ${nhl}`, `AHL ${ahl}`];

    if (prospects != null) {
      subParts.push(`PRO ${prospects}`);
    } else {
      subParts.push("PRO —");
    }

    return {
      nhl,
      ahl,
      echl,
      prospects,
      systemTotal,
      rosterSpotsLine,
      subLine: subParts.join(" · "),
      totalLine: `${systemTotal} in system`,
    };
  }, [userOrganization, rb, franchiseState, userTeamId, orgSummary, capInfo]);

  const rosterWarnings = useMemo(() => {
    return buildRosterWarnings(players, capInfo);
  }, [players, capInfo]);

  const lineGroups = useMemo(() => {
    return buildLineGroups(players);
  }, [players]);

  const tableTotalPages = Math.max(1, Math.ceil(filteredPlayers.length / TABLE_PAGE_SIZE));
  const safeTablePage = clamp(tablePage, 0, tableTotalPages - 1);
  const pageStart = safeTablePage * TABLE_PAGE_SIZE;
  const pagePlayers = filteredPlayers.slice(pageStart, pageStart + TABLE_PAGE_SIZE);
  const useFullList =
    viewMode === VIEW_MODES.BOARD ||
    viewMode === VIEW_MODES.LINES ||
    filteredPlayers.length <= TABLE_PAGE_SIZE;
  const displayPlayers = useFullList ? filteredPlayers : pagePlayers;
  const needsPagination =
    viewMode !== VIEW_MODES.LINES &&
    viewMode !== VIEW_MODES.BOARD &&
    filteredPlayers.length > TABLE_PAGE_SIZE;

  useEffect(() => {
    const maxPage = Math.max(0, Math.ceil(filteredPlayers.length / TABLE_PAGE_SIZE) - 1);
    setTablePage((page) => Math.min(page, maxPage));
  }, [filteredPlayers.length]);

  const pageButtonIndices = useMemo(() => {
    const maxButtons = 7;

    if (tableTotalPages <= maxButtons) {
      return Array.from({ length: tableTotalPages }, (_, index) => index);
    }

    const half = Math.floor(maxButtons / 2);
    let start = Math.max(0, safeTablePage - half);
    let end = Math.min(tableTotalPages, start + maxButtons);

    start = Math.max(0, end - maxButtons);

    return Array.from({ length: end - start }, (_, index) => start + index);
  }, [tableTotalPages, safeTablePage]);

  const selectedPoolTitle = useMemo(() => {
    if (searchMode === PLAYER_SEARCH_MODES.NHL_LEAGUE) {
      return "NHL League · Best Players";
    }

    if (browseSource === PLAYER_POOLS.MY_PROSPECTS) {
      const orgName = userOrganization?.name || franchiseState?.team?.name || "Organization";
      return `${orgName} · My Prospects`;
    }

    if (browseSource === PLAYER_POOLS.DRAFT_CLASS) return "Draft Class";
    if (browseSource === PLAYER_POOLS.FREE_AGENTS) return "Free Agents";
    if (browseSource === PLAYER_POOLS.OVERSEAS) return "Overseas / Unsigned";
    if (browseSource === PLAYER_POOLS.DEVELOPMENT) {
      const league = devLeagues[devLeagueIdx];
      const team = devTeams[devTeamIdx];

      return [
        league?.league_name || league?.league_code || "Development",
        team?.name || team?.team_name,
      ]
        .filter(Boolean)
        .join(" · ");
    }

    const level = orgLevel.toUpperCase();
    const name = selectedOrganization?.name || franchiseState?.team?.name || "Organization";

    return `${name} · ${level}`;
  }, [
    searchMode,
    browseSource,
    orgLevel,
    selectedOrganization,
    userOrganization,
    franchiseState,
    devLeagues,
    devLeagueIdx,
    devTeams,
    devTeamIdx,
  ]);

  const countsLabel = useMemo(() => {
    if (browseSource === PLAYER_POOLS.DRAFT_CLASS) {
      const total = draftBoard?.total ?? draftBoard?.entries?.length ?? players.length;
      const subtitle = draftBoard?.subtitle || draftBoard?.title || "Draft board";

      return `${subtitle} · ${players.length}/${total}`;
    }

    if (rb?.counts) {
      return [
        `NHL ${rb.counts.nhl_contracted ?? orgSummary.nhl}`,
        `AHL ${rb.counts.ahl_contracted ?? orgSummary.ahl}`,
        `ECHL ${rb.counts.echl_contracted ?? orgSummary.echl}`,
        `UFA ${rb.counts.free_agents ?? "—"}`,
        `Overseas ${rb.counts.overseas ?? "—"}`,
        `Dev ${rb.counts.junior_skaters ?? "—"}`,
      ].join(" · ");
    }

    return `${orgSummary.total} loaded players`;
  }, [browseSource, draftBoard, players.length, rb?.counts, orgSummary]);

  const activeTeamLabel = safeStr(
    franchiseState?.team?.name ||
      franchiseState?.team?.full_name ||
      franchiseState?.team?.fullName ||
      franchiseState?.team?.abbr ||
      franchiseState?.team?.abbreviation,
    "Franchise"
  );

  const canRefresh = typeof refreshFranchise === "function";

  const handleRefresh = useCallback(() => {
    if (canRefresh) {
      refreshFranchise();
    }
  }, [canRefresh, refreshFranchise]);

  const resetFilters = useCallback(() => {
    setSearchTerm("");
    setPositionFilter("ALL");
    setLeagueFilter("ALL");
    setStatusFilter("All");
    setTypeFilter("ALL");
    setRoleFilter("ALL");
    setSortKey("overall_desc");
    setShowCoreOnly(false);
    setShowWarningsOnly(false);
    setAdvancedFiltersOpen(false);
    setTablePage(0);
  }, []);

  const openScreen = useCallback(
    (screen) => {
      if (typeof setScreen === "function") {
        setScreen(screen);
      }
    },
    [setScreen]
  );

  const handleKeyDown = useCallback(
    (event) => {
      if (event.target?.matches?.("input, textarea, select, button")) return;

      if (event.key === "Escape") {
        event.preventDefault();

        if (drawerOpen) {
          setDrawerOpen(false);
          return;
        }

        if (profileOpen) {
          clearSelection();
          return;
        }

        openScreen(SCREENS.HUB);
        return;
      }

      if (event.key === "Enter") {
        if (!filteredPlayers.length || profileOpen) return;

        event.preventDefault();
        const currentIndex = selectedPlayerIndex >= 0 ? selectedPlayerIndex : 0;
        handleSelectPlayer(filteredPlayers[currentIndex]);
        return;
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();

        if (!filteredPlayers.length) return;

        const currentIndex = selectedPlayerIndex >= 0 ? selectedPlayerIndex : 0;
        const nextIndex = Math.max(0, currentIndex - 1);
        highlightPlayer(filteredPlayers[nextIndex]);

        const nextPage = Math.floor(nextIndex / TABLE_PAGE_SIZE);
        setTablePage(nextPage);
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();

        if (!filteredPlayers.length) return;

        const currentIndex = selectedPlayerIndex >= 0 ? selectedPlayerIndex : 0;
        const nextIndex = Math.min(filteredPlayers.length - 1, currentIndex + 1);
        highlightPlayer(filteredPlayers[nextIndex]);

        const nextPage = Math.floor(nextIndex / TABLE_PAGE_SIZE);
        setTablePage(nextPage);
        return;
      }

      if (event.key === "ArrowLeft") {
        if (!profileOpen) return;

        event.preventDefault();

        const current = PANEL_TABS.findIndex((tab) => tab.value === activeTab);
        const next = current <= 0 ? PANEL_TABS.length - 1 : current - 1;
        setActiveTab(PANEL_TABS[next].value);
        return;
      }

      if (event.key === "ArrowRight") {
        if (!profileOpen) return;

        event.preventDefault();

        const current = PANEL_TABS.findIndex((tab) => tab.value === activeTab);
        const next = current >= PANEL_TABS.length - 1 ? 0 : current + 1;
        setActiveTab(PANEL_TABS[next].value);
      }
    },
    [
      drawerOpen,
      openScreen,
      filteredPlayers,
      selectedPlayerIndex,
      handleSelectPlayer,
      highlightPlayer,
      activeTab,
      profileOpen,
      clearSelection,
    ]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);

    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  useEffect(() => {
    const onMouseDown = (event) => {
      const insideDrawer = event.target?.closest?.(".nhlrost-drawer, .nhlrost-menu-button");

      if (!insideDrawer) {
        setDrawerOpen(false);
      }
    };

    document.addEventListener("mousedown", onMouseDown);

    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  const poolOptions = useMemo(() => {
    const options = [
      { value: PLAYER_POOLS.ORGANIZATION, label: "Organization" },
      { value: PLAYER_POOLS.MY_PROSPECTS, label: "My Prospects" },
      { value: PLAYER_POOLS.FREE_AGENTS, label: "Free Agents" },
      { value: PLAYER_POOLS.OVERSEAS, label: "Overseas / Unsigned" },
      { value: PLAYER_POOLS.DEVELOPMENT, label: "Development Leagues" },
    ];

    const hasDraft =
      Array.isArray(draftBoard?.entries) ||
      Array.isArray(draftBoard?.players) ||
      Array.isArray(draftBoard?.rankings);

    if (hasDraft) {
      options.push({ value: PLAYER_POOLS.DRAFT_CLASS, label: "Draft Class" });
    }

    return options;
  }, [draftBoard]);

  const orgOptions = useMemo(() => {
    return organizations.map((org, index) => ({
      value: String(org.team_id || org.id || org.abbr || index),
      label: org.name || org.team_name || org.abbr || `Organization ${index + 1}`,
    }));
  }, [organizations]);

  const devLeagueOptions = useMemo(() => {
    return devLeagues.map((league, index) => ({
      value: String(index),
      label: league.league_name || league.league_code || `League ${index + 1}`,
    }));
  }, [devLeagues]);

  const devTeamOptions = useMemo(() => {
    return devTeams.map((team, index) => ({
      value: String(index),
      label: team.name || team.team_name || `Team ${index + 1}`,
    }));
  }, [devTeams]);

  const leaguePoolFilter = useMemo(() => {
    if (searchMode === PLAYER_SEARCH_MODES.NHL_LEAGUE) return "nhl";
    if (browseSource === PLAYER_POOLS.MY_PROSPECTS) return "rights";
    if (browseSource === PLAYER_POOLS.ORGANIZATION) {
      if (orgLevel === "ahl") return "ahl";
      if (orgLevel === "echl") return "echl";
      return "nhl";
    }
    return "nhl";
  }, [searchMode, browseSource, orgLevel]);

  const handleSearchModeChange = useCallback((value) => {
    setTablePage(0);
    setSearchMode(value);

    if (value === PLAYER_SEARCH_MODES.NHL_LEAGUE) {
      setBrowseSource(PLAYER_POOLS.ORGANIZATION);
      setOrgLevel("nhl");
      setSortKey("overall_desc");
      return;
    }

    setBrowseSource(PLAYER_POOLS.ORGANIZATION);
  }, []);

  const handleOrgTeamChange = useCallback((event) => {
    setTablePage(0);
    setOrgTeamId(event.target.value);
  }, []);

  const handleLeaguePoolChange = useCallback((value) => {
    setTablePage(0);

    if (value === "rights") {
      setBrowseSource(PLAYER_POOLS.MY_PROSPECTS);
      return;
    }

    setBrowseSource(PLAYER_POOLS.ORGANIZATION);
    setOrgLevel(value);
  }, []);

  const showTeamColumn = searchMode === PLAYER_SEARCH_MODES.NHL_LEAGUE;

  const showPoolColumn =
    showTeamColumn ||
    browseSource === PLAYER_POOLS.MY_PROSPECTS ||
    browseSource !== PLAYER_POOLS.ORGANIZATION ||
    leagueFilter === "ALL" ||
    viewMode === VIEW_MODES.TABLE ||
    viewMode === VIEW_MODES.BOARD;

  return (
    <div className="nhlrost-root">
      <RosterScreenStyles />

      <aside className="nhlrost-sidebar">
        <button
          type="button"
          className="nhlrost-brand"
          onClick={() => openScreen(SCREENS.HUB)}
          title="Back to Hub"
        >
          <span>⌂</span>
        </button>

        <nav className="nhlrost-side-nav" aria-label="Roster navigation">
          <button type="button" onClick={() => openScreen(SCREENS.HUB)}>
            <span>▦</span>
            <em>Hub</em>
          </button>

          <button type="button" className="is-active">
            <span>◫</span>
            <em>Roster</em>
          </button>

          <button type="button" onClick={() => openScreen(SCREENS.STATS)}>
            <span>▤</span>
            <em>Stats</em>
          </button>

          <button type="button" onClick={() => openScreen(SCREENS.STORYLINES)}>
            <span>📰</span>
            <em>Stories</em>
          </button>

          <button type="button" onClick={() => openScreen(SCREENS.TRADE || SCREENS.TRADE_HUB || SCREENS.HUB)}>
            <span>⇄</span>
            <em>Trade</em>
          </button>
        </nav>

        <button
          type="button"
          className="nhlrost-menu-button"
          onClick={() => setDrawerOpen((value) => !value)}
          aria-expanded={drawerOpen}
          title="Roster command drawer"
        >
          <span />
          <span />
          <span />
          <em>Menu</em>
        </button>
      </aside>

      <main className="nhlrost-main">
        <header className="nhlrost-command-bar">
          <CommandBarTeamLogo franchiseState={franchiseState} />

          <div className="nhlrost-command-bar__metrics">
            <article className="nhlrost-hud-tile">
              <div className="nhlrost-hud-tile__body">
                <small>Roster</small>
                <strong>
                  {filteredPlayers.length}/{players.length}
                </strong>
              </div>
            </article>

            <article className={`nhlrost-hud-tile ${capInfo.capSpace < 0 ? "is-danger" : ""}`}>
              <div className="nhlrost-hud-tile__body">
                <small>Cap</small>
                <strong>
                  {capInfo.capSpace < 0
                    ? `-${formatMoneyMillions(Math.abs(capInfo.capSpace))}`
                    : formatMoneyMillions(capInfo.capSpace)}
                </strong>
              </div>
            </article>

            <article className="nhlrost-hud-tile nhlrost-hud-tile--system">
              <div className="nhlrost-hud-tile__body">
                <small>Roster Spots</small>
                <strong>{systemSummary.rosterSpotsLine}</strong>
                <em>{systemSummary.subLine}</em>
              </div>
            </article>

            <article className={`nhlrost-hud-tile ${statsLite.injured ? "is-warn" : "is-good"}`}>
              <div className="nhlrost-hud-tile__body">
                <small>Inj</small>
                <strong>{statsLite.injured}</strong>
              </div>
            </article>

            <article className="nhlrost-hud-tile">
              <div className="nhlrost-hud-tile__body">
                <small>OVR</small>
                <strong>{statsLite.avgOVR ? round1(statsLite.avgOVR) : "—"}</strong>
              </div>
            </article>

            <article className="nhlrost-hud-tile">
              <div className="nhlrost-hud-tile__body">
                <small>Age</small>
                <strong>{statsLite.avgAge ? round1(statsLite.avgAge) : "—"}</strong>
              </div>
            </article>

            {rosterWarnings.length ? (
              <article className={`nhlrost-hud-tile nhlrost-attention-chip ${showWarningsOnly ? "is-active" : ""}`}>
                <div className="nhlrost-hud-tile__body">
                  <small>Alert</small>
                  <strong>{rosterWarnings.length}</strong>
                </div>
              </article>
            ) : null}
          </div>
        </header>

        <section className="nhlrost-filters-bar">
          <div className="nhlrost-filters-primary">
            <PlayerSearchModeSegmented value={searchMode} onChange={handleSearchModeChange} />

            {searchMode === PLAYER_SEARCH_MODES.TEAM_ROSTERS ? (
              <>
                <ToolbarSelect
                  id="roster-team"
                  label="Team"
                  compact
                  value={String(orgTeamId || defaultOrgId || "")}
                  onChange={handleOrgTeamChange}
                  options={orgOptions}
                />

                <LeaguePoolSegmented value={leaguePoolFilter} onChange={handleLeaguePoolChange} />
              </>
            ) : null}

            <ToolbarInput
              id="roster-search"
              label="Search"
              compact
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Search player…"
            />

            <ToolbarSelect
              id="roster-position"
              label="Position"
              compact
              value={positionFilter}
              onChange={(event) => setPositionFilter(event.target.value)}
              options={POSITION_FILTERS}
            />

            <ToolbarSelect
              id="roster-sort"
              label="Sort"
              compact
              value={sortKey}
              onChange={(event) => setSortKey(event.target.value)}
              options={SORT_KEYS}
            />
          </div>
        </section>

        <section className="nhlrost-board-shell">
          <header className="nhlrost-board-shell__head">
            <span>
              {searchMode === PLAYER_SEARCH_MODES.NHL_LEAGUE
                ? `NHL League · ${filteredPlayers.length} players`
                : `${selectedPoolTitle} · ${filteredPlayers.length} players`}
            </span>
          </header>

          <div className="nhlrost-board-shell__body">
            {browseSource === PLAYER_POOLS.DRAFT_CLASS ? (
              <DraftBoardTable
                players={displayPlayers}
                selectedPlayerKey={selectedPlayerKey}
                onSelectPlayer={handleSelectPlayer}
                pageOffset={useFullList ? 0 : pageStart}
              />
            ) : (
              <RosterBoardView
                players={displayPlayers}
                selectedPlayerKey={selectedPlayerKey}
                onSelectPlayer={handleSelectPlayer}
                showTeam={showTeamColumn}
              />
            )}
          </div>

          {needsPagination ? (
            <footer className="nhlrost-pagination">
              <button
                type="button"
                disabled={safeTablePage <= 0}
                onClick={() => setTablePage((page) => Math.max(0, page - 1))}
              >
                ‹ Prev
              </button>

              <div>
                {pageButtonIndices.map((pageIndex) => (
                  <button
                    type="button"
                    key={pageIndex}
                    className={pageIndex === safeTablePage ? "is-active" : ""}
                    onClick={() => setTablePage(pageIndex)}
                  >
                    {pageIndex + 1}
                  </button>
                ))}
              </div>

              <button
                type="button"
                disabled={safeTablePage >= tableTotalPages - 1}
                onClick={() => setTablePage((page) => Math.min(tableTotalPages - 1, page + 1))}
              >
                Next ›
              </button>
            </footer>
          ) : null}
        </section>

        {profileOpen && selectedPlayer ? (
          <PlayerProfileModal
            player={selectedPlayer}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            storylines={selectedStorylines}
            onClose={clearSelection}
          />
        ) : null}
      </main>

      {drawerOpen ? (
        <RosterCommandDrawer
          activeTeamLabel={activeTeamLabel}
          selectedPoolTitle={selectedPoolTitle}
          selectedPlayer={selectedPlayer}
          statsLite={statsLite}
          orgSummary={orgSummary}
          capInfo={capInfo}
          rosterWarnings={rosterWarnings}
          onClose={() => setDrawerOpen(false)}
          onNavigate={openScreen}
          canRefresh={canRefresh}
          onRefresh={handleRefresh}
        />
      ) : null}

      <GameFooter hints="↑↓ PLAYERS · ENTER PROFILE · ESC CLOSE · MENU" />
    </div>
  );
}
function RosterCommandDrawer({
  activeTeamLabel,
  selectedPoolTitle,
  selectedPlayer,
  statsLite,
  orgSummary,
  capInfo,
  rosterWarnings,
  onClose,
  onNavigate,
  canRefresh,
  onRefresh,
}) {
  const [tab, setTab] = useState("hub");

  return (
    <div className="nhlrost-drawer-backdrop" onMouseDown={onClose}>
      <aside className="nhlrost-drawer" onMouseDown={(event) => event.stopPropagation()}>
        <header className="nhlrost-drawer__head">
          <div>
            <p>Roster Command</p>
            <h2>{safeStr(activeTeamLabel, "Franchise")}</h2>
            <span>{selectedPoolTitle}</span>
          </div>

          <button type="button" onClick={onClose} aria-label="Close roster drawer">
            ×
          </button>
        </header>

        <nav className="nhlrost-drawer__tabs">
          <button
            type="button"
            className={tab === "hub" ? "is-active" : ""}
            onClick={() => setTab("hub")}
          >
            Hub
          </button>

          <button
            type="button"
            className={tab === "selected" ? "is-active" : ""}
            onClick={() => setTab("selected")}
          >
            Selected
          </button>

          <button
            type="button"
            className={tab === "warnings" ? "is-active" : ""}
            onClick={() => setTab("warnings")}
          >
            Warnings
          </button>

          <button
            type="button"
            className={tab === "nav" ? "is-active" : ""}
            onClick={() => setTab("nav")}
          >
            Navigation
          </button>
        </nav>

        {tab === "hub" ? (
          <div className="nhlrost-drawer__body">
            <section className="nhlrost-drawer-section">
              <h3>Roster Snapshot</h3>

              <div className="nhlrost-drawer-metric-grid">
                <article>
                  <span>Loaded</span>
                  <strong>{statsLite.count}</strong>
                </article>

                <article>
                  <span>Avg OVR</span>
                  <strong>{statsLite.avgOVR ? round1(statsLite.avgOVR) : "—"}</strong>
                </article>

                <article>
                  <span>Avg Age</span>
                  <strong>{statsLite.avgAge ? round1(statsLite.avgAge) : "—"}</strong>
                </article>

                <article>
                  <span>Avg POT</span>
                  <strong>{statsLite.avgPotential ? round0(statsLite.avgPotential) : "—"}</strong>
                </article>

                <article>
                  <span>NHL</span>
                  <strong>{orgSummary.nhl}</strong>
                </article>

                <article>
                  <span>AHL</span>
                  <strong>{orgSummary.ahl}</strong>
                </article>

                <article>
                  <span>ECHL</span>
                  <strong>{orgSummary.echl}</strong>
                </article>

                <article>
                  <span>Injured</span>
                  <strong>{statsLite.injured}</strong>
                </article>
              </div>
            </section>

            <section className="nhlrost-drawer-section">
              <h3>Cap Snapshot</h3>

              <div className="nhlrost-drawer-feed">
                <article>
                  <strong>Cap Used</strong>
                  <p>{formatMoneyMillions(capInfo.capUsed)} / {formatMoneyMillions(capInfo.capLimit)}</p>
                </article>

                <article className={capInfo.capSpace < 0 ? "is-danger" : ""}>
                  <strong>Cap Space</strong>
                  <p>
                    {capInfo.capSpace < 0
                      ? `-${formatMoneyMillions(Math.abs(capInfo.capSpace))}`
                      : formatMoneyMillions(capInfo.capSpace)}
                  </p>
                </article>

                <article>
                  <strong>Contract Slots</strong>
                  <p>{capInfo.signedContracts} / {capInfo.contractLimit}</p>
                </article>

                <article>
                  <strong>Cap Source</strong>
                  <p>{capInfo.source}</p>
                </article>
              </div>
            </section>

            <section className="nhlrost-drawer-section">
              <h3>Roster Tools</h3>

              <div className="nhlrost-drawer-actions">
                {canRefresh ? (
                  <button type="button" onClick={onRefresh}>
                    Refresh Roster Data
                  </button>
                ) : (
                  <span>Roster refresh unavailable</span>
                )}
              </div>

              <p className="nhlrost-drawer-note">
                Transaction controls are not connected yet. This screen is currently in scouting and read-only mode.
              </p>
            </section>
          </div>
        ) : null}

        {tab === "selected" ? (
          <div className="nhlrost-drawer__body">
            {selectedPlayer ? (
              <>
                <section className="nhlrost-drawer-player">
                  <PlayerAvatar player={selectedPlayer} size="xl" />

                  <div>
                    <p>Selected Player</p>
                    <h3>{selectedPlayer.name}</h3>
                    <span>
                      {selectedPlayer.position} · {selectedPlayer.age} · {selectedPlayer.teamName}
                    </span>
                  </div>
                </section>

                <section className="nhlrost-drawer-section">
                  <h3>Player Read</h3>

                  <div className="nhlrost-drawer-metric-grid">
                    <article>
                      <span>OVR</span>
                      <strong>{selectedPlayer.ovr || "—"}</strong>
                    </article>

                    <article>
                      <span>True</span>
                      <strong>{selectedPlayer.trueOverall ? round1(selectedPlayer.trueOverall) : "—"}</strong>
                    </article>

                    <article>
                      <span>Potential</span>
                      <strong>{selectedPlayer.potentialScore || "—"}</strong>
                    </article>

                    <article>
                      <span>Age</span>
                      <strong>{selectedPlayer.age}</strong>
                    </article>

                    <article>
                      <span>Morale</span>
                      <strong>{round0(selectedPlayer.morale)}</strong>
                    </article>

                    <article>
                      <span>Fatigue</span>
                      <strong>{round0(selectedPlayer.fatigue)}</strong>
                    </article>

                    <article>
                      <span>Cap</span>
                      <strong>{formatMoneyMillions(selectedPlayer.contract?.capHit)}</strong>
                    </article>

                    <article>
                      <span>Asset</span>
                      <strong>{selectedPlayer.asset?.label || "—"}</strong>
                    </article>
                  </div>
                </section>

                <section className="nhlrost-drawer-section">
                  <h3>Scouting Summary</h3>
                  <p className="nhlrost-drawer-note">{selectedPlayer.note}</p>
                </section>
              </>
            ) : (
              <EmptyPanel title="No player selected" body="No player matches the current filters." />
            )}
          </div>
        ) : null}

        {tab === "warnings" ? (
          <div className="nhlrost-drawer__body">
            <section className="nhlrost-drawer-section">
              <h3>Roster Warnings</h3>

              {rosterWarnings.length ? (
                <div className="nhlrost-drawer-feed">
                  {rosterWarnings.map((warning) => (
                    <article key={warning.key} className={toneClass(warning.tone)}>
                      <strong>{warning.title}</strong>
                      <p>{warning.body}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="nhlrost-drawer-note">
                  No roster warnings are active from the currently loaded roster data.
                </p>
              )}
            </section>

            <section className="nhlrost-drawer-section">
              <h3>What this screen checks</h3>

              <div className="nhlrost-drawer-feed">
                <article>
                  <strong>Active roster limit</strong>
                  <p>Compares loaded NHL active players against the 23-player active roster limit.</p>
                </article>

                <article>
                  <strong>Lineup minimums</strong>
                  <p>Checks whether the roster has enough forwards, defensemen, and goalies loaded.</p>
                </article>

                <article>
                  <strong>Injury impact</strong>
                  <p>Flags loaded players marked as injured, day-to-day, or LTIR.</p>
                </article>

                <article>
                  <strong>Cap state</strong>
                  <p>Shows over-cap status instead of hiding negative cap space.</p>
                </article>
              </div>
            </section>
          </div>
        ) : null}

        {tab === "nav" ? (
          <div className="nhlrost-drawer__body">
            <section className="nhlrost-drawer-section">
              <h3>Franchise Navigation</h3>

              <div className="nhlrost-drawer-nav-grid">
                <button type="button" onClick={() => onNavigate(SCREENS.HUB)}>
                  <strong>Hub</strong>
                  <span>Franchise command center</span>
                </button>

                <button type="button" onClick={() => onNavigate(SCREENS.CALENDAR)}>
                  <strong>Calendar</strong>
                  <span>Schedule and events</span>
                </button>

                <button type="button" onClick={() => onNavigate(SCREENS.STATS)}>
                  <strong>Stats</strong>
                  <span>League and team stats</span>
                </button>

                <button type="button" onClick={() => onNavigate(SCREENS.STORYLINES)}>
                  <strong>Storylines</strong>
                  <span>Player and league events</span>
                </button>

                <button type="button" onClick={() => onNavigate(SCREENS.DRAFT_CLASS || SCREENS.STATS)}>
                  <strong>Draft</strong>
                  <span>Draft class / scouting</span>
                </button>

                <button type="button" onClick={() => onNavigate(SCREENS.OFFICE || SCREENS.HUB)}>
                  <strong>Office</strong>
                  <span>Inbox, owner, finance</span>
                </button>
              </div>
            </section>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function RosterScreenStyles() {
  return (
    <style>{`
      .nhlrost-root {
        --bg: #06121d;
        --bg-2: #04101a;
        --panel: rgba(10, 28, 42, 0.94);
        --panel-2: rgba(12, 36, 54, 0.92);
        --panel-3: rgba(17, 49, 72, 0.82);
        --line: rgba(150, 190, 210, 0.16);
        --line-2: rgba(150, 220, 235, 0.28);
        --text: #e8f4fb;
        --muted: #8ba0af;
        --muted-2: #617484;
        --cyan: #00d8df;
        --blue: #62b7ff;
        --gold: #e8a536;
        --green: #48d88b;
        --red: #ff6464;
        --orange: #ff9f43;
        --purple: #b18cff;
        min-height: 100vh;
        width: 100%;
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr);
        overflow: hidden;
        background:
          radial-gradient(circle at 22% 0%, rgba(0, 216, 223, 0.13), transparent 28%),
          radial-gradient(circle at 88% 12%, rgba(232, 165, 54, 0.12), transparent 24%),
          linear-gradient(180deg, #06111b, #03080e 72%);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .nhlrost-root *,
      .nhlrost-root *::before,
      .nhlrost-root *::after {
        box-sizing: border-box;
      }

      .nhlrost-root button,
      .nhlrost-root input,
      .nhlrost-root select {
        font: inherit;
      }

      .nhlrost-root button {
        color: inherit;
      }

      .nhlrost-sidebar {
        min-height: 100vh;
        padding: 14px 8px 18px;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        background: linear-gradient(180deg, rgba(8, 23, 36, 0.98), rgba(4, 12, 20, 0.98));
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        z-index: 5;
      }

      .nhlrost-brand {
        width: 48px;
        height: 48px;
        border-radius: 16px;
        border: 1px solid rgba(0, 216, 223, 0.28);
        background: rgba(0, 216, 223, 0.08);
        display: grid;
        place-items: center;
        cursor: pointer;
      }

      .nhlrost-brand span {
        font-size: 1.1rem;
        color: var(--cyan);
      }

      .nhlrost-side-nav {
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 10px;
        align-items: stretch;
      }

      .nhlrost-side-nav button,
      .nhlrost-menu-button {
        width: 100%;
        min-height: 48px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.025);
        color: var(--muted);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 4px;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease,
          color 150ms ease;
      }

      .nhlrost-side-nav button:hover,
      .nhlrost-menu-button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.28);
        background: rgba(0, 216, 223, 0.07);
        color: var(--text);
      }

      .nhlrost-side-nav button.is-active {
        color: #ffffff;
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.1);
      }

      .nhlrost-side-nav span {
        font-size: 0.95rem;
        line-height: 1;
      }

      .nhlrost-side-nav em,
      .nhlrost-menu-button em {
        font-size: 0.52rem;
        font-style: normal;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 800;
      }

      .nhlrost-menu-button {
        margin-top: auto;
      }

      .nhlrost-menu-button > span {
        width: 24px;
        height: 2px;
        border-radius: 99px;
        background: currentColor;
      }

      .nhlrost-main {
        min-width: 0;
        min-height: 100vh;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 14px 18px 12px;
      }

      .nhlrost-panel {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          var(--panel);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.07),
          0 18px 40px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(14px);
      }

      .nhlrost-command-bar {
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: 96px minmax(0, 1fr);
        gap: 12px;
        align-items: center;
        min-height: 78px;
        padding: 10px 14px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          rgba(14, 36, 54, 0.58);
        backdrop-filter: blur(14px);
      }

      .nhlrost-command-bar__metrics {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 6px;
        min-width: 0;
        align-items: stretch;
      }

      .nhlrost-command-bar__metrics:has(.nhlrost-attention-chip) {
        grid-template-columns: repeat(7, minmax(0, 1fr));
      }

      .nhlrost-hud-tile {
        min-width: 0;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background: rgba(255, 255, 255, 0.03);
        padding: 7px 10px;
        display: flex;
        align-items: center;
      }

      .nhlrost-hud-tile__body {
        min-width: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 1px;
      }

      .nhlrost-hud-tile__body small {
        color: var(--muted);
        font-size: 0.52rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 900;
        line-height: 1;
        white-space: nowrap;
      }

      .nhlrost-hud-tile__body strong {
        font-size: 0.92rem;
        line-height: 1.05;
        color: #fff;
        white-space: nowrap;
      }

      .nhlrost-hud-tile__body em {
        font-style: normal;
        font-size: 0.52rem;
        line-height: 1.15;
        color: var(--muted-2);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
      }

      .nhlrost-hud-tile--system {
        grid-column: span 1;
      }

      .nhlrost-hud-tile.is-danger {
        border-color: rgba(255, 100, 100, 0.32);
        background: rgba(255, 100, 100, 0.06);
      }

      .nhlrost-hud-tile.is-warn {
        border-color: rgba(255, 159, 67, 0.32);
        background: rgba(255, 159, 67, 0.06);
      }

      .nhlrost-hud-tile.is-good strong {
        color: var(--green);
      }

      .nhlrost-attention-chip.is-active {
        border-color: rgba(255, 159, 67, 0.42);
        background: rgba(255, 159, 67, 0.1);
      }

      .nhlrost-filters-bar {
        flex: 0 0 auto;
        padding: 10px 14px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          rgba(14, 36, 54, 0.58);
        backdrop-filter: blur(14px);
      }

      .nhlrost-filters-primary {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
      }

      .nhlrost-search-mode-segment,
      .nhlrost-pool-segment {
        display: grid;
        gap: 6px;
        min-width: 0;
        padding: 4px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
      }

      .nhlrost-search-mode-segment {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        min-width: 240px;
      }

      .nhlrost-search-mode-segment button,
      .nhlrost-pool-segment button {
        min-height: 34px;
        border: 0;
        border-radius: 10px;
        background: transparent;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          background 140ms ease,
          color 140ms ease,
          box-shadow 140ms ease;
      }

      .nhlrost-search-mode-segment button:hover,
      .nhlrost-pool-segment button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.05);
      }

      .nhlrost-search-mode-segment button.is-active,
      .nhlrost-pool-segment button.is-active {
        color: #031018;
        background: linear-gradient(180deg, #00e2e8, #00b9c2);
        box-shadow: 0 0 16px rgba(0, 216, 223, 0.24);
      }

      .nhlrost-pool-segment {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        min-width: 280px;
      }

      .nhlrost-pool-segment button {
        min-height: 40px;
        border-radius: 10px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--muted);
        padding: 0 8px;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
        white-space: nowrap;
        transition:
          color 150ms ease,
          background 150ms ease,
          border-color 150ms ease;
      }

      .nhlrost-pool-segment button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.04);
      }

      .nhlrost-pool-segment button.is-active {
        color: #fff;
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.12);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
      }

      .nhlrost-filters-advanced {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        padding-top: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-view-modes {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 5px;
      }

      .nhlrost-view-modes__label {
        color: var(--muted);
        font-size: 0.61rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 950;
      }

      .nhlrost-view-modes__buttons {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
      }

      .nhlrost-view-modes__buttons button {
        min-height: 34px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        padding: 0 10px;
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .nhlrost-view-modes__buttons button:hover,
      .nhlrost-view-modes__buttons button.is-active {
        color: var(--text);
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.1);
      }

      .nhlrost-board-shell {
        flex: 1 1 auto;
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) auto;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(6, 18, 29, 0.78);
        overflow: hidden;
      }

      .nhlrost-board-shell__head {
        padding: 8px 14px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-board-shell__head > span {
        border-radius: 999px;
        padding: 4px 10px;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.14);
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlrost-board-shell__body {
        min-height: 0;
        overflow: auto;
        padding: 10px 12px 12px;
      }

      .nhlrost-board {
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .nhlrost-board-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlrost-board-row {
        width: 100%;
        min-height: 68px;
        display: grid;
        grid-template-columns:
          148px
          minmax(160px, 1.5fr)
          48px
          44px
          80px
          minmax(108px, 0.95fr)
          minmax(180px, 1.4fr)
          92px;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        background: rgba(255, 255, 255, 0.028);
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition:
          border-color 140ms ease,
          background 140ms ease,
          box-shadow 140ms ease;
      }

      .nhlrost-board-row:hover {
        border-color: rgba(0, 216, 223, 0.24);
        background: rgba(0, 216, 223, 0.05);
      }

      .nhlrost-board-row.is-selected {
        border-color: rgba(0, 216, 223, 0.48);
        background: rgba(0, 216, 223, 0.09);
        box-shadow: 0 0 18px rgba(0, 216, 223, 0.12);
      }

      .nhlrost-board-row__avatar {
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nhlrost-player-icon-plate {
        display: flex;
        align-items: center;
        width: 100%;
        max-width: 148px;
        height: 56px;
        position: relative;
        flex-shrink: 0;
        --plate-glow: rgba(0, 216, 223, 0.32);
        --plate-border: rgba(0, 216, 223, 0.42);
        --plate-inner: rgba(0, 216, 223, 0.14);
      }

      .nhlrost-player-icon-plate.pos-forward {
        --plate-glow: rgba(0, 216, 223, 0.42);
        --plate-border: rgba(45, 212, 191, 0.52);
        --plate-inner: rgba(0, 216, 223, 0.18);
      }

      .nhlrost-player-icon-plate.pos-defense {
        --plate-glow: rgba(96, 165, 250, 0.4);
        --plate-border: rgba(59, 130, 246, 0.55);
        --plate-inner: rgba(96, 165, 250, 0.16);
      }

      .nhlrost-player-icon-plate.pos-goalie {
        --plate-glow: rgba(196, 167, 255, 0.44);
        --plate-border: rgba(250, 204, 21, 0.48);
        --plate-inner: rgba(196, 167, 255, 0.18);
      }

      .nhlrost-player-icon-plate.pos-unknown {
        --plate-glow: rgba(148, 163, 184, 0.28);
        --plate-border: rgba(148, 163, 184, 0.38);
        --plate-inner: rgba(148, 163, 184, 0.12);
      }

      .nhlrost-player-icon-plate__number {
        width: 36px;
        min-width: 36px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: -8px;
        z-index: 2;
        transform: skewX(-10deg);
        background: linear-gradient(
          168deg,
          rgba(255, 255, 255, 0.14) 0%,
          rgba(255, 255, 255, 0.04) 48%,
          rgba(0, 0, 0, 0.22) 100%
        );
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 5px 3px 3px 7px;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.1),
          0 4px 10px rgba(0, 0, 0, 0.28);
      }

      .nhlrost-player-icon-plate__number span {
        transform: skewX(10deg);
        font-size: 1rem;
        font-weight: 900;
        color: #edf6ff;
        letter-spacing: -0.03em;
        line-height: 1;
        text-shadow: 0 1px 8px rgba(0, 0, 0, 0.45);
      }

      .nhlrost-player-icon-plate__portrait {
        position: relative;
        flex-shrink: 0;
        padding: 3px;
        border-radius: 11px 13px 10px 9px;
        background:
          linear-gradient(155deg, rgba(255, 255, 255, 0.1) 0%, transparent 42%),
          linear-gradient(325deg, var(--plate-inner), rgba(4, 10, 20, 0.88));
        border: 1px solid var(--plate-border);
        box-shadow:
          0 0 0 1px rgba(0, 0, 0, 0.45) inset,
          0 0 0 2px rgba(255, 255, 255, 0.04) inset,
          0 0 18px var(--plate-glow),
          0 6px 14px rgba(0, 0, 0, 0.38);
        clip-path: polygon(10% 0, 100% 0, 100% 100%, 0 100%, 0 14%);
        transition:
          transform 150ms ease,
          box-shadow 150ms ease;
      }

      .nhlrost-player-icon-plate__portrait::before {
        content: "";
        position: absolute;
        inset: 2px;
        border-radius: 8px 10px 8px 7px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: inset 0 0 14px var(--plate-inner);
        pointer-events: none;
        z-index: 1;
      }

      .nhlrost-player-icon-plate__portrait::after {
        content: "";
        position: absolute;
        left: 10%;
        right: 8%;
        bottom: 3px;
        height: 2px;
        border-radius: 2px;
        background: var(--arch-color, #9aa7bd);
        opacity: var(--arch-accent-opacity, 0.55);
        box-shadow: 0 0 10px var(--arch-color, #9aa7bd);
        z-index: 3;
        pointer-events: none;
      }

      .nhlrost-player-icon-plate__portrait .nhlrost-headshot.player-headshot.size-lg {
        --size: 64px;
        position: relative;
        z-index: 0;
        filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.45));
      }

      .nhlrost-board-row:hover .nhlrost-player-icon-plate__portrait {
        transform: translateY(-2px);
        box-shadow:
          0 0 0 1px rgba(0, 0, 0, 0.45) inset,
          0 0 0 2px rgba(255, 255, 255, 0.06) inset,
          0 0 24px var(--plate-glow),
          0 10px 18px rgba(0, 0, 0, 0.42);
      }

      .nhlrost-board-row.is-selected .nhlrost-player-icon-plate__portrait {
        box-shadow:
          0 0 0 1px rgba(0, 0, 0, 0.45) inset,
          0 0 0 2px rgba(255, 255, 255, 0.08) inset,
          0 0 26px var(--plate-glow),
          0 0 12px var(--arch-color, #9aa7bd);
      }

      .nhlrost-player-icon-plate__pos {
        position: absolute;
        bottom: 2px;
        right: 2px;
        z-index: 4;
        min-width: 20px;
        padding: 1px 5px;
        font-size: 0.55rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        text-align: center;
        color: #e8f7ff;
        background: rgba(5, 12, 24, 0.94);
        border: 1px solid var(--plate-border);
        border-radius: 4px;
        box-shadow: 0 0 8px var(--plate-glow);
        line-height: 1.1;
        pointer-events: none;
      }

      .nhlrost-board-row__name-line {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }

      .nhlrost-board-row__name-line strong {
        font-size: 0.98rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row__team {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.68rem;
        font-style: normal;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row.has-team .nhlrost-board-row__identity {
        min-width: 0;
      }

      .nhlrost-filters-primary .nhlrost-control--compact {
        min-width: 132px;
      }

      .nhlrost-filters-primary .nhlrost-control--compact input,
      .nhlrost-filters-primary .nhlrost-control--compact select {
        min-width: 132px;
      }

      .nhlrost-board-row__flag {
        width: 32px;
        height: 24px;
        object-fit: cover;
        border-radius: 3px;
        flex: 0 0 auto;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.14);
      }

      .nhlrost-board-row__flag-fallback,
      .nhlrost-flag-fallback {
        flex: 0 0 auto;
        min-width: 28px;
        height: 20px;
        padding: 0 5px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.58rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        color: #dff7ff;
        background: rgba(143, 211, 255, 0.12);
        border: 1px solid rgba(143, 211, 255, 0.28);
      }

      .nhlrost-flag-fallback.is-lg {
        min-width: 42px;
        height: 28px;
        font-size: 0.68rem;
      }

      .nhlrost-headshot .ph-flag {
        font-size: 0.58rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        opacity: 1;
      }

      .nhlrost-board-row__pos {
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        color: var(--muted);
      }

      .nhlrost-board-row__age {
        font-size: 0.92rem;
        font-weight: 800;
        color: rgba(232, 244, 251, 0.88);
        text-align: center;
      }

      .nhlrost-ovr-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        min-width: 44px;
        min-height: 34px;
        padding: 0 8px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 900;
        line-height: 1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: #eef4ff;
      }

      .nhlrost-ovr-pill__delta {
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.02em;
      }

      .nhlrost-ovr-pill__delta.is-up,
      .nhlrost-ovr-growth.is-up {
        color: #7dffb2;
      }

      .nhlrost-ovr-pill__delta.is-down,
      .nhlrost-ovr-growth.is-down {
        color: #ff8f8f;
      }

      .nhlrost-ovr-growth {
        font-style: normal;
        font-size: 0.78rem;
        font-weight: 900;
      }

      .nhlrost-ovr-pill.is-large {
        min-width: 52px;
        min-height: 40px;
        font-size: 1.35rem;
      }

      .nhlrost-ovr-pill.is-elite,
      .nhlrost-potential-pill.is-elite {
        color: #fff6d2;
        border-color: rgba(248, 210, 106, 0.45);
        background: rgba(248, 210, 106, 0.14);
        box-shadow: 0 0 16px rgba(248, 210, 106, 0.18);
      }

      .nhlrost-ovr-pill.is-franchise,
      .nhlrost-potential-pill.is-franchise {
        color: #dff7ff;
        border-color: rgba(143, 211, 255, 0.42);
        background: rgba(143, 211, 255, 0.12);
        box-shadow: 0 0 14px rgba(143, 211, 255, 0.16);
      }

      .nhlrost-ovr-pill.is-good,
      .nhlrost-potential-pill.is-good {
        color: #d7ffe7;
        border-color: rgba(185, 246, 202, 0.35);
        background: rgba(185, 246, 202, 0.1);
      }

      .nhlrost-ovr-pill.is-neutral,
      .nhlrost-potential-pill.is-neutral {
        color: #eef4ff;
        border-color: rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
      }

      .nhlrost-board-row__ovr strong {
        font-size: 1.45rem;
        line-height: 1;
        font-weight: 900;
      }

      .nhlrost-board-row__stats {
        color: rgba(232, 244, 251, 0.82);
        font-size: 0.76rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row__cap {
        color: var(--gold);
        font-size: 1rem;
        font-weight: 900;
        text-align: right;
        white-space: nowrap;
      }

      .nhlrost-potential-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 28px;
        padding: 0 12px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: #eef4ff;
      }

      .nhlrost-potential-pill.is-large {
        min-height: 34px;
        padding: 0 14px;
        font-size: 0.76rem;
      }

      .nhlrost-flag-badge {
        width: 28px;
        height: 20px;
        object-fit: cover;
        border-radius: 3px;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12);
      }

      .nhlrost-flag-badge.is-lg {
        width: 42px;
        height: 28px;
      }

      .nhlrost-profile-modal {
        position: fixed;
        inset: 0;
        z-index: 40;
        display: grid;
        place-items: center;
        padding: 24px;
      }

      .nhlrost-profile-modal__backdrop {
        position: absolute;
        inset: 0;
        border: 0;
        background: rgba(2, 8, 14, 0.82);
        backdrop-filter: blur(8px);
        cursor: pointer;
      }

      .nhlrost-profile-modal__panel {
        position: relative;
        width: clamp(900px, 88vw, 1180px);
        max-height: calc(100dvh - 48px);
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          rgba(8, 23, 36, 0.96);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
        overflow: hidden;
        scrollbar-gutter: stable;
      }

      .nhlrost-profile-modal__hero {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        gap: 14px;
        align-items: center;
        padding: 14px 16px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-profile-modal__visual {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__headshot.player-headshot.size-md {
        --size: 120px;
      }

      .nhlrost-profile-modal__identity-row {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }

      .nhlrost-profile-modal__identity-row h2 {
        margin: 0;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-profile-modal__meta h2 {
        margin: 0 0 4px;
        font-size: 1.25rem;
      }

      .nhlrost-profile-modal__meta p {
        margin: 0;
        color: var(--muted);
        font-size: 0.82rem;
      }

      .nhlrost-profile-modal__chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
        align-items: center;
      }

      .nhlrost-profile-modal__role,
      .nhlrost-profile-modal__health {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted);
        border-radius: 999px;
        padding: 4px 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
      }

      .nhlrost-profile-modal__close {
        position: sticky;
        top: 0;
        width: 40px;
        height: 40px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 1.4rem;
        line-height: 1;
        cursor: pointer;
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__body {
        min-height: 0;
        overflow: auto;
        padding: 12px 16px 16px;
      }

      .nhlrost-profile-scorecards {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 12px;
      }

      .nhlrost-profile-scorecard {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-profile-scorecard span {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlrost-profile-scorecard strong {
        font-size: 1.2rem;
      }

      .nhlrost-profile-scorecard em {
        font-style: normal;
        color: var(--gold);
        font-size: 0.72rem;
        font-weight: 800;
      }

      .nhlrost-contract-hero {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-contract-hero span {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlrost-contract-hero strong {
        font-size: 1.8rem;
        font-weight: 900;
      }

      .nhlrost-storyline-card {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        padding: 12px 14px;
      }

      .nhlrost-storyline-card strong {
        display: block;
        margin-bottom: 4px;
      }

      .nhlrost-storyline-card span {
        color: var(--muted);
        font-size: 0.72rem;
      }

      .nhlrost-storyline-card p {
        margin: 6px 0 0;
        color: rgba(232, 244, 251, 0.86);
        font-size: 0.82rem;
      }

      .nhlrost-inspector {
        flex: 0 0 auto;
        max-height: min(38vh, 420px);
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        border-radius: 16px;
        border: 1px solid rgba(0, 216, 223, 0.18);
        background: rgba(8, 23, 36, 0.92);
        overflow: hidden;
      }

      .nhlrost-inspector.is-collapsed {
        max-height: 56px;
        grid-template-rows: auto;
      }

      .nhlrost-inspector__head {
        padding: 8px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-inspector__identity {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
      }

      .nhlrost-inspector__identity strong {
        display: block;
        font-size: 0.92rem;
      }

      .nhlrost-inspector__identity span {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.72rem;
      }

      .nhlrost-inspector__actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex: 0 0 auto;
      }

      .nhlrost-inspector-strip {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
      }

      .nhlrost-inspector-strip__meta {
        flex: 1 1 auto;
        min-width: 0;
      }

      .nhlrost-inspector-strip__meta strong {
        display: block;
        font-size: 0.86rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-inspector-strip__meta span {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.7rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-inspector-close {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        cursor: pointer;
        font-size: 1.2rem;
        line-height: 1;
      }

      .nhlrost-team-mark {
        width: clamp(64px, 5.4vw, 74px);
        height: clamp(64px, 5.4vw, 74px);
        border-radius: 14px;
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        background: transparent;
        border: 0;
        overflow: hidden;
      }

      .nhlrost-team-logo-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        padding: 4px;
      }

      .nhlrost-team-logo-fallback {
        font-weight: 900;
        letter-spacing: 0.08em;
        font-size: 1rem;
        color: var(--cyan);
        text-shadow: 0 0 12px rgba(43, 228, 255, 0.22);
      }

      .nhlrost-chip-button,
      .nhlrost-primary-button {
        min-height: 38px;
        border-radius: 999px;
        padding: 0 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.045);
        color: var(--text);
        font-size: 0.72rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          transform 150ms ease,
          background 150ms ease,
          border-color 150ms ease,
          box-shadow 150ms ease;
      }

      .nhlrost-chip-button:hover,
      .nhlrost-primary-button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.36);
        background: rgba(0, 216, 223, 0.08);
      }

      .nhlrost-chip-button.is-active {
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.12);
        box-shadow: 0 0 12px rgba(0, 216, 223, 0.08);
      }

      .nhlrost-primary-button {
        background:
          linear-gradient(135deg, rgba(0, 216, 223, 0.3), rgba(98, 183, 255, 0.16)),
          rgba(0, 216, 223, 0.08);
        border-color: rgba(0, 216, 223, 0.45);
      }

      .nhlrost-readonly-pill {
        min-height: 38px;
        border-radius: 999px;
        padding: 0 14px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .nhlrost-stat-strip {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 12px;
      }

      .nhlrost-stat-strip article {
        min-height: 82px;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.052), rgba(255, 255, 255, 0.02)),
          rgba(7, 22, 35, 0.86);
        padding: 12px 14px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 2px;
        min-width: 0;
      }

      .nhlrost-stat-strip article.is-danger {
        border-color: rgba(255, 100, 100, 0.38);
        background:
          linear-gradient(180deg, rgba(255, 100, 100, 0.11), rgba(255, 255, 255, 0.02)),
          rgba(7, 22, 35, 0.88);
      }

      .nhlrost-stat-strip span {
        color: var(--muted);
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 900;
      }

      .nhlrost-stat-strip strong {
        font-size: 1.32rem;
        line-height: 1;
        color: #fff;
      }

      .nhlrost-stat-strip em {
        color: var(--muted-2);
        font-style: normal;
        font-size: 0.72rem;
      }

      .nhlrost-content-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(230px, 0.72fr) minmax(0, 2.1fr) minmax(260px, 0.82fr);
        gap: 14px;
        overflow: hidden;
      }

      .nhlrost-left-rail,
      .nhlrost-right-rail,
      .nhlrost-center-panel {
        min-height: 0;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 14px;
        overflow: hidden;
      }

      .nhlrost-left-rail,
      .nhlrost-right-rail {
        overflow-y: auto;
        padding-right: 2px;
      }

      .nhlrost-left-rail::-webkit-scrollbar,
      .nhlrost-right-rail::-webkit-scrollbar,
      .nhlrost-list-shell__body::-webkit-scrollbar,
      .nhlrost-board-shell__body::-webkit-scrollbar,
      .nhlrost-bottom-panel__body::-webkit-scrollbar,
      .nhlrost-inspector__body::-webkit-scrollbar,
      .nhlrost-drawer__body::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .nhlrost-left-rail::-webkit-scrollbar-thumb,
      .nhlrost-right-rail::-webkit-scrollbar-thumb,
      .nhlrost-list-shell__body::-webkit-scrollbar-thumb,
      .nhlrost-board-shell__body::-webkit-scrollbar-thumb,
      .nhlrost-bottom-panel__body::-webkit-scrollbar-thumb,
      .nhlrost-inspector__body::-webkit-scrollbar-thumb,
      .nhlrost-drawer__body::-webkit-scrollbar-thumb {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
      }

      .nhlrost-org-card,
      .nhlrost-cap-card,
      .nhlrost-selected-card {
        padding: 14px;
      }

      .nhlrost-panel > header,
      .nhlrost-panel__head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
      }

      .nhlrost-panel > header p,
      .nhlrost-panel__head p {
        margin: 0 0 4px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.64rem;
        font-weight: 900;
      }

      .nhlrost-panel > header h3,
      .nhlrost-panel__head h3 {
        margin: 0;
        font-size: 1rem;
        line-height: 1.1;
      }

      .nhlrost-panel__head > span {
        border-radius: 999px;
        padding: 5px 9px;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.16);
        font-size: 0.68rem;
        font-weight: 900;
      }

      .nhlrost-org-metrics,
      .nhlrost-stat-grid,
      .nhlrost-selected-card__mini,
      .nhlrost-adjustment-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-info-pair {
        min-width: 0;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 9px 10px;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .nhlrost-info-pair span {
        color: var(--muted);
        font-size: 0.63rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 900;
      }

      .nhlrost-info-pair strong {
        font-size: 0.86rem;
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
        word-break: break-word;
      }

      .nhlrost-info-pair.is-good strong,
      .nhlrost-mini-badge.is-good,
      .nhlrost-warning-clean span {
        color: var(--green);
      }

      .nhlrost-info-pair.is-warn strong,
      .nhlrost-mini-badge.is-warn {
        color: var(--gold);
      }

      .nhlrost-info-pair.is-bad strong,
      .nhlrost-mini-badge.is-bad,
      .nhlrost-info-pair.is-medical strong,
      .nhlrost-mini-badge.is-medical {
        color: var(--red);
      }

      .nhlrost-info-pair.is-premium strong,
      .nhlrost-mini-badge.is-premium {
        color: var(--gold);
      }

      .nhlrost-cap-meter {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        margin-bottom: 12px;
      }

      .nhlrost-cap-meter > div {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 12px;
        display: flex;
        justify-content: space-between;
        gap: 10px;
      }

      .nhlrost-cap-meter span {
        color: var(--muted);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 900;
      }

      .nhlrost-cap-meter strong {
        font-size: 0.9rem;
      }

      .nhlrost-cap-meter .is-danger strong {
        color: var(--red);
      }

      .nhlrost-cap-meter .is-good strong {
        color: var(--green);
      }

      .nhlrost-progress {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-progress__top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        color: var(--muted);
        font-size: 0.72rem;
      }

      .nhlrost-progress__top strong {
        color: var(--text);
      }

      .nhlrost-progress__track {
        height: 8px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.11);
      }

      .nhlrost-progress__track span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--cyan), var(--blue));
      }

      .nhlrost-progress.is-good .nhlrost-progress__track span {
        background: linear-gradient(90deg, #39d98a, #8bf0b2);
      }

      .nhlrost-progress.is-warn .nhlrost-progress__track span {
        background: linear-gradient(90deg, #e8a536, #ffd166);
      }

      .nhlrost-progress.is-bad .nhlrost-progress__track span,
      .nhlrost-progress.is-medical .nhlrost-progress__track span {
        background: linear-gradient(90deg, #ff6464, #ff9f9f);
      }

      .nhlrost-warning-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlrost-warning-card,
      .nhlrost-warning-clean,
      .nhlrost-action-notice {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 12px;
      }

      .nhlrost-warning-card strong,
      .nhlrost-warning-clean strong,
      .nhlrost-action-notice strong {
        display: block;
        font-size: 0.86rem;
        margin-bottom: 3px;
      }

      .nhlrost-warning-card p,
      .nhlrost-warning-clean p,
      .nhlrost-action-notice p {
        margin: 0;
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
      }

      .nhlrost-warning-card.is-bad,
      .nhlrost-action-notice.is-bad {
        border-color: rgba(255, 100, 100, 0.32);
        background: rgba(255, 100, 100, 0.075);
      }

      .nhlrost-warning-card.is-warn,
      .nhlrost-action-notice.is-warn {
        border-color: rgba(232, 165, 54, 0.32);
        background: rgba(232, 165, 54, 0.075);
      }

      .nhlrost-warning-card.is-medical,
      .nhlrost-action-notice.is-medical {
        border-color: rgba(255, 100, 100, 0.28);
        background: rgba(255, 100, 100, 0.07);
      }

      .nhlrost-warning-clean {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 2px 10px;
        align-items: center;
      }

      .nhlrost-warning-clean span {
        grid-row: span 2;
        width: 30px;
        height: 30px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background: rgba(72, 216, 139, 0.1);
        border: 1px solid rgba(72, 216, 139, 0.22);
      }

      .nhlrost-filters {
        padding: 12px;
        flex: 0 0 auto;
      }

      .nhlrost-filter-row {
        display: grid;
        grid-template-columns: repeat(7, minmax(110px, 1fr));
        gap: 10px;
        align-items: end;
      }

      .nhlrost-filter-row + .nhlrost-filter-row {
        margin-top: 10px;
      }

      .nhlrost-control {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 5px;
      }

      .nhlrost-control--compact {
        gap: 0;
      }

      .nhlrost-control--compact input,
      .nhlrost-control--compact select {
        height: 40px;
        border-radius: 12px;
        font-size: 0.8rem;
      }

      .nhlrost-control span {
        color: var(--muted);
        font-size: 0.61rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 950;
      }

      .nhlrost-control input,
      .nhlrost-control select {
        width: 100%;
        height: 38px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.11);
        outline: none;
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.025)),
          rgba(3, 11, 19, 0.86);
        color: var(--text);
        padding: 0 11px;
        font-size: 0.78rem;
        font-weight: 700;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
      }

      .nhlrost-control input::placeholder {
        color: rgba(232, 244, 251, 0.36);
      }

      .nhlrost-control input:focus,
      .nhlrost-control select:focus {
        border-color: rgba(0, 216, 223, 0.48);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.08),
          0 0 0 3px rgba(0, 216, 223, 0.08);
      }

      .nhlrost-control select option {
        background: #071625;
        color: var(--text);
      }

      .nhlrost-list-shell {
        flex: 1;
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) auto;
        overflow: hidden;
      }

      .nhlrost-list-shell__head {
        padding: 14px 16px 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .nhlrost-list-shell__head p {
        margin: 0 0 4px;
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 950;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .nhlrost-list-shell__head h3 {
        margin: 0;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
      }

      .nhlrost-list-shell__head > span {
        flex: 0 0 auto;
        border-radius: 999px;
        padding: 6px 10px;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.15);
        font-size: 0.7rem;
        font-weight: 900;
      }

      .nhlrost-list-shell__body {
        min-height: 0;
        overflow: auto;
        padding: 10px;
      }

      .nhlrost-table,
      .nhlrost-draft-table {
        min-width: 900px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-table__head,
      .nhlrost-draft-table__head {
        position: sticky;
        top: 0;
        z-index: 2;
        min-height: 34px;
        display: grid;
        align-items: center;
        gap: 8px;
        padding: 0 10px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(10, 28, 42, 0.98), rgba(6, 18, 29, 0.98));
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.6rem;
        font-weight: 950;
        backdrop-filter: blur(12px);
      }

      .nhlrost-table__head {
        grid-template-columns:
          minmax(190px, 1.7fr)
          54px
          72px
          minmax(110px, 0.9fr)
          48px
          minmax(92px, 0.85fr)
          minmax(110px, 1fr)
          minmax(190px, 1.4fr)
          minmax(92px, 0.78fr);
      }

      .nhlrost-table__head.has-pool {
        grid-template-columns:
          minmax(190px, 1.6fr)
          54px
          72px
          minmax(110px, 0.9fr)
          48px
          minmax(92px, 0.8fr)
          minmax(110px, 0.9fr)
          minmax(180px, 1.25fr)
          minmax(92px, 0.75fr)
          minmax(70px, 0.55fr);
      }

      .nhlrost-table__body,
      .nhlrost-draft-table__body {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-row,
      .nhlrost-draft-row {
        width: 100%;
        min-height: 58px;
        display: grid;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.018)),
          rgba(5, 18, 29, 0.72);
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition:
          transform 140ms ease,
          background 140ms ease,
          border-color 140ms ease,
          box-shadow 140ms ease;
      }

      .nhlrost-row {
        grid-template-columns:
          minmax(190px, 1.7fr)
          54px
          72px
          minmax(110px, 0.9fr)
          48px
          minmax(92px, 0.85fr)
          minmax(110px, 1fr)
          minmax(190px, 1.4fr)
          minmax(92px, 0.78fr);
      }

      .nhlrost-table__head.has-pool + .nhlrost-table__body .nhlrost-row {
        grid-template-columns:
          minmax(190px, 1.6fr)
          54px
          72px
          minmax(110px, 0.9fr)
          48px
          minmax(92px, 0.8fr)
          minmax(110px, 0.9fr)
          minmax(180px, 1.25fr)
          minmax(92px, 0.75fr)
          minmax(70px, 0.55fr);
      }

      .nhlrost-row:hover,
      .nhlrost-draft-row:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.26);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.07), rgba(255, 255, 255, 0.022)),
          rgba(6, 22, 35, 0.86);
      }

      .nhlrost-row.is-selected,
      .nhlrost-draft-row.is-selected {
        border-color: rgba(0, 216, 223, 0.48);
        background:
          linear-gradient(90deg, rgba(0, 216, 223, 0.12), rgba(255, 255, 255, 0.03)),
          rgba(6, 22, 35, 0.92);
        box-shadow:
          inset 3px 0 0 rgba(0, 216, 223, 0.85),
          0 0 16px rgba(0, 216, 223, 0.08);
      }

      .nhlrost-row__name {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
      }

      .nhlrost-row__name span {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .nhlrost-row__name strong {
        font-size: 0.86rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-row__name em,
      .nhlrost-row__stats,
      .nhlrost-row__pool {
        color: var(--muted);
        font-size: 0.72rem;
        font-style: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-row__pos {
        width: 38px;
        min-height: 28px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 950;
        letter-spacing: 0.05em;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.045);
      }

      .nhlrost-row__pos.pos-forward {
        color: var(--cyan);
      }

      .nhlrost-row__pos.pos-defense {
        color: var(--green);
      }

      .nhlrost-row__pos.pos-goalie {
        color: var(--purple);
      }

      .nhlrost-row__age,
      .nhlrost-row__role {
        font-size: 0.78rem;
        font-weight: 800;
      }

      .nhlrost-headshot.player-headshot {
        flex: 0 0 auto;
        filter: drop-shadow(0 8px 14px rgba(0, 0, 0, 0.42));
      }

      .nhlrost-headshot.player-headshot.size-xs {
        --size: 36px;
      }

      .nhlrost-headshot.player-headshot.size-sm {
        --size: 46px;
      }

      .nhlrost-headshot.player-headshot.size-md {
        --size: 58px;
      }

      .nhlrost-headshot.player-headshot.size-lg {
        --size: 72px;
      }

      .nhlrost-headshot.player-headshot.size-xl {
        --size: 96px;
      }

      .nhlrost-headshot .ph-flag,
      .nhlrost-headshot .ph-number {
        display: none;
      }

      .nhlrost-ovr-stack {
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 1px;
      }

      .nhlrost-ovr-stack strong {
        font-size: 1.22rem;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlrost-ovr-stack span,
      .nhlrost-ovr-stack em {
        color: var(--muted);
        font-size: 0.58rem;
        font-style: normal;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .nhlrost-ovr-drop {
        color: #ff7b7b !important;
        font-weight: 950 !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
      }

      .nhlrost-potential-stack {
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 4px;
      }

      .nhlrost-potential-stack > span {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 800;
      }

      .nhlrost-mini-badge {
        max-width: 100%;
        min-height: 24px;
        border-radius: 999px;
        padding: 4px 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.045);
        color: var(--text);
        font-size: 0.64rem;
        line-height: 1;
        font-weight: 950;
        letter-spacing: 0.06em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-archetype-tag {
        --arch-color: #9aa7bd;
        max-width: 100%;
        min-height: 26px;
        border-radius: 999px;
        padding: 5px 9px;
        display: inline-flex;
        align-items: center;
        border: 1px solid color-mix(in srgb, var(--arch-color) 72%, transparent);
        background: color-mix(in srgb, var(--arch-color) 15%, transparent);
        color: var(--arch-color);
        font-style: normal;
        font-size: 0.64rem;
        font-weight: 950;
        letter-spacing: 0.06em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlrost-card-grid,
      .nhlrost-ratings-grid-view {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
        gap: 10px;
      }

      .nhlrost-player-card,
      .nhlrost-rating-card {
        min-width: 0;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.052), rgba(255, 255, 255, 0.018)),
          rgba(5, 18, 29, 0.72);
        padding: 13px;
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          box-shadow 150ms ease,
          background 150ms ease;
      }

      .nhlrost-player-card:hover,
      .nhlrost-rating-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 216, 223, 0.26);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.065), rgba(255, 255, 255, 0.02)),
          rgba(5, 18, 29, 0.86);
      }

      .nhlrost-player-card.is-selected,
      .nhlrost-rating-card.is-selected {
        border-color: rgba(232, 165, 54, 0.56);
        box-shadow: 0 0 24px rgba(232, 165, 54, 0.08);
      }

      .nhlrost-player-card__top,
      .nhlrost-rating-card header {
        display: flex;
        align-items: center;
        gap: 11px;
      }

      .nhlrost-player-card__identity,
      .nhlrost-rating-card header > div {
        min-width: 0;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }

      .nhlrost-player-card__identity strong,
      .nhlrost-rating-card header strong {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.95rem;
      }

      .nhlrost-player-card__identity span,
      .nhlrost-rating-card header span {
        color: var(--muted);
        font-size: 0.76rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-player-card__ovr {
        flex: 0 0 auto;
        text-align: right;
      }

      .nhlrost-player-card__ovr strong {
        display: block;
        font-size: 2rem;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlrost-player-card__ovr span {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 900;
      }

      .nhlrost-player-card__badges {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlrost-player-card__metrics {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-player-card__statline {
        margin-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 10px;
        color: var(--muted);
        font-size: 0.78rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-rating-card {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .nhlrost-rating-card__bars {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlrost-rating-card footer {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--muted);
        font-size: 0.7rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 10px;
      }

      .nhlrost-rating-card footer strong {
        color: var(--cyan);
      }

      .nhlrost-draft-table__head,
      .nhlrost-draft-row {
        grid-template-columns:
          48px
          58px
          minmax(180px, 1.6fr)
          50px
          48px
          minmax(90px, 0.8fr)
          76px
          82px
          minmax(130px, 1fr)
          minmax(110px, 0.9fr);
      }

      .nhlrost-draft-row > span {
        min-width: 0;
        font-size: 0.78rem;
      }

      .nhlrost-draft-row strong {
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-draft-row em {
        color: var(--muted);
        font-size: 0.7rem;
        font-style: normal;
      }

      .nhlrost-draft-trend {
        width: fit-content;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 0.62rem;
        font-weight: 950;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
      }

      .nhlrost-draft-trend.is-up,
      .nhlrost-draft-trend.is-new {
        color: var(--green);
        border-color: rgba(72, 216, 139, 0.24);
        background: rgba(72, 216, 139, 0.07);
      }

      .nhlrost-draft-trend.is-down {
        color: var(--red);
        border-color: rgba(255, 100, 100, 0.24);
        background: rgba(255, 100, 100, 0.07);
      }

      .nhlrost-lines {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .nhlrost-line-section {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.085);
        background: rgba(255, 255, 255, 0.025);
        padding: 12px;
      }

      .nhlrost-line-section header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }

      .nhlrost-line-section header span {
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.78rem;
      }

      .nhlrost-line-section header strong {
        color: var(--muted);
        font-size: 0.72rem;
      }

      .nhlrost-line-row {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        align-items: stretch;
        gap: 10px;
        margin-top: 8px;
      }

      .nhlrost-line-row > strong {
        border-radius: 14px;
        display: grid;
        place-items: center;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.07);
        border: 1px solid rgba(0, 216, 223, 0.16);
        font-size: 0.78rem;
      }

      .nhlrost-line-row > div {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-line-section--goalies .nhlrost-line-row > div {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .nhlrost-line-chip {
        min-height: 58px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        color: var(--text);
        padding: 8px 10px;
        text-align: left;
        display: flex;
        flex-direction: column;
        gap: 3px;
        cursor: pointer;
      }

      .nhlrost-line-chip:hover {
        border-color: rgba(0, 216, 223, 0.25);
        background: rgba(0, 216, 223, 0.06);
      }

      .nhlrost-line-chip.is-selected {
        border-color: rgba(232, 165, 54, 0.52);
        background: rgba(232, 165, 54, 0.1);
      }

      .nhlrost-line-chip.is-empty {
        opacity: 0.42;
        cursor: default;
      }

      .nhlrost-line-chip span {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 950;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .nhlrost-line-chip strong {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.86rem;
      }

      .nhlrost-line-chip em {
        color: var(--muted);
        font-style: normal;
        font-size: 0.7rem;
      }

      .nhlrost-extra-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 8px;
      }

      .nhlrost-pagination {
        min-height: 48px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 8px 10px;
      }

      .nhlrost-pagination > div {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: center;
      }

      .nhlrost-pagination button {
        min-height: 30px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        padding: 0 10px;
        font-size: 0.72rem;
        font-weight: 900;
        cursor: pointer;
      }

      .nhlrost-pagination button:hover:not(:disabled),
      .nhlrost-pagination button.is-active {
        border-color: rgba(0, 216, 223, 0.4);
        background: rgba(0, 216, 223, 0.09);
      }

      .nhlrost-pagination button:disabled {
        opacity: 0.35;
        cursor: not-allowed;
      }

      .nhlrost-selected-card header {
        display: flex;
        gap: 12px;
        align-items: center;
      }

      .nhlrost-selected-card header > div {
        min-width: 0;
      }

      .nhlrost-selected-card header p {
        margin: 0 0 4px;
      }

      .nhlrost-selected-card header h3 {
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-selected-card header span {
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-selected-card__big {
        margin: 14px 0 12px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .nhlrost-selected-card__big > div {
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 12px;
      }

      .nhlrost-selected-card__big span {
        display: block;
        color: var(--muted);
        font-size: 0.64rem;
        font-weight: 950;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }

      .nhlrost-selected-card__big strong {
        display: block;
        margin-top: 4px;
        font-size: 2rem;
        line-height: 1;
        font-weight: 1000;
      }

      .nhlrost-selected-card__mini {
        margin-top: 12px;
      }

      .nhlrost-bottom-panel,
      .nhlrost-inspector {
        min-height: 0;
      }

      .nhlrost-detail-tabs {
        min-height: 50px;
        padding: 10px 12px 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .nhlrost-detail-tabs button {
        min-height: 32px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        padding: 0 12px;
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        cursor: pointer;
      }

      .nhlrost-detail-tabs button:hover,
      .nhlrost-detail-tabs button.is-active {
        color: var(--text);
        border-color: rgba(0, 216, 223, 0.4);
        background: rgba(0, 216, 223, 0.08);
      }

      .nhlrost-bottom-panel__body,
      .nhlrost-inspector__body {
        min-height: 0;
        overflow: auto;
        padding: 10px 12px 12px;
      }

      .nhlrost-player-overview {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        align-content: start;
      }

      .nhlrost-profile-zone {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
        min-width: 0;
      }

      .nhlrost-profile-zone--performance {
        grid-column: 1 / -1;
      }

      .nhlrost-profile-zone__head p {
        margin: 0 0 2px;
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlrost-profile-zone__head h3 {
        margin: 0 0 10px;
        font-size: 1rem;
        line-height: 1.2;
      }

      .nhlrost-profile-kv-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-profile-summary-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 10px;
      }

      .nhlrost-profile-summary-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
      }

      .nhlrost-profile-summary-item span {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlrost-profile-summary-item strong {
        font-size: 0.82rem;
        color: var(--gold);
      }

      .nhlrost-profile-stat-band,
      .nhlrost-stat-grid--wide {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 8px;
      }

      .nhlrost-ratings-layout,
      .nhlrost-stats-layout,
      .nhlrost-contract-layout,
      .nhlrost-development-layout,
      .nhlrost-history-layout {
        display: flex;
        flex-direction: column;
        gap: 12px;
        min-width: 0;
      }

      .nhlrost-ratings-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 8px;
      }

      .nhlrost-ratings-summary-card {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
      }

      .nhlrost-ratings-summary-card span {
        color: var(--muted);
        font-size: 0.62rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .nhlrost-ratings-summary-card strong {
        font-size: 1.1rem;
      }

      .nhlrost-ratings-summary-card em {
        font-style: normal;
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.35;
      }

      .nhlrost-detail-grid--ratings {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .nhlrost-contract-panel,
      .nhlrost-stats-band {
        width: 100%;
      }

      .nhlrost-profile-card,
      .nhlrost-profile-readout,
      .nhlrost-identity-grid,
      .nhlrost-detail-grid > .nhlrost-panel {
        padding: 14px;
      }

      .nhlrost-profile-card__hero {
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .nhlrost-profile-card__hero p,
      .nhlrost-profile-readout header p {
        margin: 0 0 5px;
        color: var(--muted);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 950;
      }

      .nhlrost-profile-card__hero h2,
      .nhlrost-profile-readout header h3 {
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.05;
      }

      .nhlrost-profile-card__hero span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.78rem;
      }

      .nhlrost-profile-card__score-row {
        margin: 14px 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-profile-card__score-row > div {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-profile-card__score-row span {
        display: block;
        color: var(--muted);
        font-size: 0.58rem;
        font-weight: 950;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .nhlrost-profile-card__score-row strong {
        display: block;
        margin-top: 3px;
        font-size: 1.22rem;
        line-height: 1;
      }

      .nhlrost-profile-card__score-row em {
        display: block;
        margin-top: 3px;
        color: var(--muted);
        font-style: normal;
        font-size: 0.65rem;
      }

      .nhlrost-profile-readout p,
      .nhlrost-muted-text {
        margin: 0;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
      }

      .nhlrost-profile-readout__badges {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlrost-identity-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-detail-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .nhlrost-rating-group {
        min-height: 180px;
      }

      .nhlrost-rating-group header {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }

      .nhlrost-rating-group header p {
        margin: 0 0 4px;
        color: var(--muted);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 950;
      }

      .nhlrost-rating-group header h3 {
        margin: 0;
      }

      .nhlrost-rating-group header span {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 900;
      }

      .nhlrost-rating-row-list,
      .nhlrost-production-bars {
        display: flex;
        flex-direction: column;
        gap: 9px;
      }

      .nhlrost-engine-score {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-bottom: 12px;
      }

      .nhlrost-engine-score > div {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-engine-score span {
        display: block;
        color: var(--muted);
        font-size: 0.58rem;
        font-weight: 950;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .nhlrost-engine-score strong {
        display: block;
        margin-top: 4px;
        font-size: 1.05rem;
      }

      .nhlrost-storyline-list,
      .nhlrost-drawer-feed {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlrost-storyline-list article,
      .nhlrost-drawer-feed article {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 12px;
      }

      .nhlrost-storyline-list strong,
      .nhlrost-drawer-feed strong {
        display: block;
        font-size: 0.86rem;
      }

      .nhlrost-storyline-list p,
      .nhlrost-drawer-feed p {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
      }

      .nhlrost-empty-panel {
        min-height: 120px;
        border-radius: 14px;
        border: 1px dashed rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.02);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 28px 20px;
        text-align: center;
      }

      .nhlrost-empty-panel.is-compact {
        min-height: 0;
        padding: 16px 14px;
        border-style: solid;
      }

      .nhlrost-empty-panel__orb {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        color: var(--muted);
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.035);
        margin-bottom: 12px;
      }

      .nhlrost-empty-panel h3 {
        margin: 0;
        font-size: 1.05rem;
      }

      .nhlrost-empty-panel p {
        margin: 6px 0 0;
        max-width: 420px;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
      }

      .nhlrost-drawer-backdrop {
        position: fixed;
        inset: 0;
        z-index: 50;
        background: rgba(0, 0, 0, 0.46);
        backdrop-filter: blur(5px);
        display: flex;
        justify-content: flex-end;
      }

      .nhlrost-drawer {
        width: min(460px, calc(100vw - 32px));
        height: 100vh;
        border-left: 1px solid rgba(255, 255, 255, 0.12);
        background:
          radial-gradient(circle at 30% 0%, rgba(0, 216, 223, 0.13), transparent 30%),
          linear-gradient(180deg, rgba(8, 23, 36, 0.98), rgba(3, 9, 15, 0.98));
        box-shadow: -24px 0 60px rgba(0, 0, 0, 0.45);
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        color: var(--text);
      }

      .nhlrost-drawer__head {
        padding: 20px 20px 14px;
        display: flex;
        justify-content: space-between;
        gap: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .nhlrost-drawer__head p {
        margin: 0 0 5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.68rem;
        font-weight: 950;
      }

      .nhlrost-drawer__head h2 {
        margin: 0;
        font-size: 1.35rem;
      }

      .nhlrost-drawer__head span {
        display: block;
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.78rem;
      }

      .nhlrost-drawer__head button {
        width: 38px;
        height: 38px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.055);
        cursor: pointer;
        font-size: 1.3rem;
      }

      .nhlrost-drawer__tabs {
        padding: 12px 16px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      .nhlrost-drawer__tabs button {
        min-height: 34px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        cursor: pointer;
        font-size: 0.68rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .nhlrost-drawer__tabs button.is-active {
        color: var(--text);
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.09);
      }

      .nhlrost-drawer__body {
        min-height: 0;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 14px;
      }

      .nhlrost-drawer-section {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: rgba(255, 255, 255, 0.035);
        padding: 14px;
      }

      .nhlrost-drawer-section h3 {
        margin: 0 0 10px;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .nhlrost-drawer-metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-drawer-metric-grid article {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-drawer-metric-grid span {
        display: block;
        color: var(--muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 950;
      }

      .nhlrost-drawer-metric-grid strong {
        display: block;
        margin-top: 4px;
        font-size: 1.08rem;
      }

      .nhlrost-drawer-player {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(135deg, rgba(0, 216, 223, 0.08), rgba(255, 255, 255, 0.035)),
          rgba(255, 255, 255, 0.03);
        padding: 14px;
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .nhlrost-drawer-player p {
        margin: 0 0 4px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.64rem;
        font-weight: 950;
      }

      .nhlrost-drawer-player h3 {
        margin: 0;
      }

      .nhlrost-drawer-player span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.78rem;
      }

      .nhlrost-drawer-note {
        margin: 0;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
      }

      .nhlrost-drawer-actions {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .nhlrost-drawer-actions button,
      .nhlrost-drawer-actions span,
      .nhlrost-drawer-nav-grid button {
        min-height: 42px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        padding: 10px 12px;
      }

      .nhlrost-drawer-actions button {
        cursor: pointer;
        font-weight: 900;
      }

      .nhlrost-drawer-nav-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .nhlrost-drawer-nav-grid button {
        text-align: left;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .nhlrost-drawer-nav-grid button:hover {
        border-color: rgba(0, 216, 223, 0.32);
        background: rgba(0, 216, 223, 0.07);
      }

      .nhlrost-drawer-nav-grid strong {
        font-size: 0.86rem;
      }

      .nhlrost-drawer-nav-grid span {
        color: var(--muted);
        font-size: 0.72rem;
      }

      .nhlrost-drawer-feed article.is-good strong,
      .nhlrost-drawer-feed article .is-good {
        color: var(--green);
      }

      .nhlrost-drawer-feed article.is-warn strong {
        color: var(--gold);
      }

      .nhlrost-drawer-feed article.is-bad strong,
      .nhlrost-drawer-feed article.is-medical strong,
      .nhlrost-drawer-feed article.is-danger strong {
        color: var(--red);
      }

      @media (max-width: 1500px) {
        .nhlrost-command-bar {
          grid-template-columns: 88px minmax(0, 1fr);
        }

        .nhlrost-player-overview {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlrost-profile-zone--contract {
          grid-column: 1 / -1;
        }

        .nhlrost-command-bar__metrics {
          grid-template-columns: repeat(6, minmax(0, 1fr));
        }

        .nhlrost-hud-tile__body strong {
          font-size: 0.86rem;
        }

        .nhlrost-filters-primary {
          grid-template-columns: minmax(260px, 1.4fr) minmax(130px, 1fr) minmax(88px, 0.6fr) minmax(100px, 0.65fr);
        }

        .nhlrost-pool-segment button {
          min-height: 38px;
          font-size: 0.68rem;
          padding: 0 6px;
        }

        .nhlrost-board-row {
          grid-template-columns:
            132px
            minmax(120px, 1.2fr)
            40px
            36px
            72px
            minmax(92px, 0.8fr)
            minmax(120px, 1fr)
            80px;
          gap: 8px;
          padding: 8px 10px;
        }
      }

      @media (max-width: 1200px) {
        .nhlrost-root {
          grid-template-columns: 64px minmax(0, 1fr);
        }

        .nhlrost-command-bar__metrics {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .nhlrost-hud-tile {
          padding: 6px;
        }

        .nhlrost-filters-primary {
          grid-template-columns: 1fr 1fr;
        }

        .nhlrost-pool-segment {
          grid-column: 1 / -1;
        }

        .nhlrost-filters-advanced {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlrost-player-overview,
        .nhlrost-detail-grid {
          grid-template-columns: 1fr;
        }

        .nhlrost-profile-zone--performance {
          grid-column: auto;
        }

        .nhlrost-detail-grid--ratings {
          grid-template-columns: 1fr;
        }

        .nhlrost-profile-modal__panel {
          width: min(100%, calc(100vw - 32px));
          max-height: calc(100dvh - 32px);
        }

        .nhlrost-board-row {
          grid-template-columns: 112px minmax(0, 1.1fr) 36px 36px 70px minmax(140px, 1.2fr) 72px;
        }

        .nhlrost-player-icon-plate {
          max-width: 120px;
          height: 50px;
        }

        .nhlrost-player-icon-plate__number {
          width: 30px;
          min-width: 30px;
          height: 44px;
        }

        .nhlrost-player-icon-plate__portrait .nhlrost-headshot.player-headshot.size-lg {
          --size: 54px;
        }

        .nhlrost-board-row__pot {
          display: none;
        }
      }

      @media (max-width: 900px) {
        .nhlrost-root {
          grid-template-columns: 1fr;
        }

        .nhlrost-sidebar {
          display: none;
        }

        .nhlrost-main {
          padding: 10px;
        }

        .nhlrost-command-bar {
          grid-template-columns: 72px minmax(0, 1fr);
        }

        .nhlrost-team-mark {
          width: 64px;
          height: 64px;
        }

        .nhlrost-command-bar__metrics {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlrost-filters-primary {
          grid-template-columns: 1fr;
        }

        .nhlrost-pool-segment {
          grid-column: auto;
        }

        .nhlrost-filters-advanced {
          grid-template-columns: 1fr;
        }

        .nhlrost-board-row {
          grid-template-columns: 108px minmax(0, 1fr) 32px 32px 56px;
        }

        .nhlrost-player-icon-plate {
          max-width: 108px;
        }

        .nhlrost-player-icon-plate__number span {
          font-size: 0.88rem;
        }

        .nhlrost-player-icon-plate__portrait .nhlrost-headshot.player-headshot.size-lg {
          --size: 48px;
        }

        .nhlrost-board-row__age,
        .nhlrost-board-row__cap {
          display: none;
        }

        .nhlrost-inspector {
          max-height: min(44vh, 360px);
        }

        .nhlrost-profile-card__score-row,
        .nhlrost-engine-score {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nhlrost-line-row {
          grid-template-columns: 1fr;
        }

        .nhlrost-line-row > div,
        .nhlrost-line-section--goalies .nhlrost-line-row > div {
          grid-template-columns: 1fr;
        }
      }
    `}</style>
  );
}