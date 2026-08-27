import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { enrichRosterPlayer } from "../game/rosterColumns";
import { GameFooter } from "../components/game/GameFooter";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { formatProspectLeague, formatProspectTeam } from "../events/prospectDevelopment/prospectDevelopmentHelpers";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { nationalityCode, ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { nearestFlagApiSize } from "../utils/countryFlags";
import {
  getBaseOverall,
  getOverallDrop,
  getOverallTooltip,
  getUniversalOverall,
} from "../utils/playerOverall";
import { getRosterMoves, moveRosterPlayer, getStatsCentral } from "../services/franchiseService";

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
  { value: "overview", label: "Overview" },
  { value: "character", label: "Character & Life" },
  { value: "performance", label: "Performance" },
  { value: "development", label: "Development" },
  { value: "contract", label: "Contract" },
  { value: "media", label: "Media" },
  { value: "career", label: "Career" },
  { value: "moves", label: "Moves" },
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
  "Suspended",
  "Leave",
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
    sv_pct: pickFirstDefined(statsRow.save_pct, statsRow.sv_pct, statsRow.savePct, statsRow.svPct),
    save_pct: pickFirstDefined(statsRow.save_pct, statsRow.sv_pct, statsRow.savePct, statsRow.svPct),
    gaa: statsRow.gaa,
    sog: pickFirstDefined(statsRow.sog, statsRow.shots),
    shots: pickFirstDefined(statsRow.shots, statsRow.sog),
    pim: statsRow.pim,
    plus_minus: pickFirstDefined(statsRow.plus_minus, statsRow.pm, statsRow.plusMinus),
    plusMinus: pickFirstDefined(statsRow.plusMinus, statsRow.plus_minus, statsRow.pm),
    toi: pickFirstDefined(statsRow.toi, statsRow.average_toi),
    hits: pickFirstDefined(statsRow.hits, statsRow.hit),
    blocks: pickFirstDefined(statsRow.blocks, statsRow.blk),
    shutouts: pickFirstDefined(statsRow.shutouts, statsRow.so),
    war: pickFirstDefined(statsRow.war),
    cfPct: pickFirstDefined(statsRow.cfPct, statsRow.cf_pct),
    cf_pct: pickFirstDefined(statsRow.cf_pct, statsRow.cfPct),
    xgfPct: pickFirstDefined(statsRow.xgfPct, statsRow.xgf_pct),
    xgf_pct: pickFirstDefined(statsRow.xgf_pct, statsRow.xgfPct),
    league_rank: pickFirstDefined(statsRow.league_rank, statsRow.pts_rank, statsRow.points_rank),
    leagueRank: pickFirstDefined(statsRow.leagueRank, statsRow.league_rank, statsRow.pts_rank),
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
    raw?.drafted_by,
    raw?.draftedBy,
    raw?.draft_team_id,
    raw?.draftTeamId,
    raw?.developed_by_team_id,
    raw?.developedByTeamId,
    raw?.rights_team_id,
    raw?.rightsTeamId,
    raw?.nhl_rights_team_id,
    raw?.nhlRightsTeamId,
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
    const affiliateName = safeStr(
      pickFirstDefined(
        player?.affiliate_team_name,
        player?.season_stats?.team_name,
        player?.season_stats?.team,
        player?.team_name,
        player?.teamName
      ),
      ""
    );
    push(player, {
      league: "AHL",
      team_name: affiliateName || orgName,
      teamName: affiliateName || orgName,
      affiliate_team_name: affiliateName || undefined,
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

  // Drafted prospects the club holds rights to but who are not on a pro roster.
  (userOrganization?.prospects || EMPTY_ARRAY).forEach((player) => {
    const path = safeStr(player?.development_path || player?.post_draft_league, "");
    push(player, {
      league: path || "PROSPECT",
      team_name: player.team_name || orgName,
      pipeline_level: path || "Prospect",
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
    franchiseState?.wjc_tournament?.user_prospects,
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

  const aav = normalizeMoneyMillions(
    pickFirstDefined(contract.aav, contract.cap_hit, contract.capHit, player?.aav, capHit)
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

  const yearsRemaining = safeNum(
    pickFirstDefined(
      contract.years_remaining,
      contract.yearsRemaining,
      player?.years_remaining,
      player?.yearsRemaining
    ),
    term
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

  const expiryYear = safeNumOrNull(
    pickFirstDefined(contract.expiry_year, contract.expiryYear, player?.expiry_year)
  );

  const rawType = pickFirstDefined(
    contract.contract_type,
    contract.contractType,
    contract.type,
    player?.contract_type,
    player?.contractType
  );
  const hasCapOrTerm = capHit > 0 || term > 0;

  const signedStatusHint = safeStr(
    pickFirstDefined(contract.signed_status, player?.signed_status, player?.rights_status),
    ""
  ).toLowerCase();

  const rawTypeLower = safeStr(rawType, "").toLowerCase();
  const explicitlyUnsigned =
    rawTypeLower === "unsigned" || (!rawTypeLower && signedStatusHint.includes("unsign"));

  // Unsigned only when there's no cap/term evidence AND the type explicitly
  // (or via rights/signed status) says Unsigned — an empty/unknown type with
  // real cap or term data should never read as unsigned.
  const isSigned = hasCapOrTerm ? true : Boolean(rawTypeLower) && !explicitlyUnsigned;

  const type = safeStr(rawType, isSigned ? "Standard" : "Unsigned");

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

  const twoWay = Boolean(pickFirstDefined(contract.two_way, contract.twoWay, player?.two_way));
  const isEntryLevel = Boolean(
    pickFirstDefined(contract.is_entry_level, contract.isEntryLevel, player?.is_entry_level)
  );
  const signingBonusM = normalizeMoneyMillions(
    pickFirstDefined(contract.signing_bonus_m, contract.signingBonusM)
  );
  const performanceBonusM = normalizeMoneyMillions(
    pickFirstDefined(contract.performance_bonus_m, contract.performanceBonusM)
  );
  const minorSalaryM = normalizeMoneyMillions(
    pickFirstDefined(contract.minor_salary_m, contract.minorSalaryM)
  );
  const startYear = safeNumOrNull(pickFirstDefined(contract.start_year, contract.startYear));

  return {
    capHit,
    aav,
    salary,
    term,
    yearsRemaining,
    expiry,
    expiryYear,
    type,
    clause,
    isSigned,
    twoWay,
    isEntryLevel,
    signingBonusM,
    performanceBonusM,
    minorSalaryM,
    startYear,
  };
}

function normalizeSeasonStats(player) {
  const s = player?.season_stats || player?.seasonStats || player?.stats || EMPTY_OBJECT;

  const gp = safeNum(pickFirstDefined(s.gp, s.games_played, s.gamesPlayed, player?.gp), 0);
  const goals = safeNum(pickFirstDefined(s.g, s.goals, player?.goals, player?.g), 0);
  const assists = safeNum(pickFirstDefined(s.a, s.assists, player?.assists, player?.a), 0);
  const points = safeNum(pickFirstDefined(s.pts, s.points, player?.points, player?.pts, goals + assists), goals + assists);
  const shots = safeNum(pickFirstDefined(s.sog, s.shots, s.shots_on_goal, s.shotsOnGoal, player?.shots), 0);
  const hits = safeNum(pickFirstDefined(s.hits, s.hit, player?.hits, player?.hit), 0);
  const blocks = safeNum(pickFirstDefined(s.blocks, s.blk, s.blocked_shots, s.blockedShots, player?.blocks, player?.blk), 0);
  const takeaways = safeNum(pickFirstDefined(s.takeaways, s.tk, player?.takeaways), 0);
  const giveaways = safeNum(pickFirstDefined(s.giveaways, s.gv, player?.giveaways), 0);
  const pim = safeNum(pickFirstDefined(s.pim, s.penalty_minutes, s.penaltyMinutes, player?.pim), 0);
    const plusMinusRaw = pickFirstDefined(
      s.plus_minus,
      s.plusMinus,
      s.pm,
      player?.plus_minus,
      player?.pm,
      s.goal_differential_on_ice,
      player?.goal_differential_on_ice
    );
    const gfOn = safeNum(pickFirstDefined(s.gf_on, s.on_ice_gf, player?.gf_on), 0);
    const gaOn = safeNum(pickFirstDefined(s.ga_on, s.on_ice_ga, player?.ga_on), 0);
    const plusMinus =
      plusMinusRaw != null && plusMinusRaw !== ""
        ? safeNum(plusMinusRaw, 0)
        : Math.round(gfOn - gaOn);

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
    shutouts: safeNum(pickFirstDefined(s.shutouts, s.so, player?.shutouts), 0),
    war: safeNumOrNull(pickFirstDefined(s.war, player?.war)),
    cfPct: safeNumOrNull(pickFirstDefined(s.cfPct, s.cf_pct, player?.cfPct)),
    xgfPct: safeNumOrNull(pickFirstDefined(s.xgfPct, s.xgf_pct, player?.xgfPct)),
    leagueRank: safeNumOrNull(pickFirstDefined(s.league_rank, s.leagueRank, s.pts_rank, s.points_rank)),
    teamRank: safeNumOrNull(pickFirstDefined(s.team_rank, s.teamRank)),
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

  const conductGames = Math.max(
    0,
    safeNum(pickFirstDefined(player?.conduct_games_remaining, player?.conductGamesRemaining), 0)
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
  const availKey = normalizeKey(
    pickFirstDefined(player?.availability_status, player?.availability, player?.status)
  );

  const onLeave =
    player?.conduct_eligible_to_play === false ||
    player?.suspended === true ||
    availKey.includes("leave") ||
    availKey.includes("suspended") ||
    statusKey.includes("leave") ||
    statusKey.includes("suspended");
  const leaveLabel =
    availKey.includes("leave") || statusKey.includes("leave")
      ? "Leave"
      : "Suspended";

  const injuredByFlag =
    player?.is_injured === true ||
    player?.injured === true ||
    player?.isInjured === true ||
    (gamesRemaining > 0 && !onLeave);

  const injuredByText =
    Boolean(injuryKey) &&
    !["healthy", "none", "available", "active"].includes(injuryKey);

  const isInjured =
    !onLeave &&
    (injuredByFlag ||
      injuredByText ||
      statusKey.includes("injured") ||
      statusKey.includes("out") ||
      statusKey.includes("day_to_day") ||
      statusKey.includes("ltir"));

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

  if (onLeave) {
    const g = conductGames || gamesRemaining;
    label = g > 0 ? `${leaveLabel} · ${g}g` : leaveLabel;
  } else if (isLTIR) {
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
    isConductLeave: onLeave,
    conductLabel: onLeave ? leaveLabel : "",
    gamesRemaining: onLeave ? conductGames || gamesRemaining : gamesRemaining,
    label,
    rawStatus,
    injuryLabel,
  };
}

function normalizeRosterStatus(player, league) {
  const health = normalizeHealth(player);

  if (health.isConductLeave) return health.conductLabel || "Suspended";
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

const TRADE_STABILITY_ESCALATION = {
  0: { shortLabel: "Monitor", label: "Monitor — early concern", tone: "neutral" },
  1: { shortLabel: "Frustrated", label: "Growing frustration", tone: "warn" },
  2: { shortLabel: "Disconnect", label: "Disconnecting from organization", tone: "warn" },
  3: { shortLabel: "Demand", label: "Formal trade demand", tone: "bad" },
  4: { shortLabel: "Crisis", label: "Locker-room crisis", tone: "bad" },
};

function formatPressureLabel(pressure) {
  if (!pressure) return "";
  return String(pressure).replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function lookupTradeStabilityFlag(franchiseState, player) {
  const flags = franchiseState?.trade_stability_roster_flags;
  if (!flags || typeof flags !== "object") return null;
  const pid = String(player?.id || player?.player_id || player?.key || "");
  if (!pid) return null;
  return flags[pid] || null;
}

function getTradeStabilityConcernBand(flag) {
  if (!flag) return null;
  const level = Number(flag.escalation_level ?? flag.escalationLevel ?? 0);
  const preset = TRADE_STABILITY_ESCALATION[level] || TRADE_STABILITY_ESCALATION[0];
  const score = flag.score != null ? Math.round(Number(flag.score)) : null;
  const pressure = flag.top_pressure || flag.topPressure || "";
  const titleParts = [preset.label];
  if (score != null) titleParts.push(`Stability ${score}`);
  if (pressure) titleParts.push(formatPressureLabel(pressure));
  return {
    ...preset,
    escalationLevel: level,
    score,
    topPressure: pressure,
    title: titleParts.join(" · "),
  };
}

function resolvePlayerTradeStabilityConcern(player, franchiseState) {
  const flag = player?.tradeStabilityFlag || lookupTradeStabilityFlag(franchiseState, player);
  return getTradeStabilityConcernBand(flag);
}

function getHealthBand(player) {
  const health = normalizeHealth(player);

  if (health.isConductLeave) return { label: health.label, tone: "bad" };
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
  return `https://flagsapi.com/${iso2}/${style}/${nearestFlagApiSize(size)}.png`;
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

function lastCareerSeasonSummary(player) {
  const seasons = Array.isArray(player?.career_seasons) ? player.career_seasons : EMPTY_ARRAY;
  if (!seasons.length) return null;

  // Prefer the most recent completed season over an in-progress current one.
  const completed = seasons.filter((row) => !row?.is_current_season);
  const pool = completed.length ? completed : seasons;
  const last = pool[pool.length - 1];
  if (!last) return null;

  const seasonLabel = safeStr(pickFirstDefined(last?.season, last?.year), "");

  if (isGoaliePosition(player.position)) {
    const gp = safeNum(pickFirstDefined(last?.gp, last?.games_played), 0);
    if (!gp) return null;
    const wins = safeNum(pickFirstDefined(last?.wins, last?.w), 0);
    const losses = safeNum(pickFirstDefined(last?.losses, last?.l), 0);
    const otl = safeNum(last?.otl, 0);
    return seasonLabel ? `${seasonLabel}: ${wins}-${losses}-${otl}` : `${wins}-${losses}-${otl}`;
  }

  const gp = safeNum(pickFirstDefined(last?.gp, last?.games_played), 0);
  const goals = safeNum(pickFirstDefined(last?.g, last?.goals), 0);
  const assists = safeNum(pickFirstDefined(last?.a, last?.assists), 0);
  const pts = safeNum(pickFirstDefined(last?.pts, last?.points), 0);
  if (!gp && !goals && !assists && !pts) return null;

  return seasonLabel ? `${seasonLabel}: ${pts} pts` : `${goals}-${assists}-${pts} pts`;
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

    if (!gp && !wins && !losses && !otl && !sv && !gaa) return lastCareerSeasonSummary(player) || "—";

    return `${gp} GP · ${wins}-${losses}-${otl} · ${sv ? formatDecimal(sv, 3) : "—"} SV% · ${gaa ? gaa.toFixed(2) : "—"} GAA`;
  }

  const gp = safeNum(stats.gp, 0);
  const goals = safeNum(stats.g, 0);
  const assists = safeNum(stats.a, 0);
  const points = safeNum(stats.pts, 0);

  if (!gp && !goals && !assists && !points) return lastCareerSeasonSummary(player) || "—";

  return `${gp} GP · ${goals} G · ${assists} A · ${points} PTS`;
}

function capHitDisplay(player) {
  const contract = player?.contract || EMPTY_OBJECT;
  const capHit = safeNum(contract.capHit, 0);

  if (!contract.isSigned && capHit <= 0) return "Unsigned";
  if (capHit <= 0) return "—";

  return formatMoneyMillions(capHit);
}

function contractSummaryDisplay(player) {
  const contract = player?.contract || EMPTY_OBJECT;

  if (!contract.isSigned) return "Unsigned";
  if (contract.capHit > 0 && contract.term > 0) return `${formatMoneyMillions(contract.capHit)} · ${contract.term} yr`;
  if (contract.capHit > 0) return formatMoneyMillions(contract.capHit);
  if (contract.term > 0) return `${contract.term} yr`;

  return "—";
}

// The enriched roster row invents a deterministic jersey number for display
// slots when no real number exists — never surface that fake value as if it
// were the player's actual sweater number.
function resolveJerseyNumber(player) {
  const real = safeNumOrNull(pickFirstDefined(player?.jersey_number, player?.jerseyNumber));
  return real !== null ? real : null;
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

  const pos = normalizePosition(player?.position || player?.pos);
  const age = safeNum(player?.age, 18);
  const trueOverall = safeNum(overallModel?.trueOverall, 0);
  const growth = inferGrowth(player);
  const ppg = safeNum(seasonStats?.ppg, 0);
  const gp = safeNum(seasonStats?.gp, 0);

  // The backend may already carry a computed 0-100 potential score
  // (potential_score, dev_potential, or a numeric `potential`). That value
  // is authoritative — never blend it with the local estimation heuristics.
  const backendScore = safeNumOrNull(
    pickFirstDefined(player?.potential_score, player?.dev_potential, player?.potential)
  );

  if (backendScore !== null && backendScore > 0) {
    const finalScore = clamp(backendScore, 0, 100);
    const rawPotentialStr = typeof rawPotential === "string" ? rawPotential.trim() : "";
    const isLabelString = rawPotentialStr && Number.isNaN(Number(rawPotentialStr));
    const label = isLabelString ? rawPotentialStr : potentialLabelFromScore(pos, finalScore, trueOverall, age);

    return {
      rawPotential: rawPotential != null ? String(rawPotential) : "",
      potentialLabel: label,
      potentialScore: round0(finalScore),
      potentialConfidence: "Backend",
      potentialBreakdown: {
        base: finalScore,
        age: 0,
        production: 0,
        ratings: 0,
      },
    };
  }

  const explicitScore = typeof rawPotential === "string" ? getPotentialScoreFromRaw(rawPotential) : 0;

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
  if (player.tradeStabilityConcern) {
    pieces.push(`Trade stability: ${player.tradeStabilityConcern.title}.`);
  }

  return pieces.join(" ");
}

// Strictly evidence-driven — only concrete rating rows (top/bottom) and
// measured analytics facts, never generic scouting buzzwords.
function buildStrengthsConcerns(player) {
  if (!player) return { strengths: [], concerns: [] };

  const groups = Array.isArray(player.rating_groups) ? player.rating_groups : EMPTY_ARRAY;
  const rows = [];
  groups.forEach((group) => {
    (group?.rows || EMPTY_ARRAY).forEach((row) => {
      const value = safeNumOrNull(row?.value);
      if (value !== null && row?.label) rows.push({ label: row.label, value });
    });
  });

  const sortedDesc = [...rows].sort((a, b) => b.value - a.value);
  const sortedAsc = [...rows].sort((a, b) => a.value - b.value);

  const strengths = sortedDesc
    .slice(0, 3)
    .filter((row) => row.value >= 78)
    .map((row) => `${row.label} rated ${round0(row.value)}`);

  const concerns = sortedAsc
    .slice(0, 3)
    .filter((row) => row.value <= 68 && row.value > 0)
    .map((row) => `${row.label} rated ${round0(row.value)}`);

  const stats = player.season_stats || EMPTY_OBJECT;
  const gp = safeNum(stats.gp, 0);
  const isGoalie = isGoaliePosition(player.position);

  if (gp >= 15) {
    if (isGoalie) {
      if (stats.svPct >= 0.918) strengths.push(`${formatDecimal(stats.svPct, 3)} SV% over ${gp} GP`);
      if (stats.svPct > 0 && stats.svPct <= 0.897) concerns.push(`${formatDecimal(stats.svPct, 3)} SV% over ${gp} GP`);
    } else {
      if (stats.shots >= 40 && stats.shootingPct <= 0.06) {
        concerns.push(`${(stats.shootingPct * 100).toFixed(1)}% shooting on ${displayStatValue(stats.shots)} shots`);
      } else if (stats.shots >= 25 && stats.shootingPct >= 0.17) {
        strengths.push(`${(stats.shootingPct * 100).toFixed(1)}% shooting on ${displayStatValue(stats.shots)} shots`);
      }
      if (stats.war != null && Number(stats.war) >= 2) strengths.push(`${Number(stats.war).toFixed(1)} WAR over ${gp} GP`);
      if (stats.war != null && Number(stats.war) <= -1) concerns.push(`${Number(stats.war).toFixed(1)} WAR over ${gp} GP`);
    }
  }

  const contract = player.contract || EMPTY_OBJECT;
  if (!contract.isSigned) {
    concerns.push("Unsigned — no active contract on file");
  } else if (contract.term > 0 && contract.term <= 1) {
    concerns.push(`Contract expires ${contract.expiry || "this season"}`);
  }
  if (contract.capHit >= 8 && getUniversalOverall(player) < 84) {
    concerns.push(`${formatMoneyMillions(contract.capHit)} cap hit against a sub-84 overall`);
  } else if (contract.capHit > 0 && contract.capHit <= 2.5 && getUniversalOverall(player) >= 80) {
    strengths.push(`${formatMoneyMillions(contract.capHit)} cap hit for an 80+ overall`);
  }

  if (player.tradeStabilityConcern) {
    const concern = player.tradeStabilityConcern;
    concerns.unshift(
      `${concern.label}${concern.score != null ? ` — stability ${concern.score}` : ""}${
        concern.topPressure ? ` (${formatPressureLabel(concern.topPressure)})` : ""
      }`
    );
  }

  return {
    strengths: strengths.slice(0, 4),
    concerns: concerns.slice(0, 4),
  };
}

// Factual bullets only — no invented recommendations. Every line traces to
// a concrete field on the player object.
function buildDecisionBullets(player) {
  if (!player) return [];

  const bullets = [];
  const contract = player.contract || EMPTY_OBJECT;
  const health = normalizeHealth(player);

  if (!contract.isSigned) {
    bullets.push({ tone: "warn", text: "Unsigned — no active contract on file" });
  } else if (contract.term > 0 && contract.term <= 1) {
    bullets.push({ tone: "warn", text: `Contract expires ${contract.expiry || "this season"}` });
  }

  const waiverExempt = pickFirstDefined(player.waiver_exempt, player.waiverExempt);
  if (waiverExempt === false) {
    bullets.push({ tone: "neutral", text: "Not waiver exempt — needs waivers to reach the minors" });
  }

  const waiverStatus = safeStr(pickFirstDefined(player.waiver_status, player.waiverStatus), "");
  if (waiverStatus && waiverStatus !== "—") {
    bullets.push({ tone: "neutral", text: `Waiver status: ${waiverStatus}` });
  }

  const rightsExpiry = pickFirstDefined(player.rights_expiry_year, player.rightsExpiryYear);
  const rightsType = safeStr(pickFirstDefined(player.rights_type, player.rightsType), "");
  if (rightsExpiry) {
    bullets.push({
      tone: "neutral",
      text: `${rightsType ? `${rightsType} rights` : "Rights"} expire ${rightsExpiry}`,
    });
  }

  if (health.isConductLeave) {
    bullets.push({ tone: "bad", text: health.label });
  } else if (health.isInjured) {
    bullets.push({ tone: health.isDayToDay ? "warn" : "bad", text: `Injured — ${health.label}` });
  }

  const elcSlideEligible = pickFirstDefined(player.elc_slide_eligible, player.elcSlideEligible);
  const slideThreshold = pickFirstDefined(player.slide_games_threshold, player.slideGamesThreshold);
  if (elcSlideEligible && slideThreshold != null) {
    bullets.push({ tone: "neutral", text: `ELC slide eligible below ${slideThreshold} GP` });
  }

  if (player.tradeStabilityConcern) {
    const concern = player.tradeStabilityConcern;
    bullets.push({
      tone: concern.tone,
      text: `Trade concern — ${concern.label}${concern.score != null ? ` (${concern.score})` : ""}`,
    });
  }

  return bullets;
}

// A single factual status line — role, contract, and (only when real data
// exists) a development direction from growth or tracked history.
function buildCommandStatusLine(player) {
  if (!player) return "No player selected";

  const parts = [];
  const role = player.roleLabel || player.role;
  if (role && role !== "—") parts.push(role);

  const contract = player.contract || EMPTY_OBJECT;
  parts.push(contract.isSigned ? contractSummaryDisplay(player) : "Unsigned");

  const growth = safeNumOrNull(player.growth);
  const history = Array.isArray(player.development_history) ? player.development_history : EMPTY_ARRAY;

  if (growth !== null && growth !== 0) {
    parts.push(`Trending ${growth > 0 ? "up" : "down"} ${formatSignedNumber(growth)} OVR this season`);
  } else if (history.length >= 2) {
    const ovrs = history
      .map((entry) => safeNumOrNull(pickFirstDefined(entry?.ovr_after, entry?.ovr)))
      .filter((value) => value !== null);
    if (ovrs.length >= 2) {
      const delta = ovrs[ovrs.length - 1] - ovrs[0];
      if (delta !== 0) parts.push(`${delta > 0 ? "Up" : "Down"} ${Math.abs(delta)} OVR across tracked seasons`);
    }
  }

  return parts.length ? parts.join(" · ") : "No connected status data";
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

  const tradeStabilityFlag = lookupTradeStabilityFlag(franchiseState, output);
  const withConcern = {
    ...output,
    tradeStabilityFlag,
    tradeStabilityConcern: getTradeStabilityConcernBand(tradeStabilityFlag),
  };

  return {
    ...withConcern,
    note: buildPlayerNote(withConcern),
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

  const tradeConcernPlayers = (players || EMPTY_ARRAY).filter((player) => player.tradeStabilityConcern);
  if (tradeConcernPlayers.length) {
    const hasCritical = tradeConcernPlayers.some(
      (player) => Number(player.tradeStabilityConcern?.escalationLevel ?? 0) >= 3
    );
    warnings.push({
      key: "trade_stability",
      tone: hasCritical ? "bad" : "warn",
      title: "Trade satisfaction concerns",
      body: `${tradeConcernPlayers.length} player${tradeConcernPlayers.length === 1 ? "" : "s"} flagged by agent monitoring.`,
    });

    tradeConcernPlayers.forEach((player) => {
      const concern = player.tradeStabilityConcern;
      warnings.push({
        key: `trade_stability:${player.id || player.key}`,
        tone: concern.tone,
        title: `${player.name} — ${concern.label}`,
        body: concern.title,
        playerId: player.id,
      });
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

function TradeStabilityConcernBadge({ player, franchiseState, compact = false }) {
  const concern = player?.tradeStabilityConcern || resolvePlayerTradeStabilityConcern(player, franchiseState);
  if (!concern) return null;

  return (
    <MiniBadge
      text={compact ? concern.shortLabel : concern.label}
      tone={concern.tone}
      title={concern.title}
    />
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

/** Horizontal metric strip for dossier stats / contract (label over value, tiles in a row). */
function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className={`nhlrost-metric ${toneClass(tone)}`}>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function MetricStrip({ children, className = "" }) {
  return <div className={`nhlrost-metric-strip ${className}`.trim()}>{children}</div>;
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

function EmptyPanel({ title = "NO SIGNAL", body = "Board channel empty — adjust filters or reload roster feed.", compact = false }) {
  return (
    <section className={`nhlrost-empty-panel ${compact ? "is-compact" : ""}`}>
      {!compact ? <div className="nhlrost-empty-panel__orb" aria-hidden="true">—</div> : null}
      <p className="nhlrost-empty-panel__phase">{compact ? "OPS STATE" : "ROSTER OPS · STANDBY"}</p>
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

function PlayerStatusStrip({ player, franchiseState }) {
  const moraleBand = getMoraleBand(player?.morale);
  const fatigueBand = getFatigueBand(player?.fatigue);
  const healthBand = getHealthBand(player);

  return (
    <div className="nhlrost-status-strip">
      <MiniBadge text={`Morale ${round0(player?.morale)}`} tone={moraleBand.tone} title={moraleBand.label} />
      <MiniBadge text={`Fatigue ${round0(player?.fatigue)}`} tone={fatigueBand.tone} title={fatigueBand.label} />
      <MiniBadge text={healthBand.label} tone={healthBand.tone} />
      <TradeStabilityConcernBadge player={player} franchiseState={franchiseState} compact />
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
    const isOut =
      player.status === "Injured" ||
      player.status === "Suspended" ||
      player.status === "Leave" ||
      health.isInjured ||
      health.isConductLeave;

    if (isOut) {
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
  const healthBand = getHealthBand(player);
  const concernLevel = Number(player?.tradeStabilityConcern?.escalationLevel ?? 0);
  const concernClass =
    concernLevel >= 3 ? "has-trade-concern is-critical" : concernLevel > 0 ? "has-trade-concern" : "";

  return (
    <button
      type="button"
      className={`nhlrost-board-row ${selected ? "is-selected" : ""}${showTeam ? " has-team" : ""} ${concernClass}`.trim()}
      onClick={() => onSelect(player)}
    >
      <span className="nhlrost-board-row__name">
        <PlayerAvatar player={player} size="sm" />
        <span>
          <span className="nhlrost-board-row__name-line">
            <RosterPlayerFlag player={player} />
            <strong>{player.name}</strong>
            {player.tradeStabilityConcern ? (
              <TradeStabilityConcernBadge player={player} compact />
            ) : null}
          </span>
          {showTeam && player.teamName && player.teamName !== "—" ? (
            <em className="nhlrost-board-row__team">{player.teamName}</em>
          ) : null}
        </span>
      </span>

      <span className={`nhlrost-board-row__pos pos-${player.positionClass}`}>{player.position}</span>

      <span className="nhlrost-board-row__ovr">
        <OvrPill player={player} />
      </span>

      <span className="nhlrost-board-row__age">{player.age ? round0(player.age) : "—"}</span>

      <span className="nhlrost-board-row__contract">{capHitDisplay(player)}</span>

      <span className="nhlrost-board-row__status">
        <MiniBadge text={player.status} tone={healthBand.tone} />
      </span>

      <span className="nhlrost-board-row__role">{player.roleLabel || player.role || "—"}</span>

      <span className="nhlrost-board-row__stats">{compactBoardStats(player)}</span>

      <span className="nhlrost-board-row__avail">
        {player.availability?.label || player.availability_status || "Active"}
      </span>
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
    <div className="nhlrost-board nhlrost-board-sheet">
      <div className="nhlrost-board-sheet__head" aria-hidden="true">
        <span>Player</span>
        <span>Pos</span>
        <span>OVR</span>
        <span>Age</span>
        <span>Contract</span>
        <span>Status</span>
        <span>Role</span>
        <span>Season</span>
        <span>Avail</span>
      </div>
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
          const posGroup = String(player.positionClass || player.position || "").toUpperCase();
          const prev = players[index - 1];
          const prevGroup = String(prev?.positionClass || prev?.position || "").toUpperCase();
          const groupLabel = posGroup.startsWith("G")
            ? "Goalies"
            : posGroup === "D" || posGroup.startsWith("LD") || posGroup.startsWith("RD")
              ? "Defence"
              : "Forwards";
          const prevLabel = prev
            ? (prevGroup.startsWith("G")
              ? "Goalies"
              : prevGroup === "D" || prevGroup.startsWith("LD") || prevGroup.startsWith("RD")
                ? "Defence"
                : "Forwards")
            : null;
          const showGroup = groupLabel !== prevLabel;
          const globalIndex = pageOffset + index;
          const scratched = Boolean(player.scratched || player.is_scratch || player.line === "scratch");
          const archetypeColor = getArchetypeColor(player.archetype);
          const healthBand = getHealthBand(player);
          const concernLevel = Number(player?.tradeStabilityConcern?.escalationLevel ?? 0);
          const concernClass =
            concernLevel >= 3 ? "has-trade-concern is-critical" : concernLevel > 0 ? "has-trade-concern" : "";

          return (
            <React.Fragment key={player.key || `${player.name}-${globalIndex}`}>
              {showGroup ? <div className="nhlrost-group-head">{groupLabel}</div> : null}
            <button
              type="button"
              className={`nhlrost-row ${selected ? "is-selected" : ""} ${scratched ? "is-scratch" : ""} ${concernClass}`.trim()}
              onClick={() => onSelectPlayer(player)}
            >
              <span className="nhlrost-row__name">
                <PlayerAvatar player={player} size="sm" />
                <span>
                  <strong>{player.jersey_number || player.jerseyNumber || player.number ? `#${player.jersey_number || player.jerseyNumber || player.number} ` : ""}{player.name}</strong>
                  {(player.locker_room_cancer || player.brady_tkachuk_chaos || (player.name_tags || []).includes("CANCER")) ? (
                    <em className="nhlrost-cancer-tag" title="Locker-room cancer">CANCER</em>
                  ) : null}
                  {player.tradeStabilityConcern ? (
                    <TradeStabilityConcernBadge player={player} compact />
                  ) : null}
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
            </React.Fragment>
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
              <TradeStabilityConcernBadge player={player} compact />
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

function PlayerOverviewPanel({ player, franchiseState }) {
  const [expandedRatings, setExpandedRatings] = useState(false);

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
  const draftYear = player.draft_year || player.draftYear;
  const draftRound = player.draft_round || player.draftRound;
  const draftOverall = player.draft_overall_pick || player.draftOverallPick;
  const draftTeamId =
    player.drafted_by_team_id ||
    player.draft_team_id ||
    player.drafted_by_team ||
    player.draftedByTeamId ||
    "";
  const draftTeamName =
    player.drafted_by_team_name ||
    player.draftedByTeamName ||
    resolveDraftTeamName(draftTeamId, franchiseState);
  const hasDraftMeta = Boolean(draftYear || draftOverall || draftTeamId || player.drafted);
  const isUndrafted = Boolean(player.undrafted) && !hasDraftMeta;
  const ratingGroups = (player.rating_groups || EMPTY_ARRAY).filter((group) => group?.rows?.length);
  const { strengths, concerns } = buildStrengthsConcerns(player);
  const tradeConcern = player.tradeStabilityConcern || resolvePlayerTradeStabilityConcern(player, franchiseState);
  const measuredToi = safeNumOrNull(
    pickFirstDefined(player.explicitMinutes, stats.toi, stats.average_toi, stats.avg_toi)
  );

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

      {hasDraftMeta ? (
        <article className="nhlrost-profile-zone nhlrost-profile-zone--draft">
          <header className="nhlrost-profile-zone__head">
            <p>Draft History</p>
            <h3>
              {draftOverall != null && draftYear
                ? `#${draftOverall} overall · ${draftYear}`
                : draftYear
                  ? String(draftYear)
                  : "Drafted"}
            </h3>
          </header>
          <div className="nhlrost-profile-kv-grid">
            <InfoPair label="Draft Year" value={draftYear || "—"} />
            <InfoPair
              label="Pick"
              value={
                draftOverall != null
                  ? draftRound != null
                    ? `R${draftRound} · #${draftOverall}`
                    : `#${draftOverall}`
                  : "—"
              }
            />
            <InfoPair label="Drafted By" value={draftTeamName || (draftTeamId ? String(draftTeamId) : "—")} />
          </div>
        </article>
      ) : isUndrafted ? (
        <article className="nhlrost-profile-zone nhlrost-profile-zone--draft">
          <header className="nhlrost-profile-zone__head">
            <p>Draft History</p>
            <h3>Undrafted</h3>
          </header>
          <p className="nhlrost-muted-text">No NHL entry draft selection on file for this player.</p>
        </article>
      ) : null}

      <article className="nhlrost-profile-zone nhlrost-profile-zone--ability">
        <header className="nhlrost-profile-zone__head">
          <p>Ability & Role</p>
          <h3>{player.explicitRole || player.roleLabel || player.role || "—"}</h3>
        </header>
        <MetricStrip>
          <Metric label="OVR" value={<OvrPill player={player} large />} />
          <Metric label="POT" value={<PotentialPill player={player} large />} />
          {player.asset?.label ? <Metric label="Asset" value={player.asset.label} /> : null}
          <Metric label="Archetype" value={player.archetype || "—"} />
          <Metric label="TOI" value={measuredToi != null ? `${round1(measuredToi)}` : "—"} />
          <Metric label="Stage" value={player.stage || "—"} />
        </MetricStrip>
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--contract">
        <header className="nhlrost-profile-zone__head">
          <p>Contract Snapshot</p>
          <h3>{capHitDisplay(player)}</h3>
        </header>
        <MetricStrip>
          <Metric label="Status" value={formatContractStatus(contract)} />
          <Metric label="Term" value={contract.term ? `${contract.term} yr` : "—"} />
          <Metric label="Expiry" value={formatContractExpiry(contract)} />
          <Metric label="Type" value={contract.type || "—"} />
          <Metric label="Clause" value={contract.clause || "—"} />
          <Metric label="Morale" value={player.morale != null ? round0(player.morale) : "—"} />
        </MetricStrip>
      </article>

      {tradeConcern ? (
        <article className="nhlrost-profile-zone nhlrost-profile-zone--trade-stability">
          <header className="nhlrost-profile-zone__head">
            <p>Trade Stability</p>
            <h3>{tradeConcern.label}</h3>
          </header>
          <MetricStrip>
            <Metric
              label="Status"
              value={<TradeStabilityConcernBadge player={player} franchiseState={franchiseState} />}
            />
            <Metric label="Score" value={tradeConcern.score != null ? tradeConcern.score : "—"} tone={tradeConcern.tone} />
            <Metric
              label="Top pressure"
              value={tradeConcern.topPressure ? formatPressureLabel(tradeConcern.topPressure) : "—"}
            />
            <Metric label="Level" value={`L${tradeConcern.escalationLevel}`} tone={tradeConcern.tone} />
          </MetricStrip>
          <p className="nhlrost-muted-text">{tradeConcern.title}</p>
        </article>
      ) : null}

      <article className="nhlrost-profile-zone nhlrost-profile-zone--performance">
        <header className="nhlrost-profile-zone__head">
          <p>Universe Season Stats</p>
          <h3>
            {hasSeasonGames
              ? `${gp} GP · ${player.league && player.league !== "—" ? player.league : "NHL"}`
              : "No games played"}
          </h3>
        </header>
        {hasSeasonGames ? (
          <MetricStrip className="nhlrost-metric-strip--stats">
            {isGoaliePosition(player.position) ? (
              <>
                <Metric label="GP" value={displayStatValue(stats.gp)} />
                <Metric
                  label="Record"
                  value={`${displayStatValue(stats.wins)}-${displayStatValue(stats.losses)}-${displayStatValue(stats.otl)}`}
                />
                <Metric label="SV%" value={stats.svPct ? formatDecimal(stats.svPct, 3) : "—"} />
                <Metric label="GAA" value={stats.gaa ? Number(stats.gaa).toFixed(2) : "—"} />
                <Metric label="Saves" value={displayStatValue(stats.saves)} />
                <Metric label="SO" value={displayStatValue(stats.shutouts)} />
              </>
            ) : (
              <>
                <Metric label="GP" value={displayStatValue(stats.gp)} />
                <Metric label="G" value={displayStatValue(stats.g)} />
                <Metric label="A" value={displayStatValue(stats.a)} />
                <Metric label="PTS" value={displayStatValue(stats.pts)} />
                <Metric label="P/GP" value={stats.ppg ? Number(stats.ppg).toFixed(2) : displayStatValue(0)} />
                <Metric label="+/-" value={stats.plusMinus != null ? formatSignedNumber(stats.plusMinus, 0) : "—"} />
                <Metric label="SOG" value={displayStatValue(stats.shots || stats.sog)} />
                <Metric label="TOI" value={stats.toi ? `${round1(stats.toi)}` : "—"} />
                {stats.war != null ? <Metric label="WAR" value={Number(stats.war).toFixed(2)} /> : null}
                {stats.cfPct != null ? (
                  <Metric
                    label="CF%"
                    value={`${Number(stats.cfPct) <= 1.5 ? (Number(stats.cfPct) * 100).toFixed(1) : Number(stats.cfPct).toFixed(1)}%`}
                  />
                ) : null}
                {stats.leagueRank != null ? <Metric label="Lg Rk" value={`#${stats.leagueRank}`} /> : null}
              </>
            )}
          </MetricStrip>
        ) : (
          <p className="nhlrost-muted-text">
            No regular-season games played yet in this franchise universe. Lines fill in as the schedule simulates.
          </p>
        )}
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--ratings">
        <header className="nhlrost-profile-zone__head">
          <p>Attribute Profile</p>
          <h3>{ratingGroups.length ? `${ratingGroups.length} rating group${ratingGroups.length === 1 ? "" : "s"}` : "No ratings loaded"}</h3>
        </header>

        {ratingGroups.length ? (
          <>
            <div className="nhlrost-overview-ratings-bars">
              {ratingGroups.map((group) => {
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

            <button
              type="button"
              className="nhlrost-ratings-expand-toggle"
              onClick={() => setExpandedRatings((value) => !value)}
              aria-expanded={expandedRatings}
            >
              {expandedRatings ? "Hide full ratings ↑" : "View full ratings ↓"}
            </button>

            {expandedRatings ? (
              <div className="nhlrost-overview-ratings-expanded">
                <RatingsPanel player={player} />
              </div>
            ) : null}
          </>
        ) : (
          <p className="nhlrost-muted-text">Backend rating groups are not available for this player.</p>
        )}
      </article>

      <article className="nhlrost-profile-zone nhlrost-profile-zone--signal">
        <header className="nhlrost-profile-zone__head">
          <p>Strengths &amp; Concerns</p>
          <h3>Evidence Read</h3>
        </header>
        <div className="nhlrost-sc-columns">
          <div className="nhlrost-sc-column">
            <span className="nhlrost-sc-column__label is-good">Strengths</span>
            {strengths.length ? (
              <ul>
                {strengths.map((line, index) => (
                  <li key={index}>{line}</li>
                ))}
              </ul>
            ) : (
              <p className="nhlrost-muted-text">No standout attributes or analytics clear the threshold yet.</p>
            )}
          </div>
          <div className="nhlrost-sc-column">
            <span className="nhlrost-sc-column__label is-warn">Concerns</span>
            {concerns.length ? (
              <ul>
                {concerns.map((line, index) => (
                  <li key={index}>{line}</li>
                ))}
              </ul>
            ) : (
              <p className="nhlrost-muted-text">No flagged risks from current ratings, analytics, or contract data.</p>
            )}
          </div>
        </div>
      </article>
    </section>
  );
}

function resolveDraftTeamName(teamId, franchiseState) {
  if (!teamId) return "";
  const tid = String(teamId).toLowerCase();
  const pools = [
    franchiseState?.roster_browser?.organizations,
    franchiseState?.organizations,
    franchiseState?.league_teams,
    franchiseState?.teams,
  ];
  for (const pool of pools) {
    if (!Array.isArray(pool)) continue;
    for (const org of pool) {
      const candidates = [
        org?.team_id,
        org?.id,
        org?.abbr,
        org?.abbreviation,
        org?.short_name,
        org?.shortName,
      ]
        .map((value) => String(value || "").toLowerCase())
        .filter(Boolean);
      if (candidates.includes(tid)) {
        return org.name || org.full_name || org.fullName || org.abbr || org.abbreviation || String(teamId);
      }
    }
  }
  return String(teamId);
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

function buildSeasonLabelFromFranchiseState(franchiseState) {
  const y = safeNumOrNull(pickFirstDefined(franchiseState?.season_year, franchiseState?.seasonYear));
  if (!y) return "";
  return `${y}-${String(y + 1).slice(-2)}`;
}

function buildCurrentSeasonCareerRow(player, stats, { isGoalie, seasonLabel }) {
  const gp = safeNum(stats.gp, 0);
  if (gp <= 0) return null;

  const row = {
    season: seasonLabel || safeStr(pickFirstDefined(stats.season, stats.seasonLabel), "Current"),
    team: safeStr(
      pickFirstDefined(
        stats.team_name,
        stats.team,
        player?.affiliate_team_name,
        player?.teamName,
        player?.team_name
      ),
      "—"
    ),
    league: safeStr(pickFirstDefined(stats.league, stats.league_code, player?.league), "NHL"),
    gp,
    is_current_season: true,
  };

  if (isGoalie) {
    row.wins = pickFirstDefined(stats.wins, stats.w);
    row.losses = pickFirstDefined(stats.losses, stats.l);
    row.otl = stats.otl;
    row.sv_pct = pickFirstDefined(stats.svPct, stats.sv_pct);
    row.gaa = stats.gaa;
    row.shutouts = stats.shutouts;
  } else {
    row.g = stats.g;
    row.a = stats.a;
    row.pts = stats.pts;
    row.plus_minus = pickFirstDefined(stats.plusMinus, stats.plus_minus);
    row.pim = stats.pim;
    row.war = stats.war;
  }

  return row;
}

function mergeCareerSeasonsWithCurrent(player, stats, options) {
  const existing = Array.isArray(player?.career_seasons) ? player.career_seasons : EMPTY_ARRAY;
  const currentRow = buildCurrentSeasonCareerRow(player, stats, options);
  if (!currentRow) return existing;

  const dedupKey = `${currentRow.season}__${safeStr(currentRow.league, "NHL").toUpperCase()}__${currentRow.team}`;
  const filtered = existing.filter((row) => {
    const key = `${safeStr(pickFirstDefined(row?.season, row?.year), "")}__${safeStr(
      pickFirstDefined(row?.league, "NHL"),
      "NHL"
    ).toUpperCase()}__${safeStr(pickFirstDefined(row?.team, row?.team_name, row?.teamName), "")}`;
    return key !== dedupKey;
  });
  return [...filtered, currentRow];
}

function ProductionPanel({ player, franchiseState }) {
  if (!player) {
    return <EmptyPanel title="No player selected" body="Select a player to view stats." />;
  }

  const stats = player.season_stats || EMPTY_OBJECT;
  const isGoalie = isGoaliePosition(player.position);
  const gp = safeNum(stats.gp, 0);
  const hasSeasonGames = gp > 0;
  const seasonLabel = buildSeasonLabelFromFranchiseState(franchiseState);
  const careerSeasons = mergeCareerSeasonsWithCurrent(player, stats, { isGoalie, seasonLabel });
  const hasCareerSeasons = careerSeasons.length > 0;
  const leagueTag = safeStr(stats.league, "").toUpperCase();
  const isAhlLine = leagueTag === "AHL";

  if (!hasSeasonGames && !hasCareerSeasons) {
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

  return (
    <section className="nhlrost-stats-layout">
      {hasSeasonGames ? (
        isGoalie ? (
          <article className="nhlrost-panel nhlrost-stats-band">
            <header className="nhlrost-panel__head">
              <div>
                <p>{isAhlLine ? "AHL Season" : "Universe Season"}</p>
                <h3>{gp} GP</h3>
              </div>
            </header>
            <MetricStrip className="nhlrost-metric-strip--stats">
              <Metric label="GP" value={displayStatValue(stats.gp)} />
              <Metric label="Record" value={`${displayStatValue(stats.wins)}-${displayStatValue(stats.losses)}-${displayStatValue(stats.otl)}`} />
              <Metric label="SV%" value={stats.svPct ? formatDecimal(stats.svPct, 3) : "—"} />
              <Metric label="GAA" value={stats.gaa ? Number(stats.gaa).toFixed(2) : "—"} />
              <Metric label="Saves" value={displayStatValue(stats.saves)} />
              <Metric label="SA" value={displayStatValue(stats.shotsAgainst)} />
              <Metric label="SO" value={displayStatValue(stats.shutouts)} />
              <Metric label="TOI/GP" value={stats.toi ? `${round1(stats.toi)}` : "—"} />
            </MetricStrip>
          </article>
        ) : (
          <article className="nhlrost-panel nhlrost-stats-band">
            <header className="nhlrost-panel__head">
              <div>
                <p>{isAhlLine ? "AHL Season" : "Universe Season"}</p>
                <h3>
                  {gp} GP · {displayStatValue(stats.pts)} PTS
                  {stats.leagueRank != null ? ` · League #${stats.leagueRank}` : ""}
                </h3>
              </div>
            </header>
            <MetricStrip className="nhlrost-metric-strip--stats">
              <Metric label="GP" value={displayStatValue(stats.gp)} />
              <Metric label="G" value={displayStatValue(stats.g)} />
              <Metric label="A" value={displayStatValue(stats.a)} />
              <Metric label="PTS" value={displayStatValue(stats.pts)} />
              <Metric label="P/GP" value={stats.ppg ? Number(stats.ppg).toFixed(2) : displayStatValue(0)} />
              <Metric label="SOG" value={displayStatValue(stats.shots)} />
              <Metric label="SH%" value={stats.shootingPct ? `${(stats.shootingPct * 100).toFixed(1)}%` : "—"} />
              <Metric label="+/-" value={stats.plusMinus != null ? formatSignedNumber(stats.plusMinus, 0) : "—"} />
              <Metric label="TOI" value={stats.toi ? `${round1(stats.toi)}` : "—"} />
              <Metric label="Hits" value={displayStatValue(stats.hits)} />
              <Metric label="Blocks" value={displayStatValue(stats.blocks)} />
              <Metric label="PIM" value={displayStatValue(stats.pim)} />
              <Metric label="WAR" value={stats.war != null ? Number(stats.war).toFixed(2) : "—"} />
              {!isAhlLine ? (
                <>
                  <Metric
                    label="CF%"
                    value={
                      stats.cfPct != null
                        ? `${Number(stats.cfPct) <= 1.5 ? (Number(stats.cfPct) * 100).toFixed(1) : Number(stats.cfPct).toFixed(1)}%`
                        : "—"
                    }
                  />
                  <Metric
                    label="xGF%"
                    value={
                      stats.xgfPct != null
                        ? `${Number(stats.xgfPct) <= 1.5 ? (Number(stats.xgfPct) * 100).toFixed(1) : Number(stats.xgfPct).toFixed(1)}%`
                        : "—"
                    }
                  />
                </>
              ) : null}
            </MetricStrip>
          </article>
        )
      ) : null}

      {hasCareerSeasons ? (
        <CareerSeasonsTable seasons={careerSeasons} isGoalie={isGoalie} />
      ) : null}
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

  const rightsType = safeStr(pickFirstDefined(player.rights_type, player.rightsType), "");
  const rightsExpiryYear = pickFirstDefined(player.rights_expiry_year, player.rightsExpiryYear);
  const orgStatus = safeStr(pickFirstDefined(player.organizational_status, player.organizationalStatus), "");
  const signedStatus = safeStr(pickFirstDefined(player.signed_status, player.signedStatus), "");
  const rightsStatus = safeStr(pickFirstDefined(player.rights_status, player.rightsStatus), "");
  const elcEligible = pickFirstDefined(player.elc_eligible, player.elcEligible);
  const elcSlideEligible = pickFirstDefined(player.elc_slide_eligible, player.elcSlideEligible);
  const slideThreshold = pickFirstDefined(player.slide_games_threshold, player.slideGamesThreshold);
  const rosterLocation = safeStr(pickFirstDefined(player.roster_location, player.rosterLocation), "");
  const inMinors = Boolean(pickFirstDefined(player.in_minors, player.inMinors));
  const waiverStatus = safeStr(pickFirstDefined(player.waiver_status, player.waiverStatus), "");
  const waiverExempt = pickFirstDefined(player.waiver_exempt, player.waiverExempt);

  const contractRights = safeStr(contract.rightsStatus || contract.rights_status, "");
  const rightsHeadline = (() => {
    if (contract.isSigned && (rightsStatus || contractRights)) {
      const code = rightsStatus || contractRights;
      if (/^[ur]fa$/i.test(code)) return `Expires as ${code.toUpperCase()}`;
      return code;
    }
    return rightsStatus || rightsType || (contract.isSigned ? "Under contract" : "Unsigned");
  })();

  const hasRightsInfo = Boolean(
    rightsType ||
      rightsExpiryYear ||
      orgStatus ||
      signedStatus ||
      rightsStatus ||
      contractRights ||
      rosterLocation ||
      waiverStatus ||
      elcEligible != null ||
      elcSlideEligible != null ||
      waiverExempt != null
  );

  return (
    <section className="nhlrost-contract-layout">
      <article className="nhlrost-panel nhlrost-contract-panel">
        <div className="nhlrost-contract-hero">
          <span>Cap Hit</span>
          <strong className={toneClass(valueTone)}>{capHitDisplay(player)}</strong>
        </div>

        <MetricStrip className="nhlrost-metric-strip--stats">
          <Metric label="AAV" value={contract.aav ? formatMoneyMillions(contract.aav) : "—"} />
          <Metric label="Salary" value={formatMoneyMillions(contract.salary)} />
          <Metric label="Term" value={contract.term ? `${contract.term} yr` : "—"} />
          <Metric label="Yrs Left" value={contract.yearsRemaining ? `${contract.yearsRemaining}` : "—"} />
          <Metric label="Expiry" value={formatContractExpiry(contract)} />
          <Metric label="Type" value={contract.type || "—"} />
          <Metric label="Clause" value={contract.clause || "—"} />
          <Metric label="Status" value={formatContractStatus(contract)} />
          {contract.twoWay ? <Metric label="Two-Way" value="Yes" /> : null}
          {contract.isEntryLevel ? <Metric label="ELC" value="Yes" /> : null}
          {contract.signingBonusM ? <Metric label="Signing" value={formatMoneyMillions(contract.signingBonusM)} /> : null}
          {contract.performanceBonusM ? (
            <Metric label="Perf Bonus" value={formatMoneyMillions(contract.performanceBonusM)} />
          ) : null}
          {contract.minorSalaryM ? <Metric label="Minor $" value={formatMoneyMillions(contract.minorSalaryM)} /> : null}
          {contract.startYear ? <Metric label="Start" value={contract.startYear} /> : null}
        </MetricStrip>
        {contract.agent?.name ? (
          <MetricStrip className="nhlrost-metric-strip--stats nhlrost-metric-strip--agent">
            <Metric label="Agent" value={contract.agent.name} />
            <Metric label="Agency" value={contract.agent.agency || "—"} />
            <Metric label="Style" value={contract.agent.style_label || contract.agent.style || "—"} />
          </MetricStrip>
        ) : null}
      </article>

      {hasRightsInfo ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Rights &amp; Roster Status</p>
              <h3>{rightsHeadline}</h3>
            </div>
          </header>
          <MetricStrip className="nhlrost-metric-strip--stats">
            {rightsType ? <Metric label="Rights" value={rightsType} /> : null}
            {rightsExpiryYear ? <Metric label="Rights Exp" value={rightsExpiryYear} /> : null}
            {orgStatus ? <Metric label="Org Status" value={orgStatus} /> : null}
            {signedStatus ? <Metric label="Signed" value={signedStatus} /> : null}
            {(rightsStatus || contractRights) ? (
              <Metric label="Expiry" value={(rightsStatus || contractRights).toUpperCase()} />
            ) : null}
            {(rosterLocation || inMinors) ? (
              <Metric label="Location" value={rosterLocation || "Minors"} />
            ) : null}
            {waiverStatus ? <Metric label="Waivers" value={waiverStatus} /> : null}
            {waiverExempt != null ? (
              <Metric label="Exempt" value={waiverExempt ? "Yes" : "No"} tone={waiverExempt ? "good" : "warn"} />
            ) : null}
            {elcEligible != null ? <Metric label="ELC Elig" value={elcEligible ? "Yes" : "No"} /> : null}
            {elcSlideEligible != null ? (
              <Metric label="Slide Elig" value={elcSlideEligible ? "Yes" : "No"} />
            ) : null}
            {slideThreshold != null ? <Metric label="Slide GP" value={`<${slideThreshold}`} /> : null}
          </MetricStrip>
        </article>
      ) : null}
    </section>
  );
}

function DevelopmentTimelineChart({ points }) {
  const width = 560;
  const height = 168;
  const padding = 28;

  const ovrs = points.map((p) => p.ovr);
  const min = Math.min(...ovrs);
  const max = Math.max(...ovrs);
  const span = Math.max(1, max - min);
  const stepX = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 0;

  const coords = points.map((point, index) => {
    const x = padding + index * stepX;
    const y = height - padding - ((point.ovr - min) / span) * (height - padding * 2);
    return { ...point, x, y };
  });

  const path = coords.map((c, index) => `${index === 0 ? "M" : "L"} ${c.x.toFixed(1)} ${c.y.toFixed(1)}`).join(" ");

  return (
    <svg
      className="nhlrost-dev-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Overall rating development curve across tracked seasons"
    >
      <path d={path} className="nhlrost-dev-chart__line" fill="none" />
      {coords.map((c, index) => (
        <g key={index}>
          <circle cx={c.x} cy={c.y} r="3.5" className="nhlrost-dev-chart__dot" />
          <text x={c.x} y={c.y - 10} textAnchor="middle" className="nhlrost-dev-chart__value">
            {c.ovr}
          </text>
          <text x={c.x} y={height - 8} textAnchor="middle" className="nhlrost-dev-chart__label">
            {c.season}
          </text>
        </g>
      ))}
    </svg>
  );
}

function DevelopmentPanel({ player }) {
  if (!player) {
    return <EmptyPanel title="No development data" body="Select a player to view development." />;
  }

  const hasMorale = player.morale != null && Number.isFinite(Number(player.morale));
  const hasFatigue = player.fatigue != null && Number.isFinite(Number(player.fatigue));
  const hasGrowth = player.growth != null && Number.isFinite(Number(player.growth));
  const seasonStartOvr = safeNumOrNull(player.season_start_ovr);

  const rawHistory = Array.isArray(player.development_history) ? player.development_history : EMPTY_ARRAY;
  const validSnapshots = rawHistory
    .map((entry, index) => {
      const ovr = safeNumOrNull(pickFirstDefined(entry?.ovr_after, entry?.ovr, entry?.ovr_before));
      if (ovr === null) return null;
      return {
        season: safeStr(pickFirstDefined(entry?.season, entry?.year), `Snapshot ${index + 1}`),
        ovr,
        delta: safeNumOrNull(entry?.delta),
        sourcePath: entry?.source_path,
      };
    })
    .filter(Boolean);

  const hasCurve = validSnapshots.length >= 2;

  return (
    <section className="nhlrost-development-layout">
      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Development</p>
            <h3>Current Read</h3>
          </div>
        </header>
        <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
          <InfoPair label="Current OVR" value={displayOverallValue(player)} />
          {seasonStartOvr !== null ? <InfoPair label="Season Start OVR" value={seasonStartOvr} /> : null}
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

      <article className="nhlrost-panel nhlrost-dev-timeline-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Development History</p>
            <h3>{hasCurve ? `${validSnapshots.length} tracked seasons` : "Overall trend"}</h3>
          </div>
        </header>

        {hasCurve ? (
          <>
            <DevelopmentTimelineChart points={validSnapshots} />
            <div className="nhlrost-dev-timeline-list">
              {validSnapshots.map((snap, index) => (
                <div key={index} className="nhlrost-dev-timeline-row">
                  <span>{snap.season}</span>
                  <strong>{snap.ovr}</strong>
                  {snap.delta !== null ? (
                    <em className={snap.delta > 0 ? "is-up" : snap.delta < 0 ? "is-down" : "is-flat"}>
                      {formatSignedNumber(snap.delta, 0)}
                    </em>
                  ) : (
                    <em className="is-flat">—</em>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : (
          <EmptyPanel
            compact
            title="Not enough tracked seasons"
            body="A development curve appears once at least two overall snapshots are recorded for this player."
          />
        )}
      </article>
    </section>
  );
}

function UsagePanel({ player, onRefresh }) {
  const [moves, setMoves] = useState(EMPTY_ARRAY);
  const [meta, setMeta] = useState(EMPTY_OBJECT);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  const playerId = String(player?.id || player?.player_id || player?.key || "");

  useEffect(() => {
    let cancelled = false;
    setError("");
    setNote("");
    setMoves(EMPTY_ARRAY);
    if (!playerId) return undefined;
    getRosterMoves(playerId)
      .then((data) => {
        if (cancelled) return;
        setMeta(data || EMPTY_OBJECT);
        setMoves(Array.isArray(data?.actions) ? data.actions : EMPTY_ARRAY);
        if (!data?.ok && data?.reason) setError(String(data.reason));
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || "Could not load roster moves");
      });
    return () => {
      cancelled = true;
    };
  }, [playerId]);

  if (!player) {
    return <EmptyPanel title="No usage selected" body="Select a player to view role and deployment." />;
  }

  const runMove = async (action, extra = {}) => {
    setBusy(action);
    setError("");
    setNote("");
    try {
      const result = await moveRosterPlayer({
        player_id: playerId,
        action,
        ...extra,
      });
      if (!result?.ok) {
        if (result?.requires_waivers) {
          const ok = window.confirm(
            `${player.name || "Player"} requires waivers to be assigned to the AHL. Place on waivers and send down?`
          );
          if (ok) {
            const forced = await moveRosterPlayer({
              player_id: playerId,
              action,
              confirm_waivers: true,
            });
            if (!forced?.ok) {
              setError(forced?.reason || "Move failed");
              return;
            }
            setNote(forced.moved || "Move completed");
            setMoves(Array.isArray(forced.available_moves) ? forced.available_moves : EMPTY_ARRAY);
            if (typeof onRefresh === "function") onRefresh();
            return;
          }
          setError("Waivers required — move cancelled");
          return;
        }
        setError(result?.reason || "Move failed");
        return;
      }
      setNote(
        [
          result.moved,
          result.slide_preserved === true ? "ELC slide preserved" : null,
          result.slide_preserved === false ? "Slide threshold already passed" : null,
          result.slide_note,
        ]
          .filter(Boolean)
          .join(" · ")
      );
      setMoves(Array.isArray(result.available_moves) ? result.available_moves : EMPTY_ARRAY);
      if (typeof onRefresh === "function") onRefresh();
    } catch (err) {
      setError(err?.message || "Move failed");
    } finally {
      setBusy("");
    }
  };

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
          <InfoPair label="Location" value={meta.location || player.league || "—"} />
          <InfoPair label="NHL GP" value={meta.nhl_gp != null ? meta.nhl_gp : "—"} />
          <InfoPair
            label="Slide threshold"
            value={meta.slide_games_threshold != null ? `<${meta.slide_games_threshold} GP` : "—"}
          />
        </div>
      </article>

      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Roster Moves</p>
            <h3>Call-ups & Assignments</h3>
          </div>
          <span>{moves.length ? `${moves.length} available` : "None"}</span>
        </header>

        {error ? <p className="nhlrost-muted-text" style={{ color: "#f0a0a0" }}>{error}</p> : null}
        {note ? <p className="nhlrost-muted-text">{note}</p> : null}

        {moves.length ? (
          <div className="nhlrost-stat-grid" style={{ gap: "0.75rem" }}>
            {moves.map((action) => (
              <button
                key={action.id}
                type="button"
                className="nhlrost-action-btn"
                disabled={Boolean(busy) || action.enabled === false}
                title={action.reason || action.slide_note || ""}
                onClick={() => runMove(action.id)}
              >
                {busy === action.id ? "Working…" : action.label}
                {action.requires_waivers ? " (waivers)" : ""}
                {action.reason ? ` — ${action.reason}` : ""}
              </button>
            ))}
          </div>
        ) : (
          <p className="nhlrost-muted-text">
            No call-up or send-down actions for this player right now. Unsigned juniors need an ELC
            first; NHL veterans may require waivers to go to the AHL.
          </p>
        )}
      </article>
    </section>
  );
}

function CareerTotalsCard({ totals, isGoalie }) {
  if (!totals || typeof totals !== "object") return null;

  const gp = pickFirstDefined(totals.gp, totals.games_played);
  if (isGoalie) {
    const wins = pickFirstDefined(totals.wins, totals.w);
    const losses = pickFirstDefined(totals.losses, totals.l);
    const svPct = pickFirstDefined(totals.sv_pct, totals.svPct);
    const gaa = totals.gaa;
    if (gp == null && wins == null && losses == null && svPct == null && gaa == null) return null;

    return (
      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Career Totals</p>
            <h3>NHL</h3>
          </div>
        </header>
        <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
          <InfoPair label="GP" value={displayStatValue(gp)} />
          <InfoPair label="W" value={displayStatValue(wins)} />
          <InfoPair label="L" value={displayStatValue(losses)} />
          <InfoPair label="OTL" value={displayStatValue(pickFirstDefined(totals.otl))} />
          <InfoPair label="SV%" value={svPct != null ? formatDecimal(svPct, 3) : "—"} />
          <InfoPair label="GAA" value={gaa != null ? Number(gaa).toFixed(2) : "—"} />
        </div>
      </article>
    );
  }

  const goals = pickFirstDefined(totals.g, totals.goals);
  const assists = pickFirstDefined(totals.a, totals.assists);
  const pts = pickFirstDefined(totals.pts, totals.points);
  if (gp == null && goals == null && assists == null && pts == null) return null;

  return (
    <article className="nhlrost-panel">
      <header className="nhlrost-panel__head">
        <div>
          <p>Career Totals</p>
          <h3>NHL</h3>
        </div>
      </header>
      <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
        <InfoPair label="GP" value={displayStatValue(gp)} />
        <InfoPair label="G" value={displayStatValue(goals)} />
        <InfoPair label="A" value={displayStatValue(assists)} />
        <InfoPair label="PTS" value={displayStatValue(pts)} />
      </div>
    </article>
  );
}

function CareerSeasonsTable({ seasons, isGoalie }) {
  if (!Array.isArray(seasons) || !seasons.length) return null;

  return (
    <article className="nhlrost-panel nhlrost-career-seasons">
      <header className="nhlrost-panel__head">
        <div>
          <p>Career</p>
          <h3>Season by Season</h3>
        </div>
      </header>
      <div className="nhlrost-table-scroll">
        <table className="nhlrost-mini-table">
          <thead>
            <tr>
              <th scope="col">Season</th>
              <th scope="col">Team</th>
              <th scope="col">Lg</th>
              <th scope="col">GP</th>
              {isGoalie ? (
                <>
                  <th scope="col">W</th>
                  <th scope="col">L</th>
                  <th scope="col">OTL</th>
                  <th scope="col">SV%</th>
                  <th scope="col">GAA</th>
                </>
              ) : (
                <>
                  <th scope="col">G</th>
                  <th scope="col">A</th>
                  <th scope="col">PTS</th>
                  <th scope="col">+/-</th>
                  <th scope="col">PIM</th>
                  <th scope="col">WAR</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {seasons.map((row, index) => (
              <tr key={index} className={row?.is_current_season ? "is-current-season" : undefined}>
                <td>{safeStr(pickFirstDefined(row?.season, row?.year), "—")}</td>
                <td>{safeStr(pickFirstDefined(row?.team, row?.team_name, row?.teamName), "—")}</td>
                <td>{safeStr(pickFirstDefined(row?.league, row?.league_name), "—")}</td>
                <td>{displayStatValue(pickFirstDefined(row?.gp, row?.games_played))}</td>
                {isGoalie ? (
                  <>
                    <td>{displayStatValue(pickFirstDefined(row?.wins, row?.w))}</td>
                    <td>{displayStatValue(pickFirstDefined(row?.losses, row?.l))}</td>
                    <td>{displayStatValue(pickFirstDefined(row?.otl))}</td>
                    <td>
                      {pickFirstDefined(row?.sv_pct, row?.svPct) != null
                        ? formatDecimal(pickFirstDefined(row?.sv_pct, row?.svPct), 3)
                        : "—"}
                    </td>
                    <td>{row?.gaa != null ? Number(row.gaa).toFixed(2) : "—"}</td>
                  </>
                ) : (
                  <>
                    <td>{displayStatValue(pickFirstDefined(row?.g, row?.goals))}</td>
                    <td>{displayStatValue(pickFirstDefined(row?.a, row?.assists))}</td>
                    <td>{displayStatValue(pickFirstDefined(row?.pts, row?.points))}</td>
                    <td>
                      {pickFirstDefined(row?.plus_minus, row?.plusMinus) != null
                        ? formatSignedNumber(pickFirstDefined(row?.plus_minus, row?.plusMinus), 0)
                        : "—"}
                    </td>
                    <td>{displayStatValue(pickFirstDefined(row?.pim))}</td>
                    <td>{row?.war != null ? Number(row.war).toFixed(2) : "—"}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function CareerAwardsCard({ awards }) {
  if (!Array.isArray(awards) || !awards.length) return null;

  return (
    <article className="nhlrost-panel">
      <header className="nhlrost-panel__head">
        <div>
          <p>Career</p>
          <h3>Awards</h3>
        </div>
      </header>
      <ul className="nhlrost-award-list">
        {awards.map((award, index) => {
          if (typeof award === "string" || typeof award === "number") {
            return <li key={index}>{award}</li>;
          }
          const name = safeStr(pickFirstDefined(award?.name, award?.award, award?.title), "Award");
          const year = pickFirstDefined(award?.season, award?.year);
          return (
            <li key={index}>
              {name}
              {year ? ` — ${year}` : ""}
            </li>
          );
        })}
      </ul>
    </article>
  );
}

function CareerTransactionsCard({ transactions }) {
  if (!Array.isArray(transactions) || !transactions.length) return null;

  return (
    <article className="nhlrost-panel">
      <header className="nhlrost-panel__head">
        <div>
          <p>Career</p>
          <h3>Transactions</h3>
        </div>
      </header>
      <div className="nhlrost-storyline-list">
        {transactions.map((tx, index) => (
          <article key={index} className="nhlrost-storyline-card">
            <strong>{safeStr(pickFirstDefined(tx?.type, tx?.headline, tx?.title), "Transaction")}</strong>
            {tx?.date || tx?.season ? <span>{tx.date || tx.season}</span> : null}
            {tx?.description || tx?.summary ? <p>{tx.description || tx.summary}</p> : null}
          </article>
        ))}
      </div>
    </article>
  );
}

function rosterMediaHeatPhrase(heat) {
  const n = Number(heat);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 20) return "Quiet";
  if (n < 45) return "Building";
  if (n < 75) return "Hot";
  return "Boiling";
}

function rosterMediaCredPhrase(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 30) return "Speculation";
  if (n < 50) return "Early chatter";
  if (n < 75) return "Credible";
  return "Strongly sourced";
}

function groupPlayerStoryArcs(events) {
  const byArc = new Map();
  (Array.isArray(events) ? events : []).forEach((ev, idx) => {
    const arcId = String(ev?.storyline_id || ev?.storylineId || ev?.id || `beat-${idx}`);
    if (!byArc.has(arcId)) byArc.set(arcId, []);
    byArc.get(arcId).push(ev);
  });
  return [...byArc.entries()]
    .map(([arcId, beats]) => {
      const sorted = [...beats].sort((a, b) =>
        String(a?.calendar_iso || a?.date || a?.season || "").localeCompare(
          String(b?.calendar_iso || b?.date || b?.season || "")
        )
      );
      const latest = sorted[sorted.length - 1] || {};
      return {
        arcId,
        beats: sorted,
        headline: latest.headline || latest.title || "Career storyline",
        heat: Math.max(...sorted.map((b) => Number(b?.heat) || 0)),
        stage: String(latest?.arc_status || latest?.status || "active").toLowerCase() === "resolved"
          ? "Archived"
          : sorted.length > 2
            ? "Escalating"
            : "Developing",
      };
    })
    .sort((a, b) => b.heat - a.heat || b.beats.length - a.beats.length);
}

function buildPlayerPublicImage(player, storylines) {
  const labels = [];
  const events = Array.isArray(storylines) ? storylines : EMPTY_ARRAY;
  const hasTrade = events.some((e) => /trade|rumor|market/i.test(`${e?.type || ""} ${e?.category || ""} ${e?.headline || ""}`));
  const hasInjury = events.some((e) => /injur/i.test(`${e?.type || ""} ${e?.category || ""}`));
  const hasConduct = events.some((e) => /legal|conduct/i.test(`${e?.type || ""} ${e?.category || ""}`));
  const hasPerformance = events.some((e) => /performance|underperform|breakout|streak/i.test(`${e?.type || ""} ${e?.category || ""}`));

  if (Number(player?.ovr) >= 88) labels.push("Franchise centerpiece");
  if (player?.captain || player?.isCaptain) labels.push("Captain");
  if (hasTrade) labels.push("Trade speculation magnet");
  if (hasInjury) labels.push("Injury narrative");
  if (hasConduct) labels.push("Off-ice scrutiny");
  if (hasPerformance) labels.push("Performance storyline");
  if (Number(player?.morale) <= 45) labels.push("Frustrated");
  else if (Number(player?.morale) >= 80) labels.push("Locker-room favourite");

  if (!labels.length) labels.push("Professional");
  return labels.slice(0, 5);
}

function MediaPanel({ player, storylines, franchiseState }) {
  if (!player) {
    return <EmptyPanel title="No player selected" body="Select a player to view media coverage." compact />;
  }

  const playerId = String(player.id || player.player_id || "");
  const narrativeUniverse = franchiseState?.narrative_universe || {};
  const playerMem = narrativeUniverse?.player_narrative_memory?.[playerId] || null;
  const events = Array.isArray(storylines) ? storylines : EMPTY_ARRAY;
  const backendArcs = Array.isArray(narrativeUniverse?.story_arcs)
    ? narrativeUniverse.story_arcs.filter((a) => String(a?.player_id || "") === playerId)
    : [];
  const arcs = backendArcs.length
    ? backendArcs.map((arc) => ({
        arcId: arc.arc_id,
        beats: (Array.isArray(arc.beats) ? arc.beats : []).map((b, index) => ({
          id: b.beat_id,
          storyline_id: b.beat_id,
          headline: b.headline,
          summary: b.summary,
          calendar_iso: b.calendar_iso,
          date: b.calendar_iso,
          credibility: b.credibility,
        })),
        headline: arc.headline || "Career storyline",
        heat: Number(arc.heat) || 0,
        stage: arc.phase || arc.status || "Developing",
      }))
    : groupPlayerStoryArcs(events);
  const memTags = Array.isArray(playerMem?.reputation_tags) ? playerMem.reputation_tags : [];
  const publicImage = memTags.length ? memTags : buildPlayerPublicImage(player, events);
  const maxHeat = Math.max(
    events.reduce((m, s) => Math.max(m, Number(s?.heat) || 0), 0),
    Number(playerMem?.media_heat) || 0
  );
  const mediaPressure = rosterMediaHeatPhrase(maxHeat);
  const backendSocial = Array.isArray(narrativeUniverse?.social_posts)
    ? narrativeUniverse.social_posts.filter((p) => {
        const blob = `${p?.related_headline || ""} ${p?.text || ""}`.toLowerCase();
        const name = safeStr(player.name, "").toLowerCase();
        return name && blob.includes(name.split(/\s+/).pop());
      }).slice(0, 6)
    : [];
  const backendReddit = Array.isArray(narrativeUniverse?.reddit_threads)
    ? narrativeUniverse.reddit_threads.filter((t) => {
        const pid = String(t?.player_id || "");
        if (pid && pid === playerId) return true;
        const blob = `${t?.title || ""} ${t?.body || ""}`.toLowerCase();
        const name = safeStr(player.name, "").toLowerCase();
        return name && blob.includes(name.split(/\s+/).pop());
      }).slice(0, 4)
    : [];
  const agentRel = narrativeUniverse?.agent_relationships?.[playerId] || null;
  const agents = Array.isArray(narrativeUniverse?.agents) ? narrativeUniverse.agents : [];
  const agentInfo = agentRel
    ? agents.find((a) => String(a?.id || "") === String(agentRel.agent_id || "")) || null
    : null;
  const privateKnowledge = Array.isArray(narrativeUniverse?.knowledge_graph)
    ? narrativeUniverse.knowledge_graph.filter(
        (node) =>
          String(node?.player_id || "") === playerId ||
          events.some((ev) => String(ev?.storyline_id || ev?.id || "") === String(node?.storyline_id || ""))
      ).slice(-4)
    : [];
  const gmKnowsMore = events.some((ev) => ev?.gm_knows_more) || privateKnowledge.some((n) => n.gm_knows_more);
  const quotes = events
    .slice(0, 5)
    .map((ev) => ({
      source: ev?.source_label || ev?.source || "League Wire",
      text: ev?.summary || ev?.short_summary || ev?.effect_summary || ev?.headline || ev?.title || "",
    }))
    .filter((q) => q.text);

  return (
    <section className="nhlrost-history-layout nhlrost-media-layout">
      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Media universe</p>
            <h3>Public image</h3>
          </div>
          {mediaPressure ? <span className="nhlrost-media-pressure">{mediaPressure}</span> : null}
        </header>
        <div className="nhlrost-media-tags">
          {publicImage.map((tag) => (
            <span key={tag} className="nhlrost-media-tag">{tag}</span>
          ))}
        </div>
      </article>

      {agentInfo ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Representation</p>
              <h3>Player agent</h3>
            </div>
          </header>
          <p className="nhlrost-muted-text">
            <strong>{agentInfo.name}</strong> · {agentInfo.agency}
            {agentRel?.trust != null ? ` · client trust ${Math.round(Number(agentRel.trust) * 100)}%` : ""}
            {agentRel?.gm_trust != null ? ` · GM trust ${Math.round(Number(agentRel.gm_trust) * 100)}%` : ""}
          </p>
          <p className="nhlrost-muted-text">
            Style: {String(agentInfo.style || "").replace(/_/g, " ")}
            {agentInfo.leak_tendency != null
              ? ` · leak tendency ${Math.round(Number(agentInfo.leak_tendency) * 100)}%`
              : ""}
          </p>
        </article>
      ) : null}

      {gmKnowsMore || privateKnowledge.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Knowledge layers</p>
              <h3>Public vs private</h3>
            </div>
          </header>
          {gmKnowsMore ? (
            <p className="nhlrost-muted-text">
              You know more than the public wire — internal facts may not match headlines.
            </p>
          ) : null}
          {privateKnowledge.map((node, index) => (
            <p key={node.storyline_id || index} className="nhlrost-muted-text">
              {node.public_headline || node.headline || "Storyline"} · public level:{" "}
              {String(node.public_level || "unknown").replace(/_/g, " ")}
              {node.gm_knows_more ? " · GM knows more" : ""}
            </p>
          ))}
        </article>
      ) : null}

      {arcs.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Story arcs</p>
              <h3>Career narrative</h3>
            </div>
          </header>
          <div className="nhlrost-media-arcs">
            {arcs.map((arc) => (
              <div key={arc.arcId} className="nhlrost-media-arc">
                <div className="nhlrost-media-arc__head">
                  <strong>{arc.headline}</strong>
                  <span>{arc.stage} · {arc.beats.length} beat{arc.beats.length === 1 ? "" : "s"}</span>
                </div>
                <ol className="nhlrost-media-timeline">
                  {arc.beats.map((beat, index) => (
                    <li key={beat.id || beat.storyline_id || index}>
                      <time>{beat.calendar_iso || beat.date || beat.season || "—"}</time>
                      <span>{beat.headline || beat.title || beat.type || "Update"}</span>
                      {beat.summary || beat.short_summary ? <p>{beat.summary || beat.short_summary}</p> : null}
                      {rosterMediaCredPhrase(beat.credibility) ? (
                        <em>{rosterMediaCredPhrase(beat.credibility)}</em>
                      ) : null}
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </article>
      ) : (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Story arcs</p>
              <h3>Career narrative</h3>
            </div>
          </header>
          <p className="nhlrost-muted-text">No recorded storyline events yet. Coverage appears as the league reacts to performance, trades, injuries, and off-ice incidents.</p>
        </article>
      )}

      {backendReddit.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>IceHole</p>
              <h3>Forum threads</h3>
            </div>
          </header>
          <div className="nhlrost-media-quotes">
            {backendReddit.map((thread, index) => (
              <blockquote key={thread.thread_id || index}>
                <strong>{thread.subreddit} · {thread.flair} · {Number(thread.upvotes || 0).toLocaleString()}↑</strong>
                <p>{thread.title}</p>
                {thread.body ? <p className="nhlrost-muted-text">{thread.body}</p> : null}
              </blockquote>
            ))}
          </div>
        </article>
      ) : null}

      {backendSocial.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Social universe</p>
              <h3>Recent posts</h3>
            </div>
          </header>
          <div className="nhlrost-media-quotes">
            {backendSocial.map((post, index) => (
              <blockquote key={post.id || index}>
                <strong>{post.author_name}{post.verified ? " ✓" : ""}{post.author_type === "agent" ? " · Agent" : ""} · {post.handle || ""}</strong>
                <p>{post.text}</p>
              </blockquote>
            ))}
          </div>
        </article>
      ) : null}

      {quotes.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>What people are saying</p>
              <h3>Media conversation</h3>
            </div>
          </header>
          <div className="nhlrost-media-quotes">
            {quotes.map((q, index) => (
              <blockquote key={`${q.source}-${index}`}>
                <strong>{q.source}</strong>
                <p>{q.text}</p>
              </blockquote>
            ))}
          </div>
        </article>
      ) : null}

      {events.length ? (
        <article className="nhlrost-panel">
          <header className="nhlrost-panel__head">
            <div>
              <p>Wire file</p>
              <h3>Recent headlines</h3>
            </div>
          </header>
          <div className="nhlrost-storyline-list">
            {events.map((event, index) => (
              <article key={event.id || event.storyline_id || index} className="nhlrost-storyline-card">
                <strong>{event.headline || event.title || event.type || "Storyline"}</strong>
                {event.calendar_iso || event.date || event.season ? (
                  <span>{event.calendar_iso || event.date || event.season}</span>
                ) : null}
                {event.summary || event.short_summary ? <p>{event.summary || event.short_summary}</p> : null}
                {event.effect_summary ? <p>{event.effect_summary}</p> : null}
                {rosterMediaHeatPhrase(event.heat) ? (
                  <em className="nhlrost-media-heat">{rosterMediaHeatPhrase(event.heat)}</em>
                ) : null}
              </article>
            ))}
          </div>
        </article>
      ) : null}
    </section>
  );
}

function CareerPanel({ player, storylines, franchiseState }) {
  if (!player) {
    return <EmptyPanel title="No career selected" body="Select a player to view career history." compact />;
  }

  const draftYear = player.draft_year || player.draftYear;
  const draftRound = player.draft_round || player.draftRound;
  const draftOverall = player.draft_overall_pick || player.draftOverallPick;
  const draftTeamId =
    player.drafted_by_team_id ||
    player.draft_team_id ||
    player.drafted_by_team ||
    player.draftedByTeamId ||
    "";
  const draftTeamName = player.drafted_by_team_name || resolveDraftTeamName(draftTeamId, franchiseState);
  const isUndrafted = Boolean(player.undrafted) || (!player.drafted && !draftYear && draftOverall == null);

  const isGoalie = isGoaliePosition(player.position);
  const seasons = Array.isArray(player.career_seasons) ? player.career_seasons : EMPTY_ARRAY;
  const awards = Array.isArray(player.career_awards) ? player.career_awards : EMPTY_ARRAY;
  const transactions = Array.isArray(player.transactions) ? player.transactions : EMPTY_ARRAY;
  const events = Array.isArray(storylines) ? storylines : EMPTY_ARRAY;

  const hasAnyCareerData = Boolean(
    player.drafted ||
      draftYear ||
      draftOverall != null ||
      isUndrafted ||
      seasons.length ||
      awards.length ||
      transactions.length ||
      events.length ||
      player.career_totals
  );

  if (!hasAnyCareerData) {
    return (
      <EmptyPanel
        compact
        title="No career record yet"
        body="Draft, season-by-season, award, and transaction data appear here once connected to franchise history."
      />
    );
  }

  return (
    <section className="nhlrost-history-layout">
      <article className="nhlrost-panel">
        <header className="nhlrost-panel__head">
          <div>
            <p>Draft Record</p>
            <h3>{isUndrafted ? "Undrafted" : draftYear ? String(draftYear) : "Drafted"}</h3>
          </div>
        </header>
        {isUndrafted ? (
          <p className="nhlrost-muted-text">Undrafted — entered the organization outside the entry draft.</p>
        ) : (
          <div className="nhlrost-stat-grid nhlrost-stat-grid--wide">
            <InfoPair label="Draft Year" value={draftYear || "—"} />
            <InfoPair label="Round" value={draftRound != null ? draftRound : "—"} />
            <InfoPair label="Overall Pick" value={draftOverall != null ? `#${draftOverall}` : "—"} />
            <InfoPair label="Drafted By" value={draftTeamName || "—"} />
          </div>
        )}
      </article>

      <CareerTotalsCard totals={player.career_totals} isGoalie={isGoalie} />
      <CareerSeasonsTable seasons={seasons} isGoalie={isGoalie} />
      <CareerAwardsCard awards={awards} />
      <CareerTransactionsCard transactions={transactions} />
    </section>
  );
}

function PlayerProfileModal({
  player,
  players,
  playerIndex,
  onSelectPlayer,
  activeTab,
  setActiveTab,
  storylines,
  onClose,
  franchiseState,
  onRefresh,
  onCallMeeting,
}) {
  const modalBodyRef = React.useRef(null);
  const rosterList = Array.isArray(players) ? players : EMPTY_ARRAY;
  const total = rosterList.length;
  const currentIndex = safeNum(playerIndex, -1);
  const hasPrev = total > 0 && currentIndex > 0;
  const hasNext = total > 0 && currentIndex >= 0 && currentIndex < total - 1;

  const goToIndex = useCallback(
    (index) => {
      if (typeof onSelectPlayer !== "function") return;
      if (index < 0 || index >= total) return;
      onSelectPlayer(index);
    },
    [onSelectPlayer, total]
  );

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") {
        onClose();
        return;
      }

      if (event.target?.matches?.("input, textarea, select")) return;

      // `[` / `]` cycle the player without stealing ArrowLeft/ArrowRight,
      // which the roster screen already uses to cycle tabs while open.
      if (event.key === "[") {
        event.preventDefault();
        goToIndex(currentIndex - 1);
        return;
      }

      if (event.key === "]") {
        event.preventDefault();
        goToIndex(currentIndex + 1);
      }
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose, goToIndex, currentIndex]);

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
  const jerseyNumber = resolveJerseyNumber(player);
  const hand = formatHandLabel(player);
  const statusLine = buildCommandStatusLine(player);
  const decisionBullets = buildDecisionBullets(player);

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
              number={jerseyNumber != null ? jerseyNumber : player.num}
              flag={flagCode || null}
            />
          </div>

          <div className="nhlrost-profile-modal__meta">
            <div className="nhlrost-profile-modal__identity-row">
              {flagCode ? <PlayerFlagBadge player={player} size="sm" /> : null}
              <h2 id="nhlrost-profile-title">{player.name}</h2>
              {jerseyNumber != null ? <span className="nhlrost-profile-modal__jersey">#{jerseyNumber}</span> : null}
            </div>
            <p>
              {getPositionDisplay(player.position)} · {player.age || "—"} · {hand}
              {teamLeague ? ` · ${teamLeague}` : ""}
              {player.status && player.status !== "Active" ? ` · ${player.status}` : ""}
            </p>

            <div className="nhlrost-profile-modal__chips">
              <OvrPill player={player} large />
              <PotentialPill player={player} large />
              {player.roleLabel || player.role ? (
                <span className="nhlrost-profile-modal__role">{player.roleLabel || player.role}</span>
              ) : null}
              <span className={`nhlrost-profile-modal__health ${toneClass(healthBand.tone)}`}>{healthBand.label}</span>
              <TradeStabilityConcernBadge player={player} franchiseState={franchiseState} />
              <span className="nhlrost-profile-modal__contract">{contractSummaryDisplay(player)}</span>
            </div>
          </div>

          <div className="nhlrost-profile-modal__nav-close">
            {total > 1 ? (
              <nav className="nhlrost-profile-modal__nav" aria-label="Browse players">
                <button
                  type="button"
                  disabled={!hasPrev}
                  onClick={() => goToIndex(currentIndex - 1)}
                  aria-label="Previous player"
                  title="Previous player ([)"
                >
                  ‹
                </button>
                <span>
                  {currentIndex >= 0 ? currentIndex + 1 : "—"} of {total}
                </span>
                <button
                  type="button"
                  disabled={!hasNext}
                  onClick={() => goToIndex(currentIndex + 1)}
                  aria-label="Next player"
                  title="Next player (])"
                >
                  ›
                </button>
              </nav>
            ) : null}

            <button type="button" className="nhlrost-profile-modal__close" onClick={onClose} aria-label="Close profile">
              ×
            </button>
          </div>
        </header>

        <p className="nhlrost-profile-modal__status-line">{statusLine}</p>

        {decisionBullets.length ? (
          <ul className="nhlrost-profile-modal__decision-strip" aria-label="Decision factors">
            {decisionBullets.map((bullet, index) => (
              <li key={index} className={toneClass(bullet.tone)}>
                {bullet.text}
              </li>
            ))}
          </ul>
        ) : null}

        <DetailTabs activeTab={activeTab} setActiveTab={setActiveTab} />

        <div className="nhlrost-profile-modal__body" ref={modalBodyRef}>
          <DetailPanelRouter
            activeTab={activeTab}
            player={player}
            storylines={storylines}
            franchiseState={franchiseState}
            onRefresh={onRefresh}
            onCallMeeting={onCallMeeting}
          />
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

function PlayerCharacterLifePanel({ player, franchiseState, onCallMeeting }) {
  if (!player) {
    return <EmptyPanel title="No player selected" body="Choose a player from the roster board." />;
  }

  const pid = String(player.id || player.player_id || player.key || "");
  const universe = franchiseState?.narrative_universe || {};
  const meetings = universe?.player_meetings || {};
  const rosterRow = (meetings.roster || []).find((r) => String(r?.player_id || "") === pid) || null;
  const dossier =
    universe?.human_dossiers?.[pid] ||
    universe?.players?.find?.((row) => String(row?.player_id || "") === pid)?.human_dossier ||
    null;

  if (!dossier) {
    return (
      <EmptyPanel
        title="Character profile unavailable"
        body="Human universe data has not synced for this player yet. Advance the calendar to populate off-ice context."
        compact
      />
    );
  }

  const charBlock = dossier.character || {};
  const stateBlock = dossier.current_state || {};
  const lifeBlock = dossier.life || {};
  const mw = dossier.mental_wellbeing || {};

  return (
    <section className="nhlrost-player-overview nhlrost-character-life">
      {rosterRow?.relationship ? (
        <article className="nhlrost-profile-zone nhlrost-profile-zone--relationship">
          <header className="nhlrost-profile-zone__head">
            <p>GM Relationship</p>
            <h3>{rosterRow.relationship.label || "—"}</h3>
          </header>
          <p className="nhlrost-muted-text">{rosterRow.relationship.detail || "No major friction on file."}</p>
          {typeof onCallMeeting === "function" ? (
            <button type="button" className="nhlrost-call-meeting-btn" onClick={() => onCallMeeting(pid)}>
              Call Meeting
            </button>
          ) : null}
        </article>
      ) : null}
      <article className="nhlrost-profile-zone nhlrost-profile-zone--character">
        <header className="nhlrost-profile-zone__head">
          <p>Character</p>
          <h3>{charBlock.headline || "—"}</h3>
        </header>
        {charBlock.summary_line ? <p className="nhlrost-muted-text">{charBlock.summary_line}</p> : null}
        <div className="nhlrost-character-trait-grid">
          {(charBlock.traits || []).map((trait) => (
            <InfoPair key={trait.label} label={trait.label} value={trait.tier || "—"} />
          ))}
        </div>
      </article>

      <article className="nhlrost-profile-zone">
        <header className="nhlrost-profile-zone__head">
          <p>Current State</p>
          <h3>
            Base {stateBlock.base_ovr ?? "—"} · Current {stateBlock.current_ovr ?? "—"}
            {stateBlock.readiness_delta ? ` (${stateBlock.readiness_delta > 0 ? "+" : ""}${stateBlock.readiness_delta})` : ""}
          </h3>
        </header>
        <MetricStrip>
          <Metric label="Morale" value={stateBlock.morale_tier || "—"} />
          <Metric label="Confidence" value={stateBlock.confidence_tier || "—"} />
          <Metric label="Role satisfaction" value={stateBlock.role_satisfaction_tier || "—"} />
          <Metric label="Pressure" value={stateBlock.pressure_label || "Settled"} tone={stateBlock.pressure_tier >= 3 ? "bad" : stateBlock.pressure_tier >= 2 ? "warn" : "neutral"} />
        </MetricStrip>
        {(dossier.pressure_drivers || []).length ? (
          <ul className="nhlrost-character-drivers">
            {dossier.pressure_drivers.map((driver) => (
              <li key={driver.label}>{driver.label}</li>
            ))}
          </ul>
        ) : null}
      </article>

      <article className="nhlrost-profile-zone">
        <header className="nhlrost-profile-zone__head">
          <p>Life</p>
          <h3>{lifeBlock.summary || "—"}</h3>
        </header>
        <MetricStrip>
          <Metric label="City attachment" value={lifeBlock.city_attachment_tier || "—"} />
          <Metric label="Home stability" value={lifeBlock.home_stability_tier || "—"} />
          <Metric label="Relocation" value={lifeBlock.relocation_tier || "—"} />
        </MetricStrip>
      </article>

      {mw?.state ? (
        <article className="nhlrost-profile-zone">
          <header className="nhlrost-profile-zone__head">
            <p>Mental wellbeing</p>
            <h3>{mw.tier || "—"}</h3>
          </header>
          <p className="nhlrost-muted-text">Private team information — not a character judgment.</p>
        </article>
      ) : null}
    </section>
  );
}

function DetailPanelRouter({ activeTab, player, storylines, franchiseState, onRefresh, onCallMeeting }) {
  if (activeTab === "overview") return <PlayerOverviewPanel player={player} franchiseState={franchiseState} />;
  if (activeTab === "character") return <PlayerCharacterLifePanel player={player} franchiseState={franchiseState} onCallMeeting={onCallMeeting} />;
  if (activeTab === "performance") return <ProductionPanel player={player} franchiseState={franchiseState} />;
  if (activeTab === "development") return <DevelopmentPanel player={player} />;
  if (activeTab === "contract") return <ContractPanel player={player} />;
  if (activeTab === "media") return <MediaPanel player={player} storylines={storylines} franchiseState={franchiseState} />;
  if (activeTab === "career") return <CareerPanel player={player} storylines={storylines} franchiseState={franchiseState} />;
  if (activeTab === "moves") return <UsagePanel player={player} onRefresh={onRefresh} />;

  return <PlayerOverviewPanel player={player} franchiseState={franchiseState} />;
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
    setPendingMeetingPlayerId,
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
  const [statsCentralPayload, setStatsCentralPayload] = useState(null);
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

  const franchiseStatsLookup = useMemo(
    () =>
      buildFranchiseStatsLookup({
        ...franchiseState,
        stats_central: franchiseState?.stats_central || statsCentralPayload || EMPTY_OBJECT,
      }),
    [franchiseState, statsCentralPayload]
  );

  useEffect(() => {
    let cancelled = false;
    getStatsCentral()
      .then((payload) => {
        if (cancelled || !payload || typeof payload !== "object") return;
        setStatsCentralPayload(payload);
      })
      .catch(() => {
        if (!cancelled) setStatsCentralPayload(null);
      });
    return () => {
      cancelled = true;
    };
  }, [franchiseState?.session_id, franchiseState?.stats_revision]);

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
          Boolean(player.tradeStabilityConcern) ||
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

  const handleSelectPlayerByIndex = useCallback(
    (index) => {
      const target = filteredPlayers[index];
      if (!target) return;
      highlightPlayer(target);
    },
    [filteredPlayers, highlightPlayer]
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
      .slice(-24)
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

    const org = franchiseState?.roster_browser?.organizations?.find(
      (o) => String(o.team_id) === String(franchiseState?.user_team_id || franchiseState?.team?.id || "")
    );
    const countSigned = (rows) =>
      (Array.isArray(rows) ? rows : []).filter((player) => {
        const c = player?.contract || {};
        const typed = String(c.type || c.contract_type || "").toUpperCase();
        if (["AHL", "ECHL", "AHL_ECHL", "PTO", "ATO", "TRYOUT"].includes(typed)) return false;
        const aav = Number(c.capHit ?? c.aav ?? c.aav_m ?? c.cap_hit_m ?? 0);
        return Boolean(c.isSigned) && aav > 0;
      }).length;
    const orgSpcCount =
      countSigned(org?.nhl) +
      countSigned(org?.ahl) +
      countSigned(org?.echl) +
      countSigned(org?.prospects);
    const backendSlots =
      snap?.nhl_spcs_used ??
      snap?.contract_slots_used ??
      franchiseState?.contract_slots?.used ??
      franchiseState?.contract_slots?.nhl_spcs_used ??
      franchiseState?.team?.nhl_spcs_used ??
      franchiseState?.team?.contract_slots_used;
    const signedContracts =
      backendSlots != null && Number.isFinite(Number(backendSlots))
        ? Number(backendSlots)
        : orgSpcCount > 0
          ? orgSpcCount
          : players.filter((player) => player.contract?.isSigned).length;

    return {
      capLimit: backendCapLimit,
      capUsed,
      capSpace,
      signedContracts,
      contractLimit: NHL_CONTRACT_RESERVE_LIMIT,
      activeLimit: NHL_ACTIVE_ROSTER_LIMIT,
      source: backendCapHit > 0 ? "Backend" : "Computed",
      label: "NHL SPCs",
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

  const handleCallMeeting = useCallback(
    (playerId) => {
      if (!playerId) return;
      setPendingMeetingPlayerId?.(String(playerId));
      openScreen(SCREENS.STORYLINES);
    },
    [openScreen, setPendingMeetingPlayerId]
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
            <span>≡</span>
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
              <button
                type="button"
                className={`nhlrost-hud-tile nhlrost-attention-chip ${showWarningsOnly ? "is-active" : ""}`}
                onClick={() => setShowWarningsOnly((value) => !value)}
                title="Filter to players with roster or trade concerns"
              >
                <div className="nhlrost-hud-tile__body">
                  <small>Alert</small>
                  <strong>{rosterWarnings.length}</strong>
                </div>
              </button>
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
            players={filteredPlayers}
            playerIndex={selectedPlayerIndex}
            onSelectPlayer={handleSelectPlayerByIndex}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            storylines={selectedStorylines}
            franchiseState={franchiseState}
            onRefresh={handleRefresh}
            onCallMeeting={handleCallMeeting}
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

      <GameFooter hints="↑↓ PLAYERS · ENTER PROFILE · [ ] PREV/NEXT PLAYER · ESC CLOSE · MENU" />
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
                  <strong>NHL SPCs</strong>
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
        --bg: var(--ops-navy, #04101a);
        --bg-2: var(--ops-navy-deep, #020a11);
        --panel: var(--ops-panel, rgba(9, 25, 38, 0.94));
        --panel-2: var(--ops-panel-2, rgba(12, 35, 52, 0.94));
        --panel-3: var(--ops-panel-3, rgba(15, 46, 66, 0.78));
        --line: var(--ops-grid, rgba(156, 218, 236, 0.14));
        --line-2: var(--ops-grid-2, rgba(115, 229, 241, 0.25));
        --text: var(--ops-text, #e9f7fb);
        --muted: var(--ops-text-secondary, #8096a8);
        --muted-2: var(--ops-text-disabled, #607789);
        --cyan: var(--ops-cyan, #13d8e7);
        --blue: var(--ops-info, #8ab4ff);
        --gold: var(--ops-gold, #e9a83c);
        --green: var(--ops-success, #52df94);
        --red: var(--ops-injury, #ff606d);
        --orange: var(--shell-orange, #e07020);
        --purple: var(--ops-gold, #e9a83c);
        min-height: 100vh;
        width: 100%;
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr);
        overflow: hidden;
        background:
          radial-gradient(circle at 22% 0%, rgba(19, 216, 231, 0.08), transparent 28%),
          radial-gradient(circle at 88% 12%, rgba(233, 168, 60, 0.06), transparent 24%),
          linear-gradient(180deg, var(--ops-navy, #06111b), var(--ops-navy-deep, #03080e) 72%);
        color: var(--text);
        font-family: var(--font-ops-ui, Inter, ui-sans-serif, system-ui, sans-serif);
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
        border-radius: 6px;
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
        border-radius: 10px;
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
        font-size: 0.6875rem;
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
        border-radius: 12px;
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
        border-radius: 8px;
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
        border-radius: 6px;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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

      button.nhlrost-attention-chip {
        cursor: pointer;
        font: inherit;
        color: inherit;
        text-align: left;
      }

      button.nhlrost-attention-chip:hover {
        border-color: rgba(255, 159, 67, 0.32);
        background: rgba(255, 159, 67, 0.08);
      }

      .nhlrost-board-row.has-trade-concern,
      .nhlrost-row.has-trade-concern {
        box-shadow: inset 3px 0 0 rgba(234, 179, 8, 0.55);
      }

      .nhlrost-board-row.has-trade-concern.is-critical,
      .nhlrost-row.has-trade-concern.is-critical {
        box-shadow: inset 3px 0 0 rgba(239, 68, 68, 0.72);
      }

      .nhlrost-board-row.has-trade-concern.is-selected,
      .nhlrost-row.has-trade-concern.is-selected {
        box-shadow:
          inset 3px 0 0 rgba(239, 68, 68, 0.72),
          inset 0 0 0 1px rgba(19, 216, 231, 0.18);
      }

      .nhlrost-profile-zone--trade-stability {
        border-color: rgba(255, 159, 67, 0.22);
        background: linear-gradient(180deg, rgba(255, 159, 67, 0.06), rgba(255, 159, 67, 0.015));
      }

      .nhlrost-filters-bar {
        flex: 0 0 auto;
        padding: 10px 14px;
        border-radius: 6px;
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
      /* Registry switches: one continuous bank of hard-edged segments so the
         roster filters read as club paperwork rather than floating pills. */
      .nhlrost-search-mode-segment,
      .nhlrost-pool-segment {
        display: grid;
        gap: 0;
        min-width: 0;
        padding: 0;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.02);
        overflow: hidden;
      }

      .nhlrost-search-mode-segment {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        min-width: 240px;
      }

      .nhlrost-search-mode-segment button,
      .nhlrost-pool-segment button {
        min-height: 32px;
        border: 0;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0;
        background: transparent;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          background 140ms ease,
          color 140ms ease;
      }

      .nhlrost-search-mode-segment button:last-child,
      .nhlrost-pool-segment button:last-child {
        border-right: 0;
      }

      .nhlrost-search-mode-segment button:hover,
      .nhlrost-pool-segment button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.05);
      }

      .nhlrost-search-mode-segment button.is-active,
      .nhlrost-pool-segment button.is-active {
        color: #031018;
        background: #00d0d8;
        box-shadow: none;
      }

      .nhlrost-pool-segment {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        min-width: 280px;
      }

      .nhlrost-pool-segment button {
        min-height: 36px;
        padding: 0 8px;
        font-weight: 900;
        letter-spacing: 0.06em;
        white-space: nowrap;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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
        border-radius: var(--radius-hud, 4px);
        border: 1px solid var(--line);
        background: var(--panel);
        overflow: hidden;
        box-shadow: var(--depth-registered, inset 0 1px 0 rgba(255, 255, 255, 0.04));
      }

      .nhlrost-board-shell__head {
        padding: 6px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--line);
        background: rgba(0, 0, 0, 0.22);
      }

      .nhlrost-board-shell__head > span {
        border-radius: var(--radius-ops, 2px);
        padding: 3px 8px;
        color: var(--cyan);
        background: var(--ops-cyan-soft, rgba(19, 216, 231, 0.13));
        border: 1px solid var(--line-2);
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }

      /* Department signature: the personnel-file edge. A punched binder
         margin runs down the sheet so the roster reads as a club document. */
      .nhlrost-board-shell__body {
        position: relative;
        min-height: 0;
        overflow: auto;
        padding: 0;
        border-left: 3px solid rgba(4, 16, 26, 0.9);
        background-image: repeating-linear-gradient(
          180deg,
          var(--line-2) 0 10px,
          transparent 10px 34px
        );
        background-repeat: no-repeat;
        background-size: 1px 100%;
        background-position: 1px 0;
      }

      .nhlrost-board,
      .nhlrost-board-sheet {
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .nhlrost-board-sheet__head {
        position: sticky;
        top: 0;
        z-index: 2;
        display: grid;
        grid-template-columns:
          minmax(168px, 1.55fr)
          44px
          52px
          40px
          72px
          minmax(72px, 0.75fr)
          minmax(84px, 0.85fr)
          minmax(128px, 1.15fr)
          minmax(64px, 0.62fr);
        align-items: center;
        gap: 8px;
        padding: 0 10px;
        min-height: 30px;
        border-bottom: 1px solid var(--line-2);
        background: rgba(4, 16, 26, 0.98);
        color: var(--muted);
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }

      .nhlrost-board-list {
        display: flex;
        flex-direction: column;
        gap: 0;
      }

      .nhlrost-board-row {
        width: 100%;
        min-height: 42px;
        display: grid;
        grid-template-columns:
          minmax(168px, 1.55fr)
          44px
          52px
          40px
          72px
          minmax(72px, 0.75fr)
          minmax(84px, 0.85fr)
          minmax(128px, 1.15fr)
          minmax(64px, 0.62fr);
        align-items: center;
        gap: 8px;
        padding: 0 10px;
        border: 0;
        border-bottom: 1px solid var(--line);
        border-radius: 0;
        background: transparent;
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition:
          background var(--motion-micro, 110ms ease),
          box-shadow var(--motion-micro, 110ms ease);
      }

      .nhlrost-board-row:nth-child(even) {
        background: rgba(255, 255, 255, 0.015);
      }

      .nhlrost-board-row:hover {
        background: var(--ops-cyan-soft, rgba(19, 216, 231, 0.13));
      }

      /* Selected personnel row lifts out of the file like a pulled tab. */
      .nhlrost-board-row.is-selected {
        background: var(--ops-table-sel, rgba(19, 216, 231, 0.13));
        box-shadow: inset 3px 0 0 var(--cyan);
        clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
      }

      .nhlrost-board-row__name {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }

      .nhlrost-board-row__name > span {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 1px;
      }

      .nhlrost-board-row__name .nhlrost-headshot.player-headshot {
        --size: 28px;
        flex: 0 0 auto;
        opacity: 0.92;
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
        border-radius: 11px 10px 10px 9px;
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
        font-size: 0.6875rem;
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
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        min-width: 0;
      }

      .nhlrost-board-row__name-line strong {
        font-size: var(--type-table-value-size, 0.8125rem);
        font-weight: 800;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row__team {
        display: block;
        color: var(--muted);
        font-size: var(--type-table-meta-size, 0.72rem);
        font-style: normal;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row__pos {
        font-size: var(--type-table-meta-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        text-align: center;
      }

      .nhlrost-board-row__age,
      .nhlrost-board-row__role,
      .nhlrost-board-row__avail {
        font-size: var(--type-table-value-size, 0.8125rem);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-board-row__contract {
        color: var(--gold);
        font-size: var(--type-financial-size, 0.875rem);
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
      }

      .nhlrost-board-row__status {
        display: flex;
        align-items: center;
        min-width: 0;
      }

      .nhlrost-board-row__stats {
        color: var(--muted);
        font-size: var(--type-table-meta-size, 0.72rem);
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-ovr-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        min-width: 36px;
        min-height: 24px;
        padding: 0 6px;
        border-radius: var(--radius-ops, 2px);
        font-size: var(--type-score-size, 1.25rem);
        font-weight: 900;
        line-height: 1;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.03);
        color: var(--text);
        box-shadow: none;
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
        font-size: 0.6875rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        color: #dff7ff;
        background: rgba(143, 211, 255, 0.12);
        border: 1px solid rgba(143, 211, 255, 0.28);
      }

      .nhlrost-flag-fallback.is-lg {
        min-width: 42px;
        height: 28px;
        font-size: 0.6875rem;
      }

      .nhlrost-headshot .ph-flag {
        font-size: 0.6875rem;
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
        border-radius: 6px;
        font-size: 0.6875rem;
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
        display: flex;
        flex-direction: column;
        border-radius: 8px;
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
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__jersey {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
      }

      .nhlrost-profile-modal__nav-close {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__nav {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .nhlrost-profile-modal__nav button {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text);
        font-size: 1rem;
        line-height: 1;
        cursor: pointer;
      }

      .nhlrost-profile-modal__nav button:disabled {
        opacity: 0.35;
        cursor: default;
      }

      .nhlrost-profile-modal__nav span {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .nhlrost-profile-modal__contract {
        font-size: 0.6875rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: var(--cyan);
        border-radius: 999px;
        padding: 4px 10px;
        border: 1px solid rgba(19, 216, 231, 0.25);
        background: rgba(19, 216, 231, 0.07);
      }

      .nhlrost-profile-modal__status-line {
        margin: 0;
        padding: 8px 16px;
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.01em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__decision-strip {
        margin: 0;
        list-style: none;
        padding: 8px 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        flex-shrink: 0;
      }

      .nhlrost-profile-modal__decision-strip li {
        position: relative;
        padding-left: 12px;
        font-size: 0.78rem;
        color: var(--text);
      }

      .nhlrost-profile-modal__decision-strip li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.5em;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--muted);
      }

      .nhlrost-profile-modal__decision-strip li.is-warn::before {
        background: var(--gold);
      }

      .nhlrost-profile-modal__decision-strip li.is-bad::before {
        background: var(--red);
      }

      .nhlrost-profile-modal__decision-strip li.is-good::before {
        background: var(--green);
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
        font-size: 0.6875rem;
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
        flex: 1 1 auto;
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
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-profile-scorecard span {
        color: var(--muted);
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .nhlrost-contract-hero strong {
        font-size: 1.8rem;
        font-weight: 900;
      }

      .nhlrost-storyline-card {
        border-radius: 10px;
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

      .nhlrost-media-pressure {
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--gold);
        border: 1px solid rgba(233, 168, 60, 0.35);
        padding: 4px 8px;
        border-radius: 4px;
      }
      .nhlrost-media-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .nhlrost-media-tag {
        font-size: 0.72rem;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid rgba(233, 168, 60, 0.28);
        background: rgba(233, 168, 60, 0.08);
        color: #f4d9a6;
      }
      .nhlrost-media-arcs {
        display: grid;
        gap: 10px;
      }
      .nhlrost-media-arc {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
        background: rgba(255, 255, 255, 0.02);
      }
      .nhlrost-media-arc__head {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: baseline;
        margin-bottom: 6px;
      }
      .nhlrost-media-arc__head span {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        white-space: nowrap;
      }
      .nhlrost-media-timeline {
        margin: 0;
        padding: 0 0 0 16px;
        display: grid;
        gap: 8px;
      }
      .nhlrost-media-timeline time {
        display: block;
        font-size: 0.68rem;
        color: var(--muted);
        font-family: var(--font-mono-data, monospace);
      }
      .nhlrost-media-timeline p {
        margin: 2px 0 0;
        font-size: 0.78rem;
        color: rgba(232, 244, 251, 0.78);
      }
      .nhlrost-media-timeline em {
        font-style: normal;
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--gold);
      }
      .nhlrost-media-quotes {
        display: grid;
        gap: 8px;
      }
      .nhlrost-media-quotes blockquote {
        margin: 0;
        padding: 10px 12px;
        border-left: 3px solid var(--cyan);
        background: rgba(19, 216, 231, 0.06);
        border-radius: 0 8px 8px 0;
      }
      .nhlrost-media-quotes blockquote strong {
        display: block;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--cyan);
        margin-bottom: 4px;
      }
      .nhlrost-media-quotes blockquote p {
        margin: 0;
        font-size: 0.82rem;
        line-height: 1.4;
      }
      .nhlrost-media-heat {
        display: inline-block;
        margin-top: 6px;
        font-style: normal;
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--gold);
      }

      .nhlrost-inspector {
        flex: 0 0 auto;
        max-height: min(38vh, 420px);
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        border-radius: var(--radius-hud, 4px);
        border: 1px solid var(--line-2);
        border-top: 2px solid var(--cyan);
        background: var(--panel);
        overflow: hidden;
        box-shadow: var(--depth-registered, inset 0 1px 0 rgba(255, 255, 255, 0.04));
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
        border-radius: 10px;
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

      /* Personnel-registry controls: squared operational keys, flat fills,
         and a 1px press instead of a floating lift. */
      .nhlrost-chip-button,
      .nhlrost-primary-button {
        min-height: 34px;
        border-radius: var(--radius-hud, 4px);
        padding: 0 13px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.045);
        color: var(--text);
        font-size: 0.72rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          background 150ms ease,
          border-color 150ms ease,
          color 150ms ease;
      }

      .nhlrost-chip-button:hover,
      .nhlrost-primary-button:hover {
        border-color: rgba(0, 216, 223, 0.36);
        background: rgba(0, 216, 223, 0.08);
      }

      .nhlrost-chip-button:active,
      .nhlrost-primary-button:active {
        transform: translateY(1px);
      }

      .nhlrost-chip-button.is-active {
        border-color: rgba(0, 216, 223, 0.42);
        background: rgba(0, 216, 223, 0.12);
        box-shadow: inset 3px 0 0 var(--ops-cyan, #13d8e7);
      }

      /* Primary roster action carries the rink cut. */
      .nhlrost-primary-button {
        background: rgba(0, 216, 223, 0.16);
        border-color: rgba(0, 216, 223, 0.45);
        border-radius: 0;
        clip-path: polygon(0 0, calc(100% - 9px) 0, 100% 9px, 100% 100%, 9px 100%, 0 calc(100% - 9px));
      }

      /* Read-only registry notice reads as a filed stamp. */
      .nhlrost-readonly-pill {
        min-height: 34px;
        border-radius: var(--radius-ops, 2px);
        padding: 0 12px;
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
        border-radius: 8px;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
        font-weight: 900;
      }

      .nhlrost-panel > header h3,
      .nhlrost-panel__head h3 {
        margin: 0;
        font-size: 1rem;
        line-height: 1.1;
      }

      .nhlrost-panel__head > span {
        border-radius: var(--radius-ops, 2px);
        padding: 3px 7px;
        letter-spacing: 0.08em;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.16);
        font-size: 0.6875rem;
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
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 9px 10px;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .nhlrost-info-pair span {
        color: var(--muted);
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 900;
      }

      .nhlrost-info-pair strong {
        font-size: 0.86rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        word-break: normal;
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
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 12px;
        display: flex;
        justify-content: space-between;
        gap: 10px;
      }

      .nhlrost-cap-meter span {
        color: var(--muted);
        font-size: 0.6875rem;
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

      /* Roster limits read against a ruled certification scale. */
      .nhlrost-progress__track {
        height: 8px;
        border-radius: 1px;
        overflow: hidden;
        background:
          repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.14) 0 1px, transparent 1px 25%),
          rgba(255, 255, 255, 0.11);
      }

      .nhlrost-progress__track span {
        display: block;
        height: 100%;
        border-radius: 0;
        background: var(--cyan);
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
        border-radius: 6px;
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
        font-size: 0.6875rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 950;
      }

      .nhlrost-control input,
      .nhlrost-control select {
        width: 100%;
        height: 38px;
        border-radius: 10px;
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
        font-size: 0.6875rem;
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
        border-radius: var(--radius-ops, 2px);
        padding: 4px 8px;
        letter-spacing: 0.08em;
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
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(10, 28, 42, 0.98), rgba(6, 18, 29, 0.98));
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.6875rem;
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
        border-radius: 6px;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
        font-weight: 800;
      }

      .nhlrost-mini-badge {
        max-width: 100%;
        min-height: 22px;
        border-radius: var(--radius-ops, 2px);
        padding: 3px 7px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.045);
        color: var(--text);
        font-size: 0.6875rem;
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
        min-height: 22px;
        border-radius: var(--radius-ops, 2px);
        padding: 3px 7px;
        display: inline-flex;
        align-items: center;
        border: 1px solid color-mix(in srgb, var(--arch-color) 72%, transparent);
        background: color-mix(in srgb, var(--arch-color) 15%, transparent);
        color: var(--arch-color);
        font-style: normal;
        font-size: 0.6875rem;
        font-weight: 950;
        letter-spacing: 0.06em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-cancer-tag {
        display: inline-block;
        margin-left: 8px;
        padding: 2px 10px;
        border-radius: 2px;
        background: #7a0f0f;
        color: #ffd0d0;
        border: 2px solid #ff3b3b;
        font-style: normal;
        font-size: 0.78rem;
        font-weight: 950;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        vertical-align: middle;
        box-shadow: 0 0 0 1px #2a0000;
      }

      .nhlrost-status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nhlrost-card-grid,
      .nhlrost-ratings-grid-view {
        display: flex;
        flex-direction: column;
        gap: 0;
        border: 1px solid var(--line);
        border-radius: var(--radius-hud, 4px);
        overflow: hidden;
      }

      .nhlrost-player-card,
      .nhlrost-rating-card {
        min-width: 0;
        border-radius: 0;
        border: 0;
        border-bottom: 1px solid var(--line);
        background: transparent;
        padding: 8px 10px;
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition: background var(--motion-micro, 110ms ease);
        box-shadow: none;
      }

      .nhlrost-player-card:hover,
      .nhlrost-rating-card:hover {
        transform: none;
        background: var(--ops-cyan-soft, rgba(19, 216, 231, 0.13));
      }

      .nhlrost-player-card.is-selected,
      .nhlrost-rating-card.is-selected {
        background: var(--ops-table-sel, rgba(19, 216, 231, 0.13));
        box-shadow: inset 3px 0 0 var(--cyan);
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
        border-radius: var(--radius-ops, 2px);
        padding: 3px 7px;
        font-size: 0.6875rem;
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
        border-radius: 8px;
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
        border-radius: 10px;
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
        border-radius: 6px;
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
        font-size: 0.6875rem;
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
        min-height: 28px;
        border-radius: var(--radius-hud, 4px);
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
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 12px;
      }

      .nhlrost-selected-card__big span {
        display: block;
        color: var(--muted);
        font-size: 0.6875rem;
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
        flex-shrink: 0;
      }

      .nhlrost-detail-tabs button {
        min-height: 32px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        padding: 0 12px;
        font-size: 0.6875rem;
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
        grid-template-columns: minmax(200px, 0.95fr) minmax(0, 1.15fr) minmax(0, 1.35fr);
        grid-template-rows: auto auto auto;
        gap: 0;
        align-content: start;
        border: 1px solid var(--line);
        border-radius: var(--radius-hud, 4px);
        overflow: hidden;
        background: rgba(0, 0, 0, 0.18);
      }

      .nhlrost-profile-zone {
        border-radius: 0;
        border: 0;
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: transparent;
        padding: 10px 12px;
        min-width: 0;
      }

      .nhlrost-profile-zone--bio {
        grid-column: 1;
        grid-row: 1 / span 2;
        background: rgba(19, 216, 231, 0.04);
        border-left: 3px solid var(--cyan);
      }

      .nhlrost-profile-zone--draft {
        grid-column: 2;
        grid-row: 1;
      }

      .nhlrost-profile-zone--ability {
        grid-column: 3;
        grid-row: 1;
        border-right: 0;
      }

      .nhlrost-profile-zone--contract {
        grid-column: 2;
        grid-row: 2;
        border-right: 1px solid var(--line);
      }

      .nhlrost-profile-zone--performance {
        grid-column: 3;
        grid-row: 2;
        border-right: 0;
        background: rgba(0, 0, 0, 0.14);
      }

      .nhlrost-profile-zone--ratings,
      .nhlrost-profile-zone--signal {
        grid-column: 1 / -1;
        border-right: 0;
      }

      .nhlrost-profile-zone--signal {
        border-bottom: 0;
        background: rgba(0, 0, 0, 0.14);
      }

      .nhlrost-metric-strip {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        align-items: stretch;
        gap: 0;
        width: 100%;
      }

      .nhlrost-metric {
        flex: 1 1 68px;
        min-width: 60px;
        max-width: 120px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 3px;
        padding: 6px 10px 6px 0;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        min-height: 0;
      }

      .nhlrost-metric:last-child {
        border-right: 0;
        padding-right: 0;
      }

      .nhlrost-metric span {
        color: var(--muted);
        font-size: 0.625rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        line-height: 1.2;
      }

      .nhlrost-metric strong {
        font-size: 0.92rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .nhlrost-metric.is-good strong {
        color: var(--green);
      }

      .nhlrost-metric.is-warn strong {
        color: var(--gold);
      }

      .nhlrost-metric.is-bad strong {
        color: #f0a0a0;
      }

      .nhlrost-metric-strip--stats .nhlrost-metric {
        flex: 1 1 56px;
        max-width: 96px;
      }

      .nhlrost-profile-zone__head p {
        margin: 0 0 2px;
        color: var(--muted);
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }

      .nhlrost-profile-zone__head h3 {
        margin: 0 0 8px;
        font-size: var(--type-ops-heading-size, 0.95rem);
        line-height: 1.2;
        font-weight: 800;
      }

      .nhlrost-profile-zone--bio .nhlrost-profile-zone__head h3 {
        font-size: 1.05rem;
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
        font-size: 0.6875rem;
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
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 0;
      }

      .nhlrost-profile-modal__body .nhlrost-stat-grid--wide {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-profile-card__score-row span {
        display: block;
        color: var(--muted);
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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
        font-size: 0.6875rem;
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
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-engine-score span {
        display: block;
        color: var(--muted);
        font-size: 0.6875rem;
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
        border-radius: 6px;
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
        border-radius: var(--radius-hud, 4px);
        border: 1px solid var(--line);
        border-left: 3px solid var(--muted);
        background: rgba(0, 0, 0, 0.22);
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        padding: 20px 18px;
        text-align: left;
      }

      .nhlrost-empty-panel.is-compact {
        min-height: 0;
        padding: 12px 14px;
        align-items: stretch;
      }

      .nhlrost-empty-panel__phase {
        margin: 0 0 6px;
        color: var(--cyan);
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .nhlrost-empty-panel__orb {
        display: none;
      }

      .nhlrost-empty-panel h3 {
        margin: 0;
        font-size: var(--type-ops-heading-size, 0.95rem);
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .nhlrost-empty-panel p {
        margin: 6px 0 0;
        max-width: 520px;
        color: var(--muted);
        font-size: var(--type-body-size, 0.875rem);
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
        font-size: 0.6875rem;
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
        border-radius: 10px;
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
        min-height: 32px;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        cursor: pointer;
        font-size: 0.6875rem;
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
        border-radius: 8px;
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
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
      }

      .nhlrost-drawer-metric-grid span {
        display: block;
        color: var(--muted);
        font-size: 0.6875rem;
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
        border-radius: 8px;
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
        font-size: 0.6875rem;
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
        border-radius: 6px;
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
          font-size: 0.6875rem;
          padding: 0 6px;
        }

        .nhlrost-board-row,
        .nhlrost-board-sheet__head {
          grid-template-columns:
            minmax(120px, 1.4fr)
            36px
            44px
            36px
            64px
            minmax(64px, 0.7fr)
            minmax(72px, 0.75fr)
            minmax(100px, 1fr)
            56px;
          gap: 6px;
          padding: 0 8px;
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

        .nhlrost-profile-zone--bio,
        .nhlrost-profile-zone--draft,
        .nhlrost-profile-zone--ability,
        .nhlrost-profile-zone--contract,
        .nhlrost-profile-zone--performance {
          grid-column: 1;
          grid-row: auto;
          border-right: 0;
        }

        .nhlrost-detail-grid--ratings {
          grid-template-columns: 1fr;
        }

        .nhlrost-profile-modal__panel {
          width: min(100%, calc(100vw - 32px));
          max-height: calc(100dvh - 32px);
        }

        .nhlrost-board-row,
        .nhlrost-board-sheet__head {
          grid-template-columns: minmax(0, 1fr) 36px 44px 36px 64px;
        }

        .nhlrost-board-row__role,
        .nhlrost-board-row__stats,
        .nhlrost-board-row__avail,
        .nhlrost-board-sheet__head span:nth-child(n+6) {
          display: none;
        }

        .nhlrost-board-row__status {
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

        .nhlrost-board-row,
        .nhlrost-board-sheet__head {
          grid-template-columns: minmax(0, 1fr) 32px 40px;
          min-height: 38px;
        }

        .nhlrost-board-row__age,
        .nhlrost-board-row__contract,
        .nhlrost-board-row__status,
        .nhlrost-board-sheet__head span:nth-child(n+4) {
          display: none;
        }

        .nhlrost-board-row__name .nhlrost-headshot.player-headshot {
          --size: 24px;
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

      /* ─── Personnel sheet corrections ─────────────────────────────────
         The roster is an official team sheet, so the header reads as one
         divided register strip instead of six free-floating metric cards,
         and every column is wide enough for its own label and values. */

      .nhlrost-command-bar__metrics {
        gap: 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-hud, 4px);
        background: rgba(4, 16, 26, 0.4);
        overflow: hidden;
      }

      .nhlrost-hud-tile {
        border: 0;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0;
        background: transparent;
        padding: 6px 12px;
      }

      .nhlrost-hud-tile:last-child {
        border-right: 0;
      }

      .nhlrost-command-bar {
        min-height: 0;
        padding: 8px 12px;
      }

      /* CONTRACT was clipped at 72px and AVAIL truncated every value while
         SEASON held a quarter of the sheet for a single em dash. */
      .nhlrost-board-sheet__head,
      .nhlrost-board-row {
        grid-template-columns:
          minmax(168px, 1.5fr)
          44px
          52px
          40px
          86px
          minmax(76px, 0.7fr)
          minmax(88px, 0.8fr)
          minmax(104px, 0.9fr)
          minmax(92px, 0.72fr);
      }

      .nhlrost-board-sheet__head span,
      .nhlrost-board-row__avail,
      .nhlrost-board-row__status {
        overflow: hidden;
        text-overflow: ellipsis;
      }

      /* Selection is a single leading rail plus a wash — one treatment, so a
         selected row cannot be confused with a hover or a status accent. */
      .nhlrost-board-row.is-selected {
        background: var(--ops-table-sel, rgba(19, 216, 231, 0.13));
        box-shadow: inset 3px 0 0 var(--ops-cyan, #13d8e7);
      }

      .nhlrost-board-row:focus-visible {
        outline-offset: -2px;
      }

      /* A pill on every one of 23 rows reads as decoration. In the board the
         normal state is a quiet mark; only exceptions keep a filled badge. */
      .nhlrost-board-row .nhlrost-mini-badge.is-good {
        min-height: 0;
        padding: 0;
        border: 0;
        background: transparent;
        color: var(--muted);
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .nhlrost-board-row .nhlrost-mini-badge.is-warn,
      .nhlrost-board-row .nhlrost-mini-badge.is-bad,
      .nhlrost-board-row .nhlrost-mini-badge.is-medical {
        border-radius: var(--radius-ops, 2px);
      }

      /* One control row: the lone sort field did not need a second band. */
      .nhlrost-filters-bar {
        padding-block: 8px;
      }

      /* ─── Personnel file (player dossier) ─────────────────────────────
         Previously eighteen identically weighted label/value cards, which
         gave age, hand, league and cap hit the same importance. Now the
         file reads as a registry: hairline definition rows under a single
         section rule, with identity and ability carrying the weight. */

      .nhlrost-profile-modal__panel {
        border-radius: var(--radius-panel, 10px);
        max-height: calc(100dvh - 24px);
      }

      .nhlrost-profile-modal__hero {
        padding: 10px 16px 8px;
        border-bottom: 2px solid var(--ops-cyan, #13d8e7);
      }

      .nhlrost-profile-modal__headshot.player-headshot.size-md {
        --size: 84px;
      }

      .nhlrost-profile-modal__meta h2 {
        font-size: 1.5rem;
        letter-spacing: 0.01em;
      }

      .nhlrost-profile-modal__role,
      .nhlrost-profile-modal__health {
        border-radius: var(--radius-ops, 2px);
        padding: 3px 8px;
      }

      .nhlrost-info-pair {
        border: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0;
        background: transparent;
        padding: 5px 2px;
        display: grid;
        grid-template-columns: minmax(78px, 0.85fr) minmax(0, 1fr);
        align-items: baseline;
        gap: 10px;
      }

      .nhlrost-info-pair span {
        letter-spacing: 0.1em;
        font-weight: 800;
      }

      .nhlrost-info-pair strong {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }

      /* Column separation so a value never reads as part of the next label. */
      .nhlrost-info-pair:nth-child(odd) {
        padding-right: 14px;
      }

      .nhlrost-info-pair:nth-child(even) {
        padding-left: 14px;
        border-left: 1px solid rgba(255, 255, 255, 0.07);
      }

      /* Ability, contract and production are the three registry blocks; the
         section rule replaces per-card borders. */
      .nhlrost-profile-scorecard {
        border: 0;
        border-left: 3px solid var(--ops-cyan, #13d8e7);
        border-radius: 0;
        background: rgba(19, 216, 231, 0.05);
      }

      .nhlrost-profile-scorecard:nth-child(2) {
        border-left-color: var(--ops-gold, #e9a83c);
        background: rgba(233, 168, 60, 0.05);
      }

      .nhlrost-profile-scorecard:nth-child(3) {
        border-left-color: var(--ops-info, #8ab4ff);
        background: rgba(138, 180, 255, 0.05);
      }

      /* Dossier tabs read as file dividers, not a pill collection. */
      .nhlrost-detail-tabs {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        gap: 0;
        padding-inline: 12px;
      }

      .nhlrost-detail-tabs button {
        border: 0;
        border-radius: 0;
        background: transparent;
        padding: 7px 13px;
        box-shadow: inset 0 -2px 0 transparent;
      }

      .nhlrost-detail-tabs button:hover {
        background: rgba(255, 255, 255, 0.03);
        box-shadow: inset 0 -2px 0 rgba(19, 216, 231, 0.35);
      }

      .nhlrost-detail-tabs button.is-active {
        background: rgba(19, 216, 231, 0.08);
        color: var(--ops-cyan, #13d8e7);
        box-shadow: inset 0 -2px 0 var(--ops-cyan, #13d8e7);
      }

      .nhlrost-profile-modal__body {
        animation: inspectorSwap 190ms var(--ease-out-expo, cubic-bezier(0.16, 1, 0.3, 1)) both;
      }

      @media (prefers-reduced-motion: reduce) {
        .nhlrost-profile-modal__body {
          animation: none !important;
        }
      }

      @media (max-height: 800px) {
        .nhlrost-command-bar {
          padding: 5px 12px;
        }

        .nhlrost-hud-tile {
          padding: 4px 10px;
        }

        .nhlrost-board-row {
          min-height: 38px;
        }
      }

      /* ─── Dossier: attribute profile, strengths/concerns, development
         curve, and career registry — flat report sections, not card walls. */

      .nhlrost-overview-ratings-bars {
        display: flex;
        flex-direction: column;
        gap: 9px;
      }

      .nhlrost-ratings-expand-toggle {
        margin-top: 10px;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        background: transparent;
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 8px 0 0;
        cursor: pointer;
        width: 100%;
        text-align: left;
      }

      .nhlrost-overview-ratings-expanded {
        margin-top: 10px;
      }

      .nhlrost-sc-columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }

      .nhlrost-sc-column {
        min-width: 0;
      }

      .nhlrost-sc-column__label {
        display: block;
        margin-bottom: 6px;
        font-size: 0.6875rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .nhlrost-sc-column__label.is-good {
        color: var(--green);
      }

      .nhlrost-sc-column__label.is-warn {
        color: var(--gold);
      }

      .nhlrost-sc-column ul {
        margin: 0;
        padding: 0;
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-sc-column li {
        position: relative;
        padding-left: 12px;
        font-size: 0.8rem;
        line-height: 1.4;
        color: var(--text);
      }

      .nhlrost-sc-column li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.55em;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: var(--muted);
      }

      .nhlrost-dev-timeline-panel {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .nhlrost-dev-chart {
        width: 100%;
        height: auto;
        margin: 4px 0 6px;
      }

      .nhlrost-dev-chart__line {
        stroke: var(--cyan);
        stroke-width: 2;
      }

      .nhlrost-dev-chart__dot {
        fill: var(--cyan);
      }

      .nhlrost-dev-chart__value {
        fill: var(--text);
        font-size: 11px;
        font-weight: 700;
      }

      .nhlrost-dev-chart__label {
        fill: var(--muted);
        font-size: 10px;
        letter-spacing: 0.04em;
      }

      .nhlrost-dev-timeline-list {
        display: flex;
        flex-direction: column;
        gap: 0;
      }

      .nhlrost-dev-timeline-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 48px 48px;
        align-items: baseline;
        gap: 10px;
        padding: 6px 2px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.8rem;
      }

      .nhlrost-dev-timeline-row span {
        color: var(--muted);
      }

      .nhlrost-dev-timeline-row strong {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }

      .nhlrost-dev-timeline-row em {
        font-style: normal;
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: 800;
        color: var(--muted);
      }

      .nhlrost-dev-timeline-row em.is-up {
        color: var(--green);
      }

      .nhlrost-dev-timeline-row em.is-down {
        color: var(--red);
      }

      .nhlrost-table-scroll {
        overflow-x: auto;
      }

      .nhlrost-mini-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
      }

      .nhlrost-mini-table th,
      .nhlrost-mini-table td {
        padding: 6px 8px;
        text-align: right;
        white-space: nowrap;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      }

      .nhlrost-mini-table th:first-child,
      .nhlrost-mini-table td:first-child,
      .nhlrost-mini-table th:nth-child(2),
      .nhlrost-mini-table td:nth-child(2),
      .nhlrost-mini-table th:nth-child(3),
      .nhlrost-mini-table td:nth-child(3) {
        text-align: left;
      }

      .nhlrost-mini-table th {
        color: var(--muted);
        font-size: 0.6875rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255, 255, 255, 0.14);
      }

      .nhlrost-award-list {
        margin: 0;
        padding: 0;
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .nhlrost-award-list li {
        position: relative;
        padding-left: 14px;
        font-size: 0.84rem;
      }

      .nhlrost-award-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.5em;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--gold);
      }

      @media (max-width: 900px) {
        .nhlrost-sc-columns {
          grid-template-columns: 1fr;
        }

        .nhlrost-profile-modal__hero {
          grid-template-columns: auto minmax(0, 1fr);
        }

        .nhlrost-profile-modal__nav-close {
          grid-column: 1 / -1;
          flex-direction: row;
          justify-content: space-between;
          align-items: center;
        }
      }

      .nhlrost-call-meeting-btn {
        margin-top: 10px;
        border: 1px solid rgba(19, 216, 231, 0.35);
        border-radius: 6px;
        background: rgba(19, 216, 231, 0.08);
        color: var(--cyan);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 8px 12px;
        cursor: pointer;
      }
      .nhlrost-call-meeting-btn:hover {
        border-color: var(--cyan);
        background: rgba(19, 216, 231, 0.14);
      }
      .nhlrost-profile-zone--relationship h3 {
        color: var(--gold);
      }
    `}</style>
  );
}