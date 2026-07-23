import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getStatsCentral } from "../services/franchiseService";
import { formatFranchiseApiError, isExpiredFranchiseSessionError } from "../services/api";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import {
  getBaseOverall,
  getOverallDrop,
  getOverallTooltip,
  getUniversalOverall,
} from "../utils/playerOverall";

import {
  attachLogosToTeamRows,
  attachTeamLogosToRows,
  getPlayerTeamLogoSrc,
  getTeamLogoSrc,
} from "../utils/teamLogos";

/*
===========================================================
STATS CENTRAL — NHL FRANCHISE MODE
===========================================================

FULL FRONTEND DISPLAY FOR THE NEW BACKEND ANALYTICS PIPELINE

Backend changes this file expects:
- engine.py is now the real game-ledger source of truth.
- franchise_sim.py sends the franchise state payload.
- player_analytics.py enriches real ledger stats.
- goals, assists, points, goalie stats, xG, Corsi, Fenwick,
  PDO, GSAx, player impact, award watch, and team analytics
  should now be read from backend rows when available.

Frontend rules:
- DO NOT invent core counting stats.
- DO NOT scale goals, assists, points, PIM, goalie wins, losses, saves.
- DO NOT fake analytics when backend provides the real values.
- DO gracefully fallback when a field is missing.
- DO keep the UI stable even if the backend payload shape changes slightly.

Visual rules:
- Background colour system matches RosterScreen.js:
  deep navy, dark panels, cyan accents, gold accents.
- Menu/tab icons are smaller and less noisy.
- Buttons are not copied directly from RosterScreen.js.
- Stats Central gets its own executive dashboard feel.

===========================================================
*/


/* =========================================================
   SMALL SAFE HELPERS
========================================================= */

function safe(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function safeInt(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.round(n);
}

function safeString(value, fallback = "") {
  if (value === null || value === undefined) return fallback;
  return String(value);
}

function cleanText(value, fallback = "—") {
  const s = safeString(value, "").trim();
  return s || fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function firstPresent(...values) {
  for (const value of values) {
    if (value !== null && value !== undefined && value !== "") {
      return value;
    }
  }
  return undefined;
}

function pickStat(...values) {
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] === null || values[i] === undefined || values[i] === "") continue;
    const n = Number(values[i]);
    if (Number.isFinite(n)) return n;
  }
  return 0;
}

function pickString(...values) {
  for (let i = 0; i < values.length; i += 1) {
    const raw = values[i];
    if (raw !== null && raw !== undefined && String(raw).trim()) {
      return String(raw);
    }
  }
  return "";
}

function pct(a, b, fallback = 0) {
  const numerator = Number(a);
  const denominator = Number(b);
  if (!Number.isFinite(numerator)) return fallback;
  if (!Number.isFinite(denominator) || denominator <= 0) return fallback;
  return numerator / denominator;
}

function pct100(a, b, fallback = 0) {
  return pct(a, b, fallback) * 100;
}

function normalizePct(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  if (Math.abs(n) > 1.5) return n / 100;
  return n;
}

function normalizePdo(value, fallback = undefined) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  if (Math.abs(n) <= 10) return n * 100;
  return n;
}

function fmtPdo(value) {
  const n = normalizePdo(value, null);
  if (n === null || n === undefined) return "—";
  return String(Math.round(n));
}

function fmtPct(value, digits = 1) {
  const n = normalizePct(value, 0);
  return `${(n * 100).toFixed(digits)}%`;
}

function fmtDecimal(value, digits = 3) {
  return safe(value, 0).toFixed(digits);
}

function fmtOne(value) {
  return safe(value, 0).toFixed(1);
}

function fmtTwo(value) {
  return safe(value, 0).toFixed(2);
}

function fmtZero(value) {
  return String(safeInt(value, 0));
}

function perGame(value, gp) {
  const games = safe(gp, 0);
  if (games <= 0) return 0;
  return safe(value, 0) / games;
}

function per60(value, toiMinutes) {
  const minutes = safe(toiMinutes, 0);
  if (minutes <= 0) return 0;
  return (safe(value, 0) / minutes) * 60;
}

function per82(value, gp) {
  const games = safe(gp, 0);
  if (games <= 0) return 0;
  return (safe(value, 0) / games) * 82;
}

function roundTo(value, digits = 2) {
  const n = safe(value, 0);
  const mult = Math.pow(10, digits);
  return Math.round(n * mult) / mult;
}

function formatSigned(value, digits = 0) {
  const n = safe(value, 0);
  const fixed = digits > 0 ? n.toFixed(digits) : String(Math.round(n));
  return n > 0 ? `+${fixed}` : fixed;
}

function formatMoney(value) {
  const n = safe(value, 0);
  if (!n) return "—";
  return `$${n.toFixed(2)}M`;
}

function normalizePosition(value) {
  const raw = safeString(value, "F").trim().toUpperCase();

  if (["GOALIE", "GOALTENDER", "NETMINDER"].includes(raw)) return "G";
  if (["DEFENSE", "DEFENCE", "DEFENSEMAN", "DEFENCEMAN"].includes(raw)) return "D";
  if (["CENTER", "CENTRE"].includes(raw)) return "C";
  if (["LEFT WING", "LEFTWING", "LWING"].includes(raw)) return "LW";
  if (["RIGHT WING", "RIGHTWING", "RWING"].includes(raw)) return "RW";

  return raw || "F";
}

function getPositionFullName(position) {
  const pos = normalizePosition(position);

  if (pos === "G") return "Position: Goalie";
  if (pos === "D") return "Position: Defense";
  if (pos === "C") return "Position: Center";
  if (pos === "LW") return "Position: Left Wing";
  if (pos === "RW") return "Position: Right Wing";
  if (pos === "F") return "Position: Forward";

  return `Position: ${pos}`;
}

function isGoalieRow(row) {
  return normalizePosition(firstPresent(row?.position, row?.pos)) === "G";
}

function isDefenseRow(row) {
  return normalizePosition(firstPresent(row?.position, row?.pos)) === "D";
}

function isForwardRow(row) {
  const pos = normalizePosition(firstPresent(row?.position, row?.pos));
  return ["C", "LW", "RW", "F"].includes(pos);
}

function playerName(row) {
  return pickString(row?.name, row?.player_name, row?.full_name, "Unknown Player");
}

function playerId(row, index = 0) {
  return pickString(
    row?.player_id,
    row?.id,
    row?.pid,
    `${playerName(row).replace(/\s+/g, "_").toLowerCase()}_${index}`
  );
}

function teamIdFromInfo(teamInfo, fallback = "USR") {
  return pickString(
    teamInfo?.team_id,
    teamInfo?.id,
    teamInfo?.abbrev,
    teamInfo?.abbr,
    teamInfo?.code,
    teamInfo?.name,
    fallback
  );
}

function teamNameFromInfo(teamInfo, fallback = "Franchise") {
  return pickString(
    teamInfo?.name,
    teamInfo?.team_name,
    teamInfo?.full_name,
    teamInfo?.abbrev,
    teamInfo?.id,
    fallback
  );
}

function fmtScore(game) {
  const home = pickStat(game?.home_goals, 0);
  const away = pickStat(game?.away_goals, 0);
  const ot = game?.overtime ? " OT" : "";
  const so = game?.shootout ? " SO" : "";
  return `${home}-${away}${ot}${so}`;
}

function getTotalTOIMinutes(row) {
  const toiSec = pickStat(row?.toi_sec, row?.time_on_ice_sec, 0);
  if (toiSec > 0) return toiSec / 60;

  return pickStat(row?.toi, row?.toi_min, row?.time_on_ice, 0);
}

function getAverageTOIMinutes(row) {
  const gp = Math.max(1, safeInt(firstPresent(row?.gp, row?.games_played, row?.games), 0));
  const totalMinutes = getTotalTOIMinutes(row);

  if (totalMinutes <= 0) return 0;

  /*
    If TOI is obviously season-total TOI, divide by GP.
    If backend already sends average TOI, keep it.
    Example:
    461.2 over 31 GP = 14:53
    18.4 already looks like ATOI, so keep it.
  */
  if (totalMinutes > 35 && gp > 0) {
    return totalMinutes / gp;
  }

  return totalMinutes;
}

function formatClockFromMinutes(minutesValue) {
  const minutes = safe(minutesValue, 0);
  if (minutes <= 0) return "—";

  const wholeMinutes = Math.floor(minutes);
  const seconds = Math.round((minutes - wholeMinutes) * 60);

  if (seconds >= 60) {
    return `${wholeMinutes + 1}:00`;
  }

  return `${wholeMinutes}:${String(seconds).padStart(2, "0")}`;
}

function formatTOI(row) {
  return formatClockFromMinutes(getTotalTOIMinutes(row));
}

function formatSmallTOI(row) {
  return formatClockFromMinutes(getAverageTOIMinutes(row));
}

function formatTOISplit(secondsValue, gpValue) {
  const seconds = safe(secondsValue, 0);
  const gp = Math.max(1, safe(gpValue, 0));

  if (seconds <= 0) return "—";

  const avgMinutes = seconds / 60 / gp;
  return formatClockFromMinutes(avgMinutes);
}

function compactNumber(value) {
  const n = safe(value, 0);
  if (Math.abs(n) >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(Math.round(n));
}

function trendTone(value, goodThreshold = 0, badThreshold = 0) {
  const n = safe(value, 0);
  if (n > goodThreshold) return "good";
  if (n < badThreshold) return "bad";
  return "neutral";
}

function rankRows(rows, key, direction = "desc") {
  return [...(rows || [])]
    .sort((a, b) => {
      const av = safe(a?.[key], 0);
      const bv = safe(b?.[key], 0);
      return direction === "asc" ? av - bv : bv - av;
    })
    .map((row, index) => ({
      ...row,
      _rank: index + 1,
    }));
}

function rowsByBackendRank(rows, rankKey, limit = 12) {
  return [...(rows || [])]
    .filter((row) => safeInt(row?.[rankKey], 0) > 0)
    .sort((a, b) => safeInt(a?.[rankKey], 9999) - safeInt(b?.[rankKey], 9999))
    .slice(0, limit)
    .map((row) => ({
      ...row,
      _rank: safeInt(row?.[rankKey], 0),
    }));
}

function hasRealNumber(value) {
  // Number(null) === 0, so null must be rejected explicitly or WAR/null stats
  // render as "0.00" instead of "—".
  if (value === null || value === undefined || value === "") return false;
  const n = Number(value);
  return Number.isFinite(n);
}

function hasRealPct(value) {
  if (value === null || value === undefined || value === "") return false;
  const n = Number(value);
  return Number.isFinite(n);
}

function fmtMaybePct(value, digits = 1) {
  if (!hasRealPct(value)) return "—";
  return fmtPct(value, digits);
}

function fmtMaybeDecimal(value, digits = 3) {
  if (!hasRealNumber(value)) return "—";
  return fmtDecimal(value, digits);
}

function fmtMaybeOne(value) {
  if (!hasRealNumber(value)) return "—";
  return fmtOne(value);
}

function fmtMaybeTwo(value) {
  if (!hasRealNumber(value)) return "—";
  return fmtTwo(value);
}

function fmtMaybeNumber(value) {
  if (!hasRealNumber(value)) return "—";
  return fmtZero(value);
}

function formatLeagueRank(rank, total = 32) {
  const n = safeInt(rank, 0);
  if (!n) return "League rank unavailable";
  return `League #${n} of ${total || 32}`;
}

function getTeamKey(team) {
  return String(
    firstPresent(
      team?.team_id,
      team?.id,
      team?.abbrev,
      team?.abbr,
      team?.name,
      ""
    )
  ).toLowerCase();
}

function rankTeamRows(teams, key, direction = "desc") {
  const rows = [...(teams || [])].filter((team) => hasRealNumber(team?.[key]));

  rows.sort((a, b) => {
    const av = safe(a?.[key], 0);
    const bv = safe(b?.[key], 0);
    return direction === "asc" ? av - bv : bv - av;
  });

  return rows.map((team, index) => ({
    ...team,
    [`${key}_league_rank`]: index + 1,
  }));
}

function findTeamRank(teams, targetTeam, key, direction = "desc") {
  const ranked = rankTeamRows(teams, key, direction);
  const targetKey = getTeamKey(targetTeam);

  const found = ranked.find((team) => getTeamKey(team) === targetKey);

  if (!found) return 0;

  return safeInt(found?.[`${key}_league_rank`], 0);
}

function deriveTeamAnalyticsFromPlayerLedger(team) {
  const t = team || {};
  return {
    ...t,
    gf: t.gf ?? t.goals_for,
    goals_for: t.goals_for ?? t.gf,
    ga: t.ga ?? t.goals_against,
    goals_against: t.goals_against ?? t.ga,
    sf: t.sf ?? t.shots_for,
    shots_for: t.shots_for ?? t.sf,
    sa: t.sa ?? t.shots_against,
    shots_against: t.shots_against ?? t.sa,
    cf_pct: hasRealPct(t.cf_pct) ? t.cf_pct : undefined,
    ff_pct: hasRealPct(t.ff_pct) ? t.ff_pct : undefined,
    xgf_pct: hasRealPct(t.xgf_pct) ? t.xgf_pct : undefined,
    sh_pct: hasRealPct(t.sh_pct) ? t.sh_pct : undefined,
    sv_pct: hasRealPct(t.sv_pct) ? t.sv_pct : undefined,
    pdo: normalizePdo(t.pdo),
    analytics_missing: {
      pp_pct: !(hasRealPct(t.pp_pct) && t.pp_pct > 0),
      pk_pct: !(hasRealPct(t.pk_pct) && t.pk_pct > 0),
      cf_pct: !hasRealPct(t.cf_pct),
      xgf_pct: !hasRealPct(t.xgf_pct),
      pdo: !hasRealNumber(normalizePdo(t.pdo)),
    },
  };
}

function attachTeamLeagueRanks(team, teams) {
  const total = teams?.length || 32;

  return {
    ...team,
    league_rank:
      safeInt(team.league_rank, 0) ||
      findTeamRank(teams, team, "points", "desc"),

    gf_league_rank: findTeamRank(teams, team, "gf", "desc"),
    ga_league_rank: findTeamRank(teams, team, "ga", "asc"),
    goal_diff_league_rank: findTeamRank(teams, team, "goal_diff", "desc"),
    pp_pct_league_rank: findTeamRank(teams, team, "pp_pct", "desc"),
    pk_pct_league_rank: findTeamRank(teams, team, "pk_pct", "desc"),
    pdo_league_rank: findTeamRank(teams, team, "pdo", "desc"),
    cf_pct_league_rank: findTeamRank(teams, team, "cf_pct", "desc"),
    xgf_pct_league_rank: findTeamRank(teams, team, "xgf_pct", "desc"),
    total_league_teams: total,
  };
}
function sortRows(rows, key, direction = "desc") {
  const dir = direction === "asc" ? 1 : -1;

  return [...(rows || [])].sort((a, b) => {
    const av = a?.[key];
    const bv = b?.[key];

    const an = Number(av);
    const bn = Number(bv);

    const aIsNumber = Number.isFinite(an);
    const bIsNumber = Number.isFinite(bn);

    if (aIsNumber && bIsNumber) {
      return (an - bn) * dir;
    }

    const as = safeString(av, "").toLowerCase();
    const bs = safeString(bv, "").toLowerCase();

    if (as < bs) return -1 * dir;
    if (as > bs) return 1 * dir;
    return 0;
  });
}

function topBy(rows, fn) {
  const arr = Array.isArray(rows) ? rows : [];
  if (!arr.length) return null;
  return [...arr].sort((a, b) => safe(fn(b), 0) - safe(fn(a), 0))[0] || null;
}

function filterRows(rows, search) {
  const q = safeString(search, "").trim().toLowerCase();
  if (!q) return rows || [];

  return (rows || []).filter((row) => {
    const haystack = [
      row?.name,
      row?.player_name,
      row?.team,
      row?.team_id,
      row?.position,
      row?.pos,
      row?.role_label,
      row?.analytics_archetype,
      row?.archetype,
      row?.player_type,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(q);
  });
}


/* =========================================================
   PAYLOAD SHAPE HELPERS
========================================================= */

function getStatsPayload(franchiseState) {
  const state = franchiseState || {};

  const statsCentral = state.stats_central || {};
  const analytics = state.analytics || {};
  const playerAnalytics = state.player_analytics || {};
  const fullAnalytics = state.full_analytics || {};
  const teamAnalytics = state.team_analytics || {};

  return {
    ...analytics,
    ...playerAnalytics,
    ...fullAnalytics,
    ...teamAnalytics,
    ...statsCentral,
  };
}

function collectRows(...groups) {
  const out = [];

  for (const group of groups) {
    if (Array.isArray(group)) {
      for (const row of group) {
        if (row && typeof row === "object") out.push(row);
      }
    }
  }

  return out;
}

function uniqueById(rows, fallbackPrefix = "row") {
  const seen = new Set();
  const out = [];

  (rows || []).forEach((row, index) => {
    const id = pickString(
      row?.player_id,
      row?.id,
      row?.pid,
      row?.team_id,
      row?.abbrev,
      `${fallbackPrefix}_${index}`
    );

    if (seen.has(id)) return;
    seen.add(id);
    out.push(row);
  });

  return out;
}

function normalizeGameRow(game, index = 0) {
  const homeId = pickString(game?.home_id, game?.home_team_id, game?.home, game?.home_team, "HOME");
  const awayId = pickString(game?.away_id, game?.away_team_id, game?.away, game?.away_team, "AWAY");

  return {
    ...game,
    game_id: pickString(game?.game_id, game?.id, `${game?.day || "d"}_${homeId}_${awayId}_${index}`),
    day: safeInt(firstPresent(game?.day, game?.calendar_day, game?.date), 0),
    calendar_iso: pickString(game?.calendar_iso, game?.iso, game?.date_iso, ""),
    home_id: homeId,
    away_id: awayId,
    home_name: pickString(game?.home_name, game?.home_team_name, game?.home_team, homeId),
    away_name: pickString(game?.away_name, game?.away_team_name, game?.away_team, awayId),
    home_goals: safeInt(game?.home_goals, 0),
    away_goals: safeInt(game?.away_goals, 0),
    home_shots: safeInt(game?.home_shots, 0),
    away_shots: safeInt(game?.away_shots, 0),
    home_xg: safe(game?.home_xg, 0),
    away_xg: safe(game?.away_xg, 0),
    overtime: Boolean(game?.overtime),
    shootout: Boolean(game?.shootout),
  };
}


/* =========================================================
   SKATER NORMALIZATION
========================================================= */

function normalizeSkater(row, index, teamId) {
  const gp = safeInt(firstPresent(row?.gp, row?.games_played, row?.games), 0);

  const g = safeInt(firstPresent(row?.g, row?.goals), 0);
  const a = safeInt(firstPresent(row?.a, row?.assists), 0);

  /*
    The backend analytics file is supposed to make:
    points = goals + assists.

    The frontend still calculates g + a as a safety fallback,
    but does not create random/scaled scoring.
  */
  const ptsBackend = firstPresent(row?.pts, row?.points);
  const pts = ptsBackend !== undefined ? safeInt(ptsBackend, g + a) : g + a;

  const sog = safeInt(firstPresent(row?.sog, row?.shots, row?.shots_on_goal), 0);
  const missedShots = safeInt(firstPresent(row?.missed_shots, row?.miss), 0);
  const blockedAttemptsFor = safeInt(
    firstPresent(row?.blocked_attempts_for, row?.blocked_shot_attempts_for, row?.bsf),
    0
  );

  const totalShotAttempts = safeInt(
    firstPresent(row?.total_shot_attempts, row?.total_shots, row?.shot_attempts),
    sog + missedShots + blockedAttemptsFor
  );

  const pim = safeInt(firstPresent(row?.pim, row?.pims, row?.penalty_minutes), 0);
  const hit = safeInt(firstPresent(row?.hit, row?.hits), 0);
  const blk = safeInt(firstPresent(row?.blk, row?.blocks, row?.blocked_shots), 0);

  const takeaways = safeInt(firstPresent(row?.tak, row?.takeaways, row?.tk), 0);
  const giveaways = safeInt(firstPresent(row?.giv, row?.giveaways, row?.gv), 0);

  const ppg = safeInt(firstPresent(row?.ppg, row?.power_play_goals), 0);
  const ppa = safeInt(firstPresent(row?.ppa, row?.power_play_assists), 0);
  const shg = safeInt(firstPresent(row?.shg, row?.short_handed_goals), 0);
  const sha = safeInt(firstPresent(row?.sha, row?.short_handed_assists), 0);

  const gwg = safeInt(firstPresent(row?.gwg, row?.game_winning_goals), 0);
  const otg = safeInt(firstPresent(row?.otg, row?.overtime_goals), 0);

  const fow = safeInt(firstPresent(row?.fow, row?.faceoff_wins), 0);
  const fol = safeInt(firstPresent(row?.fol, row?.faceoff_losses), 0);

  const toiSec = pickStat(row?.toi_sec, row?.time_on_ice_sec, 0);
  const toiMinFromSec = toiSec > 0 ? toiSec / 60 : 0;
  const toiMin = pickStat(row?.toi, row?.toi_min, row?.time_on_ice, toiMinFromSec, 0);

  const evToiSec = pickStat(row?.ev_toi_sec, row?.even_strength_toi_sec, 0);
  const ppToiSec = pickStat(row?.pp_toi_sec, row?.power_play_toi_sec, 0);
  const pkToiSec = pickStat(row?.pk_toi_sec, row?.penalty_kill_toi_sec, 0);

  const cf = pickStat(row?.cf, row?.corsi_for, row?.shot_attempts_for, 0);
  const ca = pickStat(row?.ca, row?.corsi_against, row?.shot_attempts_against, 0);

  const ff = pickStat(row?.ff, row?.fenwick_for, 0);
  const fa = pickStat(row?.fa, row?.fenwick_against, 0);

  const xgf = pickStat(row?.xgf, row?.expected_goals_for, row?.on_ice_xgf, 0);
  const xga = pickStat(row?.xga, row?.expected_goals_against, row?.on_ice_xga, 0);

  const ixg = pickStat(
    row?.ixg,
    row?.individual_xg,
    row?.individual_expected_goals,
    row?.xg,
    0
  );

  const xa = pickStat(row?.xa, row?.expected_assists, 0);

  const gfOn = pickStat(row?.gf_on, row?.on_ice_gf, row?.goals_for_on_ice, 0);
  const gaOn = pickStat(row?.ga_on, row?.on_ice_ga, row?.goals_against_on_ice, 0);

  const onIceShotsFor = pickStat(row?.on_ice_shots_for, row?.shots_for_on_ice, row?.sf_on, 0);
  const onIceShotsAgainst = pickStat(
    row?.on_ice_shots_against,
    row?.shots_against_on_ice,
    row?.sa_on,
    0
  );

  const scf = pickStat(row?.scf, row?.scoring_chances_for, 0);
  const sca = pickStat(row?.sca, row?.scoring_chances_against, 0);
  const hdcf = pickStat(row?.hdcf, row?.high_danger_chances_for, 0);
  const hdca = pickStat(row?.hdca, row?.high_danger_chances_against, 0);

  const cfSample = cf + ca;
  const cfPctRaw = firstPresent(row?.cf_pct, row?.corsi_pct, row?.corsi_percentage, row?.corsi_for_pct, row?.cf_percentage);
  const cfPct =
    cf > 0 && ca > 0
      ? cfPctRaw !== undefined
        ? normalizePct(cfPctRaw)
        : pct(cf, cfSample)
      : undefined;

  const ffPct =
    ff > 0 && fa > 0
      ? firstPresent(row?.ff_pct, row?.fenwick_pct, row?.fenwick_percentage) !== undefined
        ? normalizePct(firstPresent(row?.ff_pct, row?.fenwick_pct, row?.fenwick_percentage))
        : pct(ff, ff + fa)
      : undefined;

  const xgfPctGp = safeInt(firstPresent(row?.xgf_pct_gp), 0);
  const xgfPctSum = pickStat(row?.xgf_pct_sum, 0);
  const xgfSample = xgf + xga;
  const xgfPctRaw = firstPresent(row?.xgf_pct, row?.expected_goals_pct, row?.expected_goals_for_pct);
  const xgfPct =
    xgf > 0 && xga > 0
      ? xgfPctRaw !== undefined
        ? normalizePct(xgfPctRaw)
        : pct(xgf, xgfSample)
      : xgfPctGp > 0
        ? xgfPctSum / xgfPctGp
        : undefined;

  const gfPct =
    gfOn > 0 && gaOn > 0
      ? firstPresent(row?.gf_pct, row?.goals_for_pct) !== undefined
        ? normalizePct(firstPresent(row?.gf_pct, row?.goals_for_pct))
        : pct(gfOn, gfOn + gaOn)
      : undefined;

  const shootingPct =
    firstPresent(row?.shooting_pct, row?.sh_pct) !== undefined
      ? normalizePct(firstPresent(row?.shooting_pct, row?.sh_pct))
      : pct(g, sog);

  const faceoffPct =
    firstPresent(row?.faceoff_pct, row?.fo_pct) !== undefined
      ? normalizePct(firstPresent(row?.faceoff_pct, row?.fo_pct))
      : pct(fow, fow + fol);

  const pPerGpRaw = firstPresent(row?.points_per_game, row?.pts_per_game);
  const gPerGpRaw = firstPresent(row?.goals_per_game);
  const aPerGpRaw = firstPresent(row?.assists_per_game);
  const pPerGp = hasRealNumber(pPerGpRaw) ? Number(pPerGpRaw) : perGame(pts, gp);
  const gPerGp = hasRealNumber(gPerGpRaw) ? Number(gPerGpRaw) : perGame(g, gp);
  const aPerGp = hasRealNumber(aPerGpRaw) ? Number(aPerGpRaw) : perGame(a, gp);

  const pointsPer60Raw = firstPresent(row?.points_per_60, row?.pts_per_60, row?.p60);
  const goalsPer60Raw = firstPresent(row?.goals_per_60, row?.g_per_60);
  const assistsPer60Raw = firstPresent(row?.assists_per_60, row?.a_per_60);
  const pointsPer60 = hasRealNumber(pointsPer60Raw) ? Number(pointsPer60Raw) : per60(pts, toiMin);
  const goalsPer60 = hasRealNumber(goalsPer60Raw) ? Number(goalsPer60Raw) : per60(g, toiMin);
  const assistsPer60 = hasRealNumber(assistsPer60Raw) ? Number(assistsPer60Raw) : per60(a, toiMin);

  const finishing = hasRealNumber(row?.finishing) ? Number(row.finishing) : null;
  const shotQuality = hasRealNumber(row?.shot_quality) ? Number(row.shot_quality) : (hasRealNumber(row?.xg_per_shot) ? Number(row.xg_per_shot) : null);

  const offensiveImpact = pickStat(row?.offensive_impact, row?.offense_score, 0);
  const defensiveImpact = pickStat(row?.defensive_impact, row?.defense_score, 0);
  const specialTeamsImpact = pickStat(row?.special_teams_impact, 0);
  const transitionImpact = pickStat(row?.transition_impact, 0);
  const clutchScore = pickStat(row?.clutch_score, row?.clutch_impact, 0);

  const analyticsRatingRaw = hasRealNumber(row?.analytics_rating)
    ? Number(row.analytics_rating)
    : (hasRealNumber(row?.impact_score) ? Number(row.impact_score) : null);
  const analyticsRating = analyticsRatingRaw !== null && analyticsRatingRaw > 0 ? analyticsRatingRaw : null;
  const warRaw = firstPresent(row?.war, row?.total_impact, row?.watr, row?.WATR);
  // Always show computed WAR when present. war_valid is a sample-size flag for
  // awards/qualified leaderboards — hiding it early-season made every row look like 0.00
  // (especially once null was mis-formatted as 0.00).
  const war = hasRealNumber(warRaw) ? Number(warRaw) : null;
  const warValid = row?.war_valid !== false;

  const roleLabel = pickString(
    row?.role_label,
    row?.role,
    row?.deployment_role,
    row?.line_role,
    "Regular"
  );

  const archetype = pickString(
    row?.analytics_archetype,
    row?.player_type,
    row?.playstyle,
    row?.archetype,
    roleLabel
  );

  const position = normalizePosition(firstPresent(row?.position, row?.pos, "F"));

  return {
    ...row,

    _kind: "skater",
    _index: index,

    player_id: playerId(row, index),
    id: playerId(row, index),
    name: playerName(row),
    player_name: playerName(row),

    team_id: pickString(row?.team_id, row?.team, teamId),
    team: pickString(row?.team, row?.team_id, teamId),
    team_name: pickString(row?.team_name, row?.team_full_name),
    team_abbrev: pickString(row?.team_abbrev, row?.team_abbr, row?.abbrev),

    position,
    pos: position,

    gp,
    games_played: gp,

    g,
    goals: g,
    a,
    assists: a,
    pts,
    points: pts,

    primary_assists:
      firstPresent(
        row?.primary_assists,
        row?.primary_a,
        row?.a1
      ) !== undefined
        ? safeInt(
            firstPresent(
              row?.primary_assists,
              row?.primary_a,
              row?.a1
            ),
            0
          )
        : null,
    secondary_assists:
      firstPresent(
        row?.secondary_assists,
        row?.secondary_a,
        row?.a2
      ) !== undefined
        ? safeInt(
            firstPresent(
              row?.secondary_assists,
              row?.secondary_a,
              row?.a2
            ),
            0
          )
        : null,
    primary_points:
      firstPresent(row?.primary_points) !== undefined
        ? safeInt(row?.primary_points, 0)
        : firstPresent(
              row?.primary_assists,
              row?.primary_a,
              row?.a1
            ) !== undefined
          ? g +
            safeInt(
              firstPresent(
                row?.primary_assists,
                row?.primary_a,
                row?.a1
              ),
              0
            )
          : null,

    sog,
    shots: sog,
    shots_on_goal: sog,
    missed_shots: missedShots,
    blocked_attempts_for: blockedAttemptsFor,
    total_shot_attempts: totalShotAttempts,
    shooting_pct: shootingPct,
    sh_pct: shootingPct,

    pim,
    pims: pim,
    penalty_minutes: pim,

    hit,
    hits: hit,

    blk,
    blocks: blk,
    blocked_shots: blk,

    takeaways,
    tak: takeaways,
    giveaways,
    giv: giveaways,

    ppg,
    power_play_goals: ppg,
    ppa,
    power_play_assists: ppa,
    pp_points: safeInt(firstPresent(row?.pp_points), ppg + ppa),

    shg,
    short_handed_goals: shg,
    sha,
    short_handed_assists: sha,
    sh_points: safeInt(firstPresent(row?.sh_points), shg + sha),

    gwg,
    game_winning_goals: gwg,
    otg,
    overtime_goals: otg,

    fow,
    faceoff_wins: fow,
    fol,
    faceoff_losses: fol,
    faceoff_pct: faceoffPct,
    fo_pct: faceoffPct,

    plus_minus: safeInt(firstPresent(row?.plus_minus, row?.pm, row?.["+/-"]), 0),

    toi_sec: toiSec,
    toi: toiMin,
    toi_min: toiMin,
    ev_toi_sec: evToiSec,
    pp_toi_sec: ppToiSec,
    pk_toi_sec: pkToiSec,

    cf,
    corsi_for: cf,
    ca,
    corsi_against: ca,
    cf_pct: cfPct,
    corsi_pct: cfPct,

    ff,
    fenwick_for: ff,
    fa,
    fenwick_against: fa,
    ff_pct: ffPct,
    fenwick_pct: ffPct,

    xgf,
    expected_goals_for: xgf,
    xga,
    expected_goals_against: xga,
    xgf_pct: xgfPct,

    ixg,
    individual_xg: ixg,
    xa,
    expected_assists: xa,

    gf_on: gfOn,
    on_ice_gf: gfOn,
    ga_on: gaOn,
    on_ice_ga: gaOn,
    gf_pct: gfPct,

    on_ice_shots_for: onIceShotsFor,
    on_ice_shots_against: onIceShotsAgainst,

    scf,
    scoring_chances_for: scf,
    sca,
    scoring_chances_against: sca,
    hdcf,
    high_danger_chances_for: hdcf,
    hdca,
    high_danger_chances_against: hdca,

    points_per_game: pPerGp,
    pts_per_game: pPerGp,
    goals_per_game: gPerGp,
    assists_per_game: aPerGp,

    points_per_60: pointsPer60,
    goals_per_60: goalsPer60,
    assists_per_60: assistsPer60,

    shot_quality: shotQuality,
    finishing,

    offensive_impact: offensiveImpact,
    defensive_impact: defensiveImpact,
    special_teams_impact: specialTeamsImpact,
    transition_impact: transitionImpact,
    clutch_score: clutchScore,

    analytics_rating: analyticsRating,
    impact_score: analyticsRating,
    war,
    war_valid: warValid,
    watr: war,
    WATR: war,
    total_impact: war,

    role_label: roleLabel,
    analytics_archetype: archetype,

    age: safeInt(row?.age, 0),
    rookie: Boolean(row?.rookie || row?.is_rookie),
    is_rookie: Boolean(row?.rookie || row?.is_rookie),

    cap_hit: pickStat(row?.cap_hit, row?.cap_hit_millions, row?.salary, 0),
    overall: 0,
    ovr: 0,
    potential: pickStat(row?.potential, 0),
  };
}


/* =========================================================
   GOALIE NORMALIZATION
========================================================= */

function normalizeGoalie(row, index, teamId) {
  const gp = safeInt(firstPresent(row?.gp, row?.games_played, row?.games), 0);
  const starts = safeInt(firstPresent(row?.starts, row?.gs), gp);

  const wins = safeInt(firstPresent(row?.wins, row?.w), 0);
  const losses = safeInt(firstPresent(row?.losses, row?.l), 0);
  const otl = safeInt(firstPresent(row?.otl, row?.ot), 0);

  const sa = safeInt(firstPresent(row?.sa, row?.shots_against), 0);
  const ga = safeInt(firstPresent(row?.ga, row?.goals_against), 0);
  const savesRaw = firstPresent(row?.saves);
  const saves = savesRaw !== undefined ? safeInt(savesRaw, 0) : Math.max(0, sa - ga);

  const toiSec = pickStat(row?.toi_sec, row?.time_on_ice_sec, 0);
  const toiMinFromSec = toiSec > 0 ? toiSec / 60 : 0;
  const toiMin = pickStat(row?.toi, row?.toi_min, row?.time_on_ice, toiMinFromSec, gp * 60);

  const svPct =
    firstPresent(row?.sv_pct, row?.save_pct, row?.sv) !== undefined
      ? normalizePct(firstPresent(row?.sv_pct, row?.save_pct, row?.sv))
      : pct(saves, sa);

  const gaa =
    firstPresent(row?.gaa, row?.goals_against_average) !== undefined
      ? pickStat(row?.gaa, row?.goals_against_average, 0)
      : toiMin > 0
        ? (ga * 60) / toiMin
        : 0;

  const xga = hasRealNumber(row?.xga)
    ? Number(row.xga)
    : hasRealNumber(row?.goalie_xga)
      ? Number(row.goalie_xga)
      : hasRealNumber(row?.expected_goals_against)
        ? Number(row.expected_goals_against)
        : null;
  const gsaa = pickStat(row?.gsaa, row?.goals_saved_above_average, 0);
  const gsaxValid = row?.gsax_valid === true || (xga != null && xga > 0);
  const gsaxFromRow = hasRealNumber(row?.gsax) ? Number(row.gsax) : null;
  const gsax =
    gsaxValid && gsaxFromRow != null
      ? gsaxFromRow
      : gsaxValid && xga != null
        ? xga - ga
        : hasRealNumber(gsaa)
          ? Number(gsaa)
          : null;

  const qualityStarts = safeInt(firstPresent(row?.quality_starts, row?.qs), 0);
  const badStarts = safeInt(firstPresent(row?.bad_starts), 0);

  const qualityStartPct =
    firstPresent(row?.quality_start_pct, row?.qs_pct) !== undefined
      ? normalizePct(firstPresent(row?.quality_start_pct, row?.qs_pct))
      : pct(qualityStarts, starts);

  const hdSvPctRaw = firstPresent(row?.hd_sv_pct, row?.high_danger_save_pct);
  const mdSvPctRaw = firstPresent(row?.md_sv_pct, row?.medium_danger_save_pct);
  const ldSvPctRaw = firstPresent(row?.ld_sv_pct, row?.low_danger_save_pct);
  const hdSvPct = hdSvPctRaw !== undefined ? normalizePct(hdSvPctRaw) : null;
  const mdSvPct = mdSvPctRaw !== undefined ? normalizePct(mdSvPctRaw) : null;
  const ldSvPct = ldSvPctRaw !== undefined ? normalizePct(ldSvPctRaw) : null;

  const analyticsRatingRaw = hasRealNumber(row?.analytics_rating)
    ? Number(row.analytics_rating)
    : (hasRealNumber(row?.goalie_rating) ? Number(row.goalie_rating) : (hasRealNumber(row?.impact_score) ? Number(row.impact_score) : null));
  const analyticsRating = analyticsRatingRaw !== null && analyticsRatingRaw > 0 ? analyticsRatingRaw : null;
  const warRaw = firstPresent(row?.war, row?.total_impact, row?.watr, row?.WATR);
  const war = hasRealNumber(warRaw) ? Number(warRaw) : null;

  return {
    ...row,

    _kind: "goalie",
    _index: index,

    player_id: playerId(row, index),
    id: playerId(row, index),
    name: playerName(row),
    player_name: playerName(row),

    team_id: pickString(row?.team_id, row?.team, teamId),
    team: pickString(row?.team, row?.team_id, teamId),
    team_name: pickString(row?.team_name, row?.team_full_name),
    team_abbrev: pickString(row?.team_abbrev, row?.team_abbr, row?.abbrev),

    position: "G",
    pos: "G",

    gp,
    games_played: gp,
    starts,

    wins,
    w: wins,
    losses,
    l: losses,
    otl,

    sa,
    shots_against: sa,
    ga,
    goals_against: ga,
    saves,

    sv_pct: svPct,
    save_pct: svPct,
    gaa,

    so: safeInt(firstPresent(row?.so, row?.shutouts), 0),
    shutouts: safeInt(firstPresent(row?.so, row?.shutouts), 0),

    toi_sec: toiSec,
    toi: toiMin,
    toi_min: toiMin,

    xga,
    expected_goals_against: xga,
    gsax,
    goals_saved_above_expected: gsax,
    gsaa,

    quality_starts: qualityStarts,
    quality_start_pct: qualityStartPct,
    bad_starts: badStarts,

    hd_sv_pct: hdSvPct,
    md_sv_pct: mdSvPct,
    ld_sv_pct: ldSvPct,

    rebound_control_pct: normalizePct(row?.rebound_control_pct, 0),

    analytics_rating: analyticsRating,
    impact_score: analyticsRating,
    war,
    watr: war,
    WATR: war,
    total_impact: war,

    role_label: pickString(row?.role_label, row?.role, "Goalie"),
    analytics_archetype: pickString(row?.analytics_archetype, row?.goalie_style, row?.archetype, "Goalie"),

    age: safeInt(row?.age, 0),
    rookie: Boolean(row?.rookie || row?.is_rookie),
    is_rookie: Boolean(row?.rookie || row?.is_rookie),

    cap_hit: pickStat(row?.cap_hit, row?.cap_hit_millions, row?.salary, 0),
    overall: 0,
    ovr: 0,
    potential: pickStat(row?.potential, 0),
  };
}


/* =========================================================
   TEAM NORMALIZATION
========================================================= */

function normalizeTeam(row, index = 0) {
  const teamId = pickString(row?.team_id, row?.id, row?.abbrev, row?.abbr, row?.name, `T${index + 1}`);

  const gp = safeInt(firstPresent(row?.gp, row?.games_played, row?.games), 0);
  const wins = safeInt(firstPresent(row?.wins, row?.w), 0);
  const losses = safeInt(firstPresent(row?.losses, row?.l), 0);
  const otl = safeInt(firstPresent(row?.otl, row?.ot), 0);
  const points = safeInt(firstPresent(row?.points, row?.pts), wins * 2 + otl);

  const gf = safeInt(firstPresent(row?.gf, row?.goals_for), 0);
  const ga = safeInt(firstPresent(row?.ga, row?.goals_against), 0);

  const sf = safeInt(firstPresent(row?.sf, row?.shots_for), 0);
  const sa = safeInt(firstPresent(row?.sa, row?.shots_against), 0);

  const ppg = safeInt(firstPresent(row?.ppg, row?.power_play_goals), 0);
  const ppo = safeInt(firstPresent(row?.ppo, row?.power_play_opportunities), 0);
  const ppga = safeInt(firstPresent(row?.ppga, row?.power_play_goals_against), 0);
  const oppPpo = safeInt(
    firstPresent(row?.opp_ppo, row?.opp_power_play_opportunities, row?.times_shorthanded),
    0
  );

  const cf = pickStat(row?.cf, row?.corsi_for, row?.shot_attempts_for, 0);
  const ca = pickStat(row?.ca, row?.corsi_against, row?.shot_attempts_against, 0);
  const ff = pickStat(row?.ff, row?.fenwick_for, row?.unblocked_attempts_for, 0);
  const fa = pickStat(row?.fa, row?.fenwick_against, row?.unblocked_attempts_against, 0);
  const xgf = pickStat(row?.xgf, row?.expected_goals_for, 0);
  const xga = pickStat(row?.xga, row?.expected_goals_against, 0);

  const shPct =
    firstPresent(row?.sh_pct, row?.shooting_pct) !== undefined
      ? normalizePct(firstPresent(row?.sh_pct, row?.shooting_pct))
      : sf > 0
        ? pct(gf, sf)
        : undefined;

  const svPct =
    firstPresent(row?.sv_pct, row?.save_pct) !== undefined
      ? normalizePct(firstPresent(row?.sv_pct, row?.save_pct))
      : sa > 0
        ? pct(sa - ga, sa)
        : undefined;

  const ppPct =
    firstPresent(row?.pp_pct, row?.power_play_pct) !== undefined
      ? normalizePct(firstPresent(row?.pp_pct, row?.power_play_pct))
      : ppo > 0
        ? pct(ppg, ppo)
        : undefined;

  const pkPctRaw = firstPresent(row?.pk_pct, row?.penalty_kill_pct);
  const pkPct =
    oppPpo > 0
      ? pkPctRaw !== undefined
        ? normalizePct(pkPctRaw)
        : 1 - pct(ppga, oppPpo)
      : undefined;

  const cfSample = cf + ca;
  // Require both sides — CF-only (light-sim artifact) must not become 100%.
  const cfPctRaw = firstPresent(row?.cf_pct, row?.corsi_pct, row?.corsi_for_pct, row?.cf_percentage);
  const cfPct =
    cf > 0 && ca > 0
      ? cfPctRaw !== undefined
        ? normalizePct(cfPctRaw)
        : pct(cf, cfSample)
      : undefined;

  const ffPct =
    ff > 0 && fa > 0
      ? firstPresent(row?.ff_pct, row?.fenwick_pct) !== undefined
        ? normalizePct(firstPresent(row?.ff_pct, row?.fenwick_pct))
        : pct(ff, ff + fa)
      : undefined;

  const xgfPctGp = safeInt(firstPresent(row?.xgf_pct_gp), 0);
  const xgfPctSum = pickStat(row?.xgf_pct_sum, 0);
  const xgfSample = xgf + xga;
  const xgfPctRaw = firstPresent(row?.xgf_pct, row?.expected_goals_pct, row?.expected_goals_for_pct);
  const xgfPct =
    xgf > 0 && xga > 0
      ? xgfPctRaw !== undefined
        ? normalizePct(xgfPctRaw)
        : pct(xgf, xgfSample)
      : xgfPctGp > 0
        ? xgfPctSum / xgfPctGp
        : undefined;

  // Zero xGF with no against sample is missing data, not a real 0.0 season.
  const xgfDisplay = xgf > 0 ? xgf : undefined;
  const xgaDisplay = xga > 0 ? xga : undefined;

  const pdo =
    firstPresent(row?.pdo) !== undefined && (row?.pdo_valid !== false || (hasRealNumber(shPct) && hasRealNumber(svPct)))
      ? normalizePdo(row?.pdo)
      : hasRealNumber(shPct) && hasRealNumber(svPct)
        ? (shPct + svPct) * 100
        : undefined;

  return {
    ...row,

    _kind: "team",
    _index: index,

    team_id: teamId,
    id: teamId,
    name: pickString(row?.name, row?.team_name, row?.full_name, teamId),
    abbrev: pickString(row?.abbrev, row?.abbr, row?.team_abbrev, row?.team_abbr),
    team_abbrev: pickString(row?.team_abbrev, row?.team_abbr, row?.abbrev, row?.abbr),

    gp,
    games_played: gp,

    win_pct: gp > 0 ? wins / gp : undefined,
    points_pct: gp > 0 ? points / (gp * 2) : undefined,

    gf_per_game: gp > 0 ? gf / gp : undefined,
    ga_per_game: gp > 0 ? ga / gp : undefined,
    sf_per_game: gp > 0 ? sf / gp : undefined,
    sa_per_game: gp > 0 ? sa / gp : undefined,

    wins,
    w: wins,
    losses,
    l: losses,
    otl,
    points,
    pts: points,

    gf,
    goals_for: gf,
    ga,
    goals_against: ga,
    goal_diff: safeInt(firstPresent(row?.goal_diff, row?.gd), gf - ga),

    sf,
    shots_for: sf,
    sa,
    shots_against: sa,

    ppg,
    power_play_goals: ppg,
    ppo,
    power_play_opportunities: ppo,
    ppga,
    opp_ppo: oppPpo,

    pp_pct: ppPct,
    power_play_pct: ppPct,
    pk_pct: pkPct,
    penalty_kill_pct: pkPct,

    cf,
    ca,
    ff,
    fa,
    xgf: xgfDisplay,
    xga: xgaDisplay,

    sh_pct: shPct,
    shooting_pct: shPct,
    sv_pct: svPct,
    save_pct: svPct,
    pdo,

    cf_pct: cfPct,
    corsi_pct: cfPct,
    ff_pct: ffPct,
    fenwick_pct: ffPct,
    xgf_pct: xgfPct,

    analytics_rating: pickStat(row?.analytics_rating, row?.team_rating, row?.impact_score, 0),

    division_rank: safeInt(firstPresent(row?.division_rank, row?.div_rank), 0),
    conference_rank: safeInt(firstPresent(row?.conference_rank, row?.conf_rank), 0),
    league_rank: safeInt(firstPresent(row?.league_rank), 0),

    division: pickString(row?.division, ""),
    conference: pickString(row?.conference, ""),
  };
}
/* =========================================================
   FRONTEND IMPACT FALLBACKS
========================================================= */

function calculateFrontendImpactProxy(player) {
  const gp = Math.max(1, safe(player?.gp, 0));
  const pts = safe(player?.pts, 0);
  const toi = safe(player?.toi, 0);
  const cfPct = normalizePct(player?.cf_pct, 0);
  const xgfPct = normalizePct(player?.xgf_pct, 0);
  const gfPct = normalizePct(player?.gf_pct, 0);

  const ppg = pts / gp;
  const usage = toi > 0 ? clamp(toi / gp, 8, 28) : 15;

  const score =
    ppg * 22 +
    cfPct * 26 +
    xgfPct * 26 +
    gfPct * 16 +
    usage * 0.65;

  return roundTo(score, 2);
}

function calculateFrontendGoalieImpactProxy(goalie) {
  const svPct = normalizePct(goalie?.sv_pct, 0);
  const gaa = safe(goalie?.gaa, 0);
  const gsax = safe(goalie?.gsax, 0);
  const starts = Math.max(1, safe(goalie?.starts, goalie?.gp || 1));

  const saveComponent = (svPct - 0.88) * 480;
  const gaaComponent = (3.3 - gaa) * 12;
  const gsaxComponent = (gsax / starts) * 8;
  const workloadComponent = Math.min(12, starts * 0.18);

  return roundTo(saveComponent + gaaComponent + gsaxComponent + workloadComponent, 2);
}

function getPlayerImpactLabel(value) {
  const n = safe(value, 0);

  if (n >= 82) return "Franchise Driver";
  if (n >= 74) return "Elite Driver";
  if (n >= 66) return "Star Impact";
  if (n >= 56) return "Core Contributor";
  if (n >= 46) return "Middle Line Value";
  if (n >= 36) return "Depth Value";
  return "Replacement Level";
}

function getAnalyticsTone(value) {
  const n = safe(value, 0);

  if (n >= 74) return "elite";
  if (n >= 62) return "good";
  if (n >= 48) return "neutral";
  if (n >= 36) return "warn";
  return "bad";
}


/* =========================================================
   FULL STATS CENTRAL NORMALIZATION
========================================================= */

function normalizeStatsCentral(franchiseState) {
  const state = franchiseState || {};
  const payload = getStatsPayload(state);

  const teamInfo = state.team || state.user_team || payload.team_info || payload.user_team || {};
  const teamId = teamIdFromInfo(teamInfo);

  const rawPlayers = collectRows(
    payload.players,
    payload.skaters,
    payload.player_rows,
    payload.enriched_players,
    payload.league_leaders,
    payload.user_team_skaters,
    payload.my_skaters,
    payload.team_skaters
  );

  const rawGoalies = collectRows(
    payload.goalies,
    payload.goalie_rows,
    payload.enriched_goalies,
    payload.user_team_goalies,
    payload.my_goalies,
    payload.team_goalies
  );

  const mixedSkaters = rawPlayers.filter((p) => !isGoalieRow(p));
  const mixedGoalies = rawPlayers.filter((p) => isGoalieRow(p));

  const skaters = uniqueById(mixedSkaters, "skater").map((p, index) =>
    normalizeSkater(p, index, teamId)
  );

  const goalies = uniqueById([...rawGoalies, ...mixedGoalies], "goalie").map((p, index) =>
    normalizeGoalie(p, index, teamId)
  );

  const userSkaterRaw = collectRows(
    payload.user_team_skaters,
    payload.my_skaters,
    payload.team_skaters
  );

  const userGoalieRaw = collectRows(
    payload.user_team_goalies,
    payload.my_goalies,
    payload.team_goalies
  );

  const rosterRows = Array.isArray(state?.roster) ? state.roster : [];
  const rosterSkaterIds = new Set(
    rosterRows
      .filter(
        (row) =>
          normalizePosition(firstPresent(row?.position, row?.pos, "F")) !== "G"
      )
      .map((row) =>
        pickString(row?.player_id, row?.playerId, row?.id, row?.pid, "")
      )
      .filter(Boolean)
  );
  const rosterGoalieIds = new Set(
    rosterRows
      .filter(
        (row) =>
          normalizePosition(firstPresent(row?.position, row?.pos, "")) === "G"
      )
      .map((row) =>
        pickString(row?.player_id, row?.playerId, row?.id, row?.pid, "")
      )
      .filter(Boolean)
  );

  const userSkaterRowsByPayload = userSkaterRaw.length
    ? uniqueById(userSkaterRaw.filter((p) => !isGoalieRow(p)), "user_skater")
    : skaters.filter((p) => String(p.team_id) === String(teamId) || String(p.team) === String(teamId));
  const userGoalieRowsByPayload = userGoalieRaw.length
    ? uniqueById(userGoalieRaw, "user_goalie")
    : goalies.filter((p) => String(p.team_id) === String(teamId) || String(p.team) === String(teamId));

  const userSkatersMerged = uniqueById(
    [
      ...userSkaterRowsByPayload,
      ...skaters.filter((p) =>
        rosterSkaterIds.has(
          pickString(p?.player_id, p?.playerId, p?.id, p?.pid, "")
        )
      ),
    ],
    "user_skater_merged"
  );
  const userGoaliesMerged = uniqueById(
    [
      ...userGoalieRowsByPayload,
      ...goalies.filter((p) =>
        rosterGoalieIds.has(
          pickString(p?.player_id, p?.playerId, p?.id, p?.pid, "")
        )
      ),
    ],
    "user_goalie_merged"
  );

  const userSkaters = userSkatersMerged.map((p, index) =>
    normalizeSkater(p, index, teamId)
  );
  const userGoalies = userGoaliesMerged.map((p, index) =>
    normalizeGoalie(p, index, teamId)
  );

  const rawTeams = collectRows(
    payload.teams,
    payload.team_analytics,
    payload.league_team_stats,
    payload.league_teams
  );

  const teamsDirectory = collectRows(payload.teams_directory);
  let teamsNormalized = uniqueById([...rawTeams, ...teamsDirectory], "team").map((t, index) =>
    normalizeTeam(t, index)
  );

  if (!teamsNormalized.length) {
    const standingsRows = Array.isArray(state.standings) ? state.standings : [];
    teamsNormalized = standingsRows.map((t, index) => normalizeTeam(t, index));
  }
  const teams = attachLogosToTeamRows(teamsNormalized, state);

  const playerOverallLookup = buildPlayerOverallLookup(state);

  const skatersWithOverall = applyPlayerOverallLookup(
    skaters,
    playerOverallLookup,
    state
  );
  const goaliesWithOverall = applyPlayerOverallLookup(
    goalies,
    playerOverallLookup,
    state
  );
  const userSkatersWithOverall = applyPlayerOverallLookup(
    userSkaters,
    playerOverallLookup,
    state
  );
  const userGoaliesWithOverall = applyPlayerOverallLookup(
    userGoalies,
    playerOverallLookup,
    state
  );

  const skatersWithLogos = attachTeamLogosToRows(skatersWithOverall, teams, state);
  const goaliesWithLogos = attachTeamLogosToRows(goaliesWithOverall, teams, state);
  const userSkatersWithLogos = attachTeamLogosToRows(userSkatersWithOverall, teams, state);
  const userGoaliesWithLogos = attachTeamLogosToRows(userGoaliesWithOverall, teams, state);

  const teamRaw =
    payload.team ||
    payload.user_team_analytics ||
    payload.team_team_stats ||
    payload.team_stats ||
    teams.find((t) => String(t.team_id) === String(teamId)) ||
    {};

  const team = normalizeTeam(
    {
      ...teamRaw,
      team_id: teamId,
      name: teamNameFromInfo(teamInfo, teamRaw?.name || teamId),
    },
    0
  );

  const standingsRows = Array.isArray(state.standings) ? state.standings : [];
  const standing = standingsRows.find((row) => {
    const sid = pickString(row?.team_id, row?.id, row?.abbrev, row?.abbr, row?.name);
    return String(sid) === String(teamId);
  });

  if (standing) {
    team.wins = safeInt(firstPresent(standing.w, standing.wins, team.wins), team.wins);
    team.w = team.wins;

    team.losses = safeInt(firstPresent(standing.l, standing.losses, team.losses), team.losses);
    team.l = team.losses;

    team.otl = safeInt(firstPresent(standing.otl, team.otl), team.otl);

    team.points = safeInt(firstPresent(standing.pts, standing.points, team.points), team.points);
    team.pts = team.points;

    team.division_rank = safeInt(
      firstPresent(standing.division_rank, standing.div_rank, team.division_rank),
      team.division_rank
    );

    team.conference_rank = safeInt(
      firstPresent(standing.conference_rank, standing.conf_rank, team.conference_rank),
      team.conference_rank
    );

    team.league_rank = safeInt(
      firstPresent(standing.league_rank, team.league_rank),
      team.league_rank
    );
  }

  const derivedTeam = deriveTeamAnalyticsFromPlayerLedger(team);
  const rankedTeam = attachLogosToTeamRows(
    [attachTeamLeagueRanks(derivedTeam, teams.length ? teams : [derivedTeam])],
    state
  )[0];

  const calendarRaw = Array.isArray(payload.calendar)
    ? payload.calendar
    : Array.isArray(state.calendar)
      ? state.calendar
      : [];

  const calendar = calendarRaw.map((day, index) => ({
    ...day,
    day: safeInt(firstPresent(day?.day, day?.calendar_day, index), index),
    calendar_iso: pickString(day?.calendar_iso, day?.iso, day?.date, ""),
    segment: pickString(day?.segment, day?.season_segment, ""),
    games: Array.isArray(day?.games) ? day.games.map((g, gi) => normalizeGameRow(g, gi)) : [],
    events: Array.isArray(day?.events) ? day.events : [],
  }));

  const recentGamesRaw = Array.isArray(payload.recent_games)
    ? payload.recent_games
    : Array.isArray(payload.games)
      ? payload.games
      : Array.isArray(state.recent_games)
        ? state.recent_games
        : [];

  const recentGames = recentGamesRaw.map((g, index) => normalizeGameRow(g, index));

  const logs = collectRows(
    payload.analytics_log,
    payload.stat_log,
    payload.recent_stat_events,
    payload.news_events,
    state.timeline,
    state.notifications,
    state.storyline_events
  );

  const awardsWatch =
    payload.awards_watch ||
    state.awards_watch ||
    {};

  const meta =
    payload.meta ||
    payload.simulation_meta ||
    state.simulation_meta ||
    {};

  return {
    payload,
    teamId,
    teamInfo,
    team: rankedTeam,
    franchiseState: state,

    skaters: skatersWithLogos,
    goalies: goaliesWithLogos,
    userSkaters: userSkatersWithLogos,
    userGoalies: userGoaliesWithLogos,
    teams,

    calendar,
    recentGames,
    logs,
    awardsWatch,
    meta,

    hasCalendar: calendar.length > 0,
    hasRecentGames: recentGames.length > 0,
    hasSkaters: skaters.length > 0,
    hasGoalies: goalies.length > 0,
  };
}


/* =========================================================
   TAB CONFIG
   icons intentionally small
========================================================= */

const TABS = [
  { id: "team_stats", label: "TEAM STATS" },
  { id: "player_stats", label: "PLAYER STATS" },
  { id: "league_leaders", label: "LEAGUE LEADERS" },
];

function normalizeStatsMenu(tabId) {
  if (tabId === "team" || tabId === "overview") return "team_stats";
  if (["players", "goalies", "advanced", "compare"].includes(tabId)) return "player_stats";
  if (["leaders", "awards"].includes(tabId)) return "league_leaders";
  return ["team_stats", "player_stats", "league_leaders"].includes(tabId) ? tabId : "team_stats";
}



const PLAYER_SUBMENUS = [
  { id: "overview", label: "Overview" },
  { id: "skaters", label: "Skaters" },
  { id: "goalies", label: "Goalies" },
  { id: "analytics", label: "Analytics" },
  { id: "special_teams", label: "Special Teams" },
  { id: "trends", label: "Trends" },
  { id: "compare", label: "Compare" },
];

const LEAGUE_LEADER_VIEWS = [
  { id: "leaders", label: "Leaders" },
  { id: "awards", label: "Awards" },
];

const PLAYER_PAGE_SIZE = {
  team: 18,
  league: 16,
};

function fmtSavePct(value) {
  if (!hasRealPct(value)) return "—";
  return normalizePct(value, 0).toFixed(3).replace(/^0/, "");
}

function getPlayerTeamLabel(player) {
  return pickString(
    player?.team_abbrev,
    player?.team_abbr,
    player?.team_name,
    player?.team_id,
    player?.team,
    "—"
  );
}

function getPlayerOverall(player) {
  return getUniversalOverall(player);
}

function addRosterBrowserRowLookupEntry(lookup, row) {
  if (!row || typeof row !== "object") return;

  const id = pickString(
    row?.player_id,
    row?.playerId,
    row?.id,
    row?.pid
  );

  if (!id || lookup.has(id)) return;

  lookup.set(id, row);
}

function buildPlayerOverallLookup(franchiseState) {
  const lookup = new Map();
  const browser = franchiseState?.roster_browser;

  (browser?.organizations || []).forEach((organization) => {
    ["nhl", "ahl", "echl", "prospects"].forEach((poolKey) => {
      (organization?.[poolKey] || []).forEach((row) =>
        addRosterBrowserRowLookupEntry(lookup, row)
      );
    });
  });

  [
    browser?.free_agents,
    browser?.overseas_free_agents,
    franchiseState?.user_roster,
    franchiseState?.roster,
  ].forEach((rows) => {
    (rows || []).forEach((row) =>
      addRosterBrowserRowLookupEntry(lookup, row)
    );
  });

  collectRows(
    franchiseState?.players,
    franchiseState?.player_rows,
    franchiseState?.league_players
  ).forEach((row) => addRosterBrowserRowLookupEntry(lookup, row));

  return lookup;
}

function resolvePlayerDisplayOverall(
  row,
  franchiseState,
  rosterRowById,
  overallCache,
  index = 0
) {
  if (!row) return 0;

  const id = pickString(
    row?.player_id,
    row?.playerId,
    row?.id,
    row?.pid,
    playerId(row, index)
  );

  if (id && overallCache?.has(id)) {
    return overallCache.get(id);
  }

  /*
    Universal OVR comes from roster_browser / serialized player fields
    (effective_ovr / ovr / base_ovr). Season-stat rows often zero overall —
    never let those zeros overwrite a real roster OVR.
  */
  const rosterRow = id ? rosterRowById?.get(id) : null;
  const source = rosterRow || row;
  const overall = getUniversalOverall(source);

  if (id && overall > 0) {
    overallCache?.set(id, overall);
  }

  return overall;
}

function resolvePlayerOverallMeta(row, rosterRowById, index = 0) {
  const id = pickString(
    row?.player_id,
    row?.playerId,
    row?.id,
    row?.pid,
    playerId(row, index)
  );
  const rosterRow = id ? rosterRowById?.get(id) : null;
  const source = rosterRow || row || {};
  const ovr = getUniversalOverall(source);
  const base = getBaseOverall(source) || ovr;
  const drop = getOverallDrop(source);

  return {
    ovr,
    overall: ovr,
    base_ovr: base,
    effective_ovr: ovr,
    overall_drop: drop,
    ovr_modifiers: source?.ovr_modifiers || row?.ovr_modifiers || [],
  };
}

function applyPlayerOverallLookup(rows, rosterRowById, franchiseState) {
  if (!Array.isArray(rows) || !franchiseState) return rows;

  const overallCache = new Map();

  return rows.map((row, index) => {
    const overall = resolvePlayerDisplayOverall(
      row,
      franchiseState,
      rosterRowById,
      overallCache,
      index
    );

    if (overall <= 0) return row;

    const meta = resolvePlayerOverallMeta(row, rosterRowById, index);

    return {
      ...row,
      ...meta,
      overall,
      ovr: overall,
    };
  });
}

function playerMatchesScopeFilter(player, {
  search = "",
  position = "all",
  team = "all",
  minGp = 0,
  seasonScope = "all",
} = {}) {
  const query = safeString(search, "").trim().toLowerCase();
  const playerPosition = normalizePosition(firstPresent(player?.position, player?.pos, "F"));
  const playerTeam = normalizeTeamIdentity(
    firstPresent(
      player?.team_id,
      player?.team,
      player?.team_abbrev,
      player?.team_abbr,
      player?.team_name
    )
  );
  const rowScope = safeString(
    firstPresent(player?.stat_scope, player?.season_scope, player?.scope),
    ""
  ).toLowerCase();

  if (position !== "all") {
    if (position === "F" && !["C", "LW", "RW", "F"].includes(playerPosition)) {
      return false;
    }

    if (position !== "F" && playerPosition !== position) {
      return false;
    }
  }

  if (team !== "all" && playerTeam !== team) {
    return false;
  }

  if (safe(player?.gp, 0) < safe(minGp, 0)) {
    return false;
  }

  if (seasonScope !== "all" && rowScope && rowScope !== seasonScope) {
    return false;
  }

  if (!query) return true;

  return [
    player?.name,
    player?.player_name,
    getPlayerTeamLabel(player),
    player?.position,
    player?.role_label,
    player?.analytics_archetype,
  ]
    .join(" ")
    .toLowerCase()
    .includes(query);
}

function getPlayerPercentile(rows, player, key, direction = "desc") {
  const eligible = (rows || []).filter((row) => hasRealNumber(row?.[key]));
  if (!eligible.length || !hasRealNumber(player?.[key])) return null;

  const sorted = sortRows(eligible, key, direction);
  const index = sorted.findIndex(
    (row) => String(row.player_id || row.id) === String(player.player_id || player.id)
  );

  if (index < 0) return null;
  if (sorted.length === 1) return 100;

  return Math.round((1 - index / (sorted.length - 1)) * 100);
}

function getVisibleRange(page, pageSize, total) {
  if (!total) return { start: 0, end: 0 };

  const safePage = Math.max(1, page);
  const start = (safePage - 1) * pageSize + 1;
  const end = Math.min(total, safePage * pageSize);

  return { start, end };
}

function getPlayerInsightRows(players, type, limit = 5) {
  const rows = Array.isArray(players) ? players : [];

  if (type === "process") {
    return [...rows]
      .filter(
        (player) =>
          safe(player?.gp, 0) >= 8 &&
          hasRealPct(player?.xgf_pct) &&
          normalizePct(player?.xgf_pct, 0) >= 0.52 &&
          safe(player?.finishing, 0) < 0
      )
      .sort((a, b) => {
        const xgfDifference =
          normalizePct(b?.xgf_pct, 0) - normalizePct(a?.xgf_pct, 0);

        if (xgfDifference !== 0) return xgfDifference;
        return safe(a?.finishing, 0) - safe(b?.finishing, 0);
      })
      .slice(0, limit);
  }

  if (type === "regression") {
    return [...rows]
      .filter((player) => safe(player?.gp, 0) >= 8 && hasRealNumber(player?.finishing))
      .sort((a, b) => Math.abs(safe(b?.finishing, 0)) - Math.abs(safe(a?.finishing, 0)))
      .slice(0, limit);
  }

  if (type === "underused") {
    return [...rows]
      .filter(
        (player) =>
          safe(player?.gp, 0) >= 8 &&
          getAverageTOIMinutes(player) > 0 &&
          getAverageTOIMinutes(player) < 14 &&
          perGame(player?.pts, player?.gp) >= 0.5
      )
      .sort(
        (a, b) =>
          perGame(b?.pts, b?.gp) - perGame(a?.pts, a?.gp)
      )
      .slice(0, limit);
  }

  return [];
}

function stopRowAction(event, callback) {
  event.preventDefault();
  event.stopPropagation();
  callback();
}


/* =========================================================
   MAIN COMPONENT — START
========================================================= */

export function StatsCentralScreen() {
  const {
    franchiseState,
    setScreen,
    statsCentralTab,
    expireFranchiseSession,
    hydrateFranchiseHeavyState,
  } = useGameUI();

  const [tab, setTab] = useState(normalizeStatsMenu(statsCentralTab));
  const [scope, setScope] = useState("league");
  const [lazyStatsCentral, setLazyStatsCentral] = useState(null);
  const [statsLoadState, setStatsLoadState] = useState("idle");

  useEffect(() => {
    if (franchiseState?.roster_browser?.organizations?.length) return;

    hydrateFranchiseHeavyState({
      includeRosterBrowser: true,
      includeDraftClassRankings: false,
      includeDraftClassHud: false,
    });
  }, [
    franchiseState?.roster_browser,
    hydrateFranchiseHeavyState,
  ]);

  useEffect(() => {
    let cancelled = false;

    setStatsLoadState("loading");

    getStatsCentral()
      .then((payload) => {
        if (cancelled) return;
        setLazyStatsCentral(payload || {});
        setStatsLoadState("loaded");
      })
      .catch((error) => {
        if (cancelled) return;

        setLazyStatsCentral(null);
        setStatsLoadState("error");

        if (isExpiredFranchiseSessionError(error)) {
          expireFranchiseSession(formatFranchiseApiError(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [franchiseState?.stats_revision, expireFranchiseSession]);

  const data = useMemo(() => {
    const statsCentral =
      lazyStatsCentral ||
      (statsLoadState === "loaded"
        ? franchiseState?.stats_central
        : null) ||
      {};

    return normalizeStatsCentral({
      ...(franchiseState || {}),
      stats_central: statsCentral,
    });
  }, [franchiseState, lazyStatsCentral, statsLoadState]);

  const handleBack = useCallback(() => {
    setScreen(SCREENS.HUB);
  }, [setScreen]);

  useEffect(() => {
    if (statsCentralTab) {
      setTab(normalizeStatsMenu(statsCentralTab));
    }
  }, [statsCentralTab]);

  useEffect(() => {
    function onKeyDown(event) {
      if (
        event.target?.tagName === "INPUT" ||
        event.target?.tagName === "SELECT" ||
        event.target?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (event.key === "Escape") {
        handleBack();
      }

      if (event.key === "1") setTab("team_stats");
      if (event.key === "2") setTab("player_stats");
      if (event.key === "3") setTab("league_leaders");
    }

    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [handleBack]);

  return (
    <div className="game-screen stats-central-screen">
      <StatsCentralStyles />
      <StatsCentralRedesignStyles />

      <main className="statscentral-shell">
        <section className="sc-command-bar">
          <button
            type="button"
            className="sc-back-link"
            onClick={handleBack}
          >
            ← HUB
          </button>

          <nav className="sc-menu" aria-label="Stats Central menu">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={tab === item.id ? "is-active" : ""}
                onClick={() => setTab(item.id)}
                title={item.label}
              >
                <em>{item.label}</em>
              </button>
            ))}
          </nav>

          <div className="sc-command-context">
            <span>
              {tab === "team_stats"
                ? "League team performance"
                : tab === "player_stats"
                  ? scope === "team"
                    ? "My Team player analysis"
                    : "League player database"
                  : "League leaders"}
            </span>
          </div>
        </section>

        <section
          className={[
            "sc-content",
            tab === "team_stats" ? "is-team-stats" : "",
            tab === "player_stats" ? "is-player-stats" : "",
            tab === "league_leaders" ? "is-league-leaders" : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          {tab === "team_stats" ? (
            <TeamStatsPage
              data={data}
              loadState={statsLoadState}
            />
          ) : null}

          {tab === "player_stats" ? (
            <PlayerStatsPage
              data={data}
              scope={scope}
              onScopeChange={setScope}
              loadState={statsLoadState}
            />
          ) : null}

          {tab === "league_leaders" ? (
            <LeagueLeadersPage data={data} />
          ) : null}
        </section>
      </main>
    </div>
  );
}


function TeamStatsPage({ data, loadState }) {
  return (
    <div className="sc-team-menu-stack">
      <TeamTab
        data={data}
        loadState={loadState}
      />
    </div>
  );
}

function PlayerStatsPage({
  data,
  scope,
  onScopeChange,
  loadState,
}) {
  const [submenu, setSubmenu] = useState("overview");
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [minGp, setMinGp] = useState(scope === "league" ? 1 : 0);
  const [seasonScope, setSeasonScope] = useState("all");
  const [sortKey, setSortKey] = useState("pts");
  const [sortDir, setSortDir] = useState("desc");
  const [density, setDensity] = useState("compact");
  const [page, setPage] = useState(1);
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [pinnedIds, setPinnedIds] = useState([]);
  const [compareLeftId, setCompareLeftId] = useState("");
  const [compareRightId, setCompareRightId] = useState("");

  const activeSkaters =
    scope === "league"
      ? data.skaters || []
      : data.userSkaters || [];

  const activeGoalies =
    scope === "league"
      ? data.goalies || []
      : data.userGoalies || [];

  const allActivePlayers = useMemo(
    () => [...activeSkaters, ...activeGoalies],
    [activeSkaters, activeGoalies]
  );

  const availableTeams = useMemo(() => {
    const map = new Map();

    allActivePlayers.forEach((player) => {
      const rawValue = firstPresent(
        player?.team_id,
        player?.team,
        player?.team_abbrev,
        player?.team_abbr,
        player?.team_name
      );
      const value = normalizeTeamIdentity(rawValue);

      if (!value || map.has(value)) return;

      map.set(value, {
        value,
        label: getPlayerTeamLabel(player),
      });
    });

    return [...map.values()].sort((a, b) =>
      a.label.localeCompare(b.label)
    );
  }, [allActivePlayers]);

  const availableSeasonScopes = useMemo(() => {
    const scopes = new Set();

    allActivePlayers.forEach((player) => {
      const value = safeString(
        firstPresent(
          player?.stat_scope,
          player?.season_scope,
          player?.scope
        ),
        ""
      )
        .trim()
        .toLowerCase();

      if (value) scopes.add(value);
    });

    return [...scopes];
  }, [allActivePlayers]);

  const filterState = useMemo(
    () => ({
      search,
      position,
      team: teamFilter,
      minGp,
      seasonScope,
    }),
    [search, position, teamFilter, minGp, seasonScope]
  );

  const filteredSkaters = useMemo(
    () =>
      activeSkaters.filter((player) =>
        playerMatchesScopeFilter(player, filterState)
      ),
    [activeSkaters, filterState]
  );

  const filteredGoalies = useMemo(
    () =>
      activeGoalies.filter((player) =>
        playerMatchesScopeFilter(player, {
          ...filterState,
          position: "all",
        })
      ),
    [activeGoalies, filterState]
  );

  const sortedSkaters = useMemo(
    () => sortRows(filteredSkaters, sortKey, sortDir),
    [filteredSkaters, sortKey, sortDir]
  );

  const sortedGoalies = useMemo(() => {
    const goalieSortKey = [
      "pts",
      "g",
      "a",
      "points_per_game",
      "points_per_60",
    ].includes(sortKey)
      ? "sv_pct"
      : sortKey;

    const goalieSortDir =
      ["gaa", "losses", "ga"].includes(goalieSortKey)
        ? sortDir
        : sortDir;

    return sortRows(filteredGoalies, goalieSortKey, goalieSortDir);
  }, [filteredGoalies, sortKey, sortDir]);

  const selectedPlayer =
    allActivePlayers.find(
      (player) =>
        String(player.player_id || player.id) ===
        String(selectedPlayerId)
    ) || null;

  const changeSort = useCallback(
    (key, preferredDirection = "desc") => {
      if (sortKey === key) {
        setSortDir((current) =>
          current === "desc" ? "asc" : "desc"
        );
        return;
      }

      setSortKey(key);
      setSortDir(preferredDirection);
    },
    [sortKey]
  );

  const resetFilters = useCallback(() => {
    setSearch("");
    setPosition("all");
    setTeamFilter("all");
    setMinGp(scope === "league" ? 1 : 0);
    setSeasonScope("all");
    setPage(1);
  }, [scope]);

  const togglePin = useCallback((player) => {
    const id = String(player?.player_id || player?.id || "");

    if (!id) return;

    setPinnedIds((current) => {
      if (current.includes(id)) {
        return current.filter((value) => value !== id);
      }

      return [...current, id].slice(-3);
    });
  }, []);

  const sendToCompare = useCallback((player) => {
    const id = String(player?.player_id || player?.id || "");

    if (!id) return;

    setCompareLeftId((currentLeft) => {
      if (!currentLeft || currentLeft === id) {
        return id;
      }

      setCompareRightId(id);
      return currentLeft;
    });

    setSubmenu("compare");
    setPage(1);
  }, []);

  useEffect(() => {
    setPage(1);
  }, [
    submenu,
    scope,
    search,
    position,
    teamFilter,
    minGp,
    seasonScope,
  ]);

  useEffect(() => {
    setMinGp(scope === "league" ? 1 : 0);
    setTeamFilter("all");
    setPosition("all");
    setSelectedPlayerId("");
  }, [scope]);

  // Early season: don't leave Min GP stuck above every player's GP (empty League table).
  useEffect(() => {
    if (scope !== "league" || !activeSkaters.length) return;
    const maxGp = activeSkaters.reduce(
      (max, player) => Math.max(max, safe(player?.gp, 0)),
      0
    );
    if (maxGp > 0 && minGp > maxGp) {
      setMinGp(Math.min(1, maxGp));
    }
  }, [scope, activeSkaters, minGp]);

  if (loadState === "loading" && !allActivePlayers.length) {
    return (
      <div className="sc-player-state-message">
        <strong>Loading player statistics</strong>
        <span>Preparing the game-ledger player database.</span>
      </div>
    );
  }

  if (loadState === "error" && !allActivePlayers.length) {
    return (
      <div className="sc-player-state-message is-error">
        <strong>Player statistics unavailable</strong>
        <span>The current franchise analytics payload could not be loaded.</span>
      </div>
    );
  }

  const showFilters = [
    "skaters",
    "goalies",
    "analytics",
    "special_teams",
  ].includes(submenu);

  return (
    <div className="sc-player-workspace">
      <PlayerStatsWorkspaceHeader
        submenu={submenu}
        onSubmenuChange={setSubmenu}
        scope={scope}
        onScopeChange={onScopeChange}
        search={search}
        onSearchChange={setSearch}
        skaterCount={filteredSkaters.length}
        goalieCount={filteredGoalies.length}
        pinnedCount={pinnedIds.length}
      />

      {showFilters ? (
        <PlayerFilterBar
          scope={scope}
          position={position}
          onPositionChange={setPosition}
          teamFilter={teamFilter}
          onTeamFilterChange={setTeamFilter}
          availableTeams={availableTeams}
          minGp={minGp}
          onMinGpChange={setMinGp}
          seasonScope={seasonScope}
          onSeasonScopeChange={setSeasonScope}
          availableSeasonScopes={availableSeasonScopes}
          density={density}
          onDensityChange={setDensity}
          onReset={resetFilters}
        />
      ) : null}

      <div className="sc-player-panel">
        {submenu === "overview" ? (
          <PlayerOverviewTab
            data={data}
            scope={scope}
            players={activeSkaters}
            goalies={activeGoalies}
            pinnedIds={pinnedIds}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
            onComparePlayer={sendToCompare}
          />
        ) : null}

        {submenu === "skaters" ? (
          <PlayersTab
            players={sortedSkaters}
            scope={scope}
            sortKey={sortKey}
            sortDir={sortDir}
            changeSort={changeSort}
            density={density}
            page={page}
            onPageChange={setPage}
            selectedPlayerId={selectedPlayerId}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
            pinnedIds={pinnedIds}
            onTogglePin={togglePin}
            onComparePlayer={sendToCompare}
          />
        ) : null}

        {submenu === "goalies" ? (
          <GoaliesTab
            goalies={sortedGoalies}
            scope={scope}
            sortKey={sortKey}
            sortDir={sortDir}
            changeSort={changeSort}
            density={density}
            page={page}
            onPageChange={setPage}
            selectedPlayerId={selectedPlayerId}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
            pinnedIds={pinnedIds}
            onTogglePin={togglePin}
            onComparePlayer={sendToCompare}
          />
        ) : null}

        {submenu === "analytics" ? (
          <AdvancedTab
            players={sortedSkaters}
            scope={scope}
            sortKey={sortKey}
            sortDir={sortDir}
            changeSort={changeSort}
            density={density}
            page={page}
            onPageChange={setPage}
            selectedPlayerId={selectedPlayerId}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
            onComparePlayer={sendToCompare}
          />
        ) : null}

        {submenu === "special_teams" ? (
          <SpecialTeamsTab
            players={filteredSkaters}
            scope={scope}
            density={density}
            selectedPlayerId={selectedPlayerId}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
            onComparePlayer={sendToCompare}
          />
        ) : null}

        {submenu === "trends" ? (
          <TrendsTab
            data={data}
            players={activeSkaters}
            goalies={activeGoalies}
            games={data.recentGames}
            scope={scope}
            onSelectPlayer={(player) =>
              setSelectedPlayerId(
                String(player.player_id || player.id)
              )
            }
          />
        ) : null}

        {submenu === "compare" ? (
          <CompareTab
            players={activeSkaters}
            goalies={activeGoalies}
            leftId={compareLeftId}
            rightId={compareRightId}
            onLeftIdChange={setCompareLeftId}
            onRightIdChange={setCompareRightId}
            pinnedIds={pinnedIds}
          />
        ) : null}
      </div>

      {selectedPlayer ? (
        <PlayerDetailDrawer
          player={selectedPlayer}
          comparisonPool={allActivePlayers}
          scope={scope}
          isPinned={pinnedIds.includes(
            String(selectedPlayer.player_id || selectedPlayer.id)
          )}
          onClose={() => setSelectedPlayerId("")}
          onTogglePin={() => togglePin(selectedPlayer)}
          onCompare={() => sendToCompare(selectedPlayer)}
        />
      ) : null}
    </div>
  );
}


function PlayerStatsWorkspaceHeader({
  submenu,
  onSubmenuChange,
  scope,
  onScopeChange,
  search,
  onSearchChange,
  skaterCount,
  goalieCount,
  pinnedCount,
}) {
  return (
    <header className="sc-player-header">
      <div className="sc-player-heading">
        <span>PLAYER STATS</span>
        <strong>
          {scope === "team"
            ? "My Team"
            : "League"}
        </strong>
        <em>
          {skaterCount} skaters · {goalieCount} goalies
          {pinnedCount ? ` · ${pinnedCount} pinned` : ""}
        </em>
      </div>

      <nav
        className="sc-player-subnav"
        aria-label="Player statistics submenu"
      >
        {PLAYER_SUBMENUS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={
              submenu === item.id ? "is-active" : ""
            }
            onClick={() =>
              onSubmenuChange(item.id)
            }
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sc-player-header-actions">
        <div
          className="sc-player-scope"
          aria-label="Player statistics scope"
        >
          <button
            type="button"
            className={
              scope === "team" ? "is-active" : ""
            }
            onClick={() =>
              onScopeChange("team")
            }
          >
            My Team
          </button>

          <button
            type="button"
            className={
              scope === "league" ? "is-active" : ""
            }
            onClick={() =>
              onScopeChange("league")
            }
          >
            League
          </button>
        </div>

        <label className="sc-player-search">
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            value={search}
            onChange={(event) =>
              onSearchChange(event.target.value)
            }
            placeholder="Search players"
            aria-label="Search players"
          />
        </label>
      </div>
    </header>
  );
}

function PlayerFilterBar({
  scope,
  position,
  onPositionChange,
  teamFilter,
  onTeamFilterChange,
  availableTeams,
  minGp,
  onMinGpChange,
  seasonScope,
  onSeasonScopeChange,
  availableSeasonScopes,
  density,
  onDensityChange,
  onReset,
}) {
  return (
    <section className="sc-player-filter-bar">
      <div className="sc-player-filter-group is-positions">
        {[
          ["all", "All"],
          ["F", "F"],
          ["C", "C"],
          ["LW", "LW"],
          ["RW", "RW"],
          ["D", "D"],
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={
              position === value ? "is-active" : ""
            }
            onClick={() =>
              onPositionChange(value)
            }
          >
            {label}
          </button>
        ))}
      </div>

      {scope === "league" ? (
        <select
          value={teamFilter}
          onChange={(event) =>
            onTeamFilterChange(
              event.target.value
            )
          }
          aria-label="Filter by team"
        >
          <option value="all">All Teams</option>
          {availableTeams.map((team) => (
            <option
              key={team.value}
              value={team.value}
            >
              {team.label}
            </option>
          ))}
        </select>
      ) : null}

      <label className="sc-player-min-gp">
        <span>Min GP</span>
        <input
          type="number"
          min="0"
          max="82"
          value={minGp}
          onChange={(event) =>
            onMinGpChange(
              Math.max(
                0,
                Number(event.target.value) || 0
              )
            )
          }
        />
      </label>

      {availableSeasonScopes.length ? (
        <select
          value={seasonScope}
          onChange={(event) =>
            onSeasonScopeChange(
              event.target.value
            )
          }
          aria-label="Filter by season scope"
        >
          <option value="all">All Games</option>
          {availableSeasonScopes.map((item) => (
            <option key={item} value={item}>
              {item
                .replace(/_/g, " ")
                .replace(/\b\w/g, (letter) =>
                  letter.toUpperCase()
                )}
            </option>
          ))}
        </select>
      ) : null}

      <div className="sc-player-density">
        <button
          type="button"
          className={
            density === "compact"
              ? "is-active"
              : ""
          }
          onClick={() =>
            onDensityChange("compact")
          }
        >
          Compact
        </button>

        <button
          type="button"
          className={
            density === "comfortable"
              ? "is-active"
              : ""
          }
          onClick={() =>
            onDensityChange("comfortable")
          }
        >
          Detailed
        </button>
      </div>

      <button
        type="button"
        className="sc-player-filter-reset"
        onClick={onReset}
      >
        Reset
      </button>
    </section>
  );
}

function PagedDataTable({
  columns,
  rows,
  page,
  onPageChange,
  pageSize,
  empty,
  ...tableProps
}) {
  const totalRows = rows?.length || 0;
  const totalPages = Math.max(
    1,
    Math.ceil(totalRows / pageSize)
  );
  const safePage = Math.min(
    Math.max(1, page),
    totalPages
  );
  const startIndex =
    (safePage - 1) * pageSize;
  const pageRows = (rows || []).slice(
    startIndex,
    startIndex + pageSize
  );
  const range = getVisibleRange(
    safePage,
    pageSize,
    totalRows
  );

  useEffect(() => {
    if (page !== safePage) {
      onPageChange(safePage);
    }
  }, [page, safePage, onPageChange]);

  return (
    <div className="sc-paged-table">
      <DataTable
        columns={columns}
        rows={pageRows}
        empty={empty}
        {...tableProps}
      />

      <footer className="sc-table-pagination">
        <span>
          {totalRows
            ? `${range.start}–${range.end} of ${totalRows}`
            : "0 results"}
        </span>

        <div>
          <button
            type="button"
            disabled={safePage <= 1}
            onClick={() =>
              onPageChange(safePage - 1)
            }
          >
            Previous
          </button>

          <strong>
            {safePage} / {totalPages}
          </strong>

          <button
            type="button"
            disabled={safePage >= totalPages}
            onClick={() =>
              onPageChange(safePage + 1)
            }
          >
            Next
          </button>
        </div>
      </footer>
    </div>
  );
}

function PlayerRowActions({
  player,
  isPinned,
  onTogglePin,
  onCompare,
}) {
  return (
    <div className="sc-player-row-actions">
      <button
        type="button"
        className={isPinned ? "is-active" : ""}
        title={
          isPinned ? "Unpin player" : "Pin player"
        }
        aria-label={
          isPinned ? "Unpin player" : "Pin player"
        }
        onClick={(event) =>
          stopRowAction(event, onTogglePin)
        }
      >
        {isPinned ? "Pinned" : "Pin"}
      </button>

      <button
        type="button"
        title="Compare player"
        aria-label="Compare player"
        onClick={(event) =>
          stopRowAction(event, onCompare)
        }
      >
        Compare
      </button>
    </div>
  );
}

function fmtOverviewSavePct(value) {
  if (!hasRealPct(value)) return null;
  return fmtSavePct(value);
}

function fmtOverviewShPct(value) {
  if (!hasRealPct(value)) return null;
  return fmtPct(value, 1);
}

function getOverviewPlayoffLine(team = {}) {
  const value = firstPresent(
    team.playoff_position,
    team.playoff_spot,
    team.playoff_status,
    team.clinched_playoff_spot === true ? "Clinched" : null,
    team.in_playoffs === true ? "In playoffs" : null
  );
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "boolean") return value ? "Playoff spot" : "";
  return String(value);
}

function buildOverviewFrontOfficeRead(team = {}, players = [], goalies = []) {
  const reads = [];
  const skaters = Array.isArray(players) ? players : [];
  const totalTeams = team.total_league_teams || 32;

  if (team.gf_league_rank > 0 && team.gf_league_rank <= 8) {
    reads.push({
      label: "Strength",
      text: `Offense ranks ${formatLeagueRank(team.gf_league_rank, totalTeams)}.`,
    });
  } else if (hasRealPct(team.pp_pct) && normalizePct(team.pp_pct) >= 0.24) {
    reads.push({
      label: "Strength",
      text: `Power play converts at ${fmtMaybePct(team.pp_pct)}.`,
    });
  } else if (hasRealPct(team.xgf_pct) && normalizePct(team.xgf_pct) >= 0.52) {
    reads.push({
      label: "Strength",
      text: `Chance share holds at ${fmtMaybePct(team.xgf_pct)}.`,
    });
  } else if (hasRealPct(team.cf_pct) && normalizePct(team.cf_pct) >= 0.52) {
    reads.push({
      label: "Strength",
      text: `Shot share leads at ${fmtMaybePct(team.cf_pct)}.`,
    });
  }

  if (team.ga_league_rank > 0 && team.ga_league_rank >= 24) {
    reads.push({
      label: "Concern",
      text: `Goals against sit ${formatLeagueRank(team.ga_league_rank, totalTeams)}.`,
    });
  } else if (hasRealPct(team.pk_pct) && normalizePct(team.pk_pct) < 0.78) {
    reads.push({
      label: "Concern",
      text: `Penalty kill lagging at ${fmtMaybePct(team.pk_pct)}.`,
    });
  } else if (hasRealPct(team.sv_pct) && normalizePct(team.sv_pct) < 0.9) {
    reads.push({
      label: "Concern",
      text: `Team save rate is ${fmtOverviewSavePct(team.sv_pct)}.`,
    });
  } else if (hasRealPct(team.xgf_pct) && normalizePct(team.xgf_pct) < 0.47) {
    reads.push({
      label: "Concern",
      text: `Expected goals share only ${fmtMaybePct(team.xgf_pct)}.`,
    });
  }

  const inspectCandidate =
    [...skaters]
      .filter(
        (player) =>
          safe(player?.gp, 0) >= 8 &&
          hasRealPct(player?.xgf_pct) &&
          normalizePct(player.xgf_pct) >= 0.54 &&
          hasRealNumber(player?.finishing) &&
          safe(player.finishing, 0) < -1
      )
      .sort(
        (a, b) =>
          normalizePct(b?.xgf_pct, 0) - normalizePct(a?.xgf_pct, 0)
      )[0] ||
    [...skaters]
      .filter((player) => safe(player?.gp, 0) >= 8)
      .sort(
        (a, b) =>
          getAverageTOIMinutes(b) - getAverageTOIMinutes(a)
      )
      .find(
        (player) =>
          getAverageTOIMinutes(player) >= 18 &&
          perGame(player?.pts, player?.gp) < 0.45
      ) ||
    topBy(goalies, (goalie) =>
      hasRealNumber(goalie?.gsax) ? -Math.abs(safe(goalie.gsax, 0)) : 0
    );

  if (inspectCandidate?.name) {
    reads.push({
      label: "Inspect",
      text: `Check ${String(inspectCandidate.name).split(" ").slice(-1)[0]} usage next.`,
      player: inspectCandidate,
    });
  }

  return reads.slice(0, 3);
}

function buildOverviewRosterContribution(players = []) {
  const skaters = Array.isArray(players) ? players : [];
  const metrics = [];
  const teamPts = skaters.reduce((sum, row) => sum + safe(row?.pts, 0), 0);
  if (teamPts > 0) {
    const ranked = [...skaters].sort(
      (a, b) => safe(b?.pts, 0) - safe(a?.pts, 0)
    );
    const top = ranked[0];
    if (top && safe(top.pts, 0) > 0) {
      metrics.push({
        label: "Top scorer share",
        value: fmtPct(safe(top.pts, 0) / teamPts, 0),
        detail: `${top.name} · ${fmtZero(top.pts)} PTS`,
      });
    }
    const topThreePts = ranked
      .slice(0, 3)
      .reduce((sum, row) => sum + safe(row?.pts, 0), 0);
    if (topThreePts > 0) {
      metrics.push({
        label: "Top-three share",
        value: fmtPct(topThreePts / teamPts, 0),
        detail: `${fmtZero(topThreePts)} of ${fmtZero(teamPts)} PTS`,
      });
    }
    const defencePts = skaters
      .filter((row) => isDefenseRow(row))
      .reduce((sum, row) => sum + safe(row?.pts, 0), 0);
    metrics.push({
      label: "Defence scoring",
      value: fmtPct(defencePts / teamPts, 0),
      detail: `${fmtZero(defencePts)} PTS from D`,
    });
    const ppPts = skaters.reduce(
      (sum, row) =>
        sum +
        safe(
          firstPresent(row?.pp_points, safe(row?.ppg, 0) + safe(row?.ppa, 0)),
          0
        ),
      0
    );
    if (ppPts > 0) {
      metrics.push({
        label: "Power-play share",
        value: fmtPct(ppPts / teamPts, 0),
        detail: `${fmtZero(ppPts)} PP PTS`,
      });
    }
  }

  const positiveXgf = skaters.filter(
    (row) => hasRealPct(row?.xgf_pct) && normalizePct(row.xgf_pct) >= 0.52
  );
  if (skaters.some((row) => hasRealPct(row?.xgf_pct))) {
    metrics.push({
      label: "Above 52% xGF",
      value: String(positiveXgf.length),
      detail: `${positiveXgf.length} of ${skaters.filter((row) => hasRealPct(row?.xgf_pct)).length} with sample`,
    });
  }

  return metrics;
}

function buildOverviewTeamMetrics(team = {}, totalTeams = 32) {
  const metrics = [];
  const push = (entry) => {
    if (!entry || entry.value === null || entry.value === undefined || entry.value === "") {
      return;
    }
    metrics.push(entry);
  };

  const rankTone = (rank, asc = false) => {
    if (!rank) return "neutral";
    if ((!asc && rank <= 10) || (asc && rank <= 10)) return "good";
    if ((!asc && rank >= 24) || (asc && rank >= 24)) return "bad";
    return "neutral";
  };

  if (hasRealNumber(team.points_pct)) {
    push({
      label: "Points %",
      value: fmtPct(team.points_pct, 1),
      description: "Standings points earned",
      rank: formatLeagueRank(team.league_rank, totalTeams),
      tone: rankTone(team.league_rank),
    });
  }
  if (hasRealNumber(team.gf_per_game)) {
    push({
      label: "GF/GP",
      value: fmtMaybeOne(team.gf_per_game),
      description: "Goals for per game",
      rank: formatLeagueRank(team.gf_league_rank, totalTeams),
      tone: rankTone(team.gf_league_rank),
    });
  }
  if (hasRealNumber(team.ga_per_game)) {
    push({
      label: "GA/GP",
      value: fmtMaybeOne(team.ga_per_game),
      description: "Goals against per game",
      rank: formatLeagueRank(team.ga_league_rank, totalTeams),
      tone: rankTone(team.ga_league_rank, true),
    });
  }
  if (hasRealNumber(team.goal_diff)) {
    push({
      label: "Goal Diff",
      value: formatSigned(team.goal_diff),
      description: "Goals for minus against",
      rank: formatLeagueRank(team.goal_diff_league_rank, totalTeams),
      tone:
        team.goal_diff > 0 ? "good" : team.goal_diff < 0 ? "bad" : "neutral",
    });
  }
  if (hasRealNumber(team.sf_per_game) && safe(team.sf, 0) > 0) {
    push({
      label: "SF/GP",
      value: fmtMaybeOne(team.sf_per_game),
      description: "Shots for per game",
      rank: "",
      tone: "neutral",
    });
  }
  if (hasRealNumber(team.sa_per_game) && safe(team.sa, 0) > 0) {
    push({
      label: "SA/GP",
      value: fmtMaybeOne(team.sa_per_game),
      description: "Shots against per game",
      rank: "",
      tone: "neutral",
    });
  }
  if (hasRealPct(team.pp_pct) && !team.analytics_missing?.pp_pct) {
    push({
      label: "PP%",
      value: fmtMaybePct(team.pp_pct),
      description:
        safe(team.ppo, 0) > 0
          ? `${fmtZero(team.ppg)} PPG / ${fmtZero(team.ppo)} PPO`
          : "Power-play conversion",
      rank: formatLeagueRank(team.pp_pct_league_rank, totalTeams),
      tone: rankTone(team.pp_pct_league_rank),
    });
  }
  if (hasRealPct(team.pk_pct) && !team.analytics_missing?.pk_pct) {
    const tsh = safe(team.opp_ppo, 0);
    push({
      label: "PK%",
      value: fmtMaybePct(team.pk_pct),
      description:
        tsh > 0
          ? `${fmtZero(team.ppga)} PPGA / ${fmtZero(tsh)} TSH`
          : "Penalty-kill rate",
      rank: formatLeagueRank(team.pk_pct_league_rank, totalTeams),
      tone: rankTone(team.pk_pct_league_rank),
    });
  }
  if (hasRealPct(team.cf_pct) && !team.analytics_missing?.cf_pct) {
    push({
      label: "CF%",
      value: fmtMaybePct(team.cf_pct),
      description: "Corsi share",
      rank: formatLeagueRank(team.cf_pct_league_rank, totalTeams),
      tone:
        normalizePct(team.cf_pct) >= 0.52
          ? "good"
          : normalizePct(team.cf_pct) <= 0.47
            ? "bad"
            : "neutral",
      refPct: normalizePct(team.cf_pct),
    });
  }
  if (hasRealPct(team.xgf_pct) && !team.analytics_missing?.xgf_pct) {
    push({
      label: "xGF%",
      value: fmtMaybePct(team.xgf_pct),
      description: "Expected goals share",
      rank: formatLeagueRank(team.xgf_pct_league_rank, totalTeams),
      tone:
        normalizePct(team.xgf_pct) >= 0.52
          ? "good"
          : normalizePct(team.xgf_pct) <= 0.47
            ? "bad"
            : "neutral",
      refPct: normalizePct(team.xgf_pct),
    });
  }
  if (hasRealPct(team.sh_pct)) {
    push({
      label: "SH%",
      value: fmtOverviewShPct(team.sh_pct),
      description: "Team shooting percentage",
      rank: "",
      tone: "neutral",
    });
  }
  if (hasRealPct(team.sv_pct)) {
    push({
      label: "SV%",
      value: fmtOverviewSavePct(team.sv_pct),
      description: "Team save percentage",
      rank: formatLeagueRank(team.ga_league_rank, totalTeams),
      tone:
        normalizePct(team.sv_pct) >= 0.91
          ? "good"
          : normalizePct(team.sv_pct) < 0.89
            ? "bad"
            : "neutral",
    });
  }
  if (hasRealNumber(team.pdo) && !team.analytics_missing?.pdo) {
    push({
      label: "PDO",
      value: fmtPdo(team.pdo),
      description: "SH% + SV%",
      rank: formatLeagueRank(team.pdo_league_rank, totalTeams),
      tone: "neutral",
    });
  }

  return metrics;
}

function PlayerOverviewTab({
  data,
  scope,
  players,
  goalies,
  onSelectPlayer,
}) {
  const teams = data?.teams || [];
  const franchiseState = data?.franchiseState || null;
  const team = data?.team || {};
  const teamInfo = data?.teamInfo || {};
  const totalTeams =
    team.total_league_teams || teams.length || 32;

  const [posFilter, setPosFilter] = useState("all");
  const [sortKey, setSortKey] = useState("pts");
  const [sortDir, setSortDir] = useState("desc");
  const [sortPreset, setSortPreset] = useState("points");

  const skaters = useMemo(() => {
    const pool =
      Array.isArray(data?.userSkaters) && data.userSkaters.length
        ? data.userSkaters
        : Array.isArray(players)
          ? players
          : [];
    return pool.filter((row) => !isGoalieRow(row));
  }, [data?.userSkaters, players]);

  const goaliePool = useMemo(() => {
    if (Array.isArray(data?.userGoalies) && data.userGoalies.length) {
      return data.userGoalies;
    }
    return Array.isArray(goalies) ? goalies : [];
  }, [data?.userGoalies, goalies]);

  const filteredSkaters = useMemo(() => {
    if (posFilter === "f") return skaters.filter((row) => isForwardRow(row));
    if (posFilter === "d") return skaters.filter((row) => isDefenseRow(row));
    return skaters;
  }, [skaters, posFilter]);

  const sortedSkaters = useMemo(() => {
    if (sortKey === "toi_avg") {
      const dir = sortDir === "asc" ? 1 : -1;
      return [...filteredSkaters].sort(
        (a, b) =>
          (getAverageTOIMinutes(a) - getAverageTOIMinutes(b)) * dir
      );
    }
    if (sortKey === "overall") {
      const dir = sortDir === "asc" ? 1 : -1;
      return [...filteredSkaters].sort(
        (a, b) =>
          (getUniversalOverall(a) - getUniversalOverall(b)) * dir
      );
    }
    return sortRows(filteredSkaters, sortKey, sortDir);
  }, [filteredSkaters, sortKey, sortDir]);

  const changeSort = useCallback((key, forcedDir) => {
    setSortPreset("");
    setSortKey((prevKey) => {
      if (forcedDir) {
        setSortDir(forcedDir);
        return key;
      }
      if (prevKey === key) {
        setSortDir((prevDir) => (prevDir === "desc" ? "asc" : "desc"));
        return key;
      }
      setSortDir("desc");
      return key;
    });
  }, []);

  const applySortPreset = useCallback((preset) => {
    setSortPreset(preset);
    if (preset === "points") {
      setSortKey("pts");
      setSortDir("desc");
    } else if (preset === "goals") {
      setSortKey("g");
      setSortDir("desc");
    } else if (preset === "rate") {
      setSortKey("points_per_game");
      setSortDir("desc");
    }
  }, []);

  const topPoints = topBy(skaters, (player) => player.pts);
  const topGoals = topBy(skaters, (player) => player.g);
  const topAssists = topBy(skaters, (player) => player.a);
  const topPpg = topBy(skaters, (player) =>
    safe(player.gp, 0) >= 5 ? perGame(player.pts, player.gp) : -999
  );
  const topToi = topBy(
    skaters.filter((row) => isDefenseRow(row) || getAverageTOIMinutes(row) > 0),
    (player) =>
      isDefenseRow(player)
        ? getAverageTOIMinutes(player) + 0.01
        : getAverageTOIMinutes(player)
  );
  const topGoalie = topBy(goaliePool, (goalie) =>
    hasRealNumber(goalie.gsax)
      ? goalie.gsax
      : normalizePct(goalie.sv_pct, 0)
  );

  const teamMetrics = useMemo(
    () => buildOverviewTeamMetrics(team, totalTeams),
    [team, totalTeams]
  );
  const contributionMetrics = useMemo(
    () => buildOverviewRosterContribution(skaters),
    [skaters]
  );
  const frontOfficeReads = useMemo(
    () => buildOverviewFrontOfficeRead(team, skaters, goaliePool),
    [team, skaters, goaliePool]
  );

  const headerTeam = {
    ...team,
    ...teamInfo,
    name: teamNameFromInfo(teamInfo, team.name || "Franchise"),
    team_logo_src:
      team.team_logo_src ||
      teamInfo.team_logo_src ||
      getTeamLogoSrc(team) ||
      getTeamLogoSrc(teamInfo),
  };

  const columns = [
    {
      label: "#",
      key: "rank",
      className: "is-rank-col",
      render: (row) => {
        const index = sortedSkaters.findIndex(
          (entry) =>
            String(entry.player_id || entry.id) ===
            String(row.player_id || row.id)
        );
        return index >= 0 ? index + 1 : "—";
      },
    },
    {
      label: "Player",
      sortKey: "name",
      className: "is-player-col",
      render: (row) => (
        <OverviewTablePlayerCell
          player={row}
          teams={teams}
          franchiseState={franchiseState}
        />
      ),
      onClick: () => changeSort("name", "asc"),
    },
    {
      label: "GP",
      sortKey: "gp",
      align: "right",
      render: (row) => fmtZero(row.gp),
      onClick: () => changeSort("gp"),
    },
    {
      label: "G",
      sortKey: "g",
      align: "right",
      render: (row) => fmtZero(row.g),
      onClick: () => changeSort("g"),
    },
    {
      label: "A",
      sortKey: "a",
      align: "right",
      render: (row) => fmtZero(row.a),
      onClick: () => changeSort("a"),
    },
    {
      label: "PTS",
      sortKey: "pts",
      align: "right",
      render: (row) => (
        <strong className="sc-overview-pts">{fmtZero(row.pts)}</strong>
      ),
      onClick: () => changeSort("pts"),
    },
    {
      label: "P/GP",
      sortKey: "points_per_game",
      align: "right",
      render: (row) =>
        fmtMaybeTwo(
          hasRealNumber(row.points_per_game)
            ? row.points_per_game
            : perGame(row.pts, row.gp)
        ),
      onClick: () => changeSort("points_per_game"),
    },
    {
      label: "SOG",
      sortKey: "sog",
      align: "right",
      render: (row) => fmtZero(row.sog),
      onClick: () => changeSort("sog"),
    },
    {
      label: "SH%",
      sortKey: "shooting_pct",
      align: "right",
      render: (row) =>
        fmtOverviewShPct(
          firstPresent(row.shooting_pct, row.sh_pct)
        ) || "—",
      onClick: () => changeSort("shooting_pct"),
    },
    {
      label: "TOI/GP",
      sortKey: "toi_avg",
      align: "right",
      render: (row) => formatSmallTOI(row),
      onClick: () => changeSort("toi_avg"),
    },
    {
      label: "OVR",
      sortKey: "overall",
      align: "right",
      render: (row) => {
        const ovr = getUniversalOverall(row);
        return ovr > 0 ? fmtZero(ovr) : "—";
      },
      onClick: () => changeSort("overall"),
    },
  ];

  return (
    <div className="sc-player-overview">
      <OverviewTeamSnapshotHeader
        team={headerTeam}
        totalTeams={totalTeams}
        skaterCount={skaters.length}
        goalieCount={goaliePool.length}
        scope={scope}
      />

      <section className="sc-overview-featured-row">
        <OverviewFeaturedLeaderCard
          eyebrow="Featured Offensive Leader"
          player={topPoints}
          primaryLabel="PTS"
          primaryValue={topPoints ? fmtZero(topPoints.pts) : "—"}
          supporting={[
            topPoints ? `${fmtZero(topPoints.g)} G` : null,
            topPoints ? `${fmtZero(topPoints.a)} A` : null,
            topPoints
              ? `${fmtMaybeTwo(perGame(topPoints.pts, topPoints.gp))} P/GP`
              : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
          featured
        />

        <OverviewFeaturedLeaderCard
          eyebrow="Goals"
          player={topGoals}
          primaryLabel="G"
          primaryValue={topGoals ? fmtZero(topGoals.g) : "—"}
          supporting={[
            topGoals ? `${fmtZero(topGoals.sog)} SOG` : null,
            topGoals
              ? fmtOverviewShPct(
                  firstPresent(topGoals.shooting_pct, topGoals.sh_pct)
                )
              : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
        />

        <OverviewFeaturedLeaderCard
          eyebrow="Assists"
          player={topAssists}
          primaryLabel="A"
          primaryValue={topAssists ? fmtZero(topAssists.a) : "—"}
          supporting={[
            topAssists ? `${fmtZero(topAssists.pts)} PTS` : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
        />

        <OverviewFeaturedLeaderCard
          eyebrow="Rate Leader"
          player={topPpg}
          primaryLabel="P/GP"
          primaryValue={
            topPpg
              ? fmtMaybeTwo(perGame(topPpg.pts, topPpg.gp))
              : "—"
          }
          supporting={[
            topPpg ? `${fmtZero(topPpg.pts)} PTS` : null,
            topPpg ? `${fmtZero(topPpg.gp)} GP` : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
        />

        <OverviewFeaturedLeaderCard
          eyebrow="Defensive / TOI"
          player={topToi}
          primaryLabel="TOI"
          primaryValue={topToi ? formatSmallTOI(topToi) : "—"}
          supporting={[
            topToi
              ? normalizePosition(
                  firstPresent(topToi.position, topToi.pos, "D")
                )
              : null,
            topToi ? `${fmtZero(topToi.pts)} PTS` : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
        />

        <OverviewFeaturedLeaderCard
          eyebrow="Goalie Leader"
          player={topGoalie}
          primaryLabel={
            topGoalie && hasRealNumber(topGoalie.gsax) ? "GSAx" : "SV%"
          }
          primaryValue={
            topGoalie
              ? hasRealNumber(topGoalie.gsax)
                ? fmtMaybeOne(topGoalie.gsax)
                : fmtOverviewSavePct(topGoalie.sv_pct) || "—"
              : "—"
          }
          supporting={[
            topGoalie && hasRealPct(topGoalie.sv_pct)
              ? fmtOverviewSavePct(topGoalie.sv_pct)
              : null,
            topGoalie ? `${fmtZero(topGoalie.gp)} GP` : null,
          ].filter(Boolean)}
          teams={teams}
          franchiseState={franchiseState}
          onSelectPlayer={onSelectPlayer}
        />
      </section>

      <section className="sc-overview-main-grid">
        <OverviewScoringLeadersPanel
          rows={sortedSkaters}
          columns={columns}
          sortKey={sortKey}
          sortDir={sortDir}
          posFilter={posFilter}
          sortPreset={sortPreset}
          onPosFilter={setPosFilter}
          onSortPreset={applySortPreset}
          onSelectPlayer={onSelectPlayer}
        />

        <div className="sc-overview-side-stack">
          <OverviewTeamPerformancePanel metrics={teamMetrics} />
          <OverviewRosterContributionPanel metrics={contributionMetrics} />
          <OverviewFrontOfficeReadPanel
            reads={frontOfficeReads}
            onSelectPlayer={onSelectPlayer}
          />
        </div>
      </section>
    </div>
  );
}

function OverviewTeamSnapshotHeader({
  team = {},
  totalTeams = 32,
  skaterCount = 0,
  goalieCount = 0,
  scope = "team",
}) {
  const recordText =
    team.wins != null || team.losses != null
      ? `${safe(team.wins, 0)}-${safe(team.losses, 0)}-${safe(team.otl, 0)}`
      : "";
  const playoffLine = getOverviewPlayoffLine(team);
  const chips = [
    team.gp > 0 ? { label: "GP", value: fmtZero(team.gp) } : null,
    recordText ? { label: "Record", value: recordText } : null,
    team.points != null ? { label: "PTS", value: fmtZero(team.points) } : null,
    team.league_rank > 0
      ? {
          label: "League",
          value: formatLeagueRank(team.league_rank, totalTeams),
        }
      : null,
    team.division_rank > 0
      ? { label: "Division", value: `#${team.division_rank}` }
      : null,
    team.conference_rank > 0
      ? { label: "Conference", value: `#${team.conference_rank}` }
      : null,
    hasRealNumber(team.goal_diff)
      ? { label: "Diff", value: formatSigned(team.goal_diff) }
      : null,
    playoffLine ? { label: "Playoffs", value: playoffLine } : null,
  ].filter(Boolean);

  return (
    <section className="sc-overview-team-header">
      <div className="sc-overview-team-header-identity">
        <TeamLogoMark team={team} size="large" />
        <div>
          <span>
            {scope === "team" ? "TEAM SNAPSHOT" : "FRANCHISE CONTEXT"}
          </span>
          <h2>{teamDisplayLabel(team)}</h2>
          <p>
            {skaterCount} skaters · {goalieCount} goalies
            {team.division ? ` · ${team.division}` : ""}
            {team.conference ? ` · ${team.conference}` : ""}
          </p>
        </div>
      </div>

      {chips.length ? (
        <div className="sc-overview-team-header-chips">
          {chips.map((chip) => (
            <div key={`${chip.label}-${chip.value}`}>
              <em>{chip.label}</em>
              <strong>{chip.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function OverviewFeaturedLeaderCard({
  eyebrow,
  player,
  primaryLabel,
  primaryValue,
  supporting = [],
  teams = [],
  franchiseState = null,
  onSelectPlayer,
  featured = false,
}) {
  const ovr = getUniversalOverall(player);
  const position = normalizePosition(
    firstPresent(player?.position, player?.pos, featured ? "F" : "—")
  );

  return (
    <button
      type="button"
      className={`sc-overview-leader-tile ${featured ? "is-featured" : ""}`}
      disabled={!player}
      onClick={() => player && onSelectPlayer(player)}
    >
      <span className="sc-overview-leader-eyebrow">{eyebrow}</span>
      <div className="sc-overview-leader-tile-body">
        <div className="sc-overview-leader-portrait">
          <PlayerAvatar
            player={player}
            large={featured}
            small={!featured}
            teams={teams}
            franchiseState={franchiseState}
          />
        </div>
        <div className="sc-overview-leader-copy">
          <strong>{player?.name || "—"}</strong>
          <em>
            {player ? position : "—"}
            {ovr > 0 ? ` · ${ovr} OVR` : ""}
            {player?.gp != null ? ` · ${fmtZero(player.gp)} GP` : ""}
          </em>
          <div className="sc-overview-leader-metric">
            <b>{primaryValue}</b>
            <i>{primaryLabel}</i>
          </div>
          {supporting.length ? (
            <p>{supporting.join(" · ")}</p>
          ) : null}
        </div>
      </div>
    </button>
  );
}

function OverviewTablePlayerCell({
  player,
  teams = [],
  franchiseState = null,
}) {
  const ovr = getUniversalOverall(player);
  const position = normalizePosition(
    firstPresent(player?.position, player?.pos, "F")
  );

  return (
    <div className="sc-overview-table-player">
      <PlayerAvatar
        player={player}
        small
        teams={teams}
        franchiseState={franchiseState}
      />
      <span>
        <strong>{player?.name || "—"}</strong>
        <em>
          {position}
          {ovr > 0 ? ` · ${ovr}` : ""}
        </em>
      </span>
    </div>
  );
}

function OverviewScoringLeadersPanel({
  rows,
  columns,
  sortKey,
  sortDir,
  posFilter,
  sortPreset,
  onPosFilter,
  onSortPreset,
  onSelectPlayer,
}) {
  return (
    <section className="sc-overview-module sc-overview-scoring-leaders">
      <header className="sc-overview-module-header">
        <div>
          <span>SCORING LEADERS</span>
          <strong>Team Skater Board</strong>
        </div>
        <div className="sc-overview-board-controls">
          <div className="sc-overview-chip-group" role="group" aria-label="Position filter">
            {[
              ["all", "All"],
              ["f", "Forwards"],
              ["d", "Defence"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={posFilter === id ? "is-active" : ""}
                onClick={() => onPosFilter(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="sc-overview-chip-group" role="group" aria-label="Sort presets">
            {[
              ["points", "Points"],
              ["goals", "Goals"],
              ["rate", "Rate"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={sortPreset === id ? "is-active" : ""}
                onClick={() => onSortPreset(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <DataTable
        columns={columns}
        rows={rows}
        sortKey={sortKey}
        sortDir={sortDir}
        density="compact"
        tableClassName="sc-overview-scoring-table"
        getRowId={(row) => row.player_id || row.id}
        onRowClick={onSelectPlayer}
        empty="No skater scoring rows match the current filters."
      />
    </section>
  );
}

function OverviewTeamPerformancePanel({ metrics = [] }) {
  return (
    <section className="sc-overview-module sc-overview-team-performance">
      <header className="sc-overview-module-header">
        <div>
          <span>TEAM PERFORMANCE</span>
          <strong>Scoring Context</strong>
        </div>
      </header>

      {metrics.length ? (
        <div className="sc-overview-metric-grid">
          {metrics.map((metric) => (
            <article
              key={metric.label}
              className={`sc-overview-metric-card is-${metric.tone || "neutral"}`}
            >
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
              <em>{metric.description}</em>
              {metric.rank ? <b>{metric.rank}</b> : null}
              {typeof metric.refPct === "number" ? (
                <div
                  className="sc-overview-ref-bar"
                  aria-hidden="true"
                >
                  <i style={{ width: `${clamp(metric.refPct * 100, 0, 100)}%` }} />
                  <em />
                </div>
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <div className="sc-empty">Team performance fields are unavailable.</div>
      )}
    </section>
  );
}

function OverviewRosterContributionPanel({ metrics = [] }) {
  if (!metrics.length) return null;

  return (
    <section className="sc-overview-module sc-overview-roster-contribution">
      <header className="sc-overview-module-header">
        <div>
          <span>ROSTER CONTRIBUTION</span>
          <strong>Where Scoring Comes From</strong>
        </div>
      </header>

      <div className="sc-overview-contribution-list">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <em>{metric.detail}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function OverviewFrontOfficeReadPanel({ reads = [], onSelectPlayer }) {
  if (!reads.length) return null;

  return (
    <section className="sc-overview-module sc-overview-front-office">
      <header className="sc-overview-module-header">
        <div>
          <span>FRONT OFFICE READ</span>
          <strong>Quick Notes</strong>
        </div>
      </header>

      <div className="sc-overview-read-list">
        {reads.map((read) => {
          const content = (
            <>
              <span>{read.label}</span>
              <strong>{read.text}</strong>
            </>
          );

          if (read.player) {
            return (
              <button
                key={read.label}
                type="button"
                onClick={() => onSelectPlayer(read.player)}
              >
                {content}
              </button>
            );
          }

          return <div key={read.label}>{content}</div>;
        })}
      </div>
    </section>
  );
}

function PlayerDetailDrawer({
  player,
  comparisonPool,
  scope,
  isPinned,
  onClose,
  onTogglePin,
  onCompare,
}) {
  const isGoalie =
    normalizePosition(player.position) === "G";

  const percentileRows = isGoalie
    ? [
        ["SV%", "sv_pct", "desc"],
        ["GAA", "gaa", "asc"],
        ["GSAx", "gsax", "desc"],
        ["WAR", "war", "desc"],
      ]
    : [
        ["PTS", "pts", "desc"],
        ["P/GP", "points_per_game", "desc"],
        ["xGF%", "xgf_pct", "desc"],
        ["WAR", "war", "desc"],
      ];

  return (
    <aside
      className="sc-player-drawer"
      aria-label={`${player.name} details`}
    >
      <button
        type="button"
        className="sc-player-drawer-close"
        onClick={onClose}
        aria-label="Close player details"
      >
        ×
      </button>

      <div className="sc-player-drawer-identity">
        <PlayerAvatar player={player} />

        <div>
          <span>
            {scope === "league"
              ? getPlayerTeamLabel(player)
              : player.position}
          </span>
          <h2>{player.name}</h2>
          <p>
            {player.position}
            {player.age
              ? ` · ${player.age} years`
              : ""}
            {player.role_label
              ? ` · ${player.role_label}`
              : ""}
          </p>
        </div>
      </div>

      <div className="sc-player-drawer-actions">
        <button
          type="button"
          className={
            isPinned ? "is-active" : ""
          }
          onClick={onTogglePin}
        >
          {isPinned ? "Pinned" : "Pin Player"}
        </button>

        <button
          type="button"
          onClick={onCompare}
        >
          Compare
        </button>
      </div>

      <div className="sc-player-drawer-metrics">
        {isGoalie ? (
          <>
            <DrawerMetric
              label="GP"
              value={fmtZero(player.gp)}
            />
            <DrawerMetric
              label="SV%"
              value={fmtSavePct(
                player.sv_pct
              )}
            />
            <DrawerMetric
              label="GAA"
              value={fmtTwo(player.gaa)}
            />
            <DrawerMetric
              label="GSAx"
              value={fmtMaybeOne(
                player.gsax
              )}
            />
            <DrawerMetric
              label="QS%"
              value={fmtMaybePct(
                player.quality_start_pct
              )}
            />
            <DrawerMetric
              label="WAR"
              value={fmtMaybeTwo(player.war)}
            />
          </>
        ) : (
          <>
            <DrawerMetric
              label="GP"
              value={fmtZero(player.gp)}
            />
            <DrawerMetric
              label="PTS"
              value={fmtZero(player.pts)}
            />
            <DrawerMetric
              label="P/GP"
              value={fmtTwo(
                player.points_per_game
              )}
            />
            <DrawerMetric
              label="TOI/GP"
              value={formatSmallTOI(player)}
            />
            <DrawerMetric
              label="xGF%"
              value={fmtMaybePct(
                player.xgf_pct
              )}
            />
            <DrawerMetric
              label="WAR"
              value={fmtMaybeTwo(player.war)}
            />
          </>
        )}
      </div>

      <section className="sc-player-drawer-percentiles">
        <header>
          <span>Scope Percentiles</span>
          <strong>
            {comparisonPool.length} players
          </strong>
        </header>

        {percentileRows.map(
          ([label, key, direction]) => {
            const percentile =
              getPlayerPercentile(
                comparisonPool,
                player,
                key,
                direction
              );

            return (
              <div key={key}>
                <span>{label}</span>
                <div>
                  <i
                    style={{
                      width: `${
                        percentile || 0
                      }%`,
                    }}
                  />
                </div>
                <strong>
                  {percentile === null
                    ? "—"
                    : `${percentile}th`}
                </strong>
              </div>
            );
          }
        )}
      </section>
    </aside>
  );
}

function DrawerMetric({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SpecialTeamsTab({
  players,
  scope,
  density,
  selectedPlayerId,
  onSelectPlayer,
  onComparePlayer,
}) {
  const ppRows = useMemo(
    () =>
      [...players]
        .filter(
          (player) =>
            safe(player.pp_toi_sec, 0) > 0 ||
            safe(player.pp_points, 0) > 0
        )
        .sort((a, b) => {
          const pointDifference =
            safe(b.pp_points, 0) -
            safe(a.pp_points, 0);

          if (pointDifference !== 0) {
            return pointDifference;
          }

          return (
            safe(b.pp_toi_sec, 0) -
            safe(a.pp_toi_sec, 0)
          );
        })
        .slice(0, scope === "league" ? 16 : 10),
    [players, scope]
  );

  const pkRows = useMemo(
    () =>
      [...players]
        .filter(
          (player) =>
            safe(player.pk_toi_sec, 0) > 0 ||
            safe(player.sh_points, 0) > 0
        )
        .sort(
          (a, b) =>
            safe(b.pk_toi_sec, 0) -
            safe(a.pk_toi_sec, 0)
        )
        .slice(0, scope === "league" ? 16 : 10),
    [players, scope]
  );

  const actionColumn = {
    label: "",
    key: "actions",
    className: "is-actions-col",
    render: (row) => (
      <button
        type="button"
        className="sc-inline-compare"
        onClick={(event) =>
          stopRowAction(event, () =>
            onComparePlayer(row)
          )
        }
      >
        Compare
      </button>
    ),
  };

  const ppColumns = [
    {
      label: "Player",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope={scope}
        />
      ),
    },
    {
      label: "GP",
      key: "gp",
      align: "right",
    },
    {
      label: "PP PTS",
      align: "right",
      render: (row) =>
        fmtZero(row.pp_points),
    },
    {
      label: "PPG",
      align: "right",
      render: (row) => fmtZero(row.ppg),
    },
    {
      label: "PPA",
      align: "right",
      render: (row) => fmtZero(row.ppa),
    },
    {
      label: "PP TOI/GP",
      align: "right",
      render: (row) =>
        formatTOISplit(
          row.pp_toi_sec,
          row.gp
        ),
    },
    actionColumn,
  ];

  const pkColumns = [
    {
      label: "Player",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope={scope}
        />
      ),
    },
    {
      label: "GP",
      key: "gp",
      align: "right",
    },
    {
      label: "PK TOI/GP",
      align: "right",
      render: (row) =>
        formatTOISplit(
          row.pk_toi_sec,
          row.gp
        ),
    },
    {
      label: "SHP",
      align: "right",
      render: (row) =>
        fmtZero(row.sh_points),
    },
    {
      label: "BLK",
      align: "right",
      render: (row) => fmtZero(row.blk),
    },
    {
      label: "TAK",
      align: "right",
      render: (row) =>
        fmtZero(row.takeaways),
    },
    actionColumn,
  ];

  return (
    <div className="sc-special-teams-page">
      <section className="sc-special-team-panel">
        <header>
          <span>POWER PLAY</span>
          <strong>Man-Advantage Production</strong>
          <em>{ppRows.length} qualified</em>
        </header>

        <DataTable
          columns={ppColumns}
          rows={ppRows}
          density={density}
          selectedRowId={selectedPlayerId}
          onRowClick={onSelectPlayer}
          empty="No power-play usage has been recorded."
        />
      </section>

      <section className="sc-special-team-panel">
        <header>
          <span>PENALTY KILL</span>
          <strong>Short-Handed Usage</strong>
          <em>{pkRows.length} qualified</em>
        </header>

        <DataTable
          columns={pkColumns}
          rows={pkRows}
          density={density}
          selectedRowId={selectedPlayerId}
          onRowClick={onSelectPlayer}
          empty="No penalty-kill usage has been recorded."
        />
      </section>
    </div>
  );
}


function LeagueLeadersPage({ data }) {
  const [view, setView] = useState("leaders");

  return (
    <div className="sc-league-leaders-workspace">
      <header className="sc-league-leaders-header">
        <div>
          <span>LEAGUE CENTRAL</span>
          <strong>
            {view === "leaders"
              ? "Category leaders"
              : "Award races"}
          </strong>
        </div>

        <nav aria-label="League leaders submenu">
          {LEAGUE_LEADER_VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={view === item.id ? "is-active" : ""}
              onClick={() => setView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <div className="sc-league-leaders-panel">
        {view === "leaders" ? (
          <LeadersTab data={data} />
        ) : (
          <AwardsTab data={data} />
        )}
      </div>
    </div>
  );
}


/* =========================================================
   SMALL REUSABLE COMPONENTS
========================================================= */

function StatCard({ label, value, sub, rank, tone = "", warning = "" }) {
  return (
    <article className={`sc-stat-card ${tone ? `is-${tone}` : ""} ${warning ? "has-warning" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {rank ? <b>{rank}</b> : null}
      {sub ? <em>{sub}</em> : null}
      {warning ? <i>{warning}</i> : null}
    </article>
  );
}
function Section({ eyebrow, title, right, children, className = "" }) {
  return (
    <section className={`sc-section ${className}`}>
      <header className="sc-section-head">
        <div>
          {eyebrow ? <p>{eyebrow}</p> : null}
          <h2>{title}</h2>
        </div>
        {right ? <div className="sc-section-right">{right}</div> : null}
      </header>

      <div className="sc-section-body">
        {children}
      </div>
    </section>
  );
}

function Pill({ children, tone = "" }) {
  return (
    <span className={`sc-pill ${tone ? `is-${tone}` : ""}`}>
      {children}
    </span>
  );
}

function getPlayerHeadshotSrc(player) {
  return pickString(
    player?.headshot,
    player?.headshot_url,
    player?.headshotUrl,
    player?.portrait,
    player?.portrait_url,
    player?.image,
    player?.image_url,
    player?.photo,
    player?.photo_url
  );
}

function PlayerAvatar({
  player,
  small = false,
  large = false,
  teams = [],
  franchiseState = null,
}) {
  const pos = normalizePosition(firstPresent(player?.position, player?.pos, "F"));
  const logoSrc =
    player?.team_logo_src ||
    getPlayerTeamLogoSrc(player, teams, franchiseState || undefined);
  const headshotSrc = getPlayerHeadshotSrc(player);

  return (
    <div
      className={`sc-avatar ${headshotSrc ? "sc-avatar--headshot" : "sc-avatar--fallback"} ${small ? "is-small" : ""} ${large ? "is-large" : ""}`}
      title={playerName(player)}
    >
      {headshotSrc ? <img src={headshotSrc} alt="" /> : <span>{pos}</span>}
      {logoSrc ? (
        <i className="sc-avatar-team-logo" title={String(player?.team_name || player?.team_id || player?.team || "")}>
          <img src={logoSrc} alt="" />
        </i>
      ) : null}
    </div>
  );
}

function PlayerNameCell({
  player,
  teams = [],
  franchiseState = null,
}) {
  const logoSrc =
    player?.team_logo_src ||
    getPlayerTeamLogoSrc(
      player,
      teams,
      franchiseState || undefined
    );

  const overall = getPlayerOverall(player);
  const baseOverall = getBaseOverall(player) || overall;
  const overallDrop = getOverallDrop(player);
  const position = normalizePosition(
    firstPresent(player?.position, player?.pos, "F")
  );
  const positionLabel = getPositionFullName(position);

  return (
    <div className="sc-name-cell sc-player-name-cell-simple">
      {logoSrc ? (
        <div className="sc-player-team-logo" title={getPlayerTeamLabel(player)}>
          <img src={logoSrc} alt="" />
        </div>
      ) : (
        <div className="sc-player-team-logo is-fallback" title={getPlayerTeamLabel(player)}>
          {String(getPlayerTeamLabel(player) || "TM")
            .slice(0, 2)
            .toUpperCase()}
        </div>
      )}

      <div className="sc-player-name-copy">
        <strong>{player?.name || "Unknown Player"}</strong>

        <span className="sc-player-meta-inline">
          <em
            className="sc-player-pos-inline"
            title={positionLabel}
            aria-label={positionLabel}
          >
            {position}
          </em>

          {overall > 0 ? (
            <em
              className={`sc-player-ovr-inline${overallDrop >= 1 ? " is-dropped" : ""}`}
              title={getOverallTooltip(player)}
              aria-label={getOverallTooltip(player)}
            >
              {fmtZero(overall)}
              {overallDrop >= 1 ? (
                <span className="sc-player-ovr-drop" title={`Down ${overallDrop} from base ${baseOverall}`}>
                  ↓{overallDrop}
                </span>
              ) : null}
            </em>
          ) : null}
        </span>
      </div>
    </div>
  );
}

function teamDisplayLabel(team) {
  const abbrev = pickString(team?.abbrev, team?.team_abbrev, team?.team_abbr);
  const name = String(team?.name || team?.team_name || "").trim();
  if (name && !/^\d+$/.test(name)) return name;
  return abbrev || String(team?.team_id || team?.id || "Team");
}

function TeamLogoMark({ team, size = "small" }) {
  const logoSrc = team?.team_logo_src || getTeamLogoSrc(team);
  const label = teamDisplayLabel(team);
  const abbrev = pickString(
    team?.abbrev,
    team?.team_abbrev,
    team?.team_abbr,
    team?.team_id
  );

  const sizeClass =
    size === "large"
      ? "is-large"
      : size === "table"
        ? "is-table"
        : "is-small";

  if (logoSrc) {
    return (
      <div
        className={`sc-team-logo-mark sc-team-logo-mark--logo ${sizeClass}`}
        title={label}
      >
        <img src={logoSrc} alt="" />
      </div>
    );
  }

  return (
    <div
      className={`sc-team-logo-mark sc-team-logo-mark--fallback ${sizeClass}`}
      title={label}
    >
      {String(abbrev || "TM")
        .slice(0, 3)
        .toUpperCase()}
    </div>
  );
}

function TeamNameCell({
  team,
  isUser = false,
  overallLeagueRank = 0,
}) {
  const label = teamDisplayLabel(team);

  const subtitle =
    team?.division && team?.division !== "League"
      ? `${team.division}${team?.conference ? ` · ${team.conference}` : ""}`
      : pickString(
          team?.team_abbrev,
          team?.abbrev,
          team?.team_id,
          "League"
        );

  return (
    <div
      className={`sc-name-cell sc-team-name-cell ${
        isUser ? "is-user-team" : ""
      }`}
    >
      <TeamLogoMark team={team} size="table" />

      <div className="sc-team-name-copy">
        <div className="sc-team-name-line">
          <strong>{label}</strong>

          {isUser ? (
            <em className="sc-my-team-tag">MY TEAM</em>
          ) : null}
        </div>

        {overallLeagueRank > 0 ? (
          <small
            className="sc-team-overall-rank"
            title="Overall league standings rank"
          >
            OVERALL LEAGUE #{overallLeagueRank}
          </small>
        ) : null}

        <span>{subtitle}</span>
      </div>
    </div>
  );
}

function PlayerMiniCard({ player, metric = "pts", label = "PTS", formatter, teams = [], franchiseState = null }) {
  if (!player) {
    return (
      <div className="sc-mini-player is-empty">
        <div className="sc-avatar is-small">—</div>
        <div>
          <strong>—</strong>
          <span>No player data yet</span>
        </div>
        <b>—<small>{label}</small></b>
      </div>
    );
  }

  const value = formatter ? formatter(player?.[metric], player) : formatMetricValue(metric, player?.[metric]);

  return (
    <div className="sc-mini-player">
      <PlayerAvatar player={player} small teams={teams} franchiseState={franchiseState} />
      <div>
        <strong>{player.name}</strong>
        <span>
          {player.team_abbrev || player.team_name || player.team_id || player.team || "—"} ·{" "}
          {player.position || "F"} · {player.role_label || player.analytics_archetype || "Regular"}
        </span>
      </div>
      <b>
        {value}
        <small>{label}</small>
      </b>
    </div>
  );
}

function formatMetricValue(metric, value) {
  if (metric === "sv_pct") return fmtSavePct(value);
  if (metric === "gaa") return fmtTwo(value);

  if (
    metric === "war" ||
    metric === "watr" ||
    metric === "total_impact"
  ) {
    return fmtMaybeTwo(value);
  }

  if (
    metric === "analytics_rating" ||
    metric === "impact_score"
  ) {
    return fmtMaybeOne(value);
  }

  if (metric === "gsax") return fmtMaybeOne(value);

  if (
    metric === "xgf" ||
    metric === "xga" ||
    metric === "ixg" ||
    metric === "xa"
  ) {
    return fmtOne(value);
  }

  if (
    metric.includes("pct") ||
    metric.includes("_pct")
  ) {
    return fmtMaybePct(value);
  }

  return fmtZero(value);
}

const COLUMN_FULL_NAMES = Object.freeze({
  RK: "Rank",
  Team: "Team",
  Player: "Player",
  Goalie: "Goalie",

  GP: "Games Played",
  GS: "Games Started",
  W: "Wins",
  L: "Losses",
  OTL: "Overtime Losses",
  PTS: "Points",
  "PTS%": "Points Percentage",
  DIFF: "Goal Differential",

  GF: "Goals For",
  "GF/GP": "Goals For Per Game",
  GA: "Goals Against",
  "GA/GP": "Goals Against Per Game",
  SF: "Shots For",
  "SF/GP": "Shots For Per Game",
  SA: "Shots Against",
  "SA/GP": "Shots Against Per Game",

  "SH%": "Shooting Percentage",
  "SV%": "Save Percentage",
  SV: "Saves",
  GAA: "Goals Against Average",
  SO: "Shutouts",
  GSAx: "Goals Saved Above Expected",
  "QS%": "Quality Start Percentage",

  "PP%": "Power Play Percentage",
  PPG: "Power Play Goals",
  PPO: "Power Play Opportunities",
  "PK%": "Penalty Kill Percentage",
  PPGA: "Power Play Goals Against",
  TSH: "Times Shorthanded",

  "CF%": "Corsi For Percentage",
  "FF%": "Fenwick For Percentage",
  "xGF%": "Expected Goals For Percentage",
  xGF: "Expected Goals For",
  xGA: "Expected Goals Against",
  PDO: "Shooting Percentage Plus Save Percentage",

  OVR: "Overall Rating",
  G: "Goals",
  A: "Assists",
  "P/GP": "Points Per Game",
  "P/60": "Points Per 60 Minutes",
  "G/60": "Goals Per 60 Minutes",
  "A/60": "Assists Per 60 Minutes",
  SOG: "Shots on Goal",
  "GF%": "Goals For Percentage",
  PP: "Power Play Points",
  PK: "Penalty Kill Time on Ice",
  "FO%": "Faceoff Percentage",
  HIT: "Hits",
  BLK: "Blocked Shots",
  "TAK/GIV": "Takeaways and Giveaways",
  PIM: "Penalty Minutes",
  "TOI/GP": "Time on Ice Per Game",
  WAR: "Wins Above Replacement",
  iXG: "Individual Expected Goals",
  xA: "Expected Assists",
});

function getColumnFullName(column) {
  return (
    column?.fullLabel ||
    COLUMN_FULL_NAMES[column?.label] ||
    column?.label ||
    ""
  );
}

function DataTable({
  columns,
  rows,
  empty = "No rows found.",
  sortKey = "",
  sortDir = "desc",
  rowClassName,
  tableClassName = "",
  onRowClick = null,
  getRowId = null,
  selectedRowId = "",
  rowAriaLabel = null,
  density = "compact",
}) {
  const resolveRowId = (row, rowIndex) => {
    if (typeof getRowId === "function") {
      return String(getRowId(row, rowIndex));
    }

    return String(
      row?.player_id ||
        row?.team_id ||
        row?.id ||
        rowIndex
    );
  };

  return (
    <div
      className={`sc-table-wrap is-${density}`}
      role="region"
      aria-label="Statistics table"
    >
      <table className={`sc-table ${tableClassName}`}>
        <thead>
          <tr>
            {columns.map((column) => {
              const isSortable =
                typeof column.onClick === "function";
              const fullColumnName =
                getColumnFullName(column);
              const sortArrow = column.sortKey
                ? getSortArrow(
                    sortKey,
                    sortDir,
                    column.sortKey
                  )
                : "";

              return (
                <th
                  key={column.key || column.label}
                  title={fullColumnName}
                  aria-label={fullColumnName}
                  aria-sort={
                    column.sortKey === sortKey
                      ? sortDir === "desc"
                        ? "descending"
                        : "ascending"
                      : "none"
                  }
                  className={[
                    column.align === "right"
                      ? "is-right"
                      : "",
                    column.className || "",
                    isSortable ? "is-sortable" : "",
                    column.sortKey === sortKey
                      ? "is-sorted"
                      : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={
                    column.onClick || undefined
                  }
                  role={
                    isSortable ? "button" : undefined
                  }
                  tabIndex={isSortable ? 0 : undefined}
                  onKeyDown={
                    isSortable
                      ? (event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === " "
                          ) {
                            event.preventDefault();
                            column.onClick();
                          }
                        }
                      : undefined
                  }
                >
                  <span>
                    {column.label}
                    {sortArrow ? (
                      <em>{sortArrow}</em>
                    ) : null}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>

        <tbody>
          {rows?.length ? (
            rows.map((row, rowIndex) => {
              const rowId = resolveRowId(
                row,
                rowIndex
              );
              const isInteractive =
                typeof onRowClick === "function";
              const isSelected =
                selectedRowId !== "" &&
                String(selectedRowId) === rowId;
              const customClass =
                typeof rowClassName === "function"
                  ? rowClassName(row, rowIndex)
                  : "";

              return (
                <tr
                  key={rowId}
                  className={[
                    customClass,
                    isInteractive
                      ? "is-interactive-row"
                      : "",
                    isSelected
                      ? "is-selected-row"
                      : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={
                    isInteractive
                      ? () =>
                          onRowClick(row, rowIndex)
                      : undefined
                  }
                  tabIndex={
                    isInteractive ? 0 : undefined
                  }
                  aria-selected={
                    isInteractive
                      ? isSelected
                      : undefined
                  }
                  aria-label={
                    typeof rowAriaLabel === "function"
                      ? rowAriaLabel(row, rowIndex)
                      : undefined
                  }
                  onKeyDown={
                    isInteractive
                      ? (event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === " "
                          ) {
                            event.preventDefault();
                            onRowClick(
                              row,
                              rowIndex
                            );
                          }
                        }
                      : undefined
                  }
                >
                  {columns.map((column) => {
                    const value =
                      typeof column.render ===
                      "function"
                        ? column.render(
                            row,
                            rowIndex
                          )
                        : row[column.key];

                    return (
                      <td
                        key={
                          column.key ||
                          column.label
                        }
                        className={[
                          column.align === "right"
                            ? "is-right"
                            : "",
                          column.className || "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                      >
                        {value}
                      </td>
                    );
                  })}
                </tr>
              );
            })
          ) : (
            <tr>
              <td
                className="is-empty"
                colSpan={columns.length}
              >
                {empty}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
/* =========================================================
   OVERVIEW TAB
========================================================= */

function OverviewTab({
  data,
  topScorer,
  topGoalScorer,
  topPlaymaker,
  topImpact,
  topXg,
  topGoalie,
  topGsaxGoalie,
  selectedDay,
  setSelectedDay,
  gamesForSelectedDay,
}) {
  const team = data.team || {};
  const totalTeams = team.total_league_teams || data.teams.length || 32;

  const ppMissing = team.analytics_missing?.pp_pct;
  const pkMissing = team.analytics_missing?.pk_pct;
  const cfMissing = team.analytics_missing?.cf_pct;
  const xgfMissing = team.analytics_missing?.xgf_pct;

  const teamName = team.name || team.team_id || "Franchise";
  const recordText = `${team.wins || 0}-${team.losses || 0}-${team.otl || 0}`;
  const pointsText = `${team.points || 0} points`;

  const leagueRankText = team.league_rank
    ? `League #${team.league_rank} of ${totalTeams}`
    : "League rank unavailable";

  const divisionText = team.division_rank
    ? `Division #${team.division_rank}`
    : "Division rank unavailable";

  const conferenceText = team.conference_rank
    ? `Conference #${team.conference_rank}`
    : "Conference rank unavailable";

  const formTone =
    team.league_rank && team.league_rank <= 8
      ? "good"
      : team.league_rank && team.league_rank >= 24
        ? "bad"
        : "";

  const offenseTone =
    team.gf_league_rank && team.gf_league_rank <= 10
      ? "good"
      : team.gf_league_rank && team.gf_league_rank >= 24
        ? "bad"
        : "";

  const defenseTone =
    team.ga_league_rank && team.ga_league_rank <= 10
      ? "good"
      : team.ga_league_rank && team.ga_league_rank >= 24
        ? "bad"
        : "";

  const diffTone = team.goal_diff > 0 ? "good" : team.goal_diff < 0 ? "bad" : "";

  const possessionTone =
    team.cf_pct >= 0.52 || team.xgf_pct >= 0.52
      ? "good"
      : team.cf_pct && team.cf_pct <= 0.47
        ? "bad"
        : "";

  const goalieTone =
    team.sv_pct >= 0.91
      ? "good"
      : team.sv_pct && team.sv_pct < 0.89
        ? "bad"
        : "";

  const latestCalendarDays = Array.isArray(data.calendar)
    ? data.calendar.slice(-6).reverse()
    : [];

  return (
    <div className="sc-overview-fixed">
      <section className="sc-overview-fixed-hero">
        <div className="sc-overview-fixed-title">
          <p>FRANCHISE SNAPSHOT</p>
          <h2>{teamName}</h2>
          <span>
            {recordText} · {pointsText} · {divisionText} · {conferenceText}
          </span>
        </div>

        <div className={`sc-overview-fixed-points ${formTone ? `is-${formTone}` : ""}`}>
          <strong>{team.points || 0}</strong>
          <span>PTS</span>
          <em>{leagueRankText}</em>
        </div>
      </section>

      <section className="sc-overview-fixed-story">
        <OverviewStoryCard
          label="Offense"
          value={fmtMaybeNumber(team.gf)}
          sub="Goals For"
          rank={formatLeagueRank(team.gf_league_rank, totalTeams)}
          tone={offenseTone}
        />

        <OverviewStoryCard
          label="Defense"
          value={fmtMaybeNumber(team.ga)}
          sub="Goals Against"
          rank={formatLeagueRank(team.ga_league_rank, totalTeams)}
          tone={defenseTone}
        />

        <OverviewStoryCard
          label="Goal Diff"
          value={formatSigned(team.goal_diff)}
          sub="GF minus GA"
          rank={formatLeagueRank(team.goal_diff_league_rank, totalTeams)}
          tone={diffTone}
        />

        <OverviewStoryCard
          label="Puck Control"
          value={fmtMaybePct(team.cf_pct)}
          sub="CF%"
          rank={cfMissing ? "Needs CF/CA ledger" : formatLeagueRank(team.cf_pct_league_rank, totalTeams)}
          tone={possessionTone}
          warning={cfMissing}
        />

        <OverviewStoryCard
          label="Chance Quality"
          value={fmtMaybePct(team.xgf_pct)}
          sub="xGF%"
          rank={xgfMissing ? "Needs xGF/xGA ledger" : formatLeagueRank(team.xgf_pct_league_rank, totalTeams)}
          tone={possessionTone}
          warning={xgfMissing}
        />

        <OverviewStoryCard
          label="Goaltending"
          value={fmtMaybePct(team.sv_pct, 3)}
          sub="Team SV%"
          rank={formatLeagueRank(team.ga_league_rank, totalTeams)}
          tone={goalieTone}
        />
      </section>

      <section className="sc-overview-fixed-main">
        <div className="sc-overview-fixed-panel is-leaders">
          <header>
            <p>TEAM LEADERS</p>
            <h3>Main Drivers</h3>
          </header>

          <div className="sc-overview-fixed-driver-grid">
            <PlayerMiniCard player={topScorer} metric="pts" label="PTS" teams={data.teams} franchiseState={data.franchiseState} />
            <PlayerMiniCard player={topGoalScorer} metric="g" label="G" teams={data.teams} franchiseState={data.franchiseState} />
            <PlayerMiniCard player={topPlaymaker} metric="a" label="A" teams={data.teams} franchiseState={data.franchiseState} />
            <PlayerMiniCard player={topImpact} metric="war" label="WAR" teams={data.teams} franchiseState={data.franchiseState} />
            <PlayerMiniCard player={topXg} metric="ixg" label="iXG" teams={data.teams} franchiseState={data.franchiseState} />
          </div>
        </div>

        <div className="sc-overview-fixed-panel is-goalie">
          <header>
            <p>CREASE REPORT</p>
            <h3>Goalie Snapshot</h3>
          </header>

          <div className="sc-overview-fixed-driver-grid">
            <PlayerMiniCard player={topGoalie} metric="sv_pct" label="SV%" teams={data.teams} franchiseState={data.franchiseState} />
            <PlayerMiniCard player={topGsaxGoalie} metric="gsax" label="GSAx" teams={data.teams} franchiseState={data.franchiseState} />

            <div className="sc-overview-fixed-mini-metric">
              <span>Goals Against</span>
              <strong>{fmtMaybeNumber(team.ga)}</strong>
              <em>{formatLeagueRank(team.ga_league_rank, totalTeams)}</em>
            </div>

            <div className="sc-overview-fixed-mini-metric">
              <span>PDO</span>
              <strong>{fmtPdo(team.pdo)}</strong>
              <em>{team.analytics_missing?.pdo ? "Needs shots/saves ledger" : "SH% + SV%"}</em>
            </div>
          </div>
        </div>
      </section>

      <section className="sc-overview-fixed-bottom">
        <div className="sc-overview-fixed-panel is-special">
          <header>
            <p>SPECIAL TEAMS</p>
            <h3>PP / PK</h3>
          </header>

          <div className="sc-overview-fixed-special-grid">
            <div>
              <span>Power Play</span>
              <strong>{fmtMaybePct(team.pp_pct)}</strong>
              <em>{ppMissing ? "Needs PP opportunities" : formatLeagueRank(team.pp_pct_league_rank, totalTeams)}</em>
            </div>

            <div>
              <span>Penalty Kill</span>
              <strong>{fmtMaybePct(team.pk_pct)}</strong>
              <em>{pkMissing ? "Needs PK chances" : formatLeagueRank(team.pk_pct_league_rank, totalTeams)}</em>
            </div>
          </div>
        </div>

        <div className="sc-overview-fixed-panel is-calendar">
          <header>
            <p>CALENDAR</p>
            <h3>Recent Game Nights</h3>
          </header>

          <div className="sc-overview-fixed-calendar">
            <button
              type="button"
              className={selectedDay === null ? "is-active" : ""}
              onClick={() => setSelectedDay(null)}
            >
              <span>Latest</span>
              <em>{data.recentGames.length}</em>
            </button>

            {latestCalendarDays.map((day) => (
              <button
                key={`day-${day.day}`}
                type="button"
                className={Number(selectedDay) === Number(day.day) ? "is-active" : ""}
                onClick={() => setSelectedDay(day.day)}
              >
                <span>
                  Day {day.day}
                  {day.segment ? <small>{day.segment}</small> : null}
                </span>
                <em>{(day.games || []).length}</em>
              </button>
            ))}
          </div>
        </div>

        <div className="sc-overview-fixed-panel is-scores">
          <header>
            <p>SCORES</p>
            <h3>{selectedDay === null ? "Latest Finals" : `Day ${selectedDay} Finals`}</h3>
          </header>

          <GameScoreList games={gamesForSelectedDay} />
        </div>
      </section>
    </div>
  );
}

function OverviewStoryCard({ label, value, sub, rank, tone = "", warning = false }) {
  return (
    <article className={`sc-overview-fixed-card ${tone ? `is-${tone}` : ""} ${warning ? "has-warning" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <em>{sub}</em>
      <b>{rank}</b>
    </article>
  );
}
function GameScoreList({ games }) {
  const rows = Array.isArray(games) ? games : [];

  if (!rows.length) {
    return (
      <div className="sc-empty">
        No game results yet. Advance the calendar and games will appear here.
      </div>
    );
  }

  return (
    <div className="sc-score-list">
      {rows.map((game, index) => (
        <article key={game.game_id || index} className="sc-score-card">
          <div className="sc-score-line">
            <span className="sc-score-team">{game.home_name || game.home_id}</span>
            <strong>{game.home_goals}</strong>
            <em>—</em>
            <strong>{game.away_goals}</strong>
            <span className="sc-score-team is-away">{game.away_name || game.away_id}</span>
          </div>

          <div className="sc-score-meta">
            Day {game.day} · {fmtScore(game)}
          </div>

          <div className="sc-score-micro">
            <span>Shots {game.home_shots}-{game.away_shots}</span>
            <span>xG {fmtTwo(game.home_xg)}-{fmtTwo(game.away_xg)}</span>
          </div>
        </article>
      ))}
    </div>
  );
}

function getSortArrow(sortKey, sortDir, key) {
  if (sortKey !== key) return "";
  return sortDir === "desc" ? " ↓" : " ↑";
}

function getSkaterRankMap(players, key, direction = "desc") {
  const ranked = sortRows(players, key, direction);
  const map = new Map();

  ranked.forEach((player, index) => {
    map.set(String(player.player_id || player.id || player.name), index + 1);
  });

  return map;
}

function getPlayerRank(player, rankMap) {
  return rankMap.get(String(player.player_id || player.id || player.name)) || 0;
}

function formatRank(rank, total) {
  if (!rank) return "—";
  return `#${rank}${total ? ` / ${total}` : ""}`;
}

function getMetricTone(metric, value, row = {}) {
  const n = safe(value, 0);

  if (metric === "pts" || metric === "g" || metric === "a" || metric === "p60") {
    if (n >= 1.0) return "elite";
    if (n >= 0.7) return "good";
    if (n >= 0.4) return "neutral";
    return "warn";
  }

  if (metric === "cf_pct" || metric === "xgf_pct" || metric === "gf_pct") {
    const pctValue = normalizePct(value, 0);
    if (pctValue >= 0.55) return "elite";
    if (pctValue >= 0.52) return "good";
    if (pctValue >= 0.48) return "neutral";
    if (pctValue >= 0.45) return "warn";
    return "bad";
  }

  if (metric === "shooting_pct") {
    const pctValue = normalizePct(value, 0);
    if (pctValue >= 0.18) return "elite";
    if (pctValue >= 0.13) return "good";
    if (pctValue >= 0.08) return "neutral";
    return "warn";
  }

  if (metric === "war" || metric === "watr" || metric === "total_impact") {
    if (n >= 3.0) return "elite";
    if (n >= 1.5) return "good";
    if (n >= 0.2) return "neutral";
    if (n >= -0.5) return "warn";
    return "bad";
  }

  if (metric === "discipline") {
    if (n >= 60) return "bad";
    if (n >= 35) return "warn";
    return "neutral";
  }

  if (metric === "toi") {
    if (n >= 21) return "elite";
    if (n >= 18) return "good";
    if (n >= 13) return "neutral";
    return "depth";
  }

  return "";
}

function getSkaterIdentityLabel(row) {
  const position = normalizePosition(row?.position || row?.pos);
  const ppg = perGame(row?.pts, row?.gp);
  const g82 = per82(row?.g, row?.gp);
  const pts82 = per82(row?.pts, row?.gp);
  const avgToi = getAverageTOIMinutes(row);
  const xgfPct = normalizePct(row?.xgf_pct, 0);
  const cfPct = normalizePct(row?.cf_pct, 0);
  const impact = safe(row?.analytics_rating, 0);
  const hits = safe(row?.hit, 0);
  const blocks = safe(row?.blk, 0);
  const takeaways = safe(row?.takeaways, 0);
  const ppPoints = safe(row?.pp_points, 0);
  const shPoints = safe(row?.sh_points, 0);
  const pkToi = safe(row?.pk_toi_sec, 0);

  if (impact >= 82 && ppg >= 1) return "Franchise Driver";
  if (pts82 >= 90) return "Elite Scoring Star";
  if (g82 >= 40) return "Pure Goal Scorer";
  if (ppPoints >= 12 && ppg >= 0.65) return "Power-Play Weapon";
  if (position === "D" && avgToi >= 22 && xgfPct >= 0.5) return "Top-Pair Driver";
  if (position === "D" && blocks + hits >= 80) return "Shutdown Defender";
  if (xgfPct >= 0.54 && cfPct >= 0.52 && ppg >= 0.55) return "Two-Way Driver";
  if (takeaways >= 25 && xgfPct >= 0.5) return "Puck-Hound Creator";
  if (pkToi > 0 && shPoints > 0) return "PK Threat";
  if (hits >= 70) return "Physical Forechecker";
  if (ppg >= 0.5) return "Middle-Six Producer";
  if (avgToi >= 15) return "Reliable Regular";
  return "Depth Contributor";
}

function getSkaterRowClass(row) {
  const impact = safe(row?.analytics_rating, 0);
  const ppg = perGame(row?.pts, row?.gp);
  const avgToi = getAverageTOIMinutes(row);

  if (impact >= 82 || ppg >= 1) return "is-star-row";
  if (avgToi >= 20) return "is-usage-row";
  if (safe(row?.rookie, 0) || row?.is_rookie) return "is-rookie-row";
  return "";
}

function SkaterStatCell({ value, sub = "", tone = "", title = "" }) {
  return (
    <div className={`sc-skater-stat ${tone ? `is-${tone}` : ""}`} title={title || undefined}>
      <strong>{value}</strong>
      {sub ? <span>{sub}</span> : null}
    </div>
  );
}

function SkaterIdentityChip({ player }) {
  const label = getSkaterIdentityLabel(player);
  const tone = getAnalyticsTone(player.analytics_rating);

  return (
    <span className={`sc-skater-role-chip is-${tone}`}>
      {label}
    </span>
  );
}

function SkaterQuickCard({ label, value, sub, tone = "" }) {
  return (
    <article className={`sc-skater-quick-card ${tone ? `is-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <em>{sub}</em>
    </article>
  );
}

function hasUsefulColumnData(players, key, minPositiveRows = 1) {
  const rows = Array.isArray(players) ? players : [];
  return rows.filter((row) => safe(row?.[key], 0) > 0).length >= minPositiveRows;
}

function hasUsefulPctData(players, key, minRows = 1) {
  const rows = Array.isArray(players) ? players : [];
  return rows.filter((row) => hasRealPct(row?.[key]) && normalizePct(row?.[key], 0) > 0).length >= minRows;
}

function hasFaceoffData(players) {
  return (players || []).some((p) => safe(p.fow, 0) + safe(p.fol, 0) > 0);
}

function hasTakeGiveData(players) {
  return (players || []).some((p) => safe(p.takeaways, 0) > 0 || safe(p.giveaways, 0) > 0);
}

function hasSpecialTeamsData(players) {
  return (players || []).some(
    (p) =>
      safe(p.pp_points, 0) > 0 ||
      safe(p.pp_toi_sec, 0) > 0 ||
      safe(p.pk_toi_sec, 0) > 0 ||
      safe(p.sh_points, 0) > 0
  );
}

function getActiveColumnPresetLabel(view) {
  if (view === "core") return "Core";
  if (view === "scoring") return "Scoring";
  if (view === "analytics") return "Analytics";
  if (view === "usage") return "Usage";
  return "All";
}

function shouldShowSkaterColumn(column, view, players) {
  if (!column.viewGroups || column.viewGroups.includes("always")) return true;
  if (!column.viewGroups.includes(view) && view !== "all") return false;

  if (column.requiredData === "age" && !hasUsefulColumnData(players, "age", 1)) return false;
  if (column.requiredData === "overall" && !hasUsefulColumnData(players, "overall", 1)) return false;
  if (column.requiredData === "faceoffs" && !hasFaceoffData(players)) return false;
  if (column.requiredData === "takegive" && !hasTakeGiveData(players)) return false;
  if (column.requiredData === "specialTeams" && !hasSpecialTeamsData(players)) return false;
  if (column.requiredData === "cf_pct" && !hasUsefulPctData(players, "cf_pct", 1)) return false;
  if (column.requiredData === "xgf_pct" && !hasUsefulPctData(players, "xgf_pct", 1)) return false;
  if (column.requiredData === "gf_pct" && !hasUsefulPctData(players, "gf_pct", 1)) return false;

  return true;
}

function getStablePctTone(metric, pctValue, sampleSize = 0) {
  const pctValueNorm = normalizePct(pctValue, 0);

  if (sampleSize > 0 && sampleSize < 5) return "sample";

  if (metric === "gf_pct" && sampleSize > 0 && sampleSize < 8) return "sample";

  if (pctValueNorm >= 0.55) return "elite";
  if (pctValueNorm >= 0.52) return "good";
  if (pctValueNorm >= 0.48) return "neutral";
  if (pctValueNorm >= 0.45) return "warn";
  return "bad";
}

function getGFPercentSample(row) {
  return safe(row?.gf_on, 0) + safe(row?.ga_on, 0);
}

function formatGFPercentSub(row) {
  const sample = getGFPercentSample(row);
  if (sample <= 0) return "No GF sample";
  if (sample < 5) return `Tiny sample ${fmtZero(row.gf_on)}-${fmtZero(row.ga_on)}`;
  return `${fmtZero(row.gf_on)}-${fmtZero(row.ga_on)}`;
}

function getSkaterIdentityLabelV2(row, context = {}) {
  const backendLabel = pickString(row?.role_label, row?.analytics_archetype, row?.impact_tier, "");
  if (backendLabel) return backendLabel;

  const rank = context.rank || 99;
  const totalPlayers = context.totalPlayers || 18;
  const position = normalizePosition(row?.position || row?.pos);

  const ppg = perGame(row?.pts, row?.gp);
  const g82 = per82(row?.g, row?.gp);
  const pts82 = per82(row?.pts, row?.gp);
  const avgToi = getAverageTOIMinutes(row);
  const xgfPct = normalizePct(row?.xgf_pct, 0);
  const cfPct = normalizePct(row?.cf_pct, 0);
  const impact = safe(row?.analytics_rating, 0);
  const hits = safe(row?.hit, 0);
  const blocks = safe(row?.blk, 0);
  const takeaways = safe(row?.takeaways, 0);
  const ppPoints = safe(row?.pp_points, 0);
  const pkToi = safe(row?.pk_toi_sec, 0);

  /*
    Franchise Driver must be rare.
    Earlier logic made half the roster Franchise Drivers, which killed meaning.
  */
  if (rank <= 2 && impact >= 86 && ppg >= 0.9) return "Franchise Driver";
  if (rank <= 3 && pts82 >= 85) return "Elite Scoring Star";
  if (g82 >= 38) return "Pure Goal Scorer";
  if (ppPoints >= 10 && ppg >= 0.6) return "Power-Play Weapon";

  if (position === "D" && avgToi >= 21 && xgfPct >= 0.5) return "Top-Pair Driver";
  if (position === "D" && blocks + hits >= 75) return "Shutdown Defender";

  if (xgfPct >= 0.54 && cfPct >= 0.52 && ppg >= 0.5) return "Two-Way Driver";
  if (takeaways >= 20 && xgfPct >= 0.5) return "Puck-Hound Creator";
  if (pkToi > 0 && avgToi >= 13) return "PK Regular";
  if (hits >= 65) return "Physical Forechecker";
  if (ppg >= 0.55) return "Middle-Six Producer";
  if (avgToi >= 14) return "Reliable Regular";
  if (rank <= Math.ceil(totalPlayers * 0.75)) return "Depth Contributor";
  return "Replacement Level";
}

function SkaterIdentityChipV2({ player, rank, totalPlayers }) {
  const label = getSkaterIdentityLabelV2(player, { rank, totalPlayers });
  const tone =
    label === "Franchise Driver" || label === "Elite Scoring Star"
      ? "elite"
      : label.includes("Driver") || label.includes("Weapon") || label.includes("Two-Way")
        ? "good"
        : label.includes("Replacement")
          ? "bad"
          : label.includes("Depth")
            ? "warn"
            : "neutral";

  return (
    <span className={`sc-skater-role-chip is-${tone}`}>
      {label}
    </span>
  );
}

function SkaterSummaryChip({ label, value, sub, tone = "" }) {
  return (
    <article className={`sc-skater-summary-chip ${tone ? `is-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <em>{sub}</em>
    </article>
  );
}

/* =========================================================
   PLAYERS TAB
========================================================= */

function PlayersTab({
  players,
  sortKey,
  sortDir,
  changeSort,
  scope = "team",
  density = "compact",
  page = 1,
  onPageChange,
  selectedPlayerId = "",
  onSelectPlayer,
  pinnedIds = [],
  onTogglePin,
  onComparePlayer,
}) {
  const [view, setView] = useState("core");
  const totalPlayers = players.length;

  const activeRankMap = useMemo(
    () =>
      getSkaterRankMap(
        players,
        sortKey,
        sortDir
      ),
    [players, sortKey, sortDir]
  );

  const presets = [
    ["core", "Core"],
    ["scoring", "Scoring"],
    ["analytics", "Analytics"],
    ["usage", "Usage"],
    ["contract", "Roster"],
  ];

  const rankSub = (row, key) => {
    if (sortKey !== key) return "";

    return formatRank(
      getPlayerRank(row, activeRankMap),
      totalPlayers
    );
  };

  const actionColumn = {
    label: "",
    key: "actions",
    className: "is-actions-col",
    render: (row) => {
      const id = String(
        row.player_id || row.id
      );

      return (
        <PlayerRowActions
          player={row}
          isPinned={pinnedIds.includes(id)}
          onTogglePin={() =>
            onTogglePin(row)
          }
          onCompare={() =>
            onComparePlayer(row)
          }
        />
      );
    },
  };

  const allColumns = {
    player: {
      label: "Player",
      sortKey: "name",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope={scope}
        />
      ),
      onClick: () =>
        changeSort("name", "asc"),
    },

    gp: {
      label: "GP",
      key: "gp",
      sortKey: "gp",
      align: "right",
      onClick: () => changeSort("gp"),
    },

    g: {
      label: "G",
      sortKey: "g",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.g)}
          sub={rankSub(row, "g")}
          tone={getMetricTone(
            "g",
            perGame(row.g, row.gp),
            row
          )}
        />
      ),
      onClick: () => changeSort("g"),
    },

    a: {
      label: "A",
      sortKey: "a",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.a)}
          sub={rankSub(row, "a")}
        />
      ),
      onClick: () => changeSort("a"),
    },

    pts: {
      label: "PTS",
      sortKey: "pts",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.pts)}
          sub={rankSub(row, "pts")}
          tone={getMetricTone(
            "pts",
            perGame(row.pts, row.gp),
            row
          )}
        />
      ),
      onClick: () => changeSort("pts"),
    },

    ppg: {
      label: "P/GP",
      sortKey: "points_per_game",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtTwo(
            row.points_per_game
          )}
          sub={
            sortKey === "points_per_game"
              ? rankSub(
                  row,
                  "points_per_game"
                )
              : ""
          }
          tone={getMetricTone(
            "pts",
            row.points_per_game,
            row
          )}
        />
      ),
      onClick: () =>
        changeSort("points_per_game"),
    },

    toi: {
      label: "TOI/GP",
      sortKey: "toi",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={formatSmallTOI(row)}
          sub={rankSub(row, "toi")}
          tone={getMetricTone(
            "toi",
            getAverageTOIMinutes(row),
            row
          )}
        />
      ),
      onClick: () => changeSort("toi"),
    },

    sog: {
      label: "SOG",
      sortKey: "sog",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.sog)}
          sub={
            sortKey === "sog"
              ? rankSub(row, "sog")
              : `${fmtOne(
                  perGame(
                    row.sog,
                    row.gp
                  )
                )}/GP`
          }
        />
      ),
      onClick: () => changeSort("sog"),
    },

    shootingPct: {
      label: "SH%",
      sortKey: "shooting_pct",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={
            safe(row.sog, 0) > 0
              ? fmtMaybePct(
                  row.shooting_pct
                )
              : "—"
          }
          sub={rankSub(
            row,
            "shooting_pct"
          )}
          tone={getMetricTone(
            "shooting_pct",
            row.shooting_pct,
            row
          )}
        />
      ),
      onClick: () =>
        changeSort("shooting_pct"),
    },

    ixg: {
      label: "iXG",
      sortKey: "ixg",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybeOne(row.ixg)}
          sub={rankSub(row, "ixg")}
        />
      ),
      onClick: () => changeSort("ixg"),
    },

    finish: {
      label: "Finish",
      sortKey: "finishing",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={
            hasRealNumber(row.finishing)
              ? formatSigned(
                  row.finishing,
                  1
                )
              : "—"
          }
          sub={rankSub(
            row,
            "finishing"
          )}
          tone={
            safe(row.finishing, 0) >= 2
              ? "elite"
              : safe(
                    row.finishing,
                    0
                  ) <= -2
                ? "warn"
                : ""
          }
        />
      ),
      onClick: () =>
        changeSort("finishing"),
    },

    p60: {
      label: "P/60",
      sortKey: "points_per_60",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtTwo(
            row.points_per_60
          )}
          sub={rankSub(
            row,
            "points_per_60"
          )}
        />
      ),
      onClick: () =>
        changeSort("points_per_60"),
    },

    cfPct: {
      label: "CF%",
      sortKey: "cf_pct",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybePct(row.cf_pct)}
          sub={rankSub(row, "cf_pct")}
          tone={getStablePctTone(
            "cf_pct",
            row.cf_pct,
            safe(row.cf, 0) +
              safe(row.ca, 0)
          )}
        />
      ),
      onClick: () =>
        changeSort("cf_pct"),
    },

    xgfPct: {
      label: "xGF%",
      sortKey: "xgf_pct",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybePct(
            row.xgf_pct
          )}
          sub={rankSub(row, "xgf_pct")}
          tone={getStablePctTone(
            "xgf_pct",
            row.xgf_pct,
            safe(row.xgf, 0) +
              safe(row.xga, 0)
          )}
        />
      ),
      onClick: () =>
        changeSort("xgf_pct"),
    },

    gfPct: {
      label: "GF%",
      sortKey: "gf_pct",
      align: "right",
      render: (row) => {
        const sample =
          getGFPercentSample(row);

        return (
          <SkaterStatCell
            value={
              sample > 0
                ? fmtMaybePct(row.gf_pct)
                : "—"
            }
            sub={
              sortKey === "gf_pct"
                ? rankSub(
                    row,
                    "gf_pct"
                  )
                : sample > 0
                  ? `${fmtZero(
                      row.gf_on
                    )}-${fmtZero(
                      row.ga_on
                    )}`
                  : ""
            }
            tone={getStablePctTone(
              "gf_pct",
              row.gf_pct,
              sample
            )}
          />
        );
      },
      onClick: () =>
        changeSort("gf_pct"),
    },

    war: {
      label: "WAR",
      sortKey: "war",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybeTwo(row.war)}
          sub={rankSub(row, "war")}
          tone={getMetricTone(
            "war",
            row.war,
            row
          )}
        />
      ),
      onClick: () => changeSort("war"),
    },

    ppToi: {
      label: "PP TOI",
      sortKey: "pp_toi_sec",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={formatTOISplit(
            row.pp_toi_sec,
            row.gp
          )}
          sub={
            safe(row.pp_points, 0) > 0
              ? `${fmtZero(
                  row.pp_points
                )} PPP`
              : ""
          }
        />
      ),
      onClick: () =>
        changeSort("pp_toi_sec"),
    },

    pkToi: {
      label: "PK TOI",
      sortKey: "pk_toi_sec",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={formatTOISplit(
            row.pk_toi_sec,
            row.gp
          )}
          sub={
            safe(row.sh_points, 0) > 0
              ? `${fmtZero(
                  row.sh_points
                )} SHP`
              : ""
          }
        />
      ),
      onClick: () =>
        changeSort("pk_toi_sec"),
    },

    faceoff: {
      label: "FO%",
      sortKey: "faceoff_pct",
      align: "right",
      render: (row) => {
        const attempts =
          safe(row.fow, 0) +
          safe(row.fol, 0);

        return (
          <SkaterStatCell
            value={
              attempts > 0
                ? fmtMaybePct(
                    row.faceoff_pct
                  )
                : "—"
            }
            sub={rankSub(
              row,
              "faceoff_pct"
            )}
          />
        );
      },
      onClick: () =>
        changeSort("faceoff_pct"),
    },

    hits: {
      label: "HIT",
      sortKey: "hit",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.hit)}
          sub={rankSub(row, "hit")}
        />
      ),
      onClick: () => changeSort("hit"),
    },

    blocks: {
      label: "BLK",
      sortKey: "blk",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtZero(row.blk)}
          sub={rankSub(row, "blk")}
        />
      ),
      onClick: () => changeSort("blk"),
    },

    age: {
      label: "Age",
      sortKey: "age",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={
            row.age
              ? fmtZero(row.age)
              : "—"
          }
          sub={
            row.rookie ||
            row.is_rookie
              ? "Rookie"
              : ""
          }
        />
      ),
      onClick: () =>
        changeSort("age", "asc"),
    },

    overall: {
      label: "OVR",
      sortKey: "overall",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={
            row.overall
              ? fmtZero(row.overall)
              : "—"
          }
          sub={rankSub(
            row,
            "overall"
          )}
        />
      ),
      onClick: () =>
        changeSort("overall"),
    },

    capHit: {
      label: "Cap Hit",
      sortKey: "cap_hit",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={formatMoney(row.cap_hit)}
          sub={rankSub(
            row,
            "cap_hit"
          )}
        />
      ),
      onClick: () =>
        changeSort("cap_hit"),
    },

    role: {
      label: "Role",
      sortKey: "role_label",
      render: (row) => (
        <span className="sc-player-role-text">
          {cleanText(
            row.role_label,
            "Regular"
          )}
        </span>
      ),
      onClick: () =>
        changeSort(
          "role_label",
          "asc"
        ),
    },

    actions: actionColumn,
  };

  const presetKeys = {
    core: [
      "player",
      "gp",
      "g",
      "a",
      "pts",
      "ppg",
      "toi",
      "actions",
    ],
    scoring: [
      "player",
      "gp",
      "g",
      "a",
      "pts",
      "sog",
      "shootingPct",
      "ixg",
      "finish",
      "actions",
    ],
    analytics: [
      "player",
      "gp",
      "p60",
      "cfPct",
      "xgfPct",
      "gfPct",
      "war",
      "actions",
    ],
    usage: [
      "player",
      "gp",
      "toi",
      "ppToi",
      "pkToi",
      "faceoff",
      "hits",
      "blocks",
      "actions",
    ],
    contract: [
      "player",
      "age",
      "overall",
      "capHit",
      "role",
      "gp",
      "toi",
      "actions",
    ],
  };

  const columns = presetKeys[view]
    .map((key) => allColumns[key])
    .filter((column) => {
      if (column === allColumns.age) {
        return hasUsefulColumnData(
          players,
          "age",
          1
        );
      }

      if (column === allColumns.overall) {
        return hasUsefulColumnData(
          players,
          "overall",
          1
        );
      }

      if (column === allColumns.capHit) {
        return hasUsefulColumnData(
          players,
          "cap_hit",
          1
        );
      }

      if (column === allColumns.faceoff) {
        return hasFaceoffData(players);
      }

      return true;
    });

  const forwards = players.filter(
    (player) =>
      isForwardRow(player)
  );
  const defense = players.filter(
    (player) =>
      isDefenseRow(player)
  );

  const tableProps = {
    columns,
    sortKey,
    sortDir,
    density,
    tableClassName: "sc-skaters-table-v3",
    rowClassName: (row) =>
      getSkaterRowClass(row),
    getRowId: (row) =>
      row.player_id || row.id,
    selectedRowId: selectedPlayerId,
    onRowClick: onSelectPlayer,
    rowAriaLabel: (row) =>
      `Inspect ${row.name}`,
    empty:
      "No skaters match the current filters.",
  };

  return (
    <div className="sc-tab-page sc-skaters-page-v3">
      <header className="sc-player-table-header">
        <div>
          <span>SKATER DATABASE</span>
          <strong>
            {scope === "team"
              ? "Roster Performance"
              : "League Skaters"}
          </strong>
          <em>
            {players.length} results · sorted by{" "}
            {String(sortKey).toUpperCase()}{" "}
            {sortDir === "desc" ? "↓" : "↑"}
          </em>
        </div>

        <div className="sc-column-preset-toggle">
          {presets.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={
                view === id
                  ? "is-active"
                  : ""
              }
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      {scope === "team" ? (
        <div className="sc-roster-groups">
          <section>
            <header>
              <span>FORWARDS</span>
              <strong>{forwards.length}</strong>
            </header>

            <DataTable
              {...tableProps}
              rows={forwards}
            />
          </section>

          <section>
            <header>
              <span>DEFENCE</span>
              <strong>{defense.length}</strong>
            </header>

            <DataTable
              {...tableProps}
              rows={defense}
            />
          </section>
        </div>
      ) : (
        <PagedDataTable
          {...tableProps}
          rows={players}
          page={page}
          onPageChange={onPageChange}
          pageSize={PLAYER_PAGE_SIZE.league}
        />
      )}
    </div>
  );
}
/* =========================================================
   GOALIES TAB
========================================================= */

function GoaliesTab({
  goalies,
  scope,
  sortKey,
  sortDir,
  changeSort,
  density = "compact",
  page = 1,
  onPageChange,
  selectedPlayerId = "",
  onSelectPlayer,
  pinnedIds = [],
  onTogglePin,
  onComparePlayer,
}) {
  const goalieRows = goalies || [];

  const actionColumn = {
    label: "",
    key: "actions",
    className: "is-actions-col",
    render: (row) => {
      const id = String(
        row.player_id || row.id
      );

      return (
        <PlayerRowActions
          player={row}
          isPinned={pinnedIds.includes(id)}
          onTogglePin={() =>
            onTogglePin(row)
          }
          onCompare={() =>
            onComparePlayer(row)
          }
        />
      );
    },
  };

  const columns = [
    {
      label: "Goalie",
      sortKey: "name",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope={scope}
        />
      ),
      onClick: () =>
        changeSort("name", "asc"),
    },
    {
      label: "GP",
      key: "gp",
      sortKey: "gp",
      align: "right",
      onClick: () => changeSort("gp"),
    },
    {
      label: "GS",
      key: "starts",
      sortKey: "starts",
      align: "right",
      onClick: () =>
        changeSort("starts"),
    },
    {
      label: "W",
      key: "wins",
      sortKey: "wins",
      align: "right",
      onClick: () =>
        changeSort("wins"),
    },
    {
      label: "L",
      key: "losses",
      sortKey: "losses",
      align: "right",
      onClick: () =>
        changeSort("losses", "asc"),
    },
    {
      label: "SV%",
      sortKey: "sv_pct",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtSavePct(row.sv_pct)}
          tone={
            safe(row.sv_pct, 0) >= 0.915
              ? "elite"
              : safe(row.sv_pct, 0) >=
                  0.905
                ? "good"
                : safe(
                      row.sv_pct,
                      0
                    ) < 0.89
                  ? "bad"
                  : ""
          }
        />
      ),
      onClick: () =>
        changeSort("sv_pct"),
    },
    {
      label: "GAA",
      sortKey: "gaa",
      align: "right",
      render: (row) => fmtTwo(row.gaa),
      onClick: () =>
        changeSort("gaa", "asc"),
    },
    {
      label: "SA",
      key: "sa",
      sortKey: "sa",
      align: "right",
      onClick: () => changeSort("sa"),
    },
    {
      label: "SO",
      key: "so",
      sortKey: "so",
      align: "right",
      onClick: () => changeSort("so"),
    },
    {
      label: "GSAx",
      sortKey: "gsax",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybeOne(row.gsax)}
          tone={
            safe(row.gsax, 0) > 0
              ? "good"
              : safe(row.gsax, 0) < 0
                ? "bad"
                : ""
          }
        />
      ),
      onClick: () => changeSort("gsax"),
    },
    {
      label: "QS%",
      sortKey: "quality_start_pct",
      align: "right",
      render: (row) =>
        fmtMaybePct(
          row.quality_start_pct
        ),
      onClick: () =>
        changeSort("quality_start_pct"),
    },
    {
      label: "WAR",
      sortKey: "war",
      align: "right",
      render: (row) =>
        fmtMaybeTwo(row.war),
      onClick: () => changeSort("war"),
    },
    actionColumn,
  ];

  const qualified =
    scope === "league"
      ? goalieRows.filter(
          (goalie) =>
            safe(goalie.gp, 0) >= 5
        )
      : goalieRows;

  const topSavePct = topBy(
    qualified,
    (goalie) =>
      normalizePct(goalie.sv_pct, 0)
  );
  const topGsax = topBy(
    qualified,
    (goalie) =>
      hasRealNumber(goalie.gsax)
        ? goalie.gsax
        : -999
  );
  const workloadLeader = topBy(
    qualified,
    (goalie) => goalie.sa
  );
  const qualityLeader = topBy(
    qualified,
    (goalie) =>
      normalizePct(
        goalie.quality_start_pct,
        0
      )
  );

  const tableProps = {
    columns,
    sortKey,
    sortDir,
    density,
    tableClassName: "sc-goalie-table-v2",
    getRowId: (row) =>
      row.player_id || row.id,
    selectedRowId: selectedPlayerId,
    onRowClick: onSelectPlayer,
    rowAriaLabel: (row) =>
      `Inspect ${row.name}`,
    empty:
      "No goalies match the current filters.",
  };

  return (
    <div className="sc-tab-page sc-goalies-page-v2">
      <header className="sc-player-table-header">
        <div>
          <span>GOALIE DATABASE</span>
          <strong>
            {scope === "team"
              ? "Crease Performance"
              : "League Goalies"}
          </strong>
          <em>
            {qualified.length} qualified · sorted by{" "}
            {String(sortKey).toUpperCase()}{" "}
            {sortDir === "desc" ? "↓" : "↑"}
          </em>
        </div>
      </header>

      <section className="sc-goalie-summary-row">
        <GoalieLeaderCard
          label="Save Percentage"
          goalie={topSavePct}
          value={
            topSavePct
              ? fmtSavePct(
                  topSavePct.sv_pct
                )
              : "—"
          }
          onSelect={onSelectPlayer}
        />

        <GoalieLeaderCard
          label="Goals Saved"
          goalie={topGsax}
          value={
            topGsax
              ? fmtMaybeOne(
                  topGsax.gsax
                )
              : "—"
          }
          onSelect={onSelectPlayer}
        />

        <GoalieLeaderCard
          label="Workload"
          goalie={workloadLeader}
          value={
            workloadLeader
              ? `${fmtZero(
                  workloadLeader.sa
                )} SA`
              : "—"
          }
          onSelect={onSelectPlayer}
        />

        <GoalieLeaderCard
          label="Quality Starts"
          goalie={qualityLeader}
          value={
            qualityLeader
              ? fmtMaybePct(
                  qualityLeader.quality_start_pct
                )
              : "—"
          }
          onSelect={onSelectPlayer}
        />
      </section>

      {scope === "league" ? (
        <PagedDataTable
          {...tableProps}
          rows={qualified}
          page={page}
          onPageChange={onPageChange}
          pageSize={PLAYER_PAGE_SIZE.league}
        />
      ) : (
        <DataTable
          {...tableProps}
          rows={qualified}
        />
      )}
    </div>
  );
}



function GoalieLeaderCard({
  label,
  goalie,
  value,
  onSelect,
}) {
  return (
    <button
      type="button"
      disabled={!goalie}
      onClick={() =>
        goalie && onSelect(goalie)
      }
    >
      <span>{label}</span>
      <strong>{goalie?.name || "—"}</strong>
      <em>{value}</em>
    </button>
  );
}

/* =========================================================
   ADVANCED TAB
========================================================= */

function AdvancedTab({
  players,
  scope,
  changeSort,
  sortKey,
  sortDir,
  density = "compact",
  page = 1,
  onPageChange,
  selectedPlayerId = "",
  onSelectPlayer,
  onComparePlayer,
}) {
  const columns = [
    {
      label: "Player",
      sortKey: "name",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope={scope}
        />
      ),
      onClick: () =>
        changeSort("name", "asc"),
    },
    {
      label: "GP",
      key: "gp",
      sortKey: "gp",
      align: "right",
      onClick: () => changeSort("gp"),
    },
    {
      label: "P/60",
      sortKey: "points_per_60",
      align: "right",
      render: (row) =>
        fmtTwo(row.points_per_60),
      onClick: () =>
        changeSort("points_per_60"),
    },
    {
      label: "CF%",
      sortKey: "cf_pct",
      align: "right",
      render: (row) =>
        fmtMaybePct(row.cf_pct),
      onClick: () =>
        changeSort("cf_pct"),
    },
    {
      label: "FF%",
      sortKey: "ff_pct",
      align: "right",
      render: (row) =>
        fmtMaybePct(row.ff_pct),
      onClick: () =>
        changeSort("ff_pct"),
    },
    {
      label: "xGF%",
      sortKey: "xgf_pct",
      align: "right",
      render: (row) => (
        <SkaterStatCell
          value={fmtMaybePct(
            row.xgf_pct
          )}
          tone={getStablePctTone(
            "xgf_pct",
            row.xgf_pct,
            safe(row.xgf, 0) +
              safe(row.xga, 0)
          )}
        />
      ),
      onClick: () =>
        changeSort("xgf_pct"),
    },
    {
      label: "GF%",
      sortKey: "gf_pct",
      align: "right",
      render: (row) => {
        const sample =
          getGFPercentSample(row);

        return sample > 0
          ? fmtMaybePct(row.gf_pct)
          : "—";
      },
      onClick: () =>
        changeSort("gf_pct"),
    },
    {
      label: "iXG",
      sortKey: "ixg",
      align: "right",
      render: (row) =>
        fmtMaybeOne(row.ixg),
      onClick: () => changeSort("ixg"),
    },
    {
      label: "xA",
      sortKey: "xa",
      align: "right",
      render: (row) =>
        fmtMaybeOne(row.xa),
      onClick: () => changeSort("xa"),
    },
    {
      label: "Finish",
      sortKey: "finishing",
      align: "right",
      render: (row) =>
        hasRealNumber(row.finishing)
          ? formatSigned(
              row.finishing,
              1
            )
          : "—",
      onClick: () =>
        changeSort("finishing"),
    },
    {
      label: "WAR",
      sortKey: "war",
      align: "right",
      render: (row) => (
        <span
          className={`sc-impact-text is-${getMetricTone(
            "war",
            row.war,
            row
          )}`}
        >
          {fmtMaybeTwo(row.war)}
        </span>
      ),
      onClick: () => changeSort("war"),
    },
    {
      label: "",
      key: "actions",
      className: "is-actions-col",
      render: (row) => (
        <button
          type="button"
          className="sc-inline-compare"
          onClick={(event) =>
            stopRowAction(event, () =>
              onComparePlayer(row)
            )
          }
        >
          Compare
        </button>
      ),
    },
  ];

  const processRows = getPlayerInsightRows(
    players,
    "process",
    5
  );
  const regressionRows = getPlayerInsightRows(
    players,
    "regression",
    5
  );

  const tableProps = {
    columns,
    sortKey,
    sortDir,
    density,
    tableClassName: "sc-analytics-table-v2",
    getRowId: (row) =>
      row.player_id || row.id,
    selectedRowId: selectedPlayerId,
    onRowClick: onSelectPlayer,
    empty:
      "No advanced player analytics match the current filters.",
  };

  return (
    <div className="sc-tab-page sc-analytics-page-v2">
      <header className="sc-player-table-header">
        <div>
          <span>ADVANCED ANALYTICS</span>
          <strong>
            Possession, Expected Goals, WAR
          </strong>
          <em>
            Backend ledger metrics only
          </em>
        </div>
      </header>

      <section className="sc-analytics-insights">
        <div>
          <span>STRONG PROCESS</span>
          <strong>Results Lagging</strong>
          <HiddenValueList
            players={processRows}
            onSelectPlayer={onSelectPlayer}
          />
        </div>

        <div>
          <span>FINISHING</span>
          <strong>Regression Flags</strong>
          <RegressionList
            players={regressionRows}
            onSelectPlayer={onSelectPlayer}
          />
        </div>
      </section>

      {scope === "league" ? (
        <PagedDataTable
          {...tableProps}
          rows={players}
          page={page}
          onPageChange={onPageChange}
          pageSize={PLAYER_PAGE_SIZE.league}
        />
      ) : (
        <DataTable
          {...tableProps}
          rows={players}
        />
      )}
    </div>
  );
}

function HiddenValueList({
  players,
  onSelectPlayer = () => {},
}) {
  const rows = (players || []).slice(0, 5);

  if (!rows.length) {
    return (
      <div className="sc-empty">
        No qualifying process-value players.
      </div>
    );
  }

  return (
    <div className="sc-compact-player-list">
      {rows.map((player) => (
        <button
          key={`process-${player.player_id}`}
          type="button"
          onClick={() =>
            onSelectPlayer(player)
          }
        >
          <span>
            <strong>{player.name}</strong>
            <em>
              {fmtMaybePct(
                player.xgf_pct
              )}{" "}
              xGF% ·{" "}
              {fmtZero(player.pts)} PTS
            </em>
          </span>

          <b>
            {hasRealNumber(
              player.finishing
            )
              ? formatSigned(
                  player.finishing,
                  1
                )
              : "—"}
          </b>
        </button>
      ))}
    </div>
  );
}

function RegressionList({
  players,
  onSelectPlayer = () => {},
}) {
  const rows = (players || []).slice(0, 5);

  if (!rows.length) {
    return (
      <div className="sc-empty">
        No finishing sample is available.
      </div>
    );
  }

  return (
    <div className="sc-compact-player-list">
      {rows.map((player) => (
        <button
          key={`regression-${player.player_id}`}
          type="button"
          onClick={() =>
            onSelectPlayer(player)
          }
        >
          <span>
            <strong>{player.name}</strong>
            <em>
              {fmtZero(player.g)} G ·{" "}
              {fmtMaybeOne(player.ixg)} iXG
            </em>
          </span>

          <b
            className={
              safe(
                player.finishing,
                0
              ) >= 0
                ? "is-good"
                : "is-bad"
            }
          >
            {hasRealNumber(
              player.finishing
            )
              ? formatSigned(
                  player.finishing,
                  1
                )
              : "—"}
          </b>
        </button>
      ))}
    </div>
  );
}


/* =========================================================
   TEAM TAB
========================================================= */

const TEAM_STAT_VIEWS = [
  {
    id: "overall",
    label: "Overall",
    defaultSort: "points",
    defaultDir: "desc",
  },
  {
    id: "offense",
    label: "Offense",
    defaultSort: "gf",
    defaultDir: "desc",
  },
  {
    id: "defense",
    label: "Defense",
    defaultSort: "ga",
    defaultDir: "asc",
  },
  {
    id: "special",
    label: "Special Teams",
    defaultSort: "pp_pct",
    defaultDir: "desc",
  },
  {
    id: "analytics",
    label: "Analytics",
    defaultSort: "xgf_pct",
    defaultDir: "desc",
  },
];

const TEAM_COLUMN_PRESETS = {
  overall: [
    "rank",
    "team",
    "gp",
    "wins",
    "losses",
    "otl",
    "points",
    "points_pct",
    "goal_diff",
  ],
  offense: [
    "rank",
    "team",
    "gp",
    "gf",
    "gf_per_game",
    "sf",
    "sf_per_game",
    "sh_pct",
    "xgf",
  ],
  defense: [
    "rank",
    "team",
    "gp",
    "ga",
    "ga_per_game",
    "sa",
    "sa_per_game",
    "sv_pct",
    "xga",
  ],
  special: [
    "rank",
    "team",
    "gp",
    "pp_pct",
    "ppg",
    "ppo",
    "pk_pct",
    "ppga",
    "opp_ppo",
  ],
  analytics: [
    "rank",
    "team",
    "gp",
    "cf_pct",
    "ff_pct",
    "xgf_pct",
    "pdo",
    "goal_diff",
  ],
};

const TEAM_METRIC_DIRECTIONS = {
  gp: "desc",
  wins: "desc",
  losses: "asc",
  otl: "asc",
  points: "desc",
  points_pct: "desc",
  win_pct: "desc",

  gf: "desc",
  gf_per_game: "desc",
  sf: "desc",
  sf_per_game: "desc",
  sh_pct: "desc",
  xgf: "desc",

  ga: "asc",
  ga_per_game: "asc",
  sa: "asc",
  sa_per_game: "asc",
  sv_pct: "desc",
  xga: "asc",

  ppg: "desc",
  ppo: "desc",
  pp_pct: "desc",
  ppga: "asc",
  opp_ppo: "asc",
  pk_pct: "desc",

  cf_pct: "desc",
  ff_pct: "desc",
  xgf_pct: "desc",
  pdo: "desc",
  goal_diff: "desc",
};

function hasTeamNumber(value) {
  if (value === null || value === undefined || value === "") return false;
  return Number.isFinite(Number(value));
}

function teamRowId(team) {
  return pickString(
    team?.team_id,
    team?.id,
    team?.abbrev,
    team?.abbr,
    team?.team_abbrev,
    team?.name,
    "unknown-team"
  );
}

function normalizeTeamIdentity(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function getTeamIdentityValues(team) {
  return [
    team?.team_id,
    team?.id,
    team?.abbrev,
    team?.abbr,
    team?.team_abbrev,
    team?.team_abbr,
    team?.name,
    team?.team_name,
    team?.full_name,
  ]
    .map(normalizeTeamIdentity)
    .filter(Boolean);
}

function isUserTeamRow(team, data) {
  const userValues = new Set([
    ...getTeamIdentityValues(data?.team),
    ...getTeamIdentityValues(data?.teamInfo),
    normalizeTeamIdentity(data?.teamId),
  ]);

  return getTeamIdentityValues(team).some((value) =>
    userValues.has(value)
  );
}

function sortTeamRows(rows, key, direction = "desc") {
  const multiplier = direction === "asc" ? 1 : -1;

  return [...(rows || [])].sort((a, b) => {
    if (key === "name") {
      return (
        teamDisplayLabel(a).localeCompare(teamDisplayLabel(b)) *
        multiplier
      );
    }

    const aValid = hasTeamNumber(a?.[key]);
    const bValid = hasTeamNumber(b?.[key]);

    if (!aValid && !bValid) {
      return teamDisplayLabel(a).localeCompare(teamDisplayLabel(b));
    }

    if (!aValid) return 1;
    if (!bValid) return -1;

    const difference =
      (Number(a[key]) - Number(b[key])) * multiplier;

    if (difference !== 0) return difference;

    const pointsDifference =
      safe(b?.points, 0) - safe(a?.points, 0);

    if (pointsDifference !== 0) return pointsDifference;

    return teamDisplayLabel(a).localeCompare(teamDisplayLabel(b));
  });
}

function buildTeamRankMaps(rows) {
  const teams = Array.isArray(rows) ? rows : [];
  const maps = {
    __total: teams.length,
  };

  Object.entries(TEAM_METRIC_DIRECTIONS).forEach(
    ([metric, direction]) => {
      const ranked = teams
        .filter((team) => hasTeamNumber(team?.[metric]))
        .sort((a, b) => {
          const aValue = Number(a[metric]);
          const bValue = Number(b[metric]);

          return direction === "asc"
            ? aValue - bValue
            : bValue - aValue;
        });

      const rankMap = new Map();

      ranked.forEach((team, index) => {
        rankMap.set(teamRowId(team), index + 1);
      });

      maps[metric] = rankMap;
    }
  );

  return maps;
}

function getTeamLeagueRank(rankMaps, team, metric) {
  return (
    rankMaps?.[metric]?.get(teamRowId(team)) ||
    0
  );
}

function getTeamRankTone(metric, rank, total) {
  if (!rank || !total) return "";

  if (["gp", "ppo", "opp_ppo", "pdo"].includes(metric)) {
    return "";
  }

  if (rank <= 5) return "is-top-five";
  if (rank > Math.max(0, total - 5)) return "is-bottom-five";

  return "";
}

function formatTeamStatValue(metric, value) {
  if (!hasTeamNumber(value)) return "—";

  if (metric === "sv_pct") {
    return fmtPct(value, 3);
  }

  if (
    [
      "pp_pct",
      "pk_pct",
      "cf_pct",
      "ff_pct",
      "xgf_pct",
      "sh_pct",
      "points_pct",
      "win_pct",
    ].includes(metric)
  ) {
    return fmtPct(value, 1);
  }

  if (metric === "pdo") {
    return fmtPdo(value);
  }

  if (
    [
      "gf_per_game",
      "ga_per_game",
      "sf_per_game",
      "sa_per_game",
    ].includes(metric)
  ) {
    return fmtTwo(value);
  }

  if (["xgf", "xga"].includes(metric)) {
    return fmtOne(value);
  }

  if (metric === "goal_diff") {
    return formatSigned(value);
  }

  return fmtZero(value);
}

function TeamMetricValue({
  team,
  metric,
  rankMaps,
}) {
  const formatted = formatTeamStatValue(metric, team?.[metric]);
  const rank = getTeamLeagueRank(rankMaps, team, metric);
  const total = rankMaps?.__total || 0;
  const tone = getTeamRankTone(metric, rank, total);

  return (
    <span className={`sc-team-value ${tone}`}>
      <strong>{formatted}</strong>

      {formatted !== "—" &&
      rank &&
      (rank <= 5 || rank > Math.max(0, total - 5)) ? (
        <em>#{rank}</em>
      ) : null}
    </span>
  );
}

function filterTeamRows({
  rows,
  search,
  conference,
  division,
  scope,
  data,
}) {
  const query = String(search || "").trim().toLowerCase();

  return (rows || []).filter((team) => {
    if (scope === "team" && !isUserTeamRow(team, data)) {
      return false;
    }

    if (
      conference !== "all" &&
      normalizeTeamIdentity(team?.conference) !== conference
    ) {
      return false;
    }

    if (
      division !== "all" &&
      normalizeTeamIdentity(team?.division) !== division
    ) {
      return false;
    }

    if (!query) return true;

    const haystack = [
      teamDisplayLabel(team),
      team?.team_id,
      team?.abbrev,
      team?.team_abbrev,
      team?.conference,
      team?.division,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });
}

function getTeamColumnDefinitions({
  data,
  rankMaps,
  onSort,
}) {
  const metricColumn = (
    key,
    label,
    className = ""
  ) => ({
    key,
    label,
    sortKey: key,
    align: "right",
    className,
    onClick: () => onSort(key),
    render: (row) => (
      <TeamMetricValue
        team={row}
        metric={key}
        rankMaps={rankMaps}
      />
    ),
  });

  return {
    rank: {
      key: "view_rank",
      label: "RK",
      className: "is-rank-col",
      render: (row) => (
        <span className="sc-team-view-rank">
          {String(row._viewRank || 0).padStart(2, "0")}
        </span>
      ),
    },

    team: {
      key: "team",
      label: "Team",
      sortKey: "name",
      className: "is-team-col",
      onClick: () => onSort("name"),
      render: (row) => {
        const overallLeagueRank = getTeamLeagueRank(
          rankMaps,
          row,
          "points"
        );

        return (
          <TeamNameCell
            team={row}
            isUser={isUserTeamRow(row, data)}
            overallLeagueRank={overallLeagueRank}
          />
        );
      },
    },

    gp: metricColumn(
      "gp",
      "GP",
      "is-group-start"
    ),

    wins: metricColumn(
      "wins",
      "W"
    ),

    losses: metricColumn(
      "losses",
      "L"
    ),

    otl: metricColumn(
      "otl",
      "OTL"
    ),

    points: metricColumn(
      "points",
      "PTS"
    ),

    points_pct: metricColumn(
      "points_pct",
      "PTS%"
    ),

    goal_diff: metricColumn(
      "goal_diff",
      "DIFF",
      "is-group-start"
    ),

    gf: metricColumn(
      "gf",
      "GF",
      "is-group-start"
    ),

    gf_per_game: metricColumn(
      "gf_per_game",
      "GF/GP"
    ),

    sf: metricColumn(
      "sf",
      "SF"
    ),

    sf_per_game: metricColumn(
      "sf_per_game",
      "SF/GP"
    ),

    sh_pct: metricColumn(
      "sh_pct",
      "SH%"
    ),

    xgf: metricColumn(
      "xgf",
      "xGF",
      "is-group-start"
    ),

    ga: metricColumn(
      "ga",
      "GA",
      "is-group-start"
    ),

    ga_per_game: metricColumn(
      "ga_per_game",
      "GA/GP"
    ),

    sa: metricColumn(
      "sa",
      "SA"
    ),

    sa_per_game: metricColumn(
      "sa_per_game",
      "SA/GP"
    ),

    sv_pct: metricColumn(
      "sv_pct",
      "SV%"
    ),

    xga: metricColumn(
      "xga",
      "xGA",
      "is-group-start"
    ),

    pp_pct: metricColumn(
      "pp_pct",
      "PP%",
      "is-group-start"
    ),

    ppg: metricColumn(
      "ppg",
      "PPG"
    ),

    ppo: metricColumn(
      "ppo",
      "PPO"
    ),

    pk_pct: metricColumn(
      "pk_pct",
      "PK%",
      "is-group-start"
    ),

    ppga: metricColumn(
      "ppga",
      "PPGA"
    ),

    opp_ppo: metricColumn(
      "opp_ppo",
      "TSH"
    ),

    cf_pct: metricColumn(
      "cf_pct",
      "CF%",
      "is-group-start"
    ),

    ff_pct: metricColumn(
      "ff_pct",
      "FF%"
    ),

    xgf_pct: metricColumn(
      "xgf_pct",
      "xGF%"
    ),

    pdo: metricColumn(
      "pdo",
      "PDO",
      "is-group-start"
    ),
  };
}

function TeamStatsToolbar({
  view,
  onViewChange,
  scope,
  onScopeChange,
  search,
  onSearchChange,
  conference,
  onConferenceChange,
  division,
  onDivisionChange,
  conferences,
  divisions,
  visibleCount,
  totalCount,
  onReset,
}) {
  return (
    <section className="sc-team-toolbar">
      <div className="sc-team-toolbar-top">
        <div className="sc-team-toolbar-title">
          <span>TEAM PERFORMANCE</span>
          <strong>
            {visibleCount} of {totalCount} teams
          </strong>
        </div>

        <div
          className="sc-team-view-tabs"
          aria-label="Team statistics category"
        >
          {TEAM_STAT_VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={view === item.id ? "is-active" : ""}
              onClick={() => onViewChange(item)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sc-team-toolbar-filters">
        <div className="sc-team-scope-control">
          <button
            type="button"
            className={scope === "league" ? "is-active" : ""}
            onClick={() => onScopeChange("league")}
          >
            League
          </button>

          <button
            type="button"
            className={scope === "team" ? "is-active" : ""}
            onClick={() => onScopeChange("team")}
          >
            My Team
          </button>
        </div>

        <select
          value={conference}
          onChange={(event) =>
            onConferenceChange(event.target.value)
          }
          aria-label="Filter by conference"
        >
          <option value="all">All Conferences</option>

          {conferences.map((item) => (
            <option
              key={item}
              value={normalizeTeamIdentity(item)}
            >
              {item}
            </option>
          ))}
        </select>

        <select
          value={division}
          onChange={(event) =>
            onDivisionChange(event.target.value)
          }
          aria-label="Filter by division"
        >
          <option value="all">All Divisions</option>

          {divisions.map((item) => (
            <option
              key={item}
              value={normalizeTeamIdentity(item)}
            >
              {item}
            </option>
          ))}
        </select>

        <label className="sc-team-search">
          <span>⌕</span>

          <input
            type="search"
            value={search}
            onChange={(event) =>
              onSearchChange(event.target.value)
            }
            placeholder="Search teams"
          />
        </label>

        <button
          type="button"
          className="sc-team-reset"
          onClick={onReset}
        >
          Reset
        </button>
      </div>
    </section>
  );
}

function getTeamProfileMetrics(view) {
  if (view === "offense") {
    return [
      ["GF", "gf"],
      ["GF / GP", "gf_per_game"],
      ["SF / GP", "sf_per_game"],
      ["SH%", "sh_pct"],
      ["xGF", "xgf"],
    ];
  }

  if (view === "defense") {
    return [
      ["GA", "ga"],
      ["GA / GP", "ga_per_game"],
      ["SA / GP", "sa_per_game"],
      ["SV%", "sv_pct"],
      ["xGA", "xga"],
    ];
  }

  if (view === "special") {
    return [
      ["PP%", "pp_pct"],
      ["PP Goals", "ppg"],
      ["PP Chances", "ppo"],
      ["PK%", "pk_pct"],
      ["PP Goals Against", "ppga"],
    ];
  }

  if (view === "analytics") {
    return [
      ["CF%", "cf_pct"],
      ["FF%", "ff_pct"],
      ["xGF%", "xgf_pct"],
      ["PDO", "pdo"],
      ["Goal Diff", "goal_diff"],
    ];
  }

  return [
    ["Points", "points"],
    ["Points %", "points_pct"],
    ["Goals For", "gf"],
    ["Goals Against", "ga"],
    ["Goal Diff", "goal_diff"],
  ];
}

function TeamProfileMetric({
  team,
  label,
  metric,
  rankMaps,
}) {
  const rank = getTeamLeagueRank(rankMaps, team, metric);
  const total = rankMaps?.__total || 0;
  const tone = getTeamRankTone(metric, rank, total);

  return (
    <div className={`sc-team-profile-metric ${tone}`}>
      <span>{label}</span>
      <strong>
        {formatTeamStatValue(metric, team?.[metric])}
      </strong>
      <em>
        {rank ? `League #${rank}` : "Rank unavailable"}
      </em>
    </div>
  );
}

function TeamStatsProfile({
  team,
  data,
  view,
  rankMaps,
}) {
  if (!team) {
    return (
      <section className="sc-team-selected-profile is-empty">
        Select a team to inspect its profile.
      </section>
    );
  }

  const isUser = isUserTeamRow(team, data);
  const leagueRank = getTeamLeagueRank(
    rankMaps,
    team,
    "points"
  );

  const location = [
    team?.division,
    team?.conference,
  ]
    .filter(Boolean)
    .join(" · ");

  const metrics = getTeamProfileMetrics(view);

  return (
    <section className="sc-team-selected-profile">
      <div className="sc-team-profile-identity">
        <div className="sc-team-profile-logo">
          <TeamLogoMark team={team} size="large" />
        </div>

        <div className="sc-team-profile-copy">
          <span>
            {isUser ? "YOUR FRANCHISE" : "SELECTED TEAM"}
          </span>

          <h2>{teamDisplayLabel(team)}</h2>

          <p>
            {location || team?.team_abbrev || team?.team_id || "League"}
          </p>
        </div>
      </div>

      <div className="sc-team-profile-headline">
        <div>
          <strong>
            {fmtZero(team.wins)}-{fmtZero(team.losses)}-{fmtZero(team.otl)}
          </strong>
          <span>Record</span>
        </div>

        <div>
          <strong>{fmtZero(team.points)}</strong>
          <span>Points</span>
        </div>

        <div>
          <strong>
            {formatTeamStatValue("goal_diff", team.goal_diff)}
          </strong>
          <span>Goal Diff</span>
        </div>

        <div>
          <strong>
            {leagueRank ? `#${leagueRank}` : "—"}
          </strong>
          <span>League</span>
        </div>
      </div>

      <div className="sc-team-profile-metrics">
        {metrics.map(([label, metric]) => (
          <TeamProfileMetric
            key={metric}
            team={team}
            label={label}
            metric={metric}
            rankMaps={rankMaps}
          />
        ))}
      </div>
    </section>
  );
}

function TeamTab({ data, loadState }) {
  const rows = useMemo(() => {
    return data.teams?.length
      ? data.teams
      : data.team
        ? [data.team]
        : [];
  }, [data.teams, data.team]);

  const [view, setView] = useState("overall");
  const [scope, setScope] = useState("league");
  const [search, setSearch] = useState("");
  const [conference, setConference] = useState("all");
  const [division, setDivision] = useState("all");
  const [sortKey, setSortKey] = useState("points");
  const [sortDir, setSortDir] = useState("desc");
  const [selectedTeamId, setSelectedTeamId] = useState("");

  const rankMaps = useMemo(
    () => buildTeamRankMaps(rows),
    [rows]
  );

  useEffect(() => {
    if (!rows.length) return;

    const currentExists = rows.some(
      (team) =>
        teamRowId(team) === selectedTeamId
    );

    if (currentExists) return;

    const preferredTeam =
      rows.find((team) =>
        isUserTeamRow(team, data)
      ) || rows[0];

    setSelectedTeamId(
      teamRowId(preferredTeam)
    );
  }, [rows, selectedTeamId, data]);

  const conferences = useMemo(() => {
    return Array.from(
      new Set(
        rows
          .map((team) =>
            pickString(team?.conference)
          )
          .filter(Boolean)
      )
    ).sort();
  }, [rows]);

  const divisions = useMemo(() => {
    const eligibleRows =
      conference === "all"
        ? rows
        : rows.filter(
            (team) =>
              normalizeTeamIdentity(
                team?.conference
              ) === conference
          );

    return Array.from(
      new Set(
        eligibleRows
          .map((team) =>
            pickString(team?.division)
          )
          .filter(Boolean)
      )
    ).sort();
  }, [rows, conference]);

  useEffect(() => {
    if (division === "all") return;

    const divisionStillExists =
      divisions.some(
        (item) =>
          normalizeTeamIdentity(item) ===
          division
      );

    if (!divisionStillExists) {
      setDivision("all");
    }
  }, [division, divisions]);

  const filteredTeams = useMemo(() => {
    return filterTeamRows({
      rows,
      search,
      conference,
      division,
      scope,
      data,
    });
  }, [
    rows,
    search,
    conference,
    division,
    scope,
    data,
  ]);

  const sortedTeams = useMemo(() => {
    return sortTeamRows(
      filteredTeams,
      sortKey,
      sortDir
    ).map((team, index) => ({
      ...team,
      _viewRank: index + 1,
    }));
  }, [filteredTeams, sortKey, sortDir]);

  const selectedTeam = useMemo(() => {
    return (
      rows.find(
        (team) =>
          teamRowId(team) ===
          selectedTeamId
      ) ||
      rows.find((team) =>
        isUserTeamRow(team, data)
      ) ||
      rows[0] ||
      null
    );
  }, [rows, selectedTeamId, data]);

  const handleSort = useCallback(
    (key) => {
      if (key === sortKey) {
        setSortDir((current) =>
          current === "desc"
            ? "asc"
            : "desc"
        );
        return;
      }

      setSortKey(key);

      if (key === "name") {
        setSortDir("asc");
        return;
      }

      setSortDir(
        TEAM_METRIC_DIRECTIONS[key] ||
          "desc"
      );
    },
    [sortKey]
  );

  const handleViewChange =
    useCallback((nextView) => {
      setView(nextView.id);
      setSortKey(nextView.defaultSort);
      setSortDir(nextView.defaultDir);
    }, []);

  const resetFilters = useCallback(() => {
    setScope("league");
    setSearch("");
    setConference("all");
    setDivision("all");

    const currentView =
      TEAM_STAT_VIEWS.find(
        (item) => item.id === view
      ) || TEAM_STAT_VIEWS[0];

    setSortKey(currentView.defaultSort);
    setSortDir(currentView.defaultDir);
  }, [view]);

  const columnDefinitions = useMemo(
    () =>
      getTeamColumnDefinitions({
        data,
        rankMaps,
        onSort: handleSort,
      }),
    [data, rankMaps, handleSort]
  );

  const columns =
    TEAM_COLUMN_PRESETS[view].map(
      (columnKey) =>
        columnDefinitions[columnKey]
    );

  if (
    loadState === "loading" &&
    !rows.length
  ) {
    return (
      <div className="sc-team-state-message">
        <strong>
          Loading team statistics
        </strong>
        <span>
          Preparing league comparison data.
        </span>
      </div>
    );
  }

  if (
    loadState === "error" &&
    !rows.length
  ) {
    return (
      <div className="sc-team-state-message is-error">
        <strong>
          Team statistics unavailable
        </strong>
        <span>
          The league data could not be loaded.
        </span>
      </div>
    );
  }

  return (
    <div className="sc-team-stats-workspace">
      <TeamStatsToolbar
        view={view}
        onViewChange={handleViewChange}
        scope={scope}
        onScopeChange={setScope}
        search={search}
        onSearchChange={setSearch}
        conference={conference}
        onConferenceChange={setConference}
        division={division}
        onDivisionChange={setDivision}
        conferences={conferences}
        divisions={divisions}
        visibleCount={sortedTeams.length}
        totalCount={rows.length}
        onReset={resetFilters}
      />

      <section className="sc-team-table-panel">
        <DataTable
          columns={columns}
          rows={sortedTeams}
          sortKey={sortKey}
          sortDir={sortDir}
          tableClassName="sc-team-stats-table"
          empty="No teams match the current filters."
          getRowId={(team) =>
            teamRowId(team)
          }
          selectedRowId={selectedTeamId}
          onRowClick={(team) =>
            setSelectedTeamId(
              teamRowId(team)
            )
          }
          rowAriaLabel={(team) =>
            `Select ${teamDisplayLabel(
              team
            )}`
          }
          rowClassName={(team) =>
            isUserTeamRow(team, data)
              ? "is-user-team-row"
              : ""
          }
        />
      </section>

      <TeamStatsProfile
        team={selectedTeam}
        data={data}
        view={view}
        rankMaps={rankMaps}
      />
    </div>
  );
}

/* =========================================================
   LEADERS TAB
========================================================= */

function LeadersTab({ data }) {
  const [category, setCategory] = useState("points");
  const [page, setPage] = useState(1);

  const teams = data?.teams || [];
  const franchiseState =
    data?.franchiseState || null;

  const categories = [
    {
      id: "points",
      label: "Points",
      rows:
        rowsByBackendRank(
          data.skaters,
          "league_rank_pts",
          40
        ).length
          ? rowsByBackendRank(
              data.skaters,
              "league_rank_pts",
              40
            )
          : rankRows(
              data.skaters,
              "pts"
            ).slice(0, 40),
      metric: "pts",
      metricLabel: "PTS",
      formatter: (row) =>
        fmtZero(row.pts),
    },
    {
      id: "goals",
      label: "Goals",
      rows:
        rowsByBackendRank(
          data.skaters,
          "league_rank_goals",
          40
        ).length
          ? rowsByBackendRank(
              data.skaters,
              "league_rank_goals",
              40
            )
          : rankRows(
              data.skaters,
              "g"
            ).slice(0, 40),
      metric: "g",
      metricLabel: "G",
      formatter: (row) =>
        fmtZero(row.g),
    },
    {
      id: "assists",
      label: "Assists",
      rows:
        rowsByBackendRank(
          data.skaters,
          "league_rank_assists",
          40
        ).length
          ? rowsByBackendRank(
              data.skaters,
              "league_rank_assists",
              40
            )
          : rankRows(
              data.skaters,
              "a"
            ).slice(0, 40),
      metric: "a",
      metricLabel: "A",
      formatter: (row) =>
        fmtZero(row.a),
    },
    {
      id: "war",
      label: "WAR",
      rows:
        rowsByBackendRank(
          data.skaters,
          "league_rank_war",
          40
        ).length
          ? rowsByBackendRank(
              data.skaters,
              "league_rank_war",
              40
            )
          : rankRows(
              data.skaters.filter((row) =>
                hasRealNumber(row.war)
              ),
              "war"
            ).slice(0, 40),
      metric: "war",
      metricLabel: "WAR",
      formatter: (row) =>
        fmtMaybeTwo(row.war),
    },
    {
      id: "save_pct",
      label: "Save %",
      rows:
        rowsByBackendRank(
          data.goalies,
          "league_rank_sv_pct",
          40
        ).length
          ? rowsByBackendRank(
              data.goalies,
              "league_rank_sv_pct",
              40
            )
          : rankRows(
              data.goalies.filter(
                (row) =>
                  safe(row.gp, 0) >= 5
              ),
              "sv_pct"
            ).slice(0, 40),
      metric: "sv_pct",
      metricLabel: "SV%",
      formatter: (row) =>
        fmtSavePct(row.sv_pct),
    },
    {
      id: "gsax",
      label: "GSAx",
      rows:
        rowsByBackendRank(
          data.goalies,
          "league_rank_gsax",
          40
        ).length
          ? rowsByBackendRank(
              data.goalies,
              "league_rank_gsax",
              40
            )
          : rankRows(
              data.goalies.filter((row) =>
                hasRealNumber(row.gsax)
              ),
              "gsax"
            ).slice(0, 40),
      metric: "gsax",
      metricLabel: "GSAx",
      formatter: (row) =>
        fmtMaybeOne(row.gsax),
    },
  ];

  const active =
    categories.find(
      (item) => item.id === category
    ) || categories[0];

  useEffect(() => {
    setPage(1);
  }, [category]);

  const columns = [
    {
      label: "RK",
      key: "rank",
      className: "is-rank-col",
      render: (row, index) => (
        <span className="sc-leader-rank">
          {String(
            row._rank ||
              (page - 1) * 12 +
                index +
                1
          ).padStart(2, "0")}
        </span>
      ),
    },
    {
      label:
        active.id === "save_pct" ||
        active.id === "gsax"
          ? "Goalie"
          : "Player",
      className: "is-player-col",
      render: (row) => (
        <PlayerNameCell
          player={row}
          scope="league"
          teams={teams}
          franchiseState={franchiseState}
        />
      ),
    },
    {
      label: active.metricLabel,
      align: "right",
      render: (row) => (
        <strong className="sc-leader-primary">
          {active.formatter(row)}
        </strong>
      ),
    },
    {
      label:
        active.id === "save_pct" ||
        active.id === "gsax"
          ? "Workload"
          : "Rate",
      align: "right",
      render: (row) =>
        normalizePosition(row.position) ===
        "G"
          ? `${fmtZero(row.gp)} GP · ${fmtZero(
              row.sa
            )} SA`
          : `${fmtTwo(
              row.points_per_game
            )} P/GP · ${formatSmallTOI(
              row
            )}`,
    },
  ];

  const podium = active.rows.slice(0, 3);

  return (
    <div className="sc-leaders-workspace">
      <header className="sc-leader-category-header">
        <div>
          <span>LEAGUE LEADERS</span>
          <strong>{active.label}</strong>
        </div>

        <nav aria-label="Leader category">
          {categories.map((item) => (
            <button
              key={item.id}
              type="button"
              className={
                category === item.id
                  ? "is-active"
                  : ""
              }
              onClick={() =>
                setCategory(item.id)
              }
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <section className="sc-leader-podium">
        {podium.map((row, index) => (
          <article
            key={`podium-${active.id}-${row.player_id}`}
          >
            <span>#{index + 1}</span>
            <PlayerAvatar
              player={row}
              teams={teams}
              franchiseState={franchiseState}
            />
            <strong>{row.name}</strong>
            <em>
              {getPlayerTeamLabel(row)}
            </em>
            <b>{active.formatter(row)}</b>
          </article>
        ))}
      </section>

      <PagedDataTable
        columns={columns}
        rows={active.rows}
        page={page}
        onPageChange={setPage}
        pageSize={12}
        density="compact"
        tableClassName="sc-leader-table-v2"
        empty="No ranked rows are available."
      />
    </div>
  );
}


/* =========================================================
   AWARDS TAB — CLEAN GROUPED BOARD
========================================================= */

const AWARD_GROUPS = {
  nhl: "nhl",
  custom: "custom",
};

const NHL_AWARD_DEFINITIONS = {
  hart: {
    shortLabel: "Hart",
    label: "Hart Memorial Trophy",
    type: "skater",
    primaryMetricLabel: "WAR",
    description:
      "Most valuable player to his team. Uses production, WAR, usage, team success, and all-situation value.",
    aliases: ["hart", "hart memorial trophy", "mvp"],
  },
  art_ross: {
    shortLabel: "Art Ross",
    label: "Art Ross Trophy",
    type: "skater",
    primaryMetricLabel: "PTS",
    description: "Awarded to the league leader in total points.",
    aliases: ["art ross", "art ross trophy", "art_ross", "points leader"],
  },
  rocket: {
    shortLabel: "Rocket",
    label: "Rocket Richard Trophy",
    type: "skater",
    primaryMetricLabel: "G",
    description: "Awarded to the league leader in goals.",
    aliases: ["rocket", "rocket richard", "rocket richard trophy", "goals leader"],
  },
  norris: {
    shortLabel: "Norris",
    label: "James Norris Trophy",
    type: "skater",
    primaryMetricLabel: "WAR",
    description:
      "Best all-around defenseman. Uses WAR, points, TOI, xGF%, CF%, and defensive value.",
    aliases: ["norris", "norris trophy", "james norris"],
  },
  selke: {
    shortLabel: "Selke",
    label: "Frank J. Selke Trophy",
    type: "skater",
    primaryMetricLabel: "DEF",
    description:
      "Best defensive forward. Uses defensive value, PK usage, faceoff value, shot suppression, and two-way play.",
    aliases: ["selke", "selke trophy", "defensive forward"],
  },
  calder: {
    shortLabel: "Calder",
    label: "Calder Memorial Trophy",
    type: "skater",
    primaryMetricLabel: "Rookie",
    description:
      "Best rookie. Uses rookie production, role difficulty, WAR, and team importance.",
    aliases: ["calder", "calder trophy", "rookie"],
  },
  vezina: {
    shortLabel: "Vezina",
    label: "Vezina Trophy",
    type: "goalie",
    primaryMetricLabel: "SV%",
    description:
      "Best goaltender. Uses SV%, GAA, GSAx, quality starts, workload, and consistency.",
    aliases: ["vezina", "vezina trophy", "best goalie"],
  },
  jennings: {
    shortLabel: "Jennings",
    label: "William M. Jennings Trophy",
    type: "team",
    primaryMetricLabel: "GA",
    description:
      "Awarded to the team/goaltenders with the fewest goals against.",
    aliases: ["jennings", "william jennings", "fewest goals against"],
  },
  presidents: {
    shortLabel: "Presidents",
    label: "Presidents’ Trophy",
    type: "team",
    primaryMetricLabel: "PTS",
    description: "Awarded to the team with the most standings points.",
    aliases: ["presidents", "presidents trophy", "best team", "most points"],
  },
};

const CUSTOM_AWARD_DEFINITIONS = {
  gretzky_offense: {
    shortLabel: "Gretzky Offense",
    label: "Wayne Gretzky Best Offensive Player",
    type: "skater",
    primaryMetricLabel: "OFF",
    description:
      "Best offensive monster. Uses points, goals, assists, iXG, xA, P/60, PP production, and offensive impact.",
    aliases: [
      "gretzky",
      "wayne gretzky",
      "wayne gretzky best overall offensive player",
      "best overall offensive player",
      "best offensive player",
      "offensive player",
      "overall offensive",
    ],
  },
  mcdavid_transition: {
    shortLabel: "McDavid Transition",
    label: "Connor McDavid Transition King",
    type: "skater",
    primaryMetricLabel: "Rush",
    description:
      "Best rush and transition driver. Uses controlled entries, speed-style impact, chance creation, and production.",
    aliases: ["mcdavid", "connor mcdavid", "transition king", "rush driver", "transition"],
  },
  datsyuk_two_way: {
    shortLabel: "Datsyuk Two-Way",
    label: "Pavel Datsyuk Two-Way Wizard",
    type: "skater",
    primaryMetricLabel: "2-Way",
    description:
      "Best skill-plus-defense forward. Uses takeaways, CF%, xGF%, defensive impact, and offensive production.",
    aliases: ["datsyuk", "pavel datsyuk", "two way", "two-way", "two way wizard"],
  },
  chara_shutdown: {
    shortLabel: "Chara Shutdown",
    label: "Zdeno Chara Shutdown Defender",
    type: "skater",
    primaryMetricLabel: "DEF",
    description:
      "Best defensive defenseman. Uses blocks, hits, defensive impact, xGA suppression, and hard minutes.",
    aliases: ["chara", "zdeno chara", "shutdown defender", "shutdown defenseman"],
  },
  hasek_goalie: {
    shortLabel: "Hasek Saver",
    label: "Dominik Hasek Chaos-Saver Award",
    type: "goalie",
    primaryMetricLabel: "GSAx",
    description:
      "Goalie who steals the most value above expected. Uses GSAx, SV%, high-danger SV%, and workload.",
    aliases: ["hasek", "dominik hasek", "chaos saver", "chaos-saver", "steal value"],
  },
};

function normalizeAwardText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function awardAliasMatches(definition, awardKey, awardLabel) {
  const key = normalizeAwardText(awardKey);
  const label = normalizeAwardText(awardLabel);
  const combined = `${key} ${label}`.trim();

  return (definition.aliases || []).some((alias) => {
    const cleanAlias = normalizeAwardText(alias);
    return key === cleanAlias || label === cleanAlias || combined.includes(cleanAlias);
  });
}

function findStrictAwardDefinition(awardKey, awardLabel = "") {
  const key = normalizeAwardText(awardKey);
  const label = normalizeAwardText(awardLabel);

  for (const [definitionKey, definition] of Object.entries(NHL_AWARD_DEFINITIONS)) {
    if (
      key === normalizeAwardText(definitionKey) ||
      label === normalizeAwardText(definitionKey) ||
      awardAliasMatches(definition, awardKey, awardLabel)
    ) {
      return {
        key: definitionKey,
        group: AWARD_GROUPS.nhl,
        ...definition,
      };
    }
  }

  for (const [definitionKey, definition] of Object.entries(CUSTOM_AWARD_DEFINITIONS)) {
    if (
      key === normalizeAwardText(definitionKey) ||
      label === normalizeAwardText(definitionKey) ||
      awardAliasMatches(definition, awardKey, awardLabel)
    ) {
      return {
        key: definitionKey,
        group: AWARD_GROUPS.custom,
        ...definition,
      };
    }
  }

  return null;
}

function humanizeAwardLabel(value) {
  const raw = pickString(value, "Custom Award").replace(/[_-]+/g, " ").trim();
  return raw.replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function genericAwardDefinition(awardKey, awardLabel = "", row = null) {
  const key = normalizeAwardText(awardKey).replace(/\s+/g, "_") || "custom_award";
  const label = humanizeAwardLabel(awardLabel || awardKey);
  const position = normalizePosition(firstPresent(row?.position, row?.pos));
  const isTeam =
    row &&
    firstPresent(row?.team_id, row?.id) !== undefined &&
    firstPresent(row?.player_id, row?.player_name, row?.full_name) === undefined &&
    position !== "G";

  return {
    key,
    group: AWARD_GROUPS.custom,
    shortLabel: label,
    label,
    type: isTeam ? "team" : position === "G" ? "goalie" : "skater",
    primaryMetricLabel: "Score",
    description: "Backend award watch candidate.",
  };
}

function getAwardDefinitionList(view) {
  const source =
    view === AWARD_GROUPS.nhl
      ? NHL_AWARD_DEFINITIONS
      : CUSTOM_AWARD_DEFINITIONS;

  return Object.entries(source).map(([key, definition]) => ({
    key,
    ...definition,
  }));
}

function getAwardPrimaryMetric(row) {
  const key = String(row.award_key || "");

  if (key === "art_ross") return fmtZero(row.pts);
  if (key === "rocket" && !hasRealNumber(row.score)) return fmtZero(row.g);
  if (key === "presidents") return fmtZero(row.points);
  if (key === "jennings") return fmtZero(row.ga);
  if (key === "vezina") return fmtPct(row.sv_pct, 3);
  if (key === "hasek_goalie") return fmtMaybeOne(row.gsax);
  if (row.award_subjective && row.reason) return "Case";

  if (row.position === "G") return fmtPct(row.sv_pct, 3);
  if (row._kind === "team") return fmtZero(row.points);

  return row.score === undefined || row.score === null
    ? fmtZero(row.pts)
    : fmtOne(row.score);
}

function getAwardSecondaryLine(row) {
  if (row.position === "G") {
    return `GAA ${fmtTwo(row.gaa)} · GSAx ${fmtMaybeOne(row.gsax)} · ${fmtZero(row.wins)}W`;
  }

  if (row._kind === "team") {
    return `${fmtZero(row.wins)}-${fmtZero(row.losses)}-${fmtZero(row.otl)} · GF ${fmtZero(row.gf)} · GA ${fmtZero(row.ga)}`;
  }

  return `${fmtZero(row.g)}G · ${fmtZero(row.a)}A · ${fmtZero(row.pts)}PTS · ${fmtMaybePct(row.xgf_pct)} xGF%`;
}

function getAwardBoardSummary(rows, teamId, teamInfo = {}) {
  const teamKeys = new Set(
    [
      teamId,
      teamInfo?.team_id,
      teamInfo?.id,
      teamInfo?.abbrev,
      teamInfo?.abbr,
      teamInfo?.team_abbrev,
      teamInfo?.name,
      teamInfo?.team_name,
      teamInfo?.full_name,
    ]
      .map((value) => String(value || "").toLowerCase().trim())
      .filter(Boolean)
  );
  const leaders = rows.filter((row) => safeInt(row.rank, 0) === 1);
  const myTeamLeaders = leaders.filter((row) => {
    const rowKeys = [
      row.team_id,
      row.team,
      row.team_abbrev,
      row.abbrev,
      row.team_name,
      row.name,
    ].map((value) => String(value || "").toLowerCase().trim());
    return rowKeys.some((value) => value && teamKeys.has(value));
  });

  return {
    awardCount: new Set(rows.map((row) => row.award_key)).size,
    candidateCount: rows.length,
    leaderCount: leaders.length,
    myTeamLeaderCount: myTeamLeaders.length,
  };
}

function AwardsTab({ data }) {
  const [awardView, setAwardView] = useState(AWARD_GROUPS.nhl);

  const backendAwards = buildAwardsRowsFromPayload(data.awardsWatch, data.teamId);
  const allRows = mergeAwardRows(backendAwards, []);

  const rows = allRows
    .filter((row) => row.award_group === awardView)
    .slice(0, 25);

  const definitions = getAwardDefinitionList(awardView);
  const summary = getAwardBoardSummary(rows, data.teamId, data.teamInfo);

  const columns = [
    {
      label: "Award",
      render: (row) => (
        <div className="sc-awards-award-cell">
          <strong>{row.award_short_label || row.award_label}</strong>
          <span>{row.award_label}</span>
        </div>
      ),
    },
    {
      label: "Rank",
      align: "right",
      render: (row) => (
        <span className={`sc-awards-rank ${safeInt(row.rank, 0) === 1 ? "is-leader" : ""}`}>
          #{row.rank || 1}
        </span>
      ),
    },
    {
      label: "Candidate",
      render: (row) =>
        row._kind === "team" ? (
          <TeamNameCell team={row} />
        ) : (
          <PlayerNameCell player={row} />
        ),
    },
    {
      label: "Main",
      align: "right",
      render: (row) => (
        <span className="sc-awards-primary-metric">
          {getAwardPrimaryMetric(row)}
        </span>
      ),
    },
    {
      label: "Profile",
      render: (row) => (
        <span className="sc-awards-profile-line">
          {getAwardSecondaryLine(row)}
        </span>
      ),
    },
  ];

  const leaderRows = rows.filter((row) => safeInt(row.rank, 0) === 1).slice(0, 6);

  return (
    <div className="sc-awards-board">
      <section className="sc-awards-subtabs">
        <button
          type="button"
          className={awardView === AWARD_GROUPS.nhl ? "is-active" : ""}
          onClick={() => setAwardView(AWARD_GROUPS.nhl)}
        >
          <strong>NHL Awards</strong>
          <span>Official trophies only</span>
        </button>

        <button
          type="button"
          className={awardView === AWARD_GROUPS.custom ? "is-active" : ""}
          onClick={() => setAwardView(AWARD_GROUPS.custom)}
        >
          <strong>Made-Up Awards</strong>
          <span>Custom analytics awards only</span>
        </button>
      </section>

      <section className="sc-awards-summary-strip">
        <div>
          <span>Awards</span>
          <strong>{summary.awardCount}</strong>
        </div>
        <div>
          <span>Candidates</span>
          <strong>{summary.candidateCount}</strong>
        </div>
        <div>
          <span>Leaders</span>
          <strong>{summary.leaderCount}</strong>
        </div>
        <div>
          <span>My Team Leaders</span>
          <strong>{summary.myTeamLeaderCount}</strong>
        </div>
      </section>

      <section className="sc-awards-main-grid">
        <Section
          eyebrow="Awards Watch"
          title={
            awardView === AWARD_GROUPS.nhl
              ? "Official NHL Award Races"
              : "Made-Up Franchise Award Races"
          }
          right={<Pill>{rows.length} / 25 rows</Pill>}
        >
          <DataTable
            columns={columns}
            rows={rows}
            empty={
              awardView === AWARD_GROUPS.nhl
                ? "No official NHL award candidates found."
                : "No made-up award candidates found."
            }
          />
        </Section>

        <Section
          eyebrow="Current Leaders"
          title={awardView === AWARD_GROUPS.nhl ? "Trophy Leaders" : "Custom Award Leaders"}
        >
          <div className="sc-awards-leader-list">
            {leaderRows.length ? (
              leaderRows.map((row) => (
                <AwardLeaderCard key={`${row.award_key}-${row.player_id || row.team_id}`} row={row} />
              ))
            ) : (
              <div className="sc-empty">No leaders found for this award group.</div>
            )}
          </div>
        </Section>
      </section>

      <section className="sc-awards-rules-row">
        <Section
          eyebrow="Award Rules"
          title={awardView === AWARD_GROUPS.nhl ? "Official Criteria" : "Custom Criteria"}
        >
          <div className="sc-awards-rule-grid">
            {definitions.map((definition) => (
              <article key={definition.key} className="sc-awards-rule-card">
                <strong>{definition.shortLabel}</strong>
                <span>{definition.description}</span>
              </article>
            ))}
          </div>
        </Section>
      </section>
    </div>
  );
}

function AwardLeaderCard({ row }) {
  return (
    <article className={`sc-awards-leader-card ${row.award_group === AWARD_GROUPS.custom ? "is-custom" : ""}`}>
      <div>
        <span>{row.award_short_label || row.award_label}</span>
        <strong>{row.name || row.team_id || row.team || "—"}</strong>
        <em>{getAwardSecondaryLine(row)}</em>
      </div>

      <b>{getAwardPrimaryMetric(row)}</b>
    </article>
  );
}

function buildAwardsRowsFromPayload(awardsWatch, teamId) {
  if (!awardsWatch || typeof awardsWatch !== "object") return [];

  const rows = [];

  Object.entries(awardsWatch).forEach(([awardKey, value]) => {
    const values =
      Array.isArray(value)
        ? value
        : value && typeof value === "object"
          ? [value]
          : [];

    values.forEach((row, index) => {
      const rawAwardLabel = pickString(
        row?.award_label,
        row?.award,
        row?.trophy,
        row?.name_of_award,
        awardKey
      );

      const definition =
        findStrictAwardDefinition(awardKey, rawAwardLabel) ||
        genericAwardDefinition(awardKey, rawAwardLabel, row);

      const normalized = normalizeAwardCandidateEntity(row, index, teamId, definition.type);

      rows.push({
        ...normalized,
        award_key: definition.key,
        award_group: definition.group,
        award_type: definition.type,
        award_label: definition.label.toUpperCase(),
        award_short_label: definition.shortLabel.toUpperCase(),
        official: row?.official !== false && definition.group !== "custom",
        watch_type: row?.watch_type || (definition.group === "custom" ? "custom_franchise_award" : "official_live_race"),
        ceremony_enabled: row?.ceremony_enabled,
        display_metric: row?.display_metric || definition.metric || "",
        calculation_quality: row?.calculation_quality || "full",
        eligibility_confidence: row?.eligibility_confidence,
        award_description: definition.description,
        award_subjective: Boolean(row?.award_subjective),
        rank: safeInt(firstPresent(row?.rank), index + 1),
        score: pickStat(
          row?.score,
          row?.award_score,
          row?.analytics_rating,
          normalized.analytics_rating,
          normalized.pts,
          normalized.points,
          0
        ),
        reason: pickString(
          row?.award_rationale,
          row?.reason,
          row?.explanation,
          definition.description
        ),
      });
    });
  });

  return dedupeAwardRows(rows)
    .sort((a, b) => {
      const awardCompare = String(a.award_label).localeCompare(String(b.award_label));
      if (awardCompare !== 0) return awardCompare;
      return safeInt(a.rank, 999) - safeInt(b.rank, 999);
    })
    .slice(0, 80);
}

function mergeAwardRows(backendRows, fallbackRows) {
  const merged = [...(backendRows || [])];
  const backendKeys = new Set(merged.map((row) => row.award_key));

  (fallbackRows || []).forEach((row) => {
    if (!backendKeys.has(row.award_key)) {
      merged.push({ ...row, estimated: true });
    }
  });

  return dedupeAwardRows(merged).sort((a, b) => {
    const groupCompare = String(a.award_group).localeCompare(String(b.award_group));
    if (groupCompare !== 0) return groupCompare;
    const awardCompare = String(a.award_label).localeCompare(String(b.award_label));
    if (awardCompare !== 0) return awardCompare;
    return safeInt(a.rank, 999) - safeInt(b.rank, 999);
  });
}

function normalizeAwardCandidateEntity(row, index, teamId, awardType) {
  if (awardType === "team") {
    return normalizeTeam(row, index);
  }

  if (awardType === "goalie") {
    return normalizeGoalie(row, index, teamId);
  }

  if (isGoalieRow(row)) {
    return normalizeGoalie(row, index, teamId);
  }

  return normalizeSkater(row, index, teamId);
}

function dedupeAwardRows(rows) {
  const seen = new Set();

  return (rows || []).filter((row) => {
    const entityId = pickString(
      row.player_id,
      row.team_id,
      row.id,
      row.name,
      row.player_name,
      "unknown"
    );

    const key = `${row.award_key}_${entityId}_${row.rank}`;

    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function buildFallbackAwardsRows(data) {
  const rows = [];

  const skaters = Array.isArray(data.skaters) ? data.skaters : [];
  const goalies = Array.isArray(data.goalies) ? data.goalies : [];
  const teams = Array.isArray(data.teams) ? data.teams : [];

  rows.push(
    ...buildAwardCandidateRows("art_ross", rankRows(skaters, "pts").slice(0, 5), (p) => p.pts),
    ...buildAwardCandidateRows("rocket", rankRows(skaters, "g").slice(0, 5), (p) => p.g),
    ...buildAwardCandidateRows(
      "hart",
      [...skaters]
        .map((p) => ({
          ...p,
          _award_score: safe(p.analytics_rating, 0) + safe(p.pts, 0) * 0.45,
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "norris",
      skaters
        .filter((p) => p.position === "D")
        .map((p) => ({
          ...p,
          _award_score: safe(p.analytics_rating, 0) + safe(p.pts, 0) * 0.35,
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "selke",
      skaters
        .filter((p) => p.position !== "D" && p.position !== "G")
        .map((p) => ({
          ...p,
          _award_score:
            safe(p.defensive_impact, 0) +
            normalizePct(p.cf_pct, 0) * 40 +
            safe(p.blk, 0) * 0.1 +
            safe(p.sh_points, 0),
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "calder",
      skaters
        .filter((p) => p.rookie)
        .map((p) => ({
          ...p,
          _award_score: safe(p.pts, 0) + safe(p.analytics_rating, 0),
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "vezina",
      goalies
        .map((g) => ({
          ...g,
          _award_score:
            normalizePct(g.sv_pct, 0) * 100 +
            safe(g.gsax, 0) -
            safe(g.gaa, 0),
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (g) => g._award_score
    ),
    ...buildAwardCandidateRows(
      "jennings",
      [...teams]
        .sort((a, b) => safe(a.ga, 9999) - safe(b.ga, 9999))
        .slice(0, 5),
      (t) => -safe(t.ga, 0)
    ),
    ...buildAwardCandidateRows(
      "presidents",
      rankRows(teams, "points").slice(0, 5),
      (t) => t.points
    ),
    ...buildAwardCandidateRows(
      "gretzky_offense",
      skaters
        .map((p) => ({
          ...p,
          _award_score:
            safe(p.pts, 0) * 2 +
            safe(p.g, 0) * 1.25 +
            safe(p.ixg, 0) +
            safe(p.xa, 0) +
            safe(p.offensive_impact, 0),
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "mcdavid_transition",
      skaters
        .map((p) => ({
          ...p,
          _award_score:
            safe(p.points_per_60, p.pts_per_60 || 0) * 18 +
            safe(p.xa, 0) * 1.2 +
            normalizePct(p.xgf_pct, 0) * 45 +
            safe(p.offensive_impact, 0) * 0.6,
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "datsyuk_two_way",
      skaters
        .filter((p) => p.position !== "D" && p.position !== "G")
        .map((p) => ({
          ...p,
          _award_score:
            safe(p.takeaways, 0) * 1.5 +
            safe(p.defensive_impact, 0) +
            normalizePct(p.xgf_pct, 0) * 50 +
            safe(p.pts, 0) * 0.35,
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "chara_shutdown",
      skaters
        .filter((p) => p.position === "D")
        .map((p) => ({
          ...p,
          _award_score:
            safe(p.blk, 0) +
            safe(p.hit, 0) * 0.4 +
            safe(p.defensive_impact, 0),
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (p) => p._award_score
    ),
    ...buildAwardCandidateRows(
      "hasek_goalie",
      goalies
        .map((g) => ({
          ...g,
          _award_score:
            safe(g.gsax, 0) * 2 +
            normalizePct(g.sv_pct, 0) * 100 +
            normalizePct(g.hd_sv_pct, 0) * 35,
        }))
        .sort((a, b) => b._award_score - a._award_score)
        .slice(0, 5),
      (g) => g._award_score
    )
  );

  return dedupeAwardRows(rows);
}

function buildAwardCandidateRows(awardKey, candidates, scoreFn) {
  const definition = findStrictAwardDefinition(awardKey, awardKey);

  if (!definition) return [];

  return (candidates || []).map((candidate, index) => {
    const rank = index + 1;
    const score = safe(scoreFn(candidate), 0);

    return {
      ...candidate,
      award_key: definition.key,
      award_group: definition.group,
      award_type: definition.type,
      award_label: definition.label.toUpperCase(),
      award_short_label: definition.shortLabel.toUpperCase(),
      award_description: definition.description,
      rank,
      score,
      reason: definition.description,
    };
  });
}
/* =========================================================
   TRENDS TAB
========================================================= */

function TrendsTab({
  data,
  players,
  goalies,
  games,
  scope = "team",
  onSelectPlayer = () => {},
}) {
  const recentGames = Array.isArray(games)
    ? games.slice(-20)
    : [];
  const last10 = recentGames.slice(-10);
  const previous10 = recentGames.slice(
    -20,
    -10
  );

  const average = (rows, getter) => {
    if (!rows.length) return null;

    return (
      rows.reduce(
        (sum, row) =>
          sum + safe(getter(row), 0),
        0
      ) / rows.length
    );
  };

  const last10Goals = average(
    last10,
    (game) =>
      safe(game.home_goals) +
      safe(game.away_goals)
  );
  const previous10Goals = average(
    previous10,
    (game) =>
      safe(game.home_goals) +
      safe(game.away_goals)
  );
  const last10Shots = average(
    last10,
    (game) =>
      safe(game.home_shots) +
      safe(game.away_shots)
  );
  const previous10Shots = average(
    previous10,
    (game) =>
      safe(game.home_shots) +
      safe(game.away_shots)
  );

  const hasGoalComparison =
    last10Goals !== null &&
    previous10Goals !== null;
  const hasShotComparison =
    last10Shots !== null &&
    previous10Shots !== null;

  const playerTrendRows = [...players]
    .filter(
      (player) =>
        hasRealNumber(
          firstPresent(
            player?.last_10_points,
            player?.last10_points,
            player?.recent_points
          )
        ) ||
        hasRealNumber(
          firstPresent(
            player?.last_10_war,
            player?.last10_war,
            player?.recent_war
          )
        )
    )
    .sort(
      (a, b) =>
        safe(
          firstPresent(
            b?.last_10_points,
            b?.last10_points,
            b?.recent_points
          ),
          0
        ) -
        safe(
          firstPresent(
            a?.last_10_points,
            a?.last10_points,
            a?.recent_points
          ),
          0
        )
    )
    .slice(0, 8);

  const goalieTrendRows = [...goalies]
    .filter(
      (goalie) =>
        hasRealPct(
          firstPresent(
            goalie?.last_10_sv_pct,
            goalie?.last10_sv_pct,
            goalie?.recent_sv_pct
          )
        )
    )
    .sort(
      (a, b) =>
        normalizePct(
          firstPresent(
            b?.last_10_sv_pct,
            b?.last10_sv_pct,
            b?.recent_sv_pct
          ),
          0
        ) -
        normalizePct(
          firstPresent(
            a?.last_10_sv_pct,
            a?.last10_sv_pct,
            a?.recent_sv_pct
          ),
          0
        )
    )
    .slice(0, 4);

  const processLeaders = [...players]
    .filter(
      (player) =>
        safe(player.gp, 0) >= 8 &&
        hasRealPct(player.xgf_pct)
    )
    .sort(
      (a, b) =>
        normalizePct(b.xgf_pct, 0) -
        normalizePct(a.xgf_pct, 0)
    )
    .slice(0, 5);

  return (
    <div className="sc-tab-page sc-trends-page-v2">
      <header className="sc-player-table-header">
        <div>
          <span>TRENDS</span>
          <strong>
            {scope === "team"
              ? "My Team Movement"
              : "League Movement"}
          </strong>
          <em>
            Recent splits appear only when supplied by the backend
          </em>
        </div>
      </header>

      <section className="sc-trend-summary-row">
        <StatCard
          label="Scoring Pace"
          value={
            hasGoalComparison
              ? formatSigned(
                  last10Goals -
                    previous10Goals,
                  2
                )
              : "—"
          }
          sub="Last 10 vs previous 10"
          tone={
            hasGoalComparison
              ? trendTone(
                  last10Goals -
                    previous10Goals,
                  0.05,
                  -0.05
                )
              : ""
          }
        />

        <StatCard
          label="Shot Pace"
          value={
            hasShotComparison
              ? formatSigned(
                  last10Shots -
                    previous10Shots,
                  1
                )
              : "—"
          }
          sub="Last 10 vs previous 10"
          tone={
            hasShotComparison
              ? trendTone(
                  last10Shots -
                    previous10Shots,
                  0.5,
                  -0.5
                )
              : ""
          }
        />

        <StatCard
          label="Team PDO"
          value={fmtPdo(data.team?.pdo)}
          sub="Shooting plus save percentage"
          tone={
            hasRealNumber(
              normalizePdo(data.team?.pdo)
            )
              ? (normalizePdo(
                  data.team?.pdo
                ) ?? 100) >= 103
                ? "warn"
                : (normalizePdo(
                      data.team?.pdo
                    ) ?? 100) <= 97
                  ? "bad"
                  : "neutral"
              : ""
          }
        />

        <StatCard
          label="Team xGF%"
          value={fmtMaybePct(
            data.team?.xgf_pct
          )}
          sub="Chance share"
          tone={
            data.team?.xgf_pct >= 0.52
              ? "good"
              : data.team?.xgf_pct <= 0.48
                ? "bad"
                : "neutral"
          }
        />
      </section>

      <section className="sc-trend-content-grid">
        <div className="sc-trend-panel">
          <header>
            <span>RECENT PLAYER SPLITS</span>
            <strong>Last 10 Leaders</strong>
          </header>

          {playerTrendRows.length ? (
            <div className="sc-compact-player-list">
              {playerTrendRows.map((player) => {
                const recentPoints = safe(
                  firstPresent(
                    player?.last_10_points,
                    player?.last10_points,
                    player?.recent_points
                  ),
                  0
                );

                return (
                  <button
                    key={`trend-${player.player_id}`}
                    type="button"
                    onClick={() =>
                      onSelectPlayer(player)
                    }
                  >
                    <span>
                      <strong>{player.name}</strong>
                      <em>
                        {getPlayerTeamLabel(
                          player
                        )} · {player.position}
                      </em>
                    </span>
                    <b>
                      {fmtZero(
                        recentPoints
                      )}{" "}
                      PTS
                    </b>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="sc-empty">
              Backend last-10 player splits are not available yet.
            </div>
          )}
        </div>

        <div className="sc-trend-panel">
          <header>
            <span>PROCESS</span>
            <strong>xGF% Leaders</strong>
          </header>

          <div className="sc-compact-player-list">
            {processLeaders.map((player) => (
              <button
                key={`process-trend-${player.player_id}`}
                type="button"
                onClick={() =>
                  onSelectPlayer(player)
                }
              >
                <span>
                  <strong>{player.name}</strong>
                  <em>
                    {fmtZero(player.pts)} PTS ·{" "}
                    {formatSmallTOI(player)} TOI
                  </em>
                </span>
                <b>
                  {fmtMaybePct(
                    player.xgf_pct
                  )}
                </b>
              </button>
            ))}
          </div>
        </div>

        <div className="sc-trend-panel">
          <header>
            <span>GOALTENDING</span>
            <strong>Recent Save Rate</strong>
          </header>

          {goalieTrendRows.length ? (
            <div className="sc-compact-player-list">
              {goalieTrendRows.map((goalie) => {
                const recentSavePct =
                  firstPresent(
                    goalie?.last_10_sv_pct,
                    goalie?.last10_sv_pct,
                    goalie?.recent_sv_pct
                  );

                return (
                  <button
                    key={`goalie-trend-${goalie.player_id}`}
                    type="button"
                    onClick={() =>
                      onSelectPlayer(goalie)
                    }
                  >
                    <span>
                      <strong>{goalie.name}</strong>
                      <em>
                        {getPlayerTeamLabel(
                          goalie
                        )} · {fmtZero(
                          goalie.gp
                        )} GP
                      </em>
                    </span>
                    <b>
                      {fmtSavePct(
                        recentSavePct
                      )}
                    </b>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="sc-empty">
              Backend recent goalie splits are not available yet.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function TrendCard({ label, player, value, tone = "neutral" }) {
  return (
    <div className={`sc-list-card is-${tone}`}>
      <span>{label}</span>
      <strong>{player?.name || "—"}</strong>
      <em>{value || "—"}</em>
    </div>
  );
}


/* =========================================================
   COMPARE TAB
========================================================= */

function CompareTab({
  players,
  goalies,
  leftId,
  rightId,
  onLeftIdChange,
  onRightIdChange,
  pinnedIds = [],
}) {
  const allPlayers = useMemo(
    () => [...(players || []), ...(goalies || [])],
    [players, goalies]
  );
  const [category, setCategory] = useState("production");

  const resolvedLeftId =
    leftId || allPlayers[0]?.player_id || "";
  const left =
    allPlayers.find(
      (player) =>
        String(player.player_id) ===
        String(resolvedLeftId)
    ) ||
    allPlayers[0] ||
    null;

  const sameTypePool = left
    ? allPlayers.filter(
        (player) =>
          normalizePosition(player.position) ===
          normalizePosition(left.position)
      )
    : allPlayers;

  const resolvedRightId =
    rightId &&
    sameTypePool.some(
      (player) =>
        String(player.player_id) ===
        String(rightId)
    )
      ? rightId
      : sameTypePool.find(
          (player) =>
            String(player.player_id) !==
            String(left?.player_id)
        )?.player_id || "";

  const right =
    sameTypePool.find(
      (player) =>
        String(player.player_id) ===
        String(resolvedRightId)
    ) || null;

  useEffect(() => {
    if (!leftId && allPlayers[0]?.player_id) {
      onLeftIdChange(
        allPlayers[0].player_id
      );
    }
  }, [
    allPlayers,
    leftId,
    onLeftIdChange,
  ]);

  useEffect(() => {
    if (
      resolvedRightId &&
      String(rightId) !==
        String(resolvedRightId)
    ) {
      onRightIdChange(resolvedRightId);
    }
  }, [
    resolvedRightId,
    rightId,
    onRightIdChange,
  ]);

  const isGoalie =
    normalizePosition(left?.position) === "G";

  useEffect(() => {
    if (isGoalie) {
      setCategory("goalie");
    } else if (category === "goalie") {
      setCategory("production");
    }
  }, [isGoalie, category]);

  const categories = isGoalie
    ? [["goalie", "Goalie"]]
    : [
        ["production", "Production"],
        ["analytics", "Analytics"],
        ["usage", "Usage"],
      ];

  const rows = getCompareMetricRows(
    left,
    right,
    category
  );

  const pinnedPlayers = allPlayers.filter(
    (player) =>
      pinnedIds.includes(
        String(player.player_id || player.id)
      )
  );

  return (
    <div className="sc-tab-page sc-compare-page-v2">
      <header className="sc-player-table-header">
        <div>
          <span>COMPARE</span>
          <strong>
            Side-by-Side Decision Tool
          </strong>
          <em>
            Like-for-like player comparison
          </em>
        </div>

        <div className="sc-compare-category-tabs">
          {categories.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={
                category === id
                  ? "is-active"
                  : ""
              }
              onClick={() => setCategory(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <section className="sc-compare-selector-row">
        <label>
          <span>Left Player</span>
          <select
            value={resolvedLeftId}
            onChange={(event) => {
              onLeftIdChange(
                event.target.value
              );
              onRightIdChange("");
            }}
          >
            {allPlayers.map((player) => (
              <option
                key={`left-${player.player_id}`}
                value={player.player_id}
              >
                {player.name} ·{" "}
                {getPlayerTeamLabel(player)} ·{" "}
                {player.position}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Right Player</span>
          <select
            value={resolvedRightId}
            onChange={(event) =>
              onRightIdChange(
                event.target.value
              )
            }
          >
            {sameTypePool
              .filter(
                (player) =>
                  String(
                    player.player_id
                  ) !==
                  String(
                    left?.player_id
                  )
              )
              .map((player) => (
                <option
                  key={`right-${player.player_id}`}
                  value={player.player_id}
                >
                  {player.name} ·{" "}
                  {getPlayerTeamLabel(
                    player
                  )}{" "}
                  · {player.position}
                </option>
              ))}
          </select>
        </label>

        {pinnedPlayers.length ? (
          <div className="sc-compare-pinned">
            <span>Pinned</span>
            {pinnedPlayers.map((player) => (
              <button
                key={`pinned-${player.player_id}`}
                type="button"
                onClick={() => {
                  if (
                    normalizePosition(
                      player.position
                    ) ===
                    normalizePosition(
                      left?.position
                    )
                  ) {
                    onRightIdChange(
                      player.player_id
                    );
                  } else {
                    onLeftIdChange(
                      player.player_id
                    );
                    onRightIdChange("");
                  }
                }}
              >
                {player.name}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <section className="sc-compare-main-grid">
        <ComparePlayerCard
          player={left}
          side="left"
        />

        <div className="sc-compare-metric-board">
          {rows.map((row) => (
            <CompareMetricBar
              key={row.key}
              {...row}
            />
          ))}
        </div>

        <ComparePlayerCard
          player={right}
          side="right"
        />
      </section>
    </div>
  );
}


function getCompareMetricRows(
  left,
  right,
  category
) {
  if (!left || !right) return [];

  const metric = (
    key,
    label,
    formatter,
    options = {}
  ) => ({
    key,
    label,
    left: left?.[key],
    right: right?.[key],
    formatter,
    lowerIsBetter:
      options.lowerIsBetter || false,
  });

  if (category === "goalie") {
    return [
      metric(
        "gp",
        "Games Played",
        (value) => fmtZero(value)
      ),
      metric(
        "starts",
        "Starts",
        (value) => fmtZero(value)
      ),
      metric(
        "wins",
        "Wins",
        (value) => fmtZero(value)
      ),
      metric(
        "sv_pct",
        "Save Percentage",
        (value) => fmtSavePct(value)
      ),
      metric(
        "gaa",
        "Goals Against Average",
        (value) => fmtTwo(value),
        { lowerIsBetter: true }
      ),
      metric(
        "gsax",
        "Goals Saved Above Expected",
        (value) => fmtMaybeOne(value)
      ),
      metric(
        "quality_start_pct",
        "Quality Start Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "war",
        "WAR",
        (value) => fmtMaybeTwo(value)
      ),
    ];
  }

  if (category === "analytics") {
    return [
      metric(
        "points_per_60",
        "Points Per 60",
        (value) => fmtTwo(value)
      ),
      metric(
        "cf_pct",
        "Corsi For Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "ff_pct",
        "Fenwick For Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "xgf_pct",
        "Expected Goals For Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "gf_pct",
        "Goals For Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "ixg",
        "Individual Expected Goals",
        (value) => fmtMaybeOne(value)
      ),
      metric(
        "finishing",
        "Finishing Above Expected",
        (value) =>
          hasRealNumber(value)
            ? formatSigned(value, 1)
            : "—"
      ),
      metric(
        "war",
        "WAR",
        (value) => fmtMaybeTwo(value)
      ),
    ];
  }

  if (category === "usage") {
    return [
      {
        key: "toi",
        label: "Time On Ice Per Game",
        left: getAverageTOIMinutes(left),
        right: getAverageTOIMinutes(right),
        formatter: (value) =>
          formatClockFromMinutes(value),
        lowerIsBetter: false,
      },
      {
        key: "pp_toi",
        label: "Power-Play Time Per Game",
        left:
          safe(left.pp_toi_sec, 0) /
          60 /
          Math.max(1, safe(left.gp, 0)),
        right:
          safe(right.pp_toi_sec, 0) /
          60 /
          Math.max(1, safe(right.gp, 0)),
        formatter: (value) =>
          formatClockFromMinutes(value),
        lowerIsBetter: false,
      },
      {
        key: "pk_toi",
        label: "Penalty-Kill Time Per Game",
        left:
          safe(left.pk_toi_sec, 0) /
          60 /
          Math.max(1, safe(left.gp, 0)),
        right:
          safe(right.pk_toi_sec, 0) /
          60 /
          Math.max(1, safe(right.gp, 0)),
        formatter: (value) =>
          formatClockFromMinutes(value),
        lowerIsBetter: false,
      },
      metric(
        "faceoff_pct",
        "Faceoff Percentage",
        (value) => fmtMaybePct(value)
      ),
      metric(
        "hit",
        "Hits",
        (value) => fmtZero(value)
      ),
      metric(
        "blk",
        "Blocked Shots",
        (value) => fmtZero(value)
      ),
      metric(
        "takeaways",
        "Takeaways",
        (value) => fmtZero(value)
      ),
      metric(
        "pim",
        "Penalty Minutes",
        (value) => fmtZero(value),
        { lowerIsBetter: true }
      ),
    ];
  }

  return [
    metric(
      "gp",
      "Games Played",
      (value) => fmtZero(value)
    ),
    metric(
      "g",
      "Goals",
      (value) => fmtZero(value)
    ),
    metric(
      "a",
      "Assists",
      (value) => fmtZero(value)
    ),
    metric(
      "pts",
      "Points",
      (value) => fmtZero(value)
    ),
    metric(
      "points_per_game",
      "Points Per Game",
      (value) => fmtTwo(value)
    ),
    metric(
      "sog",
      "Shots On Goal",
      (value) => fmtZero(value)
    ),
    metric(
      "shooting_pct",
      "Shooting Percentage",
      (value) => fmtMaybePct(value)
    ),
    metric(
      "toi",
      "Time On Ice Per Game",
      (_, playerSide) => playerSide
    ),
  ].map((row) => {
    if (row.key !== "toi") return row;

    return {
      ...row,
      left: getAverageTOIMinutes(left),
      right: getAverageTOIMinutes(right),
      formatter: (value) =>
        formatClockFromMinutes(value),
    };
  });
}

function CompareMetricBar({
  label,
  left,
  right,
  formatter,
  lowerIsBetter = false,
}) {
  const leftNumber = hasRealNumber(left)
    ? Number(left)
    : null;
  const rightNumber = hasRealNumber(right)
    ? Number(right)
    : null;

  const maxValue = Math.max(
    Math.abs(leftNumber || 0),
    Math.abs(rightNumber || 0),
    0.0001
  );

  const leftWidth =
    leftNumber === null
      ? 0
      : Math.min(
          100,
          (Math.abs(leftNumber) / maxValue) *
            100
        );
  const rightWidth =
    rightNumber === null
      ? 0
      : Math.min(
          100,
          (Math.abs(rightNumber) / maxValue) *
            100
        );

  const leftBetter =
    leftNumber !== null &&
    rightNumber !== null &&
    leftNumber !== rightNumber
      ? lowerIsBetter
        ? leftNumber < rightNumber
        : leftNumber > rightNumber
      : null;

  return (
    <div className="sc-compare-metric">
      <div className="sc-compare-metric-values">
        <strong
          className={
            leftBetter === true
              ? "is-good"
              : leftBetter === false
                ? "is-bad"
                : ""
          }
        >
          {formatter(left)}
        </strong>

        <span>{label}</span>

        <strong
          className={
            leftBetter === false
              ? "is-good"
              : leftBetter === true
                ? "is-bad"
                : ""
          }
        >
          {formatter(right)}
        </strong>
      </div>

      <div className="sc-compare-bars">
        <div>
          <i style={{ width: `${leftWidth}%` }} />
        </div>
        <div>
          <i style={{ width: `${rightWidth}%` }} />
        </div>
      </div>
    </div>
  );
}

function ComparePlayerCard({
  player,
  side,
}) {
  if (!player) {
    return (
      <div
        className={`sc-compare-card is-${side}`}
      >
        <strong>—</strong>
        <span>No player selected</span>
      </div>
    );
  }

  const isGoalie =
    normalizePosition(player.position) === "G";

  return (
    <article
      className={`sc-compare-card is-${side}`}
    >
      <div className="sc-compare-card-top">
        <PlayerAvatar player={player} />

        <div>
          <span>
            {getPlayerTeamLabel(player)}
          </span>
          <strong>{player.name}</strong>
          <em>
            {player.position}
            {player.age
              ? ` · ${player.age} years`
              : ""}
          </em>
        </div>
      </div>

      <div className="sc-compare-card-grid">
        <span>
          GP <b>{fmtZero(player.gp)}</b>
        </span>

        <span>
          WAR{" "}
          <b>{fmtMaybeTwo(player.war)}</b>
        </span>

        {isGoalie ? (
          <>
            <span>
              SV%{" "}
              <b>
                {fmtSavePct(
                  player.sv_pct
                )}
              </b>
            </span>

            <span>
              GAA{" "}
              <b>{fmtTwo(player.gaa)}</b>
            </span>
          </>
        ) : (
          <>
            <span>
              PTS{" "}
              <b>{fmtZero(player.pts)}</b>
            </span>

            <span>
              xGF%{" "}
              <b>
                {fmtMaybePct(
                  player.xgf_pct
                )}
              </b>
            </span>
          </>
        )}
      </div>
    </article>
  );
}

function CompareRow({
  label,
  left,
  right,
  lowerIsBetter = false,
  digits = 0,
  hidden = false,
}) {
  if (hidden) return null;

  const format = (value) =>
    digits > 0
      ? safe(value, 0).toFixed(digits)
      : String(Math.round(safe(value, 0)));

  return (
    <CompareMetricBar
      label={label}
      left={left}
      right={right}
      lowerIsBetter={lowerIsBetter}
      formatter={format}
    />
  );
}


/* =========================================================
   LOGS TAB
========================================================= */

function LogsTab({ data, games }) {
  const logRows = Array.isArray(data.logs) ? data.logs.slice(-180).reverse() : [];
  const gameRows = Array.isArray(games) ? games.slice(-80).reverse() : [];

  return (
    <div className="sc-tab-page">
      <div className="sc-bottom-grid is-logs">
        <Section eyebrow="Timeline" title="Analytics / Storyline Logs">
          <div className="sc-log-list">
            {logRows.length ? (
              logRows.map((row, index) => (
                <LogCard key={`log-${row.id || index}`} row={row} />
              ))
            ) : (
              <div className="sc-empty">No analytics logs yet.</div>
            )}
          </div>
        </Section>

        <Section eyebrow="Game Ledger" title="Recent Games">
          <GameScoreList games={gameRows} />
        </Section>
      </div>
    </div>
  );
}

function LogCard({ row }) {
  const title = pickString(row?.headline, row?.title, row?.type, row?.event_type, "Event");
  const text = pickString(row?.text, row?.description, row?.message, "");
  const date = pickString(row?.calendar_iso, row?.date, row?.calendar_day, "");

  return (
    <article className="sc-log-card">
      <strong>{title}</strong>
      {text ? <span>{text}</span> : <span>{JSON.stringify(row)}</span>}
      {date ? <em>{date}</em> : null}
    </article>
  );
}


/* =========================================================
   FORMULAS TAB
========================================================= */

const FORMULA_SECTIONS = [
  {
    title: "Basic Counting Stats",
    tone: "green",
    items: [
      ["Goals", "Total goals scored", "G = total goals scored"],
      ["Assists", "Total assists", "A = total assists"],
      ["Points", "Goals plus assists", "P = G + A"],
      ["Games Played", "Total games played", "GP = games appeared in"],
      ["Shots on Goal", "Shots that reached the net", "SOG = shots on goal"],
      ["Hits", "Credited body checks", "HIT = total hits"],
      ["Blocks", "Opponent attempts blocked", "BLK = blocked shots"],
      ["PIM", "Penalty minutes", "PIM = penalty minutes taken"],
      ["PP Goals", "Power-play goals", "PPG = goals scored on PP"],
      ["PP Assists", "Power-play assists", "PPA = assists on PP goals"],
      ["SH Goals", "Short-handed goals", "SHG = goals while shorthanded"],
      ["SH Assists", "Short-handed assists", "SHA = assists while shorthanded"],
      ["GWG", "Game-winning goals", "GWG = winning-margin goals"],
      ["OT Goals", "Overtime goals", "OTG = overtime goals"],
      ["Faceoff Wins", "Draws won", "FOW = total faceoffs won"],
      ["Faceoff Losses", "Draws lost", "FOL = total faceoffs lost"],
      ["Takeaways", "Puck steals", "TAK = credited takeaways"],
      ["Giveaways", "Puck turnovers", "GIV = credited giveaways"],
      ["Plus Minus", "EV goal margin", "+/- = EV GF on ice − EV GA on ice"],
      ["TOI", "Time on ice", "TOI = total minutes played"],
    ],
  },
  {
    title: "Rate Stats",
    tone: "blue",
    items: [
      ["Goals/Game", "Goals per appearance", "G/GP = G ÷ GP"],
      ["Assists/Game", "Assists per appearance", "A/GP = A ÷ GP"],
      ["Points/Game", "Points per appearance", "P/GP = P ÷ GP"],
      ["Shots/Game", "Shots per appearance", "SOG/GP = SOG ÷ GP"],
      ["Hits/Game", "Hits per appearance", "HIT/GP = HIT ÷ GP"],
      ["Blocks/Game", "Blocks per appearance", "BLK/GP = BLK ÷ GP"],
      ["PIM/Game", "PIM per appearance", "PIM/GP = PIM ÷ GP"],
      ["Faceoff %", "Draw win rate", "FO% = FOW ÷ (FOW + FOL)"],
      ["Shooting %", "Goal conversion", "SH% = G ÷ SOG"],
      ["TOI/Game", "Usage per game", "TOI/GP = TOI ÷ GP"],
      ["Goals/60", "Goals scaled to 60 minutes", "G/60 = G × 60 ÷ TOI"],
      ["Assists/60", "Assists scaled to 60 minutes", "A/60 = A × 60 ÷ TOI"],
      ["Points/60", "Points scaled to 60 minutes", "P/60 = P × 60 ÷ TOI"],
      ["Shots/60", "Shots scaled to 60 minutes", "SOG/60 = SOG × 60 ÷ TOI"],
      ["Hits/60", "Hits scaled to 60 minutes", "HIT/60 = HIT × 60 ÷ TOI"],
      ["Blocks/60", "Blocks scaled to 60 minutes", "BLK/60 = BLK × 60 ÷ TOI"],
      ["Takeaways/60", "Steals scaled by ice time", "TAK/60 = TAK × 60 ÷ TOI"],
      ["Giveaways/60", "Turnovers scaled by ice time", "GIV/60 = GIV × 60 ÷ TOI"],
      ["PIM/60", "Penalty rate", "PIM/60 = PIM × 60 ÷ TOI"],
      ["EV Points/60", "Even-strength scoring rate", "EVP/60 = EVP × 60 ÷ EV TOI"],
    ],
  },
  {
    title: "Team Stats",
    tone: "gold",
    items: [
      ["Goals For", "Team goals scored", "GF = team goals"],
      ["Goals Against", "Goals allowed", "GA = opponent goals"],
      ["Goal Differential", "GF minus GA", "GD = GF − GA"],
      ["Win %", "Win rate", "Win% = W ÷ GP"],
      ["Points %", "Standings efficiency", "Pts% = points ÷ max points"],
      ["Shots For", "Team shots", "SF = shots for"],
      ["Shots Against", "Shots allowed", "SA = shots against"],
      ["Shot Differential", "SF minus SA", "Shot Diff = SF − SA"],
      ["Power Play %", "PP scoring rate", "PP% = PPG ÷ PPO"],
      ["Penalty Kill %", "PK prevention", "PK% = 1 − PPGA ÷ Opp PPO"],
      ["Team Save %", "Team goalie save rate", "SV% = (SA − GA) ÷ SA"],
      ["PDO", "Shooting plus save percentage", "PDO = SH% + SV%"],
      ["Team FO%", "Team draw win rate", "Team FO% = FOW ÷ (FOW + FOL)"],
      ["GF/Game", "Goals per game", "GF/GP = GF ÷ GP"],
      ["GA/Game", "Goals against per game", "GA/GP = GA ÷ GP"],
    ],
  },
  {
    title: "Possession",
    tone: "purple",
    items: [
      ["Corsi For", "All shot attempts for", "CF = SOG + missed + blocked attempts"],
      ["Corsi Against", "All attempts against", "CA = opponent shot attempts"],
      ["Corsi %", "Shot-attempt share", "CF% = CF ÷ (CF + CA)"],
      ["Fenwick For", "Unblocked attempts for", "FF = SOG + missed shots"],
      ["Fenwick Against", "Unblocked attempts against", "FA = opponent unblocked attempts"],
      ["Fenwick %", "Unblocked attempt share", "FF% = FF ÷ (FF + FA)"],
      ["Relative Corsi", "On-ice vs off-ice", "Rel CF% = on-ice CF% − off-ice CF%"],
      ["Attempt Differential", "CF minus CA", "CF Diff = CF − CA"],
      ["Shots For %", "Shot share", "SF% = SF ÷ (SF + SA)"],
      ["Zone Start %", "Offensive zone deployment", "ZS% = OZ starts ÷ OZ + DZ starts"],
      ["OZ Start Ratio", "O-zone share of shifts", "OZS ratio = OZ starts ÷ total shifts"],
      ["DZ Start Ratio", "D-zone share of shifts", "DZS ratio = DZ starts ÷ total shifts"],
      ["NZ Start %", "Neutral-zone start share", "NZS% = NZ starts ÷ total starts"],
      ["Possession Time %", "Puck time share", "Poss% = team possession time ÷ total time"],
      ["Controlled Entry %", "Entry quality", "Controlled Entry% = controlled entries ÷ total entries"],
    ],
  },
  {
    title: "Expected Goals",
    tone: "red",
    items: [
      ["Expected Goals", "Probability-weighted shot value", "xG = sum of shot goal probabilities"],
      ["xGF", "Expected goals for", "xGF = sum of team/player xG for"],
      ["xGA", "Expected goals against", "xGA = sum of opponent xG"],
      ["xG Differential", "Expected goal margin", "xG Diff = xGF − xGA"],
      ["xGF%", "Expected goal share (game avg)", "xGF% = average of each game's on-ice xGF%"],
      ["Goals Above Expected", "Finishing above chance quality", "Gax = Goals − xG"],
      ["Slot Shot %", "Slot chance share", "Slot Shot% = slot shots ÷ total shots"],
      ["High Danger Chances", "High-danger attempts", "HDCF = high-danger chances for"],
      ["High Danger %", "High-danger share", "HDCF% = HDCF ÷ (HDCF + HDCA)"],
      ["Medium Danger %", "Medium-danger share", "MDCF% = MDCF ÷ (MDCF + MDCA)"],
      ["Low Danger %", "Low-danger share", "LDCF% = LDCF ÷ (LDCF + LDCA)"],
      ["Rebound Shot %", "Rebound shot share", "Rebound% = rebound shots ÷ total shots"],
      ["Rush Chance %", "Rush shot share", "Rush% = rush shots ÷ total shots"],
      ["Finishing", "Goals minus individual xG", "Finishing = Goals − iXG"],
      ["Expected Shooting %", "xG per shot", "xSH% = xG ÷ SOG"],
    ],
  },
  {
    title: "Goalies",
    tone: "cyan",
    items: [
      ["Save %", "Saves divided by shots against", "SV% = saves ÷ shots against"],
      ["GAA", "Goals allowed per 60 minutes", "GAA = GA × 60 ÷ TOI"],
      ["Shutouts", "Games with zero GA", "SO = shutout games"],
      ["GSAx", "Goals saved above expected", "GSAx = xGA − GA"],
      ["HD Save %", "High-danger save rate", "HDSV% = HD saves ÷ HD shots"],
      ["MD Save %", "Medium-danger save rate", "MDSV% = MD saves ÷ MD shots"],
      ["LD Save %", "Low-danger save rate", "LDSV% = LD saves ÷ LD shots"],
      ["Rebound Control %", "Saves without rebound", "RC% = no-rebound saves ÷ total saves"],
      ["Rush Save %", "Rush chance save rate", "Rush SV% = rush saves ÷ rush shots"],
      ["Quality Start %", "Quality starts per start", "QS% = quality starts ÷ starts"],
    ],
  },
  {
    title: "WAR Value",
    tone: "orange",
    items: [
      ["WAR", "Wins above replacement", "WAR = total impact GAR ÷ goals per win"],
      ["Base WAR", "Core wins above replacement", "Base WAR = base GAR ÷ goals per win"],
      ["GAR", "Goals above replacement", "GAR = offensive GAR + defensive GAR + special teams GAR"],
      ["PAR", "Points above replacement", "PAR = player points − replacement points at usage"],
      ["GF%", "On-ice goal share", "GF% = GF_on ÷ (GF_on + GA_on)"],
      ["Skater WAR", "Total skater value", "WAR = (offensive GAR + defensive GAR + penalty GAR + faceoff GAR + possession GAR + playmaking GAR + special teams GAR) ÷ goals per win"],
      ["Goalie WAR", "Total goalie value", "Goalie WAR = (saved-goals value + quality-start value - bad-start drag) ÷ goals per win"],
      ["Offensive GAR", "Attack contribution", "Off GAR = production and individual xG above replacement at usage"],
      ["Defensive GAR", "Suppression contribution", "Def GAR = xGA suppression above replacement at usage"],
      ["Special Teams GAR", "PP/PK value", "ST GAR = special-teams production above replacement at usage"],
      ["Transition Value", "Entry/exit value", "Transition = controlled entries + exits − failed attempts"],
      ["Clutch Score", "Late/close game value", "Clutch = late goals + late assists + comeback points"],
    ],
  },
];

function FormulasTab() {
  return (
    <div className="sc-tab-page">
      <Section eyebrow="Reference" title="Hockey Analytics Formula Library">
        <div className="sc-formula-sections">
          {FORMULA_SECTIONS.map((section) => (
            <div key={section.title} className={`sc-formula-section is-${section.tone}`}>
              <header>
                <h3>{section.title}</h3>
                <span>{section.items.length} formulas</span>
              </header>

              <div className="sc-formula-list">
                {section.items.map(([name, desc, formula]) => (
                  <article key={`${section.title}-${name}`} className="sc-formula-card">
                    <strong>{name}</strong>
                    <span>{desc}</span>
                    <em>{formula}</em>
                  </article>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}


/* =========================================================
   GENERIC TIER LIST
========================================================= */

function TierList({ rows }) {
  return (
    <div className="sc-tier-list">
      {(rows || []).map((row) => (
        <div key={row.label} className={`sc-tier-row is-${row.tone || "neutral"}`}>
          <div>
            <strong>{row.label}</strong>
            <span>{row.sub}</span>
          </div>
          <b>{row.value}</b>
        </div>
      ))}
    </div>
  );
}
/* =========================================================
   STYLES
   Background system is matched to RosterScreen.js:
   - deep navy base
   - cyan glow
   - gold secondary glow
   - dark glass panels
   - NOT copying roster buttons directly
========================================================= */

function StatsCentralRedesignStyles() {
  return (
    <style>{`
      .stats-central-screen {
        height: 100vh;
        min-height: 0;
        overflow: hidden;
        background:
          radial-gradient(circle at 15% 0%, rgba(0, 206, 222, 0.08), transparent 28%),
          linear-gradient(180deg, #06131f 0%, #07111c 100%);
      }

      .statscentral-shell {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-rows: 58px minmax(0, 1fr);
        overflow: hidden;
      }

      .sc-command-bar {
        min-height: 0;
        height: 58px;
        padding: 7px 10px;
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr) minmax(180px, 270px);
        gap: 8px;
        align-items: stretch;
        border-bottom: 1px solid rgba(97, 177, 212, 0.18);
        background: rgba(5, 20, 32, 0.96);
      }

      .sc-back-link {
        min-width: 0;
        height: 42px;
        align-self: center;
        border: 1px solid rgba(116, 174, 205, 0.22);
        background: #091a29;
        color: #d9edf7;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.12em;
      }

      .sc-menu {
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 7px;
      }

      .sc-menu button {
        min-width: 0;
        height: 42px;
        align-self: center;
        padding: 0 18px;
        border: 1px solid rgba(116, 174, 205, 0.18);
        background: #081826;
        color: #8faabd;
        text-align: left;
      }

      .sc-menu button em {
        font-size: 11px;
        font-style: normal;
        font-weight: 900;
        letter-spacing: 0.14em;
      }

      .sc-menu button.is-active {
        border-color: rgba(0, 225, 235, 0.74);
        background:
          linear-gradient(90deg, rgba(0, 213, 226, 0.18), rgba(0, 213, 226, 0.03)),
          #092033;
        color: #ffffff;
        box-shadow: inset 3px 0 0 #04dce7;
      }

      .sc-command-context {
        min-width: 0;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 12px;
        color: #6f91a7;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
      }

      .sc-content {
        min-height: 0;
        overflow: hidden;
        padding: 10px;
      }

      .sc-content.is-player-stats,
      .sc-content.is-league-leaders,
      .sc-content.is-team-stats {
        height: 100%;
      }

      .sc-player-workspace,
      .sc-league-leaders-workspace {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        gap: 8px;
      }

      .sc-league-leaders-workspace {
        grid-template-rows: auto minmax(0, 1fr);
      }

      .sc-player-header {
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(150px, 205px) minmax(0, 1fr) minmax(315px, 405px);
        gap: 10px;
        align-items: center;
        padding: 8px 10px;
        border: 1px solid rgba(98, 166, 199, 0.2);
        background: #081927;
      }

      .sc-player-heading {
        min-width: 0;
        display: grid;
        line-height: 1.05;
      }

      .sc-player-heading > span,
      .sc-player-table-header > div > span,
      .sc-overview-team-header-identity > div > span,
      .sc-overview-module-header span,
      .sc-player-insight-panel > header > span,
      .sc-special-team-panel > header > span,
      .sc-trend-panel > header > span,
      .sc-league-leaders-header > div > span,
      .sc-leader-category-header > div > span {
        color: #f1b63d;
        font-size: 9px;
        font-weight: 1000;
        letter-spacing: 0.16em;
      }

      .sc-player-heading > strong {
        margin-top: 3px;
        color: #f6fbff;
        font-size: 15px;
        font-weight: 1000;
      }

      .sc-player-heading > em {
        margin-top: 4px;
        color: #6f91a7;
        font-size: 9px;
        font-style: normal;
        font-weight: 800;
      }

      .sc-player-subnav {
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 4px;
      }

      .sc-player-subnav button,
      .sc-league-leaders-header nav button,
      .sc-leader-category-header nav button,
      .sc-compare-category-tabs button {
        min-width: 0;
        min-height: 32px;
        padding: 0 8px;
        border: 1px solid transparent;
        background: transparent;
        color: #7894a7;
        font-size: 9px;
        font-weight: 950;
        letter-spacing: 0.06em;
        white-space: nowrap;
      }

      .sc-player-subnav button:hover,
      .sc-league-leaders-header nav button:hover,
      .sc-leader-category-header nav button:hover,
      .sc-compare-category-tabs button:hover {
        color: #c9e5f2;
        background: rgba(118, 177, 204, 0.07);
      }

      .sc-player-subnav button.is-active,
      .sc-league-leaders-header nav button.is-active,
      .sc-leader-category-header nav button.is-active,
      .sc-compare-category-tabs button.is-active {
        border-color: rgba(0, 218, 229, 0.46);
        background: rgba(0, 203, 216, 0.12);
        color: #f6fdff;
      }

      .sc-player-header-actions {
        min-width: 0;
        display: grid;
        grid-template-columns: auto minmax(150px, 1fr);
        gap: 7px;
      }

      .sc-player-scope,
      .sc-player-density,
      .sc-player-filter-group {
        display: flex;
        align-items: center;
        border: 1px solid rgba(102, 166, 194, 0.18);
        background: #071522;
      }

      .sc-player-scope button,
      .sc-player-density button,
      .sc-player-filter-group button {
        min-height: 30px;
        border: 0;
        background: transparent;
        color: #7794a7;
        font-size: 9px;
        font-weight: 950;
        letter-spacing: 0.05em;
      }

      .sc-player-scope button {
        padding: 0 11px;
      }

      .sc-player-scope button.is-active,
      .sc-player-density button.is-active,
      .sc-player-filter-group button.is-active {
        background: rgba(228, 175, 59, 0.14);
        color: #ffffff;
      }

      .sc-player-search,
      .sc-team-search {
        min-width: 0;
        min-height: 32px;
        display: grid;
        grid-template-columns: 24px minmax(0, 1fr);
        align-items: center;
        padding: 0 8px;
        border: 1px solid rgba(102, 166, 194, 0.2);
        background: #071522;
      }

      .sc-player-search span {
        color: #10d7e1;
        font-weight: 900;
      }

      .sc-player-search input,
      .sc-player-filter-bar select,
      .sc-player-min-gp input,
      .sc-compare-selector-row select {
        min-width: 0;
        border: 0;
        outline: 0;
        background: transparent;
        color: #dcecf4;
        font-size: 10px;
        font-weight: 800;
      }

      .sc-player-search input::placeholder {
        color: #587487;
      }

      .sc-player-filter-bar {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 6px 8px;
        border: 1px solid rgba(98, 166, 199, 0.16);
        background: rgba(7, 23, 35, 0.92);
      }

      .sc-player-filter-group button {
        min-width: 38px;
        padding: 0 8px;
      }

      .sc-player-filter-bar select,
      .sc-player-min-gp,
      .sc-player-density {
        height: 30px;
        border: 1px solid rgba(102, 166, 194, 0.18);
        background: #071522;
      }

      .sc-player-filter-bar select {
        padding: 0 26px 0 9px;
      }

      .sc-player-filter-bar select option,
      .sc-compare-selector-row select option {
        background: #091927;
        color: #e2f2f8;
      }

      .sc-player-min-gp {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 0 8px;
      }

      .sc-player-min-gp span {
        color: #6f8da0;
        font-size: 9px;
        font-weight: 900;
      }

      .sc-player-min-gp input {
        width: 40px;
        text-align: right;
      }

      .sc-player-density button {
        padding: 0 9px;
      }

      .sc-player-filter-reset {
        margin-left: auto;
        height: 30px;
        padding: 0 12px;
        border: 1px solid rgba(212, 164, 56, 0.26);
        background: rgba(212, 164, 56, 0.07);
        color: #d7a93d;
        font-size: 9px;
        font-weight: 950;
        letter-spacing: 0.06em;
      }

      .sc-player-panel,
      .sc-league-leaders-panel {
        min-height: 0;
        overflow: hidden;
      }

      .sc-tab-page {
        height: 100%;
        min-height: 0;
        overflow: hidden;
      }

      .sc-player-state-message {
        height: 100%;
        display: grid;
        place-content: center;
        gap: 6px;
        border: 1px solid rgba(98, 166, 199, 0.2);
        background: #081927;
        color: #b9d0dc;
        text-align: center;
      }

      .sc-player-state-message strong {
        color: #ffffff;
      }

      .sc-player-overview {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        gap: 10px;
        overflow: hidden;
      }

      .sc-overview-team-header {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(280px, 1fr);
        gap: 16px;
        align-items: center;
        padding: 14px 16px;
        background:
          linear-gradient(90deg, rgba(0, 204, 218, 0.12), transparent 46%),
          #081927;
      }

      .sc-overview-team-header-identity {
        min-width: 0;
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr);
        gap: 14px;
        align-items: center;
      }

      .sc-overview-team-header-identity .sc-team-logo-mark.is-large {
        width: 72px;
        height: 72px;
      }

      .sc-overview-team-header-identity span {
        color: #7ea0b3;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 0.1em;
      }

      .sc-overview-team-header-identity h2 {
        margin: 4px 0 0;
        color: #f7fcff;
        font-size: 26px;
        line-height: 1;
        font-weight: 950;
      }

      .sc-overview-team-header-identity p {
        margin: 6px 0 0;
        color: #6f90a3;
        font-size: 11px;
        font-weight: 750;
      }

      .sc-overview-team-header-chips {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
        gap: 8px;
      }

      .sc-overview-team-header-chips > div {
        min-width: 0;
        padding: 8px 10px;
        background: rgba(8, 28, 42, 0.72);
      }

      .sc-overview-team-header-chips em {
        display: block;
        color: #7897a9;
        font-size: 8px;
        font-style: normal;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sc-overview-team-header-chips strong {
        display: block;
        margin-top: 4px;
        color: #5fd4e0;
        font-size: 14px;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
      }

      .sc-overview-featured-row {
        display: grid;
        grid-template-columns: minmax(240px, 1.55fr) repeat(5, minmax(0, 1fr));
        gap: 8px;
        min-height: 0;
      }

      .sc-overview-leader-tile {
        min-width: 0;
        display: grid;
        gap: 8px;
        padding: 10px 11px;
        border: 0;
        background: #0a1b2a;
        text-align: left;
        cursor: pointer;
      }

      .sc-overview-leader-tile:hover:not(:disabled) {
        background: #0d2436;
        box-shadow: inset 0 0 0 1px rgba(0, 215, 226, 0.28);
      }

      .sc-overview-leader-tile:disabled {
        cursor: default;
        opacity: 0.7;
      }

      .sc-overview-leader-tile.is-featured {
        background:
          linear-gradient(135deg, rgba(0, 204, 218, 0.12), transparent 55%),
          #0a1f30;
      }

      .sc-overview-leader-eyebrow {
        color: #718fa2;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sc-overview-leader-tile-body {
        display: grid;
        grid-template-columns: 52px minmax(0, 1fr);
        gap: 10px;
        align-items: center;
      }

      .sc-overview-leader-tile.is-featured .sc-overview-leader-tile-body {
        grid-template-columns: 78px minmax(0, 1fr);
      }

      .sc-overview-leader-portrait {
        position: relative;
        width: 52px;
        height: 52px;
        flex: 0 0 52px;
        z-index: 1;
      }

      .sc-overview-leader-tile.is-featured .sc-overview-leader-portrait {
        width: 78px;
        height: 78px;
        flex-basis: 78px;
      }

      .sc-overview-leader-portrait .sc-avatar {
        width: 100%;
        height: 100%;
      }

      .sc-overview-leader-portrait .sc-avatar.is-large,
      .sc-avatar.is-large {
        width: 78px;
        height: 78px;
        border-radius: 14px;
      }

      .sc-overview-leader-portrait .sc-avatar-team-logo {
        z-index: 2;
      }

      .sc-overview-leader-copy {
        min-width: 0;
        display: grid;
        gap: 3px;
      }

      .sc-overview-leader-copy > strong {
        overflow: hidden;
        color: #f2f8fb;
        font-size: 13px;
        font-weight: 950;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-overview-leader-tile.is-featured .sc-overview-leader-copy > strong {
        font-size: 18px;
      }

      .sc-overview-leader-copy > em {
        overflow: hidden;
        color: #89a7b8;
        font-size: 10px;
        font-style: normal;
        font-weight: 800;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-overview-leader-metric {
        display: inline-flex;
        align-items: baseline;
        gap: 6px;
      }

      .sc-overview-leader-metric b {
        color: #f0b63b;
        font-size: 16px;
        font-weight: 950;
        font-variant-numeric: tabular-nums;
      }

      .sc-overview-leader-tile.is-featured .sc-overview-leader-metric b {
        font-size: 22px;
      }

      .sc-overview-leader-metric i {
        color: #7ea0b3;
        font-size: 9px;
        font-style: normal;
        font-weight: 900;
        letter-spacing: 0.06em;
      }

      .sc-overview-leader-copy > p {
        margin: 0;
        overflow: hidden;
        color: #6f90a3;
        font-size: 10px;
        font-weight: 750;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-overview-main-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 0.68fr) minmax(0, 0.32fr);
        gap: 10px;
        overflow: hidden;
      }

      .sc-overview-side-stack {
        min-height: 0;
        display: grid;
        grid-template-rows: minmax(0, 1.35fr) auto auto;
        gap: 8px;
        overflow: hidden;
      }

      .sc-overview-module {
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        background: #081927;
        overflow: hidden;
      }

      .sc-overview-module-header {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px 8px;
      }

      .sc-overview-module-header span {
        display: block;
        color: #748fa1;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: 0.1em;
      }

      .sc-overview-module-header strong {
        display: block;
        margin-top: 3px;
        color: #eef7fb;
        font-size: 14px;
        font-weight: 950;
      }

      .sc-overview-board-controls {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 6px;
      }

      .sc-overview-chip-group {
        display: inline-flex;
        gap: 4px;
      }

      .sc-overview-chip-group button {
        padding: 5px 8px;
        border: 0;
        background: rgba(14, 36, 52, 0.95);
        color: #8eacbd;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        cursor: pointer;
      }

      .sc-overview-chip-group button.is-active,
      .sc-overview-chip-group button:hover {
        background: rgba(0, 204, 218, 0.16);
        color: #5fd4e0;
      }

      .sc-overview-scoring-leaders .sc-table-wrap {
        min-height: 0;
        height: 100%;
        max-height: none;
        overflow-x: hidden;
        overflow-y: auto;
        border: 0;
        background: transparent;
      }

      .sc-overview-scoring-table {
        min-width: 0 !important;
        width: 100%;
      }

      .sc-overview-scoring-table thead th {
        position: static;
        height: 24px;
        padding: 0 6px;
        font-size: 8px;
        background: rgba(10, 31, 48, 0.95);
      }

      .sc-overview-scoring-table thead th.is-sorted {
        color: #5fd4e0;
      }

      .sc-overview-scoring-table td {
        height: auto;
        padding: 3px 6px;
        font-size: 10px;
      }

      .sc-overview-scoring-table tbody tr:hover td {
        background: rgba(0, 216, 223, 0.06);
      }

      .sc-overview-table-player {
        min-width: 0;
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr);
        gap: 8px;
        align-items: center;
      }

      .sc-overview-table-player .sc-avatar {
        width: 28px;
        height: 28px;
        z-index: 1;
      }

      .sc-overview-table-player span {
        min-width: 0;
        display: grid;
        gap: 1px;
      }

      .sc-overview-table-player strong {
        overflow: hidden;
        color: #f1f8fb;
        font-size: 11px;
        font-weight: 900;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-overview-table-player em {
        overflow: hidden;
        color: #7f9cb0;
        font-size: 9px;
        font-style: normal;
        font-weight: 800;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-overview-pts {
        color: #5fd4e0;
        font-weight: 950;
      }

      .sc-overview-metric-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
        padding: 0 10px 10px;
        align-content: start;
        overflow: hidden;
      }

      .sc-overview-metric-card {
        min-width: 0;
        display: grid;
        gap: 2px;
        padding: 8px 9px;
        background: rgba(8, 28, 42, 0.88);
      }

      .sc-overview-metric-card > span {
        color: #738fa1;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: 0.07em;
        text-transform: uppercase;
      }

      .sc-overview-metric-card > strong {
        color: #f2f8fb;
        font-size: 15px;
        line-height: 1.05;
        font-variant-numeric: tabular-nums;
      }

      .sc-overview-metric-card.is-good > strong {
        color: #5fd4a2;
      }

      .sc-overview-metric-card.is-bad > strong {
        color: #ff8a8a;
      }

      .sc-overview-metric-card > em {
        color: #87a4b5;
        font-size: 9px;
        font-style: normal;
        font-weight: 750;
      }

      .sc-overview-metric-card > b {
        color: #f0b63b;
        font-size: 9px;
        font-weight: 900;
      }

      .sc-overview-ref-bar {
        position: relative;
        height: 4px;
        margin-top: 4px;
        background: rgba(95, 153, 179, 0.18);
        overflow: hidden;
      }

      .sc-overview-ref-bar > i {
        display: block;
        height: 100%;
        background: #5fd4e0;
      }

      .sc-overview-ref-bar > em {
        position: absolute;
        top: 0;
        left: 50%;
        width: 1px;
        height: 100%;
        background: rgba(240, 182, 59, 0.95);
      }

      .sc-overview-contribution-list,
      .sc-overview-read-list {
        display: grid;
        gap: 6px;
        padding: 0 10px 10px;
        align-content: start;
      }

      .sc-overview-contribution-list > div,
      .sc-overview-read-list > div,
      .sc-overview-read-list > button {
        display: grid;
        gap: 2px;
        padding: 8px 9px;
        border: 0;
        background: rgba(8, 28, 42, 0.88);
        text-align: left;
        color: inherit;
      }

      .sc-overview-read-list > button {
        cursor: pointer;
      }

      .sc-overview-read-list > button:hover {
        background: rgba(0, 216, 223, 0.08);
      }

      .sc-overview-contribution-list span,
      .sc-overview-read-list span {
        color: #738fa1;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: 0.07em;
        text-transform: uppercase;
      }

      .sc-overview-contribution-list strong,
      .sc-overview-read-list strong {
        color: #eef7fb;
        font-size: 12px;
        font-weight: 900;
      }

      .sc-overview-contribution-list em {
        color: #87a4b5;
        font-size: 9px;
        font-style: normal;
      }

      .sc-player-insight-panel {
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        border: 1px solid rgba(105, 172, 202, 0.18);
        background: #081927;
      }

      .sc-player-insight-panel > header,
      .sc-special-team-panel > header,
      .sc-trend-panel > header {
        display: grid;
        gap: 3px;
        padding: 10px 11px;
        border-bottom: 1px solid rgba(105, 172, 202, 0.14);
      }

      .sc-player-insight-panel > header > strong,
      .sc-special-team-panel > header > strong,
      .sc-trend-panel > header > strong {
        color: #eff8fc;
        font-size: 12px;
      }

      .sc-player-insight-panel > div {
        min-height: 0;
        display: grid;
        align-content: start;
      }

      .sc-player-insight-panel article {
        min-width: 0;
        display: block;
        border-bottom: 1px solid rgba(95, 153, 179, 0.1);
      }

      .sc-player-insight-main {
        width: 100%;
        min-width: 0;
        display: block;
        padding: 8px 9px;
        border: 0;
        background: transparent;
        text-align: left;
        cursor: pointer;
      }

      .sc-player-insight-main > span {
        min-width: 0;
        display: grid;
        gap: 2px;
      }

      .sc-player-insight-main strong {
        overflow: hidden;
        color: #eaf5fa;
        font-size: 10px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-player-insight-main em {
        overflow: hidden;
        color: #7895a7;
        font-size: 8px;
        font-style: normal;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-player-insight-compare,
      .sc-inline-compare {
        height: 26px;
        margin-right: 7px;
        padding: 0 8px;
        border: 1px solid rgba(0, 207, 218, 0.2);
        background: rgba(0, 207, 218, 0.05);
        color: #58cbd2;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-player-table-header {
        min-height: 48px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 7px 10px;
        border: 1px solid rgba(105, 172, 202, 0.18);
        background: #081927;
      }

      .sc-player-table-header > div:first-child {
        display: grid;
        gap: 2px;
      }

      .sc-player-table-header > div:first-child > strong {
        color: #f2f9fc;
        font-size: 14px;
      }

      .sc-player-table-header > div:first-child > em {
        color: #6c8a9c;
        font-size: 8px;
        font-style: normal;
        font-weight: 800;
      }

      .sc-column-preset-toggle {
        display: flex;
        gap: 3px;
        padding: 3px;
        border: 1px solid rgba(103, 169, 198, 0.18);
        background: #071522;
      }

      .sc-column-preset-toggle button {
        min-height: 27px;
        padding: 0 9px;
        border: 0;
        background: transparent;
        color: #7692a4;
        font-size: 8px;
        font-weight: 950;
      }

      .sc-column-preset-toggle button.is-active {
        background: rgba(0, 207, 218, 0.14);
        color: #ffffff;
      }

      .sc-skaters-page-v3,
      .sc-goalies-page-v2,
      .sc-analytics-page-v2,
      .sc-trends-page-v2,
      .sc-compare-page-v2 {
        min-height: 0;
        display: grid;
        gap: 8px;
      }

      .sc-skaters-page-v3 {
        grid-template-rows: auto minmax(0, 1fr);
      }

      .sc-roster-groups {
        min-height: 0;
        display: grid;
        grid-template-rows: minmax(0, 1.75fr) minmax(0, 1fr);
        gap: 7px;
      }

      .sc-roster-groups > section {
        min-height: 0;
        display: grid;
        grid-template-rows: 26px minmax(0, 1fr);
        overflow: hidden;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #071622;
      }

      .sc-roster-groups > section > header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 9px;
        border-bottom: 1px solid rgba(105, 172, 202, 0.12);
        color: #7695a7;
        font-size: 8px;
        font-weight: 950;
        letter-spacing: 0.12em;
      }

      .sc-table-wrap {
        min-width: 0;
        max-width: 100%;
        min-height: 0;
        overflow-x: auto;
        overflow-y: auto;
        border: 1px solid rgba(104, 170, 198, 0.14);
        background: #071622;
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 202, 214, 0.35) transparent;
      }

      .sc-roster-groups .sc-table-wrap,
      .sc-paged-table .sc-table-wrap {
        height: 100%;
        min-height: 0;
        overflow-y: auto;
      }

      .sc-table {
        width: 100%;
        min-width: 780px;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
      }

      .sc-table thead th {
        position: sticky;
        top: 0;
        z-index: 8;
        height: 29px;
        padding: 0 8px;
        border-bottom: 1px solid rgba(0, 206, 218, 0.24);
        background: #06141f;
        color: #7e9caf;
        font-size: 8px;
        font-weight: 1000;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .sc-table thead th.is-sorted {
        background: #082332;
        color: #07d8e1;
      }

      .sc-table td {
        height: 40px;
        padding: 3px 8px;
        border-bottom: 1px solid rgba(91, 146, 172, 0.1);
        color: #d8e9f1;
        font-size: 12px;
      }

      .sc-table-wrap.is-comfortable .sc-table td {
        height: 48px;
        padding-top: 6px;
        padding-bottom: 6px;
        font-size: 13px;
      }

      .sc-table tbody tr:hover td {
        background: rgba(0, 189, 203, 0.045);
      }

      .sc-table tbody tr.is-selected-row td {
        background: rgba(0, 206, 218, 0.1);
        box-shadow: inset 0 1px 0 rgba(0, 206, 218, 0.16),
          inset 0 -1px 0 rgba(0, 206, 218, 0.16);
      }

      .sc-table .is-player-col {
        position: sticky;
        left: 0;
        z-index: 6;
        width: 260px;
        min-width: 260px;
        background: #071622;
      }

      .sc-table thead .is-player-col {
        z-index: 10;
        background: #06141f;
      }

      .sc-table tbody tr:hover .is-player-col {
        background: #0a202e;
      }

      .sc-table tbody tr.is-selected-row .is-player-col {
        background: #0a2b38;
      }

      .sc-table .is-actions-col {
        width: 112px;
        min-width: 112px;
      }

      .sc-player-row-actions {
        display: flex;
        justify-content: flex-end;
        gap: 4px;
      }

      .sc-player-row-actions button {
        min-height: 23px;
        padding: 0 7px;
        border: 1px solid rgba(100, 165, 192, 0.18);
        background: transparent;
        color: #708fa1;
        font-size: 7px;
        font-weight: 950;
        text-transform: uppercase;
      }

      .sc-player-row-actions button:hover,
      .sc-player-row-actions button.is-active {
        border-color: rgba(0, 213, 223, 0.4);
        color: #dffbff;
        background: rgba(0, 201, 212, 0.1);
      }

      .sc-name-cell {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .sc-player-name-cell-simple .sc-player-team-logo {
        flex: 0 0 auto;
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid rgba(156, 218, 236, 0.22);
        background: rgba(8, 24, 35, 0.94);
      }

      .sc-player-name-cell-simple .sc-player-team-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
      }

      .sc-player-name-cell-simple .sc-player-team-logo.is-fallback {
        color: #fff;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.04em;
      }

      .sc-player-name-cell-simple .sc-player-name-copy {
        min-width: 0;
        display: flex;
        align-items: baseline;
        gap: 12px;
        width: 100%;
      }

      .sc-player-name-cell-simple .sc-player-name-copy > strong {
        min-width: 0;
        overflow: hidden;
        color: #f7fcff;
        font-size: 14px;
        line-height: 1.15;
        font-weight: 950;
        letter-spacing: -0.01em;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-player-name-cell-simple .sc-player-meta-inline {
        flex: 0 0 auto;
        margin-left: auto;
        display: inline-flex;
        align-items: baseline;
        gap: 10px;
        white-space: nowrap;
      }

      .sc-player-name-cell-simple .sc-player-pos-inline {
        color: #9eb8c6;
        font-size: 12px;
        font-style: normal;
        font-weight: 850;
        letter-spacing: 0.06em;
        line-height: 1;
        cursor: help;
      }

      .sc-player-name-cell-simple .sc-player-ovr-inline {
        color: #5fd4e0;
        font-size: 13px;
        font-style: normal;
        font-weight: 900;
        letter-spacing: 0.02em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        cursor: help;
        display: inline-flex;
        align-items: baseline;
        gap: 4px;
        background: none;
        border: 0;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
      }

      .sc-player-name-cell-simple .sc-player-ovr-inline.is-dropped {
        color: #7fd7e0;
      }

      .sc-player-name-cell-simple .sc-player-ovr-drop {
        color: #ff8a8a;
        font-size: 11px;
        font-style: normal;
        font-weight: 900;
        letter-spacing: 0.01em;
        line-height: 1;
      }

      .sc-player-name-cell-simple span:not(.sc-player-meta-inline):not(.sc-player-ovr-drop) {
        color: #8facbc;
        font-size: 8px;
        font-weight: 800;
        letter-spacing: 0.06em;
      }

      .sc-name-cell > div:last-child {
        min-width: 0;
      }

      .sc-name-cell strong {
        overflow: hidden;
        display: block;
        color: #f1f8fb;
        font-size: 10px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-name-cell span:not(.sc-player-meta-inline):not(.sc-player-ovr-drop) {
        overflow: hidden;
        display: block;
        margin-top: 2px;
        color: #607f92;
        font-size: 7px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-name-cell.is-team-scope .sc-avatar-team-logo {
        display: none;
      }

      .sc-skater-stat {
        display: grid;
        justify-items: end;
        gap: 1px;
      }

      .sc-skater-stat strong {
        color: #dfeef5;
        font-size: 13px;
        font-weight: 900;
        line-height: 1.05;
      }

      .sc-skater-stat span {
        color: #557486;
        font-size: 8px;
        font-weight: 800;
      }

      .sc-skater-stat.is-elite strong,
      .sc-impact-text.is-elite {
        color: #09e3d1;
      }

      .sc-skater-stat.is-good strong,
      .sc-impact-text.is-good {
        color: #3bd2a2;
      }

      .sc-skater-stat.is-warn strong,
      .sc-impact-text.is-warn {
        color: #f0b342;
      }

      .sc-skater-stat.is-bad strong,
      .sc-impact-text.is-bad {
        color: #ff6b72;
      }

      .sc-player-role-text {
        color: #9db6c3;
        font-size: 9px;
        font-weight: 800;
      }

      .sc-paged-table {
        min-height: 0;
        height: 100%;
        display: grid;
        grid-template-rows: minmax(0, 1fr) 34px;
        overflow: hidden;
      }

      .sc-table-pagination {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 0 9px;
        border: 1px solid rgba(103, 169, 198, 0.14);
        border-top: 0;
        background: #071522;
        color: #648395;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-table-pagination > div {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .sc-table-pagination button {
        height: 23px;
        padding: 0 9px;
        border: 1px solid rgba(103, 169, 198, 0.18);
        background: transparent;
        color: #8ba6b5;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-table-pagination button:disabled {
        opacity: 0.35;
      }

      .sc-table-pagination strong {
        color: #d8e9f1;
      }

      .sc-goalies-page-v2 {
        grid-template-rows: auto auto minmax(0, 1fr);
      }

      .sc-goalie-summary-row,
      .sc-trend-summary-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 7px;
      }

      .sc-goalie-summary-row button {
        min-width: 0;
        display: grid;
        gap: 3px;
        padding: 8px 10px;
        border: 1px solid rgba(105, 172, 202, 0.17);
        background: #091a28;
        text-align: left;
      }

      .sc-goalie-summary-row span {
        color: #6c8a9d;
        font-size: 8px;
        font-weight: 900;
        text-transform: uppercase;
      }

      .sc-goalie-summary-row strong {
        overflow: hidden;
        color: #eff8fc;
        font-size: 10px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-goalie-summary-row em {
        color: #09dbc9;
        font-size: 12px;
        font-style: normal;
        font-weight: 1000;
      }

      .sc-analytics-page-v2 {
        grid-template-rows: auto auto minmax(0, 1fr);
      }

      .sc-analytics-insights {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-analytics-insights > div {
        min-width: 0;
        display: grid;
        grid-template-columns: auto minmax(130px, 1fr);
        grid-template-rows: auto auto;
        align-items: center;
        gap: 2px 12px;
        padding: 8px 10px;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-analytics-insights > div > span {
        grid-row: 1;
        color: #f0b63d;
        font-size: 8px;
        font-weight: 950;
        letter-spacing: 0.12em;
      }

      .sc-analytics-insights > div > strong {
        grid-row: 2;
        color: #edf7fb;
        font-size: 11px;
      }

      .sc-analytics-insights .sc-compact-player-list {
        grid-column: 2;
        grid-row: 1 / span 2;
      }

      .sc-compact-player-list {
        min-width: 0;
        display: grid;
        gap: 2px;
      }

      .sc-compact-player-list button {
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 8px;
        padding: 5px 7px;
        border: 0;
        border-bottom: 1px solid rgba(100, 157, 183, 0.09);
        background: transparent;
        text-align: left;
      }

      .sc-compact-player-list button:hover {
        background: rgba(0, 194, 207, 0.05);
      }

      .sc-compact-player-list button > span {
        min-width: 0;
        display: grid;
        gap: 1px;
      }

      .sc-compact-player-list strong {
        overflow: hidden;
        color: #eaf5fa;
        font-size: 9px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-compact-player-list em {
        overflow: hidden;
        color: #657f91;
        font-size: 7px;
        font-style: normal;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-compact-player-list b {
        color: #f0b63d;
        font-size: 9px;
      }

      .sc-compact-player-list b.is-good {
        color: #36d5a1;
      }

      .sc-compact-player-list b.is-bad {
        color: #ff6c74;
      }

      .sc-special-teams-page {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-special-team-panel {
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        overflow: hidden;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-special-team-panel > header > em {
        color: #668596;
        font-size: 8px;
        font-style: normal;
      }

      .sc-special-team-panel .sc-table {
        min-width: 650px;
      }

      .sc-trends-page-v2 {
        grid-template-rows: auto auto minmax(0, 1fr);
      }

      .sc-trend-summary-row .sc-stat-card {
        min-height: 68px;
        padding: 9px 11px;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-trend-content-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-trend-panel {
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-compare-page-v2 {
        grid-template-rows: auto auto minmax(0, 1fr);
      }

      .sc-compare-category-tabs {
        display: flex;
        gap: 4px;
      }

      .sc-compare-selector-row {
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
        gap: 8px;
      }

      .sc-compare-selector-row > label {
        min-width: 0;
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        align-items: center;
        gap: 8px;
        padding: 7px 9px;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-compare-selector-row > label > span {
        color: #718fa2;
        font-size: 8px;
        font-weight: 900;
        text-transform: uppercase;
      }

      .sc-compare-selector-row select {
        height: 28px;
        border: 1px solid rgba(105, 172, 202, 0.18);
        padding: 0 8px;
        background: #071522;
      }

      .sc-compare-pinned {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 0 8px;
        border: 1px solid rgba(105, 172, 202, 0.16);
        background: #081927;
      }

      .sc-compare-pinned > span {
        color: #6f8ca0;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-compare-pinned button {
        height: 24px;
        padding: 0 7px;
        border: 1px solid rgba(0, 203, 214, 0.2);
        background: transparent;
        color: #69cfd4;
        font-size: 8px;
      }

      .sc-compare-main-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(210px, 0.7fr) minmax(380px, 1.5fr) minmax(210px, 0.7fr);
        gap: 8px;
      }

      .sc-compare-card,
      .sc-compare-metric-board {
        min-height: 0;
        border: 1px solid rgba(105, 172, 202, 0.17);
        background: #081927;
      }

      .sc-compare-card {
        display: grid;
        align-content: start;
        gap: 14px;
        padding: 14px;
      }

      .sc-compare-card-top {
        display: grid;
        grid-template-columns: 58px minmax(0, 1fr);
        gap: 11px;
        align-items: center;
      }

      .sc-compare-card-top > div:last-child {
        min-width: 0;
        display: grid;
        gap: 3px;
      }

      .sc-compare-card-top span {
        color: #f0b63d;
        font-size: 8px;
        font-weight: 950;
      }

      .sc-compare-card-top strong {
        overflow: hidden;
        color: #f5fbfe;
        font-size: 15px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-compare-card-top em {
        color: #6d899b;
        font-size: 8px;
        font-style: normal;
      }

      .sc-compare-card-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 7px;
      }

      .sc-compare-card-grid span {
        display: grid;
        gap: 5px;
        padding: 8px;
        border: 1px solid rgba(105, 172, 202, 0.13);
        color: #6e8b9e;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-compare-card-grid b {
        color: #f0f8fb;
        font-size: 14px;
      }

      .sc-compare-metric-board {
        display: grid;
        align-content: stretch;
        grid-auto-rows: minmax(0, 1fr);
        overflow: hidden;
      }

      .sc-compare-metric {
        min-height: 0;
        display: grid;
        align-content: center;
        gap: 5px;
        padding: 7px 10px;
        border-bottom: 1px solid rgba(105, 172, 202, 0.1);
      }

      .sc-compare-metric-values {
        display: grid;
        grid-template-columns: 1fr minmax(120px, 1.2fr) 1fr;
        gap: 10px;
        align-items: center;
        text-align: center;
      }

      .sc-compare-metric-values strong {
        color: #edf7fb;
        font-size: 11px;
      }

      .sc-compare-metric-values strong:first-child {
        text-align: left;
      }

      .sc-compare-metric-values strong:last-child {
        text-align: right;
      }

      .sc-compare-metric-values strong.is-good {
        color: #3ad5a1;
      }

      .sc-compare-metric-values strong.is-bad {
        color: #ff6c74;
      }

      .sc-compare-metric-values span {
        color: #6c899b;
        font-size: 8px;
        font-weight: 950;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .sc-compare-bars {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }

      .sc-compare-bars > div {
        height: 3px;
        overflow: hidden;
        background: rgba(91, 145, 169, 0.13);
      }

      .sc-compare-bars > div:first-child {
        display: flex;
        justify-content: flex-end;
      }

      .sc-compare-bars i {
        display: block;
        height: 100%;
        background: #08cdd8;
      }

      .sc-compare-bars > div:last-child i {
        background: #e2ad3b;
      }

      .sc-player-drawer {
        position: fixed;
        z-index: 1200;
        top: 68px;
        right: 10px;
        bottom: 10px;
        width: min(360px, calc(100vw - 20px));
        display: grid;
        grid-template-rows: auto auto auto minmax(0, 1fr);
        gap: 12px;
        padding: 16px;
        border: 1px solid rgba(0, 214, 224, 0.4);
        background:
          linear-gradient(180deg, rgba(0, 204, 216, 0.06), transparent 30%),
          #071622;
        box-shadow: -20px 0 50px rgba(0, 0, 0, 0.38);
        overflow-y: auto;
      }

      .sc-player-drawer-close {
        position: absolute;
        top: 8px;
        right: 8px;
        width: 28px;
        height: 28px;
        border: 1px solid rgba(105, 172, 202, 0.18);
        background: transparent;
        color: #91aab8;
        font-size: 18px;
      }

      .sc-player-drawer-identity {
        display: grid;
        grid-template-columns: 64px minmax(0, 1fr);
        gap: 12px;
        align-items: center;
        padding-right: 28px;
      }

      .sc-player-drawer-identity > div:last-child {
        min-width: 0;
      }

      .sc-player-drawer-identity span {
        color: #f0b63d;
        font-size: 8px;
        font-weight: 950;
        text-transform: uppercase;
      }

      .sc-player-drawer-identity h2 {
        overflow: hidden;
        margin: 4px 0;
        color: #ffffff;
        font-size: 19px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-player-drawer-identity p {
        margin: 0;
        color: #6f8c9e;
        font-size: 9px;
      }

      .sc-player-drawer-actions {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 7px;
      }

      .sc-player-drawer-actions button {
        height: 34px;
        border: 1px solid rgba(0, 207, 218, 0.25);
        background: rgba(0, 207, 218, 0.06);
        color: #9bdde0;
        font-size: 9px;
        font-weight: 950;
      }

      .sc-player-drawer-actions button.is-active {
        border-color: rgba(230, 176, 58, 0.4);
        background: rgba(230, 176, 58, 0.1);
        color: #f0be4a;
      }

      .sc-player-drawer-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
      }

      .sc-player-drawer-metrics > div {
        display: grid;
        gap: 5px;
        padding: 8px;
        border: 1px solid rgba(105, 172, 202, 0.13);
        background: rgba(255, 255, 255, 0.015);
      }

      .sc-player-drawer-metrics span {
        color: #698798;
        font-size: 7px;
        font-weight: 900;
        text-transform: uppercase;
      }

      .sc-player-drawer-metrics strong {
        color: #eff9fc;
        font-size: 13px;
      }

      .sc-player-drawer-percentiles {
        min-height: 0;
        display: grid;
        align-content: start;
        gap: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(105, 172, 202, 0.15);
      }

      .sc-player-drawer-percentiles > header {
        display: flex;
        justify-content: space-between;
        color: #6f8d9e;
        font-size: 8px;
        font-weight: 900;
        text-transform: uppercase;
      }

      .sc-player-drawer-percentiles > div {
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr) 38px;
        gap: 8px;
        align-items: center;
      }

      .sc-player-drawer-percentiles > div > span {
        color: #91aab7;
        font-size: 8px;
        font-weight: 900;
      }

      .sc-player-drawer-percentiles > div > div {
        height: 5px;
        overflow: hidden;
        background: rgba(94, 149, 173, 0.14);
      }

      .sc-player-drawer-percentiles i {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #08cdd8, #e0ad3b);
      }

      .sc-player-drawer-percentiles > div > strong {
        color: #dfeef4;
        font-size: 8px;
        text-align: right;
      }

      .sc-league-leaders-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 10px;
        border: 1px solid rgba(105, 172, 202, 0.18);
        background: #081927;
      }

      .sc-league-leaders-header > div {
        display: grid;
        gap: 2px;
      }

      .sc-league-leaders-header > div > strong {
        color: #edf8fb;
        font-size: 14px;
      }

      .sc-league-leaders-header nav {
        display: flex;
        gap: 4px;
      }

      .sc-leaders-workspace {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        gap: 8px;
      }

      .sc-leader-category-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 7px 10px;
        border: 1px solid rgba(105, 172, 202, 0.17);
        background: #081927;
      }

      .sc-leader-category-header > div {
        display: grid;
        gap: 2px;
      }

      .sc-leader-category-header > div > strong {
        color: #f2f9fc;
        font-size: 14px;
      }

      .sc-leader-category-header nav {
        display: flex;
        gap: 3px;
      }

      .sc-leader-podium {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 7px;
      }

      .sc-leader-podium article {
        min-width: 0;
        display: grid;
        grid-template-columns: 26px 42px minmax(0, 1fr) auto;
        grid-template-rows: auto auto;
        align-items: center;
        gap: 2px 8px;
        padding: 8px 10px;
        border: 1px solid rgba(105, 172, 202, 0.17);
        background: #091a28;
      }

      .sc-leader-podium article > span {
        grid-row: 1 / span 2;
        color: #f0b63d;
        font-size: 14px;
        font-weight: 1000;
      }

      .sc-leader-podium .sc-avatar {
        grid-row: 1 / span 2;
      }

      .sc-leader-podium article > strong {
        overflow: hidden;
        color: #eef8fb;
        font-size: 10px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-leader-podium article > em {
        color: #688698;
        font-size: 7px;
        font-style: normal;
      }

      .sc-leader-podium article > b {
        grid-column: 4;
        grid-row: 1 / span 2;
        color: #08ddce;
        font-size: 15px;
      }

      .sc-leader-rank {
        color: #68889a;
        font-size: 9px;
        font-weight: 1000;
      }

      .sc-leader-primary {
        color: #09ddcf;
        font-size: 11px;
      }

      .sc-team-stats-workspace-v2 {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 1.65fr) minmax(290px, 0.75fr);
        grid-template-rows: auto minmax(0, 1fr);
        gap: 8px;
        overflow: hidden;
      }

      .sc-team-stats-workspace-v2 .sc-team-toolbar {
        grid-column: 1 / -1;
      }

      .sc-team-stats-workspace-v2 .sc-team-table-panel {
        min-height: 0;
        overflow: hidden;
      }

      .sc-team-stats-workspace-v2 .sc-team-selected-profile {
        min-height: 0;
        height: 100%;
        overflow-y: auto;
      }

      .sc-team-stats-workspace-v2 .sc-team-stats-table {
        min-width: 820px;
      }

      .sc-awards-board {
        height: 100%;
        min-height: 0;
        overflow: auto;
      }

      @media (max-width: 1320px) {
        .sc-player-header {
          grid-template-columns: 150px minmax(0, 1fr) 310px;
        }

        .sc-player-subnav button {
          padding: 0 5px;
          font-size: 8px;
        }

        .sc-overview-main-grid {
          grid-template-columns: minmax(0, 0.62fr) minmax(0, 0.38fr);
        }

        .sc-overview-featured-row {
          grid-template-columns: minmax(220px, 1.4fr) repeat(5, minmax(0, 1fr));
        }

        .sc-trend-content-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-trend-content-grid > :last-child {
          display: none;
        }

        .sc-overview-metric-grid {
          grid-template-columns: 1fr;
        }

        .sc-compare-main-grid {
          grid-template-columns: minmax(190px, 0.65fr) minmax(360px, 1.4fr) minmax(190px, 0.65fr);
        }
      }

      @media (max-width: 1050px) {
        .sc-command-bar {
          grid-template-columns: 82px minmax(0, 1fr);
        }

        .sc-command-context {
          display: none;
        }

        .sc-player-header {
          grid-template-columns: 130px minmax(0, 1fr);
        }

        .sc-player-header-actions {
          grid-column: 1 / -1;
          grid-template-columns: auto minmax(180px, 1fr);
        }

        .sc-player-filter-bar {
          overflow-x: auto;
        }

        .sc-overview-team-header {
          grid-template-columns: 1fr;
        }

        .sc-overview-main-grid {
          grid-template-columns: 1fr;
        }

        .sc-overview-featured-row {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .sc-overview-featured-row > :nth-child(n + 4) {
          display: none;
        }

        .sc-overview-side-stack {
          grid-template-rows: auto;
        }

        .sc-overview-metric-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-special-teams-page {
          grid-template-columns: 1fr;
        }

        .sc-special-teams-page > :last-child {
          display: none;
        }

        .sc-team-stats-workspace-v2 {
          grid-template-columns: 1fr;
        }

        .sc-team-stats-workspace-v2 .sc-team-selected-profile {
          display: none;
        }

        .sc-compare-main-grid {
          grid-template-columns: 1fr;
        }

        .sc-compare-card {
          display: none;
        }

        .sc-compare-selector-row {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-compare-pinned {
          display: none;
        }
      }
    `}</style>
  );
}

function StatsCentralStyles() {
  return (
    <style>{`
      .stats-central-screen {
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
        overflow: hidden;
        color: var(--text);
        background:
          radial-gradient(circle at 22% 0%, rgba(0, 216, 223, 0.13), transparent 28%),
          radial-gradient(circle at 88% 12%, rgba(232, 165, 54, 0.12), transparent 24%),
          linear-gradient(180deg, #06111b, #03080e 72%);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .stats-central-screen *,
      .stats-central-screen *::before,
      .stats-central-screen *::after {
        box-sizing: border-box;
      }

      .stats-central-screen button,
      .stats-central-screen input,
      .stats-central-screen select {
        font: inherit;
      }

      .stats-central-screen button {
        color: inherit;
      }

      .statscentral-shell {
        height: 100vh;
        min-height: 0;
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr);
        gap: 12px;
        padding: 16px 20px 12px;
        overflow: hidden;
      }

      .sc-topbar {
        min-height: 112px;
        display: grid;
        grid-template-columns: minmax(360px, 1.25fr) minmax(340px, 0.72fr) minmax(280px, 0.55fr);
        gap: 12px;
        min-width: 0;
      }

      .sc-title-card,
      .sc-control-card,
      .sc-status-card,
      .sc-section,
      .sc-stat-card,
      .sc-award-card,
      .sc-compare-card,
      .sc-team-profile-card {
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          var(--panel);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.07),
          0 18px 40px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(14px);
      }

      .sc-title-card {
        border-radius: 26px;
        padding: 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
      }

      .sc-back-button {
        width: 66px;
        height: 66px;
        flex: 0 0 auto;
        border-radius: 22px;
        border: 1px solid rgba(0, 216, 223, 0.28);
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.13), transparent 34%),
          linear-gradient(145deg, rgba(0, 216, 223, 0.13), rgba(255, 255, 255, 0.035)),
          rgba(255, 255, 255, 0.025);
        display: grid;
        place-items: center;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease,
          box-shadow 150ms ease;
      }

      .sc-back-button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.48);
        background:
          linear-gradient(145deg, rgba(0, 216, 223, 0.18), rgba(255, 255, 255, 0.04)),
          rgba(0, 216, 223, 0.06);
        box-shadow: 0 0 22px rgba(0, 216, 223, 0.08);
      }

      .sc-back-button span {
        display: block;
        color: var(--cyan);
        font-size: 1.1rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-back-button em {
        display: block;
        margin-top: -15px;
        color: var(--muted);
        font-size: 0.54rem;
        font-style: normal;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 950;
      }

      .sc-title-copy {
        min-width: 0;
      }

      .sc-title-copy p,
      .sc-section-head p {
        margin: 0 0 5px;
        color: var(--muted);
        font-size: 0.64rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 950;
      }

      .sc-title-copy h1 {
        margin: 0;
        font-size: clamp(1.9rem, 3.4vw, 3.55rem);
        line-height: 0.94;
        letter-spacing: -0.052em;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-title-copy span {
        display: block;
        margin-top: 8px;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.35;
        max-width: 780px;
      }

      .sc-control-card {
        border-radius: 26px;
        padding: 14px;
        min-width: 0;
        display: grid;
        grid-template-rows: minmax(0, 1fr) auto;
        gap: 10px;
      }

      .sc-search-box {
        min-width: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 7px;
      }

      .sc-search-box span {
        color: var(--muted);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 950;
      }

      .sc-search-box input {
        width: 100%;
        min-height: 44px;
        border-radius: 16px;
        outline: none;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.045);
        color: var(--text);
        padding: 0 13px;
        font-size: 0.78rem;
        font-weight: 800;
        transition:
          border-color 150ms ease,
          background 150ms ease,
          box-shadow 150ms ease;
      }

      .sc-search-box input::placeholder {
        color: rgba(139, 160, 175, 0.8);
      }

      .sc-search-box input:focus {
        border-color: rgba(0, 216, 223, 0.45);
        background: rgba(0, 216, 223, 0.045);
        box-shadow: 0 0 0 3px rgba(0, 216, 223, 0.08);
      }

      .sc-scope-toggle {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }

      .sc-scope-toggle button {
        min-height: 36px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.045);
        color: var(--muted);
        font-size: 0.66rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease,
          color 150ms ease;
      }

      .sc-scope-toggle button:hover {
        transform: translateY(-1px);
        color: var(--text);
        border-color: rgba(0, 216, 223, 0.3);
        background: rgba(0, 216, 223, 0.07);
      }

      .sc-scope-toggle button.is-active {
        color: #ffffff;
        border-color: rgba(232, 165, 54, 0.45);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.14), rgba(255, 255, 255, 0.035)),
          rgba(232, 165, 54, 0.08);
      }

      .sc-status-card {
        border-radius: 26px;
        padding: 12px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        min-width: 0;
      }

      .sc-status-card.is-compact-status {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 7px;
      }

      .sc-mini-status {
        min-width: 0;
        min-height: 42px;
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 8px;
        display: grid;
        grid-template-columns: 24px minmax(0, 1fr);
        align-items: center;
        gap: 8px;
      }

      .sc-mini-status-icon {
        width: 24px;
        height: 24px;
        border-radius: 9px;
        display: grid;
        place-items: center;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.18);
        font-size: 0.58rem;
        font-weight: 1000;
      }

      .sc-mini-status span {
        color: var(--muted);
        font-size: 0.56rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-mini-status strong {
        color: #fff;
        font-size: 0.92rem;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-mini-status em {
        display: block;
        margin-top: 1px;
        color: var(--muted-2);
        font-size: 0.55rem;
        font-style: normal;
      }

      .sc-menu {
        min-height: 50px;
        display: grid;
        grid-template-columns: repeat(11, minmax(0, 1fr));
        gap: 8px;
        min-width: 0;
      }

      .sc-menu button {
        min-width: 0;
        min-height: 48px;
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.085);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
          rgba(7, 22, 35, 0.78);
        color: var(--muted);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding: 0 8px;
        overflow: hidden;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          color 150ms ease,
          background 150ms ease,
          box-shadow 150ms ease;
      }

      .sc-menu button:hover {
        transform: translateY(-1px);
        color: var(--text);
        border-color: rgba(0, 216, 223, 0.3);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.08), rgba(255, 255, 255, 0.02)),
          rgba(7, 22, 35, 0.86);
      }

      .sc-menu button.is-active {
        color: #ffffff;
        border-color: rgba(0, 216, 223, 0.42);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.15), rgba(255, 255, 255, 0.035)),
          rgba(0, 216, 223, 0.07);
        box-shadow:
          inset 0 0 0 1px rgba(0, 216, 223, 0.11),
          0 12px 28px rgba(0, 216, 223, 0.07);
      }

      .sc-menu-icon {
        font-size: 0.68rem;
        line-height: 1;
        color: var(--cyan);
        opacity: 0.92;
        width: 13px;
        text-align: center;
        flex: 0 0 auto;
      }

      .sc-menu em {
        min-width: 0;
        font-size: 0.58rem;
        line-height: 1;
        font-style: normal;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-content {
        min-height: 0;
        min-width: 0;
        overflow: hidden;
      }

      .sc-tab-page {
        height: 100%;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-rows: minmax(0, 1fr) minmax(190px, 0.36fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-tab-page > .sc-section:only-child {
        min-height: 0;
      }

      .sc-section {
        min-height: 0;
        min-width: 0;
        border-radius: 24px;
        padding: 14px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }

      .sc-section-head {
        flex: 0 0 auto;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
      }

      .sc-section-head h2 {
        margin: 0;
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.08;
        letter-spacing: -0.015em;
      }

      .sc-section-right,
      .sc-section-tools {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }

      .sc-section-body {
        flex: 1;
        min-height: 0;
        min-width: 0;
        overflow: hidden;
      }

      .sc-overview {
        height: 100%;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(340px, 0.75fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-redesign {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(0, 1fr) minmax(155px, 0.38fr);
      }

      .sc-overview-main {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) minmax(0, 1fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-side {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-rows: minmax(0, 0.75fr) minmax(0, 1.25fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-hero {
        min-width: 0;
        min-height: 0;
        display: grid;
        grid-template-rows: auto auto;
        gap: 12px;
      }

      .sc-franchise-snapshot {
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          radial-gradient(circle at 12% 0%, rgba(0, 216, 223, 0.12), transparent 34%),
          linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.018)),
          rgba(10, 28, 42, 0.94);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.07),
          0 18px 40px rgba(0, 0, 0, 0.22);
        padding: 16px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 150px;
        align-items: center;
        gap: 14px;
      }

      .sc-franchise-main p {
        margin: 0 0 5px;
        color: var(--muted);
        font-size: 0.6rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 950;
      }

      .sc-franchise-main h2 {
        margin: 0;
        color: #fff;
        font-size: clamp(1.35rem, 2.2vw, 2.05rem);
        line-height: 1;
        letter-spacing: -0.04em;
      }

      .sc-franchise-main span {
        display: block;
        margin-top: 7px;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.35;
      }

      .sc-franchise-record {
        min-height: 94px;
        border-radius: 22px;
        border: 1px solid rgba(232, 165, 54, 0.24);
        background: rgba(232, 165, 54, 0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }

      .sc-franchise-record strong {
        color: var(--gold);
        font-size: 2.25rem;
        line-height: 0.9;
        font-weight: 1000;
      }

      .sc-franchise-record span {
        margin-top: 6px;
        color: #fff;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 950;
      }

      .sc-franchise-record em {
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.62rem;
        font-style: normal;
      }

      .sc-stat-grid {
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
      }

      .sc-stat-grid-clean {
        grid-template-columns: repeat(5, minmax(0, 1fr));
      }

      .sc-stat-card {
        min-width: 0;
        min-height: 102px;
        border-radius: 21px;
        padding: 12px 13px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 4px;
        overflow: hidden;
      }

      .sc-stat-card span {
        color: var(--muted);
        font-size: 0.58rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-stat-card strong {
        color: #ffffff;
        font-size: clamp(1.05rem, 1.5vw, 1.65rem);
        line-height: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-stat-card b {
        color: var(--cyan);
        font-size: 0.62rem;
        line-height: 1.15;
        font-weight: 950;
        letter-spacing: 0.04em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-stat-card em {
        color: var(--muted-2);
        font-style: normal;
        font-size: 0.69rem;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-stat-card i {
        margin-top: auto;
        color: var(--gold);
        font-size: 0.58rem;
        line-height: 1.15;
        font-style: normal;
        font-weight: 850;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-stat-card.has-warning {
        border-color: rgba(232, 165, 54, 0.2);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.055), rgba(255, 255, 255, 0.018)),
          var(--panel);
      }

      .sc-stat-card.is-good strong {
        color: var(--green);
      }

      .sc-stat-card.is-bad strong {
        color: var(--red);
      }

      .sc-stat-card.is-warn strong,
      .sc-stat-card.is-gold strong {
        color: var(--gold);
      }

      .sc-overview-feature-grid {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(0, 0.9fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-feature-section {
        min-height: 0;
      }

      .sc-driver-grid {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        overflow: auto;
        padding-right: 2px;
        align-content: start;
      }

      .sc-driver-grid .sc-mini-player {
        min-height: 76px;
      }

      .sc-crease-context-card {
        min-height: 76px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 11px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .sc-crease-context-card strong {
        color: #fff;
        font-size: 0.8rem;
        line-height: 1.1;
      }

      .sc-crease-context-card span {
        margin-top: 5px;
        color: var(--gold);
        font-size: 1rem;
        font-weight: 1000;
      }

      .sc-crease-context-card em {
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.64rem;
        font-style: normal;
      }

      .sc-overview-bottom-strip {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(280px, 0.72fr) minmax(0, 1.28fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-mini-side-section {
        padding: 12px;
        border-radius: 22px;
      }

      .sc-mini-side-section .sc-section-head {
        margin-bottom: 8px;
      }

      .sc-mini-side-section .sc-section-head h2 {
        font-size: 0.9rem;
      }

      .sc-pill {
        display: inline-flex;
        min-height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0 10px;
        color: var(--cyan);
        background: rgba(0, 216, 223, 0.08);
        border: 1px solid rgba(0, 216, 223, 0.16);
        font-size: 0.62rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
      }

      .sc-pill.is-good,
      .sc-pill.is-elite {
        color: var(--green);
        border-color: rgba(72, 216, 139, 0.28);
        background: rgba(72, 216, 139, 0.08);
      }

      .sc-pill.is-gold,
      .sc-pill.is-warn {
        color: var(--gold);
        border-color: rgba(232, 165, 54, 0.32);
        background: rgba(232, 165, 54, 0.1);
      }

      .sc-pill.is-bad {
        color: var(--red);
        border-color: rgba(255, 100, 100, 0.32);
        background: rgba(255, 100, 100, 0.08);
      }

      .sc-mini-stack {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 9px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-mini-player {
        min-height: 66px;
        display: grid;
        grid-template-columns: 38px minmax(0, 1fr) auto;
        align-items: center;
        gap: 10px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 9px 10px;
        min-width: 0;
      }

      .sc-mini-player.is-empty {
        opacity: 0.8;
      }

      .sc-avatar {
        width: 42px;
        height: 42px;
        flex: 0 0 auto;
        border-radius: 16px;
        display: grid;
        place-items: center;
        font-size: 0.72rem;
        font-weight: 1000;
        color: #ffffff;
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.26), transparent 32%),
          linear-gradient(145deg, rgba(0, 216, 223, 0.38), rgba(10, 28, 42, 0.96));
        border: 1px solid rgba(0, 216, 223, 0.34);
        box-shadow: 0 0 22px rgba(0, 216, 223, 0.13);
      }

      .sc-avatar.is-small {
        width: 34px;
        height: 34px;
        border-radius: 13px;
        font-size: 0.62rem;
      }

      .sc-avatar--logo {
        padding: 4px;
        background: rgba(8, 18, 30, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 16px rgba(0, 0, 0, 0.28);
      }

      .sc-avatar--logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
      }

      .sc-avatar--fallback {
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.26), transparent 32%),
          linear-gradient(145deg, rgba(0, 216, 223, 0.38), rgba(10, 28, 42, 0.96));
        border: 1px solid rgba(0, 216, 223, 0.34);
      }

      .sc-mini-player strong,
      .sc-name-cell strong {
        display: block;
        color: #ffffff;
        font-size: 0.8rem;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-mini-player span,
      .sc-name-cell span:not(.sc-player-meta-inline):not(.sc-player-ovr-drop) {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.66rem;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-mini-player b {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        color: var(--gold);
        font-size: 0.95rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-mini-player b small {
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.52rem;
        letter-spacing: 0.1em;
      }

      .sc-calendar-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 7px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-calendar-list-compact {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 7px;
        overflow: auto;
      }

      .sc-calendar-list button {
        min-height: 46px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.035);
        color: var(--text);
        padding: 8px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease;
      }

      .sc-calendar-list-compact button {
        min-height: 42px;
        border-radius: 14px;
      }

      .sc-calendar-list button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.25);
        background: rgba(0, 216, 223, 0.055);
      }

      .sc-calendar-list button.is-active {
        border-color: rgba(232, 165, 54, 0.4);
        background: rgba(232, 165, 54, 0.09);
      }

      .sc-calendar-list button span {
        display: flex;
        min-width: 0;
        flex-direction: column;
        align-items: flex-start;
        gap: 2px;
        color: #ffffff;
        font-size: 0.74rem;
        font-weight: 900;
      }

      .sc-calendar-list-compact button span {
        font-size: 0.66rem;
      }

      .sc-calendar-list button small {
        color: var(--muted);
        font-size: 0.56rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .sc-calendar-list button em {
        min-width: 26px;
        min-height: 26px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: rgba(0, 216, 223, 0.08);
        color: var(--cyan);
        border: 1px solid rgba(0, 216, 223, 0.13);
        font-style: normal;
        font-size: 0.68rem;
        font-weight: 950;
      }

      .sc-calendar-list-compact button em {
        min-width: 22px;
        min-height: 22px;
        font-size: 0.58rem;
      }

      .sc-score-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 9px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-mini-side-section .sc-score-list {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-score-card {
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 11px;
      }

      .sc-mini-side-section .sc-score-card {
        min-height: 74px;
      }

      .sc-score-line {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto auto minmax(0, 1fr);
        align-items: center;
        gap: 8px;
      }

      .sc-score-team {
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 900;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-score-team.is-away {
        text-align: right;
      }

      .sc-score-line strong {
        color: var(--gold);
        font-size: 0.92rem;
        font-weight: 1000;
      }

      .sc-score-line em {
        color: var(--muted);
        font-style: normal;
        font-size: 0.66rem;
      }

      .sc-score-meta,
      .sc-score-micro {
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.64rem;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .sc-score-micro span {
        color: var(--muted-2);
      }

      .sc-empty {
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        color: var(--muted);
        padding: 14px;
        font-size: 0.78rem;
        line-height: 1.4;
      }

      .sc-table-wrap {
        height: 100%;
        min-height: 0;
        min-width: 0;
        overflow: auto;
        border-radius: 19px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(4, 14, 23, 0.42);
      }

      .sc-table {
        width: 100%;
        min-width: 1060px;
        border-collapse: collapse;
      }

      .sc-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: rgba(5, 18, 29, 0.98);
        color: var(--muted);
        font-size: 0.6rem;
        line-height: 1;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        text-align: left;
        padding: 11px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        white-space: nowrap;
      }

      .sc-table th[role="button"] {
        cursor: pointer;
      }

      .sc-table th[role="button"]:hover {
        color: var(--cyan);
      }

      .sc-table td {
        color: var(--text);
        font-size: 0.74rem;
        line-height: 1.15;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        white-space: nowrap;
      }

      .sc-table tr:hover td {
        background: rgba(0, 216, 223, 0.035);
      }

      .sc-table .is-right {
        text-align: right;
      }

      .sc-table .is-empty {
        text-align: center;
        color: var(--muted);
        padding: 32px;
      }

      .sc-name-cell {
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        align-items: center;
        gap: 9px;
        min-width: 240px;
      }

      .sc-team-chip {
        width: 34px;
        height: 34px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        font-size: 0.56rem;
        font-weight: 1000;
        color: #fff;
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.26), transparent 32%),
          linear-gradient(145deg, rgba(232, 165, 54, 0.38), rgba(10, 28, 42, 0.96));
        border: 1px solid rgba(232, 165, 54, 0.34);
      }

      .sc-team-logo-mark {
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 13px;
        overflow: hidden;
      }

      .sc-team-logo-mark.is-small {
        width: 34px;
        height: 34px;
      }

      .sc-team-logo-mark.is-large {
        width: 74px;
        height: 74px;
        border-radius: 24px;
      }

      .sc-team-logo-mark--logo {
        padding: 4px;
        background: rgba(8, 18, 30, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 16px rgba(0, 0, 0, 0.28);
      }

      .sc-team-logo-mark.is-large.sc-team-logo-mark--logo {
        padding: 8px;
        box-shadow: 0 0 24px rgba(0, 0, 0, 0.28);
      }

      .sc-team-logo-mark--logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
      }

      .sc-team-logo-mark--fallback {
        font-size: 0.56rem;
        font-weight: 1000;
        color: #fff;
        background:
          radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.26), transparent 32%),
          linear-gradient(145deg, rgba(232, 165, 54, 0.38), rgba(10, 28, 42, 0.96));
        border: 1px solid rgba(232, 165, 54, 0.34);
      }

      .sc-team-logo-mark.is-large.sc-team-logo-mark--fallback {
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        box-shadow: 0 0 24px rgba(232, 165, 54, 0.13);
      }

      .sc-impact-text {
        font-weight: 1000;
      }

      .sc-impact-text.is-elite,
      .sc-impact-text.is-good {
        color: var(--green);
      }

      .sc-impact-text.is-warn {
        color: var(--gold);
      }

      .sc-impact-text.is-bad {
        color: var(--red);
      }

      .sc-bottom-grid {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        overflow: hidden;
      }

      .sc-bottom-grid.is-logs {
        height: 100%;
        grid-template-rows: minmax(0, 1fr);
      }

      .sc-tier-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 8px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-tier-row {
        min-height: 54px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 9px 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }

      .sc-tier-row strong {
        display: block;
        color: #fff;
        font-size: 0.78rem;
      }

      .sc-tier-row span {
        display: block;
        margin-top: 2px;
        color: var(--muted);
        font-size: 0.66rem;
      }

      .sc-tier-row b {
        color: var(--cyan);
        font-size: 1.1rem;
        font-weight: 1000;
      }

      .sc-tier-row.is-elite b,
      .sc-tier-row.is-good b {
        color: var(--green);
      }

      .sc-tier-row.is-warn b {
        color: var(--gold);
      }

      .sc-tier-row.is-bad b {
        color: var(--red);
      }

      .sc-danger-grid {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 9px;
        overflow: auto;
        align-content: start;
      }

      .sc-danger-card,
      .sc-list-card,
      .sc-note-card,
      .sc-log-card {
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px 11px;
      }

      .sc-danger-card strong,
      .sc-list-card strong,
      .sc-note-card strong,
      .sc-log-card strong {
        display: block;
        color: #fff;
        font-size: 0.8rem;
        line-height: 1.15;
      }

      .sc-danger-card span,
      .sc-list-card span,
      .sc-note-card span,
      .sc-log-card span {
        display: block;
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.35;
      }

      .sc-list-card em,
      .sc-log-card em {
        display: block;
        margin-top: 7px;
        color: var(--gold);
        font-style: normal;
        font-size: 0.66rem;
        font-weight: 900;
      }

      .sc-list-card em.is-good {
        color: var(--green);
      }

      .sc-list-card em.is-bad {
        color: var(--red);
      }

      .sc-list-card.is-good {
        border-color: rgba(72, 216, 139, 0.22);
        background: rgba(72, 216, 139, 0.055);
      }

      .sc-list-card.is-warn {
        border-color: rgba(232, 165, 54, 0.24);
        background: rgba(232, 165, 54, 0.06);
      }

      .sc-list-card.is-bad {
        border-color: rgba(255, 100, 100, 0.24);
        background: rgba(255, 100, 100, 0.055);
      }

      .sc-list-cards,
      .sc-note-stack,
      .sc-log-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 9px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-team-profile-card {
        height: 100%;
        border-radius: 24px;
        padding: 15px;
        display: grid;
        grid-template-columns: 74px minmax(0, 1fr);
        grid-template-rows: auto minmax(0, 1fr);
        gap: 13px;
        overflow: hidden;
      }

      .sc-team-profile-card h3 {
        margin: 0;
        color: #fff;
        font-size: 1.15rem;
        line-height: 1.08;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-team-profile-card p {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 0.75rem;
      }

      .sc-team-profile-grid {
        grid-column: 1 / -1;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-team-profile-grid span {
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 9px 10px;
        color: var(--muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 950;
      }

      .sc-team-profile-grid strong {
        display: block;
        margin-top: 4px;
        color: #fff;
        font-size: 0.84rem;
        letter-spacing: 0;
      }

      .sc-leaders-grid {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        grid-auto-rows: minmax(0, 1fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-award-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-award-card {
        min-height: 124px;
        border-radius: 22px;
        padding: 13px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 8px;
      }

      .sc-award-card span {
        color: var(--gold);
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 1000;
      }

      .sc-award-card strong {
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.12;
      }

      .sc-award-card em {
        color: var(--muted);
        font-size: 0.66rem;
        line-height: 1.3;
        font-style: normal;
      }

      .sc-award-card b {
        color: var(--cyan);
        font-size: 1.1rem;
      }

      .sc-award-card.is-subjective em {
        color: rgba(232, 220, 190, 0.88);
      }

      .sc-award-subjective-tag {
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(232, 220, 190, 0.72) !important;
      }

      .sc-awards-page {
        grid-template-rows: auto minmax(0, 1fr) minmax(220px, 0.42fr);
      }

      .sc-awards-modebar {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        min-height: 62px;
      }

      .sc-awards-modebar button {
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015)),
          rgba(7, 22, 35, 0.78);
        color: var(--text);
        cursor: pointer;
        padding: 12px 16px;
        text-align: left;
        transition: 150ms ease;
      }

      .sc-awards-modebar button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.28);
      }

      .sc-awards-modebar button.is-active {
        border-color: rgba(232, 165, 54, 0.42);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.16), rgba(255, 255, 255, 0.025)),
          rgba(232, 165, 54, 0.07);
      }

      .sc-awards-modebar strong {
        display: block;
        color: #fff;
        font-size: 0.9rem;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .sc-awards-modebar span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.68rem;
      }

      .sc-awards-bottom {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(360px, 0.85fr) minmax(0, 1.15fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-award-name-cell {
        min-width: 320px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .sc-award-name-cell span {
        color: var(--muted);
        font-size: 0.65rem;
        line-height: 1.25;
        white-space: normal;
      }

      .sc-pill.is-custom {
        color: var(--cyan);
        border-color: rgba(0, 216, 223, 0.3);
        background: rgba(0, 216, 223, 0.08);
      }

      .sc-award-definition-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 9px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-award-definition-card {
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 11px 12px;
      }

      .sc-award-definition-card strong {
        display: block;
        color: #fff;
        font-size: 0.78rem;
        line-height: 1.15;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .sc-award-definition-card span {
        display: block;
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1.35;
      }

      /* =========================================================
         AWARDS BOARD — FIXED CLEAN VERSION
      ========================================================= */

      .sc-awards-board {
        height: 100%;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-rows: 44px 58px minmax(0, 1fr) 150px;
        gap: 10px;
        overflow: hidden;
      }

      .sc-awards-subtabs {
        min-width: 0;
        min-height: 0;
        display: flex;
        gap: 8px;
      }

      .sc-awards-subtabs button {
        flex: 1;
        min-width: 0;
        min-height: 44px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.085);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.042), rgba(255, 255, 255, 0.014)),
          rgba(7, 22, 35, 0.78);
        color: var(--text);
        cursor: pointer;
        padding: 7px 12px;
        text-align: left;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease;
      }

      .sc-awards-subtabs button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.26);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.075), rgba(255, 255, 255, 0.016)),
          rgba(7, 22, 35, 0.86);
      }

      .sc-awards-subtabs button.is-active {
        border-color: rgba(232, 165, 54, 0.42);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.145), rgba(255, 255, 255, 0.022)),
          rgba(232, 165, 54, 0.065);
      }

      .sc-awards-subtabs strong {
        display: block;
        color: #fff;
        font-size: 0.72rem;
        line-height: 1;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .sc-awards-subtabs span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.58rem;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-summary-strip {
        min-width: 0;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-awards-summary-strip div {
        min-width: 0;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.012)),
          rgba(10, 28, 42, 0.84);
        padding: 9px 11px;
        overflow: hidden;
      }

      .sc-awards-summary-strip span {
        display: block;
        color: var(--muted);
        font-size: 0.56rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-summary-strip strong {
        display: block;
        margin-top: 6px;
        color: #fff;
        font-size: 1.1rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-awards-main-grid {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr);
        gap: 10px;
        overflow: hidden;
      }

      .sc-awards-main-grid > .sc-section {
        height: 100%;
      }

      .sc-awards-rules-row {
        min-height: 0;
        min-width: 0;
        overflow: hidden;
      }

      .sc-awards-rules-row > .sc-section {
        height: 100%;
        padding: 11px;
      }

      .sc-awards-rules-row .sc-section-head {
        margin-bottom: 8px;
      }

      .sc-awards-rules-row .sc-section-head h2 {
        font-size: 0.9rem;
      }

      .sc-awards-rules-row .sc-section-body {
        overflow: hidden;
      }

      .sc-awards-award-cell {
        min-width: 150px;
        max-width: 210px;
      }

      .sc-awards-award-cell strong {
        display: block;
        color: var(--gold);
        font-size: 0.69rem;
        line-height: 1.1;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-award-cell span {
        display: block;
        margin-top: 3px;
        color: var(--muted);
        font-size: 0.58rem;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-rank {
        color: var(--muted);
        font-weight: 950;
      }

      .sc-awards-rank.is-leader {
        color: var(--gold);
      }

      .sc-awards-primary-metric {
        color: var(--cyan);
        font-size: 0.82rem;
        font-weight: 1000;
      }

      .sc-awards-profile-line {
        display: block;
        max-width: 340px;
        color: var(--muted);
        font-size: 0.66rem;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-leader-list {
        height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 8px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-awards-leader-card {
        min-height: 66px;
        border-radius: 17px;
        border: 1px solid rgba(232, 165, 54, 0.16);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.075), rgba(255, 255, 255, 0.014)),
          rgba(255, 255, 255, 0.028);
        padding: 10px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 10px;
      }

      .sc-awards-leader-card.is-custom {
        border-color: rgba(0, 216, 223, 0.18);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.075), rgba(255, 255, 255, 0.014)),
          rgba(255, 255, 255, 0.028);
      }

      .sc-awards-leader-card span {
        display: block;
        color: var(--gold);
        font-size: 0.56rem;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-leader-card.is-custom span {
        color: var(--cyan);
      }

      .sc-awards-leader-card strong {
        display: block;
        margin-top: 5px;
        color: #fff;
        font-size: 0.8rem;
        line-height: 1.12;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-leader-card em {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.61rem;
        line-height: 1.18;
        font-style: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-leader-card b {
        color: var(--cyan);
        font-size: 1rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-awards-rule-grid {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
        overflow: auto;
        padding-right: 2px;
        align-content: start;
      }

      .sc-awards-rule-card {
        min-width: 0;
        min-height: 72px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.032);
        padding: 9px 10px;
        overflow: hidden;
      }

      .sc-awards-rule-card strong {
        display: block;
        color: #fff;
        font-size: 0.67rem;
        line-height: 1.1;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-awards-rule-card span {
        display: block;
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.59rem;
        line-height: 1.28;
      }

      .sc-awards-board .sc-table {
        min-width: 980px;
      }

      .sc-awards-board .sc-table td {
        padding-top: 8px;
        padding-bottom: 8px;
      }

      .sc-awards-board .sc-table th {
        padding-top: 10px;
        padding-bottom: 10px;
      }

      .sc-awards-board .sc-name-cell {
        min-width: 210px;
      }

      @media (max-width: 1250px) {
        .sc-awards-main-grid {
          grid-template-columns: 1fr;
          grid-template-rows: minmax(0, 1fr) 190px;
        }

        .sc-awards-rule-grid {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }
      }

      .sc-trend-grid {
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }

      .sc-compare-selectors {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 12px;
      }

      .sc-compare-selectors select {
        width: 100%;
        min-width: 0;
        min-height: 42px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(7, 22, 35, 0.9);
        color: var(--text);
        outline: none;
        padding: 0 12px;
        font-size: 0.76rem;
        font-weight: 800;
      }

      .sc-compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 12px;
      }

      .sc-compare-card {
        border-radius: 23px;
        padding: 13px;
        min-width: 0;
      }

      .sc-compare-card-top {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
      }

      .sc-compare-card strong {
        display: block;
        color: #fff;
        font-size: 0.92rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-compare-card span {
        color: var(--muted);
        font-size: 0.68rem;
      }

      .sc-compare-card-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-compare-card-grid span {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(255, 255, 255, 0.035);
        padding: 8px 9px;
        color: var(--muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 950;
      }

      .sc-compare-card-grid b {
        display: block;
        margin-top: 4px;
        color: #fff;
        font-size: 0.84rem;
        letter-spacing: 0;
      }

      .sc-compare-table {
        border-radius: 19px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background: rgba(4, 14, 23, 0.42);
        overflow: auto;
      }

      .sc-compare-row {
        min-height: 42px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(90px, 0.55fr) minmax(0, 1fr);
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding: 8px 12px;
      }

      .sc-compare-row span {
        color: var(--text);
        font-size: 0.78rem;
      }

      .sc-compare-row span:first-child {
        text-align: left;
      }

      .sc-compare-row span:last-child {
        text-align: right;
      }

      .sc-compare-row strong {
        text-align: center;
        color: var(--muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .sc-compare-row .is-good {
        color: var(--green);
        font-weight: 1000;
      }

      .sc-compare-row .is-bad {
        color: var(--red);
      }

      .sc-log-card strong {
        color: #fff;
      }

      .sc-log-card span {
        color: var(--muted);
      }

      .sc-log-card em {
        color: var(--gold);
      }

      .sc-formula-sections {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        overflow: auto;
        padding-right: 2px;
        align-content: start;
      }

      .sc-formula-section {
        min-width: 0;
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 13px;
      }

      .sc-formula-section header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
      }

      .sc-formula-section h3 {
        margin: 0;
        color: #fff;
        font-size: 0.92rem;
        line-height: 1.1;
      }

      .sc-formula-section header span {
        color: var(--muted);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-weight: 950;
      }

      .sc-formula-section.is-green {
        border-color: rgba(72, 216, 139, 0.16);
      }

      .sc-formula-section.is-blue {
        border-color: rgba(98, 183, 255, 0.18);
      }

      .sc-formula-section.is-gold {
        border-color: rgba(232, 165, 54, 0.2);
      }

      .sc-formula-section.is-purple {
        border-color: rgba(177, 140, 255, 0.2);
      }

      .sc-formula-section.is-red {
        border-color: rgba(255, 100, 100, 0.18);
      }

      .sc-formula-section.is-cyan {
        border-color: rgba(0, 216, 223, 0.2);
      }

      .sc-formula-section.is-orange {
        border-color: rgba(255, 159, 67, 0.2);
      }

      .sc-formula-list {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-formula-card {
        min-width: 0;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        background: rgba(4, 14, 23, 0.45);
        padding: 9px 10px;
      }

      .sc-formula-card strong {
        display: block;
        color: #fff;
        font-size: 0.74rem;
        line-height: 1.1;
      }

      .sc-formula-card span {
        display: block;
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.64rem;
        line-height: 1.28;
      }

      .sc-formula-card em {
        display: block;
        margin-top: 7px;
        color: var(--cyan);
        font-size: 0.62rem;
        line-height: 1.3;
        font-style: normal;
        font-weight: 850;
      }
                .sc-overview-fixed {
        height: 100%;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-rows: auto auto minmax(0, 1fr) minmax(132px, 0.34fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-fixed-hero {
        min-width: 0;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          radial-gradient(circle at 12% 0%, rgba(0, 216, 223, 0.14), transparent 34%),
          radial-gradient(circle at 88% 20%, rgba(232, 165, 54, 0.12), transparent 26%),
          linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.018)),
          rgba(10, 28, 42, 0.94);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.07),
          0 18px 40px rgba(0, 0, 0, 0.22);
        padding: 16px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 142px;
        align-items: center;
        gap: 14px;
        overflow: hidden;
      }

      .sc-overview-fixed-title {
        min-width: 0;
      }

      .sc-overview-fixed-title p,
      .sc-overview-fixed-panel header p {
        margin: 0 0 5px;
        color: var(--muted);
        font-size: 0.6rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 950;
      }

      .sc-overview-fixed-title h2 {
        margin: 0;
        color: #fff;
        font-size: clamp(1.45rem, 2.4vw, 2.1rem);
        line-height: 1;
        letter-spacing: -0.045em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-title span {
        display: block;
        margin-top: 7px;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.35;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-points {
        min-height: 88px;
        border-radius: 22px;
        border: 1px solid rgba(232, 165, 54, 0.26);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.14), rgba(255, 255, 255, 0.025)),
          rgba(232, 165, 54, 0.06);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }

      .sc-overview-fixed-points strong {
        color: var(--gold);
        font-size: 2.1rem;
        line-height: 0.9;
        font-weight: 1000;
      }

      .sc-overview-fixed-points span {
        margin-top: 6px;
        color: #fff;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 950;
      }

      .sc-overview-fixed-points em {
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.58rem;
        font-style: normal;
      }

      .sc-overview-fixed-points.is-good strong {
        color: var(--green);
      }

      .sc-overview-fixed-points.is-bad strong {
        color: var(--red);
      }

      .sc-overview-fixed-story {
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 10px;
      }

      .sc-overview-fixed-card {
        min-width: 0;
        min-height: 92px;
        border-radius: 20px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.052), rgba(255, 255, 255, 0.016)),
          rgba(10, 28, 42, 0.88);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.055),
          0 12px 26px rgba(0, 0, 0, 0.16);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
        overflow: hidden;
      }

      .sc-overview-fixed-card span {
        color: var(--muted);
        font-size: 0.54rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-card strong {
        color: #fff;
        font-size: 1.32rem;
        line-height: 1;
        font-weight: 1000;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-card em {
        color: var(--muted-2);
        font-size: 0.66rem;
        line-height: 1.15;
        font-style: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-card b {
        color: var(--cyan);
        font-size: 0.58rem;
        line-height: 1.15;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-card.is-good strong {
        color: var(--green);
      }

      .sc-overview-fixed-card.is-bad strong {
        color: var(--red);
      }

      .sc-overview-fixed-card.has-warning {
        border-color: rgba(232, 165, 54, 0.22);
        background:
          linear-gradient(180deg, rgba(232, 165, 54, 0.075), rgba(255, 255, 255, 0.016)),
          rgba(10, 28, 42, 0.88);
      }

      .sc-overview-fixed-card.has-warning b {
        color: var(--gold);
      }

      .sc-overview-fixed-main {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(0, 0.85fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-fixed-bottom {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: minmax(260px, 0.65fr) minmax(300px, 0.8fr) minmax(0, 1.35fr);
        gap: 12px;
        overflow: hidden;
      }

      .sc-overview-fixed-panel {
        min-height: 0;
        min-width: 0;
        border-radius: 22px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.048), rgba(255, 255, 255, 0.015)),
          var(--panel);
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.06),
          0 14px 30px rgba(0, 0, 0, 0.18);
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      .sc-overview-fixed-panel header {
        flex: 0 0 auto;
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 9px;
      }

      .sc-overview-fixed-panel header h3 {
        margin: 0;
        color: #fff;
        font-size: 0.95rem;
        line-height: 1.1;
        letter-spacing: -0.01em;
      }

      .sc-overview-fixed-driver-grid {
        flex: 1;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 9px;
        overflow: auto;
        padding-right: 2px;
        align-content: start;
      }

      .sc-overview-fixed-driver-grid .sc-mini-player {
        min-height: 72px;
      }

      .sc-overview-fixed-mini-metric {
        min-height: 72px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
      }

      .sc-overview-fixed-mini-metric span,
      .sc-overview-fixed-special-grid span {
        color: var(--muted);
        font-size: 0.57rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-mini-metric strong,
      .sc-overview-fixed-special-grid strong {
        margin-top: 5px;
        color: #fff;
        font-size: 1.12rem;
        line-height: 1;
        font-weight: 1000;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-mini-metric em,
      .sc-overview-fixed-special-grid em {
        margin-top: 5px;
        color: var(--gold);
        font-size: 0.59rem;
        line-height: 1.15;
        font-style: normal;
        font-weight: 850;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-special-grid {
        flex: 1;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-overview-fixed-special-grid div {
        min-width: 0;
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.035);
        padding: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
      }

      .sc-overview-fixed-calendar {
        flex: 1;
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 7px;
        overflow: auto;
        padding-right: 2px;
        align-content: start;
      }

      .sc-overview-fixed-calendar button {
        min-height: 38px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.035);
        color: var(--text);
        padding: 7px 9px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        transition:
          transform 150ms ease,
          border-color 150ms ease,
          background 150ms ease;
      }

      .sc-overview-fixed-calendar button:hover {
        transform: translateY(-1px);
        border-color: rgba(0, 216, 223, 0.25);
        background: rgba(0, 216, 223, 0.055);
      }

      .sc-overview-fixed-calendar button.is-active {
        border-color: rgba(232, 165, 54, 0.4);
        background: rgba(232, 165, 54, 0.09);
      }

      .sc-overview-fixed-calendar button span {
        display: flex;
        min-width: 0;
        flex-direction: column;
        align-items: flex-start;
        gap: 2px;
        color: #fff;
        font-size: 0.65rem;
        font-weight: 900;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-overview-fixed-calendar button small {
        color: var(--muted);
        font-size: 0.52rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
      }

      .sc-overview-fixed-calendar button em {
        min-width: 22px;
        min-height: 22px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: rgba(0, 216, 223, 0.08);
        color: var(--cyan);
        border: 1px solid rgba(0, 216, 223, 0.13);
        font-style: normal;
        font-size: 0.58rem;
        font-weight: 950;
      }

      .sc-overview-fixed-panel.is-scores .sc-score-list {
        flex: 1;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        overflow: auto;
        padding-right: 2px;
      }

      .sc-overview-fixed-panel.is-scores .sc-empty {
        grid-column: 1 / -1;
      }

      .sc-overview-fixed-driver-grid::-webkit-scrollbar,
      .sc-overview-fixed-calendar::-webkit-scrollbar,
      .sc-overview-fixed-panel.is-scores .sc-score-list::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .sc-overview-fixed-driver-grid::-webkit-scrollbar-thumb,
      .sc-overview-fixed-calendar::-webkit-scrollbar-thumb,
      .sc-overview-fixed-panel.is-scores .sc-score-list::-webkit-scrollbar-thumb {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
      }

      .sc-overview-fixed-driver-grid::-webkit-scrollbar-track,
      .sc-overview-fixed-calendar::-webkit-scrollbar-track,
      .sc-overview-fixed-panel.is-scores .sc-score-list::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.035);
        border-radius: 999px;
      }

      /* =========================================================
         SKATERS TAB — FIXED FUNCTIONAL BOARD
      ========================================================= */

      .sc-skaters-page {
        grid-template-rows: minmax(0, 1fr) minmax(160px, 0.26fr);
      }

      .sc-skaters-table {
        min-width: 1850px;
      }

      .sc-skaters-table th.is-player-col,
      .sc-skaters-table td.is-player-col {
        position: sticky;
        left: 0;
        z-index: 3;
        background: rgba(5, 18, 29, 0.98);
      }

      .sc-skaters-table td.is-player-col {
        z-index: 2;
        background:
          linear-gradient(90deg, rgba(5, 18, 29, 0.98), rgba(5, 18, 29, 0.92));
      }

      .sc-table th span {
        display: inline-flex;
        align-items: center;
        gap: 3px;
      }

      .sc-table th em {
        color: var(--gold);
        font-style: normal;
        font-size: 0.58rem;
      }

      .sc-table th.is-sortable {
        cursor: pointer;
        user-select: none;
      }

      .sc-table th.is-sorted {
        color: var(--cyan);
      }

      .sc-skaters-table td {
        padding-top: 7px;
        padding-bottom: 7px;
      }

      .sc-skaters-table tr:nth-child(even) td {
        background-color: rgba(255, 255, 255, 0.012);
      }

      .sc-skaters-table tr:hover td {
        background-color: rgba(0, 216, 223, 0.045);
      }

      .sc-skaters-table tr.is-star-row td {
        background-image: linear-gradient(90deg, rgba(232, 165, 54, 0.055), transparent 45%);
      }

      .sc-skaters-table tr.is-usage-row td {
        background-image: linear-gradient(90deg, rgba(0, 216, 223, 0.04), transparent 45%);
      }

      .sc-skaters-table tr.is-rookie-row td {
        background-image: linear-gradient(90deg, rgba(72, 216, 139, 0.045), transparent 45%);
      }

      .sc-skater-stat {
        min-width: 52px;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
      }

      .sc-skater-stat strong {
        color: #ffffff;
        font-size: 0.9rem;
        line-height: 1;
        font-weight: 950;
      }

      .sc-skater-stat span {
        color: var(--muted-2);
        font-size: 0.62rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: 0.03em;
      }

      .sc-skater-stat.is-elite strong {
        color: var(--gold);
      }

      .sc-skater-stat.is-good strong {
        color: var(--green);
      }

      .sc-skater-stat.is-neutral strong {
        color: var(--cyan);
      }

      .sc-skater-stat.is-warn strong {
        color: var(--orange);
      }

      .sc-skater-stat.is-bad strong {
        color: var(--red);
      }

      .sc-skater-stat.is-depth strong {
        color: var(--muted);
      }

      .sc-skater-role-chip {
        min-height: 26px;
        max-width: 190px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0 10px;
        color: var(--cyan);
        border: 1px solid rgba(0, 216, 223, 0.18);
        background: rgba(0, 216, 223, 0.075);
        font-size: 0.58rem;
        font-weight: 950;
        letter-spacing: 0.075em;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-role-chip.is-elite {
        color: var(--gold);
        border-color: rgba(232, 165, 54, 0.32);
        background: rgba(232, 165, 54, 0.1);
      }

      .sc-skater-role-chip.is-good {
        color: var(--green);
        border-color: rgba(72, 216, 139, 0.28);
        background: rgba(72, 216, 139, 0.08);
      }

      .sc-skater-role-chip.is-warn {
        color: var(--orange);
        border-color: rgba(255, 159, 67, 0.28);
        background: rgba(255, 159, 67, 0.08);
      }

      .sc-skater-role-chip.is-bad {
        color: var(--red);
        border-color: rgba(255, 100, 100, 0.28);
        background: rgba(255, 100, 100, 0.08);
      }

      .sc-skater-insight-grid {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        overflow: hidden;
      }

      .sc-skater-insight-grid .sc-section {
        padding: 12px;
        border-radius: 22px;
      }

      .sc-skater-insight-grid .sc-section-head {
        margin-bottom: 8px;
      }

      .sc-skater-insight-grid .sc-section-head h2 {
        font-size: 0.9rem;
      }

      .sc-skater-quick-grid {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-skater-quick-card {
        min-width: 0;
        min-height: 62px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.038), rgba(255, 255, 255, 0.012)),
          rgba(255, 255, 255, 0.03);
        padding: 9px 10px;
        overflow: hidden;
      }

      .sc-skater-quick-card span {
        display: block;
        color: var(--muted);
        font-size: 0.56rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-quick-card strong {
        display: block;
        margin-top: 6px;
        color: #fff;
        font-size: 1.25rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-skater-quick-card em {
        display: block;
        margin-top: 5px;
        color: var(--muted-2);
        font-size: 0.61rem;
        line-height: 1.15;
        font-style: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-quick-card.is-elite strong {
        color: var(--gold);
      }

      .sc-skater-quick-card.is-good strong {
        color: var(--green);
      }

      .sc-skater-quick-card.is-warn strong {
        color: var(--orange);
      }

      .sc-skater-quick-card.is-bad strong {
        color: var(--red);
      }

      @media (max-width: 1250px) {
        .sc-skater-insight-grid {
          grid-template-columns: 1fr;
          overflow: auto;
        }

        .sc-skaters-page {
          grid-template-rows: minmax(0, 1fr) 280px;
        }
      }

      /* =========================================================
         SKATERS TAB V2 — ADAPTIVE DENSITY FIX
      ========================================================= */

      .sc-skaters-page-v2 {
        grid-template-rows: minmax(0, 1fr) 104px;
        gap: 10px;
        overflow: hidden;
      }

      .sc-skaters-page-v2 > .sc-section {
        padding: 11px 12px;
        border-radius: 22px;
      }

      .sc-skaters-page-v2 .sc-section-head {
        margin-bottom: 8px;
        align-items: center;
      }

      .sc-skaters-page-v2 .sc-section-head h2 {
        font-size: 0.96rem;
      }

      .sc-skater-toolbar {
        gap: 6px;
        flex-wrap: nowrap;
      }

      .sc-column-preset-toggle {
        min-height: 28px;
        display: inline-grid;
        grid-template-columns: repeat(5, auto);
        gap: 4px;
        padding: 3px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.085);
        background: rgba(255, 255, 255, 0.035);
      }

      .sc-column-preset-toggle button {
        min-height: 22px;
        border: 0;
        border-radius: 999px;
        padding: 0 8px;
        background: transparent;
        color: var(--muted);
        font-size: 0.54rem;
        font-weight: 1000;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
        transition:
          color 150ms ease,
          background 150ms ease,
          transform 150ms ease;
      }

      .sc-column-preset-toggle button:hover {
        color: var(--text);
        background: rgba(0, 216, 223, 0.08);
      }

      .sc-column-preset-toggle button.is-active {
        color: #fff;
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.18), rgba(255, 255, 255, 0.03)),
          rgba(0, 216, 223, 0.08);
      }

      .sc-skaters-table-v2 {
        min-width: 1180px;
      }

      .sc-skaters-table-v2 th.is-player-col,
      .sc-skaters-table-v2 td.is-player-col {
        position: sticky;
        left: 0;
        z-index: 4;
        min-width: 220px;
        max-width: 240px;
        background: rgba(5, 18, 29, 0.98);
      }

      .sc-skaters-table-v2 td.is-player-col {
        z-index: 3;
        background:
          linear-gradient(90deg, rgba(5, 18, 29, 0.99), rgba(5, 18, 29, 0.94));
        box-shadow: 12px 0 18px rgba(0, 0, 0, 0.16);
      }

      .sc-skaters-table-v2 th {
        padding: 8px 10px;
        font-size: 0.56rem;
      }

      .sc-skaters-table-v2 td {
        padding: 6px 10px;
        font-size: 0.7rem;
      }

      .sc-skaters-table-v2 tr:nth-child(even) td {
        background-color: rgba(255, 255, 255, 0.011);
      }

      .sc-skaters-table-v2 tr:hover td {
        background-color: rgba(0, 216, 223, 0.045);
      }

      .sc-skaters-table-v2 tr.is-star-row td {
        background-image: linear-gradient(90deg, rgba(232, 165, 54, 0.05), transparent 42%);
      }

      .sc-skaters-table-v2 tr.is-usage-row td {
        background-image: linear-gradient(90deg, rgba(0, 216, 223, 0.035), transparent 42%);
      }

      .sc-skaters-table-v2 tr.is-rookie-row td {
        background-image: linear-gradient(90deg, rgba(72, 216, 139, 0.04), transparent 42%);
      }

      .sc-skaters-table-v2 .sc-name-cell {
        min-width: 205px;
        grid-template-columns: 30px minmax(0, 1fr);
        gap: 8px;
      }

      .sc-skaters-table-v2 .sc-avatar.is-small {
        width: 30px;
        height: 30px;
        border-radius: 11px;
      }

      .sc-skaters-table-v2 .sc-name-cell strong {
        font-size: 0.76rem;
      }

      .sc-skaters-table-v2 .sc-name-cell span:not(.sc-player-meta-inline):not(.sc-player-ovr-drop) {
        font-size: 0.59rem;
        color: var(--muted-2);
      }

      .sc-skaters-table-v2 .sc-player-name-cell-simple .sc-player-meta-inline {
        display: inline-flex;
        margin-top: 0;
        overflow: visible;
        color: inherit;
        font-size: inherit;
      }

      .sc-skaters-table-v2 .sc-player-name-cell-simple .sc-player-ovr-inline {
        color: #5fd4e0;
        font-size: 13px;
        font-weight: 900;
      }

      .sc-table th span {
        display: inline-flex;
        align-items: center;
        gap: 3px;
      }

      .sc-table th em {
        color: var(--gold);
        font-style: normal;
        font-size: 0.55rem;
      }

      .sc-table th.is-sortable {
        cursor: pointer;
        user-select: none;
      }

      .sc-table th.is-sorted {
        color: var(--cyan);
        background:
          linear-gradient(180deg, rgba(0, 216, 223, 0.08), rgba(5, 18, 29, 0.98));
      }

      .sc-skater-stat {
        min-width: 48px;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
      }

      .sc-skater-stat strong {
        color: #ffffff;
        font-size: 0.88rem;
        line-height: 1;
        font-weight: 950;
      }

      .sc-skater-stat span {
        max-width: 78px;
        color: var(--muted-2);
        font-size: 0.58rem;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: 0.02em;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-skater-stat.is-elite strong {
        color: var(--gold);
      }

      .sc-skater-stat.is-good strong {
        color: var(--green);
      }

      .sc-skater-stat.is-neutral strong {
        color: var(--cyan);
      }

      .sc-skater-stat.is-warn strong {
        color: var(--orange);
      }

      .sc-skater-stat.is-bad strong {
        color: var(--red);
      }

      .sc-skater-stat.is-depth strong,
      .sc-skater-stat.is-sample strong {
        color: var(--muted);
      }

      .sc-skater-stat.is-sample span {
        color: var(--gold);
      }

      .sc-skater-role-chip {
        min-height: 24px;
        max-width: 170px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0 9px;
        color: var(--cyan);
        border: 1px solid rgba(0, 216, 223, 0.18);
        background: rgba(0, 216, 223, 0.075);
        font-size: 0.54rem;
        font-weight: 950;
        letter-spacing: 0.065em;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-role-chip.is-elite {
        color: var(--gold);
        border-color: rgba(232, 165, 54, 0.32);
        background: rgba(232, 165, 54, 0.1);
      }

      .sc-skater-role-chip.is-good {
        color: var(--green);
        border-color: rgba(72, 216, 139, 0.28);
        background: rgba(72, 216, 139, 0.08);
      }

      .sc-skater-role-chip.is-neutral {
        color: var(--cyan);
        border-color: rgba(0, 216, 223, 0.22);
        background: rgba(0, 216, 223, 0.065);
      }

      .sc-skater-role-chip.is-warn {
        color: var(--orange);
        border-color: rgba(255, 159, 67, 0.28);
        background: rgba(255, 159, 67, 0.08);
      }

      .sc-skater-role-chip.is-bad {
        color: var(--red);
        border-color: rgba(255, 100, 100, 0.28);
        background: rgba(255, 100, 100, 0.08);
      }

      .sc-skater-summary-strip {
        min-height: 0;
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(12, minmax(0, 1fr));
        gap: 8px;
        overflow: hidden;
      }

      .sc-skater-summary-chip {
        min-width: 0;
        height: 104px;
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.075);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.042), rgba(255, 255, 255, 0.012)),
          rgba(10, 28, 42, 0.82);
        padding: 9px 9px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .sc-skater-summary-chip span {
        display: block;
        color: var(--muted);
        font-size: 0.52rem;
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 950;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-summary-chip strong {
        display: block;
        margin-top: 7px;
        color: #fff;
        font-size: 1.28rem;
        line-height: 1;
        font-weight: 1000;
      }

      .sc-skater-summary-chip em {
        display: block;
        margin-top: 7px;
        color: var(--muted-2);
        font-size: 0.56rem;
        line-height: 1.12;
        font-style: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .sc-skater-summary-chip.is-elite strong {
        color: var(--gold);
      }

      .sc-skater-summary-chip.is-good strong {
        color: var(--green);
      }

      .sc-skater-summary-chip.is-neutral strong {
        color: var(--cyan);
      }

      .sc-skater-summary-chip.is-warn strong {
        color: var(--orange);
      }

      .sc-skater-summary-chip.is-bad strong {
        color: var(--red);
      }

      .sc-table-wrap::-webkit-scrollbar {
        height: 9px;
        width: 9px;
      }

      .sc-table-wrap::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.035);
        border-radius: 999px;
      }

      .sc-table-wrap::-webkit-scrollbar-thumb {
        background:
          linear-gradient(90deg, rgba(0, 216, 223, 0.42), rgba(232, 165, 54, 0.35));
        border-radius: 999px;
      }

      .sc-table-wrap::-webkit-scrollbar-thumb:hover {
        background:
          linear-gradient(90deg, rgba(0, 216, 223, 0.62), rgba(232, 165, 54, 0.5));
      }

      @media (max-width: 1450px) {
        .sc-skater-summary-strip {
          grid-template-columns: repeat(6, minmax(0, 1fr));
          overflow: auto;
        }

        .sc-skater-summary-chip {
          min-width: 132px;
        }
      }

      @media (max-height: 820px) {
        .statscentral-shell {
          height: 100vh;
          gap: 9px;
          padding: 12px 16px 8px;
        }

        .sc-topbar {
          min-height: 96px;
        }

        .sc-title-card,
        .sc-control-card,
        .sc-status-card {
          border-radius: 22px;
        }

        .sc-title-copy h1 {
          font-size: clamp(1.65rem, 2.7vw, 2.75rem);
        }

        .sc-title-copy span {
          font-size: 0.7rem;
        }

        .sc-menu {
          min-height: 42px;
        }

        .sc-menu button {
          min-height: 40px;
          border-radius: 14px;
        }

        .sc-skaters-page-v2 {
          grid-template-rows: minmax(0, 1fr) 86px;
        }

        .sc-skater-summary-chip {
          height: 86px;
          padding: 7px 8px;
        }

        .sc-skater-summary-chip strong {
          margin-top: 5px;
          font-size: 1.05rem;
        }

        .sc-skater-summary-chip em {
          margin-top: 5px;
          font-size: 0.51rem;
        }
      }

      @media (max-width: 1480px) {
        .sc-overview-fixed-story {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .sc-overview-fixed-main {
          grid-template-columns: 1fr;
        }

        .sc-overview-fixed-bottom {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 1220px) {
        .sc-overview-fixed {
          height: auto;
          min-height: 860px;
          grid-template-rows: auto auto auto auto;
          overflow: visible;
        }
      }

      @media (max-width: 860px) {
        .sc-overview-fixed-hero {
          grid-template-columns: 1fr;
        }

        .sc-overview-fixed-story {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-overview-fixed-driver-grid,
        .sc-overview-fixed-special-grid,
        .sc-overview-fixed-panel.is-scores .sc-score-list {
          grid-template-columns: 1fr;
        }
      }

      /* UI-only three-menu redesign: CalendarScreen-style shell */
      .statscentral-shell {
        height: 100vh;
        grid-template-rows: auto minmax(0, 1fr);
        gap: 14px;
        padding: 18px 22px 14px;
        background:
          radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.10), transparent 30%),
          radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.07), transparent 26%),
          linear-gradient(180deg, #06131f 0%, #020a11 100%);
      }

      .sc-command-bar {
        min-height: 70px;
        display: grid;
        grid-template-columns: 112px minmax(0, 1fr) auto;
        align-items: stretch;
        gap: 12px;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(9, 25, 38, 0.94);
        padding: 10px;
      }

      .sc-back-link {
        border: 1px solid rgba(156, 218, 236, 0.16);
        background: rgba(255, 255, 255, 0.025);
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 1000;
        letter-spacing: 0.14em;
        cursor: pointer;
      }

      .sc-back-link:hover {
        color: #fff;
        border-color: rgba(19, 216, 231, 0.42);
        background: rgba(19, 216, 231, 0.08);
      }

      .sc-command-bar .sc-menu {
        min-height: 0;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .sc-command-bar .sc-menu button {
        min-height: 48px;
        border-radius: 0;
        border: 1px solid rgba(156, 218, 236, 0.12);
        background: rgba(7, 22, 35, 0.72);
        justify-content: flex-start;
        padding: 0 18px;
      }

      .sc-command-bar .sc-menu button.is-active {
        border-color: rgba(19, 216, 231, 0.45);
        background:
          linear-gradient(90deg, rgba(19, 216, 231, 0.18), rgba(19, 216, 231, 0.035)),
          rgba(7, 22, 35, 0.9);
        box-shadow: inset 3px 0 0 var(--cyan);
      }

      .sc-command-bar .sc-menu em {
        font-size: 0.82rem;
        letter-spacing: 0.16em;
      }

      .sc-command-actions {
        display: flex;
        align-items: stretch;
        justify-content: flex-end;
        gap: 10px;
      }

      .sc-command-actions .sc-scope-toggle {
        width: 188px;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        border: 1px solid rgba(156, 218, 236, 0.12);
        background: rgba(255, 255, 255, 0.025);
      }

      .sc-command-actions .sc-scope-toggle button {
        min-height: 48px;
        border-radius: 0;
        border: 0;
        background: transparent;
      }

      .sc-command-actions .sc-scope-toggle button.is-active {
        border: 0;
        background: rgba(233, 168, 60, 0.13);
        color: #fff;
      }

      .sc-quick-search {
        min-width: 48px;
        display: flex;
        align-items: stretch;
        justify-content: flex-end;
        border: 1px solid rgba(156, 218, 236, 0.12);
        background: rgba(255, 255, 255, 0.025);
      }

      .sc-quick-search.is-open {
        width: min(280px, 24vw);
      }

      .sc-quick-search input {
        min-width: 0;
        flex: 1;
        border: 0;
        outline: 0;
        background: transparent;
        color: var(--text);
        padding: 0 10px;
        font-size: 0.78rem;
        font-weight: 850;
      }

      .sc-quick-search button {
        width: 48px;
        border: 0;
        border-left: 1px solid rgba(156, 218, 236, 0.12);
        background: transparent;
        color: var(--cyan);
        cursor: pointer;
        font-size: 1.05rem;
      }

      .sc-content {
        overflow: auto;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(6, 18, 29, 0.68);
        padding: 12px;
      }

      .sc-menu-stack {
        min-height: 100%;
        display: grid;
        gap: 14px;
        align-content: start;
      }

      .sc-menu-stack > .sc-tab-page {
        height: auto;
        min-height: 540px;
        overflow: visible;
      }

      .sc-team-menu-stack > .sc-tab-page {
        min-height: calc(100vh - 240px);
      }

      .sc-section,
      .sc-stat-card,
      .sc-award-card,
      .sc-compare-card,
      .sc-team-profile-card {
        border-radius: 0;
        border-color: rgba(156, 218, 236, 0.14);
        background: rgba(9, 25, 38, 0.94);
        box-shadow: none;
        backdrop-filter: none;
      }

      .sc-section-head {
        border-bottom: 1px solid rgba(156, 218, 236, 0.12);
        padding-bottom: 10px;
      }

      .sc-section-head p {
        color: var(--gold);
        letter-spacing: 0.18em;
      }

      .sc-section-head h2 {
        font-size: 1.18rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
      }

      .sc-table-wrap {
        border-radius: 0;
        border-color: rgba(156, 218, 236, 0.12);
      }

      .sc-table th {
        background: rgba(4, 16, 26, 0.98);
        border-bottom-color: rgba(19, 216, 231, 0.22);
        font-size: 0.68rem;
      }

      .sc-table td {
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom-color: rgba(156, 218, 236, 0.09);
        font-size: 0.8rem;
      }

      .sc-name-cell {
        grid-template-columns: 58px minmax(0, 1fr);
        gap: 12px;
      }

      .sc-avatar {
        position: relative;
        width: 48px;
        height: 48px;
        border-radius: 0;
        border: 1px solid rgba(156, 218, 236, 0.18);
        background: rgba(3, 12, 20, 0.92);
        overflow: visible;
      }

      .sc-avatar.is-small {
        width: 46px;
        height: 46px;
      }

      .sc-avatar--headshot > img,
      .sc-avatar--fallback > span {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: grid;
        place-items: center;
      }

      .sc-avatar--fallback > span {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 1000;
      }

      .sc-avatar-team-logo {
        position: absolute;
        right: -8px;
        bottom: -8px;
        width: 24px;
        height: 24px;
        display: grid;
        place-items: center;
        padding: 3px;
        background: rgba(4, 16, 26, 0.98);
        border: 1px solid rgba(233, 168, 60, 0.38);
      }

      .sc-avatar-team-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
      }

      .sc-name-cell strong {
        font-size: 0.9rem;
      }

      .sc-team-logo-mark.is-small {
        width: 50px;
        height: 50px;
        border-radius: 0;
      }

      .sc-team-logo-mark.is-large {
        width: 88px;
        height: 88px;
        border-radius: 0;
      }

      .sc-pill,
      .sc-awards-subtabs button,
      .sc-scope-toggle button {
        border-radius: 0;
      }

      .sc-bottom-grid,
      .sc-leaders-grid {
        gap: 12px;
      }

      .sc-topbar,
      .sc-title-card,
      .sc-control-card,
      .sc-status-card,
      .sc-mini-status,
      .sc-menu-icon {
        display: none;
      }

      @media (max-width: 560px) {
        .sc-overview-fixed-story,
        .sc-overview-fixed-calendar {
          grid-template-columns: 1fr;
        }
      }
      .sc-table-wrap::-webkit-scrollbar,
      .sc-mini-stack::-webkit-scrollbar,
      .sc-calendar-list::-webkit-scrollbar,
      .sc-score-list::-webkit-scrollbar,
      .sc-tier-list::-webkit-scrollbar,
      .sc-danger-grid::-webkit-scrollbar,
      .sc-list-cards::-webkit-scrollbar,
      .sc-note-stack::-webkit-scrollbar,
      .sc-log-list::-webkit-scrollbar,
      .sc-award-grid::-webkit-scrollbar,
      .sc-formula-sections::-webkit-scrollbar,
      .sc-driver-grid::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .sc-table-wrap::-webkit-scrollbar-thumb,
      .sc-mini-stack::-webkit-scrollbar-thumb,
      .sc-calendar-list::-webkit-scrollbar-thumb,
      .sc-score-list::-webkit-scrollbar-thumb,
      .sc-tier-list::-webkit-scrollbar-thumb,
      .sc-danger-grid::-webkit-scrollbar-thumb,
      .sc-list-cards::-webkit-scrollbar-thumb,
      .sc-note-stack::-webkit-scrollbar-thumb,
      .sc-log-list::-webkit-scrollbar-thumb,
      .sc-award-grid::-webkit-scrollbar-thumb,
      .sc-formula-sections::-webkit-scrollbar-thumb,
      .sc-driver-grid::-webkit-scrollbar-thumb {
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
      }

      .sc-table-wrap::-webkit-scrollbar-track,
      .sc-mini-stack::-webkit-scrollbar-track,
      .sc-calendar-list::-webkit-scrollbar-track,
      .sc-score-list::-webkit-scrollbar-track,
      .sc-tier-list::-webkit-scrollbar-track,
      .sc-danger-grid::-webkit-scrollbar-track,
      .sc-list-cards::-webkit-scrollbar-track,
      .sc-note-stack::-webkit-scrollbar-track,
      .sc-log-list::-webkit-scrollbar-track,
      .sc-award-grid::-webkit-scrollbar-track,
      .sc-formula-sections::-webkit-scrollbar-track,
      .sc-driver-grid::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.035);
        border-radius: 999px;
      }

      @media (max-width: 1480px) {
        .sc-topbar {
          grid-template-columns: minmax(320px, 1.1fr) minmax(300px, 0.75fr) minmax(240px, 0.55fr);
        }

        .sc-menu {
          grid-template-columns: repeat(6, minmax(0, 1fr));
        }

        .sc-menu button {
          min-height: 44px;
        }

        .sc-stat-grid,
        .sc-stat-grid-clean {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .sc-overview-feature-grid {
          grid-template-columns: 1fr;
        }

        .sc-leaders-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          overflow: auto;
        }

        .sc-award-grid {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }
      }

      @media (max-width: 1220px) {
        .statscentral-shell {
          height: auto;
          min-height: 100vh;
          overflow: auto;
        }

        .sc-topbar {
          grid-template-columns: 1fr;
        }

        .sc-status-card {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .sc-status-card.is-compact-status {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .sc-content {
          min-height: 860px;
          overflow: visible;
        }

        .sc-overview {
          grid-template-columns: 1fr;
          overflow: visible;
        }

        .sc-overview-redesign {
          grid-template-rows: auto auto auto;
        }

        .sc-overview-main,
        .sc-overview-side {
          overflow: visible;
        }

        .sc-overview-bottom-strip {
          grid-template-columns: 1fr;
        }

        .sc-calendar-list-compact {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-tab-page {
          min-height: 860px;
        }

        .sc-stat-grid,
        .sc-stat-grid-clean {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .sc-bottom-grid {
          grid-template-columns: 1fr;
        }

        .sc-formula-sections {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 860px) {
        .statscentral-shell {
          padding: 12px;
        }

        .sc-title-card {
          align-items: flex-start;
        }

        .sc-title-copy h1 {
          font-size: 2rem;
        }

        .sc-status-card,
        .sc-status-card.is-compact-status {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-menu {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .sc-menu-icon {
          font-size: 0.64rem;
        }

        .sc-menu em {
          font-size: 0.54rem;
        }

        .sc-franchise-snapshot {
          grid-template-columns: 1fr;
        }

        .sc-stat-grid,
        .sc-stat-grid-clean {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-driver-grid {
          grid-template-columns: 1fr;
        }

        .sc-mini-side-section .sc-score-list {
          grid-template-columns: 1fr;
        }

        .sc-leaders-grid {
          grid-template-columns: 1fr;
        }

        .sc-award-grid {
          grid-template-columns: 1fr;
        }

        .sc-compare-grid,
        .sc-compare-selectors {
          grid-template-columns: 1fr;
        }

        .sc-trend-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-formula-list {
          grid-template-columns: 1fr;
        }

        .sc-team-profile-card {
          grid-template-columns: 1fr;
        }

        .sc-team-profile-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }

      @media (max-width: 560px) {
        .sc-title-card {
          flex-direction: column;
        }

        .sc-back-button {
          width: 100%;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }

        .sc-back-button em {
          margin-top: 0;
        }

        .sc-scope-toggle {
          grid-template-columns: 1fr;
        }

        .sc-menu {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-stat-grid,
        .sc-stat-grid-clean,
        .sc-trend-grid,
        .sc-danger-grid,
        .sc-calendar-list-compact {
          grid-template-columns: 1fr;
        }

        .sc-score-line {
          grid-template-columns: minmax(0, 1fr) auto auto auto;
        }

        .sc-score-team.is-away {
          grid-column: 1 / -1;
          text-align: left;
          color: var(--muted);
        }
      }

      /* =========================================================
         TEAM STATS — FULL WORKSPACE REBUILD
      ========================================================= */

      .sc-command-actions.is-empty {
        display: none;
      }

      .sc-content.is-team-stats {
        min-height: 0;
        overflow: hidden;
      }

      .sc-team-menu-stack {
        height: 100%;
        min-height: 0;
        overflow: hidden;
      }

      .sc-team-stats-workspace {
        height: 100%;
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) 154px;
        gap: 10px;
        overflow: hidden;
      }

      .sc-team-toolbar {
        flex: 0 0 auto;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background:
          linear-gradient(
            180deg,
            rgba(15, 38, 54, 0.94),
            rgba(7, 23, 35, 0.94)
          );
      }

      .sc-team-toolbar-top {
        min-height: 48px;
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        border-bottom: 1px solid rgba(156, 218, 236, 0.12);
      }

      .sc-team-toolbar-title {
        min-width: 220px;
        padding: 9px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .sc-team-toolbar-title span {
        color: var(--gold);
        font-size: 0.61rem;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.15em;
      }

      .sc-team-toolbar-title strong {
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.68rem;
        line-height: 1;
        font-weight: 850;
      }

      .sc-team-view-tabs {
        display: flex;
        align-items: stretch;
        justify-content: flex-end;
      }

      .sc-team-view-tabs button {
        min-width: 112px;
        padding: 0 16px;
        border: 0;
        border-left: 1px solid rgba(156, 218, 236, 0.12);
        background: transparent;
        color: var(--muted);
        cursor: pointer;
        font-size: 0.68rem;
        font-weight: 950;
        letter-spacing: 0.09em;
        text-transform: uppercase;
      }

      .sc-team-view-tabs button:hover {
        color: #fff;
        background: rgba(19, 216, 231, 0.06);
      }

      .sc-team-view-tabs button.is-active {
        color: #fff;
        background:
          linear-gradient(
            180deg,
            rgba(19, 216, 231, 0.13),
            rgba(19, 216, 231, 0.035)
          );
        box-shadow: inset 0 -3px 0 var(--cyan);
      }

      .sc-team-toolbar-filters {
        min-height: 44px;
        display: grid;
        grid-template-columns:
          172px
          minmax(150px, 190px)
          minmax(150px, 190px)
          minmax(180px, 1fr)
          76px;
        align-items: stretch;
        gap: 8px;
        padding: 7px 9px;
      }

      .sc-team-scope-control {
        display: grid;
        grid-template-columns: 1fr 1fr;
        border: 1px solid rgba(156, 218, 236, 0.13);
      }

      .sc-team-scope-control button {
        border: 0;
        background: rgba(255, 255, 255, 0.025);
        color: var(--muted);
        cursor: pointer;
        font-size: 0.65rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sc-team-scope-control button + button {
        border-left: 1px solid rgba(156, 218, 236, 0.13);
      }

      .sc-team-scope-control button.is-active {
        color: #fff;
        background: rgba(19, 216, 231, 0.12);
      }

      .sc-team-toolbar-filters select,
      .sc-team-search,
      .sc-team-reset {
        min-width: 0;
        border: 1px solid rgba(156, 218, 236, 0.13);
        background: rgba(3, 14, 23, 0.78);
        color: var(--text);
      }

      .sc-team-toolbar-filters select {
        outline: none;
        padding: 0 10px;
        font-size: 0.68rem;
        font-weight: 800;
      }

      .sc-team-toolbar-filters select:focus {
        border-color: rgba(19, 216, 231, 0.45);
      }

      .sc-team-toolbar-filters option {
        color: #fff;
        background: #071621;
      }

      .sc-team-search {
        display: grid;
        grid-template-columns: 34px minmax(0, 1fr);
        align-items: center;
      }

      .sc-team-search > span {
        color: var(--cyan);
        text-align: center;
        font-size: 0.9rem;
      }

      .sc-team-search input {
        min-width: 0;
        height: 100%;
        border: 0;
        outline: 0;
        background: transparent;
        color: #fff;
        padding: 0 10px 0 0;
        font-size: 0.7rem;
        font-weight: 800;
      }

      .sc-team-search input::placeholder {
        color: var(--muted-2);
      }

      .sc-team-reset {
        cursor: pointer;
        font-size: 0.64rem;
        font-weight: 950;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sc-team-reset:hover {
        color: #fff;
        border-color: rgba(19, 216, 231, 0.4);
        background: rgba(19, 216, 231, 0.08);
      }

      .sc-team-table-panel {
        min-height: 0;
        overflow: hidden;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(3, 13, 22, 0.82);
      }

      .sc-team-table-panel .sc-table-wrap {
        height: 100%;
        border: 0;
        border-radius: 0;
        overflow: auto;
        background: transparent;
      }

      .sc-team-stats-table {
        min-width: 900px;
        table-layout: fixed;
        font-variant-numeric: tabular-nums;
      }

      .sc-team-stats-table th {
        height: 38px;
        padding: 0 13px;
        background: #061722;
        color: #8facbc;
        font-size: 0.63rem;
        letter-spacing: 0.13em;
        border-bottom-color: rgba(156, 218, 236, 0.16);
      }

      .sc-team-stats-table th.is-sorted {
        color: var(--cyan);
        background: #08202c;
      }

      .sc-team-stats-table th em {
        margin-left: 4px;
        color: var(--cyan);
        font-style: normal;
      }

      .sc-team-stats-table td {
        height: 53px;
        padding: 5px 13px;
        color: #eaf6fb;
        font-size: 0.8rem;
        border-bottom-color: rgba(156, 218, 236, 0.075);
        font-variant-numeric: tabular-nums;
      }

      .sc-team-stats-table tbody tr:nth-child(even) td {
        background: rgba(255, 255, 255, 0.012);
      }

      .sc-team-stats-table tbody tr.is-interactive-row {
        cursor: pointer;
        outline: none;
      }

      .sc-team-stats-table tbody tr.is-interactive-row:hover td {
        background: rgba(19, 216, 231, 0.055);
      }

      .sc-team-stats-table tbody tr.is-interactive-row:focus-visible td {
        box-shadow: inset 0 0 0 1px rgba(19, 216, 231, 0.5);
      }

      .sc-team-stats-table tbody tr.is-selected-row td {
        background: rgba(140, 181, 202, 0.075);
        border-top: 1px solid rgba(156, 218, 236, 0.2);
        border-bottom: 1px solid rgba(156, 218, 236, 0.2);
      }

      .sc-team-stats-table tbody tr.is-user-team-row td:first-child {
        box-shadow: inset 4px 0 0 var(--cyan);
      }

      .sc-team-stats-table tbody tr.is-user-team-row td {
        background:
          linear-gradient(
            90deg,
            rgba(19, 216, 231, 0.075),
            rgba(19, 216, 231, 0.025)
          );
      }

      .sc-team-stats-table th.is-rank-col,
      .sc-team-stats-table td.is-rank-col {
        width: 58px;
        min-width: 58px;
        max-width: 58px;
        text-align: center;
      }

      .sc-team-stats-table th.is-team-col,
      .sc-team-stats-table td.is-team-col {
        width: 300px;
        min-width: 300px;
        max-width: 300px;
      }

      .sc-team-stats-table th.is-right,
      .sc-team-stats-table td.is-right {
        text-align: right;
      }

      .sc-team-stats-table th.is-group-start,
      .sc-team-stats-table td.is-group-start {
        border-left: 1px solid rgba(156, 218, 236, 0.14);
      }

      .sc-team-stats-table th:nth-child(1),
      .sc-team-stats-table td:nth-child(1) {
        position: sticky;
        left: 0;
      }

      .sc-team-stats-table th:nth-child(2),
      .sc-team-stats-table td:nth-child(2) {
        position: sticky;
        left: 58px;
      }

      .sc-team-stats-table th:nth-child(1),
      .sc-team-stats-table th:nth-child(2) {
        z-index: 5;
        background: #061722;
      }

      .sc-team-stats-table td:nth-child(1),
      .sc-team-stats-table td:nth-child(2) {
        z-index: 3;
        background: #071621;
      }

      .sc-team-stats-table tr:nth-child(even) td:nth-child(1),
      .sc-team-stats-table tr:nth-child(even) td:nth-child(2) {
        background: #081824;
      }

      .sc-team-stats-table tr.is-user-team-row td:nth-child(1),
      .sc-team-stats-table tr.is-user-team-row td:nth-child(2) {
        background: #08232d;
      }

      .sc-team-stats-table tr.is-selected-row td:nth-child(1),
      .sc-team-stats-table tr.is-selected-row td:nth-child(2) {
        background: #102630;
      }

      .sc-team-view-rank {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 950;
      }

      .sc-team-name-cell {
        min-width: 0;
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        align-items: center;
        gap: 11px;
      }

      .sc-team-logo-mark.is-table {
        width: 40px;
        height: 40px;
        border-radius: 0;
      }

      .sc-team-stats-workspace .sc-team-logo-mark--logo {
        border: 0;
        background: transparent;
        padding: 2px;
      }

      .sc-team-logo-mark img {
        width: 100%;
        height: 100%;
        object-fit: contain;
      }

      .sc-team-logo-mark--fallback.is-table {
        border: 1px solid rgba(156, 218, 236, 0.2);
        background: rgba(8, 24, 35, 0.94);
        color: #fff;
        font-size: 0.58rem;
        font-weight: 1000;
      }

      .sc-team-name-copy {
        min-width: 0;
      }

      .sc-team-overall-rank {
        display: block;
        width: fit-content;
        margin-top: 3px;
        color: #f0a23a;
        font-size: 9px;
        font-style: normal;
        font-weight: 800;
        line-height: 1;
        letter-spacing: 0.1em;
        white-space: nowrap;
      }

      .sc-team-name-line {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .sc-team-name-line strong {
        min-width: 0;
        overflow: hidden;
        color: #fff;
        font-size: 0.83rem;
        font-weight: 950;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-team-name-copy > span {
        display: block;
        margin-top: 4px;
        color: #6e8b9b;
        font-size: 0.58rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .sc-my-team-tag {
        flex: 0 0 auto;
        color: var(--cyan);
        font-size: 0.49rem;
        line-height: 1;
        font-style: normal;
        font-weight: 1000;
        letter-spacing: 0.1em;
      }

      .sc-team-value {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        gap: 5px;
        min-width: 46px;
      }

      .sc-team-value strong {
        color: #ecf7fb;
        font-size: 0.78rem;
        font-weight: 900;
      }

      .sc-team-value em {
        min-width: 22px;
        color: var(--muted-2);
        font-size: 0.5rem;
        font-style: normal;
        font-weight: 950;
      }

      .sc-team-value.is-top-five strong,
      .sc-team-profile-metric.is-top-five strong {
        color: #f3bc59;
      }

      .sc-team-value.is-top-five em,
      .sc-team-profile-metric.is-top-five em {
        color: var(--gold);
      }

      .sc-team-value.is-bottom-five strong,
      .sc-team-profile-metric.is-bottom-five strong {
        color: #e68181;
      }

      .sc-team-selected-profile {
        min-height: 0;
        display: grid;
        grid-template-columns:
          minmax(300px, 1.2fr)
          minmax(310px, 0.9fr)
          minmax(560px, 1.8fr);
        align-items: stretch;
        border: 1px solid rgba(156, 218, 236, 0.16);
        background:
          radial-gradient(
            circle at 8% 50%,
            rgba(19, 216, 231, 0.09),
            transparent 24%
          ),
          linear-gradient(
            180deg,
            rgba(13, 36, 51, 0.98),
            rgba(6, 22, 33, 0.98)
          );
        overflow: hidden;
      }

      .sc-team-selected-profile.is-empty {
        display: grid;
        place-items: center;
        color: var(--muted);
        font-size: 0.78rem;
      }

      .sc-team-profile-identity {
        min-width: 0;
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr);
        align-items: center;
        gap: 15px;
        padding: 13px 18px;
      }

      .sc-team-profile-logo {
        width: 88px;
        height: 88px;
        display: grid;
        place-items: center;
      }

      .sc-team-profile-logo .sc-team-logo-mark.is-large {
        width: 84px;
        height: 84px;
        border: 0;
        background: transparent;
      }

      .sc-team-profile-copy {
        min-width: 0;
      }

      .sc-team-profile-copy > span {
        color: var(--cyan);
        font-size: 0.55rem;
        line-height: 1;
        font-weight: 1000;
        letter-spacing: 0.14em;
      }

      .sc-team-profile-copy h2 {
        margin: 7px 0 0;
        overflow: hidden;
        color: #fff;
        font-size: 1.15rem;
        line-height: 1.05;
        font-weight: 1000;
        letter-spacing: -0.025em;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-team-profile-copy p {
        margin: 7px 0 0;
        color: var(--muted);
        font-size: 0.65rem;
        font-weight: 800;
      }

      .sc-team-profile-headline {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border-left: 1px solid rgba(156, 218, 236, 0.13);
        border-right: 1px solid rgba(156, 218, 236, 0.13);
      }

      .sc-team-profile-headline > div {
        min-width: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }

      .sc-team-profile-headline > div + div {
        border-left: 1px solid rgba(156, 218, 236, 0.1);
      }

      .sc-team-profile-headline strong {
        color: #fff;
        font-size: 1.12rem;
        line-height: 1;
        font-weight: 1000;
        font-variant-numeric: tabular-nums;
      }

      .sc-team-profile-headline span {
        margin-top: 7px;
        color: var(--muted);
        font-size: 0.53rem;
        font-weight: 950;
        letter-spacing: 0.1em;
        text-transform: uppercase;
      }

      .sc-team-profile-metrics {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
      }

      .sc-team-profile-metric {
        min-width: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 11px 13px;
      }

      .sc-team-profile-metric + .sc-team-profile-metric {
        border-left: 1px solid rgba(156, 218, 236, 0.1);
      }

      .sc-team-profile-metric span {
        color: var(--muted);
        font-size: 0.53rem;
        font-weight: 950;
        letter-spacing: 0.09em;
        text-transform: uppercase;
      }

      .sc-team-profile-metric strong {
        margin-top: 7px;
        color: #fff;
        font-size: 1rem;
        line-height: 1;
        font-weight: 1000;
        font-variant-numeric: tabular-nums;
      }

      .sc-team-profile-metric em {
        margin-top: 6px;
        overflow: hidden;
        color: #6e8b9b;
        font-size: 0.55rem;
        line-height: 1;
        font-style: normal;
        font-weight: 800;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .sc-team-state-message {
        height: 100%;
        display: grid;
        place-content: center;
        gap: 6px;
        border: 1px solid rgba(156, 218, 236, 0.14);
        background: rgba(5, 18, 28, 0.82);
        text-align: center;
      }

      .sc-team-state-message strong {
        color: #fff;
        font-size: 0.9rem;
      }

      .sc-team-state-message span {
        color: var(--muted);
        font-size: 0.7rem;
      }

      .sc-team-state-message.is-error strong {
        color: #ff8585;
      }

      .sc-team-table-panel .sc-table-wrap::-webkit-scrollbar {
        width: 9px;
        height: 9px;
      }

      .sc-team-table-panel .sc-table-wrap::-webkit-scrollbar-thumb {
        background: rgba(156, 218, 236, 0.24);
      }

      .sc-team-table-panel .sc-table-wrap::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.15);
      }

      @media (max-width: 1400px) {
        .sc-team-view-tabs button {
          min-width: 92px;
          padding: 0 10px;
          font-size: 0.61rem;
        }

        .sc-team-toolbar-filters {
          grid-template-columns:
            150px
            150px
            150px
            minmax(150px, 1fr)
            70px;
        }

        .sc-team-selected-profile {
          grid-template-columns:
            minmax(270px, 1fr)
            minmax(280px, 0.8fr)
            minmax(470px, 1.5fr);
        }

        .sc-team-stats-table th.is-team-col,
        .sc-team-stats-table td.is-team-col {
          width: 260px;
          min-width: 260px;
          max-width: 260px;
        }
      }

      @media (max-width: 1050px) {
        .sc-team-stats-workspace {
          grid-template-rows: auto minmax(0, 1fr) 205px;
        }

        .sc-team-toolbar-top {
          display: grid;
          grid-template-columns: 1fr;
        }

        .sc-team-view-tabs {
          overflow-x: auto;
          border-top: 1px solid rgba(156, 218, 236, 0.1);
        }

        .sc-team-view-tabs button {
          min-height: 42px;
          flex: 1 0 110px;
        }

        .sc-team-toolbar-filters {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .sc-team-search {
          grid-column: 1 / -1;
          min-height: 38px;
        }

        .sc-team-selected-profile {
          grid-template-columns: 1fr 1fr;
          grid-template-rows: auto auto;
        }

        .sc-team-profile-metrics {
          grid-column: 1 / -1;
        }

        .sc-team-profile-headline {
          border-right: 0;
        }
      }
    `}</style>
  );
}