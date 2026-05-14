import React, { useMemo, useState, useCallback } from "react";
import { useGameUI } from "../game/GameUIContext";
import { GameFooter } from "../components/game/GameFooter";
import { SCREENS } from "../game/constants";

/*
  StorylinesScreen.js
  Mega overhaul version.

  Rules for this file:
  - Keep existing backend/frontend connectivity untouched.
  - Read from franchiseState only.
  - Do not require new endpoints.
  - Do not break existing storyline_events.
  - Do not break existing storyline_choices.
  - Do not break onResolveStorylineChoice.
  - Do not break setScreen navigation.
  - Do not hardcode real NHL player names.
  - Fallbacks are UI-safe only and should disappear when backend data exists.
*/

/* ============================================================
   1. BASE SAFETY HELPERS
   ============================================================ */

function clamp(n, min, max) {
  const num = Number(n);
  if (!Number.isFinite(num)) return min;
  return Math.max(min, Math.min(max, num));
}

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function asObject(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}

function safeText(v, fallback = "—") {
  if (v === null || v === undefined || v === "") return fallback;
  return String(v);
}

function number(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function bool(v, fallback = false) {
  if (typeof v === "boolean") return v;
  if (v === "true") return true;
  if (v === "false") return false;
  return fallback;
}

function percent(v, fallback = 0) {
  const n = number(v, fallback);
  if (n <= 1 && n >= 0) return Math.round(n * 100);
  return Math.round(n);
}

function oneOf(...values) {
  for (const value of values) {
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return undefined;
}

function firstNonEmptyArray(...values) {
  for (const value of values) {
    if (Array.isArray(value) && value.length) return value;
  }
  return [];
}

function lower(value) {
  return String(value || "").toLowerCase();
}

function upper(value) {
  return String(value || "").toUpperCase();
}

function titleCase(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase());
}

function compactLabel(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .trim();
}

function initials(name) {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (!parts.length) return "HL";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();

  return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
}

function shortName(name) {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (!parts.length) return "Player";
  if (parts.length === 1) return parts[0];

  return `${parts[0][0]}. ${parts[parts.length - 1]}`;
}

function normalizeId(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function formatSigned(value) {
  const n = number(value, 0);
  if (n > 0) return `+${n}`;
  if (n < 0) return `${n}`;
  return "0";
}

function formatMoneyMillions(value, fallback = "—") {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return `$${n.toFixed(2)}M`;
}

function formatRecord(wins, losses, otl) {
  const w = number(wins, 0);
  const l = number(losses, 0);
  const o = number(otl, 0);
  return `${w}-${l}-${o}`;
}

function formatPct(value, fallback = "—") {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  if (n <= 1) return n.toFixed(3).replace(/^0/, "");
  return (n / 100).toFixed(3).replace(/^0/, "");
}

function formatDateShort(raw, fallback = "Today") {
  if (!raw) return fallback;

  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return String(raw);

  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function formatDateLong(raw, fallback = "Today") {
  if (!raw) return fallback;

  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return String(raw);

  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatWeekday(raw, fallback = "Current Day") {
  if (!raw) return fallback;

  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return fallback;

  return d.toLocaleDateString(undefined, {
    weekday: "short",
  });
}

function dateValue(raw) {
  if (!raw) return 0;
  const d = new Date(raw);
  const t = d.getTime();
  return Number.isNaN(t) ? 0 : t;
}

function timeAgoFromRaw(raw) {
  if (!raw) return "Today";

  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return String(raw);

  const diffMs = Date.now() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "1d ago";
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;

  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function readPath(obj, path, fallback = undefined) {
  const parts = String(path || "")
    .split(".")
    .map((x) => x.trim())
    .filter(Boolean);

  let current = obj;
  for (const part of parts) {
    if (current === null || current === undefined) return fallback;
    current = current[part];
  }

  return current === undefined ? fallback : current;
}

function flattenDeep(value) {
  if (!Array.isArray(value)) return [];
  const out = [];

  for (const item of value) {
    if (Array.isArray(item)) out.push(...flattenDeep(item));
    else if (item !== null && item !== undefined) out.push(item);
  }

  return out;
}

function uniqueBy(list, getKey) {
  const seen = new Set();
  const out = [];

  for (const item of asArray(list)) {
    const key = getKey(item);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }

  return out;
}

function sumNumbers(values) {
  return asArray(values)
    .map((v) => Number(v))
    .filter(Number.isFinite)
    .reduce((a, b) => a + b, 0);
}

function averageNumbers(values, fallback = 0) {
  const nums = asArray(values).map(Number).filter(Number.isFinite);
  if (!nums.length) return fallback;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function sortByDateDesc(a, b) {
  return dateValue(b?.calendar_iso || b?.date || b?.created_at || b?.timestamp) -
    dateValue(a?.calendar_iso || a?.date || a?.created_at || a?.timestamp);
}

function sortByDateAsc(a, b) {
  return dateValue(a?.calendar_iso || a?.date || a?.created_at || a?.timestamp) -
    dateValue(b?.calendar_iso || b?.date || b?.created_at || b?.timestamp);
}

/* ============================================================
   2. FRANCHISE STATE SELECTORS
   ============================================================ */

function getTeamName(franchiseState) {
  return (
    franchiseState?.user_team?.name ||
    franchiseState?.team?.name ||
    franchiseState?.selected_team?.name ||
    franchiseState?.franchise?.team_name ||
    franchiseState?.team_name ||
    "Franchise"
  );
}

function getTeamCity(franchiseState) {
  return (
    franchiseState?.user_team?.city ||
    franchiseState?.team?.city ||
    franchiseState?.selected_team?.city ||
    franchiseState?.franchise?.city ||
    ""
  );
}

function getTeamAbbr(franchiseState) {
  return (
    franchiseState?.user_team?.abbr ||
    franchiseState?.user_team?.abbreviation ||
    franchiseState?.team?.abbr ||
    franchiseState?.team?.abbreviation ||
    franchiseState?.selected_team?.abbr ||
    franchiseState?.selected_team?.abbreviation ||
    franchiseState?.team_abbr ||
    initials(getTeamName(franchiseState))
  );
}

function getTeamId(franchiseState) {
  return (
    franchiseState?.user_team?.id ||
    franchiseState?.user_team?.team_id ||
    franchiseState?.team?.id ||
    franchiseState?.team?.team_id ||
    franchiseState?.selected_team?.id ||
    franchiseState?.selected_team?.team_id ||
    franchiseState?.team_id ||
    getTeamAbbr(franchiseState)
  );
}

function getGMName(franchiseState) {
  return (
    franchiseState?.gm_name ||
    franchiseState?.user_gm?.name ||
    franchiseState?.franchise?.gm_name ||
    "GM Mode"
  );
}

function getSeasonLabel(franchiseState) {
  return (
    franchiseState?.season_label ||
    franchiseState?.season?.label ||
    franchiseState?.season?.name ||
    franchiseState?.calendar?.season_label ||
    "Franchise Season"
  );
}

function getCurrentRawDate(franchiseState) {
  return (
    franchiseState?.calendar?.current_date ||
    franchiseState?.calendar_iso ||
    franchiseState?.date ||
    franchiseState?.current_date ||
    franchiseState?.season_date ||
    franchiseState?.today ||
    null
  );
}

function getDateLabel(franchiseState) {
  const raw = getCurrentRawDate(franchiseState);

  if (!raw) return { main: "Today", sub: "Current Day", raw: null };

  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) {
    return { main: String(raw), sub: "Current Day", raw };
  }

  return {
    main: d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    sub: d.toLocaleDateString(undefined, {
      weekday: "short",
    }),
    raw,
  };
}

function getWeekLabel(franchiseState) {
  return (
    franchiseState?.calendar?.week_label ||
    franchiseState?.week_label ||
    franchiseState?.season_week ||
    franchiseState?.calendar_week ||
    "Week —"
  );
}

function getCalendarPhase(franchiseState) {
  return (
    franchiseState?.calendar?.phase ||
    franchiseState?.season_phase ||
    franchiseState?.phase ||
    franchiseState?.calendar?.season_phase ||
    "Regular Season"
  );
}

function getNextGame(franchiseState) {
  const schedule = collectSchedule(franchiseState);

  const next =
    franchiseState?.next_game ||
    franchiseState?.calendar?.next_game ||
    franchiseState?.upcoming_game ||
    schedule.find((g) => !g.played && !g.completed && !g.final);

  if (!next) {
    return {
      days: "—",
      opponent: "No Game",
      label: "Next Game",
      homeAway: "—",
      raw: null,
    };
  }

  const userTeamId = getTeamId(franchiseState);
  const userTeamName = getTeamName(franchiseState);

  const homeName = oneOf(next.home_team_name, next.home_name, next.homeTeamName, next.home);
  const awayName = oneOf(next.away_team_name, next.away_name, next.awayTeamName, next.away);
  const homeId = oneOf(next.home_team_id, next.home_id, next.homeTeamId);
  const awayId = oneOf(next.away_team_id, next.away_id, next.awayTeamId);

  const isHome =
    String(homeId || "").toLowerCase() === String(userTeamId || "").toLowerCase() ||
    String(homeName || "").toLowerCase() === String(userTeamName || "").toLowerCase();

  const isAway =
    String(awayId || "").toLowerCase() === String(userTeamId || "").toLowerCase() ||
    String(awayName || "").toLowerCase() === String(userTeamName || "").toLowerCase();

  let opponent =
    next.opponent_name ||
    next.opponent ||
    next.away_team_name ||
    next.home_team_name ||
    next.team_name ||
    "Opponent";

  if (isHome && awayName) opponent = awayName;
  if (isAway && homeName) opponent = homeName;

  const days =
    next.days_until ??
    next.daysAway ??
    next.days_to_game ??
    next.in_days ??
    calculateDaysUntil(getCurrentRawDate(franchiseState), next.date || next.calendar_iso);

  return {
    days,
    opponent,
    label: "Next Game",
    homeAway: isHome ? "Home" : isAway ? "Away" : "—",
    raw: next,
  };
}

function calculateDaysUntil(currentRaw, targetRaw) {
  if (!currentRaw || !targetRaw) return "—";

  const current = new Date(currentRaw);
  const target = new Date(targetRaw);

  if (Number.isNaN(current.getTime()) || Number.isNaN(target.getTime())) return "—";

  const diff = target.getTime() - current.getTime();
  return Math.max(0, Math.ceil(diff / (1000 * 60 * 60 * 24)));
}

function getMorale(franchiseState) {
  const raw =
    franchiseState?.team_morale ??
    franchiseState?.morale ??
    franchiseState?.user_team?.morale ??
    franchiseState?.team?.morale ??
    franchiseState?.chemistry?.morale ??
    franchiseState?.analytics?.team_morale ??
    58;

  return clamp(raw, 0, 100);
}

function getChemistry(franchiseState) {
  const raw =
    franchiseState?.team_chemistry ??
    franchiseState?.chemistry?.team ??
    franchiseState?.chemistry ??
    franchiseState?.user_team?.chemistry ??
    franchiseState?.team?.chemistry ??
    franchiseState?.analytics?.team_chemistry ??
    72;

  return clamp(raw, 0, 100);
}

function getFanConfidence(franchiseState) {
  const raw =
    franchiseState?.fan_confidence ??
    franchiseState?.fans?.confidence ??
    franchiseState?.user_team?.fan_confidence ??
    franchiseState?.team?.fan_confidence ??
    franchiseState?.analytics?.fan_confidence ??
    78;

  return clamp(raw, 0, 100);
}

function getGmRating(franchiseState) {
  const raw =
    franchiseState?.gm_rating ??
    franchiseState?.gm?.rating ??
    franchiseState?.user_gm?.rating ??
    franchiseState?.franchise?.gm_rating ??
    franchiseState?.analytics?.gm_rating ??
    84;

  return clamp(raw, 0, 100);
}

function getOwnerPatience(franchiseState) {
  const raw =
    franchiseState?.owner_patience ??
    franchiseState?.owner?.patience ??
    franchiseState?.ownership?.patience ??
    franchiseState?.analytics?.owner_patience ??
    70;

  return clamp(raw, 0, 100);
}

function getMediaPressure(franchiseState) {
  const raw =
    franchiseState?.media_pressure ??
    franchiseState?.media?.pressure ??
    franchiseState?.analytics?.media_pressure ??
    50;

  return clamp(raw, 0, 100);
}

function getRoomTension(franchiseState) {
  const raw =
    franchiseState?.room_tension ??
    franchiseState?.locker_room?.tension ??
    franchiseState?.analytics?.room_tension ??
    Math.max(0, 100 - getMorale(franchiseState));

  return clamp(raw, 0, 100);
}

function moraleLabel(value) {
  if (value >= 85) return "Elite";
  if (value >= 75) return "Strong";
  if (value >= 60) return "Stable";
  if (value >= 45) return "Uneasy";
  if (value >= 30) return "Negative";
  return "Crisis";
}

function chemistryLabel(value) {
  if (value >= 85) return "Excellent";
  if (value >= 70) return "Good";
  if (value >= 55) return "Average";
  if (value >= 40) return "Weak";
  return "Broken";
}

function confidenceLabel(value) {
  if (value >= 82) return "Strong";
  if (value >= 66) return "Positive";
  if (value >= 50) return "Mixed";
  if (value >= 35) return "Nervous";
  return "Hostile";
}

function pressureLabel(value) {
  if (value >= 82) return "Boiling";
  if (value >= 65) return "High";
  if (value >= 45) return "Manageable";
  if (value >= 25) return "Quiet";
  return "Calm";
}

function valueTone(value, goodHigh = true) {
  const v = number(value, 0);
  if (goodHigh) {
    if (v >= 70) return "positive";
    if (v >= 45) return "neutral";
    return "negative";
  }

  if (v >= 70) return "negative";
  if (v >= 45) return "neutral";
  return "positive";
}

/* ============================================================
   3. SCHEDULE / STANDINGS / TEAM COLLECTIONS
   ============================================================ */

function collectSchedule(franchiseState) {
  return firstNonEmptyArray(
    franchiseState?.schedule,
    franchiseState?.calendar?.schedule,
    franchiseState?.games,
    franchiseState?.calendar?.games,
    franchiseState?.season_schedule,
    franchiseState?.team_schedule
  );
}

function collectTeams(franchiseState) {
  return firstNonEmptyArray(
    franchiseState?.teams,
    franchiseState?.league_teams,
    franchiseState?.standings?.teams,
    franchiseState?.league?.teams
  );
}

function collectStandings(franchiseState) {
  const raw = firstNonEmptyArray(
    franchiseState?.standings,
    franchiseState?.standings?.teams,
    franchiseState?.league_standings,
    franchiseState?.team_standings,
    franchiseState?.analytics?.standings
  );

  if (Array.isArray(franchiseState?.standings)) return franchiseState.standings;
  if (Array.isArray(franchiseState?.standings?.teams)) return franchiseState.standings.teams;

  return raw;
}

function normalizeTeam(row, fallbackIndex = 0) {
  const wins = number(row?.wins ?? row?.w, 0);
  const losses = number(row?.losses ?? row?.l, 0);
  const otl = number(row?.otl ?? row?.ot ?? row?.overtime_losses, 0);
  const gp = number(row?.games_played ?? row?.gp, wins + losses + otl);
  const points = number(row?.points ?? row?.pts, wins * 2 + otl);
  const maxPoints = Math.max(1, gp * 2);

  return {
    id: normalizeId(row?.id || row?.team_id || row?.abbr || row?.abbreviation || fallbackIndex),
    name: row?.name || row?.team_name || row?.full_name || row?.city || `Team ${fallbackIndex + 1}`,
    city: row?.city || "",
    abbr: row?.abbr || row?.abbreviation || initials(row?.name || row?.team_name || `T${fallbackIndex + 1}`),
    conference: row?.conference || row?.conf || "—",
    division: row?.division || "—",
    wins,
    losses,
    otl,
    gp,
    points,
    pointPct: number(row?.point_pct ?? row?.points_pct ?? row?.pct, points / maxPoints),
    goalsFor: number(row?.goals_for ?? row?.gf, 0),
    goalsAgainst: number(row?.goals_against ?? row?.ga, 0),
    streak: row?.streak || row?.current_streak || "—",
    raw: row,
  };
}

function getUserTeamStanding(franchiseState) {
  const userTeamId = String(getTeamId(franchiseState) || "").toLowerCase();
  const userTeamName = String(getTeamName(franchiseState) || "").toLowerCase();
  const standings = collectStandings(franchiseState).map(normalizeTeam);

  const found =
    standings.find((t) => String(t.id || "").toLowerCase() === userTeamId) ||
    standings.find((t) => String(t.abbr || "").toLowerCase() === userTeamId) ||
    standings.find((t) => String(t.name || "").toLowerCase() === userTeamName);

  if (found) return found;

  const team = franchiseState?.user_team || franchiseState?.team || {};
  return normalizeTeam(
    {
      ...team,
      wins: team?.wins ?? franchiseState?.wins,
      losses: team?.losses ?? franchiseState?.losses,
      otl: team?.otl ?? franchiseState?.otl,
      points: team?.points ?? franchiseState?.points,
    },
    0
  );
}

function getDivisionRank(franchiseState) {
  const user = getUserTeamStanding(franchiseState);
  const standings = collectStandings(franchiseState).map(normalizeTeam);
  if (!standings.length) return "—";

  const sameDivision = standings
    .filter((t) => t.division && user.division && t.division === user.division)
    .sort((a, b) => b.points - a.points || b.pointPct - a.pointPct);

  if (!sameDivision.length) return "—";

  const idx = sameDivision.findIndex((t) => t.id === user.id || t.name === user.name || t.abbr === user.abbr);
  return idx >= 0 ? `${idx + 1}/${sameDivision.length}` : "—";
}

function getPlayoffRaceStatus(franchiseState) {
  const user = getUserTeamStanding(franchiseState);
  const standings = collectStandings(franchiseState).map(normalizeTeam);

  if (!standings.length) {
    if (user.pointPct >= 0.65) return "Comfortably In";
    if (user.pointPct >= 0.55) return "Playoff Mix";
    if (user.pointPct >= 0.48) return "Bubble";
    return "Chasing";
  }

  const sorted = [...standings].sort((a, b) => b.points - a.points || b.pointPct - a.pointPct);
  const idx = sorted.findIndex((t) => t.id === user.id || t.name === user.name || t.abbr === user.abbr);
  const rank = idx >= 0 ? idx + 1 : null;

  if (rank && rank <= 8) return "Playoff Spot";
  if (rank && rank <= 12) return "Bubble";
  if (rank && rank <= 18) return "Chasing";
  return "Long Shot";
}

function getSchedulePressure(franchiseState) {
  const schedule = collectSchedule(franchiseState);
  const currentRaw = getCurrentRawDate(franchiseState);
  const current = currentRaw ? new Date(currentRaw) : null;

  if (!schedule.length || !current || Number.isNaN(current.getTime())) {
    return {
      label: "Unknown",
      score: 45,
      gamesNext7: 0,
      backToBacks: 0,
      restDays: "—",
    };
  }

  const next7 = schedule.filter((g) => {
    const raw = g.date || g.calendar_iso || g.game_date;
    const d = new Date(raw);
    if (Number.isNaN(d.getTime())) return false;
    const diff = d.getTime() - current.getTime();
    return diff >= 0 && diff <= 7 * 24 * 60 * 60 * 1000 && !g.played && !g.completed;
  });

  const sorted = [...next7].sort((a, b) => dateValue(a.date || a.calendar_iso) - dateValue(b.date || b.calendar_iso));

  let backToBacks = 0;
  for (let i = 1; i < sorted.length; i += 1) {
    const a = new Date(sorted[i - 1].date || sorted[i - 1].calendar_iso);
    const b = new Date(sorted[i].date || sorted[i].calendar_iso);
    const diffDays = Math.round((b.getTime() - a.getTime()) / (1000 * 60 * 60 * 24));
    if (diffDays === 1) backToBacks += 1;
  }

  const score = clamp(next7.length * 13 + backToBacks * 18, 0, 100);

  return {
    label: score >= 80 ? "Brutal" : score >= 60 ? "Heavy" : score >= 35 ? "Normal" : "Light",
    score,
    gamesNext7: next7.length,
    backToBacks,
    restDays: sorted.length ? calculateDaysUntil(currentRaw, sorted[0].date || sorted[0].calendar_iso) : "—",
  };
}

/* ============================================================
   4. PLAYER COLLECTION / NORMALIZATION
   ============================================================ */

function collectPlayers(franchiseState) {
  const possible = [
    franchiseState?.players,
    franchiseState?.roster,
    franchiseState?.user_team?.players,
    franchiseState?.team?.players,
    franchiseState?.league_players,
    franchiseState?.player_stats,
    franchiseState?.skaters,
    franchiseState?.goalies,
    franchiseState?.stats?.players,
    franchiseState?.analytics?.players,
    asArray(franchiseState?.teams).flatMap((t) => asArray(t.players)),
    asArray(franchiseState?.league_teams).flatMap((t) => asArray(t.players)),
  ];

  return flattenDeep(possible).filter(Boolean);
}

function collectRosterPlayers(franchiseState) {
  const teamId = String(getTeamId(franchiseState) || "").toLowerCase();
  const teamName = String(getTeamName(franchiseState) || "").toLowerCase();

  const direct = firstNonEmptyArray(
    franchiseState?.roster,
    franchiseState?.user_team?.players,
    franchiseState?.team?.players
  );

  if (direct.length) return direct;

  return collectPlayers(franchiseState).filter((p) => {
    const pTeamId = String(p?.team_id || p?.team || p?.team_abbr || "").toLowerCase();
    const pTeamName = String(p?.team_name || "").toLowerCase();

    return pTeamId === teamId || pTeamName === teamName;
  });
}

function playerName(p) {
  return (
    p?.name ||
    p?.full_name ||
    p?.player_name ||
    `${p?.first_name || ""} ${p?.last_name || ""}`.trim() ||
    "Player"
  );
}

function playerId(p) {
  return normalizeId(
    p?.id ||
      p?.player_id ||
      p?.uid ||
      p?.slug ||
      playerName(p)
  );
}

function playerTeam(p) {
  return p?.team_abbr || p?.team || p?.team_id || p?.team_name || "—";
}

function playerTeamName(p) {
  return p?.team_name || p?.team_full_name || p?.team || "—";
}

function playerPos(p) {
  return p?.position || p?.pos || p?.primary_position || "—";
}

function playerAge(p) {
  return p?.age || p?.player_age || "—";
}

function playerOverall(p) {
  return clamp(p?.overall ?? p?.ovr ?? p?.rating ?? p?.ability ?? 70, 40, 99);
}

function playerPotential(p) {
  return clamp(p?.potential ?? p?.pot ?? p?.ceiling ?? playerOverall(p), 40, 99);
}

function playerMorale(p) {
  return clamp(p?.morale ?? p?.confidence ?? p?.player_morale ?? 55, 0, 100);
}

function playerFatigue(p) {
  return clamp(p?.fatigue ?? p?.tiredness ?? p?.energy_debt ?? 0, 0, 100);
}

function playerTradeValue(p) {
  return number(p?.trade_value ?? p?.value ?? p?.asset_value ?? playerOverall(p) * 0.8, 0);
}

function playerContractStatus(p) {
  const years =
    p?.contract_years_left ??
    p?.years_left ??
    p?.contract?.years_left ??
    p?.contract?.remaining_years;

  const cap = p?.cap_hit ?? p?.contract?.cap_hit ?? p?.salary ?? p?.aav;

  if (years !== undefined || cap !== undefined) {
    return `${years ?? "?"}Y · ${formatMoneyMillions(cap, "—")}`;
  }

  return p?.contract_status || p?.contract || "Contract —";
}

function playerStatsLine(p) {
  const pos = upper(playerPos(p));
  const gp = number(p?.games_played ?? p?.gp, 0);

  if (pos === "G" || pos === "GOALIE") {
    const wins = number(p?.wins ?? p?.w, 0);
    const losses = number(p?.losses ?? p?.l, 0);
    const svPct = p?.save_pct ?? p?.sv_pct ?? p?.savePercentage;
    const gaa = p?.gaa ?? p?.goals_against_average;

    return `${wins}-${losses} · ${svPct ? formatPct(svPct) : ".900"} SV% · ${gaa ? Number(gaa).toFixed(2) : "—"} GAA`;
  }

  const goals = number(p?.goals ?? p?.g, 0);
  const assists = number(p?.assists ?? p?.a, 0);
  const points = number(p?.points ?? p?.pts, goals + assists);

  return `${goals}G · ${assists}A · ${points}P · ${gp}GP`;
}

function normalizePlayer(p, fallbackIndex = 0) {
  const goals = number(p?.goals ?? p?.g, 0);
  const assists = number(p?.assists ?? p?.a, 0);
  const points = number(p?.points ?? p?.pts, goals + assists);
  const gp = number(p?.games_played ?? p?.gp, 0);
  const pos = playerPos(p);
  const morale = playerMorale(p);
  const fatigue = playerFatigue(p);
  const ovr = playerOverall(p);

  return {
    id: playerId(p) || `player-${fallbackIndex}`,
    name: playerName(p),
    shortName: shortName(playerName(p)),
    initials: initials(playerName(p)),
    pos,
    age: playerAge(p),
    team: playerTeam(p),
    teamName: playerTeamName(p),
    overall: ovr,
    potential: playerPotential(p),
    morale,
    fatigue,
    tradeValue: playerTradeValue(p),
    contract: playerContractStatus(p),
    goals,
    assists,
    points,
    gp,
    statsLine: playerStatsLine(p),
    injuryStatus: p?.injury_status || p?.status || "Healthy",
    isInjured: bool(p?.is_injured, false) || lower(p?.status).includes("injur") || lower(p?.injury_status).includes("injur"),
    role: p?.role || p?.line_role || p?.usage || "Roster Player",
    hand: p?.hand || p?.shoots || "—",
    recentSummary:
      p?.recent_summary ||
      p?.trend_summary ||
      p?.last_games_summary ||
      `${points} PTS in ${gp || "recent"} GP`,
    raw: p,
  };
}

function playerTrendingScore(p) {
  const points = number(p?.points ?? p?.pts, 0);
  const goals = number(p?.goals ?? p?.g, 0);
  const assists = number(p?.assists ?? p?.a, 0);
  const rating = number(p?.rating ?? p?.overall ?? p?.ovr, 0);
  const last = number(p?.last_5_points ?? p?.last5_points ?? p?.recent_points, 0);
  const morale = number(p?.morale, 50);
  const fatigue = number(p?.fatigue, 0);
  const tradeHeat = number(p?.trade_heat ?? p?.rumor_heat, 0);
  const hot = p?.hot || p?.is_hot || p?.streaking ? 10 : 0;
  const injuredPenalty = p?.is_injured || lower(p?.status).includes("injur") ? -8 : 0;

  return (
    points * 1.6 +
    goals * 1.2 +
    assists * 0.7 +
    rating * 0.08 +
    last * 3 +
    morale * 0.03 -
    fatigue * 0.05 +
    tradeHeat * 1.5 +
    hot +
    injuredPenalty
  );
}

function buildTrendingPlayers(franchiseState) {
  const players = collectPlayers(franchiseState);

  const normalized = players
    .map((p, idx) => {
      const base = normalizePlayer(p, idx);
      const pos = upper(base.pos);
      const svPct = p?.save_pct ?? p?.sv_pct ?? p?.savePercentage;

      const recent =
        p?.recent_summary ||
        p?.trend_summary ||
        (pos === "G" || pos === "GOALIE"
          ? `${svPct ? Number(svPct).toFixed(3) : ".900"} SV% recently`
          : base.recentSummary);

      const tag =
        p?.trend_label ||
        p?.status_label ||
        p?.storyline_tag ||
        (base.isInjured
          ? "INJURED"
          : p?.trade_heat || p?.rumor_heat
            ? "RUMOR"
            : base.points >= 60
              ? "HOT"
              : base.points >= 40
                ? "RISING"
                : p?.streaking
                  ? "STREAKING"
                  : "WATCH");

      return {
        ...base,
        recent,
        score: playerTrendingScore(p),
        tag,
      };
    })
    .filter((p) => p.name && p.name !== "Player")
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);

  return normalized;
}

function buildPlayerPressureList(franchiseState) {
  const players = collectRosterPlayers(franchiseState).map(normalizePlayer);

  return players
    .map((p) => {
      const score =
        (100 - p.morale) * 0.45 +
        p.fatigue * 0.25 +
        (p.isInjured ? 20 : 0) +
        (lower(p.role).includes("scratched") ? 15 : 0) +
        (lower(p.contract).includes("expir") ? 8 : 0);

      return {
        ...p,
        pressureScore: clamp(score, 0, 100),
        pressureLabel:
          score >= 75
            ? "Critical"
            : score >= 58
              ? "Hot Seat"
              : score >= 40
                ? "Watch"
                : "Stable",
      };
    })
    .sort((a, b) => b.pressureScore - a.pressureScore)
    .slice(0, 6);
}

/* ============================================================
   5. INJURY COLLECTION / NORMALIZATION
   ============================================================ */

function collectInjuries(franchiseState) {
  const possible = [
    franchiseState?.injuries,
    franchiseState?.injury_report,
    franchiseState?.injury_log,
    franchiseState?.injury_log_all,
    franchiseState?.medical?.injuries,
    franchiseState?.analytics?.injuries,
    franchiseState?.user_team?.injuries,
    franchiseState?.team?.injuries,
  ];

  return flattenDeep(possible).filter(Boolean);
}

function normalizeInjury(row, fallbackIndex = 0) {
  const player =
    row?.player ||
    row?.player_name ||
    row?.name ||
    row?.full_name ||
    row?.injured_player ||
    "Unknown Player";

  const gamesRemaining =
    row?.games_remaining ??
    row?.games_left ??
    row?.remaining_games ??
    row?.duration_games_remaining ??
    row?.days_remaining ??
    row?.remaining_days ??
    "—";

  const severity =
    row?.severity ||
    row?.tier ||
    row?.injury_tier ||
    row?.status ||
    (number(gamesRemaining, 0) >= 20
      ? "Severe"
      : number(gamesRemaining, 0) >= 8
        ? "Moderate"
        : "Minor");

  return {
    id: normalizeId(row?.id || row?.injury_id || `${player}-${fallbackIndex}`),
    player,
    team: row?.team || row?.team_abbr || row?.team_id || "—",
    position: row?.position || row?.pos || "—",
    injury: row?.injury || row?.injury_type || row?.body_part || row?.label || "Undisclosed injury",
    severity,
    status: row?.status || row?.availability || "Out",
    gamesRemaining,
    expectedReturn:
      row?.expected_return ||
      row?.return_date ||
      row?.estimated_return ||
      row?.eta ||
      "TBD",
    date:
      row?.date ||
      row?.calendar_iso ||
      row?.injury_date ||
      row?.created_at ||
      null,
    impact:
      row?.impact ||
      row?.effect_summary ||
      row?.team_impact ||
      "Lineup depth affected.",
    raw: row,
  };
}

function buildInjuryReport(franchiseState) {
  const injuries = collectInjuries(franchiseState)
    .map(normalizeInjury)
    .sort((a, b) => {
      const aSeverity = injurySeverityScore(a);
      const bSeverity = injurySeverityScore(b);
      if (bSeverity !== aSeverity) return bSeverity - aSeverity;
      return dateValue(b.date) - dateValue(a.date);
    });

  return injuries.slice(0, 8);
}

function injurySeverityScore(injury) {
  const text = lower(`${injury?.severity || ""} ${injury?.status || ""}`);
  const remaining = number(injury?.gamesRemaining, 0);

  if (text.includes("season") || text.includes("long") || text.includes("severe")) return 100;
  if (remaining >= 25) return 90;
  if (text.includes("major")) return 82;
  if (remaining >= 12) return 75;
  if (text.includes("moderate")) return 58;
  if (remaining >= 4) return 45;
  if (text.includes("minor")) return 25;
  return 35;
}

function injuryTone(injury) {
  const score = injurySeverityScore(injury);
  if (score >= 75) return "negative";
  if (score >= 45) return "warning";
  return "neutral";
}

function getInjuryCrisisScore(franchiseState) {
  const injuries = buildInjuryReport(franchiseState);
  const score = clamp(
    injuries.reduce((total, injury) => total + injurySeverityScore(injury) / 10, 0),
    0,
    100
  );

  return {
    score,
    label:
      score >= 75
        ? "Crisis"
        : score >= 50
          ? "Thin"
          : score >= 25
            ? "Manageable"
            : "Healthy",
    count: injuries.length,
  };
}

/* ============================================================
   6. STORYLINE CORE HELPERS
   ============================================================ */

function fmtDeltaMap(effects) {
  const entries = Object.entries(effects || {});
  if (!entries.length) return "";

  return entries
    .map(([k, v]) => `${k.replace(/_/g, " ")} ${Number(v) > 0 ? "+" : ""}${v}`)
    .join(" · ");
}

function getStoryPriority(ev) {
  const p = String(ev?.priority || ev?.severity || ev?.importance || "MEDIUM").toUpperCase();

  if (p.includes("CRITICAL")) return 5;
  if (p.includes("HIGH")) return 4;
  if (p.includes("MEDIUM")) return 3;
  if (p.includes("LOW")) return 2;
  if (p.includes("INFO")) return 1;

  const heat = number(ev?.heat ?? ev?.story_heat ?? ev?.urgency, 0);
  if (heat >= 90) return 5;
  if (heat >= 70) return 4;
  if (heat >= 45) return 3;
  if (heat >= 20) return 2;

  return 1;
}

function getStoryType(ev) {
  return String(ev?.type || ev?.kind || ev?.category || ev?.story_type || "storyline").toLowerCase();
}

function getHeadline(ev) {
  return ev?.headline || ev?.title || ev?.summary || ev?.name || "Storyline update";
}

function getStoryBody(ev) {
  return (
    ev?.description ||
    ev?.body ||
    ev?.detail ||
    ev?.details ||
    ev?.cause ||
    ev?.effect_summary ||
    ev?.message ||
    "A new storyline is developing around the league."
  );
}

function getStoryDate(ev) {
  return ev?.calendar_iso || ev?.date || ev?.created_at || ev?.timestamp || null;
}

function storyHasUserTeam(ev, franchiseState) {
  const teamId = String(getTeamId(franchiseState) || "").toLowerCase();
  const teamName = String(getTeamName(franchiseState) || "").toLowerCase();
  const teamAbbr = String(getTeamAbbr(franchiseState) || "").toLowerCase();

  const blob = lower(
    [
      ev?.team,
      ev?.team_id,
      ev?.team_name,
      ev?.team_abbr,
      ev?.franchise_team,
      ev?.headline,
      ev?.title,
      ev?.description,
      ev?.body,
      ev?.detail,
    ].join(" ")
  );

  return Boolean(
    blob.includes(teamId) ||
      blob.includes(teamName) ||
      blob.includes(teamAbbr)
  );
}

function getStoryTone(ev) {
  const type = getStoryType(ev);
  const headline = lower(getHeadline(ev));
  const body = lower(getStoryBody(ev));
  const effects = ev?.effects || {};
  const values = Object.values(effects).map(Number).filter(Number.isFinite);
  const total = values.reduce((a, b) => a + b, 0);
  const text = `${type} ${headline} ${body}`;

  if (
    total < 0 ||
    text.includes("injury") ||
    text.includes("injured") ||
    text.includes("drama") ||
    text.includes("tension") ||
    text.includes("frustrated") ||
    text.includes("concern") ||
    text.includes("losing") ||
    text.includes("lawsuit") ||
    text.includes("arrest") ||
    text.includes("suspended") ||
    text.includes("sideline") ||
    text.includes("crisis")
  ) {
    return "negative";
  }

  if (
    total > 0 ||
    text.includes("win") ||
    text.includes("streak") ||
    text.includes("milestone") ||
    text.includes("surge") ||
    text.includes("hot") ||
    text.includes("clinch") ||
    text.includes("extension") ||
    text.includes("award")
  ) {
    return "positive";
  }

  if (
    text.includes("trade") ||
    text.includes("rumor") ||
    text.includes("rumour") ||
    text.includes("market") ||
    text.includes("scout")
  ) {
    return "rumor";
  }

  if (
    text.includes("warning") ||
    text.includes("monitor") ||
    text.includes("question") ||
    text.includes("uncertain")
  ) {
    return "warning";
  }

  return "neutral";
}

function storyTag(ev) {
  const priority = String(ev?.priority || ev?.severity || "MEDIUM").toUpperCase();
  const type = getStoryType(ev);
  const headline = lower(getHeadline(ev));

  if (priority === "CRITICAL") return "CRISIS";
  if (priority === "HIGH") return "BREAKING";
  if (type.includes("rumor") || type.includes("rumour") || type.includes("trade") || headline.includes("rumor")) return "HOT";
  if (type.includes("injury") || headline.includes("injury") || headline.includes("sideline")) return "INJURY";
  if (type.includes("drama") || type.includes("morale") || headline.includes("frustrated")) return "DRAMA";
  if (type.includes("milestone")) return "MILESTONE";
  if (type.includes("streak")) return "STREAK";
  if (type.includes("league")) return "LEAGUE";
  return "DEVELOPING";
}

function getStorySource(ev) {
  const type = getStoryType(ev);
  const headline = lower(getHeadline(ev));

  if (ev?.source) return ev.source;
  if (ev?.reporter) return ev.reporter;
  if (type.includes("trade") || headline.includes("trade")) return "League Insider";
  if (type.includes("injury") || headline.includes("injury")) return "Team Medical Update";
  if (type.includes("drama") || headline.includes("frustrated")) return "Locker Room Source";
  if (type.includes("milestone")) return "League Desk";
  if (type.includes("waiver")) return "Transaction Wire";

  return "Hockey Operations Desk";
}

function getStoryCredibility(ev) {
  const raw = ev?.credibility ?? ev?.source_confidence ?? ev?.rumor_confidence;
  if (raw !== undefined) return clamp(raw, 0, 100);

  const type = getStoryType(ev);
  const headline = lower(getHeadline(ev));

  if (type.includes("official") || headline.includes("announced")) return 92;
  if (type.includes("injury")) return 82;
  if (type.includes("trade") || type.includes("rumor") || headline.includes("rumor")) return 55;
  if (type.includes("drama")) return 48;

  return 68;
}

function getStoryHeat(ev) {
  const raw = ev?.heat ?? ev?.story_heat ?? ev?.urgency ?? ev?.media_heat;
  if (raw !== undefined) return clamp(raw, 0, 100);

  const priority = getStoryPriority(ev);
  const tone = getStoryTone(ev);
  const effects = Object.values(ev?.effects || {}).map((x) => Math.abs(number(x, 0)));
  const effectHeat = clamp(sumNumbers(effects) * 8, 0, 40);

  let base = priority * 15;
  if (tone === "negative") base += 14;
  if (tone === "rumor") base += 10;
  if (tone === "positive") base += 4;

  return clamp(base + effectHeat, 0, 100);
}

function getStoryStatus(ev) {
  const explicit = ev?.status || ev?.story_status || ev?.resolution_status;
  if (explicit) return titleCase(explicit);

  if (ev?.resolved || ev?.is_resolved) return "Resolved";
  if (ev?.requires_action || ev?.action_required) return "Action Required";
  if (getStoryHeat(ev) >= 80) return "Escalating";
  if (getStoryHeat(ev) >= 55) return "Active";
  return "Developing";
}

function getImpactLabel(ev) {
  const effects = ev?.effects || {};
  const values = Object.values(effects).map(Number).filter(Number.isFinite);
  const total = values.reduce((a, b) => a + b, 0);

  if (total > 0) return { label: `+${total}`, tone: "positive", value: total };
  if (total < 0) return { label: `${total}`, tone: "negative", value: total };
  return { label: "—", tone: "neutral", value: 0 };
}

function getPrimaryImpactKey(ev) {
  const effects = asObject(ev?.effects);
  const entries = Object.entries(effects)
    .map(([key, value]) => [key, number(value, 0)])
    .filter(([, value]) => value !== 0)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  if (!entries.length) return "Team Morale";
  return titleCase(entries[0][0]);
}

function getStoryAffectedPlayers(ev, franchiseState) {
  const players = collectPlayers(franchiseState).map(normalizePlayer);

  const rawNames = firstNonEmptyArray(
    ev?.players,
    ev?.affected_players,
    ev?.player_names,
    ev?.involved_players
  );

  const namesFromRaw = rawNames
    .map((x) => (typeof x === "string" ? x : playerName(x)))
    .filter(Boolean);

  const blob = lower(
    [
      getHeadline(ev),
      getStoryBody(ev),
      ev?.player,
      ev?.player_name,
      ev?.name,
      ev?.primary_player,
      ev?.secondary_player,
      ...namesFromRaw,
    ].join(" ")
  );

  const matched = players.filter((p) => lower(blob).includes(lower(p.name)));

  if (matched.length) return matched.slice(0, 4);

  const directName =
    ev?.player_name ||
    ev?.player ||
    ev?.primary_player ||
    namesFromRaw[0];

  if (directName) {
    return [
      {
        id: normalizeId(directName),
        name: directName,
        shortName: shortName(directName),
        initials: initials(directName),
        pos: ev?.position || "—",
        age: "—",
        team: ev?.team || ev?.team_abbr || "—",
        teamName: ev?.team_name || "—",
        overall: 70,
        potential: 70,
        morale: 55,
        fatigue: 0,
        tradeValue: 0,
        contract: "—",
        goals: 0,
        assists: 0,
        points: 0,
        gp: 0,
        statsLine: "—",
        injuryStatus: "—",
        isInjured: false,
        role: "Involved Player",
        hand: "—",
        recentSummary: "Mentioned in storyline",
        raw: {},
      },
    ];
  }

  return [];
}

function getStoryAffectedTeams(ev, franchiseState) {
  const userTeam = {
    id: getTeamId(franchiseState),
    name: getTeamName(franchiseState),
    city: getTeamCity(franchiseState),
    abbr: getTeamAbbr(franchiseState),
  };

  const teams = collectTeams(franchiseState).map(normalizeTeam);

  const rawTeams = firstNonEmptyArray(
    ev?.teams,
    ev?.affected_teams,
    ev?.team_names,
    ev?.involved_teams
  );

  const namesFromRaw = rawTeams
    .map((x) => (typeof x === "string" ? x : x?.name || x?.team_name || x?.abbr))
    .filter(Boolean);

  const blob = lower(
    [
      getHeadline(ev),
      getStoryBody(ev),
      ev?.team,
      ev?.team_name,
      ev?.team_abbr,
      ev?.team_id,
      ...namesFromRaw,
    ].join(" ")
  );

  const matched = teams.filter((t) => {
    return (
      blob.includes(lower(t.name)) ||
      blob.includes(lower(t.abbr)) ||
      blob.includes(lower(t.id))
    );
  });

  if (matched.length) return matched.slice(0, 4);

  if (storyHasUserTeam(ev, franchiseState)) return [userTeam];

  if (ev?.team || ev?.team_name || ev?.team_abbr) {
    return [
      {
        id: ev?.team_id || ev?.team_abbr || ev?.team || "team",
        name: ev?.team_name || ev?.team || ev?.team_abbr || "Team",
        city: "",
        abbr: ev?.team_abbr || initials(ev?.team_name || ev?.team),
      },
    ];
  }

  return [];
}

function normalizeStory(ev, franchiseState, fallbackIndex = 0) {
  const id = normalizeId(
    ev?.id ||
      ev?.storyline_id ||
      ev?.event_id ||
      ev?.uid ||
      `${getHeadline(ev)}-${getStoryDate(ev) || fallbackIndex}`,
    `story-${fallbackIndex}`
  );

  const tone = getStoryTone(ev);
  const tag = storyTag(ev);
  const heat = getStoryHeat(ev);
  const priority = getStoryPriority(ev);
  const affectedPlayers = getStoryAffectedPlayers(ev, franchiseState);
  const affectedTeams = getStoryAffectedTeams(ev, franchiseState);
  const impact = getImpactLabel(ev);

  return {
    id,
    raw: ev,
    type: getStoryType(ev),
    tag,
    tone,
    priority,
    headline: getHeadline(ev),
    body: getStoryBody(ev),
    date: getStoryDate(ev),
    timeAgo: timeAgoFromRaw(getStoryDate(ev)),
    source: getStorySource(ev),
    credibility: getStoryCredibility(ev),
    heat,
    status: getStoryStatus(ev),
    cause: ev?.cause || ev?.reason || ev?.trigger || "",
    effectSummary: ev?.effect_summary || ev?.impact || ev?.result || "",
    effects: asObject(ev?.effects),
    impact,
    primaryImpact: getPrimaryImpactKey(ev),
    affectedPlayers,
    affectedTeams,
    isUserTeam: storyHasUserTeam(ev, franchiseState),
    requiresAction: bool(ev?.requires_action ?? ev?.action_required, false),
    resolved: bool(ev?.resolved ?? ev?.is_resolved, false),
    expiresOn: ev?.expires_on || ev?.deadline || ev?.expires_at || null,
    followUp:
      ev?.follow_up ||
      ev?.next_step ||
      ev?.future_hook ||
      "",
  };
}

function sortStories(a, b) {
  if (a.requiresAction !== b.requiresAction) return a.requiresAction ? -1 : 1;
  if (b.priority !== a.priority) return b.priority - a.priority;
  if (b.heat !== a.heat) return b.heat - a.heat;
  return dateValue(b.date) - dateValue(a.date);
}

function collectStorylines(franchiseState) {
  const rows = [
    ...asArray(franchiseState?.storyline_events),
    ...asArray(franchiseState?.league_news_events),
    ...asArray(franchiseState?.news_events),
  ];

  const unique = uniqueBy(rows, (ev, idx) => {
    return normalizeId(
      ev?.id ||
        ev?.storyline_id ||
        ev?.event_id ||
        `${getHeadline(ev)}-${getStoryDate(ev) || idx}`
    );
  });

  const normalized = unique
    .map((ev, idx) => normalizeStory(ev, franchiseState, idx))
    .sort(sortStories)
    .slice(0, 160);

  return normalized.length ? normalized : buildFallbackStorylines(franchiseState).map((ev, idx) => normalizeStory(ev, franchiseState, idx));
}

function buildFallbackStorylines(franchiseState) {
  const teamName = getTeamName(franchiseState);

  return [
    {
      id: "fallback-story-1",
      type: "team_drama",
      priority: "HIGH",
      headline: `${teamName} dressing room searching for answers`,
      description:
        "Team insiders believe the next few games could shape the tone of the room as pressure rises around the organization.",
      cause: "Recent results and player usage have created pressure inside the room.",
      effect_summary: "Team morale could swing based on the next result.",
      effects: { team_morale: -4 },
      calendar_iso: franchiseState?.calendar_iso || franchiseState?.date,
      requires_action: true,
    },
    {
      id: "fallback-story-2",
      type: "rumor",
      priority: "MEDIUM",
      headline: "League executives monitoring the trade market",
      description:
        "Front offices around the league are starting to check prices as the standings picture becomes clearer.",
      effect_summary: "Trade market activity is increasing.",
      effects: { trade_market_heat: 3 },
      calendar_iso: franchiseState?.calendar_iso || franchiseState?.date,
    },
    {
      id: "fallback-story-3",
      type: "league_news",
      priority: "LOW",
      headline: "Scouts preparing updated draft reports",
      description:
        "Several junior players are beginning to separate themselves as teams update their internal draft boards.",
      effect_summary: "Draft stock movement expected.",
      effects: { scouting_clarity: 2 },
      calendar_iso: franchiseState?.calendar_iso || franchiseState?.date,
    },
  ];
}

/* ============================================================
   7. STORY GROUPING / DASHBOARD BUILDERS
   ============================================================ */

function isRumorStory(story) {
  const text = lower(`${story.type} ${story.headline} ${story.body}`);
  return (
    text.includes("rumor") ||
    text.includes("rumour") ||
    text.includes("trade") ||
    text.includes("market") ||
    text.includes("scout") ||
    text.includes("available")
  );
}

function isInjuryStory(story) {
  const text = lower(`${story.type} ${story.headline} ${story.body} ${story.tag}`);
  return (
    text.includes("injury") ||
    text.includes("injured") ||
    text.includes("sideline") ||
    text.includes("out ") ||
    text.includes("medical")
  );
}

function isDramaStory(story) {
  const text = lower(`${story.type} ${story.headline} ${story.body} ${story.tag}`);
  return (
    story.tone === "negative" ||
    text.includes("drama") ||
    text.includes("morale") ||
    text.includes("tension") ||
    text.includes("frustrated") ||
    text.includes("argument") ||
    text.includes("suspended") ||
    text.includes("discipline")
  );
}

function isMilestoneStory(story) {
  const text = lower(`${story.type} ${story.headline} ${story.body} ${story.tag}`);
  return (
    text.includes("milestone") ||
    text.includes("record") ||
    text.includes("first career") ||
    text.includes("hat trick") ||
    text.includes("shutout") ||
    text.includes("award")
  );
}

function splitStorylines(storylines) {
  const leagueBuzz = [];
  const rumorMill = [];
  const teamDrama = [];
  const injuryDesk = [];
  const milestones = [];
  const userTeam = [];
  const headlines = [];
  const actionRequired = [];

  for (const story of asArray(storylines)) {
    headlines.push(story);

    if (story.requiresAction) actionRequired.push(story);
    if (story.isUserTeam) userTeam.push(story);

    if (isInjuryStory(story)) {
      injuryDesk.push(story);
    } else if (isRumorStory(story)) {
      rumorMill.push(story);
    } else if (isDramaStory(story)) {
      teamDrama.push(story);
    } else if (isMilestoneStory(story)) {
      milestones.push(story);
    } else {
      leagueBuzz.push(story);
    }
  }

  return {
    leagueBuzz: leagueBuzz.slice(0, 5),
    rumorMill: rumorMill.slice(0, 5),
    teamDrama: teamDrama.slice(0, 5),
    injuryDesk: injuryDesk.slice(0, 5),
    milestones: milestones.slice(0, 5),
    userTeam: userTeam.slice(0, 8),
    actionRequired: actionRequired.slice(0, 5),
    headlines: headlines.slice(0, 12),
  };
}

function buildMoraleTrend(franchiseState, morale) {
  const raw =
    franchiseState?.morale_trend ||
    franchiseState?.team_morale_trend ||
    franchiseState?.analytics?.morale_trend ||
    franchiseState?.history?.morale;

  if (Array.isArray(raw) && raw.length) {
    return raw.slice(-14).map((x) => number(x?.value ?? x?.morale ?? x, morale));
  }

  return [
    clamp(morale + 18, 0, 100),
    clamp(morale + 11, 0, 100),
    clamp(morale + 5, 0, 100),
    clamp(morale + 2, 0, 100),
    clamp(morale + 4, 0, 100),
    clamp(morale - 3, 0, 100),
    clamp(morale - 12, 0, 100),
    clamp(morale - 6, 0, 100),
    clamp(morale, 0, 100),
    clamp(morale - 2, 0, 100),
    clamp(morale - 4, 0, 100),
    clamp(morale - 8, 0, 100),
    clamp(morale - 5, 0, 100),
    clamp(morale, 0, 100),
  ];
}

function buildChemistryTrend(franchiseState, chemistry) {
  const raw =
    franchiseState?.chemistry_trend ||
    franchiseState?.team_chemistry_trend ||
    franchiseState?.analytics?.chemistry_trend ||
    franchiseState?.history?.chemistry;

  if (Array.isArray(raw) && raw.length) {
    return raw.slice(-14).map((x) => number(x?.value ?? x?.chemistry ?? x, chemistry));
  }

  return [
    clamp(chemistry - 8, 0, 100),
    clamp(chemistry - 6, 0, 100),
    clamp(chemistry - 4, 0, 100),
    clamp(chemistry - 5, 0, 100),
    clamp(chemistry - 2, 0, 100),
    clamp(chemistry, 0, 100),
    clamp(chemistry + 2, 0, 100),
    clamp(chemistry + 1, 0, 100),
    clamp(chemistry + 3, 0, 100),
    clamp(chemistry + 4, 0, 100),
    clamp(chemistry + 2, 0, 100),
    clamp(chemistry, 0, 100),
    clamp(chemistry - 1, 0, 100),
    clamp(chemistry, 0, 100),
  ];
}

function buildKeyFactors(storylines, franchiseState) {
  const morale = getMorale(franchiseState);
  const factors = [];

  for (const story of asArray(storylines).slice(0, 12)) {
    const effects = story?.effects || {};
    for (const [key, value] of Object.entries(effects)) {
      const n = Number(value);
      if (!Number.isFinite(n) || n === 0) continue;

      factors.push({
        label: titleCase(key),
        value: n,
        story: story.headline,
        tone: n >= 0 ? "positive" : "negative",
      });
    }
  }

  const pressure = getSchedulePressure(franchiseState);
  if (pressure.score >= 60) {
    factors.push({
      label: "Schedule Pressure",
      value: -Math.round(pressure.score / 12),
      story: `${pressure.gamesNext7} games in next 7 days`,
      tone: "negative",
    });
  }

  const injury = getInjuryCrisisScore(franchiseState);
  if (injury.count > 0) {
    factors.push({
      label: "Injury Concerns",
      value: -Math.max(1, Math.round(injury.score / 12)),
      story: `${injury.count} active injuries`,
      tone: "negative",
    });
  }

  if (!factors.length) {
    factors.push(
      {
        label: "Recent Results",
        value: morale >= 60 ? 4 : -6,
        story: "Recent form is affecting the room.",
        tone: morale >= 60 ? "positive" : "negative",
      },
      {
        label: "Injury Concerns",
        value: morale >= 70 ? -1 : -5,
        story: "Roster health is being monitored.",
        tone: "negative",
      },
      {
        label: "Practice Intensity",
        value: 3,
        story: "Coaching staff believes practices have stabilized the group.",
        tone: "positive",
      },
      {
        label: "Room Confidence",
        value: morale >= 60 ? 5 : -4,
        story: "Players are reacting to the current team direction.",
        tone: morale >= 60 ? "positive" : "negative",
      }
    );
  }

  return factors
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);
}

function buildConsequenceSnapshot(storylines, franchiseState) {
  const morale = getMorale(franchiseState);
  const chemistry = getChemistry(franchiseState);
  const fanConfidence = getFanConfidence(franchiseState);
  const gmRating = getGmRating(franchiseState);
  const ownerPatience = getOwnerPatience(franchiseState);
  const mediaPressure = getMediaPressure(franchiseState);
  const roomTension = getRoomTension(franchiseState);
  const injuryCrisis = getInjuryCrisisScore(franchiseState);
  const schedulePressure = getSchedulePressure(franchiseState);

  const activeStories = asArray(storylines);
  const actionCount = activeStories.filter((s) => s.requiresAction).length;
  const negativeCount = activeStories.filter((s) => s.tone === "negative").length;
  const rumorCount = activeStories.filter((s) => s.tone === "rumor").length;
  const heatAvg = Math.round(averageNumbers(activeStories.map((s) => s.heat), 0));

  return {
    morale,
    chemistry,
    fanConfidence,
    gmRating,
    ownerPatience,
    mediaPressure,
    roomTension,
    injuryCrisis,
    schedulePressure,
    actionCount,
    negativeCount,
    rumorCount,
    heatAvg,
    crisisIndex: clamp(
      (100 - morale) * 0.22 +
        (100 - chemistry) * 0.14 +
        mediaPressure * 0.16 +
        roomTension * 0.14 +
        injuryCrisis.score * 0.18 +
        schedulePressure.score * 0.1 +
        actionCount * 5 +
        negativeCount * 2,
      0,
      100
    ),
  };
}

/* ============================================================
   END OF CHUNK 1
   Next chunk starts with:
   - Choices normalization
   - GM decision panels
   - mini chart components
   - topbar/hero components
   - story cards
   ============================================================ */
/* ============================================================
   8. CHOICE / GM DECISION NORMALIZATION
   ============================================================ */

   function getChoiceRows(franchiseState) {
    return asArray(franchiseState?.storyline_choices);
  }
  
  function normalizeChoiceOption(option, fallbackIndex = 0) {
    const id = normalizeId(
      option?.id ||
        option?.option_id ||
        option?.choice_id ||
        option?.key ||
        `option-${fallbackIndex}`,
      `option-${fallbackIndex}`
    );
  
    const effects = asObject(option?.effects);
    const effectSummary =
      option?.effect_summary ||
      option?.impact ||
      option?.result ||
      option?.summary ||
      fmtDeltaMap(effects);
  
    const risk =
      option?.risk ??
      option?.risk_score ??
      option?.volatility ??
      option?.failure_chance ??
      null;
  
    const riskScore = risk === null || risk === undefined ? estimateChoiceRisk(option) : clamp(risk, 0, 100);
  
    return {
      id,
      label:
        option?.label ||
        option?.title ||
        option?.name ||
        option?.text ||
        `Option ${fallbackIndex + 1}`,
      description:
        option?.description ||
        option?.body ||
        option?.detail ||
        option?.explanation ||
        effectSummary ||
        "This response will shape how the storyline develops.",
      effectSummary,
      effects,
      riskScore,
      riskLabel: riskLabel(riskScore),
      tone: choiceTone(effects, riskScore),
      unavailable: bool(option?.unavailable ?? option?.disabled, false),
      unavailableReason:
        option?.unavailable_reason ||
        option?.disabled_reason ||
        option?.requirement_failed ||
        "",
      raw: option,
    };
  }
  
  function normalizeChoiceRow(row, fallbackIndex = 0) {
    const storyId = normalizeId(
      row?.storyline_id ||
        row?.story_id ||
        row?.decision_id ||
        row?.event_id ||
        row?.id ||
        `choice-row-${fallbackIndex}`
    );
  
    const options = asArray(row?.action_options || row?.options || row?.choices)
      .map(normalizeChoiceOption)
      .filter((opt) => opt.id);
  
    return {
      storylineId: storyId,
      title: row?.title || row?.headline || row?.decision_title || "",
      description:
        row?.description ||
        row?.body ||
        row?.decision_body ||
        row?.prompt ||
        "",
      deadline:
        row?.deadline ||
        row?.expires_on ||
        row?.expires_at ||
        row?.must_respond_by ||
        null,
      urgency: clamp(row?.urgency ?? row?.heat ?? row?.priority_score ?? 50, 0, 100),
      options,
      raw: row,
    };
  }
  
  function buildChoicesByStoryId(franchiseState) {
    const map = new Map();
  
    getChoiceRows(franchiseState).forEach((row, idx) => {
      const normalized = normalizeChoiceRow(row, idx);
      if (normalized.storylineId) {
        map.set(String(normalized.storylineId), normalized);
      }
    });
  
    return map;
  }
  
  function estimateChoiceRisk(option) {
    const effects = asObject(option?.effects);
    const values = Object.values(effects).map(Number).filter(Number.isFinite);
    const negative = values.filter((v) => v < 0).reduce((a, b) => a + Math.abs(b), 0);
    const positive = values.filter((v) => v > 0).reduce((a, b) => a + b, 0);
  
    let risk = negative * 8;
  
    const label = lower(`${option?.label || ""} ${option?.description || ""} ${option?.effect_summary || ""}`);
  
    if (label.includes("aggressive")) risk += 20;
    if (label.includes("public")) risk += 12;
    if (label.includes("discipline")) risk += 14;
    if (label.includes("bench")) risk += 16;
    if (label.includes("trade")) risk += 18;
    if (label.includes("ignore")) risk += 22;
    if (label.includes("private")) risk -= 6;
    if (label.includes("support")) risk -= 8;
    if (positive > negative) risk -= 4;
  
    return clamp(risk, 5, 95);
  }
  
  function riskLabel(score) {
    const v = clamp(score, 0, 100);
    if (v >= 82) return "Extreme Risk";
    if (v >= 65) return "High Risk";
    if (v >= 45) return "Medium Risk";
    if (v >= 25) return "Low Risk";
    return "Safe";
  }
  
  function choiceTone(effects, riskScore) {
    const values = Object.values(effects || {}).map(Number).filter(Number.isFinite);
    const total = values.reduce((a, b) => a + b, 0);
  
    if (riskScore >= 72) return "negative";
    if (total > 0 && riskScore < 55) return "positive";
    if (total < 0) return "negative";
    if (riskScore >= 45) return "warning";
    return "neutral";
  }
  
  function choiceCategory(option) {
    const text = lower(`${option?.label || ""} ${option?.description || ""}`);
  
    if (text.includes("press") || text.includes("public") || text.includes("media")) return "Media";
    if (text.includes("trade") || text.includes("waiver") || text.includes("call up") || text.includes("call-up")) return "Roster";
    if (text.includes("bench") || text.includes("scratch") || text.includes("line")) return "Lineup";
    if (text.includes("support") || text.includes("private") || text.includes("talk")) return "Room";
    if (text.includes("discipline") || text.includes("suspend")) return "Discipline";
  
    return "GM Call";
  }
  
  function hasChoicesForStory(story, choicesByStoryId) {
    if (!story) return false;
    const row = choicesByStoryId.get(String(story.id));
    return Boolean(row?.options?.length);
  }
  
  function getChoicesForStory(story, choicesByStoryId) {
    if (!story) return null;
    return choicesByStoryId.get(String(story.id)) || null;
  }
  
  function buildDecisionRows(storylines, choicesByStoryId) {
    return asArray(storylines)
      .map((story) => {
        const choices = getChoicesForStory(story, choicesByStoryId);
        if (!choices?.options?.length) return null;
  
        return {
          story,
          choices,
          urgencyScore: Math.max(story.heat || 0, choices.urgency || 0),
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.urgencyScore - a.urgencyScore)
      .slice(0, 6);
  }
  
  /* ============================================================
     9. VISUAL COMPONENT HELPERS
     ============================================================ */
  
  function toneClass(tone) {
    const t = lower(tone);
    if (t.includes("positive")) return "positive";
    if (t.includes("negative")) return "negative";
    if (t.includes("rumor")) return "rumor";
    if (t.includes("warning")) return "warning";
    return "neutral";
  }
  
  function toneSymbol(tone) {
    const t = toneClass(tone);
    if (t === "positive") return "↑";
    if (t === "negative") return "!";
    if (t === "rumor") return "◆";
    if (t === "warning") return "△";
    return "◇";
  }
  
  function storyIconSymbol(story) {
    const text = lower(`${story?.type || ""} ${story?.headline || ""} ${story?.tag || ""}`);
  
    if (text.includes("trade") || text.includes("rumor") || text.includes("market")) return "◆";
    if (text.includes("injury") || text.includes("sideline") || text.includes("medical")) return "+";
    if (text.includes("drama") || text.includes("morale") || text.includes("frustrated")) return "!";
    if (text.includes("milestone") || text.includes("record") || text.includes("award")) return "★";
    if (text.includes("streak") || text.includes("hot")) return "↑";
    if (text.includes("waiver")) return "↕";
    if (text.includes("draft") || text.includes("scout")) return "⌕";
    if (text.includes("suspension") || text.includes("discipline")) return "⚑";
    return toneSymbol(story?.tone);
  }
  
  function MeterRing({ value, size = 48, label, subLabel, tone = "neutral" }) {
    const v = clamp(value, 0, 100);
    const angle = Math.round((v / 100) * 360);
  
    return (
      <div className={`story-meter story-meter--${toneClass(tone)}`}>
        <div
          className="story-meter-ring"
          style={{
            width: size,
            height: size,
            background: `conic-gradient(var(--meter-color) ${angle}deg, rgba(255,255,255,.08) ${angle}deg)`,
          }}
        >
          <div className="story-meter-inner">{Math.round(v)}</div>
        </div>
        <div className="story-meter-text">
          <div>{label}</div>
          <span>{subLabel}</span>
        </div>
      </div>
    );
  }
  
  function StatChip({ label, value, tone = "neutral", sub }) {
    return (
      <div className={`stat-chip stat-chip--${toneClass(tone)}`}>
        <div className="stat-chip-label">{label}</div>
        <div className="stat-chip-value">{value}</div>
        {sub ? <div className="stat-chip-sub">{sub}</div> : null}
      </div>
    );
  }
  
  function MiniBadge({ children, tone = "neutral" }) {
    return <span className={`mini-badge mini-badge--${toneClass(tone)}`}>{children}</span>;
  }
  
  function HeatBar({ value, label = "Heat", tone = "neutral" }) {
    const v = clamp(value, 0, 100);
  
    return (
      <div className={`heatbar heatbar--${toneClass(tone)}`}>
        <div className="heatbar-top">
          <span>{label}</span>
          <strong>{Math.round(v)}</strong>
        </div>
        <div className="heatbar-track">
          <div className="heatbar-fill" style={{ width: `${v}%` }} />
        </div>
      </div>
    );
  }
  
  function ImpactPill({ value, label }) {
    const n = number(value, 0);
    const tone = n > 0 ? "positive" : n < 0 ? "negative" : "neutral";
  
    return (
      <div className={`impact-pill impact-pill--${tone}`}>
        <span>{label}</span>
        <strong>{n > 0 ? "+" : ""}{n}</strong>
      </div>
    );
  }
  
  function StoryIcon({ story, type, tone }) {
    const computedTone = tone || story?.tone || getStoryTone(story?.raw || story || {});
    const symbol = story ? storyIconSymbol(story) : storyIconSymbol({ type, tone: computedTone });
  
    return (
      <div className={`story-icon story-icon--${toneClass(computedTone)}`}>
        {symbol}
      </div>
    );
  }
  
  function PlayerAvatar({ name, size = "md", status }) {
    return (
      <div className={`player-avatar player-avatar--${size} ${status ? `player-avatar--${toneClass(status)}` : ""}`}>
        <span>{initials(name)}</span>
      </div>
    );
  }
  
  function TeamShield({ teamName, abbr, size = "md" }) {
    return (
      <div className={`team-shield team-shield--${size}`}>
        <span>{abbr || initials(teamName)}</span>
      </div>
    );
  }
  
  function EmptyState({ title = "Nothing here yet.", detail }) {
    return (
      <div className="empty-state">
        <div className="empty-state-mark">◇</div>
        <div className="empty-state-title">{title}</div>
        {detail ? <div className="empty-state-detail">{detail}</div> : null}
      </div>
    );
  }
  
  /* ============================================================
     10. CHART COMPONENTS
     ============================================================ */
  
  function MiniLineChart({ points, tone = "gold", height = 150 }) {
    const safePoints = asArray(points).length
      ? points.map((x) => clamp(x, 0, 100))
      : [72, 68, 63, 61, 62, 54, 46, 52, 57, 55, 53, 50];
  
    const width = 360;
    const max = 100;
    const min = 0;
  
    const coords = safePoints.map((p, idx) => {
      const x = (idx / Math.max(1, safePoints.length - 1)) * width;
      const y = height - ((clamp(p, min, max) - min) / (max - min)) * height;
      return [x, y];
    });
  
    const path = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
  
    return (
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={`mini-line-chart mini-line-chart--${tone}`}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id={`chartFill-${tone}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,190,54,.35)" />
            <stop offset="100%" stopColor="rgba(255,190,54,0)" />
          </linearGradient>
        </defs>
  
        {[0, 25, 50, 75, 100].map((v) => {
          const y = height - (v / 100) * height;
          return <line key={v} x1="0" x2={width} y1={y} y2={y} className="chart-grid" />;
        })}
  
        {coords.length ? (
          <>
            <path
              d={`${path} L${width},${height} L0,${height} Z`}
              className="chart-area"
              opacity="0.55"
            />
            <path d={path} className="chart-line" />
            {coords.map(([x, y], idx) => (
              <circle
                key={idx}
                cx={x}
                cy={y}
                r={idx === coords.length - 1 ? 5 : 3.4}
                className="chart-dot"
              />
            ))}
          </>
        ) : null}
      </svg>
    );
  }
  
  function SparkBars({ values, tone = "neutral" }) {
    const safeValues = asArray(values).length ? values : [30, 45, 62, 55, 71, 66, 75, 58];
    const max = Math.max(1, ...safeValues.map((x) => Math.abs(number(x, 0))));
  
    return (
      <div className={`spark-bars spark-bars--${toneClass(tone)}`}>
        {safeValues.map((value, idx) => {
          const h = Math.max(8, (Math.abs(number(value, 0)) / max) * 100);
          return (
            <div key={idx} className="spark-bar-wrap">
              <div className="spark-bar" style={{ height: `${h}%` }} />
            </div>
          );
        })}
      </div>
    );
  }
  
  function RadialGauge({ value, label, subLabel, tone = "neutral" }) {
    const v = clamp(value, 0, 100);
    const angle = Math.round((v / 100) * 360);
  
    return (
      <div className={`radial-gauge radial-gauge--${toneClass(tone)}`}>
        <div
          className="radial-gauge-ring"
          style={{
            background: `conic-gradient(var(--gauge-color) ${angle}deg, rgba(255,255,255,.08) ${angle}deg)`,
          }}
        >
          <div className="radial-gauge-inner">
            <strong>{Math.round(v)}</strong>
            <span>{label}</span>
          </div>
        </div>
        {subLabel ? <div className="radial-gauge-sub">{subLabel}</div> : null}
      </div>
    );
  }
  
  /* ============================================================
     11. LAYOUT COMPONENTS
     ============================================================ */
  
  function SectionShell({
    title,
    subtitle,
    actionLabel,
    onAction,
    children,
    className = "",
    badge,
    badgeTone = "neutral",
  }) {
    return (
      <section className={`section-shell ${className}`}>
        <div className="section-title-row">
          <div>
            <h3>{title}</h3>
            {subtitle ? <p>{subtitle}</p> : null}
          </div>
  
          <div className="section-title-actions">
            {badge ? <MiniBadge tone={badgeTone}>{badge}</MiniBadge> : null}
            {actionLabel ? (
              <button type="button" onClick={onAction} className="tiny-link">
                {actionLabel}
              </button>
            ) : null}
          </div>
        </div>
  
        {children}
      </section>
    );
  }
  
  function TopbarCell({ icon, label, value, sub, children }) {
    return (
      <div className="topbar-cell">
        {icon ? <div className="topbar-icon">{icon}</div> : null}
        {children || (
          <div className="topbar-copy">
            <div className="topbar-label">{label}</div>
            <div className="topbar-value">{value}</div>
            {sub ? <div className="topbar-label">{sub}</div> : null}
          </div>
        )}
      </div>
    );
  }
  
  function StoryTopbar({
    franchiseState,
    gmRating,
    chemistry,
    fanConfidence,
    morale,
    setScreen,
  }) {
    const teamName = getTeamName(franchiseState);
    const teamCity = getTeamCity(franchiseState);
    const teamAbbr = getTeamAbbr(franchiseState);
    const gmName = getGMName(franchiseState);
    const dateLabel = getDateLabel(franchiseState);
    const weekLabel = getWeekLabel(franchiseState);
    const nextGame = getNextGame(franchiseState);
  
    return (
      <header className="story-topbar">
        <div className="franchise-brand">
          <TeamShield teamName={teamName} abbr={teamAbbr} size="lg" />
          <div style={{ minWidth: 0 }}>
            <div className="brand-kicker">{teamCity || "Franchise"}</div>
            <div className="brand-title">{teamName}</div>
            <div className="brand-sub">{gmName}</div>
          </div>
        </div>
  
        <TopbarCell>
          <MeterRing
            value={gmRating}
            label="GM Rating"
            subLabel={gmRating >= 80 ? "Elite" : gmRating >= 65 ? "Solid" : "Under Fire"}
            tone={valueTone(gmRating)}
          />
        </TopbarCell>
  
        <TopbarCell>
          <MeterRing
            value={chemistry}
            label="Team Chemistry"
            subLabel={chemistryLabel(chemistry)}
            tone={valueTone(chemistry)}
          />
        </TopbarCell>
  
        <TopbarCell>
          <MeterRing
            value={fanConfidence}
            label="Fan Confidence"
            subLabel={confidenceLabel(fanConfidence)}
            tone={valueTone(fanConfidence)}
          />
        </TopbarCell>
  
        <TopbarCell
          icon="⌂"
          label={nextGame.label}
          value={`${nextGame.days} Days`}
          sub={`${nextGame.homeAway !== "—" ? nextGame.homeAway + " vs. " : "vs. "}${nextGame.opponent}`}
        />
  
        <TopbarCell
          icon="▦"
          label={dateLabel.sub}
          value={dateLabel.main}
          sub={weekLabel}
        />
  
        <div className="topbar-cell topbar-cell--continue">
          <button type="button" className="continue-btn" onClick={() => setScreen?.(SCREENS.CALENDAR)}>
            <span>
              Continue
              <small>Advance Day</small>
            </span>
            <span className="continue-arrow">»</span>
          </button>
        </div>
      </header>
    );
  }
  
  function PageHeader({ setScreen, activeTab, setActiveTab }) {
    const tabs = [
      { id: "overview", label: "Overview" },
      { id: "league", label: "League" },
      { id: "team", label: "Team" },
      { id: "injuries", label: "Injuries" },
      { id: "decisions", label: "Decisions" },
    ];
  
    return (
      <div className="story-header">
        <div>
          <h1 className="page-title">Storylines</h1>
          <div className="page-subtitle">Around the league. Behind the scenes. Every angle.</div>
  
          <div className="story-tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`story-tab ${activeTab === tab.id ? "story-tab--active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
  
        <div className="header-actions">
          <button type="button" className="nav-btn" onClick={() => setScreen?.(SCREENS.CALENDAR)}>
            Calendar
          </button>
          <button type="button" className="nav-btn" onClick={() => setScreen?.(SCREENS.HUB)}>
            Hub
          </button>
        </div>
      </div>
    );
  }
  
  /* ============================================================
     12. STORY CARD COMPONENTS
     ============================================================ */
  
  function StoryMetaLine({ story }) {
    return (
      <div className="story-meta-line">
        <span>{story.source}</span>
        <span>Credibility {Math.round(story.credibility)}%</span>
        <span>{story.timeAgo}</span>
        <span>{story.status}</span>
      </div>
    );
  }
  
  function StoryEffectList({ effects, limit = 4 }) {
    const entries = Object.entries(effects || {})
      .map(([key, value]) => ({
        key,
        label: titleCase(key),
        value: number(value, 0),
      }))
      .filter((x) => x.value !== 0)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, limit);
  
    if (!entries.length) {
      return (
        <div className="effect-empty">
          No direct sim effect listed yet.
        </div>
      );
    }
  
    return (
      <div className="effect-list">
        {entries.map((entry) => (
          <ImpactPill
            key={entry.key}
            label={entry.label}
            value={entry.value}
          />
        ))}
      </div>
    );
  }
  
  function AffectedPlayersRow({ players }) {
    const safePlayers = asArray(players).slice(0, 4);
  
    if (!safePlayers.length) return null;
  
    return (
      <div className="affected-row">
        <div className="affected-label">Players</div>
        <div className="affected-stack">
          {safePlayers.map((p) => (
            <div key={p.id || p.name} className="affected-player-pill">
              <PlayerAvatar name={p.name} size="xs" status={p.isInjured ? "negative" : "neutral"} />
              <span>{p.shortName || shortName(p.name)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  function AffectedTeamsRow({ teams }) {
    const safeTeams = asArray(teams).slice(0, 4);
  
    if (!safeTeams.length) return null;
  
    return (
      <div className="affected-row">
        <div className="affected-label">Teams</div>
        <div className="affected-stack">
          {safeTeams.map((team) => (
            <div key={team.id || team.abbr || team.name} className="affected-team-pill">
              <TeamShield teamName={team.name} abbr={team.abbr} size="xs" />
              <span>{team.abbr || initials(team.name)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  function CompactStoryRow({ story, selected, onSelect, rightSlot }) {
    return (
      <button
        type="button"
        className={`compact-story-row ${selected ? "compact-story-row--selected" : ""}`}
        onClick={() => onSelect?.(story.id)}
      >
        <StoryIcon story={story} />
  
        <div className="compact-story-main">
          <div className={`story-row-tag story-row-tag--${toneClass(story.tone)}`}>
            {story.tag}
          </div>
          <div className="compact-story-headline">{story.headline}</div>
          <div className="compact-story-body">{story.body}</div>
        </div>
  
        <div className="compact-story-side">
          {rightSlot || (
            <>
              <span>{story.timeAgo}</span>
              {story.impact.label !== "—" ? (
                <b className={`impact-${story.impact.tone}`}>{story.impact.label}</b>
              ) : null}
            </>
          )}
        </div>
      </button>
    );
  }
  
  function LargeStoryCard({ story, selected, onSelect }) {
    if (!story) return null;
  
    return (
      <button
        type="button"
        className={`large-story-card large-story-card--${toneClass(story.tone)} ${
          selected ? "large-story-card--selected" : ""
        }`}
        onClick={() => onSelect?.(story.id)}
      >
        <div className="large-story-top">
          <div className="large-story-tagline">
            <StoryIcon story={story} />
            <div>
              <div className={`story-row-tag story-row-tag--${toneClass(story.tone)}`}>
                {story.tag}
              </div>
              <div className="large-story-source">{story.source}</div>
            </div>
          </div>
  
          <MiniBadge tone={story.tone}>{story.status}</MiniBadge>
        </div>
  
        <div className="large-story-title">{story.headline}</div>
        <div className="large-story-body">{story.body}</div>
  
        <StoryMetaLine story={story} />
  
        <div className="large-story-bottom">
          <HeatBar value={story.heat} label="Story Heat" tone={story.tone} />
          <StoryEffectList effects={story.effects} limit={3} />
        </div>
      </button>
    );
  }
  
  function StoryListPanel({
    title,
    subtitle,
    stories,
    emptyLabel,
    emptyDetail,
    actionLabel,
    onAction,
    selectedStoryId,
    onSelectStory,
    badge,
    badgeTone,
  }) {
    return (
      <SectionShell
        title={title}
        subtitle={subtitle}
        actionLabel={actionLabel}
        onAction={onAction}
        badge={badge}
        badgeTone={badgeTone}
        className="story-list-panel"
      >
        <div className="story-list">
          {!stories.length ? (
            <EmptyState title={emptyLabel} detail={emptyDetail} />
          ) : (
            stories.map((story) => (
              <CompactStoryRow
                key={story.id}
                story={story}
                selected={String(selectedStoryId) === String(story.id)}
                onSelect={onSelectStory}
              />
            ))
          )}
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     13. HERO STORY COMPONENT
     ============================================================ */
  
  function HeroPlayerGhost({ story, morale, teamName }) {
    const player = asArray(story?.affectedPlayers)[0];
    const displayName = player?.name || teamName;
    const numberValue = player?.overall || morale;
  
    return (
      <div className={`hero-player-ghost hero-player-ghost--${toneClass(story?.tone)}`} aria-hidden="true">
        <div className="hero-player-helmet" />
        <div className="hero-player-head" />
        <div className="hero-player-body" />
        <div className="hero-player-name">{initials(displayName)}</div>
        <div className="hero-player-number">{String(Math.round(numberValue)).padStart(2, "0")}</div>
      </div>
    );
  }
  
  function HeroStory({
    story,
    morale,
    teamName,
    choices,
    onResolveStorylineChoice,
    onOpenFullStory,
  }) {
    if (!story) return null;
  
    const hasChoices = choices?.options?.length;
  
    return (
      <article className={`hero-story hero-story--${toneClass(story.tone)}`}>
        <HeroPlayerGhost story={story} morale={morale} teamName={teamName} />
  
        <div className="hero-content">
          <div className="hero-tags">
            <div className={`hero-tag hero-tag--${toneClass(story.tone)}`}>{story.tag}</div>
            <div className="hero-tag-secondary">Top Story</div>
            {story.isUserTeam ? <MiniBadge tone="warning">Your Team</MiniBadge> : null}
            {hasChoices ? <MiniBadge tone="negative">Decision</MiniBadge> : null}
          </div>
  
          <h2 className="hero-title">{story.headline}</h2>
  
          <div className="hero-body">{story.body}</div>
  
          <div className="hero-proof-grid">
            <div className="hero-proof-card">
              <span>Source</span>
              <strong>{story.source}</strong>
            </div>
            <div className="hero-proof-card">
              <span>Credibility</span>
              <strong>{Math.round(story.credibility)}%</strong>
            </div>
            <div className="hero-proof-card">
              <span>Status</span>
              <strong>{story.status}</strong>
            </div>
            <div className="hero-proof-card">
              <span>Heat</span>
              <strong>{Math.round(story.heat)}</strong>
            </div>
          </div>
  
          {story.cause ? (
            <div className="hero-cause">
              <strong>Cause:</strong> {story.cause}
            </div>
          ) : null}
  
          {story.effectSummary ? (
            <div className="hero-cause">
              <strong>Effect:</strong> {story.effectSummary}
            </div>
          ) : null}
  
          <div className="hero-bottom">
            <button
              type="button"
              className="view-story-btn"
              onClick={() => onOpenFullStory?.(story.id)}
            >
              View Full Story <span>›</span>
            </button>
  
            <div className="hero-impact">
              <div>
                <small>Primary Impact</small>
                {story.primaryImpact}
              </div>
              <strong className={`impact-${story.impact.tone}`}>
                {story.impact.label}
              </strong>
            </div>
          </div>
  
          <AffectedPlayersRow players={story.affectedPlayers} />
          <AffectedTeamsRow teams={story.affectedTeams} />
  
          {hasChoices ? (
            <div className="hero-decision-preview">
              <div className="hero-decision-title">GM Response Required</div>
              <div className="hero-decision-options">
                {choices.options.slice(0, 3).map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    className={`decision-btn decision-btn--${toneClass(opt.tone)}`}
                    disabled={opt.unavailable}
                    onClick={() => {
                      if (!opt.unavailable) {
                        onResolveStorylineChoice?.(choices.storylineId || story.id, opt.id);
                      }
                    }}
                    title={opt.effectSummary || fmtDeltaMap(opt.effects)}
                  >
                    <span>{opt.label}</span>
                    <small>{opt.unavailable ? opt.unavailableReason || "Unavailable" : opt.effectSummary || fmtDeltaMap(opt.effects)}</small>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </article>
    );
  }
  
  /* ============================================================
     14. TRENDING PLAYERS / PLAYER PRESSURE
     ============================================================ */
  
  function TrendingPlayersPanel({ players, setScreen }) {
    return (
      <SectionShell
        title="Trending Players"
        subtitle="Heat based on stats, morale, rumors, injuries, and recent form."
        actionLabel="View All"
        onAction={() => setScreen?.(SCREENS.STATS)}
        className="trending-panel"
      >
        <div className="trending-list">
          {!players.length ? (
            <EmptyState title="No player trend data yet." detail="Advance the season to generate more player movement." />
          ) : (
            players.slice(0, 5).map((p, idx) => (
              <button
                key={`${p.id}-${idx}`}
                type="button"
                className="trending-row"
                onClick={() => setScreen?.(SCREENS.STATS)}
              >
                <div className="trend-rank">{idx + 1}</div>
                <PlayerAvatar name={p.name} size="md" status={p.isInjured ? "negative" : p.morale >= 70 ? "positive" : "neutral"} />
                <div className="trend-copy">
                  <div className="trend-name">{p.name}</div>
                  <div className="trend-sub">
                    {p.pos} | {p.age} | {p.team}
                  </div>
                  <div className="trend-sub">{p.recent}</div>
                </div>
                <div className={`trend-badge trend-badge--${toneClass(p.isInjured ? "negative" : p.tag === "RUMOR" ? "rumor" : "positive")}`}>
                  {p.tag}
                </div>
                <div className="trend-arrow">↑</div>
              </button>
            ))
          )}
        </div>
      </SectionShell>
    );
  }
  
  function PlayerPressurePanel({ players }) {
    return (
      <SectionShell
        title="Locker Room Watch"
        subtitle="Players most likely to become storylines."
        badge={`${players.length} tracked`}
        badgeTone="warning"
        className="player-pressure-panel"
      >
        <div className="player-pressure-list">
          {!players.length ? (
            <EmptyState title="No pressure data yet." detail="Player morale and fatigue data will appear here once available." />
          ) : (
            players.slice(0, 6).map((p) => (
              <div key={p.id} className="pressure-player-row">
                <PlayerAvatar name={p.name} size="sm" status={p.pressureScore >= 60 ? "negative" : "neutral"} />
  
                <div className="pressure-player-copy">
                  <div className="pressure-player-name">{p.name}</div>
                  <div className="pressure-player-sub">
                    {p.pos} · {p.role} · {p.statsLine}
                  </div>
                </div>
  
                <div className="pressure-player-meter">
                  <HeatBar value={p.pressureScore} label={p.pressureLabel} tone={p.pressureScore >= 60 ? "negative" : "neutral"} />
                </div>
              </div>
            ))
          )}
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     15. MORALE / CONSEQUENCE PANELS
     ============================================================ */
  
  function MoralePanel({ morale, moraleTrend, keyFactors }) {
    return (
      <SectionShell
        title="Impact on Morale"
        subtitle="How recent stories are pushing the room."
        badge="Live"
        badgeTone={morale >= 65 ? "positive" : morale >= 45 ? "warning" : "negative"}
        className="morale-panel"
      >
        <div className="morale-content">
          <div className="morale-subtitle">Team Morale Trend</div>
  
          <div className="morale-layout">
            <div className="morale-chart-wrap">
              <MiniLineChart points={moraleTrend} tone="gold" height={190} />
              <div className="chart-labels">
                <span>Two Weeks Ago</span>
                <span>Today</span>
              </div>
            </div>
  
            <div className="morale-scorebox">
              <div className="morale-score-label">Team Morale</div>
              <div className="morale-score">{Math.round(morale)}</div>
              <div className="morale-word">{moraleLabel(morale)}</div>
            </div>
          </div>
  
          <div className="factor-list">
            <div className="factor-title">Key Factors</div>
            {keyFactors.slice(0, 6).map((f, idx) => (
              <div key={`${f.label}-${idx}`} className="factor-row" title={f.story}>
                <span>{f.label}</span>
                <strong className={f.value >= 0 ? "impact-positive" : "impact-negative"}>
                  {f.value > 0 ? "+" : ""}
                  {f.value}
                </strong>
              </div>
            ))}
          </div>
  
          <button type="button" className="panel-action-btn morale-details-btn">
            Morale Details <span>›</span>
          </button>
        </div>
      </SectionShell>
    );
  }
  
  function ConsequencePanel({ snapshot }) {
    const crisisTone =
      snapshot.crisisIndex >= 70
        ? "negative"
        : snapshot.crisisIndex >= 45
          ? "warning"
          : "positive";
  
    return (
      <SectionShell
        title="Consequence Center"
        subtitle="What the sim is currently pressuring."
        badge={pressureLabel(snapshot.mediaPressure)}
        badgeTone={crisisTone}
        className="consequence-panel"
      >
        <div className="consequence-grid">
          <RadialGauge
            value={snapshot.crisisIndex}
            label="Crisis"
            subLabel="Overall Heat"
            tone={crisisTone}
          />
  
          <div className="consequence-stat-stack">
            <StatChip
              label="Media Pressure"
              value={Math.round(snapshot.mediaPressure)}
              tone={valueTone(snapshot.mediaPressure, false)}
              sub={pressureLabel(snapshot.mediaPressure)}
            />
            <StatChip
              label="Room Tension"
              value={Math.round(snapshot.roomTension)}
              tone={valueTone(snapshot.roomTension, false)}
              sub={pressureLabel(snapshot.roomTension)}
            />
            <StatChip
              label="Owner Patience"
              value={Math.round(snapshot.ownerPatience)}
              tone={valueTone(snapshot.ownerPatience)}
              sub={snapshot.ownerPatience >= 65 ? "Safe" : snapshot.ownerPatience >= 45 ? "Watching" : "Angry"}
            />
            <StatChip
              label="Active Decisions"
              value={snapshot.actionCount}
              tone={snapshot.actionCount ? "negative" : "positive"}
              sub={snapshot.actionCount ? "Needs GM" : "Clear"}
            />
          </div>
        </div>
  
        <div className="consequence-strip">
          <div>
            <span>Story Heat</span>
            <strong>{snapshot.heatAvg}</strong>
          </div>
          <div>
            <span>Negative Stories</span>
            <strong>{snapshot.negativeCount}</strong>
          </div>
          <div>
            <span>Rumors</span>
            <strong>{snapshot.rumorCount}</strong>
          </div>
          <div>
            <span>Injury State</span>
            <strong>{snapshot.injuryCrisis.label}</strong>
          </div>
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     16. INJURY PANEL
     ============================================================ */
  
  function InjuryDeskPanel({ injuries, injuryCrisis }) {
    return (
      <SectionShell
        title="Injury Desk"
        subtitle="Roster health, expected returns, and lineup pressure."
        badge={`${injuryCrisis.count} active`}
        badgeTone={injuryCrisis.score >= 50 ? "negative" : injuryCrisis.count ? "warning" : "positive"}
        className="injury-desk-panel"
      >
        <div className="injury-summary-row">
          <RadialGauge
            value={injuryCrisis.score}
            label="Health"
            subLabel={injuryCrisis.label}
            tone={injuryCrisis.score >= 50 ? "negative" : injuryCrisis.count ? "warning" : "positive"}
          />
  
          <div className="injury-summary-copy">
            <div className="injury-summary-title">
              {injuryCrisis.count ? `${injuryCrisis.label} injury situation` : "Roster is mostly healthy"}
            </div>
            <div className="injury-summary-body">
              {injuryCrisis.count
                ? "Medical pressure is affecting lineup decisions, call-up planning, and short-term performance risk."
                : "No major injury pressure is currently being reported by the sim payload."}
            </div>
          </div>
        </div>
  
        <div className="injury-list">
          {!injuries.length ? (
            <EmptyState title="No active injuries." detail="Injury updates will appear here when the sim logs them." />
          ) : (
            injuries.slice(0, 6).map((injury) => (
              <div key={injury.id} className={`injury-row injury-row--${toneClass(injuryTone(injury))}`}>
                <PlayerAvatar name={injury.player} size="sm" status={injuryTone(injury)} />
  
                <div className="injury-copy">
                  <div className="injury-player">{injury.player}</div>
                  <div className="injury-detail">
                    {injury.position} · {injury.injury} · {injury.severity}
                  </div>
                  <div className="injury-impact">{injury.impact}</div>
                </div>
  
                <div className="injury-return">
                  <span>Return</span>
                  <strong>{injury.expectedReturn}</strong>
                  <small>{injury.gamesRemaining} games</small>
                </div>
              </div>
            ))
          )}
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     17. SCHEDULE / STANDINGS PRESSURE PANEL
     ============================================================ */
  
  function SchedulePressurePanel({ franchiseState, schedulePressure }) {
    const userStanding = getUserTeamStanding(franchiseState);
    const raceStatus = getPlayoffRaceStatus(franchiseState);
    const divisionRank = getDivisionRank(franchiseState);
  
    return (
      <SectionShell
        title="Season Pressure"
        subtitle="How the standings and calendar are feeding the narrative."
        badge={raceStatus}
        badgeTone={raceStatus === "Playoff Spot" || raceStatus === "Comfortably In" ? "positive" : raceStatus === "Bubble" ? "warning" : "negative"}
        className="season-pressure-panel"
      >
        <div className="season-pressure-grid">
          <div className="season-pressure-main">
            <div className="season-pressure-record">
              <span>Record</span>
              <strong>{formatRecord(userStanding.wins, userStanding.losses, userStanding.otl)}</strong>
              <small>{userStanding.points} PTS · {formatPct(userStanding.pointPct)} P%</small>
            </div>
  
            <div className="season-pressure-record">
              <span>Division Rank</span>
              <strong>{divisionRank}</strong>
              <small>{userStanding.division}</small>
            </div>
  
            <div className="season-pressure-record">
              <span>Race Status</span>
              <strong>{raceStatus}</strong>
              <small>{userStanding.streak !== "—" ? `Streak: ${userStanding.streak}` : "Standings pressure live"}</small>
            </div>
          </div>
  
          <div className="schedule-pressure-box">
            <HeatBar
              value={schedulePressure.score}
              label="Schedule Pressure"
              tone={schedulePressure.score >= 65 ? "negative" : schedulePressure.score >= 40 ? "warning" : "positive"}
            />
  
            <div className="schedule-pressure-facts">
              <div>
                <span>Next 7 Days</span>
                <strong>{schedulePressure.gamesNext7} games</strong>
              </div>
              <div>
                <span>Back-to-Backs</span>
                <strong>{schedulePressure.backToBacks}</strong>
              </div>
              <div>
                <span>Next Rest</span>
                <strong>{schedulePressure.restDays} days</strong>
              </div>
            </div>
          </div>
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     18. GM DECISION COMPONENTS
     ============================================================ */
  
  function DecisionOptionCard({ option, story, choices, onResolveStorylineChoice }) {
    const category = choiceCategory(option);
  
    return (
      <button
        type="button"
        disabled={option.unavailable}
        className={`decision-option-card decision-option-card--${toneClass(option.tone)} ${
          option.unavailable ? "decision-option-card--disabled" : ""
        }`}
        onClick={() => {
          if (!option.unavailable) {
            onResolveStorylineChoice?.(choices.storylineId || story.id, option.id);
          }
        }}
      >
        <div className="decision-option-top">
          <MiniBadge tone={option.tone}>{category}</MiniBadge>
          <span>{option.riskLabel}</span>
        </div>
  
        <div className="decision-option-title">{option.label}</div>
        <div className="decision-option-body">
          {option.unavailable
            ? option.unavailableReason || "This option is currently unavailable."
            : option.description}
        </div>
  
        <div className="decision-option-footer">
          <HeatBar value={option.riskScore} label="Risk" tone={option.tone} />
          <StoryEffectList effects={option.effects} limit={3} />
        </div>
      </button>
    );
  }
  
  function DecisionCard({ row, onResolveStorylineChoice, onSelectStory }) {
    const { story, choices } = row;
  
    return (
      <div className={`decision-card decision-card--${toneClass(story.tone)}`}>
        <div className="decision-card-header">
          <div>
            <div className="decision-kicker">GM Decision</div>
            <div className="decision-title">{story.headline}</div>
            <div className="decision-sub">{choices.description || story.body}</div>
          </div>
  
          <button type="button" className="decision-open-story" onClick={() => onSelectStory?.(story.id)}>
            Open Story ›
          </button>
        </div>
  
        <div className="decision-context-grid">
          <StatChip label="Story Heat" value={Math.round(story.heat)} tone={story.tone} sub={story.status} />
          <StatChip label="Credibility" value={`${Math.round(story.credibility)}%`} tone="neutral" sub={story.source} />
          <StatChip
            label="Deadline"
            value={choices.deadline ? formatDateShort(choices.deadline) : "Open"}
            tone={choices.deadline ? "warning" : "neutral"}
            sub={choices.deadline ? "Response window" : "No timer"}
          />
        </div>
  
        <div className="decision-option-grid">
          {choices.options.map((option) => (
            <DecisionOptionCard
              key={option.id}
              option={option}
              story={story}
              choices={choices}
              onResolveStorylineChoice={onResolveStorylineChoice}
            />
          ))}
        </div>
      </div>
    );
  }
  
  function DecisionsPanel({ decisionRows, onResolveStorylineChoice, onSelectStory }) {
    return (
      <SectionShell
        title="GM Decisions"
        subtitle="Choices wired to existing storyline choice payloads."
        badge={decisionRows.length ? "Action Required" : "Clear"}
        badgeTone={decisionRows.length ? "negative" : "positive"}
        className="decisions-panel"
      >
        <div className="decision-stack">
          {!decisionRows.length ? (
            <EmptyState title="No GM decisions pending." detail="When the backend sends storyline_choices, they will appear here." />
          ) : (
            decisionRows.map((row) => (
              <DecisionCard
                key={row.story.id}
                row={row}
                onResolveStorylineChoice={onResolveStorylineChoice}
                onSelectStory={onSelectStory}
              />
            ))
          )}
        </div>
      </SectionShell>
    );
  }
  
  /* ============================================================
     19. FULL STORY MODAL / DETAIL PANEL
     ============================================================ */
  
  function FullStoryModal({
    story,
    choices,
    onClose,
    onResolveStorylineChoice,
  }) {
    if (!story) return null;
  
    return (
      <div className="story-modal-backdrop" role="dialog" aria-modal="true">
        <div className={`story-modal story-modal--${toneClass(story.tone)}`}>
          <div className="story-modal-header">
            <div className="story-modal-title-block">
              <div className="hero-tags">
                <div className={`hero-tag hero-tag--${toneClass(story.tone)}`}>{story.tag}</div>
                <div className="hero-tag-secondary">{story.source}</div>
                <MiniBadge tone={story.tone}>{story.status}</MiniBadge>
              </div>
  
              <h2>{story.headline}</h2>
              <StoryMetaLine story={story} />
            </div>
  
            <button type="button" className="story-modal-close" onClick={onClose}>
              ×
            </button>
          </div>
  
          <div className="story-modal-body">
            <div className="story-modal-main">
              <p className="story-modal-lede">{story.body}</p>
  
              {story.cause ? (
                <div className="story-detail-block">
                  <h4>What caused this?</h4>
                  <p>{story.cause}</p>
                </div>
              ) : null}
  
              {story.effectSummary ? (
                <div className="story-detail-block">
                  <h4>Current sim effect</h4>
                  <p>{story.effectSummary}</p>
                </div>
              ) : null}
  
              {story.followUp ? (
                <div className="story-detail-block">
                  <h4>What could happen next?</h4>
                  <p>{story.followUp}</p>
                </div>
              ) : (
                <div className="story-detail-block">
                  <h4>What could happen next?</h4>
                  <p>
                    If this storyline stays active, future games, player morale, media pressure, fan confidence,
                    and roster decisions can continue to shape the fallout.
                  </p>
                </div>
              )}
  
              <div className="story-detail-block">
                <h4>Direct effects</h4>
                <StoryEffectList effects={story.effects} limit={8} />
              </div>
  
              <AffectedPlayersRow players={story.affectedPlayers} />
              <AffectedTeamsRow teams={story.affectedTeams} />
            </div>
  
            <aside className="story-modal-side">
              <RadialGauge value={story.heat} label="Heat" subLabel="Story Heat" tone={story.tone} />
              <RadialGauge value={story.credibility} label="Trust" subLabel="Credibility" tone="neutral" />
  
              <div className="story-modal-facts">
                <div>
                  <span>Primary Impact</span>
                  <strong>{story.primaryImpact}</strong>
                </div>
                <div>
                  <span>Date</span>
                  <strong>{formatDateShort(story.date)}</strong>
                </div>
                <div>
                  <span>Team Link</span>
                  <strong>{story.isUserTeam ? "Your Team" : "League"}</strong>
                </div>
              </div>
            </aside>
          </div>
  
          {choices?.options?.length ? (
            <div className="story-modal-decisions">
              <div className="section-title-row section-title-row--inside">
                <div>
                  <h3>GM Response</h3>
                  <p>This uses the existing onResolveStorylineChoice callback.</p>
                </div>
              </div>
  
              <div className="decision-option-grid">
                {choices.options.map((option) => (
                  <DecisionOptionCard
                    key={option.id}
                    option={option}
                    story={story}
                    choices={choices}
                    onResolveStorylineChoice={onResolveStorylineChoice}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    );
  }
  
  /* ============================================================
     20. HEADLINE TICKER
     ============================================================ */
  
  function HeadlineTicker({ headlines, onSelectStory }) {
    const safeHeadlines = asArray(headlines).slice(0, 10);
  
    return (
      <section className="headline-ticker">
        <div className="ticker-label">League Headlines</div>
  
        <div className="ticker-items">
          {safeHeadlines.length ? (
            safeHeadlines.map((story, idx) => (
              <button
                type="button"
                key={story?.id || `headline-${idx}`}
                className="ticker-item"
                onClick={() => onSelectStory?.(story.id)}
              >
                {story.headline}
              </button>
            ))
          ) : (
            <span className="ticker-placeholder">No league headlines yet.</span>
          )}
        </div>
  
        <button type="button" className="ticker-action">
          View All Headlines ›
        </button>
      </section>
    );
  }
  
  /* ============================================================
     END OF CHUNK 2
     Next chunk starts with:
     - Main StorylinesScreen component
     - tab layouts
     - JSX layout structure
     - beginning of giant CSS block
     ============================================================ */
/* ============================================================
   21. TAB CONTENT LAYOUTS
   ============================================================ */

   function OverviewTab({
    topStory,
    teamName,
    morale,
    choicesByStoryId,
    onResolveStorylineChoice,
    onOpenFullStory,
    grouped,
    selectedStoryId,
    onSelectStory,
    trendingPlayers,
    keyFactors,
    moraleTrend,
    snapshot,
    setScreen,
  }) {
    const topChoices = getChoicesForStory(topStory, choicesByStoryId);
  
    return (
      <>
        <section className="story-grid">
          <HeroStory
            story={topStory}
            morale={morale}
            teamName={teamName}
            choices={topChoices}
            onResolveStorylineChoice={onResolveStorylineChoice}
            onOpenFullStory={onOpenFullStory}
          />
  
          <TrendingPlayersPanel players={trendingPlayers} setScreen={setScreen} />
        </section>
  
        <section className="lower-grid lower-grid--overview">
          <StoryListPanel
            title="League Buzz"
            subtitle="General league movement and daily noise."
            actionLabel="View All"
            stories={grouped.leagueBuzz}
            emptyLabel="No league buzz yet."
            emptyDetail="League storylines will fill in as the sim advances."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
          />
  
          <StoryListPanel
            title="Rumor Mill"
            subtitle="Trade smoke, scouting buzz, and market movement."
            actionLabel="Trade Hub"
            onAction={() => setScreen?.(SCREENS.TRADE_HUB)}
            stories={grouped.rumorMill}
            emptyLabel="No rumors circulating yet."
            emptyDetail="Trade market stories will appear here."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.rumorMill.length ? `${grouped.rumorMill.length} hot` : null}
            badgeTone="rumor"
          />
  
          <StoryListPanel
            title="Team Drama"
            subtitle="Morale, frustration, discipline, and locker-room pressure."
            actionLabel="View All"
            stories={grouped.teamDrama}
            emptyLabel="No team drama yet."
            emptyDetail="Good. That means the room has not caught fire today."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.teamDrama.length ? `${grouped.teamDrama.length} active` : null}
            badgeTone="negative"
          />
  
          <MoralePanel
            morale={morale}
            moraleTrend={moraleTrend}
            keyFactors={keyFactors}
          />
        </section>
  
        <section className="analysis-grid">
          <ConsequencePanel snapshot={snapshot} />
  
          <StoryListPanel
            title="Your Team Feed"
            subtitle="Stories directly connected to your franchise."
            stories={grouped.userTeam}
            emptyLabel="No direct team stories."
            emptyDetail="Your team is not currently the center of the circus."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.userTeam.length ? "Tracked" : "Quiet"}
            badgeTone={grouped.userTeam.length ? "warning" : "positive"}
          />
        </section>
      </>
    );
  }
  
  function LeagueTab({
    grouped,
    storylines,
    selectedStoryId,
    onSelectStory,
  }) {
    const majorStories = asArray(storylines)
      .filter((story) => story.priority >= 4 || story.heat >= 70)
      .slice(0, 8);
  
    const positiveStories = asArray(storylines)
      .filter((story) => story.tone === "positive")
      .slice(0, 6);
  
    const negativeStories = asArray(storylines)
      .filter((story) => story.tone === "negative")
      .slice(0, 6);
  
    return (
      <>
        <section className="league-command-grid">
          <SectionShell
            title="National Desk"
            subtitle="The biggest league-wide stories by heat and priority."
            badge={`${majorStories.length} major`}
            badgeTone={majorStories.length ? "warning" : "neutral"}
            className="national-desk-panel"
          >
            <div className="large-card-grid">
              {majorStories.length ? (
                majorStories.map((story) => (
                  <LargeStoryCard
                    key={story.id}
                    story={story}
                    selected={String(selectedStoryId) === String(story.id)}
                    onSelect={onSelectStory}
                  />
                ))
              ) : (
                <EmptyState
                  title="No major national stories."
                  detail="The league desk is quiet right now."
                />
              )}
            </div>
          </SectionShell>
  
          <StoryListPanel
            title="Headline Stack"
            subtitle="Latest league items by priority."
            stories={grouped.headlines}
            emptyLabel="No headlines."
            emptyDetail="Advance the calendar to generate headlines."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
          />
        </section>
  
        <section className="triple-grid">
          <StoryListPanel
            title="Positive Momentum"
            subtitle="Awards, streaks, milestones, and rising teams."
            stories={positiveStories}
            emptyLabel="No positive stories."
            emptyDetail="Nobody is having fun yet."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={positiveStories.length ? "Good News" : null}
            badgeTone="positive"
          />
  
          <StoryListPanel
            title="League Trouble"
            subtitle="Negative events, pressure, discipline, and setbacks."
            stories={negativeStories}
            emptyLabel="No league trouble."
            emptyDetail="Suspiciously peaceful."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={negativeStories.length ? "Pressure" : null}
            badgeTone="negative"
          />
  
          <StoryListPanel
            title="Milestone Watch"
            subtitle="Records, big games, streaks, and career moments."
            stories={grouped.milestones}
            emptyLabel="No milestones yet."
            emptyDetail="Milestone stories will show up once the sim logs them."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.milestones.length ? "History" : null}
            badgeTone="positive"
          />
        </section>
      </>
    );
  }
  
  function TeamTab({
    franchiseState,
    grouped,
    playerPressure,
    snapshot,
    selectedStoryId,
    onSelectStory,
  }) {
    const teamName = getTeamName(franchiseState);
    const userStanding = getUserTeamStanding(franchiseState);
    const schedulePressure = snapshot.schedulePressure;
  
    return (
      <>
        <section className="team-command-grid">
          <SectionShell
            title={`${teamName} Situation Room`}
            subtitle="Your franchise-specific pressure, morale, and narrative map."
            badge={snapshot.crisisIndex >= 65 ? "Under Pressure" : "Stable"}
            badgeTone={snapshot.crisisIndex >= 65 ? "negative" : snapshot.crisisIndex >= 40 ? "warning" : "positive"}
            className="situation-room-panel"
          >
            <div className="situation-room-grid">
              <RadialGauge
                value={snapshot.crisisIndex}
                label="Heat"
                subLabel="Franchise Temperature"
                tone={snapshot.crisisIndex >= 65 ? "negative" : snapshot.crisisIndex >= 40 ? "warning" : "positive"}
              />
  
              <div className="situation-room-cards">
                <StatChip
                  label="Record"
                  value={formatRecord(userStanding.wins, userStanding.losses, userStanding.otl)}
                  tone={userStanding.pointPct >= 0.58 ? "positive" : userStanding.pointPct >= 0.48 ? "warning" : "negative"}
                  sub={`${userStanding.points} points`}
                />
                <StatChip
                  label="Playoff Race"
                  value={getPlayoffRaceStatus(franchiseState)}
                  tone={getPlayoffRaceStatus(franchiseState) === "Playoff Spot" ? "positive" : getPlayoffRaceStatus(franchiseState) === "Bubble" ? "warning" : "negative"}
                  sub={`Division ${getDivisionRank(franchiseState)}`}
                />
                <StatChip
                  label="Room Tension"
                  value={Math.round(snapshot.roomTension)}
                  tone={valueTone(snapshot.roomTension, false)}
                  sub={pressureLabel(snapshot.roomTension)}
                />
                <StatChip
                  label="Fan Confidence"
                  value={Math.round(snapshot.fanConfidence)}
                  tone={valueTone(snapshot.fanConfidence)}
                  sub={confidenceLabel(snapshot.fanConfidence)}
                />
              </div>
            </div>
          </SectionShell>
  
          <SchedulePressurePanel
            franchiseState={franchiseState}
            schedulePressure={schedulePressure}
          />
        </section>
  
        <section className="team-feed-grid">
          <StoryListPanel
            title="Your Team Stories"
            subtitle="Only stories connected to your franchise."
            stories={grouped.userTeam}
            emptyLabel="No direct team stories."
            emptyDetail="Your franchise is not currently in the spotlight."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.userTeam.length ? `${grouped.userTeam.length} active` : "Quiet"}
            badgeTone={grouped.userTeam.length ? "warning" : "positive"}
          />
  
          <PlayerPressurePanel players={playerPressure} />
        </section>
      </>
    );
  }
  
  function InjuriesTab({
    injuries,
    injuryCrisis,
    grouped,
    selectedStoryId,
    onSelectStory,
  }) {
    return (
      <>
        <section className="injury-command-grid">
          <InjuryDeskPanel
            injuries={injuries}
            injuryCrisis={injuryCrisis}
          />
  
          <SectionShell
            title="Medical Fallout"
            subtitle="Injury-related storylines generated by the sim."
            badge={grouped.injuryDesk.length ? "Active" : "Clear"}
            badgeTone={grouped.injuryDesk.length ? "negative" : "positive"}
            className="medical-fallout-panel"
          >
            <div className="story-list">
              {grouped.injuryDesk.length ? (
                grouped.injuryDesk.map((story) => (
                  <CompactStoryRow
                    key={story.id}
                    story={story}
                    selected={String(selectedStoryId) === String(story.id)}
                    onSelect={onSelectStory}
                  />
                ))
              ) : (
                <EmptyState
                  title="No injury storylines."
                  detail="Active injury records may still appear in the Injury Desk if the backend sends injury data."
                />
              )}
            </div>
          </SectionShell>
        </section>
  
        <section className="injury-analysis-grid">
          <SectionShell
            title="Depth Risk"
            subtitle="How injuries can pressure lineup and roster decisions."
            badge={injuryCrisis.label}
            badgeTone={injuryCrisis.score >= 50 ? "negative" : injuryCrisis.count ? "warning" : "positive"}
            className="depth-risk-panel"
          >
            <div className="depth-risk-body">
              <RadialGauge
                value={injuryCrisis.score}
                label="Risk"
                subLabel="Depth Risk"
                tone={injuryCrisis.score >= 50 ? "negative" : injuryCrisis.count ? "warning" : "positive"}
              />
  
              <div className="depth-risk-copy">
                <h4>
                  {injuryCrisis.score >= 70
                    ? "This is a real roster problem."
                    : injuryCrisis.score >= 45
                      ? "Depth is starting to matter."
                      : injuryCrisis.count
                        ? "Manageable, but worth watching."
                        : "Healthy enough for now."}
                </h4>
                <p>
                  This panel does not invent backend data. It reads the available injury payload and turns it into
                  lineup pressure, return timing, and possible GM decision context.
                </p>
              </div>
            </div>
          </SectionShell>
  
          <SectionShell
            title="Possible GM Responses"
            subtitle="Suggested actions based on injury pressure."
            className="injury-response-panel"
          >
            <div className="response-suggestion-grid">
              <div className="response-suggestion-card">
                <MiniBadge tone="neutral">Low Risk</MiniBadge>
                <strong>Shuffle Lines</strong>
                <p>Use internal depth and protect chemistry if the injury list is short.</p>
              </div>
              <div className="response-suggestion-card">
                <MiniBadge tone="warning">Medium Risk</MiniBadge>
                <strong>Call Up Prospect</strong>
                <p>Useful when the schedule is heavy or fatigue is building.</p>
              </div>
              <div className="response-suggestion-card">
                <MiniBadge tone="negative">High Risk</MiniBadge>
                <strong>Trade For Depth</strong>
                <p>Best when injuries overlap with playoff pressure or poor morale.</p>
              </div>
            </div>
          </SectionShell>
        </section>
      </>
    );
  }
  
  function DecisionsTab({
    decisionRows,
    onResolveStorylineChoice,
    onSelectStory,
    grouped,
    selectedStoryId,
  }) {
    return (
      <>
        <section className="decisions-command-grid">
          <DecisionsPanel
            decisionRows={decisionRows}
            onResolveStorylineChoice={onResolveStorylineChoice}
            onSelectStory={onSelectStory}
          />
  
          <StoryListPanel
            title="Action-Linked Stories"
            subtitle="Stories most likely to create GM choices."
            stories={grouped.actionRequired}
            emptyLabel="No action-linked stories."
            emptyDetail="When the backend sends decisions, they will be listed here."
            selectedStoryId={selectedStoryId}
            onSelectStory={onSelectStory}
            badge={grouped.actionRequired.length ? "Requires GM" : "Clear"}
            badgeTone={grouped.actionRequired.length ? "negative" : "positive"}
          />
        </section>
      </>
    );
  }
  
  /* ============================================================
     22. MAIN SCREEN COMPONENT
     ============================================================ */
  
  export default function StorylinesScreen() {
    const { franchiseState, onResolveStorylineChoice, setScreen } = useGameUI();
  
    const [selectedStoryId, setSelectedStoryId] = useState(null);
    const [activeTab, setActiveTab] = useState("overview");
    const [modalStoryId, setModalStoryId] = useState(null);
  
    const teamName = getTeamName(franchiseState);
    const gmRating = getGmRating(franchiseState);
    const chemistry = getChemistry(franchiseState);
    const fanConfidence = getFanConfidence(franchiseState);
    const morale = getMorale(franchiseState);
  
    const storylines = useMemo(() => {
      return collectStorylines(franchiseState);
    }, [franchiseState]);
  
    const choicesByStoryId = useMemo(() => {
      return buildChoicesByStoryId(franchiseState);
    }, [franchiseState?.storyline_choices]);
  
    const topStory = useMemo(() => {
      if (selectedStoryId) {
        const found = storylines.find((story) => String(story.id) === String(selectedStoryId));
        if (found) return found;
      }
  
      return storylines[0] || null;
    }, [storylines, selectedStoryId]);
  
    const modalStory = useMemo(() => {
      if (!modalStoryId) return null;
      return storylines.find((story) => String(story.id) === String(modalStoryId)) || null;
    }, [storylines, modalStoryId]);
  
    const trendingPlayers = useMemo(() => {
      return buildTrendingPlayers(franchiseState);
    }, [franchiseState]);
  
    const playerPressure = useMemo(() => {
      return buildPlayerPressureList(franchiseState);
    }, [franchiseState]);
  
    const grouped = useMemo(() => {
      return splitStorylines(storylines);
    }, [storylines]);
  
    const moraleTrend = useMemo(() => {
      return buildMoraleTrend(franchiseState, morale);
    }, [franchiseState, morale]);
  
    const chemistryTrend = useMemo(() => {
      return buildChemistryTrend(franchiseState, chemistry);
    }, [franchiseState, chemistry]);
  
    const keyFactors = useMemo(() => {
      return buildKeyFactors(storylines, franchiseState);
    }, [storylines, franchiseState]);
  
    const injuries = useMemo(() => {
      return buildInjuryReport(franchiseState);
    }, [franchiseState]);
  
    const injuryCrisis = useMemo(() => {
      return getInjuryCrisisScore(franchiseState);
    }, [franchiseState]);
  
    const snapshot = useMemo(() => {
      return buildConsequenceSnapshot(storylines, franchiseState);
    }, [storylines, franchiseState]);
  
    const decisionRows = useMemo(() => {
      return buildDecisionRows(storylines, choicesByStoryId);
    }, [storylines, choicesByStoryId]);
  
    const selectedStoryChoices = useMemo(() => {
      return getChoicesForStory(modalStory, choicesByStoryId);
    }, [modalStory, choicesByStoryId]);
  
    const handleSelectStory = useCallback((storyId) => {
      setSelectedStoryId(storyId);
    }, []);
  
    const handleOpenFullStory = useCallback((storyId) => {
      setSelectedStoryId(storyId);
      setModalStoryId(storyId);
    }, []);
  
    const handleCloseModal = useCallback(() => {
      setModalStoryId(null);
    }, []);
  
    const renderTab = () => {
      if (activeTab === "league") {
        return (
          <LeagueTab
            grouped={grouped}
            storylines={storylines}
            selectedStoryId={selectedStoryId}
            onSelectStory={handleSelectStory}
          />
        );
      }
  
      if (activeTab === "team") {
        return (
          <TeamTab
            franchiseState={franchiseState}
            grouped={grouped}
            playerPressure={playerPressure}
            snapshot={snapshot}
            selectedStoryId={selectedStoryId}
            onSelectStory={handleSelectStory}
          />
        );
      }
  
      if (activeTab === "injuries") {
        return (
          <InjuriesTab
            injuries={injuries}
            injuryCrisis={injuryCrisis}
            grouped={grouped}
            selectedStoryId={selectedStoryId}
            onSelectStory={handleSelectStory}
          />
        );
      }
  
      if (activeTab === "decisions") {
        return (
          <DecisionsTab
            decisionRows={decisionRows}
            onResolveStorylineChoice={onResolveStorylineChoice}
            onSelectStory={handleSelectStory}
            grouped={grouped}
            selectedStoryId={selectedStoryId}
          />
        );
      }
  
      return (
        <OverviewTab
          topStory={topStory}
          teamName={teamName}
          morale={morale}
          choicesByStoryId={choicesByStoryId}
          onResolveStorylineChoice={onResolveStorylineChoice}
          onOpenFullStory={handleOpenFullStory}
          grouped={grouped}
          selectedStoryId={selectedStoryId}
          onSelectStory={handleSelectStory}
          trendingPlayers={trendingPlayers}
          keyFactors={keyFactors}
          moraleTrend={moraleTrend}
          chemistryTrend={chemistryTrend}
          snapshot={snapshot}
          setScreen={setScreen}
        />
      );
    };
  
    return (
      <div className="storylines-screen">
        <style>{`
          .storylines-screen {
            --bg: #04101b;
            --bg2: #071827;
            --bg3: #020812;
            --panel: rgba(7, 24, 38, 0.92);
            --panel2: rgba(10, 33, 51, 0.94);
            --panel3: rgba(12, 43, 66, 0.9);
            --panel4: rgba(255, 255, 255, 0.045);
            --line: rgba(133, 190, 225, 0.22);
            --line2: rgba(42, 202, 255, 0.38);
            --line3: rgba(255, 255, 255, 0.08);
            --text: #edf7ff;
            --text2: #d5e4f1;
            --muted: #8da2b4;
            --muted2: #657889;
            --accent: #16d6ff;
            --blue: #2f9cff;
            --gold: #f5b92e;
            --green: #42e079;
            --red: #ff5a67;
            --orange: #ff8f3d;
            --purple: #a56cff;
            --meter-color: var(--accent);
            --gauge-color: var(--accent);
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
            --soft-shadow: 0 12px 32px rgba(0, 0, 0, 0.32);
            min-height: 100vh;
            background:
              radial-gradient(circle at 35% 0%, rgba(22, 214, 255, 0.13), transparent 30%),
              radial-gradient(circle at 88% 20%, rgba(47, 156, 255, 0.13), transparent 24%),
              radial-gradient(circle at 8% 90%, rgba(165, 108, 255, 0.08), transparent 34%),
              linear-gradient(180deg, #06131f 0%, #020812 100%);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            overflow: auto;
            padding-bottom: 24px;
          }
  
          .storylines-screen * {
            box-sizing: border-box;
          }
  
          .storylines-screen button {
            font-family: inherit;
          }
  
          .storylines-screen button:disabled {
            cursor: not-allowed;
            opacity: 0.55;
          }
  
          .storylines-screen ::selection {
            background: rgba(22, 214, 255, 0.28);
          }
  
          .story-topbar {
            min-height: 104px;
            display: grid;
            grid-template-columns: minmax(280px, 1fr) 190px 210px 230px 280px 260px 230px;
            align-items: center;
            gap: 0;
            border-bottom: 1px solid var(--line);
            background:
              linear-gradient(90deg, rgba(2, 9, 16, .96), rgba(6, 21, 34, .94)),
              repeating-linear-gradient(90deg, transparent 0, transparent 28px, rgba(255,255,255,.015) 29px);
            box-shadow: 0 12px 42px rgba(0,0,0,.38);
            position: sticky;
            top: 0;
            z-index: 20;
            backdrop-filter: blur(18px);
          }
  
          .franchise-brand {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 18px 24px;
            min-width: 0;
          }
  
          .team-shield {
            position: relative;
            display: grid;
            place-items: center;
            color: #dff8ff;
            font-weight: 950;
            letter-spacing: 1px;
            text-shadow: 0 0 14px rgba(22, 214, 255, .65);
            background:
              linear-gradient(145deg, rgba(22, 214, 255, .28), rgba(12, 43, 66, .9)),
              radial-gradient(circle at 50% 35%, rgba(255,255,255,.22), transparent 32%);
            border: 1px solid rgba(135, 210, 255, .45);
            box-shadow:
              inset 0 0 20px rgba(255,255,255,.05),
              0 0 26px rgba(22,214,255,.1);
            overflow: hidden;
          }
  
          .team-shield::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
              linear-gradient(120deg, transparent, rgba(255,255,255,.12), transparent);
            transform: translateX(-120%);
            animation: shieldSweep 5s ease-in-out infinite;
          }
  
          @keyframes shieldSweep {
            0%, 65% {
              transform: translateX(-120%);
            }
            82% {
              transform: translateX(120%);
            }
            100% {
              transform: translateX(120%);
            }
          }
  
          .team-shield--lg {
            width: 74px;
            height: 74px;
            clip-path: polygon(50% 0, 100% 22%, 86% 100%, 50% 78%, 14% 100%, 0 22%);
            font-size: 20px;
          }
  
          .team-shield--md {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            font-size: 14px;
          }
  
          .team-shield--sm {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            font-size: 12px;
          }
  
          .team-shield--xs {
            width: 24px;
            height: 24px;
            border-radius: 8px;
            font-size: 9px;
          }
  
          .brand-kicker {
            color: var(--muted);
            text-transform: uppercase;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 1.5px;
          }
  
          .brand-title {
            margin-top: 2px;
            text-transform: uppercase;
            font-size: 25px;
            line-height: 1;
            font-weight: 950;
            letter-spacing: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
  
          .brand-sub {
            margin-top: 6px;
            color: var(--accent);
            text-transform: uppercase;
            font-size: 13px;
            font-weight: 900;
            letter-spacing: 2px;
          }
  
          .topbar-cell {
            height: 68px;
            border-left: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            padding: 0 18px;
            min-width: 0;
          }
  
          .topbar-cell--continue {
            padding-right: 22px;
          }
  
          .topbar-copy {
            min-width: 0;
          }
  
          .topbar-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: var(--muted);
            font-weight: 900;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
  
          .topbar-value {
            margin-top: 5px;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 950;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
  
          .topbar-icon {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            background:
              linear-gradient(145deg, rgba(22,214,255,.16), rgba(255,255,255,.03));
            border: 1px solid var(--line);
            display: grid;
            place-items: center;
            font-size: 26px;
            color: var(--accent);
            box-shadow: inset 0 0 18px rgba(22,214,255,.05);
          }
  
          .continue-btn {
            width: 190px;
            min-height: 62px;
            border: 1px solid rgba(22, 214, 255, .55);
            background:
              radial-gradient(circle at 15% 0%, rgba(22,214,255,.22), transparent 45%),
              linear-gradient(135deg, rgba(8, 42, 75, .96), rgba(4, 20, 36, .96));
            color: #eaffff;
            border-radius: 8px;
            box-shadow:
              0 0 24px rgba(22, 214, 255, .24),
              inset 0 0 18px rgba(22, 214, 255, .1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1.8px;
            font-weight: 950;
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
          }
  
          .continue-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(22, 214, 255, .82);
            box-shadow:
              0 0 34px rgba(22, 214, 255, .3),
              inset 0 0 18px rgba(22, 214, 255, .12);
          }
  
          .continue-btn small {
            display: block;
            margin-top: 3px;
            color: var(--muted);
            letter-spacing: .5px;
            text-transform: none;
            font-weight: 800;
          }
  
          .continue-arrow {
            font-size: 28px;
            color: var(--accent);
          }
  
          .story-meter {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
            --meter-color: var(--accent);
          }
  
          .story-meter--positive {
            --meter-color: var(--green);
          }
  
          .story-meter--negative {
            --meter-color: var(--red);
          }
  
          .story-meter--warning {
            --meter-color: var(--gold);
          }
  
          .story-meter--rumor {
            --meter-color: var(--purple);
          }
  
          .story-meter-ring {
            border-radius: 999px;
            display: grid;
            place-items: center;
            box-shadow: 0 0 24px rgba(22, 214, 255, .18);
            flex: 0 0 auto;
          }
  
          .story-meter-inner {
            width: calc(100% - 10px);
            height: calc(100% - 10px);
            border-radius: 999px;
            background: #06131f;
            display: grid;
            place-items: center;
            font-size: 14px;
            font-weight: 950;
            color: var(--text);
            border: 1px solid rgba(255,255,255,.06);
          }
  
          .story-meter-text {
            min-width: 0;
          }
  
          .story-meter-text div {
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 900;
            white-space: nowrap;
          }
  
          .story-meter-text span {
            display: block;
            margin-top: 4px;
            font-size: 14px;
            font-weight: 950;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
  
          .story-main {
            padding: 18px 18px 0;
            max-width: 1920px;
            margin: 0 auto;
          }
  
          .story-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 18px;
            margin: 0 0 12px;
            padding: 0 0 0 6px;
          }
  
          .page-title {
            margin: 0;
            font-size: 46px;
            line-height: .9;
            text-transform: uppercase;
            font-weight: 950;
            letter-spacing: 4px;
            font-style: italic;
            color: #f0f7ff;
            text-shadow:
              0 3px 0 rgba(255,255,255,.08),
              0 0 24px rgba(22,214,255,.18);
          }
  
          .page-subtitle {
            margin-top: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #b9c7d5;
            font-size: 13px;
            font-weight: 900;
          }
  
          .story-tabs {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 14px;
            flex-wrap: wrap;
          }
  
          .story-tab {
            height: 34px;
            padding: 0 13px;
            border-radius: 999px;
            border: 1px solid rgba(133, 190, 225, .18);
            background: rgba(7, 24, 38, .66);
            color: var(--muted);
            cursor: pointer;
            text-transform: uppercase;
            font-size: 11px;
            font-weight: 950;
            letter-spacing: .9px;
            transition: background .16s ease, color .16s ease, border-color .16s ease, transform .16s ease;
          }
  
          .story-tab:hover {
            color: var(--text);
            border-color: var(--line2);
            transform: translateY(-1px);
          }
  
          .story-tab--active {
            color: #eaffff;
            border-color: rgba(22,214,255,.5);
            background:
              radial-gradient(circle at 20% 0%, rgba(22,214,255,.2), transparent 45%),
              rgba(22,214,255,.08);
            box-shadow: 0 0 18px rgba(22,214,255,.13);
          }
  
          .header-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: flex-end;
          }
  
          .nav-btn {
            height: 40px;
            padding: 0 16px;
            border-radius: 7px;
            border: 1px solid var(--line);
            background: rgba(7, 24, 38, .84);
            color: var(--text);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 900;
            font-size: 12px;
            cursor: pointer;
            transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
          }
  
          .nav-btn:hover {
            border-color: var(--line2);
            color: white;
            box-shadow: 0 0 18px rgba(22, 214, 255, .16);
            transform: translateY(-1px);
          }
  
          .story-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.65fr) minmax(420px, .85fr);
            gap: 10px;
          }
  
          .lower-grid {
            margin-top: 10px;
            display: grid;
            gap: 10px;
          }
  
          .lower-grid--overview {
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.02fr) minmax(0, 1.04fr) minmax(420px, .9fr);
          }
  
          .analysis-grid {
            margin-top: 10px;
            display: grid;
            grid-template-columns: minmax(0, .95fr) minmax(0, 1.05fr);
            gap: 10px;
          }
  
          .league-command-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(420px, .8fr);
            gap: 10px;
          }
  
          .triple-grid {
            margin-top: 10px;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
          }
  
          .team-command-grid,
          .injury-command-grid,
          .decisions-command-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(440px, .9fr);
            gap: 10px;
          }
  
          .team-feed-grid,
          .injury-analysis-grid {
            margin-top: 10px;
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(440px, .9fr);
            gap: 10px;
          }
  
          .section-shell {
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
              linear-gradient(180deg, rgba(8, 25, 39, .96), rgba(4, 14, 24, .96));
            overflow: hidden;
            box-shadow: var(--shadow);
            min-width: 0;
          }
  
          .section-title-row {
            min-height: 50px;
            padding: 10px 16px;
            border-bottom: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background:
              linear-gradient(90deg, rgba(10, 35, 53, .72), rgba(10,35,53,.32));
          }
  
          .section-title-row--inside {
            border-top: 1px solid var(--line);
          }
  
          .section-title-row h3 {
            margin: 0;
            text-transform: uppercase;
            font-size: 17px;
            letter-spacing: 1.3px;
            font-weight: 950;
          }
  
          .section-title-row p {
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.25;
            font-weight: 750;
          }
  
          .section-title-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
          }
  
          .tiny-link {
            border: 0;
            background: transparent;
            color: var(--muted);
            text-transform: uppercase;
            font-size: 11px;
            font-weight: 950;
            letter-spacing: .8px;
            cursor: pointer;
            white-space: nowrap;
          }
  
          .tiny-link:hover {
            color: var(--accent);
          }
  
          .mini-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 22px;
            padding: 4px 9px;
            border-radius: 999px;
            border: 1px solid rgba(133,190,225,.22);
            background: rgba(255,255,255,.045);
            color: var(--muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .8px;
            font-weight: 950;
            white-space: nowrap;
          }
  
          .mini-badge--positive {
            border-color: rgba(66, 224, 121, .35);
            background: rgba(66, 224, 121, .08);
            color: var(--green);
          }
  
          .mini-badge--negative {
            border-color: rgba(255, 90, 103, .35);
            background: rgba(255, 90, 103, .1);
            color: var(--red);
          }
  
          .mini-badge--warning {
            border-color: rgba(245, 185, 46, .38);
            background: rgba(245, 185, 46, .09);
            color: var(--gold);
          }
  
          .mini-badge--rumor {
            border-color: rgba(165, 108, 255, .38);
            background: rgba(165, 108, 255, .09);
            color: var(--purple);
          }
  
          .hero-story {
            position: relative;
            min-height: 372px;
            border: 1px solid rgba(22, 214, 255, .34);
            border-radius: 8px;
            overflow: hidden;
            background:
              linear-gradient(90deg, rgba(5, 14, 24, .98) 0%, rgba(5, 14, 24, .9) 44%, rgba(5, 14, 24, .38) 100%),
              radial-gradient(circle at 78% 45%, rgba(22, 214, 255, .26), transparent 28%),
              linear-gradient(135deg, rgba(9, 42, 68, .86), rgba(2, 10, 18, .96));
            box-shadow: var(--shadow);
          }
  
          .hero-story--negative {
            border-color: rgba(255,90,103,.34);
            background:
              linear-gradient(90deg, rgba(5, 14, 24, .98) 0%, rgba(5, 14, 24, .9) 44%, rgba(5, 14, 24, .38) 100%),
              radial-gradient(circle at 78% 45%, rgba(255,90,103,.18), transparent 30%),
              linear-gradient(135deg, rgba(65, 18, 33, .86), rgba(2, 10, 18, .96));
          }
  
          .hero-story--positive {
            border-color: rgba(66,224,121,.28);
            background:
              linear-gradient(90deg, rgba(5, 14, 24, .98) 0%, rgba(5, 14, 24, .9) 44%, rgba(5, 14, 24, .38) 100%),
              radial-gradient(circle at 78% 45%, rgba(66,224,121,.16), transparent 30%),
              linear-gradient(135deg, rgba(20, 61, 42, .72), rgba(2, 10, 18, .96));
          }
  
          .hero-story--rumor {
            border-color: rgba(165,108,255,.32);
            background:
              linear-gradient(90deg, rgba(5, 14, 24, .98) 0%, rgba(5, 14, 24, .9) 44%, rgba(5, 14, 24, .38) 100%),
              radial-gradient(circle at 78% 45%, rgba(165,108,255,.16), transparent 30%),
              linear-gradient(135deg, rgba(45, 25, 72, .72), rgba(2, 10, 18, .96));
          }
  
          .hero-story::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
              linear-gradient(110deg, rgba(255,255,255,.045), transparent 22%, transparent 70%, rgba(255,255,255,.035)),
              repeating-linear-gradient(135deg, rgba(255,255,255,.035) 0, rgba(255,255,255,.035) 1px, transparent 1px, transparent 13px);
            opacity: .42;
            pointer-events: none;
          }
  
          .hero-player-ghost {
            position: absolute;
            right: 40px;
            bottom: -28px;
            width: 350px;
            height: 310px;
            opacity: .92;
            filter: drop-shadow(0 30px 40px rgba(0,0,0,.55));
            pointer-events: none;
          }
  
          .hero-player-body {
            position: absolute;
            left: 60px;
            right: 40px;
            bottom: 0;
            height: 210px;
            background:
              linear-gradient(160deg, rgba(210,235,255,.18), rgba(17,64,95,.82) 32%, rgba(4,14,24,.95) 100%);
            clip-path: polygon(32% 0, 70% 4%, 86% 34%, 92% 100%, 8% 100%, 15% 36%);
            border: 1px solid rgba(180,230,255,.22);
            border-bottom: none;
          }
  
          .hero-player-ghost--negative .hero-player-body {
            background:
              linear-gradient(160deg, rgba(255,180,190,.16), rgba(76,28,45,.82) 32%, rgba(4,14,24,.95) 100%);
          }
  
          .hero-player-ghost--positive .hero-player-body {
            background:
              linear-gradient(160deg, rgba(190,255,215,.14), rgba(25,79,52,.82) 32%, rgba(4,14,24,.95) 100%);
          }
  
          .hero-player-ghost--rumor .hero-player-body {
            background:
              linear-gradient(160deg, rgba(220,190,255,.14), rgba(53,35,82,.82) 32%, rgba(4,14,24,.95) 100%);
          }
  
          .hero-player-head {
            position: absolute;
            left: 128px;
            top: 15px;
            width: 92px;
            height: 92px;
            border-radius: 50% 50% 45% 45%;
            background:
              radial-gradient(circle at 42% 34%, rgba(255,255,255,.28), transparent 18%),
              linear-gradient(145deg, #c99d78, #6f4432);
            border: 1px solid rgba(255,255,255,.18);
          }
  
          .hero-player-helmet {
            position: absolute;
            left: 106px;
            top: 0;
            width: 142px;
            height: 55px;
            background: linear-gradient(180deg, #071726, #02070c);
            border-radius: 55px 55px 12px 12px;
            border: 1px solid rgba(170,220,255,.25);
            transform: rotate(-5deg);
          }
  
          .hero-player-number {
            position: absolute;
            right: 62px;
            bottom: 62px;
            font-size: 92px;
            line-height: 1;
            font-weight: 950;
            color: rgba(232, 244, 255, .78);
            text-shadow: 0 4px 0 rgba(0,0,0,.45);
            transform: skew(-6deg);
          }
  
          .hero-player-name {
            position: absolute;
            right: 58px;
            bottom: 162px;
            font-size: 32px;
            font-weight: 950;
            letter-spacing: 2px;
            color: rgba(255,255,255,.8);
            transform: rotate(-2deg);
            text-transform: uppercase;
          }
  
          .hero-content {
            position: relative;
            z-index: 2;
            max-width: 760px;
            padding: 26px 24px;
          }
  
          .hero-tags {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 18px;
            flex-wrap: wrap;
          }
  
          .hero-tag {
            padding: 7px 12px;
            border-radius: 5px;
            background: linear-gradient(180deg, rgba(22,214,255,.9), rgba(11,85,112,.95));
            color: #fff;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1.2px;
            font-weight: 950;
            box-shadow: 0 0 18px rgba(22, 214, 255, .25);
          }
  
          .hero-tag--negative {
            background: linear-gradient(180deg, rgba(255,90,103,.95), rgba(136,30,44,.95));
            box-shadow: 0 0 18px rgba(255, 90, 103, .25);
          }
  
          .hero-tag--positive {
            background: linear-gradient(180deg, rgba(66,224,121,.95), rgba(22,111,55,.95));
            box-shadow: 0 0 18px rgba(66, 224, 121, .22);
          }
  
          .hero-tag--rumor {
            background: linear-gradient(180deg, rgba(165,108,255,.95), rgba(75,41,128,.95));
            box-shadow: 0 0 18px rgba(165, 108, 255, .22);
          }
  
          .hero-tag--warning {
            background: linear-gradient(180deg, rgba(245,185,46,.95), rgba(130,85,18,.95));
            box-shadow: 0 0 18px rgba(245, 185, 46, .22);
          }
  
          .hero-tag-secondary {
            color: var(--muted);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1.5px;
            font-weight: 950;
          }
  
          .hero-title {
            margin: 0;
            font-size: 37px;
            line-height: 1.08;
            text-transform: uppercase;
            font-style: italic;
            letter-spacing: 2px;
            font-weight: 950;
            max-width: 680px;
            text-shadow: 0 2px 0 rgba(0,0,0,.35);
          }
  
          .hero-body {
            margin-top: 18px;
            max-width: 570px;
            color: #c0ccd6;
            font-size: 16px;
            line-height: 1.45;
            font-weight: 650;
          }
  
          .hero-proof-grid {
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            max-width: 670px;
          }
  
          .hero-proof-card {
            min-height: 62px;
            border: 1px solid rgba(133,190,225,.16);
            background: rgba(255,255,255,.045);
            border-radius: 8px;
            padding: 10px;
          }
  
          .hero-proof-card span {
            display: block;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .8px;
            font-size: 10px;
            font-weight: 950;
          }
  
          .hero-proof-card strong {
            display: block;
            margin-top: 6px;
            color: var(--text);
            font-size: 13px;
            line-height: 1.15;
            font-weight: 950;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }
  
          .hero-cause {
            margin-top: 10px;
            max-width: 620px;
            color: #bfd0dd;
            font-size: 13px;
            line-height: 1.4;
            font-weight: 750;
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(133,190,225,.12);
            border-radius: 8px;
            padding: 10px 12px;
          }
  
          .hero-cause strong {
            color: var(--text);
          }
  
          .hero-bottom {
            margin-top: 22px;
            display: flex;
            align-items: center;
            gap: 32px;
            flex-wrap: wrap;
          }
  
          .view-story-btn,
          .panel-action-btn {
            height: 42px;
            padding: 0 18px;
            min-width: 180px;
            border-radius: 6px;
            border: 1px solid rgba(140, 205, 240, .36);
            background:
              linear-gradient(180deg, rgba(17, 50, 73, .96), rgba(6, 22, 35, .96));
            color: var(--text);
            text-transform: uppercase;
            font-weight: 950;
            letter-spacing: 1px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
          }
  
          .view-story-btn:hover,
          .panel-action-btn:hover {
            border-color: var(--line2);
            color: white;
            box-shadow: 0 0 18px rgba(22, 214, 255, .18);
            transform: translateY(-1px);
          }
  
          .hero-impact {
            display: flex;
            align-items: center;
            gap: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 950;
          }
  
          .hero-impact small {
            display: block;
            color: var(--muted);
            font-size: 12px;
          }
  
          .hero-impact strong {
            font-size: 17px;
          }
  
          .impact-negative {
            color: var(--red);
          }
  
          .impact-positive {
            color: var(--green);
          }
  
          .impact-neutral {
            color: var(--muted);
          }
  
          .hero-decision-preview {
            margin-top: 18px;
            max-width: 620px;
            border: 1px solid rgba(255,90,103,.22);
            background:
              radial-gradient(circle at 0% 0%, rgba(255,90,103,.12), transparent 48%),
              rgba(255,255,255,.035);
            border-radius: 10px;
            padding: 12px;
          }
  
          .hero-decision-title {
            text-transform: uppercase;
            color: var(--red);
            letter-spacing: 1px;
            font-size: 12px;
            font-weight: 950;
            margin-bottom: 9px;
          }
  
          .hero-decision-options {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
          }
  
          .decision-btn {
            border: 1px solid rgba(22, 214, 255, .25);
            background: rgba(22, 214, 255, .07);
            color: var(--text);
            border-radius: 6px;
            min-height: 42px;
            padding: 8px 10px;
            cursor: pointer;
            text-align: left;
            font-weight: 900;
            transition: border-color .16s ease, background .16s ease, transform .16s ease;
          }
  
          .decision-btn span {
            display: block;
            font-size: 12px;
            line-height: 1.2;
            text-transform: uppercase;
            letter-spacing: .6px;
          }
  
          .decision-btn small {
            display: block;
            margin-top: 4px;
            color: var(--muted);
            font-weight: 700;
            font-size: 11px;
            line-height: 1.25;
          }
  
          .decision-btn:hover {
            border-color: var(--line2);
            background: rgba(22, 214, 255, .12);
            transform: translateY(-1px);
          }
  
          .decision-btn--positive {
            border-color: rgba(66,224,121,.25);
            background: rgba(66,224,121,.08);
          }
  
          .decision-btn--negative {
            border-color: rgba(255,90,103,.25);
            background: rgba(255,90,103,.08);
          }
  
          .decision-btn--warning {
            border-color: rgba(245,185,46,.25);
            background: rgba(245,185,46,.08);
          }
  
          .decision-btn--rumor {
            border-color: rgba(165,108,255,.25);
            background: rgba(165,108,255,.08);
          }
  
          .affected-row {
            margin-top: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
          }
  
          .affected-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .9px;
            font-size: 11px;
            font-weight: 950;
          }
  
          .affected-stack {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
          }
  
          .affected-player-pill,
.affected-team-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(133,190,225,.16);
  background: rgba(255,255,255,.045);
  color: var(--text2);
  border-radius: 999px;
  padding: 4px 8px 4px 4px;
  font-size: 11px;
  font-weight: 850;
}

.player-avatar {
  position: relative;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background:
    radial-gradient(circle at 40% 28%, rgba(255,255,255,.42), transparent 18%),
    linear-gradient(145deg, #bd8a64, #5d3528);
  border: 1px solid rgba(255,255,255,.16);
  color: rgba(255,255,255,.9);
  font-weight: 950;
  box-shadow:
    inset 0 -12px 18px rgba(0,0,0,.25),
    0 8px 20px rgba(0,0,0,.22);
  flex: 0 0 auto;
  overflow: hidden;
}
        .player-avatar::after {
          content: "";
          position: absolute;
          inset: 0;
          background:
            linear-gradient(135deg, rgba(255,255,255,.14), transparent 45%, rgba(0,0,0,.14));
          pointer-events: none;
        }

        .player-avatar span {
          position: relative;
          z-index: 2;
        }

        .player-avatar--xs {
          width: 24px;
          height: 24px;
          font-size: 8px;
        }

        .player-avatar--sm {
          width: 38px;
          height: 38px;
          font-size: 11px;
        }

        .player-avatar--md {
          width: 46px;
          height: 46px;
          font-size: 12px;
        }

        .player-avatar--lg {
          width: 62px;
          height: 62px;
          font-size: 16px;
        }

        .player-avatar--positive {
          box-shadow:
            inset 0 -12px 18px rgba(0,0,0,.25),
            0 0 22px rgba(66,224,121,.18);
          border-color: rgba(66,224,121,.3);
        }

        .player-avatar--negative {
          box-shadow:
            inset 0 -12px 18px rgba(0,0,0,.25),
            0 0 22px rgba(255,90,103,.2);
          border-color: rgba(255,90,103,.34);
        }

        .player-avatar--warning {
          box-shadow:
            inset 0 -12px 18px rgba(0,0,0,.25),
            0 0 22px rgba(245,185,46,.18);
          border-color: rgba(245,185,46,.32);
        }

        .story-list {
          padding: 8px 12px 12px;
        }

        .story-list-panel {
          min-height: 380px;
        }

        .compact-story-row {
          width: 100%;
          display: grid;
          grid-template-columns: 50px minmax(0, 1fr) auto;
          gap: 12px;
          align-items: center;
          min-height: 76px;
          padding: 8px 0;
          border: 0;
          border-bottom: 1px solid rgba(133, 190, 225, .13);
          background: transparent;
          color: inherit;
          cursor: pointer;
          text-align: left;
          transition: background .16s ease, padding .16s ease;
        }

        .compact-story-row:last-child {
          border-bottom: 0;
        }

        .compact-story-row:hover {
          background: rgba(255,255,255,.025);
        }

        .compact-story-row--selected {
          background:
            linear-gradient(90deg, rgba(22,214,255,.08), transparent 80%);
          padding-left: 8px;
          border-left: 2px solid rgba(22,214,255,.6);
        }

        .story-icon {
          width: 42px;
          height: 42px;
          border-radius: 6px;
          display: grid;
          place-items: center;
          font-weight: 950;
          font-size: 21px;
          border: 1px solid rgba(255,255,255,.12);
          background: rgba(255,255,255,.05);
          color: var(--accent);
          box-shadow: inset 0 0 18px rgba(255,255,255,.03);
        }

        .story-icon--negative {
          background: rgba(255, 90, 103, .13);
          color: var(--red);
          border-color: rgba(255,90,103,.24);
        }

        .story-icon--positive {
          background: rgba(66, 224, 121, .12);
          color: var(--green);
          border-color: rgba(66,224,121,.24);
        }

        .story-icon--rumor {
          background: rgba(165, 108, 255, .12);
          color: var(--purple);
          border-color: rgba(165,108,255,.24);
        }

        .story-icon--warning {
          background: rgba(245, 185, 46, .12);
          color: var(--gold);
          border-color: rgba(245,185,46,.24);
        }

        .compact-story-main {
          min-width: 0;
        }

        .story-row-tag {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 1px;
          font-weight: 950;
          color: var(--gold);
        }

        .story-row-tag--negative {
          color: var(--red);
        }

        .story-row-tag--positive {
          color: var(--green);
        }

        .story-row-tag--rumor {
          color: var(--purple);
        }

        .story-row-tag--warning {
          color: var(--gold);
        }

        .compact-story-headline {
          margin-top: 3px;
          color: #dfeaf4;
          font-size: 13px;
          line-height: 1.25;
          font-weight: 900;
          transition: color .16s ease;
        }

        .compact-story-row:hover .compact-story-headline {
          color: white;
        }

        .compact-story-body {
          margin-top: 3px;
          color: #9caebe;
          font-size: 12px;
          line-height: 1.3;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .compact-story-side {
          color: var(--muted);
          font-size: 11px;
          text-align: right;
          min-width: 58px;
          display: grid;
          gap: 6px;
          justify-items: end;
        }

        .compact-story-side b {
          font-size: 13px;
        }

        .large-card-grid {
          padding: 12px;
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        .large-story-card {
          min-height: 238px;
          border: 1px solid rgba(133,190,225,.16);
          background:
            radial-gradient(circle at 100% 0%, rgba(22,214,255,.08), transparent 42%),
            rgba(255,255,255,.035);
          border-radius: 10px;
          padding: 14px;
          color: inherit;
          text-align: left;
          cursor: pointer;
          transition: transform .16s ease, border-color .16s ease, background .16s ease;
        }

        .large-story-card:hover {
          transform: translateY(-2px);
          border-color: var(--line2);
          background:
            radial-gradient(circle at 100% 0%, rgba(22,214,255,.12), transparent 42%),
            rgba(255,255,255,.05);
        }

        .large-story-card--selected {
          border-color: rgba(22,214,255,.56);
          box-shadow: 0 0 24px rgba(22,214,255,.12);
        }

        .large-story-card--negative {
          background:
            radial-gradient(circle at 100% 0%, rgba(255,90,103,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .large-story-card--positive {
          background:
            radial-gradient(circle at 100% 0%, rgba(66,224,121,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .large-story-card--rumor {
          background:
            radial-gradient(circle at 100% 0%, rgba(165,108,255,.11), transparent 42%),
            rgba(255,255,255,.035);
        }

        .large-story-top {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
        }

        .large-story-tagline {
          display: flex;
          align-items: center;
          gap: 10px;
          min-width: 0;
        }

        .large-story-source {
          margin-top: 3px;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .large-story-title {
          margin-top: 14px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 17px;
          line-height: 1.16;
          font-style: italic;
          font-weight: 950;
          letter-spacing: .8px;
        }

        .large-story-body {
          margin-top: 8px;
          color: #aebdca;
          font-size: 12px;
          line-height: 1.35;
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .story-meta-line {
          margin-top: 10px;
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
          color: var(--muted2);
          font-size: 10px;
          font-weight: 850;
          text-transform: uppercase;
          letter-spacing: .65px;
        }

        .story-meta-line span {
          display: inline-flex;
          align-items: center;
          gap: 5px;
        }

        .story-meta-line span::before {
          content: "•";
          color: var(--accent);
        }

        .story-meta-line span:first-child::before {
          content: "";
          display: none;
        }

        .large-story-bottom {
          margin-top: 12px;
          display: grid;
          gap: 10px;
        }

        .heatbar {
          --bar-color: var(--accent);
          min-width: 0;
        }

        .heatbar--positive {
          --bar-color: var(--green);
        }

        .heatbar--negative {
          --bar-color: var(--red);
        }

        .heatbar--warning {
          --bar-color: var(--gold);
        }

        .heatbar--rumor {
          --bar-color: var(--purple);
        }

        .heatbar-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          color: var(--muted);
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: .8px;
          font-weight: 950;
          margin-bottom: 5px;
        }

        .heatbar-top strong {
          color: var(--text);
        }

        .heatbar-track {
          height: 7px;
          border-radius: 999px;
          background: rgba(255,255,255,.08);
          overflow: hidden;
          border: 1px solid rgba(255,255,255,.05);
        }

        .heatbar-fill {
          height: 100%;
          border-radius: 999px;
          background:
            linear-gradient(90deg, rgba(255,255,255,.14), transparent),
            var(--bar-color);
          box-shadow: 0 0 12px color-mix(in srgb, var(--bar-color), transparent 45%);
        }

        .effect-list {
          display: flex;
          align-items: center;
          gap: 6px;
          flex-wrap: wrap;
        }

        .effect-empty {
          color: var(--muted2);
          font-size: 11px;
          font-weight: 750;
          border: 1px dashed rgba(133,190,225,.16);
          border-radius: 8px;
          padding: 8px;
        }

        .impact-pill {
          min-height: 24px;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border-radius: 999px;
          padding: 4px 8px;
          border: 1px solid rgba(133,190,225,.16);
          background: rgba(255,255,255,.045);
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: .65px;
          font-weight: 950;
          color: var(--muted);
        }

        .impact-pill strong {
          color: var(--text);
        }

        .impact-pill--positive {
          border-color: rgba(66,224,121,.28);
          background: rgba(66,224,121,.08);
          color: var(--green);
        }

        .impact-pill--negative {
          border-color: rgba(255,90,103,.28);
          background: rgba(255,90,103,.08);
          color: var(--red);
        }

        .impact-pill--neutral {
          color: var(--muted);
        }

        .trending-panel {
          min-height: 372px;
        }

        .trending-list {
          padding: 4px 12px 10px;
        }

        .trending-row {
          width: 100%;
          display: grid;
          grid-template-columns: 34px 58px minmax(0, 1fr) auto 22px;
          align-items: center;
          gap: 10px;
          min-height: 66px;
          border: 0;
          border-bottom: 1px solid rgba(133, 190, 225, .13);
          background: transparent;
          color: inherit;
          cursor: pointer;
          text-align: left;
          transition: background .16s ease, transform .16s ease;
        }

        .trending-row:last-child {
          border-bottom: 0;
        }

        .trending-row:hover {
          background: rgba(255,255,255,.025);
          transform: translateX(2px);
        }

        .trend-rank {
          color: #b8c8d7;
          font-weight: 950;
          font-size: 18px;
          text-align: center;
        }

        .trend-copy {
          min-width: 0;
        }

        .trend-name {
          color: var(--text);
          text-transform: uppercase;
          font-size: 15px;
          font-weight: 950;
          letter-spacing: .9px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .trend-sub {
          color: #9eb0bf;
          font-size: 12px;
          font-weight: 700;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .trend-badge {
          padding: 7px 12px;
          min-width: 76px;
          text-align: center;
          border-radius: 5px;
          border: 1px solid rgba(66, 224, 121, .22);
          background: rgba(66, 224, 121, .08);
          color: var(--green);
          text-transform: uppercase;
          font-size: 12px;
          font-weight: 950;
          letter-spacing: .8px;
        }

        .trend-badge--negative {
          border-color: rgba(255, 90, 103, .24);
          background: rgba(255, 90, 103, .08);
          color: var(--red);
        }

        .trend-badge--rumor {
          border-color: rgba(165, 108, 255, .24);
          background: rgba(165, 108, 255, .08);
          color: var(--purple);
        }

        .trend-arrow {
          color: var(--green);
          font-size: 22px;
          font-weight: 950;
        }

        .morale-panel {
          min-height: 420px;
        }

        .morale-content {
          padding: 14px 16px 16px;
        }

        .morale-subtitle {
          color: #c2d0dc;
          font-weight: 850;
          font-size: 13px;
          margin-bottom: 10px;
        }

        .morale-layout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 126px;
          gap: 12px;
          align-items: center;
        }

        .morale-chart-wrap {
          min-height: 190px;
          position: relative;
        }

        .mini-line-chart {
          width: 100%;
          height: 190px;
          overflow: visible;
        }

        .chart-grid {
          stroke: rgba(137, 177, 203, .16);
          stroke-width: 1;
        }

        .chart-area {
          fill: rgba(245,185,46,.16);
        }

        .chart-line {
          fill: none;
          stroke: var(--gold);
          stroke-width: 5;
          stroke-linecap: round;
          stroke-linejoin: round;
          filter: drop-shadow(0 0 6px rgba(245,185,46,.4));
        }

        .chart-dot {
          fill: var(--gold);
          stroke: rgba(7, 24, 38, .9);
          stroke-width: 2;
        }

        .mini-line-chart--green .chart-area {
          fill: rgba(66,224,121,.14);
        }

        .mini-line-chart--green .chart-line {
          stroke: var(--green);
          filter: drop-shadow(0 0 6px rgba(66,224,121,.35));
        }

        .mini-line-chart--green .chart-dot {
          fill: var(--green);
        }

        .mini-line-chart--blue .chart-area {
          fill: rgba(22,214,255,.14);
        }

        .mini-line-chart--blue .chart-line {
          stroke: var(--accent);
          filter: drop-shadow(0 0 6px rgba(22,214,255,.35));
        }

        .mini-line-chart--blue .chart-dot {
          fill: var(--accent);
        }

        .chart-labels {
          display: flex;
          justify-content: space-between;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 12px;
          font-weight: 850;
          margin-top: 4px;
        }

        .morale-scorebox {
          text-align: center;
          border-left: 1px solid rgba(133, 190, 225, .16);
          min-height: 190px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }

        .morale-score-label {
          color: var(--muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 1px;
          font-weight: 950;
        }

        .morale-score {
          margin-top: 6px;
          font-size: 54px;
          line-height: .9;
          color: ${morale >= 68 ? "var(--green)" : morale >= 52 ? "var(--gold)" : "var(--red)"};
          font-weight: 950;
        }

        .morale-word {
          margin-top: 10px;
          color: ${morale >= 68 ? "var(--green)" : morale >= 52 ? "var(--gold)" : "var(--red)"};
          text-transform: uppercase;
          font-size: 13px;
          font-weight: 950;
          letter-spacing: 1px;
        }

        .factor-list {
          margin-top: 14px;
          border-top: 1px solid rgba(133, 190, 225, .13);
          padding-top: 12px;
        }

        .factor-title {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 12px;
          letter-spacing: 1px;
          font-weight: 950;
          margin-bottom: 8px;
        }

        .factor-row {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
          min-height: 25px;
          color: #cbd7e1;
          font-size: 12px;
          font-weight: 800;
        }

        .factor-row strong {
          font-size: 13px;
        }

        .morale-details-btn {
          margin-top: 14px;
          width: 100%;
        }

        .consequence-panel {
          min-height: 360px;
        }

        .consequence-grid {
          padding: 16px;
          display: grid;
          grid-template-columns: 190px minmax(0, 1fr);
          gap: 16px;
          align-items: center;
        }

        .radial-gauge {
          --gauge-color: var(--accent);
          display: grid;
          justify-items: center;
          gap: 10px;
        }

        .radial-gauge--positive {
          --gauge-color: var(--green);
        }

        .radial-gauge--negative {
          --gauge-color: var(--red);
        }

        .radial-gauge--warning {
          --gauge-color: var(--gold);
        }

        .radial-gauge--rumor {
          --gauge-color: var(--purple);
        }

        .radial-gauge-ring {
          width: 148px;
          height: 148px;
          border-radius: 999px;
          display: grid;
          place-items: center;
          box-shadow: 0 0 30px rgba(22,214,255,.12);
        }

        .radial-gauge-inner {
          width: 118px;
          height: 118px;
          border-radius: 999px;
          background:
            radial-gradient(circle at 50% 25%, rgba(255,255,255,.08), transparent 40%),
            #06131f;
          border: 1px solid rgba(255,255,255,.07);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
        }

        .radial-gauge-inner strong {
          font-size: 34px;
          line-height: .95;
          font-weight: 950;
        }

        .radial-gauge-inner span {
          margin-top: 5px;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          font-weight: 950;
          letter-spacing: .8px;
        }

        .radial-gauge-sub {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: .8px;
          font-weight: 950;
        }

        .consequence-stat-stack {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        .stat-chip {
          border: 1px solid rgba(133,190,225,.16);
          background: rgba(255,255,255,.04);
          border-radius: 10px;
          padding: 12px;
          min-height: 78px;
        }

        .stat-chip--positive {
          border-color: rgba(66,224,121,.24);
          background: rgba(66,224,121,.07);
        }

        .stat-chip--negative {
          border-color: rgba(255,90,103,.24);
          background: rgba(255,90,103,.07);
        }

        .stat-chip--warning {
          border-color: rgba(245,185,46,.24);
          background: rgba(245,185,46,.07);
        }

        .stat-chip--rumor {
          border-color: rgba(165,108,255,.24);
          background: rgba(165,108,255,.07);
        }

        .stat-chip-label {
          color: var(--muted);
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: .85px;
          font-weight: 950;
        }

        .stat-chip-value {
          margin-top: 6px;
          color: var(--text);
          font-size: 22px;
          line-height: 1;
          font-weight: 950;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .stat-chip-sub {
          margin-top: 6px;
          color: var(--muted);
          font-size: 11px;
          line-height: 1.2;
          font-weight: 800;
        }

        .consequence-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          border-top: 1px solid rgba(133,190,225,.14);
        }

        .consequence-strip div {
          padding: 14px 12px;
          border-right: 1px solid rgba(133,190,225,.12);
        }

        .consequence-strip div:last-child {
          border-right: 0;
        }

        .consequence-strip span {
          display: block;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
        }

        .consequence-strip strong {
          display: block;
          margin-top: 5px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 15px;
          line-height: 1.15;
          font-weight: 950;
        }

        .empty-state {
          min-height: 220px;
          display: grid;
          place-items: center;
          align-content: center;
          color: var(--muted);
          font-weight: 850;
          text-transform: uppercase;
          letter-spacing: 1px;
          text-align: center;
          padding: 24px;
        }

        .empty-state-mark {
          width: 46px;
          height: 46px;
          border-radius: 14px;
          display: grid;
          place-items: center;
          border: 1px solid rgba(133,190,225,.18);
          background: rgba(255,255,255,.04);
          color: var(--accent);
          font-size: 22px;
          margin-bottom: 12px;
        }

        .empty-state-title {
          color: var(--text2);
          font-size: 13px;
          font-weight: 950;
        }

        .empty-state-detail {
          margin-top: 8px;
          max-width: 330px;
          color: var(--muted2);
          text-transform: none;
          letter-spacing: 0;
          font-size: 12px;
          line-height: 1.35;
          font-weight: 700;
        }        .injury-desk-panel {
          min-height: 430px;
        }

        .injury-summary-row {
          padding: 16px;
          display: grid;
          grid-template-columns: 170px minmax(0, 1fr);
          gap: 16px;
          align-items: center;
          border-bottom: 1px solid rgba(133,190,225,.13);
        }

        .injury-summary-copy {
          min-width: 0;
        }

        .injury-summary-title {
          color: var(--text);
          text-transform: uppercase;
          font-size: 19px;
          line-height: 1.1;
          font-weight: 950;
          letter-spacing: 1px;
          font-style: italic;
        }

        .injury-summary-body {
          margin-top: 9px;
          color: #aebdca;
          font-size: 13px;
          line-height: 1.45;
          font-weight: 750;
          max-width: 640px;
        }

        .injury-list {
          padding: 8px 12px 12px;
          display: grid;
          gap: 8px;
        }

        .injury-row {
          display: grid;
          grid-template-columns: 46px minmax(0, 1fr) 120px;
          gap: 12px;
          align-items: center;
          min-height: 82px;
          border: 1px solid rgba(133,190,225,.14);
          background: rgba(255,255,255,.035);
          border-radius: 10px;
          padding: 10px;
        }

        .injury-row--negative {
          border-color: rgba(255,90,103,.24);
          background:
            radial-gradient(circle at 100% 0%, rgba(255,90,103,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .injury-row--warning {
          border-color: rgba(245,185,46,.24);
          background:
            radial-gradient(circle at 100% 0%, rgba(245,185,46,.09), transparent 42%),
            rgba(255,255,255,.035);
        }

        .injury-copy {
          min-width: 0;
        }

        .injury-player {
          color: var(--text);
          text-transform: uppercase;
          font-size: 14px;
          font-weight: 950;
          letter-spacing: .8px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .injury-detail {
          margin-top: 3px;
          color: #aebdca;
          font-size: 12px;
          line-height: 1.25;
          font-weight: 800;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .injury-impact {
          margin-top: 4px;
          color: var(--muted);
          font-size: 11px;
          line-height: 1.25;
          font-weight: 700;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .injury-return {
          text-align: right;
          display: grid;
          gap: 3px;
          justify-items: end;
        }

        .injury-return span {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
        }

        .injury-return strong {
          color: var(--text);
          font-size: 13px;
          line-height: 1.15;
          font-weight: 950;
        }

        .injury-return small {
          color: var(--muted2);
          font-size: 10px;
          font-weight: 800;
        }

        .medical-fallout-panel {
          min-height: 430px;
        }

        .depth-risk-panel,
        .injury-response-panel {
          min-height: 300px;
        }

        .depth-risk-body {
          padding: 18px;
          display: grid;
          grid-template-columns: 170px minmax(0, 1fr);
          gap: 18px;
          align-items: center;
        }

        .depth-risk-copy h4 {
          margin: 0;
          color: var(--text);
          text-transform: uppercase;
          font-size: 19px;
          line-height: 1.1;
          font-weight: 950;
          font-style: italic;
          letter-spacing: .8px;
        }

        .depth-risk-copy p {
          margin: 10px 0 0;
          color: #aebdca;
          font-size: 13px;
          line-height: 1.45;
          font-weight: 750;
        }

        .response-suggestion-grid {
          padding: 14px;
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .response-suggestion-card {
          min-height: 150px;
          border: 1px solid rgba(133,190,225,.16);
          background:
            radial-gradient(circle at 100% 0%, rgba(22,214,255,.08), transparent 40%),
            rgba(255,255,255,.035);
          border-radius: 10px;
          padding: 14px;
        }

        .response-suggestion-card strong {
          display: block;
          margin-top: 12px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 14px;
          line-height: 1.15;
          font-weight: 950;
          letter-spacing: .7px;
        }

        .response-suggestion-card p {
          margin: 8px 0 0;
          color: var(--muted);
          font-size: 12px;
          line-height: 1.35;
          font-weight: 750;
        }

        .season-pressure-panel {
          min-height: 360px;
        }

        .season-pressure-grid {
          padding: 16px;
          display: grid;
          gap: 16px;
        }

        .season-pressure-main {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .season-pressure-record {
          min-height: 100px;
          border: 1px solid rgba(133,190,225,.16);
          background:
            linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.025));
          border-radius: 10px;
          padding: 14px;
        }

        .season-pressure-record span {
          display: block;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .85px;
          font-weight: 950;
        }

        .season-pressure-record strong {
          display: block;
          margin-top: 8px;
          color: var(--text);
          font-size: 23px;
          line-height: 1;
          font-weight: 950;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .season-pressure-record small {
          display: block;
          margin-top: 8px;
          color: var(--muted);
          font-size: 11px;
          line-height: 1.2;
          font-weight: 800;
        }

        .schedule-pressure-box {
          border-top: 1px solid rgba(133,190,225,.13);
          padding-top: 14px;
        }

        .schedule-pressure-facts {
          margin-top: 12px;
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .schedule-pressure-facts div {
          border: 1px solid rgba(133,190,225,.14);
          background: rgba(255,255,255,.035);
          border-radius: 10px;
          padding: 11px;
        }

        .schedule-pressure-facts span {
          display: block;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
        }

        .schedule-pressure-facts strong {
          display: block;
          margin-top: 6px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 15px;
          font-weight: 950;
        }

        .situation-room-panel {
          min-height: 360px;
        }

        .situation-room-grid {
          padding: 18px;
          display: grid;
          grid-template-columns: 190px minmax(0, 1fr);
          gap: 16px;
          align-items: center;
        }

        .situation-room-cards {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        .player-pressure-panel {
          min-height: 390px;
        }

        .player-pressure-list {
          padding: 12px;
          display: grid;
          gap: 8px;
        }

        .pressure-player-row {
          display: grid;
          grid-template-columns: 44px minmax(0, 1fr) 150px;
          gap: 12px;
          align-items: center;
          min-height: 72px;
          border: 1px solid rgba(133,190,225,.14);
          background: rgba(255,255,255,.035);
          border-radius: 10px;
          padding: 10px;
        }

        .pressure-player-copy {
          min-width: 0;
        }

        .pressure-player-name {
          color: var(--text);
          text-transform: uppercase;
          font-size: 13px;
          font-weight: 950;
          letter-spacing: .8px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .pressure-player-sub {
          margin-top: 4px;
          color: var(--muted);
          font-size: 11px;
          line-height: 1.25;
          font-weight: 750;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .pressure-player-meter {
          min-width: 0;
        }

        .decisions-panel {
          min-height: 430px;
        }

        .decision-stack {
          padding: 12px;
          display: grid;
          gap: 12px;
        }

        .decision-card {
          border: 1px solid rgba(133,190,225,.16);
          background:
            radial-gradient(circle at 100% 0%, rgba(22,214,255,.08), transparent 42%),
            rgba(255,255,255,.035);
          border-radius: 12px;
          padding: 14px;
        }

        .decision-card--negative {
          border-color: rgba(255,90,103,.24);
          background:
            radial-gradient(circle at 100% 0%, rgba(255,90,103,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .decision-card--warning {
          border-color: rgba(245,185,46,.24);
          background:
            radial-gradient(circle at 100% 0%, rgba(245,185,46,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .decision-card--rumor {
          border-color: rgba(165,108,255,.24);
          background:
            radial-gradient(circle at 100% 0%, rgba(165,108,255,.1), transparent 42%),
            rgba(255,255,255,.035);
        }

        .decision-card-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
        }

        .decision-kicker {
          color: var(--gold);
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: 1px;
          font-weight: 950;
        }

        .decision-title {
          margin-top: 5px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 17px;
          line-height: 1.15;
          font-weight: 950;
          font-style: italic;
          letter-spacing: .8px;
        }

        .decision-sub {
          margin-top: 7px;
          color: var(--muted);
          font-size: 12px;
          line-height: 1.35;
          font-weight: 750;
          max-width: 780px;
        }

        .decision-open-story {
          height: 34px;
          padding: 0 12px;
          border-radius: 7px;
          border: 1px solid rgba(133,190,225,.18);
          background: rgba(255,255,255,.045);
          color: var(--text);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
          cursor: pointer;
          white-space: nowrap;
        }

        .decision-open-story:hover {
          border-color: var(--line2);
          color: var(--accent);
        }

        .decision-context-grid {
          margin-top: 12px;
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 8px;
        }

        .decision-option-grid {
          margin-top: 12px;
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .decision-option-card {
          min-height: 185px;
          border: 1px solid rgba(133,190,225,.16);
          background: rgba(255,255,255,.035);
          color: inherit;
          border-radius: 10px;
          padding: 12px;
          text-align: left;
          cursor: pointer;
          transition: transform .16s ease, border-color .16s ease, background .16s ease;
        }

        .decision-option-card:hover {
          transform: translateY(-2px);
          border-color: var(--line2);
          background: rgba(255,255,255,.052);
        }

        .decision-option-card--positive {
          border-color: rgba(66,224,121,.24);
          background: rgba(66,224,121,.07);
        }

        .decision-option-card--negative {
          border-color: rgba(255,90,103,.24);
          background: rgba(255,90,103,.07);
        }

        .decision-option-card--warning {
          border-color: rgba(245,185,46,.24);
          background: rgba(245,185,46,.07);
        }

        .decision-option-card--rumor {
          border-color: rgba(165,108,255,.24);
          background: rgba(165,108,255,.07);
        }

        .decision-option-card--disabled {
          filter: grayscale(.25);
        }

        .decision-option-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
        }

        .decision-option-top span {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
          white-space: nowrap;
        }

        .decision-option-title {
          margin-top: 12px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 14px;
          line-height: 1.15;
          font-weight: 950;
          letter-spacing: .7px;
        }

        .decision-option-body {
          margin-top: 8px;
          color: #aebdca;
          font-size: 12px;
          line-height: 1.35;
          font-weight: 750;
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .decision-option-footer {
          margin-top: 12px;
          display: grid;
          gap: 9px;
        }

        .headline-ticker {
          margin-top: 18px;
          min-height: 52px;
          border: 1px solid var(--line);
          border-radius: 8px;
          overflow: hidden;
          display: grid;
          grid-template-columns: 250px minmax(0, 1fr) 220px;
          background:
            linear-gradient(90deg, rgba(7, 24, 38, .98), rgba(4, 14, 24, .96));
          box-shadow: var(--soft-shadow);
        }

        .ticker-label {
          display: flex;
          align-items: center;
          padding: 0 26px;
          color: var(--accent);
          text-transform: uppercase;
          font-size: 18px;
          font-style: italic;
          letter-spacing: 1.5px;
          font-weight: 950;
          background: linear-gradient(90deg, rgba(22,214,255,.1), transparent);
        }

        .ticker-items {
          min-width: 0;
          display: flex;
          align-items: center;
          gap: 26px;
          overflow: hidden;
          color: #becbd6;
          font-size: 13px;
          font-weight: 780;
          white-space: nowrap;
          padding: 0 14px;
        }

        .ticker-item {
          position: relative;
          border: 0;
          background: transparent;
          color: #becbd6;
          font: inherit;
          cursor: pointer;
          white-space: nowrap;
          text-align: left;
          padding: 0;
        }

        .ticker-item::before {
          content: "•";
          color: var(--accent);
          margin-right: 18px;
        }

        .ticker-item:hover {
          color: white;
        }

        .ticker-placeholder {
          color: var(--muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: .8px;
          font-weight: 900;
        }

        .ticker-action {
          border: 0;
          border-left: 1px solid var(--line);
          background: rgba(22,214,255,.04);
          color: var(--accent);
          text-transform: uppercase;
          font-size: 12px;
          letter-spacing: 1px;
          font-weight: 950;
          cursor: pointer;
        }

        .ticker-action:hover {
          background: rgba(22,214,255,.08);
          color: white;
        }

        .story-modal-backdrop {
          position: fixed;
          inset: 0;
          z-index: 80;
          background:
            radial-gradient(circle at 50% 0%, rgba(22,214,255,.12), transparent 38%),
            rgba(0, 0, 0, .72);
          backdrop-filter: blur(10px);
          display: grid;
          place-items: center;
          padding: 24px;
        }

        .story-modal {
          width: min(1280px, 100%);
          max-height: min(860px, calc(100vh - 48px));
          overflow: auto;
          border: 1px solid rgba(22,214,255,.28);
          border-radius: 14px;
          background:
            radial-gradient(circle at 88% 0%, rgba(22,214,255,.1), transparent 32%),
            linear-gradient(180deg, rgba(8, 25, 39, .98), rgba(4, 14, 24, .98));
          box-shadow:
            0 40px 120px rgba(0,0,0,.6),
            0 0 40px rgba(22,214,255,.08);
        }

        .story-modal--negative {
          border-color: rgba(255,90,103,.34);
          background:
            radial-gradient(circle at 88% 0%, rgba(255,90,103,.12), transparent 32%),
            linear-gradient(180deg, rgba(35, 13, 24, .98), rgba(4, 14, 24, .98));
        }

        .story-modal--positive {
          border-color: rgba(66,224,121,.28);
          background:
            radial-gradient(circle at 88% 0%, rgba(66,224,121,.1), transparent 32%),
            linear-gradient(180deg, rgba(12, 34, 25, .98), rgba(4, 14, 24, .98));
        }

        .story-modal--rumor {
          border-color: rgba(165,108,255,.34);
          background:
            radial-gradient(circle at 88% 0%, rgba(165,108,255,.12), transparent 32%),
            linear-gradient(180deg, rgba(26, 16, 42, .98), rgba(4, 14, 24, .98));
        }

        .story-modal-header {
          padding: 22px 22px 18px;
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 18px;
          border-bottom: 1px solid rgba(133,190,225,.16);
        }

        .story-modal-title-block {
          min-width: 0;
        }

        .story-modal-title-block h2 {
          margin: 0;
          color: var(--text);
          text-transform: uppercase;
          font-size: 34px;
          line-height: 1.05;
          font-weight: 950;
          font-style: italic;
          letter-spacing: 1.5px;
        }

        .story-modal-close {
          width: 42px;
          height: 42px;
          border-radius: 12px;
          border: 1px solid rgba(133,190,225,.18);
          background: rgba(255,255,255,.045);
          color: var(--text);
          font-size: 28px;
          line-height: 1;
          cursor: pointer;
          flex: 0 0 auto;
        }

        .story-modal-close:hover {
          border-color: rgba(255,90,103,.42);
          color: var(--red);
          background: rgba(255,90,103,.08);
        }

        .story-modal-body {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 260px;
          gap: 18px;
          padding: 20px 22px;
        }

        .story-modal-main {
          min-width: 0;
        }

        .story-modal-lede {
          margin: 0;
          color: #d4e2ee;
          font-size: 17px;
          line-height: 1.55;
          font-weight: 720;
        }

        .story-detail-block {
          margin-top: 16px;
          border: 1px solid rgba(133,190,225,.16);
          background: rgba(255,255,255,.035);
          border-radius: 12px;
          padding: 14px;
        }

        .story-detail-block h4 {
          margin: 0;
          color: var(--text);
          text-transform: uppercase;
          font-size: 13px;
          letter-spacing: 1px;
          font-weight: 950;
        }

        .story-detail-block p {
          margin: 8px 0 0;
          color: #aebdca;
          font-size: 13px;
          line-height: 1.45;
          font-weight: 750;
        }

        .story-modal-side {
          display: grid;
          gap: 14px;
          align-content: start;
        }

        .story-modal-facts {
          border: 1px solid rgba(133,190,225,.16);
          background: rgba(255,255,255,.035);
          border-radius: 12px;
          overflow: hidden;
        }

        .story-modal-facts div {
          padding: 12px;
          border-bottom: 1px solid rgba(133,190,225,.12);
        }

        .story-modal-facts div:last-child {
          border-bottom: 0;
        }

        .story-modal-facts span {
          display: block;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: .8px;
          font-weight: 950;
        }

        .story-modal-facts strong {
          display: block;
          margin-top: 6px;
          color: var(--text);
          text-transform: uppercase;
          font-size: 13px;
          line-height: 1.2;
          font-weight: 950;
        }

        .story-modal-decisions {
          padding: 0 22px 22px;
        }        .spark-bars {
          height: 78px;
          display: flex;
          align-items: end;
          gap: 5px;
          padding: 8px;
          border: 1px solid rgba(133,190,225,.14);
          background: rgba(255,255,255,.035);
          border-radius: 10px;
        }

        .spark-bar-wrap {
          flex: 1;
          height: 100%;
          display: flex;
          align-items: end;
        }

        .spark-bar {
          width: 100%;
          min-height: 8px;
          border-radius: 999px 999px 3px 3px;
          background: var(--accent);
          opacity: .82;
          box-shadow: 0 0 12px rgba(22,214,255,.16);
        }

        .spark-bars--positive .spark-bar {
          background: var(--green);
          box-shadow: 0 0 12px rgba(66,224,121,.16);
        }

        .spark-bars--negative .spark-bar {
          background: var(--red);
          box-shadow: 0 0 12px rgba(255,90,103,.16);
        }

        .spark-bars--warning .spark-bar {
          background: var(--gold);
          box-shadow: 0 0 12px rgba(245,185,46,.16);
        }

        .spark-bars--rumor .spark-bar {
          background: var(--purple);
          box-shadow: 0 0 12px rgba(165,108,255,.16);
        }

        .national-desk-panel {
          min-height: 520px;
        }

        .team-feed-grid .story-list-panel,
        .decisions-command-grid .story-list-panel,
        .injury-command-grid .story-list-panel {
          min-height: 430px;
        }

        .medical-fallout-panel .story-list {
          min-height: 360px;
        }

        .depth-risk-panel .radial-gauge-ring,
        .injury-desk-panel .radial-gauge-ring {
          width: 132px;
          height: 132px;
        }

        .depth-risk-panel .radial-gauge-inner,
        .injury-desk-panel .radial-gauge-inner {
          width: 104px;
          height: 104px;
        }

        .depth-risk-panel .radial-gauge-inner strong,
        .injury-desk-panel .radial-gauge-inner strong {
          font-size: 30px;
        }

        .story-modal .decision-option-grid {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .story-modal .decision-option-card {
          min-height: 170px;
        }

        .story-modal .affected-row {
          margin-top: 16px;
        }

        .story-modal .effect-list {
          margin-top: 10px;
        }

        .story-modal .mini-badge {
          vertical-align: middle;
        }

        .story-modal::-webkit-scrollbar,
        .storylines-screen::-webkit-scrollbar {
          width: 10px;
          height: 10px;
        }

        .story-modal::-webkit-scrollbar-track,
        .storylines-screen::-webkit-scrollbar-track {
          background: rgba(255,255,255,.03);
        }

        .story-modal::-webkit-scrollbar-thumb,
        .storylines-screen::-webkit-scrollbar-thumb {
          background: rgba(133,190,225,.22);
          border-radius: 999px;
        }

        .story-modal::-webkit-scrollbar-thumb:hover,
        .storylines-screen::-webkit-scrollbar-thumb:hover {
          background: rgba(22,214,255,.34);
        }

        .compact-story-row:focus-visible,
        .large-story-card:focus-visible,
        .trending-row:focus-visible,
        .decision-option-card:focus-visible,
        .nav-btn:focus-visible,
        .story-tab:focus-visible,
        .continue-btn:focus-visible,
        .view-story-btn:focus-visible,
        .panel-action-btn:focus-visible,
        .ticker-item:focus-visible,
        .story-modal-close:focus-visible {
          outline: 2px solid rgba(22,214,255,.75);
          outline-offset: 2px;
        }

        .compact-story-row,
        .large-story-card,
        .trending-row,
        .decision-option-card,
        .nav-btn,
        .story-tab,
        .continue-btn,
        .view-story-btn,
        .panel-action-btn,
        .ticker-item,
        .story-modal-close,
        .decision-open-story,
        .ticker-action {
          -webkit-tap-highlight-color: transparent;
        }

        @media (max-width: 1700px) {
          .story-topbar {
            grid-template-columns: minmax(280px, 1fr) 170px 190px 205px 250px 235px 210px;
          }

          .brand-title {
            font-size: 22px;
          }

          .story-meter-text span {
            font-size: 12px;
          }

          .topbar-value {
            font-size: 13px;
          }

          .hero-title {
            font-size: 33px;
          }

          .hero-proof-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            max-width: 520px;
          }

          .lower-grid--overview {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }

          .morale-panel {
            grid-column: span 2;
          }

          .analysis-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 1500px) {
          .story-topbar {
            grid-template-columns: minmax(280px, 1fr) 160px 180px 190px 220px 230px;
          }

          .story-topbar .topbar-cell:nth-last-child(1) {
            display: none;
          }

          .story-grid {
            grid-template-columns: minmax(0, 1.3fr) minmax(380px, .9fr);
          }

          .hero-player-ghost {
            right: -15px;
            opacity: .7;
          }

          .league-command-grid,
          .team-command-grid,
          .injury-command-grid,
          .decisions-command-grid,
          .team-feed-grid,
          .injury-analysis-grid {
            grid-template-columns: 1fr;
          }

          .triple-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }

          .triple-grid .story-list-panel:last-child {
            grid-column: span 2;
          }

          .large-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 1250px) {
          .story-topbar {
            position: relative;
            grid-template-columns: minmax(320px, 1fr) 1fr 1fr;
          }

          .franchise-brand {
            grid-row: span 2;
          }

          .topbar-cell {
            border-top: 1px solid var(--line);
          }

          .story-grid {
            grid-template-columns: 1fr;
          }

          .trending-panel {
            min-height: auto;
          }

          .trending-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            padding: 12px;
          }

          .trending-row {
            border: 1px solid rgba(133,190,225,.13);
            border-radius: 10px;
            padding: 8px;
            min-height: 76px;
          }

          .trending-row:last-child {
            border-bottom: 1px solid rgba(133,190,225,.13);
          }

          .hero-decision-options,
          .decision-option-grid,
          .story-modal .decision-option-grid {
            grid-template-columns: 1fr;
          }

          .decision-context-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }

          .story-modal-body {
            grid-template-columns: 1fr;
          }

          .story-modal-side {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }

          .story-modal-side .radial-gauge-ring {
            width: 126px;
            height: 126px;
          }

          .story-modal-side .radial-gauge-inner {
            width: 98px;
            height: 98px;
          }
        }

        @media (max-width: 1050px) {
          .story-main {
            padding: 14px;
          }

          .story-topbar {
            grid-template-columns: 1fr 1fr;
          }

          .franchise-brand {
            grid-column: span 2;
            grid-row: auto;
          }

          .topbar-cell {
            min-height: 82px;
            height: auto;
            justify-content: flex-start;
          }

          .story-header {
            align-items: flex-start;
            flex-direction: column;
          }

          .header-actions {
            justify-content: flex-start;
          }

          .lower-grid--overview,
          .triple-grid,
          .large-card-grid,
          .consequence-stat-stack,
          .season-pressure-main,
          .schedule-pressure-facts,
          .situation-room-cards,
          .response-suggestion-grid {
            grid-template-columns: 1fr;
          }

          .triple-grid .story-list-panel:last-child,
          .morale-panel {
            grid-column: auto;
          }

          .consequence-grid,
          .situation-room-grid,
          .injury-summary-row,
          .depth-risk-body {
            grid-template-columns: 1fr;
            text-align: left;
          }

          .consequence-grid .radial-gauge,
          .situation-room-grid .radial-gauge,
          .injury-summary-row .radial-gauge,
          .depth-risk-body .radial-gauge {
            justify-self: center;
          }

          .consequence-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }

          .consequence-strip div:nth-child(2) {
            border-right: 0;
          }

          .headline-ticker {
            grid-template-columns: 1fr;
          }

          .ticker-label,
          .ticker-action {
            min-height: 44px;
          }

          .ticker-action {
            border-left: 0;
            border-top: 1px solid var(--line);
          }

          .ticker-items {
            min-height: 48px;
            overflow-x: auto;
          }

          .story-modal-title-block h2 {
            font-size: 28px;
          }
        }

        @media (max-width: 820px) {
          .storylines-screen {
            padding-bottom: 14px;
          }

          .story-main {
            padding: 12px;
          }

          .story-topbar {
            grid-template-columns: 1fr;
          }

          .franchise-brand {
            grid-column: auto;
            padding: 14px;
          }

          .team-shield--lg {
            width: 62px;
            height: 62px;
            font-size: 17px;
          }

          .brand-title {
            font-size: 20px;
          }

          .brand-sub,
          .brand-kicker {
            font-size: 11px;
          }

          .topbar-cell {
            border-left: 0;
            padding: 12px 14px;
          }

          .topbar-cell--continue {
            justify-content: center;
          }

          .continue-btn {
            width: 100%;
          }

          .page-title {
            font-size: 36px;
            letter-spacing: 2px;
          }

          .page-subtitle {
            font-size: 11px;
            line-height: 1.3;
          }

          .story-tabs {
            gap: 6px;
          }

          .story-tab {
            height: 32px;
            padding: 0 10px;
            font-size: 10px;
          }

          .hero-story {
            min-height: auto;
          }

          .hero-content {
            padding: 20px 16px;
          }

          .hero-title {
            font-size: 26px;
            letter-spacing: 1px;
          }

          .hero-body {
            font-size: 14px;
          }

          .hero-proof-grid {
            grid-template-columns: 1fr 1fr;
          }

          .hero-player-ghost {
            opacity: .18;
            right: -120px;
            bottom: -70px;
            transform: scale(.9);
          }

          .hero-bottom {
            gap: 12px;
            align-items: stretch;
            flex-direction: column;
          }

          .view-story-btn,
          .panel-action-btn {
            width: 100%;
          }

          .hero-impact {
            justify-content: space-between;
            border: 1px solid rgba(133,190,225,.14);
            background: rgba(255,255,255,.035);
            border-radius: 8px;
            padding: 10px 12px;
          }

          .trending-list {
            grid-template-columns: 1fr;
          }

          .trending-row {
            grid-template-columns: 30px 44px minmax(0, 1fr);
          }

          .trend-badge,
          .trend-arrow {
            display: none;
          }

          .compact-story-row {
            grid-template-columns: 42px minmax(0, 1fr);
          }

          .compact-story-side {
            grid-column: 2;
            grid-row: 2;
            display: flex;
            justify-content: space-between;
            width: 100%;
            text-align: left;
          }

          .story-icon {
            width: 36px;
            height: 36px;
            font-size: 18px;
          }

          .morale-layout {
            grid-template-columns: 1fr;
          }

          .morale-scorebox {
            border-left: 0;
            border-top: 1px solid rgba(133, 190, 225, .16);
            min-height: 120px;
          }

          .decision-card-header {
            flex-direction: column;
          }

          .decision-open-story {
            width: 100%;
          }

          .decision-context-grid {
            grid-template-columns: 1fr;
          }

          .pressure-player-row,
          .injury-row {
            grid-template-columns: 42px minmax(0, 1fr);
          }

          .pressure-player-meter,
          .injury-return {
            grid-column: 2;
          }

          .injury-return {
            text-align: left;
            justify-items: start;
            grid-template-columns: repeat(3, auto);
            gap: 8px;
            align-items: center;
          }

          .story-modal-backdrop {
            padding: 10px;
          }

          .story-modal {
            max-height: calc(100vh - 20px);
            border-radius: 10px;
          }

          .story-modal-header {
            padding: 16px;
          }

          .story-modal-body,
          .story-modal-decisions {
            padding-left: 16px;
            padding-right: 16px;
          }

          .story-modal-title-block h2 {
            font-size: 24px;
          }

          .story-modal-side {
            grid-template-columns: 1fr;
          }

          .story-modal-side .radial-gauge-ring {
            width: 138px;
            height: 138px;
          }

          .story-modal-side .radial-gauge-inner {
            width: 108px;
            height: 108px;
          }
        }

        @media (max-width: 560px) {
          .story-main {
            padding: 10px;
          }

          .franchise-brand {
            gap: 12px;
          }

          .team-shield--lg {
            width: 54px;
            height: 54px;
            font-size: 15px;
          }

          .brand-title {
            font-size: 17px;
          }

          .story-meter-ring {
            width: 44px !important;
            height: 44px !important;
          }

          .story-meter-inner {
            font-size: 12px;
          }

          .topbar-icon {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            font-size: 21px;
          }

          .page-title {
            font-size: 31px;
          }

          .hero-proof-grid {
            grid-template-columns: 1fr;
          }

          .hero-tags {
            gap: 8px;
          }

          .hero-tag,
          .mini-badge {
            font-size: 9px;
          }

          .hero-title {
            font-size: 22px;
          }

          .hero-body,
          .story-modal-lede {
            font-size: 13px;
          }

          .compact-story-headline,
          .injury-player,
          .pressure-player-name,
          .trend-name {
            font-size: 12px;
          }

          .compact-story-body,
          .injury-detail,
          .injury-impact,
          .pressure-player-sub,
          .trend-sub {
            font-size: 11px;
          }

          .section-title-row {
            align-items: flex-start;
            flex-direction: column;
          }

          .section-title-actions {
            justify-content: flex-start;
          }

          .consequence-strip {
            grid-template-columns: 1fr;
          }

          .consequence-strip div {
            border-right: 0;
            border-bottom: 1px solid rgba(133,190,225,.12);
          }

          .consequence-strip div:last-child {
            border-bottom: 0;
          }

          .radial-gauge-ring {
            width: 126px;
            height: 126px;
          }

          .radial-gauge-inner {
            width: 98px;
            height: 98px;
          }

          .radial-gauge-inner strong {
            font-size: 28px;
          }

          .season-pressure-record strong,
          .stat-chip-value {
            font-size: 19px;
          }

          .story-modal-title-block h2 {
            font-size: 20px;
          }

          .story-modal-close {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            font-size: 24px;
          }

          .ticker-label {
            font-size: 15px;
            padding: 0 16px;
          }

          .ticker-items {
            gap: 18px;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .team-shield::after {
            animation: none;
          }

          .continue-btn,
          .nav-btn,
          .story-tab,
          .view-story-btn,
          .panel-action-btn,
          .trending-row,
          .compact-story-row,
          .large-story-card,
          .decision-option-card {
            transition: none;
          }

          .continue-btn:hover,
          .nav-btn:hover,
          .story-tab:hover,
          .view-story-btn:hover,
          .panel-action-btn:hover,
          .trending-row:hover,
          .large-story-card:hover,
          .decision-option-card:hover {
            transform: none;
          }
        }

        @media print {
          .story-topbar,
          .header-actions,
          .story-tabs,
          .continue-btn,
          .ticker-action,
          .story-modal-backdrop,
          .game-footer {
            display: none !important;
          }

          .storylines-screen {
            background: #fff;
            color: #000;
          }

          .section-shell,
          .hero-story,
          .headline-ticker {
            box-shadow: none;
            border-color: #ccc;
            background: #fff;
            color: #000;
          }
        }
          }
        `}</style>
  
        <StoryTopbar
          franchiseState={franchiseState}
          gmRating={gmRating}
          chemistry={chemistry}
          fanConfidence={fanConfidence}
          morale={morale}
          setScreen={setScreen}
        />
  
        <main className="story-main">
          <PageHeader
            setScreen={setScreen}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
          />
  
          {renderTab()}
  
          <HeadlineTicker
            headlines={grouped.headlines}
            onSelectStory={handleOpenFullStory}
          />
        </main>
  
        {modalStory ? (
          <FullStoryModal
            story={modalStory}
            choices={selectedStoryChoices}
            onClose={handleCloseModal}
            onResolveStorylineChoice={onResolveStorylineChoice}
          />
        ) : null}
  
        <GameFooter />
      </div>
    );
  }
  
  /* ============================================================
     END OF CHUNK 3
     Next chunk starts with:
     - remaining CSS styles
     - cards, rows, modals, responsive styles
     - final closing cleanup
     ============================================================ */
