import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { enrichRosterPlayer } from "../game/rosterColumns";
import { GameFooter } from "../components/game/GameFooter";

/**
 * RosterScreen
 *
 * Full in-file franchise roster hub.
 * No new files required.
 * Designed to be resilient against incomplete backend payloads.
 */

const EMPTY_ARRAY = [];
const EMPTY_OBJECT = Object.freeze({});

const SORT_KEYS = [
  { value: "overall_desc", label: "Overall ↓" },
  { value: "overall_asc", label: "Overall ↑" },
  { value: "age_desc", label: "Age ↓" },
  { value: "age_asc", label: "Age ↑" },
  { value: "name_asc", label: "Name A–Z" },
  { value: "name_desc", label: "Name Z–A" },
  { value: "potential_desc", label: "Potential ↓" },
  { value: "morale_desc", label: "Morale ↓" },
  { value: "fatigue_desc", label: "Fatigue ↓" },
  { value: "points_desc", label: "Points ↓" },
  { value: "goals_desc", label: "Goals ↓" },
  { value: "assists_desc", label: "Assists ↓" },
  { value: "salary_desc", label: "Cap Hit ↓" },
  { value: "term_desc", label: "Term ↓" },
];

const POSITION_FILTERS = [
  "ALL",
  "F",
  "C",
  "LW",
  "RW",
  "D",
  "G",
];

const PANEL_TABS = [
  { value: "overview", label: "Player Overview" },
  { value: "stats", label: "Stats" },
  { value: "attributes", label: "Attributes" },
  { value: "contract", label: "Contract" },
  { value: "development", label: "Development" },
  { value: "history", label: "History" },
  { value: "notes", label: "Notes" },
];

const TABLE_PAGE_SIZE = 14;

const NHL_CONTRACT_LIMIT = 23;
const DEFAULT_CAP_LIMIT = 83.5;

const POTENTIAL_ORDER = {
  Franchise: 100,
  Elite: 90,
  "Top 6": 80,
  "Top 4": 80,
  "Middle 6": 70,
  "Top 6 D": 70,
  "Bottom 6": 60,
  Depth: 50,
  AHL: 40,
  "AHL / Depth": 40,
  Unknown: 0,
  "—": 0,
};

const DEFAULT_RATING_GROUPS = [
  { title: "OFFENSE", rows: [] },
  { title: "DEFENSE", rows: [] },
  { title: "SKATING", rows: [] },
  { title: "PHYSICAL", rows: [] },
  { title: "MENTAL", rows: [] },
];

const LEAGUE_FILTERS = ["ALL", "NHL", "AHL", "ECHL"];
const STATUS_FILTERS = ["All", "Active", "Injured", "Scratched"];

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function safeStr(v, fallback = "—") {
  if (v === null || v === undefined || v === "") return fallback;
  return String(v);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function calculatePotential(p) {
  const age = safeNum(p?.age, 0);
  const ovr = safeNum(p?.ovr, safeNum(p?.overall, 0));
  if (age <= 21 && ovr >= 78) return "Elite";
  if (age <= 23 && ovr >= 74) return "Top 6";
  if (ovr >= 84) return "Top 6";
  if (ovr >= 79) return "Middle 6";
  if (ovr >= 74) return "Bottom 6";
  return "Depth";
}

function getDevelopmentStage(age) {
  if (age <= 22) return "Prospect";
  if (age <= 27) return "Prime";
  if (age <= 32) return "Veteran";
  return "Decline";
}

function getMoraleColor(morale) {
  const m = safeNum(morale, 0);
  if (m > 0.65 || m > 65) return "#4CAF50";
  if (m > 0.5 || m > 50) return "#FFC107";
  return "#F44336";
}

function formatRole(role) {
  const r = safeStr(role, "Depth");
  return r
    .replace(/_/g, " ")
    .replace(/\b(top|bottom|middle)\s*(\d)\b/gi, (_, a, b) => `${a[0].toUpperCase()}${a.slice(1)} ${b}`)
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getRosterStatus(p) {
  const gr = safeNum(
    pickFirstDefined(p?.injury_games_remaining, p?.games_remaining, p?.days_remaining),
    0
  );
  if (p?.is_injured === true || gr > 0) return "Injured";
  const injury = safeStr(p?.injury ?? p?.injury_status, "Healthy").toLowerCase();
  if (injury !== "healthy" || p?.injured) return "Injured";
  if (p?.scratched) return "Scratched";
  if (safeStr(p?.league, "NHL").toUpperCase() !== "NHL") return "Assigned";
  return "Active";
}

function getOVRColor(ovr) {
  const n = safeNum(ovr, 0);
  if (n >= 90) return "#f8d26a";
  if (n >= 85) return "#72b3ff";
  if (n >= 80) return "#eef4ff";
  return "#98a6ba";
}

function getArchetypeColor(archetype) {
  const a = safeStr(archetype, "").toUpperCase();
  if (a.includes("SNIPER")) return "#e35d5b";
  if (a.includes("PLAYMAKER")) return "#5ba9e3";
  if (a.includes("POWER")) return "#e39a45";
  if (a.includes("DEF")) return "#5cc084";
  return "#7f8aa3";
}

function getAssetValue(p) {
  const age = safeNum(p.age, 0);
  const ovr = safeNum(p.ovr, 0);
  const pot = getPotentialRank(p.potential);
  const growth = safeNum(p.growth, 0);
  if (ovr >= 87 && age <= 30) return "Elite Core";
  if (pot >= 80 && age <= 23) return "Top Prospect";
  if (ovr >= 80 && age >= 25 && growth <= 0.2) return "Trade Piece";
  if (age >= 31 || growth < -0.4) return "Declining";
  return "Depth";
}

function groupIntoLines(list) {
  const sorted = [...(list || [])].sort((a, b) => safeNum(b.ovr, 0) - safeNum(a.ovr, 0));
  const lines = [];
  for (let i = 0; i < sorted.length; i += 3) lines.push(sorted.slice(i, i + 3));
  return lines;
}

function groupIntoPairs(list) {
  const sorted = [...(list || [])].sort((a, b) => safeNum(b.ovr, 0) - safeNum(a.ovr, 0));
  const pairs = [];
  for (let i = 0; i < sorted.length; i += 2) pairs.push(sorted.slice(i, i + 2));
  return pairs;
}

const formatPlayer = (p) => {
  const normalizedMorale = safeNum(p.morale, 0) > 1 ? safeNum(p.morale, 0) / 100 : safeNum(p.morale, 0);
  const ovr = Math.round(safeNum(p.ovr, 0));
  const potential = safeStr(p.potential, "").toLowerCase() === "unknown" ? calculatePotential(p) : safeStr(p.potential, calculatePotential(p));
  const roleLabel = formatRole(p.role);
  const league = safeStr(p.league, "NHL").toUpperCase();
  const status = getRosterStatus({ ...p, league });
  return {
    ...p,
    ovr,
    potential,
    archetype: safeStr(p.archetype, "BALANCED"),
    age: safeNum(p.age, 0),
    stage: getDevelopmentStage(safeNum(p.age, 0)),
    moraleColor: getMoraleColor(normalizedMorale),
    roleLabel,
    league,
    status,
    assetTag: getAssetValue({ ...p, potential, ovr }),
    displayOVR: `${ovr} (${potential})`,
  };
};

function initialsFromName(name) {
  const raw = safeStr(name, "").trim();
  if (!raw) return "—";
  const parts = raw.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0] || ""}${parts[parts.length - 1][0] || ""}`.toUpperCase();
}

function formatMoneyMillions(v) {
  const n = safeNum(v, 0);
  if (!n) return "$0.00M";
  return `$${n.toFixed(2)}M`;
}

function formatSignedNumber(v) {
  const n = safeNum(v, 0);
  if (n > 0) return `+${n.toFixed(1)}`;
  if (n < 0) return `${n.toFixed(1)}`;
  return "0.0";
}

function normalizePosition(pos) {
  const p = safeStr(pos, "—").toUpperCase();
  if (["C", "LW", "RW", "F", "LD", "RD", "D", "G"].includes(p)) return p;
  if (p.includes("LW")) return "LW";
  if (p.includes("RW")) return "RW";
  if (p.includes("C")) return "C";
  if (p.includes("D")) return "D";
  if (p.includes("G")) return "G";
  return p;
}

function positionMatchesFilter(pos, filter) {
  if (filter === "ALL") return true;
  const p = normalizePosition(pos);
  if (filter === "F") return p === "C" || p === "LW" || p === "RW" || p === "F";
  return p === filter;
}

function getPotentialRank(potential) {
  return POTENTIAL_ORDER[safeStr(potential, "Depth")] ?? 0;
}

function getTrendMeta(trend, rankDelta) {
  const t = safeStr(trend, "SAME").toUpperCase();
  if (t === "UP") return { text: `▲${Math.abs(safeNum(rankDelta, 0)) || ""}`, cls: "is-up" };
  if (t === "DOWN") return { text: `▼${Math.abs(safeNum(rankDelta, 0)) || ""}`, cls: "is-down" };
  if (t === "NEW") return { text: "NEW", cls: "is-new" };
  return { text: "—", cls: "is-flat" };
}

function getMoraleBand(morale) {
  const m = safeNum(morale, 50);
  if (m >= 85) return { label: "Excellent", tone: "good" };
  if (m >= 70) return { label: "Good", tone: "good" };
  if (m >= 55) return { label: "Stable", tone: "neutral" };
  if (m >= 40) return { label: "Shaky", tone: "warn" };
  return { label: "Poor", tone: "bad" };
}

function getFatigueBand(fatigue) {
  const f = safeNum(fatigue, 0);
  if (f <= 15) return { label: "Fresh", tone: "good" };
  if (f <= 35) return { label: "Managed", tone: "neutral" };
  if (f <= 60) return { label: "Heavy", tone: "warn" };
  return { label: "Exhausted", tone: "bad" };
}

function getInjuryBand(status) {
  const s = safeStr(status, "Healthy");
  if (s.toLowerCase() === "healthy") return { label: "Healthy", tone: "good" };
  if (s.toLowerCase().includes("day")) return { label: s, tone: "warn" };
  return { label: s, tone: "bad" };
}

function getDevelopmentBand(delta) {
  const d = safeNum(delta, 0);
  if (d >= 1.5) return { label: "Strong Rise", tone: "good" };
  if (d > 0.2) return { label: "Trending Up", tone: "good" };
  if (d <= -1.0) return { label: "Dropping", tone: "bad" };
  if (d < -0.2) return { label: "Slight Dip", tone: "warn" };
  return { label: "Steady", tone: "neutral" };
}

function toneClass(tone) {
  if (tone === "good") return "is-good";
  if (tone === "warn") return "is-warn";
  if (tone === "bad") return "is-bad";
  return "is-neutral";
}

function pickFirstDefined(...vals) {
  for (const v of vals) {
    if (v !== undefined && v !== null && v !== "") return v;
  }
  return undefined;
}

function normalizeContract(player) {
  const contract = player?.contract || EMPTY_OBJECT;
  const salary = safeNum(
    pickFirstDefined(contract.salary, contract.cap_hit, player.salary, player.cap_hit),
    0
  );
  const term = safeNum(
    pickFirstDefined(contract.term, contract.years, player.contract_term, player.term),
    0
  );
  const type = safeStr(
    pickFirstDefined(contract.contract_type, contract.type, player.contract_type),
    "Standard"
  );
  const expiry = safeStr(
    pickFirstDefined(contract.expiry, player.contract_expiry),
    term > 0 ? `${term}Y remaining` : "Unsigned"
  );

  return {
    salary,
    term,
    type,
    expiry,
  };
}

function normalizeSeasonStats(player) {
  const s = player?.season_stats || player?.stats || EMPTY_OBJECT;
  return {
    gp: safeNum(pickFirstDefined(s.gp, s.games_played), 0),
    g: safeNum(pickFirstDefined(s.g, s.goals), 0),
    a: safeNum(pickFirstDefined(s.a, s.assists), 0),
    pts: safeNum(pickFirstDefined(s.pts, s.points), 0),
    pim: safeNum(pickFirstDefined(s.pim, s.penalty_minutes), 0),
    plusMinus: safeNum(pickFirstDefined(s.plus_minus, s.pm), 0),
    svPct: safeNum(pickFirstDefined(s.sv_pct, s.save_pct), 0),
    gaa: safeNum(pickFirstDefined(s.gaa), 0),
    wins: safeNum(pickFirstDefined(s.wins), 0),
    losses: safeNum(pickFirstDefined(s.losses), 0),
    otl: safeNum(pickFirstDefined(s.otl), 0),
  };
}

function inferRole(player) {
  return safeStr(
    pickFirstDefined(
      player.line_role,
      player.role,
      player.roster_role,
      player.depth_role
    ),
    "Depth"
  );
}

function inferSpecialTeams(player) {
  return safeStr(
    pickFirstDefined(
      player.special_teams,
      player.special_teams_role,
      player.st_role
    ),
    "Even Strength"
  );
}

function inferMinutes(player) {
  return safeNum(
    pickFirstDefined(player.minutes, player.toi, player.average_toi, player.ice_time),
    0
  );
}

function inferPotential(player) {
  return safeStr(
    pickFirstDefined(player.potential, player.potential_tier, player.ceiling),
    calculatePotential(player)
  );
}

function inferArchetype(player) {
  return safeStr(
    pickFirstDefined(player.archetype, player.player_type, player.style),
    "BALANCED"
  );
}

function inferDevCurve(player) {
  return safeStr(
    pickFirstDefined(player.development_curve, player.dev_curve, player.progression_type),
    "Normal"
  );
}

function inferMorale(player) {
  return clamp(safeNum(player.morale, 55), 0, 100);
}

function inferFatigue(player) {
  return clamp(safeNum(player.fatigue, 0), 0, 100);
}

function inferInjury(player) {
  const gr = Math.max(
    0,
    safeNum(
      pickFirstDefined(player.injury_games_remaining, player.games_remaining, player.days_remaining),
      0
    )
  );
  const isInj = player?.is_injured === true || gr > 0;
  const st = safeStr(
    pickFirstDefined(player.injury_status, player.health_status, player.status),
    ""
  ).toUpperCase();
  if (!isInj && (st === "HEALTHY" || st === "")) {
    return "Healthy";
  }
  if (st.includes("DAY")) {
    return gr > 0 ? `Day-to-day · ${gr}g` : "Day-to-day";
  }
  if (gr > 0) {
    const lab = safeStr(player.injury, "");
    if (lab) return `${lab} (${gr}g)`;
    const tier = safeStr(pickFirstDefined(player.injury_tier, player.injury_type), "");
    if (tier) return `Out (${tier}) · ${gr}g`;
    return `Out · ${gr} games`;
  }
  return safeStr(
    pickFirstDefined(player.availability_status, player.injury, player.injury_status),
    "Injured"
  );
}

function inferGrowth(player) {
  return safeNum(
    pickFirstDefined(player.growth_delta, player.dev_delta, player.overall_delta, player.delta_ovr),
    0
  );
}

function inferUsage(player) {
  return {
    role: inferRole(player),
    specialTeams: inferSpecialTeams(player),
    minutes: inferMinutes(player),
  };
}

function inferTeamName(player, franchiseState) {
  return safeStr(
    pickFirstDefined(player.team_name, player.team, franchiseState?.team?.name),
    "—"
  );
}

function inferLeague(player) {
  return safeStr(
    pickFirstDefined(player.league, player.league_name, player.league_code),
    "NHL"
  );
}

function inferHandedness(player) {
  return safeStr(
    pickFirstDefined(player.handedness, player.hand, player.shoots),
    "—"
  );
}

function inferNationality(player) {
  return safeStr(
    pickFirstDefined(player.nationality, player.nation, player.country, player.nat),
    "—"
  );
}

function inferHeight(player) {
  return safeStr(
    pickFirstDefined(player.height, player.hgt),
    "—"
  );
}

function inferWeight(player) {
  return safeStr(
    pickFirstDefined(player.weight, player.wgt),
    "—"
  );
}

function normalizeRatingGroups(player) {
  if (Array.isArray(player?.rating_groups) && player.rating_groups.length > 0) {
    return player.rating_groups.map((group) => ({
      title: safeStr(group?.title, "ATTRIBUTES"),
      rows: Array.isArray(group?.rows)
        ? group.rows.map((row, idx) => ({
            id: safeStr(row?.id, `${safeStr(group?.title, "group")}-${idx}`),
            label: safeStr(row?.label, "—"),
            v: safeNum(row?.v, 0),
          }))
        : [],
    }));
  }

  const p = player || EMPTY_OBJECT;

  const offense = [
    { id: "shooting_accuracy", label: "Shooting Accuracy", v: safeNum(p.shooting_accuracy, safeNum(p.offense, 0)) },
    { id: "shooting_power", label: "Shooting Power", v: safeNum(p.shooting_power, safeNum(p.offense, 0)) },
    { id: "finishing", label: "Finishing", v: safeNum(p.finishing, safeNum(p.offense, 0)) },
    { id: "playmaking", label: "Playmaking", v: safeNum(p.playmaking, safeNum(p.offense, 0)) },
    { id: "puck_control", label: "Puck Control", v: safeNum(p.puck_control, safeNum(p.skill, 0)) },
    { id: "offensive_iq", label: "Offensive IQ", v: safeNum(p.offensive_iq, safeNum(p.iq, 0)) },
    { id: "creativity", label: "Creativity", v: safeNum(p.creativity, safeNum(p.skill, 0)) },
  ].filter((row) => row.v > 0);

  const defense = [
    { id: "defensive_iq", label: "Defensive IQ", v: safeNum(p.defensive_iq, safeNum(p.defense, 0)) },
    { id: "stick_checking", label: "Stick Checking", v: safeNum(p.stick_checking, safeNum(p.defense, 0)) },
    { id: "shot_blocking", label: "Shot Blocking", v: safeNum(p.shot_blocking, safeNum(p.defense, 0)) },
    { id: "positioning", label: "Positioning", v: safeNum(p.positioning, safeNum(p.defense, 0)) },
    { id: "backchecking", label: "Backchecking", v: safeNum(p.backchecking, safeNum(p.defense, 0)) },
  ].filter((row) => row.v > 0);

  const skating = [
    { id: "speed", label: "Speed", v: safeNum(p.speed, safeNum(p.skating, 0)) },
    { id: "acceleration", label: "Acceleration", v: safeNum(p.acceleration, safeNum(p.skating, 0)) },
    { id: "agility", label: "Agility", v: safeNum(p.agility, safeNum(p.skating, 0)) },
    { id: "balance", label: "Balance", v: safeNum(p.balance, safeNum(p.physical, 0)) },
    { id: "endurance", label: "Endurance", v: safeNum(p.endurance, safeNum(p.conditioning, 0)) },
  ].filter((row) => row.v > 0);

  const physical = [
    { id: "strength", label: "Strength", v: safeNum(p.strength, safeNum(p.physical, 0)) },
    { id: "checking", label: "Checking", v: safeNum(p.checking, safeNum(p.physical, 0)) },
    { id: "durability", label: "Durability", v: safeNum(p.durability, safeNum(p.physical, 0)) },
    { id: "grit", label: "Grit", v: safeNum(p.grit, safeNum(p.physical, 0)) },
  ].filter((row) => row.v > 0);

  const mental = [
    { id: "hockey_iq", label: "Hockey IQ", v: safeNum(p.hockey_iq, safeNum(p.iq, 0)) },
    { id: "consistency", label: "Consistency", v: safeNum(p.consistency, safeNum(p.mental, 0)) },
    { id: "clutch", label: "Clutch", v: safeNum(p.clutch, safeNum(p.mental, 0)) },
    { id: "discipline", label: "Discipline", v: safeNum(p.discipline, safeNum(p.mental, 0)) },
    { id: "leadership", label: "Leadership", v: safeNum(p.leadership, safeNum(p.mental, 0)) },
  ].filter((row) => row.v > 0);

  const groups = [
    { title: "OFFENSE", rows: offense },
    { title: "DEFENSE", rows: defense },
    { title: "SKATING", rows: skating },
    { title: "PHYSICAL", rows: physical },
    { title: "MENTAL", rows: mental },
  ];

  const hasAnyRows = groups.some((g) => g.rows.length > 0);
  return hasAnyRows ? groups : DEFAULT_RATING_GROUPS;
}

function averageGroup(group) {
  const rows = Array.isArray(group?.rows) ? group.rows : EMPTY_ARRAY;
  if (!rows.length) return 0;
  return rows.reduce((sum, row) => sum + safeNum(row.v, 0), 0) / rows.length;
}

function ratingSummary(groups) {
  const normalized = Array.isArray(groups) ? groups : EMPTY_ARRAY;
  const find = (title) => normalized.find((g) => safeStr(g.title, "").toUpperCase() === title);
  return {
    offense: averageGroup(find("OFFENSE")),
    defense: averageGroup(find("DEFENSE")),
    skating: averageGroup(find("SKATING")),
    physical: averageGroup(find("PHYSICAL")),
    mental: averageGroup(find("MENTAL")),
  };
}

function buildPlayerNote(player) {
  const parts = [];

  const age = safeNum(player.age, 0);
  const growth = safeNum(player.growth, 0);
  const fatigue = safeNum(player.fatigue, 0);
  const injury = safeStr(player.injury, "Healthy");
  const potential = safeStr(player.potential, "Projected");
  const role = safeStr(player.role, "Depth");
  const archetype = safeStr(player.archetype, "Balanced");

  if (potential === "Franchise" || potential === "Elite") {
    parts.push("High-end ceiling asset.");
  } else if (potential === "Top 6" || potential === "Top 4") {
    parts.push("Core lineup ceiling.");
  } else if (potential === "Bottom 6" || potential === "Depth") {
    parts.push("Support-role projection.");
  }

  if (growth >= 1.5) parts.push("Currently in a strong development window.");
  else if (growth > 0.2) parts.push("Trending upward.");
  else if (growth <= -1.0) parts.push("Noticeable regression risk.");
  else if (growth < -0.2) parts.push("Slight decline showing.");
  else parts.push("Development holding steady.");

  if (age <= 21) parts.push("Still in an early growth phase.");
  else if (age >= 30) parts.push("Aging curve should be monitored.");

  if (fatigue >= 60) parts.push("Fatigue is becoming a usage concern.");
  else if (fatigue >= 35) parts.push("Workload is starting to accumulate.");

  if (injury.toLowerCase() !== "healthy") parts.push("Health status is affecting availability.");

  if (role.toLowerCase().includes("1st") || role.toLowerCase().includes("top")) {
    parts.push("Handled as a primary deployment piece.");
  }

  if (archetype !== "Balanced") {
    parts.push(`${archetype} profile influences deployment and aging.`);
  }

  return parts.join(" ");
}

function normalizeLivePlayer(player, franchiseState, idx) {
  const enriched = enrichRosterPlayer(player, idx);

  const contract = normalizeContract(player);
  const season = normalizeSeasonStats(player);
  const usage = inferUsage(player);
  const groups = normalizeRatingGroups(player);

  const normalized = {
    ...enriched,
    _draft: false,
    key: pickFirstDefined(player.id, enriched.id, `${safeStr(enriched.name, "player")}-${idx}`),
    id: pickFirstDefined(player.id, enriched.id, idx),
    name: safeStr(pickFirstDefined(player.name, enriched.name), "Unnamed Player"),
    age: safeNum(pickFirstDefined(player.age, enriched.age), 18),
    position: normalizePosition(pickFirstDefined(player.position, enriched.position)),
    ovr: safeNum(pickFirstDefined(player.overall, player.ovr, enriched.ovr), 0),
    potential: inferPotential(player),
    archetype: inferArchetype(player),
    morale: inferMorale(player),
    fatigue: inferFatigue(player),
    injury: inferInjury(player),
    contract,
    role: usage.role,
    specialTeams: usage.specialTeams,
    minutes: usage.minutes,
    dev: inferDevCurve(player),
    growth: inferGrowth(player),
    teamName: inferTeamName(player, franchiseState),
    league: inferLeague(player),
    nat: inferNationality(player),
    hgt: inferHeight(player),
    wgt: inferWeight(player),
    hand: inferHandedness(player),
    season_stats: season,
    rating_groups: groups,
    rating_summary: ratingSummary(groups),
    note: buildPlayerNote({
      age: safeNum(pickFirstDefined(player.age, enriched.age), 18),
      growth: inferGrowth(player),
      fatigue: inferFatigue(player),
      injury: inferInjury(player),
      potential: inferPotential(player),
      role: usage.role,
      archetype: inferArchetype(player),
    }),
  };
  return formatPlayer(normalized);
}

function normalizeDraftPlayer(row, idx) {
  const trend = getTrendMeta(row?.trend, row?.rank_delta);
  const contract = {
    salary: 0,
    term: 0,
    type: "Unsigned",
    expiry: "Draft rights not signed",
  };

  const groups = normalizeRatingGroups(row);

  const normalized = {
    ...row,
    _draft: true,
    key: pickFirstDefined(row.id, `draft-${idx}`),
    id: pickFirstDefined(row.id, `draft-${idx}`),
    rank: safeNum(row.rank, idx + 1),
    rank_delta: safeNum(row.rank_delta, 0),
    trend: safeStr(row.trend, "SAME"),
    trendText: trend.text,
    trendClass: trend.cls,
    name: safeStr(row.name, `Prospect ${idx + 1}`),
    age: safeNum(row.age, 18),
    position: normalizePosition(row.position),
    ovr: safeNum(row.true_ovr, 0),
    scout_grade: safeStr(row.scout_grade, "—"),
    scout_tier: safeStr(row.scout_tier, "—"),
    true_ovr: safeNum(row.true_ovr, 0),
    potential: safeStr(row.potential, calculatePotential(row)),
    archetype: safeStr(row.archetype, "BALANCED"),
    morale: 50,
    fatigue: 0,
    injury: "Healthy",
    contract,
    role: "Prospect",
    specialTeams: "Development",
    minutes: safeNum(row.minutes, 0),
    dev: safeStr(row.development_curve, "Normal"),
    growth: safeNum(row.growth_delta, 0),
    teamName: safeStr(row.team_name, "Draft Eligible"),
    league: safeStr(pickFirstDefined(row.league_name, row.league_code), "—"),
    nat: safeStr(row.nationality, safeStr(row.league_code, "—")).slice(0, 12),
    hgt: safeStr(row.height, "—"),
    wgt: safeStr(row.weight, "—"),
    hand: safeStr(row.handedness, "—"),
    season_stats: normalizeSeasonStats(row),
    rating_groups: groups,
    rating_summary: ratingSummary(groups),
    note:
      safeStr(row.notes, "") ||
      `${safeStr(row.scout_tier, "Projected")} scouting tier. ${safeStr(row.development_curve, "Normal")} growth curve. ${safeStr(row.archetype, "Balanced")} profile.`,
  };
  return formatPlayer(normalized);
}

function comparePlayers(a, b, sortKey) {
  const getName = (p) => safeStr(p.name, "").toLowerCase();
  const getOvr = (p) => safeNum(p.ovr, 0);
  const getAge = (p) => safeNum(p.age, 0);
  const getMorale = (p) => safeNum(p.morale, 0);
  const getFatigue = (p) => safeNum(p.fatigue, 0);
  const getPotential = (p) => getPotentialRank(p.potential);
  const getPoints = (p) => safeNum(p?.season_stats?.pts, 0);
  const getGoals = (p) => safeNum(p?.season_stats?.g, 0);
  const getAssists = (p) => safeNum(p?.season_stats?.a, 0);
  const getSalary = (p) => safeNum(p?.contract?.salary, 0);
  const getTerm = (p) => safeNum(p?.contract?.term, 0);

  switch (sortKey) {
    case "overall_desc":
      return getOvr(b) - getOvr(a) || getAge(a) - getAge(b);
    case "overall_asc":
      return getOvr(a) - getOvr(b) || getAge(a) - getAge(b);
    case "age_desc":
      return getAge(b) - getAge(a) || getOvr(b) - getOvr(a);
    case "age_asc":
      return getAge(a) - getAge(b) || getOvr(b) - getOvr(a);
    case "name_desc":
      return getName(b).localeCompare(getName(a));
    case "name_asc":
      return getName(a).localeCompare(getName(b));
    case "potential_desc":
      return getPotential(b) - getPotential(a) || getOvr(b) - getOvr(a);
    case "morale_desc":
      return getMorale(b) - getMorale(a) || getOvr(b) - getOvr(a);
    case "fatigue_desc":
      return getFatigue(b) - getFatigue(a) || getOvr(b) - getOvr(a);
    case "points_desc":
      return getPoints(b) - getPoints(a) || getGoals(b) - getGoals(a);
    case "goals_desc":
      return getGoals(b) - getGoals(a) || getPoints(b) - getPoints(a);
    case "assists_desc":
      return getAssists(b) - getAssists(a) || getPoints(b) - getPoints(a);
    case "salary_desc":
      return getSalary(b) - getSalary(a) || getOvr(b) - getOvr(a);
    case "term_desc":
      return getTerm(b) - getTerm(a) || getSalary(b) - getSalary(a);
    default:
      return getOvr(b) - getOvr(a);
  }
}

function statLineForPlayer(player) {
  if (!player) return "—";
  const pos = normalizePosition(player.position);

  if (pos === "G") {
    const wins = safeNum(player?.season_stats?.wins, 0);
    const losses = safeNum(player?.season_stats?.losses, 0);
    const otl = safeNum(player?.season_stats?.otl, 0);
    const sv = safeNum(player?.season_stats?.svPct, 0);
    const gaa = safeNum(player?.season_stats?.gaa, 0);

    if (!wins && !losses && !otl && !sv && !gaa) return "No goalie stats";
    return `${wins}-${losses}-${otl} · SV% ${sv ? sv.toFixed(3) : "—"} · GAA ${gaa ? gaa.toFixed(2) : "—"}`;
  }

  const gp = safeNum(player?.season_stats?.gp, 0);
  const g = safeNum(player?.season_stats?.g, 0);
  const a = safeNum(player?.season_stats?.a, 0);
  const pts = safeNum(player?.season_stats?.pts, 0);

  if (!gp && !g && !a && !pts) return "No skater stats";
  return `${gp} GP · ${g} G · ${a} A · ${pts} PTS`;
}

function gradeFromOverall(ovr) {
  const n = safeNum(ovr, 0);
  if (n >= 92) return "A+";
  if (n >= 88) return "A";
  if (n >= 84) return "A-";
  if (n >= 80) return "B+";
  if (n >= 76) return "B";
  if (n >= 72) return "B-";
  if (n >= 68) return "C+";
  if (n >= 64) return "C";
  return "D";
}

function ToolbarSelect({ id, label, value, onChange, options, tooltip }) {
  return (
    <div className="roster-toolbar__control">
      <label className="roster-toolbar__label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className="roster-toolbar__select ui-interactive"
        value={value}
        onChange={onChange}
        data-tooltip={tooltip}
      >
        {options.map((opt) => (
          <option key={opt.value ?? opt} value={opt.value ?? opt}>
            {opt.label ?? opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function ToolbarInput({ id, label, value, onChange, placeholder, tooltip }) {
  return (
    <div className="roster-toolbar__control">
      <label className="roster-toolbar__label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="roster-toolbar__input ui-interactive"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        data-tooltip={tooltip}
      />
    </div>
  );
}

function InfoPair({ label, value, tone = "neutral" }) {
  return (
    <>
      <span className="roster-info-grid__k">{label}</span>
      <span className={`roster-info-grid__v ${toneClass(tone)}`}>{value}</span>
    </>
  );
}

function ProgressBar({ label, value, max = 100 }) {
  const pct = clamp((safeNum(value, 0) / max) * 100, 0, 100);
  return (
    <div className="roster-progress-row">
      <div className="roster-progress-row__top">
        <span>{label}</span>
        <span>{safeNum(value, 0).toFixed(0)}</span>
      </div>
      <div className="roster-progress-row__bar">
        <div className="roster-progress-row__fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function MiniBadge({ text, tone = "neutral" }) {
  return <span className={`roster-mini-badge ${toneClass(tone)}`}>{text}</span>;
}

function DraftBoardTable({ players, rosterRowIndex, setRosterRowIndex, pageOffset = 0 }) {
  return (
    <>
      <div className="draft-board-header">
        <span className="d-rnk">#</span>
        <span className="d-trd">MOVE</span>
        <span className="d-name">NAME</span>
        <span className="d-pos">POS</span>
        <span className="d-age">AGE</span>
        <span className="d-lg">LEAGUE</span>
        <span className="d-ovr">TRUE</span>
        <span className="d-scout">SCOUT</span>
        <span className="d-tier">TIER</span>
        <span className="d-potential">POT</span>
        <span className="d-arch">TYPE</span>
      </div>

      <div className="draft-board-body">
        {players.map((p, idx) => {
          const globalIdx = pageOffset + idx;
          return (
          <div
            key={p.key || `${p.name}-${idx}`}
            className={`draft-board-row ui-interactive ${globalIdx === rosterRowIndex ? "is-selected" : ""}`}
            onClick={() => setRosterRowIndex(globalIdx)}
            role="row"
            data-tooltip="Scout grade vs true OVR, plus profile and development indicators"
          >
            <span className="d-rnk">{p.rank}</span>
            <span className={`d-trd draft-trend-flag ${p.trendClass}`}>{p.trendText}</span>
            <span className="d-name">{p.name}</span>
            <span className="d-pos">{p.position}</span>
            <span className="d-age">{p.age}</span>
            <span className="d-lg" title={p.league}>
              {safeStr(p.league, "—").replace("EU_J_", "")}
            </span>
            <span className="d-ovr">{p.true_ovr}</span>
            <span className="d-scout">{p.scout_grade}</span>
            <span className="d-tier">{p.scout_tier}</span>
            <span className="d-potential">{p.potential}</span>
            <span className="d-arch">{p.archetype}</span>
          </div>
        );
        })}
      </div>
    </>
  );
}

function RosterTable({ players, rosterRowIndex, setRosterRowIndex, activeRowMenu, setActiveRowMenu, pageOffset = 0 }) {
  return (
    <>
      <div className="roster-table-header roster-table-header--expanded roster-table-header--immersive roster-wf-table-header">
        <span className="col-name">NAME</span>
        <span className="col-pos">POS</span>
        <span className="col-num">OVR</span>
        <span className="col-pot">POT</span>
        <span className="col-age">AGE</span>
        <span className="col-role">ROLE</span>
        <span className="col-arch">ARCHETYPE</span>
        <span className="col-morale">MORALE</span>
        <span className="col-growth">STATUS</span>
        <span className="col-actions" aria-hidden />
      </div>

      <div className="roster-table-body roster-wf-table-body">
        {players.map((p, idx) => {
          const globalIdx = pageOffset + idx;
          const isSelected = globalIdx === rosterRowIndex;
          const moralePct = clamp(safeNum(p.morale, 0) > 1 ? safeNum(p.morale, 0) : safeNum(p.morale, 0) * 100, 0, 100);
          const archColor = getArchetypeColor(p.archetype);

          return (
            <div
              key={p.key || `${p.name}-${idx}`}
              className={`roster-row roster-row--expanded roster-row--immersive ui-interactive ${isSelected ? "is-selected" : ""}`}
              data-tooltip="Select player for full franchise breakdown"
              onClick={() => setRosterRowIndex(globalIdx)}
              role="row"
            >
              <span className="col-name">
                <strong>{p.name}</strong>
              </span>
              <span className="col-pos">{p.position}</span>
              <span className="col-num" style={{ color: getOVRColor(p.ovr) }}>
                {p.ovr}
              </span>
              <span className="col-pot">
                <MiniBadge text={p.potential} tone={getPotentialRank(p.potential) >= 80 ? "good" : "neutral"} />
              </span>
              <span className="col-age">{p.age}</span>
              <span className="col-role">{p.roleLabel || p.role}</span>
              <span className="col-arch">
                <span className="roster-arch-tag" style={{ background: `${archColor}22`, borderColor: archColor, color: archColor }}>
                  {p.archetype}
                </span>
              </span>
              <span className="col-morale">
                <span className="roster-morale-cell">
                  <span className="roster-morale-cell__bar">
                    <span className="roster-morale-cell__fill" style={{ width: `${moralePct}%`, background: p.moraleColor }} />
                  </span>
                </span>
              </span>
              <span className="col-growth">{p.status}</span>
              <span className="row-actions">
                <button
                  type="button"
                  className="row-actions__btn ui-interactive"
                  onClick={(e) => {
                    e.stopPropagation();
                    setActiveRowMenu((v) => (v === p.key ? null : p.key));
                  }}
                >
                  ⋮
                </button>
                {activeRowMenu === p.key ? (
                  <div className="row-actions__menu" onClick={(e) => e.stopPropagation()}>
                    <button type="button" className="ui-interactive" onClick={() => setRosterRowIndex(globalIdx)}>View Details</button>
                    <button type="button" className="ui-interactive" onClick={() => setActiveRowMenu(null)}>Move to AHL</button>
                    <button type="button" className="ui-interactive" onClick={() => setActiveRowMenu(null)}>Scratch</button>
                    <button type="button" className="ui-interactive" onClick={() => setActiveRowMenu(null)}>Trade Block</button>
                  </div>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </>
  );
}

function LineView({ groupedRoster, nhlLines, onSelectPlayer }) {
  return (
    <div className="roster-lineview">
      <div className="roster-lineview__summary">
        <span>NHL {groupedRoster.NHL.length}</span>
        <span>AHL {groupedRoster.AHL.length}</span>
        <span>ECHL {groupedRoster.ECHL.length}</span>
      </div>

      <div className="roster-lineview__block">
        <div className="roster-lineview__title">Forward Lines</div>
        {nhlLines.forwards.slice(0, 4).map((line, idx) => (
          <div key={`f-${idx}`} className="roster-lineview__row">
            <strong>L{idx + 1}</strong>
            <div>
              {line.map((p) => (
                <button key={p.key} type="button" className="roster-lineview__chip ui-interactive" onClick={() => onSelectPlayer(p)}>
                  {p.name}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="roster-lineview__block">
        <div className="roster-lineview__title">Defense Pairs</div>
        {nhlLines.defense.slice(0, 3).map((pair, idx) => (
          <div key={`d-${idx}`} className="roster-lineview__row">
            <strong>D{idx + 1}</strong>
            <div>
              {pair.map((p) => (
                <button key={p.key} type="button" className="roster-lineview__chip ui-interactive" onClick={() => onSelectPlayer(p)}>
                  {p.name}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="roster-lineview__block">
        <div className="roster-lineview__title">Goalies</div>
        <div className="roster-lineview__row">
          <strong>G</strong>
          <div>
            {nhlLines.goalies.map((p) => (
              <button key={p.key} type="button" className="roster-lineview__chip ui-interactive" onClick={() => onSelectPlayer(p)}>
                {p.name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function RosterCards({ players, rosterRowIndex, setRosterRowIndex, pageOffset = 0 }) {
  return (
    <div className="roster-card-grid">
      {players.map((p, idx) => {
        const globalIdx = pageOffset + idx;
        const moraleBand = getMoraleBand(p.morale);
        const fatigueBand = getFatigueBand(p.fatigue);
        const injuryBand = getInjuryBand(p.injury);

        return (
          <button
            key={p.key || `${p.name}-${idx}`}
            type="button"
            className={`roster-card ui-interactive game-panel-bevel ${globalIdx === rosterRowIndex ? "is-selected" : ""}`}
            onClick={() => setRosterRowIndex(globalIdx)}
          >
            <div className="roster-card__top">
              <div className="roster-card__crest">{initialsFromName(p.name)}</div>
              <div className="roster-card__id">
                <div className="roster-card__name">{p.name}</div>
                <div className="roster-card__sub">
                  {p.position} · {p.age} · {p.teamName}
                </div>
              </div>
              <div className="roster-card__ovr">
                <div className="roster-card__ovr-value">{p.ovr}</div>
                <div className="roster-card__ovr-grade">{gradeFromOverall(p.ovr)}</div>
              </div>
            </div>

            <div className="roster-card__badges">
              <MiniBadge text={p.potential} tone={getPotentialRank(p.potential) >= 80 ? "good" : "neutral"} />
              <MiniBadge text={p.archetype} />
              <MiniBadge text={p.role} />
            </div>

            <div className="roster-card__stats">
              <div className="roster-card__stat">
                <span>Morale</span>
                <strong className={toneClass(moraleBand.tone)}>{p.morale}</strong>
              </div>
              <div className="roster-card__stat">
                <span>Fatigue</span>
                <strong className={toneClass(fatigueBand.tone)}>{p.fatigue}</strong>
              </div>
              <div className="roster-card__stat">
                <span>Health</span>
                <strong className={toneClass(injuryBand.tone)}>{injuryBand.label}</strong>
              </div>
              <div className="roster-card__stat">
                <span>Cap</span>
                <strong>{formatMoneyMillions(p.contract.salary)}</strong>
              </div>
            </div>

            <div className="roster-card__foot">{statLineForPlayer(p)}</div>
          </button>
        );
      })}
    </div>
  );
}

function DevelopmentPanel({ player }) {
  if (!player) return null;

  const devBand = getDevelopmentBand(player.growth);
  const moraleBand = getMoraleBand(player.morale);
  const fatigueBand = getFatigueBand(player.fatigue);

  return (
    <div className="roster-detail-panel">
      <div className="roster-detail-grid">
        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">DEVELOPMENT STATUS</div>
          <div className="roster-info-grid">
            <InfoPair label="Curve" value={player.dev} />
            <InfoPair label="Growth Delta" value={formatSignedNumber(player.growth)} tone={devBand.tone} />
            <InfoPair label="State" value={devBand.label} tone={devBand.tone} />
            <InfoPair label="Potential" value={player.potential} />
            <InfoPair label="Age Window" value={player.age <= 21 ? "Early Growth" : player.age <= 27 ? "Prime Build" : "Maintenance / Decline Watch"} />
            <InfoPair label="Morale" value={`${player.morale} · ${moraleBand.label}`} tone={moraleBand.tone} />
            <InfoPair label="Fatigue" value={`${player.fatigue} · ${fatigueBand.label}`} tone={fatigueBand.tone} />
            <InfoPair label="Health" value={player.injury} tone={getInjuryBand(player.injury).tone} />
          </div>
        </div>

        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">DEVELOPMENT READOUT</div>
          <ProgressBar label="Morale" value={player.morale} />
          <ProgressBar label="Fatigue" value={100 - player.fatigue} />
          <ProgressBar label="Offense" value={player.rating_summary.offense} />
          <ProgressBar label="Defense" value={player.rating_summary.defense} />
          <ProgressBar label="Skating" value={player.rating_summary.skating} />
          <ProgressBar label="Physical" value={player.rating_summary.physical} />
          <ProgressBar label="Mental" value={player.rating_summary.mental} />
        </div>
      </div>
    </div>
  );
}

function UsagePanel({ player }) {
  if (!player) return null;

  return (
    <div className="roster-detail-panel">
      <div className="roster-detail-grid">
        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">DEPLOYMENT</div>
          <div className="roster-info-grid">
            <InfoPair label="Role" value={player.role} />
            <InfoPair label="Special Teams" value={player.specialTeams} />
            <InfoPair label="Average Minutes" value={player.minutes ? `${player.minutes.toFixed(1)} min` : "—"} />
            <InfoPair label="Health" value={player.injury} tone={getInjuryBand(player.injury).tone} />
            <InfoPair label="Fatigue Band" value={getFatigueBand(player.fatigue).label} tone={getFatigueBand(player.fatigue).tone} />
            <InfoPair label="Usage Fit" value={player.minutes >= 20 ? "Heavy" : player.minutes >= 14 ? "Regular" : player.minutes > 0 ? "Sheltered" : "Inactive"} />
          </div>
        </div>

        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">SEASON PRODUCTION</div>
          <div className="roster-info-grid">
            <InfoPair label="Games Played" value={player.season_stats.gp} />
            <InfoPair label="Goals" value={player.season_stats.g} />
            <InfoPair label="Assists" value={player.season_stats.a} />
            <InfoPair label="Points" value={player.season_stats.pts} />
            <InfoPair label="Penalty Minutes" value={player.season_stats.pim} />
            <InfoPair label="+/-" value={player.season_stats.plusMinus} />
            <InfoPair label="Save %" value={player.season_stats.svPct ? player.season_stats.svPct.toFixed(3) : "—"} />
            <InfoPair label="GAA" value={player.season_stats.gaa ? player.season_stats.gaa.toFixed(2) : "—"} />
          </div>
        </div>
      </div>
    </div>
  );
}

function ContractPanel({ player }) {
  if (!player) return null;

  const valueBand =
    player.contract.salary >= 8
      ? { label: "Premium", tone: "bad" }
      : player.contract.salary >= 4
        ? { label: "Market", tone: "neutral" }
        : { label: "Value", tone: "good" };

  return (
    <div className="roster-detail-panel">
      <div className="roster-detail-grid">
        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">CONTRACT STATUS</div>
          <div className="roster-info-grid">
            <InfoPair label="Cap Hit" value={formatMoneyMillions(player.contract.salary)} tone={valueBand.tone} />
            <InfoPair label="Term" value={`${player.contract.term} years`} />
            <InfoPair label="Type" value={player.contract.type} />
            <InfoPair label="Expiry" value={player.contract.expiry} />
            <InfoPair label="Value Tier" value={valueBand.label} tone={valueBand.tone} />
          </div>
        </div>

        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">FRANCHISE READ</div>
          <div className="roster-notes">
            <p>
              {player.contract.term <= 1
                ? "Short-term contract exposure. Extension or asset decision may be needed soon."
                : "Contract control remains in place."}
            </p>
            <p>
              {player.contract.salary >= 8
                ? "High cap concentration. Player must deliver top-line or top-pair value."
                : player.contract.salary >= 4
                  ? "Mid-tier cap commitment with regular lineup expectations."
                  : "Cap-efficient contract relative to most NHL roster slots."}
            </p>
            <p>
              {getPotentialRank(player.potential) >= 80
                ? "Ceiling and age profile suggest this player should be protected in long-range planning."
                : "Asset value depends more on role fit, usage, and consistency than star upside."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function RatingsPanel({ player }) {
  if (!player) return null;

  const groups = Array.isArray(player.rating_groups) ? player.rating_groups : EMPTY_ARRAY;

  return (
    <div className="roster-detail-panel">
      <div className="roster-detail-grid roster-detail-grid--ratings">
        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">ATTRIBUTE SUMMARY</div>
          <ProgressBar label="Offense" value={player.rating_summary.offense} />
          <ProgressBar label="Defense" value={player.rating_summary.defense} />
          <ProgressBar label="Skating" value={player.rating_summary.skating} />
          <ProgressBar label="Physical" value={player.rating_summary.physical} />
          <ProgressBar label="Mental" value={player.rating_summary.mental} />
        </div>

        {groups.map((group) => (
          <div key={group.title} className="roster-detail-card game-panel-bevel">
            <div className="roster-detail-card__title">{group.title}</div>
            <ul className="roster-ratings-group__list">
              {(group.rows || []).length ? (
                group.rows.map((row) => (
                  <li key={row.id} className="roster-ratings-row">
                    <span className="roster-ratings-row__label">{row.label}</span>
                    <span className="roster-ratings-row__v">{safeNum(row.v, 0)}</span>
                  </li>
                ))
              ) : (
                <li className="roster-ratings-row roster-ratings-row--empty">
                  <span className="roster-ratings-row__label">No attribute data available</span>
                </li>
              )}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function NotesPanel({ player }) {
  if (!player) return null;

  return (
    <div className="roster-detail-panel">
      <div className="roster-detail-grid">
        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">SCOUT / GM NOTES</div>
          <div className="roster-notes">
            <p>{player.note || "No internal notes available."}</p>
            <p>
              {player._draft
                ? "Draft board players should expose uncertainty between scout grade and true rating. Use this view to compare perceived value versus hidden talent."
                : "Live roster players should surface identity, role, contract, health, and development all in one place so the roster feels like a real management system rather than a flat browser."}
            </p>
          </div>
        </div>

        <div className="roster-detail-card game-panel-bevel">
          <div className="roster-detail-card__title">FRANCHISE FLAGS</div>
          <div className="roster-flag-list">
            <MiniBadge
              text={player.ovr >= 84 ? "Top-End Talent" : player.ovr >= 78 ? "Middle Core" : "Depth Tier"}
              tone={player.ovr >= 84 ? "good" : player.ovr >= 78 ? "neutral" : "warn"}
            />
            <MiniBadge
              text={safeNum(player.growth, 0) > 0.2 ? "Trending Up" : safeNum(player.growth, 0) < -0.2 ? "Trending Down" : "Stable Arc"}
              tone={safeNum(player.growth, 0) > 0.2 ? "good" : safeNum(player.growth, 0) < -0.2 ? "bad" : "neutral"}
            />
            <MiniBadge
              text={safeStr(player.injury, "Healthy")}
              tone={safeStr(player.injury, "Healthy").toLowerCase() === "healthy" ? "good" : "bad"}
            />
            <MiniBadge
              text={player.contract.salary >= 8 ? "Cap Heavy" : player.contract.salary >= 4 ? "Market Cost" : "Value Deal"}
              tone={player.contract.salary >= 8 ? "bad" : player.contract.salary >= 4 ? "neutral" : "good"}
            />
            <MiniBadge
              text={player.age <= 21 ? "Prospect Window" : player.age <= 27 ? "Prime Window" : "Aging Watch"}
              tone={player.age <= 27 ? "good" : "warn"}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function positionDisplayLabel(position) {
  const n = normalizePosition(position);
  if (n === "D") return "Defenseman";
  if (n === "G") return "Goalie";
  return "Forward";
}

function HistoryPanel({ player }) {
  if (!player) {
    return (
      <div className="roster-detail-panel roster-wf-bottom__router">
        <div className="roster-notes">No player selected.</div>
      </div>
    );
  }

  return (
    <div className="roster-detail-panel roster-wf-bottom__router">
      <div className="roster-detail-card game-panel-bevel">
        <div className="roster-detail-card__title">FRANCHISE HISTORY</div>
        <div className="roster-notes">
          <p>{player.note || "No tracked transactions or milestones for this player yet."}</p>
        </div>
      </div>
    </div>
  );
}

function PlayerOverviewWireframe({ player }) {
  if (!player) {
    return (
      <div className="roster-wf-overview roster-wf-overview--empty">
        <div className="roster-notes">Select a player from the list.</div>
      </div>
    );
  }

  const ss = player.season_stats || EMPTY_OBJECT;
  const moralePct = clamp(safeNum(player.morale, 0) > 1 ? safeNum(player.morale, 0) : safeNum(player.morale, 0) * 100, 0, 100);
  const isG = normalizePosition(player.position) === "G";
  const pctDisplay = isG
    ? ss.svPct
      ? `${(safeNum(ss.svPct, 0) <= 1 ? safeNum(ss.svPct, 0) * 100 : safeNum(ss.svPct, 0)).toFixed(1)}%`
      : "—"
    : "—";

  return (
    <div className="roster-wf-overview">
      <div className="roster-wf-overview__col roster-wf-overview__card game-panel-bevel">
        <div className="roster-wf-photo" aria-hidden>
          <span className="roster-wf-photo__silhouette">{initialsFromName(player.name)}</span>
        </div>
        <div className="roster-wf-overview__name">{safeStr(player.name, "—").toUpperCase()}</div>
        <div className="roster-wf-overview__sub">
          {positionDisplayLabel(player.position)} | {safeStr(player.teamName, "—")}
        </div>
        <dl className="roster-wf-mini-grid">
          <dt>OVR</dt><dd style={{ color: getOVRColor(player.ovr) }}>{player.ovr}</dd>
          <dt>POTENTIAL</dt><dd>{player.potential}</dd>
          <dt>AGE</dt><dd>{player.age}</dd>
          <dt>HEIGHT</dt><dd>{player.hgt}</dd>
          <dt>WEIGHT</dt><dd>{player.wgt}</dd>
          <dt>SHOOTS</dt><dd>{player.hand}</dd>
        </dl>
      </div>

      <div className="roster-wf-overview__col roster-wf-overview__info game-panel-bevel">
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Archetype</span><span>{player.archetype}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Role</span><span>{player.roleLabel || player.role}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">League</span><span>{player.league}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Status</span><span>{player.status}</span></div>
        <div className="roster-wf-info-line roster-wf-info-line--morale">
          <span className="roster-wf-info-line__k">Morale</span>
          <span className="roster-statusline__bar roster-wf-info-line__bar">
            <span className="roster-statusline__fill" style={{ width: `${moralePct}%`, background: player.moraleColor }} />
          </span>
        </div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Development Stage</span><span>{player.stage}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Handedness</span><span>{player.hand}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Nationality</span><span>{player.nat}</span></div>
      </div>

      <div className="roster-wf-overview__col roster-wf-overview__stats game-panel-bevel">
        <div className="roster-wf-overview__block-title">Season stats</div>
        <div className="roster-wf-stat-table">
          <div className="roster-wf-stat-table__head">
            <span>GP</span><span>G</span><span>A</span><span>PTS</span><span>+/-</span><span>PIM</span><span>SOG</span><span>%</span>
          </div>
          <div className="roster-wf-stat-table__row">
            <span>{ss.gp ?? "—"}</span>
            <span>{ss.g ?? "—"}</span>
            <span>{ss.a ?? "—"}</span>
            <span>{ss.pts ?? "—"}</span>
            <span>{ss.plusMinus ?? "—"}</span>
            <span>{ss.pim ?? "—"}</span>
            <span>—</span>
            <span>{pctDisplay}</span>
          </div>
        </div>
        <div className="roster-wf-overview__block-title">Contract overview</div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Salary</span><span>{formatMoneyMillions(player.contract.salary)}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Cap Hit</span><span>{formatMoneyMillions(player.contract.salary)}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Contract Type</span><span>{player.contract.type}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Years Remaining</span><span>{player.contract.term}Y</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Expiry Status</span><span>{player.contract.expiry}</span></div>
        <div className="roster-wf-info-line"><span className="roster-wf-info-line__k">Trade Status</span><span>{player.assetTag || "—"}</span></div>
      </div>
    </div>
  );
}

function DetailTabs({ activeTab, setActiveTab }) {
  return (
    <div className="roster-detail-tabs roster-wf-detail-tabs">
      {PANEL_TABS.map((tab) => (
        <button
          key={tab.value}
          type="button"
          className={`roster-detail-tab ui-interactive ${activeTab === tab.value ? "is-active" : ""}`}
          onClick={() => setActiveTab(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function DetailPanelRouter({ activeTab, player }) {
  if (activeTab === "overview") return <PlayerOverviewWireframe player={player} />;
  if (activeTab === "stats") return <UsagePanel player={player} />;
  if (activeTab === "attributes") return <RatingsPanel player={player} />;
  if (activeTab === "contract") return <ContractPanel player={player} />;
  if (activeTab === "development") return <DevelopmentPanel player={player} />;
  if (activeTab === "history") return <HistoryPanel player={player} />;
  return <NotesPanel player={player} />;
}

function RosterScreenStyles() {
  return (
    <style>{`
      .roster-screen--wireframe { flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }

      .roster-wf-topbar {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        margin: 0 10px 10px;
        gap: 12px;
      }
      .roster-wf-topbar__brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
      .roster-wf-topbar__logo {
        width: 40px; height: 40px; flex-shrink: 0;
        background: linear-gradient(145deg, rgba(58, 74, 120, 0.9), rgba(18, 26, 46, 0.95));
        border: 2px solid rgba(180, 190, 210, 0.45);
        border-radius: 4px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.45);
      }
      .roster-wf-topbar__titles { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
      .roster-wf-topbar__team {
        font-size: 1.02rem; font-weight: 800; letter-spacing: 0.1em;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .roster-wf-topbar__sub { font-size: 0.58rem; letter-spacing: 0.22em; opacity: 0.72; }
      .roster-wf-hamburger {
        flex-shrink: 0; width: 42px; height: 42px; border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.16); background: rgba(255,255,255,0.06);
        color: inherit; font-size: 1.15rem; line-height: 1;
      }

      .roster-wf-main {
        flex: 1;
        min-height: 0;
        margin: 0 10px;
        display: grid;
        grid-template-columns: 228px minmax(0, 1fr);
        gap: 12px;
        overflow: hidden;
      }
      .roster-wf-main--drawer { grid-template-columns: 228px minmax(0, 1fr) minmax(220px, 280px); }

      .roster-wf-sidebar {
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 10px;
        overflow: hidden;
      }
      .roster-wf-sidebox { padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; min-height: 0; }
      .roster-wf-sidebox__title {
        font-size: 10px; letter-spacing: 0.16em; font-weight: 900; opacity: 0.85;
        border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 6px;
      }
      .roster-wf-kv { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
      .roster-wf-kv li { display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px; }
      .roster-wf-kv span:first-child { opacity: 0.75; }
      .roster-wf-kv span:last-child { font-weight: 700; text-align: right; }

      .roster-wf-center {
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: 10px;
        overflow: hidden;
      }
      .roster-wf-filters { padding: 8px 10px; flex-shrink: 0; }
      .roster-wf-filters__row { margin: 0; }
      .roster-wf-reset {
        align-self: flex-end;
        margin-top: 4px;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255,140,66,0.45);
        background: rgba(255,140,66,0.12);
        color: inherit;
        font-size: 10px;
        letter-spacing: 0.14em;
        font-weight: 900;
      }

      .roster-wf-table-shell {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        padding: 0;
        overflow: hidden;
      }
      .roster-wf-table-shell__head { flex-shrink: 0; padding: 10px 12px 6px; }
      .roster-wf-table-scroll {
        flex: 1;
        min-height: 0;
        overflow: auto;
        padding: 0 8px 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .roster-screen--wireframe .roster-wf-table-scroll .roster-table-body {
        flex: 0 0 auto;
        overflow: visible;
        max-height: none;
      }
      .roster-screen--wireframe .roster-table-header.roster-wf-table-header {
        grid-template-columns: minmax(100px, 1.6fr) 44px 44px 52px 40px minmax(72px, 0.9fr) minmax(88px, 1fr) minmax(96px, 0.95fr) minmax(64px, 0.65fr) 40px;
        align-items: center;
      }
      .roster-screen--wireframe .roster-row--expanded.roster-row--immersive {
        grid-template-columns: minmax(100px, 1.6fr) 44px 44px 52px 40px minmax(72px, 0.9fr) minmax(88px, 1fr) minmax(96px, 0.95fr) minmax(64px, 0.65fr) 40px;
      }
      .roster-screen--wireframe .roster-row--immersive .col-name { flex-direction: row; align-items: center; }
      .roster-screen--wireframe .roster-row--immersive .col-pos { font-weight: 800; font-size: 11px; letter-spacing: 0.06em; }

      .roster-wf-pagination {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 8px 10px 10px;
        border-top: 1px solid rgba(255,255,255,0.08);
        font-size: 11px;
        letter-spacing: 0.08em;
      }
      .roster-wf-pagination__pages { display: inline-flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
      .roster-wf-pagination__num {
        min-width: 28px; padding: 4px 8px; border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); color: inherit;
      }
      .roster-wf-pagination__num.is-current {
        border-color: rgba(127,208,255,0.65);
        box-shadow: inset 0 0 0 1px rgba(127,208,255,0.45);
      }

      .roster-wf-drawer {
        min-height: 0;
        overflow-y: auto;
        padding: 10px 10px 14px;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .roster-wf-drawer__section { display: flex; flex-direction: column; gap: 6px; }
      .roster-wf-drawer__section--tools .roster-toolbar__control { width: 100%; }
      .roster-wf-drawer__heading {
        font-size: 10px; letter-spacing: 0.18em; font-weight: 900; opacity: 0.78;
        margin-bottom: 2px;
      }
      .roster-wf-drawer__links { display: flex; flex-direction: column; gap: 4px; }
      .roster-wf-drawer__link {
        text-align: left;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.04);
        color: inherit;
        font-size: 12px;
      }
      .roster-wf-drawer__link.is-muted { opacity: 0.45; cursor: default; }

      .roster-wf-bottom {
        flex-shrink: 0;
        margin: 10px;
        padding: 10px 12px 12px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        min-height: 200px;
        max-height: 42vh;
      }
      .roster-wf-bottom__body {
        flex: 1;
        min-height: 0;
        overflow: auto;
      }
      .roster-wf-detail-tabs {
        flex-shrink: 0;
        flex-wrap: wrap;
        gap: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 8px;
      }
      .roster-wf-detail-tabs .roster-detail-tab {
        padding: 8px 10px;
        font-size: 10px;
        letter-spacing: 0.1em;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        color: inherit;
      }
      .roster-wf-detail-tabs .roster-detail-tab.is-active {
        border-color: rgba(127,208,255,0.55);
        box-shadow: inset 0 0 0 1px rgba(127,208,255,0.35);
      }

      .roster-wf-overview {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.1fr);
        gap: 12px;
        align-items: stretch;
      }
      .roster-wf-overview--empty { padding: 12px; }
      .roster-wf-overview__col { padding: 12px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
      .roster-wf-photo {
        width: 100%; aspect-ratio: 1; max-width: 140px;
        margin: 0 auto;
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 8px;
        background: rgba(0,0,0,0.25);
        display: flex; align-items: center; justify-content: center;
      }
      .roster-wf-photo__silhouette { font-size: 28px; font-weight: 900; opacity: 0.55; }
      .roster-wf-overview__name { font-size: 15px; font-weight: 900; letter-spacing: 0.06em; text-align: center; }
      .roster-wf-overview__sub { font-size: 12px; opacity: 0.85; text-align: center; }
      .roster-wf-mini-grid {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 6px 10px;
        font-size: 12px;
        margin: 0;
      }
      .roster-wf-mini-grid dt { opacity: 0.7; margin: 0; }
      .roster-wf-mini-grid dd { margin: 0; font-weight: 700; }
      .roster-wf-info-line { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; }
      .roster-wf-info-line__k { opacity: 0.75; }
      .roster-wf-info-line--morale { align-items: center; }
      .roster-wf-info-line__bar { flex: 1; max-width: 140px; height: 8px; margin-left: auto; }
      .roster-wf-overview__block-title { font-size: 10px; letter-spacing: 0.14em; font-weight: 900; opacity: 0.8; }
      .roster-wf-stat-table { font-size: 11px; width: 100%; }
      .roster-wf-stat-table__head,
      .roster-wf-stat-table__row {
        display: grid;
        grid-template-columns: repeat(8, minmax(0, 1fr));
        gap: 4px;
        text-align: center;
      }
      .roster-wf-stat-table__head { font-weight: 900; letter-spacing: 0.06em; opacity: 0.75; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 4px; }
      .roster-wf-stat-table__row { padding-top: 4px; font-weight: 700; }

      .roster-wf-bottom__router { min-height: 120px; }

      .roster-table-header--immersive { font-weight: 900; letter-spacing: 0.08em; border-bottom: 1px solid rgba(255,255,255,0.14); }
      .roster-row--immersive { align-items: center; min-height: 44px; border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 140ms ease, box-shadow 140ms ease; position: relative; }
      .roster-row--immersive:nth-child(odd) { background: rgba(255,255,255,0.025); }
      .roster-row--immersive:hover { background: rgba(255,255,255,0.05); }
      .roster-row--immersive.is-selected { box-shadow: inset 0 0 0 1px rgba(127,208,255,0.6), 0 0 14px rgba(114,179,255,0.18); }
      .roster-row--immersive.is-selected .col-name { border-left: 3px solid #f39c12; padding-left: 6px; }
      .roster-row--immersive .col-name { display: flex; flex-direction: column; gap: 2px; }
      .roster-row--immersive .col-name strong { font-size: 13px; }
      .roster-row--immersive .col-name em { opacity: 0.75; font-size: 11px; font-style: normal; }
      .roster-row--immersive .col-num { font-size: 20px; font-weight: 1000; }
      .roster-arch-tag { display: inline-flex; align-items: center; border: 1px solid; border-radius: 999px; padding: 2px 8px; font-size: 10px; font-weight: 900; letter-spacing: 0.08em; }
      .roster-morale-cell { display: inline-flex; align-items: center; gap: 8px; min-width: 110px; }
      .roster-morale-cell__bar { width: 72px; height: 7px; border-radius: 999px; background: rgba(255,255,255,0.12); overflow: hidden; }
      .roster-morale-cell__fill { display: block; height: 100%; }

      .roster-spotlight__sectionTitle { margin-top: 10px; margin-bottom: 6px; font-size: 10px; letter-spacing: 0.16em; opacity: 0.72; font-weight: 900; }
      .roster-coreline { display: flex; align-items: baseline; gap: 10px; }
      .roster-coreline__ovr { font-size: 34px; font-weight: 1000; line-height: 1; }
      .roster-coreline__pot { font-size: 15px; font-weight: 900; opacity: 0.9; }
      .roster-statusline { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
      .roster-statusline__label { width: 52px; font-size: 11px; opacity: 0.75; }
      .roster-statusline__bar { flex: 1; height: 8px; background: rgba(255,255,255,0.12); border-radius: 999px; overflow: hidden; }
      .roster-statusline__fill { display: block; height: 100%; }

      .identity-section { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; font-size: 13px; }
      .identity-section strong { opacity: 0.78; margin-right: 4px; }

      .roster-structure { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; margin-bottom: 10px; padding: 8px 10px; background: rgba(255,255,255,0.02); }
      .roster-structure__summary { display: flex; gap: 14px; font-size: 11px; letter-spacing: 0.12em; margin-bottom: 6px; }
      .roster-structure__lines { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
      .roster-structure__block strong { display: inline-block; min-width: 104px; color: #9ec7ff; }

      .row-actions { position: relative; justify-self: end; }
      .row-actions__btn { width: 28px; height: 28px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.06); color: inherit; }
      .row-actions__menu { position: absolute; right: 0; top: 30px; min-width: 140px; display: flex; flex-direction: column; gap: 4px; background: rgba(6,14,28,0.96); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 6px; z-index: 50; }
      .row-actions__menu button { text-align: left; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; background: rgba(255,255,255,0.04); color: inherit; padding: 6px 8px; }

      .roster-lineview { border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 10px; background: rgba(255,255,255,0.02); }
      .roster-lineview__summary { display: flex; gap: 14px; margin-bottom: 8px; font-size: 12px; }
      .roster-lineview__block { margin-bottom: 8px; }
      .roster-lineview__title { font-size: 11px; letter-spacing: 0.14em; margin-bottom: 6px; opacity: 0.8; }
      .roster-lineview__row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 6px; }
      .roster-lineview__row strong { min-width: 28px; opacity: 0.8; }
      .roster-lineview__chip { margin-right: 6px; margin-bottom: 6px; border: 1px solid rgba(255,255,255,0.14); border-radius: 999px; background: rgba(255,255,255,0.05); color: inherit; padding: 4px 10px; font-size: 12px; }

      .tab-content { min-height: 0; }
    `}</style>
  );
}

export function RosterScreen() {
  const { franchiseState, rosterRowIndex, setRosterRowIndex, setScreen, refreshFranchise } = useGameUI();

  const rb = franchiseState?.roster_browser;
  const draftBoard = franchiseState?.draft_class_rankings;
  const orgs = useMemo(
    () => franchiseState?.roster_browser?.organizations || EMPTY_ARRAY,
    [franchiseState?.roster_browser]
  );
  const userId = franchiseState?.team?.id;

  const [browseSource, setBrowseSource] = useState("organization");
  const [orgTeamId, setOrgTeamId] = useState("");
  const [orgLevel, setOrgLevel] = useState("nhl");
  const [devLeagueIdx, setDevLeagueIdx] = useState(0);
  const [devTeamIdx, setDevTeamIdx] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [leagueFilter, setLeagueFilter] = useState("ALL");
  const [roleFilter, setRoleFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("All");
  const [sortKey, setSortKey] = useState("overall_desc");
  const [viewMode, setViewMode] = useState("table");
  const [activeTab, setActiveTab] = useState("overview");
  const [showOnlyCore, setShowOnlyCore] = useState(false);
  const [showHamburgerMenu, setShowHamburgerMenu] = useState(false);
  const [tablePage, setTablePage] = useState(0);
  const [centerView, setCenterView] = useState("table");
  const [activeRowMenu, setActiveRowMenu] = useState(null);

  const defaultOrgId = useMemo(() => {
    const match = orgs.find((o) => String(o.team_id) === String(userId));
    return String((match || orgs[0])?.team_id || "");
  }, [orgs, userId]);

  useEffect(() => {
    if (!orgTeamId && defaultOrgId) {
      setOrgTeamId(defaultOrgId);
    }
  }, [defaultOrgId, orgTeamId]);

  const rawPlayers = useMemo(() => {
    if (browseSource === "draft_class") {
      return EMPTY_ARRAY;
    }

    if (!rb) {
      return franchiseState?.roster || EMPTY_ARRAY;
    }

    if (browseSource === "free_agents") {
      return rb.free_agents || EMPTY_ARRAY;
    }

    if (browseSource === "overseas") {
      return rb.overseas_free_agents || EMPTY_ARRAY;
    }

    if (browseSource === "development") {
      const leagues = rb.development_leagues || EMPTY_ARRAY;
      const selectedLeague = leagues[devLeagueIdx];
      const teams = selectedLeague?.teams || EMPTY_ARRAY;
      const selectedTeam = teams[devTeamIdx];
      return selectedTeam?.players || EMPTY_ARRAY;
    }

    const org = orgs.find((o) => String(o.team_id) === String(orgTeamId)) || orgs[0];
    if (!org) return franchiseState?.roster || EMPTY_ARRAY;

    const key = orgLevel === "ahl" ? "ahl" : orgLevel === "echl" ? "echl" : "nhl";
    return org[key] || EMPTY_ARRAY;
  }, [
    rb,
    browseSource,
    devLeagueIdx,
    devTeamIdx,
    orgTeamId,
    orgLevel,
    orgs,
    franchiseState?.roster,
  ]);

  const players = useMemo(() => {
    if (browseSource === "draft_class") {
      const rows = draftBoard?.entries || EMPTY_ARRAY;
      return rows.map((row, idx) => normalizeDraftPlayer(row, idx));
    }

    return rawPlayers.map((player, idx) => normalizeLivePlayer(player, franchiseState, idx));
  }, [browseSource, draftBoard?.entries, rawPlayers, franchiseState]);

  const roleOptions = useMemo(() => {
    const roles = Array.from(new Set(players.map((p) => formatRole(p.role)).filter(Boolean)));
    return ["ALL", ...roles];
  }, [players]);

  const filteredPlayers = useMemo(() => {
    const q = safeStr(searchTerm, "").trim().toLowerCase();

    const out = players.filter((player) => {
      if (q) {
        const haystack = [
          player.name,
          player.position,
          player.archetype,
          player.teamName,
          player.league,
          player.potential,
          player.role,
          player.nat,
        ]
          .map((v) => safeStr(v, "").toLowerCase())
          .join(" ");

        if (!haystack.includes(q)) return false;
      }

      if (!positionMatchesFilter(player.position, positionFilter)) return false;

      if (leagueFilter !== "ALL" && safeStr(player.league, "").toUpperCase() !== leagueFilter) {
        return false;
      }

      if (roleFilter !== "ALL" && formatRole(player.role) !== roleFilter) return false;

      if (statusFilter !== "All" && player.status !== statusFilter) return false;

      if (showOnlyCore && getPotentialRank(player.potential) < 80 && safeNum(player.ovr, 0) < 80) {
        return false;
      }

      return true;
    });

    return [...out].sort((a, b) => comparePlayers(a, b, sortKey));
  }, [players, searchTerm, positionFilter, leagueFilter, roleFilter, statusFilter, showOnlyCore, sortKey]);

  const selectedSafeIndex = useMemo(() => {
    if (!filteredPlayers.length) return -1;
    return clamp(rosterRowIndex, 0, filteredPlayers.length - 1);
  }, [filteredPlayers.length, rosterRowIndex]);

  const selected = selectedSafeIndex >= 0 ? filteredPlayers[selectedSafeIndex] : null;
  const selectedStorylines = useMemo(() => {
    if (!selected) return EMPTY_ARRAY;
    const sid = String(selected.id || "");
    const nm = safeStr(selected.name, "").toLowerCase();
    const raw = franchiseState?.storyline_events || EMPTY_ARRAY;
    return raw
      .filter((ev) => {
        const players = Array.isArray(ev?.players) ? ev.players.map((p) => String(p).toLowerCase()) : EMPTY_ARRAY;
        const pid = String(ev?.player_id || "").toLowerCase();
        const pname = safeStr(ev?.player_name, "").toLowerCase();
        return (sid && pid === sid.toLowerCase()) || (nm && (pname === nm || players.includes(nm)));
      })
      .slice(-5)
      .reverse();
  }, [franchiseState?.storyline_events, selected]);

  const devLeagues = rb?.development_leagues || EMPTY_ARRAY;
  const devTeams = devLeagues[devLeagueIdx]?.teams || EMPTY_ARRAY;

  const countsLabel = useMemo(() => {
    if (browseSource === "draft_class") {
      return draftBoard?.subtitle
        ? `${draftBoard.subtitle} · Listed ${(draftBoard.entries || EMPTY_ARRAY).length} / ${draftBoard.total ?? "—"}`
        : "";
    }

    return rb?.counts
      ? `NHL ${rb.counts.nhl_contracted ?? "—"} · AHL ${rb.counts.ahl_contracted ?? "—"} · ECHL ${rb.counts.echl_contracted ?? "—"} · UFA ${rb.counts.free_agents ?? "—"} · Overseas ${rb.counts.overseas ?? "—"} · Dev ${rb.counts.junior_skaters ?? "—"}`
      : "";
  }, [browseSource, draftBoard?.subtitle, draftBoard?.entries, draftBoard?.total, rb?.counts]);

  const groupedRoster = useMemo(() => {
    const source = filteredPlayers;
    return {
      NHL: source.filter((p) => p.league === "NHL"),
      AHL: source.filter((p) => p.league === "AHL"),
      ECHL: source.filter((p) => p.league === "ECHL"),
    };
  }, [filteredPlayers]);

  const nhlLines = useMemo(() => {
    const nhlPlayers = groupedRoster.NHL || EMPTY_ARRAY;
    const forwards = groupIntoLines(nhlPlayers.filter((p) => !["D", "G"].includes(normalizePosition(p.position))));
    const defense = groupIntoPairs(nhlPlayers.filter((p) => normalizePosition(p.position) === "D"));
    const goalies = nhlPlayers.filter((p) => normalizePosition(p.position) === "G");
    return { forwards, defense, goalies };
  }, [groupedRoster]);

  const statsLite = useMemo(() => {
    const count = filteredPlayers.length;
    const avgOVR = count ? (filteredPlayers.reduce((s, p) => s + safeNum(p.ovr, 0), 0) / count).toFixed(1) : "—";
    const avgAge = count ? (filteredPlayers.reduce((s, p) => s + safeNum(p.age, 0), 0) / count).toFixed(1) : "—";
    return { count, avgOVR, avgAge };
  }, [filteredPlayers]);

  const rosterBreakdown = useMemo(() => {
    const countPool = (list) => {
      let forwards = 0;
      let defense = 0;
      let goalies = 0;
      for (const raw of list || EMPTY_ARRAY) {
        const pos = normalizePosition(raw?.position ?? raw?.pos);
        if (pos === "G") goalies += 1;
        else if (pos === "D") defense += 1;
        else forwards += 1;
      }
      return { forwards, defense, goalies };
    };

    const org = orgs.find((o) => String(o.team_id) === String(orgTeamId)) || orgs[0];
    if (browseSource === "organization" && org) {
      const nhl = org.nhl || EMPTY_ARRAY;
      const ahl = org.ahl || EMPTY_ARRAY;
      const echl = org.echl || EMPTY_ARRAY;
      const { forwards, defense, goalies } = countPool(nhl);
      return { forwards, defense, goalies, ahl: ahl.length, echl: echl.length };
    }

    const { forwards, defense, goalies } = countPool(
      filteredPlayers.map((p) => ({ position: p.position }))
    );
    const ahlCount = filteredPlayers.filter((p) => p.league === "AHL").length;
    const echlCount = filteredPlayers.filter((p) => p.league === "ECHL").length;
    return { forwards, defense, goalies, ahl: ahlCount, echl: echlCount };
  }, [browseSource, orgTeamId, orgs, filteredPlayers]);

  const sidebarOrgFinance = useMemo(() => {
    const org = orgs.find((o) => String(o.team_id) === String(orgTeamId)) || orgs[0];
    const capLimit =
      safeNum(franchiseState?.team?.salary_cap, 0) ||
      safeNum(franchiseState?.team?.cap_limit, 0) ||
      DEFAULT_CAP_LIMIT;
    const capHitFromPayload = safeNum(franchiseState?.team?.cap_hit, 0);

    if (browseSource === "organization" && org?.nhl) {
      const nhlList = org.nhl || EMPTY_ARRAY;
      const capUsedComputed = nhlList.reduce((sum, raw) => sum + safeNum(normalizeContract(raw).salary, 0), 0);
      const capUsed = capHitFromPayload > 0 ? capHitFromPayload : capUsedComputed;
      const injured = nhlList.filter((raw) => {
        const gr = safeNum(
          pickFirstDefined(raw?.injury_games_remaining, raw?.games_remaining, raw?.days_remaining),
          0
        );
        if (raw?.is_injured === true || gr > 0) return true;
        return safeStr(raw?.injury ?? raw?.injury_status, "Healthy").toLowerCase() !== "healthy";
      }).length;
      const ltir = nhlList.filter((raw) => /ltir/i.test(safeStr(raw?.injury ?? raw?.injury_status, ""))).length;
      const contracts = nhlList.length;
      return {
        capUsed,
        capLimit,
        capSpace: Math.max(0, capLimit - capUsed),
        injured,
        ltir,
        contracts,
        contractLimit: NHL_CONTRACT_LIMIT,
      };
    }

    const capUsedComputed = filteredPlayers.reduce((sum, p) => sum + safeNum(p?.contract?.salary, 0), 0);
    const capUsed = capHitFromPayload > 0 ? capHitFromPayload : capUsedComputed;
    const injured = filteredPlayers.filter((p) => p.status === "Injured").length;
    const ltir = filteredPlayers.filter((p) => /ltir/i.test(safeStr(p?.injury, ""))).length;
    return {
      capUsed,
      capLimit,
      capSpace: Math.max(0, capLimit - capUsed),
      injured,
      ltir,
      contracts: filteredPlayers.length,
      contractLimit: NHL_CONTRACT_LIMIT,
    };
  }, [browseSource, franchiseState?.team?.cap_hit, franchiseState?.team?.cap_limit, franchiseState?.team?.salary_cap, orgTeamId, orgs, filteredPlayers]);

  const tableTotalPages = Math.max(1, Math.ceil(filteredPlayers.length / TABLE_PAGE_SIZE));
  const tablePageSafe = clamp(tablePage, 0, tableTotalPages - 1);
  const pageStart = tablePageSafe * TABLE_PAGE_SIZE;
  const pagePlayers = filteredPlayers.slice(pageStart, pageStart + TABLE_PAGE_SIZE);

  const pageButtonIndices = useMemo(() => {
    const maxBtn = 6;
    if (tableTotalPages <= maxBtn) {
      return Array.from({ length: tableTotalPages }, (_, i) => i);
    }
    const half = Math.floor(maxBtn / 2);
    let start = Math.max(0, tablePageSafe - half);
    let end = Math.min(tableTotalPages, start + maxBtn);
    start = Math.max(0, end - maxBtn);
    return Array.from({ length: end - start }, (_, j) => start + j);
  }, [tableTotalPages, tablePageSafe]);

  const viewCombined =
    centerView === "line" ? "line" : viewMode === "cards" ? "cards" : "table";

  const setViewCombined = useCallback((v) => {
    if (v === "line") {
      setCenterView("line");
      return;
    }
    setCenterView("table");
    setViewMode(v === "cards" ? "cards" : "table");
  }, []);

  const selectPlayerByKey = useCallback(
    (player) => {
      const idx = filteredPlayers.findIndex((p) => p.key === player?.key);
      if (idx >= 0) setRosterRowIndex(idx);
      setActiveRowMenu(null);
    },
    [filteredPlayers, setRosterRowIndex]
  );

  const selectPlayerByIndex = useCallback(
    (idx) => {
      setRosterRowIndex(idx);
      setActiveRowMenu(null);
    },
    [setRosterRowIndex]
  );

  useEffect(() => {
    if (rosterRowIndex >= filteredPlayers.length) {
      setRosterRowIndex(Math.max(0, filteredPlayers.length - 1));
    }
  }, [filteredPlayers.length, rosterRowIndex, setRosterRowIndex]);

  useEffect(() => {
    setRosterRowIndex(0);
    setTablePage(0);
  }, [
    browseSource,
    orgTeamId,
    orgLevel,
    devLeagueIdx,
    devTeamIdx,
    searchTerm,
    positionFilter,
    leagueFilter,
    roleFilter,
    statusFilter,
    sortKey,
    showOnlyCore,
    viewMode,
    setRosterRowIndex,
  ]);

  useEffect(() => {
    const maxPage = Math.max(0, Math.ceil(filteredPlayers.length / TABLE_PAGE_SIZE) - 1);
    setTablePage((p) => Math.min(p, maxPage));
  }, [filteredPlayers.length]);

  useEffect(() => {
    if (devTeamIdx >= devTeams.length) {
      setDevTeamIdx(Math.max(0, devTeams.length - 1));
    }
  }, [devTeamIdx, devTeams.length]);

  const onKey = useCallback(
    (e) => {
      if (e.target?.matches?.("input, textarea, select, button")) return;

      if (e.key === "Escape") {
        e.preventDefault();
        if (showHamburgerMenu) {
          setShowHamburgerMenu(false);
          return;
        }
        setScreen(SCREENS.HUB);
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setRosterRowIndex((idx) => Math.max(0, idx - 1));
        return;
      }

      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (!filteredPlayers.length) return;
        setRosterRowIndex((idx) => Math.min(filteredPlayers.length - 1, idx + 1));
        return;
      }

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const currentIndex = PANEL_TABS.findIndex((tab) => tab.value === activeTab);
        const nextIndex = currentIndex <= 0 ? PANEL_TABS.length - 1 : currentIndex - 1;
        setActiveTab(PANEL_TABS[nextIndex].value);
        return;
      }

      if (e.key === "ArrowRight") {
        e.preventDefault();
        const currentIndex = PANEL_TABS.findIndex((tab) => tab.value === activeTab);
        const nextIndex = currentIndex >= PANEL_TABS.length - 1 ? 0 : currentIndex + 1;
        setActiveTab(PANEL_TABS[nextIndex].value);
      }
    },
    [activeTab, filteredPlayers.length, setRosterRowIndex, setScreen, showHamburgerMenu, setShowHamburgerMenu]
  );

  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);

  useEffect(() => {
    const onDocClick = (e) => {
      const inMenu = e.target?.closest?.(
        ".roster-wf__hamburger-btn, .roster-wf-drawer, .row-actions, .row-actions__menu"
      );
      if (!inMenu) {
        setShowHamburgerMenu(false);
        setActiveRowMenu(null);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const teamTitle = safeStr(franchiseState?.team?.name, "—").toUpperCase();
  const listTitle =
    browseSource === "draft_class"
      ? "PLAYER LIST (DRAFT)"
      : `PLAYER LIST (${leagueFilter === "ALL" ? "ALL" : leagueFilter})`;

  return (
    <div className="game-screen roster-screen roster-screen--franchise roster-screen--wireframe">
      <RosterScreenStyles />

      <header className="roster-wf-topbar game-panel-bevel">
        <div className="roster-wf-topbar__brand">
          <span className="roster-wf-topbar__logo" aria-hidden />
          <div className="roster-wf-topbar__titles">
            <div className="roster-wf-topbar__team">{teamTitle}</div>
            <div className="roster-wf-topbar__sub">NHL FRANCHISE</div>
          </div>
        </div>
        <button
          type="button"
          className="roster-wf-hamburger roster-wf__hamburger-btn ui-interactive"
          aria-expanded={showHamburgerMenu}
          aria-label="Open menu"
          onClick={() => setShowHamburgerMenu((v) => !v)}
        >
          ☰
        </button>
      </header>

      <div className={`roster-wf-main ${showHamburgerMenu ? "roster-wf-main--drawer" : ""}`}>
        <aside className="roster-wf-sidebar">
          <div className="roster-wf-sidebox game-panel-bevel">
            <div className="roster-wf-sidebox__title">Roster Overview</div>
            <ul className="roster-wf-kv">
              <li><span>Players</span><span>{statsLite.count}</span></li>
              <li><span>Average OVR</span><span>{statsLite.avgOVR}</span></li>
              <li><span>Average Age</span><span>{statsLite.avgAge}</span></li>
              <li>
                <span>Salary Cap</span>
                <span>
                  {formatMoneyMillions(sidebarOrgFinance.capUsed)} / {formatMoneyMillions(sidebarOrgFinance.capLimit)}
                </span>
              </li>
              <li><span>Cap Space</span><span>{formatMoneyMillions(sidebarOrgFinance.capSpace)}</span></li>
              <li>
                <span>Contracts</span>
                <span>{sidebarOrgFinance.contracts} / {sidebarOrgFinance.contractLimit}</span>
              </li>
              <li><span>Injured</span><span>{sidebarOrgFinance.injured}</span></li>
              <li><span>On LTIR</span><span>{sidebarOrgFinance.ltir}</span></li>
            </ul>
          </div>
          <div className="roster-wf-sidebox game-panel-bevel">
            <div className="roster-wf-sidebox__title">Roster Breakdown</div>
            <ul className="roster-wf-kv">
              <li><span>Forwards</span><span>{rosterBreakdown.forwards}</span></li>
              <li><span>Defensemen</span><span>{rosterBreakdown.defense}</span></li>
              <li><span>Goalies</span><span>{rosterBreakdown.goalies}</span></li>
              <li><span>AHL</span><span>{rosterBreakdown.ahl}</span></li>
              <li><span>ECHL</span><span>{rosterBreakdown.echl}</span></li>
            </ul>
          </div>
        </aside>

        <section className="roster-wf-center">
          <div className="roster-wf-filters game-panel-bevel">
            <div className="roster-toolbar__row roster-toolbar__row--wrap roster-wf-filters__row">
              <ToolbarSelect
                id="wf-view"
                label="VIEW"
                value={viewCombined}
                onChange={(e) => setViewCombined(e.target.value)}
                tooltip="Table, cards, or NHL-style lines"
                options={[
                  { value: "table", label: "Table View" },
                  { value: "cards", label: "Cards View" },
                  { value: "line", label: "Line View" },
                ]}
              />
              <ToolbarSelect
                id="wf-league"
                label="LEAGUE"
                value={leagueFilter}
                onChange={(e) => setLeagueFilter(e.target.value)}
                tooltip="NHL / AHL / ECHL roster slice"
                options={LEAGUE_FILTERS}
              />
              <ToolbarSelect
                id="wf-position"
                label="POSITION"
                value={positionFilter}
                onChange={(e) => setPositionFilter(e.target.value)}
                tooltip="Filter by position group"
                options={POSITION_FILTERS}
              />
              <ToolbarSelect
                id="wf-role"
                label="ROLE"
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
                tooltip="Filter by role label"
                options={roleOptions}
              />
              <ToolbarSelect
                id="wf-status"
                label="STATUS"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                tooltip="Availability state filter"
                options={STATUS_FILTERS}
              />
              <button
                type="button"
                className="roster-wf-reset ui-interactive"
                onClick={() => {
                  setSearchTerm("");
                  setPositionFilter("ALL");
                  setLeagueFilter("ALL");
                  setRoleFilter("ALL");
                  setStatusFilter("All");
                  setSortKey("overall_desc");
                  setShowOnlyCore(false);
                  setTablePage(0);
                }}
              >
                RESET FILTERS
              </button>
            </div>
          </div>

          <div className="roster-wf-table-shell game-panel-bevel">
            <div className="roster-table-wrap__head roster-wf-table-shell__head">
              <div className="roster-table-wrap__title">{listTitle}</div>
              <div className="roster-table-wrap__meta">
                Showing {filteredPlayers.length} / {players.length}
              </div>
            </div>

            <div className="roster-wf-table-scroll">
              {browseSource !== "draft_class" && centerView === "line" ? (
                <LineView groupedRoster={groupedRoster} nhlLines={nhlLines} onSelectPlayer={selectPlayerByKey} />
              ) : null}

              {centerView === "table" ? (
                viewMode === "cards" && browseSource !== "draft_class" ? (
                  <RosterCards
                    players={pagePlayers}
                    pageOffset={pageStart}
                    rosterRowIndex={selectedSafeIndex < 0 ? 0 : selectedSafeIndex}
                    setRosterRowIndex={selectPlayerByIndex}
                  />
                ) : browseSource === "draft_class" ? (
                  <DraftBoardTable
                    players={pagePlayers}
                    pageOffset={pageStart}
                    rosterRowIndex={selectedSafeIndex < 0 ? 0 : selectedSafeIndex}
                    setRosterRowIndex={selectPlayerByIndex}
                  />
                ) : (
                  <RosterTable
                    players={pagePlayers}
                    pageOffset={pageStart}
                    rosterRowIndex={selectedSafeIndex < 0 ? 0 : selectedSafeIndex}
                    setRosterRowIndex={selectPlayerByIndex}
                    activeRowMenu={activeRowMenu}
                    setActiveRowMenu={setActiveRowMenu}
                  />
                )
              ) : null}
            </div>

            {centerView === "table" ? (
              <div className="roster-wf-pagination">
                <button
                  type="button"
                  className="ui-interactive"
                  disabled={tablePageSafe <= 0}
                  onClick={() => setTablePage((p) => Math.max(0, p - 1))}
                >
                  &lt; PREV
                </button>
                <span className="roster-wf-pagination__pages">
                  {pageButtonIndices.map((i) => (
                    <button
                      key={`pg-${i}`}
                      type="button"
                      className={`ui-interactive roster-wf-pagination__num ${i === tablePageSafe ? "is-current" : ""}`}
                      onClick={() => setTablePage(i)}
                    >
                      {i + 1}
                    </button>
                  ))}
                </span>
                <button
                  type="button"
                  className="ui-interactive"
                  disabled={tablePageSafe >= tableTotalPages - 1}
                  onClick={() => setTablePage((p) => Math.min(tableTotalPages - 1, p + 1))}
                >
                  NEXT &gt;
                </button>
              </div>
            ) : null}
          </div>
        </section>

        {showHamburgerMenu ? (
          <aside className="roster-wf-drawer roster-wf__drawer game-panel-bevel">
            <div className="roster-wf-drawer__section">
              <div className="roster-wf-drawer__heading">League Switcher</div>
              <div className="roster-wf-drawer__links">
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setBrowseSource("organization");
                    setOrgLevel("nhl");
                    setShowHamburgerMenu(false);
                  }}
                >
                  NHL
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setBrowseSource("organization");
                    setOrgLevel("ahl");
                    setShowHamburgerMenu(false);
                  }}
                >
                  AHL
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setBrowseSource("organization");
                    setOrgLevel("echl");
                    setShowHamburgerMenu(false);
                  }}
                >
                  ECHL
                </button>
              </div>
            </div>

            <div className="roster-wf-drawer__section">
              <div className="roster-wf-drawer__heading">Roster Management</div>
              <div className="roster-wf-drawer__links">
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Roster Control</button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setCenterView("line");
                    setShowHamburgerMenu(false);
                  }}
                >
                  Line Management
                </button>
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Injuries</button>
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Waivers</button>
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Assign Players</button>
              </div>
            </div>

            <div className="roster-wf-drawer__section roster-wf-drawer__section--tools">
              <ToolbarSelect
                id="dr-pool"
                label="POOL"
                value={browseSource}
                onChange={(e) => setBrowseSource(e.target.value)}
                tooltip="Browse pool"
                options={[
                  { value: "organization", label: "NHL organization (NHL / AHL / ECHL)" },
                  { value: "free_agents", label: "Free agents (unsigned)" },
                  { value: "overseas", label: "Overseas / unsigned runway" },
                  { value: "development", label: "Junior & college leagues" },
                ]}
              />
              <ToolbarInput
                id="dr-search"
                label="PLAYER SEARCH"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Name, archetype, team, league..."
                tooltip="Text filter"
              />
              <ToolbarSelect
                id="dr-sort"
                label="SORT"
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value)}
                tooltip="Sort order"
                options={SORT_KEYS}
              />
              {browseSource === "organization" ? (
                <ToolbarSelect
                  id="dr-club"
                  label="CLUB"
                  value={orgTeamId || defaultOrgId}
                  onChange={(e) => setOrgTeamId(e.target.value)}
                  tooltip="Organization"
                  options={orgs.map((o) => ({
                    value: String(o.team_id),
                    label: o.name,
                  }))}
                />
              ) : null}
              {browseSource === "organization" ? (
                <ToolbarSelect
                  id="dr-level"
                  label="LEVEL"
                  value={orgLevel}
                  onChange={(e) => setOrgLevel(e.target.value)}
                  tooltip="Roster tier"
                  options={[
                    { value: "nhl", label: "NHL (game roster)" },
                    { value: "ahl", label: "AHL (affiliate)" },
                    { value: "echl", label: "ECHL (affiliate)" },
                  ]}
                />
              ) : null}
              {browseSource === "development" ? (
                <>
                  <ToolbarSelect
                    id="dr-dev-lg"
                    label="DEV LEAGUE"
                    value={String(devLeagueIdx)}
                    onChange={(e) => {
                      setDevLeagueIdx(Number(e.target.value));
                      setDevTeamIdx(0);
                    }}
                    tooltip="Development league"
                    options={devLeagues.map((league, idx) => ({
                      value: String(idx),
                      label: league.league_name || league.league_code || `League ${idx + 1}`,
                    }))}
                  />
                  <ToolbarSelect
                    id="dr-dev-tm"
                    label="DEV TEAM"
                    value={String(devTeamIdx)}
                    onChange={(e) => setDevTeamIdx(Number(e.target.value))}
                    tooltip="Development club"
                    options={devTeams.map((team, idx) => ({
                      value: String(idx),
                      label: team.name || `Team ${idx + 1}`,
                    }))}
                  />
                </>
              ) : null}
              <button
                type="button"
                className={`roster-toggle ui-interactive ${showOnlyCore ? "is-active" : ""}`}
                onClick={() => setShowOnlyCore((v) => !v)}
              >
                Core assets only
              </button>
              {countsLabel ? <div className="roster-toolbar__counts">{countsLabel}</div> : null}
            </div>

            <div className="roster-wf-drawer__section">
              <div className="roster-wf-drawer__heading">Player Management</div>
              <div className="roster-wf-drawer__links">
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Player Search</button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setScreen(SCREENS.STATS);
                    setShowHamburgerMenu(false);
                  }}
                >
                  Player Stats
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setActiveTab("development");
                    setShowHamburgerMenu(false);
                  }}
                >
                  Development
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setActiveTab("attributes");
                    setShowHamburgerMenu(false);
                  }}
                >
                  Attributes
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setActiveTab("contract");
                    setShowHamburgerMenu(false);
                  }}
                >
                  Contract Management
                </button>
              </div>
            </div>

            <div className="roster-wf-drawer__section">
              <div className="roster-wf-drawer__heading">Team Management</div>
              <div className="roster-wf-drawer__links">
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Depth Chart</button>
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Team Chemistry</button>
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Team Info</button>
              </div>
            </div>

            <div className="roster-wf-drawer__section">
              <div className="roster-wf-drawer__heading">Other</div>
              <div className="roster-wf-drawer__links">
                <button type="button" className="roster-wf-drawer__link ui-interactive is-muted">Notifications</button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setScreen(SCREENS.SETTINGS);
                    setShowHamburgerMenu(false);
                  }}
                >
                  Settings
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    refreshFranchise?.();
                    setShowHamburgerMenu(false);
                  }}
                >
                  Save
                </button>
                <button
                  type="button"
                  className="roster-wf-drawer__link ui-interactive"
                  onClick={() => {
                    setScreen(SCREENS.HUB);
                    setShowHamburgerMenu(false);
                  }}
                >
                  Exit
                </button>
              </div>
            </div>
          </aside>
        ) : null}
      </div>

      <footer className="roster-wf-bottom game-panel-bevel">
        {selectedStorylines.length ? (
          <div style={{ padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.12)" }}>
            <div className="roster-detail-card__title">PLAYER STORYLINE TRACK</div>
            {selectedStorylines.map((ev, i) => (
              <div key={`${ev.id || i}`} className="roster-notes" style={{ marginBottom: 6 }}>
                <p>{ev.headline || ev.title || "Storyline event"}</p>
                {ev.cause ? <p>Cause: {ev.cause}</p> : null}
                {ev.effect_summary ? <p>Effect: {ev.effect_summary}</p> : null}
              </div>
            ))}
          </div>
        ) : null}
        <DetailTabs activeTab={activeTab} setActiveTab={setActiveTab} />
        <div className="roster-wf-bottom__body tab-content">
          <DetailPanelRouter activeTab={activeTab} player={selected} />
        </div>
      </footer>

      <GameFooter hints="↑↓ LIST · ←→ DETAIL TABS · MENU · ESC HUB" />
    </div>
  );
}

export default RosterScreen;