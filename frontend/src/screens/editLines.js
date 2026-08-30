import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getFranchiseChemistry, saveFranchiseLines } from "../services/franchiseService";
import { getFranchiseSessionId, readSessionLineupCache, writeSessionLineupCache } from "../services/api";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { getBaseOverall, getOverallDrop, getOverallTooltip, getUniversalOverall } from "../utils/playerOverall";
import "../styles/playerHeadshot.css";

const LINEUP_KIND = "even_strength";
const HISTORY_LIMIT = 25;
const GROUPS = ["forwards", "defense", "goalies"];
const LINEUP_MODES = { ALL: "all", FORWARDS: "forwards", DEFENSE: "defense", GOALIES: "goalies" };
const AUTO_MODES = [["overall", "Best Overall"], ["chemistry", "Best Chemistry"], ["position", "Position Safe"], ["roles", "Balanced Roles"]];
const TABS = ["unit", "fit", "player", "warnings"];
const LINK_STRONG = 84;
const LINK_FORMING = 70;
const CHEM_DISPLAY_LIFT = 8;
const NEW_PAIR_FAMILIARITY = 34;

function numberOrNull(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function percentOrNull(value) {
  const n = numberOrNull(value);
  if (n == null) return null;
  return Math.round(n > 0 && n <= 1.5 ? n * 100 : n);
}
function clamp(value, min = 0, max = 100) { return Math.max(min, Math.min(max, Math.round(Number(value) || 0))); }
function canonicalPlayerId(value) {
  const s = String(value || "").trim();
  if (!s) return "";
  return /^\d+$/.test(s) ? `NHL_${s}` : s;
}
function shortText(value, limit = 15) { return String(value || "").trim().split(/\s+/).filter(Boolean).slice(0, limit).join(" "); }
function lastName(value) {
  const parts = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (parts.length < 2) return String(value || "");
  return parts.slice(1).join(" ");
}
function shortRole(role) {
  let value = String(role || "—").trim();
  if (!value || value === "—" || value === "-") return "";
  value = value
    .replace(/ELITE[_ ]?FRANCHISE(?:[_ ]?PLAYER)?/gi, "Elite")
    .replace(/POWER[_ ]?FORWARD/gi, "PWF")
    .replace(/TWO[-_ ]?WAY[_ ]?FORWARD/gi, "2WF")
    .replace(/TWO[-_ ]?WAY[_ ]?DEFENSEMAN/gi, "2WD")
    .replace(/DEFENSIVE[_ ]?DEFENSEMAN/gi, "DD")
    .replace(/OFFENSIVE[_ ]?DEFENSEMAN/gi, "OD")
    .replace(/PLAYMAKER/gi, "PM")
    .replace(/SNIPER/gi, "SN")
    .replace(/GRINDER/gi, "GR")
    .replace(/ENFORCER/gi, "EN")
    .replace(/GOALTENDER|GOALIE/gi, "G")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return value.slice(0, 14);
}
function roleMixLabel(players) {
  const roles = [...new Set(players.map((player) => shortRole(player.role)).filter(Boolean))];
  if (!roles.length) return "—";
  if (roles.length <= 3) return roles.join(" · ");
  return `${roles.slice(0, 3).join(" · ")} +${roles.length - 3}`;
}
function slotMeta(player) {
  const bits = [];
  const role = shortRole(player?.role);
  const hand = String(player?.handedness || "").replace(/[—-]/g, "").trim().toUpperCase();
  if (role) bits.push(role);
  if (hand === "L" || hand === "R") bits.push(hand);
  return bits.join(" · ") || "—";
}
function playerName(player, index) {
  const direct = player?.name || player?.fullName || player?.full_name || player?.player_name;
  if (direct) return String(direct);
  const combined = `${player?.first_name || player?.firstName || ""} ${player?.last_name || player?.lastName || ""}`.trim();
  return combined || `Player ${index + 1}`;
}
function normalizeKey(value) {
  return String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
}
function isHealthyStatus(value) {
  const key = normalizeKey(value);
  return !key || ["healthy", "health", "none", "available", "active", "ok", "ready"].includes(key);
}
function normalizeAvailability(player) {
  const availabilityStatus = typeof player?.availability === "string"
    ? player.availability
    : (player?.availability_status || player?.status || player?.roster_status || "");
  const statusKey = normalizeKey(availabilityStatus);
  const injuryStatusKey = normalizeKey(player?.injury_status || player?.health_status || "");
  const gamesRemaining = Math.max(
    0,
    Number(player?.injury_games_remaining ?? player?.games_remaining ?? player?.days_remaining ?? 0) || 0
  );
  const injuredByFlag =
    player?.injured === true ||
    player?.is_injured === true ||
    player?.isInjured === true ||
    gamesRemaining > 0;
  const injuredByStatus =
    (!isHealthyStatus(injuryStatusKey) && (
      injuryStatusKey.includes("injur") ||
      injuryStatusKey.includes("day_to_day") ||
      injuryStatusKey.includes("ltir") ||
      injuryStatusKey === "out"
    )) ||
    statusKey.includes("injur") ||
    statusKey.includes("day_to_day") ||
    statusKey.includes("ltir") ||
    statusKey === "out";
  const injured = injuredByFlag || injuredByStatus;
  const unavailable =
    player?.available === false ||
    player?.is_available === false ||
    player?.inactive === true ||
    player?.suspended === true ||
    player?.conduct_eligible_to_play === false ||
    statusKey.includes("unavailable") ||
    statusKey.includes("inactive") ||
    statusKey.includes("suspended") ||
    statusKey.includes("leave") ||
    statusKey === "out";
  const scratched =
    player?.scratched === true ||
    player?.is_scratched === true ||
    player?.isScratched === true ||
    statusKey.includes("scratch");
  if (injured) return { key: "injured", label: "Injured", placeable: false, reason: "Injured players cannot be placed." };
  if (player?.conduct_eligible_to_play === false || statusKey.includes("suspended")) {
    return {
      key: "suspended",
      label: statusKey.includes("leave") ? "Leave" : "Suspended",
      placeable: false,
      reason: "Player cannot dress during leave / league suspension.",
    };
  }
  if (statusKey.includes("leave")) {
    return { key: "leave", label: "Leave", placeable: false, reason: "Player is on administrative leave." };
  }
  if (unavailable) return { key: "unavailable", label: "Unavailable", placeable: false, reason: "This player is unavailable." };
  if (scratched) return { key: "scratched", label: "Scratched", placeable: true, reason: "This player is scratched." };
  return { key: "active", label: "Active", placeable: true, reason: "Player is available." };
}
function normalizePlayer(player, index) {
  const name = playerName(player, index);
  const profile = player?.chemistry_profile || player?.chemistryProfile || null;
  const overall = Math.round(numberOrNull(getUniversalOverall(player)) ?? numberOrNull(player?.rating) ?? 0);
  const baseOverall = Math.round(numberOrNull(getBaseOverall(player)) ?? overall);
  const availability = normalizeAvailability(player);
  const rawId = player?.id ?? player?.player_id ?? player?._id ?? name;
  const id = canonicalPlayerId(rawId) || String(rawId || name);
  return ensurePlayerHeadshotFields({
    ...player,
    id,
    player_id: id,
    name,
    position: String(player?.position || player?.pos || "F").toUpperCase(),
    overall,
    ovr: overall,
    effective_ovr: overall,
    base_ovr: baseOverall,
    overall_drop: Math.max(0, Math.round(numberOrNull(getOverallDrop(player)) ?? 0)),
    handedness: String(player?.handedness || player?.shoots || player?.shot || "—").toUpperCase(),
    role: player?.role || player?.player_role || player?.archetype || "—",
    chemistry_profile: profile,
    chemistry_relationships:
      player?.chemistry_relationships || player?.chemistryRelationships || profile?.chemistry_relationships || null,
    chemistry_contract_missing: !profile,
    personality: profile?.personality || player?.personality || null,
    playstyle: player?.archetype || player?.playstyle || profile?.playstyle || null,
    morale: percentOrNull(profile?.morale ?? player?.morale),
    fatigue: percentOrNull(player?.fatigue ?? player?.tiredness ?? profile?.fatigue),
    confidence: percentOrNull(profile?.confidence ?? player?.confidence),
    role_satisfaction: percentOrNull(profile?.role_satisfaction ?? player?.role_satisfaction),
    coach_trust: percentOrNull(profile?.coach_trust ?? player?.coach_trust),
    scratched: availability.key === "scratched",
    availability,
  });
}
function getRosterPlayers(props) {
  const source = props.players || props.roster || props.team?.players || props.team?.roster || props.currentTeam?.players || props.currentTeam?.roster || props.franchise?.team?.roster || props.universe?.userTeam?.roster || [];
  return Array.isArray(source) ? source.map(normalizePlayer) : [];
}
function profileValue(player, key, fallback = null) {
  const profile = player?.chemistry_profile || {};
  const raw = profile[key] ?? profile[String(key).replaceAll("_", "")] ?? player?.[key];
  return percentOrNull(raw) ?? fallback;
}
function getCountry(player) { return player?.nationality || player?.nationality_code || player?.country || player?.birth_country || player?.birthCountry || player?.nation || "—"; }
function isForward(pos) { return ["C", "LW", "RW", "F"].includes(String(pos || "").toUpperCase()); }
function isDefense(pos) { return ["LD", "RD", "D"].includes(String(pos || "").toUpperCase()); }
function isGoalie(pos) { return String(pos || "").toUpperCase() === "G"; }
function goalieSlot(slot) { return ["Starter", "Backup", "Third"].includes(slot); }
function posFit(player, slot) {
  if (!player) return false;
  if (goalieSlot(slot)) return isGoalie(player.position);
  if (["LD", "RD"].includes(slot)) return isDefense(player.position);
  if (["LW", "C", "RW"].includes(slot)) return isForward(player.position);
  return false;
}
function idealScore(player, slot) {
  if (!player) return 0;
  const pos = player.position;
  const overall = Number(player.overall) || 0;
  const elite = overall >= 84;
  if (goalieSlot(slot)) return pos === "G" ? 100 : 0;
  if (pos === slot) return 100;
  if (slot === "C" && isForward(pos)) return elite ? 88 : 78;
  if (["LW", "RW"].includes(slot) && isForward(pos)) return elite ? 90 : 82;
  if (["LD", "RD"].includes(slot) && isDefense(pos)) return elite ? 92 : 80;
  return 0;
}
function chemistryFitScore(player, slot) {
  const overall = Number(player?.overall) || 0;
  const psych = isStubPsych(player) ? 68 : (profileValue(player, "morale", 68) ?? 68);
  return clamp(idealScore(player, slot) * 0.5 + overall * 0.35 + psych * 0.15);
}

/* ---------------------------------------------------------------------------
   Player-to-player links.
   Familiarity is the real bond: chemistry_relationships[otherId] grows daily
   while two players share a line. When the backend report already scored this
   exact pair we use that; otherwise we project from familiarity + profile.
--------------------------------------------------------------------------- */
const STYLE_SYNERGY = {
  sniper: { playmaker: 11, puck_mover: 7, power_forward: 6, net_front: 6, two_way: 5, sniper: -3 },
  playmaker: { sniper: 11, net_front: 9, power_forward: 7, puck_mover: 5, two_way: 5, playmaker: -2 },
  power_forward: { playmaker: 7, sniper: 6, grinder: 5, two_way: 4, power_forward: -2 },
  net_front: { playmaker: 9, sniper: 6, puck_mover: 5, net_front: -3 },
  puck_mover: { shutdown: 12, sniper: 7, playmaker: 5, puck_mover: -3 },
  shutdown: { puck_mover: 12, offensive_d: 10, two_way: 4, shutdown: -2 },
  offensive_d: { shutdown: 10, two_way: 3, offensive_d: -4 },
  two_way: { shutdown: 5, playmaker: 5, sniper: 5, power_forward: 4, two_way: 3 },
  grinder: { power_forward: 5, enforcer: 4, two_way: 3, grinder: -2 },
  enforcer: { grinder: 4, enforcer: -4 },
};
const PERSONALITY_SYNERGY = {
  leader: { young_gun: 10, quiet_pro: 6, high_ego_star: -4, leader: -3 },
  glue_guy: { high_ego_star: 8, hot_head: 7, young_gun: 6, glue_guy: 3 },
  high_ego_star: { glue_guy: 8, quiet_pro: 4, high_ego_star: -14, hot_head: -8 },
  hot_head: { glue_guy: 7, leader: 5, hot_head: -10 },
  young_gun: { leader: 10, veteran: 8, young_gun: -3 },
  quiet_pro: { leader: 6, glue_guy: 4 },
  veteran: { young_gun: 8, veteran: 2 },
};
function styleKey(value) { return normalizeKey(value); }
const PLAYSTYLE_ALIASES = {
  pm: "playmaker", playmaker: "playmaker",
  sn: "sniper", sniper: "sniper",
  pwf: "power_forward", power_forward: "power_forward", power: "power_forward",
  gr: "grinder", grinder: "grinder", enforcer: "enforcer",
  tw: "two_way", "2wf": "two_way", "2wd": "two_way", two_way: "two_way", two_way_f: "two_way", two_way_d: "two_way",
  od: "offensive_d", offensive_d: "offensive_d", offensive_defenseman: "offensive_d",
  dd: "shutdown", defensive_d: "shutdown", defensive_defenseman: "shutdown", shutdown: "shutdown",
  puck_mover: "puck_mover",
  net_front: "net_front",
};
const PERSONALITY_ALIASES = {
  young_skilled: "young_gun", young_gun: "young_gun",
  quiet_professional: "quiet_pro", quiet_pro: "quiet_pro",
  veteran_stabilizer: "veteran", veteran: "veteran",
  intense_competitor: "hot_head", hot_head: "hot_head",
  streaky_confidence: "young_gun",
  leader: "leader", glue_guy: "glue_guy", high_ego_star: "high_ego_star",
};
function resolvePlaystyle(player) {
  const candidates = [
    player?.archetype,
    player?.role,
    player?.player_role,
    player?.playstyle,
    player?.chemistry_profile?.playstyle,
  ];
  for (const raw of candidates) {
    const key = styleKey(raw);
    if (!key) continue;
    if (PLAYSTYLE_ALIASES[key]) return PLAYSTYLE_ALIASES[key];
    for (const [alias, mapped] of Object.entries(PLAYSTYLE_ALIASES)) {
      if (key.includes(alias)) return mapped;
    }
  }
  return "";
}
function resolvePersonality(player) {
  const raw = player?.personality || player?.chemistry_profile?.personality || "";
  return PERSONALITY_ALIASES[styleKey(raw)] || styleKey(raw);
}
function isStubPsych(player) {
  const morale = profileValue(player, "morale", null);
  const confidence = profileValue(player, "confidence", null);
  const coach = profileValue(player, "coach_trust", null);
  return (morale == null || morale === 50) && (confidence == null || confidence === 50) && (coach == null || coach === 50);
}
function psychTerm(player, key) {
  if (isStubPsych(player)) return 68;
  return profileValue(player, key, 68) ?? 68;
}
function synergyFrom(table, a, b) {
  const keyA = styleKey(a);
  const keyB = styleKey(b);
  if (!keyA || !keyB) return 0;
  return Number(table?.[keyA]?.[keyB] ?? table?.[keyB]?.[keyA] ?? 0) || 0;
}
function playerLookupIds(player) {
  const ids = new Set();
  for (const raw of [player?.id, player?.player_id, player?._id]) {
    const s = String(raw || "").trim();
    if (!s) continue;
    ids.add(s);
    ids.add(canonicalPlayerId(s));
  }
  return ids;
}
function pairIndexKey(a, b) {
  const ids = [...new Set([...playerLookupIds(a), ...playerLookupIds(b)])].map(canonicalPlayerId).filter(Boolean).sort();
  if (ids.length < 2) return "";
  return `${ids[0]}|${ids[1]}`;
}
function backendPairRows(chemReport) {
  if (!chemReport) return [];
  const rows = [];
  const index = chemReport.pair_index;
  if (index && typeof index === "object") rows.push(...Object.values(index));
  const units = [...(Array.isArray(chemReport.lines) ? chemReport.lines : []), ...(Array.isArray(chemReport.pairs) ? chemReport.pairs : [])];
  for (const unit of units) {
    if (Array.isArray(unit?.pair_links)) rows.push(...unit.pair_links);
  }
  if (Array.isArray(chemReport.deployed_pair_links)) rows.push(...chemReport.deployed_pair_links);
  if (Array.isArray(chemReport.top_connections)) rows.push(...chemReport.top_connections);
  return rows;
}
function backendPairMatch(rows, a, b) {
  const wanted = new Set([...playerLookupIds(a), ...playerLookupIds(b)]);
  return rows.find((row) => {
    const idA = canonicalPlayerId(row.player_a_id || row.playerAId || "");
    const idB = canonicalPlayerId(row.player_b_id || row.playerBId || "");
    if (idA && idB && wanted.has(idA) && wanted.has(idB)) return true;
    const ids = (row.players || []).map((entry) => canonicalPlayerId(entry.id || entry.player_id || ""));
    return ids.length === 2 && ids.every((id) => wanted.has(id));
  }) || null;
}
function backendPairScore(chemReport, a, b) {
  if (!chemReport || !a || !b) return null;
  const key = pairIndexKey(a, b);
  const indexed = key ? chemReport?.pair_index?.[key] : null;
  const match = indexed || backendPairMatch(backendPairRows(chemReport), a, b);
  if (!match) return null;
  return {
    score: clamp((match.chemistry ?? 50) + CHEM_DISPLAY_LIFT),
    familiarity: numberOrNull(match.familiarity ?? match.scheme_fit?.familiarity),
    projected: false,
  };
}
function getFamiliarity(a, b) {
  const idsA = playerLookupIds(a);
  const idsB = playerLookupIds(b);
  const relA = a?.chemistry_relationships || {};
  const relB = b?.chemistry_relationships || {};
  for (const idB of idsB) {
    const direct = numberOrNull(relA[idB]);
    if (direct != null) return clamp(direct);
  }
  for (const idA of idsA) {
    const direct = numberOrNull(relB[idA]);
    if (direct != null) return clamp(direct);
  }
  return null;
}
function pairChemistry(a, b, chemReport, slotA, slotB) {
  if (!a || !b) return null;
  const live = backendPairScore(chemReport, a, b);
  const familiarity = live?.familiarity ?? getFamiliarity(a, b);
  const familiarityValue = familiarity == null ? 50 : familiarity;
  const styleA = resolvePlaystyle(a);
  const styleB = resolvePlaystyle(b);
  const distinctive = styleA && styleB && !(styleA === "two_way" && styleB === "two_way");
  const liveLooksStub = live && live.score >= 55 && live.score <= 70;
  if (live && !(distinctive && liveLooksStub)) {
    return { ...live, familiarity, fresh: familiarityValue <= NEW_PAIR_FAMILIARITY, a: a.id, b: b.id };
  }
  const morale = (psychTerm(a, "morale") + psychTerm(b, "morale")) / 2;
  const confidence = (psychTerm(a, "confidence") + psychTerm(b, "confidence")) / 2;
  const adapt = (psychTerm(a, "adaptability") + psychTerm(b, "adaptability")) / 2;
  const coach = (psychTerm(a, "coach_trust") + psychTerm(b, "coach_trust")) / 2;
  const known = new Set(["LW", "C", "RW", "LD", "RD"]);
  const position = known.has(slotA) && known.has(slotB) ? (idealScore(a, slotA) + idealScore(b, slotB)) / 2 : 70;
  const personality = synergyFrom(PERSONALITY_SYNERGY, resolvePersonality(a), resolvePersonality(b));
  const style = synergyFrom(STYLE_SYNERGY, styleA, styleB);
  const egoA = profileValue(a, "ego", 50) ?? 50;
  const egoB = profileValue(b, "ego", 50) ?? 50;
  const egoTension = egoA > 72 && egoB > 72 ? -12 : 0;
  const baseTerms = [
    { weight: 0.20, value: (morale + confidence) / 2 },
    { weight: 0.16, value: adapt },
    { weight: 0.20, value: position },
    { weight: 0.12, value: coach },
  ];
  if (familiarity != null) {
    baseTerms.unshift({ weight: 0.26, value: familiarity });
  }
  const weightSum = baseTerms.reduce((sum, term) => sum + term.weight, 0);
  const baseScore = baseTerms.reduce((sum, term) => sum + (term.weight / weightSum) * term.value, 0);
  const quality = ((Number(a.overall) || 70) + (Number(b.overall) || 70)) / 2;
  const qualityAdj = Math.max(-2, Math.min(6, (quality - 76) * 0.4));
  const score = clamp(baseScore + personality + style + egoTension + qualityAdj + CHEM_DISPLAY_LIFT);
  return { score, projected: true, familiarity, fresh: familiarityValue <= NEW_PAIR_FAMILIARITY, a: a.id, b: b.id };
}
function linkTier(score) {
  if (score == null) return "empty";
  if (score >= LINK_STRONG) return "strong";
  if (score >= LINK_FORMING) return "forming";
  return "weak";
}
function averageLinkScore(links) {
  const scores = links.map((link) => link.score).filter((score) => Number.isFinite(score));
  if (!scores.length) return null;
  return Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
}

function emptyLines(includeThird = false) {
  const goalieSlots = { Starter: "", Backup: "" };
  if (includeThird) goalieSlots.Third = "";
  return {
    forwards: [1, 2, 3, 4].map((n) => ({ id: `f${n}`, name: `Line ${n}`, slots: { LW: "", C: "", RW: "" } })),
    defense: [1, 2, 3].map((n) => ({ id: `d${n}`, name: `Pair ${n}`, slots: { LD: "", RD: "" } })),
    goalies: [{ id: "g1", name: "Goalies", slots: goalieSlots }],
  };
}
function cloneLines(state) {
  return {
    forwards: (state?.forwards || []).map((line) => ({ ...line, slots: { ...(line?.slots || {}) } })),
    defense: (state?.defense || []).map((line) => ({ ...line, slots: { ...(line?.slots || {}) } })),
    goalies: (state?.goalies || []).map((line) => ({ ...line, slots: { ...(line?.slots || {}) } })),
  };
}
function snapshot(lineState, locks) { return { lineState: cloneLines(lineState), locks: { ...(locks || {}) } }; }
function snapshotKey(value) { return JSON.stringify(value || {}); }
function slotKey(group, lineId, slot) { return `${group}:${lineId}:${slot}`; }
function descriptors(state, showThird = true) {
  const rows = [];
  for (const group of GROUPS) for (const line of state?.[group] || []) for (const slot of Object.keys(line?.slots || {})) {
    if (slot === "Third" && !showThird) continue;
    rows.push({ key: slotKey(group, line.id, slot), group, lineId: line.id, lineName: line.name, slot, playerId: String(line.slots?.[slot] || "") });
  }
  return rows;
}
function findAssignment(state, playerId) { return descriptors(state, true).find((row) => row.playerId === String(playerId || "")) || null; }
function findSlot(state, key) { return descriptors(state, true).find((row) => row.key === key) || null; }
function setSlot(state, group, lineId, slot, value) {
  return { ...state, [group]: (state[group] || []).map((line) => line.id === lineId ? { ...line, slots: { ...line.slots, [slot]: String(value || "") } } : line) };
}
function removePlayer(state, playerId) {
  let next = state;
  for (const row of descriptors(state, true)) if (row.playerId === String(playerId || "")) next = setSlot(next, row.group, row.lineId, row.slot, "");
  return next;
}
function sanitizeLineup(state, players, includeThird = false) {
  const base = emptyLines(includeThird);
  const valid = new Set(players.map((player) => String(player.id)));
  let removed = 0;
  let retained = 0;
  const cleaned = {};
  for (const group of GROUPS) {
    const source = Array.isArray(state?.[group]) ? state[group] : [];
    cleaned[group] = base[group].map((fallback, index) => {
      const incoming = source.find((line) => String(line?.id || "") === fallback.id) || source[index] || {};
      const slots = {};
      for (const slot of Object.keys(fallback.slots)) {
        const id = String(incoming?.slots?.[slot] || "");
        if (id && valid.has(id)) { slots[slot] = id; retained += 1; }
        else { slots[slot] = ""; if (id) removed += 1; }
      }
      return { ...fallback, ...incoming, id: fallback.id, name: incoming?.name || fallback.name, slots };
    });
  }
  return { lineState: cleaned, removed, retained };
}
function duplicates(state) {
  const map = new Map();
  for (const row of descriptors(state, true)) if (row.playerId) map.set(row.playerId, [...(map.get(row.playerId) || []), row]);
  return [...map.entries()].filter(([, locations]) => locations.length > 1).map(([playerId, locations]) => ({ playerId, locations }));
}
function validateLineup(state, playerMap) {
  const errors = [];
  const warnings = [];
  for (const duplicate of duplicates(state)) errors.push({ key: `dup-${duplicate.playerId}`, text: "A player appears in multiple slots.", type: "duplicate" });
  for (const row of descriptors(state, true)) {
    if (!row.playerId) { warnings.push({ key: `missing-${row.key}`, text: `${row.lineName} needs ${row.slot}.`, type: "missing" }); continue; }
    const player = playerMap[row.playerId];
    if (!player) { errors.push({ key: `stale-${row.key}`, text: "A saved player no longer exists.", type: "stale" }); continue; }
    if (!posFit(player, row.slot)) errors.push({ key: `pos-${row.key}`, text: `${player.name} cannot play ${row.slot}.`, type: "position" });
    if (!player.availability?.placeable) errors.push({ key: `avail-${row.key}`, text: `${player.name} is unavailable.`, type: "availability" });
  }
  return { errors, warnings, incomplete: warnings.some((warning) => warning.type === "missing") };
}
function candidateScore(player, slot, mode, linePlayers = []) {
  const overall = Number(player?.overall) || 0;
  const position = idealScore(player, slot);
  const fit = chemistryFitScore(player, slot);
  const morale = profileValue(player, "morale", 50) ?? 50;
  const role = String(player?.role || "").toLowerCase();
  const duplicateRole = linePlayers.some((other) => String(other?.role || "").toLowerCase() === role);
  if (mode === "chemistry") {
    // Real pair math: score the candidate against everyone already in the unit.
    const bonds = linePlayers.map((other) => pairChemistry(player, other, null)?.score).filter(Number.isFinite);
    const bond = bonds.length ? bonds.reduce((sum, value) => sum + value, 0) / bonds.length : fit;
    return bond * 0.5 + overall * 0.3 + position * 0.2;
  }
  if (mode === "position") return position * 0.58 + overall * 0.42;
  if (mode === "roles") return overall * 0.48 + position * 0.4 + (duplicateRole ? -18 : 12);
  return overall * 0.72 + position * 0.28;
}
function chooseCandidate(candidates, slot, mode, linePlayers = []) {
  return [...candidates].filter((player) => posFit(player, slot)).sort((a, b) => candidateScore(b, slot, mode, linePlayers) - candidateScore(a, slot, mode, linePlayers) || String(a.name).localeCompare(String(b.name)))[0] || null;
}
function buildBestInitialLines(players, mode = "position", includeThird = false) {
  let state = emptyLines(includeThird);
  const eligible = players.filter((player) => player.availability?.placeable);
  const used = new Set();
  for (const row of descriptors(state, includeThird)) {
    const line = state[row.group].find((item) => item.id === row.lineId);
    const linePlayers = Object.values(line?.slots || {}).map((id) => players.find((player) => player.id === id)).filter(Boolean);
    const selected = chooseCandidate(eligible.filter((player) => !used.has(player.id)), row.slot, mode, linePlayers);
    if (!selected) continue;
    state = setSlot(state, row.group, row.lineId, row.slot, selected.id);
    used.add(selected.id);
  }
  return state;
}
function autoBuildState({ current, players, locks, mode, scope, includeThird }) {
  let next = cloneLines(current);
  const eligible = players.filter((player) => player.availability?.placeable);
  const matches = (row) => !scope || (row.group === scope.group && row.lineId === scope.lineId);
  const reserved = new Set();
  for (const row of descriptors(next, includeThird)) if ((!matches(row) || locks[row.key]) && row.playerId) reserved.add(row.playerId);
  for (const row of descriptors(next, includeThird)) if (matches(row) && !locks[row.key]) next = setSlot(next, row.group, row.lineId, row.slot, "");
  for (const row of descriptors(next, includeThird)) {
    if (!matches(row) || locks[row.key]) continue;
    const line = next[row.group].find((item) => item.id === row.lineId);
    const linePlayers = Object.values(line?.slots || {}).map((id) => eligible.find((player) => player.id === id)).filter(Boolean);
    const selected = chooseCandidate(eligible.filter((player) => !reserved.has(player.id)), row.slot, mode, linePlayers);
    if (!selected) continue;
    next = setSlot(next, row.group, row.lineId, row.slot, selected.id);
    reserved.add(selected.id);
  }
  return next;
}
function previewChanges(current, next, playerMap) {
  const before = new Map(descriptors(current, true).map((row) => [row.key, row]));
  return descriptors(next, true).filter((row) => (before.get(row.key)?.playerId || "") !== row.playerId).map((row) => ({ key: row.key, label: `${row.lineName} ${row.slot}`, before: playerMap[before.get(row.key)?.playerId]?.name || "Empty", after: playerMap[row.playerId]?.name || "Empty" }));
}
function backendChemistry(chemReport, players) {
  if (!chemReport || players.length < 2) return null;
  const ids = new Set(players.flatMap((player) => [...playerLookupIds(player)]));
  const rows = [...(Array.isArray(chemReport.lines) ? chemReport.lines : []), ...(Array.isArray(chemReport.pairs) ? chemReport.pairs : [])];
  const match = rows.find((row) => {
    const rowIds = (row.players || []).map((player) => canonicalPlayerId(player.id || player.player_id || ""));
    return rowIds.length >= 2 && rowIds.every((id) => ids.has(id)) && Math.abs(rowIds.length - players.length) <= 1;
  });
  if (!match) return null;
  const scheme = match.scheme_fit || {};
  return {
    score: clamp((match.chemistry ?? 50) + CHEM_DISPLAY_LIFT),
    label: match.label || "Neutral",
    morale: percentOrNull(scheme.morale),
    roleBalance: percentOrNull(scheme.role_balance),
    positionFit: percentOrNull(scheme.position_fit),
    linemateFit: percentOrNull(scheme.linemate_compatibility),
    familiarity: percentOrNull(scheme.familiarity),
    coachFit: percentOrNull(scheme.coach_system_fit),
    usageSatisfaction: percentOrNull(scheme.usage_satisfaction),
    handednessFit: percentOrNull(scheme.handedness_fit),
    factors: Array.isArray(match.factors) ? match.factors : [],
    concerns: Array.isArray(match.concerns) ? match.concerns : [],
    projected: false,
  };
}
function unitChemistryLabel(score) {
  if (score >= 92) return "Excellent";
  if (score >= 84) return "Strong";
  if (score >= 76) return "Stable";
  if (score >= 68) return "Uneven";
  return "Weak";
}
function calculateUnitChemistry(players, unitType = "forward", chemReport = null) {
  const selected = players.filter(Boolean);
  const live = backendChemistry(chemReport, selected);
  const styles = selected.map(resolvePlaystyle).filter(Boolean);
  const liveLooksStub = live && live.score >= 70 && live.score <= 94 && styles.some((style) => style !== "two_way");
  if (live && !liveLooksStub) return live;
  if (!selected.length) return { score: 0, label: "Empty", morale: null, roleBalance: 0, positionFit: 0, linemateFit: 0, familiarity: null, coachFit: null, usageSatisfaction: null, handednessFit: null, factors: [], concerns: ["Add players to calculate fit."], projected: true };
  const avgOverall = selected.reduce((sum, player) => sum + (Number(player.overall) || 0), 0) / selected.length;
  const moraleValues = selected.map((player) => profileValue(player, "morale", null)).filter((value) => value != null);
  const morale = moraleValues.length ? moraleValues.reduce((sum, value) => sum + value, 0) / moraleValues.length : null;
  const positions = selected.map((player) => player.position);
  const roles = new Set(selected.map((player) => String(player.role || "")).filter(Boolean));
  const roleBalance = clamp((roles.size / Math.max(1, selected.length)) * 100);
  let positionFit = 70;
  let handednessFit = null;
  if (unitType === "forward") positionFit = positions.includes("C") && positions.some((pos) => pos === "LW" || pos === "F") && positions.some((pos) => pos === "RW" || pos === "F") ? 94 : 68;
  if (unitType === "defense") {
    positionFit = positions.some((pos) => pos === "LD" || pos === "D") && positions.some((pos) => pos === "RD" || pos === "D") ? 94 : 62;
    const hands = selected.map((player) => player.handedness);
    handednessFit = hands.includes("L") && hands.includes("R") ? 95 : hands.length === 2 && hands[0] === hands[1] ? 58 : 75;
  }
  if (unitType === "goalie") positionFit = selected.every((player) => isGoalie(player.position)) ? 100 : 0;
  const linemateFit = clamp(selected.reduce((sum, player) => sum + (profileValue(player, "adaptability", 50) ?? 50), 0) / selected.length);
  const bonds = [];
  for (let i = 0; i < selected.length; i += 1) {
    for (let j = i + 1; j < selected.length; j += 1) {
      const bond = pairChemistry(selected[i], selected[j], null)?.score;
      if (Number.isFinite(bond)) bonds.push(bond);
    }
  }
  const score = bonds.length
    ? clamp(bonds.reduce((sum, value) => sum + value, 0) / bonds.length)
    : clamp(avgOverall * 0.34 + (morale ?? 50) * 0.16 + roleBalance * 0.14 + positionFit * 0.22 + linemateFit * 0.14);
  const label = unitChemistryLabel(score);
  const concerns = [];
  const factors = [];
  if (positionFit < 75) concerns.push("Position balance needs work.");
  if (roleBalance < 60) concerns.push("Roles overlap too heavily.");
  const moraleIsReal = selected.some((player) => !isStubPsych(player));
  if (moraleIsReal && morale != null && morale < 48) concerns.push("Unit morale is low.");
  if (!concerns.length) factors.push("Position balance supports this unit.");
  if (roleBalance >= 70) factors.push("Roles provide useful variety.");
  return { score, label, morale: morale == null ? null : clamp(morale), roleBalance, positionFit, linemateFit, familiarity: null, coachFit: null, usageSatisfaction: null, handednessFit, factors, concerns, projected: true };
}
function unitWarnings(line, group, playerMap, chemReport, duplicateIds) {
  const warnings = [];
  const entries = Object.entries(line?.slots || {});
  const players = entries.map(([slot, id]) => ({ slot, player: playerMap[String(id || "")] })).filter((entry) => entry.player);
  for (const [slot, id] of entries) {
    if (!id) { warnings.push({ key: `missing-${line.id}-${slot}`, text: `${line.name} needs ${slot}.` }); continue; }
    const player = playerMap[String(id)];
    if (!player) { warnings.push({ key: `stale-${line.id}-${slot}`, text: "This assignment is stale." }); continue; }
    if (!posFit(player, slot)) warnings.push({ key: `invalid-${line.id}-${slot}`, text: `${player.name} cannot play ${slot}.` });
    if (!player.availability?.placeable) warnings.push({ key: `unavailable-${line.id}-${slot}`, text: `${player.name} is unavailable.` });
    if (duplicateIds.has(player.id)) warnings.push({ key: `duplicate-${line.id}-${slot}`, text: `${player.name} is duplicated.` });
  }
  if (group === "defense" && players.length === 2) {
    const hands = players.map((entry) => entry.player.handedness);
    if (hands[0] && hands[1] && hands[0] === hands[1]) warnings.push({ key: `hands-${line.id}`, text: "Same-handed defence pair." });
  }
  const type = group === "defense" ? "defense" : group === "goalies" ? "goalie" : "forward";
  const chemistry = calculateUnitChemistry(players.map((entry) => entry.player), type, chemReport);
  if (chemistry.score > 0 && chemistry.score < 58) warnings.push({ key: `chem-${line.id}`, text: "Unit chemistry is weak." });
  return warnings;
}
function sortPoolPlayers(list, sort, focusSlot) {
  const order = { G: 0, LD: 1, RD: 2, D: 3, C: 4, LW: 5, RW: 6, F: 7 };
  return [...list].sort((a, b) => {
    if (sort === "position") return (order[a.position] ?? 9) - (order[b.position] ?? 9) || a.name.localeCompare(b.name);
    if (sort === "morale") return (b.morale ?? -Infinity) - (a.morale ?? -Infinity) || b.overall - a.overall;
    if (sort === "chemistry" && focusSlot) return chemistryFitScore(b, focusSlot) - chemistryFitScore(a, focusSlot) || b.overall - a.overall;
    return b.overall - a.overall || a.name.localeCompare(b.name);
  });
}
function teamIdentity(franchiseState, props) {
  const team = franchiseState?.user_team || franchiseState?.team || franchiseState?.current_team || props.currentTeam || props.team || {};
  return {
    abbreviation: String(team?.abbreviation || team?.abbr || franchiseState?.user_team_abbreviation || franchiseState?.user_team_abbr || "TEAM").slice(0, 4),
    logo: team?.logo || team?.logo_url || team?.logoUrl || team?.team_logo || null,
  };
}

function EditLinesStyles() {
  return <style>{`
.linebuilder-root{--nv:var(--ops-navy,#06111d);--nv2:var(--ops-navy-deep,#04101a);--panel:var(--ops-panel,rgba(9,25,38,.97));--panel2:var(--ops-panel-2,rgba(12,35,52,.92));--line:var(--ops-grid,rgba(156,218,236,.16));--line2:var(--ops-grid-2,rgba(115,229,241,.3));--cyan:var(--ops-cyan,#13d8e7);--gold:var(--ops-gold,#e9a83c);--green:var(--ops-success,#52df94);--red:var(--ops-injury,#ff606d);--text:var(--ops-text,#e9f7fb);--muted:var(--ops-text-secondary,#8096a8);--muted2:var(--ops-text-disabled,#607789);width:100%;height:100vh;height:100dvh;overflow:hidden;color:var(--text);background:linear-gradient(180deg,var(--nv),var(--nv2));font-family:var(--font-ops-ui,Inter,system-ui,sans-serif);display:grid;grid-template-columns:88px minmax(0,1fr)}
.linebuilder-root *{box-sizing:border-box}
.linebuilder-root button,.linebuilder-root input{font:inherit}
.linebuilder-root button:focus-visible,.linebuilder-root input:focus-visible,.linebuilder-root [tabindex]:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}

.linebuilder-root .lb-sidebar{height:100%;padding:10px 8px;border-right:1px solid var(--line);background:rgba(4,14,24,.94);display:flex;flex-direction:column;gap:10px;overflow:hidden}
.linebuilder-root .lb-team-mark{height:46px;flex:0 0 46px;border:1px solid var(--line2);border-radius:6px;display:grid;place-items:center;color:var(--cyan);font-size:12px;font-weight:900;letter-spacing:.09em;background:rgba(19,216,231,.08)}
.linebuilder-root .lb-nav{flex:1;min-height:0;display:grid;grid-template-rows:repeat(5,minmax(0,1fr));gap:8px}
.linebuilder-root .lb-nav-btn{width:100%;height:100%;padding:6px 4px;border:1px solid transparent;border-radius:6px;color:var(--muted);background:transparent;cursor:pointer;display:grid;place-items:center;align-content:center;gap:3px}
.linebuilder-root .lb-nav-btn:hover,.linebuilder-root .lb-nav-btn.active{color:var(--cyan);border-color:var(--line2);background:rgba(19,216,231,.12);box-shadow:inset 3px 0 0 var(--cyan)}
.linebuilder-root .lb-nav-label{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.linebuilder-root .lb-glyph{width:22px;height:22px;display:inline-grid;place-items:center;border:1px solid currentColor;border-radius:3px;font-size:11px;font-weight:900;line-height:1}

.linebuilder-root .lb-shell{min-width:0;min-height:0;height:100%;padding:10px;display:grid;grid-template-rows:58px 34px minmax(0,1fr);gap:8px;overflow:hidden}
.linebuilder-root .lb-header{height:58px;border:1px solid var(--line);border-radius:8px;background:var(--panel);display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 12px;position:relative;z-index:20}
.linebuilder-root .lb-title-group{min-width:0;display:flex;align-items:center;gap:10px}
.linebuilder-root .lb-logo{width:40px;height:40px;flex:0 0 40px;border:1px solid var(--line);border-radius:6px;background:rgba(0,0,0,.2);display:grid;place-items:center;overflow:hidden;color:var(--cyan);font-size:11px;font-weight:900}
.linebuilder-root .lb-logo img{width:32px;height:32px;object-fit:contain}
.linebuilder-root .lb-title{margin:0;font-size:clamp(17px,1.7vw,21px);line-height:1;letter-spacing:.02em;font-weight:950;text-transform:uppercase;white-space:nowrap}
.linebuilder-root .lb-subtitle{margin-top:4px;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;white-space:nowrap;display:flex;align-items:center;gap:6px}
.linebuilder-root .lb-live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block}
.linebuilder-root .lb-live-dot.projected{background:var(--gold)}
.linebuilder-root .lb-actions{display:flex;align-items:center;justify-content:flex-end;gap:6px}
.linebuilder-root .lb-btn,.linebuilder-root .lb-icon{min-height:34px;border:1px solid var(--line);border-radius:6px;color:var(--text);background:rgba(255,255,255,.02);cursor:pointer;font-size:11px;font-weight:850;letter-spacing:.06em;text-transform:uppercase}
.linebuilder-root .lb-btn{padding:0 12px;display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap}
.linebuilder-root .lb-icon{width:34px;padding:0;display:inline-grid;place-items:center}
.linebuilder-root .lb-btn:hover:not(:disabled),.linebuilder-root .lb-icon:hover:not(:disabled){border-color:var(--line2);background:rgba(19,216,231,.12)}
.linebuilder-root .lb-btn:disabled,.linebuilder-root .lb-icon:disabled{opacity:.38;cursor:not-allowed}
.linebuilder-root .lb-btn.primary{color:var(--nv2);border-color:var(--cyan);background:var(--cyan);font-weight:900}
.linebuilder-root .lb-btn.subtle{color:var(--muted);border-color:var(--line);background:transparent;font-weight:750}
.linebuilder-root .lb-btn.subtle:hover:not(:disabled){color:var(--text);border-color:var(--line2)}
.linebuilder-root .lb-btn.subtle.danger{background:transparent;border-color:transparent;color:rgba(190,150,152,.85)}
.linebuilder-root .lb-btn.subtle.danger:hover:not(:disabled){background:rgba(255,96,109,.1);border-color:rgba(255,96,109,.45);color:#ffd5d5}
.linebuilder-root .lb-btn.danger{color:#ffd5d5;border-color:rgba(255,96,109,.45);background:rgba(255,96,109,.08)}
.linebuilder-root .lb-save-error{color:var(--red);font-size:11px;font-weight:800}

.linebuilder-root .lb-status{height:34px;border:1px solid var(--line);border-radius:6px;background:rgba(0,0,0,.2);padding:0 4px;display:flex;align-items:center;overflow:hidden}
.linebuilder-root .lb-pill{height:100%;padding:0 11px;border-right:1px solid var(--line);display:flex;align-items:center;gap:7px;white-space:nowrap}
.linebuilder-root .lb-pill:last-child{border-right:0}
.linebuilder-root .lb-pill-label{color:var(--muted2);font-size:11px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}
.linebuilder-root .lb-pill-value{font-size:12px;font-weight:900;font-variant-numeric:tabular-nums}
.linebuilder-root .lb-pill-value.good{color:var(--green)}
.linebuilder-root .lb-pill-value.warn{color:var(--gold)}
.linebuilder-root .lb-pill-value.bad{color:var(--red)}
.linebuilder-root .lb-chemmeter{display:flex;gap:2px;margin-left:auto;padding:0 10px}
.linebuilder-root .lb-chemseg{width:11px;height:7px;border-radius:1px;background:rgba(255,255,255,.09)}
.linebuilder-root .lb-chemseg.strong{background:var(--green)}
.linebuilder-root .lb-chemseg.forming{background:var(--gold)}
.linebuilder-root .lb-chemseg.weak{background:var(--red)}

.linebuilder-root .lb-workspace{min-width:0;min-height:0;height:100%;display:grid;grid-template-columns:240px minmax(0,1fr) 260px;gap:8px;overflow:hidden}
.linebuilder-root .lb-region{min-width:0;min-height:0;height:100%;border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden;display:grid;grid-template-rows:38px minmax(0,1fr)}
.linebuilder-root .lb-region-head{height:38px;padding:0 10px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px;background:rgba(0,0,0,.18)}
.linebuilder-root .lb-region-title{margin:0;font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}
.linebuilder-root .lb-region-note{color:var(--muted);font-size:11px;font-weight:800;white-space:nowrap}

.linebuilder-root .lb-pool{min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr);gap:6px;padding:8px;overflow:hidden}
.linebuilder-root .lb-search-wrap{position:relative}
.linebuilder-root .lb-search{width:100%;height:32px;padding:0 28px 0 9px;border:1px solid var(--line);border-radius:5px;color:var(--text);background:rgba(0,0,0,.22);outline:none;font-size:11px}
.linebuilder-root .lb-search::placeholder{color:var(--muted2)}
.linebuilder-root .lb-clear{position:absolute;top:2px;right:2px;width:28px;height:28px;border:0;border-radius:4px;color:var(--muted);background:transparent;cursor:pointer}
.linebuilder-root .lb-player-list{min-height:0;height:100%;display:flex;flex-direction:column;overflow-x:hidden;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(19,216,231,.35) rgba(0,0,0,.22)}
.linebuilder-root .lb-player{flex:0 0 auto;min-height:40px;padding:0 8px 0 6px;border:0;border-bottom:1px solid var(--line);border-left:3px solid transparent;background:transparent;display:grid;grid-template-columns:28px minmax(0,1fr) 34px;gap:7px;align-items:center;cursor:grab;text-align:left}
.linebuilder-root .lb-player:hover{background:rgba(19,216,231,.1)}
.linebuilder-root .lb-player.selected{border-left-color:var(--cyan);background:rgba(19,216,231,.14)}
.linebuilder-root .lb-player.assigned{border-left-color:rgba(128,150,168,.5);background:rgba(0,0,0,.16)}
.linebuilder-root .lb-player.assigned .lb-player-name{color:#b7c9d3}
.linebuilder-root .lb-player.disabled{cursor:not-allowed;opacity:.42}
.linebuilder-root .lb-player.locked{border-left-color:var(--gold)}
.linebuilder-root .lb-player .player-headshot{width:26px!important;height:26px!important}
.linebuilder-root .lb-player-copy{min-width:0}
.linebuilder-root .lb-player-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;font-weight:800;line-height:1.1}
.linebuilder-root .lb-player-meta{margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted2);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.linebuilder-root .lb-player-side{display:grid;justify-items:end;align-content:center;gap:3px}
.linebuilder-root .lb-player-ovr{color:var(--cyan);font-size:13px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1}
.linebuilder-root .lb-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green)}
.linebuilder-root .lb-dot.warn{background:var(--gold)}
.linebuilder-root .lb-dot.bad{background:var(--red)}

.linebuilder-root .lb-board-region{grid-template-rows:34px minmax(0,1fr)}
.linebuilder-root .lb-modebar{height:34px;padding:0 6px;border-bottom:1px solid var(--line);display:flex;align-items:stretch;background:rgba(0,0,0,.2)}
.linebuilder-root .lb-mode{padding:0 14px;border:0;border-bottom:2px solid transparent;color:var(--muted);background:transparent;cursor:pointer;display:inline-flex;align-items:center;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.linebuilder-root .lb-mode:hover{color:var(--text)}
.linebuilder-root .lb-mode.active{color:var(--cyan);border-bottom-color:var(--cyan)}
.linebuilder-root .lb-mobile-tools{display:none;margin-left:auto;gap:4px;align-items:center;padding:3px 0}

/* ---- formation board ---- */
.linebuilder-root .fm-scroll{min-height:0;height:100%;overflow-y:auto;overflow-x:hidden;padding:10px 14px 18px;scrollbar-width:thin;scrollbar-color:rgba(19,216,231,.35) rgba(0,0,0,.22)}
.linebuilder-root .fm-section{display:block;margin-bottom:6px}
.linebuilder-root .fm-section-head{display:flex;align-items:center;gap:10px;margin:10px 0 4px}
.linebuilder-root .fm-section-title{font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:var(--muted2);white-space:nowrap}
.linebuilder-root .fm-section-rule{flex:1;height:1px;background:var(--line)}
.linebuilder-root .fm-section-note{font-size:11px;font-weight:700;color:var(--muted2);white-space:nowrap}
.linebuilder-root .fm-unit{position:relative;width:100%;border-radius:8px;cursor:pointer}
.linebuilder-root .fm-unit.selected{background:rgba(19,216,231,.06);box-shadow:inset 0 0 0 1px var(--line2)}
.linebuilder-root .fm-unit.forwards{height:178px}
.linebuilder-root .fm-unit.defense{height:118px}
.linebuilder-root .fm-unit.goalies{height:118px}
.linebuilder-root .fm-unit.ppf{height:178px}
.linebuilder-root .fm-unit.ppd{height:118px}
.linebuilder-root .fm-unit.pkbox{height:224px}
.linebuilder-root .fm-unit.isolated{border:1px dashed rgba(128,150,168,.35);background:rgba(0,0,0,.12)}
.linebuilder-root .fm-links{position:absolute;inset:0;width:100%;height:100%;z-index:1;pointer-events:none}
.linebuilder-root .fm-link{stroke:rgba(128,150,168,.35);stroke-width:3;stroke-linecap:round}
.linebuilder-root .fm-link.strong{stroke:var(--green)}
.linebuilder-root .fm-link.forming{stroke:var(--gold)}
.linebuilder-root .fm-link.weak{stroke:var(--red)}
.linebuilder-root .fm-link.empty{stroke:rgba(128,150,168,.22);stroke-dasharray:3 4}
.linebuilder-root .fm-link.dashed{stroke-width:2;stroke-dasharray:2 4}
.linebuilder-root .fm-badge{position:absolute;z-index:3;transform:translate(-50%,-50%);min-width:26px;padding:2px 5px;border-radius:11px;border:1px solid currentColor;background:var(--nv2);font-size:10px;font-weight:900;text-align:center;font-variant-numeric:tabular-nums;line-height:1.15;pointer-events:none}
.linebuilder-root .fm-badge.strong{color:var(--green)}
.linebuilder-root .fm-badge.forming{color:var(--gold)}
.linebuilder-root .fm-badge.weak{color:var(--red)}
.linebuilder-root .fm-badge.empty{color:var(--muted2)}
.linebuilder-root .fm-unit-tag{position:absolute;left:0;top:50%;transform:translateY(-50%);z-index:3;display:flex;flex-direction:column;align-items:center;gap:1px;width:52px}
.linebuilder-root .fm-unit-kicker{color:var(--muted2);font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
.linebuilder-root .fm-unit-num{font-size:26px;font-weight:950;line-height:1;font-variant-numeric:tabular-nums}
.linebuilder-root .fm-unit-score{margin-top:3px;font-size:12px;font-weight:900;color:var(--cyan);font-variant-numeric:tabular-nums}
.linebuilder-root .fm-unit-score.warn{color:var(--gold)}
.linebuilder-root .fm-unit-score.bad{color:var(--red)}
.linebuilder-root .fm-unit-warn{margin-top:3px;min-width:22px;height:18px;padding:0 5px;border:1px solid rgba(233,168,60,.45);border-radius:4px;background:rgba(233,168,60,.14);color:var(--gold);font-size:10px;font-weight:900;cursor:pointer}
.linebuilder-root .fm-unit-menu-btn{margin-top:3px;min-width:22px;height:18px;border:1px solid var(--line);border-radius:4px;background:transparent;color:var(--muted);font-size:10px;font-weight:900;cursor:pointer}
.linebuilder-root .fm-unit-menu-btn:hover{color:var(--cyan);border-color:var(--line2)}

.linebuilder-root .fm-card{position:absolute;z-index:2;border:1px solid var(--line2);border-top:3px solid rgba(128,150,168,.5);border-radius:7px;background:var(--panel2);padding:7px 6px 6px;text-align:center;cursor:pointer;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px}
.linebuilder-root .fm-card:hover{border-color:var(--line2);background:rgba(19,216,231,.08)}
.linebuilder-root .fm-card.strong{border-top-color:var(--green)}
.linebuilder-root .fm-card.forming{border-top-color:var(--gold)}
.linebuilder-root .fm-card.weak{border-top-color:var(--red)}
.linebuilder-root .fm-card.empty{border:1px dashed rgba(128,150,168,.4);border-top:1px dashed rgba(128,150,168,.4);background:rgba(0,0,0,.18)}
.linebuilder-root .fm-card.fresh{border:1px dashed var(--red);border-top:3px solid var(--red);background:rgba(255,96,109,.07)}
.linebuilder-root .fm-card.selected{box-shadow:inset 0 0 0 2px var(--cyan);background:rgba(19,216,231,.14)}
.linebuilder-root .fm-card.locked{border-right:3px solid var(--gold)}
.linebuilder-root .fm-card.valid{border-color:var(--cyan);background:rgba(19,216,231,.14)}
.linebuilder-root .fm-card.swap{border-color:var(--gold);background:rgba(233,168,60,.14)}
.linebuilder-root .fm-card.invalid{border-color:var(--red);background:rgba(255,96,109,.1);cursor:not-allowed}
.linebuilder-root .fm-card.unavailable{opacity:.55}
.linebuilder-root .fm-card-slot{position:absolute;top:4px;left:6px;color:var(--muted2);font-size:10px;font-weight:900;letter-spacing:.06em;text-transform:uppercase}
.linebuilder-root .fm-card-flag{position:absolute;top:4px;right:6px;font-size:9px;font-weight:900;letter-spacing:.04em;color:var(--red)}
.linebuilder-root .fm-card-ovr{font-size:26px;font-weight:950;line-height:1;color:var(--cyan);font-variant-numeric:tabular-nums;margin-top:6px}
.linebuilder-root .fm-card.fresh .fm-card-ovr{color:var(--red);font-size:17px}
.linebuilder-root .fm-card-name{margin-top:3px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px;font-weight:900;letter-spacing:-.01em;line-height:1.05}
.linebuilder-root .fm-card-meta{margin-top:2px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted2);font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.linebuilder-root .fm-card-drop{color:var(--red);font-size:10px;font-weight:900}
.linebuilder-root .fm-pips{display:flex;justify-content:center;gap:3px;margin-top:5px}
.linebuilder-root .fm-pip{width:5px;height:5px;border-radius:50%;background:rgba(128,150,168,.35)}
.linebuilder-root .fm-pip.strong{background:var(--green)}
.linebuilder-root .fm-pip.forming{background:var(--gold)}
.linebuilder-root .fm-pip.weak{background:var(--red)}
.linebuilder-root .fm-card-empty{color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.linebuilder-root .fm-card-actions{position:absolute;bottom:3px;right:3px;display:flex;gap:3px;z-index:4}
.linebuilder-root .fm-mini{width:19px;height:19px;border:1px solid var(--line);border-radius:3px;color:var(--muted);background:rgba(0,0,0,.4);cursor:pointer;display:grid;place-items:center;font-size:10px;font-weight:900}
.linebuilder-root .fm-mini:hover{color:var(--cyan);border-color:var(--line2)}
.linebuilder-root .fm-mini.danger:hover{color:var(--red);border-color:rgba(255,96,109,.45)}
.linebuilder-root .fm-depth{position:relative;height:26px;display:flex;align-items:center;justify-content:center}
.linebuilder-root .fm-depth-line{width:0;height:100%;border-left:2px dashed rgba(128,150,168,.3)}
.linebuilder-root .fm-depth-line.strong{border-left-color:var(--green)}
.linebuilder-root .fm-depth-line.forming{border-left-color:var(--gold)}
.linebuilder-root .fm-depth-line.weak{border-left-color:var(--red)}
.linebuilder-root .fm-depth-badge{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:1px 6px;border-radius:10px;border:1px solid currentColor;background:var(--nv2);font-size:9px;font-weight:900;color:var(--muted2)}
.linebuilder-root .fm-depth-badge.strong{color:var(--green)}
.linebuilder-root .fm-depth-badge.forming{color:var(--gold)}
.linebuilder-root .fm-depth-badge.weak{color:var(--red)}
.linebuilder-root .fm-isolated-note{position:absolute;top:6px;left:10px;font-size:10px;font-weight:800;color:var(--muted2);letter-spacing:.04em}
.linebuilder-root .fm-legend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;padding:12px 6px 4px;border-top:1px solid var(--line);margin-top:12px}
.linebuilder-root .fm-legend-item{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:10px;font-weight:700}
.linebuilder-root .fm-legend-swatch{width:16px;height:3px;border-radius:2px;background:var(--green)}
.linebuilder-root .fm-legend-swatch.forming{background:var(--gold)}
.linebuilder-root .fm-legend-swatch.weak{background:var(--red)}
.linebuilder-root .fm-legend-swatch.depth{height:0;border-top:2px dashed rgba(128,150,168,.55);background:transparent}
.linebuilder-root .fm-unit-menu{position:absolute;left:56px;top:6px;z-index:40;width:160px;padding:5px;border:1px solid var(--line2);border-radius:6px;background:var(--panel);box-shadow:0 24px 70px rgba(0,0,0,.5);display:grid;gap:3px}
.linebuilder-root .lb-menu-action{min-height:29px;padding:0 8px;border:0;border-radius:4px;color:var(--text);background:transparent;cursor:pointer;text-align:left;font-size:11px;font-weight:750}
.linebuilder-root .lb-menu-action:hover:not(:disabled){background:rgba(19,216,231,.13);color:var(--cyan)}
.linebuilder-root .lb-menu-action:disabled{opacity:.35}

/* ---- inspector ---- */
.linebuilder-root .lb-inspector{min-height:0;display:grid;grid-template-rows:32px minmax(0,1fr);overflow:hidden}
.linebuilder-root .lb-tabs{padding:0 2px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:rgba(0,0,0,.16)}
.linebuilder-root .lb-tab{border:0;border-bottom:2px solid transparent;color:var(--muted);background:transparent;cursor:pointer;font-size:10px;font-weight:850;text-transform:uppercase;letter-spacing:.03em;min-height:30px}
.linebuilder-root .lb-tab.active{color:var(--cyan);border-bottom-color:var(--cyan)}
.linebuilder-root .lb-inspector-body{min-height:0;padding:10px;overflow:auto;display:flex;flex-direction:column;gap:9px;scrollbar-width:thin;scrollbar-color:rgba(19,216,231,.35) rgba(0,0,0,.22)}
.linebuilder-root .lb-inspector-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:15px;font-weight:900}
.linebuilder-root .lb-inspector-sub{margin-top:3px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}
.linebuilder-root .lb-gauge{display:flex;align-items:center;gap:10px}
.linebuilder-root .lb-gauge-score{color:var(--cyan);font-size:34px;line-height:1;font-weight:950;font-variant-numeric:tabular-nums}
.linebuilder-root .lb-gauge-score.warn{color:var(--gold)}
.linebuilder-root .lb-gauge-score.bad{color:var(--red)}
.linebuilder-root .lb-gauge-label{font-size:12px;font-weight:850;text-transform:uppercase;letter-spacing:.03em}
.linebuilder-root .lb-gauge-source{margin-top:3px;color:var(--muted2);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.linebuilder-root .lb-bar-row{display:block}
.linebuilder-root .lb-bar-head{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:3px}
.linebuilder-root .lb-bar-head span:last-child{color:var(--text);font-variant-numeric:tabular-nums}
.linebuilder-root .lb-bar{height:5px;border-radius:3px;background:rgba(255,255,255,.07);overflow:hidden}
.linebuilder-root .lb-bar-fill{height:100%;border-radius:3px;background:var(--cyan)}
.linebuilder-root .lb-bar-fill.warn{background:var(--gold)}
.linebuilder-root .lb-bar-fill.bad{background:var(--red)}
.linebuilder-root .lb-block{border-top:1px solid var(--line);padding-top:9px;display:flex;flex-direction:column;gap:6px}
.linebuilder-root .lb-block-title{font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--muted2)}
.linebuilder-root .lb-note{display:flex;gap:7px;align-items:flex-start}
.linebuilder-root .lb-note-mark{font-size:11px;font-weight:900;line-height:1.35}
.linebuilder-root .lb-note-mark.good{color:var(--green)}
.linebuilder-root .lb-note-mark.warn{color:var(--gold)}
.linebuilder-root .lb-note-mark.bad{color:var(--red)}
.linebuilder-root .lb-note-text{font-size:11px;color:var(--muted);line-height:1.4}
.linebuilder-root .lb-kv{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:26px;font-size:11px}
.linebuilder-root .lb-kv-label{color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:10px}
.linebuilder-root .lb-kv-value{font-weight:850;text-align:right;font-variant-numeric:tabular-nums}
.linebuilder-root .lb-kv-value.ok{color:var(--green)}
.linebuilder-root .lb-kv-value.warn{color:var(--gold)}
.linebuilder-root .lb-linkrow{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:28px;border-bottom:1px solid var(--line);font-size:11px}
.linebuilder-root .lb-linkrow:last-child{border-bottom:0}
.linebuilder-root .lb-linkrow-names{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-weight:700}
.linebuilder-root .lb-linkrow-score{font-weight:900;font-variant-numeric:tabular-nums}
.linebuilder-root .lb-linkrow-score.strong{color:var(--green)}
.linebuilder-root .lb-linkrow-score.forming{color:var(--gold)}
.linebuilder-root .lb-linkrow-score.weak{color:var(--red)}
.linebuilder-root .lb-inspector-player{padding:8px;border:1px solid var(--line);border-radius:6px;background:rgba(0,0,0,.16);display:grid;grid-template-columns:48px minmax(0,1fr);gap:9px;align-items:center}
.linebuilder-root .lb-inspector-player .player-headshot{width:44px!important;height:44px!important}
.linebuilder-root .lb-inspector-player-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;font-weight:900}
.linebuilder-root .lb-inspector-player-meta{margin-top:3px;color:var(--muted2);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.linebuilder-root .lb-inspector-actions{margin-top:2px;display:grid;grid-template-columns:1fr 1fr;gap:6px}
.linebuilder-root .lb-compare{min-height:42px;display:flex;align-items:center;justify-content:space-between;gap:8px;border-bottom:1px solid var(--line);padding:5px 0}
.linebuilder-root .lb-compare-name{font-size:12px;font-weight:900;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.linebuilder-root .lb-compare-meta{margin-top:2px;color:var(--muted);font-size:10px;text-transform:uppercase}
.linebuilder-root .lb-empty,.linebuilder-root .lb-loading{height:100%;display:grid;place-items:center;padding:18px;color:var(--muted);text-align:center;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.linebuilder-root .lb-skeleton{width:min(220px,80%);height:8px;border-radius:3px;background:linear-gradient(90deg,rgba(255,255,255,.03),rgba(19,216,231,.12),rgba(255,255,255,.03));background-size:200% 100%;animation:lb-shimmer 1.2s linear infinite}
@keyframes lb-shimmer{from{background-position:100% 0}to{background-position:-100% 0}}
.linebuilder-root .lb-popover{position:absolute;top:62px;right:14px;z-index:60;width:300px;max-height:min(440px,calc(100dvh - 100px));padding:9px;border:1px solid var(--line2);border-radius:8px;background:var(--panel);box-shadow:0 24px 70px rgba(0,0,0,.5);display:grid;gap:7px;overflow:hidden}
.linebuilder-root .lb-popover-title{margin:0;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.06em}
.linebuilder-root .lb-auto-options{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.linebuilder-root .lb-auto-option{min-height:32px;padding:0 7px;border:1px solid var(--line);border-radius:5px;color:var(--muted);background:rgba(0,0,0,.16);cursor:pointer;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.03em}
.linebuilder-root .lb-auto-option.active{color:var(--cyan);border-color:var(--line2);background:rgba(19,216,231,.13)}
.linebuilder-root .lb-preview{max-height:210px;display:grid;align-content:start;border:1px solid var(--line);border-radius:6px;overflow-y:auto}
.linebuilder-root .lb-preview-row{padding:5px 7px;border-bottom:1px solid var(--line);display:grid;gap:2px}
.linebuilder-root .lb-preview-label{color:var(--muted);font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.06em}
.linebuilder-root .lb-preview-change{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px}
.linebuilder-root .lb-popover-actions{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.linebuilder-root .lb-toast{position:fixed;right:18px;bottom:18px;z-index:100;min-width:230px;max-width:340px;padding:9px 11px;border:1px solid var(--line2);border-left:3px solid var(--cyan);border-radius:6px;background:var(--panel);box-shadow:0 24px 70px rgba(0,0,0,.5);display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}
.linebuilder-root .lb-toast.success{border-left-color:var(--green)}
.linebuilder-root .lb-toast.warning{border-left-color:var(--gold)}
.linebuilder-root .lb-toast.error{border-left-color:var(--red)}
.linebuilder-root .lb-toast-message{font-size:11px;font-weight:800}
.linebuilder-root .lb-toast-actions{display:flex;gap:4px}
.linebuilder-root .lb-drawer-close{display:none}
.linebuilder-root .lb-live{position:fixed;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}

@media(max-width:1400px){.linebuilder-root .lb-workspace{grid-template-columns:220px minmax(0,1fr) 240px}}
@media(max-width:1180px){
.linebuilder-root .lb-workspace{grid-template-columns:210px minmax(0,1fr)}
.linebuilder-root .lb-inspector-region{position:fixed;top:70px;right:8px;bottom:8px;width:290px;height:auto;z-index:70;transform:translateX(calc(100% + 20px));transition:transform .18s;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.linebuilder-root .lb-inspector-region.open{transform:translateX(0)}
.linebuilder-root .lb-drawer-close{display:inline-grid}
.linebuilder-root .lb-mobile-tools{display:flex}}
@media(max-width:900px){
.linebuilder-root{grid-template-columns:1fr;grid-template-rows:58px minmax(0,1fr)}
.linebuilder-root .lb-sidebar{height:58px;padding:6px 8px;border-right:0;border-bottom:1px solid var(--line);flex-direction:row;align-items:center}
.linebuilder-root .lb-nav{flex:1;display:flex;grid-template-rows:none;gap:5px}
.linebuilder-root .lb-nav-btn{min-width:56px;min-height:44px;height:auto;display:flex}
.linebuilder-root .lb-workspace{grid-template-columns:1fr}
.linebuilder-root .lb-pool-region{position:fixed;top:70px;left:8px;bottom:8px;width:260px;height:auto;z-index:72;transform:translateX(calc(-100% - 20px));transition:transform .18s;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.linebuilder-root .lb-pool-region.open{transform:translateX(0)}
.linebuilder-root .fm-card-name{font-size:12px}
.linebuilder-root .fm-card-ovr{font-size:21px}}
@media(prefers-reduced-motion:reduce){.linebuilder-root .lb-skeleton{animation:none}}
`}</style>;
}

/* ---------------------------------------------------------------------------
   Formation geometry. All values are percentages of the unit container so the
   board scales with the workspace instead of relying on fixed pixel columns.
--------------------------------------------------------------------------- */
const FWD_GEO = {
  LW: { left: 11, top: 44, width: 22, height: 52, cx: 22, cy: 70 },
  C: { left: 39, top: 4, width: 22, height: 52, cx: 50, cy: 30 },
  RW: { left: 67, top: 44, width: 22, height: 52, cx: 78, cy: 70 },
};
const DEF_GEO = {
  LD: { left: 26, top: 11, width: 21, height: 78, cx: 36.5, cy: 50 },
  RD: { left: 54, top: 11, width: 21, height: 78, cx: 64.5, cy: 50 },
};
const GOALIE_GEO = {
  Starter: { left: 22, top: 11, width: 21, height: 78 },
  Backup: { left: 46, top: 11, width: 21, height: 78 },
  Third: { left: 70, top: 11, width: 21, height: 78 },
};
const PP_GEO = {
  LW: { left: 11, top: 26, width: 21, height: 30, cx: 21.5, cy: 41 },
  C: { left: 39, top: 2, width: 21, height: 30, cx: 49.5, cy: 17 },
  RW: { left: 67, top: 26, width: 21, height: 30, cx: 77.5, cy: 41 },
  LD: { left: 26, top: 68, width: 21, height: 30, cx: 36.5, cy: 83 },
  RD: { left: 54, top: 68, width: 21, height: 30, cx: 64.5, cy: 83 },
};
const PK_GEO = {
  F1: { left: 14, top: 5, width: 26, height: 38, cx: 27, cy: 24 },
  F2: { left: 58, top: 5, width: 26, height: 38, cx: 71, cy: 24 },
  D1: { left: 14, top: 57, width: 26, height: 38, cx: 27, cy: 76 },
  D2: { left: 58, top: 57, width: 26, height: 38, cx: 71, cy: 76 },
};
const FWD_LINKS = [["LW", "C"], ["C", "RW"], ["LW", "RW"]];
const DEF_LINKS = [["LD", "RD"]];
const PP_LINKS = [["LW", "C"], ["C", "RW"], ["LW", "RW"], ["LD", "RD"]];
const PK_LINKS = [["F1", "F2"], ["D1", "D2"], ["F1", "D1"], ["F2", "D2"]];

function cardStyle(geo) {
  return { left: `${geo.left}%`, top: `${geo.top}%`, width: `${geo.width}%`, height: `${geo.height}%` };
}
function buildLinks(pairs, geo, slotPlayers, chemReport) {
  return pairs.map(([slotA, slotB]) => {
    const a = slotPlayers[slotA];
    const b = slotPlayers[slotB];
    const bond = a && b ? pairChemistry(a, b, chemReport, slotA, slotB) : null;
    const tier = bond ? linkTier(bond.score) : "empty";
    return {
      key: `${slotA}-${slotB}`,
      slotA,
      slotB,
      x1: geo[slotA].cx,
      y1: geo[slotA].cy,
      x2: geo[slotB].cx,
      y2: geo[slotB].cy,
      mx: (geo[slotA].cx + geo[slotB].cx) / 2,
      my: (geo[slotA].cy + geo[slotB].cy) / 2,
      score: bond?.score ?? null,
      projected: bond?.projected ?? true,
      familiarity: bond?.familiarity ?? null,
      fresh: Boolean(bond?.fresh),
      tier,
    };
  });
}
function pipsForSlot(links, slot) {
  return links.filter((link) => link.slotA === slot || link.slotB === slot).map((link) => link.tier);
}
function scoreTone(score) {
  if (score == null) return "";
  if (score >= LINK_STRONG) return "";
  if (score >= LINK_FORMING) return "warn";
  return "bad";
}

function Glyph({ text }) { return <span className="lb-glyph" aria-hidden="true">{text}</span>; }

function LineBuilderSidebar({ setScreen, abbreviation, activeScreen = SCREENS.EDIT_LINES }) {
  const items = [
    { label: "Roster", glyph: "R", screen: SCREENS.ROSTER },
    { label: "Lines", glyph: "L", screen: SCREENS.EDIT_LINES, active: activeScreen === SCREENS.EDIT_LINES },
    { label: "Power Play", glyph: "PP", screen: SCREENS.POWER_PLAY, active: activeScreen === SCREENS.POWER_PLAY },
    { label: "Penalty Kill", glyph: "PK", screen: SCREENS.PENALTY_KILL, active: activeScreen === SCREENS.PENALTY_KILL },
    { label: "Back", glyph: "‹", screen: SCREENS.ROSTER },
  ];
  return <aside className="lb-sidebar" aria-label="Line builder navigation">
    <div className="lb-team-mark">{abbreviation}</div>
    <nav className="lb-nav">{items.map((item) => <button type="button" key={item.label} className={`lb-nav-btn ${item.active ? "active" : ""}`} onClick={() => item.screen && setScreen(item.screen)} aria-current={item.active ? "page" : undefined} title={item.label}><Glyph text={item.glyph} /><span className="lb-nav-label">{item.label}</span></button>)}</nav>
  </aside>;
}

function LineBuilderHeader({ team, title = "Line Chemistry", subtitle, live, canUndo, canRedo, onUndo, onRedo, onClear, onReset, onAutoBuild, onSave, saving, unsaved, saveError, disabled, showHistory = true, showAutoBuild = true, resetLabel = "Reset", saveLabel }) {
  return <header className="lb-header">
    <div className="lb-title-group">
      <div className="lb-logo">{team.logo ? <img src={team.logo} alt={`${team.abbreviation} logo`} /> : team.abbreviation}</div>
      <div>
        <h1 className="lb-title">{title}</h1>
        <div className="lb-subtitle"><span className={`lb-live-dot ${live ? "" : "projected"}`} />{subtitle}</div>
      </div>
    </div>
    <div className="lb-actions">
      {showHistory ? <>
        <button type="button" className="lb-icon" onClick={onUndo} disabled={!canUndo} aria-label="Undo lineup change" title="Undo">↶</button>
        <button type="button" className="lb-icon" onClick={onRedo} disabled={!canRedo} aria-label="Redo lineup change" title="Redo">↷</button>
      </> : null}
      <button type="button" className="lb-btn subtle danger" onClick={onClear}>Clear</button>
      <button type="button" className="lb-btn subtle" onClick={onReset}>{resetLabel}</button>
      {showAutoBuild ? <button type="button" className="lb-btn" onClick={onAutoBuild} disabled={disabled}>Auto Build</button> : null}
      {saveError ? <span className="lb-save-error">Save failed</span> : null}
      <button type="button" className="lb-btn primary" onClick={onSave} disabled={saving || disabled}>{saving ? "Saving" : saveLabel || (unsaved ? "Save lines" : "Saved")}</button>
    </div>
  </header>;
}

function LineStatusStrip({ metrics, meter }) {
  return <div className="lb-status" aria-label="Lineup status">
    {metrics.map((metric) => metric.value == null ? null : (
      <div className="lb-pill" key={metric.label} title={metric.title || undefined}>
        <span className="lb-pill-label">{metric.label}</span>
        <span className={`lb-pill-value ${metric.tone || ""}`}>{metric.value}</span>
      </div>
    ))}
    {meter?.length ? <div className="lb-chemmeter" title="One segment per player-to-player link in the lineup">{meter.map((tier, index) => <span key={index} className={`lb-chemseg ${tier}`} />)}</div> : null}
  </div>;
}

function PlayerPool({ players, assignedSet, lockedSet, selectedPlayerId, search, setSearch, focusSlot, onPlayerSelect, onDragStart, open, onClose }) {
  return <section className={`lb-region lb-pool-region ${open ? "open" : ""}`}>
    <div className="lb-region-head">
      <h2 className="lb-region-title">Player pool</h2>
      <span className="lb-region-note">{players.length}</span>
      <button type="button" className="lb-icon lb-drawer-close" onClick={onClose} aria-label="Close player pool" title="Close">×</button>
    </div>
    <div className="lb-pool">
      <div className="lb-search-wrap">
        <input className="lb-search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search players" aria-label="Search players" />
        {search ? <button type="button" className="lb-clear" onClick={() => setSearch("")} aria-label="Clear search" title="Clear search">×</button> : null}
      </div>
      <div className="lb-player-list">{players.length ? players.map((player) => {
        const assigned = assignedSet.has(player.id);
        const locked = lockedSet.has(player.id);
        const disabled = !player.availability?.placeable || locked;
        const fit = focusSlot ? chemistryFitScore(player, focusSlot) : null;
        const status = player.availability?.key === "active" ? null : player.availability?.label;
        const meta = [player.position, fit != null ? `${fit}% fit` : null, status, assigned ? "Dressed" : null].filter(Boolean).join(" · ");
        return <div key={player.id} className={`lb-player ${assigned ? "assigned" : ""} ${selectedPlayerId === player.id ? "selected" : ""} ${disabled ? "disabled" : ""} ${locked ? "locked" : ""}`} draggable={!disabled} onDragStart={(event) => onDragStart(event, player.id)} onClick={() => onPlayerSelect(player.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onPlayerSelect(player.id); } }} role="button" tabIndex={0} aria-label={`${player.name}, ${player.position}, ${player.overall} overall`} title={disabled ? player.availability?.reason || "Player cannot be moved." : `${player.name} · ${player.position} · ${player.overall} overall`}>
          <PlayerHeadshot player={player} size="md" />
          <div className="lb-player-copy">
            <div className="lb-player-name">{player.name}</div>
            <div className="lb-player-meta">{meta}</div>
          </div>
          <div className="lb-player-side">
            <div className="lb-player-ovr">{player.overall}</div>
            <span className={`lb-dot ${player.availability?.key === "active" ? "" : player.availability?.placeable ? "warn" : "bad"}`} />
          </div>
        </div>;
      }) : <div className="lb-empty">No matching players</div>}</div>
    </div>
  </section>;
}

function LinkLayer({ links }) {
  return <svg className="fm-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
    {links.map((link) => (
      <line key={link.key} x1={link.x1} y1={link.y1} x2={link.x2} y2={link.y2}
        className={`fm-link ${link.tier}${link.dashed ? " dashed" : ""}`} vectorEffect="non-scaling-stroke" />
    ))}
  </svg>;
}

function LinkBadges({ links }) {
  return links.map((link) => (
    <span key={`badge-${link.key}`} className={`fm-badge ${link.tier}`} style={{ left: `${link.mx}%`, top: `${link.my}%` }}>
      {link.score == null ? "—" : link.score}
    </span>
  ));
}

function FormationCard({ descriptor, geo, player, tier, pips, locked, selected, targetState, fresh, onSelect, onDrop, onDragStart, onRemove, onToggleLock, onKeyDown, registerRef, showPips = true }) {
  const flag = player?.availability?.key === "injured" ? "MED" : player?.scratched ? "SCR" : !player?.availability?.placeable && player ? "OUT" : null;
  const classes = [
    "fm-card",
    player ? tier || "" : "empty",
    fresh ? "fresh" : "",
    selected ? "selected" : "",
    locked ? "locked" : "",
    player && !player.availability?.placeable ? "unavailable" : "",
    targetState,
  ].filter(Boolean).join(" ");
  return <div
    ref={(node) => registerRef && registerRef(descriptor.key, node)}
    className={classes}
    style={cardStyle(geo)}
    role="button"
    tabIndex={0}
    aria-label={player ? `${descriptor.slot}, ${player.name}, ${player.overall} overall` : `${descriptor.slot}, empty`}
    title={player ? shortText(getOverallTooltip(player)) : `Add ${descriptor.slot}`}
    onClick={(event) => { event.stopPropagation(); onSelect(descriptor); }}
    onKeyDown={(event) => onKeyDown && onKeyDown(event, descriptor)}
    onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
    onDrop={(event) => onDrop(event, descriptor)}
    draggable={Boolean(player) && !locked && player.availability?.placeable}
    onDragStart={(event) => player && onDragStart(event, player.id)}
  >
    <span className="fm-card-slot">{descriptor.slot}</span>
    {flag ? <span className="fm-card-flag">{flag}</span> : null}
    {player ? <>
      <span className="fm-card-ovr">{fresh ? "NEW" : player.overall}</span>
      <span className="fm-card-name" title={player.name}>{lastName(player.name)}</span>
      <span className="fm-card-meta">{fresh ? "No shared history" : slotMeta(player)}</span>
      {player.overall_drop > 0 ? <span className="fm-card-drop">-{player.overall_drop}</span> : null}
      {showPips && pips?.length ? <span className="fm-pips">{pips.map((pip, index) => <span key={index} className={`fm-pip ${pip}`} />)}</span> : null}
      {selected ? <span className="fm-card-actions">
        <button type="button" className="fm-mini" onClick={(event) => { event.stopPropagation(); onToggleLock(descriptor); }} aria-label={locked ? "Unlock slot" : "Lock slot"} title={locked ? "Unlock slot" : "Lock slot"}>{locked ? "U" : "L"}</button>
        <button type="button" className="fm-mini danger" onClick={(event) => { event.stopPropagation(); onRemove(descriptor); }} disabled={locked} aria-label="Remove player" title="Remove player">×</button>
      </span> : null}
    </> : <span className="fm-card-empty">Empty</span>}
  </div>;
}

function UnitTag({ kicker, number, score, warnings, onWarnings, onMenu }) {
  return <div className="fm-unit-tag">
    <span className="fm-unit-kicker">{kicker}</span>
    <span className="fm-unit-num">{number}</span>
    {score != null ? <span className={`fm-unit-score ${scoreTone(score)}`}>{score}%</span> : null}
    {warnings?.length ? <button type="button" className="fm-unit-warn" onClick={(event) => { event.stopPropagation(); onWarnings(); }} title={`${warnings.length} warnings`}>{warnings.length}</button> : null}
    <button type="button" className="fm-unit-menu-btn" onClick={(event) => { event.stopPropagation(); onMenu(); }} aria-label="Unit actions" title="Unit actions">···</button>
  </div>;
}

function UnitMenu({ line, group, locks, clipboard, actions }) {
  const keys = Object.keys(line.slots).map((slot) => slotKey(group, line.id, slot));
  const hasLocks = keys.some((key) => locks[key]);
  const canPaste = clipboard && JSON.stringify(Object.keys(clipboard.slots || {})) === JSON.stringify(Object.keys(line.slots || {}));
  return <div className="fm-unit-menu" role="menu" onClick={(event) => event.stopPropagation()}>
    <button type="button" className="lb-menu-action" onClick={() => actions.clear(group, line.id)}>Clear unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.autoBuild(group, line.id)}>Auto build unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.lock(group, line.id)}>Lock unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.unlock(group, line.id)} disabled={!hasLocks}>Unlock unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.copy(group, line.id)}>Copy unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.paste(group, line.id)} disabled={!canPaste}>Paste unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.reset(group, line.id)}>Reset unit</button>
    <button type="button" className="lb-menu-action" onClick={actions.close}>Close menu</button>
  </div>;
}

function DepthConnector({ tier, score }) {
  return <div className="fm-depth" aria-hidden="true">
    <div className={`fm-depth-line ${tier}`} />
    <span className={`fm-depth-badge ${tier}`}>{score == null ? "—" : score}</span>
  </div>;
}

function SectionHead({ title, note }) {
  return <div className="fm-section-head">
    <span className="fm-section-title">{title}</span>
    <span className="fm-section-rule" />
    {note ? <span className="fm-section-note">{note}</span> : null}
  </div>;
}

function FormationLegend({ depthLabel = "Line-to-line depth" }) {
  return <div className="fm-legend">
    <span className="fm-legend-item"><span className="fm-legend-swatch" />Strong link</span>
    <span className="fm-legend-item"><span className="fm-legend-swatch forming" />Forming link</span>
    <span className="fm-legend-item"><span className="fm-legend-swatch weak" />Weak / new link</span>
    <span className="fm-legend-item"><span className="fm-legend-swatch depth" />{depthLabel}</span>
  </div>;
}

function slotPlayersFor(line, playerMap) {
  const result = {};
  for (const [slot, id] of Object.entries(line?.slots || {})) result[slot] = playerMap[String(id || "")] || null;
  return result;
}

function FormationUnit({
  group, line, geo, linkPairs, playerMap, chemReport, links, warnings, chemistry,
  kicker, sizeClass, selectedUnit, selectedSlotKey, selectedPlayer, locks, menuOpen, clipboard,
  onSelectUnit, onSlotSelect, onDrop, onDragStart, onRemove, onToggleLock, onSlotKeyDown,
  registerSlotRef, onWarnings, onMenuToggle, actions, slots, showPips = true, isolated = false, isolatedNote,
}) {
  const selected = selectedUnit?.group === group && selectedUnit?.lineId === line.id;
  const unitNumber = String(line.name || "").replace(/\D+/g, "") || String(line.id).toUpperCase();
  return <div className={`fm-unit ${sizeClass} ${selected ? "selected" : ""} ${isolated ? "isolated" : ""}`} onClick={() => onSelectUnit(group, line.id)}>
    {isolated && isolatedNote ? <span className="fm-isolated-note">{isolatedNote}</span> : null}
    {links.length ? <LinkLayer links={links} /> : null}
    {links.length ? <LinkBadges links={links} /> : null}
    <UnitTag
      kicker={kicker}
      number={unitNumber}
      score={chemistry?.score ?? null}
      warnings={warnings}
      onWarnings={() => onWarnings(group, line.id)}
      onMenu={() => onMenuToggle(group, line.id)}
    />
    {slots.map((slot) => {
      const descriptor = { key: slotKey(group, line.id, slot), group, lineId: line.id, lineName: line.name, slot, playerId: String(line.slots?.[slot] || "") };
      const player = playerMap[descriptor.playerId] || null;
      const slotLinks = links.filter((link) => link.slotA === slot || link.slotB === slot);
      const tiers = slotLinks.map((link) => link.tier);
      const worst = tiers.includes("weak") ? "weak" : tiers.includes("forming") ? "forming" : tiers.includes("strong") ? "strong" : "";
      const fresh = Boolean(player) && slotLinks.length > 0 && slotLinks.every((link) => link.fresh);
      let targetState = "";
      if (selectedPlayer) targetState = !selectedPlayer.availability?.placeable || !posFit(selectedPlayer, slot) ? "invalid" : player ? "swap" : "valid";
      return <FormationCard
        key={descriptor.key}
        descriptor={descriptor}
        geo={geo[slot]}
        player={player}
        tier={worst}
        pips={tiers}
        fresh={fresh}
        locked={Boolean(locks[descriptor.key])}
        selected={selectedSlotKey === descriptor.key}
        targetState={targetState}
        onSelect={onSlotSelect}
        onDrop={onDrop}
        onDragStart={onDragStart}
        onRemove={onRemove}
        onToggleLock={onToggleLock}
        onKeyDown={onSlotKeyDown}
        registerRef={registerSlotRef}
        showPips={showPips}
      />;
    })}
    {menuOpen ? <UnitMenu line={line} group={group} locks={locks} clipboard={clipboard} actions={actions} /> : null}
  </div>;
}

function FormationBoard({
  mode, setMode, lineState, playerMap, chemReport, chemistryByUnit, warningsByUnit, linksByUnit, depthLinks,
  selectedUnit, selectedSlotKey, selectedPlayer, locks, showThird, menuKey, clipboard,
  onSelectUnit, onSlotSelect, onDrop, onDragStart, onRemove, onToggleLock, onSlotKeyDown,
  registerSlotRef, onWarnings, onMenuToggle, actions, onTogglePool, onToggleInspector, rosterEmpty,
}) {
  const options = [[LINEUP_MODES.ALL, "All"], [LINEUP_MODES.FORWARDS, "Forwards"], [LINEUP_MODES.DEFENSE, "Defence"], [LINEUP_MODES.GOALIES, "Goalies"]];
  const showForwards = mode === LINEUP_MODES.ALL || mode === LINEUP_MODES.FORWARDS;
  const showDefense = mode === LINEUP_MODES.ALL || mode === LINEUP_MODES.DEFENSE;
  const showGoalies = mode === LINEUP_MODES.ALL || mode === LINEUP_MODES.GOALIES;
  const goalieSlots = ["Starter", "Backup", ...(showThird ? ["Third"] : [])];
  const shared = {
    playerMap, chemReport, selectedUnit, selectedSlotKey, selectedPlayer, locks, clipboard,
    onSelectUnit, onSlotSelect, onDrop, onDragStart, onRemove, onToggleLock, onSlotKeyDown,
    registerSlotRef, onWarnings, onMenuToggle, actions,
  };
  return <section className="lb-region lb-board-region">
    <div className="lb-modebar">
      {options.map(([id, label], index) => (
        <button type="button" key={id} className={`lb-mode ${mode === id ? "active" : ""}`} onClick={() => setMode(id)} title={`${label} · ${index + 1}`}>{label}</button>
      ))}
      <div className="lb-mobile-tools">
        <button type="button" className="lb-icon" onClick={onTogglePool} aria-label="Open player pool" title="Players">P</button>
        <button type="button" className="lb-icon" onClick={onToggleInspector} aria-label="Open unit details" title="Details">D</button>
      </div>
    </div>
    {rosterEmpty ? <div className="lb-empty">Roster unavailable</div> : <div className="fm-scroll">
      {showForwards ? <div className="fm-section">
        <SectionHead title="Forwards" note="Every linemate pairing is linked both ways" />
        {lineState.forwards.map((line, index) => <React.Fragment key={line.id}>
          <FormationUnit
            {...shared}
            group="forwards"
            line={line}
            geo={FWD_GEO}
            linkPairs={FWD_LINKS}
            links={linksByUnit[`forwards:${line.id}`] || []}
            warnings={warningsByUnit[`forwards:${line.id}`] || []}
            chemistry={chemistryByUnit[`forwards:${line.id}`]}
            kicker="Line"
            sizeClass="forwards"
            slots={["LW", "C", "RW"]}
            menuOpen={menuKey === `forwards:${line.id}`}
          />
          {index < lineState.forwards.length - 1 ? <DepthConnector tier={depthLinks.forwards[index]?.tier || "empty"} score={depthLinks.forwards[index]?.score ?? null} /> : null}
        </React.Fragment>)}
      </div> : null}

      {showDefense ? <div className="fm-section">
        <SectionHead title="Defence" note="One link per pair" />
        {lineState.defense.map((line, index) => <React.Fragment key={line.id}>
          <FormationUnit
            {...shared}
            group="defense"
            line={line}
            geo={DEF_GEO}
            linkPairs={DEF_LINKS}
            links={linksByUnit[`defense:${line.id}`] || []}
            warnings={warningsByUnit[`defense:${line.id}`] || []}
            chemistry={chemistryByUnit[`defense:${line.id}`]}
            kicker="Pair"
            sizeClass="defense"
            slots={["LD", "RD"]}
            menuOpen={menuKey === `defense:${line.id}`}
          />
          {index < lineState.defense.length - 1 ? <DepthConnector tier={depthLinks.defense[index]?.tier || "empty"} score={depthLinks.defense[index]?.score ?? null} /> : null}
        </React.Fragment>)}
      </div> : null}

      {showGoalies ? <div className="fm-section">
        <SectionHead title="Goalies" note="Outside the chemistry system" />
        {lineState.goalies.map((line) => <FormationUnit
          key={line.id}
          {...shared}
          group="goalies"
          line={line}
          geo={GOALIE_GEO}
          linkPairs={[]}
          links={[]}
          warnings={warningsByUnit[`goalies:${line.id}`] || []}
          chemistry={null}
          kicker="Net"
          sizeClass="goalies"
          slots={goalieSlots}
          menuOpen={menuKey === `goalies:${line.id}`}
          showPips={false}
          isolated
          isolatedNote="No chemistry links — goalies are scored on their own"
        />)}
      </div> : null}

      <FormationLegend />
    </div>}
  </section>;
}

function BarRow({ label, value }) {
  if (value == null) return null;
  const pct = clamp(value);
  const tone = pct >= LINK_STRONG ? "" : pct >= LINK_FORMING ? "warn" : "bad";
  return <div className="lb-bar-row">
    <div className="lb-bar-head"><span>{label}</span><span>{pct}%</span></div>
    <div className="lb-bar"><div className={`lb-bar-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
  </div>;
}

function LineInspector({
  open, onClose, tab, setTab, selectedLine, selectedGroup, selectedPlayer, selectedSlot, playerMap,
  chemistry, warnings, unitPlayers, unitLinks, comparisonPlayers, comparing, setComparing,
  onReplace, onRemoveSelected, onToggleSelectedLock, selectedLocked, chemistryLoading, title = "Inspector",
}) {
  const average = unitPlayers.length ? Math.round(unitPlayers.reduce((sum, player) => sum + player.overall, 0) / unitPlayers.length) : null;
  const roles = roleMixLabel(unitPlayers);
  const bondAverage = averageLinkScore(unitLinks || []);
  const freshLinks = (unitLinks || []).filter((link) => link.fresh && link.score != null);
  const weakLinks = (unitLinks || []).filter((link) => link.tier === "weak" && link.score != null);
  const strongLinks = (unitLinks || []).filter((link) => link.tier === "strong");
  const isGoalieUnit = selectedGroup === "goalies";
  const linkName = (slot) => {
    const id = String(selectedLine?.slots?.[slot] || "");
    return lastName(playerMap[id]?.name || slot);
  };
  return <section className={`lb-region lb-inspector-region ${open ? "open" : ""}`}>
    <div className="lb-region-head">
      <h2 className="lb-region-title">{title}</h2>
      <span className="lb-region-note">{selectedLine?.name || "Lineup"}</span>
      <button type="button" className="lb-icon lb-drawer-close" onClick={onClose} aria-label="Close inspector" title="Close">×</button>
    </div>
    <div className="lb-inspector">
      <div className="lb-tabs">{TABS.map((id) => (
        <button type="button" key={id} className={`lb-tab ${tab === id ? "active" : ""}`} onClick={() => { setTab(id); setComparing(false); }}>{id}</button>
      ))}</div>
      <div className="lb-inspector-body">
        {comparing ? <div>
          {comparisonPlayers.length ? comparisonPlayers.map((player) => (
            <div className="lb-compare" key={player.id}>
              <div>
                <div className="lb-compare-name">{player.name}</div>
                <div className="lb-compare-meta">{player.overall} OVR · {player.fit}% fit · {player.availability.label}</div>
              </div>
              <button type="button" className="lb-btn" onClick={() => onReplace(player.id)} disabled={!player.availability.placeable}>Select</button>
            </div>
          )) : <div className="lb-empty">No valid candidates</div>}
        </div> : tab === "unit" ? <>
          <div>
            <div className="lb-inspector-title">{selectedLine?.name || "Unit"}</div>
            <div className="lb-inspector-sub">{isGoalieUnit ? "Not part of the chemistry system" : chemistryLoading ? "Loading report" : chemistry?.projected ? "Projected — save to score live" : "Live backend chemistry"}</div>
          </div>
          {isGoalieUnit ? null : <div className="lb-gauge">
            <span className={`lb-gauge-score ${scoreTone(chemistry?.score)}`}>{chemistry?.score ?? 0}</span>
            <div>
              <div className="lb-gauge-label">{shortText(chemistry?.label, 4) || "Unit"}</div>
              <div className="lb-gauge-source">Unit chemistry</div>
            </div>
          </div>}
          <div className="lb-block">
            <div className="lb-kv"><span className="lb-kv-label">Avg OVR</span><span className="lb-kv-value">{average ?? "—"}</span></div>
            <div className="lb-kv"><span className="lb-kv-label">Role mix</span><span className="lb-kv-value">{roles}</span></div>
            {isGoalieUnit ? null : <div className="lb-kv"><span className="lb-kv-label">Avg bond</span><span className="lb-kv-value">{bondAverage == null ? "—" : `${bondAverage}%`}</span></div>}
            {isGoalieUnit ? null : <div className="lb-kv"><span className="lb-kv-label">New pairings</span><span className={`lb-kv-value ${freshLinks.length ? "warn" : "ok"}`}>{freshLinks.length}</span></div>}
          </div>
          {isGoalieUnit ? null : <div className="lb-block">
            <div className="lb-block-title">Links in this unit</div>
            {(unitLinks || []).length ? unitLinks.map((link) => (
              <div className="lb-linkrow" key={link.key}>
                <span className="lb-linkrow-names">{linkName(link.slotA)} — {linkName(link.slotB)}</span>
                <span className={`lb-linkrow-score ${link.tier}`}>{link.score == null ? "—" : `${link.score}%`}</span>
              </div>
            )) : <div className="lb-note"><span className="lb-note-text">Fill the unit to score its links.</span></div>}
          </div>}
          <div className="lb-block">
            <div className="lb-block-title">Read on this unit</div>
            {strongLinks.length ? <div className="lb-note"><span className="lb-note-mark good">+</span><span className="lb-note-text">{strongLinks.length} strong complementary link{strongLinks.length > 1 ? "s" : ""} — roles on this unit fit together.</span></div> : null}
            {(chemistry?.factors || []).map((factor, index) => (
              <div className="lb-note" key={`factor-${index}`}><span className="lb-note-mark good">+</span><span className="lb-note-text">{factor}</span></div>
            ))}
            {freshLinks.length ? <div className="lb-note"><span className="lb-note-mark warn">!</span><span className="lb-note-text">{freshLinks.length} pairing{freshLinks.length > 1 ? "s have" : " has"} no shared history yet. Familiarity builds while they stay together.</span></div> : null}
            {weakLinks.length && !freshLinks.length ? <div className="lb-note"><span className="lb-note-mark warn">!</span><span className="lb-note-text">{weakLinks.length} weak link{weakLinks.length > 1 ? "s" : ""} dragging this unit down.</span></div> : null}
            {(chemistry?.concerns || []).map((concern, index) => (
              <div className="lb-note" key={`concern-${index}`}><span className="lb-note-mark warn">!</span><span className="lb-note-text">{concern}</span></div>
            ))}
            {isGoalieUnit ? <div className="lb-note"><span className="lb-note-mark">·</span><span className="lb-note-text">Goalies carry no linemate bonds. Only the starter assignment reaches the sim.</span></div> : null}
          </div>
        </> : tab === "fit" ? <div>
          <BarRow label="Position fit" value={chemistry?.positionFit} />
          <div style={{ height: 9 }} />
          <BarRow label="Role balance" value={chemistry?.roleBalance} />
          <div style={{ height: 9 }} />
          <BarRow label="Linemate fit" value={chemistry?.linemateFit} />
          <div style={{ height: 9 }} />
          <BarRow label="Familiarity" value={chemistry?.familiarity ?? bondAverage} />
          <div style={{ height: 9 }} />
          <BarRow label="Coach fit" value={chemistry?.coachFit} />
          <div style={{ height: 9 }} />
          <BarRow label="Handedness fit" value={chemistry?.handednessFit} />
        </div> : tab === "player" ? <div>
          {selectedPlayer ? <>
            <div className="lb-inspector-player">
              <PlayerHeadshot player={selectedPlayer} size="lg" />
              <div>
                <div className="lb-inspector-player-name">{selectedPlayer.name}</div>
                <div className="lb-inspector-player-meta">{selectedPlayer.position} · {selectedPlayer.overall} OVR · {selectedPlayer.availability?.label}</div>
                <div className="lb-inspector-player-meta">{shortRole(selectedPlayer.role) || "—"} · {selectedPlayer.handedness} · {getCountry(selectedPlayer)}</div>
              </div>
            </div>
            <div className="lb-block">
              <div className="lb-kv"><span className="lb-kv-label">Morale</span><span className="lb-kv-value">{selectedPlayer.morale ?? "—"}</span></div>
              <div className="lb-kv"><span className="lb-kv-label">Confidence</span><span className="lb-kv-value">{selectedPlayer.confidence ?? "—"}</span></div>
              <div className="lb-kv"><span className="lb-kv-label">Fatigue</span><span className="lb-kv-value">{selectedPlayer.fatigue ?? "—"}</span></div>
              <div className="lb-kv"><span className="lb-kv-label">Role satisfaction</span><span className="lb-kv-value">{selectedPlayer.role_satisfaction ?? "—"}</span></div>
              <div className="lb-kv"><span className="lb-kv-label">Coach trust</span><span className="lb-kv-value">{selectedPlayer.coach_trust ?? "—"}</span></div>
            </div>
          </> : <div className="lb-empty">Select a player</div>}
        </div> : <div>
          {warnings.length ? warnings.map((warning) => (
            <div className="lb-note" key={warning.key}><span className="lb-note-mark warn">!</span><span className="lb-note-text">{warning.text}</span></div>
          )) : <div className="lb-empty">No unit warnings</div>}
        </div>}
        {selectedSlot ? <div className="lb-inspector-actions">
          <button type="button" className="lb-btn" onClick={() => setComparing((current) => !current)}>{comparing ? "Close" : "Compare"}</button>
          <button type="button" className="lb-btn" onClick={() => setComparing(true)}>Replace</button>
          <button type="button" className="lb-btn" onClick={onToggleSelectedLock}>{selectedLocked ? "Unlock" : "Lock"}</button>
          <button type="button" className="lb-btn danger" onClick={onRemoveSelected} disabled={selectedLocked || !selectedPlayer}>Remove</button>
        </div> : null}
      </div>
    </div>
  </section>;
}

function AutoBuildPopover({ state, setState, changes, onApply }) {
  if (!state.open) return null;
  return <div className="lb-popover" role="dialog" aria-label="Auto build lineup">
    <h2 className="lb-popover-title">{state.scope ? "Auto build unit" : "Auto build lineup"}</h2>
    <div className="lb-auto-options">{AUTO_MODES.map(([id, label]) => (
      <button type="button" key={id} className={`lb-auto-option ${state.mode === id ? "active" : ""}`} onClick={() => setState((current) => ({ ...current, mode: id }))}>{label}</button>
    ))}</div>
    <div className="lb-preview">{changes.length ? changes.map((change) => (
      <div className="lb-preview-row" key={change.key}>
        <span className="lb-preview-label">{change.label}</span>
        <span className="lb-preview-change">{change.before} → {change.after}</span>
      </div>
    )) : <div className="lb-empty">No changes found</div>}</div>
    <div className="lb-popover-actions">
      <button type="button" className="lb-btn" onClick={() => setState((current) => ({ ...current, open: false, scope: null }))}>Cancel</button>
      <button type="button" className="lb-btn primary" onClick={onApply} disabled={!changes.length}>Apply</button>
    </div>
  </div>;
}

function LineBuilderToast({ toast, onDismiss, onDetails }) {
  if (!toast) return null;
  return <div className={`lb-toast ${toast.type}`} role="status">
    <span className="lb-toast-message">{shortText(toast.message)}</span>
    <span className="lb-toast-actions">
      {toast.onClick ? <button type="button" className="fm-mini" onClick={toast.onClick}>{toast.actionLabel || "View"}</button> : null}
      {toast.details ? <button type="button" className="fm-mini" onClick={onDetails}>i</button> : null}
      <button type="button" className="fm-mini" onClick={onDismiss} aria-label="Dismiss message" title="Dismiss">×</button>
    </span>
  </div>;
}

function EvenStrengthLines(props) {
  const { franchiseState, setScreen, setFranchiseState } = useGameUI();
  const sessionId = getFranchiseSessionId();
  const team = useMemo(() => teamIdentity(franchiseState, props), [franchiseState, props]);
  const [chemReport, setChemReport] = useState(null);
  const [chemLoading, setChemLoading] = useState(true);
  const [chemNonce, setChemNonce] = useState(0);
  const [search, setSearch] = useState("");
  const [mode, setMode] = useState(LINEUP_MODES.ALL);
  const [selectedUnit, setSelectedUnit] = useState({ group: "forwards", lineId: "f1" });
  const [selectedSlotKey, setSelectedSlotKey] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [tab, setTab] = useState("unit");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [poolOpen, setPoolOpen] = useState(false);
  const [menuKey, setMenuKey] = useState("");
  const [clipboard, setClipboard] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [locks, setLocks] = useState({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [toast, setToast] = useState(null);
  const [announcement, setAnnouncement] = useState("");
  const [autoBuild, setAutoBuild] = useState({ open: false, mode: "position", scope: null });
  const [lineState, setLineState] = useState(() => {
    const cached = readSessionLineupCache(LINEUP_KIND, sessionId);
    return cached?.forwards && cached?.defense && cached?.goalies ? cloneLines(cached) : emptyLines(false);
  });
  const [savedSnapshot, setSavedSnapshot] = useState(() => snapshot(lineState, {}));
  const hydratedSession = useRef("");
  const staleNotice = useRef("");
  const slotRefs = useRef(new Map());

  // Refetch after every save so the report stops describing the previous lineup.
  useEffect(() => {
    let active = true;
    setChemLoading(true);
    getFranchiseChemistry()
      .then((data) => { if (active) setChemReport(data || null); })
      .catch(() => { if (active) setChemReport(null); })
      .finally(() => { if (active) setChemLoading(false); });
    return () => { active = false; };
  }, [franchiseState?.session_id, chemNonce]);

  const players = useMemo(() => {
    const lean = Array.isArray(franchiseState?.roster) ? franchiseState.roster : [];
    if (lean.length) return lean.map(normalizePlayer);
    const organizations = franchiseState?.roster_browser?.organizations || [];
    const teamId = String(franchiseState?.user_team_id || "");
    const organization = organizations.find((candidate) => String(candidate?.team_id || "") === teamId) || organizations[0];
    const live = Array.isArray(organization?.nhl) ? organization.nhl : [];
    return live.length ? live.map(normalizePlayer) : getRosterPlayers(props);
  }, [franchiseState, props]);
  const playerMap = useMemo(() => players.reduce((map, player) => { map[player.id] = player; return map; }, {}), [players]);
  const showThird = useMemo(() => players.filter((player) => isGoalie(player.position)).length >= 3 || Boolean(lineState.goalies?.[0]?.slots?.Third), [players, lineState.goalies]);

  useEffect(() => {
    if (!players.length || hydratedSession.current === sessionId) return;
    const backendRaw = franchiseState?.lines?.even_strength?.lines;
    const backend = backendRaw ? sanitizeLineup(backendRaw, players, showThird) : null;
    const cacheRaw = readSessionLineupCache(LINEUP_KIND, sessionId);
    const cache = cacheRaw ? sanitizeLineup(cacheRaw, players, showThird) : null;
    const hasBackend = Boolean(backend?.retained);
    const hasCache = Boolean(cache?.retained);
    const chosen = hasBackend ? backend : hasCache ? cache : { lineState: buildBestInitialLines(players, "position", showThird), removed: 0 };
    const savedBase = hasBackend || hasCache ? chosen.lineState : emptyLines(showThird);
    hydratedSession.current = sessionId;
    setLineState(chosen.lineState);
    setLocks({});
    setHistory([]);
    setFuture([]);
    setSelectedSlotKey("");
    setSelectedPlayerId("");
    setSavedSnapshot(snapshot(savedBase, {}));
    if (chosen.removed > 0 && staleNotice.current !== sessionId) {
      staleNotice.current = sessionId;
      setToast({ type: "warning", message: `${chosen.removed} stale assignments removed.` });
    }
  }, [players, franchiseState?.lines, sessionId, showThird]);
  useEffect(() => { if (hydratedSession.current && players.length) writeSessionLineupCache(LINEUP_KIND, lineState, sessionId); }, [lineState, players.length, sessionId]);
  useEffect(() => { if (toast?.type !== "success") return undefined; const timer = window.setTimeout(() => setToast(null), 2600); return () => window.clearTimeout(timer); }, [toast]);

  const unsaved = useMemo(() => snapshotKey(snapshot(lineState, locks)) !== snapshotKey(savedSnapshot), [lineState, locks, savedSnapshot]);
  const assignedSet = useMemo(() => new Set(descriptors(lineState, showThird).map((row) => row.playerId).filter(Boolean)), [lineState, showThird]);
  const lockedSet = useMemo(() => {
    const set = new Set();
    for (const row of descriptors(lineState, showThird)) if (locks[row.key] && row.playerId) set.add(row.playerId);
    return set;
  }, [lineState, locks, showThird]);
  const duplicateIds = useMemo(() => new Set(duplicates(lineState).map((entry) => entry.playerId)), [lineState]);

  const linksByUnit = useMemo(() => {
    const result = {};
    for (const line of lineState.forwards || []) result[`forwards:${line.id}`] = buildLinks(FWD_LINKS, FWD_GEO, slotPlayersFor(line, playerMap), chemReport);
    for (const line of lineState.defense || []) result[`defense:${line.id}`] = buildLinks(DEF_LINKS, DEF_GEO, slotPlayersFor(line, playerMap), chemReport);
    for (const line of lineState.goalies || []) result[`goalies:${line.id}`] = [];
    return result;
  }, [lineState, playerMap, chemReport]);

  // Depth relationship between consecutive units: the anchor players carry it.
  const depthLinks = useMemo(() => {
    const chain = (lines, anchorSlot) => {
      const out = [];
      for (let index = 0; index < lines.length - 1; index += 1) {
        const a = playerMap[String(lines[index]?.slots?.[anchorSlot] || "")];
        const b = playerMap[String(lines[index + 1]?.slots?.[anchorSlot] || "")];
        const bond = a && b ? pairChemistry(a, b, chemReport, anchorSlot, anchorSlot) : null;
        out.push({ score: bond?.score ?? null, tier: bond ? linkTier(bond.score) : "empty" });
      }
      return out;
    };
    return { forwards: chain(lineState.forwards || [], "C"), defense: chain(lineState.defense || [], "LD") };
  }, [lineState, playerMap, chemReport]);

  const allLinks = useMemo(() => Object.values(linksByUnit).flat().filter((link) => link.score != null), [linksByUnit]);
  const chemMeter = useMemo(() => allLinks.map((link) => link.tier), [allLinks]);

  const chemistryByUnit = useMemo(() => {
    const result = {};
    for (const group of GROUPS) for (const line of lineState[group] || []) {
      const unitPlayers = Object.values(line.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean);
      const computed = calculateUnitChemistry(unitPlayers, group === "defense" ? "defense" : group === "goalies" ? "goalie" : "forward", chemReport);
      const bondAvg = averageLinkScore(linksByUnit[`${group}:${line.id}`] || []);
      if (bondAvg != null && (computed?.projected || Math.abs((computed?.score || 0) - bondAvg) >= 8)) {
        result[`${group}:${line.id}`] = { ...computed, score: bondAvg, label: unitChemistryLabel(bondAvg), projected: true };
      } else {
        result[`${group}:${line.id}`] = computed;
      }
    }
    return result;
  }, [lineState, playerMap, chemReport, linksByUnit]);
  const warningsByUnit = useMemo(() => {
    const result = {};
    for (const group of GROUPS) for (const line of lineState[group] || []) result[`${group}:${line.id}`] = unitWarnings(line, group, playerMap, chemReport, duplicateIds);
    return result;
  }, [lineState, playerMap, chemReport, duplicateIds]);
  const validation = useMemo(() => validateLineup(lineState, playerMap), [lineState, playerMap]);
  const selectedLine = useMemo(() => lineState[selectedUnit.group]?.find((line) => line.id === selectedUnit.lineId) || lineState[selectedUnit.group]?.[0] || null, [lineState, selectedUnit]);
  const selectedSlot = useMemo(() => findSlot(lineState, selectedSlotKey), [lineState, selectedSlotKey]);
  const selectedPlayer = useMemo(() => playerMap[selectedPlayerId || selectedSlot?.playerId] || null, [playerMap, selectedPlayerId, selectedSlot]);
  const unitPlayers = useMemo(() => selectedLine ? Object.values(selectedLine.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean) : [], [selectedLine, playerMap]);
  const selectedChemistry = useMemo(() => chemistryByUnit[`${selectedUnit.group}:${selectedLine?.id}`] || calculateUnitChemistry([], "forward", chemReport), [chemistryByUnit, selectedUnit, selectedLine, chemReport]);
  const selectedLinks = useMemo(() => linksByUnit[`${selectedUnit.group}:${selectedLine?.id}`] || [], [linksByUnit, selectedUnit, selectedLine]);
  const selectedWarnings = useMemo(() => warningsByUnit[`${selectedUnit.group}:${selectedLine?.id}`] || validation.errors, [warningsByUnit, selectedUnit, selectedLine, validation.errors]);
  const focusSlot = useMemo(() => selectedSlot?.slot || Object.entries(selectedLine?.slots || {}).find(([, id]) => !id)?.[0] || Object.keys(selectedLine?.slots || {})[0] || null, [selectedSlot, selectedLine]);

  const filteredPool = useMemo(() => {
    const query = search.trim().toLowerCase();
    return sortPoolPlayers(players.filter((player) => !query || `${player.name} ${player.position} ${player.role}`.toLowerCase().includes(query)), "overall", focusSlot);
  }, [players, search, focusSlot]);
  const comparisonPlayers = useMemo(() => selectedSlot ? sortPoolPlayers(players.filter((player) => posFit(player, selectedSlot.slot)).map((player) => ({ ...player, fit: chemistryFitScore(player, selectedSlot.slot) })), "overall", selectedSlot.slot).slice(0, 5) : [], [players, selectedSlot]);
  const assignedPlayers = useMemo(() => [...assignedSet].map((id) => playerMap[id]).filter(Boolean), [assignedSet, playerMap]);
  const teamChemistry = useMemo(() => {
    const scores = Object.values(chemistryByUnit || {}).map((entry) => Number(entry?.score)).filter((score) => Number.isFinite(score) && score > 0);
    if (!scores.length) return { score: 0, projected: true };
    return {
      score: Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length),
      projected: Object.values(chemistryByUnit || {}).some((entry) => entry?.projected),
    };
  }, [chemistryByUnit]);
  const statusMetrics = useMemo(() => {
    const skaters = new Set();
    for (const group of ["forwards", "defense"]) for (const line of lineState[group] || []) for (const id of Object.values(line.slots || {})) if (id) skaters.add(String(id));
    const activeGoalies = ["Starter", "Backup"].filter((slot) => lineState.goalies?.[0]?.slots?.[slot]).length;
    const scratches = players.filter((player) => player.availability?.key === "scratched").length;
    const average = assignedPlayers.length ? Math.round(assignedPlayers.reduce((sum, player) => sum + player.overall, 0) / assignedPlayers.length) : null;
    const warningCount = Object.values(warningsByUnit).reduce((sum, warnings) => sum + warnings.length, 0) + validation.errors.length;
    const freshCount = allLinks.filter((link) => link.fresh).length;
    return [
      { label: "Skaters", value: `${skaters.size}/18`, tone: skaters.size === 18 ? "good" : "warn" },
      { label: "Goalies", value: `${activeGoalies}/2`, tone: activeGoalies === 2 ? "good" : "warn" },
      { label: "Team chem", value: assignedPlayers.length ? `${teamChemistry.score}%` : null, title: "Average chemistry across all dressed lines, pairs, and goalies" },
      { label: "Avg OVR", value: average },
      { label: "New pairs", value: freshCount, tone: freshCount ? "warn" : "good", title: "Pairings with no shared history yet" },
      { label: "Scratches", value: scratches },
      { label: "Warnings", value: warningCount, tone: warningCount ? "warn" : "good" },
    ];
  }, [lineState, players, assignedPlayers, warningsByUnit, validation.errors, teamChemistry, allLinks]);
  const autoPreview = useMemo(() => autoBuild.open ? autoBuildState({ current: lineState, players, locks, mode: autoBuild.mode, scope: autoBuild.scope, includeThird: showThird }) : lineState, [autoBuild, lineState, players, locks, showThird]);
  const autoChanges = useMemo(() => previewChanges(lineState, autoPreview, playerMap), [lineState, autoPreview, playerMap]);

  const commit = useCallback((nextLines, nextLocks = locks, message = "Lineup updated.") => {
    const before = snapshot(lineState, locks);
    const after = snapshot(nextLines, nextLocks);
    if (snapshotKey(before) === snapshotKey(after)) return false;
    setHistory((current) => [...current, before].slice(-HISTORY_LIMIT));
    setFuture([]);
    setLineState(after.lineState);
    setLocks(after.locks);
    setSaveError(false);
    setAnnouncement(message);
    return true;
  }, [lineState, locks]);

  const undo = useCallback(() => {
    if (!history.length) return;
    const previous = history[history.length - 1];
    setHistory((current) => current.slice(0, -1));
    setFuture((current) => [snapshot(lineState, locks), ...current].slice(0, HISTORY_LIMIT));
    setLineState(cloneLines(previous.lineState));
    setLocks({ ...previous.locks });
    setSelectedSlotKey("");
    setSelectedPlayerId("");
    setAnnouncement("Change undone.");
  }, [history, lineState, locks]);
  const redo = useCallback(() => {
    if (!future.length) return;
    const next = future[0];
    setFuture((current) => current.slice(1));
    setHistory((current) => [...current, snapshot(lineState, locks)].slice(-HISTORY_LIMIT));
    setLineState(cloneLines(next.lineState));
    setLocks({ ...next.locks });
    setSelectedSlotKey("");
    setSelectedPlayerId("");
    setAnnouncement("Change restored.");
  }, [future, lineState, locks]);

  const placePlayer = useCallback((playerId, target) => {
    const player = playerMap[String(playerId || "")];
    if (!player || !target) return false;
    if (!player.availability?.placeable) { setToast({ type: "error", message: player.availability?.reason || "Player unavailable." }); setAnnouncement("Invalid move."); return false; }
    if (!posFit(player, target.slot)) { setToast({ type: "error", message: `${player.name} cannot play ${target.slot}.` }); setAnnouncement("Invalid position."); return false; }
    if (locks[target.key]) { setToast({ type: "warning", message: "Unlock this slot first." }); return false; }
    if (player.fatigue != null && player.fatigue >= 85 && !window.confirm("Place severely fatigued player?")) return false;
    const source = findAssignment(lineState, player.id);
    if (source && locks[source.key]) { setToast({ type: "warning", message: "Unlock the player slot first." }); return false; }
    if (source?.key === target.key) return false;
    const targetId = target.playerId || "";
    if (source && targetId) {
      const targetPlayer = playerMap[targetId];
      if (!targetPlayer || !posFit(targetPlayer, source.slot)) { setToast({ type: "error", message: "This swap breaks position rules." }); setAnnouncement("Invalid swap."); return false; }
    }
    let next = lineState;
    if (source) next = setSlot(next, source.group, source.lineId, source.slot, "");
    next = removePlayer(next, player.id);
    next = setSlot(next, target.group, target.lineId, target.slot, player.id);
    if (source && targetId) next = setSlot(next, source.group, source.lineId, source.slot, targetId);
    const changed = commit(next, locks, targetId ? "Players swapped." : "Player placed.");
    if (changed) {
      setSelectedUnit({ group: target.group, lineId: target.lineId });
      setSelectedSlotKey(target.key);
      setSelectedPlayerId("");
      setInspectorOpen(true);
    }
    return changed;
  }, [playerMap, locks, lineState, commit]);

  const onPlayerSelect = useCallback((playerId) => {
    if (selectedSlot) { placePlayer(playerId, selectedSlot); return; }
    setSelectedPlayerId((current) => current === playerId ? "" : playerId);
    setInspectorOpen(true);
    setTab("player");
    setComparing(false);
  }, [selectedSlot, placePlayer]);
  const onSlotSelect = useCallback((row) => {
    if (selectedPlayerId) { placePlayer(selectedPlayerId, row); return; }
    setSelectedUnit({ group: row.group, lineId: row.lineId });
    setSelectedSlotKey((current) => current === row.key ? "" : row.key);
    setSelectedPlayerId("");
    setInspectorOpen(true);
    setComparing(false);
  }, [selectedPlayerId, placePlayer]);
  const onDragStart = useCallback((event, playerId) => {
    const player = playerMap[playerId];
    const source = findAssignment(lineState, playerId);
    if (!player?.availability?.placeable || (source && locks[source.key])) { event.preventDefault(); return; }
    event.dataTransfer.setData("application/x-nhl-player", JSON.stringify({ pid: playerId }));
    event.dataTransfer.setData("text/plain", String(playerId));
    event.dataTransfer.effectAllowed = "move";
    setSelectedPlayerId(playerId);
  }, [playerMap, lineState, locks]);
  const onDrop = useCallback((event, target) => {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/x-nhl-player");
    const fallback = event.dataTransfer.getData("text/plain");
    let playerId = fallback;
    try { playerId = raw ? JSON.parse(raw)?.pid : fallback; } catch { playerId = fallback; }
    if (playerId) placePlayer(String(playerId), target);
  }, [placePlayer]);
  const removeSlot = useCallback((row) => {
    if (!row || locks[row.key]) return;
    const next = setSlot(lineState, row.group, row.lineId, row.slot, "");
    if (commit(next, locks, "Player removed.")) setSelectedPlayerId("");
  }, [lineState, locks, commit]);
  const toggleLock = useCallback((row) => {
    if (!row) return;
    const nextLocks = { ...locks };
    if (nextLocks[row.key]) delete nextLocks[row.key];
    else if (row.playerId) nextLocks[row.key] = true;
    commit(lineState, nextLocks, nextLocks[row.key] ? "Slot locked." : "Slot unlocked.");
  }, [lineState, locks, commit]);
  const registerSlotRef = useCallback((key, node) => { if (node) slotRefs.current.set(key, node); else slotRefs.current.delete(key); }, []);
  const onSlotKeyDown = useCallback((event, row) => {
    if (event.key === "Enter") { event.preventDefault(); onSlotSelect(row); return; }
    if (event.key === " ") { event.preventDefault(); if (selectedPlayerId) placePlayer(selectedPlayerId, row); else onSlotSelect(row); return; }
    if (event.key === "Delete") { event.preventDefault(); removeSlot(row); return; }
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const visible = descriptors(lineState, showThird).filter((item) => mode === LINEUP_MODES.ALL || item.group === mode);
    const index = visible.findIndex((item) => item.key === row.key);
    const perRow = row.group === "forwards" ? 3 : row.group === "defense" ? 2 : showThird ? 3 : 2;
    let nextIndex = index;
    if (event.key === "ArrowLeft") nextIndex -= 1;
    if (event.key === "ArrowRight") nextIndex += 1;
    if (event.key === "ArrowUp") nextIndex -= perRow;
    if (event.key === "ArrowDown") nextIndex += perRow;
    if (visible[nextIndex]) slotRefs.current.get(visible[nextIndex].key)?.focus();
  }, [onSlotSelect, selectedPlayerId, placePlayer, removeSlot, mode, lineState, showThird]);

  useEffect(() => {
    const handler = (event) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target?.tagName)) return;
      if (event.key === "Escape") { setSelectedPlayerId(""); setSelectedSlotKey(""); setMenuKey(""); setComparing(false); setAutoBuild((current) => ({ ...current, open: false, scope: null })); }
      if (event.key === "Delete" && selectedSlot) removeSlot(selectedSlot);
      if (["1", "2", "3", "4"].includes(event.key)) setMode({ 1: LINEUP_MODES.ALL, 2: LINEUP_MODES.FORWARDS, 3: LINEUP_MODES.DEFENSE, 4: LINEUP_MODES.GOALIES }[event.key]);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); if (event.shiftKey) redo(); else undo(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedSlot, removeSlot, undo, redo]);

  const openWarnings = useCallback((group, lineId) => { setSelectedUnit({ group, lineId }); setTab("warnings"); setInspectorOpen(true); setComparing(false); }, []);

  const clearUnit = useCallback((group, lineId) => {
    const line = lineState[group]?.find((item) => item.id === lineId);
    if (!line) return;
    const hasLocks = Object.keys(line.slots).some((slot) => locks[slotKey(group, lineId, slot)]);
    if (hasLocks && !window.confirm("Clear locked unit?")) return;
    let next = lineState;
    const nextLocks = { ...locks };
    for (const slot of Object.keys(line.slots)) { next = setSlot(next, group, lineId, slot, ""); delete nextLocks[slotKey(group, lineId, slot)]; }
    commit(next, nextLocks, "Unit cleared.");
    setMenuKey("");
  }, [lineState, locks, commit]);
  const lockUnit = useCallback((group, lineId, locked) => {
    const line = lineState[group]?.find((item) => item.id === lineId);
    if (!line) return;
    const nextLocks = { ...locks };
    for (const [slot, id] of Object.entries(line.slots)) {
      const key = slotKey(group, lineId, slot);
      if (locked && id) nextLocks[key] = true;
      else delete nextLocks[key];
    }
    commit(lineState, nextLocks, locked ? "Unit locked." : "Unit unlocked.");
    setMenuKey("");
  }, [lineState, locks, commit]);
  const copyUnit = useCallback((group, lineId) => {
    const line = lineState[group]?.find((item) => item.id === lineId);
    if (!line) return;
    setClipboard({ group, slots: { ...line.slots } });
    setToast({ type: "success", message: "Unit copied." });
    setMenuKey("");
  }, [lineState]);
  const pasteUnit = useCallback((group, lineId) => {
    const line = lineState[group]?.find((item) => item.id === lineId);
    if (!line || !clipboard) return;
    if (JSON.stringify(Object.keys(clipboard.slots || {})) !== JSON.stringify(Object.keys(line.slots || {}))) { setToast({ type: "error", message: "Unit positions do not match." }); return; }
    let next = lineState;
    for (const [slot, playerId] of Object.entries(clipboard.slots)) {
      const key = slotKey(group, lineId, slot);
      if (locks[key]) continue;
      const player = playerMap[String(playerId || "")];
      if (!playerId || !player || !posFit(player, slot)) { next = setSlot(next, group, lineId, slot, ""); continue; }
      const source = findAssignment(next, player.id);
      if (source && locks[source.key]) continue;
      next = removePlayer(next, player.id);
      next = setSlot(next, group, lineId, slot, player.id);
    }
    commit(next, locks, "Unit pasted.");
    setMenuKey("");
  }, [lineState, clipboard, locks, playerMap, commit]);
  const resetUnit = useCallback((group, lineId) => {
    const savedLine = savedSnapshot.lineState?.[group]?.find((item) => item.id === lineId);
    const currentLine = lineState[group]?.find((item) => item.id === lineId);
    if (!savedLine || !currentLine) return;
    let next = lineState;
    const nextLocks = { ...locks };
    for (const slot of Object.keys(currentLine.slots)) {
      next = setSlot(next, group, lineId, slot, savedLine.slots?.[slot] || "");
      const key = slotKey(group, lineId, slot);
      if (savedSnapshot.locks?.[key]) nextLocks[key] = true; else delete nextLocks[key];
    }
    commit(next, nextLocks, "Unit reset.");
    setMenuKey("");
  }, [savedSnapshot, lineState, locks, commit]);
  const actions = useMemo(() => ({
    clear: clearUnit,
    autoBuild: (group, lineId) => { setAutoBuild((current) => ({ ...current, open: true, scope: { group, lineId } })); setMenuKey(""); },
    lock: (group, lineId) => lockUnit(group, lineId, true),
    unlock: (group, lineId) => lockUnit(group, lineId, false),
    copy: copyUnit,
    paste: pasteUnit,
    reset: resetUnit,
    close: () => setMenuKey(""),
  }), [clearUnit, lockUnit, copyUnit, pasteUnit, resetUnit]);

  const resetSaved = useCallback(() => {
    if (unsaved && !window.confirm("Reset unsaved changes?")) return;
    setHistory((current) => [...current, snapshot(lineState, locks)].slice(-HISTORY_LIMIT));
    setFuture([]);
    setLineState(cloneLines(savedSnapshot.lineState));
    setLocks({ ...savedSnapshot.locks });
    setSelectedSlotKey("");
    setSelectedPlayerId("");
    setToast({ type: "success", message: "Saved lineup restored." });
  }, [unsaved, lineState, locks, savedSnapshot]);
  const clearAll = useCallback(() => {
    if (!window.confirm("Clear every lineup slot?")) return;
    commit(emptyLines(showThird), {}, "All lineup slots cleared.");
  }, [showThird, commit]);
  const applyAutoBuild = useCallback(() => {
    if (!autoChanges.length) return;
    commit(autoPreview, locks, "Auto build applied.");
    setAutoBuild((current) => ({ ...current, open: false, scope: null }));
  }, [autoChanges, autoPreview, locks, commit]);
  const saveLines = useCallback(async () => {
    if (saving || !players.length) return;
    if (validation.errors.length) {
      setSaveError(true);
      setToast({ type: "error", message: validation.errors[0].text, details: validation.errors.length > 1 });
      return;
    }
    if (validation.incomplete && !window.confirm("Save incomplete lineup?")) { setToast({ type: "warning", message: "Save cancelled." }); return; }
    setSaving(true);
    setSaveError(false);
    try {
      const response = await saveFranchiseLines({ unit_type: "even_strength", lines: lineState });
      if (response?.lines) setFranchiseState((previous) => ({ ...(previous || {}), lines: response.lines }));
      writeSessionLineupCache(LINEUP_KIND, lineState, sessionId);
      props.onSaveEditLines?.(lineState);
      props.onSave?.({ type: "editLines", lines: lineState });
      setSavedSnapshot(snapshot(lineState, locks));
      setChemNonce((value) => value + 1);
      const fallout = Array.isArray(response?.storyline_events) ? response.storyline_events : [];
      if (fallout.length) {
        setToast({
          type: "success",
          message: `Line change generated ${fallout.length} reaction(s)`,
          actionLabel: "Storylines",
          onClick: () => setScreen(SCREENS.STORYLINES),
        });
      } else {
        setToast(response?.warnings?.length ? { type: "warning", message: shortText(response.warnings[0]), details: response.warnings.length > 1 } : { type: "success", message: "Lines saved." });
      }
    } catch {
      setSaveError(true);
      setToast({ type: "error", message: "Backend save failed." });
    } finally {
      setSaving(false);
    }
  }, [saving, players.length, validation, lineState, locks, sessionId, props, setFranchiseState, setScreen]);
  const replaceSelected = useCallback((playerId) => { if (selectedSlot && placePlayer(playerId, selectedSlot)) setComparing(false); }, [selectedSlot, placePlayer]);
  const selectedLocked = selectedSlot ? Boolean(locks[selectedSlot.key]) : false;
  const rosterLoading = !franchiseState && !players.length;
  const live = !unsaved && !teamChemistry.projected;
  const subtitle = `${players.length} players · ${live ? "Live chemistry" : unsaved ? "Unsaved edits — projected scores" : "Projected chemistry"}`;

  return <div className="linebuilder-root">
    <EditLinesStyles />
    <LineBuilderSidebar setScreen={setScreen} abbreviation={team.abbreviation} activeScreen={SCREENS.EDIT_LINES} />
    <main className="lb-shell">
      <LineBuilderHeader
        team={team}
        title="Line chemistry"
        subtitle={subtitle}
        live={live}
        canUndo={history.length > 0}
        canRedo={future.length > 0}
        onUndo={undo}
        onRedo={redo}
        onClear={clearAll}
        onReset={resetSaved}
        onAutoBuild={() => setAutoBuild((current) => ({ ...current, open: !current.open, scope: null }))}
        onSave={saveLines}
        saving={saving}
        unsaved={unsaved}
        saveError={saveError}
        disabled={!players.length}
      />
      <LineStatusStrip metrics={statusMetrics} meter={chemMeter} />
      <div className="lb-workspace">
        {rosterLoading
          ? <section className="lb-region lb-pool-region"><div className="lb-region-head"><h2 className="lb-region-title">Player pool</h2></div><div className="lb-loading"><div className="lb-skeleton" /></div></section>
          : <PlayerPool players={filteredPool} assignedSet={assignedSet} lockedSet={lockedSet} selectedPlayerId={selectedPlayerId} search={search} setSearch={setSearch} focusSlot={focusSlot} onPlayerSelect={onPlayerSelect} onDragStart={onDragStart} open={poolOpen} onClose={() => setPoolOpen(false)} />}
        <FormationBoard
          mode={mode}
          setMode={setMode}
          lineState={lineState}
          playerMap={playerMap}
          chemReport={chemReport}
          chemistryByUnit={chemistryByUnit}
          warningsByUnit={warningsByUnit}
          linksByUnit={linksByUnit}
          depthLinks={depthLinks}
          selectedUnit={selectedUnit}
          selectedSlotKey={selectedSlotKey}
          selectedPlayer={selectedPlayerId ? playerMap[selectedPlayerId] : null}
          locks={locks}
          showThird={showThird}
          menuKey={menuKey}
          clipboard={clipboard}
          onSelectUnit={(group, lineId) => { setSelectedUnit({ group, lineId }); setInspectorOpen(true); }}
          onSlotSelect={onSlotSelect}
          onDrop={onDrop}
          onDragStart={onDragStart}
          onRemove={removeSlot}
          onToggleLock={toggleLock}
          onSlotKeyDown={onSlotKeyDown}
          registerSlotRef={registerSlotRef}
          onWarnings={openWarnings}
          onMenuToggle={(group, lineId) => { const key = `${group}:${lineId}`; setMenuKey((current) => current === key ? "" : key); }}
          actions={actions}
          onTogglePool={() => setPoolOpen((current) => !current)}
          onToggleInspector={() => setInspectorOpen((current) => !current)}
          rosterEmpty={!players.length && !rosterLoading}
        />
        <LineInspector
          open={inspectorOpen}
          onClose={() => setInspectorOpen(false)}
          tab={tab}
          setTab={setTab}
          selectedLine={selectedLine}
          selectedGroup={selectedUnit.group}
          selectedPlayer={selectedPlayer}
          selectedSlot={selectedSlot}
          playerMap={playerMap}
          chemistry={selectedChemistry}
          warnings={selectedWarnings}
          unitPlayers={unitPlayers}
          unitLinks={selectedLinks}
          comparisonPlayers={comparisonPlayers}
          comparing={comparing}
          setComparing={setComparing}
          onReplace={replaceSelected}
          onRemoveSelected={() => selectedSlot && removeSlot(selectedSlot)}
          onToggleSelectedLock={() => selectedSlot && toggleLock(selectedSlot)}
          selectedLocked={selectedLocked}
          chemistryLoading={chemLoading}
        />
      </div>
      <AutoBuildPopover state={autoBuild} setState={setAutoBuild} changes={autoChanges} onApply={applyAutoBuild} />
    </main>
    <LineBuilderToast toast={toast} onDismiss={() => setToast(null)} onDetails={() => { setTab("warnings"); setInspectorOpen(true); setToast(null); }} />
    <div className="lb-live" aria-live="polite">{announcement}</div>
  </div>;
}

/* ---------------------------------------------------------------------------
   Special teams. Same link language, different shapes:
   PP  = forward triangle + defence pair + a dashed point-to-bumper feed
   PK  = four-sided box, every killer linked only to the two beside them
--------------------------------------------------------------------------- */
function specialTeamsProfile(player, key, fallback = 75) {
  const profile = player?.chemistry_profile || {};
  return Number(profile[key] ?? profile[String(key).replaceAll("_", "")] ?? fallback);
}

function calculatePowerPlayChemistry(players) {
  const selected = players.filter(Boolean);
  if (!selected.length) {
    return { score: 0, label: "No unit", creativity: 0, movement: 0, finishing: 0, balance: 0, tips: ["Select players to build your power play unit."] };
  }
  const avgOverall = selected.reduce((sum, p) => sum + (Number(p.overall) || 0), 0) / selected.length;
  const creativity = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "creativity"), 0) / selected.length;
  const movement = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "puck_movement"), 0) / selected.length;
  const finishing = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "offensive_instinct"), 0) / selected.length;
  const hasCenter = selected.some((p) => p.position === "C");
  const hasDefense = selected.some((p) => isDefense(p.position));
  const hasShooter = selected.some((p) => String(p.role).toLowerCase().includes("sniper"));
  const hasPlaymaker = selected.some((p) => String(p.role).toLowerCase().includes("playmaker"));
  const hasNetFront = selected.some((p) => String(p.role).toLowerCase().includes("net"));
  const balance = (hasCenter ? 20 : 5) + (hasDefense ? 20 : 5) + (hasShooter ? 20 : 8) + (hasPlaymaker ? 20 : 8) + (hasNetFront ? 20 : 8);
  const handMixBonus = selected.some((p) => p.handedness === "L") && selected.some((p) => p.handedness === "R") ? 5 : 0;
  const score = clamp(avgOverall * 0.22 + creativity * 0.24 + movement * 0.22 + finishing * 0.2 + balance * 0.12 + handMixBonus);
  const label = score >= 90 ? "Terrifying unit" : score >= 80 ? "Dangerous PP" : score >= 70 ? "Functional unit" : score >= 60 ? "Needs a trigger" : "Disconnected";
  const tips = [];
  if (!hasShooter) tips.push("Add a true shooter so the unit has a dangerous trigger.");
  if (!hasPlaymaker) tips.push("Add a playmaker to improve puck movement and zone control.");
  if (!hasNetFront) tips.push("A net-front player helps screens, rebounds, and dirty goals.");
  if (!hasDefense) tips.push("Use at least one defenceman or power-play quarterback.");
  if (movement < 78) tips.push("Puck movement is low. This unit may become too static.");
  if (score >= 82) tips.push("Strong mix of movement, finishing, and role fit.");
  return { score, label, creativity: clamp(creativity), movement: clamp(movement), finishing: clamp(finishing), balance: clamp(balance), tips };
}

function calculatePenaltyKillChemistry(players) {
  const selected = players.filter(Boolean);
  if (!selected.length) {
    return { score: 0, label: "No unit", defensive: 0, discipline: 0, balance: 0, trust: 0, tips: ["Select players to build your penalty kill unit."] };
  }
  const avgOverall = selected.reduce((sum, p) => sum + (Number(p.overall) || 0), 0) / selected.length;
  const defensive = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "defensive_buy_in"), 0) / selected.length;
  const discipline = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "discipline"), 0) / selected.length;
  const workEthic = selected.reduce((sum, p) => sum + specialTeamsProfile(p, "work_ethic"), 0) / selected.length;
  const hasTwoForwards = selected.filter((p) => isForward(p.position)).length >= 2;
  const hasTwoDefense = selected.filter((p) => isDefense(p.position)).length >= 2;
  const balance = (hasTwoForwards ? 50 : 20) + (hasTwoDefense ? 50 : 20);
  const roleBonus = selected.some((p) => String(p.role).toLowerCase().includes("shutdown")) ? 5 : 0;
  const checkerBonus = selected.some((p) => {
    const role = String(p.role).toLowerCase();
    return role.includes("checker") || role.includes("grinder");
  }) ? 4 : 0;
  const score = clamp(avgOverall * 0.2 + defensive * 0.34 + discipline * 0.22 + workEthic * 0.14 + balance * 0.1 + roleBonus + checkerBonus);
  const label = score >= 90 ? "Elite PK identity" : score >= 80 ? "Reliable killers" : score >= 70 ? "Playable unit" : score >= 60 ? "Risky mix" : "Needs work";
  const tips = [];
  if (defensive < 78) tips.push("Defensive buy-in is low. Add a shutdown defender or two-way forward.");
  if (discipline < 76) tips.push("Discipline is dragging this unit down. Avoid penalty-prone players here.");
  if (!hasTwoDefense) tips.push("This PK unit should include two defencemen.");
  if (!hasTwoForwards) tips.push("This PK unit should include two forwards.");
  if (score >= 82) tips.push("Strong defensive trust — this unit can handle tough matchups.");
  return { score, label, defensive: clamp(defensive), discipline: clamp(discipline), balance: clamp(balance), trust: clamp(workEthic), tips };
}

function makeInitialPowerPlayLines(players) {
  const forwards = players.filter((p) => isForward(p.position));
  const defenders = players.filter((p) => isDefense(p.position));
  return [
    { id: "pp1", name: "Power Play 1", slots: { LW: forwards[0]?.id || "", C: forwards[1]?.id || "", RW: forwards[2]?.id || "", LD: defenders[0]?.id || "", RD: defenders[1]?.id || defenders[0]?.id || "" } },
    { id: "pp2", name: "Power Play 2", slots: { LW: forwards[3]?.id || forwards[0]?.id || "", C: forwards[4]?.id || forwards[1]?.id || "", RW: forwards[5]?.id || forwards[2]?.id || "", LD: defenders[2]?.id || defenders[0]?.id || "", RD: defenders[3]?.id || defenders[1]?.id || "" } },
  ];
}

function makeInitialPenaltyKillLines(players) {
  const forwards = players.filter((p) => isForward(p.position));
  const defenders = players.filter((p) => isDefense(p.position));
  return [
    { id: "pk1", name: "Penalty Kill 1", slots: { F1: forwards[0]?.id || "", F2: forwards[1]?.id || "", D1: defenders[0]?.id || "", D2: defenders[1]?.id || "" } },
    { id: "pk2", name: "Penalty Kill 2", slots: { F1: forwards[2]?.id || forwards[0]?.id || "", F2: forwards[3]?.id || forwards[1]?.id || "", D1: defenders[2]?.id || defenders[0]?.id || "", D2: defenders[3]?.id || defenders[1]?.id || "" } },
    { id: "pk3", name: "Penalty Kill 3", slots: { F1: forwards[4]?.id || forwards[0]?.id || "", F2: forwards[5]?.id || forwards[1]?.id || "", D1: defenders[4]?.id || defenders[0]?.id || "", D2: defenders[5]?.id || defenders[1]?.id || "" } },
  ];
}

function emptySpecialTeamsLines(kind) {
  if (kind === "penalty_kill") {
    return [1, 2, 3].map((n) => ({ id: `pk${n}`, name: `Penalty Kill ${n}`, slots: { F1: "", F2: "", D1: "", D2: "" } }));
  }
  return [1, 2].map((n) => ({ id: `pp${n}`, name: `Power Play ${n}`, slots: { LW: "", C: "", RW: "", LD: "", RD: "" } }));
}

function specialTeamsSlotAllowed(player, slot, kind) {
  if (!player) return false;
  if (kind === "penalty_kill") {
    if (String(slot).startsWith("D")) return isDefense(player.position);
    if (String(slot).startsWith("F")) return isForward(player.position);
    return false;
  }
  if (slot === "LD" || slot === "RD") return isDefense(player.position);
  return isForward(player.position);
}

function findSpecialTeamsAssignment(lines, playerId) {
  const id = String(playerId || "");
  for (const line of lines || []) {
    for (const [slot, value] of Object.entries(line.slots || {})) {
      if (String(value || "") === id) return { lineId: line.id, slot, key: `${line.id}:${slot}` };
    }
  }
  return null;
}

function SpecialTeamsUnit({ kind, line, index, playerMap, chemReport, chemistry, selected, selectedSlot, selectedPlayerId, onSelectUnit, onSlotSelect, onDrop, onDragStart }) {
  const isPP = kind === "power_play";
  const geo = isPP ? PP_GEO : PK_GEO;
  const pairs = isPP ? PP_LINKS : PK_LINKS;
  const slots = isPP ? ["LW", "C", "RW", "LD", "RD"] : ["F1", "F2", "D1", "D2"];
  const slotPlayers = slotPlayersFor(line, playerMap);
  const links = buildLinks(pairs, geo, slotPlayers, chemReport);
  const badgeLinks = [...links];
  if (isPP) {
    const point = links.find((link) => link.key === "LD-RD");
    const bumper = slotPlayers.C;
    const feedTier = point && bumper ? point.tier : "empty";
    links.push({ key: "feed", slotA: "LD", slotB: "C", x1: 50.5, y1: 83, x2: 49.5, y2: 34, mx: 50, my: 58, tier: feedTier, dashed: true, score: null, fresh: false });
  }
  const unitNumber = String(line.name || "").replace(/\D+/g, "") || String(index + 1);
  const selectedPlayer = playerMap[selectedPlayerId] || null;
  return <div className={`fm-unit ${isPP ? "ppf" : "pkbox"} ${selected ? "selected" : ""}`} style={isPP ? { height: 300 } : undefined} onClick={() => onSelectUnit(line.id)}>
    <LinkLayer links={links} />
    <LinkBadges links={badgeLinks} />
    <div className="fm-unit-tag">
      <span className="fm-unit-kicker">{isPP ? "PP" : "PK"}</span>
      <span className="fm-unit-num">{unitNumber}</span>
      <span className={`fm-unit-score ${scoreTone(chemistry?.score)}`}>{chemistry?.score ?? 0}%</span>
    </div>
    {slots.map((slot) => {
      const descriptor = { key: `${line.id}:${slot}`, lineId: line.id, slot, playerId: String(line.slots?.[slot] || "") };
      const player = playerMap[descriptor.playerId] || null;
      const slotLinks = badgeLinks.filter((link) => link.slotA === slot || link.slotB === slot);
      const tiers = slotLinks.map((link) => link.tier);
      const worst = tiers.includes("weak") ? "weak" : tiers.includes("forming") ? "forming" : tiers.includes("strong") ? "strong" : "";
      const fresh = Boolean(player) && slotLinks.length > 0 && slotLinks.every((link) => link.fresh);
      let targetState = "";
      if (selectedPlayer) targetState = !specialTeamsSlotAllowed(selectedPlayer, slot, kind) ? "invalid" : player ? "swap" : "valid";
      return <FormationCard
        key={descriptor.key}
        descriptor={descriptor}
        geo={geo[slot]}
        player={player}
        tier={worst}
        pips={tiers}
        fresh={fresh}
        locked={false}
        selected={selectedSlot === descriptor.key}
        targetState={targetState}
        onSelect={() => onSlotSelect(line.id, slot)}
        onDrop={(event) => onDrop(event, line.id, slot)}
        onDragStart={onDragStart}
        onRemove={() => {}}
        onToggleLock={() => {}}
        onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSlotSelect(line.id, slot); } }}
        registerRef={null}
      />;
    })}
  </div>;
}

function SpecialTeamsLines({ kind, ...props }) {
  const isPP = kind === "power_play";
  const { franchiseState, setScreen, setFranchiseState } = useGameUI();
  const sessionId = getFranchiseSessionId();
  const team = useMemo(() => teamIdentity(franchiseState, props), [franchiseState, props]);
  const activeScreen = isPP ? SCREENS.POWER_PLAY : SCREENS.PENALTY_KILL;
  const title = isPP ? "Power play" : "Penalty kill";
  const chemistryFn = isPP ? calculatePowerPlayChemistry : calculatePenaltyKillChemistry;
  const makeInitial = isPP ? makeInitialPowerPlayLines : makeInitialPenaltyKillLines;

  const [chemReport, setChemReport] = useState(null);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState(null);
  const [invalidMsg, setInvalidMsg] = useState("");
  const [unsaved, setUnsaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [selectedLineId, setSelectedLineId] = useState(isPP ? "pp1" : "pk1");
  const [selectedSlot, setSelectedSlot] = useState("");
  const [selectedSlotKey, setSelectedSlotKey] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [poolOpen, setPoolOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [lines, setLines] = useState(() => {
    try {
      const backend = franchiseState?.lines?.[kind]?.lines;
      if (Array.isArray(backend) && backend.length) return backend;
      const cached = readSessionLineupCache(kind, sessionId);
      if (Array.isArray(cached) && cached.length) return cached;
    } catch {
      // ignore bad cache
    }
    return emptySpecialTeamsLines(kind);
  });
  const hydratedSession = useRef("");

  useEffect(() => {
    let active = true;
    getFranchiseChemistry().then((data) => { if (active) setChemReport(data || null); }).catch(() => { if (active) setChemReport(null); });
    return () => { active = false; };
  }, [franchiseState?.session_id]);

  const players = useMemo(() => {
    const lean = Array.isArray(franchiseState?.roster) ? franchiseState.roster : [];
    if (lean.length) return lean.map(normalizePlayer);
    const organizations = franchiseState?.roster_browser?.organizations || [];
    const teamId = String(franchiseState?.user_team_id || "");
    const organization = organizations.find((candidate) => String(candidate?.team_id || "") === teamId) || organizations[0];
    const live = Array.isArray(organization?.nhl) ? organization.nhl : [];
    return live.length ? live.map(normalizePlayer) : getRosterPlayers(props);
  }, [franchiseState, props]);
  const playerMap = useMemo(() => players.reduce((map, player) => { map[player.id] = player; return map; }, {}), [players]);

  useEffect(() => {
    const backend = franchiseState?.lines?.[kind]?.lines;
    if (Array.isArray(backend) && backend.length) {
      setLines(backend);
      setUnsaved(false);
      hydratedSession.current = `${sessionId}:${kind}`;
    }
  }, [franchiseState?.lines?.[kind]?.lines, kind, sessionId]);

  useEffect(() => {
    const key = `${sessionId}:${kind}`;
    if (!players.length || hydratedSession.current === key) return;
    const backend = franchiseState?.lines?.[kind]?.lines;
    if (Array.isArray(backend) && backend.length) {
      setLines(backend);
      hydratedSession.current = key;
      return;
    }
    const cached = readSessionLineupCache(kind, sessionId);
    if (Array.isArray(cached) && cached.length) setLines(cached);
    else setLines(makeInitial(players));
    hydratedSession.current = key;
  }, [players, kind, sessionId, franchiseState?.lines, makeInitial]);

  useEffect(() => { if (toast?.type !== "success") return undefined; const timer = window.setTimeout(() => setToast(null), 2600); return () => window.clearTimeout(timer); }, [toast]);

  const assignedSet = useMemo(() => new Set(lines.flatMap((line) => Object.values(line.slots || {})).filter(Boolean).map(String)), [lines]);
  const emptyLockedSet = useMemo(() => new Set(), []);
  const geo = isPP ? PP_GEO : PK_GEO;
  const linkPairs = isPP ? PP_LINKS : PK_LINKS;
  const linksByLine = useMemo(() => {
    const result = {};
    for (const line of lines) result[line.id] = buildLinks(linkPairs, geo, slotPlayersFor(line, playerMap), chemReport);
    return result;
  }, [lines, playerMap, chemReport, geo, linkPairs]);
  const allLinks = useMemo(() => Object.values(linksByLine).flat().filter((link) => link.score != null), [linksByLine]);
  const chemMeter = useMemo(() => allLinks.map((link) => link.tier), [allLinks]);
  const selectedLine = useMemo(() => lines.find((line) => line.id === selectedLineId) || lines[0] || null, [lines, selectedLineId]);
  const unitPlayers = useMemo(() => selectedLine ? Object.values(selectedLine.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean) : [], [selectedLine, playerMap]);
  const unitChemistry = useMemo(() => chemistryFn(unitPlayers), [chemistryFn, unitPlayers]);
  const chemistryByLine = useMemo(() => {
    const result = {};
    for (const line of lines) {
      const linePlayers = Object.values(line.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean);
      result[line.id] = chemistryFn(linePlayers);
    }
    return result;
  }, [lines, playerMap, chemistryFn]);
  const selectedPlayer = useMemo(() => {
    if (selectedPlayerId && playerMap[selectedPlayerId]) return playerMap[selectedPlayerId];
    if (selectedLine && selectedSlot) return playerMap[String(selectedLine.slots?.[selectedSlot] || "")] || null;
    return null;
  }, [selectedPlayerId, playerMap, selectedLine, selectedSlot]);
  const filteredPool = useMemo(() => {
    const query = search.trim().toLowerCase();
    return sortPoolPlayers(players.filter((player) => !query || `${player.name} ${player.position} ${player.role}`.toLowerCase().includes(query)), "overall", null);
  }, [players, search]);
  const missingCount = useMemo(() => lines.reduce((sum, line) => sum + Object.values(line.slots || {}).filter((id) => !id).length, 0), [lines]);
  const statusMetrics = useMemo(() => {
    const filled = lines.reduce((sum, line) => sum + Object.keys(line.slots || {}).length, 0);
    const freshCount = allLinks.filter((link) => link.fresh).length;
    const average = Object.values(chemistryByLine).map((entry) => entry.score).filter(Boolean);
    return [
      { label: "Units", value: `${lines.length}` },
      { label: "Filled", value: `${assignedSet.size}/${filled}`, tone: missingCount ? "warn" : "good" },
      { label: "Avg unit", value: average.length ? `${Math.round(average.reduce((sum, value) => sum + value, 0) / average.length)}%` : null },
      { label: "New pairs", value: freshCount, tone: freshCount ? "warn" : "good" },
      { label: "Warnings", value: missingCount + (invalidMsg ? 1 : 0), tone: missingCount || invalidMsg ? "warn" : "good" },
    ];
  }, [lines, assignedSet, missingCount, invalidMsg, allLinks, chemistryByLine]);

  const placePlayer = useCallback((playerId, lineId, slot) => {
    const player = playerMap[String(playerId || "")];
    if (!player || !lineId || !slot) return false;
    if (!specialTeamsSlotAllowed(player, slot, kind)) {
      setInvalidMsg(`${player.name || "Player"} is not a valid fit for ${slot}.`);
      setToast({ type: "error", message: `${player.name} cannot play ${slot}.` });
      return false;
    }
    if (!player.availability?.placeable) {
      setToast({ type: "error", message: player.availability?.reason || "Player unavailable." });
      return false;
    }
    setInvalidMsg("");
    setLines((current) => {
      const from = findSpecialTeamsAssignment(current, player.id);
      const targetLine = current.find((line) => line.id === lineId);
      const replaced = targetLine?.slots?.[slot] || "";
      let next = current.map((line) => line.id === lineId ? { ...line, slots: { ...line.slots, [slot]: player.id } } : line);
      if (from) {
        next = next.map((line) => line.id === from.lineId ? { ...line, slots: { ...line.slots, [from.slot]: replaced && replaced !== player.id ? replaced : "" } } : line);
      }
      return next;
    });
    setUnsaved(true);
    setSaveError(false);
    setSelectedLineId(lineId);
    setSelectedSlot(slot);
    setSelectedSlotKey(`${lineId}:${slot}`);
    setSelectedPlayerId("");
    setInspectorOpen(true);
    return true;
  }, [playerMap, kind]);

  const onDragStart = useCallback((event, playerId) => {
    event.dataTransfer.setData("application/x-nhl-player", JSON.stringify({ pid: playerId }));
    event.dataTransfer.setData("text/plain", String(playerId));
    event.dataTransfer.effectAllowed = "move";
    setSelectedPlayerId(playerId);
    setInvalidMsg("");
  }, []);

  const onDropSlot = useCallback((event, lineId, slot) => {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/x-nhl-player");
    const fallback = event.dataTransfer.getData("text/plain");
    let playerId = fallback;
    try { playerId = raw ? JSON.parse(raw)?.pid : fallback; } catch { playerId = fallback; }
    if (playerId) placePlayer(String(playerId), lineId, slot);
  }, [placePlayer]);

  const onPlayerSelect = useCallback((playerId) => {
    if (selectedLineId && selectedSlot) { placePlayer(playerId, selectedLineId, selectedSlot); return; }
    setSelectedPlayerId((current) => current === playerId ? "" : playerId);
    setInspectorOpen(true);
  }, [selectedLineId, selectedSlot, placePlayer]);

  const onSlotSelect = useCallback((lineId, slot) => {
    if (selectedPlayerId) { placePlayer(selectedPlayerId, lineId, slot); return; }
    setSelectedLineId(lineId);
    setSelectedSlot((current) => current === slot && selectedLineId === lineId ? "" : slot);
    setSelectedSlotKey((current) => current === `${lineId}:${slot}` ? "" : `${lineId}:${slot}`);
    setSelectedPlayerId("");
    setInspectorOpen(true);
  }, [selectedPlayerId, selectedLineId, placePlayer]);

  const clearAll = useCallback(() => {
    if (!window.confirm(`Clear all ${isPP ? "power play" : "penalty kill"} slots?`)) return;
    setLines(emptySpecialTeamsLines(kind));
    setUnsaved(true);
    setInvalidMsg("");
  }, [kind, isPP]);

  const resetLines = useCallback(() => {
    const fresh = makeInitial(players);
    setLines(fresh);
    writeSessionLineupCache(kind, fresh, sessionId);
    setUnsaved(true);
    setInvalidMsg("");
    setToast({ type: "success", message: "Units auto-filled." });
  }, [makeInitial, players, kind, sessionId]);

  const saveLines = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    setSaveError(false);
    writeSessionLineupCache(kind, lines, sessionId);
    if (isPP) props.onSavePowerPlay?.(lines);
    else props.onSavePenaltyKill?.(lines);
    props.onSave?.({ type: isPP ? "powerPlay" : "penaltyKill", lines });
    try {
      const res = await saveFranchiseLines({ unit_type: kind, lines });
      if (res?.lines) setFranchiseState((prev) => ({ ...(prev || {}), lines: res.lines }));
      setUnsaved(false);
      const fallout = Array.isArray(res?.storyline_events) ? res.storyline_events : [];
      if (fallout.length) {
        setToast({
          type: "success",
          message: `Unit change generated ${fallout.length} reaction(s)`,
          actionLabel: "Storylines",
          onClick: () => setScreen(SCREENS.STORYLINES),
        });
      } else {
        setToast({ type: "success", message: `${isPP ? "PP" : "PK"} units saved.` });
      }
    } catch {
      setSaveError(true);
      setToast({ type: "error", message: "Backend save failed." });
    } finally {
      setSaving(false);
    }
  }, [saving, kind, lines, sessionId, isPP, props, setFranchiseState, setScreen]);

  const metricRows = isPP
    ? [["Creativity", unitChemistry.creativity], ["Puck movement", unitChemistry.movement], ["Finishing", unitChemistry.finishing], ["Role balance", unitChemistry.balance]]
    : [["Defensive buy-in", unitChemistry.defensive], ["Discipline", unitChemistry.discipline], ["Unit balance", unitChemistry.balance], ["Work ethic / trust", unitChemistry.trust]];

  const selectedLinks = linksByLine[selectedLine?.id] || [];
  const bondAverage = averageLinkScore(selectedLinks);
  const freshLinks = selectedLinks.filter((link) => link.fresh && link.score != null);
  const rosterLoading = !franchiseState && !players.length;
  const linkName = (slot) => lastName(playerMap[String(selectedLine?.slots?.[slot] || "")]?.name || slot);

  return <div className="linebuilder-root">
    <EditLinesStyles />
    <LineBuilderSidebar setScreen={setScreen} abbreviation={team.abbreviation} activeScreen={activeScreen} />
    <main className="lb-shell">
      <LineBuilderHeader
        team={team}
        title={title}
        subtitle={`${players.length} players · ${isPP ? "Creativity and puck movement weighted" : "Box coverage — every killer covers their partner"}`}
        live={!unsaved}
        showHistory={false}
        showAutoBuild={false}
        onClear={clearAll}
        onReset={resetLines}
        resetLabel="Auto fill"
        onSave={saveLines}
        saving={saving}
        unsaved={unsaved}
        saveError={saveError}
        disabled={!players.length}
        saveLabel={unsaved ? `Save ${isPP ? "PP" : "PK"}` : "Saved"}
      />
      <LineStatusStrip metrics={statusMetrics} meter={chemMeter} />
      <div className="lb-workspace">
        {rosterLoading
          ? <section className="lb-region lb-pool-region"><div className="lb-region-head"><h2 className="lb-region-title">Player pool</h2></div><div className="lb-loading"><div className="lb-skeleton" /></div></section>
          : <PlayerPool players={filteredPool} assignedSet={assignedSet} lockedSet={emptyLockedSet} selectedPlayerId={selectedPlayerId} search={search} setSearch={setSearch} focusSlot={null} onPlayerSelect={onPlayerSelect} onDragStart={onDragStart} open={poolOpen} onClose={() => setPoolOpen(false)} />}

        <section className="lb-region lb-board-region">
          <div className="lb-modebar">
            <button type="button" className="lb-mode active">{isPP ? "PP units" : "PK units"}</button>
            <div className="lb-mobile-tools">
              <button type="button" className="lb-icon" onClick={() => setPoolOpen((v) => !v)} aria-label="Open player pool" title="Players">P</button>
              <button type="button" className="lb-icon" onClick={() => setInspectorOpen((v) => !v)} aria-label="Open unit details" title="Details">D</button>
            </div>
          </div>
          {!players.length && !rosterLoading ? <div className="lb-empty">Roster unavailable</div> : <div className="fm-scroll">
            <SectionHead title={isPP ? "Power play" : "Penalty kill"} note={isPP ? "Triangle up top, pair on the points" : "Four-sided box"} />
            {lines.map((line, index) => <SpecialTeamsUnit
              key={line.id}
              kind={kind}
              line={line}
              index={index}
              playerMap={playerMap}
              chemReport={chemReport}
              chemistry={chemistryByLine[line.id]}
              selected={selectedLineId === line.id}
              selectedSlot={selectedSlotKey}
              selectedPlayerId={selectedPlayerId}
              onSelectUnit={(lineId) => { setSelectedLineId(lineId); setInspectorOpen(true); }}
              onSlotSelect={onSlotSelect}
              onDrop={onDropSlot}
              onDragStart={onDragStart}
            />)}
            <FormationLegend depthLabel={isPP ? "Point-to-bumper feed" : "Box coverage"} />
          </div>}
        </section>

        <section className={`lb-region lb-inspector-region ${inspectorOpen ? "open" : ""}`}>
          <div className="lb-region-head">
            <h2 className="lb-region-title">Inspector</h2>
            <span className="lb-region-note">{selectedLine?.name || title}</span>
            <button type="button" className="lb-icon lb-drawer-close" onClick={() => setInspectorOpen(false)} aria-label="Close inspector" title="Close">×</button>
          </div>
          <div className="lb-inspector" style={{ gridTemplateRows: "minmax(0,1fr)" }}>
            <div className="lb-inspector-body">
              <div>
                <div className="lb-inspector-title">{selectedLine?.name || title}</div>
                <div className="lb-inspector-sub">{isPP ? "Power play unit" : "Penalty kill unit"}</div>
              </div>
              <div className="lb-gauge">
                <span className={`lb-gauge-score ${scoreTone(unitChemistry.score)}`}>{unitChemistry.score}</span>
                <div>
                  <div className="lb-gauge-label">{shortText(unitChemistry.label, 4)}</div>
                  <div className="lb-gauge-source">Special teams score</div>
                </div>
              </div>
              <div className="lb-block">
                {metricRows.map(([label, value]) => <BarRow key={label} label={label} value={value} />)}
              </div>
              <div className="lb-block">
                <div className="lb-block-title">Links in this unit</div>
                {selectedLinks.length ? selectedLinks.map((link) => (
                  <div className="lb-linkrow" key={link.key}>
                    <span className="lb-linkrow-names">{linkName(link.slotA)} — {linkName(link.slotB)}</span>
                    <span className={`lb-linkrow-score ${link.tier}`}>{link.score == null ? "—" : `${link.score}%`}</span>
                  </div>
                )) : <div className="lb-note"><span className="lb-note-text">Fill the unit to score its links.</span></div>}
                <div className="lb-kv"><span className="lb-kv-label">Avg bond</span><span className="lb-kv-value">{bondAverage == null ? "—" : `${bondAverage}%`}</span></div>
              </div>
              <div className="lb-block">
                <div className="lb-block-title">Read on this unit</div>
                {freshLinks.length ? <div className="lb-note"><span className="lb-note-mark warn">!</span><span className="lb-note-text">{freshLinks.length} pairing{freshLinks.length > 1 ? "s have" : " has"} no shared history yet.</span></div> : null}
                {(unitChemistry.tips || []).map((tip, index) => (
                  <div className="lb-note" key={`tip-${index}`}><span className="lb-note-mark warn">!</span><span className="lb-note-text">{tip}</span></div>
                ))}
                {invalidMsg ? <div className="lb-note"><span className="lb-note-mark bad">×</span><span className="lb-note-text">{invalidMsg}</span></div> : null}
                {unsaved ? <div className="lb-kv"><span className="lb-kv-label">Status</span><span className="lb-kv-value warn">Unsaved changes</span></div> : null}
              </div>
              {selectedPlayer ? <div className="lb-inspector-player">
                <PlayerHeadshot player={selectedPlayer} size="lg" />
                <div>
                  <div className="lb-inspector-player-name">{selectedPlayer.name}</div>
                  <div className="lb-inspector-player-meta">{selectedPlayer.position} · {selectedPlayer.overall} OVR</div>
                  <div className="lb-inspector-player-meta">{shortRole(selectedPlayer.role) || "—"} · {selectedPlayer.handedness}</div>
                </div>
              </div> : null}
            </div>
          </div>
        </section>
      </div>
    </main>
    <LineBuilderToast toast={toast} onDismiss={() => setToast(null)} onDetails={() => setInspectorOpen(true)} />
  </div>;
}

export function PowerPlay(props) { return <SpecialTeamsLines kind="power_play" {...props} />; }
export function PenaltyKill(props) { return <SpecialTeamsLines kind="penalty_kill" {...props} />; }

export default function EditLines(props) {
  const { screen } = useGameUI();
  if (screen === SCREENS.POWER_PLAY) return <SpecialTeamsLines kind="power_play" {...props} />;
  if (screen === SCREENS.PENALTY_KILL) return <SpecialTeamsLines kind="penalty_kill" {...props} />;
  return <EvenStrengthLines {...props} />;
}