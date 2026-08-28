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
const LINEUP_MODES = { FORWARDS: "forwards", DEFENSE: "defense", GOALIES: "goalies", OVERVIEW: "overview" };
const AUTO_MODES = [["overall", "Best Overall"], ["chemistry", "Best Chemistry"], ["position", "Position Safe"], ["roles", "Balanced Roles"]];
const TABS = ["summary", "fit", "status", "warnings"];

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
function shortText(value, limit = 15) { return String(value || "").trim().split(/\s+/).filter(Boolean).slice(0, limit).join(" "); }
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
  // Prefer availability_status ("Available" / "Out") over injury_status ("HEALTHY"),
  // and never treat a truthy injury_status string alone as injured — healthy players
  // are serialized with injury_status: "HEALTHY".
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
  return ensurePlayerHeadshotFields({
    ...player,
    id: String(player?.id ?? player?.player_id ?? player?._id ?? name),
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
    chemistry_contract_missing: !profile,
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
  if (goalieSlot(slot)) return pos === "G" ? 100 : 0;
  if (pos === slot) return 100;
  if (slot === "C" && pos === "F") return 72;
  if (["LW", "RW"].includes(slot) && pos === "F") return 72;
  if (["LW", "RW"].includes(slot) && isForward(pos)) return 55;
  if (["LD", "RD"].includes(slot) && pos === "D") return 76;
  if (["LD", "RD"].includes(slot) && isDefense(pos)) return 58;
  return 0;
}
function chemistryFitScore(player, slot) {
  return clamp(idealScore(player, slot) * 0.7 + (profileValue(player, "morale", 50) ?? 50) * 0.15 + (profileValue(player, "adaptability", 50) ?? 50) * 0.15);
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
  if (mode === "chemistry") return fit * 0.48 + overall * 0.34 + morale * 0.18;
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
  const ids = new Set(players.map((player) => String(player.id)));
  const rows = [...(Array.isArray(chemReport.lines) ? chemReport.lines : []), ...(Array.isArray(chemReport.pairs) ? chemReport.pairs : [])];
  const match = rows.find((row) => {
    const rowIds = (row.players || []).map((player) => String(player.id || player.player_id || ""));
    return rowIds.length >= 2 && rowIds.every((id) => ids.has(id)) && Math.abs(rowIds.length - players.length) <= 1;
  });
  if (!match) return null;
  const scheme = match.scheme_fit || {};
  return {
    score: clamp(match.chemistry ?? 50),
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
function calculateUnitChemistry(players, unitType = "forward", chemReport = null) {
  const selected = players.filter(Boolean);
  const live = backendChemistry(chemReport, selected);
  if (live) return live;
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
  const score = clamp(avgOverall * 0.34 + (morale ?? 50) * 0.16 + roleBalance * 0.14 + positionFit * 0.22 + linemateFit * 0.14);
  const label = score >= 88 ? "Excellent" : score >= 78 ? "Strong" : score >= 68 ? "Stable" : score >= 58 ? "Uneven" : "Weak";
  const concerns = [];
  const factors = [];
  if (positionFit < 75) concerns.push("Position balance needs work.");
  if (roleBalance < 60) concerns.push("Roles overlap too heavily.");
  if (morale != null && morale < 65) concerns.push("Unit morale is low.");
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
    if (sort === "fatigue") return (a.fatigue ?? Infinity) - (b.fatigue ?? Infinity) || b.overall - a.overall;
    if (sort === "handedness") return String(a.handedness).localeCompare(String(b.handedness)) || b.overall - a.overall;
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
function modeForGroup(group) { return group === "defense" ? LINEUP_MODES.DEFENSE : group === "goalies" ? LINEUP_MODES.GOALIES : LINEUP_MODES.FORWARDS; }
function EditLinesStyles() {
  return <style>{`
.linebuilder-root{--bg:var(--ops-navy,#04101a);--panel:var(--ops-panel,rgba(9,25,38,.94));--panel2:var(--ops-panel-2,rgba(12,35,52,.94));--line:var(--ops-grid,rgba(156,218,236,.14));--line2:var(--ops-grid-2,rgba(115,229,241,.25));--cyan:var(--ops-cyan,#13d8e7);--gold:var(--ops-gold,#e9a83c);--green:var(--ops-success,#52df94);--red:var(--ops-injury,#ff606d);--text:var(--ops-text,#e9f7fb);--muted:var(--ops-text-secondary,#8096a8);--muted2:var(--ops-text-disabled,#607789);width:100%;height:100vh;height:100dvh;overflow:hidden;color:var(--text);background:linear-gradient(180deg,var(--ops-navy,#06111d),var(--ops-navy-deep,#07131f));font-family:var(--font-ops-ui,Inter,system-ui,sans-serif);display:grid;grid-template-columns:88px minmax(0,1fr)}
.linebuilder-root *{box-sizing:border-box}.linebuilder-root button,.linebuilder-root input,.linebuilder-root select{font:inherit}.linebuilder-root button:focus-visible,.linebuilder-root input:focus-visible,.linebuilder-root select:focus-visible,.linebuilder-root [tabindex]:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
.linebuilder-root .lb-sidebar{height:100%;padding:10px 8px;border-right:1px solid var(--line);background:rgba(4,14,24,.92);display:flex;flex-direction:column;gap:10px;overflow:hidden}.linebuilder-root .lb-team-mark{height:46px;flex:0 0 46px;border:1px solid var(--line2);border-radius:var(--radius-hud,4px);display:grid;place-items:center;color:var(--cyan);font-size:12px;font-weight:900;letter-spacing:.09em;background:rgba(19,216,231,.08)}.linebuilder-root .lb-nav{flex:1;min-height:0;display:grid;grid-template-rows:repeat(5,minmax(0,1fr));gap:8px}.linebuilder-root .lb-nav-btn{width:100%;min-height:0;height:100%;padding:6px 4px;border:1px solid transparent;border-radius:var(--radius-hud,4px);color:var(--muted);background:transparent;cursor:pointer;display:grid;place-items:center;align-content:center;gap:3px}.linebuilder-root .lb-nav-btn:hover,.linebuilder-root .lb-nav-btn.active{color:var(--cyan);border-color:var(--line2);background:var(--ops-cyan-soft,rgba(19,216,231,.13));box-shadow:inset 3px 0 0 var(--cyan)}.linebuilder-root .lb-nav-label{font-size: 11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.linebuilder-root .lb-glyph{width:22px;height:22px;display:inline-grid;place-items:center;border:1px solid currentColor;border-radius:var(--radius-ops,2px);font-size: 11px;font-weight:900;line-height:1}
.linebuilder-root .lb-shell{min-width:0;min-height:0;height:100%;padding:10px;display:grid;grid-template-rows:56px 34px minmax(0,1fr);gap:8px;overflow:hidden}.linebuilder-root .lb-header{height:56px;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:var(--panel);box-shadow:none;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 10px;position:relative;z-index:20}.linebuilder-root .lb-title-group{min-width:0;display:flex;align-items:center;gap:10px}.linebuilder-root .lb-logo{width:40px;height:40px;flex:0 0 40px;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:rgba(0,0,0,.18);display:grid;place-items:center;overflow:hidden;color:var(--cyan);font-size:11px;font-weight:900}.linebuilder-root .lb-logo img{width:32px;height:32px;object-fit:contain}.linebuilder-root .lb-title-copy{min-width:0;display:flex;align-items:baseline;gap:9px}.linebuilder-root .lb-title{margin:0;font-size:clamp(18px,2vw,24px);line-height:1;letter-spacing:.04em;font-weight:900;text-transform:uppercase;white-space:nowrap}.linebuilder-root .lb-roster-count{color:var(--muted);font-size: 11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}.linebuilder-root .lb-actions{display:flex;align-items:center;justify-content:flex-end;gap:6px}.linebuilder-root .lb-btn,.linebuilder-root .lb-icon{min-height:36px;border:1px solid var(--line);border-radius:var(--radius-control,6px);color:var(--text);background:rgba(255,255,255,.02);cursor:pointer;font-size: 11px;font-weight:850;letter-spacing:.06em;text-transform:uppercase}.linebuilder-root .lb-btn{padding:0 12px;display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap}.linebuilder-root .lb-icon{width:36px;padding:0;display:inline-grid;place-items:center}.linebuilder-root .lb-btn:hover:not(:disabled),.linebuilder-root .lb-icon:hover:not(:disabled){border-color:var(--line2);background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-btn:disabled,.linebuilder-root .lb-icon:disabled{opacity:.38;cursor:not-allowed}.linebuilder-root .lb-btn.primary{color:var(--ops-navy,#04101a);border-color:var(--cyan);background:var(--cyan);box-shadow:none}.linebuilder-root .lb-btn.subtle{color:var(--muted);border-color:var(--line);background:transparent;font-weight:750}.linebuilder-root .lb-btn.subtle:hover:not(:disabled){color:var(--text);border-color:var(--line2);background:rgba(255,255,255,.03)}.linebuilder-root .lb-btn.subtle.danger{color:#c98989;border-color:rgba(255,96,109,.28)}.linebuilder-root .lb-btn.subtle.danger:hover:not(:disabled){color:#ffd5d5;border-color:rgba(255,96,109,.45);background:rgba(255,96,109,.08)}.linebuilder-root .lb-btn.danger{color:#ffd5d5;border-color:rgba(255,96,109,.45);background:rgba(255,96,109,.08)}.linebuilder-root .lb-save-error{color:var(--red);font-size: 11px;font-weight:800}
.linebuilder-root .lb-status{height:34px;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:rgba(0,0,0,.16);padding:0 6px;display:flex;align-items:center;gap:0;overflow:hidden}.linebuilder-root .lb-pill{height:100%;padding:0 9px;border-right:1px solid var(--line);display:flex;align-items:center;gap:6px;white-space:nowrap}.linebuilder-root .lb-pill:last-child{border-right:0}.linebuilder-root .lb-pill-label{color:var(--muted2);font-size: 11px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}.linebuilder-root .lb-pill-value{font-size:12px;font-weight:900;font-variant-numeric:tabular-nums}
.linebuilder-root .lb-workspace{min-width:0;min-height:0;height:100%;display:grid;grid-template-columns:250px minmax(0,1fr) 290px;gap:8px;overflow:hidden}.linebuilder-root .lb-region{min-width:0;min-height:0;height:100%;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:var(--panel);box-shadow:none;overflow:hidden;display:grid;grid-template-rows:40px minmax(0,1fr)}.linebuilder-root .lb-region-head{height:40px;padding:0 10px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px;background:rgba(0,0,0,.14)}.linebuilder-root .lb-region-title{margin:0;font-size: 11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.linebuilder-root .lb-region-note{color:var(--muted);font-size: 11px;font-weight:800;white-space:nowrap}
.linebuilder-root .lb-pool{min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr);gap:6px;padding:8px;overflow:hidden}.linebuilder-root .lb-pool-toolbar{display:block}.linebuilder-root .lb-search-wrap{position:relative}.linebuilder-root .lb-search{width:100%;height:32px;padding:0 28px 0 9px;border:1px solid var(--line);border-radius:var(--radius-ops,2px);color:var(--text);background:rgba(0,0,0,.18);outline:none;font-size:11px}.linebuilder-root .lb-search::placeholder{color:var(--muted2)}.linebuilder-root .lb-clear{position:absolute;top:2px;right:2px;width:28px;height:28px;border:0;border-radius:var(--radius-ops,2px);color:var(--muted);background:transparent;cursor:pointer}.linebuilder-root .lb-clear:hover{color:var(--cyan);background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-player-list{min-height:0;height:100%;display:flex;flex-direction:column;gap:0;overflow-x:hidden;overflow-y:auto;padding-right:2px;scrollbar-width:thin;scrollbar-color:rgba(19,216,231,.35) rgba(0,0,0,.22)}.linebuilder-root .lb-player{flex:0 0 auto;min-height:38px;height:38px;padding:0 8px 0 6px;border:0;border-bottom:1px solid var(--line);border-left:3px solid transparent;border-radius:0;background:transparent;display:grid;grid-template-columns:28px minmax(0,1fr) 32px;gap:7px;align-items:center;cursor:grab}.linebuilder-root .lb-player:nth-child(even){background:rgba(255,255,255,.012)}.linebuilder-root .lb-player:hover{border-color:var(--line);background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-player.selected{border-left-color:var(--cyan);background:var(--ops-table-sel,rgba(19,216,231,.13));box-shadow:none}.linebuilder-root .lb-player.assigned{border-left-color:rgba(128,150,168,.55);background:rgba(0,0,0,.12)}.linebuilder-root .lb-player.assigned .lb-player-name{color:#b7c9d3}.linebuilder-root .lb-player.disabled{cursor:not-allowed;opacity:.4}.linebuilder-root .lb-player.locked{border-left-color:var(--gold)}.linebuilder-root .lb-player .player-headshot{width:26px!important;height:26px!important}.linebuilder-root .lb-player-copy{min-width:0}.linebuilder-root .lb-player-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:800;letter-spacing:.01em;line-height:1.1}.linebuilder-root .lb-player-meta{margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted2);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.linebuilder-root .lb-player-side{display:grid;justify-items:end;align-content:center;gap:3px}.linebuilder-root .lb-player-ovr{min-width:28px;width:100%;text-align:right;color:var(--cyan);font-size:11px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1}.linebuilder-root .lb-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green)}.linebuilder-root .lb-dot.warn{background:var(--gold)}.linebuilder-root .lb-dot.bad{background:var(--red)}
.linebuilder-root .lb-board-region{grid-template-rows:34px minmax(0,1fr)}.linebuilder-root .lb-modebar{height:34px;padding:0 8px;border-bottom:1px solid var(--line);display:flex;align-items:stretch;justify-content:flex-start;gap:0;background:rgba(0,0,0,.16)}.linebuilder-root .lb-mode{min-width:0;flex:0 0 auto;padding:0 14px;border:0;border-bottom:2px solid transparent;border-radius:0;color:var(--muted);background:transparent;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;font-size: 11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}.linebuilder-root .lb-mode:hover{color:var(--text)}.linebuilder-root .lb-mode.active{color:var(--cyan);border-bottom-color:var(--cyan);box-shadow:none}.linebuilder-root .lb-mobile-tools{display:none;margin-left:auto;gap:4px}.linebuilder-root .lb-board{min-width:0;min-height:0;height:100%;padding:0;display:grid;gap:0;overflow:hidden;align-content:stretch}.linebuilder-root .lb-board.forwards{grid-template-rows:24px repeat(4,minmax(0,1fr))}.linebuilder-root .lb-board.defense{grid-template-rows:24px repeat(3,minmax(0,1fr))}.linebuilder-root .lb-board.goalies{grid-template-rows:24px minmax(0,1fr)}.linebuilder-root .lb-board.special.pp{grid-template-rows:24px repeat(2,minmax(0,1fr))}.linebuilder-root .lb-board.special.pk{grid-template-rows:24px repeat(3,minmax(0,1fr))}.linebuilder-root .lb-col-headers{display:grid;grid-template-columns:52px minmax(0,1fr) 68px;align-items:center;padding:0 8px;margin:0;border-bottom:1px solid var(--line2);height:24px;background:rgba(0,0,0,.18)}.linebuilder-root .lb-col-headers-spacer{min-height:1px}.linebuilder-root .lb-col-headers-slots{display:grid;gap:4px;padding:0 4px}.linebuilder-root .lb-col-headers-slots.forwards{grid-template-columns:repeat(3,minmax(0,1fr))}.linebuilder-root .lb-col-headers-slots.defense,.linebuilder-root .lb-col-headers-slots.goalies{grid-template-columns:repeat(2,minmax(0,1fr))}.linebuilder-root .lb-col-headers-slots.goalies.third{grid-template-columns:repeat(3,minmax(0,1fr))}.linebuilder-root .lb-col-headers-slots.pp{grid-template-columns:repeat(5,minmax(0,1fr))}.linebuilder-root .lb-col-headers-slots.pk{grid-template-columns:repeat(4,minmax(0,1fr))}.linebuilder-root .lb-col-header{text-align:center;color:var(--muted);font-size: 11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}.linebuilder-root .lb-col-headers-chem{text-align:center;color:var(--muted);font-size: 11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase}
.linebuilder-root .lb-unit{min-width:0;min-height:0;height:100%;margin:0;border:0;border-bottom:1px solid var(--line);border-radius:0;background:transparent;display:grid;grid-template-columns:52px minmax(0,1fr) 68px;align-items:stretch;position:relative;overflow:visible}.linebuilder-root .lb-unit.stripe{background:rgba(255,255,255,.018)}.linebuilder-root .lb-unit.selected{background:var(--ops-table-sel,rgba(19,216,231,.13));box-shadow:inset 3px 0 0 var(--cyan)}.linebuilder-root .lb-unit.primary{box-shadow:inset 3px 0 0 var(--gold)}.linebuilder-root .lb-unit.primary.selected{box-shadow:inset 3px 0 0 var(--cyan)}.linebuilder-root .lb-unit-label{padding:6px 4px;border-right:1px solid var(--line);display:flex;flex-direction:column;justify-content:center;align-items:center;gap:2px;align-self:stretch;background:rgba(0,0,0,.1)}.linebuilder-root .lb-unit-kicker{color:var(--muted2);font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.linebuilder-root .lb-unit-name{font-size:16px;font-weight:950;letter-spacing:-.02em;line-height:1}.linebuilder-root .lb-unit-locks{color:var(--gold);font-size:11px;font-weight:850}.linebuilder-root .lb-slots{min-width:0;min-height:0;height:100%;padding:4px 6px;display:grid;gap:4px;align-items:stretch}.linebuilder-root .lb-slots.forwards{grid-template-columns:repeat(3,minmax(0,1fr))}.linebuilder-root .lb-slots.defense,.linebuilder-root .lb-slots.goalies{grid-template-columns:repeat(2,minmax(0,1fr))}.linebuilder-root .lb-slots.goalies.third{grid-template-columns:repeat(3,minmax(0,1fr))}.linebuilder-root .lb-slots.pp{grid-template-columns:repeat(5,minmax(0,1fr))}.linebuilder-root .lb-slots.pk{grid-template-columns:repeat(4,minmax(0,1fr))}
.linebuilder-root .lb-slot{min-width:0;min-height:0;height:100%;padding:4px 6px;border:1px solid var(--line);border-top:2px solid rgba(128,150,168,.35);border-radius:var(--radius-ops,2px);background:rgba(0,0,0,.14);display:grid;grid-template-columns:28px minmax(0,1fr) 30px;gap:6px;align-items:center;cursor:pointer;position:relative;overflow:hidden;box-shadow:none}.linebuilder-root .lb-slot.occupied{border-style:solid;border-color:var(--line);border-top-color:var(--cyan);background:rgba(0,0,0,.2)}.linebuilder-root .lb-slot.selected{border-color:var(--line2);background:var(--ops-table-sel,rgba(19,216,231,.13));box-shadow:inset 3px 0 0 var(--cyan);padding-right:44px}.linebuilder-root .lb-slot.valid{border-color:var(--line2);background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-slot.swap{border-color:var(--gold);background:var(--ops-gold-soft,rgba(233,168,60,.14))}.linebuilder-root .lb-slot.invalid{border-color:var(--red);background:rgba(255,96,109,.08)}.linebuilder-root .lb-slot.locked{border-top-color:var(--gold)}.linebuilder-root .lb-slot.starter{border-left:3px solid var(--gold)}.linebuilder-root .lb-slot.backup{border-left:3px solid var(--cyan)}.linebuilder-root .lb-slot-shot{display:grid;place-items:center}.linebuilder-root .lb-slot .player-headshot{width:26px!important;height:26px!important}.linebuilder-root .lb-slot-pos{width:28px;height:28px;border:1px dashed var(--line2);border-radius:var(--radius-ops,2px);display:grid;place-items:center;color:var(--muted);font-size:11px;font-weight:950;letter-spacing:.06em;text-transform:uppercase;background:rgba(0,0,0,.12)}.linebuilder-root .lb-slot-copy{min-width:0;display:flex;flex-direction:column;justify-content:center;gap:2px}.linebuilder-root .lb-slot-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;font-weight:800;letter-spacing:.01em;line-height:1.05}.linebuilder-root .lb-slot-meta{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted2);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.linebuilder-root .lb-slot-empty{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size: 11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}.linebuilder-root .lb-slot-side{display:grid;justify-items:end;align-content:center;gap:2px}.linebuilder-root .lb-slot-ovr{text-align:right;color:var(--cyan);font-size:12px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1}.linebuilder-root .lb-drop{color:var(--red);font-size:11px;font-weight:900}.linebuilder-root .lb-markers{display:flex;justify-content:flex-end;gap:3px}.linebuilder-root .lb-marker{width:14px;height:14px;border:1px solid currentColor;border-radius:var(--radius-ops,2px);display:grid;place-items:center;color:var(--muted);font-size:11px;font-weight:900}.linebuilder-root .lb-marker.lock{color:var(--gold)}.linebuilder-root .lb-marker.bad{color:var(--red)}.linebuilder-root .lb-slot-actions{position:absolute;top:50%;right:4px;transform:translateY(-50%);z-index:4;display:flex;gap:3px}.linebuilder-root .lb-mini{width:20px;height:20px;border:1px solid var(--line);border-radius:var(--radius-ops,2px);color:var(--muted);background:rgba(0,0,0,.22);cursor:pointer;display:grid;place-items:center;font-size: 11px;font-weight:900}.linebuilder-root .lb-mini:hover{color:var(--cyan);border-color:var(--line2)}.linebuilder-root .lb-mini.danger:hover{color:var(--red);border-color:rgba(255,96,109,.45)}
.linebuilder-root .lb-unit-status{padding:6px 6px;border-left:1px solid var(--line);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;background:rgba(0,0,0,.08)}.linebuilder-root .lb-unit-chem{color:var(--cyan);font-size:18px;line-height:1;font-weight:950;font-variant-numeric:tabular-nums}.linebuilder-root .lb-unit-chem-unit{margin-left:1px;font-size: 11px;font-weight:850;opacity:.75}.linebuilder-root .lb-unit-tools{display:flex;align-items:center;justify-content:center;gap:4px;min-height:22px}.linebuilder-root .lb-warning,.linebuilder-root .lb-menu-btn{min-width:24px;height:22px;padding:0 6px;border:1px solid var(--line);border-radius:var(--radius-ops,2px);color:var(--muted);background:transparent;cursor:pointer;font-size: 11px;font-weight:900}.linebuilder-root .lb-warning.has{color:var(--gold);border-color:rgba(233,168,60,.45);background:var(--ops-gold-soft,rgba(233,168,60,.14))}.linebuilder-root .lb-warning-count{font-size:11px;font-weight:950;font-variant-numeric:tabular-nums}.linebuilder-root .lb-unit-menu{position:absolute;top:36px;right:6px;z-index:30;width:158px;padding:5px;border:1px solid var(--line2);border-radius:var(--radius-hud,4px);background:var(--panel);box-shadow:var(--depth-overlay,0 24px 70px rgba(0,0,0,.42));display:grid;gap:3px}.linebuilder-root .lb-menu-action{min-height:30px;padding:0 8px;border:0;border-radius:var(--radius-ops,2px);color:var(--text);background:transparent;cursor:pointer;text-align:left;font-size: 11px;font-weight:750}.linebuilder-root .lb-menu-action:hover:not(:disabled){background:var(--ops-cyan-soft,rgba(19,216,231,.13));color:var(--cyan)}.linebuilder-root .lb-menu-action:disabled{opacity:.35}
.linebuilder-root .lb-overview{height:100%;padding:6px;display:grid;grid-template-rows:repeat(8,minmax(0,1fr));gap:0;overflow:hidden;border-top:1px solid var(--line)}.linebuilder-root .lb-overview-row{min-width:0;padding:4px 6px;border:0;border-bottom:1px solid var(--line);border-radius:0;color:var(--text);background:transparent;cursor:pointer;display:grid;grid-template-columns:68px minmax(0,1fr) 72px;gap:7px;align-items:center}.linebuilder-root .lb-overview-row:hover{background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-overview-name{color:var(--muted);font-size: 11px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}.linebuilder-root .lb-overview-players{min-width:0;display:flex;gap:5px;overflow:hidden}.linebuilder-root .lb-overview-player{min-width:0;flex:1;display:flex;align-items:center;gap:4px}.linebuilder-root .lb-overview-player .player-headshot{width:22px!important;height:22px!important;flex:0 0 22px}.linebuilder-root .lb-overview-player span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size: 11px;font-weight:750}.linebuilder-root .lb-overview-metrics{text-align:right}.linebuilder-root .lb-overview-score{display:block;color:var(--cyan);font-size:12px;font-weight:950}.linebuilder-root .lb-overview-warning{color:var(--gold);font-size:11px}
.linebuilder-root .lb-inspector{min-height:0;display:grid;grid-template-rows:34px minmax(0,1fr);overflow:hidden}.linebuilder-root .lb-tabs{padding:0 6px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;background:rgba(0,0,0,.12)}.linebuilder-root .lb-tab{border:0;border-bottom:2px solid transparent;border-radius:0;color:var(--muted);background:transparent;cursor:pointer;font-size: 11px;font-weight:850;text-transform:uppercase;letter-spacing:.06em;min-height:32px}.linebuilder-root .lb-tab.active{color:var(--cyan);border-bottom-color:var(--cyan);background:transparent}.linebuilder-root .lb-inspector-body{min-height:0;padding:10px;overflow:auto;display:flex;flex-direction:column;gap:8px;justify-content:flex-start;scrollbar-width:thin;scrollbar-color:rgba(19,216,231,.35) rgba(0,0,0,.22)}.linebuilder-root .lb-inspector-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;font-weight:900}.linebuilder-root .lb-inspector-sub{margin-top:3px;color:var(--muted);font-size: 11px;text-transform:uppercase;letter-spacing:.08em}.linebuilder-root .lb-inspector-player{padding:8px;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:rgba(0,0,0,.12);display:grid;grid-template-columns:52px minmax(0,1fr);gap:8px;align-items:center}.linebuilder-root .lb-inspector-player .player-headshot{width:48px!important;height:48px!important}.linebuilder-root .lb-inspector-player-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;font-weight:900}.linebuilder-root .lb-inspector-player-meta{margin-top:4px;color:var(--muted2);font-size: 11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.linebuilder-root .lb-gauge{padding:8px;border:1px solid var(--line);border-radius:var(--radius-hud,4px);background:rgba(0,0,0,.12);display:grid;grid-template-columns:56px minmax(0,1fr);gap:8px;align-items:center}.linebuilder-root .lb-gauge-score{color:var(--cyan);font-size:26px;line-height:1;font-weight:950}.linebuilder-root .lb-gauge-label{font-size:11px;font-weight:850;text-transform:uppercase;letter-spacing:.04em}.linebuilder-root .lb-gauge-source{margin-top:3px;color:var(--muted2);font-size:11px;text-transform:uppercase;letter-spacing:.06em}.linebuilder-root .lb-list{min-height:0;display:grid;align-content:start;gap:0;overflow:visible;border:1px solid var(--line);border-radius:var(--radius-hud,4px)}.linebuilder-root .lb-row{min-width:0;min-height:34px;padding:6px 8px;border:0;border-bottom:1px solid var(--line);border-radius:0;background:transparent;display:flex;align-items:center;justify-content:space-between;gap:10px}.linebuilder-root .lb-row:last-child{border-bottom:0}.linebuilder-root .lb-row-label{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size: 11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}.linebuilder-root .lb-row-value{max-width:58%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:850;font-variant-numeric:tabular-nums}.linebuilder-root .lb-row-value.ok{color:var(--green)}.linebuilder-root .lb-row-value.warn{color:var(--gold)}.linebuilder-root .lb-progress{width:84px;height:4px;border:0;border-radius:var(--radius-ops,2px);overflow:hidden;appearance:none}.linebuilder-root .lb-progress::-webkit-progress-bar{background:rgba(255,255,255,.06)}.linebuilder-root .lb-progress::-webkit-progress-value{background:var(--cyan)}.linebuilder-root .lb-progress::-moz-progress-bar{background:var(--cyan)}.linebuilder-root .lb-warning-row{justify-content:flex-start;color:#efc986;font-size: 11px}.linebuilder-root .lb-inspector-actions{margin-top:4px;display:grid;grid-template-columns:1fr 1fr;gap:6px}.linebuilder-root .lb-compare{min-height:40px}.linebuilder-root .lb-compare-copy{min-width:0}.linebuilder-root .lb-compare-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;font-weight:900}.linebuilder-root .lb-compare-meta{margin-top:3px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.linebuilder-root .lb-empty,.linebuilder-root .lb-loading{height:100%;display:grid;place-items:center;padding:18px;color:var(--muted);text-align:center;font-size: 11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.linebuilder-root .lb-loading::before{content:"SYNC";display:block;margin-bottom:6px;color:var(--cyan);font-size: 11px;letter-spacing:.14em}.linebuilder-root .lb-skeleton{width:min(220px,80%);height:8px;border-radius:var(--radius-ops,2px);background:linear-gradient(90deg,rgba(255,255,255,.03),rgba(19,216,231,.12),rgba(255,255,255,.03));background-size:200% 100%;animation:lb-shimmer 1.2s linear infinite}@keyframes lb-shimmer{from{background-position:100% 0}to{background-position:-100% 0}}
.linebuilder-root .lb-popover{position:absolute;top:56px;right:86px;z-index:50;width:300px;max-height:min(440px,calc(100dvh - 90px));padding:8px;border:1px solid var(--line2);border-radius:var(--radius-hud,4px);background:var(--panel);box-shadow:var(--depth-overlay,0 24px 70px rgba(0,0,0,.42));display:grid;gap:7px;overflow:hidden}.linebuilder-root .lb-popover-title{margin:0;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.06em}.linebuilder-root .lb-auto-options{display:grid;grid-template-columns:1fr 1fr;gap:5px}.linebuilder-root .lb-auto-option{min-height:32px;padding:0 7px;border:1px solid var(--line);border-radius:var(--radius-ops,2px);color:var(--muted);background:rgba(0,0,0,.12);cursor:pointer;font-size: 11px;font-weight:800;text-transform:uppercase;letter-spacing:.04em}.linebuilder-root .lb-auto-option.active{color:var(--cyan);border-color:var(--line2);background:var(--ops-cyan-soft,rgba(19,216,231,.13))}.linebuilder-root .lb-preview{max-height:210px;display:grid;align-content:start;gap:0;border:1px solid var(--line);border-radius:var(--radius-hud,4px);overflow-y:auto}.linebuilder-root .lb-preview-row{padding:5px 6px;border:0;border-bottom:1px solid var(--line);border-radius:0;display:grid;gap:2px}.linebuilder-root .lb-preview-label{color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.06em}.linebuilder-root .lb-preview-change{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size: 11px}.linebuilder-root .lb-popover-actions{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.linebuilder-root .lb-toast{position:fixed;right:18px;bottom:18px;z-index:100;min-width:230px;max-width:340px;min-height:42px;padding:8px 10px;border:1px solid var(--line2);border-left:3px solid var(--cyan);border-radius:var(--radius-hud,4px);background:var(--panel);box-shadow:var(--depth-overlay,0 24px 70px rgba(0,0,0,.42));display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}.linebuilder-root .lb-toast.success{border-left-color:var(--green)}.linebuilder-root .lb-toast.warning{border-left-color:var(--gold)}.linebuilder-root .lb-toast.error{border-left-color:var(--red)}.linebuilder-root .lb-toast-message{font-size: 11px;font-weight:800;letter-spacing:.02em}.linebuilder-root .lb-toast-actions{display:flex;gap:4px}.linebuilder-root .lb-drawer-close{display:none}.linebuilder-root .lb-live{position:fixed;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
@media(max-width:1280px){.linebuilder-root .lb-workspace{grid-template-columns:220px minmax(0,1fr) 260px}.linebuilder-root .lb-unit,.linebuilder-root .lb-col-headers{grid-template-columns:44px minmax(0,1fr) 60px}}
@media(max-width:1080px){.linebuilder-root{grid-template-columns:1fr;grid-template-rows:58px minmax(0,1fr)}.linebuilder-root .lb-sidebar{height:58px;padding:6px 8px;border-right:0;border-bottom:1px solid var(--line);flex-direction:row;align-items:center}.linebuilder-root .lb-team-mark{width:52px;height:44px;flex:0 0 52px}.linebuilder-root .lb-nav{flex:1;display:flex;grid-template-rows:none;gap:5px}.linebuilder-root .lb-nav-btn{min-width:58px;min-height:44px;width:auto;height:auto;padding:3px 8px;display:flex}.linebuilder-root .lb-shell{padding:8px;grid-template-rows:56px 34px minmax(0,1fr);gap:7px}.linebuilder-root .lb-header{height:56px}.linebuilder-root .lb-workspace{grid-template-columns:220px minmax(0,1fr)}.linebuilder-root .lb-inspector-region{position:fixed;top:66px;right:8px;bottom:8px;width:300px;height:auto;z-index:70;transform:translateX(calc(100% + 18px));transition:transform .18s;box-shadow:var(--depth-overlay,0 24px 70px rgba(0,0,0,.42))}.linebuilder-root .lb-inspector-region.open{transform:translateX(0)}.linebuilder-root .lb-drawer-close{display:inline-grid}.linebuilder-root .lb-mobile-tools{display:flex}}
@media(max-width:900px){.linebuilder-root .lb-workspace{grid-template-columns:1fr}.linebuilder-root .lb-pool-region{position:fixed;top:66px;left:8px;bottom:8px;width:270px;height:auto;z-index:72;transform:translateX(calc(-100% - 18px));transition:transform .18s;box-shadow:var(--depth-overlay,0 24px 70px rgba(0,0,0,.42))}.linebuilder-root .lb-pool-region.open{transform:translateX(0)}.linebuilder-root .lb-roster-count{display:none}}
/* Inspector text must fit the 200px column: tabs tighten their tracking and
   verdict values wrap instead of being cut mid-word. */
.linebuilder-root .lb-tabs{padding:0 2px}
.linebuilder-root .lb-tab{letter-spacing:.02em;padding:0 2px}
.linebuilder-root .lb-row-value{white-space:normal;text-align:right;line-height:1.3}
.linebuilder-root .lb-slot.selected{padding-right:34px}
.linebuilder-root .lb-slot-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Cyan is reserved for selection and legal placement. Merely being occupied is
   the normal state of a lineup board, so it uses a neutral rule. */
.linebuilder-root .lb-slot.occupied{border-top-color:rgba(128,150,168,.5)}
.linebuilder-root .lb-slot.occupied.selected{border-top-color:var(--cyan)}
/* Department signature: the bench board. A dashed tactical line runs behind
   each unit so its slots read as one coached combination rather than three
   unrelated cells. */
.linebuilder-root .lb-slots{position:relative}
.linebuilder-root .lb-slots::before{content:"";position:absolute;left:10px;right:10px;top:50%;height:0;border-top:1px dashed rgba(128,150,168,.28);pointer-events:none;z-index:0}
.linebuilder-root .lb-unit.selected .lb-slots::before{border-top-color:rgba(19,216,231,.45)}
.linebuilder-root .lb-unit.primary .lb-slots::before{border-top-color:rgba(233,168,60,.4)}
.linebuilder-root .lb-slot{z-index:1}
/* Unit number is stencilled like a bench nameplate. */
.linebuilder-root .lb-unit-name{font-variant-numeric:tabular-nums}
/* Bench hierarchy: saving the lineup is the only filled control on the bar.
   Clearing is destructive and stays a quiet text action until hovered. */
.linebuilder-root .lb-btn.subtle.danger{background:transparent;border-color:transparent;color:rgba(190,150,152,.85)}
.linebuilder-root .lb-btn.subtle.danger:hover:not(:disabled){background:rgba(255,96,109,.1);border-color:rgba(255,96,109,.45);color:#ffd5d5}
/* A placed player settles into the slot so the affected line is obvious. */
.linebuilder-root .lb-slot.occupied .lb-slot-name{animation:lb-slot-seat var(--motion-workspace,180ms) var(--ease-out,cubic-bezier(.2,.7,.3,1)) both}
@keyframes lb-slot-seat{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.linebuilder-root .lb-slot.occupied .lb-slot-name{animation:none}}
@media(max-width:680px){.linebuilder-root .lb-team-mark{display:none}.linebuilder-root .lb-nav-btn{min-width:0;flex:1;padding:3px 4px}.linebuilder-root .lb-nav-label{display:none}.linebuilder-root .lb-title{font-size:17px}.linebuilder-root .lb-logo{width:34px;height:34px;flex-basis:34px}.linebuilder-root .lb-actions{gap:3px}.linebuilder-root .lb-actions .lb-btn{min-height:32px;padding:0 7px;font-size: 11px}.linebuilder-root .lb-actions .lb-icon{width:32px;min-height:32px}.linebuilder-root .lb-pill{padding:0 6px}.linebuilder-root .lb-pill-label{display:none}.linebuilder-root .lb-mode{padding:0 8px;font-size: 11px}.linebuilder-root .lb-unit,.linebuilder-root .lb-col-headers{grid-template-columns:36px minmax(0,1fr) 48px}.linebuilder-root .lb-unit-name{font-size:13px}.linebuilder-root .lb-slot{grid-template-columns:22px minmax(0,1fr);gap:4px;padding:3px 4px}.linebuilder-root .lb-slot-pos{width:22px;height:22px}.linebuilder-root .lb-slot .player-headshot,.linebuilder-root .lb-slot-ovr,.linebuilder-root .lb-markers{display:none}.linebuilder-root .lb-slot-name{font-size:11px}.linebuilder-root .lb-slot-meta{font-size:11px}.linebuilder-root .lb-unit-chem{font-size:13px}.linebuilder-root .lb-col-headers-chem{display:none}}
`}</style>;
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
function LineBuilderHeader({ team, rosterCount, title = "Line Builder", canUndo, canRedo, onUndo, onRedo, onClear, onReset, onAutoBuild, onSave, saving, unsaved, saveError, disabled, showHistory = true, showAutoBuild = true, resetLabel = "Reset", saveLabel }) {
  return <header className="lb-header">
    <div className="lb-title-group"><div className="lb-logo">{team.logo ? <img src={team.logo} alt={`${team.abbreviation} logo`} /> : team.abbreviation}</div><div className="lb-title-copy"><h1 className="lb-title">{title}</h1><span className="lb-roster-count">{rosterCount} players</span></div></div>
    <div className="lb-actions">
      {showHistory ? <>
        <button type="button" className="lb-icon" onClick={onUndo} disabled={!canUndo} aria-label="Undo lineup change" title="Undo">↶</button>
        <button type="button" className="lb-icon" onClick={onRedo} disabled={!canRedo} aria-label="Redo lineup change" title="Redo">↷</button>
      </> : null}
      <button type="button" className="lb-btn subtle danger" onClick={onClear}>Clear</button>
      <button type="button" className="lb-btn subtle" onClick={onReset}>{resetLabel}</button>
      {showAutoBuild ? <button type="button" className="lb-btn" onClick={onAutoBuild} disabled={disabled}>Auto Build</button> : null}
      {saveError ? <span className="lb-save-error">Save failed</span> : null}
      <button type="button" className="lb-btn primary" onClick={onSave} disabled={saving || disabled}>{saving ? "Saving" : saveLabel || (unsaved ? "Save Changes" : "Saved")}</button>
    </div>
  </header>;
}
function LineStatusStrip({ metrics }) {
  return <div className="lb-status" aria-label="Lineup status">{metrics.map((metric) => metric.value == null ? null : <div className="lb-pill" key={metric.label} title={metric.title || undefined}><span className="lb-pill-label">{metric.label}</span><span className="lb-pill-value">{metric.value}</span></div>)}</div>;
}
function PlayerPool({ players, assignedSet, lockedSet, selectedPlayerId, search, setSearch, focusSlot, onPlayerSelect, onDragStart, open, onClose }) {
  return <section className={`lb-region lb-pool-region ${open ? "open" : ""}`}>
    <div className="lb-region-head"><h2 className="lb-region-title">Player Pool</h2><span className="lb-region-note">{players.length}</span><button type="button" className="lb-icon lb-drawer-close" onClick={onClose} aria-label="Close player pool" title="Close">×</button></div>
    <div className="lb-pool">
      <div className="lb-pool-toolbar">
        <div className="lb-search-wrap"><input className="lb-search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search players" aria-label="Search players" />{search ? <button type="button" className="lb-clear" onClick={() => setSearch("")} aria-label="Clear search" title="Clear search">×</button> : null}</div>
      </div>
      <div className="lb-player-list">{players.length ? players.map((player) => {
        const assigned = assignedSet.has(player.id);
        const locked = lockedSet.has(player.id);
        const disabled = !player.availability?.placeable || locked;
        const fit = focusSlot ? chemistryFitScore(player, focusSlot) : null;
        const status = player.availability?.key === "active" ? null : player.availability?.label;
        const meta = [player.position, fit != null ? `${fit}%` : null, status, assigned ? "In lines" : null].filter(Boolean).join(" · ");
        return <div key={player.id} className={`lb-player ${assigned ? "assigned" : ""} ${selectedPlayerId === player.id ? "selected" : ""} ${disabled ? "disabled" : ""} ${locked ? "locked" : ""}`} draggable={!disabled} onDragStart={(event) => onDragStart(event, player.id)} onClick={() => onPlayerSelect(player.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onPlayerSelect(player.id); } }} role="button" tabIndex={0} aria-label={`${player.name}, ${player.position}, ${player.overall} overall`} title={disabled ? player.availability?.reason || "Player cannot be moved." : `${player.name} · ${player.position} · ${player.overall} overall`}>
          <PlayerHeadshot player={player} size="md" /><div className="lb-player-copy"><div className="lb-player-name">{player.name}</div><div className="lb-player-meta">{meta}</div></div><div className="lb-player-side"><div className="lb-player-ovr">{player.overall}</div><span className={`lb-dot ${player.availability?.key === "active" ? "" : player.availability?.placeable ? "warn" : "bad"}`} /></div>
        </div>;
      }) : <div className="lb-empty">No matching players</div>}</div>
    </div>
  </section>;
}
function CompactPlayerSlot({ descriptor, player, locked, selected, targetState, onSelect, onDrop, onDragStart, onRemove, onToggleLock, onKeyDown, registerRef }) {
  const marker = player?.availability?.key === "injured" ? "MED" : player?.scratched ? "SCR" : null;
  const classes = ["lb-slot", player ? "occupied" : "", selected ? "selected" : "", locked ? "locked" : "", descriptor.slot === "Starter" ? "starter" : "", descriptor.slot === "Backup" ? "backup" : "", targetState].filter(Boolean).join(" ");
  return <div ref={(node) => registerRef(descriptor.key, node)} className={classes} role="button" tabIndex={0} aria-label={player ? `${descriptor.slot}, ${player.name}, ${player.overall} overall` : `${descriptor.slot}, empty`} title={player ? shortText(getOverallTooltip(player)) : `Add ${descriptor.slot}`} onClick={() => onSelect(descriptor)} onKeyDown={(event) => onKeyDown(event, descriptor)} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }} onDrop={(event) => onDrop(event, descriptor)}>
    {player ? <>
      <div className={`lb-slot-shot ${player ? "is-nameplate" : ""}`} draggable={!locked && player.availability?.placeable} onDragStart={(event) => { event.stopPropagation(); onDragStart(event, player.id); }}><PlayerHeadshot player={player} size="sm" /></div>
      <div className="lb-slot-copy"><div className="lb-slot-name" title={player.name}>{player.name}</div><div className="lb-slot-meta">{slotMeta(player)}</div></div>
      <div className="lb-slot-side"><div className="lb-slot-ovr">{player.overall}</div>{player.overall_drop > 0 ? <div className="lb-drop">-{player.overall_drop}</div> : null}<div className="lb-markers">{locked ? <span className="lb-marker lock">L</span> : null}{marker ? <span className="lb-marker bad">{marker}</span> : null}</div></div>
      {selected ? <div className="lb-slot-actions"><button type="button" className="lb-mini" onClick={(event) => { event.stopPropagation(); onToggleLock(descriptor); }} aria-label={locked ? "Unlock slot" : "Lock slot"} title={locked ? "Unlock slot" : "Lock slot"}>{locked ? "U" : "L"}</button><button type="button" className="lb-mini danger" onClick={(event) => { event.stopPropagation(); onRemove(descriptor); }} disabled={locked} aria-label="Remove player" title="Remove player">×</button></div> : null}
    </> : <><span className="lb-slot-pos">{descriptor.slot}</span><span className="lb-slot-empty">Empty</span></>}
  </div>;
}
function UnitMenu({ line, group, locks, clipboard, actions }) {
  const keys = Object.keys(line.slots).map((slot) => slotKey(group, line.id, slot));
  const hasLocks = keys.some((key) => locks[key]);
  const canPaste = clipboard && JSON.stringify(Object.keys(clipboard.slots || {})) === JSON.stringify(Object.keys(line.slots || {}));
  return <div className="lb-unit-menu" role="menu">
    <button type="button" className="lb-menu-action" onClick={() => actions.clear(group, line.id)}>Clear Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.autoBuild(group, line.id)}>Auto Build Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.lock(group, line.id)}>Lock Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.unlock(group, line.id)} disabled={!hasLocks}>Unlock Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.copy(group, line.id)}>Copy Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.paste(group, line.id)} disabled={!canPaste}>Paste Unit</button>
    <button type="button" className="lb-menu-action" onClick={() => actions.reset(group, line.id)}>Reset Unit</button>
    <button type="button" className="lb-menu-action" onClick={actions.close}>Close Menu</button>
  </div>;
}
function SlotColumnHeaders({ group, showThird }) {
  const slots = group === "defense"
    ? ["LD", "RD"]
    : group === "goalies"
      ? (showThird ? ["Starter", "Backup", "Third"] : ["Starter", "Backup"])
      : ["LW", "C", "RW"];
  return <div className={`lb-col-headers ${group}`} aria-hidden="true">
    <div className="lb-col-headers-spacer" />
    <div className={`lb-col-headers-slots ${group === "goalies" && showThird ? "goalies third" : group}`}>{slots.map((slot) => <span className="lb-col-header" key={slot}>{slot}</span>)}</div>
    <div className="lb-col-headers-chem">Chem</div>
  </div>;
}
function UnitRow({ group, line, playerMap, chemistry, warnings, selectedUnit, selectedSlotKey, selectedPlayer, locks, showThird, menuOpen, clipboard, onSelectUnit, onSlotSelect, onDrop, onDragStart, onRemove, onToggleLock, onSlotKeyDown, registerSlotRef, onWarnings, onMenuToggle, actions, stripe }) {
  const slots = Object.keys(line.slots).filter((slot) => slot !== "Third" || showThird);
  const lockedCount = slots.filter((slot) => locks[slotKey(group, line.id, slot)]).length;
  const slotClass = group === "defense" ? "defense" : group === "goalies" ? `goalies ${showThird ? "third" : ""}` : "forwards";
  const unitNumber = String(line.name || "").replace(/\D+/g, "") || line.id;
  return <div className={`lb-unit ${selectedUnit?.group === group && selectedUnit?.lineId === line.id ? "selected" : ""} ${line.id.endsWith("1") ? "primary" : ""} ${stripe ? "stripe" : ""}`} onClick={() => onSelectUnit(group, line.id)}>
    <div className="lb-unit-label"><span className="lb-unit-kicker">{group === "defense" ? "Pair" : group === "goalies" ? "Unit" : "Line"}</span><span className="lb-unit-name">{unitNumber}</span>{lockedCount ? <span className="lb-unit-locks">{lockedCount}L</span> : null}</div>
    <div className={`lb-slots ${slotClass}`}>{slots.map((slot) => {
      const descriptor = { key: slotKey(group, line.id, slot), group, lineId: line.id, lineName: line.name, slot, playerId: String(line.slots[slot] || "") };
      const player = playerMap[descriptor.playerId] || null;
      let targetState = "";
      if (selectedPlayer) targetState = !selectedPlayer.availability?.placeable || !posFit(selectedPlayer, slot) ? "invalid" : player ? "swap" : "valid";
      return <CompactPlayerSlot key={descriptor.key} descriptor={descriptor} player={player} locked={Boolean(locks[descriptor.key])} selected={selectedSlotKey === descriptor.key} targetState={targetState} onSelect={onSlotSelect} onDrop={onDrop} onDragStart={onDragStart} onRemove={onRemove} onToggleLock={onToggleLock} onKeyDown={onSlotKeyDown} registerRef={registerSlotRef} />;
    })}</div>
    <div className="lb-unit-status">
      <span className="lb-unit-chem" title={chemistry?.projected ? "Projected chemistry" : "Live chemistry"}>{chemistry?.score || 0}<span className="lb-unit-chem-unit">%</span></span>
      <div className="lb-unit-tools">
        {warnings.length ? <button type="button" className="lb-warning has" onClick={(event) => { event.stopPropagation(); onWarnings(group, line.id); }} aria-label={`${warnings.length} unit warnings`} title={`${warnings.length} warnings`}><span className="lb-warning-count">{warnings.length}</span></button> : null}
        <button type="button" className="lb-menu-btn" onClick={(event) => { event.stopPropagation(); onMenuToggle(group, line.id); }} aria-label="Open unit actions" title="Unit actions">···</button>
      </div>
    </div>
    {menuOpen ? <UnitMenu line={line} group={group} locks={locks} clipboard={clipboard} actions={actions} /> : null}
  </div>;
}
function Overview({ lineState, playerMap, chemistryByUnit, warningsByUnit, locks, showThird, onOpenUnit }) {
  const rows = [...lineState.forwards.map((line) => ({ group: "forwards", line })), ...lineState.defense.map((line) => ({ group: "defense", line })), ...lineState.goalies.map((line) => ({ group: "goalies", line }))];
  return <div className="lb-overview">{rows.map(({ group, line }) => {
    const slots = Object.keys(line.slots).filter((slot) => slot !== "Third" || showThird);
    const key = `${group}:${line.id}`;
    const lockCount = slots.filter((slot) => locks[slotKey(group, line.id, slot)]).length;
    return <button type="button" key={key} className="lb-overview-row" onClick={() => onOpenUnit(group, line.id)} aria-label={`Open ${line.name}`}><span className="lb-overview-name">{line.name}</span><span className="lb-overview-players">{slots.map((slot) => { const player = playerMap[String(line.slots[slot] || "")]; return <span className="lb-overview-player" key={slot}>{player ? <PlayerHeadshot player={player} size="sm" /> : null}<span title={player?.name || `Empty ${slot}`}>{player?.name || slot}</span></span>; })}</span><span className="lb-overview-metrics"><span className="lb-overview-score">{chemistryByUnit[key]?.score ?? 0}%</span><span className="lb-overview-warning">W{warningsByUnit[key]?.length || 0}{lockCount ? ` · L${lockCount}` : ""}</span></span></button>;
  })}</div>;
}
function LineBoard({ mode, setMode, lineState, playerMap, chemistryByUnit, warningsByUnit, selectedUnit, selectedSlotKey, selectedPlayer, locks, showThird, menuKey, clipboard, onSelectUnit, onSlotSelect, onDrop, onDragStart, onRemove, onToggleLock, onSlotKeyDown, registerSlotRef, onWarnings, onMenuToggle, actions, onOpenUnit, onTogglePool, onToggleInspector, rosterEmpty }) {
  const options = [[LINEUP_MODES.FORWARDS, "Forwards"], [LINEUP_MODES.DEFENSE, "Defence"], [LINEUP_MODES.GOALIES, "Goalies"], [LINEUP_MODES.OVERVIEW, "Overview"]];
  const activeGroup = mode === LINEUP_MODES.DEFENSE ? "defense" : mode === LINEUP_MODES.GOALIES ? "goalies" : "forwards";
  return <section className="lb-region lb-board-region">
    <div className="lb-modebar">{options.map(([id, label], index) => <button type="button" key={id} className={`lb-mode ${mode === id ? "active" : ""}`} onClick={() => setMode(id)} title={`${label} · ${index + 1}`}><span>{label}</span></button>)}<div className="lb-mobile-tools"><button type="button" className="lb-icon" onClick={onTogglePool} aria-label="Open player pool" title="Players">P</button><button type="button" className="lb-icon" onClick={onToggleInspector} aria-label="Open lineup details" title="Details">D</button></div></div>
    {rosterEmpty ? <div className="lb-empty">Roster unavailable</div> : mode === LINEUP_MODES.OVERVIEW ? <Overview lineState={lineState} playerMap={playerMap} chemistryByUnit={chemistryByUnit} warningsByUnit={warningsByUnit} locks={locks} showThird={showThird} onOpenUnit={onOpenUnit} /> : <div className={`lb-board ${activeGroup}`}><SlotColumnHeaders group={activeGroup} showThird={showThird} />{lineState[activeGroup].map((line, index) => { const key = `${activeGroup}:${line.id}`; return <UnitRow key={key} group={activeGroup} line={line} playerMap={playerMap} chemistry={chemistryByUnit[key]} warnings={warningsByUnit[key] || []} selectedUnit={selectedUnit} selectedSlotKey={selectedSlotKey} selectedPlayer={selectedPlayer} locks={locks} showThird={showThird} menuOpen={menuKey === key} clipboard={clipboard} onSelectUnit={onSelectUnit} onSlotSelect={onSlotSelect} onDrop={onDrop} onDragStart={onDragStart} onRemove={onRemove} onToggleLock={onToggleLock} onSlotKeyDown={onSlotKeyDown} registerSlotRef={registerSlotRef} onWarnings={onWarnings} onMenuToggle={onMenuToggle} actions={actions} stripe={index % 2 === 1} />; })}</div>}
  </section>;
}
function MetricRow({ label, value }) {
  if (value == null) return null;
  return <div className="lb-row"><span className="lb-row-label">{label}</span><progress className="lb-progress" value={clamp(value)} max="100" /><span className="lb-row-value">{clamp(value)}%</span></div>;
}
function LineInspector({ open, onClose, tab, setTab, selectedLine, selectedPlayer, selectedSlot, playerMap, chemistry, warnings, unitPlayers, comparisonPlayers, comparing, setComparing, onReplace, onRemoveSelected, onToggleSelectedLock, selectedLocked, chemistryLoading }) {
  const average = unitPlayers.length ? Math.round(unitPlayers.reduce((sum, player) => sum + player.overall, 0) / unitPlayers.length) : null;
  const roles = roleMixLabel(unitPlayers);
  const valid = selectedLine ? Object.entries(selectedLine.slots).every(([slot, id]) => id ? posFit(playerMap[String(id)], slot) : true) : false;
  const concern = shortText(chemistry.concerns?.[0] || "No major concern.", 8);
  const strength = shortText(chemistry.factors?.[0] || "No confirmed strength.", 8);
  return <section className={`lb-region lb-inspector-region ${open ? "open" : ""}`}>
    <div className="lb-region-head"><h2 className="lb-region-title">Inspector</h2><span className="lb-region-note">{selectedLine?.name || "Lineup"}</span><button type="button" className="lb-icon lb-drawer-close" onClick={onClose} aria-label="Close inspector" title="Close">×</button></div>
    <div className="lb-inspector"><div className="lb-tabs">{TABS.map((id) => <button type="button" key={id} className={`lb-tab ${tab === id ? "active" : ""}`} onClick={() => { setTab(id); setComparing(false); }}>{id}</button>)}</div><div className="lb-inspector-body">
      {selectedPlayer ? <div className="lb-inspector-player"><PlayerHeadshot player={selectedPlayer} size="lg" /><div><div className="lb-inspector-player-name">{selectedPlayer.name}</div><div className="lb-inspector-player-meta">{selectedPlayer.position} · {selectedPlayer.overall} OVR · {selectedPlayer.availability?.label}</div><div className="lb-inspector-player-meta">{shortRole(selectedPlayer.role) || "—"} · {selectedPlayer.handedness} · {getCountry(selectedPlayer)}</div></div></div> : <div><div className="lb-inspector-title">{selectedLine?.name || "Full Lineup"}</div><div className="lb-inspector-sub">{chemistry.projected ? "Projected chemistry" : "Backend chemistry"}</div></div>}
      {comparing ? <div className="lb-list">{comparisonPlayers.length ? comparisonPlayers.map((player) => <div className="lb-row lb-compare" key={player.id}><div className="lb-compare-copy"><div className="lb-compare-name">{player.name}</div><div className="lb-compare-meta">{player.overall} OVR · {player.fit}% fit · {player.availability.label}</div></div><button type="button" className="lb-btn" onClick={() => onReplace(player.id)} disabled={!player.availability.placeable}>Select</button></div>) : <div className="lb-empty">No valid candidates</div>}</div> : tab === "summary" ? <><div className="lb-gauge"><div className="lb-gauge-score">{chemistry.score}</div><div><div className="lb-gauge-label">{shortText(chemistry.label, 4)}</div><div className="lb-gauge-source">{chemistryLoading ? "Loading" : chemistry.projected ? "Projected unit chem" : "Live unit chem"}</div></div></div><div className="lb-list"><div className="lb-row"><span className="lb-row-label">Avg OVR</span><span className="lb-row-value">{average ?? "—"}</span></div><div className="lb-row"><span className="lb-row-label">Roles</span><span className="lb-row-value" title={roles}>{roles || "—"}</span></div><div className="lb-row"><span className="lb-row-label">Positions</span><span className={`lb-row-value ${valid ? "ok" : "warn"}`}>{valid ? "Valid" : "Review"}</span></div><div className="lb-row"><span className="lb-row-label">Strength</span><span className="lb-row-value" title={strength}>{strength}</span></div><div className="lb-row"><span className="lb-row-label">Concern</span><span className="lb-row-value" title={concern}>{concern}</span></div></div></> : tab === "fit" ? <div className="lb-list"><MetricRow label="Position Fit" value={chemistry.positionFit} /><MetricRow label="Role Balance" value={chemistry.roleBalance} /><MetricRow label="Linemate Fit" value={chemistry.linemateFit} /><MetricRow label="Coach Fit" value={chemistry.coachFit} /><MetricRow label="Familiarity" value={chemistry.familiarity} /></div> : tab === "status" ? <div className="lb-list">{selectedPlayer ? <><div className="lb-row"><span className="lb-row-label">Availability</span><span className="lb-row-value">{selectedPlayer.availability?.label || "—"}</span></div><div className="lb-row"><span className="lb-row-label">Morale</span><span className="lb-row-value">{selectedPlayer.morale ?? "—"}</span></div><div className="lb-row"><span className="lb-row-label">Fatigue</span><span className="lb-row-value">{selectedPlayer.fatigue ?? "—"}</span></div><div className="lb-row"><span className="lb-row-label">Confidence</span><span className="lb-row-value">{selectedPlayer.confidence ?? "—"}</span></div><div className="lb-row"><span className="lb-row-label">Role satisfaction</span><span className="lb-row-value">{selectedPlayer.role_satisfaction ?? "—"}</span></div><div className="lb-row"><span className="lb-row-label">Coach trust</span><span className="lb-row-value">{selectedPlayer.coach_trust ?? "—"}</span></div></> : <div className="lb-empty">Select a player</div>}</div> : <div className="lb-list">{warnings.length ? warnings.map((warning) => <div className="lb-row lb-warning-row" key={warning.key}>{warning.text}</div>) : <div className="lb-empty">No unit warnings</div>}</div>}
      {selectedSlot ? <div className="lb-inspector-actions"><button type="button" className="lb-btn" onClick={() => setComparing((current) => !current)}>{comparing ? "Close" : "Compare"}</button><button type="button" className="lb-btn" onClick={() => setComparing(true)}>Replace</button><button type="button" className="lb-btn" onClick={onToggleSelectedLock}>{selectedLocked ? "Unlock" : "Lock"}</button><button type="button" className="lb-btn danger" onClick={onRemoveSelected} disabled={selectedLocked || !selectedPlayer}>Remove</button></div> : null}
    </div></div>
  </section>;
}
function AutoBuildPopover({ state, setState, changes, onApply }) {
  if (!state.open) return null;
  return <div className="lb-popover" role="dialog" aria-label="Auto Build lineup"><h2 className="lb-popover-title">{state.scope ? "Auto Build Unit" : "Auto Build Lineup"}</h2><div className="lb-auto-options">{AUTO_MODES.map(([id, label]) => <button type="button" key={id} className={`lb-auto-option ${state.mode === id ? "active" : ""}`} onClick={() => setState((current) => ({ ...current, mode: id }))}>{label}</button>)}</div><div className="lb-preview">{changes.length ? changes.map((change) => <div className="lb-preview-row" key={change.key}><span className="lb-preview-label">{change.label}</span><span className="lb-preview-change">{change.before} → {change.after}</span></div>) : <div className="lb-empty">No changes found</div>}</div><div className="lb-popover-actions"><button type="button" className="lb-btn" onClick={() => setState((current) => ({ ...current, open: false, scope: null }))}>Cancel</button><button type="button" className="lb-btn primary" onClick={onApply} disabled={!changes.length}>Apply</button></div></div>;
}
function LineBuilderToast({ toast, onDismiss, onDetails }) {
  if (!toast) return null;
  return <div className={`lb-toast ${toast.type}`} role="status"><span className="lb-toast-message">{shortText(toast.message)}</span><span className="lb-toast-actions">{toast.details ? <button type="button" className="lb-mini" onClick={onDetails}>Details</button> : null}<button type="button" className="lb-mini" onClick={onDismiss} aria-label="Dismiss message" title="Dismiss">×</button></span></div>;
}

function EvenStrengthLines(props) {
  const { franchiseState, setScreen, setFranchiseState } = useGameUI();
  const sessionId = getFranchiseSessionId();
  const team = useMemo(() => teamIdentity(franchiseState, props), [franchiseState, props]);
  const [chemReport, setChemReport] = useState(null);
  const [chemLoading, setChemLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [mode, setMode] = useState(LINEUP_MODES.FORWARDS);
  const [selectedUnit, setSelectedUnit] = useState({ group: "forwards", lineId: "f1" });
  const [selectedSlotKey, setSelectedSlotKey] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [tab, setTab] = useState("summary");
  const [inspectorOpen, setInspectorOpen] = useState(false);
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

  useEffect(() => {
    let active = true;
    setChemLoading(true);
    getFranchiseChemistry().then((data) => { if (active) setChemReport(data || null); }).catch(() => { if (active) setChemReport(null); }).finally(() => { if (active) setChemLoading(false); });
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
  const chemistryByUnit = useMemo(() => {
    const result = {};
    for (const group of GROUPS) for (const line of lineState[group] || []) {
      const unitPlayers = Object.values(line.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean);
      result[`${group}:${line.id}`] = calculateUnitChemistry(unitPlayers, group === "defense" ? "defense" : group === "goalies" ? "goalie" : "forward", chemReport);
    }
    return result;
  }, [lineState, playerMap, chemReport]);
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
  const selectedWarnings = useMemo(() => warningsByUnit[`${selectedUnit.group}:${selectedLine?.id}`] || validation.errors, [warningsByUnit, selectedUnit, selectedLine, validation.errors]);
  const focusSlot = useMemo(() => selectedSlot?.slot || Object.entries(selectedLine?.slots || {}).find(([, id]) => !id)?.[0] || Object.keys(selectedLine?.slots || {})[0] || null, [selectedSlot, selectedLine]);

  const filteredPool = useMemo(() => {
    const query = search.trim().toLowerCase();
    return sortPoolPlayers(players.filter((player) => !query || `${player.name} ${player.position} ${player.role}`.toLowerCase().includes(query)), "overall", focusSlot);
  }, [players, search, focusSlot]);
  const comparisonPlayers = useMemo(() => selectedSlot ? sortPoolPlayers(players.filter((player) => posFit(player, selectedSlot.slot)).map((player) => ({ ...player, fit: chemistryFitScore(player, selectedSlot.slot) })), "overall", selectedSlot.slot).slice(0, 5) : [], [players, selectedSlot]);
  const assignedPlayers = useMemo(() => [...assignedSet].map((id) => playerMap[id]).filter(Boolean), [assignedSet, playerMap]);
  const teamChemistry = useMemo(() => {
    // Status bar chemistry = average of dressed unit scores (lines/pairs/goalies),
    // not a separate projected score of all 20 players mashed together.
    const scores = Object.values(chemistryByUnit || {})
      .map((entry) => Number(entry?.score))
      .filter((score) => Number.isFinite(score) && score > 0);
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
    return [
      { label: "Skaters", value: `${skaters.size}/18` },
      { label: "Goalies", value: `${activeGoalies}/2` },
      { label: "Scratches", value: scratches },
      { label: "Team Chem", value: assignedPlayers.length ? `${teamChemistry.score}%` : null, title: "Average chemistry across all dressed lines, pairs, and goalies" },
      { label: "Average", value: average },
      { label: "Warnings", value: warningCount },
    ];
  }, [lineState, players, assignedPlayers, warningsByUnit, validation.errors, teamChemistry]);
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
    setTab("summary");
    setComparing(false);
  }, [selectedSlot, placePlayer]);
  const onSlotSelect = useCallback((row) => {
    if (selectedPlayerId) { placePlayer(selectedPlayerId, row); return; }
    setSelectedUnit({ group: row.group, lineId: row.lineId });
    setSelectedSlotKey((current) => current === row.key ? "" : row.key);
    setSelectedPlayerId("");
    setInspectorOpen(true);
    setTab("summary");
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
    if (event.key === " ") { event.preventDefault(); selectedPlayerId ? placePlayer(selectedPlayerId, row) : onSlotSelect(row); return; }
    if (event.key === "Delete") { event.preventDefault(); removeSlot(row); return; }
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const activeGroup = mode === LINEUP_MODES.DEFENSE ? "defense" : mode === LINEUP_MODES.GOALIES ? "goalies" : "forwards";
    const visible = descriptors(lineState, showThird).filter((item) => mode === LINEUP_MODES.OVERVIEW || item.group === activeGroup);
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
      if (["1", "2", "3", "4"].includes(event.key)) setMode({ 1: LINEUP_MODES.FORWARDS, 2: LINEUP_MODES.DEFENSE, 3: LINEUP_MODES.GOALIES, 4: LINEUP_MODES.OVERVIEW }[event.key]);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedSlot, removeSlot, undo, redo]);

  const changeMode = useCallback((nextMode) => {
    setMode(nextMode);
    if (nextMode === LINEUP_MODES.FORWARDS) setSelectedUnit((current) => current.group === "forwards" ? current : { group: "forwards", lineId: "f1" });
    if (nextMode === LINEUP_MODES.DEFENSE) setSelectedUnit((current) => current.group === "defense" ? current : { group: "defense", lineId: "d1" });
    if (nextMode === LINEUP_MODES.GOALIES) setSelectedUnit({ group: "goalies", lineId: "g1" });
    setMenuKey("");
  }, []);
  const openUnit = useCallback((group, lineId) => { setMode(modeForGroup(group)); setSelectedUnit({ group, lineId }); setSelectedSlotKey(""); setInspectorOpen(true); }, []);
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
    commit(autoPreview, locks, "Auto Build applied.");
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
      setToast(response?.warnings?.length ? { type: "warning", message: shortText(response.warnings[0]), details: response.warnings.length > 1 } : { type: "success", message: "Lines saved." });
    } catch {
      setSaveError(true);
      setToast({ type: "error", message: "Backend save failed." });
    } finally {
      setSaving(false);
    }
  }, [saving, players.length, validation, lineState, locks, sessionId, props, setFranchiseState]);
  const replaceSelected = useCallback((playerId) => { if (selectedSlot && placePlayer(playerId, selectedSlot)) setComparing(false); }, [selectedSlot, placePlayer]);
  const selectedLocked = selectedSlot ? Boolean(locks[selectedSlot.key]) : false;
  const rosterLoading = !franchiseState && !players.length;

  return <div className="linebuilder-root">
    <EditLinesStyles />
    <LineBuilderSidebar setScreen={setScreen} abbreviation={team.abbreviation} activeScreen={SCREENS.EDIT_LINES} />
    <main className="lb-shell">
      <LineBuilderHeader team={team} rosterCount={players.length} canUndo={history.length > 0} canRedo={future.length > 0} onUndo={undo} onRedo={redo} onClear={clearAll} onReset={resetSaved} onAutoBuild={() => setAutoBuild((current) => ({ ...current, open: !current.open, scope: null }))} onSave={saveLines} saving={saving} unsaved={unsaved} saveError={saveError} disabled={!players.length} />
      <LineStatusStrip metrics={statusMetrics} />
      <div className="lb-workspace">
        {rosterLoading ? <section className="lb-region lb-pool-region"><div className="lb-region-head"><h2 className="lb-region-title">Player Pool</h2></div><div className="lb-loading"><div className="lb-skeleton" /></div></section> : <PlayerPool players={filteredPool} assignedSet={assignedSet} lockedSet={lockedSet} selectedPlayerId={selectedPlayerId} search={search} setSearch={setSearch} focusSlot={focusSlot} onPlayerSelect={onPlayerSelect} onDragStart={onDragStart} open={poolOpen} onClose={() => setPoolOpen(false)} />}
        <LineBoard mode={mode} setMode={changeMode} lineState={lineState} playerMap={playerMap} chemistryByUnit={chemistryByUnit} warningsByUnit={warningsByUnit} selectedUnit={selectedUnit} selectedSlotKey={selectedSlotKey} selectedPlayer={selectedPlayerId ? playerMap[selectedPlayerId] : null} locks={locks} showThird={showThird} menuKey={menuKey} clipboard={clipboard} onSelectUnit={(group, lineId) => { setSelectedUnit({ group, lineId }); setInspectorOpen(true); }} onSlotSelect={onSlotSelect} onDrop={onDrop} onDragStart={onDragStart} onRemove={removeSlot} onToggleLock={toggleLock} onSlotKeyDown={onSlotKeyDown} registerSlotRef={registerSlotRef} onWarnings={openWarnings} onMenuToggle={(group, lineId) => { const key = `${group}:${lineId}`; setMenuKey((current) => current === key ? "" : key); }} actions={actions} onOpenUnit={openUnit} onTogglePool={() => setPoolOpen((current) => !current)} onToggleInspector={() => setInspectorOpen((current) => !current)} rosterEmpty={!players.length && !rosterLoading} />
        <LineInspector open={inspectorOpen} onClose={() => setInspectorOpen(false)} tab={tab} setTab={setTab} selectedLine={selectedLine} selectedPlayer={selectedPlayer} selectedSlot={selectedSlot} playerMap={playerMap} chemistry={selectedChemistry} warnings={selectedWarnings} unitPlayers={unitPlayers} comparisonPlayers={comparisonPlayers} comparing={comparing} setComparing={setComparing} onReplace={replaceSelected} onRemoveSelected={() => selectedSlot && removeSlot(selectedSlot)} onToggleSelectedLock={() => selectedSlot && toggleLock(selectedSlot)} selectedLocked={selectedLocked} chemistryLoading={chemLoading} />
      </div>
      <AutoBuildPopover state={autoBuild} setState={setAutoBuild} changes={autoChanges} onApply={applyAutoBuild} />
    </main>
    <LineBuilderToast toast={toast} onDismiss={() => setToast(null)} onDetails={() => { setTab("warnings"); setInspectorOpen(true); setToast(null); }} />
    <div className="lb-live" aria-live="polite">{announcement}</div>
  </div>;
}

function specialTeamsProfile(player, key, fallback = 75) {
  const profile = player?.chemistry_profile || {};
  return Number(profile[key] ?? profile[String(key).replaceAll("_", "")] ?? fallback);
}

function calculatePowerPlayChemistry(players) {
  const selected = players.filter(Boolean);
  if (!selected.length) {
    return { score: 0, label: "No Unit", creativity: 0, movement: 0, finishing: 0, balance: 0, tips: ["Select players to build your power play unit."] };
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
  const label = score >= 90 ? "Terrifying Unit" : score >= 80 ? "Dangerous PP" : score >= 70 ? "Functional Unit" : score >= 60 ? "Needs a Trigger" : "Disconnected";
  const tips = [];
  if (!hasShooter) tips.push("Add a true shooter/sniper so the unit has a dangerous trigger.");
  if (!hasPlaymaker) tips.push("Add a playmaker to improve puck movement and zone control.");
  if (!hasNetFront) tips.push("A net-front player helps screens, rebounds, and dirty goals.");
  if (!hasDefense) tips.push("Use at least one defenseman or power-play quarterback.");
  if (movement < 78) tips.push("Puck movement is low. This unit may become too static.");
  if (score >= 82) tips.push("This power play has a strong mix of movement, finishing, and role fit.");
  return { score, label, creativity: clamp(creativity), movement: clamp(movement), finishing: clamp(finishing), balance: clamp(balance), tips };
}

function calculatePenaltyKillChemistry(players) {
  const selected = players.filter(Boolean);
  if (!selected.length) {
    return { score: 0, label: "No Unit", defensive: 0, discipline: 0, balance: 0, trust: 0, tips: ["Select players to build your penalty kill unit."] };
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
  const label = score >= 90 ? "Elite PK Identity" : score >= 80 ? "Reliable Killers" : score >= 70 ? "Playable Unit" : score >= 60 ? "Risky Mix" : "Needs Work";
  const tips = [];
  if (defensive < 78) tips.push("Defensive buy-in is low. Add a shutdown defender or two-way forward.");
  if (discipline < 76) tips.push("Discipline is dragging this unit down. Avoid penalty-prone players on the PK.");
  if (!hasTwoDefense) tips.push("This PK unit should include two defensemen.");
  if (!hasTwoForwards) tips.push("This PK unit should include two forwards.");
  if (score >= 82) tips.push("This unit has strong defensive trust and should handle tough matchups.");
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

function SpecialTeamsLines({ kind, ...props }) {
  const isPP = kind === "power_play";
  const { franchiseState, setScreen, setFranchiseState } = useGameUI();
  const sessionId = getFranchiseSessionId();
  const team = useMemo(() => teamIdentity(franchiseState, props), [franchiseState, props]);
  const activeScreen = isPP ? SCREENS.POWER_PLAY : SCREENS.PENALTY_KILL;
  const title = isPP ? "Power Play" : "Penalty Kill";
  const slotOrder = isPP ? ["LW", "C", "RW", "LD", "RD"] : ["F1", "F2", "D1", "D2"];
  const boardClass = isPP ? "special pp" : "special pk";
  const slotsClass = isPP ? "pp" : "pk";
  const chemistryFn = isPP ? calculatePowerPlayChemistry : calculatePenaltyKillChemistry;
  const makeInitial = isPP ? makeInitialPowerPlayLines : makeInitialPenaltyKillLines;

  const [search, setSearch] = useState("");
  const [toast, setToast] = useState(null);
  const [invalidMsg, setInvalidMsg] = useState("");
  const [unsaved, setUnsaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [selectedLineId, setSelectedLineId] = useState(isPP ? "pp1" : "pk1");
  const [selectedSlot, setSelectedSlot] = useState("");
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
  const selectedPlayers = useMemo(() => lines.flatMap((line) => Object.values(line.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean)), [lines, playerMap]);
  const overallChemistry = useMemo(() => chemistryFn(selectedPlayers), [chemistryFn, selectedPlayers]);
  const selectedLine = useMemo(() => lines.find((line) => line.id === selectedLineId) || lines[0] || null, [lines, selectedLineId]);
  const unitPlayers = useMemo(() => selectedLine ? Object.values(selectedLine.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean) : [], [selectedLine, playerMap]);
  const unitChemistry = useMemo(() => chemistryFn(unitPlayers), [chemistryFn, unitPlayers]);
  const selectedPlayer = useMemo(() => {
    if (selectedPlayerId && playerMap[selectedPlayerId]) return playerMap[selectedPlayerId];
    if (selectedLine && selectedSlot) return playerMap[String(selectedLine.slots?.[selectedSlot] || "")] || null;
    return null;
  }, [selectedPlayerId, playerMap, selectedLine, selectedSlot]);
  const filteredPool = useMemo(() => {
    const query = search.trim().toLowerCase();
    return players.filter((player) => !query || `${player.name} ${player.position} ${player.role}`.toLowerCase().includes(query));
  }, [players, search]);
  const missingCount = useMemo(() => lines.reduce((sum, line) => sum + Object.values(line.slots || {}).filter((id) => !id).length, 0), [lines]);
  const statusMetrics = useMemo(() => [
    { label: "Units", value: `${lines.length}` },
    { label: "Filled", value: `${assignedSet.size}/${lines.reduce((sum, line) => sum + Object.keys(line.slots || {}).length, 0)}` },
    { label: "Chemistry", value: `${overallChemistry.score}%` },
    { label: "Label", value: overallChemistry.label },
    { label: "Warnings", value: missingCount + (invalidMsg ? 1 : 0) },
  ], [lines, assignedSet, overallChemistry, missingCount, invalidMsg]);

  const placePlayer = useCallback((playerId, lineId, slot) => {
    const player = playerMap[String(playerId || "")];
    if (!player || !lineId || !slot) return false;
    if (!specialTeamsSlotAllowed(player, slot, kind)) {
      setInvalidMsg(`${player.name || "Player"} is not a valid fit for ${slot}.`);
      setToast({ type: "error", message: `${player.name} cannot play ${slot}.` });
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
    if (selectedLineId && selectedSlot) {
      placePlayer(playerId, selectedLineId, selectedSlot);
      return;
    }
    setSelectedPlayerId((current) => current === playerId ? "" : playerId);
    setInspectorOpen(true);
  }, [selectedLineId, selectedSlot, placePlayer]);

  const onSlotSelect = useCallback((lineId, slot) => {
    if (selectedPlayerId) {
      placePlayer(selectedPlayerId, lineId, slot);
      return;
    }
    setSelectedLineId(lineId);
    setSelectedSlot((current) => current === slot && selectedLineId === lineId ? "" : slot);
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
      setToast({ type: "success", message: `${isPP ? "PP" : "PK"} lines saved.` });
    } catch {
      setSaveError(true);
      setToast({ type: "error", message: "Backend save failed." });
    } finally {
      setSaving(false);
    }
  }, [saving, kind, lines, sessionId, isPP, props, setFranchiseState]);

  const metricRows = isPP
    ? [
      ["Creativity", unitChemistry.creativity],
      ["Puck Movement", unitChemistry.movement],
      ["Finishing", unitChemistry.finishing],
      ["Role Balance", unitChemistry.balance],
    ]
    : [
      ["Defensive Buy-In", unitChemistry.defensive],
      ["Discipline", unitChemistry.discipline],
      ["Unit Balance", unitChemistry.balance],
      ["Work Ethic / Trust", unitChemistry.trust],
    ];

  const rosterLoading = !franchiseState && !players.length;

  return <div className="linebuilder-root">
    <EditLinesStyles />
    <LineBuilderSidebar setScreen={setScreen} abbreviation={team.abbreviation} activeScreen={activeScreen} />
    <main className="lb-shell">
      <LineBuilderHeader
        team={team}
        rosterCount={players.length}
        title={title}
        showHistory={false}
        showAutoBuild={false}
        onClear={clearAll}
        onReset={resetLines}
        resetLabel="Auto Fill"
        onSave={saveLines}
        saving={saving}
        unsaved={unsaved}
        saveError={saveError}
        disabled={!players.length}
        saveLabel={unsaved ? `Save ${isPP ? "PP" : "PK"}` : "Saved"}
      />
      <LineStatusStrip metrics={statusMetrics} />
      <div className="lb-workspace">
        {rosterLoading
          ? <section className="lb-region lb-pool-region"><div className="lb-region-head"><h2 className="lb-region-title">Player Pool</h2></div><div className="lb-loading"><div className="lb-skeleton" /></div></section>
          : <PlayerPool players={filteredPool} assignedSet={assignedSet} lockedSet={emptyLockedSet} selectedPlayerId={selectedPlayerId} search={search} setSearch={setSearch} focusSlot={selectedSlot || null} onPlayerSelect={onPlayerSelect} onDragStart={onDragStart} open={poolOpen} onClose={() => setPoolOpen(false)} />}
        <section className="lb-region lb-board-region">
          <div className="lb-modebar">
            <button type="button" className="lb-mode active"><span>{isPP ? "PP Units" : "PK Units"}</span></button>
            <div className="lb-mobile-tools">
              <button type="button" className="lb-icon" onClick={() => setPoolOpen((v) => !v)} aria-label="Open player pool" title="Players">P</button>
              <button type="button" className="lb-icon" onClick={() => setInspectorOpen((v) => !v)} aria-label="Open unit details" title="Details">D</button>
            </div>
          </div>
          {!players.length && !rosterLoading ? <div className="lb-empty">Roster unavailable</div> : (
            <div className={`lb-board ${boardClass}`}>
              <div className={`lb-col-headers ${slotsClass}`} aria-hidden="true">
                <div className="lb-col-headers-spacer" />
                <div className={`lb-col-headers-slots ${slotsClass}`}>{slotOrder.map((slot) => <span className="lb-col-header" key={slot}>{slot}</span>)}</div>
                <div className="lb-col-headers-chem">Chem</div>
              </div>
              {lines.map((line, index) => {
                const linePlayers = Object.values(line.slots || {}).map((id) => playerMap[String(id || "")]).filter(Boolean);
                const chem = chemistryFn(linePlayers);
                const unitNumber = String(line.name || "").replace(/\D+/g, "") || String(index + 1);
                return <div key={line.id} className={`lb-unit ${selectedLineId === line.id ? "selected" : ""} ${line.id.endsWith("1") ? "primary" : ""} ${index % 2 === 1 ? "stripe" : ""}`} onClick={() => { setSelectedLineId(line.id); setInspectorOpen(true); }}>
                  <div className="lb-unit-label"><span className="lb-unit-kicker">Unit</span><span className="lb-unit-name">{unitNumber}</span></div>
                  <div className={`lb-slots ${slotsClass}`}>
                    {slotOrder.map((slot) => {
                      const playerId = String(line.slots?.[slot] || "");
                      const player = playerMap[playerId] || null;
                      const selected = selectedLineId === line.id && selectedSlot === slot;
                      let targetState = "";
                      if (selectedPlayerId && playerMap[selectedPlayerId]) {
                        targetState = !specialTeamsSlotAllowed(playerMap[selectedPlayerId], slot, kind) ? "invalid" : player ? "swap" : "valid";
                      }
                      const classes = ["lb-slot", player ? "occupied" : "", selected ? "selected" : "", targetState].filter(Boolean).join(" ");
                      return <div
                        key={`${line.id}:${slot}`}
                        className={classes}
                        role="button"
                        tabIndex={0}
                        aria-label={player ? `${slot}, ${player.name}` : `${slot}, empty`}
                        onClick={(event) => { event.stopPropagation(); onSlotSelect(line.id, slot); }}
                        onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSlotSelect(line.id, slot); } }}
                        onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
                        onDrop={(event) => onDropSlot(event, line.id, slot)}
                      >
                        {player ? <>
                          <div className="lb-slot-shot" draggable onDragStart={(event) => { event.stopPropagation(); onDragStart(event, player.id); }}><PlayerHeadshot player={player} size="sm" /></div>
                          <div className="lb-slot-copy"><div className="lb-slot-name" title={player.name}>{player.name}</div><div className="lb-slot-meta">{slotMeta(player)}</div></div>
                          <div className="lb-slot-side"><div className="lb-slot-ovr">{player.overall}</div></div>
                        </> : <><span className="lb-slot-pos">{slot}</span><span className="lb-slot-empty">Empty</span></>}
                      </div>;
                    })}
                  </div>
                  <div className="lb-unit-status"><span className="lb-unit-chem">{chem.score}<span className="lb-unit-chem-unit">%</span></span></div>
                </div>;
              })}
            </div>
          )}
        </section>
        <section className={`lb-region lb-inspector-region ${inspectorOpen ? "open" : ""}`}>
          <div className="lb-region-head">
            <h2 className="lb-region-title">Inspector</h2>
            <span className="lb-region-note">{selectedLine?.name || title}</span>
            <button type="button" className="lb-icon lb-drawer-close" onClick={() => setInspectorOpen(false)} aria-label="Close inspector" title="Close">×</button>
          </div>
          <div className="lb-inspector">
            <div className="lb-inspector-body">
              {selectedPlayer ? (
                <div className="lb-inspector-player">
                  <PlayerHeadshot player={selectedPlayer} size="lg" />
                  <div>
                    <div className="lb-inspector-player-name">{selectedPlayer.name}</div>
                    <div className="lb-inspector-player-meta">{selectedPlayer.position} · {selectedPlayer.overall} OVR</div>
                    <div className="lb-inspector-player-meta">{shortRole(selectedPlayer.role) || "—"} · {selectedPlayer.handedness}</div>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="lb-inspector-title">{selectedLine?.name || title}</div>
                  <div className="lb-inspector-sub">{isPP ? "Power play chemistry" : "Penalty kill chemistry"}</div>
                </div>
              )}
              <div className="lb-gauge">
                <div className="lb-gauge-score">{unitChemistry.score}</div>
                <div>
                  <div className="lb-gauge-label">{shortText(unitChemistry.label, 4)}</div>
                  <div className="lb-gauge-source">{isPP ? "PP unit chem" : "PK unit chem"}</div>
                </div>
              </div>
              <div className="lb-list">
                {metricRows.map(([label, value]) => <MetricRow key={label} label={label} value={value} />)}
              </div>
              <div className="lb-list">
                {(unitChemistry.tips || []).map((tip, index) => <div className="lb-row lb-warning-row" key={`tip-${index}`}>{tip}</div>)}
                {invalidMsg ? <div className="lb-row lb-warning-row">{invalidMsg}</div> : null}
                {unsaved ? <div className="lb-row"><span className="lb-row-label">Status</span><span className="lb-row-value warn">Unsaved changes</span></div> : null}
              </div>
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
