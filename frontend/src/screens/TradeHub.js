import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS, normalizeNhlAbbr } from "../game/constants";
import { submitTradePackage } from "../services/franchiseService";

const money = (value) => `$${Number(value || 0).toFixed(2)}M`;
const valueFmt = (value) => Number(value || 0).toFixed(2);
const yearsFmt = (years) => `${years} ${years === 1 ? "Year" : "Years"}`;
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

/** League calendar year used for future-pick discounting */
const TRADE_HUB_SEASON_YEAR = 2025;

const COLORS = {
  panel: "rgba(7, 13, 28, 0.78)",
  panel2: "rgba(15, 24, 48, 0.82)",
  line: "rgba(255,255,255,0.09)",
  blue: "#0b3b91",
  blue2: "#082b68",
  red: "#7b1320",
  red2: "#4f0b14",
  green: "#2a9d55",
  orange: "#e07020",
  silver: "var(--g-silver)",
  dim: "var(--g-silver-dim)",
  text: "var(--g-text)",
  neon: "var(--g-neon)",
};

function teamHueFromId(id) {
  let h = 0;
  const s = String(id || "");
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function inferLogoAbbr(teamId, teamName) {
  const a = normalizeNhlAbbr(teamId);
  if (a) return a;
  const b = normalizeNhlAbbr(teamName);
  if (b) return b;
  const t = String(teamName || teamId || "?").replace(/[^A-Za-z]/g, "");
  return (t.slice(0, 3).toUpperCase() || "?").slice(0, 3);
}

function shortTeamLabel(fullName) {
  const s = String(fullName || "").trim();
  if (!s) return "—";
  const parts = s.split(/\s+/);
  return parts[parts.length - 1] || s;
}

function statusFromStanding(st, isUser, userStrategy) {
  if (isUser) return userStrategy ? `Your club · ${userStrategy}` : "Your club";
  const gp = Number(st?.gp) || 1;
  const pts = Number(st?.pts) || 0;
  const ppg = pts / Math.max(1, gp);
  if (ppg < 0.88) return "Selling";
  if (ppg > 1.12) return "Buying";
  if (ppg > 1.03) return "Contending";
  return "Listening";
}

function needHintFromStatus(status) {
  if (String(status).includes("Selling")) return "Picks / young players";
  if (String(status).includes("Buying")) return "Veteran help / depth";
  if (String(status).includes("Contending")) return "Middle-six / D depth";
  return "Futures / cap flexibility";
}

function roughContractAav(ovr, age) {
  const o = Number(ovr) || 72;
  const a = Number(age) || 25;
  let v = 0.85 + Math.max(0, o - 72) * 0.21;
  if (a <= 23) v -= 0.55;
  if (a >= 34) v -= 0.95;
  return clamp(v, 0.75, 14.5);
}

function normalizeSkaterPosition(raw) {
  const u = String(raw || "").toUpperCase();
  if (u === "RD" || u === "LD" || u.includes("DEF"))
    return { pos: "D", handed: u.includes("RIGHT") || u === "RD" ? "Right" : "Left" };
  if (u === "RW" || u.includes("RIGHT WING")) return { pos: "RW", handed: "Right" };
  if (u === "LW" || u.includes("LEFT WING")) return { pos: "LW", handed: "Left" };
  if (u === "C" || u.includes("CENTER") || u.includes("CENTRE")) return { pos: "C", handed: "Left" };
  if (u === "G" || u.includes("GOALTEND")) return { pos: "G", handed: "Left" };
  return { pos: "C", handed: "Right" };
}

function abbrevName(full) {
  const parts = String(full || "?").trim().split(/\s+/);
  if (parts.length < 2) return String(full || "?").slice(0, 16);
  const last = parts[parts.length - 1];
  const inis = parts
    .slice(0, -1)
    .map((w) => w[0])
    .join(".");
  return `${inis}. ${last}`.slice(0, 22);
}

function apiSkaterToTradePlayer(row, teamId) {
  const fullName = String(row.name || "?");
  const { pos, handed } = normalizeSkaterPosition(row.position);
  const age = Number(row.age) || 25;
  const ovr = Number(row.ovr) || 72;
  const st = row.season_stats || {};
  const gp = Math.max(1, Number(st.gp) || 1);
  const ptsRaw = st.pts;
  const pts = Number.isFinite(Number(ptsRaw)) ? Number(ptsRaw) : (Number(st.g) || 0) + (Number(st.a) || 0);
  const ppg = pts / gp;
  const g = Number(st.g) || 0;
  const sog = Math.max(1, Number(st.sog) || 1);
  const shootPct = gp >= 4 ? (g / sog) * 100 : undefined;
  const arch = String(row.archetype || "").trim();
  const tags = [arch, age <= 22 ? "Prospect" : null, age <= 23 && ovr >= 82 ? "ELC" : null].filter(Boolean);
  const salary = roughContractAav(ovr, age);
  const years = age <= 22 ? 3 : age >= 33 ? 2 : 4;
  const defRating = pos === "D" ? clamp(4 + (ovr - 80) * 0.32, 2, 9.5) : undefined;
  return {
    id: String(row.player_id || `${teamId}-${fullName}`),
    name: abbrevName(fullName),
    fullName,
    pos,
    age,
    ovr,
    salary,
    years,
    ppg,
    clutch: clamp(Math.round((ppg - 0.65) * 2), -1, 3),
    shootPct,
    handed,
    height: String(row.height_display || "—"),
    weight: "—",
    protection: "None",
    projection: arch || `${pos} · NHL roster (contract $ est. for trade math).`,
    tags: tags.length ? tags : ["Roster"],
    defRating,
  };
}

function syntheticPicksForTeam(teamId, seasonYear, teamAbbr) {
  const tid = String(teamId);
  const y0 = Number(seasonYear) || TRADE_HUB_SEASON_YEAR;
  const ab = teamAbbr || tid.slice(0, 3);
  const row = (y, r, band, i) => ({
    id: `pick-${tid}-${y}-r${r}-${i}`,
    label: `${y} Rd ${r} (${ab})`,
    owner: tid,
    draftYear: y,
    round: r,
    firstBand: band,
  });
  return [row(y0, 1, "mid", 0), row(y0 + 1, 2, null, 1), row(y0 + 1, 3, null, 2), row(y0 + 2, 4, null, 3)];
}

/** Live league + user team from franchise session (no hard-coded rosters). */
function buildTradeData(franchiseState) {
  const rb = franchiseState?.roster_browser;
  const userTeamId = String(franchiseState?.team?.id || "");
  const seasonYear = Number(franchiseState?.season_year) || TRADE_HUB_SEASON_YEAR;
  const userStrategy = String(franchiseState?.team?.strategy || "").trim();
  if (!rb?.organizations?.length || !userTeamId) return null;

  const standingsMap = {};
  (franchiseState.standings || []).forEach((r) => {
    standingsMap[String(r.team_id)] = r;
  });

  const organizations = rb.organizations;
  const userCapSpace = Number(franchiseState?.team?.cap_space);
  const userCapHit = Number(franchiseState?.team?.cap_hit);
  const userCapLimit = Number(franchiseState?.team?.salary_cap || franchiseState?.team?.cap_limit);
  const teams = organizations.map((org) => {
    const tid = String(org.team_id);
    const nm = String(org.name || tid);
    const st = standingsMap[tid] || {};
    const abbr = inferLogoAbbr(tid, nm);
    const rec = `${Number(st.w) || 0}-${Number(st.l) || 0}-${Number(st.otl) || 0}`;
    const isUser = tid === userTeamId;
    const status = statusFromStanding(st, isUser, userStrategy);
    return {
      id: tid,
      name: nm,
      short: shortTeamLabel(nm),
      logo: abbr,
      color: `hsl(${teamHueFromId(tid)}, 44%, 30%)`,
      accent: "rgba(255,255,255,0.92)",
      capSpace: isUser && Number.isFinite(userCapSpace) ? userCapSpace : 6.0,
      retained: 0,
      capHit: isUser && Number.isFinite(userCapHit) ? userCapHit : 83.5,
      capLimit: isUser && Number.isFinite(userCapLimit) ? userCapLimit : 92.0,
      record: rec,
      status,
      need: needHintFromStatus(status),
    };
  });

  const players = {};
  const picks = {};
  organizations.forEach((org) => {
    const tid = String(org.team_id);
    const nm = String(org.name || tid);
    const abbr = inferLogoAbbr(tid, nm);
    players[tid] = (org.nhl || []).map((row) => apiSkaterToTradePlayer(row, tid));
    picks[tid] = syntheticPicksForTeam(tid, seasonYear, abbr);
  });

  return { teams, players, picks, userTeamId, seasonYear };
}

function emptyTradeForTeamIds(ids) {
  const o = {};
  ids.forEach((id) => {
    o[id] = [];
  });
  return o;
}

function inferredPpg(p) {
  if (p.ppg != null) return p.ppg;
  if (p.gp > 0 && p.points != null) return p.points / p.gp;
  return clamp(0.15 + (p.ovr - 78) * 0.055, 0.08, 1.65);
}

function defaultDefRating(p) {
  return clamp(4 + (p.ovr - 80) * 0.35, 1, 10);
}

function defRatingValue(p) {
  return p.defRating != null ? p.defRating : p.pos === "D" ? defaultDefRating(p) : 5;
}

function productionComponent(p) {
  const pos = p.pos;
  const ppg = inferredPpg(p);
  if (pos === "G") {
    const sv = p.svPct != null ? p.svPct : 0.888 + (p.ovr - 82) * 0.004;
    const gaa = p.gaa != null ? p.gaa : 3.05 - (p.ovr - 82) * 0.055;
    return (sv - 0.89) * 200 + (2.8 - gaa) * 5;
  }
  if (pos === "D") {
    const dr = defRatingValue(p);
    return 0.6 * ppg * 12 + dr * 2;
  }
  return ppg * 12;
}

function ageCurveModifier(age) {
  if (age <= 22) return clamp(5 + (22 - age) * 1.15, 5, 10);
  if (age <= 27) return clamp(3 + (27 - age) * 0.55, 3, 6);
  if (age <= 31) return clamp(0 - (age - 28) * 1, -3, 0);
  return clamp(-5 - (age - 32) * 1.15, -12, -5);
}

function expectedSalaryMillions(p) {
  const ppg = inferredPpg(p);
  const centerBump = p.pos === "C" ? 1.05 : 0;
  const wingMult = p.pos === "LW" || p.pos === "RW" ? 0.9 : 1;
  const defExtra = p.pos === "D" ? defRatingValue(p) * 0.18 : 0;
  const raw = (p.ovr - 75) * 0.5 + ppg * 7.6 * wingMult + centerBump + defExtra;
  let ageAdj = 0;
  if (p.age <= 22) ageAdj -= 1.15;
  else if (p.age >= 33) ageAdj -= 1.8 + (p.age - 33) * 0.35;
  else if (p.age >= 29) ageAdj -= (p.age - 28) * 0.28;
  return clamp(0.78 + raw * 0.58 + ageAdj, 0.85, 16.5);
}

function contractEfficiencyPoints(p) {
  return (expectedSalaryMillions(p) - p.salary) * 2;
}

function positionalValue(p) {
  if (p.pos === "C") return 3;
  if (p.pos === "D") {
    if (p.handed === "Right") return 3;
    if (p.handed === "Left") return 1;
  }
  if (p.pos === "G") {
    const perf = productionComponent(p);
    return clamp(perf * 0.35, -5, 5);
  }
  return 0;
}

function archetypeBonus(p) {
  const tags = (p.tags || []).map((t) => String(t).toLowerCase()).join(" ");
  let b = 0;
  if (tags.includes("passer") || tags.includes("playmaker")) b += 2;
  if (tags.includes("sniper") || (p.shootPct != null && p.shootPct >= 14)) b += 2;
  if (tags.includes("two-way") || tags.includes("selke") || tags.includes("defensive")) b += 3;
  if (tags.includes("grit") || tags.includes("playoff") || tags.includes("physical")) b += 2;
  return Math.min(5, b);
}

function clutchPoints(p) {
  if (p.clutch != null) return clamp(p.clutch, -2, 4);
  return 0;
}

function teamDirectionMultiplier(status) {
  if (!status) return 0;
  const s = status.toLowerCase();
  if (s.includes("sell")) return -1;
  if (s.includes("buy") || s.includes("contend")) return 1;
  if (s.includes("re-tool") || s.includes("retool")) return 0.5;
  return 0;
}

function marketDemandPoints(p, acquiringTeam, sendingTeam) {
  if (!acquiringTeam) return 0;
  const need = (acquiringTeam.need || "").toLowerCase();
  let pts = 0;
  if (/(center|centre)/.test(need) && p.pos === "C") pts += 4;
  if (/right-shot|right shot|rhd|r\.?\s*h\.?\s*d\.?/i.test(need) && p.pos === "D" && p.handed === "Right") pts += 5;
  if (/defense|defence|blue\s*line|back\s*end/.test(need) && p.pos === "D") pts += 3;
  if (/winger|wing|scoring|offence|offense/.test(need) && (p.pos === "LW" || p.pos === "RW")) pts += 2.5;
  if (/goalie|goaltender|net/.test(need) && p.pos === "G") pts += 4;
  if (/pick|draft|prospect/.test(need) && (p.age <= 22 || (p.tags || []).some((x) => /prospect|elc/i.test(x)))) pts += 3;

  const buyer = teamDirectionMultiplier(acquiringTeam.status);
  if (buyer > 0 && p.age <= 22 && (p.tags || []).some((x) => /prospect|elc/i.test(x))) pts -= 2;
  if (buyer > 0 && p.ovr >= 86 && p.salary >= 8) pts += 1.5;
  if (buyer < 0 && (p.age <= 22 || (p.tags || []).some((x) => /prospect/i.test(x)))) pts += 4;
  if (buyer < 0 && p.ovr >= 84 && p.age >= 26) pts -= 2.5;

  if (sendingTeam && sendingTeam.id !== acquiringTeam.id) {
    const sur = teamDirectionMultiplier(sendingTeam.status);
    if (sur < 0 && p.ovr >= 85 && (p.pos === "C" || p.pos === "D")) pts += 2;
  }

  return clamp(pts, -5, 6);
}

function ovrBase(ovr) {
  return (ovr - 70) * 0.8;
}

function retentionTradeBoost(retainedPct, fullSalary) {
  return (retainedPct / 100) * fullSalary * 0.5;
}

function computePlayerTradeValue(p, acquiringTeamId, sourceTeamId, td) {
  const acquiring = (td?.teams || []).find((t) => t.id === acquiringTeamId) || null;
  const sending = (td?.teams || []).find((t) => t.id === sourceTeamId) || null;
  const base = ovrBase(p.ovr);
  const production = productionComponent(p);
  const ageM = ageCurveModifier(p.age);
  const contract = contractEfficiencyPoints(p);
  const pos = positionalValue(p);
  const arch = archetypeBonus(p);
  const clutch = clutchPoints(p);
  const market = marketDemandPoints(p, acquiring, sending);
  const subtotal = base + production + ageM + contract + pos + arch + clutch + market;
  const total = Math.max(0.25, subtotal);
  return { base, production, ageM, contract, pos, arch, clutch, market, subtotal, total };
}

function roundBasePoints(round, firstBand) {
  if (round === 1) {
    if (firstBand === "overall_1") return 10;
    if (firstBand === "top5") return 9;
    if (firstBand === "mid") return 7;
    if (firstBand === "late") return 5;
    return 6;
  }
  if (round === 2) return 2.6;
  if (round === 3) return 1.25;
  if (round === 4) return 0.72;
  return clamp(0.52 - (round - 5) * 0.08, 0.12, 0.52);
}

function computePickTradeValue(pick, acquiringTeam, anchorYear) {
  const anchor = anchorYear != null ? anchorYear : TRADE_HUB_SEASON_YEAR;
  const year = pick.draftYear != null ? pick.draftYear : anchor;
  const round = pick.round != null ? pick.round : 4;
  const firstBand = pick.firstBand || null;
  let core = roundBasePoints(round, firstBand);
  if (round > 4) core = roundBasePoints(5, null);
  const yearsFuture = Math.max(0, year - anchor);
  let v = core * (1 - 0.1 * yearsFuture);
  if (acquiringTeam && /pick|draft|prospect/i.test(acquiringTeam.need || "")) v += 1.25;
  if (acquiringTeam && acquiringTeam.status && acquiringTeam.status.toLowerCase().includes("sell")) v += 0.55;
  return Math.max(0.15, v);
}

function ntcNotice(p, acquiringTeamId, sourceTeamId) {
  const prot = (p.protection || "").toLowerCase();
  if (!prot.includes("nmc") && !prot.includes("ntc")) return null;
  if (acquiringTeamId === sourceTeamId) return null;
  if (prot.includes("modified") || prot.includes("limited")) {
    return `${p.name}: ${p.protection} — verify destination is on the player's list.`;
  }
  if (prot.includes("full nmc") || (prot.includes("nmc") && !prot.includes("modified"))) {
    return `${p.name}: Full NMC — trade requires explicit approval / waiver documentation.`;
  }
  if (prot.includes("ntc")) {
    return `${p.name}: NTC — confirm destination approval.`;
  }
  return null;
}

function maxValueSpreadAllowed(nTeams) {
  if (nTeams <= 2) return 5;
  if (nTeams === 3) return 7;
  return 10;
}

function evaluateTradeValidity(visibleTeams, trade, salaryMatch, td) {
  const n = visibleTeams.length;
  const maxSpread = maxValueSpreadAllowed(n);
  const teamTotals = visibleTeams.map((team) => {
    const resolved = (trade[team.id] || []).map((a) => assetDetails(a, team.id, td));
    const value = resolved.reduce((s, x) => s + (x?.value || 0), 0);
    const cap = resolved.reduce((s, x) => s + (x?.capHit || 0), 0);
    return { team, value, cap, resolved };
  });
  const vals = teamTotals.map((t) => t.value);
  const spread = vals.length ? Math.max(...vals) - Math.min(...vals) : 0;
  const spreadOk = spread <= maxSpread;

  const capWarnings = [];
  if (salaryMatch) {
    teamTotals.forEach((row) => {
      if (row.cap > row.team.capSpace + 0.35) {
        capWarnings.push(
          `${row.team.id}: acquiring ${money(row.cap)} vs ${money(row.team.capSpace)} cap space (incoming only; salary out not netted).`,
        );
      }
    });
  }

  const ntcIssues = [];
  teamTotals.forEach((row) => {
    row.resolved.forEach((a) => {
      if (a?.assetType !== "player") return;
      const msg = ntcNotice(a, row.team.id, a.sourceTeam);
      if (msg) ntcIssues.push(msg);
    });
  });

  const directionIssues = [];
  visibleTeams.forEach((tm) => {
    if (!tm.status || !tm.status.toLowerCase().includes("sell")) return;
    const resolved = (trade[tm.id] || []).map((a) => assetDetails(a, tm.id, td));
    const vets = resolved.filter((x) => x?.assetType === "player" && x.age >= 28 && x.ovr >= 84).length;
    const picks = resolved.filter((x) => x?.assetType === "pick").length;
    if (vets >= 2 && picks === 0) {
      directionIssues.push(`${tm.id}: Selling teams usually want futures mixed into veteran-heavy packages.`);
    }
  });

  const valid = spreadOk;
  return { valid, spreadOk, spread, maxSpread, capWarnings, ntcIssues, directionIssues, teamTotals };
}

function getPlayer(id, teamId, td) {
  if (!td) return undefined;
  if (teamId) return (td.players[teamId] || []).find((p) => String(p.id) === String(id));
  return Object.values(td.players || {})
    .flat()
    .find((p) => String(p.id) === String(id));
}

function getPick(id, td) {
  if (!td) return undefined;
  return Object.values(td.picks || {})
    .flat()
    .find((p) => String(p.id) === String(id));
}

function assetDetails(asset, acquiringTeamId, td) {
  if (!td) {
    return {
      id: asset?.id,
      assetType: asset?.type === "pick" ? "pick" : "player",
      capHit: 0,
      value: 0,
      sourceTeam: asset?.team,
      tags: [],
    };
  }
  if (asset.type === "player") {
    const p = getPlayer(asset.id, asset.team, td);
    if (!p) {
      return {
        id: asset.id,
        name: "?",
        assetType: "player",
        capHit: 0,
        value: 0,
        retained: Number(asset.retained || 0),
        sourceTeam: asset.team,
        tags: [],
      };
    }
    const retained = Number(asset.retained || 0);
    const capHit = p.salary * (1 - retained / 100);
    const breakdown = computePlayerTradeValue(p, acquiringTeamId, asset.team, td);
    const retBoost = retentionTradeBoost(retained, p.salary);
    const value = Math.max(0.2, breakdown.total + retBoost);
    return {
      ...p,
      assetType: "player",
      capHit,
      value,
      valueBreakdown: { ...breakdown, retainedBoost: retBoost },
      retained,
      sourceTeam: asset.team,
    };
  }

  const pick = getPick(asset.id, td);
  if (!pick) {
    return {
      id: asset.id,
      label: "?",
      assetType: "pick",
      capHit: 0,
      retained: 0,
      value: 0,
      sourceTeam: asset.team,
    };
  }
  const acquiring = (td.teams || []).find((t) => t.id === acquiringTeamId) || null;
  const anchorY = td.seasonYear != null ? td.seasonYear : TRADE_HUB_SEASON_YEAR;
  const pickValue = computePickTradeValue(pick, acquiring, anchorY);
  return {
    ...pick,
    assetType: "pick",
    capHit: 0,
    retained: 0,
    value: pickValue,
    valueBreakdown: {
      pickCore: pickValue,
      yearsFuture: Math.max(0, (pick.draftYear || anchorY) - anchorY),
    },
    sourceTeam: asset.team,
  };
}

function TeamLogo({ team, size = 38 }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        minWidth: size,
        borderRadius: "50%",
        display: "grid",
        placeItems: "center",
        background: `radial-gradient(circle at 35% 25%, ${team.accent}55, ${team.color} 58%, #020617 100%)`,
        border: `1px solid ${team.accent}88`,
        boxShadow: `0 0 14px ${team.color}66`,
        fontFamily: "var(--g-font-head)",
        fontSize: size > 44 ? 13 : 10,
        fontWeight: 900,
        color: "#fff",
        letterSpacing: "-0.04em",
      }}
    >
      {team.logo}
    </div>
  );
}

function NHLMark() {
  return (
    <div
      style={{
        width: 34,
        height: 34,
        borderRadius: 8,
        border: "1px solid rgba(255,255,255,0.35)",
        display: "grid",
        placeItems: "center",
        transform: "skew(-8deg)",
        fontFamily: "var(--g-font-head)",
        fontWeight: 900,
        fontSize: 12,
        color: "#fff",
        background: "linear-gradient(145deg,#151b26,#05070c)",
      }}
    >
      NHL
    </div>
  );
}

function Toggle({ label, on, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        height: 42,
        minWidth: 190,
        padding: "0 12px",
        borderRadius: 6,
        border: `1px solid ${COLORS.line}`,
        background: "rgba(0,0,0,0.24)",
        color: COLORS.silver,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        cursor: "pointer",
        fontFamily: "var(--g-font-body)",
        fontWeight: 700,
      }}
    >
      <span>{label}</span>
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "4px 8px",
          borderRadius: 999,
          background: on ? "rgba(42,157,85,0.75)" : "rgba(120,120,120,0.35)",
          color: "#fff",
          fontSize: 11,
          fontFamily: "var(--g-font-head)",
        }}
      >
        {on ? "ON" : "OFF"}
        <span
          style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: "#fff",
            display: "inline-block",
          }}
        />
      </span>
    </button>
  );
}

function TopHeader({ selectedUserTeam, onBackToHub }) {
  return (
    <div
      style={{
        height: 74,
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 18px",
        borderBottom: `1px solid ${COLORS.line}`,
        background: "rgba(3,5,12,0.72)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {onBackToHub && (
          <button
            type="button"
            className="game-btn game-btn--sm"
            onClick={onBackToHub}
            style={{ marginRight: 2 }}
            title="Return to franchise hub"
          >
            ← HUB
          </button>
        )}
        <NHLMark />
        <div
          style={{
            fontFamily: "var(--g-font-head)",
            fontSize: 30,
            letterSpacing: "0.04em",
            fontWeight: 900,
          }}
        >
          TRADE HUB
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            height: 48,
            minWidth: 230,
            padding: "0 14px",
            display: "flex",
            alignItems: "center",
            gap: 10,
            border: `1px solid ${COLORS.line}`,
            borderRadius: 6,
            background: "rgba(0,0,0,0.20)",
          }}
        >
          <TeamLogo team={selectedUserTeam} size={34} />
          <div>
            <div style={{ fontSize: 13, fontWeight: 800 }}>{selectedUserTeam.name}</div>
            <div style={{ fontSize: 12, color: COLORS.dim }}>GM Mode</div>
          </div>
          <div style={{ marginLeft: "auto", color: COLORS.dim }}>⌄</div>
        </div>

        <div
          style={{
            height: 48,
            padding: "0 18px",
            display: "grid",
            placeItems: "center",
            border: `1px solid ${COLORS.line}`,
            borderRadius: 6,
            background: "rgba(0,0,0,0.20)",
            fontWeight: 800,
          }}
        >
          May 12, 2025
        </div>

        <div
          style={{
            height: 48,
            padding: "0 18px",
            border: `1px solid ${COLORS.line}`,
            borderRadius: 6,
            background: "rgba(0,0,0,0.20)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "flex-end",
          }}
        >
          <div style={{ fontSize: 11, color: COLORS.dim }}>Cap Space</div>
          <div style={{ fontSize: 16, color: "#4ade80", fontWeight: 900 }}>$8.71M</div>
        </div>

        <button className="game-btn game-btn--sm">☷</button>
      </div>
    </div>
  );
}

function Tabs({ active, setActive }) {
  const tabs = ["TRADE BUILDER", "FIND TRADE", "TRADE BOARD", "DRAFT BOARD", "TRADE HISTORY"];

  return (
    <div
      style={{
        height: 40,
        display: "flex",
        alignItems: "stretch",
        borderBottom: `1px solid ${COLORS.line}`,
        background: "rgba(7,13,28,0.78)",
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab}
          onClick={() => setActive(tab)}
          style={{
            width: 170,
            border: 0,
            borderBottom: active === tab ? `3px solid #1d7cff` : "3px solid transparent",
            background: "transparent",
            color: active === tab ? "#fff" : COLORS.dim,
            fontFamily: "var(--g-font-head)",
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

function TeamTradeCard({
  team,
  td,
  userTeamId,
  assets,
  selectedAssetKey,
  setSelectedAssetKey,
  requestRemoveAsset,
  openAddAssetForTeam,
}) {
  const resolved = assets.map((a) => assetDetails(a, team.id, td));
  const totalValue = resolved.reduce((sum, a) => sum + a.value, 0);
  const totalCap = resolved.reduce((sum, a) => sum + a.capHit, 0);
  const players = resolved.filter((a) => a.assetType === "player").length;
  const picks = resolved.filter((a) => a.assetType === "pick").length;
  const selling = team.status === "Selling";

  return (
    <div
      style={{
        borderRadius: 8,
        border: `1px solid ${COLORS.line}`,
        overflow: "hidden",
        background: COLORS.panel,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          height: 56,
          background: selling
            ? `linear-gradient(90deg, ${COLORS.red}, ${COLORS.red2})`
            : `linear-gradient(90deg, ${COLORS.blue}, ${COLORS.blue2})`,
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "0 14px",
          borderBottom: `1px solid ${COLORS.line}`,
        }}
      >
        <TeamLogo team={team} size={34} />
        <div
          style={{
            fontFamily: "var(--g-font-head)",
            fontSize: 15,
            letterSpacing: "0.04em",
            fontWeight: 900,
          }}
        >
          {team.name.toUpperCase()}
          {String(team.id) === String(userTeamId) && (
            <span
              style={{
                marginLeft: 8,
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 4,
                background: "rgba(42,157,85,0.35)",
                border: "1px solid rgba(74,222,128,0.5)",
              }}
            >
              YOUR CLUB
            </span>
          )}
        </div>
      </div>

      <div
        style={{
          padding: "12px 14px",
          display: "grid",
          gridTemplateColumns: "1fr auto",
          gap: "4px 12px",
          fontSize: 12,
          color: COLORS.silver,
          borderBottom: `1px solid ${COLORS.line}`,
        }}
      >
        <span>Cap Space</span>
        <strong>{money(team.capSpace)}</strong>
        <span>Retained Salary</span>
        <strong>{money(team.retained)}</strong>
        <span>Team Cap Hit</span>
        <strong>
          {money(team.capHit)} / {money(team.capLimit)}
        </strong>
      </div>

      <div
        style={{
          height: 26,
          padding: "0 14px",
          display: "flex",
          alignItems: "center",
          fontSize: 11,
          fontFamily: "var(--g-font-head)",
          color: COLORS.silver,
          background: "rgba(0,0,0,0.25)",
          borderBottom: `1px solid ${COLORS.line}`,
        }}
      >
        ACQUIRING
      </div>

      <div style={{ padding: 8, display: "flex", flexDirection: "column", gap: 6, flex: 1 }}>
        {resolved.map((asset) => {
          const key = `${team.id}-${asset.assetType}-${asset.id}`;
          const sourceTeam = (td?.teams || []).find((t) => t.id === asset.sourceTeam) || team;
          const canStrip =
            String(team.id) === String(userTeamId) ||
            String(asset.sourceTeam) === String(userTeamId) ||
            String(asset.sourceTeam) === String(team.id);
          const selected = key === selectedAssetKey;

          return (
            <button
              key={key}
              onClick={() => setSelectedAssetKey(key)}
              style={{
                width: "100%",
                minHeight: 58,
                borderRadius: 6,
                border: selected ? "1px solid #1797ff" : `1px solid ${COLORS.line}`,
                background: selected ? "rgba(23,151,255,0.16)" : "rgba(0,0,0,0.16)",
                color: COLORS.text,
                display: "grid",
                gridTemplateColumns: "42px 1fr auto 24px",
                alignItems: "center",
                gap: 8,
                padding: "7px 8px",
                cursor: "pointer",
                textAlign: "left",
              }}
            >
              <TeamLogo team={sourceTeam} size={30} />

              {asset.assetType === "player" ? (
                <div>
                  <div style={{ fontSize: 13, fontWeight: 900 }}>{asset.name}</div>
                  <div style={{ fontSize: 11, color: COLORS.dim }}>
                    {asset.pos} · Age: {asset.age} · OVR: {asset.ovr}
                  </div>
                  {asset.retained > 0 && (
                    <div
                      style={{
                        marginTop: 3,
                        width: "fit-content",
                        padding: "1px 5px",
                        borderRadius: 3,
                        border: "1px solid #1d7cff",
                        color: "#7ec8ff",
                        fontSize: 9,
                        fontFamily: "var(--g-font-head)",
                      }}
                    >
                      {asset.retained}% RETAINED
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 13, fontWeight: 900 }}>{asset.label}</div>
                  <div style={{ fontSize: 11, color: COLORS.dim }}>
                    Original Owner: {(td?.teams || []).find((t) => t.id === asset.owner)?.short || asset.owner}
                  </div>
                </div>
              )}

              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 13, fontWeight: 900 }}>
                  {asset.assetType === "player" ? money(asset.salary) : asset.owner}
                </div>
                <div style={{ fontSize: 11, color: COLORS.dim }}>
                  {asset.assetType === "player" ? yearsFmt(asset.years) : "Pick"}
                </div>
              </div>

              <span
                onClick={(e) => {
                  e.stopPropagation();
                  if (!canStrip) return;
                  requestRemoveAsset(team.id, asset.assetType, asset.id);
                }}
                style={{
                  color: canStrip ? COLORS.dim : "rgba(255,255,255,0.12)",
                  fontSize: 20,
                  textAlign: "center",
                  cursor: canStrip ? "pointer" : "default",
                }}
                title={
            canStrip
              ? "Remove from trade"
              : "Only assets sourced from your club or this column's team can be removed here"
          }
              >
                ×
              </span>
            </button>
          );
        })}

        <button
          onClick={() => openAddAssetForTeam(team.id)}
          style={{
            height: 44,
            borderRadius: 6,
            border: "1px dashed rgba(255,255,255,0.22)",
            background: "rgba(0,0,0,0.16)",
            color: COLORS.silver,
            fontFamily: "var(--g-font-head)",
            cursor: "pointer",
          }}
          title={
            String(team.id) === String(userTeamId)
              ? "Add players or picks you want to acquire (any NHL club)"
              : "Add what this team sends: your roster or their own roster / picks"
          }
        >
          ⊕ ADD ASSET
        </button>
      </div>

      <div
        style={{
          margin: "0 8px 10px",
          borderRadius: 5,
          overflow: "hidden",
          border: `1px solid ${COLORS.line}`,
          background: selling
            ? "linear-gradient(180deg,rgba(123,19,32,0.95),rgba(79,11,20,0.95))"
            : "linear-gradient(180deg,rgba(11,59,145,0.95),rgba(8,43,104,0.95))",
        }}
      >
        {[
          ["TOTAL TRADE VALUE", valueFmt(totalValue)],
          ["TOTAL CAP HIT", money(totalCap)],
          ["PLAYERS", players],
          ["PICKS", picks],
        ].map(([k, v]) => (
          <div
            key={k}
            style={{
              height: 29,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0 12px",
              borderBottom: `1px solid ${COLORS.line}`,
              fontFamily: "var(--g-font-head)",
              fontSize: 12,
            }}
          >
            <span>{k}</span>
            <span>{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssetsPanel({ team, td, userTeamId, openAddSpecific }) {
  const prospects = (td?.players?.[team.id] || [])
    .filter((p) => p.age <= 24 || (p.tags || []).some((x) => /prospect|elc/i.test(x)))
    .slice(0, 6);

  const picks = td?.picks?.[team.id] || [];

  return (
    <div
      style={{
        borderRadius: 8,
        border: `1px solid ${COLORS.line}`,
        overflow: "hidden",
        background: COLORS.panel,
        minHeight: 238,
      }}
    >
      <div
        style={{
          height: 38,
          background:
            team.status === "Selling"
              ? `linear-gradient(90deg, ${COLORS.red}, ${COLORS.red2})`
              : `linear-gradient(90deg, ${COLORS.blue}, ${COLORS.blue2})`,
          display: "flex",
          alignItems: "center",
          padding: "0 12px",
          fontFamily: "var(--g-font-head)",
          fontSize: 13,
          fontWeight: 900,
        }}
      >
        {team.name.toUpperCase()} ASSETS
      </div>

      <div style={{ padding: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 800 }}>
          <span>Prospects</span>
          <span>{prospects.length + 22}</span>
        </div>

        <div style={{ marginTop: 10, fontFamily: "var(--g-font-head)", fontSize: 12 }}>
          Top Prospects
        </div>

        <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
          {prospects.map((p) => (
            <button
              key={p.id}
              onClick={() => openAddSpecific(team.id, { type: "player", id: p.id, team: team.id, retained: 0 })}
              style={{
                height: 34,
                border: `1px solid ${COLORS.line}`,
                borderRadius: 5,
                background: "rgba(0,0,0,0.18)",
                color: COLORS.silver,
                display: "grid",
                gridTemplateColumns: "28px 1fr 32px 58px",
                alignItems: "center",
                gap: 6,
                padding: "0 8px",
                cursor: "pointer",
                textAlign: "left",
              }}
            >
              <span style={{ fontSize: 18 }}>👤</span>
              <span style={{ fontSize: 12 }}>{p.fullName}</span>
              <span style={{ fontSize: 11, color: COLORS.dim }}>{p.pos}</span>
              <span style={{ fontSize: 11, color: COLORS.dim }}>OVR: {p.ovr}</span>
            </button>
          ))}
        </div>

        <div style={{ marginTop: 10, fontFamily: "var(--g-font-head)", fontSize: 12 }}>
          Draft Picks
        </div>

        <div style={{ marginTop: 6, fontSize: 12, color: COLORS.dim, lineHeight: 1.55 }}>
          {picks.slice(0, 4).map((p) => (
            <div key={p.id} style={{ display: "flex", justifyContent: "space-between" }}>
              <span>{p.label}</span>
              <span>{p.owner}</span>
            </div>
          ))}
        </div>

        <button
          onClick={() => openAddSpecific(team.id, picks[0] ? { type: "pick", id: picks[0].id, team: team.id } : null)}
          style={{
            marginTop: 10,
            width: "100%",
            height: 34,
            borderRadius: 5,
            border: `1px solid ${COLORS.line}`,
            background: "rgba(255,255,255,0.06)",
            color: COLORS.silver,
            fontFamily: "var(--g-font-head)",
            cursor: "pointer",
          }}
        >
          VIEW ALL
        </button>
      </div>
    </div>
  );
}

function PlayerInfo({ selectedAsset, td }) {
  if (!selectedAsset || selectedAsset.assetType !== "player") {
    return (
      <div style={{ padding: 18, color: COLORS.dim, fontSize: 13 }}>
        Select a player asset to view contract, trade value, protection, projection, and fit notes.
      </div>
    );
  }

  const sourceTeam =
    (td?.teams || []).find((t) => t.id === selectedAsset.sourceTeam) ||
    ({
      id: selectedAsset.sourceTeam,
      name: String(selectedAsset.sourceTeam),
      short: String(selectedAsset.sourceTeam),
      logo: inferLogoAbbr(selectedAsset.sourceTeam, ""),
      color: `hsl(${teamHueFromId(selectedAsset.sourceTeam)}, 40%, 28%)`,
      accent: "#fff",
    });

  const anchorSeason = td?.seasonYear ?? TRADE_HUB_SEASON_YEAR;

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <TeamLogo team={sourceTeam} size={54} />
        <div>
          <div
            style={{
              fontFamily: "var(--g-font-head)",
              fontSize: 18,
              letterSpacing: "0.04em",
            }}
          >
            {selectedAsset.name}
          </div>
          <div style={{ color: COLORS.dim, fontSize: 12 }}>
            {selectedAsset.pos} · Age {selectedAsset.age} · OVR {selectedAsset.ovr}
          </div>
          <div style={{ color: COLORS.dim, fontSize: 12 }}>
            {selectedAsset.height} · {selectedAsset.weight} · {selectedAsset.handed}
          </div>
        </div>
        <div style={{ marginLeft: "auto", textAlign: "right" }}>
          <div style={{ fontWeight: 900 }}>{money(selectedAsset.capHit)}</div>
          <div style={{ fontSize: 11, color: COLORS.dim }}>CAP HIT</div>
        </div>
      </div>

      {selectedAsset.retained > 0 && (
        <div
          style={{
            marginTop: 12,
            padding: "5px 8px",
            width: "fit-content",
            borderRadius: 4,
            border: "1px solid #1d7cff",
            color: "#7ec8ff",
            fontSize: 11,
            fontFamily: "var(--g-font-head)",
          }}
        >
          {selectedAsset.retained}% RETAINED BY {sourceTeam.id}
        </div>
      )}

      <div style={{ marginTop: 18, display: "grid", gridTemplateColumns: "1fr auto", gap: "12px 18px" }}>
        <span>Trade Value</span>
        <strong style={{ fontSize: 20 }}>{valueFmt(selectedAsset.value)} Ⓥ</strong>
        <span>Contract</span>
        <span>{yearsFmt(selectedAsset.years)} Remaining</span>
        <span>Salary</span>
        <span>{money(selectedAsset.salary)}</span>
        <span>Cap Hit</span>
        <span>{money(selectedAsset.capHit)}</span>
        <span>Trade Protection</span>
        <span>{selectedAsset.protection}</span>
      </div>

      {selectedAsset.valueBreakdown && selectedAsset.assetType === "player" && (
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 8,
            border: `1px solid ${COLORS.line}`,
            background: "rgba(0,0,0,0.22)",
            fontSize: 11,
            color: COLORS.silver,
            lineHeight: 1.55,
          }}
        >
          <div style={{ fontWeight: 900, marginBottom: 8, color: COLORS.text }}>Value breakdown (Ⓥ)</div>
          {[
            ["OVR base", selectedAsset.valueBreakdown.base],
            ["Production", selectedAsset.valueBreakdown.production],
            ["Age curve", selectedAsset.valueBreakdown.ageM],
            ["Contract efficiency", selectedAsset.valueBreakdown.contract],
            ["Position", selectedAsset.valueBreakdown.pos],
            ["Archetype", selectedAsset.valueBreakdown.arch],
            ["Clutch / playoffs", selectedAsset.valueBreakdown.clutch],
            ["Market / team fit", selectedAsset.valueBreakdown.market],
            ["Retention boost", selectedAsset.valueBreakdown.retainedBoost],
          ].map(([label, v]) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
              <span>{label}</span>
              <span style={{ fontFamily: "var(--g-font-head)", color: v >= 0 ? "#86efac" : "#fca5a5" }}>
                {v >= 0 ? "+" : ""}
                {valueFmt(v)}
              </span>
            </div>
          ))}
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${COLORS.line}`, display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontWeight: 800 }}>Total</span>
            <strong>{valueFmt(selectedAsset.value)}</strong>
          </div>
        </div>
      )}

      {selectedAsset.valueBreakdown && selectedAsset.assetType === "pick" && (
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 8,
            border: `1px solid ${COLORS.line}`,
            background: "rgba(0,0,0,0.22)",
            fontSize: 11,
            color: COLORS.silver,
          }}
        >
          <div style={{ fontWeight: 900, marginBottom: 6, color: COLORS.text }}>Pick value</div>
          <div>
            Slot tier (1st overall / top 5 / mid / late 1st, then 2nd–4th+), −10% Ⓥ per draft year after {anchorSeason},
            plus demand if the acquiring team wants picks/prospects.
          </div>
          <div style={{ marginTop: 6 }}>
            Years ahead of anchor: <strong>{selectedAsset.valueBreakdown.yearsFuture ?? 0}</strong> · Total Ⓥ:{" "}
            <strong>{valueFmt(selectedAsset.value)}</strong>
          </div>
        </div>
      )}

      <div
        style={{
          marginTop: 18,
          paddingTop: 14,
          borderTop: `1px solid ${COLORS.line}`,
        }}
      >
        <div style={{ fontWeight: 900, marginBottom: 6 }}>Trade Projection Details</div>
        <p style={{ margin: 0, color: COLORS.silver, fontSize: 13, lineHeight: 1.45 }}>
          {selectedAsset.projection}
        </p>
      </div>

      <div style={{ marginTop: 14, display: "flex", flexWrap: "wrap", gap: 6 }}>
        {(selectedAsset.tags || []).map((tag) => (
          <span
            key={tag}
            style={{
              padding: "4px 8px",
              borderRadius: 999,
              background: "rgba(56,189,248,0.12)",
              border: "1px solid rgba(56,189,248,0.25)",
              color: "#bae6fd",
              fontSize: 11,
            }}
          >
            {tag}
          </span>
        ))}
      </div>

      <button
        style={{
          marginTop: 18,
          width: "100%",
          height: 36,
          borderRadius: 5,
          border: `1px solid ${COLORS.line}`,
          background: "rgba(255,255,255,0.06)",
          color: COLORS.silver,
          fontFamily: "var(--g-font-head)",
        }}
      >
        FULL PLAYER PROFILE
      </button>
    </div>
  );
}

function TradeSummary({ teams, trade, salaryMatch, td }) {
  const allowance = maxValueSpreadAllowed(teams.length);
  const evaluation = evaluateTradeValidity(teams, trade, salaryMatch, td);

  const teamTotals = teams.map((team) => {
    const resolved = (trade[team.id] || []).map((a) => assetDetails(a, team.id, td));
    return {
      team,
      value: resolved.reduce((sum, a) => sum + a.value, 0),
      cap: resolved.reduce((sum, a) => sum + a.capHit, 0),
      players: resolved.filter((a) => a.assetType === "player").length,
      picks: resolved.filter((a) => a.assetType === "pick").length,
    };
  });

  const highest = Math.max(...teamTotals.map((t) => t.value), 0.001);
  const lowest = Math.min(...teamTotals.map((t) => t.value));
  const spread = highest - lowest;
  const spreadOk = spread <= allowance;

  return (
    <div style={{ padding: 16 }}>
      <div style={{ fontWeight: 900, marginBottom: 10 }}>Trade Balance</div>

      {teamTotals.map((row) => (
        <div key={row.team.id} style={{ marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
            <span>{row.team.short}</span>
            <strong>{valueFmt(row.value)} Ⓥ</strong>
          </div>
          <div
            style={{
              height: 8,
              marginTop: 4,
              borderRadius: 999,
              background: "rgba(255,255,255,0.08)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${Math.max(8, (row.value / highest) * 100)}%`,
                background: row.team.status === "Selling" ? COLORS.red : "#1d7cff",
              }}
            />
          </div>
        </div>
      ))}

      <div
        style={{
          marginTop: 10,
          padding: "8px 10px",
          borderRadius: 6,
          background: "rgba(0,0,0,0.2)",
          border: `1px solid ${COLORS.line}`,
          fontSize: 11,
          color: COLORS.silver,
          lineHeight: 1.45,
        }}
      >
        Max value gap for {teams.length}-team framework: <strong>{valueFmt(allowance)} Ⓥ</strong> · Current gap:{" "}
        <strong>{valueFmt(spread)} Ⓥ</strong>
      </div>

      <div
        style={{
          marginTop: 14,
          padding: 12,
          borderRadius: 8,
          background: spreadOk ? "rgba(42,157,85,0.13)" : "rgba(224,112,32,0.13)",
          border: spreadOk ? "1px solid rgba(42,157,85,0.4)" : "1px solid rgba(224,112,32,0.4)",
          fontSize: 13,
          lineHeight: 1.45,
        }}
      >
        {spreadOk
          ? `Value spread is within the ${teams.length}-team allowance (${valueFmt(allowance)} Ⓥ).`
          : `Value gap exceeds the ${teams.length}-team limit (${valueFmt(allowance)} Ⓥ). Add picks, retention, or change the mix.`}
      </div>

      {evaluation.capWarnings.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, color: "#fdba74", lineHeight: 1.5 }}>
          {evaluation.capWarnings.map((x) => (
            <div key={x}>Cap note: {x}</div>
          ))}
        </div>
      )}

      {evaluation.ntcIssues.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, color: "#fcd34d", lineHeight: 1.5 }}>
          {evaluation.ntcIssues.map((x) => (
            <div key={x}>⚠ {x}</div>
          ))}
        </div>
      )}

      {evaluation.directionIssues.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 12, color: COLORS.dim, lineHeight: 1.5 }}>
          {evaluation.directionIssues.map((x) => (
            <div key={x}>{x}</div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 14, fontSize: 12, color: COLORS.dim, lineHeight: 1.55 }}>
        Values use OVR, production, age curve, contract efficiency vs fair AAV, position, archetypes, clutch, market fit
        for the acquiring club, and pick tier with future-year discount. Retention adds (retained % × salary × 0.5) in Ⓥ.
      </div>
    </div>
  );
}

function SidePanel({
  selectedAsset,
  showSummary,
  setShowSummary,
  teams,
  trade,
  salaryMatch,
  td,
  proposalNotice,
  onSubmitTrade,
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 0 }}>
      <div
        style={{
          borderRadius: 8,
          border: `1px solid ${COLORS.line}`,
          background: COLORS.panel,
          overflow: "hidden",
          flex: 1,
          minHeight: 0,
        }}
      >
        <div
          style={{
            height: 44,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            borderBottom: `1px solid ${COLORS.line}`,
          }}
        >
          <button
            onClick={() => setShowSummary(false)}
            style={{
              border: 0,
              borderBottom: !showSummary ? "3px solid #1d7cff" : "3px solid transparent",
              background: "rgba(255,255,255,0.03)",
              color: !showSummary ? "#fff" : COLORS.dim,
              fontFamily: "var(--g-font-head)",
              cursor: "pointer",
            }}
          >
            PLAYER INFO
          </button>
          <button
            onClick={() => setShowSummary(true)}
            style={{
              border: 0,
              borderBottom: showSummary ? "3px solid #1d7cff" : "3px solid transparent",
              background: "rgba(255,255,255,0.03)",
              color: showSummary ? "#fff" : COLORS.dim,
              fontFamily: "var(--g-font-head)",
              cursor: "pointer",
            }}
          >
            TRADE SUMMARY
          </button>
        </div>

        {showSummary ? (
          <TradeSummary teams={teams} trade={trade} salaryMatch={salaryMatch} td={td} />
        ) : (
          <PlayerInfo selectedAsset={selectedAsset} td={td} />
        )}
      </div>

      <div
        style={{
          borderRadius: 8,
          border: `1px solid ${COLORS.line}`,
          background: COLORS.panel,
          padding: 16,
        }}
      >
        <div
          style={{
            fontFamily: "var(--g-font-head)",
            marginBottom: 12,
            letterSpacing: "0.06em",
          }}
        >
          QUICK OPTIONS
        </div>

        {proposalNotice && (
          <div
            style={{
              marginBottom: 10,
              padding: "10px 12px",
              borderRadius: 6,
              fontSize: 12,
              lineHeight: 1.45,
              border:
                proposalNotice.kind === "success"
                  ? "1px solid rgba(74,222,128,0.55)"
                  : "1px solid rgba(248,113,113,0.55)",
              background:
                proposalNotice.kind === "success" ? "rgba(22,101,52,0.28)" : "rgba(127,29,29,0.28)",
              color: proposalNotice.kind === "success" ? "#bbf7d0" : "#fecaca",
            }}
          >
            {proposalNotice.message}
          </div>
        )}

        {["♙ Add Player", "♧ Add Draft Pick", "♨ Retain Salary", "✖ Remove Asset", "♲ Clear Trade"].map((x) => (
          <button
            key={x}
            style={{
              width: "100%",
              height: 32,
              marginBottom: 6,
              borderRadius: 5,
              border: `1px solid ${COLORS.line}`,
              background: "rgba(255,255,255,0.05)",
              color: COLORS.silver,
              textAlign: "left",
              padding: "0 12px",
              cursor: "pointer",
            }}
          >
            {x}
          </button>
        ))}

        <button
          type="button"
          onClick={onSubmitTrade}
          style={{
            width: "100%",
            height: 38,
            marginTop: 6,
            borderRadius: 5,
            border: "1px solid rgba(74,222,128,0.5)",
            background: "linear-gradient(180deg,rgba(42,157,85,0.95),rgba(24,102,55,0.95))",
            color: "#eaffef",
            fontFamily: "var(--g-font-head)",
            cursor: "pointer",
          }}
        >
          SUBMIT TRADE PROPOSAL
        </button>
      </div>
    </div>
  );
}

function AddAssetModal({ target, td, close, addAsset }) {
  const userTeamId = td?.userTeamId;
  const acquiringId = target?.acquiringTeamId;
  const mode = target?.mode === "counterparty" ? "counterparty" : "incoming";
  const acquiringTeam = (td?.teams || []).find((t) => t.id === acquiringId);

  const [sourceTeamId, setSourceTeamId] = useState(String(userTeamId || ""));

  useEffect(() => {
    if (!td || !acquiringId || !userTeamId) return;
    if (mode === "counterparty") {
      setSourceTeamId(String(acquiringId));
      return;
    }
    const firstOther = (td.teams || []).find((t) => t.id !== acquiringId);
    setSourceTeamId(String(firstOther?.id || userTeamId));
  }, [target, td, mode, acquiringId, userTeamId]);

  if (!target || !td || !acquiringId || !userTeamId) return null;

  const sourceTeam = (td.teams || []).find((t) => t.id === sourceTeamId) || acquiringTeam;
  const players = (td.players || {})[sourceTeamId] || [];
  const picks = (td.picks || {})[sourceTeamId] || [];
  const acqLabel = acquiringTeam?.name || acquiringId;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 5000,
        background: "rgba(0,0,0,0.68)",
        display: "grid",
        placeItems: "center",
        padding: 20,
      }}
    >
      <div
        style={{
          width: "min(860px, 100%)",
          maxHeight: "86vh",
          overflow: "hidden",
          borderRadius: 12,
          border: `1px solid ${COLORS.line}`,
          background: "linear-gradient(160deg,#101827,#05070c)",
          boxShadow: "0 30px 90px rgba(0,0,0,0.65)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            height: 58,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 16px",
            borderBottom: `1px solid ${COLORS.line}`,
          }}
        >
          <div>
            <div style={{ fontFamily: "var(--g-font-head)", fontSize: 17 }}>ADD ASSET</div>
            <div style={{ color: COLORS.dim, fontSize: 12 }}>
              {mode === "incoming"
                ? `Acquire into ${acqLabel} (any NHL roster)`
                : `Add to ${acqLabel}'s side — your outgoing or their own roster`}
            </div>
          </div>
          <button type="button" className="game-btn game-btn--sm" onClick={close}>
            CLOSE
          </button>
        </div>

        <div style={{ padding: 16, borderBottom: `1px solid ${COLORS.line}` }}>
          {mode === "incoming" ? (
            <>
              <label style={{ fontSize: 12, color: COLORS.dim, marginRight: 10 }}>Source club</label>
              <select
                value={sourceTeamId}
                onChange={(e) => setSourceTeamId(e.target.value)}
                className="hub-select"
                style={{ minWidth: 260 }}
              >
                {(td.teams || []).map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <>
              <label style={{ fontSize: 12, color: COLORS.dim, marginRight: 10 }}>Asset from</label>
              <select
                value={sourceTeamId}
                onChange={(e) => setSourceTeamId(e.target.value)}
                className="hub-select"
                style={{ minWidth: 280 }}
              >
                <option value={String(userTeamId)}>
                  {(td.teams || []).find((t) => String(t.id) === String(userTeamId))?.name || "Your club"} — you send
                </option>
                <option value={String(acquiringId)}>{acqLabel} — their roster / picks</option>
              </select>
            </>
          )}
        </div>

        <div
          style={{
            padding: 16,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
            overflow: "auto",
          }}
        >
          <div>
            <div style={{ fontFamily: "var(--g-font-head)", marginBottom: 8 }}>PLAYERS</div>
            {players.map((p) => (
              <button
                type="button"
                key={p.id}
                onClick={() => addAsset(acquiringId, { type: "player", id: p.id, team: sourceTeamId, retained: 0 })}
                style={{
                  width: "100%",
                  height: 44,
                  marginBottom: 6,
                  borderRadius: 6,
                  border: `1px solid ${COLORS.line}`,
                  background: "rgba(255,255,255,0.04)",
                  color: COLORS.silver,
                  display: "grid",
                  gridTemplateColumns: "34px 1fr auto",
                  alignItems: "center",
                  gap: 8,
                  padding: "0 10px",
                  textAlign: "left",
                  cursor: "pointer",
                }}
              >
                <TeamLogo team={sourceTeam} size={26} />
                <span>
                  <strong>{p.name}</strong>{" "}
                  <span style={{ color: COLORS.dim }}>
                    {p.pos} · OVR {p.ovr}
                  </span>
                </span>
                <span>
                  {valueFmt(
                    assetDetails(
                      { type: "player", id: p.id, team: sourceTeamId, retained: 0 },
                      acquiringId,
                      td,
                    ).value,
                  )}{" "}
                  Ⓥ
                </span>
              </button>
            ))}
          </div>

          <div>
            <div style={{ fontFamily: "var(--g-font-head)", marginBottom: 8 }}>DRAFT PICKS</div>
            {picks.map((p) => (
              <button
                type="button"
                key={p.id}
                onClick={() => addAsset(acquiringId, { type: "pick", id: p.id, team: sourceTeamId })}
                style={{
                  width: "100%",
                  height: 44,
                  marginBottom: 6,
                  borderRadius: 6,
                  border: `1px solid ${COLORS.line}`,
                  background: "rgba(255,255,255,0.04)",
                  color: COLORS.silver,
                  display: "grid",
                  gridTemplateColumns: "34px 1fr auto",
                  alignItems: "center",
                  gap: 8,
                  padding: "0 10px",
                  textAlign: "left",
                  cursor: "pointer",
                }}
              >
                <span style={{ fontSize: 20 }}>▣</span>
                <span>
                  <strong>{p.label}</strong>{" "}
                  <span style={{ color: COLORS.dim }}>
                    Owner: {(td.teams || []).find((t) => t.id === p.owner)?.short || p.owner}
                  </span>
                </span>
                <span>
                  {valueFmt(assetDetails({ type: "pick", id: p.id, team: sourceTeamId }, acquiringId, td).value)} Ⓥ
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function TradePartnerPicker({ td, userTeamId, numberOfTeams, partnerSlotIds, setPartnerSlotIds }) {
  const others = (td?.teams || []).filter((t) => String(t.id) !== String(userTeamId));
  const slots = Math.max(0, numberOfTeams - 1);
  if (!others.length || slots === 0) return null;

  return (
    <div
      style={{
        flexShrink: 0,
        padding: "8px 24px 10px",
        borderBottom: `1px solid ${COLORS.line}`,
        background: "rgba(3,5,12,0.55)",
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 12,
      }}
    >
      <span style={{ fontFamily: "var(--g-font-head)", fontSize: 12, color: COLORS.silver, letterSpacing: "0.06em" }}>
        TRADE PARTNERS
      </span>
      {Array.from({ length: slots }, (_, i) => (
        <label key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: COLORS.dim }}>
          <span>Slot {i + 1}</span>
          <select
            className="hub-select"
            style={{ minWidth: 200 }}
            value={partnerSlotIds[i] || ""}
            onChange={(e) => {
              const v = e.target.value;
              setPartnerSlotIds((prev) => {
                const next = [...prev];
                while (next.length < slots) next.push("");
                const j = next.findIndex((x, idx) => x === v && idx !== i);
                if (j >= 0 && v) [next[i], next[j]] = [next[j], next[i]];
                else next[i] = v;
                return next;
              });
            }}
          >
            {others.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
      ))}
    </div>
  );
}

function BuilderControls({
  numberOfTeams,
  setNumberOfTeams,
  salaryRetention,
  setSalaryRetention,
  salaryMatch,
  setSalaryMatch,
  tradeValidity,
}) {
  return (
    <div
      style={{
        height: 80,
        flexShrink: 0,
        display: "grid",
        gridTemplateColumns: "300px 280px 1fr 260px",
        gap: 18,
        alignItems: "center",
        padding: "0 24px",
      }}
    >
      <div>
        <div style={{ fontSize: 11, color: COLORS.dim, fontFamily: "var(--g-font-head)", marginBottom: 7 }}>
          TRADE TYPE
        </div>
        <div style={{ display: "flex" }}>
          <button
            style={{
              height: 36,
              padding: "0 20px",
              border: "1px solid #1d7cff",
              background: "#123f91",
              color: "#fff",
              borderRadius: "5px 0 0 5px",
              fontFamily: "var(--g-font-head)",
            }}
          >
            Standard Trade
          </button>
          <button
            style={{
              height: 36,
              padding: "0 20px",
              border: `1px solid ${COLORS.line}`,
              background: "rgba(0,0,0,0.22)",
              color: COLORS.silver,
              borderRadius: "0 5px 5px 0",
              fontFamily: "var(--g-font-head)",
            }}
          >
            Future Considerations
          </button>
        </div>
      </div>

      <div>
        <div style={{ fontSize: 11, color: COLORS.dim, fontFamily: "var(--g-font-head)", marginBottom: 7 }}>
          NUMBER OF TEAMS
        </div>
        <div style={{ display: "flex" }}>
          {[2, 3, 4].map((n) => (
            <button
              key={n}
              onClick={() => setNumberOfTeams(n)}
              style={{
                width: 74,
                height: 36,
                border: numberOfTeams === n ? "1px solid #1d7cff" : `1px solid ${COLORS.line}`,
                background: numberOfTeams === n ? "#1264d8" : "rgba(0,0,0,0.24)",
                color: "#fff",
                fontFamily: "var(--g-font-head)",
                cursor: "pointer",
              }}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
        <Toggle label="Salary Retention" on={salaryRetention} onClick={() => setSalaryRetention((v) => !v)} />
        <Toggle label="Salary Match" on={salaryMatch} onClick={() => setSalaryMatch((v) => !v)} />
      </div>

      <div>
        <div style={{ fontSize: 11, color: COLORS.dim, fontFamily: "var(--g-font-head)", marginBottom: 7 }}>
          TRADE VALIDITY
        </div>
        {tradeValidity?.valid ? (
          <div style={{ color: "#4ade80", fontWeight: 900 }}>✓ Trade is Valid</div>
        ) : (
          <div style={{ color: "#fca5a5", fontWeight: 800, fontSize: 12, lineHeight: 1.4 }}>
            ✕ Needs adjustment
            {!tradeValidity?.spreadOk && (
              <div style={{ marginTop: 4, color: COLORS.dim, fontWeight: 600 }}>
                Value gap {valueFmt(tradeValidity.spread)} Ⓥ exceeds max {valueFmt(tradeValidity.maxSpread)} Ⓥ for{" "}
                {numberOfTeams} teams.
              </div>
            )}
          </div>
        )}
        {tradeValidity?.valid && tradeValidity?.ntcIssues?.length > 0 && (
          <div style={{ marginTop: 6, fontSize: 11, color: "#fcd34d", fontWeight: 700 }}>⚠ Review NTC/NMC</div>
        )}
      </div>
    </div>
  );
}

export default function TradeHub() {
  const { setScreen, franchiseState, setFranchiseState } = useGameUI();
  const [activeTab, setActiveTab] = useState("TRADE BUILDER");
  const [numberOfTeams, setNumberOfTeams] = useState(4);
  const [salaryRetention, setSalaryRetention] = useState(true);
  const [salaryMatch, setSalaryMatch] = useState(true);
  const [trade, setTrade] = useState({});
  const [selectedAssetKey, setSelectedAssetKey] = useState("");
  const [showSummary, setShowSummary] = useState(false);
  const [addTarget, setAddTarget] = useState(null);
  const [partnerSlotIds, setPartnerSlotIds] = useState([]);
  const [proposalNotice, setProposalNotice] = useState(null);

  const td = useMemo(() => buildTradeData(franchiseState), [franchiseState]);

  const userTeam = useMemo(() => {
    if (!td?.teams?.length || !td.userTeamId) return null;
    return td.teams.find((t) => String(t.id) === String(td.userTeamId)) || null;
  }, [td]);

  const otherTeamIds = useMemo(() => {
    if (!td?.teams?.length) return [];
    const uid = String(td.userTeamId);
    return td.teams.filter((t) => String(t.id) !== uid).map((t) => t.id);
  }, [td]);

  useEffect(() => {
    const slots = Math.max(0, numberOfTeams - 1);
    if (!otherTeamIds.length) {
      setPartnerSlotIds([]);
      return;
    }
    setPartnerSlotIds((prev) => {
      const next = [];
      for (let i = 0; i < slots; i++) {
        const prevId = prev[i];
        const keep = prevId && otherTeamIds.some((id) => String(id) === String(prevId));
        next.push(keep ? String(prevId) : String(otherTeamIds[i % otherTeamIds.length]));
      }
      return next;
    });
  }, [numberOfTeams, otherTeamIds.join("|"), td?.userTeamId]);

  const visibleTeams = useMemo(() => {
    if (!td?.teams?.length) return [];
    const uid = String(td.userTeamId);
    const mine = td.teams.find((t) => String(t.id) === uid);
    if (!mine) return td.teams.slice(0, numberOfTeams);
    const seen = new Set([uid]);
    const partners = [];
    for (const rawId of partnerSlotIds) {
      if (partners.length >= numberOfTeams - 1) break;
      const id = String(rawId || "");
      if (!id || seen.has(id)) continue;
      const row = td.teams.find((t) => String(t.id) === id);
      if (!row) continue;
      seen.add(id);
      partners.push(row);
    }
    let fill = 0;
    while (partners.length < numberOfTeams - 1 && fill < td.teams.length) {
      const cand = td.teams[fill++];
      if (cand && !seen.has(String(cand.id))) {
        seen.add(String(cand.id));
        partners.push(cand);
      }
    }
    return [mine, ...partners.slice(0, numberOfTeams - 1)];
  }, [td, numberOfTeams, partnerSlotIds.join("|")]);

  const tradeResetKey = useMemo(
    () => (td ? `${td.userTeamId}:${numberOfTeams}:${td.teams.length}:${partnerSlotIds.join(",")}` : ""),
    [td, numberOfTeams, partnerSlotIds],
  );

  const visibleIds = useMemo(() => visibleTeams.map((t) => t.id).join("|"), [visibleTeams]);

  useEffect(() => {
    if (!td || !visibleIds) {
      setTrade({});
      setSelectedAssetKey("");
      return;
    }
    setTrade(emptyTradeForTeamIds(visibleIds.split("|")));
    setSelectedAssetKey("");
  }, [tradeResetKey, visibleIds]);

  const tradeValidity = useMemo(
    () => (td ? evaluateTradeValidity(visibleTeams, trade, salaryMatch, td) : { valid: false, spreadOk: false, spread: 0, maxSpread: 0, capWarnings: [], ntcIssues: [], directionIssues: [] }),
    [visibleTeams, trade, salaryMatch, td],
  );

  useEffect(() => {
    setProposalNotice(null);
  }, [trade]);

  const handleSubmitTrade = useCallback(async () => {
    if (!td) return;
    const total = visibleTeams.reduce((sum, team) => sum + (trade[team.id] || []).length, 0);
    if (total === 0) {
      setProposalNotice({
        kind: "error",
        message: "Add at least one player or pick on the trade boards before submitting.",
      });
      return;
    }
    if (!tradeValidity.valid) {
      setProposalNotice({
        kind: "error",
        message:
          "This package does not pass the current checks (value spread or other flags). Open Trade Summary, fix the balance, then submit again.",
      });
      return;
    }
    try {
      const payload = {};
      visibleTeams.forEach((team) => {
        payload[String(team.id)] = (trade[team.id] || []).map((a) => ({
          type: String(a.type || ""),
          id: String(a.id || ""),
          team: String(a.team || ""),
          retained: Number(a.retained || 0),
        }));
      });
      const res = await submitTradePackage({ assets_by_team: payload });
      if (res?.state) {
        setFranchiseState(res.state);
      }
      const movedPlayers = Number(res?.trade_result?.moved_players || 0);
      setProposalNotice({
        kind: "success",
        message: movedPlayers > 0 ? `Trade executed. ${movedPlayers} player(s) moved and session state refreshed.` : "Trade submitted and state refreshed.",
      });
      setShowSummary(true);
    } catch (e) {
      const d = e?.response?.data?.detail;
      const msg = typeof d === "string" ? d : "Trade submission failed.";
      setProposalNotice({ kind: "error", message: msg });
    }
  }, [td, visibleTeams, trade, tradeValidity, setFranchiseState]);

  const selectedAsset = useMemo(() => {
    if (!td) return null;
    for (const team of visibleTeams) {
      const assets = trade[team.id] || [];
      for (const raw of assets) {
        const key = `${team.id}-${raw.type}-${raw.id}`;
        if (key === selectedAssetKey) return assetDetails(raw, team.id, td);
      }
    }
    return null;
  }, [selectedAssetKey, trade, visibleTeams, td]);

  const addAsset = (targetTeamId, asset) => {
    if (!td || !asset) return;
    const uid = String(td.userTeamId);
    const tgt = String(targetTeamId);
    const src = String(asset.team);
    if (tgt === uid) {
      /* your column: any club may be the asset source */
    } else if (src === uid || src === tgt) {
      /* partner column: your outgoing or that partner's own roster */
    } else {
      return;
    }

    setTrade((prev) => {
      const existing = prev[targetTeamId] || [];
      const already = existing.some((a) => a.type === asset.type && String(a.id) === String(asset.id));

      if (already) return prev;

      return {
        ...prev,
        [targetTeamId]: [...existing, asset],
      };
    });

    setSelectedAssetKey(`${targetTeamId}-${asset.type}-${asset.id}`);
    setAddTarget(null);
  };

  const removeAsset = (targetTeamId, assetType, assetId) => {
    setTrade((prev) => ({
      ...prev,
      [targetTeamId]: (prev[targetTeamId] || []).filter((a) => !(a.type === assetType && String(a.id) === String(assetId))),
    }));
  };

  const requestRemoveAsset = (targetTeamId, assetType, assetId) => {
    if (!td) return;
    const raw = (trade[targetTeamId] || []).find((a) => a.type === assetType && String(a.id) === String(assetId));
    if (!raw) return;
    const det = assetDetails(raw, targetTeamId, td);
    if (
      String(targetTeamId) !== String(td.userTeamId) &&
      String(det.sourceTeam) !== String(td.userTeamId) &&
      String(det.sourceTeam) !== String(targetTeamId)
    )
      return;
    removeAsset(targetTeamId, assetType, assetId);
  };

  const openAddAssetForTeam = (teamId) => {
    if (!td) return;
    const mode = String(teamId) === String(td.userTeamId) ? "incoming" : "counterparty";
    setAddTarget({ acquiringTeamId: teamId, mode });
  };

  const openAddSpecific = (targetTeamId, asset) => {
    if (!td) return;
    if (!asset) {
      openAddAssetForTeam(targetTeamId);
      return;
    }
    addAsset(targetTeamId, asset);
  };

  const headerTeam =
    userTeam ||
    ({
      id: "?",
      name: franchiseState?.team?.name || "Franchise",
      short: "Hub",
      logo: "?",
      color: "#1e293b",
      accent: "#94a3b8",
    });

  if (!td) {
    return (
      <div className="game-screen hub-screen" style={{ background: "var(--g-navy0)", color: "var(--g-text)" }}>
        <TopHeader selectedUserTeam={headerTeam} onBackToHub={() => setScreen(SCREENS.HUB)} />
        <Tabs active={activeTab} setActive={setActiveTab} />
        <div
          style={{
            flex: 1,
            display: "grid",
            placeItems: "center",
            padding: 24,
            color: COLORS.silver,
            textAlign: "center",
            maxWidth: 520,
            margin: "0 auto",
            lineHeight: 1.55,
          }}
        >
          <div style={{ fontFamily: "var(--g-font-head)", fontSize: 18, marginBottom: 10 }}>TRADE FLOOR NEEDS A SAVE</div>
          <div style={{ fontSize: 13, color: COLORS.dim }}>
            Load your franchise from the hub so the trade builder can pull real team names and NHL rosters from your session.
            If you just started the app, begin or resume a franchise first.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="game-screen hub-screen" style={{ background: "var(--g-navy0)", color: "var(--g-text)" }}>
      <TopHeader selectedUserTeam={headerTeam} onBackToHub={() => setScreen(SCREENS.HUB)} />
      <Tabs active={activeTab} setActive={setActiveTab} />

      {activeTab !== "TRADE BUILDER" ? (
        <div
          style={{
            flex: 1,
            display: "grid",
            placeItems: "center",
            color: COLORS.dim,
            fontFamily: "var(--g-font-head)",
            letterSpacing: "0.08em",
          }}
        >
          {activeTab} COMING SOON
        </div>
      ) : (
        <>
          <BuilderControls
            numberOfTeams={numberOfTeams}
            setNumberOfTeams={setNumberOfTeams}
            salaryRetention={salaryRetention}
            setSalaryRetention={setSalaryRetention}
            salaryMatch={salaryMatch}
            setSalaryMatch={setSalaryMatch}
            tradeValidity={tradeValidity}
          />

          <TradePartnerPicker
            td={td}
            userTeamId={td.userTeamId}
            numberOfTeams={numberOfTeams}
            partnerSlotIds={partnerSlotIds}
            setPartnerSlotIds={setPartnerSlotIds}
          />

          <div
            style={{
              flex: 1,
              minHeight: 0,
              display: "grid",
              gridTemplateColumns: "1fr 300px",
              gap: 16,
              padding: "0 16px 10px",
              overflow: "hidden",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 0, overflow: "hidden" }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: `repeat(${numberOfTeams}, minmax(220px, 1fr))`,
                  gap: 10,
                  minHeight: 0,
                  flex: "1 1 56%",
                }}
              >
                {visibleTeams.map((team) => (
                  <TeamTradeCard
                    key={team.id}
                    team={team}
                    td={td}
                    userTeamId={td.userTeamId}
                    assets={trade[team.id] || []}
                    selectedAssetKey={selectedAssetKey}
                    setSelectedAssetKey={setSelectedAssetKey}
                    requestRemoveAsset={requestRemoveAsset}
                    openAddAssetForTeam={openAddAssetForTeam}
                  />
                ))}
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: `repeat(${numberOfTeams}, minmax(220px, 1fr))`,
                  gap: 10,
                  height: 250,
                  flexShrink: 0,
                }}
              >
                {visibleTeams.map((team) => (
                  <AssetsPanel key={team.id} team={team} td={td} userTeamId={td.userTeamId} openAddSpecific={openAddSpecific} />
                ))}
              </div>
            </div>

            <SidePanel
              selectedAsset={selectedAsset}
              showSummary={showSummary}
              setShowSummary={setShowSummary}
              teams={visibleTeams}
              trade={trade}
              salaryMatch={salaryMatch}
              td={td}
              proposalNotice={proposalNotice}
              onSubmitTrade={handleSubmitTrade}
            />
          </div>

          <div
            style={{
              height: 34,
              flexShrink: 0,
              borderTop: `1px solid ${COLORS.line}`,
              display: "flex",
              alignItems: "center",
              gap: 24,
              padding: "0 20px",
              color: COLORS.dim,
              fontSize: 12,
              background: "rgba(3,5,12,0.86)",
            }}
          >
            <span>Ⓥ Trade value = OVR base + production + age + contract vs fair AAV + position + archetype + clutch + market fit; picks tiered with −10%/yr future discount; retention adds 0.5×(ret%×$M).</span>
            <span>Projected lineups, cap hits, and trade values are estimates.</span>
            <span style={{ color: "#f59e0b", marginLeft: "auto" }}>
              ⚠ This trade cannot be processed until July 1, 2025 (Contract Year)
            </span>
          </div>
        </>
      )}

      <AddAssetModal target={addTarget} td={td} close={() => setAddTarget(null)} addAsset={addAsset} />
    </div>
  );
}