import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS, normalizeNhlAbbr } from "../game/constants";
import {
  evaluateTradePackage,
  getTradeAssets,
  getTradeHistory,
  getTradeMarket,
  requestNtcWaive,
  submitTradePackage,
} from "../services/franchiseService";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { ensurePlayerHeadshotFields, nationalityCode } from "../utils/playerHeadshots";
import PlayerHeadshot from "../components/PlayerHeadshot";
import PS1PlayerPortrait from "../components/portraits/PS1PlayerPortrait";
import { getTeamPortraitColors } from "../components/portraits/ps1PortraitUtils";
import { normalizeRosterBrowserPlayer } from "./RosterScreen";
import {
  getBaseOverall,
  getOverallDrop,
  getOverallTooltip,
  getUniversalOverall,
} from "../utils/playerOverall";

const SLOTS = 5;
const DRAG_MIME = "application/x-nhl-trade-asset";

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const safeArray = (v) => (Array.isArray(v) ? v : []);
const roundOverall = (v) => Math.round(Number(v) || 0);

function formatMoneyM(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "$0.0M";
  if (Math.abs(n) >= 10) return `$${n.toFixed(1)}M`;
  return `$${n.toFixed(2)}M`;
}

function formatMoneyShort(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "$0.0M";
  if (Math.abs(n) >= 10) return `$${n.toFixed(1)}M`;
  return `$${n.toFixed(2)}M`;
}

function normalizeMoneyMillions(value) {
  const raw = Number(value);
  if (!Number.isFinite(raw) || !raw) return 0;
  if (raw > 1_000_000) return raw / 1_000_000;
  if (raw > 1_000) return raw / 1_000_000;
  return raw;
}

function resolvePlayerCapHit(row, contract = {}, extra = {}) {
  return normalizeMoneyMillions(
    contract.capHit ??
      contract.cap_hit ??
      contract.aav ??
      contract.salary ??
      extra.capHit ??
      row?.cap_hit ??
      row?.capHit ??
      row?.aav ??
      row?.salary ??
      0,
  );
}

function resolveContractType(row, contract = {}) {
  const raw = String(
    contract.contract_type ??
      contract.contractType ??
      contract.type ??
      row?.contract_type ??
      row?.contractType ??
      "",
  ).toLowerCase();
  if (raw.includes("entry")) return "entry_level";
  if (raw) return raw;
  const capHit = resolvePlayerCapHit(row, contract);
  const age = Number(row?.age) || 99;
  if (capHit > 0 && capHit <= 1.3 && age <= 23) return "entry_level";
  return capHit > 0 ? "standard" : "unsigned";
}

function formatPlayerCapLabel(item) {
  const capHit = Number(item?.capHit) || 0;
  if (capHit > 0) return formatMoneyShort(capHit);
  const contractType = String(item?.contractType || "").toLowerCase();
  if (contractType.includes("entry")) return "ELC";
  return "—";
}

function formatPlayerTermLabel(item) {
  const years = Number(item?.years) || 0;
  if (years > 0) return `${years}Y`;
  const contractType = String(item?.contractType || "").toLowerCase();
  if (contractType.includes("unsigned")) return "UNSIGNED";
  return "UFA";
}

const PROSPECT_MAX_AGE = 21;

const VALUE_TIER_CLASS = {
  Franchise: "tier-franchise",
  FRANCHISE: "tier-franchise",
  Elite: "tier-elite",
  ELITE: "tier-elite",
  "Top Asset": "tier-top",
  "TOP ASSET": "tier-top",
  Useful: "tier-useful",
  USEFUL: "tier-useful",
  Depth: "tier-depth",
  DEPTH: "tier-depth",
  LOW: "tier-depth",
  "Negative Value": "tier-negative",
  UNKNOWN: "tier-unknown",
};

const FLAG_ISO3_TO_ISO2 = {
  CAN: "CA",
  USA: "US",
  SWE: "SE",
  FIN: "FI",
  CZE: "CZ",
  SVK: "SK",
  RUS: "RU",
  GER: "DE",
  SUI: "CH",
  AUT: "AT",
  NOR: "NO",
  DEN: "DK",
  LAT: "LV",
  BLR: "BY",
  UKR: "UA",
  FRA: "FR",
};

function assetValueLabel(item) {
  const tier = String(item?.valueTier || "").trim();
  if (tier) return tier.toUpperCase();

  const raw = Number(item?.tradeValue ?? item?.value_hint);
  if (!Number.isFinite(raw)) return "UNKNOWN";
  return valueTierFromScore(raw);
}

function valueTierFromScore(score) {
  const raw = Number(score);
  if (!Number.isFinite(raw)) return "UNKNOWN";
  if (raw >= 90) return "FRANCHISE";
  if (raw >= 75) return "ELITE";
  if (raw >= 55) return "TOP ASSET";
  if (raw >= 35) return "USEFUL";
  if (raw >= 18) return "DEPTH";
  return "LOW";
}

function roundTradeValue(raw) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 10) / 10;
}

/**
 * Stingy bar fill — ordinary NHLers look short; only true stars near-fill.
 * 15→5% · 30→12% · 45→22% · 60→36% · 75→55% · 90→82% · 98→94%
 */
function tradeValueBarPct(tv) {
  const v = Math.max(0, Number(tv) || 0);
  if (v <= 0) return 2;
  let pct;
  if (v < 30) pct = (v / 30) * 12;
  else if (v < 50) pct = 12 + ((v - 30) / 20) * 14;
  else if (v < 70) pct = 26 + ((v - 50) / 20) * 22;
  else if (v < 85) pct = 48 + ((v - 70) / 15) * 22;
  else pct = 70 + ((Math.min(v, 100) - 85) / 15) * 26;
  return clamp(Math.round(pct), 2, 96);
}

function assetValuePct(item) {
  const raw = Number(item?.tradeValue ?? item?.value_hint);
  if (!Number.isFinite(raw)) return 6;
  return tradeValueBarPct(raw);
}

/** Mirrors SimEngine `_talent_base` — aggressive depth vs star spread. */
function talentValueAnchor(ovr) {
  const o = Number(ovr) || 0;
  if (o <= 0) return 3;
  let anchor;
  if (o < 70) anchor = 3 + Math.max(0, o - 60) * 1.0;
  else if (o < 76) anchor = 10 + (o - 70) * 2.0;
  else if (o < 81) anchor = 20 + (o - 75) * 4.0;
  else if (o < 85) anchor = 40 + (o - 80) * 5.0;
  else if (o < 88) anchor = 60 + (o - 84) * 6.5;
  else if (o < 91) anchor = 80 + (o - 87) * 5.5;
  else anchor = 96 + Math.min(3, (o - 91) * 1.0);
  if (o >= 90) anchor += 4 + (o - 90) * 1.5;
  else if (o >= 87) anchor += 5 + (o - 87) * 1.5;
  else if (o >= 84) anchor += 3.5;
  else if (o < 73) anchor -= (73 - o) * 1.6;
  else if (o < 77) anchor -= (77 - o) * 0.9;
  else if (o < 80) anchor -= (80 - o) * 0.4;
  return clamp(Math.round(anchor * 10) / 10, 2, 99);
}

/** Pool sort/bar — prefer backend TV; fall back to steep OVR anchor. */
function poolPlayerValueScore(item) {
  const backend = roundTradeValue(item?.tradeValue ?? item?.value_hint);
  if (backend != null && backend > 0) return backend;
  return talentValueAnchor(roundOverall(item?.ovr)) || 12;
}

function pickRoundValueAnchor(round) {
  const rnd = Number(round) || 7;
  // Late picks stay cheap so they cannot masquerade as NHL roster chips.
  const anchors = { 1: 72, 2: 48, 3: 32, 4: 20, 5: 12, 6: 8, 7: 5 };
  return anchors[rnd] || 5;
}

/** Pick pool bar width — prefer backend TV, else round anchor. */
function poolPickValueScore(item) {
  const backend = roundTradeValue(item?.tradeValue ?? item?.value_hint);
  if (backend != null && backend > 0) return backend;
  return pickRoundValueAnchor(item?.round);
}

function assetValueTierClass(item) {
  return VALUE_TIER_CLASS[assetValueLabel(item)] || VALUE_TIER_CLASS[item?.valueTier] || "tier-unknown";
}

function resolveFlagIso2(player) {
  const enriched = ensurePlayerHeadshotFields(player || {});
  const raw =
    enriched.nationality_code ||
    enriched.nationalityCode ||
    nationalityCode(enriched.nationality || enriched.country || "") ||
    "";
  const u = String(raw).toUpperCase();
  if (/^[A-Z]{2}$/.test(u)) return u;
  return FLAG_ISO3_TO_ISO2[u] || null;
}

function tradeFlagUrl(player, size = 64) {
  const iso2 = resolveFlagIso2(player);
  if (!iso2) return null;
  return `https://flagsapi.com/${iso2}/flat/${size}.png`;
}

function qualitativeBreakdownTags(breakdown) {
  const b = breakdown || {};
  const tags = [];
  const add = (label, val) => {
    if (val == null || Number(val) === 0) return;
    const n = Number(val);
    if (n >= 8) tags.push(`Strong ${label}`);
    else if (n > 0) tags.push(`${label} Plus`);
    else if (n <= -8) tags.push(`${label} Concern`);
    else tags.push(`${label} Drag`);
  };
  add("Talent", b.talent ?? b.base);
  add("Age", b.age);
  add("Contract", b.contract);
  add("Need Fit", b.team_need);
  add("Potential", b.potential);
  add("Upside", b.prospect_upside);
  add("Rental", b.rental);
  add("ELC", b.elc);
  add("Cap", b.cap_dump);
  add("Injury", b.injury);
  add("Risk", b.risk);
  return tags;
}

function sanitizeTradeExplain(lines) {
  return safeArray(lines).filter((line) => {
    const s = String(line);
    if (/\bTV\b/i.test(s)) return false;
    if (/trade\s*value/i.test(s) && /\d+/.test(s)) return false;
    if (/value\s*[:=]?\s*\d+/i.test(s)) return false;
    if (/[+\-]\s*\d+/.test(s) && /talent|age|contract|pts|value|risk/i.test(s)) return false;
    return s.trim().length > 0;
  });
}

function TradeFlagBadge({ player, size = "sm" }) {
  const [failed, setFailed] = useState(false);
  const enriched = ensurePlayerHeadshotFields(player || {});
  const imgSize = size === "lg" ? 80 : size === "md" ? 72 : 64;
  const url = tradeFlagUrl(enriched, imgSize);
  const label = resolveFlagIso2(enriched) || "";
  const sizeClass = size === "lg" ? "is-lg" : size === "md" ? "is-md" : "";

  if (url && !failed) {
    return (
      <img
        className={`trade-flag-badge ${sizeClass}`.trim()}
        src={url}
        alt={label}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }

  if (label) {
    return <span className={`trade-flag-fallback ${sizeClass}`.trim()}>{label}</span>;
  }

  return null;
}

function TradeValueChip({ item, compact = false, className = "" }) {
  const label = assetValueLabel(item);
  const pct = assetValuePct(item);
  const tierClass = assetValueTierClass(item);
  return (
    <div className={`trade-value-chip ${tierClass} ${compact ? "compact" : ""} ${className}`.trim()}>
      <span className="trade-value-chip-label">{label}</span>
      <div className="trade-value-chip-track" aria-hidden="true">
        <div className="trade-value-chip-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

const TRADE_VALUE_FORMULA_VERSION = 3;

function resolveBackendTradeValue(row, tradeAssets, teamId) {
  const pid = String(row?.player_id || row?.id || "");
  const formulaOk = Number(tradeAssets?.formula_version) >= TRADE_VALUE_FORMULA_VERSION;
  const apiTrade = tradeAssets?.teams?.[String(teamId)]?.players?.[pid];

  // Prefer live trade-assets API (recomputed) over any stale row embedding.
  if (formulaOk && apiTrade?.trade_value != null && Number.isFinite(Number(apiTrade.trade_value))) {
    return {
      tradeValue: roundTradeValue(apiTrade.trade_value),
      valueTier: apiTrade.value_tier || null,
      breakdown: apiTrade.breakdown || {},
      explain: safeArray(apiTrade.explain),
      riskFlags: safeArray(apiTrade.risk_flags),
      contractFlags: safeArray(apiTrade.contract_flags),
      capImpact: apiTrade.cap_impact || null,
      tradeable: apiTrade.tradeable !== false,
      tradeBlockReason: String(apiTrade.trade_block_reason || ""),
      clauseLabel: String(apiTrade.clause_label || ""),
      approvedTradeTeams: safeArray(apiTrade.approved_trade_teams || apiTrade.approved_trade_team_ids),
      canTradeToPartner: apiTrade.can_trade_to_partner,
      requiresNtcWaive: Boolean(apiTrade.requires_ntc_waive),
      ntcWaived: Boolean(apiTrade.ntc_waived),
      ntcWaiverReason: String(apiTrade.ntc_waiver_reason || ""),
      source: "backend",
    };
  }

  const fromRow = row?.trade?.trade_value ?? row?.trade_value;
  if (formulaOk && fromRow != null && Number.isFinite(Number(fromRow))) {
    return {
      tradeValue: roundTradeValue(fromRow),
      valueTier: row?.trade?.value_tier || row?.value_tier || null,
      breakdown: row?.trade?.breakdown || row?.trade?.components || {},
      explain: safeArray(row?.trade?.explain),
      riskFlags: safeArray(row?.trade?.risk_flags),
      contractFlags: safeArray(row?.trade?.contract_flags),
      capImpact: row?.trade?.cap_impact || null,
      tradeable: row?.tradeable !== false,
      tradeBlockReason: String(row?.trade_block_reason || ""),
      clauseLabel: String(row?.clause_label || row?.protection || ""),
      approvedTradeTeams: safeArray(row?.approved_trade_teams || row?.approved_trade_team_ids),
      canTradeToPartner: row?.can_trade_to_partner,
      requiresNtcWaive: Boolean(row?.requires_ntc_waive),
      ntcWaived: Boolean(row?.ntc_waived),
      ntcWaiverReason: String(row?.ntc_waiver_reason || ""),
      source: "backend",
    };
  }

  // Stale session cache / old formula — show live OVR curve (no new save required).
  const ovr = roundOverall(row?.overall ?? row?.ovr ?? apiTrade?.ovr);
  if (ovr > 0) {
    const anchor = talentValueAnchor(ovr);
    return {
      tradeValue: anchor,
      valueTier: valueTierFromScore(anchor),
      breakdown: {},
      explain: ["Live OVR value curve"],
      riskFlags: safeArray(apiTrade?.risk_flags || row?.trade?.risk_flags),
      contractFlags: safeArray(apiTrade?.contract_flags || row?.trade?.contract_flags),
      capImpact: apiTrade?.cap_impact || row?.trade?.cap_impact || null,
      tradeable: apiTrade?.tradeable !== false && row?.tradeable !== false,
      tradeBlockReason: String(apiTrade?.trade_block_reason || row?.trade_block_reason || ""),
      clauseLabel: String(apiTrade?.clause_label || row?.clause_label || row?.protection || ""),
      approvedTradeTeams: safeArray(
        apiTrade?.approved_trade_teams ||
          apiTrade?.approved_trade_team_ids ||
          row?.approved_trade_teams ||
          row?.approved_trade_team_ids,
      ),
      canTradeToPartner: apiTrade?.can_trade_to_partner ?? row?.can_trade_to_partner,
      requiresNtcWaive: Boolean(apiTrade?.requires_ntc_waive || row?.requires_ntc_waive),
      ntcWaived: Boolean(apiTrade?.ntc_waived || row?.ntc_waived),
      ntcWaiverReason: String(apiTrade?.ntc_waiver_reason || row?.ntc_waiver_reason || ""),
      source: "ovr-anchor",
    };
  }

  return {
    tradeValue: null,
    valueTier: null,
    breakdown: {},
    explain: ["Needs backend value"],
    riskFlags: [],
    contractFlags: [],
    capImpact: null,
    tradeable: true,
    tradeBlockReason: "",
    clauseLabel: "",
    requiresNtcWaive: false,
    ntcWaived: false,
    ntcWaiverReason: "",
    source: "missing",
  };
}

function tradeBreakdownChips(breakdown) {
  const b = breakdown || {};
  const labels = [
    ["talent", "Talent", b.talent ?? b.base],
    ["age", "Age", b.age],
    ["contract", "Contract", b.contract],
    ["team_need", "Need", b.team_need],
    ["potential", "Potential", b.potential],
    ["prospect_upside", "Upside", b.prospect_upside],
    ["rental", "Rental", b.rental],
    ["elc", "ELC", b.elc],
    ["cap_dump", "Cap dump", b.cap_dump],
    ["injury", "Injury", b.injury],
    ["risk", "Risk", b.risk],
  ];
  return labels
    .filter(([, , v]) => v != null && Number(v) !== 0)
    .map(([key, label, v]) => ({ key, label, value: Number(v) }));
}

function computeFanReaction({ userTeam, userOutgoing, evaluation, franchiseState }) {
  const baseRaw =
    franchiseState?.team?.fan_morale ??
    franchiseState?.team?.fan_satisfaction ??
    franchiseState?.fan_morale ??
    franchiseState?.fan_satisfaction ??
    null;
  let score =
    baseRaw != null
      ? Math.round(Number(baseRaw) * (Number(baseRaw) <= 1 ? 100 : 1))
      : clamp(Math.round(42 + (Number(userTeam?.playoffOdds) || 40) * 0.38), 35, 78);

  safeArray(userOutgoing)
    .filter(Boolean)
    .forEach((p) => {
      const ovr = Number(p.ovr) || 0;
      if (p.type === "pick") {
        if (Number(p.round) === 1) score -= 10;
        else if (Number(p.round) === 2) score -= 4;
        return;
      }
      if (ovr >= 88) score -= 20;
      else if (ovr >= 84) score -= 14;
      else if (ovr >= 80) score -= 8;
      else if (ovr >= 76) score -= 4;
      if (Number(p.age) <= 24 && ovr >= 78) score -= 6;
    });

  const userNet = Number(evaluation?.asset_breakdown?.user?.net) || 0;
  if (evaluation?.accepted && userNet >= -2) score += 5;
  if (userNet < -10) score -= 6;

  return clamp(Math.round(score), 8, 98);
}

/** Prefer backend fan_reaction when present; fall back to local preview math. */
function resolveFanReaction({
  userTeam,
  userOutgoing,
  evaluation,
  franchiseState,
  hasProposed = false,
  partnerTeam = null,
}) {
  const localFactors = fanReactionFactors(
    userOutgoing,
    hasProposed ? evaluation : null,
    userTeam,
    partnerTeam,
  ).slice(0, 3);
  const backend = evaluation?.fan_reaction;
  if (backend && Number.isFinite(Number(backend.fan_reaction_score))) {
    const score = Number(backend.fan_reaction_score);
    const heat = Number.isFinite(Number(backend.fan_heat))
      ? Number(backend.fan_heat)
      : clamp(100 - score, 0, 100);
    const backendFactors = safeArray(backend.fan_factors).slice(0, 3);
    return {
      score,
      heat,
      category: backend.fan_category || fanReactionCategory(score),
      heatLabel: backend.fan_heat_label || fanHeatLabelFromHeat(heat),
      factors: backendFactors.length ? backendFactors : localFactors,
      effects: backend.fan_effects || null,
      headline: backend.fan_headline || "",
      summary: backend.fan_summary || "",
      source: "backend",
    };
  }
  const score = computeFanReaction({
    userTeam,
    userOutgoing,
    evaluation: hasProposed ? evaluation : null,
    franchiseState,
  });
  const heat = clamp(100 - score, 0, 100);
  return {
    score,
    heat,
    category: fanReactionCategory(score),
    heatLabel: fanHeatLabelFromHeat(heat),
    factors: localFactors,
    effects: null,
    headline: "",
    summary: "",
    source: "local",
  };
}

function fanHeatLabelFromHeat(heat) {
  const h = Number(heat) || 0;
  if (h >= 75) return "Furious";
  if (h >= 55) return "Backlash";
  if (h >= 30) return "Uneasy";
  return "Calm";
}

function fanEffectsSummary(effects) {
  if (!effects || typeof effects !== "object") return [];
  const rows = [];
  const conf = Number(effects.fan_confidence_delta);
  const owner = Number(effects.owner_patience_delta);
  const gm = Number(effects.gm_trust_delta);
  if (conf) rows.push(`Confidence ${conf > 0 ? "+" : ""}${Math.round(conf)}`);
  if (owner) rows.push(`Owner Patience ${owner > 0 ? "+" : ""}${Math.round(owner)}`);
  if (gm) rows.push(`GM Trust ${gm > 0 ? "+" : ""}${Math.round(gm)}`);
  return rows.slice(0, 3);
}

function FanReasonChips({ factors }) {
  const chips = safeArray(factors).slice(0, 3);
  if (!chips.length) return null;
  return (
    <div className="trade-fan-reason-chips">
      {chips.map((f) => (
        <span key={f} className="trade-fan-reason-chip">{f}</span>
      ))}
    </div>
  );
}

function inferLogoAbbr(teamId, teamName) {
  const a = normalizeNhlAbbr(teamId);
  if (a) return a;
  const b = normalizeNhlAbbr(teamName);
  if (b) return b;
  const t = String(teamName || teamId || "?").replace(/[^A-Za-z]/g, "");
  return (t.slice(0, 3).toUpperCase() || "?").slice(0, 3);
}

function teamHueFromId(id) {
  let h = 0;
  const s = String(id || "");
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function normalizeSkaterPosition(raw) {
  const u = String(raw || "").trim().toUpperCase();
  if (!u) return { pos: "—", handed: "Right" };
  if (["C", "LW", "RW", "F", "LD", "RD", "D", "G"].includes(u)) {
    const handed =
      u === "RD" || u === "RW" ? "Right" : u === "LD" || u === "LW" ? "Left" : "Left";
    return { pos: u, handed };
  }
  if (u.includes("GOAL")) return { pos: "G", handed: "Left" };
  if (u.includes("LEFT") && u.includes("DEF")) return { pos: "LD", handed: "Left" };
  if (u.includes("RIGHT") && u.includes("DEF")) return { pos: "RD", handed: "Right" };
  if (u.includes("DEF")) return { pos: "D", handed: "Left" };
  if (u.includes("CENTER") || u === "CENTRE") return { pos: "C", handed: "Left" };
  if (u.includes("LEFT") && u.includes("WING")) return { pos: "LW", handed: "Left" };
  if (u.includes("RIGHT") && u.includes("WING")) return { pos: "RW", handed: "Right" };
  if (u.includes("LW")) return { pos: "LW", handed: "Left" };
  if (u.includes("RW")) return { pos: "RW", handed: "Right" };
  if (u.includes("LD")) return { pos: "LD", handed: "Left" };
  if (u.includes("RD")) return { pos: "RD", handed: "Right" };
  if (u === "D" || u.includes(" D")) return { pos: "D", handed: "Left" };
  if (u.includes("G")) return { pos: "G", handed: "Left" };
  if (u.includes("C")) return { pos: "C", handed: "Left" };
  return { pos: u, handed: "Right" };
}

function resolveOpponentOvr(row) {
  return getUniversalOverall(row) || roundOverall(Number(row?.ovr ?? row?.overall ?? 0) || 0);
}

function roleFromOverall(ovr, pos) {
  const o = Number(ovr) || 0;
  const p = String(pos || "F").toUpperCase();
  if (p === "G") {
    if (o >= 86) return "STARTER";
    if (o >= 80) return "BACKUP";
    return "DEPTH";
  }
  if (o >= 90) return "ELITE";
  if (o >= 86) return p === "D" ? "TOP PAIR" : "1ST LINE";
  if (o >= 82) return p === "D" ? "TOP 4" : "TOP 6";
  if (o >= 78) return p === "D" ? "2ND PAIR" : "MIDDLE 6";
  if (o >= 72) return "DEPTH";
  return "PROSPECT";
}

function roundLabel(r) {
  const n = Number(r) || 0;
  if (n === 1) return "1ST";
  if (n === 2) return "2ND";
  if (n === 3) return "3RD";
  return `${n}TH`;
}

function scoutingLabel(level, pct) {
  const p = Number(pct) || 0;
  if (level === "fully_scouted" || p >= 90) return "FULL INTEL";
  if (level === "pro_scouted" || p >= 65) return "PRO SCOUTED";
  if (level === "basic_profile" || p >= 40) return "PARTIAL";
  if (level === "name_known" || p >= 15) return "NAME ONLY";
  return "LIMITED INTEL";
}

function displayOvr(player) {
  const ovr = getUniversalOverall(player) || roundOverall(Number(player?.ovr) || 0);
  if (!ovr) return "—";
  const pct = Number(player?.scouted_pct) || 0;
  const conf = String(player?.ovr_confidence || "");
  if (!player?.is_user_roster && (conf === "hidden" || pct < 25)) return "??";
  const drop = getOverallDrop(player);
  if (drop >= 1) return `${ovr}↓${drop}`;
  return String(ovr);
}

function activeTradeRumorModifier(player) {
  const mods = safeArray(player?.ovr_modifiers);
  for (const mod of mods) {
    const source = String(mod?.source || "").toLowerCase();
    const cause = String(mod?.cause_type || "").toUpperCase();
    if (
      source === "trade_rumor" ||
      cause === "TRADE_REJECTED" ||
      cause === "PLAYER_REPEATEDLY_SHOPPED" ||
      cause === "TRADE_ATTEMPTED_BY_USER"
    ) {
      return {
        amount: Number(mod?.amount) || 0,
        gamesRemaining: Number(mod?.games_remaining) || 0,
      };
    }
  }
  return null;
}

function clauseFromRow(row, contract) {
  const clause = String(
    row?.clause_label || row?.protection || contract?.clause_label || contract?.clause || "",
  ).toUpperCase();
  if (clause.includes("NMC")) return "NMC";
  if (clause.includes("NTC")) return "NTC";
  if (clause.includes("M-NTC") || clause.includes("MODIFIED")) return "M-NTC";
  return "None";
}

function clauseBlockReason(protection, tradeMeta) {
  const label = String(tradeMeta?.clauseLabel || protection || "").toUpperCase();
  if (tradeMeta?.tradeable === false && tradeMeta?.tradeBlockReason) {
    return tradeMeta.tradeBlockReason;
  }
  if (label.includes("NMC")) return "No-movement clause";
  if (label.includes("NTC") && !label.includes("M-NTC")) return "No-trade clause";
  if (label.includes("M-NTC") || label.includes("MODIFIED")) {
    const approved = safeArray(tradeMeta?.approvedTradeTeams);
    if (tradeMeta?.canTradeToPartner === false || approved.length === 0) {
      return "Modified no-trade clause — destination not approved";
    }
    if (tradeMeta?.canTradeToPartner === false) {
      return "Modified no-trade clause — destination not approved";
    }
  }
  return "";
}

function seasonStatsFromRow(row, franchiseState) {
  const pid = String(row?.player_id || row?.id || "");
  const fromRow = row?.season_stats || row?.seasonStats || row?.stats || null;
  const fromState =
    pid && franchiseState?.player_season_stats
      ? franchiseState.player_season_stats[pid] || franchiseState.player_season_stats[String(pid)]
      : null;
  const src = fromRow || fromState || {};
  const gp = Number(src.gp ?? src.games_played);
  const g = Number(src.g ?? src.goals);
  const a = Number(src.a ?? src.assists);
  const pts = Number(src.pts ?? src.points);
  const warRaw = src.war ?? src.WATR ?? src.watr ?? src.total_impact;
  const war = warRaw != null && Number.isFinite(Number(warRaw)) ? Number(warRaw) : null;
  return {
    gp: Number.isFinite(gp) ? gp : null,
    g: Number.isFinite(g) ? g : null,
    a: Number.isFinite(a) ? a : null,
    pts: Number.isFinite(pts) ? pts : (Number.isFinite(g) && Number.isFinite(a) ? g + a : null),
    war,
  };
}

function normalizePlayerFromRow(row, teamId, franchiseState, isUserTeam = false, index = 0, tradeAssets = null, partnerTeamId = null) {
  const rowContract = row?.contract || {};
  const rowCapHit = resolvePlayerCapHit(row, rowContract);
  const norm = franchiseState ? normalizeRosterBrowserPlayer(row, franchiseState, index) : null;
  const contract = norm?.contract || rowContract;
  const capHit =
    rowCapHit > 0
      ? rowCapHit
      : resolvePlayerCapHit(row, contract, norm?.contract || {});
  const years =
    Number(
      contract.term ??
        contract.years_remaining ??
        contract.years ??
        norm?.contract?.term ??
        0,
    ) || 0;
  const contractType = resolveContractType(row, contract);
  const protection = clauseFromRow(row, contract);
  const tradeMeta = resolveBackendTradeValue(row, tradeAssets, teamId);
  const ntcWaived = Boolean(tradeMeta.ntcWaived || row?.ntc_waived || row?.ntcWaived);
  const requiresNtcWaive =
    Boolean(tradeMeta.requiresNtcWaive || row?.requires_ntc_waive) ||
    ((protection === "NTC" || protection === "M-NTC") && !ntcWaived && tradeMeta.tradeable === false);
  const clauseBlocked = Boolean(clauseBlockReason(protection, tradeMeta)) && !ntcWaived;
  const blockReason =
    ntcWaived
      ? ""
      : tradeMeta.tradeBlockReason ||
        clauseBlockReason(protection, tradeMeta) ||
        (protection === "NMC" ? "No-movement clause" : "");
  const season = seasonStatsFromRow(row, franchiseState);

  const base = {
    tradeValue: tradeMeta.tradeValue,
    valueTier: tradeMeta.valueTier,
    tradeBreakdown: tradeMeta.breakdown,
    tradeExplain: tradeMeta.explain,
    tradeRiskFlags: tradeMeta.riskFlags,
    tradeContractFlags: tradeMeta.contractFlags,
    tradeCapImpact: tradeMeta.capImpact,
    tradeValueSource: tradeMeta.source,
    tradeable: (tradeMeta.tradeable !== false && !clauseBlocked) || ntcWaived,
    tradeBlockReason: blockReason,
    clauseLabel: tradeMeta.clauseLabel || protection,
    approvedTradeTeams: tradeMeta.approvedTradeTeams,
    requiresNtcWaive,
    ntcWaived,
    ntcWaiverReason: tradeMeta.ntcWaiverReason || row?.ntc_waiver_reason || "",
    gp: season.gp,
    g: season.g,
    a: season.a,
    pts: season.pts,
    war: season.war,
  };

  if (isUserTeam && norm) {
    const { handed } = normalizeSkaterPosition(norm.position || row?.position);
    return {
      id: String(norm.id || norm.key || row?.player_id || `${teamId}-${norm.name}`),
      name: String(norm.name || row?.name || "?"),
      pos: String(norm.position || normalizeSkaterPosition(row?.position).pos),
      age: Number(norm.age ?? row?.age) || 25,
      handed:
        row?.handed ??
        row?.shoots ??
        row?.catches ??
        norm?.handed ??
        norm?.shoots ??
        norm?.catches ??
        handed,
      nationality: norm?.nationality || row?.nationality || row?.country || row?.birth_country || "",
      nationalityCode:
        norm?.nationalityCode ||
        norm?.nationality_code ||
        row?.nationalityCode ||
        row?.nationality_code ||
        row?.country_code ||
        nationalityCode(norm?.nationality || row?.nationality || row?.country || "") ||
        "",
      ovr: getUniversalOverall(norm) || roundOverall(Number(norm.ovr) || 0),
      base_ovr: getBaseOverall(norm) || getUniversalOverall(norm),
      overall_drop: getOverallDrop(norm),
      ovr_modifiers: norm?.ovr_modifiers || row?.ovr_modifiers || [],
      capHit,
      years,
      contractType,
      protection,
      archetype: String(norm.archetype || row?.archetype || "").trim(),
      role: String(norm.roleLabel || norm.role || "").trim(),
      depthTier: String(row?.depth_tier || norm.assetTag || "depth").toLowerCase(),
      scoutingTag: "FULL INTEL",
      scouted_pct: 100,
      scouting_level: "fully_scouted",
      ovr_confidence: "exact",
      headshot_url: norm.headshot_url || row?.headshot_url || row?.headshotUrl || null,
      is_injured: Boolean(norm.is_injured ?? row?.is_injured),
      is_user_roster: true,
      teamId: String(teamId),
      potentialGrade: potentialGradeFromPlayer(row, roundOverall(Number(norm.ovr) || 0), Number(norm.age) || 25),
      ...base,
    };
  }

  const fullName = String(norm?.name || row?.name || "?");
  const { pos, handed } = normalizeSkaterPosition(row?.position || norm?.position);
  const age = Number(row?.age ?? norm?.age) || 25;
  const ovr =
    getUniversalOverall(norm) ||
    getUniversalOverall(row) ||
    roundOverall(resolveOpponentOvr(row));
  const scoutedPct = row?.scouted_pct != null ? Number(row.scouted_pct) : 100;
  const scoutingLevel = String(row?.scouting_level || "fully_scouted");

  return {
    id: String(row?.player_id || norm?.id || `${teamId}-${fullName}`),
    name: fullName,
    pos,
    age,
    handed:
      row?.handed ??
      row?.shoots ??
      row?.catches ??
      norm?.handed ??
      norm?.shoots ??
      norm?.catches ??
      handed,
    nationality: norm?.nationality || row?.nationality || row?.country || row?.birth_country || "",
    nationalityCode:
      norm?.nationalityCode ||
      norm?.nationality_code ||
      row?.nationalityCode ||
      row?.nationality_code ||
      row?.country_code ||
      nationalityCode(norm?.nationality || row?.nationality || row?.country || "") ||
      "",
    ovr,
    base_ovr: getBaseOverall(norm) || getBaseOverall(row) || ovr,
    overall_drop: getOverallDrop(norm) || getOverallDrop(row),
    ovr_modifiers: norm?.ovr_modifiers || row?.ovr_modifiers || [],
    capHit,
    years,
    contractType,
    protection,
    archetype: String(row?.archetype || norm?.archetype || "").trim(),
    role: roleFromOverall(ovr, pos),
    depthTier: String(row?.depth_tier || roleFromOverall(ovr, pos)).toLowerCase(),
    scoutingTag: scoutingLabel(scoutingLevel, scoutedPct),
    scouted_pct: scoutedPct,
    scouting_level: scoutingLevel,
    ovr_confidence: String(row?.ovr_confidence || "exact"),
    headshot_url: row?.headshot_url || row?.headshotUrl || norm?.headshot_url || null,
    is_injured: Boolean(row?.is_injured ?? norm?.is_injured),
    is_user_roster: false,
    teamId: String(teamId),
    potentialGrade: potentialGradeFromPlayer(row, ovr, age),
    ...base,
  };
}

function normalizePickFromBackend(pick, ownerTeamId) {
  const id = String(pick?.pick_id || pick?.id || "");
  const year = Number(pick?.year ?? pick?.draftYear) || 0;
  const round = Number(pick?.round) || 0;
  const origId = String(pick?.original_team_id || ownerTeamId);
  const tradeValueRaw =
    pick?.trade_value ?? pick?.tradeValue ?? pick?.value ?? pick?.value_hint;
  const tradeValue =
    tradeValueRaw != null && Number.isFinite(Number(tradeValueRaw))
      ? Number(tradeValueRaw)
      : null;
  return {
    id,
    pick_id: id,
    year,
    round,
    original_team_id: origId,
    current_owner_team_id: String(pick?.current_owner_team_id || ownerTeamId),
    owner: String(pick?.owner || ownerTeamId),
    protection: pick?.protection || null,
    value_hint: pick?.value_hint != null ? Number(pick.value_hint) : tradeValue,
    tradeValue,
    valueTier: pick?.value_tier || pick?.valueTier || null,
    projectedSlot: pick?.projected_slot ?? pick?.projectedSlot ?? null,
    projectedRange: pick?.projected_range || pick?.projectedRange || null,
    pickValueContext: pick?.pick_value_context || pick?.value_context || pick?.pickValueContext || null,
    originalTeamAbbr: normalizeNhlAbbr(origId) || "—",
    display: pick?.display || pick?.label || `${year} ${roundLabel(round)}`,
    type: "pick",
    teamId: String(ownerTeamId),
  };
}

function getOwnedPicks(teamId, tradeAssets, rosterOrgs) {
  const tid = String(teamId);
  const fromApi = safeArray(tradeAssets?.teams?.[tid]?.picks);
  if (fromApi.length) return fromApi.map((p) => normalizePickFromBackend(p, tid));
  const org = safeArray(rosterOrgs).find((o) => String(o.team_id) === tid);
  const fromOrg = safeArray(org?.trade_picks);
  if (fromOrg.length) return fromOrg.map((p) => normalizePickFromBackend(p, tid));
  return [];
}

function getTeamCapSummary(teamId, franchiseState, tradeAssets, rosterPlayers) {
  const tid = String(teamId);
  const userId = String(franchiseState?.team?.id || franchiseState?.user_team_id || "");
  const apiCap = tradeAssets?.teams?.[tid]?.cap || {};
  const isUser = tid === userId;
  const teamState = franchiseState?.team || {};

  let capHit = normalizeMoneyMillions(
    isUser
      ? teamState.cap_hit ?? teamState.capHit ?? apiCap.total_cap_hit ?? apiCap.totalCapHit
      : apiCap.total_cap_hit ?? apiCap.totalCapHit ?? teamState.cap_hit,
  );
  let capLimit = normalizeMoneyMillions(
    isUser
      ? teamState.salary_cap ?? teamState.cap_limit ?? apiCap.upper_limit ?? apiCap.upperLimit
      : apiCap.upper_limit ?? apiCap.upperLimit,
  );
  let capSpace = normalizeMoneyMillions(
    isUser
      ? teamState.cap_space ?? teamState.capSpace ?? apiCap.usable_cap_space ?? apiCap.usableCapSpace
      : apiCap.usable_cap_space ?? apiCap.usableCapSpace,
  );

  const rosterSum = safeArray(rosterPlayers).reduce((s, p) => s + (Number(p.capHit) || 0), 0);
  if ((!capHit || capHit <= 0) && rosterSum > 0) capHit = rosterSum;
  if (capLimit > 0 && capHit > 0 && (!Number.isFinite(capSpace) || capSpace === 0)) {
    capSpace = capLimit - capHit;
  }

  return {
    capHit: capHit > 0 ? capHit : null,
    capLimit: capLimit > 0 ? capLimit : null,
    capSpace: Number.isFinite(capSpace) ? capSpace : null,
  };
}

function getTeamRecord(teamId, standings) {
  const row = safeArray(standings).find((r) => String(r.team_id) === String(teamId));
  if (!row) return "--";
  return `${Number(row.w) || 0}-${Number(row.l) || 0}-${Number(row.otl) || 0}`;
}

function getTeamDirection(teamId, tradeAssets, franchiseState, standings) {
  const tid = String(teamId);
  const dir = tradeAssets?.teams?.[tid]?.team_direction;
  if (dir) return String(dir).toUpperCase();
  const st = safeArray(standings).find((r) => String(r.team_id) === tid);
  const gp = Number(st?.gp) || 1;
  const ppg = (Number(st?.pts) || 0) / Math.max(1, gp);
  if (ppg < 0.88) return "SELLER";
  if (ppg > 1.12) return "BUYER";
  if (ppg > 1.03) return "CONTENDER";
  const strat = String(franchiseState?.team?.strategy || "");
  if (tid === String(franchiseState?.user_team_id) && strat) return strat.toUpperCase();
  return "LISTENING";
}

function collectStandings(franchiseState) {
  if (Array.isArray(franchiseState?.standings)) return franchiseState.standings;
  if (Array.isArray(franchiseState?.standings?.teams)) return franchiseState.standings.teams;
  if (Array.isArray(franchiseState?.league_standings)) return franchiseState.league_standings;
  return [];
}

function standingRowForTeam(teamId, franchiseState) {
  const tid = String(teamId);
  return collectStandings(franchiseState).find((r) => String(r.team_id || r.id) === tid) || null;
}

function computePointsPace(row) {
  const gp = Math.max(1, Number(row?.gp) || 0);
  const pts = Number(row?.pts) || 0;
  return pts / gp;
}

function computePointsPct(row) {
  const gp = Number(row?.gp) || 0;
  const pts = Number(row?.pts) || 0;
  if (gp <= 0) return null;
  return Math.round((pts / (gp * 2)) * 1000) / 10;
}

function computeGoalDifferential(row) {
  if (!row) return null;
  const gf = row.gf ?? row.goals_for ?? row.goalsFor;
  const ga = row.ga ?? row.goals_against ?? row.goalsAgainst;
  if (gf == null || ga == null) return null;
  const diff = Number(gf) - Number(ga);
  return Number.isFinite(diff) ? diff : null;
}

function countInjuredPlayers(players) {
  return safeArray(players).filter((p) => p.is_injured).length;
}

function roundSuffix(r) {
  const n = Number(r) || 0;
  if (n === 1) return "1st";
  if (n === 2) return "2nd";
  if (n === 3) return "3rd";
  if (n <= 0) return null;
  return `${n}th`;
}

function resolveDraftYear(franchiseState) {
  const explicit =
    franchiseState?.draft_year ??
    franchiseState?.draftYear ??
    franchiseState?.upcoming_draft_year;
  if (explicit != null && Number.isFinite(Number(explicit))) return Number(explicit);
  const seasonYear = Number(franchiseState?.season_year) || new Date().getFullYear();
  return seasonYear + 1;
}

function pickMatchesYear(pick, draftYear) {
  const raw = pick?.year ?? pick?.draft_year ?? pick?.draftYear;
  if (raw == null) return false;
  return Number(raw) === Number(draftYear);
}

function formatCurrentYearPicks(picks, draftYear) {
  const list = safeArray(picks).filter((p) => pickMatchesYear(p, draftYear));
  if (!list.length) return null;
  const rounds = [...new Set(list.map((p) => Number(p.round)).filter((r) => r > 0))].sort((a, b) => a - b);
  if (!rounds.length) return null;
  const roundStr = rounds.map((r) => roundSuffix(r)).join(", ");
  if (roundStr.length > 20) {
    return `${String(draftYear).slice(-2)}: ${rounds.join(", ")}`;
  }
  return `${draftYear}: ${roundStr}`;
}

function parseLast10WinPct(lastTen) {
  const s = String(lastTen || "").trim();
  if (!s || s === "—") return null;
  const m = s.match(/(\d+)[\s\-–]+(\d+)(?:[\s\-–]+(\d+))?/);
  if (!m) return null;
  const wins = Number(m[1]);
  const losses = Number(m[2]);
  const otl = Number(m[3]) || 0;
  if (!Number.isFinite(wins) || !Number.isFinite(losses)) return null;
  const total = wins + losses + otl;
  return total > 0 ? wins / total : null;
}

function parseStreakDelta(streak) {
  const s = String(streak || "").trim().toUpperCase();
  if (!s || s === "—") return 0;
  const m = s.match(/([WL])(\d+)/);
  if (!m) return 0;
  const n = Number(m[2]) || 0;
  return m[1] === "W" ? n : -n;
}

function normalizeMarketLabel(raw) {
  const key = String(raw || "").trim().toLowerCase();
  const map = {
    cool: { label: "Cool", tone: "good", tradeLeverageHint: "Calm" },
    stable: { label: "Stable", tone: "neutral", tradeLeverageHint: "Flexible" },
    warm: { label: "Warm", tone: "warn", tradeLeverageHint: "Pressured" },
    hot: { label: "Hot", tone: "bad", tradeLeverageHint: "Pressured" },
    panic: { label: "Panic", tone: "bad", tradeLeverageHint: "Desperate" },
  };
  return map[key] || null;
}

function deriveMarketPressure(teamIntel, standingsRow, tradeContext) {
  const tid = String(teamIntel?.id || "");
  const ta = tradeContext?.tradeAssets?.teams?.[tid] || {};
  const backendLabel = ta.market_label ?? ta.market_pressure_label;
  if (backendLabel) {
    const normalized = normalizeMarketLabel(backendLabel);
    if (normalized) return normalized;
  }

  let pressure = 50;

  const odds = Number(teamIntel?.playoffOdds);
  if (Number.isFinite(odds)) {
    if (odds >= 80) pressure -= 18;
    else if (odds >= 65) pressure -= 12;
    else if (odds >= 50) pressure -= 4;
    else if (odds >= 35) pressure += 6;
    else if (odds >= 20) pressure += 14;
    else pressure += 22;
  }

  const ptsPct = Number(teamIntel?.pointsPct);
  if (Number.isFinite(ptsPct)) {
    if (ptsPct >= 58) pressure -= 10;
    else if (ptsPct >= 52) pressure -= 4;
    else if (ptsPct >= 48) pressure += 2;
    else if (ptsPct >= 44) pressure += 8;
    else pressure += 14;
  }

  const divRank = Number(teamIntel?.standings?.divisionRank);
  if (divRank > 0) {
    if (divRank <= 3) pressure -= 6;
    else if (divRank <= 5) pressure += 0;
    else if (divRank <= 7) pressure += 8;
    else pressure += 14;
  }

  const l10WinPct = parseLast10WinPct(teamIntel?.lastTen);
  if (l10WinPct != null) {
    if (l10WinPct >= 0.65) pressure -= 10;
    else if (l10WinPct >= 0.5) pressure -= 2;
    else if (l10WinPct >= 0.4) pressure += 6;
    else pressure += 14;
  }

  const streak = parseStreakDelta(teamIntel?.streak);
  if (streak <= -4) pressure += 16;
  else if (streak <= -2) pressure += 8;
  else if (streak >= 4) pressure -= 8;
  else if (streak >= 2) pressure -= 4;

  const diff = Number(teamIntel?.goalDiff);
  if (Number.isFinite(diff)) {
    if (diff >= 15) pressure -= 6;
    else if (diff >= 0) pressure -= 2;
    else if (diff >= -10) pressure += 4;
    else pressure += 10;
  }

  const inj = Number(teamIntel?.injuryCount) || 0;
  if (inj >= 5) pressure += 14;
  else if (inj >= 3) pressure += 8;
  else if (inj >= 2) pressure += 4;

  const status = String(teamIntel?.tradeDirectionLabel || teamIntel?.statusLabel || "").toUpperCase();
  if (["REBUILD", "REBUILDER", "SELLER", "TANKING"].includes(status)) pressure += 10;
  if (status === "INJURED") pressure += 12;
  if (status === "CAP TIGHT") pressure += 8;
  if (["CONTENDER", "PLAYOFF", "STANLEY CUP CONTENDER"].some((k) => status.includes(k))) pressure -= 8;
  if (status === "BUBBLE") pressure += 6;

  const cap = Number(teamIntel?.capSpace);
  if (Number.isFinite(cap) && cap < 0) pressure += 6;

  const backendHeat =
    ta.market_pressure ?? ta.market_heat ?? ta.fan_pressure ?? ta.media_pressure;
  if (backendHeat != null && Number.isFinite(Number(backendHeat))) {
    pressure += (Number(backendHeat) - 0.5) * 30;
  }

  const deadlinePhase = Number(
    tradeContext?.deadlinePhase ?? tradeContext?.tradeMarket?.deadline_phase,
  );
  if (Number.isFinite(deadlinePhase)) {
    if (deadlinePhase > 0.55) pressure += 6;
    else if (deadlinePhase > 0.25) pressure += 3;
  }

  if (standingsRow) {
    const leagueRank = Number(teamIntel?.standings?.leagueRank);
    if (leagueRank >= 28) pressure += 8;
    else if (leagueRank >= 24) pressure += 4;
  }

  if (pressure <= 28) {
    return { label: "Cool", tone: "good", tradeLeverageHint: "Calm", score: pressure };
  }
  if (pressure <= 42) {
    return { label: "Stable", tone: "neutral", tradeLeverageHint: "Flexible", score: pressure };
  }
  if (pressure <= 56) {
    return { label: "Warm", tone: "warn", tradeLeverageHint: "Pressured", score: pressure };
  }
  if (pressure <= 72) {
    return { label: "Hot", tone: "bad", tradeLeverageHint: "Pressured", score: pressure };
  }
  return { label: "Panic", tone: "bad", tradeLeverageHint: "Desperate", score: pressure };
}

function compactStatusLabel(label) {
  const s = String(label || "").toUpperCase().trim();
  if (!s || s === "—") return "—";
  if (s.includes("STANLEY") || s === "CONTENDER") return "CONTENDER";
  if (s.includes("PLAYOFF")) return "PLAYOFF";
  if (s.includes("BUBBLE")) return "BUBBLE";
  if (s.includes("REBUILD") || s.includes("TANK")) return "REBUILD";
  if (s.includes("SELL")) return "SELLER";
  if (s.includes("CAP TIGHT") || s.includes("CAP-TIGHT")) return "CAP TIGHT";
  if (s.includes("INJUR")) return "INJURED";
  if (s.includes("BUYER")) return "BUYER";
  if (s.includes("LISTEN")) return "LISTENING";
  return s;
}

function estimatePlayoffOdds(row, franchiseState) {
  // Frontend fallback only — prefer backend playoff_odds / playoff_pct from trade assets.
  if (row?.playoff_odds != null) return Math.round(Number(row.playoff_odds));
  if (row?.playoff_pct != null) return Math.round(Number(row.playoff_pct));
  const ppg = computePointsPace(row);
  if (ppg >= 1.18) return 96;
  if (ppg >= 1.1) return 82;
  if (ppg >= 1.02) return 58;
  if (ppg >= 0.95) return 38;
  if (ppg >= 0.88) return 18;
  return 6;
}

function resolvePlayoffOdds(ta, stRow, franchiseState) {
  const gp = Number(stRow?.gp) || 0;
  const ptsPct = computePointsPct(stRow);
  const raw =
    ta?.playoff_odds ??
    ta?.playoff_pct ??
    stRow?.playoff_odds ??
    stRow?.playoff_pct;
  let odds =
    raw != null && Number.isFinite(Number(raw))
      ? Math.round(Number(raw))
      : estimatePlayoffOdds(stRow, franchiseState);
  if (odds == null || !Number.isFinite(Number(odds))) return null;
  odds = Math.round(Number(odds));
  // Pace honesty: do not show contender-level odds for buried teams.
  if (gp >= 10 && ptsPct != null) {
    if (ptsPct <= 30) odds = Math.min(odds, 12);
    else if (ptsPct <= 40) odds = Math.min(odds, 28);
  }
  return odds;
}

function resolveStatusLabel(ta, direction, stRow, franchiseState) {
  const gp = Number(stRow?.gp) || 0;
  const ptsPct = computePointsPct(stRow);
  const paceStatus = contenderStatusLabel(direction, stRow, franchiseState);
  // Mid-season record beats stale backend CONTENDER labels.
  if (gp >= 10 && ptsPct != null) {
    if (ptsPct <= 43) return ptsPct <= 35 ? "TANKING" : "SELLER";
    if (ptsPct >= 55) return "CONTENDER";
  }
  const raw = ta?.outlook_label ?? ta?.contention_label ?? ta?.team_status;
  if (raw) {
    const up = String(raw).toUpperCase();
    if (gp >= 10 && ptsPct != null && ptsPct <= 43 && /CONTEND|PLAYOFF|CUP/i.test(up)) {
      return ptsPct <= 35 ? "TANKING" : "SELLER";
    }
    return up;
  }
  return String(paceStatus || "BUBBLE").toUpperCase();
}

function sidebarStatusLabel(team) {
  const st = team?.standings || {};
  const gp = Number(st.gp ?? team?.gp) || 0;
  const ptsPct =
    team?.pointsPct != null
      ? Number(team.pointsPct)
      : null;
  if (gp >= 10 && ptsPct != null && ptsPct <= 43) {
    return ptsPct <= 35 ? "TANKING" : "SELLER";
  }
  const direction = compactStatusLabel(team?.direction);
  const raw = compactStatusLabel(team?.tradeDirectionLabel || team?.statusLabel || team?.direction || "—");
  if (direction === "REBUILD" && raw === "PLAYOFF") return "REBUILD";
  if (direction === "CONTENDER" && raw === "REBUILD" && !(gp >= 10 && ptsPct != null && ptsPct <= 43)) {
    return "CONTENDER";
  }
  if (/CONTEND|CUP|PLAYOFF/i.test(raw) && gp >= 10 && ptsPct != null && ptsPct <= 43) {
    return ptsPct <= 35 ? "TANKING" : "SELLER";
  }
  return raw;
}

function deriveTradeDirectionLabel(team) {
  const dir = String(team?.direction || "").toUpperCase();
  const ppg = Number(team?.pointsPace) || 0;
  const ptsPct = team?.pointsPct != null ? Number(team.pointsPct) : null;
  const gpHint = Number(team?.standings?.gp) || 0;
  if (gpHint >= 10 && ptsPct != null && ptsPct <= 43) {
    return ptsPct <= 35 ? "TANKING" : "SELLER";
  }
  if (dir && !["LISTENING", "UNKNOWN", ""].includes(dir)) {
    if (/CONTEND/i.test(dir) && ptsPct != null && ptsPct <= 43) return "SELLER";
    return dir;
  }
  const backend = team?.statusLabel || team?.outlookLabel;
  if (backend) {
    const up = String(backend).toUpperCase();
    if (/CONTEND|PLAYOFF|CUP/i.test(up) && ptsPct != null && ptsPct <= 43) return "SELLER";
    return up;
  }
  const ovr = Number(team?.ratings?.overall) || 0;
  const cap = team?.capSpace;
  const inj = Number(team?.injuryCount) || 0;
  const pipeline = Number(team?.pipelineStrength) || 0;
  const pickCount = Number(team?.pickCount) || 0;
  if (inj >= 3) return "INJURED";
  if (cap != null && Number.isFinite(Number(cap)) && Number(cap) < 1.5) return "CAP TIGHT";
  if (ppg >= 1.05 && ovr >= 82) return "CONTENDER";
  if (ppg >= 0.95 && (Number(cap) > 3 || pickCount >= 4)) return "BUYER";
  if (ppg < 0.88 && pipeline >= 78) return "REBUILDER";
  if (ppg < 0.92) return "SELLER";
  return "BUBBLE";
}

function formatCapCompact(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const abs = formatMoneyShort(Math.abs(n));
  return n < 0 ? `-${abs}` : abs;
}

function ovrRingTone(ovr) {
  if (ovr >= 90) return "elite";
  if (ovr >= 84) return "strong";
  if (ovr >= 78) return "normal";
  if (ovr >= 70) return "muted";
  return "warn";
}

function contenderStatusLabel(direction, row, franchiseState) {
  // Frontend fallback when backend outlook_label is unavailable.
  const raw = String(direction || "").toLowerCase();
  const ppg = computePointsPace(row);
  const odds = estimatePlayoffOdds(row, franchiseState);
  if (raw === "rebuild" || ppg < 0.82) return ppg < 0.75 ? "Tanking" : "Rebuilder";
  if (raw === "contender" && ppg > 1.08 && odds >= 75) return "Stanley Cup Contender";
  if (odds >= 55 || ppg > 1.0) return "Playoff Team";
  if (odds >= 25 || ppg > 0.92) return "Bubble Team";
  if (raw === "declining") return "Rebuilder";
  return "Bubble Team";
}

function gmPersonalityLabel(direction, needsSummary, depth) {
  const raw = String(direction || "").toLowerCase();
  const map = {
    rebuild: "Prospect Hoarder",
    contender: "Aggressive Buyer",
    declining: "Cap Dumping",
    emerging: "Win Now",
    seller: "Cap Dumping",
    buyer: "Aggressive Buyer",
  };
  if (map[raw]) return map[raw];
  if (Number(depth?.pipeline_strength) >= 82) return "Prospect Hoarder";
  if (safeArray(needsSummary?.values).some((v) => /pick|prospect/i.test(v))) return "Future Assets";
  return "Pragmatic Dealer";
}

function computeTeamRatings(players) {
  const list = safeArray(players);
  const isFwd = (p) => ["C", "LW", "RW", "F"].includes(String(p?.pos || "").toUpperCase());
  const isDef = (p) => ["D", "LD", "RD"].includes(String(p?.pos || "").toUpperCase());
  const isG = (p) => String(p?.pos || "").toUpperCase() === "G";
  const avg = (arr) =>
    arr.length ? Math.round(arr.reduce((s, p) => s + (Number(p.ovr) || 0), 0) / arr.length) : 0;
  const fwds = list.filter(isFwd).sort((a, b) => (b.ovr || 0) - (a.ovr || 0));
  const defs = list.filter(isDef).sort((a, b) => (b.ovr || 0) - (a.ovr || 0));
  const gs = list.filter(isG).sort((a, b) => (b.ovr || 0) - (a.ovr || 0));
  const offense = avg([...fwds.slice(0, 6), ...defs.slice(0, 2)]);
  const defense = avg(defs.slice(0, 4));
  const goaltending = avg(gs.slice(0, 2));
  const parts = [
    { v: offense, w: 0.42 },
    { v: defense, w: 0.38 },
    { v: goaltending, w: 0.2 },
  ].filter((p) => p.v > 0);
  const weight = parts.reduce((s, p) => s + p.w, 0) || 1;
  const overall = parts.length
    ? Math.round(parts.reduce((s, p) => s + p.v * p.w, 0) / weight)
    : avg(list.slice(0, 18));
  return { overall, offense, defense, goaltending };
}

function computeStandingsRanks(teamId, franchiseState, row) {
  const standings = collectStandings(franchiseState);
  const sorted = [...standings].sort(
    (a, b) => (Number(b.pts) || 0) - (Number(a.pts) || 0) || (Number(b.w) || 0) - (Number(a.w) || 0),
  );
  const leagueRank = sorted.findIndex((r) => String(r.team_id || r.id) === String(teamId)) + 1;
  const conf = row?.conference || row?.conf || "";
  const div = row?.division || "";
  let confRank = Number(row?.conference_rank || row?.conf_rank) || 0;
  let divRank = Number(row?.division_rank || row?.div_rank) || 0;
  if (!confRank && conf) {
    const confTeams = sorted.filter((r) => (r.conference || r.conf) === conf);
    confRank = confTeams.findIndex((r) => String(r.team_id || r.id) === String(teamId)) + 1;
  }
  if (!divRank && div) {
    const divTeams = sorted.filter((r) => r.division === div);
    divRank = divTeams.findIndex((r) => String(r.team_id || r.id) === String(teamId)) + 1;
  }
  return {
    leagueRank: leagueRank || null,
    conferenceRank: confRank || null,
    divisionRank: divRank || null,
    conference: conf || "—",
    division: div || "—",
    gp: Number(row?.gp) || 0,
    pts: Number(row?.pts) || 0,
  };
}

function potentialGradeFromPlayer(row, ovr, age) {
  const explicit = row?.potential || row?.potential_tier || row?.potentialTier;
  if (explicit && String(explicit).length <= 4) return String(explicit).toUpperCase();
  const ps = Number(row?.potential_score ?? row?.potentialScore ?? 0);
  const score = ps || Number(ovr) + Math.max(0, 24 - Number(age || 25));
  if (score >= 92) return "A+";
  if (score >= 88) return "A";
  if (score >= 84) return "B+";
  if (score >= 80) return "B";
  if (score >= 76) return "C+";
  if (score >= 72) return "C";
  return "D";
}

function buildProspectsForTeam(org, teamId, franchiseState, isUser, tradeAssets) {
  const seen = new Set();
  const out = [];
  const add = (row, source, index) => {
    const age = Number(row?.age) || 99;
    if (age > PROSPECT_MAX_AGE) return;
    const pid = String(row?.player_id || row?.id || "");
    if (pid && seen.has(pid)) return;
    if (pid) seen.add(pid);
    const isNhl = source === "nhl";
    const base = normalizePlayerFromRow(row, teamId, franchiseState, isUser, index, tradeAssets);
    out.push({
      ...base,
      type: isNhl ? "player" : "prospect",
      tradeable: isNhl,
      league: isNhl ? "NHL" : String(row?.league || "AHL"),
    });
  };
  safeArray(org.nhl).forEach((row, i) => add(row, "nhl", i));
  safeArray(org.ahl).forEach((row, i) => add(row, "ahl", i + 100));
  return out.sort(compareAssetsByTradeValue);
}

function buildTeamsMeta(franchiseState, tradeAssets, tradeMarket = null) {
  const rb = franchiseState?.roster_browser;
  const userTeamId = String(franchiseState?.user_team_id || franchiseState?.team?.id || "");
  if (!rb?.organizations?.length || !userTeamId) return null;

  const standings = franchiseState?.standings || [];
  const draftYear = resolveDraftYear(franchiseState);
  const tradeContext = { tradeAssets, tradeMarket, deadlinePhase: tradeMarket?.deadline_phase };
  const players = {};
  const picks = {};
  const prospects = {};

  rb.organizations.forEach((org) => {
    const tid = String(org.team_id);
    const isUser = tid === userTeamId;
    players[tid] = safeArray(org.nhl)
      .map((row, index) => normalizePlayerFromRow(row, tid, franchiseState, isUser, index, tradeAssets))
      .sort(compareAssetsByTradeValue);
    prospects[tid] = buildProspectsForTeam(org, tid, franchiseState, isUser, tradeAssets);
    picks[tid] = getOwnedPicks(tid, tradeAssets, rb.organizations);
  });

  const teams = rb.organizations.map((org) => {
    const tid = String(org.team_id);
    const isUser = tid === userTeamId;
    const name = String(org.name || tid);
    const cap = getTeamCapSummary(tid, franchiseState, tradeAssets, players[tid]);
    const ta = tradeAssets?.teams?.[tid] || {};
    const stRow = standingRowForTeam(tid, franchiseState);
    const ranks = computeStandingsRanks(tid, franchiseState, stRow);
    const ratings = computeTeamRatings(players[tid]);
    const direction = getTeamDirection(tid, tradeAssets, franchiseState, standings);
    const needsSummary = ta.needs_summary || {};
    const depth = ta.depth || {};
    const playoffOdds = resolvePlayoffOdds(ta, stRow, franchiseState);
    const statusLabel = resolveStatusLabel(ta, direction, stRow, franchiseState);
    const healthAdjustedRating =
      ta.health_adjusted_rating != null && Number.isFinite(Number(ta.health_adjusted_rating))
        ? Math.min(
            Math.round(Number(ta.health_adjusted_rating)),
            Math.round(Number(ratings.overall) || 99) + 2,
          )
        : ratings.overall;
    const injuryCount = countInjuredPlayers(players[tid]);
    const goalDiff = computeGoalDifferential(stRow);
    const pointsPct = computePointsPct(stRow);
    const pipelineStrength =
      depth?.pipeline_strength != null && Number.isFinite(Number(depth.pipeline_strength))
        ? Math.round(Number(depth.pipeline_strength))
        : null;
    const teamPicks = picks[tid];
    const pickCount = safeArray(teamPicks).length;
    const currentYearPicks = formatCurrentYearPicks(teamPicks, draftYear);
    const intelBase = {
      id: tid,
      name,
      abbr: inferLogoAbbr(tid, name),
      hue: teamHueFromId(tid),
      isUser,
      gp: Number(stRow?.gp) || Number(ranks.gp) || 0,
      record: getTeamRecord(tid, standings),
      direction,
      statusLabel,
      gmPersonality: gmPersonalityLabel(direction, needsSummary, depth),
      capHit: cap.capHit,
      capLimit: cap.capLimit,
      capSpace: cap.capSpace,
      capDetail: ta.cap || {},
      needs: ta.needs || {},
      needsSummary,
      depth,
      rosterCount: safeArray(org.nhl).length,
      prospectCount: safeArray(prospects[tid]).length,
      ratings,
      healthAdjustedRating,
      injuryImpact: ta.injury_impact != null ? Number(ta.injury_impact) : null,
      injuryCount,
      goalDiff,
      pointsPct,
      pipelineStrength,
      pickCount,
      currentYearPicks,
      projectedPoints: ta.projected_points ?? null,
      standingsContext: ta.standings_context || null,
      standings: ranks,
      playoffOdds,
      lastTen: stRow?.last_10 || stRow?.l10 || stRow?.recent_record || "—",
      streak: stRow?.streak || stRow?.current_streak || "—",
      pointsPace: ta.points_pace != null
        ? Number(ta.points_pace).toFixed(2)
        : stRow
          ? computePointsPace(stRow).toFixed(2)
          : "—",
      ppPct: stRow?.pp_pct ?? stRow?.power_play_pct ?? null,
      pkPct: stRow?.pk_pct ?? stRow?.penalty_kill_pct ?? null,
      gfRank: stRow?.gf_rank ?? stRow?.goals_for_rank ?? null,
      gaRank: stRow?.ga_rank ?? stRow?.goals_against_rank ?? null,
      avgAge: stRow?.avg_age ?? stRow?.average_age ?? null,
    };
    const tradeDirectionLabel = deriveTradeDirectionLabel(intelBase);
    return {
      ...intelBase,
      tradeDirectionLabel,
      marketPressure: deriveMarketPressure(
        { ...intelBase, tradeDirectionLabel },
        stRow,
        tradeContext,
      ),
    };
  });

  return {
    teams,
    players,
    picks,
    prospects,
    userTeamId,
    seasonYear: Number(franchiseState?.season_year) || new Date().getFullYear(),
    draftYear,
  };
}

function assetToPayload(asset) {
  if (!asset) return null;
  if (asset.type === "pick") {
    return {
      type: "pick",
      id: String(asset.id || asset.pick_id),
      team: String(asset.teamId || asset.team),
    };
  }
  return {
    type: "player",
    id: String(asset.id),
    team: String(asset.teamId || asset.team),
    retained: Number(asset.retained_pct || asset.retained || 0),
    ntc_waived: Boolean(asset.ntcWaived || asset.ntc_waived),
  };
}

function buildAssetsByTeam(userTeamId, partnerTeamId, userOutgoing, partnerOutgoing) {
  return {
    [String(partnerTeamId)]: safeArray(userOutgoing).filter(Boolean).map(assetToPayload).filter(Boolean),
    [String(userTeamId)]: safeArray(partnerOutgoing).filter(Boolean).map(assetToPayload).filter(Boolean),
  };
}

function tagReason(reason) {
  const r = String(reason || "").toLowerCase();
  if (r.includes("cap") || r.includes("salary") || r.includes("roster maximum")) return { tag: "CAP", text: reason };
  if (r.includes("nmc") || r.includes("ntc") || r.includes("clause") || r.includes("no-trade") || r.includes("no-move"))
    return { tag: "CLAUSE", text: reason };
  if (r.includes("pick") || r.includes("registry") || r.includes("does not own")) return { tag: "PICK", text: reason };
  if (r.includes("rebuild") || r.includes("valuation") || r.includes("reject") || r.includes("interest") || r.includes("premium"))
    return { tag: "VALUE", text: reason };
  if (r.includes("duplicate") || r.includes("not found")) return { tag: "ASSET", text: reason };
  if (r.includes("retained")) return { tag: "RET", text: reason };
  return { tag: "INFO", text: reason };
}

function verdictLabel(verdict) {
  const v = String(verdict || "").toLowerCase();
  const map = {
    accepted: "ACCEPTED",
    rejected: "REJECTED",
    needs_adjustment: "NEEDS ADJUSTMENT",
    cap_illegal: "CAP ILLEGAL",
    roster_illegal: "ROSTER ILLEGAL",
    player_unavailable: "PLAYER UNAVAILABLE",
    asset_not_owned: "ASSET NOT OWNED",
    ntc_nmc_conflict: "NTC/NMC CONFLICT",
    trade_value_too_low: "TRADE VALUE TOO LOW",
    blocked: "BLOCKED",
  };
  return map[v] || (v ? v.replace(/_/g, " ").toUpperCase() : "PENDING");
}

function verdictTone(verdict) {
  const v = String(verdict || "").toLowerCase();
  if (v === "accepted") return "good";
  if (v === "needs_adjustment") return "warn";
  if (v === "rejected" || v.includes("illegal") || v.includes("conflict") || v === "blocked") return "bad";
  return "neutral";
}

function getEvaluationReasons(evaluation) {
  const blocking = safeArray(evaluation?.rejection_reasons);
  const warnings = safeArray(evaluation?.warnings);
  const all = [...blocking, ...warnings.map((w) => `[WARN] ${w}`)];
  return all.map(tagReason);
}

function fanReactionShortLabel(score) {
  const s = Number(score) || 0;
  if (s >= 80) return "Love it";
  if (s >= 60) return "Supportive";
  if (s >= 40) return "Mixed";
  if (s >= 20) return "Risk";
  return "Disaster";
}

function fanReactionCategory(score) {
  const s = Number(score) || 0;
  if (s >= 80) return "Fan Favorite Move";
  if (s >= 60) return "Supportive";
  if (s >= 40) return "Mixed";
  if (s >= 20) return "Backlash Risk";
  return "PR Disaster";
}

function fanRiskReason(userOutgoing) {
  const out = safeArray(userOutgoing).filter(Boolean);
  const star = out.find((p) => p.type !== "pick" && Number(p.ovr) >= 88);
  if (star) return "Core piece moved";
  const young = out.find((p) => p.type !== "pick" && Number(p.age) <= 24 && Number(p.ovr) >= 78);
  if (young) return "Young core moved";
  const first = out.find((p) => p.type === "pick" && Number(p.round) === 1);
  if (first) return "First-round pick moved";
  return "Popular asset moved";
}

function fanReactionFactors(userOutgoing, evaluation, userTeam = null, partnerTeam = null) {
  const factors = [];
  const out = safeArray(userOutgoing).filter(Boolean);
  const userDiv = userTeam?.standings?.division;
  const partnerDiv = partnerTeam?.standings?.division;
  if (userDiv && partnerDiv && userDiv !== "—" && userDiv === partnerDiv) {
    factors.push("RIVAL");
  }
  out.forEach((p) => {
    if (p.type === "pick") {
      if (Number(p.round) === 1) factors.push("1ST PICK");
      else if (Number(p.round) <= 3) factors.push("WEAK RETURN");
      return;
    }
    const ovr = Number(p.ovr) || 0;
    if (ovr >= 88) factors.push("STAR MOVE");
    else if (Number(p.age) <= 24 && ovr >= 78) factors.push("YOUNG CORE");
    else if (ovr >= 82) factors.push("FAN FAVORITE");
  });
  const userNet = Number(evaluation?.asset_breakdown?.user?.net) || 0;
  if (evaluation?.accepted && userNet >= -2) factors.push("FAIR DEAL");
  else if (userNet < -10 || (out.some((p) => Number(p.ovr) >= 82) && userNet < 0)) {
    factors.push("WEAK RETURN");
  }
  if (out.some((p) => p.type === "player" && Number(p.ovr) >= 80) && userNet < -8) {
    factors.push("OVERPAY");
  }
  if (!factors.length && out.length) {
    factors.push(out.some((p) => p.type !== "pick" && Number(p.ovr) >= 80) ? "STAR MOVE" : "PACKAGE MOVE");
  }
  return [...new Set(factors)].slice(0, 3);
}

function tradeOutcomeLabel(evaluation, hasAssets) {
  if (!hasAssets) return "ADD ASSETS";
  if (!evaluation) return "EVALUATING";
  const review = evaluation?.trade_review || {};
  if (review.result_label) return String(review.result_label).toUpperCase();
  const reasons = safeArray(evaluation.rejection_reasons).join(" ").toLowerCase();
  if (reasons.includes("clause") || reasons.includes("ntc") || reasons.includes("nmc")) return "CLAUSE BLOCK";
  if (!evaluation.can_execute && reasons.includes("cap")) return "CAP FAILED";
  if (!evaluation.can_execute) return "BLOCKED";
  if (!evaluation.accepted) {
    const userNet = Number(evaluation?.asset_breakdown?.user?.net) || 0;
    if (userNet < -10 || reasons.includes("value") || reasons.includes("overpay")) return "OVERPAY";
    return "REJECTED";
  }
  if (evaluation.accepted) return "ACCEPTED";
  return "PENDING";
}

function tradeOutcomeTone(evaluation, hasAssets) {
  const label = tradeOutcomeLabel(evaluation, hasAssets);
  if (label === "ACCEPTED" || label === "FAIR DEAL" || label === "CLOSE") return "good";
  if (label === "NEEDS SWEETENER" || label === "OVERPAY" || label === "CLOSE") return "warn";
  if (["BLOCKED", "CAP FAILED", "CLAUSE BLOCK", "REJECTED"].includes(label)) return "bad";
  return "neutral";
}

function findAssetInPackage(asset, leftAssets, rightAssets) {
  if (!asset) return null;
  const key = `${asset.type}-${asset.id}`;
  for (let i = 0; i < leftAssets.length; i++) {
    const a = leftAssets[i];
    if (a && `${a.type}-${a.id}` === key) return { side: "left", slotIndex: i };
  }
  for (let i = 0; i < rightAssets.length; i++) {
    const a = rightAssets[i];
    if (a && `${a.type}-${a.id}` === key) return { side: "right", slotIndex: i };
  }
  return null;
}

function assetMiniTags(item) {
  const tags = [];
  if (item.tradeable === false) tags.push("Blocked");
  else if (item.protection && item.protection !== "None") tags.push(item.protection);
  const ct = String(item.contractType || "").toLowerCase();
  if (ct.includes("entry")) tags.push("ELC");
  if (item.is_injured) tags.push("Injured");
  if (item.tradeValue != null && item.tradeValue >= 75) tags.push("High Value");
  return tags.slice(0, 2);
}

function emptySlots(n = SLOTS) {
  return Array.from({ length: n }, () => null);
}

function padSlots(assets) {
  const list = safeArray(assets).slice(0, SLOTS);
  while (list.length < SLOTS) list.push(null);
  return list;
}

function TradeLogo({ team, size = 168 }) {
  const src = resolveFranchiseTeamLogo({ name: team?.name, team_id: team?.id, id: team?.id }, team?.name);
  if (src) {
    return (
      <img
        className="trade-team-logo-img"
        src={src}
        alt=""
        style={{ width: size, height: size, objectFit: "contain" }}
      />
    );
  }
  return (
    <div className="trade-team-logo-fallback" style={{ width: size, height: size }}>
      {team?.abbr || "?"}
    </div>
  );
}

function PositionIcon({ pos }) {
  const p = String(pos || "F").toUpperCase();
  const cls =
    p === "G" ? "pos-g" : p === "D" || p === "LD" || p === "RD" ? "pos-d" : "pos-f";
  return <span className={`trade-pos-icon ${cls}`}>{p.slice(0, 2)}</span>;
}

function PickIcon({ round, year, className = "" }) {
  return (
    <div className={`trade-pick-icon ${className}`.trim()} aria-hidden="true">
      <span className="trade-pick-icon-round">R{round}</span>
      <span className="trade-pick-icon-year">{year}</span>
    </div>
  );
}

function RatingPill({ label, value, accent }) {
  return (
    <div className="trade-rating-pill">
      <span className="trade-rating-label">{label}</span>
      <span className="trade-rating-value" style={accent ? { color: accent } : undefined}>
        {value ?? "—"}
      </span>
    </div>
  );
}

function parseDragPayload(event) {
  try {
    const raw = event.dataTransfer.getData(DRAG_MIME);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function AssetCard({
  asset,
  side,
  slotIndex,
  accentHue,
  onRemove,
  onRetained,
  draggable = true,
  compact = false,
  onDragStart,
  protectedWarning = "",
}) {
  if (!asset) return null;

  if (asset.type === "pick") {
    const orig = asset.originalTeamAbbr || inferLogoAbbr(asset.original_team_id, asset.original_team_id);
    return (
      <div
        className={`trade-asset-card trade-asset-card-pick ${compact ? "compact" : ""}`}
        draggable={draggable}
        onDragStart={onDragStart}
      >
        <PickIcon round={asset.round} year={asset.year} />
        <div className="trade-asset-card-body">
          <div className="trade-asset-card-name">{asset.year} {roundLabel(asset.round)}</div>
          <div className="trade-asset-card-meta">
            ORIG {orig}
            {asset.protection ? ` · ${asset.protection}` : ""}
          </div>
          <PickValueMeter item={asset} showLabel={false} className="trade-asset-card-pick-meter" />
        </div>
        {onRemove && (
          <button type="button" className="trade-asset-card-remove" onClick={onRemove} aria-label="Remove">
            ×
          </button>
        )}
      </div>
    );
  }

  const retained = Number(asset.retained_pct || 0);
  const capHit = playerPackageCapHit(asset);
  const capDisplay = capHit > 0 ? capHit : Number(asset.capHit) || 0;
  const capImpact = playerCapImpactLabel(side, capHit);
  const headPlayer = ensurePlayerHeadshotFields({
    ...asset,
    position: asset.pos,
    headshot_id: asset.headshot_id,
    headshot_url: asset.headshot_url,
  });
  const clauseMini =
    asset.protection && asset.protection !== "None"
      ? String(asset.protection).replace(/no[- ]?trade/i, "NTC").replace(/no[- ]?movement/i, "NMC").slice(0, 6)
      : null;
  const rumorMod = activeTradeRumorModifier(asset);

  return (
    <div
      className={`trade-asset-card trade-asset-card-player ${compact ? "compact" : ""}`}
      draggable={draggable}
      onDragStart={onDragStart}
    >
      <PlayerHeadshot
        player={headPlayer}
        size={compact ? "sm" : "md"}
        className="trade-asset-headshot"
        flag={null}
        number={null}
      />
      <div className="trade-asset-ovr-focus" title={getOverallTooltip(asset)}>
        <span className="trade-asset-ovr-label">OVR</span>
        <strong className="trade-asset-ovr-number">{displayOvr(asset)}</strong>
        {rumorMod && (
          <span className="trade-asset-cap-impact bad">
            {`Rumors ${rumorMod.amount}`} · {`${rumorMod.gamesRemaining} GP left`} · Trade Heat
          </span>
        )}
      </div>
      <div className="trade-asset-card-body">
        <div className="trade-asset-card-name trade-asset-card-name-row">
          <TradeFlagBadge player={asset} size="md" />
          <span>{asset.name}</span>
          {isFranchisePlayer(asset) ? <em className="trade-asset-franchise-mark">★</em> : null}
        </div>
        {protectedWarning ? (
          <div className="trade-asset-protected-warn">{protectedWarning}</div>
        ) : null}
        <div className="trade-asset-card-meta trade-asset-card-meta-tiny">
          <PositionIcon pos={asset.pos} />
          <span>{asset.age}Y</span>
          <span>{assetCardTermLabel(asset)}</span>
          {clauseMini && <span className="trade-asset-clause-mini">{clauseMini}</span>}
        </div>
        {asset.potentialGrade && (
          <div className="trade-asset-pot-big">POT {asset.potentialGrade}</div>
        )}
      </div>
      <div className="trade-asset-card-right">
        <div className="trade-asset-aav-badge">
          <span className="trade-asset-aav-value">
            {capDisplay > 0 ? formatMoneyShort(capDisplay) : formatPlayerCapLabel(asset)}
          </span>
          <span className="trade-asset-aav-label">AAV</span>
        </div>
        {capImpact && (
          <span className={`trade-asset-cap-impact ${side === "left" ? "good" : "bad"}`}>
            {capImpact}
          </span>
        )}
        {retained > 0 && <span className="trade-ret-badge">{retained}% RET</span>}
        {onRemove && (
          <button type="button" className="trade-asset-card-remove" onClick={onRemove} aria-label="Remove">
            ×
          </button>
        )}
      </div>
    </div>
  );
}

function TradeSlot({
  slotIndex,
  asset,
  side,
  teamId,
  accentHue,
  isDropTarget,
  onDrop,
  onDragOver,
  onDragLeave,
  onRemove,
  onRetained,
  onDragStartFromSlot,
  protectedWarning = "",
}) {
  return (
    <div
      className={`trade-slot ${asset ? "filled" : "empty"} ${isDropTarget ? "drop-active" : ""} ${protectedWarning ? "protected-conflict" : ""}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="trade-slot-index">{slotIndex + 1}</div>
      {asset ? (
        <AssetCard
          asset={asset}
          side={side}
          slotIndex={slotIndex}
          accentHue={accentHue}
          onRemove={onRemove}
          onRetained={onRetained}
          protectedWarning={protectedWarning}
          onDragStart={(e) => onDragStartFromSlot(e, slotIndex, asset)}
        />
      ) : (
        <div className="trade-slot-placeholder">
          <span>DROP ASSET</span>
          <small>Player · Pick · Prospect</small>
        </div>
      )}
    </div>
  );
}

function TradeValueBadge({ item }) {
  return <TradeValueChip item={item} compact />;
}

function assetSortValue(item) {
  const tv = Number(item?.tradeValue ?? item?.value_hint);
  if (Number.isFinite(tv) && tv > 0) return tv;
  if (item?.type === "pick") return pickRoundValueAnchor(item?.round);
  if (item?.type === "player" || item?.type === "prospect") {
    return poolPlayerValueScore(item);
  }
  return talentValueAnchor(roundOverall(item?.ovr)) || 0;
}

function compareAssetsByTradeValue(a, b) {
  const diff = assetSortValue(b) - assetSortValue(a);
  if (diff !== 0) return diff;
  return (Number(b?.ovr) || 0) - (Number(a?.ovr) || 0);
}

function lookupBreakdownAsset(asset, evaluation, side, direction) {
  if (!asset || !evaluation?.asset_breakdown) return null;
  const list = safeArray(evaluation.asset_breakdown[side]?.[direction]);
  const assetId = String(asset.id || asset.player_id || "");
  const nameKey = String(asset.name || "").toLowerCase();
  return list.find((a) => {
    const bid = String(a.asset_id || a.player_id || a.id || "");
    if (assetId && bid && assetId === bid) return true;
    return String(a.name || "").toLowerCase() === nameKey;
  }) || null;
}

function resolveReviewAssetValueItem(asset, evaluation, side, direction) {
  const bdRow = lookupBreakdownAsset(asset, evaluation, side, direction);
  if (bdRow) {
    const tv = Number(bdRow.trade_value ?? bdRow.total ?? bdRow.value);
    if (Number.isFinite(tv) && tv > 0) {
      return {
        tradeValue: tv,
        valueTier: bdRow.value_tier || valueTierFromScore(tv),
      };
    }
  }
  if (asset?.type === "pick") {
    const score = poolPickValueScore(asset);
    return { tradeValue: score, valueTier: valueTierFromScore(score) };
  }
  const score = poolPlayerValueScore(asset);
  return { tradeValue: score, valueTier: valueTierFromScore(score) };
}

function reviewAssetValuePct(valueItem) {
  const raw = Number(valueItem?.tradeValue);
  if (!Number.isFinite(raw)) return 12;
  return clamp(Math.round(raw), 8, 100);
}

function shortenReviewHint(text, max = 36) {
  const s = String(text || "").trim().split(/[.!]/)[0].trim();
  if (!s) return "";
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1).trim()}…`;
}

function packageValuesFromEvaluation(evaluation, userOutgoing, partnerOutgoing) {
  const bal = evaluation?.trade_review?.trade_balance;
  if (bal != null) {
    const out = Number(bal.user_out);
    const inc = Number(bal.user_in);
    if (Number.isFinite(out) && Number.isFinite(inc)) {
      return {
        userGive: Math.round(out * 10) / 10,
        partnerGive: Math.round(inc * 10) / 10,
      };
    }
  }
  const bd = evaluation?.asset_breakdown?.user;
  if (bd) {
    const out = Number(bd.outgoing_total);
    const inc = Number(bd.incoming_total);
    if (Number.isFinite(out) && Number.isFinite(inc)) {
      return {
        userGive: Math.round(out * 10) / 10,
        partnerGive: Math.round(inc * 10) / 10,
      };
    }
  }
  return {
    userGive: Math.round(packageDisplayValue(userOutgoing) * 10) / 10,
    partnerGive: Math.round(packageDisplayValue(partnerOutgoing) * 10) / 10,
  };
}

function partnerWantHintFromBackend(wants, partnerName) {
  if (wants?.source !== "backend") return "";
  const summary = String(wants?.summary || "").trim();
  if (summary && !summary.toLowerCase().includes("no clear")) {
    return shortenReviewHint(summary, 36);
  }
  const chip = safeArray(wants?.chips)[0];
  if (chip) return `${partnerName} wants ${String(chip).toLowerCase()} help`;
  return "";
}

function packageVisualPct(value, peak) {
  const v = Number(value) || 0;
  const p = Math.max(Number(peak) || 0, 1);
  if (v <= 0) return 0;
  return clamp(Math.round((v / p) * 100), 4, 100);
}

function packageBalanceIsEven(userGive, partnerGive) {
  const left = Number(userGive) || 0;
  const right = Number(partnerGive) || 0;
  const diff = Math.abs(left - right);
  const total = Math.max(left + right, 1);
  return diff <= 3 || diff / total <= 0.08;
}

function packageDisplayValue(assets) {
  return safeArray(assets)
    .filter(Boolean)
    .reduce((sum, asset) => {
      if (asset.type === "pick") {
        const raw = Number(asset.tradeValue ?? asset.value_hint);
        return sum + (Number.isFinite(raw) ? raw : 0);
      }
      const tv = Number(asset.tradeValue);
      if (Number.isFinite(tv) && tv > 0) return sum + tv;
      return sum + assetSortValue(asset);
    }, 0);
}

function packageCapDelta(assets) {
  return safeArray(assets)
    .filter(Boolean)
    .reduce((sum, asset) => {
      if (asset.type !== "player") return sum;
      const cap = Number(asset.capHit) || 0;
      const retained = Number(asset.retained_pct || 0);
      return sum + cap * (1 - retained / 100);
    }, 0);
}

/** Live cap space after the assets currently sitting in package slots. */
function projectedTeamCapSpace(team, outgoingAssets, incomingAssets) {
  const base = Number(team?.capSpace);
  if (!Number.isFinite(base)) return null;
  const outgoing = packageCapDelta(outgoingAssets);
  const incoming = packageCapDelta(incomingAssets);
  return base + outgoing - incoming;
}

function playerPackageCapHit(asset) {
  if (!asset || asset.type !== "player") return 0;
  const cap = Number(asset.capHit) || 0;
  const retained = Number(asset.retained_pct || 0);
  return cap * (1 - retained / 100);
}

function assetCardTermLabel(asset) {
  const years = Number(asset?.years) || 0;
  if (years > 0) return `${years}Y`;
  const contractType = String(asset?.contractType || "").toLowerCase();
  if (contractType.includes("entry")) return "ELC";
  return "UFA";
}

function playerCapImpactLabel(side, capHit) {
  const hit = Number(capHit) || 0;
  if (hit <= 0) return "";
  const amt = formatMoneyShort(hit);
  return side === "left" ? `LEAVING ${amt}` : `ARRIVING ${amt}`;
}

function PackageCapStrip({ team, outgoingAssets, incomingAssets, sideLabel = "" }) {
  const baseCap = Number(team?.capSpace);
  const projected = Number.isFinite(baseCap)
    ? projectedTeamCapSpace(team, outgoingAssets, incomingAssets)
    : null;
  const delta =
    Number.isFinite(baseCap) && Number.isFinite(projected) ? projected - baseCap : null;
  const deltaGood = delta == null ? null : delta >= 0;
  const label = sideLabel || (team?.isUser ? "Your space" : "Partner space");
  const nowText = Number.isFinite(baseCap) ? formatMoneyShort(baseCap) : "—";
  const afterText = Number.isFinite(projected) ? formatMoneyShort(projected) : "—";
  const deltaText =
    delta == null
      ? "—"
      : `${delta >= 0 ? "+" : "−"}${formatMoneyShort(Math.abs(delta))}`;

  return (
    <div className="trade-package-cap-strip" title={`${label}: space now → after trade`}>
      <span className="trade-package-cap-label">{label}</span>
      <span className="trade-package-cap-flow">
        <span className="trade-package-cap-now">{nowText}</span>
        <span className="trade-package-cap-arrow" aria-hidden="true">→</span>
        <span className="trade-package-cap-after">{afterText}</span>
      </span>
      <span
        className={`trade-package-cap-delta ${
          deltaGood == null ? "" : deltaGood ? "good" : "bad"
        }`}
      >
        {deltaText}
      </span>
    </div>
  );
}

function packageBalanceQualitative(userGive, partnerGive) {
  const left = Number(userGive) || 0;
  const right = Number(partnerGive) || 0;
  if (left <= 0 && right <= 0) return "—";
  if (packageBalanceIsEven(left, right)) return "Even";
  const diff = Math.abs(left - right);
  const total = Math.max(left + right, 1);
  const ratio = diff / total;
  if (left > right) {
    if (ratio >= 0.35) return "You pay more";
    if (ratio >= 0.18) return "Heavy ask";
    return "Slight overpay";
  }
  if (ratio >= 0.35) return "They pay more";
  if (ratio >= 0.18) return "Light offer";
  return "Close";
}

function gmInterestQualitative(evaluation, partnerTeamId) {
  if (!evaluation) return "—";
  const pct = Number(evaluation.interest_level?.[partnerTeamId]) || 0;
  if (pct >= 0.66) return "High";
  if (pct >= 0.38) return "Medium";
  return "Low";
}

function proposalStatusQualitative(evaluation) {
  if (!evaluation) return "—";
  const reasons = safeArray(evaluation.rejection_reasons).join(" ").toLowerCase();
  if (evaluation.accepted && evaluation.can_execute) return "Accepted";
  if (!evaluation.can_execute) return "Blocked";
  if (!evaluation.accepted) return "Rejected";
  if (reasons.includes("reject")) return "Rejected";
  return "Rejected";
}

function shortReviewVerdict(evaluation) {
  if (!evaluation) return "PENDING";
  if (evaluation.accepted && evaluation.can_execute) return "ACCEPTED";
  if (!evaluation.can_execute) return "BLOCKED";
  if (evaluation.accepted === false) return "REJECTED";
  return "PENDING";
}

function shortReviewVerdictTone(verdict) {
  if (verdict === "ACCEPTED") return "good";
  if (verdict === "BLOCKED" || verdict === "REJECTED") return "bad";
  return "warn";
}

function balanceReviewShort(userGive, partnerGive) {
  const raw = packageBalanceQualitative(userGive, partnerGive);
  const map = {
    Even: "EVEN",
    Close: "CLOSE",
    "You pay more": "YOU PAY",
    "Heavy ask": "HEAVY",
    "They pay more": "THEY PAY",
    "Light offer": "LIGHT",
  };
  return map[raw] || String(raw || "—").toUpperCase().slice(0, 8);
}

function tradeBalancePct(userGive, partnerGive) {
  const left = Number(userGive) || 0;
  const right = Number(partnerGive) || 0;
  const total = Math.max(left + right, 1);
  const diff = Math.abs(left - right) / total;
  if (diff <= 0.08) return 50;
  if (left > right) return clamp(50 + diff * 100, 55, 95);
  return clamp(50 - diff * 100, 5, 45);
}

function gmInterestPct(evaluation, partnerTeamId) {
  const raw = Number(evaluation?.interest_level?.[partnerTeamId]);
  if (Number.isFinite(raw)) return clamp(Math.round(raw * 100), 0, 100);
  const label = gmInterestQualitative(evaluation, partnerTeamId);
  if (label === "High") return 78;
  if (label === "Medium") return 52;
  if (label === "Low") return 24;
  return 0;
}

function tradeReviewReasonSummary(evaluation) {
  const reasons = getEvaluationReasons(evaluation);
  const text = reasons.map((r) => `${r.tag || ""} ${r.text || ""}`).join(" ").toLowerCase();
  if (!evaluation) return "Awaiting result.";
  if (evaluation.accepted && evaluation.can_execute) return "Deal can execute.";
  if (text.includes("cap")) return "Cap does not fit.";
  if (text.includes("clause") || text.includes("ntc") || text.includes("nmc")) return "Clause blocks player.";
  if (text.includes("pick")) return "Pick ownership issue.";
  if (text.includes("roster") || text.includes("slot")) return "Roster limit failed.";
  if (text.includes("value") || text.includes("sweetener") || evaluation.accepted === false) return "Need better return.";
  return "Trade blocked.";
}

function partnerWantChips(partnerTeam, evaluation) {
  const chips = [];
  safeArray(evaluation?.immersion?.partner_needs).forEach((n) => chips.push(n));
  safeArray(partnerTeam?.needsSummary?.needs_short).forEach((n) => chips.push(n));
  safeArray(partnerTeam?.needsSummary?.values).forEach((n) => chips.push(n));
  safeArray(partnerTeam?.needsSummary?.shopping).forEach((n) => chips.push(n));
  return [...new Set(chips)]
    .map((x) => String(x).toUpperCase().replace(/[^A-Z0-9 ]/g, "").trim())
    .filter(Boolean)
    .slice(0, 4);
}

function partnerUntouchableChips(meta, partnerTeamId) {
  return partnerProtectionLists(meta, partnerTeamId).displayNames;
}

/** Soft = high-end core (AI may reject). Hard = legally/unavailable (cannot propose). */
function partnerProtectionLists(meta, partnerTeamId) {
  const players = safeArray(meta?.players?.[partnerTeamId]);
  const soft = [];
  const hard = [];
  players.forEach((p) => {
    const ovr = Number(p.ovr) || 0;
    const tier = String(assetValueLabel(p) || "").toUpperCase();
    const clause = String(p.protection || p.clauseLabel || "").toUpperCase();
    const isNmc = clause.includes("NMC");
    const needsNtcWaive =
      !p.ntcWaived &&
      (Boolean(p.requiresNtcWaive) || (clause.includes("NTC") && !clause.includes("M-NTC")) || clause.includes("M-NTC"));
    const unavailable =
      p.tradeable === false &&
      !needsNtcWaive && // waived path flips tradeable later; NTC without waive is hard via needsNtcWaive
      Boolean(p.tradeBlockReason || isNmc);

    const hardHit = isNmc || needsNtcWaive || unavailable;
    const softHit =
      !hardHit &&
      (ovr >= 88 ||
        (Number(p.age) <= 24 && ovr >= 82) ||
        tier.includes("FRANCHISE") ||
        tier.includes("ELITE"));

    if (hardHit) hard.push(p);
    else if (softHit) soft.push(p);
  });
  const byOvr = (a, b) => (Number(b.ovr) || 0) - (Number(a.ovr) || 0);
  hard.sort(byOvr);
  soft.sort(byOvr);
  const hardNames = hard.slice(0, 6).map((p) => p.name);
  const softNames = soft.slice(0, 6).map((p) => p.name);
  return {
    hardNames,
    softNames,
    displayNames: [...hardNames, ...softNames].slice(0, 6),
  };
}

function partnerHardProtectedConflict(partnerOutgoing, hardNames) {
  const blocked = new Set(safeArray(hardNames).map((n) => String(n).toLowerCase()));
  return safeArray(partnerOutgoing)
    .filter((a) => a?.type === "player" && blocked.has(String(a.name).toLowerCase()))
    .map((a) => a.name);
}

function partnerSoftProtectedConflict(partnerOutgoing, softNames) {
  const flagged = new Set(safeArray(softNames).map((n) => String(n).toLowerCase()));
  return safeArray(partnerOutgoing)
    .filter((a) => a?.type === "player" && flagged.has(String(a.name).toLowerCase()))
    .map((a) => a.name);
}

function capReviewStatus(userCap) {
  const after = Number(userCap?.after_usable ?? userCap?.projectedCapSpace);
  if (!Number.isFinite(after)) return "—";
  return after >= 0 ? "OK" : "BAD";
}

function gmReviewShort(evaluation, partnerTeamId) {
  const raw = gmInterestQualitative(evaluation, partnerTeamId);
  if (raw === "High") return "HIGH";
  if (raw === "Medium") return "MED";
  if (raw === "Low") return "LOW";
  return "—";
}

function fansReviewLabel(fanMeter) {
  const label = fanMeter?.heatLabel || fanHeatLabelFromHeat(fanMeter?.heat ?? (100 - Number(fanMeter?.score ?? fanMeter)));
  return String(label || "—").toUpperCase();
}

function fansReviewTone(fanMeter) {
  const heat = Number(fanMeter?.heat ?? (100 - Number(fanMeter?.score ?? fanMeter))) || 0;
  if (heat >= 55) return "bad";
  if (heat >= 30) return "warn";
  return "good";
}

function fanFactorReviewChips(factors) {
  const out = [];
  safeArray(factors).slice(0, 3).forEach((f) => {
    const s = String(f || "").toLowerCase();
    if (s.includes("star")) out.push("STAR");
    else if (s.includes("pick") || s.includes("1st") || s.includes("first")) out.push("PICK");
    else if (s.includes("captain")) out.push("CAP");
    else if (s.includes("fan")) out.push("FANS");
    else if (s.includes("rental")) out.push("RENT");
    else if (s.includes("rival")) out.push("RIVAL");
  });
  return [...new Set(out)].slice(0, 2);
}

function riskReviewChip(evaluation) {
  const reasons = getEvaluationReasons(evaluation);
  const text = reasons.map((r) => String(r.text || r.tag || r)).join(" ").toLowerCase();
  if (text.includes("cap")) return "CAP";
  if (text.includes("clause") || text.includes("ntc") || text.includes("nmc")) return "CLAUSE";
  if (text.includes("pick")) return "PICK";
  if (text.includes("value")) return "VALUE";
  if (text.includes("roster") || text.includes("slot")) return "ROSTER";
  return "LOW";
}

function reviewReasonChips(evaluation) {
  const chips = [];
  const reasons = getEvaluationReasons(evaluation);
  const text = reasons.map((r) => `${r.tag || ""} ${r.text || ""}`).join(" ").toLowerCase();
  if (text.includes("cap")) chips.push("CAP");
  if (text.includes("clause") || text.includes("ntc") || text.includes("nmc")) chips.push("CLAUSE");
  if (text.includes("pick")) chips.push("PICK");
  if (text.includes("value")) chips.push("VALUE");
  if (text.includes("star")) chips.push("STAR");
  safeArray(evaluation?.warnings).forEach((w) => {
    if (String(w).toLowerCase().includes("fan")) chips.push("FANS");
  });
  return [...new Set(chips)].slice(0, 3);
}

function leagueReviewStatus(tradeHistory, tradeMarket) {
  const recent = safeArray(tradeHistory).length
    ? safeArray(tradeHistory)
    : safeArray(tradeMarket?.recent_trades);
  return recent.length ? "ACTIVE" : "QUIET";
}

function capReviewTone(userCap) {
  return capReviewStatus(userCap) === "BAD" ? "bad" : "good";
}

function riskReviewTone(evaluation) {
  const chip = riskReviewChip(evaluation);
  return chip === "LOW" ? "good" : "bad";
}

function sortPoolByValue(list) {
  return [...safeArray(list)].sort(compareAssetsByTradeValue);
}

function pickRangeDisplay(pick) {
  const r = String(pick?.projectedRange || pick?.projected_range || "").trim().toUpperCase();
  if (!r || r === "UNKNOWN" || r === "?") return null;
  return r;
}

function pickOutlookLabel(pick) {
  const range = pickRangeDisplay(pick);
  if (range === "LOTTERY") return "Lottery upside";
  if (range === "TOP 10") return "Top-10 range";
  if (range === "MID") return "Middle of round";
  if (range === "LATE") return "Late first";
  if (range === "CONTENDER") return "Contender slot";
  return "Draft capital";
}

function groupPicksByDraftYear(picks) {
  const byYear = {};
  safeArray(picks).forEach((pick) => {
    const year = Number(pick.year) || 0;
    if (!byYear[year]) byYear[year] = [];
    byYear[year].push(pick);
  });

  return Object.keys(byYear)
    .map(Number)
    .sort((a, b) => a - b)
    .map((year) => {
      const yearPicks = [...byYear[year]].sort((a, b) => {
        const rd = (Number(a.round) || 0) - (Number(b.round) || 0);
        if (rd !== 0) return rd;
        return assetSortValue({ ...b, type: "pick" }) - assetSortValue({ ...a, type: "pick" });
      });
      return { year, count: yearPicks.length, picks: yearPicks };
    });
}

function PickValueMeter({ item, showLabel = true, className = "" }) {
  const valueItem = {
    tradeValue: item?.tradeValue ?? item?.value_hint,
    valueTier: item?.valueTier,
  };
  const pct = assetValuePct(valueItem);
  const tierClass = assetValueTierClass(valueItem);
  return (
    <div className={`trade-pick-value-meter ${tierClass} ${className}`.trim()}>
      {showLabel && <span className="trade-pick-value-label">VALUE</span>}
      <div className="trade-pick-value-track" aria-hidden="true">
        <div className="trade-pick-value-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function DraftPickYearSection({
  group,
  side,
  teamId,
  teamLookup,
  usedIds,
  onDragStart,
  onQuickAdd,
  onAssetClick,
}) {
  return (
    <>
      <div className="trade-pick-year-divider">
        <span className="trade-pick-year-divider-title">{group.year} DRAFT</span>
        <span className="trade-pick-year-divider-count">
          {group.count} PICK{group.count === 1 ? "" : "S"}
        </span>
      </div>
      {group.picks.map((pick) => (
        <AssetPoolRow
          key={pick.id}
          item={pick}
          side={side}
          teamId={teamId}
          teamLookup={teamLookup}
          usedIds={usedIds}
          onDragStart={onDragStart}
          onQuickAdd={onQuickAdd}
          onAssetClick={onAssetClick}
        />
      ))}
    </>
  );
}

function PlayerValueFocus({ item }) {
  const tvRaw =
    item?.type === "pick"
      ? poolPickValueScore(item)
      : poolPlayerValueScore(item);
  const tv = Number.isFinite(Number(tvRaw)) && Number(tvRaw) > 0 ? Number(tvRaw) : null;
  const pct = tradeValueBarPct(tv || 0);

  const tier = valueTierFromScore(tv || 0).toLowerCase().replace(/\s+/g, "-");
  const isPick = item?.type === "pick";

  return (
    <div className={`trade-player-value-focus value-${tier} ${isPick ? "is-pick-value" : ""}`.trim()}>
      <div className="trade-player-value-head">
        <span>VALUE</span>
        <strong>{tv != null ? tv.toFixed(1) : "—"}</strong>
      </div>
      <div className="trade-player-value-track" aria-hidden="true">
        <div className="trade-player-value-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function AssetPoolRow({
  item,
  side,
  teamId,
  teamLookup,
  teamLookupByAbbr,
  usedIds,
  onDragStart,
  onQuickAdd,
  onAssetClick,
}) {
  const key = `${item.type}-${item.id}`;
  const used = usedIds.has(key);
  const isAhlProspect = item.type === "prospect";
  const blocked = item.tradeable === false;
  const ntcLocked = blocked && (item.requiresNtcWaive || String(item.protection || item.clauseLabel || "").toUpperCase().includes("NTC")) && !item.ntcWaived;
  const draggable = !used && !blocked;
  const originalTeamId = String(item?.original_team_id || "");
  const originalTeamById = teamLookup?.[originalTeamId] || null;
  const pickOrigAbbr =
    normalizeNhlAbbr(item?.original_team_id) ||
    normalizeNhlAbbr(item?.originalTeamAbbr) ||
    normalizeNhlAbbr(originalTeamById?.abbr) ||
    "";
  const originalTeam = originalTeamById || (pickOrigAbbr ? teamLookupByAbbr?.[pickOrigAbbr] || null : null);
  const pickOrigLogo = resolveFranchiseTeamLogo(
    {
      team_id: originalTeam?.id ?? item?.original_team_id,
      id: originalTeam?.id ?? item?.original_team_id,
      name: originalTeam?.name || pickOrigAbbr,
      abbr: originalTeam?.abbr || pickOrigAbbr,
    },
    originalTeam?.name || pickOrigAbbr,
  );
  const blockTitle =
    item.ntcWaived
      ? "NTC waived — slightly reduced trade value"
      : item.tradeBlockReason ||
        (isAhlProspect ? "AHL prospect — NHL roster required" : blocked ? "Not tradeable" : "");

  const handleClick = (e) => {
    if (e.detail > 1) return;
    if (used || blocked || isAhlProspect) {
      onAssetClick?.(item, side, teamId);
      return;
    }
    onQuickAdd?.(item);
  };

  const statusPill = used ? (
    <span className="trade-pool-status-pill in-deal">IN DEAL</span>
  ) : item.ntcWaived ? (
    <span className="trade-pool-status-pill waived" title={blockTitle}>
      WAIVED
    </span>
  ) : ntcLocked ? (
    <span className="trade-pool-status-pill locked" title={blockTitle || "Ask to waive NTC"}>
      NTC
    </span>
  ) : blocked || isAhlProspect ? (
    <span className="trade-pool-status-pill locked" title={blockTitle}>
      LOCKED
    </span>
  ) : null;

  const seasonLine =
    item.type !== "pick" && (item.gp != null || item.pts != null || item.war != null)
      ? [
          item.gp != null ? `${item.gp} GP` : null,
          item.g != null || item.a != null || item.pts != null
            ? `${Number(item.g) || 0}-${Number(item.a) || 0}-${Number(item.pts) || 0}`
            : null,
          item.war != null && Number.isFinite(Number(item.war))
            ? `WAR ${Number(item.war).toFixed(2)}`
            : null,
        ]
          .filter(Boolean)
          .join(" · ")
      : "";

  const pickTv = Number(item?.tradeValue ?? item?.value_hint);
  const pickTvLabel =
    item.type === "pick" && Number.isFinite(pickTv) && pickTv > 0
      ? `TV ${pickTv.toFixed(1)}`
      : "";

  return (
    <div
      className={`trade-pool-row trade-pool-compact trade-pool-${item.type || "player"} ${used ? "used in-deal-row" : ""} ${isAhlProspect || blocked ? "view-only inactive" : ""}`}
      draggable={draggable}
      onDragStart={draggable ? (e) => onDragStart(e, item, "pool") : undefined}
      onDoubleClick={() => draggable && onQuickAdd(item)}
      onClick={handleClick}
      title={
        used
          ? "Already in package — remove from a deal slot"
          : blockTitle || "Click to add · drag to slot"
      }
    >
      {item.type === "pick" ? (
        <>
          <div className="trade-player-list-photo">
            <PickIcon round={item.round} year={item.year} className="trade-pool-pick-icon" />
          </div>

          <div className="trade-player-list-main">
            <div className="trade-player-list-name-row">
              <strong>
                {item.year} {roundLabel(item.round)}
                {pickTvLabel ? ` · ${pickTvLabel}` : ""}
              </strong>
            </div>
            <div className="trade-pick-origin-line">
              {pickOrigLogo ? (
                <img className="trade-pick-origin-logo" src={pickOrigLogo} alt={pickOrigAbbr || "TEAM"} />
              ) : (
                <span className="trade-pick-origin-fallback">{pickOrigAbbr || "TEAM"}</span>
              )}
            </div>

            <div className="trade-player-list-mid trade-pick-list-mid">
              <PlayerValueFocus item={item} />
            </div>
          </div>

          {statusPill}
        </>
      ) : (
        <>
          <div className="trade-player-list-photo">
            <PlayerHeadshot
              player={ensurePlayerHeadshotFields({ ...item, position: item.pos })}
              size="sm"
              className="trade-pool-headshot trade-player-clean-headshot"
              flag={null}
              number={null}
            />
          </div>

          <div className="trade-player-list-main">
            <div className="trade-player-list-name-row">
              <TradeFlagBadge player={item} size="sm" />
              <strong>{item.name}</strong>
            </div>

            <div className="trade-player-list-mid">
              <div className="trade-player-list-details">
                <span className="trade-player-detail-pos">{item.pos || "—"}</span>
                <span className="trade-player-detail-age">{Number(item.age) ? `${item.age}Y` : "—"}</span>
                <span className="trade-player-detail-cap">{formatPlayerCapLabel(item)}</span>
              </div>

              <PlayerValueFocus item={item} />
            </div>
            {seasonLine ? <div className="trade-pool-season-line">{seasonLine}</div> : null}
          </div>

          <div className="trade-player-ovr-tower">
            <span>OVR</span>
            <strong>{displayOvr(item)}</strong>
            {activeTradeRumorModifier(item) && (
              <small>
                {`Rumors ${activeTradeRumorModifier(item).amount}`} · {`${activeTradeRumorModifier(item).gamesRemaining} GP`} · Trade Heat
              </small>
            )}
          </div>
          {statusPill}
        </>
      )}
    </div>
  );
}

function AssetPool({
  teamId,
  side,
  meta,
  usedIds,
  onDragStart,
  onQuickAdd,
  onAssetClick,
}) {
  const [tab, setTab] = useState("NHL");
  const players = safeArray(meta?.players?.[teamId]);
  const picks = safeArray(meta?.picks?.[teamId]);
  const prospects = safeArray(meta?.prospects?.[teamId]);
  const teamLookup = useMemo(
    () => Object.fromEntries(safeArray(meta?.teams).map((t) => [String(t.id), t])),
    [meta?.teams],
  );
  const teamLookupByAbbr = useMemo(
    () =>
      Object.fromEntries(
        safeArray(meta?.teams)
          .map((t) => [normalizeNhlAbbr(t?.abbr || t?.id || ""), t])
          .filter(([abbr]) => Boolean(abbr)),
      ),
    [meta?.teams],
  );

  const rows = useMemo(() => {
    if (tab === "PICKS") return [];
    if (tab === "YOUTH") {
      return sortPoolByValue(prospects.map((p) => ({ ...p, type: p.type || "prospect" })));
    }
    return sortPoolByValue(players.map((p) => ({ ...p, type: "player" })));
  }, [tab, players, prospects]);

  const pickYearGroups = useMemo(() => {
    if (tab !== "PICKS") return [];
    return groupPicksByDraftYear(picks.map((p) => ({ ...p, type: "pick" })));
  }, [tab, picks]);

  return (
    <div className="trade-asset-pool">
      <div className="trade-pool-tabs">
        {[
          ["NHL", players.length],
          ["PICKS", picks.length],
          ["YOUTH", prospects.length],
        ].map(([label, count]) => (
          <button
            key={label}
            type="button"
            className={tab === label ? "active" : ""}
            onClick={() => {
              setTab(label);
            }}
          >
            {label} ({count})
          </button>
        ))}
      </div>
      <div className="trade-pool-list">
        {tab === "PICKS" ? (
          pickYearGroups.length ? (
            pickYearGroups.map((group) => (
              <DraftPickYearSection
                key={group.year}
                group={group}
                side={side}
                teamId={teamId}
                teamLookup={teamLookup}
                teamLookupByAbbr={teamLookupByAbbr}
                usedIds={usedIds}
                onDragStart={onDragStart}
                onQuickAdd={onQuickAdd}
                onAssetClick={onAssetClick}
              />
            ))
          ) : (
            <div className="trade-pool-empty">No picks loaded</div>
          )
        ) : rows.length ? (
          rows.map((item) => (
            <AssetPoolRow
              key={`${item.type}-${item.id}`}
              item={item}
              side={side}
              teamId={teamId}
              teamLookup={teamLookup}
              teamLookupByAbbr={teamLookupByAbbr}
              usedIds={usedIds}
              onDragStart={onDragStart}
              onQuickAdd={onQuickAdd}
              onAssetClick={onAssetClick}
            />
          ))
        ) : (
          <div className="trade-pool-empty">No {tab.toLowerCase()} loaded</div>
        )}
      </div>
    </div>
  );
}

function CompactBackendFeedback({ evaluation, expanded, onToggle }) {
  const reasons = getEvaluationReasons(evaluation);
  if (!reasons.length) return null;
  const visible = reasons.slice(0, 3);
  const extra = reasons.length - visible.length;

  return (
    <div className="trade-hub-feedback">
      {visible.map((r, i) => (
        <span key={`${r.tag}-${i}`} className={`trade-hub-tag trade-hub-tag-${r.tag.toLowerCase()}`}>
          [{r.tag}] {r.text}
        </span>
      ))}
      {extra > 0 && !expanded && (
        <button type="button" className="trade-hub-tag-more" onClick={onToggle}>
          +{extra} MORE
        </button>
      )}
      {expanded &&
        reasons.slice(3).map((r, i) => (
          <span key={`x-${r.tag}-${i}`} className={`trade-hub-tag trade-hub-tag-${r.tag.toLowerCase()}`}>
            [{r.tag}] {r.text}
          </span>
        ))}
    </div>
  );
}

function TradeReportPanel({ evaluation, userTeamId, partnerTeam }) {
  if (!evaluation || !userTeamId) return null;
  const verdict = evaluation.verdict || (evaluation.accepted ? "accepted" : evaluation.can_execute ? "rejected" : "blocked");
  const tone = verdictTone(verdict);
  const userBd = evaluation.asset_breakdown?.user || {};
  const conf = Math.round((Number(evaluation.scouting_confidence) || 1) * 100);
  const counters = safeArray(evaluation.suggested_counteroffers);

  return (
    <div className="trade-hub-report-panel">
      <div className={`trade-hub-verdict trade-hub-verdict-${tone}`}>
        {verdictLabel(verdict)}
      </div>
      <p className="trade-hub-explanation">{evaluation.explanation || "Add assets to evaluate this trade."}</p>
      <div className="trade-hub-report-grid">
        <div className="trade-hub-report-stat">
          <span className="trade-hub-report-label">YOU GET</span>
          <span className="trade-hub-report-value">{Math.round(Number(userBd.incoming_total) || 0)}</span>
        </div>
        <div className="trade-hub-report-stat">
          <span className="trade-hub-report-label">YOU GIVE</span>
          <span className="trade-hub-report-value">{Math.round(Number(userBd.outgoing_total) || 0)}</span>
        </div>
        <div className="trade-hub-report-stat">
          <span className="trade-hub-report-label">NET</span>
          <span className={`trade-hub-report-value ${Number(userBd.net) >= 0 ? "pos" : "neg"}`}>
            {Number(userBd.net) >= 0 ? "+" : ""}{Math.round(Number(userBd.net) || 0)}
          </span>
        </div>
        <div className="trade-hub-report-stat">
          <span className="trade-hub-report-label">SCOUT CONF</span>
          <span className="trade-hub-report-value">{conf}%</span>
        </div>
      </div>
      {counters.length > 0 && (
        <div className="trade-hub-counteroffers">
          <div className="trade-hub-counter-title">SUGGESTED ADJUSTMENTS</div>
          {counters.map((c, i) => (
            <div key={i} className="trade-hub-counter-chip">
              <strong>{c.label}</strong> — {c.explanation}
            </div>
          ))}
        </div>
      )}
      {partnerTeam && evaluation.immersion?.partner_needs?.length > 0 && (
        <div className="trade-hub-immersion-line">
          {partnerTeam.abbr} needs: {evaluation.immersion.partner_needs.join(", ")}
        </div>
      )}
    </div>
  );
}

function CapImpactPanel({ evaluation, userTeamId, partnerId }) {
  if (!evaluation?.cap_impact) return null;
  const userCap = evaluation.cap_impact[userTeamId];
  const partnerCap = evaluation.cap_impact[partnerId];
  if (!userCap && !partnerCap) return null;

  const Card = ({ label, cap }) => {
    if (!cap) return null;
    const ok = Number(cap.after_usable ?? cap.projectedCapSpace) >= 0;
    const incoming = cap.incoming_cap_m ?? cap.incoming;
    const outgoing = cap.outgoing_cap_m ?? cap.outgoing;
    return (
      <div className={`trade-hub-cap-card ${ok ? "" : "bad"}`}>
        <div className="trade-hub-cap-card-label">{label}</div>
        <div className="trade-hub-cap-big">{formatMoneyM(cap.after_usable ?? cap.projectedCapSpace)}</div>
        <div className="trade-hub-cap-sub">
          {formatMoneyM(cap.before_usable ?? cap.snapshot?.usableCapSpace)} → after trade
          {cap.delta != null && (
            <span className={Number(cap.delta) >= 0 ? "pos" : "neg"}>
              {" "}({Number(cap.delta) >= 0 ? "+" : ""}{formatMoneyShort(cap.delta)})
            </span>
          )}
        </div>
        {cap.after_deadline_space != null && (
          <div className="trade-hub-cap-sub muted">
            Deadline accrual: {formatMoneyM(cap.after_deadline_space)}
            {cap.proration_factor != null && cap.proration_factor < 0.99 && (
              <span> · prorate {(Number(cap.proration_factor) * 100).toFixed(0)}%</span>
            )}
          </div>
        )}
        {cap.ltir_relief_used && (
          <div className="trade-hub-cap-sub warn">Fits under LTIR effective limit</div>
        )}
        {(incoming != null || outgoing != null) && (
          <div className="trade-hub-cap-flow">
            {outgoing != null && <span>Out {formatMoneyShort(outgoing)}</span>}
            {incoming != null && <span>In {formatMoneyShort(incoming)}</span>}
            {cap.retained_m != null && Number(cap.retained_m) > 0 && (
              <span>Retained {formatMoneyShort(cap.retained_m)}</span>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="trade-hub-cap-panel">
      <div className="trade-hub-panel-title">CAP IMPACT</div>
      <div className="trade-hub-cap-cards">
        <Card label="YOUR TEAM" cap={userCap} />
        <Card label="PARTNER" cap={partnerCap} />
      </div>
    </div>
  );
}

function TeamNeedsPanel({ team, evaluation }) {
  if (!team) return null;
  const summary = team.needsSummary || {};
  const needs = safeArray(summary.needs_short);
  const shopping = safeArray(summary.shopping);
  const values = safeArray(summary.values);
  const impact = evaluation?.team_needs_impact?.[team.id];

  return (
    <div className="trade-hub-needs-panel">
      <div className="trade-hub-panel-title">{team.abbr} PROFILE</div>
      <div className="trade-hub-chip-row">
        <span className="trade-hub-chip trade-hub-chip-window">{team.direction}</span>
        {needs.map((n) => (
          <span key={n} className="trade-hub-chip trade-hub-chip-need">NEED: {n}</span>
        ))}
      </div>
      {shopping.length > 0 && (
        <div className="trade-hub-needs-line">Shopping: {shopping.join(" · ")}</div>
      )}
      {values.length > 0 && (
        <div className="trade-hub-needs-line">Values: {values.join(" · ")}</div>
      )}
      {impact?.strengthens?.length > 0 && (
        <div className="trade-hub-needs-line ok">Strengthens: {impact.strengthens.join(", ")}</div>
      )}
      {impact?.weakens?.length > 0 && (
        <div className="trade-hub-needs-line warn">Weakens: {impact.weakens.join(", ")}</div>
      )}
    </div>
  );
}

function ValueBreakdownPanel({ evaluation, userTeamId }) {
  const assets = evaluation?.asset_breakdown?.user;
  if (!assets) return null;

  const renderAsset = (a, sign) => {
    const valueItem = {
      tradeValue: a.trade_value ?? a.total,
      valueTier: a.value_tier,
    };
    return (
      <div key={`${sign}-${a.asset_id || a.name}`} className={`trade-hub-breakdown-row ${sign === "+" ? "in" : "out"}`}>
        <span>{sign} {a.name}</span>
        <TradeValueChip item={valueItem} compact />
      </div>
    );
  };

  return (
    <div className="trade-hub-breakdown-panel">
      <div className="trade-hub-panel-title">PACKAGE ASSETS</div>
      <div className="trade-hub-breakdown-list">
        {safeArray(assets.outgoing).map((a) => renderAsset(a, "−"))}
        {safeArray(assets.incoming).map((a) => renderAsset(a, "+"))}
      </div>
    </div>
  );
}

function LeaguePulsePanel({ market, partnerTeam }) {
  if (!market) return null;
  const recent = safeArray(market.recent_trades).slice(0, 3);
  return (
    <div className="trade-hub-league-panel">
      <div className="trade-hub-panel-title">LEAGUE PULSE</div>
      <div className="trade-hub-chip-row">
        <span className={`trade-hub-chip trade-hub-chip-market-${String(market.market_temperature || "cool").toLowerCase()}`}>
          MARKET: {market.market_temperature || "Cool"}
        </span>
        {partnerTeam && (
          <span className="trade-hub-chip">{partnerTeam.abbr} · {partnerTeam.direction}</span>
        )}
      </div>
      {recent.length > 0 ? (
        <div className="trade-hub-recent-trades">
          {recent.map((t, i) => (
            <div key={i} className="trade-hub-recent-line">
              {t.headline || t.summary || "League trade completed"}
            </div>
          ))}
        </div>
      ) : (
        <div className="trade-hub-recent-line muted">No recent trades on record</div>
      )}
    </div>
  );
}

function FanReactionBadge({ score, compact = true }) {
  const s = Number(score) || 0;
  const label = fanReactionShortLabel(s);
  const tone = s >= 70 ? "high" : s < 40 ? "low" : "mid";
  if (compact) {
    return (
      <span className={`trade-fan-badge trade-fan-badge-${tone}`} title={`Fans: ${fanReactionCategory(s)} (${s}%)`}>
        Fans: {label}
      </span>
    );
  }
  return (
    <div className={`trade-fan-meter trade-fan-meter-${tone}`}>
      <div className="trade-fan-meter-head">
        <span>Fan Pulse</span>
        <strong>{s}%</strong>
      </div>
      <div className="trade-fan-meter-track">
        <div style={{ width: `${s}%` }} />
      </div>
      <span className="trade-fan-meter-label">{fanReactionCategory(s)}</span>
    </div>
  );
}

// Fan backlash meter — reflects backend fan_reaction when available.
function FanVitriolMeter({ fanData, score, hasAssets, compact = false }) {
  const data = fanData || {};
  const safeScore = Number.isFinite(Number(data.score ?? score)) ? Number(data.score ?? score) : 50;
  const vitriol = hasAssets
    ? Number.isFinite(Number(data.heat))
      ? Number(data.heat)
      : clamp(100 - safeScore, 0, 100)
    : 0;
  const label = hasAssets ? (data.heatLabel || fanHeatLabelFromHeat(vitriol)) : "—";
  let tone = "low";
  if (vitriol >= 75) tone = "high";
  else if (vitriol >= 55) tone = "mid";
  else if (vitriol >= 30) tone = "warn";
  const factors = compact ? [] : safeArray(data.factors).slice(0, 2);

  return (
    <div className={`trade-fan-vitriol trade-fan-vitriol-${tone} ${compact ? "is-compact" : ""}`.trim()}>
      <div className="trade-fan-vitriol-head">
        <span>FAN HEAT</span>
        <strong>{label}</strong>
      </div>
      <div className="trade-fan-vitriol-track" aria-hidden="true">
        <div className="trade-fan-vitriol-fill" style={{ width: `${vitriol}%` }} />
      </div>
      {factors.length > 0 ? <FanReasonChips factors={factors} /> : null}
    </div>
  );
}

function TradeOutcomeBadge({ evaluation, hasAssets }) {
  const label = tradeOutcomeLabel(evaluation, hasAssets);
  const tone = tradeOutcomeTone(evaluation, hasAssets);
  return <span className={`trade-outcome-badge trade-outcome-${tone}`}>{label}</span>;
}

function tradeValueComparisonLabel(userGive, partnerGive, partnerAbbr) {
  const left = Number(userGive) || 0;
  const right = Number(partnerGive) || 0;

  if (left <= 0 && right <= 0) {
    return {
      headline: "BUILD DEAL",
      leftLabel: "—",
      rightLabel: "—",
      leftTone: "neutral",
      rightTone: "neutral",
    };
  }

  if (left <= 0 || right <= 0) {
    return {
      headline: "ADD BOTH SIDES",
      leftLabel: left > 0 ? left.toFixed(1) : "—",
      rightLabel: right > 0 ? right.toFixed(1) : "—",
      leftTone: "neutral",
      rightTone: "neutral",
    };
  }

  if (packageBalanceIsEven(left, right)) {
    return {
      headline: "EVEN",
      leftLabel: left.toFixed(1),
      rightLabel: right.toFixed(1),
      leftTone: "even",
      rightTone: "even",
    };
  }

  if (left > right) {
    return {
      headline: "YOU PAY MORE",
      leftLabel: left.toFixed(1),
      rightLabel: right.toFixed(1),
      leftTone: "higher",
      rightTone: "lower",
    };
  }

  return {
    headline: partnerAbbr ? `${partnerAbbr} PAYS MORE` : "THEY PAY MORE",
    leftLabel: left.toFixed(1),
    rightLabel: right.toFixed(1),
    leftTone: "lower",
    rightTone: "higher",
  };
}

function FacingTradeValueBars({
  userGive,
  partnerGive,
  userGivePct,
  partnerGivePct,
  partnerAbbr,
  hasAssets,
}) {
  const comparison = tradeValueComparisonLabel(userGive, partnerGive, partnerAbbr);
  const leftWidth = hasAssets ? userGivePct : 0;
  const rightWidth = hasAssets ? partnerGivePct : 0;
  const net = (Number(partnerGive) || 0) - (Number(userGive) || 0);
  const gap = Math.abs((Number(userGive) || 0) - (Number(partnerGive) || 0));
  const showMeta = hasAssets && ((Number(userGive) || 0) > 0 || (Number(partnerGive) || 0) > 0);

  return (
    <div className="trade-clear-value">
      <div className="trade-clear-value-head">
        <span>YOU SEND</span>
        <strong>{comparison.headline}</strong>
        <span>{partnerAbbr || "THEY"} SENDS</span>
      </div>

      <div className="trade-clear-value-body">
        <div className={`trade-clear-side trade-clear-left ${comparison.leftTone}`}>
          <div className="trade-clear-side-label">
            <span>YOU</span>
            <strong>TV {comparison.leftLabel}</strong>
          </div>
          <div className="trade-clear-track">
            <div
              className="trade-clear-fill trade-clear-fill-left"
              style={{ width: `${leftWidth}%` }}
            />
          </div>
        </div>

        <div className={`trade-clear-side trade-clear-right ${comparison.rightTone}`}>
          <div className="trade-clear-side-label">
            <span>{partnerAbbr || "THEY"}</span>
            <strong>TV {comparison.rightLabel}</strong>
          </div>
          <div className="trade-clear-track">
            <div
              className="trade-clear-fill trade-clear-fill-right"
              style={{ width: `${rightWidth}%` }}
            />
          </div>
        </div>
      </div>

      {showMeta ? (
        <div className="trade-clear-value-meta">
          Net {net >= 0 ? "+" : "−"}
          {Math.abs(net).toFixed(1)} · Gap {gap.toFixed(1)}
        </div>
      ) : null}
    </div>
  );
}

function DynamicTradeAnalysis({
  partnerAbbr,
  hasProposed,
  evaluation,
  userOutgoing,
  partnerOutgoing,
  fanMeter,
  fanData,
  onReviewClick,
  blockDetail = null,
  userTeamId = "",
  partnerId = "",
}) {
  const { userGive, partnerGive } = hasProposed && evaluation
    ? packageValuesFromEvaluation(evaluation, userOutgoing, partnerOutgoing)
    : {
      userGive: Math.round(packageDisplayValue(userOutgoing) * 10) / 10,
      partnerGive: Math.round(packageDisplayValue(partnerOutgoing) * 10) / 10,
    };
  const peak = Math.max(userGive, partnerGive, 1);
  const userGivePct = packageVisualPct(userGive, peak);
  const partnerGivePct = packageVisualPct(partnerGive, peak);
  const bothSides = userOutgoing.length > 0 && partnerOutgoing.length > 0;
  const hasAnyAssets = userOutgoing.length > 0 || partnerOutgoing.length > 0;

  let panelHeadline = "PACKAGE BALANCE";
  if (!hasAnyAssets) panelHeadline = "BUILD DEAL";
  else if (!bothSides) panelHeadline = "ADD BOTH SIDES";

  // Desk stays lean — full cap cards live in Trade Review.
  const capLine =
    hasProposed && evaluation?.cap_impact
      ? (() => {
          const yours = evaluation.cap_impact[userTeamId];
          const theirs = evaluation.cap_impact[partnerId];
          const y = yours?.after_usable ?? yours?.projectedCapSpace;
          const t = theirs?.after_usable ?? theirs?.projectedCapSpace;
          if (y == null && t == null) return null;
          return (
            <div className="trade-desk-cap-line">
              <span>After: you {y != null ? formatMoneyM(y) : "—"}</span>
              <span>·</span>
              <span>{partnerAbbr || "Them"} {t != null ? formatMoneyM(t) : "—"}</span>
            </div>
          );
        })()
      : null;

  return (
    <section className="trade-analysis-panel trade-analysis-rink">
      <div className="trade-analysis-rink-head">
        <span className="trade-analysis-headline">
          {hasProposed ? "TRADE RESULT" : panelHeadline}
        </span>
        {hasProposed && bothSides && (
          <button
            type="button"
            className="trade-analysis-review-btn"
            onClick={() => onReviewClick?.()}
          >
            Review
          </button>
        )}
      </div>

      <FacingTradeValueBars
        userGive={userGive}
        partnerGive={partnerGive}
        userGivePct={userGivePct}
        partnerGivePct={partnerGivePct}
        partnerAbbr={partnerAbbr}
        hasAssets={bothSides}
      />

      {capLine}

      <FanVitriolMeter
        fanData={fanData || fanMeter}
        score={Number(fanMeter?.score ?? fanMeter)}
        hasAssets={hasAnyAssets}
        compact
      />
    </section>
  );
}

function CapManagementPanel({ team, capImpact, evaluation, teamId, label }) {
  if (!team) return null;
  const capAfter = capImpact?.after_usable;
  const capDelta = capImpact?.delta;
  const roster = evaluation?.roster_impact?.[teamId];
  const capDetail = team.capDetail || {};
  const retUsed = capDetail.retained_slots_used ?? capDetail.retainedSlotsUsed;
  const retMax = capDetail.retained_slots_max ?? capDetail.retainedSlotsMax ?? 3;

  const rows = [
    { label: "Cap Space", value: formatMoneyM(team.capSpace), tone: team.capSpace >= 0 ? "good" : "bad" },
    { label: "After Trade", value: capAfter != null ? formatMoneyM(capAfter) : "—", tone: capAfter == null ? "" : capAfter >= 0 ? "good" : "bad" },
    { label: "Delta", value: capDelta != null ? `${capDelta >= 0 ? "+" : ""}${formatMoneyShort(capDelta)}` : "—", tone: capDelta == null ? "" : capDelta <= 0 ? "good" : "bad" },
    { label: "Roster", value: roster ? `${roster.after}` : `${team.rosterCount}`, tone: roster?.after > 23 ? "bad" : "good" },
  ];
  if (capDetail.projected_deadline_space != null) {
    rows.push({
      label: "Deadline Room",
      value: formatMoneyM(capDetail.projected_deadline_space),
      tone: Number(capDetail.projected_deadline_space) >= 0 ? "good" : "bad",
    });
  }
  if (capDetail.is_using_ltir || Number(capDetail.ltir_pool) > 0) {
    rows.push({
      label: "LTIR Pool",
      value: formatMoneyM(capDetail.ltir_pool || 0),
      tone: "warn",
    });
  }
  if (retUsed != null) {
    rows.push({
      label: "Retained Slots",
      value: `${retUsed}/${retMax}`,
      tone: Number(retUsed) >= Number(retMax) ? "bad" : "good",
    });
  }

  return (
    <div className="trade-cap-mgmt-panel">
      <div className="trade-hub-panel-title">{label} CAP</div>
      <div className="trade-cap-mgmt-grid trade-cap-mgmt-grid-compact">
        {rows.map((row) => (
          <div key={row.label} className={`trade-cap-mgmt-row ${row.tone}`}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function TradeLegacyRevisitedCards({ franchiseState }) {
  const history = safeArray(franchiseState?.team?.fan_profile?.trade_reaction_history);
  const cards = history
    .filter((e) => e && (e.review_notes?.length || e.current_verdict !== "Too Early"))
    .slice(-4)
    .reverse();
  if (!cards.length) return null;
  return (
    <>
      <div className="trade-war-subtitle">TRADE REVISITED</div>
      {cards.map((entry, i) => {
        const verdict = entry.current_verdict || entry.verdict || "Too Early";
        const delta = Number(entry.legacy_score_delta ?? (
          Number(entry.current_fan_reaction ?? 0) - Number(entry.initial_fan_reaction ?? 0)
        ));
        const note = safeArray(entry.review_notes).slice(-1)[0] || entry.incoming_assets_summary || "";
        const deltaText = Number.isFinite(delta) && delta !== 0 ? `Fan reaction ${delta > 0 ? "+" : ""}${delta}` : "";
        return (
          <div key={entry.trade_id || i} className="trade-legacy-card">
            <div className="trade-legacy-verdict">{verdict}</div>
            {note ? <div className="trade-war-line muted">{note}</div> : null}
            {deltaText ? <div className="trade-war-line trade-legacy-delta">{deltaText}</div> : null}
          </div>
        );
      })}
    </>
  );
}

function WarRoomPanel({
  market,
  tradeHistory,
  partnerTeam,
  meta,
  partnerId,
  userTeam,
  userOutgoing,
  evaluation,
  franchiseState,
}) {
  const recent = safeArray(tradeHistory).length
    ? safeArray(tradeHistory).slice(-6).reverse()
    : safeArray(market?.recent_trades).slice(0, 4);
  const picks = safeArray(meta?.picks?.[partnerId]).slice(0, 6);
  const prospects = safeArray(meta?.prospects?.[partnerId]).slice(0, 3);
  const fanReaction = resolveFanReaction({
    userTeam,
    userOutgoing,
    evaluation,
    franchiseState,
    hasProposed: Boolean(evaluation),
  });
  const fanScore = fanReaction.score;

  return (
    <div className="trade-war-room">
      <div className="trade-war-room-col">
        <div className="trade-hub-panel-title">RECENT TRADES</div>
        {recent.length ? recent.map((t, i) => (
          <div key={t.trade_id || i} className="trade-war-line">
            {t.headline || t.summary || "Trade completed"}
            {t.season_year ? <span className="trade-war-meta"> · {t.season_year}</span> : null}
          </div>
        )) : <div className="trade-war-line muted">No recent trades</div>}
        <div className="trade-war-subtitle">PARTNER PICKS</div>
        {picks.length ? picks.map((p) => (
          <div key={p.id} className="trade-war-line">
            {p.year} {roundLabel(p.round)}
            {" · "}{p.originalTeamAbbr || inferLogoAbbr(p.original_team_id, p.original_team_id)}
            {pickRangeDisplay(p) ? ` · ${pickRangeDisplay(p)}` : ""}
          </div>
        )) : <div className="trade-war-line muted">No picks loaded</div>}
      </div>
      <div className="trade-war-room-col">
        <div className="trade-hub-panel-title">FAN PULSE</div>
        <div className={`trade-war-meter ${fanScore < 40 ? "low" : fanScore >= 70 ? "high" : ""}`}>
          <span>Reaction</span>
          <div className="trade-war-meter-bar">
            <div className={fanScore < 40 ? "low" : ""} style={{ width: `${fanScore}%` }} />
          </div>
          <strong>{fanScore}%</strong>
        </div>
        <FanReasonChips factors={fanReaction.factors} />
        <TradeLegacyRevisitedCards franchiseState={franchiseState} />
        <p className="trade-war-line muted trade-fan-hint">
          {fanReaction.summary || (fanScore < 40
            ? "Fans may push back — trading popular or young core pieces."
            : fanScore >= 70
              ? "Marketable move — fan base likely supportive."
              : "Neutral — no major fan backlash expected.")}
        </p>
        {prospects.length > 0 && (
          <>
            <div className="trade-war-subtitle">TOP YOUTH (≤21)</div>
            {prospects.map((p) => (
              <div key={p.id} className="trade-war-line">
                {p.name} · {displayOvr(p)} · {assetValueLabel(p)}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function AssetContextMenu({
  asset,
  side,
  teamId,
  isUserSide,
  inPackage,
  onAdd,
  onCompare,
  onDetails,
  onRemove,
  onRetain,
  onAskPrice,
  onAskNtcWaive,
  waiveBusy,
  onClose,
}) {
  if (!asset) return null;
  const isPick = asset.type === "pick";
  const canAskWaive =
    !isPick &&
    Boolean(onAskNtcWaive) &&
    !asset.ntcWaived &&
    (asset.requiresNtcWaive ||
      String(asset.protection || asset.clauseLabel || "").toUpperCase().includes("NTC"));

  return (
    <div className="trade-ctx-overlay" onClick={onClose}>
      <div className="trade-ctx-menu" onClick={(e) => e.stopPropagation()}>
        {isPick ? (
          <div className="trade-ctx-head pick">
            <PickIcon round={asset.round} year={asset.year} />
            <div>
              <strong>{asset.year} {roundLabel(asset.round)}</strong>
              <span>{inferLogoAbbr(asset.original_team_id, asset.original_team_id)}</span>
            </div>
          </div>
        ) : (
          <div className="trade-ctx-head">
            <PlayerHeadshot
              player={ensurePlayerHeadshotFields({ ...asset, position: asset.pos })}
              size="md"
              flag={resolveFlagIso2(asset)}
            />
            <div>
              <strong className="trade-ctx-name-row">
                <TradeFlagBadge player={asset} size="sm" />
                <span>{asset.name}</span>
              </strong>
              <span>{asset.pos} · {displayOvr(asset)} OVR · {asset.age}Y</span>
            </div>
          </div>
        )}
        <div className="trade-ctx-stats">
          {isPick ? (
            <>
              <div className="trade-ctx-pick-meter">
                <PickValueMeter item={asset} />
              </div>
              <div><span>ORIGIN</span><strong>{asset.originalTeamAbbr || inferLogoAbbr(asset.original_team_id, asset.original_team_id)}</strong></div>
              <div><span>RANGE</span><strong>{pickRangeDisplay(asset) || "—"}</strong></div>
              <div><span>PROT</span><strong>{asset.protection || "—"}</strong></div>
            </>
          ) : (
            <>
              <div><span>CAP</span><strong>{formatPlayerCapLabel(asset)}</strong></div>
              <div><span>YEARS</span><strong>{asset.years > 0 ? asset.years : "—"}</strong></div>
              <div><span>VALUE</span><strong>{assetValueLabel(asset)}</strong></div>
              <div><span>ROLE</span><strong>{asset.role || roleFromOverall(asset.ovr, asset.pos)}</strong></div>
              <div><span>POT</span><strong>{asset.potentialGrade || "—"}</strong></div>
              <div><span>CLAUSE</span><strong>{asset.clauseLabel || (asset.protection && asset.protection !== "None" ? asset.protection : "—")}</strong></div>
            </>
          )}
        </div>
        {asset.ntcWaived ? (
          <div className="trade-ctx-block trade-ctx-waive-ok">
            NTC waived{asset.ntcWaiverReason ? ` — ${asset.ntcWaiverReason}` : ""}. Value slightly reduced.
          </div>
        ) : asset.tradeable === false ? (
          <div className="trade-ctx-block">{asset.tradeBlockReason || "Not tradeable"}</div>
        ) : null}
        {safeArray(asset.tradeRiskFlags).slice(0, 3).map((f, i) => (
          <span key={i} className="trade-ctx-risk-tag">{f}</span>
        ))}
        <div className="trade-ctx-actions">
          {!inPackage && asset.tradeable !== false && (
            <button type="button" onClick={onAdd}>Add</button>
          )}
          {canAskWaive && (
            <button type="button" className="primary" disabled={waiveBusy} onClick={onAskNtcWaive}>
              {waiveBusy ? "Asking…" : "Ask to Waive NTC"}
            </button>
          )}
          <button type="button" onClick={onCompare}>Compare</button>
          <button type="button" onClick={onDetails}>Details</button>
          {inPackage && <button type="button" className="danger" onClick={onRemove}>Remove</button>}
          {isUserSide && !isPick && inPackage && onRetain && (
            <button type="button" onClick={onRetain}>Retain Salary</button>
          )}
          {!isUserSide && onAskPrice && (
            <button type="button" onClick={onAskPrice}>Ask Price</button>
          )}
          <button type="button" className="ghost" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

function AssetDetailDrawer({
  asset,
  tab,
  onTabChange,
  onClose,
  evaluation,
  userTeamId,
  partnerTeamId,
  partnerTeam,
  userTeam,
  userOutgoing,
  franchiseState,
  tradeMarket,
  tradeHistory,
}) {
  if (!asset) return null;
  const isPick = asset.type === "pick";
  const valueItem = { tradeValue: asset.tradeValue ?? asset.value_hint, valueTier: asset.valueTier };
  const qualTags = qualitativeBreakdownTags(asset.tradeBreakdown);
  const explainLines = sanitizeTradeExplain(asset.tradeExplain);
  const fanScore = resolveFanReaction({ userTeam, userOutgoing, evaluation, franchiseState, hasProposed: Boolean(evaluation) });
  const partnerImpact = evaluation?.team_needs_impact?.[partnerTeamId];
  const userSlots = evaluation?.contract_slot_impact?.[userTeamId];
  const roster = evaluation?.roster_impact?.[userTeamId];
  const interest = evaluation?.interest_level?.[partnerTeamId];
  const recent = safeArray(tradeHistory).length
    ? safeArray(tradeHistory).slice(-4).reverse()
    : safeArray(tradeMarket?.recent_trades).slice(0, 4);
  const headPlayer = !isPick
    ? ensurePlayerHeadshotFields({ ...asset, position: asset.pos })
    : null;
  const handLabel = asset.handed ? String(asset.handed).charAt(0).toUpperCase() : null;
  const valueTierClass = assetValueTierClass(valueItem);

  const tabs = ["Value", "Contract", "Fit", "Fan", "Market"];

  return (
    <div className="trade-drawer-overlay trade-drawer-overlay-asset" onClick={onClose}>
      <div className="trade-drawer trade-drawer-asset" onClick={(e) => e.stopPropagation()}>
        <div className="trade-asset-detail-hero">
          <button type="button" className="trade-drawer-close trade-asset-detail-close" onClick={onClose}>×</button>
          <div className="trade-asset-hero-visual">
            {isPick ? (
              <div className="trade-asset-hero-pick">
                <PickIcon round={asset.round} year={asset.year} />
              </div>
            ) : (
              <PS1PlayerPortrait
                player={headPlayer}
                size="xl"
                className="trade-asset-hero-ps1-portrait"
                teamColors={getTeamPortraitColors(headPlayer, partnerTeam || asset)}
              />
            )}
          </div>
          <div className="trade-asset-hero-main">
            <div className="trade-asset-hero-identity">
              {!isPick && <TradeFlagBadge player={asset} size="lg" />}
              <strong>{isPick ? `${asset.year} ${roundLabel(asset.round)}` : asset.name}</strong>
              <span>
                {isPick
                  ? `ORIG ${inferLogoAbbr(asset.original_team_id, asset.original_team_id)}`
                  : `${asset.pos} · ${asset.age}Y${handLabel ? ` · ${handLabel}` : ""}`}
              </span>
            </div>
            <div className="trade-asset-hero-tiles">
              {!isPick && (
                <div className="trade-asset-hero-tile tile-ovr">
                  <span>OVR</span>
                  <strong>{displayOvr(asset)}</strong>
                </div>
              )}
              <div className="trade-asset-hero-tile tile-cap">
                <span>{isPick ? "PICK" : "CAP HIT"}</span>
                <strong>
                  {isPick
                    ? `${asset.year} ${roundSuffix(asset.round) || roundLabel(asset.round)}`
                    : formatPlayerCapLabel(asset)}
                </strong>
              </div>
              <div className={`trade-asset-hero-tile tile-value ${valueTierClass}`}>
                <span>VALUE</span>
                {isPick ? (
                  <PickValueMeter item={asset} showLabel={false} className="trade-asset-hero-pick-meter" />
                ) : (
                  <>
                    <strong>{assetValueLabel(valueItem)}</strong>
                    <div className="trade-value-chip-track hero">
                      <div className="trade-value-chip-fill" style={{ width: `${assetValuePct(valueItem)}%` }} />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="trade-drawer-tabs">
          {tabs.map((t) => (
            <button key={t} type="button" className={tab === t ? "active" : ""} onClick={() => onTabChange(t)}>{t}</button>
          ))}
        </div>
        <div className="trade-drawer-body">
          {tab === "Value" && (
            <>
              <div className="trade-asset-value-panel">
                {isPick ? (
                  <>
                    <PickValueMeter item={asset} className="trade-asset-value-panel-meter" />
                    <div className="trade-drawer-kv">
                      <span>Original Team</span>
                      <strong>{asset.originalTeamAbbr || inferLogoAbbr(asset.original_team_id, asset.original_team_id)}</strong>
                      <span>Projected Range</span>
                      <strong>{pickRangeDisplay(asset) || "—"}</strong>
                      <span>Protection</span>
                      <strong>{asset.protection || "—"}</strong>
                      <span>Value Outlook</span>
                      <strong>{pickOutlookLabel(asset)}</strong>
                      <span>Why</span>
                      <strong>{asset.pickValueContext || "—"}</strong>
                    </div>
                  </>
                ) : (
                  <>
                    <TradeValueChip item={valueItem} className="trade-asset-value-panel-chip" />
                    <div className="trade-drawer-kv">
                      <span>Value Tier</span><strong>{assetValueLabel(valueItem)}</strong>
                      <span>Role</span><strong>{asset.role || roleFromOverall(asset.ovr, asset.pos)}</strong>
                      <span>Contract Fit</span><strong>{asset.contractType || "—"}</strong>
                      <span>Need Fit</span><strong>{partnerImpact?.strengthens?.length ? "Strong" : partnerImpact?.weakens?.length ? "Weak" : "—"}</strong>
                    </div>
                    {qualTags.length > 0 && (
                      <div className="trade-hub-chip-row">
                        {qualTags.map((tag) => (
                          <span key={tag} className="trade-hub-chip">{tag}</span>
                        ))}
                      </div>
                    )}
                    {safeArray(asset.tradeRiskFlags).length > 0 && (
                      <>
                        <div className="trade-hub-panel-title">Risks</div>
                        {safeArray(asset.tradeRiskFlags).map((f, i) => (
                          <p key={i} className="trade-drawer-line warn">{f}</p>
                        ))}
                      </>
                    )}
                    {explainLines.length > 0 && (
                      <>
                        <div className="trade-hub-panel-title">Scouting Notes</div>
                        {explainLines.map((line, i) => (
                          <p key={i} className="trade-drawer-line">{line}</p>
                        ))}
                      </>
                    )}
                    {!qualTags.length && !explainLines.length && (
                      <p className="trade-drawer-muted">Qualitative value profile — exact scores hidden.</p>
                    )}
                  </>
                )}
              </div>
            </>
          )}
          {tab === "Contract" && (
            <>
              {isPick ? (
                <div className="trade-drawer-kv">
                  <span>Year</span><strong>{asset.year || "—"}</strong>
                  <span>Round</span><strong>{asset.round ? roundLabel(asset.round) : "—"}</strong>
                  <span>Protection</span><strong>{asset.protection || "—"}</strong>
                  <span>Condition</span><strong>{asset.condition || "—"}</strong>
                </div>
              ) : (
                <>
                  <div className="trade-drawer-kv">
                    <span>Cap Hit</span><strong>{formatPlayerCapLabel(asset)}</strong>
                    <span>Years</span><strong>{asset.years > 0 ? asset.years : "—"}</strong>
                    <span>Type</span><strong>{asset.contractType || "—"}</strong>
                    <span>Clause</span><strong>{asset.clauseLabel || asset.protection || "—"}</strong>
                  </div>
                  {asset.tradeBlockReason && <p className="trade-drawer-warn">{asset.tradeBlockReason}</p>}
                  {safeArray(asset.approvedTradeTeams).length > 0 && (
                    <p className="trade-drawer-line">Approved: {asset.approvedTradeTeams.join(", ")}</p>
                  )}
                  {asset.retained_pct > 0 && <p className="trade-drawer-line">Retained: {asset.retained_pct}%</p>}
                </>
              )}
            </>
          )}
          {tab === "Fit" && (
            <>
              <div className="trade-drawer-kv">
                <span>Role</span><strong>{asset.role || roleFromOverall(asset.ovr, asset.pos)}</strong>
                <span>Partner Interest</span><strong>{interest != null ? `${Math.round(interest * 100)}%` : "—"}</strong>
              </div>
              {partnerImpact?.strengthens?.length > 0 && (
                <p className="trade-drawer-line ok">Helps: {partnerImpact.strengthens.join(", ")}</p>
              )}
              {partnerImpact?.weakens?.length > 0 && (
                <p className="trade-drawer-line warn">Hurts: {partnerImpact.weakens.join(", ")}</p>
              )}
              {roster && <p className="trade-drawer-line">Roster after: {roster.after ?? "—"}</p>}
              {userSlots && (
                <p className="trade-drawer-line">Slots: {userSlots.before} → {userSlots.after}/{userSlots.limit || 50}</p>
              )}
              {partnerTeam?.needsSummary?.needs_short?.length > 0 && (
                <p className="trade-drawer-line">{partnerTeam.abbr} needs: {partnerTeam.needsSummary.needs_short.join(", ")}</p>
              )}
            </>
          )}
          {tab === "Fan" && (
            <>
              <FanReactionBadge score={fanScore.score} compact={false} />
              <p className="trade-drawer-line">{fanScore.category || fanReactionCategory(fanScore.score)}</p>
              <FanReasonChips factors={fanScore.factors} />
              {fanScore.score < 40 && <p className="trade-drawer-warn">Fan Risk: {fanRiskReason(userOutgoing)}</p>}
            </>
          )}
          {tab === "Market" && (
            <>
              <p className="trade-drawer-line">Market: {tradeMarket?.market_temperature || "—"}</p>
              <p className="trade-drawer-line">{partnerTeam?.abbr || "Partner"} · {partnerTeam?.direction || "—"}</p>
              {recent.length ? recent.map((t, i) => (
                <p key={i} className="trade-drawer-line">{t.headline || t.summary || "Trade"}</p>
              )) : <p className="trade-drawer-muted">No market comps</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function mapReviewResultTone(tone) {
  const t = String(tone || "").toLowerCase();
  if (t === "accepted") return "good";
  if (t === "close") return "warn";
  if (t === "blocked" || t === "rejected") return "bad";
  return "warn";
}

function humanizeFanFactor(raw) {
  const s = String(raw || "").trim();
  const low = s.toLowerCase();
  if (!s) return "";
  if (low.includes("star") && (low.includes(" in") || low.endsWith("in"))) return "Franchise star moved";
  if (low.includes("star player") || low === "star") return "Star player moved";
  if (low.includes("young core")) return "Young core piece moved";
  if (low.includes("first-round") || low.includes("1st")) return "First-round pick moved";
  if (low.includes("overpay")) return "Fans see an overpay";
  if (low.includes("fair")) return "Fans see fair value";
  if (low.includes("captain")) return "Captain traded";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function fanBacklashScaleLabel(heat) {
  const h = Number(heat) || 0;
  if (h >= 66) return "HIGH";
  if (h >= 33) return "MEDIUM";
  return "LOW";
}

function gmReadDisplayLabel(gmRead, evaluation, partnerTeamId) {
  const raw = String(gmRead?.label || gmReviewShort(evaluation, partnerTeamId) || "").toUpperCase();
  if (raw === "HIGH") return "Eager — close to yes";
  if (raw === "MED") return "Open, but wants premium";
  if (raw === "LOW") return "Cold — major change needed";
  return "Interest unclear";
}

function gmReadExplainer(label) {
  return gmReadDisplayLabel({ label }, null, null);
}

function playerLastName(name) {
  const parts = String(name || "").trim().split(/\s+/);
  return parts.length ? parts[parts.length - 1] : name;
}

function isFranchisePlayer(player) {
  if (!player) return false;
  const ovr = Number(player.ovr) || 0;
  const tier = String(assetValueLabel(player) || "").toUpperCase();
  return ovr >= 88 || tier.includes("FRANCHISE");
}

function untouchableReasonIcon(reason) {
  const r = String(reason || "").toLowerCase();
  if (r.includes("no-movement") || r.includes("nmc")) return "NMC";
  if (r.includes("no-trade") || r.includes("ntc")) return "NTC";
  if (r.includes("young core")) return "CORE";
  if (r.includes("captain")) return "C";
  if (r.includes("franchise") || r.includes("elite") || r.includes("cornerstone")) return "STAR";
  return "LOCK";
}

function resolvePickOriginTeam(asset, meta) {
  const teamLookup = Object.fromEntries(safeArray(meta?.teams).map((t) => [String(t.id), t]));
  const teamLookupByAbbr = Object.fromEntries(
    safeArray(meta?.teams).map((t) => [String(t.abbr || "").toUpperCase(), t]),
  );
  const originalTeamId = String(asset?.original_team_id || "");
  const byId = teamLookup[originalTeamId] || null;
  const abbr = asset.originalTeamAbbr
    || normalizeNhlAbbr(byId?.abbr)
    || normalizeNhlAbbr(originalTeamId)
    || inferLogoAbbr(originalTeamId, originalTeamId);
  const team = byId
    || teamLookupByAbbr[String(abbr).toUpperCase()]
    || { abbr: abbr !== "?" && abbr !== "—" ? abbr : null, name: abbr };
  return { team, abbr: team?.abbr || (abbr !== "?" && abbr !== "—" ? abbr : null) };
}

function reviewValueLabelForBlock(netGap, balance, isRejected, mainProblem) {
  const key = mainProblem?.primaryKey;
  if (isRejected && key === "CAP") {
    const close = Math.abs(Number(netGap) || 0) <= 4
      || ["CLOSE", "EVEN"].includes(String(balance?.label || "").toUpperCase());
    return close ? "Fair value · cap blocks" : "Cap blocks first";
  }
  return valueGapQualitativeLabel(netGap, balance?.label, isRejected, key);
}

function pickReviewShortLabel(pick) {
  const range = pickRangeDisplay(pick);
  if (range === "LOTTERY") return "Lottery chance";
  if (range === "TOP 10") return "Top-10 slot";
  if (range === "LATE") return "Late 1st";
  if (range === "MID") return "Mid-round";
  if (range === "CONTENDER") return "Contender slot";
  const round = Number(pick?.round) || 0;
  if (round === 1) return "1st-round capital";
  if (round === 2) return "2nd-round pick";
  const tier = String(assetValueLabel(pick) || "").trim();
  return tier && tier !== "UNKNOWN" ? tier : "Draft pick";
}

function resolveCapBlockContext({
  evaluation,
  userTeamId,
  partnerTeamId,
  partnerTeam,
  userTeam,
  userOutgoing,
  partnerOutgoing,
  capBefore,
  capAfter,
}) {
  const userCap = evaluation?.cap_impact?.[userTeamId];
  const partnerCap = evaluation?.cap_impact?.[partnerTeamId];
  const userAfter = Number(userCap?.after_usable ?? userCap?.projectedCapSpace ?? capAfter);
  const partnerAfter = Number(partnerCap?.after_usable ?? partnerCap?.projectedCapSpace);
  let failingTeam = "user";
  let failingAbbr = userTeam?.abbr || "YOU";
  let after = userAfter;

  if (Number.isFinite(partnerAfter) && partnerAfter < 0) {
    failingTeam = "partner";
    failingAbbr = partnerTeam?.abbr || "THEM";
    after = partnerAfter;
  } else if (Number.isFinite(userAfter) && userAfter < 0) {
    failingTeam = "user";
    failingAbbr = userTeam?.abbr || "YOU";
    after = userAfter;
  }

  const addCapPool = failingTeam === "user" ? partnerOutgoing : userOutgoing;
  const culprit = safeArray(addCapPool)
    .filter((a) => a?.type === "player")
    .sort((a, b) => playerPackageCapHit(b) - playerPackageCapHit(a))[0] || null;
  const before = failingTeam === "user"
    ? Number(userCap?.before_usable ?? capBefore)
    : Number(partnerCap?.before_usable ?? partnerTeam?.capSpace);
  const overBy = Number.isFinite(after) && after < 0 ? Math.abs(after) : 0;

  return {
    failingTeam,
    failingAbbr,
    culprit,
    before: Number.isFinite(before) ? before : capBefore,
    after: Number.isFinite(after) ? after : capAfter,
    overBy,
  };
}

function gmReadDetail(mainProblem, gmRead, evaluation, partnerTeamId) {
  const key = mainProblem?.primaryKey;
  if (key === "CAP") return "Shed salary before they re-engage.";
  if (key === "PROTECTED") return "Will not move protected core.";
  if (key === "VALUE") {
    const target = mainProblem?.blockedAsset;
    return target ? `Wants premium for ${target}.` : "Needs more value in return.";
  }
  if (key === "FIT") return "Wants a different player profile.";
  const label = String(gmRead?.label || gmReviewShort(evaluation, partnerTeamId) || "").toUpperCase();
  if (label === "LOW") return "Needs a major package change.";
  if (label === "MED") return "Add a pick or top-six piece.";
  if (label === "HIGH") return "Close — minor tweak may work.";
  return "";
}

function resolveFanTradeSubject(userOutgoing, fanReasons) {
  const players = safeArray(userOutgoing).filter((a) => a?.type === "player");
  const star = players.find((p) => Number(p.ovr) >= 88)
    || players.find((p) => Number(p.ovr) >= 82);
  if (star?.name) return `Fans erupt if ${playerLastName(star.name)} moves`;
  const first = safeArray(userOutgoing).find((a) => a?.type === "pick" && Number(a.round) === 1);
  if (first) return "Fans hate moving a 1st";
  return fanReasons[0] || "";
}

function valueGapQualitativeLabel(netGap, balanceLabel, isRejected, primaryKey = "") {
  const bal = String(balanceLabel || "").toUpperCase();
  const gap = Number(netGap) || 0;
  const key = String(primaryKey || "").toUpperCase();
  if (bal === "STEAL" || bal === "THEY PAY") return "You win value";
  if (bal === "YOU PAY" || bal === "HEAVY") return "Overpay risk";
  if (gap <= -12 || bal === "HEAVY") return "Premium required";
  if (gap <= -4 || bal === "YOU PAY") return "Short";
  if (bal === "CLOSE" || bal === "EVEN" || Math.abs(gap) <= 3) {
    if (!isRejected) return "Close";
    if (key === "CAP") return "Close — cap blocks";
    if (key === "VALUE") return "Close — needs sweetener";
    if (key === "PROTECTED") return "Close — protected";
    if (key === "FIT") return "Close — bad fit";
    return "Close — blocked";
  }
  if (gap >= 4) return "Ahead on value";
  return "Close";
}

function valueGapMarkerPct(netGap, balanceScore) {
  const score = Number(balanceScore);
  if (Number.isFinite(score)) return clamp(Math.round(score), 8, 92);
  const gap = Number(netGap) || 0;
  return clamp(50 - gap * 2.5, 8, 92);
}

function resolveMainProblem({
  evaluation,
  why,
  noTouchConflict,
  partnerOutgoing,
  userOutgoing,
  partnerTeam,
  userTeam,
  userTeamId,
  partnerTeamId,
  capBefore,
  capAfter,
  netGap,
  balance,
}) {
  if (noTouchConflict?.length) {
    const name = noTouchConflict[0];
    return {
      sentence: `${playerLastName(name)} is protected. Choose another target.`,
      type: "hard",
      negotiable: false,
      primaryKey: "PROTECTED",
      blockedAsset: name,
      subline: "Value will not fix this.",
    };
  }

  const reasons = getEvaluationReasons(evaluation);
  const blob = `${reasons.map((r) => `${r.tag || ""} ${r.text || ""}`).join(" ")} ${why?.summary || ""}`.toLowerCase();
  const incoming = safeArray(partnerOutgoing).filter((a) => a?.type === "player");
  const targetName = incoming[0]?.name ? playerLastName(incoming[0].name) : null;

  if (blob.includes("clause") || blob.includes("ntc") || blob.includes("nmc")) {
    const blocked = incoming.find((a) => {
      const c = String(a.protection || a.clauseLabel || "").toUpperCase();
      return c.includes("NMC") || c.includes("NTC");
    });
    const who = blocked?.name ? playerLastName(blocked.name) : targetName;
    return {
      sentence: who ? `${who} has a clause. Remove or rework.` : "Clause blocks this deal.",
      type: "hard",
      negotiable: false,
      primaryKey: "CLAUSE",
      blockedAsset: blocked?.name || null,
      subline: "",
    };
  }

  if (blob.includes("cap") || blob.includes("salary")) {
    const capCtx = resolveCapBlockContext({
      evaluation,
      userTeamId,
      partnerTeamId,
      partnerTeam,
      userTeam,
      userOutgoing,
      partnerOutgoing,
      capBefore,
      capAfter,
    });
    const culprit = capCtx.culprit?.name ? playerLastName(capCtx.culprit.name) : null;
    return {
      sentence: `${capCtx.failingAbbr} cannot absorb this cap.`,
      type: "hard",
      negotiable: false,
      primaryKey: "CAP",
      blockedAsset: capCtx.culprit?.name || null,
      subline: culprit
        ? `Retain salary or drop ${culprit} before value matters.`
        : "Retain salary or remove cap before value matters.",
    };
  }

  if (blob.includes("pick") && (blob.includes("own") || blob.includes("registry"))) {
    return {
      sentence: "Pick ownership issue.",
      type: "hard",
      negotiable: false,
      primaryKey: "PICK",
      blockedAsset: null,
    };
  }

  if (blob.includes("roster") || blob.includes("slot")) {
    return {
      sentence: "Roster slots do not fit.",
      type: "hard",
      negotiable: false,
      primaryKey: "ROSTER",
      blockedAsset: null,
    };
  }

  if (evaluation?.accepted && evaluation?.can_execute) {
    return {
      sentence: targetName ? `${targetName} deal clears.` : "Deal can execute.",
      type: "ok",
      negotiable: false,
      primaryKey: "ACCEPTED",
      blockedAsset: null,
    };
  }

  const gap = Number(netGap) || 0;
  if (!evaluation?.accepted || gap < -3) {
    return {
      sentence: targetName
        ? `Need more value for ${targetName}.`
        : "Package is short on value.",
      type: "soft",
      negotiable: true,
      primaryKey: "VALUE",
      blockedAsset: targetName,
    };
  }

  if (blob.includes("need") || blob.includes("fit")) {
    return {
      sentence: targetName ? `${targetName} is not what they want.` : "Roster fit is off.",
      type: "soft",
      negotiable: true,
      primaryKey: "FIT",
      blockedAsset: targetName,
    };
  }

  const valLabel = valueGapQualitativeLabel(gap, balance?.label, true);
  return {
    sentence: why?.summary || (valLabel.includes("Close") ? "Close, but not enough." : "Adjust the package."),
    type: "soft",
    negotiable: true,
    primaryKey: "VALUE",
    blockedAsset: targetName,
  };
}

function resolveReviewBlockers(evaluation, why, fanHeat = 0, noTouchConflict = []) {
  if (noTouchConflict?.length) {
    return { primary: "PROTECTED", pills: [{ key: "PROTECTED", label: "Protected", active: true }] };
  }

  const reasons = getEvaluationReasons(evaluation);
  const blob = `${reasons.map((r) => `${r.tag || ""} ${r.text || ""}`).join(" ")} ${why?.summary || ""}`.toLowerCase();
  const primaryCode = String(why?.primary_code || "").toUpperCase();
  const heat = Number(fanHeat) || 0;
  const hardCap = blob.includes("cap") || blob.includes("salary") || primaryCode === "CAP";
  const hardClause = blob.includes("clause") || blob.includes("ntc") || blob.includes("nmc") || primaryCode === "CLAUSE";

  const hierarchy = [
    { key: "CAP", label: "Cap", hit: hardCap },
    { key: "CLAUSE", label: "Clause", hit: hardClause },
    { key: "PICK", label: "Pick", hit: blob.includes("pick") && (blob.includes("own") || blob.includes("registry")) },
    { key: "ROSTER", label: "Roster", hit: blob.includes("roster") || blob.includes("slot") || primaryCode === "ROSTER" },
    {
      key: "VALUE",
      label: "Value",
      hit: !hardCap && !hardClause && (
        blob.includes("value") || blob.includes("return") || blob.includes("reject")
        || primaryCode === "VALUE" || evaluation?.accepted === false
      ),
    },
    { key: "FIT", label: "Fit", hit: blob.includes("need") || blob.includes("fit") || primaryCode === "FIT" },
    { key: "FANS", label: "Fans", hit: heat >= 55 && !hardCap && !blob.includes("value") },
  ];

  const primaryHit = hierarchy.find((d) => d.hit);
  const primary = primaryHit?.key || primaryCode || "VALUE";
  const pills = [{
    key: primary,
    label: primaryHit?.label || (primary === "VALUE" ? "Value" : primary),
    active: true,
  }];
  return { primary, pills };
}

function untouchableReasonForPlayer(player) {
  if (!player) return "Protected by partner.";
  if (player.tradeable === false) return String(player.tradeBlockReason || "Unavailable.");
  const clause = String(player.protection || player.clauseLabel || "").toUpperCase();
  if (clause.includes("NMC")) return "No-movement clause.";
  if (clause.includes("NTC")) return "No-trade clause.";
  const ovr = Number(player.ovr) || 0;
  if (ovr >= 88) return "Franchise cornerstone.";
  if (Number(player.age) <= 24 && ovr >= 82) return "Young core.";
  const tier = String(assetValueLabel(player) || "").toUpperCase();
  if (tier.includes("FRANCHISE") || tier.includes("ELITE")) return "Elite untouchable.";
  return "Core piece protected.";
}

function untouchableProtectionInfo(player, reasonText) {
  const r = String(reasonText || "").toLowerCase();
  if (player?.captain || player?.isCaptain || r.includes("captain")) {
    return { tag: "CAPTAIN", note: reasonText || "Protected by partner." };
  }
  if (r.includes("no-movement") || r.includes("nmc") || r.includes("no-trade") || r.includes("ntc")) {
    return { tag: "UNTOUCHABLE", note: reasonText || "Protected by partner." };
  }
  if (r.includes("franchise") || r.includes("cornerstone")) {
    return { tag: "FRANCHISE", note: reasonText || "Protected by partner." };
  }
  if (r.includes("young core") || r.includes("prospect")) {
    return { tag: "CORE PROSPECT", note: reasonText || "Protected by partner." };
  }
  if (r.includes("recent") || r.includes("signed")) {
    return { tag: "RECENT SIGNING", note: reasonText || "Protected by partner." };
  }
  if (r.includes("playoff") || r.includes("rental")) {
    return { tag: "PLAYOFF PIECE", note: reasonText || "Protected by partner." };
  }
  const pos = String(player?.pos || "").toUpperCase();
  const ovr = Number(player?.ovr) || 0;
  if (pos === "D" && ovr >= 84) return { tag: "TOP PAIR", note: reasonText || "Protected by partner." };
  if (ovr >= 86) return { tag: "CORE", note: reasonText || "Protected by partner." };
  return { tag: "CORE", note: reasonText || "Protected by partner." };
}

function buildUntouchableRows(names, playerLookup) {
  return dedupeReviewLines(names).map((name) => {
    const player = lookupReviewPlayer(name, playerLookup);
    const reason = untouchableReasonForPlayer(player);
    const info = untouchableProtectionInfo(player, reason);
    return { name, reason: info.note, tag: info.tag };
  });
}

function partnerShortName(team) {
  const name = String(team?.name || "").trim();
  if (name) return name.split(/\s+/)[0];
  return team?.abbr || "Partner";
}

function reviewBlockerChipLabel(primaryKey) {
  const k = String(primaryKey || "").toUpperCase();
  if (k === "PROTECTED" || k === "CLAUSE") return "UNTOUCHABLE";
  if (k === "FANS") return "FAN HEAT";
  if (["VALUE", "CAP", "FIT", "PICK", "ROSTER"].includes(k)) return k === "ROSTER" ? "FIT" : k;
  return "VALUE";
}

function reviewHeadlineCause(partnerTeam, mainProblem) {
  const abbr = partnerShortName(partnerTeam);
  const key = mainProblem?.primaryKey;
  const target = mainProblem?.blockedAsset ? playerLastName(mainProblem.blockedAsset) : null;
  if (mainProblem?.type === "ok") {
    return target ? `${abbr} accepts this deal.` : "Deal can execute.";
  }
  if (key === "PROTECTED") return `${abbr} will not trade ${target || "that player"}.`;
  if (key === "CAP") return `${abbr} cannot absorb this cap hit.`;
  if (key === "VALUE") return target ? `${abbr} needs more for ${target}.` : `${abbr} needs one stronger asset.`;
  if (key === "FIT") return `${abbr} wants a different player profile.`;
  if (key === "FANS") return "Fan backlash blocks this move.";
  if (key === "CLAUSE") return "Contract clause blocks the deal.";
  return mainProblem?.sentence || `${abbr} rejected this package.`;
}

function proposeDisabledReasonLabel(rd, externalReason) {
  const raw = String(externalReason || "").toUpperCase();
  if (raw.includes("PROTECTED") || raw.includes("UNTOUCHABLE")) return "Untouchable blocked";
  if (raw.includes("CAP")) return "Cap problem";
  if (raw.includes("CLAUSE")) return "Clause blocked";
  if (raw.includes("SWEETENER") || raw.includes("NEEDS")) return "Needs sweetener";
  if (externalReason) return externalReason;
  const key = rd.mainProblem?.primaryKey;
  if (key === "CAP") return "Cap problem";
  if (key === "PROTECTED" || key === "CLAUSE") return "Untouchable blocked";
  return "Needs sweetener";
}

function resolveDualTeamCap({
  evaluation,
  userTeamId,
  partnerTeamId,
  userTeam,
  partnerTeam,
  userOutgoing,
  partnerOutgoing,
  capBefore,
}) {
  const userCap = evaluation?.cap_impact?.[userTeamId];
  const partnerCap = evaluation?.cap_impact?.[partnerTeamId];
  let userBefore = Number(userCap?.before_usable ?? capBefore ?? userTeam?.capSpace);
  let userAfter = Number(userCap?.after_usable ?? userCap?.projectedCapSpace);
  let partnerBefore = Number(partnerCap?.before_usable ?? partnerTeam?.capSpace);
  let partnerAfter = Number(partnerCap?.after_usable ?? partnerCap?.projectedCapSpace);

  if (!Number.isFinite(userAfter) && Number.isFinite(userBefore)) {
    userAfter = userBefore + packageCapDelta(userOutgoing) - packageCapDelta(partnerOutgoing);
  }
  if (!Number.isFinite(partnerBefore) && Number.isFinite(partnerTeam?.capSpace)) {
    partnerBefore = Number(partnerTeam.capSpace);
  }
  if (!Number.isFinite(partnerAfter) && Number.isFinite(partnerBefore)) {
    partnerAfter = partnerBefore + packageCapDelta(partnerOutgoing) - packageCapDelta(userOutgoing);
  }

  const userDelta = Number.isFinite(userBefore) && Number.isFinite(userAfter) ? userAfter - userBefore : null;
  const partnerDelta = Number.isFinite(partnerBefore) && Number.isFinite(partnerAfter)
    ? partnerAfter - partnerBefore
    : null;
  const bad = (Number.isFinite(userAfter) && userAfter < 0) || (Number.isFinite(partnerAfter) && partnerAfter < 0);

  return {
    userAbbr: userTeam?.abbr || "YOU",
    partnerAbbr: partnerTeam?.abbr || "THEM",
    userBefore: Number.isFinite(userBefore) ? userBefore : null,
    userAfter: Number.isFinite(userAfter) ? userAfter : null,
    userDelta,
    partnerBefore: Number.isFinite(partnerBefore) ? partnerBefore : null,
    partnerAfter: Number.isFinite(partnerAfter) ? partnerAfter : null,
    partnerDelta,
    fits: !bad,
  };
}

function formatCapDeltaLabel(delta) {
  if (!Number.isFinite(delta) || delta === 0) return null;
  return `${delta >= 0 ? "+" : "-"}${formatMoneyShort(Math.abs(delta))}`;
}

function normalizeWantCategory(category, index) {
  const cat = String(category || "").toUpperCase();
  if (cat.includes("TOP")) return "TOP ASK";
  if (cat.includes("GOOD")) return "GOOD FIT";
  if (cat.includes("DEPTH")) return "DEPTH INTEREST";
  if (cat.includes("CAP")) return "CAP FIT";
  if (cat.includes("PICK")) return "PICK INTEREST";
  if (index === 0) return "TOP ASK";
  if (index <= 2) return "GOOD FIT";
  return "DEPTH INTEREST";
}

function buildNextStepText(rd) {
  const top = rd.fixSuggestions?.[0];
  if (top?.label && top?.text) return `${top.label.toLowerCase()} — ${top.text}`;
  if (top?.label) return top.label;
  if (rd.mainProblem?.subline) return rd.mainProblem.subline;
  return "Adjust the package and re-check.";
}

function buildWantReason(name, wants, evaluation, category, index = 0) {
  if (wants?.source === "backend") {
    if (index === 0 && wants.summary && !wants.summary.toLowerCase().includes("no clear")) {
      return shortenReviewHint(wants.summary, 40);
    }
    const chips = safeArray(wants?.chips).map((c) => String(c).toUpperCase());
    if (category === "TOP ASK" && chips.includes("DEFENSE")) return "Fills defensive need";
    if (category === "TOP ASK" && chips.includes("TOP 6")) return "Top-six scoring target";
    if (category === "TOP ASK" && chips.includes("GOALIE")) return "Goaltending upgrade";
    if (category === "GOOD FIT") return "Strong roster fit";
    if (category === "CAP FIT") return "Cap-friendly target";
    if (category === "PICK INTEREST") return "Draft capital interest";
    return "On partner shopping list";
  }
  return "";
}

function buildWantsRanked(players, summary, wants, evaluation) {
  if (wants?.source !== "backend") return [];
  return safeArray(players).map((name, i) => {
    const category = normalizeWantCategory(i === 0 ? "Top Ask" : i <= 2 ? "Good Fit" : "Depth Interest", i);
    return {
      rank: i + 1,
      name,
      category,
      note: buildWantReason(name, wants, evaluation, category, i),
    };
  });
}

function buildFixActions(evaluation, rd, noTouchConflict, partnerOutgoing, userOutgoing) {
  const actions = [];
  const mainKey = rd.mainProblem?.primaryKey || rd.blockers?.primary || "";
  const partnerName = partnerShortName(rd.partnerTeam || { abbr: "Partner" });

  if (noTouchConflict?.length) {
    noTouchConflict.forEach((name, i) => {
      actions.push({
        action: "remove",
        label: `Replace ${playerLastName(name)}`,
        hint: "Untouchable player",
        playerName: name,
        side: "incoming",
        rank: i + 1,
        fixType: "replace",
      });
    });
    actions.push({
      action: "adjust",
      label: "Add Forward",
      hint: `${partnerName} wants scoring depth`,
      rank: 9,
      fixType: "forward",
    });
    return actions.sort((a, b) => a.rank - b.rank).slice(0, 3);
  }

  if (mainKey === "CAP") {
    const capCtx = rd.capBlockContext;
    if (capCtx?.culprit?.name) {
      const side = capCtx.failingTeam === "user" ? "incoming" : "outgoing";
      actions.push({
        action: "remove",
        label: `Replace ${playerLastName(capCtx.culprit.name)}`,
        hint: "Trim cap hit first",
        playerName: capCtx.culprit.name,
        side,
        rank: 1,
        fixType: "replace",
      });
    }
    actions.push({
      action: "retain",
      label: "Add Pick",
      hint: "Retain salary to fit cap",
      rank: capCtx?.culprit ? 2 : 1,
      fixType: "pick",
    });
    actions.push({
      action: "adjust",
      label: "Replace Asset",
      hint: "Swap for lower cap hit",
      rank: 3,
      fixType: "replace",
    });
    return actions.sort((a, b) => a.rank - b.rank).slice(0, 3);
  }

  if (mainKey === "VALUE" || mainKey === "FIT") {
    const wantHint = partnerWantHintFromBackend(rd.wants, partnerName);
    safeArray(evaluation?.suggested_counteroffers).slice(0, 1).forEach((c) => {
      actions.push({
        action: "counter",
        label: "Add Pick",
        hint: shortenReviewHint(c.explanation || c.summary || "Closes value gap", 36),
        rank: 1,
        fixType: "pick",
      });
    });
    if (!actions.length) {
      actions.push({
        action: "adjust",
        label: "Add Pick",
        hint: shortenReviewHint(rd.why?.summary, 36) || "2nd-rounder closes gap",
        rank: 1,
        fixType: "pick",
      });
    }
    actions.push({
      action: "adjust",
      label: "Add Forward",
      hint: wantHint || "Raise package value",
      rank: 2,
      fixType: "forward",
    });
    const outgoingStar = safeArray(userOutgoing).find((a) => a?.type === "player");
    const fanHint = rd.fanHasBackend && rd.fanReasons?.[0]
      ? shortenReviewHint(rd.fanReasons[0], 36)
      : mainKey === "FIT" ? "Bad roster fit" : "";
    actions.push({
      action: "remove",
      label: outgoingStar?.name ? `Replace ${playerLastName(outgoingStar.name)}` : "Replace Asset",
      hint: fanHint || "Try different outgoing piece",
      playerName: outgoingStar?.name,
      side: "outgoing",
      rank: 3,
      fixType: "replace",
    });
    return actions.sort((a, b) => a.rank - b.rank).slice(0, 3);
  }

  if (mainKey === "CLAUSE") {
    const blocked = safeArray(partnerOutgoing).find((a) => {
      const c = String(a?.protection || a?.clauseLabel || "").toUpperCase();
      return a?.type === "player" && (c.includes("NMC") || c.includes("NTC"));
    }) || safeArray(userOutgoing).find((a) => {
      const c = String(a?.protection || a?.clauseLabel || "").toUpperCase();
      return a?.type === "player" && (c.includes("NMC") || c.includes("NTC"));
    });
    if (blocked?.name) {
      const isNtc = String(blocked.protection || blocked.clauseLabel || "").toUpperCase().includes("NTC")
        && !String(blocked.protection || blocked.clauseLabel || "").toUpperCase().includes("NMC");
      if (isNtc && !blocked.ntcWaived) {
        actions.push({
          action: "waive",
          label: `Ask ${playerLastName(blocked.name)} to Waive`,
          hint: "NTC can be waived",
          playerName: blocked.name,
          side: safeArray(partnerOutgoing).some((a) => a?.name === blocked.name) ? "incoming" : "outgoing",
          rank: 1,
          fixType: "ntc_waive",
        });
      }
      actions.push({
        action: "remove",
        label: `Replace ${playerLastName(blocked.name)}`,
        hint: "Clause block",
        playerName: blocked.name,
        side: safeArray(partnerOutgoing).some((a) => a?.name === blocked.name) ? "incoming" : "outgoing",
        rank: 2,
        fixType: "replace",
      });
    }
    actions.push({ action: "adjust", label: "Add Pick", hint: "Sweeten without clause player", rank: 3, fixType: "pick" });
    return actions.slice(0, 3);
  }

  if (!actions.length && rd.resultTone === "bad") {
    actions.push({ action: "adjust", label: "Add Pick", hint: "Sweeten the offer", rank: 1, fixType: "pick" });
    actions.push({ action: "adjust", label: "Add Forward", hint: "Raise package value", rank: 2, fixType: "forward" });
    actions.push({ action: "adjust", label: "Replace Asset", hint: "Try a different piece", rank: 3, fixType: "replace" });
  }
  return actions.sort((a, b) => (a.rank || 9) - (b.rank || 9)).slice(0, 3);
}

function buildFixSuggestions(evaluation, rd) {
  return buildFixActions(
    evaluation,
    rd,
    rd.noTouchConflict,
    rd.partnerOutgoing,
    rd.userOutgoing,
  ).map((a) => ({
    type: a.fixType || (a.action === "counter" ? "counter" : "hint"),
    label: a.label,
    text: a.hint || "",
    action: a.action,
    playerName: a.playerName,
    side: a.side,
    rank: a.rank,
  }));
}

function isHardProtectedTradeAsset(player) {
  if (!player || player.type === "pick") return false;
  const clause = String(player.protection || player.clauseLabel || "").toUpperCase();
  const isNmc = clause.includes("NMC");
  const needsNtcWaive =
    !player.ntcWaived &&
    (Boolean(player.requiresNtcWaive) ||
      (clause.includes("NTC") && !clause.includes("M-NTC")) ||
      clause.includes("M-NTC"));
  if (isNmc || needsNtcWaive) return true;
  if (player.tradeable === false && (player.tradeBlockReason || isNmc)) return true;
  return false;
}

function partnerUntouchableConflict(partnerOutgoing, untouchableNames) {
  const blocked = new Set(untouchableNames.map((n) => String(n).toLowerCase()));
  return safeArray(partnerOutgoing)
    .filter((a) => a?.type === "player" && blocked.has(String(a.name).toLowerCase()))
    .map((a) => a.name);
}

function formatReviewBulletLabel(raw) {
  const s = String(raw || "").trim();
  if (!s) return "";
  const pickMatch = s.match(/^(\d{4})-round(\d+)/i);
  if (pickMatch) return `${pickMatch[1]} R${pickMatch[2]}`;
  return s;
}

function isLikelyPlayerName(raw) {
  const t = formatReviewBulletLabel(raw);
  if (!t) return false;
  if (/^\d{4}\s*r\d+/i.test(t)) return false;
  if (/^(VALUE|CAP|PICK|ROSTER|CLAUSE|FIT|DEFENSE|GOALIE|TOP 6|DEPTH|RENTAL|ACCEPTED)$/i.test(t)) return false;
  return /[a-z]/i.test(t) && /\s/.test(t);
}

function dedupeReviewLines(lines) {
  const out = [];
  const seen = new Set();
  safeArray(lines).forEach((line) => {
    const formatted = formatReviewBulletLabel(line);
    const key = formatted.toLowerCase();
    if (formatted && !seen.has(key)) {
      seen.add(key);
      out.push(formatted);
    }
  });
  return out.slice(0, 5);
}

function buildReviewPlayerLookup({ meta, userTeamId, partnerTeamId, userOutgoing, partnerOutgoing }) {
  const lookup = {};
  const add = (p) => {
    if (!p?.name) return;
    lookup[p.name] = p;
    lookup[String(p.name).toLowerCase()] = p;
  };
  safeArray(userOutgoing).forEach((a) => { if (a?.type === "player") add(a); });
  safeArray(partnerOutgoing).forEach((a) => { if (a?.type === "player") add(a); });
  safeArray(meta?.players?.[userTeamId]).forEach(add);
  safeArray(meta?.players?.[partnerTeamId]).forEach(add);
  safeArray(meta?.prospects?.[userTeamId]).forEach(add);
  safeArray(meta?.prospects?.[partnerTeamId]).forEach(add);
  return lookup;
}

function lookupReviewPlayer(name, lookup) {
  if (!name || !lookup) return null;
  return lookup[name] || lookup[String(name).toLowerCase()] || null;
}

function splitReviewBullets(...parts) {
  const out = [];
  parts.forEach((p) => {
    if (Array.isArray(p)) {
      p.forEach((x) => {
        String(x || "").split(" · ").forEach((s) => {
          const t = s.trim();
          if (t) out.push(t);
        });
      });
      return;
    }
    if (typeof p === "string" && p.trim()) {
      p.split(" · ").forEach((s) => {
        const t = s.trim();
        if (t) out.push(t);
      });
    }
  });
  return dedupeReviewLines(out);
}

function TradeReviewInsightItem({ text, playerLookup, showHeadshot }) {
  const label = formatReviewBulletLabel(text);
  const player = isLikelyPlayerName(label) ? lookupReviewPlayer(label, playerLookup) : null;
  const ovr = player ? displayOvr(player) : null;
  if (player && showHeadshot) {
    return (
      <li className="trade-review-insight-player">
        <PlayerHeadshot
          player={ensurePlayerHeadshotFields(player)}
          size="sm"
          className="trade-review-insight-headshot"
          flag={null}
          number={null}
        />
        <span className="trade-review-insight-player-name">{label}</span>
        {ovr && ovr !== "—" ? <strong className="trade-review-insight-ovr">{ovr}</strong> : null}
      </li>
    );
  }
  return <li>{label}</li>;
}

function TradeReviewInsight({
  label,
  value,
  subline = "",
  lines = [],
  chips = [],
  valueClass = "",
  playerLookup = null,
  showPlayerHeadshots = false,
  meter,
  spread = false,
}) {
  const items = dedupeReviewLines(Array.isArray(lines) ? lines : splitReviewBullets(lines));
  const chipList = safeArray(chips).filter(Boolean).slice(0, 4);
  const pct = clamp(Math.round(Number(meter) || 0), 0, 100);
  const showMeter = Number.isFinite(Number(meter));
  return (
    <div className={`trade-review-insight ${valueClass} ${spread ? "spread" : ""}`}>
      <div className="trade-hub-panel-title">{label}</div>
      {value ? <div className={`trade-review-insight-value ${valueClass}`}>{value}</div> : null}
      {showMeter ? (
        <div className="trade-review-insight-meter">
          <div className="trade-review-insight-meter-fill" style={{ width: `${pct}%` }} />
        </div>
      ) : null}
      {subline ? <div className="trade-review-insight-sub">{subline}</div> : null}
      {items.length ? (
        <ul className="trade-review-insight-list">
          {items.map((item) => (
            <TradeReviewInsightItem
              key={item}
              text={item}
              playerLookup={playerLookup}
              showHeadshot={showPlayerHeadshots && isLikelyPlayerName(item)}
            />
          ))}
        </ul>
      ) : null}
      {chipList.length ? (
        <div className="trade-hub-chip-row trade-review-insight-chips">
          {chipList.map((c) => <span key={c} className="trade-hub-chip trade-hub-chip-need">{c}</span>)}
        </div>
      ) : null}
    </div>
  );
}

function reviewDisplayChips(block, preferPlayers = true) {
  const players = dedupeReviewLines(safeArray(block?.players).map(formatReviewBulletLabel));
  const chips = safeArray(block?.chips).map(formatReviewBulletLabel);
  if (preferPlayers && players.length) {
    const positional = chips.filter(
      (c) => !players.some((p) => String(p).toUpperCase() === String(c).toUpperCase()),
    );
    return dedupeReviewLines([...players, ...positional]).slice(0, 5);
  }
  return dedupeReviewLines(chips).slice(0, 5);
}

function reviewWantsPlayers(wants) {
  return dedupeReviewLines(safeArray(wants?.players).filter(isLikelyPlayerName));
}

function fanBacklashTone(fanBlock) {
  const heat = Number(fanBlock?.score) || 0;
  if (heat >= 55) return "bad";
  if (heat >= 30) return "warn";
  return "good";
}

function fanBacklashFromEvaluation(evalFan) {
  if (!evalFan || typeof evalFan !== "object") return null;
  const score = Number(
    evalFan.fan_heat ?? (100 - Number(evalFan.fan_reaction_score ?? 50)),
  ) || 0;
  return {
    label: String(evalFan.fan_heat_label || fanHeatLabelFromHeat(score)).toUpperCase(),
    score,
    reasons: safeArray(evalFan.fan_factors).slice(0, 2),
    source: "backend",
  };
}

function resolveTradeReviewData({
  evaluation,
  fanMeter,
  userTeamId,
  partnerTeamId,
  partnerTeam,
  userTeam,
  userOutgoing,
  partnerOutgoing,
  meta,
}) {
  const review = evaluation?.trade_review || null;
  const hasBackendReview = Boolean(review);
  const userCap = evaluation?.cap_impact?.[userTeamId];
  const localUserGive = Math.round(packageDisplayValue(userOutgoing));
  const localPartnerGive = Math.round(packageDisplayValue(partnerOutgoing));
  const fanHeat = Number(fanMeter?.heat ?? (100 - Number(fanMeter?.score ?? fanMeter))) || 0;

  const fallbackWhy = {
    primary_code: shortReviewVerdict(evaluation),
    summary: tradeReviewReasonSummary(evaluation),
    chips: shortReviewVerdict(evaluation) === "ACCEPTED"
      ? ["ACCEPTED"]
      : reviewReasonChips(evaluation),
    players: [],
    source: "fallback",
  };
  const fallbackWants = {
    summary: partnerWantChips(partnerTeam, evaluation).length ? "" : "No clear ask.",
    chips: partnerWantChips(partnerTeam, evaluation),
    players: [],
    source: "fallback",
  };
  const untouchableNames = partnerUntouchableChips(meta, partnerTeamId);
  const fallbackUntouchables = {
    summary: untouchableNames.length ? "Core pieces protected." : "No protected core.",
    players: untouchableNames,
    chips: [],
    source: "fallback",
  };

  const explicitAfter = Number(userCap?.after_usable ?? userCap?.projectedCapSpace);
  const explicitDelta = Number(userCap?.delta);
  const baseCap = Number(userTeam?.capSpace);
  const fallbackProjected = Number.isFinite(baseCap)
    ? baseCap + packageCapDelta(userOutgoing) - packageCapDelta(partnerOutgoing)
    : null;
  const fallbackDelta = Number.isFinite(baseCap) && Number.isFinite(fallbackProjected)
    ? fallbackProjected - baseCap
    : null;
  const afterUse = Number.isFinite(explicitAfter) ? explicitAfter : fallbackProjected;
  const deltaUse = Number.isFinite(explicitDelta) ? explicitDelta : fallbackDelta;
  const fallbackCap = {
    projected_space_m: Number.isFinite(afterUse) ? afterUse : null,
    delta_m: Number.isFinite(deltaUse) ? deltaUse : null,
    label: Number.isFinite(deltaUse)
      ? `${deltaUse >= 0 ? "+" : "-"}${formatMoneyShort(Math.abs(deltaUse))}`
      : "—",
    tone: Number.isFinite(afterUse) ? (afterUse >= 0 ? "good" : "bad") : "neutral",
  };

  const why = review?.why || fallbackWhy;
  const wants = review?.team_wants || fallbackWants;
  const untouchables = review?.untouchables || fallbackUntouchables;
  const gmRead = review?.gm_interest || (hasBackendReview
    ? { label: "—", score: 0, reasons: [], source: "backend" }
    : {
      label: gmReviewShort(evaluation, partnerTeamId),
      score: gmInterestPct(evaluation, partnerTeamId),
      reasons: [],
      source: "fallback",
    });
  const balance = review?.trade_balance || (hasBackendReview
    ? { label: "—", score: 0, summary: "", source: "backend" }
    : {
      label: balanceReviewShort(localUserGive, localPartnerGive),
      score: tradeBalancePct(localUserGive, localPartnerGive),
      summary: "",
      source: "fallback",
    });
  if (!balance.summary) {
    const net = localUserGive - localPartnerGive;
    balance.summary = net === 0
      ? "Package is near fair."
      : net > 0
        ? "You receive more value."
        : "You give more value.";
  }
  const fanBacklash = review?.fan_backlash
    || fanBacklashFromEvaluation(evaluation?.fan_reaction)
    || {
      label: fansReviewLabel(fanMeter),
      score: fanHeat,
      reasons: safeArray(fanMeter?.factors).slice(0, 2),
      source: "fallback",
    };
  const capAfter = review?.cap_after || (hasBackendReview
    ? { projected_space_m: null, delta_m: null, label: "—", tone: "neutral", source: "backend" }
    : fallbackCap);

  const playerLookup = buildReviewPlayerLookup({ meta, userTeamId, partnerTeamId, userOutgoing, partnerOutgoing });
  const untouchableRows = buildUntouchableRows(
    [...safeArray(untouchables.players), ...safeArray(untouchables.chips)],
    playerLookup,
  );
  // Only hard locks (NMC / unwaived NTC / unavailable) block proposing — high OVR is a soft risk.
  const noTouchConflict = safeArray(partnerOutgoing)
    .filter((a) => a?.type === "player" && isHardProtectedTradeAsset(lookupReviewPlayer(a.name, playerLookup) || a))
    .map((a) => a.name);
  const netGap = localUserGive - localPartnerGive;
  const isRejected = !(evaluation?.accepted && evaluation?.can_execute);
  const capBefore = Number.isFinite(baseCap) ? baseCap : null;
  const capAfterNum = Number(capAfter.projected_space_m);
  const capBlockContext = resolveCapBlockContext({
    evaluation,
    userTeamId,
    partnerTeamId,
    partnerTeam,
    userTeam,
    userOutgoing,
    partnerOutgoing,
    capBefore,
    capAfter: capAfterNum,
  });

  const blockers = resolveReviewBlockers(evaluation, why, fanBacklash.score, noTouchConflict);
  const mainProblem = resolveMainProblem({
    evaluation,
    why,
    noTouchConflict,
    partnerOutgoing,
    userOutgoing,
    partnerTeam,
    userTeam,
    userTeamId,
    partnerTeamId,
    capBefore,
    capAfter: capAfterNum,
    netGap,
    balance,
  });
  const valueLabel = reviewValueLabelForBlock(netGap, balance, isRejected, mainProblem);
  const valueMarkerPct = valueGapMarkerPct(netGap, balance.score);
  const humanFanReasons = dedupeReviewLines(
    safeArray(fanBacklash.reasons).map(humanizeFanFactor).filter(Boolean),
  ).filter((r, i, arr) => {
    const low = r.toLowerCase();
    if (low.includes("franchise star") && arr.some((x) => x.toLowerCase().includes("star player"))) return false;
    return true;
  }).slice(0, 1);
  const fanHasBackend = fanBacklash?.source === "backend"
    || Boolean(safeArray(evaluation?.fan_reaction?.fan_factors).length);
  const wantsHasBackend = wants?.source === "backend";
  const wantsRanked = buildWantsRanked(reviewWantsPlayers(wants), wants.summary, wants, evaluation);
  const metricPrimary = mainProblem.primaryKey === "CAP" ? "cap"
    : mainProblem.primaryKey === "VALUE" ? "value"
      : mainProblem.primaryKey || "value";

  const draft = {
    hasBackendReview,
    resultLabel: review?.result_label || shortReviewVerdict(evaluation),
    resultTone: mapReviewResultTone(review?.result_tone) || shortReviewVerdictTone(shortReviewVerdict(evaluation)),
    why,
    whyChips: reviewDisplayChips(why, true).filter((c) => !/^(VALUE|CAP OK|FIT|ACCEPTED)$/i.test(c)),
    wants,
    wantsPlayers: reviewWantsPlayers(wants),
    wantsRanked,
    wantsChips: safeArray(wants?.chips).slice(0, 4),
    untouchables,
    untouchableRows,
    gmRead,
    gmDisplay: gmReadDisplayLabel(gmRead, evaluation, partnerTeamId),
    gmDetail: gmReadDetail(mainProblem, gmRead, evaluation, partnerTeamId),
    balance,
    valueLabel,
    valueMarkerPct,
    fanBacklash,
    fanReasons: humanFanReasons,
    fanHasBackend,
    wantsHasBackend,
    fanSubject: fanHasBackend ? humanFanReasons[0] : "",
    capAfter,
    capBefore,
    capBlockContext,
    dualCap: resolveDualTeamCap({
      evaluation,
      userTeamId,
      partnerTeamId,
      userTeam,
      partnerTeam,
      userOutgoing,
      partnerOutgoing,
      capBefore,
    }),
    blockers,
    mainProblem,
    noTouchConflict,
    partnerOutgoing,
    userOutgoing,
    partnerTeam,
    userTeam,
    netGap,
    metricPrimary,
    fixSuggestions: [],
    fanTone: (hasBackendReview || evaluation?.fan_reaction)
      ? fanBacklashTone(fanBacklash)
      : fansReviewTone(fanMeter),
    balanceTone: isRejected && valueLabel.includes("Close") ? "warn" : "balance",
    gmTone: "gm",
    playerLookup,
  };
  draft.fixSuggestions = buildFixSuggestions(evaluation, draft);
  draft.headlineCause = reviewHeadlineCause(partnerTeam, mainProblem);
  draft.blockerChip = reviewBlockerChipLabel(mainProblem?.primaryKey);
  draft.nextStep = buildNextStepText(draft);
  draft.partnerLabel = partnerShortName(partnerTeam);

  return draft;
}

function TradeReviewReadout({ label, value, meter, subline, tone = "neutral" }) {
  const pct = clamp(Math.round(Number(meter) || 0), 0, 100);
  const showMeter = Number.isFinite(Number(meter));
  return (
    <div className={`trade-review-readout ${tone}`}>
      <div className="trade-review-readout-head">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      {subline ? <p className="trade-review-readout-sub">{subline}</p> : null}
      {showMeter ? (
        <div className="trade-review-readout-meter">
          <div className="trade-review-readout-meter-fill" style={{ width: `${pct}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function TradeReviewTextCard({ label, summary, chips, tone = "neutral" }) {
  const list = safeArray(chips).filter(Boolean).slice(0, 4);
  return (
    <div className={`trade-review-text-card ${tone}`}>
      <div className="trade-review-text-head">
        <span>{label}</span>
      </div>
      {summary ? <p className="trade-review-text-summary">{summary}</p> : null}
      {list.length > 0 && (
        <div className="trade-review-chip-row">
          {list.map((c) => <span key={c}>{c}</span>)}
        </div>
      )}
    </div>
  );
}

function TradeReviewMeter({ label, value, tone = "neutral", text }) {
  const pct = clamp(Math.round(Number(value) || 0), 0, 100);
  return (
    <div className={`trade-review-meter-card ${tone}`}>
      <div className="trade-review-meter-head">
        <span>{label}</span>
        <strong>{text}</strong>
      </div>
      <div className="trade-review-meter-track">
        <div className="trade-review-meter-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function TradeReviewInfoCard({ label, main, chips, tone = "neutral", emptyFallback = "UNKNOWN" }) {
  const list = safeArray(chips).filter(Boolean).slice(0, 4);
  return (
    <div className={`trade-review-info-card ${tone}`}>
      <div className="trade-review-info-head">
        <span>{label}</span>
      </div>
      {main ? <strong className="trade-review-info-main">{main}</strong> : null}
      <div className="trade-review-info-chips">
        {list.length ? list.map((c) => (
          <span key={c}>{c}</span>
        )) : <span>{emptyFallback}</span>}
      </div>
    </div>
  );
}

function TradeReviewCapCard({ data, userCap, userTeam, userOutgoing, partnerOutgoing }) {
  let afterText = "—";
  let deltaText = "—";
  let tone = "neutral";

  if (data && (data.projected_space_m != null || data.label)) {
    const after = Number(data.projected_space_m);
    const delta = Number(data.delta_m);
    tone = data.tone || (Number.isFinite(after) ? (after >= 0 ? "good" : "bad") : "neutral");
    afterText = Number.isFinite(after) ? formatMoneyShort(after) : "—";
    deltaText = data.label && data.label !== "—"
      ? data.label
      : Number.isFinite(delta)
        ? `${delta >= 0 ? "+" : "-"}${formatMoneyShort(Math.abs(delta))}`
        : "—";
  } else {
    const explicitAfter = Number(userCap?.after_usable ?? userCap?.projectedCapSpace);
    const explicitDelta = Number(userCap?.delta);
    const baseCap = Number(userTeam?.capSpace);
    const fallbackProjected = Number.isFinite(baseCap)
      ? baseCap + packageCapDelta(userOutgoing) - packageCapDelta(partnerOutgoing)
      : null;
    const fallbackDelta = Number.isFinite(baseCap) && Number.isFinite(fallbackProjected)
      ? fallbackProjected - baseCap
      : null;
    const after = Number.isFinite(explicitAfter) ? explicitAfter : fallbackProjected;
    const delta = Number.isFinite(explicitDelta) ? explicitDelta : fallbackDelta;
    const good = Number.isFinite(after) ? after >= 0 : Number.isFinite(delta) ? delta >= 0 : true;
    tone = good ? "good" : "bad";
    afterText = Number.isFinite(after) ? formatMoneyShort(after) : "—";
    deltaText = Number.isFinite(delta)
      ? `${delta >= 0 ? "+" : "-"}${formatMoneyShort(Math.abs(delta))}`
      : "—";
  }

  return (
    <TradeReviewInsight
      label="CAP AFTER"
      value={afterText}
      lines={[`Trade impact ${deltaText}`]}
      valueClass={tone}
    />
  );
}

function TradeReviewValueBar({ asset, evaluation, breakdownSide, breakdownDirection, compact }) {
  const valueItem = resolveReviewAssetValueItem(asset, evaluation, breakdownSide, breakdownDirection);
  const pct = reviewAssetValuePct(valueItem);
  const tierClass = assetValueTierClass(valueItem);
  const label = assetValueLabel(valueItem);
  return (
    <div className={`trade-review-anchor-value ${tierClass} ${compact ? "compact" : ""}`}>
      <span className="trade-review-anchor-value-label">{label !== "UNKNOWN" ? label : "VALUE"}</span>
      <div className="trade-player-value-track">
        <div className="trade-player-value-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function TradeReviewAnchorAsset({
  asset,
  protectedNames,
  meta,
  evaluation,
  breakdownSide,
  breakdownDirection,
  onClick,
  compact,
}) {
  if (!asset) return null;
  const isProtected = asset.type === "player"
    && protectedNames?.has(String(asset.name).toLowerCase());

  if (asset.type === "pick") {
    const origin = resolvePickOriginTeam(asset, meta);
    const range = pickRangeDisplay(asset);
    return (
      <button
        type="button"
        className={`trade-review-anchor-asset pick ${compact ? "compact" : ""}`}
        onClick={onClick}
        title={`${asset.year} Round ${asset.round}`}
      >
        <div className="trade-review-anchor-pick-top">
          <PickIcon round={asset.round} year={asset.year} className="trade-review-anchor-pick-icon" />
          <div className="trade-review-anchor-pick-main">
            <strong>{asset.year} Round {asset.round}</strong>
            <span>{pickReviewShortLabel(asset)}{range ? ` · ${range}` : ""}</span>
          </div>
          <div className="trade-review-anchor-pick-origin">
            {origin.team?.abbr || origin.abbr ? (
              <>
                <TradeLogo team={origin.team} size={compact ? 22 : 28} />
                <span>{origin.abbr || origin.team?.abbr}</span>
              </>
            ) : (
              <span className="trade-review-pick-own-fallback">Owned</span>
            )}
          </div>
        </div>
        <TradeReviewValueBar
          asset={asset}
          evaluation={evaluation}
          breakdownSide={breakdownSide}
          breakdownDirection={breakdownDirection}
          compact={compact}
        />
      </button>
    );
  }

  const role = asset.role || roleFromOverall(asset.ovr, asset.pos);
  const franchise = isFranchisePlayer(asset);
  const ovr = displayOvr(asset);
  const pos = String(asset.pos || "F").toUpperCase();
  const term = formatPlayerTermLabel(asset);

  return (
    <button
      type="button"
      className={`trade-review-anchor-asset player ${compact ? "compact" : ""} ${isProtected ? "protected" : ""}`}
      onClick={onClick}
      title={isProtected ? "Protected — cannot acquire" : "View player"}
    >
      <div className="trade-review-anchor-player-row">
        <div className="trade-review-anchor-shot">
          <PlayerHeadshot
            player={ensurePlayerHeadshotFields(asset)}
            size={compact ? "sm" : "md"}
            className="trade-review-anchor-headshot"
            flag={null}
            number={null}
          />
        </div>
        <div className="trade-review-anchor-body">
          <div className="trade-review-anchor-name-row">
            {franchise ? <em className="trade-review-franchise-mark">★</em> : null}
            <strong className="trade-review-anchor-name">{asset.name}</strong>
          </div>
          <div className="trade-review-anchor-meta">
            <span className={`trade-review-pos-badge trade-pos-icon ${pos === "G" ? "pos-g" : pos === "D" ? "pos-d" : "pos-f"}`}>
              {pos}
            </span>
            <span>{asset.age}Y</span>
            <span>{role}</span>
            <span>{formatPlayerCapLabel(asset)}</span>
            <span>{term}</span>
          </div>
          <TradeReviewValueBar
            asset={asset}
            evaluation={evaluation}
            breakdownSide={breakdownSide}
            breakdownDirection={breakdownDirection}
            compact={compact}
          />
          {isProtected ? <span className="trade-review-mini-warn">Protected</span> : null}
        </div>
        {ovr && ovr !== "—" ? (
          <div className="trade-review-anchor-ovr">
            <span>OVR</span>
            <strong>{ovr}</strong>
          </div>
        ) : null}
      </div>
    </button>
  );
}

function TradeReviewMiniAsset({ asset, protectedNames, meta, evaluation, breakdownSide, breakdownDirection, onClick }) {
  return (
    <TradeReviewAnchorAsset
      asset={asset}
      protectedNames={protectedNames}
      meta={meta}
      evaluation={evaluation}
      breakdownSide={breakdownSide}
      breakdownDirection={breakdownDirection}
      onClick={onClick}
      compact
    />
  );
}

function TradeReviewPackageSide({
  label,
  team,
  assets,
  protectedNames,
  meta,
  evaluation,
  breakdownSide,
  breakdownDirection,
  onAssetClick,
}) {
  const list = safeArray(assets).filter(Boolean);
  const multi = list.length > 1;
  return (
    <div className="trade-review-package-side anchor">
      <div className="trade-review-side-head">
        <TradeLogo team={team} size={36} />
        <span className="trade-review-side-label">{label}</span>
      </div>
      <div className="trade-review-anchor-assets">
        {list.length ? list.map((a) => (
          <TradeReviewAnchorAsset
            key={`${a.type}-${a.id}`}
            asset={a}
            protectedNames={protectedNames}
            meta={meta}
            evaluation={evaluation}
            breakdownSide={breakdownSide}
            breakdownDirection={breakdownDirection}
            onClick={onAssetClick ? () => onAssetClick(a) : undefined}
            compact={multi}
          />
        )) : <div className="trade-review-anchor-asset empty">Empty</div>}
      </div>
    </div>
  );
}

function TradeReviewValueTile({ label, markerPct, tone, primary }) {
  return (
    <div className={`trade-review-icon-tile balance ${tone} ${primary ? "primary-metric" : "secondary-metric"}`}>
      <span className="trade-review-tile-kicker">Value</span>
      <strong className="trade-review-tile-main">{label}</strong>
      <div className="trade-review-value-track">
        <div className="trade-review-value-center" />
        <div className="trade-review-value-marker" style={{ left: `${markerPct}%` }} />
      </div>
    </div>
  );
}

function TradeReviewCapTile({ capCtx, dualCap, tone, primary, capBlocked }) {
  const dc = dualCap || {};
  const bad = capBlocked || dc.fits === false;
  const userBeforeText = Number.isFinite(dc.userBefore) ? formatMoneyShort(dc.userBefore) : "—";
  const userAfterText = Number.isFinite(dc.userAfter) ? formatMoneyShort(dc.userAfter) : "—";
  const partnerBeforeText = Number.isFinite(dc.partnerBefore) ? formatMoneyShort(dc.partnerBefore) : "—";
  const partnerAfterText = Number.isFinite(dc.partnerAfter) ? formatMoneyShort(dc.partnerAfter) : "—";
  const userDeltaText = formatCapDeltaLabel(dc.userDelta);
  const partnerDeltaText = formatCapDeltaLabel(dc.partnerDelta);
  const overText = bad
    ? (capCtx?.overBy > 0
      ? `${formatMoneyShort(capCtx.overBy)} over cap`
      : `${capCtx?.failingAbbr || dc.partnerAbbr || "Team"} cannot absorb`)
    : "Cap fits";

  return (
    <div className={`trade-review-icon-tile cap ${tone} ${bad ? "bad" : ""} ${primary ? "primary-metric" : "secondary-metric"}`}>
      <span className="trade-review-tile-kicker">Cap Space</span>
      <div className="trade-review-dual-cap">
        <div className="trade-review-dual-cap-row">
          <span className="trade-review-dual-cap-team">{dc.userAbbr || "YOU"}</span>
          <strong className="trade-review-dual-cap-flow">
            {userBeforeText}
            <em>→</em>
            {userAfterText}
          </strong>
          {userDeltaText ? <span className={`trade-review-dual-cap-delta ${dc.userDelta >= 0 ? "pos" : "neg"}`}>{userDeltaText}</span> : null}
        </div>
        <div className="trade-review-dual-cap-row">
          <span className="trade-review-dual-cap-team">{dc.partnerAbbr || "THEM"}</span>
          <strong className="trade-review-dual-cap-flow">
            {partnerBeforeText}
            <em>→</em>
            {partnerAfterText}
          </strong>
          {partnerDeltaText ? <span className={`trade-review-dual-cap-delta ${dc.partnerDelta >= 0 ? "pos" : "neg"}`}>{partnerDeltaText}</span> : null}
        </div>
      </div>
      <div className={`trade-review-cap-verdict ${bad ? "bad" : "good"}`}>{overText}</div>
    </div>
  );
}

function TradeReviewContextRow({ row, playerLookup, onClick }) {
  const player = lookupReviewPlayer(row.name, playerLookup);
  const ovr = player ? displayOvr(player) : null;
  const pos = player ? String(player.pos || "F").toUpperCase() : null;
  return (
    <button type="button" className="trade-review-context-row anchor" onClick={onClick}>
      <span className={`trade-review-want-tag ${String(row.category || "").toLowerCase().replace(/\s+/g, "-")}`}>
        {row.category}
      </span>
      <span className="trade-review-context-text">
        <span className="trade-review-context-name">
          {pos ? (
            <span className={`trade-review-pos-badge sm trade-pos-icon ${pos === "G" ? "pos-g" : pos === "D" ? "pos-d" : "pos-f"}`}>
              {pos}
            </span>
          ) : null}
          {row.name}
        </span>
        <span className="trade-review-context-note">{row.note}</span>
      </span>
      {ovr && ovr !== "—" ? <span className="trade-review-context-ovr">{ovr}</span> : null}
    </button>
  );
}

function TradeReviewLockRow({ row, playerLookup, conflict, onClick }) {
  const player = lookupReviewPlayer(row.name, playerLookup);
  const ovr = player ? displayOvr(player) : null;
  const isConflict = conflict?.has(String(row.name).toLowerCase());
  return (
    <button
      type="button"
      className={`trade-review-context-row lock anchor ${isConflict ? "conflict" : ""}`}
      onClick={onClick}
    >
      <span className="trade-review-lock-tag">{row.tag || "CORE"}</span>
      <span className="trade-review-context-text">
        <span className="trade-review-context-name">{row.name}</span>
        <span className="trade-review-context-note">{row.reason}</span>
      </span>
      {ovr && ovr !== "—" ? <span className="trade-review-context-ovr">{ovr}</span> : null}
    </button>
  );
}

function TradeReviewDrawer({
  evaluation,
  userTeamId,
  partnerTeamId,
  partnerTeam,
  userTeam,
  hasAssets,
  fanMeter,
  tradeMarket,
  tradeHistory,
  comparedAssets,
  meta,
  userOutgoing,
  partnerOutgoing,
  onClose,
  onPropose,
  onReset,
  onRemoveIncoming,
  onRemoveOutgoing,
  onViewTeamNeeds,
  onAssetClick,
  proposeDisabled,
  proposeLabel,
  proposeBlockReason,
}) {
  void tradeMarket;
  void tradeHistory;
  void comparedAssets;
  void fanMeter;
  const [wantsExpanded, setWantsExpanded] = useState(false);
  const [noTouchExpanded, setNoTouchExpanded] = useState(false);

  if (!evaluation && !hasAssets) return null;

  const rd = resolveTradeReviewData({
    evaluation,
    fanMeter,
    userTeamId,
    partnerTeamId,
    partnerTeam,
    userTeam,
    userOutgoing,
    partnerOutgoing,
    meta,
  });
  const playerLookup = rd.playerLookup;
  const protectedSet = new Set(rd.untouchableRows.map((r) => String(r.name).toLowerCase()));
  const conflictSet = new Set(rd.noTouchConflict.map((n) => String(n).toLowerCase()));
  const fanPct = clamp(Math.round(Number(rd.fanBacklash.score) || 0), 0, 100);
  void fanPct;
  const accepted = rd.resultTone === "good";
  const showNoTouchExpanded = noTouchExpanded || rd.noTouchConflict.length > 0;
  const visibleWants = wantsExpanded ? rd.wantsRanked : rd.wantsRanked.slice(0, 3);
  const visibleLocks = showNoTouchExpanded ? rd.untouchableRows : rd.untouchableRows.slice(0, 2);
  const capPrimary = rd.metricPrimary === "cap";
  const wantsHidden = Math.max(0, rd.wantsRanked.length - 3);
  const locksHidden = Math.max(0, rd.untouchableRows.length - 2);
  const proposeReason = proposeDisabledReasonLabel(rd, proposeBlockReason);
  const fanReason = rd.fanHasBackend ? (rd.fanReasons[0] || "") : "";
  const partnerLabel = rd.partnerLabel || partnerShortName(partnerTeam);
  const wantsSummary = rd.wantsHasBackend && rd.wants?.summary
    && !String(rd.wants.summary).toLowerCase().includes("no clear")
    ? rd.wants.summary
    : "";

  const handleFixAction = (suggestion) => {
    if (suggestion.action === "remove" && suggestion.playerName) {
      if (suggestion.side === "outgoing" && onRemoveOutgoing) {
        onRemoveOutgoing(suggestion.playerName);
      } else if (onRemoveIncoming) {
        onRemoveIncoming(suggestion.playerName);
      }
      onClose();
      return;
    }
    if (suggestion.action === "retain") {
      onClose();
      return;
    }
    onClose();
  };

  const capTile = (
    <TradeReviewCapTile
      capCtx={rd.capBlockContext}
      dualCap={rd.dualCap}
      tone={rd.capAfter.tone || "neutral"}
      primary={capPrimary}
      capBlocked={rd.mainProblem?.primaryKey === "CAP"}
    />
  );
  const valueTile = (
    <TradeReviewValueTile
      label={rd.valueLabel}
      markerPct={rd.valueMarkerPct}
      tone={rd.balanceTone}
      primary={!capPrimary && rd.metricPrimary === "value"}
    />
  );

  return (
    <div className="trade-review-overlay" onClick={onClose}>
      <div className="trade-review-shell trade-review-board" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="trade-review-x" onClick={onClose} aria-label="Close">×</button>

        <header className="trade-review-board-head trade-review-team-head">
          <TradeLogo team={userTeam} size={40} />
          <div className="trade-review-title-block">
            <h2>Trade Review</h2>
            <div className="trade-review-team-matchup">
              <strong>{userTeam?.abbr || "YOU"}</strong>
              <span className="trade-review-match-arrow">→</span>
              <strong>{partnerTeam?.abbr || "THEM"}</strong>
            </div>
          </div>
          <TradeLogo team={partnerTeam} size={40} />
        </header>

        <div className="trade-review-board-main">
          <TradeReviewPackageSide
            label="You Send"
            team={userTeam}
            assets={userOutgoing}
            protectedNames={protectedSet}
            meta={meta}
            evaluation={evaluation}
            breakdownSide="user"
            breakdownDirection="outgoing"
            onAssetClick={onAssetClick}
          />

          <div className="trade-review-verdict-stack">
            <div className={`trade-review-verdict-core ${rd.resultTone}`}>
              <strong className="trade-review-result-word">{rd.resultLabel}</strong>
              {!accepted ? (
                <>
                  <p className="trade-review-headline-cause">{rd.headlineCause}</p>
                  <span className="trade-review-blocker-chip">{rd.blockerChip}</span>
                </>
              ) : (
                <p className="trade-review-headline-cause">{rd.headlineCause}</p>
              )}
            </div>

            {rd.noTouchConflict.length ? (
              <div className="trade-review-hard-banner">
                Cannot acquire {rd.noTouchConflict.map(playerLastName).join(", ")}
              </div>
            ) : null}

            {!accepted && rd.fixSuggestions.length ? (
              <div className="trade-review-fix-actions large">
                {rd.fixSuggestions.map((s) => (
                  <button
                    key={`${s.label}-${s.action}-${s.rank}`}
                    type="button"
                    className={`trade-review-fix-btn large ${s.type} ${s.rank === 1 ? "best" : ""}`}
                    onClick={() => handleFixAction(s)}
                  >
                    <strong>{s.label}</strong>
                    {s.text ? <span>{s.text}</span> : null}
                  </button>
                ))}
              </div>
            ) : null}

            {!accepted ? (
              <div className="trade-review-next-step">
                <span className="trade-review-next-step-label">Next Step</span>
                <p>{rd.nextStep}</p>
              </div>
            ) : null}
          </div>

          <TradeReviewPackageSide
            label="You Get"
            team={partnerTeam}
            assets={partnerOutgoing}
            protectedNames={protectedSet}
            meta={meta}
            evaluation={evaluation}
            breakdownSide="user"
            breakdownDirection="incoming"
            onAssetClick={onAssetClick}
          />
        </div>

        <div className="trade-review-icon-grid trade-review-icon-grid-4">
          {capPrimary ? capTile : valueTile}
          {capPrimary ? valueTile : capTile}
          <div className={`trade-review-icon-tile gm ${rd.gmTone} secondary-metric`}>
            <span className="trade-review-tile-kicker">Partner GM</span>
            <strong className="trade-review-tile-main">{rd.gmDisplay}</strong>
            {rd.gmDetail ? <span className="trade-review-tile-detail">{rd.gmDetail}</span> : null}
          </div>
          <div className={`trade-review-icon-tile fans ${rd.fanTone} secondary-metric`}>
            <span className="trade-review-tile-kicker">Fan Heat</span>
            <strong className="trade-review-tile-main">{rd.fanBacklash.label}</strong>
            {fanReason ? <span className="trade-review-fan-subject">{fanReason}</span> : null}
          </div>
        </div>

        <div className="trade-review-negot-context compact">
          <section className="trade-review-context-panel compact">
            <div className="trade-review-context-head">
              <h3>{partnerLabel} Wants</h3>
              {wantsHidden > 0 && !wantsExpanded ? (
                <button type="button" className="trade-review-expand-btn" onClick={() => setWantsExpanded(true)}>
                  +{wantsHidden} more
                </button>
              ) : wantsExpanded && rd.wantsRanked.length > 3 ? (
                <button type="button" className="trade-review-expand-btn" onClick={() => setWantsExpanded(false)}>
                  Less
                </button>
              ) : null}
            </div>
            {wantsSummary ? <p className="trade-review-panel-lead">{wantsSummary}</p> : null}
            {visibleWants.length ? visibleWants.map((row) => (
              <TradeReviewContextRow
                key={row.name}
                row={row}
                playerLookup={playerLookup}
                onClick={onViewTeamNeeds}
              />
            )) : (
              <p className="trade-review-panel-empty">
                {rd.wantsHasBackend ? "No specific targets listed." : "Run trade check for partner asks."}
              </p>
            )}
          </section>

          <section className={`trade-review-context-panel compact ${rd.noTouchConflict.length ? "alert" : ""}`}>
            <div className="trade-review-context-head">
              <h3>{partnerLabel} No-Touch List</h3>
              {locksHidden > 0 && !showNoTouchExpanded ? (
                <button type="button" className="trade-review-expand-btn" onClick={() => setNoTouchExpanded(true)}>
                  +{locksHidden} more
                </button>
              ) : showNoTouchExpanded && rd.untouchableRows.length > 2 ? (
                <button type="button" className="trade-review-expand-btn" onClick={() => setNoTouchExpanded(false)}>
                  Less
                </button>
              ) : null}
            </div>
            {visibleLocks.length ? visibleLocks.map((row) => (
              <TradeReviewLockRow
                key={row.name}
                row={row}
                playerLookup={playerLookup}
                conflict={conflictSet}
                onClick={onAssetClick ? () => onAssetClick(lookupReviewPlayer(row.name, playerLookup)) : undefined}
              />
            )) : (
              <p className="trade-review-panel-empty">No protected core listed.</p>
            )}
          </section>
        </div>

        <footer className="trade-review-footer trade-review-footer-negotiate">
          <button type="button" className="trade-review-btn secondary" onClick={onReset}>Reset Deal</button>
          <button type="button" className="trade-review-btn adjust-primary" onClick={onClose}>
            Adjust Package
          </button>
          <button
            type="button"
            className={`trade-review-btn propose ${accepted ? "ready" : "blocked"}`}
            disabled={proposeDisabled || !accepted}
            onClick={onPropose}
          >
            <span className="trade-review-btn-label">{proposeLabel || "Propose Trade"}</span>
            {!accepted && proposeReason ? (
              <em className="trade-review-btn-block-reason">{proposeReason}</em>
            ) : null}
          </button>
        </footer>
      </div>
    </div>
  );
}

function TeamDetailDrawer({ team, meta, partnerId, onClose }) {
  if (!team) return null;
  const st = team.standings || {};
  const capDetail = team.capDetail || {};
  const picks = safeArray(meta?.picks?.[team.id]).slice(0, 8);
  const youth = safeArray(meta?.prospects?.[team.id]).slice(0, 5);

  return (
    <div className="trade-drawer-overlay" onClick={onClose}>
      <div className="trade-drawer trade-drawer-team" onClick={(e) => e.stopPropagation()}>
        <div className="trade-drawer-head">
          <TradeLogo team={team} size={56} />
          <div>
            <strong>{team.name}</strong>
            <span>{team.record} · {team.direction}</span>
          </div>
          <button type="button" className="trade-drawer-close" onClick={onClose}>×</button>
        </div>
        <div className="trade-drawer-body">
          <div className="trade-review-grid">
            <div><span>OVR</span><strong>{team.ratings?.overall ?? "—"}</strong></div>
            <div><span>Cap Space</span><strong>{team.capSpace != null ? formatMoneyM(team.capSpace) : "—"}</strong></div>
            <div><span>Playoff</span><strong>{team.playoffOdds != null ? `${team.playoffOdds}%` : "—"}</strong></div>
            <div><span>Pace</span><strong>{team.pointsPace}</strong></div>
            <div><span>Conf</span><strong>#{st.conferenceRank || "—"}</strong></div>
            <div><span>Div</span><strong>#{st.divisionRank || "—"}</strong></div>
          </div>
          <p className="trade-drawer-line">GM: {team.gmPersonality || "—"}</p>
          {team.marketPressure?.label && (
            <p className="trade-drawer-line">
              Market: {team.marketPressure.label}
              {team.marketPressure.tradeLeverageHint
                ? ` · Leverage: ${team.marketPressure.tradeLeverageHint}`
                : ""}
            </p>
          )}
          {capDetail.projected_deadline_space != null && (
            <p className="trade-drawer-line">Deadline room: {formatMoneyM(capDetail.projected_deadline_space)}</p>
          )}
          {capDetail.retained_slots_used != null && (
            <p className="trade-drawer-line">Retained slots: {capDetail.retained_slots_used}/{capDetail.retained_slots_max ?? 3}</p>
          )}
          {Number(capDetail.ltir_pool) > 0 && (
            <p className="trade-drawer-line">LTIR pool: {formatMoneyM(capDetail.ltir_pool)}</p>
          )}
          {team.needsSummary?.needs_short?.length > 0 && (
            <div className="trade-hub-chip-row">
              {team.needsSummary.needs_short.map((n) => (
                <span key={n} className="trade-hub-chip trade-hub-chip-need">{n}</span>
              ))}
            </div>
          )}
          {picks.length > 0 && (
            <>
              <div className="trade-hub-panel-title">Picks</div>
              {picks.map((p) => (
                <p key={p.id} className="trade-drawer-line">{p.year} {roundLabel(p.round)} · {inferLogoAbbr(p.original_team_id, p.original_team_id)}</p>
              ))}
            </>
          )}
          {youth.length > 0 && (
            <>
              <div className="trade-hub-panel-title">Top Youth</div>
              {youth.map((p) => (
                <p key={p.id} className="trade-drawer-line">{p.name} · {displayOvr(p)} OVR</p>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function resolveTeamCardOvr(team) {
  const base = Number.isFinite(Number(team?.ratings?.overall))
    ? Math.round(Number(team.ratings.overall))
    : null;
  const health = team?.healthAdjustedRating != null && Number.isFinite(Number(team.healthAdjustedRating))
    ? Math.round(Number(team.healthAdjustedRating))
    : null;
  if (base == null) return health;
  if (health == null || health > base + 2) return base;
  return health;
}

function TeamOvrRing({ value }) {
  const n = Number(value);
  const display = Number.isFinite(n) ? Math.round(n) : null;
  const pct = display != null ? clamp(display, 0, 99) : 0;
  const tone = display != null ? ovrRingTone(display) : "muted";
  return (
    <div
      className={`trade-team-ovr-ring trade-team-ovr-${tone}`}
      style={{ "--ovr-pct": `${pct}%` }}
      title={display != null ? `OVR ${display}` : "OVR —"}
    >
      <div className="trade-team-ovr-ring-fill" />
      <div className="trade-team-ovr-ring-inner">
        <span className="trade-team-ovr-number">{display ?? "—"}</span>
        <span className="trade-team-ovr-label">OVR</span>
      </div>
    </div>
  );
}

function TeamIdentityCard({ team, onClick }) {
  if (!team) return null;
  const ovrDisplay = resolveTeamCardOvr(team);
  const po = team.playoffOdds;
  const hasPo = po != null && Number.isFinite(Number(po));
  const poPct = hasPo ? clamp(Math.round(Number(po)), 0, 100) : 0;
  const status = team.statusLabel || "—";

  return (
    <button type="button" className="trade-team-compact-card" onClick={onClick}>
      <div className="trade-team-logo-lifted">
        <TradeLogo team={team} size={82} />
      </div>
      <div className="trade-team-mainline">
        <div className="trade-team-abbr-big">{team.abbr || team.name}</div>
        <div className="trade-team-meta-strip">
          <TeamOvrRing value={ovrDisplay} />
          <div className="trade-team-meta-col">
            <span className={`trade-team-cap-mini ${team.capSpace >= 0 ? "ok" : "bad"}`}>
              {team.capSpace != null ? formatCapCompact(team.capSpace) : "—"}
            </span>
            <span className="trade-team-status-pill">{status}</span>
            <span className="trade-team-po-pill" title={hasPo ? `${poPct}% playoff odds` : "Playoff odds unavailable"}>
              {hasPo ? `${poPct}% PO` : "PO —"}
              {hasPo && <span className="trade-team-po-meter" style={{ width: `${poPct}%` }} />}
            </span>
          </div>
        </div>
      </div>
    </button>
  );
}

function IntelListRow({ label, value, tone }) {
  const display =
    value == null || value === "" || (typeof value === "number" && !Number.isFinite(value))
      ? "—"
      : value;
  return (
    <div className={`trade-intel-row ${tone || ""}`}>
      <span className="trade-intel-label">{label}</span>
      <strong className="trade-intel-value">{display}</strong>
    </div>
  );
}

function TeamIntelDashboard({
  team,
  onViewPlayers,
  onTeamDetail,
  isActive,
  projectedCapSpace = null,
}) {
  if (!team) return null;
  const r = team.ratings || {};
  const baseCap = Number(team.capSpace);
  const liveCap = Number.isFinite(Number(projectedCapSpace)) ? Number(projectedCapSpace) : baseCap;
  const hasLiveCap = Number.isFinite(liveCap);
  const capChanged =
    Number.isFinite(baseCap) && Number.isFinite(liveCap) && Math.abs(liveCap - baseCap) >= 0.005;
  const cap = hasLiveCap ? formatCapCompact(liveCap) : null;
  const capTone = !hasLiveCap ? "" : liveCap < 0 ? "bad" : capChanged ? (liveCap > baseCap ? "ok" : "warn") : "ok";
  const playoff =
    team.playoffOdds != null && Number.isFinite(Number(team.playoffOdds))
      ? `${team.playoffOdds}%`
      : null;
  const status = sidebarStatusLabel(team);
  const topNeed = safeArray(team.needsSummary?.needs_short || team.needsSummary?.needs)[0] || null;

  // Keep the desk readable — deep intel lives in Details / View Players.
  const rows = [
    { label: "OVR", value: r.overall, tone: "gold" },
    { label: "REC", value: team.record },
    {
      label: "SPACE",
      value: capChanged
        ? `${cap} (${liveCap >= baseCap ? "+" : ""}${formatCapCompact(liveCap - baseCap)})`
        : cap,
      tone: capTone,
    },
    { label: "STATUS", value: status },
    playoff ? { label: "PLAYOFF", value: playoff } : null,
    topNeed ? { label: "NEED", value: String(topNeed).toUpperCase() } : null,
  ].filter(Boolean);

  return (
    <div className={`trade-team-intel ${isActive ? "is-active" : ""}`}>
      <button
        type="button"
        className="trade-intel-hero"
        onClick={() => onViewPlayers?.()}
        title="View roster assets"
      >
        <TradeLogo team={team} size={68} />
        <div className="trade-intel-hero-text">
          <strong>{team.abbr || team.name}</strong>
          <span className="trade-intel-status-badge">{status}</span>
          {hasLiveCap ? (
            <span className={`trade-intel-hero-cap ${capTone}`}>
              Space {cap}
              {topNeed ? ` · Need ${String(topNeed)}` : ""}
            </span>
          ) : null}
        </div>
      </button>

      <div className="trade-intel-list trade-intel-list-lean">
        {rows.map((row) => (
          <IntelListRow key={row.label} label={row.label} value={row.value} tone={row.tone} />
        ))}
      </div>

      <div className="trade-intel-foot">
        <button type="button" className="trade-intel-view-players" onClick={() => onViewPlayers?.()}>
          View Players
        </button>
        <button type="button" className="trade-intel-detail-link" onClick={() => onTeamDetail?.(team)}>
          Details
        </button>
      </div>
    </div>
  );
}

function TeamPlayersDrawer({
  team,
  side,
  meta,
  usedIds,
  onClose,
  onDragStart,
  onQuickAdd,
  onAssetClick,
}) {
  if (!team) return null;
  const cap =
    team.capSpace != null && Number.isFinite(Number(team.capSpace))
      ? formatCapCompact(team.capSpace)
      : "—";
  const po =
    team.playoffOdds != null && Number.isFinite(Number(team.playoffOdds))
      ? `${team.playoffOdds}%`
      : "—";
  const status = sidebarStatusLabel(team);

  return (
    <div className="trade-drawer-overlay trade-players-overlay" onClick={onClose}>
      <div
        className="trade-drawer trade-drawer-players trade-players-fullscreen"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="trade-drawer-head trade-players-full-head">
          <TradeLogo team={team} size={72} />
          <div className="trade-players-header-main">
            <strong>{team.name}</strong>
            <span>{team.abbr} · Roster Assets</span>
            <div className="trade-players-intel-strip">
              <span className="trade-players-intel-pill"><span>REC</span> {team.record || "—"}</span>
              <span className="trade-players-intel-pill"><span>SPACE</span> {cap}</span>
              <span className="trade-players-intel-pill"><span>PO</span> {po}</span>
              <span className="trade-players-intel-pill"><span>STATUS</span> {status}</span>
            </div>
          </div>
          <button type="button" className="trade-drawer-close" onClick={onClose}>×</button>
        </div>
        <div className="trade-drawer-body trade-drawer-players-body trade-players-full-body">
          <AssetPool
            teamId={team.id}
            side={side}
            meta={meta}
            usedIds={usedIds}
            onDragStart={onDragStart}
            onQuickAdd={onQuickAdd}
            onAssetClick={onAssetClick}
          />
        </div>
      </div>
    </div>
  );
}

function TeamBrowserPanel({
  team,
  side,
  meta,
  usedIds,
  evaluation,
  onTeamClick,
  onViewPlayers,
  onAssetClick,
  onDragStart,
  onQuickAdd,
  isActive,
  projectedCapSpace = null,
}) {
  if (!team) return null;

  return (
    <aside className={`trade-team-intel-panel trade-team-panel-${side} ${isActive ? "is-active" : ""}`}>
      <TeamIntelDashboard
        team={team}
        onViewPlayers={onViewPlayers}
        onTeamDetail={onTeamClick}
        isActive={isActive}
        projectedCapSpace={projectedCapSpace}
      />
    </aside>
  );
}

function classifyTradeSubmitError(message) {
  const raw = String(message || "").toLowerCase();

  if (
    raw.includes("pick registry") ||
    raw.includes("draft pick") ||
    raw.includes("unresolved pick") ||
    raw.includes("owned_pick") ||
    raw.includes("current_owner")
  ) {
    return {
      label: "PICK ERROR",
      detail: "Draft pick ownership failed.",
      severity: "technical",
    };
  }

  if (raw.includes("cap") || raw.includes("salary")) {
    return {
      label: "CAP BLOCKED",
      detail: "Salary rules blocked it.",
      severity: "blocked",
    };
  }

  if (
    raw.includes("clause") ||
    raw.includes("ntc") ||
    raw.includes("nmc") ||
    raw.includes("no-trade") ||
    raw.includes("no movement")
  ) {
    return {
      label: "CLAUSE BLOCKED",
      detail: "Contract clause blocked it.",
      severity: "blocked",
    };
  }

  if (raw.includes("slot") || raw.includes("contract")) {
    return {
      label: "SLOT BLOCKED",
      detail: "Contract slots blocked it.",
      severity: "blocked",
    };
  }

  return {
    label: "TRADE FAILED",
    detail: "Server rejected trade.",
    severity: "rejected",
  };
}

function shortProposeReason(evaluation, bothSidesHaveAssets, evaluating, protectedConflict = []) {
  if (!bothSidesHaveAssets) return "Add assets to both sides";
  if (protectedConflict.length) return `Blocked: ${playerLastName(protectedConflict[0])} untouchable`;
  if (evaluating) return "Checking…";
  if (!evaluation) return "";

  const detail = evaluation?.trade_review?.block_detail || evaluation?.block_detail;
  if (detail?.message && (!evaluation.accepted || !evaluation.can_execute)) {
    return detail.message;
  }

  const reasons = safeArray(evaluation.rejection_reasons);
  if (reasons[0]) return String(reasons[0]);

  const blob = reasons.join(" ").toLowerCase();
  if (blob.includes("pick")) return "Pick ownership problem";
  if (blob.includes("cap")) return "Cap problem";
  if (blob.includes("clause")) return "Clause blocked";
  if (blob.includes("slot")) return "Roster slot risk";
  if (!evaluation.can_execute) return "Rules blocked this trade";
  if (evaluation.accepted === false) return "Partner needs a better return";
  if (evaluation.accepted) return "Ready";
  return "Pending";
}

function resolveDeskBlockDetail({
  evaluation,
  hasProposed,
  bothSidesHaveAssets,
  protectedConflict,
  softProtectedConflict,
  partnerTeam,
  userTeam,
  userOutgoing,
  partnerOutgoing,
}) {
  if (protectedConflict.length) {
    return {
      code: "PROTECTED",
      message: `${partnerTeam?.abbr || "Partner"} will not trade ${playerLastName(protectedConflict[0])}.`,
      unblock_hint: "Remove the untouchable from the package.",
      tone: "bad",
      badge: "BLOCKED",
    };
  }

  if (!bothSidesHaveAssets) {
    const missing =
      userOutgoing.length === 0 && partnerOutgoing.length === 0
        ? "both packages"
        : userOutgoing.length === 0
          ? "your package"
          : "the partner package";
    return {
      code: "INCOMPLETE",
      message: `Add assets to ${missing}.`,
      unblock_hint: "",
      tone: "warn",
      badge: null,
    };
  }

  if (hasProposed && evaluation) {
    const detail = evaluation.trade_review?.block_detail || evaluation.block_detail;
    if (detail?.message && (!evaluation.accepted || !evaluation.can_execute)) {
      return {
        code: detail.code || "BLOCKED",
        message: detail.message,
        unblock_hint: detail.unblock_hint || "",
        tone: tradeOutcomeTone(evaluation, true) === "warn" ? "warn" : "bad",
        badge: tradeOutcomeLabel(evaluation, true),
      };
    }
    if (evaluation.accepted && evaluation.can_execute) {
      return {
        code: "ACCEPTED",
        message: "Deal is ready to execute.",
        unblock_hint: "",
        tone: "good",
        badge: "ACCEPTED",
      };
    }
  }

  if (softProtectedConflict.length) {
    return {
      code: "CORE",
      message: `${playerLastName(softProtectedConflict[0])} is core — partner may reject.`,
      unblock_hint: "Swap for a lesser ask or add a sweetener.",
      tone: "warn",
      badge: "RISK",
    };
  }

  const partnerAfter = projectedTeamCapSpace(partnerTeam, partnerOutgoing, userOutgoing);
  if (Number.isFinite(partnerAfter) && partnerAfter < -0.05) {
    return {
      code: "CAP",
      message: `${partnerTeam?.abbr || "Partner"} would be −${formatMoneyShort(Math.abs(partnerAfter))} under.`,
      unblock_hint: "Retain salary or move salary the other way.",
      tone: "bad",
      badge: "BLOCKED",
    };
  }

  const userAfter = projectedTeamCapSpace(userTeam, userOutgoing, partnerOutgoing);
  if (Number.isFinite(userAfter) && userAfter < -0.05) {
    return {
      code: "CAP",
      message: `You would be −${formatMoneyShort(Math.abs(userAfter))} under.`,
      unblock_hint: "Retain salary or move salary out.",
      tone: "bad",
      badge: "BLOCKED",
    };
  }

  return null;
}

function buildTradeDecisionToast({
  accepted,
  evaluation,
  userOutgoing,
  fanMeter,
  fanReaction,
  errorMessage,
  partnerAbbr,
}) {
  const fan = fanReaction || (typeof fanMeter === "object" ? fanMeter : null);
  const fanScore = Number(fan?.score ?? fanMeter ?? 50);
  const fanFx = fan?.effects || evaluation?.fan_reaction?.fan_effects || null;
  const fanHeatLabel = fan?.heatLabel || evaluation?.fan_reaction?.fan_heat_label || "";
  const userBd = evaluation?.asset_breakdown?.user || {};
  const userNet = Number(userBd.net) || 0;
  const reasonsText = safeArray(evaluation?.rejection_reasons)
    .join(" ")
    .toLowerCase();
  const rawError = String(errorMessage || "").toLowerCase();
  const allReasons = `${reasonsText} ${rawError}`;

  if (accepted) {
    const partner = partnerAbbr || "partner";
    if (userNet >= 12) {
      return {
        type: "accepted",
        severity: "steal",
        title: "TRADE COMPLETE",
        badge: "FLEECE",
        message: `Deal with ${partner} is locked in — you won this one.`,
        modal: true,
      };
    }

    if (fanScore < 40) {
      const fx = fanEffectsSummary(fanFx);
      return {
        type: "accepted",
        severity: "risky",
        title: "TRADE COMPLETE",
        badge: "FAN RISK",
        message: fan?.summary || `Deal with ${partner} is done, but fans are restless.`,
        fanHeatLabel: fanHeatLabel || "Backlash",
        fanEffects: fx,
        modal: true,
      };
    }

    if (Math.abs(userNet) <= 4) {
      return {
        type: "accepted",
        severity: "fair",
        title: "TRADE COMPLETE",
        badge: "FAIR DEAL",
        message: `Deal with ${partner} is done.`,
        modal: true,
      };
    }

    return {
      type: "accepted",
      severity: "swing",
      title: "TRADE COMPLETE",
      badge: "BIG SWING",
      message: `Bold deal with ${partner} is complete.`,
      modal: true,
    };
  }

  if (!accepted) {
    const classified = classifyTradeSubmitError(errorMessage);

    if (classified.label === "PICK ERROR") {
      return {
        type: "rejected",
        severity: "technical",
        title: "FAILED",
        badge: "PICK ERROR",
        message: "Backend pick bug.",
      };
    }
  }

  if (allReasons.includes("cap") || allReasons.includes("salary")) {
    return {
      type: "rejected",
      severity: "blocked",
      title: "REJECTED",
      badge: "CAP BLOCK",
      message: "Clear space.",
    };
  }

  if (
    allReasons.includes("clause") ||
    allReasons.includes("ntc") ||
    allReasons.includes("nmc") ||
    allReasons.includes("no-trade") ||
    allReasons.includes("no-movement")
  ) {
    return {
      type: "rejected",
      severity: "blocked",
      title: "REJECTED",
      badge: "CLAUSE BLOCK",
      message: "Player declined.",
    };
  }

  if (evaluation?.can_execute && evaluation?.accepted === false && userNet > -8) {
    return {
      type: "rejected",
      severity: "close",
      title: "REJECTED",
      badge: "CLOSE",
      message: "Add sweetener.",
    };
  }

  return {
    type: "rejected",
    severity: "lowball",
    title: "REJECTED",
    badge: "LOWBALL",
    message: "Not enough.",
  };
}

export default function TradeHub() {
  const { setScreen, franchiseState, setFranchiseState } = useGameUI();

  const [tradeAssets, setTradeAssets] = useState(null);
  const [tradeMarket, setTradeMarket] = useState(null);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [assetsLoading, setAssetsLoading] = useState(true);
  const [assetsError, setAssetsError] = useState(false);

  const [partnerId, setPartnerId] = useState("");
  const [leftAssets, setLeftAssets] = useState(emptySlots());
  const [rightAssets, setRightAssets] = useState(emptySlots());

  const [evaluation, setEvaluation] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState("idle");

  const [retMenu, setRetMenu] = useState(null);
  const [toast, setToast] = useState("");
  const [decisionToast, setDecisionToast] = useState(null);
  const [dropTarget, setDropTarget] = useState(null);

  const [selectedAsset, setSelectedAsset] = useState(null);
  const [selectedAssetSide, setSelectedAssetSide] = useState(null);
  const [selectedAssetTeamId, setSelectedAssetTeamId] = useState(null);
  const [detailAsset, setDetailAsset] = useState(null);
  const [activeDetailTab, setActiveDetailTab] = useState("Value");
  const [selectedTeamDetail, setSelectedTeamDetail] = useState(null);
  const [selectedTradeReview, setSelectedTradeReview] = useState(false);
  const [hasProposed, setHasProposed] = useState(false);
  const [comparedAssets, setComparedAssets] = useState([]);
  const [teamPlayersMenu, setTeamPlayersMenu] = useState(null);
  const [waiveBusy, setWaiveBusy] = useState(false);
  const [waiveResult, setWaiveResult] = useState(null);

  const decisionToastTimer = useRef(null);

  const meta = useMemo(
    () => buildTeamsMeta(franchiseState, tradeAssets || franchiseState?.trade_assets, tradeMarket),
    [franchiseState, tradeAssets, tradeMarket],
  );

  const userTeam = useMemo(
    () => meta?.teams?.find((t) => String(t.id) === String(meta.userTeamId)) || null,
    [meta],
  );

  const partnerTeam = useMemo(
    () => meta?.teams?.find((t) => String(t.id) === String(partnerId)) || null,
    [meta, partnerId],
  );

  const partnerOptions = useMemo(() => {
    if (!meta?.teams) return [];
    return meta.teams.filter((t) => String(t.id) !== String(meta.userTeamId));
  }, [meta]);

  useEffect(() => {
    return () => {
      if (decisionToastTimer.current) {
        clearTimeout(decisionToastTimer.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!franchiseState?.session_id && !franchiseState?.team?.id) {
      setAssetsLoading(false);
      return;
    }
    let cancelled = false;
    setAssetsLoading(true);
    setAssetsError(false);
    Promise.all([
      getTradeAssets().catch(() => franchiseState?.trade_assets || null),
      getTradeMarket().catch(() => null),
      getTradeHistory({ limit: 20 }).catch(() => ({ history: [] })),
    ])
      .then(([assets, market, historyRes]) => {
        if (cancelled) return;
        if (assets) setTradeAssets(assets);
        else {
          setAssetsError(true);
          if (franchiseState?.trade_assets) setTradeAssets(franchiseState.trade_assets);
        }
        if (market) setTradeMarket(market);
        setTradeHistory(safeArray(historyRes?.history));
      })
      .catch(() => {
        if (!cancelled) {
          setAssetsError(true);
          if (franchiseState?.trade_assets) setTradeAssets(franchiseState.trade_assets);
        }
      })
      .finally(() => {
        if (!cancelled) setAssetsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [franchiseState?.session_id, franchiseState?.team?.id]);

  useEffect(() => {
    if (!partnerOptions.length) {
      setPartnerId("");
      return;
    }
    setPartnerId((prev) => {
      if (prev && partnerOptions.some((t) => String(t.id) === String(prev))) return prev;
      return String(partnerOptions[0].id);
    });
  }, [partnerOptions.map((t) => t.id).join("|")]);

  useEffect(() => {
    setRightAssets(emptySlots());
    setEvaluation(null);
    setSubmitStatus("idle");
    setHasProposed(false);
    setSelectedTradeReview(false);
  }, [partnerId]);

  const packageSignature = useMemo(
    () => JSON.stringify({ left: leftAssets, right: rightAssets }),
    [leftAssets, rightAssets],
  );

  useEffect(() => {
    setEvaluation(null);
    setEvaluating(false);
    setSubmitStatus("idle");
    setSelectedTradeReview(false);
    setHasProposed(false);
  }, [packageSignature]);

  const userOutgoing = useMemo(() => leftAssets.filter(Boolean), [leftAssets]);
  const partnerOutgoing = useMemo(() => rightAssets.filter(Boolean), [rightAssets]);
  const bothSidesHaveAssets = userOutgoing.length > 0 && partnerOutgoing.length > 0;

  const fanMeter = useMemo(
    () =>
      resolveFanReaction({
        userTeam,
        userOutgoing,
        evaluation: hasProposed ? evaluation : null,
        franchiseState,
        hasProposed,
        partnerTeam,
      }),
    [userTeam, partnerTeam, userOutgoing, evaluation, franchiseState, hasProposed],
  );

  const assetsPayload = useMemo(() => {
    if (!meta?.userTeamId || !partnerId || !bothSidesHaveAssets) return null;
    return buildAssetsByTeam(meta.userTeamId, partnerId, userOutgoing, partnerOutgoing);
  }, [meta?.userTeamId, partnerId, userOutgoing, partnerOutgoing, bothSidesHaveAssets]);

  const addToSide = useCallback(
    (side, slotIndex, asset) => {
      const setter = side === "left" ? setLeftAssets : setRightAssets;
      const current = side === "left" ? leftAssets : rightAssets;
      const filled = current.filter(Boolean).length;
      if (filled >= SLOTS && !current[slotIndex]) {
        setToast("PACKAGE FULL");
        setTimeout(() => setToast(""), 1800);
        return;
      }
      const dup = current.some(
        (a) => a && a.type === asset.type && String(a.id) === String(asset.id),
      );
      if (dup) return;

      setter((prev) => {
        const next = [...prev];
        next[slotIndex] = asset;
        return next;
      });
    },
    [leftAssets, rightAssets],
  );

  const removeFromSide = useCallback((side, slotIndex) => {
    const setter = side === "left" ? setLeftAssets : setRightAssets;
    setter((prev) => {
      const next = [...prev];
      next[slotIndex] = null;
      return next;
    });
  }, []);

  const setRetained = useCallback((side, slotIndex, pct) => {
    const setter = side === "left" ? setLeftAssets : setRightAssets;
    setter((prev) => {
      const next = [...prev];
      const a = next[slotIndex];
      if (!a || a.type !== "player") return prev;
      next[slotIndex] = { ...a, retained_pct: pct };
      return next;
    });
    setRetMenu(null);
  }, []);

  const leftUsedIds = useMemo(
    () => new Set(leftAssets.filter(Boolean).map((a) => `${a.type}-${a.id}`)),
    [leftAssets],
  );
  const rightUsedIds = useMemo(
    () => new Set(rightAssets.filter(Boolean).map((a) => `${a.type}-${a.id}`)),
    [rightAssets],
  );

  const prepareAssetForSide = useCallback((item, side, teamId) => {
    if (item.type === "prospect" && item.tradeable === false) return null;
    if (item.type === "pick") {
      return { ...item, type: "pick", teamId };
    }
    const clause = String(item.protection || item.clauseLabel || "").toUpperCase();
    const isNmc = clause.includes("NMC");
    const needsWaive =
      (item.requiresNtcWaive || clause.includes("NTC")) && !item.ntcWaived && !isNmc;
    if (isNmc) return null;
    if (needsWaive || item.tradeable === false) return null;
    return {
      ...item,
      type: "player",
      teamId,
      retained_pct: item.retained_pct || 0,
      ntcWaived: Boolean(item.ntcWaived),
    };
  }, []);

  const patchPlayerNtcDecision = useCallback((sourceTeamId, playerId, decision) => {
    const accepted = Boolean(decision?.accepted);
    const reason = String(decision?.reason || "");
    const penalty = Number(decision?.value_penalty_pct || 0.08);
    setTradeAssets((prev) => {
      if (!prev?.teams) return prev;
      const tid = String(sourceTeamId);
      const team = prev.teams[tid];
      if (!team?.players?.[String(playerId)]) return prev;
      const prevPlayer = team.players[String(playerId)];
      const nextVal =
        accepted && Number.isFinite(Number(prevPlayer.trade_value))
          ? Math.max(1, Math.round(Number(prevPlayer.trade_value) * (1 - penalty)))
          : prevPlayer.trade_value;
      return {
        ...prev,
        teams: {
          ...prev.teams,
          [tid]: {
            ...team,
            players: {
              ...team.players,
              [String(playerId)]: {
                ...prevPlayer,
                tradeable: accepted ? true : prevPlayer.tradeable,
                trade_block_reason: accepted ? "" : prevPlayer.trade_block_reason || decision?.value_note || "",
                requires_ntc_waive: !accepted,
                ntc_waived: accepted,
                ntc_waiver_reason: reason,
                trade_value: nextVal,
                explain: accepted
                  ? [...safeArray(prevPlayer.explain), "NTC waived — slightly reduced trade value"].slice(0, 6)
                  : prevPlayer.explain,
                risk_flags: accepted
                  ? [...safeArray(prevPlayer.risk_flags), "NTC waived — slightly reduced trade value"].slice(0, 6)
                  : prevPlayer.risk_flags,
              },
            },
          },
        },
      };
    });
    const patchAsset = (asset) => {
      if (!asset || String(asset.id) !== String(playerId)) return asset;
      const nextVal =
        accepted && Number.isFinite(Number(asset.tradeValue))
          ? Math.max(1, Math.round(Number(asset.tradeValue) * (1 - penalty)))
          : asset.tradeValue;
      return {
        ...asset,
        tradeable: accepted ? true : asset.tradeable,
        tradeBlockReason: accepted ? "" : asset.tradeBlockReason || reason,
        requiresNtcWaive: !accepted,
        ntcWaived: accepted,
        ntcWaiverReason: reason,
        tradeValue: nextVal,
        tradeRiskFlags: accepted
          ? [...safeArray(asset.tradeRiskFlags), "NTC waived — slightly reduced trade value"].slice(0, 6)
          : asset.tradeRiskFlags,
      };
    };
    setSelectedAsset((prev) => patchAsset(prev));
    setLeftAssets((prev) => prev.map(patchAsset));
    setRightAssets((prev) => prev.map(patchAsset));
  }, []);

  const handleAskNtcWaive = useCallback(async () => {
    if (!selectedAsset || selectedAsset.type === "pick" || waiveBusy) return;
    const sourceTeamId = String(selectedAsset.teamId || selectedAssetTeamId || "");
    const userId = String(meta?.userTeamId || "");
    const partner = String(partnerId || "");
    if (!sourceTeamId || !userId || !partner) {
      setToast("Pick a trade partner first");
      setTimeout(() => setToast(""), 1800);
      return;
    }
    const destinationTeamId = sourceTeamId === userId ? partner : userId;
    setWaiveBusy(true);
    try {
      const res = await requestNtcWaive({
        player_id: String(selectedAsset.id),
        source_team_id: sourceTeamId,
        destination_team_id: destinationTeamId,
      });
      const decision = res?.decision || res || {};
      patchPlayerNtcDecision(sourceTeamId, selectedAsset.id, decision);
      setWaiveResult({
        playerName: selectedAsset.name,
        accepted: Boolean(decision.accepted),
        reason: String(decision.reason || decision.value_note || ""),
        chance: decision.accept_chance,
        valueNote: decision.value_note,
      });
    } catch (e) {
      setToast(e?.response?.data?.detail || e?.message || "Waiver request failed");
      setTimeout(() => setToast(""), 2200);
    } finally {
      setWaiveBusy(false);
    }
  }, [
    selectedAsset,
    selectedAssetTeamId,
    waiveBusy,
    meta?.userTeamId,
    partnerId,
    patchPlayerNtcDecision,
  ]);

  const findOpenSlot = useCallback((side) => {
    const current = side === "left" ? leftAssets : rightAssets;
    const idx = current.findIndex((a) => !a);
    return idx >= 0 ? idx : null;
  }, [leftAssets, rightAssets]);

  const handleDragStart = useCallback((event, item, source, side, slotIndex, teamId) => {
    event.dataTransfer.setData(
      DRAG_MIME,
      JSON.stringify({ item, source, side, slotIndex, teamId }),
    );
    event.dataTransfer.effectAllowed = "move";
  }, []);

  const handleDropOnSlot = useCallback(
    (event, side, slotIndex, teamId) => {
      event.preventDefault();
      setDropTarget(null);
      const payload = parseDragPayload(event);
      if (!payload?.item) return;

      const { item, source, side: fromSide, slotIndex: fromSlot } = payload;
      const prepared = prepareAssetForSide(item, side, teamId);
      if (!prepared) {
        const clause = String(item.protection || item.clauseLabel || "").toUpperCase();
        const needsWaive =
          (item.requiresNtcWaive || clause.includes("NTC")) && !item.ntcWaived && !clause.includes("NMC");
        setToast(
          needsWaive
            ? "Ask the player to waive their NTC first"
            : item.tradeBlockReason || "Asset is not tradeable",
        );
        setTimeout(() => setToast(""), 2000);
        return;
      }

      if (source === "slot" && fromSide === side && fromSlot === slotIndex) return;

      if (source === "slot" && fromSide === side) {
        const setter = side === "left" ? setLeftAssets : setRightAssets;
        setter((prev) => {
          const next = [...prev];
          const moving = next[fromSlot];
          next[fromSlot] = next[slotIndex];
          next[slotIndex] = moving;
          return next;
        });
        return;
      }

      if (source === "slot" && fromSide !== side) {
        setToast("Move asset within its team package");
        setTimeout(() => setToast(""), 1800);
        return;
      }

      addToSide(side, slotIndex, prepared);
    },
    [addToSide, prepareAssetForSide],
  );

  const handleQuickAdd = useCallback(
    (side, teamId, item) => {
      const slotIndex = findOpenSlot(side);
      if (slotIndex == null) {
        setToast("PACKAGE FULL");
        setTimeout(() => setToast(""), 1800);
        return;
      }
      const prepared = prepareAssetForSide(item, side, teamId);
      if (!prepared) return;
      addToSide(side, slotIndex, prepared);
      setSelectedAsset(null);
      setToast("ADDED TO PACKAGE");
      setTimeout(() => setToast(""), 1400);
    },
    [addToSide, findOpenSlot, prepareAssetForSide],
  );

  const handleAssetClick = useCallback((item, side, teamId) => {
    setSelectedAsset(item);
    setSelectedAssetSide(side);
    setSelectedAssetTeamId(teamId);
  }, []);

  const handleResetPackage = useCallback(() => {
    setLeftAssets(emptySlots());
    setRightAssets(emptySlots());
    setEvaluation(null);
    setSubmitStatus("idle");
    setHasProposed(false);
    setSelectedTradeReview(false);
    setSelectedAsset(null);
    setDetailAsset(null);
  }, []);

  const packageLoc = useMemo(
    () => findAssetInPackage(selectedAsset, leftAssets, rightAssets),
    [selectedAsset, leftAssets, rightAssets],
  );

  const isUserAssetSide = selectedAssetSide === "left";

  const partnerProtection = useMemo(
    () => partnerProtectionLists(meta, partnerId),
    [meta, partnerId],
  );

  const partnerProtectedNames = partnerProtection.displayNames;

  const protectedConflict = useMemo(
    () => partnerHardProtectedConflict(partnerOutgoing, partnerProtection.hardNames),
    [partnerOutgoing, partnerProtection.hardNames],
  );

  const softProtectedConflict = useMemo(
    () => partnerSoftProtectedConflict(partnerOutgoing, partnerProtection.softNames),
    [partnerOutgoing, partnerProtection.softNames],
  );

  const protectedWarningForAsset = useCallback((asset) => {
    if (!asset || asset.type !== "player") return "";
    const name = String(asset.name).toLowerCase();
    if (protectedConflict.some((n) => String(n).toLowerCase() === name)) {
      return "Protected — cannot acquire";
    }
    if (softProtectedConflict.some((n) => String(n).toLowerCase() === name)) {
      return "Core piece — partner may reject";
    }
    return "";
  }, [protectedConflict, softProtectedConflict]);

  const removeIncomingByName = useCallback((name) => {
    const target = String(name).toLowerCase();
    setRightAssets((prev) => prev.map((a) => (
      a?.type === "player" && String(a.name).toLowerCase() === target ? null : a
    )));
  }, []);

  const removeOutgoingByName = useCallback((name) => {
    const target = String(name).toLowerCase();
    setLeftAssets((prev) => prev.map((a) => (
      a?.type === "player" && String(a.name).toLowerCase() === target ? null : a
    )));
  }, []);

  const proposeDisabled =
    !bothSidesHaveAssets ||
    !partnerId ||
    !assetsPayload ||
    submitting ||
    protectedConflict.length > 0;

  const proposeSoftBlocked =
    !proposeDisabled &&
    (softProtectedConflict.length > 0 ||
      (() => {
        const partnerAfter = projectedTeamCapSpace(partnerTeam, partnerOutgoing, userOutgoing);
        const userAfter = projectedTeamCapSpace(userTeam, userOutgoing, partnerOutgoing);
        return (
          (Number.isFinite(partnerAfter) && partnerAfter < -0.05) ||
          (Number.isFinite(userAfter) && userAfter < -0.05)
        );
      })());

  const deskBlockDetail = useMemo(
    () =>
      resolveDeskBlockDetail({
        evaluation,
        hasProposed,
        bothSidesHaveAssets,
        protectedConflict,
        softProtectedConflict,
        partnerTeam,
        userTeam,
        userOutgoing,
        partnerOutgoing,
      }),
    [
      evaluation,
      hasProposed,
      bothSidesHaveAssets,
      protectedConflict,
      softProtectedConflict,
      partnerTeam,
      userTeam,
      userOutgoing,
      partnerOutgoing,
    ],
  );

  const proposeLabel = submitting ? "PROPOSING…" : "PROPOSE TRADE";

  const shortReason =
    deskBlockDetail?.message ||
    (hasProposed
      ? shortProposeReason(evaluation, bothSidesHaveAssets, evaluating, protectedConflict)
      : "");
  const unblockHint = deskBlockDetail?.unblock_hint || "";
  const proposeOutcomeBadge = hasProposed && evaluation
    ? tradeOutcomeLabel(evaluation, bothSidesHaveAssets)
    : deskBlockDetail?.badge || "";
  const proposeOutcomeTone = hasProposed && evaluation
    ? tradeOutcomeTone(evaluation, bothSidesHaveAssets)
    : deskBlockDetail?.tone || "neutral";

  const showDecisionToast = useCallback((payload) => {
    if (decisionToastTimer.current) {
      clearTimeout(decisionToastTimer.current);
    }

    setDecisionToast(payload);

    const holdMs = payload?.modal || payload?.type === "accepted" ? 7000 : 4000;
    decisionToastTimer.current = setTimeout(() => {
      setDecisionToast(null);
    }, holdMs);
  }, []);

  const dismissDecisionToast = useCallback(() => {
    if (decisionToastTimer.current) {
      clearTimeout(decisionToastTimer.current);
      decisionToastTimer.current = null;
    }
    setDecisionToast(null);
  }, []);

  const assetToastLabel = (asset) => {
    if (!asset) return "";
    if (asset.type === "pick") return `${asset.year} ${roundLabel(asset.round)}`;
    return playerLastName(asset.name) || asset.name || "Player";
  };

  const handlePropose = useCallback(async () => {
    if (!assetsPayload || !bothSidesHaveAssets || !partnerId || submitting) return;
    setHasProposed(true);
    setSubmitting(true);
    setSubmitStatus("idle");
    setEvaluating(true);

    let ev = null;
    try {
      const evalRes = await evaluateTradePackage({ assets_by_team: assetsPayload });
      ev = evalRes?.evaluation || evalRes;
      setEvaluation(ev);
      setEvaluating(false);

      const reactionScore = resolveFanReaction({
        userTeam,
        userOutgoing,
        evaluation: ev,
        franchiseState,
        hasProposed: true,
        partnerTeam,
      });

      if (!ev?.accepted || !ev?.can_execute) {
        const rejectedToast = buildTradeDecisionToast({
          accepted: false,
          evaluation: ev,
          userOutgoing,
          fanReaction: reactionScore,
          partnerAbbr: partnerTeam?.abbr,
        });
        showDecisionToast(rejectedToast);
        setSubmitStatus("rejected");
        return;
      }

      try {
        const sentLine = userOutgoing.map(assetToastLabel).filter(Boolean).join(" · ");
        const gotLine = partnerOutgoing.map(assetToastLabel).filter(Boolean).join(" · ");

        const res = await submitTradePackage({ assets_by_team: assetsPayload });
        if (res?.state) setFranchiseState(res.state);

        const execFan = res?.trade_result?.fan_reaction || ev?.fan_reaction;
        const acceptedFan = execFan
          ? {
              score: Number(execFan.fan_reaction_score),
              heat: Number(execFan.fan_heat),
              heatLabel: execFan.fan_heat_label,
              category: execFan.fan_category,
              summary: execFan.fan_summary,
              effects: execFan.fan_effects,
              factors: safeArray(execFan.fan_factors).slice(0, 3),
            }
          : reactionScore;

        const acceptedToast = {
          ...buildTradeDecisionToast({
            accepted: true,
            evaluation: { ...ev, fan_reaction: execFan || ev?.fan_reaction },
            userOutgoing,
            fanReaction: acceptedFan,
            partnerAbbr: partnerTeam?.abbr,
          }),
          sentLine,
          gotLine,
          partnerAbbr: partnerTeam?.abbr || "Partner",
          modal: true,
        };

        showDecisionToast(acceptedToast);

        setSubmitStatus("accepted");
        setLeftAssets(emptySlots());
        setRightAssets(emptySlots());
        setEvaluation(null);
        setHasProposed(false);
        setSelectedTradeReview(false);
        Promise.all([getTradeAssets(), getTradeMarket(), getTradeHistory({ limit: 20 })])
          .then(([a, m, h]) => {
            if (a) setTradeAssets(a);
            if (m) setTradeMarket(m);
            if (h?.history) setTradeHistory(safeArray(h.history));
          })
          .catch(() => {});
      } catch (e) {
        const msg = e?.response?.data?.detail || e?.message || "Trade rejected";
        const cleanMsg = typeof msg === "string" ? msg : "Trade rejected";
        const classified = classifyTradeSubmitError(cleanMsg);

        console.warn("Trade submit failed:", cleanMsg);

        const rejectedEvaluation = {
          ...(ev || {}),
          accepted: false,
          can_execute: false,
          rejection_reasons: [classified.label],
          technical_detail: cleanMsg,
        };

        const rejectedToast = buildTradeDecisionToast({
          accepted: false,
          evaluation: rejectedEvaluation,
          userOutgoing,
          fanReaction: reactionScore,
          errorMessage: cleanMsg,
          partnerAbbr: partnerTeam?.abbr,
        });

        showDecisionToast(rejectedToast);

        setSubmitStatus("rejected");
        setEvaluation(rejectedEvaluation);
      }
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || "Evaluation failed";
      const cleanMsg = typeof msg === "string" ? msg : "Evaluation failed";
      const classified = classifyTradeSubmitError(cleanMsg);

      console.warn("Trade evaluation failed:", cleanMsg);

      const rejectedEvaluation = {
        accepted: false,
        can_execute: false,
        rejection_reasons: [classified.label],
        technical_detail: cleanMsg,
      };

      const rejectedToast = buildTradeDecisionToast({
        accepted: false,
        evaluation: rejectedEvaluation,
        userOutgoing,
        fanReaction: fanMeter,
        errorMessage: cleanMsg,
        partnerAbbr: partnerTeam?.abbr,
      });

      showDecisionToast(rejectedToast);
      setSubmitStatus("rejected");
      setEvaluation(rejectedEvaluation);
    } finally {
      setSubmitting(false);
      setEvaluating(false);
    }
  }, [
    assetsPayload,
    bothSidesHaveAssets,
    partnerId,
    submitting,
    setFranchiseState,
    userOutgoing,
    partnerOutgoing,
    fanMeter,
    userTeam,
    franchiseState,
    partnerTeam,
    showDecisionToast,
  ]);

  if (assetsLoading && !meta) {
    return (
      <div className="nhlcal-root trade-hub-root">
        <div className="trade-hub-shell">
          <div className="trade-hub-loading">LOADING TRADE DESK</div>
        </div>
        <style>{TRADE_HUB_CSS}</style>
      </div>
    );
  }

  if (!meta || !userTeam) {
    return (
      <div className="nhlcal-root trade-hub-root">
        <div className="trade-hub-shell">
          <div className="trade-hub-loading trade-hub-empty-card">
            TRADE ASSETS UNAVAILABLE
            <span>Start or refresh franchise state.</span>
            <button type="button" className="trade-hub-back-btn" onClick={() => setScreen(SCREENS.HUB)}>
              RETURN TO HUB
            </button>
          </div>
        </div>
        <style>{TRADE_HUB_CSS}</style>
      </div>
    );
  }

  const leftPadded = padSlots(leftAssets);
  const rightPadded = padSlots(rightAssets);

  const slotDragProps = (side, slotIndex, teamId) => ({
    isDropTarget: dropTarget === `${side}-${slotIndex}`,
    onDragOver: (e) => {
      e.preventDefault();
      setDropTarget(`${side}-${slotIndex}`);
    },
    onDragLeave: () => setDropTarget((v) => (v === `${side}-${slotIndex}` ? null : v)),
    onDrop: (e) => handleDropOnSlot(e, side, slotIndex, teamId),
    onDragStartFromSlot: (e, idx, asset) =>
      handleDragStart(e, asset, "slot", side, idx, teamId),
  });

  return (
    <div className="nhlcal-root trade-hub-root">
      <div className="trade-hub-shell">
      <header className="trade-hub-topbar">
        <button type="button" className="trade-hub-back-btn" onClick={() => setScreen(SCREENS.HUB)}>
          ← HUB
        </button>
        <div className="trade-hub-top-center">
          <div className="trade-hub-screen-title">TRADE CENTRE</div>
        </div>
        <div className="trade-hub-top-right">
          <select
            className="trade-hub-partner-select"
            value={partnerId}
            onChange={(e) => setPartnerId(e.target.value)}
          >
            {!partnerId && <option value="">SELECT TRADE PARTNER</option>}
            {partnerOptions.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      {!partnerId ? (
        <div className="trade-hub-main">
          <div className="trade-hub-loading trade-hub-empty-card">SELECT TRADE PARTNER</div>
        </div>
      ) : (
        <main className="trade-hub-main trade-hub-war-room-layout">
          <TeamBrowserPanel
            team={userTeam}
            side="left"
            evaluation={hasProposed ? evaluation : null}
            meta={meta}
            usedIds={leftUsedIds}
            projectedCapSpace={projectedTeamCapSpace(userTeam, leftAssets, rightAssets)}
            isActive={
              teamPlayersMenu?.teamId === meta.userTeamId ||
              String(selectedTeamDetail?.id) === String(meta.userTeamId)
            }
            onTeamClick={setSelectedTeamDetail}
            onViewPlayers={() => setTeamPlayersMenu({ teamId: meta.userTeamId, side: "left" })}
            onAssetClick={handleAssetClick}
            onDragStart={(e, item, source) =>
              handleDragStart(e, item, source, "left", null, meta.userTeamId)
            }
            onQuickAdd={(item) => handleQuickAdd("left", meta.userTeamId, item)}
          />

          <section className="trade-hub-centre trade-hub-centre-focus">
            <div className="trade-construction-grid">
              <div className="trade-package-col">
                <div className="trade-package-header">
                  <TradeLogo team={userTeam} size={40} />
                  <div className="trade-package-header-main">
                    <span>YOU SEND</span>
                    <PackageCapStrip
                      team={userTeam}
                      outgoingAssets={leftAssets}
                      incomingAssets={rightAssets}
                      sideLabel="Your space"
                    />
                  </div>
                </div>
                {leftPadded.map((asset, i) => (
                  <TradeSlot
                    key={`L-${i}`}
                    slotIndex={i}
                    asset={asset}
                    side="left"
                    teamId={meta.userTeamId}
                    accentHue={userTeam.hue}
                    onRemove={() => removeFromSide("left", i)}
                    onRetained={asset?.type === "player" ? () => setRetMenu({ side: "left", slot: i }) : undefined}
                    {...slotDragProps("left", i, meta.userTeamId)}
                  />
                ))}
              </div>

              <div className="trade-package-divider">
                <span>⇄</span>
              </div>

              <div className="trade-package-col">
                <div className="trade-package-header">
                  <TradeLogo team={partnerTeam} size={40} />
                  <div className="trade-package-header-main">
                    <span>{partnerTeam?.abbr || "PARTNER"} SENDS</span>
                    <PackageCapStrip
                      team={partnerTeam}
                      outgoingAssets={rightAssets}
                      incomingAssets={leftAssets}
                      sideLabel="Partner space"
                    />
                  </div>
                </div>
                {rightPadded.map((asset, i) => (
                  <TradeSlot
                    key={`R-${i}`}
                    slotIndex={i}
                    asset={asset}
                    side="right"
                    teamId={partnerId}
                    accentHue={partnerTeam?.hue || 200}
                    protectedWarning={protectedWarningForAsset(asset)}
                    onRemove={() => removeFromSide("right", i)}
                    onRetained={asset?.type === "player" ? () => setRetMenu({ side: "right", slot: i }) : undefined}
                    {...slotDragProps("right", i, partnerId)}
                  />
                ))}
              </div>
            </div>

            <div className="trade-hub-centre-foot">
            <DynamicTradeAnalysis
              partnerAbbr={partnerTeam?.abbr}
              hasProposed={hasProposed}
              evaluation={hasProposed ? evaluation : null}
              userOutgoing={userOutgoing}
              partnerOutgoing={partnerOutgoing}
              fanMeter={fanMeter}
              fanData={fanMeter}
              onReviewClick={() => setSelectedTradeReview(true)}
              userTeamId={meta?.userTeamId || ""}
              partnerId={partnerId || ""}
            />

            <div className="trade-hub-propose-wrap">
              {proposeOutcomeBadge && !shortReason ? (
                <span className={`trade-outcome-badge trade-outcome-${proposeOutcomeTone}`}>
                  {proposeOutcomeBadge}
                </span>
              ) : null}
              <button
                type="button"
                className={`trade-hub-propose-btn ${submitStatus === "accepted" ? "accepted" : ""} ${submitStatus === "rejected" && hasProposed ? "rejected" : ""} ${proposeSoftBlocked ? "soft-blocked" : ""}`}
                disabled={proposeDisabled}
                onClick={handlePropose}
              >
                {proposeLabel}
              </button>
            </div>

            {shortReason && submitStatus !== "accepted" && (
              <div className={`trade-hub-block-reason trade-hub-warning-reason tone-${deskBlockDetail?.tone || "warn"}`}>
                <strong>{deskBlockDetail?.code === "CAP" || deskBlockDetail?.badge === "BLOCKED" ? "Why blocked" : deskBlockDetail?.badge === "REJECTED" || deskBlockDetail?.badge === "OVERPAY" ? "Why rejected" : "Note"}</strong>
                <span>{shortReason}</span>
                {unblockHint ? <em>What would unblock: {unblockHint}</em> : null}
              </div>
            )}
            </div>

            {assetsError && (
              <div className="trade-hub-warn-inline">Cached trade data</div>
            )}

            {toast && <div className="trade-hub-toast-float">{toast}</div>}
          </section>

          <TeamBrowserPanel
            team={partnerTeam}
            side="right"
            evaluation={hasProposed ? evaluation : null}
            meta={meta}
            usedIds={rightUsedIds}
            projectedCapSpace={projectedTeamCapSpace(partnerTeam, rightAssets, leftAssets)}
            isActive={
              teamPlayersMenu?.teamId === partnerId ||
              String(selectedTeamDetail?.id) === String(partnerId)
            }
            onTeamClick={setSelectedTeamDetail}
            onViewPlayers={() => setTeamPlayersMenu({ teamId: partnerId, side: "right" })}
            onAssetClick={handleAssetClick}
            onDragStart={(e, item, source) =>
              handleDragStart(e, item, source, "right", null, partnerId)
            }
            onQuickAdd={(item) => handleQuickAdd("right", partnerId, item)}
          />
        </main>
      )}

      {teamPlayersMenu && meta && (() => {
        const menuTeam = meta.teams?.find((t) => String(t.id) === String(teamPlayersMenu.teamId));
        const menuUsedIds = teamPlayersMenu.side === "left" ? leftUsedIds : rightUsedIds;
        if (!menuTeam) return null;
        return (
          <TeamPlayersDrawer
            team={menuTeam}
            side={teamPlayersMenu.side}
            meta={meta}
            usedIds={menuUsedIds}
            onClose={() => setTeamPlayersMenu(null)}
            onDragStart={(e, item, source) =>
              handleDragStart(
                e,
                item,
                source,
                teamPlayersMenu.side,
                null,
                teamPlayersMenu.teamId,
              )
            }
            onQuickAdd={(item) =>
              handleQuickAdd(teamPlayersMenu.side, teamPlayersMenu.teamId, item)
            }
            onAssetClick={handleAssetClick}
          />
        );
      })()}

      {selectedAsset && !detailAsset && (
        <AssetContextMenu
          asset={selectedAsset}
          side={selectedAssetSide}
          teamId={selectedAssetTeamId}
          isUserSide={isUserAssetSide}
          inPackage={Boolean(packageLoc)}
          onAdd={() => {
            const side = selectedAssetSide === "left" ? "left" : "right";
            handleQuickAdd(side, selectedAssetTeamId, selectedAsset);
          }}
          onCompare={() => {
            setComparedAssets((prev) => {
              const next = [...prev.filter((a) => a.id !== selectedAsset.id), selectedAsset].slice(-2);
              setToast(next.length >= 2 ? "Compare in Trade Review" : "Pick one more to compare");
              setTimeout(() => setToast(""), 1600);
              return next;
            });
          }}
          onDetails={() => {
            setDetailAsset(selectedAsset);
            setActiveDetailTab("Value");
            setSelectedAsset(null);
          }}
          onRemove={() => {
            if (packageLoc) removeFromSide(packageLoc.side, packageLoc.slotIndex);
            setSelectedAsset(null);
          }}
          onRetain={
            isUserAssetSide && packageLoc && selectedAsset?.type === "player"
              ? () => {
                  setRetMenu({ side: packageLoc.side, slot: packageLoc.slotIndex });
                  setSelectedAsset(null);
                }
              : undefined
          }
          onAskPrice={
            hasProposed && !isUserAssetSide && safeArray(evaluation?.suggested_counteroffers).length
              ? () => {
                  setSelectedTradeReview(true);
                  setSelectedAsset(null);
                }
              : undefined
          }
          onAskNtcWaive={handleAskNtcWaive}
          waiveBusy={waiveBusy}
          onClose={() => setSelectedAsset(null)}
        />
      )}

      {waiveResult && (
        <div className="trade-ctx-overlay" onClick={() => setWaiveResult(null)}>
          <div className="trade-ctx-menu trade-ntc-waive-result" onClick={(e) => e.stopPropagation()}>
            <div className="trade-ctx-head">
              <div>
                <strong>{waiveResult.accepted ? "NTC WAIVED" : "WAIVER DECLINED"}</strong>
                <span>{waiveResult.playerName}</span>
              </div>
            </div>
            <div className={waiveResult.accepted ? "trade-ctx-block trade-ctx-waive-ok" : "trade-ctx-block"}>
              {waiveResult.reason || (waiveResult.accepted ? "Player agreed to waive." : "Player refused to waive.")}
            </div>
            {waiveResult.accepted ? (
              <p className="trade-ntc-waive-note">You can add them to the package. Trade value is slightly reduced.</p>
            ) : (
              <p className="trade-ntc-waive-note">They stay untradeable to this destination unless the situation changes.</p>
            )}
            <div className="trade-ctx-actions">
              {waiveResult.accepted && selectedAsset?.ntcWaived && (
                <button
                  type="button"
                  onClick={() => {
                    handleQuickAdd(selectedAssetSide === "left" ? "left" : "right", selectedAssetTeamId, selectedAsset);
                    setWaiveResult(null);
                    setSelectedAsset(null);
                  }}
                >
                  Add to Package
                </button>
              )}
              <button type="button" className="ghost" onClick={() => setWaiveResult(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {detailAsset && (
        <AssetDetailDrawer
          asset={detailAsset}
          tab={activeDetailTab}
          onTabChange={setActiveDetailTab}
          onClose={() => setDetailAsset(null)}
          evaluation={hasProposed ? evaluation : null}
          userTeamId={meta?.userTeamId}
          partnerTeamId={partnerId}
          partnerTeam={partnerTeam}
          userTeam={userTeam}
          userOutgoing={userOutgoing}
          franchiseState={franchiseState}
          tradeMarket={tradeMarket}
          tradeHistory={tradeHistory}
        />
      )}

      {selectedTradeReview && hasProposed && (
        <TradeReviewDrawer
          evaluation={hasProposed ? evaluation : null}
          userTeamId={meta.userTeamId}
          partnerTeamId={partnerId}
          partnerTeam={partnerTeam}
          userTeam={userTeam}
          hasAssets={bothSidesHaveAssets}
          fanMeter={fanMeter}
          tradeMarket={tradeMarket}
          tradeHistory={tradeHistory}
          comparedAssets={comparedAssets}
          userOutgoing={userOutgoing}
          partnerOutgoing={partnerOutgoing}
          meta={meta}
          onClose={() => setSelectedTradeReview(false)}
          onPropose={handlePropose}
          onReset={handleResetPackage}
          onRemoveIncoming={removeIncomingByName}
          onRemoveOutgoing={removeOutgoingByName}
          onViewTeamNeeds={() => {
            setSelectedTradeReview(false);
            setSelectedTeamDetail(partnerTeam);
          }}
          onAssetClick={(asset) => {
            if (!asset) return;
            setDetailAsset(asset);
            setActiveDetailTab("Contract");
          }}
          proposeDisabled={proposeDisabled}
          proposeLabel={proposeLabel}
          proposeBlockReason={shortReason}
        />
      )}

      {selectedTeamDetail && (
        <TeamDetailDrawer
          team={selectedTeamDetail}
          meta={meta}
          partnerId={partnerId}
          onClose={() => setSelectedTeamDetail(null)}
        />
      )}

      {retMenu && (
        <div className="trade-hub-ret-overlay" onClick={() => setRetMenu(null)}>
          <div className="trade-hub-ret-menu" onClick={(e) => e.stopPropagation()}>
            <div className="trade-hub-ret-title">RETAINED SALARY</div>
            {[0, 25, 50].map((pct) => (
              <button
                key={pct}
                type="button"
                onClick={() => setRetained(retMenu.side, retMenu.slot, pct)}
              >
                {pct}%
              </button>
            ))}
          </div>
        </div>
      )}

      </div>

      {decisionToast ? (
        decisionToast.modal || decisionToast.type === "accepted" ? (
          <div className="trade-success-overlay" onClick={dismissDecisionToast} role="presentation">
            <div
              className={`trade-success-modal trade-decision-${decisionToast.type} trade-decision-${decisionToast.severity}`}
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-label={decisionToast.title || "Trade result"}
            >
              <div className="trade-decision-glow" />
              <div className="trade-decision-kicker">
                <span>{decisionToast.badge || "DONE"}</span>
              </div>
              <strong>{decisionToast.title || "TRADE COMPLETE"}</strong>
              <p>{decisionToast.message}</p>
              {(decisionToast.sentLine || decisionToast.gotLine) && (
                <div className="trade-success-swap">
                  {decisionToast.sentLine ? (
                    <div>
                      <span>You sent</span>
                      <strong>{decisionToast.sentLine}</strong>
                    </div>
                  ) : null}
                  {decisionToast.gotLine ? (
                    <div>
                      <span>You got</span>
                      <strong>{decisionToast.gotLine}</strong>
                    </div>
                  ) : null}
                </div>
              )}
              {decisionToast.fanHeatLabel && (
                <div className="trade-fan-toast-extra">
                  <span>FANBASE REACTION</span>
                  <strong>{decisionToast.fanHeatLabel}</strong>
                  {safeArray(decisionToast.fanEffects).map((line) => (
                    <small key={line}>{line}</small>
                  ))}
                </div>
              )}
              <button type="button" className="trade-success-dismiss" onClick={dismissDecisionToast}>
                Continue
              </button>
            </div>
          </div>
        ) : (
          <div className={`trade-decision-toast trade-decision-${decisionToast.type} trade-decision-${decisionToast.severity}`}>
            <div className="trade-decision-glow" />
            <div className="trade-decision-kicker">
              <span>{decisionToast.badge}</span>
            </div>
            <strong>{decisionToast.title}</strong>
            <p>{decisionToast.message}</p>
          </div>
        )
      ) : null}

      <style>{TRADE_HUB_CSS}</style>
    </div>
  );
}

const TRADE_HUB_CSS = `
.nhlcal-root.trade-hub-root {
  --bg: #04101a;
  --bg-2: #061522;
  --panel: rgba(9, 25, 38, 0.94);
  --panel-2: rgba(12, 35, 52, 0.94);
  --panel-3: rgba(15, 46, 66, 0.78);
  --line: rgba(156, 218, 236, 0.14);
  --line-2: rgba(115, 229, 241, 0.25);
  --line-strong: rgba(73, 231, 240, 0.5);
  --text: #e9f7fb;
  --muted: #8096a8;
  --muted-2: #607789;
  --cyan: #13d8e7;
  --cyan-soft: rgba(19, 216, 231, 0.13);
  --gold: #e9a83c;
  --gold-soft: rgba(233, 168, 60, 0.14);
  --green: #52df94;
  --green-soft: rgba(82, 223, 148, 0.13);
  --red: #ff606d;
  --red-soft: rgba(255, 96, 109, 0.13);
  --blue: #8ab4ff;
  --blue-soft: rgba(138, 180, 255, 0.13);
  --purple: #c992ff;
  --purple-soft: rgba(201, 146, 255, 0.14);
  --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);

  min-height: 100vh;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  color: var(--text);
  background:
    radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.nhlcal-root.trade-hub-root *,
.nhlcal-root.trade-hub-root *::before,
.nhlcal-root.trade-hub-root *::after {
  box-sizing: border-box;
}
.nhlcal-root.trade-hub-root button {
  font-family: inherit;
}
.nhlcal-root.trade-hub-root button:focus-visible,
.nhlcal-root.trade-hub-root select:focus-visible {
  outline: 2px solid var(--line-strong);
  outline-offset: 2px;
}
.trade-hub-shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  height: 100vh;
  overflow: hidden;
}
.trade-hub-main {
  flex: 1;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
  padding: 6px 14px 4px;
}
.trade-hub-main::-webkit-scrollbar,
.trade-pool-list::-webkit-scrollbar,
.trade-hub-picker-list::-webkit-scrollbar {
  width: 10px;
}
.trade-hub-main::-webkit-scrollbar-thumb,
.trade-pool-list::-webkit-scrollbar-thumb,
.trade-hub-picker-list::-webkit-scrollbar-thumb {
  background: rgba(110, 173, 191, 0.25);
  border-radius: 999px;
}
.trade-hub-main::-webkit-scrollbar-thumb:hover,
.trade-pool-list::-webkit-scrollbar-thumb:hover,
.trade-hub-picker-list::-webkit-scrollbar-thumb:hover {
  background: rgba(110, 173, 191, 0.42);
}
.trade-hub-topbar {
  position: relative;
  z-index: 2;
  flex-shrink: 0;
  display: grid;
  grid-template-columns: 96px 1fr minmax(168px, 220px);
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--line);
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  box-shadow: var(--shadow);
}
.trade-hub-back-btn {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--line-2);
  color: var(--text);
  padding: 8px 12px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}
.trade-hub-back-btn:hover {
  transform: translateY(-1px);
  border-color: var(--line-strong);
  background: var(--cyan-soft);
}
.trade-hub-top-center { text-align: center; }
.trade-hub-screen-title {
  margin: 0;
  font-size: clamp(22px, 2.2vw, 32px);
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text);
  text-shadow: 0 0 28px rgba(19, 216, 231, 0.18);
}
.trade-hub-top-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  min-width: 0;
}
.trade-hub-partner-select {
  width: 100%;
  max-width: 220px;
  background: rgba(12, 35, 52, 0.72);
  border: 1px solid var(--line);
  color: var(--text);
  padding: 7px 10px;
  font-size: 11px;
  font-weight: 800;
  border-radius: 10px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.trade-hub-date-text {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted-2);
  letter-spacing: 0.04em;
  white-space: nowrap;
  line-height: 1.2;
}
.trade-hub-war-room-layout {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(250px, 310px) minmax(380px, 1fr) minmax(250px, 310px);
  gap: 0;
  padding: 0;
  height: 100%;
  min-height: 0;
  align-items: stretch;
}
.trade-hub-centre {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
.trade-team-panel {
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.trade-team-panel:hover {
  border-color: var(--line-2);
}
.trade-team-panel-identity {
  display: flex;
  align-items: center;
  gap: 12px;
}
.trade-team-panel-names { min-width: 0; text-align: left; }
.trade-team-panel-name {
  font-size: 16px;
  font-weight: 1000;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  line-height: 1.15;
  color: var(--text);
}
.trade-team-panel-record {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
  font-weight: 700;
}
.trade-team-badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.trade-status-badge,
.trade-conf-badge,
.trade-div-badge {
  font-size: 9px;
  padding: 5px 9px;
  border-radius: 999px;
  letter-spacing: 0.1em;
  font-weight: 900;
  text-transform: uppercase;
  border: 1px solid var(--line);
}
.trade-status-badge {
  background: var(--green-soft);
  border-color: rgba(82, 223, 148, 0.35);
  color: var(--green);
}
.trade-conf-badge { background: var(--cyan-soft); color: var(--cyan); border-color: rgba(19, 216, 231, 0.35); }
.trade-div-badge { background: var(--purple-soft); color: var(--purple); border-color: rgba(201, 146, 255, 0.35); }
.trade-team-rank-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.trade-team-rank-grid div {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 8px 10px;
  text-align: center;
}
.trade-team-rank-grid span {
  display: block;
  font-size: 8px;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  font-weight: 800;
}
.trade-team-rank-grid strong { font-size: 14px; color: var(--text); }
.trade-team-ratings {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.trade-rating-pill {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 8px 4px;
  text-align: center;
}
.trade-rating-label {
  display: block;
  font-size: 8px;
  letter-spacing: 0.14em;
  color: var(--muted);
  text-transform: uppercase;
  font-weight: 800;
}
.trade-rating-value {
  font-size: 22px;
  font-weight: 1000;
  color: var(--cyan);
  line-height: 1.1;
}
.trade-team-cap-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  font-size: 10px;
}
.trade-team-cap-strip span {
  display: block;
  font-size: 8px;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  font-weight: 800;
}
.trade-team-cap-strip strong { font-size: 12px; color: var(--text); }
.trade-team-cap-strip .ok { color: var(--green); }
.trade-team-cap-strip .bad { color: var(--red); }
.trade-team-interest {
  font-size: 11px;
  color: var(--purple);
  text-align: center;
  padding: 8px;
  background: var(--purple-soft);
  border: 1px solid rgba(201, 146, 255, 0.25);
  border-radius: 12px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.trade-team-needs-block { margin-top: 2px; }
.trade-pool-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-bottom: 8px;
}
.trade-pool-list {
  flex: 1;
  overflow-y: auto;
  max-height: min(240px, 38vh);
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-right: 2px;
}
.trade-pool-row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px 10px;
  cursor: grab;
}
.trade-pool-row:active { cursor: grabbing; }
.trade-pool-row.used { opacity: 0.35; cursor: not-allowed; }
.trade-pool-row.view-only { opacity: 0.72; cursor: default; }
.trade-pool-row.inactive { opacity: 0.48; filter: grayscale(0.35); }
.trade-pool-row-main { min-width: 0; text-align: left; }
.trade-pool-row-main strong {
  display: block;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text);
}
.trade-pool-empty {
  padding: 16px;
  text-align: center;
  font-size: 11px;
}
.trade-pick-year-divider {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 4px 4px;
  margin-top: 4px;
}
.trade-pick-year-divider:first-child {
  margin-top: 0;
  padding-top: 2px;
}
.trade-pick-year-divider-title {
  color: rgba(148, 178, 194, 0.95);
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-pick-year-divider-count {
  color: rgba(128, 150, 168, 0.85);
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.1em;
}
.trade-players-full-body .trade-pool-pick {
  grid-template-columns: 76px minmax(0, 1fr);
}
.trade-players-full-body .trade-pick-list-mid {
  width: 100%;
}
.trade-players-full-body .trade-pick-list-mid .trade-player-value-focus {
  flex: 1 1 auto;
  width: 100%;
}
.trade-players-full-body .trade-pool-pick-icon {
  width: 64px;
  height: 64px;
  min-width: 64px;
  border-radius: 14px;
  border: 1px solid rgba(233, 168, 60, 0.38);
  background:
    linear-gradient(180deg, rgba(233, 168, 60, 0.2), rgba(7, 20, 32, 0.94));
  box-shadow:
    0 8px 18px rgba(0, 0, 0, 0.32),
    inset 0 1px 0 rgba(255, 214, 102, 0.12),
    0 0 14px rgba(233, 168, 60, 0.14);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
}
.trade-players-full-body .trade-pool-pick-icon .trade-pick-icon-round {
  font-size: 18px;
  font-weight: 1000;
  color: #ffd166;
  line-height: 1;
}
.trade-players-full-body .trade-pool-pick-icon .trade-pick-icon-year {
  font-size: 10px;
  font-weight: 900;
  color: rgba(255, 214, 102, 0.82);
  letter-spacing: 0.08em;
}
.trade-players-full-body .trade-pool-pick .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-depth .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-low .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-unknown .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-useful .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-top-asset .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-elite .trade-player-value-fill,
.trade-players-full-body .trade-pool-pick .is-pick-value.value-franchise .trade-player-value-fill {
  background: linear-gradient(90deg, #e9a83c, #ffd166);
  box-shadow: 0 0 20px rgba(233, 168, 60, 0.32);
}
.trade-players-full-body .trade-pick-detail-orig {
  color: #f2fbff;
  border-color: rgba(0, 216, 223, 0.45);
  background: rgba(0, 216, 223, 0.16);
}
.trade-players-full-body .trade-pick-detail-prot {
  color: #ffe8d4;
  border-color: rgba(255, 154, 106, 0.42);
  background: rgba(255, 154, 106, 0.1);
}
.trade-players-full-body .trade-pick-detail-year {
  color: #d4faf8;
  border-color: rgba(94, 240, 245, 0.48);
  background: rgba(0, 216, 223, 0.18);
}
.trade-pick-proj-tower strong {
  font-size: 16px;
  letter-spacing: 0.04em;
}
.trade-players-full-body .trade-pick-proj-tower strong {
  font-size: 15px;
}
.trade-players-full-body .trade-pick-proj-tower.has-range strong {
  font-size: 13px;
}
.trade-pick-value-meter {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 100%;
}
.trade-pick-value-label {
  color: rgba(128, 150, 168, 0.9);
  font-size: 7px;
  font-weight: 900;
  letter-spacing: 0.1em;
}
.trade-pick-value-track {
  height: 4px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}
.trade-pick-value-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(0, 216, 223, 0.55), rgba(0, 216, 223, 0.95));
  box-shadow: 0 0 8px rgba(0, 216, 223, 0.35);
}
.trade-pick-value-meter.tier-franchise .trade-pick-value-fill,
.trade-pick-value-meter.tier-elite .trade-pick-value-fill {
  background: linear-gradient(90deg, rgba(255, 214, 102, 0.7), rgba(255, 236, 160, 0.95));
  box-shadow: 0 0 8px rgba(255, 214, 102, 0.35);
}
.trade-pick-value-meter.tier-top .trade-pick-value-fill {
  background: linear-gradient(90deg, rgba(120, 220, 255, 0.65), rgba(0, 216, 223, 0.95));
}
.trade-pick-value-meter.tier-useful .trade-pick-value-fill {
  background: linear-gradient(90deg, rgba(82, 223, 148, 0.55), rgba(82, 223, 148, 0.9));
}
.trade-pick-value-meter.tier-depth .trade-pick-value-fill,
.trade-pick-value-meter.tier-negative .trade-pick-value-fill {
  background: linear-gradient(90deg, rgba(128, 150, 168, 0.45), rgba(128, 150, 168, 0.75));
}
.trade-pick-range {
  color: rgba(128, 150, 168, 0.95);
  font-size: 7px;
  font-weight: 900;
  letter-spacing: 0.08em;
}
.trade-pick-empty {
  color: rgba(128, 150, 168, 0.55);
  font-size: 11px;
  font-weight: 700;
  text-align: center;
  padding: 2px 0 4px;
}
.trade-asset-card-pick-meter {
  margin-top: 4px;
  max-width: 100%;
}
.trade-asset-hero-pick-meter {
  margin-top: 4px;
  width: 100%;
}
.trade-asset-value-panel-meter {
  margin-bottom: 10px;
}
.trade-construction-grid {
  display: grid;
  grid-template-columns: 1fr 28px 1fr;
  gap: 8px;
  align-items: stretch;
  flex: 1 1 auto;
  min-height: 0;
}
.trade-package-header {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 8px;
  font-size: 10px;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.trade-package-header-main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 0;
}
.trade-package-header-main > span {
  font-weight: 1000;
  letter-spacing: 0.16em;
}
.trade-package-cap-strip {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 6px;
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.trade-package-cap-label {
  color: rgba(150, 176, 196, 0.92);
  letter-spacing: 0.1em;
}
.trade-package-cap-flow {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: rgba(210, 228, 240, 0.95);
}
.trade-package-cap-arrow {
  color: rgba(120, 150, 168, 0.85);
  font-weight: 700;
}
.trade-package-cap-now,
.trade-package-cap-after {
  color: rgba(180, 204, 218, 0.88);
}
.trade-package-cap-delta.good {
  color: var(--green);
  text-shadow: 0 0 12px rgba(82, 223, 148, 0.2);
}
.trade-package-cap-delta.bad {
  color: #ff9a6a;
  text-shadow: 0 0 12px rgba(255, 154, 106, 0.18);
}
.trade-package-divider {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  align-self: center;
  padding-top: 0;
}
.trade-slot {
  position: relative;
  min-height: 72px;
  margin-bottom: 6px;
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.trade-slot.drop-active { animation: trade-slot-pulse 0.9s ease infinite; }
.trade-slot-index {
  position: absolute;
  top: 6px;
  left: 8px;
  font-size: 9px;
  font-weight: 900;
  z-index: 2;
}
.trade-slot-placeholder {
  height: 100%;
  min-height: 58px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
}
.trade-asset-card {
  position: relative;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
  padding: 8px 8px 8px 20px;
  min-height: 68px;
  cursor: grab;
  border-radius: 12px;
  border: 1px solid rgba(0, 216, 223, 0.12);
  background:
    linear-gradient(180deg, rgba(8, 20, 30, 0.92), rgba(4, 12, 20, 0.95)),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.01) 0px, rgba(255,255,255,0.01) 1px, transparent 1px, transparent 5px);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.trade-asset-card-player {
  grid-template-columns: auto auto 1fr auto;
  gap: 10px;
  padding: 10px 10px 10px 20px;
  min-height: 76px;
  background:
    linear-gradient(180deg, rgba(10, 24, 36, 0.96), rgba(4, 12, 20, 0.98)),
    radial-gradient(circle at 18% 0%, rgba(19, 216, 231, 0.08), transparent 42%),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 6px);
  border-color: rgba(0, 216, 223, 0.18);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 8px 22px rgba(0, 0, 0, 0.28);
}
.trade-asset-ovr-focus {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 52px;
  padding: 6px 8px;
  border-radius: 14px;
  border: 1px solid rgba(233, 168, 60, 0.42);
  background:
    radial-gradient(circle at 50% 0%, rgba(233, 168, 60, 0.18), transparent 58%),
    linear-gradient(180deg, rgba(12, 35, 52, 0.92), rgba(5, 17, 27, 0.96));
  box-shadow: 0 0 22px rgba(233, 168, 60, 0.14);
}
.trade-asset-ovr-label {
  color: rgba(233, 168, 60, 0.82);
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.16em;
  line-height: 1;
}
.trade-asset-ovr-number {
  color: #ffe08a;
  font-size: clamp(30px, 3.2vw, 40px);
  font-weight: 1000;
  line-height: 0.95;
  letter-spacing: -0.02em;
  text-shadow:
    0 0 18px rgba(233, 168, 60, 0.34),
    0 0 32px rgba(19, 216, 231, 0.12);
}
.trade-asset-pot-big {
  display: inline-flex;
  align-items: center;
  margin-top: 6px;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(201, 146, 255, 0.48);
  background:
    radial-gradient(circle at 50% 0%, rgba(201, 146, 255, 0.2), transparent 60%),
    rgba(88, 44, 130, 0.22);
  color: #e4c4ff;
  font-size: 12px;
  font-weight: 1000;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  box-shadow: 0 0 18px rgba(201, 146, 255, 0.16);
}
.trade-asset-card-meta-tiny {
  margin-top: 4px;
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(128, 150, 168, 0.95);
}
.trade-asset-clause-mini {
  padding: 1px 5px;
  border-radius: 999px;
  border: 1px solid rgba(255, 96, 109, 0.28);
  color: #ff9aa3;
  font-size: 8px;
  font-weight: 900;
}
.trade-asset-card-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: center;
  gap: 5px;
  min-width: 72px;
}
.trade-asset-aav-badge {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 64px;
  padding: 6px 8px;
  border-radius: 10px;
  border: 1px solid rgba(19, 216, 231, 0.32);
  background:
    linear-gradient(180deg, rgba(12, 35, 52, 0.9), rgba(5, 17, 27, 0.94));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.trade-asset-aav-value {
  color: #dff8ff;
  font-size: 13px;
  font-weight: 1000;
  line-height: 1.1;
  letter-spacing: 0.02em;
}
.trade-asset-aav-label {
  margin-top: 2px;
  color: rgba(128, 150, 168, 0.95);
  font-size: 8px;
  font-weight: 900;
  letter-spacing: 0.14em;
}
.trade-asset-cap-impact {
  display: inline-flex;
  align-items: center;
  padding: 3px 7px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid transparent;
}
.trade-asset-cap-impact.good {
  color: var(--green);
  border-color: rgba(82, 223, 148, 0.32);
  background: rgba(82, 223, 148, 0.1);
}
.trade-asset-cap-impact.bad {
  color: #ff9a6a;
  border-color: rgba(255, 154, 106, 0.32);
  background: rgba(255, 154, 106, 0.1);
}
.trade-asset-pick-tier {
  align-self: center;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(0, 216, 223, 0.28);
  background: rgba(0, 216, 223, 0.08);
  color: var(--cyan);
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.trade-asset-card-player .trade-flag-badge,
.trade-asset-card-player .trade-flag-fallback {
  width: 30px;
  height: 22px;
  min-width: 30px;
  border-radius: 3px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45);
}
.trade-asset-card-player .trade-asset-card-remove {
  position: static;
  width: 20px;
  height: 20px;
  margin-top: 2px;
}
.trade-asset-headshot.player-headshot {
  border-radius: 10px;
  border: 1px solid rgba(0, 216, 223, 0.2);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
.trade-asset-card:active { cursor: grabbing; }
.trade-asset-card-body { min-width: 0; text-align: left; }
.trade-asset-card-name { font-size: 14px; line-height: 1.15; }
.trade-asset-card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin-top: 4px;
  font-size: 9px;
}
.trade-asset-card-side {
  text-align: right;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}
.trade-asset-ovr {
  font-size: 28px;
  font-weight: 1000;
  line-height: 1;
}
.trade-asset-pot {
  font-size: 9px;
  font-weight: 800;
  padding: 2px 6px;
  border-radius: 999px;
}
.trade-asset-card-remove {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 22px;
  height: 22px;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
}
.trade-pos-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 18px;
  border-radius: 6px;
  font-size: 9px;
  font-weight: 900;
}
.trade-pick-icon {
  width: 48px;
  height: 58px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  border-radius: 10px;
  border: 1px solid rgba(0, 216, 223, 0.22);
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.08), rgba(0, 0, 0, 0.25));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
.trade-pick-icon-round { font-size: 15px; font-weight: 900; color: var(--cyan); }
.trade-pick-icon-year { font-size: 10px; font-weight: 800; color: var(--muted); letter-spacing: 0.06em; }
.trade-analysis-panel {
  padding: 10px 12px;
  transition: box-shadow 0.25s ease;
}
.trade-analysis-panel.is-evaluating { animation: trade-eval-pulse 1.1s ease infinite; }
.trade-analysis-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}
.trade-analysis-title { font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; }
.trade-analysis-verdict { font-size: 12px; font-weight: 900; letter-spacing: 0.1em; text-transform: uppercase; }
.trade-scale-row { margin-bottom: 12px; }
.trade-dual-value-bars { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.trade-dual-value-row { display: grid; grid-template-columns: 120px 1fr; gap: 10px; align-items: center; }
.trade-dual-value-label { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.trade-dual-value-label span {
  font-size: 9px; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; font-weight: 800;
}
.trade-dual-value-label strong { font-size: 18px; color: var(--text); line-height: 1; }
.trade-dual-value-track {
  height: 10px; border-radius: 999px; background: rgba(255,255,255,0.06);
  border: 1px solid var(--line); overflow: hidden;
}
.trade-dual-value-fill {
  height: 100%; border-radius: 999px; transition: width 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
.trade-dual-value-fill.user { background: linear-gradient(90deg, rgba(19, 216, 231, 0.75), rgba(138, 180, 255, 0.45)); }
.trade-dual-value-fill.partner { background: linear-gradient(90deg, rgba(214, 179, 106, 0.75), rgba(255, 196, 120, 0.45)); }
.trade-dual-fill-good.user { background: linear-gradient(90deg, var(--green), #2fbf73); }
.trade-dual-fill-good.partner { background: linear-gradient(90deg, #3ecf8e, var(--green)); }
.trade-dual-fill-warn.user, .trade-dual-fill-warn.partner { background: linear-gradient(90deg, var(--gold), #d99023); }
.trade-dual-fill-bad.user, .trade-dual-fill-bad.partner { background: linear-gradient(90deg, var(--red), #c93442); }
.trade-metric-grid-compact { grid-template-columns: repeat(4, 1fr); }
.trade-analysis-note {
  margin: 8px 0 0; font-size: 12px; line-height: 1.45; color: rgba(233, 247, 251, 0.72);
  border-top: 1px solid var(--line); padding-top: 8px;
}
.trade-fan-warning {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  margin: 10px 0; padding: 12px 14px; border-radius: 10px;
  background: rgba(201, 52, 66, 0.12); border: 1px solid rgba(255, 107, 122, 0.45);
}
.trade-fan-warning-body strong { display: block; color: #ff8f98; font-size: 13px; margin-bottom: 4px; }
.trade-fan-warning-body p { margin: 0; font-size: 12px; color: rgba(255, 220, 224, 0.88); line-height: 1.4; }
.trade-fan-warning-dismiss {
  flex-shrink: 0; border: 1px solid rgba(255, 107, 122, 0.55); background: rgba(255,255,255,0.06);
  color: #ffd6da; border-radius: 8px; padding: 6px 10px; font-size: 11px; cursor: pointer;
}
.trade-pool-side-stats { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; flex-shrink: 0; }
.trade-pool-compact-stats { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
.trade-pool-name-row,
.trade-asset-card-name-row,
.trade-ctx-name-row {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.trade-flag-badge {
  width: 22px;
  height: 16px;
  object-fit: cover;
  border-radius: 2px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.45);
  flex-shrink: 0;
}
.trade-flag-badge.is-md,
.trade-flag-fallback.is-md {
  width: 30px;
  height: 22px;
  min-width: 30px;
}
.trade-flag-badge.is-lg {
  width: 34px;
  height: 24px;
}
.trade-flag-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 16px;
  padding: 0 4px;
  border-radius: 3px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(255, 255, 255, 0.06);
  font-size: 8px;
  font-weight: 900;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.trade-flag-fallback.is-lg {
  min-width: 34px;
  height: 24px;
  font-size: 10px;
}
.trade-value-chip {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  min-width: 72px;
}
.trade-value-chip.compact {
  min-width: 58px;
  gap: 3px;
}
.trade-value-chip-label {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gold);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-value-chip.compact .trade-value-chip-label { font-size: 8px; }
.trade-value-chip-track {
  width: 100%;
  height: 5px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
  border: 1px solid rgba(0, 216, 223, 0.12);
}
.trade-value-chip-track.hero {
  height: 8px;
  margin-top: 6px;
}
.trade-value-chip-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(0, 216, 223, 0.55), rgba(245, 215, 110, 0.85));
  box-shadow: 0 0 10px rgba(0, 216, 223, 0.25);
}
.trade-value-chip.tier-franchise .trade-value-chip-label { color: #ffd700; }
.trade-value-chip.tier-franchise .trade-value-chip-fill { background: linear-gradient(90deg, #c9a227, #ffd700); }
.trade-value-chip.tier-elite .trade-value-chip-label { color: #ff9f43; }
.trade-value-chip.tier-elite .trade-value-chip-fill { background: linear-gradient(90deg, #e67e22, #ff9f43); }
.trade-value-chip.tier-top .trade-value-chip-label { color: #54a0ff; }
.trade-value-chip.tier-top .trade-value-chip-fill { background: linear-gradient(90deg, #2e86de, #54a0ff); }
.trade-value-chip.tier-useful .trade-value-chip-label { color: var(--green); }
.trade-value-chip.tier-useful .trade-value-chip-fill { background: linear-gradient(90deg, #1e8449, var(--green)); }
.trade-value-chip.tier-depth .trade-value-chip-label,
.trade-value-chip.tier-unknown .trade-value-chip-label { color: var(--muted); }
.trade-value-chip.tier-depth .trade-value-chip-fill,
.trade-value-chip.tier-unknown .trade-value-chip-fill { background: linear-gradient(90deg, #4a5568, #718096); }
.trade-value-chip.tier-negative .trade-value-chip-label { color: var(--red); }
.trade-value-chip.tier-negative .trade-value-chip-fill { background: linear-gradient(90deg, #c0392b, var(--red)); }
.trade-asset-stat-stack {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  min-width: 64px;
}
.trade-asset-ovr-big {
  font-size: 22px;
  font-weight: 900;
  color: var(--gold);
  line-height: 1;
  text-shadow: 0 0 10px rgba(245, 215, 110, 0.25);
}
.trade-asset-cap-mini {
  font-size: 10px;
  font-weight: 800;
  color: var(--cyan);
  letter-spacing: 0.04em;
}
.trade-asset-hand {
  font-size: 9px;
  font-weight: 800;
  color: var(--muted);
}
.trade-pool-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; grid-column: 1 / -1; }
.trade-pool-chip { font-size: 9px; padding: 2px 6px; border-radius: 999px; background: rgba(255,255,255,0.06); color: var(--muted); }
.trade-hub-breakdown-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.trade-hub-cap-flow { display: flex; gap: 8px; font-size: 10px; color: var(--muted); margin-top: 4px; flex-wrap: wrap; }
.trade-team-rank-grid-compact { grid-template-columns: repeat(4, 1fr); }
.trade-cap-mgmt-grid-compact { grid-template-columns: repeat(2, 1fr); }
.trade-fan-hint { font-size: 11px; margin-top: 6px; }
.trade-war-meter.low strong { color: #ff8f98; }
.trade-war-meter-bar div.low { background: linear-gradient(90deg, var(--red), #ff6b7a); }
.trade-war-meter.high strong { color: var(--green); }
.trade-scale-label { display: block; font-size: 9px; letter-spacing: 0.14em; margin-bottom: 6px; }
.trade-scale-track { height: 10px; }
.trade-scale-fill-animated { transition: width 0.45s cubic-bezier(0.22, 1, 0.36, 1); }
.trade-metric-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
}
.trade-metric-card {
  text-align: center;
  padding: 10px 6px;
  transition: border-color 0.25s ease, color 0.25s ease;
}
.trade-metric-card span { display: block; font-size: 8px; letter-spacing: 0.1em; margin-bottom: 4px; }
.trade-metric-card strong { font-size: 14px; }
.trade-reaction-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 12px;
}
.trade-reaction-card { padding: 10px 12px; }
.trade-reaction-card span { display: block; font-size: 8px; letter-spacing: 0.14em; margin-bottom: 4px; }
.trade-reaction-card p { margin: 0; font-size: 11px; line-height: 1.45; }
.trade-centre-cap-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.trade-cap-mgmt-panel,
.trade-hub-report-panel,
.trade-hub-cap-panel,
.trade-hub-needs-panel,
.trade-hub-breakdown-panel,
.trade-hub-league-panel {
  padding: 16px;
}
.trade-cap-mgmt-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.trade-cap-mgmt-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  font-size: 10px;
}
.trade-cap-flex {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
  font-size: 10px;
}
.trade-cap-flex-bar {
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
}
.trade-cap-flex-fill { height: 100%; transition: width 0.4s ease; }
.trade-war-room {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  padding: 16px;
}
.trade-war-room-col { min-width: 0; }
.trade-war-line {
  font-size: 10px;
  padding: 5px 0;
  border-top: 1px solid var(--line);
  line-height: 1.4;
}
.trade-war-meter {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 8px;
  align-items: center;
  font-size: 10px;
  margin-bottom: 8px;
}
.trade-war-meter-bar {
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
}
.trade-war-meter-bar div { height: 100%; transition: width 0.35s ease; }
.trade-hub-toast-float {
  position: fixed;
  bottom: 56px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 40;
  padding: 10px 16px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.trade-decision-toast {
  position: fixed;
  top: 86px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 80;
  width: min(560px, calc(100vw - 32px));
  min-height: 112px;
  padding: 18px 22px 20px;
  border-radius: 22px;
  overflow: hidden;
  text-align: center;
  border: 1px solid rgba(0, 216, 223, 0.28);
  background:
    radial-gradient(circle at 50% 0%, rgba(19, 216, 231, 0.22), transparent 58%),
    linear-gradient(180deg, rgba(9, 27, 40, 0.98), rgba(4, 13, 22, 0.98));
  box-shadow:
    0 28px 72px rgba(0, 0, 0, 0.58),
    0 0 42px rgba(19, 216, 231, 0.14),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
  animation: tradeDecisionIn 0.28s cubic-bezier(0.22, 1, 0.36, 1);
  pointer-events: none;
}
.trade-success-overlay {
  position: fixed;
  inset: 0;
  z-index: 120;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(2, 8, 14, 0.72);
  backdrop-filter: blur(6px);
  animation: tradeDecisionIn 0.22s ease;
}
.trade-success-modal {
  position: relative;
  width: min(520px, calc(100vw - 32px));
  padding: 28px 26px 22px;
  border-radius: 22px;
  overflow: hidden;
  text-align: center;
  border: 1px solid rgba(82, 223, 148, 0.45);
  background:
    radial-gradient(circle at 50% 0%, rgba(82, 223, 148, 0.22), transparent 55%),
    linear-gradient(180deg, rgba(9, 27, 40, 0.99), rgba(4, 13, 22, 0.99));
  box-shadow:
    0 32px 80px rgba(0, 0, 0, 0.62),
    0 0 48px rgba(82, 223, 148, 0.2);
  pointer-events: auto;
}
.trade-success-modal strong {
  position: relative;
  z-index: 1;
  display: block;
  color: #f2fbff;
  font-size: clamp(24px, 2.4vw, 36px);
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-success-modal > p {
  position: relative;
  z-index: 1;
  margin: 10px auto 0;
  max-width: 420px;
  color: rgba(220, 236, 244, 0.9);
  font-size: 14px;
  font-weight: 700;
  line-height: 1.4;
}
.trade-success-swap {
  position: relative;
  z-index: 1;
  display: grid;
  gap: 10px;
  margin-top: 18px;
  text-align: left;
}
.trade-success-swap > div {
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(156, 218, 236, 0.16);
  background: rgba(0, 0, 0, 0.22);
}
.trade-success-swap span {
  display: block;
  margin-bottom: 4px;
  color: rgba(150, 176, 196, 0.92);
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-success-swap strong {
  font-size: 13px;
  letter-spacing: 0.04em;
  text-transform: none;
  font-weight: 800;
}
.trade-success-dismiss {
  position: relative;
  z-index: 1;
  margin-top: 18px;
  min-width: 160px;
  padding: 12px 18px;
  border-radius: 12px;
  border: 1px solid rgba(82, 223, 148, 0.45);
  background: rgba(82, 223, 148, 0.16);
  color: #dfffee;
  font-size: 12px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  cursor: pointer;
}
.trade-success-dismiss:hover {
  background: rgba(82, 223, 148, 0.24);
}
.trade-desk-cap-line {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(160, 190, 210, 0.9);
}
.trade-fan-vitriol.is-compact {
  gap: 4px;
  padding-top: 2px;
  padding-bottom: 2px;
}
.trade-fan-vitriol.is-compact .trade-fan-vitriol-track {
  height: 6px;
}
.trade-intel-list-lean {
  gap: 2px;
}
.trade-decision-glow {
  position: absolute;
  inset: -60px;
  opacity: 0.34;
  background:
    radial-gradient(circle at 50% 0%, rgba(19, 216, 231, 0.38), transparent 42%);
  pointer-events: none;
}
.trade-decision-kicker {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  margin-bottom: 9px;
}
.trade-decision-kicker span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 96px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(0, 216, 223, 0.3);
  background: rgba(255, 255, 255, 0.055);
  color: var(--cyan);
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.trade-decision-toast strong {
  position: relative;
  z-index: 1;
  display: block;
  color: #f2fbff;
  font-size: clamp(22px, 2.2vw, 34px);
  font-weight: 1000;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  text-shadow: 0 0 26px rgba(19, 216, 231, 0.18);
}
.trade-decision-toast p {
  position: relative;
  z-index: 1;
  margin: 9px auto 0;
  max-width: 460px;
  color: rgba(220, 236, 244, 0.88);
  font-size: 13px;
  font-weight: 800;
  line-height: 1.35;
  letter-spacing: 0.03em;
}
.trade-decision-accepted {
  border-color: rgba(82, 223, 148, 0.52);
  box-shadow:
    0 28px 72px rgba(0, 0, 0, 0.58),
    0 0 48px rgba(82, 223, 148, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.trade-decision-accepted .trade-decision-kicker span {
  color: var(--green);
  border-color: rgba(82, 223, 148, 0.42);
  background: rgba(82, 223, 148, 0.11);
}
.trade-decision-rejected {
  border-color: rgba(255, 96, 109, 0.52);
  box-shadow:
    0 28px 72px rgba(0, 0, 0, 0.58),
    0 0 48px rgba(255, 96, 109, 0.16),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.trade-decision-rejected .trade-decision-kicker span {
  color: #ff9aa3;
  border-color: rgba(255, 96, 109, 0.42);
  background: rgba(255, 96, 109, 0.11);
}
.trade-decision-steal {
  border-color: rgba(255, 209, 102, 0.62);
  box-shadow:
    0 28px 72px rgba(0, 0, 0, 0.58),
    0 0 54px rgba(233, 168, 60, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.trade-decision-steal .trade-decision-kicker span {
  color: #ffd166;
  border-color: rgba(255, 209, 102, 0.5);
  background: rgba(233, 168, 60, 0.13);
}
.trade-decision-risky .trade-decision-kicker span,
.trade-decision-close .trade-decision-kicker span {
  color: var(--gold);
  border-color: rgba(233, 168, 60, 0.48);
  background: rgba(233, 168, 60, 0.12);
}
.trade-decision-blocked .trade-decision-kicker span,
.trade-decision-lowball .trade-decision-kicker span {
  color: #ff9aa3;
  border-color: rgba(255, 96, 109, 0.42);
  background: rgba(255, 96, 109, 0.11);
}
.trade-decision-technical {
  border-color: rgba(233, 168, 60, 0.58);
  box-shadow:
    0 28px 72px rgba(0, 0, 0, 0.58),
    0 0 48px rgba(233, 168, 60, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.trade-decision-technical .trade-decision-kicker span {
  color: var(--gold);
  border-color: rgba(233, 168, 60, 0.48);
  background: rgba(233, 168, 60, 0.12);
}
@keyframes tradeDecisionIn {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(-16px) scale(0.96);
    filter: blur(4px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0) scale(1);
    filter: blur(0);
  }
}
.trade-asset-pool {
  flex: 1;
  min-height: 180px;
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}
.trade-pool-tabs button {
  padding: 8px 4px;
  font-size: 9px;
  letter-spacing: 0.1em;
  font-weight: 900;
  text-transform: uppercase;
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
  color: var(--muted);
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
}
.trade-pool-tabs button:hover {
  transform: translateY(-1px);
  border-color: var(--line-2);
  color: var(--text);
}
.trade-pool-tabs button.active {
  color: var(--cyan);
  border-color: var(--line-strong);
  background:
    linear-gradient(180deg, rgba(19, 216, 231, 0.16), rgba(19, 216, 231, 0.04)),
    var(--panel-2);
  box-shadow: 0 0 18px rgba(19, 216, 231, 0.12);
}
.trade-pool-row {
  border: 1px solid var(--line);
  background:
    linear-gradient(180deg, rgba(18, 42, 61, 0.45), rgba(6, 20, 31, 0.34)),
    radial-gradient(circle at 100% 0%, rgba(19, 216, 231, 0.05), transparent 52%);
  border-radius: 12px;
  transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
.trade-pool-row:hover:not(.used):not(.view-only) {
  border-color: var(--line-strong);
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
  background:
    linear-gradient(180deg, rgba(16, 44, 62, 0.84), rgba(7, 25, 38, 0.82));
}
.trade-pool-player { border-left: 3px solid var(--green); }
.trade-pool-pick { border-left: 3px solid var(--cyan); }
.trade-pool-prospect { border-left: 3px solid var(--purple); }
.trade-pool-row-main span { color: var(--muted); }
.trade-pool-cap { color: var(--text); }
.trade-pool-empty {
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 12px;
  background: var(--panel-3);
}
.trade-package-col {
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px;
  box-shadow: var(--shadow);
  min-height: 0;
  overflow-y: auto;
}
.trade-package-header {
  color: var(--muted);
  font-weight: 900;
  letter-spacing: 0.18em;
}
.trade-package-divider { color: var(--muted-2); }
.trade-slot.empty {
  border: 2px dashed var(--line-2);
  background: var(--panel-3);
  border-radius: 12px;
}
.trade-slot.filled { border: 1px solid var(--line); border-radius: 12px; }
.trade-slot.drop-active {
  border-color: var(--line-strong);
  box-shadow: 0 0 20px rgba(19, 216, 231, 0.18);
}
@keyframes trade-slot-pulse {
  0%, 100% { box-shadow: 0 0 0 rgba(19, 216, 231, 0.08); }
  50% { box-shadow: 0 0 18px rgba(19, 216, 231, 0.24); }
}
.trade-slot-index { color: var(--muted-2); }
.trade-slot-placeholder {
  color: var(--muted);
  letter-spacing: 0.14em;
}
.trade-slot-placeholder small { color: var(--muted-2); }
.trade-asset-card {
  background:
    linear-gradient(180deg, rgba(18, 42, 61, 0.55), rgba(6, 20, 31, 0.5)),
    radial-gradient(circle at 0% 0%, rgba(19, 216, 231, 0.06), transparent 40%);
  border-radius: 12px;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.trade-asset-card-player { border-left: 3px solid var(--green); }
.trade-asset-card-pick { border-left: 3px solid var(--cyan); }
.trade-asset-card-name {
  color: var(--text);
  font-weight: 900;
  letter-spacing: 0.04em;
}
.trade-asset-card-meta { color: var(--muted); }
.trade-ret-badge {
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid rgba(233, 168, 60, 0.35);
  background: rgba(233, 168, 60, 0.1);
  color: var(--gold);
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.08em;
}
.trade-asset-card-remove {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
  transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}
.trade-asset-card-remove:hover {
  transform: translateY(-1px);
  border-color: var(--red);
  color: var(--red);
}
.trade-pos-icon.pos-f { background: var(--green-soft); color: var(--green); }
.trade-pos-icon.pos-d { background: var(--blue-soft); color: var(--blue); }
.trade-pos-icon.pos-g { background: var(--gold-soft); color: var(--gold); }
.trade-pick-icon {
  background:
    linear-gradient(160deg, rgba(19, 216, 231, 0.22), rgba(15, 46, 66, 0.9));
  border: 1px solid rgba(19, 216, 231, 0.35);
  border-radius: 10px;
}
.trade-pick-icon-round { color: var(--cyan); }
.trade-pick-icon-year { color: var(--muted); }
.trade-analysis-panel {
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
}
@keyframes trade-eval-pulse {
  0%, 100% { box-shadow: var(--shadow); }
  50% { box-shadow: 0 0 24px rgba(19, 216, 231, 0.14), var(--shadow); }
}
.trade-analysis-title { color: var(--cyan); font-weight: 900; }
.trade-analysis-verdict-good { color: var(--green); }
.trade-analysis-verdict-warn { color: var(--gold); }
.trade-analysis-verdict-bad { color: var(--red); }
.trade-analysis-verdict-neutral { color: var(--muted); }
.trade-scale-label { color: var(--muted); text-transform: uppercase; font-weight: 800; }
.trade-hub-value-track {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line);
  border-radius: 999px;
}
.trade-hub-value-center { background: var(--line-strong); }
.trade-hub-value-fill-neutral { background: linear-gradient(90deg, rgba(19, 216, 231, 0.55), rgba(138, 180, 255, 0.35)); }
.trade-hub-value-fill-good { background: linear-gradient(90deg, var(--green), #2fbf73); }
.trade-hub-value-fill-warn { background: linear-gradient(90deg, var(--gold), #d99023); }
.trade-hub-value-fill-bad { background: linear-gradient(90deg, var(--red), #c93442); }
.trade-metric-card {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
}
.trade-metric-card span { color: var(--muted); text-transform: uppercase; font-weight: 800; }
.trade-metric-card.good { border-color: rgba(82, 223, 148, 0.4); color: var(--green); }
.trade-metric-card.warn { border-color: rgba(233, 168, 60, 0.4); color: var(--gold); }
.trade-metric-card.bad { border-color: rgba(255, 96, 109, 0.4); color: var(--red); }
.trade-reaction-card {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
}
.trade-reaction-card span { color: var(--muted); text-transform: uppercase; font-weight: 800; }
.trade-reaction-card p { color: rgba(233, 247, 251, 0.82); }
.trade-reaction-card.gm { border-left: 3px solid var(--purple); }
.trade-reaction-card.owner { border-left: 3px solid var(--gold); }
.trade-cap-mgmt-panel,
.trade-hub-report-panel,
.trade-hub-cap-panel,
.trade-hub-needs-panel,
.trade-hub-breakdown-panel,
.trade-hub-league-panel,
.trade-war-room {
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
}
.trade-hub-panel-title {
  color: var(--cyan);
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.trade-cap-mgmt-row {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
}
.trade-cap-mgmt-row span { color: var(--muted); text-transform: uppercase; font-weight: 800; font-size: 9px; }
.trade-cap-mgmt-row.good strong { color: var(--green); }
.trade-cap-mgmt-row.bad strong { color: var(--red); }
.trade-cap-flex-bar {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line);
}
.trade-cap-flex-fill {
  background: linear-gradient(90deg, var(--green), var(--cyan));
}
.trade-war-line { color: rgba(233, 247, 251, 0.72); border-top-color: var(--line); }
.trade-war-line.muted { color: var(--muted-2); }
.trade-war-line.accent { color: var(--purple); }
.trade-war-subtitle { color: var(--cyan); }
.trade-war-meter-bar {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line);
}
.trade-war-meter-bar div {
  background: linear-gradient(90deg, var(--gold), var(--red));
}
.trade-hub-toast-float {
  background: var(--panel);
  border: 1px solid rgba(233, 168, 60, 0.45);
  color: var(--gold);
  border-radius: 12px;
  box-shadow: var(--shadow);
}
.trade-hub-team-logo-fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--panel-3);
  border: 1px solid var(--line);
  color: var(--text);
  font-weight: 1000;
  font-size: 28px;
}
.trade-ret-badge { font-size: 9px; color: var(--gold); margin-top: 2px; font-weight: 800; }
.trade-ret-btn,
.trade-hub-ret-btn {
  margin-top: 4px;
  font-size: 8px;
  padding: 4px 8px;
  background: var(--panel-3);
  border: 1px solid var(--line-2);
  border-radius: 8px;
  color: var(--text);
  cursor: pointer;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}
.trade-ret-btn:hover,
.trade-hub-ret-btn:hover {
  transform: translateY(-1px);
  border-color: var(--gold);
  background: var(--gold-soft);
  color: var(--gold);
}
.trade-hub-value-track {
  position: relative;
  height: 10px;
  overflow: hidden;
}
.trade-hub-value-center {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 2px;
  transform: translateX(-50%);
  z-index: 2;
}
.trade-hub-value-fill {
  height: 100%;
  transition: width 0.35s ease, background 0.25s ease;
}
.trade-pool-clause {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.08em;
  padding: 2px 5px;
  border-radius: 4px;
  border: 1px solid rgba(233, 168, 60, 0.45);
  color: var(--gold);
  background: var(--gold-soft);
}
.trade-pool-clause-nmc { border-color: rgba(255, 90, 90, 0.5); color: #ffb4b4; background: rgba(255, 60, 60, 0.12); }
.trade-pool-clause-ntc { border-color: rgba(233, 168, 60, 0.55); }
.trade-pool-clause-mntc { border-color: rgba(140, 180, 255, 0.45); color: #b8d4ff; }
.trade-war-meta { opacity: 0.55; font-size: 10px; }
.trade-hub-cap-sub.warn { color: var(--gold); }
.trade-hub-cap-sub.muted { opacity: 0.72; font-size: 10px; }
.trade-cap-mgmt-row.warn strong { color: var(--gold); }
.trade-hub-propose-btn {
  display: block;
  width: min(680px, 92%);
  min-height: 48px;
  margin: 4px auto 0;
  padding: 14px 22px;
  font-size: clamp(14px, 1.3vw, 20px);
  font-weight: 1000;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #07111a;
  border: 1px solid rgba(255, 222, 142, 0.65);
  border-radius: 18px;
  background:
    radial-gradient(circle at 20% 0%, rgba(255,255,255,0.55), transparent 28%),
    linear-gradient(180deg, #ffd166, #e9a83c 52%, #b97818);
  box-shadow:
    0 20px 48px rgba(233, 168, 60, 0.28),
    0 0 32px rgba(233, 168, 60, 0.14),
    inset 0 1px 0 rgba(255,255,255,0.55);
  cursor: pointer;
  transition:
    transform 0.18s ease,
    filter 0.18s ease,
    box-shadow 0.18s ease,
    opacity 0.18s ease;
}
.trade-hub-propose-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  filter: brightness(1.06);
  box-shadow:
    0 24px 58px rgba(233, 168, 60, 0.34),
    0 0 42px rgba(233, 168, 60, 0.22),
    inset 0 1px 0 rgba(255,255,255,0.6);
}
.trade-hub-propose-btn:active:not(:disabled) {
  transform: translateY(0);
}
.trade-hub-propose-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.trade-hub-propose-btn.ready,
.trade-hub-propose-btn.accepted {
  background:
    radial-gradient(circle at 20% 0%, rgba(255,255,255,0.45), transparent 28%),
    linear-gradient(180deg, #7ae7ad, #52df94 55%, #25975d);
  border-color: rgba(122, 231, 173, 0.7);
  color: #03130c;
  box-shadow:
    0 20px 50px rgba(82, 223, 148, 0.24),
    0 0 36px rgba(82, 223, 148, 0.15);
}
.trade-hub-propose-btn.rejected {
  background:
    radial-gradient(circle at 20% 0%, rgba(255,255,255,0.35), transparent 28%),
    linear-gradient(180deg, #ff9aa3, #ff606d 55%, #bc2634);
  border-color: rgba(255, 96, 109, 0.65);
  color: #fff;
  box-shadow:
    0 20px 50px rgba(255, 96, 109, 0.24),
    0 0 36px rgba(255, 96, 109, 0.12);
}
.trade-hub-feedback {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 12px;
  max-width: 820px;
  margin-left: auto;
  margin-right: auto;
}
.trade-hub-tag {
  font-size: 10px;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--panel-3);
  border: 1px solid var(--line);
  line-height: 1.3;
  max-width: 100%;
  font-weight: 700;
}
.trade-hub-tag-cap { border-color: rgba(255, 96, 109, 0.45); color: var(--red); background: var(--red-soft); }
.trade-hub-tag-clause { border-color: rgba(233, 168, 60, 0.45); color: var(--gold); background: var(--gold-soft); }
.trade-hub-tag-value { border-color: rgba(201, 146, 255, 0.45); color: var(--purple); background: var(--purple-soft); }
.trade-hub-tag-pick { border-color: rgba(19, 216, 231, 0.45); color: var(--cyan); background: var(--cyan-soft); }
.trade-hub-tag-more {
  font-size: 10px;
  background: transparent;
  border: 1px dashed var(--line-2);
  color: var(--muted);
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}
.trade-hub-tag-more:hover {
  transform: translateY(-1px);
  border-color: var(--line-strong);
  color: var(--text);
}
.trade-hub-ret-overlay {
  position: fixed;
  inset: 0;
  z-index: 60;
  background: rgba(2, 10, 17, 0.78);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
}
.trade-hub-ret-menu {
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.98), rgba(5, 17, 27, 0.98)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line-2);
  border-radius: 16px;
  padding: 16px;
  min-width: 200px;
  box-shadow: var(--shadow);
}
.trade-hub-ret-title {
  font-size: 10px;
  letter-spacing: 0.15em;
  margin-bottom: 12px;
  color: var(--cyan);
  font-weight: 900;
  text-transform: uppercase;
}
.trade-hub-ret-menu button {
  display: block;
  width: 100%;
  margin-bottom: 8px;
  padding: 10px 12px;
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 12px;
  color: var(--text);
  cursor: pointer;
  font-weight: 900;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}
.trade-hub-ret-menu button:hover {
  transform: translateY(-1px);
  border-color: var(--line-strong);
  background: var(--cyan-soft);
  color: var(--cyan);
}
.trade-hub-loading {
  position: relative;
  z-index: 2;
  min-height: 50vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  font-size: 16px;
  font-weight: 1000;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text);
  animation: trade-loading-pulse 1.4s ease infinite;
}
@keyframes trade-loading-pulse {
  0%, 100% { opacity: 0.72; }
  50% { opacity: 1; color: var(--cyan); }
}
.trade-hub-empty-card {
  max-width: 520px;
  margin: 40px auto;
  padding: 28px 24px;
  background:
    linear-gradient(180deg, rgba(9, 27, 40, 0.94), rgba(5, 17, 27, 0.94)),
    radial-gradient(circle at 66% 20%, rgba(19, 216, 231, 0.07), transparent 35%);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
  text-align: center;
}
.trade-hub-loading span {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: none;
  color: var(--muted);
  animation: none;
}
.trade-hub-warn-inline {
  text-align: center;
  font-size: 11px;
  color: var(--gold);
  margin-top: 8px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.trade-hub-block-reason {
  text-align: center;
  font-size: 11px;
  color: var(--gold);
  font-weight: 800;
  margin-top: 8px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 10px 14px;
  border: 1px solid rgba(233, 168, 60, 0.35);
  border-radius: 12px;
  background: var(--gold-soft);
  max-width: 720px;
  margin-left: auto;
  margin-right: auto;
}
.trade-hub-warning-reason {
  width: fit-content;
  max-width: min(420px, 90%);
  margin: 8px auto 0;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(233, 168, 60, 0.38);
  background: rgba(233, 168, 60, 0.1);
  color: var(--gold);
  font-size: 11px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  text-align: center;
  opacity: 0.92;
}
.trade-technical-details {
  margin-top: 10px;
  border: 1px solid rgba(255, 96, 109, 0.22);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255, 96, 109, 0.06);
}
.trade-technical-details summary {
  cursor: pointer;
  color: var(--gold);
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-technical-details p {
  margin: 8px 0 0;
  color: rgba(220, 236, 244, 0.78);
  font-size: 11px;
  line-height: 1.35;
  word-break: break-word;
}
.trade-analysis-rink {
  padding: 8px 10px 7px;
  border-radius: 14px;
  border: 1px solid rgba(0, 216, 223, 0.18);
  background:
    radial-gradient(circle at 50% 50%, rgba(0, 216, 223, 0.08), transparent 48%),
    linear-gradient(180deg, rgba(9, 27, 40, 0.96), rgba(4, 13, 22, 0.96));
  box-shadow:
    0 14px 34px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.045);
}
.trade-analysis-rink-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}
.trade-analysis-review-btn {
  border: 1px solid rgba(0, 216, 223, 0.22);
  background: rgba(255, 255, 255, 0.045);
  color: rgba(230, 246, 252, 0.88);
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
}
.trade-analysis-review-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.trade-clear-value {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.trade-clear-value-head {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 8px;
}
.trade-clear-value-head span {
  color: rgba(128, 150, 168, 0.95);
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-clear-value-head span:last-child {
  text-align: right;
}
.trade-clear-value-head strong {
  padding: 5px 12px;
  border-radius: 999px;
  border: 1px solid rgba(0, 216, 223, 0.28);
  background: rgba(255, 255, 255, 0.055);
  color: #e9f7fb;
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  white-space: nowrap;
  box-shadow: 0 0 20px rgba(19, 216, 231, 0.08);
}
.trade-clear-value-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  align-items: stretch;
  gap: 8px;
}
.trade-analysis-headline {
  color: rgba(220, 236, 244, 0.92);
  font-size: 11px;
  font-weight: 1000;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.trade-pool-status-pill {
  flex-shrink: 0;
  align-self: center;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.trade-pool-status-pill.in-deal {
  color: var(--cyan);
  border: 1px solid rgba(0, 216, 223, 0.28);
  background: rgba(0, 216, 223, 0.08);
}
.trade-pool-status-pill.waived {
  color: #9ec9ff;
  border: 1px solid rgba(96, 170, 255, 0.45);
  background: rgba(96, 170, 255, 0.12);
}
.trade-pool-status-pill.locked {
  color: rgba(128, 150, 168, 0.95);
  border: 1px solid rgba(128, 150, 168, 0.22);
  background: rgba(255, 255, 255, 0.04);
}
.trade-clear-side {
  min-width: 0;
  padding: 7px 9px;
  border-radius: 12px;
  border: 1px solid rgba(0, 216, 223, 0.16);
  background:
    linear-gradient(180deg, rgba(12, 35, 52, 0.78), rgba(5, 17, 27, 0.86));
}
.trade-clear-side-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 5px;
}
.trade-clear-side-label span {
  color: rgba(128, 150, 168, 0.95);
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.trade-clear-side-label strong {
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: rgba(220, 236, 244, 0.9);
}
.trade-clear-track {
  height: 12px;
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(0, 216, 223, 0.18);
  background: rgba(255, 255, 255, 0.045);
  box-shadow: inset 0 1px 6px rgba(0, 0, 0, 0.42);
}
.trade-clear-fill {
  height: 100%;
  min-width: 8px;
  border-radius: 999px;
  transition: width 0.42s cubic-bezier(0.22, 1, 0.36, 1);
}
.trade-clear-fill-left {
  background: linear-gradient(90deg, #13d8e7, #8ab4ff);
  box-shadow: 0 0 22px rgba(19, 216, 231, 0.28);
}
.trade-clear-fill-right {
  background: linear-gradient(90deg, #e9a83c, #ffd166);
  box-shadow: 0 0 22px rgba(233, 168, 60, 0.28);
}
.trade-clear-side.higher {
  border-color: rgba(82, 223, 148, 0.5);
  box-shadow:
    0 0 24px rgba(82, 223, 148, 0.14),
    inset 0 0 18px rgba(82, 223, 148, 0.04);
}
.trade-clear-side.higher .trade-clear-side-label strong {
  color: var(--green);
}
.trade-clear-side.lower {
  opacity: 0.62;
}
.trade-clear-side.lower .trade-clear-side-label strong {
  color: rgba(128, 150, 168, 0.95);
}
.trade-clear-side.even {
  border-color: rgba(0, 216, 223, 0.38);
}
.trade-clear-side.even .trade-clear-side-label strong {
  color: var(--cyan);
}
.trade-clear-side.neutral {
  opacity: 0.72;
}
.trade-fan-vitriol {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(156, 218, 236, 0.12);
}
.trade-fan-vitriol-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 4px;
}
.trade-fan-vitriol-head span {
  color: rgba(128, 150, 168, 0.95);
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.trade-fan-vitriol-head strong {
  color: rgba(233, 247, 251, 0.92);
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.trade-fan-vitriol-track {
  height: 6px;
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(255, 96, 109, 0.18);
  background: rgba(255, 255, 255, 0.045);
  box-shadow: inset 0 1px 5px rgba(0, 0, 0, 0.38);
}
.trade-fan-vitriol-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #e9a83c, #ff606d);
  box-shadow: 0 0 18px rgba(255, 96, 109, 0.22);
  transition: width 0.42s cubic-bezier(0.22, 1, 0.36, 1);
}
.trade-fan-vitriol-low .trade-fan-vitriol-fill {
  background: linear-gradient(90deg, #52df94, #e9a83c);
}
.trade-fan-vitriol-warn .trade-fan-vitriol-fill {
  background: linear-gradient(90deg, #e9a83c, #ffb454);
}
.trade-fan-vitriol-mid .trade-fan-vitriol-fill,
.trade-fan-vitriol-high .trade-fan-vitriol-fill {
  background: linear-gradient(90deg, #e9a83c, #ff606d);
}
.trade-fan-reason-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
.trade-fan-reason-chip {
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid rgba(255, 180, 120, 0.35);
  color: rgba(255, 220, 200, 0.92);
  background: rgba(255, 120, 60, 0.08);
}
.trade-fan-toast-extra {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.trade-fan-toast-extra span {
  font-size: 9px;
  letter-spacing: 0.14em;
  color: var(--muted);
}
.trade-fan-toast-extra strong {
  font-size: 13px;
  color: #ff8f98;
}
.trade-fan-toast-extra small {
  font-size: 10px;
  color: rgba(255, 220, 224, 0.85);
}
.trade-hub-insight-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.trade-hub-report-panel,
.trade-hub-cap-panel,
.trade-hub-needs-panel,
.trade-hub-breakdown-panel,
.trade-hub-league-panel {
  padding: 16px;
}
.trade-hub-verdict {
  font-size: 20px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 8px;
  animation: trade-verdict-in 0.35s ease;
}
.trade-hub-verdict-good { color: var(--green); }
.trade-hub-verdict-warn { color: var(--gold); }
.trade-hub-verdict-bad { color: var(--red); }
.trade-hub-verdict-neutral { color: var(--muted); }
.trade-hub-explanation {
  font-size: 12px;
  color: var(--muted);
  line-height: 1.45;
  margin: 0 0 12px;
}
.trade-hub-report-stat {
  text-align: center;
  padding: 8px 6px;
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
}
.trade-hub-report-label {
  display: block;
  font-size: 8px;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin-bottom: 4px;
  text-transform: uppercase;
  font-weight: 800;
}
.trade-hub-report-value {
  font-size: 16px;
  font-weight: 1000;
  color: var(--text);
}
.trade-hub-report-value.pos { color: var(--green); }
.trade-hub-report-value.neg { color: var(--red); }
.trade-hub-counter-chip {
  font-size: 10px;
  color: rgba(233, 247, 251, 0.78);
  padding: 6px 0;
  border-top: 1px solid var(--line);
  line-height: 1.35;
}
.trade-hub-immersion-line {
  margin-top: 8px;
  font-size: 10px;
  color: var(--purple);
  letter-spacing: 0.04em;
}
.trade-hub-cap-card {
  padding: 12px;
  background: var(--panel-3);
  border-radius: 12px;
  border: 1px solid var(--line);
}
.trade-hub-cap-card.bad {
  border-color: rgba(255, 96, 109, 0.45);
  background: var(--red-soft);
  animation: trade-cap-pulse 1.5s ease infinite;
}
.trade-hub-cap-card-label {
  font-size: 8px;
  letter-spacing: 0.15em;
  color: var(--muted);
  text-transform: uppercase;
  font-weight: 800;
}
.trade-hub-cap-big {
  font-size: 18px;
  font-weight: 1000;
  margin: 6px 0;
  color: var(--text);
}
.trade-hub-cap-sub {
  font-size: 9px;
  color: var(--muted);
}
.trade-hub-cap-sub .pos { color: var(--green); }
.trade-hub-cap-sub .neg { color: var(--red); }
.trade-hub-chip {
  font-size: 9px;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--panel-3);
  border: 1px solid var(--line);
  letter-spacing: 0.06em;
  font-weight: 800;
  text-transform: uppercase;
}
.trade-hub-chip-need { border-color: rgba(201, 146, 255, 0.4); color: var(--purple); background: var(--purple-soft); }
.trade-hub-chip-window { border-color: rgba(19, 216, 231, 0.4); color: var(--cyan); background: var(--cyan-soft); }
.trade-hub-chip-market-hot { border-color: rgba(255, 96, 109, 0.45); color: var(--red); background: var(--red-soft); }
.trade-hub-chip-market-warm { border-color: rgba(233, 168, 60, 0.45); color: var(--gold); background: var(--gold-soft); }
.trade-hub-chip-market-cool { border-color: rgba(138, 180, 255, 0.4); color: var(--blue); background: var(--blue-soft); }
.trade-hub-needs-line {
  font-size: 10px;
  color: var(--muted);
  margin-top: 4px;
  line-height: 1.4;
}
.trade-hub-needs-line.ok { color: var(--green); }
.trade-hub-needs-line.warn { color: var(--gold); }
.trade-hub-breakdown-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  padding: 6px 8px;
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
}
.trade-hub-breakdown-row.in span:last-child { color: var(--green); font-weight: 800; }
.trade-hub-breakdown-row.out span:last-child { color: var(--red); font-weight: 800; }
.trade-hub-recent-line {
  font-size: 10px;
  color: var(--muted);
  padding: 4px 0;
  border-top: 1px solid var(--line);
}
.trade-hub-recent-line.muted { font-style: italic; color: var(--muted-2); }
.trade-hub-report-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.trade-hub-counteroffers { margin-top: 10px; }
.trade-hub-counter-title {
  font-size: 9px;
  letter-spacing: 0.15em;
  color: var(--muted);
  margin-bottom: 6px;
  text-transform: uppercase;
  font-weight: 800;
}
.trade-hub-breakdown-list { display: flex; flex-direction: column; gap: 6px; }
.trade-hub-cap-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.trade-hub-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.trade-hub-recent-trades { margin-top: 6px; }
@keyframes trade-verdict-in {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes trade-cap-pulse {
  0%, 100% { box-shadow: 0 0 0 rgba(255, 96, 109, 0); }
  50% { box-shadow: 0 0 12px rgba(255, 96, 109, 0.25); }
}
@media (max-width: 1200px) {
  .trade-hub-war-room-layout {
    grid-template-columns: 1fr;
  }
  .trade-team-panel { max-height: none; }
  .trade-war-room { grid-template-columns: 1fr; }
  .trade-metric-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 960px) {
  .trade-construction-grid {
    grid-template-columns: 1fr;
  }
  .trade-package-divider {
    padding: 4px 0;
    transform: rotate(90deg);
  }
  .trade-centre-cap-row { grid-template-columns: 1fr; }
  .trade-reaction-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .trade-team-ratings { grid-template-columns: repeat(2, 1fr); }
  .trade-team-rank-grid { grid-template-columns: repeat(2, 1fr); }
  .trade-metric-grid { grid-template-columns: repeat(2, 1fr); }
}

/* —— Overhaul: progressive disclosure —— */
.trade-hub-centre-focus {
  gap: 6px;
  justify-content: flex-start;
  padding: 4px 8px 6px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.trade-hub-centre-foot {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.trade-hub-propose-wrap {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 6px;
}
.trade-hub-propose-wrap .trade-outcome-badge {
  align-self: center;
}
.trade-hub-propose-btn.soft-blocked {
  border-color: rgba(255, 180, 90, 0.45);
  box-shadow: inset 0 0 0 1px rgba(255, 180, 90, 0.18);
}
.trade-hub-centre-foot .trade-hub-block-reason {
  margin-top: 2px;
  padding: 8px 12px;
  font-size: 10px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  line-height: 1.35;
}
.trade-hub-centre-foot .trade-hub-block-reason strong {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 9px;
  color: rgba(210, 228, 240, 0.78);
}
.trade-hub-centre-foot .trade-hub-block-reason em {
  font-style: normal;
  color: rgba(160, 190, 210, 0.88);
  font-size: 9px;
}
.trade-hub-centre-foot .trade-hub-block-reason.tone-bad {
  border-color: rgba(255, 120, 100, 0.35);
}
.trade-hub-centre-foot .trade-hub-block-reason.tone-warn {
  border-color: rgba(255, 180, 90, 0.35);
}
.trade-clear-value-meta {
  text-align: center;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(150, 176, 196, 0.95);
}
.trade-block-detail {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(255, 120, 100, 0.28);
  background: rgba(40, 12, 12, 0.35);
  font-size: 10px;
  line-height: 1.35;
}
.trade-block-detail.tone-warn {
  border-color: rgba(255, 180, 90, 0.3);
  background: rgba(40, 28, 8, 0.35);
}
.trade-block-detail strong {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 9px;
}
.trade-block-detail em {
  font-style: normal;
  opacity: 0.85;
}
.trade-pool-season-line {
  margin-top: 3px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: rgba(140, 168, 188, 0.92);
}
.trade-pool-row.in-deal-row {
  opacity: 0.72;
  background: rgba(0, 216, 223, 0.07);
  border-color: rgba(0, 216, 223, 0.22);
  box-shadow: inset 0 0 0 1px rgba(0, 216, 223, 0.08);
  cursor: default;
}
.trade-intel-hero-cap {
  display: block;
  margin-top: 4px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(160, 190, 210, 0.95);
}
.trade-intel-hero-cap.ok { color: var(--green); }
.trade-intel-hero-cap.warn { color: #ffb45a; }
.trade-intel-hero-cap.bad { color: #ff9a6a; }
.trade-outcome-overpay,
.trade-outcome-warn {
  color: #ffb45a;
  border-color: rgba(255, 180, 90, 0.4);
  background: rgba(255, 180, 90, 0.1);
}
.trade-team-browser {
  padding: 12px;
  gap: 8px;
}
.trade-team-compact-card {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: flex-start;
  gap: 10px 14px;
  width: 100%;
  min-height: 108px;
  max-height: 118px;
  padding: 10px 12px 12px;
  background: linear-gradient(165deg, rgba(12, 28, 48, 0.95) 0%, rgba(8, 18, 32, 0.98) 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  cursor: pointer;
  color: inherit;
  text-align: left;
  overflow: visible;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.trade-team-compact-card:hover {
  border-color: var(--line-strong);
  box-shadow: 0 0 18px rgba(19, 216, 231, 0.1);
}
.trade-team-logo-lifted {
  margin-top: -14px;
  flex-shrink: 0;
  filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.45));
}
.trade-team-logo-lifted .trade-team-logo-img,
.trade-team-logo-lifted .trade-team-logo-fallback {
  width: 80px !important;
  height: 80px !important;
}
.trade-team-mainline {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 2px;
}
.trade-team-abbr-big {
  font-size: 17px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text);
  line-height: 1;
}
.trade-team-meta-strip {
  display: flex;
  align-items: center;
  gap: 12px;
}
.trade-team-meta-col {
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 0;
}
.trade-team-ovr-ring {
  --ovr-pct: 0%;
  position: relative;
  width: 58px;
  height: 58px;
  flex-shrink: 0;
  border-radius: 50%;
  background: conic-gradient(
    var(--ovr-ring-color, var(--cyan)) 0 var(--ovr-pct),
    rgba(255, 255, 255, 0.08) var(--ovr-pct) 100%
  );
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}
.trade-team-ovr-ring-fill {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  pointer-events: none;
}
.trade-team-ovr-ring-inner {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: rgba(6, 14, 26, 0.92);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  line-height: 1;
}
.trade-team-ovr-number {
  font-size: 18px;
  font-weight: 900;
  color: var(--text);
}
.trade-team-ovr-label {
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.14em;
  color: var(--muted-2);
  margin-top: 2px;
}
.trade-team-ovr-elite {
  --ovr-ring-color: var(--gold);
  box-shadow: 0 0 14px rgba(255, 196, 77, 0.25);
}
.trade-team-ovr-strong { --ovr-ring-color: #2dd4bf; }
.trade-team-ovr-normal { --ovr-ring-color: var(--cyan); }
.trade-team-ovr-muted { --ovr-ring-color: rgba(148, 163, 184, 0.85); }
.trade-team-ovr-warn { --ovr-ring-color: rgba(248, 113, 113, 0.75); }
.trade-team-cap-mini {
  font-size: 14px;
  font-weight: 800;
  letter-spacing: 0.02em;
  line-height: 1.1;
}
.trade-team-cap-mini.ok { color: var(--green); }
.trade-team-cap-mini.bad { color: var(--red); }
.trade-team-status-pill {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--cyan);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 140px;
  line-height: 1.15;
}
.trade-team-po-pill {
  position: relative;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  color: var(--muted);
  padding-bottom: 4px;
  line-height: 1.15;
}
.trade-team-po-meter {
  position: absolute;
  left: 0;
  bottom: 0;
  height: 3px;
  max-width: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--green));
  opacity: 0.9;
}
.trade-team-compact-name {
  font-size: 14px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.trade-team-compact-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 10px;
  color: var(--muted);
  margin-top: 4px;
  font-weight: 700;
}
.trade-team-compact-meta .ok { color: var(--green); }
.trade-team-compact-meta .bad { color: var(--red); }
.trade-team-interest-compact {
  font-size: 10px;
  text-align: center;
  color: var(--purple);
  font-weight: 800;
  letter-spacing: 0.06em;
}
.trade-pool-compact { grid-template-columns: auto 1fr auto; padding: 6px 8px; }
.trade-pool-compact-stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 3px;
}
.trade-pool-mini-tag {
  font-size: 8px;
  font-weight: 800;
  padding: 2px 5px;
  border-radius: 4px;
  border: 1px solid var(--line);
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.trade-pool-list { max-height: min(240px, 38vh); }
.trade-compare-strip { margin: 12px 0; }
.trade-compare-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  font-size: 12px;
  border-bottom: 1px solid var(--line);
}
.trade-compare-row strong { color: var(--cyan); }
.trade-pool-tabs button.active {
  background: var(--cyan-soft);
  border-color: var(--line-strong);
  color: var(--cyan);
}
.trade-analysis-clickable {
  width: 100%;
  text-align: left;
  cursor: pointer;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel);
  padding: 12px;
  color: inherit;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.trade-analysis-clickable:hover:not(:disabled) {
  border-color: var(--line-strong);
  box-shadow: 0 0 20px rgba(19, 216, 231, 0.08);
}
.trade-analysis-clickable:disabled { cursor: default; opacity: 0.7; }
.trade-analysis-hint {
  font-size: 9px;
  color: var(--muted-2);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.trade-outcome-badge {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.1em;
  padding: 5px 10px;
  border-radius: 8px;
  border: 1px solid var(--line);
}
.trade-outcome-good { background: var(--green-soft); color: var(--green); border-color: rgba(82,223,148,0.35); }
.trade-outcome-warn { background: var(--gold-soft); color: var(--gold); }
.trade-outcome-bad { background: var(--red-soft); color: var(--red); }
.trade-outcome-neutral { background: var(--panel-3); color: var(--muted); }
.trade-fan-badge {
  font-size: 10px;
  font-weight: 800;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
}
.trade-fan-badge-high { color: var(--green); border-color: rgba(82,223,148,0.35); }
.trade-fan-badge-mid { color: var(--gold); }
.trade-fan-badge-low { color: var(--red); }
.trade-fan-alert-compact {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 11px;
  font-weight: 800;
  color: var(--red);
  background: var(--red-soft);
  border: 1px solid rgba(255,96,109,0.35);
  border-radius: 10px;
  padding: 8px 12px;
}
.trade-fan-alert-compact button {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  font-size: 16px;
}
.trade-ctx-overlay,
.trade-drawer-overlay {
  position: fixed;
  inset: 0;
  z-index: 120;
  background: rgba(2, 10, 17, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.trade-ctx-menu {
  width: min(340px, 92vw);
  background: var(--panel-2);
  border: 1px solid var(--line-2);
  border-radius: 14px;
  padding: 14px;
  box-shadow: var(--shadow);
}
.trade-ctx-head {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.trade-ctx-head strong { display: block; font-size: 15px; }
.trade-ctx-head span { font-size: 11px; color: var(--muted); }
.trade-ctx-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 10px;
}
.trade-ctx-stats div {
  background: var(--panel-3);
  border-radius: 8px;
  padding: 6px 8px;
  font-size: 10px;
}
.trade-ctx-stats strong { display: block; font-size: 13px; margin-top: 2px; }
.trade-ctx-pick-meter {
  grid-column: 1 / -1;
  background: transparent !important;
  padding: 0 !important;
}
.trade-ctx-block {
  font-size: 11px;
  color: var(--red);
  margin-bottom: 8px;
  padding: 6px 8px;
  background: var(--red-soft);
  border-radius: 8px;
}
.trade-ctx-block.trade-ctx-waive-ok {
  color: #8fd7a8;
  background: rgba(56, 160, 96, 0.16);
}
.trade-ntc-waive-result .trade-ntc-waive-note {
  margin: 0 0 8px;
  font-size: 11px;
  color: var(--muted);
  line-height: 1.35;
}
.trade-ctx-actions button.primary {
  color: #dff7ff;
  border-color: rgba(96, 190, 255, 0.5);
  background: rgba(40, 110, 170, 0.35);
}
.trade-ctx-risk-tag {
  display: inline-block;
  font-size: 9px;
  margin: 0 4px 4px 0;
  padding: 3px 6px;
  border-radius: 6px;
  border: 1px solid var(--line);
  color: var(--gold);
}
.trade-ctx-actions {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
  margin-top: 10px;
}
.trade-ctx-actions button {
  padding: 8px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--panel-3);
  color: var(--text);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  cursor: pointer;
}
.trade-ctx-actions button.danger { color: var(--red); border-color: rgba(255,96,109,0.4); }
.trade-ctx-actions button.ghost { grid-column: span 2; }
.trade-drawer {
  width: min(68vw, 720px);
  max-height: 88vh;
  background: var(--panel-2);
  border: 1px solid var(--line-2);
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.trade-drawer-review { width: min(72vw, 800px); }
.trade-drawer.trade-review-fullscreen {
  width: calc(100vw - 24px);
  height: calc(100vh - 24px);
  max-width: none;
  max-height: none;
  border-radius: 14px;
  padding: 0;
  overflow: hidden;
}
.trade-review-overlay {
  position: fixed;
  inset: 0;
  z-index: 140;
  background: rgba(2, 10, 17, 0.98);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: stretch;
  justify-content: stretch;
  padding: 0;
}
.trade-review-shell {
  width: 100%;
  height: 100%;
  max-width: none;
  max-height: none;
  background: var(--panel);
  border: none;
  border-radius: 0;
  box-shadow: none;
  display: grid;
  gap: 6px;
  padding: 10px 14px 12px;
  overflow: hidden;
  color: var(--text);
  position: relative;
  box-sizing: border-box;
}
.trade-review-shell.trade-review-board {
  grid-template-rows: auto minmax(0, 1.05fr) auto minmax(0, 0.42fr) auto;
}
.trade-review-team-head {
  display: grid;
  grid-template-columns: 48px 1fr 48px;
  align-items: center;
  gap: 10px;
  padding: 2px 44px 4px;
}
.trade-review-anchor-value .trade-player-value-fill {
  background: linear-gradient(90deg, var(--cyan), #8ab4ff);
  box-shadow: 0 0 10px rgba(19, 216, 231, 0.25);
}
.trade-review-anchor-value.tier-elite .trade-player-value-fill,
.trade-review-anchor-value.tier-franchise .trade-player-value-fill {
  background: linear-gradient(90deg, var(--gold), #ffd166);
  box-shadow: 0 0 10px rgba(233, 168, 60, 0.25);
}
.trade-review-anchor-value.tier-top .trade-player-value-fill {
  background: linear-gradient(90deg, #13d8e7, var(--green));
}
.trade-review-panel-lead {
  margin: 0 0 4px;
  font-size: 11px;
  line-height: 1.3;
  color: rgba(233, 247, 251, 0.88);
}
.trade-review-package-side.anchor {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
  height: 100%;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.22);
  box-sizing: border-box;
}
.trade-review-side-label {
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--gold);
}
.trade-review-anchor-assets {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  min-height: 0;
}
.trade-review-anchor-asset {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--line-2);
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.28);
  color: inherit;
  cursor: pointer;
  text-align: left;
  padding: 10px;
}
.trade-review-anchor-asset.empty {
  display: grid;
  place-items: center;
  min-height: 120px;
  color: var(--muted);
  font-size: 14px;
  font-weight: 700;
  cursor: default;
}
.trade-review-anchor-asset.compact {
  padding: 6px 8px;
}
.trade-review-anchor-asset.protected {
  border-color: rgba(255, 96, 109, 0.55);
  background: rgba(255, 96, 109, 0.08);
}
.trade-review-anchor-player-row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: stretch;
}
.trade-review-anchor-shot {
  flex-shrink: 0;
}
.trade-review-anchor-shot .trade-review-anchor-headshot,
.trade-review-anchor-shot .player-headshot {
  --size: 72px;
}
.trade-review-anchor-asset.compact .trade-review-anchor-shot .player-headshot {
  --size: 48px;
}
.trade-review-anchor-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.trade-review-anchor-name-row {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
}
.trade-review-anchor-name {
  font-size: clamp(15px, 1.4vw, 20px);
  font-weight: 900;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-anchor-asset.compact .trade-review-anchor-name {
  font-size: 14px;
}
.trade-review-anchor-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 700;
  color: rgba(233, 247, 251, 0.92);
}
.trade-review-anchor-asset.compact .trade-review-anchor-meta {
  font-size: 11px;
  gap: 6px;
}
.trade-review-anchor-ovr {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 56px;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid rgba(19, 216, 231, 0.35);
  background: rgba(19, 216, 231, 0.08);
}
.trade-review-anchor-ovr span {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: var(--muted);
}
.trade-review-anchor-ovr strong {
  font-size: clamp(28px, 3vw, 40px);
  font-weight: 1000;
  line-height: 1;
  color: var(--cyan);
}
.trade-review-anchor-asset.compact .trade-review-anchor-ovr strong {
  font-size: 24px;
}
.trade-review-anchor-value {
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin-top: 2px;
}
.trade-review-anchor-value-label {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-anchor-value .trade-player-value-track {
  height: 7px;
}
.trade-review-anchor-pick-top {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 6px;
}
.trade-review-anchor-pick-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.trade-review-anchor-pick-main strong {
  font-size: 16px;
  font-weight: 900;
  color: var(--text);
}
.trade-review-anchor-pick-main span {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
}
.trade-review-result-word {
  font-size: clamp(32px, 4vw, 52px) !important;
  font-weight: 1000;
  letter-spacing: 0.12em;
  line-height: 1;
}
.trade-review-headline-cause {
  margin: 6px 0 0;
  font-size: clamp(14px, 1.5vw, 18px);
  font-weight: 700;
  line-height: 1.25;
  color: var(--text);
  max-width: 36ch;
}
.trade-review-blocker-chip {
  margin-top: 8px;
  display: inline-block;
  font-size: 11px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid rgba(255, 96, 109, 0.5);
  background: rgba(255, 96, 109, 0.12);
  color: var(--red);
}
.trade-review-verdict-core.good .trade-review-blocker-chip {
  border-color: rgba(82, 223, 148, 0.5);
  background: rgba(82, 223, 148, 0.12);
  color: var(--green);
}
.trade-review-fix-actions.large {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}
.trade-review-fix-btn.large {
  min-height: 64px;
  padding: 10px 12px;
  border-radius: 10px;
}
.trade-review-fix-btn.large strong {
  font-size: 13px;
  letter-spacing: 0.08em;
}
.trade-review-fix-btn.large span {
  font-size: 11px;
  line-height: 1.25;
}
.trade-review-next-step {
  padding: 10px 12px;
  border-radius: 10px;
  border: 2px solid rgba(19, 216, 231, 0.45);
  background: linear-gradient(180deg, rgba(19, 216, 231, 0.14), rgba(19, 216, 231, 0.04));
  text-align: left;
}
.trade-review-next-step-label {
  display: block;
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--cyan);
  margin-bottom: 4px;
}
.trade-review-next-step p {
  margin: 0;
  font-size: 14px;
  font-weight: 800;
  line-height: 1.3;
  color: var(--text);
}
.trade-review-dual-cap {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 4px;
  text-align: left;
}
.trade-review-dual-cap-row {
  display: grid;
  grid-template-columns: 36px 1fr auto;
  gap: 6px;
  align-items: center;
}
.trade-review-dual-cap-team {
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.trade-review-dual-cap-flow {
  font-size: clamp(12px, 1.2vw, 15px);
  font-weight: 1000;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 4px;
}
.trade-review-dual-cap-flow em {
  font-style: normal;
  color: var(--muted);
  font-size: 12px;
}
.trade-review-dual-cap-delta {
  font-size: 11px;
  font-weight: 900;
}
.trade-review-dual-cap-delta.pos { color: var(--green); }
.trade-review-dual-cap-delta.neg { color: var(--red); }
.trade-review-negot-context.compact {
  gap: 8px;
  align-content: start;
}
.trade-review-context-panel.compact {
  padding: 6px 8px;
  max-height: none;
}
.trade-review-context-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 2px;
}
.trade-review-context-head h3 {
  margin: 0;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gold);
}
.trade-review-panel-empty {
  margin: 4px 0;
  font-size: 11px;
  color: var(--muted);
}
.trade-review-context-row.anchor {
  grid-template-columns: auto minmax(0, 1fr) auto;
  min-height: 32px;
  padding: 4px 0;
}
.trade-review-context-row.lock.anchor {
  grid-template-columns: auto minmax(0, 1fr) auto;
}
.trade-review-want-tag,
.trade-review-lock-tag {
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.08em;
  padding: 4px 6px;
  border-radius: 5px;
  border: 1px solid var(--line);
  background: rgba(0, 0, 0, 0.25);
  color: var(--gold);
  white-space: nowrap;
  min-width: 68px;
  text-align: center;
}
.trade-review-lock-tag {
  color: var(--red);
  border-color: rgba(255, 96, 109, 0.35);
  background: rgba(255, 96, 109, 0.08);
}
.trade-review-context-ovr {
  font-size: 16px;
  font-weight: 1000;
  color: var(--cyan);
  min-width: 28px;
  text-align: right;
}
.trade-review-pos-badge.sm {
  min-width: 20px;
  height: 18px;
  font-size: 9px;
  margin-right: 4px;
}
.trade-review-tile-main {
  font-size: clamp(13px, 1.2vw, 16px);
}
.trade-review-cap-verdict {
  margin-top: 6px;
  font-size: 13px;
  font-weight: 1000;
}
.trade-review-fan-subject {
  font-size: 12px;
  margin-top: 6px;
}
.trade-review-btn.adjust-primary {
  font-size: 13px;
  min-height: 48px;
  box-shadow: 0 0 20px rgba(19, 216, 231, 0.2);
}
.trade-review-btn-block-reason {
  font-size: 11px;
}
.trade-review-title-block {
  text-align: center;
}
.trade-review-title-block h2 {
  margin: 0 0 4px;
  font-size: clamp(17px, 2vw, 22px);
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text);
}
.trade-review-team-matchup {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.trade-review-team-matchup strong {
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: var(--muted);
}
.trade-review-team-matchup span {
  color: var(--muted-2);
  font-size: 12px;
}
.trade-review-main-sub {
  margin: 2px 0 0;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.3;
  color: var(--gold);
}
.trade-review-verdict-chip.primary {
  border-color: rgba(255, 96, 109, 0.55);
  color: var(--red);
  background: var(--red-soft);
}
.trade-review-icon-tile.primary-metric {
  border-color: rgba(255, 96, 109, 0.55);
  background: rgba(255, 96, 109, 0.1);
  box-shadow: 0 0 0 1px rgba(255, 96, 109, 0.2);
}
.trade-review-icon-tile.primary-metric.cap.bad {
  border-color: rgba(255, 96, 109, 0.7);
  background: rgba(255, 96, 109, 0.16);
}
.trade-review-icon-tile.secondary-metric {
  opacity: 0.88;
}
.trade-review-cap-flow {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 4px;
  align-items: center;
  margin-top: 6px;
}
.trade-review-cap-flow .cap-step {
  display: flex;
  flex-direction: column;
  gap: 2px;
  text-align: center;
}
.trade-review-cap-flow .cap-step em {
  font-size: 8px;
  font-style: normal;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-cap-flow .cap-step strong {
  font-size: 13px;
  font-weight: 1000;
  color: var(--text);
}
.trade-review-cap-flow .cap-step.after.bad strong {
  color: var(--red);
  font-size: 15px;
}
.trade-review-cap-flow .cap-arrow {
  color: var(--muted);
  font-size: 12px;
}
.trade-review-cap-verdict {
  margin-top: 5px;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  text-align: center;
}
.trade-review-cap-verdict.bad {
  color: var(--red);
  font-size: 12px;
}
.trade-review-cap-verdict.good {
  color: var(--green);
}
.trade-review-tile-detail,
.trade-review-fan-subject {
  display: block;
  margin-top: 4px;
  font-size: 10px;
  font-weight: 600;
  line-height: 1.25;
  color: var(--muted);
}
.trade-review-fan-subject {
  font-size: 11px;
  color: rgba(233, 247, 251, 0.88);
}
.trade-review-fix-btn.best {
  border-color: rgba(19, 216, 231, 0.65);
  background: rgba(19, 216, 231, 0.14);
}
.trade-review-package-side.compact {
  padding: 4px 6px;
  gap: 4px;
  align-self: start;
  height: auto;
  min-height: 0;
}
.trade-review-side-head {
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: flex-start;
  padding-bottom: 2px;
}
.trade-review-side-label {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text);
}
.trade-review-mini-assets {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.trade-review-mini-asset.compact.horizontal {
  width: 100%;
  box-sizing: border-box;
}
.trade-review-mini-asset.player.compact.horizontal {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(0,0,0,0.2);
  text-align: left;
  cursor: pointer;
  color: inherit;
}
.trade-review-player-shot-wrap {
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.trade-review-player-shot-wrap .trade-review-mini-headshot {
  width: 38px !important;
  height: 38px !important;
}
.trade-review-mini-top {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  width: 100%;
}
.trade-review-pos-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 20px;
  padding: 0 5px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}
.trade-review-mini-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 11px;
  font-weight: 700;
  color: rgba(233, 247, 251, 0.9);
  margin-top: 2px;
}
.trade-review-mini-asset.pick.compact.horizontal {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  min-height: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(0,0,0,0.2);
  cursor: pointer;
  color: inherit;
  text-align: left;
}
.trade-review-pick-icon-sm.trade-pick-icon {
  width: 34px;
  height: 40px;
  min-height: 0;
}
.trade-review-pick-icon-sm .trade-pick-icon-round {
  font-size: 14px;
}
.trade-review-pick-icon-sm .trade-pick-icon-year {
  font-size: 9px;
}
.trade-review-pick-body {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.trade-review-pick-meta {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  line-height: 1.2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-pick-origin {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  flex-shrink: 0;
}
.trade-review-pick-origin span {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.trade-review-pick-own-fallback {
  font-size: 9px;
  font-weight: 800;
  color: var(--muted);
}
.trade-review-mini-name {
  font-size: 13px;
  font-weight: 800;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}
.trade-review-ovr-badge {
  margin-left: auto;
  flex-shrink: 0;
}
.trade-review-board-main {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 1.15fr) minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
}
.trade-review-verdict-stack {
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: center;
}
.trade-review-verdict-core {
  min-height: 0;
  padding: 14px 16px;
}
.trade-review-icon-grid-4 {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  align-items: stretch;
  gap: 8px;
  padding-top: 0;
}
.trade-review-icon-tile {
  min-height: 96px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  text-align: center;
  padding: 10px 10px 8px;
  box-sizing: border-box;
}
.trade-review-tile-kicker {
  font-size: 10px;
}
.trade-review-board {
  position: relative;
  height: 100%;
  display: grid;
  grid-template-rows: auto minmax(0, 1.05fr) auto minmax(0, 0.42fr) auto;
  gap: 6px;
  padding: 0;
  overflow: hidden;
  box-sizing: border-box;
}
.trade-review-verdict-guidance {
  flex: 1;
  min-height: 48px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px dashed rgba(19, 216, 231, 0.28);
  background: rgba(19, 216, 231, 0.05);
  text-align: left;
}
.trade-review-guidance-label {
  display: block;
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--cyan);
  margin-bottom: 4px;
}
.trade-review-verdict-guidance p {
  margin: 0;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.3;
  color: var(--text);
}
.trade-review-guidance-note {
  margin-top: 4px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--gold) !important;
}
.trade-review-icon-grid-4 {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  align-items: stretch;
  gap: 6px;
  padding-top: 0;
}
.trade-review-icon-tile {
  min-height: 102px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  text-align: center;
  padding: 8px 8px 6px;
  box-sizing: border-box;
}
.trade-review-tile-kicker {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
.trade-review-tile-main {
  font-size: 13px;
  font-weight: 1000;
  line-height: 1.2;
  color: var(--text);
  margin: 0 0 4px;
}
.trade-review-tile-main-compact {
  font-size: 12px;
  line-height: 1.25;
}
.trade-review-expand-btn {
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid rgba(19, 216, 231, 0.45);
  background: rgba(19, 216, 231, 0.1);
  color: var(--cyan);
}
.trade-review-context-toggle:hover .trade-review-expand-btn {
  background: rgba(19, 216, 231, 0.2);
}
.trade-review-context-row.lock {
  grid-template-columns: 36px minmax(0, 1fr) auto;
  min-height: 34px;
}
.trade-review-lock-icon {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.06em;
  padding: 3px 6px;
  border-radius: 5px;
  border: 1px solid var(--line);
  background: rgba(0,0,0,0.25);
  color: var(--gold);
  text-align: center;
}
.trade-review-btn {
  font-size: 12px;
}
.trade-review-btn-label {
  font-size: 12px;
  font-weight: 1000;
  letter-spacing: 0.1em;
}
.trade-review-btn.propose.blocked {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  min-height: 52px;
}
.trade-review-btn-block-reason {
  font-size: 10px;
  font-style: normal;
  font-weight: 700;
  line-height: 1.25;
  color: var(--red);
  max-width: 100%;
  text-align: center;
  padding: 0 6px;
}
.trade-review-negot-context {
  min-height: 0;
}
.trade-review-team-matchup {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}
.trade-review-team-matchup strong {
  font-size: clamp(16px, 2vw, 22px);
  font-weight: 1000;
  letter-spacing: 0.14em;
}
.trade-review-team-matchup span {
  color: var(--muted);
  font-size: 14px;
}
.trade-review-verdict-stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}
.trade-review-verdict-type {
  font-size: 9px;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-main-problem {
  margin: 4px 0 0;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.25;
  color: var(--text);
}
.trade-review-hard-banner {
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid rgba(255, 96, 109, 0.55);
  background: var(--red-soft);
  color: var(--red);
  font-size: 12px;
  font-weight: 800;
  text-align: center;
  letter-spacing: 0.04em;
}
.trade-review-fix-actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 4px;
}
.trade-review-fix-btn {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(19, 216, 231, 0.35);
  background: rgba(19, 216, 231, 0.08);
  color: var(--text);
  cursor: pointer;
  text-align: left;
}
.trade-review-fix-btn strong {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.04em;
  color: var(--cyan);
}
.trade-review-fix-btn span {
  font-size: 10px;
  color: var(--muted);
  line-height: 1.2;
}
.trade-review-fix-btn:hover {
  background: rgba(19, 216, 231, 0.16);
  border-color: rgba(19, 216, 231, 0.55);
}
.trade-review-fix-btn.counter {
  border-color: rgba(233, 168, 60, 0.45);
  background: rgba(233, 168, 60, 0.1);
}
.trade-review-fix-btn.counter strong { color: var(--gold); }
.trade-review-icon-grid-4 {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.trade-review-value-track {
  position: relative;
  height: 6px;
  margin-top: 6px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--red-soft), rgba(255,255,255,0.1) 45%, rgba(255,255,255,0.1) 55%, rgba(82,223,148,0.25));
  overflow: visible;
}
.trade-review-value-center {
  position: absolute;
  left: 50%;
  top: -2px;
  width: 2px;
  height: 10px;
  background: rgba(255,255,255,0.35);
  transform: translateX(-50%);
}
.trade-review-value-marker {
  position: absolute;
  top: 50%;
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--cyan);
  border: 1px solid rgba(255,255,255,0.5);
  transform: translate(-50%, -50%);
  box-shadow: 0 0 8px rgba(19,216,231,0.5);
}
.trade-review-cap-compare {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 4px;
  font-size: 10px;
  color: var(--muted);
  font-weight: 700;
}
.trade-review-fan-track.compact {
  height: 5px;
  margin-top: 4px;
}
.trade-review-fan-hint {
  display: block;
  margin-top: 3px;
  font-size: 9px;
  color: var(--muted);
  line-height: 1.2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-negot-context {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  min-height: 0;
}
.trade-review-context-panel {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 4px 6px;
  min-height: 0;
  overflow: hidden;
}
.trade-review-context-panel.alert {
  border-color: rgba(255, 96, 109, 0.45);
}
.trade-review-context-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 2px 0 4px;
  border: none;
  background: transparent;
  cursor: pointer;
  color: inherit;
}
.trade-review-context-toggle h3 {
  margin: 0;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text);
}
.trade-review-context-toggle span {
  font-size: 10px;
  font-weight: 800;
  color: var(--muted);
}
.trade-review-context-row {
  width: 100%;
  display: grid;
  grid-template-columns: auto 28px minmax(0, 1fr) auto;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  border: none;
  border-top: 1px solid var(--line);
  background: transparent;
  cursor: pointer;
  text-align: left;
  color: inherit;
}
.trade-review-context-row.lock {
  grid-template-columns: 20px minmax(0, 1fr) auto;
}
.trade-review-context-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.trade-review-context-row:first-of-type { border-top: none; }
.trade-review-context-row:hover { background: rgba(255,255,255,0.03); }
.trade-review-context-row.conflict {
  background: rgba(255, 96, 109, 0.08);
}
.trade-review-context-cat {
  font-size: 8px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--gold);
  min-width: 52px;
}
.trade-review-context-name {
  font-size: 12px;
  font-weight: 800;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-context-note {
  font-size: 10px;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-context-ovr {
  font-size: 13px;
  font-weight: 1000;
  font-style: normal;
  color: var(--cyan);
}
.trade-review-lock-icon {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.06em;
  padding: 3px 6px;
  border-radius: 5px;
  border: 1px solid var(--line);
  background: rgba(0,0,0,0.25);
  color: var(--gold);
  text-align: center;
  line-height: 1.2;
}
.trade-review-mini-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.trade-review-mini-warn {
  font-size: 9px;
  font-weight: 800;
  color: var(--red);
  letter-spacing: 0.04em;
}
.trade-review-mini-asset.protected {
  border-color: rgba(255, 96, 109, 0.55);
  background: rgba(255, 96, 109, 0.1);
}
.trade-review-mini-asset.player {
  border: none;
  background: rgba(0,0,0,0.18);
  cursor: pointer;
  text-align: left;
  color: inherit;
  width: 100%;
}
.trade-review-franchise-mark,
.trade-asset-franchise-mark {
  color: var(--gold);
  font-style: normal;
  margin-right: 3px;
}
.trade-review-footer-negotiate {
  grid-template-columns: 0.8fr 1.4fr 1fr;
}
.trade-review-btn.adjust-primary {
  background: linear-gradient(180deg, rgba(19, 216, 231, 0.35), rgba(19, 216, 231, 0.12));
  border-color: rgba(19, 216, 231, 0.55);
  color: var(--cyan);
  font-size: 11px;
}
.trade-review-btn.secondary {
  background: var(--panel-3);
  color: var(--muted);
}
.trade-review-btn.propose {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  background: var(--panel-3);
  border-color: var(--line);
  color: var(--muted);
}
.trade-review-btn.propose.ready {
  background: linear-gradient(180deg, rgba(82, 223, 148, 0.35), rgba(82, 223, 148, 0.12));
  border-color: rgba(82, 223, 148, 0.55);
  color: var(--green);
}
.trade-review-btn.propose em {
  font-size: 9px;
  font-style: normal;
  color: var(--red);
  font-weight: 700;
  line-height: 1.2;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-asset-protected-warn {
  margin-top: 4px;
  padding: 3px 6px;
  border-radius: 6px;
  background: rgba(255, 180, 70, 0.14);
  border: 1px solid rgba(255, 180, 70, 0.4);
  color: #ffc978;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.04em;
}
.trade-slot.protected-conflict {
  outline: 1px solid rgba(255, 180, 70, 0.4);
  border-radius: 10px;
}
.trade-slot.protected-conflict:has(.trade-asset-protected-warn) {
  outline-color: rgba(255, 180, 70, 0.45);
}
.trade-review-shell::-webkit-scrollbar,
.trade-review-context-panel::-webkit-scrollbar {
  width: 6px;
}
.trade-review-shell::-webkit-scrollbar-thumb,
.trade-review-context-panel::-webkit-scrollbar-thumb {
  background: rgba(19, 216, 231, 0.35);
  border-radius: 999px;
}
.trade-review-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-bottom: 2px;
  border-bottom: 1px solid var(--line);
}
.trade-review-topbar h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text);
}
.trade-review-topbar span {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.06em;
}
.trade-review-close-btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid rgba(19, 216, 231, 0.45);
  background: linear-gradient(180deg, rgba(19, 216, 231, 0.18), rgba(19, 216, 231, 0.06));
  color: var(--cyan);
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
}
.trade-review-close-btn:hover {
  background: rgba(19, 216, 231, 0.24);
}
.trade-review-hero {
  display: grid;
  grid-template-columns: 1fr minmax(220px, 0.9fr) 1fr;
  gap: 6px;
  align-items: stretch;
  min-height: 0;
}
.trade-review-asset-strip {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 0;
}
.trade-review-strip-label {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-strip-items {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-height: 0;
}
.trade-review-strip-card {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(0, 0, 0, 0.22);
  min-height: 44px;
}
.trade-review-strip-card.pick {
  grid-template-columns: 1fr;
  justify-items: center;
  color: var(--gold);
  font-weight: 800;
}
.trade-review-strip-meta {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.trade-review-strip-meta strong {
  font-size: 13px;
  font-weight: 800;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-strip-meta span {
  font-size: 12px;
  font-weight: 900;
  color: var(--cyan);
}
.trade-review-strip-empty {
  font-size: 11px;
  color: var(--muted);
}
.trade-review-result {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: center;
  justify-content: center;
}
.trade-review-result-kicker {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-result strong {
  font-size: 24px;
  font-weight: 1000;
  letter-spacing: 0.06em;
  line-height: 1;
}
.trade-review-result p {
  margin: 0;
  font-size: 12px;
  line-height: 1.3;
  color: rgba(233, 247, 251, 0.9);
}
.trade-review-result.good { border-color: rgba(82, 223, 148, 0.5); color: var(--green); }
.trade-review-result.warn { border-color: rgba(233, 168, 60, 0.5); color: var(--gold); }
.trade-review-result.bad { border-color: rgba(255, 96, 109, 0.55); color: var(--red); }
.trade-review-blocker-pills {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px;
}
.trade-review-blocker-pills span {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 7px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted-2);
  background: rgba(0, 0, 0, 0.2);
}
.trade-review-blocker-pills span.active {
  color: var(--text);
  border-color: rgba(255, 96, 109, 0.55);
  background: var(--red-soft);
}
.trade-review-deal-gap {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 8px;
}
.trade-review-deal-gap-labels {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: var(--muted);
}
.trade-review-deal-gap-labels strong {
  color: var(--text);
  font-size: 14px;
}
.trade-review-deal-gap-chip {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
}
.trade-review-deal-gap-chip.pos { color: var(--green); border-color: rgba(82, 223, 148, 0.4); }
.trade-review-deal-gap-chip.neg { color: var(--red); border-color: rgba(255, 96, 109, 0.4); }
.trade-review-deal-gap-track {
  display: flex;
  gap: 4px;
  height: 10px;
  margin-top: 6px;
  align-items: stretch;
}
.trade-review-deal-bar {
  border-radius: 999px;
  min-width: 4px;
  transition: width 0.25s ease;
}
.trade-review-deal-bar.give { background: linear-gradient(90deg, var(--red), #ff8f9d); }
.trade-review-deal-bar.get { background: linear-gradient(90deg, var(--green), #7dffc0); }
.trade-review-deal-gap-note {
  margin: 4px 0 0;
  font-size: 11px;
  color: var(--muted);
}
.trade-review-metrics {
  display: grid;
  grid-template-columns: 1.2fr 1fr 0.9fr;
  gap: 6px;
}
.trade-review-metric {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-height: 0;
}
.trade-review-metric-label {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-metric strong {
  font-size: 18px;
  font-weight: 1000;
  line-height: 1.1;
  color: var(--text);
}
.trade-review-metric em {
  font-size: 11px;
  font-style: normal;
  color: var(--muted);
  line-height: 1.25;
}
.trade-review-metric ul {
  margin: 2px 0 0;
  padding-left: 16px;
  font-size: 11px;
  color: rgba(233, 247, 251, 0.88);
}
.trade-review-metric.fan.bad strong { color: var(--red); }
.trade-review-metric.fan.warn strong { color: var(--gold); }
.trade-review-metric.fan.good strong { color: var(--green); }
.trade-review-metric.cap.good strong { color: var(--green); }
.trade-review-metric.cap.bad strong { color: var(--red); }
.trade-review-metric.gm strong { color: var(--purple); }
.trade-review-fan-scale {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 6px;
  align-items: center;
  font-size: 9px;
  color: var(--muted-2);
  text-transform: uppercase;
  font-weight: 800;
}
.trade-review-fan-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
  border: 1px solid var(--line);
}
.trade-review-fan-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--green), var(--gold), var(--red));
}
.trade-review-panels {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
  min-height: 0;
  overflow: hidden;
}
.trade-review-panel {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 0;
  overflow: auto;
}
.trade-review-panel h3 {
  margin: 0;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text);
}
.trade-review-panel-lead {
  margin: 0;
  font-size: 11px;
  line-height: 1.3;
  color: var(--muted);
}
.trade-review-player-row {
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  border-top: 1px solid var(--line);
}
.trade-review-player-row:first-of-type {
  border-top: none;
}
.trade-review-player-rank {
  font-size: 10px;
  font-weight: 900;
  color: var(--gold);
  min-width: 18px;
}
.trade-review-player-row-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.trade-review-player-row-text strong {
  font-size: 13px;
  font-weight: 800;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-player-row-text span {
  font-size: 10px;
  color: var(--muted);
  line-height: 1.2;
}
.trade-review-player-row-ovr {
  font-size: 14px;
  font-weight: 1000;
  font-style: normal;
  color: var(--cyan);
}
.trade-review-tag {
  display: inline-block;
  margin-top: 2px;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 7px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
}
.trade-review-fix {
  background: var(--panel-2);
  border: 1px solid rgba(19, 216, 231, 0.28);
  border-radius: 10px;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.trade-review-fix h3 {
  margin: 0;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--cyan);
}
.trade-review-fix-row {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 8px;
  font-size: 11px;
  padding-top: 3px;
  border-top: 1px solid var(--line);
}
.trade-review-fix-row:first-of-type { border-top: none; padding-top: 0; }
.trade-review-fix-row strong { color: var(--text); font-weight: 800; }
.trade-review-fix-row span { color: var(--muted); line-height: 1.3; }
.trade-review-fix-row.counter strong { color: var(--gold); }
.trade-review-warning {
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(255, 96, 109, 0.5);
  background: var(--red-soft);
  color: var(--red);
  font-size: 11px;
  font-weight: 700;
}
.trade-review-footer {
  display: grid;
  grid-template-columns: 1fr 1fr 1.2fr;
  gap: 6px;
  padding-top: 2px;
  border-top: 1px solid var(--line);
}
.trade-review-btn {
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--line);
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
}
.trade-review-btn.ghost {
  background: var(--panel-3);
  color: var(--text);
}
.trade-review-btn.primary {
  background: linear-gradient(180deg, rgba(82, 223, 148, 0.35), rgba(82, 223, 148, 0.12));
  border-color: rgba(82, 223, 148, 0.55);
  color: var(--green);
}
.trade-review-btn.primary.disabled,
.trade-review-btn.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  color: var(--muted);
  border-color: var(--line);
  background: var(--panel-3);
}
.trade-review-clean {
  position: relative;
  height: 100%;
  display: grid;
  grid-template-rows: auto 1fr auto;
  padding: 18px;
  overflow: hidden;
  box-sizing: border-box;
}
.trade-review-clean-head {
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 8px 48px 16px;
}
.trade-review-clean-head strong {
  font-size: clamp(28px, 3vw, 46px);
  font-weight: 1000;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.trade-review-x {
  position: absolute;
  top: 18px;
  right: 22px;
  z-index: 5;
  width: 44px;
  height: 44px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  color: var(--text);
  font-size: 26px;
  line-height: 1;
  cursor: pointer;
}
.trade-review-x:hover {
  background: rgba(255,255,255,0.12);
}
.trade-review-board {
  position: relative;
  height: 100%;
  display: grid;
  grid-template-rows: auto minmax(0, 0.24fr) minmax(0, 0.22fr) minmax(0, 1fr);
  gap: 3px;
  padding: 3px;
  overflow: hidden;
  box-sizing: border-box;
}
.trade-review-board-head {
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 0 36px;
}
.trade-review-board-head strong {
  font-size: clamp(18px, 2vw, 28px);
  font-weight: 1000;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text);
  line-height: 1.1;
}
.trade-review-board-main {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 0.82fr) minmax(300px, 1.36fr) minmax(0, 0.82fr);
  gap: 8px;
  align-items: start;
}
.trade-review-board-verdict {
  border-radius: 10px;
  display: grid;
  place-items: center;
  text-align: center;
  gap: 2px;
  padding: 3px;
  border: 1px solid var(--line);
  background: var(--panel-3);
}
.trade-review-board-verdict span {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: var(--muted);
  line-height: 1;
}
.trade-review-board-verdict strong {
  font-size: clamp(18px, 2.4vw, 30px);
  font-weight: 1000;
  letter-spacing: 0.06em;
  line-height: 1;
}
.trade-review-board-verdict.good {
  color: var(--green);
  border-color: rgba(82,223,148,0.46);
}
.trade-review-board-verdict.warn {
  color: var(--gold);
  border-color: rgba(233,168,60,0.48);
}
.trade-review-board-verdict.bad {
  color: var(--red);
  border-color: rgba(255,96,109,0.48);
}
.trade-review-board-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  min-height: 0;
}
.trade-review-insight-top-row,
.trade-review-insight-bottom-row {
  display: grid;
  gap: 3px;
  min-height: 0;
}
.trade-review-insight-top-row {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  align-items: stretch;
}
.trade-review-insight-bottom-row {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: stretch;
  height: 100%;
}
.trade-review-insight {
  min-height: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.trade-review-insight.spread {
  height: 100%;
}
.trade-review-insight .trade-hub-panel-title {
  font-size: 10px;
  margin: 0;
  line-height: 1.1;
}
.trade-review-insight-value {
  font-size: 14px;
  font-weight: 1000;
  line-height: 1.1;
  color: var(--text);
  margin: 0;
  letter-spacing: 0.04em;
}
.trade-review-insight-sub {
  font-size: 11px;
  line-height: 1.25;
  color: var(--muted);
  margin: 0;
}
.trade-review-insight-meter {
  height: 7px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
  margin: 0;
}
.trade-review-insight-meter-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--cyan);
  transition: width 0.25s ease;
}
.trade-review-insight.good .trade-review-insight-meter-fill { background: var(--green); }
.trade-review-insight.warn .trade-review-insight-meter-fill { background: var(--gold); }
.trade-review-insight.bad .trade-review-insight-meter-fill { background: var(--red); }
.trade-review-insight-value.good { color: var(--green); }
.trade-review-insight-value.warn { color: var(--gold); }
.trade-review-insight-value.bad { color: var(--red); }
.trade-review-insight-value.balance { color: var(--blue, #5a9fff); }
.trade-review-insight-value.gm { color: var(--purple, #c992ff); }
.trade-review-insight-value.ask { color: var(--gold); }
.trade-review-insight-value.no { color: var(--red); }
.trade-review-insight-list {
  list-style: disc;
  margin: 0;
  padding: 0 0 0 16px;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.trade-review-insight-list > li {
  font-size: 12px;
  font-weight: 600;
  line-height: 1.2;
  color: rgba(233, 247, 251, 0.9);
  margin: 0;
}
.trade-review-insight-player {
  list-style: none;
  margin-left: -16px;
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 6px;
  min-height: 34px;
  padding: 2px 0;
}
.trade-review-insight-player-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-review-insight-ovr {
  font-size: 14px;
  font-weight: 1000;
  color: var(--cyan);
  line-height: 1;
}
.trade-review-insight-headshot {
  flex-shrink: 0;
}
.trade-review-insight-chips {
  margin: 0;
  gap: 3px;
}
.trade-review-readout,
.trade-review-text-card {
  min-height: 96px;
  border-radius: 16px;
  padding: 12px 14px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.035);
  border-top: 2px solid rgba(255,255,255,0.12);
}
.trade-review-readout-head,
.trade-review-text-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
}
.trade-review-readout-head span,
.trade-review-text-head span {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.13em;
  color: var(--muted);
  text-transform: uppercase;
}
.trade-review-readout-head strong {
  font-size: 15px;
  font-weight: 1000;
  color: var(--text);
}
.trade-review-readout-sub,
.trade-review-text-summary {
  margin: 8px 0 0;
  font-size: 11px;
  line-height: 1.35;
  color: var(--muted);
}
.trade-review-readout-meter {
  height: 6px;
  border-radius: 999px;
  margin-top: 10px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}
.trade-review-readout-meter-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--cyan);
  transition: width 0.25s ease;
}
.trade-review-readout.good {
  border-top-color: rgba(82,223,148,0.55);
}
.trade-review-readout.good .trade-review-readout-meter-fill {
  background: var(--green);
}
.trade-review-readout.warn {
  border-top-color: rgba(233,168,60,0.55);
}
.trade-review-readout.warn .trade-review-readout-meter-fill {
  background: var(--gold);
}
.trade-review-readout.bad {
  border-top-color: rgba(255,96,109,0.55);
}
.trade-review-readout.bad .trade-review-readout-meter-fill {
  background: var(--red);
}
.trade-review-readout.balance {
  border-top-color: rgba(90,159,255,0.55);
}
.trade-review-readout.balance .trade-review-readout-meter-fill {
  background: var(--blue, #5a9fff);
}
.trade-review-readout.gm {
  border-top-color: rgba(201,146,255,0.55);
}
.trade-review-readout.gm .trade-review-readout-meter-fill {
  background: var(--purple, #c992ff);
}
.trade-review-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.trade-review-chip-row span {
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.06em;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(0,0,0,0.2);
  color: var(--text);
}
.trade-review-text-card.good {
  border-top-color: rgba(82,223,148,0.5);
}
.trade-review-text-card.bad {
  border-top-color: rgba(255,96,109,0.5);
}
.trade-review-text-card.ask {
  border-top-color: rgba(233,168,60,0.5);
}
.trade-review-text-card.no {
  border-top-color: rgba(255,96,109,0.38);
}
.trade-review-meter-card,
.trade-review-info-card,
.trade-review-cap-card {
  min-height: 96px;
  border-radius: 18px;
  padding: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
}
.trade-review-meter-head,
.trade-review-info-head,
.trade-review-cap-head,
.trade-review-cap-delta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: baseline;
}
.trade-review-meter-head span,
.trade-review-info-head span,
.trade-review-cap-head span,
.trade-review-cap-delta span {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
}
.trade-review-meter-head strong,
.trade-review-info-card strong,
.trade-review-cap-head strong,
.trade-review-cap-delta strong {
  font-size: 15px;
  font-weight: 1000;
  color: var(--text);
}
.trade-review-info-main {
  display: block;
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.3;
}
.trade-review-meter-track {
  height: 9px;
  border-radius: 999px;
  margin-top: 12px;
  background: rgba(255,255,255,0.09);
  overflow: hidden;
}
.trade-review-meter-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--cyan);
  box-shadow: 0 0 14px rgba(19,216,231,0.32);
  transition: width 0.25s ease;
}
.trade-review-meter-card.good .trade-review-meter-fill {
  background: var(--green);
  box-shadow: 0 0 14px rgba(82,223,148,0.28);
}
.trade-review-meter-card.warn .trade-review-meter-fill {
  background: var(--gold);
  box-shadow: 0 0 14px rgba(233,168,60,0.28);
}
.trade-review-meter-card.bad .trade-review-meter-fill {
  background: var(--red);
  box-shadow: 0 0 14px rgba(255,96,109,0.28);
}
.trade-review-meter-card.gm .trade-review-meter-fill {
  background: var(--purple, #c992ff);
  box-shadow: 0 0 14px rgba(201,146,255,0.28);
}
.trade-review-meter-card.balance .trade-review-meter-fill {
  background: var(--blue, #5a9fff);
  box-shadow: 0 0 14px rgba(90,159,255,0.28);
}
.trade-review-info-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}
.trade-review-info-chips span {
  padding: 5px 8px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.08em;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.18);
  color: var(--text);
}
.trade-review-info-card.good {
  border-color: rgba(82,223,148,0.38);
}
.trade-review-info-card.bad {
  border-color: rgba(255,96,109,0.38);
}
.trade-review-info-card.ask {
  border-color: rgba(233,168,60,0.38);
}
.trade-review-info-card.no {
  border-color: rgba(255,96,109,0.32);
}
.trade-review-cap-card.good {
  border-color: rgba(82,223,148,0.42);
  background: rgba(82,223,148,0.08);
}
.trade-review-cap-card.bad {
  border-color: rgba(255,96,109,0.42);
  background: rgba(255,96,109,0.09);
}
.trade-review-cap-delta {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.1);
}
.trade-review-cap-card.good .trade-review-cap-delta strong {
  color: var(--green);
}
.trade-review-cap-card.bad .trade-review-cap-delta strong {
  color: var(--red);
}
.trade-review-clean-main {
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(200px, 1fr) minmax(220px, 0.85fr) minmax(200px, 1fr);
  gap: 16px;
  align-items: stretch;
}
.trade-review-package-side {
  height: auto;
  min-height: 0;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: var(--panel-3);
  padding: 3px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.trade-review-package-side > span {
  text-align: center;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: var(--cyan);
  text-transform: uppercase;
  line-height: 1.1;
}
.trade-review-mini-assets {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 0 0 auto;
  min-height: 0;
}
.trade-review-pick-main {
  font-size: 12px;
  font-weight: 900;
  color: var(--gold);
  line-height: 1.1;
}
.trade-review-ovr-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 26px;
  padding: 2px 5px;
  border-radius: 6px;
  background: rgba(19, 216, 231, 0.16);
  border: 1px solid rgba(19, 216, 231, 0.4);
  font-size: 12px;
  font-weight: 1000;
  font-style: normal;
  color: var(--cyan);
  line-height: 1;
  flex-shrink: 0;
}
.trade-review-ovr-badge.sm {
  min-width: 24px;
  font-size: 11px;
}
.trade-review-context-panel.light {
  background: transparent;
  border-color: rgba(255,255,255,0.08);
}
.trade-review-context-row.compact {
  grid-template-columns: minmax(0, 1fr) auto;
  min-height: 32px;
  padding: 5px 0;
}
.trade-review-context-cat-inline {
  display: inline-block;
  margin-right: 6px;
  font-size: 8px;
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--gold);
  font-style: normal;
}
.trade-review-icon-tile > span:not(.trade-review-tile-kicker):not(.trade-review-tile-detail):not(.trade-review-fan-subject) {
  display: none;
}
.trade-review-icon-tile .icon {
  display: none;
}
.trade-review-mini-asset {
  min-height: 0;
  flex: 1;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(0,0,0,0.18);
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 6px;
  padding: 3px 6px;
  font-size: 12px;
  font-weight: 800;
}
.trade-review-mini-asset.pick {
  grid-template-columns: 1fr;
  justify-items: center;
  color: var(--gold);
}
.trade-review-mini-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text);
}
.trade-review-mini-ovr {
  font-size: 14px;
  font-weight: 1000;
  color: var(--cyan);
  line-height: 1;
}
.trade-review-mini-headshot {
  flex-shrink: 0;
}
.trade-review-mini-asset.pick {
  color: var(--gold);
  border-color: rgba(233,168,60,0.34);
  background: rgba(233,168,60,0.1);
}
.trade-review-mini-asset.overflow,
.trade-review-mini-asset.empty {
  grid-template-columns: 1fr;
  justify-items: center;
  color: var(--muted);
  font-size: 13px;
}
.trade-review-verdict-core {
  min-height: 120px;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 4px;
  border: 2px solid rgba(255,255,255,0.14);
  background: radial-gradient(circle at 50% 0%, rgba(19,216,231,0.14), rgba(255,255,255,0.035));
  padding: 10px 12px;
}
.trade-review-verdict-core strong {
  font-size: clamp(22px, 2.8vw, 36px);
  font-weight: 1000;
  letter-spacing: 0.1em;
  line-height: 1;
}
.trade-review-verdict-core.good {
  color: var(--green);
  border-color: rgba(82,223,148,0.48);
  background: radial-gradient(circle at 50% 0%, rgba(82,223,148,0.16), rgba(255,255,255,0.035));
}
.trade-review-verdict-core.warn {
  color: var(--gold);
  border-color: rgba(233,168,60,0.48);
}
.trade-review-verdict-core.bad {
  color: var(--red);
  border-color: rgba(255,96,109,0.48);
  background: radial-gradient(circle at 50% 0%, rgba(255,96,109,0.14), rgba(255,255,255,0.035));
}
.trade-review-verdict-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
}
.trade-review-verdict-chip {
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.12em;
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(0,0,0,0.25);
  color: var(--muted);
}
.trade-review-icon-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  padding-top: 16px;
}
.trade-review-icon-tile {
  min-height: 84px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  display: grid;
  place-items: center;
  text-align: center;
  padding: 10px 6px;
}
.trade-review-icon-tile .icon {
  font-size: 22px;
  line-height: 1;
}
.trade-review-icon-tile > span {
  display: block;
  margin-top: 4px;
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.13em;
  color: var(--muted);
}
.trade-review-icon-tile > strong {
  display: block;
  margin-top: 3px;
  font-size: 15px;
  font-weight: 1000;
  color: var(--text);
}
.trade-review-fan-chips {
  display: flex;
  gap: 4px;
  justify-content: center;
  margin-top: 2px;
}
.trade-review-fan-chip {
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  padding: 2px 5px;
  border-radius: 4px;
  border: 1px solid rgba(255,255,255,0.14);
  color: var(--muted);
}
.trade-review-icon-tile.balance {
  border-color: rgba(19,216,231,0.42);
  background: rgba(19,216,231,0.1);
}
.trade-review-icon-tile.balance > strong { color: var(--cyan); }
.trade-review-icon-tile.fans.good {
  border-color: rgba(82,223,148,0.42);
  background: rgba(82,223,148,0.1);
}
.trade-review-icon-tile.fans.good > strong { color: var(--green); }
.trade-review-icon-tile.fans.warn {
  border-color: rgba(233,168,60,0.42);
  background: rgba(233,168,60,0.12);
}
.trade-review-icon-tile.fans.warn > strong { color: var(--gold); }
.trade-review-icon-tile.fans.bad {
  border-color: rgba(255,96,109,0.42);
  background: rgba(255,96,109,0.12);
}
.trade-review-icon-tile.fans.bad > strong { color: var(--red); }
.trade-review-icon-tile.cap.good {
  border-color: rgba(82,223,148,0.42);
  background: rgba(82,223,148,0.1);
}
.trade-review-icon-tile.cap.good > strong { color: var(--green); }
.trade-review-icon-tile.cap.bad,
.trade-review-icon-tile.risk.bad {
  border-color: rgba(255,96,109,0.42);
  background: rgba(255,96,109,0.12);
}
.trade-review-icon-tile.cap.bad > strong,
.trade-review-icon-tile.risk.bad > strong { color: var(--red); }
.trade-review-icon-tile.risk.good > strong { color: var(--green); }
.trade-review-icon-tile.gm {
  border-color: rgba(201,146,255,0.42);
  background: rgba(201,146,255,0.1);
}
.trade-review-icon-tile.gm > strong { color: #e8c8ff; }
.trade-review-icon-tile.league {
  border-color: rgba(138,180,255,0.42);
  background: rgba(138,180,255,0.1);
}
.trade-review-icon-tile.league > strong { color: #a8c8ff; }
.trade-drawer.trade-review-fullscreen .trade-drawer-body {
  padding: 16px 18px;
}
.trade-review-facts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 10px 0;
}
.trade-review-facts div {
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(0, 216, 223, 0.14);
  background: rgba(255, 255, 255, 0.03);
}
.trade-review-facts span {
  display: block;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
}
.trade-review-facts strong {
  display: block;
  margin-top: 3px;
  font-size: 13px;
  color: #e8f8fc;
}
.trade-review-list {
  margin-top: 10px;
}
.trade-drawer-overlay-asset {
  padding: 0;
  align-items: stretch;
  justify-content: stretch;
  background: rgba(2, 8, 14, 0.94);
  backdrop-filter: blur(8px);
}
.trade-drawer.trade-drawer-asset {
  width: 100vw;
  height: 100vh;
  max-width: none;
  max-height: none;
  border-radius: 0;
  border: none;
  border-top: 1px solid rgba(0, 216, 223, 0.18);
  background:
    linear-gradient(165deg, rgba(8, 22, 34, 0.98), rgba(4, 12, 20, 0.99)),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 6px);
  box-shadow: none;
}
.trade-asset-detail-hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(320px, 1.15fr) minmax(300px, 0.85fr);
  gap: clamp(20px, 3vw, 40px);
  align-items: stretch;
  min-height: min(52vh, 560px);
  padding: clamp(20px, 3vh, 36px) clamp(24px, 4vw, 48px);
  border-bottom: 1px solid rgba(0, 216, 223, 0.16);
  background:
    linear-gradient(180deg, rgba(0, 216, 223, 0.08), transparent 72%),
    radial-gradient(ellipse at 22% 42%, rgba(245, 215, 110, 0.1), transparent 55%),
    radial-gradient(ellipse at 78% 20%, rgba(0, 216, 223, 0.06), transparent 45%);
  flex-shrink: 0;
}
.trade-asset-detail-close {
  position: fixed;
  top: 18px;
  right: 22px;
  z-index: 130;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(0, 0, 0, 0.5);
  color: #e8f4f8;
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.trade-asset-detail-close:hover {
  background: rgba(0, 216, 223, 0.16);
  border-color: rgba(0, 216, 223, 0.45);
}
.trade-asset-hero-visual {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: min(46vh, 500px);
  padding: clamp(12px, 2vh, 24px);
  border-radius: 20px;
  border: 1px solid rgba(0, 216, 223, 0.2);
  background:
    linear-gradient(180deg, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.15)),
    radial-gradient(circle at 50% 30%, rgba(0, 216, 223, 0.1), transparent 60%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 16px 48px rgba(0, 0, 0, 0.35);
}
.trade-asset-hero-main {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: clamp(20px, 3vh, 32px);
  min-width: 0;
}
.trade-asset-hero-pick {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.trade-asset-hero-pick .trade-pick-icon {
  width: min(42vw, 320px);
  height: min(48vh, 380px);
  border-radius: 20px;
  border: 2px solid rgba(0, 216, 223, 0.3);
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.12), rgba(0, 0, 0, 0.35));
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
}
.trade-asset-hero-pick .trade-pick-icon-round {
  font-size: clamp(48px, 8vw, 96px);
  line-height: 1;
}
.trade-asset-hero-pick .trade-pick-icon-year {
  font-size: clamp(18px, 2.5vw, 32px);
  margin-top: 8px;
}
.trade-asset-hero-headshot.player-headshot.size-xl {
  --size: clamp(280px, 42vw, 380px);
  width: var(--size);
  height: var(--size);
  max-width: 100%;
  border-radius: 20px;
  border: 2px solid rgba(0, 216, 223, 0.32);
  box-shadow:
    0 24px 64px rgba(0, 0, 0, 0.5),
    0 0 40px rgba(0, 216, 223, 0.12);
}
.trade-asset-hero-headshot.player-headshot.size-xl .ph-flag {
  width: clamp(36px, 4vw, 52px) !important;
  height: clamp(26px, 3vw, 38px) !important;
}
.trade-asset-hero-identity {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.trade-asset-hero-identity strong {
  font-size: clamp(28px, 4.5vw, 52px);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #f2f8fb;
  text-shadow: 0 0 24px rgba(0, 216, 223, 0.25);
  line-height: 1.05;
}
.trade-asset-hero-identity span {
  font-size: clamp(13px, 1.6vw, 18px);
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(170, 205, 220, 0.9);
}
.trade-asset-hero-tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: clamp(12px, 2vw, 18px);
}
.trade-asset-hero-tile {
  min-width: 0;
  padding: clamp(18px, 2.5vh, 28px) clamp(16px, 2vw, 24px);
  border-radius: 16px;
  border: 1px solid rgba(0, 216, 223, 0.22);
  background: rgba(0, 0, 0, 0.32);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
  text-align: center;
}
.trade-asset-hero-tile span {
  display: block;
  font-size: clamp(10px, 1.2vw, 13px);
  font-weight: 900;
  letter-spacing: 0.16em;
  color: rgba(160, 198, 214, 0.88);
  margin-bottom: 8px;
}
.trade-asset-hero-tile strong {
  display: block;
  font-size: clamp(32px, 5vw, 56px);
  font-weight: 900;
  line-height: 1;
  color: #f5f9fc;
  text-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
}
.trade-asset-hero-tile.tile-ovr strong { color: #f5d76e; }
.trade-asset-hero-tile.tile-cap strong { color: #5ef0f5; font-size: clamp(22px, 3.5vw, 40px); }
.trade-asset-hero-tile.tile-value { grid-column: 1 / -1; }
.trade-asset-hero-tile.tile-value strong {
  font-size: clamp(18px, 2.5vw, 28px);
  letter-spacing: 0.1em;
}
.trade-asset-hero-tile .trade-value-chip-track.hero {
  height: 12px;
  margin-top: 12px;
  max-width: 100%;
}
.trade-drawer.trade-drawer-asset .trade-drawer-body {
  flex: 1;
  min-height: 0;
  padding: clamp(16px, 2.5vh, 28px) clamp(24px, 4vw, 48px);
}
.trade-drawer.trade-drawer-asset .trade-drawer-tabs {
  padding: 10px clamp(24px, 4vw, 48px);
  flex-shrink: 0;
}
.trade-drawer.trade-drawer-asset .trade-drawer-tabs button {
  padding: 12px 10px;
  font-size: 11px;
}
.trade-asset-value-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.trade-asset-value-panel-chip {
  align-self: flex-start;
  min-width: 180px;
}
.trade-asset-value-panel-chip .trade-value-chip-label { font-size: 12px; }
.trade-asset-value-panel-chip .trade-value-chip-track { height: 10px; }
.trade-drawer-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}
.trade-drawer-head strong { display: block; font-size: 16px; letter-spacing: 0.06em; }
.trade-drawer-head span { font-size: 11px; color: var(--muted); }
.trade-drawer-close {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--muted);
  font-size: 22px;
  cursor: pointer;
}
.trade-drawer-tabs {
  display: flex;
  gap: 4px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--line);
}
.trade-drawer-tabs button {
  flex: 1;
  padding: 8px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  cursor: pointer;
}
.trade-drawer-tabs button.active {
  background: var(--cyan-soft);
  border-color: var(--line-2);
  color: var(--cyan);
}
.trade-drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px;
}
.trade-drawer-foot {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--line);
}
.trade-drawer-foot .trade-hub-propose-btn { flex: 1; margin: 0; }
.trade-drawer-btn {
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: var(--panel-3);
  color: var(--text);
  font-size: 11px;
  font-weight: 800;
  cursor: pointer;
}
.trade-drawer-kv {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}
.trade-drawer-kv span { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.trade-drawer-kv strong { display: block; font-size: 14px; margin-top: 2px; }
.trade-drawer-line { font-size: 12px; color: var(--muted); margin: 6px 0; }
.trade-drawer-line.ok { color: var(--green); }
.trade-drawer-line.warn { color: var(--gold); }
.trade-drawer-muted { font-size: 11px; color: var(--muted-2); font-style: italic; }
.trade-drawer-warn { font-size: 11px; color: var(--red); margin: 8px 0; }
.trade-review-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 14px;
}
.trade-review-grid div {
  background: var(--panel-3);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
  text-align: center;
}
.trade-review-grid span { display: block; font-size: 8px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
.trade-review-grid strong { font-size: 16px; }
.trade-review-grid .pos { color: var(--green); }
.trade-review-grid .neg { color: var(--red); }
.trade-review-legal {
  display: flex;
  gap: 8px;
  margin: 10px 0;
  flex-wrap: wrap;
}
.trade-review-legal span {
  font-size: 10px;
  font-weight: 800;
  padding: 5px 10px;
  border-radius: 8px;
  border: 1px solid var(--line);
}
.trade-review-legal .ok { color: var(--green); }
.trade-review-legal .bad { color: var(--red); }
.trade-fan-meter { margin: 12px 0; }
.trade-fan-meter-head {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 800;
  margin-bottom: 6px;
}
.trade-fan-meter-track {
  height: 8px;
  background: var(--panel-3);
  border-radius: 999px;
  overflow: hidden;
}
.trade-fan-meter-track div {
  height: 100%;
  background: linear-gradient(90deg, var(--red), var(--gold), var(--green));
  border-radius: 999px;
}
.trade-fan-meter-label { font-size: 10px; color: var(--muted); margin-top: 4px; display: block; }

/* Team Intel — full-height list panels */
.trade-team-intel-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  min-width: 0;
  padding: 0;
  margin: 0;
  border: none;
  border-radius: 0;
  background:
    linear-gradient(180deg, rgba(5, 14, 24, 0.96), rgba(2, 8, 16, 0.98)),
    repeating-linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.012) 0px,
      rgba(255, 255, 255, 0.012) 1px,
      transparent 1px,
      transparent 5px
    );
  box-shadow: inset 0 1px 0 rgba(0, 216, 223, 0.06);
  overflow: hidden;
  transition: box-shadow 0.2s ease, background 0.2s ease;
}
.trade-team-intel-panel.is-active {
  background:
    linear-gradient(180deg, rgba(6, 20, 32, 0.98), rgba(3, 12, 22, 0.99)),
    repeating-linear-gradient(
      90deg,
      rgba(0, 216, 223, 0.03) 0px,
      rgba(0, 216, 223, 0.03) 1px,
      transparent 1px,
      transparent 5px
    );
  box-shadow:
    inset 0 0 32px rgba(0, 216, 223, 0.07),
    inset 0 1px 0 rgba(0, 216, 223, 0.18);
}
.trade-team-panel-left {
  border-right: 1px solid rgba(0, 216, 223, 0.16);
}
.trade-team-panel-right {
  border-left: 1px solid rgba(0, 216, 223, 0.16);
}
.trade-team-intel {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  flex: 1;
}
.trade-intel-hero {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  padding: 18px 16px 14px;
  border: none;
  border-bottom: 1px solid rgba(0, 216, 223, 0.14);
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.04), transparent);
  cursor: pointer;
  color: inherit;
  text-align: left;
  flex-shrink: 0;
  transition: background 0.15s ease;
}
.trade-intel-hero:hover {
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.09), rgba(0, 216, 223, 0.02));
}
.trade-team-intel.is-active .trade-intel-hero {
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.1), rgba(0, 216, 223, 0.02));
}
.trade-intel-hero-text {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.trade-intel-hero-text strong {
  font-size: 20px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #f0f8fc;
  text-shadow:
    0 0 12px rgba(0, 216, 223, 0.35),
    0 1px 2px rgba(0, 0, 0, 0.8);
}
.trade-intel-status-badge {
  display: inline-block;
  align-self: flex-start;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #f5d76e;
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid rgba(245, 215, 110, 0.35);
  background: rgba(245, 215, 110, 0.08);
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.trade-intel-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 4px 0;
}
.trade-intel-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 38px;
  padding: 11px 16px;
  border-bottom: 1px solid rgba(0, 216, 223, 0.08);
  transition: background 0.12s ease;
}
.trade-intel-row:hover {
  background: rgba(0, 216, 223, 0.04);
}
.trade-intel-label {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(160, 198, 214, 0.88);
  flex-shrink: 0;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-intel-value {
  font-size: 15px;
  font-weight: 900;
  color: #eef6fa;
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.55);
  letter-spacing: 0.03em;
}
.trade-intel-row.gold .trade-intel-value {
  color: #f5d76e;
  text-shadow: 0 0 10px rgba(245, 215, 110, 0.35), 0 1px 2px rgba(0, 0, 0, 0.6);
}
.trade-intel-row.ok .trade-intel-value {
  color: #5ee89a;
  text-shadow: 0 0 8px rgba(94, 232, 154, 0.25), 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-intel-row.bad .trade-intel-value {
  color: #ff6b6b;
  text-shadow: 0 0 8px rgba(255, 107, 107, 0.25), 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-intel-row.warn .trade-intel-value {
  color: #ffb347;
  text-shadow: 0 0 8px rgba(255, 179, 71, 0.25), 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-intel-row.neutral .trade-intel-value {
  color: #b8d4e0;
}
.trade-intel-row.good .trade-intel-value {
  color: #7ee8d0;
  text-shadow: 0 0 8px rgba(126, 232, 208, 0.2), 0 1px 2px rgba(0, 0, 0, 0.5);
}
.trade-intel-foot {
  display: flex;
  flex-direction: column;
  gap: 0;
  flex-shrink: 0;
  border-top: 1px solid rgba(0, 216, 223, 0.16);
  background: rgba(0, 0, 0, 0.15);
}
.trade-intel-view-players {
  width: 100%;
  padding: 16px 14px;
  border: none;
  border-bottom: 1px solid rgba(0, 216, 223, 0.1);
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.14), rgba(0, 216, 223, 0.06));
  color: #5ef0f5;
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
  text-shadow: 0 0 10px rgba(0, 216, 223, 0.4), 0 1px 2px rgba(0, 0, 0, 0.6);
}
.trade-intel-view-players:hover {
  background: linear-gradient(180deg, rgba(0, 216, 223, 0.22), rgba(0, 216, 223, 0.1));
  box-shadow: inset 0 0 20px rgba(0, 216, 223, 0.12);
}
.trade-intel-detail-link {
  width: 100%;
  padding: 13px 14px;
  border: none;
  background: transparent;
  color: rgba(160, 190, 205, 0.85);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  cursor: pointer;
  transition: color 0.15s ease, background 0.15s ease;
}
.trade-intel-detail-link:hover {
  color: #e8f2f8;
  background: rgba(255, 255, 255, 0.04);
}
.trade-players-overlay {
  align-items: center;
  justify-content: center;
  padding: 18px;
}
.trade-drawer-players.trade-players-fullscreen {
  width: min(1500px, calc(100vw - 36px));
  height: min(900px, calc(100vh - 36px));
  max-width: none;
  max-height: none;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 22px;
  border: 1px solid rgba(0, 216, 223, 0.35);
  background:
    radial-gradient(circle at 15% 0%, rgba(0, 216, 223, 0.14), transparent 34%),
    radial-gradient(circle at 88% 10%, rgba(233, 168, 60, 0.1), transparent 30%),
    linear-gradient(180deg, rgba(7, 23, 36, 0.98), rgba(3, 11, 18, 0.98));
  box-shadow:
    0 32px 90px rgba(0, 0, 0, 0.65),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.trade-players-full-head {
  flex-shrink: 0;
  min-height: 92px;
  padding: 18px 22px;
  border-bottom: 1px solid rgba(0, 216, 223, 0.18);
  background:
    linear-gradient(180deg, rgba(11, 35, 52, 0.9), rgba(5, 17, 27, 0.72));
}
.trade-players-full-head .trade-team-logo-img,
.trade-players-full-head .trade-team-logo-fallback {
  filter: drop-shadow(0 0 18px rgba(0, 216, 223, 0.22));
}
.trade-players-header-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
  flex: 1;
}
.trade-players-header-main strong {
  font-size: 22px;
  font-weight: 1000;
  letter-spacing: 0.04em;
}
.trade-players-header-main > span {
  color: rgba(180, 205, 218, 0.76);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.trade-players-intel-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.trade-players-intel-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 9px;
  border-radius: 999px;
  border: 1px solid rgba(0, 216, 223, 0.18);
  background: rgba(255, 255, 255, 0.045);
  color: #d8eef5;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.trade-players-intel-pill span {
  color: rgba(135, 165, 180, 0.9);
}
.trade-players-full-head .trade-drawer-close {
  margin-left: auto;
  width: 38px;
  height: 38px;
  border-radius: 12px;
  font-size: 20px;
  border: 1px solid rgba(0, 216, 223, 0.22);
  background: rgba(255, 255, 255, 0.045);
  color: rgba(220, 240, 248, 0.85);
}
.trade-players-full-head .trade-drawer-close:hover {
  color: #ffffff;
  border-color: rgba(255, 96, 109, 0.55);
  background: rgba(255, 96, 109, 0.12);
}
.trade-players-full-body {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  padding: 18px;
}
.trade-players-full-body .trade-asset-pool {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  border-top: none;
  padding-top: 0;
}
.trade-players-full-body .trade-pool-tabs {
  flex-shrink: 0;
  margin-bottom: 14px;
}
.trade-players-full-body .trade-pool-tabs button {
  min-height: 42px;
  font-size: 11px;
}
.trade-players-full-body .trade-pool-list {
  flex: 1;
  min-height: 0;
  max-height: none;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 2px 10px 10px 2px;
}
.trade-players-full-body .trade-pool-row {
  min-height: 112px;
  padding: 12px 14px;
  border-radius: 14px;
  grid-template-columns: 76px minmax(0, 1fr) 88px;
  gap: 14px;
  align-items: center;
  overflow: visible;
  position: relative;
  z-index: 0;
  background:
    linear-gradient(90deg, rgba(10, 29, 44, 0.96), rgba(7, 20, 32, 0.92)),
    radial-gradient(circle at 0% 50%, rgba(0, 216, 223, 0.12), transparent 36%);
  border: 1px solid rgba(0, 216, 223, 0.18);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.04),
    0 10px 22px rgba(0, 0, 0, 0.16);
}
.trade-players-full-body .trade-pool-row:hover:not(.used):not(.view-only) {
  z-index: 1;
  border-color: rgba(0, 216, 223, 0.42);
  box-shadow:
    0 12px 28px rgba(0, 0, 0, 0.22),
    inset 0 0 20px rgba(0, 216, 223, 0.03);
}
.trade-players-full-body .trade-pool-row.is-focused {
  border-color: rgba(0, 216, 223, 0.62);
  background:
    linear-gradient(90deg, rgba(0, 216, 223, 0.14), rgba(7, 20, 32, 0.94)),
    radial-gradient(circle at 0% 50%, rgba(0, 216, 223, 0.18), transparent 40%);
  box-shadow:
    0 16px 36px rgba(0, 0, 0, 0.28),
    inset 0 0 28px rgba(0, 216, 223, 0.06);
}
.trade-player-list-photo {
  width: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.trade-players-full-body .trade-player-clean-headshot.player-headshot {
  --size: 64px;
  border-radius: 14px;
  border: 1px solid rgba(0, 216, 223, 0.22);
  box-shadow:
    0 8px 18px rgba(0, 0, 0, 0.32),
    inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.trade-players-full-body .trade-player-clean-headshot .ph-flag,
.trade-players-full-body .trade-player-clean-headshot .ph-number,
.trade-players-full-body .trade-player-clean-headshot .ph-badge,
.trade-players-full-body .trade-player-clean-headshot .player-headshot-flag,
.trade-players-full-body .trade-player-clean-headshot .player-headshot-badge,
.trade-players-full-body .trade-player-clean-headshot .player-headshot-country,
.trade-players-full-body .trade-player-clean-headshot .headshot-flag,
.trade-players-full-body .trade-player-clean-headshot .headshot-badge {
  display: none !important;
}
.trade-player-list-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  justify-content: center;
}
.trade-pick-origin-line {
  display: flex;
  align-items: center;
  min-height: 20px;
}
.trade-pick-origin-logo {
  width: 22px;
  height: 22px;
  object-fit: contain;
  filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.35));
}
.trade-pick-origin-fallback {
  color: rgba(160, 190, 208, 0.95);
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.trade-players-full-body .trade-player-list-main {
  gap: 8px;
  align-self: center;
  width: 100%;
}
.trade-players-full-body .trade-player-list-mid {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  min-width: 0;
}
.trade-players-full-body .trade-player-list-details {
  display: flex;
  flex-wrap: nowrap;
  flex-shrink: 0;
  gap: 10px;
}
.trade-player-list-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.trade-player-list-name-row strong {
  min-width: 0;
  color: #f2fbff;
  font-size: 16px;
  font-weight: 1000;
  letter-spacing: 0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.trade-player-list-details {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.trade-player-list-details span {
  min-width: 52px;
  padding: 7px 12px;
  border-radius: 8px;
  border: 1px solid rgba(0, 216, 223, 0.32);
  background: rgba(0, 216, 223, 0.1);
  color: #e8f8fc;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  text-align: center;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 12px rgba(0, 216, 223, 0.08);
}
.trade-players-full-body .trade-player-list-details span {
  min-width: 84px;
  padding: 12px 18px;
  border-radius: 12px;
  font-size: 17px;
  font-weight: 1000;
  letter-spacing: 0.06em;
  border-width: 1px;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.1),
    0 0 18px rgba(0, 216, 223, 0.12);
}
.trade-players-full-body .trade-player-detail-pos {
  color: #f2fbff;
  border-color: rgba(0, 216, 223, 0.45);
  background: rgba(0, 216, 223, 0.16);
}
.trade-players-full-body .trade-player-detail-age {
  color: #e8f4ff;
  border-color: rgba(138, 180, 255, 0.42);
  background: rgba(138, 180, 255, 0.12);
}
.trade-players-full-body .trade-player-detail-cap {
  color: #d4faf8;
  border-color: rgba(94, 240, 245, 0.48);
  background: rgba(0, 216, 223, 0.18);
  min-width: 96px;
}
.trade-player-value-focus {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.trade-players-full-body .trade-player-value-focus {
  display: flex !important;
  flex: 1 1 auto;
  min-width: 0;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  margin: 0;
  padding: 0;
  border: none;
  background: transparent;
  box-shadow: none;
  opacity: 1;
  visibility: visible;
  position: relative;
  z-index: 2;
}
.trade-players-full-body .trade-player-value-head {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  line-height: 1;
}
.trade-players-full-body .trade-player-value-head span {
  color: rgba(148, 178, 194, 0.95);
  font-size: 8px;
  font-weight: 1000;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.trade-players-full-body .trade-player-value-track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(100, 130, 150, 0.28);
  background: rgba(0, 12, 20, 0.78);
  box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.5);
}
.trade-players-full-body .trade-player-value-fill {
  height: 100%;
  min-width: 0;
  max-width: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #4a6070, #6a8498);
  box-shadow: none;
  transition: width 0.28s ease;
}
.trade-players-full-body .trade-player-value-focus.value-franchise .trade-player-value-fill {
  background: linear-gradient(90deg, #d4922a, #ffd166);
  box-shadow: 0 0 12px rgba(233, 168, 60, 0.35);
}
.trade-players-full-body .trade-player-value-focus.value-elite .trade-player-value-fill {
  background: linear-gradient(90deg, #e8892f, #ffc978);
  box-shadow: 0 0 10px rgba(255, 159, 67, 0.28);
}
.trade-players-full-body .trade-player-value-focus.value-top-asset .trade-player-value-fill {
  background: linear-gradient(90deg, #2a6fb8, #54a0ff);
  box-shadow: none;
}
.trade-players-full-body .trade-player-value-focus.value-useful .trade-player-value-fill {
  background: linear-gradient(90deg, #2d6b48, #4aaa72);
  box-shadow: none;
}
.trade-players-full-body .trade-player-value-focus.value-depth .trade-player-value-fill,
.trade-players-full-body .trade-player-value-focus.value-low .trade-player-value-fill,
.trade-players-full-body .trade-player-value-focus.value-unknown .trade-player-value-fill {
  background: linear-gradient(90deg, #3a4554, #5a6878);
  box-shadow: none;
  opacity: 0.85;
}
.trade-player-value-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.trade-player-value-head span {
  color: rgba(200, 235, 245, 0.98);
  font-size: 11px;
  font-weight: 1000;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.trade-player-value-track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid rgba(100, 130, 150, 0.3);
  background: rgba(0, 12, 20, 0.78);
  box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.5);
}
.trade-player-value-fill {
  height: 100%;
  min-width: 0;
  max-width: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #4a6070, #6a8498);
  box-shadow: none;
}
.trade-players-full-body .trade-pool-row.trade-pool-row-has-add {
  grid-template-columns: 76px minmax(0, 1fr) 88px 92px;
  align-items: center;
  gap: 14px;
}
.trade-players-full-body .trade-pool-player .trade-player-ovr-tower,
.trade-players-full-body .trade-pool-player .trade-pool-row-actions {
  align-self: center;
  position: relative;
  z-index: 2;
}
.trade-pool-row-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
}
.trade-pool-add-btn {
  min-width: 78px;
  padding: 10px 8px;
  border-radius: 10px;
  border: 1px solid rgba(0, 216, 223, 0.42);
  background:
    linear-gradient(180deg, rgba(0, 216, 223, 0.22), rgba(0, 216, 223, 0.08)),
    rgba(4, 16, 26, 0.92);
  color: #e8f8fc;
  font-size: 10px;
  font-weight: 1000;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition:
    transform 0.16s ease,
    border-color 0.16s ease,
    box-shadow 0.16s ease,
    background 0.16s ease;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 16px rgba(0, 216, 223, 0.08);
}
.trade-pool-add-btn:hover:not(.in-package):not(.locked) {
  transform: translateY(-1px);
  border-color: rgba(0, 216, 223, 0.72);
  background:
    linear-gradient(180deg, rgba(0, 216, 223, 0.32), rgba(0, 216, 223, 0.12)),
    rgba(4, 16, 26, 0.96);
  box-shadow:
    0 8px 22px rgba(0, 216, 223, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
.trade-pool-add-btn.in-package,
.trade-pool-add-btn.locked {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: default;
  opacity: 0.72;
  border-color: rgba(128, 150, 168, 0.28);
  background: rgba(255, 255, 255, 0.04);
  color: rgba(180, 205, 218, 0.82);
  box-shadow: none;
}
.trade-pool-add-btn.in-package {
  color: rgba(82, 223, 148, 0.88);
  border-color: rgba(82, 223, 148, 0.28);
  background: rgba(82, 223, 148, 0.08);
}
.trade-players-full-body .trade-pool-pick.trade-pool-row-has-add {
  grid-template-columns: 76px minmax(0, 1fr) auto;
}
.trade-player-ovr-tower {
  height: 68px;
  width: 68px;
  min-width: 68px;
  max-width: 68px;
  border-radius: 8px;
  border: 1px solid rgba(0, 216, 223, 0.32);
  background: rgba(0, 216, 223, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  flex-shrink: 0;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 12px rgba(0, 216, 223, 0.08);
}
.trade-player-ovr-tower span {
  color: rgba(135, 165, 180, 0.95);
  font-size: 9px;
  font-weight: 1000;
  letter-spacing: 0.16em;
  line-height: 1;
}
.trade-player-ovr-tower strong {
  color: #e8f8fc;
  font-size: 28px;
  font-weight: 1000;
  line-height: 1;
}
@media (max-height: 860px) {
  .trade-slot {
    min-height: 64px;
    margin-bottom: 5px;
  }
  .trade-slot-placeholder,
  .trade-asset-card {
    min-height: 60px;
  }
  .trade-asset-card-player {
    min-height: 68px;
  }
  .trade-package-col {
    padding: 8px;
  }
  .trade-analysis-rink {
    padding: 8px 10px 6px;
  }
  .trade-hub-propose-btn {
    min-height: 44px;
    padding: 12px 18px;
  }
}
.trade-players-full-body .trade-pool-player .trade-pool-mini-tag,
.trade-players-full-body .trade-pool-player .trade-value-chip-label {
  display: none !important;
}
@media (max-width: 1180px) {
  .trade-review-board {
    overflow-y: auto;
  }
  .trade-review-board-main {
    grid-template-columns: 1fr;
  }
  .trade-review-board-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .trade-review-insight-top-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .trade-review-insight-bottom-row {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 900px) {
  .trade-review-overlay { padding: 4px; }
  .trade-review-shell { padding: 6px; }
  .trade-review-hero { grid-template-columns: 1fr; }
  .trade-review-metrics { grid-template-columns: 1fr; }
  .trade-review-panels { grid-template-columns: 1fr; }
  .trade-review-footer-negotiate { grid-template-columns: 1fr; }
  .trade-review-negot-context { grid-template-columns: 1fr; }
  .trade-review-icon-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trade-review-fix-row { grid-template-columns: 1fr; }
  .trade-drawer.trade-review-fullscreen {
    width: calc(100vw - 12px);
    height: calc(100vh - 12px);
    border-radius: 10px;
  }
  .trade-review-clean-main {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto;
  }
  .trade-review-board-main {
    grid-template-columns: 1fr;
  }
  .trade-review-board-verdict {
    min-height: 120px;
    order: -1;
  }
  .trade-review-verdict-core {
    min-height: 140px;
    order: -1;
  }
  .trade-review-icon-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .trade-review-facts {
    grid-template-columns: 1fr;
  }
  .trade-drawer-players.trade-players-fullscreen {
    width: calc(100vw - 18px);
    height: calc(100vh - 18px);
    border-radius: 16px;
  }
  .trade-players-full-head {
    min-height: auto;
    padding: 14px;
  }
  .trade-players-full-body {
    padding: 12px;
  }
  .trade-players-full-body .trade-pool-list {
    display: flex;
    flex-direction: column;
  }
  .trade-players-full-body .trade-pool-row {
    grid-template-columns: 62px minmax(0, 1fr) 78px;
    gap: 10px;
    padding: 10px;
    min-height: 104px;
  }
  .trade-players-full-body .trade-pool-row.trade-pool-row-has-add {
    grid-template-columns: 56px minmax(0, 1fr) 68px 72px;
  }
  .trade-players-full-body .trade-player-list-mid {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
  }
  .trade-players-full-body .trade-player-list-details {
    flex-wrap: wrap;
  }
  .trade-pool-add-btn {
    min-width: 68px;
    padding: 8px 6px;
    font-size: 9px;
  }
  .trade-players-full-body .trade-player-clean-headshot.player-headshot {
    --size: 54px;
  }
  .trade-player-list-name-row strong {
    font-size: 14px;
  }
  .trade-players-full-body .trade-player-list-details span {
    min-width: 72px;
    padding: 10px 14px;
    font-size: 15px;
  }
  .trade-player-ovr-tower {
    width: 78px;
    min-width: 78px;
    max-width: 78px;
    height: 64px;
  }
  .trade-player-ovr-tower strong {
    font-size: 28px;
  }
  .trade-players-full-body .trade-player-value-track {
    height: 11px;
  }
  .trade-asset-detail-hero {
    grid-template-columns: 1fr;
    min-height: auto;
  }
  .trade-asset-hero-visual {
    min-height: min(42vh, 420px);
  }
  .trade-asset-hero-headshot.player-headshot.size-xl {
    --size: clamp(220px, 72vw, 340px);
  }
  .trade-asset-hero-pick .trade-pick-icon {
    width: min(72vw, 280px);
    height: min(38vh, 340px);
  }
  .trade-clear-value-head {
    grid-template-columns: 1fr;
    text-align: center;
  }
  .trade-clear-value-head span:last-child {
    text-align: center;
  }
  .trade-clear-value-body {
    grid-template-columns: 1fr;
  }
}
`;
