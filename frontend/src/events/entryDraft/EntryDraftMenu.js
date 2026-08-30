import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useGameUI } from "../../game/GameUIContext";
import {
  acceptEntryDraftTrade,
  completeEntryDraft,
  getEntryDraftState,
  simEntryDraftToUserPick,
  startEntryDraft,
  submitCpuDraftPick,
  submitDraftPick,
} from "../../services/franchiseService";
import { getTeamAbbreviation, getTeamLogoSrc } from "../../utils/teamLogos";
import { flagApiUrl, resolveCountryCode } from "../../utils/countryFlags";
import FanReactionFeed from "../../components/franchise/social/FanReactionFeed";
import { buildDraftFanTweets, buildDraftPickReactionTweet } from "../awardsNight/awardHelpers";
import {
  formatPick,
  getPlayerName,
  getPlayerPosition,
  pickFranchiseData,
  safeArray,
} from "../shared/eventHelpers";
import "../../styles/nhlcalShell.css";
import { uiPortalTarget } from "../../utils/fluidUiScale";

const PREFIX = "edraft";
const ROUNDS = 7;

const LOADING_COPY = {
  start: "Preparing…",
  cpu: "Advancing pick…",
  simNext: "Advancing pick…",
  simUser: "Simulating to your pick…",
  simRound: "Simulating round…",
  submit: "Submitting selection…",
  complete: "Completing draft…",
};

/* ------------------------------------------------------------------ */
/* Icons — same broadcast glyph register as CalendarScreen / nhlcal    */
/* ------------------------------------------------------------------ */

const ICON_GLYPHS = {
  back: "←",
  next: "▶",
  down: "▾",
  up: "▴",
  search: "⌕",
  close: "✕",
  check: "✓",
  clock: "◷",
  pin: "⌖",
  compare: "⇄",
  trade: "⇄",
  sim: "▶▶",
  step: "▶",
  round: "↻",
  target: "◎",
  shield: "⛨",
  crease: "▭",
  chart: "▤",
  book: "☰",
  alert: "⚠",
  flame: "✦",
  minus: "−",
  plus: "+",
};

const ICON_TONES = {
  check: "gold",
  search: "cyan",
  trade: "cyan",
  chart: "blue",
  shield: "gold",
  book: "blue",
  flame: "gold",
  target: "cyan",
  compare: "cyan",
  clock: "gold",
  sim: "cyan",
  next: "gold",
  pin: "blue",
  alert: "danger",
  crease: "blue",
  round: "cyan",
  step: "cyan",
};

function Icon({ name, size = 14, className = "", tone, well = false }) {
  const glyph = ICON_GLYPHS[name];
  if (!glyph) return null;
  const resolvedTone = tone || ICON_TONES[name] || "neutral";
  if (well) {
    return (
      <span className={`${PREFIX}-icon-well tone-${resolvedTone} ${className}`.trim()} aria-hidden="true">
        <span style={{ fontSize: size }}>{glyph}</span>
      </span>
    );
  }
  return (
    <span
      className={`${PREFIX}-action-icon tone-${resolvedTone} ${className}`.trim()}
      aria-hidden="true"
      style={{ fontSize: size }}
    >
      {glyph}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Data helpers — unchanged contracts with the backend                 */
/* ------------------------------------------------------------------ */

function liveDraftData(franchiseState, eventData) {
  const fromState = pickFranchiseData(franchiseState, {}, ["draft", "offseason.draft"]);
  const fromEvent = pickFranchiseData({}, eventData, ["draft", "offseason.draft"]);
  if (!fromState && !fromEvent) return {};
  if (!fromState) return fromEvent;
  if (!fromEvent) return fromState;
  const stateN = safeArray(fromState.completed_picks || fromState.draft_results).length;
  const eventN = safeArray(fromEvent.completed_picks || fromEvent.draft_results).length;
  if (stateN !== eventN) return stateN > eventN ? fromState : fromEvent;
  const statePick = Number(fromState.overall_pick || 0);
  const eventPick = Number(fromEvent.overall_pick || 0);
  return statePick >= eventPick ? fromState : fromEvent;
}

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function draftYearLabel(draft) {
  return draft?.draft_year || draft?.draftYear || "";
}

function getId(p) {
  return p?.key || p?.prospect_id || p?.player_id || p?.id || `${getPlayerName(p)}-${p?.rank}`;
}

function num(v, fallback = null) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function getRank(p) {
  return num(p?.rank ?? p?.final_rank ?? p?.public_rank ?? p?.draft_rank);
}

function getPublicRankAtPick(p) {
  return num(p?.public_rank_at_pick ?? p?.final_rank ?? p?.rank ?? p?.public_rank);
}

function getPreseasonRank(p) {
  return num(p?.preseason_rank);
}

function getTeamRank(p) {
  return num(p?.team_board_rank ?? p?.internal_rank ?? p?.team_rank);
}

function getPickValueDelta(p) {
  const pub = getPublicRankAtPick(p);
  const slot = num(p?.overall_pick);
  if (pub == null || slot == null) return null;
  return slot - pub;
}

function describePickMovement(p) {
  const delta = getPickValueDelta(p);
  if (delta == null) return null;
  const abs = Math.abs(delta);
  if (delta >= 12) return { label: `Fell ${abs} spots`, tone: "steal", tag: "Steal" };
  if (delta >= 5) return { label: `Slid ${abs} spots`, tone: "value", tag: "Value" };
  if (delta <= -15) return { label: `Reached ${abs} early`, tone: "reach", tag: "Reach" };
  if (delta <= -8) return { label: `Taken ${abs} early`, tone: "early", tag: "Early" };
  if (abs <= 4) return { label: "On board", tone: "default", tag: null };
  if (delta > 0) return { label: `Fell ${abs}`, tone: "value", tag: "Value" };
  return { label: `Early by ${abs}`, tone: "early", tag: "Early" };
}

function getPickDisplayTag(p) {
  const backendLabel = String(p?.selection_label || p?.pick_classification || "").trim();
  const tags = safeArray(p?.pick_tags);
  const labelMap = {
    Steal: "steal",
    Value: "value",
    Expected: "expected",
    Early: "early",
    Reach: "reach",
    "Off Board": "offboard",
    "Off the Board": "offboard",
    BPA: "bpa",
    "Need Fit": "need",
    "Need Pick": "need",
  };
  if (backendLabel && labelMap[backendLabel] !== undefined) {
    return { tag: backendLabel === "Off the Board" ? "Off Board" : backendLabel, tone: labelMap[backendLabel] };
  }
  if (p?.was_off_board || tags.includes("Off Board") || tags.includes("Off the Board")) {
    return { tag: "Off Board", tone: "offboard" };
  }
  if (p?.was_steal || tags.includes("Steal")) return { tag: "Steal", tone: "steal" };
  if (p?.was_value || tags.includes("Value")) return { tag: "Value", tone: "value" };
  if (p?.was_reach || tags.includes("Reach")) return { tag: "Reach", tone: "reach" };
  if (p?.was_early || tags.includes("Early")) return { tag: "Early", tone: "early" };
  if (p?.was_expected || tags.includes("Expected")) return { tag: "Expected", tone: "expected" };
  if (p?.was_bpa || tags.includes("BPA")) return { tag: "BPA", tone: "bpa" };
  if (p?.was_team_need || tags.includes("Need Fit") || tags.includes("Need Pick")) {
    return { tag: "Need", tone: "need" };
  }
  const movement = describePickMovement(p);
  if (movement?.tag) return { tag: movement.tag, tone: movement.tone };
  return { tag: null, tone: "default" };
}

function enrichPickFromBoard(pick, boardEntries) {
  if (!pick) return null;
  const pid = String(pick.prospect_id || pick.key || "");
  const fromBoard = safeArray(boardEntries).find((e) => String(e.key || e.prospect_id) === pid);
  return { ...(fromBoard || {}), ...pick };
}

function getRisk(p) {
  const v = p?.risk ?? p?.risk_level ?? p?.draft_risk;
  if (v == null || v === "") return null;
  return String(v);
}

function getConfidence(p) {
  const raw = p?.scouting_confidence ?? p?.confidence ?? p?.scout_confidence;
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function getConfidenceRange(p) {
  const lo = num(p?.confidence_low ?? p?.scouting_confidence_low);
  const hi = num(p?.confidence_high ?? p?.scouting_confidence_high);
  if (lo != null && hi != null) return [lo, hi];
  return null;
}

function isSoftConfidence(p) {
  const c = getConfidence(p);
  return c != null && c < 45 && !getConfidenceRange(p);
}

function getConfidenceTitle(p) {
  const range = getConfidenceRange(p);
  if (range) {
    return `Scouting confidence band: ${Math.round(range[0])}–${Math.round(range[1])}%`;
  }
  const c = getConfidence(p);
  if (c == null) return null;
  if (c < 45) {
    return `Soft confidence (~${Math.round(c)}%) — wider scouting variance on this prospect`;
  }
  return `Scout confidence: ${Math.round(c)}%`;
}

function formatConfidence(p) {
  const range = getConfidenceRange(p);
  if (range) return `${Math.round(range[0])}–${Math.round(range[1])}%`;
  const c = getConfidence(p);
  if (c == null) return null;
  if (c < 45) return `~${Math.round(c)}%`;
  return `${Math.round(c)}%`;
}

function getTeamFit(p) {
  const raw = p?.team_fit_score ?? p?.fit_score ?? p?.need_fit_score ?? p?.organizational_fit;
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function normalizeStockLabel(p) {
  const raw = p?.stock_label || p?.stock_movement_label || p?.stock;
  if (raw == null || raw === "") {
    const move = num(p?.stock_movement ?? p?.movement);
    if (move == null) return null;
    if (move > 0) return "Rising";
    if (move < 0) return "Falling";
    return "Stable";
  }
  const label = String(raw).toLowerCase();
  if (label.includes("rise") || label.includes("up") || label.includes("trend")) return "Rising";
  if (label.includes("fall") || label.includes("down")) return "Falling";
  return "Stable";
}

function getDefiningTrait(p) {
  if (!p) return null;
  const dossier = p.dossier || {};
  const v =
    dossier?.player_comparison?.archetype ||
    p.projected_role ||
    p.expected_role ||
    p.rights_card?.expected_role ||
    p.scouting_trait ||
    p.player_type ||
    p.archetype ||
    p.style;
  return v ? String(v) : null;
}

function getScoutedPotentialLabel(p) {
  if (!p) return null;
  const d = p.dossier?.potential_range || p.potential_range;
  if (d && typeof d === "object" && !Array.isArray(d) && d.low != null && d.high != null) {
    return `${Math.round(d.low)}–${Math.round(d.high)}`;
  }
  const range = safeArray(p.scouted_potential_range || (Array.isArray(d) ? d : null));
  if (range.length >= 2 && Number.isFinite(Number(range[0])) && Number.isFinite(Number(range[1]))) {
    return `${Math.round(range[0])}–${Math.round(range[1])}`;
  }
  if (p.scouted_potential != null && p.scouted_potential !== "") return String(p.scouted_potential);
  if (p.potential_label) return String(p.potential_label);
  if (p.potential_tier) return String(p.potential_tier);
  return null;
}

function formatCurrentOvr(p) {
  if (!p) return null;
  return resolveCurrentOvrLabel(p);
}

function readinessFromScore(n) {
  if (n >= 78) return "NHL Ready";
  if (n >= 68) return "Close";
  if (n >= 55) return "Developing";
  if (n >= 40) return "Long-term project";
  return "At Risk";
}

function formatNhlReadiness(p) {
  const ready = p?.nhl_readiness ?? p?.dossier?.nhl_readiness ?? p?.dossier?.readiness;
  if (ready != null && ready !== "") {
    if (typeof ready === "object") {
      if (ready.label) return String(ready.label);
      if (ready.score != null && Number.isFinite(Number(ready.score))) {
        return readinessFromScore(Number(ready.score));
      }
    }
    const n = Number(ready);
    if (Number.isFinite(n)) {
      if (n > 1.5 && n <= 100) return readinessFromScore(n);
      if (n <= 0) return "NHL Ready";
      if (n <= 8) return `${Math.round(n)} Years Away`;
    }
    const s = String(ready);
    if (/ready|close|develop|project|risk/i.test(s)) return s;
  }
  const label = p?.dossier?.readinessLabel || p?.readinessLabel;
  return label ? String(label) : null;
}

function formatNhlArrivalEta(p) {
  const etaObj = p?.dossier?.eta;
  if (etaObj && typeof etaObj === "object") {
    if (etaObj.years === 0 || String(etaObj.label || "").toLowerCase() === "now") return "NHL Ready / Now";
    if (etaObj.years != null) return `${Math.round(Number(etaObj.years))} Years Away`;
    if (etaObj.label) return String(etaObj.label);
  }
  const years = Number(p?.nhl_eta_years ?? p?.nhl_eta ?? p?.rights_card?.eta);
  if (Number.isFinite(years)) return years <= 0 ? "NHL Ready / Now" : `${Math.round(years)} Years Away`;
  if (p?.nhl_eta_label) return String(p.nhl_eta_label);
  return null;
}

function getNhlEta(p) {
  const value = p?.nhl_eta ?? p?.eta ?? p?.nhl_eta_years ?? p?.development_eta ?? p?.rights_card?.eta;
  if (value == null || value === "") return null;
  if (typeof value === "object") {
    if (value.label === "Now" || value.years === 0) return "NHL Ready";
    if (value.label) return String(value.label);
    if (value.years != null && value.years !== "") return `${value.years}Y`;
    return null;
  }
  return String(value);
}

function formatNhlPotentialEta(p) {
  const peak = p?.potential_eta ?? p?.dossier?.potential_eta ?? p?.dossier?.peak_eta;
  if (peak != null && peak !== "") {
    if (typeof peak === "object") {
      if (peak.label) return String(peak.label);
      if (peak.years != null) return `${Math.round(Number(peak.years))}y to peak`;
    }
    return String(peak);
  }
  const etaObj = p?.dossier?.eta;
  if (etaObj && typeof etaObj === "object" && etaObj.peak_years != null) {
    return `${Math.round(Number(etaObj.peak_years))}y to peak`;
  }
  return getNhlEta(p);
}

function isPhilosophyNeedLabel(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("upside") || s.includes("philosophy") || s.includes("bpa")) return true;
  if (s.includes("high") && s.includes("swing")) return true;
  return false;
}

function getDevelopmentPath(p) {
  return p?.development_path || null;
}

function getComparable(p) {
  const v = p?.comparable_player || p?.player_comparable || p?.style_comparable;
  return v ? String(v) : null;
}

function getScoutSummary(p) {
  return p?.scout_summary || p?.scout_quote || p?.summary || null;
}

function getBackendWhyWorks(p) {
  return p?.why_pick_makes_sense || p?.why_this_pick_makes_sense || null;
}

function getBackendWhyFails(p) {
  return p?.why_pick_could_fail || p?.bust_reason || null;
}

function shortTeamName(name) {
  if (!name) return "";
  const s = String(name);
  if (s.includes(" ")) {
    const parts = s.trim().split(/\s+/);
    return parts[parts.length - 1];
  }
  return s;
}

function teamAbbrev(teamId, teamName) {
  return (
    getTeamAbbreviation({ team_id: teamId, name: teamName, id: teamId }) ||
    String(teamName || teamId || "").slice(0, 3).toUpperCase()
  );
}

function roundInt(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(n) : null;
}

function formatOvrRange(low, high) {
  const lo = roundInt(low);
  const hi = roundInt(high);
  if (lo == null && hi == null) return null;
  if (lo != null && hi != null) return lo === hi ? `${lo}` : `${lo}–${hi}`;
  return lo != null ? `${lo}` : `${hi}`;
}

function resolveCurrentOvrLabel(p) {
  if (!p) return null;
  if (p.ovr_revealed && p.true_ovr != null) return `${roundInt(p.true_ovr)}`;
  const lo = roundInt(
    p.floor_grade ??
    p.dossier?.overallRangeLow ??
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[0] : null) ??
    p.current_ovr
  );
  const hi = roundInt(
    p.dossier?.overallRangeHigh ??
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[1] : null) ??
    p.current_ovr
  );
  if (lo != null || hi != null) return formatOvrRange(lo, hi);
  const est = roundInt(p.scouted_overall_estimate ?? p.current_ovr_estimate);
  return est != null ? `${est}` : null;
}

function resolvePotentialOvrLabel(p) {
  if (!p) return null;
  if (p.ceiling_hidden || p.dossier?.ceilingHidden) return null;
  const lo = roundInt(
    p.dossier?.scoutedPotentialLow ??
    p.potential_range?.low ??
    (Array.isArray(p.scouted_potential_range) ? p.scouted_potential_range[0] : null)
  );
  let hi = roundInt(
    p.dossier?.scoutedPotentialHigh ??
    p.potential_range?.high ??
    (Array.isArray(p.scouted_potential_range) ? p.scouted_potential_range[1] : null)
  );
  const rank = getRank(p);
  const maxSpan = rank != null && rank <= 10 ? 6 : rank != null && rank <= 32 ? 5 : rank != null && rank <= 64 ? 4 : 3;
  if (hi != null && lo != null && hi - lo > maxSpan) hi = lo + maxSpan;
  const label = formatOvrRange(lo, hi);
  if (label) return label;
  return getScoutedPotentialLabel(p);
}

function resolveProductionStats(prospect, dossierStats = {}) {
  const src = dossierStats && typeof dossierStats === "object" ? dossierStats : {};
  const actual = prospect?.actual_stats && typeof prospect.actual_stats === "object" ? prospect.actual_stats : {};
  const pickStat = (...vals) => {
    for (const v of vals) {
      if (v == null || v === "") continue;
      const n = roundInt(v);
      if (n != null) return n;
    }
    return null;
  };
  const games = pickStat(
    prospect?.gp,
    prospect?.games_played,
    actual.gp,
    actual.games,
    actual.games_played,
    src.games,
    src.gp,
  ) ?? 0;
  const goals = pickStat(prospect?.goals, actual.goals, src.goals) ?? 0;
  const assists = pickStat(prospect?.assists, actual.assists, src.assists) ?? 0;
  const points = pickStat(prospect?.points, actual.points, src.points) ?? (goals + assists);
  const ppgRaw = prospect?.ppg ?? prospect?.points_per_game ?? actual.ppg ?? src.ppg;
  const ppg = ppgRaw != null
    ? Number(ppgRaw)
    : (games > 0 ? points / games : null);
  return {
    games,
    goals,
    assists,
    points,
    ppg: Number.isFinite(ppg) ? ppg : null,
    primary: pickStat(prospect?.primary_points, actual.primary_points, src.primary_points),
    savePct: prospect?.save_pct ?? actual.save_pct ?? src.save_pct,
    gaa: prospect?.gaa ?? actual.gaa ?? src.gaa,
    shutouts: pickStat(prospect?.shutouts, actual.shutouts, src.shutouts),
    wins: pickStat(prospect?.wins, actual.wins, src.wins),
  };
}

function formatStatCell(label, value) {
  if (value == null || value === "") return null;
  if (label === "SV%") {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return n > 1 ? n.toFixed(1) : (n * 100).toFixed(1);
  }
  if (label === "GAA" || label === "P/GP") {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(1) : null;
  }
  return String(roundInt(value) ?? value);
}

function needLabel(n) {
  return n?.category || n?.position || n?.need || n?.label || null;
}

function getUpcomingOrder(draftOrder, completedCount, userTeamId, windowSize = 12) {
  const order = safeArray(draftOrder);
  const upcoming = order.slice(completedCount + 1);
  const next = upcoming.slice(0, windowSize);
  const userNext = upcoming.find((s) => String(s.team_id) === String(userTeamId));
  if (userNext && !next.some((s) => s.overall_pick === userNext.overall_pick)) {
    return [...next, userNext];
  }
  return next;
}

function isGoalie(p) {
  return String(getPlayerPosition(p) || "").toUpperCase() === "G";
}

const SKATER_CHAPTER_ORDER = [
  ["Overall", "overall"],
  ["Offence", "offence"],
  ["Defence", "defence"],
  ["Character", "character"],
  ["Mental", "mental"],
  ["Transition", "transition"],
  ["Physical", "physical"],
  ["Potential", "potential"],
];

const GOALIE_CHAPTER_ORDER = [
  ["Overall", "overall"],
  ["Glove", "glove"],
  ["Blocker", "blocker"],
  ["Stick", "stick"],
  ["Potential", "potential"],
];

function chapterRatingsHidden(prospect) {
  if (!prospect) return true;
  if (prospect.ceiling_hidden || prospect.dossier?.ceilingHidden) return true;
  const cp = prospect.chapter_profile || prospect.dossier?.chapter_profile;
  if (cp?.hidden) return true;
  if (prospect.dossier?.chapterProfileHidden) return true;
  const scout = num(prospect.scouted_percentage ?? prospect.dossier?.dedicatedScoutFile ? 72 : 0) ?? 0;
  const rank = getRank(prospect);
  if (scout >= 72) return false;
  if (rank != null && rank > 64 && scout < 55) return true;
  if (rank != null && rank > 32 && scout < 45) return true;
  if (rank != null && rank > 15 && scout < 35) return true;
  return false;
}

function chapterRatingsFogged(prospect) {
  if (!prospect || chapterRatingsHidden(prospect)) return false;
  const cp = prospect.chapter_profile || prospect.dossier?.chapter_profile;
  return Boolean(cp?.fogged || prospect.dossier?.chapterProfileFogged);
}

function resolveChapterMap(prospect) {
  if (!prospect) return {};
  if (chapterRatingsHidden(prospect)) return {};

  const profileChapters =
    (prospect.chapter_profile?.chapters && typeof prospect.chapter_profile.chapters === "object"
      ? prospect.chapter_profile.chapters
      : null)
    || (prospect.dossier?.chapter_profile?.chapters && typeof prospect.dossier.chapter_profile.chapters === "object"
      ? prospect.dossier.chapter_profile.chapters
      : null)
    || (prospect.chapters && typeof prospect.chapters === "object" ? prospect.chapters : null)
    || {};

  const fogged = chapterRatingsFogged(prospect);
  const ovr = roundInt(
    prospect.overall
    ?? prospect.effective_ovr
    ?? prospect.scouted_overall_estimate
    ?? prospect.dossier?.overallRangeLow
    ?? prospect.dossier?.now_range?.low
  );
  const pot = roundInt(
    prospect.potential
    ?? prospect.potential_score
    ?? prospect.dossier?.scoutedPotentialHigh
    ?? prospect.dossier?.peak_range?.high
    ?? (Array.isArray(prospect.potential_range) ? prospect.potential_range[1] : null)
  );

  const readChapter = (key) => {
    const raw = profileChapters[key];
    if (raw == null || raw === "") return null;
    if (typeof raw === "object" && raw.band) {
      const lo = roundInt(raw.low);
      const hi = roundInt(raw.high);
      if (lo != null && hi != null) return { band: true, lo, hi, mid: Math.round((lo + hi) / 2) };
    }
    return chapterValue(profileChapters, key);
  };

  if (isGoalie(prospect)) {
    return {
      overall: readChapter("overall") ?? (fogged ? null : ovr),
      glove: readChapter("glove") ?? (fogged ? null : ovr),
      blocker: readChapter("blocker") ?? (fogged ? null : ovr),
      stick: readChapter("stick") ?? (fogged ? null : (ovr != null ? Math.max(0, ovr - 2) : null)),
      potential: readChapter("potential") ?? (fogged ? null : pot),
    };
  }

  if (fogged) {
    return {
      offence: readChapter("offence"),
      defence: readChapter("defence"),
      character: readChapter("character"),
      mental: readChapter("mental"),
      transition: readChapter("transition"),
      physical: readChapter("physical"),
    };
  }

  const offenceFallback = roundInt(prospect.offence ?? prospect.shooting ?? prospect.passing);
  const defenceFallback = roundInt(prospect.defence ?? prospect.defense);
  const mentalFallback = roundInt(prospect.mental ?? prospect.hockey_iq);
  const transitionFallback = roundInt(prospect.transition ?? prospect.skating);
  const physicalFallback = roundInt(prospect.physical ?? prospect.physicality);
  const characterFallback =
    roundInt(prospect.character_score)
    ?? mentalFallback
    ?? ovr;

  return {
    overall: readChapter("overall") ?? ovr,
    offence: readChapter("offence") ?? offenceFallback ?? ovr,
    defence: readChapter("defence") ?? defenceFallback ?? ovr,
    character: readChapter("character") ?? characterFallback,
    mental: readChapter("mental") ?? mentalFallback ?? ovr,
    transition: readChapter("transition") ?? transitionFallback ?? ovr,
    physical: readChapter("physical") ?? physicalFallback ?? ovr,
    potential: readChapter("potential") ?? pot,
  };
}

function chapterValue(chapters, key) {
  if (!chapters || typeof chapters !== "object") return null;
  const aliases = key === "defence" ? ["defence", "defense"] : [key];
  for (const alias of aliases) {
    const raw = chapters[alias];
    if (raw != null && raw !== "") {
      const n = roundInt(raw);
      if (n != null) return n;
    }
  }
  return null;
}

/* Chapter-based rating rows (authoritative schema). */
function chapterAttributeRows(prospect) {
  if (!prospect || chapterRatingsHidden(prospect)) return [];
  const chapters = resolveChapterMap(prospect);
  const order = isGoalie(prospect) ? GOALIE_CHAPTER_ORDER : SKATER_CHAPTER_ORDER;
  return order
    .map(([label, key]) => {
      const value = chapters[key];
      if (value == null || value === "") return null;
      if (typeof value === "object" && value.band) {
        return [label, value];
      }
      const n = roundInt(value);
      return n != null ? [label, n] : null;
    })
    .filter(Boolean);
}

function attributeRows(prospect) {
  return chapterAttributeRows(prospect);
}

function buildDraftGrade(userPicks) {
  const picks = safeArray(userPicks);
  if (!picks.length) return null;
  let score = 75;
  for (const p of picks) {
    if (p.pick_classification === "Steal" || p.was_steal) score += 5;
    if (p.pick_classification === "Reach" || p.was_reach) score -= 5;
    if (p.pick_classification === "BPA") score += 3;
    if (p.pick_classification === "Need Pick") score += 2;
    if (p.pick_classification === "Goalie Gamble") score -= 2;
  }
  score = Math.max(0, Math.min(100, score));
  if (score >= 94) return "A+";
  if (score >= 90) return "A";
  if (score >= 86) return "A-";
  if (score >= 82) return "B+";
  if (score >= 78) return "B";
  if (score >= 74) return "B-";
  if (score >= 70) return "C+";
  if (score >= 65) return "C";
  if (score >= 60) return "C-";
  if (score >= 50) return "D";
  return "F";
}

function clip15(text) {
  return String(text || "").trim().split(/\s+/).filter(Boolean).slice(0, 15).join(" ");
}

function pubDelta(pick) {
  let d = pick?.public_rank_delta;
  if (d == null) d = pick?.public_board_delta;
  if (d == null) {
    const o = Number(pick?.overall_pick || 0);
    const f = Number(pick?.final_rank || 0);
    d = o && f ? o - f : null;
  }
  return d == null || Number.isNaN(Number(d)) ? null : Math.round(Number(d));
}

function fmtSigned(v, digits = 0) {
  if (v == null || Number.isNaN(Number(v))) return null;
  const n = Number(v);
  const s = digits ? n.toFixed(digits) : String(Math.round(n));
  return n > 0 ? `+${s}` : s;
}

function numOrDash(v) {
  return v == null || v === "" || Number.isNaN(Number(v)) ? "—" : Math.round(Number(v));
}

function ovrVal(p) {
  return numOrDash(p?.floor_grade ?? p?.true_ovr ?? p?.scouted_ovr);
}

function potVal(p) {
  const v = p?.ceiling_grade ?? p?.potential_score;
  if (v != null && !Number.isNaN(Number(v))) return Math.round(Number(v));
  return p?.potential_grade || "—";
}

function projVal(p) {
  const raw = p?.player_type || p?.development_path || p?.nhl_readiness;
  if (raw) return String(raw);
  return p?.nhl_eta != null ? `${p.nhl_eta}y` : "—";
}

function teamAccent(team) {
  const abbr = String(getTeamAbbreviation(team) || "").toUpperCase();
  if (!abbr) return "var(--ed-gold)";
  let hash = 0;
  for (let i = 0; i < abbr.length; i += 1) hash = abbr.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 62% 52%)`;
}

const RECAP_POS_COLORS = {
  C: "#4fb0ff",
  LW: "#6ce5b0",
  RW: "#ffd166",
  D: "#c792ea",
  G: "#ff7b72",
  W: "#6ce5b0",
};

/* ------------------------------------------------------------------ */
/* Stylesheet                                                          */
/* ------------------------------------------------------------------ */

const SHEET = `
.edraft-root{
  --bg:#04101a;
  --bg-2:#061522;
  --panel:rgba(9,25,38,.94);
  --panel-2:rgba(12,35,52,.94);
  --panel-3:rgba(15,46,66,.78);
  --line:rgba(156,218,236,.14);
  --line-soft:rgba(156,218,236,.08);
  --line-2:rgba(115,229,241,.25);
  --line-strong:rgba(73,231,240,.5);
  --text:#e9f7fb;
  --muted:#8096a8;
  --muted-2:#607789;
  --cyan:#13d8e7;
  --cyan-soft:rgba(19,216,231,.13);
  --gold:#e9a83c;
  --gold-soft:rgba(233,168,60,.14);
  --green:#52df94;
  --green-soft:rgba(82,223,148,.13);
  --red:#ff606d;
  --red-soft:rgba(255,96,109,.13);
  --orange:#ff8a4c;
  --orange-soft:rgba(255,138,76,.13);
  --blue:#8ab4ff;
  --blue-soft:rgba(138,180,255,.13);
  --purple:#8ab4ff;
  --purple-soft:rgba(138,180,255,.14);
  --shadow:0 24px 70px rgba(0,0,0,.42);
  --depth-registered:inset 0 1px 0 rgba(255,255,255,.04);
  --ed-bg:#020a11;
  --ed-panel:var(--panel);
  --ed-panel-2:var(--panel-2);
  --ed-raise:var(--panel-3);
  --ed-line:var(--line);
  --ed-line-strong:var(--line-strong);
  --ed-line-soft:rgba(156,218,236,.08);
  --ed-ink:var(--text);
  --ed-ink-2:var(--muted);
  --ed-ink-3:var(--muted-2);
  --ed-gold:var(--gold);
  --ed-cyan:var(--cyan);
  --ed-good:var(--green);
  --ed-bad:var(--red);
  --ed-warn:var(--gold);
  --ed-purple:var(--blue);
  --ed-cyan-soft:var(--cyan-soft);
  --ed-gold-soft:var(--gold-soft);
  --ed-green-soft:var(--green-soft);
  --ed-red-soft:var(--red-soft);
  --ed-blue-soft:var(--blue-soft);
  --ed-head:var(--font-broadcast-display,"Archivo Black","Teko","Barlow Condensed",system-ui,sans-serif);
  --ed-body:var(--font-ops-ui,Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif);
  position:relative;
  display:flex;
  flex-direction:column;
  min-height:100dvh;
  height:100%;
  max-height:100dvh;
  background:
    radial-gradient(circle at 24% 0%, rgba(19,216,231,.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233,168,60,.08), transparent 26%),
    linear-gradient(180deg,#06131f 0%,#020a11 100%);
  color:var(--text);
  font-family:var(--ed-body);
  font-size:14px;
  letter-spacing:.01em;
  overflow:hidden;
}
.edraft-root *{box-sizing:border-box;}
.edraft-root button{font-family:inherit;}
.edraft-action-icon{
  display:inline-flex;width:1.1em;justify-content:center;align-items:center;
  line-height:1;flex:none;font-weight:1000;opacity:1;
}
.edraft-action-icon.tone-neutral{color:var(--text);}
.edraft-action-icon.tone-cyan{color:var(--cyan);}
.edraft-action-icon.tone-gold{color:var(--gold);}
.edraft-action-icon.tone-blue{color:var(--blue);}
.edraft-action-icon.tone-green{color:var(--green);}
.edraft-action-icon.tone-danger{color:var(--red);}
.nhlcal-advance-button .edraft-action-icon,
.nhlcal-advance-button-secondary .edraft-action-icon{color:inherit;}
.edraft-icon-well{
  width:28px;height:28px;display:grid;place-items:center;flex:none;
  border-radius:8px;background:rgba(148,185,205,.12);
  border:1px solid rgba(148,185,205,.14);color:var(--text);
  font-weight:1000;line-height:1;
  box-shadow:var(--depth-registered);
}
.edraft-icon-well.tone-cyan{color:var(--cyan);background:var(--cyan-soft);border-color:rgba(19,216,231,.32);box-shadow:0 0 8px rgba(19,216,231,.28);}
.edraft-icon-well.tone-gold{color:var(--gold);background:var(--gold-soft);border-color:rgba(233,168,60,.32);box-shadow:0 0 8px rgba(233,168,60,.24);}
.edraft-icon-well.tone-blue{color:var(--blue);background:var(--blue-soft);border-color:rgba(138,180,255,.32);box-shadow:0 0 8px rgba(138,180,255,.22);}
.edraft-icon-well.tone-green{color:var(--green);background:var(--green-soft);border-color:rgba(82,223,148,.32);box-shadow:0 0 8px rgba(82,223,148,.22);}
.edraft-icon-well.tone-danger{color:var(--red);background:var(--red-soft);border-color:rgba(255,96,109,.32);box-shadow:0 0 8px rgba(255,96,109,.22);}
.edraft-icon-well.tone-neutral{color:var(--text);background:rgba(148,185,205,.12);border-color:rgba(148,185,205,.18);}
.edraft-tabular{font-variant-numeric:tabular-nums;letter-spacing:.02em;}
.edraft-muted{color:var(--muted);font-size:12.5px;}

/* ---------- command bar ---------- */
.edraft-command{
  position:relative;z-index:3;
  display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:14px;
  padding:12px;
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.08), transparent 42%),
    var(--panel);
  border-bottom:1px solid var(--line);
  box-shadow:var(--depth-registered);
}
.edraft-command-left{display:flex;align-items:center;gap:12px;min-width:0;}
.edraft-titles h1{
  margin:0;font-family:var(--ed-head);font-weight:900;
  font-size:20px;line-height:1;letter-spacing:.05em;text-transform:uppercase;color:var(--text);
}
.edraft-titles p{margin:2px 0 0;font-size:11px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;font-weight:800;}
.edraft-command-mid{display:flex;align-items:center;justify-content:center;min-width:0;}
.edraft-phase{
  margin:0;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;color:var(--gold);
  text-shadow:0 0 18px rgba(233,168,60,.28);
}
.edraft-phase.is-cpu{color:var(--cyan);text-shadow:0 0 18px rgba(19,216,231,.28);}
.edraft-command-right{display:flex;align-items:center;gap:8px;}
.edraft-root .nhlcal-quick-link{
  border:1px solid var(--line);border-radius:4px;background:rgba(12,31,47,.72);
  color:var(--text);padding:9px 14px;font-size:11px;font-weight:900;
  letter-spacing:.06em;text-transform:uppercase;transition:border-color .2s ease,background .2s ease,transform .2s ease;
}
.edraft-root .nhlcal-quick-link:hover{
  border-color:var(--line-strong);background:rgba(19,216,231,.12);color:var(--cyan);transform:translateY(-1px);
}
.edraft-root .nhlcal-advance-button{
  height:40px;min-width:108px;border:0;border-radius:0;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);
  background:var(--gold);color:#1b1002;text-transform:uppercase;
  letter-spacing:.12em;font-size:11px;font-weight:1000;
}
.edraft-root .nhlcal-advance-button:hover{background:#f0b44a;}
.edraft-root .nhlcal-advance-button-secondary{
  height:40px;min-width:108px;border:1px solid rgba(19,216,231,.28);border-radius:0;
  background:rgba(7,22,35,.88);color:var(--text);text-transform:uppercase;
  letter-spacing:.08em;font-size:11px;font-weight:1000;
}
.edraft-root .nhlcal-advance-button-secondary:hover:not(:disabled){
  border-color:rgba(19,216,231,.52);background:rgba(19,216,231,.11);color:var(--cyan);
}

/* Shared nhlcal buttons — compact variants inside draft surfaces */
.edraft-dock-actions .nhlcal-advance-button,
.edraft-dock-actions .nhlcal-advance-button-secondary,
.edraft-onclock .nhlcal-advance-button,
.edraft-dossier-actions .nhlcal-advance-button,
.edraft-sheet-foot .nhlcal-advance-button,
.edraft-sheet-foot .nhlcal-advance-button-secondary{
  height:40px;min-width:108px;font-size:11px;letter-spacing:.08em;
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
}
.edraft-dossier-actions .nhlcal-advance-button.edraft-btn--draft{
  flex:1;min-height:46px;min-width:0;height:46px;font-size:12px;
}
.edraft-board-draft-btn{
  height:32px;min-width:0;width:100%;padding:0 6px;font-size:10px;letter-spacing:.08em;
  display:inline-flex;align-items:center;justify-content:center;gap:4px;
  box-shadow:none;
}
.edraft-btn--danger{color:var(--red)!important;border-color:rgba(255,96,109,.42)!important;}
.edraft-link{
  background:none;border:0;padding:0;color:var(--cyan);cursor:pointer;
  font:inherit;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;font-weight:900;
}
.edraft-link:hover{color:var(--text);}

/* ---------- on-deck rail ---------- */
.edraft-ondeck{
  display:flex;align-items:stretch;gap:0;
  border-bottom:1px solid var(--line);
  background:
    radial-gradient(circle at 100% 14%, rgba(19,216,231,.14), transparent 34%),
    linear-gradient(180deg,rgba(5,16,26,.98),rgba(3,10,17,.98));
}
.edraft-ondeck-label{
  display:flex;align-items:center;gap:6px;padding:0 14px;flex:none;
  font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan);font-weight:1000;
  border-right:1px solid var(--line);
}
.edraft-ondeck-track{display:flex;gap:0;overflow-x:auto;scrollbar-width:thin;}
.edraft-ondeck-track::-webkit-scrollbar{height:4px;}
.edraft-ondeck-track::-webkit-scrollbar-thumb{background:rgba(19,216,231,.28);}
.edraft-slot{
  display:flex;align-items:center;gap:7px;padding:8px 13px;flex:none;
  border-right:1px solid var(--line-soft);
  transition:background .18s ease,box-shadow .18s ease;
}
.edraft-slot-num{font-family:var(--ed-head);font-size:14px;color:var(--muted);}
.edraft-slot-abbr{font-size:12px;font-weight:800;letter-spacing:.05em;color:var(--text);}
.edraft-slot-via{display:block;font-size:9.5px;color:var(--muted);letter-spacing:.06em;}
.edraft-slot.is-next{
  background:
    radial-gradient(circle at 0% 0%, rgba(233,168,60,.18), transparent 56%),
    var(--gold-soft);
  box-shadow:inset 2px 0 0 var(--gold),0 0 10px rgba(233,168,60,.18);
}
.edraft-slot.is-user{
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.16), transparent 56%),
    var(--cyan-soft);
  box-shadow:inset 2px 0 0 var(--cyan),0 0 10px rgba(19,216,231,.22);
}
.edraft-slot.is-user .edraft-slot-abbr{color:var(--cyan);}
.edraft-slot.is-next .edraft-slot-abbr{color:var(--gold);}

/* ---------- floor layout ---------- */
.edraft-floor{
  flex:1;min-height:0;position:relative;z-index:2;
  display:grid;grid-template-columns:minmax(200px,14vw) minmax(0,1fr) minmax(260px,20vw);gap:8px;
  padding:12px;background:transparent;
}
.edraft-stage{
  background:
    radial-gradient(circle at 0% 0%, rgba(138,180,255,.08), transparent 40%),
    rgba(6,21,34,.82);
  border:1px solid var(--line);border-radius:8px;
  box-shadow:var(--depth-registered);
  display:flex;flex-direction:column;min-height:0;overflow:hidden;
}
.edraft-stage .edraft-scroll{flex:1;min-height:0;display:flex;flex-direction:column;}
.edraft-pane{
  background:rgba(6,21,34,.82);
  border:1px solid var(--line);border-radius:8px;
  box-shadow:var(--depth-registered);
  display:flex;flex-direction:column;min-height:0;min-width:0;overflow:hidden;
}
.edraft-pane-head{
  display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:10px 12px;border-bottom:1px solid var(--line);flex:none;
  background:rgba(5,17,27,.72);
}
.edraft-pane-head h3{
  margin:0;font-family:var(--ed-head);font-weight:1000;font-size:15px;
  letter-spacing:.13em;text-transform:uppercase;color:var(--cyan);
}
.edraft-pane-head em{font-style:normal;font-size:11px;color:var(--cyan);font-weight:900;opacity:.85;}
.edraft-scroll{flex:1;min-height:0;overflow-y:auto;scrollbar-width:thin;}
.edraft-scroll::-webkit-scrollbar{width:6px;}
.edraft-scroll::-webkit-scrollbar-thumb{background:rgba(150,180,208,.2);}

/* ---------- tabs ---------- */
.edraft-tabs{display:flex;border-bottom:1px solid var(--ed-line-soft);flex:none;}
.edraft-tabs button{
  flex:1;padding:8px 4px;background:none;border:0;border-bottom:2px solid transparent;
  color:var(--ed-ink-3);font:inherit;font-size:11px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;cursor:pointer;
}
.edraft-tabs button:hover{color:var(--cyan);}
.edraft-tabs button.is-active{
  color:var(--cyan);border-bottom-color:var(--cyan);
  background:var(--cyan-soft);font-weight:1000;
  box-shadow:inset 0 -2px 0 var(--cyan);
}

/* ---------- segmented + inputs ---------- */
.edraft-seg{display:flex;width:100%;border:1px solid var(--line);}
.edraft-seg button{
  flex:1;padding:4px 9px;background:rgba(14,35,50,.9);border:0;color:var(--muted);
  font:inherit;font-size:10.5px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;
}
.edraft-seg button + button{border-left:1px solid var(--line);}
.edraft-seg button.is-active{background:var(--cyan-soft);color:var(--cyan);font-weight:1000;border-color:var(--line-strong);}
.edraft-field{
  display:flex;align-items:center;gap:6px;padding:0 9px;width:100%;
  background:rgba(14,35,50,.9);border:1px solid var(--line);
}
.edraft-field .edraft-action-icon,.edraft-field .edraft-icon-well{color:var(--cyan);}
.edraft-input{
  width:100%;padding:6px 0;background:none;border:0;color:var(--text);
  font:inherit;font-size:12.5px;outline:none;
}
.edraft-input::placeholder{color:var(--muted-2);}
.edraft-select{
  width:100%;padding:5px 7px;background:rgba(14,35,50,.9);border:1px solid var(--line);
  color:var(--text);font:inherit;font-size:11.5px;outline:none;
}
.edraft-board-tools-row{width:100%;}

/* ---------- draft log ---------- */
.edraft-log-tools{display:flex;flex-direction:column;gap:7px;padding:9px 12px;border-bottom:1px solid var(--ed-line-soft);flex:none;}
.edraft-log-round{
  padding:6px 12px;font-family:var(--ed-head);font-size:12px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--ed-ink-3);background:rgba(255,255,255,.028);
  border-block:1px solid var(--ed-line-soft);position:sticky;top:0;z-index:1;backdrop-filter:blur(4px);
}
.edraft-log-hint{
  margin:0;padding:6px 12px 0;font-size:11px;color:var(--ed-ink-3);letter-spacing:.04em;
}
.edraft-log-row{
  display:grid;grid-template-columns:34px 22px minmax(0,1fr) auto auto;align-items:center;gap:8px;
  width:100%;padding:8px 12px;background:none;border:0;border-bottom:1px solid var(--ed-line-soft);
  text-align:left;color:inherit;font:inherit;cursor:pointer;
}
.edraft-log-row:hover{background:rgba(19,216,231,.06);}
.edraft-log-row.is-selected{
  background:var(--cyan-soft);
  box-shadow:inset 0 0 0 2px rgba(19,216,231,.55);
}
.edraft-log-row.is-user{box-shadow:inset 2px 0 0 var(--gold);}
.edraft-log-view{
  font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ed-cyan);opacity:.72;
}
.edraft-log-row.is-selected .edraft-log-view{opacity:1;font-weight:600;}
.edraft-log-num{font-family:var(--ed-head);font-size:16px;color:var(--ed-ink-3);}
.edraft-log-body{min-width:0;}
.edraft-log-body strong{display:block;font-size:14px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.edraft-log-body span{display:block;font-size:11.5px;color:var(--ed-ink-3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.edraft-empty{padding:22px 14px;color:var(--ed-ink-3);font-size:12.5px;text-align:center;}

/* ---------- badges ---------- */
.edraft-badge{
  display:inline-flex;align-items:center;padding:2px 7px;flex:none;
  font-size:9.5px;font-weight:900;letter-spacing:.11em;text-transform:uppercase;
  border:1px solid currentColor;
}
.edraft-badge.tone-steal{color:var(--green);background:var(--green-soft);border-color:rgba(82,223,148,.38);box-shadow:0 0 8px rgba(82,223,148,.18);}
.edraft-badge.tone-value,.edraft-badge.tone-safe{color:var(--green);background:var(--green-soft);border-color:rgba(82,223,148,.28);}
.edraft-badge.tone-reach{color:var(--orange);background:var(--orange-soft);border-color:rgba(255,138,76,.38);box-shadow:0 0 8px rgba(255,138,76,.18);}
.edraft-badge.tone-early{color:var(--muted);background:rgba(128,150,168,.12);border-color:rgba(128,150,168,.22);}
.edraft-badge.tone-expected,.edraft-badge.tone-bpa{color:var(--cyan);background:var(--cyan-soft);border-color:rgba(19,216,231,.38);}
.edraft-badge.tone-need{color:var(--blue);background:var(--blue-soft);border-color:rgba(138,180,255,.38);}
.edraft-badge.tone-offboard{color:var(--red);background:var(--red-soft);border-color:rgba(255,96,109,.28);}
.edraft-badge.tone-goalie{color:var(--red);background:var(--red-soft);border-color:rgba(255,96,109,.28);}
.edraft-badge.tone-default{color:var(--muted);background:rgba(128,150,168,.1);border-color:rgba(128,150,168,.18);}

/* ---------- centre stage ---------- */
.edraft-clockband{
  display:grid;grid-template-columns:auto auto 1fr auto;align-items:center;gap:12px;
  padding:10px 12px;flex:none;
  background:linear-gradient(180deg,rgba(11,31,45,.72),rgba(7,22,34,.72));
  border-bottom:1px solid var(--line);
}
.edraft-clockband.is-user{
  border-bottom-color:rgba(233,168,60,.42);
  background:
    radial-gradient(circle at 0% 0%, rgba(233,168,60,.18), transparent 56%),
    linear-gradient(180deg,rgba(42,28,20,.78),rgba(18,20,28,.74));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}
.edraft-onclock-num{font-family:var(--ed-head);font-size:24px;line-height:1;color:var(--gold);text-shadow:0 0 16px rgba(233,168,60,.32);}
.edraft-onclock-num small{display:block;font-size:9px;letter-spacing:.18em;color:var(--muted);font-weight:1000;}
.edraft-onclock-team h2{margin:0;font-family:var(--ed-head);font-size:18px;font-weight:1000;letter-spacing:.04em;line-height:1.1;color:var(--text);}
.edraft-onclock-state{
  display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:1000;
  letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);
}
.edraft-onclock-state.is-user{color:var(--gold);}
.edraft-onclock-meta{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px;}
.edraft-chip{
  display:inline-flex;align-items:center;gap:6px;padding:2px 8px;
  font-size:10px;letter-spacing:.06em;color:var(--text);
  background:rgba(255,255,255,.04);border:1px solid var(--line);
}
.edraft-chip .edraft-icon-well{width:18px;height:18px;border-radius:6px;font-size:10px;}
.edraft-chip.is-need{color:var(--blue);border-color:rgba(138,180,255,.32);background:var(--blue-soft);}
.edraft-clock{font-family:var(--ed-head);font-size:20px;color:var(--cyan);text-shadow:0 0 12px rgba(19,216,231,.28);}
.edraft-batch-banner{
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  padding:8px 12px;font-size:12px;color:rgba(255,239,211,.96);
  background:
    radial-gradient(circle at 0% 0%, rgba(233,168,60,.18), transparent 56%),
    var(--gold-soft);
  border-bottom:1px solid rgba(233,168,60,.32);
}
.edraft-batch-banner strong{color:var(--gold);}

/* ---------- dossier ---------- */
.edraft-dossier{
  flex:1;min-height:0;padding:14px 12px 12px;display:flex;flex-direction:column;gap:12px;
  justify-content:flex-start;
}
.edraft-dossier-blank{
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
  flex:1;min-height:280px;color:var(--ed-ink-3);text-align:center;padding:24px;
}
.edraft-dossier-blank strong{font-family:var(--ed-head);font-size:16px;letter-spacing:.08em;text-transform:uppercase;color:var(--ed-ink-2);}
.edraft-idband{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:12px;}
.edraft-idcopy{min-width:0;}
.edraft-idflag{width:40px;height:28px;object-fit:cover;border:1px solid var(--ed-line);}
.edraft-idband h2{margin:0;font-family:var(--ed-head);font-size:26px;font-weight:1000;line-height:1;letter-spacing:.02em;color:var(--text);}
.edraft-idband p{margin:3px 0 0;font-size:13px;color:var(--muted);letter-spacing:.03em;}

.edraft-arc{
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:14px;
  padding:12px 16px;
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.1), transparent 50%),
    radial-gradient(circle at 100% 0%, rgba(233,168,60,.1), transparent 50%),
    rgba(6,21,34,.72);
  border:1px solid var(--line);border-radius:8px;
  box-shadow:var(--depth-registered);
}
.edraft-arc-node{display:flex;flex-direction:column;gap:2px;}
.edraft-arc-node span{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:900;}
.edraft-arc-node strong{font-family:var(--ed-head);font-size:28px;line-height:1;color:var(--text);}
.edraft-arc-node.is-peak strong{color:var(--blue);text-shadow:0 0 12px rgba(138,180,255,.22);}
.edraft-arc-node.is-right{text-align:right;align-items:flex-end;}
.edraft-arc-mid{display:flex;flex-direction:column;align-items:center;gap:4px;color:var(--muted);}
.edraft-arc-mid i{display:block;width:64px;height:1px;background:var(--line);}
.edraft-arc-mid em{font-style:normal;font-size:11px;color:var(--green);font-weight:1000;letter-spacing:.04em;}

.edraft-dossier-grid{
  flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) minmax(180px,.75fr);gap:12px;align-items:stretch;
}
.edraft-dossier-main,.edraft-dossier-side{display:flex;flex-direction:column;gap:14px;min-height:0;height:100%;}
.edraft-dossier-main .edraft-sec:last-child{flex:1;display:flex;flex-direction:column;min-height:0;}
.edraft-dossier-side .edraft-facts--stack{
  display:flex;flex-direction:column;gap:1px;background:var(--ed-line-soft);flex:1;min-height:0;
}
.edraft-dossier-side .edraft-fact{flex:1;display:flex;flex-direction:column;justify-content:center;padding:12px 14px;}
.edraft-fact.tone-good strong{color:var(--green);}
.edraft-fact.tone-warn strong{color:var(--orange);}
.edraft-dossier-actions{
  display:flex;flex-wrap:wrap;gap:8px;padding-top:4px;margin-top:auto;flex-shrink:0;
  border-top:1px solid var(--ed-line-soft);
}

.edraft-sec{display:flex;flex-direction:column;gap:8px;}
.edraft-sec-head{
  display:flex;align-items:center;gap:8px;padding-bottom:6px;border-bottom:1px solid var(--line);
}
.edraft-sec-head h4{
  margin:0;flex:1;font-family:var(--ed-head);font-size:12px;font-weight:1000;
  letter-spacing:.14em;text-transform:uppercase;color:var(--cyan);
}
.edraft-sec-head em{font-style:normal;font-size:10px;color:var(--muted);letter-spacing:.06em;}
.edraft-dna{display:flex;flex-direction:column;gap:7px;}
.edraft-dna-row{
  display:grid;grid-template-columns:minmax(80px,1fr) minmax(0,2fr) 36px;align-items:center;gap:10px;
}
.edraft-dna-row span{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:800;}
.edraft-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:1px;background:var(--line);}
.edraft-fact{
  display:flex;flex-direction:column;gap:3px;padding:10px 12px;background:rgba(6,21,34,.82);
}
.edraft-fact span{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:800;}
.edraft-fact strong{font-size:13px;color:var(--text);font-weight:700;line-height:1.3;}

.edraft-prod-strip{
  flex:1;display:grid;grid-template-columns:repeat(auto-fit,minmax(72px,1fr));gap:8px;align-items:stretch;
}
.edraft-prod-cell{
  display:flex;flex-direction:column;justify-content:center;gap:4px;padding:12px 10px;
  background:rgba(14,35,50,.9);border:1px solid var(--line);text-align:center;border-radius:8px;height:100%;
}
.edraft-prod-cell strong{font-family:var(--ed-head);font-size:28px;line-height:1;color:var(--text);}
.edraft-prod-cell span{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:800;}
.edraft-dna-track{height:6px;background:rgba(255,255,255,.07);border-radius:999px;overflow:hidden;}
.edraft-dna-track i{
  display:block;height:100%;background:var(--cyan);opacity:1;
  box-shadow:0 0 8px rgba(19,216,231,.45);
}
.edraft-dna-row.is-goalie .edraft-dna-track i{background:var(--red);box-shadow:0 0 8px rgba(255,96,109,.35);}
.edraft-dna-row strong{font-size:14px;text-align:right;color:var(--text);font-weight:900;}
.edraft-dock{
  flex:none;display:flex;align-items:center;justify-content:flex-end;gap:8px;
  padding:10px 12px;
  background:
    radial-gradient(circle at 100% 100%, rgba(19,216,231,.08), transparent 40%),
    var(--panel);
  border-top:1px solid var(--line);
  box-shadow:var(--depth-registered);
}
.edraft-btn--danger:hover{background:var(--red-soft)!important;}

.edraft-split{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.edraft-list{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:5px;}
.edraft-list li{
  position:relative;padding-left:13px;font-size:12.5px;line-height:1.42;color:var(--ed-ink-2);
}
.edraft-list li::before{content:"";position:absolute;left:0;top:7px;width:5px;height:5px;background:var(--ed-good);}
.edraft-list.is-risk li::before{background:var(--ed-bad);}
.edraft-list li b{display:block;color:var(--ed-ink);font-weight:600;}

.edraft-report{
  padding:10px 13px;font-size:12.8px;line-height:1.5;color:var(--muted);
  background:rgba(6,21,34,.72);border-left:2px solid var(--gold);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}
.edraft-report cite{display:block;margin-top:5px;font-style:normal;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted-2);}

.edraft-path{display:flex;align-items:center;gap:7px;flex-wrap:wrap;}
.edraft-path-node{
  padding:4px 11px;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;
  background:rgba(255,255,255,.04);border:1px solid var(--line);color:var(--muted);
}
.edraft-path-node.is-end{border-color:rgba(233,168,60,.42);color:var(--gold);box-shadow:0 0 8px rgba(233,168,60,.16);}
.edraft-path-sep{color:var(--muted-2);}

.edraft-meter{display:grid;grid-template-columns:110px 1fr 42px;align-items:center;gap:9px;}
.edraft-meter span{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted-2);}
.edraft-meter-track{height:5px;background:rgba(255,255,255,.07);border-radius:999px;overflow:hidden;}
.edraft-meter-track i{display:block;height:100%;background:var(--cyan);box-shadow:0 0 8px rgba(19,216,231,.45);}
.edraft-meter.is-risk .edraft-meter-track i{background:var(--red);box-shadow:0 0 8px rgba(255,96,109,.35);}
.edraft-meter strong{font-size:12px;text-align:right;color:var(--text);}

.edraft-callout{
  padding:9px 12px;font-size:12.2px;line-height:1.45;color:var(--muted);
  background:var(--blue-soft);border:1px solid rgba(138,180,255,.28);
}
.edraft-callout.is-warn{background:var(--red-soft);border-color:rgba(255,96,109,.28);color:var(--text);}

/* ---------- action dock ---------- */
.edraft-dock--root{position:relative;z-index:6;flex-shrink:0;}
.edraft-dock-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.edraft-error{
  margin:0;padding:6px 16px;color:var(--red);font-size:12px;
  background:var(--red-soft);border-top:1px solid rgba(255,96,109,.28);
}
.edraft-confirm-dialog{
  width:min(420px,100%);padding:18px 20px;
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.08), transparent 42%),
    var(--panel);
  border:1px solid var(--line);box-shadow:var(--shadow);
}
.edraft-confirm-dialog h3{
  margin:0 0 8px;font-family:var(--ed-head);font-size:17px;font-weight:1000;
  letter-spacing:.06em;text-transform:uppercase;color:var(--text);
}
.edraft-confirm-dialog p{margin:0 0 14px;font-size:13px;line-height:1.45;color:var(--ed-ink-2);}
.edraft-confirm-actions{display:flex;align-items:center;justify-content:flex-end;gap:8px;}

/* ---------- board ---------- */
.edraft-board-tools{display:flex;flex-direction:column;gap:7px;padding:9px 12px;border-bottom:1px solid var(--ed-line-soft);flex:none;}
.edraft-board-tools-row{display:flex;align-items:center;gap:6px;justify-content:space-between;}
.edraft-need-line{display:flex;flex-wrap:wrap;gap:4px;}
.edraft-board-cols{
  display:grid;grid-template-columns:32px minmax(0,1fr) 28px 72px 44px;gap:6px;
  padding:6px 12px;font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ed-ink-3);border-bottom:1px solid var(--ed-line-soft);flex:none;
}
.edraft-board-row{
  display:grid;grid-template-columns:32px minmax(0,1fr) 28px 72px 44px;gap:6px;align-items:center;
  width:100%;padding:8px 12px;border-bottom:1px solid var(--ed-line-soft);
  text-align:left;color:inherit;font:inherit;background:none;border-left:0;border-right:0;border-top:0;cursor:pointer;
}
.edraft-board-row:hover{background:rgba(19,216,231,.06);}
.edraft-board-row.is-selected{
  background:var(--gold-soft);
  box-shadow:inset 0 0 0 2px rgba(233,168,60,.55);
}
.edraft-board-row:focus-visible{outline:2px solid var(--gold);outline-offset:-2px;box-shadow:0 0 8px rgba(233,168,60,.24);}
.edraft-board-rank{font-family:var(--ed-head);font-size:15px;color:var(--muted);}
.edraft-board-name strong{display:block;font-size:14px;font-weight:700;line-height:1.2;color:var(--text);}
.edraft-board-name em{display:block;font-size:11px;color:var(--muted);font-style:normal;margin-top:2px;}
.edraft-board-pos{font-size:13px;font-weight:900;color:var(--cyan);}
.edraft-board-band{font-size:13px;color:var(--text);font-weight:1000;}
.edraft-board-stock{font-size:12px;color:var(--muted);}
.edraft-board-stock.is-soft{color:var(--muted);}
.edraft-board-stock.up{color:var(--green);font-weight:1000;}
.edraft-board-stock.down{color:var(--red);font-weight:1000;}
.edraft-board-cols.is-user-pick{grid-template-columns:32px minmax(0,1fr) 28px 72px 44px 36px;}
.edraft-board-row-wrap{
  display:grid;grid-template-columns:minmax(0,1fr) 40px;align-items:stretch;
  border-bottom:1px solid var(--ed-line-soft);
}
.edraft-board-row-wrap.is-selected .edraft-board-row{
  background:var(--gold-soft);
  box-shadow:inset 0 0 0 2px rgba(233,168,60,.55);
}
.edraft-board-row-wrap .edraft-board-row{border-bottom:0;}
.edraft-board-row-wrap .edraft-board-draft-btn{
  align-self:center;margin-right:6px;
  border:1px solid rgba(19,216,231,.35);background:rgba(19,216,231,.08);color:var(--cyan);
  border-radius:4px;cursor:pointer;
}
.edraft-board-row-wrap .edraft-board-draft-btn:hover{background:rgba(19,216,231,.18);}
.edraft-dossier-actions--draft{margin-top:8px;padding-top:10px;}
.edraft-drafted-banner{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:8px 14px;margin:0 12px;
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.14), transparent 56%),
    var(--cyan-soft);
  border:1px solid rgba(19,216,231,.28);
  font-size:12px;color:var(--muted);
}
.edraft-drafted-banner strong{color:var(--cyan);}
.edraft-chip{opacity:1;}
.edraft-board-name{min-width:0;}

/* ---------- feed (flat rows, no bubbles) ---------- */
.edraft-feed{padding:0;}
.edraft-feed .edraft-feed-row{
  display:grid;grid-template-columns:26px minmax(0,1fr);gap:9px;
  padding:9px 12px;border-bottom:1px solid var(--ed-line-soft);
}
.edraft-feed-av{
  width:26px;height:26px;display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,.06);border:1px solid var(--ed-line-soft);
  font-size:10px;font-weight:700;letter-spacing:.04em;color:var(--ed-ink-2);
}
.edraft-feed-meta{display:flex;gap:6px;align-items:baseline;font-size:10.5px;color:var(--ed-ink-3);}
.edraft-feed-meta b{color:var(--ed-ink);font-size:11.5px;}
.edraft-feed-row p{margin:3px 0 0;font-size:12.3px;line-height:1.45;color:var(--ed-ink-2);}

/* ---------- overlays ---------- */
.edraft-scrim{
  position:fixed;inset:0;z-index:14000;display:flex;align-items:center;justify-content:center;
  padding:24px;
  background:
    radial-gradient(circle at 50% 40%, rgba(19,216,231,.08), transparent 50%),
    rgba(3,6,10,.82);
  backdrop-filter:blur(3px);
}
.edraft-sheet{
  width:min(1080px,100%);max-height:88vh;display:flex;flex-direction:column;
  background:
    radial-gradient(circle at 0% 0%, rgba(19,216,231,.08), transparent 42%),
    var(--panel);
  border:1px solid var(--line);box-shadow:var(--shadow);
}
.edraft-sheet-head{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:12px 16px;border-bottom:1px solid var(--line);flex:none;
  background:
    radial-gradient(circle at 100% 0%, rgba(233,168,60,.08), transparent 40%),
    var(--panel-2);
}
.edraft-sheet-head h2,.edraft-sheet-head h3{
  margin:0;font-family:var(--ed-head);font-weight:500;font-size:21px;letter-spacing:.05em;text-transform:uppercase;
}
.edraft-sheet-head p{margin:2px 0 0;font-size:11.5px;color:var(--ed-ink-3);letter-spacing:.09em;}
.edraft-sheet-body{flex:1;min-height:0;overflow-y:auto;scrollbar-width:thin;}
.edraft-sheet-body::-webkit-scrollbar{width:7px;}
.edraft-sheet-body::-webkit-scrollbar-thumb{background:rgba(150,180,208,.2);}
.edraft-sheet-foot{
  display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-wrap:wrap;
  padding:11px 16px;border-top:1px solid var(--ed-line);flex:none;background:rgba(9,14,20,.9);
}
.edraft-confirm{margin-right:auto;font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ed-gold);}

.edraft-pick-modal{display:grid;grid-template-columns:300px minmax(0,1fr);flex:1;min-height:0;}
.edraft-pick-list{border-right:1px solid var(--ed-line-soft);display:flex;flex-direction:column;min-height:0;}
.edraft-pick-detail{display:flex;flex-direction:column;min-height:0;overflow-y:auto;scrollbar-width:thin;}
.edraft-pick-detail::-webkit-scrollbar{width:7px;}
.edraft-pick-detail::-webkit-scrollbar-thumb{background:rgba(150,180,208,.2);}
.edraft-pick-row{
  display:grid;grid-template-columns:20px 34px minmax(0,1fr) 26px auto;gap:7px;align-items:center;
  width:100%;padding:8px 12px;background:none;border:0;border-bottom:1px solid var(--ed-line-soft);
  text-align:left;font:inherit;color:inherit;cursor:pointer;
}
.edraft-pick-row:hover{background:rgba(19,216,231,.06);}
.edraft-pick-row.is-selected{
  background:var(--gold-soft);
  box-shadow:inset 0 0 0 2px rgba(233,168,60,.55);
}
.edraft-pick-row strong{font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.edraft-flag{width:18px;height:13px;object-fit:cover;border:1px solid var(--ed-line-soft);}

/* ---------- cpu reveal ---------- */
.edraft-reveal{
  position:absolute;inset:0;z-index:40;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:7px;text-align:center;
  background:radial-gradient(700px 380px at 50% 42%,rgba(20,34,48,.98),rgba(5,8,13,.99));
}
.edraft-reveal p{margin:0;font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:var(--ed-ink-3);}
.edraft-reveal h2{margin:2px 0;font-family:var(--ed-head);font-size:25px;font-weight:500;letter-spacing:.06em;}
.edraft-reveal h1{margin:2px 0;font-family:var(--ed-head);font-size:52px;font-weight:500;line-height:1;color:var(--ed-gold);}
.edraft-reveal-meta{display:flex;gap:7px;flex-wrap:wrap;justify-content:center;margin-top:5px;}

/* ---------- logos ---------- */
.edraft-logo{display:inline-flex;align-items:center;justify-content:center;flex:none;}
.edraft-logo img{width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 0 6px rgba(19,216,231,.28));}
.edraft-logo.xs{width:15px;height:15px;}
.edraft-logo.sm{width:21px;height:21px;}
.edraft-logo.md{width:31px;height:31px;}
.edraft-logo.lg{width:52px;height:52px;}
.edraft-logo-fb{
  font-family:var(--ed-head);font-size:11px;letter-spacing:.05em;color:var(--ed-ink-3);
  background:rgba(255,255,255,.05);border:1px solid var(--ed-line-soft);
}

/* ---------- intro ---------- */
.edraft-intro{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:13px;padding:40px 24px;text-align:center;position:relative;z-index:2;
}
.edraft-seal{
  width:112px;height:112px;display:flex;align-items:center;justify-content:center;
  font-family:var(--ed-head);font-size:29px;letter-spacing:.05em;color:var(--gold);
  border:1px solid rgba(233,168,60,.42);
  background:
    radial-gradient(circle at 50% 30%, rgba(233,168,60,.18), transparent 70%),
    rgba(6,21,34,.82);
  box-shadow:0 0 18px rgba(233,168,60,.22);
  clip-path:polygon(50% 0,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
}
.edraft-intro h1{margin:0;font-family:var(--ed-head);font-size:58px;font-weight:500;line-height:1;letter-spacing:.05em;text-transform:uppercase;}
.edraft-intro p{margin:0;color:var(--ed-ink-2);font-size:13.5px;letter-spacing:.09em;}
.edraft-intro-lines{display:flex;flex-direction:column;gap:5px;max-width:620px;margin-top:6px;}
.edraft-intro-lines span{
  font-size:12.5px;color:var(--muted);padding:6px 12px;
  background:rgba(6,21,34,.72);border-left:2px solid var(--gold);
  text-align:left;box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}

/* ---------- recap ---------- */
.edraft-recap{flex:1;min-height:0;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;position:relative;z-index:2;}
.edraft-recap-top{display:grid;grid-template-columns:270px minmax(0,1fr);gap:12px;}
.edraft-grade{
  display:flex;flex-direction:column;align-items:center;gap:7px;padding:18px 14px;
  background:linear-gradient(180deg,rgba(20,32,45,.9),rgba(11,17,25,.9));
  border:1px solid var(--ed-line);border-top:2px solid var(--rc-accent,var(--ed-gold));
}
.edraft-grade img{width:56px;height:56px;object-fit:contain;}
.edraft-grade-kicker{font-size:9.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--ed-ink-3);}
.edraft-grade-letter{font-family:var(--ed-head);font-size:76px;line-height:.9;color:var(--rc-accent,var(--ed-gold));}
.edraft-grade-meta{display:flex;gap:9px;font-size:11px;color:var(--ed-ink-3);}
.edraft-grade p{margin:4px 0 0;font-size:12px;color:var(--ed-ink-2);text-align:center;line-height:1.4;}
.edraft-metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--ed-line-soft);}
.edraft-metric{background:var(--ed-panel);padding:10px 11px;}
.edraft-metric strong{display:block;font-family:var(--ed-head);font-size:26px;line-height:1;}
.edraft-metric span{display:block;margin-top:2px;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ed-ink-3);}
.edraft-metric.good strong{color:var(--ed-good);}
.edraft-metric.bad strong{color:var(--ed-bad);}
.edraft-card{background:var(--ed-panel);border:1px solid var(--ed-line-soft);display:flex;flex-direction:column;}
.edraft-card-head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;border-bottom:1px solid var(--ed-line-soft);}
.edraft-card-head h4{margin:0;font-family:var(--ed-head);font-size:14px;font-weight:500;letter-spacing:.15em;text-transform:uppercase;}
.edraft-card-head span{font-size:10.5px;color:var(--ed-ink-3);}
.edraft-mid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;}
.edraft-dna-bar{display:flex;height:20px;margin:11px 12px 6px;overflow:hidden;}
.edraft-dna-seg{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#08111a;}
.edraft-dna-legend{display:flex;flex-wrap:wrap;gap:9px;padding:0 12px 11px;font-size:10.5px;color:var(--ed-ink-3);}
.edraft-dna-legend span{display:inline-flex;align-items:center;gap:4px;}
.edraft-dna-legend i{width:8px;height:8px;}
.edraft-needs{margin:0;padding:6px 12px 11px;list-style:none;display:flex;flex-direction:column;gap:5px;}
.edraft-needs li{display:grid;grid-template-columns:14px minmax(0,1fr) auto;gap:7px;align-items:center;font-size:12px;}
.edraft-needs li.is-filled{color:var(--ed-good);}
.edraft-needs li.is-open{color:var(--ed-ink-3);}
.edraft-table{display:flex;flex-direction:column;}
.edraft-thead,.edraft-trow{
  display:grid;grid-template-columns:34px 54px minmax(0,1.6fr) 40px 44px 44px minmax(0,1.2fr) 56px 18px;
  gap:7px;align-items:center;padding:7px 12px;
}
.edraft-thead{font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ed-ink-3);border-bottom:1px solid var(--ed-line-soft);}
.edraft-trow{border-bottom:1px solid var(--ed-line-soft);cursor:pointer;font-size:12.3px;}
.edraft-trow:hover{background:rgba(255,255,255,.04);}
.edraft-trow.is-best{background:var(--gold-soft);box-shadow:inset 0 0 0 2px rgba(233,168,60,.42);}
.edraft-trow .pos{color:var(--ed-good);}
.edraft-trow .neg{color:var(--ed-bad);}
.edraft-bottom{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
.edraft-vrow{
  display:flex;align-items:center;justify-content:space-between;gap:9px;width:100%;
  padding:7px 12px;border-bottom:1px solid var(--ed-line-soft);
  background:none;border-inline:0;border-top:0;color:inherit;font:inherit;font-size:12.2px;
  text-align:left;cursor:pointer;
}
.edraft-vrow:hover{background:rgba(255,255,255,.04);}
.edraft-vrow .pos{color:var(--ed-good);}
.edraft-vrow .neg{color:var(--ed-bad);}
.edraft-story{
  display:grid;grid-template-columns:6px minmax(0,1fr) 14px;gap:8px;align-items:start;width:100%;
  padding:8px 12px;background:none;border:0;border-bottom:1px solid var(--ed-line-soft);
  text-align:left;color:var(--ed-ink-2);font:inherit;font-size:12.2px;line-height:1.42;cursor:pointer;
}
.edraft-story i{width:6px;height:6px;margin-top:5px;background:var(--ed-gold);}
.edraft-story span{overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;}
.edraft-story.is-open span{-webkit-line-clamp:unset;}

.edraft-overlay-msg{
  position:absolute;inset:0;z-index:35;display:flex;align-items:center;justify-content:center;
  background:rgba(5,9,14,.72);font-family:var(--ed-head);font-size:20px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ed-ink-2);
}

@media (max-width:1400px){
  .edraft-floor{grid-template-columns:232px minmax(0,1fr) 302px;}
  .edraft-bottom{grid-template-columns:repeat(2,1fr);}
}
@media (max-width:1180px){
  .edraft-floor{grid-template-columns:minmax(0,1fr) 300px;}
  .edraft-pane--log{display:none;}
  .edraft-recap-top{grid-template-columns:1fr;}
  .edraft-metrics{grid-template-columns:repeat(3,1fr);}
}
`;

/* ------------------------------------------------------------------ */
/* Small presentational pieces                                         */
/* ------------------------------------------------------------------ */

function TeamLogo({ teamId, teamName, size = "md" }) {
  const src = getTeamLogoSrc({ team_id: teamId, name: teamName });
  if (!src) {
    return <span className={`${PREFIX}-logo ${size} ${PREFIX}-logo-fb`}>{teamAbbrev(teamId, teamName) || "TM"}</span>;
  }
  return (
    <span className={`${PREFIX}-logo ${size}`}>
      <img src={src} alt="" loading="lazy" />
    </span>
  );
}

function ProspectFlag({ country, code, width = 18 }) {
  const iso = resolveCountryCode(code || country);
  const src = iso ? `https://flagcdn.com/w160/${iso.toLowerCase()}.png` : flagApiUrl(code || country, 32);
  if (!src) return <span className={`${PREFIX}-flag`} aria-hidden="true" />;
  return (
    <img
      className={`${PREFIX}-flag`}
      src={src}
      alt={iso ? `${iso} flag` : ""}
      style={{ width, height: Math.round(width * 0.72) }}
      loading="lazy"
      onError={(e) => { e.currentTarget.style.visibility = "hidden"; }}
    />
  );
}

function PickBadge({ tag, tone }) {
  if (!tag) return null;
  return <span className={`${PREFIX}-badge tone-${tone || "default"}`}>{tag}</span>;
}

function Chip({ children, variant, icon }) {
  if (children == null || children === "") return null;
  return (
    <span className={`${PREFIX}-chip${variant ? ` is-${variant}` : ""}`}>
      {icon ? <Icon name={icon} size={11} tone={ICON_TONES[icon]} well /> : null}
      {children}
    </span>
  );
}

function SectionHead({ title, meta, icon, tone }) {
  return (
    <div className={`${PREFIX}-sec-head`}>
      {icon ? <Icon name={icon} size={13} tone={tone || ICON_TONES[icon]} well /> : null}
      <h4>{title}</h4>
      {meta ? <em>{meta}</em> : null}
    </div>
  );
}

function Fact({ label, value, tone }) {
  if (value == null || value === "") return null;
  return (
    <div className={`${PREFIX}-fact${tone ? ` tone-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function resolveAvailableProspect(selected, available) {
  const rows = safeArray(available);
  if (!rows.length) return null;
  if (selected) {
    const sid = getId(selected);
    const match = rows.find((p) => getId(p) === sid);
    if (match) return match;
  }
  return rows[0];
}

function ConfirmDialog({ open, title, message, confirmLabel = "Confirm", danger = false, loading = false, onConfirm, onCancel }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onCancel?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return createPortal(
    <div className={`${PREFIX}-scrim`} role="dialog" aria-modal="true" aria-label={title} onClick={onCancel}>
      <div className={`${PREFIX}-confirm-dialog`} onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        {message ? <p>{message}</p> : null}
        <div className={`${PREFIX}-confirm-actions`}>
          <button type="button" className="nhlcal-quick-link" disabled={loading} onClick={onCancel}>Cancel</button>
          <button
            type="button"
            className={danger ? "nhlcal-quick-link edraft-btn--danger" : "nhlcal-advance-button"}
            disabled={loading}
            onClick={onConfirm}
          >
            {loading ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* Command bar + on-deck rail                                          */
/* ------------------------------------------------------------------ */

function CommandBar({
  franchiseState,
  draft,
  currentRound,
  overallNow,
  completedCount,
  totalPicks,
  draftDone,
  isUserPick,
  onBack,
}) {
  const phaseLabel = draftDone
    ? "Draft complete"
    : isUserPick
      ? `Round ${Math.min(currentRound, ROUNDS)} · Your pick #${overallNow || "—"} · ${completedCount}/${totalPicks}`
      : `Round ${Math.min(currentRound, ROUNDS)} · Pick #${overallNow || "—"} · ${completedCount}/${totalPicks}`;

  return (
    <header className={`${PREFIX}-command`}>
      <div className={`${PREFIX}-command-left`}>
        <button type="button" className="nhlcal-quick-link" onClick={onBack}>
          ← Hub
        </button>
        <div className={`${PREFIX}-titles`}>
          <h1>Entry Draft</h1>
          <p>{[seasonLabel(franchiseState), draftYearLabel(draft), draft?.location].filter(Boolean).join(" · ")}</p>
        </div>
      </div>

      <div className={`${PREFIX}-command-mid`}>
        <p className={`${PREFIX}-phase${isUserPick && !draftDone ? "" : " is-cpu"}`}>{phaseLabel}</p>
      </div>

      <div className={`${PREFIX}-command-right`}>
        {draft?.class_strength ? <Chip>{draft.class_strength}</Chip> : null}
      </div>
    </header>
  );
}

function DraftActionDock({
  isUserPick,
  draftDone,
  loading,
  loadingOp,
  stageProspect,
  available,
  tradeOffers,
  onSimPick,
  onSimToUser,
  onSimFullDraft,
  onMakeSelection,
  onDraftProspect,
  onTradeDown,
}) {
  const dockProspect = resolveAvailableProspect(stageProspect, available);
  return (
    <div className={`${PREFIX}-dock ${PREFIX}-dock--root`}>
      <div className={`${PREFIX}-dock-actions`}>
            {isUserPick ? (
              <>
                <button
                  type="button"
                  className="nhlcal-advance-button"
                  disabled={loading}
                  onClick={() => (dockProspect ? onDraftProspect?.(dockProspect) : onMakeSelection?.())}
                >
                  <Icon name="check" size={12} />
                  {dockProspect ? `Draft ${getPlayerName(dockProspect)}` : "Select prospect"}
                </button>
                <button type="button" className="nhlcal-advance-button nhlcal-advance-button-secondary" disabled={loading} onClick={onMakeSelection}>
                  Browse board
                </button>
                <button type="button" className="nhlcal-quick-link" disabled={loading} onClick={onTradeDown}>
                  Trade down{tradeOffers?.length ? ` (${tradeOffers.length})` : ""}
                </button>
              </>
            ) : (
          <>
            <button type="button" className="nhlcal-advance-button" disabled={loading || draftDone} onClick={onSimPick}>
              {loadingOp === "cpu" ? LOADING_COPY.cpu : "Sim pick"}
            </button>
            <button type="button" className="nhlcal-advance-button nhlcal-advance-button-secondary" disabled={loading || draftDone} onClick={onSimToUser}>
              {loadingOp === "simUser" ? LOADING_COPY.simUser : "Sim to my pick"}
            </button>
            <button type="button" className="nhlcal-quick-link edraft-btn--danger" disabled={loading || draftDone} onClick={onSimFullDraft}>
              {loadingOp === "complete" ? LOADING_COPY.complete : "Complete draft"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function OnDeckRail({ upcoming, userTeamId }) {
  const slots = safeArray(upcoming);
  if (!slots.length) return null;
  return (
    <div className={`${PREFIX}-ondeck`}>
      <span className={`${PREFIX}-ondeck-label`}>On deck</span>
      <div className={`${PREFIX}-ondeck-track`}>
        {slots.map((slot, idx) => {
          const isUser = String(slot.team_id) === String(userTeamId);
          return (
            <div
              key={slot.overall_pick}
              className={`${PREFIX}-slot${isUser ? " is-user" : ""}${idx === 0 ? " is-next" : ""}`}
              title={slot.team_name || ""}
            >
              <span className={`${PREFIX}-slot-num ${PREFIX}-tabular`}>{slot.overall_pick}</span>
              <TeamLogo teamId={slot.team_id} teamName={slot.team_name} size="sm" />
              <span>
                <span className={`${PREFIX}-slot-abbr`}>{teamAbbrev(slot.team_id, slot.team_name)}</span>
                {slot.is_traded && (slot.via_team_name || slot.via_team_id) ? (
                  <span className={`${PREFIX}-slot-via`}>via {teamAbbrev(slot.via_team_id, slot.via_team_name)}</span>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Left pane — draft log + floor buzz                                  */
/* ------------------------------------------------------------------ */

function LeftPane({ completed, selectedPick, onSelectPick, userTeamId, tweets, feedEnabled }) {
  const [pane, setPane] = useState("log");
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const rows = safeArray(completed);

  const filtered = useMemo(() => {
    let list = rows.slice().reverse();
    if (filter === "mine") list = list.filter((p) => String(p.team_id) === String(userTeamId));
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((p) => {
        const name = String(p.prospect_name || getPlayerName(p) || "").toLowerCase();
        const team = String(p.team_name || "").toLowerCase();
        return name.includes(q) || team.includes(q) || String(p.overall_pick).includes(q);
      });
    }
    return list;
  }, [rows, filter, search, userTeamId]);

  const grouped = useMemo(() => {
    const out = [];
    let lastRound = null;
    for (const p of filtered) {
      if (p.round !== lastRound) {
        out.push({ type: "round", round: p.round, key: `r-${p.round}-${p.overall_pick}` });
        lastRound = p.round;
      }
      out.push({ type: "pick", pick: p, key: `p-${p.overall_pick}` });
    }
    return out;
  }, [filtered]);

  return (
    <aside className={`${PREFIX}-pane ${PREFIX}-pane--log`}>
      <div className={`${PREFIX}-tabs`}>
        <button type="button" className={pane === "log" ? "is-active" : ""} onClick={() => setPane("log")}>
          Draft Log
        </button>
        <button type="button" className={pane === "buzz" ? "is-active" : ""} onClick={() => setPane("buzz")}>
          Floor Buzz
        </button>
      </div>

      {pane === "log" ? (
        <>
          {rows.length >= 18 ? (
            <div className={`${PREFIX}-log-tools`}>
              <div className={`${PREFIX}-seg`}>
                <button type="button" className={filter === "all" ? "is-active" : ""} onClick={() => setFilter("all")}>All</button>
                <button type="button" className={filter === "mine" ? "is-active" : ""} onClick={() => setFilter("mine")}>Mine</button>
              </div>
              <label className={`${PREFIX}-field`}>
                <input
                  type="search"
                  className={`${PREFIX}-input`}
                  placeholder="Player, club, pick"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  aria-label="Search draft log"
                />
              </label>
            </div>
          ) : rows.length ? (
            <div className={`${PREFIX}-log-tools`}>
              <div className={`${PREFIX}-seg`}>
                <button type="button" className={filter === "all" ? "is-active" : ""} onClick={() => setFilter("all")}>All</button>
                <button type="button" className={filter === "mine" ? "is-active" : ""} onClick={() => setFilter("mine")}>Mine</button>
              </div>
            </div>
          ) : null}

          <div className={`${PREFIX}-scroll`}>
            {!rows.length ? (
              <p className={`${PREFIX}-empty`}>No selections yet.</p>
            ) : (
              grouped.map((row) => {
                if (row.type === "round") {
                  return <div key={row.key} className={`${PREFIX}-log-round`}>Round {row.round}</div>;
                }
                const p = row.pick;
                const { tag, tone } = getPickDisplayTag(p);
                const selected = selectedPick?.overall_pick === p.overall_pick;
                const mine = String(p.team_id) === String(userTeamId);
                return (
                  <button
                    type="button"
                    key={row.key}
                    className={`${PREFIX}-log-row${selected ? " is-selected" : ""}${mine ? " is-user" : ""}`}
                    onClick={() => onSelectPick?.(p)}
                    title="View scouting card"
                    aria-label={`View ${p.prospect_name || getPlayerName(p)} draft card`}
                  >
                    <span className={`${PREFIX}-log-num ${PREFIX}-tabular`}>{p.overall_pick}</span>
                    <TeamLogo teamId={p.team_id} teamName={p.team_name} size="sm" />
                    <span className={`${PREFIX}-log-body`}>
                      <strong>{p.prospect_name || getPlayerName(p)}</strong>
                      <span>
                        {[
                          p.position || getPlayerPosition(p),
                          teamAbbrev(p.team_id, p.team_name),
                          p.is_traded && p.via_team_name ? `via ${teamAbbrev(p.via_team_id, p.via_team_name)}` : null,
                          getPublicRankAtPick(p) != null ? `Brd ${getPublicRankAtPick(p)}` : null,
                        ].filter(Boolean).join(" · ")}
                      </span>
                    </span>
                    <PickBadge tag={tag} tone={tone} />
                  </button>
                );
              })
            )}
          </div>
        </>
      ) : (
        <div className={`${PREFIX}-scroll ${PREFIX}-feed`}>
          {feedEnabled && safeArray(tweets).length ? (
            <FanReactionFeed
              enabled
              reactions={tweets}
              eventType="entry_draft"
              visibleCount={6}
              intervalMs={7000}
              maxTweets={28}
              feedLabel="Draft Floor"
              feedSubLabel="Live reactions"
              className={`${PREFIX}-feed-inner`}
            />
          ) : (
            <p className={`${PREFIX}-empty`}>The floor goes quiet until the first name is called.</p>
          )}
        </div>
      )}
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/* Right pane — best available board                                   */
/* ------------------------------------------------------------------ */

function BoardPane({
  available,
  selectedId,
  onSelectProspect,
  isUserPick = false,
  onDraftProspect,
}) {
  const [search, setSearch] = useState("");
  const [posFilter, setPosFilter] = useState("all");
  const [sortBy, setSortBy] = useState("backend");
  const listRef = useRef(null);

  const filtered = useMemo(() => {
    let rows = safeArray(available).slice();

    if (posFilter !== "all") {
      rows = rows.filter((p) => {
        const pos = getPlayerPosition(p);
        if (posFilter === "W") return ["LW", "RW", "W"].includes(pos);
        return pos === posFilter;
      });
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((p) => String(getPlayerName(p) || "").toLowerCase().includes(q));
    }
    if (sortBy === "public") rows.sort((a, b) => (getRank(a) ?? 9999) - (getRank(b) ?? 9999));
    else if (sortBy === "team") rows.sort((a, b) => (getTeamRank(a) ?? 9999) - (getTeamRank(b) ?? 9999));
    else if (sortBy === "position") rows.sort((a, b) => String(getPlayerPosition(a)).localeCompare(String(getPlayerPosition(b))));
    else if (sortBy === "confidence") rows.sort((a, b) => (getConfidence(b) ?? -1) - (getConfidence(a) ?? -1));
    return rows;
  }, [available, posFilter, search, sortBy]);

  useEffect(() => {
    if (!listRef.current || !selectedId) return;
    const el = listRef.current.querySelector('[data-selected="true"]');
    if (el && typeof el.scrollIntoView === "function") el.scrollIntoView({ block: "nearest" });
  }, [selectedId, filtered]);

  return (
    <aside className={`${PREFIX}-pane`}>
      <div className={`${PREFIX}-pane-head`}>
        <h3>Best Available</h3>
        <em className={`${PREFIX}-tabular`}>{filtered.length}</em>
      </div>

      <div className={`${PREFIX}-board-tools`}>
        <label className={`${PREFIX}-field`}>
          <Icon name="search" size={13} tone="cyan" />
          <input
            type="search"
            className={`${PREFIX}-input`}
            placeholder="Search prospects"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search prospects"
          />
        </label>

        <div className={`${PREFIX}-seg`}>
          {["all", "C", "W", "D", "G"].map((f) => (
            <button key={f} type="button" className={posFilter === f ? "is-active" : ""} onClick={() => setPosFilter(f)}>
              {f === "all" ? "All" : f}
            </button>
          ))}
        </div>

        <div className={`${PREFIX}-board-tools-row`}>
          <select className={`${PREFIX}-select`} value={sortBy} onChange={(e) => setSortBy(e.target.value)} aria-label="Sort board">
            <option value="backend">Board order</option>
            <option value="public">Public rank</option>
            <option value="team">Your rank</option>
            <option value="position">Position</option>
            <option value="confidence">Confidence</option>
          </select>
        </div>
      </div>

      <div className={`${PREFIX}-board-cols${isUserPick ? " is-user-pick" : ""}`} aria-hidden="true">
        <span>Brd</span><span>Prospect</span><span>Pos</span><span>OVR–Pot</span><span>Conf</span>
        {isUserPick ? <span>Draft</span> : null}
      </div>

      <div className={`${PREFIX}-scroll`} ref={listRef} role="listbox" aria-label="Best available prospects">
        {!filtered.length ? (
          <p className={`${PREFIX}-empty`}>No prospects match those filters.</p>
        ) : (
          filtered.map((p) => {
            const pub = getRank(p);
            const ovr = resolveCurrentOvrLabel(p);
            const pot = resolvePotentialOvrLabel(p);
            const conf = formatConfidence(p);
            const confTitle = getConfidenceTitle(p);
            const softConf = isSoftConfidence(p);
            const selected = selectedId === getId(p);
            return (
              <div
                key={getId(p)}
                className={`${PREFIX}-board-row-wrap${selected ? " is-selected" : ""}`}
                data-selected={selected ? "true" : undefined}
              >
                <button
                  type="button"
                  role="option"
                  aria-selected={selected}
                  className={`${PREFIX}-board-row${selected ? " is-selected" : ""}`}
                  title={p.stock_reason || confTitle || undefined}
                  onClick={() => onSelectProspect?.(p)}
                >
                  <span className={`${PREFIX}-board-rank ${PREFIX}-tabular`}>{pub != null ? pub : "—"}</span>
                  <span className={`${PREFIX}-board-name`}>
                    <strong>{getPlayerName(p)}</strong>
                  </span>
                  <span className={`${PREFIX}-board-pos`}>{getPlayerPosition(p)}</span>
                  <span className={`${PREFIX}-board-band ${PREFIX}-tabular`}>
                    {[ovr, pot].filter(Boolean).join(" › ") || "—"}
                  </span>
                  <span
                    className={`${PREFIX}-board-stock ${PREFIX}-tabular${softConf ? " is-soft" : ""}`}
                    title={confTitle || undefined}
                  >
                    {conf || "—"}
                  </span>
                </button>
                {isUserPick && onDraftProspect ? (
                  <button
                    type="button"
                    className={`${PREFIX}-board-draft-btn`}
                    title={`Draft ${getPlayerName(p)}`}
                    aria-label={`Draft ${getPlayerName(p)}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDraftProspect(p);
                    }}
                  >
                    <Icon name="check" size={11} />
                  </button>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/* Centre stage — scouting dossier (position aware)                    */
/* ------------------------------------------------------------------ */

function SkillDna({ prospect, confidence, omitKeys = [] }) {
  const omit = new Set(omitKeys.map((k) => String(k).toLowerCase()));
  const hidden = chapterRatingsHidden(prospect);
  const fogged = chapterRatingsFogged(prospect);
  const rows = chapterAttributeRows(prospect).filter(([label]) => !omit.has(String(label).toLowerCase()));
  const goalie = isGoalie(prospect);
  const soft = fogged || (Number.isFinite(Number(confidence)) && Number(confidence) < 45);
  if (hidden) {
    return (
      <section className={`${PREFIX}-sec`}>
        <SectionHead
          title={goalie ? "Goalie chapters" : "Chapter ratings"}
          meta="Scout to reveal"
          icon="chart"
          tone="blue"
        />
        <p className={`${PREFIX}-muted`}>Chapter grades stay hidden until you build a dedicated scouting file on this prospect.</p>
      </section>
    );
  }
  if (!rows.length) return null;
  return (
    <section className={`${PREFIX}-sec`}>
      <SectionHead
        title={goalie ? "Goalie chapters" : "Chapter ratings"}
        meta={soft ? (fogged ? "Estimated bands" : "Soft confidence band") : null}
        icon="chart"
        tone="blue"
      />
      <div className={`${PREFIX}-dna`}>
        {rows.map(([label, value]) => {
          const band = typeof value === "object" && value?.band;
          const v = band
            ? Math.max(0, Math.min(100, Math.round(Number(value.mid ?? ((value.lo + value.hi) / 2)))))
            : Math.max(0, Math.min(100, Math.round(Number(value))));
          const labelText = band ? `${value.lo}–${value.hi}` : String(v);
          return (
            <div
              key={label}
              className={`${PREFIX}-dna-row${goalie ? " is-goalie" : ""}${soft ? " is-soft" : ""}`}
            >
              <span>{label}</span>
              <div className={`${PREFIX}-dna-track`}><i style={{ width: `${v}%` }} /></div>
              <strong className={`${PREFIX}-tabular`} title={soft ? getConfidenceTitle(prospect) || "Estimated band" : undefined}>
                {labelText}
              </strong>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function StageDossier({
  prospect,
  currentPickOverall,
  onFullReport,
  onShortlist,
  onPin,
  shortlisted,
  pinned,
}) {
  if (!prospect) {
    return (
      <div className={`${PREFIX}-dossier-blank`}>
        <strong>No prospect selected</strong>
        <p className={`${PREFIX}-muted`}>Pick a name off the board or click a pick in the draft log.</p>
      </div>
    );
  }

  const d = prospect.dossier || {};
  const goalie = isGoalie(prospect);
  const name = getPlayerName(prospect);
  const pos = getPlayerPosition(prospect);
  const nationality = d.nationality || prospect.nationality || prospect.country || null;
  const natCode = prospect.country_code || prospect.nat || prospect.nationality_code || nationality;
  const league = d.league || prospect.league || prospect.league_name || null;
  const club = d.team || prospect.team_name || prospect.team || null;
  const ageRaw = prospect.age ?? d.age;
  const shoots = prospect.handedness || prospect.shoots || prospect.shot_side || prospect.catches || null;
  const height = prospect.height || d.height || null;
  const weight = (prospect.weight || d.weight) ? `${roundInt(prospect.weight || d.weight)} lb` : null;

  const pub = getRank(prospect);
  const tb = getTeamRank(prospect);
  const curLabel = resolveCurrentOvrLabel(prospect);
  const potLabel = resolvePotentialOvrLabel(prospect);
  const curParts = String(curLabel || "").split("–").map((x) => roundInt(x)).filter((x) => x != null);
  const potParts = String(potLabel || "").split("–").map((x) => roundInt(x)).filter((x) => x != null);
  const curMid = curParts.length ? curParts.reduce((a, b) => a + b, 0) / curParts.length : null;
  const peakMid = potParts.length ? Math.max(...potParts) : roundInt(prospect.ceiling_grade);
  const growth = curMid != null && peakMid != null ? Math.max(0, Math.round(peakMid - curMid)) : null;

  const projection = d.projection || {};
  const archetype = d.player_comparison?.archetype || getDefiningTrait(prospect) || null;
  const readiness = formatNhlReadiness(prospect);
  const prod = resolveProductionStats(prospect, d.stats);
  const productionRows = goalie
    ? [
        prod.games ? ["GP", prod.games] : null,
        prod.savePct != null ? ["SV%", prod.savePct] : null,
        prod.gaa != null ? ["GAA", prod.gaa] : null,
        prod.wins ? ["W", prod.wins] : null,
        prod.shutouts ? ["SO", prod.shutouts] : null,
      ].filter(Boolean)
    : [
        prod.games ? ["GP", prod.games] : null,
        prod.goals || prod.goals === 0 ? ["G", prod.goals] : null,
        prod.assists || prod.assists === 0 ? ["A", prod.assists] : null,
        prod.points || prod.points === 0 ? ["PTS", prod.points] : null,
        prod.ppg != null ? ["P/GP", prod.ppg] : null,
      ].filter(Boolean);

  const reachDelta = pub != null && currentPickOverall != null ? currentPickOverall - pub : null;
  const identityLine = [
    pos,
    ageRaw != null ? `Age ${roundInt(ageRaw) ?? ageRaw}` : null,
    shoots ? `${shoots}${goalie ? " catch" : " shot"}` : null,
    nationality,
  ].filter(Boolean).join(" · ");

  const showActions = onShortlist || onPin || onFullReport;

  return (
    <div className={`${PREFIX}-dossier`}>
      <header className={`${PREFIX}-idband`}>
        <ProspectFlag country={nationality} code={natCode} width={40} />
        <div className={`${PREFIX}-idcopy`}>
          <h2>{name}</h2>
          <p>{identityLine}</p>
          <p className={`${PREFIX}-muted`}>{[club, league, height, weight].filter(Boolean).join(" · ")}</p>
        </div>
      </header>

      {(curLabel || potLabel) ? (
        <div className={`${PREFIX}-arc`}>
          <div className={`${PREFIX}-arc-node`}>
            <span>Current OVR</span>
            <strong className={`${PREFIX}-tabular`}>{curLabel || "—"}</strong>
          </div>
          <div className={`${PREFIX}-arc-mid`}>
            <i />
            {growth != null ? <em>+{growth} upside</em> : null}
          </div>
          <div className={`${PREFIX}-arc-node is-peak is-right`}>
            <span>Projected ceiling</span>
            <strong className={`${PREFIX}-tabular`}>{potLabel || "—"}</strong>
          </div>
        </div>
      ) : null}

      <div className={`${PREFIX}-dossier-grid`}>
        <div className={`${PREFIX}-dossier-main`}>
          <SkillDna prospect={prospect} confidence={getConfidence(prospect)} omitKeys={["Overall", "Potential"]} />

          {productionRows.length ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="Production" meta={league || null} icon="chart" tone="blue" />
              <div className={`${PREFIX}-prod-strip`}>
                {productionRows.map(([label, value]) => {
                  const formatted = formatStatCell(label, value);
                  if (formatted == null) return null;
                  return (
                    <div key={label} className={`${PREFIX}-prod-cell`}>
                      <span>{label}</span>
                      <strong className={`${PREFIX}-tabular`}>{formatted}</strong>
                    </div>
                  );
                })}
              </div>
            </section>
          ) : null}
        </div>

        <aside className={`${PREFIX}-dossier-side`}>
          <div className={`${PREFIX}-facts ${PREFIX}-facts--stack`}>
            <Fact label={goalie ? "Starter path" : "Projected role"} value={projection.label || archetype} />
            <Fact label="NHL readiness" value={readiness} />
            <Fact label="Public board" value={pub != null ? `#${pub}` : null} />
            <Fact label="Your board" value={tb != null ? `#${tb}` : null} />
            <Fact
              label="Value vs slot"
              tone={reachDelta == null ? undefined : reachDelta < 0 ? "warn" : reachDelta > 0 ? "good" : undefined}
              value={
                reachDelta == null
                  ? null
                  : reachDelta > 0
                    ? `Fell ${reachDelta}`
                    : reachDelta < 0
                      ? `Reach ${Math.abs(reachDelta)}`
                      : "On board"
              }
            />
          </div>
        </aside>
      </div>

      {showActions ? (
        <div className={`${PREFIX}-dossier-actions`}>
          {onShortlist ? (
            <button type="button" className="nhlcal-advance-button-secondary nhlcal-advance-button" onClick={() => onShortlist(prospect)}>
              {shortlisted ? "Remove shortlist" : "Shortlist"}
            </button>
          ) : null}
          {onPin ? (
            <button type="button" className="nhlcal-advance-button-secondary nhlcal-advance-button" onClick={() => onPin(prospect)}>
              {pinned ? "Unpin" : "Pin"}
            </button>
          ) : null}
          {onFullReport ? (
            <button type="button" className="nhlcal-quick-link" onClick={() => onFullReport(prospect)}>
              Full report
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Full report sheet (available prospect)                              */
/* ------------------------------------------------------------------ */

function FullReportSheet({ prospect, onClose, onDraft, isUserPick, loading }) {
  const [confirm, setConfirm] = useState(false);
  const closeRef = useRef(null);

  useEffect(() => {
    closeRef.current?.focus();
    setConfirm(false);
  }, [prospect]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!prospect) return null;
  const d = prospect.dossier || {};
  const notes = safeArray(prospect.scouting_event_notes);
  const history = safeArray(prospect.stock_history);
  const rights = prospect.rights_card || {};
  const character = d.character_read || null;

  return createPortal(
    <div className={`${PREFIX}-scrim`} role="presentation" onClick={onClose}>
      <div
        className={`${PREFIX}-sheet`}
        role="dialog"
        aria-modal="true"
        aria-label={`${getPlayerName(prospect)} full scouting report`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h2>{getPlayerName(prospect)}</h2>
            <p>{[getPlayerPosition(prospect), prospect.league || d.league, prospect.nationality || d.nationality].filter(Boolean).join(" · ")}</p>
          </div>
          <button type="button" className="nhlcal-quick-link" ref={closeRef} onClick={onClose}>
            <Icon name="close" size={12} /> Close
          </button>
        </header>

        <div className={`${PREFIX}-sheet-body`}>
          <StageDossier prospect={prospect} currentPickOverall={null} />

          {character ? (
            <div className={`${PREFIX}-dossier`}>
              <section className={`${PREFIX}-sec`}>
                <SectionHead
                  icon="shield"
                  title="Character read"
                  meta={character.confidence ? `${character.confidence}% confidence` : null}
                />
                <p className={`${PREFIX}-report`}>{character.headline || "Mixed reports"}</p>
                {safeArray(character.traits).length ? (
                  <div className={`${PREFIX}-facts`}>
                    {safeArray(character.traits).map((t) => (
                      <Fact key={t.label} label={t.label} value={t.tier || null} />
                    ))}
                  </div>
                ) : null}
                {character.interview_notes ? <p className={`${PREFIX}-callout`}>{character.interview_notes}</p> : null}
              </section>
            </div>
          ) : null}

          {notes.length ? (
            <div className={`${PREFIX}-dossier`}>
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="book" title="Staff notes" />
                <ul className={`${PREFIX}-list`}>
                  {notes.slice(0, 8).map((n, i) => (
                    <li key={i}>{typeof n === "string" ? n : n.text || n.note}</li>
                  ))}
                </ul>
              </section>
            </div>
          ) : null}

          {history.length ? (
            <div className={`${PREFIX}-dossier`}>
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="chart" title="Stock history" />
                <ul className={`${PREFIX}-list`}>
                  {history.slice(-8).map((h, i) => {
                    const r = num(h.rank ?? h.public_rank);
                    return (
                      <li key={i}>
                        <b>#{r != null ? r : "—"} · {h.date || h.stage || h.event_source || "Update"}</b>
                        {h.reason || h.stock_label || ""}
                      </li>
                    );
                  })}
                </ul>
              </section>
            </div>
          ) : null}

          {(rights.expected_role || rights.development_path || rights.rights_status) ? (
            <div className={`${PREFIX}-dossier`}>
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="shield" title="Rights outlook" />
                <div className={`${PREFIX}-facts`}>
                  <Fact label="Org role" value={rights.expected_role} />
                  <Fact label="Returning to" value={rights.returning_to || rights.development_path} />
                  <Fact label="Rights status" value={rights.rights_status} />
                  <Fact label="ELC" value={rights.elc_decision} />
                </div>
              </section>
            </div>
          ) : null}
        </div>

        {isUserPick ? (
          <footer className={`${PREFIX}-sheet-foot`}>
            {confirm ? (
              <>
                <span className={`${PREFIX}-confirm`}>Confirm this selection?</span>
                <button type="button" className="nhlcal-quick-link" onClick={() => setConfirm(false)}>Back</button>
                <button type="button" className={`nhlcal-advance-button`} disabled={loading} onClick={() => onDraft?.(prospect)}>
                  <Icon name="check" size={12} /> {loading ? LOADING_COPY.submit : "Submit card"}
                </button>
              </>
            ) : (
              <button type="button" className={`nhlcal-advance-button`} disabled={loading} onClick={() => setConfirm(true)}>
                <Icon name="check" size={12} /> Select {getPlayerName(prospect)}
              </button>
            )}
          </footer>
        ) : null}
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* Completed pick profile sheet                                        */
/* ------------------------------------------------------------------ */

function ReactionRow({ tweet }) {
  if (!tweet) return null;
  const fan = tweet.fan || {};
  const initials = String(fan.displayName || "RW")
    .trim().split(/\s+/).filter(Boolean).slice(0, 2)
    .map((p) => p[0] || "").join("").toUpperCase() || "RW";
  return (
    <div className={`${PREFIX}-feed-row`}>
      <span className={`${PREFIX}-feed-av`} aria-hidden="true">{initials}</span>
      <div>
        <div className={`${PREFIX}-feed-meta`}>
          <b>{fan.displayName}</b>
          {fan.handle ? <span>{fan.handle}</span> : null}
          {tweet.createdAtLabel ? <span>{tweet.createdAtLabel}</span> : null}
        </div>
        <p>{tweet.text}</p>
      </div>
    </div>
  );
}

function PickProfileSheet({ pick, reactionTweet, onClose }) {
  useEffect(() => {
    if (!pick) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pick, onClose]);

  if (!pick) return null;

  const name = pick.prospect_name || getPlayerName(pick);
  const pos = pick.position || getPlayerPosition(pick);
  const goalie = String(pos || "").toUpperCase() === "G";
  const nationality = pick.nationality || null;
  const natCode = pick.country_code || pick.nat || nationality;
  const overall = num(pick.overall_pick);
  const pub = getPublicRankAtPick(pick);
  const movement = describePickMovement(pick);
  const { tag, tone } = getPickDisplayTag(pick);
  const rights = pick.rights_card || {};
  const tags = safeArray(pick.pick_tags).filter(Boolean);

  const profile = [
    ["Age", num(pick.age)],
    ["Position", pos],
    [goalie ? "Catches" : "Shoots", pick.shoots || pick.handedness],
    ["League", pick.league || pick.league_name],
    ["Nationality", nationality],
    ["Eligibility", num(pick.draft_eligibility_year)],
  ].filter(([, v]) => v != null && v !== "");

  const projection = [
    ["Current OVR", num(pick.floor_grade) != null ? Math.round(num(pick.floor_grade)) : null],
    ["Projected ceiling", num(pick.ceiling_grade) != null ? Math.round(num(pick.ceiling_grade)) : null],
    ["Talent grade", pick.potential_grade],
    [goalie ? "Starter path" : "Projected role", pick.player_type || getDefiningTrait(pick)],
    ["NHL readiness", pick.nhl_readiness || formatNhlReadiness(pick)],
    ["Peak window", formatNhlPotentialEta(pick) || getNhlEta(pick)],
    ["Scout confidence", formatConfidence(pick)],
    ["Bust risk", pick.risk_score ?? getRisk(pick)],
  ].filter(([, v]) => v != null && v !== "");

  const value = [
    ["Overall pick", overall != null ? `#${overall}` : null],
    ["Round", num(pick.round) != null ? (num(pick.pick_in_round) != null ? `${pick.round} · #${pick.pick_in_round}` : String(pick.round)) : null],
    ["Public board", pub != null ? `#${pub}` : null],
    ["Preseason", getPreseasonRank(pick) != null ? `#${getPreseasonRank(pick)}` : null],
    ["Value vs slot", fmtSigned(getPickValueDelta(pick))],
    ["Movement", movement?.label],
  ].filter(([, v]) => v != null && v !== "");

  const rightsRows = [
    ["Rights through", rights.rights_through],
    ["Rights status", rights.rights_status],
    ["Returning to", rights.returning_to || pick.development_path],
    ["Org role", rights.expected_role],
    ["ELC", rights.elc_decision],
    ["Signing deadline", rights.rights_signing_deadline],
  ].filter(([, v]) => v != null && v !== "");

  return createPortal(
    <div className={`${PREFIX}-scrim`} role="presentation" onClick={onClose}>
      <div
        className={`${PREFIX}-sheet`}
        role="dialog"
        aria-modal="true"
        aria-label={`${name} draft profile`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className={`${PREFIX}-sheet-head`}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <ProspectFlag country={nationality} code={natCode} width={34} />
            <div>
              <h3>{name}</h3>
              <p>
                {[
                  overall != null ? `Pick #${overall}` : null,
                  pick.team_name,
                  pick.is_traded && pick.via_team_name ? `via ${pick.via_team_name}` : null,
                ].filter(Boolean).join(" · ")}
              </p>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <PickBadge tag={tag} tone={tone} />
            <button type="button" className="nhlcal-quick-link" onClick={onClose}>
              <Icon name="close" size={12} /> Close
            </button>
          </div>
        </header>

        <div className={`${PREFIX}-sheet-body`}>
          <div className={`${PREFIX}-dossier`}>
            {profile.length ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="book" title="Player profile" />
                <div className={`${PREFIX}-facts`}>
                  {profile.map(([l, v]) => <Fact key={l} label={l} value={v} />)}
                </div>
              </section>
            ) : null}

            {projection.length ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon={goalie ? "crease" : "chart"} title="Scouting & projection" />
                <div className={`${PREFIX}-facts`}>
                  {projection.map(([l, v]) => <Fact key={l} label={l} value={v} />)}
                </div>
              </section>
            ) : null}

            {value.length ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="target" title="Draft value" />
                <div className={`${PREFIX}-facts`}>
                  {value.map(([l, v]) => <Fact key={l} label={l} value={v} />)}
                </div>
              </section>
            ) : null}

            {pick.pick_reason ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="shield" title="Why this pick" />
                <p className={`${PREFIX}-report`}>{pick.pick_reason}</p>
              </section>
            ) : null}

            {(getBackendWhyWorks(pick) || getBackendWhyFails(pick)) ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="compare" title="Analysis" />
                {getBackendWhyWorks(pick) ? <p className={`${PREFIX}-callout`}>{getBackendWhyWorks(pick)}</p> : null}
                {getBackendWhyFails(pick) ? <p className={`${PREFIX}-callout is-warn`}>{getBackendWhyFails(pick)}</p> : null}
              </section>
            ) : null}

            {rightsRows.length ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="shield" title="Rights & development" />
                <div className={`${PREFIX}-facts`}>
                  {rightsRows.map(([l, v]) => <Fact key={l} label={l} value={v} />)}
                </div>
              </section>
            ) : null}

            {reactionTweet ? (
              <section className={`${PREFIX}-sec`}>
                <SectionHead icon="flame" title="Floor reaction" />
                <div className={`${PREFIX}-feed`}>
                  <ReactionRow tweet={reactionTweet} />
                </div>
              </section>
            ) : null}

            {tags.length ? (
              <div className={`${PREFIX}-need-line`}>
                {tags.slice(0, 6).map((t) => <Chip key={t}>{t}</Chip>)}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* Selection modal                                                     */
/* ------------------------------------------------------------------ */

function ComparisonTable({ prospects }) {
  const rows = safeArray(prospects).slice(0, 3);
  if (!rows.length) return <p className={`${PREFIX}-empty`}>Add up to three prospects to compare.</p>;
  const compareRows = [
    ["Public rank", (p) => (getRank(p) != null ? `#${getRank(p)}` : null)],
    ["Your rank", (p) => (getTeamRank(p) != null ? `#${getTeamRank(p)}` : null)],
    ["Position", (p) => getPlayerPosition(p) || null],
    ["League", (p) => p.league || p.league_name || null],
    ["Current", (p) => formatCurrentOvr(p)],
    ["Ceiling", (p) => getScoutedPotentialLabel(p)],
    ["Risk", (p) => getRisk(p)],
    ["Confidence", (p) => formatConfidence(p)],
    ["Fit", (p) => (getTeamFit(p) != null ? String(Math.round(getTeamFit(p))) : null)],
    ["ETA", (p) => getNhlEta(p)],
    ["Comparable", (p) => getComparable(p)],
  ].filter(([, render]) => rows.some((p) => render(p)));

  return (
    <div className={`${PREFIX}-dossier`}>
      <div className={`${PREFIX}-table`}>
        <div className={`${PREFIX}-thead`} style={{ gridTemplateColumns: `120px repeat(${rows.length},minmax(0,1fr))` }}>
          <span>Metric</span>
          {rows.map((p) => <span key={getId(p)}>{getPlayerName(p)}</span>)}
        </div>
        {compareRows.map(([label, render]) => (
          <div key={label} className={`${PREFIX}-trow`} style={{ gridTemplateColumns: `120px repeat(${rows.length},minmax(0,1fr))`, cursor: "default" }}>
            <span className={`${PREFIX}-muted`}>{label}</span>
            {rows.map((p) => <span key={`${getId(p)}-${label}`}>{render(p) || "—"}</span>)}
          </div>
        ))}
      </div>
    </div>
  );
}

function SelectionModal({ open, prospects, compareIds, onCompare, onDraft, onClose, loading, currentPick, draft, onTradeDown, focusProspect }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [mode, setMode] = useState("report");
  const [confirm, setConfirm] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const filtered = useMemo(() => {
    let rows = safeArray(prospects);
    if (filter === "C") rows = rows.filter((p) => getPlayerPosition(p) === "C");
    else if (filter === "D") rows = rows.filter((p) => getPlayerPosition(p) === "D");
    else if (filter === "G") rows = rows.filter((p) => getPlayerPosition(p) === "G");
    else if (filter === "W") rows = rows.filter((p) => ["LW", "RW", "W"].includes(getPlayerPosition(p)));
    else if (filter === "rising") rows = rows.filter((p) => num(p.rank_movement ?? p.stock_delta, 0) > 0);
    else if (filter === "falling") rows = rows.filter((p) => num(p.rank_movement ?? p.stock_delta, 0) < 0);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((p) => {
        const name = String(getPlayerName(p) || "").toLowerCase();
        const league = String(p.league || p.dossier?.league || "").toLowerCase();
        const country = String(p.nationality || p.country_code || "").toLowerCase();
        return name.includes(q) || league.includes(q) || country.includes(q);
      });
    }
    return rows;
  }, [prospects, filter, search]);

  const detail = selected || filtered[0] || null;
  const compareProspects = safeArray(prospects).filter((p) => compareIds.includes(getId(p))).slice(0, 3);

  useEffect(() => {
    if (!open) {
      setConfirm(false);
      setSubmitting(false);
      setMode("report");
      return;
    }
    if (focusProspect) {
      setSelected(focusProspect);
      setConfirm(false);
      setMode("report");
    }
  }, [open, focusProspect]);

  if (!open) return null;

  const handleDraft = async () => {
    if (!detail || submitting || loading) return;
    setSubmitting(true);
    try {
      await onDraft(detail);
    } finally {
      setSubmitting(false);
      setConfirm(false);
    }
  };

  return createPortal(
    <div className={`${PREFIX}-scrim`} role="dialog" aria-modal="true" aria-label="Make selection">
      <div className={`${PREFIX}-sheet`} style={{ width: "min(1240px,100%)", maxHeight: "92vh" }}>
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h2>You are on the clock</h2>
            <p>
              {[
                formatPick(currentPick?.overall_pick || draft?.overall_pick),
                draft?.current_team_name,
                currentPick?.round ? `Round ${currentPick.round}` : null,
              ].filter(Boolean).join(" · ")}
            </p>
          </div>
          <div style={{ display: "flex", gap: 7 }}>
            {typeof onTradeDown === "function" ? (
              <button type="button" className="nhlcal-quick-link" onClick={() => { onClose?.(); onTradeDown(); }}>
                <Icon name="trade" size={12} /> Trade down
              </button>
            ) : null}
            <button type="button" className="nhlcal-quick-link" onClick={onClose}>
              <Icon name="close" size={12} /> Close
            </button>
          </div>
        </header>

        <div className={`${PREFIX}-pick-modal`}>
          <div className={`${PREFIX}-pick-list`}>
            <div className={`${PREFIX}-board-tools`}>
              <label className={`${PREFIX}-field`}>
                <Icon name="search" size={13} />
                <input
                  type="search"
                  className={`${PREFIX}-input`}
                  placeholder="Name, league, country"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  aria-label="Search prospects"
                />
              </label>
              <div className={`${PREFIX}-seg`}>
                {["all", "C", "W", "D", "G", "rising", "falling"].map((f) => (
                  <button key={f} type="button" className={filter === f ? "is-active" : ""} onClick={() => setFilter(f)}>
                    {f === "all" ? "All" : f === "rising" ? "Up" : f === "falling" ? "Down" : f}
                  </button>
                ))}
              </div>
            </div>
            <div className={`${PREFIX}-scroll`} role="listbox" aria-label="Available prospects">
              {filtered.map((p) => {
                const d = p.dossier || {};
                const move = num(p.rank_movement ?? p.stock_delta ?? d.rankMovement);
                const active = getId(selected || detail) === getId(p);
                return (
                  <button
                    type="button"
                    key={getId(p)}
                    role="option"
                    aria-selected={active}
                    className={`${PREFIX}-pick-row${active ? " is-selected" : ""}`}
                    onClick={() => { setSelected(p); setConfirm(false); setMode("report"); }}
                  >
                    <ProspectFlag country={p.nationality || d.nationality} code={p.country_code || d.countryCode} />
                    <span className={`${PREFIX}-board-rank ${PREFIX}-tabular`}>{getRank(p) != null ? `#${getRank(p)}` : "—"}</span>
                    <strong>{getPlayerName(p)}</strong>
                    <span className={`${PREFIX}-board-pos`}>{getPlayerPosition(p)}</span>
                    {move != null && move !== 0 ? (
                      <span className={`${PREFIX}-board-stock ${move > 0 ? "up" : "down"}`}>{fmtSigned(move)}</span>
                    ) : <span />}
                  </button>
                );
              })}
              {!filtered.length ? <p className={`${PREFIX}-empty`}>No prospects match.</p> : null}
            </div>
          </div>

          <div className={`${PREFIX}-pick-detail`}>
            <div className={`${PREFIX}-tabs`}>
              <button type="button" className={mode === "report" ? "is-active" : ""} onClick={() => setMode("report")}>
                Scouting file
              </button>
              <button type="button" className={mode === "compare" ? "is-active" : ""} onClick={() => setMode("compare")}>
                Compare ({compareProspects.length})
              </button>
            </div>
            {mode === "report" ? (
              <StageDossier
                prospect={detail}
                currentPickOverall={currentPick?.overall_pick || draft?.overall_pick || null}
              />
            ) : (
              <ComparisonTable prospects={compareProspects.length ? compareProspects : (detail ? [detail] : [])} />
            )}
          </div>
        </div>

        <footer className={`${PREFIX}-sheet-foot`}>
          {detail ? (
            <button type="button" className="nhlcal-quick-link" onClick={() => onCompare(detail)}>
              <Icon name="compare" size={12} /> {compareIds.includes(getId(detail)) ? "Remove from compare" : "Add to compare"}
            </button>
          ) : null}
          {confirm ? (
            <>
              <span className={`${PREFIX}-confirm`}>Submit the card?</span>
              <button type="button" className="nhlcal-quick-link" onClick={() => setConfirm(false)}>Back</button>
              <button type="button" className={`nhlcal-advance-button`} disabled={loading || submitting} onClick={handleDraft}>
                <Icon name="check" size={12} /> {loading || submitting ? LOADING_COPY.submit : `Draft ${getPlayerName(detail)}`}
              </button>
            </>
          ) : (
            <button
              type="button"
              className={`nhlcal-advance-button`}
              disabled={loading || submitting || !detail}
              onClick={() => setConfirm(true)}
            >
              <Icon name="check" size={12} /> {detail ? `Draft ${getPlayerName(detail)}` : "Select a prospect"}
            </button>
          )}
        </footer>
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* Trade down                                                          */
/* ------------------------------------------------------------------ */

function TradePanel({ draft, onClose, onAccept, accepting }) {
  const offers = safeArray(draft?.trade_offers || draft?.draft_day_trade_offers || draft?.pick_trade_offers);
  return createPortal(
    <div className={`${PREFIX}-scrim`} role="dialog" aria-modal="true" aria-label="Trade down">
      <div className={`${PREFIX}-sheet`} style={{ width: "min(760px,100%)" }}>
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h3>Trade down</h3>
            <p>Clubs bidding to climb into your slot</p>
          </div>
          <button type="button" className="nhlcal-quick-link" onClick={onClose}>
            <Icon name="close" size={12} /> Close
          </button>
        </header>
        <div className={`${PREFIX}-sheet-body`}>
          {!offers.length ? (
            <p className={`${PREFIX}-empty`}>
              Nobody is paying to move up. Only clubs with a real board priority still available will bid — keep the pick or make your selection.
            </p>
          ) : (
            <div className={`${PREFIX}-dossier`}>
              {offers.map((offer, i) => {
                const candidates = safeArray(offer.target_candidates);
                const incoming = safeArray(
                  offer.incoming_assets?.length
                    ? offer.incoming_assets
                    : String(offer.assets_in || "").split(/\s*[·+]\s*/).map((s) => s.trim()).filter(Boolean)
                );
                const outgoing = safeArray(
                  offer.outgoing_assets?.length
                    ? offer.outgoing_assets
                    : [offer.on_clock_overall_pick ? `#${offer.on_clock_overall_pick} pick` : "On-clock pick"]
                );
                return (
                  <section key={i} className={`${PREFIX}-sec`}>
                    <SectionHead
                      icon="trade"
                      title={offer.team_name || offer.from_team_name || "Club"}
                      meta={offer.partner_overall_pick ? `Move to #${offer.partner_overall_pick}` : null}
                    />
                    <div className={`${PREFIX}-split`}>
                      <div>
                        <p className={`${PREFIX}-muted`}>You send</p>
                        <ul className={`${PREFIX}-list is-risk`}>
                          {outgoing.map((a, idx) => <li key={`o-${idx}`}>{String(a)}</li>)}
                        </ul>
                      </div>
                      <div>
                        <p className={`${PREFIX}-muted`}>You receive</p>
                        {incoming.length ? (
                          <ul className={`${PREFIX}-list`}>
                            {incoming.map((a, idx) => <li key={`i-${idx}`}>{String(a)}</li>)}
                          </ul>
                        ) : <p className={`${PREFIX}-muted`}>No package attached</p>}
                      </div>
                    </div>
                    {candidates.length ? (
                      <p className={`${PREFIX}-callout`}>
                        Rumoured targets: {candidates.map((c) => `${c.name || "Prospect"}${c.position ? ` (${c.position})` : ""}`).join(", ")}. Only one of them is their true priority.
                      </p>
                    ) : null}
                    <div className={`${PREFIX}-need-line`}>
                      {offer.slot_value_gap != null ? <Chip icon="chart">{`Chart gap ${Number(offer.slot_value_gap).toFixed(1)}`}</Chip> : null}
                      {offer.value || offer.value_grade ? <Chip icon="target">{offer.value || offer.value_grade}</Chip> : null}
                    </div>
                    <div>
                      <button
                        type="button"
                        className={`nhlcal-advance-button`}
                        disabled={accepting || !incoming.length}
                        onClick={() => onAccept?.(offer)}
                      >
                        <Icon name="check" size={12} /> {accepting ? "Accepting…" : "Accept and move down"}
                      </button>
                    </div>
                  </section>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* CPU reveal + round recap                                            */
/* ------------------------------------------------------------------ */

function CpuReveal({ pick, onDone, fast }) {
  useEffect(() => {
    const r = Number(pick?.round) || 1;
    const pace = fast ? 400 : r === 1 ? 2200 : r <= 3 ? 1400 : 700;
    const t = window.setTimeout(onDone, pace);
    return () => window.clearTimeout(t);
  }, [pick, onDone, fast]);

  if (!pick) return null;
  const pub = getPublicRankAtPick(pick);
  const movement = describePickMovement(pick);
  const { tag, tone } = getPickDisplayTag(pick);

  return (
    <div className={`${PREFIX}-reveal`}>
      <p>With the {formatPick(pick.overall_pick)} selection</p>
      <TeamLogo teamId={pick.team_id} teamName={pick.team_name} size="lg" />
      <h2>{pick.team_name}</h2>
      {pick.is_traded && (pick.via_team_name || pick.via_team_id) ? (
        <p>via {teamAbbrev(pick.via_team_id, pick.via_team_name)}</p>
      ) : null}
      <p>select</p>
      <h1>{pick.prospect_name}</h1>
      <div className={`${PREFIX}-reveal-meta`}>
        {pick.position ? <Chip>{pick.position}</Chip> : null}
        {pick.league ? <Chip>{pick.league}</Chip> : null}
        {pub != null ? <Chip icon="target">{`Board #${pub}`}</Chip> : null}
        {movement ? <Chip>{movement.label}</Chip> : null}
        <PickBadge tag={tag} tone={tone} />
      </div>
    </div>
  );
}

function RoundRecapPanel({ recap, onContinue }) {
  if (!recap) return null;
  return (
    <div className={`${PREFIX}-reveal`}>
      <p>Round complete</p>
      <h1>Round {recap.round}</h1>
      {recap.headline ? <p className={`${PREFIX}-muted`} style={{ maxWidth: 560 }}>{recap.headline}</p> : null}
      {safeArray(recap.user_picks).length ? (
        <div className={`${PREFIX}-reveal-meta`}>
          {safeArray(recap.user_picks).map((p, i) => (
            <Chip key={i} icon="check">{`${formatPick(p.overall_pick)} ${p.prospect_name}`}</Chip>
          ))}
        </div>
      ) : null}
      <button type="button" className={`nhlcal-advance-button`} onClick={onContinue}>
        <Icon name="next" size={12} /> Continue draft
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Recap                                                               */
/* ------------------------------------------------------------------ */

function Metric({ label, value, tone = "" }) {
  return (
    <div className={`${PREFIX}-metric ${tone}`}>
      <strong className={`${PREFIX}-tabular`}>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function StoryLine({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <button type="button" className={`${PREFIX}-story${open ? " is-open" : ""}`} onClick={() => setOpen((o) => !o)}>
      <i aria-hidden="true" />
      <span>{text}</span>
      <Icon name={open ? "up" : "down"} size={12} />
    </button>
  );
}

function ValueRow({ pick, tone, onOpen }) {
  const d = pubDelta(pick);
  return (
    <button type="button" className={`${PREFIX}-vrow`} onClick={onOpen}>
      <span>
        <b>{teamAbbrev(pick.team_id, pick.team_name)}</b> {formatPick(pick.overall_pick)} · {pick.prospect_name}
      </span>
      <span>
        {pick.final_rank != null ? <span className={`${PREFIX}-muted`}>Brd #{pick.final_rank} </span> : null}
        {d != null ? <span className={tone}>{fmtSigned(d)}</span> : null}
      </span>
    </button>
  );
}

function RecapShow({ draft, completed, userPicks }) {
  const r = draft?.recap || {};
  const [profilePick, setProfilePick] = useState(null);
  const picks = safeArray(userPicks).slice().sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));

  const teamId = picks[0]?.team_id ?? picks[0]?.team;
  const teamName = picks[0]?.team_name;
  const teamObj = { id: teamId, team_id: teamId, name: teamName, team_name: teamName };
  const logo = getTeamLogoSrc(teamObj);
  const accent = teamAccent(teamObj);

  const grade = r.user_grade || draft?.user_draft_grade || buildDraftGrade(picks) || "—";
  const summary = r.grade_summary ? clip15(r.grade_summary) : "";
  const posBreakdown = r.user_position_breakdown || r.position_breakdown || {};
  const steals = safeArray(r.best_steals || safeArray(completed).filter((p) => p.was_steal)).slice(0, 5);
  const reaches = safeArray(r.biggest_reaches || safeArray(completed).filter((p) => p.was_reach)).slice(0, 5);
  const headlines = safeArray(r.headlines).slice(0, 6);
  const needsReport = safeArray(r.needs_report);
  const stillOnBoard = safeArray(r.still_on_board);
  const bestId = String(r.best_user_pick?.prospect_id ?? "");

  const avgRaw = r.user_avg_value_delta != null
    ? Number(r.user_avg_value_delta)
    : (picks.length ? picks.reduce((s, p) => s + (pubDelta(p) || 0), 0) / picks.length : 0);

  const dnaEntries = Object.entries(posBreakdown).filter(([, v]) => Number(v) > 0);
  const dnaTotal = dnaEntries.reduce((s, [, v]) => s + Number(v || 0), 0);

  return (
    <div className={`${PREFIX}-recap`} style={{ "--rc-accent": accent }}>
      <div className={`${PREFIX}-recap-top`}>
        <div className={`${PREFIX}-grade`}>
          {logo ? <img src={logo} alt={teamName || "Team"} /> : null}
          <span className={`${PREFIX}-grade-kicker`}>Draft grade</span>
          <strong className={`${PREFIX}-grade-letter`}>{grade}</strong>
          <div className={`${PREFIX}-grade-meta`}>
            {r.user_grade_score != null ? <span>{r.user_grade_score}/100</span> : null}
            {r.user_class_rank ? <span>{r.user_class_rank}</span> : null}
          </div>
          {summary ? <p>{summary}</p> : null}
        </div>

        <div>
          <div className={`${PREFIX}-metrics`}>
            <Metric label="Picks" value={r.user_pick_count ?? picks.length} />
            <Metric label="Steals" value={r.user_steal_count ?? 0} tone="good" />
            <Metric label="Reaches" value={r.user_reach_count ?? 0} tone="bad" />
            <Metric label="Value adds" value={r.user_value_count ?? 0} tone="good" />
            <Metric label="Needs filled" value={r.user_need_count ?? 0} />
            <Metric label="Avg value" value={fmtSigned(avgRaw, 1) ?? "0.0"} tone={avgRaw >= 0 ? "good" : "bad"} />
          </div>

          <div className={`${PREFIX}-mid`}>
            <div className={`${PREFIX}-card`}>
              <div className={`${PREFIX}-card-head`}><h4>Class DNA</h4><span>{dnaTotal} picks</span></div>
              {dnaTotal ? (
                <>
                  <div className={`${PREFIX}-dna-bar`}>
                    {dnaEntries.map(([pos, v]) => {
                      const w = (Number(v) / dnaTotal) * 100;
                      return (
                        <div
                          key={pos}
                          className={`${PREFIX}-dna-seg`}
                          style={{ width: `${w}%`, background: RECAP_POS_COLORS[pos] || "#9cb2c4" }}
                          title={`${pos}: ${v}`}
                        >
                          {w >= 12 ? `${pos} ${v}` : ""}
                        </div>
                      );
                    })}
                  </div>
                  <div className={`${PREFIX}-dna-legend`}>
                    {dnaEntries.map(([pos, v]) => (
                      <span key={pos}><i style={{ background: RECAP_POS_COLORS[pos] || "#9cb2c4" }} />{pos} {v}</span>
                    ))}
                  </div>
                </>
              ) : <p className={`${PREFIX}-empty`}>No positional data.</p>}
            </div>

            <div className={`${PREFIX}-card`}>
              <div className={`${PREFIX}-card-head`}><h4>Needs report</h4><span>{needsReport.length} tracked</span></div>
              {needsReport.length ? (
                <ul className={`${PREFIX}-needs`}>
                  {needsReport.slice(0, 5).map((n, i) => (
                    <li key={i} className={n.filled ? "is-filled" : "is-open"} title={n.detail || n.category}>
                      <Icon name={n.filled ? "check" : "minus"} size={12} />
                      <span>{n.category}</span>
                      <span>{n.filled ? "Addressed" : "Open"}</span>
                    </li>
                  ))}
                </ul>
              ) : <p className={`${PREFIX}-empty`}>No pressing needs — the depth chart holds.</p>}
            </div>
          </div>
        </div>
      </div>

      <div className={`${PREFIX}-card`}>
        <div className={`${PREFIX}-card-head`}><h4>Your class</h4><span>Open a name for the full file</span></div>
        <div className={`${PREFIX}-table`}>
          <div className={`${PREFIX}-thead`}>
            <span>Rd</span><span>Pick</span><span>Player</span><span>Pos</span><span>OVR</span><span>Pot</span><span>Projection</span><span>Value</span><span />
          </div>
          {picks.length ? picks.map((p, i) => {
            const d = pubDelta(p);
            const isBest = bestId && String(p.prospect_id) === bestId;
            return (
              <div
                key={i}
                className={`${PREFIX}-trow${isBest ? " is-best" : ""}`}
                role="button"
                tabIndex={0}
                onClick={() => setProfilePick(p)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setProfilePick(p); } }}
              >
                <span className={`${PREFIX}-tabular`}>{p.round}</span>
                <span className={`${PREFIX}-tabular`}>{formatPick(p.overall_pick)}</span>
                <span>{p.prospect_name}</span>
                <span>{p.position || "—"}</span>
                <span className={`${PREFIX}-tabular`}>{ovrVal(p)}</span>
                <span className={`${PREFIX}-tabular`}>{potVal(p)}</span>
                <span>{projVal(p)}</span>
                <span className={d == null ? "" : d > 0 ? "pos" : d < 0 ? "neg" : ""}>{d == null ? "—" : fmtSigned(d)}</span>
                <Icon name="next" size={12} />
              </div>
            );
          }) : <p className={`${PREFIX}-empty`}>No selections were made.</p>}
        </div>
      </div>

      <div className={`${PREFIX}-bottom`}>
        <div className={`${PREFIX}-card`}>
          <div className={`${PREFIX}-card-head`}><h4>Best steals</h4></div>
          {steals.length ? steals.map((p, i) => (
            <ValueRow key={i} pick={p} tone="pos" onOpen={() => setProfilePick(p)} />
          )) : <p className={`${PREFIX}-empty`}>The board stayed honest.</p>}
        </div>

        <div className={`${PREFIX}-card`}>
          <div className={`${PREFIX}-card-head`}><h4>Biggest reaches</h4></div>
          {reaches.length ? reaches.map((p, i) => (
            <ValueRow key={i} pick={p} tone="neg" onOpen={() => setProfilePick(p)} />
          )) : <p className={`${PREFIX}-empty`}>A disciplined night.</p>}
        </div>

        <div className={`${PREFIX}-card`}>
          <div className={`${PREFIX}-card-head`}><h4>Class storylines</h4></div>
          {headlines.length ? headlines.map((h, i) => <StoryLine key={i} text={h} />)
            : <p className={`${PREFIX}-empty`}>A quiet draft night.</p>}
        </div>

        <div className={`${PREFIX}-card`}>
          <div className={`${PREFIX}-card-head`}><h4>Still on the board</h4></div>
          {stillOnBoard.length ? stillOnBoard.slice(0, 5).map((e, i) => (
            <div key={i} className={`${PREFIX}-vrow`} style={{ cursor: "default" }}>
              <span>{e.name}{e.position ? ` · ${e.position}` : ""}</span>
              <span className={`${PREFIX}-muted`}>{e.projected} › {e.status}</span>
            </div>
          )) : <p className={`${PREFIX}-empty`}>The board cleared out.</p>}
        </div>
      </div>

      {profilePick ? (
        <PickProfileSheet
          pick={profilePick}
          reactionTweet={buildDraftPickReactionTweet(profilePick, { seed: `${draft?.draft_year || "draft"}-recap-pick` })}
          onClose={() => setProfilePick(null)}
        />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Root                                                                */
/* ------------------------------------------------------------------ */

export default function EntryDraftMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const { mergeFranchiseState } = useGameUI();

  const rawDraft = liveDraftData(franchiseState, eventData);
  const [draft, setDraft] = useState(rawDraft || {});
  const [stage, setStage] = useState(() => {
    if (rawDraft?.draft_completed) return "recap";
    if (rawDraft?.draft_started) return "live";
    return "intro";
  });

  const [loadingOp, setLoadingOp] = useState(null);
  const [userModalOpen, setUserModalOpen] = useState(false);
  const [cpuReveal, setCpuReveal] = useState(null);
  const [batchMeta, setBatchMeta] = useState(null);
  const [roundRecapView, setRoundRecapView] = useState(null);
  const [simMode] = useState("manual");
  const [compareIds, setCompareIds] = useState([]);
  const [error, setError] = useState("");
  const [tradePanelOpen, setTradePanelOpen] = useState(false);
  const [tradeAccepting, setTradeAccepting] = useState(false);
  const [selectedCompletedPick, setSelectedCompletedPick] = useState(null);
  const [selectedAvailable, setSelectedAvailable] = useState(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [shortlistIds, setShortlistIds] = useState([]);
  const [pinnedIds, setPinnedIds] = useState([]);
  const [confirmDialog, setConfirmDialog] = useState(null);

  const simLock = useRef(false);
  const lastRoundRef = useRef(1);
  const userModalDismissedPickRef = useRef(null);
  const openUserAfterRevealRef = useRef(false);
  const draftHydratedRef = useRef(Boolean(rawDraft?.draft_started));
  const loading = Boolean(loadingOp);

  useEffect(() => {
    if (!rawDraft || !Object.keys(rawDraft).length) return;
    setDraft((prev) => {
      const prevCount = safeArray(prev?.completed_picks || prev?.draft_results).length;
      const nextCount = safeArray(rawDraft.completed_picks || rawDraft.draft_results).length;
      if (!draftHydratedRef.current) {
        draftHydratedRef.current = true;
        return rawDraft;
      }
      if (!prev?.draft_started && rawDraft.draft_started) return rawDraft;
      if (nextCount < prevCount) return prev;
      if (nextCount === prevCount && Number(rawDraft.overall_pick || 0) < Number(prev.overall_pick || 0)) return prev;
      if (nextCount > prevCount || Number(rawDraft.overall_pick || 0) > Number(prev.overall_pick || 0)) return rawDraft;
      return prev;
    });
  }, [rawDraft]);

  const completed = safeArray(draft.completed_picks || draft.draft_results);
  const available = safeArray(draft.available_prospects);
  const currentPick = draft.current_pick;
  const currentRound = Number(currentPick?.round || draft.current_round || 1);
  const isUserPick = Boolean(draft.is_user_pick);
  const draftDone = Boolean(draft.draft_completed);
  const uid = String(franchiseState?.user_team_id || "");
  const userPicks = useMemo(() => completed.filter((p) => String(p.team_id) === uid), [completed, uid]);
  const roundRecaps = draft.round_recaps || {};
  const orderNote = draft.draft_order_note;
  const storylines = safeArray(draft.storylines);
  const needs = safeArray(draft.team_needs);
  const phil = draft.team_philosophy || {};
  const draftOrder = safeArray(draft.draft_order);
  const totalPicks = draft.total_picks || 224;
  const overallNow = currentPick?.overall_pick || draft.overall_pick;
  const tradeOffers = safeArray(draft?.trade_offers || draft?.draft_day_trade_offers || draft?.pick_trade_offers);

  const upcomingPicks = useMemo(
    () => getUpcomingOrder(draftOrder, completed.length, uid, 14),
    [draftOrder, completed.length, uid]
  );


  const positionalNeeds = useMemo(
    () => needs.map(needLabel).filter((n) => n && !isPhilosophyNeedLabel(n)),
    [needs]
  );
  const primaryNeed = positionalNeeds[0] || null;

  const stageProspect = useMemo(
    () => resolveAvailableProspect(selectedAvailable, available),
    [selectedAvailable, available]
  );

  useEffect(() => {
    if (!selectedAvailable) return;
    const ids = new Set(safeArray(available).map(getId));
    if (!ids.has(getId(selectedAvailable))) {
      setSelectedAvailable(null);
    }
  }, [available, selectedAvailable, overallNow]);

  const draftSocialTweets = useMemo(() => {
    if (!completed.length) return [];
    const season = franchiseState?.season_year || franchiseState?.seasonYear || draft?.draft_year || "draft";
    return buildDraftFanTweets(completed, {
      maxTweets: 28,
      seed: `${season}-entry-draft`,
      draftContext: {
        franchiseSeed: String(franchiseState?.seed || franchiseState?.franchise_seed || season),
        primaryNeed,
        teamNeeds: needs,
        recentPositions: completed.slice(-5).map((p) => String(p.position || "").toUpperCase()),
      },
    });
  }, [completed, franchiseState, draft?.draft_year, primaryNeed, needs]);

  const selectedPickReaction = useMemo(() => {
    if (!selectedCompletedPick) return null;
    const overall = Number(selectedCompletedPick.overall_pick);
    const fromFeed = draftSocialTweets.find(
      (t) => Number(t?.context?.overallPick) === overall
        || String(t?.context?.winnerLabel || "").toLowerCase() === String(selectedCompletedPick.prospect_name || "").toLowerCase()
    );
    if (fromFeed) return fromFeed;
    const season = franchiseState?.season_year || franchiseState?.seasonYear || draft?.draft_year || "draft";
    return buildDraftPickReactionTweet(selectedCompletedPick, {
      seed: `${season}-entry-draft`,
      draftContext: {
        franchiseSeed: String(franchiseState?.seed || franchiseState?.franchise_seed || season),
        primaryNeed,
        teamNeeds: needs,
      },
    });
  }, [selectedCompletedPick, draftSocialTweets, franchiseState, draft?.draft_year, primaryNeed, needs]);

  const applyResponse = useCallback((res, { batch = false } = {}) => {
    const nextDraft = res?.draft ?? res?.state?.draft ?? null;
    if (res?.state) {
      mergeFranchiseState(nextDraft ? { ...res.state, draft: nextDraft } : res.state);
    } else if (nextDraft) {
      mergeFranchiseState({ draft: nextDraft });
    }
    if (nextDraft) setDraft(nextDraft);
    if (batch && res?.batch_summary) setBatchMeta(res.batch_summary);
    else if (res?.batch_summary) setBatchMeta(res.batch_summary);
    if (res?.draft?.recap) setDraft((d) => ({ ...d, recap: res.draft.recap }));
    else if (res?.recap) setDraft((d) => ({ ...d, recap: res.recap }));
  }, [mergeFranchiseState]);

  const acceptTradeOffer = useCallback(async (offer) => {
    setTradeAccepting(true);
    setError("");
    try {
      const res = await acceptEntryDraftTrade(offer);
      applyResponse(res);
      setTradePanelOpen(false);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Trade failed");
    } finally {
      setTradeAccepting(false);
    }
  }, [applyResponse]);

  const handleStart = useCallback(async () => {
    setLoadingOp("start");
    setError("");
    try {
      const res = draft.draft_started ? { draft } : await startEntryDraft();
      applyResponse(res);
      setStage("board");
      window.setTimeout(() => setStage("live"), 1600);
    } catch (e) {
      setError(e?.message || "Failed to start draft");
    } finally {
      setLoadingOp(null);
    }
  }, [applyResponse, draft]);

  const runCpuPick = useCallback(async () => {
    if (simLock.current) return;
    simLock.current = true;
    setLoadingOp("cpu");
    setError("");
    try {
      const res = await submitCpuDraftPick();
      applyResponse(res);
      if (res?.draft?.is_user_pick) {
        userModalDismissedPickRef.current = null;
      }
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || "CPU pick failed";
      const msg = String(Array.isArray(detail) ? detail.join(" ") : detail);
      if (/user.*on the clock|make a selection/i.test(msg)) {
        try {
          const sync = await getEntryDraftState();
          applyResponse(sync);
          userModalDismissedPickRef.current = null;
          setError("");
        } catch {
          setError(msg);
        }
      } else {
        setError(msg);
      }
    } finally {
      simLock.current = false;
      setLoadingOp(null);
    }
  }, [applyResponse]);

  const simToUser = useCallback(async () => {
    if (simLock.current || draftDone) return;
    if (isUserPick) {
      setError("You are already on the clock.");
      return;
    }
    simLock.current = true;
    setLoadingOp("simUser");
    setError("");
    setBatchMeta(null);
    try {
      const res = await simEntryDraftToUserPick();
      applyResponse(res, { batch: true });
      const simulated = safeArray(res?.simulated_picks);
      openUserAfterRevealRef.current = Boolean(res?.draft?.is_user_pick);
      if (simulated.length) {
        setCpuReveal(simulated[simulated.length - 1]);
      } else if (openUserAfterRevealRef.current) {
        openUserAfterRevealRef.current = false;
        userModalDismissedPickRef.current = null;
      } else {
        setError("Already at your pick or draft is complete.");
      }
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || "Sim failed";
      const msg = String(Array.isArray(detail) ? detail.join(" ") : detail);
      if (/user.*on the clock|make a selection/i.test(msg)) {
        try {
          const sync = await getEntryDraftState();
          applyResponse(sync);
          userModalDismissedPickRef.current = null;
          setError("");
        } catch {
          setError(msg);
        }
      } else {
        setError(msg);
      }
    } finally {
      simLock.current = false;
      setLoadingOp(null);
    }
  }, [applyResponse, draftDone, isUserPick]);

  const executeSimFullDraft = useCallback(async () => {
    if (simLock.current || draftDone) return;
    simLock.current = true;
    setLoadingOp("complete");
    setError("");
    try {
      const res = await completeEntryDraft();
      applyResponse(res, { batch: true });
      setStage("recap");
    } catch (e) {
      setError(e?.message || "Complete draft failed");
    } finally {
      setLoadingOp(null);
      simLock.current = false;
    }
  }, [applyResponse, draftDone]);

  const requestSimFullDraft = useCallback(() => {
    if (simLock.current || draftDone) return;
    setConfirmDialog({
      title: "Complete draft",
      message: "Complete the entire remaining draft? This may skip your picks.",
      confirmLabel: "Complete draft",
      danger: true,
      onConfirm: () => {
        setConfirmDialog(null);
        executeSimFullDraft();
      },
    });
  }, [draftDone, executeSimFullDraft]);

  const handleUserDraft = useCallback(async (prospect) => {
    if (!prospect) {
      setError("Select a prospect to draft.");
      return;
    }
    if (draftDone) {
      setError("Draft is already complete.");
      return;
    }
    if (!isUserPick) {
      setError("It is not your pick on the clock.");
      return;
    }
    if (simLock.current) return;
    simLock.current = true;
    const pid = getId(prospect);
    setLoadingOp("submit");
    setError("");
    try {
      const res = await submitDraftPick({
        player_id: pid,
        pick_round: currentPick?.round || currentRound,
        pick_overall: currentPick?.overall_pick || draft.overall_pick,
        request_id: `${draft?.draft_id || "draft"}:${currentPick?.pick_id || currentPick?.overall_pick || draft.overall_pick}:${pid}`,
      });
      applyResponse(res);
      setUserModalOpen(false);
      setReportOpen(false);
      userModalDismissedPickRef.current = null;
      setSelectedAvailable(null);
    } catch (e) {
      setError(e?.message || "Pick failed");
    } finally {
      simLock.current = false;
      setLoadingOp(null);
    }
  }, [applyResponse, currentPick, currentRound, draft.overall_pick, draft?.draft_id, draftDone, isUserPick]);

  const simCurrentPick = useCallback(async () => {
    if (draftDone || loading || simLock.current) return;
    if (isUserPick) {
      const best = available[0];
      if (!best) {
        setError("No available prospect to auto-pick.");
        return;
      }
      setConfirmDialog({
        title: "Auto-select",
        message: "Auto-select the top available prospect for your pick?",
        confirmLabel: `Draft ${getPlayerName(best)}`,
        onConfirm: () => {
          setConfirmDialog(null);
          handleUserDraft(best);
        },
      });
      return;
    }
    await runCpuPick();
  }, [draftDone, loading, isUserPick, available, handleUserDraft, runCpuPick]);

  useEffect(() => {
    if (!isUserPick) userModalDismissedPickRef.current = null;
  }, [isUserPick, currentPick?.overall_pick, draft.overall_pick]);

  useEffect(() => {
    if (!isUserPick || draftDone || tradePanelOpen || cpuReveal) return;
    const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
    if (!pickKey) return;
    if (userModalDismissedPickRef.current === pickKey) return;
    setUserModalOpen(true);
  }, [isUserPick, draftDone, tradePanelOpen, cpuReveal, currentPick?.overall_pick, draft.overall_pick]);

  useEffect(() => {
    const prev = lastRoundRef.current;
    if (currentRound > prev && roundRecaps[String(prev)]) setRoundRecapView(roundRecaps[String(prev)]);
    lastRoundRef.current = currentRound;
  }, [currentRound, roundRecaps]);

  useEffect(() => {
    if (draftDone && stage === "live") setStage("recap");
  }, [draftDone, stage]);

  const onCompare = useCallback((p) => {
    const id = getId(p);
    setCompareIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-3)));
  }, []);

  const toggleShortlist = useCallback((p) => {
    const id = getId(p);
    setShortlistIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  const togglePin = useCallback((p) => {
    const id = getId(p);
    setPinnedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-12)));
  }, []);

  const clockTimer = draft.clock_seconds ?? draft.pick_clock ?? currentPick?.clock_seconds ?? currentPick?.time_remaining;

  const centerProspect = selectedCompletedPick || stageProspect;
  const viewingCompletedPick = Boolean(selectedCompletedPick);

  const openDraftModal = useCallback(() => {
    userModalDismissedPickRef.current = null;
    setUserModalOpen(true);
  }, []);

  const promptDraftProspect = useCallback((prospect) => {
    if (!prospect || draftDone || !isUserPick || simLock.current) return;
    const name = getPlayerName(prospect);
    const pickLabel = formatPick(currentPick?.overall_pick || draft.overall_pick);
    setConfirmDialog({
      title: "Submit pick",
      message: `Draft ${name} with ${pickLabel}?`,
      confirmLabel: `Draft ${name}`,
      onConfirm: () => {
        setConfirmDialog(null);
        handleUserDraft(prospect);
      },
    });
  }, [
    draftDone,
    isUserPick,
    currentPick?.overall_pick,
    draft.overall_pick,
    handleUserDraft,
  ]);

  return (
    <section className={`${PREFIX}-root`}>
      <style>{SHEET}</style>

      <CommandBar
        franchiseState={franchiseState}
        draft={draft}
        currentRound={currentRound}
        overallNow={overallNow}
        completedCount={completed.length}
        totalPicks={totalPicks}
        draftDone={draftDone}
        isUserPick={isUserPick}
        onBack={onBack}
      />

      {stage === "intro" ? (
        <main className={`${PREFIX}-intro`}>
          <div className={`${PREFIX}-seal`}>{draftYearLabel(draft) || "NHL"}</div>
          <h1>Entry Draft</h1>
          {(draft.location || draft.class_strength) ? (
            <p>{[draft.location, draft.class_strength].filter(Boolean).join(" · ")}</p>
          ) : null}
          {orderNote ? <p className={`${PREFIX}-muted`}>{orderNote}</p> : null}
          {storylines.length ? (
            <div className={`${PREFIX}-intro-lines`}>
              {storylines.map((s, i) => <span key={i}>{s}</span>)}
            </div>
          ) : null}
          {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
          <button type="button" className={`nhlcal-advance-button`} disabled={loading} onClick={handleStart}>
            <Icon name="next" size={13} />
            {loadingOp === "start" ? LOADING_COPY.start : draft.draft_started ? "Enter draft floor" : "Open the floor"}
          </button>
        </main>
      ) : null}

      {stage === "live" ? (
        <>
          <OnDeckRail upcoming={upcomingPicks} userTeamId={uid} />

          {batchMeta ? (
            <div className={`${PREFIX}-batch-banner`}>
              <span>Simulated <strong>{batchMeta.picks_made ?? 0}</strong> picks</span>
              {batchMeta.biggest_steal?.prospect_name ? (
                <span>Steal: <strong>{batchMeta.biggest_steal.prospect_name}</strong></span>
              ) : null}
              {batchMeta.biggest_reach?.prospect_name ? (
                <span>Reach: <strong>{batchMeta.biggest_reach.prospect_name}</strong></span>
              ) : null}
              <button type="button" className={`${PREFIX}-link`} onClick={() => setBatchMeta(null)}>Dismiss</button>
            </div>
          ) : null}

          <div className={`${PREFIX}-floor`}>
            <LeftPane
              completed={completed}
              selectedPick={selectedCompletedPick}
              userTeamId={uid}
              tweets={draftSocialTweets}
              feedEnabled={completed.length > 0}
              onSelectPick={(p) => {
                const enriched = enrichPickFromBoard(p, draft.public_draft_board || draft.draft_class_rankings?.entries);
                setSelectedCompletedPick(enriched);
                setSelectedAvailable(null);
              }}
            />

            <section className={`${PREFIX}-stage`}>
              {loading && !cpuReveal ? (
                <div className={`${PREFIX}-overlay-msg`}>{LOADING_COPY[loadingOp] || "Working…"}</div>
              ) : null}
              {cpuReveal ? (
                <CpuReveal
                  pick={cpuReveal}
                  fast={simMode === "fast"}
                  onDone={() => {
                    setCpuReveal(null);
                    simLock.current = false;
                    if (openUserAfterRevealRef.current) {
                      openUserAfterRevealRef.current = false;
                      userModalDismissedPickRef.current = null;
                      setUserModalOpen(true);
                    }
                  }}
                />
              ) : null}
              {roundRecapView ? (
                <RoundRecapPanel recap={roundRecapView} onContinue={() => setRoundRecapView(null)} />
              ) : null}

              {!draftDone ? (
                <>
                  <div className={`${PREFIX}-clockband${isUserPick ? " is-user" : ""}`}>
                    <div className={`${PREFIX}-onclock-num ${PREFIX}-tabular`}>
                      <small>Pick</small>
                      {overallNow || "—"}
                    </div>
                    <TeamLogo teamId={draft.current_team_id} teamName={draft.current_team_name} size="lg" />
                    <div className={`${PREFIX}-onclock-team`}>
                      <span className={`${PREFIX}-onclock-state${isUserPick ? " is-user" : ""}`}>
                        {isUserPick ? "Your pick" : "CPU selecting"}
                      </span>
                      <h2>{draft.current_team_name || "Team"}</h2>
                      <div className={`${PREFIX}-onclock-meta`}>
                        {draft.is_traded_pick && (draft.via_team_name || draft.via_team_id) ? (
                          <Chip>{`via ${teamAbbrev(draft.via_team_id, draft.via_team_name)}`}</Chip>
                        ) : null}
                      </div>
                    </div>
                    {clockTimer != null && Number.isFinite(Number(clockTimer)) ? (
                      <span className={`${PREFIX}-clock ${PREFIX}-tabular`}>
                        {Math.round(Number(clockTimer))}s
                      </span>
                    ) : <span />}
                  </div>

                  {viewingCompletedPick ? (
                    <div className={`${PREFIX}-drafted-banner`}>
                      <span>
                        Drafted <strong>#{selectedCompletedPick.overall_pick}</strong>
                        {" · "}
                        {selectedCompletedPick.prospect_name || getPlayerName(selectedCompletedPick)}
                        {" · "}
                        {selectedCompletedPick.team_name}
                      </span>
                      <button type="button" className={`${PREFIX}-link`} onClick={() => setSelectedCompletedPick(null)}>
                        Back to board
                      </button>
                    </div>
                  ) : null}

                  <div className={`${PREFIX}-scroll`}>
                    <StageDossier
                      prospect={centerProspect}
                      currentPickOverall={viewingCompletedPick ? selectedCompletedPick?.overall_pick : overallNow}
                      shortlisted={!viewingCompletedPick && stageProspect ? shortlistIds.includes(getId(stageProspect)) : false}
                      pinned={!viewingCompletedPick && stageProspect ? pinnedIds.includes(getId(stageProspect)) : false}
                      onShortlist={viewingCompletedPick ? undefined : toggleShortlist}
                      onPin={viewingCompletedPick ? undefined : togglePin}
                      onFullReport={viewingCompletedPick ? undefined : (p) => { setSelectedAvailable(p); setReportOpen(true); }}
                    />
                  </div>

                  {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
                </>
              ) : (
                <div className={`${PREFIX}-reveal`}>
                  <p>Draft complete</p>
                  <h1>{completed.length}</h1>
                  <p className={`${PREFIX}-muted`}>selections made</p>
                </div>
              )}
            </section>

            <BoardPane
              available={available}
              selectedId={selectedAvailable ? getId(selectedAvailable) : null}
              isUserPick={isUserPick}
              onDraftProspect={promptDraftProspect}
              onSelectProspect={(p) => {
                setSelectedAvailable(p);
                setSelectedCompletedPick(null);
              }}
            />
          </div>

          <DraftActionDock
            isUserPick={isUserPick}
            draftDone={draftDone}
            loading={loading}
            loadingOp={loadingOp}
            stageProspect={stageProspect}
            available={available}
            tradeOffers={tradeOffers}
            onSimPick={simCurrentPick}
            onSimToUser={simToUser}
            onSimFullDraft={requestSimFullDraft}
            onMakeSelection={openDraftModal}
            onDraftProspect={promptDraftProspect}
            onTradeDown={() => setTradePanelOpen(true)}
          />
        </>
      ) : null}

      {stage === "recap" ? (
        <>
          <RecapShow draft={draft} completed={completed} userPicks={userPicks} />
          <div className={`${PREFIX}-dock ${PREFIX}-dock--root`}>
            <div className={`${PREFIX}-dock-actions`}>
              <button type="button" className={`nhlcal-advance-button`} onClick={onContinue}>
                Continue offseason
              </button>
            </div>
          </div>
        </>
      ) : null}

      {reportOpen && selectedAvailable ? (
        <FullReportSheet
          prospect={selectedAvailable}
          isUserPick={isUserPick}
          loading={loading}
          onDraft={handleUserDraft}
          onClose={() => setReportOpen(false)}
        />
      ) : null}

      {tradePanelOpen ? (
        <TradePanel
          draft={draft}
          accepting={tradeAccepting}
          onClose={() => setTradePanelOpen(false)}
          onAccept={acceptTradeOffer}
        />
      ) : null}

      <ConfirmDialog
        open={Boolean(confirmDialog)}
        title={confirmDialog?.title || ""}
        message={confirmDialog?.message || ""}
        confirmLabel={confirmDialog?.confirmLabel || "Confirm"}
        danger={Boolean(confirmDialog?.danger)}
        loading={loading}
        onConfirm={() => confirmDialog?.onConfirm?.()}
        onCancel={() => setConfirmDialog(null)}
      />

      <SelectionModal
        open={userModalOpen && isUserPick && !draftDone && !tradePanelOpen}
        prospects={available}
        compareIds={compareIds}
        currentPick={currentPick}
        draft={draft}
        loading={loading}
        focusProspect={selectedAvailable}
        onCompare={onCompare}
        onDraft={handleUserDraft}
        onTradeDown={() => setTradePanelOpen(true)}
        onClose={() => {
          const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
          if (pickKey) userModalDismissedPickRef.current = pickKey;
          setUserModalOpen(false);
        }}
      />
    </section>
  );
}