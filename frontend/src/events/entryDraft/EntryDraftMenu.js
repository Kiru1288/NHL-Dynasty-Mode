import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useGameUI } from "../../game/GameUIContext";
import {
  acceptEntryDraftTrade,
  completeEntryDraft,
  getDraftProspectProfile,
  getEntryDraftState,
  simEntryDraftToUserPick,
  startEntryDraft,
  submitCpuDraftPick,
  submitDraftPick,
} from "../../services/franchiseService";
import { getTeamAbbreviation, getTeamLogoSrc } from "../../utils/teamLogos";
import { flagApiUrl, resolveCountryCode } from "../../utils/countryFlags";
import { buildDraftFanTweets, buildDraftPickReactionTweet } from "../awardsNight/awardHelpers";
import { resolveWjcDossierBlock } from "../../screens/prospectDossierHelpers";
import { buildProspectDossierModel } from "../../components/dossier/prospectDossierModel";
import { ProjectionRange, RankChange, StockTrajectory } from "../../components/franchise/commandVisuals";
import { DossierCommandSheet, DossierLayoutStyles } from "../../components/dossier/ProspectDossierView";
import {
  formatPick,
  getPlayerName,
  getPlayerPosition,
  pickFranchiseData,
  safeArray,
} from "../shared/eventHelpers";
import { uiPortalTarget } from "../../utils/fluidUiScale";
import {
  CertaintyChip,
  ProspectClassContext,
  ProspectVisualStyles,
  certaintyTier,
  prospectCertainty,
  useProspectClass,
} from "../../components/dossier/ProspectVisuals";

const PREFIX = "edraft";
const ROUNDS = 7;
const USER_PICK_CLOCK_SECONDS = 300;
const CPU_PICK_CLOCK_SECONDS = 18;
const PICK_TIMEOUT_BONUS_SECONDS = 120;
const PICK_TIMEOUTS_PER_PICK = 1;

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

function findProspectRow(pick, pool) {
  const pid = String(pick?.prospect_id || pick?.key || pick?.selected_prospect_id || "");
  const name = String(pick?.prospect_name || pick?.name || "").trim().toLowerCase();
  return safeArray(pool).find((e) => {
    const id = String(e?.key || e?.prospect_id || e?.id || e?.playerId || "");
    if (pid && id && pid === id) return true;
    if (name && String(e?.name || e?.prospect_name || "").trim().toLowerCase() === name) return true;
    return false;
  }) || null;
}

function enrichPickFromBoard(pick, draftOrEntries) {
  if (!pick) return null;
  const isDraft = Boolean(draftOrEntries) && !Array.isArray(draftOrEntries);
  const draft = isDraft ? draftOrEntries : null;
  const fromAvail = draft ? findProspectRow(pick, draft.available_prospects) : null;
  const fromBoard = findProspectRow(
    pick,
    isDraft
      ? (draft.public_draft_board || draft.draft_class_rankings?.entries || [])
      : draftOrEntries
  ) || (draft ? findProspectRow(pick, draft.draft_class_rankings?.entries) : null);

  const juniorTeam =
    pick.junior_team ||
    fromAvail?.junior_team ||
    fromAvail?.dossier?.team ||
    fromAvail?.dossier?.club ||
    fromBoard?.team_name ||
    fromBoard?.team ||
    fromBoard?.dossier?.team;
  const juniorLeague =
    pick.junior_league ||
    fromAvail?.league ||
    fromBoard?.league ||
    fromAvail?.dossier?.league ||
    fromBoard?.dossier?.league ||
    pick.league;
  const ledger =
    (pick.actual_stats && typeof pick.actual_stats === "object" ? pick.actual_stats : null) ||
    (fromAvail?.actual_stats && typeof fromAvail.actual_stats === "object" ? fromAvail.actual_stats : null) ||
    (fromBoard?.actual_stats && typeof fromBoard.actual_stats === "object" ? fromBoard.actual_stats : null);
  const chapters = pick.chapter_profile || fromAvail?.chapter_profile || fromBoard?.chapter_profile;
  const dossier = {
    ...(fromBoard?.dossier || {}),
    ...(fromAvail?.dossier || {}),
    ...(pick.dossier || {}),
  };

  const merged = {
    ...(fromBoard || {}),
    ...(fromAvail || {}),
    ...pick,
    name: pick.name || pick.prospect_name || fromAvail?.name || fromBoard?.name,
    prospect_name: pick.prospect_name || pick.name || fromAvail?.name || fromBoard?.name,
    junior_team: juniorTeam,
    junior_league: juniorLeague,
    club: juniorTeam,
    nhl_team_name: pick.overall_pick != null ? pick.team_name : pick.nhl_team_name,
  };
  if (Object.keys(dossier).length) merged.dossier = dossier;
  if (chapters) merged.chapter_profile = chapters;
  if (ledger) {
    merged.actual_stats = ledger;
    for (const key of ["gp", "games_played", "goals", "assists", "points", "ppg", "wins", "save_pct", "gaa", "shutouts"]) {
      if (ledger[key] != null && ledger[key] !== "") merged[key] = ledger[key];
    }
  } else {
    for (const key of ["gp", "games_played", "goals", "assists", "points", "ppg"]) {
      const boardVal = fromAvail?.[key] ?? fromBoard?.[key];
      if ((merged[key] == null || Number(merged[key]) === 0) && boardVal != null && Number(boardVal) !== 0) {
        merged[key] = boardVal;
      }
    }
  }
  return merged;
}

function takeFilled(...vals) {
  for (const v of vals) {
    if (v == null || v === "") continue;
    return v;
  }
  return null;
}

function applyProspectProfile(pick, profile) {
  if (!pick) return null;
  if (!profile || typeof profile !== "object") return pick;
  const stats = profile.stats && typeof profile.stats === "object" ? profile.stats : {};
  const chapters = profile.chapter_profile || pick.chapter_profile;
  const chapterMap = (chapters && typeof chapters === "object" && chapters.chapters) || {};
  const takeStat = (...vals) => {
    const present = vals.filter((v) => v != null && v !== "");
    const nonzero = present.find((v) => Number(v) !== 0);
    return nonzero != null ? nonzero : (present[0] ?? null);
  };
  return {
    ...pick,
    dossier: { ...(pick.dossier || {}), ...profile },
    name: takeFilled(pick.name, pick.prospect_name, profile.name, profile.fullName),
    prospect_name: takeFilled(pick.prospect_name, pick.name, profile.name, profile.fullName),
    first_name: takeFilled(pick.first_name, profile.firstName, profile.first_name),
    last_name: takeFilled(pick.last_name, profile.lastName, profile.last_name),
    age: takeFilled(pick.age, profile.age),
    height: takeFilled(pick.height, profile.heightDisplay, profile.height),
    weight: takeFilled(pick.weight, profile.weight),
    handedness: takeFilled(pick.handedness, profile.handedness),
    shoots: takeFilled(pick.shoots, pick.handedness, profile.handedness),
    nationality: takeFilled(pick.nationality, profile.nationality),
    country_code: takeFilled(pick.country_code, profile.country_code, profile.countryCode),
    junior_team: takeFilled(pick.junior_team, profile.team, profile.club, profile.currentClub),
    junior_league: takeFilled(pick.junior_league, profile.league, profile.league_display, profile.league_parent),
    club: takeFilled(pick.club, pick.junior_team, profile.club, profile.team),
    chapter_profile: chapters,
    offence: takeFilled(pick.offence, chapterMap.offence),
    defence: takeFilled(pick.defence, chapterMap.defence, chapterMap.defense),
    character: takeFilled(pick.character, pick.character_score, chapterMap.character),
    mental: takeFilled(pick.mental, chapterMap.mental),
    transition: takeFilled(pick.transition, chapterMap.transition),
    physical: takeFilled(pick.physical, chapterMap.physical),
    glove: takeFilled(pick.glove, chapterMap.glove),
    blocker: takeFilled(pick.blocker, chapterMap.blocker),
    stick: takeFilled(pick.stick, chapterMap.stick),
    actual_stats: (pick.actual_stats && Number(pick.actual_stats.gp || pick.actual_stats.games) > 0)
      ? pick.actual_stats
      : {
          gp: stats.games ?? stats.gp,
          games: stats.games ?? stats.gp,
          games_played: stats.games ?? stats.gp,
          goals: stats.goals,
          assists: stats.assists,
          points: stats.points,
          ppg: stats.ppg,
        },
    gp: takeStat(pick.gp, stats.games, stats.gp),
    goals: takeStat(pick.goals, stats.goals),
    assists: takeStat(pick.assists, stats.assists),
    points: takeStat(pick.points, stats.points),
    ppg: takeFilled(pick.ppg, stats.ppg),
    analytics: pick.analytics || profile.analytics || stats.analytics,
    wjc_stats: pick.wjc_stats || profile.wjcStats || profile.wjcPerformance,
  };
}

function useProspectFile(pick, revision) {
  const pid = String(pick?.prospect_id || pick?.key || pick?.selected_prospect_id || pick?.playerId || "");
  const [profile, setProfile] = useState(null);
  useEffect(() => {
    if (!pid) {
      setProfile(null);
      return undefined;
    }
    let live = true;
    getDraftProspectProfile(pid, { prospectRevision: revision })
      .then((loaded) => {
        if (live) setProfile(loaded);
      })
      .catch(() => {
        if (live) setProfile(null);
      });
    return () => {
      live = false;
    };
  }, [pid, revision]);
  return applyProspectProfile(pick, profile);
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

function getBoardProspectSubline(p) {
  if (!p) return "";
  const club = p.junior_team || p.team || p.club || p.dossier?.team || p.dossier?.club;
  const league = p.league || p.junior_league || p.dossier?.league;
  const trait = getDefiningTrait(p);
  const nat = p.nationality || p.country || p.dossier?.nationality;
  const parts = [club, league].filter(Boolean);
  if (!parts.length && nat) parts.push(String(nat));
  const base = parts.join(" · ");
  if (trait && base) return `${base} · ${trait}`;
  if (trait) return trait;
  return base;
}

function getBoardStockClass(p) {
  const label = normalizeStockLabel(p);
  if (label === "Rising") return "up";
  if (label === "Falling") return "down";
  return "";
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


function isPhilosophyNeedLabel(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("upside") || s.includes("philosophy") || s.includes("bpa")) return true;
  if (s.includes("high") && s.includes("swing")) return true;
  return false;
}


function getComparable(p) {
  const v = p?.comparable_player || p?.player_comparable || p?.style_comparable;
  return v ? String(v) : null;
}


function getBackendWhyWorks(p) {
  return p?.why_pick_makes_sense || p?.why_this_pick_makes_sense || null;
}

function getBackendWhyFails(p) {
  return p?.why_pick_could_fail || p?.bust_reason || null;
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
  if (p.ability_revealed && p.drafted_overall != null) return `${roundInt(p.drafted_overall)}`;
  const scoutPct = num(p.scouted_percentage ?? (p.dossier?.dedicatedScoutFile ? 72 : null));
  const revealed = Boolean(p.ovr_revealed) || (scoutPct != null && scoutPct >= 72);
  if (revealed) {
    const exact = roundInt(p.drafted_overall ?? p.true_ovr ?? p.overall ?? p.current_ovr_estimate);
    if (exact != null) return `${exact}`;
  }
  const lo = roundInt(
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[0] : null) ??
    p.dossier?.overallRangeLow ??
    p.dossier?.now_range?.low ??
    p.public_ovr_low
  );
  const hi = roundInt(
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[1] : null) ??
    p.dossier?.overallRangeHigh ??
    p.dossier?.now_range?.high ??
    p.public_ovr_high
  );
  if (lo != null || hi != null) return formatOvrRange(lo, hi);
  const est = roundInt(p.scouted_overall_estimate ?? p.current_ovr_estimate);
  return est != null ? `${est}` : null;
}

function resolvePotentialOvrLabel(p) {
  if (!p) return null;
  if (p.ceiling_hidden || p.dossier?.ceilingHidden || p.dossier?.peak_range?.hidden) return null;
  if (p.ability_revealed && p.drafted_potential != null) return `${roundInt(p.drafted_potential)}`;
  const lo = roundInt(
    p.dossier?.scoutedPotentialLow ??
    p.dossier?.peak_range?.low ??
    p.potential_range?.low ??
    (Array.isArray(p.scouted_potential_range) ? p.scouted_potential_range[0] : null)
  );
  let hi = roundInt(
    p.dossier?.scoutedPotentialHigh ??
    p.dossier?.peak_range?.high ??
    p.potential_range?.high ??
    (Array.isArray(p.scouted_potential_range) ? p.scouted_potential_range[1] : null)
  );
  const rank = getRank(p);
  const maxSpan = rank != null && rank <= 10 ? 6 : rank != null && rank <= 32 ? 5 : rank != null && rank <= 64 ? 4 : 3;
  // Only a real scouting file earns a tight ceiling range; a thin file keeps its
  // full spread (clamping it made every prospect look fully scouted).
  const filePct = num(p.scouted_percentage) ?? 0;
  if (filePct >= 72 && hi != null && lo != null && hi - lo > maxSpan) hi = lo + maxSpan;
  const label = formatOvrRange(lo, hi);
  if (label) return label;
  return getScoutedPotentialLabel(p);
}

/** First `count` sentences of a scouting paragraph (the dossier was a wall of text). */
/** Staff certainty on this prospect's OVR and potential (see ProspectVisuals). */
function getProspectCertainty(p) {
  if (!p) return { ovr: 0, pot: 0 };
  return prospectCertainty({
    scoutPct: num(p.scouted_percentage ?? p.dossier?.scouted_percentage),
    publicConf: getConfidence(p),
    revealed: Boolean(p.ovr_revealed),
    abilityRevealed: Boolean(p.ability_revealed && p.drafted_overall != null),
    ceilingHidden: Boolean(p.ceiling_hidden || p.dossier?.ceilingHidden || p.dossier?.peak_range?.hidden),
    rank: getRank(p),
  });
}

function resolveProductionStats(prospect, dossierStats = {}) {
  const src = dossierStats && typeof dossierStats === "object" ? dossierStats : {};
  const actual = prospect?.actual_stats && typeof prospect.actual_stats === "object" ? prospect.actual_stats : {};
  const season = prospect?.season_stats && typeof prospect.season_stats === "object" ? prospect.season_stats : {};
  const projected = prospect?.projected_stats && typeof prospect.projected_stats === "object" ? prospect.projected_stats : {};
  const pickStat = (...vals) => {
    const present = [];
    for (const v of vals) {
      if (v == null || v === "") continue;
      const n = Number(v);
      if (!Number.isFinite(n)) continue;
      present.push(n);
    }
    if (!present.length) return null;
    const nonzero = present.find((n) => n !== 0);
    return roundInt(nonzero != null ? nonzero : present[0]);
  };
  const games = pickStat(
    actual.gp,
    actual.games,
    actual.games_played,
    src.games,
    src.gp,
    season.gp,
    season.games_played,
    projected.gp,
    projected.projected_gp,
    prospect?.gp,
    prospect?.games_played,
  );
  const goals = pickStat(actual.goals, src.goals, season.goals, projected.goals, projected.projected_goals, prospect?.goals);
  const assists = pickStat(actual.assists, src.assists, season.assists, projected.assists, projected.projected_assists, prospect?.assists);
  const points = pickStat(actual.points, src.points, season.points, projected.points, projected.projected_points, prospect?.points) ?? (
    goals != null && assists != null ? goals + assists : null
  );
  const ppgRaw = actual.ppg ?? actual.points_per_game ?? src.ppg ?? season.ppg ?? projected.ppg ?? projected.projected_ppg ?? prospect?.ppg ?? prospect?.points_per_game;
  const ppg = ppgRaw != null
    ? Number(ppgRaw)
    : (games > 0 && points != null ? points / games : null);
  return {
    games,
    goals,
    assists,
    points,
    ppg: Number.isFinite(ppg) ? ppg : null,
    primary: pickStat(actual.primary_points, src.primary_points, prospect?.primary_points),
    savePct: actual.save_pct ?? src.save_pct ?? season.save_pct ?? prospect?.save_pct,
    gaa: actual.gaa ?? src.gaa ?? season.gaa ?? prospect?.gaa,
    shutouts: pickStat(actual.shutouts, src.shutouts, prospect?.shutouts),
    wins: pickStat(actual.wins, src.wins, prospect?.wins),
  };
}

function formatStatCell(label, value) {
  if (value == null || value === "") return null;
  if (label === "SV%") {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const frac = n > 1 ? n / 100 : n;
    return frac.toFixed(3).replace(/^0/, "");
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

function buildTeamHaulColumns(draftOrder, completed) {
  const picks = safeArray(completed);
  const byTeam = new Map();
  for (const p of picks) {
    const id = String(p.team_id || p.team || "");
    if (!id) continue;
    if (!byTeam.has(id)) byTeam.set(id, []);
    byTeam.get(id).push(p);
  }
  for (const list of byTeam.values()) {
    list.sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));
  }
  const seen = new Set();
  const cols = [];
  for (const slot of safeArray(draftOrder)) {
    const id = String(slot.team_id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    cols.push({
      team_id: id,
      team_name: slot.team_name,
      picks: byTeam.get(id) || [],
    });
  }
  for (const [id, list] of byTeam) {
    if (seen.has(id)) continue;
    cols.push({
      team_id: id,
      team_name: list[0]?.team_name,
      picks: list,
    });
  }
  return cols;
}

function scoutedOvrLabel(p) {
  if (!p) return null;
  const lo = roundInt(
    p.scouted_overall_estimate ??
    p.current_ovr_estimate ??
    p.floor_grade ??
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[0] : null)
  );
  const hi = roundInt(
    p.dossier?.overallRangeHigh ??
    (Array.isArray(p.current_ovr_range) ? p.current_ovr_range[1] : null)
  );
  if (lo != null && hi != null && lo !== hi) return `${lo}–${hi}`;
  if (lo != null) return `${lo}`;
  if (hi != null) return `${hi}`;
  return null;
}

function revealedOvrValue(p) {
  return roundInt(p?.drafted_overall ?? p?.true_ovr ?? p?.overall);
}

function isGoalie(p) {
  return String(getPlayerPosition(p) || "").toUpperCase() === "G";
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

function recapVerdict(pick) {
  const delta = pubDelta(pick);
  if (delta == null) return null;
  const label = delta >= 10 ? "VALUE" : delta <= -10 ? "REACH" : "EXPECTED";
  const tone = label === "VALUE" ? "value" : label === "REACH" ? "reach" : "expected";
  const shown = delta > 0 ? `+${delta}` : delta < 0 ? `−${Math.abs(delta)}` : "0";
  return { label, tone, text: `${label} ${shown}` };
}

function recapConfidence(p) {
  const point = getConfidence(p);
  if (point != null) return { text: `${Math.round(point)}%`, muted: point < 45 };
  const range = getConfidenceRange(p);
  if (!range) return null;
  return { text: `${Math.round(range[0])}–${Math.round(range[1])}%`, muted: true };
}

function projectionLeagueLabel(p) {
  const raw = p?.development_path || p?.rights_card?.development_path || p?.dossier?.development_path;
  if (raw == null || String(raw).trim() === "") return null;
  return String(raw).trim();
}

function isStealOrReachLine(text) {
  return /\b(steal|steals|reach|reaches)\b/i.test(String(text || ""));
}

const SCOUTED_NOTE = "Public rank is the consensus board at the moment of selection. Scouted OVR is the fogged pre-draft band; revealed OVR is the true ability unlocked once the player is off the board.";

function potVal(p) {
  const v = p?.drafted_potential ?? p?.ceiling_grade ?? p?.potential_score;
  if (v != null && !Number.isNaN(Number(v))) return Math.round(Number(v));
  return p?.potential_grade || "—";
}

/* ------------------------------------------------------------------ */
/* Stylesheet                                                          */
/* ------------------------------------------------------------------ */



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

function PickBadge({ tag, tone, className = "" }) {
  if (!tag) return null;
  return <span className={`${PREFIX}-text-pop tone-${tone || "default"}${className ? ` ${className}` : ""}`}>{tag}</span>;
}

function pickOvrPopTone(tone) {
  if (tone === "steal" || tone === "value") return tone === "steal" ? "steal" : "value";
  if (tone === "reach" || tone === "early") return "reach";
  if (tone === "expected" || tone === "bpa") return "expected";
  if (tone === "need") return "blue";
  if (tone === "offboard") return "offboard";
  return "cyan";
}

function LastPickSpotlight({ pick, onOpen }) {
  if (!pick) return null;
  const { tag, tone } = getPickDisplayTag(pick);
  const movement = describePickMovement(pick);
  const ovr = revealedOvrValue(pick) ?? resolveCurrentOvrLabel(pick);
  const pub = getPublicRankAtPick(pick);
  const name = pick.prospect_name || getPlayerName(pick);
  const pos = pick.position || getPlayerPosition(pick);
  const ovrTone = pickOvrPopTone(tone);

  const onKey = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onOpen?.(pick);
    }
  };

  return (
    <div
      className={`${PREFIX}-pick-spotlight${tone && tone !== "default" ? ` tone-${tone}` : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onOpen?.(pick)}
      onKeyDown={onKey}
      aria-label={`View ${name}, pick ${pick.overall_pick}`}
    >
      <div className={`${PREFIX}-pick-spotlight-pick`}>
        <span>Pick</span>
        <b className={`${PREFIX}-tabular`}>{pick.overall_pick ?? "—"}</b>
      </div>
      <div className={`${PREFIX}-pick-spotlight-main`}>
        <p className={`${PREFIX}-pick-spotlight-kicker`}>Last selection</p>
        <div className={`${PREFIX}-pick-spotlight-team`}>
          <TeamLogo teamId={pick.team_id} teamName={pick.team_name} size="md" />
          <h3>{pick.team_name}</h3>
        </div>
        <p className={`${PREFIX}-pick-spotlight-name`}>{name}</p>
        <div className={`${PREFIX}-pick-spotlight-meta`}>
          {pos ? <span>{pos}</span> : null}
          {pick.league ? <span>{pick.league}</span> : null}
          {pub != null ? <span>Board #{pub}</span> : null}
          {movement?.label ? <span>{movement.label}</span> : null}
        </div>
      </div>
      <div className={`${PREFIX}-pick-spotlight-side`}>
        <span className={`${PREFIX}-haul-ovr ${PREFIX}-pick-spotlight-ovr ${PREFIX}-tabular ${PREFIX}-num-pop tone-${ovrTone}`}>
          {ovr || "—"}
        </span>
        {tag ? <PickBadge tag={tag} tone={tone} className={`${PREFIX}-pick-spotlight-tag`} /> : null}
      </div>
    </div>
  );
}

function Chip({ children, variant, icon }) {
  if (children == null || children === "") return null;
  return (
    <span className={`${PREFIX}-chip${variant ? ` is-${variant}` : ""}`}>
      {icon ? <Icon name={icon} size={11} tone={ICON_TONES[icon]} /> : null}
      {children}
    </span>
  );
}

function formatPickClockSeconds(totalSeconds) {
  const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return `${r}s`;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function useDraftPickClock(overallPick, draftDone, clockEnabled, isUserPick, clockPaused, onExpired) {
  const duration = isUserPick ? USER_PICK_CLOCK_SECONDS : CPU_PICK_CLOCK_SECONDS;
  const [secondsLeft, setSecondsLeft] = useState(duration);
  const [timeoutsUsed, setTimeoutsUsed] = useState(0);
  const endsAtRef = useRef(null);
  const expiredRef = useRef(false);
  const pausedRef = useRef(false);
  const onExpiredRef = useRef(onExpired);
  onExpiredRef.current = onExpired;
  pausedRef.current = Boolean(clockPaused);

  useEffect(() => {
    if (draftDone || !clockEnabled || overallPick == null || overallPick === "") {
      setSecondsLeft(duration);
      setTimeoutsUsed(0);
      endsAtRef.current = null;
      expiredRef.current = false;
      return undefined;
    }
    setSecondsLeft(duration);
    setTimeoutsUsed(0);
    expiredRef.current = false;
    endsAtRef.current = Date.now() + duration * 1000;
    const tick = () => {
      if (!endsAtRef.current || pausedRef.current) return;
      const left = Math.max(0, Math.ceil((endsAtRef.current - Date.now()) / 1000));
      setSecondsLeft(left);
      if (left <= 0) {
        const handled = onExpiredRef.current?.();
        if (handled !== false) expiredRef.current = true;
      }
    };
    tick();
    const id = window.setInterval(tick, 250);
    return () => window.clearInterval(id);
  }, [overallPick, draftDone, clockEnabled, isUserPick, duration]);

  const callTimeout = useCallback(() => {
    if (!isUserPick || timeoutsUsed >= PICK_TIMEOUTS_PER_PICK || !endsAtRef.current) return false;
    endsAtRef.current += PICK_TIMEOUT_BONUS_SECONDS * 1000;
    setTimeoutsUsed((n) => n + 1);
    const left = Math.max(0, Math.ceil((endsAtRef.current - Date.now()) / 1000));
    setSecondsLeft(left);
    if (left > 0) expiredRef.current = false;
    return true;
  }, [isUserPick, timeoutsUsed]);

  const canTimeout = isUserPick && clockEnabled && !draftDone && timeoutsUsed < PICK_TIMEOUTS_PER_PICK;

  return { secondsLeft, timeoutsUsed, callTimeout, canTimeout };
}

function PickClockDisplay({
  secondsLeft,
  isUserPick,
  draftDone,
  clockEnabled,
  clockTeamLabel,
  loading,
  canTimeout,
  onTimeout,
}) {
  if (draftDone || !clockEnabled) return null;
  const urgent = isUserPick ? secondsLeft <= 60 : secondsLeft <= 5;
  const label = formatPickClockSeconds(secondsLeft);
  const timerLabel = isUserPick ? "Your pick clock" : `${clockTeamLabel || "Team"} pick clock`;
  return (
    <div className={`${PREFIX}-pick-timer-wrap`}>
      <div
        className={`${PREFIX}-pick-timer${isUserPick ? " is-user" : " is-cpu"}${urgent ? " is-urgent" : ""}`}
        aria-live="polite"
        role="timer"
        aria-label={`${timerLabel} ${label} remaining`}
      >
        <span className={`${PREFIX}-pick-timer-label`}>{timerLabel}</span>
        <span className={`${PREFIX}-pick-timer-time ${PREFIX}-tabular`}>{label}</span>
      </div>
      {isUserPick ? (
        <button
          type="button"
          className={`nhlcal-quick-link ${PREFIX}-pick-timeout`}
          disabled={loading || !canTimeout}
          onClick={onTimeout}
          title={canTimeout ? "Add two minutes (one timeout per pick)" : "Timeout already used this pick"}
        >
          {canTimeout ? "Timeout +2:00" : "Timeout used"}
        </button>
      ) : null}
    </div>
  );
}

function SectionHead({ title, meta, icon, tone }) {
  const metaNode = meta == null || meta === ""
    ? null
    : typeof meta === "string"
      ? <em>{meta}</em>
      : <span className={`${PREFIX}-sec-meta`}>{meta}</span>;
  return (
    <div className={`${PREFIX}-sec-head`}>
      {icon ? <Icon name={icon} size={13} tone={tone || ICON_TONES[icon]} /> : null}
      <h4>{title}</h4>
      {metaNode}
    </div>
  );
}

function factValuePopClass(value, tone) {
  const s = String(value).trim();
  const numeric = typeof value === "number"
    || /^#?\d+([.]\d+)?%?$/.test(s)
    || /^[\d.]+[–-][\d.]+$/.test(s);
  let popTone = "cyan";
  if (tone === "good") popTone = "green";
  else if (tone === "warn") popTone = "orange";
  else if (!numeric) popTone = "default";
  const base = numeric ? `${PREFIX}-num-pop` : `${PREFIX}-text-pop`;
  return `${PREFIX}-fact-value ${base} tone-${popTone}`;
}

function Fact({ label, value, tone }) {
  if (value == null || value === "") return null;
  return (
    <div className={`${PREFIX}-fact${tone ? ` tone-${tone}` : ""}`}>
      <span className={`${PREFIX}-fact-label`}>{label}</span>
      <strong className={factValuePopClass(value, tone)}>{value}</strong>
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
  pickSecondsLeft,
  clockEnabled,
  clockTeamLabel,
  canPickTimeout,
  onPickTimeout,
  clockLoading,
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
        <PickClockDisplay
          secondsLeft={pickSecondsLeft}
          isUserPick={isUserPick}
          draftDone={draftDone}
          clockEnabled={clockEnabled}
          clockTeamLabel={clockTeamLabel}
          loading={clockLoading}
          canTimeout={canPickTimeout}
          onTimeout={onPickTimeout}
        />
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
  onOpenBoard,
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
                {dockProspect ? (
                  <button type="button" className="nhlcal-advance-button-secondary" disabled={loading} onClick={onMakeSelection}>
                    <Icon name="search" size={12} />
                    Select prospect
                  </button>
                ) : null}
                <button type="button" className="nhlcal-advance-button-secondary" disabled={loading} onClick={onTradeDown}>
                  <Icon name="trade" size={12} />
                  Trade down{tradeOffers?.length ? ` (${tradeOffers.length})` : ""}
                </button>
              </>
            ) : (
          <>
            <button type="button" className="nhlcal-advance-button" disabled={loading || draftDone} onClick={onSimPick}>
              <Icon name="sim" size={12} />
              {loadingOp === "cpu" ? LOADING_COPY.cpu : "Sim pick"}
            </button>
            <button type="button" className="nhlcal-advance-button-secondary" disabled={loading || draftDone} onClick={onSimToUser}>
              <Icon name="step" size={12} />
              {loadingOp === "simUser" ? LOADING_COPY.simUser : "Sim to my pick"}
            </button>
            <button type="button" className="nhlcal-advance-button-secondary" disabled={loading || draftDone} onClick={onOpenBoard}>
              <Icon name="search" size={12} />
              Select prospect
            </button>
            <button type="button" className="nhlcal-advance-button-secondary edraft-btn--danger" disabled={loading || draftDone} onClick={onSimFullDraft}>
              <Icon name="round" size={12} tone="danger" />
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

function FloorBuzzFeed({ tweets, completed, userTeamId, onSelectPick }) {
  const groups = useMemo(() => {
    const pickByOverall = new Map();
    safeArray(completed).forEach((pick) => {
      const overall = Number(pick?.overall_pick);
      if (Number.isFinite(overall)) pickByOverall.set(overall, pick);
    });

    const byKey = new Map();
    safeArray(tweets).forEach((tweet) => {
      const overall = Number(tweet?.context?.overallPick);
      const pick = Number.isFinite(overall) ? pickByOverall.get(overall) : null;
      const playerName = pick?.prospect_name || getPlayerName(pick) || tweet?.context?.winnerLabel || "Prospect";
      const teamName = pick?.team_name || tweet?.context?.winnerTeamName || "";
      const key = Number.isFinite(overall) ? `p-${overall}` : `t-${tweet.id}`;
      if (!byKey.has(key)) {
        byKey.set(key, {
          key,
          pick,
          overall: Number.isFinite(overall) ? overall : null,
          playerName,
          teamId: pick?.team_id,
          teamName,
          position: pick?.position || getPlayerPosition(pick),
          tweets: [],
        });
      }
      byKey.get(key).tweets.push(tweet);
    });

    return Array.from(byKey.values()).sort((a, b) => (b.overall || 0) - (a.overall || 0));
  }, [tweets, completed]);

  if (!groups.length) return null;

  return (
    <div className={`${PREFIX}-buzz`}>
      {groups.map((group) => {
        const mine = String(group.teamId || "") === String(userTeamId || "");
        const { tag, tone } = group.pick ? getPickDisplayTag(group.pick) : { tag: null, tone: "default" };
        const teamLabel = teamAbbrev(group.teamId, group.teamName) || group.teamName || "NHL";
        return (
          <article key={group.key} className={`${PREFIX}-buzz-pick${mine ? " is-user" : ""}`}>
            <button
              type="button"
              className={`${PREFIX}-buzz-head`}
              onClick={() => group.pick && onSelectPick?.(group.pick)}
              title={group.pick ? `View ${group.playerName} draft card` : undefined}
            >
              <span className={`${PREFIX}-buzz-num ${PREFIX}-tabular`}>
                {group.overall != null ? `#${group.overall}` : "—"}
              </span>
              <TeamLogo teamId={group.teamId} teamName={group.teamName} size="sm" />
              <span className={`${PREFIX}-buzz-who`}>
                <strong>{group.playerName}</strong>
                <span>{[teamLabel, group.position].filter(Boolean).join(" · ")}</span>
              </span>
              <PickBadge tag={tag} tone={tone} />
            </button>
            {group.tweets.map((tweet) => (
              <ReactionRow key={tweet.id} tweet={tweet} />
            ))}
          </article>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Left pane — draft log + floor buzz                                  */
/* ------------------------------------------------------------------ */

function DraftedOvrChip({ pick }) {
  const ovr = resolveCurrentOvrLabel(pick);
  if (!ovr) return null;
  const pot = roundInt(pick?.drafted_potential ?? pick?.ceiling_grade);
  return (
    <span className={`${PREFIX}-log-ovr`} title="Overall now / projected peak">
      <b className={`${PREFIX}-tabular ${PREFIX}-num-pop tone-cyan`}>{ovr}</b>
      {pot != null ? <i className={`${PREFIX}-tabular ${PREFIX}-num-pop tone-gold`}>{pot}</i> : null}
    </span>
  );
}

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
                    <DraftedOvrChip pick={p} />
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
            <FloorBuzzFeed
              tweets={tweets}
              completed={rows}
              userTeamId={userTeamId}
              onSelectPick={onSelectPick}
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
  hideHead = false,
  rich = false,
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

  const richClass = rich ? " is-rich" : "";
  const posCounts = useMemo(() => {
    const counts = { C: 0, W: 0, D: 0, G: 0 };
    safeArray(available).forEach((p) => {
      const pos = getPlayerPosition(p);
      if (pos === "C") counts.C += 1;
      else if (pos === "G") counts.G += 1;
      else if (pos === "D") counts.D += 1;
      else if (["LW", "RW", "W"].includes(pos)) counts.W += 1;
    });
    return counts;
  }, [available]);

  return (
    <aside className={`${PREFIX}-pane`}>
      {hideHead ? null : (
        <div className={`${PREFIX}-pane-head`}>
          <h3>Best Available</h3>
          <em className={`${PREFIX}-tabular`}>{filtered.length}</em>
        </div>
      )}

      {rich ? (
        <div className={`${PREFIX}-board-summary`} aria-label="Board snapshot">
          <span><strong>{safeArray(available).length}</strong> on public board</span>
          <span><strong>{filtered.length}</strong> in view</span>
          <span>C <strong>{posCounts.C}</strong></span>
          <span>W <strong>{posCounts.W}</strong></span>
          <span>D <strong>{posCounts.D}</strong></span>
          <span>G <strong>{posCounts.G}</strong></span>
        </div>
      ) : null}

      <div className={`${PREFIX}-board-tools${richClass}`}>
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

      <div className={`${PREFIX}-board-cols${isUserPick ? " is-user-pick" : ""}${richClass}`} aria-hidden="true">
        {rich ? (
          <>
            <span>Brd</span>
            <span>Prospect</span>
            <span>Pos</span>
            <span>Age</span>
            <span>You</span>
            <span>OVR · Pot</span>
            <span>Stock</span>
            <span>Sure?</span>
            <span>Risk</span>
            {isUserPick ? <span>Draft</span> : null}
          </>
        ) : (
          <>
            <span>Brd</span><span>Prospect</span><span>Pos</span><span>OVR–Pot</span><span>Sure?</span>
            {isUserPick ? <span>Draft</span> : null}
          </>
        )}
      </div>

      <div
        className={`${PREFIX}-scroll ${PREFIX}-scroll-surface`}
        ref={listRef}
        role="listbox"
        aria-label="Best available prospects"
      >
        {!filtered.length ? (
          <p className={`${PREFIX}-empty`}>No prospects match those filters.</p>
        ) : (
          filtered.map((p) => {
            const pub = getRank(p);
            const ovr = resolveCurrentOvrLabel(p);
            const pot = resolvePotentialOvrLabel(p);
            const cert = getProspectCertainty(p);
            const confTitle = `${certaintyTier(cert.ovr).label}: ${cert.ovr}% sure of OVR, ${cert.pot}% sure of potential`
              + (getConfidenceTitle(p) ? ` · ${getConfidenceTitle(p)}` : "");
            const softConf = cert.ovr < 55;
            const selected = selectedId === getId(p);
            const teamRank = getTeamRank(p);
            const age = num(p?.age ?? p?.dossier?.age);
            const subline = getBoardProspectSubline(p);
            const stockLabel = normalizeStockLabel(p);
            const stockClass = getBoardStockClass(p);
            const risk = getRisk(p);
            const riskHigh = risk && /high|bust|major/i.test(String(risk));
            const nat = p.nationality || p.country || p.dossier?.nationality;
            const natCode = p.country_code || p.nat || nat;
            const fit = getTeamFit(p);
            return (
              <div
                key={getId(p)}
                className={`${PREFIX}-board-row-wrap${selected ? " is-selected" : ""}${richClass}`}
                data-selected={selected ? "true" : undefined}
              >
                <button
                  type="button"
                  role="option"
                  aria-selected={selected}
                  className={`${PREFIX}-board-row${selected ? " is-selected" : ""}${richClass}`}
                  title={[p.stock_reason, confTitle, fit != null ? `Org fit ${Math.round(fit)}` : null].filter(Boolean).join(" · ") || undefined}
                  onClick={() => onSelectProspect?.(p)}
                >
                  <span className={`${PREFIX}-board-rank ${PREFIX}-tabular`}>{pub != null ? pub : "—"}</span>
                  <span className={`${PREFIX}-board-name`}>
                    {rich ? (
                      <span className={`${PREFIX}-board-name-cell`}>
                        <ProspectFlag country={nat} code={natCode} width={22} />
                        <span>
                          <strong>{getPlayerName(p)}</strong>
                          {subline ? <span className={`${PREFIX}-board-sub`}>{subline}</span> : null}
                        </span>
                      </span>
                    ) : (
                      <strong>{getPlayerName(p)}</strong>
                    )}
                  </span>
                  <span className={`${PREFIX}-board-pos`}>{getPlayerPosition(p)}</span>
                  {rich ? (
                    <>
                      <span className={`${PREFIX}-board-age ${PREFIX}-tabular`}>
                        {age != null ? roundInt(age) ?? age : "—"}
                      </span>
                      <span className={`${PREFIX}-board-you ${PREFIX}-tabular`} title="Your board rank">
                        {teamRank != null ? `#${teamRank}` : "—"}
                      </span>
                    </>
                  ) : null}
                  <span
                    className={`${PREFIX}-board-band ${PREFIX}-tabular ${PREFIX}-num-pop tone-cyan${softConf ? " is-soft" : ""}`}
                    title={confTitle}
                  >
                    {[ovr ? (softConf ? `~${ovr}` : ovr) : null, pot ? (cert.pot < 45 ? `~${pot}` : pot) : null].filter(Boolean).join(" · ") || "—"}
                  </span>
                  {rich ? (
                    <span className={`${PREFIX}-board-stock ${PREFIX}-tabular${stockClass ? ` ${stockClass}` : ""}`}>
                      {stockLabel || "—"}
                    </span>
                  ) : null}
                  <span
                    className={`${PREFIX}-board-stock ${PREFIX}-tabular${softConf ? " is-soft" : ""}`}
                    title={confTitle || undefined}
                  >
                    <CertaintyChip certainty={cert.ovr} />
                  </span>
                  {rich ? (
                    <span className={`${PREFIX}-board-risk${riskHigh ? " is-high" : ""}`} title="Bust / risk read">
                      {risk || "—"}
                    </span>
                  ) : null}
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

function TeamHaulBoard({
  draftOrder,
  completed,
  userTeamId,
  clockTeamId,
  selectedPick,
  onSelectPick,
  floor = false,
}) {
  const cols = useMemo(
    () => buildTeamHaulColumns(draftOrder, completed),
    [draftOrder, completed]
  );
  const selectedOverall = selectedPick?.overall_pick;
  const latestOverall = useMemo(() => {
    let max = null;
    safeArray(completed).forEach((p) => {
      const n = Number(p?.overall_pick);
      if (Number.isFinite(n) && (max == null || n > max)) max = n;
    });
    return max;
  }, [completed]);
  return (
    <div className={`${PREFIX}-haul${floor ? ` ${PREFIX}-haul--floor` : ""}`} role="list" aria-label="Drafted players by team">
      {cols.map((col) => {
        const isUser = String(col.team_id) === String(userTeamId);
        const isClock = clockTeamId && String(col.team_id) === String(clockTeamId);
        return (
          <div
            key={col.team_id}
            className={`${PREFIX}-haul-col${isUser ? " is-user" : ""}${isClock ? " is-clock" : ""}`}
            role="listitem"
          >
            <div className={`${PREFIX}-haul-head`}>
              <TeamLogo teamId={col.team_id} teamName={col.team_name} size="sm" />
              <div>
                <strong>{teamAbbrev(col.team_id, col.team_name)}</strong>
                <span>{col.picks.length} selected</span>
              </div>
            </div>
            <div className={`${PREFIX}-haul-list`}>
              {!col.picks.length ? (
                <p className={`${PREFIX}-haul-empty`}>No picks yet</p>
              ) : (
                col.picks.map((p) => {
                  const selected = selectedOverall === p.overall_pick;
                  const isLatest = latestOverall != null && Number(p.overall_pick) === latestOverall;
                  const { tag, tone } = getPickDisplayTag(p);
                  const ovr = revealedOvrValue(p) ?? resolveCurrentOvrLabel(p);
                  const ovrTone = pickOvrPopTone(tone);
                  const toneClass = tone && tone !== "default" ? ` tone-${tone}` : "";
                  return (
                    <button
                      type="button"
                      key={p.overall_pick || getId(p)}
                      className={`${PREFIX}-haul-pick${selected ? " is-selected" : ""}${isLatest ? " is-latest" : ""}${toneClass}`}
                      onClick={() => onSelectPick?.(p)}
                      title={`${p.prospect_name || getPlayerName(p)} · pick ${p.overall_pick}`}
                    >
                      <b className={`${PREFIX}-tabular`}>{p.overall_pick}</b>
                      <span>
                        <strong>{p.prospect_name || getPlayerName(p)}</strong>
                        <em>
                          {[p.position || getPlayerPosition(p), p.round ? `R${p.round}` : null]
                            .filter(Boolean)
                            .join(" · ")}
                        </em>
                        {floor && tag ? (
                          <PickBadge tag={tag} tone={tone} className={`${PREFIX}-haul-pick-tag`} />
                        ) : null}
                      </span>
                      <span className={`${PREFIX}-haul-ovr ${PREFIX}-tabular ${PREFIX}-num-pop tone-${ovrTone}`}>{ovr || "—"}</span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ───────────── War room desk: the centre of the floor ─────────────
   Pinned scouting cards for the best players left, a draft-trend whiteboard and
   room chatter, all derived from data already on the screen (no extra requests). */
const WR_POS_GROUP = (pos) => (pos === "LW" || pos === "RW" || pos === "W" ? "W" : pos);
const WR_POS_NAME = { C: "centres", W: "wingers", D: "defencemen", G: "goalies" };

function warRoomTag(p, overallNow, isBiggestFaller) {
  const stock = normalizeStockLabel(p);
  const risk = String(getRisk(p) || "");
  const age = num(p?.age ?? p?.dossier?.age);
  const rank = getRank(p);
  const fell = rank != null && overallNow ? Number(overallNow) - rank : 0;
  if (isBiggestFaller && fell >= 5) return { label: "Steal", tone: "lime" };
  if (fell >= 4) return { label: `Fell from #${rank}`, tone: "sky" };
  if (stock === "Rising") return { label: "Riser", tone: "orange" };
  if (stock === "Falling") return { label: "Sliding", tone: "coral" };
  if (/high|bust|major/i.test(risk)) return { label: "Raw", tone: "coral" };
  if (/low|safe/i.test(risk) || (age != null && age >= 19)) return { label: "Safe", tone: "sky" };
  return { label: "Watch", tone: "yellow" };
}

function warRoomNote(p) {
  const trait = getDefiningTrait(p);
  if (trait) return trait.length > 34 ? `${trait.slice(0, 32)}…` : trait;
  const league = p?.league || p?.junior_league || p?.dossier?.league;
  return league ? `Tracked in ${league}` : "Scouts split";
}

function WarRoomDesk({ available, completed, draftOrder, userTeamId, overallNow, clockTeamName, onOpenProspect, totalPicks, completedCount }) {
  const top = useMemo(
    () =>
      safeArray(available)
        .slice()
        .sort((a, b) => (getRank(a) ?? 9999) - (getRank(b) ?? 9999))
        .slice(0, 10),
    [available]
  );
  const recent = useMemo(
    () =>
      safeArray(completed)
        .slice()
        .sort((a, b) => Number(b.overall_pick || 0) - Number(a.overall_pick || 0))
        .slice(0, 6),
    [completed]
  );
  const run = useMemo(() => {
    const counts = {};
    recent.forEach((p) => {
      const g = WR_POS_GROUP(p.position || getPlayerPosition(p));
      if (g) counts[g] = (counts[g] || 0) + 1;
    });
    const [pos, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];
    return pos && n >= 3 ? { pos, n } : null;
  }, [recent]);
  const identity = useMemo(() => {
    const pool = safeArray(available)
      .slice()
      .sort((a, b) => (getRank(a) ?? 9999) - (getRank(b) ?? 9999))
      .slice(0, 40);
    if (pool.length < 10) return null;
    const counts = { C: 0, W: 0, D: 0, G: 0 };
    pool.forEach((p) => {
      const g = WR_POS_GROUP(getPlayerPosition(p));
      if (g in counts) counts[g] += 1;
    });
    const share = { C: 0.25, W: 0.36, D: 0.32, G: 0.07 };
    const ratio = Object.keys(counts).map((k) => [k, counts[k] / pool.length / share[k]]);
    ratio.sort((a, b) => b[1] - a[1]);
    return { deep: WR_POS_NAME[ratio[0][0]], thin: WR_POS_NAME[ratio[ratio.length - 1][0]] };
  }, [available]);
  const nextUser = useMemo(() => {
    const now = Number(overallNow || 0);
    const slot = safeArray(draftOrder).find(
      (s) => String(s.team_id || "") === String(userTeamId) && Number(s.overall_pick || s.pick || 0) >= now
    );
    const n = slot ? Number(slot.overall_pick || slot.pick || 0) : null;
    return n ? { pick: n, away: Math.max(0, n - now) } : null;
  }, [draftOrder, userTeamId, overallNow]);
  const chatter = useMemo(() => {
    const out = [];
    recent.forEach((p) => {
      const { tag } = getPickDisplayTag(p);
      const team = teamAbbrev(p.team_id, p.team_name);
      const name = p.prospect_name || getPlayerName(p);
      if (/reach/i.test(String(tag || ""))) out.push(`Murmurs on the floor: ${team} reached for ${name}.`);
      else if (/steal|value/i.test(String(tag || ""))) out.push(`Scouts nodding — ${team} got value with ${name}.`);
    });
    if (run) out.push(`${run.n} ${WR_POS_NAME[run.pos]} in the last ${recent.length} picks — a run is on.`);
    if (top[0]) out.push(`${getPlayerName(top[0])} is still on the board. Phones are ringing.`);
    if (clockTeamName) out.push(`${clockTeamName} huddling at the table.`);
    if (identity) out.push(`This class is deep in ${identity.deep}, thin at ${identity.thin}.`);
    return out.length ? out : ["The floor is quiet. Every club is waiting on the next card."];
  }, [recent, run, top, clockTeamName, identity]);

  if (!top.length) return null;
  return (
    <section className={`${PREFIX}-wr`} aria-label="War room desk">
      <div className={`${PREFIX}-wr-cards`}>
        <p className={`${PREFIX}-wr-kicker`}>Top of the board</p>
        <div className={`${PREFIX}-wr-spread`}>
          {top.map((p, i) => {
            const biggest = top.reduce((best, q) => ((getRank(q) ?? 9999) < (getRank(best) ?? 9999) ? q : best), top[0]);
            const tag = warRoomTag(p, overallNow, p === biggest);
            const pot = resolvePotentialOvrLabel(p);
            const ovr = resolveCurrentOvrLabel(p);
            const potN = num(String(pot || "").replace(/[^\d.]/g, ""));
            const stock = normalizeStockLabel(p);
            const nat = p.nationality || p.country || p.dossier?.nationality;
            return (
              <button
                type="button"
                key={getId(p)}
                className={`${PREFIX}-wr-card${i === 0 ? " is-pinned" : ""}`}
                style={{ "--tilt": `${[-1.2, 0.8, -0.5, 1.1, -0.9, 0.6, -1.0, 0.9, -0.4, 1.2][i] || 0}deg`, "--i": i }}
                onClick={() => onOpenProspect?.(p)}
              >
                <span className={`${PREFIX}-wr-sticky tone-${tag.tone}`}>{tag.label}</span>
                <span className={`${PREFIX}-wr-rank`}>#{getRank(p) ?? i + 1}</span>
                <strong className={`${PREFIX}-wr-name`}>{getPlayerName(p)}</strong>
                <span className={`${PREFIX}-wr-meta`}>
                  <ProspectFlag country={nat} code={p.country_code || p.nat || nat} width={16} />
                  {getPlayerPosition(p)}
                  {stock === "Rising" ? <em className="up">▲</em> : stock === "Falling" ? <em className="down">▼</em> : null}
                </span>
                <span className={`${PREFIX}-wr-heat`} title={`Potential ${pot || "?"}`}>
                  <i style={{ width: `${potN ? Math.max(8, Math.min(100, (potN - 60) * 2.6)) : 30}%` }} />
                </span>
                <span className={`${PREFIX}-wr-nums`}>{[ovr, pot].filter(Boolean).join(" · ") || "—"}</span>
                <span className={`${PREFIX}-wr-note`}>{warRoomNote(p)}</span>
              </button>
            );
          })}
        </div>
      </div>
      <aside className={`${PREFIX}-wr-board`}>
        <p className={`${PREFIX}-wr-kicker`}>Whiteboard</p>
        {nextUser ? (
          <p className={`${PREFIX}-wr-line is-you`}>
            Your next pick: <b>#{nextUser.pick}</b>
            {nextUser.away ? ` · ${nextUser.away} away` : " · you're up"}
          </p>
        ) : null}
        {run ? (
          <p className={`${PREFIX}-wr-line`}>
            <b>{WR_POS_NAME[run.pos].replace(/^./, (c) => c.toUpperCase())} run</b> — {run.n} of the last {recent.length}
          </p>
        ) : null}
        {identity ? (
          <p className={`${PREFIX}-wr-line`}>
            Class: deep <b>{identity.deep}</b>, thin <b>{identity.thin}</b>
          </p>
        ) : null}
        <p className={`${PREFIX}-wr-line`}>
          <b>{Math.max(0, Number(totalPicks || 224) - Number(completedCount || 0))}</b> picks left · {safeArray(available).length} eligible players
        </p>
      </aside>
      <div className={`${PREFIX}-wr-ticker`} aria-label="Draft room chatter">
        <span className={`${PREFIX}-wr-ticker-flag`}>Floor</span>
        <div className={`${PREFIX}-wr-ticker-view`}>
          <div className={`${PREFIX}-wr-ticker-track`} key={chatter.join("|")}>
            {[...chatter, ...chatter].map((line, i) => (
              <span key={i}>{line}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function BoardPopup({
  open,
  available,
  selectedId,
  isUserPick,
  onSelectProspect,
  onDraftProspect,
  onClose,
  ignoreEscape = false,
}) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape" && !ignoreEscape) onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, ignoreEscape]);

  if (!open) return null;
  return createPortal(
    <div className={`${PREFIX}-scrim`} role="dialog" aria-modal="true" aria-label="Best available" onClick={onClose}>
      <div className={`${PREFIX}-sheet ${PREFIX}-sheet--board`} onClick={(e) => e.stopPropagation()}>
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h2>Best available</h2>
            <p>Public board · scouting file · org ranks — no click required</p>
          </div>
          <button type="button" className="nhlcal-quick-link" onClick={onClose}>
            <Icon name="close" size={12} /> Close
          </button>
        </header>
        <BoardPane
          available={available}
          selectedId={selectedId}
          isUserPick={isUserPick}
          hideHead
          rich
          onSelectProspect={onSelectProspect}
          onDraftProspect={onDraftProspect}
        />
      </div>
    </div>,
    uiPortalTarget()
  );
}

function CompletedDossier({ pick, draft, extra, onClose, revision }) {
  const seeded = useMemo(() => enrichPickFromBoard(pick, draft) || pick, [pick, draft]);
  const prospect = useProspectFile(seeded, revision);
  if (!pick) return null;
  return (
    <DossierPopup
      prospect={prospect || seeded}
      currentPickOverall={pick.overall_pick}
      extra={extra}
      onClose={onClose}
    />
  );
}

function DossierPopup({
  prospect,
  currentPickOverall,
  onClose,
  extra,
  stacked = false,
  isUserPick = false,
  loading = false,
  onDraft,
}) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!prospect) return null;
  return createPortal(
    <div
      className={`${PREFIX}-scrim is-dock${stacked ? " is-stack" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-label="Prospect file"
      onClick={onClose}
    >
      <div className={`${PREFIX}-sheet ${PREFIX}-sheet--dossier`} onClick={(e) => e.stopPropagation()}>
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h2>Prospect file</h2>
            {prospect.overall_pick != null ? (
              <p>
                {[
                  prospect.nhl_team_name || prospect.team_name,
                  formatPick(prospect.overall_pick),
                ].filter(Boolean).join(" · ")}
              </p>
            ) : null}
          </div>
          <button type="button" className="nhlcal-advance-button-secondary edraft-sheet-close" onClick={onClose}>
            <Icon name="close" size={12} /> {stacked ? "Back to board" : "Close"}
          </button>
        </header>
        <div className={`${PREFIX}-sheet-body`}>
          {extra || null}
          <StageDossier prospect={prospect} currentPickOverall={currentPickOverall} />
        </div>
        {isUserPick && onDraft ? (
          <footer className={`${PREFIX}-sheet-foot`}>
            <button
              type="button"
              className="nhlcal-advance-button"
              disabled={loading}
              onClick={() => onDraft(prospect)}
            >
              <Icon name="check" size={12} />
              {loading ? LOADING_COPY.submit : `Draft ${getPlayerName(prospect)}`}
            </button>
          </footer>
        ) : null}
      </div>
    </div>,
    uiPortalTarget()
  );
}

/* ------------------------------------------------------------------ */
/* Centre stage — scouting dossier (position aware)                    */
/* ------------------------------------------------------------------ */


function StageDossier({
  prospect,
  currentPickOverall,
  onFullReport,
  onShortlist,
  onPin,
  shortlisted,
  pinned,
}) {
  const classPool = useProspectClass();
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
  const drafted = prospect.overall_pick != null;
  const name = prospect.prospect_name || getPlayerName(prospect);
  const pos = getPlayerPosition(prospect);
  const nationality = d.nationality || prospect.nationality || prospect.country || null;
  const natCode = d.country_code || prospect.country_code || prospect.nat || prospect.nationality_code || nationality;
  const league = prospect.junior_league || d.league || prospect.league || prospect.league_name || null;
  const nhlClub = prospect.nhl_team_name || (drafted ? prospect.team_name : null);
  const club = prospect.junior_team || d.team || d.club || d.currentClub || (!drafted ? (prospect.team_name || prospect.team) : null);
  const ageRaw = prospect.age ?? d.age;
  const shoots = prospect.handedness || prospect.shoots || prospect.shot_side || prospect.catches || d.handedness || null;
  const height = prospect.height || d.height || d.heightDisplay || null;
  const weight = (prospect.weight || d.weight) ? `${roundInt(prospect.weight || d.weight)} lb` : null;

  const pub = getRank(prospect);
  const dossierModel = buildProspectDossierModel(prospect, d, {
    allProspects: classPool,
    userScoutPct: num(prospect.scouted_percentage ?? prospect.dedicatedScoutingPct ?? d.userScoutPct) ?? 0,
  });

  const archetypeObj = d.archetype && typeof d.archetype === "object" ? d.archetype : null;
  const archetype = archetypeObj?.label || d.player_comparison?.archetype || getDefiningTrait(prospect) || null;
  const archetypeBlurb = archetypeObj?.blurb || d.micro_summary || null;
  const playStyle = d.play_style_block?.label || d.playStyle || d.play_style || null;
  const etaLabel = (d.eta && typeof d.eta === "object" ? d.eta.label : d.eta) || d.estimatedNhlArrival || null;
  const scoutReport = d.scout_report || d.micro_summary || prospect.storyline || null;
  const projectionNotes = safeArray(d.projectionNotes || d.projection_notes)
    .map((n) => (typeof n === "string" ? n : n?.text || n?.note || ""))
    .filter((n) => String(n).trim())
    .slice(0, 3);
  const character = d.character_read || null;
  const comparison = d.player_comparison && typeof d.player_comparison === "object" ? d.player_comparison : null;
  const rights = prospect.rights_card && typeof prospect.rights_card === "object" ? prospect.rights_card : {};
  const wjcBlock = resolveWjcDossierBlock({ ...d, wjcStats: d.wjcStats || d.wjcPerformance || prospect.wjc_stats }, goalie);
  const playoff = d.playoffStats && Number(d.playoffStats.games) > 0 ? d.playoffStats : null;
  const { tag: pickTag, tone: pickTone } = drafted ? getPickDisplayTag(prospect) : { tag: null, tone: "default" };
  const prod = resolveProductionStats(prospect, d.stats);
  const hasBox = prod.games != null && Number(prod.games) > 0;

  const reachDelta = pub != null && currentPickOverall != null ? currentPickOverall - pub : null;
  const identityLine = [
    pos,
    ageRaw != null ? `Age ${roundInt(ageRaw) ?? ageRaw}` : null,
    shoots ? `${shoots}${goalie ? " catch" : " shot"}` : null,
    height,
    weight,
    nationality,
  ].filter(Boolean).join(" · ");

  const showActions = onShortlist || onPin || onFullReport;
  const slotNote = pickTag
    || (reachDelta == null ? null : reachDelta > 0 ? `Fell ${reachDelta}` : reachDelta < 0 ? `Reach ${Math.abs(reachDelta)}` : "On board");

  const productionLine = !hasBox ? null : goalie
    ? {
        line: [
          prod.savePct != null ? `${formatStatCell("SV%", prod.savePct)} SV%` : null,
          prod.gaa != null ? `${formatStatCell("GAA", prod.gaa)} GAA` : null,
        ].filter(Boolean).join(" · ") || `${prod.games} GP`,
        context: [`${prod.games} GP`, prod.wins != null ? `${prod.wins} W` : null, prod.shutouts ? `${prod.shutouts} SO` : null, league]
          .filter(Boolean).join(" · "),
      }
    : {
        line: `${prod.goals ?? 0}G–${prod.assists ?? 0}A–${prod.points ?? 0}P${prod.ppg != null ? ` · ${Number(prod.ppg).toFixed(2)} P/GP` : ""}`,
        context: [`${prod.games} GP`, league, d.stats?.sampleNote && !/\bGP sample\b/i.test(d.stats.sampleNote) ? d.stats.sampleNote : null]
          .filter(Boolean).join(" · "),
      };
  const rightsLine = [
    rights.expected_role,
    rights.returning_to || rights.development_path || null,
    rights.rights_status,
    rights.elc_decision ? `ELC ${String(rights.elc_decision).toLowerCase()}` : null,
    rights.rights_through ? `through ${rights.rights_through}` : null,
  ].filter(Boolean).join(" · ") || null;
  const hasMore = Boolean(
    archetypeBlurb || scoutReport || wjcBlock || playoff || projectionNotes.length || character?.interview_notes
    || comparison?.name || comparison?.label || comparison?.blurb
    || (drafted && (prospect.pick_reason || getBackendWhyWorks(prospect) || getBackendWhyFails(prospect))),
  );

  return (
    <div className={`${PREFIX}-dossier`}>
      <header className={`${PREFIX}-idband`}>
        <ProspectFlag country={nationality} code={natCode} width={40} />
        <div className={`${PREFIX}-idcopy`}>
          <h2>{name}</h2>
          <p>{identityLine}</p>
          <p className={`${PREFIX}-muted`}>
            {[club, league, nhlClub ? `Drafted by ${nhlClub}` : null].filter(Boolean).join(" · ")}
          </p>
        </div>
      </header>

      <ProspectVisualStyles />
      <DossierLayoutStyles />
      <DossierCommandSheet
        model={dossierModel}
        tag={drafted ? [nhlClub, prospect.overall_pick != null ? formatPick(prospect.overall_pick) : null].filter(Boolean).join(" · ") : null}
        portrait={<ProspectFlag country={nationality} code={natCode} width={36} />}
        slotNote={slotNote}
        eta={etaLabel || formatNhlArrivalEta(prospect)}
        playStyle={playStyle || archetype}
        production={productionLine}
        currentPick={currentPickOverall}
        footnote={rightsLine}
      />

      {hasMore ? (
        <details className="cs-more" open={drafted}>
          <summary>{drafted ? "Pick notes, history and full report" : "Full report, history and notes"}</summary>
          {archetypeBlurb || scoutReport ? (
            <p className={`${PREFIX}-report`}>{archetypeBlurb || scoutReport}</p>
          ) : null}
      <div className={`${PREFIX}-dossier-grid`}>
        <div className={`${PREFIX}-dossier-main`}>

          {wjcBlock ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="World Juniors" meta={wjcBlock.grade || wjcBlock.headline} icon="target" tone="blue" />
              {wjcBlock.headline ? <p className={`${PREFIX}-muted`}>{wjcBlock.headline}</p> : null}
              {wjcBlock.summary ? <p className={`${PREFIX}-report`}>{wjcBlock.summary}</p> : null}
              <div className={`${PREFIX}-prod-strip`}>
                {wjcBlock.tiles.map((tile) => (
                  <div key={`wjc-${tile.label}`} className={`${PREFIX}-prod-cell`}>
                    <span>{tile.label}</span>
                    <strong className={`${PREFIX}-tabular ${PREFIX}-num-pop tone-cyan`}>{tile.value}</strong>
                  </div>
                ))}
              </div>
              {wjcBlock.impactLines?.length ? (
                <ul className={`${PREFIX}-list`}>
                  {wjcBlock.impactLines.map((line) => <li key={line}>{line}</li>)}
                </ul>
              ) : null}
            </section>
          ) : null}

          {playoff ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="Playoff run" icon="flame" />
              <div className={`${PREFIX}-prod-strip`}>
                {[["GP", playoff.games], ["G", playoff.goals], ["A", playoff.assists], ["PTS", playoff.points], playoff.ppg != null ? ["P/GP", playoff.ppg] : null]
                  .filter(Boolean)
                  .map(([label, value]) => {
                    const formatted = formatStatCell(label, value);
                    if (formatted == null) return null;
                    return (
                      <div key={`po-${label}`} className={`${PREFIX}-prod-cell`}>
                        <span>{label}</span>
                        <strong className={`${PREFIX}-tabular ${PREFIX}-num-pop tone-cyan`}>{formatted}</strong>
                      </div>
                    );
                  })}
              </div>
            </section>
          ) : null}

          {projectionNotes.length ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="Projection notes" icon="book" />
              <ul className={`${PREFIX}-list`}>
                {projectionNotes.map((n, i) => <li key={`n-${i}`}>{n}</li>)}
              </ul>
            </section>
          ) : null}

          {comparison?.name || comparison?.label || comparison?.blurb ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="Comparison" meta={comparison.archetype || null} icon="compare" />
              <p className={`${PREFIX}-report`}>
                {comparison.name || comparison.label}
                {comparison.blurb ? ` — ${comparison.blurb}` : ""}
              </p>
            </section>
          ) : null}

          {character?.interview_notes ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead title="Interview notes" />
              <p className={`${PREFIX}-report`}>{character.interview_notes}</p>
            </section>
          ) : null}

          {drafted && (prospect.pick_reason || getBackendWhyWorks(prospect) || getBackendWhyFails(prospect)) ? (
            <section className={`${PREFIX}-sec`}>
              <SectionHead
                title="Why this pick"
                meta={pickTag ? <PickBadge tag={pickTag} tone={pickTone} /> : null}
                icon="shield"
              />
              {prospect.pick_reason ? <p className={`${PREFIX}-report`}>{prospect.pick_reason}</p> : null}
              {getBackendWhyWorks(prospect) ? <p className={`${PREFIX}-callout`}>{getBackendWhyWorks(prospect)}</p> : null}
              {getBackendWhyFails(prospect) ? <p className={`${PREFIX}-callout is-warn`}>{getBackendWhyFails(prospect)}</p> : null}
            </section>
          ) : null}

        </div>
      </div>
        </details>
      ) : null}

      {showActions ? (
        <div className={`${PREFIX}-dossier-actions`}>
          {onShortlist ? (
            <button type="button" className="nhlcal-advance-button-secondary" onClick={() => onShortlist(prospect)}>
              {shortlisted ? "Remove shortlist" : "Shortlist"}
            </button>
          ) : null}
          {onPin ? (
            <button type="button" className="nhlcal-advance-button-secondary" onClick={() => onPin(prospect)}>
              {pinned ? "Unpin" : "Pin"}
            </button>
          ) : null}
          {onFullReport ? (
            <button type="button" className="nhlcal-advance-button-secondary" onClick={() => onFullReport(prospect)}>
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
        className={`${PREFIX}-sheet ${PREFIX}-sheet--dossier`}
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
          <button type="button" className="nhlcal-advance-button-secondary edraft-sheet-close" ref={closeRef} onClick={onClose}>
            <Icon name="close" size={12} /> Close
          </button>
        </header>

        <div className={`${PREFIX}-sheet-body ${PREFIX}-sheet--dossier-body`}>
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
              <button type="button" className={`nhlcal-advance-button`} disabled={loading} onClick={() => onDraft?.(prospect)}>
                <Icon name="check" size={12} /> Draft {getPlayerName(prospect)}
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
              onClick={() => detail && onDraft(detail)}
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

function DraftTradeValueBar({ sendValue, receiveValue, sendLabel = "You send", receiveLabel = "You receive" }) {
  const send = Math.max(0, Number(sendValue) || 0);
  const recv = Math.max(0, Number(receiveValue) || 0);
  const total = Math.max(send + recv, 0.01);
  const recvPct = Math.round((recv / total) * 100);
  const sendPct = 100 - recvPct;
  const delta = Math.abs(send - recv);
  const fair = delta <= Math.max(3.5, total * 0.1);
  return (
    <div className={`${PREFIX}-trade-meter`} aria-label="Trade value comparison">
      <div className={`${PREFIX}-trade-meter__labels`}>
        <span>{sendLabel} · {send ? send.toFixed(1) : "—"}</span>
        <em>{fair ? "Balanced chart" : recv > send ? "Value tilts your way" : "They pay to climb"}</em>
        <span>{receiveLabel} · {recv ? recv.toFixed(1) : "—"}</span>
      </div>
      <div className={`${PREFIX}-trade-meter__track`}>
        <span className={`${PREFIX}-trade-meter__recv`} style={{ width: `${recvPct}%` }} />
        <span className={`${PREFIX}-trade-meter__send`} style={{ width: `${sendPct}%` }} />
      </div>
    </div>
  );
}

function TradeDownOfferCard({ offer, expanded, onToggle, onAccept, accepting }) {
  const candidates = safeArray(offer.candidate_intel?.length ? offer.candidate_intel : offer.target_candidates);
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
  const meter = offer.value_meter || {};
  const sendVal = meter.user_send_value ?? offer.on_clock_chart_value;
  const recvVal = meter.user_receive_value ?? offer.incoming_chart_value;
  const styleClass = offer.offer_style === "desperate" ? " is-desperate" : offer.offer_style === "pay_up" ? " is-pay_up" : "";
  const intelLines = safeArray(offer.intel_lines);
  const partnerId = offer.from_team_id || offer.from_team_name;

  return (
    <article className={`${PREFIX}-trade-card${styleClass}`}>
      <button type="button" className={`${PREFIX}-trade-head`} onClick={onToggle} aria-expanded={expanded}>
        <TeamLogo teamId={partnerId} teamName={offer.team_name || offer.from_team_name} size="md" />
        <span className={`${PREFIX}-trade-head-main`}>
          <strong>{offer.team_name || offer.from_team_name || "Club"}</strong>
          <span>
            {[
              offer.partner_overall_pick ? `Move to #${offer.partner_overall_pick}` : null,
              offer.slots_moved ? `+${offer.slots_moved} slots` : null,
              offer.urgency_label || offer.urgency,
            ].filter(Boolean).join(" · ")}
          </span>
        </span>
        <span className={`${PREFIX}-trade-badges`}>
          {offer.value ? <Chip icon="target">{offer.value}</Chip> : null}
          {offer.slot_value_gap != null ? (
            <Chip icon="chart">{`Gap ${Number(offer.slot_value_gap).toFixed(1)}`}</Chip>
          ) : null}
        </span>
      </button>
      {expanded ? (
        <div className={`${PREFIX}-trade-body`}>
          {(sendVal != null || recvVal != null) ? (
            <DraftTradeValueBar
              sendValue={sendVal}
              receiveValue={recvVal}
              sendLabel="You send"
              receiveLabel="You receive"
            />
          ) : null}
          <div className={`${PREFIX}-trade-split`}>
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
          {intelLines.length ? (
            <div className={`${PREFIX}-trade-intel`}>
              <strong>{offer.reason_headline || offer.pitch_headline || "Floor intel"}</strong>
              <ul>
                {intelLines.map((line, idx) => <li key={idx}>{line}</li>)}
              </ul>
            </div>
          ) : null}
          {candidates.length ? (
            <div>
              <p className={`${PREFIX}-muted`}>Rumoured targets — one is real</p>
              <div className={`${PREFIX}-trade-targets`}>
                {candidates.map((c, idx) => (
                  <div
                    key={c.prospect_id || c.name || idx}
                    className={`${PREFIX}-trade-target${c.need_fit ? " is-fit" : ""}`}
                  >
                    <strong>{c.name || "Prospect"}{c.position ? ` (${c.position})` : ""}</strong>
                    {c.public_rank_band ? <em>{c.public_rank_band}</em> : null}
                    {c.scout_buzz ? <span>{c.scout_buzz}</span> : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          <div className={`${PREFIX}-trade-actions`}>
            <button
              type="button"
              className={`nhlcal-advance-button`}
              disabled={accepting || !incoming.length}
              onClick={() => onAccept?.(offer)}
            >
              <Icon name="check" size={12} /> {accepting ? "Accepting…" : "Accept and move down"}
            </button>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function TradePanel({ draft, onClose, onAccept, accepting }) {
  const offers = safeArray(draft?.trade_offers || draft?.draft_day_trade_offers || draft?.pick_trade_offers);
  const [expandedId, setExpandedId] = useState(null);
  useEffect(() => {
    if (!offers.length) {
      setExpandedId(null);
      return;
    }
    const firstKey = offers[0]?.offer_id || "0";
    setExpandedId((prev) => (prev == null || !offers.some((o, i) => (o.offer_id || String(i)) === prev) ? firstKey : prev));
  }, [offers]);
  return createPortal(
    <div className={`${PREFIX}-scrim`} role="dialog" aria-modal="true" aria-label="Trade down">
      <div className={`${PREFIX}-sheet`} style={{ width: "min(760px,100%)" }}>
        <header className={`${PREFIX}-sheet-head`}>
          <div>
            <h3>Trade down</h3>
            <p>Clubs bidding to climb, including teams holding later picks</p>
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
            <div className={`${PREFIX}-trade-stack`}>
              {offers.map((offer, i) => {
                const key = offer.offer_id || `offer-${i}`;
                return (
                  <TradeDownOfferCard
                    key={key}
                    offer={offer}
                    expanded={expandedId === key}
                    onToggle={() => setExpandedId((prev) => (prev === key ? null : key))}
                    onAccept={onAccept}
                    accepting={accepting}
                  />
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

  const ovr = revealedOvrValue(pick) ?? resolveCurrentOvrLabel(pick);
  const ovrTone = pickOvrPopTone(tone);
  const revealTone = tone && tone !== "default" ? ` tone-${tone}` : "";

  return (
    <div className={`${PREFIX}-reveal${revealTone}`}>
      <div className={`${PREFIX}-reveal-card`}>
        <p>With the {formatPick(pick.overall_pick)} selection</p>
        <TeamLogo teamId={pick.team_id} teamName={pick.team_name} size="lg" />
        <h2>{pick.team_name}</h2>
        {pick.is_traded && (pick.via_team_name || pick.via_team_id) ? (
          <p>via {teamAbbrev(pick.via_team_id, pick.via_team_name)}</p>
        ) : null}
        <p>selects</p>
        <h1>{pick.prospect_name}</h1>
        {ovr ? (
          <span className={`${PREFIX}-tabular ${PREFIX}-num-pop tone-${ovrTone}`} style={{ fontSize: "1.85rem" }}>
            {ovr} OVR
          </span>
        ) : null}
        {tag ? <PickBadge tag={tag} tone={tone} className={`${PREFIX}-reveal-tag`} /> : null}
        <div className={`${PREFIX}-reveal-meta`}>
          {pick.position ? <span className={`${PREFIX}-text-pop tone-cyan`}>{pick.position}</span> : null}
          {pick.league ? <span className={`${PREFIX}-text-pop tone-muted`}>{pick.league}</span> : null}
          {pub != null ? <span className={`${PREFIX}-text-pop tone-blue`}>{`Board #${pub}`}</span> : null}
          {movement?.label ? <span className={`${PREFIX}-text-pop tone-${movement.tone || "default"}`}>{movement.label}</span> : null}
        </div>
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
            <Chip key={p?.overall_pick ?? i} icon="check">{`${formatPick(p.overall_pick)} ${p.prospect_name}`}</Chip>
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
/* Recap — draft-night broadcast (drx-*). Styles: EntryDraftRecap.css. */

function RecapDash() {
  return <span className="drx-dash">—</span>;
}

function recapCapTone(grade) {
  const g = String(grade || "").trim().toUpperCase();
  if (/^[AB]/.test(g)) return "good";
  if (/^[DF]/.test(g) || g === "—") return "bad";
  return "mid";
}

function parseLeagueRank(rankStr) {
  const m = String(rankStr || "").match(/(\d+)/);
  const n = m ? Number(m[1]) : null;
  return Number.isFinite(n) && n >= 1 && n <= 32 ? n : null;
}

function recapOrdinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

function buildGmHeadline(r, grade) {
  const steals = Number(r.user_steal_count) || 0;
  const reaches = Number(r.user_reach_count) || 0;
  const picks = Number(r.user_pick_count) || 0;
  const pl = (n, w, wp = `${w}s`) => `${n} ${n === 1 ? w : wp}`;
  const g = String(grade || "").trim().toUpperCase();
  const band = /^A/.test(g) ? "A" : /^B/.test(g) ? "B" : /^C/.test(g) ? "C" : "D";
  const lines = {
    A: [
      `${pl(steals, "steal")} and a board that broke your way. This class travels.`,
      `The room wanted noise. You walked out with ${pl(picks, "keeper")}.`,
    ],
    B: [
      reaches
        ? `${pl(reaches, "reach", "reaches")}, but the ledger still balances.`
        : `Disciplined night. ${pl(steals, "steal")} to point at.`,
      `${pl(picks, "pick")} in, ${pl(steals, "steal")} banked. The ceiling decides the grade from here.`,
    ],
    C: [
      `Mixed reviews: ${pl(reaches, "reach", "reaches")}, ${pl(steals, "steal")}.`,
      `The war room split on this ${picks}-pick class. Development has to carry it.`,
    ],
    D: [
      reaches && !steals
        ? `${pl(reaches, "reach", "reaches")}, zero steals. Scouts will remember this one.`
        : `Rough night: ${pl(reaches, "reach", "reaches")}, ${pl(steals, "steal")}.`,
      `${pl(picks, "pick")} and the board won most of them.`,
    ],
  };
  const pool = lines[band];
  return pool[(picks + steals + reaches) % pool.length];
}

function selectFranchisePick(picks, bestId) {
  if (!picks.length) return null;
  const sorted = picks.slice().sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));
  const roundOne = sorted.find((p) => Number(p.round) === 1 && Number(p.overall_pick) <= 32);
  if (roundOne) return roundOne;
  if (bestId) {
    const best = picks.find((p) => String(p.prospect_id) === bestId);
    if (best) return best;
  }
  return sorted[0];
}

function recapNumeric(val) {
  if (val == null || val === "" || Number.isNaN(Number(val))) return null;
  return Math.round(Number(val));
}

function recapLowConfidence(p) {
  const point = getConfidence(p);
  if (point != null && point < 40) return true;
  const range = getConfidenceRange(p);
  return Boolean(range && range[1] < 40);
}

function buildScoutNote(pick) {
  const scouted = scoutedOvrLabel(pick);
  const revealed = recapNumeric(revealedOvrValue(pick));
  const conf = getConfidence(pick);
  if (conf != null && conf < 40) return `Boom-or-bust: ${Math.round(conf)}% confidence`;
  if (scouted && revealed != null) return `Scouted ${scouted}, landed at ${revealed}`;
  if (scouted) return `Scouted ${scouted}`;
  const pub = getPublicRankAtPick(pick);
  if (pub != null && revealed != null) return `Board #${pub} · revealed ${revealed}`;
  return null;
}

function bestValueInsight(picks) {
  let best = null;
  let bestD = -Infinity;
  picks.forEach((p) => {
    const d = pubDelta(p);
    if (d != null && d > bestD) {
      bestD = d;
      best = p;
    }
  });
  if (!best || bestD < 5) return null;
  const rd = best.round != null ? `Round ${best.round}` : `Pick ${formatPick(best.overall_pick)}`;
  return `Best value: ${rd} (${fmtSigned(bestD)})`;
}

function stillBoardRank(entry) {
  const rank = entry?.projected_rank ?? entry?.final_rank ?? entry?.rank;
  if (rank != null && rank !== "" && Number.isFinite(Number(rank))) return `#${rank}`;
  const projected = entry?.projected;
  if (!projected || projected === "Unranked") return "";
  const text = String(projected).trim();
  if (!text) return "";
  if (/^brd\b/i.test(text)) return text.replace(/^brd\s*/i, "");
  return text.startsWith("#") ? text : `#${text}`;
}

function prefersReducedMotion() {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function useCountUp(target, active) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!active || target == null || Number.isNaN(Number(target))) {
      setVal(target == null ? null : Number(target));
      return undefined;
    }
    const end = Math.round(Number(target));
    if (prefersReducedMotion() || typeof requestAnimationFrame !== "function") {
      setVal(end);
      return undefined;
    }
    const start = performance.now();
    const dur = 700;
    let frame = 0;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / dur);
      setVal(Math.round(end * (1 - (1 - t) ** 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, active]);
  return val;
}

function DrxLogo({ teamId, teamName, size = "md" }) {
  const src = getTeamLogoSrc({ team_id: teamId, name: teamName });
  const abbr = getTeamAbbreviation({ team_id: teamId, name: teamName }) || teamAbbrev(teamId, teamName) || "TM";
  return (
    <span className={`drx-logo drx-logo--${size}`} title={teamName || abbr}>
      {src ? <img src={src} alt="" loading="lazy" /> : <b>{abbr}</b>}
    </span>
  );
}

function DrxRankLadder({ rank }) {
  if (!rank) return null;
  return (
    <div className="drx-ladder-wrap" aria-label={`Class ranked ${recapOrdinal(rank)} of 32`}>
      <div className="drx-ladder">
        {Array.from({ length: 32 }, (_, i) => {
          const n = i + 1;
          const cls = [n <= 8 ? "is-top" : "", n >= 25 ? "is-bottom" : "", n === rank ? "is-you" : ""]
            .filter(Boolean)
            .join(" ");
          return <i key={n} className={cls || undefined} />;
        })}
      </div>
      <div className="drx-ladder-label">
        <span>1st</span>
        <b>{recapOrdinal(rank)} of 32 classes</b>
        <span>32nd</span>
      </div>
    </div>
  );
}

function DrxRink({ posBreakdown, needItems }) {
  const count = (k) => Number(posBreakdown?.[k] || 0);
  const lw = count("LW");
  const rw = count("RW") || (lw ? 0 : count("W"));
  const spots = [
    { key: "LW", n: lw },
    { key: "C", n: count("C") },
    { key: "RW", n: rw },
    { key: "D", n: count("D") },
    { key: "G", n: count("G") },
  ];
  const needByPos = {};
  needItems.forEach((n) => {
    const cat = String(n.category || "").toUpperCase();
    if (cat.includes("GOAL")) needByPos.G = n;
    else if (cat.includes("DEF")) needByPos.D = n;
    else if (cat.includes("WING")) needByPos.RW = n;
    else if (cat.includes("CENT")) needByPos.C = n;
  });
  return (
    <div className="drx-rink-wrap">
      <div className="drx-rink" aria-label="Class DNA by position">
        {spots.map((s) => {
          const need = needByPos[s.key];
          return (
            <div
              key={s.key}
              className={`drx-spot drx-spot--${s.key.toLowerCase()}${s.n ? " has" : ""}${need && !need.filled ? " is-need" : ""}`}
              title={need ? needLabel(need) : undefined}
            >
              <b>{s.n || "–"}</b>
              <small>{s.key}</small>
            </div>
          );
        })}
      </div>
      <div className="drx-rink-cap">
        <span className="drx-kicker">Class DNA</span>
        {needItems.map((n, i) => (
          <em key={n.category || i} className={n.filled ? "is-filled" : "is-open"}>
            {needLabel(n)} · {n.filled ? "filled" : "open"}
          </em>
        ))}
      </div>
    </div>
  );
}

function potTier(pot, pos) {
  if (pot == null) return null;
  const P = String(pos || "").toUpperCase();
  const grp = P.includes("G") ? "G" : /^(LD|RD|D)$/.test(P) || P.startsWith("D") ? "D" : "F";
  const names = {
    F: ["First-line ceiling", "Top-six ceiling", "Middle-six ceiling", "Depth ceiling"],
    D: ["Top-pair ceiling", "Top-four ceiling", "Third-pair ceiling", "Depth ceiling"],
    G: ["Franchise starter", "Starter ceiling", "Backup ceiling", "Depth ceiling"],
  }[grp];
  if (pot >= 88) return { key: "elite", label: names[0] };
  if (pot >= 84) return { key: "high", label: names[1] };
  if (pot >= 79) return { key: "mid", label: names[2] };
  return { key: "low", label: names[3] };
}

const LANE_MIN = 60;
const LANE_MAX = 95;
function lanePct(v) {
  return Math.max(0, Math.min(100, ((Number(v) - LANE_MIN) / (LANE_MAX - LANE_MIN)) * 100));
}
function laneZones(pos) {
  const P = String(pos || "").toUpperCase();
  const grp = P.includes("G") ? "G" : P.startsWith("D") || /^(LD|RD)$/.test(P) ? "D" : "F";
  const names = {
    F: ["Depth", "Middle six", "Top six", "First line"],
    D: ["Depth", "Third pair", "Top four", "Top pair"],
    G: ["Depth", "Backup", "Starter", "Franchise"],
  }[grp];
  const cuts = [LANE_MIN, 79, 84, 88, LANE_MAX];
  return names.map((label, i) => ({
    label,
    key: ["low", "mid", "high", "elite"][i],
    left: lanePct(cuts[i]),
    width: lanePct(cuts[i + 1]) - lanePct(cuts[i]),
  }));
}

/* Development lane: the lineup roles laid out left to right; a dot for where
   he is now, a marker for where he can get to. Hockey roles, not a gauge. */
function DrxLane({ ovr, pot, pos, labels = false }) {
  const tier = potTier(pot, pos);
  const zones = laneZones(pos);
  const now = ovr != null ? lanePct(ovr) : null;
  const cap = pot != null ? lanePct(Math.max(pot, ovr ?? pot)) : null;
  return (
    <span className={`drx-lane tier-${tier ? tier.key : "none"}${labels ? " has-labels" : ""}`}>
      <span className="drx-lane__track">
        {zones.map((z) => (
          <span
            key={z.key}
            className={`drx-lane__zone is-${z.key}${tier && tier.key === z.key ? " is-target" : ""}`}
            style={{ left: `${z.left}%`, width: `${z.width}%` }}
          >
            {labels ? <em>{z.label}</em> : null}
          </span>
        ))}
        {now != null && cap != null ? (
          <span className="drx-lane__climb" style={{ left: `${now}%`, width: `${Math.max(0, cap - now)}%` }} />
        ) : null}
        {now != null ? <span className="drx-lane__now" style={{ left: `${now}%` }} title={`Now ${ovr}`} /> : null}
        {cap != null ? <span className="drx-lane__cap" style={{ left: `${cap}%` }} title={`Potential ${pot}`} /> : null}
      </span>
    </span>
  );
}

function DrxPot({ ovr, pot, pos, big = false }) {
  const tier = potTier(pot, pos);
  return (
    <span className={`drx-pot tier-${tier ? tier.key : "none"}${big ? " is-big" : ""}`}>
      <strong>{pot ?? "—"}</strong>
      <small>{ovr != null ? `Potential · now ${ovr}` : "Potential"}</small>
    </span>
  );
}

function DrxTier({ pot, pos, big = false }) {
  const tier = potTier(pot, pos);
  if (!tier) return null;
  return <span className={`drx-tier tier-${tier.key}${big ? " is-big" : ""}`}>{tier.label}</span>;
}

function verdictDisplay(verdict) {
  if (!verdict) return "";
  return verdict.text.replace(/^[A-Z]+/, (w) => w.charAt(0) + w.slice(1).toLowerCase());
}

function DrxStamp({ verdict, big = false, pub = null }) {
  if (!verdict) return null;
  return (
    <span className={`drx-stamp drx-stamp--${verdict.tone}${big ? " is-big" : ""}`}>
      <b>{verdictDisplay(verdict)}</b>
      {pub != null ? <small>Board rank #{pub}</small> : null}
    </span>
  );
}

function pickNationality(pick) {
  return pick.nationality || pick.country || pick.dossier?.nationality || null;
}

function pickParts(pick) {
  return {
    verdict: recapVerdict(pick),
    ovr: recapNumeric(revealedOvrValue(pick)),
    pot: recapNumeric(potVal(pick)),
    league: projectionLeagueLabel(pick),
    name: pick.prospect_name || getPlayerName(pick) || "",
    pos: pick.position || getPlayerPosition(pick) || "",
    pub: getPublicRankAtPick(pick),
    low: recapLowConfidence(pick),
    nat: pickNationality(pick),
  };
}

function DrxFranchiseCard({ pick, onOpen, teamName, draftYear }) {
  const { verdict, ovr, pot, league, name, pos, pub, low, nat } = pickParts(pick);
  const tier = potTier(pot, pos);
  const conf = getConfidence(pick);
  const pickOrd = pick.overall_pick != null ? recapOrdinal(Number(pick.overall_pick)) : null;
  return (
    <button type="button" className={`drx-fcard tier-${tier ? tier.key : "none"}`} onClick={() => onOpen(pick)}>
      <span className="drx-fcard__call">
        {pickOrd ? `With the ${pickOrd} pick${draftYear ? ` in the ${draftYear} draft` : ""}, ` : ""}
        {teamName ? `the ${teamName} select` : "you selected"}
      </span>
      <span className="drx-fcard__name">
        {nat ? <ProspectFlag country={nat} width={30} /> : null}
        <span>{name || <RecapDash />}</span>
      </span>
      <span className="drx-fcard__meta">
        {[pos, league, buildScoutNote(pick)].filter(Boolean).join(" · ")}
      </span>
      <span className="drx-fcard__grid">
        <DrxPot ovr={ovr} pot={pot} pos={pos} big />
        <span className="drx-fcard__lane">
          {tier ? <span className={`drx-tier tier-${tier.key} is-big`}>{tier.label}</span> : null}
          <DrxLane ovr={ovr} pot={pot} pos={pos} labels />
        </span>
        <span className="drx-fcard__facts">
          <DrxStamp verdict={verdict} pub={pub} big />
          {conf != null ? (
            <span className={`drx-fact${low ? " is-low" : ""}`}><b>{Math.round(conf)}%</b> scouting confidence</span>
          ) : null}
          <span className="drx-open">Open his file</span>
        </span>
      </span>
    </button>
  );
}

function DrxSlip({ pick, index, isBest, onOpen }) {
  const { verdict, ovr, pot, league, name, pos, pub, low, nat } = pickParts(pick);
  const tier = potTier(pot, pos);
  return (
    <button
      type="button"
      className={`drx-slip${isBest ? " is-best" : ""}`}
      style={{ "--i": index }}
      title={buildScoutNote(pick) || SCOUTED_NOTE}
      onClick={() => onOpen(pick)}
    >
      <span className="drx-slip__stub">
        <b>{pick.overall_pick ?? "—"}</b>
        <small>Round {pick.round ?? "—"}</small>
      </span>
      <span className="drx-slip__id">
        <span className="drx-slip__name">
          {nat ? <ProspectFlag country={nat} width={20} /> : null}
          <span>{name || <RecapDash />}</span>
        </span>
        <span className="drx-slip__sub">
          <span className="drx-slip__pos">{pos}</span>
          {tier ? <span className={`drx-tier tier-${tier.key}`}>{tier.label}</span> : null}
          {league ? <em>{league}</em> : null}
          {low ? <span className="drx-lowconf">{buildScoutNote(pick) || "Low confidence"}</span> : null}
        </span>
      </span>
      <DrxLane ovr={ovr} pot={pot} pos={pos} />
      <DrxPot ovr={ovr} pot={pot} pos={pos} />
      <DrxStamp verdict={verdict} pub={pub} />
    </button>
  );
}

function DrxTeamBoard({ teamId, teamName, completed, onOpen, onClose }) {
  const rows = safeArray(completed)
    .filter((p) => String(p.team_id) === String(teamId))
    .sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));
  return (
    <div className="drx-teamboard">
      <div className="drx-teamboard__head">
        <DrxLogo teamId={teamId} teamName={teamName} size="md" />
        <h4>{teamName || teamAbbrev(teamId, teamName)}</h4>
        <span>{rows.length} pick{rows.length === 1 ? "" : "s"}</span>
        <button type="button" className="drx-card__more" onClick={onClose}>Close</button>
      </div>
      {rows.length ? (
        <div className="drx-slips">
          {rows.map((p, i) => (
            <DrxSlip key={p.pick_id || p.overall_pick || p.prospect_id || i} pick={p} index={i} onOpen={onOpen} />
          ))}
        </div>
      ) : (
        <p className="drx-empty">No picks for this team.</p>
      )}
    </div>
  );
}

function DrxCeilingMap({ picks }) {
  const counts = { elite: 0, high: 0, mid: 0, low: 0 };
  const labels = {};
  picks.forEach((p) => {
    const pos = p.position || getPlayerPosition(p) || "";
    const t = potTier(recapNumeric(potVal(p)), pos);
    if (!t) return;
    counts[t.key] += 1;
    if (!labels[t.key]) labels[t.key] = t.key === "elite" ? "Elite" : t.key === "high" ? "High" : t.key === "mid" ? "Middle" : "Depth";
  });
  const total = counts.elite + counts.high + counts.mid + counts.low;
  if (!total) return null;
  return (
    <div className="drx-ceilmap" aria-label="Class ceiling">
      <span className="drx-kicker">Class ceiling</span>
      <div className="drx-ceilmap__bar">
        {["elite", "high", "mid", "low"].map((k) => (counts[k] ? (
          <i key={k} className={`tier-${k}`} style={{ flexGrow: counts[k] }} title={`${counts[k]} ${labels[k]}`} />
        ) : null))}
      </div>
      <div className="drx-ceilmap__legend">
        {["elite", "high", "mid", "low"].map((k) => (counts[k] ? (
          <span key={k} className={`tier-${k}`}><b>{counts[k]}</b> {labels[k].toLowerCase()}</span>
        ) : null))}
      </div>
    </div>
  );
}

function moverBoardRank(p) {
  const b = getPublicRankAtPick(p);
  return b != null ? b : num(p?.final_rank);
}

function DrxMover({ pick, tone, lead, isUser, onOpen }) {
  const d = pubDelta(pick);
  const signed = fmtSigned(d);
  const name = pick?.prospect_name || getPlayerName(pick) || "";
  const pos = pick?.position || "";
  const abbr = teamAbbrev(pick?.team_id, pick?.team_name);
  const board = moverBoardRank(pick);
  const spots = d != null ? Math.abs(d) : null;
  const travel = spots != null ? Math.min(100, Math.max(12, (spots / 150) * 100)) : 40;
  return (
    <button
      type="button"
      className={`drx-mover tone-${tone}${lead ? " is-lead" : ""}${isUser ? " is-user" : ""}`}
      onClick={() => onOpen(pick)}
    >
      <DrxLogo teamId={pick?.team_id} teamName={pick?.team_name} size={lead ? "lg" : "sm"} />
      <span className="drx-mover__body">
        <strong>{name || <RecapDash />}</strong>
        <small>
          {[pos, abbr].filter(Boolean).join(" · ")}
          {!lead && board != null ? ` · #${board} → #${pick?.overall_pick ?? "—"}` : ""}
          {isUser ? <em className="drx-you">Your pick</em> : null}
        </small>
        {lead ? (
          <span className="drx-move-track">
            <span>Board #{board ?? "—"}</span>
            <i aria-hidden="true"><b style={{ width: `${travel}%` }} /></i>
            <span>Pick #{pick?.overall_pick ?? "—"}</span>
          </span>
        ) : null}
        {lead && spots != null ? (
          <small className="drx-mover__move">{tone === "pos" ? `Slid ${spots} spots` : `Jumped ${spots} spots`}</small>
        ) : null}
      </span>
      <span className="drx-mover__delta">{signed ?? "—"}</span>
    </button>
  );
}

function DrxMoverCard({ title, kind, rows, tone, empty, userTeamId, onOpen }) {
  const [expanded, setExpanded] = useState(false);
  const shown = rows.slice(0, expanded ? 10 : 3);
  return (
    <section className={`drx-card drx-card--${kind}`}>
      <header className="drx-card__head">
        <h4>{title}</h4>
        <span className="drx-count">{rows.length}</span>
      </header>
      <div className={`drx-card__list${expanded ? " is-expanded" : ""}`}>
        {shown.map((p, i) => (
          <DrxMover
            key={p?.pick_id || p?.overall_pick || p?.prospect_id || i}
            pick={p}
            tone={tone}
            lead={i === 0}
            isUser={String(p?.team_id) === String(userTeamId)}
            onOpen={onOpen}
          />
        ))}
        {!shown.length ? <p className="drx-empty">{empty}</p> : null}
      </div>
      {rows.length > 3 ? (
        <button type="button" className="drx-card__more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : `View all ${rows.length}`} <Icon name={expanded ? "up" : "down"} size={11} />
        </button>
      ) : null}
    </section>
  );
}

function DrxBoardCard({ rows }) {
  const [expanded, setExpanded] = useState(false);
  const shown = rows.slice(0, expanded ? 10 : 3);
  return (
    <section className="drx-card drx-card--board">
      <header className="drx-card__head">
        <h4>Still on the board</h4>
        <span className="drx-count">{rows.length}</span>
      </header>
      <div className={`drx-card__list${expanded ? " is-expanded" : ""}`}>
        {shown.length ? shown.map((entry, i) => {
          const name = entry?.name || entry?.prospect_name || "";
          const pos = entry?.position ? String(entry.position).trim() : "";
          const league = entry?.league || entry?.current_league || "";
          const passed = Boolean(entry?.user_passed || entry?.passed_by_user);
          return (
            <div key={entry?.key || entry?.prospect_id || name || i} className={`drx-avail${passed ? " is-user" : ""}`}>
              <span className="drx-avail__id">
                <strong>
                  {pos ? <span className="drx-pos">{pos}</span> : null}
                  <span>{name || <RecapDash />}</span>
                </strong>
                <small>
                  {[league, entry?.status].filter(Boolean).join(" · ") || "Undrafted"}
                  {passed ? <em className="drx-you">You passed</em> : null}
                </small>
              </span>
              <span className="drx-avail__rank">
                <small>Board</small>
                <b>{stillBoardRank(entry) || "—"}</b>
              </span>
            </div>
          );
        }) : <p className="drx-empty">The board cleared out.</p>}
      </div>
      {rows.length > 3 ? (
        <button type="button" className="drx-card__more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show less" : `View all ${rows.length}`} <Icon name={expanded ? "up" : "down"} size={11} />
        </button>
      ) : null}
    </section>
  );
}

function RecapShow({ draft, completed, userPicks, draftOrder, userTeamId, franchiseState, onContinue }) {
  const r = draft?.recap || {};
  const [profilePick, setProfilePick] = useState(null);
  const [focusedTeamId, setFocusedTeamId] = useState(null);
  const [sortBy, setSortBy] = useState("pick");
  const picks = safeArray(userPicks).slice().sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));

  const orderSlot = safeArray(draftOrder).find((s) => String(s.team_id) === String(userTeamId));
  const teamId = picks[0]?.team_id ?? picks[0]?.team ?? userTeamId;
  const teamName = picks[0]?.team_name || orderSlot?.team_name || "";
  const teamLogo = getTeamLogoSrc({ team_id: teamId, name: teamName });

  const grade = r.user_grade || draft?.user_draft_grade || buildDraftGrade(picks) || "—";
  const capTone = recapCapTone(grade);
  const posBreakdown = r.user_position_breakdown || r.position_breakdown || {};
  const steals = Array.isArray(r.best_steals) ? r.best_steals : safeArray(completed).filter((p) => p.was_steal);
  const reaches = Array.isArray(r.biggest_reaches) ? r.biggest_reaches : safeArray(completed).filter((p) => p.was_reach);
  const extraStory = safeArray(r.headlines).find((line) => line && !isStealOrReachLine(line) && !/best value/i.test(line)) || "";
  const needItems = safeArray(r.needs_report).filter((n) => n?.category);
  const stillOnBoard = safeArray(r.still_on_board).filter((entry) => {
    const name = entry?.name || entry?.prospect_name;
    return Boolean(name) || Boolean(stillBoardRank(entry));
  });
  const bestId = String(r.best_user_pick?.prospect_id ?? "");
  const valueInsight = bestValueInsight(picks);

  const leagueTeams = [];
  const seenTeams = new Set();
  safeArray(draftOrder).forEach((slot) => {
    const key = String(slot?.team_id ?? "");
    if (!key || seenTeams.has(key)) return;
    seenTeams.add(key);
    leagueTeams.push(slot);
  });

  const avgRaw = r.user_avg_value_delta != null
    ? Number(r.user_avg_value_delta)
    : (picks.length ? picks.reduce((s, p) => s + (pubDelta(p) || 0), 0) / picks.length : 0);
  const avgText = Number.isFinite(avgRaw) ? (fmtSigned(avgRaw, 1) ?? "0.0") : null;
  const avgTone = avgRaw > 0 ? "pos" : avgRaw < 0 ? "neg" : "flat";
  const gradeScoreRaw = r.user_grade_score != null && r.user_grade_score !== "" ? Number(r.user_grade_score) : null;
  const scoreShown = useCountUp(gradeScoreRaw, true);
  const classRank = parseLeagueRank(r.user_class_rank);
  const pickCount = r.user_pick_count ?? picks.length;
  const stealCount = Number(r.user_steal_count ?? 0);
  const reachCount = Number(r.user_reach_count ?? 0);

  const franchisePick = selectFranchisePick(picks, bestId);
  const franchiseKey = franchisePick
    ? String(franchisePick.pick_id || franchisePick.overall_pick || franchisePick.prospect_id)
    : "";
  const restPicks = franchisePick
    ? picks.filter((p) => String(p.pick_id || p.overall_pick || p.prospect_id) !== franchiseKey)
    : picks;
  const compactPicks = sortBy === "pot"
    ? restPicks.slice().sort((a, b) => (recapNumeric(potVal(b)) ?? 0) - (recapNumeric(potVal(a)) ?? 0))
    : restPicks;

  const openPick = (p) => setProfilePick(enrichPickFromBoard(p, draft));
  const rightsCount = picks.filter((p) => p?.prospect_file_complete).length;
  const footerStatus = rightsCount > 0
    ? `${rightsCount} pick${rightsCount === 1 ? "" : "s"} signed to rights`
    : `${pickCount} pick${pickCount === 1 ? "" : "s"} on the card`;


  return (
    <div className={`drx tone-${capTone}`}>
      <header className="drx-hero">
        {teamLogo ? (
          <span className="drx-hero__mark" aria-hidden="true"><img src={teamLogo} alt="" /></span>
        ) : null}
        <div className="drx-grade-col">
          <div className="drx-grade" aria-label={`Draft grade ${grade}`}>
            <strong>{grade}</strong>
          </div>
          {gradeScoreRaw != null ? (
            <span className="drx-grade__score">{scoreShown != null ? `${scoreShown}/100` : "—/100"}</span>
          ) : null}
        </div>

        <div className="drx-hero__story">
          <p className="drx-kicker drx-kicker--team">
            <DrxLogo teamId={teamId} teamName={teamName} size="xs" />
            <span>{teamName || teamAbbrev(teamId, teamName)} · {seasonLabel(franchiseState)} Entry Draft · Final</span>
          </p>
          <h2 className="drx-headline">{buildGmHeadline(r, grade)}</h2>
          <DrxRankLadder rank={classRank} />
          <DrxCeilingMap picks={picks} />
        </div>

        <div className="drx-board">
          <div className={`drx-board__main tone-${avgTone}`}>
            <span>Avg value</span>
            <strong>{avgText || "—"}</strong>
            <em>{avgRaw >= 0 ? "Beat the board" : "Missed the board"}</em>
          </div>
          <div className="drx-board__row">
            <div>
              <strong>{pickCount}</strong>
              <span>Picks</span>
            </div>
            <div className={stealCount ? "tone-pos" : undefined}>
              <strong>{stealCount}</strong>
              <span>Steals</span>
            </div>
            <div className={reachCount ? "tone-neg" : undefined}>
              <strong>{reachCount}</strong>
              <span>Reaches</span>
            </div>
          </div>
        </div>

        <DrxRink posBreakdown={posBreakdown} needItems={needItems} />
      </header>

      <div className="drx-body">
        <section className="drx-class" aria-label="Your class">
          <div className="drx-head">
            <h3>Your class</h3>
            <span title={SCOUTED_NOTE}>{picks.length} picks</span>
            {picks.length > 2 ? (
              <div className="drx-sort" role="tablist" aria-label="Sort your class">
                <button type="button" role="tab" aria-selected={sortBy === "pick"} className={sortBy === "pick" ? "is-on" : undefined} onClick={() => setSortBy("pick")}>By pick</button>
                <button type="button" role="tab" aria-selected={sortBy === "pot"} className={sortBy === "pot" ? "is-on" : undefined} onClick={() => setSortBy("pot")}>By potential</button>
              </div>
            ) : null}
          </div>
          {picks.length ? (
            <>
              {franchisePick ? (
                <DrxFranchiseCard pick={franchisePick} onOpen={openPick} teamName={teamName} draftYear={draftYearLabel(draft)} />
              ) : null}
              <div className="drx-slips">
                {compactPicks.map((p, i) => (
                  <DrxSlip
                    key={p.pick_id || p.overall_pick || p.prospect_id || i}
                    pick={p}
                    index={i}
                    isBest={Boolean(bestId) && String(p.prospect_id) === bestId}
                    onOpen={openPick}
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="drx-empty-state">
              <h4>No picks on the card</h4>
              <p>The war room stayed quiet. You did not make a selection this draft.</p>
            </div>
          )}
        </section>

        <aside className="drx-rail" aria-label="League recap">
          {valueInsight ? <p className="drx-insight">{valueInsight}</p> : null}
          <DrxMoverCard title="Best steals" kind="steal" rows={steals} tone="pos" empty="The board stayed honest." userTeamId={userTeamId} onOpen={openPick} />
          <DrxMoverCard title="Biggest reaches" kind="reach" rows={reaches} tone="neg" empty="A disciplined night across the league." userTeamId={userTeamId} onOpen={openPick} />
          <DrxBoardCard rows={stillOnBoard} />
          {extraStory ? <p className="drx-insight is-story">{extraStory}</p> : null}
        </aside>
      </div>

      {leagueTeams.length ? (
        <section className="drx-league">
          <div className="drx-league__bar">
            <h4>Around the league</h4>
            <span className="drx-league__hint">{focusedTeamId ? "" : "Pick a team to see its class"}</span>
            <div className="drx-league__logos" role="tablist" aria-label="Teams">
              {leagueTeams.map((slot) => {
                const isUser = String(slot.team_id) === String(userTeamId);
                const focused = String(slot.team_id) === String(focusedTeamId);
                return (
                  <button
                    key={slot.team_id}
                    type="button"
                    role="tab"
                    aria-selected={focused}
                    className={`drx-league__team${isUser ? " is-user" : ""}${focused ? " is-focused" : ""}`}
                    title={slot.team_name || teamAbbrev(slot.team_id, slot.team_name)}
                    onClick={() => setFocusedTeamId(focused ? null : slot.team_id)}
                  >
                    <DrxLogo teamId={slot.team_id} teamName={slot.team_name} size="sm" />
                  </button>
                );
              })}
            </div>
          </div>
          {focusedTeamId ? (
            <DrxTeamBoard
              teamId={focusedTeamId}
              teamName={(leagueTeams.find((t) => String(t.team_id) === String(focusedTeamId)) || {}).team_name}
              completed={completed}
              onOpen={openPick}
              onClose={() => setFocusedTeamId(null)}
            />
          ) : null}
        </section>
      ) : null}

      <footer className="drx-footer">
        <span className="drx-footer__status">
          <DrxLogo teamId={teamId} teamName={teamName} size="xs" />
          {footerStatus}
        </span>
        <button type="button" className="nhlcal-advance-button drx-continue" onClick={onContinue}>
          Continue offseason <Icon name="next" size={12} />
        </button>
      </footer>

      {profilePick ? (
        <CompletedDossier pick={profilePick} draft={draft} onClose={() => setProfilePick(null)} />
      ) : null}
    </div>
  );
}

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
  const [boardOpen, setBoardOpen] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState(null);

  const simLock = useRef(false);
  const lastRoundRef = useRef(1);
  const userModalDismissedPickRef = useRef(null);
  const openUserAfterRevealRef = useRef(false);
  const draftHydratedRef = useRef(Boolean(rawDraft?.draft_started));
  const clockAutoPickPickRef = useRef(null);
  const clockAdvanceInFlightRef = useRef(false);
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
      const nextFiles = safeArray(rawDraft.completed_picks || rawDraft.draft_results).filter((p) => p?.prospect_file_complete).length;
      const prevFiles = safeArray(prev?.completed_picks || prev?.draft_results).filter((p) => p?.prospect_file_complete).length;
      if (nextFiles > prevFiles) return rawDraft;
      return prev;
    });
  }, [rawDraft]);

  const completed = safeArray(draft.completed_picks || draft.draft_results);
  const lastCompletedPick = useMemo(() => {
    let best = null;
    let max = -1;
    completed.forEach((p) => {
      const n = Number(p?.overall_pick);
      if (Number.isFinite(n) && n > max) {
        max = n;
        best = p;
      }
    });
    return best;
  }, [completed]);
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

  const runCpuPick = useCallback(async ({ reveal = true } = {}) => {
    if (simLock.current) return false;
    simLock.current = true;
    setLoadingOp("cpu");
    setError("");
    try {
      const res = await submitCpuDraftPick();
      applyResponse(res);
      if (reveal && res?.pick_result) {
        setCpuReveal(res.pick_result);
      }
      if (res?.draft?.is_user_pick) {
        userModalDismissedPickRef.current = null;
      }
      return true;
    } catch (e) {
      clockAutoPickPickRef.current = null;
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
      return false;
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

  const handleUserDraft = useCallback(async (prospect, { autoOnClock = false } = {}) => {
    const pick = prospect || (autoOnClock ? (resolveAvailableProspect(selectedAvailable, available) || available[0]) : null);
    if (!pick) {
      setError(autoOnClock ? "Pick clock expired — no prospect available to auto-select." : "Select a prospect to draft.");
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
    const pid = getId(pick);
    setLoadingOp("submit");
    setError("");
    try {
      const res = await submitDraftPick({
        player_id: pid || "",
        pick_round: currentPick?.round || currentRound,
        pick_overall: currentPick?.overall_pick || draft.overall_pick,
        request_id: `${draft?.draft_id || "draft"}:${currentPick?.pick_id || currentPick?.overall_pick || draft.overall_pick}:${pid || "auto"}`,
      });
      applyResponse(res);
      setUserModalOpen(false);
      setReportOpen(false);
      setBoardOpen(false);
      userModalDismissedPickRef.current = null;
      setSelectedAvailable(null);
    } catch (e) {
      if (autoOnClock) clockAutoPickPickRef.current = null;
      setError(e?.message || "Pick failed");
    } finally {
      simLock.current = false;
      setLoadingOp(null);
    }
  }, [
    applyResponse,
    available,
    currentPick,
    currentRound,
    draft.overall_pick,
    draft?.draft_id,
    draftDone,
    isUserPick,
    selectedAvailable,
  ]);

  const pickClockEnabled = stage === "live" && !draftDone && overallNow != null && overallNow !== "";
  const pickClockPaused = Boolean(cpuReveal) || tradePanelOpen;

  const onPickClockExpired = useCallback(() => {
    if (draftDone || !pickClockEnabled) return true;
    const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
    if (pickKey && clockAutoPickPickRef.current === pickKey) return true;
    if (simLock.current || loading || clockAdvanceInFlightRef.current) return false;

    if (isUserPick) {
      setTradePanelOpen(false);
      setCpuReveal(null);
      setUserModalOpen(false);
      setBoardOpen(false);
      setReportOpen(false);
      setConfirmDialog(null);

      const pick = resolveAvailableProspect(selectedAvailable, available) || available[0];
      if (!pick && !available.length) {
        setError("Pick clock expired — no prospect available to auto-select.");
        return true;
      }
      if (pickKey) clockAutoPickPickRef.current = pickKey;
      clockAdvanceInFlightRef.current = true;
      void handleUserDraft(pick, { autoOnClock: true }).finally(() => {
        clockAdvanceInFlightRef.current = false;
      });
      return false;
    }

    if (pickKey) clockAutoPickPickRef.current = pickKey;
    clockAdvanceInFlightRef.current = true;
    void runCpuPick({ reveal: false }).finally(() => {
      clockAdvanceInFlightRef.current = false;
    });
    return false;
  }, [
    draftDone,
    pickClockEnabled,
    isUserPick,
    loading,
    currentPick?.overall_pick,
    draft.overall_pick,
    selectedAvailable,
    available,
    handleUserDraft,
    runCpuPick,
  ]);

  useEffect(() => {
    clockAutoPickPickRef.current = null;
    clockAdvanceInFlightRef.current = false;
  }, [overallNow]);

  const {
    secondsLeft: pickSecondsLeft,
    callTimeout: callPickTimeout,
    canTimeout: canPickTimeout,
  } = useDraftPickClock(
    overallNow,
    draftDone,
    pickClockEnabled,
    isUserPick,
    pickClockPaused,
    onPickClockExpired,
  );

  const clockTeamLabel = teamAbbrev(draft.current_team_id, draft.current_team_name)
    || draft.current_team_name
    || "Team";

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



  const openDraftModal = useCallback(() => {
    userModalDismissedPickRef.current = null;
    setUserModalOpen(true);
  }, []);

  const promptDraftProspect = useCallback((prospect) => {
    if (!prospect || draftDone || !isUserPick || simLock.current) return;
    const name = getPlayerName(prospect);
    const pickLabel = formatPick(currentPick?.overall_pick || draft.overall_pick);
    // One "are you sure", from any entry point (board, dossier, report, pick card).
    // Every overlay closes first so nothing is left to dismiss after the pick.
    const restore = { modal: userModalOpen, report: reportOpen, board: boardOpen, selected: selectedAvailable };
    setUserModalOpen(false);
    setReportOpen(false);
    setBoardOpen(false);
    setSelectedAvailable(null);
    setConfirmDialog({
      title: "Are you sure?",
      message: `Select ${name} with the ${pickLabel} pick? This can't be undone.`,
      confirmLabel: `Draft ${name}`,
      onConfirm: () => {
        setConfirmDialog(null);
        handleUserDraft(prospect);
      },
      onCancel: () => {
        setConfirmDialog(null);
        setUserModalOpen(restore.modal);
        setReportOpen(restore.report);
        setBoardOpen(restore.board);
        setSelectedAvailable(restore.selected);
      },
    });
  }, [
    draftDone,
    isUserPick,
    currentPick?.overall_pick,
    draft.overall_pick,
    handleUserDraft,
    userModalOpen,
    reportOpen,
    boardOpen,
    selectedAvailable,
  ]);

  const classPool = [...available, ...safeArray(completed)];

  return (
    <ProspectClassContext.Provider value={classPool}>
    <section className={`${PREFIX}-root`}>
      {null}
      <ProspectVisualStyles />

      <CommandBar
        franchiseState={franchiseState}
        draft={draft}
        currentRound={currentRound}
        overallNow={overallNow}
        completedCount={completed.length}
        totalPicks={totalPicks}
        draftDone={draftDone}
        isUserPick={isUserPick}
        pickSecondsLeft={pickSecondsLeft}
        clockEnabled={pickClockEnabled}
        clockTeamLabel={clockTeamLabel}
        canPickTimeout={canPickTimeout}
        onPickTimeout={callPickTimeout}
        clockLoading={loading}
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
                setSelectedCompletedPick(enrichPickFromBoard(p, draft));
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
                          <span className={`${PREFIX}-onclock-via`}>
                            via {teamAbbrev(draft.via_team_id, draft.via_team_name)}
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <span aria-hidden="true" />
                  </div>

                  {!cpuReveal && lastCompletedPick ? (
                    <LastPickSpotlight
                      pick={lastCompletedPick}
                      onOpen={(p) => {
                        setSelectedCompletedPick(enrichPickFromBoard(p, draft));
                        setSelectedAvailable(null);
                      }}
                    />
                  ) : null}

                  <WarRoomDesk
                    available={available}
                    completed={completed}
                    draftOrder={draftOrder}
                    userTeamId={uid}
                    overallNow={overallNow}
                    clockTeamName={draft.current_team_name}
                    totalPicks={totalPicks}
                    completedCount={completed.length}
                    onOpenProspect={(p) => {
                      setSelectedAvailable(p);
                      setSelectedCompletedPick(null);
                    }}
                  />

                  <TeamHaulBoard
                    floor
                    draftOrder={draftOrder}
                    completed={completed}
                    userTeamId={uid}
                    clockTeamId={draft.current_team_id}
                    selectedPick={selectedCompletedPick}
                    onSelectPick={(p) => {
                      setSelectedCompletedPick(enrichPickFromBoard(p, draft));
                      setSelectedAvailable(null);
                    }}
                  />

                  {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
                </>
              ) : (
                <TeamHaulBoard
                  draftOrder={draftOrder}
                  completed={completed}
                  userTeamId={uid}
                  selectedPick={selectedCompletedPick}
                  onSelectPick={(p) => {
                    setSelectedCompletedPick(enrichPickFromBoard(p, draft));
                  }}
                />
              )}
            </section>
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
            onOpenBoard={() => setBoardOpen(true)}
          />
        </>
      ) : null}

      {stage === "recap" ? (
        <RecapShow
          draft={draft}
          completed={completed}
          userPicks={userPicks}
          draftOrder={draftOrder}
          userTeamId={uid}
          franchiseState={franchiseState}
          onContinue={onContinue}
        />
      ) : null}

      {reportOpen && selectedAvailable ? (
        <FullReportSheet
          prospect={selectedAvailable}
          isUserPick={isUserPick}
          loading={loading}
          onDraft={promptDraftProspect}
          onClose={() => setReportOpen(false)}
        />
      ) : null}

      <BoardPopup
        open={boardOpen && !draftDone}
        available={available}
        selectedId={selectedAvailable ? getId(selectedAvailable) : null}
        isUserPick={isUserPick}
        ignoreEscape={Boolean(selectedAvailable)}
        onDraftProspect={promptDraftProspect}
        onSelectProspect={(p) => {
          setSelectedAvailable(p);
          setSelectedCompletedPick(null);
          setReportOpen(false);
        }}
        onClose={() => {
          setBoardOpen(false);
          setSelectedAvailable(null);
        }}
      />

      {selectedAvailable && boardOpen ? (
        <DossierPopup
          prospect={selectedAvailable}
          currentPickOverall={overallNow}
          stacked
          isUserPick={isUserPick}
          loading={loading}
          onDraft={promptDraftProspect}
          onClose={() => setSelectedAvailable(null)}
        />
      ) : null}

      {selectedCompletedPick ? (
        <CompletedDossier
          pick={selectedCompletedPick}
          draft={draft}
          revision={franchiseState?.prospect_revision}
          extra={selectedPickReaction ? (
            <section className={`${PREFIX}-sec`} style={{ padding: "0 4px 8px" }}>
              <SectionHead icon="flame" title="Floor reaction" />
              <ReactionRow tweet={selectedPickReaction} />
            </section>
          ) : null}
          onClose={() => setSelectedCompletedPick(null)}
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
        onCancel={() => (confirmDialog?.onCancel ? confirmDialog.onCancel() : setConfirmDialog(null))}
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
        onDraft={promptDraftProspect}
        onTradeDown={() => setTradePanelOpen(true)}
        onClose={() => {
          const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
          if (pickKey) userModalDismissedPickRef.current = pickKey;
          setUserModalOpen(false);
        }}
      />
    </section>
    </ProspectClassContext.Provider>
  );
}
