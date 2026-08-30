import { api, baseURL, getFranchiseSessionId, isNetworkError, isTimeoutError } from "./api";

/** Avoid infinite spinner if backend never responds (axios default timeout is 0 = none). */
const FRANCHISE_START_TIMEOUT_MS = 900_000;
let inflightFranchiseStatePromise = null;
let inflightFranchiseStateSessionId = null;
let inflightFranchiseStateHeavyPromise = null;
let inflightFranchiseStateHeavyKey = null;
let inflightFranchiseNarrativePromise = null;

const prospectProfileCache = new Map();
const inflightProspectProfiles = new Map();

export function resetFranchiseStateCache() {
  inflightFranchiseStatePromise = null;
  inflightFranchiseStateSessionId = null;
  prospectProfileCache.clear();
  inflightProspectProfiles.clear();
}

export async function listTeams() {
  // SimEngine boot on this route can take minutes; cap wait so UI never hangs forever.
  const { data } = await api.get("/api/franchise/teams", { timeout: 300000 });
  return data.teams || [];
}

export async function startFranchise(payload) {
  try {
    const res = await api.post("/api/franchise/start", payload, {
      timeout: FRANCHISE_START_TIMEOUT_MS,
      validateStatus: () => true,
    });
    const data = res.data;
    const status = res.status;
    if (status < 200 || status >= 300) {
      const msg =
        (data && typeof data.message === "string" && data.message) ||
        (data && typeof data.detail === "string" && data.detail) ||
        `Begin Franchise failed: HTTP ${status}`;
      console.error("[startFranchise] HTTP error", status, data);
      throw new Error(msg);
    }
    if (data && data.ok === false) {
      console.error("[startFranchise] backend returned ok:false", data);
      throw new Error(data.message || "Begin Franchise failed");
    }
    return data;
  } catch (e) {
    if (e?.code === "ECONNABORTED") {
      console.error("[startFranchise] timeout", baseURL);
      throw new Error(
        `Begin Franchise timed out after ${Math.round(FRANCHISE_START_TIMEOUT_MS / 60000)} minutes. Check the backend terminal for progress.`
      );
    }
    console.error("[startFranchise] request failed", e);
    throw e;
  }
}

/** @param {{ mode?: string, count?: number, auto_resolve?: boolean }} [opts] */
export async function advanceFranchise(opts = {}) {
  const { mode = "day", count = 1, auto_resolve = true } = opts;
  const m = String(mode || "day").toLowerCase();
  const heavy =
    m === "season" ||
    m === "days" ||
    m === "games" ||
    (m === "day" && Number(count) > 1) ||
    Number(count) > 1;
  try {
    const { data } = await api.post(
      "/api/franchise/advance",
      {
        mode,
        count,
        auto_resolve,
      },
      { timeout: heavy ? 600000 : 180000 }
    );
    return data;
  } catch (e) {
    if (isTimeoutError(e)) {
      const err = new Error(
        heavy
          ? "Season/bulk advance timed out. The sim may still have finished — refresh Hub, then retry or open Playoffs if the season ended."
          : "Day advance timed out. Retry Advance Day — the backend may still be processing."
      );
      err.code = "ECONNABORTED";
      err.cause = e;
      throw err;
    }
    if (isNetworkError(e)) {
      const err = new Error(
        "Connection dropped during advance (often a backend reload while simming). Check that the API is up, then retry Advance — avoid saving backend files mid-sim."
      );
      err.code = "ERR_NETWORK";
      err.cause = e;
      throw err;
    }
    throw e;
  }
}

export async function enterPlayoffs() {
  const { data } = await api.post("/api/franchise/playoffs/enter");
  return data;
}

export async function playoffAction(action, body = {}) {
  const heavy = ["sim_rest", "finish", "complete", "sim_round"].includes(
    String(action || "").toLowerCase()
  );
  const { data } = await api.post(
    "/api/franchise/playoffs/action",
    {
      action,
      ...body,
    },
    { timeout: heavy ? 300000 : 180000 }
  );
  return data;
}

/** @param {{ target?: string }} [payload] */
export async function advanceSeasonPhase(payload = {}) {
  const { data } = await api.post("/api/franchise/season/advance-phase", payload);
  return data;
}

/** @param {{ stage?: string }} [payload] */
export async function enterOffseasonStage(payload = {}) {
  return continueOffseason(payload);
}

export async function continueOffseason(payload = {}) {
  const { data } = await api.post("/api/franchise/offseason/continue", payload, {
    timeout: 300000,
  });
  return data;
}

/** @param {{ stage?: string }} [payload] */
export async function reopenOffseasonStage(payload = {}) {
  const { data } = await api.post("/api/franchise/offseason/reopen-stage", payload, {
    timeout: 120000,
  });
  return data;
}

export async function getFreeAgencyDesk() {
  const { data } = await api.get("/api/franchise/free-agency/desk", {
    timeout: 120000,
  });
  return data;
}

export async function generateNextSeason() {
  const { data } = await api.post("/api/franchise/next-season/generate");
  return data;
}

export async function getLeagueOperations() {
  const { data } = await api.get("/api/franchise/league-operations");
  return data;
}

export async function getFranchiseState({ crisisTick = false } = {}) {
  const sid = getFranchiseSessionId();
  if (!sid) {
    resetFranchiseStateCache();
    throw new Error("No franchise session");
  }
  if (
    !crisisTick &&
    inflightFranchiseStatePromise &&
    inflightFranchiseStateSessionId === sid
  ) {
    return inflightFranchiseStatePromise;
  }
  inflightFranchiseStateSessionId = sid;
  const params = crisisTick ? { crisis_tick: 1 } : undefined;
  inflightFranchiseStatePromise = api
    .get("/api/franchise/state", { params })
    .then((res) => res.data)
    .finally(() => {
      if (inflightFranchiseStateSessionId === sid) {
        resetFranchiseStateCache();
      }
    });
  return inflightFranchiseStatePromise;
}

export async function getFranchiseCrisis() {
  const { data } = await api.get("/api/franchise/crisis");
  return data;
}

export async function getFranchiseNarrative() {
  const sid = getFranchiseSessionId();
  if (!sid) throw new Error("No franchise session");
  if (inflightFranchiseNarrativePromise) {
    return inflightFranchiseNarrativePromise;
  }
  inflightFranchiseNarrativePromise = api
    .get("/api/franchise/narrative")
    .then((res) => res.data)
    .finally(() => {
      inflightFranchiseNarrativePromise = null;
    });
  return inflightFranchiseNarrativePromise;
}

export async function getStatsCentral() {
  const { data } = await api.get("/api/franchise/stats-central");
  return data;
}

export async function getDraftClassDetail() {
  const { data } = await api.get("/api/franchise/draft-class/detail");
  return data;
}

export async function getDraftProspectProfile(prospectId, options = {}) {
  const id = String(prospectId || "").trim();
  if (!id) return null;
  const rev = options.prospectRevision;
  const cached = prospectProfileCache.get(id);
  if (cached && (!rev || cached.rev === rev)) {
    return cached.profile;
  }
  const inflight = inflightProspectProfiles.get(id);
  if (inflight) return inflight;
  const promise = api
    .get(`/api/franchise/draft-class/prospect/${encodeURIComponent(id)}/profile`, { timeout: 60000 })
    .then((res) => {
      const profile = res.data?.profile || null;
      if (profile) {
        prospectProfileCache.set(id, {
          rev: rev ?? profile.prospect_revision ?? null,
          profile,
        });
      }
      return profile;
    })
    .finally(() => {
      inflightProspectProfiles.delete(id);
    });
  inflightProspectProfiles.set(id, promise);
  return promise;
}

export function prefetchDraftProspectProfile(prospectId, options = {}) {
  return getDraftProspectProfile(prospectId, options).catch(() => null);
}

export async function getFranchiseStateHeavy(options = {}) {
  const params = {
    include_roster_browser: options.includeRosterBrowser !== false,
    include_draft_class_rankings: options.includeDraftClassRankings !== false,
    include_draft_class_hud: options.includeDraftClassHud !== false,
  };
  if (options.includeNhlCalendarFull) {
    params.include_nhl_calendar_full = true;
  }
  const cacheKey = JSON.stringify(params);
  if (inflightFranchiseStateHeavyPromise && inflightFranchiseStateHeavyKey === cacheKey) {
    return inflightFranchiseStateHeavyPromise;
  }
  inflightFranchiseStateHeavyKey = cacheKey;
  inflightFranchiseStateHeavyPromise = api
    .get("/api/franchise/state/heavy", { params, timeout: 180000 })
    .then((res) => res.data)
    .finally(() => {
      if (inflightFranchiseStateHeavyKey === cacheKey) {
        inflightFranchiseStateHeavyPromise = null;
        inflightFranchiseStateHeavyKey = null;
      }
    });
  return inflightFranchiseStateHeavyPromise;
}

export async function getContractOffice() {
  const { data } = await api.get("/api/franchise/contract-office");
  return data;
}

export async function getFreeAgentDetail(playerId) {
  const { data } = await api.get(`/api/franchise/free-agents/${encodeURIComponent(playerId)}`);
  return data;
}

async function postContractAction(path, payload) {
  const { data } = await api.post(`/api/franchise/contracts/${path}`, payload);
  return data;
}

export function offerContract(payload) {
  return postContractAction("offer", payload);
}

export function reSignContract(payload) {
  return postContractAction("re-sign", payload);
}

/** Preview player response without persisting a deal. */
export function evaluateContractOffer(payload) {
  return postContractAction("re-sign", { ...payload, evaluate_only: true });
}

export function signFreeAgent(payload) {
  return postContractAction("sign-free-agent", payload);
}

export function qualifyRfa(payload) {
  return postContractAction("qualify-rfa", payload);
}

export function releaseRfaRights(payload) {
  return postContractAction("release-rights", payload);
}

export function buyoutContract(payload) {
  return postContractAction("buyout", payload);
}

export function waiveContract(payload) {
  return postContractAction("waive", payload);
}

export function buryContract(payload) {
  return postContractAction("bury", payload);
}

export async function getRosterMoves(playerId) {
  const { data } = await api.get("/api/franchise/roster/moves", {
    params: { player_id: playerId },
  });
  return data;
}

export async function moveRosterPlayer(payload) {
  const { data } = await api.post("/api/franchise/roster/move", payload || {});
  return data;
}

export function signElcContract(payload) {
  return postContractAction("sign-elc", payload);
}

/** Preview structured ELC offer (acceptance, cap, slots) — no mutation. */
export function previewElcOffer(payload) {
  return postContractAction("preview-elc-offer", payload);
}

/** Submit structured ELC offer (persists exact terms on accept). */
export function submitElcOffer(payload) {
  return postContractAction("submit-elc-offer", payload);
}

/** Prospect Rights stage decision (ELC, keep path, expire, etc.). */
export function prospectRightsDecision(payload) {
  return postContractAction("prospect-rights", payload);
}

export function evaluateElcSigning(payload) {
  return postContractAction("evaluate-elc", payload);
}

export function submitOfferSheet(payload) {
  return postContractAction("offer-sheet", payload);
}

export function matchOfferSheet(payload) {
  return postContractAction("match-offer-sheet", payload);
}

export function declineOfferSheet(payload) {
  return postContractAction("decline-offer-sheet", payload);
}

export function fileArbitration(payload) {
  return postContractAction("arbitration-file", payload);
}

export function settleArbitration(payload) {
  return postContractAction("arbitration-settle", payload);
}

export async function advanceFreeAgencyDay(days = 1) {
  const { data } = await api.post("/api/franchise/free-agency/advance-day", {
    days: Math.max(1, Number(days) || 1),
  });
  return data;
}

/** Advance exclusive own-FA negotiating window and resolve pending offers. */
export async function advanceContractNegotiationDay(days = 1) {
  const { data } = await api.post("/api/franchise/contracts/advance-day", {
    days: Math.max(1, Number(days) || 1),
  });
  return data;
}

export async function getFranchiseChemistry() {
  const { data } = await api.get("/api/franchise/chemistry");
  return data;
}

export async function getFranchiseLines() {
  const { data } = await api.get("/api/franchise/lines");
  return data;
}

/** @param {{unit_type: string, lines: any}} payload */
export async function saveFranchiseLines(payload) {
  const { data } = await api.post("/api/franchise/lines", payload);
  return data;
}

/** @param {string} gameId */
export async function getFranchiseGame(gameId) {
  const { data } = await api.get(`/api/franchise/game/${encodeURIComponent(gameId)}`);
  return data.game;
}

export async function advanceDay() {
  return advanceFranchise({ mode: "day", count: 1, auto_resolve: true });
}

export async function submitDecision(decisionId, choiceId) {
  const { data } = await api.post("/api/franchise/decision", {
    decision_id: decisionId,
    choice_id: choiceId,
  });
  return data;
}

export async function submitStorylineChoice(storylineId, choiceId) {
  const { data } = await api.post("/api/franchise/storyline/choice", {
    storyline_id: storylineId,
    choice_id: choiceId,
  });
  return data;
}

/** Resolve player-initiated universe meeting. */
export async function resolvePlayerMeeting(interactionId, choiceId) {
  const { data } = await api.post("/api/franchise/player-meetings/resolve", {
    interaction_id: interactionId,
    choice_id: choiceId,
  });
  return data;
}

/** Start GM-initiated player meeting. */
export async function startPlayerMeeting(playerId, interactionType) {
  const { data } = await api.post("/api/franchise/player-meetings/start", {
    player_id: playerId,
    interaction_type: interactionType,
  });
  return data;
}

/** Resolve in-progress GM-initiated meeting. */
export async function advancePlayerMeeting(meetingId, choiceId) {
  const { data } = await api.post("/api/franchise/player-meetings/resolve", {
    meeting_id: meetingId,
    choice_id: choiceId,
  });
  return data;
}

/** Fetch player meeting detail + available interactions. */
export async function getPlayerMeetingDetail(playerId) {
  const { data } = await api.get(`/api/franchise/player-meetings/player/${encodeURIComponent(playerId)}`);
  return data;
}

/** @param {string[]} ids */
export async function dismissFranchisePopups(ids) {
  const { data } = await api.post("/api/franchise/popup/dismiss", { ids: ids || [] });
  return data;
}

/** @param {{ assets_by_team: Record<string, Array<{type:string,id:string,team:string,retained?:number}>> }} payload */
export async function submitTradePackage(payload) {
  const { data } = await api.post("/api/franchise/trade", payload || { assets_by_team: {} });
  return data;
}

/** Evaluate a trade package without executing (backend-authoritative). */
export async function evaluateTradePackage(payload) {
  const { data } = await api.post("/api/franchise/trade/evaluate", payload || { assets_by_team: {} });
  return data;
}

/** Ask an NTC/M-NTC player to waive for a destination team. */
export async function requestNtcWaive(payload) {
  const { data } = await api.post("/api/franchise/trade/ntc-waive", payload || {});
  return data;
}

export async function getTradeAssets(options = {}) {
  const force = options?.force !== false;
  const { data } = await api.get("/api/franchise/trade/assets", {
    params: { force: force ? 1 : 0, v: 3 },
  });
  return data;
}

export async function getTradeHistory(params = {}) {
  const { data } = await api.get("/api/franchise/trade/history", { params });
  return data;
}

export async function getTradeMarket() {
  const { data } = await api.get("/api/franchise/trade/market");
  return data;
}

export async function getEntryDraftState() {
  const { data } = await api.get("/api/franchise/entry-draft/state");
  return data;
}

export async function startEntryDraft() {
  const { data } = await api.post("/api/franchise/entry-draft/start");
  return data;
}

/** @param {{ player_id: string, pick_round?: number, pick_overall?: number }} body */
export async function submitDraftPick(body) {
  const { data } = await api.post("/api/franchise/draft/pick", body);
  return data;
}

export async function submitCpuDraftPick() {
  const { data } = await api.post("/api/franchise/entry-draft/cpu-pick");
  return data;
}

export async function simEntryDraftRound() {
  const { data } = await api.post("/api/franchise/entry-draft/sim-round");
  return data;
}

export async function simEntryDraftToUserPick() {
  const { data } = await api.post("/api/franchise/entry-draft/sim-to-user-pick");
  return data;
}

export async function completeEntryDraft() {
  const { data } = await api.post("/api/franchise/entry-draft/complete");
  return data;
}

export async function acceptEntryDraftTrade(offer) {
  const { data } = await api.post("/api/franchise/entry-draft/accept-trade", { offer: offer || {} });
  return data;
}

export async function getEntryDraftResults() {
  const { data } = await api.get("/api/franchise/entry-draft/results");
  return data;
}

export async function getDraftCombineState() {
  const { data } = await api.get("/api/franchise/draft-combine/state");
  return data;
}

/** @param {{ prospect_id: string, meeting_type?: 'interview'|'dinner' }} body */
export async function submitCombineMeeting(body) {
  const { data } = await api.post("/api/franchise/draft-combine/meeting", body);
  return data;
}

export async function getSocialFeed(sessionId) {
  const sid = sessionId || getFranchiseSessionId();
  const { data } = await api.get(`/api/franchise/${encodeURIComponent(sid)}/social-feed`);
  return data;
}

export async function getBurnerState(sessionId) {
  const sid = sessionId || getFranchiseSessionId();
  const { data } = await api.get(`/api/franchise/${encodeURIComponent(sid)}/burner`);
  return data;
}

export async function previewBurnerPost(text, marketKey, sessionId) {
  const sid = sessionId || getFranchiseSessionId();
  const { data } = await api.post(`/api/franchise/${encodeURIComponent(sid)}/burner/preview`, {
    text,
    market_key: marketKey || undefined,
  });
  return data;
}

export async function postBurnerMessage(text, marketKey, sessionId) {
  const sid = sessionId || getFranchiseSessionId();
  const { data } = await api.post(`/api/franchise/${encodeURIComponent(sid)}/burner/post`, {
    text,
    market_key: marketKey || undefined,
  });
  return data;
}

