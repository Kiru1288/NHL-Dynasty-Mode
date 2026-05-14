import { api, baseURL } from "./api";

/** Avoid infinite spinner if backend never responds (axios default timeout is 0 = none). */
const FRANCHISE_START_TIMEOUT_MS = 900_000;

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
  const { data } = await api.post("/api/franchise/advance", {
    mode,
    count,
    auto_resolve,
  });
  return data;
}

export async function getFranchiseState() {
  const { data } = await api.get("/api/franchise/state");
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
