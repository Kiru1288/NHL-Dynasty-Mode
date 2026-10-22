/** Thin client for /api/franchise/identity/* and /api/franchise/legacy/* (services/legacy_hall.py). */
import { api } from "../../services/api";

const get = (url, params) => api.get(url, { params, timeout: 20000 }).then((r) => r.data);
const post = (url, body) => api.post(url, body, { timeout: 20000 }).then((r) => r.data);

export const legacyApi = {
  identity: (teamId) => get("/api/franchise/identity", teamId ? { team_id: teamId } : undefined),
  validate: (identity) => post("/api/franchise/identity/validate", { identity }),
  rebrand: (identity, reason, requestId) => post("/api/franchise/identity/rebrand", { identity, reason, request_id: requestId }),
  savePreset: (name, uniform) => post("/api/franchise/identity/preset", { name, uniform }),
  setPhilosophy: (philosophy) => post("/api/franchise/identity/philosophy", { philosophy }),
  captaincy: (teamId) => get("/api/franchise/identity/captaincy", teamId ? { team_id: teamId } : undefined),
  appoint: (captainId, alternateIds) => post("/api/franchise/identity/captaincy", { captain_id: captainId, alternate_ids: alternateIds }),
  franchise: (teamId) => get("/api/franchise/legacy/franchise", teamId ? { team_id: teamId } : undefined),
  league: () => get("/api/franchise/legacy/league"),
  season: (season) => get(`/api/franchise/legacy/season/${season}`),
  player: (pid) => get(`/api/franchise/legacy/player/${encodeURIComponent(pid)}`),
  leaders: (params) => get("/api/franchise/legacy/leaders", params),
  drafts: (params) => get("/api/franchise/legacy/drafts", params),
  trades: (params) => get("/api/franchise/legacy/trades", params),
  rivalries: (teamId) => get("/api/franchise/legacy/rivalries", teamId ? { team_id: teamId } : undefined),
  timeline: (params) => get("/api/franchise/legacy/timeline", params),
  trivia: (teamId) => get("/api/franchise/legacy/trivia", teamId ? { team_id: teamId } : undefined),
  retireNumber: (playerId, number) => post("/api/franchise/legacy/retire-number", { player_id: playerId, number }),
};

export function apiError(e) {
  const d = e?.response?.data?.detail;
  if (d && Array.isArray(d.errors)) return d.errors.join(" · ");
  if (typeof d === "string") return d;
  return e?.message || "Request failed";
}

/** One-shot hand-off between Identity and Legacy Hall (e.g. "open this player's dossier"). */
let pendingFocus = null;
export function setLegacyFocus(focus) {
  pendingFocus = focus;
}
export function takeLegacyFocus() {
  const f = pendingFocus;
  pendingFocus = null;
  return f;
}

export function seasonLabel(y) {
  const n = Number(y);
  return Number.isFinite(n) ? `${n}-${String(n + 1).slice(-2)}` : "";
}
