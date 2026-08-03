import axios from "axios";
import { record as perfRecord } from "./perfProfiler";

export const baseURL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export const SESSION_STORAGE_KEY = "nhl_franchise_session_id";
export const API_INSTANCE_STORAGE_KEY = "nhl_franchise_api_instance_id";
export const API_CODE_REVISION_STORAGE_KEY = "nhl_franchise_api_code_revision";

/** Frontend expects these /api/health code.features — missing means stale uvicorn. */
export const REQUIRED_BACKEND_FEATURES = [
  "lineup_persistence",
  "lines_route",
  "chemistry_profile_contract",
  "saved_line_deployment",
  "stale_save_invalidation",
];

/** Prefixed keys that must never bleed across sessions / backend restarts. */
export const FRANCHISE_LINEUP_STORAGE_PREFIXES = [
  "nhl_franchise_even_strength_lines_",
  "nhl_franchise_power_play_lines_",
  "nhl_franchise_penalty_kill_lines_",
];

export const api = axios.create({
  baseURL,
  timeout: 0,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  config.metadata = { ...(config.metadata || {}), start: performance.now() };
  const url = config.url || "";
  if (url.includes("/franchise/start") || url.includes("/franchise/teams")) {
    return config;
  }
  const sid = localStorage.getItem(SESSION_STORAGE_KEY);
  if (sid) {
    config.headers = config.headers || {};
    config.headers["X-Franchise-Session"] = sid;
  }
  return config;
});

export function isNetworkError(err) {
  if (err.response) return false;
  return (
    err.code === "ECONNABORTED" ||
    err.code === "ERR_NETWORK" ||
    err.code === "ECONNRESET" ||
    (err.message && err.message.toLowerCase().includes("network error"))
  );
}

export function isTimeoutError(err) {
  if (err.response) return false;
  const msg = String(err?.message || "").toLowerCase();
  return err.code === "ECONNABORTED" || msg.includes("timeout");
}

/** Remove every client cache that can make UI look like an old save after backend changes. */
export function clearFranchiseClientCaches() {
  try {
    localStorage.removeItem(SESSION_STORAGE_KEY);
    const doomed = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (!key) continue;
      if (
        FRANCHISE_LINEUP_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix)) ||
        key === "nhl_franchise_even_strength_lines_v1" ||
        key === "nhl_franchise_power_play_lines_v1" ||
        key === "nhl_franchise_penalty_kill_lines_v1" ||
        key.startsWith("nhl_franchise_even_strength_lines_v2:") ||
        key.startsWith("nhl_franchise_power_play_lines_v2:") ||
        key.startsWith("nhl_franchise_penalty_kill_lines_v2:")
      ) {
        doomed.push(key);
      }
    }
    doomed.forEach((key) => localStorage.removeItem(key));
  } catch {
    // ignore quota / private mode
  }
}

export function clearFranchiseSession() {
  clearFranchiseClientCaches();
}

export function lineupStorageKey(kind, sessionId) {
  const sid = String(sessionId || getFranchiseSessionId() || "anon").trim() || "anon";
  return `nhl_franchise_${kind}_lines_v2:${sid}`;
}

export function readSessionLineupCache(kind, sessionId) {
  try {
    const key = lineupStorageKey(kind, sessionId);
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function writeSessionLineupCache(kind, value, sessionId) {
  try {
    const key = lineupStorageKey(kind, sessionId);
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore
  }
}

export async function resetFranchiseServerSessions() {
  try {
    await api.post("/api/franchise/reset", null, {
      timeout: 8000,
      validateStatus: (status) => status >= 200 && status < 500,
    });
  } catch {
    // The local session still needs to be cleared even if the backend is offline.
  }
}

export function isExpiredFranchiseSessionError(err) {
  const status = err?.response?.status;
  if (status !== 404 && status !== 400) return false;
  const detail = String(err?.response?.data?.detail || err?.response?.data?.message || "");
  if (/unknown or expired franchise session/i.test(detail)) return true;
  if (status === 400 && /missing x-franchise-session/i.test(detail)) return true;
  const url = String(err?.config?.url || "");
  return status === 404 && url.includes("/api/franchise/") && !url.includes("/franchise/teams");
}

function rememberBackendIdentity(instanceId, codeRevision) {
  if (instanceId) localStorage.setItem(API_INSTANCE_STORAGE_KEY, instanceId);
  if (codeRevision) localStorage.setItem(API_CODE_REVISION_STORAGE_KEY, codeRevision);
}

/**
 * Drop saved session id + client lineup caches when the backend process restarted
 * or its live code fingerprint changed (uvicorn --reload / new python process).
 */
export async function syncFranchiseSessionWithBackend() {
  try {
    const { data } = await api.get("/api/health", { timeout: 8000 });
    const instanceId = String(data?.instance_id || "").trim();
    const codeRevision = String(data?.code_revision || data?.code?.revision || "").trim();
    const features = data?.code?.features || {};
    const prevInstance = localStorage.getItem(API_INSTANCE_STORAGE_KEY);
    const prevRevision = localStorage.getItem(API_CODE_REVISION_STORAGE_KEY);
    const missingFeatures = REQUIRED_BACKEND_FEATURES.filter((f) => !features?.[f]);
    // Old uvicorn still serving yesterday's process has no revision / new features.
    const staleBackend = !codeRevision || missingFeatures.length > 0;

    const sid = getFranchiseSessionId();
    const backendChanged =
      staleBackend ||
      (instanceId && prevInstance && prevInstance !== instanceId) ||
      (codeRevision && prevRevision && prevRevision !== codeRevision);

    if (backendChanged) {
      clearFranchiseClientCaches();
      // Drop server-side in-memory saves too — otherwise /state can still revive them.
      await resetFranchiseServerSessions();
      rememberBackendIdentity(instanceId, codeRevision);
      return true;
    }

    rememberBackendIdentity(instanceId, codeRevision);

    if (!sid) return false;

    const probe = await api.get("/api/franchise/state", {
      timeout: 8000,
      validateStatus: (status) => status === 200 || status === 404 || status === 400,
    });
    if (probe.status === 404 || probe.status === 400) {
      clearFranchiseClientCaches();
      rememberBackendIdentity(instanceId, codeRevision);
      return true;
    }

    return false;
  } catch {
    return false;
  }
}

function noteBackendIdentityFromHeaders(headers) {
  if (!headers) return;
  const instanceId = String(headers["x-api-instance-id"] || "").trim();
  const codeRevision = String(headers["x-api-code-revision"] || "").trim();
  if (!instanceId && !codeRevision) return;

  const prevInstance = localStorage.getItem(API_INSTANCE_STORAGE_KEY);
  const prevRevision = localStorage.getItem(API_CODE_REVISION_STORAGE_KEY);
  const changed =
    (instanceId && prevInstance && prevInstance !== instanceId) ||
    (codeRevision && prevRevision && prevRevision !== codeRevision);

  if (changed) {
    clearFranchiseClientCaches();
    rememberBackendIdentity(instanceId, codeRevision);
    // Fire-and-forget: purge obsolete in-memory saves on the server.
    resetFranchiseServerSessions();
    if (typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent("nhl-franchise-backend-changed", {
          detail: { instanceId, codeRevision },
        })
      );
    }
    return;
  }

  rememberBackendIdentity(instanceId, codeRevision);
}

function _perfNoteAxios(config, error) {
  try {
    const start = config?.metadata?.start;
    if (start == null) return;
    const ms = performance.now() - start;
    const url = String(config?.url || config?.baseURL || "unknown");
    const method = String(config?.method || "get").toUpperCase();
    const bytes = error
      ? 0
      : Number(
          (typeof config?.__responseSize === "number" && config.__responseSize) ||
            (config?.__responseData && JSON.stringify(config.__responseData).length) ||
            0
        );
    perfRecord(`api.${method} ${url.split("?")[0]}`, ms, {
      status: error?.response?.status || config?.__status,
      bytes: bytes || undefined,
      error: error ? String(error.code || error.message || "error") : undefined,
    });
  } catch {
    // never break API on profiler failure
  }
}

api.interceptors.response.use(
  (response) => {
    noteBackendIdentityFromHeaders(response?.headers);
    try {
      if (response?.config) {
        response.config.__status = response.status;
        // Avoid double-stringify cost on huge payloads — sample Content-Length when present.
        const cl = response.headers?.["content-length"] || response.headers?.["Content-Length"];
        if (cl) response.config.__responseSize = Number(cl);
      }
    } catch {
      // ignore
    }
    _perfNoteAxios(response?.config);
    return response;
  },
  (error) => {
    noteBackendIdentityFromHeaders(error?.response?.headers);
    _perfNoteAxios(error?.config, error);
    if (isExpiredFranchiseSessionError(error)) {
      clearFranchiseClientCaches();
    }
    return Promise.reject(error);
  }
);

export function setFranchiseSessionId(id) {
  localStorage.setItem(SESSION_STORAGE_KEY, id);
}

export function getFranchiseSessionId() {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

/** User-facing message for failed franchise API calls */
export function formatFranchiseApiError(err) {
  if (isTimeoutError(err)) {
    return (
      "The franchise API timed out on that request. The server is often still running — " +
      "retry the action, or use Resume Offseason Timeline from Hub if you're past the Cup."
    );
  }
  if (isNetworkError(err)) {
    return (
      `Lost connection to the franchise API (${baseURL}). ` +
      "If the backend was just reloading code, refresh and try again. " +
      "If health is down, start backend/start_api.ps1."
    );
  }
  if (err.response?.status === 404) {
    if (isExpiredFranchiseSessionError(err)) {
      // Expired sessions silently return the user to setup — no banner text.
      return "";
    }
    return (
      `404 from ${baseURL} — often an old API without /api/franchise. ` +
      `Stop uvicorn/Python on port 8000, then run backend/start_api.ps1. ` +
      `Check ${baseURL}/api/health for mode: interactive_franchise.`
    );
  }
  const data = err.response?.data;
  const d = data?.detail ?? data?.message;
  if (typeof d === "string") return d;
  if (d) return JSON.stringify(d);
  return err.message || String(err);
}
