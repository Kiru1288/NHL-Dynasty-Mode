import axios from "axios";

const baseURL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export const SESSION_STORAGE_KEY = "nhl_franchise_session_id";

export const api = axios.create({
  baseURL,
  timeout: 0,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
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
    (err.message && err.message.toLowerCase().includes("network"))
  );
}

export function clearFranchiseSession() {
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

export function setFranchiseSessionId(id) {
  localStorage.setItem(SESSION_STORAGE_KEY, id);
}

export function getFranchiseSessionId() {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

/** User-facing message for failed franchise API calls */
export function formatFranchiseApiError(err) {
  if (isNetworkError(err)) {
    return `API offline (${baseURL}). Start the backend (backend/start_api.ps1 or uvicorn).`;
  }
  if (err.response?.status === 404) {
    return (
      `404 from ${baseURL} — often an old API without /api/franchise. ` +
      `Stop uvicorn/Python on port 8000, then run backend/start_api.ps1. ` +
      `Check ${baseURL}/api/health for mode: interactive_franchise.`
    );
  }
  const d = err.response?.data?.detail;
  if (typeof d === "string") return d;
  if (d) return JSON.stringify(d);
  return err.message || String(err);
}
