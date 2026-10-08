/**
 * Team identity data — one fetch shared by every screen.
 * Backend: GET /api/franchise/team-identity (services/team_identity_service.py).
 * Identities move slowly (season momentum), so a short TTL cache is plenty.
 */
import { api } from "../../services/api";

const TTL_MS = 60 * 1000;
let cache = { at: 0, byKey: null, promise: null };
const listeners = new Set();

function indexTeams(teams) {
  const byKey = new Map();
  (Array.isArray(teams) ? teams : []).forEach((t) => {
    if (!t) return;
    if (t.team_id != null) byKey.set(String(t.team_id), t);
    if (t.abbreviation) byKey.set(String(t.abbreviation).toUpperCase(), t);
    if (t.team_name) byKey.set(String(t.team_name).toLowerCase(), t);
  });
  return byKey;
}

export function loadTeamIdentities({ force = false } = {}) {
  const fresh = cache.byKey && Date.now() - cache.at < TTL_MS;
  if (!force && fresh) return Promise.resolve(cache.byKey);
  if (cache.promise) return cache.promise;
  cache.promise = api
    .get("/api/franchise/team-identity", { timeout: 20000 })
    .then(({ data }) => {
      cache = { at: Date.now(), byKey: indexTeams(data?.teams), promise: null };
      listeners.forEach((fn) => fn(cache.byKey));
      return cache.byKey;
    })
    .catch(() => {
      cache.promise = null;
      return cache.byKey || new Map();
    });
  return cache.promise;
}

export function invalidateTeamIdentities() {
  cache.at = 0;
}

export function peekTeamIdentities() {
  return cache.byKey;
}

export function subscribeTeamIdentities(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function findTeamIdentity(byKey, { teamId, abbr, name } = {}) {
  if (!byKey) return null;
  return (
    (teamId != null && byKey.get(String(teamId))) ||
    (abbr && byKey.get(String(abbr).toUpperCase())) ||
    (name && byKey.get(String(name).toLowerCase())) ||
    null
  );
}
