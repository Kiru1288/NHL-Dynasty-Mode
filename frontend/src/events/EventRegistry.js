/**
 * Maps EVENT_TYPES → lazy component loaders or static component refs.
 */
import { EVENT_TYPES } from "./eventTypes";
import { registerFranchiseEventTypes } from "./franchiseEventResolver";

const registry = Object.create(null);

export function registerEventType(typeKey, spec) {
  if (!typeKey || !spec) return;
  registry[typeKey] = spec;
}

export function getEventRegistration(typeKey) {
  return registry[typeKey] ?? null;
}

export function listRegisteredEventTypes() {
  return Object.keys(registry);
}

registerFranchiseEventTypes(registerEventType);

export { EVENT_TYPES };
