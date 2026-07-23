/**
 * Observes sim/calendar progression and enqueues events for the UI layer.
 * Should call into EventRegistry + navigation/modal host — not implemented yet.
 *
 * Scaffold only — no subscriptions or timers.
 */
export function createEventTriggerManager() {
  return {
    dispose() {},
  };
}
