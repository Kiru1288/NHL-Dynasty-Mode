/** Shared timing + class helpers for franchise cinematic blur transitions. */
export const FRANCHISE_EVENT_TRANSITION_MS = 560;
export const FRANCHISE_EVENT_TRANSITION_HALF_MS = Math.round(FRANCHISE_EVENT_TRANSITION_MS / 2);

export const FRANCHISE_EVENT_LAYER_CLASSES = {
  root: "franchise-event-layer",
  overlay: "franchise-event-overlay",
  content: "franchise-event-content",
  entering: "is-entering",
  visible: "is-visible",
  exiting: "is-exiting",
  swappingOut: "is-swapping-out",
  swappingIn: "is-swapping-in",
  phaseBlur: "is-phase-blur",
};

export function prefersReducedMotion() {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function transitionDelay(ms = FRANCHISE_EVENT_TRANSITION_MS) {
  return prefersReducedMotion() ? 0 : ms;
}
