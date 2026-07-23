/**
 * Public barrel for the events system (scaffold).
 * Import from "src/events" when wiring the app shell — not connected yet.
 */
export { EVENT_TYPES } from "./eventTypes";
export { getEventDateAnchors } from "./eventDates";
export { noop as eventNoop } from "./eventUtils";
export { registerEventType, getEventRegistration, listRegisteredEventTypes } from "./EventRegistry";
export { default as EventRouter } from "./EventRouter";
export { default as EventModalShell } from "./EventModalShell";
export { default as FranchiseEventOverlay, getCurrentFranchiseEvent, getFranchisePhaseCta } from "./FranchiseEventOverlay";
export { createEventTriggerManager } from "./EventTriggerManager";
