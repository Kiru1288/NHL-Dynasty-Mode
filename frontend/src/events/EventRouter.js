import React from "react";
import { getEventRegistration } from "./EventRegistry";

/**
 * Chooses which event UI subtree to mount from EventRegistry.
 */
export default function EventRouter({
  typeKey,
  franchiseState,
  eventData,
  onContinue,
  onBack,
  onClose,
  onEnterPlayoffs,
  playoffData,
}) {
  const entry = typeKey ? getEventRegistration(typeKey) : null;
  if (!entry?.component) return null;

  const Component = entry.component;
  const data = eventData ?? (entry.getEventData ? entry.getEventData(franchiseState) : {});

  if (typeKey === "playoffs_start") {
    return (
      <Component
        franchiseState={franchiseState}
        playoffData={playoffData || data}
        onEnterPlayoffs={onEnterPlayoffs}
        onClose={onClose}
        onBack={onBack}
      />
    );
  }

  return (
    <Component
      franchiseState={franchiseState}
      eventData={data}
      onContinue={onContinue}
      onBack={onBack}
      onClose={onClose}
    />
  );
}
