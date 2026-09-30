import React, { Suspense } from "react";
import { getEventRegistration } from "./EventRegistry";

const loadingStyle = {
  minHeight: "100vh",
  display: "grid",
  placeItems: "center",
  background: "#050a12",
  color: "rgba(201,168,106,0.85)",
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  fontSize: 12,
  fontWeight: 800,
};

export default function EventRouter(props) {
  return (
    <Suspense fallback={<div style={loadingStyle}>Loading…</div>}>
      <EventRouterInner {...props} />
    </Suspense>
  );
}

/**
 * Chooses which event UI subtree to mount from EventRegistry.
 */
function EventRouterInner({
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
        onContinue={onContinue}
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
