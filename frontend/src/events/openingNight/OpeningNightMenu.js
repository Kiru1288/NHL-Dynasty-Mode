import React from "react";
import ComingSeasonNight from "../shared/ComingSeasonNight";

export default function OpeningNightMenu({
  franchiseState = {},
  onContinue,
  onBack,
}) {
  return (
    <ComingSeasonNight
      phaseLabel="OPENING NIGHT"
      title="Opening Night"
      eyebrow="Season drop"
      body="Opening night is on the calendar. The puck-drop broadcast will land here — for now, return to the hub and keep the club ready."
      ctaLabel="Return to Hub"
      franchiseState={franchiseState}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}
