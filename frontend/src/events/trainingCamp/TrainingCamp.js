import React from "react";
import ComingSeasonNight from "../shared/ComingSeasonNight";

export default function TrainingCamp({
  franchiseState = {},
  onContinue,
  onBack,
}) {
  return (
    <ComingSeasonNight
      phaseLabel="TRAINING CAMP"
      title="Training Camp"
      eyebrow="Roster battles"
      body="Camp reports and roster battles will live here. Return to the hub and keep the club moving."
      ctaLabel="Continue"
      franchiseState={franchiseState}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}
