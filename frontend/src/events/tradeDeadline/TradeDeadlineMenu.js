import React from "react";
import ComingSeasonNight from "../shared/ComingSeasonNight";

export default function TradeDeadlineMenu({
  franchiseState = {},
  onContinue,
  onBack,
}) {
  return (
    <ComingSeasonNight
      phaseLabel="TRADE DEADLINE"
      title="Deadline Night"
      eyebrow="March wire"
      body="Deadline night is on the calendar. The trade desk broadcast will land here — for now, keep working the club from the hub."
      ctaLabel="Return to Hub"
      franchiseState={franchiseState}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}
