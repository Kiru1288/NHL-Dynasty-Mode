import React from "react";
import { NextSeasonRevealEventMenu } from "../offseasonEventMenus";
import OffseasonTimeline from "../offseasonTimeline";

/** Permanent Preseason event entry — bridges next season reveal into camp/preseason. */
export default function Preseason({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  return (
    <>
      <NextSeasonRevealEventMenu
        franchiseState={franchiseState}
        eventData={eventData}
        onContinue={onContinue}
        onBack={onBack}
      />
      <div style={{ position: "fixed", bottom: 88, left: 0, right: 0, zIndex: 25, padding: "0 1rem" }}>
        <OffseasonTimeline franchiseState={franchiseState} />
      </div>
    </>
  );
}
