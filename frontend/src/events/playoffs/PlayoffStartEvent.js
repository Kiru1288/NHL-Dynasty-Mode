import React from "react";
import PlayoffStartMenu from "./PlayoffStartMenu";

/** Stanley Cup playoff hub — interactive bracket + series simulation desk. */
export default function PlayoffStartEvent({
  franchiseState,
  playoffData,
  onEnterPlayoffs,
  onClose,
  onBack,
}) {
  return (
    <div className="po-start-menu-host franchise-event-phase-host" role="presentation">
      <PlayoffStartMenu
        franchiseState={franchiseState}
        playoffData={playoffData}
        onEnterPlayoffs={onEnterPlayoffs}
        onBack={onBack || onClose}
      />
    </div>
  );
}
