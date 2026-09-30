import React, { useCallback } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import FreeAgencyMenu from "../events/freeAgency/FreeAgencyMenu";

/**
 * Standalone Free Agency Wire — same UI as the offseason timeline desk,
 * available from Hub without mutating offseason_stage. The menu fetches the live desk itself.
 */
export default function FreeAgency() {
  const { franchiseState, setScreen } = useGameUI();

  const onBack = useCallback(() => {
    setScreen(SCREENS.HUB);
  }, [setScreen]);

  return (
    <div className="game-screen free-agency-screen" style={{ height: "100%", minHeight: 0 }}>
      <FreeAgencyMenu
        franchiseState={franchiseState}
        eventData={{
          free_agency_market: franchiseState?.free_agency_market,
          free_agents: franchiseState?.free_agents,
        }}
        standalone
        onBack={onBack}
        onContinue={onBack}
        ctaLabel="Back to Hub"
      />
    </div>
  );
}
