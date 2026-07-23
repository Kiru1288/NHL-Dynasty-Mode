import React, { useCallback } from "react";
import EventRouter from "./EventRouter";
import { getCurrentFranchiseEvent } from "./franchiseEventResolver";

/**
 * Full-screen franchise phase event host (playoffs + offseason chain).
 */
export default function FranchiseEventOverlay({
  franchiseState,
  onClose,
  onContinueOffseason,
  onGenerateNextSeason,
  onEnterPlayoffs,
  onAdvancePhase,
}) {
  const current = getCurrentFranchiseEvent(franchiseState);

  const handleContinueFlow = useCallback(async () => {
    const key = current?.key;
    try {
      if (key === "roster_cleanup") {
        if (!franchiseState?.flags?.can_generate_next_season) {
          return;
        }
        if (typeof onGenerateNextSeason === "function") await onGenerateNextSeason();
        return;
      }
      if (key === "playoffs_start") {
        if (typeof onEnterPlayoffs === "function") await onEnterPlayoffs();
        return;
      }
      if (typeof onContinueOffseason === "function") {
        await onContinueOffseason();
        return;
      }
      if (typeof onAdvancePhase === "function") await onAdvancePhase();
    } catch (err) {
      throw err;
    }
  }, [current?.key, franchiseState, onContinueOffseason, onGenerateNextSeason, onEnterPlayoffs, onAdvancePhase]);

  if (!current?.component) return null;

  return (
    <div className="franchise-event-overlay" role="presentation">
      <EventRouter
        typeKey={current.key}
        franchiseState={franchiseState}
        eventData={current.eventData}
        onContinue={handleContinueFlow}
        onBack={onClose}
        onClose={onClose}
        onEnterPlayoffs={onEnterPlayoffs}
        playoffData={current.eventData}
      />
    </div>
  );
}

export { getCurrentFranchiseEvent, getFranchisePhaseCta } from "./franchiseEventResolver";
