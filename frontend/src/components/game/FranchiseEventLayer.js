import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import FranchiseEventOverlay, {
  getCurrentFranchiseEvent,
} from "../../events/FranchiseEventOverlay";

function playoffsAreComplete(franchiseState) {
  return Boolean(
    franchiseState?.playoffs_done ||
      franchiseState?.flags?.playoffs_done ||
      franchiseState?.flags?.playoffs_simulated
  );
}

/**
 * Global cinematic franchise events (playoffs + offseason).
 * Sits above ShowcasePopupLayer so menus appear instead of raw JSON popups.
 *
 * Players can leave the timeline to Hub World (trades, stats, office) without
 * advancing the stage; reopen via Hub phase CTA / openFranchiseEvent.
 */
export function FranchiseEventLayer() {
  const {
    franchiseState,
    onContinueOffseason,
    onReopenOffseasonStage,
    onGenerateNextSeason,
    onEnterPlayoffs,
    onAdvanceSeasonPhase,
    franchiseEventForceOpen,
    setFranchiseEventForceOpen,
    setScreen,
  } = useGameUI();

  const [dismissed, setDismissed] = useState(false);
  const [pinnedOpen, setPinnedOpen] = useState(false);
  const event = useMemo(
    () => getCurrentFranchiseEvent(franchiseState),
    [franchiseState]
  );

  const eventKey = event?.key;
  const phase = String(
    franchiseState?.season_phase || franchiseState?.phase || ""
  ).toLowerCase();
  const stage = String(franchiseState?.offseason_stage || "").toLowerCase();

  useEffect(() => {
    setDismissed(false);
  }, [eventKey, phase, stage]);

  useEffect(() => {
    if (!franchiseEventForceOpen) return;
    setDismissed(false);
    setPinnedOpen(true);
    setFranchiseEventForceOpen(false);
  }, [franchiseEventForceOpen, setFranchiseEventForceOpen]);

  const phaseAllowsAuto =
    ["playoff_ready", "post_cup", "offseason", "playoffs"].includes(phase) ||
    (phase === "complete" && playoffsAreComplete(franchiseState));

  const shouldShow =
    Boolean(event) && !dismissed && (pinnedOpen || phaseAllowsAuto);

  const handleLeaveToHub = () => {
    setDismissed(true);
    setPinnedOpen(false);
    if (typeof setScreen === "function") {
      setScreen(SCREENS.HUB);
    }
  };

  if (!shouldShow || !event) return null;

  return (
    <div className="franchise-event-layer register-ops" data-register="ops" role="presentation">
      <style>{FRANCHISE_EVENT_LAYER_CSS}</style>
      <FranchiseEventOverlay
        franchiseState={franchiseState}
        onClose={handleLeaveToHub}
        onContinueOffseason={onContinueOffseason}
        onReopenOffseasonStage={onReopenOffseasonStage}
        onGenerateNextSeason={onGenerateNextSeason}
        onEnterPlayoffs={onEnterPlayoffs}
        onAdvancePhase={onAdvanceSeasonPhase}
      />
    </div>
  );
}

const FRANCHISE_EVENT_LAYER_CSS = `
.franchise-event-layer {
  background: var(--ops-navy-deep);
  color: var(--ops-text);
  font-family: var(--font-ops-ui);
  isolation: isolate;
}
.franchise-event-overlay {
  background: var(--ops-navy-deep);
}
.franchise-event-phase-host {
  height: 100%;
  min-height: 0;
  max-height: 100%;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.franchise-event-phase-host > * {
  flex: 1;
  min-height: 0;
  max-height: 100%;
}
`;
