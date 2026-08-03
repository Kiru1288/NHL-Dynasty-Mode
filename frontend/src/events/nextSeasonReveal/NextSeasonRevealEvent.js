import React from "react";
import { NextSeasonRevealEventMenu } from "../offseasonEventMenus";

export default function NextSeasonRevealEvent(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      <style>{NEXT_SEASON_HOST_CSS}</style>
      <NextSeasonRevealEventMenu {...props} />
    </div>
  );
}

const NEXT_SEASON_HOST_CSS = `
.franchise-event-phase-host .nextseason-root {
  min-height: 0;
  height: 100%;
  max-height: 100%;
}
.franchise-event-phase-host .nextseason-stage {
  min-height: 0;
  max-height: calc(100% - 140px);
}
.franchise-event-phase-host .nextseason-panel {
  border-radius: var(--radius-panel-lg);
  max-height: calc(100% - 80px);
}
.franchise-event-phase-host .nextseason-card {
  border-radius: var(--radius-panel);
}
`;
