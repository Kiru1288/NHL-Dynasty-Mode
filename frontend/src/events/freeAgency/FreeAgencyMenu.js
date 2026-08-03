import React from "react";
import { FreeAgencyEventMenu } from "../offseasonEventMenus";

export default function FreeAgencyMenu(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      <style>{FA_MENU_HOST_CSS}</style>
      <FreeAgencyEventMenu {...props} />
    </div>
  );
}

const FA_MENU_HOST_CSS = `
.franchise-event-phase-host .fa-root {
  min-height: 0;
  height: 100%;
  max-height: 100%;
}
.franchise-event-phase-host .fa-stage {
  height: calc(100% - 64px);
  max-height: calc(100% - 64px);
  min-height: 0;
}
.franchise-event-phase-host .fa-panel,
.franchise-event-phase-host .fa-desk {
  border-radius: var(--radius-panel-lg);
  max-height: 100%;
}
.franchise-event-phase-host .fa-list,
.franchise-event-phase-host .fa-capbar,
.franchise-event-phase-host .fa-stats > div,
.franchise-event-phase-host .fa-nego-panel {
  border-radius: var(--radius-panel);
}
.franchise-event-phase-host .fa-search,
.franchise-event-phase-host .fa-sort {
  border-radius: var(--radius-card);
}
.franchise-event-phase-host .fa-row {
  border-radius: var(--radius-control);
}
.franchise-event-phase-host .fa-deal-modal {
  border-radius: var(--radius-panel-lg);
}
`;
