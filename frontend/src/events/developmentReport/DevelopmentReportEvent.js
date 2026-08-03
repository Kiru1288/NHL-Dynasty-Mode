import React from "react";
import ProspectDevelopmentMenu from "../prospectDevelopment/ProspectDevelopmentMenu";

/** Offseason development camp — uses the full Prospect Development page. */
export default function DevelopmentReportEvent(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      <style>{DEV_REPORT_HOST_CSS}</style>
      <ProspectDevelopmentMenu {...props} />
    </div>
  );
}

const DEV_REPORT_HOST_CSS = `
.franchise-event-phase-host .nhlcal-root {
  min-height: 0;
  height: 100%;
  max-height: 100%;
}
.franchise-event-phase-host .nhlcal-sidebar {
  min-height: 0;
  height: 100%;
  max-height: 100%;
}
`;
