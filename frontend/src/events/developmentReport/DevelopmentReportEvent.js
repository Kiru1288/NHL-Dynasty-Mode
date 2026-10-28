import React from "react";
import ProspectDevelopmentMenu from "../prospectDevelopment/ProspectDevelopmentMenu";

/** Offseason development camp — uses the full Prospect Development page. */
export default function DevelopmentReportEvent(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      {null}
      <ProspectDevelopmentMenu {...props} />
    </div>
  );
}


