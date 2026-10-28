import React from "react";
import { FreeAgencyEventMenu } from "../offseasonEventMenus";

export default function FreeAgencyMenu(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      {null}
      <FreeAgencyEventMenu {...props} />
    </div>
  );
}


