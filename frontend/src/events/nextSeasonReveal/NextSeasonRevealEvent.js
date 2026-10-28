import React from "react";
import { NextSeasonRevealEventMenu } from "../offseasonEventMenus";

export default function NextSeasonRevealEvent(props) {
  return (
    <div className="franchise-event-phase-host register-ops" data-register="ops">
      {null}
      <NextSeasonRevealEventMenu {...props} />
    </div>
  );
}


