import React from "react";
import WorldJuniorsMenu from "./WorldJuniorsMenu";

/** Full-screen opaque shell — calendar must not bleed through. */
export default function WorldJuniorsEvent(props) {
  return (
    <div className="wjc-event-shell">
      <WorldJuniorsMenu {...props} />
    </div>
  );
}
