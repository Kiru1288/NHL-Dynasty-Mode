import React from "react";
import WorldJuniorsRevamp from "./WorldJuniorsRevamp";

/** Full-screen opaque shell. The holiday rink village replaced the old WorldJuniorsMenu dashboard (kept in the repo). */
export default function WorldJuniorsEvent(props) {
  return (
    <div className="wjc-event-shell">
      <WorldJuniorsRevamp {...props} />
    </div>
  );
}
