import React from "react";

export function Loader({ label = "Running simulation…" }) {
  return (
    <div className="ui-loader" role="status" aria-live="polite">
      <div className="ui-loader__bar" />
      <p className="ui-loader__text">{label}</p>
    </div>
  );
}
