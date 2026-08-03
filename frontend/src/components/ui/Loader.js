import React from "react";

const KICKERS = {
  ops: "FEED ACQUISITION",
  office: "OFFICE REGISTRATION",
  cinematic: "EVENT PACKAGE",
};

export function Loader({
  label = "Synchronizing league data…",
  register = "ops",
  kicker,
}) {
  const family = KICKERS[register] ? register : "ops";

  return (
    <div
      className={`ui-loader fcn-load fcn-load--${family}`}
      role="status"
      aria-live="polite"
    >
      <span className="ops-state__kicker">{kicker || KICKERS[family]}</span>
      <div className="fcn-load__track" aria-hidden="true">
        {family === "ops" ? (
          <>
            <span />
            <span />
            <span />
          </>
        ) : null}
      </div>
      <p className="ui-loader__text fcn-load__label">{label}</p>
    </div>
  );
}
