import React from "react";
import { getTimelineSteps } from "./timelineConfig";

export default function OffseasonTimeline({ franchiseState, className = "" }) {
  const steps = getTimelineSteps(franchiseState);

  return (
    <nav className={`franchise-timeline ${className}`.trim()} aria-label="Franchise progression">
      <ol className="franchise-timeline-track">
        {steps.map((step) => (
          <li
            key={step.id}
            className={`franchise-timeline-step is-${step.state}${step.status === "planned" ? " is-planned" : ""}`}
            title={step.status === "planned" ? "Planned franchise milestone" : step.label}
          >
            <span className="franchise-timeline-dot" aria-hidden="true">
              {step.state === "done" ? "✓" : ""}
            </span>
            <span className="franchise-timeline-label">{step.label}</span>
          </li>
        ))}
      </ol>
    </nav>
  );
}
