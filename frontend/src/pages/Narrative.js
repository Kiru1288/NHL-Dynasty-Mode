import React from "react";
import { Panel } from "../components/ui/Panel";

export function Narrative() {
  return (
    <div>
      <h1 className="page-title">Narrative</h1>
      <p className="page-sub">Placeholder — storylines, news feed, and player journeys.</p>
      <Panel title="Story center" subtitle="Coming soon">
        <p style={{ margin: 0, color: "var(--text-muted)" }}>
          Perspective logs already capture GM-facing narrative; universe timeline lives in standard runs.
        </p>
      </Panel>
    </div>
  );
}
