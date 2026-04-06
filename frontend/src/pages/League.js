import React from "react";
import { Panel } from "../components/ui/Panel";

export function League() {
  return (
    <div>
      <h1 className="page-title">League</h1>
      <p className="page-sub">Placeholder — wire standings, schedule, and league stats from saved JSON runs later.</p>
      <Panel title="League overview" subtitle="Coming soon">
        <p style={{ margin: 0, color: "var(--text-muted)" }}>
          Consume <code>universe_year_*.json</code> from a completed run folder via a future endpoint.
        </p>
      </Panel>
    </div>
  );
}
