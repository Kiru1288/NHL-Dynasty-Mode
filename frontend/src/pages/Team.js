import React from "react";
import { Panel } from "../components/ui/Panel";

export function Team() {
  return (
    <div>
      <h1 className="page-title">Team</h1>
      <p className="page-sub">Placeholder — roster, cap, and line combinations.</p>
      <Panel title="Club dashboard" subtitle="Coming soon">
        <p style={{ margin: 0, color: "var(--text-muted)" }}>
          Target: team-scoped views tied to SimEngine state snapshots.
        </p>
      </Panel>
    </div>
  );
}
