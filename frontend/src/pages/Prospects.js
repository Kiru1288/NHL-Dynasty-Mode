import React from "react";
import { Panel } from "../components/ui/Panel";

export function Prospects() {
  return (
    <div>
      <h1 className="page-title">Prospects</h1>
      <p className="page-sub">Placeholder — draft board, pipeline tiers, scouting summaries.</p>
      <Panel title="Prospect operations" subtitle="Coming soon">
        <p style={{ margin: 0, color: "var(--text-muted)" }}>
          Hook into draft logs in <code>timeline_log.txt</code> and structured JSON when exposed.
        </p>
      </Panel>
    </div>
  );
}
