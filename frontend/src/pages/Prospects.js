import React, { useEffect, useState } from "react";
import { Panel } from "../components/ui/Panel";
import { api } from "../services/api";

const muted = { margin: 0, color: "var(--text-muted)" };
const cellStyle = { padding: "6px 10px", borderBottom: "1px solid rgba(255,255,255,0.06)", whiteSpace: "nowrap" };
const headStyle = { ...cellStyle, textAlign: "left", fontSize: 12, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--text-muted)" };

function round0(v) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n) : null;
}

function fmtPpg(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : "—";
}

function stockBadge(p) {
  const d = Number(p.stock_delta || 0);
  const label = p.stock_label || (d > 0 ? "Rising" : d < 0 ? "Falling" : "Holding");
  const color = d > 0 ? "#3ddc84" : d < 0 ? "#ff6b6b" : "var(--text-muted)";
  return <span style={{ color }}>{d > 0 ? "▲" : d < 0 ? "▼" : "•"} {label}</span>;
}

function statLine(p) {
  const mode = p.stats_mode || (Number(p.gp || 0) > 0 ? "current" : "projected");
  if (mode === "current") {
    return `${p.gp ?? 0} GP · ${p.goals ?? 0}G ${p.assists ?? 0}A ${p.points ?? 0}P · ${fmtPpg(p.ppg)} PPG`;
  }
  if (p.projected_points != null || p.projected_ppg != null) {
    return `${p.projected_gp ?? "—"} GP · ${p.projected_points ?? "—"}P · ${fmtPpg(p.projected_ppg)} PPG`;
  }
  return "—";
}

export function Prospects() {
  const [prospects, setProspects] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    api
      .get("/api/franchise/scouting/prospects")
      .then(({ data }) => {
        if (!alive) return;
        setProspects(Array.isArray(data?.prospects) ? data.prospects : []);
      })
      .catch(() => {
        if (!alive) return;
        setProspects([]);
        setError("Scouting data unavailable. Start or load a franchise to populate the draft board.");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (prospects === null) {
    return (
      <div>
        <h1 className="page-title">Prospects</h1>
        <Panel title="Draft board"><p style={muted}>Loading scouting data…</p></Panel>
      </div>
    );
  }

  if (!prospects.length) {
    return (
      <div>
        <h1 className="page-title">Prospects</h1>
        <Panel title="Draft board" subtitle="No data">
          <p style={muted}>{error || "No prospects available yet for this franchise."}</p>
        </Panel>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title">Prospects</h1>
      <p className="page-sub">Draft board — scouting-limited view; stats marked projected until games are played.</p>
      <Panel title="Draft board" subtitle={`${prospects.length} prospects`}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
            <thead>
              <tr>
                <th style={headStyle}>#</th>
                <th style={headStyle}>Name</th>
                <th style={headStyle}>Pos</th>
                <th style={headStyle}>League</th>
                <th style={headStyle}>Nation</th>
                <th style={headStyle}>Potential</th>
                <th style={headStyle}>Risk</th>
                <th style={headStyle}>Stock</th>
                <th style={headStyle}>Stats</th>
                <th style={headStyle}>Mode</th>
              </tr>
            </thead>
            <tbody>
              {prospects.slice(0, 100).map((p) => {
                const lo = round0(p.ovr_range?.low);
                const hi = round0(p.ovr_range?.high);
                const pot = p.ovr_revealed && p.true_ovr != null
                  ? String(round0(p.true_ovr))
                  : lo != null && hi != null
                  ? `${lo}–${hi}`
                  : "—";
                const mode = p.stats_mode || (Number(p.gp || 0) > 0 ? "current" : "projected");
                return (
                  <tr key={p.id || p.key || p.name}>
                    <td style={cellStyle}>{p.rank || "—"}</td>
                    <td style={cellStyle}>{p.name}</td>
                    <td style={cellStyle}>{p.position}</td>
                    <td style={cellStyle}>{p.league || "—"}</td>
                    <td style={cellStyle}>{p.country || "—"}</td>
                    <td style={cellStyle}>{pot}</td>
                    <td style={cellStyle}>{p.risk || "—"}</td>
                    <td style={cellStyle}>{stockBadge(p)}</td>
                    <td style={cellStyle}>{statLine(p)}</td>
                    <td style={{ ...cellStyle, color: "var(--text-muted)" }}>
                      {mode === "current" ? "Season to date" : "Projected"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
