import React, { useEffect, useState } from "react";
import { Panel } from "../components/ui/Panel";
import { api } from "../services/api";

const muted = { margin: 0, color: "var(--text-muted)" };
const rowStyle = { display: "flex", justifyContent: "space-between", gap: 12, padding: "4px 0" };

function round0(v) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n) : null;
}

function money(v) {
  const n = Number(v);
  return Number.isFinite(n) ? `$${n.toFixed(1)}M` : "—";
}

export function Team() {
  const [state, setState] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    api
      .get("/api/franchise/state")
      .then(({ data }) => {
        if (alive) setState(data || {});
      })
      .catch(() => {
        if (!alive) return;
        setState({});
        setError("Team data unavailable. Start or load a franchise first.");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (state === null) {
    return (
      <div>
        <h1 className="page-title">Team</h1>
        <Panel title="Club dashboard"><p style={muted}>Loading team data…</p></Panel>
      </div>
    );
  }

  const team = state.team || null;
  if (!team) {
    return (
      <div>
        <h1 className="page-title">Team</h1>
        <Panel title="Club dashboard" subtitle="No data">
          <p style={muted}>{error || "No active franchise session."}</p>
        </Panel>
      </div>
    );
  }

  const rec = team.record || {};
  const standings = state.standings || [];
  const rankIdx = standings.findIndex((s) => String(s.team_id) === String(team.id));
  const seasonStarted = standings.some((s) => Number(s.gp || 0) > 0);
  const zeroGpTeams = seasonStarted ? standings.filter((s) => Number(s.gp || 0) === 0) : [];

  const sc = state.stats_central || {};
  const leaders = (sc.user_leaders || []).slice(0, 5);
  const goalies = (sc.goalie_leaders || []).filter((g) => String(g.team_id) === String(team.id)).slice(0, 2);
  const roster = state.roster || [];
  const injuries = (state.injuries || []).slice(0, 8);

  return (
    <div>
      <h1 className="page-title">{team.name || "Team"}</h1>
      <p className="page-sub">
        {team.coach ? `Coach ${team.coach} · ` : ""}
        {rec.gp != null ? `${rec.w ?? 0}-${rec.l ?? 0}-${rec.otl ?? 0} (${rec.pts ?? 0} pts)` : "Season not started"}
        {rankIdx >= 0 ? ` · #${rankIdx + 1} in league` : ""}
      </p>

      {zeroGpTeams.length > 0 && (
        <Panel title="Data warning" subtitle="Validation">
          <p style={{ ...muted, color: "#ff6b6b" }}>
            {zeroGpTeams.length} team(s) show 0 GP while the season is underway — check the schedule sync.
          </p>
        </Panel>
      )}

      <Panel title="Cap snapshot">
        <div style={rowStyle}><span>Cap hit</span><strong>{money(team.cap_hit_m ?? team.capHit ?? team.cap_hit)}</strong></div>
        <div style={rowStyle}><span>Cap space</span><strong>{money(team.cap_space_m ?? team.capSpace ?? team.cap_space)}</strong></div>
        <div style={rowStyle}><span>Roster size</span><strong>{roster.length || "—"}</strong></div>
      </Panel>

      <Panel title="Scoring leaders" subtitle={leaders.length ? "Season to date" : undefined}>
        {leaders.length ? (
          leaders.map((p, i) => (
            <div key={p.player_id || i} style={rowStyle}>
              <span>{p.name} {p.position ? `(${p.position})` : ""}</span>
              <strong>{p.g ?? p.goals ?? 0}G {p.a ?? p.assists ?? 0}A {(p.pts ?? p.points ?? (Number(p.g || 0) + Number(p.a || 0)))}P</strong>
            </div>
          ))
        ) : (
          <p style={muted}>No game-derived stats yet — advance the calendar.</p>
        )}
      </Panel>

      <Panel title="Goaltending">
        {goalies.length ? (
          goalies.map((g, i) => (
            <div key={g.player_id || i} style={rowStyle}>
              <span>{g.name}</span>
              <strong>
                {g.w ?? g.wins ?? 0}W · {g.save_pct != null ? Number(g.save_pct).toFixed(3) : "—"} SV% ·{" "}
                {g.gaa != null ? Number(g.gaa).toFixed(2) : "—"} GAA
              </strong>
            </div>
          ))
        ) : (
          <p style={muted}>No goalie stats recorded yet.</p>
        )}
      </Panel>

      {injuries.length > 0 && (
        <Panel title="Injuries">
          {injuries.map((inj, i) => (
            <div key={i} style={rowStyle}>
              <span>{typeof inj === "string" ? inj : inj.player || inj.name || "Player"}</span>
              <span style={{ color: "var(--text-muted)" }}>
                {typeof inj === "object" ? inj.status || inj.detail || "" : ""}
              </span>
            </div>
          ))}
        </Panel>
      )}

      {roster.length > 0 && (
        <Panel title="Roster summary" subtitle={`${roster.length} players`}>
          {roster.slice(0, 10).map((p, i) => (
            <div key={p.id || i} style={rowStyle}>
              <span>{p.name} {p.position ? `(${p.position})` : ""}</span>
              <strong>{round0(p.ovr) ?? "—"} OVR</strong>
            </div>
          ))}
          {roster.length > 10 && <p style={muted}>+{roster.length - 10} more on the full roster screen.</p>}
        </Panel>
      )}
    </div>
  );
}
