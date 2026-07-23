import React, { useEffect, useState } from "react";
import { Panel } from "../components/ui/Panel";
import { api } from "../services/api";

const muted = { margin: 0, color: "var(--text-muted)" };
const listStyle = { margin: 0, paddingLeft: 18, display: "grid", gap: 6 };

function eventText(ev) {
  if (typeof ev === "string") return ev;
  return String(ev?.headline || ev?.message || ev?.title || ev?.text || "");
}

export function Narrative() {
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
        setError("League narrative unavailable. Start or load a franchise first.");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (state === null) {
    return (
      <div>
        <h1 className="page-title">Narrative</h1>
        <Panel title="Story center"><p style={muted}>Loading storylines…</p></Panel>
      </div>
    );
  }

  const storylines = (state.storyline_events || []).map(eventText).filter(Boolean);
  const timeline = (state.timeline || []).map(eventText).filter(Boolean);
  const notifications = (state.notifications || []).map(eventText).filter(Boolean);
  const injuries = (state.injuries || []).map(eventText).filter(Boolean);
  const empty = !storylines.length && !timeline.length && !notifications.length && !injuries.length;

  if (empty) {
    return (
      <div>
        <h1 className="page-title">Narrative</h1>
        <Panel title="Story center" subtitle="No data">
          <p style={muted}>{error || "No storylines yet — advance the calendar to generate league news."}</p>
        </Panel>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title">Narrative</h1>
      <p className="page-sub">League storylines and news from your franchise session.</p>
      {storylines.length > 0 && (
        <Panel title="Storylines" subtitle={`${storylines.length} active`}>
          <ul style={listStyle}>
            {storylines.slice(0, 20).map((s, i) => (
              <li key={`s-${i}`}>{s}</li>
            ))}
          </ul>
        </Panel>
      )}
      {notifications.length > 0 && (
        <Panel title="League news">
          <ul style={listStyle}>
            {notifications.slice(0, 15).map((s, i) => (
              <li key={`n-${i}`}>{s}</li>
            ))}
          </ul>
        </Panel>
      )}
      {injuries.length > 0 && (
        <Panel title="Injury report">
          <ul style={listStyle}>
            {injuries.slice(0, 12).map((s, i) => (
              <li key={`i-${i}`}>{s}</li>
            ))}
          </ul>
        </Panel>
      )}
      {timeline.length > 0 && (
        <Panel title="Recent timeline">
          <ul style={listStyle}>
            {timeline.slice(-12).reverse().map((s, i) => (
              <li key={`t-${i}`} style={{ color: "var(--text-muted)" }}>{s}</li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
