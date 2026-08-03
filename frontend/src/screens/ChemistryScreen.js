import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { getFranchiseChemistry } from "../services/franchiseService";
import { SCREENS } from "../game/constants";
import "./ChemistryScreen.css";

function scoreClass(score) {
  const s = Number(score) || 0;
  if (s >= 90) return "chemistry-score-elite";
  if (s >= 75) return "chemistry-score-strong";
  if (s >= 60) return "chemistry-score-connected";
  if (s >= 45) return "chemistry-score-neutral";
  if (s >= 30) return "chemistry-score-awkward";
  return "chemistry-score-broken";
}

function toneFor(score) {
  const s = Number(score) || 0;
  if (s >= 75) return "high";
  if (s >= 45) return "mid";
  return "low";
}

/* Coaching-board strength notation rather than a progress bar: five notches
   carry the reading, the numeral carries the precision, the tone carries the
   verdict. */
function Meter({ label, value }) {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  const filled = Math.ceil(safe / 20);
  return (
    <div className="chemistry-meter" data-tone={toneFor(safe)}>
      <div className="chemistry-meter-row">
        <span>{label}</span>
        <strong>{safe}</strong>
      </div>
      <div className="chemistry-notation" role="img" aria-label={`${label} ${safe} of 100`}>
        {[0, 1, 2, 3, 4].map((i) => (
          <i key={i} className={i < filled ? "is-on" : ""} />
        ))}
      </div>
    </div>
  );
}

function GroupCard({ title, rows = [] }) {
  if (!rows.length) return null;
  return (
    <section className="chemistry-group">
      <h3 className="chemistry-section-title">{title}</h3>
      <div className="chemistry-grid">
        {rows.map((row, idx) => (
          <article className="chemistry-line-card" data-tone={toneFor(row.chemistry)} key={`${title}-${idx}`}>
            <header className="chemistry-line-top">
              <span>{row.slot || "Unit"}</span>
              <span className={`chemistry-score-mark ${scoreClass(row.chemistry)}`}>
                {row.chemistry} · {row.label}
              </span>
            </header>
            {/* Unit read as a link chain: who is bonded to whom, in order. */}
            <div className="chemistry-line-players">
              {(row.players || []).length
                ? (row.players || []).map((p, i) => (
                    <React.Fragment key={`${p.name}-${i}`}>
                      {i > 0 ? <span className="chemistry-link-mark" aria-hidden="true" /> : null}
                      <span className="chemistry-node">
                        <strong>{p.name}</strong>
                        <em>{p.position}</em>
                      </span>
                    </React.Fragment>
                  ))
                : "No players"}
            </div>
            <p>{row.identity || (row.source === "session.lines" ? "From your saved lines." : "Projected from current roster order.")}</p>
            <p className="chemistry-risk">{row.risk || ""}</p>
            {row.scheme_fit ? (
              <div className="chemistry-grid" style={{ marginTop: 8 }}>
                <Meter label="Pos Fit" value={row.scheme_fit.position_fit} />
                <Meter label="Linemates" value={row.scheme_fit.linemate_compatibility} />
                <Meter label="Role" value={row.scheme_fit.role_balance} />
                <Meter label="Coach Sys" value={row.scheme_fit.coach_system_fit} />
                <Meter label="Familiarity" value={row.scheme_fit.familiarity} />
                <Meter label="Morale" value={row.scheme_fit.morale} />
                <Meter label="Usage" value={row.scheme_fit.usage_satisfaction} />
              </div>
            ) : null}
            <div className="chemistry-chip-row">
              {(row.factors || []).slice(0, 3).map((f, i) => (
                <span className="chemistry-factor-mark" key={`${f}-${i}`}>{f}</span>
              ))}
              {(row.concerns || []).slice(0, 2).map((c, i) => (
                <span className="chemistry-concern-mark" key={`${c}-${i}`}>{c}</span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function ChemistryScreen() {
  const { setScreen, franchiseState } = useGameUI();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    getFranchiseChemistry()
      .then((data) => {
        if (alive) setReport(data || null);
      })
      .catch(() => {
        if (alive) {
          setError("Chemistry report unavailable until the franchise room data initializes.");
          setReport(null);
        }
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const room = report?.room || {};
  const lines = Array.isArray(report?.lines) ? report.lines : [];
  const pairs = Array.isArray(report?.pairs) ? report.pairs : [];
  const goalies = Array.isArray(report?.goalies) ? report.goalies : [];
  const topConnections = Array.isArray(report?.top_connections) ? report.top_connections : [];
  const concerns = Array.isArray(report?.concerns) ? report.concerns : [];
  const pressure = Array.isArray(report?.storyline_pressure) ? report.storyline_pressure : [];

  const headline = useMemo(() => {
    const label = room?.label || "Neutral";
    const overall = Number(room?.overall) || 50;
    return `${label} room pulse at ${overall}`;
  }, [room?.label, room?.overall]);

  return (
    <div className="game-screen chemistry-screen">
      <div className="chemistry-hero">
        <div>
          <h2>Room Chemistry Report</h2>
          <p>{headline}</p>
          <small>
            {report?.team_name || franchiseState?.team?.name || "Team"} · Last updated {report?.as_of_date || "today"}
            {report?.line_source ? ` · Lines: ${report.line_source}` : ""}
          </small>
        </div>
        <button type="button" onClick={() => setScreen(SCREENS.HUB)}>Back To Hub</button>
      </div>

      {loading ? <div className="chemistry-empty">Loading chemistry report...</div> : null}
      {!loading && error ? <div className="chemistry-empty">{error}</div> : null}

      {!loading && !error ? (
        <>
          <div className="chemistry-legend" aria-label="Chemistry tier guide">
            <span className="is-high">High · 75+ elite fit</span>
            <span className="is-mid">Medium · 45–74 workable</span>
            <span className="is-low">Low · below 45 friction</span>
          </div>

          <section className="chemistry-room-card">
            <div className="chemistry-room-header">
              <h3>Room Pulse</h3>
              <span className={`chemistry-score-mark ${scoreClass(room.overall)}`}>
                {room?.overall ?? 50} · {room?.label || "Neutral"}
              </span>
            </div>
            <div className="chemistry-grid">
              <Meter label="Morale" value={room.morale} />
              <Meter label="Confidence" value={room.confidence} />
              <Meter label="Role Satisfaction" value={room.role_satisfaction} />
              <Meter label="Leadership" value={room.leadership} />
              <Meter label="Tension" value={room.tension} />
              <Meter label="Buy-In" value={room.buy_in} />
              <Meter label="Coach Trust" value={room.coach_trust} />
              <Meter label="Chaos Resistance" value={room.chaos_resistance} />
            </div>
          </section>

          <GroupCard title="Forward Line Chemistry" rows={lines} />
          <GroupCard title="Defense Pair Chemistry" rows={pairs} />

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Goalie Room Fit</h3>
            <div className="chemistry-grid">
              {goalies.length ? goalies.map((g) => (
                <article className="chemistry-line-card" key={g.player_id || g.name}>
                  <header className="chemistry-line-top">
                    <span>{g.name}</span>
                    <span className={`chemistry-score-mark ${scoreClass(g.chemistry)}`}>
                      {g.chemistry} · {g.label}
                    </span>
                  </header>
                  <p>Confidence {g.confidence} · Pressure response {g.pressure_response}</p>
                </article>
              )) : <article className="chemistry-line-card">No goalie room fit data yet.</article>}
            </div>
          </section>

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Top Connections</h3>
            <div className="chemistry-grid">
              {topConnections.slice(0, 6).map((c, i) => (
                <article className="chemistry-line-card" key={`${c.player_a_id}-${c.player_b_id}-${i}`}>
                  <header className="chemistry-line-top">
                    <span>{c.player_a_name} + {c.player_b_name}</span>
                    <span className={`chemistry-score-mark ${scoreClass(c.chemistry)}`}>{c.chemistry}</span>
                  </header>
                  <p>{c.label}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Room Concerns</h3>
            <div className="chemistry-chip-row">
              {(concerns.length ? concerns : ["Projected from current roster order."]).map((c, i) => (
                <span className="chemistry-concern-mark" key={`${c}-${i}`}>{c}</span>
              ))}
            </div>
          </section>

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Storyline Pressure</h3>
            <div className="chemistry-grid">
              {(pressure.length ? pressure : [{ text: "No active storyline pressure." }]).map((p, i) => (
                <article className="chemistry-line-card" key={`${p.text}-${i}`}>
                  <p>{p.text}</p>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

