import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { getFranchiseChemistry } from "../services/franchiseService";
import { SCREENS } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { CHEMISTRY_BANDS, chemistryLabel, chemistryTone } from "../utils/chemistryScale";
import "./ChemistryScreen.css";

const LINE_SOURCE_LABELS = {
  "session.lines": "Your saved lines",
  roster_projection: "Projected lines",
};

/** Same unit names as the Line Builder ("Line 1", "Pair 1"), not F1 / D1. */
function unitName(slot, fallback) {
  const raw = String(slot || "").trim();
  const m = raw.match(/^([FD])\s*(\d+)$/i);
  if (!m) return raw || fallback;
  return `${m[1].toUpperCase() === "F" ? "Line" : "Pair"} ${m[2]}`;
}

function scoreClass(score) {
  return `chemistry-score-${chemistryLabel(score).toLowerCase()}`;
}

function toneFor(score) {
  return chemistryTone(Number(score) || 0) || "low";
}

function ScoreMark({ score, label }) {
  if (score == null) return null;
  return (
    <span className={`chemistry-score-mark ${scoreClass(score)}`}>
      {score} · {label || chemistryLabel(score)}
    </span>
  );
}

function SideNavButton({ active, icon, label, onClick }) {
  return (
    <button
      type="button"
      className={`nhlcal-side-button${active ? " is-active" : ""}`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      <span className="nhlcal-side-icon">{icon}</span>
      <span className="nhlcal-side-label">{label}</span>
    </button>
  );
}

/* Coaching-board strength notation rather than a progress bar: five notches
   carry the reading, the numeral carries the precision, the tone carries the
   verdict. */
function Meter({ label, value, invert = false }) {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  const filled = Math.ceil(safe / 20);
  // Tension-style stats read better low, so their colour follows 100 - value.
  return (
    <div className="chemistry-meter" data-tone={toneFor(invert ? 100 - safe : safe)}>
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

function GroupCard({ title, rows = [], unitFallback }) {
  if (!rows.length) return null;
  return (
    <section className="chemistry-group">
      <h3 className="chemistry-section-title">{title}</h3>
      <div className="chemistry-grid chemistry-grid--units">
        {rows.map((row, idx) => (
          <article className="chemistry-line-card" data-tone={toneFor(row.chemistry)} key={`${title}-${idx}`}>
            <header className="chemistry-line-top">
              <span>{unitName(row.slot, `${unitFallback} ${idx + 1}`)}</span>
              <ScoreMark score={row.chemistry} label={row.label} />
            </header>
            {/* One player per row so every card lines up regardless of name length. */}
            <ul className="chemistry-line-players">
              {(row.players || []).length ? (
                (row.players || []).map((p, i) => (
                  <li className="chemistry-node" key={`${p.name}-${i}`}>
                    <em>{p.position}</em>
                    <strong>{p.name}</strong>
                  </li>
                ))
              ) : (
                <li className="chemistry-node">No players</li>
              )}
            </ul>
            <p>{row.identity || (row.source === "session.lines" ? "From your saved lines." : "Projected from current roster order.")}</p>
            {row.risk ? <p className="chemistry-risk">{row.risk}</p> : null}
            {(row.factors || []).length || (row.concerns || []).length ? (
              <div className="chemistry-chip-row">
                {(row.factors || []).slice(0, 3).map((f, i) => (
                  <span className="chemistry-factor-mark" key={`${f}-${i}`}>{f}</span>
                ))}
                {(row.concerns || []).slice(0, 2).map((c, i) => (
                  <span className="chemistry-concern-mark" key={`${c}-${i}`}>{c}</span>
                ))}
              </div>
            ) : null}
            {row.scheme_fit ? (
              <div className="chemistry-grid chemistry-worksheet">
                <Meter label="Pos Fit" value={row.scheme_fit.position_fit} />
                <Meter label="Linemates" value={row.scheme_fit.linemate_compatibility} />
                <Meter label="Role" value={row.scheme_fit.role_balance} />
                <Meter label="Coach Sys" value={row.scheme_fit.coach_system_fit} />
                <Meter label="Familiarity" value={row.scheme_fit.familiarity} />
                <Meter label="Morale" value={row.scheme_fit.morale} />
                <Meter label="Usage" value={row.scheme_fit.usage_satisfaction} />
              </div>
            ) : null}
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
  // The projection banner already says this; don't repeat it as a concern chip.
  const concerns = (Array.isArray(report?.concerns) ? report.concerns : []).filter(
    (c) => !/no saved even-strength lines/i.test(String(c))
  );
  const pressure = Array.isArray(report?.storyline_pressure) ? report.storyline_pressure : [];

  const teamName = report?.team_name || franchiseState?.team?.name || "Team";
  const teamLogo = resolveFranchiseTeamLogo(franchiseState?.team, teamName);
  const projected = report?.line_source && report.line_source !== "session.lines";

  const headline = useMemo(() => {
    const label = room?.label || "Neutral";
    const overall = Number(room?.overall) || 50;
    return `${label} room pulse at ${overall}`;
  }, [room?.label, room?.overall]);

  return (
    <div className="nhlcal-root chemistry-root">
      <aside className="nhlcal-sidebar">
        <button type="button" className="nhlcal-brand-button" onClick={() => setScreen(SCREENS.HUB)} title="Office">
          <span className="nhlcal-shield-icon">⌂</span>
        </button>
        <nav className="nhlcal-side-nav" aria-label="Chemistry navigation">
          <SideNavButton icon="▦" label="Office" onClick={() => setScreen(SCREENS.HUB)} />
          <SideNavButton icon="◫" label="Calendar" onClick={() => setScreen(SCREENS.CALENDAR)} />
          <SideNavButton icon="◉" label="Roster" onClick={() => setScreen(SCREENS.ROSTER)} />
          <SideNavButton icon="▥" label="Lines" onClick={() => setScreen(SCREENS.EDIT_LINES)} />
          <SideNavButton active icon="◍" label="Chemistry" />
        </nav>
      </aside>

    <main className="chemistry-screen">
      <div className="chemistry-hero">
        <div className="chemistry-hero-identity">
          <span className="chemistry-team-logo">
            {teamLogo ? <img src={teamLogo} alt={`${teamName} logo`} /> : null}
          </span>
          <div>
            <p className="chemistry-kicker">{teamName}</p>
            <h1>Room Chemistry</h1>
            <p>{headline}</p>
            <small>
              Last updated {report?.as_of_date || "today"}
              {report?.line_source ? ` · ${LINE_SOURCE_LABELS[report.line_source] || "Projected lines"}` : ""}
            </small>
          </div>
        </div>
        <button type="button" onClick={() => setScreen(SCREENS.EDIT_LINES)}>Edit Lines</button>
      </div>

      {!loading && !error && projected ? (
        <div className="chemistry-projection-note" role="status">
          <span>
            <strong>Projected lines.</strong> You haven't saved even-strength lines yet, so these units come from roster
            order. Save your lines in the Line Builder to see chemistry for the lineup you actually dress.
          </span>
          <button type="button" onClick={() => setScreen(SCREENS.EDIT_LINES)}>Open Line Builder</button>
        </div>
      ) : null}

      {loading ? <div className="chemistry-empty">Loading chemistry report...</div> : null}
      {!loading && error ? <div className="chemistry-empty">{error}</div> : null}

      {!loading && !error ? (
        <>
          <div className="chemistry-legend" aria-label="Chemistry tier guide">
            {CHEMISTRY_BANDS.map((band) => (
              <span key={band.tone} className={`is-${band.tone}`}>{band.label}</span>
            ))}
          </div>

          <section className="chemistry-room-card">
            <div className="chemistry-room-header">
              <h3>Room Pulse</h3>
              <ScoreMark score={room?.overall ?? 50} label={room?.label} />
            </div>
            <div className="chemistry-grid">
              <Meter label="Morale" value={room.morale} />
              <Meter label="Confidence" value={room.confidence} />
              <Meter label="Role Satisfaction" value={room.role_satisfaction} />
              <Meter label="Leadership" value={room.leadership} />
              <Meter label="Tension" value={room.tension} invert />
              <Meter label="Buy-In" value={room.buy_in} />
              <Meter label="Coach Trust" value={room.coach_trust} />
              <Meter label="Chaos Resistance" value={room.chaos_resistance} />
            </div>
          </section>

          <GroupCard title="Forward Lines" rows={lines} unitFallback="Line" />
          <GroupCard title="Defence Pairs" rows={pairs} unitFallback="Pair" />

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Goalie Room Fit</h3>
            <div className="chemistry-grid chemistry-grid--units">
              {goalies.length ? goalies.map((g) => (
                <article className="chemistry-line-card" data-tone={toneFor(g.chemistry)} key={g.player_id || g.name}>
                  <header className="chemistry-line-top">
                    <span>{g.name}</span>
                    <ScoreMark score={g.chemistry} label={g.label} />
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
                <article
                  className="chemistry-line-card chemistry-connection-card"
                  data-tone={toneFor(c.chemistry)}
                  key={`${c.player_a_id}-${c.player_b_id}-${i}`}
                >
                  <ul className="chemistry-line-players">
                    <li className="chemistry-node"><strong>{c.player_a_name}</strong></li>
                    <li className="chemistry-node"><strong>{c.player_b_name}</strong></li>
                  </ul>
                  <ScoreMark score={c.chemistry} label={c.label} />
                </article>
              ))}
            </div>
          </section>

          <section className="chemistry-group">
            <h3 className="chemistry-section-title">Room Concerns</h3>
            <div className="chemistry-chip-row">
              {(concerns.length ? concerns : ["No room concerns flagged."]).map((c, i) => (
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
    </main>
    </div>
  );
}

