import React from "react";
import OffseasonTimeline from "../offseasonTimeline";
import "../awardsNight/AwardsNight.css";

/**
 * Permanent Hall of Fame event — scaffold for future induction ceremonies.
 * Will support: induction classes, career comparisons, legacy tracking.
 */
export default function HallOfFame({ franchiseState = {}, onContinue, onBack }) {
  const inductees = franchiseState?.hall_of_fame?.inductees || franchiseState?.hof_inductees || [];

  return (
    <section className="an-root">
      <header className="an-topbar">
        <button type="button" className="an-ghost-btn" onClick={onBack}>← Back</button>
        <div className="an-status-pill">HALL OF FAME</div>
        <div className="an-season" />
      </header>
      <main className="an-stage" style={{ gridTemplateColumns: "1fr" }}>
        <div className="an-center">
          <p className="an-ceremony-kicker">Franchise Legacy</p>
          <h1 className="an-ceremony-title">Hall of Fame</h1>
          {inductees.length ? (
            <ul className="an-rail-list">
              {inductees.map((row, i) => (
                <li key={i} className="an-rail-item">{row.name || row.player_name || "Inductee"}</li>
              ))}
            </ul>
          ) : (
            <p className="an-empty">No Hall of Fame class this season. This event will expand as franchise history grows.</p>
          )}
        </div>
      </main>
      <div className="an-timeline-wrap">
        <OffseasonTimeline franchiseState={franchiseState} />
      </div>
      <footer className="an-footer">
        <div className="an-footer-actions">
          <button type="button" className="an-cta-btn" onClick={onContinue}>Continue</button>
        </div>
      </footer>
    </section>
  );
}
