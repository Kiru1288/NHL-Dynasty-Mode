import React from "react";
import OffseasonTimeline from "../offseasonTimeline";
import "../awardsNight/AwardsNight.css";

/** Permanent Training Camp event — scaffold for roster battles and camp reports. */
export default function TrainingCamp({ franchiseState = {}, onContinue, onBack }) {
  return (
    <section className="an-root">
      <header className="an-topbar">
        <button type="button" className="an-ghost-btn" onClick={onBack}>← Back</button>
        <div className="an-status-pill">TRAINING CAMP</div>
        <div className="an-season" />
      </header>
      <main className="an-stage" style={{ gridTemplateColumns: "1fr" }}>
        <div className="an-center">
          <p className="an-ceremony-kicker">Roster Battles</p>
          <h1 className="an-ceremony-title">Training Camp</h1>
          <p className="an-empty">Camp reports will live here as the franchise timeline expands.</p>
        </div>
      </main>
      <div className="an-timeline-wrap"><OffseasonTimeline franchiseState={franchiseState} /></div>
      <footer className="an-footer">
        <div className="an-footer-actions">
          <button type="button" className="an-cta-btn" onClick={onContinue}>Continue</button>
        </div>
      </footer>
    </section>
  );
}
