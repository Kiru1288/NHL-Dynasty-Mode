import React, { useMemo } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import {
  formatStatValue,
  getLeagueTotals,
  getRetireeAge,
  getRetireeName,
  getRetireeOverall,
  getRetireePosition,
  getRetireeTeam,
  isGoalieRetiree,
  normalizeRetireesPayload,
  retireeToHeadshotPlayer,
  sortRetirees,
} from "./retirementHelpers";
import "./RetirementsBoard.css";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function StatCell({ label, value }) {
  return (
    <div className="retirement-stat-cell">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatGrid({ title, columns, goalie = false }) {
  return (
    <div className="retirement-stat-group">
      <h3>{title}</h3>
      <div className={`retirement-stat-grid${goalie ? " is-goalie" : ""}`}>
        {columns.map((col) => (
          <StatCell key={`${title}-${col.label}`} label={col.label} value={col.value} />
        ))}
      </div>
    </div>
  );
}

function skaterColumns(totals) {
  return [
    { label: "GP", value: formatStatValue(totals.gp, { decimals: 0 }) },
    { label: "G", value: formatStatValue(totals.g, { decimals: 0 }) },
    { label: "A", value: formatStatValue(totals.a, { decimals: 0 }) },
    { label: "PTS", value: formatStatValue(totals.pts, { decimals: 0 }) },
    { label: "+/-", value: formatStatValue(totals.pm, { decimals: 0 }) },
    { label: "PIM", value: formatStatValue(totals.pim, { decimals: 0 }) },
  ];
}

function goalieColumns(totals) {
  return [
    { label: "GP", value: formatStatValue(totals.gp, { decimals: 0 }) },
    { label: "W", value: formatStatValue(totals.w, { decimals: 0 }) },
    { label: "L", value: formatStatValue(totals.l, { decimals: 0 }) },
    { label: "SV%", value: formatStatValue(totals.sv_pct, { pct: true, decimals: 1 }) },
    { label: "GAA", value: formatStatValue(totals.gaa, { decimals: 2 }) },
  ];
}

function RetirementRow({ row, index }) {
  const name = getRetireeName(row);
  const team = getRetireeTeam(row);
  const position = getRetireePosition(row);
  const age = getRetireeAge(row);
  const overall = getRetireeOverall(row);
  const goalie = isGoalieRetiree(row);
  const nhl = getLeagueTotals(row, "nhl");
  const ahl = getLeagueTotals(row, "ahl");
  const player = ensurePlayerHeadshotFields(retireeToHeadshotPlayer(row));
  const statCols = goalie ? goalieColumns : skaterColumns;

  return (
    <article className="retirement-row" style={{ "--row-index": index }}>
      <div className="retirement-player-cell">
        <div className="retirement-headshot">
          <PlayerHeadshot player={player} size="md" variant="card" mood="neutral" />
        </div>
        <div className="retirement-player-meta">
          <strong className="retirement-player-name" title={name}>
            {name}
          </strong>
          <span className="retirement-player-team" title={team}>
            {team}
          </span>
          <span className="retirement-player-tags">
            {position}
            {age != null ? ` · Age ${age}` : ""}
          </span>
        </div>
      </div>

      <div className="retirement-ovr" aria-label={`Overall ${overall ?? "unknown"}`}>
        <span>OVR</span>
        <strong>{overall ?? "—"}</strong>
      </div>

      <div className="retirement-stats-wrap">
        <StatGrid title="NHL Career" columns={statCols(nhl)} goalie={goalie} />
        <StatGrid title="AHL Career" columns={statCols(ahl)} goalie={goalie} />
      </div>
    </article>
  );
}

export default function RetirementsBoard({ franchiseState = {}, retirees = [], onContinue, onBack }) {
  const sorted = useMemo(() => sortRetirees(normalizeRetireesPayload(retirees)), [retirees]);

  return (
    <section className="retirement-page">
      <div className="retirement-page__bg" aria-hidden="true" />
      <header className="retirement-header">
        <button type="button" className="retirement-back-btn" onClick={onBack}>
          ← Back
        </button>
        <h1>Retirements</h1>
        <span className="retirement-season">{seasonLabel(franchiseState)}</span>
      </header>

      <main className="retirement-list" aria-label="Retired players">
        {sorted.length ? (
          sorted.map((row, index) => (
            <RetirementRow key={`${row.player_id || row.name}-${index}`} row={row} index={index} />
          ))
        ) : (
          <p className="retirement-empty-state">No retirements this year.</p>
        )}
      </main>

      <footer className="retirement-actions">
        <button type="button" className="retirement-continue-btn" onClick={onContinue}>
          Continue to Salary Cap
        </button>
      </footer>
    </section>
  );
}
