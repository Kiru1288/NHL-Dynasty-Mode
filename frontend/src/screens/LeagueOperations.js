import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getLeagueOperations } from "../services/franchiseService";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

const EMPTY_OBJ = Object.freeze({});
const EMPTY_ARR = Object.freeze([]);

const ACTION_STATES = ["Stable", "Monitoring", "Negotiating", "Vote Required"];
const SORT_KEYS = ["revenue", "trend", "market", "risk"];
const WORKSPACES = [
  { id: "overview", label: "Overview" },
  { id: "cba", label: "CBA" },
  { id: "markets", label: "Markets" },
  { id: "risk", label: "Franchise Risk" },
];
const MARKET_FILTERS = [
  { id: "all", label: "All" },
  { id: "growing", label: "Growing" },
  { id: "losing", label: "Losing" },
  { id: "reloc", label: "Relocation Risk" },
];
const CALM_MOODS = new Set(["calm", "quiet", "stable"]);

const LO_STYLES = `
.lo-screen {
  --lo-panel: var(--ops-panel, rgba(9, 25, 38, 0.94));
  --lo-line: var(--ops-grid, rgba(156, 218, 236, 0.14));
  --lo-text: var(--ops-text, #e9f7fb);
  --lo-muted: var(--ops-text-secondary, #8096a8);
  --lo-cyan: var(--ops-cyan, #13d8e7);
  --lo-gold: var(--ops-gold, #e9a83c);
  --lo-green: var(--ops-success, #52df94);
  --lo-red: var(--ops-injury, #ff606d);
  --lo-yellow: var(--ops-gold, #e9a83c);
  --lo-orange: #ff8c3c;
  flex: 1;
  height: 100%;
  max-height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 8px 14px 10px !important;
  color: var(--lo-text);
  background:
    radial-gradient(ellipse at 14% 0%, rgba(19, 216, 231, 0.06), transparent 42%),
    radial-gradient(ellipse at 86% 4%, rgba(233, 168, 60, 0.05), transparent 36%),
    linear-gradient(180deg, var(--ops-black, #061522) 0%, var(--ops-navy-deep, #020a11) 100%);
  font-family: var(--font-ops-ui, Inter, ui-sans-serif, system-ui, sans-serif);
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  gap: 8px;
  scrollbar-width: thin;
  scrollbar-color: rgba(156, 218, 236, 0.28) rgba(4, 14, 22, 0.95);
}
.lo-screen * { box-sizing: border-box; }
.lo-screen *::-webkit-scrollbar { width: 8px; height: 8px; }
.lo-screen *::-webkit-scrollbar-track {
  background: rgba(4, 14, 22, 0.95);
  border-radius: 8px;
}
.lo-screen *::-webkit-scrollbar-thumb {
  background: rgba(156, 218, 236, 0.28);
  border-radius: 8px;
  border: 2px solid rgba(4, 14, 22, 0.95);
}
.lo-screen *::-webkit-scrollbar-thumb:hover {
  background: rgba(19, 216, 231, 0.45);
}
.lo-screen *::-webkit-scrollbar-corner { background: rgba(4, 14, 22, 0.95); }

.lo-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  height: 48px;
  max-height: 54px;
  padding: 0 2px;
}

.lo-header-left {
  display: flex;
  align-items: baseline;
  gap: 12px;
  min-width: 0;
}

.lo-header-left h1 {
  margin: 0;
  font-size: var(--type-ops-heading-size, 0.95rem);
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--lo-text);
  white-space: nowrap;
}

.lo-header-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}

.lo-header-meta .season { color: var(--lo-text); font-weight: 800; }
.lo-header-meta .status { color: var(--lo-cyan); font-weight: 800; }
.lo-header-meta .lo-header-note {
  color: var(--lo-muted);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.6875rem;
}

.lo-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.lo-impact-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 10px;
  border-top: 1px solid var(--lo-line);
  border-bottom: 1px solid var(--lo-line);
  border-radius: 0;
  background: rgba(6, 21, 34, 0.55);
  max-height: 40px;
}

.lo-impact-strip img { width: 22px; height: 22px; object-fit: contain; }

.lo-impact-strip .abbr-fallback {
  width: 22px;
  height: 22px;
  display: grid;
  place-items: center;
  font-size: 0.6875rem;
  font-weight: 900;
  color: var(--lo-gold);
  background: rgba(233, 168, 60, 0.12);
  border-radius: 5px;
}

.lo-impact-pill {
  display: flex;
  flex-direction: column;
  gap: 0;
  line-height: 1.15;
}

.lo-impact-pill span {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--lo-muted);
}

.lo-impact-pill b {
  font-size: 0.76rem;
  font-weight: 800;
  color: var(--lo-gold);
  font-variant-numeric: tabular-nums;
}

.lo-impact-pill b.warn { color: var(--lo-red); }
.lo-impact-pill b.ok { color: var(--lo-green); }

.lo-header button,
.lo-tabs button,
.lo-filters button,
.lo-issue,
.lo-risk-card,
.lo-table th button {
  font: inherit;
}

.lo-back {
  height: 34px;
  padding: 0 14px;
  border: 1px solid var(--lo-line);
  border-radius: 8px;
  background: transparent;
  color: var(--lo-text);
  font-weight: 700;
  font-size: 0.72rem;
  cursor: pointer;
}
.lo-back:hover,
.lo-back:focus-visible {
  border-color: var(--lo-cyan);
  outline: none;
  box-shadow: 0 0 0 2px rgba(19, 216, 231, 0.25);
}

.lo-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 0 2px;
  border-bottom: 1px solid var(--lo-line);
}

.lo-tabs button {
  height: 32px;
  padding: 0 14px;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  background: transparent;
  color: var(--lo-muted);
  font-size: 0.76rem;
  font-weight: 800;
  cursor: pointer;
}
.lo-tabs button:hover { color: var(--lo-text); }
.lo-tabs button:focus-visible {
  color: var(--lo-cyan);
  outline: none;
  box-shadow: inset 0 0 0 2px rgba(19, 216, 231, 0.35);
}
.lo-tabs button.active {
  color: var(--lo-cyan);
  border-bottom-color: var(--lo-cyan);
}

.lo-workspace {
  min-height: 0;
  height: 100%;
  overflow: hidden;
  display: grid;
}

.lo-empty {
  display: grid;
  place-items: center;
  color: var(--lo-muted);
  font-size: 0.78rem;
  font-weight: 700;
  min-height: 80px;
}

/* —— Overview —— */
.lo-ov {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto auto auto;
  gap: 10px;
}

.lo-forecast {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 6px;
  min-height: 0;
  min-width: 0;
  height: 100%;
}

.lo-forecast-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.lo-forecast-head h2 {
  margin: 0;
  font-size: 0.88rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--lo-gold);
}

.lo-forecast-meta {
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
}

.lo-forecast-meta .now {
  font-size: 1.45rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  color: #fff;
  line-height: 1;
}

.lo-forecast-meta .arrow { color: var(--lo-muted); font-size: 0.9rem; }

.lo-forecast-meta .next {
  font-size: 1.45rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  color: var(--lo-gold);
  line-height: 1;
}

.lo-forecast-meta .delta {
  font-size: 0.78rem;
  font-weight: 800;
  color: var(--lo-green);
}
.lo-forecast-meta .delta.neg { color: var(--lo-red); }

.lo-chart {
  position: relative;
  height: 100%;
  min-height: 160px;
  min-width: 0;
}
.lo-chart svg { width: 100%; height: 100%; display: block; overflow: visible; }
.lo-chart-band { fill: rgba(233, 168, 60, 0.12); }
.lo-chart-line { fill: none; stroke: var(--lo-gold); stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
.lo-chart-dot { fill: var(--lo-gold); }
.lo-chart-grid { stroke: rgba(156, 218, 236, 0.08); stroke-width: 1; }
.lo-chart-label {
  fill: var(--lo-muted);
  font-size: 11px;
  font-weight: 700;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}
.lo-chart-value {
  fill: var(--lo-text);
  font-size: 11px;
  font-weight: 800;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

.lo-confidence-text {
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--lo-muted);
}

.lo-summary-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.lo-summary {
  display: grid;
  gap: 2px;
  min-width: 0;
}
.lo-summary span {
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--lo-muted);
}
.lo-summary b {
  font-size: 1.35rem;
  font-weight: 900;
  color: #fff;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}
.lo-summary b.gold { color: var(--lo-gold); }

.lo-sentiment {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  min-height: 28px;
  padding: 6px 0;
  border-top: 1px solid var(--lo-line);
  border-bottom: 1px solid var(--lo-line);
}
.lo-sentiment-label {
  font-size: 0.76rem;
  font-weight: 800;
  color: var(--lo-muted);
}
.lo-sentiment-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.76rem;
  font-weight: 700;
}
.lo-sentiment-item em {
  font-style: normal;
  color: var(--lo-muted);
}
.lo-sentiment-item b { color: var(--lo-text); }
.lo-sentiment-empty {
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--lo-muted);
}

.lo-priority {
  display: grid;
  grid-template-rows: auto auto;
  gap: 0;
  min-height: 0;
}
.lo-priority-title {
  margin: 0 0 2px;
  padding-bottom: 3px;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lo-muted);
  border-bottom: 1px solid var(--lo-line);
}
/* League docket: one filing line per decision, not three tall cards. */
.lo-priority-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0;
  min-height: 0;
  align-content: start;
}
.lo-priority-card {
  display: grid;
  grid-template-columns: minmax(120px, 176px) minmax(88px, 116px) minmax(0, 1fr) auto;
  align-items: baseline;
  gap: 12px;
  padding: 7px 10px 7px 9px;
  border-top: 0;
  border-bottom: 1px solid var(--lo-line);
  border-left: 2px solid var(--lo-gold);
  border-radius: 0;
  background: transparent;
  text-align: left;
  color: inherit;
  cursor: pointer;
  min-width: 0;
  align-content: start;
}
.lo-priority-card:hover,
.lo-priority-card:focus-visible {
  border-left-color: var(--lo-cyan);
  outline: none;
  background: rgba(19, 216, 231, 0.06);
}
.lo-priority-card strong {
  font-size: 0.82rem;
  font-weight: 800;
  color: #fff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lo-priority-meta {
  display: flex;
  justify-content: flex-start;
  gap: 8px;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}
.lo-priority-card p {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
  line-height: 1.35;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lo-priority-card .state {
  justify-self: end;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--lo-gold);
}

.lo-state-stable { color: var(--lo-muted) !important; }
.lo-state-monitoring { color: var(--lo-yellow) !important; }
.lo-state-negotiating { color: var(--lo-orange) !important; }
.lo-state-vote { color: var(--lo-gold) !important; }

/* —— CBA —— */
.lo-cba {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 360px);
  gap: 14px;
}

.lo-cba-left {
  min-height: 0;
  height: 100%;
  overflow: hidden;
  display: grid;
  grid-template-rows: minmax(120px, 0.42fr) minmax(0, 1fr);
  gap: 12px;
}

.lo-cba-panel {
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}

/* Department signature: the filing registry. Section titles are league
   document headings, opened by a registry mark rather than an icon. */
.lo-section-title {
  position: relative;
  margin: 0 0 6px;
  padding-left: 16px;
  font-size: 0.84rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--lo-muted);
}

.lo-section-title::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0.18em;
  bottom: 0.18em;
  width: 6px;
  border: 1px solid var(--lo-gold, #e9a83c);
  border-right: 0;
  opacity: 0.8;
}

.lo-timeline-scroll {
  min-height: 0;
  height: 100%;
  overflow: auto;
  padding-right: 4px;
}

.lo-timeline {
  display: grid;
  gap: 0;
  position: relative;
  padding-left: 14px;
}
.lo-timeline::before {
  content: "";
  position: absolute;
  left: 3px;
  top: 6px;
  bottom: 6px;
  width: 1px;
  background: rgba(156, 218, 236, 0.18);
}
.lo-timeline-item {
  position: relative;
  padding: 0 0 12px 12px;
}
.lo-timeline-item:last-child { padding-bottom: 0; }
.lo-timeline-item::before {
  content: "";
  position: absolute;
  left: -14px;
  top: 5px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--lo-muted);
  box-shadow: 0 0 0 3px rgba(7, 20, 32, 1);
}
.lo-timeline-item.active::before { background: var(--lo-gold); }
.lo-timeline-item.warn::before { background: var(--lo-orange); }
.lo-timeline-item span {
  display: block;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
  margin-bottom: 2px;
}
.lo-timeline-item b {
  display: block;
  font-size: 0.8rem;
  font-weight: 800;
  color: #fff;
}
.lo-timeline-item em {
  display: block;
  margin-top: 2px;
  font-style: normal;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}

.lo-issues-scroll {
  min-height: 0;
  overflow: auto;
  display: grid;
  gap: 4px;
  align-content: start;
  padding-right: 4px;
}

.lo-issue {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 4px 10px;
  padding: 10px 12px;
  border-radius: var(--radius-control, 6px);
  cursor: pointer;
  text-align: left;
  border-top: 1px solid transparent;
  border-bottom: 1px solid var(--lo-line);
  background: transparent;
  color: inherit;
  width: 100%;
}
.lo-issue:hover { background: rgba(19, 216, 231, 0.05); border-bottom-color: var(--lo-cyan); }
.lo-issue:focus-visible {
  outline: none;
  border-color: var(--lo-cyan);
  box-shadow: 0 0 0 2px rgba(19, 216, 231, 0.3);
}
.lo-issue.selected {
  background: rgba(19, 216, 231, 0.18);
  border-color: var(--lo-cyan);
  box-shadow: inset 3px 0 0 var(--lo-cyan);
}
.lo-issue-title {
  font-size: 0.8rem;
  font-weight: 800;
  color: #fff;
}
.lo-issue-scale {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}
.lo-issue-state {
  grid-column: 2;
  grid-row: 1 / span 2;
  align-self: center;
  font-size: 0.72rem;
  font-weight: 800;
  white-space: nowrap;
  color: var(--lo-muted);
}

.lo-inspector {
  min-height: 0;
  height: 100%;
  overflow: auto;
  display: grid;
  gap: 10px;
  align-content: start;
  padding: 12px 14px;
  border-top: 1px solid var(--lo-line);
  border-bottom: 1px solid var(--lo-line);
  border-radius: 0;
  background: rgba(6, 21, 34, 0.55);
}
.lo-inspector-mark {
  font-size: 0.6875rem;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--lo-muted);
}
.lo-inspector-empty {
  color: var(--lo-muted);
  font-size: 0.74rem;
  font-weight: 700;
}
.lo-inspector h4 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 900;
  color: #fff;
}
.lo-inspector-kicker {
  font-size: 0.74rem;
  font-weight: 800;
  color: var(--lo-cyan);
}
.lo-inspector-rows { display: grid; gap: 8px; }
.lo-inspector-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 0.74rem;
}
.lo-inspector-row span { color: var(--lo-muted); font-weight: 700; }
.lo-inspector-row b { font-weight: 800; color: #fff; }
.lo-inspector-note {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
  line-height: 1.4;
}

.lo-support-bars { display: grid; gap: 10px; margin-top: 4px; }
.lo-support-row { display: grid; gap: 4px; }
.lo-support-top {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}
.lo-support-top b { color: var(--lo-text); font-weight: 800; }
/* Support is filed against a ruled register, matching league documents. */
.lo-support-track {
  height: 8px;
  border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(255, 255, 255, 0.06);
  overflow: hidden;
}
.lo-support-track > i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--lo-cyan);
}
.lo-support-track.owners > i { background: var(--lo-gold); }

/* —— Markets —— */
.lo-markets {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 8px;
}

.lo-markets-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.lo-filters {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.lo-filters button {
  height: 30px;
  padding: 0 12px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--lo-muted);
  font-size: 0.72rem;
  font-weight: 800;
  cursor: pointer;
}
.lo-filters button:hover { color: var(--lo-text); background: rgba(255, 255, 255, 0.06); }
.lo-filters button:focus-visible {
  outline: none;
  border-color: var(--lo-cyan);
  box-shadow: 0 0 0 2px rgba(19, 216, 231, 0.25);
}
.lo-filters button.active {
  color: var(--lo-cyan);
  border-color: rgba(19, 216, 231, 0.45);
  background: rgba(19, 216, 231, 0.1);
}

.lo-markets-body {
  min-height: 0;
  height: 100%;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 12px;
}

.lo-table-wrap {
  min-height: 0;
  height: 100%;
  overflow: auto;
  border: 1px solid var(--lo-line);
  border-radius: 8px;
}

.lo-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
.lo-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #0a1a28;
}
.lo-table th,
.lo-table td {
  padding: 9px 10px;
  text-align: left;
  vertical-align: middle;
  border-bottom: 1px solid rgba(156, 218, 236, 0.06);
}
.lo-table th {
  font-size: 0.74rem;
  font-weight: 800;
  color: var(--lo-muted);
  border-bottom-color: rgba(156, 218, 236, 0.14);
  white-space: nowrap;
}
.lo-table th button {
  all: unset;
  cursor: pointer;
  font: inherit;
  color: inherit;
  letter-spacing: inherit;
  text-transform: inherit;
}
.lo-table th button:hover,
.lo-table th button:focus-visible { color: var(--lo-cyan); outline: none; }
.lo-table th .sorted { color: var(--lo-cyan); }

.lo-table td {
  font-size: 0.76rem;
  font-weight: 700;
}
.lo-table tbody tr {
  cursor: pointer;
  transition: background 0.12s ease;
}
.lo-table tbody tr:hover { background: rgba(255, 255, 255, 0.03); }
.lo-table tbody tr:focus-visible {
  outline: 2px solid var(--lo-cyan);
  outline-offset: -2px;
}
.lo-table tr.selected { background: rgba(19, 216, 231, 0.1); }
.lo-table tr.threatened { background: rgba(255, 96, 109, 0.06); }
.lo-table tr.threatened.selected { background: rgba(19, 216, 231, 0.12); }
.lo-table tr.user td:first-child { box-shadow: inset 2px 0 0 var(--lo-gold); }

.lo-col-team { width: 28%; }
.lo-col-rev { width: 16%; text-align: right !important; font-variant-numeric: tabular-nums; color: var(--lo-gold); }
.lo-col-trend { width: 16%; text-align: right !important; font-variant-numeric: tabular-nums; }
.lo-col-market { width: 18%; }
.lo-col-risk { width: 18%; }
.lo-table th.lo-col-rev,
.lo-table th.lo-col-trend { text-align: right; }

.lo-team-cell {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.lo-team-cell img { width: 22px; height: 22px; object-fit: contain; flex-shrink: 0; }
.lo-team-cell .abbr-fallback {
  width: 22px;
  height: 22px;
  display: grid;
  place-items: center;
  font-size: 0.6875rem;
  font-weight: 900;
  color: var(--lo-muted);
  flex-shrink: 0;
}
.lo-team-cell strong {
  font-weight: 800;
  white-space: nowrap;
}
.lo-event-mark {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--lo-gold);
}
.lo-event-mark.risk { background: var(--lo-red); }
.lo-event-mark.rev { background: var(--lo-cyan); }

.lo-trend.up { color: var(--lo-green); }
.lo-trend.down { color: var(--lo-red); }
.lo-trend.flat { color: var(--lo-muted); }

.lo-risk-high { color: var(--lo-red); font-weight: 900; }
.lo-risk-med { color: var(--lo-yellow); font-weight: 800; }
.lo-risk-low { color: var(--lo-muted); font-weight: 700; }

.lo-dossier {
  min-height: 0;
  overflow: auto;
}

/* —— Franchise Risk —— */
.lo-risk {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  grid-template-rows: minmax(0, 1fr);
  gap: 12px;
}
.lo-risk.has-detail {
  grid-template-columns: repeat(3, minmax(0, 1fr)) minmax(260px, 320px);
  grid-template-rows: minmax(0, 1fr);
}

.lo-risk-col {
  min-height: 0;
  height: 100%;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 6px;
  border: 1px solid var(--lo-line);
  border-radius: 10px;
  padding: 10px;
  background: rgba(8, 22, 34, 0.4);
}

.lo-risk-col h3 {
  margin: 0;
  font-size: 0.84rem;
  font-weight: 800;
  color: var(--lo-muted);
}
.lo-risk-col.immediate h3 { color: var(--lo-red); }
.lo-risk-col.monitor h3 { color: var(--lo-yellow); }
.lo-risk-col.stable h3 { color: var(--lo-cyan); }

.lo-risk-list {
  min-height: 0;
  overflow: auto;
  display: grid;
  gap: 6px;
  align-content: start;
  padding-right: 2px;
}
.lo-risk-list .lo-inspector-empty {
  min-height: 120px;
  height: 100%;
  display: grid;
  place-items: center;
}

.lo-risk-card {
  display: grid;
  gap: 4px;
  padding: 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.03);
  text-align: left;
  color: inherit;
  cursor: pointer;
  width: 100%;
}
.lo-risk-card:hover,
.lo-risk-card:focus-visible {
  border-color: var(--lo-cyan);
  outline: none;
  background: rgba(19, 216, 231, 0.08);
}
.lo-risk-card.selected {
  border-color: var(--lo-cyan);
  background: rgba(19, 216, 231, 0.12);
}
.lo-risk-card-top {
  display: flex;
  align-items: center;
  gap: 8px;
}
.lo-risk-card-top strong {
  font-size: 0.84rem;
  font-weight: 800;
  color: #fff;
}
.lo-risk-card-meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--lo-muted);
}
.lo-risk-detail {
  min-height: 0;
  height: 100%;
  overflow: hidden;
}

@media (max-width: 1100px) {
  .lo-cba { grid-template-columns: 1fr; }
  .lo-markets-body { grid-template-columns: 1fr; }
  .lo-risk { grid-template-columns: 1fr; }
  .lo-priority-list { grid-template-columns: 1fr; }
  .lo-summary-row { grid-template-columns: 1fr; }
}
`;

function InlineStyles() {
  return <style>{LO_STYLES}</style>;
}

function asObject(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? v : EMPTY_OBJ;
}

function asArray(v) {
  return Array.isArray(v) ? v : EMPTY_ARR;
}

function num(v, fb = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fb;
}

function fmtMoneyM(v) {
  return `$${Math.round(num(v, 0))}M`;
}

function fmtCap(v) {
  return `$${num(v, 0).toFixed(1)}M`;
}

function riskClass(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("critical") || s.includes("high")) return "lo-risk-high";
  if (s.includes("med")) return "lo-risk-med";
  return "lo-risk-low";
}

function isThreatened(team) {
  const score = num(team.relocation_risk, 0);
  const label = String(team.relocation_risk_label || team.risk_label || "").toLowerCase();
  return score >= 0.55 || label.includes("critical") || label.includes("high");
}

function deriveDisplayStatus(team) {
  const profit = num(team.profit, 0);
  const risk = num(team.relocation_risk, 0);
  if (profit >= 18) return "Surge";
  if (profit >= 8) return "Profit";
  if (profit >= 2) return "Thin";
  if (profit >= -2) return "Even";
  if (risk >= 0.55 || profit < -10) return "Risk";
  if (profit < -2) return "Loss";
  return "Flat";
}

function deriveRelocReason(row) {
  const pressure = String(row.pressure || row.market_pressure || "").trim();
  const reason = String(row.reason || "").trim();
  if (reason && reason !== "Market" && reason !== "Stable") return reason;
  if (pressure && pressure !== "Stable") {
    const map = {
      Gate: "Gate",
      Arena: "Arena",
      Revenue: "Revenue",
      "Small Market": "Small Market",
      Owner: "Owner",
      Attendance: "Fan Drop",
      Lease: "Lease",
    };
    return map[pressure] || pressure;
  }
  const tags = asArray(row.reason_tags);
  if (tags.includes("Arena Drag")) return "Arena";
  if (tags.includes("Weak Gate")) return "Gate";
  if (tags.includes("Fan Freeze")) return "Fan Drop";
  if (tags.includes("Small Loss")) return "Revenue";
  if (row.market_tier === "Small") return "Small Market";
  return reason || "Lease";
}

function deriveCapDrivers(data) {
  const existing = asArray(data.cap_drivers);
  if (existing.length) return existing;

  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const smallBe = num(data.small_market_break_even, 0.5);
  const losing = num(data.losing_teams_count, 0);
  const cbaP = num(asObject(data.cba).pressure, 0.3);
  const teams = asArray(data.teams);
  const avgFan = teams.reduce((s, t) => s + num(t.fan_sentiment, 55), 0) / Math.max(teams.length, 1);

  return [
    { label: "Escrow", sign: escrow >= 1 ? "+" : "-" },
    { label: "Revenue", sign: health >= 0.55 ? "+" : "-" },
    { label: "Small Teams", sign: smallBe >= 0.65 ? "+" : "-" },
    { label: "Loss Teams", sign: losing <= 10 ? "+" : "-" },
    { label: "CBA", sign: cbaP < 0.55 ? "+" : "-" },
    { label: "Fan Heat", sign: avgFan >= 52 ? "+" : "-" },
  ];
}

function deriveOwnerMood(data) {
  const m = asObject(data.owner_mood);
  if (m.owners && m.owners !== "—") return m;

  const capType = String(data.cap_change_type || "");
  const health = num(data.revenue_health, 0.5);
  const cbaP = num(asObject(data.cba).pressure, 0.3);
  const losing = num(data.losing_teams_count, 0);

  let owners = "Calm";
  if (cbaP >= 0.65) owners = "Cautious";
  else if (cbaP >= 0.45) owners = "Watchful";

  let players = "Stable";
  if (capType.includes("Drop") || capType.includes("Freeze")) players = "Angry";
  else if (capType.includes("Jump")) players = "Pushy";
  else if (cbaP >= 0.5) players = "Restless";

  let fans = "Stable";
  if (health >= 0.62) fans = "Strong";
  else if (health < 0.42) fans = "Cold";

  let media = "Quiet";
  if (losing >= 18 || cbaP >= 0.6) media = "Loud";
  else if (cbaP >= 0.4) media = "Watching";

  return { owners, players, fans, media };
}

function moodMeterValue(label) {
  const s = String(label || "").toLowerCase();
  if (["angry", "loud", "cold", "cautious"].some((k) => s.includes(k))) return 28;
  if (["restless", "watchful", "watching", "pushy", "cool"].some((k) => s.includes(k))) return 48;
  if (["strong", "hot", "calm"].some((k) => s.includes(k))) return 82;
  if (["stable", "quiet", "warm"].some((k) => s.includes(k))) return 68;
  return 55;
}

function moodMeterColor(value) {
  if (value < 40) return "var(--lo-red)";
  if (value < 55) return "var(--lo-orange)";
  if (value < 72) return "var(--lo-cyan)";
  return "var(--lo-green)";
}

function deriveTeamEvent(team) {
  if (isThreatened(team)) {
    return { kind: "risk", label: "Relocation" };
  }
  const stars = asArray(team.superstar_tags);
  if (stars.length) {
    return { kind: "star", label: "Star" };
  }
  const yoy = num(team.revenue_yoy_delta, 0);
  const dir = String(team.revenue_yoy_direction || "");
  if (
    Math.abs(yoy) >= 8
    || (dir === "up" && Math.abs(yoy) >= 5)
    || (dir === "down" && Math.abs(yoy) >= 5)
  ) {
    return { kind: "rev", label: (yoy >= 0 || dir === "up") ? "Spike" : "Drop" };
  }
  return null;
}

function normalizeTeams(rawTeams) {
  const sorted = [...asArray(rawTeams)].sort((a, b) => num(b.revenue, 0) - num(a.revenue, 0));
  const maxRev = Math.max(...sorted.map((t) => num(t.revenue, 0)), 1);

  return sorted.map((t, index) => {
    const rank = num(t.rank, 0) > 0 ? num(t.rank, 0) : index + 1;
    const riskScore = num(t.relocation_risk, 0);
    const riskLabel = t.relocation_risk_label || (riskScore >= 0.55 ? "High" : riskScore >= 0.35 ? "Med" : "Low");
    return {
      ...t,
      rank,
      display_status: t.revenue_status || deriveDisplayStatus(t),
      revenue_bar_pct: num(t.revenue_bar_pct, 0) > 0 ? num(t.revenue_bar_pct, 0) : (num(t.revenue, 0) / maxRev) * 100,
      relocation_risk: riskScore,
      relocation_risk_label: riskLabel,
      reloc_reason: deriveRelocReason(t),
      threatened: isThreatened({ ...t, relocation_risk: riskScore, relocation_risk_label: riskLabel }),
      event: deriveTeamEvent({ ...t, relocation_risk: riskScore, relocation_risk_label: riskLabel }),
    };
  });
}

function synthesizeCapSeries(data) {
  const existing =
    asArray(data.cap_forecast).length
      ? asArray(data.cap_forecast)
      : asArray(data.cap_history).length
        ? asArray(data.cap_history)
        : asArray(data.projected_seasons);

  if (existing.length >= 3) {
    return existing.slice(0, 6).map((row, i) => {
      const year = num(row.year ?? row.season ?? row.season_year, num(data.current_year, 2025) + i);
      const cap = num(row.cap ?? row.salary_cap ?? row.value ?? row.projected, num(data.salary_cap, 88));
      const low = num(row.low ?? row.cap_low, cap * 0.97);
      const high = num(row.high ?? row.cap_high, cap * 1.03);
      return { year, label: String(year), cap, low, high };
    });
  }

  const currentYear = num(data.current_year, 2025);
  const current = num(data.salary_cap ?? data.cap ?? data.current_cap, 88);
  const projected = num(data.projected_salary_cap, current + num(data.cap_change, 0));
  const change = projected - current;
  const type = String(data.cap_change_type || "");
  let growth = change;
  if (Math.abs(growth) < 0.05) {
    if (type.includes("Jump")) growth = current * 0.05;
    else if (type.includes("Rise")) growth = current * 0.025;
    else if (type.includes("Drop")) growth = -current * 0.015;
    else growth = current * 0.01;
  }

  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const uncertainty = Math.max(0.8, (Math.abs(1 - escrow) * 8 + Math.abs(0.55 - health) * 10 + 1.2));

  const series = [];
  for (let i = 0; i < 5; i++) {
    const year = currentYear + i;
    const decay = Math.pow(0.92, i);
    const cap = i === 0 ? current : current + growth * i * (0.85 + decay * 0.2);
    const band = uncertainty * (0.7 + i * 0.35);
    series.push({
      year,
      label: String(year),
      cap: Math.round(cap * 10) / 10,
      low: Math.round((cap - band) * 10) / 10,
      high: Math.round((cap + band) * 10) / 10,
    });
  }

  if (series.length > 1) {
    series[1].cap = Math.round(projected * 10) / 10;
    series[1].low = Math.round((projected - uncertainty) * 10) / 10;
    series[1].high = Math.round((projected + uncertainty) * 10) / 10;
  }

  return series;
}

function deriveConfidence(data, series) {
  const next = series[1] || series[0] || { low: 0, high: 0, cap: 0 };
  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const width = Math.max(0.5, next.high - next.low);
  const mid = next.cap || (next.low + next.high) / 2;
  const confidencePct = Math.round(Math.max(40, Math.min(92, 88 - Math.abs(1 - escrow) * 35 - Math.abs(0.55 - health) * 40)));
  const confidenceLevel = confidencePct >= 75 ? "High" : confidencePct >= 55 ? "Medium" : "Low";
  return {
    low: next.low,
    high: next.high,
    mid,
    width,
    label: `${fmtCap(next.low)} – ${fmtCap(next.high)}`,
    confidencePct,
    confidenceLevel,
    rangeLabel: `Projected Range ${fmtCap(next.low)}–${fmtCap(next.high)} · ${confidenceLevel} Confidence`,
  };
}

function assignActionState(row) {
  const status = String(row.status || row.state || "").toLowerCase();
  const likelihood = num(row.likelihood ?? row.support_pct, 0);
  const owners = num(row.owner_support, 0);
  const players = num(row.player_support, 0);
  const gap = Math.abs(owners - players);
  const voteReady = likelihood >= 72 && gap <= 12;

  if (status.includes("vote") || voteReady) return "Vote Required";
  if (status === "open" || status.includes("negotiat")) return "Negotiating";
  if (status === "likely" || status.includes("monitor")) return "Monitoring";
  if (status === "long") return "Stable";
  if (gap >= 18) return "Negotiating";
  if (likelihood >= 48) return "Monitoring";
  if (ACTION_STATES.includes(row.status)) return row.status;
  return "Stable";
}

function issueUrgency(issue) {
  const stateRank = { "Vote Required": 4, Negotiating: 3, Monitoring: 2, Stable: 1 };
  const impact = num(issue.likelihood, 0) * 0.55 + num(issue.support_pct, 0) * 0.25 + (stateRank[issue.action_state] || 1) * 18;
  return impact;
}

function clipWords(text, max = 15) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "";
  return words.slice(0, max).join(" ");
}

function issueConsequence(issue) {
  const note = String(issue.consequence || issue.impact || issue.summary || issue.detail || "").trim();
  if (note) return clipWords(note, 15);

  const status = String(issue.status || issue.action_state || "Open");
  const likelihood = num(issue.likelihood, 0);
  const owners = num(issue.owner_support, 0);
  const players = num(issue.player_support, 0);
  const name = String(issue.name || "Issue");

  if (owners || players) {
    const lean =
      owners > players + 6
        ? "Owners currently lead."
        : players > owners + 6
          ? "Players currently lead."
          : "Sides are nearly even.";
    return clipWords(`${name}: ${status} at ${likelihood}%. ${lean}`, 15);
  }

  if (status === "Vote Required") return clipWords(`${name} may reshape next-season costs.`, 15);
  if (status === "Negotiating" || status === "Open") return clipWords(`${name} talks can shift cap room.`, 15);
  if (status === "Monitoring" || status === "Likely") return clipWords(`${name} is ${likelihood}% likely this cycle.`, 15);
  return clipWords(`${name} sits on a longer runway.`, 15);
}

function issueDeadline(issue) {
  const explicit = issue.deadline || issue.when || issue.window;
  if (explicit) return String(explicit).slice(0, 16);
  const status = String(issue.status || "").toLowerCase();
  if (status === "likely") return "Near term";
  if (status === "open") return "Active window";
  if (status === "long") return "Long runway";
  if (issue.bargaining_deadline) return String(issue.bargaining_deadline).slice(0, 10);
  return "Ongoing";
}

function deriveActiveIssues(data) {
  const cba = asObject(data.cba);
  const bargainingDeadline = cba.bargaining_deadline || null;
  const fromNeg = asArray(cba.potential_changes).map((row) => ({
    ...row,
    id: `cba-${row.name}`,
    source: "CBA",
    kind: "issue",
    bargaining_deadline: bargainingDeadline,
  }));
  const fromRules = asArray(data.rule_changes).map((row) => ({
    ...row,
    id: `rule-${row.name}`,
    source: "Rule",
    kind: "issue",
    bargaining_deadline: bargainingDeadline,
  }));

  const merged = new Map();
  [...fromNeg, ...fromRules].forEach((row) => {
    const key = String(row.name || row.id);
    if (!merged.has(key)) merged.set(key, row);
  });

  return [...merged.values()]
    .map((row) => {
      const action_state = assignActionState(row);
      const owners = num(row.owner_support, 0);
      const players = num(row.player_support, 0);
      const support = num(row.support_pct ?? row.likelihood, Math.round((owners + players) / 2));
      const enriched = {
        ...row,
        action_state,
        support_pct: support,
        scale_label:
          owners || players
            ? `Owners ${owners}% · Players ${players}%`
            : `Support ${support}%`,
      };
      return {
        ...enriched,
        deadline: issueDeadline(enriched),
        consequence: issueConsequence(enriched),
      };
    })
    .sort((a, b) => issueUrgency(b) - issueUrgency(a))
    .slice(0, 5);
}

function deriveCbaTimeline(data) {
  const cba = asObject(data.cba);
  const year = num(data.current_year, 2025);
  const endYear = num(cba.end_year, year + num(cba.years_remaining, 3));
  const startYear = num(cba.start_year, endYear - 7);
  const deadline = String(cba.bargaining_deadline || `${endYear - 1}-06-30`).slice(0, 10);
  const yearsLeft = num(cba.years_remaining, Math.max(0, endYear - year));
  const pressure = String(cba.pressure_level || "Low");
  const items = [
    {
      id: "start",
      when: String(startYear),
      title: cba.current_agreement || `CBA ${startYear}–${endYear}`,
      detail: "Current agreement in force",
      tone: "",
    },
    {
      id: "now",
      when: `Season ${year}`,
      title: "Active term",
      detail: `${yearsLeft} year${yearsLeft === 1 ? "" : "s"} remaining · Pressure ${pressure}`,
      tone: yearsLeft <= 2 || pressure === "High" ? "warn" : "active",
    },
    {
      id: "talks",
      when: deadline,
      title: "Negotiation window",
      detail: yearsLeft <= 2 ? "Formal talks expected" : "Milestone: bargaining opens",
      tone: yearsLeft <= 2 ? "warn" : "",
    },
    {
      id: "expiry",
      when: String(endYear),
      title: "Agreement expiry",
      detail: `Ends after ${endYear - 1}–${endYear} season`,
      tone: yearsLeft <= 1 ? "warn" : "",
    },
  ];

  const relevantRules = asArray(cba.key_rules).filter((rule) => {
    const r = String(rule).toLowerCase();
    const pressureHot = num(cba.pressure, 0) >= 0.55;
    if (!pressureHot) return false;
    return r.includes("escrow") || r.includes("cap") || r.includes("share") || r.includes("tax");
  });

  if (relevantRules.length) {
    items.push({
      id: "rule-shift",
      when: "Rule shift",
      title: relevantRules[0],
      detail: "Elevated under current CBA pressure",
      tone: "active",
    });
  }

  return items;
}

function normalizeLeagueOps(raw) {
  const data = asObject(raw);
  if (!Object.keys(data).length) return EMPTY_OBJ;

  const teams = normalizeTeams(data.teams);
  const cap_forecast = synthesizeCapSeries(data);
  const confidence = deriveConfidence(data, cap_forecast);

  return {
    ...data,
    teams,
    salary_cap: num(data.salary_cap ?? data.cap?.salary_cap, 0),
    projected_salary_cap: num(data.projected_salary_cap ?? data.cap?.projected_salary_cap, 0),
    cap_change: num(data.cap_change ?? data.cap?.cap_change, 0),
    cap_change_type: data.cap_change_type || data.cap?.cap_change_type || "Flat Cap",
    cap_drivers: deriveCapDrivers({ ...data, teams }),
    owner_mood: deriveOwnerMood(data),
    league_status:
      data.league_health_label
      || asObject(data.relocation).league_stability
      || data.league_status
      || data.cap_change_type
      || "Stable",
    cap_forecast,
    confidence,
    active_issues: deriveActiveIssues(data),
    cba_timeline: deriveCbaTimeline(data),
  };
}

function TeamLogo({ team, abbr, size = 22 }) {
  const url = resolveFranchiseTeamLogo(team, abbr);
  if (url) return <img src={url} alt="" style={{ width: size, height: size, objectFit: "contain" }} />;
  return <span className="abbr-fallback">{String(abbr || "?").slice(0, 3)}</span>;
}

function chartAxisLabel(points, i) {
  if (!points.length) return "";
  if (i === 0) return "Current";
  if (i === 1) return "Next Season";
  if (i === points.length - 1) return "Long-Term";
  return "";
}

function CapForecastChart({ series }) {
  const points = asArray(series);
  if (!points.length) return <div className="lo-empty">No forecast</div>;

  const padL = 44;
  // Reserve room for the final season's value label so it is not cut off at
  // the right edge of the forecast frame.
  const padR = 44;
  const padT = 22;
  const padB = 30;
  const w = 720;
  const h = 220;
  const caps = points.flatMap((p) => [p.low, p.high, p.cap]);
  const minY = Math.min(...caps) - 1;
  const maxY = Math.max(...caps) + 1;
  const span = Math.max(maxY - minY, 1);

  const xAt = (i) => padL + (i * (w - padL - padR)) / Math.max(points.length - 1, 1);
  const yAt = (v) => padT + ((maxY - v) / span) * (h - padT - padB);

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i)} ${yAt(p.cap)}`).join(" ");
  const band = [
    ...points.map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i)} ${yAt(p.high)}`),
    ...[...points].reverse().map((p, i) => {
      const idx = points.length - 1 - i;
      return `L ${xAt(idx)} ${yAt(p.low)}`;
    }),
    "Z",
  ].join(" ");

  const ticks = [minY + span * 0.25, minY + span * 0.5, minY + span * 0.75];

  return (
    <div className="lo-chart" aria-label="Projected salary cap by season">
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
        {ticks.map((t) => (
          <line key={t} className="lo-chart-grid" x1={padL} x2={w - padR} y1={yAt(t)} y2={yAt(t)} />
        ))}
        <path className="lo-chart-band" d={band} />
        <path className="lo-chart-line" d={line} />
        {points.map((p, i) => {
          const axis = chartAxisLabel(points, i);
          const anchor = i === 0 ? "start" : i === points.length - 1 ? "end" : "middle";
          return (
            <g key={p.year}>
              <circle className="lo-chart-dot" cx={xAt(i)} cy={yAt(p.cap)} r={axis ? 4 : 2.6} />
              {axis ? (
                <>
                  <text className="lo-chart-label" x={xAt(i)} y={h - 8} textAnchor={anchor}>
                    {axis}
                  </text>
                  <text className="lo-chart-value" x={xAt(i)} y={yAt(p.cap) - 10} textAnchor={anchor}>
                    {fmtCap(p.cap)}
                  </text>
                </>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function stateClass(actionState) {
  if (actionState === "Vote Required") return "lo-state-vote";
  const s = String(actionState || "").toLowerCase();
  if (s === "monitoring") return "lo-state-monitoring";
  if (s === "negotiating") return "lo-state-negotiating";
  if (s === "stable") return "lo-state-stable";
  return "";
}

function unusualMoods(mood) {
  const m = asObject(mood);
  return [
    ["Owners", m.owners],
    ["Players", m.players],
    ["Fans", m.fans],
    ["Media", m.media],
  ].filter(([, value]) => {
    const s = String(value || "").toLowerCase();
    return s && !CALM_MOODS.has(s);
  });
}

function YourImpactStrip({ user }) {
  const u = asObject(user);
  const risk = String(u.boycott_risk || u.relocation_risk_label || "Low");
  const riskLower = risk.toLowerCase();
  const riskTone = riskLower.includes("high") || riskLower.includes("critical") ? "warn" : riskLower.includes("med") ? "" : "ok";
  const influence =
    num(u.cap_contribution_m, 0) !== 0
      ? `${num(u.cap_contribution_m, 0) >= 0 ? "+" : ""}${num(u.cap_contribution_m, 0).toFixed(1)}M`
      : u.market_tier || u.market_health || "—";

  return (
    <div className="lo-impact-strip" title="Your Impact">
      <TeamLogo team={u} abbr={u.abbreviation || u.name} size={22} />
      <div className="lo-impact-pill">
        <span>Revenue</span>
        <b>{fmtMoneyM(u.revenue)}</b>
      </div>
      <div className="lo-impact-pill">
        <span>Influence</span>
        <b>{influence}</b>
      </div>
      <div className="lo-impact-pill">
        <span>Warning</span>
        <b className={riskTone}>{risk}</b>
      </div>
    </div>
  );
}

function TeamDossier({ team }) {
  const t = asObject(team);
  if (!Object.keys(t).length) {
    return (
      <div className="lo-inspector lo-dossier">
        <div className="lo-inspector-empty">Select a team for market detail</div>
      </div>
    );
  }

  const yoy = num(t.revenue_yoy_delta, 0);
  const yoyDir = t.revenue_yoy_direction || "flat";
  const prior = Math.max(0, num(t.revenue, 0) - yoy);
  const attendance = num(t.attendance_rate, 0);
  const fan = num(t.fan_sentiment, 0);
  const profit = num(t.profit, 0);
  const tags = asArray(t.reason_tags);
  const ownership =
    t.owner_mood
    || t.ownership
    || (tags.find((tag) => /owner|boycott|patience/i.test(String(tag))) || null)
    || (String(t.market_pressure || "") === "Owner" ? "Owner pressure" : null)
    || (fan < 42 ? "Fan unrest" : fan >= 68 ? "Hot gate" : "Steady");
  const arena = t.reloc_reason || t.market_pressure || t.pressure || "Stable";

  return (
    <div className="lo-inspector lo-dossier">
      <span className="lo-inspector-kicker">Market Dossier</span>
      <h4>
        {t.abbreviation || t.name} · {t.market_tier || "Market"}
      </h4>
      <div className="lo-inspector-rows">
        <div className="lo-inspector-row">
          <span>Prior revenue</span>
          <b style={{ color: "var(--lo-gold)" }}>{fmtMoneyM(prior)}</b>
        </div>
        <div className="lo-inspector-row">
          <span>Current revenue</span>
          <b style={{ color: "var(--lo-gold)" }}>{fmtMoneyM(t.revenue)}</b>
        </div>
        <div className="lo-inspector-row">
          <span>YoY change</span>
          <b className={`lo-trend ${yoyDir === "up" || yoy > 0 ? "up" : yoyDir === "down" || yoy < 0 ? "down" : "flat"}`}>
            {yoyDir === "flat" && Math.abs(yoy) < 1
              ? "Flat"
              : `${yoy >= 0 ? "+" : ""}${Math.round(yoy)}M`}
          </b>
        </div>
        <div className="lo-inspector-row">
          <span>Profit</span>
          <b className={`lo-trend ${profit > 0 ? "up" : profit < 0 ? "down" : "flat"}`}>
            {profit >= 0 ? "+" : ""}{Math.round(profit)}M
          </b>
        </div>
        <div className="lo-inspector-row">
          <span>Attendance</span>
          <b>{attendance > 0 ? `${Math.round(attendance * 100)}%` : "—"}</b>
        </div>
        <div className="lo-inspector-row"><span>Arena pressure</span><b>{arena}</b></div>
        <div className="lo-inspector-row"><span>Ownership</span><b>{ownership}</b></div>
        <div className="lo-inspector-row">
          <span>Relocation risk</span>
          <b className={riskClass(t.relocation_risk_label)}>
            {t.relocation_risk_label} · {Math.round(num(t.relocation_risk, 0) * 100)}%
          </b>
        </div>
      </div>
      {tags.length ? (
        <p className="lo-inspector-note">{clipWords(tags.slice(0, 2).join(" · "), 15)}</p>
      ) : null}
      {t.event ? (
        <p className="lo-inspector-note">{clipWords(`${t.event.label} may shift local revenue outlook.`, 15)}</p>
      ) : null}
      {t.threatened ? (
        <p className="lo-inspector-note">Relocation pressure elevated. Watch gate and arena.</p>
      ) : null}
    </div>
  );
}

function IssueInspector({ issue }) {
  const row = asObject(issue);
  if (!Object.keys(row).length) {
    return (
      <div className="lo-inspector">
        <div className="lo-inspector-empty">Select an issue for detail</div>
      </div>
    );
  }

  const owners = num(row.owner_support, 0);
  const players = num(row.player_support, 0);
  const hasPair = owners > 0 || players > 0;

  return (
    <div className="lo-inspector">
      <span className="lo-inspector-mark">Policy Brief · Display Only</span>
      <span className="lo-inspector-kicker">{row.source || "Issue"} · {row.action_state}</span>
      <h4>{row.name}</h4>
      <div className="lo-inspector-rows">
        <div className="lo-inspector-row"><span>Status</span><b>{row.status || row.action_state}</b></div>
        <div className="lo-inspector-row"><span>Deadline</span><b>{issueDeadline(row)}</b></div>
        <div className="lo-inspector-row"><span>Likelihood</span><b>{num(row.likelihood, 0)}%</b></div>
      </div>
      {hasPair ? (
        <div className="lo-support-bars">
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Owners</em><b>{owners}%</b></div>
            <div className="lo-support-track owners"><i style={{ width: `${Math.min(100, owners)}%` }} /></div>
          </div>
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Players</em><b>{players}%</b></div>
            <div className="lo-support-track"><i style={{ width: `${Math.min(100, players)}%` }} /></div>
          </div>
        </div>
      ) : (
        <div className="lo-support-bars">
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Support</em><b>{num(row.support_pct, 0)}%</b></div>
            <div className="lo-support-track"><i style={{ width: `${Math.min(100, num(row.support_pct, 0))}%` }} /></div>
          </div>
        </div>
      )}
      <p className="lo-inspector-note">{issueConsequence(row)}</p>
    </div>
  );
}

function OverviewWorkspace({ data, onOpenIssue }) {
  const confidence = asObject(data.confidence);
  const capChange = num(data.cap_change, 0);
  const drivers = asArray(data.cap_drivers);
  const plusCount = drivers.filter((d) => d.sign === "+").length;
  const escrowPct = Math.min(100, Math.round(num(data.escrow_progress, 0) * 100));
  const revenueLabel =
    data.league_revenue_b != null
      ? `$${num(data.league_revenue_b, 0).toFixed(2)}B`
      : fmtMoneyM(data.league_revenue);
  const moods = unusualMoods(data.owner_mood);
  const topIssues = asArray(data.active_issues).slice(0, 3);

  return (
    <div className="lo-ov">
      <div className="lo-forecast">
        <div className="lo-forecast-head">
          <h2>Cap Forecast</h2>
          <div className="lo-forecast-meta">
            <span className="now">{fmtCap(data.salary_cap)}</span>
            <span className="arrow">→</span>
            <span className="next">{fmtCap(data.projected_salary_cap)}</span>
            <span className={`delta${capChange < 0 ? " neg" : ""}`}>
              {capChange >= 0 ? "+" : ""}
              {fmtCap(capChange)} · {data.cap_change_type || "Flat Cap"}
            </span>
          </div>
        </div>
        <CapForecastChart series={data.cap_forecast} />
        <div className="lo-confidence-text">{confidence.rangeLabel || "—"}</div>
      </div>

      <div className="lo-summary-row">
        <div className="lo-summary">
          <span>Escrow %</span>
          <b>{escrowPct}%</b>
        </div>
        <div className="lo-summary">
          <span>League Revenue</span>
          <b className="gold">{revenueLabel}</b>
        </div>
        <div className="lo-summary">
          <span>Cap Drivers</span>
          <b>{plusCount}/{Math.max(drivers.length, 1)} lifting</b>
        </div>
      </div>

      <div className="lo-sentiment">
        <span className="lo-sentiment-label">League Sentiment</span>
        {moods.length ? (
          moods.map(([label, value]) => {
            const meter = moodMeterValue(value);
            return (
              <span key={label} className="lo-sentiment-item">
                <em>{label}</em>
                <b style={{ color: moodMeterColor(meter) }}>{value}</b>
              </span>
            );
          })
        ) : (
          <span className="lo-sentiment-empty">No unusual mood signals</span>
        )}
      </div>

      <div className="lo-priority">
        <h3 className="lo-priority-title">Priority Decisions</h3>
        <div className="lo-priority-list">
          {topIssues.length ? (
            topIssues.map((issue) => {
              const id = issue.id || issue.name;
              const consequence = issueConsequence(issue);
              const prefix = `${issue.name}: `;
              const detail = consequence.startsWith(prefix)
                ? consequence.slice(prefix.length)
                : consequence;
              return (
                <button
                  key={id}
                  type="button"
                  className="lo-priority-card"
                  onClick={() => onOpenIssue(issue)}
                >
                  <strong>{issue.name}</strong>
                  <div className="lo-priority-meta">
                    <span>{issueDeadline(issue)}</span>
                  </div>
                  <p>{detail}</p>
                  <span className={`state ${stateClass(issue.action_state)}`}>{issue.action_state}</span>
                </button>
              );
            })
          ) : (
            <div className="lo-empty">No priority decisions</div>
          )}
        </div>
      </div>
    </div>
  );
}

function CbaWorkspace({ data, selectedIssueId, onSelectIssue }) {
  const issues = asArray(data.active_issues);
  const timeline = asArray(data.cba_timeline);
  const selected = issues.find((i) => (i.id || i.name) === selectedIssueId) || null;

  return (
    <div className="lo-cba">
      <div className="lo-cba-left">
        <div className="lo-cba-panel">
          <h3 className="lo-section-title">CBA Timeline</h3>
          <div className="lo-timeline-scroll">
            <div className="lo-timeline">
              {timeline.length ? (
                timeline.map((item) => (
                  <div key={item.id} className={`lo-timeline-item ${item.tone || ""}`.trim()}>
                    <span>{item.when}</span>
                    <b>{item.title}</b>
                    <em>{item.detail}</em>
                  </div>
                ))
              ) : (
                <div className="lo-inspector-empty">No CBA milestones</div>
              )}
            </div>
          </div>
        </div>

        <div className="lo-cba-panel">
          <h3 className="lo-section-title">Negotiations</h3>
          <div className="lo-issues-scroll">
            {issues.length ? (
              issues.map((issue) => {
                const id = issue.id || issue.name;
                const selectedCls = selectedIssueId === id ? " selected" : "";
                return (
                  <button
                    key={id}
                    type="button"
                    className={`lo-issue${selectedCls}`}
                    onClick={() => onSelectIssue(issue)}
                  >
                    <span className="lo-issue-title">{issue.name}</span>
                    <span className="lo-issue-scale">{issue.scale_label}</span>
                    <span className={`lo-issue-state ${stateClass(issue.action_state)}`}>{issue.action_state}</span>
                  </button>
                );
              })
            ) : (
              <div className="lo-inspector-empty">No active issues</div>
            )}
          </div>
        </div>
      </div>

      <IssueInspector issue={selected} />
    </div>
  );
}

function filterTeams(teams, filterId) {
  const list = asArray(teams);
  if (filterId === "growing") {
    return list.filter((t) => {
      const yoy = num(t.revenue_yoy_delta, 0);
      const dir = t.revenue_yoy_direction || "";
      return dir === "up" || yoy > 0;
    });
  }
  if (filterId === "losing") {
    return list.filter((t) => {
      const yoy = num(t.revenue_yoy_delta, 0);
      const dir = t.revenue_yoy_direction || "";
      return dir === "down" || yoy < 0 || String(t.display_status) === "Loss";
    });
  }
  if (filterId === "reloc") {
    return list.filter((t) => num(t.relocation_risk, 0) >= 0.35 || t.threatened);
  }
  return list;
}

function MarketsWorkspace({
  teams,
  userTeamId,
  sortKey,
  sortDir,
  onSort,
  filterId,
  onFilter,
  selectedTeamId,
  onSelectTeam,
}) {
  const filtered = filterTeams(teams, filterId);
  const selected = filtered.find((t) => t.id === selectedTeamId) || asArray(teams).find((t) => t.id === selectedTeamId) || null;

  const trendText = (t) => {
    const yoy = num(t.revenue_yoy_delta, 0);
    const dir = t.revenue_yoy_direction || "flat";
    if (dir === "flat" && Math.abs(yoy) < 1) return { text: "—", dir: "flat" };
    return {
      text: `${yoy >= 0 ? "+" : ""}${Math.round(yoy)}M`,
      dir: dir === "up" || yoy > 0 ? "up" : dir === "down" || yoy < 0 ? "down" : "flat",
    };
  };

  const sortLabel = (key, label) => (
    <button type="button" className={sortKey === key ? "sorted" : ""} onClick={() => onSort(key)}>
      {label}
      {sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
    </button>
  );

  return (
    <div className="lo-markets">
      <div className="lo-markets-toolbar">
        <div className="lo-filters" role="tablist" aria-label="Market filters">
          {MARKET_FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={filterId === f.id ? "active" : ""}
              onClick={() => onFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="lo-markets-body">
        <div className="lo-table-wrap">
          <table className="lo-table">
            <thead>
              <tr>
                <th className="lo-col-team">Team</th>
                <th className="lo-col-rev">{sortLabel("revenue", "Revenue")}</th>
                <th className="lo-col-trend">{sortLabel("trend", "Change")}</th>
                <th className="lo-col-market">{sortLabel("market", "Market")}</th>
                <th className="lo-col-risk">{sortLabel("risk", "Risk")}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length ? (
                filtered.map((t) => {
                  const trend = trendText(t);
                  const isUser = String(t.id) === String(userTeamId);
                  const selectedCls = selectedTeamId === t.id;
                  return (
                    <tr
                      key={t.id}
                      tabIndex={0}
                      className={[
                        t.threatened ? "threatened" : "",
                        selectedCls ? "selected" : "",
                        isUser ? "user" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      onClick={() => onSelectTeam(t)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelectTeam(t);
                        }
                      }}
                    >
                      <td className="lo-col-team">
                        <div className="lo-team-cell">
                          <TeamLogo team={t} abbr={t.abbreviation} />
                          <strong>{t.abbreviation}</strong>
                          {t.event ? (
                            <span
                              className={`lo-event-mark ${t.event.kind === "risk" ? "risk" : t.event.kind === "rev" ? "rev" : ""}`}
                              title={t.event.label}
                            />
                          ) : null}
                        </div>
                      </td>
                      <td className="lo-col-rev">{fmtMoneyM(t.revenue)}</td>
                      <td className={`lo-col-trend lo-trend ${trend.dir}`}>{trend.text}</td>
                      <td className="lo-col-market">{t.market_tier || "—"}</td>
                      <td className={`lo-col-risk ${riskClass(t.relocation_risk_label)}`}>
                        {t.relocation_risk_label}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={5}>
                    <div className="lo-empty">No matching teams</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <TeamDossier team={selected} />
      </div>
    </div>
  );
}

function hasMeaningfulPressure(team) {
  const label = String(team.relocation_risk_label || "").toLowerCase();
  if (label.includes("low") && !team.threatened) return false;
  const reloc = num(team.relocation_risk, 0);
  const profit = num(team.profit, 0);
  const status = String(team.display_status || "");
  return reloc >= 0.35 || team.threatened || profit < -5 || status === "Loss" || status === "Risk";
}

function riskBucket(team) {
  const reloc = num(team.relocation_risk, 0);
  const label = String(team.relocation_risk_label || "").toLowerCase();
  if (team.threatened || reloc >= 0.55 || label.includes("high") || label.includes("critical")) {
    return "immediate";
  }
  if (reloc >= 0.42) return "monitor";
  return "stable";
}

function FranchiseRiskWorkspace({ teams, selectedTeamId, onSelectTeam }) {
  const pressured = asArray(teams).filter(hasMeaningfulPressure);
  const immediate = pressured.filter((t) => riskBucket(t) === "immediate");
  const monitor = pressured.filter((t) => riskBucket(t) === "monitor");
  const stable = pressured.filter((t) => riskBucket(t) === "stable");
  const selected =
    pressured.find((t) => t.id === selectedTeamId)
    || asArray(teams).find((t) => t.id === selectedTeamId)
    || null;

  const renderCards = (list) => (
    list.length ? (
      list.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`lo-risk-card${selectedTeamId === t.id ? " selected" : ""}`}
          onClick={() => onSelectTeam(t)}
        >
          <div className="lo-risk-card-top">
            <TeamLogo team={t} abbr={t.abbreviation} size={20} />
            <strong>{t.abbreviation}</strong>
          </div>
          <div className="lo-risk-card-meta">
            <span className={riskClass(t.relocation_risk_label)}>{t.relocation_risk_label}</span>
            <span>{t.reloc_reason || t.display_status}</span>
          </div>
          <div className="lo-risk-card-meta">
            <span style={{ color: "var(--lo-gold)" }}>{fmtMoneyM(t.revenue)}</span>
            <span>{Math.round(num(t.relocation_risk, 0) * 100)}%</span>
          </div>
        </button>
      ))
    ) : (
      <div className="lo-inspector-empty">None</div>
    )
  );

  return (
    <div className={`lo-risk${selected ? " has-detail" : ""}`}>
      <div className="lo-risk-col immediate">
        <h3>Immediate</h3>
        <div className="lo-risk-list">{renderCards(immediate)}</div>
      </div>
      <div className="lo-risk-col monitor">
        <h3>Monitor</h3>
        <div className="lo-risk-list">{renderCards(monitor)}</div>
      </div>
      <div className="lo-risk-col stable">
        <h3>Stable</h3>
        <div className="lo-risk-list">{renderCards(stable)}</div>
      </div>
      {selected ? (
        <div className="lo-risk-detail">
          <TeamDossier team={selected} />
        </div>
      ) : null}
    </div>
  );
}

function sortTeams(teams, sortKey, sortDir) {
  const list = [...asArray(teams)];
  const dir = sortDir === "asc" ? 1 : -1;
  list.sort((a, b) => {
    if (sortKey === "trend") {
      return (num(a.revenue_yoy_delta, 0) - num(b.revenue_yoy_delta, 0)) * dir;
    }
    if (sortKey === "market") {
      const order = { Large: 3, Mid: 2, Small: 1 };
      return ((order[a.market_tier] || 0) - (order[b.market_tier] || 0)) * dir;
    }
    if (sortKey === "risk") {
      return (num(a.relocation_risk, 0) - num(b.relocation_risk, 0)) * dir;
    }
    return (num(a.revenue, 0) - num(b.revenue, 0)) * dir;
  });
  return list;
}

export default function LeagueOperations() {
  const { franchiseState, setScreen } = useGameUI();
  const [remote, setRemote] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [workspace, setWorkspace] = useState("overview");
  const [sortKey, setSortKey] = useState("revenue");
  const [sortDir, setSortDir] = useState("desc");
  const [marketFilter, setMarketFilter] = useState("all");
  const [selectedIssueId, setSelectedIssueId] = useState(null);
  const [selectedTeamId, setSelectedTeamId] = useState(null);

  const cached = useMemo(() => asObject(franchiseState?.league_operations), [franchiseState]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(false);

    getLeagueOperations()
      .then((res) => {
        if (!alive) return;
        setRemote(asObject(res?.league_operations));
        setLoading(false);
      })
      .catch(() => {
        if (!alive) return;
        setRemote(null);
        setError(true);
        setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, []);

  const data = useMemo(() => {
    const raw =
      (remote && Object.keys(remote).length)
        ? remote
        : (cached && Object.keys(cached).length)
          ? cached
          : EMPTY_OBJ;
    return normalizeLeagueOps(raw);
  }, [remote, cached]);

  const hasData = asArray(data.teams).length > 0;
  const userTeamId = franchiseState?.user_team_id || franchiseState?.team?.id;

  const sortedTeams = useMemo(
    () => sortTeams(data.teams, sortKey, sortDir),
    [data.teams, sortKey, sortDir]
  );

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") setScreen(SCREENS.HUB);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setScreen]);

  useEffect(() => {
    if (workspace !== "cba") return;
    const issues = asArray(data.active_issues);
    if (!issues.length) return;
    const stillValid = issues.some((i) => (i.id || i.name) === selectedIssueId);
    if (!stillValid) {
      const first = issues[0];
      setSelectedIssueId(first.id || first.name);
    }
  }, [workspace, data.active_issues, selectedIssueId]);

  const handleSort = (key) => {
    if (!SORT_KEYS.includes(key)) return;
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir(key === "market" ? "asc" : "desc");
    }
  };

  const leagueStatus = data.league_status || "Stable";

  return (
    <div className="game-screen hub-screen calendar-screen cal-fr lo-screen">
      <InlineStyles />

      <header className="lo-header">
        <div className="lo-header-left">
          <h1>League Operations</h1>
          <div className="lo-header-meta">
            <span className="season">{data.season || "—"}</span>
            <span aria-hidden="true">·</span>
            <span className="status">{leagueStatus}</span>
            <span aria-hidden="true">·</span>
            <span className="lo-header-note">Intelligence Display</span>
          </div>
        </div>
        <div className="lo-header-right">
          <YourImpactStrip user={data.user_team} />
          <button type="button" className="lo-back" onClick={() => setScreen(SCREENS.HUB)}>
            Back
          </button>
        </div>
      </header>

      <nav className="lo-tabs" aria-label="League Operations workspaces">
        {WORKSPACES.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={workspace === tab.id ? "active" : ""}
            onClick={() => setWorkspace(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {!hasData && !loading ? (
        <div className="lo-empty">{error ? "No league data" : "No league data"}</div>
      ) : (
        <main className="lo-workspace" aria-live="polite">
          {workspace === "overview" ? (
            <OverviewWorkspace
              data={data}
              onOpenIssue={(issue) => {
                setSelectedIssueId(issue.id || issue.name);
                setWorkspace("cba");
              }}
            />
          ) : null}
          {workspace === "cba" ? (
            <CbaWorkspace
              data={data}
              selectedIssueId={selectedIssueId}
              onSelectIssue={(issue) => setSelectedIssueId(issue.id || issue.name)}
            />
          ) : null}
          {workspace === "markets" ? (
            <MarketsWorkspace
              teams={sortedTeams}
              userTeamId={userTeamId}
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
              filterId={marketFilter}
              onFilter={setMarketFilter}
              selectedTeamId={selectedTeamId}
              onSelectTeam={(t) => setSelectedTeamId(t.id)}
            />
          ) : null}
          {workspace === "risk" ? (
            <FranchiseRiskWorkspace
              teams={sortedTeams}
              selectedTeamId={selectedTeamId}
              onSelectTeam={(t) => setSelectedTeamId(t.id)}
            />
          ) : null}
        </main>
      )}
    </div>
  );
}
