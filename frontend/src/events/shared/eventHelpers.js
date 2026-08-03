/** Shared helpers for cinematic franchise events. */

export function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

export function safeArray(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

export function getTeamId(team) {
  return String(team?.team_id || team?.teamId || team?.id || "").trim();
}

export function getTeamName(team) {
  if (!team) return "—";
  if (typeof team === "string") return team;
  return (
    team.full_name ||
    team.fullName ||
    team.name ||
    team.team_name ||
    team.teamName ||
    getTeamId(team) ||
    "—"
  );
}

export function getPlayerName(player) {
  if (!player) return "—";
  if (typeof player === "string") return player;
  return (
    player.name ||
    player.full_name ||
    player.fullName ||
    player.player_name ||
    player.playerName ||
    "—"
  );
}

export function getPlayerOverall(player) {
  const n = Number(player?.overall ?? player?.ovr ?? player?.rating);
  // Cards always show whole-number OVR (83.4 -> 83, 87.6 -> 88).
  return Number.isFinite(n) ? Math.round(n) : null;
}

export function getPlayerPosition(player) {
  return String(player?.position || player?.pos || player?.primary_position || "—").toUpperCase();
}

export function formatMoney(value) {
  if (value == null || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Math.abs(n) >= 1) return `$${n.toFixed(2)}M`;
  return `$${(n * 1_000_000).toFixed(0)}`;
}

export function formatPick(pick) {
  if (pick == null) return "—";
  if (typeof pick === "object") {
    const num = pick.pick ?? pick.overall ?? pick.pick_number;
    const rnd = pick.round ?? pick.rnd;
    if (num != null && rnd != null) return `${rnd}.${num}`;
    if (num != null) return `#${num}`;
  }
  return `#${pick}`;
}

export function pickFranchiseData(franchiseState, eventData, keys) {
  const list = Array.isArray(keys) ? keys : [keys];
  for (const key of list) {
    const parts = String(key).split(".");
    let cur = eventData;
    for (const p of parts) {
      cur = cur?.[p];
    }
    if (cur !== undefined && cur !== null && cur !== "") return cur;
    cur = franchiseState;
    for (const p of parts) {
      cur = cur?.[p];
    }
    if (cur !== undefined && cur !== null && cur !== "") return cur;
    const off = franchiseState?.offseason;
    if (off) {
      cur = off;
      for (const p of parts) {
        cur = cur?.[p];
      }
      if (cur !== undefined && cur !== null && cur !== "") return cur;
    }
  }
  return undefined;
}

export function buildCinematicCss(prefix) {
  const p = prefix;
  const draftReviewExtras =
    p === "draftreview" || p === "prospectrights" || p === "resign"
      ? `
.${p}-root {
  --bg: #04101a;
  --bg-2: #061522;
  --panel: rgba(9, 25, 38, 0.94);
  --panel-2: rgba(12, 35, 52, 0.94);
  --panel-3: rgba(15, 46, 66, 0.78);
  --line: rgba(156, 218, 236, 0.14);
  --line-2: rgba(115, 229, 241, 0.25);
  --line-strong: rgba(73, 231, 240, 0.5);
  --text: #e9f7fb;
  --muted: #8096a8;
  --muted-2: #607789;
  --cyan: #13d8e7;
  --cyan-soft: rgba(19, 216, 231, 0.13);
  --gold: #e9a83c;
  --gold-soft: rgba(233, 168, 60, 0.14);
  --green: #52df94;
  --green-soft: rgba(82, 223, 148, 0.13);
  --red: #ff606d;
  --red-soft: rgba(255, 96, 109, 0.13);
  --blue: #8ab4ff;
  --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);
  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
  background:
    radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
}
.${p}-root button { font-family: inherit; }
.${p}-spotlight { opacity: 0.45; }
.${p}-topbar { padding: 0.7rem 1.35rem; }
.${p}-phase-text {
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  border: 0;
  background: transparent;
  padding: 0;
}
.${p}-ghost-btn {
  border-radius: var(--radius-hud, 4px);
  background: rgba(12, 31, 47, 0.72);
  border: 1px solid var(--line);
  padding: 0.4rem 0.85rem;
  font-weight: 700;
}
.${p}-ghost-btn:hover {
  border-color: var(--line-strong);
  background: rgba(19, 216, 231, 0.12);
}
.${p}-ghost-btn:active {
  transform: translateY(1px);
}
.${p}-leave-btn {
  border: 0;
  background: transparent;
  color: var(--muted);
  padding: 0.55rem 0.35rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: none;
  border-radius: 0;
}
.${p}-leave-btn:hover {
  color: var(--text);
  background: transparent;
  border-color: transparent;
  transform: none;
}
.${p}-stage { gap: 1.15rem; padding: 0 1.35rem 5.5rem; grid-template-columns: 1fr min(320px, 30vw); }
.${p}-reveal { justify-content: flex-start; padding: 0.65rem 0 0.25rem; overflow: auto; min-height: 0; }
.${p}-title { display: none; text-shadow: none; }
.${p}-eyebrow { display: none; }
.${p}-workspace { display: flex; flex-direction: column; gap: 0.7rem; flex: 1; min-height: 0; width: 100%; }
.${p}-haul-block { display: flex; flex-direction: column; gap: 0.35rem; scroll-margin-top: 0.5rem; }
.${p}-haul-verdict {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.45rem 0.75rem;
}
.${p}-haul-grade {
  font-size: clamp(1.55rem, 2.6vw, 2.05rem);
  font-weight: 900;
  letter-spacing: 0.04em;
  color: var(--gold);
  line-height: 1;
}
.${p}-haul-grade-label {
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text);
}
.${p}-haul-reason { margin: 0; font-size: 0.84rem; color: rgba(233, 247, 251, 0.78); line-height: 1.4; max-width: 78ch; }
.${p}-haul-meta {
  display: flex; flex-wrap: wrap; gap: 0.2rem 0;
  align-items: baseline;
  font-size: 0.72rem;
  color: var(--muted);
  line-height: 1.45;
}
.${p}-haul-meta-item {
  display: inline-flex; align-items: baseline; gap: 0.28rem;
  max-width: 22ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  margin-right: 0.55rem;
}
.${p}-haul-meta-item strong {
  color: rgba(233, 247, 251, 0.9);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.${p}-haul-meta-item.is-grade strong { color: var(--gold); }
.${p}-analysis-row { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
.${p}-analysis-chip {
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  color: rgba(233, 247, 251, 0.78);
  border: 0; border-left: 2px solid rgba(156, 218, 236, 0.28);
  background: transparent; padding: 0.12rem 0.45rem 0.12rem 0.5rem; border-radius: 0;
}
.${p}-more-signals {
  border: 0; background: transparent; color: var(--cyan); cursor: pointer;
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 0.12rem 0.2rem; font-family: inherit;
}
.${p}-more-signals:hover { color: var(--text); }
.${p}-grid {
  display: grid;
  grid-template-columns: 1.08fr 1fr;
  grid-template-rows: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: 0;
  flex: 1; min-height: 0;
  border-top: 1px solid var(--line);
}
.${p}-grid.has-one-row { grid-template-rows: minmax(0, 1fr); }
.${p}-pane {
  background: transparent;
  border: 0;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  border-radius: 0;
  padding: 0.7rem 0.85rem 0.75rem;
  min-height: 0;
  overflow: auto;
  display: flex; flex-direction: column; gap: 0.32rem;
  scroll-margin-top: 0.5rem;
}
.${p}-pane:nth-child(2n) { border-right: 0; }
.${p}-pane-label {
  font-size: 0.6875rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); font-weight: 900; margin: 0;
}
.${p}-pick-kicker {
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--cyan); margin: 0;
}
.${p}-pick-name {
  font-size: clamp(1.55rem, 2.8vw, 2.15rem);
  font-weight: 900; margin: 0; line-height: 1.05; color: var(--text);
  letter-spacing: 0.02em;
}
.${p}-pick-sub { font-size: 0.8rem; color: rgba(233, 247, 251, 0.78); margin: 0; }
.${p}-pick-meta { font-size: 0.72rem; color: var(--muted); margin: 0; }
.${p}-grade-row { display: flex; flex-wrap: wrap; gap: 0.45rem 0.65rem; align-items: center; margin-top: 0.15rem; }
.${p}-grade-mark {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 2rem; height: 1.35rem; padding: 0 0.35rem;
  border-radius: 6px;
  border: 1px solid rgba(233, 168, 60, 0.35);
  background: var(--gold-soft);
  color: #ffd88d;
  font-size: 0.72rem; font-weight: 900; font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}
.${p}-grade-label, .${p}-secondary-meta {
  font-size: 0.76rem; font-weight: 700; color: rgba(233, 247, 251, 0.78);
}
.${p}-risk {
  font-size: 0.7rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--muted);
}
.${p}-risk.tone-high { color: var(--red); }
.${p}-risk.tone-med { color: var(--gold); }
.${p}-risk.tone-low { color: var(--green); }
.${p}-review-line { font-size: 0.78rem; color: rgba(233, 247, 251, 0.82); margin: 0.1rem 0 0; line-height: 1.35; }
.${p}-dest-hero {
  font-size: clamp(1.15rem, 2vw, 1.45rem); font-weight: 900; margin: 0; color: var(--text);
}
.${p}-dest-label { font-size: 0.6875rem; color: var(--gold); font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; margin: 0; }
.${p}-plan-row, .${p}-fit-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem 0.85rem; margin-top: 0.15rem;
}
.${p}-plan-kv { margin: 0; }
.${p}-plan-kv span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); font-weight: 800;
}
.${p}-plan-kv strong {
  display: block; font-size: 0.78rem; color: var(--text); font-weight: 800;
  margin-top: 0.1rem; line-height: 1.3;
}
.${p}-season-obj {
  margin: 0.15rem 0 0; font-size: 0.8rem; color: rgba(233, 247, 251, 0.85); line-height: 1.35;
}
.${p}-path-timeline {
  display: flex; align-items: stretch; gap: 0; margin-top: 0.45rem;
  padding-top: 0.35rem;
  border-top: 1px solid var(--line);
}
.${p}-path-step {
  flex: 1; min-width: 0; text-align: left; padding: 0.15rem 0.45rem 0.15rem 0.55rem;
  border: 0; border-radius: 0; background: transparent;
  border-left: 2px solid rgba(156, 218, 236, 0.22);
  display: flex; flex-direction: column; justify-content: center; gap: 0.12rem;
  opacity: 0.55;
}
.${p}-path-step.is-next, .${p}-path-step.is-current {
  opacity: 1; border-left-color: var(--gold);
}
.${p}-path-step.is-future, .${p}-path-step.is-projection { opacity: 0.45; }
.${p}-path-stage {
  display: block; font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); font-weight: 900;
}
.${p}-path-detail {
  display: block; font-size: 0.72rem; color: var(--text); font-weight: 700; line-height: 1.25;
}
.${p}-path-arrow { display: none; }
.${p}-alt-toggle {
  align-self: flex-start; margin-top: 0.2rem; border: 0; background: transparent;
  color: var(--cyan); cursor: pointer; font-size: 0.6875rem; font-weight: 800;
  letter-spacing: 0.04em; text-transform: uppercase; padding: 0; font-family: inherit;
}
.${p}-alt-toggle:hover { color: var(--text); }
.${p}-alt-path { margin: 0.1rem 0 0; font-size: 0.74rem; color: rgba(233, 247, 251, 0.78); line-height: 1.35; }
.${p}-stat-row { display: flex; flex-wrap: wrap; gap: 0.45rem 0.9rem; }
.${p}-stat { margin: 0; min-width: 3rem; }
.${p}-stat span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); font-weight: 800;
}
.${p}-stat strong {
  display: block; font-size: 0.95rem; font-weight: 900; color: var(--text);
  margin-top: 0.08rem; font-variant-numeric: tabular-nums;
}
.${p}-scout-head { margin: 0; font-size: 0.84rem; font-weight: 800; color: var(--text); }
.${p}-note-list { margin: 0.15rem 0 0; padding-left: 1rem; color: rgba(233, 247, 251, 0.78); font-size: 0.72rem; }
.${p}-note-list li { margin: 0.1rem 0; }
.${p}-context { font-size: 0.74rem; color: rgba(233, 247, 251, 0.72); margin: 0.12rem 0 0; line-height: 1.35; }
.${p}-rights-callout {
  margin-top: 0.2rem; padding: 0.45rem 0.55rem;
  border-left: 3px solid var(--gold);
  background: var(--gold-soft);
}
.${p}-rights-callout span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: #ffd88d; font-weight: 900;
}
.${p}-rights-callout strong {
  display: block; margin-top: 0.12rem; font-size: 0.84rem; color: var(--text); font-weight: 800;
}
.${p}-panel {
  border-radius: 12px; border-color: var(--line);
  box-shadow: var(--shadow); padding: 0.65rem 0.65rem 0.45rem;
  backdrop-filter: blur(12px); background: var(--panel);
  max-height: calc(100vh - 180px);
}
.${p}-panel h2 {
  position: sticky; top: 0; z-index: 2;
  margin: 0 0 0.4rem; padding: 0.15rem 0 0.45rem;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
  font-size: 0.72rem; letter-spacing: 0.12em; font-weight: 900;
}
.${p}-rail { gap: 0.2rem; padding-right: 0.1rem; }
.${p}-rail-hint {
  margin: 0.35rem 0 0; padding-top: 0.35rem;
  border-top: 1px solid var(--line);
  font-size: 0.6875rem; color: var(--muted); letter-spacing: 0.04em;
}
.${p}-rail-btn {
  display: block; width: 100%; text-align: left; cursor: pointer;
  background: transparent; border: 0; padding: 0; color: inherit; font: inherit; border-radius: 0;
}
.${p}-rail-btn:focus-visible { outline: 2px solid rgba(19, 216, 231, 0.65); outline-offset: 2px; }
.${p}-haul-card {
  display: grid; grid-template-columns: 3.4rem 1fr auto; gap: 0.45rem; align-items: start;
  padding: 0.4rem 0.45rem 0.4rem 0.55rem;
  border: 0; border-radius: 0; background: transparent;
  border-left: 2px solid transparent;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.${p}-haul-card.is-active {
  background: var(--cyan-soft);
  border-left-color: var(--gold);
  box-shadow: none;
}
.${p}-haul-card:hover { background: rgba(19, 216, 231, 0.07); }
.${p}-haul-rank {
  font-size: 0.6875rem; font-weight: 900; color: var(--cyan);
  letter-spacing: 0.02em; line-height: 1.25; padding-top: 0.1rem;
  font-variant-numeric: tabular-nums;
}
.${p}-haul-body strong {
  display: block; font-size: 0.84rem; margin-bottom: 0.08rem; color: var(--text); font-weight: 800;
}
.${p}-haul-body .${p}-card-details {
  flex-direction: column; align-items: flex-start; gap: 0.08rem;
  font-size: 0.6875rem; color: var(--muted);
}
.${p}-haul-grade-letter {
  font-size: 0.78rem; font-weight: 900; min-width: 1.4rem; text-align: right;
  padding-top: 0.08rem;
}
.${p}-haul-grade-letter.tone-a { color: var(--green); }
.${p}-haul-grade-letter.tone-b { color: #56dc75; }
.${p}-haul-grade-letter.tone-c { color: var(--gold); }
.${p}-haul-grade-letter.tone-d { color: var(--red); }
.${p}-haul-grade-letter.tone-n { color: var(--muted); }
.${p}-footer { padding: 0.55rem 1.35rem 0.85rem; }
.${p}-ticker { margin-bottom: 0.55rem; }
.${p}-ticker-tab {
  border: 0; background: transparent; cursor: pointer; font-family: inherit;
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); padding: 0.2rem 0.1rem; border-bottom: 2px solid transparent;
}
.${p}-ticker-tab:hover { color: var(--text); }
.${p}-ticker-tab.is-active { color: var(--text); border-bottom-color: var(--gold); }
.${p}-footer-actions.is-split { justify-content: space-between; }
.${p}-footer-actions.is-split .${p}-cta-btn { margin-left: auto; }
.${p}-cta-btn {
  border-radius: 7px; min-width: 190px; letter-spacing: 0.1em; font-size: 11px;
  box-shadow: 0 14px 36px rgba(217, 144, 35, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}
.${p}-skip-btn { border-radius: var(--radius-ops, 2px); }
.${p}-decision-list { display: flex; flex-direction: column; gap: 0.28rem; margin-top: 0.2rem; }
.${p}-decision-btn {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 0.5rem;
  width: 100%; text-align: left; cursor: pointer;
  border: 1px solid var(--line); border-radius: 8px;
  background: rgba(12, 31, 47, 0.55);
  color: var(--text); padding: 0.45rem 0.55rem;
  font-family: inherit; font-size: 0.76rem; font-weight: 700;
}
.${p}-decision-btn:hover:not(:disabled):not(.is-disabled) {
  border-color: rgba(19, 216, 231, 0.35);
  background: rgba(19, 216, 231, 0.08);
}
.${p}-decision-btn.is-recommended {
  border-color: rgba(233, 168, 60, 0.45);
  background: var(--gold-soft);
}
.${p}-decision-btn.is-selected {
  border-color: rgba(19, 216, 231, 0.55);
  box-shadow: inset 0 0 0 1px rgba(19, 216, 231, 0.25);
}
.${p}-decision-btn.is-disabled, .${p}-decision-btn:disabled {
  opacity: 0.45; color: var(--muted); cursor: not-allowed;
}
.${p}-decision-btn .${p}-decision-meta {
  display: block; margin-top: 0.12rem;
  font-size: 0.6875rem; font-weight: 600; color: var(--muted); line-height: 1.3;
}
.${p}-decision-tag {
  flex-shrink: 0; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em;
  text-transform: uppercase; color: #ffd88d;
}
.${p}-decision-detail {
  margin-top: 0.45rem; padding: 0.55rem 0.6rem;
  border: 1px solid rgba(156, 218, 236, 0.18);
  border-radius: 10px; background: rgba(8, 22, 34, 0.72);
}
.${p}-decision-speech {
  position: relative; margin: 0 0 0.55rem;
  padding: 0.55rem 0.7rem 0.55rem 0.85rem;
  border-left: 3px solid rgba(233, 168, 60, 0.65);
  background: rgba(233, 168, 60, 0.08);
  color: rgba(245, 236, 214, 0.92);
  font-size: 0.74rem; font-weight: 600; line-height: 1.4;
}
.${p}-decision-speech strong {
  display: block; margin-bottom: 0.15rem;
  font-size: 0.6875rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: #ffd88d; font-weight: 900;
}
.${p}-decision-tradeoffs {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem;
}
.${p}-decision-col { min-width: 0; }
.${p}-decision-col-label {
  display: block; margin-bottom: 0.2rem;
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-decision-col ul {
  margin: 0; padding-left: 0.95rem;
  color: rgba(233, 247, 251, 0.82); font-size: 0.6875rem; line-height: 1.35;
}
.${p}-decision-col.is-pros ul { color: rgba(170, 232, 196, 0.92); }
.${p}-decision-col.is-cons ul { color: rgba(255, 196, 170, 0.92); }
.${p}-decision-col li { margin: 0.12rem 0; }
.${p}-decision-apply {
  margin-top: 0.55rem; width: 100%;
  border: 1px solid rgba(233, 168, 60, 0.45); border-radius: 8px;
  background: linear-gradient(180deg, rgba(233, 168, 60, 0.28), rgba(233, 168, 60, 0.12));
  color: #ffe7b5; padding: 0.5rem 0.65rem;
  font-family: inherit; font-size: 0.72rem; font-weight: 800;
  letter-spacing: 0.06em; text-transform: uppercase; cursor: pointer;
}
.${p}-decision-apply:disabled {
  opacity: 0.45; cursor: not-allowed;
}
.${p}-decision-feedback {
  margin: 0.4rem 0 0; font-size: 0.7rem; line-height: 1.35; color: rgba(233, 247, 251, 0.8);
}
.${p}-decision-feedback.is-error { color: #ffb4a0; }
.${p}-decision-feedback.is-ok { color: #9be7b8; }
@media (max-width: 720px) {
  .${p}-decision-tradeoffs { grid-template-columns: 1fr; }
}
.${p}-alert-list { display: flex; flex-direction: column; gap: 0.25rem; margin-top: 0.15rem; }
.${p}-alert-item {
  margin: 0; padding: 0.35rem 0.45rem;
  border-left: 2px solid rgba(156, 218, 236, 0.28);
  font-size: 0.72rem; color: rgba(233, 247, 251, 0.78); line-height: 1.35;
}
.${p}-alert-item.is-warn { border-left-color: var(--gold); color: #ffd88d; }
.${p}-env-reasons { margin: 0.1rem 0 0; padding-left: 1rem; color: rgba(233, 247, 251, 0.78); font-size: 0.72rem; }
.${p}-env-reasons li { margin: 0.08rem 0; }
.${p}-path-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.2rem; }
.${p}-path-chip {
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); border-left: 2px solid rgba(156, 218, 236, 0.28);
  padding: 0.1rem 0.4rem;
}
.${p}-path-chip.is-current { color: var(--text); border-left-color: var(--gold); }
.${p}-filter-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.15rem; }
.${p}-filter-chip {
  border: 1px solid var(--line); background: rgba(12, 31, 47, 0.55); color: var(--muted);
  border-radius: var(--radius-ops, 2px); padding: 0.22rem 0.5rem; font-size: 0.6875rem; font-weight: 800;
  letter-spacing: 0.04em; text-transform: uppercase; cursor: pointer; font-family: inherit;
}
.${p}-filter-chip em { font-style: normal; margin-left: 0.35rem; color: var(--cyan); }
.${p}-filter-chip.is-active, .${p}-haul-meta .${p}-filter-chip.is-active {
  border-color: rgba(233, 168, 60, 0.45); color: var(--text); background: var(--gold-soft);
}
.${p}-table-layout {
  display: grid; grid-template-columns: minmax(0, 1fr) min(320px, 32vw);
  gap: 0.75rem; min-height: 0; flex: 1;
}
.${p}-table-wrap {
  min-height: 0; overflow: auto; border: 1px solid var(--line); border-radius: 10px;
  background: rgba(8, 22, 34, 0.55);
}
.${p}-table { width: 100%; border-collapse: collapse; font-size: 0.74rem; }
.${p}-table thead th {
  position: sticky; top: 0; z-index: 2; background: rgba(9, 25, 38, 0.98);
  text-align: left; padding: 0.45rem 0.5rem; font-size: 0.6875rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); font-weight: 900; cursor: pointer;
  border-bottom: 1px solid var(--line); white-space: nowrap;
}
.${p}-table tbody td {
  padding: 0.35rem 0.5rem; border-bottom: 1px solid rgba(156, 218, 236, 0.08);
  vertical-align: middle; color: var(--text);
}
.${p}-table tbody tr { cursor: pointer; }
.${p}-table tbody tr:hover { background: rgba(19, 216, 231, 0.06); }
.${p}-table tbody tr.is-selected {
  background: rgba(19, 216, 231, 0.12);
  box-shadow: inset 3px 0 0 var(--gold);
}
.${p}-player-cell { display: flex; align-items: center; gap: 0.45rem; min-width: 0; }
.${p}-player-cell strong { font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.${p}-row-action {
  border: 1px solid rgba(233, 168, 60, 0.35); background: var(--gold-soft); color: #ffd88d;
  border-radius: 6px; padding: 0.2rem 0.45rem; font-size: 0.6875rem; font-weight: 800;
  cursor: pointer; font-family: inherit; text-transform: uppercase; letter-spacing: 0.04em;
}
.${p}-muted { color: var(--muted); }
.${p}-detail-panel {
  min-height: 0; overflow: auto; border: 1px solid var(--line); border-radius: 10px;
  background: var(--panel); padding: 0.65rem; display: flex; flex-direction: column; gap: 0.45rem;
}
.${p}-detail-head { display: flex; gap: 0.55rem; align-items: center; }
.${p}-detail-head h3 { margin: 0; font-size: 0.95rem; font-weight: 900; }
.${p}-detail-head p { margin: 0.15rem 0 0; font-size: 0.7rem; color: var(--muted); }
.${p}-detail-tabs { display: flex; gap: 0.35rem; }
.${p}-detail-tabs button {
  flex: 1; border: 1px solid var(--line); background: transparent; color: var(--muted);
  border-radius: 6px; padding: 0.35rem; font-size: 0.6875rem; font-weight: 800;
  letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer; font-family: inherit;
}
.${p}-detail-tabs button.is-active { color: var(--text); border-color: var(--gold); background: var(--gold-soft); }
.${p}-detail-tabs button:disabled { opacity: 0.4; cursor: not-allowed; }
.${p}-detail-body { display: flex; flex-direction: column; gap: 0.4rem; }
.${p}-field { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.6875rem; color: var(--muted); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
.${p}-field input[type="number"] {
  border: 1px solid var(--line); background: rgba(4, 16, 26, 0.7); color: var(--text);
  border-radius: 6px; padding: 0.4rem 0.5rem; font-size: 0.85rem; font-weight: 700;
}
.${p}-check-row { display: flex; flex-wrap: wrap; gap: 0.55rem; font-size: 0.72rem; color: rgba(233,247,251,0.82); }
.${p}-check-row label { display: inline-flex; align-items: center; gap: 0.25rem; }
.${p}-negotiate-actions { display: flex; gap: 0.4rem; align-items: center; }
.${p}-negotiate-actions .${p}-cta-btn { min-width: 0; flex: 1; padding: 0.65rem 0.75rem; font-size: 0.6875rem; }
.${p}-pending-shot { width: 2.2rem; }
.${p}-panel { max-height: calc(100vh - 170px); }
.${p}-stage { grid-template-columns: 1fr min(280px, 28vw); gap: 1rem; padding-bottom: 5.25rem; }
.${p}-cta-btn:disabled { opacity: 0.55; cursor: not-allowed; }
@media (max-width: 1200px) {
  .${p}-table-layout { grid-template-columns: 1fr; }
  .${p}-detail-panel { max-height: 280px; }
}
@media (max-width: 1366px) {
  .${p}-fit-grid { grid-template-columns: 1fr; }
  .${p}-pick-meta { display: none; }
  .${p}-haul-meta-item:nth-child(n+5) { display: none; }
}
@media (max-width: 1100px) {
  .${p}-grid { grid-template-columns: 1fr 1fr; }
  .${p}-path-detail { font-size: 0.6875rem; }
  .${p}-stage { grid-template-columns: 1fr min(280px, 34vw); }
}
@media (max-width: 900px) {
  .${p}-pane:nth-child(2n) { border-right: 1px solid var(--line); }
  .${p}-path-timeline { flex-direction: column; gap: 0.35rem; }
  .${p}-path-step { border-left-width: 2px; padding-left: 0.55rem; }
}
`
      : "";
  const draftReviewOnly =
    p === "draftreview"
      ? `
.${p}-root { overflow: hidden; height: 100vh; max-height: 100vh; }
.${p}-reveal {
  overflow: hidden; min-height: 0; flex: 1;
  padding: 0.35rem 0 0.15rem; justify-content: flex-start;
}
.${p}-workspace {
  display: flex; flex-direction: column;
  gap: 0.35rem; overflow: hidden; min-height: 0; flex: 1;
  max-height: calc(100vh - 148px);
}
.${p}-stage {
  min-height: 0; height: calc(100vh - 118px);
  max-height: calc(100vh - 118px);
  padding: 0 1.15rem 4.35rem; gap: 0.7rem;
  align-items: stretch;
}
.${p}-footer { padding: 0.3rem 1.15rem 0.5rem; }
.${p}-ticker { margin-bottom: 0.25rem; }
.${p}-panel {
  max-height: calc(100vh - 140px); height: calc(100vh - 140px);
  padding: 0.4rem 0.4rem 0.3rem; min-height: 0;
  border-color: rgba(233, 168, 60, 0.22);
}
.${p}-panel h2 {
  margin-bottom: 0.25rem; padding-bottom: 0.25rem; font-size: 0.6875rem; color: var(--gold);
}
.${p}-rail {
  gap: 0.12rem; overflow-y: auto; min-height: 0;
  scrollbar-width: thin; scrollbar-color: rgba(19, 216, 231, 0.35) transparent;
}
.${p}-rail::-webkit-scrollbar { width: 5px; }
.${p}-rail::-webkit-scrollbar-thumb {
  background: rgba(19, 216, 231, 0.35); border-radius: 999px;
}
.${p}-haul-banner {
  display: grid; grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 0.45rem 0.7rem; align-items: center;
  padding: 0.35rem 0.5rem; border-radius: 10px;
  background: linear-gradient(105deg, rgba(233, 168, 60, 0.14), rgba(9, 25, 38, 0.3) 42%, rgba(19, 216, 231, 0.06));
  border: 1px solid rgba(233, 168, 60, 0.26);
  flex-shrink: 0;
}
.${p}-broadcast-strip {
  display: flex; align-items: center; gap: 0.55rem; flex-shrink: 0;
  padding: 0.18rem 0.35rem; border-radius: 6px;
  background: linear-gradient(90deg, rgba(180, 28, 36, 0.35), rgba(8, 22, 34, 0.55));
  border: 1px solid rgba(255, 96, 109, 0.28);
}
.${p}-broadcast-live {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase;
  color: #ffb4a0; padding: 0.08rem 0.35rem; border-radius: 3px;
  background: rgba(180, 28, 36, 0.55); border: 1px solid rgba(255, 96, 109, 0.45);
  animation: ${p}-onair-pulse 1.8s ease-in-out infinite;
}
.${p}-broadcast-show {
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  color: rgba(233, 247, 251, 0.92);
}
.${p}-broadcast-seg {
  margin-left: auto; font-size: 0.6875rem; font-weight: 700; color: var(--muted);
  letter-spacing: 0.04em; text-transform: uppercase;
}
@keyframes ${p}-onair-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}
.${p}-haul-banner-grade {
  width: 3.4rem; height: 3.4rem; border-radius: 10px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: rgba(2, 10, 17, 0.55); border: 2px solid rgba(233, 168, 60, 0.55);
}
.${p}-haul-banner-grade strong {
  font-size: 1.45rem; font-weight: 900; color: #ffd88d; line-height: 1;
}
.${p}-haul-banner-grade span {
  margin-top: 0.08rem; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em;
  text-transform: uppercase; color: rgba(255, 216, 141, 0.75);
}
.${p}-haul-banner-grade.tone-a { border-color: rgba(82, 223, 148, 0.55); }
.${p}-haul-banner-grade.tone-a strong { color: #9be7b8; }
.${p}-haul-banner-grade.tone-b { border-color: rgba(86, 220, 117, 0.45); }
.${p}-haul-banner-grade.tone-b strong { color: #9be7b8; }
.${p}-haul-banner-grade.tone-c { border-color: rgba(233, 168, 60, 0.55); }
.${p}-haul-banner-grade.tone-d { border-color: rgba(255, 96, 109, 0.5); }
.${p}-haul-banner-grade.tone-d strong { color: #ffb4a0; }
.${p}-haul-banner-copy { min-width: 0; }
.${p}-haul-banner-copy strong {
  display: block; font-size: 0.86rem; font-weight: 900; color: var(--text);
}
.${p}-haul-banner-copy p {
  margin: 0.08rem 0 0; font-size: 0.7rem; color: rgba(233, 247, 251, 0.78); line-height: 1.3;
  display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden;
}
.${p}-haul-chips { display: flex; flex-wrap: wrap; gap: 0.22rem; margin-top: 0.2rem; }
.${p}-haul-chip {
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  color: rgba(233, 247, 251, 0.85); padding: 0.1rem 0.35rem; border-radius: var(--radius-ops, 2px);
  background: rgba(12, 31, 47, 0.65); border: 1px solid rgba(156, 218, 236, 0.18);
}
.${p}-haul-metrics { display: flex; gap: 0.45rem; }
.${p}-haul-metric-card { min-width: 3.6rem; text-align: center; padding: 0.1rem 0.2rem; }
.${p}-haul-metric-card span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); font-weight: 800;
}
.${p}-haul-metric-card strong {
  display: block; margin-top: 0.08rem; font-size: 0.86rem; font-weight: 900; color: var(--text);
  max-width: 8ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-left: auto; margin-right: auto;
}
.${p}-review-main {
  display: flex; flex-direction: column; gap: 0.35rem;
  min-height: 0; flex: 1; overflow: hidden;
}
.${p}-draft-card {
  display: grid; grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 0.5rem 0.7rem; align-items: center;
  padding: 0.4rem 0.55rem; border-radius: 10px;
  background: linear-gradient(180deg, rgba(15, 46, 66, 0.5), rgba(8, 22, 34, 0.7));
  border: 1px solid rgba(156, 218, 236, 0.14);
  flex-shrink: 0;
}
.${p}-pos-mark {
  width: 3.4rem; height: 3.4rem; border-radius: 10px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: rgba(19, 216, 231, 0.1); border: 1px solid rgba(19, 216, 231, 0.35);
}
.${p}-pos-mark strong { font-size: 1.15rem; font-weight: 900; color: var(--cyan); line-height: 1; }
.${p}-pos-mark span {
  margin-top: 0.12rem; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-pos-mark.pos-f { background: rgba(82, 223, 148, 0.1); border-color: rgba(82, 223, 148, 0.35); }
.${p}-pos-mark.pos-f strong { color: #9be7b8; }
.${p}-pos-mark.pos-d { background: rgba(138, 180, 255, 0.12); border-color: rgba(138, 180, 255, 0.35); }
.${p}-pos-mark.pos-d strong { color: #8ab4ff; }
.${p}-pos-mark.pos-g { background: rgba(233, 168, 60, 0.12); border-color: rgba(233, 168, 60, 0.4); }
.${p}-pos-mark.pos-g strong { color: #ffd88d; }
.${p}-draft-card-body { min-width: 0; }
.${p}-hero-kicker {
  margin: 0; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--cyan);
}
.${p}-hero-name {
  margin: 0.06rem 0 0; font-size: clamp(1.35rem, 2.4vw, 1.85rem); font-weight: 900;
  line-height: 1.05; color: var(--text); letter-spacing: 0.01em;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.${p}-hero-sub {
  margin: 0.12rem 0 0; font-size: 0.74rem; color: rgba(233, 247, 251, 0.86); line-height: 1.25;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-hero-meta {
  margin: 0.1rem 0 0; font-size: 0.6875rem; color: rgba(180, 206, 220, 0.9); line-height: 1.25;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-value-banner {
  display: inline-flex; align-items: center; gap: 0.3rem; margin-top: 0.22rem;
  padding: 0.16rem 0.45rem; border-radius: var(--radius-ops, 2px); font-size: 0.6875rem; font-weight: 900;
  letter-spacing: 0.04em; text-transform: uppercase; border: 1px solid transparent;
}
.${p}-value-banner.tone-steal {
  color: #9be7b8; background: rgba(82, 223, 148, 0.14); border-color: rgba(82, 223, 148, 0.35);
}
.${p}-value-banner.tone-value {
  color: #9be7b8; background: rgba(82, 223, 148, 0.1); border-color: rgba(82, 223, 148, 0.28);
}
.${p}-value-banner.tone-reach {
  color: #ffb4a0; background: rgba(255, 96, 109, 0.12); border-color: rgba(255, 96, 109, 0.35);
}
.${p}-value-banner.tone-expected {
  color: #ffd88d; background: rgba(233, 168, 60, 0.12); border-color: rgba(233, 168, 60, 0.35);
}
.${p}-value-banner.tone-neutral {
  color: rgba(233, 247, 251, 0.85); background: rgba(19, 216, 231, 0.08); border-color: rgba(19, 216, 231, 0.25);
}
.${p}-grade-shield {
  width: 3.7rem; height: 3.7rem; border-radius: 50%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  border: 2px solid rgba(233, 168, 60, 0.5); background: rgba(233, 168, 60, 0.12);
  flex-shrink: 0;
}
.${p}-grade-shield strong { font-size: 1.2rem; font-weight: 900; color: #ffd88d; line-height: 1; }
.${p}-grade-shield span {
  margin-top: 0.08rem; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.05em;
  text-transform: uppercase; color: rgba(255, 216, 141, 0.8); text-align: center;
  max-width: 3.3rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.${p}-grade-shield.tone-a { border-color: rgba(82, 223, 148, 0.55); background: rgba(82, 223, 148, 0.12); }
.${p}-grade-shield.tone-a strong, .${p}-grade-shield.tone-a span { color: #9be7b8; }
.${p}-grade-shield.tone-b { border-color: rgba(86, 220, 117, 0.45); background: rgba(86, 220, 117, 0.1); }
.${p}-grade-shield.tone-b strong, .${p}-grade-shield.tone-b span { color: #9be7b8; }
.${p}-grade-shield.tone-c { border-color: rgba(233, 168, 60, 0.5); }
.${p}-grade-shield.tone-d { border-color: rgba(255, 96, 109, 0.5); background: rgba(255, 96, 109, 0.12); }
.${p}-grade-shield.tone-d strong, .${p}-grade-shield.tone-d span { color: #ffb4a0; }
.${p}-stage-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.95fr);
  grid-template-rows: minmax(0, 1fr) auto auto;
  gap: 0.35rem 0.5rem; min-height: 0; flex: 1; overflow: hidden;
}
.${p}-projection-card, .${p}-prod-card, .${p}-roadmap-card, .${p}-bottom-band {
  border-radius: 10px; border: 1px solid rgba(156, 218, 236, 0.14);
  background: rgba(8, 22, 34, 0.55); padding: 0.4rem 0.5rem; min-width: 0; min-height: 0;
}
.${p}-projection-card, .${p}-prod-card { overflow: hidden; }
.${p}-section-label {
  margin: 0; font-size: 0.6875rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); font-weight: 900;
}
.${p}-projection-role {
  margin: 0.15rem 0 0; font-size: clamp(0.95rem, 1.5vw, 1.15rem); font-weight: 900;
  color: var(--text); line-height: 1.15;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.${p}-projection-eta {
  margin: 0.15rem 0 0; font-size: 0.76rem; font-weight: 800; color: var(--gold);
}
.${p}-meter-stack {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.28rem 0.55rem; margin-top: 0.3rem;
}
.${p}-meter-row { min-width: 0; }
.${p}-meter-head {
  display: flex; justify-content: space-between; gap: 0.35rem; align-items: baseline;
  font-size: 0.6875rem; color: rgba(233, 247, 251, 0.82); font-weight: 700;
}
.${p}-meter-head strong { color: var(--text); font-weight: 900; }
/* Event meters read as ruled scales, matching the ops ledger language. */
.${p}-meter-track {
  margin-top: 0.1rem; height: 0.32rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(12, 35, 52, 0.9);
  overflow: hidden;
}
.${p}-meter-fill {
  display: block; height: 100%; border-radius: 0;
  background: linear-gradient(90deg, #13d8e7, #52df94);
}
.${p}-meter-fill.tone-high { background: linear-gradient(90deg, #2fbf78, #52df94); }
.${p}-meter-fill.tone-med { background: linear-gradient(90deg, #d99023, #e9a83c); }
.${p}-meter-fill.tone-low { background: linear-gradient(90deg, #ff606d, #ff8f98); }
.${p}-meter-fill.tone-neutral { background: linear-gradient(90deg, #13d8e7, #8ab4ff); }
.${p}-prod-card { display: flex; flex-direction: column; gap: 0.25rem; }
.${p}-prod-hero { display: flex; align-items: baseline; gap: 0.35rem; margin-top: 0.1rem; }
.${p}-prod-hero strong {
  font-size: clamp(1.35rem, 2.2vw, 1.75rem); font-weight: 900; color: var(--cyan); line-height: 1;
  font-variant-numeric: tabular-nums;
}
.${p}-prod-hero span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
}
.${p}-prod-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.25rem; }
.${p}-prod-stat {
  margin: 0; padding: 0.22rem 0.25rem; border-radius: 7px; background: rgba(12, 31, 47, 0.55);
  text-align: center;
}
.${p}-prod-stat span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); font-weight: 800;
}
.${p}-prod-stat strong {
  display: block; margin-top: 0.08rem; font-size: 0.92rem; font-weight: 900; color: var(--text);
  font-variant-numeric: tabular-nums;
}
.${p}-prod-stat.is-ppg { background: rgba(19, 216, 231, 0.12); }
.${p}-prod-stat.is-ppg strong { color: var(--cyan); }
.${p}-prod-context {
  margin: 0; font-size: 0.6875rem; color: rgba(180, 206, 220, 0.92); line-height: 1.25;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-roadmap-card { grid-column: 1 / -1; flex-shrink: 0; padding: 0.35rem 0.5rem; }
.${p}-roadmap { display: flex; align-items: stretch; gap: 0; margin-top: 0.25rem; }
.${p}-road-step {
  flex: 1; min-width: 0; position: relative;
  padding: 0.2rem 0.4rem 0.2rem 0.5rem;
  border-left: 3px solid rgba(156, 218, 236, 0.2); opacity: 0.55;
}
.${p}-road-step.is-next, .${p}-road-step.is-current {
  opacity: 1; border-left-color: var(--gold); background: rgba(233, 168, 60, 0.08);
}
.${p}-road-step.is-projection { opacity: 0.9; border-left-color: var(--cyan); }
.${p}-road-step-label {
  display: block; font-size: 0.6875rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); font-weight: 900;
}
.${p}-road-step-title {
  display: block; margin-top: 0.08rem; font-size: 0.78rem; font-weight: 900; color: var(--text);
  line-height: 1.15; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-road-step-detail {
  display: block; margin-top: 0.06rem; font-size: 0.6875rem; color: rgba(233, 247, 251, 0.78); line-height: 1.2;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-road-step-eta {
  display: inline-block; margin-top: 0.12rem; font-size: 0.6875rem; font-weight: 800; color: var(--gold);
}
.${p}-bottom-band {
  grid-column: 1 / -1; flex-shrink: 0;
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.35fr);
  gap: 0.45rem 0.65rem; align-items: start;
}
.${p}-scout-col { min-width: 0; }
.${p}-scout-col h3 {
  margin: 0 0 0.18rem; font-size: 0.6875rem; letter-spacing: 0.08em; text-transform: uppercase;
  font-weight: 900;
}
.${p}-scout-col.is-pro h3 { color: #9be7b8; }
.${p}-scout-col.is-con h3 { color: #ffb4a0; }
.${p}-scout-item {
  margin: 0 0 0.12rem; padding-left: 0.4rem; border-left: 2px solid rgba(156, 218, 236, 0.25);
  font-size: 0.6875rem; line-height: 1.25; color: rgba(233, 247, 251, 0.88);
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.${p}-scout-col.is-pro .${p}-scout-item { border-left-color: rgba(82, 223, 148, 0.55); }
.${p}-scout-col.is-con .${p}-scout-item { border-left-color: rgba(255, 96, 109, 0.55); }
.${p}-org-col { min-width: 0; }
.${p}-org-top {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem 0.45rem; margin-top: 0.15rem;
}
.${p}-org-kv { margin: 0; min-width: 0; }
.${p}-org-kv span {
  display: block; font-size: 0.6875rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); font-weight: 800;
}
.${p}-org-kv strong {
  margin-top: 0.06rem; font-size: 0.7rem; font-weight: 900; color: var(--text);
  line-height: 1.2; overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical;
}
.${p}-depth-bars { display: flex; flex-direction: column; gap: 0.16rem; margin-top: 0.25rem; }
.${p}-depth-bar-row {
  display: grid; grid-template-columns: 4.2rem minmax(0, 1fr) 1.3rem; gap: 0.3rem; align-items: center;
}
.${p}-depth-bar-row span { font-size: 0.6875rem; font-weight: 800; color: var(--muted); }
.${p}-depth-bar-row em {
  font-style: normal; font-size: 0.6875rem; font-weight: 900; color: var(--text); text-align: right;
}
.${p}-depth-track {
  height: 0.32rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(12, 35, 52, 0.9);
  overflow: hidden;
}
.${p}-depth-fill {
  display: block; height: 100%; border-radius: 0; background: var(--cyan, #13d8e7);
}
.${p}-org-note {
  margin: 0.2rem 0 0; font-size: 0.6875rem; color: rgba(180, 206, 220, 0.92); line-height: 1.25;
  display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden;
}
.${p}-more-details { flex-shrink: 0; }
.${p}-more-toggle {
  border: 0; background: transparent; color: var(--cyan); cursor: pointer;
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;
  padding: 0; font-family: inherit;
}
.${p}-more-toggle:hover, .${p}-more-toggle:focus-visible { color: var(--text); }
.${p}-more-panel {
  margin-top: 0.2rem; padding: 0.35rem 0.45rem; max-height: 4.5rem; overflow: auto;
  border: 1px solid rgba(156, 218, 236, 0.16); border-radius: 8px;
  background: rgba(8, 22, 34, 0.72); font-size: 0.6875rem; color: rgba(233, 247, 251, 0.82);
  line-height: 1.3;
}
.${p}-more-panel p { margin: 0 0 0.25rem; }
.${p}-more-panel p:last-child { margin-bottom: 0; }
.${p}-sr-only, .${p}-root .sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
.${p}-haul-card {
  grid-template-columns: 2.2rem 2rem minmax(0, 1fr) auto;
  gap: 0.28rem; padding: 0.28rem 0.3rem; align-items: center; border-radius: 7px;
}
.${p}-haul-card.is-active {
  background: rgba(19, 216, 231, 0.14);
  border-left: 3px solid var(--gold);
  box-shadow: inset 0 0 0 1px rgba(233, 168, 60, 0.2);
}
.${p}-haul-pos {
  width: 1.75rem; height: 1.75rem; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.6875rem; font-weight: 900; color: var(--cyan);
  background: rgba(19, 216, 231, 0.1); border: 1px solid rgba(19, 216, 231, 0.25);
}
.${p}-haul-pos.pos-f { color: #9be7b8; background: rgba(82, 223, 148, 0.1); border-color: rgba(82, 223, 148, 0.3); }
.${p}-haul-pos.pos-d { color: #8ab4ff; background: rgba(138, 180, 255, 0.1); border-color: rgba(138, 180, 255, 0.3); }
.${p}-haul-pos.pos-g { color: #ffd88d; background: rgba(233, 168, 60, 0.12); border-color: rgba(233, 168, 60, 0.35); }
.${p}-haul-rank {
  font-size: 0.6875rem; font-weight: 900; color: var(--muted);
  letter-spacing: 0.02em; line-height: 1.15; font-variant-numeric: tabular-nums;
}
.${p}-haul-body strong {
  font-size: 0.74rem; margin-bottom: 0.02rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-haul-body .${p}-card-details {
  flex-direction: row; flex-wrap: nowrap; gap: 0.28rem; font-size: 0.6875rem; overflow: hidden;
}
.${p}-haul-body .${p}-card-details span {
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-haul-grade-letter { font-size: 0.84rem; font-weight: 900; min-width: 1.3rem; text-align: right; }
@media (max-width: 1200px) {
  .${p}-haul-banner { grid-template-columns: auto minmax(0, 1fr); }
  .${p}-haul-metrics { display: none; }
  .${p}-bottom-band { grid-template-columns: 1fr 1fr; }
  .${p}-org-col { grid-column: 1 / -1; }
}
@media (max-width: 900px) {
  .${p}-root, .${p}-reveal, .${p}-workspace, .${p}-stage, .${p}-panel { height: auto; max-height: none; }
  .${p}-reveal { overflow: auto; }
  .${p}-workspace { max-height: none; overflow: visible; }
  .${p}-stage-grid { grid-template-columns: 1fr; grid-template-rows: auto; overflow: visible; }
  .${p}-draft-card { grid-template-columns: auto minmax(0, 1fr); }
  .${p}-grade-shield { grid-column: 2; justify-self: end; }
  .${p}-hero-name, .${p}-hero-sub, .${p}-hero-meta { white-space: normal; }
  .${p}-bottom-band { grid-template-columns: 1fr; }
  .${p}-meter-stack { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .${p}-haul-card, .${p}-meter-fill { transition: none; }
}
`
      : "";

  const resignOnly =
    p === "resign"
      ? `
.${p}-root {
  --orange: #f0a35a;
  --orange-soft: rgba(240, 163, 90, 0.16);
  --cyan-glow: 0 0 0 1px rgba(19, 216, 231, 0.45), 0 0 18px rgba(19, 216, 231, 0.2);
  height: 100vh; max-height: 100vh; overflow: hidden;
}
.${p}-stage {
  grid-template-columns: minmax(0, 1fr) min(280px, 24vw);
  gap: 0.85rem;
  padding: 0 1.1rem 4.25rem;
  height: calc(100vh - 64px);
  max-height: calc(100vh - 64px);
  min-height: 0;
  align-items: stretch;
}
.${p}-reveal {
  min-height: 0; overflow: hidden; flex: 1;
  padding: 0.35rem 0 0; justify-content: flex-start;
}
.${p}-workspace {
  gap: 0.55rem; min-height: 0; flex: 1; overflow: hidden;
  max-height: 100%;
}
.${p}-footer.is-compact {
  padding: 0.55rem 1.2rem 0.7rem;
  background: linear-gradient(0deg, rgba(2, 10, 17, 0.98), rgba(6, 19, 31, 0.94));
}
.${p}-footer.is-compact .${p}-footer-actions {
  width: 100%;
  align-items: center;
}
.${p}-footer.is-compact .${p}-cta-btn {
  margin-left: auto;
  min-width: 220px;
  padding: 0.75rem 1.4rem;
}
.${p}-footer.is-compact .${p}-cta-btn:disabled {
  opacity: 0.4; filter: grayscale(0.2); box-shadow: none;
}
.${p}-panel {
  border: 1px solid rgba(156, 218, 236, 0.24);
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
  background: linear-gradient(180deg, rgba(10, 28, 42, 0.96), rgba(6, 18, 28, 0.96));
  max-height: 100%;
  min-height: 0;
  padding: 0.55rem 0.55rem 0.4rem;
}
.${p}-panel h2 {
  color: var(--gold); letter-spacing: 0.14em; font-size: 0.6875rem; margin-bottom: 0.4rem;
}
.${p}-office-bar {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 0.85rem;
  padding: 0.35rem 0.15rem 0.1rem;
  border-bottom: 1px solid rgba(156, 218, 236, 0.12);
}
.${p}-office-count {
  font-size: 1.05rem; font-weight: 1000; color: var(--text); letter-spacing: 0.02em;
}
.${p}-office-count span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); margin-left: 0.25rem;
}
.${p}-office-meta, .${p}-office-slots {
  font-size: 0.72rem; font-weight: 700; color: var(--muted);
}
.${p}-office-cap {
  font-size: 0.82rem; font-weight: 900; color: var(--green);
}
.${p}-office-warn {
  font-style: normal; font-size: 0.7rem; color: var(--gold); width: 100%;
}
.${p}-filter-row { gap: 0.3rem; }
.${p}-filter-chip {
  padding: 0.3rem 0.6rem; font-size: 0.6875rem; letter-spacing: 0.08em;
  border-radius: var(--radius-ops, 2px); border-color: rgba(156, 218, 236, 0.18);
  background: rgba(10, 26, 40, 0.65);
  transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
}
.${p}-filter-chip:hover {
  border-color: rgba(19, 216, 231, 0.35); color: var(--text);
}
.${p}-filter-chip.is-active {
  border-color: rgba(233, 168, 60, 0.5); color: #fff3d6;
  background: rgba(233, 168, 60, 0.14);
}
.${p}-filter-chip em { margin-left: 0.3rem; color: var(--cyan); font-style: normal; }
.${p}-table-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(340px, 36%);
  gap: 0.75rem; min-height: 0; flex: 1; overflow: hidden;
}
.${p}-table-wrap {
  min-height: 0; overflow: auto;
  border: 1px solid rgba(156, 218, 236, 0.22);
  border-radius: 10px;
  background: rgba(6, 18, 28, 0.7);
}
.${p}-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.${p}-table thead th {
  position: sticky; top: 0; z-index: 2;
  padding: 0.5rem 0.55rem; font-size: 0.6875rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); font-weight: 900;
  background: rgba(9, 25, 38, 0.98);
  border-bottom: 1px solid rgba(156, 218, 236, 0.16);
  white-space: nowrap; cursor: pointer;
}
.${p}-table tbody td {
  padding: 0.48rem 0.55rem; font-size: 0.8rem;
  border-bottom: 1px solid rgba(156, 218, 236, 0.07);
  vertical-align: middle; color: var(--text);
}
.${p}-table tbody tr {
  opacity: 0.82; cursor: pointer;
  transition: background 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
}
.${p}-table tbody tr:nth-child(even) { background: rgba(255, 255, 255, 0.015); }
.${p}-table tbody tr:hover {
  opacity: 1; background: rgba(19, 216, 231, 0.07);
}
.${p}-table tbody tr.is-actionable { opacity: 0.94; }
.${p}-table tbody tr.is-selected {
  opacity: 1;
  background: linear-gradient(90deg, rgba(19, 216, 231, 0.18), rgba(19, 216, 231, 0.05) 50%);
  box-shadow: inset 3px 0 0 var(--gold), var(--cyan-glow);
}
.${p}-player-cell { display: flex; align-items: center; gap: 0.4rem; min-width: 0; }
.${p}-player-cell strong {
  font-size: 0.86rem; font-weight: 900;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-shot-frame {
  display: inline-flex; width: 2.1rem; height: 2.1rem; border-radius: 999px; padding: 1px;
  background: linear-gradient(145deg, rgba(19, 216, 231, 0.5), rgba(233, 168, 60, 0.3));
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.${p}-shot-frame .player-headshot { border-radius: 999px; overflow: hidden; }
.${p}-ovr { font-weight: 1000; font-size: 0.86rem; }
.${p}-ovr.tone-gold { color: #f0c35a; }
.${p}-ovr.tone-blue { color: #8ab4ff; }
.${p}-ovr.tone-grey { color: #9aa8b5; }
.${p}-status-chip {
  display: inline-flex; padding: 0.12rem 0.4rem; border-radius: var(--radius-ops, 2px);
  border: 1px solid rgba(156, 218, 236, 0.18); background: rgba(12, 31, 47, 0.5);
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
}
.${p}-status-chip.tone-accepted {
  color: #7dffb4; background: rgba(82, 223, 148, 0.12); border-color: rgba(82, 223, 148, 0.35);
}
.${p}-status-chip.tone-rejected {
  color: #ff8f98; background: rgba(255, 96, 109, 0.12); border-color: rgba(255, 96, 109, 0.35);
}
.${p}-status-chip.tone-countered {
  color: #ffd36a; background: rgba(233, 168, 60, 0.12); border-color: rgba(233, 168, 60, 0.35);
}
.${p}-status-chip.tone-pending {
  color: #8ab4ff; background: rgba(138, 180, 255, 0.12); border-color: rgba(138, 180, 255, 0.32);
}
.${p}-status-chip.tone-released {
  color: #c9d4de; background: rgba(156, 176, 196, 0.12); border-color: rgba(156, 176, 196, 0.32);
}
.${p}-status-chip.tone-lapsed {
  color: #b8a48a; background: rgba(184, 164, 138, 0.12); border-color: rgba(184, 164, 138, 0.32);
}
.${p}-interest-pill {
  display: inline-flex; align-items: center; gap: 0.28rem;
  padding: 0.12rem 0.42rem; border-radius: var(--radius-ops, 2px);
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.06em; text-transform: uppercase;
  border: 1px solid transparent;
}
.${p}-interest-pill::before {
  content: ""; width: 0.35rem; height: 0.35rem; border-radius: 999px; background: currentColor;
}
.${p}-interest-pill.tone-high {
  color: #7dffb4; background: rgba(82, 223, 148, 0.1); border-color: rgba(82, 223, 148, 0.3);
}
.${p}-interest-pill.tone-med {
  color: #ffd36a; background: rgba(233, 168, 60, 0.1); border-color: rgba(233, 168, 60, 0.3);
}
.${p}-interest-pill.tone-low {
  color: #ff8f98; background: rgba(255, 96, 109, 0.1); border-color: rgba(255, 96, 109, 0.3);
}
.${p}-row-action {
  padding: 0.28rem 0.55rem; font-size: 0.6875rem; border-radius: 7px;
  border: 1px solid rgba(233, 168, 60, 0.45);
  background: rgba(233, 168, 60, 0.14); color: #ffe4a8;
  font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  cursor: pointer; font-family: inherit;
}
.${p}-row-action:hover:not(:disabled) { filter: brightness(1.08); }
.${p}-muted { color: var(--muted); }
.${p}-detail-panel {
  min-height: 0; overflow: hidden;
  display: flex; flex-direction: column; gap: 0.5rem;
  border: 1px solid rgba(73, 231, 240, 0.32);
  border-radius: 12px;
  background:
    radial-gradient(circle at 16% 0%, rgba(19, 216, 231, 0.1), transparent 40%),
    linear-gradient(180deg, rgba(14, 38, 56, 0.97), rgba(8, 22, 34, 0.97));
  padding: 0.7rem 0.75rem 0.7rem;
  box-shadow: 0 0 0 1px rgba(19, 216, 231, 0.06), 0 16px 40px rgba(0, 0, 0, 0.36);
}
.${p}-identity {
  display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.7rem; align-items: center;
  flex: 0 0 auto;
}
.${p}-identity-shot {
  display: flex; width: 4.4rem; height: 4.4rem; border-radius: 999px; padding: 2px;
  background:
    radial-gradient(circle at 32% 22%, rgba(255, 255, 255, 0.35), transparent 42%),
    linear-gradient(150deg, rgba(19, 216, 231, 0.55), rgba(233, 168, 60, 0.35));
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
}
.${p}-identity-shot .player-headshot { border-radius: 999px; overflow: hidden; }
.${p}-identity-copy { min-width: 0; display: flex; flex-direction: column; gap: 0.1rem; }
.${p}-ovr-badge {
  display: inline-flex; align-items: baseline; gap: 0.3rem;
  font-size: 1.25rem; font-weight: 1000; line-height: 1;
}
.${p}-ovr-badge em {
  font-style: normal; font-size: 0.6875rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); font-weight: 900;
}
.${p}-ovr-badge.tone-gold { color: #f0c35a; }
.${p}-ovr-badge.tone-blue { color: #8ab4ff; }
.${p}-ovr-badge.tone-grey { color: #9aa8b5; }
.${p}-identity-copy h3 {
  margin: 0; font-size: 1.1rem; font-weight: 1000; letter-spacing: 0.03em; line-height: 1.15;
}
.${p}-identity-copy p {
  margin: 0; font-size: 0.72rem; color: var(--muted); font-weight: 700;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-detail-tabs { display: flex; gap: 0.3rem; flex: 0 0 auto; }
.${p}-detail-tabs button {
  flex: 1; border: 1px solid var(--line); background: transparent; color: var(--muted);
  border-radius: 7px; padding: 0.38rem; font-size: 0.6875rem; font-weight: 800;
  letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer; font-family: inherit;
}
.${p}-detail-tabs button.is-active {
  color: var(--text); border-color: var(--gold); background: var(--gold-soft);
}
.${p}-detail-tabs button:disabled { opacity: 0.35; cursor: not-allowed; }
.${p}-detail-body {
  display: flex; flex-direction: column; gap: 0.45rem; min-height: 0; flex: 1; overflow: auto;
}
.${p}-nego-body {
  overflow: hidden; gap: 0.55rem;
}
.${p}-nego-scroll {
  display: flex; flex-direction: column; gap: 0.5rem; min-height: 0; flex: 1; overflow: auto;
  padding-right: 0.15rem;
}
.${p}-cap-banner {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem;
  padding: 0.55rem 0.6rem; border-radius: 10px;
  border: 1px solid rgba(82, 223, 148, 0.25);
  background: rgba(82, 223, 148, 0.08);
}
.${p}-cap-banner > div { display: flex; flex-direction: column; gap: 0.1rem; }
.${p}-cap-banner span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-cap-banner strong { font-size: 1.15rem; font-weight: 1000; line-height: 1.1; }
.${p}-cap-banner strong.is-green { color: var(--green); }
.${p}-cap-banner strong.is-red { color: var(--red); }
.${p}-offer-compare {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem;
}
.${p}-offer-compare > div {
  padding: 0.5rem 0.55rem; border-radius: 9px;
  border: 1px solid rgba(156, 218, 236, 0.14);
  background: rgba(6, 18, 28, 0.5);
  display: flex; flex-direction: column; gap: 0.12rem;
}
.${p}-offer-compare span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-offer-compare strong {
  display: flex; align-items: baseline; gap: 0.35rem;
  font-size: 1.05rem; font-weight: 1000; color: var(--text);
}
.${p}-offer-compare em {
  font-style: normal; font-size: 0.72rem; font-weight: 800; color: var(--muted);
}
.${p}-offer-diff-line {
  margin: 0; font-size: 0.78rem; font-weight: 800;
}
.${p}-offer-diff-line.is-green { color: var(--green); }
.${p}-offer-diff-line.is-warn { color: var(--orange); }
.${p}-offer-diff-line.is-cyan { color: var(--cyan); }
.${p}-special-actions { display: flex; flex-direction: column; gap: 0.3rem; }
.${p}-deal-controls {
  display: flex; flex-direction: column; gap: 0.55rem;
  padding: 0.55rem 0.6rem; border-radius: 10px;
  border: 1px solid rgba(156, 218, 236, 0.14);
  background: rgba(6, 18, 28, 0.4);
}
.${p}-field {
  display: flex; flex-direction: column; gap: 0.25rem;
  font-size: 0.6875rem; color: var(--muted); font-weight: 800;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.${p}-select-wrap {
  position: relative; border-radius: 8px;
  border: 1px solid rgba(156, 218, 236, 0.2);
  background: rgba(8, 24, 36, 0.72);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.${p}-select-wrap:hover {
  border-color: rgba(19, 216, 231, 0.4);
  box-shadow: 0 0 12px rgba(19, 216, 231, 0.1);
}
.${p}-select-wrap::after {
  content: ""; position: absolute; right: 0.75rem; top: 50%;
  width: 0.45rem; height: 0.45rem;
  border-right: 2px solid var(--gold); border-bottom: 2px solid var(--gold);
  transform: translateY(-65%) rotate(45deg); pointer-events: none;
}
.${p}-select-wrap select {
  appearance: none; -webkit-appearance: none; width: 100%; border: 0;
  background: transparent; color: var(--text); font: inherit;
  font-size: 0.84rem; font-weight: 800;
  padding: 0.55rem 1.9rem 0.55rem 0.7rem; cursor: pointer;
}
.${p}-slider-block, .${p}-term-block {
  display: flex; flex-direction: column; gap: 0.3rem;
}
.${p}-slider-head {
  display: flex; justify-content: space-between; align-items: baseline;
}
.${p}-slider-head span, .${p}-mini-label {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-slider-head strong { font-size: 1rem; font-weight: 1000; color: var(--gold); }
/* Salary control is a contract-demand ruler with a squared term marker. */
.${p}-salary-slider {
  -webkit-appearance: none; appearance: none; width: 100%; height: 0.3rem;
  border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.16) 0 1px, transparent 1px 20%),
    linear-gradient(90deg, rgba(19, 216, 231, 0.25), rgba(233, 168, 60, 0.45));
  outline: none; cursor: pointer;
}
.${p}-salary-slider::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 0.7rem; height: 1.1rem; border-radius: 2px;
  border: 1px solid #fff6d8;
  background: #e9a83c;
  box-shadow: none; cursor: pointer;
}
.${p}-salary-slider::-moz-range-thumb {
  width: 0.7rem; height: 1.1rem; border-radius: 2px; border: 1px solid #fff6d8;
  background: #e9a83c; cursor: pointer;
}
.${p}-term-seg {
  display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 0.22rem;
}
.${p}-term-seg button {
  border: 1px solid rgba(156, 218, 236, 0.18);
  background: rgba(8, 22, 34, 0.7); color: var(--muted);
  border-radius: 7px; padding: 0.42rem 0.1rem;
  font-size: 0.78rem; font-weight: 900; font-family: inherit; cursor: pointer;
}
.${p}-term-seg button:hover:not(:disabled) {
  border-color: rgba(19, 216, 231, 0.35); color: var(--text);
}
.${p}-term-seg button.is-active {
  color: #1b1002; border-color: rgba(233, 168, 60, 0.65);
  background: linear-gradient(180deg, #f4bd52, #d99023);
}
.${p}-check-row {
  display: flex; flex-wrap: wrap; gap: 0.65rem;
  font-size: 0.72rem; color: rgba(233, 247, 251, 0.82);
}
.${p}-check-row label { display: inline-flex; align-items: center; gap: 0.25rem; }
.${p}-dossier-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
}
.${p}-dossier-grid > div {
  padding: 0.45rem 0.5rem; border-radius: 8px;
  border: 1px solid rgba(156, 218, 236, 0.12); background: rgba(6, 18, 28, 0.45);
  display: flex; flex-direction: column; gap: 0.1rem;
}
.${p}-dossier-grid span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-dossier-grid strong { font-size: 0.9rem; font-weight: 900; color: var(--text); }
.${p}-negotiate-actions {
  display: flex; flex-direction: column; gap: 0.35rem;
  flex: 0 0 auto; margin-top: auto;
  padding-top: 0.35rem;
  border-top: 1px solid rgba(156, 218, 236, 0.12);
}
.${p}-preview-btn {
  width: 100%; justify-content: center; border-radius: 9px;
  border: 1px solid rgba(156, 218, 236, 0.32);
  background: transparent; color: rgba(233, 247, 251, 0.88);
  padding: 0.55rem 0.85rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase;
}
.${p}-preview-btn:hover {
  border-color: rgba(19, 216, 231, 0.45); background: rgba(19, 216, 231, 0.07);
}
.${p}-submit-btn, .${p}-negotiate-actions .${p}-cta-btn {
  width: 100%; flex: none; min-width: 0;
  padding: 0.82rem 1rem; font-size: 0.76rem; border-radius: 9px;
  letter-spacing: 0.12em;
}
.${p}-rail-btn {
  border: 0; background: transparent; padding: 0; text-align: left;
  cursor: pointer; font: inherit; color: inherit; width: 100%;
}
.${p}-queue-card {
  display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.5rem;
  padding: 0.55rem 0.6rem; border-radius: 10px;
  border: 1px solid rgba(156, 218, 236, 0.12);
  background: rgba(10, 28, 42, 0.72);
  transition: border-color 0.15s ease, background 0.15s ease;
}
.${p}-queue-card:hover { border-color: rgba(19, 216, 231, 0.3); }
.${p}-queue-card.is-active {
  border-color: rgba(233, 168, 60, 0.5);
  background: rgba(233, 168, 60, 0.1);
}
.${p}-queue-card .${p}-ovr { font-size: 1.1rem; line-height: 1; }
.${p}-queue-body { display: flex; flex-direction: column; gap: 0.12rem; min-width: 0; }
.${p}-queue-body strong {
  font-size: 0.84rem; font-weight: 900;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-queue-status {
  font-size: 0.6875rem; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.${p}-queue-body em {
  font-style: normal; font-size: 0.78rem; font-weight: 900; color: var(--gold);
}
.${p}-continue-note { margin: 0; font-size: 0.7rem; flex: 0 0 auto; }
.${p}-context { font-size: 0.72rem; color: rgba(233, 247, 251, 0.72); margin: 0; line-height: 1.35; }
.${p}-office-window {
  font-size: 0.72rem; font-weight: 800; color: var(--cyan);
  letter-spacing: 0.04em; text-transform: uppercase;
}
.${p}-office-flash {
  font-style: normal; font-size: 0.72rem; color: var(--gold); width: 100%;
}
.${p}-sim-day-btn {
  border: 1px solid rgba(19, 216, 231, 0.4);
  background: rgba(19, 216, 231, 0.12); color: var(--text);
  border-radius: var(--radius-hud, 4px); padding: 0.28rem 0.75rem;
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.1em;
  text-transform: uppercase; cursor: pointer; font-family: inherit;
}
.${p}-sim-day-btn:hover:not(:disabled) {
  border-color: rgba(19, 216, 231, 0.65); background: rgba(19, 216, 231, 0.2);
}
.${p}-sim-day-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.${p}-nego-meter {
  display: flex; flex-direction: column; gap: 0.28rem;
  padding: 0.5rem 0.55rem; border-radius: 10px;
  border: 1px solid rgba(156, 218, 236, 0.16);
  background: rgba(6, 18, 28, 0.45);
}
.${p}-nego-meter-head {
  display: flex; justify-content: space-between; align-items: baseline;
}
.${p}-nego-meter-head span {
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-nego-meter-head strong { font-size: 1.05rem; font-weight: 1000; color: var(--text); }
.${p}-nego-meter-track {
  position: relative; height: 0.55rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(12, 31, 47, 0.9);
  overflow: hidden;
}
.${p}-nego-meter-fill {
  display: block; height: 100%; border-radius: 0;
  background: linear-gradient(90deg, #ff606d, #f0a35a 40%, #52df94 70%, #f4bd52);
  transition: opacity 0.2s ease;
}
.${p}-nego-meter-fill.tone-bad { background: linear-gradient(90deg, #ff606d, #f0a35a); }
.${p}-nego-meter-fill.tone-mid { background: linear-gradient(90deg, #f0a35a, #e9a83c); }
.${p}-nego-meter-fill.tone-good { background: linear-gradient(90deg, #52df94, #13d8e7); }
.${p}-nego-meter-fill.tone-instant { background: linear-gradient(90deg, #f4bd52, #fff0c2); }
.${p}-nego-meter-mark {
  position: absolute; top: -2px; bottom: -2px; width: 2px;
  background: rgba(233, 247, 251, 0.55); transform: translateX(-50%);
}
.${p}-nego-meter-mark.is-instant { background: var(--gold); }
.${p}-nego-meter-note {
  margin: 0; font-size: 0.7rem; color: rgba(233, 247, 251, 0.72); line-height: 1.3;
}
.${p}-check-row.is-clauses label.is-on { color: var(--gold); font-weight: 800; }
.${p}-check-row.is-clauses em {
  font-style: normal; margin-left: 0.25rem; font-size: 0.6875rem;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--cyan);
}
@media (max-width: 1280px) {
  .${p}-table-layout { grid-template-columns: minmax(0, 1fr) minmax(300px, 38%); }
  .${p}-stage { grid-template-columns: minmax(0, 1fr) min(250px, 26vw); }
}
@media (max-width: 1100px) {
  .${p}-table-layout { grid-template-columns: 1fr; }
  .${p}-detail-panel { max-height: min(46vh, 460px); }
  .${p}-stage { grid-template-columns: 1fr; height: auto; max-height: none; overflow: auto; }
  .${p}-root { height: auto; max-height: none; overflow: auto; }
}
`
      : "";

  const faOnly =
    p === "fa"
      ? `
.${p}-root {
  --orange: #f0a35a; --green: #52df94; --cyan: #13d8e7; --gold: #e9a83c; --red: #ff606d;
  height: 100vh; max-height: 100vh; overflow: hidden;
}
.${p}-stage {
  grid-template-columns: minmax(0, 1fr) min(280px, 24vw);
  gap: 0.85rem; padding: 0 1.1rem 4.1rem;
  height: calc(100vh - 64px); max-height: calc(100vh - 64px); min-height: 0;
}
.${p}-reveal {
  display: flex; flex-direction: column; justify-content: flex-start;
  min-height: 0; height: 100%; overflow: hidden; padding-top: 0.35rem;
}
.${p}-workspace {
  display: flex; flex-direction: column;
  gap: 0.5rem; min-height: 0; flex: 1; overflow: hidden;
}
.${p}-footer.is-compact { padding: 0.55rem 1.2rem 0.7rem; }
.${p}-panel {
  max-height: 100%; height: 100%; min-height: 0;
  display: flex; flex-direction: column; overflow: hidden;
}
.${p}-rail {
  min-height: 0; flex: 1; overflow-y: auto; overflow-x: hidden;
  scrollbar-width: thin; scrollbar-color: rgba(19, 216, 231, 0.45) transparent;
  padding-right: 0.2rem;
}
.${p}-rail::-webkit-scrollbar { width: 8px; }
.${p}-rail::-webkit-scrollbar-thumb {
  background: rgba(19, 216, 231, 0.4); border-radius: 999px;
}
.${p}-fa-prev { display: inline-flex; }
.${p}-fa-prev .team-logo-badge,
.${p}-fa-logos .team-logo-badge,
.${p}-fa-offer .team-logo-badge {
  width: var(--team-logo-size, 28px); height: var(--team-logo-size, 28px);
  border-radius: 999px; overflow: hidden; display: inline-flex;
  align-items: center; justify-content: center;
  background: rgba(8,22,34,0.85); border: 1px solid rgba(156,218,236,0.2);
  flex: 0 0 auto;
}
.${p}-fa-prev .team-logo-badge img,
.${p}-fa-logos .team-logo-badge img,
.${p}-fa-offer .team-logo-badge img {
  width: 100%; height: 100%; object-fit: contain;
}
.${p}-fa-prev .team-logo-badge span,
.${p}-fa-logos .team-logo-badge span,
.${p}-fa-offer .team-logo-badge span {
  font-size: 0.6875rem; font-weight: 900; color: var(--text);
}
.${p}-fa-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.75rem;
  padding: 0.35rem 0.1rem; border-bottom: 1px solid rgba(156,218,236,0.12);
  font-size: 0.72rem; color: var(--muted); font-weight: 700;
}
.${p}-fa-bar strong { color: var(--text); font-size: 0.9rem; letter-spacing: 0.04em; text-transform: uppercase; }
.${p}-fa-cap { color: var(--green); font-weight: 900; }
.${p}-fa-bar .is-ok { color: var(--green); }
.${p}-fa-bar .is-bad { color: var(--orange); }
.${p}-sim-btn {
  border: 1px solid rgba(19,216,231,0.35); background: rgba(19,216,231,0.1);
  color: var(--text); border-radius: var(--radius-hud, 4px); padding: 0.28rem 0.7rem;
  font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase;
  cursor: pointer; font-family: inherit;
}
.${p}-sim-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.${p}-fa-tools { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.${p}-fa-search {
  flex: 1; min-width: 12rem; border: 1px solid rgba(156,218,236,0.2);
  background: rgba(8,22,34,0.7); color: var(--text); border-radius: 8px;
  padding: 0.45rem 0.7rem; font: inherit; font-size: 0.82rem;
}
.${p}-fa-filters { display: flex; flex-wrap: wrap; gap: 0.25rem; }
.${p}-fa-chip {
  border: 1px solid rgba(156,218,236,0.18); background: rgba(10,26,40,0.65);
  color: var(--muted); border-radius: var(--radius-ops, 2px); padding: 0.28rem 0.55rem;
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  cursor: pointer; font-family: inherit;
}
.${p}-fa-chip.is-active { border-color: rgba(233,168,60,0.5); color: #fff3d6; background: rgba(233,168,60,0.14); }
.${p}-fa-sort {
  border: 1px solid rgba(156,218,236,0.2); background: rgba(8,22,34,0.7);
  color: var(--text); border-radius: 8px; padding: 0.4rem 0.55rem; font: inherit; font-size: 0.72rem;
}
.${p}-fa-layout {
  display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(360px, 40%);
  gap: 0.75rem; min-height: 0; flex: 1; overflow: hidden; align-items: stretch;
}
.${p}-fa-list {
  min-height: 0; max-height: 100%; height: 100%;
  overflow-y: auto; overflow-x: hidden;
  border: 1px solid rgba(156,218,236,0.2);
  border-radius: 10px; background: rgba(6,18,28,0.65); padding: 0.35rem;
  scrollbar-width: thin; scrollbar-color: rgba(19, 216, 231, 0.5) rgba(8,22,34,0.6);
}
.${p}-fa-list::-webkit-scrollbar { width: 10px; }
.${p}-fa-list::-webkit-scrollbar-track {
  background: rgba(8,22,34,0.55); border-radius: 999px;
}
.${p}-fa-list::-webkit-scrollbar-thumb {
  background: rgba(19, 216, 231, 0.45); border-radius: 999px;
  border: 2px solid rgba(8,22,34,0.55);
}
.${p}-fa-count {
  margin: 0.15rem 0.35rem 0.4rem; font-size: 0.6875rem; font-weight: 900;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
}
.${p}-fa-row {
  width: 100%; display: grid;
  grid-template-columns: auto minmax(0,1fr) auto auto auto;
  gap: 0.45rem; align-items: center; text-align: left;
  border: 1px solid transparent; background: transparent; color: inherit;
  border-radius: 9px; padding: 0.4rem 0.45rem; cursor: pointer; font: inherit;
  margin-bottom: 0.15rem;
}
.${p}-fa-row:hover { background: rgba(19,216,231,0.06); }
.${p}-fa-row.is-selected {
  background: linear-gradient(90deg, rgba(19,216,231,0.16), rgba(19,216,231,0.04));
  border-color: rgba(19,216,231,0.28);
  box-shadow: inset 3px 0 0 var(--gold);
}
.${p}-fa-row-body { min-width: 0; display: flex; flex-direction: column; gap: 0.08rem; }
.${p}-fa-row-body strong {
  font-size: 0.86rem; font-weight: 900; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-fa-row-body em { font-style: normal; font-size: 0.6875rem; color: var(--muted); }
.${p}-fa-row-meta { font-size: 0.6875rem; color: rgba(233,247,251,0.65); text-transform: capitalize; }
.${p}-fa-logos { display: inline-flex; gap: 0.15rem; align-items: center; }
.${p}-fa-logos.is-large { gap: 0.35rem; margin-top: 0.25rem; }
.${p}-fa-ask { font-size: 0.82rem; font-weight: 900; color: var(--gold); white-space: nowrap; }
.${p}-fa-star { color: var(--muted); font-size: 0.9rem; padding: 0.15rem; }
.${p}-fa-star.is-on { color: var(--gold); }
.${p}-fa-desk {
  min-height: 0; overflow: auto; border: 1px solid rgba(73,231,240,0.3);
  border-radius: 12px; padding: 0.7rem 0.75rem;
  background:
    radial-gradient(circle at 16% 0%, rgba(19,216,231,0.1), transparent 40%),
    linear-gradient(180deg, rgba(14,38,56,0.97), rgba(8,22,34,0.97));
  display: flex; flex-direction: column; gap: 0.55rem;
  box-shadow: 0 16px 40px rgba(0,0,0,0.36);
}
.${p}-fa-identity {
  display: grid; grid-template-columns: auto minmax(0,1fr); gap: 0.7rem; align-items: center;
}
.${p}-fa-shot {
  width: 4.4rem; height: 4.4rem; border-radius: 999px; padding: 2px;
  background: linear-gradient(150deg, rgba(19,216,231,0.55), rgba(233,168,60,0.35));
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.${p}-fa-shot .player-headshot { border-radius: 999px; overflow: hidden; }
.${p}-fa-ovr { font-size: 1.15rem; font-weight: 1000; color: #f0c35a; }
.${p}-fa-identity h3 {
  margin: 0; font-size: 1.15rem; font-weight: 1000; letter-spacing: 0.03em; line-height: 1.15;
}
.${p}-fa-identity p { margin: 0.1rem 0 0; font-size: 0.72rem; color: var(--muted); font-weight: 700; }
.${p}-fa-prev-line { color: rgba(233,247,251,0.7) !important; }
.${p}-fa-capbar {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
  padding: 0.5rem 0.55rem; border-radius: 10px;
  border: 1px solid rgba(82,223,148,0.22); background: rgba(82,223,148,0.07);
}
.${p}-fa-capbar span { font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.${p}-fa-capbar strong { display: block; font-size: 1.05rem; font-weight: 1000; }
.${p}-fa-capbar strong.is-green { color: var(--green); }
.${p}-fa-capbar strong.is-red { color: var(--red); }
.${p}-fa-captrack {
  grid-column: 1 / -1; height: 0.35rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(8,22,34,0.8);
  overflow: hidden;
}
.${p}-fa-captrack span { display: block; height: 100%; background: linear-gradient(90deg, #13d8e7, #e9a83c); }
.${p}-nego-meter {
  display: flex; flex-direction: column; gap: 0.25rem;
  padding: 0.5rem 0.55rem; border-radius: 10px;
  border: 1px solid rgba(156,218,236,0.16); background: rgba(6,18,28,0.45);
}
.${p}-nego-meter-head { display: flex; justify-content: space-between; }
.${p}-nego-meter-head span { font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.${p}-nego-meter-head strong { font-size: 1rem; font-weight: 1000; }
.${p}-nego-meter-track {
  height: 0.5rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(12,31,47,0.9);
  overflow: hidden;
}
.${p}-nego-meter-fill { display: block; height: 100%; border-radius: 0; background: linear-gradient(90deg,#ff606d,#f0a35a 40%,#52df94 70%,#f4bd52); }
.${p}-nego-meter-fill.tone-bad { background: linear-gradient(90deg,#ff606d,#f0a35a); }
.${p}-nego-meter-fill.tone-mid { background: linear-gradient(90deg,#f0a35a,#e9a83c); }
.${p}-nego-meter-fill.tone-good { background: linear-gradient(90deg,#52df94,#13d8e7); }
.${p}-nego-meter-fill.tone-instant { background: linear-gradient(90deg,#f4bd52,#fff0c2); }
.${p}-nego-meter-note { margin: 0; font-size: 0.7rem; color: rgba(233,247,251,0.72); }
.${p}-fa-stats {
  display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 0.3rem;
}
.${p}-fa-stats > div {
  padding: 0.35rem; border-radius: 8px; border: 1px solid rgba(156,218,236,0.12);
  background: rgba(6,18,28,0.45); text-align: center;
}
.${p}-fa-stats span { display: block; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em; color: var(--muted); }
.${p}-fa-stats strong { font-size: 0.85rem; }
.${p}-fa-pctiles { display: flex; flex-direction: column; gap: 0.28rem; }
.${p}-fa-pctile {
  display: grid; grid-template-columns: 3.2rem 1fr 2.4rem; gap: 0.35rem; align-items: center;
  font-size: 0.6875rem;
}
.${p}-fa-pctile span { color: var(--muted); font-weight: 800; letter-spacing: 0.04em; }
.${p}-fa-pctile-track {
  height: 0.35rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0 1px, transparent 1px 25%),
    rgba(8, 22, 34, 0.85);
  overflow: hidden; border: 1px solid rgba(156, 218, 236, 0.12);
}
.${p}-fa-pctile-track i {
  display: block; height: 100%; border-radius: 0;
  background: linear-gradient(90deg, #13d8e7, #52df94 70%, #f4bd52);
}
.${p}-fa-pctile strong { text-align: right; font-size: 0.72rem; color: var(--text); }
.${p}-fa-history { display: flex; flex-direction: column; gap: 0.3rem; }
.${p}-fa-history-row {
  display: grid; grid-template-columns: 1fr; gap: 0.1rem;
  padding: 0.35rem 0.45rem; border-radius: 8px;
  border: 1px solid rgba(156, 218, 236, 0.12); background: rgba(6, 18, 28, 0.4);
}
.${p}-fa-history-row strong { font-size: 0.78rem; }
.${p}-fa-history-row em { font-style: normal; font-size: 0.6875rem; color: var(--muted); }
.${p}-fa-history-row span { font-size: 0.7rem; color: rgba(233, 247, 251, 0.78); }
.${p}-fa-offers { display: flex; flex-direction: column; gap: 0.3rem; }
.${p}-mini-label { font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.${p}-fa-offer {
  display: grid; grid-template-columns: auto auto 1fr; gap: 0.4rem; align-items: center;
  padding: 0.35rem 0.45rem; border-radius: 8px; border: 1px solid rgba(156,218,236,0.12);
  background: rgba(6,18,28,0.4);
}
.${p}-fa-offer strong { font-size: 0.78rem; }
.${p}-fa-offer em { font-style: normal; font-size: 0.72rem; color: var(--gold); justify-self: end; }
.${p}-fa-controls { display: flex; flex-direction: column; gap: 0.45rem; }
.${p}-field { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.6875rem; color: var(--muted); font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.${p}-select-wrap {
  position: relative; border-radius: 8px; border: 1px solid rgba(156,218,236,0.2);
  background: rgba(8,24,36,0.72);
}
.${p}-select-wrap::after {
  content: ""; position: absolute; right: 0.7rem; top: 50%; width: 0.4rem; height: 0.4rem;
  border-right: 2px solid var(--gold); border-bottom: 2px solid var(--gold);
  transform: translateY(-65%) rotate(45deg); pointer-events: none;
}
.${p}-select-wrap select {
  appearance: none; width: 100%; border: 0; background: transparent; color: var(--text);
  font: inherit; font-size: 0.84rem; font-weight: 800; padding: 0.5rem 1.8rem 0.5rem 0.65rem;
}
.${p}-slider-block, .${p}-term-block { display: flex; flex-direction: column; gap: 0.28rem; }
.${p}-slider-head { display: flex; justify-content: space-between; }
.${p}-slider-head span { font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.${p}-slider-head strong { font-size: 0.95rem; font-weight: 1000; color: var(--gold); }
.${p}-salary-slider {
  -webkit-appearance: none; appearance: none; width: 100%; height: 0.3rem; border-radius: 1px;
  background:
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.16) 0 1px, transparent 1px 20%),
    linear-gradient(90deg, rgba(19,216,231,0.25), rgba(233,168,60,0.45));
  outline: none;
}
.${p}-salary-slider::-webkit-slider-thumb {
  -webkit-appearance: none; width: 0.7rem; height: 1.1rem; border-radius: 2px; border: 1px solid #fff6d8;
  background: #e9a83c; cursor: pointer;
}
/* Term selector is a contract-length ruler: one continuous scale of years. */
.${p}-term-seg {
  display: grid; grid-template-columns: repeat(8, minmax(0,1fr)); gap: 0;
  border: 1px solid rgba(156,218,236,0.18); border-radius: var(--radius-ops, 2px); overflow: hidden;
}
.${p}-term-seg button {
  border: 0; border-right: 1px solid rgba(156,218,236,0.18);
  background: rgba(8,22,34,0.7); color: var(--muted);
  border-radius: 0; padding: 0.4rem 0.1rem; font-size: 0.75rem; font-weight: 900;
  font-variant-numeric: tabular-nums; cursor: pointer; font-family: inherit;
}
.${p}-term-seg button:last-child { border-right: 0; }
.${p}-term-seg button.is-active {
  color: #1b1002;
  background: #e9a83c;
}
.${p}-check-row { display: flex; gap: 0.7rem; font-size: 0.72rem; color: rgba(233,247,251,0.82); }
.${p}-check-row label { display: inline-flex; align-items: center; gap: 0.25rem; }
.${p}-check-row.is-clauses label.is-on { color: var(--gold); font-weight: 800; }
.${p}-negotiate-actions { display: flex; flex-direction: column; gap: 0.35rem; }
.${p}-submit-btn, .${p}-negotiate-actions .${p}-cta-btn {
  width: 100%; min-width: 0; padding: 0.82rem 1rem; font-size: 0.76rem; border-radius: 9px;
}
.${p}-wire-item {
  margin: 0 0 0.45rem; padding: 0.45rem 0.5rem; border-radius: 8px;
  border: 1px solid rgba(156,218,236,0.12); background: rgba(10,28,42,0.7);
  font-size: 0.72rem; color: rgba(233,247,251,0.82); line-height: 1.35;
}
.${p}-context { font-size: 0.72rem; color: rgba(233,247,251,0.72); margin: 0; }
.${p}-warn { color: var(--gold); font-size: 0.8rem; }
@media (max-width: 1200px) {
  .${p}-fa-layout { grid-template-columns: 1fr; }
  .${p}-fa-desk { max-height: min(48vh, 480px); }
  .${p}-stage { grid-template-columns: 1fr; height: auto; max-height: none; }
  .${p}-root { height: auto; max-height: none; overflow: auto; }
}
`
      : "";

  const prospectRightsOnly =
    p === "prospectrights"
      ? `
.${p}-root { overflow: hidden; height: 100vh; max-height: 100vh; }
.${p}-reveal {
  overflow: hidden; min-height: 0; flex: 1;
  padding: 0.3rem 0 0.1rem; justify-content: flex-start;
}
.${p}-workspace {
  display: flex; flex-direction: column;
  gap: 0.4rem; overflow: hidden; min-height: 0; flex: 1;
  max-height: calc(100vh - 148px);
}
.${p}-stage {
  min-height: 0; height: calc(100vh - 118px);
  max-height: calc(100vh - 118px);
  padding: 0 1.1rem 4.35rem; gap: 0.65rem;
  align-items: stretch;
}
.${p}-footer { padding: 0.3rem 1.15rem 0.5rem; }
.${p}-ticker { margin-bottom: 0.2rem; }
.${p}-panel {
  max-height: calc(100vh - 140px); height: calc(100vh - 140px);
  padding: 0.35rem 0.35rem 0.25rem; min-height: 0;
  border-color: rgba(233, 168, 60, 0.22);
}
.${p}-panel h2 {
  margin-bottom: 0.2rem; padding-bottom: 0.2rem; font-size: 0.6875rem; color: var(--gold);
}
.${p}-rail {
  gap: 0.12rem; overflow-y: auto; min-height: 0;
  scrollbar-width: thin; scrollbar-color: rgba(19, 216, 231, 0.35) transparent;
}
.${p}-rail::-webkit-scrollbar { width: 5px; }
.${p}-rail::-webkit-scrollbar-thumb {
  background: rgba(19, 216, 231, 0.35); border-radius: 999px;
}
.${p}-impact-strip {
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.35rem; flex-shrink: 0;
}
.${p}-impact-card {
  padding: 0.4rem 0.45rem; border-radius: 10px;
  background: linear-gradient(180deg, rgba(15, 46, 66, 0.55), rgba(8, 22, 34, 0.72));
  border: 1px solid rgba(156, 218, 236, 0.14);
  text-align: center; min-width: 0;
}
.${p}-impact-card span {
  display: block; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-impact-card strong {
  display: block; margin-top: 0.12rem; font-size: 1.05rem; font-weight: 900;
  color: var(--text); line-height: 1.1;
}
.${p}-impact-card em {
  display: block; margin-top: 0.08rem; font-style: normal; font-size: 0.6875rem;
  font-weight: 700; color: var(--cyan);
}
.${p}-impact-card.is-warn { border-color: rgba(233, 168, 60, 0.4); }
.${p}-impact-card.is-warn strong { color: #ffd88d; }
.${p}-nego-grid {
  display: grid; grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.25fr) minmax(0, 0.95fr);
  gap: 0.4rem; min-height: 0; flex: 1; overflow: hidden;
}
.${p}-nego-col {
  display: flex; flex-direction: column; gap: 0.35rem; min-height: 0; overflow: hidden;
}
.${p}-player-card {
  flex-shrink: 0; padding: 0.45rem 0.55rem; border-radius: 12px;
  background:
    linear-gradient(145deg, rgba(233, 168, 60, 0.12), transparent 42%),
    linear-gradient(180deg, rgba(15, 46, 66, 0.55), rgba(8, 22, 34, 0.78));
  border: 1px solid rgba(233, 168, 60, 0.28);
}
.${p}-player-card-top {
  display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.5rem; align-items: center;
}
.${p}-pos-badge {
  width: 3.2rem; height: 3.2rem; border-radius: 10px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: rgba(19, 216, 231, 0.1); border: 1px solid rgba(19, 216, 231, 0.35);
}
.${p}-pos-badge strong { font-size: 1.1rem; font-weight: 900; color: var(--cyan); line-height: 1; }
.${p}-pos-badge span {
  margin-top: 0.1rem; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-player-card h2 {
  margin: 0; font-size: 1.15rem; font-weight: 900; color: var(--text); line-height: 1.15;
}
.${p}-player-card .${p}-kicker {
  margin: 0 0 0.15rem; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--cyan);
}
.${p}-stat-grid {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.28rem; margin-top: 0.4rem;
}
.${p}-stat-cell {
  padding: 0.28rem 0.3rem; border-radius: 8px;
  background: rgba(2, 10, 17, 0.4); border: 1px solid rgba(156, 218, 236, 0.1);
  text-align: center; min-width: 0;
}
.${p}-stat-cell span {
  display: block; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-stat-cell strong {
  display: block; margin-top: 0.08rem; font-size: 0.82rem; font-weight: 900; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.${p}-contract-doc {
  flex-shrink: 0; padding: 0.5rem 0.55rem; border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(233, 168, 60, 0.16), rgba(9, 25, 38, 0.55) 38%),
    linear-gradient(180deg, rgba(15, 46, 66, 0.5), rgba(8, 22, 34, 0.75));
  border: 1px solid rgba(233, 168, 60, 0.35);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 216, 141, 0.12);
}
.${p}-contract-doc-head {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;
  margin-bottom: 0.35rem;
}
.${p}-contract-doc-head h3 {
  margin: 0; font-size: 0.72rem; font-weight: 900; letter-spacing: 0.12em;
  text-transform: uppercase; color: #ffd88d;
}
.${p}-contract-doc-head p {
  margin: 0.15rem 0 0; font-size: 0.6875rem; color: rgba(233, 247, 251, 0.78);
}
.${p}-contract-hero {
  display: flex; gap: 0.75rem; align-items: baseline; margin: 0.2rem 0 0.4rem;
}
.${p}-contract-hero strong {
  font-size: 1.55rem; font-weight: 900; color: var(--text); line-height: 1;
}
.${p}-contract-hero span {
  font-size: 0.78rem; font-weight: 800; color: var(--gold);
}
.${p}-contract-rows {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.22rem 0.5rem;
}
.${p}-contract-row {
  display: flex; justify-content: space-between; gap: 0.35rem;
  padding: 0.18rem 0; border-bottom: 1px solid rgba(156, 218, 236, 0.08);
  font-size: 0.6875rem;
}
.${p}-contract-row span { color: var(--muted); font-weight: 700; }
.${p}-contract-row strong { color: var(--text); font-weight: 800; text-align: right; }
.${p}-assignment-select {
  background: rgba(8, 22, 34, 0.85); border: 1px solid rgba(156, 218, 236, 0.18);
  color: var(--text); font-size: 0.6875rem; font-weight: 800; border-radius: 6px;
  padding: 0.12rem 0.3rem; max-width: 60%;
}
.${p}-assignment-select:disabled { opacity: 0.55; cursor: not-allowed; }
.${p}-assignment-select option:disabled { color: rgba(233, 247, 251, 0.4); }
.${p}-offer-scroll {
  display: flex; flex-direction: column; gap: 0.28rem; min-height: 0; flex: 1; overflow-y: auto;
  padding-right: 0.1rem; scrollbar-width: thin;
}
.${p}-offer-label {
  margin: 0; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); flex-shrink: 0;
}
.${p}-offer-btn {
  display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.25rem 0.4rem;
  text-align: left; padding: 0.4rem 0.45rem; border-radius: 10px; cursor: pointer;
  background: rgba(8, 22, 34, 0.72); border: 1px solid rgba(156, 218, 236, 0.12);
  color: var(--text); transition: border-color 0.15s ease, background 0.15s ease;
}
.${p}-offer-btn:hover:not(:disabled) { border-color: rgba(19, 216, 231, 0.35); }
.${p}-offer-btn.is-selected {
  border-color: rgba(233, 168, 60, 0.55);
  background: linear-gradient(180deg, rgba(233, 168, 60, 0.14), rgba(8, 22, 34, 0.8));
}
.${p}-offer-btn.is-recommended { box-shadow: inset 3px 0 0 var(--gold); }
.${p}-offer-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.${p}-offer-btn strong {
  display: block; font-size: 0.78rem; font-weight: 900; color: var(--text);
}
.${p}-offer-btn p {
  margin: 0.08rem 0 0; font-size: 0.6875rem; color: rgba(233, 247, 251, 0.72); line-height: 1.3;
}
.${p}-offer-tag {
  align-self: start; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.06em;
  text-transform: uppercase; color: #1b1002; background: var(--gold);
  padding: 0.12rem 0.3rem; border-radius: var(--radius-ops, 2px);
}
.${p}-reasons {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; min-height: 0; flex: 1; overflow: hidden;
}
.${p}-reason-col {
  padding: 0.35rem 0.4rem; border-radius: 10px; min-height: 0; overflow: hidden;
  background: rgba(8, 22, 34, 0.65); border: 1px solid rgba(156, 218, 236, 0.12);
  display: flex; flex-direction: column;
}
.${p}-reason-col.is-sign { border-color: rgba(82, 223, 148, 0.28); }
.${p}-reason-col.is-wait { border-color: rgba(233, 168, 60, 0.28); }
.${p}-reason-col h4 {
  margin: 0 0 0.25rem; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.08em;
  text-transform: uppercase; flex-shrink: 0;
}
.${p}-reason-col.is-sign h4 { color: #9be7b8; }
.${p}-reason-col.is-wait h4 { color: #ffd88d; }
.${p}-reason-col ul {
  margin: 0; padding: 0 0 0 0.9rem; overflow-y: auto; min-height: 0; flex: 1;
}
.${p}-reason-col li {
  font-size: 0.6875rem; color: rgba(233, 247, 251, 0.85); line-height: 1.35; margin-bottom: 0.18rem;
}
.${p}-timeline {
  flex-shrink: 0; display: flex; align-items: stretch; gap: 0; padding: 0.35rem 0.2rem;
  overflow-x: auto; border-radius: 10px;
  background: rgba(8, 22, 34, 0.55); border: 1px solid rgba(156, 218, 236, 0.12);
}
.${p}-tl-step {
  flex: 1; min-width: 4.2rem; text-align: center; position: relative; padding: 0.15rem 0.25rem;
}
.${p}-tl-step::after {
  content: ""; position: absolute; top: 0.55rem; right: -1px; width: calc(50% + 2px);
  height: 2px; background: rgba(19, 216, 231, 0.25);
}
.${p}-tl-step:last-child::after { display: none; }
.${p}-tl-dot {
  width: 0.55rem; height: 0.55rem; border-radius: 999px; margin: 0.3rem auto 0.25rem;
  background: var(--cyan); box-shadow: 0 0 0 3px rgba(19, 216, 231, 0.15);
  position: relative; z-index: 1;
}
.${p}-tl-step.is-future .${p}-tl-dot { background: var(--muted); box-shadow: none; }
.${p}-tl-step.is-deadline .${p}-tl-dot { background: var(--gold); box-shadow: 0 0 0 3px rgba(233, 168, 60, 0.2); }
.${p}-tl-step span {
  display: block; font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.${p}-tl-step strong {
  display: block; margin-top: 0.08rem; font-size: 0.6875rem; font-weight: 800; color: var(--text);
  line-height: 1.2;
}
.${p}-apply-bar {
  flex-shrink: 0; display: flex; flex-direction: column; gap: 0.25rem;
  padding: 0.4rem 0.45rem; border-radius: 10px;
  background: rgba(8, 22, 34, 0.72); border: 1px solid rgba(233, 168, 60, 0.3);
}
.${p}-apply-bar .${p}-preview {
  display: flex; flex-wrap: wrap; gap: 0.35rem 0.65rem; font-size: 0.6875rem; color: rgba(233, 247, 251, 0.78);
}
.${p}-apply-bar .${p}-preview strong { color: var(--text); }
.${p}-decision-apply {
  border: 0; border-radius: 8px; padding: 0.7rem 1rem; width: 100%;
  background: linear-gradient(180deg, #f4bd52, #d99023);
  color: #1b1002; font-weight: 1000; letter-spacing: 0.1em; text-transform: uppercase;
  cursor: pointer; font-size: 0.78rem;
}
.${p}-decision-apply:disabled { opacity: 0.5; cursor: not-allowed; }
.${p}-decision-feedback { margin: 0; font-size: 0.6875rem; font-weight: 700; }
.${p}-decision-feedback.is-error { color: #ff8f98; }
.${p}-decision-feedback.is-ok { color: #9be7b8; }
.${p}-accept-meter { margin-top: 0.4rem; padding-top: 0.35rem; border-top: 1px solid rgba(156, 218, 236, 0.12); }
.${p}-accept-meter-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.6875rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.2rem;
}
.${p}-accept-meter-head strong { font-size: 0.95rem; color: var(--text); letter-spacing: 0; text-transform: none; }
.${p}-agent-wants {
  margin: 0.3rem 0 0; padding: 0; list-style: none;
  display: flex; flex-wrap: wrap; gap: 0.2rem 0.55rem;
}
.${p}-agent-wants li { font-size: 0.6875rem; color: rgba(233, 247, 251, 0.82); font-weight: 700; }
.${p}-rail-priority {
  display: block; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--gold); margin-top: 0.05rem;
}
.${p}-rail-priority.is-calm { color: var(--cyan); }
.${p}-rail-priority.is-soft { color: var(--muted); }
.${p}-haul-card { padding: 0.35rem 0.4rem; gap: 0.4rem; }
.${p}-alert-banner {
  flex-shrink: 0; margin: 0; padding: 0.28rem 0.45rem; border-radius: 8px;
  font-size: 0.6875rem; color: #ffd88d;
  background: rgba(233, 168, 60, 0.1); border: 1px solid rgba(233, 168, 60, 0.28);
}
@media (max-width: 1200px) {
  .${p}-nego-grid { grid-template-columns: 1fr 1.1fr; }
  .${p}-nego-col.is-side { display: none; }
  .${p}-impact-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 900px) {
  .${p}-root, .${p}-reveal, .${p}-workspace, .${p}-stage, .${p}-panel { height: auto; max-height: none; }
  .${p}-reveal { overflow: auto; }
  .${p}-workspace { max-height: none; overflow: visible; }
  .${p}-nego-grid { grid-template-columns: 1fr; overflow: visible; }
  .${p}-nego-col.is-side { display: flex; }
  .${p}-impact-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
`
      : "";

  const cleanupOnly =
    p === "cleanup"
      ? `
.${p}-root {
  background:
    radial-gradient(circle at 16% 0%, rgba(82, 223, 148, 0.09), transparent 28%),
    radial-gradient(circle at 90% 12%, rgba(255, 96, 109, 0.07), transparent 24%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
}
.${p}-title--compliance {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.08em;
}
.${p}-title--compliance::after {
  content: "";
  display: block;
  width: 72px;
  height: 3px;
  margin: 0.4rem 0 0;
  background: linear-gradient(90deg, #52df94, #13d8e7);
}
.${p}-status-pill {
  border-color: rgba(82, 223, 148, 0.35);
  color: #52df94;
  background: rgba(82, 223, 148, 0.1);
}
.${p}-chip { border-color: rgba(82, 223, 148, 0.22); }
.${p}-card.is-active { border-color: rgba(82, 223, 148, 0.45); box-shadow: inset 3px 0 0 #52df94; }
`
      : "";

  const nextseasonOnly =
    p === "nextseason"
      ? `
.${p}-root {
  background:
    radial-gradient(circle at 50% 18%, rgba(233, 168, 60, 0.14), transparent 44%),
    radial-gradient(circle at 10% 82%, rgba(19, 216, 231, 0.08), transparent 32%),
    linear-gradient(180deg, #081828 0%, #020a11 100%);
}
.${p}-spotlight {
  background: radial-gradient(circle at 50% 32%, rgba(233, 168, 60, 0.14), transparent 52%);
  opacity: 1;
}
.${p}-title--countdown {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.06em;
  text-shadow: 0 0 28px rgba(233, 168, 60, 0.22);
}
.${p}-title--countdown::before {
  content: "SEASON";
  display: block;
  font-size: 0.38em;
  letter-spacing: 0.32em;
  color: var(--gold);
  margin-bottom: 0.2rem;
}
.${p}-eyebrow { color: var(--gold); }
.${p}-chip { border-color: rgba(233, 168, 60, 0.32); }
`
      : "";

  const draftFloorOnly =
    p === "draft"
      ? `
.${p}-title--floor {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.1em;
}
.${p}-title--floor::before {
  content: "PROSPECT";
  display: block;
  font-size: 0.4em;
  letter-spacing: 0.28em;
  color: var(--cyan);
  margin-bottom: 0.15rem;
}
.${p}-status-pill { border-color: rgba(233, 168, 60, 0.35); color: var(--gold); background: var(--gold-soft, rgba(233, 168, 60, 0.14)); }
`
      : "";

  const titleVariantCss = `
.${p}-title--ledger {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.07em;
  border-left: 4px solid var(--gold);
  padding-left: 0.65rem;
  text-align: left;
  text-shadow: none;
}
.${p}-title--market {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.08em;
  text-shadow: none;
}
.${p}-title--market::after {
  content: "";
  display: block;
  width: min(220px, 48vw);
  height: 2px;
  margin: 0.35rem 0 0;
  background: linear-gradient(90deg, var(--cyan), transparent);
}
.${p}-title--review {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.06em;
  color: var(--gold);
  text-shadow: none;
}
.${p}-title--rights {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.07em;
  border-bottom: 2px solid rgba(233, 168, 60, 0.45);
  padding-bottom: 0.35rem;
  text-shadow: none;
}
.${p}-title--ceremony {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.12em;
  text-align: center;
}
.${p}-title--ceremony::before {
  content: "";
  display: block;
  width: 48px;
  height: 2px;
  margin: 0 auto 0.55rem;
  background: var(--gold);
}
.${p}-title--lottery {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.14em;
  background: linear-gradient(90deg, transparent, rgba(19,216,231,0.12), transparent);
  padding: 0.35rem 0.75rem;
  text-shadow: none;
}
.${p}-title--awards {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.1em;
  color: #f4d08a;
  border-bottom: 1px solid rgba(233, 168, 60, 0.35);
  padding-bottom: 0.4rem;
  text-shadow: 0 1px 0 rgba(0,0,0,0.45);
}
.${p}-title--compliance {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.08em;
  border-left: 3px solid var(--green, #52df94);
  padding-left: 0.65rem;
  text-shadow: none;
}
.${p}-title--countdown {
  font-family: "Archivo Black", "Rajdhani", "Barlow Condensed", "Arial Narrow", sans-serif;
  letter-spacing: 0.16em;
}
.${p}-title--countdown::after {
  content: "SEASON";
  display: block;
  margin-top: 0.25rem;
  font-size: 0.42em;
  letter-spacing: 0.28em;
  color: var(--gold);
}
.${p}-root--ledger { --gold: #e9a83c; }
.${p}-root--market {
  background:
    radial-gradient(circle at 82% 0%, rgba(19, 216, 231, 0.11), transparent 30%),
    radial-gradient(circle at 12% 18%, rgba(233, 168, 60, 0.07), transparent 26%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
}
.${p}-root--floor {
  background:
    radial-gradient(circle at 50% 0%, rgba(233, 168, 60, 0.1), transparent 34%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
}
`;

  return `
.${p}-root {
  --panel: rgba(9, 25, 38, 0.94);
  --panel-2: rgba(12, 35, 52, 0.94);
  --line: rgba(156, 218, 236, 0.14);
  --cyan: #13d8e7;
  --gold: #e9a83c;
  --text: #e9f7fb;
  --muted: #8096a8;
  position: relative;
  min-height: 100dvh;
  min-height: 100vh;
  overflow: hidden;
  color: var(--text);
  background:
    radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
  isolation: isolate;
  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    sans-serif;
}
.${p}-bg, .${p}-bg-scrim, .${p}-bg-noise, .${p}-spotlight { position: absolute; inset: 0; pointer-events: none; }
.${p}-bg { z-index: 0; }
.${p}-bg-scrim { z-index: 1; background: linear-gradient(180deg, rgba(6, 19, 31, 0.35) 0%, rgba(2, 10, 17, 0.92) 100%); }
.${p}-bg-noise { z-index: 2; opacity: 0.05; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.${p}-spotlight { z-index: 3; background: radial-gradient(circle at 50% 28%, rgba(19, 216, 231, 0.1), transparent 48%); opacity: 0.85; }
.${p}-topbar { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); gap: 0.75rem; }
.${p}-topbar-left { display: flex; align-items: center; gap: 0.5rem; }
.${p}-ghost-btn { background: rgba(12, 31, 47, 0.72); border: 1px solid var(--line); color: var(--text); padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 700; transition: border-color var(--motion-micro, 110ms ease), background var(--motion-micro, 110ms ease); }
.${p}-ghost-btn:hover { border-color: rgba(73, 231, 240, 0.5); background: rgba(19, 216, 231, 0.08); }
.${p}-ghost-btn:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
/* Event phase reads as a league stamp: framed, squared, filed. */
.${p}-live-pill, .${p}-status-pill { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.7rem; border-radius: 2px; background: rgba(19, 216, 231, 0.08); border: 1px solid rgba(19, 216, 231, 0.32); font-size: 0.72rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan); }
.${p}-live-pill::before, .${p}-status-pill::before { content: ""; width: 6px; height: 6px; border-radius: 1px; background: currentColor; flex-shrink: 0; }
.${p}-phase-text { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.${p}-phase-spacer { min-width: 1rem; }
.${p}-season { font-size: 0.85rem; color: var(--muted); font-weight: 700; }
.${p}-stage { position: relative; z-index: 10; display: grid; grid-template-columns: 1fr min(380px, 34vw); gap: 1.5rem; padding: 0 1.5rem 6rem; min-height: calc(100dvh - 140px); }
@media (max-width: 900px) { .${p}-stage { grid-template-columns: 1fr; } }
@media (max-height: 800px) {
  .${p}-topbar { padding: 0.65rem 1rem; }
  .${p}-stage { padding: 0 1rem 5rem; gap: 1rem; min-height: calc(100dvh - 110px); }
  .${p}-title { font-size: clamp(1.6rem, 4vw, 2.4rem); }
  .${p}-panel { max-height: calc(100dvh - 160px); }
}
.${p}-reveal { display: flex; flex-direction: column; justify-content: center; padding: 2rem 0; }
.${p}-eyebrow { font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--cyan); margin: 0 0 0.5rem; font-weight: 900; }
/* Event titles register onto a rule instead of glowing. */
.${p}-title { position: relative; font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 900; letter-spacing: 0.06em; margin: 0 0 1rem; padding-bottom: 0.5rem; text-transform: uppercase; color: var(--text); text-shadow: none; }
.${p}-title::after { content: ""; position: absolute; left: 0; bottom: 0; width: 96px; height: 3px; background: var(--gold); }
.${p}-hero-name { font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800; margin: 0.5rem 0; color: var(--text); }
.${p}-hero-headline { font-size: 0.95rem; color: var(--muted); margin: 0.25rem 0 0.5rem; max-width: 42ch; }
.${p}-summary-line { font-size: 0.85rem; color: rgba(233, 247, 251, 0.78); margin-top: 0.75rem; }
.${p}-section { margin-bottom: 1rem; }
.${p}-section-title { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0 0 0.5rem; font-weight: 900; }
.${p}-headline { font-size: 0.8rem; color: var(--muted); margin: 0.25rem 0; line-height: 1.35; }
.${p}-chip-legend { border-color: rgba(233, 168, 60, 0.45); color: #f4d08a; }
.${p}-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
/* Metadata reads as equipment plates, not pills. */
.${p}-chip { padding: 0.25rem 0.55rem; border-radius: 2px; background: rgba(12, 35, 52, 0.8); border: 1px solid var(--line); font-size: 0.78rem; font-weight: 800; letter-spacing: 0.04em; color: rgba(233, 247, 251, 0.82); font-variant-numeric: tabular-nums; }
.${p}-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; backdrop-filter: blur(12px); padding: 1rem; max-height: calc(100dvh - 200px); display: flex; flex-direction: column; box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32); }
.${p}-panel h2 { font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 0 0 0.75rem; font-weight: 900; }
.${p}-rail { overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; flex: 1; padding-right: 0.25rem; }
.${p}-rail-hint { margin: 0.5rem 0 0; font-size: 0.72rem; color: var(--muted); }
.${p}-card { display: grid; grid-template-columns: auto 1fr; gap: 0.75rem; padding: 0.75rem; border-radius: 6px; background: linear-gradient(180deg, rgba(11, 31, 45, 0.72), rgba(7, 22, 34, 0.72)); border: 1px solid rgba(156, 218, 236, 0.1); transition: border-color var(--motion-micro, 110ms ease); }
.${p}-card.is-active { border-color: rgba(19, 216, 231, 0.45); box-shadow: inset 3px 0 0 var(--cyan); }
.${p}-card-rank { font-size: 0.75rem; font-weight: 900; color: var(--cyan); min-width: 2rem; }
.${p}-card-body strong { display: block; font-size: 0.95rem; margin-bottom: 0.25rem; color: var(--text); }
.${p}-card-details { display: flex; flex-wrap: wrap; gap: 0.35rem; font-size: 0.72rem; color: var(--muted); }
.${p}-empty { color: var(--muted); font-size: 0.9rem; padding: 1rem; text-align: left; border: 1px dashed var(--line); border-radius: 6px; }
.${p}-footer { position: fixed; bottom: 0; left: 0; right: 0; z-index: var(--z-sticky, 30); display: grid; grid-template-rows: auto auto; gap: 0.5rem; background: linear-gradient(0deg, rgba(2, 10, 17, 0.98), rgba(6, 19, 31, 0.9)); border-top: 1px solid var(--line); padding: 0.65rem 1.35rem 0.85rem; }
.${p}-footer.is-compact { grid-template-rows: auto; padding: 0.55rem 1.35rem 0.75rem; }
.${p}-ticker { margin-bottom: 0; text-align: center; overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
.${p}-ticker-track { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.45rem 1rem; min-width: min-content; }
.${p}-ticker-group { display: contents; }
.${p}-ticker-group span { font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }
.${p}-ticker-tab { border: 0; background: transparent; cursor: pointer; font-family: inherit; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); padding: 0.2rem 0.1rem; }
.${p}-ticker-tab.is-active { color: var(--text); }
.${p}-footer-actions { display: flex; justify-content: center; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
.${p}-footer-actions.is-split { justify-content: space-between; width: 100%; }
.${p}-skip-btn { background: rgba(12, 31, 47, 0.72); border: 1px solid var(--line); color: var(--muted); padding: 0.7rem 1.15rem; border-radius: 2px; cursor: pointer; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; transition: border-color 0.2s ease, color 0.2s ease; }
.${p}-skip-btn:hover { border-color: rgba(73, 231, 240, 0.5); color: var(--text); }
/* Broadcast action: the event's committing control. Rink-cut corner, flat
   deadline gold, no gloss and no lift. */
.${p}-cta-btn { position: relative; border: 0; border-radius: 0; clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%); padding: 0.8rem 2rem; min-width: 200px; background: #e9a83c; color: #1b1002; cursor: pointer; font-weight: 1000; letter-spacing: 0.14em; text-transform: uppercase; box-shadow: none; transition: background 0.2s ease, transform 0.11s ease; }
.${p}-cta-btn:hover:not(:disabled) { background: #f4c66e; }
.${p}-cta-btn:active:not(:disabled) { transform: translateY(1px); }
.${p}-cta-btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
.${p}-meter { position: relative; height: 6px; border-radius: 0; background: rgba(12, 35, 52, 0.8); overflow: hidden; margin-top: 0.5rem; background-image: repeating-linear-gradient(90deg, rgba(255,255,255,0.14) 0 1px, transparent 1px 25%); }
.${p}-meter span { display: block; height: 100%; background: #52df94; border-radius: 0; }
.${p}-warn { color: var(--gold); font-size: 0.8rem; }
.${p}-delta-up { color: #52df94; }
.${p}-delta-down { color: #ff8f98; }
.${p}-deal-overlay {
  position: fixed; inset: 0; z-index: var(--z-popup, 12000);
  display: grid; place-items: center;
  background: rgba(2, 8, 14, 0.72);
  backdrop-filter: blur(4px);
  padding: 1rem;
}
.${p}-deal-modal {
  width: min(420px, 94vw);
  border-radius: 8px;
  border: 1px solid rgba(156, 218, 236, 0.22);
  background: linear-gradient(180deg, rgba(12, 35, 52, 0.98), rgba(6, 18, 28, 0.98));
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
  padding: 1.15rem 1.2rem 1.05rem;
  text-align: center;
}
.${p}-deal-modal.is-accept { border-color: rgba(82, 223, 148, 0.45); }
.${p}-deal-modal.is-deny { border-color: rgba(255, 96, 109, 0.45); }
.${p}-deal-modal.is-pending { border-color: rgba(233, 168, 60, 0.45); }
.${p}-deal-kicker {
  margin: 0 0 0.35rem; font-size: 0.6875rem; font-weight: 900;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
}
.${p}-deal-modal h3 {
  margin: 0 0 0.35rem; font-size: 1.35rem; letter-spacing: 0.04em; text-transform: uppercase;
}
.${p}-deal-modal.is-accept h3 { color: #52df94; }
.${p}-deal-modal.is-deny h3 { color: #ff8f98; }
.${p}-deal-modal.is-pending h3 { color: #f4bd52; }
.${p}-deal-player { margin: 0 0 0.45rem; font-size: 0.95rem; font-weight: 800; color: var(--text); }
.${p}-deal-body { margin: 0 0 0.55rem; font-size: 0.84rem; color: rgba(233, 247, 251, 0.82); line-height: 1.4; }
.${p}-deal-terms { margin: 0 0 0.85rem; font-size: 0.9rem; font-weight: 900; color: var(--gold); }
.${p}-deal-modal .${p}-cta-btn { width: 100%; min-width: 0; }
${titleVariantCss}
${draftReviewExtras}
${draftReviewOnly}
${prospectRightsOnly}
${resignOnly}
${faOnly}
${cleanupOnly}
${nextseasonOnly}
${draftFloorOnly}
@media (prefers-reduced-motion: reduce) {
  .${p}-spotlight { opacity: 0.35; }
  .${p}-ghost-btn:hover, .${p}-skip-btn:hover, .${p}-cta-btn:hover:not(:disabled) { transform: none; }
}
`;
}
