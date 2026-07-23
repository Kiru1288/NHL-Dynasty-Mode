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
    p === "draftreview"
      ? `
.${p}-workspace { display: flex; flex-direction: column; gap: 0.45rem; flex: 1; min-height: 0; width: 100%; max-width: 100%; }
.${p}-haul-block { display: flex; flex-direction: column; gap: 0.25rem; }
.${p}-haul-strip { display: flex; flex-wrap: wrap; gap: 0.35rem 0.95rem; align-items: baseline; padding: 0.2rem 0 0.1rem; }
.${p}-haul-item { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.${p}-haul-item strong { color: var(--cyan); font-weight: 900; margin-right: 0.3rem; }
.${p}-haul-item.is-gold strong { color: var(--gold); }
.${p}-haul-reason { margin: 0; font-size: 0.78rem; color: rgba(233, 247, 251, 0.78); line-height: 1.35; max-width: 92ch; }
.${p}-analysis-row { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.${p}-analysis-chip { font-size: 0.65rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: rgba(233, 247, 251, 0.82); border: 1px solid rgba(156, 218, 236, 0.16); background: rgba(12, 35, 52, 0.7); padding: 0.22rem 0.5rem; border-radius: 999px; }
.${p}-grid { display: grid; grid-template-columns: 1.05fr 1fr; grid-template-rows: minmax(0, 1.15fr) minmax(0, 1fr); gap: 0.55rem; flex: 1; min-height: 0; }
.${p}-pane { background: linear-gradient(180deg, rgba(11, 31, 45, 0.72), rgba(7, 22, 34, 0.78)); border: 1px solid rgba(156, 218, 236, 0.12); border-radius: 10px; padding: 0.55rem 0.75rem; min-height: 0; overflow: hidden; display: flex; flex-direction: column; gap: 0.28rem; }
.${p}-pane-label { font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); font-weight: 900; margin: 0; }
.${p}-pick-kicker { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cyan); margin: 0; }
.${p}-pick-name { font-size: clamp(1.15rem, 2.1vw, 1.65rem); font-weight: 900; margin: 0; line-height: 1.12; color: var(--text); }
.${p}-pick-sub { font-size: 0.78rem; color: rgba(233, 247, 251, 0.78); margin: 0; }
.${p}-pick-meta { font-size: 0.72rem; color: var(--muted); margin: 0; }
.${p}-grade-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin-top: 0.1rem; }
.${p}-grade-pill { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.22rem 0.55rem; border-radius: 999px; border: 1px solid rgba(233, 168, 60, 0.4); background: rgba(233, 168, 60, 0.1); color: #f4d08a; font-size: 0.72rem; font-weight: 900; }
.${p}-verdict { font-size: 0.76rem; font-weight: 800; color: var(--cyan); }
.${p}-risk { font-size: 0.7rem; font-weight: 800; color: var(--gold); }
.${p}-review-line { font-size: 0.76rem; color: rgba(233, 247, 251, 0.82); margin: 0.1rem 0 0; line-height: 1.3; }
.${p}-dest-hero { font-size: clamp(1.05rem, 1.8vw, 1.35rem); font-weight: 900; margin: 0; color: var(--text); }
.${p}-dest-label { font-size: 0.68rem; color: var(--gold); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin: 0; }
.${p}-plan-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.28rem 0.65rem; margin-top: 0.1rem; }
.${p}-fit-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.28rem 0.65rem; }
.${p}-plan-kv { margin: 0; }
.${p}-plan-kv span { display: block; font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 800; }
.${p}-plan-kv strong { display: block; font-size: 0.76rem; color: var(--text); font-weight: 800; margin-top: 0.08rem; line-height: 1.25; }
.${p}-obj { margin-top: 0.15rem; }
.${p}-path-timeline { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 0.2rem; align-items: stretch; margin-top: 0.2rem; }
.${p}-path-step { text-align: center; padding: 0.32rem 0.3rem; border-radius: 8px; border: 1px solid rgba(156, 218, 236, 0.12); background: rgba(6, 19, 31, 0.55); min-width: 0; }
.${p}-path-step.is-next { border-color: rgba(19, 216, 231, 0.45); box-shadow: 0 0 16px rgba(19, 216, 231, 0.08); }
.${p}-path-step.is-future { opacity: 0.92; }
.${p}-path-step.is-projection { opacity: 0.78; }
.${p}-path-stage { display: block; font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 900; }
.${p}-path-detail { display: block; font-size: 0.7rem; color: var(--text); font-weight: 700; margin-top: 0.15rem; line-height: 1.2; }
.${p}-path-arrow { align-self: center; color: var(--muted); font-weight: 900; font-size: 0.8rem; }
.${p}-alt-path { margin: 0.15rem 0 0; font-size: 0.74rem; color: rgba(233, 247, 251, 0.8); line-height: 1.3; }
.${p}-alt-path span { color: var(--gold); font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.62rem; margin-right: 0.35rem; }
.${p}-stat-row { display: flex; flex-wrap: wrap; gap: 0.45rem 0.75rem; }
.${p}-stat { margin: 0; min-width: 3.2rem; }
.${p}-stat span { display: block; font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 800; }
.${p}-stat strong { display: block; font-size: 0.9rem; font-weight: 900; color: var(--text); margin-top: 0.08rem; }
.${p}-scout-head { margin: 0; font-size: 0.84rem; font-weight: 800; color: var(--text); }
.${p}-note-list { margin: 0.15rem 0 0; padding-left: 1rem; color: rgba(233, 247, 251, 0.78); font-size: 0.72rem; }
.${p}-note-list li { margin: 0.1rem 0; }
.${p}-context { font-size: 0.72rem; color: rgba(233, 247, 251, 0.75); margin: 0.15rem 0 0; line-height: 1.3; }
.${p}-rail-btn { display: block; width: 100%; text-align: left; cursor: pointer; background: transparent; border: 0; padding: 0; color: inherit; font: inherit; border-radius: 10px; }
.${p}-rail-btn:focus-visible { outline: 2px solid rgba(19, 216, 231, 0.65); outline-offset: 2px; }
.${p}-rail-btn .${p}-card { width: 100%; }
.${p}-rail-btn .${p}-card-details { flex-direction: column; align-items: flex-start; gap: 0.15rem; }
.${p}-reveal { justify-content: flex-start; padding: 0.55rem 0 0.35rem; overflow: hidden; min-height: 0; }
.${p}-title { font-size: clamp(1.4rem, 2.8vw, 2.1rem); margin: 0 0 0.25rem; }
@media (max-width: 1366px) {
  .${p}-fit-grid { grid-template-columns: 1fr; }
  .${p}-pick-meta { display: none; }
  .${p}-haul-item:nth-child(n+6) { display: none; }
}
@media (max-width: 1100px) {
  .${p}-grid { grid-template-columns: 1fr 1fr; gap: 0.45rem; }
  .${p}-path-detail { font-size: 0.66rem; }
}
`
      : "";
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
  min-height: 100vh;
  overflow: hidden;
  color: var(--text);
  background:
    radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
    linear-gradient(180deg, #06131f 0%, #020a11 100%);
  isolation: isolate;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.${p}-bg, .${p}-bg-scrim, .${p}-bg-noise, .${p}-spotlight { position: absolute; inset: 0; pointer-events: none; }
.${p}-bg { z-index: 0; }
.${p}-bg-scrim { z-index: 1; background: linear-gradient(180deg, rgba(6, 19, 31, 0.35) 0%, rgba(2, 10, 17, 0.92) 100%); }
.${p}-bg-noise { z-index: 2; opacity: 0.05; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.${p}-spotlight { z-index: 3; background: radial-gradient(circle at 50% 28%, rgba(19, 216, 231, 0.1), transparent 48%); opacity: 0.85; }
.${p}-topbar { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); gap: 0.75rem; }
.${p}-topbar-left { display: flex; align-items: center; gap: 0.5rem; }
.${p}-ghost-btn { background: rgba(12, 31, 47, 0.72); border: 1px solid var(--line); color: var(--text); padding: 0.5rem 1rem; border-radius: 999px; cursor: pointer; font-weight: 700; transition: border-color 0.2s ease, transform 0.2s ease; }
.${p}-ghost-btn:hover { border-color: rgba(73, 231, 240, 0.5); transform: translateY(-1px); }
.${p}-live-pill, .${p}-status-pill { display: inline-flex; align-items: center; padding: 0.35rem 0.85rem; border-radius: 999px; background: rgba(19, 216, 231, 0.1); border: 1px solid rgba(19, 216, 231, 0.28); font-size: 0.72rem; font-weight: 900; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cyan); }
.${p}-season { font-size: 0.85rem; color: var(--muted); font-weight: 700; }
.${p}-stage { position: relative; z-index: 10; display: grid; grid-template-columns: 1fr min(380px, 34vw); gap: 1.5rem; padding: 0 1.5rem 6rem; min-height: calc(100vh - 140px); }
@media (max-width: 900px) { .${p}-stage { grid-template-columns: 1fr; } }
.${p}-reveal { display: flex; flex-direction: column; justify-content: center; padding: 2rem 0; }
.${p}-eyebrow { font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--cyan); margin: 0 0 0.5rem; font-weight: 900; }
.${p}-title { font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 900; letter-spacing: 0.06em; margin: 0 0 1rem; text-transform: uppercase; color: var(--text); text-shadow: 0 0 24px rgba(19, 216, 231, 0.12); }
.${p}-hero-name { font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800; margin: 0.5rem 0; color: var(--text); }
.${p}-hero-headline { font-size: 0.95rem; color: var(--muted); margin: 0.25rem 0 0.5rem; max-width: 42ch; }
.${p}-summary-line { font-size: 0.85rem; color: rgba(233, 247, 251, 0.78); margin-top: 0.75rem; }
.${p}-section { margin-bottom: 1rem; }
.${p}-section-title { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0 0 0.5rem; font-weight: 900; }
.${p}-headline { font-size: 0.8rem; color: var(--muted); margin: 0.25rem 0; line-height: 1.35; }
.${p}-chip-legend { border-color: rgba(233, 168, 60, 0.45); color: #f4d08a; }
.${p}-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.${p}-chip { padding: 0.35rem 0.75rem; border-radius: 999px; background: rgba(12, 35, 52, 0.8); border: 1px solid var(--line); font-size: 0.8rem; font-weight: 700; color: rgba(233, 247, 251, 0.82); }
.${p}-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; backdrop-filter: blur(12px); padding: 1rem; max-height: calc(100vh - 200px); display: flex; flex-direction: column; box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32); }
.${p}-panel h2 { font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 0 0 0.75rem; font-weight: 900; }
.${p}-rail { overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; flex: 1; padding-right: 0.25rem; }
.${p}-card { display: grid; grid-template-columns: auto 1fr; gap: 0.75rem; padding: 0.75rem; border-radius: 10px; background: linear-gradient(180deg, rgba(11, 31, 45, 0.72), rgba(7, 22, 34, 0.72)); border: 1px solid rgba(156, 218, 236, 0.1); transition: border-color 0.2s ease; }
.${p}-card.is-active { border-color: rgba(19, 216, 231, 0.45); box-shadow: 0 0 20px rgba(19, 216, 231, 0.1); }
.${p}-card-rank { font-size: 0.75rem; font-weight: 900; color: var(--cyan); min-width: 2rem; }
.${p}-card-body strong { display: block; font-size: 0.95rem; margin-bottom: 0.25rem; color: var(--text); }
.${p}-card-details { display: flex; flex-wrap: wrap; gap: 0.35rem; font-size: 0.72rem; color: var(--muted); }
.${p}-empty { color: var(--muted); font-size: 0.9rem; padding: 1rem; text-align: center; }
.${p}-footer { position: fixed; bottom: 0; left: 0; right: 0; z-index: 20; background: linear-gradient(0deg, rgba(2, 10, 17, 0.98), rgba(6, 19, 31, 0.9)); border-top: 1px solid var(--line); padding: 0.75rem 1.5rem 1rem; }
.${p}-ticker { margin-bottom: 0.75rem; text-align: center; }
.${p}-ticker-track { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.65rem 1.25rem; }
.${p}-ticker-group { display: contents; }
.${p}-ticker-group span { font-size: 0.68rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }
.${p}-footer-actions { display: flex; justify-content: center; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
.${p}-skip-btn { background: rgba(12, 31, 47, 0.72); border: 1px solid var(--line); color: var(--muted); padding: 0.75rem 1.25rem; border-radius: 999px; cursor: pointer; font-weight: 700; transition: border-color 0.2s ease, transform 0.2s ease; }
.${p}-skip-btn:hover { border-color: rgba(73, 231, 240, 0.5); color: var(--text); transform: translateY(-1px); }
.${p}-cta-btn { border: 0; border-radius: 7px; padding: 0.85rem 2rem; min-width: 200px; background: linear-gradient(180deg, #f4bd52, #d99023), radial-gradient(circle at 20% 0%, rgba(255, 255, 255, 0.5), transparent 30%); color: #1b1002; cursor: pointer; font-weight: 1000; letter-spacing: 0.12em; text-transform: uppercase; box-shadow: 0 14px 36px rgba(217, 144, 35, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.35); transition: transform 0.2s ease, filter 0.2s ease; }
.${p}-cta-btn:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.04); }
.${p}-cta-btn:disabled { opacity: 0.55; cursor: not-allowed; box-shadow: none; }
.${p}-meter { height: 6px; border-radius: 999px; background: rgba(12, 35, 52, 0.8); overflow: hidden; margin-top: 0.5rem; }
.${p}-meter span { display: block; height: 100%; background: linear-gradient(90deg, #52df94, #2fbf78); border-radius: 999px; }
.${p}-warn { color: var(--gold); font-size: 0.8rem; }
.${p}-delta-up { color: #52df94; }
.${p}-delta-down { color: #ff8f98; }
${draftReviewExtras}
`;
}
