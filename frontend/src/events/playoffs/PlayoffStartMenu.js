import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveFranchiseTeamLogo, toLogoUrl } from "../../utils/teamLogos";
import { enterPlayoffs, playoffAction, playoffSeriesPreview } from "../../services/franchiseService";
import { isNetworkError, formatFranchiseApiError, isTimeoutError } from "../../services/api";
import { useGameUI } from "../../game/GameUIContext";
import TeamLogoBadge from "../../components/ui/TeamLogoBadge";
import PlayerHeadshot from "../../components/PlayerHeadshot";

/**
 * Stanley Cup Playoffs bracket hub (Broadcast Operations register).
 * West R1 → R2 → Final → CUP ← East Final ← R2 ← R1
 *
 * All styles are scoped under .pf-root (game.css, "Playoffs broadcast redesign").
 */

/* ───────────────────────── constants ───────────────────────── */

const ROUND_NAMES = { 1: "First Round", 2: "Second Round", 3: "Conference Final", 4: "Stanley Cup Final" };
const ROUND_SHORT = { 1: "R1", 2: "R2", 3: "CF", 4: "Final" };
const SERIES_PER_CONF = { 1: 4, 2: 2, 3: 1 };
const MAX_FF_DAYS = 150; // hard cap on sequential advance_day calls
const FF_PAINT_MS = 60; // let React paint the bracket between days
const STALL_LIMIT = 3; // consecutive responses with no day change → stop
// Sim speeds: slow/medium play one night per request with a pause so every round
// (Round 1 included) visibly plays out; fast batches many nights per request.
const SIM_SPEEDS = {
  slow: { label: "Slow", daysPerCall: 1, pauseMs: 1100, popupMs: 0 },
  medium: { label: "Medium", daysPerCall: 1, pauseMs: 380, popupMs: 0 },
  // Fast still plays one night per request (every series, one game at a time); it only shortens the pause.
  fast: { label: "Fast", daysPerCall: 1, pauseMs: 140, popupMs: 2600 },
};
const SPEED_KEY = "po_sim_speed";
const MAX_POPUPS = 5;

function readSpeed() {
  try {
    const v = window.localStorage.getItem(SPEED_KEY);
    return SIM_SPEEDS[v] ? v : "medium";
  } catch {
    return "medium";
  }
}

/** Popup copy for a backend playoff event (franchise_playoffs._detect_playoff_events). */
function describePlayoffEvent(ev, teamFor) {
  const w = teamFor(ev.team_id)?.abbrev || "TBD";
  const l = teamFor(ev.opp_id)?.abbrev || "TBD";
  const g = `Game ${ev.game}`;
  const otLabel = (n) => (n >= 2 ? `${n}OT` : "OT");
  switch (ev.kind) {
    case "user_eliminated":
      return { tone: "loss", tag: "Eliminated", title: `${l} eliminated`, body: `${w} close it out ${ev.series_score}. Season over.` };
    case "series_choke":
      return { tone: "alert", tag: "Collapse", title: `${l} blow a ${ev.blown} lead`, body: `${w} storm back to win the series ${ev.series_score}.` };
    case "upset_series":
      return { tone: "alert", tag: "Upset", title: `${w} topple ${l}`, body: `Lower seed advances ${ev.series_score}.` };
    case "sweep":
      return { tone: "info", tag: "Sweep", title: `${w} sweep ${l}`, body: `Four straight. ${g} final ${ev.score}.` };
    case "game7_result":
      return { tone: "gold", tag: "Game 7", title: `${w} win Game 7`, body: `${w} ${ev.score} over ${l}${ev.ot_periods ? " in OT" : ""}.` };
    case "game7_set":
      return { tone: "gold", tag: "Game 7", title: `${w} vs ${l} going the distance`, body: "Series tied 3-3. Winner takes all." };
    case "marathon_ot":
      return { tone: "gold", tag: otLabel(ev.ot_periods), title: `${w} win a ${otLabel(ev.ot_periods)} marathon`, body: `${g} vs ${l}, final ${ev.score}.` };
    case "ot_elimination":
      return { tone: "gold", tag: "OT Winner", title: `${w} end it in overtime`, body: `Series-clinching OT winner vs ${l}, ${ev.score}.` };
    case "upset_game":
      return { tone: "info", tag: "Stunner", title: `${w} stun ${l}`, body: `${g}, ${ev.score}. Series ${ev.series_score}.` };
    default:
      return null;
  }
}

/* ───────────────────────── pure helpers ───────────────────────── */

const cx = (...parts) => parts.filter(Boolean).join(" ");

function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function hasKeys(obj) {
  return Boolean(obj) && typeof obj === "object" && Object.keys(obj).length > 0;
}

/** Number or null — never NaN. Fixes the "DAY NAN" labels. */
function numOrNull(value) {
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function getTeamId(team) {
  if (typeof team === "string" || typeof team === "number") return String(team);
  return String(team?.team_id || team?.teamId || team?.id || "").trim();
}

function asTeamList(value) {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value.teams)) return value.teams;
  if (Array.isArray(value.standings)) return value.standings;
  if (Array.isArray(value.rows)) return value.rows;
  return Object.values(value).filter((row) => row && typeof row === "object" && getTeamId(row));
}

function buildTeamLookup(franchiseState = {}, payload = {}) {
  const map = new Map();
  const pools = [
    payload?.playoff_teams,
    payload?.teams,
    franchiseState?.playoff_payload?.playoff_teams,
    franchiseState?.playoff_payload?.teams,
    franchiseState?.league_teams,
    franchiseState?.leagueTeams,
    franchiseState?.standings,
    franchiseState?.standings_table,
  ];
  for (const pool of pools) {
    for (const row of asTeamList(pool)) {
      const id = getTeamId(row);
      if (!id) continue;
      map.set(id, { ...(map.get(id) || {}), ...row });
    }
  }
  return map;
}

function formatTeamRecord(row = {}) {
  const wins = numOrNull(firstDefined(row.w, row.wins, row.record?.w, row.record?.wins));
  const losses = numOrNull(firstDefined(row.l, row.losses, row.record?.l, row.record?.losses));
  const otl = numOrNull(firstDefined(row.otl, row.ot, row.overtime_losses, row.record?.otl)) ?? 0;
  if (wins === null || losses === null) return "";
  return `${wins}-${losses}-${otl}`;
}

function resolveTeam(id, lookup, franchiseState) {
  const tid = String(id || "");
  const row = lookup.get(tid) || {};
  const name = firstDefined(row.full_name, row.name, row.team_name, row.abbrev, tid) || tid;
  const abbrev = firstDefined(row.abbrev, row.abbreviation, String(name).slice(0, 3).toUpperCase());
  const logo =
    toLogoUrl(row.logo || row.logo_url) ||
    resolveFranchiseTeamLogo({ ...row, team_id: tid, name, abbrev }, franchiseState);
  return {
    ...row,
    team_id: tid,
    name,
    abbrev,
    logo,
    record: formatTeamRecord(row),
    pts: numOrNull(firstDefined(row.pts, row.points)),
    seed: firstDefined(row.seed, row.playoff_seed),
  };
}

function roundName(round) {
  return ROUND_NAMES[round] || `Round ${round}`;
}

function confKey(series) {
  const c = String(series?.conference || "").toLowerCase();
  if (c.includes("west")) return "West";
  if (c.includes("east")) return "East";
  return series?.conference || "League";
}

function isHighSeedHome(gameNumber) {
  return [1, 2, 5, 7].includes(Number(gameNumber));
}

function userTeamId(franchiseState = {}) {
  return String(
    franchiseState.user_team_id ||
      franchiseState.userTeamId ||
      franchiseState.team?.team_id ||
      franchiseState.team?.id ||
      ""
  );
}

function seriesWinnerId(series) {
  if (!series) return "";
  if (series.winner_id) return String(series.winner_id);
  const wh = Number(series.wins_high || 0);
  const wl = Number(series.wins_low || 0);
  if (series.status === "complete" || wh >= 4 || wl >= 4) {
    if (wh === wl) return "";
    return String(wh > wl ? series.team_high_id : series.team_low_id);
  }
  return "";
}

/** empty | pending | scheduled | live | complete */
function seriesState(series, isLive) {
  if (!series) return "empty";
  if (!series.team_high_id || !series.team_low_id) return "pending";
  if (series.status === "complete" || seriesWinnerId(series)) return "complete";
  if (isLive && series.status === "active") return "live";
  return "scheduled";
}

function nextGameNumber(series) {
  return numOrNull(series?.next_game) || (series?.game_log?.length || 0) + 1;
}

function seriesLabel(series, high, low) {
  const wh = Number(series?.wins_high || 0);
  const wl = Number(series?.wins_low || 0);
  const h = high?.abbrev || "TBD";
  const l = low?.abbrev || "TBD";
  const winner = seriesWinnerId(series);
  if (winner) {
    const highWon = winner === String(series.team_high_id);
    return `${highWon ? h : l} wins ${Math.max(wh, wl)}–${Math.min(wh, wl)}`;
  }
  if (wh === wl) return `Series tied ${wh}–${wl}`;
  return wh > wl ? `${h} leads ${wh}–${wl}` : `${l} leads ${wl}–${wh}`;
}

function emptySlot(conf, round, slot) {
  return {
    series_id: `${conf || "CUP"}-R${round}-${slot}`,
    round_index: round,
    conference: conf,
    bracket_slot: slot,
    team_high_id: "",
    team_low_id: "",
    wins_high: 0,
    wins_low: 0,
    status: "pending",
    game_log: [],
    next_game: 1,
  };
}

/** Preview bracket from playoff_ready payload (R1 filled, later rounds empty). No invented schedule data. */
function buildPreviewSeries(payload = {}) {
  const r1src = payload.first_round || payload.matchups || payload.series || [];
  const rows = [];
  const byConf = {};

  (r1src || []).forEach((m) => {
    const conf = confKey(m);
    byConf[conf] = byConf[conf] || [];
    const slot = byConf[conf].length;
    const row = {
      ...m,
      series_id: m.series_id || `${conf}-R1-${slot}`,
      round_index: 1,
      conference: m.conference || conf,
      bracket_slot: slot,
      team_high_id: String(m.team_high_id || m.home_id || ""),
      team_low_id: String(m.team_low_id || m.away_id || ""),
      seed_high: m.seed_high || m.seedHigh,
      seed_low: m.seed_low || m.seedLow,
      wins_high: 0,
      wins_low: 0,
      status: "scheduled",
      game_log: [],
      next_game: 1,
      scheduled_day: numOrNull(m.scheduled_day),
      preview: true,
    };
    byConf[conf].push(row);
    rows.push(row);
  });

  const confs = Object.keys(byConf).length ? Object.keys(byConf) : ["West", "East"];
  for (const conf of confs) {
    if (conf === "League") continue;
    const n = (byConf[conf] || []).length || 4;
    for (let slot = 0; slot < Math.max(1, Math.floor(n / 2)); slot += 1) {
      rows.push(emptySlot(conf, 2, slot));
    }
    rows.push(emptySlot(conf, 3, 0));
  }
  rows.push(emptySlot(null, 4, 0));
  return rows;
}

function normalizeSeries(s, uid) {
  const hi = s.team_high_id !== undefined && s.team_high_id !== null ? String(s.team_high_id) : "";
  const lo = s.team_low_id !== undefined && s.team_low_id !== null ? String(s.team_low_id) : "";
  return {
    ...s,
    series_id: String(s.series_id),
    round_index: Number(s.round_index) || 1,
    team_high_id: hi,
    team_low_id: lo,
    wins_high: Number(s.wins_high || 0),
    wins_low: Number(s.wins_low || 0),
    game_log: Array.isArray(s.game_log) ? s.game_log : [],
    is_user_series: Boolean(uid) && (hi === uid || lo === uid),
  };
}

const bySlot = (a, b) => Number(a.bracket_slot || 0) - Number(b.bracket_slot || 0);

function feederText(feeder, teamFor) {
  if (!feeder) return "TBD";
  const w = seriesWinnerId(feeder);
  if (w) return teamFor(w)?.abbrev || "TBD";
  const h = teamFor(feeder.team_high_id)?.abbrev;
  const l = teamFor(feeder.team_low_id)?.abbrev;
  return h && l ? `${h}/${l}` : "TBD";
}

/** The two series that feed an empty bracket slot, matched to the row they fill. */
function pendingRowFeeders(series, feeders) {
  if (!feeders) return [null, null];
  const [f0, f1] = feeders;
  const contains = (f, id) => Boolean(f && id) && (String(f.team_high_id) === id || String(f.team_low_id) === id);
  let fHigh = f0;
  let fLow = f1;
  const hi = series.team_high_id;
  const lo = series.team_low_id;
  if (hi && contains(f0, hi)) fLow = f1;
  else if (hi && contains(f1, hi)) fLow = f0;
  if (lo && contains(f0, lo)) fHigh = f1;
  else if (lo && contains(f1, lo)) fHigh = f0;
  return [fHigh, fLow];
}

function isBlockedResponse(res) {
  return Boolean(res?.result?.blocked || res?.blocked);
}

function isFinishedResponse(res) {
  const st = res?.state || {};
  return (
    Boolean(st?.playoff_live?.completed) ||
    Boolean(st?.champion_id) ||
    st?.season_phase === "post_cup" ||
    st?.phase === "post_cup" ||
    res?.result?.finish?.status === "post_cup" ||
    Boolean(res?.result?.finish?.champion_id)
  );
}

function describeError(e, fallback) {
  if (isTimeoutError(e) || isNetworkError(e)) {
    return formatFranchiseApiError(e) || "Lost connection to the sim server.";
  }
  const detail = formatFranchiseApiError(e) || e?.response?.data?.detail || e?.message || fallback;
  return typeof detail === "string" ? detail : JSON.stringify(detail);
}

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

/* ───────────────────────── small pieces ───────────────────────── */

const TWO_WORD_NICKS = ["Maple Leafs", "Red Wings", "Blue Jackets", "Golden Knights"];
/** "Vancouver Canucks" -> { city: "Vancouver", nick: "Canucks" } */
function splitTeamName(team) {
  const full = String(team?.name || team?.abbrev || "").trim();
  if (!full) return { city: "", nick: "" };
  const two = TWO_WORD_NICKS.find((n) => full.endsWith(n));
  if (two) return { city: full.slice(0, full.length - two.length).trim(), nick: two };
  const parts = full.split(" ");
  if (parts.length < 2) return { city: "", nick: full };
  return { city: parts.slice(0, -1).join(" "), nick: parts[parts.length - 1] };
}

function cityOf(team) {
  return splitTeamName(team).city || team?.abbrev || "";
}

function nickOf(team) {
  return splitTeamName(team).nick || team?.abbrev || "TBD";
}

function teamAccent(team) {
  const c = firstDefined(team?.primary_color, team?.color_primary, team?.colors?.primary, team?.color, team?.team_color);
  return typeof c === "string" && /^#|^rgb/.test(c) ? c : null;
}

/** What is at stake in the next game of a live series. */
function seriesStakes(series, high, low) {
  const wh = Number(series?.wins_high || 0);
  const wl = Number(series?.wins_low || 0);
  if (wh === 3 && wl === 3) return { kind: "g7", text: "Game 7" };
  if (wh === 3 || wl === 3) {
    const trail = wh === 3 ? low : high;
    return { kind: "elim", text: `${trail?.abbrev || "Trailing club"} facing elimination` };
  }
  return null;
}

// Named rivalries (same list the backend preview uses), keyed by sorted abbreviations.
const RIVALRY_NAMES = {
  "CGY|EDM": "Battle of Alberta",
  "OTT|TOR": "Battle of Ontario",
  "BOS|MTL": "Original Six rivalry",
  "MTL|TOR": "Original Six rivalry",
  "CHI|DET": "Original Six rivalry",
  "BOS|TOR": "Original Six rivalry",
  "NYI|NYR": "Battle of New York",
  "NJD|NYR": "Hudson River rivalry",
  "PHI|PIT": "Battle of Pennsylvania",
  "PIT|WSH": "Capitals vs Penguins",
  "ANA|LAK": "Freeway Face-Off",
  "FLA|TBL": "Battle of Florida",
  "COL|DET": "Avalanche vs Red Wings",
  "CGY|VAN": "Pacific grudge match",
  "MTL|QUE": "Battle of Quebec",
};

function rivalryName(a, b) {
  if (!a || !b) return null;
  return RIVALRY_NAMES[[String(a), String(b)].sort().join("|")] || null;
}

/** A one-line identity for a matchup, from the two clubs' regular seasons. */
function matchupIntel(high, low) {
  if (!high || !low) return null;
  const riv = rivalryName(high.abbrev, low.abbrev);
  const hp = numOrNull(high.pts);
  const lp = numOrNull(low.pts);
  const gap = hp !== null && lp !== null ? hp - lp : null;
  const g = gap === null ? null : Math.abs(gap);
  const apart = g === null ? "" : g === 0 ? "level on points" : `${g} pt${g === 1 ? "" : "s"} apart`;
  if (riv) return { kind: "rivalry", text: riv, detail: apart };
  if (g === null) return null;
  if (g >= 12) return { kind: "upset", text: "Upset watch", detail: apart };
  if (g <= 3) return { kind: "even", text: "Coin flip", detail: apart };
  return { kind: "lean", text: `${(gap > 0 ? high : low).abbrev} favoured`, detail: apart };
}

function streakText(s) {
  const m = /^([WL])(\d+)$/.exec(String(s || ""));
  if (!m) return "";
  return `${m[1] === "W" ? "won" : "lost"} ${m[2]} straight`;
}

function gamesFromRecord(rec) {
  const parts = String(rec || "").split("-").map((x) => Number(x));
  return parts.length >= 2 && parts.every((x) => Number.isFinite(x)) ? parts.reduce((a, b) => a + b, 0) : 0;
}

function sentenceCase(text) {
  const s = String(text || "").trim();
  if (!s) return "";
  if (s !== s.toUpperCase()) return s;
  // Keep team abbreviations (2-3 letters) upper case.
  return s
    .toLowerCase()
    .replace(/\b([a-z]{2,3})\b/g, (w, _x, i, all) => {
      const orig = text.trim().substr(i, w.length);
      return /^[A-Z]{2,3}$/.test(orig) && !["THE", "AND", "CAN", "OFF", "ALL", "HAS", "ITS", "FOR"].includes(orig) ? orig : w;
    })
    .replace(/^./, (c) => c.toUpperCase());
}

function HomeIce({ title = "Home ice for the next game" }) {
  return (
    <svg className="pf-home" viewBox="0 0 12 12" width="11" height="11" role="img" aria-label={title}>
      <title>{title}</title>
      <path d="M1.5 6 6 2l4.5 4v4.5h-3V8h-3v2.5h-3Z" fill="currentColor" />
    </svg>
  );
}

function TrophyGlyph({ size = 34 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="pf-trophy-glyph">
      <path d="M9 4h14v5a7 7 0 0 1-14 0V4Z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M9 6H5.5a3.5 3.5 0 0 0 3.8 4.4M23 6h3.5a3.5 3.5 0 0 1-3.8 4.4" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M16 16v5M12 21h8l1.5 3h-11L12 21Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M9 24h14v4H9z" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function CupCenterpiece({ size = 86, lit }) {
  return (
    <svg width={size} height={size * 1.25} viewBox="0 0 64 80" aria-hidden="true" className={cx("pf-cup-svg", lit && "is-lit")}>
      <defs>
        <linearGradient id="pfCupMetal" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="#7d8a94" />
          <stop offset="0.45" stopColor="#f4f7f9" />
          <stop offset="0.6" stopColor="#c9d3da" />
          <stop offset="1" stopColor="#6c7a85" />
        </linearGradient>
      </defs>
      <path d="M18 4h28l-2 10c0 7-5 12-12 12s-12-5-12-12L18 4Z" fill="url(#pfCupMetal)" />
      <rect x="28" y="26" width="8" height="8" fill="url(#pfCupMetal)" />
      <path d="M22 34h20l3 8H19l3-8Z" fill="url(#pfCupMetal)" />
      <path d="M18 42h28v9H18z" fill="url(#pfCupMetal)" opacity="0.92" />
      <path d="M15 51h34v11H15z" fill="url(#pfCupMetal)" opacity="0.85" />
      <path d="M12 62h40v14H12z" fill="url(#pfCupMetal)" opacity="0.78" />
      <path d="M12 66h40M12 70h40M15 55h34M18 46h28" stroke="rgba(20,30,40,0.35)" strokeWidth="0.8" />
    </svg>
  );
}

function TeamMark({ team, size = 22, dimmed }) {
  if (!team?.team_id) {
    return <span className="pf-mark is-empty" style={{ width: size, height: size }} />;
  }
  return (
    <span className={cx("pf-mark", dimmed && "is-dimmed")} style={{ width: size, height: size }}>
      <TeamLogoBadge teamLogo={team.logo || ""} teamName={team.name || team.abbrev || "Team"} size={size} variant="badge" />
    </span>
  );
}

/** One cell per game: which club won it (score and OT on hover). */
function SeriesTrack({ series, high, low }) {
  const log = Array.isArray(series.game_log) ? series.game_log : [];
  const hi = String(series.team_high_id || "");
  const done = Boolean(seriesWinnerId(series));
  const next = nextGameNumber(series);
  return (
    <div className="pf-track" aria-label="Game results">
      {Array.from({ length: 7 }, (_, i) => {
        const g = log.find((x) => Number(x.game) === i + 1);
        if (!g) {
          return (
            <span key={i} className={cx("pf-track-cell", done ? "is-moot" : i + 1 === next && "is-next")}>
              {done ? "" : `G${i + 1}`}
            </span>
          );
        }
        const hs = Number(g.home_score);
        const as = Number(g.away_score);
        const win = String(g.winner_id || (hs > as ? g.home_id : g.away_id));
        const t = win === hi ? high : low;
        return (
          <span
            key={i}
            className={cx("pf-track-cell", win === hi ? "is-high" : "is-low", g.ot && "is-ot")}
            title={`Game ${i + 1}: ${t?.abbrev || ""} won ${Math.max(hs, as)}-${Math.min(hs, as)}${g.ot ? " in overtime" : ""}`}
          >
            {t?.abbrev || "W"}
          </span>
        );
      })}
    </div>
  );
}

/** Series leaders (points from per-game stars) and goalie lines (per-game saves) from the game log. */
function seriesPlayerSummary(series) {
  const log = Array.isArray(series?.game_log) ? series.game_log : [];
  const pts = new Map();
  const gk = new Map();
  for (const g of log) {
    for (const st of Array.isArray(g.stars) ? g.stars : []) {
      const k = String(st.player_id || st.name);
      const r = pts.get(k) || { name: st.name, team_id: String(st.team_id || ""), pts: 0 };
      r.pts += Number(st.points) || 0;
      pts.set(k, r);
    }
    for (const gl of Array.isArray(g.goalies) ? g.goalies : []) {
      const k = String(gl.player_id || gl.name);
      const r = gk.get(k) || { name: gl.name, team_id: String(gl.team_id || ""), gp: 0, sv: 0, sa: 0, ga: 0 };
      r.gp += 1;
      r.sv += Number(gl.saves) || 0;
      r.sa += Number(gl.shots) || 0;
      r.ga += Number(gl.ga) || 0;
      gk.set(k, r);
    }
  }
  const leaders = [...pts.values()].sort((a, b) => b.pts - a.pts).slice(0, 4);
  const goalies = [...gk.values()].sort((a, b) => b.gp - a.gp).slice(0, 2);
  return { leaders, goalies };
}

/** An empty bracket row: the series that will fill it, and where that series stands. */
function FeederSlot({ feeder, teamFor }) {
  if (!feeder) {
    return (
      <span className="pf-feed is-tbd">
        <span className="pf-feed-ghost" aria-hidden="true" />
        <span className="pf-feed-copy">
          <b>To be decided</b>
        </span>
      </span>
    );
  }
  const w = seriesWinnerId(feeder);
  if (w) {
    const t = teamFor(w);
    return (
      <span className="pf-feed is-set">
        <TeamMark team={t} size={30} />
        <span className="pf-feed-copy">
          <small>Advanced</small>
          <b>{t?.name || t?.abbrev || "TBD"}</b>
        </span>
      </span>
    );
  }
  const h = teamFor(feeder.team_high_id);
  const l = teamFor(feeder.team_low_id);
  if (!h || !l) {
    return (
      <span className="pf-feed is-tbd">
        <span className="pf-feed-ghost" aria-hidden="true" />
        <span className="pf-feed-copy">
          <small>Winner of</small>
          <b>{ROUND_NAMES[feeder.round_index] || "Earlier round"}</b>
        </span>
      </span>
    );
  }
  const wh = Number(feeder.wins_high || 0);
  const wl = Number(feeder.wins_low || 0);
  const status = wh + wl ? seriesLabel(feeder, h, l) : "Not started";
  return (
    <span className="pf-feed">
      <span className="pf-feed-pair" aria-hidden="true">
        <TeamMark team={h} size={22} />
        <TeamMark team={l} size={22} />
      </span>
      <span className="pf-feed-copy">
        <small>Winner of</small>
        <b>
          {h.abbrev} – {l.abbrev}
        </b>
      </span>
      <em className="pf-feed-state">{status}</em>
    </span>
  );
}

function TeamRow({ team, seed, wins, showWins, feeder, teamFor, isHome, isWinner, isLoser, isLeader }) {
  if (!team) {
    return (
      <div className="pf-row is-empty">
        <FeederSlot feeder={feeder} teamFor={teamFor} />
      </div>
    );
  }
  const accent = teamAccent(team);
  const city = cityOf(team);
  return (
    <div
      style={accent ? { "--pf-team": accent } : undefined}
      className={cx("pf-row", isWinner && "is-winner", isLoser && "is-out", isLeader && "is-lead")}
    >
      <span className="pf-seed" title={seed ? `Playoff seed ${seed}` : undefined}>
        {seed ?? ""}
      </span>
      <TeamMark team={team} size={32} dimmed={isLoser} />
      <span className="pf-row-name">
        <small>
          {city}
          {team.record ? <span className="pf-row-rec-inline"> · {team.record}</span> : null}
          {isHome ? <HomeIce /> : null}
        </small>
        <strong title={team.name}>{nickOf(team)}</strong>
      </span>
      {showWins ? (
        <b className="pf-row-wins">{wins}</b>
      ) : (
        <span className="pf-row-rec" title="Regular-season record">
          {team.record || ""}
        </span>
      )}
    </div>
  );
}

/* ───────────────────────── bracket series ───────────────────────── */

function SeriesCard({ series, high, low, state, feeders, teamFor, selected, justSet, playoffDay, onSelect }) {
  const next = nextGameNumber(series);
  const home = isHighSeedHome(next) ? high : low;
  const day = numOrNull(series.scheduled_day);
  const tonight = state === "live" && day !== null && day === playoffDay;
  const winner = seriesWinnerId(series);
  const wh = series.wins_high;
  const wl = series.wins_low;
  const showWins = state === "live" || state === "complete";
  const [fdHigh, fdLow] = state === "pending" ? pendingRowFeeders(series, feeders) : [null, null];
  const stakes = state === "live" ? seriesStakes(series, high, low) : null;
  const swept = state === "complete" && Math.min(wh, wl) === 0;
  const intel = state === "scheduled" ? matchupIntel(high, low) : null;
  const homeId = state === "live" || state === "scheduled" ? home?.team_id : null;

  let meta;
  if (state === "pending") meta = series.round_index === 4 ? "Awaiting conference champions" : `Awaiting ${ROUND_NAMES[series.round_index - 1] || "previous round"}`;
  else if (state === "complete") meta = swept ? "Sweep" : `Decided in ${wh + wl}`;
  else if (tonight) meta = `Game ${next} · Tonight`;
  else if (state === "live") meta = day !== null ? `Game ${next} · Day ${day + 1}` : `Game ${next}`;
  else meta = day !== null ? `Game 1 · Day ${day + 1}` : "Game 1";

  let status = "";
  if (state === "complete") status = `${(winner === series.team_high_id ? high : low)?.abbrev || "Winner"} advances`;
  else if (stakes) status = stakes.text;
  else if (state === "live") status = seriesLabel(series, high, low);
  else if (state === "scheduled") status = "Best of 7";

  return (
    <button
      type="button"
      className={cx(
        "pf-series",
        `is-${state}`,
        selected && "is-selected",
        series.is_user_series && "is-user",
        tonight && "is-tonight",
        justSet && "is-just-set",
        stakes && `is-${stakes.kind}`,
        `is-round-${series.round_index}`
      )}
      disabled={state === "pending"}
      aria-pressed={selected}
      aria-label={`${roundName(series.round_index)}: ${high?.name || "TBD"} vs ${low?.name || "TBD"}`}
      onClick={() => onSelect?.(series.series_id)}
    >
      {series.is_user_series ? <span className="pf-series-mine">Your franchise</span> : null}
      <div className="pf-series-head">
        <span className="pf-series-meta">
          {tonight ? <i className="pf-live-dot" aria-hidden="true" /> : null}
          {meta}
        </span>
        {status ? <span className={cx("pf-series-status", stakes && `is-${stakes.kind}`, state === "complete" && "is-done")}>{status}</span> : null}
      </div>
      <TeamRow
        team={high}
        seed={firstDefined(series.seed_high, high?.seed)}
        wins={wh}
        showWins={showWins}
        feeder={fdHigh}
        teamFor={state === "pending" ? teamFor : null}
        isHome={Boolean(homeId) && homeId === high?.team_id}
        isWinner={Boolean(winner) && winner === series.team_high_id}
        isLoser={Boolean(winner) && winner !== series.team_high_id}
        isLeader={state === "live" && wh > wl}
      />
      <TeamRow
        team={low}
        seed={firstDefined(series.seed_low, low?.seed)}
        wins={wl}
        showWins={showWins}
        feeder={fdLow}
        teamFor={state === "pending" ? teamFor : null}
        isHome={Boolean(homeId) && homeId === low?.team_id}
        isWinner={Boolean(winner) && winner === series.team_low_id}
        isLoser={Boolean(winner) && winner !== series.team_low_id}
        isLeader={state === "live" && wl > wh}
      />
      {showWins ? <SeriesTrack series={series} high={high} low={low} /> : null}
      {intel ? (
        <div className={cx("pf-series-intel", `is-${intel.kind}`)}>
          <b>{intel.text}</b>
          {intel.detail ? <span>{intel.detail}</span> : null}
        </div>
      ) : null}
    </button>
  );
}

/* ───────────────────────── series centre ───────────────────────── */

/* Series preview: the story of the matchup, fetched once per series per result. */
const PREVIEW_CACHE = new Map();

function useSeriesPreview(highId, lowId, day) {
  const key = highId && lowId ? `${highId}|${lowId}|${day ?? ""}` : "";
  const [data, setData] = useState(() => (key ? PREVIEW_CACHE.get(key) || null : null));
  useEffect(() => {
    if (!key) {
      setData(null);
      return undefined;
    }
    const hit = PREVIEW_CACHE.get(key);
    if (hit) {
      setData(hit);
      return undefined;
    }
    let alive = true;
    setData(null);
    playoffSeriesPreview(highId, lowId)
      .then((d) => {
        if (!d) return;
        if (PREVIEW_CACHE.size > 40) PREVIEW_CACHE.clear();
        PREVIEW_CACHE.set(key, d);
        if (alive) setData(d);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [key, highId, lowId]);
  return data;
}

function Block({ title, aside, children, className }) {
  return (
    <section className={cx("pf-block", className)}>
      <header className="pf-block-head">
        <h5>{title}</h5>
        {aside ? <span>{aside}</span> : null}
      </header>
      {children}
    </section>
  );
}

function FormDots({ form, fallback }) {
  const f = Array.isArray(form) ? form : [];
  if (!f.length) return <span className="pf-form-text">{fallback || "—"}</span>;
  const word = { W: "Win", L: "Loss", O: "OT/SO loss" };
  return (
    <span className="pf-form" aria-label={`Last ${f.length}: ${f.join(" ")}`}>
      {f.map((r, i) => (
        <i key={i} className={`is-${String(r).toLowerCase()}`} title={word[r] || r}>
          {r === "O" ? "OT" : r}
        </i>
      ))}
    </span>
  );
}

function CompareRow({ label, a, b, fmt = (v) => v, lowerIsBetter = false, bar = true }) {
  const av = Number(a) || 0;
  const bv = Number(b) || 0;
  const aEdge = lowerIsBetter ? av < bv : av > bv;
  const bEdge = lowerIsBetter ? bv < av : bv > av;
  const tot = Math.abs(av) + Math.abs(bv) || 1;
  return (
    <div className="pf-cmp">
      <b className={aEdge ? "is-edge" : ""}>{fmt(av)}</b>
      <div className="pf-cmp-mid">
        <span>{label}</span>
        {bar ? (
          <div className="pf-cmp-bar">
            <i className="is-a" style={{ width: `${(Math.abs(av) / tot) * 100}%` }} />
            <i className="is-b" style={{ width: `${(Math.abs(bv) / tot) * 100}%` }} />
          </div>
        ) : null}
      </div>
      <b className={bEdge ? "is-edge" : ""}>{fmt(bv)}</b>
    </div>
  );
}

const DUEL_ROLES = {
  star: ["Top scorers", "Each club's leading scorer: the player the other side has to slow down."],
  goalie: ["In goal", "The starters with the most games this season."],
  xfactor: ["X-factors", "Secondary scorers whose production swings close games."],
};

function DuelSide({ p, team, align }) {
  const player = useMemo(
    () => (p ? { ...p, id: p.player_id || p.id, position: p.pos, overall: p.ovr || undefined } : null),
    [p]
  );
  if (!p) return <div className={cx("pf-duel-side", `is-${align}`, "is-none")}>No player data</div>;
  return (
    <div className={cx("pf-duel-side", `is-${align}`)}>
      <span className="pf-duel-face">
        <PlayerHeadshot player={player} size="sm" preferPhoto showFlag={false} />
      </span>
      <span className="pf-duel-who">
        <b>{p.name}</b>
        <small>
          {team?.abbrev}
          {p.pos ? ` · ${p.pos}` : ""}
          {p.age ? ` · ${p.age} yrs` : ""}
        </small>
      </span>
      <span className="pf-duel-line">{p.line}</span>
      {p.why ? <span className="pf-duel-why">{p.why}</span> : null}
    </div>
  );
}

function PlayerDuel({ role, a, b, high, low }) {
  const [label, blurb] = DUEL_ROLES[role] || [role, ""];
  return (
    <div className="pf-duel">
      <p className="pf-duel-role">
        <b>{label}</b>
        <span>{blurb}</span>
      </p>
      <div className="pf-duel-row">
        <DuelSide p={a} team={high} align="l" />
        <span className="pf-duel-vs">vs</span>
        <DuelSide p={b} team={low} align="r" />
      </div>
    </div>
  );
}

function PreviewLoading() {
  return (
    <div className="pf-loading" aria-busy="true">
      <span className="pf-spinner" /> Building the series report…
    </div>
  );
}

function OverviewTab({ p, high, low, part = "story" }) {
  if (!p) return <PreviewLoading />;
  const H = p.teams?.high || {};
  const L = p.teams?.low || {};
  const h2h = p.h2h;
  const met = h2h && h2h.high + h2h.low > 0;
  let h2hAside = "";
  if (met) h2hAside = h2h.high === h2h.low ? "Split" : `${(h2h.high > h2h.low ? high : low)?.abbrev} advantage`;
  return (
    <div className="pf-tab-body">
      {part === "story" ? (
        <>
      <article className="pf-story">
        {p.tags?.length ? <p className="pf-story-tags">{p.tags.join(" · ")}</p> : null}
        <h4>{sentenceCase(p.headline)}</h4>
        {p.story?.length ? <p>{p.story.join(" ")}</p> : null}
      </article>

      <Block title="Series projection" aside={p.pick ? `Projected: ${p.pick}` : null}>
        <div className="pf-proj">
          <div className="pf-proj-side">
            <TeamMark team={high} size={30} />
            <span>
              <b>{p.odds?.high ?? "—"}%</b>
              <small>{high?.abbrev}</small>
            </span>
          </div>
          <div className="pf-proj-bar" role="img" aria-label={`${high?.abbrev} ${p.odds?.high}%, ${low?.abbrev} ${p.odds?.low}%`}>
            <i style={{ width: `${p.odds?.high ?? 50}%` }} />
          </div>
          <div className="pf-proj-side is-r">
            <span>
              <b>{p.odds?.low ?? "—"}%</b>
              <small>{low?.abbrev}</small>
            </span>
            <TeamMark team={low} size={30} />
          </div>
        </div>
        <p className="pf-note">
          Chance to win the series, from each club's regular-season points share and goal differential plus a small home-ice edge
          {p.projection?.p_game ? `. ${high?.abbrev} would win a single game ${p.projection.p_game}% of the time.` : "."}
        </p>
      </Block>

      <Block title="Season series" aside={h2hAside}>
        {met ? (
          <div className="pf-h2h">
            <div className="pf-h2h-score">
              <span>
                <b>{h2h.high}</b>
                <small>{high?.abbrev} wins</small>
              </span>
              <i>–</i>
              <span>
                <b>{h2h.low}</b>
                <small>{low?.abbrev} wins</small>
              </span>
            </div>
            <ol className="pf-h2h-games">
              {(h2h.games || []).map((g, i) => (
                <li key={i} className={g.won ? "is-h" : "is-l"}>
                  <small>Meeting {i + 1}</small>
                  <b>{g.won ? high?.abbrev : low?.abbrev}</b>
                  <span>
                    {Math.max(g.gf, g.ga)}–{Math.min(g.gf, g.ga)}
                    {g.extra ? " OT" : ""}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <p className="pf-note">{p.season_series || "Didn't meet this season"}</p>
        )}
      </Block>

        </>
      ) : null}
      {part === "history" ? (
        <>
      <Block title="Cup history">
        <div className="pf-history">
          {[
            [high, H],
            [low, L],
          ].map(([t, T], i) => (
            <div key={t?.team_id || i} className={cx("pf-history-row", T.drought?.never && "is-never")}>
              <TeamMark team={t} size={26} />
              <b>{nickOf(t)}</b>
              <span>{T.drought?.label || "—"}</span>
            </div>
          ))}
        </div>
        {p.history ? <p className="pf-note">{p.history}</p> : null}
      </Block>

      <Block title="Pressure index" aside="0–100">
        {[
          [high, H],
          [low, L],
        ].map(([t, T], i) => (
          <div key={t?.team_id || i} className="pf-press">
            <div className="pf-press-top">
              <b>{t?.abbrev}</b>
              <div className="pf-press-bar">
                <i style={{ width: `${Math.max(0, Math.min(100, Number(T.pressure) || 0))}%` }} className={Number(T.pressure) >= 80 ? "is-hot" : ""} />
              </div>
              <em>{T.pressure ?? "—"}</em>
            </div>
            {T.pressure_why?.length ? <p className="pf-press-why">{T.pressure_why.join(" · ")}</p> : null}
          </div>
        ))}
        <p className="pf-note">How much is riding on this series. It rises for favourites, strong seasons, long Cup droughts and veteran cores.</p>
      </Block>

      {p.injuries?.length ? (
        <Block title="Injury watch" className="is-alert">
          <ul className="pf-injuries">
            {p.injuries.map((x) => (
              <li key={`${x.team}-${x.name}`}>
                <b>{x.name}</b>
                <span>
                  {x.team}
                  {x.ovr ? ` · ${Math.round(x.ovr)} OVR` : ""}
                </span>
              </li>
            ))}
          </ul>
        </Block>
      ) : null}
        </>
      ) : null}
    </div>
  );
}

function MatchupTab({ p, high, low, part = "matchup" }) {
  if (!p) return <PreviewLoading />;
  const H = p.teams?.high || {};
  const L = p.teams?.low || {};
  const gpH = Number(H.gp) || gamesFromRecord(H.record) || 1;
  const gpL = Number(L.gp) || gamesFromRecord(L.record) || 1;
  const duels = [
    ["star", H.star, L.star],
    ["goalie", H.goalie, L.goalie],
    ["xfactor", H.xfactor, L.xfactor],
  ].filter(([, a, b]) => a || b);
  const signed = (v) => (v > 0 ? `+${v}` : `${v}`);
  return (
    <div className="pf-tab-body">
      {part === "matchup" ? (
        <>
      <Block title="Recent form" aside="Last 10 games">
        {[
          [high, H],
          [low, L],
        ].map(([t, T], i) => (
          <div key={t?.team_id || i} className="pf-form-row">
            <span className="pf-form-team">
              <TeamMark team={t} size={22} />
              <b>{t?.abbrev}</b>
            </span>
            <FormDots form={T.form} fallback={T.last10} />
            <span className="pf-form-sum">
              {T.last10}
              {streakText(T.streak) ? <small>{streakText(T.streak)}</small> : null}
            </span>
          </div>
        ))}
      </Block>

      <Block title="Team comparison" aside="Regular season">
        <div className="pf-cmp-head">
          <span>
            <TeamMark team={high} size={22} /> {high?.abbrev}
          </span>
          <span>
            {low?.abbrev} <TeamMark team={low} size={22} />
          </span>
        </div>
        <CompareRow label="Points" a={H.pts} b={L.pts} />
        <CompareRow label="Goals for per game" a={(H.gf || 0) / gpH} b={(L.gf || 0) / gpL} fmt={(v) => v.toFixed(2)} />
        <CompareRow label="Goals against per game" a={(H.ga || 0) / gpH} b={(L.ga || 0) / gpL} fmt={(v) => v.toFixed(2)} lowerIsBetter />
        <CompareRow label="Goal differential" a={(H.gf || 0) - (H.ga || 0)} b={(L.gf || 0) - (L.ga || 0)} fmt={signed} bar={false} />
      </Block>

        </>
      ) : null}
      {part === "players" && duels.length ? (
        <Block title="Key matchups">
          {duels.map(([role, a, b]) => (
            <PlayerDuel key={role} role={role} a={a} b={b} high={high} low={low} />
          ))}
        </Block>
      ) : null}
    </div>
  );
}

function TeamSide({ team, seed, align, mine }) {
  return (
    <div className={cx("pf-board-team", `is-${align}`)}>
      <TeamMark team={team} size={60} />
      <span className="pf-board-name">
        <small>{cityOf(team)}</small>
        <b>{nickOf(team)}</b>
      </span>
      <span className="pf-board-rec">
        {team?.record || "—"}
        {team?.pts !== null && team?.pts !== undefined ? ` · ${team.pts} pts` : ""}
        {seed ? ` · Seed ${seed}` : ""}
      </span>
      {mine ? <span className="pf-board-mine">Your franchise</span> : null}
    </div>
  );
}

function OverviewDesk({ o, onCollapse }) {
  return (
    <aside className="pf-desk is-overview">
      <header className="pf-desk-top">
        <div>
          <p className="pf-lead-in">Series centre</p>
          <div className="pf-desk-head">{o.title || "The road to the Cup"}</div>
        </div>
        <button type="button" className="pf-icon-act" aria-label="Collapse series centre" title="Collapse" onClick={onCollapse}>
          ⟩
        </button>
      </header>
      <dl className="pf-strip">
        <div>
          <dt>Teams alive</dt>
          <dd>{o.alive ?? 16}</dd>
        </div>
        <div>
          <dt>Series in play</dt>
          <dd>{o.active ?? 0}</dd>
        </div>
        <div>
          <dt>Eliminated</dt>
          <dd>{o.out ?? 0}</dd>
        </div>
      </dl>
      {[
        ["Tonight", o.tonight, ""],
        ["Facing elimination", o.watch, "is-hot"],
        ["Decided", o.done, "is-done"],
      ].map(([label, rows, tone]) =>
        rows?.length ? (
          <Block key={label} title={label}>
            {rows.map((t) => (
              <button key={t.id} type="button" className={cx("pf-ov-line", tone)} onClick={() => o.onPick?.(t.id)}>
                <span>{t.label}</span>
                <em>{t.note}</em>
              </button>
            ))}
          </Block>
        ) : null
      )}
      <p className="pf-note">Select any series in the bracket for its scouting report, game log and sim controls.</p>
    </aside>
  );
}

function SeriesDesk({ series, high, low, state, canAct, busy, playoffDay, uid, onClear, onAction, overview, onCollapse }) {
  const [tab, setTab] = useState("auto");
  const sidKey = series?.series_id || "";
  useEffect(() => setTab("auto"), [sidKey]);
  const wh = Number(series?.wins_high || 0);
  const wl = Number(series?.wins_low || 0);
  const p = useSeriesPreview(
    state !== "pending" ? series?.team_high_id : "",
    state !== "pending" ? series?.team_low_id : "",
    `${wh}-${wl}`
  );
  if (!series) return <OverviewDesk o={overview || {}} onCollapse={onCollapse} />;

  const games = Array.isArray(series.game_log) ? series.game_log : [];
  const next = nextGameNumber(series);
  const day = numOrNull(series.scheduled_day);
  const tonight = state === "live" && day !== null && day === playoffDay;
  const home = isHighSeedHome(next) ? high : low;
  const winner = seriesWinnerId(series);
  const winTeam = winner ? (winner === series.team_high_id ? high : low) : null;
  const abbrFor = (id) => {
    const s = String(id || "");
    if (s && s === series.team_high_id) return high?.abbrev || "H";
    if (s && s === series.team_low_id) return low?.abbrev || "A";
    return s.slice(0, 3).toUpperCase() || "—";
  };
  const scheduleDates = Array.isArray(series.schedule_dates) ? series.schedule_dates : [];
  const summary = seriesPlayerSummary(series);
  const conf = confKey(series);

  let boardState;
  if (state === "pending") boardState = "Matchup not set";
  else if (winTeam) boardState = `${winTeam.abbrev} wins the series`;
  else if (wh === wl) boardState = wh ? "Series tied" : "Series tied · not started";
  else boardState = `${(wh > wl ? high : low)?.abbrev} leads the series`;

  let nextLine;
  if (state === "pending") nextLine = "Waiting on the previous round.";
  else if (winTeam) nextLine = series.round_index >= 4 ? `${winTeam.name} win the Stanley Cup.` : `${winTeam.name} advance to the ${roundName(series.round_index + 1)}.`;
  else if (state === "scheduled") nextLine = `Game 1 in ${cityOf(home)}${day !== null ? ` · Day ${day + 1}` : ""}. Puck drops when the round opens.`;
  else if (tonight) nextLine = `Tonight: Game ${next} in ${cityOf(home)}.`;
  else nextLine = `Next: Game ${next} in ${cityOf(home)}${day !== null ? ` · Day ${day + 1}` : ""}.`;

  const hasLeaders = summary.leaders.length || summary.goalies.length;
  const tabs = [
    state !== "pending" ? ["overview", "Story"] : null,
    state !== "pending" ? ["history", "History"] : null,
    state !== "pending" ? ["matchup", "Matchup"] : null,
    state !== "pending" ? ["players", "Players"] : null,
    ["games", "Game log"],
    hasLeaders ? ["leaders", "Leaders"] : null,
  ].filter(Boolean);
  const cur = tab !== "auto" && tabs.some(([k]) => k === tab) ? tab : games.length ? "games" : tabs[0][0];

  return (
    <aside className={cx("pf-desk", series.is_user_series && "is-user")}>
      <header className="pf-desk-top">
        <div>
          <p className="pf-lead-in">Series centre</p>
          <div className="pf-desk-head">
            {roundName(series.round_index)}
            {series.round_index === 4 || conf === "League" ? "" : <small> · {conf}ern Conference</small>}
          </div>
        </div>
        <div className="pf-desk-acts">
          <button type="button" className="pf-text-act" onClick={onClear} title="Show every series">
            All series
          </button>
          <button type="button" className="pf-icon-act" aria-label="Collapse series centre" title="Collapse" onClick={onCollapse}>
            ⟩
          </button>
        </div>
      </header>

      <div className="pf-board">
        <TeamSide team={high} seed={series.seed_high} align="l" mine={series.is_user_series && Boolean(uid) && high?.team_id === uid} />
        <div className="pf-board-mid">
          <span className="pf-board-lab">Series</span>
          <div className="pf-board-score">
            <b className={wh > wl ? "is-lead" : ""}>{state === "pending" ? "–" : wh}</b>
            <i>–</i>
            <b className={wl > wh ? "is-lead" : ""}>{state === "pending" ? "–" : wl}</b>
          </div>
          <span className={cx("pf-board-state", winTeam && "is-done")}>{boardState}</span>
        </div>
        <TeamSide team={low} seed={series.seed_low} align="r" mine={series.is_user_series && Boolean(uid) && low?.team_id === uid} />
      </div>

      <p className={cx("pf-desk-next", tonight && "is-tonight")}>{nextLine}</p>

      {state !== "pending" ? (
        <dl className="pf-strip">
          <div>
            <dt>Home ice</dt>
            <dd>{cityOf(high) || high?.abbrev}</dd>
          </div>
          <div>
            <dt>Format</dt>
            <dd>Best of 7</dd>
          </div>
          <div>
            <dt>Games played</dt>
            <dd>{wh + wl}</dd>
          </div>
        </dl>
      ) : null}

      {canAct && state === "live" ? (
        <div className="pf-desk-run">
          {series.is_user_series ? (
            <button
              type="button"
              className="pf-act is-primary"
              disabled={busy}
              onClick={() => onAction("play_user_game", { series_id: series.series_id }, "Playing your game…")}
            >
              Play Game {next}
            </button>
          ) : null}
          <button
            type="button"
            className={cx("pf-act", series.is_user_series ? "is-quiet" : "is-primary")}
            disabled={busy}
            onClick={() => onAction("sim_series", { series_id: series.series_id }, "Simming series…")}
          >
            Sim this series
          </button>
        </div>
      ) : null}

      <div className="pf-tabs" role="tablist" aria-label="Series details">
        {tabs.map(([k, l]) => (
          <button key={k} type="button" role="tab" aria-selected={cur === k} className={cx("pf-tab", cur === k && "is-on")} onClick={() => setTab(k)}>
            {l}
            {k === "games" && games.length ? <em>{games.length}</em> : null}
          </button>
        ))}
      </div>

      {cur === "overview" ? <OverviewTab p={p} high={high} low={low} part="story" /> : null}
      {cur === "history" ? <OverviewTab p={p} high={high} low={low} part="history" /> : null}
      {cur === "matchup" ? <MatchupTab p={p} high={high} low={low} part="matchup" /> : null}
      {cur === "players" ? <MatchupTab p={p} high={high} low={low} part="players" /> : null}
      {cur === "games" ? (
        <div className="pf-tab-body">
          <ol className="pf-games">
            {Array.from({ length: 7 }, (_, i) => {
              const n = i + 1;
              const g = games.find((x) => Number(x.game) === n);
              const hIsHome = isHighSeedHome(n);
              const schedHome = hIsHome ? high : low;
              const schedAway = hIsHome ? low : high;
              const tipDay = numOrNull(scheduleDates[i]);
              const isTonightGame = state === "live" && n === next && tonight;
              const moot = !g && (state === "complete" || Boolean(winTeam));
              if (g) {
                const hs = Number(g.home_score);
                const as = Number(g.away_score);
                return (
                  <li key={n} className={cx("pf-game", "is-played", g.ot && "is-ot")}>
                    <span className="pf-game-n">Game {n}</span>
                    <span className="pf-game-line">
                      <b className={as > hs ? "w" : ""}>
                        {abbrFor(g.away_id)} {as}
                      </b>
                      <i>at</i>
                      <b className={hs > as ? "w" : ""}>
                        {abbrFor(g.home_id)} {hs}
                      </b>
                    </span>
                    <span className="pf-game-tag">{g.ot ? "Final · OT" : "Final"}</span>
                  </li>
                );
              }
              return (
                <li key={n} className={cx("pf-game", isTonightGame && "is-tonight", moot && "is-moot", n > 4 && !moot && "is-ifneeded")}>
                  <span className="pf-game-n">Game {n}</span>
                  <span className="pf-game-line">
                    {state === "pending" ? (
                      "—"
                    ) : (
                      <>
                        {schedAway?.abbrev || "TBD"} <i>at</i> {schedHome?.abbrev || "TBD"}
                      </>
                    )}
                  </span>
                  <span className="pf-game-tag">
                    {moot ? "Not needed" : isTonightGame ? "Tonight" : tipDay !== null ? `Day ${tipDay + 1}${n > 4 ? " · if needed" : ""}` : n > 4 ? "If needed" : ""}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>
      ) : null}
      {cur === "leaders" && hasLeaders ? (
        <div className="pf-tab-body">
          {summary.leaders.length ? (
            <Block title="Series scoring" aside="Points">
              {summary.leaders.map((r) => (
                <div key={r.name + r.team_id} className="pf-leader">
                  <span className="pf-leader-team">{abbrFor(r.team_id)}</span>
                  <span className="pf-leader-name">{r.name}</span>
                  <b>{r.pts}</b>
                </div>
              ))}
            </Block>
          ) : null}
          {summary.goalies.length ? (
            <Block title="Goaltending" aside="SV% · GAA">
              {summary.goalies.map((r) => (
                <div key={r.name + r.team_id} className="pf-leader">
                  <span className="pf-leader-team">{abbrFor(r.team_id)}</span>
                  <span className="pf-leader-name">{r.name}</span>
                  <b>
                    {r.sa ? (r.sv / r.sa).toFixed(3).replace(/^0/, "") : "—"}
                    <small> {(r.ga / Math.max(1, r.gp)).toFixed(2)}</small>
                  </b>
                </div>
              ))}
            </Block>
          ) : null}
        </div>
      ) : null}
    </aside>
  );
}

/* ───────────────────────── main component ───────────────────────── */

export default function PlayoffStartMenu({
  franchiseState = {},
  playoffData = {},
  onEnterPlayoffs,
  onContinue,
  onBack,
}) {
  const { mergeFranchiseState, setFranchiseState } = useGameUI() || {};
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [notice, setNotice] = useState(null); // { tone: "info" | "error", text, seriesId? }
  const [fastForwarding, setFastForwarding] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [justSetIds, setJustSetIds] = useState(() => new Set());
  const [speed, setSpeedState] = useState(readSpeed);
  const [popups, setPopups] = useState([]);
  const [confirm, setConfirm] = useState(null); // { key, text, run }
  const [deskCollapsed, setDeskCollapsed] = useState(false);
  const speedRef = useRef(speed);
  const popupTimersRef = useRef(new Map());

  const busyRef = useRef(false);
  const abortRef = useRef(false);
  const mountedRef = useRef(true);
  const prevTeamsRef = useRef(new Map());
  const selectedRef = useRef(null);
  const followedUserRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current = true;
    };
  }, []);

  /* ── derived state ── */

  const phase = String(franchiseState?.season_phase || franchiseState?.phase || "").toLowerCase();
  const live = franchiseState?.playoff_live || playoffData?.live_state || null;
  const liveSeriesRaw = Array.isArray(live?.series) && live.series.length ? live.series : null;
  const isLive = Boolean(live?.started) && (phase === "playoffs" || phase === "playoff_ready");
  const uid = String(live?.user_team_id || userTeamId(franchiseState) || "");

  // playoffData defaults to {} — only prefer it when it actually carries data.
  const payload = useMemo(
    () => (hasKeys(playoffData) ? playoffData : franchiseState?.playoff_payload || {}),
    [playoffData, franchiseState?.playoff_payload]
  );

  const lookup = useMemo(() => buildTeamLookup(franchiseState, payload), [franchiseState, payload]);

  const teamFor = useMemo(() => {
    const cache = new Map();
    return (id) => {
      const tid = id === undefined || id === null ? "" : String(id);
      if (!tid) return null;
      if (!cache.has(tid)) cache.set(tid, resolveTeam(tid, lookup, franchiseState));
      return cache.get(tid);
    };
  }, [lookup, franchiseState]);

  // Live series win whenever they exist — the bracket must not reset to a preview after the Cup.
  const seriesList = useMemo(() => {
    const src = liveSeriesRaw || buildPreviewSeries(payload);
    return src.map((s) => normalizeSeries(s, uid));
  }, [liveSeriesRaw, payload, uid]);

  const bracket = useMemo(() => {
    const pick = (conf, round) =>
      seriesList.filter((s) => s.round_index === round && confKey(s) === conf).sort(bySlot);
    const structured = pick("West", 1).length === 4 && pick("East", 1).length === 4;
    if (!structured) {
      return {
        structured: false,
        rounds: [1, 2, 3, 4].map((r) => seriesList.filter((s) => s.round_index === r).sort(bySlot)),
      };
    }
    const fill = (conf, round) => {
      const rows = pick(conf, round);
      return Array.from({ length: SERIES_PER_CONF[round] }, (_, i) => rows[i] || emptySlot(conf, round, i));
    };
    return {
      structured: true,
      West: { 1: fill("West", 1), 2: fill("West", 2), 3: fill("West", 3) },
      East: { 1: fill("East", 1), 2: fill("East", 2), 3: fill("East", 3) },
      cup: seriesList.find((s) => s.round_index === 4) || emptySlot(null, 4, 0),
    };
  }, [seriesList]);

  const getFeeders = useCallback(
    (round, conf, slot) => {
      if (!bracket.structured) return null;
      if (round === 4) return [bracket.West[3][0], bracket.East[3][0]];
      if (round < 2) return null;
      const prev = bracket[conf]?.[round - 1];
      return prev ? [prev[slot * 2], prev[slot * 2 + 1]] : null;
    },
    [bracket]
  );

  const playoffDay = numOrNull(live?.playoff_day) ?? 0;
  const championId = String(live?.champion_id || playoffData?.champion_id || franchiseState?.champion_id || "");
  const cupComplete =
    Boolean(championId) ||
    Boolean(live?.completed) ||
    phase === "post_cup" ||
    phase === "offseason" ||
    Boolean(franchiseState?.playoffs_done || franchiseState?.flags?.playoffs_done);
  const canAct = isLive && !cupComplete;

  const r1Filled = seriesList.filter((s) => s.round_index === 1 && s.team_high_id && s.team_low_id).length;

  const userCurrent = useMemo(
    () =>
      seriesList
        .filter((s) => s.is_user_series && s.team_high_id && s.team_low_id)
        .sort((a, b) => b.round_index - a.round_index)[0] || null,
    [seriesList]
  );
  const userWinner = seriesWinnerId(userCurrent);
  const userEliminated = Boolean(userCurrent) && Boolean(userWinner) && userWinner !== uid;
  const userActive = userCurrent && seriesState(userCurrent, isLive) === "live" ? userCurrent : null;
  const userAlive = Boolean(userCurrent) && !userEliminated && !cupComplete;
  const userGameTonight = Boolean(userActive) && numOrNull(userActive.scheduled_day) === playoffDay;

  const currentRound = useMemo(() => {
    const active = seriesList.filter((s) => seriesState(s, isLive) === "live").map((s) => s.round_index);
    if (active.length) return Math.min(...active);
    if (cupComplete) return 5;
    return 1;
  }, [seriesList, isLive, cupComplete]);

  /* ── selection ── */

  const selected = useMemo(
    () => (selectedId ? seriesList.find((s) => s.series_id === selectedId) || null : null),
    [selectedId, seriesList]
  );

  useEffect(() => {
    if (selected) selectedRef.current = selected;
  }, [selected]);

  // Series ids change preview → live; re-map by teams instead of silently dropping the selection.
  useEffect(() => {
    if (!selectedId || selected) return;
    const prev = selectedRef.current;
    const match =
      prev &&
      prev.team_high_id &&
      seriesList.find(
        (s) =>
          s.round_index === prev.round_index &&
          s.team_high_id === prev.team_high_id &&
          s.team_low_id === prev.team_low_id
      );
    setSelectedId(match ? match.series_id : null);
  }, [selectedId, selected, seriesList]);

  // Follow the user's club into each new round (unless the user picked a different series).
  const userCurrentId = userCurrent?.series_id || null;
  useEffect(() => {
    if (!userCurrentId || followedUserRef.current === userCurrentId) return;
    const prevFollowed = followedUserRef.current;
    followedUserRef.current = userCurrentId;
    setSelectedId((cur) => (!cur || cur === prevFollowed ? userCurrentId : cur));
  }, [userCurrentId]);

  // Pulse series whose matchup just got set.
  useEffect(() => {
    const prev = prevTeamsRef.current;
    const next = new Map();
    const gained = [];
    for (const s of seriesList) {
      const key = `${s.team_high_id}|${s.team_low_id}`;
      next.set(s.series_id, key);
      const old = prev.get(s.series_id);
      if (old !== undefined && old !== key && s.team_high_id && s.team_low_id) gained.push(s.series_id);
    }
    prevTeamsRef.current = next;
    if (!gained.length) return undefined;
    setJustSetIds(new Set(gained));
    const t = window.setTimeout(() => setJustSetIds(new Set()), 900);
    return () => window.clearTimeout(t);
  }, [seriesList]);

  useEffect(() => {
    if (isLive && live?.intro_seen === false) {
      playoffAction("mark_intro_seen").catch(() => {});
    }
  }, [isLive, live?.intro_seen]);

  /* ── sim speed + event popups ── */

  const setSpeed = useCallback((v) => {
    speedRef.current = v;
    setSpeedState(v);
    try {
      window.localStorage.setItem(SPEED_KEY, v);
    } catch {
      /* ignore */
    }
  }, []);

  const dismissPopup = useCallback((id) => {
    const t = popupTimersRef.current.get(id);
    if (t) window.clearTimeout(t);
    popupTimersRef.current.delete(id);
    setPopups((cur) => cur.filter((p) => p.id !== id));
  }, []);

  useEffect(() => {
    const timers = popupTimersRef.current;
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, []);

  /** Queue popups from res.result.events; never blocks the sim loop. */
  const enqueueEvents = useCallback(
    (res) => {
      const events = res?.result?.events;
      if (!Array.isArray(events) || !events.length || !mountedRef.current) return;
      const ttl = SIM_SPEEDS[speedRef.current]?.popupMs || 0;
      const fresh = [];
      for (const ev of events) {
        const copy = describePlayoffEvent(ev, teamFor);
        if (!copy) continue;
        const id = `${ev.id || ev.kind}-${Math.random().toString(36).slice(2, 7)}`;
        fresh.push({ id, user: Boolean(ev.involves_user), ...copy });
        // User elimination always waits for a click; everything else auto-dismisses in fast mode.
        if (ttl && ev.kind !== "user_eliminated") {
          popupTimersRef.current.set(id, window.setTimeout(() => dismissPopup(id), ttl));
        }
      }
      if (fresh.length) setPopups((cur) => [...cur, ...fresh].slice(-MAX_POPUPS));
    },
    [teamFor, dismissPopup]
  );

  /* ── actions ── */

  const applyState = useCallback(
    (res) => {
      const state = res?.state;
      if (!state) return;
      if (typeof mergeFranchiseState === "function") mergeFranchiseState(state);
      else if (typeof setFranchiseState === "function") setFranchiseState(state);
    },
    [mergeFranchiseState, setFranchiseState]
  );

  /** Single busy owner: guards double clicks, never lets a nested call clear busy early. */
  const withBusy = useCallback(async (label, fn) => {
    if (busyRef.current) return null;
    busyRef.current = true;
    setBusy(true);
    setBusyLabel(label);
    setNotice(null);
    try {
      return await fn();
    } catch (e) {
      if (mountedRef.current) setNotice({ tone: "error", text: describeError(e, "Playoff action failed") });
      return null;
    } finally {
      busyRef.current = false;
      if (mountedRef.current) {
        setBusy(false);
        setBusyLabel("");
      }
    }
  }, []);

  const enterRaw = useCallback(async () => {
    const res = typeof onEnterPlayoffs === "function" ? await onEnterPlayoffs() : await enterPlayoffs();
    applyState(res);
    return res;
  }, [onEnterPlayoffs, applyState]);

  const surfaceBlocked = useCallback((res) => {
    const reason = res?.result?.reason || res?.reason || "Your series plays tonight — play or sim your game first.";
    const sid = res?.result?.series?.series_id || res?.series?.series_id || null;
    setNotice({ tone: "info", text: reason, seriesId: sid ? String(sid) : null });
    if (sid) setSelectedId(String(sid));
  }, []);

  const handleEnter = useCallback(() => withBusy("Opening Round 1…", enterRaw), [withBusy, enterRaw]);

  const runAction = useCallback(
    (action, body = {}, label = "Working…") =>
      withBusy(label, async () => {
        const res = await playoffAction(action, body);
        applyState(res);
        enqueueEvents(res);
        if (isBlockedResponse(res)) {
          surfaceBlocked(res);
          return res;
        }
        const sid = res?.result?.series?.series_id;
        if (sid) setSelectedId(String(sid));
        return res;
      }),
    [withBusy, applyState, surfaceBlocked, enqueueEvents]
  );

  const simSeriesTarget = useMemo(() => {
    const activeRows = seriesList.filter((s) => seriesState(s, isLive) === "live");
    if (selected && seriesState(selected, isLive) === "live") return selected;
    return (
      userActive ||
      activeRows.find((s) => numOrNull(s.scheduled_day) === playoffDay) ||
      activeRows[0] ||
      null
    );
  }, [seriesList, selected, userActive, isLive, playoffDay]);

  const handleSimSeries = useCallback(() => {
    if (!simSeriesTarget) {
      setNotice({ tone: "info", text: "No active series to sim." });
      return;
    }
    setSelectedId(simSeriesTarget.series_id);
    runAction("sim_series", { series_id: simSeriesTarget.series_id }, "Simming series…");
  }, [simSeriesTarget, runAction]);

  /**
   * Day-by-day sim at the chosen speed. Every round (Round 1 included) plays out
   * the same way. stopAt: "user_game" (next game of the user's club), "round_end", or "cup".
   * Fast mode batches many nights per request (backend advance_days).
   */
  const runSim = useCallback(
    (stopAt) =>
      withBusy("Simming playoffs…", async () => {
        abortRef.current = false;
        setFastForwarding(true);
        try {
          if (!isLive && !cupComplete) await enterRaw();
          let lastDay = null;
          let stalls = 0;
          for (let i = 0; i < MAX_FF_DAYS; i += 1) {
            if (abortRef.current) {
              if (mountedRef.current) setNotice({ tone: "info", text: "Sim stopped." });
              return null;
            }
            const cfg = SIM_SPEEDS[speedRef.current] || SIM_SPEEDS.medium;
            const res = await playoffAction("advance_days", {
              count: cfg.daysPerCall,
              stop_at: stopAt === "cup" ? "" : stopAt,
            });
            applyState(res);
            enqueueEvents(res);
            if (isBlockedResponse(res)) {
              surfaceBlocked(res);
              return res;
            }
            if (isFinishedResponse(res)) return res;
            const reason = res?.result?.stop_reason;
            if (reason === "user_game" || reason === "round_end" || reason === "user_eliminated") {
              if (reason === "user_game") surfaceBlocked(res);
              return res;
            }
            const day = numOrNull(res?.state?.playoff_live?.playoff_day);
            if (day !== null && mountedRef.current) setBusyLabel(`Simming playoffs · Day ${day + 1}`);
            if (day !== null && day === lastDay) {
              stalls += 1;
              if (stalls >= STALL_LIMIT) {
                setNotice({ tone: "error", text: `Playoff day stopped advancing at Day ${day + 1}. Sim halted.` });
                return res;
              }
            } else {
              stalls = 0;
            }
            lastDay = day;
            await sleep(cfg.pauseMs || FF_PAINT_MS);
          }
          setNotice({ tone: "error", text: `Stopped after ${MAX_FF_DAYS} sim steps.` });
          return null;
        } finally {
          if (mountedRef.current) setFastForwarding(false);
        }
      }),
    [withBusy, isLive, cupComplete, enterRaw, applyState, surfaceBlocked, enqueueEvents]
  );

  const handleFastForward = useCallback(
    () => runSim(userAlive ? "user_game" : "cup"),
    [runSim, userAlive]
  );
  const handleSimRound = useCallback(() => runSim("round_end"), [runSim]);

  const handleContinue = useCallback(
    () =>
      withBusy(phase === "post_cup" ? "Opening awards…" : "Wrapping postseason…", async () => {
        if (phase === "playoffs" || phase === "playoff_ready") {
          const res = await playoffAction("finish");
          applyState(res);
          const reached =
            res?.state?.season_phase === "post_cup" ||
            res?.state?.phase === "post_cup" ||
            res?.result?.finish?.status === "post_cup";
          if (!reached) return res;
        }
        if (typeof onContinue === "function") await onContinue();
        return null;
      }),
    [withBusy, phase, applyState, onContinue]
  );

  /* ── header content ── */

  const champion = championId ? teamFor(championId) : null;
  const seasonRaw = firstDefined(franchiseState?.season_label, franchiseState?.season_year, franchiseState?.season);
  const seasonLabel = typeof seasonRaw === "string" || typeof seasonRaw === "number" ? String(seasonRaw) : "";
  const me = uid ? teamFor(uid) : null;

  const overview = (() => {
    const lab = (x) => `${teamFor(x.team_high_id)?.abbrev || "TBD"} – ${teamFor(x.team_low_id)?.abbrev || "TBD"}`;
    const rows = seriesList.filter((x) => x.team_high_id && x.team_low_id);
    const liveRows = rows.filter((x) => seriesState(x, isLive) === "live");
    const doneRows = rows.filter((x) => seriesState(x, isLive) === "complete");
    const out = new Set(doneRows.map((x) => (seriesWinnerId(x) === String(x.team_high_id) ? String(x.team_low_id) : String(x.team_high_id))));
    const r1Teams = new Set();
    seriesList.forEach((x) => {
      if (Number(x.round_index) === 1) {
        if (x.team_high_id) r1Teams.add(String(x.team_high_id));
        if (x.team_low_id) r1Teams.add(String(x.team_low_id));
      }
    });
    const total = r1Teams.size || 16;
    const tonight = liveRows
      .filter((x) => numOrNull(x.scheduled_day) === playoffDay)
      .map((x) => ({ id: x.series_id, label: lab(x), note: `Game ${nextGameNumber(x)} · ${seriesLabel(x, teamFor(x.team_high_id), teamFor(x.team_low_id))}` }));
    const watch = liveRows
      .filter((x) => Number(x.wins_high) === 3 || Number(x.wins_low) === 3)
      .map((x) => ({
        id: x.series_id,
        label: lab(x),
        note: Number(x.wins_high) === 3 && Number(x.wins_low) === 3 ? "Game 7" : seriesLabel(x, teamFor(x.team_high_id), teamFor(x.team_low_id)),
      }));
    const done = doneRows
      .filter((x) => Number(x.round_index) === currentRound || cupComplete)
      .slice(-6)
      .map((x) => ({ id: x.series_id, label: lab(x), note: seriesLabel(x, teamFor(x.team_high_id), teamFor(x.team_low_id)) }));
    return {
      title: cupComplete ? "Cup decided" : isLive ? roundName(currentRound) : "The road to the Cup",
      alive: cupComplete ? 1 : Math.max(1, total - out.size),
      active: liveRows.length,
      out: out.size,
      tonight,
      watch,
      done,
      onPick: setSelectedId,
    };
  })();

  const counts = useMemo(() => {
    const per = { 1: { done: 0, total: 8 }, 2: { done: 0, total: 4 }, 3: { done: 0, total: 2 }, 4: { done: 0, total: 1 } };
    const conf = { West: {}, East: {} };
    const out = new Set();
    const r1Teams = { West: new Set(), East: new Set() };
    let games = 0;
    for (const s of seriesList) {
      const r = Number(s.round_index) || 1;
      const st = seriesState(s, isLive);
      const c = confKey(s);
      games += Array.isArray(s.game_log) ? s.game_log.length : 0;
      if (r === 1 && r1Teams[c]) {
        if (s.team_high_id) r1Teams[c].add(String(s.team_high_id));
        if (s.team_low_id) r1Teams[c].add(String(s.team_low_id));
      }
      if (conf[c]) {
        conf[c][r] = conf[c][r] || { done: 0, live: 0, total: SERIES_PER_CONF[r] || 1 };
        if (st === "complete") conf[c][r].done += 1;
        if (st === "live") conf[c][r].live += 1;
      }
      if (st === "complete") {
        if (per[r]) per[r].done += 1;
        const w = seriesWinnerId(s);
        out.add(w === String(s.team_high_id) ? String(s.team_low_id) : String(s.team_high_id));
      }
    }
    const alive = (c) => [...r1Teams[c]].filter((t) => !out.has(t)).length;
    const decided = Object.values(per).reduce((a, b) => a + b.done, 0);
    return { per, conf, games, decided, alive: { West: alive("West"), East: alive("East") }, r1: { West: r1Teams.West.size, East: r1Teams.East.size } };
  }, [seriesList, isLive]);

  const totalTeams = counts.r1.West + counts.r1.East || 16;

  let headline;
  let subline;
  if (champion) {
    headline = `${nickOf(champion)} win the Stanley Cup`;
    subline = `${counts.games} playoff games · ${seasonLabel || "this season"}`;
  } else if (cupComplete) {
    headline = "The Cup has been decided";
    subline = `${counts.games} playoff games played`;
  } else if (isLive) {
    headline = `${roundName(currentRound)} · Day ${playoffDay + 1}`;
    subline = `${overview.alive} teams remaining · ${counts.decided} of 15 series decided · ${counts.games} games played`;
  } else {
    headline = "The road to the Cup";
    subline = r1Filled ? `${totalTeams} teams · ${r1Filled} opening-round series · one champion` : "The bracket is being set";
  }

  const userChip = (() => {
    if (!me) return null;
    if (championId && championId === uid) return { tone: "gold", text: "Stanley Cup champions" };
    if (!userCurrent) return r1Filled >= 8 ? { tone: "muted", text: "Missed the playoffs" } : null;
    if (userEliminated) return { tone: "muted", text: `Out in the ${roundName(userCurrent.round_index)}` };
    const st = seriesState(userCurrent, isLive);
    const opp = teamFor(userCurrent.team_high_id === uid ? userCurrent.team_low_id : userCurrent.team_high_id);
    if (st === "complete") return { tone: "live", text: `Advanced past ${opp?.abbrev || "opponent"}` };
    if (st === "scheduled") return { tone: "live", text: `vs ${opp?.abbrev || "TBD"} · ${roundName(userCurrent.round_index)}` };
    const label = seriesLabel(userCurrent, teamFor(userCurrent.team_high_id), teamFor(userCurrent.team_low_id));
    return { tone: userGameTonight ? "gold" : "live", text: `${label}${userGameTonight ? " · tonight" : ""}` };
  })();

  /* ── bracket helpers ── */

  const pathIds = useMemo(() => {
    const ids = new Set();
    if (!userCurrent || !bracket.structured) return ids;
    seriesList.forEach((s) => {
      if (s.is_user_series && s.team_high_id && s.team_low_id) ids.add(s.series_id);
    });
    if (userEliminated) return ids;
    const side = bracket[confKey(userCurrent)];
    if (side) {
      let slot = Math.max(0, (side[userCurrent.round_index] || []).findIndex((x) => x.series_id === userCurrent.series_id));
      for (let rr = userCurrent.round_index + 1; rr <= 3; rr += 1) {
        slot = Math.floor(slot / 2);
        const s = side[rr]?.[slot];
        if (s) ids.add(s.series_id);
      }
    }
    if (bracket.cup) ids.add(bracket.cup.series_id);
    return ids;
  }, [userCurrent, userEliminated, bracket, seriesList]);

  const renderCard = (series, feeders) => {
    const state = seriesState(series, isLive);
    return (
      <SeriesCard
        series={{ ...series, game_log: Array.isArray(series.game_log) ? series.game_log : [] }}
        high={teamFor(series.team_high_id)}
        low={teamFor(series.team_low_id)}
        state={state}
        feeders={feeders}
        teamFor={teamFor}
        selected={selected?.series_id === series.series_id}
        justSet={justSetIds.has(series.series_id)}
        playoffDay={playoffDay}
        onSelect={setSelectedId}
      />
    );
  };

  const roundStatus = (round, conf) => {
    const c = conf ? counts.conf[conf]?.[round] : null;
    const done = c ? c.done : counts.per[round]?.done || 0;
    const total = c ? c.total : counts.per[round]?.total || 1;
    if (done >= total) return { tone: "done", text: total === 1 ? "Decided" : "Complete" };
    if (isLive && currentRound === round) return { tone: "current", text: `In progress · ${done} of ${total} decided` };
    if (!isLive && round === 1 && !cupComplete) return { tone: "next", text: "Opens next" };
    return { tone: "locked", text: "Locked" };
  };

  const sideRounds = (conf) => (conf === "East" ? [3, 2, 1] : [1, 2, 3]);
  const roundLabel = (round) => (round === 3 ? "Conference Final" : ROUND_NAMES[round]);

  const renderHeads = (conf) =>
    sideRounds(conf).map((round) => {
      const st = roundStatus(round, conf);
      return (
        <div key={`h-${conf}-${round}`} className={cx("pf-colhead", `is-${st.tone}`, conf === "East" && "is-east")}>
          <b>{roundLabel(round)}</b>
          <small>{st.text}</small>
        </div>
      );
    });

  const renderSideColumns = (conf) => {
    const side = bracket[conf];
    return sideRounds(conf).map((round) => {
      const rows = side[round];
      const slot = (s, idx) => {
        const st = seriesState(s, isLive);
        return (
          <div
            key={s.series_id}
            className={cx(
              "pf-slot",
              "has-out",
              round > 1 && "has-in",
              round < 3 ? (idx % 2 === 0 ? "is-top" : "is-bottom") : "is-straight",
              st === "complete" && "is-lit",
              st !== "pending" && "is-set",
              pathIds.has(s.series_id) && "is-path"
            )}
          >
            {renderCard(s, getFeeders(round, conf, idx))}
          </div>
        );
      };
      let body;
      if (round === 3) body = slot(rows[0], 0);
      else {
        body = [];
        for (let p = 0; p < rows.length; p += 2) {
          body.push(
            <div key={`pair-${p}`} className="pf-pair">
              {slot(rows[p], p)}
              {slot(rows[p + 1], p + 1)}
            </div>
          );
        }
      }
      return (
        <div key={`${conf}-${round}`} className={cx("pf-col", conf === "West" ? "is-west" : "is-east", `is-r${round}`)}>
          {body}
        </div>
      );
    });
  };

  const cupLine = (() => {
    if (currentRound === 4 && isLive) return "The Final is on. Best of seven for the Cup.";
    if (isLive) return `${overview.alive} teams left · ${15 - counts.decided} series to go`;
    return "Four rounds. Sixteen teams. One champion.";
  })();

  const renderCupColumn = () => (
    <div className="pf-col is-cup">
      <div className="pf-cup-anchor">
        <div className={cx("pf-cup-hero", champion && "is-won", currentRound === 4 && isLive && "is-final")}>
          <span className="pf-cup-light" aria-hidden="true" />
          <CupCenterpiece size={champion ? 140 : 118} lit={Boolean(champion) || currentRound === 4} />
          <div className="pf-cup-name">{champion ? nickOf(champion) : "The Stanley Cup"}</div>
          <p className="pf-cup-line">{champion ? `${champion.name} · champions${seasonLabel ? ` ${seasonLabel}` : ""}` : cupLine}</p>
        </div>
        <div className={cx("pf-slot", "is-final", "has-in", pathIds.has(bracket.cup.series_id) && "is-path", seriesState(bracket.cup, isLive) === "complete" && "is-lit")}>
          {renderCard(bracket.cup, getFeeders(4))}
        </div>
        <div className="pf-cup-progress" aria-label={`${counts.decided} of 15 series decided`}>
          <div className="pf-ticks">
            {[1, 2, 3, 4].flatMap((r) =>
              Array.from({ length: counts.per[r].total }, (_, i) => (
                <i key={`${r}-${i}`} className={cx(`is-r${r}`, i < counts.per[r].done && "is-on")} />
              ))
            )}
          </div>
          <span>
            {counts.decided} of 15 series decided
          </span>
        </div>
      </div>
    </div>
  );

  const stepState = (round) => {
    if (cupComplete || round < currentRound) return "done";
    if (round === currentRound && isLive) return "current";
    if (!isLive && round === 1) return "next";
    return "locked";
  };

  // The featured series: the Final once it is set, else your club, else the biggest game tonight.
  const featured = (() => {
    const cup = bracket.structured ? bracket.cup : seriesList.find((s) => s.round_index === 4);
    if (cup && cup.team_high_id && cup.team_low_id) return cup;
    if (userCurrent && !userEliminated) return userCurrent;
    const liveRows = seriesList.filter((s) => seriesState(s, isLive) === "live");
    return (
      liveRows.find((s) => Number(s.wins_high) === 3 || Number(s.wins_low) === 3) ||
      liveRows.find((s) => numOrNull(s.scheduled_day) === playoffDay) ||
      null
    );
  })();

  // Your path to the Cup: who stands in each round still ahead.
  const userRoad = (() => {
    if (!userCurrent || !bracket.structured || userEliminated || cupComplete) return [];
    const conf = confKey(userCurrent);
    const side = bracket[conf];
    const other = conf === "West" ? "East" : "West";
    if (!side) return [];
    const r = userCurrent.round_index;
    const slot = Math.max(0, (side[r] || []).findIndex((x) => x.series_id === userCurrent.series_id));
    const oppId = userCurrent.team_high_id === uid ? userCurrent.team_low_id : userCurrent.team_high_id;
    const say = (txt) => {
      if (!txt || txt === "TBD") return "To be decided";
      return txt.includes("/") ? `${txt.replace("/", " / ")} winner` : `vs ${txt}`;
    };
    const steps = [{ round: r, label: `vs ${nickOf(teamFor(oppId))}`, now: true, done: Boolean(seriesWinnerId(userCurrent)) }];
    let s2 = slot;
    for (let rr = r + 1; rr <= 3; rr += 1) {
      const sib = (side[rr - 1] || [])[s2 ^ 1];
      steps.push({ round: rr, label: say(sib ? feederText(sib, teamFor) : "TBD") });
      s2 = Math.floor(s2 / 2);
    }
    if (r < 4) {
      const otherFinal = (bracket[other]?.[3] || [])[0];
      const t = otherFinal ? feederText(otherFinal, teamFor) : "TBD";
      steps.push({ round: 4, label: t === "TBD" || t.includes("/") ? `${other}ern champion` : `vs ${t}` });
    }
    return steps;
  })();

  const askConfirm = useCallback((key, text, run) => setConfirm({ key, text, run }), []);

  const nextUp = (() => {
    if (cupComplete) {
      return {
        tone: "gold",
        over: "Season complete",
        head: champion ? `${champion.name} win the Stanley Cup` : "The Cup has been decided",
        sub: champion && champion.team_id === uid ? "Your club are champions." : "The awards are next.",
        cta: { label: "Continue to awards", onClick: handleContinue },
      };
    }
    if (!isLive) {
      const home = userCurrent ? teamFor(userCurrent.team_high_id) : null;
      return {
        tone: "live",
        over: userCurrent ? "Featured series · Your franchise" : "Opening round",
        head: userCurrent ? "" : `${r1Filled} opening-round series are set`,
        sub: userCurrent
          ? `First Round · Game 1 in ${cityOf(home)} · ${nickOf(home)} hold home ice`
          : "Your club missed the playoffs. Watch it play out or sim to the Cup.",
        cta: { label: "Start Round 1", onClick: handleEnter },
        note: "Opens the round. Games are then played one night at a time.",
        alt: userCurrent ? null : { label: "Sim entire playoffs", onClick: () => askConfirm("all", "Sim every round through the Stanley Cup Final? You can stop it at any time.", handleFastForward) },
      };
    }
    if (userGameTonight && userActive) {
      const n = nextGameNumber(userActive);
      const home = isHighSeedHome(n) ? teamFor(userActive.team_high_id) : teamFor(userActive.team_low_id);
      return {
        tone: "gold",
        over: `Tonight · ${roundName(currentRound)}`,
        head: "",
        sub: `Game ${n} in ${cityOf(home)} · ${seriesLabel(userActive, teamFor(userActive.team_high_id), teamFor(userActive.team_low_id))}`,
        cta: { label: `Play Game ${n}`, onClick: () => runAction("play_user_game", { series_id: userActive.series_id }, "Playing your game…") },
        note: "Other series wait until your game is played or simmed.",
      };
    }
    if (userActive) {
      const n = nextGameNumber(userActive);
      const d = numOrNull(userActive.scheduled_day);
      const home = isHighSeedHome(n) ? teamFor(userActive.team_high_id) : teamFor(userActive.team_low_id);
      return {
        tone: "live",
        over: `Featured series · Your franchise`,
        head: "",
        sub: `Next: Game ${n} in ${cityOf(home)}${d !== null ? ` on Day ${d + 1}` : ""} · ${seriesLabel(userActive, teamFor(userActive.team_high_id), teamFor(userActive.team_low_id))}`,
        cta: { label: "Sim to my next game", onClick: handleFastForward },
        note: "Plays the other series night by night until your club is up.",
      };
    }
    const tonightCount = overview.tonight.length;
    const fs = featured;
    return {
      tone: userEliminated ? "muted" : "live",
      over: fs ? (fs.round_index === 4 ? "Stanley Cup Final" : "Featured series") : roundName(currentRound),
      head: fs ? "" : userEliminated ? "Your season is over" : tonightCount ? `${tonightCount} ${tonightCount === 1 ? "game" : "games"} tonight` : "Between games",
      sub: fs
        ? `${roundName(fs.round_index)} · ${seriesLabel(fs, teamFor(fs.team_high_id), teamFor(fs.team_low_id))}${userEliminated ? " · your season is over" : ""}`
        : userEliminated
          ? "The rest of the bracket plays on."
          : `${overview.active} series still alive in this round.`,
      cta: userEliminated
        ? { label: "Sim to the Cup", onClick: () => askConfirm("cup", "Sim the rest of the playoffs through the Final? You can stop it at any time.", handleFastForward) }
        : { label: "Sim next game", onClick: () => runAction("advance_day", { skip_off_days: true }, "Playing the next game in every series…") },
      note: userEliminated ? "" : "Every live series plays its next game.",
    };
  })();

  // Desk falls back to the user's own series so it is always on screen.
  const deskSeries =
    selected ||
    (userCurrent ? { ...userCurrent, game_log: Array.isArray(userCurrent.game_log) ? userCurrent.game_log : [] } : null);

  const pair = simSeriesTarget
    ? `${teamFor(simSeriesTarget.team_high_id)?.abbrev}–${teamFor(simSeriesTarget.team_low_id)?.abbrev}`
    : "";

  const stageRound = featured ? featured.round_index : cupComplete ? 4 : Math.min(4, currentRound || 1);
  const featHigh = featured ? teamFor(featured.team_high_id) : null;
  const featLow = featured ? teamFor(featured.team_low_id) : null;
  const featState = featured ? seriesState(featured, isLive) : null;
  const featNext = featured ? nextGameNumber(featured) : 1;
  const featHome = featured && featState !== "complete" ? (isHighSeedHome(featNext) ? featHigh : featLow) : null;
  const featWinner = featured ? seriesWinnerId(featured) : "";

  const renderFeatTeam = (team, seed, align, wins, otherWins) => (
    <div
      className={cx(
        "pf-feat-team",
        `is-${align}`,
        featWinner && featWinner === team?.team_id && "is-winner",
        featWinner && featWinner !== team?.team_id && "is-out",
        team?.team_id === uid && "is-mine"
      )}
    >
      <TeamMark team={team} size={76} dimmed={Boolean(featWinner) && featWinner !== team?.team_id} />
      <span className="pf-feat-name">
        <small>
          {cityOf(team)}
          {featHome && featHome.team_id === team?.team_id ? <HomeIce /> : null}
        </small>
        <b>{nickOf(team)}</b>
        <em>
          {seed ? `Seed ${seed} · ` : ""}
          {team?.record || ""}
          {team?.pts !== null && team?.pts !== undefined ? ` · ${team.pts} pts` : ""}
        </em>
      </span>
    </div>
  );

  return (
    <div className={cx("pf-root", champion && "is-crowned", deskCollapsed && "is-desk-collapsed")} data-round={currentRound || 1}>
      {popups.length ? (
        <div className="pf-pop-stack" aria-live="polite">
          {popups.map((p) => (
            <div key={p.id} className={cx("pf-pop", `is-${p.tone}`, p.user && "is-user")} role="status">
              <div className="pf-pop-top">
                <span className="pf-pop-label">{p.tag}</span>
                <button type="button" className="pf-pop-x" aria-label="Dismiss" onClick={() => dismissPopup(p.id)}>
                  ×
                </button>
              </div>
              <strong className="pf-pop-head">{p.title}</strong>
              <span className="pf-pop-body">{p.body}</span>
            </div>
          ))}
        </div>
      ) : null}

      <header className="pf-top">
        <button type="button" className="pf-hub" onClick={onBack} aria-label="Back to the franchise hub">
          <span aria-hidden="true">←</span> Hub
        </button>
        <div className="pf-brand">
          <span className="pf-brand-mark" aria-hidden="true">
            <TrophyGlyph size={30} />
          </span>
          <div className="pf-brand-copy">
            <p className="pf-season">Stanley Cup Playoffs{seasonLabel ? ` · ${seasonLabel}` : ""}</p>
            <div className="pf-headline" role="heading" aria-level={1}>
              {headline}
            </div>
            <p className="pf-subline">{subline}</p>
          </div>
        </div>

        <ol className="pf-prog" aria-label="Playoff progression">
          {[1, 2, 3, 4].map((r) => {
            const st = stepState(r);
            const c = counts.per[r];
            let note = "Locked";
            if (st === "done") note = "Complete";
            else if (st === "current") note = `${c.done} of ${c.total} decided`;
            else if (st === "next") note = "Opens next";
            return (
              <li key={r} className={cx("pf-prog-step", `is-${st}`)} aria-current={st === "current" ? "step" : undefined}>
                <span className="pf-prog-num">{st === "done" ? "✓" : String(r).padStart(2, "0")}</span>
                <span className="pf-prog-copy">
                  <b>{r === 4 ? "Stanley Cup Final" : r === 3 ? "Conference Finals" : ROUND_NAMES[r]}</b>
                  <small>{note}</small>
                </span>
              </li>
            );
          })}
        </ol>

        {userChip ? (
          <button
            type="button"
            className={cx("pf-mine", `is-${userChip.tone}`)}
            disabled={!userCurrent}
            onClick={() => {
              if (userCurrent) {
                setSelectedId(userCurrent.series_id);
                setDeskCollapsed(false);
              }
            }}
            title={userCurrent ? "Open your series" : undefined}
          >
            <TeamMark team={me} size={34} />
            <span>
              <small>Your franchise · {me?.abbrev}</small>
              <b>{userChip.text}</b>
            </span>
          </button>
        ) : null}
      </header>

      {notice ? (
        <div className={cx("pf-notice", `is-${notice.tone}`)} role={notice.tone === "error" ? "alert" : "status"}>
          <span className="pf-notice-text">{notice.text}</span>
          {notice.seriesId && userActive && notice.seriesId === userActive.series_id && canAct ? (
            <button
              type="button"
              className="pf-act is-primary is-sm"
              disabled={busy}
              onClick={() => runAction("play_user_game", { series_id: userActive.series_id }, "Playing your game…")}
            >
              Play game
            </button>
          ) : null}
          <button type="button" className="pf-icon-act" aria-label="Dismiss" onClick={() => setNotice(null)}>
            ×
          </button>
        </div>
      ) : null}

      <section className="pf-body">
        <div className="pf-main">
          <section className={cx("pf-stage", `is-${nextUp.tone}`, featured?.round_index === 4 && "is-final", champion && "is-crowned")}>
            <div className="pf-stage-round" aria-label={`Round ${stageRound} of 4`}>
              <small>Round</small>
              <b>{stageRound}</b>
              <span>of 4</span>
            </div>
            <div className="pf-stage-main">
              <p className="pf-stage-over">{nextUp.over}</p>
              {featured && featHigh && featLow ? (
                <button
                  type="button"
                  className="pf-feat"
                  onClick={() => {
                    setSelectedId(featured.series_id);
                    setDeskCollapsed(false);
                  }}
                  title="Open this series in the series centre"
                >
                  {renderFeatTeam(featHigh, firstDefined(featured.seed_high, featHigh?.seed), "l")}
                  <div className="pf-feat-score">
                    <div>
                      <b className={featured.wins_high > featured.wins_low ? "is-lead" : ""}>{featured.wins_high}</b>
                      <i>:</i>
                      <b className={featured.wins_low > featured.wins_high ? "is-lead" : ""}>{featured.wins_low}</b>
                    </div>
                    <small>Series score</small>
                    <em>
                      {featState === "complete"
                        ? "Final"
                        : `Game ${featNext}${featState === "live" && numOrNull(featured.scheduled_day) === playoffDay ? " · Tonight" : ""}`}
                    </em>
                  </div>
                  {renderFeatTeam(featLow, firstDefined(featured.seed_low, featLow?.seed), "r")}
                </button>
              ) : (
                <div className="pf-stage-head">{nextUp.head}</div>
              )}
              <p className="pf-stage-line">{nextUp.sub}</p>
              {userRoad.length ? (
                <ol className="pf-path" aria-label="Your path to the Cup">
                  <li className="pf-path-lab">Path to the Cup</li>
                  {userRoad.map((st) => (
                    <li key={st.round} className={cx("pf-path-step", st.now && "is-now", st.done && "is-done")}>
                      <small>{st.round === 3 ? "Conference Final" : ROUND_NAMES[st.round]}</small>
                      <b>{st.label}</b>
                    </li>
                  ))}
                </ol>
              ) : null}
            </div>
            <div className="pf-stage-side">
              <dl className="pf-status">
                <div>
                  <dt>Teams remaining</dt>
                  <dd>{overview.alive}</dd>
                </div>
                <div>
                  <dt>Series decided</dt>
                  <dd>
                    {counts.decided}
                    <small>/15</small>
                  </dd>
                </div>
                <div>
                  <dt>Games played</dt>
                  <dd>{counts.games}</dd>
                </div>
              </dl>
              <button type="button" className="pf-go" disabled={busy} onClick={nextUp.cta.onClick}>
                {busy ? busyLabel || "Working…" : nextUp.cta.label}
              </button>
              {nextUp.note ? <small className="pf-go-note">{nextUp.note}</small> : null}
              {nextUp.alt ? (
                <button type="button" className="pf-text-act" disabled={busy} onClick={nextUp.alt.onClick}>
                  {nextUp.alt.label}
                </button>
              ) : null}
            </div>
          </section>

          <div className="pf-bracket-wrap">
            {bracket.structured ? (
              <div className="pf-bracket">
                <div className="pf-conf is-west">
                  <b>Western Conference</b>
                  <span>
                    {counts.alive.West} of {counts.r1.West || 8} teams alive
                  </span>
                </div>
                <div className="pf-conf is-mid">
                  <b>Stanley Cup Final</b>
                  <span>{roundStatus(4).text}</span>
                </div>
                <div className="pf-conf is-east">
                  <b>Eastern Conference</b>
                  <span>
                    {counts.alive.East} of {counts.r1.East || 8} teams alive
                  </span>
                </div>
                {renderHeads("West")}
                <div className="pf-colhead is-cup" aria-hidden="true" />
                {renderHeads("East")}
                {renderSideColumns("West")}
                {renderCupColumn()}
                {renderSideColumns("East")}
              </div>
            ) : (
              <div className="pf-bracket is-flat">
                {bracket.rounds.map((rows, i) => (
                  <div key={i} className="pf-flat-col">
                    <div className="pf-colhead">
                      <b>{roundName(i + 1)}</b>
                    </div>
                    {rows.length ? (
                      rows.map((s) => (
                        <div key={s.series_id} className="pf-slot">
                          {renderCard(s, null)}
                        </div>
                      ))
                    ) : (
                      <p className="pf-note">Awaiting series</p>
                    )}
                  </div>
                ))}
              </div>
            )}
            <p className="pf-legend">
              <span>
                <i className="pf-legend-seed">1</i> Playoff seed
              </span>
              <span>
                <HomeIce title="Home ice" /> Home ice for the next game
              </span>
              <span>
                <i className="pf-legend-mine" /> Your franchise and its path
              </span>
              <span>Select a series for its full report</span>
            </p>
          </div>
        </div>

        {deskCollapsed ? (
          <aside className="pf-desk is-collapsed">
            <button type="button" className="pf-rail-open" onClick={() => setDeskCollapsed(false)} aria-label="Open the series centre">
              <span aria-hidden="true">⟨</span>
              <b>Series centre</b>
            </button>
          </aside>
        ) : (
          <SeriesDesk
            series={deskSeries}
            high={teamFor(deskSeries?.team_high_id)}
            low={teamFor(deskSeries?.team_low_id)}
            state={seriesState(deskSeries, isLive)}
            canAct={canAct}
            busy={busy}
            playoffDay={playoffDay}
            uid={uid}
            onClear={() => setSelectedId(null)}
            onAction={runAction}
            overview={overview}
            onCollapse={() => setDeskCollapsed(true)}
          />
        )}
      </section>

      <footer className="pf-dock">
        {confirm ? (
          <div className="pf-confirm" role="alertdialog" aria-label="Confirm simulation">
            <span>{confirm.text}</span>
            <button
              type="button"
              className="pf-act is-primary is-sm"
              onClick={() => {
                const run = confirm.run;
                setConfirm(null);
                run();
              }}
            >
              Yes, sim it
            </button>
            <button type="button" className="pf-act is-quiet is-sm" onClick={() => setConfirm(null)}>
              Cancel
            </button>
          </div>
        ) : isLive && !cupComplete ? (
          <div className="pf-dock-group" role="group" aria-label="Simulation">
            <span className="pf-dock-lab">Simulate</span>
            <button
              type="button"
              className="pf-act is-quiet"
              disabled={busy || userGameTonight}
              title={userGameTonight ? "Your game is tonight. Play or sim it first." : "Every live series plays its next game, on the same night"}
              onClick={() => runAction("advance_day", { skip_off_days: true }, "Playing the next game in every series…")}
            >
              Next game
            </button>
            <button
              type="button"
              className="pf-act is-quiet"
              disabled={busy || !simSeriesTarget}
              title="Finish the selected series (or the next active one)"
              onClick={handleSimSeries}
            >
              {pair ? `Finish ${pair}` : "Finish series"}
            </button>
            {userAlive ? (
              <button
                type="button"
                className="pf-act is-quiet"
                disabled={busy || userGameTonight}
                title="Sim night by night until your club's next game"
                onClick={handleFastForward}
              >
                To my next game
              </button>
            ) : null}
            <button
              type="button"
              className="pf-act is-quiet"
              disabled={busy || (userAlive && userGameTonight)}
              title="Sim night by night until the current round is over"
              onClick={() =>
                askConfirm(
                  "round",
                  `Sim the rest of the ${roundName(currentRound)}? ${overview.active} series still in play${userAlive ? ", including yours" : ""}.`,
                  handleSimRound
                )
              }
            >
              Rest of round
            </button>
            {!userAlive ? (
              <button
                type="button"
                className="pf-act is-quiet"
                disabled={busy}
                onClick={() => askConfirm("cup", "Sim the rest of the playoffs through the Final? You can stop it at any time.", handleFastForward)}
              >
                To the Cup
              </button>
            ) : null}
          </div>
        ) : (
          <span className="pf-dock-lab">{cupComplete ? "Playoffs complete" : "The round opens when you start it."}</span>
        )}

        <div className="pf-speed" role="group" aria-label="Sim speed">
          <span className="pf-dock-lab">Sim speed</span>
          {Object.entries(SIM_SPEEDS).map(([key, cfg]) => (
            <button key={key} type="button" className={cx("pf-speed-opt", speed === key && "is-on")} aria-pressed={speed === key} onClick={() => setSpeed(key)}>
              {cfg.label}
            </button>
          ))}
        </div>

        {busy ? (
          <span className="pf-dock-status" role="status">
            <i className="pf-spinner" aria-hidden="true" />
            {busyLabel || "Working…"}
            {fastForwarding ? (
              <button
                type="button"
                className="pf-act is-quiet is-sm"
                onClick={() => {
                  abortRef.current = true;
                }}
              >
                Stop
              </button>
            ) : null}
          </span>
        ) : null}
      </footer>
    </div>
  );
}
