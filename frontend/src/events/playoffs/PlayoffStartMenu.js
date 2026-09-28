import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveFranchiseTeamLogo, toLogoUrl } from "../../utils/teamLogos";
import { enterPlayoffs, playoffAction } from "../../services/franchiseService";
import { isNetworkError, formatFranchiseApiError, isTimeoutError } from "../../services/api";
import { useGameUI } from "../../game/GameUIContext";
import "../../styles/nhlcalShell.css";

/**
 * Stanley Cup Playoffs bracket hub (Broadcast Operations register).
 * West R1 → R2 → Final → CUP ← East Final ← R2 ← R1
 *
 * All styles are scoped under .po-hub-root so nothing leaks into CalendarScreen.
 */

/* ───────────────────────── constants ───────────────────────── */

const ROUND_NAMES = { 1: "Round 1", 2: "Round 2", 3: "Conference Final", 4: "Stanley Cup Final" };
const ROUND_SHORT = { 1: "R1", 2: "R2", 3: "CF", 4: "Final" };
const SERIES_PER_CONF = { 1: 4, 2: 2, 3: 1 };
const MAX_FF_DAYS = 150; // hard cap on sequential advance_day calls
const FF_PAINT_MS = 60; // let React paint the bracket between days
const STALL_LIMIT = 3; // consecutive responses with no day change → stop

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

/** Which feeder fills which empty row — keeps a half-set series honest. */
function pendingRowLabels(series, feeders, teamFor) {
  if (!feeders) return ["TBD", "TBD"];
  const [f0, f1] = feeders;
  const contains = (f, id) =>
    Boolean(f && id) && (String(f.team_high_id) === id || String(f.team_low_id) === id);
  let fHigh = f0;
  let fLow = f1;
  const hi = series.team_high_id;
  const lo = series.team_low_id;
  if (hi && contains(f0, hi)) fLow = f1;
  else if (hi && contains(f1, hi)) fLow = f0;
  if (lo && contains(f0, lo)) fHigh = f1;
  else if (lo && contains(f1, lo)) fHigh = f0;
  return [`W ${feederText(fHigh, teamFor)}`, `W ${feederText(fLow, teamFor)}`];
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

function SideNavButton({ active, icon, label, onClick }) {
  return (
    <button
      type="button"
      className={`nhlcal-side-button${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      <span className="nhlcal-side-icon">{icon}</span>
      <span className="nhlcal-side-label">{label}</span>
    </button>
  );
}

function TrophyGlyph({ size = 34 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="po-trophy-glyph">
      <path d="M9 4h14v5a7 7 0 0 1-14 0V4Z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M9 6H5.5a3.5 3.5 0 0 0 3.8 4.4M23 6h3.5a3.5 3.5 0 0 1-3.8 4.4" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M16 16v5M12 21h8l1.5 3h-11L12 21Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M9 24h14v4H9z" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function TeamMark({ team, size = 22, dimmed }) {
  if (!team?.team_id) {
    return <span className="po-mark is-empty" style={{ width: size, height: size }} />;
  }
  return (
    <span className={cx("po-mark", dimmed && "is-dimmed")} style={{ width: size, height: size }}>
      {team.logo ? (
        <img src={team.logo} alt="" />
      ) : (
        <span className="po-mark-fallback">{String(team.abbrev || "?").slice(0, 3)}</span>
      )}
    </span>
  );
}

function TeamRow({ team, seed, wins, showWins, placeholder, isWinner, isLoser, isLeader }) {
  const empty = !team;
  return (
    <div
      className={cx(
        "po-row",
        empty && "is-empty",
        isWinner && "is-winner",
        isLoser && "is-out",
        isLeader && "is-lead"
      )}
    >
      <span className="po-seed">{empty ? "" : seed ?? ""}</span>
      <TeamMark team={team} size={22} dimmed={isLoser} />
      <span className="po-row-name">
        <strong>{empty ? placeholder : team.abbrev}</strong>
        {!empty && team.record ? <small>{team.record}</small> : null}
      </span>
      <em className="po-row-wins">{showWins ? wins : ""}</em>
    </div>
  );
}

/* ───────────────────────── bracket card ───────────────────────── */

function SeriesCard({ series, high, low, state, feeders, teamFor, selected, justSet, playoffDay, onSelect }) {
  const next = nextGameNumber(series);
  const home = isHighSeedHome(next) ? high : low;
  const day = numOrNull(series.scheduled_day);
  const tonight = state === "live" && day !== null && day === playoffDay;
  const winner = seriesWinnerId(series);
  const wh = series.wins_high;
  const wl = series.wins_low;
  const showWins = state === "live" || state === "complete";
  const [phHigh, phLow] = state === "pending" ? pendingRowLabels(series, feeders, teamFor) : ["TBD", "TBD"];

  let meta;
  if (state === "pending") meta = "Awaiting";
  else if (state === "complete") meta = "Final";
  else if (tonight) meta = `G${next} · Tonight`;
  else if (state === "live") meta = day !== null ? `G${next} · Day ${day + 1}` : `Game ${next}`;
  else meta = day !== null ? `G1 · Day ${day + 1}` : "Game 1";
  const metaRight = state === "live" || state === "scheduled" ? `@${home?.abbrev || "—"}` : "";

  const footer =
    state === "complete" || (state === "live" && series.game_log.length)
      ? seriesLabel(series, high, low)
      : null;

  return (
    <button
      type="button"
      className={cx(
        "po-card",
        `is-${state}`,
        selected && "is-selected",
        series.is_user_series && "is-user",
        tonight && "is-tonight",
        justSet && "is-just-set"
      )}
      disabled={state === "pending"}
      aria-pressed={selected}
      aria-label={`${roundName(series.round_index)}: ${high?.abbrev || "TBD"} vs ${low?.abbrev || "TBD"}`}
      onClick={() => onSelect?.(series.series_id)}
    >
      <div className="po-card-meta">
        <span className="po-card-meta-left">
          {series.is_user_series ? <span className="po-you">You</span> : null}
          {tonight ? <i className="po-dot" aria-hidden="true" /> : null}
          <span>{meta}</span>
        </span>
        {metaRight ? <span>{metaRight}</span> : null}
      </div>
      <TeamRow
        team={high}
        seed={firstDefined(series.seed_high, high?.seed)}
        wins={wh}
        showWins={showWins}
        placeholder={phHigh}
        isWinner={Boolean(winner) && winner === series.team_high_id}
        isLoser={Boolean(winner) && winner !== series.team_high_id}
        isLeader={state === "live" && wh > wl}
      />
      <TeamRow
        team={low}
        seed={firstDefined(series.seed_low, low?.seed)}
        wins={wl}
        showWins={showWins}
        placeholder={phLow}
        isWinner={Boolean(winner) && winner === series.team_low_id}
        isLoser={Boolean(winner) && winner !== series.team_low_id}
        isLeader={state === "live" && wl > wh}
      />
      {footer ? <div className="po-card-foot">{footer}</div> : null}
    </button>
  );
}

/* ───────────────────────── series desk ───────────────────────── */

function SeriesDesk({ series, high, low, state, canAct, busy, playoffDay, onClear, onAction }) {
  if (!series) {
    return (
      <aside className="po-desk">
        <p className="po-eyebrow">Series Desk</p>
        <h3 className="po-desk-title">No series selected</h3>
        <p className="po-desk-muted">
          Select any matchup in the bracket to see the series score, the seven-game slate, and sim controls.
        </p>
      </aside>
    );
  }

  const games = series.game_log;
  const next = nextGameNumber(series);
  const day = numOrNull(series.scheduled_day);
  const tonight = state === "live" && day !== null && day === playoffDay;
  const home = isHighSeedHome(next) ? high : low;
  const away = isHighSeedHome(next) ? low : high;
  const wh = series.wins_high;
  const wl = series.wins_low;
  const abbrFor = (id) => {
    const s = String(id || "");
    if (s && s === series.team_high_id) return high?.abbrev || "H";
    if (s && s === series.team_low_id) return low?.abbrev || "A";
    return s.slice(0, 3).toUpperCase() || "—";
  };
  const scheduleDates = Array.isArray(series.schedule_dates) ? series.schedule_dates : [];

  let nextLine = null;
  if (state === "pending") nextLine = "Waiting on the previous round.";
  else if (state === "complete") nextLine = "Series complete — winner advances.";
  else if (state === "scheduled") nextLine = `Game 1 at ${home?.abbrev || "home"}${day !== null ? ` · Day ${day + 1}` : ""}. Starts when Round 1 opens.`;
  else if (tonight) nextLine = `Tonight · Game ${next}: ${away?.abbrev} at ${home?.abbrev}`;
  else nextLine = `Next · Game ${next} at ${home?.abbrev}${day !== null ? ` · Day ${day + 1}` : ""}`;

  return (
    <aside className="po-desk">
      <div className="po-desk-head">
        <div>
          <p className="po-eyebrow">
            {roundName(series.round_index)}
            {series.round_index === 4 ? "" : ` · ${confKey(series)}`}
          </p>
          <h3 className="po-desk-title">
            {high?.abbrev || "TBD"} vs {low?.abbrev || "TBD"}
          </h3>
        </div>
        <button type="button" className="po-btn po-btn--ghost po-btn--icon" aria-label="Clear selection" onClick={onClear}>
          ×
        </button>
      </div>

      <div className="po-desk-score">
        {[{ team: high, wins: wh, lead: wh > wl }, null, { team: low, wins: wl, lead: wl > wh }].map((side, i) =>
          side ? (
            <div key={i} className="po-desk-team">
              <TeamMark team={side.team} size={40} />
              <strong>{side.team?.abbrev || "TBD"}</strong>
              <small>
                {side.team?.record || "—"}
                {side.team?.pts !== null && side.team?.pts !== undefined ? ` · ${side.team.pts} pts` : ""}
              </small>
            </div>
          ) : (
            <div key={i} className="po-desk-wins">
              <span className={wh > wl ? "is-lead" : ""}>{state === "pending" ? "–" : wh}</span>
              <span className="po-desk-sep">:</span>
              <span className={wl > wh ? "is-lead" : ""}>{state === "pending" ? "–" : wl}</span>
            </div>
          )
        )}
      </div>

      <div>
        <p className="po-desk-status">{state === "pending" ? "Matchup not set" : seriesLabel(series, high, low)}</p>
        <p className="po-desk-muted">{nextLine}</p>
      </div>

      {state !== "pending" ? (
        <div className="po-facts">
          <span className="po-fact">
            Home ice <b>{high?.abbrev}</b>
          </span>
          <span className="po-fact">
            Format <b>Best of 7</b>
          </span>
          {series.is_user_series ? <span className="po-fact is-gold">Your club</span> : null}
        </div>
      ) : null}

      <div className="po-games">
        {Array.from({ length: 7 }, (_, i) => {
          const n = i + 1;
          const g = games.find((x) => Number(x.game) === n);
          const hIsHome = isHighSeedHome(n);
          const schedHome = hIsHome ? high?.abbrev : low?.abbrev;
          const schedAway = hIsHome ? low?.abbrev : high?.abbrev;
          const tipDay = numOrNull(scheduleDates[i]);
          const isTonightGame = state === "live" && n === next && tonight;
          const moot = !g && state === "complete";

          if (g) {
            const hs = Number(g.home_score);
            const as = Number(g.away_score);
            return (
              <div key={n} className={cx("po-game", "is-played", g.ot && "is-ot")}>
                <span className="po-game-n">G{n}</span>
                <span className="po-game-line">
                  <b className={as > hs ? "w" : ""}>
                    {abbrFor(g.away_id)} {as}
                  </b>
                  {" @ "}
                  <b className={hs > as ? "w" : ""}>
                    {abbrFor(g.home_id)} {hs}
                  </b>
                </span>
                <span className="po-game-tag">{g.ot ? "OT" : "Final"}</span>
              </div>
            );
          }
          return (
            <div key={n} className={cx("po-game", isTonightGame && "is-tonight", moot && "is-moot")}>
              <span className="po-game-n">G{n}</span>
              <span className="po-game-line">
                {state === "pending" ? "—" : `${schedAway || "TBD"} @ ${schedHome || "TBD"}`}
              </span>
              <span className="po-game-tag">
                {moot ? "Not needed" : isTonightGame ? "Tonight" : tipDay !== null ? `Day ${tipDay + 1}` : ""}
              </span>
            </div>
          );
        })}
      </div>

      {canAct && state === "live" ? (
        <div className="po-desk-actions">
          {series.is_user_series ? (
            <button
              type="button"
              className="po-btn po-btn--primary"
              disabled={busy}
              onClick={() => onAction("play_user_game", { series_id: series.series_id }, "Playing your game…")}
            >
              Play Game {next}
            </button>
          ) : null}
          <button
            type="button"
            className={cx("po-btn", series.is_user_series ? "po-btn--secondary" : "po-btn--primary")}
            disabled={busy}
            onClick={() => onAction("sim_series", { series_id: series.series_id }, "Simming series…")}
          >
            Sim Series
          </button>
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
        if (isBlockedResponse(res)) {
          surfaceBlocked(res);
          return res;
        }
        const sid = res?.result?.series?.series_id;
        if (sid) setSelectedId(String(sid));
        return res;
      }),
    [withBusy, applyState, surfaceBlocked]
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
   * Day-by-day sim. Stops on: finish, blocked (user's game tonight), stall, or Stop.
   * With the user alive this is "Sim to My Next Game"; eliminated/out it runs to the Cup.
   */
  const handleFastForward = useCallback(() => {
    const allowSimRest = !userAlive;
    return withBusy("Simming playoffs…", async () => {
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
          const res = await playoffAction("advance_day");
          applyState(res);
          if (isBlockedResponse(res)) {
            surfaceBlocked(res);
            return res;
          }
          if (isFinishedResponse(res)) return res;
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
          await sleep(FF_PAINT_MS);
        }
        if (allowSimRest) {
          setBusyLabel("Finishing Cup run…");
          const res = await playoffAction("sim_rest");
          applyState(res);
          return res;
        }
        setNotice({ tone: "error", text: `Stopped after ${MAX_FF_DAYS} days without reaching your next game.` });
        return null;
      } finally {
        if (mountedRef.current) setFastForwarding(false);
      }
    });
  }, [userAlive, withBusy, isLive, cupComplete, enterRaw, applyState, surfaceBlocked]);

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

  let eyebrow = `Stanley Cup Playoffs${seasonLabel ? ` · ${seasonLabel}` : ""}`;
  let title;
  if (champion) {
    eyebrow = `Stanley Cup Champions${seasonLabel ? ` · ${seasonLabel}` : ""}`;
    title = champion.name;
  } else if (cupComplete) title = "Cup decided";
  else if (isLive) title = `${roundName(currentRound)} · Day ${playoffDay + 1}`;
  else title = r1Filled ? "Round 1 matchups set" : "Bracket pending";

  const userChip = (() => {
    const me = uid ? teamFor(uid) : null;
    if (!me) return null;
    if (championId && championId === uid) return { tone: "gold", text: `${me.abbrev} · Cup champions` };
    if (!userCurrent) return r1Filled >= 8 ? { tone: "muted", text: `${me.abbrev} · Missed playoffs` } : null;
    if (userEliminated) return { tone: "muted", text: `${me.abbrev} · Out in ${ROUND_SHORT[userCurrent.round_index]}` };
    const st = seriesState(userCurrent, isLive);
    const opp = teamFor(userCurrent.team_high_id === uid ? userCurrent.team_low_id : userCurrent.team_high_id);
    if (st === "complete") return { tone: "cyan", text: `${me.abbrev} advanced` };
    if (st === "scheduled") return { tone: "cyan", text: `${me.abbrev} vs ${opp?.abbrev || "TBD"}` };
    const label = seriesLabel(userCurrent, teamFor(userCurrent.team_high_id), teamFor(userCurrent.team_low_id));
    return { tone: userGameTonight ? "gold" : "cyan", text: `${label}${userGameTonight ? " · Tonight" : ""}` };
  })();

  /* ── render helpers ── */

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

  const roundHeadClass = (round) =>
    cx("po-col-head", canAct && currentRound === round && "is-current");

  const renderSideColumns = (conf) => {
    const side = bracket[conf];
    const cols = [
      { round: 1, title: `${conf} R1` },
      { round: 2, title: `${conf} R2` },
      { round: 3, title: `${conf} Final` },
    ];
    const ordered = conf === "East" ? [...cols].reverse() : cols;
    return ordered.map(({ round, title: colTitle }) => {
      const rows = side[round];
      const slot = (s, idx) => (
        <div key={s.series_id} className={cx("po-slot", round > 1 && "has-in", round === 3 && "has-out")}>
          {renderCard(s, getFeeders(round, conf, idx))}
        </div>
      );
      let body;
      if (round === 3) body = slot(rows[0], 0);
      else {
        const pairs = [];
        for (let p = 0; p < rows.length; p += 2) {
          pairs.push(
            <div key={`pair-${p}`} className="po-pair">
              {slot(rows[p], p)}
              {slot(rows[p + 1], p + 1)}
            </div>
          );
        }
        body = pairs;
      }
      return (
        <div key={`${conf}-${round}`} className={cx("po-col", conf === "West" ? "is-west" : "is-east")}>
          <header className={roundHeadClass(round)}>{colTitle}</header>
          <div className="po-col-body">{body}</div>
        </div>
      );
    });
  };

  const renderCupColumn = () => (
    <div className="po-col is-cup">
      <header className={roundHeadClass(4)}>Stanley Cup Final</header>
      <div className="po-col-body">
        <div className="po-slot has-in">
          <div className="po-cup-wrap">
            <div className="po-cup-crest">
              <TrophyGlyph />
            </div>
            {renderCard(bracket.cup, getFeeders(4))}
            {champion ? <div className="po-cup-champ">{champion.name}</div> : null}
          </div>
        </div>
      </div>
    </div>
  );

  const stepState = (round) => {
    if (cupComplete || round < currentRound) return "done";
    if (round === currentRound && isLive) return "current";
    return "upcoming";
  };

  const pair = simSeriesTarget
    ? `${teamFor(simSeriesTarget.team_high_id)?.abbrev}–${teamFor(simSeriesTarget.team_low_id)?.abbrev}`
    : "";

  return (
    <div className="nhlcal-root po-hub-root register-ops" data-register="ops">
      <style>{PO_HUB_CSS}</style>

      <aside className="nhlcal-sidebar">
        <button type="button" className="nhlcal-brand-button po-brand" onClick={onBack} aria-label="Return to hub">
          <TrophyGlyph size={24} />
        </button>
        <nav className="nhlcal-side-nav">
          <SideNavButton active icon="⌘" label="Bracket" />
          <SideNavButton icon="⌂" label="Hub" onClick={onBack} />
        </nav>
      </aside>

      <main className="nhlcal-main">
        <header className="po-top">
          <div className="po-top-title">
            <p className="po-eyebrow">{eyebrow}</p>
            <h1 className="po-title">{title}</h1>
          </div>

          <ol className="po-steps" aria-label="Playoff rounds">
            {[1, 2, 3, 4].map((r) => {
              const st = stepState(r);
              return (
                <li key={r} className={cx("po-step", `is-${st}`)}>
                  <span className="po-step-mark" aria-hidden="true">
                    {st === "done" ? "✓" : r}
                  </span>
                  {ROUND_SHORT[r]}
                </li>
              );
            })}
          </ol>

          {userChip ? <span className={cx("po-chip", `is-${userChip.tone}`)}>{userChip.text}</span> : null}

          <button type="button" className="po-btn po-btn--ghost" onClick={onBack}>
            ← Hub
          </button>
        </header>

        {notice ? (
          <div className={cx("po-notice", `is-${notice.tone}`)} role={notice.tone === "error" ? "alert" : "status"}>
            <span className="po-notice-text">{notice.text}</span>
            {notice.seriesId && userActive && notice.seriesId === userActive.series_id && canAct ? (
              <button
                type="button"
                className="po-btn po-btn--primary po-btn--sm"
                disabled={busy}
                onClick={() =>
                  runAction("play_user_game", { series_id: userActive.series_id }, "Playing your game…")
                }
              >
                Play Game
              </button>
            ) : null}
            <button
              type="button"
              className="po-btn po-btn--ghost po-btn--icon"
              aria-label="Dismiss"
              onClick={() => setNotice(null)}
            >
              ×
            </button>
          </div>
        ) : null}

        <section className="po-body">
          <div className="po-bracket-panel">
            {bracket.structured ? (
              <div className="po-bracket">
                {renderSideColumns("West")}
                {renderCupColumn()}
                {renderSideColumns("East")}
              </div>
            ) : (
              <div className="po-bracket is-flat">
                {bracket.rounds.map((rows, i) => (
                  <div key={i} className="po-col">
                    <header className={roundHeadClass(i + 1)}>{roundName(i + 1)}</header>
                    <div className="po-col-body po-flat-stack">
                      {rows.length ? (
                        rows.map((s) => <div key={s.series_id}>{renderCard(s, null)}</div>)
                      ) : (
                        <div className="po-empty-slot">Awaiting series</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <SeriesDesk
            series={selected}
            high={teamFor(selected?.team_high_id)}
            low={teamFor(selected?.team_low_id)}
            state={seriesState(selected, isLive)}
            canAct={canAct}
            busy={busy}
            playoffDay={playoffDay}
            onClear={() => setSelectedId(null)}
            onAction={runAction}
          />
        </section>

        <footer className="po-foot">
          {cupComplete ? (
            <button type="button" className="po-btn po-btn--primary" disabled={busy} onClick={handleContinue}>
              Continue to Awards
            </button>
          ) : !isLive ? (
            <>
              <button type="button" className="po-btn po-btn--primary" disabled={busy} onClick={handleEnter}>
                Start Round 1
              </button>
              {!userCurrent ? (
                <button type="button" className="po-btn po-btn--secondary" disabled={busy} onClick={handleFastForward}>
                  Sim Entire Playoffs
                </button>
              ) : null}
            </>
          ) : (
            <>
              {userGameTonight ? (
                <button
                  type="button"
                  className="po-btn po-btn--primary"
                  disabled={busy}
                  onClick={() =>
                    runAction("play_user_game", { series_id: userActive.series_id }, "Playing your game…")
                  }
                >
                  Play Game {nextGameNumber(userActive)}
                </button>
              ) : null}
              <button
                type="button"
                className={cx("po-btn", userGameTonight ? "po-btn--secondary" : "po-btn--primary")}
                disabled={busy || userGameTonight}
                title={userGameTonight ? "Your game is tonight — play or sim it first" : "Play tonight's slate, then advance one day"}
                onClick={() => runAction("advance_day", {}, "Simming tonight's slate…")}
              >
                Sim Day
              </button>
              <button
                type="button"
                className="po-btn po-btn--secondary"
                disabled={busy || !simSeriesTarget}
                title="Finish the selected series (or the next active series)"
                onClick={handleSimSeries}
              >
                {pair ? `Sim ${pair}` : "Sim Series"}
              </button>
              <button
                type="button"
                className="po-btn po-btn--secondary"
                disabled={busy || userGameTonight}
                title={
                  userAlive
                    ? "Sim day by day until your club's next game"
                    : "Sim day by day through the Stanley Cup Final"
                }
                onClick={handleFastForward}
              >
                {userAlive ? "Sim to My Next Game" : "Sim to Cup"}
              </button>
            </>
          )}

          {busy ? (
            <span className="po-foot-status" role="status">
              <i className="po-spinner" aria-hidden="true" />
              {busyLabel || "Working…"}
              {fastForwarding ? (
                <button
                  type="button"
                  className="po-btn po-btn--ghost po-btn--sm"
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
      </main>
    </div>
  );
}

/* ───────────────────────── styles (scoped) ───────────────────────── */

const PO_HUB_CSS = `
.po-start-menu-host { height: 100%; min-height: 0; width: 100%; overflow: hidden; }

.po-hub-root {
  --po-display: var(--font-broadcast-display, "Archivo Black", "Arial Black", sans-serif);
  --po-ui: var(--font-broadcast-ui, var(--font-ui, "Barlow Semi Condensed", "Inter", "Segoe UI", system-ui, sans-serif));
  --po-gold: var(--gold, #e9a83c);
  --po-cyan: var(--cyan, #13d8e7);
  --po-text: var(--text, #e8f1f6);
  --po-muted: var(--muted, #8aa3b3);
  --po-line: var(--line, rgba(120, 180, 210, 0.16));
  --po-line-2: var(--line-2, rgba(120, 180, 210, 0.3));
  --po-wire: rgba(19, 216, 231, 0.34);
  --po-radius: var(--radius-card, 8px);
  --po-radius-sm: var(--radius-control, 6px);
  --po-panel: var(--panel, rgba(6, 20, 32, 0.96));
  height: 100%;
  min-height: 0;
  max-height: 100%;
  overflow: hidden;
  font-family: var(--po-ui);
  color: var(--po-text);
}
.po-hub-root .nhlcal-main {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}
.po-hub-root .po-brand { display: grid; place-items: center; color: var(--po-gold); }

/* ── type primitives ── */
.po-hub-root .po-eyebrow {
  margin: 0;
  font: 800 10px/1.2 var(--po-ui);
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--po-gold);
}

/* ── buttons: one system for the whole screen ── */
.po-hub-root .po-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 34px;
  padding: 0 14px;
  border: 1px solid transparent;
  border-radius: var(--po-radius-sm);
  font: 800 11px/1 var(--po-ui);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.po-hub-root .po-btn:focus-visible { outline: 2px solid var(--po-cyan); outline-offset: 2px; }
.po-hub-root .po-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.po-hub-root .po-btn--primary { background: var(--po-gold); border-color: var(--po-gold); color: #1a1204; }
.po-hub-root .po-btn--primary:not(:disabled):hover { background: #f4bb57; }
.po-hub-root .po-btn--secondary { background: rgba(19, 216, 231, 0.07); border-color: rgba(19, 216, 231, 0.42); color: var(--po-cyan); }
.po-hub-root .po-btn--secondary:not(:disabled):hover { background: rgba(19, 216, 231, 0.14); }
.po-hub-root .po-btn--ghost { background: transparent; border-color: var(--po-line-2); color: var(--po-text); }
.po-hub-root .po-btn--ghost:not(:disabled):hover { border-color: var(--po-cyan); color: var(--po-cyan); }
.po-hub-root .po-btn--sm { height: 26px; padding: 0 10px; font-size: 10px; }
.po-hub-root .po-btn--icon { width: 28px; height: 28px; padding: 0; font-size: 16px; letter-spacing: 0; }

/* ── top bar ── */
.po-hub-root .po-top {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 16px;
  padding: 12px 16px 10px;
  border-bottom: 1px solid var(--po-line);
  flex-shrink: 0;
}
.po-hub-root .po-top-title { flex: 1 1 240px; min-width: 0; }
.po-hub-root .po-title {
  margin: 4px 0 0;
  font-family: var(--po-display);
  font-weight: 400;
  font-size: clamp(18px, 1.6vw, 24px);
  letter-spacing: 0.04em;
  line-height: 1.05;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.po-hub-root .po-steps { display: flex; gap: 4px; margin: 0; padding: 0; list-style: none; }
.po-hub-root .po-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 10px 0 4px;
  border: 1px solid var(--po-line);
  border-radius: 999px;
  font: 800 10px/1 var(--po-ui);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--po-muted);
}
.po-hub-root .po-step-mark {
  display: inline-grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.05);
  font-size: 10px;
}
.po-hub-root .po-step.is-done { color: var(--po-text); border-color: var(--po-line-2); }
.po-hub-root .po-step.is-done .po-step-mark { background: rgba(19, 216, 231, 0.18); color: var(--po-cyan); }
.po-hub-root .po-step.is-current { color: var(--po-gold); border-color: rgba(233, 168, 60, 0.6); background: rgba(233, 168, 60, 0.08); }
.po-hub-root .po-step.is-current .po-step-mark { background: var(--po-gold); color: #1a1204; }
.po-hub-root .po-chip {
  display: inline-flex;
  align-items: center;
  height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  border: 1px solid var(--po-line-2);
  font: 800 10.5px/1 var(--po-ui);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.po-hub-root .po-chip.is-gold { color: var(--po-gold); border-color: rgba(233, 168, 60, 0.6); }
.po-hub-root .po-chip.is-cyan { color: var(--po-cyan); border-color: rgba(19, 216, 231, 0.45); }
.po-hub-root .po-chip.is-muted { color: var(--po-muted); }

/* ── notice ── */
.po-hub-root .po-notice {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 10px 16px 0;
  padding: 6px 6px 6px 12px;
  border: 1px solid;
  border-radius: var(--po-radius);
  font: 700 12px/1.3 var(--po-ui);
  flex-shrink: 0;
}
.po-hub-root .po-notice-text { flex: 1; min-width: 0; }
.po-hub-root .po-notice.is-info { border-color: rgba(233, 168, 60, 0.5); background: rgba(233, 168, 60, 0.08); color: var(--po-text); }
.po-hub-root .po-notice.is-error { border-color: rgba(255, 96, 96, 0.5); background: rgba(255, 80, 80, 0.08); color: #ffc2c2; }

/* ── body layout ── */
.po-hub-root .po-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 12px;
  padding: 12px 16px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.po-hub-root .po-bracket-panel {
  display: flex;
  min-height: 0;
  min-width: 0;
  padding: 10px 12px 12px;
  border: 1px solid var(--po-line);
  border-radius: var(--radius-panel, 10px);
  background: linear-gradient(180deg, rgba(8, 26, 40, 0.92), rgba(4, 14, 22, 0.96));
  overflow: auto;
}

/* ── bracket grid ── */
.po-hub-root .po-bracket {
  --g: 18px;
  display: grid;
  grid-template-columns: repeat(3, minmax(108px, 1fr)) minmax(124px, 1.08fr) repeat(3, minmax(108px, 1fr));
  column-gap: var(--g);
  width: 100%;
  min-height: 440px;
}
.po-hub-root .po-bracket.is-flat { grid-template-columns: repeat(4, minmax(128px, 1fr)); }
.po-hub-root .po-col { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.po-hub-root .po-col-head {
  padding-bottom: 6px;
  margin-bottom: 2px;
  border-bottom: 1px solid var(--po-line);
  font: 800 10px/1.2 var(--po-ui);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  text-align: center;
  color: var(--po-muted);
  white-space: nowrap;
}
.po-hub-root .po-col-head.is-current { color: var(--po-cyan); border-bottom-color: rgba(19, 216, 231, 0.5); }
.po-hub-root .po-col.is-cup .po-col-head { color: var(--po-gold); }
.po-hub-root .po-col-body { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.po-hub-root .po-pair { position: relative; flex: 1; display: flex; flex-direction: column; min-height: 0; }
.po-hub-root .po-slot { position: relative; flex: 1; display: flex; align-items: center; min-height: 100px; }
.po-hub-root .po-flat-stack { gap: 8px; justify-content: space-around; padding-top: 6px; }

/* connectors: pair bracket joins two card centres (25% / 75%), stubs carry it on at 50% */
.po-hub-root .po-pair::after {
  content: "";
  position: absolute;
  top: 25%;
  bottom: 25%;
  width: calc(var(--g) / 2);
  border: 2px solid var(--po-wire);
  pointer-events: none;
}
.po-hub-root .po-col.is-west .po-pair::after { right: calc(var(--g) / -2); border-left: 0; border-radius: 0 6px 6px 0; }
.po-hub-root .po-col.is-east .po-pair::after { left: calc(var(--g) / -2); border-right: 0; border-radius: 6px 0 0 6px; }
.po-hub-root .po-slot.has-in::before,
.po-hub-root .po-slot.has-out::after,
.po-hub-root .po-col.is-cup .po-slot::after {
  content: "";
  position: absolute;
  top: 50%;
  width: calc(var(--g) / 2);
  border-top: 2px solid var(--po-wire);
  transform: translateY(-1px);
  pointer-events: none;
}
.po-hub-root .po-col.is-west .po-slot.has-in::before { left: calc(var(--g) / -2); }
.po-hub-root .po-col.is-east .po-slot.has-in::before { right: calc(var(--g) / -2); }
.po-hub-root .po-col.is-west .po-slot.has-out::after { right: calc(var(--g) / -2); }
.po-hub-root .po-col.is-east .po-slot.has-out::after { left: calc(var(--g) / -2); }
.po-hub-root .po-col.is-cup .po-slot::before { left: calc(var(--g) / -2); }
.po-hub-root .po-col.is-cup .po-slot::after { right: calc(var(--g) / -2); }

/* cup column */
.po-hub-root .po-cup-wrap { position: relative; width: 100%; }
.po-hub-root .po-cup-crest {
  position: absolute;
  bottom: calc(100% + 10px);
  left: 50%;
  transform: translateX(-50%);
  color: var(--po-gold);
  opacity: 0.9;
}
.po-hub-root .po-cup-champ {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  right: 0;
  text-align: center;
  font-family: var(--po-display);
  font-size: 12px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--po-gold);
}
.po-hub-root .po-col.is-cup .po-card { border-color: rgba(233, 168, 60, 0.4); }

/* ── series card ── */
.po-hub-root .po-card {
  position: relative;
  display: grid;
  gap: 1px;
  width: 100%;
  padding: 6px 8px 7px;
  text-align: left;
  border: 1px solid var(--po-line);
  border-radius: var(--po-radius);
  background: rgba(10, 30, 45, 0.96);
  color: var(--po-text);
  font-family: var(--po-ui);
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease;
}
.po-hub-root .po-card:not(:disabled):hover { border-color: var(--po-line-2); background: rgba(14, 38, 56, 0.98); }
.po-hub-root .po-card:focus-visible { outline: 2px solid var(--po-cyan); outline-offset: 2px; }
.po-hub-root .po-card.is-selected { border-color: var(--po-cyan); box-shadow: inset 0 0 0 1px var(--po-cyan); }
.po-hub-root .po-card.is-user { border-left: 3px solid var(--po-gold); padding-left: 6px; }
.po-hub-root .po-card.is-pending { background: transparent; border-style: dashed; cursor: default; }
.po-hub-root .po-card.is-complete { background: rgba(8, 22, 34, 0.92); }
.po-hub-root .po-card.is-just-set { animation: poSet 0.85s ease; }
@keyframes poSet {
  0% { border-color: var(--po-cyan); background: rgba(19, 216, 231, 0.14); }
  100% { background: rgba(10, 30, 45, 0.96); }
}
.po-hub-root .po-card-meta {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 2px;
  font: 800 9.5px/1.3 var(--po-ui);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--po-muted);
  white-space: nowrap;
}
.po-hub-root .po-card-meta-left { display: inline-flex; align-items: center; gap: 5px; min-width: 0; overflow: hidden; }
.po-hub-root .po-card.is-tonight .po-card-meta-left { color: var(--po-gold); }
.po-hub-root .po-you {
  padding: 1px 4px;
  border-radius: 3px;
  background: var(--po-gold);
  color: #1a1204;
  font-size: 9px;
}
.po-hub-root .po-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--po-gold); flex-shrink: 0; }
.po-hub-root .po-row {
  display: grid;
  grid-template-columns: 14px 22px minmax(0, 1fr) auto;
  gap: 6px;
  align-items: center;
  min-height: 26px;
}
.po-hub-root .po-seed {
  font: 800 10px/1 var(--po-ui);
  color: var(--po-muted);
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.po-hub-root .po-row-name { display: flex; flex-direction: column; min-width: 0; line-height: 1.15; }
.po-hub-root .po-row-name strong {
  font: 800 12.5px/1.15 var(--po-ui);
  letter-spacing: 0.03em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.po-hub-root .po-row-name small { font: 600 10px/1.2 var(--po-ui); color: var(--po-muted); font-variant-numeric: tabular-nums; }
.po-hub-root .po-row-wins {
  min-width: 12px;
  font: 800 15px/1 var(--po-ui);
  font-style: normal;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--po-muted);
}
.po-hub-root .po-row.is-lead strong,
.po-hub-root .po-row.is-lead .po-row-wins { color: var(--po-cyan); }
.po-hub-root .po-row.is-winner strong,
.po-hub-root .po-row.is-winner .po-row-wins { color: var(--po-gold); }
.po-hub-root .po-row.is-out { opacity: 0.4; }
.po-hub-root .po-row.is-empty strong { font: 700 10.5px/1.2 var(--po-ui); letter-spacing: 0.04em; color: var(--po-muted); }
.po-hub-root .po-card-foot {
  margin-top: 3px;
  padding-top: 4px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  font: 700 10.5px/1.2 var(--po-ui);
  color: var(--po-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.po-hub-root .po-mark {
  display: inline-grid;
  place-items: center;
  flex-shrink: 0;
  overflow: hidden;
  border-radius: 5px;
  background: rgba(255, 255, 255, 0.04);
}
.po-hub-root .po-mark img { width: 100%; height: 100%; object-fit: contain; }
.po-hub-root .po-mark.is-empty { border: 1px dashed var(--po-line-2); background: transparent; }
.po-hub-root .po-mark.is-dimmed { filter: grayscale(0.85); }
.po-hub-root .po-mark-fallback { font: 900 10px/1 var(--po-ui); }
.po-hub-root .po-empty-slot {
  display: grid;
  place-items: center;
  min-height: 64px;
  border: 1px dashed var(--po-line);
  border-radius: var(--po-radius);
  font: 700 11px/1 var(--po-ui);
  color: var(--po-muted);
}

/* ── series desk ── */
.po-hub-root .po-desk {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
  padding: 12px;
  border: 1px solid var(--po-line);
  border-radius: var(--radius-panel, 10px);
  background: var(--po-panel);
  overflow: auto;
}
.po-hub-root .po-desk-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.po-hub-root .po-desk-title {
  margin: 4px 0 0;
  font-family: var(--po-display);
  font-weight: 400;
  font-size: 16px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.po-hub-root .po-desk-muted { margin: 2px 0 0; font: 600 11.5px/1.4 var(--po-ui); color: var(--po-muted); }
.po-hub-root .po-desk-status { margin: 0; font: 800 13px/1.3 var(--po-ui); }
.po-hub-root .po-desk-score {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 8px;
  padding: 10px 6px;
  border: 1px solid var(--po-line);
  border-radius: var(--po-radius);
  background: rgba(0, 0, 0, 0.18);
}
.po-hub-root .po-desk-team { display: grid; justify-items: center; gap: 3px; min-width: 0; text-align: center; }
.po-hub-root .po-desk-team strong { font: 800 13px/1.1 var(--po-ui); letter-spacing: 0.03em; }
.po-hub-root .po-desk-team small { font: 600 10.5px/1.2 var(--po-ui); color: var(--po-muted); font-variant-numeric: tabular-nums; }
.po-hub-root .po-desk-wins { display: flex; align-items: baseline; gap: 6px; font: 800 28px/1 var(--po-ui); font-variant-numeric: tabular-nums; }
.po-hub-root .po-desk-wins .is-lead { color: var(--po-cyan); }
.po-hub-root .po-desk-sep { font-size: 18px; color: var(--po-muted); }
.po-hub-root .po-facts { display: flex; flex-wrap: wrap; gap: 6px; }
.po-hub-root .po-fact {
  padding: 3px 7px;
  border: 1px solid var(--po-line);
  border-radius: 4px;
  font: 700 10px/1.2 var(--po-ui);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--po-muted);
}
.po-hub-root .po-fact b { color: var(--po-text); font-weight: 800; }
.po-hub-root .po-fact.is-gold { color: var(--po-gold); border-color: rgba(233, 168, 60, 0.5); }
.po-hub-root .po-games { display: grid; gap: 4px; }
.po-hub-root .po-game {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
  height: 30px;
  padding: 0 8px;
  border: 1px solid var(--po-line);
  border-radius: var(--po-radius-sm);
  background: rgba(0, 0, 0, 0.16);
  font: 700 11.5px/1 var(--po-ui);
  font-variant-numeric: tabular-nums;
}
.po-hub-root .po-game-n { font-weight: 800; color: var(--po-muted); }
.po-hub-root .po-game-line { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--po-muted); }
.po-hub-root .po-game-line b { font-weight: 700; color: var(--po-text); }
.po-hub-root .po-game-line b.w { font-weight: 800; color: var(--po-cyan); }
.po-hub-root .po-game-tag { font: 800 9.5px/1 var(--po-ui); letter-spacing: 0.1em; text-transform: uppercase; color: var(--po-muted); }
.po-hub-root .po-game.is-ot .po-game-tag,
.po-hub-root .po-game.is-tonight .po-game-tag { color: var(--po-gold); }
.po-hub-root .po-game.is-tonight { border-color: rgba(233, 168, 60, 0.55); }
.po-hub-root .po-game.is-moot { opacity: 0.35; }
.po-hub-root .po-desk-actions { display: grid; gap: 6px; margin-top: auto; }

/* ── footer ── */
.po-hub-root .po-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-top: 1px solid var(--po-line);
  background: rgba(4, 16, 26, 0.94);
  flex-shrink: 0;
}
.po-hub-root .po-foot-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
  font: 800 10.5px/1 var(--po-ui);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--po-cyan);
}
.po-hub-root .po-spinner {
  width: 12px;
  height: 12px;
  border: 2px solid rgba(19, 216, 231, 0.25);
  border-top-color: var(--po-cyan);
  border-radius: 50%;
  animation: poSpin 0.8s linear infinite;
}
@keyframes poSpin { to { transform: rotate(360deg); } }

/* ── responsive ── */
@media (max-width: 1500px) {
  .po-hub-root .po-body { grid-template-columns: minmax(0, 1fr) 264px; }
}
@media (max-width: 1100px) {
  .po-hub-root .po-body { grid-template-columns: 1fr; overflow: auto; }
  .po-hub-root .po-steps { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .po-hub-root .po-card,
  .po-hub-root .po-btn { transition: none; }
  .po-hub-root .po-card.is-just-set,
  .po-hub-root .po-spinner { animation: none; }
}
`;
