import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  resolveFranchiseTeamLogo,
  toLogoUrl,
} from "../../utils/teamLogos";
import { enterPlayoffs, playoffAction, continueOffseason } from "../../services/franchiseService";
import { isNetworkError, formatFranchiseApiError, isTimeoutError } from "../../services/api";
import { useGameUI } from "../../game/GameUIContext";
import "../../styles/nhlcalShell.css";

/**
 * Playable Stanley Cup bracket hub — CalendarScreen (nhlcal) visual language.
 * West R1 → R2 → Final → Cup ← East Final ← R2 ← R1
 */

function firstDefined(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
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

function buildTeamLookup(franchiseState = {}, playoffData = {}) {
  const map = new Map();
  const pools = [
    playoffData?.playoff_teams,
    playoffData?.teams,
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
  const wins = Number(firstDefined(row.w, row.wins, row.record?.w, row.record?.wins));
  const losses = Number(firstDefined(row.l, row.losses, row.record?.l, row.record?.losses));
  const otl = Number(firstDefined(row.otl, row.ot, row.overtime_losses, row.record?.otl, 0));
  if (!Number.isFinite(wins) || !Number.isFinite(losses)) return "";
  return `${wins}-${losses}-${Number.isFinite(otl) ? otl : 0}`;
}

function resolveTeam(id, lookup, franchiseState) {
  const tid = String(id || "");
  const row = lookup.get(tid) || {};
  const name =
    firstDefined(row.full_name, row.name, row.team_name, row.abbrev, tid) || tid;
  const abbrev = firstDefined(
    row.abbrev,
    row.abbreviation,
    String(name).slice(0, 3).toUpperCase()
  );
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
    pts: firstDefined(row.pts, row.points),
    seed: row.seed || row.playoff_seed,
  };
}

function seriesLabel(series, highTeam, lowTeam) {
  const wh = Number(series?.wins_high || 0);
  const wl = Number(series?.wins_low || 0);
  const hi = highTeam?.abbrev || "Home";
  const lo = lowTeam?.abbrev || "Away";
  if (series?.status === "complete" || wh >= 4 || wl >= 4) {
    return wh > wl ? `${hi} wins ${wh}–${wl}` : `${lo} wins ${wl}–${wh}`;
  }
  if (wh === 0 && wl === 0) return "Series tied 0–0";
  if (wh === wl) return `Series tied ${wh}–${wl}`;
  return wh > wl ? `${hi} leads ${wh}–${wl}` : `${lo} leads ${wl}–${wh}`;
}

function roundTitle(round) {
  return (
    { 1: "Round 1", 2: "Round 2", 3: "Conf. Final", 4: "Stanley Cup Final" }[round] ||
    `Round ${round}`
  );
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

function emptySlot(conf, round, slot) {
  return {
    series_id: `${conf || "LEA"}-R${round}-${slot}`,
    round_index: round,
    conference: conf,
    bracket_slot: slot,
    team_high_id: "",
    team_low_id: "",
    wins_high: 0,
    wins_low: 0,
    status: "pending",
    game_log: [],
    is_user_series: false,
    next_game: 1,
  };
}

/** Preview bracket from playoff_ready payload (R1 filled, later rounds empty). */
function buildPreviewSeries(payload = {}, userId = "") {
  const r1src = payload.first_round || payload.matchups || payload.series || [];
  const rows = [];
  const byConf = {};

  (r1src || []).forEach((m, i) => {
    const conf = confKey(m);
    const hi = String(m.team_high_id || m.home_id || "");
    const lo = String(m.team_low_id || m.away_id || "");
    byConf[conf] = byConf[conf] || [];
    const slot = byConf[conf].length;
    const row = {
      ...m,
      series_id: m.series_id || `${conf}-R1-${slot}`,
      round_index: 1,
      conference: m.conference || conf,
      bracket_slot: slot,
      team_high_id: hi,
      team_low_id: lo,
      seed_high: m.seed_high || m.seedHigh,
      seed_low: m.seed_low || m.seedLow,
      wins_high: Number(m.wins_high || 0),
      wins_low: Number(m.wins_low || 0),
      status: "active",
      game_log: Array.isArray(m.game_log) ? m.game_log : [],
      is_user_series: Boolean(userId) && (userId === hi || userId === lo),
      next_game: 1,
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
  rows.push({
    ...emptySlot(null, 4, 0),
    series_id: "CUP-R4-0",
    conference: null,
  });
  return rows;
}

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

function TeamMark({ team, size = 34, dimmed }) {
  if (!team?.team_id) {
    return (
      <span className="po-hub-tbd" style={{ width: size, height: size }}>
        TBD
      </span>
    );
  }
  return (
    <span
      className={`po-hub-team-mark${dimmed ? " is-dimmed" : ""}`}
      style={{ width: size, height: size }}
    >
      {team.logo ? (
        <img src={team.logo} alt="" />
      ) : (
        <span className="po-hub-fallback">{String(team.abbrev || "?").slice(0, 3)}</span>
      )}
    </span>
  );
}

function BracketSeriesCard({
  series,
  highTeam,
  lowTeam,
  selected,
  onSelect,
  isUser,
  justSet,
  playoffDay = 0,
}) {
  const complete = series?.status === "complete";
  const pending = series?.status === "pending" || !series?.team_high_id;
  const loser = complete
    ? series.wins_high > series.wins_low
      ? lowTeam
      : highTeam
    : null;
  const leaderHigh = !complete && series?.wins_high > series?.wins_low;
  const leaderLow = !complete && series?.wins_low > series?.wins_high;
  const nextGame = Number(series?.next_game || (series?.game_log?.length || 0) + 1);
  const homeAbbr = isHighSeedHome(nextGame) ? highTeam?.abbrev : lowTeam?.abbrev;
  const games = Array.isArray(series?.game_log) ? series.game_log : [];
  const lastGame = games.length ? games[games.length - 1] : null;
  const scheduledDay = series?.scheduled_day;
  const dueTonight = !pending && Number(scheduledDay) === Number(playoffDay);
  const track = String(series?.schedule_track || "").toUpperCase();

  return (
    <button
      type="button"
      className={[
        "po-hub-series-card",
        selected ? "is-selected" : "",
        complete ? "is-complete" : "",
        pending ? "is-pending" : "",
        isUser ? "is-user" : "",
        series?.status === "active" ? "is-active" : "",
        dueTonight ? "is-tonight" : "",
        justSet ? "is-just-set" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={() => onSelect?.(series)}
      disabled={pending && !series?.team_high_id}
    >
      <div className="po-hub-series-meta">
        {isUser ? <span className="po-hub-user-tag">YOUR SERIES</span> : <span>{track || "Best of 7"}</span>}
        {!pending ? (
          <span className={`po-hub-home-tag${dueTonight ? " is-tonight" : ""}`}>
            {dueTonight ? "TONIGHT" : `Day ${Number(scheduledDay) + 1}`}
            {" · "}@{homeAbbr || "HOME"}
          </span>
        ) : (
          <span>Awaiting winner</span>
        )}
      </div>

      <div
        className={`po-hub-series-row${
          loser && loser.team_id === highTeam?.team_id ? " is-out" : ""
        }${leaderHigh ? " is-lead" : ""}${!pending ? " has-home" : ""}`}
      >
        <span className="po-hub-seed">{highTeam?.seed || series?.seed_high || "—"}</span>
        <TeamMark
          team={highTeam}
          size={28}
          dimmed={loser && loser.team_id === highTeam?.team_id}
        />
        <div className="po-hub-series-info">
          <strong>{highTeam?.abbrev || "TBD"}</strong>
          <small>{highTeam?.record || (pending ? "—" : "Playoffs")}</small>
        </div>
        <em>{pending ? "—" : series?.wins_high ?? 0}</em>
      </div>

      <div
        className={`po-hub-series-row${
          loser && loser.team_id === lowTeam?.team_id ? " is-out" : ""
        }${leaderLow ? " is-lead" : ""}`}
      >
        <span className="po-hub-seed">{lowTeam?.seed || series?.seed_low || "—"}</span>
        <TeamMark
          team={lowTeam}
          size={28}
          dimmed={loser && loser.team_id === lowTeam?.team_id}
        />
        <div className="po-hub-series-info">
          <strong>{lowTeam?.abbrev || "TBD"}</strong>
          <small>{lowTeam?.record || (pending ? "—" : "Playoffs")}</small>
        </div>
        <em>{pending ? "—" : series?.wins_low ?? 0}</em>
      </div>

      <div className="po-hub-series-scoreline">
        {pending ? "Winner advances →" : seriesLabel(series, highTeam, lowTeam)}
      </div>
      {lastGame ? (
        <div className="po-hub-series-last">
          G{lastGame.game}: {lastGame.home_score}–{lastGame.away_score}
          {lastGame.ot ? " OT" : ""}
        </div>
      ) : null}
    </button>
  );
}

function seriesAbbr(teamId, highTeam, lowTeam) {
  const id = String(teamId || "");
  if (id && id === String(highTeam?.team_id || "")) return highTeam?.abbrev || "H";
  if (id && id === String(lowTeam?.team_id || "")) return lowTeam?.abbrev || "A";
  return String(id).slice(0, 3).toUpperCase() || "—";
}

function SeriesSidePanel({
  series,
  highTeam,
  lowTeam,
  busy,
  isLive,
  playoffDay = 0,
  onClose,
  onAction,
  onEnter,
}) {
  if (!series) {
    return (
      <aside className="nhlcal-rail-panel po-hub-side">
        <div className="nhlcal-panel-head">
          <p>Series Desk</p>
          <h3>Select a series</h3>
        </div>
        <p className="po-hub-muted">
          Click a Round 1 matchup to open the series score, seven-game slate, and play/sim controls.
        </p>
      </aside>
    );
  }

  const games = Array.isArray(series.game_log) ? series.game_log : [];
  const slots = Array.from({ length: 7 }, (_, i) => {
    const g = games.find((x) => Number(x.game) === i + 1);
    const date = Array.isArray(series.schedule_dates) ? series.schedule_dates[i] : null;
    return g || { game: i + 1, empty: true, scheduled_day: date };
  });
  const seriesActive = series.status === "active";
  const complete = series.status === "complete";
  const pending = series.status === "pending" || !series.team_high_id;
  const nextGame = Number(series.next_game || games.length + 1);
  const homeAbbr = isHighSeedHome(nextGame) ? highTeam?.abbrev : lowTeam?.abbrev;
  const awayAbbr = isHighSeedHome(nextGame) ? lowTeam?.abbrev : highTeam?.abbrev;
  const dueTonight = seriesActive && Number(series.scheduled_day) === Number(playoffDay);

  return (
    <aside className="nhlcal-rail-panel po-hub-side">
      <div className="nhlcal-panel-head">
        <p>
          {roundTitle(series.round_index)}
          {series.round_index === 4 ? "" : ` · ${confKey(series)}`}
          {series.is_user_series ? " · Your club" : ""}
        </p>
        <h3>
          {highTeam?.abbrev || "TBD"} vs {lowTeam?.abbrev || "TBD"}
        </h3>
        <button type="button" className="nhlcal-quick-link" onClick={() => onClose?.()}>
          Clear
        </button>
      </div>

      <div className="po-hub-side-score">
        <div>
          <TeamMark team={highTeam} size={42} />
          <strong>{highTeam?.abbrev || "TBD"}</strong>
          <small>{highTeam?.record || "—"}</small>
          <em>{pending ? "—" : series.wins_high ?? 0}</em>
        </div>
        <span>SERIES</span>
        <div>
          <TeamMark team={lowTeam} size={42} />
          <strong>{lowTeam?.abbrev || "TBD"}</strong>
          <small>{lowTeam?.record || "—"}</small>
          <em>{pending ? "—" : series.wins_low ?? 0}</em>
        </div>
      </div>

      <p className="po-hub-side-status">
        {pending ? "Waiting for previous round" : seriesLabel(series, highTeam, lowTeam)}
      </p>
      {!pending ? (
        <p className="po-hub-side-next">
          {complete
            ? "Series complete — winner advances"
            : dueTonight
              ? `TONIGHT · Game ${nextGame} at ${homeAbbr || "HOME"} (${awayAbbr || "AWAY"} visits)`
              : `Next tip Day ${Number(series.scheduled_day || 0) + 1} · Game ${nextGame} at ${homeAbbr || "HOME"}`}
        </p>
      ) : null}

      <div className="po-hub-game-slots">
        {slots.map((g) => {
          const homeIsNext = isHighSeedHome(g.game);
          const scheduledHome = homeIsNext ? highTeam?.abbrev : lowTeam?.abbrev;
          const scheduledAway = homeIsNext ? lowTeam?.abbrev : highTeam?.abbrev;
          const tipDay =
            g.scheduled_day != null
              ? Number(g.scheduled_day)
              : Array.isArray(series.schedule_dates)
                ? series.schedule_dates[g.game - 1]
                : null;
          return (
            <div
              key={g.game}
              className={`po-hub-game-slot${g.empty ? " is-empty" : ""}${g.ot ? " is-ot" : ""}${
                !g.empty && Number(g.game) === games.length ? " is-latest" : ""
              }`}
            >
              <span>G{g.game}</span>
              {g.empty ? (
                <strong className="po-hub-game-pending">
                  {pending
                    ? "—"
                    : `${scheduledAway || "A"} @ ${scheduledHome || "H"}${
                        tipDay != null ? ` · D${Number(tipDay) + 1}` : ""
                      }`}
                </strong>
              ) : (
                <strong>
                  {seriesAbbr(g.home_id, highTeam, lowTeam)} {g.home_score}–{g.away_score}{" "}
                  {seriesAbbr(g.away_id, highTeam, lowTeam)}
                  {g.ot ? " OT" : ""}
                </strong>
              )}
              <small>
                {g.empty
                  ? tipDay != null && Number(tipDay) === Number(playoffDay)
                    ? "Tonight"
                    : "Unplayed"
                  : "Final"}
              </small>
            </div>
          );
        })}
      </div>

      <div className="po-hub-side-actions">
        {!isLive ? (
          <button
            type="button"
            className="nhlcal-advance-button"
            disabled={busy}
            onClick={onEnter}
          >
            Start Round 1
          </button>
        ) : (
          <>
            {series.is_user_series ? (
              <button
                type="button"
                className="nhlcal-advance-button"
                disabled={!seriesActive || busy}
                title="Play the next game in this series"
                onClick={() => onAction("play_user_game", { series_id: series.series_id })}
              >
                Play Game
              </button>
            ) : null}
            <button
              type="button"
              className={
                series.is_user_series
                  ? "nhlcal-advance-button-secondary"
                  : "nhlcal-advance-button"
              }
              disabled={!seriesActive || busy}
              title="Finish this series"
              onClick={() => onAction("sim_series", { series_id: series.series_id })}
            >
              Sim Series
            </button>
          </>
        )}
      </div>
    </aside>
  );
}

export default function PlayoffStartMenu({
  franchiseState = {},
  playoffData = {},
  onEnterPlayoffs,
  onBack,
}) {
  const { mergeFranchiseState, setFranchiseState, openFranchiseEvent } = useGameUI() || {};
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [justSetIds, setJustSetIds] = useState(() => new Set());
  const prevTeamsRef = useRef(new Map());
  const autoEnterRef = useRef(false);
  const autoOpenAwardsRef = useRef(false);

  const phase = String(franchiseState?.season_phase || franchiseState?.phase || "").toLowerCase();
  const live = franchiseState?.playoff_live || playoffData?.live_state || null;
  const isLive = Boolean(live?.started) && (phase === "playoffs" || phase === "playoff_ready");
  const uid = userTeamId(franchiseState);
  const payload = playoffData || franchiseState?.playoff_payload || {};

  const lookup = useMemo(
    () => buildTeamLookup(franchiseState, payload),
    [franchiseState, payload]
  );

  const seriesList = useMemo(() => {
    if (isLive && Array.isArray(live?.series) && live.series.length) {
      return live.series.map((s) => ({
        ...s,
        is_user_series:
          Boolean(s.is_user_series) ||
          (uid &&
            (String(s.team_high_id) === uid || String(s.team_low_id) === uid)),
      }));
    }
    return buildPreviewSeries(payload, uid);
  }, [isLive, live, payload, uid]);

  const selected = seriesList.find((s) => s.series_id === selectedId) || null;

  const teamFor = useCallback(
    (id) => resolveTeam(id, lookup, franchiseState),
    [lookup, franchiseState]
  );

  const bySlot = (a, b) => Number(a.bracket_slot || 0) - Number(b.bracket_slot || 0);

  const west = useMemo(
    () => ({
      r1: seriesList.filter((s) => Number(s.round_index) === 1 && confKey(s) === "West").sort(bySlot),
      r2: seriesList.filter((s) => Number(s.round_index) === 2 && confKey(s) === "West").sort(bySlot),
      r3: seriesList.filter((s) => Number(s.round_index) === 3 && confKey(s) === "West").sort(bySlot),
    }),
    [seriesList]
  );

  const east = useMemo(
    () => ({
      r1: seriesList.filter((s) => Number(s.round_index) === 1 && confKey(s) === "East").sort(bySlot),
      r2: seriesList.filter((s) => Number(s.round_index) === 2 && confKey(s) === "East").sort(bySlot),
      r3: seriesList.filter((s) => Number(s.round_index) === 3 && confKey(s) === "East").sort(bySlot),
    }),
    [seriesList]
  );

  const flatR1 = seriesList.filter((s) => Number(s.round_index) === 1);
  const useFlat = !west.r1.length && !east.r1.length && flatR1.length;
  const cup = seriesList.find((s) => Number(s.round_index) === 4) || null;
  const championId = live?.champion_id || playoffData?.champion_id || franchiseState?.champion_id;
  const playoffDay = Number(live?.playoff_day || 0);
  const cupComplete =
    Boolean(championId) ||
    Boolean(live?.completed) ||
    phase === "post_cup" ||
    phase === "offseason" ||
    Boolean(franchiseState?.playoffs_done || franchiseState?.flags?.playoffs_done);
  const userSeriesActive = seriesList.find((s) => s.is_user_series && s.status === "active");
  const userSeriesDone = seriesList.find((s) => s.is_user_series && s.status === "complete");
  const userSeries = userSeriesActive || null;
  const userEliminated =
    Boolean(userSeriesDone) &&
    uid &&
    (String(userSeriesDone.loser_id || "") === uid ||
      (String(userSeriesDone.winner_id || "") &&
        String(userSeriesDone.winner_id) !== uid));

  const applyState = useCallback(
    (res) => {
      const state = res?.state;
      if (!state) return;
      if (typeof mergeFranchiseState === "function") mergeFranchiseState(state);
      else if (typeof setFranchiseState === "function") setFranchiseState(state);
    },
    [mergeFranchiseState, setFranchiseState]
  );

  const handoffToOffseason = useCallback(
    async (res) => {
      const st = res?.state || {};
      const finished =
        st.season_phase === "post_cup" ||
        st.phase === "post_cup" ||
        st.season_phase === "offseason" ||
        res?.result?.finish?.status === "post_cup";
      if (!finished) return;
      autoOpenAwardsRef.current = true;
      if (typeof openFranchiseEvent === "function") {
        openFranchiseEvent();
      }
    },
    [openFranchiseEvent]
  );

  const runEnter = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const res =
        typeof onEnterPlayoffs === "function" ? await onEnterPlayoffs() : await enterPlayoffs();
      applyState(res);
    } catch (e) {
      setError(formatFranchiseApiError(e) || e?.message || "Failed to start playoffs");
    } finally {
      setBusy(false);
    }
  }, [onEnterPlayoffs, applyState]);

  const runAction = useCallback(
    async (action, body = {}) => {
      setBusy(true);
      setError("");
      try {
        if (!isLive && !cupComplete) {
          await runEnter();
          if (action === "enter" || action === "start") return;
        }
        const res = await playoffAction(action, body);
        if (res?.result?.blocked || res?.blocked) {
          const reason =
            res?.result?.reason || res?.reason || "Your series tips tonight — sim your game first.";
          setError(reason);
          const blockId = res?.result?.series?.series_id || res?.series?.series_id;
          if (blockId) setSelectedId(blockId);
        }
        applyState(res);
        if (res?.result?.series?.series_id) {
          setSelectedId(res.result.series.series_id);
        }
        await handoffToOffseason(res);
        if (res?.result?.finish?.status === "post_cup" || res?.state?.season_phase === "post_cup") {
          setError("");
        }
        return res;
      } catch (e) {
        const detail =
          formatFranchiseApiError(e) ||
          e?.response?.data?.detail ||
          e?.message ||
          `Action failed: ${action}`;
        setError(typeof detail === "string" ? detail : JSON.stringify(detail));
        return null;
      } finally {
        setBusy(false);
      }
    },
    [isLive, cupComplete, runEnter, applyState, handoffToOffseason]
  );

  const runSimSeries = useCallback(async () => {
    const activeRows = seriesList.filter(
      (s) => s.status === "active" && s.team_high_id && s.team_low_id
    );
    const target =
      (selected?.status === "active" && selected?.team_high_id ? selected : null) ||
      (userSeries?.status === "active" ? userSeries : null) ||
      activeRows.find((s) => Number(s.scheduled_day) === playoffDay) ||
      activeRows[0] ||
      null;
    if (!target) {
      setError("No active series to sim — select a Round 1 matchup first.");
      return;
    }
    setSelectedId(target.series_id);
    await runAction("sim_series", { series_id: target.series_id });
  }, [seriesList, selected, userSeries, playoffDay, runAction]);

  const runFastForwardPlayoffs = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      if (!isLive && !cupComplete) {
        await runEnter();
      }
      const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
      let finished = Boolean(cupComplete);
      for (let i = 0; i < 220 && !finished; i += 1) {
        const res = await playoffAction("advance_day");
        applyState(res);
        const st = res?.state || {};
        finished =
          Boolean(st?.playoff_live?.completed) ||
          Boolean(st?.champion_id) ||
          st?.season_phase === "post_cup" ||
          st?.phase === "post_cup" ||
          res?.result?.finish?.status === "post_cup" ||
          Boolean(res?.result?.finish?.champion_id);
        if (finished) {
          await handoffToOffseason(res);
          break;
        }
        // Sped-up visual: pause briefly so the bracket can paint between days.
        await sleep(55);
      }
      if (!finished) {
        const res = await playoffAction("sim_rest");
        applyState(res);
        await handoffToOffseason(res);
      }
    } catch (e) {
      setError(
        formatFranchiseApiError(e) ||
          e?.response?.data?.detail ||
          e?.message ||
          "Sim Playoffs failed"
      );
    } finally {
      setBusy(false);
    }
  }, [isLive, cupComplete, runEnter, applyState, handoffToOffseason]);

  const runContinueOffseason = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      // If Cup not finished yet, finish remaining games first.
      if (!cupComplete && isLive) {
        const res = await playoffAction("sim_rest");
        applyState(res);
        await handoffToOffseason(res);
        if (res?.state?.season_phase === "post_cup" || res?.result?.finish?.status === "post_cup") {
          return;
        }
      }
      const res = await continueOffseason({
        from_stage: String(
          /* post-cup / awards handoff into offseason timeline */
          "awards"
        ),
      });
      applyState(res);
      if (typeof openFranchiseEvent === "function") openFranchiseEvent();
    } catch (e) {
      setError(
        isTimeoutError(e) || isNetworkError(e)
          ? formatFranchiseApiError(e)
          : formatFranchiseApiError(e) || e?.message || "Offseason continue failed"
      );
    } finally {
      setBusy(false);
    }
  }, [cupComplete, isLive, applyState, handoffToOffseason, openFranchiseEvent]);

  // Land on the live bracket immediately from playoff_ready.
  useEffect(() => {
    if (autoEnterRef.current) return;
    if (phase !== "playoff_ready" || isLive || busy) return;
    autoEnterRef.current = true;
    runEnter();
  }, [phase, isLive, busy, runEnter]);

  useEffect(() => {
    if (isLive && live?.intro_seen === false) {
      playoffAction("mark_intro_seen").catch(() => {});
    }
  }, [isLive, live?.intro_seen]);

  const autoSelectRef = useRef(false);

  useEffect(() => {
    if (autoSelectRef.current) return;
    const prefer = seriesList.find((s) => s.is_user_series && s.status !== "pending");
    if (!prefer) return;
    autoSelectRef.current = true;
    setSelectedId(prefer.series_id);
  }, [seriesList]);

  useEffect(() => {
    const prev = prevTeamsRef.current;
    const next = new Map();
    const gained = [];
    for (const s of seriesList) {
      const key = `${s.team_high_id || ""}|${s.team_low_id || ""}`;
      next.set(s.series_id, key);
      const old = prev.get(s.series_id);
      if (old !== undefined && old !== key && s.team_high_id && s.team_low_id) {
        gained.push(s.series_id);
      }
    }
    prevTeamsRef.current = next;
    if (!gained.length) return undefined;
    setJustSetIds(new Set(gained));
    const t = window.setTimeout(() => setJustSetIds(new Set()), 900);
    return () => window.clearTimeout(t);
  }, [seriesList]);

  const renderColumn = (title, rows, connector) => (
    <div className={`po-hub-col${connector ? ` connector-${connector}` : ""}`}>
      <header>{title}</header>
      <div className="po-hub-col-stack">
        {(rows || []).length ? (
          rows.map((s) => (
            <BracketSeriesCard
              key={s.series_id}
              series={s}
              highTeam={teamFor(s.team_high_id)}
              lowTeam={teamFor(s.team_low_id)}
              selected={selected?.series_id === s.series_id}
              isUser={Boolean(s.is_user_series)}
              justSet={justSetIds.has(s.series_id)}
              playoffDay={playoffDay}
              onSelect={(ser) => setSelectedId(ser.series_id)}
            />
          ))
        ) : (
          <div className="po-hub-empty-slot">Awaiting series</div>
        )}
      </div>
    </div>
  );

  return (
    <div className="nhlcal-root po-hub-root register-ops" data-register="ops">
      <style>{PO_HUB_CSS}</style>
      <aside className="nhlcal-sidebar">
        <button type="button" className="nhlcal-brand-button" onClick={onBack}>
          <span className="nhlcal-shield-icon" />
        </button>
        <nav className="nhlcal-side-nav">
          <SideNavButton active icon="⌘" label="Bracket" />
          <SideNavButton icon="⌂" label="Hub" onClick={onBack} />
        </nav>
      </aside>

      <main className="nhlcal-main">
        <section className="nhlcal-topbar po-hub-topbar">
          <div className="nhlcal-team-block">
            <p className="nhlcal-eyebrow">Stanley Cup Playoffs</p>
            {championId ? (
              <h1>{teamFor(championId).name} — Champions</h1>
            ) : cupComplete ? (
              <h1>Cup decided</h1>
            ) : null}
          </div>
          <div className="nhlcal-action-cluster">
            <button type="button" className="po-hub-hub-link" onClick={onBack}>
              ← Return to Hub
            </button>
          </div>
        </section>

        {error ? <p className="po-hub-error">{error}</p> : null}

        <section className="po-hub-layout">
          <div className="po-hub-bracket nhlcal-calendar-panel">
            {useFlat ? (
              <div className="po-hub-bracket-row po-hub-bracket-row-flat">
                {renderColumn("Round 1", flatR1, "right")}
                {renderColumn("Round 2", seriesList.filter((s) => Number(s.round_index) === 2), "both")}
                {renderColumn("Conf. Finals", seriesList.filter((s) => Number(s.round_index) === 3), "both")}
                {renderColumn("Stanley Cup", cup ? [cup] : [], "left")}
              </div>
            ) : (
              <div className="po-hub-bracket-row">
                {renderColumn("West R1", west.r1, "right")}
                {renderColumn("West R2", west.r2, "both")}
                {renderColumn("West Final", west.r3, "both")}
                {renderColumn("Stanley Cup Final", cup ? [cup] : [], "cup")}
                {renderColumn("East Final", east.r3, "both")}
                {renderColumn("East R2", east.r2, "both")}
                {renderColumn("East R1", east.r1, "left")}
              </div>
            )}
          </div>

          <SeriesSidePanel
            series={selected}
            highTeam={teamFor(selected?.team_high_id)}
            lowTeam={teamFor(selected?.team_low_id)}
            busy={busy}
            isLive={isLive}
            playoffDay={playoffDay}
            onClose={() => setSelectedId(null)}
            onAction={runAction}
            onEnter={runEnter}
          />
        </section>

        <footer className="po-hub-actionbar">
          {cupComplete ? (
            <>
              <button
                type="button"
                className="nhlcal-advance-button"
                disabled={busy}
                onClick={runContinueOffseason}
              >
                Continue to Awards / Offseason
              </button>
              <button
                type="button"
                className="po-hub-hub-link"
                disabled={busy}
                onClick={onBack}
              >
                ← Return to Hub
              </button>
            </>
          ) : !isLive ? (
            <button
              type="button"
              className="nhlcal-advance-button"
              disabled={busy}
              onClick={runEnter}
            >
              Start Round 1
            </button>
          ) : (
            <>
              <button
                type="button"
                className="nhlcal-advance-button"
                disabled={busy}
                title="Play all games scheduled for tonight, then advance one day"
                onClick={() => runAction("advance_day")}
              >
                <span className="po-hub-action-icon" aria-hidden>
                  ▶
                </span>
                Sim Day
              </button>
              <button
                type="button"
                className="nhlcal-advance-button-secondary"
                disabled={busy}
                title="Finish the selected series (or the next active series)"
                onClick={() => {
                  runSimSeries();
                }}
              >
                <span className="po-hub-action-icon" aria-hidden>
                  ⏩
                </span>
                Sim Series
              </button>
              <button
                type="button"
                className={userEliminated || !userSeries ? "nhlcal-advance-button" : "nhlcal-advance-button-secondary"}
                disabled={busy}
                title="Sped-up sim through the Stanley Cup (bracket updates as days fly by)"
                onClick={() => runFastForwardPlayoffs()}
              >
                <span className="po-hub-action-icon" aria-hidden>
                  ⏭
                </span>
                {userEliminated || !userSeries ? "Sim Playoffs → Awards" : "Sim Playoffs"}
              </button>
            </>
          )}
          {busy ? <span className="po-hub-busy">Simming…</span> : null}
        </footer>
      </main>
    </div>
  );
}

const PO_HUB_CSS = `
.po-hub-root.nhlcal-root {
  min-height: 0;
  height: 100%;
  max-height: 100%;
}
.po-hub-root {
  height: 100%;
  min-height: 0;
  max-height: 100%;
  overflow: hidden;
}
.po-start-menu-host {
  height: 100%;
  min-height: 0;
  max-height: 100%;
  width: 100%;
  overflow: hidden;
}
.nhlcal-main {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  max-height: 100%;
  overflow: hidden;
}
.po-hub-topbar { align-items: flex-start; gap: 12px; padding: 10px 14px 6px; flex-shrink: 0; }
.po-hub-directive {
  margin: 6px 0 0;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  max-width: 760px;
  opacity: 0.88;
}
.po-hub-hub-link {
  border: 1px solid var(--line-2);
  background: rgba(8, 24, 36, 0.9);
  color: var(--text);
  border-radius: var(--radius-card);
  padding: 10px 14px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  cursor: pointer;
}
.po-hub-hub-link:hover { border-color: var(--line-strong); color: var(--cyan); }
.po-hub-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 10px;
  padding: 0 12px 8px;
  min-height: 0;
  flex: 1;
  overflow: hidden;
}
.po-hub-bracket {
  overflow: hidden;
  padding: 8px 10px;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.po-hub-bracket-banner {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
  padding: 6px 10px;
  border: 1px solid rgba(233,168,60,0.28);
  border-radius: var(--radius-card);
  background: linear-gradient(90deg, rgba(233,168,60,0.12), rgba(19,216,231,0.08), rgba(233,168,60,0.12));
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  flex-shrink: 0;
}
.po-hub-bracket-banner strong { color: var(--gold); font-size: 11px; }
.po-hub-bracket-banner span { text-align: center; color: var(--cyan); opacity: 0.9; }
.po-hub-bracket-row {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 0 12px;
  align-items: stretch;
  flex: 1;
  min-height: 0;
  position: relative;
}
.po-hub-bracket-row-flat { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.po-hub-col { position: relative; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.po-hub-col header {
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text);
  font-weight: 1000;
  margin-bottom: 6px;
  text-align: center;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--line);
  flex-shrink: 0;
}
.po-hub-col-stack {
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: space-evenly;
  flex: 1;
  min-height: 0;
}
.po-hub-col.connector-right .po-hub-col-stack::after,
.po-hub-col.connector-both .po-hub-col-stack::after,
.po-hub-col.connector-cup .po-hub-col-stack::after {
  content: "";
  position: absolute;
  top: 36px;
  bottom: 10px;
  right: -7px;
  width: 7px;
  border-right: 2px solid rgba(19,216,231,0.28);
  pointer-events: none;
}
.po-hub-col.connector-left .po-hub-col-stack::before,
.po-hub-col.connector-both .po-hub-col-stack::before,
.po-hub-col.connector-cup .po-hub-col-stack::before {
  content: "";
  position: absolute;
  top: 36px;
  bottom: 10px;
  left: -7px;
  width: 7px;
  border-left: 2px solid rgba(19,216,231,0.28);
  pointer-events: none;
}
.po-hub-series-card {
  width: 100%;
  text-align: left;
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  background: linear-gradient(180deg, rgba(12,35,52,0.98), rgba(6,18,28,0.98));
  color: var(--text);
  padding: 6px 7px;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
  position: relative;
  flex: 0 1 auto;
}
.po-hub-series-card:hover { border-color: var(--line-2); transform: translateY(-1px); }
.po-hub-series-card.is-selected { border-color: var(--line-strong); box-shadow: 0 0 0 1px rgba(19,216,231,0.3); }
.po-hub-series-card.is-user {
  border-color: rgba(233,168,60,0.7);
  background: linear-gradient(180deg, rgba(55,36,8,0.72), rgba(12,35,52,0.98));
  box-shadow: inset 3px 0 0 var(--gold);
}
.po-hub-series-card.is-active { outline: 1px solid rgba(19,216,231,0.22); }
.po-hub-series-card.is-complete { opacity: 0.9; }
.po-hub-series-card.is-pending { opacity: 0.48; }
.po-hub-series-card.is-just-set { animation: poHubAdvance 0.85s ease; }
@keyframes poHubAdvance {
  0% { transform: scale(0.96); box-shadow: 0 0 0 0 rgba(19,216,231,0.45); }
  55% { transform: scale(1.02); box-shadow: 0 0 0 6px rgba(19,216,231,0); }
  100% { transform: scale(1); }
}
.po-hub-series-meta {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  font-size: 11px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
  font-weight: 800;
}
.po-hub-series-card.is-tonight { outline: 1px solid rgba(233,168,60,0.55); }
.po-hub-home-tag.is-tonight { color: var(--gold); font-weight: 1000; }
.po-hub-series-row {
  display: grid;
  grid-template-columns: 18px 28px 1fr 16px;
  gap: 5px;
  align-items: center;
  padding: 3px 0;
}
.po-hub-series-row.is-out { opacity: 0.35; }
.po-hub-series-row.is-lead strong { color: var(--cyan); }
.po-hub-series-row.has-home .po-hub-seed { color: var(--gold); }
.po-hub-seed {
  font-size: 12px;
  font-weight: 1000;
  color: var(--text);
  text-align: center;
}
.po-hub-series-info { display: flex; flex-direction: column; min-width: 0; gap: 0; }
.po-hub-series-info strong { font-size: 12px; font-weight: 1000; letter-spacing: 0.02em; }
.po-hub-series-info small { color: #b7c9d6; font-size: 11px; font-weight: 800; }
.po-hub-series-row em {
  font-style: normal;
  font-weight: 1000;
  text-align: right;
  color: var(--text);
  font-size: 13px;
}
.po-hub-series-scoreline {
  margin-top: 2px;
  font-size: 11px;
  color: #d5e6f0;
  font-weight: 800;
}
.po-hub-series-last {
  margin-top: 1px;
  font-size: 11px;
  color: var(--cyan);
  font-weight: 800;
}
.po-hub-team-mark {
  display: inline-grid;
  place-items: center;
  border-radius: var(--radius-hud);
  overflow: hidden;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--line);
  flex-shrink: 0;
}
.po-hub-team-mark img { width: 100%; height: 100%; object-fit: contain; }
.po-hub-team-mark.is-dimmed { filter: grayscale(0.85); opacity: 0.45; }
.po-hub-fallback { font-size: 11px; font-weight: 900; }
.po-hub-tbd {
  display: inline-grid;
  place-items: center;
  font-size: 11px;
  color: var(--muted);
  font-weight: 800;
  border: 1px dashed var(--line);
  border-radius: var(--radius-hud);
}
.po-hub-empty-slot {
  border: 1px dashed var(--line);
  border-radius: var(--radius-card);
  min-height: 64px;
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
}
.po-hub-side {
  min-height: 0;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
  background: var(--panel);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.po-hub-side .nhlcal-panel-head { display: grid; gap: 3px; margin-bottom: 8px; flex-shrink: 0; }
.po-hub-side .nhlcal-panel-head h3 { margin: 0; font-size: 15px; }
.po-hub-side .nhlcal-panel-head p {
  margin: 0;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.po-hub-side-score {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 6px;
  align-items: center;
  margin-bottom: 6px;
  flex-shrink: 0;
}
.po-hub-side-score > div { display: grid; gap: 2px; justify-items: center; text-align: center; }
.po-hub-side-score strong { font-size: 12px; }
.po-hub-side-score small { color: #b7c9d6; font-size: 11px; font-weight: 800; }
.po-hub-side-score em { font-style: normal; font-size: 20px; font-weight: 1000; color: var(--cyan); }
.po-hub-side-score > span { font-size: 11px; letter-spacing: 0.12em; color: var(--muted); }
.po-hub-side-status, .po-hub-side-next, .po-hub-muted {
  color: #b7c9d6;
  font-size: 11px;
  margin: 0 0 8px;
  font-weight: 700;
  flex-shrink: 0;
}
.po-hub-game-slots {
  display: grid;
  gap: 4px;
  margin-bottom: 8px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.po-hub-game-slot {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 6px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: var(--radius-control);
  padding: 4px 6px;
  background: rgba(0,0,0,0.18);
  font-size: 11px;
}
.po-hub-game-slot.is-empty { opacity: 0.5; }
.po-hub-game-slot.is-ot strong { color: var(--gold); }
.po-hub-game-slot.is-latest strong { color: var(--cyan); }
.po-hub-game-pending { font-weight: 700; color: var(--muted); font-size: 11px; }
.po-hub-side-actions { display: grid; gap: 6px; flex-shrink: 0; }
.po-hub-actionbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 8px 14px 12px;
  border-top: 1px solid var(--line);
  background: rgba(4,16,26,0.94);
  flex-shrink: 0;
}
.po-hub-actionbar .nhlcal-advance-button,
.po-hub-actionbar .nhlcal-advance-button-secondary,
.po-hub-side-actions .nhlcal-advance-button,
.po-hub-side-actions .nhlcal-advance-button-secondary {
  height: 40px;
  min-width: 108px;
  font-size: 11px;
  letter-spacing: 0.08em;
}
.po-hub-actionbar .nhlcal-advance-button,
.po-hub-actionbar .nhlcal-advance-button-secondary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 132px;
  font-size: 11px;
}
.po-hub-action-icon {
  display: inline-flex;
  width: 1.1em;
  justify-content: center;
  opacity: 0.9;
  font-size: 13px;
}
.po-hub-busy {
  color: var(--cyan);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.po-hub-error { color: var(--red); padding: 0 14px; font-weight: 700; flex-shrink: 0; }
.nhlcal-eyebrow {
  margin: 0;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 900;
}
.nhlcal-topbar h1 {
  margin: 2px 0 0;
  font-family: var(--font-broadcast-display, "Archivo Black", sans-serif);
  font-size: clamp(1.35rem, 2.4vw, 1.85rem);
  font-weight: 400;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  line-height: 1.05;
}
.nhlcal-topbar h1::after {
  content: "";
  display: block;
  width: 56px;
  height: 2px;
  margin-top: 4px;
  background: linear-gradient(90deg, var(--gold), var(--cyan));
}
.po-hub-topbar .nhlcal-eyebrow { color: var(--gold); }
@media (max-width: 1100px) {
  .po-hub-layout { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .po-hub-series-card,
  .po-hub-series-card:hover,
  .po-hub-series-card.is-just-set {
    animation: none !important;
    transform: none !important;
    transition: none !important;
  }
}
`;
