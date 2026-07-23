const EMPTY_ARR = Object.freeze([]);

function asArray(v) {
  return Array.isArray(v) ? v : EMPTY_ARR;
}

function firstNumber(...values) {
  for (const v of values) {
    if (v === null || v === undefined || v === "") continue;
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function normKey(v) {
  return String(v || "").trim().toLowerCase();
}

export function getTeamAbbreviation(team) {
  if (team == null || team === "") return "—";
  if (typeof team === "string") {
    const s = team.trim().toUpperCase();
    if (s.length <= 4 && s.length >= 2) return s;
    const parts = s.split(/\s+/);
    return (parts[parts.length - 1] || s).slice(0, 3) || "—";
  }
  if (typeof team !== "object") return "—";
  const abbr =
    team.abbreviation ||
    team.abbr ||
    team.abbrev ||
    team.code ||
    team.short_name ||
    team.shortName;
  if (abbr) return String(abbr).toUpperCase();
  const name = team.name || team.full_name || team.fullName || team.display_name || "";
  if (name) {
    const parts = String(name).trim().split(/\s+/);
    return (parts[parts.length - 1] || name).slice(0, 3).toUpperCase();
  }
  return "—";
}

function readScore(game, side) {
  const homeKeys = ["homeScore", "home_score", "home_goals", "homeGoals", "homeGoals"];
  const awayKeys = ["awayScore", "away_score", "away_goals", "awayGoals", "awayGoals"];
  const keys = side === "home" ? homeKeys : awayKeys;
  const nested = game?.score;
  const nestedVal = nested && typeof nested === "object" ? nested[side] : null;
  return firstNumber(game?.[keys[0]], game?.[keys[1]], game?.[keys[2]], game?.[keys[3]], nestedVal);
}

export function isGameLive(game) {
  if (!game || typeof game !== "object") return false;
  if (game.is_live === true || game.isLive === true || game.in_progress === true) return true;
  const status = normKey(game.status || game.game_status || game.state || game.gameState);
  return ["live", "in_progress", "in progress", "in-progress", "playing", "active"].includes(status);
}

export function isGameFinal(game) {
  if (!game || typeof game !== "object") return false;
  if (isGameLive(game)) return false;

  const status = normKey(game.status || game.game_status || game.state || game.gameState);
  const scheduled = ["scheduled", "pregame", "preview", "upcoming", "not_started", "not-started", "ns"].includes(status);
  if (scheduled) return false;

  const home = readScore(game, "home");
  const away = readScore(game, "away");
  if (home === null || away === null) return false;
  if (home === away) return false;
  if (home === 0 && away === 0 && !game.completed && !game.is_final && !game.isFinal && !game.simmed) {
    return false;
  }

  const finalFlag =
    game.completed === true ||
    game.is_final === true ||
    game.isFinal === true ||
    game.simmed === true ||
    game.played === true;

  if (finalFlag) return true;

  if (["final", "completed", "complete", "played", "done", "simmed", "finished"].includes(status)) {
    return true;
  }

  // Calendar slate rows often include scores only after simulation.
  if (!scheduled && !isGameLive(game)) return true;

  return false;
}

function isShootout(game) {
  return Boolean(
    game?.shootout ||
      game?.went_so ||
      game?.wentSO ||
      game?.so ||
      normKey(game?.result_type || game?.resultType).includes("so") ||
      normKey(game?.period_final || game?.periodFinal).includes("so")
  );
}

function isOvertime(game) {
  if (isShootout(game)) return false;
  return Boolean(
    game?.overtime ||
      game?.went_ot ||
      game?.wentOT ||
      game?.ot ||
      normKey(game?.result_type || game?.resultType).includes("ot") ||
      normKey(game?.period_final || game?.periodFinal).includes("ot")
  );
}

export function formatGameTime(value) {
  if (!value) return "TBD";
  const raw = String(value).trim();
  if (/^\d{1,2}:\d{2}\s?(AM|PM)$/i.test(raw)) {
    return raw.toUpperCase().replace(/\s+/, " ");
  }
  if (/^\d{1,2}:\d{2}$/.test(raw)) {
    const [hourRaw, minute] = raw.split(":");
    let hour = Number(hourRaw);
    if (!Number.isFinite(hour)) return raw;
    const suffix = hour >= 12 ? "PM" : "AM";
    hour = hour % 12 || 12;
    return `${hour}:${minute} ${suffix}`;
  }
  return raw;
}

export function formatScheduledGame(game) {
  const away = game.awayAbbr || "—";
  const home = game.homeAbbr || "—";
  const time = formatGameTime(game.gameTime);
  return `${away} @ ${home} ${time}`;
}

export function formatFinalGame(game) {
  const home = Number(game.homeScore);
  const away = Number(game.awayScore);
  const homeWins = home > away;
  const winner = homeWins ? game.homeAbbr : game.awayAbbr;
  const loser = homeWins ? game.awayAbbr : game.homeAbbr;
  const winScore = homeWins ? home : away;
  const loseScore = homeWins ? away : home;
  let prefix = "FINAL";
  if (game.shootout) prefix = "FINAL/SO";
  else if (game.overtime) prefix = "FINAL/OT";
  return `${prefix}: ${winner} ${winScore}, ${loser} ${loseScore}`;
}

export function formatLiveGame(game) {
  const away = game.awayAbbr || "—";
  const home = game.homeAbbr || "—";
  const awayScore = game.awayScore ?? 0;
  const homeScore = game.homeScore ?? 0;
  const period = game.periodLabel || "LIVE";
  return `LIVE: ${away} ${awayScore}, ${home} ${homeScore} — ${period}`;
}

function normalizeRawGame(raw, index = 0) {
  if (!raw || typeof raw !== "object") return null;

  const awayAbbr = getTeamAbbreviation(
    raw.away_abbr ||
      raw.awayAbbr ||
      raw.away_team ||
      raw.awayTeam ||
      raw.away_name ||
      raw.awayName ||
      raw.away_id ||
      raw.away
  );
  const homeAbbr = getTeamAbbreviation(
    raw.home_abbr ||
      raw.homeAbbr ||
      raw.home_team ||
      raw.homeTeam ||
      raw.home_name ||
      raw.homeName ||
      raw.home_id ||
      raw.home
  );

  const homeScore = readScore(raw, "home");
  const awayScore = readScore(raw, "away");

  const gameTime =
    raw.time ||
    raw.start_time ||
    raw.startTime ||
    raw.puck_drop ||
    raw.puckDrop ||
    raw.game_time ||
    raw.gameTime ||
    "";

  const periodLabel =
    raw.period_label ||
    raw.periodLabel ||
    raw.period ||
    raw.clock ||
    raw.time_remaining ||
    raw.timeRemaining ||
    "";

  const normalized = {
    id: String(raw.id || raw.game_id || raw.gameId || `${awayAbbr}-${homeAbbr}-${index}`),
    awayAbbr,
    homeAbbr,
    homeScore,
    awayScore,
    gameTime: String(gameTime || ""),
    status: raw.status || raw.game_status || raw.state || "",
    periodLabel: String(periodLabel || ""),
    overtime: isOvertime(raw),
    shootout: isShootout(raw),
    simmed: Boolean(raw.simmed || raw.played || raw.completed),
    raw,
  };

  normalized.isLive = isGameLive(raw) || isGameLive(normalized);
  normalized.isFinal = !normalized.isLive && isGameFinal({ ...raw, ...normalized });
  normalized.isScheduled = !normalized.isLive && !normalized.isFinal;

  return normalized;
}

function collectRawGamesFromPayload(payload) {
  if (!payload || typeof payload !== "object") return [];

  const directLists = [
    payload.games,
    payload.todays_games,
    payload.todaysGames,
    payload.daily_games,
    payload.dailyGames,
    payload.fixtures,
    payload.game_results,
    payload.gameResults,
    payload.completed_games,
    payload.completedGames,
    payload.schedule,
  ];

  for (const list of directLists) {
    if (Array.isArray(list) && list.length) return list;
  }

  const full = asArray(payload.nhl_calendar_full || payload.nhlCalendarFull);
  const todayMeta = payload.nhl_today || payload.nhlToday || {};
  const curIdx = todayMeta.calendar_index ?? todayMeta.calendarIndex;

  const todayRow =
    full.find((d) => d?.is_today_cursor) ||
    full.find((d) => Number(d?.calendar_index) === Number(curIdx)) ||
    full.find((d) => String(d?.iso || "") === String(todayMeta.iso || ""));

  if (todayRow && Array.isArray(todayRow.games) && todayRow.games.length) {
    return todayRow.games;
  }

  const upcoming = asArray(payload.schedule_upcoming || payload.scheduleUpcoming);
  const todayIso = String(todayMeta.iso || todayRow?.iso || "").slice(0, 10);
  if (todayIso && upcoming.length) {
    const block = upcoming.find((d) => String(d?.iso || d?.date || "").slice(0, 10) === todayIso);
    if (block && Array.isArray(block.games) && block.games.length) return block.games;
  }

  return [];
}

export function normalizeTickerGames(payload) {
  try {
    const rawGames = collectRawGamesFromPayload(payload);
    if (!rawGames.length) return [];

    const seen = new Set();
    const out = [];

    rawGames.forEach((raw, index) => {
      const row = normalizeRawGame(raw, index);
      if (!row) return;
      const key = row.id || `${row.awayAbbr}@${row.homeAbbr}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push(row);
    });

    return out;
  } catch {
    return [];
  }
}

export function buildTickerItemText(game) {
  if (!game) return "";
  if (game.isLive) return formatLiveGame(game);
  if (game.isFinal) return formatFinalGame(game);
  return formatScheduledGame(game);
}

export function buildTickerItems(games) {
  const list = asArray(games);
  if (!list.length) {
    return [{ id: "empty", text: "NO GAMES TODAY", state: "empty" }];
  }

  return list.map((game) => ({
    id: game.id,
    text: buildTickerItemText(game),
    state: game.isLive ? "live" : game.isFinal ? "final" : "scheduled",
  }));
}

export function deriveTickerStateLabel(games) {
  const list = asArray(games);
  if (!list.length) return "TODAY";
  if (list.some((g) => g.isLive)) return "LIVE";
  if (list.every((g) => g.isFinal)) return "FINAL";
  if (list.some((g) => g.isFinal)) return "RESULTS";
  return "TODAY";
}
