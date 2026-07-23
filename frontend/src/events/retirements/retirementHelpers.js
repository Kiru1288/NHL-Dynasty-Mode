import { firstDefined, getPlayerName, getPlayerOverall, getPlayerPosition, safeArray } from "../shared/eventHelpers";

function dig(obj, path) {
  if (!obj || !path) return undefined;
  return String(path)
    .split(".")
    .reduce((cur, key) => (cur == null ? undefined : cur[key]), obj);
}

export function formatStatValue(value, { pct = false, decimals = 2 } = {}) {
  if (value === undefined || value === null || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (pct) {
    const normalized = n > 0 && n <= 1 ? n * 100 : n;
    return `${normalized.toFixed(decimals)}%`;
  }
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(decimals);
}

function readTotalsBlock(row, keys) {
  for (const key of keys) {
    const block = dig(row, key) ?? dig(row?.player, key);
    if (block && typeof block === "object" && !Array.isArray(block)) return block;
  }
  return null;
}

export function getRetireeName(row) {
  return getPlayerName(row?.player || row);
}

export function getRetireeOverall(row) {
  return getPlayerOverall(row?.player || row);
}

export function getRetireePosition(row) {
  return getPlayerPosition(row?.player || row);
}

export function getRetireeAge(row) {
  const age = firstDefined(row?.age, row?.player?.age);
  return age != null ? age : null;
}

export function getRetireeTeam(row) {
  return (
    firstDefined(
      row?.team_name,
      row?.team,
      row?.team_abbrev,
      row?.retired_from,
      row?.final_team,
      row?.player?.team_name,
      row?.player?.team
    ) || "UFA"
  );
}

function buildFlatNhlTotals(row) {
  const g = firstDefined(row?.goals, row?.career_goals);
  const a = firstDefined(row?.assists, row?.career_assists);
  const pts = firstDefined(row?.points, row?.career_points, g != null && a != null ? Number(g) + Number(a) : null);
  return {
    gp: firstDefined(row?.games_played, row?.career_games, row?.gp),
    g,
    a,
    pts,
    pm: firstDefined(row?.plus_minus, row?.plusMinus, row?.plus_minus_total),
    pim: firstDefined(row?.pim, row?.penalty_minutes, row?.penaltyMinutes),
    w: firstDefined(row?.goalie_wins, row?.wins, row?.w),
    l: firstDefined(row?.losses, row?.goalie_losses, row?.l),
    otl: firstDefined(row?.otl, row?.overtime_losses, row?.ot_losses),
    sv_pct: firstDefined(row?.save_percentage, row?.sv_pct, row?.save_pct, row?.sv),
    gaa: firstDefined(row?.gaa, row?.goals_against_average),
    so: firstDefined(row?.shutouts, row?.so),
  };
}

export function getLeagueTotals(row, league = "nhl") {
  const isNhl = league === "nhl";
  const block =
    readTotalsBlock(row, isNhl ? ["nhl_totals", "career_nhl", "stats.nhl"] : ["ahl_totals", "career_ahl", "stats.ahl"]) ||
    readTotalsBlock(row?.player, isNhl ? ["nhl_totals", "career_nhl", "stats.nhl"] : ["ahl_totals", "career_ahl", "stats.ahl"]);

  if (block) {
    const g = firstDefined(block.g, block.goals);
    const a = firstDefined(block.a, block.assists);
    return {
      gp: firstDefined(block.gp, block.games_played, block.games),
      g,
      a,
      pts: firstDefined(block.pts, block.points, g != null && a != null ? Number(g) + Number(a) : null),
      pm: firstDefined(block.pm, block.plus_minus, block.plusMinus),
      pim: firstDefined(block.pim, block.penalty_minutes, block.penaltyMinutes),
      w: firstDefined(block.w, block.wins, block.goalie_wins),
      l: firstDefined(block.l, block.losses, block.goalie_losses),
      otl: firstDefined(block.otl, block.overtime_losses, block.ot_losses),
      sv_pct: firstDefined(block.sv_pct, block.save_percentage, block.save_pct, block.sv),
      gaa: firstDefined(block.gaa, block.goals_against_average),
      so: firstDefined(block.so, block.shutouts),
    };
  }

  if (isNhl) return buildFlatNhlTotals(row);
  return {
    gp: null,
    g: null,
    a: null,
    pts: null,
    pm: null,
    pim: null,
    w: null,
    l: null,
    otl: null,
    sv_pct: null,
    gaa: null,
    so: null,
  };
}

export function getNhlGamesPlayed(row) {
  const totals = getLeagueTotals(row, "nhl");
  const gp = Number(totals.gp);
  return Number.isFinite(gp) ? gp : 0;
}

export function isGoalieRetiree(row) {
  const pos = getRetireePosition(row);
  if (pos === "G") return true;
  const nhl = getLeagueTotals(row, "nhl");
  const ahl = getLeagueTotals(row, "ahl");
  return [nhl, ahl].some(
    (totals) =>
      totals.gaa != null ||
      totals.sv_pct != null ||
      (totals.w != null && totals.g == null && totals.a == null)
  );
}

export function normalizeRetireesPayload(raw) {
  if (!raw) return [];
  const rows = Array.isArray(raw) ? raw : safeArray(raw.all || raw);
  const seen = new Set();
  const out = [];

  rows.forEach((row) => {
    if (!row || row.confirmed === false) return;
    const key = String(firstDefined(row.player_id, row.id, getRetireeName(row)) || "");
    if (key && seen.has(key)) return;
    if (key) seen.add(key);
    out.push(row);
  });

  return out;
}

export function sortRetirees(rows) {
  return [...rows].sort((a, b) => {
    const ovrDiff = (getRetireeOverall(b) ?? -1) - (getRetireeOverall(a) ?? -1);
    if (ovrDiff !== 0) return ovrDiff;

    const gpDiff = getNhlGamesPlayed(b) - getNhlGamesPlayed(a);
    if (gpDiff !== 0) return gpDiff;

    return getRetireeName(a).localeCompare(getRetireeName(b));
  });
}

export function retireeToHeadshotPlayer(row) {
  const base = row?.player && typeof row.player === "object" ? { ...row.player } : { ...row };
  return {
    ...base,
    name: getRetireeName(row),
    position: getRetireePosition(row),
    overall: getRetireeOverall(row),
    age: getRetireeAge(row),
    player_id: firstDefined(row?.player_id, row?.player?.player_id, row?.id),
    team_name: getRetireeTeam(row),
  };
}
