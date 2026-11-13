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
  // Your own club's players can be "considering" (not confirmed yet). They used
  // to be dropped here, so user-team retirees went missing from the page.
  const rows = Array.isArray(raw)
    ? raw
    : [...safeArray(raw.all), ...safeArray(raw.team), ...(raw.all ? [] : safeArray(raw))];
  const seen = new Set();
  const out = [];

  rows.forEach((row) => {
    if (!row) return;
    if (row.confirmed === false && !(Array.isArray(raw?.team) && raw.team.includes(row))) return;
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

/* ---------- Tribute screen helpers (fields are optional: old saves lack them) ---------- */

function num(v) {
  if (v === undefined || v === null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Career totals: real NHL career when the backend sends it, else what the save recorded. */
export function getCareerTotals(row) {
  const c = row?.career_totals;
  if (c && typeof c === "object" && num(c.gp) > 0) {
    return {
      source: "nhl",
      gp: num(c.gp),
      g: num(c.g),
      a: num(c.a),
      pts: num(c.pts),
      pm: num(c.pm),
      w: num(c.w),
      so: num(c.so),
      sv_pct: num(c.sv_pct),
      gaa: num(c.gaa),
      seasons: num(c.seasons),
      first: c.first_season || null,
      last: c.last_season || null,
      teams: Array.isArray(c.teams) ? c.teams.filter(Boolean) : [],
    };
  }
  const t = getLeagueTotals(row, "nhl");
  return {
    source: "save",
    gp: num(t.gp),
    g: num(t.g),
    a: num(t.a),
    pts: num(t.pts),
    pm: num(t.pm),
    w: num(t.w),
    so: num(t.so),
    sv_pct: num(t.sv_pct),
    gaa: num(t.gaa),
    seasons: null,
    first: null,
    last: null,
    teams: [],
  };
}

export function isMinorLeagueRetiree(row) {
  if (row?.level) return true;
  return row?.retirement_type === "depth_retirement" && row?.seasons_played == null;
}

export function isUserTeamRetiree(row, userTeamId) {
  if (row?.is_user_team) return true;
  return userTeamId != null && userTeamId !== "" && String(row?.team_id ?? "") === String(userTeamId);
}

export function tributeScore(row) {
  const c = getCareerTotals(row);
  const ovr = getRetireeOverall(row) ?? 0;
  const goalie = isGoalieRetiree(row);
  const production = goalie ? (c.w || 0) * 2 : c.pts || 0;
  return ovr * 10 + production * 0.06 + (c.gp || 0) * 0.01 + (Number(row?.cups_won) || 0) * 25;
}

export function fmtInt(v) {
  const n = num(v);
  return n == null ? "—" : Math.round(n).toLocaleString("en-US");
}

export function fmtSigned(v) {
  const n = num(v);
  if (n == null) return "—";
  return n > 0 ? `+${Math.round(n)}` : String(Math.round(n));
}

export function fmtSvPct(v) {
  const n = num(v);
  if (n == null || n <= 0) return "—";
  const frac = n > 1.5 ? n / 100 : n;
  return frac.toFixed(3).replace(/^0/, "");
}

export function fmtGaa(v) {
  const n = num(v);
  return n == null || n <= 0 ? "—" : n.toFixed(2);
}

export function fmtMillions(v) {
  const n = num(v);
  if (n == null || n <= 0) return null;
  return `$${n >= 10 ? n.toFixed(1) : n.toFixed(2).replace(/0$/, "")}M`;
}

export function draftLine(row) {
  const d = row?.draft && typeof row.draft === "object" ? row.draft : null;
  const legacy = row?.draft_info && typeof row.draft_info === "object" ? row.draft_info : {};
  const year = d?.draft_year ?? legacy.draft_year;
  const round = d?.draft_round ?? legacy.draft_round;
  const pick = d?.draft_overall_pick ?? legacy.draft_pick ?? legacy.draft_overall_pick;
  const team = d?.draft_team_abbr;
  if (d?.undrafted) return "Undrafted";
  if (!year && !round && !pick) return null;
  const parts = [];
  if (year) parts.push(String(year));
  if (round) parts.push(`round ${round}`);
  if (pick) parts.push(`#${pick} overall`);
  return `${parts.join(", ")}${team ? ` by ${team}` : ""}`;
}

export function contractLine(row) {
  const c = row?.last_contract;
  if (!c || typeof c !== "object") return null;
  const aav = fmtMillions(c.aav_millions);
  if (!aav) return null;
  const yrs = num(c.years_remaining) || 0;
  if (yrs > 0) {
    const total = fmtMillions(c.walked_away_millions);
    return `${aav} AAV, ${yrs} year${yrs === 1 ? "" : "s"} left${total ? ` (${total} walked away from)` : ""}`;
  }
  return `${aav} AAV, expiring`;
}

/** Trophy names grouped with counts, Stanley Cups first. */
export function trophySummary(row) {
  const list = Array.isArray(row?.trophies) ? row.trophies : [];
  const counts = new Map();
  list.forEach((t) => {
    const name = String(t?.award_name || "").trim();
    if (!name) return;
    counts.set(name, (counts.get(name) || 0) + 1);
  });
  const out = [...counts.entries()].map(([name, n]) => ({ name, n }));
  out.sort((a, b) => {
    const ac = /stanley cup/i.test(a.name) ? 1 : 0;
    const bc = /stanley cup/i.test(b.name) ? 1 : 0;
    if (ac !== bc) return bc - ac;
    return b.n - a.n || a.name.localeCompare(b.name);
  });
  return out;
}

export function finalSeasonLine(row, goalie) {
  const f = row?.final_season;
  if (!f || typeof f !== "object" || !num(f.gp)) return null;
  if (goalie) {
    return [
      `${fmtInt(f.gp)} GP`,
      f.wins != null ? `${fmtInt(f.wins)} W` : null,
      f.svPct != null ? `${fmtSvPct(f.svPct)} SV%` : null,
      f.gaa != null ? `${fmtGaa(f.gaa)} GAA` : null,
    ].filter(Boolean).join(" · ");
  }
  return [
    `${fmtInt(f.gp)} GP`,
    `${fmtInt(f.g)} G`,
    `${fmtInt(f.a)} A`,
    `${fmtInt(f.pts)} PTS`,
    f.plusMinus != null ? fmtSigned(f.plusMinus) : null,
  ].filter(Boolean).join(" · ");
}
