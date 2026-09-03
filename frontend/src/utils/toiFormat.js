/** Shared TOI helpers — always derive ATOI from toi_sec / gp when possible. */

function safeNum(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function safeInt(value, fallback = 0) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
}

function firstPresent(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

export function getGamesPlayed(row) {
  return Math.max(0, safeInt(firstPresent(row?.gp, row?.games_played, row?.games), 0));
}

export function getTotalTOISeconds(row) {
  return Math.max(0, safeNum(firstPresent(row?.toi_sec, row?.time_on_ice_sec, row?.toi_total_sec), 0));
}

export function getAverageTOIMinutes(row) {
  const gp = getGamesPlayed(row);
  const toiSec = getTotalTOISeconds(row);
  if (toiSec > 0 && gp > 0) {
    return toiSec / 60 / gp;
  }

  const precomputed = safeNum(
    firstPresent(row?.toi, row?.avg_toi, row?.average_toi, row?.toi_per_game, row?.toi_min),
    0
  );
  if (precomputed > 0 && precomputed <= 30) {
    return precomputed;
  }
  if (precomputed > 0 && gp > 0) {
    return precomputed / gp;
  }
  return precomputed;
}

export function formatClockFromMinutes(minutesValue) {
  const minutes = safeNum(minutesValue, 0);
  if (minutes <= 0) return "—";

  const wholeMinutes = Math.floor(minutes);
  const seconds = Math.round((minutes - wholeMinutes) * 60);

  if (seconds >= 60) {
    return `${wholeMinutes + 1}:00`;
  }

  return `${wholeMinutes}:${String(seconds).padStart(2, "0")}`;
}

export function formatAverageTOI(row) {
  return formatClockFromMinutes(getAverageTOIMinutes(row));
}

export function formatTotalTOI(row) {
  const toiSec = getTotalTOISeconds(row);
  if (toiSec > 0) {
    return formatClockFromMinutes(toiSec / 60);
  }
  return formatClockFromMinutes(safeNum(firstPresent(row?.toi, row?.time_on_ice), 0));
}
