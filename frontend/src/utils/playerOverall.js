/**
 * Universal player overall — one source of truth across Roster, Stats Central,
 * Edit Lines, and Trade Hub.
 *
 * Backend fields (preferred order for display):
 *   effective_ovr → ovr → overall → base_ovr
 *
 * Scale: values <= 1.5 are treated as 0–1 fractions and mapped to 0–99.
 */

function toFiniteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function normalizeOverallScale(value) {
  const n = toFiniteNumber(value);
  if (n === null || n <= 0) return 0;
  if (n <= 1.5) return Math.max(1, Math.min(99, Math.round(n * 99)));
  return Math.max(1, Math.min(99, Math.round(n)));
}

function firstPositiveOverall(...candidates) {
  for (const candidate of candidates) {
    const scaled = normalizeOverallScale(candidate);
    if (scaled > 0) return scaled;
  }
  return 0;
}

/**
 * Canonical display overall for a player row from any screen/API shape.
 */
export function getUniversalOverall(player) {
  if (!player || typeof player !== "object") return 0;

  return firstPositiveOverall(
    player.effective_ovr,
    player.effectiveOvr,
    player.ovr,
    player.overall,
    player.display_overall,
    player.displayOverall,
    player.true_overall,
    player.trueOverall,
    player.base_ovr,
    player.baseOvr,
    player.rating
  );
}

/**
 * Pre-modifier / ratings-blend base overall when available.
 */
export function getBaseOverall(player) {
  if (!player || typeof player !== "object") return 0;

  return firstPositiveOverall(
    player.base_ovr,
    player.baseOvr,
    player.true_overall,
    player.trueOverall,
    player.ovr,
    player.overall
  );
}

/**
 * Points dropped from base → current (storyline / rumor / conduct modifiers).
 */
export function getOverallDrop(player) {
  const base = getBaseOverall(player);
  const current = getUniversalOverall(player);
  if (base <= 0 || current <= 0) return 0;
  return Math.max(0, base - current);
}

export function getOverallTooltip(player) {
  const current = getUniversalOverall(player);
  const base = getBaseOverall(player);
  const drop = getOverallDrop(player);

  if (current <= 0) return "Overall rating unavailable";
  if (drop > 0 && base > 0) {
    return `Overall ${current} (base ${base}, ↓${drop})`;
  }
  return `Overall rating ${current}`;
}

/**
 * Attach universal overall fields onto a player object for consistent UI/sort.
 */
export function withUniversalOverall(player) {
  if (!player || typeof player !== "object") return player;

  const ovr = getUniversalOverall(player);
  const baseOvr = getBaseOverall(player) || ovr;
  const drop = Math.max(0, baseOvr - ovr);

  return {
    ...player,
    ovr,
    overall: ovr,
    base_ovr: baseOvr,
    effective_ovr: ovr,
    overall_drop: drop,
  };
}
