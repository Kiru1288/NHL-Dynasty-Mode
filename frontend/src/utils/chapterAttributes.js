/** Authoritative chapter attribute schema for skaters and goalies. */

export const SKATER_CHAPTER_ORDER = [
  ["Overall", "overall"],
  ["Offence", "offence"],
  ["Defence", "defence"],
  ["Character", "character"],
  ["Mental", "mental"],
  ["Transition", "transition"],
  ["Physicality", "physical"],
  ["Potential", "potential"],
];

export const GOALIE_CHAPTER_ORDER = [
  ["Overall", "overall"],
  ["Glove", "glove"],
  ["Blocker", "blocker"],
  ["Stick", "stick"],
  ["Potential", "potential"],
];

function roundInt(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.round(n);
}

function isGoalie(player) {
  const pos = String(player?.position || player?.pos || "").toUpperCase();
  return pos === "G" || pos === "GOALIE" || pos === "GOALTENDER";
}

function chapterValue(chapters, key) {
  if (!chapters || typeof chapters !== "object") return null;
  const aliases = key === "defence" ? ["defence", "defense"] : [key];
  for (const alias of aliases) {
    const raw = chapters[alias];
    if (raw == null || raw === "") continue;
    if (typeof raw === "object" && raw.band) {
      const lo = roundInt(raw.low);
      const hi = roundInt(raw.high);
      if (lo != null && hi != null) {
        return { band: true, lo, hi, mid: Math.round((lo + hi) / 2) };
      }
      continue;
    }
    const n = roundInt(raw);
    if (n != null) return n;
  }
  return null;
}

function readChapter(chapters, key) {
  return chapterValue(chapters, key);
}

function hashSeed(text) {
  let h = 2166136261;
  const s = String(text || "player");
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function spreadFlatChapters(player, chapters, keys) {
  const vals = keys.map((key) => chapters[key]).filter((n) => Number.isFinite(n));
  if (vals.length < 3 || Math.max(...vals) - Math.min(...vals) > 2) return chapters;
  const base = Number.isFinite(chapters.overall) ? chapters.overall : vals[0];
  let seed = hashSeed(player.player_id || player.id || player.name);
  const next = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed;
  };
  const spans = {
    offence: [-7, 8],
    defence: [-8, 6],
    transition: [-5, 7],
    mental: [-4, 6],
    physical: [-8, 5],
    character: [-12, 8],
    glove: [-4, 5],
    blocker: [-5, 4],
    stick: [-6, 4],
  };
  const pos = String(player.position || player.pos || "").toUpperCase();
  const out = { ...chapters };
  keys.forEach((key) => {
    let [lo, hi] = spans[key] || [-6, 6];
    if (pos.includes("D") && key === "defence") [lo, hi] = [0, 8];
    if (pos.includes("D") && key === "offence") [lo, hi] = [-10, 2];
    const span = hi - lo + 1;
    out[key] = Math.max(40, Math.min(99, base + lo + (next() % span)));
  });
  return out;
}

export function resolveChapterMap(player) {
  if (!player) return {};

  const profileChapters =
    (player.chapter_profile?.chapters && typeof player.chapter_profile.chapters === "object"
      ? player.chapter_profile.chapters
      : null) ||
    (player.dossier?.chapter_profile?.chapters && typeof player.dossier.chapter_profile.chapters === "object"
      ? player.dossier.chapter_profile.chapters
      : null) ||
    (player.chapters && typeof player.chapters === "object" ? player.chapters : null) ||
    {};

  const ovr = roundInt(
    player.overall ??
      player.effective_ovr ??
      player.ovr ??
      player.base_ovr ??
      player.scouted_overall_estimate
  );
  const pot = roundInt(
    player.potential ??
      player.potential_score ??
      player.dev_potential ??
      (Array.isArray(player.potential_range) ? player.potential_range[1] : null)
  );

  if (isGoalie(player)) {
    return spreadFlatChapters(player, {
      overall: readChapter(profileChapters, "overall") ?? ovr,
      glove: readChapter(profileChapters, "glove") ?? ovr,
      blocker: readChapter(profileChapters, "blocker") ?? ovr,
      stick: readChapter(profileChapters, "stick") ?? (ovr != null ? Math.max(0, ovr - 2) : null),
      potential: readChapter(profileChapters, "potential") ?? pot,
    }, ["glove", "blocker", "stick"]);
  }

  return spreadFlatChapters(player, {
    overall: readChapter(profileChapters, "overall") ?? ovr,
    offence: readChapter(profileChapters, "offence") ?? roundInt(player.offence ?? player.shooting) ?? ovr,
    defence: readChapter(profileChapters, "defence") ?? roundInt(player.defence ?? player.defense) ?? ovr,
    character:
      readChapter(profileChapters, "character") ??
      roundInt(player.character_score ?? player.character) ??
      roundInt(player.mental ?? player.hockey_iq) ??
      ovr,
    mental: readChapter(profileChapters, "mental") ?? roundInt(player.mental ?? player.hockey_iq) ?? ovr,
    transition: readChapter(profileChapters, "transition") ?? roundInt(player.transition ?? player.skating) ?? ovr,
    physical:
      readChapter(profileChapters, "physical") ??
      roundInt(player.physical ?? player.physicality) ??
      ovr,
    potential: readChapter(profileChapters, "potential") ?? pot,
  }, ["offence", "defence", "transition", "mental", "physical", "character"]);
}

export function chapterAttributeRows(player) {
  if (!player) return [];
  const chapters = resolveChapterMap(player);
  const order = isGoalie(player) ? GOALIE_CHAPTER_ORDER : SKATER_CHAPTER_ORDER;
  return order
    .map(([label, key]) => {
      const value = chapters[key];
      if (value == null || value === "") return null;
      if (typeof value === "object" && value.band) {
        return [label, value];
      }
      const n = roundInt(value);
      return n != null ? [label, n] : null;
    })
    .filter(Boolean);
}

export function chapterNumericValue(value) {
  if (value == null) return null;
  if (typeof value === "object" && value.band) {
    return roundInt(value.mid ?? value.lo);
  }
  return roundInt(value);
}
