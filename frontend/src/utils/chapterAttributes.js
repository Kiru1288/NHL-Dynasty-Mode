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
    return {
      overall: readChapter(profileChapters, "overall") ?? ovr,
      glove: readChapter(profileChapters, "glove") ?? ovr,
      blocker: readChapter(profileChapters, "blocker") ?? ovr,
      stick: readChapter(profileChapters, "stick") ?? (ovr != null ? Math.max(0, ovr - 2) : null),
      potential: readChapter(profileChapters, "potential") ?? pot,
    };
  }

  return {
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
  };
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
