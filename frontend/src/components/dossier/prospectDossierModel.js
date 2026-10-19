/**
 * One dossier view-model. Every number on the prospect dossier is read from
 * the object this builds. Components must not format raw profile/player fields.
 *
 * Contradictions in the payload (collapsed ranges, peaks over 99, a floor above
 * the current OVR, league coverage labeled as "your file") are reconciled here.
 */

import { certaintyTier, percentileOf, skillTier } from "./ProspectVisuals";
import { resolveCountryCode } from "../../utils/countryFlags";
import {
  buildScoutingDeskEntries,
  formatDeskGrade,
  resolveWjcDossierBlock,
} from "../../screens/prospectDossierHelpers";
import { weeklyTrajectoryPoints } from "../../screens/draftWarRoom";

const REVEAL_AT = 72;
const RATING_CAP = 99;

const SKATER_CHAPTERS = [
  ["offence", "Offence", "OFF"],
  ["defence", "Defence", "DEF"],
  ["transition", "Transition", "TRN"],
  ["mental", "Mental", "MEN"],
  ["physical", "Physicality", "PHY"],
  ["character", "Character", "CHR"],
];

const GOALIE_CHAPTERS = [
  ["glove", "Glove", "GLV"],
  ["blocker", "Blocker", "BLK"],
  ["stick", "Stick", "STK"],
];

const CHARACTER_TRAITS = [
  { key: "compete", label: "Competitive Drive", match: /compet|drive/i },
  { key: "coach", label: "Coachability", match: /coach/i },
  { key: "lead", label: "Leadership", match: /leader/i },
  { key: "work", label: "Work Ethic", match: /work|ethic|motor/i },
  { key: "social", label: "Social Adjustment", match: /social|adjust/i },
];

const TIER_RANK = {
  awful: 0,
  bad: 1,
  weak: 2,
  average: 3,
  good: 4,
  "very-good": 5,
  great: 6,
  elite: 7,
  "world-class": 8,
};

function num(v) {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function round(v) {
  const n = num(v);
  return n == null ? null : Math.round(n);
}

export function tierOf(value) {
  return skillTier(value);
}

export function formatRange(low, high) {
  const a = round(low);
  const b = round(high);
  if (a == null && b == null) return "—";
  if (a == null) return String(b);
  if (b == null) return String(a);
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  if (lo === hi) return String(lo);
  return `${lo}\u2013${hi}`;
}

/** A rating band: one number when the ends match or the read is exact, otherwise "74–90". */
export function formatRating(band) {
  if (!band || band.hidden) return "—";
  if (band.locked) return "Scout to reveal";
  if (band.exact && band.value != null) return String(Math.round(band.value));
  if (band.low != null && band.high != null) return formatRange(band.low, band.high);
  if (band.value != null) return String(Math.round(band.value));
  return "—";
}

export function formatPct(value) {
  const n = num(value);
  if (n == null) return "—";
  return `${Math.round(n)}%`;
}

export function formatRank(value) {
  const n = num(value);
  if (n == null || n <= 0) return "—";
  return `#${Math.round(n)}`;
}

export function formatDelta(value) {
  const n = num(value);
  if (n == null || Math.round(n) === 0) return null;
  const r = Math.round(n);
  if (r > 0) return `+${r}`;
  return `\u2212${Math.abs(r)}`;
}

export function formatPercentile(p) {
  const n = num(p);
  if (n == null) return "—";
  const rounded = Math.round(n);
  if (rounded >= 50) return `Top ${Math.max(1, 100 - rounded)}%`;
  return `Bottom ${Math.max(1, rounded)}%`;
}

function clipWords(text, max = 120) {
  const t = String(text || "").replace(/\s+/g, " ").trim();
  if (t.length <= max) return t;
  const slice = t.slice(0, max + 1);
  const space = slice.lastIndexOf(" ");
  const cut = (space > 40 ? slice.slice(0, space) : t.slice(0, max)).trim();
  return cut.replace(/[.,;:\u2014-]+$/, "");
}

function isGoaliePos(pos) {
  const p = String(pos || "").toUpperCase();
  return p === "G" || p.includes("GOAL");
}

function ageOf(player, profile) {
  const raw = player?.age ?? profile?.age ?? profile?.identity_badges?.age;
  const n = num(String(raw ?? "").replace(/[^\d.]/g, ""));
  return n != null && n > 0 ? Math.round(n) : null;
}

function filePctOf(player, profile, override) {
  if (override != null && Number.isFinite(Number(override))) {
    return clamp(Math.round(Number(override)), 0, 100);
  }
  const n = num(
    profile?.userScoutPct
    ?? player?.dedicatedScoutingPct
    ?? player?.scoutedPercentage
    ?? player?.scouted_percentage
    ?? 0,
  );
  return clamp(Math.round(n || 0), 0, 100);
}

function stageFor(pct) {
  if (pct <= 0) return "UNSCOUTED";
  if (pct < 36) return "EARLY";
  if (pct < REVEAL_AT) return "DEVELOPING";
  return "COMPLETE";
}

function consensusOf(player, profile) {
  const n = num(
    profile?.scout_confidence
    ?? player?.publicScoutingConfidence
    ?? player?.public_scouting_confidence
    ?? player?.scoutingConfidence
    ?? player?.scouting_confidence,
  );
  if (n == null) return null;
  return clamp(Math.round(n), 0, 100);
}

function buildOvr(player, profile, filePct) {
  const revealed = filePct >= REVEAL_AT || Boolean(
    player?.ovrRevealed || player?.ovr_revealed || player?.ability_revealed,
  );
  const range = player?.current_ovr_range;
  let low = round(
    profile?.now_range?.low
    ?? profile?.overallRangeLow
    ?? player?.ovrRange?.low
    ?? (Array.isArray(range) ? range[0] : null)
    ?? player?.public_ovr_low,
  );
  let high = round(
    profile?.now_range?.high
    ?? profile?.overallRangeHigh
    ?? player?.ovrRange?.high
    ?? (Array.isArray(range) ? range[1] : null)
    ?? player?.public_ovr_high,
  );
  if (low != null) low = clamp(low, 0, RATING_CAP);
  if (high != null) high = clamp(high, 0, RATING_CAP);
  if (low != null && high != null && low > high) [low, high] = [high, low];

  const exactSrc = round(
    profile?.scoutedOverall
    ?? profile?.currentOvrEstimate
    ?? profile?.scouted_overall
    ?? player?.ovrHint
    ?? player?.drafted_overall
    ?? player?.true_ovr
    ?? player?.overall
    ?? player?.current_ovr_estimate,
  );

  if (revealed && exactSrc != null && exactSrc > 0) {
    const value = clamp(exactSrc, 0, RATING_CAP);
    return { value, low: value, high: value, exact: true, revealed: true };
  }
  if (low != null && high != null) {
    if (low === high) return { value: low, low, high, exact: true, revealed: false };
    const value = clamp(Math.round((low + high) / 2), low, high);
    return { value, low, high, exact: false, revealed: false };
  }
  if (exactSrc != null && exactSrc > 0) {
    const value = clamp(exactSrc, 0, RATING_CAP);
    return { value, low: value, high: value, exact: true, revealed: false };
  }
  return { value: null, low: null, high: null, exact: false, revealed: false };
}

function buildPeak(player, profile, ovr) {
  const hidden = Boolean(
    profile?.ceilingHidden
    || profile?.potential?.hidden
    || profile?.peak_range?.hidden
    || player?.ceilingHidden
    || player?.ceiling_hidden,
  );
  if (hidden) return { value: null, low: null, high: null, exact: false, hidden: true };

  const potRange = player?.potential_range || player?.scouted_potential_range;
  const rangeLow = num(
    profile?.peak_range?.low
    ?? profile?.scoutedPotentialLow
    ?? player?.potentialRange?.low
    ?? (Array.isArray(potRange) ? potRange[0] : null),
  );
  const rangeHigh = num(
    profile?.peak_range?.high
    ?? profile?.scoutedPotentialHigh
    ?? player?.potentialRange?.high
    ?? (Array.isArray(potRange) ? potRange[1] : null),
  );
  const rating = num(profile?.potential?.rating ?? player?.potentialScore ?? player?.potential_score);
  const floor = num(profile?.potential?.floor ?? player?.floorScore);

  let high = rangeHigh != null ? rangeHigh : rating;
  let low = rangeLow;
  let lowFromFloor = false;
  if (low == null && floor != null) {
    low = floor;
    lowFromFloor = true;
  }
  if (high == null && low != null) high = low;
  if (low == null && high != null) low = high;
  if (low == null || high == null) {
    return { value: null, low: null, high: null, exact: false, hidden: false };
  }

  low = Math.round(Math.min(low, high));
  high = Math.min(RATING_CAP, Math.round(Math.max(high, low)));
  const ovrNow = ovr?.value != null ? ovr.value : ovr?.low;
  if (lowFromFloor && ovrNow != null && low > ovrNow) {
    low = high;
  }
  if (ovr?.low != null) low = Math.max(low, ovr.low);
  low = clamp(low, 0, RATING_CAP);
  if (low > high) low = high;
  const exact = low === high;
  const value = exact ? low : clamp(Math.round((low + high) / 2), low, high);
  return { value, low, high, exact, hidden: false };
}

function riskLevel(volatility, filePct, peak) {
  const vol = String(volatility || "").toLowerCase();
  const spread = !peak?.hidden && peak?.low != null && peak?.high != null ? peak.high - peak.low : 0;
  const wide = spread >= 8;
  const thin = filePct < 36;
  if (/high|volatile|bust|elevat/.test(vol) || thin) return "High";
  if ((/low|safe|stable/.test(vol) && filePct >= 55) || (filePct >= REVEAL_AT && !wide)) return "Low";
  return "Medium";
}

function chaptersOf(player, profile) {
  return player?.chapterProfile?.chapters
    || player?.chapter_profile?.chapters
    || profile?.chapter_profile?.chapters
    || profile?.chapterProfile?.chapters
    || null;
}

function chapterHidden(player, profile) {
  return Boolean(
    player?.chapterProfile?.hidden
    || player?.chapter_profile?.hidden
    || profile?.chapterProfileHidden
    || profile?.chapter_profile?.hidden
    || player?.chapter_profile?.fogged
    || profile?.chapterProfileFogged,
  );
}

function readRaw(chapters, key) {
  if (!chapters || typeof chapters !== "object") return null;
  const aliases = key === "defence"
    ? ["defence", "defense"]
    : key === "physical"
      ? ["physical", "physicality"]
      : [key];
  for (const alias of aliases) {
    const raw = chapters[alias];
    if (raw != null && raw !== "") return raw;
  }
  return null;
}

function sortMidOf(raw, filePct) {
  if (raw == null || typeof raw === "boolean") return null;
  if (typeof raw === "object") {
    const lo = num(raw.low ?? raw.lo);
    const hi = num(raw.high ?? raw.hi);
    if (lo == null || hi == null) return null;
    return (Math.min(lo, hi) + Math.max(lo, hi)) / 2;
  }
  const v = num(raw);
  if (v == null || v <= 0) return null;
  if (filePct < REVEAL_AT) return null;
  return v;
}

function skillFromRaw(key, label, short, raw, filePct, hidden) {
  const blank = {
    key, label, short, value: null, low: null, high: null, exact: false, locked: true, tier: null, percentile: null, classAvg: null,
  };
  if (hidden || raw == null) return blank;
  if (typeof raw === "object") {
    const lo = round(raw.low ?? raw.lo);
    const hi = round(raw.high ?? raw.hi);
    if (lo == null || hi == null) return blank;
    const low = clamp(Math.min(lo, hi), 0, RATING_CAP);
    const high = clamp(Math.max(lo, hi), 0, RATING_CAP);
    if (low === high) {
      return {
        key, label, short, value: low, low, high, exact: true, locked: false, tier: skillTier(low), percentile: null, classAvg: null,
      };
    }
    return {
      key,
      label,
      short,
      value: null,
      low,
      high,
      exact: false,
      locked: false,
      tier: skillTier((low + high) / 2),
      percentile: null,
      classAvg: null,
    };
  }
  const v = round(raw);
  if (v == null || v <= 0) return blank;
  const value = clamp(v, 0, RATING_CAP);
  if (filePct < REVEAL_AT) return blank;
  return {
    key, label, short, value, low: value, high: value, exact: true, locked: false, tier: skillTier(value), percentile: null, classAvg: null,
  };
}

function buildSkills(player, profile, filePct, allProspects, goalie) {
  const hidden = chapterHidden(player, profile);
  const order = goalie ? GOALIE_CHAPTERS : SKATER_CHAPTERS;
  const own = chaptersOf(player, profile) || {};
  const peers = (allProspects || []).filter((p) => isGoaliePos(p?.position || p?.pos) === goalie);
  const peerRaws = peers.map((p) => chaptersOf(p, p?.profile || p?.dossier) || {});

  return order.map(([key, label, short]) => {
    const skill = skillFromRaw(key, label, short, readRaw(own, key), filePct, hidden);
    const mids = peerRaws
      .map((chs) => sortMidOf(readRaw(chs, key), REVEAL_AT))
      .filter((v) => v != null);
    skill.classAvg = mids.length ? mids.reduce((a, b) => a + b, 0) / mids.length : null;
    const mine = skill.exact ? skill.value : (skill.low != null && skill.high != null ? (skill.low + skill.high) / 2 : null);
    skill.percentile = mine == null ? null : percentileOf(mids, mine);
    return skill;
  });
}

function tierRank(tier) {
  if (!tier?.key) return null;
  return TIER_RANK[tier.key] ?? null;
}

function wordTier(raw) {
  if (raw == null || raw === "") return null;
  const n = num(raw);
  if (n != null) return skillTier(n);
  const s = String(raw).toLowerCase();
  if (s.includes("world")) return skillTier(94);
  if (s.includes("elite")) return skillTier(88);
  if (/\bgreat\b/.test(s)) return skillTier(82);
  if (s.includes("very good") || s.includes("very_good")) return skillTier(76);
  if (/\bgood\b/.test(s)) return skillTier(68);
  if (s.includes("average")) return skillTier(60);
  if (s.includes("weak") || s.includes("below")) return skillTier(52);
  if (s.includes("bad") || s.includes("awful") || s.includes("poor")) return skillTier(40);
  return null;
}

function buildCharacter(profile, filePct) {
  const traitsIn = Array.isArray(profile?.character_read?.traits) ? profile.character_read.traits : [];
  const traits = [];
  CHARACTER_TRAITS.forEach((spec) => {
    const row = traitsIn.find((t) => spec.match.test(String(t?.label || t?.key || "")));
    if (!row) return;
    const tier = wordTier(row.tier ?? row.score ?? row.value);
    if (!tier || tier.key === "unknown") return;
    traits.push({ key: spec.key, label: spec.label, tier });
  });
  const flagged = traits.some((t) => (tierRank(t.tier) ?? 9) <= TIER_RANK.weak);
  const visible = traits.length > 0 && (filePct >= 36 || flagged);
  return { traits, visible, flagged };
}

function phrasePercentile(p) {
  const label = formatPercentile(p);
  if (label === "—") return null;
  return `${label.charAt(0).toLowerCase()}${label.slice(1)} of class`;
}

function buildNotes(skills, character) {
  const ranked = skills.filter((s) => !s.locked && (s.percentile != null || s.tier));
  const byPct = (a, b) => (b.percentile ?? -1) - (a.percentile ?? -1);
  const strengths = [];
  ranked
    .filter((s) => s.percentile != null && s.percentile >= 67)
    .sort(byPct)
    .slice(0, 2)
    .forEach((s) => {
      const detail = phrasePercentile(s.percentile);
      if (detail) strengths.push({ key: s.key, label: s.label, detail });
    });
  if (strengths.length < 2) {
    ranked
      .filter((s) => (tierRank(s.tier) ?? 0) >= TIER_RANK.great && !strengths.some((x) => x.key === s.key))
      .sort((a, b) => (tierRank(b.tier) ?? 0) - (tierRank(a.tier) ?? 0))
      .slice(0, 2 - strengths.length)
      .forEach((s) => {
        strengths.push({ key: s.key, label: s.label, detail: `${s.tier.label.toLowerCase()} on the league scale` });
      });
  }

  const concerns = [];
  ranked
    .filter((s) => s.percentile != null && s.percentile < 40)
    .sort((a, b) => a.percentile - b.percentile)
    .slice(0, 2)
    .forEach((s) => {
      const detail = phrasePercentile(s.percentile);
      if (detail) concerns.push({ key: s.key, label: s.label, detail });
    });
  character.traits
    .filter((t) => (tierRank(t.tier) ?? 9) <= TIER_RANK.weak)
    .forEach((t) => {
      if (concerns.length >= 3) return;
      if (concerns.some((c) => c.key === t.key)) return;
      concerns.push({ key: t.key, label: t.label, detail: t.tier.label.toLowerCase() });
    });
  ranked
    .filter((s) => (tierRank(s.tier) ?? 9) <= TIER_RANK.weak && !concerns.some((c) => c.key === s.key))
    .slice(0, Math.max(0, 3 - concerns.length))
    .forEach((s) => {
      concerns.push({ key: `tier-${s.key}`, label: s.label, detail: s.tier.label.toLowerCase() });
    });

  return {
    strengths: strengths.slice(0, 3),
    concerns: concerns.slice(0, 3),
  };
}

function roleFromPeak(peakValue, pos) {
  if (peakValue == null) return null;
  const p = String(pos || "").toUpperCase();
  let role;
  if (peakValue >= 92) role = "Franchise";
  else if (peakValue >= 88) role = "Elite";
  else if (peakValue >= 84) role = "Top line";
  else if (peakValue >= 80) role = "Top-6";
  else if (peakValue >= 75) role = "Mid-6";
  else if (peakValue >= 70) role = "Bottom-6";
  else role = "AHL";
  if (p === "G" || p.includes("GOAL")) {
    if (role === "Elite" || role === "Top line") return "Starter";
    if (role === "Top-6") return "Tandem";
    if (role === "Mid-6" || role === "Bottom-6" || role === "AHL") return "AHL";
  } else if (p === "D" || p === "LD" || p === "RD" || p.includes("DEF")) {
    if (role === "Top line") return "Top pair";
    if (role === "Top-6") return "Top-4";
    if (role === "Mid-6") return "Bottom pair";
  }
  return role;
}

function readinessLabel(odds, ovrNow, age) {
  if (ovrNow == null && odds == null) return null;
  if (odds != null && odds >= 85 && ovrNow != null && ovrNow >= 80) {
    if (age != null && age <= 18 && ovrNow < 88) return "1 year away";
    return "NHL ready";
  }
  if (ovrNow != null && ovrNow >= 76) return "Close";
  if (age != null && age <= 18) return "Project";
  if (odds != null && odds >= 50) return "In the mix";
  return "Long shot";
}

function outcomesOf(profile) {
  const dist = profile?.outcome_distribution || profile?.outcomeDistribution || null;
  const raw = Array.isArray(dist?.segments) ? dist.segments : Array.isArray(dist) ? dist : null;
  if (raw) {
    return raw
      .map((seg) => ({
        key: String(seg.key || seg.label || "outcome"),
        label: String(seg.label || seg.key || "Outcome"),
        pct: num(seg.pct ?? seg.weight ?? seg.w ?? seg.value ?? seg.probability),
      }))
      .filter((s) => s.pct != null && s.pct > 0);
  }
  if (dist && typeof dist === "object") {
    return Object.entries(dist)
      .filter(([k, v]) => k !== "label" && num(v) != null && num(v) > 0)
      .map(([k, v]) => ({ key: k, label: k.replace(/_/g, " "), pct: Number(v) }));
  }
  return [];
}

function posPhrase(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "RW" || p === "LW" || p === "W" || p === "F") return "winger";
  if (p === "C") return "center";
  if (p === "D" || p === "LD" || p === "RD" || p.includes("DEF")) return "defenseman";
  if (isGoaliePos(p)) return "goalie";
  return "prospect";
}

function humanize(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  return s
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function archetypeOf(player, profile) {
  return humanize(
    profile?.archetype?.label
    || (typeof profile?.archetype === "string" ? profile.archetype : null)
    || profile?.player_comparison?.archetype
    || player?.prospectRole
    || profile?.prospect_role,
  );
}

function buildSentence({ file, role, archetype, pos }) {
  const arch = archetype ? archetype.toLowerCase() : posPhrase(pos);
  const ceiling = role?.projected ? role.projected.toLowerCase() : null;
  const rawWho = ceiling && ceiling !== arch ? `${role.projected} ${arch}` : (archetype || role?.projected || arch || "This prospect");
  const who = rawWho.charAt(0).toUpperCase() + rawWho.slice(1);
  let text;
  if (file.pct <= 0) {
    text = `${who}; you've seen 0% of him — the ceiling is a guess.`;
  } else if (file.consensusPct != null && file.pct >= REVEAL_AT && file.pct > file.consensusPct + 8) {
    text = `Your file leads the league (${file.pct}% vs ${file.consensusPct}% viewings). ${who}.`;
  } else if (file.pct >= REVEAL_AT) {
    text = `${who}. Your file is in — these numbers are real.`;
  } else {
    text = `${who}. File ${file.pct}% — exact OVR unlocks at 72%.`;
  }
  return clipWords(text, 120);
}

function handedness(player, profile) {
  const raw = profile?.identity_badges?.handedness
    || profile?.handedness
    || player?.handedness
    || player?.shoots
    || player?.shot_side
    || player?.catches
    || "";
  const h = String(raw).trim().toLowerCase();
  if (!h) return null;
  const side = h.startsWith("l") ? "Left" : h.startsWith("r") ? "Right" : h;
  return isGoaliePos(player?.position || profile?.position) ? `Catches ${side.toLowerCase()}` : `Shoots ${side.toLowerCase()}`;
}

function heightLabel(player, profile) {
  const cm = num(player?.heightCm ?? profile?.height_cm);
  if (cm != null && cm > 0) {
    const inches = Math.round(cm / 2.54);
    return `${Math.floor(inches / 12)}'${inches % 12}"`;
  }
  const raw = profile?.identity_badges?.height || profile?.height || player?.height;
  return raw ? String(raw) : null;
}

function weightLabel(player, profile) {
  const raw = profile?.identity_badges?.weight || profile?.weight || player?.weight;
  const n = num(String(raw ?? "").replace(/[^\d.]/g, ""));
  if (n == null || n <= 0) return null;
  return `${Math.round(n)} lb`;
}

function seasonLine(player, profile, goalie) {
  const stats = profile?.stats || {};
  const gp = num(player?.gp ?? stats.games ?? player?.games);
  if (goalie) {
    const parts = [];
    if (gp != null && gp > 0) parts.push(`${Math.round(gp)} GP`);
    const sv = player?.savePct ?? stats.savePct ?? stats.save_pct;
    if (sv != null && sv !== "") parts.push(`${sv} SV%`);
    const gaa = player?.gaa ?? stats.gaa;
    if (gaa != null && gaa !== "") parts.push(`${gaa} GAA`);
    return parts.length ? parts.join(" · ") : null;
  }
  if (gp == null || gp <= 0) return null;
  const g = round(player?.goals ?? stats.goals) ?? 0;
  const a = round(player?.assists ?? stats.assists) ?? 0;
  const p = round(player?.points ?? stats.points);
  const pts = p == null ? g + a : p;
  return `${g}G-${a}A-${pts}P in ${Math.round(gp)} GP`;
}

function pctCell(label, value) {
  const n = num(value);
  if (n == null) return null;
  const shown = n <= 1.5 ? n * 100 : n;
  return { label, value: `${shown.toFixed(1)}%` };
}

function signedCell(label, value, digits) {
  const n = num(value);
  if (n == null) return null;
  const text = n > 0 ? `+${n.toFixed(digits)}` : n.toFixed(digits);
  return { label, value: text };
}

function numCell(label, value, digits = 0) {
  const n = num(value);
  if (n == null) return null;
  return { label, value: n.toFixed(digits) };
}

function buildAnalytics(player, profile, goalie) {
  const stats = profile?.stats || {};
  const a = {
    ...(player?.analytics && typeof player.analytics === "object" ? player.analytics : {}),
    ...(stats?.analytics && typeof stats.analytics === "object" ? stats.analytics : {}),
    ...(profile?.analytics && typeof profile.analytics === "object" ? profile.analytics : {}),
  };
  const gp = num(player?.gp ?? stats.games);
  const sampleThin = Boolean(profile?.sampleThin || stats?.sampleThin || (gp != null && gp > 0 && gp < 15));
  const cells = [];
  const push = (cell) => {
    if (cell && cell.value != null && cell.value !== "") cells.push(cell);
  };
  if (goalie) {
    push(gp != null && gp > 0 ? { label: "GP", value: String(Math.round(gp)) } : null);
    push(numCell("W", player?.wins ?? stats.wins, 0));
    push(player?.savePct != null ? { label: "SV%", value: String(player.savePct) } : numCell("SV%", stats.save_pct, 3));
    push(player?.gaa != null ? { label: "GAA", value: String(player.gaa) } : numCell("GAA", stats.gaa, 2));
    push(signedCell("GSAx", a.gsax ?? a.goals_saved_above_expected, 2));
    push(numCell("QS", a.quality_starts, 0));
    push(numCell("SO", player?.shutouts ?? stats.shutouts, 0));
  } else {
    const g = round(player?.goals ?? stats.goals);
    const ast = round(player?.assists ?? stats.assists);
    const pts = round(player?.points ?? stats.points);
    push(gp != null && gp > 0 ? { label: "GP", value: String(Math.round(gp)) } : null);
    if (gp != null && gp > 0 && (g != null || ast != null || pts != null)) {
      push({ label: "G-A-P", value: `${g ?? 0}-${ast ?? 0}-${pts ?? ((g ?? 0) + (ast ?? 0))}` });
    }
    const ppg = num(stats.ppg ?? player?.ppg ?? (gp > 0 && pts != null ? pts / gp : null));
    push(ppg != null && gp > 0 ? { label: "P/GP", value: ppg.toFixed(2) } : null);
    push(pctCell("SH%", a.shooting_pct ?? a.sh_pct));
    push(numCell("Prim P", a.primary_points ?? a.primary_pts, 0));
    push(signedCell("WAR", a.war ?? a.WAR ?? a.player_war, 2));
    push(pctCell("xGF%", a.xgf_pct ?? a.xGF_pct));
    const cf = num(a.cf_pct ?? a.corsi_pct);
    const xgf = num(a.xgf_pct ?? a.xGF_pct);
    const cfExplicit = a.cf_pct != null || a.corsi_pct != null || a.corsi_percentage != null;
    if (cf != null && (cfExplicit || xgf == null || cf !== xgf)) push(pctCell("CF%", cf));
    push(numCell("Shots/G", a.shot_rate ?? a.shots_per_game, 2));
    push(signedCell("+/-", player?.plusMinus ?? a.plus_minus ?? a.plusMinus, 0));
  }
  return { cells, sampleThin, gp: gp != null ? Math.round(gp) : null };
}

function buildReports(player, profile) {
  const desk = buildScoutingDeskEntries(player, profile, {
    gp: num(player?.gp ?? profile?.stats?.games) || 0,
    analytics: profile?.analytics || player?.analytics || null,
  });
  const entries = (desk?.entries || []).map((row, index) => ({
    key: `${row.scout || "scout"}-${index}`,
    scout: row.scout || "Scout",
    meta: row.meta || "",
    quote: row.locked ? null : (row.quote || null),
    grade: row.locked ? null : formatDeskGrade(row),
    locked: Boolean(row.locked),
  }));
  const returned = entries.filter((e) => !e.locked).length;
  return { entries, returned };
}

function buildRank(player, profile) {
  const board = round(
    player?.boardRank ?? player?.rank ?? player?.draft_rank ?? player?.draftRank ?? player?.scoutRank ?? profile?.rank,
  );
  const consensus = round(
    player?.publicRank ?? player?.consensus_rank ?? player?.consensusRank ?? player?.public_rank ?? profile?.consensus_rank,
  );
  const deltaRaw = num(player?.draftStock?.deltaRank ?? player?.weeklyDelta ?? player?.stock);
  const delta = deltaRaw == null || Math.round(deltaRaw) === 0 ? null : Math.round(deltaRaw);
  return {
    board: board != null && board > 0 ? board : null,
    consensus: consensus != null && consensus > 0 ? consensus : null,
    delta,
  };
}

function buildIdentity(player, profile, goalie) {
  const first = player?.firstName || player?.first_name || "";
  const last = player?.lastName || player?.last_name || "";
  const combined = `${first} ${last}`.trim();
  const name = combined || player?.prospect_name || player?.name || "Prospect";
  const pos = player?.position || profile?.position || profile?.identity_badges?.position || "—";
  const age = ageOf(player, profile);
  const nation = profile?.country || player?.country || player?.nationality || profile?.nationality || null;
  const club = profile?.team || player?.teamDisplay || player?.junior_team || player?.team || player?.teamName || null;
  const league = profile?.league || player?.leagueDisplay || player?.junior_league || player?.league || null;
  return {
    name,
    pos: String(pos),
    age,
    shoots: handedness(player, profile),
    height: heightLabel(player, profile),
    weight: weightLabel(player, profile),
    nation: nation || null,
    flagCode: resolveCountryCode(nation || player?.countryCode || player?.nat) || null,
    club: club || null,
    league: league || null,
    seasonLine: seasonLine(player, profile, goalie),
    archetype: archetypeOf(player, profile),
    overage: Boolean(player?.overager) || (age != null && age >= 19),
  };
}

/**
 * @param {object} player board row or entry-draft prospect
 * @param {object} profile scouting profile (or prospect.dossier)
 * @param {{ allProspects?: object[], userScoutPct?: number }} [options]
 */
export function buildProspectDossierModel(player, profile, options = {}) {
  const src = player || {};
  const prof = profile || {};
  const goalie = isGoaliePos(src.position || prof.position);
  const pct = filePctOf(src, prof, options.userScoutPct);
  const consensusPct = consensusOf(src, prof);
  const file = {
    pct,
    stage: stageFor(pct),
    nextUnlock: pct < REVEAL_AT ? { atPct: REVEAL_AT, unlocks: "exact OVR" } : null,
    consensusPct,
    certaintyLabel: certaintyTier(pct).label,
  };
  const ovr = buildOvr(src, prof, pct);
  const peak = buildPeak(src, prof, ovr);
  const skills = buildSkills(src, prof, pct, options.allProspects || [], goalie);
  const character = buildCharacter(prof, pct);
  const notes = buildNotes(skills, character);
  const age = ageOf(src, prof);
  const odds = num(prof?.potential?.nhl_probability ?? prof?.nhl_probability ?? src?.nhl_probability);
  const etaRaw = prof?.eta?.label || prof?.eta?.text || (typeof prof?.eta === "string" ? prof.eta : null) || prof?.nhl_eta || src?.nhlEta || null;
  const ovrNow = ovr.exact ? ovr.value : (ovr.low != null && ovr.low >= 80 ? ovr.low : ovr.value);
  let eta = etaRaw && !/unknown/i.test(String(etaRaw)) ? String(etaRaw) : null;
  const ready = readinessLabel(odds, ovrNow, age);
  if (!eta && ready === "NHL ready") eta = "Now";
  if (!eta && ready === "1 year away") eta = "1 year";
  const role = {
    projected: peak.hidden ? null : roleFromPeak(peak.high ?? peak.value, src.position || prof.position),
    readiness: ready,
    eta,
    nhlOdds: odds != null ? clamp(Math.round(odds), 0, 100) : null,
    outcomes: outcomesOf(prof),
  };
  const identity = buildIdentity(src, prof, goalie);
  const volatility = peak.hidden ? "" : String(prof?.developmentVolatility || src?.riskLabel || prof?.gem?.label || "");
  const rank = buildRank(src, prof);
  const model = {
    identity,
    file,
    ovr,
    peak,
    skills,
    tierOf,
    role,
    rank,
    risk: riskLevel(volatility, pct, peak),
    strengths: notes.strengths,
    concerns: notes.concerns,
    character,
    analytics: buildAnalytics(src, prof, goalie),
    reports: buildReports(src, prof),
    trajectory: weeklyTrajectoryPoints(prof, src),
    wjc: resolveWjcDossierBlock(prof, goalie),
    goalie,
  };
  const sentence = buildSentence({
    file,
    role,
    archetype: identity.archetype,
    pos: identity.pos,
  });
  return {
    ...model,
    sentence,
    display: {
      ovr: formatRating(ovr),
      peak: formatRating(peak),
      file: formatPct(file.pct),
      consensus: file.consensusPct == null ? null : formatPct(file.consensusPct),
      rank: formatRank(rank.board),
      delta: formatDelta(rank.delta),
      nhlOdds: role.nhlOdds == null ? "—" : formatPct(role.nhlOdds),
    },
  };
}

function assertModel(model, errors, label) {
  const { ovr, peak, skills } = model;
  if (ovr.low != null && ovr.value != null && ovr.high != null) {
    if (!(ovr.low <= ovr.value && ovr.value <= ovr.high)) errors.push(`${label}: ovr bounds`);
  }
  if (peak.high != null && peak.high > 99) errors.push(`${label}: peak cap`);
  if (peak.low != null && ovr.low != null && peak.low < ovr.low) errors.push(`${label}: peak.low < ovr.low`);
  skills.forEach((s) => {
    if (!s.exact && s.value != null) errors.push(`${label}: fogged ${s.key} exposes ${s.value}`);
  });
  if (model.sentence.length > 120) errors.push(`${label}: sentence length`);
}

/** Invariants for the reconciled Dalton file plus a revealed skater, a clamped peak, and a goalie. */
export function __selfTest() {
  const errors = [];
  const dalton = {
    id: "dalton",
    firstName: "Dalton",
    lastName: "Robichaud",
    position: "RW",
    age: 17,
    rank: 1,
    publicRank: 1,
    gp: 6,
    goals: 5,
    assists: 5,
    points: 10,
    draftStock: { deltaRank: 0 },
    chapterProfile: {
      chapters: {
        offence: { band: true, low: 74, high: 90 },
        defence: { band: true, low: 60, high: 72 },
        transition: 88,
        mental: { band: true, low: 70, high: 82 },
        physical: { band: true, low: 66, high: 78 },
        character: { band: true, low: 64, high: 80 },
      },
    },
  };
  const daltonProfile = {
    userScoutPct: 0,
    scout_confidence: 90,
    now_range: { low: 84, high: 84 },
    peak_range: { low: 94, high: 99 },
    scoutedOverall: 84,
    archetype: { label: "POWER FORWARD" },
    potential: { rating: 99, floor: 89, nhl_probability: 92 },
    strengths: ["Shot tool — Shot grades 92 — World-Class", "Vision tool — Vision grades 87 — Elite"],
    concerns: [],
    character_read: {
      headline: "Recognized leader",
      traits: [
        { label: "Work Ethic", tier: "Awful" },
        { label: "Coachability", tier: "Weak" },
      ],
    },
    development_trajectory: "Now 84 OVR · peak 99 · floor 89 · likely outcome franchise",
    developmentVolatility: "HIGH",
  };
  const peers = [dalton, { ...dalton, id: "b", chapterProfile: dalton.chapterProfile }, { ...dalton, id: "c", chapterProfile: dalton.chapterProfile }];
  const m = buildProspectDossierModel(dalton, daltonProfile, { allProspects: peers, userScoutPct: 0 });
  assertModel(m, errors, "dalton");
  if (!m.ovr.exact || m.ovr.value !== 84 || m.display.ovr !== "84") errors.push("dalton: collapsed ovr should be 84");
  if (m.peak.high !== 99 || m.peak.low !== 94) errors.push(`dalton: peak ${m.peak.low}-${m.peak.high}`);
  if (m.peak.floor != null) errors.push("dalton: floor leaked");
  if (m.display.peak !== "94\u201399") errors.push(`dalton: peak text ${m.display.peak}`);
  if (m.file.pct !== 0 || m.file.stage !== "UNSCOUTED") errors.push("dalton: file");
  if (m.file.consensusPct !== 90) errors.push("dalton: consensus");
  if (m.rank.delta != null) errors.push("dalton: zero delta");
  if (m.identity.seasonLine !== "5G-5A-10P in 6 GP") errors.push(`dalton: season ${m.identity.seasonLine}`);
  const character = m.skills.find((s) => s.key === "character");
  if (!character || character.exact || character.value != null) errors.push("dalton: character exact leak");
  const transition = m.skills.find((s) => s.key === "transition");
  if (!transition || transition.value != null) errors.push("dalton: unrevealed transition leak");
  const blob = [...m.strengths, ...m.concerns].map((n) => `${n.label} ${n.detail}`).join(" | ");
  if (/tool|shot grades|vision grades|skating/i.test(blob)) errors.push(`dalton: tool leak ${blob}`);
  if (!m.concerns.length) errors.push("dalton: empty concerns");
  if (m.role.readiness === "Developing" || /unknown/i.test(String(m.role.eta || ""))) errors.push("dalton: readiness");
  if (m.role.nhlOdds !== 92) errors.push("dalton: odds");

  const revealed = buildProspectDossierModel(
    { ...dalton, id: "rev", dedicatedScoutingPct: 80 },
    { ...daltonProfile, userScoutPct: 80, now_range: { low: 79, high: 89 } },
    { allProspects: peers, userScoutPct: 80 },
  );
  assertModel(revealed, errors, "revealed");
  if (!revealed.ovr.exact || revealed.ovr.value !== 84 || !revealed.ovr.revealed) errors.push("revealed: ovr");
  const revTransition = revealed.skills.find((s) => s.key === "transition");
  if (!revTransition?.exact || revTransition.value !== 88) errors.push("revealed: transition");

  const clamped = buildProspectDossierModel(
    { id: "cap", position: "C", age: 18, rank: 12 },
    {
      userScoutPct: 40,
      now_range: { low: 70, high: 78 },
      peak_range: { low: 96, high: 105 },
      potential: { rating: 105, floor: 91, nhl_probability: 50 },
    },
    { userScoutPct: 40 },
  );
  assertModel(clamped, errors, "clamp");
  if (clamped.peak.high !== 99) errors.push("clamp: high");
  if (clamped.ovr.exact) errors.push("clamp: wide ovr marked exact");
  if (clamped.display.ovr !== "70\u201378") errors.push(`clamp: ovr text ${clamped.display.ovr}`);

  const goalie = buildProspectDossierModel(
    {
      id: "g",
      firstName: "A",
      lastName: "Net",
      position: "G",
      age: 18,
      rank: 20,
      gp: 24,
      savePct: ".912",
      chapterProfile: { chapters: { glove: { low: 70, high: 80 }, blocker: 60, stick: { low: 55, high: 68 } } },
    },
    { userScoutPct: 10, scoutedOverall: 62, now_range: { low: 58, high: 66 }, potential: { nhl_probability: 40 } },
    { userScoutPct: 10 },
  );
  assertModel(goalie, errors, "goalie");
  if (!goalie.goalie || goalie.skills.length !== 3) errors.push("goalie: chapters");
  if (goalie.skills.some((s) => s.key === "blocker" && s.value != null)) errors.push("goalie: blocker leak");
  if (!goalie.identity.seasonLine || !goalie.identity.seasonLine.includes("24 GP")) errors.push("goalie: season");

  return { ok: errors.length === 0, errors };
}
