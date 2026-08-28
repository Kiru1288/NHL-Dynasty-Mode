/**
 * Draft War Room helpers — pyramid tiers, publications, overage/character
 * stock effects, weekly movers, and brochure-dossier copy.
 */

import { goalieStrengthCopy, goalieWeaknessCopy } from "./prospectDossierHelpers";

export const PYRAMID_TIERS = [
  { key: "transcendent", label: "Transcendent", rarity: "1 / 100,000", color: "#ffd24a", ink: "#1a1204", order: 1, minPeak: 96 },
  { key: "generational", label: "Generational", rarity: "Once a decade", color: "#ff6b2d", ink: "#1a0900", order: 2, minPeak: 93 },
  { key: "franchise", label: "Franchise", rarity: "Cornerstone", color: "#c084fc", ink: "#14081f", order: 3, minPeak: 90 },
  { key: "elite", label: "Elite", rarity: "Elite driver", color: "#22d3ee", ink: "#041318", order: 4, minPeak: 87 },
  { key: "star", label: "Star", rarity: "Top-line piece", color: "#34d399", ink: "#04140e", order: 5, minPeak: 84 },
  { key: "core", label: "Core", rarity: "Everyday NHLer", color: "#3b82f6", ink: "#06101f", order: 6, minPeak: 80 },
  { key: "depth", label: "Depth", rarity: "Bottom six / pair", color: "#8aa0b8", ink: "#0b1218", order: 7, minPeak: 76 },
  { key: "nhler", label: "NHLer", rarity: "Roster NHLer", color: "#64748b", ink: "#0b1016", order: 8, minPeak: 72 },
  { key: "longshot", label: "Longshot", rarity: "Long-term flyer", color: "#a78b6a", ink: "#140f08", order: 9, minPeak: 0 },
];

export const PYRAMID_BY_KEY = Object.fromEntries(PYRAMID_TIERS.map((t) => [t.key, t]));

export const PUBLICATIONS = [
  { id: "athletic", label: "The Athletic", author: "Staff board", bias: "analytics" },
  { id: "mckenzie", label: "Bob McKenzie", author: "TSN", bias: "consensus" },
  { id: "wheeler", label: "Scott Wheeler", author: "The Athletic", bias: "production" },
  { id: "pronman", label: "Corey Pronman", author: "The Athletic", bias: "tools" },
];

export const WORKSPACE_TABS = [
  { id: "board", label: "Scout Board" },
  { id: "stats", label: "Stats" },
];

export const BOARD_SOURCES = [
  { id: "scout", label: "Our Scouts" },
  { id: "central", label: "Central Scouting" },
  { id: "athletic", label: "The Athletic" },
  { id: "mckenzie", label: "Bob McKenzie" },
  { id: "wheeler", label: "Scott Wheeler" },
  { id: "pronman", label: "Corey Pronman" },
];

const CHARACTER_INCIDENTS = [
  {
    id: "locker_fight",
    title: "Locker-room haymaker",
    stockHit: 14,
    hitPctHit: 11,
    story: (name) =>
      `${name} swung a Bauer bag at a linemate after a missed empty-netter in Kamloops. The bag won. Teammates now call him Carry-On, and two NHL clubs quietly moved him off their top-20.`,
  },
  {
    id: "hot_head",
    title: "Hot head vs. the clock",
    stockHit: 10,
    hitPctHit: 8,
    story: (name) =>
      `${name} earned a misconduct arguing with a timekeeper who said the period was over. It was. Scouts noted the compete; GMs noted the lawyer fees.`,
  },
  {
    id: "diva",
    title: "Walk-up-song standoff",
    stockHit: 12,
    hitPctHit: 10,
    story: (name) =>
      `${name} refused a defensive-zone faceoff until the PA played his walk-up song — a 47-second TikTok remix of his own last name. Coach sat him. Twitter did not.`,
  },
  {
    id: "curfew",
    title: "Billet curfew, nacho edition",
    stockHit: 8,
    hitPctHit: 6,
    story: (name) =>
      `${name} was clocked sneaking into the billet house at 2:14 a.m. with a family-size nacho tray and a goldfish in a souvenir cup. The goldfish is fine. The development curve is not.`,
  },
  {
    id: "bus_nickelback",
    title: "Ninety-minute Nickelback set",
    stockHit: 6,
    hitPctHit: 5,
    story: (name) =>
      `After a 1–8 road swing, ${name} led the team bus in a 90-minute Nickelback set. Leadership, technically. Several scouts asked if “presence” can be a red flag.`,
  },
  {
    id: "gps_own_goal",
    title: "GPS caption, own-goal follow-through",
    stockHit: 9,
    hitPctHit: 7,
    story: (name) =>
      `${name} posted “refs in this league couldn’t find the net with GPS,” then scored an own-goal that night. Analytics still like the shot volume. The room does not.`,
  },
  {
    id: "stick_rack",
    title: "Three-stick tantrum",
    stockHit: 11,
    hitPctHit: 9,
    story: (name) =>
      `${name} snapped three sticks on the bench after a shift, then asked the trainer for a fourth. Trainer said no. ${name} said, quote, “this is why we don’t make playoffs.” They were 8 points up in the standings.`,
  },
  {
    id: "stipend_diva",
    title: "Stipend hierarchy",
    stockHit: 13,
    hitPctHit: 10,
    story: (name) =>
      `${name} told a linemate he wouldn’t pass to anyone making less than him. He is on a $50 weekly stipend. The linemate has 18 more points.`,
  },
  {
    id: "hotel_iron",
    title: "Hotel iron incident",
    stockHit: 7,
    hitPctHit: 6,
    story: (name) =>
      `Team staff found ${name} ironing a playoff beard onto a pillowcase at 1 a.m. “Visualization,” he called it. The hotel called it a fire hazard. Scouts called it a character follow-up.`,
  },
  {
    id: "ref_dad",
    title: "Asked if dad was working",
    stockHit: 8,
    hitPctHit: 7,
    story: (name) =>
      `${name} asked the referee if his dad was working the game — the referee’s dad was in the stands. In a suit. As the league’s discipline chair. Combine interviews got spicier.`,
  },
];

export function hashSeed(input) {
  const s = String(input ?? "");
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h >>> 0);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

export function prospectAge(player) {
  const n = Number(player?.age);
  return Number.isFinite(n) ? n : 18;
}

export function isOverager(player) {
  return prospectAge(player) >= 19;
}

export function overageRankPenalty(player) {
  const age = prospectAge(player);
  if (age <= 18) return 0;
  if (player?.isTranscendent && age === 19) return 4;
  if (age === 19) return 18;
  if (age === 20) return 34;
  return 52;
}

export function overageStockNote(player) {
  if (!isOverager(player)) return null;
  const age = prospectAge(player);
  if (age >= 20) {
    return `Overager (${age}) — second-year eligible. NHL clubs almost never take this profile inside the top 20; the extra year of dominance is priced in, and the runway is shorter.`;
  }
  return `Overager (${age}) — first extra year of junior. Production is real, but draft stock takes a major hit versus true 17–18-year-olds in the same range.`;
}

function peakFromPlayer(player) {
  const score = Number(player?.potentialScore ?? player?.potential?.rating ?? player?.profile?.potential?.rating);
  if (Number.isFinite(score) && score > 0) return score;
  const lo = Number(player?.potentialRange?.low);
  const hi = Number(player?.potentialRange?.high);
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo) return (lo + hi) / 2;
  const rank = Number(player?.publicRank ?? player?.rank) || 200;
  if (rank <= 1) return 95;
  if (rank <= 3) return 92;
  if (rank <= 8) return 88;
  if (rank <= 16) return 85;
  if (rank <= 32) return 82;
  if (rank <= 64) return 78;
  if (rank <= 96) return 74;
  return 70;
}

export function resolvePyramidTier(player) {
  if (player?.isTranscendent) return PYRAMID_BY_KEY.transcendent;
  const peak = peakFromPlayer(player);
  const rank = Number(player?.publicRank ?? player?.scoutRank ?? player?.rank) || 999;
  let pick = PYRAMID_TIERS[PYRAMID_TIERS.length - 1];
  for (const tier of PYRAMID_TIERS) {
    if (peak >= tier.minPeak) {
      pick = tier;
      break;
    }
  }
  // Rank cap: a 4th-rounder with a juicy ceiling still isn't a franchise pick on the board.
  if (pick.key === "transcendent" && rank > 2) pick = PYRAMID_BY_KEY.generational;
  if (pick.key === "generational" && rank > 5) pick = PYRAMID_BY_KEY.franchise;
  if (pick.key === "franchise" && rank > 12) pick = PYRAMID_BY_KEY.elite;
  if (pick.key === "elite" && rank > 24) pick = PYRAMID_BY_KEY.star;
  if (isOverager(player) && pick.order <= 4) {
    pick = PYRAMID_TIERS[Math.min(PYRAMID_TIERS.length - 1, pick.order)] || pick;
  }
  return pick;
}

function pickIncident(player) {
  const idx = hashSeed(`${player?.id || player?.name}-incident`) % CHARACTER_INCIDENTS.length;
  return CHARACTER_INCIDENTS[idx];
}

export function resolveCharacterFile(player, profile = null) {
  const charRead = profile?.character_read || null;
  const forced = Boolean(
    player?.characterConcerns
    || player?.character_concerns
    || profile?.character_concerns
    || player?.isBustRisk,
  );
  const headline = charRead?.headline || null;
  const interviewNotes = charRead?.interview_notes || null;
  const traits = Array.isArray(charRead?.traits) ? charRead.traits : [];
  const weakTraits = traits.filter((t) => {
    const tier = String(t?.tier || "").toLowerCase();
    return tier === "below average" || tier === "mixed reports" || tier === "unknown";
  });
  // Only flag when the backend explicitly marks character concerns — not from trait tiers alone.
  const flagged = forced;

  const traitSummary = traits.length
    ? traits.slice(0, 3).map((t) => `${t.label}: ${t.tier}`).join(" · ")
    : null;
  const story = interviewNotes
    || traitSummary
    || (headline ? `Character read — ${headline}.` : null)
    || (forced ? "Character concerns flagged on the central scouting file." : "No material character flags in the current report.");

  if (!flagged && !forced) {
    return {
      flagged: false,
      title: headline || "Clean file",
      story,
      stockHit: 0,
      hitPctHit: 0,
    };
  }

  const concernTitle = weakTraits[0]?.label
    ? `${weakTraits[0].label} — ${weakTraits[0].tier}`
    : (headline || "Character follow-up");
  return {
    flagged: true,
    title: concernTitle,
    story,
    stockHit: forced ? 12 : 6,
    hitPctHit: forced ? 9 : 5,
    id: forced ? "backend_flag" : "character_read",
  };
}

export function classIsWeak(prospects) {
  const list = Array.isArray(prospects) ? prospects : [];
  const topPeak = Math.max(0, ...list.slice(0, 8).map((p) => peakFromPlayer(p)));
  const hasStarPower = list.some((p) => p?.isTranscendent || peakFromPlayer(p) >= 93);
  return list.length > 0 && !hasStarPower && topPeak < 92;
}

function weeklyDeltaFor(player, weakClass) {
  const stock = player?.draftStock || {};
  let delta = Number(stock.deltaRank ?? player?.stock) || 0;
  const heat = Number(stock.stockHeat);
  if ((!delta || delta === 0) && Number.isFinite(heat) && heat !== 0) delta = heat;
  const publicRank = Number(player?.publicRank ?? player?.rank) || 999;
  if (publicRank === 1 && delta < 0 && !weakClass) {
    delta = 0;
  }
  const char = player?.characterFile;
  if (char?.flagged) delta -= Math.round(char.stockHit / 3);
  if (isOverager(player) && publicRank <= 20) delta -= 4;
  return delta;
}

export function enrichProspectsForWarRoom(prospects, profilesById = {}) {
  const list = Array.isArray(prospects) ? prospects : [];
  const weak = classIsWeak(list);
  const withFiles = list.map((p) => {
    const profile = p.profile || profilesById[p.id] || null;
    const characterFile = resolveCharacterFile(p, profile);
    const publicRank = Number(p.rank) || 999;
    const age = prospectAge(p);
    const charPenalty = characterFile.flagged ? characterFile.stockHit : 0;
    const scoutScore = publicRank + overageRankPenalty(p) + charPenalty;
    return {
      ...p,
      profile,
      publicRank,
      scoutScore,
      age,
      overager: age >= 19,
      characterFile,
      rankHistory: p.rankHistory || profile?.rankHistory || profile?.stock_history || p.stockHistory || [],
      preseasonRank: p.preseasonRank ?? profile?.preseasonRank ?? null,
      midseasonRank: p.midseasonRank ?? profile?.midseasonRank ?? null,
    };
  });

  const scoutOrder = [...withFiles].sort((a, b) => a.scoutScore - b.scoutScore || a.publicRank - b.publicRank);
  const scoutRankById = new Map(scoutOrder.map((p, i) => [p.id, i + 1]));

  return withFiles.map((p) => {
    const scoutRank = scoutRankById.get(p.id) || p.publicRank;
    const weeklyDelta = weeklyDeltaFor({ ...p, publicRank: p.publicRank }, weak);
    const pyramidTier = resolvePyramidTier({ ...p, rank: scoutRank, publicRank: p.publicRank });
    const draftStock = {
      ...(p.draftStock || {}),
      deltaRank: weeklyDelta,
      direction: weeklyDelta > 0 ? "UP" : weeklyDelta < 0 ? "DOWN" : "STABLE",
      stockUnit: "rank",
      stockMode: "weekly_heat",
      label: weeklyDelta > 0 ? `Wk +${weeklyDelta}` : weeklyDelta < 0 ? `Wk ${weeklyDelta}` : "Wk 0",
      available: true,
    };
    const publicRank = Number(p.publicRank) || 999;
    const delta = scoutRank - publicRank;
    const boardDivergence = Math.abs(delta) >= 5
      ? { scoutRank, publicRank, delta }
      : null;
    const pos = String(p.position || "").toUpperCase();
    const displayedCeiling = Number(p.potentialScore ?? p.potentialRange?.high ?? p.potentialRange?.low);
    const ceilingVal = Number.isFinite(displayedCeiling) && displayedCeiling > 0
      ? displayedCeiling
      : peakFromPlayer(p);
    const goalieBoardCap = pos === "G" && publicRank >= 33 && (
      ceilingVal >= 80 || Boolean(p.ceilingHidden)
    );
    let consensusFloorApplied = false;
    if (publicRank >= 1 && publicRank <= 32) {
      const floor = 86 - (publicRank - 1) * (86 - 70) / 31;
      const shown = Number.isFinite(displayedCeiling) && displayedCeiling > 0
        ? displayedCeiling
        : null;
      if (shown != null && shown <= floor + 0.55 && shown >= floor - 0.15) {
        consensusFloorApplied = true;
      }
    }
    return {
      ...p,
      scoutRank,
      pyramidTier,
      draftStock,
      weeklyDelta,
      characterConcerns: Boolean(p.characterConcerns || p.character_concerns),
      boardDivergence,
      goalieBoardCap,
      consensusFloorApplied,
    };
  });
}

export function publicationScore(player, pubId) {
  const base = Number(player?.publicRank ?? player?.rank) || 200;
  const peak = peakFromPlayer(player);
  const skating = Number(player?.skating) || 60;
  const shooting = Number(player?.shooting) || 60;
  const iq = Number(player?.hockeyIQ) || 60;
  const ppg = Number(player?.ppg) || 0;
  const war = Number(player?.analytics?.war ?? player?.profile?.analytics?.war) || 0;
  const xgf = Number(player?.analytics?.xgf_pct ?? player?.profile?.analytics?.xgf_pct) || 50;
  const jitter = (hashSeed(`${pubId}-${player?.id}`) % 9) - 4;
  let score = base;
  if (pubId === "mckenzie") {
    score = base + jitter * 0.35;
  } else if (pubId === "pronman") {
    score = base - (skating + shooting + iq - 210) / 12 + jitter;
    if (isOverager(player)) score += 8;
  } else if (pubId === "wheeler") {
    score = base - ppg * 6 - (peak - 80) * 0.25 + jitter;
    if (isOverager(player)) score += 10;
  } else if (pubId === "athletic") {
    score = base - war * 3 - (xgf - 50) / 8 + jitter;
  }
  if (player?.characterFile?.flagged) score += player.characterFile.stockHit * 0.45;
  return score;
}

export function rankProspectsForSource(prospects, sourceId) {
  const list = Array.isArray(prospects) ? prospects : [];
  const copy = [...list];
  if (sourceId === "scout") {
    copy.sort((a, b) => (a.scoutRank || a.rank) - (b.scoutRank || b.rank));
    return copy.map((p, i) => ({ ...p, boardRank: i + 1, rank: p.scoutRank || i + 1 }));
  }
  if (sourceId === "central") {
    copy.sort((a, b) => (Number(a.centralRank) || a.publicRank || 999) - (Number(b.centralRank) || b.publicRank || 999));
    return copy.map((p, i) => ({ ...p, boardRank: i + 1, rank: Number(p.centralRank) || i + 1 }));
  }
  if (sourceId === "consensus") {
    copy.sort((a, b) => (a.publicRank || 999) - (b.publicRank || 999));
    return copy.map((p, i) => ({ ...p, boardRank: i + 1, rank: p.publicRank || i + 1 }));
  }
  copy.sort((a, b) => publicationScore(a, sourceId) - publicationScore(b, sourceId));
  return copy.map((p, i) => ({ ...p, boardRank: i + 1, rank: i + 1 }));
}

export function groupByPyramid(prospects) {
  const groups = [];
  const seen = new Map();
  (prospects || []).forEach((p) => {
    const tier = p.pyramidTier || resolvePyramidTier(p);
    const key = tier.key;
    if (!seen.has(key)) {
      const g = { ...tier, prospects: [] };
      seen.set(key, g);
      groups.push(g);
    }
    seen.get(key).prospects.push(p);
  });
  return groups.sort((a, b) => a.order - b.order);
}

export function weeklyStockMovers(prospects, { minAbs = 1, limit = 80 } = {}) {
  const rows = (prospects || [])
    .map((p) => {
      const delta = Number(p.weeklyDelta ?? p.draftStock?.deltaRank) || 0;
      return {
        key: p.id,
        name: `${p.firstName || ""} ${p.lastName || ""}`.trim() || p.name || "Unknown",
        rank: Number(p.scoutRank ?? p.rank) || 0,
        publicRank: Number(p.publicRank ?? p.rank) || 0,
        delta,
        overager: Boolean(p.overager),
        tier: p.pyramidTier?.key || "",
      };
    })
    .filter((r) => Math.abs(r.delta) >= minAbs)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.rank - b.rank);
  return rows.slice(0, limit);
}

export function confMeaning(pct) {
  if (pct == null || !Number.isFinite(Number(pct))) {
    return { band: "No file", detail: "Our scouts have not opened a dedicated file yet." };
  }
  const n = Math.round(Number(pct));
  if (n >= 91) return { band: "Locked read", detail: `${n}% file — in-person, video, and interviews agree. Ceiling is no longer a guess.` };
  if (n >= 76) return { band: "In file", detail: `${n}% — regional scout plus video. Projection is usable; late-season looks still matter.` };
  if (n >= 56) return { band: "Regional coverage", detail: `${n}% file — enough looks to rank him, not enough to bet a top pick without another viewing.` };
  if (n >= 36) return { band: "Limited looks", detail: `${n}% file — mostly video / boxcars. Treat the number as a range, not a grade.` };
  if (n >= 15) return { band: "First look", detail: `${n}% file — one or two viewings. Do not draft off this.` };
  return { band: "No file", detail: `${n}% — name on a list, not a scouting report.` };
}

function isGoaliePlayer(player) {
  const p = String(player?.position || "").toUpperCase();
  return p === "G" || p.includes("GOAL");
}

export function skillDevelopmentNotes(tools, player) {
  const goalie = isGoaliePlayer(player);
  const rows = (tools || []).map((t) => {
    const mid = Number(t.raw != null ? t.raw : t.mid);
    const weak = Number.isFinite(mid) && mid < 70;
    const plus = Number.isFinite(mid) && mid >= 82;
    let reach = "Average translation";
    if (t.locked || mid == null) reach = "Unscouted";
    else if (goalie && weak && t.label === "Positioning") reach = "Development drag — angles are the first thing that fail at pro pace";
    else if (goalie && weak && t.label === "Rebound") reach = "Development drag — second chances decide goalie careers";
    else if (!goalie && weak && t.label === "Skating") reach = "Development drag — pace is the first thing that fails at pro speed";
    else if (weak) reach = "Needs a real jump to hold this projection";
    else if (plus) reach = "Already a carrying tool";
    else reach = "On-track if the work stays honest";
    return { ...t, weak, plus, reach };
  });
  const skating = rows.find((t) => t.label === "Skating");
  const positioning = rows.find((t) => t.label === "Positioning");
  const overager = isOverager(player);
  const char = player?.characterFile;
  let developOdds = 62;
  if (goalie) {
    if (positioning?.weak) developOdds -= 12;
    if (rows.find((t) => t.label === "Rebound")?.weak) developOdds -= 10;
  } else if (skating?.weak) developOdds -= 14;
  if (overager) developOdds -= 16;
  if (char?.flagged) developOdds -= char.hitPctHit || 8;
  if ((Number(player?.workEthic) || 0) >= 80) developOdds += 8;
  if ((Number(player?.coachability) || 0) >= 80) developOdds += 6;
  if (prospectAge(player) <= 17) developOdds += 7;
  developOdds = clamp(developOdds, 12, 92);
  return {
    rows,
    developOdds,
    skatingWeak: Boolean(goalie ? positioning?.weak : skating?.weak),
  };
}

export function projectionOutcomes(player, profile, developOdds = 60) {
  const pot = profile?.potential && typeof profile.potential === "object" ? profile.potential : {};
  const range = player?.potentialRange || pot.range || null;
  const hidden = Boolean(profile?.ceilingHidden || pot.hidden || player?.ceilingHidden);
  if (hidden) {
    return {
      peak: null,
      expected: null,
      worst: null,
      shootPastPct: null,
      hitPeakPct: null,
      nhlOdds: null,
      band: null,
      source: "fog",
      note: "Ceiling still fogged — assign a scout before treating these as real numbers.",
    };
  }

  const rating = Number(pot.rating ?? player?.potentialScore);
  const floorRaw = Number(pot.floor ?? player?.floorScore);
  const lo = Number(range?.low);
  const hi = Number(range?.high);
  const hasLo = Number.isFinite(lo) && lo > 0;
  const hasHi = Number.isFinite(hi) && hi > 0;
  const hasRating = Number.isFinite(rating) && rating > 0;
  const hasFloor = Number.isFinite(floorRaw) && floorRaw > 0;

  if (!hasRating && !hasHi && !hasLo && !hasFloor) {
    return {
      peak: null,
      expected: null,
      worst: null,
      shootPastPct: null,
      hitPeakPct: null,
      nhlOdds: Number(pot.nhl_probability ?? pot.probability) || null,
      band: pot.band || pot.label || null,
      source: "missing",
      note: "No backend ceiling on this file yet.",
    };
  }

  const peak = Math.round(hasHi ? hi : (hasRating ? rating : (hasFloor ? floorRaw + 8 : 0)));
  const worst = Math.round(hasFloor ? floorRaw : (hasLo ? lo : Math.max(58, peak - 10)));
  const mid = hasLo && hasHi ? (lo + hi) / 2 : (hasRating ? rating : (worst + peak) / 2);
  let expected = Math.round(hasRating && (!hasHi || rating <= hi) ? rating : mid);
  expected = clamp(expected, Math.min(worst, peak), Math.max(worst, peak));
  if (expected === peak && peak - worst >= 3) expected = peak - Math.max(1, Math.round((peak - worst) * 0.35));
  if (expected === worst && peak - worst >= 3) expected = worst + Math.max(1, Math.round((peak - worst) * 0.4));

  const nhlOdds = Number(pot.nhl_probability ?? pot.probability);
  const band = pot.band || pot.label || player?.potentialLabel || null;
  let shootPast = Number.isFinite(nhlOdds) ? Math.round(Math.max(2, (100 - nhlOdds) * 0.12)) : Math.round(clamp(developOdds / 8, 2, 18));
  if (String(band).toLowerCase().includes("boom")) shootPast += 8;
  if (isOverager(player)) shootPast -= 6;
  if (player?.characterFile?.flagged) shootPast -= 5;
  shootPast = clamp(shootPast, 1, 22);

  const hitPeakPct = Number.isFinite(nhlOdds)
    ? clamp(Math.round(nhlOdds * 0.55), 8, 72)
    : clamp(developOdds - 8, 10, 70);

  return {
    peak,
    expected,
    worst: Math.min(worst, expected),
    shootPastPct: shootPast,
    hitPeakPct,
    nhlOdds: Number.isFinite(nhlOdds) ? Math.round(nhlOdds) : null,
    band,
    source: "backend",
    note: null,
  };
}

export function strengthCopy(player, tools) {
  if (isGoaliePlayer(player)) return goalieStrengthCopy(player, tools);
  const name = `${player?.firstName || ""} ${player?.lastName || ""}`.trim() || "He";
  const plus = (tools || []).filter((t) => t.plus).map((t) => t.label.toLowerCase());
  if (player?.completion < 40) return [`${name} is still a first-look file — strengths are rumours until we get another viewing.`];
  const lines = [];
  if (plus.includes("skating")) lines.push("Separates in open ice; the first three strides are already NHL-fast.");
  if (plus.includes("shot")) lines.push("Release jumps on goalies. Slot and the circle are both dangerous.");
  if (plus.includes("vision") || plus.includes("iq")) lines.push("Sees the second option before the first one dies. Makes linemates honest.");
  if (plus.includes("defense")) lines.push("Detail away from the puck is ahead of the age curve — kills plays, not just time.");
  if (plus.includes("physical")) lines.push("Wins walls and net-fronts without turning into a passenger after contact.");
  if (!lines.length) {
    lines.push("Projectable frame and enough processing to stay in the lineup conversation.");
    lines.push("Habits in transition are already usable; the carrying tool is still arriving.");
  }
  return lines.slice(0, 4);
}

export function weaknessCopy(player, tools, skatingWeak) {
  if (isGoaliePlayer(player)) return goalieWeaknessCopy(player, tools);
  const name = `${player?.firstName || ""} ${player?.lastName || ""}`.trim() || "He";
  if (player?.completion < 40) return [`More looks needed before we tattoo a weakness onto ${name}.`];
  const weak = (tools || []).filter((t) => t.weak).map((t) => t.label.toLowerCase());
  const lines = [];
  if (skatingWeak) lines.push("Skating is the tell. If the pace doesn’t jump, the rest of the tools play a league down.");
  if (weak.includes("shot")) lines.push("Shot selection gets greedy; goalies see it too early.");
  if (weak.includes("defense")) lines.push("Defensive reads still chase the puck instead of the next play.");
  if (weak.includes("physical")) lines.push("Gets moved. Pro net-fronts will not be polite about it.");
  if (weak.includes("vision") || weak.includes("iq")) lines.push("Forces seams that aren’t there; turnovers arrive dressed as ambition.");
  if (isOverager(player)) lines.push("The extra year of junior inflates the boxcars. Same production at 17 would be a different conversation.");
  if (!lines.length) {
    lines.push("Needs pro pace and a thicker second effort before the projection is safe.");
  }
  return lines.slice(0, 4);
}

export function weeklyTrajectoryPoints(profile, player) {
  const raw = profile?.rankHistory
    || profile?.stock_history
    || player?.rankHistory
    || player?.stockHistory
    || [];
  const points = [];
  const push = (rank, label, value) => {
    const r = Number(rank);
    if (!Number.isFinite(r) || r <= 0) return;
    points.push({
      x: points.length,
      rank: r,
      value: value != null && Number.isFinite(Number(value)) ? Number(value) : null,
      label: String(label || `Wk ${points.length + 1}`),
    });
  };

  if (Array.isArray(raw) && raw.length) {
    raw.forEach((entry, i) => {
      if (entry == null) return;
      if (typeof entry === "number" && Number.isFinite(entry)) {
        push(entry, `Wk ${i + 1}`);
        return;
      }
      if (typeof entry !== "object") return;
      const rank = Number(entry.rank ?? entry.public_rank ?? entry.board_rank ?? entry.central_rank ?? entry.value);
      const label = entry.date_label || entry.label || entry.event || entry.phase || entry.date || `Wk ${i + 1}`;
      const heat = entry.stock_heat ?? entry.stockHeat ?? entry.delta ?? entry.delta_rank;
      push(rank, label, heat);
    });
  }

  const current = Number(
    player?.scoutRank ?? player?.rank ?? profile?.currentRank ?? profile?.rank ?? 0
  );
  const weekly = Number(player?.weeklyDelta ?? player?.draftStock?.deltaRank) || 0;
  const pre = Number(profile?.preseasonRank ?? player?.preseasonRank);
  const mid = Number(profile?.midseasonRank ?? player?.midseasonRank);

  const uniqueRanks = new Set(points.map((p) => p.rank));
  const usable = points.filter((p) => Number.isFinite(p.rank) && p.rank > 0);
  if (usable.length < 3 || uniqueRanks.size <= 1) {
    const start = Number.isFinite(pre) && pre > 0
      ? pre
      : Math.max(1, Math.round(current + (weekly !== 0 ? weekly * 2 : (current > 8 ? 6 : 2))));
    const middle = Number.isFinite(mid) && mid > 0
      ? mid
      : Math.max(1, Math.round((start + current) / 2));
    const weekAgo = Math.max(1, Math.round(current + (weekly !== 0 ? weekly : (start > current ? 1 : -1))));
    const now = Math.max(1, Math.round(current || start));
    return [
      { x: 0, rank: start, label: "Preseason" },
      { x: 1, rank: middle, label: "Midseason" },
      { x: 2, rank: weekAgo, label: "Last week" },
      { x: 3, rank: now, label: "This week" },
    ];
  }

  if (points[0]) points[0] = { ...points[0], label: points[0].label || "Preseason" };
  const last = points[points.length - 1];
  if (last) points[points.length - 1] = { ...last, rank: current || last.rank, label: "This week" };
  if (weekly !== 0 && points.length >= 2) {
    const prev = points[points.length - 2];
    points[points.length - 2] = { ...prev, rank: Math.max(1, Math.round((current || prev.rank) + weekly)), label: prev.label || "Last week" };
  }
  return points.map((p, i) => ({ ...p, x: i }));
}

export function sourceCaption(sourceId) {
  if (sourceId === "scout") return "Our scouts";
  if (sourceId === "central") return "NHL Central Scouting";
  if (sourceId === "consensus") return "Public consensus (reference only)";
  const pub = PUBLICATIONS.find((p) => p.id === sourceId);
  return pub ? `${pub.label} board` : "Draft board";
}
