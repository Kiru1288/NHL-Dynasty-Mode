/**
 * Position-aware prospect dossier content — archetypes, zone grades, copy, scouting desk.
 */

import { hashSeed, isOverager } from "./draftWarRoom";

export const GOALIE_ARCHETYPES = [
  { key: "long_levered_wall", label: "LONG-LEVERED WALL", blurb: "Wins with reach and depth. The puck still finds seams he hasn't learned to close." },
  { key: "athletic_stopper", label: "ATHLETIC STOPPER", blurb: "Covers ground in a hurry. Lives on reflex saves when structure breaks down." },
  { key: "positional_anchor", label: "POSITIONAL ANCHOR", blurb: "Square and patient. Beats you with angles before the desperation save." },
  { key: "hybrid_tracker", label: "HYBRID TRACKER", blurb: "Blends butterfly economy with stand-up reads. Still choosing his default mode." },
  { key: "puck_moving_netminder", label: "PUCK-MOVING NETMINDER", blurb: "Acts like a third defenseman. The risk is when he handles under pressure." },
  { key: "project_stopper", label: "PROJECT STOPPER", blurb: "Frame and compete grade ahead of the technical package." },
];

export const SKATER_ARCHETYPES = {
  F: [
    { key: "transition_playdriver", label: "TRANSITION PLAYDRIVER", blurb: "Carries the puck out. Everything starts on his stick." },
    { key: "two_way_forward", label: "TWO-WAY FORWARD", blurb: "Details at both ends. The offense is steady, not explosive." },
    { key: "power_forward", label: "POWER FORWARD", blurb: "Wins inside ice and finishes through contact." },
    { key: "sniper", label: "SNIPER", blurb: "Release is the weapon. Needs volume and lanes to stay dangerous." },
    { key: "playmaker", label: "PLAYMAKER", blurb: "Slows the game down and feeds the second layer." },
  ],
  D: [
    { key: "transition_playdriver", label: "TRANSITION PLAYDRIVER", blurb: "Carries the puck out. Everything starts on his stick." },
    { key: "offensive_d", label: "OFFENSIVE DEFENSEMAN", blurb: "Quarterbacks from the blue line. Defense is still catching up." },
    { key: "shutdown_d", label: "SHUTDOWN DEFENDER", blurb: "Kills plays early. The puck-moving layer is still thin." },
    { key: "two_way_d", label: "TWO-WAY DEFENSEMAN", blurb: "Balanced usage profile with no glaring hole." },
    { key: "mobile_d", label: "MOBILE DEFENDER", blurb: "Skating carries the projection until the physical game arrives." },
  ],
  default: [
    { key: "transition_playdriver", label: "TRANSITION PLAYDRIVER", blurb: "Carries the puck out. Everything starts on his stick." },
    { key: "two_way_forward", label: "TWO-WAY FORWARD", blurb: "Details at both ends. The offense is steady, not explosive." },
  ],
};

export const GOALIE_PLAY_STYLES = ["Butterfly", "Hybrid", "Athletic", "Stand-up", "Puck-moving"];
export const SKATER_PLAY_STYLES = {
  F: ["North-south", "East-west", "Power", "Skill", "Two-way"],
  D: ["Puck-moving", "Shutdown", "Transition", "Offensive", "Physical"],
  default: ["Two-way", "Skill", "Power"],
};

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function posBucket(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "G" || p.includes("GOAL")) return "G";
  if (p === "D" || p === "LD" || p === "RD" || p.includes("DEF")) return "D";
  if (p === "C") return "C";
  if (p === "LW" || p === "RW" || p === "W" || p === "F") return "F";
  return "UNK";
}

export function letterGradeFromValue(value) {
  const n = num(value);
  if (n == null || n <= 0) return "—";
  if (n >= 92) return "A+";
  if (n >= 88) return "A";
  if (n >= 84) return "A-";
  if (n >= 80) return "B+";
  if (n >= 76) return "B";
  if (n >= 72) return "B-";
  if (n >= 68) return "C+";
  if (n >= 64) return "C";
  if (n >= 60) return "C-";
  if (n >= 56) return "D+";
  if (n >= 52) return "D";
  return "D-";
}

/** Letter grade for scouting-desk cells (numeric OVR or preformatted letter). */
export function formatDeskGrade(row) {
  if (!row || row.locked) return "??";
  if (row.grade == null || row.grade === "") return "—";
  const n = num(row.grade);
  if (n != null && n > 20) return letterGradeFromValue(n);
  const s = String(row.grade).trim();
  if (/^[A-F][+-]?$/.test(s)) return s;
  return s || "—";
}

/** Analytics model score from production + rank — stable, not random. */
export function deriveModelGradeFromAnalytics(player, profile, analytics = null) {
  const a = analytics || profile?.analytics || profile?.stats?.analytics || {};
  const rank = num(player?.rank ?? profile?.rank) || 99;
  const ovr = num(
    profile?.scoutedOverall
    ?? player?.ovrHint
    ?? profile?.currentOvrEstimate
    ?? profile?.scouted_overall_estimate
  );
  let score = ovr ?? 62;
  const war = num(a.war);
  const gp = num(player?.gp ?? profile?.stats?.games) || 0;
  const pts = num(player?.points ?? profile?.stats?.points);
  const ppg = num(a.ppg ?? profile?.stats?.ppg ?? (gp > 0 && pts != null ? pts / gp : null));
  if (war != null) score += war * 3.5;
  if (ppg != null) {
    if (ppg >= 1.5) score += 8;
    else if (ppg >= 1.0) score += 4;
    else if (ppg >= 0.7) score += 2;
  }
  if (rank <= 3) score += 8;
  else if (rank <= 8) score += 5;
  else if (rank <= 15) score += 2;
  return clamp(Math.round(score), 52, 96);
}

function pickAttr(source, ...keys) {
  for (const k of keys) {
    const v = source?.[k];
    if (v != null && Number.isFinite(Number(v)) && Number(v) > 0) return Number(v);
  }
  return null;
}

function estimateGoalieAttr(player, profile, key, seedOffset = 0) {
  const src = { ...player, ...(profile?.attributes || {}) };
  const direct = {
    glove: pickAttr(src, "glove", "glove_score"),
    blocker: pickAttr(src, "blocker", "blocker_score"),
    reflexes: pickAttr(src, "reflexes", "reflex_score"),
    rebound_control: pickAttr(src, "rebound_control", "reboundControl", "rebound_score"),
    positioning: pickAttr(src, "positioning", "positioning_score"),
    puck_handling: pickAttr(src, "puck_handling", "puckHandling", "puck_handling_score"),
    skating: pickAttr(src, "skating", "skating_score", "mobility"),
    compete: pickAttr(src, "compete", "competitiveness"),
    poise: pickAttr(src, "poise", "mental_toughness", "composure"),
  };
  if (direct[key] != null) return direct[key];
  const ovr = num(profile?.scoutedOverall ?? player?.ovrHint ?? player?.true_ovr) || 62;
  const jitter = (hashSeed(`${player?.id}-${key}-${seedOffset}`) % 13) - 6;
  const base = {
    glove: ovr + 2,
    blocker: ovr - 1,
    reflexes: ovr + 1,
    rebound_control: ovr - 2,
    positioning: ovr,
    puck_handling: ovr - 6,
    skating: ovr - 3,
    compete: num(player?.compete) || ovr - 1,
    poise: num(player?.poise) || ovr,
  };
  return clamp(Math.round((base[key] ?? ovr) + jitter), 45, 94);
}

export function resolveGoalieToolRows(player, profile, completion = 55, wideFog = false) {
  const keys = [
    ["Glove", "glove", 1],
    ["Blocker", "blocker", 2],
    ["Reflexes", "reflexes", 3],
    ["Rebound", "rebound_control", 4],
    ["Positioning", "positioning", 5],
    ["Puck-handling", "puck_handling", 6],
  ];
  return keys.map(([label, key, seed]) => {
    const raw = estimateGoalieAttr(player, profile, key, seed);
    const display = attributeDisplayForTools(raw, completion, seed, { wideFog });
    const mid = display.range ? (display.range[0] + display.range[1]) / 2 : (display.locked ? null : raw);
    return {
      label,
      text: display.text,
      locked: display.locked,
      raw,
      mid: Number.isFinite(mid) ? mid : raw,
      low: display.range ? display.range[0] : null,
      high: display.range ? display.range[1] : null,
    };
  });
}

function attributeDisplayForTools(exactValue, completion, attrSeed = 0, { wideFog = false } = {}) {
  const v = Number(exactValue);
  const hasVal = Number.isFinite(v) && v > 0;
  const base = hasVal ? v : 50;
  const fogBoost = wideFog ? 1.65 : 1.0;
  if (completion <= 18) {
    return { text: "?", range: null, locked: true };
  }
  if (completion <= 55 || wideFog) {
    const spread = Math.round(12 * fogBoost);
    const low = clamp(base - spread + (hashSeed(base + attrSeed) % 5), 40, 96);
    const high = clamp(low + spread + (wideFog ? 4 : 0), low, 96);
    return { text: `${low}–${high}`, range: [low, high], locked: false };
  }
  if (completion <= 80) {
    const spread = 6;
    const low = clamp(base - spread + (hashSeed(base + attrSeed + 7) % 3), 45, 96);
    const high = clamp(low + spread, low, 96);
    return { text: `${low}–${high}`, range: [low, high], locked: false };
  }
  if (completion < 95) {
    const spread = 2;
    const low = clamp(base - spread, 45, 96);
    const high = clamp(base + spread, low, 96);
    return { text: `${low}–${high}`, range: [low, high], locked: false };
  }
  return { text: String(Math.round(base)), range: [base, base], locked: false };
}

export function resolveArchetype(player, profile, tools, isGoalie) {
  const comparison = profile?.player_comparison;
  if (comparison?.archetype && !isGoalie) {
    return {
      label: String(comparison.archetype).toUpperCase(),
      blurb: profile?.micro_summary || comparison.summary || "Scouting file still building the style read.",
      source: "backend",
    };
  }
  const seed = hashSeed(`${player?.id}-arch`);
  if (isGoalie) {
    const pool = GOALIE_ARCHETYPES;
    const rebound = num(tools?.find((t) => t.label === "Rebound")?.mid);
    const positioning = num(tools?.find((t) => t.label === "Positioning")?.mid);
    const puck = num(tools?.find((t) => t.label === "Puck-handling")?.mid);
    let idx = seed % pool.length;
    if (positioning != null && positioning >= 78) idx = pool.findIndex((a) => a.key === "positional_anchor");
    else if (rebound != null && rebound >= 78) idx = pool.findIndex((a) => a.key === "long_levered_wall");
    else if (puck != null && puck >= 72) idx = pool.findIndex((a) => a.key === "puck_moving_netminder");
    else if (num(tools?.find((t) => t.label === "Reflexes")?.mid) >= 80) idx = pool.findIndex((a) => a.key === "athletic_stopper");
    if (idx < 0) idx = seed % pool.length;
    const pick = pool[idx];
    return { label: pick.label, blurb: pick.blurb, source: "derived" };
  }

  const bucket = posBucket(player?.position);
  const pool = bucket === "D" ? SKATER_ARCHETYPES.D : SKATER_ARCHETYPES.F;
  const skating = num(tools?.find((t) => t.label === "Skating")?.mid);
  const shot = num(tools?.find((t) => t.label === "Shot")?.mid);
  const vision = num(tools?.find((t) => t.label === "Vision")?.mid);
  const defense = num(tools?.find((t) => t.label === "Defense")?.mid);

  const roleRaw = String(
    player?.prospectRole
    || profile?.prospect_role
    || profile?.play_style
    || profile?.playStyle
    || comparison?.play_style
    || player?.playstyle
    || player?.play_style
    || "",
  ).trim();
  const roleKey = roleRaw.toLowerCase().replace(/[\s-]+/g, "_");

  const roleToArchetypeKey = {
    defensive_defenseman: "shutdown_d",
    shutdown_defenseman: "shutdown_d",
    offensive_defenseman: "offensive_d",
    two_way_defenseman: "two_way_d",
    power_forward: "power_forward",
    sniper: "sniper",
    playmaker: "playmaker",
    two_way_forward: "two_way_forward",
    two_way_center: "two_way_forward",
    two_way_winger: "two_way_forward",
  };

  const mappedKey = roleToArchetypeKey[roleKey]
    || (roleKey.includes("defensive") || roleKey.includes("shutdown") ? "shutdown_d" : null)
    || (roleKey.includes("offensive") || roleKey.includes("puck_moving") ? "offensive_d" : null)
    || (roleKey.includes("two_way") || roleKey.includes("two-way") ? (bucket === "D" ? "two_way_d" : "two_way_forward") : null);
  if (mappedKey) {
    const roleIdx = pool.findIndex((a) => a.key === mappedKey);
    if (roleIdx >= 0) {
      const pick = pool[roleIdx];
      return { label: pick.label, blurb: pick.blurb, source: "backend" };
    }
  }

  let idx = seed % pool.length;
  if (bucket === "D") {
    if (defense != null && defense >= 78 && (skating == null || defense >= skating + 2)) {
      idx = pool.findIndex((a) => a.key === "shutdown_d");
    } else if (shot != null && shot >= 76 && (defense == null || shot >= defense)) {
      idx = pool.findIndex((a) => a.key === "offensive_d");
    } else if (skating != null && skating >= 80 && vision != null && vision >= 76 && (defense == null || skating >= defense)) {
      idx = pool.findIndex((a) => a.key === "transition_playdriver");
    } else if (skating != null && skating >= 78) {
      idx = pool.findIndex((a) => a.key === "mobile_d");
    }
  } else if (skating != null && skating >= 80 && vision != null && vision >= 76) {
    idx = pool.findIndex((a) => a.key === "transition_playdriver");
  } else if (shot != null && shot >= 82) {
    idx = pool.findIndex((a) => a.key === "sniper");
  } else if (vision != null && vision >= 82) {
    idx = pool.findIndex((a) => a.key === "playmaker");
  }
  if (idx < 0) idx = seed % pool.length;
  const pick = pool[idx] || SKATER_ARCHETYPES.default[0];
  return { label: pick.label, blurb: pick.blurb, source: comparison?.archetype ? "backend" : "derived" };
}

export function humanizePlayStyleLabel(raw) {
  if (raw == null || raw === "") return null;
  const s = String(raw).trim();
  if (!s) return null;
  const key = s.toUpperCase().replace(/[\s-]+/g, "_");
  const lookup = {
    TWO_WAY_F: "Two-way forward",
    TWO_WAY_D: "Two-way defenseman",
    TWO_WAY_W: "Two-way winger",
    POWER_FORWARD: "Power forward",
    SNIPER: "Sniper",
    PLAYMAKER: "Playmaker",
    GRINDER: "Grinder",
    OFFENSIVE_D: "Offensive defenseman",
    SHUTDOWN_D: "Shutdown defenseman",
    PUCK_MOVING_G: "Puck-moving goalie",
    BUTTERFLY: "Butterfly",
    HYBRID: "Hybrid",
    ATHLETIC: "Athletic",
  };
  if (lookup[key]) return lookup[key];
  if (key.includes("_") || (s === s.toUpperCase() && s.length > 2)) {
    return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
  return s;
}

export function resolvePlayStyleTag(player, profile, tools, isGoalie) {
  const backendRaw = profile?.player_comparison?.play_style
    || profile?.play_style
    || profile?.playStyle
    || player?.goalieStyle
    || player?.analytics?.goalie_style;
  if (backendRaw) {
    return { label: humanizePlayStyleLabel(backendRaw), source: "backend" };
  }
  const seed = hashSeed(`${player?.id}-style`);
  if (isGoalie) {
    const positioning = num(tools?.find((t) => t.label === "Positioning")?.mid) || 0;
    const puck = num(tools?.find((t) => t.label === "Puck-handling")?.mid) || 0;
    const reflex = num(tools?.find((t) => t.label === "Reflexes")?.mid) || 0;
    if (puck >= 72) return { label: "Puck-moving", source: "derived" };
    if (reflex >= 80 && positioning < 72) return { label: "Athletic", source: "derived" };
    if (positioning >= 76) return { label: GOALIE_PLAY_STYLES[seed % 2 === 0 ? 1 : 0], source: "derived" };
    return { label: GOALIE_PLAY_STYLES[seed % GOALIE_PLAY_STYLES.length], source: "derived" };
  }
  const bucket = posBucket(player?.position);
  const pool = bucket === "D" ? SKATER_PLAY_STYLES.D : SKATER_PLAY_STYLES.F;
  return { label: pool[seed % pool.length], source: "derived" };
}

export function resolveOvrBands(profile, player, ceilingHidden) {
  const hidden = Boolean(ceilingHidden || profile?.peak_range?.hidden);
  const nowLow = num(profile?.now_range?.low ?? profile?.overallRangeLow ?? player?.ovrRange?.low);
  const nowHigh = num(profile?.now_range?.high ?? profile?.overallRangeHigh ?? player?.ovrRange?.high);
  const peakLow = num(
    profile?.peak_range?.low
    ?? profile?.scoutedPotentialLow
    ?? player?.potentialRange?.low
  );
  const peakHigh = num(
    profile?.peak_range?.high
    ?? profile?.scoutedPotentialHigh
    ?? player?.potentialRange?.high
  );

  const nowText = nowLow != null && nowHigh != null
    ? `${Math.round(nowLow)}–${Math.round(nowHigh)}`
    : null;
  const peakText = !hidden && peakHigh != null ? String(Math.round(peakHigh)) : null;

  let headroom = num(profile?.headroom_delta ?? profile?.headroomDelta);
  if (headroom == null && !hidden && nowHigh != null && peakHigh != null) {
    headroom = Math.round(peakHigh - nowHigh);
  }

  const fileDepthLabel = profile?.file_depth_label
    || (nowText ? `Now ${nowText} OVR` : null);
  const peakRangeLabel = !hidden && peakLow != null && peakHigh != null
    ? `Peak ${Math.round(peakLow)}–${Math.round(peakHigh)} OVR`
    : null;

  return {
    nowLow,
    nowHigh,
    nowText,
    peakLow,
    peakHigh,
    peakText,
    headroom,
    fileDepthLabel,
    peakRangeLabel,
    hidden,
  };
}

export function resolveProjectedRangeLabel(player, profile, ceilingHidden) {
  const bands = resolveOvrBands(profile, player, ceilingHidden);
  if (bands.fileDepthLabel) {
    return { text: bands.fileDepthLabel, source: "backend" };
  }
  if (bands.hidden && bands.nowText) {
    return { text: `Now ${bands.nowText} OVR`, source: "range" };
  }
  return { text: null, source: "missing" };
}

export function resolveCreaseZoneGrades(player, profile, tools) {
  const rebound = tools?.find((t) => t.label === "Rebound");
  const positioning = tools?.find((t) => t.label === "Positioning");
  const puck = tools?.find((t) => t.label === "Puck-handling");
  const composite = (parts) => {
    const rows = parts.filter(Boolean);
    if (!rows.length || rows.some((t) => t.locked)) return { locked: true, value: null };
    const vals = rows.map((t) => t.mid ?? t.raw).filter((n) => Number.isFinite(n));
    if (!vals.length) return { locked: true, value: null };
    return { locked: false, value: vals.reduce((a, b) => a + b, 0) / vals.length };
  };
  return {
    rebound: composite([rebound, tools?.find((t) => t.label === "Reflexes")]),
    angles: composite([positioning]),
    range: composite([puck, tools?.find((t) => t.label === "Glove")]),
  };
}

export function zoneGradeWord(n) {
  if (n == null || !Number.isFinite(n)) return "Unknown";
  if (n >= 85) return "Elite";
  if (n >= 75) return "Solid";
  if (n >= 62) return "Serviceable";
  if (n >= 52) return "Shaky";
  return "Raw";
}

/** Tier key + fill intensity for zone map tiles — drives color differentiation. */
export function zoneTierMeta(value) {
  const n = num(value);
  if (n == null) return { tier: "fog", fill: 0.1 };
  if (n >= 85) return { tier: "elite", fill: 0.78 };
  if (n >= 75) return { tier: "high", fill: 0.58 };
  if (n >= 62) return { tier: "solid", fill: 0.42 };
  if (n >= 52) return { tier: "developing", fill: 0.28 };
  return { tier: "raw", fill: 0.18 };
}

/** In-universe file caveats — player-facing, not dev QA copy. */
export function dossierFileNotes(player, profile, scoutingDesk, { offIceEstimated = false } = {}) {
  const notes = [];
  if (offIceEstimated) {
    notes.push("Limited off-ice reads on file — character tiers still developing.");
  }
  if (scoutingDesk?.source === "empty") {
    notes.push("No scouting desk entries on file yet — backend will populate as the season progresses.");
  }
  const conf = num(profile?.scout_confidence ?? player?.scoutingConfidence);
  if (conf != null && conf < 45) {
    notes.push("File depth remains thin — ceiling and usage projection may shift with more looks.");
  }
  return notes;
}

export function resolveBottomStatStrip(player, profile, tools, isGoalie) {
  const gradeForTool = (tool, ...playerKeys) => {
    const raw = tool?.raw != null && Number.isFinite(Number(tool.raw)) ? Number(tool.raw) : null;
    if (raw != null) return letterGradeFromValue(raw);
    for (const key of playerKeys) {
      const v = num(player?.[key]);
      if (v != null) return letterGradeFromValue(v);
    }
    const mid = tool?.mid != null && Number.isFinite(Number(tool.mid)) ? Number(tool.mid) : null;
    return letterGradeFromValue(mid);
  };
  if (isGoalie) {
    const glove = tools?.find((t) => t.label === "Glove");
    const blocker = tools?.find((t) => t.label === "Blocker");
    const skating = estimateGoalieAttr(player, profile, "skating", 7);
    const compete = estimateGoalieAttr(player, profile, "compete", 8);
    return [
      { label: "Glove", grade: gradeForTool(glove, "glove"), tone: "gold" },
      { label: "Blocker", grade: gradeForTool(blocker, "blocker"), tone: "cyan" },
      { label: "Skating", grade: letterGradeFromValue(skating), tone: "gold" },
      { label: "Compete", grade: letterGradeFromValue(compete), tone: "amber" },
    ];
  }
  const pick = (label) => tools?.find((t) => t.label === label);
  return [
    { label: "Skating", grade: gradeForTool(pick("Skating"), "skating"), tone: "gold" },
    { label: "Shot", grade: gradeForTool(pick("Shot"), "shooting"), tone: "cyan" },
    { label: "Vision", grade: gradeForTool(pick("Vision"), "passing"), tone: "gold" },
    { label: "Physical", grade: gradeForTool(pick("Physical"), "physical"), tone: "amber" },
  ];
}

export function developmentTrajectoryNarrative(player, profile, tools, skillNotes, isGoalie) {
  const notes = Array.isArray(profile?.projectionNotes) ? profile.projectionNotes : [];
  if (profile?.developmentProfile && notes.length) {
    const bits = notes.slice(0, 2).map((n) => (typeof n === "string" ? n : n?.fact || n?.title)).filter(Boolean);
    if (bits.length) return bits.join(" ");
  }
  if (profile?.micro_summary) return profile.micro_summary;
  const age = num(player?.age) || 18;
  const overager = isOverager(player);
  if (isGoalie) {
    const angles = tools?.find((t) => t.label === "Positioning");
    const rebound = tools?.find((t) => t.label === "Rebound");
    if (angles?.plus && rebound?.weak) {
      return "Technically trending toward a positional starter path, but rebound control still lags the rest of the package.";
    }
    if (rebound?.plus) {
      return "Already calms the crease on second chances — the next jump is handling traffic and pro pace.";
    }
    if (overager) {
      return `At ${age}, the technical base needs to show NHL pace soon; mental/compete grades will decide the path.`;
    }
    return "Physical tools are ahead of the reads. Year-over-year growth should show up in angles and post-save recovery first.";
  }
  if (skillNotes?.skatingWeak) {
    return "Skating is the drag on the rest of the toolkit — physical maturity may help, but pro pace is the swing factor.";
  }
  if (overager) {
    return `Older for the class (${age}Y). Present production is real; remaining runway is mostly mental and strength gains.`;
  }
  if (age <= 17) {
    return "Young for the class with room to add mass and pace — trajectory depends on whether the carrying skill keeps scaling.";
  }
  return "Balanced development curve: tools are tracking, but pro translation still hinges on pace and consistency shift-to-shift.";
}

export function goalieStrengthCopy(player, tools) {
  const name = `${player?.firstName || ""} ${player?.lastName || ""}`.trim() || "He";
  if (player?.completion < 40) return [`${name} is still a first-look goalie file — strengths are rumours until we get another viewing.`];
  const byLabel = Object.fromEntries((tools || []).map((t) => [t.label, t]));
  const lines = [];
  if (byLabel.Positioning?.plus) lines.push("Squares early and kills angle before the shot — structure is already pro-caliber for his age.");
  if (byLabel.Reflexes?.plus) lines.push("Second-effort saves bail him out when depth breaks down.");
  if (byLabel.Glove?.plus) lines.push("Glove hand is a weapon on rush chances and lateral releases.");
  if (byLabel.Rebound?.plus) lines.push("Deadens pucks on first contact — forwards aren't feasting on second chances.");
  if (byLabel["Puck-handling"]?.plus) lines.push("Comfortable stopping pucks behind the net and starting breakouts.");
  if (!lines.length) {
    lines.push("Projectable frame with enough compete to stay in the conversation.");
    lines.push("Shows flashes when the crease is organized — consistency is the next layer.");
  }
  return lines.slice(0, 4);
}

export function goalieWeaknessCopy(player, tools) {
  const name = `${player?.firstName || ""} ${player?.lastName || ""}`.trim() || "He";
  if (player?.completion < 40) return [`More looks needed before we tattoo a weakness onto ${name}.`];
  const byLabel = Object.fromEntries((tools || []).map((t) => [t.label, t]));
  const lines = [];
  if (byLabel.Rebound?.weak) lines.push("Rebounds leak into dangerous ice — recovery paths aren't automatic yet.");
  if (byLabel.Positioning?.weak) lines.push("Depth and angle management still chase the play instead of leading it.");
  if (byLabel["Puck-handling"]?.weak) lines.push("Forecheck pressure turns puck-handling into a turnover risk.");
  if (byLabel.Reflexes?.weak) lines.push("Relies on size over quick hands — east-west plays stress him.");
  if (byLabel.Blocker?.weak) lines.push("Blocker side can be exposed on catch-and-release looks.");
  if (isOverager(player)) lines.push("Extra junior year inflates the stat line — same SV% at 18 would move the needle.");
  if (!lines.length) lines.push("Needs pro pace reps before the projection is safe.");
  return lines.slice(0, 4);
}


export function buildScoutingDeskEntries(player, profile, { gp = 0, analytics = null } = {}) {
  const backend = profile?.scouting_history || profile?.scoutingHistory || profile?.scout_reports;
  if (Array.isArray(backend) && backend.length) {
    return {
      entries: backend.map((row) => ({
        scout: row.scout || row.name || "Scout",
        meta: row.meta || row.region || row.league || "",
        quote: row.quote || row.note || row.summary || "",
        grade: row.grade ?? row.rating ?? null,
        gradeLabel: row.grade_label || row.gradeLabel || "GRADE",
        tone: row.tone || "neutral",
        locked: Boolean(row.locked),
      })),
      source: "backend",
    };
  }
  return { entries: [], source: "empty" };
}

function isGoaliePosition(pos) {
  const p = String(pos || "").toUpperCase();
  return p === "G" || p.includes("GOAL");
}

export function outcomeRibbonSegmentsForPosition(outcomes, isGoalie = false, outcomeDistribution = null) {
  if (outcomeDistribution?.segments?.length) {
    return outcomeDistribution.segments.map((seg) => ({
      key: seg.key,
      label: seg.label,
      w: seg.weight ?? seg.w ?? seg.pct,
      pct: seg.pct ?? seg.weight ?? seg.w,
    }));
  }
  return null;
}

/** Fields the UI expects but the backend may not yet expose. */
export function dossierBackendGaps(player, profile, scoutingDesk, { ovrBands = null, offIceEstimated = false } = {}) {
  const gaps = [];
  if (isGoaliePosition(player?.position)) {
    const attrs = ["glove", "blocker", "reflexes", "rebound_control", "positioning", "puck_handling"];
    const missing = attrs.filter((k) => pickAttr(player, k, `${k}_score`) == null && pickAttr(profile, k) == null);
    if (missing.length) gaps.push(`Goalie sub-attributes not on API (${missing.join(", ")}) — grades are estimated from OVR.`);
  }
  if (offIceEstimated) {
    gaps.push("Off-ice character/leadership numbers are estimated — character_read tiers are still fogged.");
  }
  if (scoutingDesk?.source === "empty") gaps.push("Scouting desk — awaiting backend scouting_history rows.");
  if (!profile?.outcome_distribution && !profile?.outcomeDistribution) {
    gaps.push("Outcome distribution ribbon — derived from NHL probability until backend ships outcome_distribution.");
  }
  if (!profile?.development_trajectory && !profile?.developmentTrajectory) {
    gaps.push("Development trajectory narrative — built from projectionNotes/micro_summary.");
  }
  const playRaw = profile?.play_style || profile?.playStyle;
  if (playRaw && /[A-Z_]{3,}/.test(String(playRaw)) && process.env.NODE_ENV === "development") {
    gaps.push(`Play style enum "${playRaw}" was not humanized by backend — using client fallback.`);
  }
  if (ovrBands?.headroom == null && !ovrBands?.hidden && ovrBands?.nowHigh != null && ovrBands?.peakHigh == null) {
    gaps.push("Peak OVR band missing — headroom delta unavailable.");
  }
  return gaps;
}

/** Fast client-side profile from board row — opens dossier instantly while API enriches. */
export function buildStubProspectProfile(player) {
  if (!player?.id) return null;
  const potLow = player.potentialRange?.low ?? player.potentialScore;
  const potHigh = player.potentialRange?.high ?? player.potentialScore;
  const dedicatedPct = player.dedicatedScoutingPct ?? null;
  return {
    _stub: true,
    stats: {
      games: player.gp,
      goals: player.goals,
      assists: player.assists,
      points: player.points,
      ppg: player.ppg,
    },
    overallRangeLow: player.ovrRange?.low,
    overallRangeHigh: player.ovrRange?.high,
    scoutedOverall: player.ovrHint,
    scoutedPotentialLow: potLow,
    scoutedPotentialHigh: potHigh,
    scout_confidence: player.scoutingConfidence ?? player.completion,
    dedicatedScoutFile: dedicatedPct != null ? dedicatedPct >= 20 : (Number(player.completion) || 0) >= 20,
    ceilingHidden: Boolean(player.ceilingHidden),
    character_score: player.characterScore ?? player.character,
    character_concerns: Boolean(player.characterConcerns),
    chapter_profile: player.chapterProfile || null,
    analytics: player.analytics || null,
    is_transcendent: Boolean(player.isTranscendent),
    prospect_role: player.prospectRole,
    play_style: player.playerType,
    potential: player.potentialScore != null
      ? { rating: player.potentialScore, hidden: Boolean(player.ceilingHidden) }
      : (player.ceilingHidden ? { hidden: true } : null),
    archetype: player.prospectRole ? { label: player.prospectRole } : null,
    rank: player.rank,
    intel_label: player.intelLabel,
    risk: player.riskLabel,
  };
}
