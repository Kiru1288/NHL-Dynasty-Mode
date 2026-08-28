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
  let idx = seed % pool.length;
  if (skating != null && skating >= 80 && vision != null && vision >= 76) idx = 0;
  else if (shot != null && shot >= 82) idx = pool.findIndex((a) => a.key === "sniper");
  else if (vision != null && vision >= 82) idx = pool.findIndex((a) => a.key === "playmaker");
  else if (bucket === "D" && shot != null && shot >= 76) idx = pool.findIndex((a) => a.key === "offensive_d");
  if (idx < 0) idx = seed % pool.length;
  const pick = pool[idx] || SKATER_ARCHETYPES.default[0];
  return { label: pick.label, blurb: pick.blurb, source: comparison?.archetype ? "backend" : "derived" };
}

export function resolvePlayStyleTag(player, profile, tools, isGoalie) {
  const backend = profile?.player_comparison?.play_style
    || profile?.play_style
    || profile?.playStyle
    || player?.goalieStyle
    || player?.analytics?.goalie_style;
  if (backend) return { label: String(backend), source: "backend" };
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

export function resolveProjectedRangeLabel(player, profile, ceilingHidden) {
  const conf = num(profile?.scout_confidence ?? player?.scoutingConfidence ?? player?.completion);
  const ovrLow = num(profile?.overallRangeLow ?? player?.ovrRange?.low);
  const ovrHigh = num(profile?.overallRangeHigh ?? player?.ovrRange?.high);
  const potLow = num(profile?.scoutedPotentialLow ?? player?.potentialRange?.low);
  const potHigh = num(profile?.scoutedPotentialHigh ?? player?.potentialRange?.high);
  if (ceilingHidden) {
    if (ovrLow != null && ovrHigh != null) {
      return { text: `${Math.round(ovrLow)}–${Math.round(ovrHigh)} OVR floor/now`, source: "range" };
    }
    return { text: null, source: "missing" };
  }
  if (potLow != null && potHigh != null) {
    return { text: `${Math.round(potLow)}–${Math.round(potHigh)} OVR ceiling/floor`, source: "backend" };
  }
  if (ovrLow != null && ovrHigh != null) {
    return { text: `${Math.round(ovrLow)}–${Math.round(ovrHigh)} OVR range`, source: "derived" };
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
        viewings: row.viewings ?? row.viewing_count ?? null,
        hitRate: row.hit_rate ?? row.hitRate ?? null,
        quote: row.quote || row.note || row.summary || "",
        grade: row.grade ?? row.rating ?? null,
        gradeLabel: row.grade_label || row.gradeLabel || "HIS GRADE",
        tone: row.tone || "neutral",
        locked: Boolean(row.locked),
      })),
      source: "backend",
      psychLocked: backend.every((r) => !String(r.scout || r.name || "").toLowerCase().includes("psych")),
    };
  }

  const conf = num(profile?.scout_confidence ?? player?.scoutingConfidence ?? player?.completion) || 52;
  const entries = [];
  const assigned = player?.assignedScout || profile?.assigned_scout || profile?.assignedScout;
  const gradeBase = num(
    profile?.scoutedOverall
    ?? profile?.scouted_overall_estimate
    ?? player?.ovrHint
    ?? player?.trueOvr
  ) || null;
  const modelGrade = deriveModelGradeFromAnalytics(player, profile, analytics);

  if (assigned || profile?.projectionNotes || profile?.micro_summary) {
    entries.push({
      scout: assigned || "Lead scout",
      meta: assigned ? "Assigned file" : "Central scouting",
      viewings: gp > 0 ? Math.max(1, Math.min(8, Math.round(conf / 14))) : null,
      hitRate: Math.round(conf),
      quote: profile?.projectionNotes || profile?.micro_summary || profile?.microSummary || "File in progress.",
      grade: gradeBase,
      gradeLabel: "HIS GRADE",
      tone: "green",
      locked: false,
    });
  }

  const strengthEvidence = profile?.strengthsEvidence || profile?.strengths_evidence || [];
  (Array.isArray(strengthEvidence) ? strengthEvidence : []).slice(0, 1).forEach((ev) => {
    const title = typeof ev === "string" ? ev : (ev?.title || ev?.label || "");
    const detail = typeof ev === "object" ? (ev?.detail || ev?.note || "") : "";
    if (!title) return;
    entries.push({
      scout: "Skills desk",
      meta: "Strengths read",
      viewings: null,
      hitRate: null,
      quote: detail ? `${title} — ${detail}` : title,
      grade: gradeBase,
      gradeLabel: "TOOLS",
      tone: "cyan",
      locked: false,
    });
  });

  const weaknessEvidence = profile?.weaknessesEvidence || profile?.weaknesses_evidence || [];
  (Array.isArray(weaknessEvidence) ? weaknessEvidence : []).slice(0, 1).forEach((ev) => {
    const title = typeof ev === "string" ? ev : (ev?.title || ev?.label || "");
    const detail = typeof ev === "object" ? (ev?.detail || ev?.note || "") : "";
    if (!title) return;
    entries.push({
      scout: "Regional scout",
      meta: "Development gap",
      viewings: null,
      hitRate: null,
      quote: detail ? `${title} — ${detail}` : title,
      grade: gradeBase != null ? Math.max(45, gradeBase - 6) : null,
      gradeLabel: "HIS GRADE",
      tone: "amber",
      locked: false,
    });
  });

  const charRead = profile?.character_read;
  if (charRead?.interview_notes) {
    entries.push({
      scout: "Character interview",
      meta: "Private setting",
      viewings: 1,
      hitRate: charRead.confidence ?? null,
      quote: charRead.interview_notes,
      grade: null,
      gradeLabel: "READ",
      tone: "green",
      locked: false,
    });
  } else {
    entries.push({
      scout: "Psych. interview",
      meta: "Not commissioned",
      viewings: null,
      hitRate: null,
      quote: null,
      grade: null,
      gradeLabel: "",
      tone: "muted",
      locked: true,
    });
  }

  if (profile?.stockReason || player?.stockReason) {
    entries.push({
      scout: "Analytics dept.",
      meta: "Stock movement",
      viewings: null,
      hitRate: Math.round(conf),
      quote: profile?.stockReason || player?.stockReason,
      grade: modelGrade,
      gradeLabel: "MODEL",
      tone: "cyan",
      locked: false,
    });
  } else if (analytics && (analytics.war != null || analytics.ppg != null || player?.gp > 0)) {
    const gp = num(player?.gp) || 0;
    const pts = num(player?.points) || 0;
    const ppg = num(analytics.ppg ?? (gp > 0 ? pts / gp : null));
    const quote = gp > 0 && ppg != null
      ? `${pts}P in ${gp} GP · ${ppg.toFixed(2)} PPG`
      : "Production model on file.";
    entries.push({
      scout: "Analytics dept.",
      meta: "Production model",
      viewings: null,
      hitRate: analytics.war != null ? Math.round(clamp(50 + Number(analytics.war) * 8, 45, 88)) : Math.round(conf),
      quote,
      grade: modelGrade,
      gradeLabel: "MODEL",
      tone: "cyan",
      locked: false,
    });
  }

  if (!entries.length) {
    return { entries: [], source: "empty", psychLocked: true };
  }

  return { entries, source: "profile", psychLocked: entries.some((e) => e.locked) };
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
  if (!outcomes) return null;
  const nhl = Math.max(0, Math.min(100, Number(outcomes.nhlOdds) || 0));
  if (isGoalie) {
    const non = Math.max(0, 100 - nhl);
    const bust = Math.round(non * 0.38);
    const minor = Math.max(0, non - bust);
    const backup = Math.min(Math.round(nhl * 0.55), nhl);
    const platoon = Math.max(0, Math.min(Math.round(nhl * 0.35), nhl - backup));
    const starter = Math.max(0, nhl - backup - platoon);
    const segs = [
      { key: "bust", label: "Bust", w: bust },
      { key: "ahl", label: "AHL/ECHL", w: minor },
      { key: "mid", label: "NHL Backup", w: backup },
      { key: "top", label: "Platoon", w: platoon },
      { key: "star", label: "Starter", w: starter },
    ];
    const sum = segs.reduce((s, x) => s + x.w, 0);
    if (sum <= 0) return null;
    return segs.map((x) => ({ ...x, pct: (x.w / sum) * 100 }));
  }
  const keys = ["peak", "expected", "worst", "nhlOdds", "hitPeakPct", "shootPastPct"];
  if (keys.some((k) => outcomes[k] == null || !Number.isFinite(Number(outcomes[k])))) return null;
  const non = Math.max(0, 100 - nhl);
  const bust = Math.round(non * 0.42);
  const ahl = Math.max(0, non - bust);
  const star = Math.min(Number(outcomes.shootPastPct), Math.round(nhl * 0.28));
  const top6 = Math.min(Number(outcomes.hitPeakPct), Math.max(0, nhl - star));
  const mid6 = Math.max(0, nhl - star - top6);
  const segs = [
    { key: "bust", label: "Bust", w: bust },
    { key: "ahl", label: "AHL", w: ahl },
    { key: "mid", label: "Mid-6", w: mid6 },
    { key: "top", label: "Top-6", w: top6 },
    { key: "star", label: "Star+", w: star },
  ];
  const sum = segs.reduce((s, x) => s + x.w, 0);
  if (sum <= 0) return null;
  return segs.map((x) => ({ ...x, pct: (x.w / sum) * 100 }));
}

/** Fields the UI expects but the backend may not yet expose. */
export function dossierBackendGaps(player, profile, scoutingDesk) {
  const gaps = [];
  if (isGoaliePosition(player?.position)) {
    const attrs = ["glove", "blocker", "reflexes", "rebound_control", "positioning", "puck_handling"];
    const missing = attrs.filter((k) => pickAttr(player, k, `${k}_score`) == null && pickAttr(profile, k) == null);
    if (missing.length) gaps.push(`Goalie sub-attributes not on API (${missing.join(", ")}) — grades are estimated from OVR.`);
  }
  if (scoutingDesk?.source === "empty") gaps.push("Scouting desk — assign a scout or commission interviews to populate reports.");
  if (!profile?.outcome_distribution && !profile?.outcomeDistribution) gaps.push("Outcome distribution ribbon — derived from NHL probability until backend ships outcome_distribution.");
  if (!profile?.development_trajectory && !profile?.developmentTrajectory) gaps.push("Development trajectory narrative — built from projectionNotes/micro_summary.");
  return gaps;
}
