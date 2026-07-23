/** Normalize backend development-report records for the Organizational Development Review UI. */

import {
  normalizeProspectTeamName,
  resolveLeagueFromTeam,
} from "../../data/prospectLeagueTeams";

const EMPTY = Object.freeze([]);
const READINESS_ORDER = {
  "NHL Ready": 0,
  Close: 1,
  Developing: 2,
  "Long-Term": 3,
  "At Risk": 4,
};
const TREND_ORDER = {
  Breakout: 0,
  Improved: 1,
  Stable: 2,
  Stalled: 3,
  Regressed: 4,
};
const NOTABLE_ORDER = {
  "Top Riser": 0,
  "Newly NHL Ready": 1,
  Regressed: 2,
  Stalled: 3,
  "Late Bloomer": 4,
  "High Risk": 5,
};

export const DEV_FILTERS = [
  { id: "all", label: "All" },
  { id: "ready", label: "Ready" },
  { id: "improved", label: "Improved" },
  { id: "stalled", label: "Stalled" },
  { id: "regressed", label: "Regressed" },
];

export const POSITION_FILTERS = [
  { id: "all", label: "All Positions" },
  { id: "F", label: "F" },
  { id: "D", label: "D" },
  { id: "G", label: "G" },
];

export const SORT_OPTIONS = [
  { id: "ovr", label: "Current OVR" },
  { id: "ovr_delta", label: "OVR change" },
  { id: "potential", label: "Potential" },
  { id: "age", label: "Age" },
  { id: "readiness", label: "Readiness" },
  { id: "development", label: "Biggest movers" },
];

export const ORG_GROUPS = [
  "NHL / AHL",
  "ECHL",
  "Junior / NCAA",
  "Europe",
  "Unsigned",
];

function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function str(v, fallback = "") {
  if (v == null || v === "") return fallback;
  return String(v);
}

const LEAGUE_CODE_MAP = {
  CHL_OHL: "OHL",
  CHL_WHL: "WHL",
  CHL_QMJHL: "QMJHL",
  USHL: "USHL",
  NCAA: "NCAA",
  AHL: "AHL",
  OHL: "OHL",
  WHL: "WHL",
  QMJHL: "QMJHL",
};

function mapLeagueCode(code) {
  const raw = str(code, "").trim();
  if (!raw) return "";
  const upper = raw.toUpperCase();
  return LEAGUE_CODE_MAP[upper] || raw;
}

export function formatProspectLeague(raw = {}) {
  const display = str(raw?.league_display ?? raw?.current_league_id ?? raw?.league, "").trim();
  if (display) {
    const mapped = mapLeagueCode(display);
    return mapped.length <= 18 ? mapped : mapped.slice(0, 18);
  }
  const teamName = formatProspectTeam(raw);
  const fromTeam = resolveLeagueFromTeam(teamName);
  if (fromTeam?.leagueDisplay) return fromTeam.leagueDisplay;
  return mapLeagueCode(raw?.league_code) || "—";
}

export function formatProspectTeam(raw = {}, fallback = "") {
  const name = str(
    raw?.team_name ?? raw?.teamName ?? raw?.team ?? raw?.current_team ?? fallback,
    ""
  );
  const leagueId = str(raw?.current_league_id ?? raw?.league_id ?? raw?.league, "").toUpperCase();
  const orgGroup = str(raw?.org_group, "");
  // NHL / AHL / ECHL org labels are already the franchise name — never remap
  // "Ottawa Senators" → "Ottawa 67's" via city token matching.
  if (
    leagueId === "NHL" || leagueId === "AHL" || leagueId === "ECHL"
    || orgGroup.startsWith("NHL")
    || /senators|bruins|canadiens|maple leafs|oilers|flames|canucks|jets|wild|predators|blues|blackhawks|red wings|sabres|penguins|flyers|rangers|islanders|devils|capitals|hurricanes|panthers|lightning|blue jackets|stars|avalanche|coyotes|golden knights|kraken|sharks|ducks|kings/i.test(name)
  ) {
    return name || "—";
  }
  return normalizeProspectTeamName(name) || name || "—";
}

const ATTR_PREFIXES = [
  "off_",
  "pm_",
  "def_",
  "phy_",
  "skg_",
  "iqm_",
  "pc_",
  "dev_",
  "per_",
  "st_",
  "g_",
];

const META_ATTR_KEYS = new Set([
  "dev_potential",
  "dev_ceiling",
  "dev_growth_rate",
  "dev_work_ethic",
  "dev_coachability",
  "dev_learning_ability",
  "potential",
  "overall",
  "ovr",
]);

function isSkillAttr(key) {
  const k = str(key, "").toLowerCase();
  return k && !META_ATTR_KEYS.has(k) && !k.startsWith("_");
}

function attrLabel(key) {
  let k = str(key, "").toLowerCase();
  if (!k) return "—";
  for (const prefix of ATTR_PREFIXES) {
    if (k.startsWith(prefix)) {
      k = k.slice(prefix.length);
      break;
    }
  }
  return k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatOvrDelta(delta) {
  const n = Math.round(num(delta));
  if (n === 0) return "0";
  return n > 0 ? `+${n}` : `${n}`;
}

function formatAttributeDelta(val) {
  const n = Math.round(num(val));
  if (!Number.isFinite(n) || n === 0) return null;
  return n > 0 ? `+${n}` : `${n}`;
}

function formatAttrValue(val) {
  const n = num(val);
  if (!Number.isFinite(n)) return "—";
  return String(Math.round(n));
}

function buildAttributeRows(record, limit = null) {
  const deltas = record?.attribute_deltas || record?.attributeDeltas || {};
  const prev = record?.previous_attributes || {};
  const cur = record?.current_attributes || {};
  const rows = Object.entries(deltas)
    .map(([key, val]) => {
      if (!isSkillAttr(key)) return null;
      const formatted = formatAttributeDelta(val);
      if (!formatted) return null;
      const before = prev[key];
      const after = cur[key];
      return {
        key,
        label: attrLabel(key),
        delta: num(val),
        display: formatted,
        before: before != null ? formatAttrValue(before) : null,
        after: after != null ? formatAttrValue(after) : null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return limit != null ? rows.slice(0, limit) : rows;
}

function topAttributeDeltas(record, limit = 6) {
  return buildAttributeRows(record, limit);
}

function allAttributeDeltas(record) {
  return buildAttributeRows(record, 12);
}

function isForwardPosition(pos) {
  const p = str(pos, "").toUpperCase();
  return p && !["G", "D", "LD", "RD", "DEF", "DEFENSE"].includes(p);
}

function isDefensePosition(pos) {
  const p = str(pos, "").toUpperCase();
  return ["D", "LD", "RD", "DEF", "DEFENSE"].includes(p);
}

/** Map normalized backend record to UI row — no client-side hockey conclusions. */
export function normalizeDevelopmentRecord(record = {}, extras = {}) {
  if (!record || typeof record !== "object") return null;

  const legacy = Boolean(extras.legacy);
  const id = str(record.player_id ?? record.id, "");
  if (!id) return null;

  const prevOvr = Math.round(num(record.previous_overall, num(record.current_overall)));
  const curOvr = Math.round(num(record.current_overall, prevOvr));
  const delta = Math.round(num(
    record.overall_delta,
    curOvr - prevOvr
  ));
  const potential = num(record.potential, 0);

  return {
    id,
    name: str(record.player_name ?? record.name, "Unknown"),
    position: str(record.position, "—").toUpperCase(),
    age: record.age ?? "—",
    potential: potential > 0 ? Math.round(potential) : null,
    team: formatProspectTeam(record, record.team_name),
    league: formatProspectLeague(record),
    orgGroup: str(record.org_group, "Junior / NCAA"),
    previousOvr: prevOvr,
    ovr: curOvr,
    ovrDelta: delta,
    ovrDeltaLabel: formatOvrDelta(delta),
    readinessTier: str(record.readiness_tier, legacy ? "—" : "Developing"),
    readinessScore: num(record.readiness_score, 0),
    developmentTrend: str(record.development_trend, legacy ? "—" : "Stable"),
    developmentPhase: str(record.development_phase, ""),
    primaryReason: str(record.primary_reason, ""),
    secondaryReasons: Array.isArray(record.secondary_reasons)
      ? record.secondary_reasons.map(String)
      : EMPTY,
    notable: Boolean(record.notable),
    notableCategory: str(record.notable_category, ""),
    signed: Boolean(record.signed),
    goalie: Boolean(record.goalie ?? record.position === "G"),
    legacy,
    incomplete: legacy || (!record.primary_reason && !record.development_trend),
    topAttributes: topAttributeDeltas(record),
    allAttributes: allAttributeDeltas(record),
    seasonStats: record.season_stats || {},
    leagueContext: record.league_adjusted_context || {},
    developmentHistory: Array.isArray(record.development_history)
      ? record.development_history
      : EMPTY,
    attributeDeltas: record.attribute_deltas || {},
    pool: str(record.pool, ""),
    raw: record,
  };
}

/** Legacy thin riser/faller row when schema_version < 2. */
function legacyRowFromDelta(row = {}, delta = 0) {
  return normalizeDevelopmentRecord(
    {
      player_id: row.player_id ?? row.id,
      player_name: row.name,
      position: row.position || "F",
      age: row.age,
      current_overall: num(row.overall ?? row.ovr),
      previous_overall: num(row.overall ?? row.ovr) - num(delta),
      overall_delta: num(delta),
      development_trend: delta > 0 ? "Improved" : delta < 0 ? "Regressed" : "Stable",
      readiness_tier: "",
      primary_reason: "",
      org_group: "NHL / AHL",
      signed: true,
    },
    { legacy: true }
  );
}

function rosterEnrichmentMap(franchiseState = {}) {
  const map = new Map();
  const push = (player) => {
    const id = str(player?.player_id ?? player?.id, "");
    if (!id || map.has(id)) return;
    map.set(id, player);
  };

  for (const org of franchiseState?.roster_browser?.organizations || []) {
    for (const key of ["nhl", "ahl", "echl", "prospects"]) {
      for (const p of org?.[key] || []) push(p);
    }
  }
  for (const block of franchiseState?.roster_browser?.development_leagues || []) {
    for (const tm of block?.teams || []) {
      for (const p of tm?.players || []) push(p);
    }
  }
  return map;
}

function enrichFromRoster(row, enrichMap) {
  if (!row || !enrichMap?.has(row.id)) return row;
  const src = enrichMap.get(row.id);
  return {
    ...row,
    team: row.team !== "—" ? row.team : formatProspectTeam(src),
    league: row.league !== "—" ? row.league : formatProspectLeague(src),
  };
}

/** Primary list from normalized backend development records. */
export function extractDevelopmentPlayers(franchiseState = {}, eventData = {}) {
  const devReport =
    eventData?.development_report ?? franchiseState?.development_report ?? {};
  const schemaVersion = num(devReport.schema_version, 1);
  const rows = [];
  const seen = new Set();
  const enrichMap = rosterEnrichmentMap(franchiseState);

  const pushRecord = (record, extras = {}) => {
    const norm = normalizeDevelopmentRecord(record, extras);
    if (!norm) return;
    const key = norm.id.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    rows.push(enrichFromRoster(norm, enrichMap));
  };

  if (schemaVersion >= 2 && Array.isArray(devReport.organization_players)) {
    for (const record of devReport.organization_players) {
      pushRecord(record);
    }
  } else {
    for (const p of devReport.risers || []) {
      const row = legacyRowFromDelta(p, num(p.delta, 0));
      if (row) {
        const key = row.id.toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          rows.push(row);
        }
      }
    }
    for (const p of devReport.fallers || []) {
      const row = legacyRowFromDelta(p, num(p.delta, 0));
      if (row) {
        const key = row.id.toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          rows.push(row);
        }
      }
    }
    for (const p of devReport.prospects_ready || []) {
      pushRecord(p, { legacy: true });
    }
    for (const p of devReport.org_prospect_deltas || []) {
      pushRecord(
        {
          player_id: p.player_id,
          player_name: p.name,
          current_overall: p.overall,
          previous_overall: num(p.overall) - num(p.delta),
          overall_delta: num(p.delta),
          readiness_tier: p.readiness_tier || "",
          readiness_score: num(p.nhl_readiness),
          current_league_id: p.league_id,
          development_trend: num(p.delta) > 0 ? "Improved" : num(p.delta) < 0 ? "Regressed" : "Stable",
          org_group: "Unsigned",
          signed: false,
        },
        { legacy: true }
      );
    }
  }

  return sortDevelopmentPlayers(rows, "ovr");
}

/** League-wide NHL seasonal growth (other teams) for the League tab. */
export function extractLeagueNhlDevelopment(franchiseState = {}, eventData = {}) {
  const devReport =
    eventData?.development_report ?? franchiseState?.development_report ?? {};
  const rows = [];
  const seen = new Set();
  const push = (raw) => {
    const norm = normalizeDevelopmentRecord({
      ...raw,
      player_id: raw.player_id ?? raw.id,
      player_name: raw.player_name ?? raw.name,
      current_overall: raw.current_overall ?? raw.overall,
      previous_overall: raw.previous_overall
        ?? (num(raw.overall) - num(raw.delta ?? raw.overall_delta)),
      overall_delta: raw.overall_delta ?? raw.delta,
      current_league_id: "NHL",
      org_group: "NHL / AHL",
      team_name: raw.team_name || raw.team,
      attribute_deltas: raw.attribute_deltas || {},
    });
    if (!norm) return;
    const key = norm.id.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    rows.push(norm);
  };
  for (const p of devReport.league_risers || []) push(p);
  for (const p of devReport.league_fallers || []) push(p);
  return sortDevelopmentPlayers(rows, "ovr_delta");
}

/** Reconcile displayed trend with authoritative OVR movement. */
export function reconciledDevelopmentTrend(player = {}) {
  const delta = num(player.ovrDelta);
  const raw = str(player.developmentTrend, "Stable");
  if (delta >= 1.2) return "Breakout";
  if (delta >= 0.35) return "Improved";
  if (delta <= -0.8) return "Regressed";
  if (delta <= -0.35) return "Regressed";
  if (Math.abs(delta) < 0.15) {
    if (raw === "Breakout" || raw === "Regressed") return "Stable";
    return raw || "Stable";
  }
  return delta > 0 ? "Improved" : "Regressed";
}

/** Chips for list rows — trend first; readiness only when it doesn't conflict. */
export function rowStatusChips(player = {}) {
  const trend = reconciledDevelopmentTrend(player);
  const chips = [{ kind: "trend", label: trend, className: trendClass(trend) }];
  const tier = str(player.readinessTier, "");
  const showReady =
    tier === "NHL Ready"
    && (player.notableCategory === "Newly NHL Ready"
      || num(player.ovrDelta) >= 0.15
      || trend !== "Regressed");
  if (showReady) {
    chips.push({ kind: "readiness", label: tier, className: readinessClass(tier) });
  } else if (tier && tier !== "NHL Ready" && tier !== "—") {
    chips.push({ kind: "readiness", label: tier, className: readinessClass(tier) });
  }
  return chips;
}

export function sortDevelopmentPlayers(list, sortId = "development") {
  const sorted = [...list];
  sorted.sort((a, b) => {
    switch (sortId) {
      case "readiness":
        return (
          (READINESS_ORDER[a.readinessTier] ?? 9) - (READINESS_ORDER[b.readinessTier] ?? 9)
          || num(b.readinessScore) - num(a.readinessScore)
        );
      case "ovr":
        return num(b.ovr) - num(a.ovr) || num(b.ovrDelta) - num(a.ovrDelta);
      case "potential":
        return num(b.potential, -1) - num(a.potential, -1) || num(b.ovr) - num(a.ovr);
      case "ovr_delta":
        return num(b.ovrDelta) - num(a.ovrDelta) || num(b.ovr) - num(a.ovr);
      case "age":
        return num(a.age, 99) - num(b.age, 99);
      case "development":
      default:
        return (
          (a.notable ? 0 : 1) - (b.notable ? 1 : 0)
          || (NOTABLE_ORDER[a.notableCategory] ?? 9) - (NOTABLE_ORDER[b.notableCategory] ?? 9)
          || num(b.ovrDelta) - num(a.ovrDelta)
          || (READINESS_ORDER[a.readinessTier] ?? 9) - (READINESS_ORDER[b.readinessTier] ?? 9)
          || num(b.ovr) - num(a.ovr)
        );
    }
  });
  return sorted;
}

export function filterDevelopmentPlayers(list, filterId, positionId = "all") {
  let out = list;
  if (positionId && positionId !== "all") {
    out = out.filter((p) => {
      if (positionId === "G") return p.position === "G";
      if (positionId === "D") return isDefensePosition(p.position);
      if (positionId === "F") return isForwardPosition(p.position);
      return true;
    });
  }
  if (!filterId || filterId === "all") return out;
  return out.filter((p) => {
    const trend = reconciledDevelopmentTrend(p);
    switch (filterId) {
      case "ready":
        return p.readinessTier === "NHL Ready";
      case "improved":
        return trend === "Improved" || trend === "Breakout";
      case "stalled":
        return trend === "Stalled" || p.notableCategory === "Stalled";
      case "regressed":
        return trend === "Regressed" || p.notableCategory === "Regressed";
      default:
        return true;
    }
  });
}

export function summarizeDevelopmentReport(devReport = {}, players = []) {
  const summary = devReport.summary || {};
  if (summary.line) {
    return {
      line: summary.line,
      improved: num(summary.improved),
      nhlReady: num(summary.nhl_ready),
      stalled: num(summary.stalled),
      regressed: num(summary.regressed),
      total: num(summary.total, players.length),
    };
  }
  const improved = players.filter(
    (p) => {
      const t = reconciledDevelopmentTrend(p);
      return t === "Improved" || t === "Breakout";
    }
  ).length;
  const nhlReady = players.filter((p) => p.readinessTier === "NHL Ready").length;
  const stalled = players.filter((p) => reconciledDevelopmentTrend(p) === "Stalled").length;
  const regressed = players.filter((p) => reconciledDevelopmentTrend(p) === "Regressed").length;
  const parts = [];
  if (improved) parts.push(`${improved} improved`);
  if (nhlReady) parts.push(`${nhlReady} NHL ready`);
  if (stalled) parts.push(`${stalled} stalled`);
  if (regressed) parts.push(`${regressed} regressed`);
  return {
    line: parts.length ? parts.join(" · ") : "No notable movement",
    improved,
    nhlReady,
    stalled,
    regressed,
    total: players.length,
  };
}

export function overviewGroups(players = []) {
  const risers = players
    .filter((p) => p.ovrDelta >= 0.4 || p.notableCategory === "Top Riser")
    .sort((a, b) => num(b.ovrDelta) - num(a.ovrDelta))
    .slice(0, 5);
  const regressed = players
    .filter((p) => p.ovrDelta <= -0.4 || p.notableCategory === "Regressed")
    .sort((a, b) => num(a.ovrDelta) - num(b.ovrDelta))
    .slice(0, 5);
  const newlyReady = players
    .filter((p) => p.notableCategory === "Newly NHL Ready")
    .slice(0, 5);
  const stalled = players
    .filter((p) => p.developmentTrend === "Stalled" || p.notableCategory === "Stalled")
    .slice(0, 5);
  return { risers, regressed, newlyReady, stalled };
}

export function leagueMovers(devReport = {}) {
  const risers = (devReport.league_risers || devReport.risers || []).slice(0, 8);
  const fallers = (devReport.league_fallers || devReport.fallers || []).slice(0, 8);
  return { risers, fallers };
}

export function groupPlayersByOrg(players = []) {
  const groups = {};
  for (const g of ORG_GROUPS) groups[g] = [];
  for (const p of players) {
    const key = ORG_GROUPS.includes(p.orgGroup) ? p.orgGroup : "Junior / NCAA";
    groups[key].push(p);
  }
  return groups;
}

export function readinessClass(tier) {
  const t = str(tier, "").toLowerCase();
  if (t === "nhl ready") return "ready";
  if (t === "close") return "close";
  if (t === "at risk") return "risk";
  if (t === "stalled" || t === "regressed") return "warn";
  return "neutral";
}

export function trendClass(trend) {
  const t = str(trend, "").toLowerCase();
  if (t === "breakout" || t === "improved") return "up";
  if (t === "stalled" || t === "regressed") return "warn";
  return "neutral";
}

export function ovrDeltaClass(delta) {
  const n = num(delta);
  if (n >= 0.4) return "up";
  if (n <= -0.4) return "down";
  return "neutral";
}

export function formatOvrDeltaDisplay(delta) {
  return formatOvrDelta(delta);
}

export function formatSeasonStats(player) {
  const s = player?.seasonStats || {};
  const gp = num(s.gp);
  if (!gp) return "";
  if (player.goalie) {
    const bits = [`${gp} GP`];
    if (s.starts) bits.push(`${s.starts} GS`);
    if (s.save_pct != null) bits.push(`${s.save_pct} SV%`);
    if (s.gaa != null) bits.push(`${s.gaa} GAA`);
    return bits.join(" · ");
  }
  const pts = num(s.points, num(s.goals) + num(s.assists));
  const line = `${gp} GP · ${num(s.goals)} G · ${num(s.assists)} A · ${pts} P`;
  return s.ppg ? `${line} · ${s.ppg} P/GP` : line;
}

// Backward-compatible exports for any legacy imports
export const FILTERS = DEV_FILTERS;
export function filterProspects(list, filterId) {
  return filterDevelopmentPlayers(list, filterId, "all");
}
export function summarizeProspects(list) {
  const s = summarizeDevelopmentReport({}, list);
  return {
    rising: s.improved,
    nhlReady: s.nhlReady,
    stalled: s.stalled,
    highRisk: list.filter((p) => p.notableCategory === "High Risk").length,
    lateBloomers: list.filter((p) => p.notableCategory === "Late Bloomer").length,
  };
}
export function trendChipClass(trend) {
  const c = trendClass(trend);
  if (c === "up") return "pd-chip--up";
  if (c === "warn") return "pd-chip--warn";
  return "pd-chip--neutral";
}
