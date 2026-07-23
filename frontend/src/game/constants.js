/** Fixed logical resolution (scaled to fit viewport). */
export const GAME_W = 1280;
export const GAME_H = 720;

export const SCREENS = {
  SETUP: "setup",
  HUB: "hub",
  ROSTER: "roster",
  CALENDAR: "calendar",
  STORYLINES: "storylines",
  STATS: "stats",
  TRADE: "trade",
  DRAFT_CLASS: "draft_class",
  DRAFT_LOTTERY: "draft_lottery",
  TEAM_NEEDS: "team_needs",
  SCOUTING: "scouting",
  CHEMISTRY: "chemistry",
  EDIT_LINES: "edit_lines",
  POWER_PLAY: "power_play",
  PENALTY_KILL: "penalty_kill",
  CAP_LEDGER: "cap_ledger",
  OFFICE: "office",
  GM_WORLD: "gm_world",
  LEAGUE_OPERATIONS: "league_operations",
  SETTINGS: "settings",
  PLACEHOLDER: "placeholder",
};

/** Hub left-rail entries (controller order). */
export const HUB_MENU = [
  { id: "roster", label: "ROSTER" },
  { id: "calendar", label: "CALENDAR" },
  { id: "stats", label: "SCOUTING / INTEL" },
  { id: "ops", label: "TRADE FLOOR" },
  { id: "draft_class", label: "DRAFT CLASS" },
  { id: "office", label: "GM OFFICE" },
  { id: "settings", label: "SYSTEMS" },
  { id: "new", label: "NEW FRANCHISE" },
];

/** Discrete slider rows (local UI rules — not wired to sim yet). */
export const SETTINGS_ROWS = [
  { key: "roughing", label: "Roughing" },
  { key: "hooking", label: "Hooking" },
  { key: "slashing", label: "Slashing" },
  { key: "interference", label: "Interference" },
];

/** Canonical NHL code -> full team name (single source of truth). */
export const NHL_TEAM_NAME_BY_ABBR = Object.freeze({
  ANA: "Anaheim Ducks",
  BOS: "Boston Bruins",
  BUF: "Buffalo Sabres",
  CGY: "Calgary Flames",
  CAR: "Carolina Hurricanes",
  CHI: "Chicago Blackhawks",
  COL: "Colorado Avalanche",
  CBJ: "Columbus Blue Jackets",
  DAL: "Dallas Stars",
  DET: "Detroit Red Wings",
  EDM: "Edmonton Oilers",
  FLA: "Florida Panthers",
  LAK: "Los Angeles Kings",
  MIN: "Minnesota Wild",
  MTL: "Montreal Canadiens",
  NSH: "Nashville Predators",
  NJD: "New Jersey Devils",
  NYI: "New York Islanders",
  NYR: "New York Rangers",
  OTT: "Ottawa Senators",
  PHI: "Philadelphia Flyers",
  PIT: "Pittsburgh Penguins",
  SEA: "Seattle Kraken",
  SJS: "San Jose Sharks",
  STL: "St. Louis Blues",
  TBL: "Tampa Bay Lightning",
  TOR: "Toronto Maple Leafs",
  UTA: "Utah Hockey Club",
  VAN: "Vancouver Canucks",
  VGK: "Vegas Golden Knights",
  WSH: "Washington Capitals",
  WPG: "Winnipeg Jets",
});

/** Immediate 32-club list for setup UI; API may replace with engine ids when /franchise/teams returns. */
export function buildDefaultFranchiseTeamList() {
  return Object.entries(NHL_TEAM_NAME_BY_ABBR).map(([abbr, name]) => ({
    team_id: abbr,
    name,
  }));
}

export const NHL_ABBR_SET = new Set(Object.keys(NHL_TEAM_NAME_BY_ABBR));

export const NHL_ABBR_ALIASES = Object.freeze({
  MON: "MTL",
  TB: "TBL",
  LV: "VGK",
  LAS: "VGK",
  PHX: "UTA",
  ARI: "UTA",
  UTH: "UTA",
  LA: "LAK",
  SJ: "SJS",
  ANH: "ANA",
  NJS: "NJD",
  WAS: "WSH",
  WIN: "WPG",
});

export const NHL_RIVALRY_PAIRS = Object.freeze([
  ["BOS", "MTL"],
  ["TOR", "MTL"],
  ["TOR", "OTT"],
  ["NYR", "NYI"],
  ["NYR", "NJD"],
  ["NYR", "PHI"],
  ["PIT", "PHI"],
  ["PIT", "WSH"],
  ["CHI", "DET"],
  ["EDM", "CGY"],
  ["LAK", "ANA"],
  ["LAK", "SJS"],
  ["VAN", "CGY"],
  ["WPG", "MIN"],
  ["SEA", "VAN"],
  ["TBL", "FLA"],
]);

function _nhlPairKey(a, b) {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

export const NHL_RIVALRY_LOOKUP = new Set(
  NHL_RIVALRY_PAIRS.map(([a, b]) => _nhlPairKey(String(a), String(b)))
);

const NHL_NAME_TO_ABBR = Object.freeze(
  Object.fromEntries(
    Object.entries(NHL_TEAM_NAME_BY_ABBR).map(([abbr, name]) => [String(name).toLowerCase(), String(abbr)])
  )
);

export function normalizeNhlAbbr(raw) {
  let s = String(raw || "").trim().toUpperCase().replace(/[^A-Z]/g, "");
  if (!s) return "";
  if (NHL_ABBR_ALIASES[s]) s = NHL_ABBR_ALIASES[s];
  if (NHL_ABBR_SET.has(s)) return s;
  const tri = s.slice(0, 3);
  if (NHL_ABBR_SET.has(tri)) return tri;
  return "";
}

export function teamNameToNhlAbbr(teamNameRaw) {
  const nm = String(teamNameRaw || "").trim().toLowerCase();
  if (!nm) return "";
  if (NHL_NAME_TO_ABBR[nm]) return NHL_NAME_TO_ABBR[nm];
  for (const [full, ab] of Object.entries(NHL_NAME_TO_ABBR)) {
    if (nm.includes(full)) return ab;
  }
  return "";
}

export function isRivalryMatchup(a, b) {
  const x = normalizeNhlAbbr(a);
  const y = normalizeNhlAbbr(b);
  if (!x || !y || x === y) return false;
  return NHL_RIVALRY_LOOKUP.has(_nhlPairKey(x, y));
}

/** Plain string for timeline / notification / feed items (API may send structured objects). */
export function franchiseFeedText(item) {
  if (item == null) return "";
  if (typeof item === "string" || typeof item === "number" || typeof item === "boolean") {
    return String(item);
  }
  if (typeof item !== "object") return String(item);
  const o = item;
  const t = o.text ?? o.headline ?? o.title ?? o.message ?? o.body ?? o.description;
  if (t != null && String(t).trim() !== "") return String(t);
  try {
    return JSON.stringify(o);
  } catch {
    return "[Franchise feed item]";
  }
}
