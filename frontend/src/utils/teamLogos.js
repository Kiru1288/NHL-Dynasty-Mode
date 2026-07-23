/**
 * Resolve NHL team logo assets from frontend/src/logos for any screen.
 */

const LOGO_CONTEXT = (() => {
  try {
    return require.context("../logos", false, /\.(png|jpe?g|webp|svg)$/i);
  } catch (_err) {
    return null;
  }
})();

const TEAM_LOGO_NAME_OVERRIDES = {
  ANA: "Anaheim",
  ARI: "Arizona",
  BOS: "Boston",
  BUF: "Buffalo",
  CGY: "Calgary",
  CAR: "Carolina",
  CHI: "Chicago",
  COL: "Colorado",
  CBJ: "Columbus",
  DAL: "Dallas",
  DET: "Detroit",
  EDM: "Edmonton",
  FLA: "Florida",
  LAK: "Los Angeles",
  MIN: "Minnesota",
  MTL: "Montreal",
  MON: "Montreal",
  NSH: "Nashville",
  NJD: "New Jersey",
  NYI: "NY Islanders",
  NYR: "NY Rangers",
  OTT: "Ottawa",
  PHI: "Philadelphia",
  PIT: "Pittsburgh",
  SJS: "San Jose",
  SEA: "Seattle",
  STL: "St. Louis",
  TAM: "Tampa Bay",
  TBL: "Tampa Bay",
  TOR: "Toronto",
  UTA: "Arizona",
  UHC: "Arizona",
  VAN: "Vancouver",
  VGK: "Vegas",
  WSH: "Washington",
  WPG: "Winnipeg",
};

/** Full NHL display names (normalized) -> logo file stem key */
const NHL_CANONICAL_NAME_TO_LOGO = {
  anaheimducks: "Anaheim",
  arizonacoyotes: "Arizona",
  bostonbruins: "Boston",
  buffalosabres: "Buffalo",
  calgaryflames: "Calgary",
  carolinahurricanes: "Carolina",
  chicagoblackhawks: "Chicago",
  coloradoavalanche: "Colorado",
  columbusbluejackets: "Columbus",
  dallasstars: "Dallas",
  detroitredwings: "Detroit",
  edmontonoilers: "Edmonton",
  floridapanthers: "Florida",
  losangeleskings: "Los Angeles",
  minnesotawild: "Minnesota",
  montrealcanadiens: "Montreal",
  nashvillepredators: "Nashville",
  newjerseydevils: "New Jersey",
  newyorkislanders: "NY Islanders",
  newyorkrangers: "NY Rangers",
  ottawasenators: "Ottawa",
  philadelphiaflyers: "Philadelphia",
  pittsburghpenguins: "Pittsburgh",
  sanjosesharks: "San Jose",
  seattlekraken: "Seattle",
  stlouisblues: "St. Louis",
  tampabaylightning: "Tampa Bay",
  torontomapleleafs: "Toronto",
  utahhockeyclub: "Arizona",
  utahmammoth: "Arizona",
  utah: "Arizona",
  vancouvercanucks: "Vancouver",
  vegasgoldenknights: "Vegas",
  washingtoncapitals: "Washington",
  winnipegjets: "Winnipeg",
};

const TEAM_LOGO_MAP = (() => {
  const map = new Map();
  if (!LOGO_CONTEXT) return map;

  LOGO_CONTEXT.keys().forEach((key) => {
    const src = LOGO_CONTEXT(key);
    const rawFile = String(key || "").replace(/^.\//, "");
    const stem = rawFile.replace(/\.[^.]+$/, "");
    const cleaned = stem.replace(/\s+\d+$/, "").trim();
    const normalized = normalizeLogoToken(cleaned);

    if (normalized && !map.has(normalized)) {
      map.set(normalized, src);
    }
  });

  return map;
})();

export function normalizeLogoToken(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "")
    .trim();
}

function firstPresent(...values) {
  for (const value of values) {
    if (value !== null && value !== undefined && value !== "") {
      return value;
    }
  }
  return undefined;
}

export function getTeamAbbreviation(team) {
  if (!team) return "";

  if (typeof team === "string") {
    const s = String(team).trim();
    if (!s) return "";
    if (s.length <= 4 && !s.includes(" ")) return s.toUpperCase();
    if (s.includes(" ")) return s.split(/\s+/)[0].slice(0, 3).toUpperCase();
    return s.slice(0, 3).toUpperCase();
  }

  const rawExplicit = firstPresent(
    team.abbrev,
    team.abbr,
    team.abbreviation,
    team.short_name,
    team.shortName,
    team.code,
    team.team_abbrev
  );

  if (rawExplicit) return String(rawExplicit).slice(0, 4).toUpperCase();

  const city = String(team.city || team.location || team.market || "").trim();
  if (city) return city.slice(0, 3).toUpperCase();

  const nm = String(team.name || team.team_name || team.full_name || "").trim();
  if (nm) {
    if (nm.includes(" ")) return nm.split(/\s+/)[0].slice(0, 3).toUpperCase();
    return nm.slice(0, 3).toUpperCase();
  }

  return String(team.id || team.team_id || "").slice(0, 3).toUpperCase();
}

function logoStemFromCanonicalName(value) {
  const key = normalizeLogoToken(value);
  if (!key) return "";
  return NHL_CANONICAL_NAME_TO_LOGO[key] || "";
}

export function getTeamLogoSrc(team) {
  if (!team) return null;

  if (typeof team === "string") {
    const raw = String(team).trim();
    if (!raw) return null;
    const fromCanon = logoStemFromCanonicalName(raw);
    if (fromCanon) {
      const src = TEAM_LOGO_MAP.get(normalizeLogoToken(fromCanon));
      if (src) return src;
    }
    const abbr = raw.length <= 4 && !raw.includes(" ") ? raw.toUpperCase() : "";
    const preferred = TEAM_LOGO_NAME_OVERRIDES[abbr] || "";
    if (preferred) {
      const src = TEAM_LOGO_MAP.get(normalizeLogoToken(preferred));
      if (src) return src;
    }
    return TEAM_LOGO_MAP.get(normalizeLogoToken(raw));
  }

  const rowAbbr = String(
    firstPresent(team.team_abbrev, team.team_abbr, team.abbrev, team.abbr) || ""
  ).toUpperCase();

  if (rowAbbr && TEAM_LOGO_NAME_OVERRIDES[rowAbbr]) {
    const src = TEAM_LOGO_MAP.get(normalizeLogoToken(TEAM_LOGO_NAME_OVERRIDES[rowAbbr]));
    if (src) return src;
  }

  const abbr = String(getTeamAbbreviation(team) || "").toUpperCase();
  const preferredName = TEAM_LOGO_NAME_OVERRIDES[abbr] || logoStemFromCanonicalName(team.name || team.team_name);

  const candidates = [
    preferredName,
    logoStemFromCanonicalName(team.full_name || team.fullName),
    logoStemFromCanonicalName(team.name),
    logoStemFromCanonicalName(team.team_name),
    logoStemFromCanonicalName([team.city, team.name].filter(Boolean).join(" ")),
    team.full_name,
    team.fullName,
    team.name,
    team.team_name,
    team.nickname,
    [team.city, team.name].filter(Boolean).join(" "),
    team.city,
    abbr,
  ]
    .map((value) => String(value || "").trim())
    .filter(Boolean);

  for (let i = 0; i < candidates.length; i += 1) {
    const src = TEAM_LOGO_MAP.get(normalizeLogoToken(candidates[i]));
    if (src) return src;
  }

  return null;
}

/** Webpack/Cra logo imports may be a string URL or `{ default: url }`. */
export function toLogoUrl(src) {
  if (!src) return "";
  if (typeof src === "string") return src;
  if (typeof src === "object" && src.default != null) {
    return toLogoUrl(src.default);
  }
  return "";
}

/**
 * Resolve a displayable logo URL for franchise UI (office hub, headers, etc.).
 * Prefers bundled NHL logos from team name/abbrev before remote logo fields.
 */
export function resolveFranchiseTeamLogo(team, teamNameFallback = "") {
  const teamLike =
    team && typeof team === "object"
      ? team
      : { name: teamNameFallback || String(team || ""), team_name: teamNameFallback || String(team || "") };

  const bundled = getTeamLogoSrc(teamLike);
  if (bundled) return toLogoUrl(bundled);

  if (teamNameFallback && !teamLike.name) {
    const fromName = getTeamLogoSrc({
      name: teamNameFallback,
      team_name: teamNameFallback,
    });
    if (fromName) return toLogoUrl(fromName);
  }

  if (team && typeof team === "object") {
    const remote = firstPresent(
      team.logo,
      team.logo_url,
      team.logoUrl,
      team.team_logo,
      team.team_logo_src,
      team.image,
      team.crest,
      team.logoPath,
      team.logo_path
    );
    if (remote) return toLogoUrl(remote);
  }

  return "";
}

function teamRichnessScore(team) {
  if (!team || typeof team !== "object") return 0;

  let score = 0;
  const name = String(firstPresent(team.name, team.team_name, team.full_name) || "").trim();

  if (name && !/^\d+$/.test(name)) score += 4;
  if (team.full_name || team.fullName) score += 2;
  if (firstPresent(team.abbrev, team.abbr, team.code, team.team_abbrev)) score += 2;
  if (team.city) score += 1;
  if (getTeamLogoSrc(team)) score += 8;

  return score;
}

function mergeTeamPool(...lists) {
  const byId = new Map();

  const absorb = (team) => {
    if (!team || typeof team !== "object") return;

    const id = String(firstPresent(team.team_id, team.id) || "");
    if (!id) return;

    const existing = byId.get(id);
    if (!existing || teamRichnessScore(team) > teamRichnessScore(existing)) {
      byId.set(id, team);
    }
  };

  lists.forEach((list) => {
    (list || []).forEach(absorb);
  });

  return Array.from(byId.values());
}

export function collectFranchiseTeams(franchiseState = {}, normalizedTeams = []) {
  const statsDirectory =
    franchiseState?.stats_central?.teams_directory ||
    franchiseState?.statsCentral?.teams_directory ||
    [];

  const standings = (franchiseState?.standings || []).map((row) => ({
    team_id: row.team_id,
    id: row.team_id,
    name: row.name,
    team_name: row.name,
    full_name: row.name,
    abbrev: row.abbrev,
    abbr: row.abbr,
  }));

  const orgs = (franchiseState?.roster_browser?.organizations || []).map((org) => ({
    team_id: org.team_id,
    id: org.team_id,
    name: org.name,
    team_name: org.name,
    full_name: org.name,
  }));

  const rosterBrowser = franchiseState?.roster_browser;
  const rosterExtras = rosterBrowser
    ? [
        ...(rosterBrowser.teams || []),
        ...Object.values(rosterBrowser.league_teams || rosterBrowser.by_team || {}),
      ]
    : [];

  return mergeTeamPool(
    statsDirectory,
    orgs,
    standings,
    franchiseState?.teams,
    franchiseState?.league_teams,
    franchiseState?.all_teams,
    rosterExtras,
    normalizedTeams
  );
}

export function buildTeamLogoIndex(franchiseState = {}, normalizedTeams = []) {
  const index = new Map();

  collectFranchiseTeams(franchiseState, normalizedTeams).forEach((team) => {
    const id = String(firstPresent(team.team_id, team.id) || "");
    const logoSrc = getTeamLogoSrc(team);

    if (id && logoSrc) {
      index.set(id, logoSrc);
    }
  });

  return index;
}

export function resolvePlayerTeam(player, teams = [], franchiseState = {}) {
  if (!player) return null;

  const teamId = String(firstPresent(player.team_id, player.team, player.teamId) || "");
  const pool = collectFranchiseTeams(franchiseState, teams);

  const matches = pool.filter((t) => String(firstPresent(t.team_id, t.id) || "") === teamId);
  const byId =
    matches.sort((a, b) => teamRichnessScore(b) - teamRichnessScore(a))[0] || null;
  if (byId) return byId;

  const abbrev = String(
    firstPresent(player.team_abbrev, player.abbrev, player.team_abbr) || ""
  ).toUpperCase();

  if (abbrev) {
    const byAbbr = pool.find(
      (t) => String(getTeamAbbreviation(t) || "").toUpperCase() === abbrev
    );
    if (byAbbr) return byAbbr;
  }

  const teamName = String(firstPresent(player.team_name, player.team) || "").trim();
  if (teamName && !/^\d+$/.test(teamName)) {
    const lower = teamName.toLowerCase();
    const byName = pool.find((t) => {
      const nm = String(t.name || t.team_name || t.full_name || "").toLowerCase();
      return nm && nm === lower;
    });
    if (byName) return byName;
  }

  return {
    team_id: teamId,
    id: teamId,
    abbrev,
    abbr: abbrev,
    name: teamName || abbrev || teamId,
    team_name: teamName,
    city: player.team_city,
    full_name: teamName,
  };
}

export function getPlayerTeamLogoSrc(player, teams = [], franchiseState = {}) {
  if (!player) return null;

  const abbrev = String(
    firstPresent(player.team_abbrev, player.team_abbr, player.abbrev) || ""
  ).toUpperCase();

  if (abbrev && TEAM_LOGO_NAME_OVERRIDES[abbrev]) {
    const fromAbbr = getTeamLogoSrc({ abbrev, team_abbrev: abbrev });
    if (fromAbbr) return fromAbbr;
  }

  const teamName = String(firstPresent(player.team_name, player.team) || "").trim();
  if (teamName && !/^\d+$/.test(teamName)) {
    const fromName = getTeamLogoSrc(teamName);
    if (fromName) return fromName;
  }

  return getTeamLogoSrc(resolvePlayerTeam(player, teams, franchiseState));
}

export function attachLogosToTeamRows(teams, franchiseState = {}) {
  const payloadDirectory =
    franchiseState?.stats_central?.teams_directory ||
    franchiseState?.statsCentral?.teams_directory ||
    [];
  const pool = collectFranchiseTeams(franchiseState, [...(teams || []), ...(payloadDirectory || [])]);
  const logoIndex = buildTeamLogoIndex(franchiseState, [...(teams || []), ...(payloadDirectory || [])]);

  return (teams || []).map((team) => {
    const teamId = String(firstPresent(team?.team_id, team?.id) || "");
    const poolRow = pool.find((t) => String(firstPresent(t.team_id, t.id) || "") === teamId);

    const enriched = {
      ...team,
      team_id: teamId || team?.team_id,
      id: teamId || team?.id,
      name: firstPresent(
        team?.name,
        team?.team_name,
        poolRow?.name,
        poolRow?.team_name,
        teamId
      ),
      abbrev: firstPresent(
        team?.abbrev,
        team?.abbr,
        team?.team_abbrev,
        team?.team_abbr,
        poolRow?.abbrev,
        poolRow?.team_abbrev,
        getTeamAbbreviation(poolRow || team)
      ),
      team_abbrev: firstPresent(
        team?.team_abbrev,
        team?.team_abbr,
        team?.abbrev,
        team?.abbr,
        poolRow?.team_abbrev,
        poolRow?.abbrev
      ),
    };

    const team_logo_src =
      team?.team_logo_src ||
      (teamId ? logoIndex.get(teamId) : null) ||
      getTeamLogoSrc(enriched) ||
      getTeamLogoSrc(poolRow);

    if (!team_logo_src) return enriched;

    return { ...enriched, team_logo_src };
  });
}

export function attachTeamLogosToRows(rows, teams = [], franchiseState = {}) {
  const payloadDirectory =
    franchiseState?.stats_central?.teams_directory ||
    franchiseState?.statsCentral?.teams_directory ||
    [];
  const logoIndex = buildTeamLogoIndex(franchiseState, [
    ...(teams || []),
    ...(payloadDirectory || []),
  ]);

  return (rows || []).map((row) => {
    const teamId = String(firstPresent(row.team_id, row.team, row.teamId) || "");
    const team = resolvePlayerTeam(row, teams, franchiseState);
    const team_logo_src =
      row.team_logo_src ||
      (teamId ? logoIndex.get(teamId) : null) ||
      getPlayerTeamLogoSrc(row, teams, franchiseState);

    if (!team_logo_src) return row;

    return {
      ...row,
      team_logo_src,
      team_name: firstPresent(row.team_name, team?.name, team?.team_name, team?.full_name),
      team_abbrev: firstPresent(
        row.team_abbrev,
        row.team_abbr,
        team?.abbrev,
        team?.abbr,
        getTeamAbbreviation(team)
      ),
    };
  });
}
