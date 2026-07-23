/**
 * Prospect team → league lookup (mirrors SimEngine prospect_league_teams.py).
 * Fixes display when backend rows have mismatched team/league (e.g. Rimouski + NCAA).
 */

function normalizeTeamKey(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function addTeam(lookup, leagueCode, leagueDisplay, city, name) {
  const entry = { leagueCode, leagueDisplay, teamName: name, city };
  [name, city, name.split(" ").slice(-1)[0]].forEach((token) => {
    const key = normalizeTeamKey(token);
    if (key && !lookup[key]) lookup[key] = entry;
  });
  lookup[normalizeTeamKey(name)] = entry;
}

function buildLookup() {
  const lookup = {};
  const add = (code, display, teams) => {
    teams.forEach(([city, name]) => addTeam(lookup, code, display, city, name));
  };

  add("CHL_QMJHL", "QMJHL", [
    ["Rimouski", "Rimouski Oceanic"],
    ["Quebec City", "Quebec Remparts"],
    ["Halifax", "Halifax Mooseheads"],
    ["Moncton", "Moncton Wildcats"],
    ["Shawinigan", "Shawinigan Cataractes"],
    ["Gatineau", "Gatineau Olympiques"],
    ["Drummondville", "Drummondville Voltigeurs"],
    ["Charlottetown", "Charlottetown Islanders"],
    ["Saint John", "Saint John Sea Dogs"],
    ["Sherbrooke", "Sherbrooke Phoenix"],
    ["Rouyn-Noranda", "Rouyn-Noranda Huskies"],
    ["Val-d'Or", "Val-d'Or Foreurs"],
    ["Victoriaville", "Victoriaville Tigres"],
    ["Baie-Comeau", "Baie-Comeau Drakkar"],
    ["Blainville-Boisbriand", "Blainville-Boisbriand Armada"],
    ["Sydney", "Cape Breton Eagles"],
    ["Chicoutimi", "Chicoutimi Sagueneens"],
    ["St. John's", "Newfoundland Regiment"],
  ]);

  add("CHL_OHL", "OHL", [
    ["London", "London Knights"],
    ["Kitchener", "Kitchener Rangers"],
    ["Windsor", "Windsor Spitfires"],
    ["Ottawa", "Ottawa 67's"],
    ["Barrie", "Barrie Colts"],
    ["Saginaw", "Saginaw Spirit"],
    ["Kingston", "Kingston Frontenacs"],
    ["Guelph", "Guelph Storm"],
    ["Peterborough", "Peterborough Petes"],
    ["Oshawa", "Oshawa Generals"],
    ["Sudbury", "Sudbury Wolves"],
    ["North Bay", "North Bay Battalion"],
    ["Sarnia", "Sarnia Sting"],
    ["Erie", "Erie Otters"],
    ["Flint", "Flint Firebirds"],
    ["Niagara", "Niagara IceDogs"],
    ["Owen Sound", "Owen Sound Attack"],
    ["Brantford", "Brantford Bulldogs"],
    ["Brampton", "Brampton Steelheads"],
    ["Sault Ste. Marie", "Soo Greyhounds"],
  ]);

  add("CHL_WHL", "WHL", [
    ["Seattle", "Seattle Thunderbirds"],
    ["Portland", "Portland Winterhawks"],
    ["Kelowna", "Kelowna Rockets"],
    ["Calgary", "Calgary Hitmen"],
    ["Edmonton", "Edmonton Oil Kings"],
    ["Regina", "Regina Pats"],
    ["Saskatoon", "Saskatoon Blades"],
    ["Brandon", "Brandon Wheat Kings"],
    ["Medicine Hat", "Medicine Hat Tigers"],
    ["Lethbridge", "Lethbridge Hurricanes"],
    ["Red Deer", "Red Deer Rebels"],
    ["Prince George", "Prince George Cougars"],
    ["Vancouver", "Vancouver Giants"],
    ["Victoria", "Victoria Royals"],
    ["Spokane", "Spokane Chiefs"],
    ["Tri-City", "Tri-City Americans"],
    ["Moose Jaw", "Moose Jaw Warriors"],
    ["Swift Current", "Swift Current Broncos"],
    ["Kamloops", "Kamloops Blazers"],
    ["Everett", "Everett Silvertips"],
    ["Penticton", "Penticton Vees"],
    ["Prince Albert", "Prince Albert Raiders"],
    ["Wenatchee", "Wenatchee Wild"],
  ]);

  add("USHL", "USHL", [
    ["Omaha", "Omaha Lancers"],
    ["Dubuque", "Dubuque Fighting Saints"],
    ["Fargo", "Fargo Force"],
    ["Chicago", "Chicago Steel"],
    ["Green Bay", "Green Bay Gamblers"],
    ["Sioux Falls", "Sioux Falls Stampede"],
    ["Youngstown", "Youngstown Phantoms"],
    ["Madison", "Madison Capitols"],
    ["Waterloo", "Waterloo Black Hawks"],
    ["Des Moines", "Des Moines Buccaneers"],
    ["Cedar Rapids", "Cedar Rapids RoughRiders"],
    ["Lincoln", "Lincoln Stars"],
    ["Muskegon", "Muskegon Lumberjacks"],
    ["Sioux City", "Sioux City Musketeers"],
    ["Kearney", "Tri-City Storm"],
    ["Plymouth", "USNTDP Juniors"],
  ]);

  add("NCAA", "NCAA", [
    ["Boston", "Boston College"],
    ["Boston", "Boston University"],
    ["Ann Arbor", "Michigan"],
    ["Grand Forks", "North Dakota"],
    ["Denver", "Denver"],
    ["Hamden", "Quinnipiac"],
    ["Minneapolis", "Minnesota"],
    ["Madison", "Wisconsin"],
    ["Providence", "Providence College"],
    ["Ithaca", "Cornell"],
    ["New Haven", "Yale"],
    ["Cambridge", "Harvard"],
  ]);

  add("EU_J_SHL", "J20 Nationell", [
    ["Gothenburg", "Frolunda HC"],
    ["Stockholm", "Djurgardens IF"],
    ["Malmo", "Malmo Redhawks"],
    ["Lulea", "Lulea HF"],
    ["Skelleftea", "Skelleftea AIK"],
    ["Timra", "Timra IK"],
  ]);

  add("EU_J_LIIGA", "U20 SM-sarja", [
    ["Tampere", "Tappara"],
    ["Helsinki", "HIFK"],
    ["Turku", "TPS"],
    ["Oulu", "Karpat"],
    ["Jyvaskyla", "JYP"],
  ]);

  add("EU_J_KHL_JR", "MHL", [
    ["Saint Petersburg", "SKA Saint Petersburg"],
    ["Moscow", "CSKA Moscow"],
    ["Kazan", "Ak Bars Kazan"],
    ["Yaroslavl", "Lokomotiv Yaroslavl"],
    ["Novosibirsk", "Sibir Novosibirsk"],
  ]);

  return lookup;
}

const TEAM_LOOKUP = buildLookup();

export function resolveLeagueFromTeam(teamName) {
  const raw = String(teamName || "").trim();
  if (!raw) return null;

  const direct = TEAM_LOOKUP[normalizeTeamKey(raw)];
  if (direct) return { ...direct };

  for (const token of raw.split(/\s+/)) {
    const hit = TEAM_LOOKUP[normalizeTeamKey(token)];
    if (hit) return { ...hit };
  }

  const key = normalizeTeamKey(raw);
  const partial = Object.entries(TEAM_LOOKUP).filter(
    ([k]) => key.includes(k) || k.includes(key)
  );
  if (partial.length === 1) return { ...partial[0][1] };
  return null;
}

export function normalizeProspectTeamName(teamName) {
  const hit = resolveLeagueFromTeam(teamName);
  return hit?.teamName || String(teamName || "").trim();
}

/** Merge team-derived league into a prospect/draft row for display. */
export function applyProspectLeagueTeamFix(row = {}) {
  const teamRaw = row.team_name || row.team || row.teamName || "";
  const hit = resolveLeagueFromTeam(teamRaw);
  if (!hit) return row;

  const out = { ...row };
  out.team_name = hit.teamName;
  out.team = hit.teamName;
  out.teamName = hit.teamName;

  const backendCode = String(row.league_code || row.leagueCode || "").toUpperCase();
  if (backendCode !== hit.leagueCode) {
    out.league_code = hit.leagueCode;
    out.leagueCode = hit.leagueCode;
    out.league_display = hit.leagueDisplay;
    out.leagueDisplay = hit.leagueDisplay;
    out.league = hit.leagueDisplay;
  }
  return out;
}

export { TEAM_LOOKUP, normalizeTeamKey };
