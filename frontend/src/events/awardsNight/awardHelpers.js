import { firstDefined, pickFranchiseData, safeArray } from "../shared/eventHelpers";
import { resolveFranchiseTeamLogo, toLogoUrl } from "../../utils/teamLogos";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";

/** Canonical display metadata keyed by normalized award id. */
export const AWARD_CATALOG = {
  presidents: {
    label: "Presidents' Trophy",
    short: "PREZ",
    kind: "team",
    order: 1,
    ceremonyOrder: 12,
    accent: "#93c5fd",
    glow: "rgba(147,197,253,.35)",
    trophyTone: "silver-blue",
    ceremonyTitle: "League Standard",
    stageLine: "Best regular-season record.",
    displayMetric: "PTS",
  },
  stanley: {
    label: "Stanley Cup",
    short: "CUP",
    kind: "team",
    order: 2,
    ceremonyOrder: 99,
    accent: "#fbbf24",
    glow: "rgba(251,191,36,.42)",
    trophyTone: "championship-gold",
    ceremonyTitle: "Champions Crowned",
    stageLine: "The final team standing.",
    displayMetric: "Champion",
  },
  conference_champions: {
    label: "Conference Champions",
    short: "CONF",
    kind: "team",
    order: 2.5,
    ceremonyOrder: 13,
    accent: "#93c5fd",
    glow: "rgba(147,197,253,.28)",
    trophyTone: "silver-blue",
    ceremonyTitle: "Conference Crowns",
    stageLine: "East and West champions.",
    displayMetric: "Champion",
  },
  art_ross: {
    label: "Art Ross Trophy",
    short: "ROSS",
    kind: "player",
    order: 3,
    ceremonyOrder: 8,
    accent: "#f472b6",
    glow: "rgba(244,114,182,.35)",
    trophyTone: "rose-gold",
    ceremonyTitle: "Scoring King",
    stageLine: "League leader in points.",
    displayMetric: "PTS",
  },
  rocket: {
    label: "Maurice Richard Trophy",
    short: "GOAL",
    kind: "player",
    order: 4,
    ceremonyOrder: 7,
    accent: "#fb7185",
    glow: "rgba(251,113,133,.36)",
    trophyTone: "goal-red",
    ceremonyTitle: "Goal Scoring Crown",
    stageLine: "Most goals in the league.",
    displayMetric: "G",
  },
  norris: {
    label: "James Norris Memorial Trophy",
    short: "NORR",
    kind: "player",
    order: 5,
    ceremonyOrder: 6,
    accent: "#60a5fa",
    glow: "rgba(96,165,250,.35)",
    trophyTone: "defense-blue",
    ceremonyTitle: "Blue Line Royalty",
    stageLine: "Premier defenseman.",
    displayMetric: "Ballot points",
  },
  hart: {
    label: "Hart Memorial Trophy",
    short: "HART",
    kind: "player",
    order: 6,
    ceremonyOrder: 10,
    accent: "#c084fc",
    glow: "rgba(192,132,252,.38)",
    trophyTone: "mvp-purple",
    ceremonyTitle: "Most Valuable Player",
    stageLine: "Most valuable all-around season.",
    displayMetric: "Ballot points",
  },
  selke: {
    label: "Frank J. Selke Trophy",
    short: "SELK",
    kind: "player",
    order: 7,
    ceremonyOrder: 3,
    accent: "#34d399",
    glow: "rgba(52,211,153,.34)",
    trophyTone: "two-way-green",
    ceremonyTitle: "Two-Way Standard",
    stageLine: "Elite defensive forward impact.",
    displayMetric: "Ballot points",
  },
  calder: {
    label: "Calder Memorial Trophy",
    short: "CALD",
    kind: "player",
    order: 8,
    ceremonyOrder: 1,
    accent: "#a3e635",
    glow: "rgba(163,230,53,.34)",
    trophyTone: "rookie-lime",
    ceremonyTitle: "Rookie Breakout",
    stageLine: "Top first-year player.",
    displayMetric: "Ballot points",
  },
  vezina: {
    label: "Vezina Trophy",
    short: "VEZI",
    kind: "player",
    order: 9,
    ceremonyOrder: 5,
    accent: "#38bdf8",
    glow: "rgba(56,189,248,.34)",
    trophyTone: "crease-cyan",
    ceremonyTitle: "Crease King",
    stageLine: "Best goaltending season.",
    displayMetric: "Ballot points",
  },
  conn_smythe: {
    label: "Conn Smythe Trophy",
    short: "SMYT",
    kind: "player",
    order: 10,
    ceremonyOrder: 11,
    accent: "#fbbf24",
    glow: "rgba(251,191,36,.34)",
    trophyTone: "championship-gold",
    ceremonyTitle: "Playoff MVP",
    stageLine: "Most valuable in the postseason.",
    displayMetric: "Playoff ballot points",
  },
  jennings: {
    label: "William M. Jennings Trophy",
    short: "JENN",
    kind: "player",
    order: 11,
    ceremonyOrder: 4,
    accent: "#67e8f9",
    glow: "rgba(103,232,249,.3)",
    trophyTone: "crease-cyan",
    ceremonyTitle: "Fewest Goals Against",
    stageLine: "Team goals-against leaders.",
    displayMetric: "Team GA",
  },
  lady_byng: {
    label: "Lady Byng Memorial Trophy",
    short: "BYNG",
    kind: "player",
    order: 12,
    ceremonyOrder: 2,
    accent: "#f9a8d4",
    glow: "rgba(249,168,212,.3)",
    trophyTone: "rose-gold",
    ceremonyTitle: "Gentlemanly Play",
    stageLine: "Skill with exceptional discipline.",
    displayMetric: "Ballot points",
  },
  ted_lindsay: {
    label: "Ted Lindsay Award",
    short: "LIND",
    kind: "player",
    order: 13,
    ceremonyOrder: 9,
    accent: "#c084fc",
    glow: "rgba(192,132,252,.3)",
    trophyTone: "mvp-purple",
    ceremonyTitle: "Players' Choice",
    stageLine: "Outstanding player by peer weighting.",
    displayMetric: "Ballot points",
  },
  masterton: {
    label: "Bill Masterton Memorial Trophy",
    short: "MAST",
    kind: "player",
    order: 14,
    ceremonyOrder: 14,
    accent: "#fdba74",
    glow: "rgba(253,186,116,.28)",
    trophyTone: "rose-gold",
    ceremonyTitle: "Perseverance",
    stageLine: "Dedication and adversity.",
    displayMetric: "Selection",
  },
  messier: {
    label: "Mark Messier Leadership Award",
    short: "MESS",
    kind: "player",
    order: 15,
    ceremonyOrder: 15,
    accent: "#93c5fd",
    glow: "rgba(147,197,253,.28)",
    trophyTone: "silver-blue",
    ceremonyTitle: "Leadership",
    stageLine: "Leadership and character.",
    displayMetric: "Selection",
  },
  jack_adams: {
    label: "Jack Adams Award",
    short: "ADAM",
    kind: "coach",
    order: 16,
    ceremonyOrder: 16,
    accent: "#86efac",
    glow: "rgba(134,239,172,.28)",
    trophyTone: "two-way-green",
    ceremonyTitle: "Coach of the Year",
    stageLine: "Bench leadership.",
    displayMetric: "Ballot points",
  },
  all_star_1: {
    label: "First NHL All-Star Team",
    short: "AS1",
    kind: "player",
    order: 17,
    ceremonyOrder: 17,
    accent: "#fde68a",
    glow: "rgba(253,230,138,.28)",
    trophyTone: "championship-gold",
    ceremonyTitle: "First Team",
    stageLine: "Season-end First All-Star Team.",
    displayMetric: "Selection",
  },
  all_star_2: {
    label: "Second NHL All-Star Team",
    short: "AS2",
    kind: "player",
    order: 18,
    ceremonyOrder: 18,
    accent: "#e2e8f0",
    glow: "rgba(226,232,240,.24)",
    trophyTone: "silver-blue",
    ceremonyTitle: "Second Team",
    stageLine: "Season-end Second All-Star Team.",
    displayMetric: "Selection",
  },
};

const AWARD_ALIASES = [
  ["presidents", ["presidents", "president", "presidents trophy", "presidents' trophy"]],
  ["stanley", ["stanley", "stanley cup", "cup"]],
  ["conference_champions", ["conference champions", "conference champion", "east champion", "west champion"]],
  ["art_ross", ["art ross", "art ross trophy"]],
  ["rocket", ["rocket", "rocket richard", "maurice richard", "richard trophy"]],
  ["norris", ["norris", "norris trophy", "james norris"]],
  ["hart", ["hart", "hart memorial", "mvp"]],
  ["selke", ["selke", "frank j selke"]],
  ["calder", ["calder", "calder memorial", "rookie of the year"]],
  ["vezina", ["vezina", "vezina trophy"]],
  ["conn_smythe", ["conn smythe", "smythe", "playoff mvp"]],
  ["jennings", ["jennings", "william jennings", "william m. jennings"]],
  ["lady_byng", ["lady byng", "byng"]],
  ["ted_lindsay", ["ted lindsay", "lindsay"]],
  ["masterton", ["masterton", "bill masterton"]],
  ["messier", ["messier", "mark messier"]],
  ["jack_adams", ["jack adams", "adams"]],
  ["all_star_1", ["first nhl all-star team", "first all-star", "all star first"]],
  ["all_star_2", ["second nhl all-star team", "second all-star", "all star second"]],
];

const RANDOM_FAN_API_URL = "https://randomuser.me/api/";

const FALLBACK_FAN_FIRST_NAMES = [
  "Mason",
  "Logan",
  "Avery",
  "Nolan",
  "Riley",
  "Carter",
  "Hudson",
  "Owen",
  "Theo",
  "Wyatt",
  "Miles",
  "Cole",
  "Jules",
  "Drew",
  "Quinn",
  "Reese",
  "Blake",
  "Hayden",
  "Rowan",
  "Casey",
  "Devon",
  "Parker",
  "Jamie",
  "Morgan",
];

const FALLBACK_FAN_LAST_NAMES = [
  "Puckett",
  "Crossbar",
  "Benches",
  "Stickside",
  "Bluepaint",
  "Icer",
  "Dumpin",
  "Chase",
  "Sauce",
  "Barnburner",
  "Fivehole",
  "Overtime",
  "Rinkwell",
  "Glassman",
  "Boardley",
  "Slotter",
  "Netfront",
  "Clapper",
];

const FAN_HANDLE_WORDS = [
  "puckwatch",
  "creaseburner",
  "neutralzone",
  "boardbattle",
  "capfriendlyish",
  "benchnoise",
  "slotshot",
  "rinkrat",
  "hockeypanic",
  "dumpandchange",
  "softdump",
  "powerplaymerchant",
  "wildtake",
  "goalienation",
  "statline",
  "forecheckfeed",
  "deadlinebrain",
  "overtimeclub",
  "pressboxwatch",
  "zoneentry",
];

const FAN_PERSONAS = [
  "diehard",
  "homer",
  "skeptic",
  "stat nerd",
  "chaos fan",
  "old-school fan",
  "prospect watcher",
  "talk-radio caller",
  "rival fan",
  "playoff worrier",
  "boxscore scout",
  "front-office critic",
];

const FAN_MARKETS = [
  "League Feed",
  "RinkSide",
  "North Stand",
  "Lower Bowl",
  "Pressbox Replies",
  "After Hours Hockey",
  "Puck Forum",
  "Fan Line",
  "Neutral Zone",
  "Late Night Thread",
];

export const AWARD_FAN_REACTION_TEMPLATES = {
  generic: [
    "{winner} winning {award} feels right. The {topStat} number makes it hard to argue.",
    "I need the full voting breakdown, but {winner} taking {award} is not shocking at all.",
    "{award} discourse is about to be unbearable and honestly I am here for it.",
    "{winner} just added a real legacy line tonight. {legacy}",
    "People can debate the winner, but {winner} had the season everyone noticed.",
    "The finalists were strong, but {winner} always felt like the name they were building toward.",
    "That {award} race was closer than people want to admit.",
    "{winner} winning is going to age either perfectly or terribly. No middle ground.",
    "You can tell who watched the games and who only read the stat table from this {award} reaction.",
    "{winner} just became part of league history. That is the point of awards night.",
  ],
  presidents: [
    "{winner} winning the Presidents' Trophy after that regular season is fair. {topStat} says enough.",
    "Best regular-season team gets the hardware. People can hate it, but {winner} earned this.",
    "{winner} set the pace all year. Now the real pressure starts.",
    "The Presidents' Trophy is nice, but {winner} knows nobody relaxes until the playoffs are done.",
    "{winner} fans should enjoy this for about five minutes before everyone starts yelling about the curse.",
    "{winner} was the standard for 82 games. That matters, even if people pretend it does not.",
    "Regular season monsters. Now {winner} has to make it mean something.",
    "The league spent months chasing {winner}. This award is not complicated.",
  ],
  stanley: [
    "{winner} surviving the playoff grind and lifting the Cup is the whole story. No notes.",
    "{winner} fans are never going to let anyone forget this run.",
    "The Cup is home with {winner}. That sentence still feels insane.",
    "{winner} did not just win. They outlasted everyone.",
    "Every risky move looks genius when {winner} ends the year with the Cup.",
    "{winner} just turned a long season into franchise history.",
    "This is why you go all in. {winner} finished the job.",
    "The final team standing is {winner}. That is the only argument that matters tonight.",
  ],
  art_ross: [
    "{winner} winning the Art Ross with {topStat} is absurd production.",
    "Scoring race ends with {winner} on top. That tracks.",
    "{winner} led the league in points and still somehow people will call it quiet.",
    "The Art Ross going to {winner} feels like the least surprising part of awards night.",
    "{winner} owned the scoring race for long stretches. This is deserved.",
    "You do not luck into the Art Ross. {winner} was a problem every night.",
    "{winner} just won the league's production crown. That season was pure offense.",
    "The point total was not just good. It shaped the whole season around {winner}.",
  ],
  rocket: [
    "{winner} winning the Rocket is perfect. Goal scorers are supposed to feel inevitable.",
    "{topStat} for {winner}. That is not a hot streak, that is a season-long warning sign.",
    "Every goalie in the league is relieved {winner}'s Rocket season is finally over.",
    "{winner} owned the goal column. No overthinking needed.",
    "You could feel the Rocket race bending toward {winner} for months.",
    "The Maurice Richard Trophy going to {winner} just makes sense.",
    "{winner} scored like every game mattered because apparently it did.",
    "Goal scoring is still the loudest stat in hockey, and {winner} had the loudest season.",
  ],
  norris: [
    "{winner} winning the Norris is going to start arguments, but the season was real.",
    "Blue line minutes, production, pressure. {winner} checked every box.",
    "{winner} taking the Norris feels like a vote for control more than just points.",
    "The Norris debate is always messy, but {winner} had the resume.",
    "{winner} was basically running games from the blue line all year.",
    "Defensemen do not get casual seasons like this. {winner} earned the Norris conversation.",
    "If you watched {winner} every night, this Norris vote makes sense.",
    "The stat sheet helped, but {winner}'s impact was bigger than one number.",
  ],
  hart: [
    "{winner} winning the Hart means the league saw what everyone else saw.",
    "Most valuable player debates are always toxic, but {winner} had the season.",
    "{winner} getting the Hart is going to make one fanbase furious and another fanbase impossible to talk to.",
    "The Hart going to {winner} feels like the headline of the entire season.",
    "Take {winner} off that team and everything looks different. That is value.",
    "{winner} had the kind of season that turns every game into an argument.",
    "This Hart race was nasty, but {winner} was always the main character.",
    "That is a legacy award for {winner}. Not just a good-season award.",
  ],
  selke: [
    "{winner} winning the Selke is for the people who watch the little details.",
    "The Selke is never the loudest award, but {winner} earned it shift by shift.",
    "{winner} tilted the ice without needing every highlight to prove it.",
    "Two-way monster season from {winner}. The Selke makes sense.",
    "The box score does not always explain why {winner} won this, and that is kind of the point.",
    "{winner} got rewarded for doing the annoying winning-hockey stuff all year.",
    "Selke discourse is always funny because half the value happens before the shot even exists.",
    "Coaches probably love {winner}'s season more than fans realize.",
  ],
  calder: [
    "{winner} winning the Calder is how a fanbase starts dreaming way too early.",
    "Rookie of the year for {winner}. The future just got louder.",
    "{winner} did not look like a rookie for most of this season.",
    "The Calder going to {winner} feels like the start of something bigger.",
    "Every team thinks their rookie is different. {winner} actually backed it up.",
    "{winner} just turned a first impression into hardware.",
    "The league officially has a new problem if {winner} keeps building from here.",
    "Calder winner today, unrealistic fan expectations tomorrow. That is the sport.",
  ],
  vezina: [
    "{winner} winning the Vezina makes sense. Goaltending carried nights it had no business carrying.",
    "{topStat} from {winner} is exactly why this Vezina vote happened.",
    "Goalie awards are chaos, but {winner} gave voters a pretty clean answer.",
    "{winner} stole enough games to make this feel obvious.",
    "The Vezina going to {winner} is a reminder that goalies still break seasons.",
    "Every contender wants the year {winner} just had in net.",
    "{winner} was not just good. He changed what his team could survive.",
    "Vezina debates get weird, but {winner}'s season had the saves, wins, and moments.",
  ],
};

function normalizeKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/['ù]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function normalizePersonName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function asNumber(value, fallback = null) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function roundStat(value) {
  const n = asNumber(value);
  if (n === null) return "ù";
  return String(Math.round(n));
}

function oneDecimal(value) {
  const n = asNumber(value);
  if (n === null) return "ù";
  return n.toFixed(1);
}

function percent(value) {
  const n = asNumber(value);
  if (n === null) return "ù";
  if (n <= 1) return `${Math.round(n * 100)}%`;
  return `${Math.round(n)}%`;
}

function compactRecord(team) {
  const wins = asNumber(firstDefined(team?.wins, team?.w));
  const losses = asNumber(firstDefined(team?.losses, team?.l));
  const ot = asNumber(firstDefined(team?.ot_losses, team?.otl, team?.ot));
  if (wins === null || losses === null) return "";
  return ot === null ? `${wins}-${losses}` : `${wins}-${losses}-${ot}`;
}

function getPlayerName(player) {
  return String(player?.name || player?.full_name || player?.fullName || "ù").trim();
}

function getTeamName(team) {
  return String(
    team?.full_name ||
      team?.fullName ||
      team?.name ||
      team?.team_name ||
      team?.winnerLabel ||
      "ù"
  ).trim();
}

function hashString(value) {
  const str = String(value || "seed");
  let hash = 2166136261;

  for (let i = 0; i < str.length; i += 1) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

function seededFloat(seedValue) {
  let t = hashString(seedValue) + 0x6d2b79f5;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function seededInt(seedValue, min, max) {
  const low = Number(min) || 0;
  const high = Number(max) || low;
  if (high <= low) return low;
  return Math.floor(seededFloat(seedValue) * (high - low + 1)) + low;
}

function seededPick(list, seedValue, fallback = "") {
  const arr = safeArray(list).filter(Boolean);
  if (!arr.length) return fallback;
  return arr[seededInt(seedValue, 0, arr.length - 1)] || fallback;
}

function compactHandlePart(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "")
    .slice(0, 20);
}

function trimTweet(text, maxLength = 180) {
  const clean = String(text || "")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,!?;:])/g, "$1")
    .trim();

  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, maxLength - 1).trim()}ù`;
}

function replaceTemplate(template, values) {
  return String(template || "").replace(/\{([a-zA-Z0-9_]+)\}/g, (_, key) => {
    const value = values?.[key];
    return value === undefined || value === null || value === "" ? "ù" : String(value);
  });
}

export function resolveAwardKey(rawName) {
  const key = normalizeKey(rawName);
  for (const [id, aliases] of AWARD_ALIASES) {
    if (aliases.some((a) => key === a || key.includes(a))) return id;
  }
  return key.replace(/\s+/g, "_") || "award";
}

export function getAwardCatalogEntry(rawName) {
  const id = resolveAwardKey(rawName);
  return (
    AWARD_CATALOG[id] || {
      label: String(rawName || "Award").trim(),
      short:
        String(rawName || "AWD")
          .replace(/[^A-Za-z]/g, "")
          .slice(0, 4)
          .toUpperCase() || "AWD",
      kind: /president|stanley/i.test(String(rawName)) ? "team" : "player",
      order: 99,
      ceremonyOrder: 99,
      accent: "#94a3b8",
      glow: "rgba(148,163,184,.30)",
      trophyTone: "neutral-silver",
      ceremonyTitle: "Season Hardware",
      stageLine: "Award winner announced.",
    }
  );
}

function collectPlayerRows(franchiseState) {
  const seen = new Set();
  const rows = [];

  const push = (row) => {
    if (!row || typeof row !== "object") return;
    const id = String(row.player_id || row.id || "").trim();
    const name = String(row.name || row.full_name || row.fullName || "").trim();
    const sig = id || `${normalizePersonName(name)}:${String(row.team_id || row.teamId || "")}`;
    if (!sig || seen.has(sig)) return;
    seen.add(sig);
    rows.push(row);
  };

  const sc = franchiseState?.stats_central || {};
  for (const list of [
    sc.players,
    sc.skaters,
    sc.goalies,
    sc.leaders,
    sc.goalie_leaders,
    sc.user_leaders,
    franchiseState?.league_leaders,
    franchiseState?.player_leaders,
  ]) {
    safeArray(list).forEach(push);
  }

  safeArray(franchiseState?.roster).forEach(push);

  const browser = franchiseState?.roster_browser;
  if (browser && typeof browser === "object") {
    for (const block of Object.values(browser)) {
      if (Array.isArray(block)) block.forEach(push);
      else if (block?.players) safeArray(block.players).forEach(push);
    }
  }

  for (const team of safeArray(franchiseState?.league_teams)) {
    safeArray(team?.roster).forEach(push);
    safeArray(team?.players).forEach(push);
  }

  return rows;
}

export function resolveWinnerPlayer(award, franchiseState) {
  const winnerName = String(
    firstDefined(award?.winner_name, award?.winnerName, award?.winner) || ""
  ).trim();
  const teamId = String(firstDefined(award?.winner_team_id, award?.winnerTeamId) || "").trim();
  const playerId = String(firstDefined(award?.winner_player_id, award?.winnerPlayerId) || "").trim();

  const rows = collectPlayerRows(franchiseState);

  if (playerId) {
    const byId = rows.find((r) => String(r.player_id || r.id || "") === playerId);
    if (byId) return ensurePlayerHeadshotFields(byId);
  }

  if (!winnerName) return null;

  const target = normalizePersonName(winnerName);

  const exact =
    rows.find(
      (r) =>
        normalizePersonName(r.name || r.full_name || r.fullName) === target &&
        (!teamId || String(r.team_id || r.teamId || "") === teamId)
    ) || rows.find((r) => normalizePersonName(r.name || r.full_name || r.fullName) === target);

  if (exact) return ensurePlayerHeadshotFields(exact);

  return ensurePlayerHeadshotFields({
    name: winnerName,
    team_id: teamId,
    player_id: playerId,
    position: award?.winner_position || award?.position || "F",
  });
}

export function resolveWinnerTeam(award, franchiseState) {
  const teamId = String(firstDefined(award?.winner_team_id, award?.winnerTeamId) || "").trim();
  const winnerName = String(
    firstDefined(award?.winner_name, award?.winnerName, award?.winner) || ""
  ).trim();
  const winnerStats = award?.winner_stats || award?.winnerStats || {};

  let base = null;

  if (teamId) {
    const match = safeArray(franchiseState?.league_teams).find(
      (t) => String(t.id || t.team_id || t.teamId || "") === teamId
    );
    if (match) {
      base = {
        ...match,
        team_id: teamId,
        full_name:
          match.full_name ||
          match.fullName ||
          match.name ||
          match.team_name ||
          winnerName,
      };
    }
  }

  if (!base) {
    const fromStandings = safeArray(franchiseState?.standings).find(
      (t) => String(t.team_id || t.id || "") === teamId
    );

    if (fromStandings) {
      base = {
        ...fromStandings,
        team_id: teamId,
        full_name: fromStandings.name || fromStandings.team_name || winnerName,
        abbreviation: fromStandings.abbreviation || fromStandings.abbr,
      };
    }
  }

  if (!base) {
    base = {
      team_id: teamId,
      full_name: winnerName || "ù",
      name: winnerName || "ù",
    };
  }

  const record = base.record || {};
  const standingsRow = safeArray(franchiseState?.standings).find(
    (t) => String(t.team_id || t.id || "") === teamId
  );

  return {
    ...base,
    ...record,
    ...winnerStats,
    wins: firstDefined(winnerStats.wins, winnerStats.w, base.wins, base.w, standingsRow?.w),
    losses: firstDefined(winnerStats.losses, winnerStats.l, base.losses, base.l, standingsRow?.l),
    otl: firstDefined(winnerStats.otl, winnerStats.ot_losses, base.otl, standingsRow?.otl),
    points: firstDefined(winnerStats.points, winnerStats.pts, base.points, base.pts, standingsRow?.pts),
    goals_for: firstDefined(winnerStats.goals_for, winnerStats.gf, base.goals_for, base.gf),
    goals_against: firstDefined(winnerStats.goals_against, winnerStats.ga, base.goals_against, base.ga),
    goal_diff: firstDefined(winnerStats.goal_diff, base.goal_diff),
    record: winnerStats.record || compactRecord({
      wins: firstDefined(winnerStats.wins, winnerStats.w, base.wins, base.w, standingsRow?.w),
      losses: firstDefined(winnerStats.losses, winnerStats.l, base.losses, base.l, standingsRow?.l),
      ot_losses: firstDefined(winnerStats.otl, base.otl, standingsRow?.otl),
      w: standingsRow?.w,
      l: standingsRow?.l,
      otl: standingsRow?.otl,
    }),
  };
}

export function getAwardWinnerLabel(award, franchiseState) {
  const meta = getAwardCatalogEntry(award?.name);
  if (meta.kind === "team") {
    return getTeamName(resolveWinnerTeam(award, franchiseState));
  }
  return String(firstDefined(award?.winner_name, award?.winnerName, award?.winner) || "ù");
}

export function getTeamLogoSrc(team, franchiseState) {
  const name =
    team?.full_name ||
    team?.fullName ||
    team?.name ||
    team?.team_name ||
    "";

  return (
    toLogoUrl(team?.team_logo_src || team?.logo || team?.logo_url || team?.logoUrl) ||
    resolveFranchiseTeamLogo(team, name) ||
    ""
  );
}

function buildPlayerStatCards(award, player) {
  const key = award.awardKey;
  const source = { ...player, ...award };

  const goals = firstDefined(source.goals, source.g);
  const assists = firstDefined(source.assists, source.a);
  const points = firstDefined(source.points, source.pts);
  const plusMinus = firstDefined(source.plus_minus, source.plusMinus);
  const games = firstDefined(source.games_played, source.gp, source.games);
  const wins = firstDefined(source.wins, source.w);
  const savePct = firstDefined(source.save_pct, source.savePct, source.sv_pct);
  const gaa = firstDefined(source.gaa, source.goals_against_average);
  const shutouts = firstDefined(source.shutouts, source.so);
  const toi = firstDefined(source.toi, source.time_on_ice, source.avg_toi);
  const blocks = firstDefined(source.blocks, source.blocked_shots);
  const takeaways = firstDefined(source.takeaways, source.tk);
  const faceoffPct = firstDefined(source.faceoff_pct, source.fo_pct, source.faceoffPct);

  if (key === "vezina") {
    const svRaw = savePct;
    let svDisplay = "ù";

    if (svRaw != null && svRaw !== "") {
      const n = Number(svRaw);
      if (Number.isFinite(n)) {
        svDisplay = n <= 1 ? (n * 100).toFixed(1) : n.toFixed(1);
      }
    }

    return [
      { label: "Wins", value: roundStat(wins), tone: "primary" },
      { label: "Save %", value: svDisplay, suffix: "%" },
      { label: "GAA", value: gaa ? oneDecimal(gaa) : "ù" },
      { label: "Shutouts", value: roundStat(shutouts) },
    ];
  }

  if (key === "norris") {
    return [
      { label: "Points", value: roundStat(points), tone: "primary" },
      { label: "Goals", value: roundStat(goals) },
      { label: "Avg TOI", value: toi ? String(toi) : "ù" },
      { label: "Blocks", value: roundStat(blocks) },
    ];
  }

  if (key === "selke") {
    return [
      { label: "Points", value: roundStat(points), tone: "primary" },
      { label: "Takeaways", value: roundStat(takeaways) },
      { label: "Faceoffs", value: percent(faceoffPct) },
      {
        label: "+/-",
        value:
          plusMinus !== undefined
            ? `${Number(plusMinus) > 0 ? "+" : ""}${roundStat(plusMinus)}`
            : "ù",
      },
    ];
  }

  if (key === "rocket") {
    return [
      { label: "Goals", value: roundStat(goals), tone: "primary" },
      { label: "Games", value: roundStat(games) },
      { label: "Assists", value: roundStat(assists) },
      { label: "Points", value: roundStat(points) },
    ];
  }

  return [
    { label: "Points", value: roundStat(points), tone: "primary" },
    { label: "Goals", value: roundStat(goals) },
    { label: "Assists", value: roundStat(assists) },
    {
      label: "+/-",
      value:
        plusMinus !== undefined
          ? `${Number(plusMinus) > 0 ? "+" : ""}${roundStat(plusMinus)}`
          : "ù",
    },
  ];
}

function buildTeamStatCards(award, team) {
  const points = firstDefined(team?.points, team?.pts);
  const wins = firstDefined(team?.wins, team?.w);
  const losses = firstDefined(team?.losses, team?.l);
  const goalsFor = firstDefined(team?.goals_for, team?.gf);
  const goalsAgainst = firstDefined(team?.goals_against, team?.ga);
  const record = team?.record || compactRecord(team);
  const goalDiff =
    team?.goal_diff !== undefined && team?.goal_diff !== null
      ? Number(team.goal_diff)
      : goalsFor !== undefined && goalsAgainst !== undefined
        ? Number(goalsFor) - Number(goalsAgainst)
        : null;

  if (award.awardKey === "stanley") {
    return [
      { label: "Champions", value: "Cup", tone: "primary" },
      { label: "Playoffs", value: firstDefined(award?.playoff_record, team?.playoff_record) || "Won" },
      { label: "Record", value: record || "ù" },
      { label: "Season", value: roundStat(points), suffix: " pts" },
    ];
  }

  return [
    { label: "Points", value: roundStat(points), tone: "primary" },
    { label: "Wins", value: roundStat(wins) },
    { label: "Record", value: record || `${roundStat(wins)}-${roundStat(losses)}` },
    {
      label: "Goal Diff",
      value:
        goalDiff !== null && Number.isFinite(goalDiff)
          ? `${goalDiff > 0 ? "+" : ""}${goalDiff}`
          : "ù",
    },
  ];
}

function parseTeamStatsFromRationale(rationale) {
  const m = String(rationale || "").match(/\((\d+)\s*pts,\s*(\d+)-(\d+)-(\d+),\s*GD\s*([+-]?\d+)\)/i);
  if (!m) return null;

  return {
    points: Number(m[1]),
    pts: Number(m[1]),
    wins: Number(m[2]),
    w: Number(m[2]),
    losses: Number(m[3]),
    l: Number(m[3]),
    otl: Number(m[4]),
    goal_diff: Number(m[5]),
    record: `${m[2]}-${m[3]}-${m[4]}`,
  };
}

export const CEREMONY_RAIL_GROUPS = [
  { id: "team", label: "Team Awards", awardKeys: ["presidents"] },
  {
    id: "player",
    label: "Player Awards",
    awardKeys: ["calder", "selke", "vezina", "norris", "rocket", "art_ross", "hart"],
  },
  { id: "championship", label: "Championship", awardKeys: ["stanley"] },
];

export const SEASON_MILESTONES = [
  { id: "regular", label: "Regular Season" },
  { id: "playoffs", label: "Playoffs" },
  { id: "awards", label: "Awards" },
  { id: "draft_lottery", label: "Draft Lottery" },
  { id: "draft", label: "Draft" },
  { id: "free_agency", label: "Free Agency" },
];

function sortedStandings(franchiseState) {
  return [...safeArray(franchiseState?.standings)].sort(
    (a, b) => asNumber(b.pts, 0) - asNumber(a.pts, 0) || asNumber(b.w, 0) - asNumber(a.w, 0)
  );
}

function findStandingsRow(teamId, franchiseState) {
  const id = String(teamId || "").trim();
  if (!id) return null;
  return (
    sortedStandings(franchiseState).find((row) => String(row.team_id || row.id || "") === id) || null
  );
}

function getNhlRank(teamId, franchiseState) {
  const standings = sortedStandings(franchiseState);
  const idx = standings.findIndex((row) => String(row.team_id || row.id || "") === String(teamId || ""));
  return idx >= 0 ? idx + 1 : null;
}

function ordinal(n) {
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  const mod100 = num % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${num}th`;
  const mod10 = num % 10;
  if (mod10 === 1) return `${num}st`;
  if (mod10 === 2) return `${num}nd`;
  if (mod10 === 3) return `${num}rd`;
  return `${num}th`;
}

function seasonYearLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}ù${Number(y) + 1}` : "This season";
}

function teamShortName(name = "") {
  const parts = String(name).trim().split(/\s+/);
  return parts[parts.length - 1] || String(name);
}

export function getCeremonyRevealStatus(slideIndex, activeIndex, totalSlides) {
  if (slideIndex < activeIndex) return "revealed";
  if (slideIndex === activeIndex) return activeIndex >= totalSlides - 1 ? "final" : "active";
  if (slideIndex === activeIndex + 1) return "up-next";
  return "locked";
}

export function getCeremonyGroupId(awardKey) {
  const group = CEREMONY_RAIL_GROUPS.find((g) => g.awardKeys.includes(awardKey));
  return group?.id || "player";
}

export function buildCeremonyRailGroups(slides) {
  return CEREMONY_RAIL_GROUPS.map((group) => ({
    ...group,
    items: safeArray(slides)
      .map((slide, index) => ({ slide, index }))
      .filter(({ slide }) => group.awardKeys.includes(slide.awardKey)),
  })).filter((group) => group.items.length);
}

function buildHeroBadges(award, entity, franchiseState) {
  const key = award.awardKey;
  const source = { ...entity, ...(award.winner_stats || {}) };
  const badges = [];

  if (key === "presidents" || key === "stanley") {
    const teamId = String(award.winner_team_id || award.winnerTeamId || entity?.team_id || "");
    const rank = getNhlRank(teamId, franchiseState);
    const pts = firstDefined(source.points, source.pts);
    const gd = firstDefined(source.goal_diff, source.goalDiff);
    const wins = firstDefined(source.wins, source.w);
    const division = firstDefined(source.division, source.division_name, source.divisionName);

    if (rank) badges.push({ label: `${ordinal(rank)} NHL`, tone: "gold" });
    if (pts !== undefined) badges.push({ label: `${roundStat(pts)} PTS`, tone: "primary" });
    if (gd !== undefined && gd !== null) {
      const n = Number(gd);
      badges.push({ label: `${n > 0 ? "+" : ""}${n} DIFF`, tone: "accent" });
    }
    if (wins !== undefined) badges.push({ label: `${roundStat(wins)} WINS`, tone: "neutral" });
    if (division) badges.push({ label: `${division} Champion`, tone: "neutral" });
    return badges;
  }

  const points = firstDefined(source.points, source.pts);
  const goals = firstDefined(source.goals, source.g);
  const assists = firstDefined(source.assists, source.a);
  const wins = firstDefined(source.wins, source.w);
  const savePct = firstDefined(source.save_pct, source.savePct, source.sv_pct);
  const plusMinus = firstDefined(source.plus_minus, source.plusMinus);
  const gp = firstDefined(source.games_played, source.gp, source.games);

  if (key === "rocket" && goals !== undefined) badges.push({ label: `${roundStat(goals)} G`, tone: "primary" });
  else if (points !== undefined) badges.push({ label: `${roundStat(points)} PTS`, tone: "primary" });
  if (goals !== undefined && key !== "rocket") badges.push({ label: `${roundStat(goals)} G`, tone: "neutral" });
  if (assists !== undefined) badges.push({ label: `${roundStat(assists)} A`, tone: "neutral" });
  if (wins !== undefined && key === "vezina") badges.push({ label: `${roundStat(wins)} W`, tone: "gold" });
  if (savePct !== undefined && key === "vezina") {
    const n = Number(savePct);
    const sv = Number.isFinite(n) ? (n <= 1 ? (n * 100).toFixed(1) : n.toFixed(1)) : "ù";
    badges.push({ label: `${sv}% SV`, tone: "accent" });
  }
  if (plusMinus !== undefined) {
    const n = Number(plusMinus);
    badges.push({ label: `${n > 0 ? "+" : ""}${roundStat(plusMinus)}`, tone: "accent" });
  }
  if (gp !== undefined) badges.push({ label: `${roundStat(gp)} GP`, tone: "neutral" });
  if (award.winnerTeamName) badges.push({ label: teamShortName(award.winnerTeamName), tone: "neutral" });

  return badges.slice(0, 5);
}

function buildWhyTheyWon(award, entity, franchiseState) {
  const key = award.awardKey;
  const source = { ...entity, ...(award.winner_stats || {}) };
  const bullets = [];

  if (key === "presidents" || key === "stanley") {
    const teamId = String(award.winner_team_id || award.winnerTeamId || entity?.team_id || "");
    const row = findStandingsRow(teamId, franchiseState) || source;
    const rank = getNhlRank(teamId, franchiseState);
    const pts = firstDefined(row.points, row.pts, source.points, source.pts);
    const gd = firstDefined(row.goal_diff, source.goal_diff);
    const record = row.record || compactRecord(row) || source.record;
    const home = firstDefined(row.home_record, row.homeRecord);
    const away = firstDefined(row.road_record, row.away_record, row.awayRecord);
    const l10 = firstDefined(row.last_10, row.last10, row.l10);

    if (rank && pts !== undefined) bullets.push(`${ordinal(rank)} in NHL points (${roundStat(pts)}).`);
    if (record) bullets.push(`Finished ${record} in the regular season.`);
    if (gd !== undefined && gd !== null) {
      const n = Number(gd);
      bullets.push(`Goal differential of ${n > 0 ? "+" : ""}${n}.`);
    }
    if (home) bullets.push(`Home record: ${home}.`);
    if (away) bullets.push(`Road record: ${away}.`);
    if (l10) bullets.push(`Last 10: ${l10}.`);
    if (key === "stanley") bullets.push("Outlasted every playoff opponent to lift the Cup.");
    return bullets.slice(0, 6);
  }

  const points = firstDefined(source.points, source.pts);
  const goals = firstDefined(source.goals, source.g);
  const wins = firstDefined(source.wins, source.w);
  const savePct = firstDefined(source.save_pct, source.savePct);
  const blocks = firstDefined(source.blocks, source.blocked_shots);
  const takeaways = firstDefined(source.takeaways, source.tk);
  const gp = firstDefined(source.games_played, source.gp);

  if (key === "art_ross" && points !== undefined) bullets.push(`League scoring leader with ${roundStat(points)} points.`);
  if (key === "rocket" && goals !== undefined) bullets.push(`Led the NHL with ${roundStat(goals)} goals.`);
  if (key === "hart" && points !== undefined) bullets.push(`Drove team success with ${roundStat(points)} points.`);
  if (key === "norris" && points !== undefined) bullets.push(`${roundStat(points)} points from the blue line.`);
  if (key === "selke") {
    if (takeaways !== undefined) bullets.push(`${roundStat(takeaways)} takeaways in ${gp ? roundStat(gp) : "the"} games.`);
    if (points !== undefined) bullets.push(`${roundStat(points)} points while playing elite defense.`);
  }
  if (key === "calder" && points !== undefined) bullets.push(`Top rookie production: ${roundStat(points)} points.`);
  if (key === "vezina") {
    if (wins !== undefined) bullets.push(`${roundStat(wins)} wins behind the crease.`);
    if (savePct !== undefined) {
      const n = Number(savePct);
      const sv = Number.isFinite(n) ? (n <= 1 ? (n * 100).toFixed(1) : n.toFixed(1)) : null;
      if (sv) bullets.push(`${sv}% save percentage on the season.`);
    }
  }
  if (blocks !== undefined && key === "norris") bullets.push(`${roundStat(blocks)} blocked shots.`);

  const runnerUp = getRunnerUpText(award);
  if (runnerUp && runnerUp !== "the rest of the field") bullets.push(`Finished ahead of ${runnerUp}.`);

  return bullets.slice(0, 6);
}

function buildSeasonSnapshot(award, entity, franchiseState) {
  const key = award.awardKey;
  const source = { ...entity, ...(award.winner_stats || {}) };
  const rows = [];

  if (key === "presidents" || key === "stanley") {
    const teamId = String(award.winner_team_id || award.winnerTeamId || entity?.team_id || "");
    const row = findStandingsRow(teamId, franchiseState) || source;
    const rank = getNhlRank(teamId, franchiseState);
    if (rank) rows.push({ label: "NHL Rank", value: ordinal(rank) });
    rows.push(
      { label: "Points", value: roundStat(firstDefined(row.pts, row.points, source.points)) },
      { label: "Wins", value: roundStat(firstDefined(row.w, row.wins, source.wins)) },
      { label: "Goals For", value: roundStat(firstDefined(row.gf, row.goals_for, source.goals_for)) },
      { label: "Goals Against", value: roundStat(firstDefined(row.ga, row.goals_against, source.goals_against)) }
    );
    const pp = firstDefined(row.pp_pct, row.power_play_pct, source.pp_pct);
    const pk = firstDefined(row.pk_pct, row.penalty_kill_pct, source.pk_pct);
    const sv = firstDefined(row.save_pct, row.team_save_pct, source.save_pct);
    if (pp !== undefined) rows.push({ label: "Power Play", value: percent(pp) });
    if (pk !== undefined) rows.push({ label: "Penalty Kill", value: percent(pk) });
    if (sv !== undefined) rows.push({ label: "Save %", value: percent(sv) });
    return rows.filter((r) => r.value && r.value !== "ù").slice(0, 8);
  }

  rows.push(
    { label: "Points", value: roundStat(firstDefined(source.points, source.pts)) },
    { label: "Goals", value: roundStat(firstDefined(source.goals, source.g)) },
    { label: "Assists", value: roundStat(firstDefined(source.assists, source.a)) },
    { label: "Games", value: roundStat(firstDefined(source.games_played, source.gp)) }
  );
  if (key === "vezina") {
    rows.push(
      { label: "Wins", value: roundStat(firstDefined(source.wins, source.w)) },
      { label: "Save %", value: percent(firstDefined(source.save_pct, source.savePct)) },
      { label: "GAA", value: oneDecimal(firstDefined(source.gaa, source.goals_against_average)) }
    );
  }
  return rows.filter((r) => r.value && r.value !== "ù").slice(0, 8);
}

function resolveTeamNameById(teamId, franchiseState) {
  const id = String(teamId || "").trim();
  if (!id) return "";
  const team = safeArray(franchiseState?.league_teams).find(
    (t) => String(t.id || t.team_id || t.teamId || "") === id
  );
  return team ? getTeamName(team) : "";
}

function buildPreviousWinners(award, franchiseState) {
  const entries = [];
  const currentLabel = seasonYearLabel(franchiseState);
  entries.push({ season: currentLabel, winner: award.winnerLabel, isCurrent: true });

  if (award.awardKey === "stanley") {
    const history = [...safeArray(franchiseState?.season_history)].reverse().slice(0, 3);
    history.forEach((row) => {
      const y = row?.season_year;
      const champion = resolveTeamNameById(row?.champion_id, franchiseState);
      if (y && champion) entries.push({ season: `${y}ù${Number(y) + 1}`, winner: champion });
    });
  }

  const seen = new Set();
  return entries
    .filter((entry) => {
      const key = `${entry.season}:${entry.winner}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return Boolean(entry.winner);
    })
    .slice(0, 4);
}

export function buildVotingDetail(award) {
  const cards = [...safeArray(award?.candidateCards)].sort((a, b) => (a.rank || 99) - (b.rank || 99));
  if (!cards.length) return null;

  const winner = cards.find((c) => c.isWinner) || cards[0];
  const runnerUp = cards.find((c) => c !== winner);
  const winnerVotes = asNumber(winner?.votes, null);
  const runnerVotes = asNumber(runnerUp?.votes, null);

  let headline = "";
  let marginLabel = "";

  if (winnerVotes !== null && runnerVotes !== null) {
    const gap = winnerVotes - runnerVotes;
    if (award.awardKey === "presidents") {
      headline = gap === 0 ? "Tied on points at the top" : gap === 1 ? "Won by 1 point" : `Won by ${gap} points`;
      marginLabel = gap === 0 ? "Even" : gap > 0 ? `+${gap} pt` : `${gap} pt`;
    } else if (award.awardKey === "rocket") {
      headline = gap === 1 ? "Won by 1 goal" : `Won by ${gap} goals`;
      marginLabel = `+${gap} G`;
    } else {
      headline = gap === 1 ? "Won by 1 vote point" : `Won by ${gap} vote points`;
      marginLabel = `+${gap}`;
    }
  } else if (winner?.voteLabel) {
    headline = `${winner.label} led the field`;
    marginLabel = winner.voteLabel;
  }

  return {
    headline,
    marginLabel,
    winnerVotes,
    runnerUpVotes: runnerVotes,
    runnerUpName: runnerUp?.label || "",
    rankings: cards.map((card) => ({
      rank: card.rank,
      label: card.label,
      teamName: card.teamName || "",
      voteLabel: card.voteLabel || card.stat || "",
      votes: card.votes,
      isWinner: card.isWinner,
      teamLogoSrc: card.teamLogoSrc || "",
      player: card.player || null,
    })),
  };
}

function buildAwardRationale(award, entity, franchiseState) {
  const existing = String(award.rationale || award.reason || "").trim();
  if (existing && !/legacy still being written|new chapter enters league history/i.test(existing)) {
    return existing;
  }

  const key = award.awardKey;
  const name = award.winnerLabel || getPlayerName(entity) || getTeamName(entity);
  const short = teamShortName(name);

  if (key === "presidents" || key === "stanley") {
    const teamId = String(award.winner_team_id || award.winnerTeamId || entity?.team_id || "");
    const row = findStandingsRow(teamId, franchiseState) || { ...entity, ...(award.winner_stats || {}) };
    const rank = getNhlRank(teamId, franchiseState);
    const pts = firstDefined(row.pts, row.points, entity?.points);
    const gd = firstDefined(row.goal_diff, entity?.goal_diff);
    const record = row.record || compactRecord(row) || entity?.record;

    if (key === "stanley") {
      return `${short} won the Stanley Cup${record ? ` after a ${record} regular season` : ""}.`;
    }

    if (rank && pts !== undefined && gd !== undefined && gd !== null) {
      const n = Number(gd);
      return `${short} finished ${ordinal(rank)} in the NHL with ${roundStat(pts)} points and a ${n > 0 ? "+" : ""}${n} goal differential.`;
    }
    if (rank && pts !== undefined) {
      return `${short} finished ${ordinal(rank)} in the NHL with ${roundStat(pts)} points.`;
    }
    if (record && pts !== undefined) {
      return `${short} led the league at ${roundStat(pts)} points with a ${record} record.`;
    }
  }

  const points = firstDefined(entity?.points, entity?.pts, award.winner_stats?.points);
  const goals = firstDefined(entity?.goals, entity?.g, award.winner_stats?.goals);
  const wins = firstDefined(entity?.wins, entity?.w, award.winner_stats?.wins);

  const lines = {
    art_ross: points !== undefined ? `${name} led the NHL with ${roundStat(points)} points.` : `${name} finished as the league scoring leader.`,
    rocket: goals !== undefined ? `${name} led the NHL with ${roundStat(goals)} goals.` : `${name} owned the goal column all season.`,
    norris: points !== undefined ? `${name} posted ${roundStat(points)} points from defense to take the Norris.` : `${name} controlled play from the blue line.`,
    hart: points !== undefined ? `${name}'s ${roundStat(points)}-point season made him the league's MVP.` : `${name} delivered the defining individual season.`,
    selke: `${name} separated himself with elite two-way play all season.`,
    calder: points !== undefined ? `${name} led all rookies with ${roundStat(points)} points.` : `${name} was the league's top first-year skater.`,
    vezina: wins !== undefined ? `${name} backstopped ${roundStat(wins)} wins to earn the Vezina.` : `${name} gave his team elite goaltending all year.`,
  };

  return lines[key] || `${name} takes home the ${award.awardLabel || "award"}.`;
}

function buildLegacyLine(award, entity) {
  const source = { ...entity, ...award };
  const age = firstDefined(source.age, source.player_age);
  const draftYear = firstDefined(source.draft_year, source.draftYear);
  const draftPick = firstDefined(source.draft_pick, source.draftPick, source.overall_pick);

  if (age && draftYear && draftPick) return `Age ${age} ù Drafted ${draftYear} (${draftPick})`;
  if (age) return `Age ${age}`;
  return "";
}

function resolveCandidateTeamName(item, franchiseState) {
  const direct = String(item?.team_name || item?.teamName || "").trim();
  if (direct) return direct;

  const teamId = String(item?.team_id || item?.teamId || "").trim();
  if (teamId) {
    const team = safeArray(franchiseState?.league_teams).find(
      (t) => String(t.id || t.team_id || t.teamId || "") === teamId
    );
    if (team) return getTeamName(team);
  }

  return "";
}

function candidateVoteLabel(awardKey, votes) {
  const n = asNumber(votes);
  if (n === null) return "";

  if (awardKey === "presidents" || awardKey === "stanley") {
    return `${n} pts`;
  }

  if (awardKey === "rocket") {
    return `${n} goals`;
  }

  if (["art_ross", "calder", "hart", "norris", "selke"].includes(awardKey)) {
    return `${n} vote pts`;
  }

  return `${n} vote pts`;
}

function buildCandidateCards(award, franchiseState) {
  let source = safeArray(award.candidates).length
    ? safeArray(award.candidates)
    : safeArray(award.finalists);

  if (!source.length && award.awardKey === "presidents") {
    source = safeArray(franchiseState?.standings)
      .slice(0, 5)
      .map((row, index) => ({
        name: row.name || row.team_name,
        team_id: row.team_id || row.id,
        team_name: row.name || row.team_name,
        votes: asNumber(row.pts, 0),
        rank: index + 1,
        is_winner: index === 0,
        points: asNumber(row.pts, 0),
        record: compactRecord({
          wins: row.w,
          losses: row.l,
          ot_losses: row.otl,
          w: row.w,
          l: row.l,
          otl: row.otl,
        }),
      }));
  }

  if (!source.length) return [];

  return source.map((item, index) => {
    if (typeof item === "string") {
      const fakeAward = { ...award, winner_name: item };
      const player = resolveWinnerPlayer(fakeAward, franchiseState);
      const teamName = String(player?.team_name || player?.teamName || "").trim();

      return {
        rank: index + 1,
        name: item,
        player,
        label: item,
        teamName,
        votes: null,
        voteLabel: "",
        stat: teamName || "",
        subline: index === 0 ? "Winner" : "Finalist",
        isWinner: index === 0,
      };
    }

    const isWinner = Boolean(item.is_winner ?? item.isWinner ?? index === 0);
    const rank = asNumber(item.rank, index + 1);
    const name = getPlayerName(item) || getTeamName(item);
    const teamName = resolveCandidateTeamName(item, franchiseState);
    const votes = asNumber(item.votes, null);
    const isTeamAward = award.awardKind === "team";

    let player = null;
    if (!isTeamAward) {
      player = ensurePlayerHeadshotFields(item);
      if (!player?.name) {
        player = resolveWinnerPlayer(
          { ...award, winner_name: name, winner_player_id: item.player_id },
          franchiseState
        );
      }
    }

    const points = firstDefined(item.points, item.pts);
    const goals = firstDefined(item.goals, item.g);

    const statLine = isTeamAward
      ? item.record
        ? `${item.record} ù ${roundStat(firstDefined(item.points, item.pts))} pts`
        : teamName
      : points
        ? `${roundStat(points)} PTS`
        : goals
          ? `${roundStat(goals)} G`
          : teamName;

    return {
      rank,
      name,
      player,
      label: name,
      teamName,
      teamLogoSrc: isTeamAward
        ? getTeamLogoSrc(
            { team_id: item.team_id, full_name: teamName || name, name: teamName || name },
            franchiseState
          )
        : getTeamLogoSrc({ team_id: item.team_id, full_name: teamName }, franchiseState),
      votes,
      voteLabel: candidateVoteLabel(award.awardKey, votes),
      stat: statLine,
      subline: isWinner ? "Winner" : `Finalist ù ${candidateVoteLabel(award.awardKey, votes) || "ù"}`,
      isWinner,
    };
  });
}

function buildFinalistCards(award, franchiseState) {
  return buildCandidateCards(award, franchiseState).slice(0, 3);
}

function getTopStatCard(award) {
  return (
    safeArray(award?.statCards).find((s) => s?.tone === "primary" && s?.value && s.value !== "ù") ||
    safeArray(award?.statCards).find((s) => s?.value && s.value !== "ù") ||
    null
  );
}

function getTopStatText(award) {
  const top = getTopStatCard(award);
  if (!top) return "the numbers";

  return `${top.value}${top.suffix || ""} ${top.label}`.trim();
}

function getRunnerUpText(award) {
  const finalist =
    safeArray(award?.finalistCards).find((f) => !f?.isWinner) ||
    safeArray(award?.candidateCards).find((f) => !f?.isWinner);

  if (!finalist?.label) return "the rest of the field";
  return finalist.voteLabel ? `${finalist.label} with ${finalist.voteLabel}` : finalist.label;
}

function getAwardContextValues(award) {
  return {
    award: award?.awardLabel || "the award",
    awardShort: award?.awardShort || "AWD",
    winner: award?.winnerLabel || "the winner",
    winnerTeam: award?.winnerTeamName || "their team",
    topStat: getTopStatText(award),
    runnerUp: getRunnerUpText(award),
    rationale: award?.rationale || "The case was strong.",
    legacy: award?.legacyLine || "The legacy keeps building.",
    stageLine: award?.stageLine || "Award winner announced.",
  };
}

function getReactionTone(award, seed) {
  const tonesByAward = {
    stanley: ["celebration", "shock", "legacy", "hype"],
    presidents: ["skeptic", "respect", "pressure", "regular-season"],
    hart: ["debate", "legacy", "hype", "argument"],
    vezina: ["goalie-chaos", "respect", "debate", "stolen-games"],
    norris: ["debate", "film-room", "argument", "respect"],
    calder: ["future", "hype", "projection", "hope"],
    selke: ["nerd", "respect", "details", "coach-brain"],
    rocket: ["goal-scorer", "hype", "fear", "pure-offense"],
    art_ross: ["statline", "production", "hype", "points-race"],
  };

  return seededPick(
    tonesByAward[award?.awardKey] || ["reaction", "debate", "hype"],
    `${award?.awardKey}:${award?.winnerLabel}:${seed}:tone`,
    "reaction"
  );
}

function buildFanHandle(first, last, seed) {
  const apiName = `${first || ""}${last || ""}`;
  const word = seededPick(FAN_HANDLE_WORDS, `${apiName}:${seed}:word`, "puckwatch");
  const number = seededInt(`${apiName}:${seed}:number`, 11, 989);
  const compactFirst = compactHandlePart(first).slice(0, 10);
  const compactLast = compactHandlePart(last).slice(0, 10);

  const options = [
    `${word}${number}`,
    `${compactFirst}${word}${seededInt(`${seed}:small`, 1, 99)}`,
    `${compactLast}_${word}`,
    `${word}_${compactFirst}`,
  ].filter((x) => x && x.length > 4);

  return `@${seededPick(options, `${apiName}:${seed}:handle`, `${word}${number}`)}`;
}

function normalizeFanProfile(raw, index = 0, seed = "awards") {
  const first =
    raw?.name?.first ||
    raw?.first ||
    seededPick(FALLBACK_FAN_FIRST_NAMES, `${seed}:${index}:first`, "Rink");

  const last =
    raw?.name?.last ||
    raw?.last ||
    seededPick(FALLBACK_FAN_LAST_NAMES, `${seed}:${index}:last`, "Watcher");

  const displayName = `${first} ${last}`.trim();

  const apiUsername = compactHandlePart(raw?.login?.username || raw?.username);
  const handle = apiUsername
    ? `@${apiUsername}`
    : buildFanHandle(first, last, `${seed}:${index}`);

  return {
    id: raw?.login?.uuid || raw?.id || `fan-${hashString(`${seed}:${index}:${displayName}`)}`,
    displayName,
    handle,
    avatarSrc:
      raw?.picture?.thumbnail ||
      raw?.picture?.medium ||
      raw?.picture?.large ||
      raw?.avatarSrc ||
      "",
    persona: raw?.persona || seededPick(FAN_PERSONAS, `${seed}:${index}:persona`, "fan"),
    market: raw?.market || seededPick(FAN_MARKETS, `${seed}:${index}:market`, "League Feed"),
    verified: Boolean(raw?.verified),
  };
}

export function buildFallbackAwardFans(count = 24, seed = "awards-night") {
  const total = Math.max(1, Number(count) || 24);

  return Array.from({ length: total }, (_, index) => {
    const first = seededPick(FALLBACK_FAN_FIRST_NAMES, `${seed}:${index}:first`, "Rink");
    const last = seededPick(FALLBACK_FAN_LAST_NAMES, `${seed}:${index}:last`, "Watcher");

    return normalizeFanProfile(
      {
        id: `fallback-fan-${index}`,
        first,
        last,
      },
      index,
      seed
    );
  });
}

export async function fetchAwardFanProfiles(options = {}) {
  const {
    count = 24,
    seed = "awards-night",
    nationality = "",
    timeoutMs = 4500,
  } = options || {};

  const total = Math.max(1, Math.min(Number(count) || 24, 100));
  const controller =
    typeof AbortController !== "undefined" ? new AbortController() : null;

  let timeoutId = null;
  if (controller && timeoutMs) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }

  try {
    const params = new URLSearchParams({
      results: String(total),
      seed: String(seed),
      inc: "name,login,picture,nat",
      noinfo: "true",
    });

    if (nationality) params.set("nat", String(nationality));

    const response = await fetch(`${RANDOM_FAN_API_URL}?${params.toString()}`, {
      signal: controller?.signal,
    });

    if (!response.ok) {
      throw new Error(`Random fan API failed with ${response.status}`);
    }

    const data = await response.json();
    const fans = safeArray(data?.results).map((row, index) =>
      normalizeFanProfile(row, index, seed)
    );

    return fans.length ? fans : buildFallbackAwardFans(total, seed);
  } catch (error) {
    return buildFallbackAwardFans(total, seed);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

function buildAwardReactionText(award, fan, index, seed) {
  const templates = [
    ...safeArray(AWARD_FAN_REACTION_TEMPLATES[award?.awardKey]),
    ...safeArray(AWARD_FAN_REACTION_TEMPLATES.generic),
  ];

  const template = seededPick(
    templates,
    `${seed}:${award?.awardKey}:${award?.winnerLabel}:${fan?.handle}:${index}:template`,
    "{winner} wins {award}. The discourse starts now."
  );

  const values = getAwardContextValues(award);
  return trimTweet(replaceTemplate(template, values), 190);
}

function buildTweetMetrics(seed) {
  const likes = seededInt(`${seed}:likes`, 8, 980);
  const reposts = seededInt(`${seed}:reposts`, 0, Math.max(4, Math.floor(likes / 5)));
  const replies = seededInt(`${seed}:replies`, 0, Math.max(3, Math.floor(likes / 8)));
  const quotes = seededInt(`${seed}:quotes`, 0, Math.max(2, Math.floor(likes / 12)));

  return {
    replies,
    reposts,
    quotes,
    likes,
  };
}

export function buildAwardsFanTweets(awards, options = {}) {
  const {
    fans = null,
    maxTweets = 18,
    tweetsPerAward = 2,
    seed = "awards-night",
    includeSummaryTweets = true,
  } = options || {};

  const awardList = safeArray(awards).filter(Boolean);
  if (!awardList.length) {
    return [
      {
        id: "awards-empty-feed",
        type: "award_reaction",
        source: "fan",
        awardKey: "awards",
        awardLabel: "Awards Night",
        createdAtLabel: "now",
        fan: normalizeFanProfile({ first: "Rink", last: "Watcher" }, 0, seed),
        displayName: "Rink Watcher",
        handle: "@puckwatch117",
        avatarSrc: "",
        persona: "league fan",
        market: "League Feed",
        tone: "empty",
        text: "Awards Night is open, but no winners have hit the feed yet.",
        metrics: {
          replies: 0,
          reposts: 0,
          quotes: 0,
          likes: 1,
        },
      },
    ];
  }

  const fanPool = safeArray(fans).length
    ? safeArray(fans).map((fan, index) => normalizeFanProfile(fan, index, seed))
    : buildFallbackAwardFans(Math.max(24, awardList.length * 3), seed);

  const tweets = [];
  const perAward = Math.max(1, Number(tweetsPerAward) || 2);

  if (includeSummaryTweets) {
    const hart = awardList.find((a) => a.awardKey === "hart");
    const stanley = awardList.find((a) => a.awardKey === "stanley");
    const lead = stanley || hart || awardList[0];
    const fan = fanPool[0] || normalizeFanProfile({}, 0, seed);
    const metricSeed = `${seed}:summary:${lead?.awardKey}:${lead?.winnerLabel}`;

    tweets.push({
      id: `tweet-summary-${hashString(metricSeed)}`,
      type: "award_reaction",
      source: "fan",
      awardKey: lead?.awardKey || "awards",
      awardLabel: lead?.awardLabel || "Awards Night",
      createdAtLabel: "now",
      fan,
      displayName: fan.displayName,
      handle: fan.handle,
      avatarSrc: fan.avatarSrc,
      persona: fan.persona,
      market: fan.market,
      tone: "main-feed",
      text: trimTweet(
        lead
          ? `Awards Night already has a main character: ${lead.winnerLabel} and the ${lead.awardLabel}.`
          : "Awards Night is already giving the league something to argue about.",
        180
      ),
      metrics: buildTweetMetrics(metricSeed),
    });
  }

  awardList.forEach((award, awardIndex) => {
    for (let i = 0; i < perAward; i += 1) {
      const fanIndex = (awardIndex * perAward + i + 1) % fanPool.length;
      const fan = fanPool[fanIndex] || fanPool[0] || normalizeFanProfile({}, fanIndex, seed);
      const metricSeed = `${seed}:${award.awardKey}:${award.winnerLabel}:${fan.handle}:${i}`;
      const text = buildAwardReactionText(award, fan, i, seed);

      tweets.push({
        id: `tweet-${hashString(metricSeed)}`,
        type: "award_reaction",
        source: "fan",
        awardKey: award.awardKey,
        awardLabel: award.awardLabel,
        createdAtLabel: seededPick(
          ["now", "12s", "28s", "45s", "1m", "2m", "4m"],
          `${metricSeed}:time`,
          "now"
        ),
        fan,
        displayName: fan.displayName,
        handle: fan.handle,
        avatarSrc: fan.avatarSrc,
        persona: fan.persona,
        market: fan.market,
        tone: getReactionTone(award, metricSeed),
        text,
        metrics: buildTweetMetrics(metricSeed),
        context: {
          winnerLabel: award.winnerLabel,
          winnerTeamName: award.winnerTeamName || "",
          topStat: getTopStatText(award),
          runnerUp: getRunnerUpText(award),
        },
      });
    }
  });

  return tweets
    .filter((tweet) => tweet?.text)
    .slice(0, Math.max(1, Number(maxTweets) || 18));
}

export function buildAwardSocialPulse(awards, options = {}) {
  const tweets = buildAwardsFanTweets(awards, {
    maxTweets: options.maxTweets || 12,
    tweetsPerAward: options.tweetsPerAward || 1,
    seed: options.seed || "awards-night-pulse",
    fans: options.fans,
  });

  const awardList = safeArray(awards);
  const debateAwards = awardList.filter((award) =>
    ["hart", "norris", "vezina", "calder", "selke"].includes(award.awardKey)
  );

  const totalLikes = tweets.reduce((sum, tweet) => sum + Number(tweet?.metrics?.likes || 0), 0);
  const hottestAward = debateAwards[0] || awardList.find((a) => a.awardKey === "stanley") || awardList[0];

  return {
    title: "Fan Pulse",
    headline: hottestAward
      ? `${hottestAward.winnerLabel} has the feed moving`
      : "Awards Night has the feed moving",
    subline: hottestAward
      ? `${hottestAward.awardLabel} is driving the loudest reactions.`
      : "Fans are reacting across the league.",
    tweets,
    totalLikes,
    debateCount: debateAwards.length,
    hottestAwardKey: hottestAward?.awardKey || "",
  };
}

export function normalizeAwardsPayload(franchiseState, eventData) {
  const rawRoot = pickFranchiseData(franchiseState, eventData, [
    "awards",
    "awards_payload",
    "season_summary.awards",
  ]);

  const ceremony = rawRoot?.ceremony || {};
  const backendCatalog = ceremony?.catalog || rawRoot?.ceremony?.catalog || null;
  const revealOrder = Array.isArray(ceremony?.reveal_order) ? ceremony.reveal_order : null;

  const officialResults = safeArray(rawRoot?.official_results);
  const legacyItems = safeArray(rawRoot?.items || rawRoot?.awards || rawRoot).filter(
    (row) => row && typeof row === "object" && (row.name || row.winner_name || row.winnerName || row.award_id)
  );

  const sourceRows = officialResults.length ? officialResults : legacyItems;

  const normalized = sourceRows
    .map((row) => {
      const status = String(row.status || "complete").toLowerCase();
      const awardKey =
        resolveAwardKey(row.award_id || row.awardId || row.name) ||
        String(row.award_id || "").trim();
      const catalogFromBackend = backendCatalog?.[awardKey] || null;
      const meta = {
        ...getAwardCatalogEntry(row.name || row.award_id || awardKey),
        ...(catalogFromBackend
          ? {
              label: catalogFromBackend.name || undefined,
              displayMetric: catalogFromBackend.display_metric || undefined,
            }
          : {}),
      };

      const displayMetric =
        row.display_metric ||
        meta.displayMetric ||
        (meta.kind === "team" ? "PTS" : "Ballot points");

      if (status === "unavailable" || status === "pending") {
        return {
          ...row,
          awardKey: awardKey || resolveAwardKey(row.name),
          awardLabel: meta.label || row.name || "Award",
          awardShort: meta.short || "AWD",
          awardKind: meta.kind || row.recipient_type || "player",
          awardOrder: meta.order || 50,
          ceremonyOrder: meta.ceremonyOrder ?? meta.order ?? 50,
          awardAccent: meta.accent,
          awardGlow: meta.glow,
          trophyTone: meta.trophyTone,
          ceremonyTitle: meta.ceremonyTitle || meta.label,
          stageLine: row.unavailable_reason || row.public_rationale || row.rationale || "Required season data was unavailable.",
          status,
          unavailable: true,
          winnerLabel: "Not awarded",
          rationale: row.unavailable_reason || row.public_rationale || "Required season data was unavailable.",
          finalists: [],
          candidates: [],
          winners: [],
          winnerEntity: null,
          winnerPlayer: null,
          winnerTeam: null,
          statCards: [],
          displayMetric,
          calculationQuality: row.calculation_quality || "unavailable",
          includeInReveal: false,
        };
      }

      const winners = safeArray(row.winners).filter(Boolean);
      const primaryWinner = winners[0] || null;
      const base = {
        ...row,
        awardKey: awardKey || resolveAwardKey(row.name),
        awardLabel: meta.label || row.name,
        awardShort: meta.short,
        awardKind: meta.kind || row.recipient_type || "player",
        awardOrder: meta.order,
        ceremonyOrder: meta.ceremonyOrder ?? meta.order,
        awardAccent: meta.accent,
        awardGlow: meta.glow,
        trophyTone: meta.trophyTone,
        ceremonyTitle: meta.ceremonyTitle,
        stageLine: meta.stageLine,
        winnerLabel: getAwardWinnerLabel(
          {
            ...row,
            winner_name: row.winner_name || primaryWinner?.name,
            winners,
          },
          franchiseState
        ),
        rationale: String(row.public_rationale || row.rationale || row.reason || "").trim(),
        finalists: safeArray(row.finalists).filter(Boolean),
        candidates: safeArray(row.full_results || row.candidates || row.finalists).filter(Boolean),
        winners,
        shared: Boolean(row.shared),
        winner_stats: row.winner_stats || row.winnerStats || null,
        winner_team_name: String(row.winner_team_name || row.winnerTeamName || primaryWinner?.team_name || "").trim(),
        displayMetric,
        calculationQuality: row.calculation_quality || "full",
        voting: row.voting || null,
        includeInReveal: true,
      };

      const parsedStats =
        (meta.kind === "team" || row.recipient_type === "team") && !base.winner_stats
          ? parseTeamStatsFromRationale(base.rationale)
          : null;

      const entity =
        meta.kind === "team" || row.recipient_type === "team"
          ? { ...resolveWinnerTeam({ ...base, winner_stats: base.winner_stats || parsedStats }, franchiseState) }
          : resolveWinnerPlayer(base, franchiseState);

      const statCards =
        meta.kind === "team" || row.recipient_type === "team"
          ? buildTeamStatCards(base, entity)
          : buildPlayerStatCards(base, entity);

      const normalizedAward = {
        ...base,
        winnerEntity: entity,
        winnerPlayer: meta.kind === "player" || meta.kind === "coach" || meta.kind === "goalie" ? entity : null,
        winnerTeam: meta.kind === "team" ? entity : null,
        winnerTeamName:
          base.winner_team_name ||
          (meta.kind !== "team"
            ? resolveCandidateTeamName(
                { team_id: base.winner_team_id, team_name: base.winner_team_name },
                franchiseState
              ) || String(entity?.team_name || entity?.teamName || "").trim()
            : getTeamName(entity)),
        winnerLogoSrc: meta.kind === "team" ? getTeamLogoSrc(entity, franchiseState) : "",
        winnerTeamLogoSrc: getTeamLogoSrc(
          meta.kind === "team" ? entity : resolveWinnerTeam(base, franchiseState),
          franchiseState
        ),
        statCards,
        rationale: buildAwardRationale(base, entity, franchiseState),
        legacyLine: buildLegacyLine(base, entity),
        heroBadges: buildHeroBadges(base, entity, franchiseState),
        whyTheyWon: buildWhyTheyWon(base, entity, franchiseState),
        seasonSnapshot: buildSeasonSnapshot(base, entity, franchiseState),
        previousWinners: buildPreviousWinners(base, franchiseState),
        votingDetail: null,
        candidateCards: buildCandidateCards(base, franchiseState),
        finalistCards: buildFinalistCards(base, franchiseState),
      };

      normalizedAward.votingDetail = buildVotingDetail(normalizedAward);

      return {
        ...normalizedAward,
        socialContext: getAwardContextValues(normalizedAward),
      };
    })
    .filter(Boolean);

  const byKey = new Map(normalized.map((a) => [a.awardKey, a]));
  if (revealOrder?.length) {
    const ordered = [];
    for (const key of revealOrder) {
      const row = byKey.get(key);
      if (row && row.includeInReveal !== false && !row.unavailable) ordered.push(row);
    }
    for (const row of normalized) {
      if (row.unavailable) continue;
      if (!ordered.includes(row) && row.includeInReveal !== false) ordered.push(row);
    }
    // Keep unavailable cards after revealable winners for intentional display.
    for (const row of normalized) {
      if (row.unavailable) ordered.push(row);
    }
    return ordered;
  }

  return normalized.sort(
    (a, b) =>
      Number(Boolean(a.unavailable)) - Number(Boolean(b.unavailable)) ||
      (a.ceremonyOrder ?? a.awardOrder) - (b.ceremonyOrder ?? b.awardOrder) ||
      String(a.awardLabel).localeCompare(String(b.awardLabel))
  );
}

export function buildAwardTickerItems(awards) {
  if (!awards.length) return ["AWARDS NIGHT", "SEASON HARDWARE", "NO WINNERS REPORTED"];

  const chunks = [];

  for (const award of awards) {
    chunks.push(`${award.awardLabel} ù ${award.winnerLabel}`);

    if (award.awardKey === "stanley") {
      chunks.push(`${award.winnerLabel} crowned Stanley Cup champions`);
    }

    if (award.statCards?.length) {
      const top = award.statCards.find((s) => s.tone === "primary") || award.statCards[0];
      if (top?.value && top.value !== "ù") {
        chunks.push(`${award.winnerLabel}: ${top.value}${top.suffix || ""} ${top.label}`);
      }
    }

    if (award.finalistCards?.length) {
      const finalists = award.finalistCards
        .slice(0, 3)
        .map((f) => f.label)
        .filter(Boolean)
        .join(", ");

      if (finalists) {
        chunks.push(`${award.awardShort} finalists: ${finalists}`);
      }
    }

    if (award.candidates?.length) {
      for (const cand of award.candidates.slice(0, 5)) {
        if (typeof cand === "string") continue;
        const cName = getPlayerName(cand) || getTeamName(cand);
        const cTeam = cand.team_name || cand.teamName || "";
        const voteTxt = candidateVoteLabel(award.awardKey, cand.votes);

        if (cName && voteTxt) {
          chunks.push(`${cName}${cTeam ? ` (${cTeam})` : ""}: ${voteTxt}`);
        }
      }
    } else if (award.finalists?.length) {
      chunks.push(
        `Finalists: ${award.finalists
          .slice(0, 3)
          .map((f) => (typeof f === "string" ? f : getPlayerName(f) || getTeamName(f)))
          .join(", ")}`
      );
    }
  }

  return chunks;
}

export function pickFeaturedAward(awards) {
  if (!awards.length) return null;

  return (
    awards.find((a) => a.awardKey === "hart") ||
    awards.find((a) => a.awardKey === "stanley") ||
    awards[0]
  );
}

export function buildAwardsCeremonySlides(awards) {
  const ordered = [...safeArray(awards)]
    .filter((a) => !a?.unavailable && a?.status !== "unavailable" && a?.status !== "pending" && a?.includeInReveal !== false)
    .sort((a, b) => (a.ceremonyOrder ?? a.awardOrder) - (b.ceremonyOrder ?? b.awardOrder));

  return ordered.map((award, index) => ({
    id: `${award.awardKey}-${index}`,
    awardKey: award.awardKey,
    title: award.ceremonyTitle || award.awardLabel,
    awardLabel: award.awardLabel,
    awardShort: award.awardShort,
    winnerLabel: award.winnerLabel,
    winnerTeamName: award.winnerTeamName || "",
    accent: award.awardAccent,
    glow: award.awardGlow,
    trophyTone: award.trophyTone,
    stageLine: award.stageLine,
    rationale: award.rationale,
    legacyLine: award.legacyLine,
    heroBadges: award.heroBadges || [],
    whyTheyWon: award.whyTheyWon || [],
    seasonSnapshot: award.seasonSnapshot || [],
    previousWinners: award.previousWinners || [],
    votingDetail: award.votingDetail || null,
    statCards: award.statCards || [],
    candidateCards: award.candidateCards || [],
    finalistCards: award.finalistCards || [],
    winnerPlayer: award.winnerPlayer || null,
    winnerTeam: award.winnerTeam || null,
    winnerLogoSrc: award.winnerLogoSrc || "",
    winnerTeamLogoSrc: award.winnerTeamLogoSrc || "",
    awardKind: award.awardKind,
    displayMetric: award.displayMetric || "",
    calculationQuality: award.calculationQuality || "full",
    socialContext: award.socialContext || getAwardContextValues(award),
    revealDelayMs: 450 + index * 90,
    cinematicWeight:
      award.awardKey === "stanley"
        ? "championship"
        : award.awardKey === "hart"
          ? "main-event"
          : "standard",
  }));
}

export function buildAwardsNightSummary(awards) {
  const list = safeArray(awards);
  const stanley = list.find((a) => a.awardKey === "stanley");
  const hart = list.find((a) => a.awardKey === "hart");
  const artRoss = list.find((a) => a.awardKey === "art_ross");
  const vezina = list.find((a) => a.awardKey === "vezina");
  const socialPulse = buildAwardSocialPulse(list, {
    maxTweets: 10,
    tweetsPerAward: 1,
    seed: "awards-night-summary",
  });

  return {
    headline: stanley
      ? `${stanley.winnerLabel} finish the job`
      : hart
        ? `${hart.winnerLabel} owns Awards Night`
        : "Season hardware handed out",
    subline: hart
      ? `${hart.winnerLabel} captures the Hart Memorial Trophy.`
      : "The league closes the year with its biggest individual honors.",
    heroAwards: [stanley, hart, artRoss, vezina].filter(Boolean),
    count: list.length,
    socialPulse,
    fanTweets: socialPulse.tweets,
  };
}
/* ============================================================================
 * ENTRY DRAFT ù shared Award Show Twitter Universe extension
 * Reuses seededFloat/seededPick/normalizeFanProfile/buildFallbackAwardFans.
 * ============================================================================ */

export const DRAFT_REACTION_TEMPLATES = [
  { id: "draft_major_steal_01", tags: ["majorSteal"], text: "{playerName} at {pickNumber}? That is serious value for {teamAbbreviation}." },
  { id: "draft_major_steal_02", tags: ["majorSteal"], text: "The public board had {playerName} at {publicRank}. He lasted until {pickNumber}." },
  { id: "draft_major_steal_03", tags: ["majorSteal"], text: "{teamAbbreviation} waited, stayed patient, and landed one of the board's biggest fallers." },
  { id: "draft_major_steal_04", tags: ["majorSteal"], text: "How was {playerName} still there? No complaints from me." },
  { id: "draft_minor_value_01", tags: ["minorValue"], text: "A little value for {teamAbbreviation}. {playerName} went later than expected." },
  { id: "draft_minor_value_02", tags: ["minorValue"], text: "Not a dramatic fall, but {teamAbbreviation} beat the public board here." },
  { id: "draft_minor_value_03", tags: ["minorValue"], text: "Solid spot for {playerName}. The range makes sense, and the value is there." },
  { id: "draft_expected_01", tags: ["expectedPick"], text: "{playerName} lands almost exactly where the public board expected." },
  { id: "draft_expected_02", tags: ["expectedPick", "needFit"], text: "The range fits. The position fits. A straightforward pick for {teamAbbreviation}." },
  { id: "draft_expected_03", tags: ["expectedPick"], text: "No surprise here. {playerName} always felt like the pick." },
  { id: "draft_slight_reach_01", tags: ["slightReach"], text: "A bit earlier than expected, but {teamAbbreviation} clearly wanted {playerName}." },
  { id: "draft_slight_reach_02", tags: ["slightReach"], text: "I had {playerName} later, though the tools make the gamble understandable." },
  { id: "draft_slight_reach_03", tags: ["slightReach"], text: "Maybe early, but I can see what they are betting on." },
  { id: "draft_major_reach_01", tags: ["majorReach"], text: "That is the first major board surprise of the draft." },
  { id: "draft_major_reach_02", tags: ["majorReach"], text: "{playerName} was ranked {publicRank}. {teamAbbreviation} took him at {pickNumber}." },
  { id: "draft_major_reach_03", tags: ["majorReach"], text: "I need someone to explain why {playerName} had to be the pick here." },
  { id: "draft_major_reach_04", tags: ["majorReach"], text: "Our rival just left a lot of talent on the board." },
  { id: "draft_off_board_01", tags: ["offBoard"], text: "We are officially off the public board. {teamAbbreviation} goes its own way." },
  { id: "draft_off_board_02", tags: ["offBoard"], text: "That name was not expected this early. Time to check the notes." },
  { id: "draft_off_board_03", tags: ["offBoard"], text: "I will be honest: I did not have {playerName} on my radar." },
  { id: "draft_need_01", tags: ["needFit"], text: "{teamAbbreviation} needed help at {teamNeed}. {playerName} directly addresses it." },
  { id: "draft_need_02", tags: ["needFit"], text: "Need met. {teamAbbreviation} adds a {position} without forcing the board." },
  { id: "draft_need_03", tags: ["needFit"], text: "Finally. They drafted the position everyone knew they needed." },
  { id: "draft_bpa_01", tags: ["bestAvailable"], text: "Best player left on the public board. Easy value for {teamAbbreviation}." },
  { id: "draft_bpa_02", tags: ["bestAvailable"], text: "Forget position. {playerName} was simply the strongest name remaining." },
  { id: "draft_bpa_03", tags: ["bestAvailable"], text: "They ignored the depth chart and trusted the board. Hard to argue." },
  { id: "draft_pos_surprise_01", tags: ["positionSurprise"], text: "Interesting. {teamAbbreviation} was already deep at {position}." },
  { id: "draft_pos_surprise_02", tags: ["positionSurprise"], text: "The player makes sense. The position is the surprise." },
  { id: "draft_pos_surprise_03", tags: ["positionSurprise"], text: "Another {position}? That was not on my draft-night bingo card." },
  { id: "draft_first_goalie_01", tags: ["firstGoalie"], text: "{playerName} is the first goalie off the board at pick {pickNumber}." },
  { id: "draft_first_goalie_02", tags: ["firstGoalie"], text: "The goalie market is open. {teamAbbreviation} makes the first move." },
  { id: "draft_early_goalie_01", tags: ["earlyGoalie"], text: "A goalie this early will divide the room." },
  { id: "draft_early_goalie_02", tags: ["earlyGoalie"], text: "High-risk position, high-confidence pick. {teamAbbreviation} did not hesitate." },
  { id: "draft_early_goalie_03", tags: ["earlyGoalie"], text: "That is early for a goalie. They better love him." },
  { id: "draft_late_goalie_01", tags: ["lateGoalieValue"], text: "Late-round goalie value for {teamAbbreviation}. That is a worthwhile swing." },
  { id: "draft_late_goalie_02", tags: ["lateGoalieValue", "needFit"], text: "The organization needed another goalie prospect. This is a sensible spot." },
  { id: "draft_wjc_01", tags: ["wjcStandout"], text: "{playerName} turned heads at the WJC. Now he goes {pickNumber}." },
  { id: "draft_wjc_02", tags: ["wjcStandout"], text: "A strong WJC for {country}, and now a major draft-night moment." },
  { id: "draft_wjc_03", tags: ["wjcProducer"], text: "{wjcPoints} WJC points helped push {playerName} into this range." },
  { id: "draft_wjc_04", tags: ["wjcHero"], text: "I watched the whole WJC. This kid can play." },
  { id: "draft_scoring_01", tags: ["juniorScoringLeader"], text: "The {league} scoring leader is headed to {teamName}." },
  { id: "draft_scoring_02", tags: ["juniorScoringLeader"], text: "Production was never the question with {playerName}." },
  { id: "draft_scoring_03", tags: ["juniorScoringLeader"], text: "{teamAbbreviation} adds one of junior hockey's most productive forwards." },
  { id: "draft_playoff_01", tags: ["playoffPerformer"], text: "{playerName} raised his game in the playoffs. Scouts noticed." },
  { id: "draft_playoff_02", tags: ["playoffPerformer"], text: "Strong regular season. Even better playoffs. That matters." },
  { id: "draft_playoff_03", tags: ["playoffPerformer"], text: "Give me the player who showed up when the games got harder." },
  { id: "draft_injury_01", tags: ["injuredProspect"], text: "The talent is clear. The health question shaped where {playerName} landed." },
  { id: "draft_injury_02", tags: ["injuredProspect"], text: "A calculated swing if {playerName} returns to full strength." },
  { id: "draft_injury_03", tags: ["injuredProspect"], text: "I like the player. I am nervous about the missed time." },
  { id: "draft_undersized_01", tags: ["undersized"], text: "The size pushed him down. The puck skill kept him on the board." },
  { id: "draft_undersized_02", tags: ["undersized"], text: "Smaller player, major pace. {playerName} wins with his feet." },
  { id: "draft_undersized_03", tags: ["undersized"], text: "He is small. He is also very hard to catch." },
  { id: "draft_physical_01", tags: ["physicalProspect"], text: "Size, strength, and a direct game. {teamAbbreviation} wanted a heavier prospect." },
  { id: "draft_physical_02", tags: ["physicalProspect"], text: "{teamAbbreviation} adds some needed weight to the prospect pool." },
  { id: "draft_physical_03", tags: ["physicalProspect"], text: "That is a lot of player coming over the boards." },
  { id: "draft_skater_01", tags: ["eliteSkater"], text: "The skating is the selling point. {playerName} creates separation immediately." },
  { id: "draft_skater_02", tags: ["eliteSkater"], text: "One of the fastest players in the class is off the board." },
  { id: "draft_skater_03", tags: ["eliteSkater"], text: "You cannot teach those feet." },
  { id: "draft_skate_concern_01", tags: ["skatingConcern"], text: "The hands work. The skating will decide how far {playerName} goes." },
  { id: "draft_skate_concern_02", tags: ["skatingConcern"], text: "Strong production, but the pace remains the question." },
  { id: "draft_def_concern_01", tags: ["defensiveConcern"], text: "The offence is exciting. The defensive detail needs work." },
  { id: "draft_off_concern_01", tags: ["offensiveConcern"], text: "Reliable player, but where does the scoring come from?" },
  { id: "draft_safe_01", tags: ["safeFloor"], text: "Not flashy, but {playerName} does a lot of NHL things already." },
  { id: "draft_safe_02", tags: ["safeFloor"], text: "This looks like a low-drama path toward an NHL role." },
  { id: "draft_safe_03", tags: ["safeFloor"], text: "Maybe not a star. Feels like a player who will help." },
  { id: "draft_upside_01", tags: ["highUpside"], text: "This is a bet on what {playerName} could become, not what he is today." },
  { id: "draft_upside_02", tags: ["highUpside"], text: "Young, skilled, unfinished. The ceiling drove this selection." },
  { id: "draft_upside_03", tags: ["highUpside"], text: "Swing big. Figure out the details later." },
  { id: "draft_overager_01", tags: ["overager"], text: "Older than most of the class, but much more polished than last year." },
  { id: "draft_overager_02", tags: ["overager"], text: "{playerName} returned, produced, and forced his way onto the board." },
  { id: "draft_young_01", tags: ["youngProspect"], text: "One of the youngest players available. There is plenty of runway here." },
  { id: "draft_young_02", tags: ["youngProspect"], text: "Do not expect him soon. {teamAbbreviation} is thinking several years ahead." },
  { id: "draft_local_01", tags: ["localPlayer"], text: "A local connection for {teamAbbreviation}. {playerName} knows this market well." },
  { id: "draft_local_02", tags: ["localPlayer"], text: "The hometown kid gets the call. That will play well in the building." },
  { id: "draft_cdn_01", tags: ["canadianTeamCanadian"], text: "A Canadian prospect stays north with {teamName}." },
  { id: "draft_cdn_02", tags: ["canadianTeamCanadian"], text: "Canadian kid, Canadian club. Easy story to cheer for." },
  { id: "draft_usa_01", tags: ["americanTeamAmerican"], text: "Another American prospect joins {teamName}." },
  { id: "draft_usa_02", tags: ["americanTeamAmerican"], text: "Homegrown talent for {teamAbbreviation}. I like it." },
  { id: "draft_euro_01", tags: ["europeanProspect"], text: "{playerName} brings a strong {league} track record to {teamName}." },
  { id: "draft_euro_02", tags: ["europeanProspect"], text: "A proud draft-night moment for {country}." },
  { id: "draft_euro_03", tags: ["europeanProspect", "longProject"], text: "The tools are there. The North American adjustment may take time." },
  { id: "draft_rival_01", tags: ["majorSteal"], text: "I hate that our rival got {playerName} that late." },
  { id: "draft_rival_02", tags: ["majorReach"], text: "That pick works for me. They left better players available." },
  { id: "draft_reporter_01", tags: ["expectedPick"], text: "The scouts pushed hard for {playerName}. Management listened." },
  { id: "draft_reporter_02", tags: ["longProject"], text: "This is not an immediate roster play. It is a development investment." },
  { id: "draft_reporter_03", tags: ["nhlReady"], text: "Do not be surprised if {playerName} gets a long camp look soon." },
  { id: "draft_analyst_01", tags: ["expectedPick", "needFit"], text: "One of the cleaner pick-and-player fits of the round." },
  { id: "draft_analyst_02", tags: ["slightReach"], text: "Reasonable player. Debatable spot." },
  { id: "draft_analyst_03", tags: ["majorReach"], text: "A surprise, not an indefensible one." },
  { id: "draft_disagreement_01", tags: ["slightReach", "majorReach"], text: "The room and the public board saw {playerName} very differently." },
  { id: "draft_disagreement_02", tags: ["offBoard"], text: "This is where draft boards stop looking alike." },
  { id: "draft_fan_opt_01", tags: ["minorValue", "majorSteal", "needFit"], text: "I am talking myself into this pick very quickly." },
  { id: "draft_fan_opt_02", tags: ["highUpside"], text: "There is a star outcome here. Let the development staff work." },
  { id: "draft_fan_panic_01", tags: ["majorReach"], text: "Someone tell me there is a plan here." },
  { id: "draft_fan_panic_02", tags: ["majorReach"], text: "We waited all night for that?" },
  { id: "draft_ignore_need_01", tags: ["ignoredNeed"], text: "The need at {teamNeed} remains. {teamAbbreviation} chose value elsewhere." },
  { id: "draft_ignore_need_02", tags: ["ignoredNeed"], text: "Good player. Still no answer at {teamNeed}." },
  { id: "draft_fill_need_01", tags: ["needFit"], text: "The most obvious need is no longer untouched." },
  { id: "draft_run_d_01", tags: ["defenseRun"], text: "Defencemen are moving quickly. That is {recentPositionCount} in this stretch." },
  { id: "draft_run_c_01", tags: ["centreRun"], text: "The centre market is thinning out in a hurry." },
  { id: "draft_run_g_01", tags: ["goalieRun"], text: "Now we have a goalie run. Teams are protecting their targets." },
  { id: "draft_fall_01", tags: ["prospectFall"], text: "{playerName} is still available. That was not expected." },
  { id: "draft_day_two_01", tags: ["dayTwoValue"], text: "First-round talent on day two. That is strong value for {teamAbbreviation}." },
  { id: "draft_seventh_01", tags: ["seventhRound"], text: "Seventh-round tools bet. Nothing wrong with swinging here." },
  { id: "draft_nhl_ready_01", tags: ["nhlReady"], text: "One of the draft's most NHL-ready players is headed to {teamName}." },
  { id: "draft_project_01", tags: ["longProject"], text: "Raw tools, long runway. This selection is about development." },
  { id: "draft_recap_01", tags: ["roundComplete"], text: "Round {round} is complete. Value picks emerged, and the board took a few sharp turns." },
  { id: "draft_first_recap_01", tags: ["firstRoundComplete"], text: "The first round is complete. Day two begins with plenty of talent available." },
  { id: "draft_trade_up_01", tags: ["tradeUp"], text: "{acquiringTeam} moved up to {pickNumber}. They clearly had a target." },
  { id: "draft_trade_down_01", tags: ["tradeDown"], text: "{movingTeam} moves back and adds {tradedAssets}." },
  { id: "draft_lupul_homage", tags: ["lupulHomage"], text: "I want the leafs to keep lupul soley based upon the fact that he banged phangeuds wife" },
];

const CANADIAN_TEAM_IDS = new Set([
  "TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG", "SEA",
]);
const AMERICAN_TEAM_IDS = new Set([
  "BOS", "BUF", "DET", "FLA", "TBL", "CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH",
  "CHI", "DAL", "MIN", "NSH", "STL", "COL", "ARI", "UTA", "ANA", "LAK", "SJS", "VGK",
]);

function draftToken(values, key, fallback = "") {
  const v = values?.[key];
  if (v === undefined || v === null || v === "") return fallback;
  return String(v);
}

function fillDraftTemplate(template, values) {
  return String(template || "").replace(/\{([a-zA-Z0-9_]+)\}/g, (_, key) => draftToken(values, key, ""));
}

export function classifyDraftSelectionEvent(pick, draftContext = {}) {
  const tags = new Set();
  const pickNumber = Number(pick?.overall_pick || pick?.pickNumber || 0);
  const publicRank = Number(pick?.public_rank_at_pick ?? pick?.final_rank ?? pick?.public_rank ?? pick?.rank);
  const hasPublic = Number.isFinite(publicRank) && publicRank > 0 && publicRank < 500;
  // Spec: publicRankDifference = pickNumber - publicRank
  const delta = hasPublic ? pickNumber - publicRank : null;
  const label = String(pick?.selection_label || pick?.pick_classification || "").toLowerCase();
  const position = String(pick?.position || "").toUpperCase();
  const teamAbbr = String(pick?.team_abbreviation || pick?.team_id || "").toUpperCase();
  const country = String(pick?.nationality || pick?.country || pick?.country_code || "");
  const age = Number(pick?.age || 0);
  const wjcGp = Number(pick?.wjc_gp || pick?.wjcStats?.games || 0);
  const wjcPts = Number(pick?.wjc_points || pick?.wjcStats?.points || 0);
  const primaryNeed = String(
    draftContext?.primaryNeed ||
    pick?.board_snapshot?.team_need?.category ||
    (Array.isArray(draftContext?.teamNeeds) && draftContext.teamNeeds[0]?.category) ||
    ""
  );

  // Unranked / beyond tracked board only
  if (!hasPublic || label.includes("off board") || label === "off_board") tags.add("offBoard");
  else if (delta >= 12 || label === "steal") tags.add("majorSteal");
  else if (delta >= 4) tags.add("minorValue");
  else if (delta <= -10 || label === "reach") tags.add("majorReach");
  else if (delta <= -4 || label === "early") tags.add("slightReach");
  else tags.add("expectedPick");

  if (pick?.was_bpa || label === "bpa") tags.add("bestAvailable");
  if (pick?.was_team_need || (primaryNeed && position && primaryNeed.toLowerCase().includes(
    position === "C" ? "center" : position === "G" ? "goalie" : position === "D" ? "defense" : "wing"
  ))) tags.add("needFit");
  else if (primaryNeed && !tags.has("needFit") && tags.has("bestAvailable")) tags.add("ignoredNeed");

  if (position === "G") {
    if (draftContext?.goaliesSelectedBefore === 0) tags.add("firstGoalie");
    if (pickNumber <= 32) tags.add("earlyGoalie");
    if (pickNumber >= 97 && hasPublic && delta >= 8) tags.add("lateGoalieValue");
  }
  if (wjcGp > 0 && wjcPts >= 7) tags.add("wjcHero");
  else if (wjcGp > 0 && wjcPts >= 4) tags.add("wjcStandout");
  else if (wjcGp > 0 && wjcPts >= 2) tags.add("wjcProducer");

  if (Number(pick?.playoff_gp || pick?.playoffStats?.games || 0) > 0 &&
      Number(pick?.playoff_points || pick?.playoffStats?.points || 0) >= 6) {
    tags.add("playoffPerformer");
  }
  if (pick?.injury_flag || pick?.medical_concerns || Number(pick?.games_missed || 0) >= 20) {
    tags.add("injuredProspect");
  }
  const hcm = Number(pick?.height_cm || 0);
  if ((position === "D" && hcm > 0 && hcm < 183) || (["C","LW","RW","W"].includes(position) && hcm > 0 && hcm < 175)) {
    tags.add("undersized");
  }
  if (hcm >= 193 || Number(pick?.weight || 0) >= 215) tags.add("physicalProspect");
  if (String(pick?.developmentProfile || pick?.development_profile || "").toLowerCase().includes("safe")) tags.add("safeFloor");
  if (String(pick?.developmentProfile || pick?.risk || "").toLowerCase().includes("upside") || String(pick?.risk || "") === "High") tags.add("highUpside");
  if (age >= 20) tags.add("overager");
  if (age > 0 && age <= 17) tags.add("youngProspect");
  if (String(pick?.nhl_readiness || pick?.estimatedNhlArrival || "").toLowerCase().match(/now|ready|immediate|1y/)) tags.add("nhlReady");
  if (String(pick?.developmentProfile || "").toLowerCase().includes("raw") || Number(pick?.nhl_eta || 0) >= 4) tags.add("longProject");
  if (pickNumber >= 97 && hasPublic && Number(pick?.final_rank || publicRank) <= 32) tags.add("dayTwoValue");
  if (pickNumber >= 193) tags.add("seventhRound");

  const nat = country.toUpperCase();
  if ((nat.includes("CAN") || nat === "CA") && (CANADIAN_TEAM_IDS.has(teamAbbr) || /toronto|montreal|ottawa|vancouver|calgary|edmonton|winnipeg/.test(String(pick?.team_name || "").toLowerCase()))) {
    tags.add("canadianTeamCanadian");
  }
  if ((nat.includes("USA") || nat === "US") && AMERICAN_TEAM_IDS.has(teamAbbr)) {
    tags.add("americanTeamAmerican");
  }
  if (nat && !nat.includes("CAN") && nat !== "CA" && !nat.includes("USA") && nat !== "US") {
    tags.add("europeanProspect");
  }

  const recent = safeArray(draftContext?.recentPositions);
  if (recent.filter((p) => p === "D").length >= 3) tags.add("defenseRun");
  if (recent.filter((p) => p === "C").length >= 3) tags.add("centreRun");
  if (recent.filter((p) => p === "G").length >= 2) tags.add("goalieRun");

  // 1% homage ù independent of pick context, seeded so it is deterministic per pick.
  const homageRoll = seededFloat(`${draftContext?.franchiseSeed || "draft"}:lupul:${pickNumber}:${pick?.prospect_id || pick?.prospect_name || ""}`);
  if (homageRoll < 0.01) tags.add("lupulHomage");

  return {
    tags: Array.from(tags),
    publicRankDifference: delta,
    selectionLabel: pick?.selection_label || pick?.pick_classification || null,
    valueLabel: tags.has("majorSteal") ? "Steal" : tags.has("minorValue") ? "Value" : tags.has("majorReach") ? "Reach" : tags.has("slightReach") ? "Early" : tags.has("offBoard") ? "Off Board" : "Expected",
  };
}

export function buildDraftFanTweets(picks, options = {}) {
  const {
    fans = null,
    maxTweets = 24,
    seed = "entry-draft",
    draftContext = {},
  } = options || {};

  const pickList = safeArray(picks).filter(Boolean);
  if (!pickList.length) return [];

  const fanPool = safeArray(fans).length
    ? safeArray(fans).map((fan, index) => normalizeFanProfile(fan, index, seed))
    : buildFallbackAwardFans(Math.max(24, pickList.length * 2), seed);

  const tweets = [];
  const usedTemplates = new Set();
  const usedNormalized = new Set();

  pickList.slice(-40).forEach((pick, pickIndex) => {
    const classified = classifyDraftSelectionEvent(pick, {
      ...draftContext,
      recentPositions: safeArray(draftContext.recentPositions).length
        ? draftContext.recentPositions
        : pickList.slice(Math.max(0, pickIndex - 4), pickIndex + 1).map((p) => String(p.position || "").toUpperCase()),
      goaliesSelectedBefore: pickList.slice(0, pickIndex).filter((p) => String(p.position || "").toUpperCase() === "G").length,
    });

    const values = {
      playerName: pick.prospect_name || pick.name || "the prospect",
      firstName: String(pick.prospect_name || pick.name || "").split(" ")[0] || "He",
      lastName: String(pick.prospect_name || pick.name || "").split(" ").slice(-1)[0] || "",
      teamName: pick.team_name || pick.teamName || "the club",
      teamAbbreviation: pick.team_abbreviation || pick.team_id || "NHL",
      pickNumber: pick.overall_pick || pick.pickNumber || "",
      round: pick.round || "",
      publicRank: pick.public_rank_at_pick ?? pick.final_rank ?? pick.rank ?? "unranked",
      publicRankDifference: classified.publicRankDifference ?? "",
      position: pick.position || "",
      country: pick.nationality || pick.country || "",
      juniorTeam: pick.league || pick.club || "",
      league: pick.league || "",
      wjcPoints: pick.wjc_points || pick.wjcStats?.points || "",
      teamNeed: draftContext.primaryNeed || pick?.board_snapshot?.team_need?.category || "depth",
      selectionLabel: classified.selectionLabel || classified.valueLabel || "",
      recentPositionCount: String(safeArray(draftContext.recentPositions).length || 3),
      acquiringTeam: draftContext.acquiringTeam || pick.team_name || "",
      movingTeam: draftContext.movingTeam || "",
      tradedAssets: draftContext.tradedAssets || "future assets",
      developmentTimeline: pick.nhl_eta ? `${pick.nhl_eta}yr` : pick.estimatedNhlArrival || "",
      readinessLabel: pick.nhl_readiness || "",
    };

    // Prefer special tags first, then grade tags
    const priorityTags = [
      "lupulHomage", "majorSteal", "majorReach", "offBoard", "firstGoalie", "wjcHero",
      "tradeUp", "tradeDown", "needFit", "bestAvailable", "dayTwoValue",
      "minorValue", "slightReach", "expectedPick",
    ];
    const active = priorityTags.filter((t) => classified.tags.includes(t));
    const candidates = DRAFT_REACTION_TEMPLATES.filter((t) =>
      t.tags.some((tag) => classified.tags.includes(tag))
      && !usedTemplates.has(t.id)
      && (t.id !== "draft_lupul_homage" || classified.tags.includes("lupulHomage"))
    );
    // Lupul homage always wins when tagged
    let chosen = null;
    if (classified.tags.includes("lupulHomage")) {
      chosen = DRAFT_REACTION_TEMPLATES.find((t) => t.id === "draft_lupul_homage");
    }
    if (!chosen) {
      const pool = candidates.length ? candidates : DRAFT_REACTION_TEMPLATES.filter((t) => t.tags.includes("expectedPick"));
      chosen = seededPick(pool, `${seed}:${values.pickNumber}:${values.playerName}:${active[0] || "x"}`, pool[0]);
    }
    if (!chosen) return;

    let text = trimTweet(fillDraftTemplate(chosen.text, values), 160);
    // Skip templates that still have empty critical tokens
    if (/\{[a-zA-Z]+\}/.test(text) || !text.trim()) return;
    const normalized = text.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ")
      .replace(String(values.playerName || "").toLowerCase(), "{player}")
      .replace(String(values.teamAbbreviation || "").toLowerCase(), "{team}");
    if (usedNormalized.has(normalized) && chosen.id !== "draft_lupul_homage") return;

    usedTemplates.add(chosen.id);
    usedNormalized.add(normalized);

    const fanIndex = (pickIndex + tweets.length) % fanPool.length;
    const fan = fanPool[fanIndex] || normalizeFanProfile({}, fanIndex, seed);
    const metricSeed = `${seed}:draft:${values.pickNumber}:${values.playerName}:${chosen.id}`;

    tweets.push({
      id: `draft-tweet-${hashString(metricSeed)}`,
      type: "draft_reaction",
      source: "fan",
      awardKey: "entry_draft",
      awardLabel: `Pick ${values.pickNumber}`,
      createdAtLabel: seededPick(["now", "8s", "22s", "41s", "1m", "2m"], `${metricSeed}:time`, "now"),
      fan,
      displayName: fan.displayName,
      handle: fan.handle,
      avatarSrc: fan.avatarSrc,
      persona: fan.persona,
      market: fan.market,
      tone: classified.valueLabel === "Steal" ? "homer" : classified.valueLabel === "Reach" ? "skeptic" : "reaction",
      text,
      metrics: buildTweetMetrics(metricSeed),
      context: {
        winnerLabel: values.playerName,
        winnerTeamName: values.teamName,
        topStat: `Pick ${values.pickNumber}`,
        selectionLabel: classified.valueLabel,
        tags: classified.tags,
      },
    });
  });

  return tweets.slice(-maxTweets);
}
