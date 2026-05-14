import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { teamNameToNhlAbbr } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

/**
 * SetupScreen
 * -----------------------------------------------------------------------------
 * Full franchise setup replacement screen inspired by the Halo 3 lobby / setup
 * presentation style shown in the reference image.
 *
 * Goals of this rewrite:
 * - Bigger, more premium first impression for franchise mode
 * - Circular team wheel selector that feels like a console menu
 * - Real NHL team metadata, richer descriptions, and visual identity cues
 * - Fully keyboard navigable without breaking existing GameUIContext flows
 * - Keeps integration surface small:
 *    - uses teams / setupTeamIndex / setSetupTeamIndex
 *    - uses setupGamesPerTeam / setSetupGamesPerTeam
 *    - uses gmName / setGmName
 *    - uses beginFranchise / loading / error
 * - Does not require any additional files
 *
 * Notes:
 * - This file intentionally includes inline styling objects + helper components so
 *   the whole screen can be dropped in without creating new files.
 * - It gracefully falls back if backend team data differs from expected shape.
 * - It supports both select-style control and visual wheel navigation.
 */

const EASTERN_ORDER = [
  "BOS",
  "BUF",
  "DET",
  "FLA",
  "MTL",
  "OTT",
  "TBL",
  "TOR",
  "CAR",
  "CBJ",
  "NJD",
  "NYI",
  "NYR",
  "PHI",
  "PIT",
  "WSH",
];

const WESTERN_ORDER = [
  "UTA",
  "ANA",
  "CGY",
  "CHI",
  "COL",
  "DAL",
  "EDM",
  "LAK",
  "MIN",
  "NSH",
  "SEA",
  "SJS",
  "STL",
  "VAN",
  "VGK",
  "WPG",
];

const DEFAULT_TEAM_ORDER = [...EASTERN_ORDER, ...WESTERN_ORDER];

const TEAM_META = {
  ANA: {
    code: "ANA",
    city: "Anaheim",
    nickname: "Ducks",
    fullName: "Anaheim Ducks",
    conference: "Western",
    division: "Pacific",
    primary: "#F47A38",
    secondary: "#B09862",
    accent: "#CFC493",
    text: "#F3F7FF",
    shortPitch:
      "Rebuild patiently, grow the young core, and turn long-term upside into a playoff-caliber machine.",
    styleTags: ["Youth Movement", "Prospect Pipeline", "Patience Required"],
    arena: "Honda Center",
    market: "Large",
    pressure: "Medium",
    historyTier: "Modern",
    offense: 80,
    defense: 79,
    goaltending: 79,
    overall: 80,
  },
  ARI: {
    code: "ARI",
    city: "Arizona",
    nickname: "Coyotes",
    fullName: "Arizona Coyotes",
    conference: "Western",
    division: "Central",
    primary: "#8C2633",
    secondary: "#E2D6B5",
    accent: "#111111",
    text: "#F6F8FC",
    shortPitch:
      "A resource-management challenge built around development, asset value, and long-term culture building.",
    styleTags: ["Asset Play", "Underdog", "Long-Term Build"],
    arena: "Mullett Arena",
    market: "Small",
    pressure: "Low",
    historyTier: "Modern",
    offense: 77,
    defense: 76,
    goaltending: 77,
    overall: 77,
  },
  UTA: {
    code: "UTA",
    city: "Utah",
    nickname: "Hockey Club",
    fullName: "Utah Hockey Club",
    conference: "Western",
    division: "Central",
    primary: "#6CACE4",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#F4FAFF",
    shortPitch:
      "A fresh market identity with cap flexibility and a mandate to prove the model fast in a league that does not wait.",
    styleTags: ["New Era", "Market Energy", "Build Your Standard"],
    arena: "Delta Center",
    market: "Mid",
    pressure: "High",
    historyTier: "New",
    offense: 83,
    defense: 82,
    goaltending: 82,
    overall: 82,
  },
  BOS: {
    code: "BOS",
    city: "Boston",
    nickname: "Bruins",
    fullName: "Boston Bruins",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#FFB81C",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#F6F8FC",
    shortPitch:
      "Heavy expectations, massive history, and no patience for mediocrity. Compete now or hear about it daily.",
    styleTags: ["Original Six", "Win Now", "Pressure Cooker"],
    arena: "TD Garden",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 88,
    defense: 90,
    goaltending: 88,
    overall: 89,
  },
  BUF: {
    code: "BUF",
    city: "Buffalo",
    nickname: "Sabres",
    fullName: "Buffalo Sabres",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#003087",
    secondary: "#FFB81C",
    accent: "#FFFFFF",
    text: "#F4F8FF",
    shortPitch:
      "Young talent, impatient fans, and a brutal mandate: stop being a maybe and finally become real.",
    styleTags: ["Breakout Candidate", "Youth Core", "Pressure Rising"],
    arena: "KeyBank Center",
    market: "Mid",
    pressure: "High",
    historyTier: "Storied",
    offense: 84,
    defense: 81,
    goaltending: 80,
    overall: 82,
  },
  CGY: {
    code: "CGY",
    city: "Calgary",
    nickname: "Flames",
    fullName: "Calgary Flames",
    conference: "Western",
    division: "Pacific",
    primary: "#C8102E",
    secondary: "#F1BE48",
    accent: "#111111",
    text: "#FFF8FA",
    shortPitch:
      "A volatile roster with a proud market. Balance now-value against the temptation to reshape everything.",
    styleTags: ["Crossroads", "Canadian Market", "Identity Search"],
    arena: "Scotiabank Saddledome",
    market: "Large",
    pressure: "High",
    historyTier: "Strong",
    offense: 82,
    defense: 82,
    goaltending: 81,
    overall: 82,
  },
  CAR: {
    code: "CAR",
    city: "Carolina",
    nickname: "Hurricanes",
    fullName: "Carolina Hurricanes",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#CC0000",
    secondary: "#111111",
    accent: "#A2AAAD",
    text: "#FFF8F8",
    shortPitch:
      "Analytics darling, deep structure, and a perennial contender profile built on relentless systems play.",
    styleTags: ["Contender", "Structure", "Analytics"],
    arena: "Lenovo Center",
    market: "Mid",
    pressure: "High",
    historyTier: "Modern",
    offense: 88,
    defense: 90,
    goaltending: 86,
    overall: 88,
  },
  CBJ: {
    code: "CBJ",
    city: "Columbus",
    nickname: "Blue Jackets",
    fullName: "Columbus Blue Jackets",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#002654",
    secondary: "#CE1126",
    accent: "#A4A9AD",
    text: "#F4F8FF",
    shortPitch:
      "A market still chasing consistency. Build trust, stabilize development, and make the room matter again.",
    styleTags: ["Reset", "Development", "Culture Build"],
    arena: "Nationwide Arena",
    market: "Mid",
    pressure: "Medium",
    historyTier: "Modern",
    offense: 79,
    defense: 78,
    goaltending: 78,
    overall: 79,
  },
  CHI: {
    code: "CHI",
    city: "Chicago",
    nickname: "Blackhawks",
    fullName: "Chicago Blackhawks",
    conference: "Western",
    division: "Central",
    primary: "#CF0A2C",
    secondary: "#111111",
    accent: "#FFD100",
    text: "#FFF7F8",
    shortPitch:
      "Historic brand, big spotlight, and a future-centered build that can accelerate fast if managed correctly.",
    styleTags: ["Original Six", "Superstar Potential", "Rebuild"],
    arena: "United Center",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 80,
    defense: 77,
    goaltending: 78,
    overall: 79,
  },
  COL: {
    code: "COL",
    city: "Colorado",
    nickname: "Avalanche",
    fullName: "Colorado Avalanche",
    conference: "Western",
    division: "Central",
    primary: "#6F263D",
    secondary: "#236192",
    accent: "#A2AAAD",
    text: "#FAF5F8",
    shortPitch:
      "Elite talent, elite expectation. You are not building toward contention — you are already there.",
    styleTags: ["Cup Window", "Star Power", "Fast Attack"],
    arena: "Ball Arena",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 91,
    defense: 88,
    goaltending: 86,
    overall: 89,
  },
  DAL: {
    code: "DAL",
    city: "Dallas",
    nickname: "Stars",
    fullName: "Dallas Stars",
    conference: "Western",
    division: "Central",
    primary: "#006847",
    secondary: "#8F8F8C",
    accent: "#111111",
    text: "#F6FFFB",
    shortPitch:
      "Veteran quality mixed with enough youth to sustain the window. Precision management turns very good into terrifying.",
    styleTags: ["Contender", "Balanced Core", "Deep Lineup"],
    arena: "American Airlines Center",
    market: "Large",
    pressure: "High",
    historyTier: "Strong",
    offense: 89,
    defense: 87,
    goaltending: 86,
    overall: 88,
  },
  DET: {
    code: "DET",
    city: "Detroit",
    nickname: "Red Wings",
    fullName: "Detroit Red Wings",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#CE1126",
    secondary: "#FFFFFF",
    accent: "#A2AAAD",
    text: "#FFF8FA",
    shortPitch:
      "One of hockey's grand brands. The rebuild is judged less by patience and more by visible progress.",
    styleTags: ["Original Six", "Historic Brand", "Rebuild Pressure"],
    arena: "Little Caesars Arena",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 83,
    defense: 82,
    goaltending: 81,
    overall: 82,
  },
  EDM: {
    code: "EDM",
    city: "Edmonton",
    nickname: "Oilers",
    fullName: "Edmonton Oilers",
    conference: "Western",
    division: "Pacific",
    primary: "#041E42",
    secondary: "#FF4C00",
    accent: "#FFFFFF",
    text: "#F2F8FF",
    shortPitch:
      "Generational firepower means one thing: banners or disappointment. Every move is magnified instantly.",
    styleTags: ["Cup Window", "Canadian Pressure", "Top-End Talent"],
    arena: "Rogers Place",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 93,
    defense: 83,
    goaltending: 82,
    overall: 88,
  },
  FLA: {
    code: "FLA",
    city: "Florida",
    nickname: "Panthers",
    fullName: "Florida Panthers",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#041E42",
    secondary: "#C8102E",
    accent: "#B9975B",
    text: "#F6F9FF",
    shortPitch:
      "Mean, deep, and fully dangerous. This is a franchise that expects a heavyweight identity every night.",
    styleTags: ["Contender", "Heavy Hockey", "Deep Core"],
    arena: "Amerant Bank Arena",
    market: "Large",
    pressure: "High",
    historyTier: "Modern",
    offense: 90,
    defense: 88,
    goaltending: 86,
    overall: 89,
  },
  LAK: {
    code: "LAK",
    city: "Los Angeles",
    nickname: "Kings",
    fullName: "Los Angeles Kings",
    conference: "Western",
    division: "Pacific",
    primary: "#111111",
    secondary: "#A2AAAD",
    accent: "#FFFFFF",
    text: "#F7F8FB",
    shortPitch:
      "Big market, strong structure, and enough ambition to chase bigger things than safe playoff appearances.",
    styleTags: ["Structure", "Big Market", "Competitive Window"],
    arena: "Crypto.com Arena",
    market: "Large",
    pressure: "High",
    historyTier: "Strong",
    offense: 85,
    defense: 86,
    goaltending: 84,
    overall: 85,
  },
  MIN: {
    code: "MIN",
    city: "Minnesota",
    nickname: "Wild",
    fullName: "Minnesota Wild",
    conference: "Western",
    division: "Central",
    primary: "#154734",
    secondary: "#A6192E",
    accent: "#EAAA00",
    text: "#F6FFF9",
    shortPitch:
      "A hockey-mad market that demands substance. Win over fans with depth, discipline, and a real plan.",
    styleTags: ["Identity Team", "Depth", "Steady Build"],
    arena: "Xcel Energy Center",
    market: "Mid",
    pressure: "High",
    historyTier: "Strong",
    offense: 84,
    defense: 85,
    goaltending: 84,
    overall: 84,
  },
  MTL: {
    code: "MTL",
    city: "Montreal",
    nickname: "Canadiens",
    fullName: "Montreal Canadiens",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#AF1E2D",
    secondary: "#001E62",
    accent: "#FFFFFF",
    text: "#FFF8FA",
    shortPitch:
      "No market is louder, no history is heavier. Rebuild smart, because every move becomes national conversation.",
    styleTags: ["Original Six", "Historic Pressure", "Rebuild"],
    arena: "Bell Centre",
    market: "Massive",
    pressure: "Very High",
    historyTier: "Legendary",
    offense: 80,
    defense: 80,
    goaltending: 81,
    overall: 80,
  },
  NJD: {
    code: "NJD",
    city: "New Jersey",
    nickname: "Devils",
    fullName: "New Jersey Devils",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#CE1126",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#FFF7F9",
    shortPitch:
      "Fast, skilled, and built to attack. Nail the details and this team becomes a genuine eastern nightmare.",
    styleTags: ["Skill Team", "Speed", "Ascending"],
    arena: "Prudential Center",
    market: "Large",
    pressure: "High",
    historyTier: "Strong",
    offense: 88,
    defense: 83,
    goaltending: 81,
    overall: 84,
  },
  NSH: {
    code: "NSH",
    city: "Nashville",
    nickname: "Predators",
    fullName: "Nashville Predators",
    conference: "Western",
    division: "Central",
    primary: "#FFB81C",
    secondary: "#041E42",
    accent: "#FFFFFF",
    text: "#FFFDF7",
    shortPitch:
      "A loud building, a proud market, and a roster that can tilt either toward aggression or patient reshaping.",
    styleTags: ["Re-Tool", "Market Energy", "Competitive Identity"],
    arena: "Bridgestone Arena",
    market: "Mid",
    pressure: "Medium",
    historyTier: "Strong",
    offense: 82,
    defense: 82,
    goaltending: 81,
    overall: 82,
  },
  NYI: {
    code: "NYI",
    city: "New York",
    nickname: "Islanders",
    fullName: "New York Islanders",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#00539B",
    secondary: "#F47D30",
    accent: "#FFFFFF",
    text: "#F4FAFF",
    shortPitch:
      "A team forever straddling urgency and identity. Decide whether to squeeze the present or reinvent the future.",
    styleTags: ["Crossroads", "Veteran Core", "Decision Time"],
    arena: "UBS Arena",
    market: "Large",
    pressure: "High",
    historyTier: "Strong",
    offense: 81,
    defense: 83,
    goaltending: 83,
    overall: 82,
  },
  NYR: {
    code: "NYR",
    city: "New York",
    nickname: "Rangers",
    fullName: "New York Rangers",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#0038A8",
    secondary: "#CE1126",
    accent: "#FFFFFF",
    text: "#F4F8FF",
    shortPitch:
      "Madison Square Garden means stars, scrutiny, and zero hiding. Make the right moves or live in headlines.",
    styleTags: ["Big Market", "Contender", "Spotlight"],
    arena: "Madison Square Garden",
    market: "Massive",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 89,
    defense: 86,
    goaltending: 88,
    overall: 88,
  },
  OTT: {
    code: "OTT",
    city: "Ottawa",
    nickname: "Senators",
    fullName: "Ottawa Senators",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#C52032",
    secondary: "#C2912C",
    accent: "#111111",
    text: "#FFF8F8",
    shortPitch:
      "Young pieces, sharp expectations, and a market desperate for proof that the climb is finally real.",
    styleTags: ["Young Core", "Pressure Rising", "Canadian Market"],
    arena: "Canadian Tire Centre",
    market: "Mid",
    pressure: "High",
    historyTier: "Strong",
    offense: 84,
    defense: 82,
    goaltending: 80,
    overall: 82,
  },
  PHI: {
    code: "PHI",
    city: "Philadelphia",
    nickname: "Flyers",
    fullName: "Philadelphia Flyers",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#F74902",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#FFF9F5",
    shortPitch:
      "A hard-edged market that hates drifting. Set a real identity fast and make the rebuild feel purposeful.",
    styleTags: ["Culture Reset", "Historic Market", "Rebuild"],
    arena: "Wells Fargo Center",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 80,
    defense: 80,
    goaltending: 79,
    overall: 80,
  },
  PIT: {
    code: "PIT",
    city: "Pittsburgh",
    nickname: "Penguins",
    fullName: "Pittsburgh Penguins",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#FCB514",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#FFFDF7",
    shortPitch:
      "An aging era, a decorated standard, and one giant question: squeeze one more run or start the next chapter?",
    styleTags: ["Legacy Team", "Transition", "High Expectations"],
    arena: "PPG Paints Arena",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 84,
    defense: 81,
    goaltending: 81,
    overall: 82,
  },
  SEA: {
    code: "SEA",
    city: "Seattle",
    nickname: "Kraken",
    fullName: "Seattle Kraken",
    conference: "Western",
    division: "Pacific",
    primary: "#001628",
    secondary: "#99D9D9",
    accent: "#E9072B",
    text: "#F2FBFF",
    shortPitch:
      "A modern franchise with room to define itself. Shape the personality early and own the expansion story.",
    styleTags: ["Expansion Identity", "Modern Brand", "Flexible Path"],
    arena: "Climate Pledge Arena",
    market: "Large",
    pressure: "Medium",
    historyTier: "New",
    offense: 81,
    defense: 81,
    goaltending: 80,
    overall: 81,
  },
  SJS: {
    code: "SJS",
    city: "San Jose",
    nickname: "Sharks",
    fullName: "San Jose Sharks",
    conference: "Western",
    division: "Pacific",
    primary: "#006D75",
    secondary: "#111111",
    accent: "#FFFFFF",
    text: "#F4FFFF",
    shortPitch:
      "One of the cleanest blank slates in hockey. Development, patience, and asset discipline define this run.",
    styleTags: ["Deep Rebuild", "Prospects", "Future Focus"],
    arena: "SAP Center",
    market: "Large",
    pressure: "Medium",
    historyTier: "Strong",
    offense: 75,
    defense: 74,
    goaltending: 74,
    overall: 75,
  },
  STL: {
    code: "STL",
    city: "St. Louis",
    nickname: "Blues",
    fullName: "St. Louis Blues",
    conference: "Western",
    division: "Central",
    primary: "#002F87",
    secondary: "#FCB514",
    accent: "#041E42",
    text: "#F6F9FF",
    shortPitch:
      "A proud, demanding market where competitive mediocrity solves nothing. Pick a lane and commit hard.",
    styleTags: ["Re-Tool", "Tradition", "Decision Point"],
    arena: "Enterprise Center",
    market: "Mid",
    pressure: "High",
    historyTier: "Strong",
    offense: 82,
    defense: 82,
    goaltending: 81,
    overall: 82,
  },
  TBL: {
    code: "TBL",
    city: "Tampa Bay",
    nickname: "Lightning",
    fullName: "Tampa Bay Lightning",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#002868",
    secondary: "#FFFFFF",
    accent: "#A2AAAD",
    text: "#F4F8FF",
    shortPitch:
      "Championship habits still exist here. The challenge is extending a standard most organizations never reach.",
    styleTags: ["Championship Standard", "Veteran Core", "Still Dangerous"],
    arena: "Amalie Arena",
    market: "Large",
    pressure: "Very High",
    historyTier: "Elite",
    offense: 87,
    defense: 85,
    goaltending: 86,
    overall: 86,
  },
  TOR: {
    code: "TOR",
    city: "Toronto",
    nickname: "Maple Leafs",
    fullName: "Toronto Maple Leafs",
    conference: "Eastern",
    division: "Atlantic",
    primary: "#00205B",
    secondary: "#FFFFFF",
    accent: "#A2AAAD",
    text: "#F4F8FF",
    shortPitch:
      "Original Six franchise with a passionate fanbase and a rich history. Push for the Cup and build a lasting legacy.",
    styleTags: ["Original Six", "Massive Market", "Cup Pressure"],
    arena: "Scotiabank Arena",
    market: "Massive",
    pressure: "Maximum",
    historyTier: "Legendary",
    offense: 89,
    defense: 87,
    goaltending: 86,
    overall: 88,
  },
  VAN: {
    code: "VAN",
    city: "Vancouver",
    nickname: "Canucks",
    fullName: "Vancouver Canucks",
    conference: "Western",
    division: "Pacific",
    primary: "#00205B",
    secondary: "#00843D",
    accent: "#FFFFFF",
    text: "#F2F8FF",
    shortPitch:
      "A loud market, volatile expectations, and a roster that can aim high if the details are managed properly.",
    styleTags: ["Canadian Pressure", "Momentum Team", "High Ceiling"],
    arena: "Rogers Arena",
    market: "Large",
    pressure: "Very High",
    historyTier: "Strong",
    offense: 87,
    defense: 84,
    goaltending: 84,
    overall: 85,
  },
  VGK: {
    code: "VGK",
    city: "Vegas",
    nickname: "Golden Knights",
    fullName: "Vegas Golden Knights",
    conference: "Western",
    division: "Pacific",
    primary: "#B4975A",
    secondary: "#333F42",
    accent: "#C8102E",
    text: "#FEFBF4",
    shortPitch:
      "An aggressive organization with zero fear. If there is a big move to make, Vegas usually makes it.",
    styleTags: ["Aggressive", "Contender", "Bold Management"],
    arena: "T-Mobile Arena",
    market: "Large",
    pressure: "High",
    historyTier: "New Power",
    offense: 88,
    defense: 87,
    goaltending: 86,
    overall: 87,
  },
  WPG: {
    code: "WPG",
    city: "Winnipeg",
    nickname: "Jets",
    fullName: "Winnipeg Jets",
    conference: "Western",
    division: "Central",
    primary: "#041E42",
    secondary: "#7B303E",
    accent: "#AC162C",
    text: "#F4F8FF",
    shortPitch:
      "A fiercely loyal market, travel realities, and a team that can win if the structure holds together.",
    styleTags: ["Contender Edge", "Market Passion", "Tough Environment"],
    arena: "Canada Life Centre",
    market: "Small",
    pressure: "High",
    historyTier: "Strong",
    offense: 86,
    defense: 85,
    goaltending: 87,
    overall: 86,
  },
  WSH: {
    code: "WSH",
    city: "Washington",
    nickname: "Capitals",
    fullName: "Washington Capitals",
    conference: "Eastern",
    division: "Metropolitan",
    primary: "#041E42",
    secondary: "#C8102E",
    accent: "#FFFFFF",
    text: "#F5F8FF",
    shortPitch:
      "Legacy-chasing present, succession-planning future. This job is about timing decline without wasting relevance.",
    styleTags: ["Legacy Era", "Transition", "Cap Pressure"],
    arena: "Capital One Arena",
    market: "Large",
    pressure: "High",
    historyTier: "Elite",
    offense: 83,
    defense: 81,
    goaltending: 81,
    overall: 82,
  },
};

const OPTION_GROUPS = {
  seasonLength: [
    { label: "82 Games", value: 82, note: "Full NHL-style season" },
    { label: "62 Games", value: 62, note: "Expanded mid-length season" },
    { label: "42 Games", value: 42, note: "Compact schedule" },
    { label: "15 Games", value: 15, note: "Quick sandbox season" },
  ],
  difficulty: [
    { label: "Rookie", value: "rookie" },
    { label: "Pro", value: "pro" },
    { label: "All-Star", value: "allstar" },
    { label: "Legend", value: "legend" },
  ],
  onOff: [
    { label: "On", value: true },
    { label: "Off", value: false },
  ],
  scouting: [
    { label: "Basic", value: "basic" },
    { label: "Standard", value: "standard" },
    { label: "Detailed", value: "detailed" },
  ],
  ownerGoals: [
    { label: "On", value: true },
    { label: "Off", value: false },
    { label: "Relaxed", value: "relaxed" },
  ],
  startingYear: [
    { label: "2024-25", value: "2024-25" },
    { label: "2025-26", value: "2025-26" },
    { label: "2026-27", value: "2026-27" },
  ],
};

const FIELD_KEYS = [
  "team",
  "gmName",
  "seasonLength",
  "difficulty",
  "salaryCap",
  "tradeDeadline",
  "injuries",
  "prospectScouting",
  "expansionDraft",
  "ownerGoals",
  "startingYear",
  "start",
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeCode(raw) {
  if (!raw && raw !== 0) return "";
  const text = String(raw).trim().toUpperCase();
  if (TEAM_META[text]) return text;
  const aliases = {
    UTA: "UTA",
    UTAH: "UTA",
    YOTES: "ARI",
    COLUMBUS: "CBJ",
    JACKETS: "CBJ",
    BLUEJACKETS: "CBJ",
    TAMPA: "TBL",
    LIGHTNING: "TBL",
    VEGAS: "VGK",
    KNIGHTS: "VGK",
    GOLDENKNIGHTS: "VGK",
    LOSANGELES: "LAK",
    KINGS: "LAK",
    NEWJERSEY: "NJD",
    DEVILS: "NJD",
    ISLANDERS: "NYI",
    RANGERS: "NYR",
    PHILADELPHIA: "PHI",
    PENGUINS: "PIT",
    CAPITALS: "WSH",
    HABS: "MTL",
    LEAFS: "TOR",
    MAPLELEAFS: "TOR",
    CANUCKS: "VAN",
    SHARKS: "SJS",
    KRAKEN: "SEA",
    AVALANCHE: "COL",
    DUCKS: "ANA",
    FLAMES: "CGY",
    OILERS: "EDM",
    PREDATORS: "NSH",
    JETS: "WPG",
    WILD: "MIN",
    STARS: "DAL",
    BLUES: "STL",
    SABRES: "BUF",
    BRUINS: "BOS",
    REDWINGS: "DET",
    REDWINGS: "DET",
    PANTHERS: "FLA",
    SENATORS: "OTT",
    CANADIENS: "MTL",
    HURRICANES: "CAR",
    FLYERS: "PHI",
  };
  return aliases[text] || text.slice(0, 3);
}

function getTeamMetaFromAny(team) {
  if (!team) return TEAM_META.TOR;

  const fromApiName = teamNameToNhlAbbr(team.name || "");
  if (fromApiName && TEAM_META[fromApiName]) return TEAM_META[fromApiName];

  const directCandidates = [
    team.code,
    team.teamCode,
    team.abbreviation,
    team.team_id,
    team.id,
    team.shortName,
  ]
    .map(normalizeCode)
    .filter(Boolean);

  for (const candidate of directCandidates) {
    if (TEAM_META[candidate]) return TEAM_META[candidate];
  }

  const byName = `${team.city || ""} ${team.name || ""} ${team.nickname || ""}`
    .trim()
    .toLowerCase();

  const entry = Object.values(TEAM_META).find((meta) => {
    const haystack = `${meta.city} ${meta.nickname} ${meta.fullName}`.toLowerCase();
    return byName && (haystack.includes(byName) || byName.includes(meta.city.toLowerCase()) || byName.includes(meta.nickname.toLowerCase()));
  });

  return entry || TEAM_META.TOR;
}

function buildOrderedTeams(teams) {
  if (!Array.isArray(teams) || teams.length === 0) return [];

  const enriched = teams.map((team, index) => {
    const meta = getTeamMetaFromAny(team);
    return {
      raw: team,
      index,
      meta,
      code: meta.code,
      name: team.name || meta.fullName,
    };
  });

  const claimed = new Set();
  const ordered = [];
  DEFAULT_TEAM_ORDER.forEach((code) => {
    const match = enriched.find((item) => item.code === code && !claimed.has(item.index));
    if (match) {
      ordered.push(match);
      claimed.add(match.index);
    }
  });

  enriched.forEach((item) => {
    if (!claimed.has(item.index)) {
      ordered.push(item);
      claimed.add(item.index);
    }
  });

  return ordered;
}

function findOrderedIndexFromSetupIndex(orderedTeams, setupIndex) {
  const found = orderedTeams.findIndex((item) => item.index === setupIndex);
  return found >= 0 ? found : 0;
}

function valueLabel(options, currentValue) {
  const match = options.find((option) => option.value === currentValue);
  return match ? match.label : String(currentValue);
}

function percentToBarGradient(percent, colorA, colorB) {
  const clamped = clamp(percent, 0, 100);
  return `linear-gradient(90deg, ${colorA} 0%, ${colorB} ${clamped}%, rgba(255,255,255,0.08) ${clamped}%, rgba(255,255,255,0.04) 100%)`;
}

function useWindowWidth() {
  const [width, setWidth] = useState(() => (typeof window !== "undefined" ? window.innerWidth : 1440));

  useEffect(() => {
    function onResize() {
      setWidth(window.innerWidth);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return width;
}

function TeamGlyph({ code, size = 40, color = "#ffffff", subColor = "rgba(255,255,255,0.2)" }) {
  const common = {
    width: size,
    height: size,
    display: "block",
  };

  const textStyle = {
    fontFamily: "Inter, Arial, sans-serif",
    fontWeight: 900,
    letterSpacing: "0.08em",
  };

  return (
    <svg viewBox="0 0 100 100" style={common} aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id={`g-${code}`} cx="50%" cy="40%" r="70%">
          <stop offset="0%" stopColor={color} stopOpacity="0.24" />
          <stop offset="100%" stopColor={subColor} stopOpacity="0.08" />
        </radialGradient>
      </defs>
      <circle cx="50" cy="50" r="46" fill={`url(#g-${code})`} stroke={color} strokeOpacity="0.35" />
      <circle cx="50" cy="50" r="35" fill="rgba(0,0,0,0.18)" stroke={color} strokeOpacity="0.2" />
      <text x="50" y="57" textAnchor="middle" fill={color} style={{ ...textStyle, fontSize: 24 }}>
        {code}
      </text>
    </svg>
  );
}

function HaloPanel({ children, style, className = "" }) {
  return (
    <div
      className={className}
      style={{
        position: "relative",
        borderRadius: 20,
        border: "1px solid rgba(130,180,255,0.18)",
        background:
          "linear-gradient(180deg, rgba(8,20,45,0.84) 0%, rgba(4,12,29,0.92) 100%)",
        boxShadow:
          "inset 0 0 0 1px rgba(255,255,255,0.03), 0 22px 70px rgba(0,0,0,0.38), 0 0 30px rgba(37,86,180,0.12)",
        overflow: "hidden",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(135deg, rgba(105,165,255,0.08) 0%, transparent 28%, transparent 72%, rgba(105,165,255,0.05) 100%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 1,
          borderRadius: 19,
          border: "1px solid rgba(255,255,255,0.035)",
          pointerEvents: "none",
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
}

function SectionLabel({ children, right }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        marginBottom: 14,
      }}
    >
      <div
        style={{
          color: "#A9C6FF",
          fontSize: 13,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          fontWeight: 800,
        }}
      >
        {children}
      </div>
      {right ? <div>{right}</div> : null}
    </div>
  );
}

function StatBar({ label, value, fillA, fillB }) {
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          color: "#D9E7FF",
          fontSize: 13,
          fontWeight: 700,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        <span>{label}</span>
        <span style={{ color: "#8CB4FF" }}>{value}</span>
      </div>
      <div
        style={{
          height: 10,
          borderRadius: 999,
          border: "1px solid rgba(120,170,255,0.18)",
          background: percentToBarGradient(value, fillA, fillB),
          boxShadow: "inset 0 0 10px rgba(255,255,255,0.05)",
        }}
      />
    </div>
  );
}

function OptionChip({ children, active = false, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        appearance: "none",
        border: active ? "1px solid rgba(143,189,255,0.7)" : "1px solid rgba(118,162,237,0.2)",
        background: active
          ? "linear-gradient(180deg, rgba(80,138,255,0.26) 0%, rgba(35,79,166,0.3) 100%)"
          : "linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)",
        color: active ? "#F4F8FF" : "#A9C6FF",
        borderRadius: 999,
        padding: "9px 14px",
        fontWeight: 700,
        fontSize: 12,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        whiteSpace: "normal",
        lineHeight: 1.15,
        textAlign: "center",
        maxWidth: "100%",
        cursor: "pointer",
        boxShadow: active ? "0 0 18px rgba(86,140,255,0.2)" : "none",
      }}
    >
      {children}
    </button>
  );
}

function SetupFieldRow({
  label,
  value,
  active,
  onClick,
  children,
  hint,
  accent,
  disabled = false,
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      style={{
        width: "100%",
        display: "grid",
        gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 0.9fr)",
        alignItems: "center",
        gap: 14,
        padding: "15px 18px",
        background: active
          ? "linear-gradient(90deg, rgba(85,139,255,0.18) 0%, rgba(33,69,150,0.18) 100%)"
          : "linear-gradient(90deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.01) 100%)",
        border: active
          ? `1px solid ${accent || "rgba(135,185,255,0.48)"}`
          : "1px solid rgba(117,157,228,0.12)",
        borderRadius: 14,
        color: disabled ? "rgba(192,210,243,0.4)" : "#EDF4FF",
        textAlign: "left",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "transform 120ms ease, border-color 120ms ease, background 120ms ease",
        minWidth: 0,
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontSize: 15,
            fontWeight: 800,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {label}
        </div>
        {hint ? (
          <div style={{ marginTop: 5, color: "#7E9FD8", fontSize: 12 }}>{hint}</div>
        ) : null}
      </div>
      <div style={{ minWidth: 0 }}>
        {children || (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-end",
              gap: 10,
              color: "#DCE9FF",
              fontSize: 16,
              fontWeight: 700,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            <span>{value}</span>
            <span style={{ color: "#7EA9FF", fontSize: 18 }}>▾</span>
          </div>
        )}
      </div>
    </button>
  );
}

function TeamWheel({ orderedTeams, selectedOrderedIndex, onSelect, large = false }) {
  const radius = large ? 258 : 226;
  const outerRadius = large ? 322 : 280;
  const center = large ? 360 : 320;
  const size = center * 2;
  const slotCount = Math.max(orderedTeams.length, 1);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        maxWidth: size,
        aspectRatio: "1 / 1",
        margin: "0 auto",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: "7%",
          borderRadius: "50%",
          border: "1px solid rgba(135,181,255,0.18)",
          boxShadow:
            "0 0 0 2px rgba(255,255,255,0.025) inset, 0 0 90px rgba(34,89,196,0.16), inset 0 0 70px rgba(17,62,153,0.14)",
          background:
            "radial-gradient(circle at 50% 50%, rgba(25,77,182,0.18) 0%, rgba(5,12,24,0.12) 38%, rgba(3,7,15,0.75) 100%)",
        }}
      />
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", height: "100%" }}>
        <defs>
          <radialGradient id="wheel-center-glow" cx="50%" cy="45%" r="60%">
            <stop offset="0%" stopColor="rgba(120,180,255,0.42)" />
            <stop offset="35%" stopColor="rgba(86,140,255,0.2)" />
            <stop offset="100%" stopColor="rgba(5,10,20,0.02)" />
          </radialGradient>
          <linearGradient id="wheel-ring-line" x1="0%" x2="100%" y1="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(168,208,255,0.3)" />
            <stop offset="100%" stopColor="rgba(88,142,255,0.08)" />
          </linearGradient>
        </defs>

        <circle cx={center} cy={center} r={outerRadius} fill="none" stroke="url(#wheel-ring-line)" strokeWidth="2" />
        <circle cx={center} cy={center} r={radius + 38} fill="none" stroke="rgba(122,170,255,0.16)" strokeWidth="1.5" />
        <circle cx={center} cy={center} r={radius - 30} fill="none" stroke="rgba(122,170,255,0.12)" strokeWidth="1" />
        <circle cx={center} cy={center} r={radius - 64} fill="url(#wheel-center-glow)" stroke="rgba(132,178,255,0.18)" strokeWidth="2" />

        {orderedTeams.map((team, idx) => {
          const selected = idx === selectedOrderedIndex;
          const angle = (Math.PI * 2 * idx) / slotCount - Math.PI / 2;
          const x = center + Math.cos(angle) * radius;
          const y = center + Math.sin(angle) * radius;
          const dividerX = center + Math.cos(angle) * (radius + 70);
          const dividerY = center + Math.sin(angle) * (radius + 70);

          return (
            <g key={`${team.code}-${team.raw?.team_id ?? team.index}-${idx}`}>
              <line
                x1={center}
                y1={center}
                x2={dividerX}
                y2={dividerY}
                stroke={selected ? "rgba(164,210,255,0.38)" : "rgba(120,155,220,0.1)"}
                strokeWidth={selected ? 2 : 1}
              />
              <circle
                cx={x}
                cy={y}
                r={selected ? 39 : 32}
                fill={selected ? "rgba(79,129,245,0.24)" : "rgba(11,18,38,0.88)"}
                stroke={selected ? team.meta.secondary : "rgba(140,176,235,0.18)"}
                strokeWidth={selected ? 3 : 1.3}
                style={{ cursor: "pointer" }}
                onClick={() => onSelect(idx)}
              />
              <foreignObject x={x - 24} y={y - 24} width={48} height={48} style={{ pointerEvents: "none" }}>
                <div style={{ width: 48, height: 48, display: "grid", placeItems: "center" }}>
                  <TeamGlyph
                    code={team.code}
                    size={selected ? 44 : 36}
                    color={selected ? team.meta.secondary : "#DDE9FF"}
                    subColor={selected ? team.meta.primary : "rgba(255,255,255,0.18)"}
                  />
                </div>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function TeamCenterBadge({ meta }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: "26%",
        borderRadius: "50%",
        display: "grid",
        placeItems: "center",
        textAlign: "center",
        padding: 24,
      }}
    >
      <div
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "50%",
          border: "1px solid rgba(122,170,255,0.16)",
          background:
            `radial-gradient(circle at 50% 35%, ${meta.primary}30 0%, rgba(6,10,18,0.82) 60%, rgba(1,3,7,0.98) 100%)`,
          boxShadow: `0 0 60px ${meta.primary}22, inset 0 0 0 1px rgba(255,255,255,0.03)`,
          display: "grid",
          placeItems: "center",
          padding: 20,
        }}
      >
        <div>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 10 }}>
            <TeamGlyph code={meta.code} size={84} color={meta.secondary || "#fff"} subColor={meta.primary} />
          </div>
          <div
            style={{
              color: "#E6F1FF",
              fontWeight: 900,
              fontSize: 18,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              lineHeight: 1.2,
            }}
          >
            {meta.city}
            <br />
            {meta.nickname}
          </div>
          <div
            style={{
              marginTop: 12,
              color: "#84ACFF",
              fontSize: 14,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            OVR {meta.overall} | OFF {meta.offense} | DEF {meta.defense} | GOA {meta.goaltending}
          </div>
        </div>
      </div>
    </div>
  );
}

function FooterLegend() {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 14,
        color: "#A8C3F5",
        fontSize: 13,
      }}
    >
      <LegendButton color="#F44336" letter="B" label="Back" />
      <LegendButton color="#FFC107" letter="Y" label="Randomize" />
      <LegendButton color="#2196F3" letter="X" label="Reset Name" />
      <LegendButton color="#B0BEC5" letter="≡" label="Save Preset" />
      <LegendButton color="#8BC34A" letter="A" label="Start Franchise" />
    </div>
  );
}

function LegendButton({ color, letter, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          width: 24,
          height: 24,
          borderRadius: "50%",
          display: "grid",
          placeItems: "center",
          background: color,
          color: "#091120",
          fontWeight: 900,
          fontSize: 13,
          boxShadow: `0 0 14px ${color}55`,
        }}
      >
        {letter}
      </div>
      <span>{label}</span>
    </div>
  );
}

function InlineSelect({ options, value, onChange, activeColor }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end" }}>
      <div
        style={{
          display: "flex",
          gap: 6,
          flexWrap: "wrap",
          justifyContent: "flex-end",
        }}
      >
        {options.map((option) => (
          <OptionChip
            key={String(option.value)}
            active={option.value === value}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </OptionChip>
        ))}
      </div>
    </div>
  );
}

export function SetupScreen() {
  const {
    teams,
    setupTeamIndex,
    setSetupTeamIndex,
    setupGamesPerTeam,
    setSetupGamesPerTeam,
    gmName,
    setGmName,
    beginFranchise,
    loading,
    loadTeams,
    error,
  } = useGameUI();

  useEffect(() => {
    loadTeams();
  }, [loadTeams]);

  const orderedTeams = useMemo(() => buildOrderedTeams(teams), [teams]);
  const orderedIndex = useMemo(
    () => findOrderedIndexFromSetupIndex(orderedTeams, setupTeamIndex),
    [orderedTeams, setupTeamIndex]
  );

  const selected = orderedTeams[orderedIndex] || null;
  const selectedMeta = selected?.meta || TEAM_META.TOR;
  const windowWidth = useWindowWidth();
  const compact = windowWidth < 1260;
  const veryCompact = windowWidth < 980;
  const startButtonRef = useRef(null);
  const gmInputRef = useRef(null);
  const [activeField, setActiveField] = useState("team");
  const [difficulty, setDifficulty] = useState("allstar");
  const [salaryCap, setSalaryCap] = useState(true);
  const [tradeDeadline, setTradeDeadline] = useState(true);
  const [injuries, setInjuries] = useState(true);
  const [prospectScouting, setProspectScouting] = useState("detailed");
  const [expansionDraft, setExpansionDraft] = useState(true);
  const [ownerGoals, setOwnerGoals] = useState(true);
  const [startingYear, setStartingYear] = useState("2024-25");
  const [statusText, setStatusText] = useState("Ready to create your franchise.");

  useEffect(() => {
    if (selected?.index !== undefined && selected.index !== setupTeamIndex) {
      setSetupTeamIndex(selected.index);
    }
  }, [orderedIndex, orderedTeams, selected, setSetupTeamIndex, setupTeamIndex]);

  const setTeamByOrderedIndex = useCallback(
    (nextOrderedIndex) => {
      if (!orderedTeams.length) return;
      const safeIndex = ((nextOrderedIndex % orderedTeams.length) + orderedTeams.length) % orderedTeams.length;
      const team = orderedTeams[safeIndex];
      if (!team) return;
      setSetupTeamIndex(team.index);
      setStatusText(`Selected ${team.meta.fullName}.`);
    },
    [orderedTeams, setSetupTeamIndex]
  );

  const cycleFieldValue = useCallback(
    (fieldKey, direction) => {
      const dir = direction >= 0 ? 1 : -1;

      if (fieldKey === "team") {
        setTeamByOrderedIndex(orderedIndex + dir);
        return;
      }

      if (fieldKey === "seasonLength") {
        const options = OPTION_GROUPS.seasonLength;
        const currentIndex = options.findIndex((item) => item.value === setupGamesPerTeam);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + dir + options.length) % options.length;
        setSetupGamesPerTeam(options[nextIndex].value);
        setStatusText(`Season length set to ${options[nextIndex].label}.`);
        return;
      }

      if (fieldKey === "difficulty") {
        const options = OPTION_GROUPS.difficulty;
        const currentIndex = options.findIndex((item) => item.value === difficulty);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + dir + options.length) % options.length;
        setDifficulty(options[nextIndex].value);
        setStatusText(`Difficulty set to ${options[nextIndex].label}.`);
        return;
      }

      if (fieldKey === "salaryCap") {
        setSalaryCap((prev) => !prev);
        setStatusText(`Salary cap ${!salaryCap ? "enabled" : "disabled"}.`);
        return;
      }

      if (fieldKey === "tradeDeadline") {
        setTradeDeadline((prev) => !prev);
        setStatusText(`Trade deadline ${!tradeDeadline ? "enabled" : "disabled"}.`);
        return;
      }

      if (fieldKey === "injuries") {
        setInjuries((prev) => !prev);
        setStatusText(`Injuries ${!injuries ? "enabled" : "disabled"}.`);
        return;
      }

      if (fieldKey === "prospectScouting") {
        const options = OPTION_GROUPS.scouting;
        const currentIndex = options.findIndex((item) => item.value === prospectScouting);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + dir + options.length) % options.length;
        setProspectScouting(options[nextIndex].value);
        setStatusText(`Scouting set to ${options[nextIndex].label}.`);
        return;
      }

      if (fieldKey === "expansionDraft") {
        setExpansionDraft((prev) => !prev);
        setStatusText(`Expansion draft ${!expansionDraft ? "enabled" : "disabled"}.`);
        return;
      }

      if (fieldKey === "ownerGoals") {
        const options = OPTION_GROUPS.ownerGoals;
        const currentIndex = options.findIndex((item) => item.value === ownerGoals);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + dir + options.length) % options.length;
        setOwnerGoals(options[nextIndex].value);
        setStatusText(`Owner goals set to ${options[nextIndex].label}.`);
        return;
      }

      if (fieldKey === "startingYear") {
        const options = OPTION_GROUPS.startingYear;
        const currentIndex = options.findIndex((item) => item.value === startingYear);
        const nextIndex = currentIndex < 0 ? 0 : (currentIndex + dir + options.length) % options.length;
        setStartingYear(options[nextIndex].value);
        setStatusText(`Starting year set to ${options[nextIndex].label}.`);
      }
    },
    [
      difficulty,
      expansionDraft,
      injuries,
      orderedIndex,
      ownerGoals,
      prospectScouting,
      salaryCap,
      setSetupGamesPerTeam,
      setupGamesPerTeam,
      setTeamByOrderedIndex,
      startingYear,
      tradeDeadline,
    ]
  );

  const randomizeSetup = useCallback(() => {
    if (!orderedTeams.length) return;
    const randomTeam = Math.floor(Math.random() * orderedTeams.length);
    const randomNames = [
      "Alex Mercer",
      "Jordan Hayes",
      "Sam Bennett",
      "Morgan Ellis",
      "Connor Blake",
      "Ryan Shepherd",
      "Jamie Carter",
      "Taylor Quinn",
    ];

    setTeamByOrderedIndex(randomTeam);
    setGmName(randomNames[Math.floor(Math.random() * randomNames.length)]);

    const seasonOption = OPTION_GROUPS.seasonLength[Math.floor(Math.random() * OPTION_GROUPS.seasonLength.length)];
    setSetupGamesPerTeam(seasonOption.value);
    setDifficulty(OPTION_GROUPS.difficulty[Math.floor(Math.random() * OPTION_GROUPS.difficulty.length)].value);
    setSalaryCap(Math.random() > 0.15);
    setTradeDeadline(Math.random() > 0.25);
    setInjuries(Math.random() > 0.12);
    setProspectScouting(OPTION_GROUPS.scouting[Math.floor(Math.random() * OPTION_GROUPS.scouting.length)].value);
    setExpansionDraft(Math.random() > 0.5);
    setOwnerGoals(OPTION_GROUPS.ownerGoals[Math.floor(Math.random() * OPTION_GROUPS.ownerGoals.length)].value);
    setStartingYear(OPTION_GROUPS.startingYear[Math.floor(Math.random() * OPTION_GROUPS.startingYear.length)].value);
    setStatusText("Randomized setup configuration.");
  }, [orderedTeams.length, setGmName, setSetupGamesPerTeam, setTeamByOrderedIndex]);

  const resetName = useCallback(() => {
    setGmName("John Anderson");
    setStatusText("General manager name reset to John Anderson.");
  }, [setGmName]);

  const onStart = useCallback(() => {
    setStatusText(`Authorizing franchise control for ${selectedMeta.fullName}.`);
    beginFranchise();
  }, [beginFranchise, selectedMeta.fullName]);

  useEffect(() => {
    function onKeyDown(event) {
      const targetTag = event.target?.tagName;
      const isInputFocused = targetTag === "INPUT" || targetTag === "TEXTAREA";

      if (isInputFocused && activeField === "gmName") {
        if (event.key === "Enter") {
          event.preventDefault();
          startButtonRef.current?.focus?.();
          setActiveField("start");
        }
        return;
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();
        const current = FIELD_KEYS.indexOf(activeField);
        const next = clamp(current - 1, 0, FIELD_KEYS.length - 1);
        setActiveField(FIELD_KEYS[next]);
        if (FIELD_KEYS[next] === "gmName") {
          setTimeout(() => gmInputRef.current?.focus?.(), 0);
        }
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        const current = FIELD_KEYS.indexOf(activeField);
        const next = clamp(current + 1, 0, FIELD_KEYS.length - 1);
        setActiveField(FIELD_KEYS[next]);
        if (FIELD_KEYS[next] === "gmName") {
          setTimeout(() => gmInputRef.current?.focus?.(), 0);
        }
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        cycleFieldValue(activeField, -1);
        return;
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        cycleFieldValue(activeField, 1);
        return;
      }

      if (event.key === "Enter") {
        event.preventDefault();
        if (activeField === "gmName") {
          gmInputRef.current?.focus?.();
          return;
        }
        if (activeField === "start") {
          onStart();
          return;
        }
        if (activeField !== "team") {
          cycleFieldValue(activeField, 1);
        }
        return;
      }

      if (event.key.toLowerCase() === "y") {
        event.preventDefault();
        randomizeSetup();
        return;
      }

      if (event.key.toLowerCase() === "x") {
        event.preventDefault();
        resetName();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeField, cycleFieldValue, onStart, randomizeSetup, resetName]);

  const bodyPadding = veryCompact ? 18 : compact ? 22 : 28;

  return (
    <div
      className="game-screen setup-screen"
      style={{
        minHeight: "100vh",
        color: "#EAF2FF",
        background:
          `radial-gradient(circle at 50% 45%, ${selectedMeta.primary}22 0%, rgba(7,17,38,0.88) 28%, rgba(3,9,21,0.96) 58%, #010611 100%)`,
        position: "relative",
        overflowX: "hidden",
        overflowY: "auto",
      }}
    >
      <BackdropLines accent={selectedMeta.primary} />

      <div style={{ position: "relative", zIndex: 1 }}>
        <GameHeader teamName="FRANCHISE MODE" sectionTitle="SETUP" />

        <div
          style={{
            padding: veryCompact ? "14px 16px 18px" : compact ? "18px 20px 20px" : "22px 28px 24px",
            maxWidth: 1660,
            margin: "0 auto",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: veryCompact
                ? "1fr"
                : compact
                ? "minmax(300px, 0.9fr) minmax(0, 1.1fr)"
                : "minmax(340px, 0.95fr) minmax(0, 1.15fr)",
              gap: veryCompact ? 18 : compact ? 22 : 28,
              alignItems: "start",
            }}
          >
            <HaloPanel style={{ padding: bodyPadding }}>
              <div style={{ display: "grid", gap: 18 }}>
                <div>
                  <div
                    style={{
                      fontSize: veryCompact ? 34 : 48,
                      lineHeight: 0.95,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      fontWeight: 900,
                      color: "#D8E8FF",
                    }}
                  >
                    Setup
                  </div>
                  <div
                    style={{
                      marginTop: 14,
                      color: "#9DBAEF",
                      fontSize: veryCompact ? 15 : 17,
                      lineHeight: 1.55,
                      maxWidth: 420,
                    }}
                  >
                    Create your franchise. Set your team, general manager, and configure the rules that define your save.
                  </div>
                </div>

                <div style={{ display: "grid", gap: 10 }}>
                  <SetupFieldRow
                    label="Pick Your Team"
                    value={selectedMeta.fullName}
                    active={activeField === "team"}
                    onClick={() => setActiveField("team")}
                    accent={`color-mix(in srgb, ${selectedMeta.secondary} 70%, white 30%)`}
                    hint="Cycle with left and right, or click a team in the wheel"
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "flex-end",
                        gap: 12,
                        minWidth: 0,
                      }}
                    >
                      <TeamGlyph code={selectedMeta.code} size={34} color={selectedMeta.secondary} subColor={selectedMeta.primary} />
                      <div
                        style={{
                          color: "#F0F5FF",
                          fontWeight: 800,
                          fontSize: 16,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {selectedMeta.fullName}
                      </div>
                      <div
                        style={{
                          width: 28,
                          height: 28,
                          borderRadius: "50%",
                          border: "1px solid rgba(146,194,255,0.35)",
                          color: "#9CC0FF",
                          display: "grid",
                          placeItems: "center",
                          fontWeight: 900,
                          fontSize: 13,
                        }}
                      >
                        A
                      </div>
                    </div>
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="General Manager Name"
                    active={activeField === "gmName"}
                    onClick={() => {
                      setActiveField("gmName");
                      gmInputRef.current?.focus?.();
                    }}
                    hint="Type your executive name"
                  >
                    <input
                      ref={gmInputRef}
                      value={gmName}
                      onChange={(e) => setGmName(e.target.value)}
                      onFocus={() => setActiveField("gmName")}
                      placeholder="Enter GM name"
                      style={{
                        width: "100%",
                        borderRadius: 12,
                        border: "1px solid rgba(130,176,255,0.2)",
                        background: "rgba(4,10,22,0.82)",
                        color: "#F3F7FF",
                        padding: "12px 14px",
                        fontSize: 16,
                        fontWeight: 700,
                        outline: "none",
                        boxShadow: activeField === "gmName" ? `0 0 0 1px ${selectedMeta.secondary}55` : "none",
                      }}
                    />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Season Length"
                    value={valueLabel(OPTION_GROUPS.seasonLength, setupGamesPerTeam)}
                    active={activeField === "seasonLength"}
                    onClick={() => setActiveField("seasonLength")}
                    hint="Sets regular season duration"
                  >
                    <InlineSelect
                      options={OPTION_GROUPS.seasonLength}
                      value={setupGamesPerTeam}
                      onChange={setSetupGamesPerTeam}
                      activeColor={selectedMeta.primary}
                    />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Difficulty"
                    value={valueLabel(OPTION_GROUPS.difficulty, difficulty)}
                    active={activeField === "difficulty"}
                    onClick={() => setActiveField("difficulty")}
                    hint="Economy and management challenge"
                  >
                    <InlineSelect options={OPTION_GROUPS.difficulty} value={difficulty} onChange={setDifficulty} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Salary Cap"
                    value={salaryCap ? "On" : "Off"}
                    active={activeField === "salaryCap"}
                    onClick={() => setActiveField("salaryCap")}
                    hint="Enable financial realism"
                  >
                    <InlineSelect options={OPTION_GROUPS.onOff} value={salaryCap} onChange={setSalaryCap} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Trade Deadline"
                    value={tradeDeadline ? "On" : "Off"}
                    active={activeField === "tradeDeadline"}
                    onClick={() => setActiveField("tradeDeadline")}
                    hint="Deadline freeze and late-season urgency"
                  >
                    <InlineSelect options={OPTION_GROUPS.onOff} value={tradeDeadline} onChange={setTradeDeadline} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Injuries"
                    value={injuries ? "On" : "Off"}
                    active={activeField === "injuries"}
                    onClick={() => setActiveField("injuries")}
                    hint="Turn roster attrition on or off"
                  >
                    <InlineSelect options={OPTION_GROUPS.onOff} value={injuries} onChange={setInjuries} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Prospect Scouting"
                    value={valueLabel(OPTION_GROUPS.scouting, prospectScouting)}
                    active={activeField === "prospectScouting"}
                    onClick={() => setActiveField("prospectScouting")}
                    hint="Amount of scouting detail in draft season"
                  >
                    <InlineSelect
                      options={OPTION_GROUPS.scouting}
                      value={prospectScouting}
                      onChange={setProspectScouting}
                    />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Expansion Draft"
                    value={expansionDraft ? "On" : "Off"}
                    active={activeField === "expansionDraft"}
                    onClick={() => setActiveField("expansionDraft")}
                    hint="Future league shake-up support"
                  >
                    <InlineSelect options={OPTION_GROUPS.onOff} value={expansionDraft} onChange={setExpansionDraft} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Owner Goals"
                    value={valueLabel(OPTION_GROUPS.ownerGoals, ownerGoals)}
                    active={activeField === "ownerGoals"}
                    onClick={() => setActiveField("ownerGoals")}
                    hint="How strongly ownership drives expectations"
                  >
                    <InlineSelect options={OPTION_GROUPS.ownerGoals} value={ownerGoals} onChange={setOwnerGoals} />
                  </SetupFieldRow>

                  <SetupFieldRow
                    label="Starting Year"
                    value={valueLabel(OPTION_GROUPS.startingYear, startingYear)}
                    active={activeField === "startingYear"}
                    onClick={() => setActiveField("startingYear")}
                    hint="Controls your narrative launch point"
                  >
                    <InlineSelect options={OPTION_GROUPS.startingYear} value={startingYear} onChange={setStartingYear} />
                  </SetupFieldRow>
                </div>

                <div style={{ display: "grid", gap: 12 }}>
                  <div style={{ color: "#8DAFEC", fontSize: 14 }}>{statusText}</div>
                  {error ? (
                    <div
                      style={{
                        padding: "12px 14px",
                        borderRadius: 12,
                        border: "1px solid rgba(255,110,110,0.28)",
                        background: "rgba(80,10,15,0.4)",
                        color: "#FFD9D9",
                        fontWeight: 700,
                      }}
                    >
                      {error}
                    </div>
                  ) : null}
                  <button
                    ref={startButtonRef}
                    type="button"
                    onClick={onStart}
                    onFocus={() => setActiveField("start")}
                    disabled={loading || !teams.length || !orderedTeams.length}
                    style={{
                      appearance: "none",
                      border: `1px solid ${selectedMeta.secondary}88`,
                      background: loading
                        ? "linear-gradient(180deg, rgba(80,90,120,0.45) 0%, rgba(38,44,61,0.8) 100%)"
                        : `linear-gradient(180deg, ${selectedMeta.primary}BB 0%, rgba(12,30,74,0.95) 100%)`,
                      color: "#F7FBFF",
                      borderRadius: 14,
                      padding: veryCompact ? "16px 18px" : "18px 20px",
                      fontSize: veryCompact ? 16 : 18,
                      fontWeight: 900,
                      textTransform: "uppercase",
                      letterSpacing: "0.14em",
                      cursor: loading ? "wait" : "pointer",
                      boxShadow: loading ? "none" : `0 14px 34px ${selectedMeta.primary}35`,
                    }}
                  >
                    {loading ? "Authorizing…" : "Begin Franchise"}
                  </button>
                </div>
              </div>
            </HaloPanel>

            <div style={{ display: "grid", gap: veryCompact ? 18 : 22 }}>
              <HaloPanel style={{ padding: bodyPadding, minHeight: compact ? undefined : 820 }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: veryCompact ? "1fr" : "1fr",
                    gap: 18,
                  }}
                >
                  <div style={{ position: "relative", minHeight: veryCompact ? 520 : compact ? 640 : 760 }}>
                    <SectionLabel
                      right={
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                          {selectedMeta.styleTags.map((tag) => (
                            <span
                              key={tag}
                              style={{
                                padding: "7px 10px",
                                borderRadius: 999,
                                border: "1px solid rgba(130,173,242,0.18)",
                                color: "#B9D2FF",
                                fontSize: 12,
                                fontWeight: 700,
                                background: "rgba(255,255,255,0.03)",
                              }}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      }
                    >
                      Team Selection Matrix
                    </SectionLabel>

                    <TeamWheel
                      orderedTeams={orderedTeams}
                      selectedOrderedIndex={orderedIndex}
                      onSelect={setTeamByOrderedIndex}
                      large={!compact}
                    />
                    <TeamCenterBadge meta={selectedMeta} />
                  </div>

                  <HaloPanel
                    style={{
                      padding: veryCompact ? 18 : 22,
                      marginTop: compact ? 0 : -10,
                      background:
                        "linear-gradient(180deg, rgba(7,17,40,0.88) 0%, rgba(3,9,21,0.96) 100%)",
                    }}
                  >
                    <div style={{ display: "grid", gap: 18 }}>
                      <div>
                        <div
                          style={{
                            color: "#D9EBFF",
                            fontSize: veryCompact ? 24 : 32,
                            fontWeight: 900,
                            textTransform: "uppercase",
                            letterSpacing: "0.08em",
                            lineHeight: 1.05,
                          }}
                        >
                          {selectedMeta.fullName}
                        </div>
                        <div style={{ marginTop: 10, color: "#8FB2EC", fontSize: 16, lineHeight: 1.7 }}>
                          {selectedMeta.shortPitch}
                        </div>
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: veryCompact ? "1fr 1fr" : "repeat(4, minmax(0, 1fr))",
                          gap: 12,
                        }}
                      >
                        <DataCard title="Conference" value={selectedMeta.conference} accent={selectedMeta.primary} />
                        <DataCard title="Division" value={selectedMeta.division} accent={selectedMeta.primary} />
                        <DataCard title="Market" value={selectedMeta.market} accent={selectedMeta.primary} />
                        <DataCard title="Pressure" value={selectedMeta.pressure} accent={selectedMeta.primary} />
                        <DataCard title="Arena" value={selectedMeta.arena} accent={selectedMeta.primary} />
                        <DataCard title="History" value={selectedMeta.historyTier} accent={selectedMeta.primary} />
                        <DataCard title="Mode" value={valueLabel(OPTION_GROUPS.difficulty, difficulty)} accent={selectedMeta.primary} />
                        <DataCard title="Year" value={startingYear} accent={selectedMeta.primary} />
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: compact ? "1fr" : "1fr 1fr",
                          gap: 18,
                        }}
                      >
                        <HaloPanel style={{ padding: 18 }}>
                          <SectionLabel>Team Ratings</SectionLabel>
                          <div style={{ display: "grid", gap: 14 }}>
                            <StatBar label="Overall" value={selectedMeta.overall} fillA={selectedMeta.primary} fillB={selectedMeta.secondary} />
                            <StatBar label="Offense" value={selectedMeta.offense} fillA={selectedMeta.primary} fillB={selectedMeta.secondary} />
                            <StatBar label="Defense" value={selectedMeta.defense} fillA={selectedMeta.primary} fillB={selectedMeta.secondary} />
                            <StatBar label="Goaltending" value={selectedMeta.goaltending} fillA={selectedMeta.primary} fillB={selectedMeta.secondary} />
                          </div>
                        </HaloPanel>

                        <HaloPanel style={{ padding: 18 }}>
                          <SectionLabel>Save Summary</SectionLabel>
                          <div style={{ display: "grid", gap: 12 }}>
                            <SummaryLine label="General Manager" value={gmName || "Unnamed Executive"} />
                            <SummaryLine label="Season Format" value={valueLabel(OPTION_GROUPS.seasonLength, setupGamesPerTeam)} />
                            <SummaryLine label="Difficulty" value={valueLabel(OPTION_GROUPS.difficulty, difficulty)} />
                            <SummaryLine label="Salary Cap" value={salaryCap ? "Enabled" : "Disabled"} />
                            <SummaryLine label="Trade Deadline" value={tradeDeadline ? "Enabled" : "Disabled"} />
                            <SummaryLine label="Injuries" value={injuries ? "Enabled" : "Disabled"} />
                            <SummaryLine label="Scouting" value={valueLabel(OPTION_GROUPS.scouting, prospectScouting)} />
                            <SummaryLine label="Expansion Draft" value={expansionDraft ? "Enabled" : "Disabled"} />
                            <SummaryLine label="Owner Goals" value={valueLabel(OPTION_GROUPS.ownerGoals, ownerGoals)} />
                          </div>
                        </HaloPanel>
                      </div>
                    </div>
                  </HaloPanel>
                </div>
              </HaloPanel>

              <HaloPanel style={{ padding: bodyPadding }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: veryCompact ? "1fr" : "1.4fr 1fr",
                    gap: 16,
                    alignItems: "center",
                  }}
                >
                  <div>
                    <div
                      style={{
                        color: "#DCEBFF",
                        fontSize: veryCompact ? 17 : 20,
                        fontWeight: 800,
                        letterSpacing: "0.05em",
                        textTransform: "uppercase",
                      }}
                    >
                      Ready Room Controls
                    </div>
                    <div style={{ marginTop: 8, color: "#95B2E8", fontSize: 14, lineHeight: 1.65 }}>
                      Keyboard support: ↑ and ↓ move between fields, ← and → change values, Enter activates the highlighted control. Press Y to randomize and X to reset the GM name.
                    </div>
                  </div>
                  <div style={{ justifySelf: veryCompact ? "start" : "end" }}>
                    <FooterLegend />
                  </div>
                </div>
              </HaloPanel>
            </div>
          </div>
        </div>

        <GameFooter />
      </div>
    </div>
  );
}

function DataCard({ title, value, accent }) {
  return (
    <div
      style={{
        borderRadius: 14,
        border: "1px solid rgba(127,171,238,0.16)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.015) 100%)",
        padding: "14px 15px",
        minHeight: 86,
        boxShadow: `inset 0 0 0 1px rgba(255,255,255,0.02), 0 0 18px ${accent}12`,
      }}
    >
      <div
        style={{
          color: "#8FADE0",
          fontSize: 12,
          fontWeight: 800,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
        }}
      >
        {title}
      </div>
      <div
        style={{
          marginTop: 8,
          color: "#EFF6FF",
          fontSize: 18,
          lineHeight: 1.3,
          fontWeight: 800,
        }}
      >
        {value}
      </div>
    </div>
  );
}

function SummaryLine({ label, value }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        paddingBottom: 9,
        borderBottom: "1px solid rgba(122,170,255,0.09)",
      }}
    >
      <span
        style={{
          color: "#8FADE0",
          fontSize: 12,
          fontWeight: 800,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
      <span style={{ color: "#EFF6FF", fontSize: 14, fontWeight: 700, textAlign: "right" }}>{value}</span>
    </div>
  );
}

function BackdropLines({ accent }) {
  return (
    <>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(180deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
          backgroundSize: "120px 120px",
          opacity: 0.14,
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            `radial-gradient(circle at 20% 14%, ${accent}22 0%, transparent 24%), radial-gradient(circle at 85% 12%, rgba(90,145,255,0.18) 0%, transparent 18%), radial-gradient(circle at 76% 84%, rgba(90,145,255,0.12) 0%, transparent 20%), radial-gradient(circle at 12% 78%, rgba(90,145,255,0.08) 0%, transparent 16%)`,
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          opacity: 0.15,
          background:
            "linear-gradient(135deg, transparent 0%, rgba(151,197,255,0.12) 8%, transparent 16%, transparent 100%)",
          transform: "translateX(-12%)",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          boxShadow: "inset 0 0 180px rgba(0,0,0,0.62)",
          pointerEvents: "none",
        }}
      />
    </>
  );
}

export default SetupScreen;
