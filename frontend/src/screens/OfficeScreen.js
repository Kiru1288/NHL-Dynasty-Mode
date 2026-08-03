import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { HUB_MENU, SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";

/**
 * OFFICE CREATION / FRANCHISE INFRASTRUCTURE SCREEN
 * -------------------------------------------------
 * Full replacement screen:
 * - keeps the project's existing glass / ESPN-EA shell language
 * - expands office progression into a prestige-driven hub
 * - keeps everything in ONE file for easy copy-paste use
 * - intentionally avoids creating new files
 *
 * Notes:
 * - Designed to feel native to the existing CSS already in the project.
 * - Uses many existing class names from your current stylesheet so visual language stays consistent.
 * - A few additional semantic class names are included in markup for future CSS refinement, but the layout
 *   still works with the current stylesheet.
 */

/* -------------------------------------------------------------------------- */
/*                                CONFIG DATA                                 */
/* -------------------------------------------------------------------------- */

const OFFICE_TIERS = [
  {
    id: 1,
    key: "small_outpost",
    shortLabel: "Tier 1",
    label: "Small Outpost",
    theme: "Groundwork",
    prestigeMin: 0,
    prestigeMax: 19,
    prestigeNeeded: 0,
    unlockText: "Default starting office",
    mood: "Survival",
    description:
      "A cramped, low-budget front office with dim lights, old furniture, and the feeling of a franchise just trying to survive.",
    atmosphere:
      "Concrete walls. Worn flooring. One tired lamp. Filing cabinets that have seen too many failed rebuilds.",
    perkText:
      "Baseline command center. No prestige bonus. The grind starts here.",
    visualBullets: [
      "Cramped room",
      "Concrete or worn walls",
      "Metal desk",
      "Old filing cabinets",
      "Dim lighting",
      "Papers scattered",
      "Bulletin board",
    ],
    functionalSummary:
      "A gritty underdog starting point that visually reinforces early-franchise struggle.",
    accent: "orange",
  },
  {
    id: 2,
    key: "functional_hq",
    shortLabel: "Tier 2",
    label: "Functional HQ",
    theme: "Stability",
    prestigeMin: 20,
    prestigeMax: 39,
    prestigeNeeded: 20,
    unlockText: "Unlock at 20 Prestige",
    mood: "Structure",
    description:
      "A cleaner, more organized workspace with better lighting and a little real belief around the building.",
    atmosphere:
      "Basic monitors, improved layout, less clutter, better floor finish, cleaner surfaces.",
    perkText:
      "Signals competence. Front office starts to feel organized rather than improvised.",
    visualBullets: [
      "Cleaner walls",
      "Better lighting",
      "Basic monitors",
      "Organized desk",
      "Fewer loose papers",
      "Improved seating",
      "Visible team branding",
    ],
    functionalSummary:
      "Your team begins looking professional enough that players, scouts, and staff notice.",
    accent: "neon",
  },
  {
    id: 3,
    key: "established_hq",
    shortLabel: "Tier 3",
    label: "Established HQ",
    theme: "Contender",
    prestigeMin: 40,
    prestigeMax: 59,
    prestigeNeeded: 40,
    unlockText: "Unlock at 40 Prestige",
    mood: "Respect",
    description:
      "A confident headquarters with visible staff activity, upgraded tech, and a room that looks like serious hockey decisions are made here.",
    atmosphere:
      "Glass accents, multi-display command desk, team marks, visible staff movement, cleaner architecture.",
    perkText:
      "You look legit now. The room finally reflects a franchise with direction.",
    visualBullets: [
      "Glass elements",
      "Staff visible",
      "Multiple screens",
      "Branding visible",
      "Cleaner lines",
      "Sharper lighting",
      "Operational confidence",
    ],
    functionalSummary:
      "This is where rebuild turns into respectability and organizational identity begins to stick.",
    accent: "purple",
  },
  {
    id: 4,
    key: "premier_hq",
    shortLabel: "Tier 4",
    label: "Premier HQ",
    theme: "Powerhouse",
    prestigeMin: 60,
    prestigeMax: 79,
    prestigeNeeded: 60,
    unlockText: "Unlock at 60 Prestige",
    mood: "Dominance",
    description:
      "Luxury surfaces, premium furnishings, interactive displays, and a skyline-worthy atmosphere built for a perennial contender.",
    atmosphere:
      "LED lighting, premium materials, skyline glass, interactive planning stations, executive presentation.",
    perkText:
      "This office screams control, confidence, and expectation.",
    visualBullets: [
      "Skyline view",
      "Luxury furniture",
      "LED lighting",
      "Interactive displays",
      "Cleaner architecture",
      "High-end finish",
      "Executive energy",
    ],
    functionalSummary:
      "The franchise is no longer trying to prove it belongs. It already does.",
    accent: "silver",
  },
  {
    id: 5,
    key: "legacy_hq",
    shortLabel: "Tier 5",
    label: "Legacy Headquarters",
    theme: "Legacy",
    prestigeMin: 80,
    prestigeMax: 100,
    prestigeNeeded: 80,
    unlockText: "Unlock at 80 Prestige",
    mood: "Dynasty",
    description:
      "A massive, cinematic headquarters with trophy displays, franchise history walls, and the aura of a team that expects banners.",
    atmosphere:
      "Hall of fame wall. Trophy cases. Signature lighting. Monumental scale. Dynasty energy.",
    perkText:
      "The office itself becomes a statement: this franchise matters.",
    visualBullets: [
      "Trophy cases",
      "Hall of Fame walls",
      "Massive office",
      "Cinematic lighting",
      "Premium displays",
      "Historic memorabilia",
      "Dynasty presentation",
    ],
    functionalSummary:
      "The final evolution. No longer a workspace — a monument to the franchise.",
    accent: "gold",
  },
];

const PRESTIGE_SOURCES = [
  {
    key: "wins",
    label: "Winning Games",
    weight: 1.6,
    description: "Every win pushes the franchise closer to credibility and influence.",
  },
  {
    key: "playoffPush",
    label: "Playoff Success",
    weight: 10,
    description: "Postseason appearances dramatically raise organizational status.",
  },
  {
    key: "cupLegacy",
    label: "Championships",
    weight: 22,
    description: "Nothing changes your reputation faster than banners.",
  },
  {
    key: "playerGrowth",
    label: "Player Development",
    weight: 6,
    description: "Turning prospects into core pieces builds prestige behind the scenes.",
  },
  {
    key: "smartMoves",
    label: "Smart Trades / Asset Management",
    weight: 5,
    description: "A sharp front office earns league-wide respect.",
  },
  {
    key: "fanEnergy",
    label: "Fan Engagement",
    weight: 4,
    description: "Buzz matters. Relevance matters. Momentum matters.",
  },
];

const CUSTOMIZATION_GROUPS = [
  {
    key: "wallColor",
    label: "Wall Color",
    category: "Structure",
    options: [
      {
        id: "worn_concrete",
        label: "Worn Concrete",
        tierMin: 1,
        vibe: "Industrial rebuild energy",
        tags: ["Rough", "Underdog", "Barebones"],
      },
      {
        id: "steel_navy",
        label: "Steel Navy",
        tierMin: 1,
        vibe: "Cold, focused, defensive atmosphere",
        tags: ["Serious", "Muted", "Traditional"],
      },
      {
        id: "charcoal_clean",
        label: "Charcoal Clean",
        tierMin: 2,
        vibe: "Cleaner structure without losing edge",
        tags: ["Professional", "Balanced", "Modern"],
      },
      {
        id: "executive_graphite",
        label: "Executive Graphite",
        tierMin: 3,
        vibe: "A controlled, established look",
        tags: ["Elite", "Sharp", "Authority"],
      },
      {
        id: "legacy_paneling",
        label: "Legacy Paneling",
        tierMin: 5,
        vibe: "Museum-grade tradition and prestige",
        tags: ["Historic", "Premium", "Dynasty"],
      },
    ],
  },
  {
    key: "flooring",
    label: "Flooring",
    category: "Structure",
    options: [
      {
        id: "scuffed_tile",
        label: "Scuffed Tile",
        tierMin: 1,
        vibe: "Cheap, practical, no-frills",
        tags: ["Starter", "Budget", "Rugged"],
      },
      {
        id: "dark_vinyl",
        label: "Dark Vinyl",
        tierMin: 1,
        vibe: "Stable but still economical",
        tags: ["Clean", "Simple", "Reliable"],
      },
      {
        id: "matte_wood",
        label: "Matte Wood",
        tierMin: 2,
        vibe: "Warmth and professionalism",
        tags: ["Balanced", "Modern", "Comfort"],
      },
      {
        id: "polished_hardwood",
        label: "Polished Hardwood",
        tierMin: 4,
        vibe: "Executive class finish",
        tags: ["Premium", "Luxury", "Finish"],
      },
      {
        id: "heritage_inlay",
        label: "Heritage Inlay",
        tierMin: 5,
        vibe: "Custom dynasty centerpiece",
        tags: ["Custom", "Legacy", "Elite"],
      },
    ],
  },
  {
    key: "desk",
    label: "Desk",
    category: "Furniture",
    options: [
      {
        id: "metal_basic",
        label: "Metal Utility Desk",
        tierMin: 1,
        vibe: "Functional and ugly in a charming way",
        tags: ["Basic", "Cold", "Starter"],
      },
      {
        id: "compact_maple",
        label: "Compact Maple Desk",
        tierMin: 2,
        vibe: "First signs of stability",
        tags: ["Tidy", "Improved", "Structured"],
      },
      {
        id: "executive_lshape",
        label: "Executive L-Desk",
        tierMin: 3,
        vibe: "Real hockey decisions get made here",
        tags: ["Command", "Organized", "Authority"],
      },
      {
        id: "glass_command",
        label: "Glass Command Desk",
        tierMin: 4,
        vibe: "High-end control center",
        tags: ["Modern", "Luxury", "Tech"],
      },
      {
        id: "legacy_showpiece",
        label: "Legacy Showpiece Desk",
        tierMin: 5,
        vibe: "A centerpiece worthy of a dynasty",
        tags: ["Monumental", "Iconic", "Prestige"],
      },
    ],
  },
  {
    key: "chair",
    label: "Chair",
    category: "Furniture",
    options: [
      {
        id: "folded_task",
        label: "Task Chair",
        tierMin: 1,
        vibe: "Bare minimum comfort",
        tags: ["Cheap", "Starter", "Functional"],
      },
      {
        id: "mesh_pro",
        label: "Mesh Pro Chair",
        tierMin: 2,
        vibe: "Better posture, better planning",
        tags: ["Upgrade", "Modern", "Support"],
      },
      {
        id: "executive_leather",
        label: "Executive Leather Chair",
        tierMin: 3,
        vibe: "The seat of a serious GM",
        tags: ["Authority", "Comfort", "Status"],
      },
      {
        id: "captains_lounge",
        label: "Captain's Lounge Chair",
        tierMin: 4,
        vibe: "Luxury with presence",
        tags: ["Premium", "Bold", "Luxury"],
      },
      {
        id: "legacy_throne",
        label: "Legacy Throne",
        tierMin: 5,
        vibe: "Ridiculous in the best possible way",
        tags: ["Dynasty", "Mythic", "Statement"],
      },
    ],
  },
  {
    key: "decor",
    label: "Decor",
    category: "Identity",
    options: [
      {
        id: "minimal",
        label: "Minimal Decor",
        tierMin: 1,
        vibe: "Nothing extra. Survive first.",
        tags: ["Bare", "Lean", "No-frills"],
      },
      {
        id: "team_posters",
        label: "Team Posters",
        tierMin: 1,
        vibe: "A small push toward identity",
        tags: ["Basic", "Pride", "Cheap"],
      },
      {
        id: "framed_history",
        label: "Framed History",
        tierMin: 3,
        vibe: "The room begins to tell a story",
        tags: ["Identity", "Tradition", "Narrative"],
      },
      {
        id: "executive_branding",
        label: "Executive Branding",
        tierMin: 4,
        vibe: "Corporate strength meets team culture",
        tags: ["Professional", "Strong", "Polished"],
      },
      {
        id: "hall_of_fame",
        label: "Hall of Fame Display",
        tierMin: 5,
        vibe: "This franchise is a cathedral now",
        tags: ["Legacy", "Historic", "Elite"],
      },
    ],
  },
];

const OFFICE_ADDITIONS = [
  {
    id: "scouting_board",
    label: "Scouting Board",
    short: "SCOUT",
    cost: 25000,
    icon: "SB",
    bonusLabel: "Scouting Accuracy",
    description:
      "Boosts scouting precision and reduces fog-of-war around prospects, hidden value, and later-round sleepers.",
    gameplayImpact: [
      "Improves prospect certainty",
      "Reduces hidden scouting error",
      "Makes board planning feel smarter",
    ],
    unlockTier: 1,
    flavor:
      "A wall of names, magnets, tape, and draft obsession. Hockey nerd heaven.",
  },
  {
    id: "analytics_station",
    label: "Analytics Station",
    short: "DATA",
    cost: 35000,
    icon: "AS",
    bonusLabel: "Advanced Evaluation",
    description:
      "Unlocks deeper stat context, sharper trade evaluation framing, and better roster fit analysis.",
    gameplayImpact: [
      "Advanced stat framing",
      "Improves trade-value context",
      "Better performance snapshots",
    ],
    unlockTier: 1,
    flavor:
      "Second monitor energy. Charts. Numbers. Someone saying 'underlying play-driving' every 14 minutes.",
  },
  {
    id: "meeting_room",
    label: "Meeting Room",
    short: "MEET",
    cost: 50000,
    icon: "MR",
    bonusLabel: "Morale / Communication",
    description:
      "Improves player and staff communication flow and supports morale-sensitive franchise interactions.",
    gameplayImpact: [
      "Supports morale systems",
      "Player/staff meeting immersion",
      "Better narrative decision staging",
    ],
    unlockTier: 2,
    flavor:
      "The room where hard conversations, deadline plans, and draft arguments happen.",
  },
  {
    id: "recovery_area",
    label: "Recovery Area",
    short: "REC",
    cost: 40000,
    icon: "RA",
    bonusLabel: "Recovery / Durability",
    description:
      "Improves recovery framing, shortens injury downtime conceptually, and supports stamina restoration systems.",
    gameplayImpact: [
      "Improved recovery support",
      "Better wellness optics",
      "Adds player-care identity",
    ],
    unlockTier: 2,
    flavor:
      "A practical investment that says your franchise stops treating players like disposable parts.",
  },
  {
    id: "archives_room",
    label: "Archives Room",
    short: "ARCH",
    cost: 45000,
    icon: "AR",
    bonusLabel: "Franchise Legacy",
    description:
      "Tracks team history, unlocks long-term memory flavor, and deepens the feeling of running a real organization.",
    gameplayImpact: [
      "Legacy immersion",
      "Historical framing",
      "Supports dynasty storytelling",
    ],
    unlockTier: 3,
    flavor:
      "Old media guides, retired photos, draft books, and the ghosts of every past decision.",
  },
  {
    id: "war_room",
    label: "War Room",
    short: "WAR",
    cost: 75000,
    icon: "WR",
    bonusLabel: "Draft / Deadline Command",
    description:
      "A dedicated decision bunker that sharpens draft-day confidence and deadline command-center identity.",
    gameplayImpact: [
      "Better draft atmosphere",
      "Trade deadline immersion",
      "High-stakes decision theater",
    ],
    unlockTier: 4,
    flavor:
      "This is where you stare at the board, stop breathing for six seconds, and either change the franchise or ruin it.",
  },
];

const OFFICE_PREVIEW_OBJECTS = [
  {
    id: "lighting",
    label: "Lighting State",
    tierTexts: [
      "Dim fluorescent survival lighting",
      "Cleaner overhead office lighting",
      "Sharper balanced management lighting",
      "High-end LED executive lighting",
      "Cinematic legacy spotlight system",
    ],
  },
  {
    id: "roomScale",
    label: "Room Scale",
    tierTexts: [
      "Cramped room",
      "Functional working office",
      "Expanded command space",
      "Executive premium suite",
      "Monumental headquarters floor",
    ],
  },
  {
    id: "staffPresence",
    label: "Staff Presence",
    tierTexts: [
      "Mostly empty",
      "Limited support presence",
      "Visible assistant movement",
      "Active executive ecosystem",
      "Full legacy front office operation",
    ],
  },
];

const AMBIENT_NOTES = [
  "AC hum under cheap fluorescent lights",
  "Muted arena noise filtering through the walls",
  "Monitor glow reflecting off paperwork",
  "Soft idle camera sway for a cinematic preview feel",
  "Occasional light flicker in the lower tiers",
  "Dust motes visible in the beam when the room is darker",
];

const PREVIEW_BACKDROP_BY_TIER = {
  1: "Concrete / worn utility backdrop",
  2: "Structured office wall backdrop",
  3: "Glass-accented management backdrop",
  4: "Skyline executive backdrop",
  5: "Legacy hall / trophy wall backdrop",
};

const TEAM_BRAND_MOTTOS = [
  "Build from nothing.",
  "Turn survival into structure.",
  "Turn structure into respect.",
  "Turn respect into power.",
  "Turn power into legacy.",
];

/* -------------------------------------------------------------------------- */
/*                               HELPER METHODS                               */
/* -------------------------------------------------------------------------- */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatCurrency(num) {
  try {
    return new Intl.NumberFormat("en-CA", {
      style: "currency",
      currency: "CAD",
      maximumFractionDigits: 0,
    }).format(Number(num) || 0);
  } catch {
    return `$${Number(num || 0).toLocaleString()}`;
  }
}

function cycleOptionIndex(currentIndex, dir, total) {
  if (!total) return 0;
  let next = currentIndex + dir;
  if (next < 0) next = total - 1;
  if (next >= total) next = 0;
  return next;
}

function findTierByPrestige(prestige) {
  return (
    OFFICE_TIERS.find((tier) => prestige >= tier.prestigeMin && prestige <= tier.prestigeMax) ||
    OFFICE_TIERS[0]
  );
}

function getNextTier(currentTierId) {
  return OFFICE_TIERS.find((tier) => tier.id === currentTierId + 1) || null;
}

function getTierProgressPercent(prestige, currentTier) {
  if (!currentTier) return 0;
  if (currentTier.id === 5) return 100;
  const span = Math.max(1, currentTier.prestigeMax - currentTier.prestigeMin + 1);
  const value = prestige - currentTier.prestigeMin;
  return clamp(Math.round((value / span) * 100), 0, 100);
}

function getLockedTierCount(currentTierId) {
  return OFFICE_TIERS.filter((tier) => tier.id > currentTierId).length;
}

function tierUnlocked(currentTierId, unlockTierId) {
  return currentTierId >= unlockTierId;
}

function calcRecordPoints(rec) {
  const w = Number(rec?.w) || 0;
  const otl = Number(rec?.otl) || 0;
  return w * 2 + otl;
}

function calcTotalGames(rec) {
  const w = Number(rec?.w) || 0;
  const l = Number(rec?.l) || 0;
  const otl = Number(rec?.otl) || 0;
  return w + l + otl;
}

function calcWinPct(rec) {
  const gp = calcTotalGames(rec);
  if (!gp) return 0;
  return (Number(rec?.w) || 0) / gp;
}

function buildPrestigeModel(franchiseState) {
  const team = franchiseState?.team || {};
  const rec = team?.record || {};
  const wins = Number(rec?.w) || 0;
  const losses = Number(rec?.l) || 0;
  const otl = Number(rec?.otl) || 0;
  const gp = wins + losses + otl;
  const championships = Number(franchiseState?.championships) || 0;
  const phase = String(franchiseState?.phase || "");
  const capPressure = Number(team?.cap_pressure?.replace?.("%", "")) || 0;

  const inPlayoffs = /playoff|postseason/i.test(phase);
  const winPct = gp ? wins / gp : 0;

  const playoffPush = inPlayoffs ? 1 : 0;
  const cupLegacy = championships;
  const playerGrowth = clamp(Math.round((wins * 0.12) + (winPct * 10)), 0, 12);
  const smartMoves = clamp(Math.round((100 - capPressure) / 16), 0, 6);
  const fanEnergy = clamp(Math.round((wins * 0.08) + (playoffPush * 3) + (championships * 4)), 0, 10);

  const detailed = {
    wins,
    losses,
    otl,
    gp,
    winPct,
    playoffPush,
    cupLegacy,
    playerGrowth,
    smartMoves,
    fanEnergy,
    capPressure,
  };

  let raw =
    wins * PRESTIGE_SOURCES[0].weight +
    playoffPush * PRESTIGE_SOURCES[1].weight +
    cupLegacy * PRESTIGE_SOURCES[2].weight +
    playerGrowth * PRESTIGE_SOURCES[3].weight +
    smartMoves * PRESTIGE_SOURCES[4].weight +
    fanEnergy * PRESTIGE_SOURCES[5].weight;

  raw += winPct * 14;
  raw += gp > 0 ? 8 : 0;
  raw += championships ? 5 : 0;

  const prestige = clamp(Math.round(raw), 0, 100);

  return { prestige, detailed };
}

function getAccentClassName(tierId) {
  switch (tierId) {
    case 1:
      return "is-accent-orange";
    case 2:
      return "is-accent-neon";
    case 3:
      return "is-accent-purple";
    case 4:
      return "is-accent-silver";
    case 5:
      return "is-accent-gold";
    default:
      return "";
  }
}

function getTierFlavorLine(tier) {
  if (!tier) return "Starting point.";
  return `${tier.label} · ${tier.theme} · ${tier.mood}`;
}

function getDefaultCustomizationForTier() {
  const result = {};
  for (const group of CUSTOMIZATION_GROUPS) {
    result[group.key] = 0;
  }
  return result;
}

function getCustomizationSelection(group, index) {
  return group?.options?.[index] || group?.options?.[0] || null;
}

function isCustomizationOptionLocked(currentTierId, option) {
  return (option?.tierMin || 1) > currentTierId;
}

function summarizeVisualState({ tier, customization, additionsOwned }) {
  const pieces = [];

  pieces.push(`Backdrop: ${PREVIEW_BACKDROP_BY_TIER[tier.id]}`);

  for (const group of CUSTOMIZATION_GROUPS) {
    const selection = getCustomizationSelection(group, customization[group.key]);
    if (selection) {
      pieces.push(`${group.label}: ${selection.label}`);
    }
  }

  const ownedLabels = OFFICE_ADDITIONS.filter((a) => additionsOwned[a.id]).map((a) => a.label);
  pieces.push(
    ownedLabels.length
      ? `Installed additions: ${ownedLabels.join(", ")}`
      : "Installed additions: None yet"
  );

  return pieces;
}

function buildOfficeBudgetEstimate(franchiseState) {
  const team = franchiseState?.team || {};
  const rec = team?.record || {};
  const wins = Number(rec?.w) || 0;
  const points = calcRecordPoints(rec);

  const base = 90000;
  const performance = wins * 3500 + points * 1250;
  const championships = (Number(franchiseState?.championships) || 0) * 60000;
  const phaseBonus = /playoff|postseason/i.test(String(franchiseState?.phase || "")) ? 45000 : 0;

  return Math.round(base + performance + championships + phaseBonus);
}

/* -------------------------------------------------------------------------- */
/*                             PRESENTATION BITS                              */
/* -------------------------------------------------------------------------- */

function OfficeCustomizationRow({
  group,
  selectedIndex,
  onChange,
  currentTierId,
}) {
  const selection = getCustomizationSelection(group, selectedIndex);
  const optionCount = group.options.length;

  const goLeft = () => {
    let next = selectedIndex;
    for (let i = 0; i < optionCount; i += 1) {
      next = cycleOptionIndex(next, -1, optionCount);
      const option = group.options[next];
      if (!isCustomizationOptionLocked(currentTierId, option)) {
        onChange(next);
        return;
      }
    }
  };

  const goRight = () => {
    let next = selectedIndex;
    for (let i = 0; i < optionCount; i += 1) {
      next = cycleOptionIndex(next, 1, optionCount);
      const option = group.options[next];
      if (!isCustomizationOptionLocked(currentTierId, option)) {
        onChange(next);
        return;
      }
    }
  };

  return (
    <div className="hub-office-cat hub-glass hub-glass--inner" style={{ padding: 10, borderRadius: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          marginBottom: 8,
        }}
      >
        <div>
          <div className="hub-office-rail__h" style={{ marginBottom: 2 }}>
            {group.label}
          </div>
          <div className="hub-office-rail__p" style={{ margin: 0 }}>
            {group.category}
          </div>
        </div>
        <span className="hub-live-status__chip">{selection?.label || "—"}</span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "36px minmax(0,1fr) 36px",
          gap: 8,
          alignItems: "center",
        }}
      >
        <button
          type="button"
          className="game-btn game-btn--secondary game-btn--sm ui-interactive"
          onClick={goLeft}
          data-tooltip={`Previous ${group.label}`}
        >
          ◀
        </button>

        <div className="hub-glass hub-glass--inner" style={{ padding: 10, borderRadius: 8 }}>
          <div
            style={{
              fontFamily: "var(--g-font-head)",
              fontSize: "0.72rem",
              letterSpacing: "0.08em",
              color: "var(--g-text)",
              marginBottom: 4,
            }}
          >
            {selection?.label || "Unavailable"}
          </div>
          <div
            style={{
              fontSize: "0.6875rem",
              color: "var(--g-silver-dim)",
              lineHeight: 1.35,
              marginBottom: 6,
            }}
          >
            {selection?.vibe || "No preview description available."}
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
            }}
          >
            {(selection?.tags || []).map((tag) => (
              <span key={tag} className="hub-live-status__chip">
                {tag}
              </span>
            ))}
          </div>
        </div>

        <button
          type="button"
          className="game-btn game-btn--secondary game-btn--sm ui-interactive"
          onClick={goRight}
          data-tooltip={`Next ${group.label}`}
        >
          ▶
        </button>
      </div>
    </div>
  );
}

function OfficeTierCard({ tier, currentTierId, prestige }) {
  const unlocked = currentTierId >= tier.id;
  const current = currentTierId === tier.id;

  return (
    <div
      className={`hub-tier-preview ui-interactive ${current ? "is-current" : ""}`}
      data-tooltip={`${tier.label} · ${tier.unlockText}`}
      style={{
        minHeight: 168,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        opacity: unlocked ? 1 : 0.65,
        position: "relative",
      }}
    >
      <div>
        <div className={`hub-tier-preview__viz hub-tier-preview__viz--${Math.min(tier.id, 5)}`} />
        <div
          style={{
            fontFamily: "var(--g-font-head)",
            fontSize: "0.7rem",
            letterSpacing: "0.08em",
            color: "var(--g-text)",
            marginBottom: 4,
          }}
        >
          {tier.label}
        </div>
        <div className="hub-tier-preview__lbl" style={{ marginBottom: 6 }}>
          {tier.theme} · {tier.mood}
        </div>
        <div
          style={{
            fontSize: "0.6875rem",
            lineHeight: 1.35,
            color: "var(--g-silver-dim)",
          }}
        >
          {tier.description}
        </div>
      </div>

      <div style={{ marginTop: 10 }}>
        <div className="hub-live-status__chip" style={{ marginBottom: 6 }}>
          {unlocked ? (current ? "CURRENT OFFICE" : "UNLOCKED") : "LOCKED"}
        </div>
        <div
          style={{
            fontSize: "0.6875rem",
            letterSpacing: "0.08em",
            color: unlocked ? "var(--g-neon)" : "var(--g-silver-dim)",
          }}
        >
          {unlocked ? `Prestige ${prestige}/100` : `Requires ${tier.prestigeNeeded} Prestige`}
        </div>
      </div>

      {!unlocked && (
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            fontFamily: "var(--font-ops-ui, Inter, sans-serif)",
            fontSize: "0.6875rem",
            fontWeight: 800,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--office-text-secondary, rgba(220, 216, 208, 0.58))",
            opacity: 0.9,
          }}
          aria-hidden="true"
        >
          LOCK
        </div>
      )}
    </div>
  );
}

function OfficeAdditionCard({
  addition,
  currentTierId,
  owned,
  estimatedBudget,
  onToggle,
}) {
  const unlockedByTier = currentTierId >= addition.unlockTier;
  const affordable = estimatedBudget >= addition.cost;
  const locked = !unlockedByTier;

  const statusText = locked
    ? `Unlocks at Tier ${addition.unlockTier}`
    : owned
    ? "Installed"
    : affordable
    ? "Available"
    : "Too expensive right now";

  return (
    <div className="hub-reward-tile hub-glass hub-glass--inner" style={{ padding: 12, textAlign: "left" }}>
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 10,
          marginBottom: 10,
        }}
      >
        <div style={{ display: "flex", gap: 10, minWidth: 0 }}>
          <div
            className="hub-glass hub-glass--inner"
            style={{
              width: 42,
              height: 42,
              borderRadius: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: "var(--g-font-head)",
              fontSize: "0.8rem",
              color: "var(--g-text)",
              flexShrink: 0,
            }}
          >
            {addition.icon}
          </div>
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                fontFamily: "var(--g-font-head)",
                fontSize: "0.72rem",
                letterSpacing: "0.08em",
                color: "var(--g-text)",
                marginBottom: 4,
              }}
            >
              {addition.label}
            </div>
            <div style={{ fontSize: "0.6875rem", color: "var(--g-silver-dim)" }}>
              {addition.bonusLabel}
            </div>
          </div>
        </div>

        <span className="hub-live-status__chip">{addition.short}</span>
      </div>

      <div
        style={{
          fontSize: "0.6875rem",
          color: "var(--g-silver-dim)",
          lineHeight: 1.45,
          marginBottom: 10,
        }}
      >
        {addition.description}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
        {addition.gameplayImpact.map((impact) => (
          <span key={impact} className="hub-live-status__chip">
            {impact}
          </span>
        ))}
      </div>

      <div
        style={{
          fontSize: "0.6875rem",
          lineHeight: 1.4,
          color: "var(--g-silver)",
          marginBottom: 10,
        }}
      >
        {addition.flavor}
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <div>
          <div
            style={{
              fontFamily: "var(--g-font-head)",
              fontSize: "0.72rem",
              color: "var(--g-orange)",
              marginBottom: 2,
            }}
          >
            {formatCurrency(addition.cost)}
          </div>
          <div style={{ fontSize: "0.6875rem", color: "var(--g-silver-dim)" }}>{statusText}</div>
        </div>

        <button
          type="button"
          className={`game-btn ${owned ? "game-btn--secondary" : "game-btn--primary"} game-btn--sm ui-interactive`}
          onClick={() => onToggle(addition.id)}
          disabled={locked}
          data-tooltip={locked ? `Requires Tier ${addition.unlockTier}` : owned ? "Remove from preview state" : "Add to preview state"}
        >
          {locked ? "LOCKED" : owned ? "REMOVE" : "INSTALL"}
        </button>
      </div>
    </div>
  );
}

function OfficePreviewCanvas({
  tier,
  customization,
  additionsOwned,
  teamName,
  prestige,
  currentMotto,
}) {
  const visualSummary = summarizeVisualState({ tier, customization, additionsOwned });

  return (
    <section className="office-screen__panel office-screen__panel--wide hub-glass hub-glass--inner">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.35fr) minmax(220px, 0.75fr)",
          gap: 16,
          minHeight: 420,
        }}
      >
        <div
          className={`hub-glass hub-glass--inner ${getAccentClassName(tier.id)}`}
          style={{
            borderRadius: 14,
            padding: 16,
            position: "relative",
            overflow: "hidden",
            minHeight: 420,
            background:
              tier.id === 1
                ? "linear-gradient(145deg, rgba(34,22,12,.62), rgba(6,10,18,.94))"
                : tier.id === 2
                ? "linear-gradient(145deg, rgba(18,40,52,.44), rgba(6,10,18,.94))"
                : tier.id === 3
                ? "linear-gradient(145deg, rgba(46,24,62,.42), rgba(6,10,18,.94))"
                : tier.id === 4
                ? "linear-gradient(145deg, rgba(42,42,52,.52), rgba(6,10,18,.94))"
                : "linear-gradient(145deg, rgba(72,48,12,.48), rgba(6,10,18,.94))",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "radial-gradient(circle at 18% 12%, rgba(255,255,255,.08), transparent 34%), radial-gradient(circle at 80% 78%, rgba(56,189,248,.12), transparent 38%)",
              pointerEvents: "none",
            }}
          />

          <div
            style={{
              position: "absolute",
              top: 18,
              left: 18,
              right: 18,
              display: "flex",
              justifyContent: "space-between",
              gap: 8,
              zIndex: 2,
              flexWrap: "wrap",
            }}
          >
            <span className="hub-live-status__chip">{tier.label}</span>
            <span className="hub-live-status__chip">Prestige {prestige}/100</span>
            <span className="hub-live-status__chip">{teamName || "Franchise HQ"}</span>
          </div>

          <div
            style={{
              position: "absolute",
              inset: "56px 16px 16px 16px",
              borderRadius: 12,
              overflow: "hidden",
              background:
                tier.id === 1
                  ? "linear-gradient(180deg, rgba(80,62,48,.16), rgba(14,18,28,.82))"
                  : tier.id === 2
                  ? "linear-gradient(180deg, rgba(50,90,100,.12), rgba(14,18,28,.82))"
                  : tier.id === 3
                  ? "linear-gradient(180deg, rgba(82,46,112,.12), rgba(14,18,28,.84))"
                  : tier.id === 4
                  ? "linear-gradient(180deg, rgba(110,110,150,.1), rgba(14,18,28,.82))"
                  : "linear-gradient(180deg, rgba(160,128,60,.13), rgba(14,18,28,.84))",
            }}
          >
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "flex-end",
                justifyContent: "center",
                paddingBottom: 22,
              }}
            >
              <div
                style={{
                  width: "78%",
                  maxWidth: 720,
                  aspectRatio: "16 / 9",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    left: "2%",
                    right: "2%",
                    bottom: 0,
                    height: "16%",
                    borderRadius: "50%",
                    background:
                      tier.id === 1
                        ? "radial-gradient(circle, rgba(140,120,90,.18), transparent 68%)"
                        : tier.id === 5
                        ? "radial-gradient(circle, rgba(255,210,90,.18), transparent 68%)"
                        : "radial-gradient(circle, rgba(56,189,248,.16), transparent 68%)",
                    filter: "blur(8px)",
                  }}
                />

                {/* rear wall */}
                <div
                  style={{
                    position: "absolute",
                    inset: "2% 0 24% 0",
                    borderRadius: 14,
                    border: "1px solid rgba(255,255,255,.06)",
                    background:
                      tier.id === 1
                        ? "linear-gradient(180deg, rgba(80,80,80,.22), rgba(26,26,32,.76))"
                        : tier.id === 2
                        ? "linear-gradient(180deg, rgba(90,110,120,.18), rgba(22,28,36,.78))"
                        : tier.id === 3
                        ? "linear-gradient(180deg, rgba(70,70,110,.18), rgba(22,22,36,.82))"
                        : tier.id === 4
                        ? "linear-gradient(180deg, rgba(110,110,130,.16), rgba(24,24,36,.82))"
                        : "linear-gradient(180deg, rgba(120,100,54,.16), rgba(20,18,14,.84))",
                  }}
                />

                {/* window / wall feature */}
                <div
                  style={{
                    position: "absolute",
                    right: tier.id >= 4 ? "6%" : "10%",
                    top: tier.id >= 4 ? "10%" : "16%",
                    width: tier.id >= 4 ? "26%" : "14%",
                    height: tier.id >= 4 ? "40%" : "20%",
                    borderRadius: 10,
                    background:
                      tier.id >= 4
                        ? "linear-gradient(180deg, rgba(56,189,248,.22), rgba(10,16,24,.16))"
                        : "rgba(255,255,255,.04)",
                    border: "1px solid rgba(255,255,255,.08)",
                    boxShadow: tier.id >= 4 ? "0 0 18px rgba(56,189,248,.12)" : "none",
                  }}
                />

                {/* desk */}
                <div
                  style={{
                    position: "absolute",
                    left: tier.id >= 4 ? "26%" : "30%",
                    right: tier.id >= 4 ? "26%" : "30%",
                    bottom: "14%",
                    height: tier.id >= 4 ? "20%" : "17%",
                    borderRadius: "14px 14px 8px 8px",
                    background:
                      tier.id === 1
                        ? "linear-gradient(180deg, rgba(90,96,110,.58), rgba(34,38,48,.88))"
                        : tier.id === 2
                        ? "linear-gradient(180deg, rgba(114,92,72,.56), rgba(44,34,24,.88))"
                        : tier.id === 3
                        ? "linear-gradient(180deg, rgba(80,88,110,.54), rgba(34,38,54,.9))"
                        : tier.id === 4
                        ? "linear-gradient(180deg, rgba(140,150,170,.48), rgba(40,48,70,.92))"
                        : "linear-gradient(180deg, rgba(170,150,100,.52), rgba(56,42,18,.92))",
                    border: "1px solid rgba(255,255,255,.08)",
                    zIndex: 2,
                  }}
                />

                {/* chair */}
                <div
                  style={{
                    position: "absolute",
                    left: "43%",
                    right: "43%",
                    bottom: "7%",
                    height: "16%",
                    borderRadius: "10px 10px 4px 4px",
                    background:
                      tier.id === 1
                        ? "rgba(54,60,68,.86)"
                        : tier.id === 5
                        ? "rgba(100,72,26,.88)"
                        : "rgba(44,52,74,.9)",
                    border: "1px solid rgba(255,255,255,.08)",
                    zIndex: 1,
                  }}
                />

                {/* monitors */}
                <div
                  style={{
                    position: "absolute",
                    left: tier.id >= 3 ? "34%" : "38%",
                    width: tier.id >= 3 ? "10%" : "7%",
                    bottom: "27%",
                    height: tier.id >= 2 ? "9%" : "6%",
                    borderRadius: 6,
                    background:
                      tier.id >= 2
                        ? "linear-gradient(180deg, rgba(56,189,248,.56), rgba(18,28,40,.9))"
                        : "rgba(150,150,160,.22)",
                    boxShadow: tier.id >= 2 ? "0 0 12px rgba(56,189,248,.16)" : "none",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    right: tier.id >= 3 ? "34%" : "38%",
                    width: tier.id >= 3 ? "10%" : "7%",
                    bottom: "27%",
                    height: tier.id >= 2 ? "9%" : "6%",
                    borderRadius: 6,
                    background:
                      tier.id >= 2
                        ? "linear-gradient(180deg, rgba(56,189,248,.56), rgba(18,28,40,.9))"
                        : "rgba(150,150,160,.22)",
                    boxShadow: tier.id >= 2 ? "0 0 12px rgba(56,189,248,.16)" : "none",
                  }}
                />

                {/* trophy cases at tier 5 */}
                {tier.id >= 5 && (
                  <>
                    <div
                      style={{
                        position: "absolute",
                        left: "5%",
                        bottom: "14%",
                        width: "14%",
                        height: "34%",
                        borderRadius: 10,
                        background: "linear-gradient(180deg, rgba(255,214,100,.18), rgba(24,20,12,.88))",
                        border: "1px solid rgba(255,214,100,.28)",
                        boxShadow: "0 0 20px rgba(255,214,100,.12)",
                      }}
                    />
                    <div
                      style={{
                        position: "absolute",
                        right: "5%",
                        bottom: "14%",
                        width: "14%",
                        height: "34%",
                        borderRadius: 10,
                        background: "linear-gradient(180deg, rgba(255,214,100,.18), rgba(24,20,12,.88))",
                        border: "1px solid rgba(255,214,100,.28)",
                        boxShadow: "0 0 20px rgba(255,214,100,.12)",
                      }}
                    />
                  </>
                )}

                {/* wall board */}
                <div
                  style={{
                    position: "absolute",
                    left: tier.id === 1 ? "8%" : "10%",
                    top: "16%",
                    width: tier.id >= 3 ? "18%" : "14%",
                    height: tier.id >= 3 ? "22%" : "16%",
                    borderRadius: 8,
                    border: "1px solid rgba(255,255,255,.08)",
                    background:
                      additionsOwned.scouting_board
                        ? "linear-gradient(180deg, rgba(255,255,255,.12), rgba(14,18,28,.92))"
                        : "rgba(255,255,255,.04)",
                  }}
                />

                {/* addition signal nodes */}
                <div
                  style={{
                    position: "absolute",
                    left: "8%",
                    right: "8%",
                    bottom: "3%",
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 6,
                    zIndex: 3,
                  }}
                >
                  {OFFICE_ADDITIONS.map((a) => {
                    const on = additionsOwned[a.id];
                    return (
                      <div
                        key={a.id}
                        title={a.label}
                        style={{
                          width: 30,
                          height: 10,
                          borderRadius: 999,
                          background: on ? "rgba(224,112,32,.86)" : "rgba(148,163,184,.24)",
                          boxShadow: on ? "0 0 10px rgba(224,112,32,.26)" : "none",
                          border: "1px solid rgba(255,255,255,.08)",
                        }}
                      />
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <div
            style={{
              position: "absolute",
              left: 18,
              bottom: 18,
              right: 18,
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0,1fr))",
              gap: 8,
              zIndex: 2,
            }}
          >
            {OFFICE_PREVIEW_OBJECTS.map((obj) => (
              <div key={obj.id} className="hub-glass hub-glass--inner" style={{ padding: 8, borderRadius: 10 }}>
                <div
                  style={{
                    fontSize: "0.6875rem",
                    letterSpacing: "0.12em",
                    color: "var(--g-silver-dim)",
                    marginBottom: 4,
                    textTransform: "uppercase",
                  }}
                >
                  {obj.label}
                </div>
                <div
                  style={{
                    fontSize: "0.6875rem",
                    lineHeight: 1.35,
                    color: "var(--g-text)",
                  }}
                >
                  {obj.tierTexts[tier.id - 1]}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            minWidth: 0,
          }}
        >
          <div className="hub-glass hub-glass--inner" style={{ padding: 14, borderRadius: 12 }}>
            <div className="hub-office-rail__h">Live Office Preview</div>
            <div className="hub-office-rail__p" style={{ marginBottom: 10 }}>
              Your franchise’s brain, personality, and evolution — visualized.
            </div>
            <div className="hub-live-status__chip" style={{ marginBottom: 8 }}>
              {getTierFlavorLine(tier)}
            </div>
            <div
              style={{
                fontSize: "0.72rem",
                lineHeight: 1.45,
                color: "var(--g-silver)",
                marginBottom: 10,
              }}
            >
              {tier.atmosphere}
            </div>
            <div
              style={{
                fontSize: "0.6875rem",
                color: "var(--g-silver-dim)",
                lineHeight: 1.45,
              }}
            >
              Motto: {currentMotto}
            </div>
          </div>

          <div className="hub-glass hub-glass--inner" style={{ padding: 14, borderRadius: 12 }}>
            <div className="hub-office-rail__h">Current Visual State</div>
            <ul className="hub-perk-list">
              {visualSummary.map((line) => (
                <li key={line} className="hub-perk-row">
                  <span className="hub-perk-row__txt">{line}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="hub-glass hub-glass--inner" style={{ padding: 14, borderRadius: 12 }}>
            <div className="hub-office-rail__h">Ambient Layer</div>
            <ul className="hub-perk-list">
              {AMBIENT_NOTES.map((note) => (
                <li key={note} className="hub-perk-row">
                  <span className="hub-perk-row__txt">{note}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

function PrestigeBreakdownCard({ prestigeData, currentTier, nextTier }) {
  const d = prestigeData.detailed;

  return (
    <section className="office-screen__panel hub-glass hub-glass--inner">
      <h3 className="hub-office-rail__h">Prestige System</h3>
      <p className="hub-office-rail__p">
        Prestige is your franchise reputation. It powers office evolution from survival to dynasty.
      </p>

      <div className="hub-reward-grid" style={{ marginBottom: 10 }}>
        <div className="hub-reward-tile hub-glass hub-glass--inner">
          <span className="hub-reward-tile__k">Current Tier</span>
          <span className="hub-reward-tile__v" style={{ fontSize: "0.95rem" }}>
            {currentTier.id}
          </span>
        </div>
        <div className="hub-reward-tile hub-glass hub-glass--inner">
          <span className="hub-reward-tile__k">Prestige</span>
          <span className="hub-reward-tile__v">{prestigeData.prestige}</span>
        </div>
        <div className="hub-reward-tile hub-glass hub-glass--inner">
          <span className="hub-reward-tile__k">Mood</span>
          <span className="hub-reward-tile__v" style={{ fontSize: "0.82rem" }}>
            {currentTier.mood}
          </span>
        </div>
        <div className="hub-reward-tile hub-glass hub-glass--inner">
          <span className="hub-reward-tile__k">Next Goal</span>
          <span className="hub-reward-tile__v" style={{ fontSize: "0.74rem" }}>
            {nextTier ? nextTier.label : "MAXED"}
          </span>
        </div>
      </div>

      <div className="hub-progress">
        <span className="hub-progress__title">
          {nextTier
            ? `${currentTier.label} → ${nextTier.label}`
            : "Legacy Headquarters fully unlocked"}
        </span>
        <div className="hub-progress__track">
          <div
            className="hub-progress__fill hub-progress__fill--gold"
            style={{ width: `${getTierProgressPercent(prestigeData.prestige, currentTier)}%` }}
          />
        </div>
        <span className="hub-progress__cap">
          {nextTier
            ? `${getTierProgressPercent(prestigeData.prestige, currentTier)}% through current prestige band`
            : "You have reached the final office tier"}
        </span>
      </div>

      <div style={{ marginTop: 12 }}>
        <div className="hub-office-rail__h">Prestige Drivers</div>
        <ul className="hub-perk-list">
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Winning Games</span>
            <span className="hub-perk-row__txt">{d.wins} wins powering early credibility</span>
          </li>
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Playoff Status</span>
            <span className="hub-perk-row__txt">
              {d.playoffPush ? "Playoff intensity is boosting your status" : "No active playoff aura right now"}
            </span>
          </li>
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Championship Legacy</span>
            <span className="hub-perk-row__txt">
              {d.cupLegacy} championships attached to franchise identity
            </span>
          </li>
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Development Signal</span>
            <span className="hub-perk-row__txt">
              Player growth index: {d.playerGrowth}
            </span>
          </li>
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Asset Management</span>
            <span className="hub-perk-row__txt">
              Smart move signal: {d.smartMoves}
            </span>
          </li>
          <li className="hub-perk-row">
            <span className="hub-perk-row__tier">Fan Energy</span>
            <span className="hub-perk-row__txt">
              Momentum / relevance signal: {d.fanEnergy}
            </span>
          </li>
        </ul>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/*                               MAIN COMPONENT                               */
/* -------------------------------------------------------------------------- */

export function OfficeScreen() {
  const {
    franchiseState,
    hubMenuIndex,
    setHubMenuIndex,
    openHubMenu,
    error,
    onAdvanceDay,
    advancing,
    refreshFranchise,
    setScreen,
  } = useGameUI();

  const hubMenuIndexRef = useRef(hubMenuIndex);
  hubMenuIndexRef.current = hubMenuIndex;

  const team = franchiseState?.team || {};
  const rec = team?.record || {};
  const teamName = team?.name || "Franchise";
  const franchisePhase = String(franchiseState?.phase || "Regular Season");
  const championships = Number(franchiseState?.championships) || 0;

  const [customization, setCustomization] = useState(getDefaultCustomizationForTier);
  const [appliedCustomization, setAppliedCustomization] = useState(getDefaultCustomizationForTier);
  const [additionsOwned, setAdditionsOwned] = useState({
    scouting_board: false,
    analytics_station: false,
    meeting_room: false,
    recovery_area: false,
    archives_room: false,
    war_room: false,
  });
  const [lastAppliedStamp, setLastAppliedStamp] = useState("No changes applied yet");
  const [previewMode, setPreviewMode] = useState("cinematic");
  const [viewLockedOnly, setViewLockedOnly] = useState(false);

  const prestigeData = useMemo(() => buildPrestigeModel(franchiseState), [franchiseState]);
  const currentTier = useMemo(() => findTierByPrestige(prestigeData.prestige), [prestigeData.prestige]);
  const nextTier = useMemo(() => getNextTier(currentTier.id), [currentTier.id]);
  const estimatedBudget = useMemo(() => buildOfficeBudgetEstimate(franchiseState), [franchiseState]);

  const recordLine = `${Number(rec?.w) || 0}-${Number(rec?.l) || 0}-${Number(rec?.otl) || 0}`;
  const points = calcRecordPoints(rec);
  const totalGames = calcTotalGames(rec);
  const winPct = calcWinPct(rec);
  const currentMotto = TEAM_BRAND_MOTTOS[Math.max(0, currentTier.id - 1)];
  const lockedTierCount = getLockedTierCount(currentTier.id);

  const visibleTierCards = useMemo(() => {
    if (!viewLockedOnly) return OFFICE_TIERS;
    return OFFICE_TIERS.filter((tier) => tier.id > currentTier.id);
  }, [viewLockedOnly, currentTier.id]);

  const availableAdditions = useMemo(() => {
    return OFFICE_ADDITIONS.map((addition) => ({
      ...addition,
      installed: !!additionsOwned[addition.id],
      tierUnlocked: tierUnlocked(currentTier.id, addition.unlockTier),
    }));
  }, [additionsOwned, currentTier.id]);

  const installedAdditionCount = useMemo(
    () => Object.values(additionsOwned).filter(Boolean).length,
    [additionsOwned]
  );

  const onRailClick = useCallback(
    (idx) => {
      setHubMenuIndex(idx);
    },
    [setHubMenuIndex]
  );

  const onRailDblClick = useCallback(
    (idx) => {
      openHubMenu(idx);
    },
    [openHubMenu]
  );

  const updateCustomization = useCallback((groupKey, optionIndex) => {
    setCustomization((prev) => ({
      ...prev,
      [groupKey]: optionIndex,
    }));
  }, []);

  const applyCustomization = useCallback(() => {
    setAppliedCustomization(customization);
    setLastAppliedStamp(`Applied ${new Date().toLocaleString()}`);
  }, [customization]);

  const resetPreviewToApplied = useCallback(() => {
    setCustomization(appliedCustomization);
  }, [appliedCustomization]);

  const toggleAddition = useCallback((additionId) => {
    setAdditionsOwned((prev) => ({
      ...prev,
      [additionId]: !prev[additionId],
    }));
  }, []);

  useEffect(() => {
    function onKey(e) {
      if (e.target.matches("input, textarea, select, button")) return;

      if (e.key === "Escape") {
        e.preventDefault();
        setScreen(SCREENS.HUB);
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setHubMenuIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setHubMenuIndex((i) => Math.min(HUB_MENU.length - 1, i + 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        openHubMenu(hubMenuIndexRef.current);
      } else if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        resetPreviewToApplied();
      } else if (e.key.toLowerCase() === "a") {
        e.preventDefault();
        applyCustomization();
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [applyCustomization, openHubMenu, resetPreviewToApplied, setHubMenuIndex, setScreen]);

  const previewNarrative = useMemo(() => {
    const selectionLines = CUSTOMIZATION_GROUPS.map((group) => {
      const sel = getCustomizationSelection(group, customization[group.key]);
      return `${group.label}: ${sel?.label || "Unknown"}`;
    });

    const installed = OFFICE_ADDITIONS.filter((a) => additionsOwned[a.id]).map((a) => a.label);

    return [
      `You currently occupy the ${currentTier.label}.`,
      currentTier.description,
      `The room theme is ${currentTier.theme.toLowerCase()} with a ${currentTier.mood.toLowerCase()} emotional tone.`,
      `Selected build: ${selectionLines.join(" · ")}.`,
      installed.length
        ? `Installed gameplay additions: ${installed.join(", ")}.`
        : "No gameplay additions are installed yet.",
    ].join(" ");
  }, [additionsOwned, currentTier, customization]);

  return (
    <div className="game-screen hub-screen hub-screen--takeover office-screen register-office" data-register="office">
      <div className="hub-franchise-shell">
        <aside className="hub-sidebar hub-glass" aria-label="Franchise command rail">
          <div className="hub-sidebar__brand">
            <span className="hub-sidebar__espn">FCN</span>
            <span className="hub-sidebar__title">NHL FRANCHISE</span>
            <span className="hub-sidebar__tag">GM TAKEOVER</span>
          </div>

          <nav className="hub-sidebar__nav">
            {HUB_MENU.map((item, idx) => (
              <div
                key={item.id}
                className={`hub-sidebar__link ui-interactive ${idx === hubMenuIndex ? "is-selected" : ""}`}
                data-tooltip={`${item.label} · Enter to open`}
                onClick={() => onRailClick(idx)}
                onDoubleClick={() => onRailDblClick(idx)}
                role="button"
                tabIndex={-1}
              >
                {item.label}
              </div>
            ))}
          </nav>
        </aside>

        <div className="hub-workspace">
          <header className="office-screen__header hub-glass">
            <div style={{ minWidth: 0 }}>
              <span className="office-screen__kicker">Office Sanctum // Infrastructure</span>
              <h1 className="office-screen__title">Franchise HQ Progression</h1>
              <p className="office-screen__sub">
                Prestige-driven headquarters evolution — from survival workspace to legacy monument. Preview and configure; apply when ready.
              </p>
            </div>

            <div className="office-screen__chips">
              <span className="hub-live-status__chip">{teamName}</span>
              <span className="hub-live-status__chip">REC {recordLine}</span>
              <span className="hub-live-status__chip">PTS {points}</span>
              <span className="hub-live-status__chip">{franchisePhase}</span>
            </div>
          </header>

          <div className="office-screen__body hub-glass">
            {error && <div className="game-toast game-toast--err hub-live-toast">{error}</div>}

            <div className="office-screen__grid">
              {/* CURRENT OFFICE / PRESTIGE SUMMARY */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Current Office</h3>
                <p className="hub-office-rail__p">
                  Tier-based progression hub. The office evolves from underdog survival to dynasty dominance.
                </p>

                <div className="hub-office-tier-card hub-glass hub-glass--inner">
                  <div className="hub-office-tier-card__now">
                    <span className="hub-office-tier-card__label">Current Tier</span>
                    <span className="hub-office-tier-card__name">{currentTier.label}</span>
                  </div>

                  <div className="hub-office-tier-card__next">
                    <span className="hub-office-tier-card__label">Theme</span>
                    <span className="hub-office-tier-card__name">
                      {currentTier.theme} · {currentTier.mood}
                    </span>
                  </div>

                  <div className="hub-progress">
                    <div className="hub-progress__track">
                      <div
                        className="hub-progress__fill"
                        style={{ width: `${getTierProgressPercent(prestigeData.prestige, currentTier)}%` }}
                      />
                    </div>
                    <span className="hub-progress__cap">
                      Prestige {prestigeData.prestige}/100
                      {nextTier ? ` · ${nextTier.prestigeNeeded - prestigeData.prestige > 0 ? nextTier.prestigeNeeded - prestigeData.prestige : 0} to next unlock` : " · Final tier reached"}
                    </span>
                  </div>
                </div>

                <ul className="hub-perk-list" style={{ marginTop: 10 }}>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Office Identity</span>
                    <span className="hub-perk-row__txt">{currentTier.description}</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Atmosphere</span>
                    <span className="hub-perk-row__txt">{currentTier.atmosphere}</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Perk Signal</span>
                    <span className="hub-perk-row__txt">{currentTier.perkText}</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Functional Summary</span>
                    <span className="hub-perk-row__txt">{currentTier.functionalSummary}</span>
                  </li>
                </ul>
              </section>

              {/* FRANCHISE SNAPSHOT */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Franchise Snapshot</h3>
                <div className="hub-reward-grid">
                  <div className="hub-reward-tile hub-glass hub-glass--inner">
                    <span className="hub-reward-tile__k">Games</span>
                    <span className="hub-reward-tile__v">{totalGames}</span>
                  </div>
                  <div className="hub-reward-tile hub-glass hub-glass--inner">
                    <span className="hub-reward-tile__k">Win %</span>
                    <span className="hub-reward-tile__v">
                      {Math.round(winPct * 100)}
                    </span>
                  </div>
                  <div className="hub-reward-tile hub-glass hub-glass--inner">
                    <span className="hub-reward-tile__k">Cups</span>
                    <span className="hub-reward-tile__v">{championships}</span>
                  </div>
                  <div className="hub-reward-tile hub-glass hub-glass--inner">
                    <span className="hub-reward-tile__k">Additions</span>
                    <span className="hub-reward-tile__v">{installedAdditionCount}</span>
                  </div>
                </div>

                <div style={{ marginTop: 10 }}>
                  <div className="hub-progress hub-progress--tight">
                    <span className="hub-progress__title">Estimated office investment capacity</span>
                    <div className="hub-progress__track">
                      <div
                        className="hub-progress__fill hub-progress__fill--gold"
                        style={{
                          width: `${clamp(Math.round((estimatedBudget / 350000) * 100), 8, 100)}%`,
                        }}
                      />
                    </div>
                    <span className="hub-progress__cap">{formatCurrency(estimatedBudget)} projected budget weight</span>
                  </div>
                </div>

                <ul className="hub-perk-list" style={{ marginTop: 10 }}>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Current Motto</span>
                    <span className="hub-perk-row__txt">{currentMotto}</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Phase</span>
                    <span className="hub-perk-row__txt">{franchisePhase}</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Locked Headquarters</span>
                    <span className="hub-perk-row__txt">{lockedTierCount} tiers still ahead of you</span>
                  </li>
                </ul>
              </section>

              {/* VIEW / MODE */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">View Controls</h3>
                <p className="hub-office-rail__p">
                  Keep this as a progression hub, a cinematic preview, or a clean system board.
                </p>

                <div className="hub-gm-style-row" style={{ marginBottom: 10 }}>
                  {["cinematic", "systems", "hybrid"].map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      className={`hub-gm-style-chip ui-interactive ${previewMode === mode ? "is-on" : ""}`}
                      onClick={() => setPreviewMode(mode)}
                    >
                      {mode.toUpperCase()}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  className="game-btn game-btn--secondary ui-interactive"
                  onClick={() => setViewLockedOnly((prev) => !prev)}
                  style={{ width: "100%", marginBottom: 10 }}
                >
                  {viewLockedOnly ? "SHOW ALL TIERS" : "SHOW LOCKED TIERS ONLY"}
                </button>

                <button
                  type="button"
                  className="game-btn game-btn--secondary ui-interactive"
                  onClick={resetPreviewToApplied}
                  style={{ width: "100%", marginBottom: 10 }}
                >
                  RESET TO APPLIED LOOK
                </button>

                <button
                  type="button"
                  className="game-btn game-btn--primary ui-interactive"
                  onClick={applyCustomization}
                  style={{ width: "100%" }}
                >
                  APPLY CHANGES
                </button>

                <p
                  style={{
                    marginTop: 10,
                    marginBottom: 0,
                    fontSize: "0.6875rem",
                    lineHeight: 1.4,
                    color: "var(--g-silver-dim)",
                  }}
                >
                  {lastAppliedStamp}
                </p>
              </section>

              {/* PRESTIGE SYSTEM */}
              <PrestigeBreakdownCard
                prestigeData={prestigeData}
                currentTier={currentTier}
                nextTier={nextTier}
              />

              {/* PREVIEW CANVAS */}
              <OfficePreviewCanvas
                tier={currentTier}
                customization={customization}
                additionsOwned={additionsOwned}
                teamName={teamName}
                prestige={prestigeData.prestige}
                currentMotto={currentMotto}
              />

              {/* LEFT-SIDE CUSTOMIZATION SYSTEM */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Customization Options</h3>
                <p className="hub-office-rail__p">
                  Real-time office shaping across structure, furniture, and identity. Higher-tier options stay locked
                  until your prestige catches up.
                </p>

                <div className="hub-office-cat-list">
                  {CUSTOMIZATION_GROUPS.map((group) => (
                    <OfficeCustomizationRow
                      key={group.key}
                      group={group}
                      selectedIndex={customization[group.key]}
                      onChange={(index) => updateCustomization(group.key, index)}
                      currentTierId={currentTier.id}
                    />
                  ))}
                </div>
              </section>

              {/* TIER STACK */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">View Other Offices</h3>
                <p className="hub-office-rail__p">
                  Every office tier should feel like a different world. Greyed-out tiers represent future prestige goals.
                </p>

                <div
                  className="hub-tier-strip"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: 10,
                  }}
                >
                  {visibleTierCards.map((tier) => (
                    <OfficeTierCard
                      key={tier.id}
                      tier={tier}
                      currentTierId={currentTier.id}
                      prestige={prestigeData.prestige}
                    />
                  ))}
                </div>
              </section>

              {/* ADDITIONS / GAMEPLAY INTEGRATION */}
              <section className="office-screen__panel office-screen__panel--wide hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Office Additions</h3>
                <p className="hub-office-rail__p">
                  These are not just cosmetic. Each addition is meant to reinforce mechanics like scouting, morale,
                  development, trade evaluation, recovery, and long-term franchise identity.
                </p>

                <div
                  className="hub-reward-grid"
                  style={{
                    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                    gap: 10,
                  }}
                >
                  {availableAdditions.map((addition) => (
                    <OfficeAdditionCard
                      key={addition.id}
                      addition={addition}
                      currentTierId={currentTier.id}
                      owned={!!additionsOwned[addition.id]}
                      estimatedBudget={estimatedBudget}
                      onToggle={toggleAddition}
                    />
                  ))}
                </div>
              </section>

              {/* TIER VISUAL PROGRESSION */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Visual Progression</h3>
                <p className="hub-office-rail__p">
                  The office should move from “this place is rough… but it’s mine” to “this franchise is a dynasty.”
                </p>
                <ul className="hub-perk-list">
                  {OFFICE_TIERS.map((tier) => (
                    <li key={tier.id} className="hub-perk-row">
                      <span className="hub-perk-row__tier">
                        {tier.shortLabel} · {tier.label}
                      </span>
                      <span className="hub-perk-row__txt">
                        {tier.visualBullets.join(" · ")}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>

              {/* PSYCHOLOGICAL / DESIGN INTENT */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Design Intent</h3>
                <p className="hub-office-rail__p">
                  This system should create attachment to the franchise, reward long-term play, and make office growth
                  feel like a visible reflection of success.
                </p>
                <ul className="hub-perk-list">
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Tier 1 Thought</span>
                    <span className="hub-perk-row__txt">“This is rough… I need to build.”</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Tier 3 Thought</span>
                    <span className="hub-perk-row__txt">“Okay, we’re legit now.”</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Tier 5 Thought</span>
                    <span className="hub-perk-row__txt">“This is a dynasty.”</span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Core Purpose</span>
                    <span className="hub-perk-row__txt">
                      Going from a basement startup to running the Yankees of hockey.
                    </span>
                  </li>
                </ul>
              </section>

              {/* GAMEPLAY CONNECTIONS */}
              <section className="office-screen__panel hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">System Integration</h3>
                <p className="hub-office-rail__p">
                  The office exists to support broader franchise systems — not sit as a disconnected cosmetic page.
                </p>
                <ul className="hub-perk-list">
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Player Development</span>
                    <span className="hub-perk-row__txt">
                      Office maturity should visually support progression, coaching investment, and development identity.
                    </span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Scouting</span>
                    <span className="hub-perk-row__txt">
                      Draft prep feels better when the room itself looks built for prospect evaluation.
                    </span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Trade AI / Decision Making</span>
                    <span className="hub-perk-row__txt">
                      Analytics Station and War Room push the fantasy that serious front-office work is happening.
                    </span>
                  </li>
                  <li className="hub-perk-row">
                    <span className="hub-perk-row__tier">Morale / Injury Systems</span>
                    <span className="hub-perk-row__txt">
                      Meeting Room and Recovery Area translate directly into believable franchise support structures.
                    </span>
                  </li>
                </ul>
              </section>

              {/* PREVIEW NARRATIVE */}
              <section className="office-screen__panel office-screen__panel--wide hub-glass hub-glass--inner">
                <h3 className="hub-office-rail__h">Narrative Readout</h3>
                <p
                  style={{
                    margin: 0,
                    fontSize: "0.78rem",
                    color: "var(--g-silver)",
                    lineHeight: 1.55,
                  }}
                >
                  {previewNarrative}
                </p>
              </section>
            </div>
          </div>

          <footer className="hub-action-bar hub-glass">
            <div className="hub-action-bar__buttons">
              <button
                type="button"
                className="game-btn game-btn--primary game-btn--advance ui-interactive"
                data-tooltip="Resolves the next calendar step when no decisions are pending"
                disabled={!franchiseState?.flags?.can_advance || advancing}
                onClick={onAdvanceDay}
              >
                {advancing ? "ADVANCING…" : "ADVANCE DAY"}
              </button>

              <button
                type="button"
                className="game-btn game-btn--secondary ui-interactive"
                data-tooltip="Pull the latest franchise payload from the API"
                disabled={advancing}
                onClick={refreshFranchise}
              >
                REFRESH
              </button>

              <button
                type="button"
                className="game-btn game-btn--secondary ui-interactive"
                data-tooltip="Apply current office customization selections"
                onClick={applyCustomization}
                disabled={advancing}
              >
                APPLY OFFICE CHANGES
              </button>
            </div>

            <div className="hub-action-bar__hints">
              ESC COMMAND DECK · ↑↓ SELECT · ENTER CONFIRM · A APPLY · R RESET PREVIEW
            </div>
          </footer>
        </div>
      </div>

      <GameFooter />
      <style>{OFFICE_SCREEN_CSS}</style>
    </div>
  );
}

const OFFICE_SCREEN_CSS = `
.office-screen.register-office {
  background:
    radial-gradient(circle at 18% 0%, rgba(201, 168, 106, 0.06), transparent 32%),
    linear-gradient(180deg, var(--office-bg, #101218), var(--office-bg-deep, #0c0e14));
  color: var(--office-text, #ece8e0);
}

.office-screen.register-office .hub-franchise-shell {
  background: transparent;
}

.office-screen.register-office .hub-sidebar,
.office-screen.register-office .office-screen__header,
.office-screen.register-office .office-screen__body,
.office-screen.register-office .hub-action-bar,
.office-screen.register-office .hub-glass,
.office-screen.register-office .hub-glass--inner {
  border-color: var(--office-line, rgba(255, 255, 255, 0.08)) !important;
  border-radius: var(--radius-hud, 4px) !important;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.03), transparent 24%),
    var(--office-desk, rgba(14, 16, 22, 0.9)) !important;
  box-shadow: var(--depth-hud, 0 18px 42px rgba(0, 0, 0, 0.48)) !important;
  backdrop-filter: blur(14px);
}

.office-screen.register-office .hub-sidebar__link.is-selected {
  background: rgba(201, 168, 106, 0.12) !important;
  color: var(--office-brass, #c9a86a) !important;
  box-shadow: inset 3px 0 0 rgba(201, 168, 106, 0.55);
}

.office-screen.register-office .hub-live-status__chip {
  border-radius: var(--radius-ops, 2px) !important;
  background: rgba(201, 168, 106, 0.08) !important;
  color: var(--office-brass, #c9a86a) !important;
  border: 1px solid rgba(201, 168, 106, 0.22);
}

.office-screen.register-office .office-screen__kicker {
  font-family: var(--font-mono-data, "IBM Plex Mono", monospace);
  font-size: 11px;
  letter-spacing: 0.16em;
  color: var(--office-brass, #c9a86a);
}

.office-screen.register-office .office-screen__title {
  font-family: var(--font-office-display, "Archivo Black", sans-serif);
  letter-spacing: 0.05em;
  color: var(--office-text, #ece8e0);
}

.office-screen.register-office .office-screen__sub,
.office-screen.register-office .hub-office-rail__p {
  font-family: var(--font-ops-ui, Inter, sans-serif);
  color: var(--office-text-secondary, rgba(220, 216, 208, 0.58));
}

.office-screen.register-office .hub-office-rail__h {
  font-family: var(--font-office-display, "Archivo Black", sans-serif);
  font-size: 0.82rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--office-text, #ece8e0);
}

.office-screen.register-office .hub-progress__fill {
  background: linear-gradient(90deg, rgba(201, 168, 106, 0.45), var(--office-brass, #c9a86a)) !important;
}

.office-screen.register-office .hub-progress__fill--gold {
  background: linear-gradient(90deg, rgba(201, 168, 106, 0.55), var(--office-brass, #c9a86a)) !important;
}

.office-screen.register-office .game-btn--primary {
  border-color: rgba(201, 168, 106, 0.45) !important;
  background: rgba(201, 168, 106, 0.14) !important;
  color: var(--office-text, #ece8e0) !important;
}

.office-screen.register-office .game-btn--secondary {
  border-color: var(--office-line, rgba(255, 255, 255, 0.08)) !important;
  color: var(--office-text-secondary, rgba(220, 216, 208, 0.72)) !important;
}

.office-screen.register-office .hub-reward-tile__v {
  color: var(--office-brass, #c9a86a) !important;
}

@media (prefers-reduced-motion: reduce) {
  .office-screen.register-office * {
    transition-duration: 0.01ms !important;
  }
}
`;

export default OfficeScreen;