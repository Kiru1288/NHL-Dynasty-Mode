import React, {
    Suspense,
    useCallback,
    useMemo,
    useRef,
    useState,
    useEffect,
  } from "react";
  
  import { Canvas, useFrame, useThree } from "@react-three/fiber";
  
  import {
    Html,
    OrbitControls,
    Text,
    RoundedBox,
    ContactShadows,
    Environment,
    Edges,
    SoftShadows,
    Sparkles,
    AccumulativeShadows,
    RandomizedLight,
    CameraShake,
    useGLTF,
  } from "@react-three/drei";
  
  import {
    EffectComposer,
    Bloom,
    Vignette,
    Noise,
  } from "@react-three/postprocessing";
  
  import { motion, AnimatePresence } from "framer-motion";
  import * as THREE from "three";
  import TeamLogoBadge from "../components/ui/TeamLogoBadge";
  import PlayerHeadshot from "../components/PlayerHeadshot";
  import { resolveFranchiseTeamLogo, toLogoUrl } from "../utils/teamLogos";
  import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
  import { SCREENS } from "../game/constants";
  import "./FirstPersonOfficeHub.css";
  import officeFontBold from "../styles/ArchivoBlack-Regular.ttf";
  import retroOfficePackGlb from "../styles/Retro Office Pack/Itch Upload/90s Retro Office Pack.glb";

  const OFFICE_HITBOXES = {
    dashboard: [0.72, 0.38, 0.38],
    messages: [0.42, 0.24, 0.42],
    calendar: [0.68, 0.18, 0.74],
    scouting: [0.48, 0.16, 0.42],
    contracts: [0.48, 0.16, 0.42],
    stats: [0.42, 0.16, 0.52],
    gameDayPuck: [0.24, 0.14, 0.24],
    news: [0.58, 0.16, 0.42],
    tasks: [0.42, 0.16, 0.52],
    teamIdentity: [1.25, 1.35, 0.24],
    lines: [0.9, 1.35, 0.42],
    standings: [1.28, 0.88, 0.24],
    leagueCentral: [1.45, 0.78, 0.24],
    draft: [1.65, 1.12, 0.24],
    awards: [0.82, 0.62, 0.28],
    arenaWindow: [1.85, 1.08, 0.24],
  };

  /** First-person executive seated eye line — command desk focal point */
  const OFFICE_CAMERA = {
    position: [0, 1.62, 3.85],
    target: [0, 1.36, -0.35],
    fov: 46,
    minDistance: 2.5,
    maxDistance: 6.8,
  };

  const OFFICE_PALETTE = {
    void: "#12151c",
    wall: "#1e2430",
    panel: "#181c26",
    walnut: "#2a2018",
    gunmetal: "#3a4048",
    leather: "#1a1816",
    gold: "#c9a86a",
    goldDim: "#8a7348",
    monitor: "#0f1824",
    monitorGlow: "#1a4a68",
    alert: "#8a3028",
  };

  const USE_RETRO_OFFICE_PACK = false;
  const USE_PROCEDURAL_ROOM_SHELL = true;

  /** Bundled from src — source: styles/Retro Office Pack/Itch Upload/ */
  const RETRO_OFFICE_MODEL_PATH = retroOfficePackGlb;

  const RETRO_OFFICE_TRANSFORM = {
    position: [0, 0, -0.95],
    rotation: [0, Math.PI, 0],
    scale: 0.82,
  };

  const pictureContext = (() => {
    try {
      return require.context("../pictures", false, /\.(png|jpe?g|webp|svg)$/i);
    } catch (err) {
      return null;
    }
  })();
  
  function getOfficePictures() {
    if (!pictureContext) return [];
  
    return pictureContext.keys().map((key) => {
      const asset = pictureContext(key);
  
      return {
        key,
        src: asset?.default || asset,
        name: key.replace("./", "").replace(/\.[^/.]+$/, ""),
      };
    });
  }

  function safeText(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
  }
  
  function initialsFromTeam(teamName = "NHL") {
    return String(teamName)
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((word) => word[0])
      .join("")
      .toUpperCase();
  }
  
  function formatRecord(record) {
    if (!record) return "0-0-0";
    if (typeof record === "string") return record;
  
    return `${record.w ?? record.wins ?? 0}-${record.l ?? record.losses ?? 0}-${
      record.otl ?? record.ot ?? record.overtime_losses ?? 0
    }`;
  }
  
  function formatMoney(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "string" && value.startsWith("$")) return value;

    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);

    const abs = Math.abs(n);
    if (abs >= 1000000) return `$${(n / 1000000).toFixed(2)}M`;
    if (abs >= 1000) return `$${(n / 1000).toFixed(0)}K`;
    return `$${n.toFixed(0)}`;
  }

  function titleCaseWords(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (m) => m.toUpperCase());
  }

  function formatOfficeMode(mode) {
    return titleCaseWords(mode || "regular");
  }

  function formatStandingsLine(line) {
    const text = String(line || "Standings");
    return text.replace(/^(\d+)\s/, (_, n) => {
      const num = Number(n);
      const mod10 = num % 10;
      const mod100 = num % 100;
      const suffix =
        mod100 >= 11 && mod100 <= 13
          ? "th"
          : mod10 === 1
            ? "st"
            : mod10 === 2
              ? "nd"
              : mod10 === 3
                ? "rd"
                : "th";
      return `${num}${suffix} `;
    });
  }

  function formatNextGameLabel(nextGame, phase = "") {
    const text = String(nextGame || "");
    if (text && text !== "No game listed" && text !== "Upcoming Game") return text;
    const ph = String(phase || "").toLowerCase();
    if (ph.includes("offseason")) return "Offseason — no game scheduled";
    if (ph.includes("complete")) return "Season complete";
    return "No game on schedule";
  }

  function officeSafeNumber(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function officeSafeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function officePhaseText(franchiseState) {
    const ph = franchiseState?.season_phase || franchiseState?.phase || "";
    const stage = franchiseState?.offseason_stage;
    const ui = franchiseState?.nhl_today?.ui_phase;
    if (ph === "offseason" && stage) return `${ph} ${stage}`;
    if (ui) return String(ui);
    return String(ph || "regular");
  }

  function countOfficeInjuries(franchiseState, team) {
    const pools = [
      franchiseState?.injuries,
      franchiseState?.medical?.injuries,
      franchiseState?.team?.injuries,
      franchiseState?.user_team?.injuries,
      team?.injuries,
    ];
    for (const pool of pools) {
      if (Array.isArray(pool)) return pool.length;
    }
    return officeSafeNumber(franchiseState?.injury_count, 0);
  }

  function parseStreakDirection(streak) {
    if (!streak) return null;
    const text = String(streak).toLowerCase();
    if (text.includes("w") || text.includes("win")) return "win";
    if (text.includes("l") || text.includes("loss")) return "loss";
    const n = officeSafeNumber(streak, NaN);
    if (Number.isFinite(n)) return n > 0 ? "win" : n < 0 ? "loss" : null;
    return null;
  }

  function parseStreakLength(streak) {
    if (!streak) return 0;
    const match = String(streak).match(/(\d+)/);
    return match ? officeSafeNumber(match[1], 0) : 0;
  }

  function deriveOfficeMood(franchiseState, team, officeSummary = {}) {
    const fs = franchiseState || {};
    const ph = String(fs.season_phase || fs.phase || "regular").toLowerCase();
    const stage = String(fs.offseason_stage || "").toLowerCase();
    const uiPhase = String(fs.nhl_today?.ui_phase || "").toLowerCase();
    const combined = `${ph} ${stage} ${uiPhase}`;

    const isOffseason = ph === "offseason" || combined.includes("offseason");
    const isPlayoffs =
      ph === "playoffs" ||
      ph === "playoff_ready" ||
      ph === "post_cup" ||
      combined.includes("playoff");
    const isCupFinal =
      combined.includes("cup final") ||
      combined.includes("stanley cup final") ||
      stage.includes("cup_final") ||
      stage.includes("final");
    const isTradeDeadline =
      combined.includes("trade deadline") ||
      combined.includes("deadline") ||
      uiPhase.includes("deadline") ||
      stage.includes("trade_deadline");
    const isDraftWeek =
      combined.includes("draft") &&
      (stage.includes("draft") || uiPhase.includes("draft") || isOffseason);
    const isFreeAgency =
      stage.includes("free_agency") ||
      stage.includes("free agency") ||
      combined.includes("free agency") ||
      combined.includes("free_agency");

    const streak =
      team?.streak ||
      team?.current_streak ||
      fs.streak ||
      fs.current_streak ||
      officeSummary?.streak;
    const streakDir = parseStreakDirection(streak);
    const streakLen = parseStreakLength(streak);
    const isLosingStreak = streakDir === "loss" && streakLen >= 3;
    const isHotStreak = streakDir === "win" && streakLen >= 3;

    const injuryCount = countOfficeInjuries(fs, team);
    const hasInjuryCrisis = injuryCount >= 3;

    const ownerConf = officeSafeNumber(
      fs.owner_confidence ??
        fs.owner?.confidence ??
        fs.owner?.approval ??
        fs.management?.owner_confidence,
      NaN
    );
    const hasOwnerPressure =
      (Number.isFinite(ownerConf) && ownerConf < 45) ||
      officeSafeNumber(officeSummary?.pendingTasks, 0) >= 4;

    const unread = officeSafeNumber(
      officeSummary?.unreadMessages ??
        fs.unread_messages ??
        fs.unreadMessages,
      0
    );
    const pending = officeSafeNumber(
      officeSummary?.pendingTasks ??
        fs.pending_tasks ??
        fs.pendingTasks,
      0
    );
    const hasUrgentDecisions =
      pending > 0 ||
      unread > 0 ||
      hasOwnerPressure ||
      hasInjuryCrisis ||
      officeSafeArray(fs.urgent_decisions).length > 0;

    let teamForm = "steady";
    if (isHotStreak) teamForm = "hot";
    else if (isLosingStreak) teamForm = "cold";
    else if (isPlayoffs) teamForm = "stakes";

    let pressureLevel = "low";
    if (hasOwnerPressure || isCupFinal || (isPlayoffs && isLosingStreak)) {
      pressureLevel = "critical";
    } else if (
      isTradeDeadline ||
      hasInjuryCrisis ||
      isLosingStreak ||
      hasUrgentDecisions
    ) {
      pressureLevel = "high";
    } else if (isDraftWeek || isFreeAgency || isPlayoffs || pending > 0) {
      pressureLevel = "medium";
    }

    let officeMode = "regular_season";
    if (isCupFinal) officeMode = "cup_final";
    else if (isPlayoffs) officeMode = "playoffs";
    else if (isTradeDeadline) officeMode = "trade_deadline";
    else if (isDraftWeek) officeMode = "draft_week";
    else if (isFreeAgency) officeMode = "free_agency";
    else if (isOffseason) officeMode = "offseason";
    else if (ph === "preseason") officeMode = "preseason";

    return {
      seasonPhase: ph || "regular",
      officeMode,
      pressureLevel,
      teamForm,
      isTradeDeadline,
      isDraftWeek,
      isFreeAgency,
      isPlayoffs,
      isOffseason,
      isCupFinal,
      isLosingStreak,
      isHotStreak,
      hasInjuryCrisis,
      hasOwnerPressure,
      hasUrgentDecisions,
      injuryCount,
      unreadMessages: unread,
      pendingTasks: pending,
    };
  }

  function buildOfficeUrgentItems(franchiseState, team, officeSummary = {}) {
    const items = [];
    const fs = franchiseState || {};
    const mood = deriveOfficeMood(fs, team, officeSummary);
    const push = (item) => {
      if (!item?.id || !item?.title) return;
      items.push({
        severity: "low",
        detail: "",
        target: OFFICE_NAV_TARGETS.DASHBOARD,
        ...item,
      });
    };

    const unread = mood.unreadMessages;
    if (unread > 0) {
      push({
        id: "unread-messages",
        type: "messages",
        severity: unread >= 5 ? "high" : "medium",
        title: `${unread} unread message${unread === 1 ? "" : "s"}`,
        detail: "Trade calls and league noise may need a response.",
        target: OFFICE_NAV_TARGETS.INBOX,
      });
    }

    const pending = mood.pendingTasks;
    if (pending > 0) {
      push({
        id: "pending-tasks",
        type: "tasks",
        severity: pending >= 3 ? "high" : "medium",
        title: `${pending} decision${pending === 1 ? "" : "s"} on the desk`,
        detail: "Front office priorities are waiting for your call.",
        target: OFFICE_NAV_TARGETS.TASKS,
      });
    }

    if (mood.hasOwnerPressure) {
      push({
        id: "owner-pressure",
        type: "owner",
        severity: "high",
        title: "Owner pressure rising",
        detail: "Leadership confidence may be narrowing your runway.",
        target: OFFICE_NAV_TARGETS.OWNER,
      });
    }

    if (mood.hasInjuryCrisis) {
      push({
        id: "injury-crisis",
        type: "injuries",
        severity: "high",
        title: "Injury report requires attention",
        detail: `${mood.injuryCount} active injuries are affecting roster stability.`,
        target: OFFICE_NAV_TARGETS.INJURIES,
      });
    }

    const capRaw =
      team?.cap_space ??
      team?.capSpace ??
      fs.cap_space ??
      officeSummary?.capSpaceRaw;
    const capNum = officeSafeNumber(capRaw, NaN);
    if (Number.isFinite(capNum) && capNum < 1500000) {
      push({
        id: "cap-tight",
        type: "contracts",
        severity: capNum < 0 ? "critical" : "medium",
        title: capNum < 0 ? "Cap space is underwater" : "Cap space is tight",
        detail: "Contract moves may require creativity before the next major decision.",
        target: OFFICE_NAV_TARGETS.SALARY_CAP,
      });
    }

    if (mood.isTradeDeadline) {
      push({
        id: "trade-deadline",
        type: "trade",
        severity: "high",
        title: "Trade market activity detected",
        detail: "Deadline pressure is live. Calls and offers may not wait.",
        target: OFFICE_NAV_TARGETS.TRADE_CALLS,
      });
    }

    if (mood.isDraftWeek) {
      push({
        id: "draft-board",
        type: "draft",
        severity: "medium",
        title: "Draft board update available",
        detail: "Final tier review is recommended before selections lock in.",
        target: OFFICE_NAV_TARGETS.DRAFT_BOARD,
      });
    }

    if (mood.isFreeAgency || mood.isOffseason) {
      push({
        id: "contract-decisions",
        type: "contracts",
        severity: "medium",
        title: "Contract decisions pending",
        detail: "RFAs, UFAs, and extension timing are on the clock.",
        target: OFFICE_NAV_TARGETS.CONTRACTS,
      });
    }

    const storyCount = officeSafeNumber(officeSummary?.activeStorylines, 0);
    if (storyCount > 0) {
      push({
        id: "storylines",
        type: "news",
        severity: "low",
        title: `${storyCount} active storyline${storyCount === 1 ? "" : "s"}`,
        detail: "Locker room and league narratives may need management.",
        target: OFFICE_NAV_TARGETS.STORYLINES,
      });
    }

    const nextGame = officeSummary?.nextGame;
    if (nextGame && nextGame !== "No game listed") {
      push({
        id: "next-game",
        type: "game",
        severity: mood.isPlayoffs ? "high" : "low",
        title: "Next game preparation available",
        detail: `Upcoming: ${nextGame}`,
        target: OFFICE_NAV_TARGETS.GAME_PREVIEW,
      });
    }

    if (mood.isLosingStreak) {
      push({
        id: "losing-streak",
        type: "performance",
        severity: "medium",
        title: "Losing streak flagged by staff",
        detail: "Analytics and coaching are tracking a slide in form.",
        target: OFFICE_NAV_TARGETS.TEAM_STATS,
      });
    }

    const tradeOffers = officeSafeArray(fs.trade_offers || fs.incoming_trades);
    if (tradeOffers.length > 0) {
      push({
        id: "trade-offers",
        type: "trade",
        severity: "medium",
        title: `${tradeOffers.length} trade offer${tradeOffers.length === 1 ? "" : "s"} on file`,
        detail: "Some proposals expire after the next game or phase advance.",
        target: OFFICE_NAV_TARGETS.TRADE_CALLS,
      });
    }

    const severityRank = { critical: 0, high: 1, medium: 2, low: 3 };
    return items.sort(
      (a, b) =>
        (severityRank[a.severity] ?? 9) - (severityRank[b.severity] ?? 9)
    );
  }

  const LOW_POWER_STORAGE_KEY = "nhlOfficeLowPowerMode";

  function detectWebGLSupport() {
    try {
      const canvas = document.createElement("canvas");
      return !!(
        window.WebGLRenderingContext &&
        (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
      );
    } catch (err) {
      return false;
    }
  }
    
  const OFFICE_PANEL_IDS = {
    DASHBOARD: "dashboard",
    MESSAGES: "messages",
    CALENDAR: "calendar",
    SCOUTING: "scouting",
    CONTRACTS: "contracts",
    STATS: "stats",
    LINES: "lines",
    NEWS: "news",
    AWARDS: "awards",
    DRAFT: "draft",
    STANDINGS: "standings",
    GAME_DAY: "gameDay",
    TEAM_IDENTITY: "teamIdentity",
    TASKS: "tasks",
    LEAGUE_CENTRAL: "leagueCentral",
  };

  const OFFICE_NAV_TARGETS = {
    DASHBOARD: "dashboard",
    SIM_NEXT_GAME: "sim-next-game",
    TEAM_REPORT: "team-report",
    OWNER_GOALS: "owner-goals",
    ROSTER: "roster",
    INJURIES: "injuries",

    INBOX: "inbox",
    TRADE_CALLS: "trade-calls",
    STAFF: "staff",
    OWNER: "owner",

    CALENDAR: "calendar",
    SIM_TO_DATE: "sim-to-date",
    EVENTS: "events",
    NEXT_GAME: "next-game",

    DRAFT_CLASS: "draft-class",
    SCOUTING: "scouting",
    WATCHLIST: "watchlist",
    ASSIGN_SCOUTS: "assign-scouts",

    CONTRACTS: "contracts",
    EXTENSIONS: "extensions",
    FREE_AGENCY: "free-agency",
    SALARY_CAP: "salary-cap",

    SKATER_STATS: "skater-stats",
    GOALIE_STATS: "goalie-stats",
    TEAM_STATS: "team-stats",
    ADVANCED_STATS: "advanced-stats",

    LINES: "lines",
    POWERPLAY: "powerplay",
    PENALTYKILL: "penaltykill",
    DEPTH_CHART: "depth-chart",

    STORYLINES: "storylines",
    LEAGUE_NEWS: "league-news",
    GAME_RECAPS: "recaps",
    RUMORS: "rumors",

    AWARDS: "awards",
    RECORDS: "records",
    HISTORY: "history",
    RETIRED_NUMBERS: "retired-numbers",

    DRAFT_BOARD: "draft-board",
    PROSPECT_RANKINGS: "prospect-rankings",
    TEAM_NEEDS: "team-needs",
    DRAFT_LOTTERY: "draft-lottery",

    STANDINGS: "standings",
    PLAYOFF_RACE: "playoff-race",
    POWER_RANKINGS: "power-rankings",
    DIVISION: "division",

    GAME_PREVIEW: "game-preview",
    SIM_GAME: "sim-game",
    BROADCAST: "broadcast",
    MATCHUP: "matchup",

    TEAM_PROFILE: "team-profile",
    FANBASE: "fanbase",
    MORALE: "morale",
    OWNERSHIP: "ownership",

    TASKS: "tasks",
    OBJECTIVES: "objectives",
    URGENT_DECISIONS: "urgent-decisions",
    STAFF_NOTES: "staff-notes",

    LEAGUE_CENTRAL: "league-central",
  };

  /*
   * ============================================================================
   * LEAGUE OPERATIONS — CONNECTION AUDIT (placeholder, not built yet)
   * ============================================================================
   * 1) WHERE THE UI LIVES
   *    - 3D Hub World wall: InteractiveGroup id="leagueCentral" (~line 4631)
   *      renders BroadcastScoreboard (static wall text + user record/next game).
   *    - Panel copy/registry: PANEL_CONTENT[OFFICE_PANEL_IDS.LEAGUE_CENTRAL],
   *      PLACEHOLDER_COPY.leagueCentral, FRANCHISE_COMMAND_REGISTRY id "league-central".
   *    - Full-screen placeholder: App.js CommandPlaceholderScreen via SCREENS.PLACEHOLDER.
   *
   * 2) WHAT OPENS IT
   *    - Clicking the 3D "League Operations" wall calls handleOpenPanel("leagueCentral").
   *    - PANEL_TO_COMMAND_TARGET maps leagueCentral → "league-central", so onNavigate
   *      runs BEFORE the in-office OfficePanel overlay (panel copy is bypassed).
   *    - HubScreen.handleNavigate → navigateFranchiseCommand → resolveCommandTarget
   *      ("league-central") → GameUIContext.openCommandPlaceholder → setScreen(PLACEHOLDER).
   *    - HubScreen OFFICE_PANEL_TO_SCREEN[LEAGUE_CENTRAL] also maps to PLACEHOLDER.
   *    - Sub-actions (Scores / League News / Broadcast / Game Recaps) are registered in
   *      OFFICE_NAV_TARGETS but only reachable if OfficePanel opens; most map to other
   *      placeholders (leagueNews, gameRecaps, arenaWindow for Broadcast).
   *
   * 3) DATA IT CURRENTLY RECEIVES
   *    - Wall decoration only: record + nextGame props from HubScreen (franchise state).
   *    - HubScreen derives record from team/franchiseState; nextGame from schedule_upcoming
   *      / next_game fields (see HubScreen findNextGame).
   *    - Placeholder screen gets static PLACEHOLDER_COPY.leagueCentral (title/subtitle/
   *      description) via commandPlaceholder in GameUIContext — no franchise payload.
   *    - No dedicated league_operations API call or route param.
   *
   * 4) BACKEND / SIM DATA TO POWER THE REAL FEATURE (already on franchise state)
   *    - build_state_payload (backend/services/franchise_sim.py): standings,
   *      schedule_upcoming, nhl_calendar_full (daily slates + final scores),
   *      storyline_events / notifications (league_news popup_scope), stats_central,
   *      game_results-derived scores. No league_central payload yet.
   *
   * 5) FILES TO EDIT WHEN BUILDING THE REAL FEATURE
   *    - frontend/src/screens/LeagueOperations.js (new) or extend Stats/Calendar patterns
   *    - frontend/src/screens/FirstPersonOfficeHub.js — wire league-central to real screen
   *      (FRANCHISE_COMMAND_REGISTRY, COMMAND_TARGET_ROUTES, PANEL_TO_COMMAND_TARGET)
   *    - frontend/src/screens/HubScreen.js — OFFICE_PANEL_TO_SCREEN mapping
   *    - frontend/src/game/constants.js — SCREENS entry if new route
   *    - frontend/src/App.js — mount real screen instead of CommandPlaceholderScreen
   *    - Optional backend: league_operations slice in build_state_payload if UI needs
   *      curated nightly scores / recaps / headlines bundle (no endpoint required today).
   * ============================================================================
   */

  export const FRANCHISE_COMMAND_GROUPS = {
    primary: { id: "primary", label: "Primary Commands" },
    operations: { id: "operations", label: "Hockey Operations" },
    frontOffice: { id: "frontOffice", label: "League & Front Office" },
    future: { id: "future", label: "Reserved / Coming Soon" },
  };

  const PLACEHOLDER_COPY = {
    gmPhone: {
      title: "GM Phone",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for inbox, trade calls, owner messages, and staff updates.",
    },
    legacyWall: {
      title: "Legacy Wall",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for awards, records, team history, and retired numbers.",
    },
    arenaWindow: {
      title: "Arena Window",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for game preview, broadcast, matchup reports, and game-day prep.",
    },
    decisionDesk: {
      title: "Decision Desk",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for tasks, objectives, urgent decisions, and staff notes.",
    },
    cultureWall: {
      title: "Franchise Culture Wall",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for team profile, fanbase, morale dashboards, and ownership direction.",
    },
    leagueCentral: {
      title: "League Operations",
      subtitle: "CBA desk, cap forecast, and team revenue.",
      description:
        "League-wide economics — salary cap growth, escrow, relocation risk, and team money.",
    },
    inbox: {
      title: "Inbox",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for GM inbox and front office messaging.",
    },
    ownerDesk: {
      title: "Owner Desk",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for owner goals, approval, and executive pressure.",
    },
    leagueNews: {
      title: "League News",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for league-wide news wire and transaction feed.",
    },
    gameRecaps: {
      title: "Game Recaps",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for nightly scoresheets and game recaps.",
    },
    assignScouts: {
      title: "Assign Scouts",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for scout assignments and coverage maps.",
    },
  };

  export const FRANCHISE_COMMAND_REGISTRY = [
    {
      id: "command-center",
      label: "Franchise Office",
      eyebrow: "Command Center",
      description: "Executive hub, franchise overview, and office systems.",
      group: "primary",
      target: "command-center",
      type: "hub",
      highlight: true,
      enabled: true,
    },
    {
      id: "roster",
      label: "Roster",
      eyebrow: "Personnel",
      description: "NHL roster, depth chart, roles, and injury list.",
      group: "primary",
      target: "roster",
      type: "navigate",
      screen: SCREENS.ROSTER,
      highlight: true,
      enabled: true,
    },
    {
      id: "calendar",
      label: "Calendar",
      eyebrow: "Schedule",
      description: "Season schedule, upcoming games, and league dates.",
      group: "primary",
      target: "calendar",
      type: "navigate",
      screen: SCREENS.CALENDAR,
      highlight: true,
      enabled: true,
    },
    {
      id: "strategy-board",
      label: "Strategy Board",
      eyebrow: "Tactics",
      description: "Forward lines, defensive pairs, and matchup planning.",
      group: "primary",
      target: "strategy-board",
      type: "navigate",
      screen: SCREENS.EDIT_LINES,
      highlight: true,
      enabled: true,
    },
    {
      id: "standings",
      label: "Standings",
      eyebrow: "League Table",
      description: "Division standings, playoff race, and points picture.",
      group: "primary",
      target: "standings",
      type: "navigate",
      screen: SCREENS.STATS,
      tab: "team",
      highlight: true,
      enabled: true,
    },
    {
      id: "draft-war-room",
      label: "Draft War Room",
      eyebrow: "Draft Mode",
      description: "Draft board prep, tiers, and selection strategy.",
      group: "primary",
      target: "draft-war-room",
      type: "navigate",
      screen: SCREENS.DRAFT_CLASS,
      highlight: true,
      enabled: true,
    },
    {
      id: "lines",
      label: "Lines",
      eyebrow: "Lineup",
      description: "Edit even-strength lines and defensive pairs.",
      group: "operations",
      target: "lines",
      type: "navigate",
      screen: SCREENS.EDIT_LINES,
      enabled: true,
    },
    {
      id: "power-play",
      label: "Power Play",
      eyebrow: "Special Teams",
      description: "Power play units and deployment.",
      group: "operations",
      target: "powerplay",
      type: "navigate",
      screen: SCREENS.POWER_PLAY,
      enabled: true,
    },
    {
      id: "penalty-kill",
      label: "Penalty Kill",
      eyebrow: "Special Teams",
      description: "Penalty kill units and structure.",
      group: "operations",
      target: "penaltykill",
      type: "navigate",
      screen: SCREENS.PENALTY_KILL,
      enabled: true,
    },
    {
      id: "scouting",
      label: "Scouting",
      eyebrow: "Amateur Ops",
      description: "Prospect reports, watchlists, and draft intel.",
      group: "operations",
      target: "scouting",
      type: "navigate",
      screen: SCREENS.SCOUTING,
      enabled: true,
    },
    {
      id: "team-needs",
      label: "Team Needs",
      eyebrow: "Roster Planning",
      description: "Positional needs and draft priorities.",
      group: "operations",
      target: "team-needs",
      type: "navigate",
      screen: SCREENS.TEAM_NEEDS,
      enabled: true,
    },
    {
      id: "contracts",
      label: "Contracts",
      eyebrow: "Cap Ledger",
      description: "Active contracts, extensions, and cap hits.",
      group: "operations",
      target: "contracts",
      type: "navigate",
      screen: SCREENS.CAP_LEDGER,
      tab: "contracts",
      enabled: true,
    },
    {
      id: "stats-analytics",
      label: "Stats / Analytics",
      eyebrow: "Performance Intel",
      description: "Skater, goalie, team, and advanced analytics.",
      group: "frontOffice",
      target: "stats-analytics",
      type: "navigate",
      screen: SCREENS.STATS,
      tab: "overview",
      enabled: true,
    },
    {
      id: "trade-hub",
      label: "Trade Hub",
      eyebrow: "Trade Floor",
      description: "Trade calls, offers, and roster moves.",
      group: "frontOffice",
      target: "trade-hub",
      type: "navigate",
      screen: SCREENS.TRADE,
      enabled: true,
    },
    {
      id: "free-agency",
      label: "Free Agency",
      eyebrow: "Cap Ledger",
      description: "UFA/RFA market, bids, and signings.",
      group: "frontOffice",
      target: "free-agency",
      type: "navigate",
      screen: SCREENS.CAP_LEDGER,
      tab: "freeAgency",
      enabled: true,
    },
    {
      id: "storylines",
      label: "Storylines / News",
      eyebrow: "Narrative",
      description: "League storylines, drama, and narrative beats.",
      group: "frontOffice",
      target: "storylines",
      type: "navigate",
      screen: SCREENS.STORYLINES,
      enabled: true,
    },
    {
      id: "chemistry",
      label: "Chemistry / Morale",
      eyebrow: "Locker Room",
      description: "Line chemistry, morale, and culture metrics.",
      group: "frontOffice",
      target: "chemistry",
      type: "navigate",
      screen: SCREENS.CHEMISTRY,
      enabled: true,
    },
    {
      id: "draft-class",
      label: "Draft Class",
      eyebrow: "Prospects",
      description: "Full draft class rankings and scouting dossiers.",
      group: "frontOffice",
      target: "draft-class",
      type: "navigate",
      screen: SCREENS.DRAFT_CLASS,
      enabled: true,
    },
    {
      id: "league-central",
      label: "League Operations",
      eyebrow: "League Economics",
      description: "CBA desk, cap forecast, team revenue, and relocation watch.",
      group: "frontOffice",
      target: "league-central",
      type: "navigate",
      screen: SCREENS.LEAGUE_OPERATIONS,
      enabled: true,
    },
    {
      id: "gm-phone",
      label: "GM Phone",
      eyebrow: "Communications",
      description: "Inbox, trade calls, owner messages, and staff updates.",
      group: "future",
      target: "gm-phone",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.gmPhone,
      enabled: true,
    },
    {
      id: "legacy-wall",
      label: "Legacy Wall",
      eyebrow: "History",
      description: "Awards, records, banners, and retired numbers.",
      group: "future",
      target: "legacy-wall",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.legacyWall,
      enabled: true,
    },
    {
      id: "arena-window",
      label: "Arena Window",
      eyebrow: "Game Day",
      description: "Game preview, broadcast, and matchup prep.",
      group: "future",
      target: "arena-window",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.arenaWindow,
      enabled: true,
    },
    {
      id: "decision-desk",
      label: "Decision Desk",
      eyebrow: "Tasks",
      description: "Pending decisions, objectives, and urgent items.",
      group: "future",
      target: "decision-desk",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.decisionDesk,
      enabled: true,
    },
  ];

  function commandToRoute(cmd) {
    if (!cmd) return null;
    if (cmd.type === "hub") return { type: "hub" };
    if (cmd.type === "placeholder") {
      return {
        type: "placeholder",
        placeholder: { ...cmd.placeholder, targetId: cmd.target },
      };
    }
    if (cmd.type === "navigate" && cmd.screen) {
      const route = { type: "screen", screen: cmd.screen };
      if (cmd.screen === SCREENS.CAP_LEDGER && cmd.tab) route.capTab = cmd.tab;
      if (cmd.screen === SCREENS.STATS && cmd.tab) route.statsTab = cmd.tab;
      return route;
    }
    return null;
  }

  const COMMAND_TARGET_ROUTES = (() => {
    const routes = {};

    FRANCHISE_COMMAND_REGISTRY.forEach((cmd) => {
      const route = commandToRoute(cmd);
      if (route) routes[cmd.target] = route;
    });

    const screenRoute = (screen, extras = {}) => ({ type: "screen", screen, ...extras });
    const ph = (copy, targetId) => ({
      type: "placeholder",
      placeholder: { ...copy, targetId },
    });

    routes[OFFICE_NAV_TARGETS.DASHBOARD] = { type: "hub" };
    routes[OFFICE_NAV_TARGETS.ROSTER] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.INJURIES] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.DEPTH_CHART] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.CALENDAR] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.EVENTS] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.NEXT_GAME] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.DRAFT_CLASS] = screenRoute(SCREENS.DRAFT_CLASS);
    routes[OFFICE_NAV_TARGETS.DRAFT_BOARD] = screenRoute(SCREENS.DRAFT_CLASS);
    routes[OFFICE_NAV_TARGETS.PROSPECT_RANKINGS] = screenRoute(SCREENS.DRAFT_CLASS);
    routes["draft-war-room"] = screenRoute(SCREENS.DRAFT_CLASS);
    routes["strategy-board"] = screenRoute(SCREENS.EDIT_LINES);
    routes["stats-analytics"] = screenRoute(SCREENS.STATS, { statsTab: "overview" });
    routes["trade-hub"] = screenRoute(SCREENS.TRADE);
    routes["command-center"] = { type: "hub" };
    routes[OFFICE_NAV_TARGETS.DRAFT_LOTTERY] = screenRoute(SCREENS.DRAFT_LOTTERY);
    routes[OFFICE_NAV_TARGETS.TEAM_NEEDS] = screenRoute(SCREENS.TEAM_NEEDS);
    routes[OFFICE_NAV_TARGETS.SCOUTING] = screenRoute(SCREENS.SCOUTING);
    routes[OFFICE_NAV_TARGETS.WATCHLIST] = screenRoute(SCREENS.SCOUTING);
    routes[OFFICE_NAV_TARGETS.CONTRACTS] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "contracts" });
    routes[OFFICE_NAV_TARGETS.EXTENSIONS] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "contracts" });
    routes[OFFICE_NAV_TARGETS.FREE_AGENCY] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "freeAgency" });
    routes[OFFICE_NAV_TARGETS.SALARY_CAP] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "salaryCap" });
    routes[OFFICE_NAV_TARGETS.SKATER_STATS] = screenRoute(SCREENS.STATS, { statsTab: "players" });
    routes[OFFICE_NAV_TARGETS.GOALIE_STATS] = screenRoute(SCREENS.STATS, { statsTab: "goalies" });
    routes[OFFICE_NAV_TARGETS.TEAM_STATS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.ADVANCED_STATS] = screenRoute(SCREENS.STATS, { statsTab: "advanced" });
    routes[OFFICE_NAV_TARGETS.STANDINGS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.PLAYOFF_RACE] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.POWER_RANKINGS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.DIVISION] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.LINES] = screenRoute(SCREENS.EDIT_LINES);
    routes[OFFICE_NAV_TARGETS.POWERPLAY] = screenRoute(SCREENS.POWER_PLAY);
    routes[OFFICE_NAV_TARGETS.PENALTYKILL] = screenRoute(SCREENS.PENALTY_KILL);
    routes[OFFICE_NAV_TARGETS.STORYLINES] = screenRoute(SCREENS.STORYLINES);
    routes[OFFICE_NAV_TARGETS.RUMORS] = screenRoute(SCREENS.STORYLINES);
    routes[OFFICE_NAV_TARGETS.MORALE] = screenRoute(SCREENS.CHEMISTRY);
    routes[OFFICE_NAV_TARGETS.TRADE_CALLS] = screenRoute(SCREENS.TRADE);

    routes[OFFICE_NAV_TARGETS.INBOX] = ph(PLACEHOLDER_COPY.inbox, OFFICE_NAV_TARGETS.INBOX);
    routes[OFFICE_NAV_TARGETS.STAFF] = ph(PLACEHOLDER_COPY.gmPhone, OFFICE_NAV_TARGETS.STAFF);
    routes[OFFICE_NAV_TARGETS.OWNER] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.OWNER);
    routes[OFFICE_NAV_TARGETS.TEAM_REPORT] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.TEAM_REPORT);
    routes[OFFICE_NAV_TARGETS.OWNER_GOALS] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.OWNER_GOALS);
    routes[OFFICE_NAV_TARGETS.TASKS] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.TASKS);
    routes[OFFICE_NAV_TARGETS.OBJECTIVES] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.OBJECTIVES);
    routes[OFFICE_NAV_TARGETS.URGENT_DECISIONS] = ph(
      PLACEHOLDER_COPY.decisionDesk,
      OFFICE_NAV_TARGETS.URGENT_DECISIONS
    );
    routes[OFFICE_NAV_TARGETS.STAFF_NOTES] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.STAFF_NOTES);
    routes[OFFICE_NAV_TARGETS.AWARDS] = screenRoute(SCREENS.STATS, { statsTab: "overview" });
    routes[OFFICE_NAV_TARGETS.RECORDS] = ph(PLACEHOLDER_COPY.legacyWall, OFFICE_NAV_TARGETS.RECORDS);
    routes[OFFICE_NAV_TARGETS.HISTORY] = ph(PLACEHOLDER_COPY.legacyWall, OFFICE_NAV_TARGETS.HISTORY);
    routes[OFFICE_NAV_TARGETS.RETIRED_NUMBERS] = ph(
      PLACEHOLDER_COPY.legacyWall,
      OFFICE_NAV_TARGETS.RETIRED_NUMBERS
    );
    routes[OFFICE_NAV_TARGETS.GAME_PREVIEW] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.GAME_PREVIEW);
    routes[OFFICE_NAV_TARGETS.BROADCAST] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.BROADCAST);
    routes[OFFICE_NAV_TARGETS.MATCHUP] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.MATCHUP);
    routes[OFFICE_NAV_TARGETS.SIM_GAME] = null;
    routes[OFFICE_NAV_TARGETS.TEAM_PROFILE] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.TEAM_PROFILE);
    routes[OFFICE_NAV_TARGETS.FANBASE] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.FANBASE);
    routes[OFFICE_NAV_TARGETS.OWNERSHIP] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.OWNERSHIP);
    routes[OFFICE_NAV_TARGETS.LEAGUE_CENTRAL] = screenRoute(SCREENS.LEAGUE_OPERATIONS);
    routes[OFFICE_NAV_TARGETS.LEAGUE_NEWS] = ph(PLACEHOLDER_COPY.leagueNews, OFFICE_NAV_TARGETS.LEAGUE_NEWS);
    routes[OFFICE_NAV_TARGETS.GAME_RECAPS] = ph(PLACEHOLDER_COPY.gameRecaps, OFFICE_NAV_TARGETS.GAME_RECAPS);
    routes[OFFICE_NAV_TARGETS.ASSIGN_SCOUTS] = ph(
      PLACEHOLDER_COPY.assignScouts,
      OFFICE_NAV_TARGETS.ASSIGN_SCOUTS
    );
    routes["gm-phone"] = ph(PLACEHOLDER_COPY.gmPhone, "gm-phone");
    routes["legacy-wall"] = ph(PLACEHOLDER_COPY.legacyWall, "legacy-wall");
    routes["arena-window"] = ph(PLACEHOLDER_COPY.arenaWindow, "arena-window");
    routes["decision-desk"] = ph(PLACEHOLDER_COPY.decisionDesk, "decision-desk");
    routes["culture-wall"] = ph(PLACEHOLDER_COPY.cultureWall, "culture-wall");

    return routes;
  })();

  export function resolveCommandTarget(target) {
    if (!target) return null;
    return COMMAND_TARGET_ROUTES[target] || null;
  }

  const QUICK_MENU_BADGE_TITLES = {
    Deadline: "Trade deadline window",
    Draft: "Draft week priority",
    FA: "Free agency period",
    Offseason: "Offseason operations",
    Pressure: "Owner pressure elevated",
    Injuries: "Injury crisis active",
    Playoffs: "Playoff push",
    Slide: "Losing streak flagged",
    Urgent: "Urgent desk item",
  };

  export {
    OFFICE_PANEL_IDS,
    OFFICE_NAV_TARGETS,
    deriveOfficeMood,
    buildOfficeUrgentItems,
    getDynamicPanelCopy,
    getContextualCommandRegistry,
    LOW_POWER_STORAGE_KEY,
  };

  const PANEL_TO_COMMAND_TARGET = {
    [OFFICE_PANEL_IDS.DASHBOARD]: "command-center",
    [OFFICE_PANEL_IDS.MESSAGES]: "gm-phone",
    [OFFICE_PANEL_IDS.CALENDAR]: "calendar",
    [OFFICE_PANEL_IDS.SCOUTING]: "scouting",
    [OFFICE_PANEL_IDS.CONTRACTS]: "contracts",
    [OFFICE_PANEL_IDS.STATS]: "stats-analytics",
    [OFFICE_PANEL_IDS.LINES]: "lines",
    [OFFICE_PANEL_IDS.NEWS]: "storylines",
    [OFFICE_PANEL_IDS.AWARDS]: "stats-analytics",
    [OFFICE_PANEL_IDS.DRAFT]: "draft-war-room",
    [OFFICE_PANEL_IDS.STANDINGS]: "standings",
    [OFFICE_PANEL_IDS.GAME_DAY]: "arena-window",
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: "culture-wall",
    [OFFICE_PANEL_IDS.TASKS]: "decision-desk",
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: "league-central",
  };

  const OFFICE_INTERACTIVE_PANEL_IDS = [
    OFFICE_PANEL_IDS.DASHBOARD,
    OFFICE_PANEL_IDS.MESSAGES,
    OFFICE_PANEL_IDS.CALENDAR,
    OFFICE_PANEL_IDS.SCOUTING,
    OFFICE_PANEL_IDS.CONTRACTS,
    OFFICE_PANEL_IDS.STATS,
    OFFICE_PANEL_IDS.NEWS,
    OFFICE_PANEL_IDS.TASKS,
    OFFICE_PANEL_IDS.TEAM_IDENTITY,
    OFFICE_PANEL_IDS.LINES,
    OFFICE_PANEL_IDS.STANDINGS,
    OFFICE_PANEL_IDS.LEAGUE_CENTRAL,
    OFFICE_PANEL_IDS.DRAFT,
    OFFICE_PANEL_IDS.AWARDS,
    OFFICE_PANEL_IDS.GAME_DAY,
  ];
  
  const PANEL_CONTENT = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      title: "Command Interface",
      eyebrow: "Executive Command Screen",
      description:
        "Review your franchise overview, roster status, owner goals, cap pressure, injuries, staff notes, and next decisions.",
      actions: [
        ["Sim Next Game", OFFICE_NAV_TARGETS.SIM_NEXT_GAME],
        ["Team Report", OFFICE_NAV_TARGETS.TEAM_REPORT],
        ["Owner Goals", OFFICE_NAV_TARGETS.OWNER_GOALS],
        ["Roster Status", OFFICE_NAV_TARGETS.ROSTER],
        ["Injury Watch", OFFICE_NAV_TARGETS.INJURIES],
      ],
    },
  
    [OFFICE_PANEL_IDS.MESSAGES]: {
      title: "GM Phone",
      eyebrow: "Trade Calls / Inbox",
      description:
        "Trade calls, owner messages, staff updates, league communication, and urgent front office notes.",
      actions: [
        ["Inbox", OFFICE_NAV_TARGETS.INBOX],
        ["Trade Calls", OFFICE_NAV_TARGETS.TRADE_CALLS],
        ["Staff Updates", OFFICE_NAV_TARGETS.STAFF],
        ["Owner Messages", OFFICE_NAV_TARGETS.OWNER],
      ],
    },
  
    [OFFICE_PANEL_IDS.CALENDAR]: {
      title: "Season Calendar",
      eyebrow: "Desk Calendar",
      description:
        "Review the schedule, upcoming games, league events, simulation dates, and important deadlines.",
      actions: [
        ["Schedule", OFFICE_NAV_TARGETS.CALENDAR],
        ["Sim to Date", OFFICE_NAV_TARGETS.SIM_TO_DATE],
        ["Important Dates", OFFICE_NAV_TARGETS.EVENTS],
        ["Next Game", OFFICE_NAV_TARGETS.NEXT_GAME],
      ],
    },
  
    [OFFICE_PANEL_IDS.SCOUTING]: {
      title: "Scouting Room",
      eyebrow: "Scouting Kit",
      description:
        "Review prospects, assignments, scouting reports, draft rankings, and watchlists.",
      actions: [
        ["Draft Class", OFFICE_NAV_TARGETS.DRAFT_CLASS],
        ["Scouting Reports", OFFICE_NAV_TARGETS.SCOUTING],
        ["Watchlist", OFFICE_NAV_TARGETS.WATCHLIST],
        ["Assign Scouts", OFFICE_NAV_TARGETS.ASSIGN_SCOUTS],
      ],
    },
  
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      title: "Contract Office",
      eyebrow: "Cap Ledger",
      description:
        "Manage contracts, free agency, salary cap, and roster money.",
      actions: [
        ["Contracts", OFFICE_NAV_TARGETS.CONTRACTS],
        ["Extensions", OFFICE_NAV_TARGETS.EXTENSIONS],
        ["Free Agency", OFFICE_NAV_TARGETS.FREE_AGENCY],
        ["Salary Cap", OFFICE_NAV_TARGETS.SALARY_CAP],
      ],
    },
  
    [OFFICE_PANEL_IDS.STATS]: {
      title: "Analytics Room",
      eyebrow: "Analytics Tablet",
      description:
        "Study player performance, team analytics, xGF%, CF%, PDO, power play, penalty kill, and trends.",
      actions: [
        ["Skater Stats", OFFICE_NAV_TARGETS.SKATER_STATS],
        ["Goalie Stats", OFFICE_NAV_TARGETS.GOALIE_STATS],
        ["Team Analytics", OFFICE_NAV_TARGETS.TEAM_STATS],
        ["Advanced Metrics", OFFICE_NAV_TARGETS.ADVANCED_STATS],
      ],
    },
  
    [OFFICE_PANEL_IDS.LINES]: {
      title: "Line Strategy Board",
      eyebrow: "Rink Whiteboard",
      description:
        "Edit forward lines, defensive pairs, special teams, matchup plans, and tactical setup.",
      actions: [
        ["Edit Lines", OFFICE_NAV_TARGETS.LINES],
        ["Power Play", OFFICE_NAV_TARGETS.POWERPLAY],
        ["Penalty Kill", OFFICE_NAV_TARGETS.PENALTYKILL],
        ["Depth Chart", OFFICE_NAV_TARGETS.DEPTH_CHART],
      ],
    },
  
    [OFFICE_PANEL_IDS.NEWS]: {
      title: "League Storylines",
      eyebrow: "Newspaper Stack",
      description:
        "Read headlines, rumors, game recaps, player drama, league movement, and front office noise.",
      actions: [
        ["Storylines", OFFICE_NAV_TARGETS.STORYLINES],
        ["League News", OFFICE_NAV_TARGETS.LEAGUE_NEWS],
        ["Game Recaps", OFFICE_NAV_TARGETS.GAME_RECAPS],
        ["Rumors", OFFICE_NAV_TARGETS.RUMORS],
      ],
    },
  
    [OFFICE_PANEL_IDS.AWARDS]: {
      title: "Legacy Wall",
      eyebrow: "Trophy Shelf",
      description:
        "View awards, records, team history, retired numbers, banners, and legacy moments.",
      actions: [
        ["Awards", OFFICE_NAV_TARGETS.AWARDS],
        ["Records", OFFICE_NAV_TARGETS.RECORDS],
        ["Team History", OFFICE_NAV_TARGETS.HISTORY],
        ["Retired Numbers", OFFICE_NAV_TARGETS.RETIRED_NUMBERS],
      ],
    },
  
    [OFFICE_PANEL_IDS.DRAFT]: {
      title: "Draft War Room",
      eyebrow: "Physical Draft Board",
      description:
        "Prepare for the draft with rankings, team needs, scouting lists, lottery odds, and prospect tiers.",
      actions: [
        ["Draft Board", OFFICE_NAV_TARGETS.DRAFT_BOARD],
        ["Prospect Rankings", OFFICE_NAV_TARGETS.PROSPECT_RANKINGS],
        ["Scouting", OFFICE_NAV_TARGETS.SCOUTING],
        ["Team Needs", OFFICE_NAV_TARGETS.TEAM_NEEDS],
        ["Draft Lottery", OFFICE_NAV_TARGETS.DRAFT_LOTTERY],
      ],
    },

    [OFFICE_PANEL_IDS.STANDINGS]: {
      title: "League Standings",
      eyebrow: "Standings Wall",
      description:
        "Track division races, playoff odds, conference battles, league rankings, and power movement.",
      actions: [
        ["Standings", OFFICE_NAV_TARGETS.STANDINGS],
        ["Playoff Race", OFFICE_NAV_TARGETS.PLAYOFF_RACE],
        ["Power Rankings", OFFICE_NAV_TARGETS.POWER_RANKINGS],
        ["Division View", OFFICE_NAV_TARGETS.DIVISION],
      ],
    },
  
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      title: "Arena Window",
      eyebrow: "Arena View",
      description:
        "Prepare for the next matchup, review lines, watch broadcast, check injuries, and simulate.",
      actions: [
        ["Game Preview", OFFICE_NAV_TARGETS.GAME_PREVIEW],
        ["Sim Game", OFFICE_NAV_TARGETS.SIM_GAME],
        ["Broadcast", OFFICE_NAV_TARGETS.BROADCAST],
        ["Matchup Report", OFFICE_NAV_TARGETS.MATCHUP],
      ],
    },
  
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      title: "Franchise Culture Wall",
      eyebrow: "Logo Wall",
      description:
        "Review team branding, culture, fanbase, morale, ownership direction, and long-term identity.",
      actions: [
        ["Team Profile", OFFICE_NAV_TARGETS.TEAM_PROFILE],
        ["Fanbase", OFFICE_NAV_TARGETS.FANBASE],
        ["Morale", OFFICE_NAV_TARGETS.MORALE],
        ["Ownership", OFFICE_NAV_TARGETS.OWNERSHIP],
      ],
    },
  
    [OFFICE_PANEL_IDS.TASKS]: {
      title: "Decision Desk",
      eyebrow: "Clipboard",
      description:
        "Review pending decisions, reminders, urgent items, owner pressure, and front office priorities.",
      actions: [
        ["Tasks", OFFICE_NAV_TARGETS.TASKS],
        ["Objectives", OFFICE_NAV_TARGETS.OBJECTIVES],
        ["Urgent Decisions", OFFICE_NAV_TARGETS.URGENT_DECISIONS],
        ["Staff Notes", OFFICE_NAV_TARGETS.STAFF_NOTES],
      ],
    },

    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      title: "League Operations",
      eyebrow: "League Economics",
      description: "CBA rules, cap growth, team revenue, and relocation risk.",
      actions: [
        ["Cap Forecast", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["Team Money", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["CBA Desk", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["Relocation Watch", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
      ],
    },
  };

  const PANEL_STAFF_SPEAKERS = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      role: "Assistant GM",
      fallback: "Start with what matters most today, then work outward.",
    },
    [OFFICE_PANEL_IDS.MESSAGES]: {
      role: "Assistant GM",
      fallback: "If the phone keeps ringing, something in the market is moving.",
    },
    [OFFICE_PANEL_IDS.CALENDAR]: {
      role: "Assistant GM",
      fallback: "The calendar is the truth. Miss a date and the league will not wait.",
    },
    [OFFICE_PANEL_IDS.SCOUTING]: {
      role: "Head Scout",
      fallback: "The public board is not your board. Review the tiers before draft day.",
    },
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      role: "Cap Specialist",
      fallback: "Do not approve anything long-term until we know the projected cap hit.",
    },
    [OFFICE_PANEL_IDS.STATS]: {
      role: "Analytics Director",
      fallback: "The standings are one story. The underlying numbers may be another.",
    },
    [OFFICE_PANEL_IDS.LINES]: {
      role: "Head Coach",
      fallback: "Matchups and minutes matter as much as names on the card.",
    },
    [OFFICE_PANEL_IDS.NEWS]: {
      role: "Assistant GM",
      fallback: "Narrative pressure is real even when the box score looks fine.",
    },
    [OFFICE_PANEL_IDS.AWARDS]: {
      role: "Assistant GM",
      fallback: "Legacy wins recruiting battles you never see on the scoresheet.",
    },
    [OFFICE_PANEL_IDS.DRAFT]: {
      role: "Head Scout",
      fallback: "Your board should be opinionated, tiered, and ready for chaos.",
    },
    [OFFICE_PANEL_IDS.STANDINGS]: {
      role: "Analytics Director",
      fallback: "Playoff probability and division math are tightening every week.",
    },
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      role: "Head Coach",
      fallback: "Game-day prep is where culture meets execution.",
    },
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      role: "Owner",
      fallback: "The building feels what the franchise believes about itself.",
    },
    [OFFICE_PANEL_IDS.TASKS]: {
      role: "Assistant GM",
      fallback: "Urgent does not always mean important, but ignored urgent becomes expensive.",
    },
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      role: "Assistant GM",
      fallback: "League money drives the next cap number.",
    },
  };

  const PANEL_PRESSURE_COPY = {
    [OFFICE_PANEL_IDS.CONTRACTS]:
      "Ignored contracts can turn into arbitration pressure or expensive July panic.",
    [OFFICE_PANEL_IDS.SCOUTING]:
      "If the board is stale by draft week, your staff may miss risers.",
    [OFFICE_PANEL_IDS.LINES]:
      "If injuries are ignored, fatigue and role mismatch can snowball.",
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]:
      "Low confidence can narrow your rebuild runway.",
    [OFFICE_PANEL_IDS.MESSAGES]:
      "Some offers expire after the next game or phase advance.",
    [OFFICE_PANEL_IDS.TASKS]:
      "Deferred decisions tend to arrive louder and more expensive.",
  };

  const PANEL_CAMERA_TARGETS = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      position: [0, 1.62, 2.55],
      target: [0, 1.12, 0.45],
    },
    [OFFICE_PANEL_IDS.MESSAGES]: {
      position: [-0.95, 1.58, 2.35],
      target: [-1.35, 1.08, 0.72],
    },
    [OFFICE_PANEL_IDS.CALENDAR]: {
      position: [0.85, 1.58, 2.35],
      target: [1.28, 1.06, 0.74],
    },
    [OFFICE_PANEL_IDS.SCOUTING]: {
      position: [-0.55, 1.58, 2.15],
      target: [-0.88, 1.06, 0.22],
    },
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      position: [0.55, 1.58, 2.15],
      target: [0.9, 1.06, 0.2],
    },
    [OFFICE_PANEL_IDS.STATS]: {
      position: [0.45, 1.62, 2.35],
      target: [0.72, 1.1, 0.68],
    },
    [OFFICE_PANEL_IDS.NEWS]: {
      position: [-0.65, 1.6, 2.45],
      target: [-1.05, 1.08, 1.05],
    },
    [OFFICE_PANEL_IDS.TASKS]: {
      position: [1.0, 1.6, 2.45],
      target: [1.38, 1.08, 1.1],
    },
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      position: [1.05, 1.82, 1.35],
      target: [1.62, 2.45, -3.2],
    },
    [OFFICE_PANEL_IDS.LINES]: {
      position: [-0.95, 1.82, 1.35],
      target: [-2.35, 2.0, -3.2],
    },
    [OFFICE_PANEL_IDS.STANDINGS]: {
      position: [-1.15, 1.45, 0.85],
      target: [-2.65, 0.92, -3.2],
    },
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      position: [1.05, 1.82, 1.35],
      target: [2.35, 2.05, -3.2],
    },
    [OFFICE_PANEL_IDS.DRAFT]: {
      position: [-2.35, 1.72, 0.15],
      target: [-4.2, 1.72, -1.35],
    },
    [OFFICE_PANEL_IDS.AWARDS]: {
      position: [2.35, 1.55, 0.15],
      target: [4.2, 1.52, -1.55],
    },
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      position: [2.15, 1.68, 0.55],
      target: [3.65, 1.62, -0.75],
    },
  };

  function getDynamicPanelCopy(
    panelId,
    basePanel,
    franchiseState,
    team,
    officeMood,
    urgentItems
  ) {
    const panel = basePanel || PANEL_CONTENT[panelId] || {};
    const mood = officeMood || deriveOfficeMood(franchiseState, team);
    const urgent = officeSafeArray(urgentItems);
    const urgentCount = urgent.length;
    const phase = officePhaseText(franchiseState);
    const topUrgent = urgent[0]?.title || "routine franchise maintenance";
    const speaker = PANEL_STAFF_SPEAKERS[panelId] || {
      role: "Assistant GM",
      fallback: "Keep the room calm and the decisions sharp.",
    };

    let description = panel.description || "";
    let staffNote = speaker.fallback;
    let pressureLine = PANEL_PRESSURE_COPY[panelId] || "";

    if (panelId === OFFICE_PANEL_IDS.DASHBOARD) {
      description = `Your front office has ${urgentCount} urgent item${urgentCount === 1 ? "" : "s"}, current phase is ${phase}, and the next major decision is ${topUrgent}.`;
      staffNote = `We have ${urgentCount} fires and ${mood.pendingTasks || 0} queued decisions. Triage before you get pulled into noise.`;
    } else if (panelId === OFFICE_PANEL_IDS.MESSAGES) {
      const tradeTone =
        mood.isTradeDeadline || mood.unreadMessages > 2 ? "active" : "quiet";
      description = `${mood.unreadMessages || 0} unread messages. Trade calls are ${tradeTone} depending on league phase.`;
      staffNote =
        mood.isTradeDeadline
          ? "Phones are hot. Filter noise and protect your leverage."
          : speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.CONTRACTS) {
      if (mood.isFreeAgency || mood.isOffseason) {
        description =
          "Contract season is live. Market pressure, comparables, and cap timing are all moving.";
      }
      const capRaw =
        team?.cap_space ?? team?.capSpace ?? franchiseState?.cap_space;
      const capNum = officeSafeNumber(capRaw, NaN);
      if (Number.isFinite(capNum) && capNum < 2000000) {
        description = `Cap space is tight at ${formatMoney(capRaw)}. Every move needs a second look.`;
        pressureLine = PANEL_PRESSURE_COPY[panelId];
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.SCOUTING) {
      if (mood.isDraftWeek || mood.isOffseason) {
        description =
          "Final board review is recommended before selections and tier calls lock in.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.LINES) {
      if (mood.hasInjuryCrisis) {
        description =
          "Lineup decisions are unstable with multiple injuries affecting roles and minutes.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.STATS) {
      if (mood.isLosingStreak) {
        description =
          "Analytics staff is flagging performance issues beneath the recent results.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.TEAM_IDENTITY) {
      if (mood.hasOwnerPressure) {
        description =
          "Ownership expectations are elevated. Culture and results are being measured together.";
        staffNote =
          "They are watching the room as closely as the standings. Keep the message consistent.";
        pressureLine = PANEL_PRESSURE_COPY[panelId];
      }
    } else if (panelId === OFFICE_PANEL_IDS.TASKS) {
      description = `${mood.pendingTasks || 0} pending decisions and ${urgentCount} urgent desk items need executive attention.`;
      staffNote = speaker.fallback;
    }

    if (panelId === OFFICE_PANEL_IDS.MESSAGES && mood.isTradeDeadline) {
      pressureLine = PANEL_PRESSURE_COPY[panelId];
    }
    if (panelId === OFFICE_PANEL_IDS.LINES && mood.hasInjuryCrisis) {
      pressureLine = PANEL_PRESSURE_COPY[panelId];
    }

    return {
      ...panel,
      description,
      staffNote,
      staffRole: speaker.role,
      pressureLine: pressureLine || PANEL_PRESSURE_COPY[panelId] || null,
    };
  }

  const QUICK_MENU_BADGE_RULES = {
    [OFFICE_PANEL_IDS.MESSAGES]: ["trade_deadline", "owner_pressure"],
    [OFFICE_PANEL_IDS.CONTRACTS]: ["free_agency", "offseason", "trade_deadline"],
    [OFFICE_PANEL_IDS.DRAFT]: ["draft_week"],
    [OFFICE_PANEL_IDS.SCOUTING]: ["draft_week"],
    [OFFICE_PANEL_IDS.TASKS]: ["owner_pressure"],
    [OFFICE_PANEL_IDS.LINES]: ["injury_crisis"],
    [OFFICE_PANEL_IDS.GAME_DAY]: ["playoffs", "injury_crisis"],
    [OFFICE_PANEL_IDS.STATS]: ["losing_streak"],
  };

  function getQuickMenuBadge(panelId, officeMood, urgentItems) {
    const mood = officeMood || {};
    const rules = QUICK_MENU_BADGE_RULES[panelId] || [];
    const urgent = officeSafeArray(urgentItems);

    if (rules.includes("trade_deadline") && mood.isTradeDeadline) return "Deadline";
    if (rules.includes("draft_week") && mood.isDraftWeek) return "Draft";
    if (rules.includes("free_agency") && mood.isFreeAgency) return "FA";
    if (rules.includes("offseason") && mood.isOffseason) return "Offseason";
    if (rules.includes("owner_pressure") && mood.hasOwnerPressure) return "Pressure";
    if (rules.includes("injury_crisis") && mood.hasInjuryCrisis) return "Injuries";
    if (rules.includes("playoffs") && mood.isPlayoffs) return "Playoffs";
    if (rules.includes("losing_streak") && mood.isLosingStreak) return "Slide";

    const panelUrgent = urgent.find((item) => {
      if (panelId === OFFICE_PANEL_IDS.MESSAGES) return item.type === "messages" || item.type === "trade";
      if (panelId === OFFICE_PANEL_IDS.TASKS) return item.type === "tasks";
      if (panelId === OFFICE_PANEL_IDS.CONTRACTS) return item.type === "contracts";
      if (panelId === OFFICE_PANEL_IDS.SCOUTING || panelId === OFFICE_PANEL_IDS.DRAFT) {
        return item.type === "draft";
      }
      if (panelId === OFFICE_PANEL_IDS.LINES) return item.type === "injuries";
      if (panelId === OFFICE_PANEL_IDS.GAME_DAY) return item.type === "game";
      return false;
    });

    if (panelUrgent?.severity === "critical" || panelUrgent?.severity === "high") {
      return "Urgent";
    }

    return "";
  }

  function getContextualCommandRegistry(baseRegistry, officeMood, urgentItems) {
    const mood = officeMood || {};
    const priority = [];

    const pushIds = (ids) => {
      ids.forEach((id) => {
        if (!priority.includes(id)) priority.push(id);
      });
    };

    if (mood.isTradeDeadline) {
      pushIds(["trade-hub", "league-central", "contracts", "stats-analytics", "lines"]);
    } else if (mood.isDraftWeek) {
      pushIds(["draft-war-room", "draft-class", "scouting", "team-needs", "contracts"]);
    } else if (mood.isFreeAgency || mood.isOffseason) {
      pushIds(["free-agency", "contracts", "scouting", "team-needs", "draft-class"]);
    } else if (mood.isPlayoffs) {
      pushIds(["strategy-board", "lines", "standings", "stats-analytics", "storylines"]);
    } else {
      pushIds(["calendar", "strategy-board", "lines", "standings", "roster"]);
    }

    if (mood.hasOwnerPressure) {
      pushIds(["decision-desk", "command-center", "storylines"]);
    }
    if (mood.hasInjuryCrisis) {
      pushIds(["lines", "roster", "strategy-board"]);
    }

    const rank = new Map(priority.map((id, index) => [id, index]));

    return officeSafeArray(baseRegistry)
      .map((item) => ({
        ...item,
        badge: getQuickMenuBadgeForCommand(item, mood, urgentItems),
      }))
      .sort((a, b) => {
        const aRank = rank.has(a.id) ? rank.get(a.id) : 99;
        const bRank = rank.has(b.id) ? rank.get(b.id) : 99;
        return aRank - bRank;
      });
  }

  function getQuickMenuBadgeForCommand(cmd, officeMood, urgentItems) {
    const panelId = cmd.panelId || cmd.id;
    return getQuickMenuBadge(panelId, officeMood, urgentItems);
  }

  function validateOfficeNavigation() {
    if (process.env.NODE_ENV === "production") return;

    const panelIds = new Set(Object.keys(PANEL_CONTENT));

    OFFICE_INTERACTIVE_PANEL_IDS.forEach((panelId) => {
      if (!panelIds.has(panelId)) {
        console.warn(
          "[OfficeNav] Interactive object opens missing panel:",
          panelId
        );
      }
    });

    const suspiciousPairs = [
      { labelIncludes: "game preview", targetMustInclude: "game-preview" },
      { labelIncludes: "calendar", targetMustInclude: "calendar" },
      { labelIncludes: "schedule", targetMustInclude: "calendar" },
      { labelIncludes: "broadcast", targetMustInclude: "broadcast" },
      { labelIncludes: "standings", targetMustInclude: "standings" },
      { labelIncludes: "draft board", targetMustInclude: "draft-board" },
      { labelIncludes: "salary cap", targetMustInclude: "salary-cap" },
      { labelIncludes: "contracts", targetMustInclude: "contracts" },
      { labelIncludes: "lines", targetMustInclude: "lines" },
      { labelIncludes: "power play", targetMustInclude: "powerplay" },
      { labelIncludes: "penalty kill", targetMustInclude: "penaltykill" },
    ];

    Object.entries(PANEL_CONTENT).forEach(([panelId, panel]) => {
      if (!panel?.title || !Array.isArray(panel.actions)) {
        console.warn("[OfficeNav] Bad panel config:", panelId, panel);
        return;
      }

      const seenTargets = new Set();

      panel.actions.forEach(([label, target]) => {
        if (!label || !target) {
          console.warn("[OfficeNav] Empty action label/target:", panelId, label, target);
        }

        if (seenTargets.has(target)) {
          console.warn("[OfficeNav] Duplicate target in panel:", panelId, target);
        }

        seenTargets.add(target);

        const normalizedLabel = String(label).toLowerCase();
        const normalizedTarget = String(target).toLowerCase();

        suspiciousPairs.forEach((rule) => {
          if (
            normalizedLabel.includes(rule.labelIncludes) &&
            !normalizedTarget.includes(rule.targetMustInclude)
          ) {
            console.warn(
              `[OfficeNav] Suspicious action mapping in ${panelId}: "${label}" -> "${target}". Expected target to include "${rule.targetMustInclude}".`
            );
          }
        });
      });
    });
  }
  
  const QUICK_MENU = FRANCHISE_COMMAND_REGISTRY;
  
  function CameraRig({
    resetToken,
    activePanel,
    lowPowerMode = false,
    prefersReducedMotion = false,
  }) {
    const controlsRef = useRef(null);
    const { camera } = useThree();
    const focusRef = useRef({
      position: new THREE.Vector3(...OFFICE_CAMERA.position),
      target: new THREE.Vector3(...OFFICE_CAMERA.target),
    });
    const [camX, camY, camZ] = OFFICE_CAMERA.position;
    const [tgtX, tgtY, tgtZ] = OFFICE_CAMERA.target;

    useEffect(() => {
      const snap = prefersReducedMotion || lowPowerMode;
      const panelTarget = activePanel ? PANEL_CAMERA_TARGETS[activePanel] : null;
      const nextPos = panelTarget?.position || OFFICE_CAMERA.position;
      const nextTarget = panelTarget?.target || OFFICE_CAMERA.target;

      focusRef.current.position.set(...nextPos);
      focusRef.current.target.set(...nextTarget);

      if (snap) {
        camera.position.set(...nextPos);
        if (controlsRef.current) {
          controlsRef.current.target.set(...nextTarget);
          controlsRef.current.update();
        } else {
          camera.lookAt(...nextTarget);
        }
      }
    }, [
      resetToken,
      activePanel,
      camera,
      lowPowerMode,
      prefersReducedMotion,
      camX,
      camY,
      camZ,
      tgtX,
      tgtY,
      tgtZ,
    ]);

    useFrame(() => {
      if (!controlsRef.current) return;
      const snap = prefersReducedMotion || lowPowerMode;
      const lerpFactor = snap ? 1 : 0.06;

      camera.position.lerp(focusRef.current.position, lerpFactor);
      controlsRef.current.target.lerp(focusRef.current.target, lerpFactor);
      controlsRef.current.update();
    });

    return (
      <>
        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom
          minDistance={OFFICE_CAMERA.minDistance}
          maxDistance={OFFICE_CAMERA.maxDistance}
          zoomSpeed={0.38}
          enableDamping
          dampingFactor={0.11}
          rotateSpeed={0.28}
          minPolarAngle={Math.PI / 2.88}
          maxPolarAngle={Math.PI / 2.04}
          minAzimuthAngle={-0.58}
          maxAzimuthAngle={0.58}
          target={OFFICE_CAMERA.target}
        />

        null
      </>
    );
  }
  function HoverLabel({ visible, label, description, badge }) {
    if (!visible) return null;
  
    return (
      <Html center distanceFactor={8.5} position={[0, 0.32, 0]}>
        <div className="office-object-label">
          <strong>{label}</strong>
          <span>{description}</span>
          {badge ? <em>{badge}</em> : null}
        </div>
      </Html>
    );
  }
  function BlinkingNotificationLight({ active, position = [0, 0, 0], color = "#d94a41" }) {
    const ref = useRef();

    useFrame((state) => {
      if (!ref.current || !active) return;
      const pulse = 0.35 + Math.sin(state.clock.elapsedTime * 5.5) * 0.25;
      ref.current.material.emissiveIntensity = pulse;
    });

    if (!active) return null;

    return (
      <mesh ref={ref} position={position} raycast={() => null}>
        <sphereGeometry args={[0.028, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.45}
          roughness={0.35}
        />
      </mesh>
    );
  }

  function InteractiveGroup({
    id,
    label,
    description,
    badge,
    children,
    position = [0, 0, 0],
    rotation = [0, 0, 0],
    scale = 1,
    hoveredId,
    setHoveredId,
    onOpen,
    hoverScale = 1.008,
    hoverLift = 0.0015,
    hitBoxArgs,
    hitBoxPosition = [0, 0.18, 0],
    openId = id,
    lowPowerMode = false,
  }) {
    const groupRef = useRef();
    const scaleVec = useRef(new THREE.Vector3(1, 1, 1));
    const isHovered = hoveredId === id;
    const effectiveHoverScale = lowPowerMode ? 1.004 : hoverScale;
    const effectiveHoverLift = lowPowerMode ? 0.0006 : hoverLift;

    useFrame((state) => {
      if (!groupRef.current) return;

      const targetScale = isHovered ? effectiveHoverScale : 1;

      groupRef.current.scale.lerp(
        scaleVec.current.set(
          targetScale * scale,
          targetScale * scale,
          targetScale * scale
        ),
        lowPowerMode ? 0.22 : 0.14
      );
  
      if (isHovered) {
        groupRef.current.position.y =
          position[1] +
          Math.sin(state.clock.elapsedTime * 2.2) * effectiveHoverLift;
      } else {
        groupRef.current.position.y +=
          (position[1] - groupRef.current.position.y) * 0.12;
      }
    });
  
    return (
      <group
        ref={groupRef}
        position={position}
        rotation={rotation}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredId(id);
          document.body.classList.add("office-cursor-active");
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHoveredId(null);
          document.body.classList.remove("office-cursor-active");
        }}
        onClick={(e) => {
          e.stopPropagation();
          onOpen(openId);
        }}
      >
        <mesh position={hitBoxPosition} renderOrder={999}>
          <boxGeometry
            args={hitBoxArgs || OFFICE_HITBOXES[id] || [0.75, 0.5, 0.75]}
          />
          <meshBasicMaterial
            transparent
            opacity={0}
            depthWrite={false}
            color="#ffffff"
          />
        </mesh>
  
        {children(isHovered)}
  
        <HoverLabel
          visible={isHovered}
          label={label}
          description={description}
          badge={badge}
        />
      </group>
    );
  }
  
  function GlowMaterial({
    color = "#1b2536",
    emissive = "#000000",
    intensity = 0.2,
    roughness = 0.55,
    metalness = 0.1,
  }) {
    return (
      <meshStandardMaterial
        color={color}
        emissive={emissive}
        emissiveIntensity={intensity}
        roughness={roughness}
        metalness={metalness}
      />
    );
  }
  
  function WallText({
    children,
    position,
    rotation = [0, 0, 0],
    size = 0.12,
    color = "#f8ead5",
    anchorX = "center",
    anchorY = "middle",
    maxWidth = 2,
  }) {
    return (
      <Text
        font={officeFontBold}
        position={position}
        rotation={rotation}
        fontSize={size}
        color={color}
        anchorX={anchorX}
        anchorY={anchorY}
        maxWidth={maxWidth}
        textAlign="center"
      >
        {children}
      </Text>
    );
  }
  
  function ScreenGlassMaterial({ hovered }) {
    return (
      <meshPhysicalMaterial
        color={hovered ? "#081420" : OFFICE_PALETTE.monitor}
        emissive={hovered ? "#1e5a82" : OFFICE_PALETTE.monitorGlow}
        emissiveIntensity={hovered ? 0.48 : 0.26}
        roughness={0.12}
        metalness={0.18}
        clearcoat={0.82}
        clearcoatRoughness={0.14}
        transparent
        opacity={0.96}
      />
    );
  }

  function SmokedGlassMaterial({ opacity = 0.14, hovered = false }) {
    return (
      <meshPhysicalMaterial
        color={hovered ? "#1a2230" : "#0e1218"}
        emissive={hovered ? "#2a3a4a" : "#101820"}
        emissiveIntensity={hovered ? 0.12 : 0.05}
        roughness={0.08}
        metalness={0.22}
        transparent
        opacity={opacity}
        clearcoat={0.9}
        clearcoatRoughness={0.08}
      />
    );
  }

  function WoodMaterial({
    color = "#4b2a19",
    roughness = 0.44,
    metalness = 0.05,
  }) {
    return (
      <meshStandardMaterial color={color} roughness={roughness} metalness={metalness} />
    );
  }

  function PaperMaterial({ color = "#f0ead8", roughness = 0.78 }) {
    return <meshStandardMaterial color={color} roughness={roughness} metalness={0.02} />;
  }

  function LeatherMaterial({ color = "#231f1d", roughness = 0.72 }) {
    return <meshStandardMaterial color={color} roughness={roughness} metalness={0.03} />;
  }

  function MetalMaterial({
    color = "#8a7350",
    roughness = 0.28,
    metalness = 0.72,
  }) {
    return (
      <meshStandardMaterial color={color} roughness={roughness} metalness={metalness} />
    );
  }

  function PlasticMaterial({ color = "#1a1f28", roughness = 0.52 }) {
    return <meshStandardMaterial color={color} roughness={roughness} metalness={0.08} />;
  }

  function GlassMaterial({ opacity = 0.22 }) {
    return (
      <meshPhysicalMaterial
        color="#c8dce8"
        roughness={0.08}
        metalness={0.04}
        transparent
        opacity={opacity}
        clearcoat={0.85}
        clearcoatRoughness={0.12}
      />
    );
  }

  function WoodGrainLines({ width = 4.35, depth = 1.38, y = 1.031, z = 1.03, count = 16 }) {
    return (
      <group>
        {Array.from({ length: count }).map((_, i) => (
          <mesh
            key={`grain-${i}`}
            position={[-width / 2 + (i / (count - 1)) * width, y, z]}
            raycast={() => null}
          >
            <boxGeometry args={[0.005, 0.002, depth]} />
            <meshStandardMaterial
              color="#352018"
              roughness={0.62}
              transparent
              opacity={0.28}
            />
          </mesh>
        ))}
      </group>
    );
  }

  function FloorPlanks({ width = 9, depth = 8, count = 22 }) {
    return (
      <group position={[0, 0.004, -1.1]} rotation={[-Math.PI / 2, 0, 0]}>
        {Array.from({ length: count }).map((_, i) => (
          <mesh
            key={`plank-${i}`}
            position={[(-width / 2 + (i / (count - 1)) * width), 0, 0]}
            raycast={() => null}
          >
            <planeGeometry args={[0.012, depth]} />
            <meshBasicMaterial color="#1a100a" transparent opacity={0.22} />
          </mesh>
        ))}
      </group>
    );
  }

  function OfficeRug() {
    return (
      <group position={[0, 0.012, 0.35]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <mesh receiveShadow>
          <planeGeometry args={[3.1, 2.0]} />
          <meshStandardMaterial color="#12100e" roughness={0.94} metalness={0.01} />
        </mesh>
        <mesh position={[0, 0.001, 0]}>
          <planeGeometry args={[2.85, 1.75]} />
          <meshStandardMaterial color="#181410" roughness={0.9} transparent opacity={0.42} />
        </mesh>
      </group>
    );
  }

  function ExecutiveChairSilhouette() {
    return (
      <group position={[0, 0.78, 2.95]} raycast={() => null}>
        <mesh position={[0, 0.62, 0]} castShadow>
          <boxGeometry args={[0.88, 1.18, 0.07]} />
          <LeatherMaterial color={OFFICE_PALETTE.leather} roughness={0.9} />
        </mesh>
        {[-0.48, 0.48].map((x) => (
          <mesh key={`arm-${x}`} position={[x, 0.14, 0.18]} rotation={[0.42, 0, 0]}>
            <boxGeometry args={[0.12, 0.05, 0.38]} />
            <LeatherMaterial color="#080807" roughness={0.88} />
          </mesh>
        ))}
        <mesh position={[0, 0.08, 0.32]}>
          <boxGeometry args={[0.72, 0.16, 0.42]} />
          <LeatherMaterial color="#0d0c0b" roughness={0.86} />
        </mesh>
      </group>
    );
  }

  function CeilingLightStrip() {
    return (
      <group position={[0, 3.98, -1.2]} raycast={() => null}>
        <mesh>
          <boxGeometry args={[5.2, 0.04, 0.18]} />
          <MetalMaterial color="#1a1c22" roughness={0.35} metalness={0.72} />
        </mesh>
        <mesh position={[0, -0.018, 0]}>
          <boxGeometry args={[4.8, 0.012, 0.1]} />
          <meshStandardMaterial
            color="#f0ddb0"
            emissive="#d4b060"
            emissiveIntensity={0.62}
            roughness={0.55}
          />
        </mesh>
      </group>
    );
  }

  function WallDisplayFrame({ width = 1.85, height = 1.08, children, accent = "#c9a86a" }) {
    return (
      <group>
        <mesh position={[0, 0, -0.028]} castShadow raycast={() => null}>
          <boxGeometry args={[width + 0.12, height + 0.12, 0.05]} />
          <MetalMaterial color="#14161c" roughness={0.42} metalness={0.68} />
        </mesh>
        <mesh position={[0, 0, -0.012]} raycast={() => null}>
          <boxGeometry args={[width, height, 0.03]} />
          <meshStandardMaterial color="#080a0e" roughness={0.72} metalness={0.08} />
        </mesh>
        <mesh position={[0, height / 2 + 0.02, 0.01]} raycast={() => null}>
          <boxGeometry args={[width - 0.08, 0.012, 0.008]} />
          <meshStandardMaterial
            color={accent}
            emissive={accent}
            emissiveIntensity={0.18}
            roughness={0.45}
          />
        </mesh>
        {children}
      </group>
    );
  }

  function Baseboards() {
    const trimMat = (
      <meshStandardMaterial color="#1a1512" roughness={0.68} metalness={0.04} />
    );

    return (
      <group raycast={() => null}>
        <mesh position={[0, 0.08, -3.48]} receiveShadow>
          <boxGeometry args={[8.6, 0.14, 0.06]} />
          {trimMat}
        </mesh>
        <mesh position={[-4.38, 0.08, -0.15]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
          <boxGeometry args={[6.7, 0.14, 0.06]} />
          {trimMat}
        </mesh>
        <mesh position={[4.38, 0.08, -0.15]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
          <boxGeometry args={[6.7, 0.14, 0.06]} />
          {trimMat}
        </mesh>
      </group>
    );
  }

  function WallPanelStrips() {
    return (
      <group position={[0, 2.05, -3.41]} raycast={() => null}>
        {[-3.2, -1.6, 0, 1.6, 3.2].map((x) => (
          <mesh key={`panel-${x}`} position={[x, 0, 0.02]}>
            <boxGeometry args={[0.028, 3.8, 0.012]} />
            <meshStandardMaterial color="#16181f" roughness={0.78} metalness={0.06} />
          </mesh>
        ))}
        {[-1.2, 0.5, 2.1].map((y) => (
          <mesh key={`seam-${y}`} position={[0, y, 0.022]}>
            <boxGeometry args={[8.4, 0.012, 0.01]} />
            <meshStandardMaterial color="#0e1016" roughness={0.82} metalness={0.04} />
          </mesh>
        ))}
        <mesh position={[0, -1.55, 0.03]}>
          <boxGeometry args={[6.8, 0.018, 0.01]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.goldDim}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={0.08}
            roughness={0.5}
            metalness={0.35}
          />
        </mesh>
      </group>
    );
  }

  function SmallRivets({ radius = 0.75, count = 4 }) {
    const positions = [
      [-radius, radius],
      [radius, radius],
      [-radius, -radius],
      [radius, -radius],
    ].slice(0, count);

    return positions.map(([x, y], i) => (
      <mesh key={`rivet-${i}`} position={[x, y, 0.04]} raycast={() => null}>
        <cylinderGeometry args={[0.012, 0.012, 0.018, 8]} />
        <MetalMaterial color="#6a5a42" roughness={0.35} metalness={0.78} />
      </mesh>
    ));
  }

  function WallFrame({ width = 2.28, height = 1.24, depth = 0.05, children }) {
    return (
      <group>
        <mesh position={[0, 0, -0.01]} castShadow raycast={() => null}>
          <boxGeometry args={[width, height, depth]} />
          <WoodMaterial color="#151210" roughness={0.55} />
        </mesh>
        <mesh position={[0, 0, 0.018]} castShadow raycast={() => null}>
          <boxGeometry args={[width - 0.14, height - 0.14, 0.022]} />
          <MetalMaterial color="#3a342c" roughness={0.42} metalness={0.55} />
        </mesh>
        {children}
      </group>
    );
  }

  function DeskLamp({ position = [-1.88, 1.034, 0.38] }) {
    return (
      <group position={position} raycast={() => null}>
        <mesh position={[0, 0.018, 0]} castShadow>
          <cylinderGeometry args={[0.055, 0.065, 0.028, 16]} />
          <MetalMaterial color="#5c4a32" roughness={0.32} metalness={0.68} />
        </mesh>
        <mesh position={[0.04, 0.09, -0.02]} rotation={[0.35, 0, -0.42]} castShadow>
          <cylinderGeometry args={[0.008, 0.008, 0.14, 8]} />
          <MetalMaterial color="#4a4034" roughness={0.38} metalness={0.62} />
        </mesh>
        <mesh position={[0.08, 0.16, -0.04]} rotation={[0.15, 0, 0]} castShadow>
          <cylinderGeometry args={[0.09, 0.11, 0.08, 20, 1, true]} />
          <meshStandardMaterial
            color="#2a2218"
            roughness={0.78}
            metalness={0.04}
            side={THREE.DoubleSide}
          />
        </mesh>
        <mesh position={[0.08, 0.145, -0.04]}>
          <circleGeometry args={[0.055, 20]} />
          <meshStandardMaterial
            color="#ffd8a0"
            emissive="#ffb86a"
            emissiveIntensity={0.35}
            roughness={0.5}
          />
        </mesh>
        <pointLight
          position={[0.08, 0.12, -0.04]}
          intensity={0.55}
          color="#ffcc88"
          distance={1.8}
          castShadow={false}
        />
      </group>
    );
  }

  function DeskPen({ position = [1.05, 1.034, 0.92] }) {
    return (
      <group position={position} rotation={[0, -0.4, 0]} raycast={() => null}>
        <mesh rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.006, 0.006, 0.14, 8]} />
          <PlasticMaterial color="#1c2533" />
        </mesh>
        <mesh position={[0.07, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.009, 0.009, 0.025, 8]} />
          <MetalMaterial color="#c4a86a" roughness={0.25} metalness={0.8} />
        </mesh>
      </group>
    );
  }

  function DeskClutter() {
    return (
      <group raycast={() => null}>
        <group position={[0.55, 1.034, 0.92]} rotation={[0, 0.12, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.22, 0.004, 0.28]} />
            <PaperMaterial color="#d8d0c0" />
          </mesh>
          <mesh position={[0.03, 0.004, 0.04]} rotation={[0, -0.08, 0.02]} castShadow>
            <boxGeometry args={[0.18, 0.003, 0.22]} />
            <PaperMaterial color="#ece6d8" />
          </mesh>
        </group>
      </group>
    );
  }

  function DeskDrawerFaces() {
    const drawers = [
      [-1.42, 0.78, 1.62],
      [-1.42, 0.64, 1.62],
      [1.42, 0.78, 1.62],
      [1.42, 0.64, 1.62],
    ];

    return (
      <group raycast={() => null}>
        {drawers.map(([x, y, z], i) => (
          <group key={`drawer-${i}`} position={[x, y, z]}>
            <mesh castShadow>
              <boxGeometry args={[0.72, 0.22, 0.04]} />
              <WoodMaterial color="#3a2114" roughness={0.48} />
            </mesh>
            <mesh position={[0, 0, 0.028]}>
              <boxGeometry args={[0.58, 0.008, 0.012]} />
              <MetalMaterial color="#7a6848" roughness={0.3} metalness={0.75} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  function MarkerTray({ position = [0, -0.48, 0.07] }) {
    const markers = ["#d83e37", "#2c65b9", "#111111", "#2e8b45"];
    return (
      <group position={position} raycast={() => null}>
        <mesh>
          <boxGeometry args={[1.35, 0.035, 0.06]} />
          <MetalMaterial color="#4a4a4a" roughness={0.35} metalness={0.6} />
        </mesh>
        {markers.map((color, i) => (
          <mesh key={color} position={[-0.42 + i * 0.28, 0.03, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.012, 0.012, 0.09, 8]} />
            <PlasticMaterial color={color} roughness={0.45} />
          </mesh>
        ))}
        <mesh position={[0.52, 0.028, 0]}>
          <boxGeometry args={[0.08, 0.04, 0.04]} />
          <meshStandardMaterial color="#d8d0c4" roughness={0.75} />
        </mesh>
      </group>
    );
  }
  
  function useTeamLogoTexture(teamLogo) {
    const [texture, setTexture] = useState(null);
    const [loadFailed, setLoadFailed] = useState(false);
    const logoUrl = toLogoUrl(teamLogo);

    useEffect(() => {
      if (!logoUrl) {
        setTexture(null);
        setLoadFailed(false);
        return undefined;
      }

      let active = true;
      setLoadFailed(false);

      const loader = new THREE.TextureLoader();
      loader.load(
        logoUrl,
        (tex) => {
          if (!active) {
            tex.dispose();
            return;
          }
          tex.colorSpace = THREE.SRGBColorSpace;
          setTexture(tex);
        },
        undefined,
        () => {
          if (active) {
            setTexture(null);
            setLoadFailed(true);
          }
        }
      );

      return () => {
        active = false;
      };
    }, [logoUrl]);

    useEffect(() => {
      return () => {
        texture?.dispose();
      };
    }, [texture]);

    return { texture, loadFailed };
  }

  function TeamLogoPlane({
    teamLogo,
    teamName,
    hovered = false,
    width = 1.15,
    height = 1.15,
    opacity = 1,
    circularFallback = true,
  }) {
    const { texture, loadFailed } = useTeamLogoTexture(teamLogo);
    const initials = initialsFromTeam(teamName);
    const showTexture = texture && !loadFailed;

    if (showTexture) {
      return (
        <mesh raycast={() => null}>
          <planeGeometry args={[width, height]} />
          <meshBasicMaterial
            map={texture}
            transparent
            opacity={opacity}
            toneMapped={false}
            depthWrite={opacity >= 0.95}
          />
        </mesh>
      );
    }

    const fallbackRadius = Math.min(width, height) * 0.5;

    return (
      <group>
        {circularFallback ? (
          <mesh raycast={() => null}>
            <circleGeometry args={[fallbackRadius, 64]} />
            <meshStandardMaterial
              color={hovered ? "#f3d78a" : "#1c2533"}
              roughness={0.4}
              metalness={0.16}
              transparent={opacity < 1}
              opacity={opacity}
            />
          </mesh>
        ) : (
          <mesh raycast={() => null}>
            <planeGeometry args={[width, height]} />
            <meshStandardMaterial
              color={hovered ? "#2a3548" : "#121820"}
              roughness={0.45}
              metalness={0.12}
              transparent={opacity < 1}
              opacity={opacity}
            />
          </mesh>
        )}

        <WallText
          position={[0, 0, 0.02]}
          size={Math.min(width, height) * 0.19}
          color="#fff4d8"
          maxWidth={width * 0.9}
        >
          {initials}
        </WallText>
      </group>
    );
  }

  /** Flat logo decal for desk mat, laptop, binders */
  function TeamLogoDecal({
    teamLogo,
    teamName,
    position = [0, 0, 0],
    rotation = [0, 0, 0],
    width = 0.35,
    height = 0.35,
    opacity = 0.22,
    hovered = false,
  }) {
    return (
      <group position={position} rotation={rotation}>
        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={width}
          height={height}
          opacity={opacity}
          circularFallback={false}
        />
      </group>
    );
  }
  
  function LaptopObject({ hovered, teamName, teamLogo }) {
    const screenRef = useRef(null);
    const pulseRef = useRef(null);

    const commandGroups = [
      {
        label: "TEAM OPERATIONS",
        color: "#5a8aaa",
        items: [
          ["OVERVIEW", "Franchise pulse"],
          ["ROSTER", "Active roster"],
          ["LINES", "Matchups"],
        ],
      },
      {
        label: "FRONT OFFICE",
        color: "#8a7348",
        items: [
          ["SCOUTING", "Draft intel"],
          ["CONTRACTS", "Cap ledger"],
          ["TRADE DESK", "Active calls"],
        ],
      },
      {
        label: "LEAGUE / ANALYTICS",
        color: "#4a7a8a",
        items: [
          ["CALENDAR", "Schedule"],
          ["ANALYTICS", "Advanced stats"],
          ["HEADLINES", "League news"],
        ],
      },
    ];

    const statusStrip = [
      ["CAP ROOM", "AVAILABLE"],
      ["INJURY", "WATCH"],
      ["OWNER", "GOALS"],
      ["ALERTS", "LIVE"],
    ];

    useFrame((state) => {
      const t = state.clock.elapsedTime;

      if (screenRef.current?.material) {
        screenRef.current.material.emissiveIntensity = hovered
          ? 0.44 + Math.sin(t * 1.8) * 0.04
          : 0.28 + Math.sin(t * 1.1) * 0.02;
      }

      if (pulseRef.current?.material) {
        pulseRef.current.material.opacity = hovered
          ? 0.14 + Math.sin(t * 2.2) * 0.04
          : 0.08 + Math.sin(t * 1.4) * 0.02;
      }
    });

    return (
      <group>
        <RoundedBox
          position={[0, 0.02, 0.02]}
          args={[1.92, 0.08, 0.88]}
          radius={0.04}
          smoothness={8}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <MetalMaterial color="#0a0c10" roughness={0.38} metalness={0.62} />
        </RoundedBox>

        <mesh position={[0, 0.058, 0.02]} raycast={() => null}>
          <boxGeometry args={[1.78, 0.006, 0.78]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.goldDim}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={hovered ? 0.14 : 0.06}
            roughness={0.35}
            metalness={0.55}
          />
        </mesh>

        <group position={[0, 0.48, -0.18]} rotation={[-0.22, 0, 0]}>
          <RoundedBox
            args={[1.68, 1.02, 0.07]}
            radius={0.04}
            smoothness={10}
            castShadow
            receiveShadow
            raycast={() => null}
          >
            <meshPhysicalMaterial
              color="#050810"
              emissive={hovered ? "#0e2840" : "#061420"}
              emissiveIntensity={hovered ? 0.22 : 0.1}
              roughness={0.32}
              metalness={0.45}
              clearcoat={0.7}
              clearcoatRoughness={0.15}
            />
            <Edges color={hovered ? "#5a8aaa" : "#1a3040"} />
          </RoundedBox>

          <RoundedBox
            ref={screenRef}
            position={[0, 0, 0.042]}
            args={[1.54, 0.88, 0.014]}
            radius={0.028}
            smoothness={8}
            raycast={() => null}
          >
            <ScreenGlassMaterial hovered={hovered} />
          </RoundedBox>

          <mesh ref={pulseRef} position={[0, 0, 0.036]} raycast={() => null}>
            <planeGeometry args={[1.62, 0.94]} />
            <meshBasicMaterial color="#4a9ac8" transparent opacity={0.08} depthWrite={false} />
          </mesh>

          <RoundedBox
            position={[0, 0.4, 0.048]}
            args={[1.44, 0.05, 0.01]}
            radius={0.01}
            smoothness={4}
            raycast={() => null}
          >
            <meshStandardMaterial
              color="#0c1824"
              emissive="#1a3048"
              emissiveIntensity={0.2}
              roughness={0.45}
            />
          </RoundedBox>

          <WallText position={[-0.66, 0.402, 0.056]} size={0.02} color="#6a8a9a" anchorX="left">
            FRANCHISE COMMAND
          </WallText>

          <WallText position={[0.58, 0.402, 0.056]} size={0.016} color={hovered ? "#7eb896" : "#5a8a6a"}>
            LIVE
          </WallText>

          <TeamLogoDecal
            teamLogo={teamLogo}
            teamName={teamName}
            position={[-0.58, 0.28, 0.055]}
            width={0.14}
            height={0.14}
            opacity={0.92}
            hovered={hovered}
          />

          <WallText
            position={[-0.42, 0.29, 0.058]}
            size={0.038}
            color="#e8f0f4"
            anchorX="left"
          >
            {teamName}
          </WallText>

          <WallText
            position={[-0.42, 0.24, 0.058]}
            size={0.018}
            color="#6a8a9a"
            anchorX="left"
          >
            EXECUTIVE DASHBOARD
          </WallText>

          {commandGroups.map((group, gi) => {
            const x = -0.48 + gi * 0.48;
            return (
              <group key={group.label} position={[x, -0.02, 0.056]}>
                <WallText position={[0, 0.14, 0]} size={0.014} color={group.color}>
                  {group.label}
                </WallText>
                {group.items.map(([label, sub], ii) => {
                  const y = 0.06 - ii * 0.1;
                  return (
                    <group key={label} position={[0, y, 0]}>
                      <RoundedBox
                        args={[0.42, 0.072, 0.008]}
                        radius={0.008}
                        smoothness={3}
                        raycast={() => null}
                      >
                        <meshStandardMaterial
                          color={hovered ? "#0c1c2a" : "#081420"}
                          emissive={hovered ? "#1a4060" : "#0c2438"}
                          emissiveIntensity={hovered ? 0.28 : 0.14}
                          roughness={0.48}
                        />
                      </RoundedBox>
                      <WallText position={[-0.16, 0.012, 0.008]} size={0.016} color="#d8e8f0" anchorX="left">
                        {label}
                      </WallText>
                      <WallText position={[-0.16, -0.018, 0.008]} size={0.011} color="#5a7a8a" anchorX="left">
                        {sub}
                      </WallText>
                    </group>
                  );
                })}
              </group>
            );
          })}

          {statusStrip.map(([top, bottom], index) => {
            const x = -0.54 + index * 0.36;
            return (
              <group key={top} position={[x, -0.38, 0.056]}>
                <RoundedBox args={[0.3, 0.05, 0.008]} radius={0.006} smoothness={3} raycast={() => null}>
                  <meshStandardMaterial
                    color="#060c14"
                    emissive="#102838"
                    emissiveIntensity={hovered ? 0.2 : 0.1}
                    roughness={0.5}
                  />
                </RoundedBox>
                <WallText position={[0, 0.01, 0.008]} size={0.012} color="#8aaaba">
                  {top}
                </WallText>
                <WallText position={[0, -0.012, 0.008]} size={0.01} color="#5a7a88">
                  {bottom}
                </WallText>
              </group>
            );
          })}

          <mesh position={[0.68, 0.26, 0.056]} raycast={() => null}>
            <circleGeometry args={[0.018, 20]} />
            <meshBasicMaterial color={hovered ? "#c94a44" : "#6a2824"} />
          </mesh>
          <WallText position={[0.68, 0.22, 0.058]} size={0.01} color="#a86a64">
            ALERT
          </WallText>
        </group>

        <pointLight
          position={[0, 0.35, 0.1]}
          intensity={hovered ? 0.42 : 0.22}
          color="#4a8ab8"
          distance={1.6}
        />
      </group>
    );
  }
  
  function PhoneObject({
    hovered,
    unreadMessages,
    hasTradeActivity = false,
    callerLabel = "LEAGUE GM",
  }) {
    const badgeText = Number(unreadMessages || 0) > 0 ? String(unreadMessages) : "";
    const notify = hasTradeActivity || Number(unreadMessages || 0) > 0;
  
    return (
      <group rotation={[0, -0.18, 0]}>
        <RoundedBox args={[0.58, 0.12, 0.9]} radius={0.06} smoothness={7}>
          <GlowMaterial
            color={hovered ? "#252a32" : "#11141a"}
            emissive={hovered ? "#4b87aa" : "#000000"}
            intensity={0.28}
            roughness={0.5}
          />
        </RoundedBox>
  
        <mesh position={[0, 0.073, -0.22]}>
          <boxGeometry args={[0.42, 0.012, 0.22]} />
          <meshStandardMaterial
            color="#071018"
            emissive="#5cc8ff"
            emissiveIntensity={hovered ? 0.55 : 0.25}
          />
        </mesh>
  
        <WallText
          position={[0, 0.088, -0.22]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.045}
          color="#daf5ff"
        >
          TRADE CALLS
        </WallText>

        <WallText
          position={[0, 0.09, -0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.022}
          color="#9fdfff"
        >
          {safeText(callerLabel, "LEAGUE GM")}
        </WallText>

        <BlinkingNotificationLight active={notify} position={[-0.24, 0.1, -0.42]} />
  
        {[0, 1, 2].map((row) =>
          [0, 1, 2].map((col) => (
            <mesh
              key={`${row}-${col}`}
              position={[-0.16 + col * 0.16, 0.083, 0.09 + row * 0.12]}
            >
              <cylinderGeometry args={[0.025, 0.025, 0.014, 18]} />
              <meshStandardMaterial color="#343b45" roughness={0.4} />
            </mesh>
          ))
        )}
  
        {badgeText ? (
          <group position={[0.26, 0.18, -0.42]}>
            <mesh>
              <sphereGeometry args={[0.09, 22, 22]} />
              <meshStandardMaterial
                color="#d94a41"
                emissive="#b72a20"
                emissiveIntensity={0.4}
              />
            </mesh>
  
            <WallText position={[0, 0, 0.095]} size={0.08} color="#ffffff">
              {badgeText}
            </WallText>
          </group>
        ) : null}
      </group>
    );
  }
  
  function ScoutingKitObject({ hovered, teamLogo, teamName, draftWeek = false }) {
    return (
      <group rotation={[0, 0.16, 0]}>
        <RoundedBox args={[0.92, 0.08, 0.66]} radius={0.035} smoothness={5}>
          <GlowMaterial
            color={hovered ? "#d9b15a" : "#8e6b37"}
            emissive={hovered ? "#b58222" : "#000000"}
            intensity={hovered ? 0.18 : 0}
            roughness={0.72}
          />
        </RoundedBox>
  
        <mesh position={[-0.22, 0.038, -0.28]}>
          <boxGeometry args={[0.45, 0.035, 0.16]} />
          <meshStandardMaterial color="#e5bd69" roughness={0.65} />
        </mesh>
  
        <mesh position={[0.08, 0.062, -0.02]}>
          <boxGeometry args={[0.72, 0.018, 0.42]} />
          <meshStandardMaterial color="#efe5ce" roughness={0.8} />
        </mesh>
  
        <WallText
          position={[0.02, 0.08, -0.02]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.06}
          color="#1c1710"
        >
          SCOUTING
        </WallText>

        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[0.28, 0.082, -0.22]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.12}
          height={0.12}
          opacity={0.72}
          hovered={hovered}
        />
  
        <group position={[-0.23, 0.11, 0.18]} rotation={[-Math.PI / 2, 0, 0]}>
          <mesh>
            <cylinderGeometry args={[0.07, 0.07, 0.12, 24]} />
            <meshStandardMaterial color="#111820" roughness={0.36} />
          </mesh>
  
          <mesh position={[0.14, 0, 0]}>
            <cylinderGeometry args={[0.07, 0.07, 0.12, 24]} />
            <meshStandardMaterial color="#111820" roughness={0.36} />
          </mesh>
  
          <mesh position={[0.07, 0, 0]}>
            <boxGeometry args={[0.08, 0.035, 0.035]} />
            <meshStandardMaterial color="#303945" roughness={0.45} />
          </mesh>
        </group>
  
        <WallText
          position={[0.17, 0.087, 0.18]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.028}
          color="#49351c"
        >
          INTL TRIP NOTES
        </WallText>

        {[
          ["TIER 1", -0.18, -0.02, "#c9a86a"],
          ["RISERS", 0.02, 0.02, "#7eb896"],
          ["WATCH", 0.2, -0.04, "#7eb8d4"],
        ].map(([label, x, z, color]) => (
          <group key={label} position={[x, 0.084, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <mesh raycast={() => null}>
              <boxGeometry args={[0.14, 0.09, 0.004]} />
              <meshStandardMaterial color="#efe5ce" roughness={0.82} />
            </mesh>
            <WallText position={[0, 0.004, 0.004]} size={0.018} color={color}>
              {label}
            </WallText>
          </group>
        ))}

        {draftWeek ? (
          <mesh position={[0, 0.095, 0]} raycast={() => null}>
            <planeGeometry args={[0.72, 0.42]} />
            <meshBasicMaterial color="#c9a86a" transparent opacity={0.08} depthWrite={false} />
          </mesh>
        ) : null}
      </group>
    );
  }
  
  function ContractLedgerObject({ hovered, capSpace, capPressure = false }) {
    return (
      <group rotation={[0, -0.08, 0]}>
        <RoundedBox args={[0.98, 0.075, 0.7]} radius={0.03} smoothness={6}>
          <GlowMaterial
            color={hovered ? "#8ed0ad" : "#355c50"}
            emissive={hovered ? "#5db88b" : "#000000"}
            intensity={hovered ? 0.18 : 0}
            roughness={0.68}
          />
        </RoundedBox>
  
        <mesh position={[0.04, 0.06, 0]}>
          <boxGeometry args={[0.72, 0.02, 0.5]} />
          <meshStandardMaterial color="#f2ead6" roughness={0.82} />
        </mesh>
  
        <WallText
          position={[0.04, 0.083, -0.18]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.048}
          color="#172820"
        >
          CAP LEDGER
        </WallText>
  
        <WallText
          position={[0.04, 0.086, -0.04]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.035}
          color="#375244"
        >
          ROOM: {formatMoney(capSpace)}
        </WallText>
  
        {[0, 1, 2].map((i) => (
          <mesh key={i} position={[0.04, 0.087, 0.08 + i * 0.09]}>
            <boxGeometry args={[0.48, 0.005, 0.012]} />
            <meshStandardMaterial color="#77a98c" roughness={0.7} />
          </mesh>
        ))}

        {[
          ["RFA", -0.18, 0.1],
          ["UFA", 0.02, 0.14],
          ["NMC", 0.2, 0.08],
          ["CAP", -0.05, 0.2],
        ].map(([label, x, z]) => (
          <WallText
            key={label}
            position={[0.04 + x, 0.092, z]}
            rotation={[-Math.PI / 2, 0, 0]}
            size={0.022}
            color="#2d4a3d"
          >
            {label}
          </WallText>
        ))}

        <mesh position={[0.22, 0.09, 0.24]} rotation={[-Math.PI / 2, 0, 0.12]} raycast={() => null}>
          <boxGeometry args={[0.18, 0.12, 0.004]} />
          <PaperMaterial color="#f7f1df" />
        </mesh>

        <mesh position={[0.22, 0.091, 0.3]} rotation={[-Math.PI / 2, 0, 0.12]} raycast={() => null}>
          <boxGeometry args={[0.1, 0.004, 0.004]} />
          <meshStandardMaterial color="#4a4034" roughness={0.7} />
        </mesh>

        {capPressure ? (
          <WallText
            position={[0.22, 0.095, 0.24]}
            rotation={[-Math.PI / 2, 0, 0.12]}
            size={0.028}
            color="#b72a20"
          >
            CAP WARN
          </WallText>
        ) : null}
  
        <group position={[-0.33, 0.105, 0.2]} rotation={[0, 0, 0.65]}>
          <mesh>
            <cylinderGeometry args={[0.018, 0.018, 0.42, 16]} />
            <meshStandardMaterial color="#111111" roughness={0.35} />
          </mesh>
  
          <mesh position={[0, 0.23, 0]}>
            <cylinderGeometry args={[0.015, 0.015, 0.06, 16]} />
            <meshStandardMaterial color="#d0a24a" metalness={0.4} roughness={0.25} />
          </mesh>
        </group>
      </group>
    );
  }
  
  function TabletObject({ hovered }) {
    return (
      <group rotation={[0, 0.24, 0]}>
        <RoundedBox args={[0.72, 0.055, 0.92]} radius={0.055} smoothness={7}>
          <GlowMaterial
            color="#090d12"
            emissive={hovered ? "#67c9ff" : "#1e6487"}
            intensity={hovered ? 0.55 : 0.25}
            roughness={0.42}
          />
        </RoundedBox>
  
        <mesh position={[0, 0.035, 0]}>
          <boxGeometry args={[0.58, 0.012, 0.72]} />
          <meshStandardMaterial
            color="#071725"
            emissive="#184d6a"
            emissiveIntensity={0.35}
          />
        </mesh>
  
        <WallText
          position={[0, 0.05, -0.25]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.055}
          color="#d9f8ff"
        >
          STATS
        </WallText>
  
        <WallText
          position={[0, 0.052, -0.03]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.035}
          color="#95e8ff"
        >
          CF% xGF% PDO
        </WallText>
  
        {[0, 1, 2, 3].map((i) => (
          <mesh
            key={i}
            position={[-0.22 + i * 0.15, 0.055, 0.24]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <planeGeometry args={[0.08, 0.08 + i * 0.03]} />
            <meshBasicMaterial
              color="#64d6ff"
              transparent
              opacity={hovered ? 0.32 : 0.18}
            />
          </mesh>
        ))}
      </group>
    );
  }
  
  function CalendarObject({ hovered, currentDate, nextGame, teamLogo, teamName }) {
    const calendarDays = [
      ["", "", "1", "2", "3", "4", "5"],
      ["6", "7", "8", "9", "10", "11", "12"],
      ["13", "14", "15", "16", "17", "18", "19"],
      ["20", "21", "22", "23", "24", "25", "26"],
      ["27", "28", "29", "30", "31", "", ""],
    ];
  
    const markedDays = {
      "6": "home",
      "11": "away",
      "14": "meeting",
      "20": "travel",
      "25": "deadline",
    };
  
    return (
      <group rotation={[0, -0.22, 0]}>
        <mesh position={[0.035, -0.018, 0.035]} rotation={[0, 0, -0.015]}>
          <boxGeometry args={[0.82, 0.035, 0.98]} />
          <meshStandardMaterial color="#cfc3aa" roughness={0.88} />
        </mesh>
  
        <RoundedBox args={[0.88, 0.06, 1.02]} radius={0.035} smoothness={6}>
          <meshStandardMaterial
            color={hovered ? "#fff3d4" : "#eee2c4"}
            roughness={0.78}
            metalness={0.03}
            emissive={hovered ? "#d9a441" : "#000000"}
            emissiveIntensity={hovered ? 0.08 : 0}
          />
        </RoundedBox>
  
        <mesh position={[0, 0.047, -0.43]}>
          <boxGeometry args={[0.88, 0.028, 0.16]} />
          <meshStandardMaterial
            color={hovered ? "#d94338" : "#a92f2b"}
            roughness={0.5}
            metalness={0.08}
            emissive={hovered ? "#5c0e0b" : "#000000"}
            emissiveIntensity={hovered ? 0.2 : 0}
          />
        </mesh>
  
        {[-0.32, -0.16, 0, 0.16, 0.32].map((x) => (
          <mesh key={x} position={[x, 0.071, -0.5]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.035, 0.006, 8, 22]} />
            <meshStandardMaterial color="#d8d8d8" metalness={0.55} roughness={0.32} />
          </mesh>
        ))}
  
        <WallText
          position={[0, 0.078, -0.43]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.048}
          color="#ffffff"
        >
          SEASON CALENDAR
        </WallText>

        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[0.3, 0.076, -0.38]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.11}
          height={0.11}
          opacity={0.8}
          hovered={hovered}
        />
  
        <WallText
          position={[-0.27, 0.075, -0.27]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.039}
          color="#1d1a16"
          anchorX="left"
        >
          OCTOBER
        </WallText>
  
        <WallText
          position={[0.27, 0.075, -0.27]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.028}
          color="#69513b"
        >
          {safeText(currentDate, "Today")}
        </WallText>
  
        {["S", "M", "T", "W", "T", "F", "S"].map((day, i) => (
          <WallText
            key={`${day}-${i}`}
            position={[-0.33 + i * 0.11, 0.078, -0.16]}
            rotation={[-Math.PI / 2, 0, 0]}
            size={0.027}
            color="#8d332b"
          >
            {day}
          </WallText>
        ))}
  
        {[0, 1, 2, 3, 4, 5].map((row) => (
          <mesh key={`row-${row}`} position={[0, 0.066, -0.1 + row * 0.095]}>
            <boxGeometry args={[0.74, 0.004, 0.006]} />
            <meshStandardMaterial color="#c9bda5" roughness={0.85} />
          </mesh>
        ))}
  
        {[0, 1, 2, 3, 4, 5, 6, 7].map((col) => (
          <mesh
            key={`col-${col}`}
            position={[-0.385 + col * 0.11, 0.066, 0.095]}
          >
            <boxGeometry args={[0.004, 0.004, 0.47]} />
            <meshStandardMaterial color="#c9bda5" roughness={0.85} />
          </mesh>
        ))}
  
        {calendarDays.map((week, row) =>
          week.map((day, col) => {
            if (!day) return null;
  
            const eventType = markedDays[day];
  
            const eventColor =
              eventType === "home"
                ? "#2f79d8"
                : eventType === "away"
                ? "#d83d32"
                : eventType === "meeting"
                ? "#d6a02e"
                : eventType === "travel"
                ? "#4a9d68"
                : eventType === "deadline"
                ? "#8e4ee6"
                : null;
  
            return (
              <group
                key={`${row}-${col}-${day}`}
                position={[-0.33 + col * 0.11, 0.079, -0.06 + row * 0.095]}
              >
                <WallText
                  position={[0, 0, 0]}
                  rotation={[-Math.PI / 2, 0, 0]}
                  size={0.028}
                  color={eventType ? "#15110d" : "#4b4034"}
                >
                  {day}
                </WallText>
  
                {eventType ? (
                  <mesh
                    position={[0.032, 0.002, 0.026]}
                    rotation={[-Math.PI / 2, 0, 0]}
                  >
                    <circleGeometry args={[0.014, 18]} />
                    <meshStandardMaterial
                      color={eventColor}
                      emissive={eventColor}
                      emissiveIntensity={hovered ? 0.32 : 0.12}
                      roughness={0.45}
                    />
                  </mesh>
                ) : null}
              </group>
            );
          })
        )}
  
        <group
          position={[0.29, 0.09, 0.34]}
          rotation={[-Math.PI / 2, 0, -0.08]}
        >
          <mesh>
            <planeGeometry args={[0.28, 0.2]} />
            <meshStandardMaterial
              color={hovered ? "#ffe985" : "#f5d86d"}
              roughness={0.75}
              side={THREE.DoubleSide}
            />
          </mesh>
  
          <WallText position={[0, 0.035, 0.004]} size={0.021} color="#241b0f">
            NEXT GAME
          </WallText>
  
          <WallText
            position={[0, -0.025, 0.004]}
            size={0.017}
            color="#39291a"
            maxWidth={0.24}
          >
            {safeText(nextGame, "No game listed")}
          </WallText>
        </group>
      </group>
    );
  }

  function NewspaperObject({ hovered, activeStorylines }) {
    return (
      <group rotation={[0, -0.36, 0]}>
        {[0, 1, 2].map((i) => (
          <mesh key={i} position={[i * 0.025, i * 0.02, i * -0.025]} raycast={() => null}>
            <boxGeometry args={[0.82, 0.025, 0.56]} />
            <meshStandardMaterial
              color={hovered ? "#f1ead9" : "#d7cfbf"}
              roughness={0.88}
            />
          </mesh>
        ))}
  
        <WallText
          position={[0.04, 0.085, -0.16]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.055}
          color="#151515"
        >
          LEAGUE DAILY
        </WallText>
  
        <WallText
          position={[0.04, 0.087, 0.1]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.04}
          color="#303030"
        >
          {Number(activeStorylines || 0)} active storylines
        </WallText>
      </group>
    );
  }
  
  function ClipboardObject({ hovered, pendingTasks }) {
    return (
      <group rotation={[0, 0.5, 0]}>
        <RoundedBox args={[0.58, 0.055, 0.76]} radius={0.025} smoothness={5} raycast={() => null}>
          <GlowMaterial color={hovered ? "#fff6d7" : "#e7dcbd"} roughness={0.8} />
        </RoundedBox>
  
        <mesh position={[0, 0.055, -0.31]} raycast={() => null}>
          <boxGeometry args={[0.32, 0.04, 0.08]} />
          <meshStandardMaterial color="#20252d" roughness={0.5} metalness={0.16} />
        </mesh>
  
        <WallText
          position={[0, 0.076, -0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.043}
          color="#1b1b1b"
        >
          DECISIONS
        </WallText>
  
        {[0, 1, 2].map((i) => (
          <group key={i} position={[-0.16, 0.077, 0.07 + i * 0.12]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
              <circleGeometry args={[0.018, 18]} />
              <meshStandardMaterial color={i === 0 ? "#d9473e" : "#7a8a77"} />
            </mesh>
  
            <mesh position={[0.16, 0, 0]} raycast={() => null}>
              <boxGeometry args={[0.24, 0.004, 0.014]} />
              <meshStandardMaterial color="#6a5b45" roughness={0.8} />
            </mesh>
          </group>
        ))}
  
        <WallText
          position={[0, 0.079, 0.3]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.033}
          color="#3a3327"
        >
          {Number(pendingTasks || 0)} pending
        </WallText>
      </group>
    );
  }
  
  function CoffeeAndPuck() {
    return (
      <group position={[1.58, 1.058, 0.22]} raycast={() => null}>
        <mesh castShadow receiveShadow>
          <cylinderGeometry args={[0.13, 0.13, 0.045, 32]} />
          <meshStandardMaterial color="#060606" roughness={0.38} metalness={0.12} />
        </mesh>
        <mesh position={[0, 0.028, 0]}>
          <torusGeometry args={[0.118, 0.005, 8, 32]} />
          <meshStandardMaterial color="#1a1a1a" roughness={0.32} metalness={0.18} />
        </mesh>
      </group>
    );
  }

  function Desk({ children, teamName, teamLogo }) {
    return (
      <group>
        {[[-1.55, 0.36, 1.05], [1.55, 0.36, 1.05], [-1.55, 0.36, 0.48], [1.55, 0.36, 0.48]].map(
          ([x, y, z], i) => (
            <mesh key={`leg-${i}`} position={[x, y, z]} castShadow receiveShadow raycast={() => null}>
              <boxGeometry args={[0.1, 0.72, 0.1]} />
              <MetalMaterial color="#14161c" roughness={0.42} metalness={0.78} />
            </mesh>
          )
        )}

        <RoundedBox
          position={[0, 0.72, 0.92]}
          args={[4.2, 0.32, 1.42]}
          radius={0.05}
          smoothness={8}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <WoodMaterial color={OFFICE_PALETTE.walnut} roughness={0.52} />
        </RoundedBox>

        <mesh position={[0, 0.84, 1.68]} castShadow raycast={() => null}>
          <boxGeometry args={[3.9, 0.22, 0.05]} />
          <WoodMaterial color="#18120e" roughness={0.54} />
        </mesh>

        <DeskDrawerFaces />

        <RoundedBox
          position={[0, 0.94, 0.92]}
          args={[4.38, 0.08, 1.55]}
          radius={0.04}
          smoothness={8}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <WoodMaterial color="#241810" roughness={0.42} metalness={0.06} />
        </RoundedBox>

        {[-2.12, 2.12].map((x) => (
          <mesh key={`edge-${x}`} position={[x, 0.978, 0.92]} raycast={() => null}>
            <boxGeometry args={[0.008, 0.014, 1.48]} />
            <meshStandardMaterial color={OFFICE_PALETTE.goldDim} roughness={0.22} metalness={0.72} />
          </mesh>
        ))}

        <mesh position={[0, 0.976, 0.92]} receiveShadow raycast={() => null}>
          <boxGeometry args={[1.85, 0.01, 0.88]} />
          <LeatherMaterial color="#121010" roughness={0.82} />
        </mesh>

        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[0, 0.982, 0.92]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.28}
          height={0.28}
          opacity={0.08}
        />

        <RoundedBox
          position={[0, 0.87, 1.72]}
          args={[0.95, 0.1, 0.05]}
          radius={0.015}
          smoothness={5}
          raycast={() => null}
        >
          <MetalMaterial color="#0a0c10" roughness={0.38} metalness={0.55} />
        </RoundedBox>

        <DeskPen position={[1.12, 0.982, 0.78]} />
        <DeskClutter />

        {children}
      </group>
    );
  }
  
  function getBestPlayer(players = []) {
    const rating = (player) =>
      Number(
        player?.overall ||
          player?.ovr ||
          player?.rating ||
          player?.trueOverall ||
          player?.true_ovr ||
          player?.calculatedOverall ||
          0
      );

    return [...players].sort((a, b) => rating(b) - rating(a))[0];
  }

  function getPlayerHeadshot(player) {
    return player?.headshot || player?.headshotUrl || player?.image || player?.portrait || player?.faceUrl || player?.picture || "";
  }

  function getPlayerName(player) {
    return (
      player?.name ||
      player?.full_name ||
      `${player?.first_name || player?.firstName || ""} ${player?.last_name || player?.lastName || ""}`.trim() ||
      "Franchise Player"
    );
  }

  function WallPlayerPortrait({ player, imageUrl }) {
    const resolvedPlayer = useMemo(() => ensurePlayerHeadshotFields(player || {}), [player]);

    const pictureTexture = useMemo(() => {
      if (!imageUrl) return null;
      const loader = new THREE.TextureLoader();
      const tex = loader.load(imageUrl);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    }, [imageUrl]);

    useEffect(() => {
      return () => {
        pictureTexture?.dispose();
      };
    }, [pictureTexture]);

    if (pictureTexture) {
      return (
        <mesh position={[0, 0, 0.045]} raycast={() => null}>
          <boxGeometry args={[2.02, 1.02, 0.02]} />
          <meshBasicMaterial map={pictureTexture} toneMapped={false} />
        </mesh>
      );
    }

    if (!player) {
      return (
        <mesh position={[0, 0, 0.045]} raycast={() => null}>
          <boxGeometry args={[2.02, 1.02, 0.02]} />
          <meshStandardMaterial
            color="#142638"
            emissive="#102943"
            emissiveIntensity={0.1}
            roughness={0.55}
          />
        </mesh>
      );
    }

    return (
      <>
        <mesh position={[0, 0, 0.045]} raycast={() => null}>
          <boxGeometry args={[2.02, 1.02, 0.02]} />
          <meshStandardMaterial color="#142638" roughness={0.55} />
        </mesh>
        <Html transform position={[0, 0.02, 0.06]} scale={0.42} center style={{ pointerEvents: "none" }}>
          <div className="office-wall-portrait">
            <PlayerHeadshot player={resolvedPlayer} size="xl" variant="card" />
          </div>
        </Html>
      </>
    );
  }

  function RoomShell() {
    return (
      <group>
        <mesh position={[0, 0, -1.1]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[9, 8]} />
          <meshStandardMaterial color="#1e1a16" roughness={0.72} metalness={0.04} />
        </mesh>

        <mesh position={[0, 0.008, -1.1]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[8.6, 7.6]} />
          <meshStandardMaterial color="#26221e" roughness={0.68} />
        </mesh>

        <FloorPlanks />
        <OfficeRug />
        <Baseboards />
  
        <mesh position={[0, 2.1, -3.55]} receiveShadow castShadow raycast={() => null}>
          <boxGeometry args={[8.8, 4.2, 0.12]} />
          <meshStandardMaterial color={OFFICE_PALETTE.wall} roughness={0.82} metalness={0.06} />
        </mesh>

        <WallPanelStrips />
  
        <mesh
          position={[-4.43, 2.1, -0.15]}
          rotation={[0, Math.PI / 2, 0]}
          receiveShadow
          raycast={() => null}
        >
          <boxGeometry args={[6.9, 4.2, 0.12]} />
          <meshStandardMaterial color="#222830" roughness={0.84} metalness={0.05} />
        </mesh>
  
        <mesh
          position={[4.43, 2.1, -0.15]}
          rotation={[0, Math.PI / 2, 0]}
          receiveShadow
          raycast={() => null}
        >
          <boxGeometry args={[6.9, 4.2, 0.12]} />
          <meshStandardMaterial color="#20242c" roughness={0.84} metalness={0.05} />
        </mesh>
  
        <mesh position={[0, 4.15, -0.2]} rotation={[Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[8.8, 7]} />
          <meshStandardMaterial color="#1a1c24" roughness={0.9} />
        </mesh>

        <mesh position={[0, 4.08, -0.15]} raycast={() => null}>
          <boxGeometry args={[8.4, 0.05, 0.08]} />
          <MetalMaterial color="#1a1c22" roughness={0.38} metalness={0.72} />
        </mesh>

        <CeilingLightStrip />

        {[-4.38, 4.38].map((x) => (
          <group key={`side-trim-${x}`} position={[x, 2.05, -0.15]}>
            <mesh position={[0, 0, 0.08]} raycast={() => null}>
              <boxGeometry args={[0.04, 3.6, 0.02]} />
              <meshStandardMaterial
                color={OFFICE_PALETTE.goldDim}
                emissive={OFFICE_PALETTE.goldDim}
                emissiveIntensity={0.14}
                roughness={0.45}
                metalness={0.42}
              />
            </mesh>
          </group>
        ))}

        <mesh position={[0, 2.85, -3.42]} raycast={() => null}>
          <planeGeometry args={[1.4, 0.22]} />
          <meshBasicMaterial color="#c9a86a" transparent opacity={0.03} depthWrite={false} />
        </mesh>
      </group>
    );
  }
  
  function WallLogo({ hovered, teamLogo, teamName, scale = 1.18 }) {
    return (
      <group scale={[scale, scale, scale]}>
        <mesh position={[0, 0, -0.04]} castShadow raycast={() => null}>
          <boxGeometry args={[1.95, 1.95, 0.06]} />
          <WoodMaterial color="#1a1612" roughness={0.58} />
        </mesh>

        <mesh position={[0, 0, -0.018]} raycast={() => null}>
          <boxGeometry args={[1.72, 1.72, 0.04]} />
          <MetalMaterial color="#2a2620" roughness={0.45} metalness={0.42} />
        </mesh>

        <mesh position={[0, 0, -0.015]} raycast={() => null}>
          <circleGeometry args={[0.86, 64]} />
          <meshStandardMaterial
            color={hovered ? "#3a3228" : "#1c2028"}
            metalness={0.22}
            roughness={0.48}
          />
        </mesh>

        <mesh position={[0, 0, -0.02]} raycast={() => null}>
          <planeGeometry args={[2.05, 2.05]} />
          <meshBasicMaterial
            color={hovered ? "#ffd8a0" : "#c9a86a"}
            transparent
            opacity={hovered ? 0.14 : 0.08}
            depthWrite={false}
          />
        </mesh>
  
        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={1.28}
          height={1.28}
        />

        <SmallRivets radius={0.78} />
  
        <WallText position={[0, -0.87, 0.045]} size={0.068} color="#c9a86a">
          FRONT OFFICE
        </WallText>

        <mesh position={[0, 0.95, 0.02]} raycast={() => null}>
          <planeGeometry args={[1.4, 0.2]} />
          <meshBasicMaterial color="#ffd8a0" transparent opacity={0.06} depthWrite={false} />
        </mesh>
      </group>
    );
  }

  /** Restrained franchise crest — smoked-glass wall emblem, not a giant pasted logo */
  function WallHeroLogo({ teamLogo, teamName, hovered = false }) {
    return (
      <group position={[0, 3.02, -3.44]} scale={[0.68, 0.68, 0.68]} raycast={() => null}>
        <mesh position={[0, 0, -0.04]}>
          <boxGeometry args={[1.35, 1.35, 0.045]} />
          <MetalMaterial color="#12141a" roughness={0.48} metalness={0.62} />
        </mesh>

        <mesh position={[0, 0, -0.022]}>
          <boxGeometry args={[1.18, 1.18, 0.028]} />
          <SmokedGlassMaterial opacity={0.22} hovered={hovered} />
        </mesh>

        <mesh position={[0, 0, -0.016]}>
          <planeGeometry args={[1.28, 1.28]} />
          <meshBasicMaterial
            color={hovered ? "#d4b878" : "#8a7348"}
            transparent
            opacity={hovered ? 0.1 : 0.05}
            depthWrite={false}
          />
        </mesh>

        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={0.72}
          height={0.72}
          opacity={hovered ? 0.88 : 0.72}
        />

        <SmallRivets radius={0.48} count={4} />

        <pointLight
          position={[0, 0, 0.28]}
          intensity={hovered ? 0.38 : 0.18}
          color="#c9a86a"
          distance={1.8}
        />
      </group>
    );
  }
  
  function TrophyShelf({ hovered, championshipCount = 0 }) {
    const legacyLabel =
      championshipCount > 0
        ? `${championshipCount} BANNERS`
        : "LEGACY WALL";

    return (
      <group>
        <mesh position={[0, -0.42, -0.02]} castShadow raycast={() => null}>
          <boxGeometry args={[1.42, 0.55, 0.04]} />
          <WoodMaterial color="#2a1810" roughness={0.55} />
        </mesh>

        {[-0.62, 0.62].map((x) => (
          <mesh key={`bracket-${x}`} position={[x, -0.22, -0.04]} raycast={() => null}>
            <boxGeometry args={[0.06, 0.28, 0.05]} />
            <MetalMaterial color="#5a4a38" roughness={0.35} metalness={0.65} />
          </mesh>
        ))}

        <mesh position={[0, -0.36, 0]}>
          <boxGeometry args={[1.35, 0.08, 0.28]} />
          <WoodMaterial color="#3b2317" roughness={0.48} />
        </mesh>

        {[0, 1, 2].map((i) => (
          <group key={i} position={[-0.42 + i * 0.42, -0.08, 0]}>
            <mesh position={[0, -0.2, 0]} castShadow>
              <cylinderGeometry args={[0.09, 0.13, 0.08, 24]} />
              <MetalMaterial color="#6b481c" roughness={0.38} metalness={0.45} />
            </mesh>

            <mesh position={[0, 0.02, 0]} castShadow>
              <cylinderGeometry args={[0.08, 0.12, 0.28, 24]} />
              <meshStandardMaterial
                color={hovered ? "#d4b05a" : "#b89230"}
                metalness={0.48}
                roughness={0.32}
                emissive={hovered ? "#6e5200" : "#000000"}
                emissiveIntensity={hovered ? 0.08 : 0}
              />
            </mesh>

            <mesh position={[0, 0.2, 0]} castShadow>
              <sphereGeometry args={[0.13, 20, 18]} />
              <meshStandardMaterial
                color={hovered ? "#d4b05a" : "#c99837"}
                metalness={0.42}
                roughness={0.28}
              />
            </mesh>
          </group>
        ))}

        <WallText position={[0, 0.46, 0.02]} size={0.062} color="#c9a86a">
          {legacyLabel}
        </WallText>

        {[0, 1].map((i) => (
          <group key={`plaque-${i}`} position={[-0.35 + i * 0.7, 0.18, 0.03]}>
            <mesh raycast={() => null}>
              <boxGeometry args={[0.28, 0.16, 0.012]} />
              <MetalMaterial color="#5a4a38" roughness={0.35} metalness={0.62} />
            </mesh>
            <WallText position={[0, 0, 0.012]} size={0.022} color="#e8dcc0">
              {i === 0 ? "HISTORY" : "RECORDS"}
            </WallText>
          </group>
        ))}
      </group>
    );
  }

  function HockeySticks() {
    return (
      <group position={[-3.75, 0.75, -2.88]} rotation={[0, 0, -0.15]}>
        {[0, 1, 2].map((i) => (
          <group
            key={i}
            position={[i * 0.08, 0, i * 0.035]}
            rotation={[0, 0, i * 0.13]}
          >
            <mesh position={[0, 0.63, 0]} rotation={[0, 0, 0.08]}>
              <boxGeometry args={[0.035, 1.45, 0.035]} />
              <meshStandardMaterial color="#5f3a21" roughness={0.58} />
            </mesh>

            <mesh position={[0.1, -0.1, 0]} rotation={[0, 0, 0.55]}>
              <boxGeometry args={[0.34, 0.045, 0.045]} />
              <meshStandardMaterial color="#1b1b1b" roughness={0.5} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  function RinkWhiteboard({ hovered }) {
    return (
      <WallDisplayFrame width={1.78} height={1.02} accent="#5a8aaa">
        <RoundedBox
          position={[0, 0, 0.02]}
          args={[1.62, 0.88, 0.04]}
          radius={0.02}
          smoothness={6}
          raycast={() => null}
        >
          <meshStandardMaterial
            color={hovered ? "#0c1824" : "#081420"}
            emissive={hovered ? "#143048" : "#0a2030"}
            emissiveIntensity={hovered ? 0.28 : 0.14}
            roughness={0.48}
            metalness={0.08}
          />
        </RoundedBox>

        <mesh position={[0, 0, 0.048]} raycast={() => null}>
          <planeGeometry args={[1.48, 0.74]} />
          <GlassMaterial opacity={0.06} />
        </mesh>

        <WallText position={[0, 0.36, 0.055]} size={0.042} color="#8aaaba">
          STRATEGY BOARD
        </WallText>

        <WallText position={[0, 0.28, 0.055]} size={0.022} color="#5a7a88">
          LINES • SPECIAL TEAMS • DEPTH
        </WallText>

        {[
          ["1LW", -0.42, 0.08, "#6a8a9a"],
          ["1C", 0, 0.02, "#8a7348"],
          ["1RW", 0.42, 0.08, "#6a8a9a"],
          ["LD", -0.22, -0.14, "#4a6a7a"],
          ["RD", 0.22, -0.14, "#4a6a7a"],
        ].map(([label, x, y, color]) => (
          <group key={label} position={[x, y, 0.056]}>
            <mesh raycast={() => null}>
              <circleGeometry args={[0.048, 20]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={hovered ? 0.35 : 0.18}
                roughness={0.42}
              />
            </mesh>
            <WallText position={[0, 0, 0.012]} size={0.022} color="#d8e8f0">
              {label}
            </WallText>
          </group>
        ))}

        {["PP1", "PK1", "SCRATCHES"].map((label, i) => (
          <WallText
            key={label}
            position={[-0.5 + i * 0.5, -0.34, 0.056]}
            size={0.024}
            color={i === 2 ? "#8a5a54" : "#5a8aaa"}
          >
            {label}
          </WallText>
        ))}
      </WallDisplayFrame>
    );
  }
  
  function PhysicalDraftBoard({ hovered, draftWeek = false }) {
    const teams = ["TEAM 1", "TEAM 2", "TEAM 3", "TEAM 4", "TEAM 5", "TEAM 6"];
    const names = [
      "PROSPECT 1",
      "PROSPECT 2",
      "PROSPECT 3",
      "PROSPECT 4",
      "PROSPECT 5",
      "PROSPECT 6",
      "PROSPECT 7",
      "PROSPECT 8",
      "PROSPECT 9",
      "PROSPECT 10",
      "PROSPECT 11",
      "PROSPECT 12",
    ];
  
    const colors = ["#f0d75e", "#8fd2ba", "#89c8e7", "#d7879b"];
  
    return (
      <group>
        <mesh position={[0, 0, -0.025]} castShadow raycast={() => null}>
          <boxGeometry args={[1.92, 1.35, 0.05]} />
          <WoodMaterial color="#2a2418" roughness={0.58} />
        </mesh>

        <RoundedBox args={[1.85, 1.28, 0.08]} radius={0.035} smoothness={6} raycast={() => null}>
          <meshStandardMaterial
            color={hovered ? "#ebe8dc" : "#d4cfc2"}
            roughness={0.72}
            metalness={0.02}
          />
        </RoundedBox>
  
        <mesh position={[-0.83, 0.49, 0.055]} raycast={() => null}>
          <boxGeometry args={[0.18, 0.17, 0.018]} />
          <meshStandardMaterial color="#27304a" />
        </mesh>
  
        <WallText position={[-0.83, 0.49, 0.074]} size={0.027} color="#f9e7a6">
          DRAFT
        </WallText>
  
        <WallText position={[0.12, 0.56, 0.075]} size={0.058} color="#111111">
          FRANCHISE DRAFT BOARD
        </WallText>
  
        {teams.map((team, col) => (
          <group key={team} position={[-0.62 + col * 0.25, 0.41, 0.075]}>
            <WallText position={[0, 0, 0]} size={0.027} color="#111111">
              {team}
            </WallText>
          </group>
        ))}
  
        {Array.from({ length: 12 }).map((_, row) => (
          <group key={row} position={[0, 0.31 - row * 0.065, 0.077]}>
            <WallText position={[-0.85, 0, 0]} size={0.025} color="#111111">
              {row + 1}
            </WallText>
  
            {teams.map((team, col) => {
              const color = colors[(row + col) % colors.length];
              const name = names[(row + col) % names.length];
  
              return (
                <group key={`${team}-${row}`} position={[-0.62 + col * 0.25, 0, 0]}>
                  <mesh position={[0, 0, 0.004]} raycast={() => null}>
                    <cylinderGeometry args={[0.012, 0.012, 0.018, 8]} />
                    <meshStandardMaterial color="#c94a44" roughness={0.5} />
                  </mesh>
                  <mesh position={[0, 0, 0.012]} raycast={() => null}>
                    <boxGeometry args={[0.23, 0.052, 0.009]} />
                    <meshBasicMaterial color={color} />
                  </mesh>
  
                  <WallText position={[0, 0.003, 0.012]} size={0.016} color="#0b0b0b">
                    {name}
                  </WallText>
                </group>
              );
            })}
          </group>
        ))}
  
        <WallText position={[0, -0.52, 0.075]} size={0.032} color="#3b3b3b">
          LOTTERY • NEEDS • TIERS • WATCHLIST
        </WallText>

        {["TIER 1", "RISERS", "WATCH"].map((label, i) => (
          <WallText
            key={label}
            position={[-0.45 + i * 0.45, 0.62, 0.076]}
            size={0.024}
            color="#6a5528"
          >
            {label}
          </WallText>
        ))}

        {draftWeek ? (
          <mesh position={[0, 0, 0.08]} raycast={() => null}>
            <planeGeometry args={[1.7, 1.15]} />
            <meshBasicMaterial color="#c9a86a" transparent opacity={0.07} depthWrite={false} />
          </mesh>
        ) : null}
      </group>
    );
  }
  
  function StandingsWallBoard({ hovered, standingsRank }) {
    const teams = ["ATL", "MET", "CEN", "PAC", "WC1", "WC2"];
  
    return (
      <group>
        <mesh position={[0, 0, -0.02]} castShadow raycast={() => null}>
          <boxGeometry args={[1.54, 1.02, 0.04]} />
          <WoodMaterial color="#151820" roughness={0.55} />
        </mesh>

        <RoundedBox args={[1.46, 0.95, 0.06]} radius={0.035} smoothness={6} raycast={() => null}>
          <meshStandardMaterial
            color={hovered ? "#1a2230" : "#111821"}
            emissive={hovered ? "#1e3a52" : "#000000"}
            emissiveIntensity={hovered ? 0.12 : 0}
            roughness={0.55}
            metalness={0.08}
          />
        </RoundedBox>
  
        <WallText position={[0, 0.35, 0.05]} size={0.058} color="#b8d4e8">
          STANDINGS
        </WallText>
  
        {teams.map((team, i) => (
          <group key={team} position={[0, 0.2 - i * 0.105, 0.06]}>
            <mesh raycast={() => null}>
              <boxGeometry args={[1.15, 0.075, 0.012]} />
              <meshBasicMaterial color={i % 2 ? "#1a2432" : "#141c28"} />
            </mesh>

            <mesh position={[0, 0, 0.018]} raycast={() => null}>
              <boxGeometry args={[1.12, 0.002, 0.004]} />
              <meshBasicMaterial color="#2a3848" transparent opacity={0.5} />
            </mesh>
  
            <WallText position={[-0.42, 0, 0.016]} size={0.032} color="#dff5ff">
              {team}
            </WallText>
  
            <WallText position={[0.32, 0, 0.016]} size={0.027} color="#f7d98f">
              {i === 0 ? safeText(standingsRank, "Race") : `${92 - i * 5} PTS`}
            </WallText>
          </group>
        ))}
      </group>
    );
  }
  
  // League Operations 3D wall — chart-style economics icon.
  function LeagueEconomyChart({ hovered }) {
    const bars = [
      { x: -0.52, h: 0.22, color: "#52df94" },
      { x: -0.18, h: 0.34, color: "#13d8e7" },
      { x: 0.16, h: 0.28, color: "#8ab4ff" },
      { x: 0.5, h: 0.42, color: "#e9a83c" },
    ];

    return (
      <WallDisplayFrame width={1.78} height={1.02} accent={OFFICE_PALETTE.gold}>
        <RoundedBox
          position={[0, 0, 0.02]}
          args={[1.62, 0.88, 0.04]}
          radius={0.02}
          smoothness={6}
          raycast={() => null}
        >
          <meshStandardMaterial
            color="#060810"
            emissive={hovered ? "#142838" : "#0a1828"}
            emissiveIntensity={hovered ? 0.32 : 0.16}
            roughness={0.44}
            metalness={0.1}
          />
        </RoundedBox>

        <WallText position={[0, 0.36, 0.055]} size={0.038} color="#c9a86a">
          LEAGUE OPS
        </WallText>

        <WallText position={[0, 0.28, 0.055]} size={0.02} color="#6a8a9a">
          CBA • CAP • REVENUE
        </WallText>

        {bars.map((bar) => (
          <mesh
            key={bar.x}
            position={[bar.x, -0.02 + bar.h / 2 - 0.12, 0.055]}
            raycast={() => null}
          >
            <boxGeometry args={[0.18, bar.h, 0.02]} />
            <meshStandardMaterial
              color={bar.color}
              emissive={bar.color}
              emissiveIntensity={hovered ? 0.55 : 0.28}
              roughness={0.35}
            />
          </mesh>
        ))}

        <mesh position={[0, -0.22, 0.054]} raycast={() => null}>
          <boxGeometry args={[1.1, 0.02, 0.01]} />
          <meshStandardMaterial color="#1a3040" emissive="#1a3040" emissiveIntensity={0.2} />
        </mesh>

        <WallText position={[0, -0.32, 0.062]} size={0.02} color="#7a9aaa">
          CAP FORECAST • TEAM MONEY
        </WallText>
      </WallDisplayFrame>
    );
  }

  function BroadcastScoreboard({ hovered, record, nextGame }) {
    return (
      <WallDisplayFrame width={1.78} height={1.02} accent={OFFICE_PALETTE.gold}>
        <RoundedBox
          position={[0, 0, 0.02]}
          args={[1.62, 0.88, 0.04]}
          radius={0.02}
          smoothness={6}
          raycast={() => null}
        >
          <meshStandardMaterial
            color="#060810"
            emissive={hovered ? "#142838" : "#0a1828"}
            emissiveIntensity={hovered ? 0.32 : 0.16}
            roughness={0.44}
            metalness={0.1}
          />
        </RoundedBox>

        <mesh position={[0, 0, 0.048]} raycast={() => null}>
          <planeGeometry args={[1.48, 0.74]} />
          <GlassMaterial opacity={0.07} />
        </mesh>

        <WallText position={[0, 0.36, 0.055]} size={0.042} color="#c9a86a">
          LEAGUE OPERATIONS
        </WallText>

        <WallText position={[0, 0.28, 0.055]} size={0.022} color="#6a8a9a">
          STANDINGS • SCORES • HEADLINES
        </WallText>

        <WallText position={[0, 0.06, 0.055]} size={0.034} color="#8aaaba">
          Record {safeText(record)}
        </WallText>

        <WallText position={[0, -0.06, 0.055]} size={0.028} color="#d8e0e8">
          Next {safeText(nextGame, "No game listed")}
        </WallText>

        <mesh position={[0, -0.3, 0.054]} raycast={() => null}>
          <boxGeometry args={[1.32, 0.06, 0.01]} />
          <meshStandardMaterial
            color="#101820"
            emissive="#1a3040"
            emissiveIntensity={hovered ? 0.22 : 0.1}
            roughness={0.5}
          />
        </mesh>

        <WallText position={[0, -0.3, 0.062]} size={0.022} color="#7a9aaa">
          BROADCAST • NEWS • LEAGUE FEED
        </WallText>
      </WallDisplayFrame>
    );
  }
  
  function ArenaWindowObject({ hovered, nextGame, seasonYear }) {
    return (
      <group>
        <mesh position={[0, 0, -0.05]} castShadow raycast={() => null}>
          <boxGeometry args={[2.32, 1.46, 0.08]} />
          <WoodMaterial color="#2a2418" roughness={0.58} />
        </mesh>

        <mesh raycast={() => null}>
          <boxGeometry args={[2.18, 1.32, 0.035]} />
          <meshStandardMaterial
            color={hovered ? "#1a4a6a" : "#122838"}
            emissive="#143850"
            emissiveIntensity={hovered ? 0.28 : 0.14}
            transparent
            opacity={0.88}
            roughness={0.35}
          />
        </mesh>

        <mesh position={[0, 0, 0.022]} raycast={() => null}>
          <planeGeometry args={[2.05, 1.18]} />
          <GlassMaterial opacity={0.14} />
        </mesh>

        {[-0.72, 0, 0.72].map((x) => (
          <mesh key={`mullion-${x}`} position={[x, 0, 0.028]} raycast={() => null}>
            <boxGeometry args={[0.04, 1.28, 0.02]} />
            <WoodMaterial color="#1a1612" roughness={0.62} />
          </mesh>
        ))}

        <mesh position={[0, 0.62, 0.028]} raycast={() => null}>
          <boxGeometry args={[2.12, 0.04, 0.02]} />
          <WoodMaterial color="#1a1612" roughness={0.62} />
        </mesh>
  
        <mesh position={[0, -0.48, 0.04]} castShadow raycast={() => null}>
          <boxGeometry args={[2.1, 0.12, 0.14]} />
          <WoodMaterial color="#2a2018" roughness={0.55} />
        </mesh>
  
        {[-0.55, 0, 0.55].map((x) => (
          <mesh key={x} position={[x, 0.38, 0.045]} raycast={() => null}>
            <circleGeometry args={[0.06, 16]} />
            <meshBasicMaterial
              color="#e8dcc0"
              transparent
              opacity={hovered ? 0.75 : 0.4}
            />
          </mesh>
        ))}
  
        <WallText position={[0, 0.22, 0.05]} size={0.068} color="#b8dce8">
          GAME DAY
        </WallText>
  
        <WallText position={[0, 0.02, 0.05]} size={0.042} color="#8ec0d8">
          {safeText(nextGame, "Next matchup")}
        </WallText>
  
        <WallText position={[0, -0.17, 0.05]} size={0.04} color="#f3d895">
          {safeText(seasonYear, "Season")}
        </WallText>
      </group>
    );
  }
  
  // Phase 2 idea:
  // Split the full GLB in Blender into individual runtime props:
  // desk.glb, chair.glb, crt-monitor.glb, phone.glb, keyboard-mouse.glb,
  // binders.glb, filing-cabinet.glb, lamp.glb, mugs.glb, printer.glb.
  // Then replace procedural desk-zone props one by one while keeping InteractiveGroup hitboxes.
  function RetroOfficeModel({
    enabled = USE_RETRO_OFFICE_PACK,
    lowPowerMode = false,
    transform = RETRO_OFFICE_TRANSFORM,
  }) {
    const { scene } = useGLTF(RETRO_OFFICE_MODEL_PATH);

    const clonedScene = useMemo(() => {
      if (!scene) return null;
      return scene.clone(true);
    }, [scene]);

    useEffect(() => {
      if (!clonedScene) return;

      clonedScene.traverse((child) => {
        if (!child.isMesh) return;

        child.castShadow = !lowPowerMode;
        child.receiveShadow = true;

        // Critical: imported art should never block current InteractiveGroup hitboxes.
        child.raycast = () => null;

        const materials = Array.isArray(child.material)
          ? child.material
          : child.material
            ? [child.material]
            : [];

        materials.forEach((mat) => {
          if (!mat) return;
          mat.needsUpdate = true;

          if ("roughness" in mat && mat.roughness == null) {
            mat.roughness = 0.72;
          }

          if ("metalness" in mat && mat.metalness == null) {
            mat.metalness = 0.05;
          }
        });
      });

      if (process.env.NODE_ENV !== "production") {
        const meshNames = [];
        clonedScene.traverse((child) => {
          if (child.isMesh && child.name) meshNames.push(child.name);
        });
        console.log("[RetroOfficePack] Loaded mesh count:", meshNames.length);
        console.log("[RetroOfficePack] Sample meshes:", meshNames.slice(0, 40));
      }
    }, [clonedScene, lowPowerMode]);

    if (!enabled || !clonedScene) return null;

    return (
      <group
        position={transform.position}
        rotation={transform.rotation}
        scale={transform.scale}
        raycast={() => null}
      >
        <primitive object={clonedScene} />
      </group>
    );
  }

  useGLTF.preload(RETRO_OFFICE_MODEL_PATH);

  function OfficeScene({
    teamName,
    teamLogo,
    seasonYear,
    currentDate,
    record,
    capSpace,
    nextGame,
    standingsRank,
    unreadMessages,
    pendingTasks,
    activeStorylines,
    hoveredId,
    setHoveredId,
    handleOpenPanel,
    resetToken,
    officePictures,
    bestPlayer,
    officeMood,
    activePanel,
    lowPowerMode = false,
    prefersReducedMotion = false,
    championshipCount = 0,
    capPressure = false,
  }) {
    const mood = officeMood || {};
    const tradeActivity =
      mood.isTradeDeadline ||
      Number(unreadMessages || 0) > 0 ||
      mood.hasUrgentDecisions;

    return (
      <>
        <color attach="background" args={[OFFICE_PALETTE.void]} />
        <fog attach="fog" args={[OFFICE_PALETTE.void, 11, 20]} />
  
        <CameraRig
          resetToken={resetToken}
          activePanel={activePanel}
          lowPowerMode={lowPowerMode}
          prefersReducedMotion={prefersReducedMotion}
        />
        {!lowPowerMode ? <SoftShadows size={20} samples={12} focus={0.48} /> : null}
        <Environment preset="apartment" environmentIntensity={0.58} />
  
        <hemisphereLight intensity={0.38} color="#dce8f4" groundColor="#3a342c" />
        <ambientLight intensity={0.34} color="#c8d0dc" />
  
        <directionalLight
          position={[1.8, 4.8, 1.6]}
          intensity={0.82}
          color={mood.isPlayoffs ? "#a8c8dc" : "#e8d4a8"}
          castShadow={!lowPowerMode}
          shadow-mapSize-width={lowPowerMode ? 1024 : 2048}
          shadow-mapSize-height={lowPowerMode ? 1024 : 2048}
          shadow-bias={-0.00025}
        />
  
        <pointLight
          position={[0, 1.55, 0.15]}
          intensity={0.92}
          color="#6aa8d8"
          distance={3.4}
        />
  
        <pointLight
          position={[0, 3.05, -3.2]}
          intensity={0.62}
          color="#e8d0a0"
          distance={4.2}
        />
  
        <pointLight
          position={[-2.4, 2.1, -2.8]}
          intensity={0.38}
          color="#6a9ac0"
          distance={5.2}
        />
  
        <pointLight
          position={[2.4, 2.1, -2.8]}
          intensity={0.38}
          color="#6a9ac0"
          distance={5.2}
        />

        <spotLight
          position={[0, 3.6, 0.8]}
          angle={0.42}
          penumbra={0.65}
          intensity={0.48}
          color="#f0ddb0"
          distance={7}
          castShadow={false}
        />
  
        {!lowPowerMode ? (
          <AccumulativeShadows
            temporal
            frames={48}
            color="#1a1814"
            colorBlend={0.85}
            opacity={0.34}
            scale={8}
            position={[0, 0.018, 0]}
          >
            <RandomizedLight
              amount={4}
              radius={2.8}
              ambient={0.28}
              intensity={0.62}
              position={[1.6, 4.2, 1.8]}
              color="#c8a868"
            />
          </AccumulativeShadows>
        ) : null}

        {USE_RETRO_OFFICE_PACK ? <RetroOfficeModel lowPowerMode={lowPowerMode} /> : null}

        <RoomShell />

        <WallHeroLogo teamLogo={teamLogo} teamName={teamName} hovered={hoveredId === "dashboard"} />

        <Desk teamName={teamName} teamLogo={teamLogo}>
          <InteractiveGroup
            id="dashboard"
            label="Command Interface"
            description="Franchise overview, roster status, owner goals, cap pressure, and executive reports"
            position={[0, 1.02, 0.38]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.dashboard}
            hitBoxPosition={[0, 0.18, 0.04]}
          >
            {(hovered) => (
              <LaptopObject
                hovered={hovered}
                teamName={teamName}
                teamLogo={teamLogo}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="messages"
            label="GM Phone"
            description="Trade calls, owner messages, and staff inbox"
            badge={Number(unreadMessages || 0) > 0 ? `${unreadMessages} unread` : ""}
            position={[-1.32, 1.0, 0.62]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.messages}
            hitBoxPosition={[0, 0.09, 0]}
            lowPowerMode={lowPowerMode}
          >
            {(hovered) => (
              <PhoneObject
                hovered={hovered}
                unreadMessages={unreadMessages}
                hasTradeActivity={tradeActivity}
                callerLabel={mood.isTradeDeadline ? "TRADE DESK" : "LEAGUE GM"}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="calendar"
            label="Desk Calendar"
            description="Open schedule and simulation dates"
            position={[1.28, 1.0, 0.62]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.calendar}
            hitBoxPosition={[0, 0.075, 0]}
          >
            {(hovered) => (
              <CalendarObject
                hovered={hovered}
                currentDate={currentDate}
                nextGame={nextGame}
                teamLogo={teamLogo}
                teamName={teamName}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="scouting"
            label="Scouting Kit"
            description="Draft class, reports, watchlist, and scouts"
            position={[-0.78, 1.0, 0.08]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.scouting}
            hitBoxPosition={[0, 0.07, 0]}
          >
            {(hovered) => (
              <ScoutingKitObject
                hovered={hovered}
                teamLogo={teamLogo}
                teamName={teamName}
                draftWeek={mood.isDraftWeek}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="contracts"
            label="Contract Office"
            description="Contracts, free agency, and salary cap"
            badge={`Cap: ${formatMoney(capSpace)}`}
            position={[0.82, 1.0, 0.08]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.contracts}
            hitBoxPosition={[0, 0.07, 0]}
          >
            {(hovered) => (
              <ContractLedgerObject
                hovered={hovered}
                capSpace={capSpace}
                capPressure={capPressure}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="stats"
            label="Analytics Room"
            description="Skater stats, goalie stats, xGF%, CF%, PDO"
            position={[0.55, 1.0, 0.48]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.stats}
            hitBoxPosition={[0, 0.075, 0]}
          >
            {(hovered) => <TabletObject hovered={hovered} />}
          </InteractiveGroup>

          <InteractiveGroup
            id="news"
            label="League Storylines"
            description="Storylines, rumors, headlines, and recaps"
            badge={`${Number(activeStorylines || 0)} stories`}
            position={[-0.42, 1.0, 0.42]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.news}
            hitBoxPosition={[0, 0.075, 0]}
          >
            {(hovered) => (
              <NewspaperObject
                hovered={hovered}
                activeStorylines={activeStorylines}
              />
            )}
          </InteractiveGroup>

          <InteractiveGroup
            id="tasks"
            label="Decision Desk"
            description="Tasks, objectives, reminders, and urgent decisions"
            badge={`${Number(pendingTasks || 0)} tasks`}
            position={[0.42, 1.0, 0.42]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hitBoxArgs={OFFICE_HITBOXES.tasks}
            hitBoxPosition={[0, 0.075, 0]}
          >
            {(hovered) => (
              <ClipboardObject hovered={hovered} pendingTasks={pendingTasks} />
            )}
          </InteractiveGroup>

          <CoffeeAndPuck />
        </Desk>

        <InteractiveGroup
          id="teamIdentity"
          label="Franchise Culture Wall"
          description="Team identity, fanbase, morale, ownership"
          position={[1.55, 2.48, -3.46]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.teamIdentity}
          hitBoxPosition={[0, 0, 0]}
        >
          {(hovered) => (
            <WallLogo
              hovered={hovered}
              teamLogo={teamLogo}
              teamName={teamName}
              scale={0.88}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="lines"
          label="Line Strategy Board"
          description="Edit lines, special teams, depth chart, and strategy"
          position={[-2.55, 2.12, -3.46]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.lines}
          hitBoxPosition={[0, 0, 0]}
        >
          {(hovered) => <RinkWhiteboard hovered={hovered} />}
        </InteractiveGroup>

        <InteractiveGroup
          id="standings"
          label="League Standings"
          description="Division race, playoff odds, power rankings"
          position={[-2.75, 0.95, -3.45]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.standings}
          hitBoxPosition={[0, 0.44, 0]}
        >
          {(hovered) => (
            <StandingsWallBoard
              hovered={hovered}
              standingsRank={standingsRank}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="leagueCentral"
          label="League Operations"
          description="CBA desk, cap forecast, and team revenue"
          position={[2.55, 2.12, -3.45]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.leagueCentral}
          hitBoxPosition={[0, 0, 0]}
        >
          {(hovered) => (
            <LeagueEconomyChart hovered={hovered} />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="draft"
          label="Draft War Room"
          description="Prospects, rankings, watchlist, lottery, and team needs"
          position={[-4.34, 1.82, -1.45]}
          rotation={[0, Math.PI / 2, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.draft}
          hitBoxPosition={[0, 0, 0]}
        >
          {(hovered) => (
            <PhysicalDraftBoard hovered={hovered} draftWeek={mood.isDraftWeek} />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="awards"
          label="Legacy Wall"
          description="Awards, records, banners, and franchise history"
          position={[4.34, 1.55, -1.65]}
          rotation={[0, -Math.PI / 2, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.awards}
          hitBoxPosition={[0, 0.1, 0]}
        >
          {(hovered) => (
            <TrophyShelf
              hovered={hovered}
              championshipCount={championshipCount}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="gameDay"
          label="Arena Window"
          description="Game preview, simulation, broadcast, and matchup report"
          position={[3.76, 1.65, -0.85]}
          rotation={[0, -Math.PI / 2, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.arenaWindow}
          hitBoxPosition={[0, 0, 0]}
        >
          {(hovered) => (
            <ArenaWindowObject
              hovered={hovered}
              nextGame={nextGame}
              seasonYear={seasonYear}
            />
          )}
        </InteractiveGroup>
  
        <ContactShadows
          position={[0, 0.012, 0]}
          opacity={0.68}
          scale={7.5}
          blur={3.2}
          far={4.5}
          color="#000000"
        />
      </>
    );
  }
  
  function OfficeHud({
    teamName,
    teamLogo,
    seasonYear,
    currentDate,
    record,
    capSpace,
    nextGame,
    standingsRank,
    officeMood,
    franchisePulse = null,
    urgentItems = [],
    onUrgentSelect,
    onReset,
    onQuickMenu,
    onExitOffice,
    lowPowerMode = false,
    onToggleLowPower,
    prefersReducedMotion = false,
  }) {
    const mood = officeMood || {};
    const urgent = officeSafeArray(urgentItems);
    const pulse = franchisePulse && typeof franchisePulse === "object" ? franchisePulse : {};

    return (
      <div className="office-hud">
        <div className="office-hud-card office-hud-card--left">
          <div className="office-hud-card__brand">
            <TeamLogoBadge
              teamLogo={teamLogo}
              teamName={teamName}
              size={52}
              variant="badge"
            />
            <div className="office-hud-card__copy">
              <span>EXECUTIVE SUITE</span>
              <strong>{teamName}</strong>
              <small>
                {safeText(seasonYear, "Season")} · {safeText(currentDate, "Today")}
              </small>
            </div>
          </div>
        </div>

        <div className="office-hud-card office-hud-card--right">
          <div className="office-hud-card__brand">
            <TeamLogoBadge
              teamLogo={teamLogo}
              teamName={teamName}
              size={44}
              variant="circle"
            />
            <div className="office-hud-card__copy">
              <span>FRANCHISE PULSE</span>
              <strong>{safeText(record, "0-0-0")}</strong>
              <small>
                {safeText(standingsRank, "Standings")} · Cap {formatMoney(capSpace)}
              </small>
              {pulse.revenue_label ? (
                <em>
                  Rev {safeText(pulse.revenue_label, "—")} · Fans {safeText(pulse.fan_label, "—")}
                </em>
              ) : null}
              {pulse.cap_pull_label ? (
                <em>
                  Cap Pull {safeText(pulse.cap_pull_label, "—")} · Boycott {safeText(pulse.boycott_risk, "Low")}
                </em>
              ) : null}
              <em>Next · {safeText(nextGame, "No game listed")}</em>
              {mood.officeMode ? (
                <em className="office-hud-mode">
                  Mode: {safeText(mood.officeMode, "regular").replace(/_/g, " ")}
                </em>
              ) : null}
            </div>
          </div>
        </div>

        <div className="office-urgent-desk">
          <span>Priority Briefing</span>
          {urgent.length ? (
            <ul className="office-urgent-desk__list">
              {urgent.slice(0, 5).map((item) => (
                <li
                  key={item.id}
                  className={`office-urgent-desk__item office-urgent-desk__item--${item.severity || "low"}`}
                >
                  <button
                    type="button"
                    onClick={() => onUrgentSelect?.(item.target, item)}
                  >
                    <strong>{item.title}</strong>
                    <small>{item.detail}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="office-urgent-desk__empty">
              No urgent fires on the desk. That either means you are doing well, or
              the league is waiting to ruin your week.
            </p>
          )}
        </div>

        <div className="office-control-bar">
          <div className="office-control-bar__utilities">
            <button type="button" className="office-control-bar__utility" onClick={onReset}>
              Reset View
            </button>

            {onToggleLowPower ? (
              <button type="button" className="office-control-bar__utility" onClick={onToggleLowPower}>
                {lowPowerMode ? "Full Detail" : "Performance"}
              </button>
            ) : null}

            {onExitOffice ? (
              <button type="button" className="office-control-bar__utility" onClick={onExitOffice}>
                Exit
              </button>
            ) : null}
          </div>

          <button type="button" className="office-control-bar__primary" onClick={onQuickMenu}>
            Command Menu
          </button>
        </div>

        <div className="office-instructions">
          <span>Look · Zoom · Select office systems</span>
        </div>
      </div>
    );
  }

  function OfficePanel({
    activePanel,
    teamName,
    teamLogo,
    record,
    capSpace,
    nextGame,
    standingsRank,
    onClose,
    onNavigate,
    panelCopy,
    briefingNote,
    urgentItems = [],
  }) {
    const panel = panelCopy || (activePanel ? PANEL_CONTENT[activePanel] : null);

    if (!panel) return null;

    const panelUrgent = officeSafeArray(urgentItems).filter((item) => {
      if (activePanel === OFFICE_PANEL_IDS.MESSAGES) {
        return item.type === "messages" || item.type === "trade";
      }
      if (activePanel === OFFICE_PANEL_IDS.CONTRACTS) return item.type === "contracts";
      if (activePanel === OFFICE_PANEL_IDS.SCOUTING || activePanel === OFFICE_PANEL_IDS.DRAFT) {
        return item.type === "draft";
      }
      if (activePanel === OFFICE_PANEL_IDS.LINES) return item.type === "injuries";
      if (activePanel === OFFICE_PANEL_IDS.TASKS) return item.type === "tasks";
      return false;
    });

    return (
      <AnimatePresence>
        <motion.aside
          className="office-panel"
          initial={{ opacity: 0, x: 80, scale: 0.96 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: 80, scale: 0.96 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
        >
          <TeamLogoBadge
            className="office-panel-watermark"
            teamLogo={teamLogo}
            teamName={teamName}
            size={200}
            variant="watermark"
            opacity={0.1}
          />

          <div className="office-panel-header">
            <TeamLogoBadge
              teamLogo={teamLogo}
              teamName={teamName}
              size={72}
              variant="framed"
            />

            <div>
              <span>{panel.eyebrow}</span>
              <h2>{panel.title}</h2>
            </div>

            <button type="button" className="office-panel-close" onClick={onClose}>
              ×
            </button>
          </div>

          {briefingNote ? (
            <p className="office-panel-briefing">{briefingNote}</p>
          ) : null}

          <p>{panel.description}</p>

          {panel.staffNote ? (
            <div className="office-panel-staff-note">
              <span>{panel.staffRole || "Staff Note"}</span>
              <p>{panel.staffNote}</p>
            </div>
          ) : null}

          {panel.pressureLine ? (
            <p className="office-panel-pressure">
              <strong>If ignored:</strong> {panel.pressureLine}
            </p>
          ) : null}

          {panelUrgent.length ? (
            <div className="office-panel-urgent">
              <span>Desk Priority</span>
              <ul>
                {panelUrgent.slice(0, 3).map((item) => (
                  <li key={item.id}>
                    <button type="button" onClick={() => onNavigate(item.target)}>
                      {item.title}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="office-panel-stats">
            <div>
              <span>Record</span>
              <strong>{safeText(record, "0-0-0")}</strong>
            </div>

            <div>
              <span>Cap</span>
              <strong>{formatMoney(capSpace)}</strong>
            </div>

            <div>
              <span>Standing</span>
              <strong>{safeText(standingsRank, "—")}</strong>
            </div>
          </div>

          <div className="office-panel-next">
            <span>Next Game</span>
            <strong>{safeText(nextGame, "No game listed")}</strong>
          </div>

          <div className="office-panel-actions">
            {panel.actions.map(([label, target]) => (
              <button key={target} type="button" onClick={() => onNavigate(target)}>
                {label}
              </button>
            ))}
          </div>
        </motion.aside>
      </AnimatePresence>
    );
  }

  function QuickMenu({
    open,
    onClose,
    onNavigate,
    onSimNextGame,
    simDisabled = false,
    menuItems = QUICK_MENU,
  }) {
    const grouped = useMemo(() => {
      const buckets = {};
      officeSafeArray(menuItems).forEach((item) => {
        const groupId = item.group || "primary";
        if (!buckets[groupId]) buckets[groupId] = [];
        buckets[groupId].push(item);
      });
      return buckets;
    }, [menuItems]);

    const groupOrder = ["primary", "operations", "frontOffice", "future"];

    return (
      <AnimatePresence>
        {open ? (
          <motion.div
            className="office-quick-menu"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="office-quick-menu-card"
              initial={{ y: 32, scale: 0.98 }}
              animate={{ y: 0, scale: 1 }}
              exit={{ y: 32, scale: 0.98 }}
            >
              <div className="office-quick-menu-head">
                <div>
                  <span className="office-quick-menu-kicker">Franchise Command</span>
                  <h3>Select an area of hockey operations</h3>
                </div>

                <button type="button" className="office-quick-menu-close" onClick={onClose} aria-label="Close">
                  ×
                </button>
              </div>

              {onSimNextGame ? (
                <div className="office-quick-menu-sim">
                  <button
                    type="button"
                    className="office-quick-menu-sim-btn"
                    disabled={simDisabled}
                    onClick={() => {
                      onSimNextGame();
                      onClose();
                    }}
                  >
                    <strong>Sim Next Game</strong>
                    <small>Advance franchise to the next scheduled game</small>
                  </button>
                </div>
              ) : null}

              {groupOrder.map((groupId) => {
                const items = grouped[groupId];
                if (!items?.length) return null;
                const meta = FRANCHISE_COMMAND_GROUPS[groupId] || { label: groupId };

                return (
                  <section key={groupId} className={`office-quick-menu-section office-quick-menu-section--${groupId}`}>
                    <header className="office-quick-menu-section-head">
                      <span>{meta.label}</span>
                    </header>

                    <div className="office-quick-menu-grid">
                      {items.map((item) => {
                        const isPlaceholder = item.type === "placeholder";
                        const isHub = item.type === "hub";

                        return (
                          <button
                            type="button"
                            key={item.id}
                            className={[
                              "office-quick-menu-item",
                              item.highlight ? "is-highlight" : "",
                              isPlaceholder ? "is-placeholder" : "",
                              item.enabled === false ? "is-disabled" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                            onClick={() => {
                              if (item.enabled === false) return;
                              onNavigate?.(item.target);
                              onClose();
                            }}
                          >
                            <span className="office-quick-menu-item-eyebrow">
                              {item.eyebrow}
                              {item.badge ? ` • ${item.badge}` : ""}
                              {isPlaceholder ? " • Soon" : ""}
                            </span>
                            <strong>{item.label}</strong>
                            <small>{item.description}</small>
                            {isHub ? <em className="office-quick-menu-item-note">Stay in office</em> : null}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                );
              })}
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    );
  }
  
  function WebGLFallback({
    onNavigate,
    teamName,
    teamLogo,
    phase,
    record,
    capSpace,
    urgentItems = [],
    menuItems = QUICK_MENU,
    onSimNextGame,
    simDisabled = false,
  }) {
    const urgent = officeSafeArray(urgentItems);
    const grouped = useMemo(() => {
      const buckets = {};
      officeSafeArray(menuItems).forEach((item) => {
        const groupId = item.group || "primary";
        if (!buckets[groupId]) buckets[groupId] = [];
        buckets[groupId].push(item);
      });
      return buckets;
    }, [menuItems]);
    const groupOrder = ["primary", "operations", "frontOffice", "future"];

    return (
      <div className="office-fallback">
        <div className="office-fallback-hero">
          <TeamLogoBadge
            teamLogo={teamLogo}
            teamName={teamName}
            size={72}
            variant="framed"
          />
          <div>
            <span>Executive Office Fallback</span>
            <h2>{safeText(teamName, "Franchise Club")}</h2>
            <p>
              WebGL could not load, but your command center is still online. Phase{" "}
              {safeText(phase, "—")} • Record {safeText(record, "0-0-0")} • Cap{" "}
              {formatMoney(capSpace)}
            </p>
          </div>
        </div>

        <div className="office-fallback-briefing">
          <span>Executive Briefing</span>
          {urgent.length ? (
            <ul className="office-urgent-desk__list">
              {urgent.slice(0, 6).map((item) => (
                <li
                  key={item.id}
                  className={`office-urgent-desk__item office-urgent-desk__item--${item.severity || "low"}`}
                >
                  <button
                    type="button"
                    onClick={() => onNavigate?.(item.target)}
                  >
                    <strong>{item.title}</strong>
                    <small>{item.detail}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="office-urgent-desk__empty">
              No urgent fires on the desk. That either means you are doing well, or
              the league is waiting to ruin your week.
            </p>
          )}
        </div>

        {onSimNextGame ? (
          <div className="office-quick-menu-sim office-fallback-sim">
            <button
              type="button"
              className="office-quick-menu-sim-btn"
              disabled={simDisabled}
              onClick={onSimNextGame}
            >
              <strong>Sim Next Game</strong>
              <small>Advance franchise to the next scheduled game</small>
            </button>
          </div>
        ) : null}

        {groupOrder.map((groupId) => {
          const items = grouped[groupId];
          if (!items?.length) return null;
          const meta = FRANCHISE_COMMAND_GROUPS[groupId] || { label: groupId };
          return (
            <section key={groupId} className="office-fallback-section">
              <header className="office-quick-menu-section-head">
                <span>{meta.label}</span>
              </header>
              <div className="office-fallback-grid">
                {items.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={[
                      item.highlight ? "is-highlight" : "",
                      item.type === "placeholder" ? "is-placeholder" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => onNavigate?.(item.target)}
                  >
                    <span>
                      {item.eyebrow}
                      {item.badge ? ` • ${item.badge}` : ""}
                    </span>
                    <strong>{item.label}</strong>
                    <small>{item.description}</small>
                  </button>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    );
  }
  
  class OfficeErrorBoundary extends React.Component {
    constructor(props) {
      super(props);
      this.state = { hasError: false };
    }
  
    static getDerivedStateFromError() {
      return { hasError: true };
    }
  
    componentDidCatch(error) {
      console.error("Office scene crashed:", error);
    }
  
    render() {
      if (this.state.hasError) {
        return (
          <WebGLFallback
            onNavigate={this.props.onNavigate}
            teamName={this.props.teamName}
            teamLogo={this.props.teamLogo}
            phase={this.props.phase}
            record={this.props.record}
            capSpace={this.props.capSpace}
            urgentItems={this.props.urgentItems}
            menuItems={this.props.menuItems}
            onSimNextGame={this.props.onSimNextGame}
            simDisabled={this.props.simDisabled}
          />
        );
      }
  
      return this.props.children;
    }
  }
  
  export default function FirstPersonOfficeHub({
    teamName = "Franchise Club",
    teamLogo = "",
    seasonYear = "Season",
    currentDate = "Today",
    record = "0-0-0",
    capSpace = "—",
    nextGame = "No game listed",
    standingsRank = "Standings",
    unreadMessages = 0,
    pendingTasks = 0,
    activeStorylines = 0,
    franchiseState = null,
    team = null,
    officeMood: officeMoodProp = null,
    urgentItems: urgentItemsProp = null,
    officeSummary = null,
    panelRequest = null,
    onNavigate,
    onOpenPanel,
    onExitOffice,
    onPanelRequestHandled,
    onSimNextGame,
    simDisabled = false,
    players = [],
  }) {
    const [hoveredId, setHoveredId] = useState(null);
    const [activePanel, setActivePanel] = useState(null);
    const [showQuickMenu, setShowQuickMenu] = useState(false);
    const [resetToken, setResetToken] = useState(0);
    const [briefingNote, setBriefingNote] = useState("");
    const canvasHostRef = useRef(null);
    const [canvasReady, setCanvasReady] = useState(false);
    const [lowPowerMode, setLowPowerMode] = useState(() => {
      try {
        return localStorage.getItem(LOW_POWER_STORAGE_KEY) === "1";
      } catch (err) {
        return false;
      }
    });
    const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

    const bestPlayer = useMemo(() => getBestPlayer(players), [players]);
  
    const officePictures = useMemo(() => getOfficePictures(), []);
    const normalizedRecord = useMemo(() => formatRecord(record), [record]);
    const effectiveTeamLogo = useMemo(() => {
      const fromName = resolveFranchiseTeamLogo(
        { name: teamName, team_name: teamName },
        teamName
      );
      if (fromName) return fromName;
      return toLogoUrl(teamLogo);
    }, [teamLogo, teamName]);

    const summary = useMemo(
      () => ({
        unreadMessages,
        pendingTasks,
        activeStorylines,
        nextGame,
        record: normalizedRecord,
        capSpace,
        capSpaceRaw:
          team?.cap_space ??
          team?.capSpace ??
          franchiseState?.cap_space,
        ...(officeSummary || {}),
      }),
      [
        unreadMessages,
        pendingTasks,
        activeStorylines,
        nextGame,
        normalizedRecord,
        capSpace,
        team,
        franchiseState,
        officeSummary,
      ]
    );

    const officeMood = useMemo(
      () =>
        officeMoodProp ||
        deriveOfficeMood(franchiseState, team, summary),
      [officeMoodProp, franchiseState, team, summary]
    );

    const urgentItems = useMemo(
      () =>
        urgentItemsProp ||
        buildOfficeUrgentItems(franchiseState, team, summary),
      [urgentItemsProp, franchiseState, team, summary]
    );

    const contextualCommandRegistry = useMemo(
      () => getContextualCommandRegistry(FRANCHISE_COMMAND_REGISTRY, officeMood, urgentItems),
      [officeMood, urgentItems]
    );

    const activePanelCopy = useMemo(() => {
      if (!activePanel) return null;
      return getDynamicPanelCopy(
        activePanel,
        PANEL_CONTENT[activePanel],
        franchiseState,
        team,
        officeMood,
        urgentItems
      );
    }, [activePanel, franchiseState, team, officeMood, urgentItems]);

    const championshipCount = useMemo(() => {
      const cups =
        franchiseState?.championships ??
        franchiseState?.stanley_cups ??
        team?.championships ??
        team?.stanley_cups;
      if (Array.isArray(cups)) return cups.length;
      return officeSafeNumber(cups, 0);
    }, [franchiseState, team]);

    const franchisePulse = useMemo(
      () => franchiseState?.franchise_pulse || null,
      [franchiseState]
    );

    const capPressure = useMemo(() => {
      const raw =
        team?.cap_space ??
        team?.capSpace ??
        franchiseState?.cap_space ??
        summary.capSpaceRaw;
      const n = officeSafeNumber(raw, NaN);
      return Number.isFinite(n) && n < 2000000;
    }, [team, franchiseState, summary]);

    const webglSupported = useMemo(() => detectWebGLSupport(), []);
  
    const handleOpenPanel = useCallback(
      (panelId) => {
        const commandTarget = PANEL_TO_COMMAND_TARGET[panelId];
        if (commandTarget && onNavigate) {
          onNavigate(commandTarget);
          return;
        }

        if (!PANEL_CONTENT[panelId]) {
          console.warn("[OfficeNav] Missing panel:", panelId);
          if (onNavigate) onNavigate(panelId);
          return;
        }

        setActivePanel(panelId);

        if (onOpenPanel) {
          onOpenPanel(panelId);
        }
      },
      [onNavigate, onOpenPanel]
    );
  
    const handleNavigate = useCallback(
      (target) => {
        if (onNavigate) {
          onNavigate(target);
        } else {
          console.log("Navigate:", target);
        }
        setActivePanel(null);
        setBriefingNote("");
      },
      [onNavigate]
    );

    const handleToggleLowPower = useCallback(() => {
      setLowPowerMode((prev) => {
        const next = !prev;
        try {
          localStorage.setItem(LOW_POWER_STORAGE_KEY, next ? "1" : "0");
        } catch (err) {
          /* ignore storage failures */
        }
        return next;
      });
    }, []);

    useEffect(() => {
      if (!panelRequest?.panelId) return;
      handleOpenPanel(panelRequest.panelId);
      if (panelRequest.note) {
        setBriefingNote(panelRequest.note);
      }
      onPanelRequestHandled?.();
    }, [panelRequest, handleOpenPanel, onPanelRequestHandled]);

    useEffect(() => {
      if (typeof window === "undefined" || !window.matchMedia) return undefined;
      const media = window.matchMedia("(prefers-reduced-motion: reduce)");
      const apply = () => setPrefersReducedMotion(Boolean(media.matches));
      apply();
      media.addEventListener?.("change", apply);
      return () => media.removeEventListener?.("change", apply);
    }, []);

    useEffect(() => {
      let active = true;

      const enableCanvas = () => {
        if (active && canvasHostRef.current && webglSupported) {
          setCanvasReady(true);
        }
      };

      // Defer mount until the host div exists — avoids R3F connect() hitting null under StrictMode.
      const frameId = window.requestAnimationFrame(enableCanvas);

      return () => {
        active = false;
        window.cancelAnimationFrame(frameId);
        setCanvasReady(false);
      };
    }, [webglSupported]);
  
    useEffect(() => {
      validateOfficeNavigation();
    }, []);
  
    useEffect(() => {
      return () => document.body.classList.remove("office-cursor-active");
    }, []);
  
    useEffect(() => {
      const onKeyDown = (e) => {
        const key = e.key.toLowerCase();
  
        if (key === "escape") {
          if (showQuickMenu) {
            setShowQuickMenu(false);
            return;
          }
          setActivePanel(null);
          setBriefingNote("");
        }
  
        if (key === "m") {
          setShowQuickMenu((open) => !open);
        }
  
        if (key === "r") {
          setResetToken((v) => v + 1);
        }
      };
  
      window.addEventListener("keydown", onKeyDown);
      return () => window.removeEventListener("keydown", onKeyDown);
    }, [showQuickMenu]);
  
    const showFallback = !webglSupported;
    const effectiveLowPower = lowPowerMode || prefersReducedMotion;
    const phaseLabelText = officePhaseText(franchiseState) || seasonYear;

    if (showFallback) {
      return (
        <section
          className="office-hub office-hub--fallback"
          data-office-mode={officeMood.officeMode}
          data-season-phase={officeMood.seasonPhase}
          data-pressure={officeMood.pressureLevel}
        >
          <WebGLFallback
            onNavigate={handleNavigate}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            phase={phaseLabelText}
            record={normalizedRecord}
            capSpace={capSpace}
            urgentItems={urgentItems}
            menuItems={contextualCommandRegistry}
            onSimNextGame={onSimNextGame}
            simDisabled={simDisabled}
          />
          <OfficePanel
            activePanel={activePanel}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            record={normalizedRecord}
            capSpace={capSpace}
            nextGame={nextGame}
            standingsRank={standingsRank}
            panelCopy={activePanelCopy}
            briefingNote={briefingNote}
            urgentItems={urgentItems}
            onClose={() => {
              setActivePanel(null);
              setBriefingNote("");
            }}
            onNavigate={handleNavigate}
          />
        </section>
      );
    }

    return (
      <section
        className={`office-hub ${effectiveLowPower ? "office-hub--low-power" : ""}`}
        data-hovered={hoveredId || "none"}
        data-office-mode={officeMood.officeMode}
        data-season-phase={officeMood.seasonPhase}
        data-pressure={officeMood.pressureLevel}
        data-team-form={officeMood.teamForm}
        data-deadline={officeMood.isTradeDeadline ? "true" : "false"}
        data-draft-week={officeMood.isDraftWeek ? "true" : "false"}
        data-free-agency={officeMood.isFreeAgency ? "true" : "false"}
        data-playoffs={officeMood.isPlayoffs ? "true" : "false"}
        data-offseason={officeMood.isOffseason ? "true" : "false"}
        data-injury-crisis={officeMood.hasInjuryCrisis ? "true" : "false"}
        data-owner-pressure={officeMood.hasOwnerPressure ? "true" : "false"}
      >
        <div className="office-canvas" ref={canvasHostRef}>
          <OfficeErrorBoundary
            onOpenPanel={handleOpenPanel}
            onNavigate={handleNavigate}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            phase={phaseLabelText}
            record={normalizedRecord}
            capSpace={capSpace}
            urgentItems={urgentItems}
            menuItems={contextualCommandRegistry}
            onSimNextGame={onSimNextGame}
            simDisabled={simDisabled}
          >
            {canvasReady ? (
            <Canvas
              shadows={!effectiveLowPower}
              dpr={effectiveLowPower ? [1, 1] : [1, 2]}
              eventSource={canvasHostRef}
              camera={{
                position: OFFICE_CAMERA.position,
                fov: OFFICE_CAMERA.fov,
              }}
              gl={{
                antialias: !effectiveLowPower,
                powerPreference: effectiveLowPower ? "default" : "high-performance",
              }}
              onPointerMissed={() => {
                setHoveredId(null);
                document.body.classList.remove("office-cursor-active");
              }}
              onCreated={({ gl }) => {
                gl.toneMapping = THREE.ACESFilmicToneMapping;
                gl.toneMappingExposure = 1.28;
                gl.outputColorSpace = THREE.SRGBColorSpace;
              }}
            >
              <Suspense fallback={null}>
                <OfficeScene
                  teamName={teamName}
                  teamLogo={effectiveTeamLogo}
                  seasonYear={seasonYear}
                  currentDate={currentDate}
                  record={normalizedRecord}
                  capSpace={capSpace}
                  nextGame={nextGame}
                  standingsRank={standingsRank}
                  unreadMessages={unreadMessages}
                  pendingTasks={pendingTasks}
                  activeStorylines={activeStorylines}
                  hoveredId={hoveredId}
                  setHoveredId={setHoveredId}
                  handleOpenPanel={handleOpenPanel}
                  resetToken={resetToken}
                  officePictures={officePictures}
                  bestPlayer={bestPlayer}
                  officeMood={officeMood}
                  activePanel={activePanel}
                  lowPowerMode={effectiveLowPower}
                  prefersReducedMotion={prefersReducedMotion}
                  championshipCount={championshipCount}
                  capPressure={capPressure}
                />
  
                {!effectiveLowPower ? (
                  <EffectComposer multisampling={4}>
                    <Bloom
                      intensity={0.16}
                      luminanceThreshold={0.52}
                      luminanceSmoothing={0.9}
                    />

                    <Vignette eskil={false} offset={0.32} darkness={0.12} />

                    <Noise opacity={0.003} />
                  </EffectComposer>
                ) : (
                  <EffectComposer multisampling={0}>
                    <Vignette eskil={false} offset={0.32} darkness={0.12} />
                  </EffectComposer>
                )}
              </Suspense>
            </Canvas>
            ) : null}
          </OfficeErrorBoundary>
        </div>
  
        <div className="office-vignette" aria-hidden="true" />
  
        <OfficeHud
          teamName={teamName}
          teamLogo={effectiveTeamLogo}
          seasonYear={seasonYear}
          currentDate={currentDate}
          record={normalizedRecord}
          capSpace={capSpace}
          nextGame={nextGame}
          standingsRank={standingsRank}
          officeMood={officeMood}
          franchisePulse={franchisePulse}
          urgentItems={urgentItems}
          onUrgentSelect={handleNavigate}
          lowPowerMode={effectiveLowPower}
          prefersReducedMotion={prefersReducedMotion}
          onToggleLowPower={handleToggleLowPower}
          onReset={() => setResetToken((v) => v + 1)}
          onQuickMenu={() => setShowQuickMenu(true)}
          onExitOffice={onExitOffice}
        />

        <OfficePanel
          activePanel={activePanel}
          teamName={teamName}
          teamLogo={effectiveTeamLogo}
          record={normalizedRecord}
          capSpace={capSpace}
          nextGame={nextGame}
          standingsRank={standingsRank}
          panelCopy={activePanelCopy}
          briefingNote={briefingNote}
          urgentItems={urgentItems}
          onClose={() => {
            setActivePanel(null);
            setBriefingNote("");
          }}
          onNavigate={handleNavigate}
        />

        <QuickMenu
          open={showQuickMenu}
          onClose={() => setShowQuickMenu(false)}
          onNavigate={handleNavigate}
          onSimNextGame={onSimNextGame}
          simDisabled={simDisabled}
          menuItems={contextualCommandRegistry}
        />
      </section>
    );
  }