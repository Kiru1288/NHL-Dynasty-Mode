import React, {
    Fragment,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
  } from "react";
  import { useGameUI } from "../game/GameUIContext";
  import { SCREENS } from "../game/constants";
  import { baseURL as API_BASE, SESSION_STORAGE_KEY } from "../services/api";
  import { formatProspectLeague, formatProspectTeam } from "../events/prospectDevelopment/prospectDevelopmentHelpers";
  
  /**
   * Scouting.js
   * ---------------------------------------------------------------------------
   * NHL Franchise Mode scouting command screen.
   *
   * Core goals:
   * - No hardcoded team entities.
   * - No hardcoded player entities.
   * - No hardcoded prospect entities.
   * - No hardcoded images.
   * - Uses backend + franchiseState.realData when present.
   * - Keeps UI information light and game-like.
   * - Uses short labels, cards, icons, meters, actions.
   * - Embedded CSS included in later chunk.
   *
   * Expected backend endpoints:
   * GET  /api/franchise/scouting/state
   * GET  /api/franchise/scouting/world
   * GET  /api/franchise/scouting/prospects
   * GET  /api/franchise/scouting/assignments
   * POST /api/franchise/scouting/assign
   * POST /api/franchise/scouting/cancel
   * POST /api/franchise/scouting/interview
   * POST /api/franchise/scouting/dinner
   * POST /api/franchise/scouting/combine
   * POST /api/franchise/scouting/private-workout
   * POST /api/franchise/scouting/request-medical
   * POST /api/franchise/scouting/focus
   */
  
  /* -------------------------------------------------------------------------- */
  /* Constants                                                                  */
  /* -------------------------------------------------------------------------- */
  
  const DEFAULT_HEADERS = {
    "Content-Type": "application/json",
  };
  
  const EMPTY_ARRAY = Object.freeze([]);
  const EMPTY_OBJECT = Object.freeze({});
  
  const STORAGE_PREFIX = "nhlfm:scouting";
  
  const VIEW_MODES = Object.freeze({
    OVERVIEW: "overview",
    GLOBE: "globe",
    BOARD: "board",
    WATCHLIST: "watchlist",
    REPORTS: "reports",
    SCOUTS: "scouts",
    PLAYER: "player",
  });
  
  const SCOUTING_PHASES = Object.freeze({
    EARLY: "early",
    MID: "mid",
    LATE: "late",
    COMBINE: "combine",
    DRAFT_WEEK: "draft_week",
    OFFSEASON: "offseason",
  });
  
  const SCOUTING_ACTIONS = Object.freeze({
    PLAYER_FOCUS: "player_focus",
    REGION_SWEEP: "region_sweep",
    LIVE_VIEWING: "live_viewing",
    VIDEO_REVIEW: "video_review",
    CHARACTER_CHECK: "character_check",
    ANALYTICS: "analytics",
    INTERVIEW: "interview",
    DINNER: "dinner",
    COMBINE: "combine",
    PRIVATE_WORKOUT: "private_workout",
    MEDICAL: "medical",
  });
  
  const SCOUTING_INTENSITY = Object.freeze({
    LIGHT: "light",
    NORMAL: "normal",
    HEAVY: "heavy",
    ALL_IN: "all_in",
  });
  
  const SORT_KEYS = Object.freeze({
    RANK: "rank",
    NAME: "name",
    POSITION: "position",
    COUNTRY: "country",
    REGION: "region",
    SCOUTED: "scouted",
    UPSIDE: "upside",
    RISK: "risk",
  });
  
  const FILTER_DEFAULTS = Object.freeze({
    search: "",
    position: "all",
    country: "all",
    region: "all",
    league: "all",
    coverage: "all",
    onlyWatchlist: false,
    onlyNeedsWork: false,
    sortKey: SORT_KEYS.RANK,
    sortDirection: "asc",
  });
  
  const ENDPOINTS = Object.freeze({
    state: "/api/franchise/scouting/state",
    world: "/api/franchise/scouting/world",
    prospects: "/api/franchise/scouting/prospects",
    assignments: "/api/franchise/scouting/assignments",
    assign: "/api/franchise/scouting/assign",
    cancel: "/api/franchise/scouting/cancel",
    interview: "/api/franchise/scouting/interview",
    dinner: "/api/franchise/scouting/dinner",
    combine: "/api/franchise/scouting/combine",
    privateWorkout: "/api/franchise/scouting/private-workout",
    medical: "/api/franchise/scouting/request-medical",
    focus: "/api/franchise/scouting/focus",
  });
  
  const VIEW_TABS = [
    { mode: VIEW_MODES.OVERVIEW, label: "Overview" },
    { mode: VIEW_MODES.GLOBE, label: "Map" },
    { mode: VIEW_MODES.BOARD, label: "Board" },
    { mode: VIEW_MODES.WATCHLIST, label: "Watchlist" },
    { mode: VIEW_MODES.REPORTS, label: "Reports" },
    { mode: VIEW_MODES.SCOUTS, label: "Scouts" },
  ];
  
  /* -------------------------------------------------------------------------- */
  /* Utility helpers                                                            */
  /* -------------------------------------------------------------------------- */
  
  function cx(...classes) {
    return classes.filter(Boolean).join(" ");
  }
  
  function safeObject(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return value;
  }
  
  function safeArray(value) {
    if (Array.isArray(value)) return value;
    if (Array.isArray(value?.data)) return value.data;
    if (Array.isArray(value?.items)) return value.items;
    if (Array.isArray(value?.results)) return value.results;
    return [];
  }
  
  function stringOr(value, fallback = "") {
    if (typeof value === "string") return value;
    if (value == null) return fallback;
    return String(value);
  }
  
  function numberOr(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }
  
  function clamp(value, min, max) {
    const n = Number(value);
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, n));
  }
  
  function percentage(value, fallback = 0) {
    return clamp(numberOr(value, fallback), 0, 100);
  }
  
  function slugify(value) {
    return stringOr(value)
      .toLowerCase()
      .replace(/['"]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }
  
  function titleCase(input) {
    return stringOr(input)
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/\w\S*/g, (txt) => {
        return txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase();
      });
  }
  
  function compareText(a, b) {
    return stringOr(a).localeCompare(stringOr(b), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  }
  
  function getNested(obj, paths, fallback = undefined) {
    const source = safeObject(obj);
    const list = Array.isArray(paths) ? paths : [paths];
  
    for (const path of list) {
      if (!path) continue;
  
      const parts = String(path).split(".");
      let cur = source;
      let ok = true;
  
      for (const part of parts) {
        if (cur == null || typeof cur !== "object" || !(part in cur)) {
          ok = false;
          break;
        }
        cur = cur[part];
      }
  
      if (ok && cur !== undefined && cur !== null) return cur;
    }
  
    return fallback;
  }
  
  function formatMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (Math.abs(n) >= 1000000) return `$${(n / 1000000).toFixed(1)}M`;
    if (Math.abs(n) >= 1000) return `$${Math.round(n / 1000)}K`;
    return `$${Math.round(n)}`;
  }
  
  function formatDateLike(value) {
    if (!value) return "—";
  
    try {
      const d = new Date(value);
      if (Number.isNaN(d.getTime())) return stringOr(value, "—");
  
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return stringOr(value, "—");
    }
  }
  
  function makeOption(value, label, count = null) {
    return {
      value: stringOr(value),
      label: stringOr(label || value),
      count,
    };
  }
  
  function uniqueOptions(items, getter, allLabel = "All") {
    const counts = new Map();
  
    safeArray(items).forEach((item) => {
      const raw = getter(item);
      const value = stringOr(raw).trim();
      if (!value) return;
      counts.set(value, (counts.get(value) || 0) + 1);
    });
  
    const options = [...counts.entries()]
      .sort((a, b) => compareText(a[0], b[0]))
      .map(([value, count]) => makeOption(value, titleCase(value), count));
  
    return [makeOption("all", allLabel, safeArray(items).length), ...options];
  }
  
  function apiUrl(path) {
    if (!path) return API_BASE;
    if (/^https?:\/\//i.test(path)) return path;
    return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  }
  
  function franchiseHeaders() {
    const headers = { ...DEFAULT_HEADERS };
  
    try {
      const sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
      if (sessionId) headers["X-Franchise-Session"] = sessionId;
    } catch {
      // Storage can fail in private browsing.
    }
  
    return headers;
  }
  
  async function apiGet(path, options = {}) {
    const res = await fetch(apiUrl(path), {
      method: "GET",
      headers: franchiseHeaders(),
      credentials: "include",
      signal: options.signal,
    });
  
    const text = await res.text().catch(() => "");
  
    if (!res.ok) {
      if (text.trim().startsWith("<")) {
        throw new Error(`Backend returned HTML for ${path}.`);
      }
  
      throw new Error(text || `GET ${path} failed.`);
    }
  
    const trimmed = text.trim();
    if (!trimmed) return {};
  
    if (trimmed.startsWith("<")) {
      throw new Error(`Backend returned HTML for ${path}.`);
    }
  
    try {
      return JSON.parse(trimmed);
    } catch {
      throw new Error(`Invalid JSON from ${path}.`);
    }
  }
  
  async function apiPost(path, body, options = {}) {
    const res = await fetch(apiUrl(path), {
      method: "POST",
      headers: franchiseHeaders(),
      credentials: "include",
      body: JSON.stringify(body || {}),
      signal: options.signal,
    });
  
    const text = await res.text().catch(() => "");
  
    if (!res.ok) {
      if (text.trim().startsWith("<")) {
        throw new Error(`Backend returned HTML for ${path}.`);
      }
  
      throw new Error(text || `POST ${path} failed.`);
    }
  
    if (!text.trim()) return {};
  
    try {
      return JSON.parse(text);
    } catch {
      return {};
    }
  }
  
  function stableHash(input) {
    const s = stringOr(input);
    let hash = 0;
  
    for (let i = 0; i < s.length; i += 1) {
      hash = (hash << 5) - hash + s.charCodeAt(i);
      hash |= 0;
    }
  
    return Math.abs(hash);
  }
  
  /* -------------------------------------------------------------------------- */
  /* Hockey helpers                                                             */
  /* -------------------------------------------------------------------------- */
  
  function phaseFromState(state) {
    const raw = stringOr(
      getNested(state, [
        "phase",
        "scouting_phase",
        "scoutingPhase",
        "season.phase",
        "calendar.phase",
        "league.phase",
      ]),
      ""
    ).toLowerCase();
  
    if (raw.includes("combine")) return SCOUTING_PHASES.COMBINE;
    if (raw.includes("draft") && raw.includes("week")) return SCOUTING_PHASES.DRAFT_WEEK;
    if (raw.includes("late")) return SCOUTING_PHASES.LATE;
    if (raw.includes("mid")) return SCOUTING_PHASES.MID;
    if (raw.includes("off")) return SCOUTING_PHASES.OFFSEASON;
  
    return raw || SCOUTING_PHASES.EARLY;
  }
  
  function phaseLabel(phase) {
    switch (phase) {
      case SCOUTING_PHASES.EARLY:
        return "Early Season";
      case SCOUTING_PHASES.MID:
        return "Midseason";
      case SCOUTING_PHASES.LATE:
        return "Late Season";
      case SCOUTING_PHASES.COMBINE:
        return "Combine";
      case SCOUTING_PHASES.DRAFT_WEEK:
        return "Draft Week";
      case SCOUTING_PHASES.OFFSEASON:
        return "Offseason";
      default:
        return titleCase(phase || "Scouting");
    }
  }
  
  function phaseCue(phase) {
    switch (phase) {
      case SCOUTING_PHASES.COMBINE:
        return "Interviews matter now.";
      case SCOUTING_PHASES.DRAFT_WEEK:
        return "Final board lock.";
      case SCOUTING_PHASES.LATE:
        return "Confirm playoff habits.";
      case SCOUTING_PHASES.MID:
        return "Fix blind spots.";
      case SCOUTING_PHASES.OFFSEASON:
        return "Audit scouting misses.";
      default:
        return "Build the board.";
    }
  }
  
  function actionLabel(action) {
    switch (action) {
      case SCOUTING_ACTIONS.PLAYER_FOCUS:
        return "Player Focus";
      case SCOUTING_ACTIONS.REGION_SWEEP:
        return "Region Sweep";
      case SCOUTING_ACTIONS.LIVE_VIEWING:
        return "Live View";
      case SCOUTING_ACTIONS.VIDEO_REVIEW:
        return "Video";
      case SCOUTING_ACTIONS.CHARACTER_CHECK:
        return "Character";
      case SCOUTING_ACTIONS.ANALYTICS:
        return "Analytics";
      case SCOUTING_ACTIONS.INTERVIEW:
        return "Interview";
      case SCOUTING_ACTIONS.DINNER:
        return "Dinner";
      case SCOUTING_ACTIONS.COMBINE:
        return "Combine";
      case SCOUTING_ACTIONS.PRIVATE_WORKOUT:
        return "Workout";
      case SCOUTING_ACTIONS.MEDICAL:
        return "Medical";
      default:
        return titleCase(action || "Scout");
    }
  }
  
  function intensityLabel(value) {
    switch (value) {
      case SCOUTING_INTENSITY.LIGHT:
        return "Light";
      case SCOUTING_INTENSITY.NORMAL:
        return "Normal";
      case SCOUTING_INTENSITY.HEAVY:
        return "Heavy";
      case SCOUTING_INTENSITY.ALL_IN:
        return "All-In";
      default:
        return titleCase(value || "Normal");
    }
  }
  
  function intensityMultiplier(value) {
    switch (value) {
      case SCOUTING_INTENSITY.LIGHT:
        return 0.7;
      case SCOUTING_INTENSITY.HEAVY:
        return 1.45;
      case SCOUTING_INTENSITY.ALL_IN:
        return 2;
      case SCOUTING_INTENSITY.NORMAL:
      default:
        return 1;
    }
  }
  
  function actionBaseCost(action) {
    switch (action) {
      case SCOUTING_ACTIONS.REGION_SWEEP:
        return 8000;
      case SCOUTING_ACTIONS.LIVE_VIEWING:
        return 4500;
      case SCOUTING_ACTIONS.VIDEO_REVIEW:
        return 1200;
      case SCOUTING_ACTIONS.CHARACTER_CHECK:
        return 2500;
      case SCOUTING_ACTIONS.ANALYTICS:
        return 1800;
      case SCOUTING_ACTIONS.INTERVIEW:
        return 3000;
      case SCOUTING_ACTIONS.DINNER:
        return 7000;
      case SCOUTING_ACTIONS.COMBINE:
        return 5000;
      case SCOUTING_ACTIONS.PRIVATE_WORKOUT:
        return 9500;
      case SCOUTING_ACTIONS.MEDICAL:
        return 4000;
      case SCOUTING_ACTIONS.PLAYER_FOCUS:
      default:
        return 3500;
    }
  }
  
  function riskTone(value) {
    const numeric = Number(value);
    const text = stringOr(value).toLowerCase();
  
    if (Number.isFinite(numeric)) {
      if (numeric >= 75) return "danger";
      if (numeric >= 50) return "warn";
      if (numeric >= 25) return "watch";
      return "good";
    }
  
    if (text.includes("extreme")) return "danger";
    if (text.includes("high")) return "danger";
    if (text.includes("medium")) return "warn";
    if (text.includes("moderate")) return "warn";
    if (text.includes("low")) return "good";
  
    return "neutral";
  }
  
  function coverageTone(value) {
    const n = percentage(value);
  
    if (n >= 90) return "elite";
    if (n >= 70) return "good";
    if (n >= 45) return "watch";
    if (n >= 20) return "warn";
    return "danger";
  }
  
  function positionGroup(pos) {
    const p = stringOr(pos).toUpperCase();
  
    if (["C", "LW", "RW", "F", "W"].includes(p)) return "F";
    if (["LD", "RD", "D"].includes(p)) return "D";
    if (["G", "GOALIE"].includes(p)) return "G";
  
    return p || "—";
  }
  
  function prospectNeedLabel(prospect) {
    const p = safeObject(prospect);
  
    if (p.scouted < 25) return "Unknown";
    if (p.scouted < 50) return "Needs View";
    if (safeArray(p.redFlags).length) return "Flagged";
    if (p.volatility >= 60) return "Volatile";
    if (p.scouted >= 90) return "Final";
    return "Track";
  }
  
  function estimateAssignmentCost({ action, intensity, country, prospect }) {
    const base = actionBaseCost(action);
    const countryCost = numberOr(country?.cost, 0);
    const playerCost = numberOr(prospect?.costEstimate, 0);
  
    const difficulty =
      1 +
      percentage(country?.difficulty) / 220 +
      percentage(country?.safetyRisk) / 260 +
      percentage(country?.politicalRisk) / 300;
  
    return Math.round(
      (base + countryCost * 0.35 + playerCost * 0.45) *
        intensityMultiplier(intensity) *
        clamp(difficulty, 0.85, 2.5)
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* ID helpers                                                                 */
  /* -------------------------------------------------------------------------- */
  
  function getProspectId(player) {
    return (
      stringOr(player?.id).trim() ||
      stringOr(player?.player_id).trim() ||
      stringOr(player?.playerId).trim() ||
      stringOr(player?.prospect_id).trim() ||
      stringOr(player?.prospectId).trim() ||
      stringOr(player?.uuid).trim() ||
      slugify(`${player?.name || ""}-${player?.birthdate || ""}-${player?.league || ""}`)
    );
  }
  
  function getScoutId(scout) {
    return (
      stringOr(scout?.id).trim() ||
      stringOr(scout?.scout_id).trim() ||
      stringOr(scout?.scoutId).trim() ||
      stringOr(scout?.staff_id).trim() ||
      stringOr(scout?.staffId).trim() ||
      stringOr(scout?.uuid).trim() ||
      slugify(`${scout?.name || ""}-${scout?.role || ""}`)
    );
  }
  
  function getCountryId(country) {
    return (
      stringOr(country?.id).trim() ||
      stringOr(country?.country_id).trim() ||
      stringOr(country?.countryId).trim() ||
      stringOr(country?.iso3).trim() ||
      stringOr(country?.iso2).trim() ||
      stringOr(country?.code).trim() ||
      slugify(country?.name || country?.country || country?.label)
    );
  }
  
  function getRegionId(region) {
    return (
      stringOr(region?.id).trim() ||
      stringOr(region?.region_id).trim() ||
      stringOr(region?.regionId).trim() ||
      stringOr(region?.code).trim() ||
      slugify(region?.name || region?.region || region?.label)
    );
  }
  
  function getAssignmentId(assignment) {
    return (
      stringOr(assignment?.id).trim() ||
      stringOr(assignment?.assignment_id).trim() ||
      stringOr(assignment?.assignmentId).trim() ||
      stringOr(assignment?.uuid).trim() ||
      slugify(
        `${assignment?.scout_id || assignment?.scoutId || ""}-${
          assignment?.target_id || assignment?.targetId || ""
        }-${assignment?.created_at || assignment?.createdAt || ""}`
      )
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Normalizers                                                                */
  /* -------------------------------------------------------------------------- */
  
  function normalizeProspect(raw, index = 0) {
    const player = safeObject(raw);
  
    const id = getProspectId(player) || `prospect-${index}`;
  
    const name =
      stringOr(
        getNested(player, [
          "name",
          "full_name",
          "fullName",
          "identity.full_name",
          "identity.name",
          "profile.name",
        ]),
        "Unnamed Prospect"
      ) || "Unnamed Prospect";
  
    const position = stringOr(
      getNested(player, ["position", "pos", "primary_position", "profile.position"]),
      "—"
    ).toUpperCase();
  
    const country = stringOr(
      getNested(player, [
        "country",
        "nationality",
        "birth_country",
        "birthCountry",
        "identity.country",
        "profile.country",
        "bio.country",
      ]),
      "Unknown"
    );
  
    const region = stringOr(
      getNested(player, [
        "region",
        "scouting_region",
        "scoutingRegion",
        "geo.region",
        "profile.region",
      ]),
      ""
    );
  
    const league = formatProspectLeague(player)
      || formatProspectLeague({
        league_display: getNested(player, ["league_display", "leagueDisplay"]),
        league_code: getNested(player, ["league_code", "leagueCode"]),
        league_name: getNested(player, ["league_name", "leagueName"]),
        league: stringOr(
          getNested(player, [
            "league",
            "current_league",
            "currentLeague",
            "team.league",
            "profile.league",
          ]),
          ""
        ),
      });

    const team = formatProspectTeam(
      {
        team_name: getNested(player, ["team_name", "teamName"]),
        team: getNested(player, ["team", "current_team", "currentTeam", "club", "profile.team"]),
        club: getNested(player, ["club"]),
      },
      stringOr(
        getNested(player, ["team_name", "teamName", "team", "current_team", "currentTeam", "club", "profile.team"]),
        ""
      )
    );
  
    const rank = numberOr(
      getNested(player, [
        "rank",
        "draft_rank",
        "draftRank",
        "consensus_rank",
        "consensusRank",
        "draft_profile.rank",
        "draftProfile.rank",
      ]),
      index + 1
    );
  
    const scouted = percentage(
      getNested(player, [
        "scouted",
        "scouted_percentage",
        "scoutedPercentage",
        "scouting.scouted",
        "scouting.progress",
        "scouting.percent",
        "scouting.percentage",
        "scouting.completion",
      ]),
      0
    );
  
    const upside = percentage(
      getNested(player, [
        "upside",
        "potential_score",
        "potentialScore",
        "scouting.upside",
        "ratings.upside",
        "draft_profile.upside",
        "draftProfile.upside",
      ]),
      0
    );
  
    const floor = percentage(
      getNested(player, [
        "floor",
        "floor_score",
        "floorScore",
        "scouting.floor",
        "ratings.floor",
        "draft_profile.floor",
        "draftProfile.floor",
      ]),
      0
    );
  
    const risk = getNested(player, [
      "risk",
      "risk_score",
      "riskScore",
      "scouting.risk",
      "draft_profile.risk",
      "draftProfile.risk",
    ]);
  
    const projection = stringOr(
      getNested(player, [
        "projection",
        "nhl_projection",
        "nhlProjection",
        "draft_profile.projection",
        "draftProfile.projection",
        "scouting.projection",
      ]),
      "Unknown"
    );
  
    const age = numberOr(getNested(player, ["age", "profile.age", "bio.age"]), null);
  
    const height = stringOr(
      getNested(player, ["height", "profile.height", "bio.height", "measurements.height"]),
      ""
    );
  
    const weight = stringOr(
      getNested(player, ["weight", "profile.weight", "bio.weight", "measurements.weight"]),
      ""
    );
  
    const handedness = stringOr(
      getNested(player, ["handedness", "shoots", "catches", "profile.shoots"]),
      ""
    );
  
    const birthdate = getNested(player, ["birthdate", "birth_date", "dob", "bio.birthdate"]);
  
    const headshotUrl = stringOr(
      getNested(player, [
        "headshot",
        "headshot_url",
        "headshotUrl",
        "image",
        "image_url",
        "imageUrl",
        "portrait",
        "portrait_url",
        "portraitUrl",
        "media.headshot",
        "media.headshot_url",
        "media.headshotUrl",
      ]),
      ""
    );
  
    const watchlist = Boolean(
      getNested(player, ["watchlist", "is_watchlisted", "isWatchlisted", "scouting.watchlist"])
    );
  
    const traits = safeArray(
      getNested(player, ["traits", "player_traits", "playerTraits", "scouting.traits"])
    );
  
    const redFlags = safeArray(
      getNested(player, [
        "red_flags",
        "redFlags",
        "risk_flags",
        "riskFlags",
        "scouting.red_flags",
        "scouting.redFlags",
      ])
    );
  
    const notes = safeArray(
      getNested(player, ["notes", "reports", "scouting.notes", "scouting.reports"])
    );
  
    const skills = safeObject(
      getNested(player, [
        "skills",
        "ratings",
        "scouting.skills",
        "scouting.ratings",
        "draft_profile.skills",
        "draftProfile.skills",
      ])
    );
  
    const volatility = percentage(
      getNested(player, [
        "volatility",
        "draft_volatility",
        "draftVolatility",
        "stock_volatility",
        "stockVolatility",
        "scouting.volatility",
      ]),
      0
    );
  
    const draftStock = stringOr(
      getNested(player, [
        "draft_stock",
        "draftStock",
        "stock",
        "scouting.stock",
        "scouting.draft_stock",
      ]),
      "Stable"
    );
  
    const costEstimate = numberOr(
      getNested(player, [
        "cost",
        "scouting_cost",
        "scoutingCost",
        "travel_cost",
        "travelCost",
        "scouting.cost",
      ]),
      0
    );
  
    const combineStatus = stringOr(
      getNested(player, [
        "combine_status",
        "combineStatus",
        "scouting.combine_status",
        "scouting.combineStatus",
      ]),
      "Not Started"
    );
  
    const interviewStatus = stringOr(
      getNested(player, [
        "interview_status",
        "interviewStatus",
        "scouting.interview_status",
        "scouting.interviewStatus",
      ]),
      "Not Started"
    );
  
    const dinnerStatus = stringOr(
      getNested(player, [
        "dinner_status",
        "dinnerStatus",
        "scouting.dinner_status",
        "scouting.dinnerStatus",
      ]),
      "Not Started"
    );
  
    const needsWork =
      scouted < 60 ||
      Boolean(
        getNested(player, [
          "needs_attention",
          "needsAttention",
          "scouting.needs_attention",
          "scouting.flagged",
        ])
      );
  
    return {
      raw: player,
      id,
      name,
      position,
      positionGroup: positionGroup(position),
      country,
      countryKey: slugify(country),
      region,
      regionKey: slugify(region),
      league,
      team,
      rank,
      scouted,
      upside,
      floor,
      risk,
      riskTone: riskTone(risk),
      projection,
      age,
      height,
      weight,
      handedness,
      birthdate,
      headshotUrl,
      watchlist,
      traits,
      redFlags,
      notes,
      skills,
      volatility,
      draftStock,
      costEstimate,
      combineStatus,
      interviewStatus,
      dinnerStatus,
      needsWork,
    };
  }
  
  function normalizeScout(raw, index = 0) {
    const scout = safeObject(raw);
  
    const id = getScoutId(scout) || `scout-${index}`;
  
    const name = stringOr(
      getNested(scout, ["name", "full_name", "fullName", "profile.name"]),
      "Unnamed Scout"
    );
  
    const role = stringOr(
      getNested(scout, ["role", "type", "specialty", "profile.role"]),
      "Scout"
    );
  
    const region = stringOr(
      getNested(scout, ["region", "assigned_region", "assignedRegion", "profile.region"]),
      ""
    );
  
    const country = stringOr(
      getNested(scout, ["country", "assigned_country", "assignedCountry", "profile.country"]),
      ""
    );
  
    const headshotUrl = stringOr(
      getNested(scout, [
        "headshot",
        "headshot_url",
        "headshotUrl",
        "image",
        "image_url",
        "imageUrl",
        "portrait",
        "portrait_url",
        "portraitUrl",
        "media.headshot",
      ]),
      ""
    );
  
    const strengths = safeArray(
      getNested(scout, ["strengths", "specialties", "profile.strengths"])
    );
  
    const languages = safeArray(getNested(scout, ["languages", "profile.languages"]));
  
    const availability = stringOr(
      getNested(scout, ["availability", "status", "profile.availability"]),
      "available"
    );
  
    const workload = percentage(
      getNested(scout, ["workload", "load", "assignment_load", "assignmentLoad"]),
      0
    );
  
    const accuracy = percentage(
      getNested(scout, ["accuracy", "evaluation_accuracy", "evaluationAccuracy"]),
      50
    );
  
    const regionKnowledge = percentage(
      getNested(scout, ["region_knowledge", "regionKnowledge", "profile.region_knowledge"]),
      50
    );
  
    const character = percentage(
      getNested(scout, [
        "character_evaluation",
        "characterEvaluation",
        "profile.character_evaluation",
      ]),
      50
    );
  
    const analytics = percentage(
      getNested(scout, ["analytics", "analytics_score", "analyticsScore", "profile.analytics"]),
      50
    );
  
    return {
      raw: scout,
      id,
      name,
      role,
      region,
      country,
      headshotUrl,
      strengths,
      languages,
      availability,
      workload,
      accuracy,
      regionKnowledge,
      character,
      analytics,
    };
  }
  
  function normalizeCountry(raw, index = 0) {
    const country = safeObject(raw);
  
    const id = getCountryId(country) || `country-${index}`;
  
    const name = stringOr(
      getNested(country, ["name", "country", "label", "display_name", "displayName"]),
      "Unknown"
    );
  
    const region = stringOr(
      getNested(country, ["region", "scouting_region", "scoutingRegion", "continent"]),
      ""
    );
  
    const prospectCount = numberOr(
      getNested(country, [
        "prospect_count",
        "prospectCount",
        "players",
        "player_count",
        "playerCount",
      ]),
      0
    );
  
    const scoutedAverage = percentage(
      getNested(country, [
        "scouted_average",
        "scoutedAverage",
        "average_scouted",
        "averageScouted",
        "coverage",
      ]),
      0
    );
  
    const cost = numberOr(
      getNested(country, ["cost", "travel_cost", "travelCost", "scouting_cost", "scoutingCost"]),
      0
    );
  
    const effort = percentage(
      getNested(country, ["effort", "travel_effort", "travelEffort"]),
      0
    );
  
    const difficulty = percentage(
      getNested(country, [
        "difficulty",
        "logistics_difficulty",
        "logisticsDifficulty",
        "access_difficulty",
        "accessDifficulty",
      ]),
      0
    );
  
    const safetyRisk = percentage(
      getNested(country, [
        "safety_risk",
        "safetyRisk",
        "security_risk",
        "securityRisk",
        "risk",
      ]),
      0
    );
  
    const politicalRisk = percentage(
      getNested(country, [
        "political_risk",
        "politicalRisk",
        "admin_risk",
        "adminRisk",
        "government_risk",
        "governmentRisk",
      ]),
      0
    );
  
    const sourceRisk = percentage(
      getNested(country, [
        "source_risk",
        "sourceRisk",
        "corruption_risk",
        "corruptionRisk",
        "reliability_risk",
        "reliabilityRisk",
      ]),
      0
    );
  
    const notes = safeArray(
      getNested(country, ["notes", "travel_notes", "travelNotes", "scouting_notes"])
    );
  
    const lat = numberOr(getNested(country, ["lat", "latitude", "geo.lat"]), null);
    const lon = numberOr(
      getNested(country, ["lon", "lng", "longitude", "geo.lon", "geo.lng"]),
      null
    );
  
    const x = numberOr(getNested(country, ["x", "map_x", "mapX", "geo.x"]), null);
    const y = numberOr(getNested(country, ["y", "map_y", "mapY", "geo.y"]), null);
  
    return {
      raw: country,
      id,
      name,
      nameKey: slugify(name),
      region,
      regionKey: slugify(region),
      prospectCount,
      scoutedAverage,
      cost,
      effort,
      difficulty,
      safetyRisk,
      politicalRisk,
      sourceRisk,
      notes,
      lat,
      lon,
      x,
      y,
    };
  }
  
  function normalizeRegion(raw, index = 0) {
    const region = safeObject(raw);
  
    const id = getRegionId(region) || `region-${index}`;
  
    const name = stringOr(
      getNested(region, ["name", "region", "label", "display_name", "displayName"]),
      "Unknown"
    );
  
    const prospectCount = numberOr(
      getNested(region, ["prospect_count", "prospectCount", "players", "player_count"]),
      0
    );
  
    const countryCount = numberOr(
      getNested(region, ["country_count", "countryCount", "countries"]),
      0
    );
  
    const scoutedAverage = percentage(
      getNested(region, ["coverage", "scouted_average", "scoutedAverage"]),
      0
    );
  
    const cost = numberOr(
      getNested(region, ["cost", "average_cost", "averageCost", "travel_cost"]),
      0
    );
  
    const difficulty = percentage(
      getNested(region, ["difficulty", "logistics_difficulty", "logisticsDifficulty"]),
      0
    );
  
    return {
      raw: region,
      id,
      name,
      nameKey: slugify(name),
      prospectCount,
      countryCount,
      scoutedAverage,
      cost,
      difficulty,
    };
  }
  
  function normalizeAssignment(raw, index = 0) {
    const assignment = safeObject(raw);
  
    const id = getAssignmentId(assignment) || `assignment-${index}`;
  
    const targetType = stringOr(
      getNested(assignment, ["target_type", "targetType", "type"]),
      "player"
    );
  
    const targetId = stringOr(
      getNested(assignment, [
        "target_id",
        "targetId",
        "player_id",
        "playerId",
        "prospect_id",
        "prospectId",
        "country_id",
        "countryId",
        "region_id",
        "regionId",
      ]),
      ""
    );
  
    const scoutId = stringOr(
      getNested(assignment, ["scout_id", "scoutId", "staff_id", "staffId"]),
      ""
    );
  
    const action = stringOr(
      getNested(assignment, ["action", "assignment_type", "assignmentType"]),
      SCOUTING_ACTIONS.PLAYER_FOCUS
    );
  
    const intensity = stringOr(
      getNested(assignment, ["intensity", "effort"]),
      SCOUTING_INTENSITY.NORMAL
    );
  
    const status = stringOr(getNested(assignment, ["status", "state"]), "active");
  
    const progress = percentage(
      getNested(assignment, ["progress", "completion", "percent_complete", "percentComplete"]),
      0
    );
  
    const cost = numberOr(getNested(assignment, ["cost", "budget", "estimated_cost"]), 0);
  
    const createdAt = getNested(assignment, ["created_at", "createdAt", "date"]);
    const dueDate = getNested(assignment, ["due_date", "dueDate", "deadline"]);
  
    return {
      raw: assignment,
      id,
      targetType,
      targetId,
      scoutId,
      action,
      intensity,
      status,
      progress,
      cost,
      createdAt,
      dueDate,
    };
  }
  
  /* -------------------------------------------------------------------------- */
  /* Derived world data                                                         */
  /* -------------------------------------------------------------------------- */
  
  function deriveCountriesFromProspects(prospects) {
    const map = new Map();
  
    safeArray(prospects).forEach((prospect) => {
      const key = slugify(prospect.country || "Unknown");
      if (!key) return;
  
      const existing =
        map.get(key) ||
        normalizeCountry({
          id: key,
          name: prospect.country || "Unknown",
          region: prospect.region || "",
          prospect_count: 0,
          scouted_average: 0,
        });
  
      const nextCount = existing.prospectCount + 1;
  
      const nextAverage =
        (existing.scoutedAverage * existing.prospectCount + percentage(prospect.scouted)) /
        nextCount;
  
      map.set(key, {
        ...existing,
        prospectCount: nextCount,
        scoutedAverage: nextAverage,
        region: existing.region || prospect.region || "",
        regionKey: slugify(existing.region || prospect.region || ""),
      });
    });
  
    return [...map.values()].sort((a, b) => compareText(a.name, b.name));
  }
  
  function deriveRegionsFromCountries(countries) {
    const map = new Map();
  
    safeArray(countries).forEach((country) => {
      const key = slugify(country.region || "Unknown");
      if (!key) return;
  
      const existing =
        map.get(key) ||
        normalizeRegion({
          id: key,
          name: country.region || "Unknown",
          country_count: 0,
          prospect_count: 0,
          scouted_average: 0,
          cost: 0,
          difficulty: 0,
        });
  
      const nextCountryCount = existing.countryCount + 1;
      const nextProspectCount = existing.prospectCount + numberOr(country.prospectCount, 0);
  
      const nextCoverage =
        (existing.scoutedAverage * existing.countryCount + percentage(country.scoutedAverage)) /
        nextCountryCount;
  
      const nextCost =
        (existing.cost * existing.countryCount + numberOr(country.cost, 0)) / nextCountryCount;
  
      const nextDifficulty =
        (existing.difficulty * existing.countryCount + percentage(country.difficulty)) /
        nextCountryCount;
  
      map.set(key, {
        ...existing,
        countryCount: nextCountryCount,
        prospectCount: nextProspectCount,
        scoutedAverage: nextCoverage,
        cost: nextCost,
        difficulty: nextDifficulty,
      });
    });
  
    return [...map.values()].sort((a, b) => compareText(a.name, b.name));
  }
  
  function worldPointFromCountry(country, index, total) {
    const c = safeObject(country);
  
    if (Number.isFinite(c.x) && Number.isFinite(c.y)) {
      return {
        x: clamp(c.x, 5, 95),
        y: clamp(c.y, 6, 94),
      };
    }
  
    if (Number.isFinite(c.lat) && Number.isFinite(c.lon)) {
      return {
        x: clamp(((c.lon + 180) / 360) * 100, 5, 95),
        y: clamp(((90 - c.lat) / 180) * 100, 6, 94),
      };
    }
  
    const hash = stableHash(c.id || c.name || `${index}`);
    const ring = total > 0 ? index / total : 0;
    const angle = ((hash % 360) * Math.PI) / 180;
    const radius = 24 + (hash % 18) + ring * 14;
  
    return {
      x: clamp(50 + Math.cos(angle) * radius, 8, 92),
      y: clamp(50 + Math.sin(angle) * radius * 0.55, 10, 90),
    };
  }
  
  /* -------------------------------------------------------------------------- */
  /* realData extraction                                                        */
  /* -------------------------------------------------------------------------- */
  
  function extractRealData(franchiseState) {
    const state = safeObject(franchiseState);
    const draftRankings = safeObject(state.draft_class_rankings || state.draftClassRankings);
    const draftEntries = safeArray(draftRankings.entries);
    const devLeagues = safeArray(safeObject(state.roster_browser).development_leagues);

    const team = safeObject(state.team);

    return {
      state,
      realData: null,
      team,
      prospects: draftEntries.length > 0 ? draftEntries : [],
      scouts: safeArray(state.scouts || state.staff || state.scouting_state?.scouts),
      countries: safeArray(state.scouting_state?.countries),
      regions: safeArray(state.scouting_state?.regions),
      assignments: safeArray(state.scouting_state?.assignments || state.scoutingAssignments || state.assignments),
      development_leagues: devLeagues,
      draft_class_rankings: draftRankings,
    };
  }
  
  function getTeamName(team) {
    return (
      stringOr(team?.name).trim() ||
      stringOr(team?.team_name).trim() ||
      stringOr(team?.teamName).trim() ||
      stringOr(team?.full_name).trim() ||
      "Franchise Club"
    );
  }
  
  function getTeamCity(team) {
    return (
      stringOr(team?.city).trim() ||
      stringOr(team?.market).trim() ||
      stringOr(team?.location).trim() ||
      ""
    );
  }
  
  function getTeamLogo(team) {
    return stringOr(
      getNested(team, [
        "logo",
        "logo_url",
        "logoUrl",
        "team_logo",
        "teamLogo",
        "media.logo",
        "assets.logo",
        "branding.logo",
      ]),
      ""
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Hooks                                                                      */
  /* -------------------------------------------------------------------------- */
  
  function useMountedRef() {
    const mounted = useRef(false);
  
    useEffect(() => {
      mounted.current = true;
  
      return () => {
        mounted.current = false;
      };
    }, []);
  
    return mounted;
  }
  
  function useLocalStorageState(key, initialValue) {
    const [value, setValue] = useState(() => {
      try {
        const raw = window.localStorage.getItem(key);
        if (!raw) return initialValue;
        return JSON.parse(raw);
      } catch {
        return initialValue;
      }
    });
  
    useEffect(() => {
      try {
        window.localStorage.setItem(key, JSON.stringify(value));
      } catch {
        // ignore
      }
    }, [key, value]);
  
    return [value, setValue];
  }
  
  function useDebouncedValue(value, delay = 160) {
    const [debounced, setDebounced] = useState(value);
  
    useEffect(() => {
      const id = window.setTimeout(() => setDebounced(value), delay);
      return () => window.clearTimeout(id);
    }, [value, delay]);
  
    return debounced;
  }
  
  function useAsyncData(loader, deps) {
    const mountedRef = useMountedRef();
  
    const [state, setState] = useState({
      loading: true,
      error: "",
      data: null,
      updatedAt: null,
    });
  
    useEffect(() => {
      const controller = new AbortController();
  
      setState((prev) => ({
        ...prev,
        loading: true,
        error: "",
      }));
  
      Promise.resolve()
        .then(() => loader(controller.signal))
        .then((data) => {
          if (!mountedRef.current || controller.signal.aborted) return;
  
          setState({
            loading: false,
            error: "",
            data,
            updatedAt: new Date().toISOString(),
          });
        })
        .catch((err) => {
          if (!mountedRef.current || controller.signal.aborted) return;
  
          setState({
            loading: false,
            error: err?.message || "Scouting failed to load.",
            data: null,
            updatedAt: new Date().toISOString(),
          });
        });
  
      return () => controller.abort();
    }, deps);
  
    return state;
  }
  
  /* -------------------------------------------------------------------------- */
  /* Loader                                                                     */
  /* -------------------------------------------------------------------------- */
  
  async function loadScoutingPayload(signal, franchiseState) {
    const real = extractRealData(franchiseState);
  
    const safeFetch = async (path) => {
      try {
        return await apiGet(path, { signal });
      } catch (err) {
        return {
          __error: err?.message || `Missing endpoint: ${path}`,
        };
      }
    };
  
    const [stateRaw, worldRaw, prospectsRaw, assignmentsRaw] = await Promise.all([
      safeFetch(ENDPOINTS.state),
      safeFetch(ENDPOINTS.world),
      safeFetch(ENDPOINTS.prospects),
      safeFetch(ENDPOINTS.assignments),
    ]);
  
    const state = {
      ...safeObject(real.state),
      ...safeObject(real.realData?.state),
      ...safeObject(stateRaw?.state || stateRaw?.data || stateRaw),
    };
  
    const prospectSource =
      safeArray(prospectsRaw?.prospects).length > 0
        ? prospectsRaw.prospects
        : safeArray(prospectsRaw?.players).length > 0
          ? prospectsRaw.players
          : safeArray(prospectsRaw?.draft_class).length > 0
            ? prospectsRaw.draft_class
            : safeArray(prospectsRaw?.draftClass).length > 0
              ? prospectsRaw.draftClass
              : safeArray(prospectsRaw?.data).length > 0
                ? prospectsRaw.data
                : real.prospects;
  
    const prospects = safeArray(prospectSource).map(normalizeProspect);
  
    const scoutSource =
      safeArray(stateRaw?.scouts).length > 0
        ? stateRaw.scouts
        : safeArray(stateRaw?.staff).length > 0
          ? stateRaw.staff
          : safeArray(state?.scouts).length > 0
            ? state.scouts
            : safeArray(state?.staff).length > 0
              ? state.staff
              : real.scouts;
  
    const scouts = safeArray(scoutSource).map(normalizeScout);
  
    const assignmentSource =
      safeArray(assignmentsRaw?.assignments).length > 0
        ? assignmentsRaw.assignments
        : safeArray(assignmentsRaw?.data).length > 0
          ? assignmentsRaw.data
          : safeArray(assignmentsRaw?.items).length > 0
            ? assignmentsRaw.items
            : real.assignments;
  
    const assignments = safeArray(assignmentSource).map(normalizeAssignment);
  
    const countrySource =
      safeArray(worldRaw?.countries).length > 0
        ? worldRaw.countries
        : safeArray(worldRaw?.world?.countries).length > 0
          ? worldRaw.world.countries
          : safeArray(worldRaw?.data?.countries).length > 0
            ? worldRaw.data.countries
            : real.countries;
  
    const explicitCountries = safeArray(countrySource).map(normalizeCountry);
    const countries = explicitCountries.length
      ? explicitCountries
      : deriveCountriesFromProspects(prospects);
  
    const regionSource =
      safeArray(worldRaw?.regions).length > 0
        ? worldRaw.regions
        : safeArray(worldRaw?.world?.regions).length > 0
          ? worldRaw.world.regions
          : safeArray(worldRaw?.data?.regions).length > 0
            ? worldRaw.data.regions
            : real.regions;
  
    const explicitRegions = safeArray(regionSource).map(normalizeRegion);
    const regions = explicitRegions.length ? explicitRegions : deriveRegionsFromCountries(countries);
  
    const metaNotes = [
      stateRaw?.__error,
      worldRaw?.__error,
      prospectsRaw?.__error,
      assignmentsRaw?.__error,
    ].filter(Boolean);
  
    return {
      state,
      phase: phaseFromState(state),
      team: real.team,
      prospects,
      scouts,
      assignments,
      countries,
      regions,
      meta: {
        notes: metaNotes,
        source: explicitCountries.length ? "backend" : "derived",
        generatedAt:
          worldRaw?.generated_at ||
          worldRaw?.generatedAt ||
          state?.generated_at ||
          state?.generatedAt ||
          null,
      },
    };
  }
  
  /* -------------------------------------------------------------------------- */
  /* Small UI primitives                                                        */
  /* -------------------------------------------------------------------------- */
  
  function TeamLogo({ team, size = "md" }) {
    const logo = getTeamLogo(team);
    const name = getTeamName(team);
  
    return (
      <div className={cx("scout-team-logo", `scout-team-logo--${size}`)}>
        {logo ? (
          <img src={logo} alt={name} />
        ) : (
          <span>{name.slice(0, 2).toUpperCase()}</span>
        )}
      </div>
    );
  }
  
  function PersonAvatar({ src, name, label, size = "md" }) {
    return (
      <div className={cx("scout-avatar", `scout-avatar--${size}`)}>
        {src ? (
          <img src={src} alt={name || label || "Avatar"} />
        ) : (
          <span>{stringOr(label || name).slice(0, 2).toUpperCase()}</span>
        )}
      </div>
    );
  }
  
  function StatTile({ label, value, sub, tone = "blue", icon }) {
    return (
      <article className={cx("scout-stat-tile", `tone-${tone}`)}>
        <div className="scout-stat-tile__icon">{icon || label.slice(0, 2)}</div>
        <div>
          <span>{label}</span>
          <strong>{value}</strong>
          {sub ? <small>{sub}</small> : null}
        </div>
      </article>
    );
  }
  
  function CompactProgress({ value, tone = "blue", showValue = true }) {
    const v = percentage(value);
  
    return (
      <div className={cx("scout-compact-progress", `tone-${tone}`)}>
        {showValue ? <span>{Math.round(v)}%</span> : null}
        <div>
          <i style={{ width: `${v}%` }} />
        </div>
      </div>
    );
  }
  
  function ProgressRing({ value, label, tone = "blue" }) {
    const v = percentage(value);
    const radius = 32;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (v / 100) * circumference;
  
    return (
      <div className={cx("scout-ring", `tone-${tone}`)}>
        <svg viewBox="0 0 84 84" aria-hidden="true">
          <circle className="scout-ring__track" cx="42" cy="42" r={radius} />
          <circle
            className="scout-ring__fill"
            cx="42"
            cy="42"
            r={radius}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div>
          <strong>{Math.round(v)}%</strong>
          <span>{label}</span>
        </div>
      </div>
    );
  }
  
  function SelectField({ label, value, onChange, options }) {
    return (
      <label className="scout-field">
        <span>{label}</span>
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {safeArray(options).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
              {option.count != null && option.value !== "all" ? ` (${option.count})` : ""}
            </option>
          ))}
        </select>
      </label>
    );
  }
  
  function TextField({ label, value, onChange, placeholder }) {
    return (
      <label className="scout-field">
        <span>{label}</span>
        <input
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      </label>
    );
  }
  
  function EmptyState({ title, text, icon = "◎" }) {
    return (
      <div className="scout-empty">
        <div>{icon}</div>
        <h3>{title}</h3>
        {text ? <p>{text}</p> : null}
      </div>
    );
  }
  
  function LoadingState() {
    return (
      <div className="scout-loading">
        <div className="scout-loading-rink">
          <span />
          <i />
        </div>
        <strong>Loading scouting...</strong>
        <p>Board · map · reports</p>
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Main component                                                             */
  /* -------------------------------------------------------------------------- */
  
  export default function Scouting() {
    const { setScreen, franchiseState } = useGameUI();
    const mountedRef = useMountedRef();
  
    const [refreshToken, setRefreshToken] = useState(0);
  
    const [viewMode, setViewMode] = useLocalStorageState(
      `${STORAGE_PREFIX}:view`,
      VIEW_MODES.OVERVIEW
    );
  
    const [filters, setFilters] = useLocalStorageState(
      `${STORAGE_PREFIX}:filters`,
      FILTER_DEFAULTS
    );
  
    const [selectedCountryId, setSelectedCountryId] = useLocalStorageState(
      `${STORAGE_PREFIX}:country`,
      ""
    );
  
    const [selectedProspectId, setSelectedProspectId] = useLocalStorageState(
      `${STORAGE_PREFIX}:prospect`,
      ""
    );
  
    const [selectedScoutId, setSelectedScoutId] = useLocalStorageState(
      `${STORAGE_PREFIX}:scout`,
      ""
    );
  
    const [selectedAction, setSelectedAction] = useLocalStorageState(
      `${STORAGE_PREFIX}:action`,
      SCOUTING_ACTIONS.PLAYER_FOCUS
    );
  
    const [selectedIntensity, setSelectedIntensity] = useLocalStorageState(
      `${STORAGE_PREFIX}:intensity`,
      SCOUTING_INTENSITY.NORMAL
    );
  
    const [panelOpen, setPanelOpen] = useState(true);
    const [busyAction, setBusyAction] = useState("");
    const [toast, setToast] = useState({ type: "", text: "" });
  
    const scoutingLoad = useAsyncData(
      (signal) => loadScoutingPayload(signal, franchiseState),
      [refreshToken, franchiseState]
    );
  
    const payload = scoutingLoad.data || EMPTY_OBJECT;
  
    const team = payload.team || extractRealData(franchiseState).team || EMPTY_OBJECT;
    const teamName = getTeamName(team);
    const teamCity = getTeamCity(team);
  
    const phase = payload.phase || SCOUTING_PHASES.EARLY;
    const prospects = payload.prospects || EMPTY_ARRAY;
    const scouts = payload.scouts || EMPTY_ARRAY;
    const assignments = payload.assignments || EMPTY_ARRAY;
    const countries = payload.countries || EMPTY_ARRAY;
    const regions = payload.regions || EMPTY_ARRAY;
    const state = payload.state || EMPTY_OBJECT;
    const meta = payload.meta || EMPTY_OBJECT;
  
    const debouncedSearch = useDebouncedValue(filters.search, 160);
  
    const prospectById = useMemo(() => {
      const map = new Map();
      prospects.forEach((prospect) => map.set(prospect.id, prospect));
      return map;
    }, [prospects]);
  
    const countryById = useMemo(() => {
      const map = new Map();
      countries.forEach((country) => map.set(country.id, country));
      return map;
    }, [countries]);
  
    const scoutById = useMemo(() => {
      const map = new Map();
      scouts.forEach((scout) => map.set(scout.id, scout));
      return map;
    }, [scouts]);
  
    const selectedCountry = selectedCountryId ? countryById.get(selectedCountryId) || null : null;
    const selectedProspect = selectedProspectId
      ? prospectById.get(selectedProspectId) || null
      : null;
    const selectedScout = selectedScoutId ? scoutById.get(selectedScoutId) || null : null;
  
    const selectedProspectCountry = useMemo(() => {
      if (!selectedProspect) return null;
  
      return (
        countries.find((country) => country.nameKey === selectedProspect.countryKey) ||
        countries.find((country) => country.name === selectedProspect.country) ||
        null
      );
    }, [selectedProspect, countries]);
  
    const selectedCountryProspects = useMemo(() => {
      if (!selectedCountry) return EMPTY_ARRAY;
  
      return prospects
        .filter((prospect) => prospect.countryKey === selectedCountry.nameKey)
        .sort((a, b) => numberOr(a.rank, 9999) - numberOr(b.rank, 9999));
    }, [selectedCountry, prospects]);
  
    const filteredProspects = useMemo(() => {
      const search = debouncedSearch.trim().toLowerCase();
  
      const filtered = prospects.filter((prospect) => {
        if (search) {
          const haystack = [
            prospect.name,
            prospect.position,
            prospect.country,
            prospect.region,
            prospect.league,
            prospect.team,
            prospect.projection,
            prospect.draftStock,
            ...safeArray(prospect.traits),
            ...safeArray(prospect.redFlags),
          ]
            .join(" ")
            .toLowerCase();
  
          if (!haystack.includes(search)) return false;
        }
  
        if (filters.position !== "all" && prospect.position !== filters.position) return false;
        if (filters.country !== "all" && prospect.country !== filters.country) return false;
        if (filters.region !== "all" && prospect.region !== filters.region) return false;
        if (filters.league !== "all" && prospect.league !== filters.league) return false;
  
        if (filters.coverage !== "all") {
          if (filters.coverage === "unknown" && prospect.scouted >= 25) return false;
          if (filters.coverage === "partial" && (prospect.scouted < 25 || prospect.scouted >= 70)) {
            return false;
          }
          if (filters.coverage === "solid" && (prospect.scouted < 70 || prospect.scouted >= 90)) {
            return false;
          }
          if (filters.coverage === "final" && prospect.scouted < 90) return false;
        }
  
        if (filters.onlyWatchlist && !prospect.watchlist) return false;
        if (filters.onlyNeedsWork && !prospect.needsWork) return false;
  
        return true;
      });
  
      filtered.sort((a, b) => {
        let result = 0;
  
        switch (filters.sortKey) {
          case SORT_KEYS.NAME:
            result = compareText(a.name, b.name);
            break;
          case SORT_KEYS.POSITION:
            result = compareText(a.position, b.position);
            break;
          case SORT_KEYS.COUNTRY:
            result = compareText(a.country, b.country);
            break;
          case SORT_KEYS.REGION:
            result = compareText(a.region, b.region);
            break;
          case SORT_KEYS.SCOUTED:
            result = numberOr(b.scouted, 0) - numberOr(a.scouted, 0);
            break;
          case SORT_KEYS.UPSIDE:
            result = numberOr(b.upside, 0) - numberOr(a.upside, 0);
            break;
          case SORT_KEYS.RISK:
            result = percentage(b.risk) - percentage(a.risk);
            break;
          case SORT_KEYS.RANK:
          default:
            result = numberOr(a.rank, 9999) - numberOr(b.rank, 9999);
            break;
        }
  
        return filters.sortDirection === "desc" ? -result : result;
      });
  
      return filtered;
    }, [prospects, debouncedSearch, filters]);
  
    const watchlistProspects = useMemo(() => {
      return prospects
        .filter((prospect) => prospect.watchlist || prospect.needsWork || prospect.scouted < 50)
        .sort((a, b) => numberOr(a.rank, 9999) - numberOr(b.rank, 9999));
    }, [prospects]);
  
    const options = useMemo(() => {
      return {
        positionOptions: uniqueOptions(prospects, (p) => p.position, "All Positions"),
        countryOptions: uniqueOptions(prospects, (p) => p.country, "All Countries"),
        regionOptions: uniqueOptions(prospects, (p) => p.region, "All Regions"),
        leagueOptions: uniqueOptions(prospects, (p) => p.league, "All Leagues"),
      };
    }, [prospects]);
  
    const dashboardStats = useMemo(() => {
      const total = prospects.length;
  
      const avgCoverage = total
        ? prospects.reduce((sum, prospect) => sum + percentage(prospect.scouted), 0) / total
        : 0;
  
      const underScouted = prospects.filter((prospect) => prospect.scouted < 50).length;
      const highRisk = prospects.filter((prospect) => prospect.riskTone === "danger").length;
      const watchlist = prospects.filter((prospect) => prospect.watchlist).length;
  
      const activeAssignments = assignments.filter((assignment) => {
        const status = stringOr(assignment.status).toLowerCase();
        return ["active", "pending", "in_progress", "in progress"].includes(status);
      }).length;
  
      const finalReady = prospects.filter((prospect) => prospect.scouted >= 90).length;
  
      const budget = numberOr(
        getNested(state, [
          "budget",
          "scouting_budget",
          "scoutingBudget",
          "front_office.scouting_budget",
          "frontOffice.scoutingBudget",
        ]),
        0
      );
  
      return {
        total,
        avgCoverage,
        underScouted,
        highRisk,
        watchlist,
        activeAssignments,
        finalReady,
        budget,
        countries: countries.length,
        regions: regions.length,
        scouts: scouts.length,
      };
    }, [prospects, assignments, countries, regions, scouts, state]);
  
    const worldPoints = useMemo(() => {
      return countries.map((country, index) => {
        const point = worldPointFromCountry(country, index, countries.length);
  
        const relatedProspects = prospects.filter(
          (prospect) => prospect.countryKey === country.nameKey
        );
  
        const topProspect = relatedProspects
          .slice()
          .sort((a, b) => numberOr(a.rank, 9999) - numberOr(b.rank, 9999))[0];
  
        return {
          ...country,
          x: point.x,
          y: point.y,
          relatedProspects,
          topProspect,
        };
      });
    }, [countries, prospects]);
  
    const assignmentRows = useMemo(() => {
      return assignments.map((assignment) => {
        const scout = scoutById.get(assignment.scoutId) || null;
        const prospect = prospectById.get(assignment.targetId) || null;
  
        const country =
          countryById.get(assignment.targetId) ||
          countries.find((c) => c.nameKey === prospect?.countryKey) ||
          null;
  
        return {
          ...assignment,
          scout,
          prospect,
          country,
        };
      });
    }, [assignments, scoutById, prospectById, countryById, countries]);
  
    const selectedCountryForCost = selectedProspectCountry || selectedCountry;
  
    const estimatedCost = useMemo(() => {
      return estimateAssignmentCost({
        action: selectedAction,
        intensity: selectedIntensity,
        country: selectedCountryForCost,
        prospect: selectedProspect,
      });
    }, [selectedAction, selectedIntensity, selectedCountryForCost, selectedProspect]);
  
    const updateFilter = useCallback(
      (key, value) => {
        setFilters((prev) => ({
          ...prev,
          [key]: value,
        }));
      },
      [setFilters]
    );
  
    const resetFilters = useCallback(() => {
      setFilters(FILTER_DEFAULTS);
    }, [setFilters]);
  
    const showToast = useCallback(
      (type, text) => {
        setToast({ type, text });
  
        window.clearTimeout(showToast._id);
        showToast._id = window.setTimeout(() => {
          if (!mountedRef.current) return;
          setToast({ type: "", text: "" });
        }, 4200);
      },
      [mountedRef]
    );
  
    const runCommand = useCallback(
      async (name, path, body) => {
        setBusyAction(name);
  
        try {
          const result = await apiPost(path, body);
  
          if (!mountedRef.current) return null;
  
          showToast(
            "success",
            stringOr(result?.message, `${titleCase(name)} sent.`)
          );
  
          setRefreshToken((token) => token + 1);
          return result;
        } catch (err) {
          if (!mountedRef.current) return null;
  
          showToast("danger", err?.message || `${titleCase(name)} failed.`);
          return null;
        } finally {
          if (mountedRef.current) setBusyAction("");
        }
      },
      [mountedRef, showToast]
    );
  
    const selectCountry = useCallback(
      (countryId) => {
        setSelectedCountryId(countryId);
  
        const country = countryById.get(countryId);
  
        if (country) {
          const firstProspect = prospects
            .filter((prospect) => prospect.countryKey === country.nameKey)
            .sort((a, b) => numberOr(a.rank, 9999) - numberOr(b.rank, 9999))[0];
  
          if (firstProspect) {
            setSelectedProspectId(firstProspect.id);
          }
        }
      },
      [countryById, prospects, setSelectedCountryId, setSelectedProspectId]
    );
  
    const selectProspect = useCallback(
      (prospectId) => {
        setSelectedProspectId(prospectId);
        setViewMode(VIEW_MODES.PLAYER);
  
        const prospect = prospectById.get(prospectId);
  
        if (prospect) {
          const country = countries.find((c) => c.nameKey === prospect.countryKey);
          if (country) setSelectedCountryId(country.id);
        }
      },
      [setSelectedProspectId, setViewMode, prospectById, countries, setSelectedCountryId]
    );
  
    const assignScouting = useCallback(() => {
      const targetType = selectedProspect ? "player" : selectedCountry ? "country" : "";
      const targetId = selectedProspect?.id || selectedCountry?.id || "";
  
      if (!targetId) {
        showToast("danger", "Pick a player or country.");
        return;
      }
  
      runCommand("assign scouting", ENDPOINTS.assign, {
        scout_id: selectedScoutId || null,
        target_type: targetType,
        target_id: targetId,
        action: selectedAction,
        intensity: selectedIntensity,
        estimated_cost: estimatedCost,
        context: {
          phase,
          prospect_id: selectedProspect?.id || null,
          country_id: selectedCountry?.id || selectedProspectCountry?.id || null,
        },
      });
    }, [
      selectedProspect,
      selectedCountry,
      selectedScoutId,
      selectedAction,
      selectedIntensity,
      estimatedCost,
      selectedProspectCountry,
      phase,
      runCommand,
      showToast,
    ]);
  
    const runProspectAction = useCallback(
      (action) => {
        if (!selectedProspect) {
          showToast("danger", "Pick a prospect.");
          return;
        }
  
        const pathMap = {
          [SCOUTING_ACTIONS.INTERVIEW]: ENDPOINTS.interview,
          [SCOUTING_ACTIONS.DINNER]: ENDPOINTS.dinner,
          [SCOUTING_ACTIONS.COMBINE]: ENDPOINTS.combine,
          [SCOUTING_ACTIONS.PRIVATE_WORKOUT]: ENDPOINTS.privateWorkout,
          [SCOUTING_ACTIONS.MEDICAL]: ENDPOINTS.medical,
          [SCOUTING_ACTIONS.PLAYER_FOCUS]: ENDPOINTS.focus,
        };
  
        runCommand(actionLabel(action), pathMap[action] || ENDPOINTS.focus, {
          prospect_id: selectedProspect.id,
          player_id: selectedProspect.id,
          action,
          intensity: selectedIntensity,
          scout_id: selectedScoutId || null,
          estimated_cost: estimateAssignmentCost({
            action,
            intensity: selectedIntensity,
            country: selectedProspectCountry,
            prospect: selectedProspect,
          }),
          phase,
        });
      },
      [
        selectedProspect,
        selectedIntensity,
        selectedScoutId,
        selectedProspectCountry,
        phase,
        runCommand,
        showToast,
      ]
    );
  
    const cancelAssignment = useCallback(
      (assignment) => {
        runCommand("cancel assignment", ENDPOINTS.cancel, {
          assignment_id: assignment.id,
        });
      },
      [runCommand]
    );
  
    const goBack = useCallback(() => {
      if (SCREENS.HUB) setScreen(SCREENS.HUB);
    }, [setScreen]);
  
    const loadError = scoutingLoad.error;
  
    return (
      <div className="scout-root">
        <ScoutingStyles />
  
        <aside className="scout-sidebar">
          <button className="scout-home-button" type="button" onClick={goBack}>
            <span>⌂</span>
          </button>
  
          <nav className="scout-side-nav" aria-label="Scouting navigation">
            <SideButton label="Hub" icon="▦" onClick={() => setScreen(SCREENS.HUB)} />
            <SideButton label="Calendar" icon="◫" onClick={() => setScreen(SCREENS.CALENDAR)} />
            <SideButton label="Scouting" icon="◎" active onClick={() => setViewMode(VIEW_MODES.OVERVIEW)} />
            <SideButton label="Draft" icon="▤" onClick={() => setScreen(SCREENS.DRAFT_CLASS)} />
            <SideButton label="Office" icon="◆" onClick={() => setScreen(SCREENS.OFFICE)} />
          </nav>
  
          <button
            className="scout-settings"
            type="button"
            onClick={() => setScreen(SCREENS.SETTINGS)}
          >
            <span>⚙</span>
          </button>
        </aside>
  
        <main className="scout-main">
          <section className="scout-screen-head">
            <div className="scout-title-block">
              <TeamLogo team={team} />
              <div>
                <p>{teamCity || "Franchise"}</p>
                <h1>{teamName}</h1>
                <span>
                  Scouting · {phaseLabel(phase)}
                </span>
              </div>
            </div>
  
            <div className="scout-head-actions">
              <button type="button" onClick={() => setRefreshToken((t) => t + 1)}>
                {scoutingLoad.loading ? "Loading" : "Refresh"}
              </button>
              <button type="button" onClick={() => setPanelOpen((value) => !value)}>
                {panelOpen ? "Hide Panel" : "Panel"}
              </button>
              <button type="button" onClick={() => setScreen(SCREENS.DRAFT_CLASS)}>
                Draft Class
              </button>
            </div>
          </section>
  
          <section className="scout-top-strip">
            <StatTile
              icon="BD"
              label="Board"
              value={dashboardStats.total}
              sub={`${dashboardStats.finalReady} final`}
              tone="blue"
            />
            <StatTile
              icon="CF"
              label="Confidence"
              value={`${Math.round(dashboardStats.avgCoverage)}%`}
              sub={`${dashboardStats.underScouted} weak`}
              tone={dashboardStats.avgCoverage >= 70 ? "green" : "gold"}
            />
            <StatTile
              icon="SC"
              label="Scouts"
              value={dashboardStats.scouts}
              sub={`${dashboardStats.activeAssignments} active`}
              tone="cyan"
            />
            <StatTile
              icon="BG"
              label="Budget"
              value={dashboardStats.budget ? formatMoney(dashboardStats.budget) : "—"}
              sub={`${dashboardStats.countries} countries`}
              tone="purple"
            />
          </section>
  
          {(toast.text || loadError || safeArray(meta.notes).length > 0) && (
            <section className="scout-alert-stack">
              {toast.text ? (
                <div className={cx("scout-alert", `is-${toast.type || "info"}`)}>
                  {toast.text}
                </div>
              ) : null}
  
              {loadError ? (
                <div className="scout-alert is-danger">
                  {String(loadError).includes("HTML")
                    ? `Backend issue at ${API_BASE}.`
                    : loadError}
                </div>
              ) : null}
  
              {safeArray(meta.notes)
                .slice(0, 1)
                .map((note, index) => (
                  <div className="scout-alert is-warn" key={`${note}-${index}`}>
                    {note}
                  </div>
                ))}
            </section>
          )}
  
          <section className="scout-tabs" role="tablist" aria-label="Scouting views">
            {VIEW_TABS.map((tab) => (
              <button
                key={tab.mode}
                type="button"
                className={viewMode === tab.mode ? "is-active" : ""}
                onClick={() => setViewMode(tab.mode)}
              >
                {tab.label}
              </button>
            ))}
          </section>
  
          <section className={cx("scout-layout", panelOpen && "has-panel")}>
            <section className="scout-workspace">
              {scoutingLoad.loading ? (
                <LoadingState />
              ) : (
                <Fragment>
                  {viewMode === VIEW_MODES.OVERVIEW && (
                    <OverviewView
                      phase={phase}
                      stats={dashboardStats}
                      prospects={prospects}
                      countries={countries}
                      regions={regions}
                      assignments={assignmentRows}
                      onSelectProspect={selectProspect}
                      onSelectCountry={selectCountry}
                      setViewMode={setViewMode}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.GLOBE && (
                    <GlobeView
                      worldPoints={worldPoints}
                      regions={regions}
                      selectedCountryId={selectedCountryId}
                      selectedCountry={selectedCountry}
                      selectedCountryProspects={selectedCountryProspects}
                      onSelectCountry={selectCountry}
                      onSelectProspect={selectProspect}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.BOARD && (
                    <BoardView
                      prospects={filteredProspects}
                      filters={filters}
                      updateFilter={updateFilter}
                      resetFilters={resetFilters}
                      options={options}
                      selectedProspectId={selectedProspectId}
                      onSelectProspect={selectProspect}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.WATCHLIST && (
                    <WatchlistView
                      prospects={watchlistProspects}
                      onSelectProspect={selectProspect}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.REPORTS && (
                    <ReportsView
                      phase={phase}
                      state={state}
                      prospects={prospects}
                      countries={countries}
                      regions={regions}
                      assignments={assignmentRows}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.SCOUTS && (
                    <ScoutsView
                      scouts={scouts}
                      assignments={assignmentRows}
                      selectedScoutId={selectedScoutId}
                      setSelectedScoutId={setSelectedScoutId}
                    />
                  )}
  
                  {viewMode === VIEW_MODES.PLAYER && (
                    <PlayerView
                      prospect={selectedProspect}
                      country={selectedProspectCountry}
                      prospects={filteredProspects}
                      phase={phase}
                      onSelectProspect={selectProspect}
                      runProspectAction={runProspectAction}
                      busyAction={busyAction}
                    />
                  )}
                </Fragment>
              )}
            </section>
  
            {panelOpen ? (
              <aside className="scout-command-rail">
                <CommandPanel
                  scouts={scouts}
                  selectedScout={selectedScout}
                  selectedScoutId={selectedScoutId}
                  setSelectedScoutId={setSelectedScoutId}
                  selectedAction={selectedAction}
                  setSelectedAction={setSelectedAction}
                  selectedIntensity={selectedIntensity}
                  setSelectedIntensity={setSelectedIntensity}
                  selectedCountry={selectedCountry}
                  selectedProspect={selectedProspect}
                  estimatedCost={estimatedCost}
                  phase={phase}
                  busyAction={busyAction}
                  assignScouting={assignScouting}
                />
              </aside>
            ) : null}
          </section>
        </main>
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Sidebar                                                                    */
  /* -------------------------------------------------------------------------- */
  
  function SideButton({ active, icon, label, onClick }) {
    return (
      <button
        type="button"
        className={cx("scout-side-button", active && "is-active")}
        onClick={onClick}
        title={label}
      >
        <span>{icon}</span>
        <small>{label}</small>
      </button>
    );
  }
  /* -------------------------------------------------------------------------- */
/* Overview                                                                   */
/* -------------------------------------------------------------------------- */

function OverviewView({
    phase,
    stats,
    prospects,
    countries,
    regions,
    assignments,
    onSelectProspect,
    onSelectCountry,
    setViewMode,
  }) {
    const topProspects = useMemo(() => {
      return prospects
        .slice()
        .sort((a, b) => numberOr(a.rank, 9999) - numberOr(b.rank, 9999))
        .slice(0, 5);
    }, [prospects]);
  
    const coverageGaps = useMemo(() => {
      return countries
        .slice()
        .sort((a, b) => percentage(a.scoutedAverage) - percentage(b.scoutedAverage))
        .slice(0, 5);
    }, [countries]);
  
    const hardTrips = useMemo(() => {
      return countries
        .slice()
        .sort((a, b) => {
          const br = Math.max(b.safetyRisk, b.politicalRisk, b.sourceRisk, b.difficulty);
          const ar = Math.max(a.safetyRisk, a.politicalRisk, a.sourceRisk, a.difficulty);
          return br - ar;
        })
        .slice(0, 5);
    }, [countries]);
  
    const boardNeeds = useMemo(() => {
      const map = new Map();
  
      prospects.forEach((prospect) => {
        const key = prospect.position || "—";
        map.set(key, (map.get(key) || 0) + 1);
      });
  
      return [...map.entries()]
        .sort((a, b) => compareText(a[0], b[0]))
        .slice(0, 8);
    }, [prospects]);
  
    return (
      <div className="scout-overview">
        <header className="scout-section-head">
          <div>
            <span>Overview</span>
            <h2>{phaseCue(phase)}</h2>
          </div>
  
          <button type="button" onClick={() => setViewMode(VIEW_MODES.BOARD)}>
            View Board
          </button>
        </header>
  
        <section className="scout-overview-grid">
          <article className="scout-map-card scout-card">
            <div className="scout-card-head">
              <h3>Coverage Map</h3>
              <span>{stats.countries} countries</span>
            </div>
  
            <MiniWorldMap countries={countries} onSelectCountry={onSelectCountry} />
  
            <div className="scout-map-legend">
              <span>
                <i className="dot good" />
                Strong
              </span>
              <span>
                <i className="dot warn" />
                Thin
              </span>
              <span>
                <i className="dot danger" />
                Risk
              </span>
            </div>
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Position Needs</h3>
              <span>Board mix</span>
            </div>
  
            <div className="position-need-grid">
              {boardNeeds.length ? (
                boardNeeds.map(([position, count]) => {
                  const share = prospects.length ? (count / prospects.length) * 100 : 0;
  
                  return (
                    <div className="position-need-card" key={position}>
                      <span>{position}</span>
                      <strong>{count}</strong>
                      <CompactProgress value={share} tone="green" showValue={false} />
                    </div>
                  );
                })
              ) : (
                <EmptyState title="No board" text="No prospects returned." icon="BD" />
              )}
            </div>
          </article>
  
          <article className="scout-card scout-card--wide">
            <div className="scout-card-head">
              <h3>Top Prospects</h3>
              <button type="button" onClick={() => setViewMode(VIEW_MODES.BOARD)}>
                View All
              </button>
            </div>
  
            <div className="top-prospect-row">
              {topProspects.length ? (
                topProspects.map((prospect) => (
                  <ProspectTile
                    key={prospect.id}
                    prospect={prospect}
                    onClick={() => onSelectProspect(prospect.id)}
                  />
                ))
              ) : (
                <EmptyState title="No prospects" text="Draft class unavailable." icon="DP" />
              )}
            </div>
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Coverage Gaps</h3>
              <span>Lowest</span>
            </div>
  
            <CompactCountryList
              countries={coverageGaps}
              metric={(country) => `${Math.round(country.scoutedAverage)}%`}
              tone="danger"
              onSelectCountry={onSelectCountry}
            />
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Hard Trips</h3>
              <span>Risk</span>
            </div>
  
            <CompactCountryList
              countries={hardTrips}
              metric={(country) =>
                Math.round(
                  Math.max(
                    country.safetyRisk,
                    country.politicalRisk,
                    country.sourceRisk,
                    country.difficulty
                  )
                )
              }
              tone="warn"
              onSelectCountry={onSelectCountry}
            />
          </article>
  
          <article className="scout-card scout-card--wide">
            <div className="scout-card-head">
              <h3>Active Work</h3>
              <span>{assignments.length}</span>
            </div>
  
            <div className="overview-assignment-strip">
              {assignments.length ? (
                assignments.slice(0, 6).map((assignment) => (
                  <AssignmentMiniCard key={assignment.id} assignment={assignment} />
                ))
              ) : (
                <EmptyState title="No assignments" text="Pick a scout target." icon="AS" />
              )}
            </div>
          </article>
        </section>
      </div>
    );
  }
  
  function MiniWorldMap({ countries, onSelectCountry }) {
    const points = useMemo(() => {
      return countries.slice(0, 28).map((country, index) => {
        const point = worldPointFromCountry(country, index, countries.length);
  
        return {
          ...country,
          x: point.x,
          y: point.y,
        };
      });
    }, [countries]);
  
    return (
      <div className="mini-world">
        <div className="mini-world__grid" />
  
        {points.map((country) => {
          const risk = Math.max(
            country.safetyRisk,
            country.politicalRisk,
            country.sourceRisk,
            country.difficulty
          );
  
          return (
            <button
              key={country.id}
              type="button"
              className={cx(
                "mini-world-pin",
                country.scoutedAverage >= 70 && "is-good",
                country.scoutedAverage < 45 && "is-gap",
                risk >= 65 && "is-risk"
              )}
              style={{
                left: `${country.x}%`,
                top: `${country.y}%`,
              }}
              onClick={() => onSelectCountry(country.id)}
              title={`${country.name} · ${Math.round(country.scoutedAverage)}%`}
            >
              <span />
            </button>
          );
        })}
      </div>
    );
  }
  
  function ProspectTile({ prospect, onClick }) {
    return (
      <button type="button" className="prospect-tile" onClick={onClick}>
        <span className="prospect-rank">#{prospect.rank}</span>
  
        <PersonAvatar
          src={prospect.headshotUrl}
          name={prospect.name}
          label={prospect.position}
          size="lg"
        />
  
        <div className="prospect-tile__body">
          <strong>{prospect.name}</strong>
          <small>
            {prospect.position} · {prospect.country}
          </small>
        </div>
  
        <div className={cx("prospect-grade", coverageTone(prospect.scouted))}>
          <strong>{gradeFromCoverage(prospect.scouted)}</strong>
          <span>{Math.round(prospect.scouted)}%</span>
        </div>
      </button>
    );
  }
  
  function gradeFromCoverage(value) {
    const v = percentage(value);
  
    if (v >= 92) return "A";
    if (v >= 82) return "A-";
    if (v >= 72) return "B+";
    if (v >= 62) return "B";
    if (v >= 50) return "B-";
    if (v >= 35) return "C";
    return "?";
  }
  
  function CompactCountryList({ countries, metric, tone, onSelectCountry }) {
    if (!countries.length) {
      return <EmptyState title="No countries" text="World data unavailable." icon="GL" />;
    }
  
    return (
      <div className="compact-country-list">
        {countries.map((country, index) => (
          <button key={country.id} type="button" onClick={() => onSelectCountry(country.id)}>
            <b>{index + 1}</b>
            <span>{country.name}</span>
            <i className={tone}>{metric(country)}</i>
          </button>
        ))}
      </div>
    );
  }
  
  function AssignmentMiniCard({ assignment }) {
    const targetName =
      assignment.prospect?.name ||
      assignment.country?.name ||
      assignment.targetId ||
      "Unknown";
  
    return (
      <article className="assignment-mini">
        <span>{actionLabel(assignment.action)}</span>
        <strong>{targetName}</strong>
        <CompactProgress value={assignment.progress} tone="blue" />
      </article>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Globe                                                                      */
  /* -------------------------------------------------------------------------- */
  
  function GlobeView({
    worldPoints,
    regions,
    selectedCountryId,
    selectedCountry,
    selectedCountryProspects,
    onSelectCountry,
    onSelectProspect,
  }) {
    const [hoveredCountryId, setHoveredCountryId] = useState("");
  
    const hoveredCountry = useMemo(() => {
      return worldPoints.find((country) => country.id === hoveredCountryId) || null;
    }, [hoveredCountryId, worldPoints]);
  
    const activeCountry = hoveredCountry || selectedCountry;
  
    return (
      <div className="globe-view">
        <header className="scout-section-head">
          <div>
            <span>World Map</span>
            <h2>Scout by country</h2>
          </div>
        </header>
  
        <section className="globe-layout">
          <article className="scout-card globe-card">
            <div className="interactive-globe" role="img" aria-label="Scouting globe">
              <div className="interactive-globe__halo" />
              <div className="interactive-globe__sphere">
                <div className="globe-line longitude a" />
                <div className="globe-line longitude b" />
                <div className="globe-line longitude c" />
                <div className="globe-line latitude a" />
                <div className="globe-line latitude b" />
                <div className="globe-line latitude c" />
                <div className="globe-scan" />
  
                {worldPoints.map((country) => {
                  const selected = country.id === selectedCountryId;
                  const risk = Math.max(
                    country.safetyRisk,
                    country.politicalRisk,
                    country.sourceRisk,
                    country.difficulty
                  );
                  const gap = country.scoutedAverage < 45;
  
                  return (
                    <button
                      key={country.id}
                      type="button"
                      className={cx(
                        "globe-pin",
                        selected && "is-selected",
                        gap && "is-gap",
                        risk >= 65 && "is-risk"
                      )}
                      style={{
                        left: `${country.x}%`,
                        top: `${country.y}%`,
                      }}
                      onClick={() => onSelectCountry(country.id)}
                      onMouseEnter={() => setHoveredCountryId(country.id)}
                      onMouseLeave={() => setHoveredCountryId("")}
                      title={`${country.name} · ${country.prospectCount} prospects`}
                    >
                      <span className="globe-pin__pulse" />
                      <span className="globe-pin__core" />
                      <span className="globe-pin__label">{country.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
  
            <div className="globe-legend">
              <span>
                <i className="dot good" />
                Covered
              </span>
              <span>
                <i className="dot warn" />
                Gap
              </span>
              <span>
                <i className="dot danger" />
                Risk
              </span>
            </div>
          </article>
  
          <article className="scout-card country-intel-card">
            {activeCountry ? (
              <CountryIntel
                country={activeCountry}
                prospects={
                  activeCountry.id === selectedCountry?.id
                    ? selectedCountryProspects
                    : activeCountry.relatedProspects || EMPTY_ARRAY
                }
                onSelectProspect={onSelectProspect}
              />
            ) : (
              <EmptyState title="Select Country" text="Pins reveal targets." icon="MAP" />
            )}
          </article>
        </section>
  
        <section className="region-strip">
          {regions.slice(0, 12).map((region) => (
            <article className="region-chip" key={region.id}>
              <div>
                <span>{region.name}</span>
                <strong>{region.prospectCount}</strong>
              </div>
              <CompactProgress value={region.scoutedAverage} tone="blue" />
            </article>
          ))}
        </section>
      </div>
    );
  }
  
  function CountryIntel({ country, prospects, onSelectProspect }) {
    const risk = Math.max(
      country.safetyRisk,
      country.politicalRisk,
      country.sourceRisk,
      country.difficulty
    );
  
    return (
      <div className="country-intel-lite">
        <header>
          <div>
            <span>Country Intel</span>
            <h3>{country.name}</h3>
            <p>{country.region || "No region"}</p>
          </div>
  
          <div className={cx("risk-score", riskTone(risk))}>
            <strong>{Math.round(risk)}</strong>
            <span>risk</span>
          </div>
        </header>
  
        <section className="country-metrics-lite">
          <MiniMetric label="Players" value={country.prospectCount} />
          <MiniMetric label="Coverage" value={`${Math.round(country.scoutedAverage)}%`} />
          <MiniMetric label="Cost" value={formatMoney(country.cost)} />
          <MiniMetric label="Effort" value={`${Math.round(country.effort)}%`} />
        </section>
  
        <section className="country-bars-lite">
          <LabeledBar label="Coverage" value={country.scoutedAverage} tone="blue" />
          <LabeledBar label="Logistics" value={country.difficulty} tone="gold" />
          <LabeledBar label="Safety" value={country.safetyRisk} tone="danger" />
          <LabeledBar label="Admin" value={country.politicalRisk} tone="purple" />
        </section>
  
        <section className="country-players-lite">
          <div className="country-players-lite__head">
            <h4>Players</h4>
            <span>{safeArray(prospects).length}</span>
          </div>
  
          {safeArray(prospects).length ? (
            safeArray(prospects)
              .slice(0, 7)
              .map((prospect) => (
                <button
                  key={prospect.id}
                  type="button"
                  className="country-player-row"
                  onClick={() => onSelectProspect(prospect.id)}
                >
                  <b>#{prospect.rank}</b>
                  <span>
                    <strong>{prospect.name}</strong>
                    <small>
                      {prospect.position} · {prospectNeedLabel(prospect)}
                    </small>
                  </span>
                  <i>{Math.round(prospect.scouted)}%</i>
                </button>
              ))
          ) : (
            <EmptyState title="No players" text="No matches returned." icon="PL" />
          )}
        </section>
      </div>
    );
  }
  
  function MiniMetric({ label, value }) {
    return (
      <div className="mini-metric">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    );
  }
  
  function LabeledBar({ label, value, tone = "blue" }) {
    return (
      <div className={cx("labeled-bar", `tone-${tone}`)}>
        <div>
          <span>{label}</span>
          <b>{Math.round(percentage(value))}%</b>
        </div>
        <CompactProgress value={value} tone={tone} showValue={false} />
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Board                                                                      */
  /* -------------------------------------------------------------------------- */
  
  function BoardView({
    prospects,
    filters,
    updateFilter,
    resetFilters,
    options,
    selectedProspectId,
    onSelectProspect,
  }) {
    return (
      <div className="board-view">
        <header className="scout-section-head">
          <div>
            <span>Draft Board</span>
            <h2>Find the gaps</h2>
          </div>
  
          <button type="button" onClick={resetFilters}>
            Reset
          </button>
        </header>
  
        <section className="board-filters">
          <TextField
            label="Search"
            value={filters.search}
            onChange={(value) => updateFilter("search", value)}
            placeholder="Name, country, league..."
          />
  
          <SelectField
            label="Position"
            value={filters.position}
            onChange={(value) => updateFilter("position", value)}
            options={options.positionOptions}
          />
  
          <SelectField
            label="Country"
            value={filters.country}
            onChange={(value) => updateFilter("country", value)}
            options={options.countryOptions}
          />
  
          <SelectField
            label="Region"
            value={filters.region}
            onChange={(value) => updateFilter("region", value)}
            options={options.regionOptions}
          />
  
          <SelectField
            label="League"
            value={filters.league}
            onChange={(value) => updateFilter("league", value)}
            options={options.leagueOptions}
          />
  
          <SelectField
            label="Coverage"
            value={filters.coverage}
            onChange={(value) => updateFilter("coverage", value)}
            options={[
              makeOption("all", "All"),
              makeOption("unknown", "Unknown"),
              makeOption("partial", "Partial"),
              makeOption("solid", "Solid"),
              makeOption("final", "Final"),
            ]}
          />
  
          <SelectField
            label="Sort"
            value={filters.sortKey}
            onChange={(value) => updateFilter("sortKey", value)}
            options={[
              makeOption(SORT_KEYS.RANK, "Rank"),
              makeOption(SORT_KEYS.NAME, "Name"),
              makeOption(SORT_KEYS.POSITION, "Position"),
              makeOption(SORT_KEYS.COUNTRY, "Country"),
              makeOption(SORT_KEYS.REGION, "Region"),
              makeOption(SORT_KEYS.SCOUTED, "Coverage"),
              makeOption(SORT_KEYS.UPSIDE, "Upside"),
              makeOption(SORT_KEYS.RISK, "Risk"),
            ]}
          />
  
          <button
            type="button"
            className={cx("filter-toggle", filters.onlyWatchlist && "is-on")}
            onClick={() => updateFilter("onlyWatchlist", !filters.onlyWatchlist)}
          >
            Watch
          </button>
  
          <button
            type="button"
            className={cx("filter-toggle", filters.onlyNeedsWork && "is-on")}
            onClick={() => updateFilter("onlyNeedsWork", !filters.onlyNeedsWork)}
          >
            Needs Work
          </button>
  
          <button
            type="button"
            className="filter-toggle"
            onClick={() =>
              updateFilter("sortDirection", filters.sortDirection === "asc" ? "desc" : "asc")
            }
          >
            {filters.sortDirection === "asc" ? "Asc" : "Desc"}
          </button>
        </section>
  
        <section className="board-table-wrap">
          <table className="board-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Country</th>
                <th>Projection</th>
                <th>Scouted</th>
                <th>Upside</th>
                <th>Risk</th>
                <th>Status</th>
              </tr>
            </thead>
  
            <tbody>
              {prospects.map((prospect) => (
                <tr
                  key={prospect.id}
                  className={selectedProspectId === prospect.id ? "is-selected" : ""}
                  onClick={() => onSelectProspect(prospect.id)}
                >
                  <td>
                    <span className="rank-badge">#{prospect.rank}</span>
                  </td>
  
                  <td>
                    <div className="board-player-cell">
                      <PersonAvatar
                        src={prospect.headshotUrl}
                        name={prospect.name}
                        label={prospect.position}
                      />
                      <div>
                        <strong>{prospect.name}</strong>
                        <small>
                          {prospect.position} · {prospect.league || "—"}
                        </small>
                      </div>
                    </div>
                  </td>
  
                  <td>
                    <div className="stack-cell">
                      <strong>{prospect.country}</strong>
                      <small>{prospect.region || "—"}</small>
                    </div>
                  </td>
  
                  <td>
                    <span className="soft-pill">{prospect.projection}</span>
                  </td>
  
                  <td>
                    <CompactProgress
                      value={prospect.scouted}
                      tone={coverageTone(prospect.scouted)}
                    />
                  </td>
  
                  <td>
                    <CompactProgress value={prospect.upside} tone="blue" />
                  </td>
  
                  <td>
                    <span className={cx("risk-pill", prospect.riskTone)}>
                      {typeof prospect.risk === "number"
                        ? `${Math.round(prospect.risk)}%`
                        : titleCase(prospect.risk || "—")}
                    </span>
                  </td>
  
                  <td>
                    <span className="soft-pill">{prospectNeedLabel(prospect)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
  
          {!prospects.length ? (
            <EmptyState title="No matches" text="Try fewer filters." icon="SR" />
          ) : null}
        </section>
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Watchlist                                                                  */
  /* -------------------------------------------------------------------------- */
  
  function WatchlistView({ prospects, onSelectProspect }) {
    return (
      <div className="watchlist-view">
        <header className="scout-section-head">
          <div>
            <span>Watchlist</span>
            <h2>Players needing eyes</h2>
          </div>
        </header>
  
        {prospects.length ? (
          <section className="watchlist-grid">
            {prospects.map((prospect) => (
              <button
                key={prospect.id}
                type="button"
                className="watch-card"
                onClick={() => onSelectProspect(prospect.id)}
              >
                <div className="watch-card__top">
                  <span>#{prospect.rank}</span>
                  <b>{prospectNeedLabel(prospect)}</b>
                </div>
  
                <div className="watch-card__main">
                  <PersonAvatar
                    src={prospect.headshotUrl}
                    name={prospect.name}
                    label={prospect.position}
                    size="lg"
                  />
  
                  <div>
                    <strong>{prospect.name}</strong>
                    <small>
                      {prospect.position} · {prospect.country}
                    </small>
                  </div>
                </div>
  
                <div className="watch-card__bars">
                  <LabeledBar label="Scouted" value={prospect.scouted} tone="blue" />
                  <LabeledBar label="Upside" value={prospect.upside} tone="green" />
                </div>
              </button>
            ))}
          </section>
        ) : (
          <EmptyState title="Clear list" text="No urgent files." icon="✓" />
        )}
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Reports                                                                    */
  /* -------------------------------------------------------------------------- */
  
  function ReportsView({ phase, state, prospects, countries, regions, assignments }) {
    const report = useMemo(() => {
      const byPosition = new Map();
      const byCountry = new Map();
  
      prospects.forEach((prospect) => {
        byPosition.set(prospect.position, (byPosition.get(prospect.position) || 0) + 1);
        byCountry.set(prospect.country, (byCountry.get(prospect.country) || 0) + 1);
      });
  
      const topCountries = [...byCountry.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8);
  
      const coverageGaps = countries
        .slice()
        .sort((a, b) => percentage(a.scoutedAverage) - percentage(b.scoutedAverage))
        .slice(0, 8);
  
      const hardTravel = countries
        .slice()
        .sort((a, b) => {
          const br = Math.max(b.safetyRisk, b.politicalRisk, b.sourceRisk, b.difficulty);
          const ar = Math.max(a.safetyRisk, a.politicalRisk, a.sourceRisk, a.difficulty);
          return br - ar;
        })
        .slice(0, 8);
  
      const highUpside = prospects
        .slice()
        .sort((a, b) => percentage(b.upside) - percentage(a.upside))
        .slice(0, 8);
  
      const lowConfidence = prospects
        .slice()
        .sort((a, b) => percentage(a.scouted) - percentage(b.scouted))
        .slice(0, 8);
  
      return {
        byPosition,
        topCountries,
        coverageGaps,
        hardTravel,
        highUpside,
        lowConfidence,
      };
    }, [prospects, countries]);
  
    const avgCoverage = prospects.length
      ? prospects.reduce((sum, prospect) => sum + percentage(prospect.scouted), 0) /
        prospects.length
      : 0;
  
    const finalReady = prospects.filter((prospect) => prospect.scouted >= 90).length;
    const upsideGaps = prospects.filter(
      (prospect) => prospect.upside >= 70 && prospect.scouted < 60
    ).length;
    const weakCountries = countries.filter((country) => country.scoutedAverage < 45).length;
  
    return (
      <div className="reports-view">
        <header className="scout-section-head">
          <div>
            <span>Reports</span>
            <h2>Scouting room notes</h2>
          </div>
        </header>
  
        <section className="report-grid">
          <ReportPanel title="Board Mix" subtitle="Positions">
            <div className="position-report-grid">
              {[...report.byPosition.entries()]
                .sort((a, b) => compareText(a[0], b[0]))
                .map(([position, count]) => (
                  <div className="position-report-card" key={position}>
                    <span>{position}</span>
                    <strong>{count}</strong>
                  </div>
                ))}
            </div>
          </ReportPanel>
  
          <ReportPanel title="Source Countries" subtitle="Volume">
            <ReportList
              items={report.topCountries}
              render={(item) => (
                <Fragment>
                  <span>{item[0]}</span>
                  <strong>{item[1]}</strong>
                </Fragment>
              )}
            />
          </ReportPanel>
  
          <ReportPanel title="Coverage Gaps" subtitle="Weak files">
            <ReportList
              items={report.coverageGaps}
              render={(country) => (
                <Fragment>
                  <span>{country.name}</span>
                  <strong>{Math.round(country.scoutedAverage)}%</strong>
                </Fragment>
              )}
            />
          </ReportPanel>
  
          <ReportPanel title="Hard Travel" subtitle="Risk">
            <ReportList
              items={report.hardTravel}
              render={(country) => (
                <Fragment>
                  <span>{country.name}</span>
                  <strong>
                    {Math.round(
                      Math.max(
                        country.safetyRisk,
                        country.politicalRisk,
                        country.sourceRisk,
                        country.difficulty
                      )
                    )}
                  </strong>
                </Fragment>
              )}
            />
          </ReportPanel>
  
          <ReportPanel title="Upside Swings" subtitle="Debate">
            <ReportList
              items={report.highUpside}
              render={(prospect) => (
                <Fragment>
                  <span>
                    #{prospect.rank} {prospect.name}
                  </span>
                  <strong>{Math.round(prospect.upside)}%</strong>
                </Fragment>
              )}
            />
          </ReportPanel>
  
          <ReportPanel title="Low Confidence" subtitle="Need eyes">
            <ReportList
              items={report.lowConfidence}
              render={(prospect) => (
                <Fragment>
                  <span>
                    #{prospect.rank} {prospect.name}
                  </span>
                  <strong>{Math.round(prospect.scouted)}%</strong>
                </Fragment>
              )}
            />
          </ReportPanel>
  
          <ReportPanel title="Director Memo" subtitle={phaseLabel(phase)} wide>
            <div className="director-memo">
              <div className="memo-stats">
                <span>
                  <b>{Math.round(avgCoverage)}%</b> avg coverage
                </span>
                <span>
                  <b>{finalReady}</b> final-ready
                </span>
                <span>
                  <b>{upsideGaps}</b> upside gaps
                </span>
                <span>
                  <b>{weakCountries}</b> weak countries
                </span>
                <span>
                  <b>{assignments.length}</b> assignments
                </span>
                <span>
                  <b>{regions.length}</b> regions
                </span>
              </div>
  
              <div className="memo-meta">
                <span>Date: {formatDateLike(state?.date || state?.current_date)}</span>
                <span>Season: {state?.season || state?.year || "—"}</span>
              </div>
            </div>
          </ReportPanel>
        </section>
      </div>
    );
  }
  
  function ReportPanel({ title, subtitle, children, wide = false }) {
    return (
      <article className={cx("report-panel", wide && "is-wide")}>
        <div className="report-panel__head">
          <h3>{title}</h3>
          <span>{subtitle}</span>
        </div>
        {children}
      </article>
    );
  }
  
  function ReportList({ items, render }) {
    const list = safeArray(items);
  
    if (!list.length) {
      return <EmptyState title="No data" text="Nothing returned." icon="—" />;
    }
  
    return (
      <div className="report-list">
        {list.map((item, index) => (
          <div className="report-row" key={index}>
            <b>{index + 1}</b>
            {render(item)}
          </div>
        ))}
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Scouts                                                                     */
  /* -------------------------------------------------------------------------- */
  
  function ScoutsView({ scouts, assignments, selectedScoutId, setSelectedScoutId }) {
    const assignmentByScout = useMemo(() => {
      const map = new Map();
  
      assignments.forEach((assignment) => {
        const key = assignment.scoutId || "";
        if (!key) return;
        map.set(key, [...(map.get(key) || []), assignment]);
      });
  
      return map;
    }, [assignments]);
  
    return (
      <div className="scouts-view">
        <header className="scout-section-head">
          <div>
            <span>Scouts</span>
            <h2>Manage workload</h2>
          </div>
        </header>
  
        {scouts.length ? (
          <section className="scout-staff-grid">
            {scouts.map((scout) => {
              const scoutAssignments = assignmentByScout.get(scout.id) || EMPTY_ARRAY;
  
              return (
                <button
                  key={scout.id}
                  type="button"
                  className={cx("staff-card", selectedScoutId === scout.id && "is-selected")}
                  onClick={() => setSelectedScoutId(scout.id)}
                >
                  <div className="staff-card__top">
                    <PersonAvatar
                      src={scout.headshotUrl}
                      name={scout.name}
                      label={scout.role}
                      size="lg"
                    />
  
                    <div>
                      <strong>{scout.name}</strong>
                      <span>{scout.role}</span>
                    </div>
                  </div>
  
                  <div className="staff-card__meta">
                    {scout.region ? <span>{scout.region}</span> : null}
                    {scout.country ? <span>{scout.country}</span> : null}
                    <span>{titleCase(scout.availability)}</span>
                  </div>
  
                  <div className="staff-card__bars">
                    <LabeledBar label="Load" value={scout.workload} tone="gold" />
                    <LabeledBar label="Accuracy" value={scout.accuracy} tone="blue" />
                    <LabeledBar label="Character" value={scout.character} tone="purple" />
                  </div>
  
                  <div className="staff-card__footer">
                    <span>{scoutAssignments.length} active</span>
                    <span>{safeArray(scout.languages).slice(0, 2).join(", ") || "—"}</span>
                  </div>
                </button>
              );
            })}
          </section>
        ) : (
          <EmptyState title="No scouts" text="Staff data not returned." icon="SC" />
        )}
      </div>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Player                                                                     */
  /* -------------------------------------------------------------------------- */
  
  function PlayerView({
    prospect,
    country,
    prospects,
    phase,
    onSelectProspect,
    runProspectAction,
    busyAction,
  }) {
    if (!prospect) {
      return (
        <div className="player-view-empty">
          <EmptyState title="Pick a player" text="Open the board or map." icon="PF" />
  
          <div className="quick-player-list">
            {prospects.slice(0, 8).map((p) => (
              <button key={p.id} type="button" onClick={() => onSelectProspect(p.id)}>
                <b>#{p.rank}</b>
                <span>{p.name}</span>
                <small>{p.position}</small>
              </button>
            ))}
          </div>
        </div>
      );
    }
  
    const skills = Object.entries(safeObject(prospect.skills))
      .filter(([, value]) => Number.isFinite(Number(value)))
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 12);
  
    return (
      <div className="player-view">
        <section className="player-hero-card">
          <div className="player-identity">
            <PersonAvatar
              src={prospect.headshotUrl}
              name={prospect.name}
              label={prospect.position}
              size="xl"
            />
  
            <div>
              <span>Prospect File</span>
              <h2>{prospect.name}</h2>
              <p>
                #{prospect.rank} · {prospect.position} · {prospect.country}
              </p>
  
              <div className="player-tags">
                <span>{prospect.projection}</span>
                <span>{prospectNeedLabel(prospect)}</span>
                <span>{prospect.draftStock}</span>
              </div>
            </div>
          </div>
  
          <div className="player-rank-card">
            <span>Rank</span>
            <strong>#{prospect.rank}</strong>
            <small>{Math.round(prospect.scouted)}% scouted</small>
          </div>
        </section>
  
        <section className="player-file-grid">
          <article className="scout-card player-card-wide">
            <div className="scout-card-head">
              <h3>Confidence</h3>
              <span>{phaseLabel(phase)}</span>
            </div>
  
            <div className="ring-grid">
              <ProgressRing
                value={prospect.scouted}
                label="Scouted"
                tone={coverageTone(prospect.scouted)}
              />
              <ProgressRing value={prospect.upside} label="Upside" tone="blue" />
              <ProgressRing value={prospect.floor} label="Floor" tone="green" />
              <ProgressRing value={prospect.volatility} label="Volatility" tone="gold" />
            </div>
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Bio</h3>
            </div>
  
            <dl className="player-bio">
              <BioLine label="Age" value={prospect.age ?? "—"} />
              <BioLine label="Height" value={prospect.height || "—"} />
              <BioLine label="Weight" value={prospect.weight || "—"} />
              <BioLine label="Hand" value={prospect.handedness || "—"} />
              <BioLine label="Birth" value={formatDateLike(prospect.birthdate)} />
              <BioLine label="League" value={prospect.league || "—"} />
            </dl>
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Country</h3>
            </div>
  
            {country ? (
              <div className="country-mini">
                <strong>{country.name}</strong>
                <span>{country.region || "—"}</span>
                <LabeledBar label="Coverage" value={country.scoutedAverage} tone="blue" />
                <small>{formatMoney(country.cost)} trip</small>
              </div>
            ) : (
              <EmptyState title="No country" text="World data missing." icon="GL" />
            )}
          </article>
  
          <article className="scout-card player-card-wide">
            <div className="scout-card-head">
              <h3>Skills</h3>
              <span>{skills.length}</span>
            </div>
  
            {skills.length ? (
              <div className="skill-grid">
                {skills.map(([key, value]) => (
                  <div className="skill-row" key={key}>
                    <span>{titleCase(key)}</span>
                    <CompactProgress value={value} tone="blue" />
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState title="No ratings" text="Scout to unlock." icon="SK" />
            )}
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Traits</h3>
            </div>
  
            {safeArray(prospect.traits).length ? (
              <div className="tag-cloud">
                {safeArray(prospect.traits).slice(0, 12).map((trait, index) => (
                  <span key={`${trait}-${index}`}>{titleCase(trait)}</span>
                ))}
              </div>
            ) : (
              <EmptyState title="Unknown" text="No traits yet." icon="TR" />
            )}
          </article>
  
          <article className="scout-card">
            <div className="scout-card-head">
              <h3>Flags</h3>
            </div>
  
            {safeArray(prospect.redFlags).length ? (
              <div className="flag-list">
                {safeArray(prospect.redFlags).slice(0, 8).map((flag, index) => (
                  <span key={`${flag}-${index}`}>{titleCase(flag)}</span>
                ))}
              </div>
            ) : (
              <div className="clean-file">No major flags</div>
            )}
          </article>
  
          <article className="scout-card player-card-wide">
            <div className="scout-card-head">
              <h3>Draft Actions</h3>
              <span>Interactive</span>
            </div>
  
            <div className="draft-action-grid">
              <DraftActionButton
                title="Interview"
                label="Personality read"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.INTERVIEW)}
              />
              <DraftActionButton
                title="Dinner"
                label="Final fit check"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.DINNER)}
              />
              <DraftActionButton
                title="Combine"
                label="Testing data"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.COMBINE)}
              />
              <DraftActionButton
                title="Workout"
                label="One more look"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.PRIVATE_WORKOUT)}
              />
              <DraftActionButton
                title="Medical"
                label="Risk review"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.MEDICAL)}
              />
              <DraftActionButton
                title="Focus"
                label="Priority file"
                disabled={busyAction}
                onClick={() => runProspectAction(SCOUTING_ACTIONS.PLAYER_FOCUS)}
              />
            </div>
          </article>
  
          <article className="scout-card player-card-wide">
            <div className="scout-card-head">
              <h3>Notes</h3>
              <span>{safeArray(prospect.notes).length}</span>
            </div>
  
            {safeArray(prospect.notes).length ? (
              <div className="note-list">
                {safeArray(prospect.notes)
                  .slice(0, 10)
                  .map((note, index) => (
                    <ScoutNote key={`note-${index}`} note={note} index={index} />
                  ))}
              </div>
            ) : (
              <EmptyState title="No notes" text="Assign a scout." icon="NT" />
            )}
          </article>
        </section>
      </div>
    );
  }
  
  function BioLine({ label, value }) {
    return (
      <div>
        <dt>{label}</dt>
        <dd>{value}</dd>
      </div>
    );
  }
  
  function DraftActionButton({ title, label, disabled, onClick }) {
    return (
      <button type="button" className="draft-action-button" disabled={disabled} onClick={onClick}>
        <strong>{title}</strong>
        <span>{label}</span>
      </button>
    );
  }
  
  function ScoutNote({ note, index }) {
    const obj = typeof note === "object" && note !== null ? note : { text: String(note) };
  
    const text = stringOr(obj.text || obj.note || obj.summary || obj.body, "No note.");
    const author = stringOr(obj.author || obj.scout || obj.scout_name || obj.scoutName, "Scout");
    const date = obj.date || obj.created_at || obj.createdAt || obj.game_date || obj.gameDate;
  
    return (
      <article className="scout-note">
        <b>{index + 1}</b>
        <div>
          <header>
            <strong>{author}</strong>
            <span>{formatDateLike(date)}</span>
          </header>
          <p>{text}</p>
        </div>
      </article>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Command panel                                                              */
  /* -------------------------------------------------------------------------- */
  
  function CommandPanel({
    scouts,
    selectedScout,
    selectedScoutId,
    setSelectedScoutId,
    selectedAction,
    setSelectedAction,
    selectedIntensity,
    setSelectedIntensity,
    selectedCountry,
    selectedProspect,
    estimatedCost,
    phase,
    busyAction,
    assignScouting,
  }) {
    const target = selectedProspect
      ? {
          type: "Prospect",
          name: selectedProspect.name,
          sub: `${selectedProspect.position} · ${Math.round(selectedProspect.scouted)}%`,
        }
      : selectedCountry
        ? {
            type: "Country",
            name: selectedCountry.name,
            sub: `${selectedCountry.prospectCount} players`,
          }
        : {
            type: "Target",
            name: "Pick one",
            sub: "Map or board",
          };
  
    return (
      <section className="command-panel">
        <header className="command-panel__head">
          <span>Assignment</span>
          <h3>Scout Command</h3>
        </header>
  
        <article className="command-target">
          <span>{target.type}</span>
          <strong>{target.name}</strong>
          <small>{target.sub}</small>
        </article>
  
        <SelectField
          label="Scout"
          value={selectedScoutId}
          onChange={setSelectedScoutId}
          options={[
            makeOption("", "Auto Assign"),
            ...safeArray(scouts).map((scout) =>
              makeOption(scout.id, `${scout.name} · ${scout.role}`)
            ),
          ]}
        />
  
        {selectedScout ? (
          <article className="selected-scout">
            <div className="selected-scout__top">
              <PersonAvatar
                src={selectedScout.headshotUrl}
                name={selectedScout.name}
                label={selectedScout.role}
              />
              <div>
                <strong>{selectedScout.name}</strong>
                <span>{selectedScout.role}</span>
              </div>
            </div>
  
            <LabeledBar label="Load" value={selectedScout.workload} tone="gold" />
            <LabeledBar label="Accuracy" value={selectedScout.accuracy} tone="blue" />
            <LabeledBar label="Character" value={selectedScout.character} tone="purple" />
  
            <div className="selected-scout__chips">
              {selectedScout.region ? <span>{selectedScout.region}</span> : null}
              {selectedScout.country ? <span>{selectedScout.country}</span> : null}
              {safeArray(selectedScout.languages)
                .slice(0, 3)
                .map((language) => (
                  <span key={language}>{language}</span>
                ))}
            </div>
          </article>
        ) : null}
  
        <SelectField
          label="Action"
          value={selectedAction}
          onChange={setSelectedAction}
          options={[
            makeOption(SCOUTING_ACTIONS.PLAYER_FOCUS, "Player Focus"),
            makeOption(SCOUTING_ACTIONS.REGION_SWEEP, "Region Sweep"),
            makeOption(SCOUTING_ACTIONS.LIVE_VIEWING, "Live View"),
            makeOption(SCOUTING_ACTIONS.VIDEO_REVIEW, "Video"),
            makeOption(SCOUTING_ACTIONS.CHARACTER_CHECK, "Character"),
            makeOption(SCOUTING_ACTIONS.ANALYTICS, "Analytics"),
            makeOption(SCOUTING_ACTIONS.INTERVIEW, "Interview"),
            makeOption(SCOUTING_ACTIONS.DINNER, "Dinner"),
            makeOption(SCOUTING_ACTIONS.COMBINE, "Combine"),
            makeOption(SCOUTING_ACTIONS.PRIVATE_WORKOUT, "Workout"),
            makeOption(SCOUTING_ACTIONS.MEDICAL, "Medical"),
          ]}
        />
  
        <div className="intensity-picker">
          <span>Intensity</span>
          <div>
            {Object.values(SCOUTING_INTENSITY).map((value) => (
              <button
                key={value}
                type="button"
                className={selectedIntensity === value ? "is-active" : ""}
                onClick={() => setSelectedIntensity(value)}
              >
                {intensityLabel(value)}
              </button>
            ))}
          </div>
        </div>
  
        <article className="cost-card">
          <span>Estimated Cost</span>
          <strong>{formatMoney(estimatedCost)}</strong>
          <small>{phaseCue(phase)}</small>
        </article>
  
        <button
          type="button"
          className="assign-button"
          disabled={Boolean(busyAction)}
          onClick={assignScouting}
        >
          {busyAction ? "Sending..." : "Assign Scout"}
        </button>
      </section>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* Assignments helper cards                                                   */
  /* -------------------------------------------------------------------------- */
  
  function AssignmentCard({ assignment, onCancel, busyAction }) {
    const targetName =
      assignment.prospect?.name ||
      assignment.country?.name ||
      stringOr(assignment.targetId, "Unknown");
  
    const scoutName = assignment.scout?.name || "Unassigned";
  
    return (
      <article className="assignment-card">
        <header>
          <div>
            <span>{actionLabel(assignment.action)}</span>
            <h3>{targetName}</h3>
            <p>{scoutName}</p>
          </div>
  
          <b>{titleCase(assignment.status)}</b>
        </header>
  
        <LabeledBar label="Progress" value={assignment.progress} tone="blue" />
  
        <footer>
          <span>{formatMoney(assignment.cost)}</span>
          <button type="button" disabled={busyAction} onClick={() => onCancel(assignment)}>
            Cancel
          </button>
        </footer>
      </article>
    );
  }
  
  /* -------------------------------------------------------------------------- */
  /* CSS goes in next chunk                                                     */
  /* -------------------------------------------------------------------------- */

  function ScoutingStyles() {
    return (
      <style>{`
        .scout-root {
          --scout-bg: #030b13;
          --scout-bg-2: #061423;
          --scout-panel: rgba(7, 20, 34, 0.94);
          --scout-panel-2: rgba(10, 29, 47, 0.92);
          --scout-panel-3: rgba(14, 39, 62, 0.82);
          --scout-line: rgba(150, 205, 235, 0.16);
          --scout-line-2: rgba(150, 205, 235, 0.28);
          --scout-text: #eef8ff;
          --scout-muted: rgba(220, 237, 248, 0.66);
          --scout-faint: rgba(220, 237, 248, 0.46);
          --scout-blue: #39b9ff;
          --scout-cyan: #22e2ef;
          --scout-green: #5cf29c;
          --scout-gold: #ffd166;
          --scout-purple: #b892ff;
          --scout-orange: #ff9f43;
          --scout-red: #ff4f63;
          --scout-radius: 18px;
          --scout-radius-lg: 24px;
          --scout-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
  
          min-height: 100vh;
          width: 100%;
          display: grid;
          grid-template-columns: 86px minmax(0, 1fr);
          color: var(--scout-text);
          background:
            radial-gradient(circle at 18% 0%, rgba(57, 185, 255, 0.12), transparent 30%),
            radial-gradient(circle at 90% 12%, rgba(255, 79, 99, 0.09), transparent 26%),
            linear-gradient(180deg, #061522 0%, #020810 100%);
          font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
          overflow: hidden;
        }
  
        .scout-root *,
        .scout-root *::before,
        .scout-root *::after {
          box-sizing: border-box;
        }
  
        .scout-root button,
        .scout-root input,
        .scout-root select {
          font: inherit;
        }
  
        .scout-root button {
          cursor: pointer;
        }
  
        .scout-root button:disabled {
          cursor: not-allowed;
          opacity: 0.55;
        }
  
        .scout-root img {
          display: block;
          max-width: 100%;
        }
  
        .scout-sidebar {
          min-height: 100vh;
          background:
            linear-gradient(180deg, rgba(6, 17, 28, 0.98), rgba(2, 8, 15, 0.98));
          border-right: 1px solid var(--scout-line);
          display: flex;
          flex-direction: column;
          align-items: stretch;
        }
  
        .scout-home-button,
        .scout-settings,
        .scout-side-button {
          border: 0;
          background: transparent;
          color: var(--scout-muted);
        }
  
        .scout-home-button {
          height: 92px;
          display: grid;
          place-items: center;
          border-bottom: 1px solid var(--scout-line);
        }
  
        .scout-home-button span {
          width: 36px;
          height: 40px;
          display: grid;
          place-items: center;
          color: var(--scout-text);
          border: 1px solid rgba(238, 248, 255, 0.44);
          clip-path: polygon(50% 0, 92% 18%, 92% 72%, 50% 100%, 8% 72%, 8% 18%);
          font-weight: 900;
        }
  
        .scout-side-nav {
          display: grid;
          gap: 4px;
          padding: 14px 0;
        }
  
        .scout-side-button {
          min-height: 66px;
          position: relative;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 4px;
          transition: 160ms ease;
        }
  
        .scout-side-button span {
          font-size: 22px;
          line-height: 1;
        }
  
        .scout-side-button small {
          font-size: 10px;
          font-weight: 900;
          letter-spacing: 0.02em;
        }
  
        .scout-side-button:hover,
        .scout-side-button.is-active {
          color: var(--scout-cyan);
          background: linear-gradient(90deg, rgba(34, 226, 239, 0.13), transparent);
        }
  
        .scout-side-button.is-active::before {
          content: "";
          position: absolute;
          left: 0;
          top: 12px;
          bottom: 12px;
          width: 3px;
          border-radius: 999px;
          background: var(--scout-cyan);
          box-shadow: 0 0 18px rgba(34, 226, 239, 0.8);
        }
  
        .scout-settings {
          margin-top: auto;
          height: 82px;
          display: grid;
          place-items: center;
          border-top: 1px solid var(--scout-line);
        }
  
        .scout-settings span {
          font-size: 22px;
        }
  
        .scout-settings:hover {
          color: var(--scout-text);
        }
  
        .scout-main {
          min-width: 0;
          height: 100vh;
          overflow: auto;
          padding: 22px;
        }
  
        .scout-main::-webkit-scrollbar,
        .scout-workspace::-webkit-scrollbar,
        .scout-command-rail::-webkit-scrollbar,
        .board-table-wrap::-webkit-scrollbar {
          width: 10px;
          height: 10px;
        }
  
        .scout-main::-webkit-scrollbar-thumb,
        .scout-workspace::-webkit-scrollbar-thumb,
        .scout-command-rail::-webkit-scrollbar-thumb,
        .board-table-wrap::-webkit-scrollbar-thumb {
          background: rgba(150, 205, 235, 0.22);
          border-radius: 999px;
        }
  
        .scout-screen-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 18px;
          margin-bottom: 14px;
        }
  
        .scout-title-block {
          min-width: 0;
          display: flex;
          align-items: center;
          gap: 16px;
        }
  
        .scout-team-logo {
          flex: 0 0 auto;
          display: grid;
          place-items: center;
          overflow: hidden;
          border-radius: 18px;
          border: 1px solid var(--scout-line-2);
          background:
            radial-gradient(circle at 34% 20%, rgba(255, 255, 255, 0.18), transparent 36%),
            rgba(255, 255, 255, 0.06);
          box-shadow: inset 0 0 28px rgba(255, 255, 255, 0.04);
        }
  
        .scout-team-logo--md {
          width: 70px;
          height: 70px;
        }
  
        .scout-team-logo--sm {
          width: 46px;
          height: 46px;
          border-radius: 14px;
        }
  
        .scout-team-logo--lg {
          width: 92px;
          height: 92px;
          border-radius: 24px;
        }
  
        .scout-team-logo img {
          width: 82%;
          height: 82%;
          object-fit: contain;
        }
  
        .scout-team-logo span {
          font-size: 17px;
          font-weight: 1000;
          letter-spacing: 0.06em;
          color: var(--scout-cyan);
        }
  
        .scout-title-block p {
          margin: 0 0 2px;
          color: var(--scout-muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          font-weight: 1000;
        }
  
        .scout-title-block h1 {
          margin: 0;
          font-size: clamp(30px, 3.1vw, 48px);
          line-height: 0.95;
          letter-spacing: 0.045em;
          text-transform: uppercase;
        }
  
        .scout-title-block span {
          display: inline-flex;
          margin-top: 7px;
          color: var(--scout-red);
          font-size: 14px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
  
        .scout-head-actions {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 8px;
        }
  
        .scout-head-actions button,
        .scout-section-head button,
        .report-panel__head button {
          min-height: 38px;
          padding: 0 14px;
          border-radius: 10px;
          border: 1px solid var(--scout-line);
          background: rgba(8, 24, 39, 0.88);
          color: var(--scout-text);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          transition: 160ms ease;
        }
  
        .scout-head-actions button:hover,
        .scout-section-head button:hover,
        .report-panel__head button:hover {
          border-color: rgba(34, 226, 239, 0.34);
          background: rgba(34, 226, 239, 0.1);
        }
  
        .scout-top-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin-bottom: 12px;
        }
  
        .scout-stat-tile {
          min-height: 82px;
          border: 1px solid var(--scout-line);
          border-radius: 16px;
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.02)),
            rgba(7, 21, 35, 0.86);
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 14px;
          overflow: hidden;
          position: relative;
        }
  
        .scout-stat-tile::after {
          content: "";
          position: absolute;
          right: -32px;
          top: -38px;
          width: 100px;
          height: 100px;
          border-radius: 999px;
          background: var(--tile-color);
          opacity: 0.15;
          filter: blur(14px);
        }
  
        .scout-stat-tile__icon {
          width: 42px;
          height: 42px;
          border-radius: 14px;
          display: grid;
          place-items: center;
          color: var(--tile-color);
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.08);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.04em;
        }
  
        .scout-stat-tile > div:last-child {
          min-width: 0;
          position: relative;
          z-index: 1;
        }
  
        .scout-stat-tile span {
          display: block;
          color: var(--scout-muted);
          font-size: 10px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
  
        .scout-stat-tile strong {
          display: block;
          margin-top: 2px;
          color: var(--scout-text);
          font-size: 23px;
          line-height: 1;
        }
  
        .scout-stat-tile small {
          display: block;
          margin-top: 6px;
          color: var(--scout-faint);
          font-size: 10px;
          font-weight: 800;
          text-transform: uppercase;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
  
        .scout-stat-tile.tone-blue {
          --tile-color: var(--scout-blue);
        }
  
        .scout-stat-tile.tone-cyan {
          --tile-color: var(--scout-cyan);
        }
  
        .scout-stat-tile.tone-green {
          --tile-color: var(--scout-green);
        }
  
        .scout-stat-tile.tone-gold {
          --tile-color: var(--scout-gold);
        }
  
        .scout-stat-tile.tone-purple {
          --tile-color: var(--scout-purple);
        }
  
        .scout-stat-tile.tone-danger {
          --tile-color: var(--scout-red);
        }
  
        .scout-alert-stack {
          display: grid;
          gap: 8px;
          margin-bottom: 12px;
        }
  
        .scout-alert {
          padding: 12px 14px;
          border-radius: 14px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.055);
          color: var(--scout-text);
          font-size: 13px;
          line-height: 1.4;
        }
  
        .scout-alert.is-success {
          border-color: rgba(92, 242, 156, 0.38);
          background: rgba(92, 242, 156, 0.1);
        }
  
        .scout-alert.is-danger {
          border-color: rgba(255, 79, 99, 0.42);
          background: rgba(255, 79, 99, 0.1);
        }
  
        .scout-alert.is-warn {
          border-color: rgba(255, 209, 102, 0.38);
          background: rgba(255, 209, 102, 0.1);
        }
  
        .scout-tabs {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          border: 1px solid var(--scout-line);
          border-radius: 16px;
          overflow: hidden;
          margin-bottom: 12px;
          background: rgba(7, 21, 35, 0.84);
        }
  
        .scout-tabs button {
          min-height: 46px;
          border: 0;
          border-bottom: 3px solid transparent;
          background: transparent;
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          transition: 160ms ease;
        }
  
        .scout-tabs button:hover {
          color: var(--scout-text);
          background: rgba(255, 255, 255, 0.035);
        }
  
        .scout-tabs button.is-active {
          color: var(--scout-text);
          border-bottom-color: var(--scout-red);
          background: linear-gradient(180deg, rgba(255, 79, 99, 0.11), rgba(255, 79, 99, 0.025));
        }
  
        .scout-layout {
          display: grid;
          grid-template-columns: minmax(0, 1fr);
          gap: 12px;
          align-items: start;
        }
  
        .scout-layout.has-panel {
          grid-template-columns: minmax(0, 1fr) 360px;
        }
  
        .scout-workspace {
          min-width: 0;
          max-height: calc(100vh - 244px);
          overflow: auto;
          border: 1px solid var(--scout-line);
          border-radius: var(--scout-radius);
          background:
            linear-gradient(180deg, rgba(9, 26, 42, 0.94), rgba(4, 13, 23, 0.94));
          box-shadow: var(--scout-shadow);
          padding: 16px;
        }
  
        .scout-command-rail {
          min-width: 0;
          max-height: calc(100vh - 244px);
          overflow: auto;
        }
  
        .scout-card,
        .command-panel,
        .report-panel {
          border: 1px solid var(--scout-line);
          border-radius: var(--scout-radius);
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.02)),
            rgba(8, 23, 37, 0.82);
          box-shadow: 0 18px 50px rgba(0, 0, 0, 0.2);
        }
  
        .scout-card {
          padding: 16px;
          min-width: 0;
        }
  
        .scout-card--wide,
        .player-card-wide {
          grid-column: 1 / -1;
        }
  
        .scout-card-head,
        .scout-section-head,
        .report-panel__head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 14px;
        }
  
        .scout-card-head h3,
        .scout-section-head h2,
        .report-panel__head h3 {
          margin: 0;
          color: var(--scout-text);
          text-transform: uppercase;
          letter-spacing: 0.045em;
        }
  
        .scout-card-head h3 {
          font-size: 17px;
        }
  
        .scout-section-head h2 {
          margin-top: 4px;
          font-size: 20px;
        }
  
        .report-panel__head h3 {
          font-size: 17px;
        }
  
        .scout-card-head span,
        .scout-section-head span,
        .report-panel__head span {
          color: var(--scout-muted);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }
  
        .scout-card-head button {
          min-height: 32px;
          padding: 0 10px;
          border-radius: 9px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.055);
          color: var(--scout-text);
          font-size: 10px;
          font-weight: 1000;
          text-transform: uppercase;
        }
  
        .scout-overview-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
          gap: 14px;
        }
  
        .scout-map-card {
          min-height: 360px;
        }
  
        .mini-world {
          position: relative;
          min-height: 300px;
          overflow: hidden;
          border-radius: 18px;
          border: 1px solid rgba(150, 205, 235, 0.12);
          background:
            radial-gradient(circle at 46% 48%, rgba(57, 185, 255, 0.15), transparent 42%),
            radial-gradient(ellipse at 18% 38%, rgba(92, 242, 156, 0.08), transparent 26%),
            radial-gradient(ellipse at 66% 42%, rgba(92, 242, 156, 0.08), transparent 25%),
            rgba(0, 0, 0, 0.16);
        }
  
        .mini-world__grid {
          position: absolute;
          inset: 0;
          opacity: 0.35;
          background-image:
            linear-gradient(rgba(255, 255, 255, 0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px);
          background-size: 28px 28px;
          mask-image: radial-gradient(circle at 50% 50%, black 0%, transparent 76%);
        }
  
        .mini-world-pin {
          position: absolute;
          width: 14px;
          height: 14px;
          padding: 0;
          transform: translate(-50%, -50%);
          border: 0;
          border-radius: 999px;
          background: transparent;
        }
  
        .mini-world-pin span {
          position: absolute;
          inset: 3px;
          border-radius: 999px;
          background: var(--scout-blue);
          box-shadow: 0 0 14px rgba(57, 185, 255, 0.8);
        }
  
        .mini-world-pin::before {
          content: "";
          position: absolute;
          inset: -5px;
          border-radius: 999px;
          border: 1px solid rgba(57, 185, 255, 0.35);
          animation: scoutPulse 1.8s ease-out infinite;
        }
  
        .mini-world-pin.is-good span {
          background: var(--scout-green);
          box-shadow: 0 0 14px rgba(92, 242, 156, 0.85);
        }
  
        .mini-world-pin.is-gap span {
          background: var(--scout-gold);
          box-shadow: 0 0 14px rgba(255, 209, 102, 0.85);
        }
  
        .mini-world-pin.is-risk span {
          background: var(--scout-red);
          box-shadow: 0 0 14px rgba(255, 79, 99, 0.85);
        }
  
        @keyframes scoutPulse {
          from {
            opacity: 0.8;
            transform: scale(0.6);
          }
  
          to {
            opacity: 0;
            transform: scale(1.8);
          }
        }
  
        .scout-map-legend,
        .globe-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 12px;
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 800;
        }
  
        .scout-map-legend span,
        .globe-legend span {
          display: inline-flex;
          align-items: center;
          gap: 7px;
        }
  
        .dot {
          width: 9px;
          height: 9px;
          border-radius: 999px;
          display: inline-block;
        }
  
        .dot.good {
          background: var(--scout-green);
          box-shadow: 0 0 10px rgba(92, 242, 156, 0.75);
        }
  
        .dot.warn {
          background: var(--scout-gold);
          box-shadow: 0 0 10px rgba(255, 209, 102, 0.75);
        }
  
        .dot.danger {
          background: var(--scout-red);
          box-shadow: 0 0 10px rgba(255, 79, 99, 0.75);
        }
  
        .position-need-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }
  
        .position-need-card {
          min-height: 112px;
          border-radius: 16px;
          border: 1px solid rgba(150, 205, 235, 0.14);
          background: rgba(0, 0, 0, 0.16);
          padding: 14px;
          display: grid;
          align-content: space-between;
          gap: 10px;
        }
  
        .position-need-card span {
          color: var(--scout-muted);
          font-size: 13px;
          font-weight: 1000;
        }
  
        .position-need-card strong {
          font-size: 32px;
          line-height: 1;
        }
  
        .top-prospect-row {
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 10px;
        }
  
        .prospect-tile {
          min-width: 0;
          min-height: 212px;
          position: relative;
          display: grid;
          justify-items: center;
          align-content: start;
          gap: 10px;
          padding: 14px;
          border-radius: 18px;
          border: 1px solid rgba(150, 205, 235, 0.14);
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.015)),
            rgba(0, 0, 0, 0.14);
          color: var(--scout-text);
          text-align: center;
          overflow: hidden;
          transition: 160ms ease;
        }
  
        .prospect-tile:hover {
          transform: translateY(-2px);
          border-color: rgba(57, 185, 255, 0.34);
          background: rgba(57, 185, 255, 0.1);
        }
  
        .prospect-rank {
          position: absolute;
          left: 10px;
          top: 10px;
          min-width: 30px;
          height: 28px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          border: 1px solid rgba(255, 79, 99, 0.36);
          color: var(--scout-red);
          background: rgba(255, 79, 99, 0.1);
          font-size: 12px;
          font-weight: 1000;
        }
  
        .scout-avatar {
          display: grid;
          place-items: center;
          overflow: hidden;
          flex: 0 0 auto;
          border-radius: 16px;
          border: 1px solid rgba(150, 205, 235, 0.18);
          background:
            radial-gradient(circle at 34% 24%, rgba(255, 255, 255, 0.22), transparent 34%),
            linear-gradient(135deg, rgba(57, 185, 255, 0.28), rgba(184, 146, 255, 0.12));
        }
  
        .scout-avatar--md {
          width: 48px;
          height: 48px;
        }
  
        .scout-avatar--lg {
          width: 76px;
          height: 76px;
          border-radius: 22px;
        }
  
        .scout-avatar--xl {
          width: 110px;
          height: 110px;
          border-radius: 30px;
        }
  
        .scout-avatar img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
  
        .scout-avatar span {
          font-size: 13px;
          font-weight: 1000;
          letter-spacing: 0.04em;
          color: var(--scout-text);
        }
  
        .scout-avatar--lg span {
          font-size: 18px;
        }
  
        .scout-avatar--xl span {
          font-size: 27px;
        }
  
        .prospect-tile__body {
          min-width: 0;
          width: 100%;
        }
  
        .prospect-tile__body strong,
        .prospect-tile__body small {
          display: block;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
  
        .prospect-tile__body strong {
          font-size: 14px;
          text-transform: uppercase;
        }
  
        .prospect-tile__body small {
          margin-top: 4px;
          color: var(--scout-muted);
          font-size: 12px;
        }
  
        .prospect-grade {
          width: 68px;
          height: 68px;
          display: grid;
          place-items: center;
          align-content: center;
          border-radius: 999px;
          border: 5px solid currentColor;
          background: rgba(0, 0, 0, 0.18);
        }
  
        .prospect-grade strong {
          font-size: 22px;
          line-height: 1;
        }
  
        .prospect-grade span {
          font-size: 11px;
          font-weight: 900;
          color: var(--scout-muted);
        }
  
        .prospect-grade.elite,
        .prospect-grade.good {
          color: var(--scout-green);
        }
  
        .prospect-grade.watch {
          color: var(--scout-blue);
        }
  
        .prospect-grade.warn {
          color: var(--scout-gold);
        }
  
        .prospect-grade.danger {
          color: var(--scout-red);
        }
  
        .compact-country-list {
          display: grid;
          gap: 8px;
        }
  
        .compact-country-list button {
          min-height: 44px;
          display: grid;
          grid-template-columns: 34px minmax(0, 1fr) auto;
          align-items: center;
          gap: 10px;
          border: 1px solid rgba(150, 205, 235, 0.12);
          border-radius: 13px;
          background: rgba(0, 0, 0, 0.13);
          color: var(--scout-text);
          text-align: left;
          padding: 8px 10px;
          transition: 150ms ease;
        }
  
        .compact-country-list button:hover {
          transform: translateX(2px);
          border-color: rgba(57, 185, 255, 0.3);
          background: rgba(57, 185, 255, 0.08);
        }
  
        .compact-country-list b {
          width: 24px;
          height: 24px;
          display: grid;
          place-items: center;
          border-radius: 7px;
          color: var(--scout-red);
          background: rgba(255, 79, 99, 0.1);
          border: 1px solid rgba(255, 79, 99, 0.28);
          font-size: 12px;
        }
  
        .compact-country-list span {
          min-width: 0;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .compact-country-list i {
          font-style: normal;
          font-weight: 1000;
        }
  
        .compact-country-list i.warn {
          color: var(--scout-gold);
        }
  
        .compact-country-list i.danger {
          color: var(--scout-red);
        }
  
        .overview-assignment-strip {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }
  
        .assignment-mini {
          min-height: 104px;
          border-radius: 16px;
          border: 1px solid rgba(150, 205, 235, 0.12);
          background: rgba(0, 0, 0, 0.13);
          padding: 13px;
          display: grid;
          align-content: space-between;
          gap: 10px;
        }
  
        .assignment-mini span {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
  
        .assignment-mini strong {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          font-size: 16px;
        }
  
        .scout-compact-progress {
          display: grid;
          grid-template-columns: 42px minmax(70px, 1fr);
          gap: 8px;
          align-items: center;
        }
  
        .scout-compact-progress span {
          color: var(--progress-color);
          font-size: 12px;
          font-weight: 1000;
        }
  
        .scout-compact-progress div {
          height: 8px;
          border-radius: 999px;
          overflow: hidden;
          background: rgba(255, 255, 255, 0.08);
        }
  
        .scout-compact-progress i {
          display: block;
          height: 100%;
          border-radius: inherit;
          background: var(--progress-color);
          box-shadow: 0 0 12px var(--progress-glow);
        }
  
        .scout-compact-progress.tone-blue {
          --progress-color: var(--scout-blue);
          --progress-glow: rgba(57, 185, 255, 0.62);
        }
  
        .scout-compact-progress.tone-cyan {
          --progress-color: var(--scout-cyan);
          --progress-glow: rgba(34, 226, 239, 0.62);
        }
  
        .scout-compact-progress.tone-green,
        .scout-compact-progress.tone-good,
        .scout-compact-progress.tone-elite {
          --progress-color: var(--scout-green);
          --progress-glow: rgba(92, 242, 156, 0.62);
        }
  
        .scout-compact-progress.tone-gold,
        .scout-compact-progress.tone-warn,
        .scout-compact-progress.tone-watch {
          --progress-color: var(--scout-gold);
          --progress-glow: rgba(255, 209, 102, 0.62);
        }
  
        .scout-compact-progress.tone-purple {
          --progress-color: var(--scout-purple);
          --progress-glow: rgba(184, 146, 255, 0.62);
        }
  
        .scout-compact-progress.tone-danger,
        .scout-compact-progress.tone-red {
          --progress-color: var(--scout-red);
          --progress-glow: rgba(255, 79, 99, 0.62);
        }
  
        .globe-layout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 380px;
          gap: 14px;
        }
  
        .globe-card {
          min-height: 610px;
          display: grid;
          place-items: center;
          position: relative;
          overflow: hidden;
          background:
            radial-gradient(circle at 50% 46%, rgba(57, 185, 255, 0.18), transparent 48%),
            rgba(255, 255, 255, 0.035);
        }
  
        .interactive-globe {
          width: min(70vh, 620px);
          max-width: 94%;
          aspect-ratio: 1;
          position: relative;
          display: grid;
          place-items: center;
        }
  
        .interactive-globe__halo {
          position: absolute;
          inset: 2%;
          border-radius: 999px;
          background: radial-gradient(circle, rgba(57, 185, 255, 0.18), transparent 68%);
          filter: blur(10px);
          animation: globeHalo 4s ease-in-out infinite alternate;
        }
  
        @keyframes globeHalo {
          from {
            opacity: 0.55;
            transform: scale(0.98);
          }
  
          to {
            opacity: 1;
            transform: scale(1.03);
          }
        }
  
        .interactive-globe__sphere {
          position: relative;
          width: 86%;
          height: 86%;
          border-radius: 999px;
          overflow: hidden;
          border: 1px solid rgba(190, 228, 255, 0.28);
          background:
            radial-gradient(circle at 35% 28%, rgba(255, 255, 255, 0.22), transparent 16%),
            radial-gradient(circle at 58% 60%, rgba(57, 185, 255, 0.24), transparent 46%),
            linear-gradient(135deg, rgba(10, 52, 84, 0.98), rgba(3, 18, 32, 0.98));
          box-shadow:
            inset 0 0 84px rgba(0, 0, 0, 0.58),
            0 0 84px rgba(57, 185, 255, 0.22);
        }
  
        .interactive-globe__sphere::before {
          content: "";
          position: absolute;
          inset: 8%;
          border-radius: 999px;
          background:
            radial-gradient(ellipse at 30% 34%, rgba(92, 242, 156, 0.2), transparent 19%),
            radial-gradient(ellipse at 68% 33%, rgba(92, 242, 156, 0.16), transparent 20%),
            radial-gradient(ellipse at 54% 70%, rgba(92, 242, 156, 0.12), transparent 24%);
          opacity: 0.72;
        }
  
        .globe-line {
          position: absolute;
          pointer-events: none;
          border: 1px solid rgba(210, 235, 255, 0.14);
        }
  
        .globe-line.longitude {
          top: 5%;
          bottom: 5%;
          width: 28%;
          border-radius: 999px;
        }
  
        .globe-line.longitude.a {
          left: 18%;
        }
  
        .globe-line.longitude.b {
          left: 36%;
        }
  
        .globe-line.longitude.c {
          right: 18%;
        }
  
        .globe-line.latitude {
          left: 5%;
          right: 5%;
          height: 22%;
          border-radius: 999px;
        }
  
        .globe-line.latitude.a {
          top: 22%;
        }
  
        .globe-line.latitude.b {
          top: 39%;
        }
  
        .globe-line.latitude.c {
          bottom: 22%;
        }
  
        .globe-scan {
          position: absolute;
          inset: -20% 46%;
          width: 8%;
          background: linear-gradient(90deg, transparent, rgba(57, 185, 255, 0.25), transparent);
          filter: blur(4px);
          animation: globeScan 5.5s linear infinite;
          transform-origin: center;
        }
  
        @keyframes globeScan {
          from {
            transform: rotate(0deg);
          }
  
          to {
            transform: rotate(360deg);
          }
        }
  
        .globe-pin {
          position: absolute;
          width: 20px;
          height: 20px;
          padding: 0;
          border: 0;
          background: transparent;
          transform: translate(-50%, -50%);
          z-index: 4;
        }
  
        .globe-pin__pulse {
          position: absolute;
          inset: -9px;
          border-radius: 999px;
          border: 1px solid rgba(57, 185, 255, 0.35);
          animation: scoutPulse 1.8s ease-out infinite;
        }
  
        .globe-pin__core {
          position: absolute;
          inset: 4px;
          border-radius: 999px;
          background: var(--scout-blue);
          box-shadow: 0 0 18px rgba(57, 185, 255, 0.95);
        }
  
        .globe-pin.is-gap .globe-pin__core {
          background: var(--scout-gold);
          box-shadow: 0 0 18px rgba(255, 209, 102, 0.95);
        }
  
        .globe-pin.is-risk .globe-pin__core {
          background: var(--scout-red);
          box-shadow: 0 0 18px rgba(255, 79, 99, 0.95);
        }
  
        .globe-pin.is-selected .globe-pin__core {
          inset: 1px;
          background: white;
          box-shadow: 0 0 24px rgba(255, 255, 255, 0.95);
        }
  
        .globe-pin__label {
          position: absolute;
          left: 20px;
          top: -8px;
          max-width: 160px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          padding: 4px 8px;
          border-radius: 999px;
          color: var(--scout-text);
          background: rgba(0, 0, 0, 0.48);
          border: 1px solid rgba(255, 255, 255, 0.15);
          font-size: 11px;
          font-weight: 900;
          opacity: 0;
          pointer-events: none;
          transform: translateY(4px);
          transition: 160ms ease;
        }
  
        .globe-pin:hover .globe-pin__label,
        .globe-pin.is-selected .globe-pin__label {
          opacity: 1;
          transform: translateY(0);
        }
  
        .country-intel-card {
          min-height: 610px;
          overflow: hidden;
        }
  
        .country-intel-lite {
          height: 100%;
          display: grid;
          align-content: start;
          gap: 14px;
        }
  
        .country-intel-lite header {
          display: flex;
          justify-content: space-between;
          gap: 14px;
        }
  
        .country-intel-lite header span {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
  
        .country-intel-lite h3 {
          margin: 5px 0 3px;
          font-size: 30px;
          letter-spacing: -0.04em;
        }
  
        .country-intel-lite p {
          margin: 0;
          color: var(--scout-muted);
        }
  
        .risk-score {
          width: 74px;
          height: 74px;
          border-radius: 20px;
          display: grid;
          place-items: center;
          align-content: center;
          border: 1px solid var(--scout-line);
          background: rgba(0, 0, 0, 0.14);
        }
  
        .risk-score strong {
          font-size: 27px;
          line-height: 1;
        }
  
        .risk-score span {
          margin-top: 3px;
          color: var(--scout-muted);
          font-size: 10px;
          text-transform: uppercase;
        }
  
        .risk-score.danger {
          color: var(--scout-red);
          border-color: rgba(255, 79, 99, 0.34);
          background: rgba(255, 79, 99, 0.1);
        }
  
        .risk-score.warn,
        .risk-score.watch {
          color: var(--scout-gold);
          border-color: rgba(255, 209, 102, 0.34);
          background: rgba(255, 209, 102, 0.1);
        }
  
        .risk-score.good {
          color: var(--scout-green);
          border-color: rgba(92, 242, 156, 0.34);
          background: rgba(92, 242, 156, 0.1);
        }
  
        .country-metrics-lite {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 9px;
        }
  
        .mini-metric {
          min-height: 76px;
          padding: 12px;
          border-radius: 15px;
          border: 1px solid rgba(150, 205, 235, 0.12);
          background: rgba(0, 0, 0, 0.13);
          display: grid;
          align-content: center;
          gap: 4px;
        }
  
        .mini-metric span {
          color: var(--scout-muted);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
        }
  
        .mini-metric strong {
          font-size: 21px;
        }
  
        .country-bars-lite {
          display: grid;
          gap: 10px;
        }
  
        .labeled-bar {
          display: grid;
          gap: 6px;
        }
  
        .labeled-bar > div:first-child {
          display: flex;
          justify-content: space-between;
          gap: 10px;
        }
  
        .labeled-bar span {
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 900;
        }
  
        .labeled-bar b {
          color: var(--scout-text);
          font-size: 12px;
        }
  
        .country-players-lite {
          min-height: 0;
          display: grid;
          gap: 8px;
        }
  
        .country-players-lite__head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
        }
  
        .country-players-lite__head h4 {
          margin: 0;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
  
        .country-players-lite__head span {
          color: var(--scout-muted);
          font-weight: 1000;
        }
  
        .country-player-row {
          display: grid;
          grid-template-columns: 40px minmax(0, 1fr) 48px;
          align-items: center;
          gap: 10px;
          min-height: 54px;
          padding: 8px 10px;
          border-radius: 14px;
          border: 1px solid rgba(150, 205, 235, 0.12);
          background: rgba(0, 0, 0, 0.13);
          color: var(--scout-text);
          text-align: left;
          transition: 150ms ease;
        }
  
        .country-player-row:hover {
          transform: translateX(2px);
          border-color: rgba(57, 185, 255, 0.32);
          background: rgba(57, 185, 255, 0.08);
        }
  
        .country-player-row b {
          color: var(--scout-red);
        }
  
        .country-player-row strong,
        .country-player-row small {
          display: block;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
  
        .country-player-row small {
          margin-top: 2px;
          color: var(--scout-muted);
        }
  
        .country-player-row i {
          justify-self: end;
          color: var(--scout-blue);
          font-style: normal;
          font-weight: 1000;
        }
  
        .region-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin-top: 14px;
        }
  
        .region-chip {
          padding: 12px;
          border-radius: 16px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.04);
          display: grid;
          gap: 9px;
        }
  
        .region-chip > div {
          display: flex;
          justify-content: space-between;
          gap: 10px;
        }
  
        .region-chip span {
          color: var(--scout-muted);
          font-weight: 900;
        }
  
        .region-chip strong {
          font-size: 21px;
        }
  
        .board-filters {
          display: grid;
          grid-template-columns: minmax(220px, 1.4fr) repeat(6, minmax(120px, 1fr)) repeat(3, auto);
          gap: 9px;
          align-items: end;
          margin-bottom: 14px;
        }
  
        .scout-field {
          min-width: 0;
          display: grid;
          gap: 6px;
        }
  
        .scout-field span {
          color: var(--scout-muted);
          font-size: 10px;
          font-weight: 1000;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
  
        .scout-field input,
        .scout-field select {
          width: 100%;
          min-height: 42px;
          border-radius: 12px;
          border: 1px solid var(--scout-line);
          background: rgba(0, 0, 0, 0.2);
          color: var(--scout-text);
          outline: none;
          padding: 0 12px;
        }
  
        .scout-field input::placeholder {
          color: rgba(220, 237, 248, 0.36);
        }
  
        .scout-field input:focus,
        .scout-field select:focus {
          border-color: rgba(57, 185, 255, 0.52);
          box-shadow: 0 0 0 3px rgba(57, 185, 255, 0.11);
        }
  
        .scout-field select option {
          background: #071524;
          color: var(--scout-text);
        }
  
        .filter-toggle {
          min-height: 42px;
          padding: 0 12px;
          border-radius: 12px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.045);
          color: var(--scout-muted);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          white-space: nowrap;
          transition: 150ms ease;
        }
  
        .filter-toggle:hover,
        .filter-toggle.is-on {
          color: var(--scout-text);
          border-color: rgba(255, 79, 99, 0.36);
          background: rgba(255, 79, 99, 0.1);
        }
  
        .board-table-wrap {
          overflow: auto;
          border-radius: 18px;
          border: 1px solid var(--scout-line);
          background: rgba(0, 0, 0, 0.15);
        }
  
        .board-table {
          width: 100%;
          min-width: 1040px;
          border-collapse: collapse;
        }
  
        .board-table th,
        .board-table td {
          padding: 13px 14px;
          text-align: left;
          border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
  
        .board-table th {
          position: sticky;
          top: 0;
          z-index: 2;
          color: var(--scout-muted);
          background: rgba(6, 18, 30, 0.98);
          font-size: 10px;
          font-weight: 1000;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
  
        .board-table tbody tr {
          cursor: pointer;
          transition: 120ms ease;
        }
  
        .board-table tbody tr:hover,
        .board-table tbody tr.is-selected {
          background: rgba(57, 185, 255, 0.08);
        }
  
        .rank-badge {
          display: inline-flex;
          min-width: 36px;
          height: 30px;
          align-items: center;
          justify-content: center;
          border-radius: 9px;
          color: var(--scout-red);
          background: rgba(255, 79, 99, 0.1);
          border: 1px solid rgba(255, 79, 99, 0.26);
          font-weight: 1000;
        }
  
        .board-player-cell {
          min-width: 260px;
          display: flex;
          align-items: center;
          gap: 11px;
        }
  
        .board-player-cell strong,
        .board-player-cell small,
        .stack-cell strong,
        .stack-cell small {
          display: block;
        }
  
        .board-player-cell small,
        .stack-cell small {
          margin-top: 3px;
          color: var(--scout-muted);
        }
  
        .soft-pill,
        .risk-pill {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          padding: 6px 10px;
          border: 1px solid rgba(255, 255, 255, 0.09);
          background: rgba(255, 255, 255, 0.065);
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 850;
          white-space: nowrap;
        }
  
        .risk-pill.danger {
          color: var(--scout-red);
          background: rgba(255, 79, 99, 0.1);
          border-color: rgba(255, 79, 99, 0.26);
        }
  
        .risk-pill.warn,
        .risk-pill.watch {
          color: var(--scout-gold);
          background: rgba(255, 209, 102, 0.1);
          border-color: rgba(255, 209, 102, 0.26);
        }
  
        .risk-pill.good {
          color: var(--scout-green);
          background: rgba(92, 242, 156, 0.1);
          border-color: rgba(92, 242, 156, 0.26);
        }
  
        .watchlist-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
        }
  
        .watch-card {
          min-height: 245px;
          border-radius: 18px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.04);
          color: var(--scout-text);
          padding: 14px;
          display: grid;
          gap: 12px;
          text-align: left;
          transition: 160ms ease;
        }
  
        .watch-card:hover {
          transform: translateY(-2px);
          border-color: rgba(57, 185, 255, 0.32);
          background: rgba(57, 185, 255, 0.08);
        }
  
        .watch-card__top,
        .watch-card__footer {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          align-items: center;
        }
  
        .watch-card__top span {
          color: var(--scout-red);
          font-weight: 1000;
        }
  
        .watch-card__top b {
          color: var(--scout-gold);
          font-size: 11px;
          text-transform: uppercase;
        }
  
        .watch-card__main {
          display: flex;
          align-items: center;
          gap: 12px;
        }
  
        .watch-card__main strong,
        .watch-card__main small {
          display: block;
        }
  
        .watch-card__main small {
          margin-top: 3px;
          color: var(--scout-muted);
        }
  
        .watch-card__bars {
          display: grid;
          gap: 10px;
        }
  
        .report-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }
  
        .report-panel {
          padding: 16px;
          min-height: 260px;
        }
  
        .report-panel.is-wide {
          grid-column: 1 / -1;
          min-height: 0;
        }
  
        .position-report-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
        }
  
        .position-report-card {
          min-height: 86px;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 4px;
          border-radius: 16px;
          border: 1px solid rgba(57, 185, 255, 0.2);
          background:
            radial-gradient(circle at 50% 20%, rgba(57, 185, 255, 0.16), transparent 42%),
            rgba(0, 0, 0, 0.15);
        }
  
        .position-report-card span {
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 1000;
        }
  
        .position-report-card strong {
          font-size: 30px;
          line-height: 1;
        }
  
        .report-list {
          display: grid;
          gap: 8px;
        }
  
        .report-row {
          min-height: 42px;
          display: grid;
          grid-template-columns: 34px minmax(0, 1fr) auto;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 13px;
          border: 1px solid rgba(255, 255, 255, 0.07);
          background: rgba(0, 0, 0, 0.13);
        }
  
        .report-row b {
          width: 24px;
          height: 24px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          color: var(--scout-blue);
          background: rgba(57, 185, 255, 0.1);
        }
  
        .report-row span {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: var(--scout-muted);
        }
  
        .report-row strong {
          color: var(--scout-text);
        }
  
        .director-memo {
          display: grid;
          gap: 12px;
        }
  
        .memo-stats,
        .memo-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 9px;
        }
  
        .memo-stats span,
        .memo-meta span {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          min-height: 36px;
          padding: 0 11px;
          border-radius: 999px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.045);
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 850;
        }
  
        .memo-stats b {
          color: var(--scout-text);
        }
  
        .scout-staff-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
        }
  
        .staff-card {
          min-height: 265px;
          padding: 15px;
          border-radius: 18px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.04);
          color: var(--scout-text);
          display: grid;
          gap: 12px;
          text-align: left;
          transition: 160ms ease;
        }
  
        .staff-card:hover,
        .staff-card.is-selected {
          transform: translateY(-2px);
          border-color: rgba(57, 185, 255, 0.34);
          background: rgba(57, 185, 255, 0.08);
        }
  
        .staff-card__top {
          display: flex;
          align-items: center;
          gap: 12px;
        }
  
        .staff-card__top strong,
        .staff-card__top span {
          display: block;
        }
  
        .staff-card__top span {
          margin-top: 3px;
          color: var(--scout-muted);
        }
  
        .staff-card__meta,
        .staff-card__footer,
        .selected-scout__chips {
          display: flex;
          flex-wrap: wrap;
          gap: 7px;
        }
  
        .staff-card__meta span,
        .staff-card__footer span,
        .selected-scout__chips span {
          display: inline-flex;
          align-items: center;
          min-height: 28px;
          padding: 0 9px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.09);
          background: rgba(255, 255, 255, 0.055);
          color: var(--scout-muted);
          font-size: 11px;
          font-weight: 850;
        }
  
        .staff-card__bars {
          display: grid;
          gap: 9px;
        }
  
        .staff-card__footer {
          margin-top: auto;
          justify-content: space-between;
        }
  
        .player-view-empty {
          min-height: 520px;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 16px;
        }
  
        .quick-player-list {
          width: min(620px, 100%);
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 8px;
        }
  
        .quick-player-list button {
          min-height: 44px;
          display: grid;
          grid-template-columns: 46px minmax(0, 1fr) 44px;
          align-items: center;
          gap: 8px;
          border-radius: 13px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.045);
          color: var(--scout-text);
          text-align: left;
          padding: 8px 10px;
        }
  
        .quick-player-list b {
          color: var(--scout-red);
        }
  
        .quick-player-list span {
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
  
        .quick-player-list small {
          color: var(--scout-muted);
          text-align: right;
        }
  
        .player-hero-card {
          display: flex;
          justify-content: space-between;
          align-items: stretch;
          gap: 16px;
          margin-bottom: 14px;
          padding: 16px;
          border-radius: var(--scout-radius-lg);
          border: 1px solid var(--scout-line);
          background:
            radial-gradient(circle at 12% 18%, rgba(57, 185, 255, 0.15), transparent 35%),
            rgba(255, 255, 255, 0.045);
        }
  
        .player-identity {
          display: flex;
          align-items: center;
          gap: 16px;
          min-width: 0;
        }
  
        .player-identity > div:last-child {
          min-width: 0;
        }
  
        .player-identity span:first-child {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
  
        .player-identity h2 {
          margin: 5px 0 5px;
          font-size: clamp(28px, 3vw, 42px);
          line-height: 0.98;
          letter-spacing: -0.045em;
        }
  
        .player-identity p {
          margin: 0;
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .player-tags {
          display: flex;
          flex-wrap: wrap;
          gap: 7px;
          margin-top: 11px;
        }
  
        .player-tags span {
          display: inline-flex;
          align-items: center;
          min-height: 30px;
          padding: 0 10px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.09);
          background: rgba(255, 255, 255, 0.055);
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 850;
        }
  
        .player-rank-card {
          min-width: 142px;
          display: grid;
          place-items: center;
          align-content: center;
          border-radius: 22px;
          border: 1px solid rgba(57, 185, 255, 0.28);
          background: rgba(57, 185, 255, 0.09);
          padding: 16px;
        }
  
        .player-rank-card span,
        .player-rank-card small {
          color: var(--scout-muted);
          font-weight: 900;
        }
  
        .player-rank-card strong {
          font-size: 42px;
          line-height: 1;
        }
  
        .player-file-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }
  
        .ring-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
        }
  
        .scout-ring {
          min-height: 145px;
          position: relative;
          display: grid;
          place-items: center;
          border-radius: 18px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(0, 0, 0, 0.13);
        }
  
        .scout-ring svg {
          width: 94px;
          height: 94px;
          transform: rotate(-90deg);
        }
  
        .scout-ring__track,
        .scout-ring__fill {
          fill: none;
          stroke-width: 7;
        }
  
        .scout-ring__track {
          stroke: rgba(255, 255, 255, 0.08);
        }
  
        .scout-ring__fill {
          stroke: var(--ring-color);
          stroke-linecap: round;
          transition: stroke-dashoffset 260ms ease;
        }
  
        .scout-ring > div {
          position: absolute;
          display: grid;
          place-items: center;
          text-align: center;
        }
  
        .scout-ring strong {
          font-size: 23px;
          line-height: 1;
        }
  
        .scout-ring span {
          margin-top: 4px;
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 900;
        }
  
        .scout-ring.tone-blue,
        .scout-ring.tone-watch {
          --ring-color: var(--scout-blue);
        }
  
        .scout-ring.tone-green,
        .scout-ring.tone-good,
        .scout-ring.tone-elite {
          --ring-color: var(--scout-green);
        }
  
        .scout-ring.tone-gold,
        .scout-ring.tone-warn {
          --ring-color: var(--scout-gold);
        }
  
        .scout-ring.tone-danger,
        .scout-ring.tone-red {
          --ring-color: var(--scout-red);
        }
  
        .player-bio {
          margin: 0;
          display: grid;
          gap: 7px;
        }
  
        .player-bio div {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          padding: 9px 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
  
        .player-bio div:last-child {
          border-bottom: 0;
        }
  
        .player-bio dt {
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .player-bio dd {
          margin: 0;
          color: var(--scout-text);
          font-weight: 1000;
          text-align: right;
        }
  
        .country-mini {
          display: grid;
          gap: 9px;
        }
  
        .country-mini strong {
          font-size: 24px;
        }
  
        .country-mini span,
        .country-mini small {
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .skill-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px 16px;
        }
  
        .skill-row {
          display: grid;
          grid-template-columns: 150px minmax(0, 1fr);
          align-items: center;
          gap: 10px;
        }
  
        .skill-row > span {
          min-width: 0;
          color: var(--scout-muted);
          font-weight: 850;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
  
        .tag-cloud,
        .flag-list {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
  
        .tag-cloud span,
        .flag-list span {
          display: inline-flex;
          align-items: center;
          min-height: 32px;
          padding: 0 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 850;
        }
  
        .tag-cloud span {
          color: var(--scout-blue);
          border: 1px solid rgba(57, 185, 255, 0.24);
          background: rgba(57, 185, 255, 0.09);
        }
  
        .flag-list span {
          color: var(--scout-red);
          border: 1px solid rgba(255, 79, 99, 0.24);
          background: rgba(255, 79, 99, 0.09);
        }
  
        .clean-file {
          min-height: 90px;
          display: grid;
          place-items: center;
          border-radius: 16px;
          border: 1px solid rgba(92, 242, 156, 0.24);
          background: rgba(92, 242, 156, 0.08);
          color: var(--scout-green);
          font-weight: 1000;
        }
  
        .draft-action-grid {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 10px;
        }
  
        .draft-action-button {
          min-height: 92px;
          display: grid;
          align-content: center;
          gap: 6px;
          text-align: center;
          border-radius: 16px;
          border: 1px solid var(--scout-line);
          background: rgba(0, 0, 0, 0.13);
          color: var(--scout-text);
          padding: 12px;
          transition: 160ms ease;
        }
  
        .draft-action-button:hover {
          transform: translateY(-2px);
          border-color: rgba(57, 185, 255, 0.34);
          background: rgba(57, 185, 255, 0.08);
        }
  
        .draft-action-button strong {
          font-size: 14px;
          text-transform: uppercase;
        }
  
        .draft-action-button span {
          color: var(--scout-muted);
          font-size: 12px;
          font-weight: 850;
        }
  
        .note-list {
          display: grid;
          gap: 10px;
        }
  
        .scout-note {
          display: grid;
          grid-template-columns: 34px minmax(0, 1fr);
          gap: 10px;
        }
  
        .scout-note > b {
          width: 30px;
          height: 30px;
          display: grid;
          place-items: center;
          border-radius: 999px;
          color: var(--scout-blue);
          background: rgba(57, 185, 255, 0.1);
          border: 1px solid rgba(57, 185, 255, 0.22);
        }
  
        .scout-note > div {
          padding-bottom: 11px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
  
        .scout-note header {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 5px;
        }
  
        .scout-note header span {
          color: var(--scout-muted);
          font-size: 12px;
        }
  
        .scout-note p {
          margin: 0;
          color: var(--scout-muted);
          line-height: 1.5;
          font-size: 13px;
        }
  
        .command-panel {
          padding: 16px;
          display: grid;
          gap: 14px;
        }
  
        .command-panel__head {
          display: grid;
          gap: 4px;
        }
  
        .command-panel__head span {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
  
        .command-panel__head h3 {
          margin: 0;
          font-size: 19px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
  
        .command-target {
          display: grid;
          gap: 4px;
          padding: 14px;
          border-radius: 17px;
          border: 1px solid rgba(57, 185, 255, 0.24);
          background:
            radial-gradient(circle at 12% 22%, rgba(57, 185, 255, 0.15), transparent 44%),
            rgba(57, 185, 255, 0.075);
        }
  
        .command-target span,
        .cost-card span {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }
  
        .command-target strong {
          font-size: 21px;
          line-height: 1.1;
        }
  
        .command-target small {
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .selected-scout {
          display: grid;
          gap: 10px;
          padding: 13px;
          border-radius: 17px;
          border: 1px solid var(--scout-line);
          background: rgba(0, 0, 0, 0.13);
        }
  
        .selected-scout__top {
          display: flex;
          align-items: center;
          gap: 10px;
        }
  
        .selected-scout__top strong,
        .selected-scout__top span {
          display: block;
        }
  
        .selected-scout__top span {
          margin-top: 2px;
          color: var(--scout-muted);
        }
  
        .intensity-picker {
          display: grid;
          gap: 7px;
        }
  
        .intensity-picker > span {
          color: var(--scout-muted);
          font-size: 10px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }
  
        .intensity-picker > div {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 7px;
        }
  
        .intensity-picker button {
          min-height: 39px;
          border-radius: 12px;
          border: 1px solid var(--scout-line);
          background: rgba(255, 255, 255, 0.045);
          color: var(--scout-muted);
          font-size: 11px;
          font-weight: 1000;
          transition: 150ms ease;
        }
  
        .intensity-picker button:hover,
        .intensity-picker button.is-active {
          color: var(--scout-text);
          border-color: rgba(255, 79, 99, 0.38);
          background: rgba(255, 79, 99, 0.1);
        }
  
        .cost-card {
          display: grid;
          gap: 6px;
          padding: 15px;
          border-radius: 18px;
          border: 1px solid rgba(255, 209, 102, 0.26);
          background:
            radial-gradient(circle at 16% 22%, rgba(255, 209, 102, 0.14), transparent 42%),
            rgba(255, 209, 102, 0.06);
        }
  
        .cost-card span {
          color: var(--scout-gold);
        }
  
        .cost-card strong {
          font-size: 34px;
          line-height: 1;
        }
  
        .cost-card small {
          color: var(--scout-muted);
          font-weight: 850;
        }
  
        .assign-button {
          min-height: 54px;
          border: 0;
          border-radius: 17px;
          color: #02101c;
          background: linear-gradient(135deg, #6bd2ff, #5cf29c);
          box-shadow:
            0 16px 38px rgba(57, 185, 255, 0.23),
            inset 0 -2px 0 rgba(0, 0, 0, 0.16);
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          transition: 160ms ease;
        }
  
        .assign-button:hover {
          transform: translateY(-2px);
          box-shadow:
            0 20px 44px rgba(57, 185, 255, 0.32),
            inset 0 -2px 0 rgba(0, 0, 0, 0.16);
        }
  
        .assignment-card {
          border: 1px solid var(--scout-line);
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.04);
          padding: 14px;
          display: grid;
          gap: 12px;
        }
  
        .assignment-card header,
        .assignment-card footer {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
        }
  
        .assignment-card header span {
          color: var(--scout-blue);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }
  
        .assignment-card h3 {
          margin: 4px 0 2px;
          font-size: 18px;
        }
  
        .assignment-card p {
          margin: 0;
          color: var(--scout-muted);
        }
  
        .assignment-card header b {
          display: inline-flex;
          padding: 6px 9px;
          border-radius: 999px;
          color: var(--scout-green);
          background: rgba(92, 242, 156, 0.1);
          border: 1px solid rgba(92, 242, 156, 0.22);
          font-size: 12px;
        }
  
        .assignment-card footer span {
          color: var(--scout-gold);
          font-weight: 1000;
        }
  
        .assignment-card footer button {
          min-height: 32px;
          padding: 0 11px;
          border-radius: 10px;
          border: 1px solid rgba(255, 79, 99, 0.3);
          background: rgba(255, 79, 99, 0.09);
          color: var(--scout-red);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
        }
  
        .scout-empty {
          min-height: 170px;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 9px;
          text-align: center;
          padding: 22px;
          border-radius: 18px;
          border: 1px dashed rgba(150, 205, 235, 0.22);
          background: rgba(255, 255, 255, 0.025);
        }
  
        .scout-empty > div {
          width: 54px;
          height: 54px;
          display: grid;
          place-items: center;
          border-radius: 999px;
          border: 1px solid rgba(57, 185, 255, 0.24);
          color: var(--scout-blue);
          background: rgba(57, 185, 255, 0.08);
          font-weight: 1000;
        }
  
        .scout-empty h3 {
          margin: 0;
          font-size: 16px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
  
        .scout-empty p {
          max-width: 420px;
          margin: 0;
          color: var(--scout-muted);
          font-size: 13px;
          line-height: 1.45;
        }
  
        .scout-loading {
          min-height: 420px;
          display: grid;
          place-items: center;
          align-content: center;
          gap: 14px;
          text-align: center;
        }
  
        .scout-loading-rink {
          position: relative;
          width: 250px;
          height: 124px;
          border-radius: 999px;
          border: 1px solid rgba(238, 248, 255, 0.18);
          background: rgba(255, 255, 255, 0.035);
          overflow: hidden;
        }
  
        .scout-loading-rink span {
          position: absolute;
          top: 0;
          left: 50%;
          width: 2px;
          height: 100%;
          background: rgba(57, 185, 255, 0.32);
        }
  
        .scout-loading-rink i {
          position: absolute;
          top: 53px;
          left: 26px;
          width: 18px;
          height: 18px;
          border-radius: 999px;
          background: #05070b;
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
          animation: scoutPuck 1.35s ease-in-out infinite alternate;
        }
  
        @keyframes scoutPuck {
          from {
            transform: translateX(0);
          }
  
          to {
            transform: translateX(180px);
          }
        }
  
        .scout-loading strong {
          font-size: 20px;
        }
  
        .scout-loading p {
          margin: 0;
          color: var(--scout-muted);
        }
  
        @media (max-width: 1560px) {
          .scout-layout.has-panel {
            grid-template-columns: minmax(0, 1fr) 330px;
          }
  
          .scout-overview-grid {
            grid-template-columns: 1fr 0.86fr;
          }
  
          .top-prospect-row {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
  
          .board-filters {
            grid-template-columns: repeat(5, minmax(0, 1fr));
          }
  
          .watchlist-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
  
          .draft-action-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }
  
        @media (max-width: 1280px) {
          .scout-root {
            grid-template-columns: 76px minmax(0, 1fr);
          }
  
          .scout-screen-head {
            align-items: flex-start;
            flex-direction: column;
          }
  
          .scout-head-actions {
            justify-content: flex-start;
          }
  
          .scout-top-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
  
          .scout-layout.has-panel {
            grid-template-columns: 1fr;
          }
  
          .scout-workspace,
          .scout-command-rail {
            max-height: none;
          }
  
          .globe-layout {
            grid-template-columns: 1fr;
          }
  
          .country-intel-card {
            min-height: 0;
          }
  
          .region-strip {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
  
          .scout-staff-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }
  
        @media (max-width: 980px) {
          .scout-root {
            grid-template-columns: 1fr;
            overflow: auto;
          }
  
          .scout-sidebar {
            min-height: auto;
            height: auto;
            flex-direction: row;
            align-items: center;
            border-right: 0;
            border-bottom: 1px solid var(--scout-line);
          }
  
          .scout-home-button,
          .scout-settings {
            width: 72px;
            height: 66px;
            border: 0;
          }
  
          .scout-side-nav {
            flex: 1;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            padding: 0;
          }
  
          .scout-side-button {
            min-height: 66px;
          }
  
          .scout-side-button.is-active::before {
            left: 14px;
            right: 14px;
            top: auto;
            bottom: 0;
            width: auto;
            height: 3px;
          }
  
          .scout-main {
            height: auto;
            min-height: calc(100vh - 67px);
            overflow: visible;
            padding: 16px;
          }
  
          .scout-tabs {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
  
          .scout-overview-grid,
          .report-grid,
          .player-file-grid {
            grid-template-columns: 1fr;
          }
  
          .top-prospect-row,
          .overview-assignment-strip,
          .watchlist-grid,
          .scout-staff-grid,
          .ring-grid,
          .position-report-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
  
          .globe-card {
            min-height: 500px;
          }
  
          .region-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
  
          .player-hero-card {
            flex-direction: column;
          }
  
          .player-rank-card {
            min-width: 0;
            min-height: 120px;
          }
  
          .skill-grid {
            grid-template-columns: 1fr;
          }
        }
  
        @media (max-width: 680px) {
          .scout-main {
            padding: 12px;
          }
  
          .scout-title-block {
            align-items: flex-start;
          }
  
          .scout-team-logo--md {
            width: 58px;
            height: 58px;
            border-radius: 16px;
          }
  
          .scout-title-block h1 {
            font-size: 27px;
          }
  
          .scout-top-strip,
          .scout-tabs,
          .top-prospect-row,
          .overview-assignment-strip,
          .watchlist-grid,
          .scout-staff-grid,
          .ring-grid,
          .position-report-grid,
          .region-strip,
          .country-metrics-lite,
          .draft-action-grid,
          .quick-player-list,
          .position-need-grid,
          .board-filters {
            grid-template-columns: 1fr;
          }
  
          .scout-side-button small {
            display: none;
          }
  
          .scout-side-button span {
            font-size: 20px;
          }
  
          .scout-home-button,
          .scout-settings {
            width: 58px;
          }
  
          .scout-workspace {
            padding: 12px;
            border-radius: 16px;
          }
  
          .interactive-globe {
            width: 100%;
          }
  
          .globe-card {
            min-height: 420px;
          }
  
          .globe-pin__label {
            display: none;
          }
  
          .country-player-row {
            grid-template-columns: 36px minmax(0, 1fr);
          }
  
          .country-player-row i {
            grid-column: 2;
            justify-self: start;
          }
  
          .player-identity {
            flex-direction: column;
            align-items: flex-start;
          }
  
          .player-identity h2 {
            font-size: 30px;
          }
  
          .scout-avatar--xl {
            width: 90px;
            height: 90px;
            border-radius: 24px;
          }
  
          .skill-row {
            grid-template-columns: 1fr;
          }
  
          .report-row {
            grid-template-columns: 34px minmax(0, 1fr);
          }
  
          .report-row strong {
            grid-column: 2;
          }
  
          .scout-compact-progress {
            grid-template-columns: 38px minmax(70px, 1fr);
          }
        }
      `}</style>
    );
  }