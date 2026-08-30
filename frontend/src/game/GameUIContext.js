import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  advanceFranchise,
  advanceFreeAgencyDay,
  advanceSeasonPhase,
  continueOffseason,
  dismissFranchisePopups,
  generateNextSeason,
  getFranchiseCrisis,
  getFranchiseNarrative,
  getFranchiseState,
  getFranchiseStateHeavy,
  listTeams,
  reopenOffseasonStage,
  resetFranchiseStateCache,
  startFranchise,
  submitDecision,
  submitStorylineChoice,
  enterPlayoffs,
} from "../services/franchiseService";
import {
  clearFranchiseSession,
  formatFranchiseApiError,
  getFranchiseSessionId,
  isExpiredFranchiseSessionError,
  readFranchiseHubSnapshot,
  resetFranchiseServerSessions,
  resolveApiBaseUrl,
  setFranchiseSessionId,
  syncFranchiseSessionWithBackend,
  writeFranchiseHubSnapshot,
} from "../services/api";
import { markNavigation, record as perfRecord } from "../services/perfProfiler";
import { HUB_MENU, SCREENS, buildDefaultFranchiseTeamList } from "./constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import {
  activeBreakingAlerts,
  breakingAlertKey,
  readDismissedBreakingKeys,
  writeDismissedBreakingKeys,
} from "../utils/breakingAlerts";
import hubWallTextureSrc from "../pictures/gray-abstract-texture-background.jpg";
import officeFontBold from "../styles/ArchivoBlack-Regular.ttf";
import { ShowcasePopupLayer } from "../components/game/ShowcasePopupLayer";
import { TradeDemandCrisisOverlay } from "../components/game/TradeDemandCrisisOverlay";
import { FranchiseEventLayer } from "../components/game/FranchiseEventLayer";
import WorldJuniorsEvent from "../events/worldJuniors/WorldJuniorsEvent";
import { resolveWorldJuniorsPayload } from "../events/worldJuniors/WorldJuniorsMenu";

const GameUIContext = createContext(null);
export const FranchiseStateContext = createContext(null);

export function useFranchiseState() {
  return useContext(FranchiseStateContext);
}

export const INJURIES_STORAGE_KEY = "nhl_franchise_injuries_enabled";

function franchiseTeamKey(team) {
  if (!team || typeof team !== "object") {
    return "";
  }
  return String(
    team.abbr ||
      team.abbreviation ||
      team.team_id ||
      team.teamId ||
      team.id ||
      team.name ||
      ""
  )
    .trim()
    .toUpperCase();
}

function remapTeamIndex(prevTeams, prevIndex, nextTeams) {
  if (!Array.isArray(nextTeams) || !nextTeams.length) {
    return -1;
  }
  if (prevIndex == null || prevIndex < 0 || !Array.isArray(prevTeams)) {
    return -1;
  }
  const code = franchiseTeamKey(prevTeams[prevIndex]);
  if (!code) {
    return Math.min(prevIndex, nextTeams.length - 1);
  }
  const found = nextTeams.findIndex((team) => franchiseTeamKey(team) === code);
  return found >= 0 ? found : -1;
}

function sameTeamRoster(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }
  return left.every((team, index) => franchiseTeamKey(team) === franchiseTeamKey(right[index]));
}

function readInjuriesPref() {
  try {
    const stored = localStorage.getItem(INJURIES_STORAGE_KEY);
    if (stored != null) return stored === "true";
  } catch {
    /* ignore */
  }
  return true;
}

export function useGameUI() {
  const ctx = useContext(GameUIContext);
  if (!ctx) throw new Error("useGameUI outside GameUIProvider");
  return ctx;
}

/* ---------------------------------------------------------------------------
   HUB WARM-UP
   ---------------------------------------------------------------------------
   The opening hallway, office cinematic and franchise setup are all real
   playable time. That time is used to pull the hub's expensive resources into
   the browser cache so the final transition is not a cold start.

   Each stage is tracked independently so the setup loading state can describe
   what is still settling in plain language, and so the hub transition can be
   held until the core scene is genuinely prepared.
*/

export const HUB_WARMUP_STAGES = Object.freeze({
  ENVIRONMENT: "environment",
  CRESTS: "crests",
  OPERATIONS: "operations",
});

export const HUB_WARMUP_LABELS = Object.freeze({
  environment: "Executive office",
  crests: "League identity",
  operations: "Hockey operations records",
});

function prefetchImage(src, timeoutMs = 25000) {
  return new Promise((resolve) => {
    if (!src || typeof window === "undefined") {
      resolve(false);
      return;
    }

    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      resolve(ok);
    };

    const timer = window.setTimeout(() => finish(false), timeoutMs);
    const img = new Image();
    img.decoding = "async";
    img.onload = () => finish(true);
    img.onerror = () => finish(false);
    img.src = src;
  });
}

function prefetchResource(src, timeoutMs = 40000) {
  return new Promise((resolve) => {
    if (!src || typeof window === "undefined" || typeof fetch !== "function") {
      resolve(false);
      return;
    }

    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      resolve(ok);
    };

    const timer = window.setTimeout(() => finish(false), timeoutMs);
    fetch(src, { mode: "cors", credentials: "same-origin" })
      .then((res) => finish(Boolean(res && res.ok)))
      .catch(() => finish(false));
  });
}

function prefetchImagesBatched(sources, batchSize = 6) {
  const list = Array.from(new Set((sources || []).filter(Boolean)));
  if (!list.length) return Promise.resolve(true);

  return new Promise((resolve) => {
    let index = 0;
    const runBatch = () => {
      const slice = list.slice(index, index + batchSize);
      index += batchSize;
      if (!slice.length) {
        resolve(true);
        return;
      }
      Promise.allSettled(slice.map((src) => prefetchImage(src))).finally(() => {
        if (typeof window.requestIdleCallback === "function") {
          window.requestIdleCallback(runBatch, { timeout: 1200 });
        } else {
          window.setTimeout(runBatch, 48);
        }
      });
    };
    runBatch();
  });
}

function playGlobalBreakingSting(level) {
  if (typeof window === "undefined") return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = level === "league_defining" ? "sawtooth" : "square";
    osc.frequency.value = level === "league_defining" ? 880 : 660;
    gain.gain.value = 0.035;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.3);
    osc.stop(ctx.currentTime + 0.35);
  } catch {
    /* optional */
  }
}

function AdvancingOverlay() {
  return (
    <div
      className="franchise-advancing-overlay"
      role="status"
      aria-live="polite"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 11500,
        display: "grid",
        placeItems: "center",
        background: "rgba(4, 8, 14, 0.42)",
        backdropFilter: "blur(2px)",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          padding: "14px 18px",
          borderRadius: 8,
          border: "1px solid rgba(19, 216, 231, 0.35)",
          background: "rgba(9, 25, 38, 0.96)",
          color: "#dffcff",
          fontSize: 12,
          fontWeight: 800,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
        }}
      >
        Simulating league day…
      </div>
    </div>
  );
}

function HubBootstrapScreen({ label = "Restoring franchise…" }) {
  return (
    <div
      style={{
        minHeight: "100%",
        display: "grid",
        placeItems: "center",
        background: "#0c0e14",
        color: "rgba(201,168,106,0.85)",
        fontFamily: 'var(--font-office-display, "Archivo Black", sans-serif)',
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        fontSize: 12,
        fontWeight: 800,
      }}
    >
      {label}
    </div>
  );
}

function leanStateIdentity(prev, next) {
  if (!prev || !next) return false;
  return (
    prev.stats_revision === next.stats_revision &&
    prev.narrative_revision === next.narrative_revision &&
    prev.prospect_revision === next.prospect_revision &&
    prev.calendar_cursor === next.calendar_cursor &&
    prev.franchise_today_iso === next.franchise_today_iso &&
    prev.scouting_as_of_iso === next.scouting_as_of_iso &&
    prev.phase === next.phase &&
    prev.season_phase === next.season_phase &&
    prev.offseason_stage === next.offseason_stage &&
    prev.next_important_event === next.next_important_event &&
    prev.session_id === next.session_id
  );
}

const LEAN_MERGE_PROGRESS_KEYS = [
  "playoff_live",
  "playoff_payload",
  "champion_id",
  "phase",
  "season_phase",
  "offseason_stage",
  "next_important_event",
  "awards_payload",
  "awards",
  "retirements_payload",
  "retirements",
  "salary_cap_payload",
  "salary_cap",
  "development_report_payload",
  "development_report",
  "draft_lottery_payload",
  "draft_lottery",
  "draft_combine_payload",
  "draft_combine",
  "draft_state",
  "draft_payload",
  "draft",
  "draft_review_payload",
  "draft_review",
  "prospect_rights_payload",
  "prospect_rights",
  "resign_payload",
  "contracts",
  "free_agency_market_payload",
  "free_agency_market",
  "free_agents",
  "roster_cleanup_payload",
  "roster_cleanup",
  "draft_state",
  "draft",
  "draft_completed",
  "draft_started",
  "next_season_generated",
];

function applyLeanProgressFields(merged, incoming) {
  for (const key of LEAN_MERGE_PROGRESS_KEYS) {
    if (incoming[key] !== undefined && incoming[key] !== null) {
      merged[key] = incoming[key];
    }
  }
  return merged;
}

function isNonemptyNarrativeUniverse(nu) {
  return nu && typeof nu === "object" && !Array.isArray(nu) && Object.keys(nu).length > 0;
}

function mergeNarrativeUniverse(prior, incoming) {
  if (!incoming || typeof incoming !== "object" || !("narrative_universe" in incoming)) {
    return prior?.narrative_universe;
  }
  const next = incoming.narrative_universe;
  if (!next || typeof next !== "object" || Array.isArray(next)) {
    return next ?? prior?.narrative_universe;
  }
  if (!isNonemptyNarrativeUniverse(next) && isNonemptyNarrativeUniverse(prior?.narrative_universe)) {
    return prior.narrative_universe;
  }
  return next;
}

function mergeFranchisePayload(prev, incoming) {
  if (!incoming || typeof incoming !== "object") return prev;
  const prior = prev && typeof prev === "object" ? prev : {};
  if (leanStateIdentity(prior, incoming)) {
    const merged = {
      ...prior,
      trade_demand_crisis: incoming.trade_demand_crisis ?? prior.trade_demand_crisis,
      pending_decisions: incoming.pending_decisions ?? prior.pending_decisions,
      pendingDecisions: incoming.pendingDecisions ?? prior.pendingDecisions,
      pending_ui_popups: incoming.pending_ui_popups ?? prior.pending_ui_popups,
      narrative_summary: incoming.narrative_summary ?? prior.narrative_summary,
      flags: incoming.flags ?? prior.flags,
    };
    return applyLeanProgressFields(merged, incoming);
  }
  const revisionChanged =
    incoming?.stats_revision != null && incoming.stats_revision !== prior.stats_revision;
  const prospectRevChanged =
    incoming?.prospect_revision != null && incoming.prospect_revision !== prior.prospect_revision;
  const calendarChanged =
    incoming?.franchise_today_iso != null && incoming.franchise_today_iso !== prior.franchise_today_iso;
  const cursorChanged =
    incoming?.calendar_cursor != null && incoming.calendar_cursor !== prior.calendar_cursor;
  const scoutingChanged =
    incoming?.scouting_as_of_iso != null && incoming.scouting_as_of_iso !== prior.scouting_as_of_iso;
  const draftStale = prospectRevChanged || calendarChanged || cursorChanged || scoutingChanged;
  return {
    ...prior,
    ...incoming,
    roster: incoming?.roster ?? prior.roster,
    lines: incoming?.lines ?? prior.lines,
    roster_browser: incoming?.roster_browser ?? (revisionChanged ? undefined : prior.roster_browser),
    draft_class_rankings: incoming?.draft_class_rankings ?? (draftStale ? undefined : prior.draft_class_rankings),
    draft_class_hud: incoming?.draft_class_hud ?? (draftStale ? undefined : prior.draft_class_hud),
    offseason_stage: incoming.offseason_stage ?? prior.offseason_stage,
    playoff_live: incoming.playoff_live ?? prior.playoff_live,
    playoff_payload: incoming.playoff_payload ?? prior.playoff_payload,
    draft: incoming.draft ?? prior.draft,
    draft_state: incoming.draft_state ?? prior.draft_state,
    narrative_universe: mergeNarrativeUniverse(prior, incoming),
  };
}

function BreakingNewsLayer({ franchiseState, screen, setScreen }) {
  const sessionId = String(franchiseState?.session_id || getFranchiseSessionId() || "anon");
  const [dismissedBreaking, setDismissedBreaking] = useState(() =>
    readDismissedBreakingKeys(sessionId)
  );
  const alerts = Array.isArray(franchiseState?.narrative_universe?.breaking_alerts)
    ? franchiseState.narrative_universe.breaking_alerts
    : Array.isArray(franchiseState?.narrative_summary?.breaking_alerts)
      ? franchiseState.narrative_summary.breaking_alerts
      : [];
  const pending = activeBreakingAlerts(alerts, dismissedBreaking);
  const active = pending[0] || null;

  useEffect(() => {
    setDismissedBreaking(readDismissedBreakingKeys(sessionId));
  }, [sessionId]);

  const dismissAlerts = useCallback(
    (list) => {
      const batch = Array.isArray(list) ? list : [];
      if (!batch.length) return;
      setDismissedBreaking((prev) => {
        const next = new Set(prev);
        batch.forEach((alert) => {
          const key = breakingAlertKey(alert);
          if (key) next.add(key);
        });
        writeDismissedBreakingKeys(sessionId, next);
        return next;
      });
    },
    [sessionId]
  );

  useEffect(() => {
    if (!active?.level || screen === SCREENS.STORYLINES) return;
    playGlobalBreakingSting(active.level);
  }, [active?.storyline_id, active?.headline, active?.level, screen]);

  if (!active || screen === SCREENS.STORYLINES || screen === SCREENS.SETUP || !franchiseState) return null;

  return (
    <div
      className="nhl-breaking-global"
      role="status"
      style={{
        position: "fixed",
        top: 12,
        right: 12,
        zIndex: 11000,
        width: "min(360px, calc(100vw - 24px))",
        border: "1px solid rgba(255, 96, 109, 0.55)",
        borderTop: "3px solid #ff606d",
        background: "linear-gradient(180deg, rgba(40, 8, 12, 0.98), rgba(9, 25, 38, 0.98))",
        padding: "12px 14px",
        boxShadow: "0 12px 32px rgba(0,0,0,.4)",
      }}
    >
      <p style={{ margin: "0 0 4px", fontSize: 10, fontWeight: 900, letterSpacing: ".12em", textTransform: "uppercase", color: "#ff606d" }}>
        Breaking · {String(active.level || "major").replace(/_/g, " ")}
        {pending.length > 1 ? ` · ${pending.length} alerts` : ""}
      </p>
      <strong style={{ display: "block", fontSize: 13, lineHeight: 1.35, marginBottom: 8 }}>
        {String(active.headline || "Major league development")}
      </strong>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          type="button"
          onClick={() => {
            dismissAlerts(pending);
            setScreen?.(SCREENS.STORYLINES);
          }}
          style={{
            border: "1px solid rgba(73, 231, 240, 0.5)",
            borderRadius: 6,
            background: "rgba(19, 216, 231, 0.18)",
            color: "#13d8e7",
            padding: "6px 10px",
            fontSize: 11,
            fontWeight: 800,
            cursor: "pointer",
          }}
        >
          Open wire
        </button>
        <button
          type="button"
          onClick={() => dismissAlerts(pending)}
          style={{
            border: "1px solid rgba(156, 218, 236, 0.2)",
            borderRadius: 6,
            background: "transparent",
            color: "#e9f7fb",
            padding: "6px 10px",
            fontSize: 11,
            fontWeight: 800,
            cursor: "pointer",
          }}
        >
          Dismiss{pending.length > 1 ? " all" : ""}
        </button>
      </div>
    </div>
  );
}

export function GameUIProvider({ children }) {
  const [screen, setScreenState] = useState(() =>
    getFranchiseSessionId() ? SCREENS.HUB : SCREENS.SETUP
  );
  const screenRef = useRef(screen);
  const setScreen = useCallback((next) => {
    const to = typeof next === "function" ? next(screenRef.current) : next;
    const from = screenRef.current;
    if (to !== from) {
      markNavigation(from, to);
    }
    screenRef.current = to;
    setScreenState(to);
  }, []);
  const [hubMenuIndex, setHubMenuIndex] = useState(1);
  const [rosterRowIndex, setRosterRowIndex] = useState(0);
  const [settingsRowIndex, setSettingsRowIndex] = useState(0);
  const [setupTeamIndex, setSetupTeamIndex] = useState(-1);
  const [setupGamesPerTeam, setSetupGamesPerTeam] = useState(82);
  const [capLedgerTab, setCapLedgerTab] = useState("contracts");
  const [statsCentralTab, setStatsCentralTab] = useState("overview");
  const [commandPlaceholder, setCommandPlaceholder] = useState(null);

  const [teams, setTeams] = useState([]);
  const teamsRef = useRef(teams);
  teamsRef.current = teams;
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [gmName, setGmName] = useState("");
  const [playerUniverse, setPlayerUniverse] = useState("generated");
  const [injuriesEnabled, setInjuriesEnabledState] = useState(readInjuriesPref);
  const [franchiseState, setFranchiseState] = useState(() => {
    if (!getFranchiseSessionId()) return null;
    const snap = readFranchiseHubSnapshot();
    return snap && typeof snap === "object" ? snap : null;
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionBootstrapping, setSessionBootstrapping] = useState(() =>
    Boolean(getFranchiseSessionId())
  );
  const [advancing, setAdvancing] = useState(false);
  const [franchiseEventForceOpen, setFranchiseEventForceOpen] = useState(false);
  const [worldJuniorsOpen, setWorldJuniorsOpen] = useState(false);
  const [wjcEventSnapshot, setWjcEventSnapshot] = useState(null);
  const [pendingDraftProspectId, setPendingDraftProspectId] = useState(null);
  const [pendingMeetingPlayerId, setPendingMeetingPlayerId] = useState(null);
  const [pendingSocialNav, setPendingSocialNav] = useState(null);
  const [hubWarmup, setHubWarmup] = useState(() => ({
    [HUB_WARMUP_STAGES.ENVIRONMENT]: "waiting",
    [HUB_WARMUP_STAGES.CRESTS]: "waiting",
    [HUB_WARMUP_STAGES.OPERATIONS]: "waiting",
  }));
  const hubWarmupRef = useRef({ started: new Set(), promises: [] });
  const narrativeHydratedRef = useRef(false);

  useEffect(() => {
    narrativeHydratedRef.current = false;
  }, [franchiseState?.session_id]);

  const [ruleSliders, setRuleSliders] = useState({
    roughing: 50,
    hooking: 50,
    slashing: 50,
    interference: 50,
  });

  const expireFranchiseSession = useCallback(() => {
    clearFranchiseSession();
    resetFranchiseStateCache();
    setFranchiseState(null);
    setScreen(SCREENS.SETUP);
    // Silently return to setup — no expired-session banner.
    setError(null);
  }, []);

  const handleFranchiseApiError = useCallback(
    (e, { rethrow = false } = {}) => {
      if (isExpiredFranchiseSessionError(e)) {
        expireFranchiseSession(formatFranchiseApiError(e));
        if (rethrow) throw e;
        return true;
      }
      setError(formatFranchiseApiError(e));
      if (rethrow) throw e;
      return false;
    },
    [expireFranchiseSession]
  );

  const refreshFranchise = useCallback(async ({ crisisTick = false } = {}) => {
    if (!getFranchiseSessionId()) return;
    setError(null);
    const t0 = performance.now();
    try {
      const s = await getFranchiseState({ crisisTick });
      // Authoritative lean state always wins for identity fields that can change
      // with backend code (roster contracts, lines, morale, etc.).
      setFranchiseState((prev) => {
        if (!prev || typeof prev !== "object") {
          writeFranchiseHubSnapshot(s);
          return s;
        }
        const merged = mergeFranchisePayload(prev, s);
        writeFranchiseHubSnapshot(merged);
        return merged;
      });
      perfRecord("ui.refresh_franchise", performance.now() - t0);
    } catch (e) {
      perfRecord("ui.refresh_franchise", performance.now() - t0, { error: true });
      if (handleFranchiseApiError(e)) return;
    }
  }, [handleFranchiseApiError]);

  useEffect(() => {
    if (!franchiseState?.trade_demand_crisis) return undefined;
    const tick = async () => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      try {
        const res = await getFranchiseCrisis();
        if (res?.trade_demand_crisis !== undefined) {
          setFranchiseState((prev) =>
            prev ? { ...prev, trade_demand_crisis: res.trade_demand_crisis } : prev
          );
        }
      } catch {
        // Ignore transient crisis poll failures.
      }
    };
    tick();
    const timer = setInterval(tick, 2000);
    const onVisibility = () => {
      if (document.visibilityState === "visible") tick();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [franchiseState?.trade_demand_crisis?.demand_id]);

  const mergeFranchiseState = useCallback((nextState) => {
    if (!nextState || typeof nextState !== "object") return;
    setFranchiseState((prev) => {
      const merged = mergeFranchisePayload(prev, nextState);
      writeFranchiseHubSnapshot(merged);
      return merged;
    });
  }, []);

  const hydrateFranchiseNarrative = useCallback(async (options = {}) => {
    const force = options?.force === true;
    if (!getFranchiseSessionId()) return null;
    if (!force && narrativeHydratedRef.current) return null;
    narrativeHydratedRef.current = true;
    try {
      const data = await getFranchiseNarrative();
      setFranchiseState((prev) => {
        if (!prev) return prev;
        const merged = {
          ...prev,
          narrative_revision: data?.narrative_revision ?? prev.narrative_revision,
          narrative_universe: data?.narrative_universe ?? prev.narrative_universe,
        };
        writeFranchiseHubSnapshot(merged);
        return merged;
      });
      return data;
    } catch (e) {
      narrativeHydratedRef.current = false;
      handleFranchiseApiError(e);
      return null;
    }
  }, [handleFranchiseApiError]);

  const hydrateFranchiseHeavyState = useCallback(
    async ({
      includeRosterBrowser = true,
      includeDraftClassRankings = true,
      includeDraftClassHud = true,
      includeNhlCalendarFull = false,
    } = {}) => {
      if (!getFranchiseSessionId()) return null;
      let skipRoster = false;
      let skipDraft = false;
      let skipCalendar = false;
      setFranchiseState((prev) => {
        if (includeRosterBrowser && prev?.roster_browser) skipRoster = true;
        if (includeDraftClassRankings && prev?.draft_class_rankings) {
          const boardRev = Number(
            prev.draft_class_rankings.prospect_revision ?? prev.draft_class_rankings.revision ?? NaN
          );
          const liveRev = Number(prev.prospect_revision ?? NaN);
          const boardIso = String(prev.draft_class_rankings.stats_as_of_iso || prev.draft_class_rankings.scouting_as_of_iso || "");
          const liveIso = String(
            prev.franchise_today_iso
            || prev.scouting_as_of_iso
            || prev.nhl_today?.iso
            || ""
          );
          const boardCursor = Number(prev.draft_class_rankings.calendar_cursor ?? NaN);
          const liveCursor = Number(prev.calendar_cursor ?? NaN);
          const hudIso = String(prev.draft_class_hud?.scouting_as_of_iso || prev.draft_class_hud?.franchise_today_iso || "");
          const revMatch = Number.isFinite(boardRev) && Number.isFinite(liveRev) && boardRev === liveRev;
          const isoMatch = !boardIso || !liveIso || boardIso === liveIso;
          const cursorMatch =
            !Number.isFinite(boardCursor) ||
            !Number.isFinite(liveCursor) ||
            boardCursor === liveCursor;
          const hudMatch = !hudIso || !liveIso || hudIso === liveIso || hudIso === String(prev.scouting_as_of_iso || "");
          skipDraft = revMatch && isoMatch && cursorMatch && hudMatch;
        }
        if (includeNhlCalendarFull && Array.isArray(prev?.nhl_calendar_full) && prev.nhl_calendar_full.length > 120) {
          skipCalendar = true;
        }
        return prev;
      });
      const needRoster = includeRosterBrowser && !skipRoster;
      const needDraft = (includeDraftClassRankings || includeDraftClassHud) && !skipDraft;
      const needCalendar = includeNhlCalendarFull && !skipCalendar;
      if (!needRoster && !needDraft && !needCalendar) return null;
      try {
        const heavy = await getFranchiseStateHeavy({
          includeRosterBrowser: needRoster,
          includeDraftClassRankings: needDraft && includeDraftClassRankings,
          includeDraftClassHud: needDraft && includeDraftClassHud,
          includeNhlCalendarFull: needCalendar,
        });
        setFranchiseState((prev) => {
          const merged = { ...(prev || {}), ...(heavy || {}) };
          writeFranchiseHubSnapshot(merged);
          return merged;
        });
        return heavy;
      } catch (e) {
        if (handleFranchiseApiError(e)) return null;
        return null;
      }
    },
    [handleFranchiseApiError]
  );

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const sid = getFranchiseSessionId();
      if (!sid) {
        if (!cancelled) {
          setSessionBootstrapping(false);
          setScreen(SCREENS.SETUP);
        }
        return;
      }

      setSessionBootstrapping(true);
      const backendRestarted = await syncFranchiseSessionWithBackend();
      if (cancelled) return;

      if (backendRestarted) {
        clearFranchiseSession();
        resetFranchiseStateCache();
        setFranchiseState(null);
        setScreen(SCREENS.SETUP);
        setError(null);
        setSessionBootstrapping(false);
        return;
      }

      setScreen(SCREENS.HUB);
      try {
        await refreshFranchise();
        await hydrateFranchiseHeavyState({
          includeRosterBrowser: true,
          includeDraftClassRankings: false,
          includeDraftClassHud: false,
        });
      } finally {
        if (!cancelled) setSessionBootstrapping(false);
      }
    })();

    const onBackendChanged = () => {
      expireFranchiseSession(
        "Backend process changed while the app was open. Start a new franchise."
      );
    };
    window.addEventListener("nhl-franchise-backend-changed", onBackendChanged);

    return () => {
      cancelled = true;
      window.removeEventListener("nhl-franchise-backend-changed", onBackendChanged);
    };
  }, [refreshFranchise, expireFranchiseSession, hydrateFranchiseHeavyState]);

  const loadTeams = useCallback(async () => {
    const fallback = buildDefaultFranchiseTeamList();
    if (!teamsRef.current.length) {
      setTeams(fallback);
    }
    setTeamsLoading(true);
    try {
      await resolveApiBaseUrl();
      const t = await listTeams();
      const list = Array.isArray(t) && t.length > 0 ? t : fallback;
      const current = teamsRef.current.length ? teamsRef.current : fallback;
      if (!sameTeamRoster(current, list)) {
        setSetupTeamIndex((index) => remapTeamIndex(current, index, list));
        setTeams(list);
      }
    } catch (e) {
      console.warn(
        "Franchise teams API unavailable; using local 32-club list.",
        e
      );
    } finally {
      setTeamsLoading(false);
    }
  }, []);

  const setInjuriesEnabled = useCallback((enabled) => {
    const next = Boolean(enabled);
    setInjuriesEnabledState(next);
    try {
      localStorage.setItem(INJURIES_STORAGE_KEY, String(next));
    } catch {
      /* ignore */
    }
  }, []);

  const markHubWarmup = useCallback((stage, status) => {
    setHubWarmup((current) =>
      current[stage] === status ? current : { ...current, [stage]: status }
    );
  }, []);

  const primeHubAssets = useCallback(
    (stage) => {
      const store = hubWarmupRef.current;
      if (!stage || store.started.has(stage)) return;

      let job = null;

      if (stage === HUB_WARMUP_STAGES.ENVIRONMENT) {
        job = Promise.all([
          prefetchImage(hubWallTextureSrc),
          prefetchResource(officeFontBold),
        ]);
      } else if (stage === HUB_WARMUP_STAGES.CRESTS) {
        const roster = teams.length ? teams : buildDefaultFranchiseTeamList();
        const crests = Array.from(
          new Set(
            roster
              .map((club) => resolveFranchiseTeamLogo(club, club?.name))
              .filter(Boolean)
          )
        );
        job = prefetchImagesBatched(crests);
      } else if (stage === HUB_WARMUP_STAGES.OPERATIONS) {
        /*
          Records can only be pulled once a session exists. Before that this
          stage stays queued rather than reporting itself finished, so the
          real hydration still happens when the franchise actually starts.
        */
        if (!getFranchiseSessionId()) {
          return;
        }
        job = hydrateFranchiseHeavyState({ includeRosterBrowser: true });
      }

      if (!job) {
        return;
      }

      store.started.add(stage);
      markHubWarmup(stage, "loading");

      store.promises.push(
        Promise.resolve(job)
          .catch(() => null)
          .then(() => {
            markHubWarmup(stage, "ready");
          })
      );
    },
    [markHubWarmup, teams, hydrateFranchiseHeavyState]
  );

  const awaitHubReady = useCallback(async () => {
    const store = hubWarmupRef.current;
    // Drain repeatedly: a warm-up stage may queue another while it settles.
    for (let pass = 0; pass < 4; pass += 1) {
      const pending = store.promises.slice();
      if (!pending.length) break;
      await Promise.allSettled(pending);
      if (store.promises.length === pending.length) break;
    }
  }, []);

  const beginFranchise = useCallback(async () => {
    if (!teams.length) {
      const message = "No team selected. Check that the API is running, then try again.";
      setError(message);
      return { ok: false, error: message };
    }
    const selectedTeam = teams[setupTeamIndex];
    if (!selectedTeam) {
      const message = "Choose a club before opening hockey operations.";
      setError(message);
      return { ok: false, error: message };
    }
    setError(null);
    setLoading(true);
    try {
      await resolveApiBaseUrl({ force: true });
      clearFranchiseSession();
      resetFranchiseStateCache();
      setFranchiseState(null);
      await resetFranchiseServerSessions();
      const t = selectedTeam;
      const teamQuery = String(
        t?.team_id ??
          t?.teamId ??
          t?.id ??
          t?.abbr ??
          t?.abbreviation ??
          t?.name ??
          ""
      ).trim();
      if (!teamQuery) {
        throw new Error("No valid team id was found for the selected team.");
      }
      const universe =
        playerUniverse === "real_nhl" ? "real_nhl" : "generated";
      const res = await startFranchise({
        team_query: teamQuery,
        head_coach_name: gmName.trim() || "General Manager",
        coach_archetype: "balanced",
        games_per_team: Number(setupGamesPerTeam) || 82,
        injuries_enabled: injuriesEnabled,
        player_universe: universe,
      });
      const nextSessionId = String(res?.session_id || "").trim();
      let nextState = res?.state ?? res?.franchiseState;
      if (nextState == null && res && typeof res === "object") {
        const looksLikeApiEnvelope =
          Object.prototype.hasOwnProperty.call(res, "session_id") &&
          (Object.prototype.hasOwnProperty.call(res, "ok") ||
            Object.prototype.hasOwnProperty.call(res, "state"));
        if (!looksLikeApiEnvelope) {
          nextState = res;
        }
      }
      if (!nextSessionId) {
        throw new Error("Backend returned no session id.");
      }
      setFranchiseSessionId(nextSessionId);
      if (!nextState || typeof nextState !== "object") {
        // Some backend variants return only session id from /start.
        // In that case, fetch the authoritative state and continue.
        nextState = await getFranchiseState();
      }
      if (!nextState || typeof nextState !== "object") {
        throw new Error("Backend returned no franchise state.");
      }
      setFranchiseState(nextState);
      writeFranchiseHubSnapshot(nextState);
      setHubMenuIndex(1);
      setScreen(SCREENS.HUB);
      // Finish hub warmup in the background — don't keep the loading screen up for it.
      primeHubAssets(HUB_WARMUP_STAGES.ENVIRONMENT);
      primeHubAssets(HUB_WARMUP_STAGES.CRESTS);
      primeHubAssets(HUB_WARMUP_STAGES.OPERATIONS);
      void awaitHubReady();
      return { ok: true };
    } catch (e) {
      console.error("[beginFranchise]", e);
      const message = formatFranchiseApiError(e);
      setError(message);
      return { ok: false, error: message };
    } finally {
      setLoading(false);
    }
  }, [
    teams,
    setupTeamIndex,
    gmName,
    setupGamesPerTeam,
    injuriesEnabled,
    playerUniverse,
    primeHubAssets,
    awaitHubReady,
  ]);

  const onAdvanceFranchise = useCallback(
    async ({ mode = "day", count = 1, auto_resolve: autoResolve } = {}) => {
      if (!franchiseState) return null;
      const phase = String(franchiseState.phase || franchiseState.season_phase || "");
      if (phase === "complete") return null;
      // Free agency: Hub day/week/month advances the living FA market clock.
      if (["post_cup", "offseason"].includes(phase)) {
        const stage = String(franchiseState.offseason_stage || "");
        const faOpen =
          stage === "free_agency" ||
          Boolean(franchiseState.free_agency_open) ||
          String(franchiseState.free_agency_market?.market_status || "") === "open";
        if (!faOpen) return null;
        const faMode = String(mode || "day").toLowerCase();
        const faCount = Math.max(1, Number(count) || 1);
        let days = faCount;
        if (faMode === "week" || faMode === "weeks") days = 7 * faCount;
        else if (faMode === "month" || faMode === "months") days = 30 * faCount;
        else if (faMode === "season") days = 14;
        setAdvancing(true);
        setError(null);
        try {
          const res = await advanceFreeAgencyDay(days);
          if (res?.state) mergeFranchiseState(res.state);
          return res;
        } catch (e) {
          handleFranchiseApiError(e);
          return null;
        } finally {
          setAdvancing(false);
        }
      }
      const m = String(mode || "day").toLowerCase();
      const c = Math.max(1, Number(count) || 1);
      const soloDay = m === "day" && c === 1;
      if (soloDay && !franchiseState?.flags?.can_advance) return null;
      const multiDayBlock =
        m === "days" || m === "games" || (m === "day" && c > 1);
      const seasonBlock = m === "season";
      const bulkSim = multiDayBlock || seasonBlock;
      const effectiveAuto = autoResolve === undefined ? Boolean(bulkSim) : Boolean(autoResolve);
      setAdvancing(true);
      setError(null);
      let res = null;
      try {
        if (bulkSim) {
          if (!effectiveAuto && franchiseState?.flags && !franchiseState.flags.can_advance) return null;
          const targetMode = seasonBlock ? "season" : m === "games" ? "games" : "days";
          const targetCount = seasonBlock ? 1 : c;
          res = await advanceFranchise({
            mode: targetMode,
            count: targetCount,
            auto_resolve: effectiveAuto,
          });
          mergeFranchiseState(res.state);
        } else {
          res = await advanceFranchise({ mode: "day", count: 1, auto_resolve: effectiveAuto });
          mergeFranchiseState(res.state);
        }
      } catch (e) {
        // Never rethrow — Axios Network Error must not open the React crash overlay.
        handleFranchiseApiError(e);
        return null;
      } finally {
        setAdvancing(false);
      }
      return res;
    },
    [franchiseState, handleFranchiseApiError]
  );

  const onAdvanceDay = useCallback(async () => {
    await onAdvanceFranchise({ mode: "day", count: 1, auto_resolve: true });
  }, [onAdvanceFranchise]);

  const onEnterPlayoffs = useCallback(async () => {
    if (!franchiseState) return null;
    setAdvancing(true);
    setError(null);
    try {
      const res = await enterPlayoffs();
      if (res?.state) {
        mergeFranchiseState(res.state);
        setFranchiseEventForceOpen(true);
      }
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      return null;
    } finally {
      setAdvancing(false);
    }
  }, [franchiseState, handleFranchiseApiError, setFranchiseEventForceOpen]);

  const onAdvanceSeasonPhase = useCallback(async (payload = {}) => {
    setAdvancing(true);
    setError(null);
    try {
      const res = await advanceSeasonPhase(payload);
      if (res?.state) mergeFranchiseState(res.state);
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      return null;
    } finally {
      setAdvancing(false);
    }
  }, [handleFranchiseApiError]);

  const onContinueOffseason = useCallback(async () => {
    setAdvancing(true);
    setError(null);
    try {
      const fromStage = String(
        franchiseState?.offseason_stage ||
          (String(franchiseState?.phase || franchiseState?.season_phase || "").toLowerCase() === "post_cup"
            ? "awards"
            : "") ||
          ""
      );
      const res = await continueOffseason({ from_stage: fromStage });
      if (res?.state) {
        mergeFranchiseState(res.state);
        const nextPhase = String(res.state.phase || res.state.season_phase || "").toLowerCase();
        // After Enter Preseason, land on the hub — do not reopen last year's
        // awards / playoff cinematic from a stale pending popup.
        if (nextPhase !== "preseason" && nextPhase !== "regular") {
          setFranchiseEventForceOpen(true);
        } else {
          setFranchiseEventForceOpen(false);
        }
      }
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      return null;
    } finally {
      setAdvancing(false);
    }
  }, [franchiseState, handleFranchiseApiError, setFranchiseEventForceOpen]);

  const onReopenOffseasonStage = useCallback(
    async (stage = "free_agency") => {
      setAdvancing(true);
      setError(null);
      try {
        const res = await reopenOffseasonStage({ stage });
        if (res?.state) {
          mergeFranchiseState(res.state);
          setFranchiseEventForceOpen(true);
        }
        return res;
      } catch (e) {
        handleFranchiseApiError(e);
        return null;
      } finally {
        setAdvancing(false);
      }
    },
    [handleFranchiseApiError, setFranchiseEventForceOpen]
  );

  const openFranchiseEvent = useCallback(() => {
    setFranchiseEventForceOpen(true);
  }, [setFranchiseEventForceOpen]);

  const openWorldJuniors = useCallback(() => {
    const state = franchiseState;
    const pops =
      Array.isArray(state?.pending_ui_popups) && state.pending_ui_popups.length
        ? state.pending_ui_popups
        : Array.isArray(state?.pendingUiPopups)
          ? state.pendingUiPopups
          : [];
    const wjcPop = pops.find(
      (pop) => pop && (pop.kind === "wjc_tournament" || pop.wjc_live === true || pop.wjc_phase)
    );
    const resolved = resolveWorldJuniorsPayload(state, wjcPop || null);
    setWjcEventSnapshot(resolved.hasData && resolved.raw ? { ...resolved.raw } : null);
    setWorldJuniorsOpen(true);
  }, [franchiseState]);

  const openDraftClassFromWjc = useCallback(
    (prospectId = null) => {
      if (prospectId) setPendingDraftProspectId(String(prospectId));
      setWorldJuniorsOpen(false);
      setWjcEventSnapshot(null);
      setScreen(SCREENS.DRAFT_CLASS);
    },
    [setScreen]
  );

  const onGenerateNextSeason = useCallback(async () => {
    setAdvancing(true);
    setError(null);
    try {
      const res = await generateNextSeason();
      if (res?.state) {
        mergeFranchiseState(res.state);
        // Roster Check → Generate lands on the September hub, not another cinematic.
        setFranchiseEventForceOpen(false);
      }
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      return null;
    } finally {
      setAdvancing(false);
    }
  }, [handleFranchiseApiError, setFranchiseEventForceOpen]);

  const onResolveDecision = useCallback(async (decisionId, choiceId) => {
    setError(null);
    try {
      const res = await submitDecision(decisionId, choiceId);
      mergeFranchiseState(res.state);
    } catch (e) {
      handleFranchiseApiError(e);
    }
  }, [handleFranchiseApiError]);

  const onResolveStorylineChoice = useCallback(async (storylineId, choiceId) => {
    setError(null);
    try {
      const res = await submitStorylineChoice(storylineId, choiceId);
      mergeFranchiseState(res.state);
      await hydrateFranchiseNarrative?.({ force: true });
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      throw e;
    }
  }, [handleFranchiseApiError, hydrateFranchiseNarrative]);

  const onDismissShowcasePopups = useCallback(async (ids) => {
    const rawIds = ids || [];
    const dropFirst = rawIds.some((x) => String(x || "").startsWith("__drop_first__:"));
    const cleanIds = rawIds.map((x) => String(x || "").trim()).filter((x) => x && !x.startsWith("__drop_first__:"));
    if (!getFranchiseSessionId() || (!cleanIds.length && !dropFirst)) return;
    setError(null);
    const drop = new Set(cleanIds);
    setFranchiseState((prev) => {
      if (!prev) return prev;
      let pending = prev.pending_ui_popups || [];
      if (dropFirst && pending.length) {
        pending = pending.slice(1);
      } else if (drop.size) {
        pending = pending.filter((p) => !drop.has(String(p?.id || "")));
      }
      const merged = { ...prev, pending_ui_popups: pending };
      writeFranchiseHubSnapshot(merged);
      return merged;
    });
    if (!cleanIds.length) return;
    try {
      await dismissFranchisePopups(cleanIds);
    } catch (e) {
      handleFranchiseApiError(e);
    }
  }, [handleFranchiseApiError]);

  const closeWorldJuniors = useCallback(async () => {
    setWorldJuniorsOpen(false);
    setWjcEventSnapshot(null);
    const state = franchiseState;
    const pops =
      Array.isArray(state?.pending_ui_popups) && state.pending_ui_popups.length
        ? state.pending_ui_popups
        : Array.isArray(state?.pendingUiPopups)
          ? state.pendingUiPopups
          : [];
    const wjcPop = pops.find(
      (pop) => pop && (pop.kind === "wjc_tournament" || pop.wjc_live === true || pop.wjc_phase)
    );
    if (wjcPop?.id) {
      try {
        const res = await dismissFranchisePopups([wjcPop.id]);
        if (res?.state) mergeFranchiseState(res.state);
      } catch {
        /* overlay already closed */
      }
    }
  }, [franchiseState]);

  const goNewFranchise = useCallback(() => {
    clearFranchiseSession();
    resetFranchiseStateCache();
    setFranchiseState(null);
    setScreen(SCREENS.SETUP);
    resetFranchiseServerSessions();
  }, []);

  const openCommandPlaceholder = useCallback((payload) => {
    setCommandPlaceholder({
      title: payload?.title || "Coming Soon",
      subtitle:
        payload?.subtitle || "This feature does not have a dedicated screen yet.",
      description: payload?.description || "",
      targetId: payload?.targetId || payload?.target || "",
    });
    setScreen(SCREENS.PLACEHOLDER);
  }, []);

  const openHubMenu = useCallback(
    (idx) => {
      setHubMenuIndex(idx);
      const item = HUB_MENU[idx];
      if (!item) return;
      if (item.id === "roster") {
        setRosterRowIndex(0);
        setScreen(SCREENS.ROSTER);
      } else if (item.id === "calendar") {
        setScreen(SCREENS.CALENDAR);
      } else if (item.id === "stats") {
        setScreen(SCREENS.STATS);
      } else if (item.id === "ops") {
        setScreen(SCREENS.TRADE);
      } else if (item.id === "draft_class") {
        setScreen(SCREENS.DRAFT_CLASS);
      } else if (item.id === "scouting") {
        setScreen(SCREENS.SCOUTING);
      } else if (item.id === "chemistry") {
        setScreen(SCREENS.CHEMISTRY);
      } else if (item.id === "office") {
        setScreen(SCREENS.HUB);
      } else if (item.id === "settings") {
        setSettingsRowIndex(0);
        setScreen(SCREENS.SETTINGS);
      } else if (item.id === "new") {
        goNewFranchise();
      }
    },
    [goNewFranchise]
  );

  const adjustSlider = useCallback((key, delta) => {
    setRuleSliders((prev) => {
      const step = 5;
      const v = Math.round((prev[key] + delta * step) / step) * step;
      return { ...prev, [key]: Math.max(0, Math.min(100, v)) };
    });
  }, []);

  const value = useMemo(
    () => ({
      screen,
      setScreen,
      hubMenuIndex,
      setHubMenuIndex,
      rosterRowIndex,
      setRosterRowIndex,
      settingsRowIndex,
      setSettingsRowIndex,
      setupTeamIndex,
      setSetupTeamIndex,
      setupGamesPerTeam,
      setSetupGamesPerTeam,
      teams,
      teamsLoading,
      loadTeams,
      gmName,
      setGmName,
      playerUniverse,
      setPlayerUniverse,
      injuriesEnabled,
      setInjuriesEnabled,
      franchiseState,
      setFranchiseState,
      error,
      setError,
      loading,
      sessionBootstrapping,
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      hydrateFranchiseHeavyState,
      hydrateFranchiseNarrative,
      mergeFranchiseState,
      expireFranchiseSession,
      beginFranchise,
      onAdvanceDay,
      onAdvanceFranchise,
      onEnterPlayoffs,
      onAdvanceSeasonPhase,
      onContinueOffseason,
      onReopenOffseasonStage,
      onGenerateNextSeason,
      openFranchiseEvent,
      franchiseEventForceOpen,
      setFranchiseEventForceOpen,
      worldJuniorsOpen,
      openWorldJuniors,
      closeWorldJuniors,
      wjcEventSnapshot,
      pendingDraftProspectId,
      setPendingDraftProspectId,
      pendingMeetingPlayerId,
      setPendingMeetingPlayerId,
      pendingSocialNav,
      setPendingSocialNav,
      openDraftClassFromWjc,
      onResolveDecision,
      onResolveStorylineChoice,
      onDismissShowcasePopups,
      openHubMenu,
      goNewFranchise,
      capLedgerTab,
      setCapLedgerTab,
      statsCentralTab,
      setStatsCentralTab,
      commandPlaceholder,
      setCommandPlaceholder,
      openCommandPlaceholder,
      hubWarmup,
      primeHubAssets,
      awaitHubReady,
    }),
    [
      screen,
      hubMenuIndex,
      rosterRowIndex,
      settingsRowIndex,
      setupTeamIndex,
      setupGamesPerTeam,
      teams,
      teamsLoading,
      loadTeams,
      gmName,
      playerUniverse,
      injuriesEnabled,
      setInjuriesEnabled,
      franchiseState,
      error,
      loading,
      sessionBootstrapping,
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      hydrateFranchiseHeavyState,
      hydrateFranchiseNarrative,
      mergeFranchiseState,
      expireFranchiseSession,
      beginFranchise,
      onAdvanceDay,
      onAdvanceFranchise,
      onEnterPlayoffs,
      onAdvanceSeasonPhase,
      onContinueOffseason,
      onReopenOffseasonStage,
      onGenerateNextSeason,
      openFranchiseEvent,
      franchiseEventForceOpen,
      setFranchiseEventForceOpen,
      worldJuniorsOpen,
      openWorldJuniors,
      closeWorldJuniors,
      wjcEventSnapshot,
      pendingDraftProspectId,
      setPendingDraftProspectId,
      pendingMeetingPlayerId,
      setPendingMeetingPlayerId,
      pendingSocialNav,
      setPendingSocialNav,
      openDraftClassFromWjc,
      onResolveDecision,
      onResolveStorylineChoice,
      onDismissShowcasePopups,
      openHubMenu,
      goNewFranchise,
      capLedgerTab,
      statsCentralTab,
      commandPlaceholder,
      openCommandPlaceholder,
      hubWarmup,
      primeHubAssets,
      awaitHubReady,
    ]
  );

  return (
    <FranchiseStateContext.Provider value={franchiseState}>
      <GameUIContext.Provider value={value}>
        {children}
      {advancing ? <AdvancingOverlay /> : null}
      <BreakingNewsLayer franchiseState={franchiseState} screen={screen} setScreen={setScreen} />
      <ShowcasePopupLayer />
      <TradeDemandCrisisOverlay />
      <FranchiseEventLayer />
      {worldJuniorsOpen ? (
        <div className="wjc-event-shell wjc-event-shell--global" role="presentation">
          <WorldJuniorsEvent
            franchiseState={franchiseState}
            eventData={wjcEventSnapshot}
            onClose={closeWorldJuniors}
            onBackToHub={closeWorldJuniors}
            onSimNextTournamentDay={onAdvanceDay}
            onOpenDraftBoard={openDraftClassFromWjc}
          />
        </div>
      ) : null}
      </GameUIContext.Provider>
    </FranchiseStateContext.Provider>
  );
}
