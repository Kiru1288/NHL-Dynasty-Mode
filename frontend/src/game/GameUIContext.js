import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  advanceFranchise,
  advanceFreeAgencyDay,
  advanceSeasonPhase,
  continueOffseason,
  dismissFranchisePopups,
  generateNextSeason,
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
  resetFranchiseServerSessions,
  resolveApiBaseUrl,
  setFranchiseSessionId,
  syncFranchiseSessionWithBackend,
} from "../services/api";
import { markNavigation, record as perfRecord } from "../services/perfProfiler";
import { HUB_MENU, SCREENS, buildDefaultFranchiseTeamList } from "./constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import hubWallTextureSrc from "../pictures/gray-abstract-texture-background.jpg";
import officeFontBold from "../styles/ArchivoBlack-Regular.ttf";
import { ShowcasePopupLayer } from "../components/game/ShowcasePopupLayer";
import { FranchiseEventLayer } from "../components/game/FranchiseEventLayer";
import WorldJuniorsEvent from "../events/worldJuniors/WorldJuniorsEvent";
import { resolveWorldJuniorsPayload } from "../events/worldJuniors/WorldJuniorsMenu";

const GameUIContext = createContext(null);

export const INJURIES_STORAGE_KEY = "nhl_franchise_injuries_enabled";

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
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [gmName, setGmName] = useState("");
  const [playerUniverse, setPlayerUniverse] = useState("generated");
  const [injuriesEnabled, setInjuriesEnabledState] = useState(readInjuriesPref);
  const [franchiseState, setFranchiseState] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [advancing, setAdvancing] = useState(false);
  const [franchiseEventForceOpen, setFranchiseEventForceOpen] = useState(false);
  const [worldJuniorsOpen, setWorldJuniorsOpen] = useState(false);
  const [wjcEventSnapshot, setWjcEventSnapshot] = useState(null);
  const [pendingDraftProspectId, setPendingDraftProspectId] = useState(null);
  const [hubWarmup, setHubWarmup] = useState(() => ({
    [HUB_WARMUP_STAGES.ENVIRONMENT]: "waiting",
    [HUB_WARMUP_STAGES.CRESTS]: "waiting",
    [HUB_WARMUP_STAGES.OPERATIONS]: "waiting",
  }));
  const hubWarmupRef = useRef({ started: new Set(), promises: [] });

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

  const refreshFranchise = useCallback(async () => {
    if (!getFranchiseSessionId()) return;
    setError(null);
    const t0 = performance.now();
    try {
      const s = await getFranchiseState();
      // Authoritative lean state always wins for identity fields that can change
      // with backend code (roster contracts, lines, morale, etc.).
      setFranchiseState((prev) => {
        if (!prev || typeof prev !== "object") return s;
        const revisionChanged =
          s?.stats_revision != null && s.stats_revision !== prev.stats_revision;
        return {
          ...prev,
          ...s,
          roster: s?.roster ?? prev.roster,
          lines: s?.lines ?? prev.lines,
          // Preserve heavy domains until explicitly refreshed, but drop them when
          // lean state advances (trades/games days) so screens re-hydrate fresh data.
          roster_browser: s?.roster_browser ?? (revisionChanged ? undefined : prev.roster_browser),
          draft_class_rankings: s?.draft_class_rankings ?? prev.draft_class_rankings,
          draft_class_hud: s?.draft_class_hud ?? prev.draft_class_hud,
        };
      });
      perfRecord("ui.refresh_franchise", performance.now() - t0);
    } catch (e) {
      perfRecord("ui.refresh_franchise", performance.now() - t0, { error: true });
      if (handleFranchiseApiError(e)) return;
    }
  }, [handleFranchiseApiError]);

  const mergeFranchiseState = useCallback((nextState) => {
    if (!nextState || typeof nextState !== "object") return;
    setFranchiseState((prev) => {
      const prior = prev && typeof prev === "object" ? prev : {};
      const revisionChanged =
        nextState?.stats_revision != null &&
        nextState.stats_revision !== prior.stats_revision;
      return {
        ...prior,
        ...nextState,
        roster: nextState?.roster ?? prior.roster,
        lines: nextState?.lines ?? prior.lines,
        roster_browser:
          nextState?.roster_browser ?? (revisionChanged ? undefined : prior.roster_browser),
        draft_class_rankings: nextState?.draft_class_rankings ?? prior.draft_class_rankings,
        draft_class_hud: nextState?.draft_class_hud ?? prior.draft_class_hud,
      };
    });
  }, []);

  const hydrateFranchiseHeavyState = useCallback(
    async ({
      includeRosterBrowser = true,
      includeDraftClassRankings = true,
      includeDraftClassHud = true,
    } = {}) => {
      if (!getFranchiseSessionId()) return null;
      try {
        const heavy = await getFranchiseStateHeavy({
          includeRosterBrowser,
          includeDraftClassRankings,
          includeDraftClassHud,
        });
        setFranchiseState((prev) => ({
          ...(prev || {}),
          ...(heavy || {}),
        }));
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
      const backendRestarted = await syncFranchiseSessionWithBackend();
      if (cancelled) return;

      if (backendRestarted) {
        clearFranchiseSession();
        resetFranchiseStateCache();
        setFranchiseState(null);
        setScreen(SCREENS.SETUP);
        setError(null);
        return;
      }

      if (getFranchiseSessionId()) {
        setScreen(SCREENS.HUB);
        refreshFranchise();
      } else {
        setScreen(SCREENS.SETUP);
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
  }, [refreshFranchise, expireFranchiseSession]);

  const loadTeams = useCallback(async () => {
    const fallback = buildDefaultFranchiseTeamList();
    setTeams(fallback);
    setSetupTeamIndex((i) =>
      i < 0 ? -1 : Math.min(i, Math.max(0, fallback.length - 1))
    );
    setTeamsLoading(true);
    try {
      await resolveApiBaseUrl();
      const t = await listTeams();
      const list = Array.isArray(t) && t.length > 0 ? t : fallback;
      setTeams(list);
      setSetupTeamIndex((i) =>
        i < 0 ? -1 : Math.min(i, Math.max(0, list.length - 1))
      );
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
        job = Promise.all(crests.map((src) => prefetchImage(src)));
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
      setHubMenuIndex(1);
      // The hallway and the appointment already paid for most of this. Finish
      // whatever the hub still needs before handing the player a live office.
      primeHubAssets(HUB_WARMUP_STAGES.ENVIRONMENT);
      primeHubAssets(HUB_WARMUP_STAGES.CRESTS);
      primeHubAssets(HUB_WARMUP_STAGES.OPERATIONS);
      await awaitHubReady();
      setScreen(SCREENS.HUB);
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
    } catch (e) {
      handleFranchiseApiError(e);
    }
  }, [handleFranchiseApiError]);

  const onDismissShowcasePopups = useCallback(async (ids) => {
    if (!getFranchiseSessionId() || !ids || !ids.length) return;
    setError(null);
    try {
      const res = await dismissFranchisePopups(ids);
      mergeFranchiseState(res.state);
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
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      hydrateFranchiseHeavyState,
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
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      hydrateFranchiseHeavyState,
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
    <GameUIContext.Provider value={value}>
      {children}
      <ShowcasePopupLayer />
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
  );
}
