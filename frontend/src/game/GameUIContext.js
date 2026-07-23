import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  advanceFranchise,
  advanceSeasonPhase,
  continueOffseason,
  dismissFranchisePopups,
  generateNextSeason,
  getFranchiseState,
  getFranchiseStateHeavy,
  listTeams,
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
  setFranchiseSessionId,
  syncFranchiseSessionWithBackend,
} from "../services/api";
import { HUB_MENU, SCREENS, buildDefaultFranchiseTeamList } from "./constants";
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

export function GameUIProvider({ children }) {
  const [screen, setScreen] = useState(() =>
    getFranchiseSessionId() ? SCREENS.HUB : SCREENS.SETUP
  );
  const [hubMenuIndex, setHubMenuIndex] = useState(1);
  const [rosterRowIndex, setRosterRowIndex] = useState(0);
  const [settingsRowIndex, setSettingsRowIndex] = useState(0);
  const [setupTeamIndex, setSetupTeamIndex] = useState(0);
  const [setupGamesPerTeam, setSetupGamesPerTeam] = useState(82);
  const [capLedgerTab, setCapLedgerTab] = useState("contracts");
  const [statsCentralTab, setStatsCentralTab] = useState("overview");
  const [commandPlaceholder, setCommandPlaceholder] = useState(null);

  const [teams, setTeams] = useState([]);
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [gmName, setGmName] = useState("Pat Quinn");
  const [injuriesEnabled, setInjuriesEnabledState] = useState(readInjuriesPref);
  const [franchiseState, setFranchiseState] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [advancing, setAdvancing] = useState(false);
  const [franchiseEventForceOpen, setFranchiseEventForceOpen] = useState(false);
  const [worldJuniorsOpen, setWorldJuniorsOpen] = useState(false);
  const [wjcEventSnapshot, setWjcEventSnapshot] = useState(null);
  const [pendingDraftProspectId, setPendingDraftProspectId] = useState(null);

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
    } catch (e) {
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
    setSetupTeamIndex((i) => Math.min(i, Math.max(0, fallback.length - 1)));
    setTeamsLoading(true);
    try {
      const t = await listTeams();
      const list = Array.isArray(t) && t.length > 0 ? t : fallback;
      setTeams(list);
      setSetupTeamIndex((i) => Math.min(i, Math.max(0, list.length - 1)));
    } catch (e) {
      setError(formatFranchiseApiError(e));
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

  const beginFranchise = useCallback(async () => {
    if (!teams.length) {
      setError("No team selected. Check that the API is running, then try again.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      clearFranchiseSession();
      resetFranchiseStateCache();
      setFranchiseState(null);
      await resetFranchiseServerSessions();
      const t = teams[setupTeamIndex];
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
      const res = await startFranchise({
        team_query: teamQuery,
        head_coach_name: gmName.trim() || "General Manager",
        coach_archetype: "balanced",
        games_per_team: Number(setupGamesPerTeam) || 82,
        injuries_enabled: injuriesEnabled,
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
      setScreen(SCREENS.HUB);
    } catch (e) {
      console.error("[beginFranchise]", e);
      setError(formatFranchiseApiError(e));
    } finally {
      setLoading(false);
    }
  }, [teams, setupTeamIndex, gmName, setupGamesPerTeam, injuriesEnabled]);

  const onAdvanceFranchise = useCallback(
    async ({ mode = "day", count = 1, auto_resolve: autoResolve } = {}) => {
      if (!franchiseState) return null;
      const phase = String(franchiseState.phase || franchiseState.season_phase || "");
      if (phase === "complete") return null;
      if (["post_cup", "offseason"].includes(phase)) return null;
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
      if (res?.state) mergeFranchiseState(res.state);
      return res;
    } catch (e) {
      handleFranchiseApiError(e);
      return null;
    } finally {
      setAdvancing(false);
    }
  }, [handleFranchiseApiError]);

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
