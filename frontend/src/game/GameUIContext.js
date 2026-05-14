import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  advanceFranchise,
  dismissFranchisePopups,
  getFranchiseState,
  listTeams,
  startFranchise,
  submitDecision,
  submitStorylineChoice,
} from "../services/franchiseService";
import {
  clearFranchiseSession,
  formatFranchiseApiError,
  getFranchiseSessionId,
  setFranchiseSessionId,
} from "../services/api";
import { HUB_MENU, SCREENS, buildDefaultFranchiseTeamList } from "./constants";
import { ShowcasePopupLayer } from "../components/game/ShowcasePopupLayer";

const GameUIContext = createContext(null);

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

  const [teams, setTeams] = useState([]);
  const [teamsLoading, setTeamsLoading] = useState(false);
  const [gmName, setGmName] = useState("Pat Quinn");
  const [franchiseState, setFranchiseState] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [advancing, setAdvancing] = useState(false);

  const [ruleSliders, setRuleSliders] = useState({
    roughing: 50,
    hooking: 50,
    slashing: 50,
    interference: 50,
  });

  const refreshFranchise = useCallback(async () => {
    if (!getFranchiseSessionId()) return;
    setError(null);
    try {
      const s = await getFranchiseState();
      setFranchiseState(s);
    } catch (e) {
      if (e.response?.status === 404 || e.response?.status === 400) {
        clearFranchiseSession();
        setFranchiseState(null);
        setScreen(SCREENS.SETUP);
        return;
      }
      setError(formatFranchiseApiError(e));
    }
  }, []);

  useEffect(() => {
    if (getFranchiseSessionId()) {
      setScreen(SCREENS.HUB);
      refreshFranchise();
    } else {
      setScreen(SCREENS.SETUP);
    }
  }, [refreshFranchise]);

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

  const beginFranchise = useCallback(async () => {
    if (!teams.length) {
      setError("No team selected. Check that the API is running, then try again.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
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
  }, [teams, setupTeamIndex, gmName, setupGamesPerTeam]);

  const onAdvanceFranchise = useCallback(
    async ({ mode = "day", count = 1, auto_resolve: autoResolve } = {}) => {
      if (!franchiseState || franchiseState.phase === "complete") return;
      const m = String(mode || "day").toLowerCase();
      const c = Math.max(1, Number(count) || 1);
      const soloDay = m === "day" && c === 1;
      if (soloDay && !franchiseState?.flags?.can_advance) return;
      const multiDayBlock =
        m === "days" || m === "games" || (m === "day" && c > 1);
      const seasonBlock = m === "season";
      const bulkSim = multiDayBlock || seasonBlock;
      const effectiveAuto = autoResolve === undefined ? Boolean(bulkSim) : Boolean(autoResolve);
      setAdvancing(true);
      setError(null);
      try {
        if (bulkSim) {
          if (!effectiveAuto && franchiseState?.flags && !franchiseState.flags.can_advance) return;
          const targetMode = seasonBlock ? "season" : m === "games" ? "games" : "days";
          const targetCount = seasonBlock ? 1 : c;
          const res = await advanceFranchise({
            mode: targetMode,
            count: targetCount,
            auto_resolve: effectiveAuto,
          });
          setFranchiseState(res.state);
        } else {
          const res = await advanceFranchise({ mode: "day", count: 1, auto_resolve: effectiveAuto });
          setFranchiseState(res.state);
        }
      } catch (e) {
        const d = e.response?.data?.detail;
        const msg = typeof d === "string" ? d : JSON.stringify(d || e.message);
        try {
          if (getFranchiseSessionId()) {
            const s = await getFranchiseState();
            setFranchiseState(s);
          }
        } catch {
          /* keep prior state */
        }
        setError(msg);
      } finally {
        setAdvancing(false);
      }
    },
    [franchiseState]
  );

  const onAdvanceDay = useCallback(async () => {
    await onAdvanceFranchise({ mode: "day", count: 1, auto_resolve: true });
  }, [onAdvanceFranchise]);

  const onResolveDecision = useCallback(async (decisionId, choiceId) => {
    setError(null);
    try {
      const res = await submitDecision(decisionId, choiceId);
      setFranchiseState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      const msg = typeof d === "string" ? d : JSON.stringify(d || e.message);
      try {
        if (getFranchiseSessionId()) {
          const s = await getFranchiseState();
          setFranchiseState(s);
        }
      } catch {
        /* ignore */
      }
      setError(msg);
    }
  }, []);

  const onResolveStorylineChoice = useCallback(async (storylineId, choiceId) => {
    setError(null);
    try {
      const res = await submitStorylineChoice(storylineId, choiceId);
      setFranchiseState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      const msg = typeof d === "string" ? d : JSON.stringify(d || e.message);
      setError(msg);
    }
  }, []);

  const onDismissShowcasePopups = useCallback(async (ids) => {
    if (!getFranchiseSessionId() || !ids || !ids.length) return;
    setError(null);
    try {
      const res = await dismissFranchisePopups(ids);
      setFranchiseState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      setError(typeof d === "string" ? d : JSON.stringify(d || e.message));
    }
  }, []);

  const goNewFranchise = useCallback(() => {
    clearFranchiseSession();
    setFranchiseState(null);
    setScreen(SCREENS.SETUP);
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
      } else if (item.id === "office") {
        setScreen(SCREENS.OFFICE);
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
      franchiseState,
      setFranchiseState,
      error,
      setError,
      loading,
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      beginFranchise,
      onAdvanceDay,
      onAdvanceFranchise,
      onResolveDecision,
      onResolveStorylineChoice,
      onDismissShowcasePopups,
      openHubMenu,
      goNewFranchise,
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
      franchiseState,
      error,
      loading,
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      beginFranchise,
      onAdvanceDay,
      onAdvanceFranchise,
      onResolveDecision,
      onResolveStorylineChoice,
      onDismissShowcasePopups,
      openHubMenu,
      goNewFranchise,
    ]
  );

  return (
    <GameUIContext.Provider value={value}>
      {children}
      <ShowcasePopupLayer />
    </GameUIContext.Provider>
  );
}
