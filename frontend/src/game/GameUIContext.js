import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  advanceDay,
  getFranchiseState,
  listTeams,
  startFranchise,
  submitDecision,
} from "../services/franchiseService";
import {
  clearFranchiseSession,
  formatFranchiseApiError,
  getFranchiseSessionId,
  setFranchiseSessionId,
} from "../services/api";
import { HUB_MENU, SCREENS } from "./constants";

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
  const [setupArchetypeIndex, setSetupArchetypeIndex] = useState(0);

  const [teams, setTeams] = useState([]);
  const [coachName, setCoachName] = useState("Pat Quinn");
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

  const archetypes = useMemo(
    () => ["balanced", "development", "defense_first", "aggressive", "players_coach"],
    []
  );

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
    try {
      const t = await listTeams();
      setTeams(t);
      setSetupTeamIndex((i) => Math.min(i, Math.max(0, t.length - 1)));
    } catch (e) {
      setError(formatFranchiseApiError(e));
    }
  }, []);

  useEffect(() => {
    if (screen === SCREENS.SETUP) {
      loadTeams();
    }
  }, [screen, loadTeams]);

  const beginFranchise = useCallback(async () => {
    if (!teams.length) return;
    setError(null);
    setLoading(true);
    try {
      const t = teams[setupTeamIndex];
      const arch = archetypes[setupArchetypeIndex] || "balanced";
      const res = await startFranchise({
        team_query: String(t.team_id),
        head_coach_name: coachName.trim() || "Head Coach",
        coach_archetype: arch,
      });
      setFranchiseSessionId(res.session_id);
      setFranchiseState(res.state);
      setHubMenuIndex(1);
      setScreen(SCREENS.HUB);
    } catch (e) {
      setError(formatFranchiseApiError(e));
    } finally {
      setLoading(false);
    }
  }, [teams, setupTeamIndex, archetypes, setupArchetypeIndex, coachName]);

  const onAdvanceDay = useCallback(async () => {
    if (!franchiseState?.flags?.can_advance) return;
    setAdvancing(true);
    setError(null);
    try {
      const res = await advanceDay();
      setFranchiseState(res.state);
    } catch (e) {
      const d = e.response?.data?.detail;
      setError(typeof d === "string" ? d : JSON.stringify(d || e.message));
    } finally {
      setAdvancing(false);
    }
  }, [franchiseState?.flags?.can_advance]);

  const onResolveDecision = useCallback(async (decisionId, choiceId) => {
    setError(null);
    try {
      const res = await submitDecision(decisionId, choiceId);
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
      } else if (item.id === "ops") {
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
      setupArchetypeIndex,
      setSetupArchetypeIndex,
      teams,
      coachName,
      setCoachName,
      archetypes,
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
      onResolveDecision,
      openHubMenu,
      goNewFranchise,
    }),
    [
      screen,
      hubMenuIndex,
      rosterRowIndex,
      settingsRowIndex,
      setupTeamIndex,
      setupArchetypeIndex,
      teams,
      coachName,
      archetypes,
      franchiseState,
      error,
      loading,
      advancing,
      ruleSliders,
      adjustSlider,
      refreshFranchise,
      beginFranchise,
      onAdvanceDay,
      onResolveDecision,
      openHubMenu,
      goNewFranchise,
    ]
  );

  return <GameUIContext.Provider value={value}>{children}</GameUIContext.Provider>;
}
