import React, { useEffect, useRef } from "react";
import { GameUIProvider, useGameUI } from "./game/GameUIContext";
import { SCREENS } from "./game/constants";
import setupTheme from "./soundtrack/JJ's Energy - Felix Weber (FIFA 2014 World Cup Brazil OST).mp3";
import { GameCanvas } from "./components/game/GameCanvas";
import { SetupScreen } from "./screens/SetupScreen";
import { HubScreen } from "./screens/HubScreen";
import { RosterScreen } from "./screens/RosterScreen";
import { StatsCentralScreen } from "./screens/StatsCentralScreen";
import CalendarScreen from "./screens/CalendarScreen";
import { SettingsScreen } from "./screens/SettingsScreen";
import { OfficeScreen } from "./screens/OfficeScreen";
import TradeHub from "./screens/TradeHub";
import DraftClass from "./screens/DraftClass";
import DraftLottery from "./screens/DraftLottery";
import TeamNeeds from "./screens/TeamNeeds";
import StorylinesScreen from "./screens/StorylinesScreen";
import ChemistryScreen from "./screens/ChemistryScreen";
import EditLines from "./screens/editLines";
import Scouting from "./screens/Scouting";
import CapLedger from "./screens/CapLedger";
import LeagueOperations from "./screens/LeagueOperations";

/** TEMP: remove with frontend/src/dev/EventMenuReplay.js after menu QA */
import EventMenuReplay from "./dev/EventMenuReplay";

const isEventMenuReplay =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("replayEvents") === "1";

const SETUP_MUSIC_VOLUME = 0.28;
const HUB_FADE_MS = 4500;

function useSetupSoundtrack(screen) {
  const audioRef = useRef(null);
  const fadeFrameRef = useRef(null);

  useEffect(() => {
    const audio = new Audio(setupTheme);
    audio.loop = true;
    audio.volume = SETUP_MUSIC_VOLUME;
    audio.preload = "auto";
    audioRef.current = audio;

    return () => {
      if (fadeFrameRef.current != null) {
        cancelAnimationFrame(fadeFrameRef.current);
      }
      audio.pause();
      audio.src = "";
      audioRef.current = null;
    };
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const cancelFade = () => {
      if (fadeFrameRef.current != null) {
        cancelAnimationFrame(fadeFrameRef.current);
        fadeFrameRef.current = null;
      }
    };

    const tryPlay = () => {
      audio.play().catch(() => {
        const resume = () => {
          audio.play().catch(() => {});
          window.removeEventListener("pointerdown", resume);
          window.removeEventListener("keydown", resume);
        };
        window.addEventListener("pointerdown", resume, { once: true });
        window.addEventListener("keydown", resume, { once: true });
      });
    };

    const fadeOutAndStop = () => {
      cancelFade();
      const startVolume = audio.volume;
      const start = performance.now();

      const tick = (now) => {
        const progress = Math.min(1, (now - start) / HUB_FADE_MS);
        audio.volume = Math.max(0, startVolume * (1 - progress));
        if (progress < 1) {
          fadeFrameRef.current = requestAnimationFrame(tick);
        } else {
          audio.pause();
          audio.currentTime = 0;
          audio.volume = SETUP_MUSIC_VOLUME;
          fadeFrameRef.current = null;
        }
      };

      fadeFrameRef.current = requestAnimationFrame(tick);
    };

    if (screen === SCREENS.SETUP) {
      cancelFade();
      audio.volume = SETUP_MUSIC_VOLUME;
      if (audio.paused) {
        tryPlay();
      }
      return cancelFade;
    }

    if (screen === SCREENS.HUB) {
      if (!audio.paused) {
        fadeOutAndStop();
      }
      return cancelFade;
    }

    cancelFade();
    audio.pause();
    audio.currentTime = 0;
    audio.volume = SETUP_MUSIC_VOLUME;
    return cancelFade;
  }, [screen]);
}

function GameRoot() {
  const { screen, hydrateFranchiseHeavyState, franchiseState } = useGameUI();
  useSetupSoundtrack(screen);

  useEffect(() => {
    const needsRosterBrowser = [
      SCREENS.ROSTER,
      SCREENS.TRADE,
      SCREENS.TEAM_NEEDS,
      SCREENS.CAP_LEDGER,
      SCREENS.EDIT_LINES,
      SCREENS.POWER_PLAY,
      SCREENS.PENALTY_KILL,
      SCREENS.SCOUTING,
      SCREENS.DRAFT_CLASS,
      SCREENS.HUB,
    ].includes(screen);
    const needsDraft = [SCREENS.DRAFT_CLASS, SCREENS.SCOUTING].includes(screen);
    if (!needsRosterBrowser && !needsDraft) return;
    hydrateFranchiseHeavyState({
      includeRosterBrowser: needsRosterBrowser,
      includeDraftClassRankings: needsDraft,
      includeDraftClassHud: needsDraft,
    });
  }, [screen, hydrateFranchiseHeavyState, franchiseState?.stats_revision]);

  return (
    <GameCanvas>
      {screen === SCREENS.SETUP && <SetupScreen />}
      {screen === SCREENS.HUB && <HubScreen />}
      {screen === SCREENS.ROSTER && <RosterScreen />}
      {screen === SCREENS.CALENDAR && <CalendarScreen />}
      {screen === SCREENS.STORYLINES && <StorylinesScreen />}
      {screen === SCREENS.CHEMISTRY && <ChemistryScreen />}
      {screen === SCREENS.EDIT_LINES && <EditLines />}
      {screen === SCREENS.POWER_PLAY && <EditLines />}
      {screen === SCREENS.PENALTY_KILL && <EditLines />}
      {screen === SCREENS.STATS && <StatsCentralScreen />}
      {screen === SCREENS.TRADE && <TradeHub />}
      {screen === SCREENS.DRAFT_CLASS && <DraftClass />}
      {screen === SCREENS.DRAFT_LOTTERY && <DraftLottery />}
      {screen === SCREENS.TEAM_NEEDS && <TeamNeeds />}
      {screen === SCREENS.SCOUTING && <Scouting />}
      {screen === SCREENS.OFFICE && <OfficeScreen />}
      {screen === SCREENS.LEAGUE_OPERATIONS && <LeagueOperations />}
      {screen === SCREENS.GM_WORLD && <LeagueOperations />}
      {screen === SCREENS.CAP_LEDGER && <CapLedger />}
      {screen === SCREENS.SETTINGS && <SettingsScreen />}
      {screen === SCREENS.PLACEHOLDER && <CommandPlaceholderScreen />}
    </GameCanvas>
  );
}

function CommandPlaceholderScreen() {
  const { commandPlaceholder, setScreen } = useGameUI();
  const payload = commandPlaceholder || {
    title: "Coming Soon",
    subtitle: "This feature does not have a dedicated screen yet.",
    description: "",
    targetId: "",
  };

  return (
    <div
      className="command-placeholder-screen"
      style={{
        minHeight: "100%",
        display: "grid",
        placeItems: "center",
        padding: 24,
        background:
          "radial-gradient(circle at 50% 0%, rgba(201,168,106,0.12), transparent 40%), linear-gradient(180deg, #10131a, #07080c)",
        color: "#f3ead8",
      }}
    >
      <div
        className="command-placeholder-card"
        style={{
          width: "min(720px, 100%)",
          padding: "28px 26px",
          borderRadius: 14,
          border: "1px solid rgba(201,168,106,0.28)",
          background: "rgba(8,10,14,0.92)",
          boxShadow: "0 24px 64px rgba(0,0,0,0.45)",
        }}
      >
        <span
          style={{
            display: "inline-block",
            marginBottom: 10,
            color: "#c9a86a",
            fontSize: 11,
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            fontWeight: 800,
          }}
        >
          Not Built Yet
        </span>
        <h1 style={{ margin: "0 0 10px", fontSize: "clamp(28px, 4vw, 42px)", lineHeight: 1.05 }}>
          {payload.title}
        </h1>
        <p style={{ margin: "0 0 12px", color: "rgba(243,234,216,0.78)", fontSize: 16 }}>
          {payload.subtitle}
        </p>
        {payload.description ? (
          <p style={{ margin: "0 0 16px", color: "rgba(243,234,216,0.62)", lineHeight: 1.5 }}>
            {payload.description}
          </p>
        ) : null}
        {payload.targetId ? (
          <code
            style={{
              display: "inline-block",
              marginBottom: 18,
              padding: "6px 10px",
              borderRadius: 8,
              background: "rgba(255,255,255,0.04)",
              color: "rgba(201,168,106,0.85)",
              fontSize: 12,
            }}
          >
            Target: {payload.targetId}
          </code>
        ) : null}
        <button
          type="button"
          onClick={() => setScreen(SCREENS.HUB)}
          style={{
            marginTop: 8,
            minHeight: 42,
            padding: "0 16px",
            borderRadius: 999,
            border: "1px solid rgba(201,168,106,0.38)",
            background: "rgba(201,168,106,0.1)",
            color: "#fff7e7",
            fontWeight: 800,
            cursor: "pointer",
          }}
        >
          ← Back to Franchise Office
        </button>
      </div>
    </div>
  );
}

export default function App() {
  if (isEventMenuReplay) {
    return <EventMenuReplay />;
  }

  return (
    <GameUIProvider>
      <GameRoot />
    </GameUIProvider>
  );
}
