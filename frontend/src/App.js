import React, { Suspense, useEffect, useRef } from "react";
import { GameUIProvider, useGameUI } from "./game/GameUIContext";
import { SCREENS } from "./game/constants";
import setupTheme from "./soundtrack/JJ's Energy - Felix Weber (FIFA 2014 World Cup Brazil OST).mp3";
import { GameCanvas } from "./components/game/GameCanvas";
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
import FreeAgency from "./screens/FreeAgency";
import LeagueOperations from "./screens/LeagueOperations";

/** TEMP: remove with frontend/src/dev/EventMenuReplay.js after menu QA */
import EventMenuReplay from "./dev/EventMenuReplay";

const SetupScreen = React.lazy(() =>
  import("./screens/SetupScreen").then((m) => ({ default: m.SetupScreen }))
);

function SetupScreenFallback() {
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
      Loading franchise setup…
    </div>
  );
}

const isEventMenuReplay =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("replayEvents") === "1";

const SETUP_MUSIC_VOLUME = 0.28;
const HUB_FADE_MS = 4500;

function useSetupSoundtrack(screen) {
  const audioRef = useRef(null);
  const fadeFrameRef = useRef(null);

  useEffect(() => {
    if (screen !== SCREENS.SETUP) return undefined;

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
  }, [screen]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const cancelFade = () => {
      if (fadeFrameRef.current != null) {
        cancelAnimationFrame(fadeFrameRef.current);
        fadeFrameRef.current = null;
      }
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
      const tryPlay = () => {
        audio.play().catch(() => {});
      };
      tryPlay();
      window.addEventListener("pointerdown", tryPlay);
      window.addEventListener("keydown", tryPlay);
      return () => {
        cancelFade();
        window.removeEventListener("pointerdown", tryPlay);
        window.removeEventListener("keydown", tryPlay);
      };
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
      SCREENS.HUB,
    ].includes(screen);
    const needsDraft = [SCREENS.DRAFT_CLASS, SCREENS.SCOUTING].includes(screen);
    if (!needsRosterBrowser && !needsDraft) return;
    hydrateFranchiseHeavyState({
      includeRosterBrowser: needsRosterBrowser,
      includeDraftClassRankings: needsDraft,
      includeDraftClassHud: needsDraft,
    });
  }, [screen, hydrateFranchiseHeavyState, franchiseState?.stats_revision, franchiseState?.prospect_revision]);

  return (
    <GameCanvas>
      {screen === SCREENS.SETUP && (
        <Suspense fallback={<SetupScreenFallback />}>
          <SetupScreen />
        </Suspense>
      )}
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
      {screen === SCREENS.FREE_AGENCY && <FreeAgency />}
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
      className="command-placeholder-screen register-office"
      data-register="office"
      style={{
        minHeight: "100%",
        display: "grid",
        placeItems: "center",
        padding: 24,
        background:
          "radial-gradient(circle at 50% 18%, rgba(201,168,106,0.08), transparent 36%), linear-gradient(180deg, #141820 0%, #0c0e14 100%)",
        color: "var(--office-text, #ece8e0)",
        fontFamily: 'var(--font-office-display, "Archivo Black", sans-serif)',
      }}
    >
      <div
        className="command-placeholder-card"
        style={{
          width: "min(520px, 100%)",
          padding: "20px 22px",
          borderRadius: 6,
          border: "1px solid rgba(255,255,255,0.08)",
          background: "rgba(8,10,14,0.94)",
          boxShadow: "0 18px 42px rgba(0,0,0,0.48)",
          position: "relative",
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 2,
            background: "linear-gradient(90deg, transparent, #c9a86a, transparent)",
            opacity: 0.55,
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            marginBottom: 10,
          }}
        >
          <span
            style={{
              color: "#c9a86a",
              fontSize: 11,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              fontWeight: 800,
            }}
          >
            Office · Filing
          </span>
          <span className="fcn-stamp fcn-stamp--office">Not in service</span>
        </div>
        <h1
          style={{
            margin: "0 0 8px",
            fontSize: "clamp(22px, 3vw, 32px)",
            lineHeight: 1.05,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          {payload.title}
        </h1>
        <p
          style={{
            margin: "0 0 10px",
            color: "rgba(220,216,208,0.72)",
            fontSize: 14,
            fontFamily: "Inter, system-ui, sans-serif",
            fontWeight: 600,
          }}
        >
          {payload.subtitle}
        </p>
        {payload.description ? (
          <p
            style={{
              margin: "0 0 14px",
              color: "rgba(220,216,208,0.55)",
              lineHeight: 1.45,
              fontSize: 13,
              fontFamily: "Inter, system-ui, sans-serif",
            }}
          >
            {payload.description}
          </p>
        ) : null}
        {payload.targetId ? (
          <code
            style={{
              display: "inline-block",
              marginBottom: 16,
              padding: "4px 8px",
              borderRadius: 2,
              background: "rgba(255,255,255,0.04)",
              color: "rgba(201,168,106,0.85)",
              fontSize: 11,
              letterSpacing: "0.06em",
            }}
          >
            Target: {payload.targetId}
          </code>
        ) : null}
        <button
          type="button"
          className="ui-btn ui-btn--executive"
          onClick={() => setScreen(SCREENS.HUB)}
          style={{
            marginTop: 4,
            minHeight: 40,
            padding: "0 16px",
            cursor: "pointer",
          }}
        >
          Return to Office
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
