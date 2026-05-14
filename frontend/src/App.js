import React from "react";
import { GameUIProvider, useGameUI } from "./game/GameUIContext";
import { SCREENS } from "./game/constants";
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
import StorylinesScreen from "./screens/StorylinesScreen";

function GameRoot() {
  const { screen } = useGameUI();

  return (
    <GameCanvas>
      {screen === SCREENS.SETUP && <SetupScreen />}
      {screen === SCREENS.HUB && <HubScreen />}
      {screen === SCREENS.ROSTER && <RosterScreen />}
      {screen === SCREENS.CALENDAR && <CalendarScreen />}
      {screen === SCREENS.STORYLINES && <StorylinesScreen />}
      {screen === SCREENS.STATS && <StatsCentralScreen />}
      {screen === SCREENS.TRADE && <TradeHub />}
      {screen === SCREENS.DRAFT_CLASS && <DraftClass />}
      {screen === SCREENS.OFFICE && <OfficeScreen />}
      {screen === SCREENS.SETTINGS && <SettingsScreen />}
    </GameCanvas>
  );
}

export default function App() {
  return (
    <GameUIProvider>
      <GameRoot />
    </GameUIProvider>
  );
}
