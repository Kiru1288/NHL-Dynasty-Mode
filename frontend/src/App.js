import React from "react";
import { GameUIProvider, useGameUI } from "./game/GameUIContext";
import { SCREENS } from "./game/constants";
import { GameCanvas } from "./components/game/GameCanvas";
import { SetupScreen } from "./screens/SetupScreen";
import { HubScreen } from "./screens/HubScreen";
import { RosterScreen } from "./screens/RosterScreen";
import { SettingsScreen } from "./screens/SettingsScreen";

function GameRoot() {
  const { screen } = useGameUI();

  return (
    <GameCanvas>
      {screen === SCREENS.SETUP && <SetupScreen />}
      {screen === SCREENS.HUB && <HubScreen />}
      {screen === SCREENS.ROSTER && <RosterScreen />}
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
